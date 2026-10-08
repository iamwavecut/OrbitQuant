"""FP32 pointwise arithmetic for Kandinsky 6, fused before the final cast."""

from __future__ import annotations

import torch

from orbitquant.kernels.triton_cuda import _load_triton

triton, tl = _load_triton()


@triton.jit
def _qk_rms_kernel(X, Weight, Y, R, EPS: tl.constexpr):
    rows = tl.program_id(0) * 8 + tl.arange(0, 8)
    cols = tl.arange(0, 64)
    x = tl.load(X + rows[:, None] * 64 + cols[None, :], rows[:, None] < R, other=0).to(tl.float32)
    weight = tl.load(Weight + cols).to(tl.float32)
    inverse = tl.rsqrt(tl.sum(x * x, 1) / 64 + EPS)
    out = x * inverse[:, None] * weight[None, :]
    tl.store(Y + rows[:, None] * 64 + cols[None, :], out, rows[:, None] < R)


def qk_rms_norm(x, weight, eps):
    """Normalize contiguous BF16 attention heads of width 64, eight rows per block."""
    out = torch.empty_like(x)
    rows = x.numel() // 64
    _qk_rms_kernel[(triton.cdiv(rows, 8),)](
        x, weight, out, rows, EPS=eps, num_warps=4, enable_fp_fusion=False
    )
    return out


@triton.jit
def _norm_modulate_kernel(
    X, Scale, Shift, Y, D: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * D + col, col < D, other=0).to(tl.float32)
    mean = tl.sum(x, 0) / D
    centered = tl.where(col < D, x - mean, 0.0)
    variance = tl.sum(centered * centered, 0) / D
    normalized = (x - mean) * tl.rsqrt(variance + EPS)
    scale = tl.load(Scale + col, col < D, other=0).to(tl.float32)
    shift = tl.load(Shift + col, col < D, other=0).to(tl.float32)
    tl.store(Y + row * D + col, normalized * (1.0 + scale) + shift, col < D)


def norm_modulate(x, norm, scale, shift):
    """Fuse non-affine LayerNorm and modulation, with FP32 reduction and final cast."""
    width = x.shape[-1]
    if (
        _affine_ready(x, x, scale, shift)
        and type(norm) is torch.nn.LayerNorm
        and "forward" not in norm.__dict__
        and not norm._forward_hooks
        and not norm._forward_pre_hooks
        and not norm.elementwise_affine
        and tuple(norm.normalized_shape) == (width,)
        and width <= 8192
    ):
        out = torch.empty_like(x)
        _norm_modulate_kernel[(x.numel() // width,)](
            x,
            scale,
            shift,
            out,
            D=width,
            EPS=norm.eps,
            BLOCK=triton.next_power_of_2(width),
            num_warps=4,
            enable_fp_fusion=False,
        )
        return out
    return scale_shift(norm(x.float()), x, scale, shift)


@triton.jit
def _affine_kernel(
    X, Y, Scale, Shift, Out, count, D: tl.constexpr, GATE: tl.constexpr, BLOCK: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < count
    cols = offsets % D
    y = tl.load(Y + offsets, mask, other=0).to(tl.float32)
    scale = tl.load(Scale + cols).to(tl.float32)
    if GATE:
        x = tl.load(X + offsets, mask, other=0).to(tl.float32)
        result = x + scale * y
    else:
        shift = tl.load(Shift + cols).to(tl.float32)
        result = y * (scale + 1.0) + shift
    tl.store(Out + offsets, result, mask)


def _affine_ready(x, other, *vectors):
    return (
        not torch.is_grad_enabled()
        and x.device.type == "cuda"
        and x.ndim >= 2
        and x.numel() > 0
        and x.shape[0] == 1
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and x.is_contiguous()
        and other.device == x.device
        and other.shape == x.shape
        and other.is_contiguous()
        and all(
            v.device == x.device and v.numel() == x.shape[-1] and v.is_contiguous() for v in vectors
        )
    )


def gate_sum(x, out, gate):
    if _affine_ready(x, out, gate):
        result = torch.empty_like(x)
        _affine_kernel[(triton.cdiv(x.numel(), 2048),)](
            x,
            out,
            gate,
            gate,
            result,
            x.numel(),
            D=x.shape[-1],
            GATE=True,
            BLOCK=2048,
            num_warps=4,
            enable_fp_fusion=False,
        )
        return result
    if x.ndim > 2 and gate.ndim == 2:
        gate = gate.reshape((gate.shape[0],) + (1,) * (x.ndim - 2) + (gate.shape[-1],))
    return (x.float() + gate.float() * out.float()).to(x.dtype)


def scale_shift(normed, x, scale, shift):
    if _affine_ready(x, normed, scale, shift):
        result = torch.empty_like(x)
        _affine_kernel[(triton.cdiv(x.numel(), 2048),)](
            x,
            normed,
            scale,
            shift,
            result,
            x.numel(),
            D=x.shape[-1],
            GATE=False,
            BLOCK=2048,
            num_warps=4,
            enable_fp_fusion=False,
        )
        return result
    if x.ndim > 2 and scale.ndim == 2:
        shape = (scale.shape[0],) + (1,) * (x.ndim - 2) + (scale.shape[-1],)
        scale, shift = scale.reshape(shape), shift.reshape(shape)
    return (normed * (scale.float() + 1.0) + shift.float()).to(x.dtype)


@triton.jit
def _rotary_kernel(X, Rope, Out, pairs, HEADS: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < pairs
    seq = offsets // (HEADS * (D // 2))
    channel = offsets % (D // 2)
    pos = seq * (2 * D) + channel * 4
    x0 = tl.load(X + offsets * 2, mask, other=0).to(tl.float32)
    x1 = tl.load(X + offsets * 2 + 1, mask, other=0).to(tl.float32)
    r00 = tl.load(Rope + pos, mask, other=0).to(tl.float32)
    r01 = tl.load(Rope + pos + 1, mask, other=0).to(tl.float32)
    r10 = tl.load(Rope + pos + 2, mask, other=0).to(tl.float32)
    r11 = tl.load(Rope + pos + 3, mask, other=0).to(tl.float32)
    tl.store(Out + offsets * 2, r00 * x0 + r01 * x1, mask)
    tl.store(Out + offsets * 2 + 1, r10 * x0 + r11 * x1, mask)


def rotary(x, rope):
    if (
        not torch.is_grad_enabled()
        and x.device.type == "cuda"
        and x.ndim == 4
        and x.numel() > 0
        and x.shape[0] == 1
        and x.shape[-1] % 2 == 0
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and x.is_contiguous()
        and rope.device == x.device
        and rope.shape == (x.shape[1], 1, x.shape[-1] // 2, 2, 2)
        and rope.is_contiguous()
    ):
        result = torch.empty_like(x)
        _rotary_kernel[(triton.cdiv(x.numel() // 2, 1024),)](
            x,
            rope,
            result,
            x.numel() // 2,
            HEADS=x.shape[2],
            D=x.shape[-1],
            BLOCK=1024,
            num_warps=4,
            enable_fp_fusion=False,
        )
        return result
    pairs = x.reshape(*x.shape[:-1], -1, 1, 2).float()
    return (rope.float() * pairs).sum(-1).reshape(x.shape).to(x.dtype)
