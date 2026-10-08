"""Contiguous video padding and RGB conversion for CUDA inference."""

from __future__ import annotations

import torch

from orbitquant.kernels.triton_cuda import _load_triton

triton, tl = _load_triton()
from triton.language.extra import libdevice  # noqa: E402


@triton.jit
def _pad_kernel(
    X,
    Y,
    N,
    T: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    OT: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    LEFT: tl.constexpr,
    TOP: tl.constexpr,
    FRONT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ox, oy, ot = i % OW, i // OW % OH, i // (OW * OH) % OT
    bc = i // (OW * OH * OT)
    ix = tl.minimum(tl.maximum(ox - LEFT, 0), W - 1)
    iy = tl.minimum(tl.maximum(oy - TOP, 0), H - 1)
    it = tl.minimum(tl.maximum(ot - FRONT, 0), T - 1)
    value = tl.load(X + ((bc * T + it) * H + iy) * W + ix, i < N, other=0)
    tl.store(Y + i, value, i < N)


def replicate_video_padding(x, padding):
    """Pad contiguous NCTHW input; unsupported inputs use PyTorch."""
    if (
        torch.is_grad_enabled()
        or x.device.type != "cuda"
        or x.ndim != 5
        or not x.is_contiguous()
        or x.numel() == 0
        or len(padding) != 6
        or any(not isinstance(p, int) or p < 0 for p in padding)
        or x.dtype not in (torch.float16, torch.bfloat16, torch.float32)
    ):
        return torch.nn.functional.pad(x, padding, mode="replicate")
    b, c, t, h, w = x.shape
    left, right, top, bottom, front, back = padding
    shape = (b, c, t + front + back, h + top + bottom, w + left + right)
    count = b * c * shape[2] * shape[3] * shape[4]
    if count >= 2**31:
        return torch.nn.functional.pad(x, padding, mode="replicate")
    y = torch.empty(shape, device=x.device, dtype=x.dtype)
    _pad_kernel[(triton.cdiv(count, 2048),)](
        x,
        y,
        count,
        T=t,
        H=h,
        W=w,
        OT=shape[2],
        OH=shape[3],
        OW=shape[4],
        LEFT=left,
        TOP=top,
        FRONT=front,
        BLOCK=2048,
        num_warps=4,
    )
    return y


@triton.jit
def _rgb_kernel(X, Y, N, T: tl.constexpr, H: tl.constexpr, W: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    pixel, channel = i // 3, i % 3
    frame, batch = pixel // (H * W) % T, pixel // (H * W * T)
    offset = ((batch * 3 + channel) * T + frame) * H * W + pixel % (H * W)
    x = tl.load(X + offset, i < N, other=0)
    # Match both eager dtype round trips before FP32 NumPy multiplication.
    half = (x.to(tl.float32) * 0.5).to(x.dtype).to(tl.float32)
    normalized = (half + 0.5).to(x.dtype).to(tl.float32)
    value = libdevice.rint(tl.minimum(tl.maximum(normalized, 0.0), 1.0) * 255.0)
    tl.store(Y + i, value.to(tl.uint8), i < N)


def video_to_rgb8(video):
    """Return contiguous NTHWC RGB bytes on the input CUDA device.

    The runtime caller validates layout, channels, dtype and inference mode.
    """
    b, _, t, h, w = video.shape
    out = torch.empty((b, t, h, w, 3), device=video.device, dtype=torch.uint8)
    _rgb_kernel[(triton.cdiv(out.numel(), 1024),)](
        video,
        out,
        out.numel(),
        T=t,
        H=h,
        W=w,
        BLOCK=1024,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out


@triton.jit
def _group_partials(X, P, SIZE: tl.constexpr, PARTS: tl.constexpr, BLOCK: tl.constexpr):
    group, part = tl.program_id(0), tl.program_id(1)
    i = part * BLOCK + tl.arange(0, BLOCK)
    count = tl.minimum(SIZE - part * BLOCK, BLOCK)
    x = tl.load(X + group * SIZE + i, i < SIZE, other=0).to(tl.float32)
    mean = tl.sum(x, 0) / count
    delta = tl.where(i < SIZE, x - mean, 0.0)
    m2 = tl.sum(delta * delta, 0)
    tl.store(P + group * PARTS * 2 + part * 2, mean)
    tl.store(P + group * PARTS * 2 + part * 2 + 1, m2)


@triton.jit
def _group_statistics(
    P,
    S,
    SIZE: tl.constexpr,
    PARTS: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK: tl.constexpr,
    EPS: tl.constexpr,
):
    group = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    n = tl.minimum(tl.maximum(SIZE - i * CHUNK, 0), CHUNK).to(tl.float32)
    means = tl.load(P + group * PARTS * 2 + i * 2, i < PARTS, other=0)
    m2 = tl.load(P + group * PARTS * 2 + i * 2 + 1, i < PARTS, other=0)
    mean = tl.sum(means * n, 0) / SIZE
    variance = tl.sum(m2 + n * (means - mean) * (means - mean), 0) / SIZE
    tl.store(S + group * 2, mean)
    tl.store(S + group * 2 + 1, tl.rsqrt(variance + EPS))


@triton.jit
def _group_output(
    X,
    W,
    B,
    S,
    Y,
    N,
    C: tl.constexpr,
    SPATIAL: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    group = i // GROUP_SIZE
    channel = i // SPATIAL % C
    x = tl.load(X + i, i < N, other=0).to(tl.float32)
    weight = tl.load(W + channel, i < N, other=0).to(tl.float32)
    bias = tl.load(B + channel, i < N, other=0).to(tl.float32)
    mean = tl.load(S + group * 2, i < N, other=0)
    inv = tl.load(S + group * 2 + 1, i < N, other=0)
    scale = inv * weight
    normalized = (x * scale + (bias - mean * scale)).to(Y.dtype.element_ty).to(tl.float32)
    activated = normalized / (1.0 + tl.exp(-normalized))
    tl.store(Y + i, activated, i < N)


def group_norm_silu(x, norm):
    """Fuse large BF16 GroupNorm and SiLU; preserve eager fallback contracts."""
    if (
        x.device.type != "cuda"
        or not x.is_contiguous()
        or torch.is_grad_enabled()
        or x.dtype != torch.bfloat16
        or x.ndim < 3
        or not 131072 <= x.numel() < 2**31
        or type(norm) is not torch.nn.GroupNorm
        or not norm.affine
        or norm.weight.dtype != x.dtype
        or norm.bias.dtype != x.dtype
        or norm.weight.device != x.device
        or norm.bias.device != x.device
        or norm._forward_hooks
        or norm._forward_pre_hooks
        or "forward" in norm.__dict__
    ):
        return torch.nn.functional.silu(norm(x))
    b, c = x.shape[:2]
    spatial = x.numel() // (b * c)
    group_size = c // norm.num_groups * spatial
    groups = b * norm.num_groups
    if triton.cdiv(group_size, 4096) > 32768:
        return torch.nn.functional.silu(norm(x))
    parts = triton.cdiv(group_size, 4096)
    partial = torch.empty((groups, parts, 2), device=x.device, dtype=torch.float32)
    stats = torch.empty((groups, 2), device=x.device, dtype=x.dtype)
    y = torch.empty_like(x)
    _group_partials[(groups, parts)](
        x, partial, SIZE=group_size, PARTS=parts, BLOCK=4096, num_warps=4
    )
    _group_statistics[(groups,)](
        partial,
        stats,
        SIZE=group_size,
        PARTS=parts,
        CHUNK=4096,
        BLOCK=triton.next_power_of_2(parts),
        EPS=norm.eps,
        num_warps=4,
    )
    _group_output[(triton.cdiv(x.numel(), 1024),)](
        x,
        norm.weight,
        norm.bias,
        stats,
        y,
        x.numel(),
        C=c,
        SPATIAL=spatial,
        GROUP_SIZE=group_size,
        BLOCK=1024,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return y
