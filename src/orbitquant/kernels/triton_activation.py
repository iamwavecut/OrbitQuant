"""Activation-side Triton kernels for fused DiT blocks.

* RPBH + Lloyd-Max activation quantization straight to INT8 surrogate codes, optionally fed by
  a prologue that replaces the elementwise ops in front of a projection (RMSNorm with a
  ``1 + weight`` scale followed by AdaLN modulation, or an elementwise product such as an
  attention output gate). Every value eager code rounds to bf16 is rounded at the same point.
* per-token absmax INT8 for W8A8 projections (``Int8RowLinear`` numerics),
* per-head RMSNorm + interleaved rotary embedding for Q/K,
* packed W4 -> INT8 surrogate weight decode.
"""

from __future__ import annotations

import torch

from orbitquant.kernels.triton_cuda import _load_triton, fit_int8_centroid_surrogate

triton, tl = _load_triton()

from triton.language.extra import libdevice  # noqa: E402

PROLOGUE_NONE = 0
PROLOGUE_NORM_MODULATE = 1
PROLOGUE_MULTIPLY = 2


@triton.jit
def _bf16(x):
    return x.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _prologue_kernel(
    x_ptr,
    x2_ptr,
    w_ptr,
    scale_ptr,
    shift_ptr,
    y_ptr,
    norms_ptr,
    x_stride,
    x2_stride,
    D: tl.constexpr,
    DP2: tl.constexpr,
    rms_eps,
    PROLOGUE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, DP2)
    mask = offs < D
    x = tl.load(x_ptr + row * x_stride + offs, mask=mask, other=0.0).to(tl.float32)
    if PROLOGUE == 1:
        mean_sq = libdevice.div_rn(tl.sum(x * x, axis=0), D * 1.0)
        inv = libdevice.rsqrt(mean_sq + rms_eps)
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32) + 1.0
        normed = _bf16(x * inv * w)
        scale = tl.load(scale_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        shift = tl.load(shift_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = _bf16(_bf16(_bf16(1.0 + scale) * normed) + shift)
    elif PROLOGUE == 2:
        x2 = tl.load(x2_ptr + row * x2_stride + offs, mask=mask, other=0.0).to(tl.float32)
        y = _bf16(x * x2)
    else:
        y = x
    tl.store(y_ptr + row * D + offs, y.to(tl.bfloat16), mask=mask)
    tl.store(norms_ptr + row, tl.sqrt(tl.sum(y * y, axis=0)))


@triton.jit
def _fwht_local_stage(values, block_size: tl.constexpr, stage_width: tl.constexpr):
    butterfly = tl.reshape(values, (block_size // (stage_width * 2), 2, stage_width))
    butterfly = tl.permute(butterfly, (0, 2, 1))
    left, right = tl.split(butterfly)
    butterfly = tl.join(left + right, left - right)
    butterfly = tl.permute(butterfly, (0, 2, 1))
    return tl.reshape(butterfly, (block_size,))


@triton.jit
def _rpbh_int8_kernel(
    y_ptr,
    norms_ptr,
    perm_ptr,
    signs_ptr,
    bounds_ptr,
    codes_ptr,
    q_ptr,
    D: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    OB: tl.constexpr,
    STAGES: tl.constexpr,
    LEVELS: tl.constexpr,
    eps,
    inv_sqrt_block,
):
    row_block = tl.program_id(0)
    row = (row_block // NUM_BLOCKS).to(tl.int64)
    block = row_block % NUM_BLOCKS
    cols = block * OB + tl.arange(0, OB)
    src = tl.load(perm_ptr + cols).to(tl.int32)
    signs = tl.load(signs_ptr + cols).to(tl.float32)
    norm = tl.load(norms_ptr + row).to(tl.float32)
    values = tl.load(y_ptr + row * D + src).to(tl.float32)
    values = values * signs / (norm + eps)
    if STAGES > 0:
        values = _fwht_local_stage(values, OB, 1)
    if STAGES > 1:
        values = _fwht_local_stage(values, OB, 2)
    if STAGES > 2:
        values = _fwht_local_stage(values, OB, 4)
    if STAGES > 3:
        values = _fwht_local_stage(values, OB, 8)
    if STAGES > 4:
        values = _fwht_local_stage(values, OB, 16)
    if STAGES > 5:
        values = _fwht_local_stage(values, OB, 32)
    if STAGES > 6:
        values = _fwht_local_stage(values, OB, 64)
    if STAGES > 7:
        values = _fwht_local_stage(values, OB, 128)
    if STAGES > 8:
        values = _fwht_local_stage(values, OB, 256)
    if STAGES > 9:
        values = _fwht_local_stage(values, OB, 512)
    if STAGES > 10:
        values = _fwht_local_stage(values, OB, 1024)
    if STAGES > 11:
        values = _fwht_local_stage(values, OB, 2048)
    values *= inv_sqrt_block
    indices = tl.zeros((OB,), dtype=tl.int32)
    for level in tl.static_range(0, LEVELS - 1):
        indices += (values > tl.load(bounds_ptr + level)).to(tl.int32)
    tl.store(q_ptr + row * D + cols, tl.load(codes_ptr + indices).to(tl.int8))


class ActivationQuantizer:
    """RPBH/Lloyd-Max INT8 surrogate activation quantizer of one OrbitQuantLinear input.

    Projections that share an input (Q/K/V/gate, gate/up) share one quantizer: the codes and
    token norms are identical, so they are produced once.
    """

    def __init__(self, rotation, codebook, eps: float, device: torch.device):
        self.dim = int(rotation.dim)
        self.block = int(rotation.block_size)
        self.num_blocks = int(rotation.num_blocks)
        if self.block > 4096:
            raise ValueError("fused activation quantization supports RPBH blocks up to 4096")
        self.inv_sqrt_block = float(rotation.normalization)
        self.eps = float(eps)
        self.perm = rotation.permutation.to(device=device, dtype=torch.int32).contiguous()
        self.signs = rotation.signs.to(device=device, dtype=torch.int8).contiguous()
        self.bounds = codebook.boundaries.to(device=device, dtype=torch.float32).contiguous()
        self.levels = int(codebook.centroids.numel())
        codes, self.scale = fit_int8_centroid_surrogate(codebook.centroids)
        self.codes = codes.to(device=device, dtype=torch.int8).contiguous()

    def __call__(
        self,
        x,
        *,
        prologue=PROLOGUE_NONE,
        x2=None,
        norm_weight=None,
        mod_scale=None,
        mod_shift=None,
        rms_eps=1e-5,
    ):
        """Return ``(codes int8 [rows, dim], token_norms fp32 [rows])``."""
        if x.dim() != 2 or x.stride(1) != 1:
            x = x.reshape(-1, self.dim).contiguous()
        rows = x.shape[0]
        if x2 is not None and (x2.dim() != 2 or x2.stride(1) != 1):
            x2 = x2.reshape(rows, self.dim).contiguous()
        y = torch.empty((rows, self.dim), device=x.device, dtype=torch.bfloat16)
        norms = torch.empty(rows, device=x.device, dtype=torch.float32)
        q = torch.empty((rows, self.dim), device=x.device, dtype=torch.int8)
        if rows == 0:
            return q, norms
        _prologue_kernel[(rows,)](
            x,
            x2 if x2 is not None else x,
            norm_weight if norm_weight is not None else norms,
            mod_scale if mod_scale is not None else x,
            mod_shift if mod_shift is not None else x,
            y,
            norms,
            x.stride(0),
            x2.stride(0) if x2 is not None else 0,
            D=self.dim,
            DP2=triton.next_power_of_2(self.dim),
            rms_eps=float(rms_eps),
            PROLOGUE=int(prologue),
            num_warps=8,
        )
        _rpbh_int8_kernel[(rows * self.num_blocks,)](
            y,
            norms,
            self.perm,
            self.signs,
            self.bounds,
            self.codes,
            q,
            D=self.dim,
            NUM_BLOCKS=self.num_blocks,
            OB=self.block,
            STAGES=self.block.bit_length() - 1,
            LEVELS=self.levels,
            eps=self.eps,
            inv_sqrt_block=self.inv_sqrt_block,
            num_warps=8 if self.block >= 512 else 4,
        )
        return q, norms


@triton.jit
def _rows_int8_kernel(x_ptr, q_ptr, scale_ptr, D: tl.constexpr, DP2: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, DP2)
    mask = offs < D
    x = tl.load(x_ptr + row * D + offs, mask=mask, other=0.0).to(tl.float32)
    scale = libdevice.div_rn(tl.maximum(tl.max(tl.abs(x), axis=0), 1e-12), 127.0)
    q = libdevice.rint(libdevice.div_rn(x, scale))
    q = tl.minimum(tl.maximum(q, -127.0), 127.0)
    tl.store(q_ptr + row * D + offs, q.to(tl.int8), mask=mask)
    tl.store(scale_ptr + row, scale)


def quantize_rows_int8_with_triton(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token absmax INT8 with ``Int8RowLinear`` numerics (IEEE divisions, round-half-even)."""
    dim = x.shape[-1]
    x = x.reshape(-1, dim).contiguous()
    rows = x.shape[0]
    q = torch.empty((rows, dim), device=x.device, dtype=torch.int8)
    scales = torch.empty(rows, device=x.device, dtype=torch.float32)
    if rows:
        _rows_int8_kernel[(rows,)](
            x, q, scales, D=dim, DP2=triton.next_power_of_2(dim), num_warps=8
        )
    return q, scales


@triton.jit
def _qk_norm_rope_kernel(
    src_ptr,
    src_stride,
    out_ptr,
    w_ptr,
    cos_ptr,
    sin_ptr,
    rms_eps,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    half = tl.arange(0, HEAD_DIM // 2)
    base = src_ptr + row * src_stride + head * HEAD_DIM
    even = tl.load(base + 2 * half).to(tl.float32)
    odd = tl.load(base + 2 * half + 1).to(tl.float32)
    mean_sq = libdevice.div_rn(
        tl.sum(even * even, axis=0) + tl.sum(odd * odd, axis=0), HEAD_DIM * 1.0
    )
    inv = libdevice.rsqrt(mean_sq + rms_eps)
    even = _bf16(even * inv * (tl.load(w_ptr + 2 * half).to(tl.float32) + 1.0))
    odd = _bf16(odd * inv * (tl.load(w_ptr + 2 * half + 1).to(tl.float32) + 1.0))
    cos_even = tl.load(cos_ptr + row * HEAD_DIM + 2 * half)
    cos_odd = tl.load(cos_ptr + row * HEAD_DIM + 2 * half + 1)
    sin_even = tl.load(sin_ptr + row * HEAD_DIM + 2 * half)
    sin_odd = tl.load(sin_ptr + row * HEAD_DIM + 2 * half + 1)
    out_even = even * cos_even + (-odd) * sin_even
    out_odd = odd * cos_odd + even * sin_odd
    dst = out_ptr + (row * HEADS + head) * HEAD_DIM
    tl.store(dst + 2 * half, out_even.to(tl.bfloat16))
    tl.store(dst + 2 * half + 1, out_odd.to(tl.bfloat16))


def qk_norm_rope_with_triton(src, heads, head_dim, weight, cos, sin, rms_eps):
    """Per-head RMSNorm (``1 + weight`` scale) then interleaved rotary embedding.

    ``src`` is a ``[rows, >= heads * head_dim]`` strided view; returns
    ``[1, rows, heads, head_dim]`` bf16. Multiplies and adds are not contracted into FMAs, as
    in the eager elementwise ops.
    """
    rows = src.shape[0]
    out = torch.empty((1, rows, heads, head_dim), device=src.device, dtype=torch.bfloat16)
    if rows:
        _qk_norm_rope_kernel[(rows, heads)](
            src,
            src.stride(0),
            out,
            weight,
            cos,
            sin,
            float(rms_eps),
            HEADS=heads,
            HEAD_DIM=head_dim,
            num_warps=1,
            enable_fp_fusion=False,
        )
    return out


@triton.jit
def _decode_w4_kernel(packed_ptr, codes_ptr, out_ptr, total_bytes, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total_bytes
    byte = tl.load(packed_ptr + offs, mask=mask, other=0).to(tl.int32)
    low = tl.load(codes_ptr + (byte & 15))
    high = tl.load(codes_ptr + ((byte >> 4) & 15))
    pair = tl.reshape(tl.join(low, high), (2 * BLOCK,))
    out_offs = tl.program_id(0).to(tl.int64) * (2 * BLOCK) + tl.arange(0, 2 * BLOCK)
    tl.store(out_ptr + out_offs, pair.to(tl.int8), mask=out_offs < 2 * total_bytes)


def decode_w4_to_int8_with_triton(packed, codes, out_features, in_features, out=None):
    """Packed W4 indices (low nibble first) -> INT8 surrogate weight ``[out, in]``."""
    if out is None:
        out = torch.empty((out_features, in_features), device=packed.device, dtype=torch.int8)
    total = out_features * in_features // 2
    block = 1024
    _decode_w4_kernel[(triton.cdiv(total, block),)](
        packed.reshape(-1), codes, out, total, BLOCK=block
    )
    return out
