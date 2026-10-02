"""Activation-side Triton kernels for fused DiT blocks.

* RPBH + Lloyd-Max activation quantization straight to INT8 surrogate codes, optionally fed by
  a prologue that replaces the elementwise ops in front of a projection: an RMSNorm or
  LayerNorm followed by AdaLN modulation (per tensor or selected per row from a table), or an
  elementwise product such as an attention output gate. Every value eager code rounds to bf16
  is rounded at the same point.
* per-token absmax INT8 for W8A8 projections (``Int8RowLinear`` numerics),
* per-head RMSNorm + rotary embedding for Q/K (interleaved pairs or rotate-half, optionally on
  a leading slice of the head, optionally zero-padded to a wider head),
* post-norm gated residual updates (``res + gate * norm(x)``),
* packed W2/W3/W4 -> INT8 surrogate weight decode.
"""

from __future__ import annotations

import torch

from orbitquant.kernels.triton_cuda import _load_triton, fit_int8_centroid_surrogate

triton, tl = _load_triton()

from triton.language.extra import libdevice  # noqa: E402

PROLOGUE_NONE = 0
PROLOGUE_NORM_MODULATE = 1  # NORM_RMS_ONE_PLUS + MOD_SHIFT_SCALE (Krea 2)
PROLOGUE_MULTIPLY = 2

# Norm in front of a projection. The RMSNorm variants differ in where eager code rounds:
# torch.nn.RMSNorm rounds once (``x * inv * w``), diffusers' RMSNorm rounds ``x * inv`` before
# the weight multiply.
NORM_NONE = 0
NORM_RMS_ONE_PLUS = 1  # x * inv * (1 + w), one rounding
NORM_RMS = 2  # bf16(x * inv) * w (diffusers RMSNorm)
NORM_RMS_PLAIN = 3  # x * inv, no weight
NORM_LAYER = 4  # (x - mean) * rsqrt(var + eps), no affine
NORM_RMS_TORCH = 5  # x * inv * w, one rounding (torch.nn.RMSNorm)

# AdaLN modulation after the norm.
MOD_NONE = 0
MOD_SHIFT_SCALE = 1  # (1 + scale) * n + shift
MOD_SCALE = 2  # n * scale (the caller already folded the 1 +)
MOD_ONE_PLUS_SCALE = 3  # (1 + scale) * n

_PROLOGUES = {
    PROLOGUE_NONE: (NORM_NONE, MOD_NONE),
    PROLOGUE_NORM_MODULATE: (NORM_RMS_ONE_PLUS, MOD_SHIFT_SCALE),
}


@triton.jit
def _bf16(x):
    return x.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _normalize(x, w_ptr, offs, mask, D: tl.constexpr, rms_eps, NORM: tl.constexpr):
    if NORM == 4:
        mean = libdevice.div_rn(tl.sum(x, axis=0), D * 1.0)
        centered = tl.where(mask, x - mean, 0.0)
        var = libdevice.div_rn(tl.sum(centered * centered, axis=0), D * 1.0)
        return _bf16(centered * libdevice.rsqrt(var + rms_eps))
    if NORM == 0:
        return x
    mean_sq = libdevice.div_rn(tl.sum(x * x, axis=0), D * 1.0)
    inv = libdevice.rsqrt(mean_sq + rms_eps)
    if NORM == 1:
        return _bf16(x * inv * (tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32) + 1.0))
    if NORM == 2:
        return _bf16(_bf16(x * inv) * tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32))
    if NORM == 5:
        return _bf16(x * inv * tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32))
    return _bf16(x * inv)


@triton.jit
def _prologue_kernel(
    x_ptr,
    x2_ptr,
    w_ptr,
    scale_ptr,
    shift_ptr,
    index_ptr,
    y_ptr,
    norms_ptr,
    x_stride,
    x2_stride,
    mod_stride,
    D: tl.constexpr,
    DP2: tl.constexpr,
    rms_eps,
    MULTIPLY: tl.constexpr,
    NORM: tl.constexpr,
    MOD: tl.constexpr,
    ROW_INDEXED: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, DP2)
    mask = offs < D
    x = tl.load(x_ptr + row * x_stride + offs, mask=mask, other=0.0).to(tl.float32)
    if MULTIPLY:
        x2 = tl.load(x2_ptr + row * x2_stride + offs, mask=mask, other=0.0).to(tl.float32)
        y = _bf16(x * x2)
    else:
        y = _normalize(x, w_ptr, offs, mask, D, rms_eps, NORM)
        if MOD != 0:
            mod_row = tl.load(index_ptr + row).to(tl.int64) * mod_stride if ROW_INDEXED else 0
            scale = tl.load(scale_ptr + mod_row + offs, mask=mask, other=0.0).to(tl.float32)
            if MOD == 1:
                shift = tl.load(shift_ptr + mod_row + offs, mask=mask, other=0.0).to(tl.float32)
                y = _bf16(_bf16(_bf16(1.0 + scale) * y) + shift)
            elif MOD == 2:
                y = _bf16(y * scale)
            else:
                y = _bf16(_bf16(1.0 + scale) * y)
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
def _fwht(values, OB: tl.constexpr, STAGES: tl.constexpr):
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
    if STAGES > 12:
        values = _fwht_local_stage(values, OB, 4096)
    if STAGES > 13:
        values = _fwht_local_stage(values, OB, 8192)
    return values


# Widest transform held in one tensor. Wider RPBH blocks (8192, 16384) are rotated as 4096-wide
# chunks whose last one or two butterfly stages pair whole chunks, which are plain elementwise
# ops in registers; one 16384-wide tensor needs more shared memory than consumer GPUs have.
FWHT_CHUNK = 4096
_CHUNK = tl.constexpr(FWHT_CHUNK)
_CHUNK_STAGES = tl.constexpr(FWHT_CHUNK.bit_length() - 1)


@triton.jit
def _rotated_chunk(
    y_ptr, perm_ptr, signs_ptr, row, start, denom, D: tl.constexpr, NORMALIZE: tl.constexpr
):
    cols = start + tl.arange(0, _CHUNK)
    src = tl.load(perm_ptr + cols).to(tl.int32)
    signs = tl.load(signs_ptr + cols).to(tl.float32)
    values = tl.load(y_ptr + row * D + src).to(tl.float32) * signs
    if NORMALIZE:
        values = values / denom
    return _fwht(values, _CHUNK, _CHUNK_STAGES)


@triton.jit
def _rotated_chunks(
    y_ptr,
    perm_ptr,
    signs_ptr,
    row,
    start,
    denom,
    D: tl.constexpr,
    CHUNKS: tl.constexpr,
    CROSS: tl.constexpr,
    NORMALIZE: tl.constexpr,
):
    # CHUNKS consecutive chunks from ``start``; CROSS stages (widths 4096, 8192) pair chunks.
    v0 = _rotated_chunk(y_ptr, perm_ptr, signs_ptr, row, start, denom, D, NORMALIZE)
    v1 = _rotated_chunk(y_ptr, perm_ptr, signs_ptr, row, start + _CHUNK, denom, D, NORMALIZE)
    v2 = v0
    v3 = v1
    if CHUNKS == 4:
        v2 = _rotated_chunk(
            y_ptr, perm_ptr, signs_ptr, row, start + 2 * _CHUNK, denom, D, NORMALIZE
        )
        v3 = _rotated_chunk(
            y_ptr, perm_ptr, signs_ptr, row, start + 3 * _CHUNK, denom, D, NORMALIZE
        )
    if CROSS > 0:
        v0, v1 = v0 + v1, v0 - v1
        if CHUNKS == 4:
            v2, v3 = v2 + v3, v2 - v3
    if CROSS > 1:
        v0, v2 = v0 + v2, v0 - v2
        v1, v3 = v1 + v3, v1 - v3
    return v0, v1, v2, v3


@triton.jit
def _absmax_int8_store(values, q_ptr, row, start, scale, D: tl.constexpr):
    q = libdevice.rint(libdevice.div_rn(values, scale))
    q = tl.minimum(tl.maximum(q, -127.0), 127.0)
    tl.store(q_ptr + row * D + start + tl.arange(0, _CHUNK), q.to(tl.int8))


@triton.jit
def _rpbh_absmax_int8_kernel(
    y_ptr,
    perm_ptr,
    signs_ptr,
    q_ptr,
    scale_ptr,
    D: tl.constexpr,
    DP2: tl.constexpr,
    STAGES: tl.constexpr,
    CHUNKS: tl.constexpr,
    inv_sqrt_block,
):
    # One row per program: the INT8 scale needs the absolute maximum of the whole rotated row.
    # D is a multiple of the RPBH block, so the butterflies of the padded row never pair a
    # channel with another block. Blocks wider than a chunk imply D == DP2 (no padding).
    row = tl.program_id(0).to(tl.int64)
    if CHUNKS == 1:
        cols = tl.arange(0, DP2)
        mask = cols < D
        src = tl.load(perm_ptr + cols, mask=mask, other=0).to(tl.int32)
        signs = tl.load(signs_ptr + cols, mask=mask, other=0).to(tl.float32)
        values = tl.load(y_ptr + row * D + src, mask=mask, other=0.0).to(tl.float32) * signs
        values = _fwht(values, DP2, STAGES) * inv_sqrt_block
        scale = libdevice.div_rn(tl.maximum(tl.max(tl.abs(values), axis=0), 1e-12), 127.0)
        q = libdevice.rint(libdevice.div_rn(values, scale))
        q = tl.minimum(tl.maximum(q, -127.0), 127.0)
        tl.store(q_ptr + row * D + cols, q.to(tl.int8), mask=mask)
    else:
        v0, v1, v2, v3 = _rotated_chunks(
            y_ptr, perm_ptr, signs_ptr, row, 0, 1.0, D, CHUNKS, STAGES - _CHUNK_STAGES, False
        )
        v0 *= inv_sqrt_block
        v1 *= inv_sqrt_block
        v2 *= inv_sqrt_block
        v3 *= inv_sqrt_block
        peak = tl.maximum(tl.max(tl.abs(v0), axis=0), tl.max(tl.abs(v1), axis=0))
        if CHUNKS == 4:
            peak = tl.maximum(
                peak, tl.maximum(tl.max(tl.abs(v2), axis=0), tl.max(tl.abs(v3), axis=0))
            )
        scale = libdevice.div_rn(tl.maximum(peak, 1e-12), 127.0)
        _absmax_int8_store(v0, q_ptr, row, 0, scale, D)
        _absmax_int8_store(v1, q_ptr, row, _CHUNK, scale, D)
        if CHUNKS == 4:
            _absmax_int8_store(v2, q_ptr, row, 2 * _CHUNK, scale, D)
            _absmax_int8_store(v3, q_ptr, row, 3 * _CHUNK, scale, D)
    tl.store(scale_ptr + row, scale)


@triton.jit
def _codes_store(
    values,
    bounds_ptr,
    codes_ptr,
    q_ptr,
    row,
    start,
    inv_sqrt_block,
    D: tl.constexpr,
    WIDTH: tl.constexpr,
    LEVELS: tl.constexpr,
):
    values *= inv_sqrt_block
    indices = tl.zeros((WIDTH,), dtype=tl.int32)
    for level in tl.static_range(0, LEVELS - 1):
        indices += (values > tl.load(bounds_ptr + level)).to(tl.int32)
    tl.store(
        q_ptr + row * D + start + tl.arange(0, WIDTH), tl.load(codes_ptr + indices).to(tl.int8)
    )


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
    norm = tl.load(norms_ptr + row).to(tl.float32)
    if OB <= _CHUNK:
        cols = block * OB + tl.arange(0, OB)
        src = tl.load(perm_ptr + cols).to(tl.int32)
        signs = tl.load(signs_ptr + cols).to(tl.float32)
        values = tl.load(y_ptr + row * D + src).to(tl.float32)
        values = _fwht(values * signs / (norm + eps), OB, STAGES)
        _codes_store(
            values, bounds_ptr, codes_ptr, q_ptr, row, block * OB, inv_sqrt_block, D, OB, LEVELS
        )
    else:
        start = block * OB
        v0, v1, v2, v3 = _rotated_chunks(
            y_ptr,
            perm_ptr,
            signs_ptr,
            row,
            start,
            norm + eps,
            D,
            OB // _CHUNK,
            STAGES - _CHUNK_STAGES,
            True,
        )
        _codes_store(
            v0, bounds_ptr, codes_ptr, q_ptr, row, start, inv_sqrt_block, D, _CHUNK, LEVELS
        )
        _codes_store(
            v1,
            bounds_ptr,
            codes_ptr,
            q_ptr,
            row,
            start + _CHUNK,
            inv_sqrt_block,
            D,
            _CHUNK,
            LEVELS,
        )
        if OB // _CHUNK == 4:
            _codes_store(
                v2,
                bounds_ptr,
                codes_ptr,
                q_ptr,
                row,
                start + 2 * _CHUNK,
                inv_sqrt_block,
                D,
                _CHUNK,
                LEVELS,
            )
            _codes_store(
                v3,
                bounds_ptr,
                codes_ptr,
                q_ptr,
                row,
                start + 3 * _CHUNK,
                inv_sqrt_block,
                D,
                _CHUNK,
                LEVELS,
            )


class ActivationQuantizer:
    """RPBH/Lloyd-Max INT8 surrogate activation quantizer of one OrbitQuantLinear input.

    Projections that share an input (Q/K/V/gate, gate/up) share one quantizer: the codes and
    token norms are identical, so they are produced once. Without a codebook the rotated input
    is quantized to INT8 with one absmax scale per token (8-bit activations) and the call
    returns that scale instead of the token norm.
    """

    def __init__(self, rotation, codebook, eps: float, device: torch.device):
        self.dim = int(rotation.dim)
        self.block = int(rotation.block_size)
        self.num_blocks = int(rotation.num_blocks)
        if self.block > 16384:
            raise ValueError("fused activation quantization supports RPBH blocks up to 16384")
        self.inv_sqrt_block = float(rotation.normalization)
        self.eps = float(eps)
        self.perm = rotation.permutation.to(device=device, dtype=torch.int32).contiguous()
        self.signs = rotation.signs.to(device=device, dtype=torch.int8).contiguous()
        self.absmax = codebook is None
        if self.absmax:
            if self.block > FWHT_CHUNK and self.dim not in (2 * FWHT_CHUNK, 4 * FWHT_CHUNK):
                raise ValueError(
                    "8-bit activations with RPBH blocks above 4096 need an 8192 or 16384 wide input"
                )
            self.levels, self.scale = 256, 1.0
            return
        self.bounds = codebook.boundaries.to(device=device, dtype=torch.float32).contiguous()
        self.levels = int(codebook.centroids.numel())
        codes, self.scale = fit_int8_centroid_surrogate(codebook.centroids)
        self.codes = codes.to(device=device, dtype=torch.int8).contiguous()

    def __call__(
        self,
        x,
        *,
        prologue=PROLOGUE_NONE,
        norm=None,
        mod=None,
        x2=None,
        norm_weight=None,
        mod_scale=None,
        mod_shift=None,
        mod_index=None,
        rms_eps=1e-5,
        prologue_only=False,
    ):
        """Return ``(codes int8 [rows, dim], token_norms fp32 [rows])``, or with
        ``prologue_only`` the bf16 prologue output instead of the codes.

        ``prologue`` selects a preset (``PROLOGUE_MULTIPLY`` multiplies by ``x2``); ``norm`` and
        ``mod`` override its norm and modulation. With ``mod_index`` the modulation vectors are
        rows ``mod_index[row]`` of the ``mod_scale``/``mod_shift`` tables.
        """
        if x.dim() != 2 or x.stride(1) != 1:
            x = x.reshape(-1, self.dim).contiguous()
        rows = x.shape[0]
        multiply = prologue == PROLOGUE_MULTIPLY
        if multiply:
            norm_mode, mod_mode = NORM_NONE, MOD_NONE
        else:
            norm_mode, mod_mode = _PROLOGUES[prologue]
        if norm is not None:
            norm_mode = norm
        if mod is not None:
            mod_mode = mod
        if x2 is not None and (x2.dim() != 2 or x2.stride(1) != 1):
            x2 = x2.reshape(rows, self.dim).contiguous()
        if mod_mode != MOD_NONE:
            # Per-tensor vectors are [dim]; indexed tables are [rows, dim] sharing one stride.
            mod_scale = mod_scale.reshape(-1, self.dim).contiguous()
            if mod_shift is not None:
                mod_shift = mod_shift.reshape(-1, self.dim).contiguous()
        y = torch.empty((rows, self.dim), device=x.device, dtype=torch.bfloat16)
        norms = torch.empty(rows, device=x.device, dtype=torch.float32)
        q = torch.empty((rows, self.dim), device=x.device, dtype=torch.int8)
        if rows == 0:
            return (y if prologue_only else q), norms
        _prologue_kernel[(rows,)](
            x,
            x2 if x2 is not None else x,
            norm_weight if norm_weight is not None else norms,
            mod_scale if mod_scale is not None else x,
            mod_shift if mod_shift is not None else x,
            mod_index if mod_index is not None else norms,
            y,
            norms,
            x.stride(0),
            x2.stride(0) if x2 is not None else 0,
            mod_scale.stride(0) if mod_index is not None else 0,
            D=self.dim,
            DP2=triton.next_power_of_2(self.dim),
            rms_eps=float(rms_eps),
            MULTIPLY=multiply,
            NORM=int(norm_mode),
            MOD=int(mod_mode),
            ROW_INDEXED=mod_index is not None,
            num_warps=8,
            # Under FP contraction the compiler folds the bf16 round trips that mirror the
            # eager ops' rounding (extf(truncf(x)) -> x) and fuses the multiply-adds.
            enable_fp_fusion=False,
        )
        if prologue_only:
            return y, norms
        if self.absmax:
            row_width = triton.next_power_of_2(self.dim)
            _rpbh_absmax_int8_kernel[(rows,)](
                y,
                self.perm,
                self.signs,
                q,
                norms,
                D=self.dim,
                DP2=row_width,
                STAGES=self.block.bit_length() - 1,
                CHUNKS=row_width // FWHT_CHUNK if self.block > FWHT_CHUNK else 1,
                inv_sqrt_block=self.inv_sqrt_block,
                num_warps=16 if row_width >= 16384 else 8,
            )
            return q, norms
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
            num_warps=16 if self.block >= 8192 else 8 if self.block >= 512 else 4,
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


QK_WEIGHT_NONE = 0
QK_WEIGHT_ONE_PLUS = 1  # x * inv * (1 + w), one rounding (Krea 2)
QK_WEIGHT_DIFFUSERS = 2  # bf16(x * inv) * w (diffusers RMSNorm)
QK_WEIGHT_TORCH = 3  # x * inv * w, one rounding (torch.nn.RMSNorm)

ROPE_INTERLEAVED = 0  # rotates channel pairs (2i, 2i + 1)
ROPE_HALF = 1  # rotate_half: channel i pairs with i + rotary_dim / 2


@triton.jit
def _qk_normalize(x, inv, w_ptr, offs, mask, WEIGHT: tl.constexpr):
    if WEIGHT == 1:
        return _bf16(x * inv * (tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32) + 1.0))
    if WEIGHT == 2:
        return _bf16(_bf16(x * inv) * tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32))
    if WEIGHT == 3:
        return _bf16(x * inv * tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32))
    return _bf16(x * inv)


@triton.jit
def _qk_norm_rope_kernel(
    src_ptr,
    src_stride,
    out_ptr,
    out_row_stride,
    out_head_stride,
    w_ptr,
    cos_ptr,
    sin_ptr,
    rope_stride,
    rms_eps,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    OUT_DIM: tl.constexpr,
    DP2: tl.constexpr,
    ROT: tl.constexpr,
    WEIGHT: tl.constexpr,
    STYLE: tl.constexpr,
    ROPE_BF16: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    offs = tl.arange(0, DP2)
    mask = offs < HEAD_DIM
    base = src_ptr + row * src_stride + head * HEAD_DIM
    x = tl.load(base + offs, mask=mask, other=0.0).to(tl.float32)
    mean_sq = libdevice.div_rn(tl.sum(x * x, axis=0), HEAD_DIM * 1.0)
    inv = libdevice.rsqrt(mean_sq + rms_eps)
    y = _qk_normalize(x, inv, w_ptr, offs, mask, WEIGHT)
    if ROT > 0:
        rot_mask = offs < ROT
        if STYLE == 0:
            partner = offs ^ 1
            sign = tl.where(offs % 2 == 0, -1.0, 1.0)
        else:
            partner = tl.where(offs < ROT // 2, offs + ROT // 2, offs - ROT // 2)
            sign = tl.where(offs < ROT // 2, -1.0, 1.0)
        xp = tl.load(base + partner, mask=rot_mask, other=0.0).to(tl.float32)
        yp = _qk_normalize(xp, inv, w_ptr, partner, rot_mask, WEIGHT)
        cos = tl.load(cos_ptr + row * rope_stride + offs, mask=rot_mask, other=0.0).to(tl.float32)
        sin = tl.load(sin_ptr + row * rope_stride + offs, mask=rot_mask, other=0.0).to(tl.float32)
        if ROPE_BF16:
            rotated = _bf16(_bf16(y * _bf16(cos)) + _bf16(sign * yp * _bf16(sin)))
        else:
            rotated = y * cos + sign * yp * sin
        y = tl.where(rot_mask, rotated, y)
    y = tl.where(mask, y, 0.0)
    dst = out_ptr + row * out_row_stride + head * out_head_stride
    tl.store(dst + offs, y.to(tl.bfloat16), mask=offs < OUT_DIM)


def qk_norm_rope_with_triton(
    src,
    heads,
    head_dim,
    weight,
    cos,
    sin,
    rms_eps,
    *,
    weight_mode=QK_WEIGHT_ONE_PLUS,
    style=ROPE_INTERLEAVED,
    rotary_dim=None,
    rope_bf16=False,
    out=None,
    out_dim=None,
):
    """Per-head RMSNorm then rotary embedding.

    ``src`` is a ``[rows, >= heads * head_dim]`` strided view; ``cos``/``sin`` are
    ``[rows, rotary_dim]`` (``None`` skips the rotation). Channels past ``rotary_dim`` pass
    through. Returns ``[1, rows, heads, out_dim]`` bf16 (``out_dim >= head_dim`` zero-pads each
    head); ``out`` may be a row slice of a larger buffer. Multiplies and adds are not contracted
    into FMAs, as in the eager elementwise ops.
    """
    rows = src.shape[0]
    out_dim = head_dim if out_dim is None else int(out_dim)
    if out is None:
        out = torch.empty((1, rows, heads, out_dim), device=src.device, dtype=torch.bfloat16)
    rotary = 0 if cos is None else int(rotary_dim or cos.shape[-1])
    if rows:
        _qk_norm_rope_kernel[(rows, heads)](
            src,
            src.stride(0),
            out,
            out.stride(1),
            out.stride(2),
            weight if weight is not None else src,
            cos if cos is not None else src,
            sin if sin is not None else src,
            cos.stride(0) if cos is not None else 0,
            float(rms_eps),
            HEADS=heads,
            HEAD_DIM=head_dim,
            OUT_DIM=out_dim,
            DP2=triton.next_power_of_2(max(head_dim, out_dim)),
            ROT=rotary,
            WEIGHT=int(weight_mode) if weight is not None else QK_WEIGHT_NONE,
            STYLE=int(style),
            ROPE_BF16=bool(rope_bf16),
            num_warps=1 if head_dim <= 128 else 2,
            enable_fp_fusion=False,
        )
    return out


@triton.jit
def _postnorm_residual_kernel(
    res_ptr,
    x_ptr,
    w_ptr,
    gate_ptr,
    index_ptr,
    out_ptr,
    res_stride,
    x_stride,
    gate_stride,
    D: tl.constexpr,
    DP2: tl.constexpr,
    rms_eps,
    NORM: tl.constexpr,
    ROW_INDEXED: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, DP2)
    mask = offs < D
    x = tl.load(x_ptr + row * x_stride + offs, mask=mask, other=0.0).to(tl.float32)
    n = _normalize(x, w_ptr, offs, mask, D, rms_eps, NORM)
    gate_row = tl.load(index_ptr + row).to(tl.int64) * gate_stride if ROW_INDEXED else 0
    gate = tl.load(gate_ptr + gate_row + offs, mask=mask, other=0.0).to(tl.float32)
    res = tl.load(res_ptr + row * res_stride + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + row * res_stride + offs, (res + _bf16(gate * n)).to(tl.bfloat16), mask=mask)


def postnorm_residual_with_triton(
    residual, x, norm_weight, gate, rms_eps, *, norm=NORM_RMS, gate_index=None, out=None
):
    """``residual + gate * norm(x)`` row by row (sandwich-norm blocks), in place by default.

    ``gate`` is a ``[dim]`` vector, or a ``[rows, dim]`` table addressed by ``gate_index``.
    """
    dim = x.shape[-1]
    rows = x.numel() // dim
    x = x.reshape(rows, dim)
    if x.stride(1) != 1:
        x = x.contiguous()
    res = residual.reshape(rows, dim)
    out = res if out is None else out.reshape(rows, dim)
    gate = gate.reshape(-1, dim).contiguous()
    if rows:
        _postnorm_residual_kernel[(rows,)](
            res,
            x,
            norm_weight if norm_weight is not None else x,
            gate,
            gate_index if gate_index is not None else gate,
            out,
            res.stride(0),
            x.stride(0),
            gate.stride(0),
            D=dim,
            DP2=triton.next_power_of_2(dim),
            rms_eps=float(rms_eps),
            NORM=int(norm),
            ROW_INDEXED=gate_index is not None,
            num_warps=8,
            enable_fp_fusion=False,
        )
    return residual


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


@triton.jit
def _decode_lowbit_kernel(
    packed_ptr,
    codes_ptr,
    out_ptr,
    total_values,
    total_bytes,
    BITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total_values
    bit = offs * BITS
    first = bit >> 3
    shift = (bit & 7).to(tl.int32)
    low = tl.load(packed_ptr + first, mask=mask, other=0).to(tl.int32)
    high = tl.load(packed_ptr + first + 1, mask=mask & (first + 1 < total_bytes), other=0)
    word = low | (high.to(tl.int32) << 8)
    index = (word >> shift) & ((1 << BITS) - 1)
    tl.store(out_ptr + offs, tl.load(codes_ptr + index).to(tl.int8), mask=mask)


def decode_lowbit_to_int8_with_triton(packed, codes, bits, out_features, in_features, out=None):
    """Packed low-bit indices (LSB-first bit stream, ``orbitquant.packing``) -> INT8 surrogate
    weight ``[out, in]``."""
    if out is None:
        out = torch.empty((out_features, in_features), device=packed.device, dtype=torch.int8)
    if bits == 4:
        total = out_features * in_features // 2
        block = 1024
        _decode_w4_kernel[(triton.cdiv(total, block),)](
            packed.reshape(-1), codes, out, total, BLOCK=block
        )
        return out
    values = out_features * in_features
    block = 2048
    _decode_lowbit_kernel[(triton.cdiv(values, block),)](
        packed.reshape(-1), codes, out, values, packed.numel(), BITS=int(bits), BLOCK=block
    )
    return out


def decode_w4_to_int8_with_triton(packed, codes, out_features, in_features, out=None):
    """Packed W4 indices (low nibble first) -> INT8 surrogate weight ``[out, in]``."""
    return decode_lowbit_to_int8_with_triton(packed, codes, 4, out_features, in_features, out=out)
