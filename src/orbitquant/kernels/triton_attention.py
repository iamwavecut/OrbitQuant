"""Non-causal grouped-query attention with INT8 Q.K^T and FP16-accumulated P.V (Triton).

This is the SageAttention (v1) scheme. GeForce Ada and Blackwell cards run BF16/FP16 tensor ops
with an FP32 accumulator at half the FP16-accumulator rate and INT8 at four times the BF16
rate, so FlashAttention-style BF16 kernels leave most of the tensor throughput unused:

* Q and K are quantized to INT8 per block of tokens and head; K first loses its per-channel
  mean over the sequence, which shifts every score of a query by the same amount and leaves
  the softmax unchanged. The softmax scale and ``log2(e)`` are folded into the Q scale so the
  kernel exponentiates with ``exp2``.
* The online softmax runs in FP32.
* P and V are FP16; each ``BLOCK_N``-token tile of ``P @ V`` accumulates in FP16 and is added
  to an FP32 accumulator. V is converted once up front: converting each tile inside the loop
  costs about 10% of the kernel time.

Tensors use the ``[batch=1, tokens, heads, head_dim]`` layout of DiT projections; rows may be
strided (e.g. a column slice of a fused Q|K|V GEMM output). Keys and values may be a different
(e.g. compacted, padding-free) sequence than the queries.
"""

from __future__ import annotations

import math

import torch

from orbitquant.kernels.triton_cuda import _load_triton
from orbitquant.kernels.triton_int8_gemm import autotune

triton, tl = _load_triton()

LOG2E = 1.4426950408889634
BLOCK_Q = 128
BLOCK_KV = 64


@triton.jit
def _quantize_blocks_kernel(
    x_ptr,
    mean_ptr,
    out_ptr,
    scale_ptr,
    rows,
    stride_xr,
    stride_xh,
    stride_or,
    stride_oh,
    multiplier,
    HAS_MEAN: tl.constexpr,
    BLOCK: tl.constexpr,
    D: tl.constexpr,
):
    block = tl.program_id(0)
    head = tl.program_id(1)
    offs_r = block * BLOCK + tl.arange(0, BLOCK)
    offs_d = tl.arange(0, D)
    mask = offs_r[:, None] < rows
    rows_x = offs_r[:, None].to(tl.int64) * stride_xr + head * stride_xh
    x = tl.load(x_ptr + rows_x + offs_d[None, :], mask=mask, other=0.0).to(tl.float32)
    if HAS_MEAN:
        x = tl.where(mask, x - tl.load(mean_ptr + head * D + offs_d)[None, :], 0.0)
    x = x * multiplier
    scale = tl.maximum(tl.max(tl.max(tl.abs(x), axis=1), axis=0) / 127.0, 1e-20)
    q = x / scale
    q = tl.where(q >= 0, q + 0.5, q - 0.5).to(tl.int8)
    rows_o = offs_r[:, None].to(tl.int64) * stride_or + head * stride_oh
    tl.store(out_ptr + rows_o + offs_d[None, :], q, mask=mask)
    tl.store(scale_ptr + head * tl.cdiv(rows, BLOCK) + block, scale)


@triton.jit
def _attention_tile(
    acc,
    l_i,
    m_i,
    q,
    qk_scale,
    k_ptrs,
    v_ptrs,
    n_valid,
    MASKED: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offs_n = tl.arange(0, BLOCK_N)
    k = tl.load(k_ptrs, mask=offs_n[None, :] < n_valid, other=0) if MASKED else tl.load(k_ptrs)
    qk = tl.dot(q, k, out_dtype=tl.int32).to(tl.float32) * qk_scale
    if MASKED:
        qk = tl.where(offs_n[None, :] < n_valid, qk, float("-inf"))
    m_new = tl.maximum(m_i, tl.max(qk, 1))
    p = tl.math.exp2(qk - m_new[:, None])
    alpha = tl.math.exp2(m_i - m_new)
    l_i = l_i * alpha + tl.sum(p, 1)
    acc = acc * alpha[:, None]
    v = tl.load(v_ptrs, mask=offs_n[:, None] < n_valid, other=0.0) if MASKED else tl.load(v_ptrs)
    acc += tl.dot(p.to(tl.float16), v, out_dtype=tl.float16).to(tl.float32)
    return acc, l_i, m_new


@triton.jit
def _attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    q_scale_ptr,
    k_scale_ptr,
    o_ptr,
    seq,
    seq_kv,
    stride_qr,
    stride_qh,
    stride_kr,
    stride_kh,
    stride_vr,
    stride_vh,
    stride_or,
    stride_oh,
    GROUPS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    head = tl.program_id(1)
    kv_head = head // GROUPS
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    q_rows = offs_m[:, None].to(tl.int64) * stride_qr + head * stride_qh
    q = tl.load(q_ptr + q_rows + offs_d[None, :], mask=offs_m[:, None] < seq, other=0)
    q_scale = tl.load(q_scale_ptr + head * tl.cdiv(seq, BLOCK_M) + start_m)
    k_scales = k_scale_ptr + kv_head * tl.cdiv(seq_kv, BLOCK_N)
    k_ptrs = k_ptr + offs_n[None, :].to(tl.int64) * stride_kr + kv_head * stride_kh
    k_ptrs += offs_d[:, None]
    v_ptrs = v_ptr + offs_n[:, None].to(tl.int64) * stride_vr + kv_head * stride_vh
    v_ptrs += offs_d[None, :]
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)
    full = seq_kv // BLOCK_N
    for block in range(0, full):
        qk_scale = q_scale * tl.load(k_scales + block)
        acc, l_i, m_i = _attention_tile(
            acc, l_i, m_i, q, qk_scale, k_ptrs, v_ptrs, BLOCK_N, False, BLOCK_N
        )
        k_ptrs += BLOCK_N * stride_kr
        v_ptrs += BLOCK_N * stride_vr
    if full * BLOCK_N < seq_kv:
        qk_scale = q_scale * tl.load(k_scales + full)
        acc, l_i, m_i = _attention_tile(
            acc, l_i, m_i, q, qk_scale, k_ptrs, v_ptrs, seq_kv - full * BLOCK_N, True, BLOCK_N
        )
    out = acc / l_i[:, None]
    o_rows = offs_m[:, None].to(tl.int64) * stride_or + head * stride_oh
    tl.store(
        o_ptr + o_rows + offs_d[None, :],
        out.to(o_ptr.dtype.element_ty),
        mask=offs_m[:, None] < seq,
    )


_attention = autotune(
    [triton.Config({}, num_warps=w, num_stages=s) for w in (4, 8) for s in (2, 3, 4)],
    ["GROUPS", "D", "BLOCK_M"],
)(_attention_kernel)


def _query_block(dim: int) -> int:
    # The FP32 accumulator is BLOCK_M x D: wide heads take shorter query tiles.
    return BLOCK_Q if dim <= 128 else 64


def _quantize_blocks(x, block, multiplier, mean=None):
    rows, heads, dim = x.shape
    out = torch.empty((rows, heads, dim), device=x.device, dtype=torch.int8)
    scales = torch.empty((heads, triton.cdiv(rows, block)), device=x.device, dtype=torch.float32)
    _quantize_blocks_kernel[(triton.cdiv(rows, block), heads)](
        x,
        mean if mean is not None else scales,
        out,
        scales,
        rows,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        float(multiplier),
        HAS_MEAN=mean is not None,
        BLOCK=block,
        D=dim,
        num_warps=4,
    )
    return out, scales


def int8_attention_with_triton(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """``softmax(q k^T * sm_scale) v`` for ``[1, S, Hq, D]`` queries and ``[1, Skv, Hkv, D]``
    keys/values (``Hq`` a multiple of ``Hkv``, ``D`` a power of two, last dim contiguous).
    Returns ``[1, S, Hq, D]`` in the query dtype. A head padded with zero channels to a power of
    two gives the unpadded result in its leading channels when ``sm_scale`` is the unpadded one."""
    if query.dim() != 4 or query.shape[0] != 1:
        raise ValueError("expected [1, tokens, heads, head_dim] tensors")
    q, k, v = query[0], key[0], value[0]
    seq, heads, dim = q.shape
    seq_kv, kv_heads = k.shape[0], k.shape[1]
    if heads % kv_heads:
        raise ValueError(f"{heads} query heads are not a multiple of {kv_heads} kv heads")
    if dim & (dim - 1) or q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError("head_dim must be a power of two and contiguous")
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(dim)
    block_q = _query_block(dim)
    q_codes, q_scales = _quantize_blocks(q, block_q, sm_scale * LOG2E)
    k_codes, k_scales = _quantize_blocks(k, BLOCK_KV, 1.0, k.mean(dim=0, dtype=torch.float32))
    v16 = v if v.dtype == torch.float16 else v.to(torch.float16)
    out = torch.empty((1, seq, heads, dim), device=q.device, dtype=query.dtype)
    o = out[0]
    _attention[(triton.cdiv(seq, block_q), heads)](
        q_codes,
        k_codes,
        v16,
        q_scales,
        k_scales,
        o,
        seq,
        seq_kv,
        q_codes.stride(0),
        q_codes.stride(1),
        k_codes.stride(0),
        k_codes.stride(1),
        v16.stride(0),
        v16.stride(1),
        o.stride(0),
        o.stride(1),
        GROUPS=heads // kv_heads,
        D=dim,
        BLOCK_M=block_q,
        BLOCK_N=BLOCK_KV,
    )
    return out
