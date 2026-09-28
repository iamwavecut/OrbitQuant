"""INT8 x INT8 GEMM with the scale epilogue fused (Triton).

The large-row W4A4 path used ``torch._int_mm`` (int32 result in global memory) followed by a
separate scale kernel. On GeForce cards the int32 round trip costs as much as a quarter of the
GEMM; this kernel keeps the accumulator in registers and writes the scaled result in the output
dtype. Integer accumulation is exact, and the epilogue multiplies in the same order as the
kernels it replaces, so the plain epilogue is bit-identical to them:

* ``scale_mode=0`` (OrbitQuant W4A4 surrogates): ``((acc * a_scale) * b_scale) * alpha + bias``
* ``scale_mode=1`` (per-row INT8, ``Int8RowLinear``): ``acc * (a_scale * b_scale) + bias``

Optional epilogues fuse the elementwise work that follows a projection in DiT blocks. Each
rounds to bf16 exactly where the eager ops it replaces do:

* ``EPILOGUE_SWIGLU``: rows of ``b`` interleave gate and up in chunks of ``BLOCK_N // 2``;
  the kernel writes ``bf16(silu(bf16(gate))) * bf16(up)`` (half as many columns).
* ``EPILOGUE_RESIDUAL_GATE``: ``out = res + bf16(gate[n] * bf16(y))`` (in place allowed).
* ``EPILOGUE_SIGMOID_TAIL``: columns ``>= sig_from`` store ``sigmoid(bf16(y))``.
"""

from __future__ import annotations

import inspect

import torch

from orbitquant.kernels.triton_cuda import _load_triton

triton, tl = _load_triton()

from triton.language.extra import libdevice  # noqa: E402

EPILOGUE_PLAIN = 0
EPILOGUE_SWIGLU = 1
EPILOGUE_RESIDUAL_GATE = 2
EPILOGUE_SIGMOID_TAIL = 3

# The SwiGLU epilogue pairs gate/up halves inside one N tile; weights packed for it interleave
# rows in chunks of this size.
SWIGLU_INTERLEAVE = 128

_SHAPES = [
    (128, 128, 64, 4, 4),
    (128, 128, 128, 4, 3),
    (128, 128, 128, 8, 3),
    (128, 256, 64, 8, 3),
    (256, 128, 64, 8, 3),
    (64, 256, 128, 4, 3),
    (128, 64, 128, 4, 4),
    (64, 128, 128, 4, 4),
]


def autotune(configs, key, restore_value=None):
    """``triton.autotune`` that stores its choices in the Triton cache, so a restarted process
    does not benchmark every configuration again (when the installed Triton supports it)."""
    extra = {}
    if "cache_results" in inspect.signature(triton.autotune).parameters:
        extra["cache_results"] = True
    if restore_value:
        extra["restore_value"] = restore_value
    return triton.autotune(configs=configs, key=key, **extra)


def _configs(block_n: int | None = None):
    return [
        triton.Config(
            {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": 8},
            num_warps=warps,
            num_stages=stages,
        )
        for bm, bn, bk, warps, stages in _SHAPES
        if block_n is None or bn == block_n
    ]


@triton.jit
def _bf16(x):
    return x.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _int8_scaled_gemm_body(
    a_ptr,
    b_ptr,
    out_ptr,
    a_scale_ptr,
    b_scale_ptr,
    bias_ptr,
    res_ptr,
    gate_ptr,
    M,
    N,
    K,
    stride_om,
    col_offset,
    M_BUCKET,
    alpha,
    sig_from,
    HAS_BIAS: tl.constexpr,
    SCALE_MODE: tl.constexpr,
    EPILOGUE: tl.constexpr,
    EVEN_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_n = offs_n < N
    a_ptrs = a_ptr + offs_m[:, None].to(tl.int64) * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :].to(tl.int64) * K + offs_k[:, None]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            a = tl.load(a_ptrs, mask=mask_m[:, None], other=0)
            b = tl.load(b_ptrs, mask=mask_n[None, :], other=0)
        else:
            k_mask = offs_k + k * BLOCK_K < K
            a = tl.load(a_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=0)
            b = tl.load(b_ptrs, mask=mask_n[None, :] & k_mask[:, None], other=0)
        acc = tl.dot(a, b, acc, out_dtype=tl.int32)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K

    a_s = tl.load(a_scale_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
    b_s = tl.load(b_scale_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    if SCALE_MODE == 1:
        y = acc.to(tl.float32) * (a_s[:, None] * b_s[None, :])
    else:
        y = acc.to(tl.float32) * a_s[:, None] * b_s[None, :] * alpha
    if HAS_BIAS:
        y += tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)[None, :]
    rows = offs_m[:, None].to(tl.int64) * stride_om
    if EPILOGUE == 1:
        pairs = tl.permute(tl.reshape(_bf16(y), (BLOCK_M, 2, BLOCK_N // 2)), (0, 2, 1))
        g, u = tl.split(pairs)
        h = _bf16(libdevice.div_rn(g, 1.0 + libdevice.exp(-g))) * u
        offs_h = col_offset + pid_n * (BLOCK_N // 2) + tl.arange(0, BLOCK_N // 2)
        tl.store(
            out_ptr + rows + offs_h[None, :],
            h.to(out_ptr.dtype.element_ty),
            mask=mask_m[:, None] & (offs_h[None, :] < col_offset + N // 2),
        )
    elif EPILOGUE == 2:
        cols = (col_offset + offs_n)[None, :]
        mask = mask_m[:, None] & mask_n[None, :]
        gate = tl.load(gate_ptr + col_offset + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        res = tl.load(res_ptr + rows + cols, mask=mask, other=0.0).to(tl.float32)
        out = res + _bf16(gate[None, :] * _bf16(y))
        tl.store(out_ptr + rows + cols, out.to(out_ptr.dtype.element_ty), mask=mask)
    elif EPILOGUE == 3:
        y = _bf16(y)
        if pid_n * BLOCK_N >= sig_from:
            y = libdevice.div_rn(1.0, 1.0 + libdevice.exp(-y))
        tl.store(
            out_ptr + rows + (col_offset + offs_n)[None, :],
            y.to(out_ptr.dtype.element_ty),
            mask=mask_m[:, None] & mask_n[None, :],
        )
    else:
        tl.store(
            out_ptr + rows + (col_offset + offs_n)[None, :],
            y.to(out_ptr.dtype.element_ty),
            mask=mask_m[:, None] & mask_n[None, :],
        )


_KEY = ["M_BUCKET", "N", "K", "SCALE_MODE", "HAS_BIAS", "EPILOGUE"]
# The residual-gate epilogue usually updates its output in place: every benchmark run of the
# autotuner would add the projection to the residual again.
_int8_scaled_gemm = autotune(_configs(), _KEY, restore_value=["out_ptr"])(_int8_scaled_gemm_body)
_int8_scaled_gemm_swiglu = autotune(_configs(block_n=2 * SWIGLU_INTERLEAVE), _KEY)(
    _int8_scaled_gemm_body
)


def _row_bucket(rows: int) -> int:
    return 1 << max(5, (rows - 1).bit_length())


def matmul_int8_scaled_with_triton(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    *,
    alpha: float = 1.0,
    bias: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    col_offset: int = 0,
    output_dtype: torch.dtype = torch.bfloat16,
    scale_mode: int = 0,
    epilogue: int = EPILOGUE_PLAIN,
    residual: torch.Tensor | None = None,
    gate: torch.Tensor | None = None,
    sig_from: int = 0,
) -> torch.Tensor:
    """Scaled ``a @ b.T`` for INT8 ``a [M, K]`` and ``b [N, K]``.

    ``out`` may be a wider row-major buffer; the result lands in columns
    ``[col_offset, col_offset + N)`` (``N // 2`` for SwiGLU) so chunked weight decodes write in
    place. The residual-gate epilogue reads ``residual`` with the same layout as ``out`` and may
    alias it.
    """
    if a.dtype != torch.int8 or b.dtype != torch.int8:
        raise ValueError("matmul_int8_scaled_with_triton expects INT8 operands")
    rows, k = a.shape
    n = b.shape[0]
    if b.shape[1] != k:
        raise ValueError(f"inner dimensions differ: {k} vs {b.shape[1]}")
    if epilogue == EPILOGUE_SWIGLU and n % (2 * SWIGLU_INTERLEAVE):
        raise ValueError(f"SwiGLU epilogue needs N divisible by {2 * SWIGLU_INTERLEAVE}")
    if epilogue == EPILOGUE_RESIDUAL_GATE and (residual is None or gate is None):
        raise ValueError("residual-gate epilogue needs residual and gate")
    out_cols = n // 2 if epilogue == EPILOGUE_SWIGLU else n
    if out is None:
        out = torch.empty((rows, out_cols), device=a.device, dtype=output_dtype)
    if rows == 0:
        return out
    if a.stride(1) != 1 or a.stride(0) != k or b.stride(1) != 1 or b.stride(0) != k:
        a = a.contiguous()
        b = b.contiguous()
    if residual is not None and residual.stride() != out.stride():
        raise ValueError("residual must share the output layout")
    a_scale = a_scale.to(torch.float32).contiguous()
    b_scale = b_scale.to(torch.float32).contiguous()
    kernel = _int8_scaled_gemm_swiglu if epilogue == EPILOGUE_SWIGLU else _int8_scaled_gemm

    def grid(meta):
        return (triton.cdiv(rows, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),)

    kernel[grid](
        a,
        b,
        out,
        a_scale,
        b_scale,
        bias if bias is not None else a_scale,
        residual if residual is not None else out,
        gate if gate is not None else a_scale,
        rows,
        n,
        k,
        out.stride(0),
        int(col_offset),
        _row_bucket(rows),
        float(alpha),
        int(sig_from),
        HAS_BIAS=bias is not None,
        SCALE_MODE=int(scale_mode),
        EPILOGUE=int(epilogue),
        EVEN_K=k % 128 == 0,
    )
    return out
