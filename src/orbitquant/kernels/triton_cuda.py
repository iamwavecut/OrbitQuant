from __future__ import annotations

import torch

from orbitquant.codebooks import LloydMaxCodebook


def _load_triton():
    try:
        import triton
        import triton.language as tl
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Triton is required for the triton_cuda backend") from exc
    return triton, tl


triton, tl = _load_triton()


@triton.jit
def _codebook_rescale_kernel(
    rotated_ptr,
    norms_ptr,
    centroids_ptr,
    boundaries_ptr,
    output_ptr,
    total: tl.constexpr,
    dim: tl.constexpr,
    block_size: tl.constexpr,
    levels: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < total
    values = tl.load(rotated_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    indices = tl.zeros((block_size,), dtype=tl.int32)
    for idx in tl.static_range(0, levels - 1):
        boundary = tl.load(boundaries_ptr + idx)
        indices += (values > boundary).to(tl.int32)
    centroids = tl.load(centroids_ptr + indices)
    rows = offsets // dim
    norms = tl.load(norms_ptr + rows, mask=mask, other=0.0).to(tl.float32)
    tl.store(output_ptr + offsets, centroids * norms, mask=mask)


@triton.jit
def _dequantize_packed_weight_kernel(
    packed_ptr,
    row_norms_ptr,
    centroids_ptr,
    output_ptr,
    total: tl.constexpr,
    in_features: tl.constexpr,
    bits: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < total
    bit_starts = offsets * bits
    byte_indices = bit_starts // 8
    bit_offsets = bit_starts % 8
    low = tl.load(packed_ptr + byte_indices, mask=mask, other=0).to(tl.uint32)
    needs_high = bit_offsets + bits > 8
    high = tl.load(packed_ptr + byte_indices + 1, mask=mask & needs_high, other=0).to(
        tl.uint32
    )
    raw = low | (high << 8)
    indices = (raw >> bit_offsets) & ((1 << bits) - 1)
    rows = offsets // in_features
    norms = tl.load(row_norms_ptr + rows, mask=mask, other=0.0).to(tl.float32)
    centroids = tl.load(centroids_ptr + indices)
    tl.store(output_ptr + offsets, centroids * norms, mask=mask)


def quantize_rotated_activations_with_triton(
    rotated: torch.Tensor,
    norms: torch.Tensor,
    codebook: LloydMaxCodebook,
) -> torch.Tensor:
    if not rotated.is_cuda:
        raise RuntimeError("triton_cuda backend requires CUDA tensors")
    triton, _ = _load_triton()
    rotated_contiguous = rotated.contiguous()
    flat = rotated_contiguous.reshape(-1)
    row_norms = norms.contiguous().reshape(-1).to(device=rotated.device, dtype=torch.float32)
    centroids = codebook.centroids.to(device=rotated.device, dtype=torch.float32).contiguous()
    boundaries = codebook.boundaries.to(device=rotated.device, dtype=torch.float32).contiguous()
    output = torch.empty_like(flat, dtype=torch.float32)
    block_size = 256
    grid = (triton.cdiv(flat.numel(), block_size),)
    _codebook_rescale_kernel[grid](
        flat,
        row_norms,
        centroids,
        boundaries,
        output,
        total=flat.numel(),
        dim=rotated.shape[-1],
        block_size=block_size,
        levels=centroids.numel(),
    )
    return output.reshape_as(rotated_contiguous)


def dequantize_packed_weight_with_triton(
    packed_weight_indices: torch.Tensor,
    row_norms: torch.Tensor,
    codebook: LloydMaxCodebook,
    *,
    bits: int,
    out_features: int,
    in_features: int,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("triton_cuda backend requires CUDA tensors")
    triton, _ = _load_triton()
    target_device = torch.device(device)
    if target_device.type != "cuda":
        raise RuntimeError("triton_cuda weight dequant requires a CUDA device")
    total = out_features * in_features
    packed = packed_weight_indices.to(device=target_device, dtype=torch.uint8).contiguous()
    norms = row_norms.to(device=target_device, dtype=torch.float32).contiguous()
    centroids = codebook.centroids.to(device=target_device, dtype=torch.float32).contiguous()
    output = torch.empty(total, device=target_device, dtype=torch.float32)
    if total == 0:
        return output.reshape(out_features, in_features)

    block_size = 256
    grid = (triton.cdiv(total, block_size),)
    _dequantize_packed_weight_kernel[grid](
        packed,
        norms,
        centroids,
        output,
        total=total,
        in_features=in_features,
        bits=bits,
        block_size=block_size,
    )
    return output.reshape(out_features, in_features)
