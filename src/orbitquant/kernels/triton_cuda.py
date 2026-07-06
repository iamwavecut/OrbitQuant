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
