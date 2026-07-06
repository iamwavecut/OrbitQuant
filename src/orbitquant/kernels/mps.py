from __future__ import annotations

from functools import lru_cache

import torch

from orbitquant.codebooks import LloydMaxCodebook

_MPS_CODEBOOK_RESCALE_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

kernel void orbitquant_codebook_rescale(
    const device float* rotated [[buffer(0)]],
    const device float* norms [[buffer(1)]],
    const device float* centroids [[buffer(2)]],
    const device float* boundaries [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant int& total [[buffer(5)]],
    constant int& dim [[buffer(6)]],
    constant int& levels [[buffer(7)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid >= uint(total)) {
    return;
  }
  float value = rotated[tid];
  int index = 0;
  for (int idx = 0; idx < levels - 1; ++idx) {
    if (value > boundaries[idx]) {
      index += 1;
    }
  }
  uint row = tid / uint(dim);
  output[tid] = centroids[index] * norms[row];
}
"""


def mps_metal_available() -> bool:
    return bool(torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader"))


@lru_cache(maxsize=1)
def _codebook_rescale_shader():
    if not mps_metal_available():
        raise RuntimeError("MPS Metal shader backend is not available in this environment")
    return torch.mps.compile_shader(_MPS_CODEBOOK_RESCALE_SOURCE)


def quantize_rotated_activations_with_mps(
    rotated: torch.Tensor,
    norms: torch.Tensor,
    codebook: LloydMaxCodebook,
) -> torch.Tensor:
    if rotated.device.type != "mps":
        raise RuntimeError("mps backend requires MPS tensors")
    rotated_contiguous = rotated.to(torch.float32).contiguous()
    flat = rotated_contiguous.reshape(-1)
    if flat.numel() == 0:
        return torch.empty_like(rotated_contiguous, dtype=torch.float32)

    row_norms = norms.contiguous().reshape(-1).to(device=rotated.device, dtype=torch.float32)
    centroids = codebook.centroids.to(device=rotated.device, dtype=torch.float32).contiguous()
    boundaries = codebook.boundaries.to(device=rotated.device, dtype=torch.float32).contiguous()
    output = torch.empty_like(flat, dtype=torch.float32)

    shader = _codebook_rescale_shader()
    shader.orbitquant_codebook_rescale(
        flat,
        row_norms,
        centroids,
        boundaries,
        output,
        torch.tensor(flat.numel(), dtype=torch.int32, device=rotated.device),
        torch.tensor(rotated.shape[-1], dtype=torch.int32, device=rotated.device),
        torch.tensor(centroids.numel(), dtype=torch.int32, device=rotated.device),
        threads=[flat.numel(), 1, 1],
        group_size=[min(flat.numel(), 256), 1, 1],
    )
    return output.reshape_as(rotated_contiguous)
