from __future__ import annotations

import torch

from orbitquant.codebooks import LloydMaxCodebook
from orbitquant.rotations import RPBHRotation


def quantize_activations(
    x: torch.Tensor,
    *,
    rotation: RPBHRotation,
    codebook: LloydMaxCodebook,
    eps: float,
) -> torch.Tensor:
    original_dtype = x.dtype
    work = x.to(torch.float32)
    norms = work.norm(dim=-1, keepdim=True).clamp_min(eps)
    unit = work / norms
    rotated = rotation.apply_to_activations(unit)
    quantized = codebook.quantize(rotated)
    return (quantized * norms).to(original_dtype)
