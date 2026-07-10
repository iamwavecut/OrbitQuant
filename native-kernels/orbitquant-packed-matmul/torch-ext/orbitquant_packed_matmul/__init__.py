from __future__ import annotations

import torch

from ._ops import ops

__all__ = ["matmul_packed_weight"]


def matmul_packed_weight(
    x: torch.Tensor,
    packed_weight_indices: torch.Tensor,
    row_norms: torch.Tensor,
    centroids: torch.Tensor,
    *,
    bits: int,
    out_features: int,
    in_features: int,
    bias: torch.Tensor | None = None,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 128,
) -> torch.Tensor:
    if bits <= 0 or bits > 8:
        raise ValueError("bits must be in [1, 8]")
    if x.shape[-1] != in_features:
        raise ValueError(f"expected input last dimension {in_features}, got {x.shape[-1]}")
    if block_m <= 0 or block_n <= 0 or block_k <= 0:
        raise ValueError("packed matmul tile sizes must be positive")

    original_shape = x.shape
    x_2d = x.contiguous().reshape(-1, in_features)
    out = torch.empty((x_2d.shape[0], out_features), device=x.device, dtype=x.dtype)
    packed = packed_weight_indices.to(device=x.device, dtype=torch.uint8).contiguous()
    auxiliary_dtype = torch.bfloat16 if x.device.type == "cuda" else torch.float32
    norms = row_norms.to(device=x.device, dtype=auxiliary_dtype).contiguous()
    centroid_values = centroids.to(device=x.device, dtype=torch.float32).contiguous()
    if bias is None:
        bias_values = torch.empty((1,), device=x.device, dtype=torch.float32)
        has_bias = False
    else:
        bias_dtype = x.dtype if x.device.type == "cuda" else torch.float32
        bias_values = bias.to(device=x.device, dtype=bias_dtype).contiguous()
        has_bias = True

    ops.matmul_packed_weight(
        out,
        x_2d,
        packed,
        norms,
        centroid_values,
        bias_values,
        has_bias,
        bits,
        out_features,
        in_features,
        block_m,
        block_n,
        block_k,
    )
    return out.reshape(*original_shape[:-1], out_features)
