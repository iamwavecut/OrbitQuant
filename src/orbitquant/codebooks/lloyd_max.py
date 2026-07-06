from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch


def _coordinate_density(grid: torch.Tensor, dim: int) -> torch.Tensor:
    if dim < 2:
        raise ValueError("dim must be at least 2")
    alpha = (dim - 3) / 2
    inside = (1 - grid.square()).clamp_min(0)
    if alpha == 0:
        density = torch.ones_like(grid)
    else:
        density = torch.exp(alpha * torch.log(inside.clamp_min(1e-45)))
    density[grid.abs() >= 1] = 0
    return density


@dataclass(frozen=True)
class LloydMaxCodebook:
    dim: int
    bits: int
    centroids: torch.Tensor
    boundaries: torch.Tensor
    algorithm_version: int = 1

    def quantize_indices(self, values: torch.Tensor) -> torch.Tensor:
        boundaries = self.boundaries.to(device=values.device, dtype=values.dtype)
        return torch.bucketize(values, boundaries).to(torch.uint8)

    def quantize(self, values: torch.Tensor) -> torch.Tensor:
        indices = self.quantize_indices(values).to(torch.long)
        centroids = self.centroids.to(device=values.device, dtype=values.dtype)
        return centroids[indices]


def _weighted_centroid(
    grid: torch.Tensor, density: torch.Tensor, left: float, right: float
) -> float:
    mask = (grid >= left) & (grid <= right)
    weights = density[mask]
    points = grid[mask]
    total = weights.sum()
    if total <= 0:
        return (left + right) / 2
    return float((points * weights).sum() / total)


@lru_cache(maxsize=256)
def get_codebook(dim: int, bits: int) -> LloydMaxCodebook:
    if bits <= 0 or bits > 8:
        raise ValueError("bits must be in [1, 8]")

    levels = 2**bits
    grid = torch.linspace(
        -1 + 1e-6, 1 - 1e-6, 20001, dtype=torch.float64, device="cpu"
    )
    density = _coordinate_density(grid, dim)
    density = density / density.sum()

    cdf = torch.cumsum(density, dim=0)
    quantiles = (torch.arange(levels, dtype=torch.float64, device="cpu") + 0.5) / levels
    centroids = torch.empty(levels, dtype=torch.float64, device="cpu")
    for idx, quantile in enumerate(quantiles):
        grid_idx = int(torch.searchsorted(cdf, quantile).clamp(max=grid.numel() - 1))
        centroids[idx] = grid[grid_idx]

    for _ in range(80):
        boundaries = (centroids[:-1] + centroids[1:]) / 2
        edges = torch.cat(
            (
                torch.tensor([-1.0], dtype=torch.float64, device="cpu"),
                boundaries,
                torch.tensor([1.0], dtype=torch.float64, device="cpu"),
            )
        )
        next_centroids = torch.empty_like(centroids)
        for idx in range(levels):
            next_centroids[idx] = _weighted_centroid(
                grid, density, float(edges[idx]), float(edges[idx + 1])
            )
        # Enforce exact symmetry; the target distribution is symmetric.
        next_centroids = (next_centroids - torch.flip(next_centroids, dims=[0])) / 2
        if torch.max(torch.abs(next_centroids - centroids)) < 1e-10:
            centroids = next_centroids
            break
        centroids = next_centroids

    boundaries = (centroids[:-1] + centroids[1:]) / 2
    return LloydMaxCodebook(
        dim=dim,
        bits=bits,
        centroids=centroids.to(torch.float32),
        boundaries=boundaries.to(torch.float32),
    )
