from __future__ import annotations

import os
from contextlib import suppress
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

_ALGORITHM_VERSION = 1
_DISABLE_CACHE_VALUES = {"1", "true", "yes", "on"}


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
    algorithm_version: int = _ALGORITHM_VERSION

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


def _disk_cache_disabled() -> bool:
    return os.environ.get("ORBITQUANT_DISABLE_CODEBOOK_DISK_CACHE", "").lower() in (
        _DISABLE_CACHE_VALUES
    )


def _codebook_cache_root() -> Path | None:
    if _disk_cache_disabled():
        return None
    override = os.environ.get("ORBITQUANT_CODEBOOK_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home:
        return Path(xdg_cache_home).expanduser() / "orbitquant" / "codebooks"
    return Path.home() / ".cache" / "orbitquant" / "codebooks"


def codebook_cache_path(dim: int, bits: int) -> Path | None:
    root = _codebook_cache_root()
    if root is None:
        return None
    return root / f"lloyd_max_v{_ALGORITHM_VERSION}_dim{dim}_bits{bits}.safetensors"


def _load_cached_codebook(dim: int, bits: int) -> LloydMaxCodebook | None:
    path = codebook_cache_path(dim, bits)
    if path is None or not path.exists():
        return None
    try:
        tensors = load_file(path)
        version = int(tensors["algorithm_version"].item())
        centroids = tensors["centroids"].to(torch.float32)
        boundaries = tensors["boundaries"].to(torch.float32)
    except Exception:
        return None
    if version != _ALGORITHM_VERSION:
        return None
    if centroids.shape != (2**bits,) or boundaries.shape != (2**bits - 1,):
        return None
    return LloydMaxCodebook(
        dim=dim,
        bits=bits,
        centroids=centroids.cpu(),
        boundaries=boundaries.cpu(),
        algorithm_version=version,
    )


def _write_cached_codebook(codebook: LloydMaxCodebook) -> None:
    path = codebook_cache_path(codebook.dim, codebook.bits)
    if path is None:
        return
    tmp_path: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f"{path.name}.tmp")
        save_file(
            {
                "centroids": codebook.centroids.detach().cpu().to(torch.float32),
                "boundaries": codebook.boundaries.detach().cpu().to(torch.float32),
                "algorithm_version": torch.tensor(
                    [codebook.algorithm_version], dtype=torch.int32
                ),
            },
            tmp_path,
        )
        tmp_path.replace(path)
    except Exception:
        if tmp_path is not None:
            with suppress(Exception):
                tmp_path.unlink(missing_ok=True)


def _generate_codebook(dim: int, bits: int) -> LloydMaxCodebook:
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
        algorithm_version=_ALGORITHM_VERSION,
    )


@lru_cache(maxsize=256)
def get_codebook(dim: int, bits: int) -> LloydMaxCodebook:
    cached = _load_cached_codebook(dim, bits)
    if cached is not None:
        return cached
    codebook = _generate_codebook(dim, bits)
    _write_cached_codebook(codebook)
    return codebook


def clear_codebook_cache() -> None:
    get_codebook.cache_clear()
