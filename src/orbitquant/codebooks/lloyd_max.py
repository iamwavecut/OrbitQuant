from __future__ import annotations

import hashlib
import math
import os
from contextlib import suppress
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

_ALGORITHM_VERSION = 2
_SUPPORTED_ALGORITHM_VERSIONS = {1, 2}
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


def codebook_cache_path(
    dim: int, bits: int, algorithm_version: int = _ALGORITHM_VERSION
) -> Path | None:
    root = _codebook_cache_root()
    if root is None:
        return None
    return root / f"lloyd_max_v{algorithm_version}_dim{dim}_bits{bits}.safetensors"


def _codebook_checksum(
    *,
    dim: int,
    bits: int,
    algorithm_version: int,
    centroids: torch.Tensor,
    boundaries: torch.Tensor,
) -> str:
    digest = hashlib.sha256()
    digest.update(
        f"lloyd_max_v{algorithm_version}:dim={dim}:bits={bits}".encode("ascii")
    )
    for tensor in (centroids, boundaries):
        contiguous = tensor.detach().cpu().to(torch.float32).contiguous()
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def _checksum_tensor(value: str) -> torch.Tensor:
    return torch.tensor(list(value.encode("ascii")), dtype=torch.uint8)


def _checksum_string(value: torch.Tensor) -> str:
    return bytes(value.detach().cpu().to(torch.uint8).tolist()).decode("ascii")


def _valid_cached_codebook_tensors(
    *,
    dim: int,
    bits: int,
    algorithm_version: int,
    centroids: torch.Tensor,
    boundaries: torch.Tensor,
    checksum: str,
) -> bool:
    if centroids.shape != (2**bits,) or boundaries.shape != (2**bits - 1,):
        return False
    if centroids.dtype != torch.float32 or boundaries.dtype != torch.float32:
        return False
    if not torch.isfinite(centroids).all() or not torch.isfinite(boundaries).all():
        return False
    if not torch.all(centroids[1:] > centroids[:-1]):
        return False
    if boundaries.numel() and not torch.all(boundaries[1:] > boundaries[:-1]):
        return False
    if centroids.numel() and (centroids[0] < -1 or centroids[-1] > 1):
        return False
    if not torch.allclose(centroids, -torch.flip(centroids, dims=[0]), atol=1e-5):
        return False
    expected_boundaries = (centroids[:-1] + centroids[1:]) / 2
    if not torch.allclose(boundaries, expected_boundaries, atol=1e-6):
        return False
    expected_checksum = _codebook_checksum(
        dim=dim,
        bits=bits,
        algorithm_version=algorithm_version,
        centroids=centroids,
        boundaries=boundaries,
    )
    return checksum == expected_checksum


def _load_cached_codebook(
    dim: int, bits: int, algorithm_version: int
) -> LloydMaxCodebook | None:
    path = codebook_cache_path(dim, bits, algorithm_version)
    if path is None or not path.exists():
        return None
    try:
        tensors = load_file(path)
        version = int(tensors["algorithm_version"].item())
        cached_dim = int(tensors["dim"].item())
        cached_bits = int(tensors["bits"].item())
        centroids = tensors["centroids"].to(torch.float32)
        boundaries = tensors["boundaries"].to(torch.float32)
        checksum = _checksum_string(tensors["cache_checksum"])
    except Exception:
        return None
    if version != algorithm_version or cached_dim != dim or cached_bits != bits:
        return None
    if not _valid_cached_codebook_tensors(
        dim=dim,
        bits=bits,
        algorithm_version=algorithm_version,
        centroids=centroids,
        boundaries=boundaries,
        checksum=checksum,
    ):
        return None
    return LloydMaxCodebook(
        dim=dim,
        bits=bits,
        centroids=centroids.cpu(),
        boundaries=boundaries.cpu(),
        algorithm_version=version,
    )


def _write_cached_codebook(codebook: LloydMaxCodebook) -> None:
    path = codebook_cache_path(
        codebook.dim, codebook.bits, codebook.algorithm_version
    )
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
                "dim": torch.tensor([codebook.dim], dtype=torch.int32),
                "bits": torch.tensor([codebook.bits], dtype=torch.int32),
                "algorithm_version": torch.tensor(
                    [codebook.algorithm_version], dtype=torch.int32
                ),
                "cache_checksum": _checksum_tensor(
                    _codebook_checksum(
                        dim=codebook.dim,
                        bits=codebook.bits,
                        algorithm_version=codebook.algorithm_version,
                        centroids=codebook.centroids,
                        boundaries=codebook.boundaries,
                    )
                ),
            },
            tmp_path,
        )
        tmp_path.replace(path)
    except Exception:
        if tmp_path is not None:
            with suppress(Exception):
                tmp_path.unlink(missing_ok=True)


def _generate_legacy_v1_codebook(dim: int, bits: int) -> LloydMaxCodebook:
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
        algorithm_version=1,
    )


_BETA_EPSILON = 3e-14
_BETA_FPMIN = 1e-300


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _BETA_FPMIN:
        d = _BETA_FPMIN
    d = 1.0 / d
    result = d

    for iteration in range(1, 10_001):
        doubled = 2 * iteration
        coefficient = (
            iteration * (b - iteration) * x / ((qam + doubled) * (a + doubled))
        )
        d = 1.0 + coefficient * d
        if abs(d) < _BETA_FPMIN:
            d = _BETA_FPMIN
        c = 1.0 + coefficient / c
        if abs(c) < _BETA_FPMIN:
            c = _BETA_FPMIN
        d = 1.0 / d
        result *= d * c

        coefficient = -(
            (a + iteration)
            * (qab + iteration)
            * x
            / ((a + doubled) * (qap + doubled))
        )
        d = 1.0 + coefficient * d
        if abs(d) < _BETA_FPMIN:
            d = _BETA_FPMIN
        c = 1.0 + coefficient / c
        if abs(c) < _BETA_FPMIN:
            c = _BETA_FPMIN
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) < _BETA_EPSILON:
            return result

    raise RuntimeError("incomplete beta continued fraction did not converge")


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_factor = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    factor = math.exp(log_factor)
    if x < (a + 1.0) / (a + b + 2.0):
        return factor * _beta_continued_fraction(a, b, x) / a
    return 1.0 - factor * _beta_continued_fraction(b, a, 1.0 - x) / b


def _positive_coordinate_quantile(dim: int, probability: float) -> float:
    beta_b = (dim - 1) / 2
    lower = 0.0
    upper = 1.0
    for _ in range(96):
        midpoint = (lower + upper) / 2
        cdf = _regularized_incomplete_beta(0.5, beta_b, midpoint * midpoint)
        if cdf < probability:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2


def _positive_interval_centroid(dim: int, left: float, right: float) -> float:
    beta_b = (dim - 1) / 2
    left_cdf = _regularized_incomplete_beta(0.5, beta_b, left * left)
    if right >= 1.0:
        interval_probability = _regularized_incomplete_beta(
            beta_b, 0.5, 1.0 - left * left
        )
    else:
        right_cdf = _regularized_incomplete_beta(0.5, beta_b, right * right)
        interval_probability = right_cdf - left_cdf
    mass = 0.5 * interval_probability
    if mass <= 0.0:
        raise RuntimeError("Lloyd-Max interval has zero probability mass")

    left_log = beta_b * math.log1p(-(left * left)) if left < 1.0 else -math.inf
    right_log = beta_b * math.log1p(-(right * right)) if right < 1.0 else -math.inf
    unnormalized_moment = (
        0.0
        if left_log == -math.inf
        else math.exp(left_log) * -math.expm1(right_log - left_log)
    )
    density_normalization = math.exp(
        math.lgamma(dim / 2) - 0.5 * math.log(math.pi) - math.lgamma(beta_b)
    )
    moment = density_normalization * unnormalized_moment / (2 * beta_b)
    return moment / mass


def _generate_exact_v2_codebook(dim: int, bits: int) -> LloydMaxCodebook:
    levels = 2**bits
    positive_levels = levels // 2
    positive_centroids = [
        _positive_coordinate_quantile(dim, (index + 0.5) / positive_levels)
        for index in range(positive_levels)
    ]

    tolerance = 1e-13
    max_iterations = max(1_024, 6 * levels * levels)
    for _ in range(max_iterations):
        edges = [0.0]
        edges.extend(
            (positive_centroids[index] + positive_centroids[index + 1]) / 2
            for index in range(positive_levels - 1)
        )
        edges.append(1.0)
        next_centroids = [
            _positive_interval_centroid(dim, edges[index], edges[index + 1])
            for index in range(positive_levels)
        ]
        delta = max(
            abs(next_value - old_value)
            for next_value, old_value in zip(
                next_centroids, positive_centroids, strict=True
            )
        )
        if delta < tolerance:
            positive_centroids = next_centroids
            break
        accelerated = [
            old_value + 1.9 * (next_value - old_value)
            for next_value, old_value in zip(
                next_centroids, positive_centroids, strict=True
            )
        ]
        if all(
            0.0
            < accelerated[index]
            < (accelerated[index + 1] if index + 1 < positive_levels else 1.0)
            for index in range(positive_levels)
        ):
            positive_centroids = accelerated
        else:
            positive_centroids = next_centroids
    else:
        raise RuntimeError(
            f"Lloyd-Max did not converge for dim={dim}, bits={bits} after "
            f"{max_iterations} iterations"
        )

    centroids = torch.tensor(
        [-value for value in reversed(positive_centroids)] + positive_centroids,
        dtype=torch.float64,
        device="cpu",
    )
    boundaries = (centroids[:-1] + centroids[1:]) / 2
    return LloydMaxCodebook(
        dim=dim,
        bits=bits,
        centroids=centroids.to(torch.float32),
        boundaries=boundaries.to(torch.float32),
        algorithm_version=2,
    )


def _generate_codebook(
    dim: int, bits: int, algorithm_version: int = _ALGORITHM_VERSION
) -> LloydMaxCodebook:
    if bits <= 0 or bits > 8:
        raise ValueError("bits must be in [1, 8]")
    if algorithm_version == 1:
        legacy = _generate_legacy_v1_codebook(dim, bits)
        return LloydMaxCodebook(
            dim=legacy.dim,
            bits=legacy.bits,
            centroids=legacy.centroids,
            boundaries=legacy.boundaries,
            algorithm_version=1,
        )
    if algorithm_version == 2:
        return _generate_exact_v2_codebook(dim, bits)
    raise ValueError(
        f"algorithm_version must be one of {sorted(_SUPPORTED_ALGORITHM_VERSIONS)}"
    )


@lru_cache(maxsize=256)
def get_codebook(
    dim: int, bits: int, algorithm_version: int = _ALGORITHM_VERSION
) -> LloydMaxCodebook:
    if algorithm_version not in _SUPPORTED_ALGORITHM_VERSIONS:
        raise ValueError(
            f"algorithm_version must be one of {sorted(_SUPPORTED_ALGORITHM_VERSIONS)}"
        )
    cached = _load_cached_codebook(dim, bits, algorithm_version)
    if cached is not None:
        return cached
    codebook = _generate_codebook(dim, bits, algorithm_version)
    _write_cached_codebook(codebook)
    return codebook


def clear_codebook_cache() -> None:
    get_codebook.cache_clear()
