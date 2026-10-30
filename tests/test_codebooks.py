import time

import pytest
import torch
from safetensors.torch import load_file, save_file

import orbitquant.codebooks.lloyd_max as lloyd_max_module
from orbitquant.codebooks import clear_codebook_cache, codebook_cache_path, get_codebook


def test_coordinate_density_matches_unit_sphere_marginal_shape():
    grid = torch.tensor([-1.0, -0.5, 0.0, 0.75, 1.0], dtype=torch.float64)

    density = lloyd_max_module._coordinate_density(grid, dim=7)

    expected_shape = (1 - grid.square()).clamp_min(0).square()
    expected_shape[grid.abs() >= 1] = 0
    assert torch.allclose(density, expected_shape)
    assert density[2] > density[1] > density[3] > density[0]
    assert density[0] == 0
    assert density[-1] == 0


def test_lloyd_max_codebook_is_symmetric_and_sorted():
    codebook = get_codebook(dim=32, bits=4)

    assert codebook.dim == 32
    assert codebook.bits == 4
    assert codebook.centroids.shape == (16,)
    assert codebook.boundaries.shape == (15,)
    assert torch.all(codebook.centroids[1:] > codebook.centroids[:-1])
    assert torch.all(codebook.boundaries[1:] > codebook.boundaries[:-1])
    assert torch.allclose(codebook.centroids, -torch.flip(codebook.centroids, dims=[0]), atol=1e-5)


def test_lloyd_max_codebook_matches_deterministic_unit_sphere_oracle(monkeypatch):
    monkeypatch.setenv("ORBITQUANT_DISABLE_CODEBOOK_DISK_CACHE", "1")
    clear_codebook_cache()

    codebook = get_codebook(dim=32, bits=3)

    expected_centroids = torch.tensor(
        [
            -0.36626825,
            -0.23246057,
            -0.13175610,
            -0.04285152,
            0.04285152,
            0.13175610,
            0.23246057,
            0.36626825,
        ],
        dtype=torch.float32,
    )
    expected_boundaries = torch.tensor(
        [
            -0.29936442,
            -0.18210834,
            -0.08730381,
            0.0,
            0.08730381,
            0.18210834,
            0.29936442,
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(codebook.centroids, expected_centroids, atol=1e-6, rtol=0)
    torch.testing.assert_close(codebook.boundaries, expected_boundaries, atol=1e-6, rtol=0)

    generator = torch.Generator().manual_seed(123)
    samples = torch.randn(4096, 32, generator=generator)
    coordinates = (samples / samples.norm(dim=-1, keepdim=True)).flatten()
    lloyd_mse = torch.mean((codebook.quantize(coordinates) - coordinates) ** 2)
    uniform_centroids = torch.tensor(
        [-0.875, -0.625, -0.375, -0.125, 0.125, 0.375, 0.625, 0.875],
        dtype=torch.float32,
    )
    uniform_boundaries = (uniform_centroids[:-1] + uniform_centroids[1:]) / 2
    uniform_indices = torch.bucketize(coordinates, uniform_boundaries)
    uniform_mse = torch.mean((uniform_centroids[uniform_indices] - coordinates) ** 2)

    assert lloyd_mse < uniform_mse * 0.2


def test_legacy_codebook_version_preserves_existing_artifact_centroids(monkeypatch):
    monkeypatch.setenv("ORBITQUANT_DISABLE_CODEBOOK_DISK_CACHE", "1")
    clear_codebook_cache()

    legacy = get_codebook(dim=32, bits=3, algorithm_version=1)

    expected = torch.tensor(
        [
            -0.36600566,
            -0.23213057,
            -0.13146324,
            -0.04273151,
            0.04273151,
            0.13146324,
            0.23213057,
            0.36600566,
        ],
        dtype=torch.float32,
    )
    assert legacy.algorithm_version == 1
    torch.testing.assert_close(legacy.centroids, expected, atol=1e-6, rtol=0)


def test_v2_codebook_satisfies_lloyd_max_centroid_condition(monkeypatch):
    monkeypatch.setenv("ORBITQUANT_DISABLE_CODEBOOK_DISK_CACHE", "1")
    clear_codebook_cache()
    codebook = get_codebook(dim=128, bits=6, algorithm_version=2)
    positive = codebook.centroids[codebook.centroids.numel() // 2 :].double()
    edges = torch.cat(
        (
            torch.zeros(1, dtype=torch.float64),
            (positive[:-1] + positive[1:]) / 2,
            torch.ones(1, dtype=torch.float64),
        )
    )

    conditional_means = torch.tensor(
        [
            lloyd_max_module._positive_interval_centroid(
                128, float(edges[index]), float(edges[index + 1])
            )
            for index in range(positive.numel())
        ],
        dtype=torch.float64,
    )

    torch.testing.assert_close(positive, conditional_means, atol=1e-7, rtol=0)


def _float32_ulps(left, right):
    return int((left.view(torch.int32) - right.view(torch.int32)).abs().max())


@pytest.mark.parametrize(("dim", "bits"), [(64, 4), (128, 2), (3072, 4), (128, 6)])
def test_v3_newton_codebook_reaches_the_v2_fixed_point(monkeypatch, dim, bits):
    monkeypatch.setenv("ORBITQUANT_DISABLE_CODEBOOK_DISK_CACHE", "1")
    clear_codebook_cache()
    v2 = get_codebook(dim=dim, bits=bits, algorithm_version=2)
    v3 = get_codebook(dim=dim, bits=bits, algorithm_version=3)

    # v2 stops its linear iteration short of the fixed point by up to one float32 ulp.
    assert v3.algorithm_version == 3
    assert _float32_ulps(v3.centroids, v2.centroids) <= 1
    assert _float32_ulps(v3.boundaries, v2.boundaries) <= 1


def test_v3_builds_eight_bit_codebooks_where_v2_does_not_converge(monkeypatch):
    # v2 needs 393216 iterations and still fails for d = 7680 at 8 bits.
    monkeypatch.setenv("ORBITQUANT_DISABLE_CODEBOOK_DISK_CACHE", "1")
    clear_codebook_cache()
    dim = 7680
    start = time.perf_counter()
    codebook = get_codebook(dim=dim, bits=8, algorithm_version=3)
    elapsed = time.perf_counter() - start

    positive = codebook.centroids[codebook.centroids.numel() // 2 :].double()
    edges = [0.0, *((positive[:-1] + positive[1:]) / 2).tolist(), 1.0]
    conditional_means = torch.tensor(
        [
            lloyd_max_module._positive_interval_centroid(dim, edges[index], edges[index + 1])
            for index in range(positive.numel())
        ],
        dtype=torch.float64,
    )
    assert codebook.centroids.numel() == 256
    # Two float32 epsilons: the stored tables are float32 roundings of the fixed point.
    torch.testing.assert_close(positive, conditional_means, atol=0, rtol=2 * 2**-23)
    assert elapsed < 10.0


def test_lloyd_max_quantization_error_decreases_with_bits():
    torch.manual_seed(0)
    values = torch.randn(2048)
    values = values / values.norm() * (2048**0.5)
    values = values.clamp(-0.75, 0.75)

    q2 = get_codebook(dim=64, bits=2).quantize(values)
    q3 = get_codebook(dim=64, bits=3).quantize(values)
    q4 = get_codebook(dim=64, bits=4).quantize(values)

    mse2 = torch.mean((q2 - values) ** 2)
    mse3 = torch.mean((q3 - values) ** 2)
    mse4 = torch.mean((q4 - values) ** 2)

    assert mse4 < mse3 < mse2


def test_lloyd_max_codebook_reuses_valid_persistent_disk_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("ORBITQUANT_CODEBOOK_CACHE_DIR", str(tmp_path))
    clear_codebook_cache()
    cache_path = codebook_cache_path(dim=40, bits=2)
    assert cache_path is not None

    generated = get_codebook(dim=40, bits=2)
    clear_codebook_cache()

    def fail_generation(*args, **kwargs):
        raise AssertionError("valid disk cache should be used without regeneration")

    monkeypatch.setattr(lloyd_max_module, "_generate_codebook", fail_generation)

    restored = get_codebook(dim=40, bits=2)

    assert torch.equal(restored.centroids, generated.centroids)
    assert torch.equal(restored.boundaries, generated.boundaries)


def test_lloyd_max_codebook_rejects_untrusted_legacy_disk_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("ORBITQUANT_CODEBOOK_CACHE_DIR", str(tmp_path))
    clear_codebook_cache()
    cache_path = codebook_cache_path(dim=40, bits=2)
    assert cache_path is not None
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    centroids = torch.tensor([-0.3, -0.1, 0.1, 0.3], dtype=torch.float32)
    boundaries = torch.tensor([-0.2, 0.0, 0.2], dtype=torch.float32)
    save_file(
        {
            "centroids": centroids,
            "boundaries": boundaries,
            "algorithm_version": torch.tensor([1], dtype=torch.int32),
        },
        cache_path,
    )

    codebook = get_codebook(dim=40, bits=2)
    tensors = load_file(cache_path)

    assert not torch.equal(codebook.centroids, centroids)
    assert not torch.equal(codebook.boundaries, boundaries)
    assert int(tensors["dim"].item()) == 40
    assert int(tensors["bits"].item()) == 2
    assert "cache_checksum" in tensors


def test_lloyd_max_codebook_writes_persistent_disk_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("ORBITQUANT_CODEBOOK_CACHE_DIR", str(tmp_path))
    clear_codebook_cache()
    cache_path = codebook_cache_path(dim=48, bits=3)
    assert cache_path is not None

    codebook = get_codebook(dim=48, bits=3)

    assert cache_path.exists()
    tensors = load_file(cache_path)
    assert int(tensors["dim"].item()) == 48
    assert int(tensors["bits"].item()) == 3
    assert "cache_checksum" in tensors
    clear_codebook_cache()
    restored = get_codebook(dim=48, bits=3)
    assert torch.equal(restored.centroids, codebook.centroids)
    assert torch.equal(restored.boundaries, codebook.boundaries)
