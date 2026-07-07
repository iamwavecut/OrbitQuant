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
    expected_boundaries = torch.tensor(
        [
            -0.29906812,
            -0.18179691,
            -0.08709738,
            0.0,
            0.08709738,
            0.18179691,
            0.29906812,
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
