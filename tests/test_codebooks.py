import torch
from safetensors.torch import load_file, save_file

import orbitquant.codebooks.lloyd_max as lloyd_max_module
from orbitquant.codebooks import clear_codebook_cache, codebook_cache_path, get_codebook


def test_lloyd_max_codebook_is_symmetric_and_sorted():
    codebook = get_codebook(dim=32, bits=4)

    assert codebook.dim == 32
    assert codebook.bits == 4
    assert codebook.centroids.shape == (16,)
    assert codebook.boundaries.shape == (15,)
    assert torch.all(codebook.centroids[1:] > codebook.centroids[:-1])
    assert torch.all(codebook.boundaries[1:] > codebook.boundaries[:-1])
    assert torch.allclose(codebook.centroids, -torch.flip(codebook.centroids, dims=[0]), atol=1e-5)


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
