import torch

from orbitquant.codebooks import get_codebook


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
