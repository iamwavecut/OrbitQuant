import pytest
import torch

from orbitquant.codebooks import get_codebook
from orbitquant.functional import quantize_activations
from orbitquant.kernels import available_backends, quantize_activations_kernel, select_backend
from orbitquant.rotations import RPBHRotation


def test_cpu_activation_kernel_matches_reference_functional_path():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 16)
    rotation = RPBHRotation(dim=16, seed=3, block_size=8)
    codebook = get_codebook(dim=16, bits=4)

    expected = quantize_activations(x, rotation=rotation, codebook=codebook, eps=1e-12)
    actual = quantize_activations_kernel(
        x, rotation=rotation, codebook=codebook, eps=1e-12, backend="cpu"
    )

    assert torch.allclose(actual, expected)


def test_backend_selection_is_explicit_and_fails_loud_for_unavailable_backends():
    assert available_backends()["cpu"] is True
    assert select_backend(torch.device("cpu"), requested="auto") == "cpu"

    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA"):
            select_backend(torch.device("cpu"), requested="triton_cuda")

    if not torch.backends.mps.is_available():
        with pytest.raises(RuntimeError, match="MPS"):
            select_backend(torch.device("cpu"), requested="mps")
