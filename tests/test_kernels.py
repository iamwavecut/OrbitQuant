import torch

import orbitquant.kernels.dispatch as dispatch_module
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
        try:
            select_backend(torch.device("cpu"), requested="triton_cuda")
        except RuntimeError as exc:
            assert "CUDA" in str(exc)
        else:
            raise AssertionError("unavailable CUDA/Triton backend was accepted")

    if not torch.backends.mps.is_available():
        try:
            select_backend(torch.device("cpu"), requested="mps")
        except RuntimeError as exc:
            assert "MPS" in str(exc)
        else:
            raise AssertionError("unavailable MPS backend was accepted")


def test_backend_selection_accepts_injected_availability_for_gpu_paths():
    backends = {"cpu": True, "mps": False, "triton_cuda": True}

    assert (
        select_backend(torch.device("cuda"), requested="auto", backends=backends)
        == "triton_cuda"
    )
    assert select_backend(torch.device("mps"), requested="auto", backends=backends) == "cpu"


def test_triton_cuda_dispatch_uses_backend_function_with_reference_equivalent_output(monkeypatch):
    torch.manual_seed(0)
    x = torch.randn(2, 3, 16)
    rotation = RPBHRotation(dim=16, seed=3, block_size=8)
    codebook = get_codebook(dim=16, bits=4)
    calls = []

    def fake_triton_backend(input_tensor, *, rotation, codebook, eps):
        calls.append(input_tensor.shape)
        return quantize_activations(input_tensor, rotation=rotation, codebook=codebook, eps=eps)

    monkeypatch.setattr(dispatch_module, "_triton_cuda_quantize_activations", fake_triton_backend)
    monkeypatch.setattr(
        dispatch_module,
        "available_backends",
        lambda: {"cpu": True, "mps": False, "triton_cuda": True},
    )

    expected = quantize_activations(x, rotation=rotation, codebook=codebook, eps=1e-12)
    actual = quantize_activations_kernel(
        x, rotation=rotation, codebook=codebook, eps=1e-12, backend="triton_cuda"
    )

    assert torch.allclose(actual, expected)
    assert calls == [x.shape]
