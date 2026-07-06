import pytest
import torch

import orbitquant.kernels.dispatch as dispatch_module
from orbitquant.codebooks import get_codebook
from orbitquant.functional import quantize_activations
from orbitquant.kernels import (
    available_backends,
    backend_capabilities,
    quantize_activations_kernel,
    select_backend,
)
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


def test_backend_capabilities_report_partial_and_fallback_kernel_status(monkeypatch):
    monkeypatch.setattr(dispatch_module, "_mps_metal_available", lambda: False)
    capabilities = backend_capabilities(
        backends={"cpu": True, "mps": True, "triton_cuda": True}
    )

    assert capabilities["cpu"]["available"] is True
    assert capabilities["cpu"]["optimized"] is False
    assert capabilities["cpu"]["implementation"] == "torch_reference"
    assert capabilities["mps"]["available"] is True
    assert capabilities["mps"]["optimized"] is False
    assert capabilities["mps"]["implementation"] == "torch_reference_mps"
    assert capabilities["triton_cuda"]["available"] is True
    assert capabilities["triton_cuda"]["optimized"] is True
    assert capabilities["triton_cuda"]["optimized_stage"] == "codebook_lookup_rescale"
    assert capabilities["triton_cuda"]["full_fusion"] is False


def test_backend_capabilities_report_mps_metal_partial_kernel(monkeypatch):
    monkeypatch.setattr(dispatch_module, "_mps_metal_available", lambda: True)

    capabilities = backend_capabilities(
        backends={"cpu": True, "mps": True, "triton_cuda": False}
    )

    assert capabilities["mps"]["available"] is True
    assert capabilities["mps"]["optimized"] is True
    assert capabilities["mps"]["implementation"] == "metal_codebook_rescale"
    assert capabilities["mps"]["optimized_stage"] == "codebook_lookup_rescale"
    assert capabilities["mps"]["full_fusion"] is False


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


def test_mps_dispatch_uses_backend_function_with_reference_equivalent_output(monkeypatch):
    torch.manual_seed(0)
    x = torch.randn(2, 3, 16)
    rotation = RPBHRotation(dim=16, seed=3, block_size=8)
    codebook = get_codebook(dim=16, bits=4)
    calls = []

    def fake_mps_backend(input_tensor, *, rotation, codebook, eps):
        calls.append(input_tensor.shape)
        return quantize_activations(input_tensor, rotation=rotation, codebook=codebook, eps=eps)

    monkeypatch.setattr(dispatch_module, "_mps_quantize_activations", fake_mps_backend)
    monkeypatch.setattr(
        dispatch_module,
        "available_backends",
        lambda: {"cpu": True, "mps": True, "triton_cuda": False},
    )

    expected = quantize_activations(x, rotation=rotation, codebook=codebook, eps=1e-12)
    actual = quantize_activations_kernel(
        x, rotation=rotation, codebook=codebook, eps=1e-12, backend="mps"
    )

    assert torch.allclose(actual, expected)
    assert calls == [x.shape]


def test_mps_backend_matches_reference_without_full_reference_fallback(monkeypatch):
    if not dispatch_module._mps_metal_available():
        pytest.skip("MPS Metal shader backend is not available")

    torch.manual_seed(0)
    x = torch.randn(4, 5, 16, device="mps", dtype=torch.float32)
    rotation = RPBHRotation(dim=16, seed=3, block_size=8)
    codebook = get_codebook(dim=16, bits=4)
    expected = quantize_activations(x, rotation=rotation, codebook=codebook, eps=1e-12)

    def fail_reference(*args, **kwargs):
        raise AssertionError("mps backend should not call the full reference path")

    monkeypatch.setattr(dispatch_module, "_reference_quantize_activations", fail_reference)

    actual = quantize_activations_kernel(
        x, rotation=rotation, codebook=codebook, eps=1e-12, backend="mps"
    )

    assert torch.allclose(actual.cpu(), expected.cpu())


def test_mps_codebook_kernel_matches_bucketize_boundary_semantics():
    if not dispatch_module._mps_metal_available():
        pytest.skip("MPS Metal shader backend is not available")

    from orbitquant.kernels.mps import quantize_rotated_activations_with_mps

    codebook = get_codebook(dim=16, bits=4)
    rotated = codebook.boundaries.reshape(1, -1).to(device="mps", dtype=torch.float32)
    norms = torch.ones(1, 1, device="mps", dtype=torch.float32)

    expected = codebook.quantize(rotated)
    actual = quantize_rotated_activations_with_mps(rotated, norms, codebook)

    assert torch.equal(actual.cpu(), expected.cpu())


def test_triton_cuda_backend_matches_reference_without_full_reference_fallback(monkeypatch):
    if not torch.cuda.is_available() or not available_backends()["triton_cuda"]:
        pytest.skip("CUDA/Triton backend is not available")

    torch.manual_seed(0)
    x = torch.randn(4, 5, 16, device="cuda", dtype=torch.float32)
    rotation = RPBHRotation(dim=16, seed=3, block_size=8)
    codebook = get_codebook(dim=16, bits=4)
    expected = quantize_activations(x, rotation=rotation, codebook=codebook, eps=1e-12)

    def fail_reference(*args, **kwargs):
        raise AssertionError("triton_cuda backend should not call the full reference path")

    monkeypatch.setattr(dispatch_module, "_reference_quantize_activations", fail_reference)

    actual = quantize_activations_kernel(
        x, rotation=rotation, codebook=codebook, eps=1e-12, backend="triton_cuda"
    )

    assert torch.allclose(actual, expected)
