import sys
from types import SimpleNamespace

import pytest
import torch

import orbitquant.kernels.dispatch as dispatch_module
import orbitquant.layers as layers_module
from orbitquant.config import OrbitQuantConfig
from orbitquant.layers import OrbitQuantLinear


def test_orbit_linear_debug_rotation_matches_source_linear():
    torch.manual_seed(0)
    source = torch.nn.Linear(16, 7)
    x = torch.randn(2, 5, 16)
    config = OrbitQuantConfig(
        weight_bits=4,
        activation_bits=4,
        runtime_mode="debug_no_quant",
        rotation_seed=11,
        block_size=8,
    )

    quantized = OrbitQuantLinear.from_linear(source, config=config, module_name="block.attn.to_q")

    expected = source(x)
    actual = quantized(x)

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_orbit_linear_quantized_forward_is_finite_and_shape_preserving():
    torch.manual_seed(1)
    source = torch.nn.Linear(16, 7)
    x = torch.randn(2, 5, 16)
    config = OrbitQuantConfig(weight_bits=4, activation_bits=4, rotation_seed=11, block_size=8)

    quantized = OrbitQuantLinear.from_linear(source, config=config, module_name="block.ff.linear")
    actual = quantized(x)

    assert actual.shape == (2, 5, 7)
    assert torch.isfinite(actual).all()
    assert not any(parameter.requires_grad for parameter in quantized.parameters())
    assert "_rotation_permutation" not in quantized.state_dict()
    assert "_activation_codebook_centroids" not in quantized.state_dict()


def test_orbit_linear_passes_configured_activation_kernel_backend(monkeypatch):
    torch.manual_seed(2)
    source = torch.nn.Linear(16, 7)
    x = torch.randn(2, 5, 16)
    config = OrbitQuantConfig(
        weight_bits=4,
        activation_bits=4,
        rotation_seed=11,
        block_size=8,
        activation_kernel_backend="cpu",
    )
    quantized = OrbitQuantLinear.from_linear(source, config=config, module_name="block.ff.linear")
    original_kernel = layers_module.quantize_activations_kernel
    seen_backends = []

    def wrapped_kernel(*args, **kwargs):
        seen_backends.append(kwargs["backend"])
        return original_kernel(*args, **kwargs)

    monkeypatch.setattr(layers_module, "quantize_activations_kernel", wrapped_kernel)

    quantized(x)

    assert seen_backends == ["cpu"]


def test_orbit_linear_reuses_cuda_activation_constant_buffers(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    torch.manual_seed(2)
    source = torch.nn.Linear(16, 7, device="cuda", dtype=torch.float32)
    x = torch.randn(2, 5, 16, device="cuda")
    config = OrbitQuantConfig(
        weight_bits=4,
        activation_bits=4,
        rotation_seed=11,
        block_size=8,
        activation_kernel_backend="triton_cuda",
    )
    quantized = OrbitQuantLinear.from_linear(
        source, config=config, module_name="block.ff.linear"
    )
    seen_constant_ids = []

    def fake_kernel(input_tensor, *, constant_tensors, **kwargs):
        assert input_tensor.is_cuda
        assert set(constant_tensors) == {"permutation", "signs", "centroids", "boundaries"}
        assert all(tensor.is_cuda for tensor in constant_tensors.values())
        seen_constant_ids.append(
            {name: id(tensor) for name, tensor in constant_tensors.items()}
        )
        return torch.zeros_like(input_tensor)

    monkeypatch.setattr(layers_module, "quantize_activations_kernel", fake_kernel)

    quantized(x)
    quantized(x)

    assert seen_constant_ids[0] == seen_constant_ids[1]


def test_orbit_linear_caches_dequantized_weight(monkeypatch):
    torch.manual_seed(2)
    source = torch.nn.Linear(16, 7)
    x = torch.randn(2, 5, 16)
    config = OrbitQuantConfig(weight_bits=4, activation_bits=4, rotation_seed=11, block_size=8)
    quantized = OrbitQuantLinear.from_linear(source, config=config, module_name="block.ff.linear")
    calls = 0
    original_unpack = layers_module.unpack_lowbit

    def counted_unpack(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_unpack(*args, **kwargs)

    monkeypatch.setattr(layers_module, "unpack_lowbit", counted_unpack)

    quantized(x)
    quantized(x)
    assert calls == 1

    quantized.clear_dequantized_cache()
    quantized(x)
    assert calls == 2


def test_orbit_linear_mps_weight_dequant_uses_kernel_without_cpu_unpack(monkeypatch):
    if not dispatch_module._mps_metal_available():
        pytest.skip("MPS Metal shader backend is not available")

    torch.manual_seed(3)
    source = torch.nn.Linear(16, 7)
    config = OrbitQuantConfig(weight_bits=4, activation_bits=4, rotation_seed=11, block_size=8)
    quantized = OrbitQuantLinear.from_linear(source, config=config, module_name="block.ff.linear")
    expected = quantized._dequantize_weight(device=torch.device("cpu"), dtype=torch.float32)
    quantized.clear_dequantized_cache()

    def fail_unpack(*args, **kwargs):
        raise AssertionError("MPS weight dequant should not call CPU unpack_lowbit")

    monkeypatch.setattr(layers_module, "unpack_lowbit", fail_unpack)

    actual = quantized._dequantize_weight(device=torch.device("mps"), dtype=torch.float32)

    assert torch.allclose(actual.cpu(), expected)


def test_orbit_linear_cuda_weight_dequant_dispatches_to_triton_without_cpu_unpack(
    monkeypatch,
):
    torch.manual_seed(4)
    source = torch.nn.Linear(16, 7)
    config = OrbitQuantConfig(weight_bits=4, activation_bits=4, rotation_seed=11, block_size=8)
    quantized = OrbitQuantLinear.from_linear(source, config=config, module_name="block.ff.linear")
    expected = quantized._dequantize_weight(device=torch.device("cpu"), dtype=torch.float32)
    quantized.clear_dequantized_cache()
    calls = []

    def fake_triton_dequant(*args, **kwargs):
        calls.append(kwargs)
        return expected.clone()

    monkeypatch.setitem(
        sys.modules,
        "orbitquant.kernels.triton_cuda",
        SimpleNamespace(dequantize_packed_weight_with_triton=fake_triton_dequant),
    )

    def fail_unpack(*args, **kwargs):
        raise AssertionError("CUDA weight dequant should not call CPU unpack_lowbit")

    monkeypatch.setattr(layers_module, "unpack_lowbit", fail_unpack)

    actual = quantized._dequantize_weight(device=torch.device("cuda"), dtype=torch.float32)

    assert torch.allclose(actual, expected)
    assert calls == [
        {
            "bits": 4,
            "out_features": 7,
            "in_features": 16,
            "device": torch.device("cuda"),
        }
    ]
