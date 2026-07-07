import copy
import sys
from types import SimpleNamespace

import pytest
import torch

import orbitquant.kernels.dispatch as dispatch_module
import orbitquant.layers as layers_module
from orbitquant.codebooks import clear_codebook_cache
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


def test_orbit_linear_state_dict_contains_no_activation_calibration_state():
    torch.manual_seed(1)
    source = torch.nn.Linear(16, 7)
    config = OrbitQuantConfig(weight_bits=4, activation_bits=4, rotation_seed=11, block_size=8)

    quantized = OrbitQuantLinear.from_linear(source, config=config, module_name="block.ff.linear")

    assert set(quantized.state_dict()) == {"bias", "packed_weight_indices", "row_norms"}


def test_orbit_linear_shares_codebooks_by_dimension_and_bits_not_module_name():
    torch.manual_seed(1)
    source_a = torch.nn.Linear(16, 7)
    source_b = torch.nn.Linear(16, 9)
    config = OrbitQuantConfig(weight_bits=4, activation_bits=3, rotation_seed=11, block_size=8)

    quantized_a = OrbitQuantLinear.from_linear(
        source_a, config=config, module_name="blocks.0.attn.to_q"
    )
    quantized_b = OrbitQuantLinear.from_linear(
        source_b, config=config, module_name="blocks.37.ff.linear_out"
    )

    assert quantized_a.weight_codebook is quantized_b.weight_codebook
    assert quantized_a.activation_codebook is quantized_b.activation_codebook


def test_orbit_linear_weight_indices_quantize_rotated_unit_directions():
    torch.manual_seed(2)
    source = torch.nn.Linear(16, 7)
    config = OrbitQuantConfig(weight_bits=4, activation_bits=4, rotation_seed=11, block_size=8)

    quantized = OrbitQuantLinear.from_linear(source, config=config, module_name="block.ff.linear")

    assert quantized.row_norms is not None
    assert quantized.row_norms.dtype == torch.bfloat16

    weight = source.weight.detach().to(torch.float32)
    row_norms = weight.norm(dim=-1).clamp_min(config.activation_eps)
    rotated_after_normalize = quantized.rotation.apply_to_weight(weight / row_norms[:, None])
    expected_indices = quantized.weight_codebook.quantize_indices(rotated_after_normalize)
    actual_indices = layers_module.unpack_lowbit(
        quantized.packed_weight_indices,
        bits=quantized.weight_bits,
        length=quantized.out_features * quantized.in_features,
    ).reshape(quantized.out_features, quantized.in_features)

    assert torch.equal(actual_indices, expected_indices)


def test_orbit_linear_stores_raw_zero_row_norm_and_dequantizes_zero_row():
    source = torch.nn.Linear(16, 7, bias=False)
    with torch.no_grad():
        source.weight[0].zero_()
    config = OrbitQuantConfig(weight_bits=4, activation_bits=4, rotation_seed=11, block_size=8)

    quantized = OrbitQuantLinear.from_linear(source, config=config, module_name="block.ff.linear")
    dequantized = quantized._dequantize_weight(device=torch.device("cpu"), dtype=torch.float32)

    assert quantized.row_norms is not None
    assert quantized.row_norms[0].item() == 0
    assert torch.equal(dequantized[0], torch.zeros_like(dequantized[0]))


def test_orbit_linear_quantized_forward_matches_manual_paper_equation(monkeypatch):
    monkeypatch.setenv("ORBITQUANT_DISABLE_CODEBOOK_DISK_CACHE", "1")
    clear_codebook_cache()
    torch.manual_seed(7)
    source = torch.nn.Linear(24, 5)
    with torch.no_grad():
        source.weight[1].zero_()
    x = torch.randn(2, 3, 24)
    x[0, 0].zero_()
    config = OrbitQuantConfig(
        weight_bits=3,
        activation_bits=3,
        rotation_seed=13,
        block_size="paper",
        activation_kernel_backend="cpu",
    )
    quantized = OrbitQuantLinear.from_linear(source, config=config, module_name="block.ff.linear")

    weight = source.weight.detach().to(torch.float32)
    rotated_weight = quantized.rotation.apply_to_weight(weight)
    raw_row_norms = rotated_weight.norm(dim=-1)
    weight_unit = rotated_weight / raw_row_norms.clamp_min(config.activation_eps)[:, None]
    weight_indices = quantized.weight_codebook.quantize_indices(weight_unit)
    dequantized_weight = (
        quantized.row_norms.to(torch.float32)[:, None]
        * quantized.weight_codebook.centroids[weight_indices.to(torch.long)]
    )

    work = x.to(torch.float32)
    token_norms = work.norm(dim=-1, keepdim=True)
    activation_unit = work / (token_norms + config.activation_eps)
    rotated_activation = quantized.rotation.apply_to_activations(activation_unit)
    dequantized_activation = token_norms * quantized.activation_codebook.quantize(
        rotated_activation
    )
    expected = torch.nn.functional.linear(
        dequantized_activation,
        dequantized_weight,
        source.bias.detach().to(torch.float32),
    )

    actual = quantized(x)

    assert quantized.rotation.block_size == 8
    assert torch.equal(actual[0, 0], source.bias.detach())
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


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


def test_orbit_linear_mps_forward_matches_reference_without_cpu_unpack(monkeypatch):
    if not dispatch_module._mps_metal_available():
        pytest.skip("MPS Metal shader backend is not available")

    torch.manual_seed(6)
    source = torch.nn.Linear(16, 7)
    x = torch.randn(2, 5, 16)
    config = OrbitQuantConfig(
        weight_bits=4,
        activation_bits=4,
        rotation_seed=11,
        block_size=8,
        activation_kernel_backend="cpu",
    )
    reference = OrbitQuantLinear.from_linear(
        source, config=config, module_name="block.ff.linear"
    )
    actual_layer = copy.deepcopy(reference).to("mps")
    actual_layer.activation_kernel_backend = "mps"

    expected = reference(x)

    def fail_unpack(*args, **kwargs):
        raise AssertionError("MPS forward should not call CPU unpack_lowbit")

    monkeypatch.setattr(layers_module, "unpack_lowbit", fail_unpack)

    actual = actual_layer(x.to("mps"))

    assert actual.device.type == "mps"
    assert actual.shape == expected.shape
    assert torch.allclose(actual.cpu(), expected, atol=1e-4, rtol=1e-4)


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


def test_orbit_linear_triton_packed_matmul_runtime_rejects_non_cuda_before_quantization(
    monkeypatch,
):
    torch.manual_seed(5)
    source = torch.nn.Linear(16, 7)
    config = OrbitQuantConfig(
        weight_bits=4,
        activation_bits=4,
        rotation_seed=11,
        block_size=8,
        runtime_mode="triton_packed_matmul",
        activation_kernel_backend="cpu",
    )
    quantized = OrbitQuantLinear.from_linear(source, config=config, module_name="block.ff.linear")
    x = torch.randn(2, 5, 16)

    def fail_activation_quantization(*args, **kwargs):
        raise AssertionError("device validation should run before activation quantization")

    monkeypatch.setattr(
        layers_module,
        "quantize_activations_kernel",
        fail_activation_quantization,
    )

    with pytest.raises(RuntimeError, match="requires CUDA input tensors"):
        quantized(x)


def test_orbit_linear_triton_packed_matmul_runtime_avoids_weight_dequant_cache(monkeypatch):
    torch.manual_seed(5)
    source = torch.nn.Linear(16, 7)
    config = OrbitQuantConfig(
        weight_bits=4,
        activation_bits=4,
        rotation_seed=11,
        block_size=8,
        runtime_mode="triton_packed_matmul",
        activation_kernel_backend="triton_cuda",
        packed_matmul_block_m=32,
        packed_matmul_block_n=64,
        packed_matmul_block_k=64,
        packed_matmul_num_warps=8,
    )
    quantized = OrbitQuantLinear.from_linear(source, config=config, module_name="block.ff.linear")
    x = torch.randn(2, 5, 16)
    calls = []

    def fake_activation_kernel(input_tensor, **kwargs):
        return input_tensor

    def fake_matmul(input_tensor, packed_weight_indices, row_norms, codebook, **kwargs):
        calls.append(
            {
                "shape": tuple(input_tensor.shape),
                "bits": kwargs["bits"],
                "out_features": kwargs["out_features"],
                "in_features": kwargs["in_features"],
                "bias_is_none": kwargs["bias"] is None,
                "block_m": kwargs["block_m"],
                "block_n": kwargs["block_n"],
                "block_k": kwargs["block_k"],
                "num_warps": kwargs["num_warps"],
            }
        )
        return torch.zeros(
            *input_tensor.shape[:-1],
            kwargs["out_features"],
            dtype=input_tensor.dtype,
        )

    def fail_dequant(*args, **kwargs):
        raise AssertionError(
            "triton_packed_matmul runtime should not materialize dequantized weight"
        )

    monkeypatch.setattr(layers_module, "quantize_activations_kernel", fake_activation_kernel)
    monkeypatch.setattr(quantized, "_validate_triton_packed_matmul_input", lambda x: None)
    monkeypatch.setitem(
        sys.modules,
        "orbitquant.kernels.triton_cuda",
        SimpleNamespace(matmul_packed_weight_with_triton=fake_matmul),
    )
    monkeypatch.setattr(quantized, "_dequantize_weight", fail_dequant)

    actual = quantized(x)

    assert actual.shape == (2, 5, 7)
    assert calls == [
        {
            "shape": (2, 5, 16),
            "bits": 4,
            "out_features": 7,
            "in_features": 16,
            "bias_is_none": False,
            "block_m": 32,
            "block_n": 64,
            "block_k": 64,
            "num_warps": 8,
        }
    ]


@pytest.mark.parametrize("use_bias", [True, False])
def test_orbit_linear_triton_packed_matmul_runtime_matches_dequant_bf16(use_bias):
    if not torch.cuda.is_available() or not dispatch_module.available_backends()["triton_cuda"]:
        pytest.skip("CUDA/Triton backend is not available")

    torch.manual_seed(7 + int(use_bias))
    source = torch.nn.Linear(32, 9, bias=use_bias, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(2, 5, 32, device="cuda", dtype=torch.bfloat16)
    config = OrbitQuantConfig(
        weight_bits=4,
        activation_bits=4,
        rotation_seed=11,
        block_size=8,
        activation_kernel_backend="triton_cuda",
        runtime_mode="dequant_bf16",
        packed_matmul_block_m=16,
        packed_matmul_block_n=16,
        packed_matmul_block_k=32,
        packed_matmul_num_warps=4,
    )
    reference = OrbitQuantLinear.from_linear(
        source, config=config, module_name="block.ff.linear"
    )
    packed = copy.deepcopy(reference)
    packed.runtime_mode = "triton_packed_matmul"

    expected = reference(x)
    actual = packed(x)

    assert actual.is_cuda
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert torch.allclose(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)
