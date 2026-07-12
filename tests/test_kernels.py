import pytest
import torch

import orbitquant.kernels.dispatch as dispatch_module
from orbitquant.codebooks import get_codebook
from orbitquant.config import OrbitQuantConfig
from orbitquant.functional import quantize_activations
from orbitquant.kernels import (
    available_backends,
    backend_capabilities,
    quantize_activations_kernel,
    select_backend,
)
from orbitquant.kernels.executorch_vulkan import ExecuTorchVulkanW4A4Linear
from orbitquant.layers import OrbitQuantLinear
from orbitquant.modeling import quantize_linear_modules
from orbitquant.packing import pack_lowbit, unpack_lowbit
from orbitquant.rotations import RPBHRotation


def _make_executorch_vulkan_export_layer() -> ExecuTorchVulkanW4A4Linear:
    source = torch.nn.Linear(24, 7, bias=True)
    layer = OrbitQuantLinear.from_linear(
        source,
        config=OrbitQuantConfig(
            weight_bits=4,
            activation_bits=4,
            block_size="paper",
            runtime_mode="dequant_bf16",
        ),
        module_name="probe",
    )
    return ExecuTorchVulkanW4A4Linear(layer)


def test_executorch_vulkan_export_keeps_exact_w4a4_as_one_custom_op():
    exported = torch.export.export(
        _make_executorch_vulkan_export_layer(),
        (torch.randn(2, 3, 24, dtype=torch.float32),),
        strict=True,
    )

    call_targets = {node.target for node in exported.graph.nodes if node.op == "call_function"}
    assert torch.ops.orbitquant_vulkan.linear_w4a4_exact.default in call_targets


def test_executorch_vulkan_export_rejects_unsupported_bfloat16_activations():
    layer = _make_executorch_vulkan_export_layer()

    with pytest.raises(TypeError, match="float16 or float32"):
        layer(torch.randn(2, 24, dtype=torch.bfloat16))


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


def test_activation_quantization_is_per_token_scale_equivariant_and_batch_independent():
    torch.manual_seed(11)
    x = torch.randn(2, 4, 16).clamp_min(-2).clamp_max(2)
    scales = torch.tensor(
        [[[0.5], [1.0], [2.0], [4.0]], [[1.5], [3.0], [0.75], [2.5]]],
        dtype=x.dtype,
    )
    rotation = RPBHRotation(dim=16, seed=3, block_size=8)
    codebook = get_codebook(dim=16, bits=4)

    baseline = quantize_activations(x, rotation=rotation, codebook=codebook, eps=1e-12)
    scaled = quantize_activations(x * scales, rotation=rotation, codebook=codebook, eps=1e-12)

    assert torch.allclose(scaled, baseline * scales, atol=1e-6, rtol=1e-6)

    outlier = torch.full((1, 1, 16), 10_000.0)
    with_outlier = quantize_activations(
        torch.cat((x.reshape(-1, 16), outlier.reshape(-1, 16)), dim=0),
        rotation=rotation,
        codebook=codebook,
        eps=1e-12,
    )

    assert torch.equal(with_outlier[:-1], baseline.reshape(-1, 16))


def test_activation_normalization_adds_paper_epsilon_to_token_norm():
    class IdentityCodebook:
        def quantize(self, values):
            return values

    torch.manual_seed(12)
    x = torch.randn(2, 3, 16)
    rotation = RPBHRotation(dim=16, seed=3, block_size=8)

    actual = quantize_activations(
        x,
        rotation=rotation,
        codebook=IdentityCodebook(),
        eps=0.5,
    )

    norms = x.float().norm(dim=-1, keepdim=True)
    expected = rotation.apply_to_activations(x.float() / (norms + 0.5)) * norms
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_activation_quantization_rescales_with_raw_norm_so_zero_tokens_stay_zero():
    x = torch.zeros(2, 3, 16)
    rotation = RPBHRotation(dim=16, seed=3, block_size=8)
    codebook = get_codebook(dim=16, bits=4)

    actual = quantize_activations(x, rotation=rotation, codebook=codebook, eps=1e-10)

    assert torch.equal(actual, torch.zeros_like(x))


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
    monkeypatch.setattr(dispatch_module, "_native_packed_matmul_available", lambda: False)
    monkeypatch.setattr(dispatch_module, "_native_cpu_packed_matmul_available", lambda: False)
    monkeypatch.setattr(dispatch_module, "_native_cpu_activation_available", lambda: False)
    monkeypatch.setattr(dispatch_module, "_native_cpu_adaln_available", lambda: False)
    capabilities = backend_capabilities(backends={"cpu": True, "mps": True, "triton_cuda": True})

    assert capabilities["cpu"]["available"] is True
    assert capabilities["cpu"]["claim_status"] == "reference_only"
    assert capabilities["cpu"]["optimized"] is False
    assert capabilities["cpu"]["implemented_stage"] == (
        "activation_norm_rpbh_quant_rescale,packed_weight_matmul,adaln_rtn_packed_matmul"
    )
    assert capabilities["cpu"]["optimized_stage"] is None
    assert capabilities["cpu"]["weight_dequant_optimized"] is False
    assert capabilities["cpu"]["weight_pack_optimized"] is False
    assert capabilities["cpu"]["lowbit_unpack_optimized"] is False
    assert capabilities["cpu"]["weight_quant_optimized"] is False
    assert capabilities["cpu"]["adaln_quant_optimized"] is False
    assert capabilities["cpu"]["adaln_dequant_optimized"] is False
    assert capabilities["cpu"]["implementation"] == "torch_reference"
    assert capabilities["cpu"]["hf_kernel_builder_compliant"] is False
    assert capabilities["mps"]["available"] is True
    assert capabilities["mps"]["claim_status"] == "reference_only"
    assert capabilities["mps"]["optimized"] is False
    assert (
        capabilities["mps"]["implemented_stage"]
        == "activation_norm_rpbh_quant_rescale,packed_weight_dequant,packed_weight_matmul"
    )
    assert capabilities["mps"]["optimized_stage"] is None
    assert capabilities["mps"]["weight_dequant_optimized"] is False
    assert capabilities["mps"]["weight_pack_optimized"] is False
    assert capabilities["mps"]["lowbit_unpack_optimized"] is False
    assert capabilities["mps"]["weight_quant_optimized"] is False
    assert capabilities["mps"]["adaln_quant_optimized"] is False
    assert capabilities["mps"]["adaln_dequant_optimized"] is False
    assert capabilities["mps"]["implementation"] == "torch_reference_mps"
    assert capabilities["mps"]["upstream_native_mps_op"] is False
    assert capabilities["mps"]["hf_kernel_builder_compliant"] is False
    assert capabilities["triton_cuda"]["available"] is True
    assert capabilities["triton_cuda"]["claim_status"] == "partial_optimized"
    assert capabilities["triton_cuda"]["optimized"] is True
    expected_triton_stage = (
        "activation_norm_rpbh_quant_rescale,packed_weight_dequant,"
        "packed_weight_matmul,lowbit_pack,lowbit_unpack,weight_rotation_fwht_quant_pack,"
        "adaln_rtn_quant_pack,adaln_rtn_dequant,adaln_rtn_packed_matmul"
    )
    assert capabilities["triton_cuda"]["implemented_stage"] == expected_triton_stage
    assert capabilities["triton_cuda"]["optimized_stage"] == expected_triton_stage
    assert capabilities["triton_cuda"]["weight_dequant_optimized"] is True
    assert capabilities["triton_cuda"]["weight_pack_optimized"] is True
    assert capabilities["triton_cuda"]["lowbit_unpack_optimized"] is True
    assert capabilities["triton_cuda"]["weight_quant_optimized"] is True
    assert capabilities["triton_cuda"]["adaln_quant_optimized"] is True
    assert capabilities["triton_cuda"]["adaln_dequant_optimized"] is True
    assert capabilities["triton_cuda"]["adaln_packed_matmul_optimized"] is True
    assert capabilities["triton_cuda"]["full_fusion"] is False
    assert capabilities["triton_cuda"]["implementation"] == "python_triton_orbitquant_pipeline"
    assert capabilities["triton_cuda"]["package_format"] == "python_triton"
    assert capabilities["triton_cuda"]["hf_kernel_builder_compliant"] is False


def test_backend_capabilities_report_mps_metal_partial_kernel(monkeypatch):
    monkeypatch.setattr(dispatch_module, "_mps_metal_available", lambda: True)
    monkeypatch.setattr(dispatch_module, "_native_packed_matmul_available", lambda: True)
    monkeypatch.setattr(dispatch_module, "_native_cpu_packed_matmul_available", lambda: False)
    monkeypatch.setattr(dispatch_module, "_native_cpu_activation_available", lambda: False)
    monkeypatch.setattr(dispatch_module, "_native_cpu_adaln_available", lambda: False)

    capabilities = backend_capabilities(backends={"cpu": True, "mps": True, "triton_cuda": False})

    assert capabilities["mps"]["available"] is True
    assert capabilities["mps"]["claim_status"] == "partial_optimized"
    assert capabilities["mps"]["optimized"] is True
    assert (
        capabilities["mps"]["implementation"]
        == "torch_mps_compile_shader_fused_activation+native_packed_matmul"
    )
    assert capabilities["mps"]["package_format"] == "torch.mps.compile_shader,native_kernel_package"
    assert (
        capabilities["mps"]["optimized_stage"]
        == "activation_norm_rpbh_quant_rescale,packed_weight_dequant,packed_weight_matmul"
    )
    assert (
        capabilities["mps"]["implemented_stage"]
        == "activation_norm_rpbh_quant_rescale,packed_weight_dequant,packed_weight_matmul"
    )
    assert capabilities["mps"]["upstream_native_mps_op"] is False
    assert capabilities["mps"]["hf_kernel_builder_compliant"] is False
    assert capabilities["mps"]["weight_dequant_optimized"] is True
    assert capabilities["mps"]["weight_pack_optimized"] is False
    assert capabilities["mps"]["lowbit_unpack_optimized"] is False
    assert capabilities["mps"]["weight_quant_optimized"] is False
    assert capabilities["mps"]["adaln_quant_optimized"] is False
    assert capabilities["mps"]["adaln_dequant_optimized"] is False
    assert capabilities["mps"]["full_fusion"] is False
    assert capabilities["triton_cuda"]["available"] is False
    assert capabilities["triton_cuda"]["claim_status"] == "unavailable"
    assert capabilities["triton_cuda"]["optimized"] is False
    assert capabilities["triton_cuda"]["implemented_stage"] == (
        "activation_norm_rpbh_quant_rescale,packed_weight_dequant,"
        "packed_weight_matmul,lowbit_pack,lowbit_unpack,weight_rotation_fwht_quant_pack,"
        "adaln_rtn_quant_pack,adaln_rtn_dequant,adaln_rtn_packed_matmul"
    )
    assert capabilities["triton_cuda"]["optimized_stage"] is None
    assert capabilities["triton_cuda"]["weight_dequant_optimized"] is False
    assert capabilities["triton_cuda"]["weight_pack_optimized"] is False
    assert capabilities["triton_cuda"]["lowbit_unpack_optimized"] is False
    assert capabilities["triton_cuda"]["weight_quant_optimized"] is False
    assert capabilities["triton_cuda"]["adaln_quant_optimized"] is False
    assert capabilities["triton_cuda"]["adaln_dequant_optimized"] is False
    assert capabilities["triton_cuda"]["adaln_packed_matmul_optimized"] is False
    assert capabilities["triton_cuda"]["hf_kernel_builder_compliant"] is False


def test_backend_capabilities_report_mps_shader_without_native_matmul(monkeypatch):
    monkeypatch.setattr(dispatch_module, "_mps_metal_available", lambda: True)
    monkeypatch.setattr(dispatch_module, "_native_packed_matmul_available", lambda: False)
    monkeypatch.setattr(dispatch_module, "_native_cpu_packed_matmul_available", lambda: False)
    monkeypatch.setattr(dispatch_module, "_native_cpu_activation_available", lambda: False)
    monkeypatch.setattr(dispatch_module, "_native_cpu_adaln_available", lambda: False)

    capabilities = backend_capabilities(backends={"cpu": True, "mps": True, "triton_cuda": False})

    assert capabilities["mps"]["claim_status"] == "partial_optimized"
    assert capabilities["mps"]["optimized"] is True
    assert capabilities["mps"]["implementation"] == "torch_mps_compile_shader_fused_activation"
    assert capabilities["mps"]["package_format"] == "torch.mps.compile_shader"
    assert (
        capabilities["mps"]["optimized_stage"]
        == "activation_norm_rpbh_quant_rescale,packed_weight_dequant"
    )
    assert (
        capabilities["mps"]["implemented_stage"]
        == "activation_norm_rpbh_quant_rescale,packed_weight_dequant,packed_weight_matmul"
    )
    assert capabilities["mps"]["weight_dequant_optimized"] is True


def test_backend_capabilities_report_mps_native_matmul_without_shader(monkeypatch):
    monkeypatch.setattr(dispatch_module, "_mps_metal_available", lambda: False)
    monkeypatch.setattr(dispatch_module, "_native_packed_matmul_available", lambda: True)
    monkeypatch.setattr(dispatch_module, "_native_cpu_packed_matmul_available", lambda: False)
    monkeypatch.setattr(dispatch_module, "_native_cpu_activation_available", lambda: False)
    monkeypatch.setattr(dispatch_module, "_native_cpu_adaln_available", lambda: False)

    capabilities = backend_capabilities(backends={"cpu": True, "mps": True, "triton_cuda": False})

    assert capabilities["mps"]["claim_status"] == "partial_optimized"
    assert capabilities["mps"]["optimized"] is True
    assert capabilities["mps"]["implementation"] == "native_packed_matmul"
    assert capabilities["mps"]["package_format"] == "native_kernel_package"
    assert capabilities["mps"]["optimized_stage"] == "packed_weight_matmul"
    assert capabilities["mps"]["weight_dequant_optimized"] is False


def test_backend_capabilities_label_cpu_native_matmul_as_partial_not_fused(monkeypatch):
    monkeypatch.setattr(dispatch_module, "_mps_metal_available", lambda: False)
    monkeypatch.setattr(dispatch_module, "_native_packed_matmul_available", lambda: False)
    monkeypatch.setattr(dispatch_module, "_native_cpu_packed_matmul_available", lambda: True)
    monkeypatch.setattr(dispatch_module, "_native_cpu_activation_available", lambda: False)
    monkeypatch.setattr(dispatch_module, "_native_cpu_adaln_available", lambda: False)

    capabilities = backend_capabilities(backends={"cpu": True, "mps": False, "triton_cuda": False})

    assert capabilities["cpu"]["claim_status"] == "partial_optimized"
    assert capabilities["cpu"]["optimized"] is True
    assert capabilities["cpu"]["full_fusion"] is False
    assert capabilities["cpu"]["optimized_stage"] == "packed_weight_matmul"
    assert capabilities["cpu"]["implementation"] == (
        "native_exact_packed_matmul+torch_activation_reference"
    )
    assert capabilities["cpu"]["hf_kernel_builder_compliant"] is True


def test_backend_capabilities_report_native_cpu_activation_and_matmul(monkeypatch):
    monkeypatch.setattr(dispatch_module, "_mps_metal_available", lambda: False)
    monkeypatch.setattr(dispatch_module, "_native_packed_matmul_available", lambda: False)
    monkeypatch.setattr(dispatch_module, "_native_cpu_packed_matmul_available", lambda: True)
    monkeypatch.setattr(dispatch_module, "_native_cpu_activation_available", lambda: True)
    monkeypatch.setattr(dispatch_module, "_native_cpu_adaln_available", lambda: False)

    capabilities = backend_capabilities(backends={"cpu": True, "mps": False, "triton_cuda": False})

    assert capabilities["cpu"]["claim_status"] == "partial_optimized"
    assert capabilities["cpu"]["optimized"] is True
    assert capabilities["cpu"]["full_fusion"] is False
    assert capabilities["cpu"]["optimized_stage"] == (
        "activation_norm_rpbh_quant_rescale,packed_weight_matmul"
    )
    assert capabilities["cpu"]["implementation"] == (
        "native_exact_activation+native_exact_packed_matmul"
    )
    assert capabilities["cpu"]["hf_kernel_builder_compliant"] is True


def test_backend_capabilities_report_complete_native_cpu_kernel_surface(monkeypatch):
    monkeypatch.setattr(dispatch_module, "_mps_metal_available", lambda: False)
    monkeypatch.setattr(dispatch_module, "_native_packed_matmul_available", lambda: False)
    monkeypatch.setattr(dispatch_module, "_native_cpu_packed_matmul_available", lambda: True)
    monkeypatch.setattr(dispatch_module, "_native_cpu_activation_available", lambda: True)
    monkeypatch.setattr(dispatch_module, "_native_cpu_adaln_available", lambda: True)

    capabilities = backend_capabilities(backends={"cpu": True, "mps": False, "triton_cuda": False})

    assert capabilities["cpu"]["optimized_stage"] == (
        "activation_norm_rpbh_quant_rescale,packed_weight_matmul,adaln_rtn_packed_matmul"
    )
    assert capabilities["cpu"]["adaln_quant_optimized"] is False
    assert capabilities["cpu"]["adaln_dequant_optimized"] is True
    assert capabilities["cpu"]["implementation"] == (
        "native_exact_activation+native_exact_packed_matmul+native_packed_adaln_int4"
    )
    assert "without a full floating-point cache" in capabilities["cpu"]["notes"]


def test_backend_selection_accepts_injected_availability_for_gpu_paths(monkeypatch):
    monkeypatch.setattr(dispatch_module, "_torch_uses_hip", lambda: False)
    backends = {"cpu": True, "mps": False, "triton_cuda": True}

    assert (
        select_backend(torch.device("cuda"), requested="auto", backends=backends) == "triton_cuda"
    )
    assert select_backend(torch.device("mps"), requested="auto", backends=backends) == "cpu"


def test_backend_selection_separates_hip_from_cuda(monkeypatch):
    monkeypatch.setattr(dispatch_module, "_torch_uses_hip", lambda: True)
    backends = {
        "cpu": True,
        "mps": False,
        "triton_cuda": False,
        "triton_rocm": True,
    }

    assert select_backend(torch.device("cuda"), requested="auto", backends=backends) == "cpu"
    assert (
        select_backend(torch.device("cuda"), requested="triton_rocm", backends=backends)
        == "triton_rocm"
    )


def test_backend_selection_keeps_xpu_explicit_until_hardware_proof():
    backends = {
        "cpu": True,
        "mps": False,
        "triton_cuda": False,
        "triton_rocm": False,
        "triton_xpu": True,
    }

    assert select_backend(torch.device("xpu"), requested="auto", backends=backends) == "cpu"
    assert (
        select_backend(torch.device("xpu"), requested="triton_xpu", backends=backends)
        == "triton_xpu"
    )


def test_available_backends_reports_xpu_only_with_device_and_triton(monkeypatch):
    monkeypatch.setattr(dispatch_module, "_xpu_available", lambda: True)
    monkeypatch.setattr(dispatch_module, "_triton_available", lambda: True)

    assert available_backends()["triton_xpu"] is True


def test_available_backends_does_not_report_cuda_triton_on_hip(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(dispatch_module, "_triton_available", lambda: True)
    monkeypatch.setattr(dispatch_module, "_torch_uses_hip", lambda: True)

    backends = available_backends()

    assert backends["triton_cuda"] is False
    assert backends["triton_rocm"] is True


def test_rocm_capability_remains_experimental_until_hardware_proof():
    capabilities = backend_capabilities(
        backends={
            "cpu": True,
            "mps": False,
            "triton_cuda": False,
            "triton_rocm": True,
        }
    )

    rocm = capabilities["triton_rocm"]
    assert rocm["available"] is True
    assert rocm["claim_status"] == "experimental_unverified"
    assert rocm["optimized"] is False
    assert rocm["optimized_stage"] is None
    assert rocm["weight_dequant_optimized"] is False
    assert rocm["adaln_packed_matmul_optimized"] is False
    assert "without loading CUDA-native extensions" in rocm["notes"]


def test_xpu_capability_remains_experimental_until_hardware_proof():
    capabilities = backend_capabilities(
        backends={
            "cpu": True,
            "mps": False,
            "triton_cuda": False,
            "triton_rocm": False,
            "triton_xpu": True,
        }
    )

    xpu = capabilities["triton_xpu"]
    assert xpu["available"] is True
    assert xpu["claim_status"] == "experimental_unverified"
    assert xpu["optimized"] is False
    assert xpu["optimized_stage"] is None
    assert xpu["weight_dequant_optimized"] is False
    assert xpu["adaln_packed_matmul_optimized"] is False
    assert xpu["device_types"] == ["xpu"]
    assert "explicit-only" in xpu["notes"]


@pytest.mark.parametrize("bits", [2, 3, 4, 6])
def test_triton_rocm_lowbit_pack_unpack_matches_reference(bits):
    if not torch.cuda.is_available() or not getattr(torch.version, "hip", None):
        pytest.skip("a PyTorch ROCm runtime is required")
    if not available_backends().get("triton_rocm", False):
        pytest.skip("the ROCm-compatible Triton package is not importable")

    from orbitquant.kernels.triton_cuda import (
        pack_lowbit_with_triton,
        unpack_lowbit_with_triton,
    )

    values = torch.arange(0, 1 << bits, dtype=torch.uint8).repeat(5)[:37]
    expected_packed = pack_lowbit(values, bits=bits)
    actual_packed = pack_lowbit_with_triton(values.to("cuda"), bits=bits)
    actual_values = unpack_lowbit_with_triton(
        actual_packed,
        bits=bits,
        length=values.numel(),
    )

    assert torch.equal(actual_packed.cpu(), expected_packed)
    assert torch.equal(actual_values.cpu(), values)


@pytest.mark.parametrize("bits", [2, 3, 4, 6])
def test_triton_xpu_lowbit_pack_unpack_matches_reference(bits):
    xpu = getattr(torch, "xpu", None)
    if xpu is None or not xpu.is_available():
        pytest.skip("a PyTorch XPU runtime is required")
    if not available_backends().get("triton_xpu", False):
        pytest.skip("the Intel XPU Triton package is not importable")

    from orbitquant.kernels.triton_cuda import (
        pack_lowbit_with_triton,
        unpack_lowbit_with_triton,
    )

    values = torch.arange(0, 1 << bits, dtype=torch.uint8).repeat(5)[:37]
    expected_packed = pack_lowbit(values, bits=bits)
    actual_packed = pack_lowbit_with_triton(values.to("xpu"), bits=bits)
    actual_values = unpack_lowbit_with_triton(
        actual_packed,
        bits=bits,
        length=values.numel(),
    )

    assert torch.equal(actual_packed.cpu(), expected_packed)
    assert torch.equal(actual_values.cpu(), values)


def test_triton_availability_requires_importable_triton_modules(monkeypatch):
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "triton" or name.startswith("triton."):
            raise RuntimeError("broken triton import")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert dispatch_module._triton_available() is False
    assert available_backends()["triton_cuda"] is False


def test_mps_shader_source_declares_fused_activation_and_dequant_kernels():
    import orbitquant.kernels.mps as mps_module

    source = mps_module._MPS_KERNEL_SOURCE

    assert "orbitquant_row_norm_bfloat16" in source
    assert "orbitquant_row_norm_bfloat16_wide" in source
    assert "orbitquant_wide_activation_threads = 512" in source
    assert "orbitquant_fused_activation_bfloat16" in source
    assert "threadgroup float values" in source
    assert "threadgroup_barrier" in source
    assert "orbitquant_codebook_rescale" in source
    assert "orbitquant_dequantize_packed_weight" in source
    assert "linear" not in source.lower()


def test_triton_dispatch_uses_backend_function_with_reference_equivalent_output(monkeypatch):
    torch.manual_seed(0)
    x = torch.randn(2, 3, 16)
    rotation = RPBHRotation(dim=16, seed=3, block_size=8)
    codebook = get_codebook(dim=16, bits=4)
    calls = []

    def fake_triton_backend(input_tensor, *, rotation, codebook, eps, constant_tensors=None):
        calls.append(input_tensor.shape)
        assert constant_tensors is None
        return quantize_activations(input_tensor, rotation=rotation, codebook=codebook, eps=eps)

    monkeypatch.setattr(dispatch_module, "_triton_quantize_activations", fake_triton_backend)
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

    def fake_mps_backend(input_tensor, *, rotation, codebook, eps, constant_tensors=None):
        calls.append(input_tensor.shape)
        assert constant_tensors is None
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


def test_mps_dispatch_passes_preloaded_constants_to_backend(monkeypatch):
    torch.manual_seed(0)
    x = torch.randn(2, 3, 16)
    rotation = RPBHRotation(dim=16, seed=3, block_size=8)
    codebook = get_codebook(dim=16, bits=4)
    constants = {
        "permutation": rotation.permutation.clone(),
        "signs": rotation.signs.clone(),
        "centroids": codebook.centroids.clone(),
        "boundaries": codebook.boundaries.clone(),
    }
    calls = []

    def fake_mps_backend(input_tensor, *, rotation, codebook, eps, constant_tensors=None):
        calls.append(constant_tensors)
        return quantize_activations(input_tensor, rotation=rotation, codebook=codebook, eps=eps)

    monkeypatch.setattr(dispatch_module, "_mps_quantize_activations", fake_mps_backend)
    monkeypatch.setattr(
        dispatch_module,
        "available_backends",
        lambda: {"cpu": True, "mps": True, "triton_cuda": False},
    )

    expected = quantize_activations(x, rotation=rotation, codebook=codebook, eps=1e-12)
    actual = quantize_activations_kernel(
        x,
        rotation=rotation,
        codebook=codebook,
        eps=1e-12,
        backend="mps",
        constant_tensors=constants,
    )

    assert torch.allclose(actual, expected)
    assert calls == [constants]


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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("dim,block_size", [(16, 8), (3072, "paper"), (4096, "paper")])
def test_mps_fused_activation_matches_reference_without_torch_rotation(
    monkeypatch, dtype, dim, block_size
):
    if not dispatch_module._mps_metal_available():
        pytest.skip("MPS Metal shader backend is not available")

    torch.manual_seed(0)
    x = torch.randn(2, dim, device="mps", dtype=dtype)
    rotation = RPBHRotation(dim=dim, seed=3, block_size=block_size)
    codebook = get_codebook(dim=dim, bits=4)
    expected = quantize_activations(x, rotation=rotation, codebook=codebook, eps=1e-12)

    def fail_rotation(*args, **kwargs):
        raise AssertionError("MPS fused activation path should not call PyTorch RPBH")

    monkeypatch.setattr(RPBHRotation, "apply_to_activations", fail_rotation)
    actual = quantize_activations_kernel(
        x, rotation=rotation, codebook=codebook, eps=1e-12, backend="mps"
    )

    atol = 2e-3 if dtype == torch.float32 else 0.0
    assert torch.allclose(actual.cpu(), expected.cpu(), atol=atol, rtol=0.0)


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


@pytest.mark.parametrize("bits", [2, 3, 4, 6])
def test_mps_weight_dequant_kernel_matches_reference_for_supported_bits(bits):
    if not dispatch_module._mps_metal_available():
        pytest.skip("MPS Metal shader backend is not available")

    from orbitquant.kernels.mps import dequantize_packed_weight_with_mps

    out_features = 5
    in_features = 16
    codebook = get_codebook(dim=in_features, bits=bits)
    indices = (torch.arange(out_features * in_features, dtype=torch.uint8) % (2**bits)).reshape(
        out_features, in_features
    )
    packed = pack_lowbit(indices, bits=bits)
    row_norms = torch.linspace(0.5, 1.5, out_features, dtype=torch.float32)
    unpacked = unpack_lowbit(packed, bits=bits, length=indices.numel()).reshape_as(indices)
    expected = row_norms[:, None] * codebook.centroids[unpacked.to(torch.long)]

    actual = dequantize_packed_weight_with_mps(
        packed,
        row_norms,
        codebook,
        bits=bits,
        out_features=out_features,
        in_features=in_features,
    )

    assert torch.allclose(actual.cpu(), expected)


@pytest.mark.parametrize("bits", [2, 3, 4, 6])
def test_triton_weight_dequant_kernel_matches_reference_for_supported_bits(bits):
    if not torch.cuda.is_available() or not available_backends()["triton_cuda"]:
        pytest.skip("CUDA/Triton backend is not available")

    from orbitquant.kernels.triton_cuda import dequantize_packed_weight_with_triton

    out_features = 5
    in_features = 16
    codebook = get_codebook(dim=in_features, bits=bits)
    indices = (torch.arange(out_features * in_features, dtype=torch.uint8) % (2**bits)).reshape(
        out_features, in_features
    )
    packed = pack_lowbit(indices, bits=bits)
    row_norms = torch.linspace(0.5, 1.5, out_features, dtype=torch.float32)
    unpacked = unpack_lowbit(packed, bits=bits, length=indices.numel()).reshape_as(indices)
    expected = row_norms[:, None] * codebook.centroids[unpacked.to(torch.long)]

    actual = dequantize_packed_weight_with_triton(
        packed,
        row_norms,
        codebook,
        bits=bits,
        out_features=out_features,
        in_features=in_features,
    )

    assert torch.allclose(actual.cpu(), expected)


@pytest.mark.parametrize("bits", [2, 3, 4, 6])
def test_triton_packed_weight_matmul_matches_dequantized_linear(bits):
    if not torch.cuda.is_available() or not available_backends()["triton_cuda"]:
        pytest.skip("CUDA/Triton backend is not available")

    from orbitquant.kernels.triton_cuda import matmul_packed_weight_with_triton

    torch.manual_seed(11 + bits)
    tokens = 9
    in_features = 16
    out_features = 7
    codebook = get_codebook(dim=in_features, bits=bits)
    indices = (torch.arange(out_features * in_features, dtype=torch.uint8) % (2**bits)).reshape(
        out_features, in_features
    )
    packed = pack_lowbit(indices, bits=bits).to("cuda")
    row_norms = torch.linspace(0.5, 1.5, out_features, dtype=torch.bfloat16, device="cuda")
    x = torch.randn(tokens, in_features, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(out_features, device="cuda", dtype=torch.bfloat16)

    unpacked = unpack_lowbit(packed.cpu(), bits=bits, length=indices.numel()).reshape_as(indices)
    weight = row_norms.float().cpu()[:, None] * codebook.centroids[unpacked.to(torch.long)]
    expected = torch.nn.functional.linear(x.float().cpu(), weight, bias.float().cpu())

    actual = matmul_packed_weight_with_triton(
        x,
        packed,
        row_norms,
        codebook,
        bits=bits,
        out_features=out_features,
        in_features=in_features,
        bias=bias,
        block_m=16,
        block_n=16,
        block_k=32,
        num_warps=4,
    )

    assert actual.is_cuda
    assert actual.dtype == x.dtype
    assert actual.shape == (tokens, out_features)
    assert torch.allclose(actual.float().cpu(), expected, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("bits", [2, 3, 4, 6])
def test_triton_lowbit_pack_unpack_stays_on_cuda_and_matches_reference(bits):
    if not torch.cuda.is_available() or not available_backends()["triton_cuda"]:
        pytest.skip("CUDA/Triton backend is not available")

    values_cpu = (torch.arange(0, 1027, dtype=torch.uint8) * 3) % (2**bits)
    expected = pack_lowbit(values_cpu, bits=bits)

    packed = pack_lowbit(values_cpu.to("cuda"), bits=bits)
    unpacked = unpack_lowbit(packed, bits=bits, length=values_cpu.numel())

    assert packed.is_cuda
    assert unpacked.is_cuda
    assert torch.equal(packed.cpu(), expected)
    assert torch.equal(unpacked.cpu(), values_cpu)


def test_triton_lowbit_pack_can_skip_redundant_range_validation():
    if not torch.cuda.is_available() or not available_backends()["triton_cuda"]:
        pytest.skip("CUDA/Triton backend is not available")

    values = torch.tensor([0, 1, 2, 3], device="cuda", dtype=torch.uint8)

    packed = pack_lowbit(values, bits=2, validate=False)
    unpacked = unpack_lowbit(packed, bits=2, length=values.numel())

    assert packed.is_cuda
    assert unpacked.is_cuda
    assert torch.equal(unpacked.cpu(), values.cpu())


@pytest.mark.parametrize("bits", [2, 3, 4, 6])
def test_triton_weight_quant_indices_match_reference_rotation_path(bits):
    if not torch.cuda.is_available() or not available_backends()["triton_cuda"]:
        pytest.skip("CUDA/Triton backend is not available")

    from orbitquant.kernels.triton_cuda import quantize_weight_indices_with_triton

    torch.manual_seed(123)
    weight = torch.randn(9, 32, device="cuda", dtype=torch.float32)
    rotation = RPBHRotation(dim=32, seed=5, block_size=8)
    codebook = get_codebook(dim=32, bits=bits)
    row_norms = weight.norm(dim=-1).clamp_min(1e-12)
    rotated = rotation.apply_to_weight(weight)
    expected = codebook.quantize_indices(rotated / row_norms[:, None])

    actual = quantize_weight_indices_with_triton(
        weight,
        row_norms,
        rotation=rotation,
        codebook=codebook,
    )

    assert actual.is_cuda
    assert torch.equal(actual.cpu(), expected.cpu())


@pytest.mark.parametrize("bits", [2, 3, 4, 6])
def test_triton_weight_quant_pack_matches_reference_two_step_path(bits):
    if not torch.cuda.is_available() or not available_backends()["triton_cuda"]:
        pytest.skip("CUDA/Triton backend is not available")

    from orbitquant.kernels.triton_cuda import quantize_weight_packed_with_triton

    torch.manual_seed(124)
    weight = torch.randn(9, 32, device="cuda", dtype=torch.float32)
    rotation = RPBHRotation(dim=32, seed=5, block_size=8)
    codebook = get_codebook(dim=32, bits=bits)
    row_norms = weight.norm(dim=-1).clamp_min(1e-12)
    rotated = rotation.apply_to_weight(weight)
    expected_indices = codebook.quantize_indices(rotated / row_norms[:, None])
    expected = pack_lowbit(expected_indices, bits=bits)

    actual = quantize_weight_packed_with_triton(
        weight,
        row_norms,
        rotation=rotation,
        codebook=codebook,
        bits=bits,
    )

    assert actual.is_cuda
    assert torch.equal(actual.cpu(), expected.cpu())


def test_triton_row_norms_match_reference_for_bfloat16_weight_input():
    if not torch.cuda.is_available() or not available_backends()["triton_cuda"]:
        pytest.skip("CUDA/Triton backend is not available")

    from orbitquant.kernels.triton_cuda import row_norms_with_triton

    torch.manual_seed(125)
    weight = torch.randn(11, 32, device="cuda", dtype=torch.bfloat16)
    expected = weight.float().norm(dim=-1).clamp_min(1e-12)

    actual = row_norms_with_triton(weight, eps=1e-12)

    assert actual.is_cuda
    assert actual.dtype == torch.float32
    assert torch.allclose(actual, expected, atol=1e-4, rtol=1e-4)


def test_triton_weight_quant_pack_accepts_bfloat16_without_reference_rotation_path(monkeypatch):
    if not torch.cuda.is_available() or not available_backends()["triton_cuda"]:
        pytest.skip("CUDA/Triton backend is not available")

    from orbitquant.kernels.triton_cuda import (
        quantize_weight_packed_with_triton,
        row_norms_with_triton,
    )

    torch.manual_seed(126)
    weight = torch.randn(9, 32, device="cuda", dtype=torch.bfloat16)
    rotation = RPBHRotation(dim=32, seed=5, block_size=8)
    codebook = get_codebook(dim=32, bits=4)
    row_norms = row_norms_with_triton(weight, eps=1e-12)
    rotated = rotation.apply_to_weight(weight.float())
    expected_indices = codebook.quantize_indices(rotated / row_norms[:, None])
    expected = pack_lowbit(expected_indices, bits=4)

    def fail_rotation(*args, **kwargs):
        raise AssertionError("triton_cuda weight quantization should not call PyTorch RPBH")

    monkeypatch.setattr(RPBHRotation, "apply_to_weight", fail_rotation)

    actual = quantize_weight_packed_with_triton(
        weight,
        row_norms,
        rotation=rotation,
        codebook=codebook,
        bits=4,
    )

    assert actual.is_cuda
    assert torch.equal(actual.cpu(), expected.cpu())


def test_triton_weight_quantization_reuses_cuda_constant_tensors():
    if not torch.cuda.is_available() or not available_backends()["triton_cuda"]:
        pytest.skip("CUDA/Triton backend is not available")

    from orbitquant.kernels.triton_cuda import (
        _weight_quantization_constants,
        clear_triton_constant_cache,
    )

    clear_triton_constant_cache()
    rotation = RPBHRotation(dim=32, seed=5, block_size=8)
    codebook = get_codebook(dim=32, bits=4)

    first = _weight_quantization_constants(
        rotation=rotation,
        codebook=codebook,
        device=torch.device("cuda"),
    )
    second = _weight_quantization_constants(
        rotation=rotation,
        codebook=codebook,
        device=torch.device("cuda"),
    )

    assert all(tensor.is_cuda for tensor in first)
    assert [tensor.data_ptr() for tensor in first] == [tensor.data_ptr() for tensor in second]


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


def test_triton_cuda_activation_kernel_matches_reference_without_torch_rotation(monkeypatch):
    if not torch.cuda.is_available() or not available_backends()["triton_cuda"]:
        pytest.skip("CUDA/Triton backend is not available")

    torch.manual_seed(0)
    x = torch.randn(3, 7, 32, device="cuda", dtype=torch.float32)
    rotation = RPBHRotation(dim=32, seed=11, block_size=8)
    codebook = get_codebook(dim=32, bits=3)
    expected = quantize_activations(x, rotation=rotation, codebook=codebook, eps=1e-12)

    def fail_rotation(*args, **kwargs):
        raise AssertionError("triton_cuda activation path should not call PyTorch RPBH rotation")

    monkeypatch.setattr(RPBHRotation, "apply_to_activations", fail_rotation)

    actual = quantize_activations_kernel(
        x, rotation=rotation, codebook=codebook, eps=1e-12, backend="triton_cuda"
    )

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_triton_cuda_activation_kernel_preserves_bfloat16_dtype():
    if not torch.cuda.is_available() or not available_backends()["triton_cuda"]:
        pytest.skip("CUDA/Triton backend is not available")

    torch.manual_seed(1)
    x = torch.randn(2, 5, 16, device="cuda", dtype=torch.bfloat16)
    rotation = RPBHRotation(dim=16, seed=3, block_size=8)
    codebook = get_codebook(dim=16, bits=4)
    expected = quantize_activations(x, rotation=rotation, codebook=codebook, eps=1e-12)
    actual = quantize_activations_kernel(
        x, rotation=rotation, codebook=codebook, eps=1e-12, backend="triton_cuda"
    )

    assert actual.dtype == torch.bfloat16
    assert torch.allclose(actual.float(), expected.float(), atol=1e-2, rtol=1e-2)


def test_triton_cuda_activation_kernel_matches_paper_sized_rpbh_block():
    if not torch.cuda.is_available() or not available_backends()["triton_cuda"]:
        pytest.skip("CUDA/Triton backend is not available")

    torch.manual_seed(2)
    x = torch.randn(2, 3072, device="cuda", dtype=torch.float32)
    rotation = RPBHRotation(dim=3072, seed=0, block_size="paper")
    codebook = get_codebook(dim=3072, bits=4)
    expected = quantize_activations(x, rotation=rotation, codebook=codebook, eps=1e-12)
    actual = quantize_activations_kernel(
        x, rotation=rotation, codebook=codebook, eps=1e-12, backend="triton_cuda"
    )

    assert rotation.block_size == 1024
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_triton_cuda_activation_kernel_matches_large_rpbh_block():
    if not torch.cuda.is_available() or not available_backends()["triton_cuda"]:
        pytest.skip("CUDA/Triton backend is not available")

    torch.manual_seed(3)
    x = torch.randn(3, 4096, device="cuda", dtype=torch.bfloat16)
    rotation = RPBHRotation(dim=4096, seed=0, block_size="paper")
    codebook = get_codebook(dim=4096, bits=4)
    expected = quantize_activations(x, rotation=rotation, codebook=codebook, eps=1e-12)
    actual = quantize_activations_kernel(
        x, rotation=rotation, codebook=codebook, eps=1e-12, backend="triton_cuda"
    )

    assert rotation.block_size == 4096
    assert actual.dtype == torch.bfloat16
    assert torch.allclose(actual.float(), expected.float(), atol=1e-2, rtol=1e-2)


def test_triton_cuda_activation_kernel_matches_very_large_rpbh_block():
    if not torch.cuda.is_available() or not available_backends()["triton_cuda"]:
        pytest.skip("CUDA/Triton backend is not available")

    torch.manual_seed(4)
    x = torch.randn(2, 16384, device="cuda", dtype=torch.bfloat16)
    rotation = RPBHRotation(dim=16384, seed=0, block_size="paper")
    codebook = get_codebook(dim=16384, bits=4)
    expected = quantize_activations(x, rotation=rotation, codebook=codebook, eps=1e-12)
    actual = quantize_activations_kernel(
        x, rotation=rotation, codebook=codebook, eps=1e-12, backend="triton_cuda"
    )

    assert rotation.block_size == 16384
    assert actual.dtype == torch.bfloat16
    assert torch.allclose(actual.float(), expected.float(), atol=1e-2, rtol=1e-2)


def test_cuda_quantize_linear_modules_keeps_packed_buffers_on_gpu_until_serialization():
    if not torch.cuda.is_available() or not available_backends()["triton_cuda"]:
        pytest.skip("CUDA/Triton backend is not available")

    torch.manual_seed(3)
    model = torch.nn.Module()
    model.transformer_blocks = torch.nn.ModuleList(
        [torch.nn.ModuleDict({"attn": torch.nn.ModuleDict({"to_q": torch.nn.Linear(32, 32)})})]
    )
    config = OrbitQuantConfig(block_size=8, target_policy="generic_dit")

    summary = quantize_linear_modules(model, config, quantization_device="cuda")

    quantized = model.transformer_blocks[0]["attn"]["to_q"]
    assert isinstance(quantized, OrbitQuantLinear)
    assert summary.quantization_device == "cuda"
    assert summary.weight_quantization_backend == "triton_cuda"
    assert quantized.packed_weight_indices is not None
    assert quantized.row_norms is not None
    assert quantized.packed_weight_indices.is_cuda
    assert quantized.row_norms.is_cuda
    assert quantized._rotation_permutation.is_cuda
    assert quantized._rotation_signs.is_cuda
    assert quantized._activation_codebook_centroids.is_cuda
    assert quantized._activation_codebook_boundaries.is_cuda


def _build_w4a4_triton_fixture(in_features: int, out_features: int) -> dict:
    from orbitquant.kernels.triton_cuda import (
        fit_int8_centroid_surrogate,
        quantize_weight_packed_with_triton,
        row_norms_with_triton,
    )

    torch.manual_seed(7)
    weight = torch.randn(out_features, in_features, device="cuda", dtype=torch.bfloat16)
    rotation = RPBHRotation(dim=in_features, seed=0, block_size="paper")
    weight_codebook = get_codebook(dim=in_features, bits=4)
    activation_codebook = get_codebook(dim=in_features, bits=4)
    row_norms = row_norms_with_triton(weight, eps=1e-10)
    packed = quantize_weight_packed_with_triton(
        weight, row_norms, rotation=rotation, codebook=weight_codebook, bits=4
    )
    activation_codes, activation_scale = fit_int8_centroid_surrogate(
        activation_codebook.centroids
    )
    weight_codes, weight_scale = fit_int8_centroid_surrogate(weight_codebook.centroids)
    return {
        "rotation": rotation,
        "weight_codebook": weight_codebook,
        "activation_codebook": activation_codebook,
        "row_norms": row_norms.to(torch.bfloat16),
        "packed": packed,
        "activation_codes": activation_codes.to("cuda"),
        "activation_scale": activation_scale,
        "weight_codes": weight_codes.to("cuda"),
        "weight_scale": weight_scale,
        "in_features": in_features,
        "out_features": out_features,
    }


@pytest.mark.parametrize("rows", [1, 16, 17, 31, 33, 77])
def test_w4a4_int_mm_path_supports_arbitrary_row_counts(rows):
    if not torch.cuda.is_available() or not available_backends()["triton_cuda"]:
        pytest.skip("CUDA/Triton backend is not available")

    from orbitquant.kernels.triton_cuda import (
        dequantize_packed_weight_with_triton,
        matmul_packed_w4a4_with_int_mm,
        quantize_activations_packed_w4_with_triton,
        quantize_activations_with_triton,
    )

    fixture = _build_w4a4_triton_fixture(in_features=512, out_features=512)
    x = torch.randn(rows, fixture["in_features"], device="cuda", dtype=torch.bfloat16)
    packed_x, token_norms = quantize_activations_packed_w4_with_triton(
        x,
        rotation=fixture["rotation"],
        codebook=fixture["activation_codebook"],
        eps=1e-10,
    )

    actual = matmul_packed_w4a4_with_int_mm(
        packed_x,
        fixture["packed"],
        token_norms,
        fixture["row_norms"],
        fixture["activation_codes"],
        fixture["weight_codes"],
        activation_scale=fixture["activation_scale"],
        weight_scale=fixture["weight_scale"],
        out_features=fixture["out_features"],
        in_features=fixture["in_features"],
        output_dtype=torch.bfloat16,
    )

    dequantized = dequantize_packed_weight_with_triton(
        fixture["packed"],
        fixture["row_norms"],
        fixture["weight_codebook"],
        bits=4,
        out_features=fixture["out_features"],
        in_features=fixture["in_features"],
    ).to(torch.bfloat16)
    quantized_values = quantize_activations_with_triton(
        x,
        rotation=fixture["rotation"],
        codebook=fixture["activation_codebook"],
        eps=1e-10,
    )
    expected = torch.nn.functional.linear(quantized_values.to(torch.bfloat16), dequantized)

    assert actual.shape == (rows, fixture["out_features"])
    relative_error = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert relative_error <= 3e-2


@pytest.mark.parametrize("rows", [32, 400])
def test_int8_activation_lowbit_fused_matches_packed_fused(rows):
    if not torch.cuda.is_available() or not available_backends()["triton_cuda"]:
        pytest.skip("CUDA/Triton backend is not available")

    from orbitquant.kernels.triton_cuda import (
        matmul_int8_activations_packed_lowbit_fused_with_triton,
        matmul_packed_w4a4_fused_with_triton,
        quantize_activations_packed_w4_with_triton,
    )

    fixture = _build_w4a4_triton_fixture(in_features=512, out_features=512)
    x = torch.randn(rows, fixture["in_features"], device="cuda", dtype=torch.bfloat16)
    packed_x, token_norms = quantize_activations_packed_w4_with_triton(
        x,
        rotation=fixture["rotation"],
        codebook=fixture["activation_codebook"],
        eps=1e-10,
    )
    bias = torch.randn(fixture["out_features"], device="cuda", dtype=torch.bfloat16)
    packed_result = matmul_packed_w4a4_fused_with_triton(
        packed_x,
        fixture["packed"],
        token_norms,
        fixture["row_norms"],
        fixture["activation_codes"],
        fixture["weight_codes"],
        activation_scale=fixture["activation_scale"],
        weight_scale=fixture["weight_scale"],
        out_features=fixture["out_features"],
        in_features=fixture["in_features"],
        bias=bias,
        output_dtype=torch.bfloat16,
    )

    # Decode the packed nibbles into the same INT8 surrogates the fused
    # kernel would produce, then run the direct-activation kernel.
    low = (packed_x & 15).to(torch.long)
    high = (packed_x >> 4).to(torch.long)
    int8_x = torch.empty(
        (rows, fixture["in_features"]), device="cuda", dtype=torch.int8
    )
    int8_x[:, 0::2] = fixture["activation_codes"][low]
    int8_x[:, 1::2] = fixture["activation_codes"][high]

    direct_result = matmul_int8_activations_packed_lowbit_fused_with_triton(
        int8_x,
        fixture["packed"],
        token_norms,
        fixture["row_norms"],
        fixture["weight_codes"],
        weight_bits=4,
        activation_scale=fixture["activation_scale"],
        weight_scale=fixture["weight_scale"],
        out_features=fixture["out_features"],
        in_features=fixture["in_features"],
        bias=bias,
        output_dtype=torch.bfloat16,
    )
    torch.testing.assert_close(direct_result, packed_result)


@pytest.mark.parametrize("rows", [32, 400])
def test_w2a4_fused_triton_matmul_matches_int8_emulation(rows):
    if not torch.cuda.is_available() or not available_backends()["triton_cuda"]:
        pytest.skip("CUDA/Triton backend is not available")

    from orbitquant.kernels.triton_cuda import (
        matmul_int8_activations_packed_lowbit_fused_with_triton,
        matmul_packed_w2a4_fused_with_triton,
    )

    torch.manual_seed(0)
    in_features, out_features = 512, 512
    act_indices = torch.randint(0, 16, (rows, in_features), dtype=torch.uint8)
    weight_indices = torch.randint(0, 4, (out_features, in_features), dtype=torch.uint8)
    act_flat = act_indices.flatten()
    packed_x = (act_flat[0::2] | (act_flat[1::2] << 4)).reshape(
        rows, in_features // 2
    ).cuda()
    w_flat = weight_indices.flatten()
    packed_w = (
        w_flat[0::4] | (w_flat[1::4] << 2) | (w_flat[2::4] << 4) | (w_flat[3::4] << 6)
    ).cuda()
    activation_codes = torch.arange(-8, 8, dtype=torch.int8, device="cuda")
    weight_codes = torch.tensor([-7, -2, 2, 7], dtype=torch.int8, device="cuda")
    token_norms = (torch.rand(rows, device="cuda") + 0.5).float()
    row_norms = (torch.rand(out_features, device="cuda") + 0.5).float()
    scale = 0.01
    act_i8 = activation_codes[act_indices.long().cuda()]
    w_i8 = weight_codes[weight_indices.long().cuda()]
    expected = (
        act_i8.float() @ w_i8.float().t()
        * token_norms[:, None] * row_norms[None, :] * scale
    )

    packed_result = matmul_packed_w2a4_fused_with_triton(
        packed_x,
        packed_w,
        token_norms,
        row_norms,
        activation_codes,
        weight_codes,
        activation_scale=scale,
        weight_scale=1.0,
        out_features=out_features,
        in_features=in_features,
        output_dtype=torch.bfloat16,
    )
    assert (packed_result.float() - expected).abs().max().item() < 0.5

    direct_result = matmul_int8_activations_packed_lowbit_fused_with_triton(
        act_i8,
        packed_w,
        token_norms,
        row_norms,
        weight_codes,
        weight_bits=2,
        activation_scale=scale,
        weight_scale=1.0,
        out_features=out_features,
        in_features=in_features,
        output_dtype=torch.bfloat16,
    )
    torch.testing.assert_close(direct_result, packed_result)


@pytest.mark.parametrize("rows", [17, 48])
def test_w4a4_int_mm_path_supports_unaligned_rows_on_large_shape(rows):
    if not torch.cuda.is_available() or not available_backends()["triton_cuda"]:
        pytest.skip("CUDA/Triton backend is not available")

    from orbitquant.kernels.triton_cuda import (
        dequantize_packed_weight_with_triton,
        matmul_packed_w4a4_with_int_mm,
        quantize_activations_packed_w4_with_triton,
        quantize_activations_with_triton,
    )

    fixture = _build_w4a4_triton_fixture(in_features=4096, out_features=2048)
    x = torch.randn(rows, fixture["in_features"], device="cuda", dtype=torch.bfloat16)
    packed_x, token_norms = quantize_activations_packed_w4_with_triton(
        x,
        rotation=fixture["rotation"],
        codebook=fixture["activation_codebook"],
        eps=1e-10,
    )

    actual = matmul_packed_w4a4_with_int_mm(
        packed_x,
        fixture["packed"],
        token_norms,
        fixture["row_norms"],
        fixture["activation_codes"],
        fixture["weight_codes"],
        activation_scale=fixture["activation_scale"],
        weight_scale=fixture["weight_scale"],
        out_features=fixture["out_features"],
        in_features=fixture["in_features"],
        output_dtype=torch.bfloat16,
        chunk_out_features=2048,
    )

    dequantized = dequantize_packed_weight_with_triton(
        fixture["packed"],
        fixture["row_norms"],
        fixture["weight_codebook"],
        bits=4,
        out_features=fixture["out_features"],
        in_features=fixture["in_features"],
    ).to(torch.bfloat16)
    quantized_values = quantize_activations_with_triton(
        x,
        rotation=fixture["rotation"],
        codebook=fixture["activation_codebook"],
        eps=1e-10,
    )
    expected = torch.nn.functional.linear(quantized_values.to(torch.bfloat16), dequantized)
    relative_error = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert relative_error <= 3e-2


def test_adaln_triton_default_config_compiles_on_flux_modulation_shape():
    if not torch.cuda.is_available() or not available_backends()["triton_cuda"]:
        pytest.skip("CUDA/Triton backend is not available")

    from orbitquant.kernels.triton_cuda import (
        dequantize_adaln_weight_with_triton,
        matmul_packed_adaln_int4_with_triton,
        quantize_adaln_weight_with_triton,
    )

    torch.manual_seed(3)
    in_features, out_features = 3072, 18432
    weight = torch.randn(out_features, in_features, device="cuda", dtype=torch.float32)
    packed, scales = quantize_adaln_weight_with_triton(weight, group_size=64)
    x = torch.randn(1, in_features, device="cuda", dtype=torch.bfloat16)

    actual = matmul_packed_adaln_int4_with_triton(
        x,
        packed,
        scales,
        out_features=out_features,
        in_features=in_features,
        group_size=64,
    )

    reference_weight = dequantize_adaln_weight_with_triton(
        packed,
        scales,
        out_features=out_features,
        in_features=in_features,
        group_size=64,
    ).to(torch.bfloat16)
    expected = torch.nn.functional.linear(x, reference_weight)
    relative_error = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert relative_error <= 5e-3


@pytest.mark.parametrize("rows", [1, 3, 16, 17])
def test_adaln_triton_small_rows_match_dequantized_reference(rows):
    if not torch.cuda.is_available() or not available_backends()["triton_cuda"]:
        pytest.skip("CUDA/Triton backend is not available")

    from orbitquant.kernels.triton_cuda import (
        dequantize_adaln_weight_with_triton,
        matmul_packed_adaln_int4_with_triton,
        quantize_adaln_weight_with_triton,
    )

    torch.manual_seed(4)
    in_features, out_features = 256, 192
    weight = torch.randn(out_features, in_features, device="cuda", dtype=torch.float32)
    packed, scales = quantize_adaln_weight_with_triton(weight, group_size=64)
    bias = torch.randn(out_features, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(rows, in_features, device="cuda", dtype=torch.bfloat16)

    actual = matmul_packed_adaln_int4_with_triton(
        x,
        packed,
        scales,
        out_features=out_features,
        in_features=in_features,
        group_size=64,
        bias=bias,
    )

    reference_weight = dequantize_adaln_weight_with_triton(
        packed,
        scales,
        out_features=out_features,
        in_features=in_features,
        group_size=64,
    ).to(torch.bfloat16)
    expected = torch.nn.functional.linear(x, reference_weight, bias)
    relative_error = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert relative_error <= 5e-3


@pytest.mark.parametrize("dim", [2048, 4096])
def test_triton_cuda_activation_kernel_matches_reference_for_large_blocks(dim):
    if not torch.cuda.is_available() or not available_backends()["triton_cuda"]:
        pytest.skip("CUDA/Triton backend is not available")

    torch.manual_seed(5)
    x = torch.randn(3, dim, device="cuda", dtype=torch.float32)
    rotation = RPBHRotation(dim=dim, seed=0, block_size="paper")
    codebook = get_codebook(dim=dim, bits=4)
    expected = quantize_activations(x, rotation=rotation, codebook=codebook, eps=1e-12)
    actual = quantize_activations_kernel(
        x, rotation=rotation, codebook=codebook, eps=1e-12, backend="triton_cuda"
    )

    assert rotation.block_size == dim
    assert torch.allclose(actual, expected, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("dim", [2048, 4096])
def test_triton_packed_w4_activation_quant_matches_values_path_for_large_blocks(dim):
    if not torch.cuda.is_available() or not available_backends()["triton_cuda"]:
        pytest.skip("CUDA/Triton backend is not available")

    from orbitquant.kernels.triton_cuda import (
        quantize_activations_packed_w4_with_triton,
        quantize_activations_with_triton,
    )

    torch.manual_seed(6)
    rows = 3
    x = torch.randn(rows, dim, device="cuda", dtype=torch.bfloat16)
    rotation = RPBHRotation(dim=dim, seed=0, block_size="paper")
    codebook = get_codebook(dim=dim, bits=4)

    packed_x, token_norms = quantize_activations_packed_w4_with_triton(
        x, rotation=rotation, codebook=codebook, eps=1e-10
    )
    values = quantize_activations_with_triton(
        x, rotation=rotation, codebook=codebook, eps=1e-10
    )

    codes = torch.empty(rows, dim, dtype=torch.uint8, device="cuda")
    codes[:, 0::2] = packed_x & 15
    codes[:, 1::2] = packed_x >> 4
    centroids = codebook.centroids.to(device="cuda")
    reconstructed = (centroids[codes.long()] * token_norms[:, None]).to(values.dtype)

    assert torch.equal(reconstructed, values)


def test_triton_activation_kernel_accepts_int32_permutation_constants():
    if not torch.cuda.is_available() or not available_backends()["triton_cuda"]:
        pytest.skip("CUDA/Triton backend is not available")

    from orbitquant.kernels.triton_cuda import quantize_activations_with_triton

    torch.manual_seed(8)
    dim = 64
    x = torch.randn(4, dim, device="cuda", dtype=torch.float32)
    rotation = RPBHRotation(dim=dim, seed=1, block_size=32)
    codebook = get_codebook(dim=dim, bits=4)
    expected = quantize_activations(x, rotation=rotation, codebook=codebook, eps=1e-12)

    constants = {
        "permutation": rotation.permutation.to(device="cuda", dtype=torch.int32),
        "signs": rotation.signs.to(device="cuda"),
        "centroids": codebook.centroids.to(device="cuda"),
        "boundaries": codebook.boundaries.to(device="cuda"),
    }
    actual = quantize_activations_with_triton(
        x, rotation=rotation, codebook=codebook, eps=1e-12, constant_tensors=constants
    )

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_triton_packed_matmul_recovers_from_shared_memory_overflow():
    if not torch.cuda.is_available() or not available_backends()["triton_cuda"]:
        pytest.skip("CUDA/Triton backend is not available")

    from orbitquant.kernels.triton_cuda import matmul_packed_weight_with_triton

    torch.manual_seed(9)
    in_features, out_features, rows = 4096, 512, 64
    codebook = get_codebook(dim=in_features, bits=4)
    indices = (
        torch.arange(out_features * in_features, dtype=torch.int64) % 16
    ).to(torch.uint8)
    packed = pack_lowbit(indices, bits=4).to("cuda")
    row_norms = torch.linspace(0.5, 1.5, out_features, dtype=torch.bfloat16, device="cuda")
    x = torch.randn(rows, in_features, device="cuda", dtype=torch.bfloat16)

    weight = row_norms.float().cpu()[:, None] * codebook.centroids[
        indices.reshape(out_features, in_features).to(torch.long)
    ]
    expected = torch.nn.functional.linear(x.float().cpu(), weight)

    actual = matmul_packed_weight_with_triton(
        x,
        packed,
        row_norms,
        codebook,
        bits=4,
        out_features=out_features,
        in_features=in_features,
        block_m=64,
        block_n=256,
        block_k=128,
        num_warps=8,
    )

    assert torch.allclose(actual.float().cpu(), expected, atol=1e-1, rtol=1e-2)


@pytest.mark.parametrize("rows", [32, 64, 400])
def test_w4a4_fused_triton_matmul_matches_dequantized_reference(rows):
    if not torch.cuda.is_available() or not available_backends()["triton_cuda"]:
        pytest.skip("CUDA/Triton backend is not available")

    from orbitquant.kernels.triton_cuda import (
        dequantize_packed_weight_with_triton,
        matmul_packed_w4a4_fused_with_triton,
        quantize_activations_packed_w4_with_triton,
        quantize_activations_with_triton,
    )

    fixture = _build_w4a4_triton_fixture(in_features=512, out_features=512)
    x = torch.randn(rows, fixture["in_features"], device="cuda", dtype=torch.bfloat16)
    packed_x, token_norms = quantize_activations_packed_w4_with_triton(
        x,
        rotation=fixture["rotation"],
        codebook=fixture["activation_codebook"],
        eps=1e-10,
    )
    bias = torch.randn(fixture["out_features"], device="cuda", dtype=torch.bfloat16)

    actual = matmul_packed_w4a4_fused_with_triton(
        packed_x,
        fixture["packed"],
        token_norms,
        fixture["row_norms"],
        fixture["activation_codes"],
        fixture["weight_codes"],
        activation_scale=fixture["activation_scale"],
        weight_scale=fixture["weight_scale"],
        out_features=fixture["out_features"],
        in_features=fixture["in_features"],
        bias=bias,
        output_dtype=torch.bfloat16,
    )

    dequantized = dequantize_packed_weight_with_triton(
        fixture["packed"],
        fixture["row_norms"],
        fixture["weight_codebook"],
        bits=4,
        out_features=fixture["out_features"],
        in_features=fixture["in_features"],
    ).to(torch.bfloat16)
    quantized_values = quantize_activations_with_triton(
        x,
        rotation=fixture["rotation"],
        codebook=fixture["activation_codebook"],
        eps=1e-10,
    )
    expected = torch.nn.functional.linear(
        quantized_values.to(torch.bfloat16), dequantized, bias
    )
    relative_error = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert relative_error <= 3e-2


def test_w4a4_int_mm_decoded_weight_cache_matches_per_forward_decode():
    if not torch.cuda.is_available() or not available_backends()["triton_cuda"]:
        pytest.skip("CUDA/Triton backend is not available")

    from orbitquant.kernels.triton_cuda import (
        decode_packed_w4_weight_to_int8,
        matmul_packed_w4a4_with_int_mm,
        quantize_activations_packed_w4_with_triton,
    )

    fixture = _build_w4a4_triton_fixture(in_features=512, out_features=512)
    x = torch.randn(64, fixture["in_features"], device="cuda", dtype=torch.bfloat16)
    packed_x, token_norms = quantize_activations_packed_w4_with_triton(
        x,
        rotation=fixture["rotation"],
        codebook=fixture["activation_codebook"],
        eps=1e-10,
    )
    shared_kwargs = dict(
        activation_scale=fixture["activation_scale"],
        weight_scale=fixture["weight_scale"],
        out_features=fixture["out_features"],
        in_features=fixture["in_features"],
        output_dtype=torch.bfloat16,
    )

    baseline = matmul_packed_w4a4_with_int_mm(
        packed_x,
        fixture["packed"],
        token_norms,
        fixture["row_norms"],
        fixture["activation_codes"],
        fixture["weight_codes"],
        **shared_kwargs,
    )
    decoded = decode_packed_w4_weight_to_int8(
        fixture["packed"],
        fixture["weight_codes"],
        out_features=fixture["out_features"],
        in_features=fixture["in_features"],
    )
    cached = matmul_packed_w4a4_with_int_mm(
        packed_x,
        fixture["packed"],
        token_norms,
        fixture["row_norms"],
        fixture["activation_codes"],
        fixture["weight_codes"],
        decoded_weight=decoded,
        **shared_kwargs,
    )

    assert torch.equal(baseline, cached)


def test_config_w4a4_int8_weight_cache_flag_roundtrips():
    config = OrbitQuantConfig(w4a4_int8_weight_cache=True)
    assert config.w4a4_int8_weight_cache is True
    restored = OrbitQuantConfig.from_dict(config.to_dict())
    assert restored.w4a4_int8_weight_cache is True
    assert OrbitQuantConfig().w4a4_int8_weight_cache is False
