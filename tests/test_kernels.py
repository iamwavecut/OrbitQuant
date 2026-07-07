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
from orbitquant.layers import OrbitQuantLinear
from orbitquant.modeling import quantize_linear_modules
from orbitquant.packing import pack_lowbit, unpack_lowbit
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
    assert capabilities["cpu"]["claim_status"] == "reference_only"
    assert capabilities["cpu"]["optimized"] is False
    assert capabilities["cpu"]["weight_dequant_optimized"] is False
    assert capabilities["cpu"]["weight_pack_optimized"] is False
    assert capabilities["cpu"]["lowbit_unpack_optimized"] is False
    assert capabilities["cpu"]["weight_quant_optimized"] is False
    assert capabilities["cpu"]["adaln_quant_optimized"] is False
    assert capabilities["cpu"]["adaln_dequant_optimized"] is False
    assert capabilities["cpu"]["implementation"] == "torch_reference"
    assert capabilities["mps"]["available"] is True
    assert capabilities["mps"]["claim_status"] == "reference_only"
    assert capabilities["mps"]["optimized"] is False
    assert capabilities["mps"]["weight_dequant_optimized"] is False
    assert capabilities["mps"]["weight_pack_optimized"] is False
    assert capabilities["mps"]["lowbit_unpack_optimized"] is False
    assert capabilities["mps"]["weight_quant_optimized"] is False
    assert capabilities["mps"]["adaln_quant_optimized"] is False
    assert capabilities["mps"]["adaln_dequant_optimized"] is False
    assert capabilities["mps"]["implementation"] == "torch_reference_mps"
    assert capabilities["triton_cuda"]["available"] is True
    assert capabilities["triton_cuda"]["claim_status"] == "partial_optimized"
    assert capabilities["triton_cuda"]["optimized"] is True
    assert capabilities["triton_cuda"]["optimized_stage"] == (
        "activation_norm_rpbh_quant_rescale,packed_weight_dequant,"
        "packed_weight_matmul,lowbit_pack,lowbit_unpack,weight_rotation_fwht_quant_pack,"
        "adaln_rtn_quant_pack,adaln_rtn_dequant"
    )
    assert capabilities["triton_cuda"]["weight_dequant_optimized"] is True
    assert capabilities["triton_cuda"]["weight_pack_optimized"] is True
    assert capabilities["triton_cuda"]["lowbit_unpack_optimized"] is True
    assert capabilities["triton_cuda"]["weight_quant_optimized"] is True
    assert capabilities["triton_cuda"]["adaln_quant_optimized"] is True
    assert capabilities["triton_cuda"]["adaln_dequant_optimized"] is True
    assert capabilities["triton_cuda"]["full_fusion"] is False


def test_backend_capabilities_report_mps_metal_partial_kernel(monkeypatch):
    monkeypatch.setattr(dispatch_module, "_mps_metal_available", lambda: True)

    capabilities = backend_capabilities(
        backends={"cpu": True, "mps": True, "triton_cuda": False}
    )

    assert capabilities["mps"]["available"] is True
    assert capabilities["mps"]["claim_status"] == "partial_optimized"
    assert capabilities["mps"]["optimized"] is True
    assert capabilities["mps"]["implementation"] == "metal_codebook_rescale"
    assert capabilities["mps"]["optimized_stage"] == "codebook_lookup_rescale,packed_weight_dequant"
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
    assert capabilities["triton_cuda"]["weight_dequant_optimized"] is False
    assert capabilities["triton_cuda"]["weight_pack_optimized"] is False
    assert capabilities["triton_cuda"]["lowbit_unpack_optimized"] is False
    assert capabilities["triton_cuda"]["weight_quant_optimized"] is False
    assert capabilities["triton_cuda"]["adaln_quant_optimized"] is False
    assert capabilities["triton_cuda"]["adaln_dequant_optimized"] is False


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

    def fake_triton_backend(input_tensor, *, rotation, codebook, eps, constant_tensors=None):
        calls.append(input_tensor.shape)
        assert constant_tensors is None
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


def test_cuda_quantize_linear_modules_keeps_packed_buffers_on_gpu_until_serialization():
    if not torch.cuda.is_available() or not available_backends()["triton_cuda"]:
        pytest.skip("CUDA/Triton backend is not available")

    torch.manual_seed(3)
    model = torch.nn.Module()
    model.transformer_blocks = torch.nn.ModuleList(
        [
            torch.nn.ModuleDict(
                {"attn": torch.nn.ModuleDict({"to_q": torch.nn.Linear(32, 32)})}
            )
        ]
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
