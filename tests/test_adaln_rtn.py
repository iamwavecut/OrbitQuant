import pytest
import torch

from orbitquant.adaln import (
    RTNInt4Linear,
    _quantize_adaln_weight_reference,
)
from orbitquant.config import OrbitQuantConfig
from orbitquant.kernels import available_backends


def test_int4_rtn_linear_preserves_shape_and_freezes_parameters():
    torch.manual_seed(0)
    source = torch.nn.Linear(17, 9)
    x = torch.randn(2, 3, 17)
    config = OrbitQuantConfig(adaln_group_size=8)

    quantized = RTNInt4Linear.from_linear(source, config=config, module_name="block.modulation")
    actual = quantized(x)

    assert actual.shape == (2, 3, 9)
    assert torch.isfinite(actual).all()
    assert not any(parameter.requires_grad for parameter in quantized.parameters())
    assert quantized.group_size == 8


def test_int4_rtn_default_matches_paper_group64_and_bf16_activation_path():
    torch.manual_seed(0)
    source = torch.nn.Linear(65, 9)
    x = torch.randn(2, 3, 65, dtype=torch.bfloat16)
    config = OrbitQuantConfig()

    quantized = RTNInt4Linear.from_linear(source, config=config, module_name="block.modulation")
    actual = quantized(x)

    assert quantized.group_size == 64
    assert quantized.scales.dtype == torch.bfloat16
    assert actual.dtype == torch.bfloat16
    assert actual.shape == (2, 3, 9)
    assert torch.isfinite(actual).all()


def test_int4_rtn_rejects_non_positive_group_size():
    source = torch.nn.Linear(16, 8)
    config = OrbitQuantConfig(adaln_group_size=1)

    quantized = RTNInt4Linear.from_linear(source, config=config, module_name="block.modulation")

    assert quantized.packed_weight.dtype == torch.uint8


def test_int4_rtn_cuda_quantize_path_matches_reference():
    if not torch.cuda.is_available() or not available_backends()["triton_cuda"]:
        pytest.skip("CUDA/Triton backend is not available")

    from orbitquant.kernels.triton_cuda import quantize_adaln_weight_with_triton

    torch.manual_seed(1)
    weight = torch.randn(7, 19, device="cuda", dtype=torch.float32)

    expected_packed, expected_scales = _quantize_adaln_weight_reference(
        weight.cpu(), group_size=8
    )
    actual_packed, actual_scales = quantize_adaln_weight_with_triton(weight, group_size=8)

    assert actual_packed.is_cuda
    assert actual_scales.is_cuda
    assert torch.equal(actual_packed.cpu(), expected_packed)
    assert torch.allclose(actual_scales.cpu(), expected_scales)


def test_int4_rtn_cuda_dequant_path_matches_reference():
    if not torch.cuda.is_available() or not available_backends()["triton_cuda"]:
        pytest.skip("CUDA/Triton backend is not available")

    from orbitquant.kernels.triton_cuda import dequantize_adaln_weight_with_triton

    torch.manual_seed(2)
    source = torch.nn.Linear(19, 7).to("cuda")
    config = OrbitQuantConfig(adaln_group_size=8)
    quantized = RTNInt4Linear.from_linear(source, config=config, module_name="block.modulation")

    expected = quantized._dequantize_weight(device=torch.device("cpu"), dtype=torch.float32)
    quantized.clear_dequantized_cache()
    actual = dequantize_adaln_weight_with_triton(
        quantized.packed_weight,
        quantized.scales,
        out_features=quantized.out_features,
        in_features=quantized.in_features,
        group_size=quantized.group_size,
        device="cuda",
    )

    assert actual.is_cuda
    assert torch.allclose(actual.cpu(), expected)
