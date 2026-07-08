from __future__ import annotations

import pytest
import torch
from orbitquant_packed_matmul import matmul_packed_weight


def _pack(values: torch.Tensor, bits: int) -> torch.Tensor:
    flat = values.detach().to(device="cpu", dtype=torch.uint8).flatten()
    packed = torch.zeros((flat.numel() * bits + 7) // 8, dtype=torch.uint8)
    for value_index, value in enumerate(flat.tolist()):
        bit_start = value_index * bits
        byte_index = bit_start // 8
        shift = bit_start % 8
        packed[byte_index] |= (value << shift) & 0xFF
        if shift + bits > 8:
            packed[byte_index + 1] |= value >> (8 - shift)
    return packed


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    pytest.skip("CUDA or MPS is required")


def _mps_device() -> str:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is required")
    return "mps"


@pytest.mark.kernels_ci
@pytest.mark.parametrize("bits", [2, 3, 4, 6])
@pytest.mark.parametrize("in_features", [16, 19])
@pytest.mark.parametrize("with_bias", [False, True])
def test_matmul_packed_weight_matches_dequantized_reference(
    bits: int, in_features: int, with_bias: bool
) -> None:
    device = _device()
    dtype = torch.float16 if device == "mps" else torch.bfloat16
    rows = 9
    out_features = 7
    x = torch.randn(rows, in_features, device=device, dtype=dtype)
    indices = (
        torch.arange(out_features * in_features, dtype=torch.uint8).reshape(
            out_features, in_features
        )
        % (2**bits)
    )
    packed = _pack(indices, bits).to(device)
    row_norms = torch.linspace(0.5, 1.5, out_features, device=device)
    centroids = torch.linspace(-1.0, 1.0, 2**bits, device=device)
    bias = torch.randn(out_features, device=device, dtype=dtype) if with_bias else None

    expected_weight = row_norms.cpu()[:, None] * centroids.cpu()[indices.long()]
    expected_bias = None if bias is None else bias.float().cpu()
    expected = torch.nn.functional.linear(x.float().cpu(), expected_weight, expected_bias)
    actual = matmul_packed_weight(
        x,
        packed,
        row_norms,
        centroids,
        bits=bits,
        out_features=out_features,
        in_features=in_features,
        bias=bias,
        block_m=16,
        block_n=16,
        block_k=32,
    )

    assert actual.device.type == device
    assert actual.dtype == x.dtype
    assert actual.shape == (rows, out_features)
    assert torch.allclose(actual.float().cpu(), expected, atol=2e-2, rtol=2e-2)


@pytest.mark.kernels_ci
def test_matmul_packed_weight_explicit_mps_path_matches_dequantized_reference() -> None:
    device = _mps_device()
    bits = 4
    rows = 5
    in_features = 19
    out_features = 7
    x = torch.randn(rows, in_features, device=device, dtype=torch.float16)
    indices = (
        torch.arange(out_features * in_features, dtype=torch.uint8).reshape(
            out_features, in_features
        )
        % (2**bits)
    )
    packed = _pack(indices, bits).to(device)
    row_norms = torch.linspace(0.5, 1.5, out_features, device=device)
    centroids = torch.linspace(-1.0, 1.0, 2**bits, device=device)
    bias = torch.randn(out_features, device=device, dtype=torch.float16)

    expected_weight = row_norms.cpu()[:, None] * centroids.cpu()[indices.long()]
    expected = torch.nn.functional.linear(x.float().cpu(), expected_weight, bias.float().cpu())
    actual = matmul_packed_weight(
        x,
        packed,
        row_norms,
        centroids,
        bits=bits,
        out_features=out_features,
        in_features=in_features,
        bias=bias,
        block_m=16,
        block_n=16,
        block_k=32,
    )

    assert actual.device.type == "mps"
    assert actual.dtype == torch.float16
    assert actual.shape == (rows, out_features)
    assert torch.allclose(actual.float().cpu(), expected, atol=2e-2, rtol=2e-2)
