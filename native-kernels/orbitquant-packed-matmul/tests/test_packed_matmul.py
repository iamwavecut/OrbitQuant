from __future__ import annotations

import pytest
import torch
from orbitquant_packed_matmul import (
    matmul_packed_w4a4_int8,
    matmul_packed_weight,
    quantize_activations_int8,
    quantize_activations_packed_w4,
)


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


def _mps_bfloat16_device() -> str:
    device = _mps_device()
    try:
        torch.zeros(1, device=device, dtype=torch.bfloat16)
    except Exception as exc:
        pytest.skip(f"MPS bfloat16 tensors are not supported by this PyTorch runtime: {exc}")
    return device


def _cuda_device() -> str:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    return "cuda"


def _row_norms(device: str, out_features: int) -> torch.Tensor:
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    return torch.linspace(0.5, 1.5, out_features, device=device, dtype=dtype)


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
    indices = torch.arange(out_features * in_features, dtype=torch.uint8).reshape(
        out_features, in_features
    ) % (2**bits)
    packed = _pack(indices, bits).to(device)
    row_norms = _row_norms(device, out_features)
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
@pytest.mark.parametrize("bits", [2, 3, 4, 6])
@pytest.mark.parametrize("rows", [1, 2, 3, 8, 9, 15])
@pytest.mark.parametrize(
    ("in_features", "out_features"),
    [(32, 32), (37, 29)],
)
def test_matmul_packed_weight_short_sequence_matches_reference(
    bits: int,
    rows: int,
    in_features: int,
    out_features: int,
) -> None:
    device = _device()
    dtype = torch.float16 if device == "mps" else torch.bfloat16
    x = torch.randn(rows, in_features, device=device, dtype=dtype)
    indices = torch.randint(0, 2**bits, (out_features, in_features), dtype=torch.uint8)
    packed = _pack(indices, bits).to(device)
    row_norms = _row_norms(device, out_features)
    centroids = torch.linspace(-1.0, 1.0, 2**bits, device=device)
    bias = torch.randn(out_features, device=device, dtype=dtype)

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
    )

    assert actual.shape == (rows, out_features)
    assert torch.allclose(actual.float().cpu(), expected, atol=3e-2, rtol=3e-2)


@pytest.mark.kernels_ci
def test_matmul_packed_weight_explicit_mps_path_matches_dequantized_reference() -> None:
    device = _mps_device()
    bits = 4
    rows = 5
    in_features = 19
    out_features = 7
    x = torch.randn(rows, in_features, device=device, dtype=torch.float16)
    indices = torch.arange(out_features * in_features, dtype=torch.uint8).reshape(
        out_features, in_features
    ) % (2**bits)
    packed = _pack(indices, bits).to(device)
    row_norms = _row_norms(device, out_features)
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


@pytest.mark.kernels_ci
def test_matmul_packed_weight_explicit_mps_bfloat16_path_matches_dequantized_reference() -> None:
    device = _mps_bfloat16_device()
    bits = 4
    rows = 5
    in_features = 19
    out_features = 7
    x = torch.randn(rows, in_features, device=device, dtype=torch.bfloat16)
    indices = torch.arange(out_features * in_features, dtype=torch.uint8).reshape(
        out_features, in_features
    ) % (2**bits)
    packed = _pack(indices, bits).to(device)
    row_norms = _row_norms(device, out_features)
    centroids = torch.linspace(-1.0, 1.0, 2**bits, device=device)
    bias = torch.randn(out_features, device=device, dtype=torch.bfloat16)

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
    assert actual.dtype == torch.bfloat16
    assert actual.shape == (rows, out_features)
    assert torch.allclose(actual.float().cpu(), expected, atol=3e-2, rtol=3e-2)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("bits", [2, 3, 4, 6])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_matmul_packed_weight_mps_aligned_mma_path_matches_dequantized_reference(
    bits: int,
    dtype: torch.dtype,
) -> None:
    device = _mps_bfloat16_device() if dtype == torch.bfloat16 else _mps_device()
    rows = in_features = out_features = 32
    x = torch.randn(rows, in_features, device=device, dtype=dtype)
    indices = torch.arange(out_features * in_features, dtype=torch.uint8).reshape(
        out_features, in_features
    ) % (2**bits)
    packed = _pack(indices, bits).to(device)
    row_norms = _row_norms(device, out_features)
    centroids = torch.linspace(-1.0, 1.0, 2**bits, device=device)
    bias = torch.randn(out_features, device=device, dtype=dtype)

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
    )

    tolerance = 3e-2 if dtype == torch.bfloat16 else 2e-2
    assert actual.dtype == dtype
    assert torch.allclose(actual.float().cpu(), expected, atol=tolerance, rtol=tolerance)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("bits", [2, 3, 4, 6])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_matmul_packed_weight_cuda_mma64_path_matches_dequantized_reference(
    bits: int,
    dtype: torch.dtype,
) -> None:
    device = _cuda_device()
    rows = 65
    in_features = 64
    out_features = 70
    x = torch.randn(rows, in_features, device=device, dtype=dtype)
    indices = torch.arange(out_features * in_features, dtype=torch.uint8).reshape(
        out_features, in_features
    ) % (2**bits)
    packed = _pack(indices, bits).to(device)
    row_norms = _row_norms(device, out_features)
    centroids = torch.linspace(-1.0, 1.0, 2**bits, device=device)
    bias = torch.randn(out_features, device=device, dtype=dtype)

    expected_weight = (row_norms[:, None] * centroids[indices.long().to(device)]).to(dtype)
    expected = torch.nn.functional.linear(x, expected_weight, bias)
    actual = matmul_packed_weight(
        x,
        packed,
        row_norms,
        centroids,
        bits=bits,
        out_features=out_features,
        in_features=in_features,
        bias=bias,
    )

    tolerance = 3e-2 if dtype == torch.bfloat16 else 2e-2
    assert actual.dtype == dtype
    assert torch.allclose(actual.float(), expected.float(), atol=tolerance, rtol=tolerance)


@pytest.mark.kernels_ci
@pytest.mark.parametrize(
    ("rows", "out_features", "tile_m", "tile_n"),
    [
        (128, 128, 128, 128),
        (256, 128, 256, 128),
        (128, 256, 128, 256),
        (130, 258, 128, 128),
    ],
)
@pytest.mark.parametrize("weight_k_major", [False, True])
def test_matmul_packed_w4a4_async_matches_sync_and_float_reference(
    rows: int,
    out_features: int,
    tile_m: int,
    tile_n: int,
    weight_k_major: bool,
) -> None:
    device = _cuda_device()
    in_features = 256
    torch.manual_seed(0)
    activation_indices = torch.randint(0, 16, (rows, in_features), device=device, dtype=torch.uint8)
    weight_indices = torch.randint(
        0, 16, (out_features, in_features), device=device, dtype=torch.uint8
    )
    packed_activations = (
        activation_indices[:, 0::2] | (activation_indices[:, 1::2] << 4)
    ).contiguous()
    row_major_weights = (weight_indices[:, 0::2] | (weight_indices[:, 1::2] << 4)).contiguous()
    packed_weights = row_major_weights.T.contiguous() if weight_k_major else row_major_weights
    codes = torch.tensor(
        [-104, -79, -62, -48, -36, -25, -15, -5, 5, 15, 25, 36, 48, 62, 79, 104],
        device=device,
        dtype=torch.int8,
    )
    token_norms = torch.linspace(0.5, 1.0, rows, device=device)
    row_norms = torch.linspace(0.5, 1.5, out_features, device=device, dtype=torch.bfloat16)
    activation_scale = 0.005
    weight_scale = 0.005

    kwargs = {
        "activation_scale": activation_scale,
        "weight_scale": weight_scale,
        "out_features": out_features,
        "in_features": in_features,
        "tile_m": tile_m,
        "tile_n": tile_n,
    }
    sync = matmul_packed_w4a4_int8(
        packed_activations,
        packed_weights,
        token_norms,
        row_norms,
        codes,
        codes,
        async_packed=False,
        weight_k_major=weight_k_major,
        **kwargs,
    )
    asynchronous = matmul_packed_w4a4_int8(
        packed_activations,
        packed_weights,
        token_norms,
        row_norms,
        codes,
        codes,
        async_packed=True,
        weight_k_major=weight_k_major,
        **kwargs,
    )

    assert torch.equal(asynchronous, sync)

    activation_values = codes[activation_indices.long()].float()
    weight_values = codes[weight_indices.long()].float()
    reference = activation_values @ weight_values.T
    reference *= token_norms[:, None]
    reference *= row_norms.float()[None, :]
    reference *= activation_scale * weight_scale

    assert torch.allclose(asynchronous.float(), reference, atol=0.25, rtol=1e-2)


@pytest.mark.kernels_ci
@pytest.mark.parametrize(("dim", "threads"), [(512, 128), (4096, 256), (16384, 512)])
def test_quantize_activations_packed_w4_matches_torch_reference(
    dim: int,
    threads: int,
) -> None:
    device = _cuda_device()
    rows = 2
    x = torch.zeros((rows, dim), device=device, dtype=torch.bfloat16)
    x[0, 3] = 1
    x[1, dim - 7] = -1
    permutation = torch.randperm(dim, device=device)
    signs = torch.where(
        torch.arange(dim, device=device) % 2 == 0,
        torch.ones(dim, device=device, dtype=torch.int8),
        -torch.ones(dim, device=device, dtype=torch.int8),
    )
    boundaries = torch.linspace(-0.2, 0.2, 15, device=device)
    eps = 1e-12
    inv_sqrt_block = dim**-0.5

    packed, norms = quantize_activations_packed_w4(
        x,
        permutation,
        signs,
        boundaries,
        eps=eps,
        inv_sqrt_block=inv_sqrt_block,
        threads=threads,
    )

    work = x.float()[:, permutation] * signs.float()
    expected_norms = work.norm(dim=-1)
    work /= expected_norms[:, None] + eps
    width = 1
    while width < dim:
        blocks = work.reshape(rows, -1, width * 2)
        left = blocks[..., :width]
        right = blocks[..., width:]
        work = torch.cat((left + right, left - right), dim=-1).reshape(rows, dim)
        width *= 2
    indices = torch.bucketize(work * inv_sqrt_block, boundaries).to(torch.uint8)
    expected_packed = (indices[:, 0::2] | (indices[:, 1::2] << 4)).contiguous()

    assert torch.equal(packed, expected_packed)
    assert torch.allclose(norms, expected_norms, atol=1e-6, rtol=1e-6)


@pytest.mark.kernels_ci
@pytest.mark.parametrize(("dim", "threads"), [(512, 128), (4096, 256), (16384, 512)])
def test_quantize_activations_int8_matches_packed_codes(
    dim: int,
    threads: int,
) -> None:
    device = _cuda_device()
    x = torch.randn((2, dim), device=device, dtype=torch.bfloat16)
    permutation = torch.randperm(dim, device=device)
    signs = torch.where(
        torch.arange(dim, device=device) % 2 == 0,
        torch.ones(dim, device=device, dtype=torch.int8),
        -torch.ones(dim, device=device, dtype=torch.int8),
    )
    boundaries = torch.linspace(-0.2, 0.2, 15, device=device)
    codes = torch.tensor(
        [-120, -92, -68, -49, -34, -22, -12, -4, 4, 12, 22, 34, 49, 68, 92, 120],
        device=device,
        dtype=torch.int8,
    )
    kwargs = {
        "eps": 1e-12,
        "inv_sqrt_block": dim**-0.5,
        "threads": threads,
    }

    packed, packed_norms = quantize_activations_packed_w4(
        x,
        permutation,
        signs,
        boundaries,
        **kwargs,
    )
    quantized, int8_norms = quantize_activations_int8(
        x,
        permutation,
        signs,
        boundaries,
        codes,
        **kwargs,
    )

    indices = torch.empty((2, dim), device=device, dtype=torch.long)
    indices[:, 0::2] = packed & 15
    indices[:, 1::2] = packed >> 4
    expected = codes[indices]

    assert torch.equal(quantized, expected)
    assert torch.equal(int8_norms, packed_norms)


@pytest.mark.kernels_ci
def test_quantize_activations_int8_matches_blocked_rpbh_reference() -> None:
    device = _cuda_device()
    rows = 2
    dim = 12288
    block_size = 4096
    x = torch.randn((rows, dim), device=device, dtype=torch.bfloat16)
    permutation = torch.randperm(dim, device=device)
    signs = torch.where(
        torch.arange(dim, device=device) % 2 == 0,
        torch.ones(dim, device=device, dtype=torch.int8),
        -torch.ones(dim, device=device, dtype=torch.int8),
    )
    boundaries = torch.linspace(-0.2, 0.2, 15, device=device)
    codes = torch.tensor(
        [-120, -92, -68, -49, -34, -22, -12, -4, 4, 12, 22, 34, 49, 68, 92, 120],
        device=device,
        dtype=torch.int8,
    )

    quantized, norms = quantize_activations_int8(
        x,
        permutation,
        signs,
        boundaries,
        codes,
        eps=1e-12,
        inv_sqrt_block=block_size**-0.5,
        threads=512,
    )

    work = x.float()[:, permutation] * signs.float()
    expected_norms = work.norm(dim=-1)
    work = (work / expected_norms[:, None]).reshape(rows, -1, block_size)
    width = 1
    while width < block_size:
        blocks = work.reshape(rows, -1, width * 2)
        left = blocks[..., :width]
        right = blocks[..., width:]
        work = torch.cat((left + right, left - right), dim=-1).reshape(
            rows, -1, block_size
        )
        width *= 2
    indices = torch.bucketize(
        work.reshape(rows, dim) * (block_size**-0.5), boundaries
    )
    expected = codes[indices]

    assert torch.equal(quantized, expected)
    assert torch.allclose(norms, expected_norms, atol=1e-5, rtol=1e-6)
