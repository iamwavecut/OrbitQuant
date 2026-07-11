import math

import pytest
import torch
from torch.nn import functional as F

from orbitquant.adaln import RTNInt4Linear
from orbitquant.codebooks import get_codebook
from orbitquant.config import OrbitQuantConfig
from orbitquant.layers import OrbitQuantLinear
from orbitquant.rotations import RPBHRotation


def _sylvester_hadamard(size: int) -> torch.Tensor:
    matrix = torch.ones((1, 1), dtype=torch.float32)
    while matrix.shape[0] < size:
        matrix = torch.cat(
            (
                torch.cat((matrix, matrix), dim=1),
                torch.cat((matrix, -matrix), dim=1),
            ),
            dim=0,
        )
    if matrix.shape != (size, size):
        raise AssertionError(f"{size} is not a power of two")
    return matrix / math.sqrt(size)


def _dense_paper_row_rotation(rotation: RPBHRotation) -> torch.Tensor:
    """Build Pi.T from Equation 9 without the production FWHT implementation."""

    dim = rotation.dim
    permutation_gather = torch.zeros((dim, dim), dtype=torch.float32)
    permutation_gather[
        rotation.permutation,
        torch.arange(dim, dtype=torch.long),
    ] = 1
    signs = torch.diag(rotation.signs.to(torch.float32))
    block = _sylvester_hadamard(rotation.block_size)
    block_hadamard = torch.block_diag(*([block] * rotation.num_blocks))
    return permutation_gather @ signs @ block_hadamard


def _nearest_centroid(
    values: torch.Tensor,
    centroids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    distances = (values.unsqueeze(-1) - centroids).abs()
    indices = distances.argmin(dim=-1)
    return indices, centroids[indices]


def _unpack_little_endian_indices(
    packed: torch.Tensor,
    *,
    bits: int,
    count: int,
) -> torch.Tensor:
    """Decode the artifact bitstream without orbitquant.packing helpers."""

    payload = packed.detach().cpu().to(torch.uint8).tolist()
    mask = (1 << bits) - 1
    decoded: list[int] = []
    for index in range(count):
        bit_offset = index * bits
        byte_offset = bit_offset // 8
        shift = bit_offset % 8
        word = payload[byte_offset]
        if shift + bits > 8:
            word |= payload[byte_offset + 1] << 8
        decoded.append((word >> shift) & mask)
    return torch.tensor(decoded, dtype=torch.long)


def _beta_interval_mean(dim: int, left: float, right: float) -> float:
    """Numerically integrate Equation 2 without production beta helpers."""

    grid = torch.linspace(left, right, 20_001, dtype=torch.float64)
    density = (1 - grid.square()).clamp_min(0).pow((dim - 3) / 2)
    mass = torch.trapezoid(density, grid)
    moment = torch.trapezoid(grid * density, grid)
    return float(moment / mass)


def test_rpbh_matches_independent_dense_equation_9_oracle():
    torch.manual_seed(101)
    rotation = RPBHRotation(dim=24, seed=17, block_size="paper")
    row_rotation = _dense_paper_row_rotation(rotation)
    activations = torch.randn(2, 3, 24)
    weight = torch.randn(7, 24)

    expected_activations = activations @ row_rotation
    expected_weight = weight @ row_rotation

    torch.testing.assert_close(
        rotation.apply_to_activations(activations),
        expected_activations,
        atol=2e-6,
        rtol=2e-6,
    )
    torch.testing.assert_close(
        rotation.apply_to_weight(weight),
        expected_weight,
        atol=2e-6,
        rtol=2e-6,
    )
    torch.testing.assert_close(
        row_rotation @ row_rotation.T,
        torch.eye(rotation.dim),
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        F.linear(expected_activations, expected_weight),
        F.linear(activations, weight),
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.parametrize(
    ("weight_bits", "activation_bits", "with_bias"),
    [
        (4, 4, True),
        (3, 3, False),
        (2, 4, True),
        (2, 3, False),
        (4, 6, True),
    ],
)
def test_reference_runtime_matches_independent_algorithm_1_oracle(
    weight_bits: int,
    activation_bits: int,
    with_bias: bool,
):
    torch.manual_seed(202)
    source = torch.nn.Linear(24, 5, bias=with_bias)
    with torch.no_grad():
        source.weight[1].zero_()
    activations = torch.randn(2, 2, 24)
    activations[0, 0].zero_()
    config = OrbitQuantConfig(
        weight_bits=weight_bits,
        activation_bits=activation_bits,
        rotation_seed=29,
        block_size="paper",
        runtime_mode="dequant_bf16",
        activation_kernel_backend="cpu",
    )
    layer = OrbitQuantLinear.from_linear(
        source,
        config=config,
        module_name="transformer_blocks.0.attn.to_q",
    )

    row_rotation = _dense_paper_row_rotation(layer.rotation)
    weight = source.weight.detach().to(torch.float32)
    rotated_weight = weight @ row_rotation
    raw_row_norms = rotated_weight.norm(dim=-1)
    safe_row_norms = raw_row_norms.clamp_min(config.activation_eps)
    weight_directions = rotated_weight / safe_row_norms[:, None]
    weight_centroids = layer.weight_codebook.centroids.to(torch.float32)
    expected_weight_indices, quantized_weight_directions = _nearest_centroid(
        weight_directions,
        weight_centroids,
    )
    actual_weight_indices = _unpack_little_endian_indices(
        layer.packed_weight_indices,
        bits=weight_bits,
        count=source.out_features * source.in_features,
    ).reshape_as(expected_weight_indices)
    stored_row_norms = raw_row_norms.to(torch.bfloat16)
    quantized_weight = stored_row_norms.to(torch.float32)[:, None] * quantized_weight_directions

    rotated_activations = activations.to(torch.float32) @ row_rotation
    token_norms = rotated_activations.norm(dim=-1, keepdim=True)
    activation_directions = rotated_activations / (token_norms + config.activation_eps)
    _, quantized_activation_directions = _nearest_centroid(
        activation_directions,
        layer.activation_codebook.centroids.to(torch.float32),
    )
    quantized_activations = token_norms * quantized_activation_directions
    bias = None if source.bias is None else source.bias.detach().to(torch.float32)
    expected = F.linear(quantized_activations, quantized_weight, bias)

    actual = layer(activations)

    assert torch.equal(actual_weight_indices, expected_weight_indices)
    assert torch.equal(layer.row_norms, stored_row_norms)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    expected_zero_token = torch.zeros(source.out_features) if bias is None else bias
    torch.testing.assert_close(actual[0, 0], expected_zero_token, atol=0, rtol=0)


@pytest.mark.parametrize("bits", [2, 3, 4, 6])
def test_lloyd_max_v2_satisfies_independent_beta_centroid_condition(bits: int):
    dim = 64
    codebook = get_codebook(dim=dim, bits=bits, algorithm_version=2)
    positive = codebook.centroids[codebook.centroids.numel() // 2 :].to(torch.float64)
    edges = torch.cat(
        (
            torch.zeros(1, dtype=torch.float64),
            (positive[:-1] + positive[1:]) / 2,
            torch.ones(1, dtype=torch.float64),
        )
    )
    numerical_means = torch.tensor(
        [
            _beta_interval_mean(dim, float(edges[index]), float(edges[index + 1]))
            for index in range(positive.numel())
        ],
        dtype=torch.float64,
    )

    torch.testing.assert_close(positive, numerical_means, atol=2e-6, rtol=0)
    torch.testing.assert_close(
        codebook.boundaries,
        (codebook.centroids[:-1] + codebook.centroids[1:]) / 2,
        atol=2e-8,
        rtol=0,
    )


def test_adaln_group64_rtn_matches_independent_weight_only_contract():
    source = torch.nn.Linear(70, 3, bias=True)
    with torch.no_grad():
        values = torch.linspace(-1.5, 1.3, source.weight.numel())
        source.weight.copy_(values.reshape_as(source.weight))
        source.bias.copy_(torch.tensor([-0.25, 0.0, 0.5]))
    layer = RTNInt4Linear.from_linear(
        source,
        config=OrbitQuantConfig(),
        module_name="transformer_blocks.0.norm1.linear",
    )

    num_groups = math.ceil(source.in_features / 64)
    padded = torch.zeros(source.out_features, num_groups * 64, dtype=torch.float32)
    padded[:, : source.in_features] = source.weight.detach().to(torch.float32)
    grouped = padded.reshape(source.out_features, num_groups, 64)
    scales = grouped.abs().amax(dim=-1).clamp_min(1e-12) / 7
    signed_codes = torch.round(grouped / scales[..., None]).clamp(-8, 7).to(torch.long)
    actual_codes = _unpack_little_endian_indices(
        layer.packed_weight,
        bits=4,
        count=source.out_features * num_groups * 64,
    ).reshape_as(signed_codes)
    stored_scales = scales.to(torch.bfloat16)
    dequantized_weight = (
        signed_codes.to(torch.float32) * stored_scales.to(torch.float32)[..., None]
    ).reshape(source.out_features, num_groups * 64)[:, : source.in_features]
    activations = torch.randn(2, 4, source.in_features, dtype=torch.bfloat16)
    expected = F.linear(
        activations,
        dequantized_weight.to(torch.bfloat16),
        source.bias.detach().to(torch.bfloat16),
    )

    assert layer.group_size == 64
    assert torch.equal(actual_codes, signed_codes + 8)
    assert torch.equal(layer.scales, stored_scales)
    torch.testing.assert_close(layer(activations), expected, atol=0, rtol=0)
