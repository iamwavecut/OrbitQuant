import pytest
import torch

from orbitquant.adaln import (
    _quantize_adaln_weight_bounded,
    _quantize_adaln_weight_reference,
)
from orbitquant.codebooks import get_codebook
from orbitquant.layers import _quantize_weight_bounded, _quantize_weight_pack
from orbitquant.rotations import get_rpbh_rotation
from orbitquant.streaming import iter_aligned_row_tiles


@pytest.mark.parametrize("bits", [2, 3, 4, 6])
def test_row_tiled_orbitquant_matches_full_matrix_reference_with_padding(bits):
    torch.manual_seed(bits)
    weight = torch.randn(19, 18, dtype=torch.bfloat16)
    rotation = get_rpbh_rotation(18, seed=7, block_size="paper")
    codebook = get_codebook(18, bits, algorithm_version=2)
    reference_weight = weight.to(torch.float32)
    reference_norms = reference_weight.norm(dim=-1)
    expected = _quantize_weight_pack(
        reference_weight,
        reference_norms,
        rotation=rotation,
        codebook=codebook,
        bits=bits,
        eps=1e-10,
    )

    actual, actual_norms = _quantize_weight_bounded(
        weight,
        rotation=rotation,
        codebook=codebook,
        bits=bits,
        eps=1e-10,
        row_tile_size=3,
    )

    assert actual.numel() == (weight.numel() * bits + 7) // 8
    assert torch.equal(actual, expected)
    assert torch.equal(actual_norms, reference_norms.to(torch.bfloat16))


def test_row_tiles_align_non_final_payloads_to_byte_boundaries():
    tiles = list(iter_aligned_row_tiles(19, values_per_row=18, bits=3, max_rows=3))

    assert tiles == [(0, 4), (4, 8), (8, 12), (12, 16), (16, 19)]
    assert all((end - start) * 18 * 3 % 8 == 0 for start, end in tiles[:-1])


def test_row_tiled_adaln_matches_full_matrix_reference_with_group_padding():
    torch.manual_seed(11)
    weight = torch.randn(19, 18, dtype=torch.bfloat16)
    expected_packed, expected_scales = _quantize_adaln_weight_reference(
        weight,
        group_size=7,
    )

    actual_packed, actual_scales = _quantize_adaln_weight_bounded(
        weight,
        group_size=7,
        row_tile_size=3,
    )

    assert torch.equal(actual_packed, expected_packed)
    assert torch.equal(actual_scales, expected_scales.to(torch.bfloat16))
