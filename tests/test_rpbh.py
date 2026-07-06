import torch

from orbitquant.rotations import RPBHRotation


def test_rpbh_preserves_norms_and_is_deterministic():
    x = torch.randn(3, 5, 16)

    rotation_a = RPBHRotation(dim=16, seed=123, block_size=4)
    rotation_b = RPBHRotation(dim=16, seed=123, block_size=4)

    y_a = rotation_a.apply_to_activations(x)
    y_b = rotation_b.apply_to_activations(x)

    assert torch.allclose(y_a, y_b)
    assert torch.allclose(y_a.norm(dim=-1), x.norm(dim=-1), atol=1e-5, rtol=1e-5)


def test_rpbh_folds_weight_with_activation_rotation_identity():
    x = torch.randn(4, 16)
    weight = torch.randn(7, 16)
    bias = torch.randn(7)

    rotation = RPBHRotation(dim=16, seed=7, block_size=8)

    baseline = torch.nn.functional.linear(x, weight, bias)
    rotated_x = rotation.apply_to_activations(x)
    rotated_weight = rotation.apply_to_weight(weight)
    folded = torch.nn.functional.linear(rotated_x, rotated_weight, bias)

    assert torch.allclose(folded, baseline, atol=1e-5, rtol=1e-5)


def test_rpbh_paper_block_size_uses_largest_power_of_two_divisor():
    rotation = RPBHRotation(dim=24, seed=0, block_size="paper")

    assert rotation.block_size == 8
    assert rotation.num_blocks == 3
