import torch

from orbitquant.rotations import RPBHRotation, clear_rotation_cache, get_rpbh_rotation


def test_rpbh_preserves_norms_and_is_deterministic():
    x = torch.randn(3, 5, 16)

    rotation_a = RPBHRotation(dim=16, seed=123, block_size=4)
    rotation_b = RPBHRotation(dim=16, seed=123, block_size=4)

    y_a = rotation_a.apply_to_activations(x)
    y_b = rotation_b.apply_to_activations(x)

    assert torch.allclose(y_a, y_b)
    assert torch.allclose(y_a.norm(dim=-1), x.norm(dim=-1), atol=1e-5, rtol=1e-5)


def test_rpbh_permutation_spreads_block_local_mass_before_hadamard():
    rotation = RPBHRotation(dim=32, seed=0, block_size=4)
    x = torch.zeros(32)
    x[:4] = 1

    y = rotation.apply_to_activations(x)
    block_energy = y.reshape(rotation.num_blocks, rotation.block_size).square().sum(dim=-1)

    assert not torch.equal(rotation.permutation, torch.arange(rotation.dim))
    assert (block_energy > 1e-6).sum().item() > 1


def test_rpbh_matches_paper_order_permute_sign_block_hadamard_normalize():
    rotation = RPBHRotation(dim=8, seed=5, block_size=4)
    x = torch.arange(16, dtype=torch.float32).reshape(2, 8) - 3
    hadamard4 = torch.tensor(
        [
            [1, 1, 1, 1],
            [1, -1, 1, -1],
            [1, 1, -1, -1],
            [1, -1, -1, 1],
        ],
        dtype=torch.float32,
    )

    permuted = x.index_select(dim=-1, index=rotation.permutation)
    signed = permuted * rotation.signs.to(dtype=torch.float32)
    blocks = signed.reshape(2, rotation.num_blocks, rotation.block_size)
    expected = torch.matmul(blocks, hadamard4.T) * rotation.normalization
    expected = expected.reshape_as(x)

    assert torch.allclose(rotation.apply_to_activations(x), expected)


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


def test_rpbh_inverse_restores_rotated_activations():
    x = torch.randn(3, 5, 16)
    rotation = RPBHRotation(dim=16, seed=19, block_size=8)

    restored = rotation.apply_inverse_to_activations(rotation.apply_to_activations(x))

    assert torch.allclose(restored, x, atol=1e-5, rtol=1e-5)


def test_rpbh_paper_block_size_uses_largest_power_of_two_divisor():
    rotation = RPBHRotation(dim=24, seed=0, block_size="paper")

    assert rotation.block_size == 8
    assert rotation.num_blocks == 3


def test_rpbh_rotation_cache_reuses_dimension_seed_block_instances():
    clear_rotation_cache()

    first = get_rpbh_rotation(dim=32, seed=17, block_size=8)
    second = get_rpbh_rotation(dim=32, seed=17, block_size=8)
    other_seed = get_rpbh_rotation(dim=32, seed=18, block_size=8)

    assert first is second
    assert first is not other_seed
