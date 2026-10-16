import pytest
import torch

from orbitquant.codebooks.lloyd_max import get_codebook
from orbitquant.config import OrbitQuantConfig
from orbitquant.gptq import (
    HessianCollector,
    gptq_round,
    load_rounded_codes,
    round_groups,
    update_state_dict,
)
from orbitquant.layers import OrbitQuantLinear
from orbitquant.modeling import quantize_model
from orbitquant.packing import unpack_lowbit
from orbitquant.rotations.rpbh import get_rpbh_rotation

DIM = 64


def _config():
    return OrbitQuantConfig(
        weight_bits=4,
        activation_bits=4,
        block_size=DIM,
        target_policy="universal",
        runtime_mode="dequant_bf16",
    )


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q = torch.nn.Linear(DIM, DIM, bias=False)
        self.k = torch.nn.Linear(DIM, 2 * DIM, bias=False)
        self.o = torch.nn.Linear(DIM, DIM, bias=False)

    def forward(self, x):
        return self.o(torch.tanh(self.q(x)) + self.k(x)[..., :DIM])


def _calibration(n=512):
    # Correlated inputs: a few strong directions plus noise, as in real activations.
    torch.manual_seed(1)
    basis = torch.randn(4, DIM)
    return torch.randn(n, 4) @ basis * 3 + 0.1 * torch.randn(n, DIM)


def test_gptq_beats_nearest_rounding_on_correlated_inputs():
    torch.manual_seed(0)
    weight = torch.randn(32, DIM)
    rotation = get_rpbh_rotation(dim=DIM, seed=0, block_size=DIM)
    x = rotation.apply_to_activations(_calibration())
    hessian = x.T @ x
    rotated = rotation.apply_to_weight(weight)
    codes, scale = gptq_round(rotated, hessian, bits=4)
    codebook = get_codebook(DIM, 4, 2)
    gptq = codebook.centroids[codes.long()] * scale[:, None]
    nearest = (
        codebook.centroids[codebook.quantize_indices(rotated / scale[:, None]).long()]
        * scale[:, None]
    )

    def error(approx):
        diff = rotated - approx
        return torch.sum((diff @ hessian) * diff) / torch.sum((rotated @ hessian) * rotated)

    assert codes.dtype == torch.uint8 and codes.shape == rotated.shape
    assert torch.equal(scale, rotated.norm(dim=1).to(torch.bfloat16).float())
    assert error(gptq) < 0.5 * error(nearest)


def test_hessian_collector_shares_one_hessian_per_group():
    block = _Block()
    config = _config()
    x = _calibration(64)
    with HessianCollector(block, [["q", "k"], ["o"]], config) as hessians:
        block(x)
    rotation = get_rpbh_rotation(dim=DIM, seed=config.rotation_seed, block_size=config.block_size)
    xr = rotation.apply_to_activations(x)
    torch.testing.assert_close(hessians["q"], xr.T @ xr)
    assert hessians["k"] is hessians["q"]
    assert hessians.tokens["q"] == 64
    assert not block.q._forward_pre_hooks


def test_rounded_codes_load_into_the_quantized_model():
    torch.manual_seed(0)
    block = _Block()
    config = _config()
    groups = [["q", "k"], ["o"]]
    with HessianCollector(block, groups, config) as hessians:
        block(_calibration())
    codes = round_groups(block, groups, hessians, config)
    assert codes["q"].gptq_error < codes["q"].rtn_error
    quantize_model(block, config, quantization_device=None)
    load_rounded_codes(block, codes)
    module = block.k
    assert isinstance(module, OrbitQuantLinear)
    unpacked = unpack_lowbit(module.packed_weight_indices, bits=4, length=2 * DIM * DIM).reshape(
        2 * DIM, DIM
    )
    expected = unpack_lowbit(codes["k"].packed, bits=4, length=2 * DIM * DIM).reshape(2 * DIM, DIM)
    assert torch.equal(unpacked, expected)
    centroids = get_codebook(DIM, 4, 2).centroids
    dequantized = module._dequantize_weight(
        device=torch.device("cpu"), dtype=torch.float32, remember=False
    )
    torch.testing.assert_close(
        dequantized, centroids[expected.long()] * codes["k"].row_norms.float()[:, None]
    )


def test_sequential_rounding_writes_rounded_weights_back():
    torch.manual_seed(0)
    block = _Block()
    config = _config()
    with HessianCollector(block, [["o"]], config) as hessians:
        block(_calibration())
    codes = round_groups(block, [["o"]], hessians, config, sequential=True)
    rotation = get_rpbh_rotation(dim=DIM, seed=config.rotation_seed, block_size=config.block_size)
    indices = unpack_lowbit(codes["o"].packed, bits=4, length=DIM * DIM).reshape(DIM, DIM).long()
    rounded = get_codebook(DIM, 4, 2).centroids[indices] * codes["o"].row_norms.float()[:, None]
    torch.testing.assert_close(block.o.weight.data, rotation.apply_inverse_to_weight(rounded))


def test_update_state_dict_keeps_shapes_and_dtypes():
    torch.manual_seed(0)
    block = _Block()
    config = _config()
    with HessianCollector(block, [["o"]], config) as hessians:
        block(_calibration())
    codes = round_groups(block, [["o"]], hessians, config)
    state = {
        "o.packed_weight_indices": torch.zeros(DIM * DIM // 2, dtype=torch.uint8),
        "o.row_norms": torch.zeros(DIM, dtype=torch.bfloat16),
    }
    update_state_dict(state, codes)
    assert state["o.packed_weight_indices"].shape == (DIM * DIM // 2,)
    assert state["o.row_norms"].dtype == torch.bfloat16
    assert torch.equal(state["o.packed_weight_indices"], codes["o"].packed)


def test_batched_rounding_matches_one_group_at_a_time():
    torch.manual_seed(0)
    rotation = get_rpbh_rotation(dim=DIM, seed=0, block_size=DIM)
    weights, hessians = [], []
    for seed in range(3):
        torch.manual_seed(seed)
        weights.append(rotation.apply_to_weight(torch.randn(16, DIM)))
        x = rotation.apply_to_activations(_calibration(256) + seed)
        hessians.append(x.T @ x)
    codes, scale = gptq_round(torch.stack(weights), torch.stack(hessians), bits=4)
    for g in range(3):
        one_codes, one_scale = gptq_round(weights[g], hessians[g], bits=4)
        assert torch.equal(codes[g], one_codes)
        assert torch.equal(scale[g], one_scale)


def test_round_groups_batches_same_shape_groups():
    torch.manual_seed(0)
    blocks = torch.nn.ModuleList([_Block() for _ in range(3)])
    config = _config()
    groups = [[f"{i}.q", f"{i}.k"] for i in range(3)] + [[f"{i}.o"] for i in range(3)]
    x = _calibration()
    with HessianCollector(blocks, groups, config) as hessians:
        for block in blocks:
            block(x)
    batched = round_groups(blocks, groups, hessians, config, release=False, batch=8)
    single = round_groups(blocks, groups, hessians, config, batch=1)
    for name, codes in batched.items():
        assert torch.equal(codes.packed, single[name].packed)
        assert torch.equal(codes.row_norms, single[name].row_norms)


def test_damping_grows_until_the_hessian_factorizes():
    torch.manual_seed(0)
    rotation = get_rpbh_rotation(dim=DIM, seed=0, block_size=DIM)
    weight = rotation.apply_to_weight(torch.randn(8, DIM))
    hessian = torch.eye(DIM)
    # Four negative eigenvalues of -0.5: positive definite only from about 0.55 relative damping.
    hessian[range(DIM - 4, DIM), range(DIM - 4, DIM)] = -0.5
    codes, scale, damping = gptq_round(weight, hessian, bits=4, return_damping=True)
    assert codes.shape == weight.shape
    assert damping.item() == pytest.approx(1.0)
