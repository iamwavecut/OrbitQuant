import pytest
import torch

from orbitquant.adaln import RTNInt4Linear, _quantize_adaln_weight_reference
from orbitquant.codebooks import get_codebook
from orbitquant.codebooks.lloyd_max import codebook_retained_energy
from orbitquant.config import OrbitQuantConfig
from orbitquant.gptq import HessianCollector, load_rounded_codes, round_groups
from orbitquant.layers import OrbitQuantLinear, output_gain
from orbitquant.modeling import quantize_model
from orbitquant.packing import unpack_lowbit


@pytest.mark.parametrize(("bits", "expected"), [(2, 0.8826), (3, 0.9655), (4, 0.9905)])
def test_retained_energy_is_one_minus_the_lloyd_max_distortion(bits, expected):
    codebook = get_codebook(3072, bits, 2)

    assert codebook_retained_energy(codebook) == pytest.approx(expected, abs=1e-4)


def test_retained_energy_matches_a_monte_carlo_estimate():
    torch.manual_seed(0)
    codebook = get_codebook(256, 3, 2)
    u = torch.randn(4096, 256)
    u = u / u.norm(dim=-1, keepdim=True)
    estimate = float((u * codebook.quantize(u)).sum(dim=-1).mean())

    assert codebook_retained_energy(codebook) == pytest.approx(estimate, abs=2e-3)


def test_gain_correction_scales_row_norms_and_keeps_codes():
    torch.manual_seed(0)
    linear = torch.nn.Linear(128, 16)
    plain = OrbitQuantConfig(weight_bits=2, activation_bits=4, runtime_mode="dequant_bf16")
    corrected = OrbitQuantConfig(
        weight_bits=2,
        activation_bits=4,
        runtime_mode="dequant_bf16",
        output_gain_correction=True,
    )

    base = OrbitQuantLinear.from_linear(linear, config=plain, module_name="proj")
    gained = OrbitQuantLinear.from_linear(linear, config=corrected, module_name="proj")

    gain = 1.0 / (
        codebook_retained_energy(get_codebook(128, 2, 2))
        * codebook_retained_energy(get_codebook(128, 4, 2))
    )
    assert gained.output_gain == pytest.approx(gain)
    assert output_gain(plain, 128) == 1.0
    assert torch.equal(gained.packed_weight_indices, base.packed_weight_indices)
    torch.testing.assert_close(
        gained.row_norms.float(), (base.row_norms.float() * gain).bfloat16().float()
    )


def test_gain_correction_reduces_error_for_inputs_aligned_with_weight_rows():
    torch.manual_seed(0)
    linear = torch.nn.Linear(256, 32, bias=False).requires_grad_(False)
    weight = linear.weight
    x = weight[torch.randint(0, 32, (64,))] * 5 + 0.05 * torch.randn(64, 256)
    exact = linear(x)

    def relative_error(correct):
        config = OrbitQuantConfig(
            weight_bits=2,
            activation_bits=4,
            runtime_mode="dequant_bf16",
            output_gain_correction=correct,
        )
        layer = OrbitQuantLinear.from_linear(linear, config=config, module_name="proj")
        return float((layer(x) - exact).pow(2).sum() / exact.pow(2).sum())

    assert relative_error(True) < relative_error(False)


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q = torch.nn.Linear(64, 64, bias=False)

    def forward(self, x):
        return self.q(x)


def test_gptq_codes_keep_the_gain_correction():
    torch.manual_seed(0)
    block = _Block()
    config = OrbitQuantConfig(
        block_size=64,
        target_policy="universal",
        runtime_mode="dequant_bf16",
        output_gain_correction=True,
    )
    with HessianCollector(block, [["q"]], config) as hessians:
        block(torch.randn(256, 64))
    codes = round_groups(block, [["q"]], hessians, config)
    quantize_model(block, config, quantization_device=None)
    load_rounded_codes(block, codes)

    expected = (codes["q"].row_norms.float() * block.q.output_gain).bfloat16()
    assert torch.equal(block.q.row_norms, expected)


def _adaln_error(packed, scales, weight, group_size=64):
    codes = unpack_lowbit(packed, bits=4, length=weight.numel()).reshape(weight.shape)
    codes = codes.to(torch.float32) - 8
    grouped = codes.reshape(weight.shape[0], -1, group_size)
    restored = grouped * scales.to(torch.float32)[..., None]
    return (restored.reshape(weight.shape) - weight).square().reshape(
        weight.shape[0], -1, group_size
    ).sum(dim=-1)


def test_mse_adaln_scales_never_lose_to_absmax_and_win_on_average():
    torch.manual_seed(0)
    weight = torch.randn(96, 256) * torch.rand(96, 1)
    absmax = RTNInt4Linear.from_weight(
        weight,
        bias=None,
        in_features=256,
        out_features=96,
        source_weight_layout="out_in",
        config=OrbitQuantConfig(),
        module_name="norm1.linear",
    )
    mse = RTNInt4Linear.from_weight(
        weight,
        bias=None,
        in_features=256,
        out_features=96,
        source_weight_layout="out_in",
        config=OrbitQuantConfig(adaln_scale="mse"),
        module_name="norm1.linear",
    )

    absmax_error = _adaln_error(absmax.packed_weight, absmax.scales, weight)
    mse_error = _adaln_error(mse.packed_weight, mse.scales, weight)

    assert mse.packed_weight.shape == absmax.packed_weight.shape
    assert mse.scales.dtype == torch.bfloat16
    assert bool((mse_error <= absmax_error * (1 + 1e-6)).all())
    assert float(mse_error.sum()) < 0.85 * float(absmax_error.sum())


def test_absmax_adaln_default_is_unchanged():
    torch.manual_seed(0)
    weight = torch.randn(8, 128)
    reference_packed, reference_scales = _quantize_adaln_weight_reference(weight, group_size=64)
    layer = RTNInt4Linear.from_weight(
        weight,
        bias=None,
        in_features=128,
        out_features=8,
        source_weight_layout="out_in",
        config=OrbitQuantConfig(),
        module_name="norm1.linear",
    )

    assert torch.equal(layer.packed_weight, reference_packed)
    assert torch.equal(layer.scales, reference_scales.to(torch.bfloat16))


def test_config_rejects_unknown_adaln_scale():
    with pytest.raises(ValueError, match="adaln_scale"):
        OrbitQuantConfig(adaln_scale="percentile")
