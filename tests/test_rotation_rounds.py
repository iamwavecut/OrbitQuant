import pytest
import torch

from orbitquant.artifacts import (
    load_orbitquant_artifact,
    save_orbitquant_artifact,
    validate_orbitquant_artifact,
)
from orbitquant.config import OrbitQuantConfig
from orbitquant.gptq import HessianCollector, load_rounded_codes, round_groups
from orbitquant.layers import OrbitQuantLinear
from orbitquant.modeling import (
    dequantize_quantized_linear_modules,
    quantize_linear_modules,
    quantize_model,
)
from orbitquant.rotations import get_pre_rotation, resolve_rotation_rounds
from orbitquant.rotations.rpbh import PRE_ROTATION_SEED_OFFSET


def test_auto_rounds_add_a_first_round_only_for_narrow_paper_blocks():
    assert resolve_rotation_rounds(3360, rotation_rounds="auto") == 2  # h = 32
    assert resolve_rotation_rounds(3840, rotation_rounds="auto") == 2  # h = 256
    assert resolve_rotation_rounds(3072, rotation_rounds="auto") == 1  # h = 1024
    assert resolve_rotation_rounds(256, rotation_rounds="auto") == 1  # h = d
    assert resolve_rotation_rounds(3072, rotation_rounds=2) == 2
    assert get_pre_rotation(3072, rotation_rounds=1) is None
    pre = get_pre_rotation(3360, seed=5, rotation_rounds="auto")
    assert pre.seed == 5 + PRE_ROTATION_SEED_OFFSET
    assert pre.block_size == 32


def test_config_rejects_unknown_rotation_rounds():
    with pytest.raises(ValueError, match="rotation_rounds"):
        OrbitQuantConfig(rotation_rounds=3)
    with pytest.raises(ValueError, match="rotation_rounds"):
        OrbitQuantConfig(rotation_rounds=True)


def test_two_round_rotation_cancels_inside_the_layer():
    torch.manual_seed(0)
    linear = torch.nn.Linear(96, 24)
    x = torch.randn(5, 96)
    config = OrbitQuantConfig(rotation_rounds=2, runtime_mode="debug_no_quant")

    layer = OrbitQuantLinear.from_linear(linear, config=config, module_name="proj")

    assert layer.pre_rotation is not None
    torch.testing.assert_close(layer(x), linear(x), rtol=1e-4, atol=1e-4)


def _outlier_tokens(dim: int, tokens: int = 64) -> torch.Tensor:
    generator = torch.Generator().manual_seed(3)
    x = torch.randn(tokens, dim, generator=generator)
    # Eight neighbouring channels at 20x: one RPBH block of 32 holds them all.
    x[:, 64:72] *= 20.0
    return x


def test_two_rounds_cut_a4_error_for_clustered_outliers_at_narrow_blocks():
    torch.manual_seed(0)
    dim = 3360
    linear = torch.nn.Linear(dim, 48, bias=False)
    x = _outlier_tokens(dim)
    linear.requires_grad_(False)
    exact = linear(x)

    def relative_error(rounds):
        config = OrbitQuantConfig(
            weight_bits=8,
            activation_bits=4,
            rotation_rounds=rounds,
            runtime_mode="dequant_bf16",
        )
        layer = OrbitQuantLinear.from_linear(linear, config=config, module_name="proj")
        error = (layer(x) - exact).pow(2).sum() / exact.pow(2).sum()
        return float(error)

    one_round = relative_error(1)
    two_rounds = relative_error("auto")

    assert two_rounds < 0.5 * one_round


def test_dequantized_export_returns_to_the_source_basis_with_two_rounds():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(96, 32))
    weight = model[0].weight.detach().clone()
    config = OrbitQuantConfig(
        weight_bits=8, rotation_rounds=2, target_policy="universal", runtime_mode="dequant_bf16"
    )
    quantize_model(model, config, quantization_device=None)

    dequantize_quantized_linear_modules(model)

    error = (model[0].weight - weight).pow(2).sum() / weight.pow(2).sum()
    assert float(error) < 1e-3


class _Block(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = torch.nn.Linear(dim, dim, bias=False)
        self.o = torch.nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.o(torch.tanh(self.q(x)))


def test_gptq_rounds_in_the_two_round_basis():
    torch.manual_seed(0)
    dim = 64
    block = _Block(dim)
    reference = _Block(dim)
    reference.load_state_dict(block.state_dict())
    config = OrbitQuantConfig(
        block_size=16,
        rotation_rounds=2,
        target_policy="universal",
        runtime_mode="debug_no_activation_quant",
    )
    basis = torch.randn(4, dim)
    calibration = torch.randn(512, 4) @ basis * 3 + 0.1 * torch.randn(512, dim)
    groups = [["q"], ["o"]]
    with HessianCollector(block, groups, config) as hessians:
        block(calibration)
    codes = round_groups(block, groups, hessians, config)
    quantize_model(block, config, quantization_device=None)
    load_rounded_codes(block, codes)

    assert codes["q"].gptq_error < codes["q"].rtn_error
    with torch.no_grad():
        exact = reference.q(calibration)
        error = (block.q(calibration) - exact).pow(2).sum() / exact.pow(2).sum()
    assert float(error) < 0.02


class _ArtifactModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList(
            [torch.nn.ModuleDict({"attn": torch.nn.ModuleDict({"to_q": torch.nn.Linear(96, 32)})})]
        )


def test_two_round_artifact_validates_and_restores_the_same_outputs(tmp_path):
    torch.manual_seed(0)
    source = _ArtifactModel()
    config = OrbitQuantConfig(rotation_rounds="auto", runtime_mode="dequant_bf16")
    summary = quantize_linear_modules(source, config)
    save_orbitquant_artifact(
        source,
        tmp_path,
        config=config,
        source_model_id="example/model",
        source_revision="abc123",
        source_license="apache-2.0",
        summary=summary,
    )

    result = validate_orbitquant_artifact(tmp_path)
    restored = _ArtifactModel()
    manifest = load_orbitquant_artifact(restored, tmp_path)

    assert result["rotation_validation"] == "checked"
    assert manifest.rotation_rounds == "auto"
    x = torch.randn(3, 96)
    layer = restored.transformer_blocks[0]["attn"]["to_q"]
    assert layer.pre_rotation is not None
    torch.testing.assert_close(layer(x), source.transformer_blocks[0]["attn"]["to_q"](x))


def test_fused_runtime_refuses_two_round_rotations():
    pytest.importorskip("triton")
    from orbitquant.fused.groups import PackedGroup, RowSource
    from orbitquant.fused.runtime import FusedRuntime

    with pytest.raises(ValueError, match="one RPBH round"):
        FusedRuntime.from_config(OrbitQuantConfig(rotation_rounds=2))
    holder = torch.nn.Module()
    holder.proj = OrbitQuantLinear.from_linear(
        torch.nn.Linear(96, 32),
        config=OrbitQuantConfig(rotation_rounds=2, runtime_mode="dequant_bf16"),
        module_name="proj",
    )
    with pytest.raises(ValueError, match="one RPBH round"):
        PackedGroup.from_sources(holder, [RowSource("proj")])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_two_round_layer_on_cuda_matches_the_cpu_reference():
    torch.manual_seed(0)
    linear = torch.nn.Linear(3360, 256).to(torch.bfloat16)
    x = _outlier_tokens(3360).to(torch.bfloat16)
    reference = OrbitQuantLinear.from_linear(
        linear,
        config=OrbitQuantConfig(rotation_rounds="auto", runtime_mode="dequant_bf16"),
        module_name="proj",
    )
    layer = OrbitQuantLinear.from_linear(
        linear, config=OrbitQuantConfig(rotation_rounds="auto"), module_name="proj"
    ).to("cuda")

    with torch.no_grad():
        expected = reference(x.float())
        output = layer(x.to("cuda")).float().cpu()

    error = (output - expected).pow(2).sum() / expected.pow(2).sum()
    assert float(error) < 5e-3
