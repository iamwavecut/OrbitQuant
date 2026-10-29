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
from orbitquant.outliers import find_outlier_channels


def _massive_tokens(dim=256, tokens=64, channel=17, factor=300.0):
    generator = torch.Generator().manual_seed(4)
    x = torch.randn(tokens, dim, generator=generator)
    x[:, channel] *= factor
    return x


def test_config_normalizes_and_bounds_outlier_channels():
    config = OrbitQuantConfig(outlier_channels={"proj": [9, 3, 9]})

    assert config.outlier_channels == {"proj": [3, 9]}
    with pytest.raises(ValueError, match="outlier_channels"):
        OrbitQuantConfig(outlier_channels={"proj": []})
    with pytest.raises(ValueError, match="outlier_channels"):
        OrbitQuantConfig(outlier_channels={"proj": [-1]})


def test_outlier_split_keeps_the_exact_product_without_quantization():
    torch.manual_seed(0)
    linear = torch.nn.Linear(256, 32)
    # The side GEMM keeps the columns in BF16, the dtype of DiT checkpoints.
    linear.weight.data = linear.weight.data.bfloat16().float()
    x = _massive_tokens()
    config = OrbitQuantConfig(runtime_mode="debug_no_quant", outlier_channels={"proj": [17]})

    layer = OrbitQuantLinear.from_linear(linear, config=config, module_name="proj")

    with torch.no_grad():
        torch.testing.assert_close(layer(x), linear(x), rtol=1e-3, atol=1e-2)


def test_keeping_a_massive_channel_out_restores_the_quantization_error():
    torch.manual_seed(0)
    linear = torch.nn.Linear(256, 32, bias=False).requires_grad_(False)
    x = _massive_tokens()
    exact = linear(x)
    clean = x.clone()
    clean[:, 17] = 0

    def relative_error(channels, inputs, reference):
        config = OrbitQuantConfig(
            weight_bits=8,
            activation_bits=4,
            runtime_mode="dequant_bf16",
            outlier_channels={"proj": channels} if channels else {},
        )
        layer = OrbitQuantLinear.from_linear(linear, config=config, module_name="proj")
        return float((layer(inputs) - reference).pow(2).sum() / reference.pow(2).sum())

    # Error on the channels other than the massive one, with and without the split.
    rest = exact - torch.outer(x[:, 17], linear.weight[:, 17])
    plain = float(
        (
            OrbitQuantLinear.from_linear(
                linear,
                config=OrbitQuantConfig(
                    weight_bits=8, activation_bits=4, runtime_mode="dequant_bf16"
                ),
                module_name="proj",
            )(x)
            - exact
        ).pow(2).sum()
        / rest.pow(2).sum()
    )
    split = relative_error([17], x, exact) * float(exact.pow(2).sum() / rest.pow(2).sum())
    floor = relative_error(None, clean, linear(clean))

    assert split < 1.5 * floor
    assert plain > 5 * split


def test_find_outlier_channels_reports_the_massive_channel():
    model = torch.nn.Sequential(torch.nn.Linear(256, 64), torch.nn.GELU(), torch.nn.Linear(64, 8))
    x = _massive_tokens()

    found = find_outlier_channels(
        model, lambda: model(x), config=OrbitQuantConfig(target_policy="universal")
    )

    assert found == {"0": [17]}


def test_outlier_artifact_validates_and_restores_the_same_outputs(tmp_path):
    torch.manual_seed(0)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer_blocks = torch.nn.ModuleList(
                [
                    torch.nn.ModuleDict(
                        {"attn": torch.nn.ModuleDict({"to_q": torch.nn.Linear(256, 32)})}
                    )
                ]
            )

    name = "transformer_blocks.0.attn.to_q"
    source = Model()
    config = OrbitQuantConfig(runtime_mode="dequant_bf16", outlier_channels={name: [17, 40]})
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

    validate_orbitquant_artifact(tmp_path)
    restored = Model()
    load_orbitquant_artifact(restored, tmp_path)

    layer = restored.transformer_blocks[0]["attn"]["to_q"]
    assert layer.outlier_channels.tolist() == [17, 40]
    x = _massive_tokens()
    with torch.no_grad():
        torch.testing.assert_close(layer(x), source.transformer_blocks[0]["attn"]["to_q"](x))


def test_dequantized_export_puts_the_outlier_columns_back():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(256, 32))
    weight = model[0].weight.detach().clone()
    config = OrbitQuantConfig(
        weight_bits=8,
        target_policy="universal",
        runtime_mode="dequant_bf16",
        outlier_channels={"0": [17]},
    )
    quantize_model(model, config, quantization_device=None)

    dequantize_quantized_linear_modules(model)

    torch.testing.assert_close(
        model[0].weight[:, 17], weight[:, 17].bfloat16().float(), rtol=0, atol=0
    )
    error = (model[0].weight - weight).pow(2).sum() / weight.pow(2).sum()
    assert float(error) < 1e-3


def test_gptq_rounds_only_the_quantized_columns():
    torch.manual_seed(0)

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q = torch.nn.Linear(256, 64, bias=False)

        def forward(self, x):
            return self.q(x)

    block = Block()
    reference = Block()
    reference.load_state_dict(block.state_dict())
    config = OrbitQuantConfig(
        target_policy="universal",
        runtime_mode="debug_no_activation_quant",
        outlier_channels={"q": [17]},
    )
    x = _massive_tokens(tokens=512)
    with HessianCollector(block, [["q"]], config) as hessians:
        block(x)
    codes = round_groups(block, [["q"]], hessians, config, sequential=True)
    torch.testing.assert_close(block.q.weight[:, 17], reference.q.weight[:, 17])
    quantize_model(block, config, quantization_device=None)
    load_rounded_codes(block, codes)

    with torch.no_grad():
        exact = reference.q(x)
        error = (block.q(x) - exact).pow(2).sum() / exact.pow(2).sum()
    assert float(error) < 0.02


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_outlier_split_on_cuda_matches_the_cpu_reference():
    torch.manual_seed(0)
    linear = torch.nn.Linear(1024, 256).to(torch.bfloat16)
    x = _massive_tokens(dim=1024).to(torch.bfloat16)
    channels = {"proj": [17]}
    reference = OrbitQuantLinear.from_linear(
        linear,
        config=OrbitQuantConfig(runtime_mode="dequant_bf16", outlier_channels=channels),
        module_name="proj",
    )
    layer = OrbitQuantLinear.from_linear(
        linear, config=OrbitQuantConfig(outlier_channels=channels), module_name="proj"
    ).to("cuda")

    with torch.no_grad():
        expected = reference(x.float())
        output = layer(x.to("cuda")).float().cpu()

    error = (output - expected).pow(2).sum() / expected.pow(2).sum()
    assert float(error) < 5e-3
