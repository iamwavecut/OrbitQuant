import json

import pytest
import torch

from orbitquant.config import OrbitQuantConfig
from orbitquant.layers import OrbitQuantLinear
from orbitquant.modeling import quantize_model
from orbitquant.quantizer import register_hf_quantizers

diffusers = pytest.importorskip("diffusers")
configuration_utils = pytest.importorskip("diffusers.configuration_utils")


class TinyDiffusersTransformer(diffusers.ModelMixin, diffusers.ConfigMixin):
    config_name = "config.json"

    @configuration_utils.register_to_config
    def __init__(self, in_features: int = 16, out_features: int = 16):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {
                        "attn": torch.nn.ModuleDict(
                            {"to_q": torch.nn.Linear(in_features, out_features)}
                        )
                    }
                )
            ]
        )
        self.proj_out = torch.nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj_out(self.transformer_blocks[0]["attn"]["to_q"](x))


def test_diffusers_modelmixin_from_pretrained_quantizes_on_load(tmp_path):
    register_hf_quantizers()
    model = TinyDiffusersTransformer()
    model.save_pretrained(tmp_path)

    loaded = TinyDiffusersTransformer.from_pretrained(
        tmp_path,
        quantization_config=OrbitQuantConfig(block_size=8),
    )

    assert isinstance(loaded.transformer_blocks[0]["attn"]["to_q"], OrbitQuantLinear)
    assert isinstance(loaded.proj_out, torch.nn.Linear)
    x = torch.randn(2, 3, 16)
    assert torch.isfinite(loaded(x)).all()


def test_diffusers_modelmixin_save_pretrained_round_trips_pre_quantized_model(
    tmp_path,
    monkeypatch,
):
    register_hf_quantizers()
    source_dir = tmp_path / "source"
    quantized_dir = tmp_path / "quantized"
    model = TinyDiffusersTransformer()
    model.save_pretrained(source_dir)
    quantization_config = OrbitQuantConfig(
        block_size=8,
        target_policy="generic_dit",
        runtime_mode="debug_no_activation_quant",
        activation_kernel_backend="cpu",
        activation_eps=1e-8,
    )

    quantized = TinyDiffusersTransformer.from_pretrained(
        source_dir,
        quantization_config=quantization_config,
    )
    quantized.save_pretrained(quantized_dir)
    saved_config = json.loads((quantized_dir / "config.json").read_text())

    def fail_from_linear(cls, *args, **kwargs):
        raise AssertionError("pre-quantized restore should not requantize Linear weights")

    def fail_post_load_quantization(*args, **kwargs):
        raise AssertionError("pre-quantized restore fell back to post-load quantization")

    monkeypatch.setattr(OrbitQuantLinear, "from_linear", classmethod(fail_from_linear))
    monkeypatch.setattr(
        "orbitquant.quantizer.quantize_linear_modules",
        fail_post_load_quantization,
    )

    restored = TinyDiffusersTransformer.from_pretrained(quantized_dir)

    saved_quantization_config = saved_config["quantization_config"]
    assert saved_quantization_config["quant_method"] == "orbitquant"
    assert saved_quantization_config["runtime_mode"] == "debug_no_activation_quant"
    assert saved_quantization_config["activation_kernel_backend"] == "cpu"
    assert saved_quantization_config["activation_eps"] == 1e-8
    restored_linear = restored.transformer_blocks[0]["attn"]["to_q"]
    assert isinstance(restored_linear, OrbitQuantLinear)
    assert restored_linear.runtime_mode == "debug_no_activation_quant"
    assert restored_linear.activation_kernel_backend == "cpu"
    assert restored_linear.activation_eps == 1e-8
    assert isinstance(restored.proj_out, torch.nn.Linear)
    x = torch.randn(2, 3, 16)
    assert torch.isfinite(restored(x)).all()


def test_diffusers_manual_quantize_model_persists_quantization_config(tmp_path):
    model = TinyDiffusersTransformer()
    config = OrbitQuantConfig(
        block_size=8,
        runtime_mode="dequant_bf16",
        activation_kernel_backend="cpu",
    )

    quantize_model(model, config, quantization_device="cpu")
    model.save_pretrained(tmp_path)
    saved_config = json.loads((tmp_path / "config.json").read_text())
    restored = TinyDiffusersTransformer.from_pretrained(tmp_path)

    assert saved_config["quantization_config"] == config.to_dict()
    assert isinstance(restored.transformer_blocks[0]["attn"]["to_q"], OrbitQuantLinear)
