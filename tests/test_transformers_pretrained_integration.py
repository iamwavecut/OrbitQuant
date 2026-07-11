import json

import pytest
import torch

from orbitquant.adaln import RTNInt4Linear
from orbitquant.config import OrbitQuantConfig
from orbitquant.layers import OrbitQuantLinear
from orbitquant.quantizer import register_hf_quantizers

transformers = pytest.importorskip("transformers")


class TinyTransformersConfig(transformers.PretrainedConfig):
    model_type = "orbitquant-tiny"

    def __init__(self, hidden_size: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size


class TinyTransformersModel(transformers.PreTrainedModel):
    config_class = TinyTransformersConfig
    base_model_prefix = ""

    def __init__(self, config: TinyTransformersConfig):
        super().__init__(config)
        self.transformer_blocks = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {
                        "attn": torch.nn.ModuleDict(
                            {
                                "to_q": torch.nn.Linear(
                                    config.hidden_size, config.hidden_size
                                )
                            }
                        ),
                        "modulation": torch.nn.Linear(config.hidden_size, config.hidden_size),
                    }
                )
            ]
        )
        self.proj_out = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.post_init()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.transformer_blocks[0]["attn"]["to_q"](x)
        return self.proj_out(hidden)


def test_transformers_pretrained_from_pretrained_quantizes_on_load(tmp_path):
    register_hf_quantizers()
    model = TinyTransformersModel(TinyTransformersConfig())
    model.save_pretrained(tmp_path)

    loaded = TinyTransformersModel.from_pretrained(
        tmp_path,
        quantization_config=OrbitQuantConfig(block_size=8),
    )

    assert isinstance(loaded.transformer_blocks[0]["attn"]["to_q"], OrbitQuantLinear)
    assert isinstance(loaded.transformer_blocks[0]["modulation"], RTNInt4Linear)
    assert isinstance(loaded.proj_out, torch.nn.Linear)
    quantized = loaded.transformer_blocks[0]["attn"]["to_q"]
    assert not quantized._derived_constants_valid
    x = torch.randn(2, 3, 16)
    assert torch.isfinite(loaded(x)).all()
    assert quantized._derived_constants_valid
    assert torch.equal(
        torch.unique(quantized._rotation_signs),
        torch.tensor([-1, 1], dtype=torch.int8),
    )


def test_transformers_pretrained_streams_full_precision_weight_into_packed_tensors(
    tmp_path,
    monkeypatch,
):
    register_hf_quantizers()
    model = TinyTransformersModel(TinyTransformersConfig())
    model.save_pretrained(tmp_path)

    def fail_post_load_quantization(*args, **kwargs):
        raise AssertionError("Transformers load fell back to post-load quantization")

    monkeypatch.setattr(
        "orbitquant.quantizer.quantize_linear_modules",
        fail_post_load_quantization,
    )

    loaded, loading_info = TinyTransformersModel.from_pretrained(
        tmp_path,
        quantization_config=OrbitQuantConfig(block_size=8),
        output_loading_info=True,
    )

    quantized = loaded.transformer_blocks[0]["attn"]["to_q"]
    modulation = loaded.transformer_blocks[0]["modulation"]
    assert not loading_info["missing_keys"]
    assert not loading_info["unexpected_keys"]
    assert isinstance(quantized, OrbitQuantLinear)
    assert isinstance(modulation, RTNInt4Linear)
    assert quantized.packed_weight_indices is not None
    assert quantized.row_norms is not None
    assert modulation.packed_weight is not None
    assert modulation.scales is not None
    assert not quantized._derived_constants_valid
    x = torch.randn(2, 3, 16)
    assert torch.isfinite(loaded(x)).all()
    assert quantized._derived_constants_valid
    assert torch.equal(
        torch.unique(quantized._rotation_signs),
        torch.tensor([-1, 1], dtype=torch.int8),
    )


def test_transformers_pretrained_save_pretrained_round_trips_pre_quantized_model(
    tmp_path,
    monkeypatch,
):
    register_hf_quantizers()
    source_dir = tmp_path / "source"
    quantized_dir = tmp_path / "quantized"
    model = TinyTransformersModel(TinyTransformersConfig())
    model.save_pretrained(source_dir)
    quantization_config = OrbitQuantConfig(
        block_size=8,
        target_policy="generic_dit",
        runtime_mode="debug_no_activation_quant",
        activation_kernel_backend="cpu",
        activation_eps=1e-8,
    )

    quantized = TinyTransformersModel.from_pretrained(
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

    restored = TinyTransformersModel.from_pretrained(quantized_dir)

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
    assert isinstance(restored.transformer_blocks[0]["modulation"], RTNInt4Linear)
    assert isinstance(restored.proj_out, torch.nn.Linear)
    x = torch.randn(2, 3, 16)
    assert torch.isfinite(restored(x)).all()
