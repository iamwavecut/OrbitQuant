import gc
import json
import weakref

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
    x = torch.randn(2, 3, 16)
    assert torch.isfinite(loaded(x)).all()


def test_transformers_pretrained_streams_full_precision_weight_into_packed_tensors(
    tmp_path,
    monkeypatch,
):
    register_hf_quantizers()
    hidden_size = 128
    model = TinyTransformersModel(TinyTransformersConfig(hidden_size=hidden_size))
    config = OrbitQuantConfig(block_size=8, weight_row_tile_size=2)
    expected = OrbitQuantLinear.from_linear(
        model.transformer_blocks[0]["attn"]["to_q"],
        config=config,
        module_name="transformer_blocks.0.attn.to_q",
    )
    model.save_pretrained(tmp_path)

    def fail_post_load_quantization(*args, **kwargs):
        raise AssertionError("Transformers load fell back to post-load quantization")

    monkeypatch.setattr(
        "orbitquant.quantizer.quantize_linear_modules",
        fail_post_load_quantization,
    )

    loaded, loading_info = TinyTransformersModel.from_pretrained(
        tmp_path,
        quantization_config=config,
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
    assert torch.equal(quantized.packed_weight_indices, expected.packed_weight_indices)
    assert torch.equal(quantized.row_norms, expected.row_norms)
    assert modulation.packed_weight is not None
    assert modulation.scales is not None
    assert loaded.hf_quantizer.released_source_tensor_bytes > 0
    assert loaded.hf_quantizer.source_page_release_failures == 0
    x = torch.randn(2, 3, hidden_size)
    assert torch.isfinite(loaded(x)).all()


def test_transformers_pretrained_streams_sharded_safetensors(tmp_path, monkeypatch):
    register_hf_quantizers()
    model = TinyTransformersModel(TinyTransformersConfig())
    model.save_pretrained(tmp_path, max_shard_size="1KB")
    assert (tmp_path / "model.safetensors.index.json").is_file()

    def fail_post_load_quantization(*args, **kwargs):
        raise AssertionError("Transformers load fell back to post-load quantization")

    monkeypatch.setattr(
        "orbitquant.quantizer.quantize_linear_modules",
        fail_post_load_quantization,
    )
    source_refs = []
    original_from_weight = OrbitQuantLinear.from_weight.__func__

    def recording_from_weight(cls, weight, **kwargs):
        source_refs.append(weakref.ref(weight))
        return original_from_weight(cls, weight, **kwargs)

    monkeypatch.setattr(
        OrbitQuantLinear,
        "from_weight",
        classmethod(recording_from_weight),
    )

    loaded, loading_info = TinyTransformersModel.from_pretrained(
        tmp_path,
        quantization_config=OrbitQuantConfig(block_size=8, weight_row_tile_size=2),
        output_loading_info=True,
    )

    assert not loading_info["missing_keys"]
    assert not loading_info["unexpected_keys"]
    assert isinstance(loaded.transformer_blocks[0]["attn"]["to_q"], OrbitQuantLinear)
    gc.collect()
    assert source_refs
    assert all(ref() is None for ref in source_refs)


def test_transformers_streaming_respects_auto_device_map_and_max_memory(tmp_path):
    register_hf_quantizers()
    TinyTransformersModel(TinyTransformersConfig()).save_pretrained(tmp_path)

    loaded = TinyTransformersModel.from_pretrained(
        tmp_path,
        quantization_config=OrbitQuantConfig(block_size=8),
        device_map="auto",
        max_memory={"cpu": "64MB"},
    )

    quantized = loaded.transformer_blocks[0]["attn"]["to_q"]
    assert quantized.packed_weight_indices.device.type == "cpu"
    assert quantized.row_norms.device.type == "cpu"
    assert loaded.proj_out.weight.device.type == "cpu"


def test_transformers_streaming_respects_disk_offload_for_packed_state(tmp_path):
    register_hf_quantizers()
    source_dir = tmp_path / "source"
    offload_dir = tmp_path / "offload"
    TinyTransformersModel(TinyTransformersConfig()).save_pretrained(source_dir)

    loaded = TinyTransformersModel.from_pretrained(
        source_dir,
        quantization_config=OrbitQuantConfig(
            block_size=8,
            runtime_mode="debug_no_activation_quant",
            activation_kernel_backend="cpu",
        ),
        device_map={"transformer_blocks": "disk", "proj_out": "cpu"},
        offload_folder=offload_dir,
    )
    quantized = loaded.transformer_blocks[0]["attn"]["to_q"]

    assert quantized.packed_weight_indices.device.type == "meta"
    assert quantized.row_norms.device.type == "meta"
    assert hasattr(quantized, "_hf_hook")
    assert torch.isfinite(loaded(torch.randn(2, 3, 16))).all()
    assert quantized.packed_weight_indices.device.type == "meta"
    assert quantized._dequantized_weight_cache is None


def test_transformers_on_the_fly_bounded_mode_rejects_pickle_checkpoint(tmp_path):
    register_hf_quantizers()
    model = TinyTransformersModel(TinyTransformersConfig())
    model.save_pretrained(tmp_path)
    torch.save(model.state_dict(), tmp_path / "pytorch_model.bin")
    (tmp_path / "model.safetensors").unlink()

    with pytest.raises(RuntimeError, match="requires safetensors checkpoints"):
        TinyTransformersModel.from_pretrained(
            tmp_path,
            quantization_config=OrbitQuantConfig(block_size=8),
            use_safetensors=False,
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
