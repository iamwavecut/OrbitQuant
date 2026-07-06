from collections import OrderedDict

import pytest
import torch

from orbitquant.adaln import RTNInt4Linear
from orbitquant.config import OrbitQuantConfig
from orbitquant.layers import OrbitQuantLinear
from orbitquant.quantizer import OrbitQuantizer, register_hf_quantizers


class TinyQuantizerTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {"attn": torch.nn.ModuleDict({"to_q": torch.nn.Linear(16, 16)})}
                )
            ]
        )
        self.proj_out = torch.nn.Linear(16, 16)


def test_quantizer_adapter_reports_no_calibration_requirement():
    quantizer = OrbitQuantizer(OrbitQuantConfig())

    assert quantizer.requires_parameters_quantization is True
    assert quantizer.requires_calibration is False
    assert quantizer.is_serializable() is True


def test_quantizer_preserves_hf_pre_quantized_constructor_semantics():
    default_quantizer = OrbitQuantizer(OrbitQuantConfig())
    on_the_fly_quantizer = OrbitQuantizer(OrbitQuantConfig(), pre_quantized=False)

    assert default_quantizer.pre_quantized is True
    assert on_the_fly_quantizer.pre_quantized is False


def test_hf_registration_is_best_effort_without_optional_dependencies():
    result = register_hf_quantizers()

    assert set(result) == {"diffusers", "transformers"}


def test_hf_registration_populates_auto_mappings_when_dependencies_are_installed():
    result = register_hf_quantizers()

    if result["diffusers"]:
        import diffusers.quantizers.auto as diffusers_auto

        assert diffusers_auto.AUTO_QUANTIZER_MAPPING["orbitquant"] is OrbitQuantizer
        assert diffusers_auto.AUTO_QUANTIZATION_CONFIG_MAPPING["orbitquant"] is OrbitQuantConfig

    if result["transformers"]:
        import transformers.quantizers.auto as transformers_auto

        assert transformers_auto.AUTO_QUANTIZER_MAPPING["orbitquant"] is OrbitQuantizer
        assert transformers_auto.AUTO_QUANTIZATION_CONFIG_MAPPING["orbitquant"] is OrbitQuantConfig


def test_quantizer_inherits_hf_base_classes_when_dependencies_are_installed():
    result = register_hf_quantizers()
    quantizer = OrbitQuantizer(OrbitQuantConfig())

    if result["diffusers"]:
        from diffusers.quantizers.base import DiffusersQuantizer

        assert isinstance(quantizer, DiffusersQuantizer)

    if result["transformers"]:
        from transformers.quantizers import HfQuantizer

        assert isinstance(quantizer, HfQuantizer)


def test_quantizer_reports_target_weight_parameters_only():
    model = TinyQuantizerTransformer()
    quantizer = OrbitQuantizer(OrbitQuantConfig(block_size=8))

    assert quantizer.param_needs_quantization(
        model, "transformer_blocks.0.attn.to_q.weight"
    )
    assert not quantizer.param_needs_quantization(model, "proj_out.weight")
    assert not quantizer.param_needs_quantization(model, "transformer_blocks.0.attn.to_q.bias")


def test_pre_quantized_quantizer_prepares_empty_quantized_module_skeletons():
    model = TinyQuantizerTransformer()
    model.transformer_blocks[0]["modulation"] = torch.nn.Linear(16, 32)
    quantizer = OrbitQuantizer(OrbitQuantConfig(block_size=8), pre_quantized=True)

    quantizer._process_model_before_weight_loading(model)

    orbit_layer = model.transformer_blocks[0]["attn"]["to_q"]
    adaln_layer = model.transformer_blocks[0]["modulation"]
    assert isinstance(orbit_layer, OrbitQuantLinear)
    assert isinstance(adaln_layer, RTNInt4Linear)
    assert orbit_layer.packed_weight_indices.numel() == 128
    assert orbit_layer.row_norms.shape == (16,)
    assert adaln_layer.packed_weight.numel() == 1024
    assert adaln_layer.scales.shape == (32, 1)


def test_on_the_fly_quantizer_quantizes_after_weight_loading_only():
    model = TinyQuantizerTransformer()
    quantizer = OrbitQuantizer(OrbitQuantConfig(block_size=8), pre_quantized=False)

    quantizer._process_model_before_weight_loading(model)
    assert isinstance(model.transformer_blocks[0]["attn"]["to_q"], torch.nn.Linear)
    assert not quantizer.check_if_quantized_param(
        model,
        model.transformer_blocks[0]["attn"]["to_q"].weight,
        "transformer_blocks.0.attn.to_q.weight",
        {},
    )

    quantizer._process_model_after_weight_loading(model)
    assert isinstance(model.transformer_blocks[0]["attn"]["to_q"], OrbitQuantLinear)


def test_pre_quantized_skeleton_accepts_packed_state_dict_strictly():
    torch.manual_seed(0)
    config = OrbitQuantConfig(block_size=8)
    source = TinyQuantizerTransformer()
    source.transformer_blocks[0]["modulation"] = torch.nn.Linear(16, 32)

    on_the_fly_quantizer = OrbitQuantizer(config, pre_quantized=False)
    on_the_fly_quantizer._process_model_before_weight_loading(source)
    on_the_fly_quantizer._process_model_after_weight_loading(source)
    source_state = {key: value.detach().clone() for key, value in source.state_dict().items()}

    restored = TinyQuantizerTransformer()
    restored.transformer_blocks[0]["modulation"] = torch.nn.Linear(16, 32)
    pre_quantized_quantizer = OrbitQuantizer(config, pre_quantized=True)
    pre_quantized_quantizer._process_model_before_weight_loading(restored)

    incompatible = restored.load_state_dict(source_state, strict=True)

    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    assert isinstance(restored.transformer_blocks[0]["attn"]["to_q"], OrbitQuantLinear)
    assert isinstance(restored.transformer_blocks[0]["modulation"], RTNInt4Linear)
    restored_state = restored.state_dict()
    assert torch.equal(
        restored_state["transformer_blocks.0.attn.to_q.packed_weight_indices"],
        source_state["transformer_blocks.0.attn.to_q.packed_weight_indices"],
    )
    assert torch.equal(
        restored_state["transformer_blocks.0.modulation.packed_weight"],
        source_state["transformer_blocks.0.modulation.packed_weight"],
    )
    x = torch.randn(2, 3, 16)
    assert torch.isfinite(restored.transformer_blocks[0]["attn"]["to_q"](x)).all()
    assert torch.isfinite(restored.transformer_blocks[0]["modulation"](x)).all()


def test_pre_quantized_quantizer_materializes_streamed_packed_tensors():
    torch.manual_seed(0)
    config = OrbitQuantConfig(block_size=8)
    source = TinyQuantizerTransformer()
    source.transformer_blocks[0]["modulation"] = torch.nn.Linear(16, 32)
    source_quantizer = OrbitQuantizer(config, pre_quantized=False)
    source_quantizer._process_model_before_weight_loading(source)
    source_quantizer._process_model_after_weight_loading(source)
    source_state = {key: value.detach().clone() for key, value in source.state_dict().items()}

    restored = TinyQuantizerTransformer()
    restored.transformer_blocks[0]["modulation"] = torch.nn.Linear(16, 32)
    streaming_quantizer = OrbitQuantizer(config, pre_quantized=True)
    streaming_quantizer._process_model_before_weight_loading(restored)
    unexpected_keys = list(source_state)

    streamed_keys = [
        "transformer_blocks.0.attn.to_q.packed_weight_indices",
        "transformer_blocks.0.attn.to_q.row_norms",
        "transformer_blocks.0.attn.to_q.bias",
        "transformer_blocks.0.modulation.packed_weight",
        "transformer_blocks.0.modulation.scales",
        "transformer_blocks.0.modulation.bias",
    ]
    for key in streamed_keys:
        value = source_state[key]
        assert streaming_quantizer.check_if_quantized_param(
            restored, value, key, source_state, param_device="cpu"
        )
        streaming_quantizer.create_quantized_param(
            restored, value, key, "cpu", source_state, unexpected_keys
        )

    restored_state = restored.state_dict()
    for key in streamed_keys:
        assert torch.equal(restored_state[key], source_state[key])
        assert key not in unexpected_keys


def test_diffusers_meta_loader_streams_pre_quantized_tensors_through_quantizer():
    model_loading_utils = pytest.importorskip("diffusers.models.model_loading_utils")
    torch.manual_seed(0)
    config = OrbitQuantConfig(block_size=8)
    source = TinyQuantizerTransformer()
    source.transformer_blocks[0]["modulation"] = torch.nn.Linear(16, 32)
    source_quantizer = OrbitQuantizer(config, pre_quantized=False)
    source_quantizer._process_model_before_weight_loading(source)
    source_quantizer._process_model_after_weight_loading(source)
    source_state = OrderedDict(
        (key, value.detach().clone()) for key, value in source.state_dict().items()
    )

    restored = TinyQuantizerTransformer()
    restored.transformer_blocks[0]["modulation"] = torch.nn.Linear(16, 32)
    streaming_quantizer = OrbitQuantizer(config, pre_quantized=True)
    streaming_quantizer._process_model_before_weight_loading(restored)
    streamed_keys = [
        "transformer_blocks.0.attn.to_q.packed_weight_indices",
        "transformer_blocks.0.attn.to_q.row_norms",
        "transformer_blocks.0.modulation.packed_weight",
        "transformer_blocks.0.modulation.scales",
    ]
    unexpected_keys = list(streamed_keys)

    model_loading_utils.load_model_dict_into_meta(
        restored,
        source_state,
        hf_quantizer=streaming_quantizer,
        unexpected_keys=unexpected_keys,
    )

    restored_state = restored.state_dict()
    for key in streamed_keys:
        assert torch.equal(restored_state[key], source_state[key])
        assert key not in unexpected_keys
    x = torch.randn(2, 3, 16)
    assert torch.isfinite(restored.transformer_blocks[0]["attn"]["to_q"](x)).all()
    assert torch.isfinite(restored.transformer_blocks[0]["modulation"](x)).all()


def test_quantizer_dequantize_restores_torch_linear_modules_for_debugging():
    torch.manual_seed(1)
    model = TinyQuantizerTransformer()
    model.transformer_blocks[0]["modulation"] = torch.nn.Linear(16, 32)
    quantizer = OrbitQuantizer(OrbitQuantConfig(block_size=8), pre_quantized=False)

    quantizer._process_model_before_weight_loading(model)
    quantizer._process_model_after_weight_loading(model)
    assert isinstance(model.transformer_blocks[0]["attn"]["to_q"], OrbitQuantLinear)
    assert isinstance(model.transformer_blocks[0]["modulation"], RTNInt4Linear)

    dequantized = quantizer._dequantize(model)

    orbit_layer = dequantized.transformer_blocks[0]["attn"]["to_q"]
    adaln_layer = dequantized.transformer_blocks[0]["modulation"]
    assert isinstance(orbit_layer, torch.nn.Linear)
    assert isinstance(adaln_layer, torch.nn.Linear)
    assert orbit_layer.weight.requires_grad is False
    assert adaln_layer.weight.requires_grad is False
    x = torch.randn(2, 3, 16)
    assert torch.isfinite(orbit_layer(x)).all()
    assert torch.isfinite(adaln_layer(x)).all()
