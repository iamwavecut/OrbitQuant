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

    quantizer._process_model_after_weight_loading(model)
    assert isinstance(model.transformer_blocks[0]["attn"]["to_q"], OrbitQuantLinear)
