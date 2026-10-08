import pytest
import torch

from orbitquant import OrbitQuantConfig, quantize_model
from orbitquant.layers import OrbitQuantLinear
from orbitquant.policies import classify_linear_modules, resolve_target_policy
from orbitquant.quantizer import OrbitQuantizer


class Kandinsky6Transformer3DModel(torch.nn.Module):
    """Small fixture using the upstream multimodal projection paths."""

    def __init__(self):
        super().__init__()
        for prefix in ("video", "audio"):
            setattr(
                self,
                f"{prefix}_time_embeddings",
                torch.nn.ModuleDict({"timestep_embedder": torch.nn.Linear(16, 16)}),
            )
            setattr(
                self,
                f"{prefix}_text_embeddings",
                torch.nn.ModuleDict({"in_layer": torch.nn.Linear(16, 16)}),
            )
            setattr(
                self,
                f"{prefix}_text_transformer_blocks",
                torch.nn.ModuleList(
                    [
                        torch.nn.ModuleDict(
                            {
                                "text_modulation": torch.nn.ModuleDict(
                                    {"out_layer": torch.nn.Linear(16, 96)}
                                ),
                                "attn": torch.nn.ModuleDict({"to_query": torch.nn.Linear(16, 16)}),
                                "feed_forward": torch.nn.Sequential(torch.nn.Linear(16, 32)),
                            }
                        )
                    ]
                ),
            )
        self.visual_transformer_blocks = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {
                        "video_dec_block": torch.nn.ModuleDict(
                            {
                                "visual_modulation": torch.nn.ModuleDict(
                                    {"out_layer": torch.nn.Linear(16, 144)}
                                ),
                                "self_attention": torch.nn.ModuleDict(
                                    {"out_layer": torch.nn.Linear(16, 16)}
                                ),
                            }
                        ),
                        "audio_dec_block": torch.nn.ModuleDict(
                            {
                                "cross_attention": torch.nn.ModuleDict(
                                    {"to_value": torch.nn.Linear(16, 16)}
                                ),
                            }
                        ),
                        "va_cross_attention": torch.nn.ModuleDict(
                            {"to_key": torch.nn.Linear(16, 16)}
                        ),
                        "av_cross_attention": torch.nn.ModuleDict(
                            {"out_layer": torch.nn.Linear(16, 16)}
                        ),
                        "va_modulation": torch.nn.ModuleDict(
                            {"out_layer": torch.nn.Linear(16, 48)}
                        ),
                    }
                )
            ]
        )
        for name in ("out_layer", "audio_out_layer"):
            setattr(
                self,
                name,
                torch.nn.ModuleDict(
                    {
                        "modulation": torch.nn.ModuleDict({"out_layer": torch.nn.Linear(16, 32)}),
                        "out_layer": torch.nn.Linear(16, 8),
                    }
                ),
            )
        self.unrelated_attention = torch.nn.Linear(16, 16)


def test_kandinsky6_auto_policy_preserves_conditioning_and_output_boundaries():
    model = Kandinsky6Transformer3DModel()
    config = OrbitQuantConfig()
    assert resolve_target_policy(model, config) == "kandinsky6"
    decisions = classify_linear_modules(model, config)
    for name, decision in decisions.items():
        if "modulation" in name or "time_embeddings" in name:
            assert decision.action == "bf16_skip", name
            assert decision.dtype == "float32", name
        elif "transformer_blocks" in name:
            assert decision.action == "orbitquant", name
        else:
            assert decision.action == "bf16_skip", name


def test_kandinsky6_conversion_keeps_fp32_modulation_and_quantizes_cross_modal_paths():
    model = Kandinsky6Transformer3DModel().to(torch.bfloat16)
    summary = quantize_model(model, OrbitQuantConfig(), quantization_device="cpu")
    assert not summary.adaln_modules
    assert isinstance(
        model.visual_transformer_blocks[0].av_cross_attention.out_layer, OrbitQuantLinear
    )
    assert model.visual_transformer_blocks[0].va_modulation.out_layer.weight.dtype == torch.float32
    assert model.audio_time_embeddings.timestep_embedder.weight.dtype == torch.float32
    assert model.audio_out_layer.out_layer.weight.dtype == torch.bfloat16


def test_kandinsky6_config_round_trip():
    config = OrbitQuantConfig(target_policy="kandinsky6")
    assert OrbitQuantConfig.from_dict(config.to_dict()).target_policy == "kandinsky6"


@pytest.mark.parametrize("pre_quantized", [True, False])
def test_kandinsky6_loader_preserves_fp32_values_before_diffusers_cast(pre_quantized):
    model = Kandinsky6Transformer3DModel()
    quantizer = OrbitQuantizer(OrbitQuantConfig(), pre_quantized=pre_quantized)
    if not pre_quantized:
        quantizer._diffusers_streaming_quantization = True
    for tensor_name in ("weight", "bias"):
        name = f"visual_transformer_blocks.0.va_modulation.out_layer.{tensor_name}"
        module = model.visual_transformer_blocks[0].va_modulation.out_layer
        original = getattr(module, tensor_name).detach().clone()
        assert not torch.equal(original, original.to(torch.bfloat16).float())
        cast = original.to(torch.bfloat16)
        assert quantizer.check_if_quantized_param(model, cast, name)
        quantizer.create_quantized_param(model, cast, name, "cpu", {name: original})
        torch.testing.assert_close(getattr(module, tensor_name), original, rtol=0, atol=0)


def test_kandinsky6_real_diffusers_joint_forward_and_prequantized_reload(tmp_path):
    diffusers = pytest.importorskip("diffusers")
    cls = getattr(diffusers, "Kandinsky6Transformer3DModel", None)
    if cls is None:
        pytest.skip("requires Diffusers with Kandinsky 6 support")
    torch.manual_seed(7)
    model = cls(
        in_visual_dim=4,
        out_visual_dim=4,
        in_text_dim=16,
        in_text_dim2=8,
        time_dim=16,
        model_dim=32,
        ff_dim=64,
        num_text_blocks=1,
        num_visual_blocks=1,
        axes_dims=(4, 4, 8),
        in_audio_dim=4,
        out_audio_dim=4,
        model_dim_a=32,
        time_dim_a=16,
        ff_dim_a=64,
        axes_dims_a=(4, 4, 8),
        ca_rope=True,
        cross_gates=True,
        fix_modulation=True,
    ).eval()
    quantize_model(model, OrbitQuantConfig(runtime_mode="dequant_bf16"), quantization_device="cpu")
    inputs = {
        "hidden_states": torch.randn(1, 1, 2, 2, 9),
        "encoder_hidden_states": torch.randn(1, 3, 16),
        "pooled_projections": torch.randn(1, 8),
        "timestep": torch.tensor([500.0]),
        "audio_hidden_states": torch.randn(1, 3, 4),
        "encoder_attention_mask": torch.tensor([[True, True, False]]),
    }
    with torch.no_grad():
        expected = model(**inputs)
    assert torch.isfinite(expected.sample).all()
    assert torch.isfinite(expected.audio_sample).all()
    model.save_pretrained(tmp_path, safe_serialization=True)
    restored = cls.from_pretrained(tmp_path, torch_dtype=torch.float32)
    assert isinstance(
        restored.visual_transformer_blocks[0].av_cross_attention.to_query, OrbitQuantLinear
    )
    assert (
        restored.visual_transformer_blocks[0].va_modulation.out_layer.weight.dtype == torch.float32
    )
    with torch.no_grad():
        actual = restored(**inputs)
    torch.testing.assert_close(actual.sample, expected.sample, rtol=0, atol=0)
    torch.testing.assert_close(actual.audio_sample, expected.audio_sample, rtol=0, atol=0)
    inputs.pop("audio_hidden_states")
    with torch.no_grad():
        assert restored(**inputs).audio_sample is None
