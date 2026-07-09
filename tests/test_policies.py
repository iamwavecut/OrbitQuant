import pytest
import torch

from orbitquant.config import OrbitQuantConfig
from orbitquant.policies import classify_linear_modules


class TinyDenoiser(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {
                        "attn": torch.nn.ModuleDict(
                            {
                                "to_q": torch.nn.Linear(16, 16),
                                "to_k": torch.nn.Linear(16, 16),
                                "to_v": torch.nn.Linear(16, 16),
                            }
                        ),
                        "ff": torch.nn.ModuleDict(
                            {
                                "linear_1": torch.nn.Linear(16, 32),
                                "linear_2": torch.nn.Linear(32, 16),
                            }
                        ),
                        "modulation": torch.nn.Linear(16, 32),
                    }
                )
            ]
        )
        self.x_embedder = torch.nn.Linear(8, 16)
        self.proj_out = torch.nn.Linear(16, 8)


def test_generic_policy_quantizes_block_linears_and_skips_embedder_and_final_head():
    model = TinyDenoiser()
    config = OrbitQuantConfig(target_policy="generic_dit")

    decisions = classify_linear_modules(model, config)

    assert decisions["transformer_blocks.0.attn.to_q"].action == "orbitquant"
    assert decisions["transformer_blocks.0.ff.linear_1"].action == "orbitquant"
    assert decisions["transformer_blocks.0.modulation"].action == "adaln_int4_rtn"
    assert decisions["x_embedder"].action == "bf16_skip"
    assert decisions["proj_out"].action == "bf16_skip"


def test_generic_policy_does_not_treat_norm1_linear_as_adaln_without_model_evidence():
    model = torch.nn.Module()
    model.transformer_blocks = torch.nn.ModuleList(
        [
            torch.nn.ModuleDict(
                {"norm1": torch.nn.ModuleDict({"linear": torch.nn.Linear(16, 16)})}
            )
        ]
    )
    config = OrbitQuantConfig(target_policy="generic_dit")

    decisions = classify_linear_modules(model, config)

    assert decisions["transformer_blocks.0.norm1.linear"].action != "adaln_int4_rtn"


def test_generic_policy_uses_modules_dtype_dict_as_explicit_skip_override():
    model = TinyDenoiser()
    config = OrbitQuantConfig(
        target_policy="generic_dit",
        modules_dtype_dict={"float16": ["transformer_blocks.0.attn.to_q"]},
    )

    decisions = classify_linear_modules(model, config)

    decision = decisions["transformer_blocks.0.attn.to_q"]
    assert decision.action == "bf16_skip"
    assert decision.dtype == "float16"
    assert decision.reason == "explicit modules_dtype_dict override"
    assert decisions["transformer_blocks.0.attn.to_k"].action == "orbitquant"


def _diffusers_class(class_name: str):
    diffusers = pytest.importorskip("diffusers")
    cls = getattr(diffusers, class_name, None)
    if cls is None:
        pytest.skip(f"{class_name} is not available in this diffusers version")
    return cls


def _decisions(model: torch.nn.Module, target_policy: str):
    return classify_linear_modules(model, OrbitQuantConfig(target_policy=target_policy))


def _assert_actions(decisions, expected):
    for name, action in expected.items():
        assert decisions[name].action == action


def test_flux_policy_matches_paper_layer_scope_on_diffusers_model():
    FluxTransformer2DModel = _diffusers_class("FluxTransformer2DModel")
    model = FluxTransformer2DModel(
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=16,
        pooled_projection_dim=16,
        axes_dims_rope=(4, 6, 6),
    )

    decisions = _decisions(model, "flux")

    _assert_actions(
        decisions,
        {
            "transformer_blocks.0.attn.to_q": "orbitquant",
            "transformer_blocks.0.attn.to_k": "orbitquant",
            "transformer_blocks.0.attn.to_v": "orbitquant",
            "transformer_blocks.0.attn.to_out.0": "orbitquant",
            "transformer_blocks.0.attn.add_q_proj": "orbitquant",
            "transformer_blocks.0.attn.add_k_proj": "orbitquant",
            "transformer_blocks.0.attn.add_v_proj": "orbitquant",
            "transformer_blocks.0.attn.to_add_out": "orbitquant",
            "transformer_blocks.0.ff.net.0.proj": "orbitquant",
            "transformer_blocks.0.ff.net.2": "orbitquant",
            "transformer_blocks.0.ff_context.net.0.proj": "orbitquant",
            "transformer_blocks.0.ff_context.net.2": "orbitquant",
            "single_transformer_blocks.0.proj_mlp": "orbitquant",
            "single_transformer_blocks.0.proj_out": "orbitquant",
            "single_transformer_blocks.0.attn.to_q": "orbitquant",
            "single_transformer_blocks.0.attn.to_k": "orbitquant",
            "single_transformer_blocks.0.attn.to_v": "orbitquant",
            "transformer_blocks.0.norm1.linear": "adaln_int4_rtn",
            "transformer_blocks.0.norm1_context.linear": "adaln_int4_rtn",
            "single_transformer_blocks.0.norm.linear": "adaln_int4_rtn",
            "time_text_embed.timestep_embedder.linear_1": "bf16_skip",
            "context_embedder": "bf16_skip",
            "x_embedder": "bf16_skip",
            "norm_out.linear": "bf16_skip",
            "proj_out": "bf16_skip",
        },
    )


def test_flux2_policy_matches_paper_layer_scope_on_diffusers_model():
    Flux2Transformer2DModel = _diffusers_class("Flux2Transformer2DModel")
    model = Flux2Transformer2DModel(
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=16,
        timestep_guidance_channels=16,
        axes_dims_rope=(4, 4, 4, 4),
    )

    decisions = _decisions(model, "flux2")

    _assert_actions(
        decisions,
        {
            "double_stream_modulation_img.linear": "adaln_int4_rtn",
            "double_stream_modulation_txt.linear": "adaln_int4_rtn",
            "single_stream_modulation.linear": "adaln_int4_rtn",
            "transformer_blocks.0.attn.to_q": "orbitquant",
            "transformer_blocks.0.attn.to_k": "orbitquant",
            "transformer_blocks.0.attn.to_v": "orbitquant",
            "transformer_blocks.0.attn.to_out.0": "orbitquant",
            "transformer_blocks.0.attn.add_q_proj": "orbitquant",
            "transformer_blocks.0.attn.add_k_proj": "orbitquant",
            "transformer_blocks.0.attn.add_v_proj": "orbitquant",
            "transformer_blocks.0.attn.to_add_out": "orbitquant",
            "transformer_blocks.0.ff.linear_in": "orbitquant",
            "transformer_blocks.0.ff.linear_out": "orbitquant",
            "transformer_blocks.0.ff_context.linear_in": "orbitquant",
            "transformer_blocks.0.ff_context.linear_out": "orbitquant",
            "single_transformer_blocks.0.attn.to_qkv_mlp_proj": "orbitquant",
            "single_transformer_blocks.0.attn.to_out": "orbitquant",
            "time_guidance_embed.timestep_embedder.linear_1": "bf16_skip",
            "context_embedder": "bf16_skip",
            "x_embedder": "bf16_skip",
            "norm_out.linear": "bf16_skip",
            "proj_out": "bf16_skip",
        },
    )


def test_wan_policy_matches_paper_layer_scope_on_diffusers_model():
    WanTransformer3DModel = _diffusers_class("WanTransformer3DModel")
    model = WanTransformer3DModel(
        num_layers=1,
        num_attention_heads=2,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=16,
        ffn_dim=32,
        cross_attn_norm=True,
    )

    decisions = _decisions(model, "wan")

    _assert_actions(
        decisions,
        {
            "blocks.0.attn1.to_q": "orbitquant",
            "blocks.0.attn1.to_k": "orbitquant",
            "blocks.0.attn1.to_v": "orbitquant",
            "blocks.0.attn1.to_out.0": "orbitquant",
            "blocks.0.attn2.to_q": "orbitquant",
            "blocks.0.attn2.to_k": "orbitquant",
            "blocks.0.attn2.to_v": "orbitquant",
            "blocks.0.attn2.to_out.0": "orbitquant",
            "blocks.0.ffn.net.0.proj": "orbitquant",
            "blocks.0.ffn.net.2": "orbitquant",
            "condition_embedder.time_embedder.linear_1": "bf16_skip",
            "condition_embedder.text_embedder.linear_1": "bf16_skip",
            "proj_out": "bf16_skip",
        },
    )


def test_z_image_policy_matches_paper_layer_scope_on_diffusers_model():
    ZImageTransformer2DModel = _diffusers_class("ZImageTransformer2DModel")
    model = ZImageTransformer2DModel(
        dim=32,
        n_layers=1,
        n_refiner_layers=1,
        n_heads=2,
        n_kv_heads=2,
        cap_feat_dim=16,
        axes_dims=[4, 6, 6],
        axes_lens=[64, 64, 64],
    )

    decisions = _decisions(model, "z_image")

    _assert_actions(
        decisions,
        {
            "noise_refiner.0.attention.to_q": "orbitquant",
            "noise_refiner.0.attention.to_k": "orbitquant",
            "noise_refiner.0.attention.to_v": "orbitquant",
            "noise_refiner.0.attention.to_out.0": "orbitquant",
            "noise_refiner.0.feed_forward.w1": "orbitquant",
            "noise_refiner.0.feed_forward.w2": "orbitquant",
            "noise_refiner.0.feed_forward.w3": "orbitquant",
            "noise_refiner.0.adaLN_modulation.0": "adaln_int4_rtn",
            "context_refiner.0.attention.to_q": "orbitquant",
            "context_refiner.0.attention.to_k": "orbitquant",
            "context_refiner.0.attention.to_v": "orbitquant",
            "context_refiner.0.attention.to_out.0": "orbitquant",
            "context_refiner.0.feed_forward.w1": "orbitquant",
            "context_refiner.0.feed_forward.w2": "orbitquant",
            "context_refiner.0.feed_forward.w3": "orbitquant",
            "layers.0.attention.to_q": "orbitquant",
            "layers.0.attention.to_k": "orbitquant",
            "layers.0.attention.to_v": "orbitquant",
            "layers.0.attention.to_out.0": "orbitquant",
            "layers.0.feed_forward.w1": "orbitquant",
            "layers.0.feed_forward.w2": "orbitquant",
            "layers.0.feed_forward.w3": "orbitquant",
            "layers.0.adaLN_modulation.0": "adaln_int4_rtn",
            "all_x_embedder.2-1": "bf16_skip",
            "t_embedder.mlp.0": "bf16_skip",
            "cap_embedder.1": "bf16_skip",
            "all_final_layer.2-1.linear": "bf16_skip",
            "all_final_layer.2-1.adaLN_modulation.1": "bf16_skip",
        },
    )
