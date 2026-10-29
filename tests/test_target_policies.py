import pytest
import torch

from orbitquant.config import OrbitQuantConfig
from orbitquant.policies import classify_linear_modules, resolve_target_policy


class TinyFluxSingleBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.single_transformer_blocks = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {
                        "proj_mlp": torch.nn.Linear(16, 64),
                        "proj_out": torch.nn.Linear(80, 16),
                    }
                )
            ]
        )
        self.norm_out = torch.nn.ModuleDict({"linear": torch.nn.Linear(16, 32)})
        self.proj_out = torch.nn.Linear(16, 8)


def test_flux_single_block_proj_out_is_quantized_but_final_proj_out_is_skipped():
    decisions = classify_linear_modules(
        TinyFluxSingleBlock(), OrbitQuantConfig(target_policy="flux")
    )

    assert decisions["single_transformer_blocks.0.proj_mlp"].action == "orbitquant"
    assert decisions["single_transformer_blocks.0.proj_out"].action == "orbitquant"
    assert decisions["norm_out.linear"].action == "bf16_skip"
    assert decisions["proj_out"].action == "bf16_skip"


class TinyFlux2Names(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.double_stream_modulation_img = torch.nn.ModuleDict(
            {"linear": torch.nn.Linear(16, 96)}
        )
        self.double_stream_modulation_txt = torch.nn.ModuleDict(
            {"linear": torch.nn.Linear(16, 96)}
        )
        self.single_stream_modulation = torch.nn.ModuleDict({"linear": torch.nn.Linear(16, 48)})
        self.transformer_blocks = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {
                        "attn": torch.nn.ModuleDict(
                            {
                                "to_q": torch.nn.Linear(16, 16),
                                "add_k_proj": torch.nn.Linear(16, 16),
                            }
                        ),
                        "double_stream_modulation_img": torch.nn.Linear(16, 96),
                    }
                )
            ]
        )
        self.single_transformer_blocks = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {"attn": torch.nn.ModuleDict({"to_qkv_mlp_proj": torch.nn.Linear(16, 96)})}
                )
            ]
        )
        self.context_embedder = torch.nn.Linear(16, 16)


class TinyFlux2FalsePositiveNames(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.final_transformer_blocks = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {
                        "attn": torch.nn.ModuleDict(
                            {"to_q": torch.nn.Linear(16, 16)}
                        )
                    }
                )
            ]
        )
        self.transformer_blocks_final = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {
                        "attn": torch.nn.ModuleDict(
                            {"to_q": torch.nn.Linear(16, 16)}
                        )
                    }
                )
            ]
        )
        self.prefix_single_stream_modulation = torch.nn.ModuleDict(
            {"linear": torch.nn.Linear(16, 48)}
        )


def test_flux2_policy_covers_fused_and_text_conditioning_projections():
    decisions = classify_linear_modules(TinyFlux2Names(), OrbitQuantConfig(target_policy="flux2"))

    assert decisions["transformer_blocks.0.attn.to_q"].action == "orbitquant"
    assert decisions["transformer_blocks.0.attn.add_k_proj"].action == "orbitquant"
    assert decisions["single_transformer_blocks.0.attn.to_qkv_mlp_proj"].action == "orbitquant"
    assert decisions["double_stream_modulation_img.linear"].action == "adaln_int4_rtn"
    assert decisions["double_stream_modulation_txt.linear"].action == "adaln_int4_rtn"
    assert decisions["single_stream_modulation.linear"].action == "adaln_int4_rtn"
    assert decisions["transformer_blocks.0.double_stream_modulation_img"].action == "adaln_int4_rtn"
    assert decisions["context_embedder"].action == "bf16_skip"


def test_flux2_policy_does_not_match_boundary_or_prefixed_modulation_substrings():
    decisions = classify_linear_modules(
        TinyFlux2FalsePositiveNames(), OrbitQuantConfig(target_policy="flux2")
    )

    assert decisions["final_transformer_blocks.0.attn.to_q"].action == "bf16_skip"
    assert decisions["transformer_blocks_final.0.attn.to_q"].action == "bf16_skip"
    assert decisions["prefix_single_stream_modulation.linear"].action == "bf16_skip"


class TinyZImageNames(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {
                        "attention": torch.nn.ModuleDict({"to_q": torch.nn.Linear(16, 16)}),
                        "feed_forward": torch.nn.ModuleDict(
                            {"net": torch.nn.ModuleList([torch.nn.Linear(16, 32)])}
                        ),
                        "adaLN_modulation": torch.nn.Sequential(torch.nn.Linear(16, 64)),
                    }
                )
            ]
        )
        self.t_embedder = torch.nn.ModuleDict(
            {"mlp": torch.nn.ModuleList([torch.nn.Linear(16, 16)])}
        )
        self.final_layer = torch.nn.ModuleDict({"linear": torch.nn.Linear(16, 8)})
        self.all_final_layer = torch.nn.ModuleDict(
            {"adaLN_modulation": torch.nn.Sequential(torch.nn.Linear(16, 64))}
        )


def test_z_image_policy_covers_attention_ffn_and_adaln_but_skips_final_layer():
    decisions = classify_linear_modules(
        TinyZImageNames(), OrbitQuantConfig(target_policy="z_image")
    )

    assert decisions["transformer_blocks.0.attention.to_q"].action == "orbitquant"
    assert decisions["transformer_blocks.0.feed_forward.net.0"].action == "orbitquant"
    assert decisions["transformer_blocks.0.adaLN_modulation.0"].action == "adaln_int4_rtn"
    assert decisions["t_embedder.mlp.0"].action == "bf16_skip"
    assert decisions["final_layer.linear"].action == "bf16_skip"
    assert decisions["all_final_layer.adaLN_modulation.0"].action == "bf16_skip"


def test_z_image_policy_matches_current_diffusers_tiny_module_names():
    pytest.importorskip("diffusers")
    from diffusers import ZImageTransformer2DModel

    model = ZImageTransformer2DModel(
        in_channels=4,
        dim=256,
        n_layers=1,
        n_refiner_layers=1,
        n_heads=2,
        n_kv_heads=2,
        cap_feat_dim=32,
        axes_dims=[32, 48, 48],
        axes_lens=[64, 32, 32],
    )

    decisions = classify_linear_modules(model, OrbitQuantConfig(target_policy="z_image"))

    assert decisions["noise_refiner.0.attention.to_q"].action == "orbitquant"
    assert decisions["noise_refiner.0.feed_forward.w1"].action == "orbitquant"
    assert decisions["noise_refiner.0.adaLN_modulation.0"].action == "adaln_int4_rtn"
    assert decisions["context_refiner.0.attention.to_out.0"].action == "orbitquant"
    assert decisions["context_refiner.0.feed_forward.w3"].action == "orbitquant"
    assert decisions["layers.0.attention.to_k"].action == "orbitquant"
    assert decisions["layers.0.feed_forward.w2"].action == "orbitquant"
    assert decisions["layers.0.adaLN_modulation.0"].action == "adaln_int4_rtn"
    assert decisions["all_x_embedder.2-1"].action == "bf16_skip"
    assert decisions["t_embedder.mlp.0"].action == "bf16_skip"
    assert decisions["cap_embedder.1"].action == "bf16_skip"
    assert decisions["all_final_layer.2-1.linear"].action == "bf16_skip"
    assert decisions["all_final_layer.2-1.adaLN_modulation.1"].action == "bf16_skip"


class TinyWanNames(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {
                        "attn1": torch.nn.ModuleDict(
                            {
                                "to_q": torch.nn.Linear(16, 16),
                                "to_k": torch.nn.Linear(16, 16),
                                "to_v": torch.nn.Linear(16, 16),
                                "to_out": torch.nn.Sequential(torch.nn.Linear(16, 16)),
                            }
                        ),
                        "attn2": torch.nn.ModuleDict(
                            {
                                "to_q": torch.nn.Linear(16, 16),
                                "to_k": torch.nn.Linear(16, 16),
                                "to_v": torch.nn.Linear(16, 16),
                                "to_out": torch.nn.Sequential(torch.nn.Linear(16, 16)),
                            }
                        ),
                        "ffn": torch.nn.ModuleDict(
                            {
                                "net": torch.nn.ModuleList(
                                    [torch.nn.Linear(16, 32), torch.nn.Linear(32, 16)]
                                )
                            }
                        ),
                        "norm1": torch.nn.ModuleDict({"linear": torch.nn.Linear(16, 32)}),
                    }
                )
            ]
        )
        self.time_proj = torch.nn.Linear(16, 16)


def test_wan_policy_covers_self_cross_attention_and_ffn_but_skips_time_projection():
    decisions = classify_linear_modules(TinyWanNames(), OrbitQuantConfig(target_policy="wan"))

    assert decisions["blocks.0.attn1.to_q"].action == "orbitquant"
    assert decisions["blocks.0.attn1.to_k"].action == "orbitquant"
    assert decisions["blocks.0.attn1.to_v"].action == "orbitquant"
    assert decisions["blocks.0.attn1.to_out.0"].action == "orbitquant"
    assert decisions["blocks.0.attn2.to_q"].action == "orbitquant"
    assert decisions["blocks.0.attn2.to_k"].action == "orbitquant"
    assert decisions["blocks.0.attn2.to_v"].action == "orbitquant"
    assert decisions["blocks.0.attn2.to_out.0"].action == "orbitquant"
    assert decisions["blocks.0.ffn.net.0"].action == "orbitquant"
    assert decisions["blocks.0.ffn.net.1"].action == "orbitquant"
    assert decisions["blocks.0.norm1.linear"].action == "bf16_skip"
    assert decisions["time_proj"].action == "bf16_skip"
    assert {decision.action for decision in decisions.values()}.isdisjoint({"adaln_int4_rtn"})


def test_target_policy_changes_transformer_block_scope():
    decisions = classify_linear_modules(TinyWanNames(), OrbitQuantConfig(target_policy="flux"))

    assert decisions["blocks.0.attn1.to_q"].action == "bf16_skip"


def test_flux_policy_matches_current_diffusers_tiny_module_names():
    pytest.importorskip("diffusers")
    from diffusers import FluxTransformer2DModel

    model = FluxTransformer2DModel(
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=16,
        pooled_projection_dim=8,
        axes_dims_rope=(4, 4, 0),
    )

    decisions = classify_linear_modules(model, OrbitQuantConfig(target_policy="flux"))

    assert decisions["transformer_blocks.0.norm1.linear"].action == "adaln_int4_rtn"
    assert decisions["single_transformer_blocks.0.norm.linear"].action == "adaln_int4_rtn"
    assert decisions["single_transformer_blocks.0.proj_out"].action == "orbitquant"
    assert decisions["norm_out.linear"].action == "bf16_skip"
    assert decisions["proj_out"].action == "bf16_skip"


class AutoFluxTransformer2DModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {
                        "norm1": torch.nn.ModuleDict(
                            {"linear": torch.nn.Linear(16, 32)}
                        ),
                        "attn": torch.nn.ModuleDict(
                            {"to_q": torch.nn.Linear(16, 16)}
                        ),
                    }
                )
            ]
        )
        self.proj_out = torch.nn.Linear(16, 8)


def test_auto_policy_infers_flux_from_transformer_class_name():
    model = AutoFluxTransformer2DModel()
    config = OrbitQuantConfig(target_policy="auto")

    decisions = classify_linear_modules(model, config)

    assert resolve_target_policy(model, config) == "flux"
    assert decisions["transformer_blocks.0.norm1.linear"].action == "adaln_int4_rtn"
    assert decisions["transformer_blocks.0.attn.to_q"].action == "orbitquant"
    assert decisions["proj_out"].action == "bf16_skip"


def test_flux2_policy_matches_current_diffusers_tiny_module_names():
    pytest.importorskip("diffusers")
    from diffusers import Flux2Transformer2DModel

    model = Flux2Transformer2DModel(
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=16,
        timestep_guidance_channels=8,
        mlp_ratio=2.0,
        axes_dims_rope=(2, 2, 2, 2),
        guidance_embeds=False,
    )

    decisions = classify_linear_modules(model, OrbitQuantConfig(target_policy="flux2"))

    assert decisions["double_stream_modulation_img.linear"].action == "adaln_int4_rtn"
    assert decisions["single_stream_modulation.linear"].action == "adaln_int4_rtn"
    assert decisions["single_transformer_blocks.0.attn.to_qkv_mlp_proj"].action == "orbitquant"
    assert decisions["norm_out.linear"].action == "bf16_skip"
    assert decisions["proj_out"].action == "bf16_skip"


class AutoFlux2Transformer2DModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.double_stream_modulation_img = torch.nn.ModuleDict(
            {"linear": torch.nn.Linear(16, 96)}
        )
        self.single_transformer_blocks = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {"attn": torch.nn.ModuleDict({"to_qkv_mlp_proj": torch.nn.Linear(16, 96)})}
                )
            ]
        )


def test_auto_policy_infers_flux2_from_transformer_class_name():
    model = AutoFlux2Transformer2DModel()
    config = OrbitQuantConfig(target_policy="auto")

    decisions = classify_linear_modules(model, config)

    assert resolve_target_policy(model, config) == "flux2"
    assert decisions["double_stream_modulation_img.linear"].action == "adaln_int4_rtn"
    assert decisions["single_transformer_blocks.0.attn.to_qkv_mlp_proj"].action == "orbitquant"


def test_wan_policy_matches_current_diffusers_tiny_module_names():
    pytest.importorskip("diffusers")
    from diffusers import WanTransformer3DModel

    model = WanTransformer3DModel(
        num_layers=1,
        num_attention_heads=2,
        attention_head_dim=8,
        text_dim=16,
        freq_dim=8,
        ffn_dim=32,
    )

    decisions = classify_linear_modules(model, OrbitQuantConfig(target_policy="wan"))

    assert decisions["condition_embedder.time_embedder.linear_1"].action == "bf16_skip"
    assert decisions["blocks.0.attn1.to_q"].action == "orbitquant"
    assert decisions["blocks.0.attn1.to_k"].action == "orbitquant"
    assert decisions["blocks.0.attn1.to_v"].action == "orbitquant"
    assert decisions["blocks.0.attn1.to_out.0"].action == "orbitquant"
    assert decisions["blocks.0.attn2.to_q"].action == "orbitquant"
    assert decisions["blocks.0.attn2.to_k"].action == "orbitquant"
    assert decisions["blocks.0.attn2.to_v"].action == "orbitquant"
    assert decisions["blocks.0.attn2.to_out.0"].action == "orbitquant"
    assert decisions["blocks.0.ffn.net.0.proj"].action == "orbitquant"
    assert decisions["blocks.0.ffn.net.2"].action == "orbitquant"
    assert decisions["proj_out"].action == "bf16_skip"


class AutoZImageTransformer2DModel(TinyZImageNames):
    pass


class AutoWanTransformer3DModel(TinyWanNames):
    pass


def test_auto_policy_infers_z_image_and_wan_from_transformer_class_names():
    z_image = AutoZImageTransformer2DModel()
    wan = AutoWanTransformer3DModel()

    assert resolve_target_policy(z_image, OrbitQuantConfig(target_policy="auto")) == "z_image"
    assert resolve_target_policy(wan, OrbitQuantConfig(target_policy="auto")) == "wan"

    z_decisions = classify_linear_modules(z_image, OrbitQuantConfig(target_policy="auto"))
    wan_decisions = classify_linear_modules(wan, OrbitQuantConfig(target_policy="auto"))

    assert z_decisions["transformer_blocks.0.adaLN_modulation.0"].action == "adaln_int4_rtn"
    assert z_decisions["final_layer.linear"].action == "bf16_skip"
    assert wan_decisions["blocks.0.attn2.to_k"].action == "orbitquant"
    assert {decision.action for decision in wan_decisions.values()}.isdisjoint(
        {"adaln_int4_rtn"}
    )


class AdaLayerNormZero(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 96)


class PixArtAlphaTextProjection(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = torch.nn.Linear(16, 16)


class TinyParentClassNames(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.caption_projection = PixArtAlphaTextProjection()
        self.transformer_blocks = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {
                        "norm2": AdaLayerNormZero(),
                        "attn1": torch.nn.ModuleDict({"to_q": torch.nn.Linear(16, 16)}),
                    }
                )
            ]
        )


def test_universal_policy_detects_adaln_and_embedders_by_parent_class():
    decisions = classify_linear_modules(TinyParentClassNames(), OrbitQuantConfig())

    assert decisions["transformer_blocks.0.norm2.linear"].action == "adaln_int4_rtn"
    assert decisions["transformer_blocks.0.attn1.to_q"].action == "orbitquant"
    assert decisions["caption_projection.linear_1"].action == "bf16_skip"


def test_target_policy_version_1_keeps_name_token_classification_for_old_artifacts():
    config = OrbitQuantConfig.from_dict({"weight_bits": 4, "activation_bits": 4})
    decisions = classify_linear_modules(TinyParentClassNames(), config)

    assert config.target_policy_version == 1
    assert decisions["transformer_blocks.0.norm2.linear"].action == "orbitquant"
    assert decisions["caption_projection.linear_1"].action == "orbitquant"


def _tiny_cogvideox():
    from diffusers import CogVideoXTransformer3DModel

    return CogVideoXTransformer3DModel(
        num_attention_heads=2,
        attention_head_dim=8,
        in_channels=4,
        out_channels=4,
        time_embed_dim=8,
        text_embed_dim=16,
        num_layers=1,
        sample_width=8,
        sample_height=8,
        sample_frames=5,
        patch_size=2,
        temporal_compression_ratio=4,
        max_text_seq_length=8,
    )


def test_cogvideox_policy_keeps_layernorm_zero_projections_at_int4_rtn():
    pytest.importorskip("diffusers")
    model = _tiny_cogvideox()
    config = OrbitQuantConfig(target_policy="auto")

    decisions = classify_linear_modules(model, config)

    assert resolve_target_policy(model, config) == "cogvideox"
    assert decisions["transformer_blocks.0.norm1.linear"].action == "adaln_int4_rtn"
    assert decisions["transformer_blocks.0.norm2.linear"].action == "adaln_int4_rtn"
    for name in (
        "transformer_blocks.0.attn1.to_q",
        "transformer_blocks.0.attn1.to_k",
        "transformer_blocks.0.attn1.to_v",
        "transformer_blocks.0.attn1.to_out.0",
        "transformer_blocks.0.ff.net.0.proj",
        "transformer_blocks.0.ff.net.2",
    ):
        assert decisions[name].action == "orbitquant"
    for name in (
        "patch_embed.text_proj",
        "time_embedding.linear_1",
        "time_embedding.linear_2",
        "norm_out.linear",
        "proj_out",
    ):
        assert decisions[name].action == "bf16_skip"


def test_cogvideox_auto_policy_stays_universal_for_version_1_artifacts():
    pytest.importorskip("diffusers")
    model = _tiny_cogvideox()
    config = OrbitQuantConfig.from_dict({"weight_bits": 4, "activation_bits": 4})

    assert resolve_target_policy(model, config) == "universal"
    decisions = classify_linear_modules(model, config)
    assert decisions["transformer_blocks.0.norm1.linear"].action == "orbitquant"


def test_hunyuan_video_policy_matches_current_diffusers_tiny_module_names():
    pytest.importorskip("diffusers")
    from diffusers import HunyuanVideoTransformer3DModel

    model = HunyuanVideoTransformer3DModel(
        in_channels=4,
        out_channels=4,
        num_attention_heads=2,
        attention_head_dim=8,
        num_layers=1,
        num_single_layers=1,
        num_refiner_layers=1,
        patch_size=1,
        patch_size_t=1,
        text_embed_dim=16,
        pooled_projection_dim=8,
        rope_axes_dim=(2, 2, 4),
    )
    config = OrbitQuantConfig(target_policy="auto")

    decisions = classify_linear_modules(model, config)

    assert resolve_target_policy(model, config) == "hunyuan_video"
    for name in (
        "transformer_blocks.0.norm1.linear",
        "transformer_blocks.0.norm1_context.linear",
        "single_transformer_blocks.0.norm.linear",
    ):
        assert decisions[name].action == "adaln_int4_rtn"
    for name in (
        "transformer_blocks.0.attn.to_q",
        "transformer_blocks.0.attn.add_k_proj",
        "transformer_blocks.0.attn.to_add_out",
        "transformer_blocks.0.ff_context.net.2",
        "single_transformer_blocks.0.proj_mlp",
        "single_transformer_blocks.0.proj_out",
    ):
        assert decisions[name].action == "orbitquant"
    for name in (
        "context_embedder.token_refiner.refiner_blocks.0.attn.to_q",
        "time_text_embed.guidance_embedder.linear_1",
        "norm_out.linear",
        "proj_out",
    ):
        assert decisions[name].action == "bf16_skip"


def test_universal_policy_keeps_sd3_and_pixart_modulation_at_int4_rtn():
    pytest.importorskip("diffusers")
    from diffusers import PixArtTransformer2DModel, SD3Transformer2DModel

    sd3 = SD3Transformer2DModel(
        sample_size=8,
        patch_size=2,
        in_channels=4,
        num_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=16,
        caption_projection_dim=16,
        pooled_projection_dim=8,
        out_channels=4,
    )
    pixart = PixArtTransformer2DModel(
        sample_size=8,
        num_layers=1,
        patch_size=2,
        attention_head_dim=8,
        num_attention_heads=2,
        in_channels=4,
        out_channels=4,
        cross_attention_dim=16,
        caption_channels=16,
    )

    sd3_decisions = classify_linear_modules(sd3, OrbitQuantConfig())
    pixart_decisions = classify_linear_modules(pixart, OrbitQuantConfig())

    assert sd3_decisions["transformer_blocks.0.norm1.linear"].action == "adaln_int4_rtn"
    assert sd3_decisions["transformer_blocks.0.norm1_context.linear"].action == "adaln_int4_rtn"
    assert sd3_decisions["norm_out.linear"].action == "bf16_skip"
    assert pixart_decisions["adaln_single.linear"].action == "adaln_int4_rtn"
    assert pixart_decisions["caption_projection.linear_1"].action == "bf16_skip"
    assert pixart_decisions["transformer_blocks.0.attn2.to_k"].action == "orbitquant"
