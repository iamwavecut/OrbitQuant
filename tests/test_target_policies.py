import pytest
import torch

from orbitquant.config import OrbitQuantConfig
from orbitquant.policies import classify_linear_modules


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


class TinyWanNames(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {
                        "attn1": torch.nn.ModuleDict({"to_q": torch.nn.Linear(16, 16)}),
                        "attn2": torch.nn.ModuleDict({"add_k_proj": torch.nn.Linear(16, 16)}),
                        "ffn": torch.nn.ModuleDict(
                            {"net": torch.nn.ModuleList([torch.nn.Linear(16, 32)])}
                        ),
                    }
                )
            ]
        )
        self.time_proj = torch.nn.Linear(16, 16)


def test_wan_policy_covers_self_cross_attention_and_ffn_but_skips_time_projection():
    decisions = classify_linear_modules(TinyWanNames(), OrbitQuantConfig(target_policy="wan"))

    assert decisions["blocks.0.attn1.to_q"].action == "orbitquant"
    assert decisions["blocks.0.attn2.add_k_proj"].action == "orbitquant"
    assert decisions["blocks.0.ffn.net.0"].action == "orbitquant"
    assert decisions["time_proj"].action == "bf16_skip"


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
    assert decisions["single_transformer_blocks.0.proj_out"].action == "orbitquant"
    assert decisions["norm_out.linear"].action == "bf16_skip"
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
    assert decisions["blocks.0.attn2.to_q"].action == "orbitquant"
    assert decisions["blocks.0.ffn.net.0.proj"].action == "orbitquant"
    assert decisions["proj_out"].action == "bf16_skip"
