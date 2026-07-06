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
        self.proj_out = torch.nn.Linear(16, 8)


def test_flux_single_block_proj_out_is_quantized_but_final_proj_out_is_skipped():
    decisions = classify_linear_modules(
        TinyFluxSingleBlock(), OrbitQuantConfig(target_policy="flux")
    )

    assert decisions["single_transformer_blocks.0.proj_mlp"].action == "orbitquant"
    assert decisions["single_transformer_blocks.0.proj_out"].action == "orbitquant"
    assert decisions["proj_out"].action == "bf16_skip"


class TinyFlux2Names(torch.nn.Module):
    def __init__(self):
        super().__init__()
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
        self.final_layer = torch.nn.ModuleDict({"linear": torch.nn.Linear(16, 8)})


def test_z_image_policy_covers_attention_ffn_and_adaln_but_skips_final_layer():
    decisions = classify_linear_modules(
        TinyZImageNames(), OrbitQuantConfig(target_policy="z_image")
    )

    assert decisions["transformer_blocks.0.attention.to_q"].action == "orbitquant"
    assert decisions["transformer_blocks.0.feed_forward.net.0"].action == "orbitquant"
    assert decisions["transformer_blocks.0.adaLN_modulation.0"].action == "adaln_int4_rtn"
    assert decisions["final_layer.linear"].action == "bf16_skip"


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
