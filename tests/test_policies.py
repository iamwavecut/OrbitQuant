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
