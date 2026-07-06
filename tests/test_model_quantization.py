import torch

from orbitquant import OrbitQuantConfig, OrbitQuantLinear
from orbitquant.adaln import RTNInt4Linear
from orbitquant.modeling import quantize_linear_modules


class TinyPipelineTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {
                        "attn": torch.nn.ModuleDict({"to_q": torch.nn.Linear(16, 16)}),
                        "modulation": torch.nn.Linear(16, 32),
                    }
                )
            ]
        )
        self.proj_out = torch.nn.Linear(16, 16)


def test_quantize_linear_modules_replaces_orbit_and_adaln_targets_only():
    model = TinyPipelineTransformer()
    config = OrbitQuantConfig(block_size=8)

    summary = quantize_linear_modules(model, config)

    assert isinstance(model.transformer_blocks[0]["attn"]["to_q"], OrbitQuantLinear)
    assert isinstance(model.transformer_blocks[0]["modulation"], RTNInt4Linear)
    assert isinstance(model.proj_out, torch.nn.Linear)
    assert summary.quantized_modules == ["transformer_blocks.0.attn.to_q"]
    assert summary.adaln_modules == ["transformer_blocks.0.modulation"]
    assert summary.skipped_modules == ["proj_out"]
