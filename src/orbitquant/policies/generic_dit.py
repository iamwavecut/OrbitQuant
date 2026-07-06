from __future__ import annotations

from dataclasses import dataclass

import torch

from orbitquant.config import OrbitQuantConfig


@dataclass(frozen=True)
class PolicyDecision:
    name: str
    action: str
    reason: str


_SKIP_TOKENS = (
    "embed",
    "embedding",
    "time_text_embed",
    "time_in",
    "time_out",
    "timestep",
    "proj_out",
    "final",
    "unpatchify",
    "vae",
    "text_encoder",
    "scheduler",
    "image_processor",
    "safety_checker",
)

_BLOCK_TOKENS = (
    "transformer_blocks",
    "single_transformer_blocks",
    "blocks",
    "layers",
    "attn",
    "attention",
    "ff",
    "feed_forward",
    "mlp",
)

_MODULATION_TOKENS = ("adaln", "modulation", "mod", "norm1_context", "norm1")


def classify_linear_modules(
    model: torch.nn.Module, config: OrbitQuantConfig
) -> dict[str, PolicyDecision]:
    decisions: dict[str, PolicyDecision] = {}
    explicit_skips = tuple(config.modules_to_not_convert)

    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        lowered = name.lower()
        in_transformer_block = any(token in lowered for token in _BLOCK_TOKENS)
        if explicit_skips and any(token in name for token in explicit_skips):
            decisions[name] = PolicyDecision(name, "bf16_skip", "explicit modules_to_not_convert")
        elif any(token in lowered for token in _MODULATION_TOKENS):
            decisions[name] = PolicyDecision(
                name, "adaln_int4_rtn", "dynamic modulation projection"
            )
        elif in_transformer_block:
            decisions[name] = PolicyDecision(name, "orbitquant", "transformer block linear")
        elif any(token in lowered for token in _SKIP_TOKENS):
            decisions[name] = PolicyDecision(
                name, "bf16_skip", "non-transformer or boundary module"
            )
        else:
            decisions[name] = PolicyDecision(
                name, "bf16_skip", "outside known transformer block policy"
            )

    return decisions
