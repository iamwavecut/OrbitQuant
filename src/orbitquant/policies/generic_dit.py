from __future__ import annotations

from dataclasses import dataclass

import torch

from orbitquant.config import OrbitQuantConfig


@dataclass(frozen=True)
class PolicyDecision:
    name: str
    action: str
    reason: str
    dtype: str | None = None


_HARD_SKIP_TOKENS = (
    "embed",
    "embedding",
    "time_text_embed",
    "time_in",
    "time_out",
    "timestep",
    "t_embedder",
    "time_proj",
    "vae",
    "text_encoder",
    "scheduler",
    "image_processor",
    "safety_checker",
)

_BOUNDARY_SKIP_TOKENS = ("proj_out", "final", "unpatchify")

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


def _dtype_override_for_module(name: str, config: OrbitQuantConfig) -> str | None:
    for dtype_name, module_patterns in config.modules_dtype_dict.items():
        if any(pattern in name for pattern in module_patterns):
            return dtype_name
    return None


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
        dtype_override = _dtype_override_for_module(name, config)
        if dtype_override is not None:
            decisions[name] = PolicyDecision(
                name,
                "bf16_skip",
                "explicit modules_dtype_dict override",
                dtype=dtype_override,
            )
        elif explicit_skips and any(token in name for token in explicit_skips):
            decisions[name] = PolicyDecision(name, "bf16_skip", "explicit modules_to_not_convert")
        elif any(token in lowered for token in _HARD_SKIP_TOKENS):
            decisions[name] = PolicyDecision(
                name, "bf16_skip", "embedding, timestep, or non-denoiser module"
            )
        elif any(token in lowered for token in _MODULATION_TOKENS):
            decisions[name] = PolicyDecision(
                name, "adaln_int4_rtn", "dynamic modulation projection"
            )
        elif in_transformer_block:
            decisions[name] = PolicyDecision(name, "orbitquant", "transformer block linear")
        elif any(token in lowered for token in _BOUNDARY_SKIP_TOKENS):
            decisions[name] = PolicyDecision(
                name, "bf16_skip", "non-transformer or boundary module"
            )
        else:
            decisions[name] = PolicyDecision(
                name, "bf16_skip", "outside known transformer block policy"
            )

    return decisions
