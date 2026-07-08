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
    "embedder",
    "time_text_embed",
    "time_guidance_embed",
    "condition_embedder",
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

_BOUNDARY_SKIP_TOKENS = ("proj_out", "norm_out", "final", "final_layer", "unpatchify")

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

_MODULATION_TOKENS = ("adaln", "modulation")


@dataclass(frozen=True)
class PolicyRules:
    block_tokens: tuple[str, ...] = _BLOCK_TOKENS
    modulation_tokens: tuple[str, ...] = _MODULATION_TOKENS
    modulation_scopes: tuple[str, ...] = _BLOCK_TOKENS
    top_level_modulation_tokens: tuple[str, ...] = ()
    hard_skip_tokens: tuple[str, ...] = _HARD_SKIP_TOKENS
    boundary_skip_tokens: tuple[str, ...] = _BOUNDARY_SKIP_TOKENS
    projection_tokens: tuple[str, ...] = ()


_POLICY_RULES: dict[str, PolicyRules] = {
    "auto": PolicyRules(),
    "generic_dit": PolicyRules(),
    "flux": PolicyRules(
        block_tokens=("transformer_blocks", "single_transformer_blocks"),
        modulation_tokens=("adaln", "modulation", "norm1_context", "norm1", ".norm.linear"),
        modulation_scopes=("transformer_blocks", "single_transformer_blocks"),
        projection_tokens=(
            "to_q",
            "to_k",
            "to_v",
            "to_out",
            "add_q_proj",
            "add_k_proj",
            "add_v_proj",
            "to_add_out",
            "proj_mlp",
            "proj_out",
            "ff",
            "ff_context",
        ),
    ),
    "flux2": PolicyRules(
        block_tokens=("transformer_blocks", "single_transformer_blocks"),
        modulation_scopes=("transformer_blocks", "single_transformer_blocks"),
        top_level_modulation_tokens=(
            "double_stream_modulation_img",
            "double_stream_modulation_txt",
            "single_stream_modulation",
        ),
        projection_tokens=(
            "to_q",
            "to_k",
            "to_v",
            "to_out",
            "add_q_proj",
            "add_k_proj",
            "add_v_proj",
            "to_add_out",
            "to_qkv_mlp_proj",
            "linear_in",
            "linear_out",
        ),
    ),
    "z_image": PolicyRules(
        block_tokens=("noise_refiner", "context_refiner", "layers", "transformer_blocks"),
        modulation_scopes=("noise_refiner", "context_refiner", "layers", "transformer_blocks"),
        projection_tokens=(
            "attention",
            "feed_forward",
            "to_q",
            "to_k",
            "to_v",
            "to_out",
            "w1",
            "w2",
            "w3",
        ),
    ),
    "wan": PolicyRules(
        block_tokens=("blocks",),
        modulation_tokens=(),
        modulation_scopes=(),
        projection_tokens=("attn1", "attn2", "ffn", "to_q", "to_k", "to_v", "to_out"),
    ),
}


def _dtype_override_for_module(name: str, config: OrbitQuantConfig) -> str | None:
    for dtype_name, module_patterns in config.modules_dtype_dict.items():
        if any(pattern in name for pattern in module_patterns):
            return dtype_name
    return None


def _contains_any(value: str, tokens: tuple[str, ...]) -> bool:
    return any(token in value for token in tokens)


def _path_components(value: str) -> tuple[str, ...]:
    return tuple(part for part in value.split(".") if part)


def _contains_path_component(value: str, tokens: tuple[str, ...]) -> bool:
    components = set(_path_components(value))
    return any(token in components for token in tokens)


def _contains_component_substring(value: str, tokens: tuple[str, ...]) -> bool:
    return any(token in component for component in _path_components(value) for token in tokens)


def _contains_component_or_path_substring(value: str, tokens: tuple[str, ...]) -> bool:
    components = _path_components(value)
    for token in tokens:
        if "." in token:
            if token in value:
                return True
        elif any(token in component for component in components):
            return True
    return False


def resolve_target_policy(model: torch.nn.Module, config: OrbitQuantConfig) -> str:
    if config.target_policy != "auto":
        return config.target_policy

    class_name = model.__class__.__name__.lower()
    if "flux2" in class_name:
        return "flux2"
    if "zimage" in class_name or "z_image" in class_name:
        return "z_image"
    if "wantransformer" in class_name or class_name.startswith("wan"):
        return "wan"
    if "flux" in class_name:
        return "flux"
    return "generic_dit"


def _policy_rules(model: torch.nn.Module, config: OrbitQuantConfig) -> PolicyRules:
    return _POLICY_RULES.get(resolve_target_policy(model, config), _POLICY_RULES["generic_dit"])


def _is_top_level_modulation(lowered: str, rules: PolicyRules) -> bool:
    return _contains_path_component(lowered, rules.top_level_modulation_tokens)


def _is_scoped_modulation(lowered: str, rules: PolicyRules) -> bool:
    return _contains_path_component(
        lowered, rules.modulation_scopes
    ) and _contains_component_or_path_substring(
        lowered,
        rules.modulation_tokens,
    )


def _is_transformer_projection(lowered: str, rules: PolicyRules) -> bool:
    if not _contains_path_component(lowered, rules.block_tokens):
        return False
    return not rules.projection_tokens or _contains_path_component(
        lowered, rules.projection_tokens
    )


def classify_linear_modules(
    model: torch.nn.Module, config: OrbitQuantConfig
) -> dict[str, PolicyDecision]:
    decisions: dict[str, PolicyDecision] = {}
    explicit_skips = tuple(config.modules_to_not_convert)
    rules = _policy_rules(model, config)

    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        lowered = name.lower()
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
        elif _contains_any(lowered, rules.hard_skip_tokens):
            decisions[name] = PolicyDecision(
                name, "bf16_skip", "embedding, timestep, or non-denoiser module"
            )
        elif _contains_any(lowered, rules.boundary_skip_tokens) and not _is_transformer_projection(
            lowered, rules
        ):
            decisions[name] = PolicyDecision(
                name, "bf16_skip", "non-transformer or boundary module"
            )
        elif _is_top_level_modulation(lowered, rules) or _is_scoped_modulation(lowered, rules):
            decisions[name] = PolicyDecision(
                name, "adaln_int4_rtn", "dynamic modulation projection"
            )
        elif _is_transformer_projection(lowered, rules):
            decisions[name] = PolicyDecision(name, "orbitquant", "transformer block linear")
        elif _contains_any(lowered, rules.boundary_skip_tokens):
            decisions[name] = PolicyDecision(
                name, "bf16_skip", "non-transformer or boundary module"
            )
        else:
            decisions[name] = PolicyDecision(
                name, "bf16_skip", "outside known transformer block policy"
            )

    return decisions
