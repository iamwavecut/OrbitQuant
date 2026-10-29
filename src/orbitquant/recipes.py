from __future__ import annotations

from typing import Any

from orbitquant.config import OrbitQuantConfig

_RECIPES: dict[str, tuple[int, int]] = {
    "w2a3": (2, 3),
    "w2a4": (2, 4),
    "w3a3": (3, 3),
    "w4a4": (4, 4),
    "w4a6": (4, 6),
    "w4a8": (4, 8),
}
# Recipe fields beyond the bit widths. W4A8 uses per-token absmax INT8
# activations: every fast CUDA path is an INT8 GEMM, so 8-bit activations cost
# no GEMM time over 4-bit ones.
_RECIPE_FIELDS: dict[str, dict[str, Any]] = {
    "w4a8": {"activation_quantizer": "absmax_int8"},
}


def recipe(name: str, **overrides: Any) -> OrbitQuantConfig:
    """Create a universal OrbitQuant config from a named bit profile."""

    normalized_name = name.lower().replace("-", "")
    try:
        weight_bits, activation_bits = _RECIPES[normalized_name]
    except KeyError as exc:
        raise ValueError(f"unknown recipe {name!r}; choose one of {sorted(_RECIPES)}") from exc
    values = {
        "weight_bits": weight_bits,
        "activation_bits": activation_bits,
        "target_policy": "universal",
        **_RECIPE_FIELDS.get(normalized_name, {}),
    }
    values.update(overrides)
    return OrbitQuantConfig(**values)


__all__ = ["recipe"]
