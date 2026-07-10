from __future__ import annotations

from typing import Any

from orbitquant.config import OrbitQuantConfig

_RECIPES: dict[str, tuple[int, int]] = {
    "w2a3": (2, 3),
    "w2a4": (2, 4),
    "w3a3": (3, 3),
    "w4a4": (4, 4),
    "w4a6": (4, 6),
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
    }
    values.update(overrides)
    return OrbitQuantConfig(**values)


__all__ = ["recipe"]
