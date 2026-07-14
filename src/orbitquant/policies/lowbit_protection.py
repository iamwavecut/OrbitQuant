"""Mixed-bit protection for very low weight bit widths.

At 2-bit weights every layer quantizes at the scalar Lloyd-Max floor (~34%
relative error), and deep DiT stacks fail cumulatively rather than through a
single culprit. Measured on Ideogram-v4-Instant (34 blocks, 8-step distilled):
quantizing all block projections to W2 collapses the output to noise, while
keeping the first and last 4 blocks intact restores a coherent image even
with more layers quantized overall — the boundary blocks act as error
amplifiers. This module upgrades the weight bit width of boundary-block and
out-of-block projections after policy classification, architecture-agnostic:
block indices are recovered from the module paths themselves.
"""

from __future__ import annotations

import re
from dataclasses import replace

from orbitquant.config import OrbitQuantConfig
from orbitquant.policies.generic_dit import PolicyDecision

_BLOCK_INDEX_RE = re.compile(r"(?:^|\.)(?P<container>[a-zA-Z_][a-zA-Z0-9_]*)\.(?P<index>\d+)\.")
_SEPARATE_QK_LEAVES = {
    "add_k_proj",
    "add_q_proj",
    "k",
    "k_proj",
    "key",
    "key_proj",
    "q",
    "q_proj",
    "query",
    "query_proj",
    "to_k",
    "to_q",
}


def _block_key(name: str) -> tuple[str, int] | None:
    """Return the outermost (container, index) pair of an indexed module path."""
    match = _BLOCK_INDEX_RE.search(name)
    if match is None:
        return None
    return match.group("container"), int(match.group("index"))


def resolve_protected_block_count(config: OrbitQuantConfig) -> int:
    if config.lowbit_boundary_protection == "auto":
        return config.lowbit_protected_blocks if config.weight_bits <= 2 else 0
    return int(config.lowbit_boundary_protection)


def resolve_interior_protection(
    config: OrbitQuantConfig, *, target_policy: str
) -> bool:
    if config.weight_bits != 2:
        return False
    if config.lowbit_interior_protection == "auto":
        return target_policy == "universal"
    return bool(config.lowbit_interior_protection)


def apply_lowbit_boundary_protection(
    decisions: dict[str, PolicyDecision], config: OrbitQuantConfig
) -> dict[str, PolicyDecision]:
    """Upgrade boundary and out-of-block orbitquant decisions to protected bits.

    Only decisions with ``action == "orbitquant"`` are touched, and only when
    the resolved protected block count is positive and the protected bit width
    exceeds ``config.weight_bits``. Returns the same dict for chaining.
    """

    protected_blocks = resolve_protected_block_count(config)
    protected_bits = max(config.lowbit_protected_bits, config.weight_bits)
    if protected_blocks <= 0 or protected_bits == config.weight_bits:
        return decisions

    per_container: dict[str, set[int]] = {}
    keys: dict[str, tuple[str, int] | None] = {}
    for name, decision in decisions.items():
        if decision.action != "orbitquant":
            continue
        key = _block_key(name)
        keys[name] = key
        if key is not None:
            per_container.setdefault(key[0], set()).add(key[1])

    for name, decision in decisions.items():
        if decision.action != "orbitquant":
            continue
        key = keys.get(name)
        if key is None:
            protected = True
            reason = "out-of-block projection"
        else:
            container, index = key
            indices = per_container[container]
            first = min(indices)
            last = max(indices)
            effective_protected_blocks = (
                protected_blocks
                if len(indices) <= 2
                else min(protected_blocks, (len(indices) - 1) // 2)
            )
            protected = (
                index < first + effective_protected_blocks
                or index > last - effective_protected_blocks
            )
            reason = f"boundary block {container}.{index}"
        if protected:
            decisions[name] = replace(
                decision,
                weight_bits=protected_bits,
                reason=(
                    f"{decision.reason}; low-bit boundary protection "
                    f"({reason} upgraded to W{protected_bits})"
                ),
            )
    return decisions


def apply_lowbit_interior_protection(
    decisions: dict[str, PolicyDecision],
    config: OrbitQuantConfig,
    *,
    target_policy: str,
) -> dict[str, PolicyDecision]:
    """Use W3 for sensitive interior projections while retaining separate Q/K at W2."""

    if not resolve_interior_protection(config, target_policy=target_policy):
        return decisions

    for name, decision in decisions.items():
        if decision.action != "orbitquant" or _block_key(name) is None:
            continue
        if decision.weight_bits is not None and decision.weight_bits > config.weight_bits:
            continue
        if name.rsplit(".", 1)[-1].lower() in _SEPARATE_QK_LEAVES:
            continue
        decisions[name] = replace(
            decision,
            weight_bits=3,
            reason=(
                f"{decision.reason}; low-bit interior protection "
                "(non-Q/K projection upgraded to W3)"
            ),
        )
    return decisions


def apply_lowbit_protection(
    decisions: dict[str, PolicyDecision],
    config: OrbitQuantConfig,
    *,
    target_policy: str,
) -> dict[str, PolicyDecision]:
    apply_lowbit_boundary_protection(decisions, config)
    return apply_lowbit_interior_protection(
        decisions,
        config,
        target_policy=target_policy,
    )
