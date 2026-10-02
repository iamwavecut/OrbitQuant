"""Block adapters: which projections each block fuses and the forward that runs them.

An adapter module defines:

* ``FAMILY``: the name recorded in fused checkpoints;
* ``MODEL_CLASSES``: transformer class names it serves;
* ``blocks(model)``: ``(path, kind)`` of every fused block;
* ``groups(kind, block)``: ``{name: GroupSpec}`` for a block of that kind;
* ``forward(kind)``: the replacement ``forward`` (same signature as the original block);
* ``prepare(block, kind)``: per-block settings derived from the loaded weights (optional).
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field

from orbitquant.fused.groups import RowSource

_MODULES = ("krea2", "flux2", "ideogram4", "minimax_h3", "qwenimage21", "boogu")


@dataclass(frozen=True)
class GroupSpec:
    sources: tuple[RowSource, ...]
    swiglu: bool = False
    # Dense (unquantized) sources become per-row INT8 unless the checkpoint asks for BF16.
    dense: str | None = None
    extra: dict = field(default_factory=dict)


# INT8 Q/K quantization loses the small channels of heads whose Q/K RMSNorm scales one channel
# far above the rest (Krea 2 block 0: 36x the median); such blocks keep BF16 attention.
INT8_ATTENTION_MAX_NORM_SPREAD = 3.0


def int8_attention_allowed(*norm_weights, offset: float = 0.0) -> bool:
    """Whether a block's Q/K RMSNorm scales (``weight + offset``) are flat enough for INT8 Q/K."""
    import torch

    scales = torch.cat([w.reshape(-1) for w in norm_weights]).float().add(offset).abs()
    return bool(scales.max() <= INT8_ATTENTION_MAX_NORM_SPREAD * scales.median())


def specs(**groups) -> dict[str, GroupSpec]:
    """``specs(qkv=["attn.to_q", ...], gateup=(["ff.gate", "ff.up"], {"swiglu": True}))``."""
    out = {}
    for name, value in groups.items():
        options = {}
        if isinstance(value, tuple):
            value, options = value
        sources = tuple(s if isinstance(s, RowSource) else RowSource(s) for s in value)
        out[name] = GroupSpec(sources, **options)
    return out


def adapter_for_class(class_name: str):
    for name in _MODULES:
        try:
            module = importlib.import_module(f"orbitquant.fused.families.{name}")
        except ModuleNotFoundError:
            continue
        if class_name in module.MODEL_CLASSES:
            return module
    raise ValueError(f"no fused block adapter for {class_name}")


def adapter_for_family(family: str):
    try:
        module = importlib.import_module(f"orbitquant.fused.families.{family}")
    except ModuleNotFoundError as exc:
        raise ValueError(f"unknown fused family {family!r}") from exc
    return module
