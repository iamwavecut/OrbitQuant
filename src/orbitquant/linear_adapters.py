from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

WeightLayout = Literal["out_in", "in_out"]


@dataclass(frozen=True)
class LinearAdapter:
    module_type: type[nn.Module]
    weight_layout: WeightLayout = "out_in"
    in_features_attr: str = "in_features"
    out_features_attr: str = "out_features"
    name: str | None = None


@dataclass(frozen=True)
class LinearModuleSpec:
    in_features: int
    out_features: int
    weight_layout: WeightLayout
    adapter_name: str


_ADAPTERS: list[LinearAdapter] = []


def register_linear_adapter(
    module_type: type[nn.Module],
    *,
    weight_layout: WeightLayout = "out_in",
    in_features_attr: str = "in_features",
    out_features_attr: str = "out_features",
    name: str | None = None,
) -> None:
    """Register a linear-compatible module whose forward is equivalent to ``F.linear``.

    ``weight_layout`` describes the source parameter. The replacement always
    stores the canonical PyTorch ``[out_features, in_features]`` layout.
    """

    if not isinstance(module_type, type) or not issubclass(module_type, nn.Module):
        raise TypeError("module_type must be a torch.nn.Module subclass")
    if weight_layout not in {"out_in", "in_out"}:
        raise ValueError("weight_layout must be 'out_in' or 'in_out'")
    adapter = LinearAdapter(
        module_type=module_type,
        weight_layout=weight_layout,
        in_features_attr=in_features_attr,
        out_features_attr=out_features_attr,
        name=name,
    )
    _ADAPTERS[:] = [item for item in _ADAPTERS if item.module_type is not module_type]
    _ADAPTERS.append(adapter)


def _adapter_for_module(module: nn.Module) -> LinearAdapter | None:
    for adapter in reversed(_ADAPTERS):
        if isinstance(module, adapter.module_type):
            return adapter
    return None


def linear_module_spec(module: nn.Module) -> LinearModuleSpec | None:
    adapter = _adapter_for_module(module)
    if adapter is None:
        return None
    try:
        in_features = int(getattr(module, adapter.in_features_attr))
        out_features = int(getattr(module, adapter.out_features_attr))
        weight = module.weight
    except (AttributeError, TypeError, ValueError):
        return None
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        return None
    expected_shape = (
        (out_features, in_features)
        if adapter.weight_layout == "out_in"
        else (in_features, out_features)
    )
    if tuple(weight.shape) != expected_shape:
        return None
    return LinearModuleSpec(
        in_features=in_features,
        out_features=out_features,
        weight_layout=adapter.weight_layout,
        adapter_name=adapter.name or adapter.module_type.__name__,
    )


def is_linear_module(module: nn.Module) -> bool:
    return linear_module_spec(module) is not None


def is_unregistered_linear_candidate(module: nn.Module) -> bool:
    if is_linear_module(module) or isinstance(module, nn.Embedding):
        return False
    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        return False
    class_name = type(module).__name__.lower()
    return "linear" in class_name or (
        hasattr(module, "in_features") and hasattr(module, "out_features")
    )


def canonical_linear_weight(module: nn.Module) -> torch.Tensor:
    spec = linear_module_spec(module)
    if spec is None:
        raise TypeError(f"no OrbitQuant linear adapter registered for {type(module).__name__}")
    weight = module.weight
    if spec.weight_layout == "in_out":
        return weight.transpose(0, 1).contiguous()
    return weight


register_linear_adapter(nn.Linear)

try:
    from transformers.pytorch_utils import Conv1D
except Exception:
    pass
else:
    register_linear_adapter(
        Conv1D,
        weight_layout="in_out",
        in_features_attr="nx",
        out_features_attr="nf",
        name="transformers.Conv1D",
    )


__all__ = [
    "LinearAdapter",
    "LinearModuleSpec",
    "WeightLayout",
    "canonical_linear_weight",
    "is_linear_module",
    "is_unregistered_linear_candidate",
    "linear_module_spec",
    "register_linear_adapter",
]
