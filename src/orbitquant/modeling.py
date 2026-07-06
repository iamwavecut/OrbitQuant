from __future__ import annotations

from dataclasses import dataclass, field

import torch

from orbitquant.adaln import RTNInt4Linear
from orbitquant.config import OrbitQuantConfig
from orbitquant.layers import OrbitQuantLinear
from orbitquant.policies import classify_linear_modules


@dataclass
class QuantizationSummary:
    quantized_modules: list[str] = field(default_factory=list)
    adaln_modules: list[str] = field(default_factory=list)
    skipped_modules: list[str] = field(default_factory=list)


def _parent_and_child(model: torch.nn.Module, module_name: str) -> tuple[torch.nn.Module, str]:
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        if part.isdigit() and isinstance(parent, (torch.nn.ModuleList, torch.nn.Sequential)):
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    return parent, parts[-1]


def _set_child(parent: torch.nn.Module, child_name: str, module: torch.nn.Module) -> None:
    if child_name.isdigit() and isinstance(parent, (torch.nn.ModuleList, torch.nn.Sequential)):
        parent[int(child_name)] = module
    elif isinstance(parent, torch.nn.ModuleDict):
        parent[child_name] = module
    else:
        setattr(parent, child_name, module)


def quantize_linear_modules(
    model: torch.nn.Module, config: OrbitQuantConfig
) -> QuantizationSummary:
    decisions = classify_linear_modules(model, config)
    modules = dict(model.named_modules())
    summary = QuantizationSummary()

    for name, decision in decisions.items():
        module = modules[name]
        if not isinstance(module, torch.nn.Linear):
            continue
        if decision.action == "orbitquant":
            replacement = OrbitQuantLinear.from_linear(module, config=config, module_name=name)
            parent, child_name = _parent_and_child(model, name)
            _set_child(parent, child_name, replacement)
            summary.quantized_modules.append(name)
        elif decision.action == "adaln_int4_rtn":
            replacement = RTNInt4Linear.from_linear(module, config=config, module_name=name)
            parent, child_name = _parent_and_child(model, name)
            _set_child(parent, child_name, replacement)
            summary.adaln_modules.append(name)
        else:
            summary.skipped_modules.append(name)

    return summary


def prepare_prequantized_linear_modules(
    model: torch.nn.Module, config: OrbitQuantConfig
) -> QuantizationSummary:
    decisions = classify_linear_modules(model, config)
    modules = dict(model.named_modules())
    summary = QuantizationSummary()

    for name, decision in decisions.items():
        module = modules[name]
        if not isinstance(module, torch.nn.Linear):
            continue
        if decision.action == "orbitquant":
            replacement = OrbitQuantLinear.empty_from_linear(
                module, config=config, module_name=name
            )
            parent, child_name = _parent_and_child(model, name)
            _set_child(parent, child_name, replacement)
            summary.quantized_modules.append(name)
        elif decision.action == "adaln_int4_rtn":
            replacement = RTNInt4Linear.empty_from_linear(
                module, config=config, module_name=name
            )
            parent, child_name = _parent_and_child(model, name)
            _set_child(parent, child_name, replacement)
            summary.adaln_modules.append(name)
        else:
            summary.skipped_modules.append(name)

    return summary
