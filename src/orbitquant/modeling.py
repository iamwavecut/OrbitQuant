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


def _first_tensor_device(module: torch.nn.Module) -> torch.device:
    for parameter in module.parameters(recurse=False):
        return parameter.device
    for buffer in module.buffers(recurse=False):
        return buffer.device
    return torch.device("cpu")


def _frozen_linear_from_weight(
    *,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    in_features: int,
    out_features: int,
    device: torch.device,
) -> torch.nn.Linear:
    replacement = torch.nn.Linear(
        in_features,
        out_features,
        bias=bias is not None,
        device=device,
        dtype=torch.float32,
    )
    with torch.no_grad():
        replacement.weight.copy_(weight.to(device=device, dtype=torch.float32))
        if bias is not None and replacement.bias is not None:
            replacement.bias.copy_(bias.to(device=device, dtype=torch.float32))
    replacement.requires_grad_(False)
    return replacement


def dequantize_quantized_linear_modules(model: torch.nn.Module) -> torch.nn.Module:
    for name, module in list(model.named_modules()):
        if isinstance(module, OrbitQuantLinear):
            device = _first_tensor_device(module)
            rotated_weight = module._dequantize_weight(device=device, dtype=torch.float32)
            weight = module.rotation.apply_inverse_to_weight(rotated_weight)
            bias = None if module.bias is None else module.bias.detach()
            replacement = _frozen_linear_from_weight(
                weight=weight,
                bias=bias,
                in_features=module.in_features,
                out_features=module.out_features,
                device=device,
            )
            parent, child_name = _parent_and_child(model, name)
            _set_child(parent, child_name, replacement)
        elif isinstance(module, RTNInt4Linear):
            device = _first_tensor_device(module)
            weight = module._dequantize_weight(device=device, dtype=torch.float32)
            bias = None if module.bias is None else module.bias.detach()
            replacement = _frozen_linear_from_weight(
                weight=weight,
                bias=bias,
                in_features=module.in_features,
                out_features=module.out_features,
                device=device,
            )
            parent, child_name = _parent_and_child(model, name)
            _set_child(parent, child_name, replacement)
    return model
