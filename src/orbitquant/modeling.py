from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from orbitquant.adaln import RTNInt4Linear
from orbitquant.config import OrbitQuantConfig
from orbitquant.kernels import available_backends
from orbitquant.layers import OrbitQuantLinear
from orbitquant.policies import PolicyDecision, classify_linear_modules

_TORCH_DTYPE_BY_NAME = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


@dataclass
class QuantizationSummary:
    quantized_modules: list[str] = field(default_factory=list)
    adaln_modules: list[str] = field(default_factory=list)
    skipped_modules: list[str] = field(default_factory=list)
    quantization_device: str = "auto"
    weight_quantization_backend: str = "torch_reference"
    quantization_staging_mode: str = "streaming"
    elapsed_seconds: float = 0.0
    orbitquant_seconds: float = 0.0
    adaln_seconds: float = 0.0
    device_transfer_seconds: float = 0.0
    module_device_transfer_count: int = 0
    source_linear_device_counts: dict[str, int] = field(default_factory=dict)
    quantized_buffer_device_counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class QuantizationPrewarmSummary:
    orbitquant_modules: int
    adaln_modules: int
    total_modules: int
    elapsed_seconds: float
    device: str
    dtype: str


def _parent_and_child(model: torch.nn.Module, module_name: str) -> tuple[torch.nn.Module, str]:
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        if part.isdigit() and isinstance(parent, torch.nn.ModuleList | torch.nn.Sequential):
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    return parent, parts[-1]


def _set_child(parent: torch.nn.Module, child_name: str, module: torch.nn.Module) -> None:
    if child_name.isdigit() and isinstance(parent, torch.nn.ModuleList | torch.nn.Sequential):
        parent[int(child_name)] = module
    elif isinstance(parent, torch.nn.ModuleDict):
        parent[child_name] = module
    else:
        setattr(parent, child_name, module)


def _apply_dtype_override(module: torch.nn.Module, decision: PolicyDecision) -> None:
    if decision.dtype is None:
        return
    module.to(dtype=_TORCH_DTYPE_BY_NAME[decision.dtype])


def _auto_quantization_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _quantization_device(device: str | torch.device | None) -> torch.device | None:
    if device is None:
        return None
    if device == "auto":
        return _auto_quantization_device()
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA quantization device requested but CUDA is not available")
    if torch_device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS quantization device requested but MPS is not available")
    return torch_device


def _weight_quantization_backend(device: torch.device | None) -> str:
    if device is None:
        return "module_device"
    if device.type == "cuda":
        if not available_backends()["triton_cuda"]:
            raise RuntimeError(
                "CUDA quantization requires the Triton CUDA backend. Install triton "
                "or use --device cpu for the reference quantization path."
            )
        return "triton_cuda"
    if device.type == "mps":
        return "torch_reference_mps"
    return "torch_reference"


def _count_tensor_device(summary: QuantizationSummary, tensor: torch.Tensor | None) -> None:
    if tensor is None:
        return
    key = str(tensor.device)
    summary.quantized_buffer_device_counts[key] = (
        summary.quantized_buffer_device_counts.get(key, 0) + 1
    )


def _record_orbitquant_buffers(summary: QuantizationSummary, module: OrbitQuantLinear) -> None:
    _count_tensor_device(summary, module.packed_weight_indices)
    _count_tensor_device(summary, module.row_norms)
    _count_tensor_device(summary, module.debug_weight)
    if module.bias is not None:
        _count_tensor_device(summary, module.bias)


def _record_adaln_buffers(summary: QuantizationSummary, module: RTNInt4Linear) -> None:
    _count_tensor_device(summary, module.packed_weight)
    _count_tensor_device(summary, module.scales)
    if module.bias is not None:
        _count_tensor_device(summary, module.bias)


def _record_source_linear_device(summary: QuantizationSummary, module: torch.nn.Linear) -> None:
    key = str(module.weight.device)
    summary.source_linear_device_counts[key] = summary.source_linear_device_counts.get(key, 0) + 1


def _move_module_to_device(
    module: torch.nn.Module,
    target_device: torch.device,
    summary: QuantizationSummary,
) -> None:
    if _first_recursive_tensor_device(module) == target_device:
        return
    transfer_started_at = time.perf_counter()
    module.to(device=target_device)
    _synchronize_if_needed(target_device)
    summary.device_transfer_seconds += time.perf_counter() - transfer_started_at
    summary.module_device_transfer_count += 1


def quantize_linear_modules(
    model: torch.nn.Module,
    config: OrbitQuantConfig,
    *,
    quantization_device: str | torch.device | None = "auto",
    staging_mode: str = "streaming",
) -> QuantizationSummary:
    if staging_mode not in {"streaming", "component"}:
        raise ValueError("staging_mode must be 'streaming' or 'component'")

    decisions = classify_linear_modules(model, config)
    target_device = _quantization_device(quantization_device)
    summary = QuantizationSummary(
        quantization_device="preserve" if target_device is None else str(target_device),
        weight_quantization_backend=_weight_quantization_backend(target_device),
        quantization_staging_mode=staging_mode,
    )
    started_at = time.perf_counter()

    for name in decisions:
        module = model.get_submodule(name)
        if isinstance(module, torch.nn.Linear):
            _record_source_linear_device(summary, module)

    if target_device is not None and staging_mode == "component":
        _move_module_to_device(model, target_device, summary)

    for name, decision in decisions.items():
        module = model.get_submodule(name)
        if not isinstance(module, torch.nn.Linear):
            continue
        if decision.action == "orbitquant":
            if target_device is not None and staging_mode == "streaming":
                _move_module_to_device(module, target_device, summary)
            module_started_at = time.perf_counter()
            replacement = OrbitQuantLinear.from_linear(module, config=config, module_name=name)
            _synchronize_if_needed(_first_tensor_device(replacement))
            summary.orbitquant_seconds += time.perf_counter() - module_started_at
            _record_orbitquant_buffers(summary, replacement)
            parent, child_name = _parent_and_child(model, name)
            _set_child(parent, child_name, replacement)
            summary.quantized_modules.append(name)
        elif decision.action == "adaln_int4_rtn":
            if target_device is not None and staging_mode == "streaming":
                _move_module_to_device(module, target_device, summary)
            module_started_at = time.perf_counter()
            replacement = RTNInt4Linear.from_linear(module, config=config, module_name=name)
            _synchronize_if_needed(_first_tensor_device(replacement))
            summary.adaln_seconds += time.perf_counter() - module_started_at
            _record_adaln_buffers(summary, replacement)
            parent, child_name = _parent_and_child(model, name)
            _set_child(parent, child_name, replacement)
            summary.adaln_modules.append(name)
        else:
            _apply_dtype_override(module, decision)
            summary.skipped_modules.append(name)

    if target_device is not None:
        _synchronize_if_needed(target_device)
    summary.elapsed_seconds = time.perf_counter() - started_at
    return summary


def prepare_prequantized_linear_modules(
    model: torch.nn.Module, config: OrbitQuantConfig
) -> QuantizationSummary:
    decisions = classify_linear_modules(model, config)
    summary = QuantizationSummary()

    for name, decision in decisions.items():
        module = model.get_submodule(name)
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
            _apply_dtype_override(module, decision)
            summary.skipped_modules.append(name)

    return summary


def _first_tensor_device(module: torch.nn.Module) -> torch.device:
    for parameter in module.parameters(recurse=False):
        return parameter.device
    for buffer in module.buffers(recurse=False):
        return buffer.device
    return torch.device("cpu")


def _first_recursive_tensor_device(module: torch.nn.Module) -> torch.device:
    for parameter in module.parameters(recurse=True):
        return parameter.device
    for buffer in module.buffers(recurse=True):
        return buffer.device
    return torch.device("cpu")


def _first_floating_tensor_dtype(module: torch.nn.Module) -> torch.dtype:
    for parameter in module.parameters(recurse=False):
        if parameter.is_floating_point():
            return parameter.dtype
    for buffer in module.buffers(recurse=False):
        if buffer.is_floating_point():
            return buffer.dtype
    return torch.bfloat16


def _synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif device.type == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


def prewarm_quantized_linear_modules(
    model: torch.nn.Module,
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> QuantizationPrewarmSummary:
    """Materialize lazy dequantized weight caches for quantized linear modules."""

    target_device = None if device is None else _quantization_device(device)
    started_at = time.perf_counter()
    orbitquant_modules = 0
    adaln_modules = 0
    last_device = target_device or torch.device("cpu")
    last_dtype = dtype or torch.bfloat16
    synced_devices: dict[str, torch.device] = {}

    for module in model.modules():
        if not isinstance(module, OrbitQuantLinear | RTNInt4Linear):
            continue
        module_device = target_device or _first_tensor_device(module)
        module_dtype = dtype or _first_floating_tensor_dtype(module)
        device_key = str(module_device)
        if device_key not in synced_devices:
            _synchronize_if_needed(module_device)
            synced_devices[device_key] = module_device
        module._dequantize_weight(device=module_device, dtype=module_dtype)
        last_device = module_device
        last_dtype = module_dtype
        if isinstance(module, OrbitQuantLinear):
            orbitquant_modules += 1
        else:
            adaln_modules += 1

    for module_device in synced_devices.values():
        _synchronize_if_needed(module_device)

    elapsed_seconds = time.perf_counter() - started_at
    return QuantizationPrewarmSummary(
        orbitquant_modules=orbitquant_modules,
        adaln_modules=adaln_modules,
        total_modules=orbitquant_modules + adaln_modules,
        elapsed_seconds=elapsed_seconds,
        device=str(last_device),
        dtype=str(last_dtype).removeprefix("torch."),
    )


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
