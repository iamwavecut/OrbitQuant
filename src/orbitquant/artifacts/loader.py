from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from orbitquant.adaln import RTNInt4Linear
from orbitquant.artifacts.checksums import validate_checksums
from orbitquant.artifacts.manifest import OrbitQuantManifest
from orbitquant.artifacts.validator import validate_required_artifact_files
from orbitquant.config import OrbitQuantConfig
from orbitquant.layers import OrbitQuantLinear
from orbitquant.modeling import _parent_and_child, _set_child


def _get_module(model: torch.nn.Module, module_name: str) -> torch.nn.Module:
    module = model
    for part in module_name.split("."):
        if part.isdigit() and isinstance(module, (torch.nn.ModuleList, torch.nn.Sequential)):
            module = module[int(part)]
        elif isinstance(module, torch.nn.ModuleDict):
            module = module[part]
        else:
            module = getattr(module, part)
    return module


def load_orbitquant_artifact(
    model: torch.nn.Module,
    artifact_dir: str | Path,
    *,
    strict: bool = True,
) -> OrbitQuantManifest:
    artifact_path = Path(artifact_dir)
    validate_required_artifact_files(artifact_path)
    config = OrbitQuantConfig.from_dict(
        json.loads((artifact_path / "quantization_config.json").read_text(encoding="utf-8"))
    )
    manifest = OrbitQuantManifest.from_dict(
        json.loads((artifact_path / "orbitquant_manifest.json").read_text(encoding="utf-8"))
    )
    validate_checksums(artifact_path, manifest.checksums)

    for name in manifest.quantized_modules:
        module = _get_module(model, name)
        if not isinstance(module, torch.nn.Linear):
            raise TypeError(f"expected Linear at {name}, got {type(module).__name__}")
        replacement = OrbitQuantLinear.empty_from_linear(module, config=config, module_name=name)
        parent, child_name = _parent_and_child(model, name)
        _set_child(parent, child_name, replacement)

    for name in manifest.adaln_modules:
        module = _get_module(model, name)
        if not isinstance(module, torch.nn.Linear):
            raise TypeError(f"expected Linear at {name}, got {type(module).__name__}")
        replacement = RTNInt4Linear.empty_from_linear(module, config=config, module_name=name)
        parent, child_name = _parent_and_child(model, name)
        _set_child(parent, child_name, replacement)

    state_dict = load_file(artifact_path / "model.safetensors")
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if strict and (missing or unexpected):
        raise RuntimeError(f"artifact state mismatch: missing={missing}, unexpected={unexpected}")
    return manifest
