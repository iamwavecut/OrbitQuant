from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from orbitquant.adaln import RTNInt4Linear
from orbitquant.artifacts.checksums import sha256_file
from orbitquant.artifacts.manifest import OrbitQuantManifest
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


def _validate_manifest_checksums(artifact_path: Path, manifest: OrbitQuantManifest) -> None:
    for relative_path, expected in manifest.checksums.items():
        path = artifact_path / relative_path
        if not path.is_file():
            raise RuntimeError(f"artifact checksum target missing: {relative_path}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"artifact checksum mismatch for {relative_path}: "
                f"expected {expected}, got {actual}"
            )


def load_orbitquant_artifact(
    model: torch.nn.Module,
    artifact_dir: str | Path,
    *,
    strict: bool = True,
) -> OrbitQuantManifest:
    artifact_path = Path(artifact_dir)
    config = OrbitQuantConfig.from_dict(
        json.loads((artifact_path / "quantization_config.json").read_text(encoding="utf-8"))
    )
    manifest = OrbitQuantManifest.from_dict(
        json.loads((artifact_path / "orbitquant_manifest.json").read_text(encoding="utf-8"))
    )
    _validate_manifest_checksums(artifact_path, manifest)

    for name in manifest.quantized_modules:
        module = _get_module(model, name)
        if not isinstance(module, torch.nn.Linear):
            raise TypeError(f"expected Linear at {name}, got {type(module).__name__}")
        replacement = OrbitQuantLinear.from_linear(module, config=config, module_name=name)
        parent, child_name = _parent_and_child(model, name)
        _set_child(parent, child_name, replacement)

    for name in manifest.adaln_modules:
        module = _get_module(model, name)
        if not isinstance(module, torch.nn.Linear):
            raise TypeError(f"expected Linear at {name}, got {type(module).__name__}")
        replacement = RTNInt4Linear.from_linear(module, config=config, module_name=name)
        parent, child_name = _parent_and_child(model, name)
        _set_child(parent, child_name, replacement)

    state_dict = load_file(artifact_path / "model.safetensors")
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if strict and (missing or unexpected):
        raise RuntimeError(f"artifact state mismatch: missing={missing}, unexpected={unexpected}")
    return manifest
