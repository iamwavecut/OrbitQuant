from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from orbitquant.artifacts.checksums import sha256_file, write_sha256sums
from orbitquant.artifacts.manifest import OrbitQuantManifest
from orbitquant.artifacts.model_card import render_model_card
from orbitquant.config import OrbitQuantConfig


def _module_shapes(model: torch.nn.Module) -> dict[str, list[int]]:
    shapes: dict[str, list[int]] = {}
    for name, tensor in model.state_dict().items():
        shapes[name] = list(tensor.shape)
    return shapes


def _summary_list(summary: Any, field: str) -> list[str]:
    value = getattr(summary, field, [])
    return list(value)


def save_orbitquant_artifact(
    model: torch.nn.Module,
    output_dir: str | Path,
    *,
    config: OrbitQuantConfig,
    source_model_id: str,
    source_revision: str,
    source_license: str,
    summary: Any,
) -> OrbitQuantManifest:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tensor_path = output_path / "model.safetensors"
    state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    save_file(state_dict, tensor_path)
    config_path = output_path / "quantization_config.json"
    config_path.write_text(json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8")

    skipped = _summary_list(summary, "skipped_modules")
    checksums = {
        "model.safetensors": sha256_file(tensor_path),
        "quantization_config.json": sha256_file(config_path),
    }
    manifest = OrbitQuantManifest.from_config(
        config,
        source_model_id=source_model_id,
        source_revision=source_revision,
        source_license=source_license,
        quantized_modules=_summary_list(summary, "quantized_modules"),
        adaln_modules=_summary_list(summary, "adaln_modules"),
        skipped_modules=skipped,
        module_shapes=_module_shapes(model),
        checksums=checksums,
    )

    (output_path / "orbitquant_manifest.json").write_text(
        json.dumps(manifest.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    (output_path / "README.md").write_text(render_model_card(manifest), encoding="utf-8")
    write_sha256sums(output_path)
    return manifest
