from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from safetensors.torch import load_file

from orbitquant.artifacts.checksums import validate_checksums
from orbitquant.artifacts.manifest import OrbitQuantManifest
from orbitquant.config import OrbitQuantConfig

_REQUIRED_ARTIFACT_FILES = (
    "README.md",
    "SHA256SUMS",
    "model.safetensors",
    "quantization_config.json",
    "orbitquant_manifest.json",
    "orbitquant_codebooks.safetensors",
    "orbitquant_rotations.safetensors",
    "prompts.json",
    "benchmark/summary.json",
)


def validate_required_artifact_files(artifact_path: Path) -> None:
    missing = [
        relative_path
        for relative_path in _REQUIRED_ARTIFACT_FILES
        if not (artifact_path / relative_path).is_file()
    ]
    if missing:
        raise RuntimeError(f"required artifact file missing: {missing}")


def validate_orbitquant_artifact(artifact_dir: str | Path) -> dict[str, Any]:
    artifact_path = Path(artifact_dir)
    validate_required_artifact_files(artifact_path)
    config = OrbitQuantConfig.from_dict(
        json.loads((artifact_path / "quantization_config.json").read_text(encoding="utf-8"))
    )
    manifest = OrbitQuantManifest.from_dict(
        json.loads((artifact_path / "orbitquant_manifest.json").read_text(encoding="utf-8"))
    )
    validate_checksums(artifact_path, manifest.checksums)
    state_dict = load_file(artifact_path / "model.safetensors")

    expected_shapes = manifest.module_shapes
    missing = sorted(set(expected_shapes) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(expected_shapes))
    shape_mismatches = {
        name: {"expected": expected_shapes[name], "actual": list(state_dict[name].shape)}
        for name in sorted(set(expected_shapes) & set(state_dict))
        if expected_shapes[name] != list(state_dict[name].shape)
    }
    if missing or unexpected or shape_mismatches:
        raise RuntimeError(
            "artifact tensor shape mismatch: "
            f"missing={missing}, unexpected={unexpected}, shape_mismatches={shape_mismatches}"
        )

    return {
        "valid": True,
        "artifact_dir": str(artifact_path),
        "source_model_id": manifest.source_model_id,
        "source_revision": manifest.source_revision,
        "source_license": manifest.source_license,
        "weight_bits": config.weight_bits,
        "activation_bits": config.activation_bits,
        "target_policy": config.target_policy,
        "runtime_mode": config.runtime_mode,
        "activation_kernel_backend": config.activation_kernel_backend,
        "tensor_count": len(state_dict),
        "quantized_module_count": len(manifest.quantized_modules),
        "adaln_module_count": len(manifest.adaln_modules),
        "skipped_module_count": len(manifest.skipped_modules),
        "required_files": list(_REQUIRED_ARTIFACT_FILES),
        "checksums": manifest.checksums,
    }
