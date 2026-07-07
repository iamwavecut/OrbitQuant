from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from safetensors.torch import load_file

from orbitquant.artifacts.checksums import validate_checksums, validate_sha256sums
from orbitquant.artifacts.manifest import OrbitQuantManifest
from orbitquant.config import OrbitQuantConfig

_REQUIRED_ARTIFACT_FILES = (
    "README.md",
    "SHA256SUMS",
    "model_index.json",
    "model.safetensors",
    "quantization_config.json",
    "orbitquant_manifest.json",
    "orbitquant_codebooks.safetensors",
    "orbitquant_rotations.safetensors",
    "prompts.json",
    "benchmark/summary.json",
)

_SHA256SUMS_REQUIRED_EXTRA_ENTRIES = ("README.md", "orbitquant_manifest.json")


def validate_required_artifact_files(artifact_path: Path) -> None:
    missing = [
        relative_path
        for relative_path in _REQUIRED_ARTIFACT_FILES
        if not (artifact_path / relative_path).is_file()
    ]
    if missing:
        raise RuntimeError(f"required artifact file missing: {missing}")


def _mismatch(name: str, expected: Any, actual: Any) -> str | None:
    return None if expected == actual else f"{name}: expected {expected!r}, got {actual!r}"


def _validate_model_index(
    model_index: dict[str, Any],
    *,
    config: OrbitQuantConfig,
    manifest: OrbitQuantManifest,
) -> None:
    expected = {
        "_class_name": "OrbitQuantComponentArtifact",
        "artifact_format": "orbitquant-v1",
        "quant_method": "orbitquant",
        "source_model_id": manifest.source_model_id,
        "source_revision": manifest.source_revision,
        "source_license": manifest.source_license,
        "weight_name": "model.safetensors",
        "quantization_config": "quantization_config.json",
        "manifest": "orbitquant_manifest.json",
        "codebooks": "orbitquant_codebooks.safetensors",
        "rotations": "orbitquant_rotations.safetensors",
        "weight_bits": config.weight_bits,
        "activation_bits": config.activation_bits,
        "target_policy": config.target_policy,
        "runtime_mode": config.runtime_mode,
        "activation_kernel_backend": config.activation_kernel_backend,
    }
    if manifest.quantization_device != "unknown" or "quantization_device" in model_index:
        expected["quantization_device"] = manifest.quantization_device
    if (
        manifest.weight_quantization_backend != "unknown"
        or "weight_quantization_backend" in model_index
    ):
        expected["weight_quantization_backend"] = manifest.weight_quantization_backend
    if (
        manifest.quantization_staging_mode != "unknown"
        or "quantization_staging_mode" in model_index
    ):
        expected["quantization_staging_mode"] = manifest.quantization_staging_mode
    mismatches = [
        mismatch
        for key, value in expected.items()
        if (mismatch := _mismatch(key, value, model_index.get(key))) is not None
    ]
    component = model_index.get("component")
    if not isinstance(component, str) or not component:
        mismatches.append(f"component: expected non-empty string, got {component!r}")
    if mismatches:
        raise RuntimeError("model_index mismatch: " + "; ".join(mismatches))


def validate_orbitquant_artifact(
    artifact_dir: str | Path,
    *,
    validate_checksums_enabled: bool = True,
    validate_tensors: bool = True,
) -> dict[str, Any]:
    artifact_path = Path(artifact_dir)
    validate_required_artifact_files(artifact_path)
    model_index = json.loads((artifact_path / "model_index.json").read_text(encoding="utf-8"))
    config = OrbitQuantConfig.from_dict(
        json.loads((artifact_path / "quantization_config.json").read_text(encoding="utf-8"))
    )
    manifest = OrbitQuantManifest.from_dict(
        json.loads((artifact_path / "orbitquant_manifest.json").read_text(encoding="utf-8"))
    )
    if validate_checksums_enabled:
        validate_checksums(artifact_path, manifest.checksums)
        sha256sums_entries = validate_sha256sums(
            artifact_path,
            required_paths=tuple(manifest.checksums) + _SHA256SUMS_REQUIRED_EXTRA_ENTRIES,
        )
    else:
        sha256sums_entries = {}
    _validate_model_index(model_index, config=config, manifest=manifest)
    expected_shapes = manifest.module_shapes
    if validate_tensors:
        state_dict = load_file(artifact_path / "model.safetensors")
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
        "component": model_index["component"],
        "runtime_mode": config.runtime_mode,
        "activation_kernel_backend": config.activation_kernel_backend,
        "adaln_group_size": manifest.adaln_group_size,
        "quantization_device": manifest.quantization_device,
        "weight_quantization_backend": manifest.weight_quantization_backend,
        "quantization_staging_mode": manifest.quantization_staging_mode,
        "tensor_count": len(expected_shapes),
        "tensor_validation": "checked" if validate_tensors else "skipped",
        "checksum_validation": "checked" if validate_checksums_enabled else "skipped",
        "sha256sums_validation": "checked" if validate_checksums_enabled else "skipped",
        "sha256sums_entry_count": len(sha256sums_entries),
        "quantized_module_count": len(manifest.quantized_modules),
        "adaln_module_count": len(manifest.adaln_modules),
        "skipped_module_count": len(manifest.skipped_modules),
        "required_files": list(_REQUIRED_ARTIFACT_FILES),
        "checksums": manifest.checksums,
    }


def _module_list_mismatch(
    name: str, expected: list[str], actual: list[str]
) -> dict[str, Any] | None:
    expected_set = set(expected)
    actual_set = set(actual)
    missing = sorted(expected_set - actual_set)
    unexpected = sorted(actual_set - expected_set)
    if not missing and not unexpected and len(expected) == len(actual):
        return None
    return {
        "name": name,
        "expected_count": len(expected),
        "actual_count": len(actual),
        "missing": missing,
        "unexpected": unexpected,
    }


def validate_artifact_policy_inventory(
    artifact_dir: str | Path,
    inventory_path: str | Path,
) -> dict[str, Any]:
    artifact_path = Path(artifact_dir)
    inventory_file = Path(inventory_path)
    manifest = OrbitQuantManifest.from_dict(
        json.loads((artifact_path / "orbitquant_manifest.json").read_text(encoding="utf-8"))
    )
    inventory = json.loads(inventory_file.read_text(encoding="utf-8"))
    scalar_mismatches = [
        mismatch
        for key, expected, actual in (
            ("source_model_id", manifest.source_model_id, inventory.get("source_model_id")),
            ("target_policy", manifest.target_policy, inventory.get("target_policy")),
        )
        if (mismatch := _mismatch(key, expected, actual)) is not None
    ]
    module_mismatches = [
        mismatch
        for mismatch in (
            _module_list_mismatch(
                "quantized_modules",
                list(inventory.get("quantized_modules", [])),
                manifest.quantized_modules,
            ),
            _module_list_mismatch(
                "adaln_modules",
                list(inventory.get("adaln_modules", [])),
                manifest.adaln_modules,
            ),
            _module_list_mismatch(
                "skipped_modules",
                list(inventory.get("skipped_modules", [])),
                manifest.skipped_modules,
            ),
        )
        if mismatch is not None
    ]
    if scalar_mismatches or module_mismatches:
        raise RuntimeError(
            "artifact policy inventory mismatch: "
            f"scalars={scalar_mismatches}, modules={module_mismatches}"
        )
    action_counts = inventory.get("action_counts") or {}
    return {
        "valid": True,
        "artifact_dir": str(artifact_path),
        "inventory_path": str(inventory_file),
        "source_model_id": manifest.source_model_id,
        "target_policy": manifest.target_policy,
        "component": inventory.get("component"),
        "load_mode": inventory.get("load_mode"),
        "linear_module_count": inventory.get("linear_module_count"),
        "action_counts": action_counts,
        "quantized_module_count": len(manifest.quantized_modules),
        "adaln_module_count": len(manifest.adaln_modules),
        "skipped_module_count": len(manifest.skipped_modules),
    }
