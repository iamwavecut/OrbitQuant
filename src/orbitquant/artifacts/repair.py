from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from orbitquant.artifacts.checksums import (
    is_ignored_artifact_relative_path,
    sha256_file,
    write_sha256sums_from_manifest,
)
from orbitquant.artifacts.manifest import OrbitQuantManifest
from orbitquant.artifacts.model_card import render_model_card
from orbitquant.artifacts.validator import validate_orbitquant_artifact
from orbitquant.config import OrbitQuantConfig


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def repair_artifact_metadata(
    artifact_dir: str | Path,
    *,
    quantization_device: str | None = None,
    weight_quantization_backend: str | None = None,
    validate_tensors: bool = True,
) -> dict[str, Any]:
    """Refresh metadata-only files after schema/provenance changes.

    This does not rewrite packed weights, codebooks, rotations, prompts, assets, or
    metric rows. It updates manifest/model_index/benchmark summary/README and then
    rewrites SHA256SUMS so the artifact remains self-consistent.
    """

    artifact_path = Path(artifact_dir)
    before = validate_orbitquant_artifact(
        artifact_path,
        validate_checksums_enabled=True,
        validate_tensors=validate_tensors,
    )
    manifest_path = artifact_path / "orbitquant_manifest.json"
    model_index_path = artifact_path / "model_index.json"
    benchmark_path = artifact_path / "benchmark" / "summary.json"
    config_path = artifact_path / "quantization_config.json"

    config = OrbitQuantConfig.from_dict(_read_json(config_path))
    manifest = OrbitQuantManifest.from_dict(_read_json(manifest_path))
    model_index = _read_json(model_index_path)
    benchmark_summary = _read_json(benchmark_path)

    next_quantization_device = quantization_device or manifest.quantization_device
    next_weight_backend = weight_quantization_backend or manifest.weight_quantization_backend
    next_staging_mode = manifest.quantization_staging_mode

    manifest = OrbitQuantManifest(
        source_model_id=manifest.source_model_id,
        source_revision=manifest.source_revision,
        source_license=manifest.source_license,
        weight_bits=manifest.weight_bits,
        activation_bits=manifest.activation_bits,
        rotation_seed=manifest.rotation_seed,
        block_size=manifest.block_size,
        block_size_policy=manifest.block_size_policy,
        codebook_version=manifest.codebook_version,
        target_policy=manifest.target_policy,
        runtime_mode=manifest.runtime_mode,
        activation_kernel_backend=manifest.activation_kernel_backend,
        quantization_device=next_quantization_device,
        weight_quantization_backend=next_weight_backend,
        quantization_staging_mode=next_staging_mode,
        quantized_modules=manifest.quantized_modules,
        adaln_modules=manifest.adaln_modules,
        skipped_modules=manifest.skipped_modules,
        module_shapes=manifest.module_shapes,
        checksums={
            relative_path: digest
            for relative_path, digest in manifest.checksums.items()
            if not is_ignored_artifact_relative_path(relative_path)
        },
    )

    model_index["quantization_device"] = next_quantization_device
    model_index["weight_quantization_backend"] = next_weight_backend
    model_index["quantization_staging_mode"] = next_staging_mode
    benchmark_summary["quantization_device"] = next_quantization_device
    benchmark_summary["weight_quantization_backend"] = next_weight_backend
    benchmark_summary["quantization_staging_mode"] = next_staging_mode

    _write_json(model_index_path, model_index)
    _write_json(benchmark_path, benchmark_summary)
    manifest.checksums.update(
        {
            "model_index.json": sha256_file(model_index_path),
            "benchmark/summary.json": sha256_file(benchmark_path),
            "quantization_config.json": sha256_file(config_path),
        }
    )
    _write_json(manifest_path, manifest.to_dict())
    (artifact_path / "README.md").write_text(render_model_card(manifest), encoding="utf-8")
    write_sha256sums_from_manifest(artifact_path, manifest.checksums)

    after = validate_orbitquant_artifact(
        artifact_path,
        validate_checksums_enabled=True,
        validate_tensors=validate_tensors,
    )
    return {
        "artifact_dir": str(artifact_path),
        "updated": {
            "quantization_device": next_quantization_device,
            "weight_quantization_backend": next_weight_backend,
            "quantization_staging_mode": next_staging_mode,
        },
        "before": before,
        "after": after,
        "config_bits": f"W{config.weight_bits}A{config.activation_bits}",
    }
