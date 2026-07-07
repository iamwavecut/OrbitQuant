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
from orbitquant.artifacts.validator import validate_required_artifact_files

_MANAGED_NON_MANIFEST_CHECKSUMS = {"README.md", "SHA256SUMS", "orbitquant_manifest.json"}


def _read_manifest(artifact_path: Path) -> OrbitQuantManifest:
    return OrbitQuantManifest.from_dict(
        json.loads((artifact_path / "orbitquant_manifest.json").read_text(encoding="utf-8"))
    )


def _read_benchmark_summary(artifact_path: Path) -> dict[str, Any]:
    return json.loads(
        (artifact_path / "benchmark" / "summary.json").read_text(encoding="utf-8")
    )


def refresh_artifact_checksums(artifact_dir: str | Path) -> dict[str, Any]:
    """Rebuild manifest and SHA256SUMS checksums from current artifact files."""

    artifact_path = Path(artifact_dir)
    validate_required_artifact_files(artifact_path)
    manifest = _read_manifest(artifact_path)
    benchmark_summary = _read_benchmark_summary(artifact_path)
    checksums: dict[str, str] = {}
    for path in sorted(artifact_path.rglob("*")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(artifact_path).as_posix()
        if relative_path in _MANAGED_NON_MANIFEST_CHECKSUMS:
            continue
        if is_ignored_artifact_relative_path(relative_path):
            continue
        checksums[relative_path] = sha256_file(path)

    payload = manifest.to_dict()
    payload["checksums"] = checksums
    updated_manifest = OrbitQuantManifest.from_dict(payload)
    (artifact_path / "orbitquant_manifest.json").write_text(
        json.dumps(updated_manifest.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    (artifact_path / "README.md").write_text(
        render_model_card(updated_manifest, benchmark_summary=benchmark_summary),
        encoding="utf-8",
    )
    write_sha256sums_from_manifest(artifact_path, checksums)
    return {
        "artifact_dir": str(artifact_path),
        "checksum_count": len(checksums),
        "sha256sums_path": str(artifact_path / "SHA256SUMS"),
    }
