from __future__ import annotations

import json
from pathlib import Path

from orbitquant.artifacts.checksums import sha256_file, validate_checksums, write_sha256sums
from orbitquant.artifacts.manifest import OrbitQuantManifest
from orbitquant.artifacts.validator import validate_required_artifact_files


def _read_manifest(artifact_path: Path) -> OrbitQuantManifest:
    return OrbitQuantManifest.from_dict(
        json.loads((artifact_path / "orbitquant_manifest.json").read_text(encoding="utf-8"))
    )


def _relative_asset_path(artifact_path: Path, asset_path: Path) -> str:
    try:
        relative = asset_path.resolve().relative_to(artifact_path.resolve())
    except ValueError as exc:
        raise ValueError("artifact asset must be inside the artifact directory") from exc
    if relative.parts[0] != "assets":
        raise ValueError("artifact asset must live under assets/")
    return relative.as_posix()


def record_artifact_asset(artifact_dir: str | Path, asset_path: str | Path) -> str:
    artifact_path = Path(artifact_dir)
    asset = Path(asset_path)
    if not asset.is_file():
        raise RuntimeError(f"artifact asset missing: {asset}")

    validate_required_artifact_files(artifact_path)
    manifest = _read_manifest(artifact_path)
    validate_checksums(artifact_path, manifest.checksums)

    relative_path = _relative_asset_path(artifact_path, asset)
    checksums = dict(manifest.checksums)
    checksums[relative_path] = sha256_file(asset)
    payload = manifest.to_dict()
    payload["checksums"] = checksums
    (artifact_path / "orbitquant_manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    write_sha256sums(artifact_path)
    return relative_path
