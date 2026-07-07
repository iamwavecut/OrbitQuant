from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from orbitquant.artifacts.assets import record_artifact_asset
from orbitquant.artifacts.validator import validate_orbitquant_artifact

_IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png", ".webp"}
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _safe_name(value: Any) -> str:
    return _SAFE_NAME_RE.sub("-", str(value)).strip("-") or "unknown"


def _prompt_id(metadata: dict[str, Any]) -> str:
    prompt_record = metadata.get("prompt_record")
    if isinstance(prompt_record, dict) and prompt_record.get("id") is not None:
        return str(prompt_record["id"])
    return "prompt"


def _output_path(artifact_path: Path, metadata: dict[str, Any]) -> Path | None:
    value = metadata.get("output_path")
    if not value:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else artifact_path / path


def _artifact_path(artifact_path: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else artifact_path / path


def _comparison_source_path(artifact_path: Path, metadata: dict[str, Any]) -> Path | None:
    output_path = _output_path(artifact_path, metadata)
    if output_path is not None and output_path.suffix.lower() in _IMAGE_SUFFIXES:
        return output_path

    contact_sheet_path = metadata.get("contact_sheet_path")
    if contact_sheet_path:
        path = _artifact_path(artifact_path, contact_sheet_path)
        if path.suffix.lower() in _IMAGE_SUFFIXES:
            return path

    for asset_path in metadata.get("asset_paths", []):
        path = _artifact_path(artifact_path, asset_path)
        if "contact_sheet" in path.name and path.suffix.lower() in _IMAGE_SUFFIXES:
            return path
    return None


def _comparison_key(metadata: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(metadata.get("suite", "")),
        str(metadata.get("seed", "")),
        _prompt_id(metadata),
    )


def _indexed_records(artifact_path: Path, split: str) -> dict[tuple[str, str, str], dict[str, Any]]:
    records = {}
    for record in _read_jsonl(artifact_path / "benchmark" / f"{split}.metrics.jsonl"):
        metadata = dict(record.get("metadata", {}))
        path = _comparison_source_path(artifact_path, metadata)
        if path is None or not path.is_file():
            continue
        records[_comparison_key(metadata)] = {"metadata": metadata, "path": path}
    return records


def create_artifact_image_comparisons(
    artifact_dir: str | Path,
    *,
    validate_checksums_enabled: bool = True,
    refresh_checksums_enabled: bool = True,
) -> list[str]:
    from orbitquant.eval.assets import create_image_comparison_sheet

    artifact_path = Path(artifact_dir)
    validate_orbitquant_artifact(
        artifact_path,
        validate_checksums_enabled=validate_checksums_enabled,
        validate_tensors=validate_checksums_enabled,
    )
    original_records = _indexed_records(artifact_path, "original")
    orbitquant_records = _indexed_records(artifact_path, "orbitquant")
    created = []
    for key, orbitquant in sorted(orbitquant_records.items()):
        original = original_records.get(key)
        if original is None:
            continue
        suite, seed, prompt_id = key
        bit_setting = orbitquant["metadata"].get("bit_setting", "orbitquant")
        output_path = (
            artifact_path
            / "assets"
            / (
                "original_vs_orbitquant_"
                f"{_safe_name(suite)}_seed{_safe_name(seed)}_"
                f"{_safe_name(bit_setting)}_{_safe_name(prompt_id)}.webp"
            )
        )
        create_image_comparison_sheet(
            original["path"],
            orbitquant["path"],
            output_path,
            labels=("BF16", f"OrbitQuant {bit_setting}"),
        )
        created.append(
            record_artifact_asset(
                artifact_path,
                output_path,
                validate_checksums_enabled=validate_checksums_enabled,
                refresh_checksums_enabled=refresh_checksums_enabled,
            )
        )
    return created
