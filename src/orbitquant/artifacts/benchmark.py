from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from orbitquant.artifacts.checksums import sha256_file, validate_checksums, write_sha256sums
from orbitquant.artifacts.manifest import OrbitQuantManifest
from orbitquant.artifacts.validator import validate_required_artifact_files

_VALID_METRIC_SPLITS = {"original", "orbitquant"}


def _normalize_json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _normalize_json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_normalize_json_value(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass
    return value


def _csv_value(value: Any) -> Any:
    normalized = _normalize_json_value(value)
    if isinstance(normalized, dict | list):
        return json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return normalized


def _read_manifest(artifact_path: Path) -> OrbitQuantManifest:
    return OrbitQuantManifest.from_dict(
        json.loads((artifact_path / "orbitquant_manifest.json").read_text(encoding="utf-8"))
    )


def _read_summary(summary_path: Path) -> dict[str, Any]:
    if not summary_path.is_file():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _record_count(jsonl_path: Path) -> int:
    return sum(1 for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip())


def _write_manifest_with_refreshed_checksums(
    artifact_path: Path, manifest: OrbitQuantManifest, relative_paths: tuple[str, ...]
) -> None:
    checksums = dict(manifest.checksums)
    for relative_path in relative_paths:
        checksums[relative_path] = sha256_file(artifact_path / relative_path)
    payload = manifest.to_dict()
    payload["checksums"] = checksums
    (artifact_path / "orbitquant_manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def record_artifact_metrics(
    artifact_dir: str | Path,
    *,
    split: str,
    metrics: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_split = split.lower()
    if normalized_split not in _VALID_METRIC_SPLITS:
        raise ValueError("split must be one of: original, orbitquant")
    if not metrics:
        raise ValueError("metrics must not be empty")

    artifact_path = Path(artifact_dir)
    validate_required_artifact_files(artifact_path)
    manifest = _read_manifest(artifact_path)
    validate_checksums(artifact_path, manifest.checksums)

    record = {
        "split": normalized_split,
        "metrics": _normalize_json_value(metrics),
        "metadata": {} if metadata is None else _normalize_json_value(metadata),
    }
    summary_path = artifact_path / "benchmark" / "summary.json"
    jsonl_path = artifact_path / "benchmark" / f"{normalized_split}.metrics.jsonl"
    csv_path = artifact_path / "benchmark" / f"{normalized_split}.metrics.csv"

    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, separators=(",", ":")) + "\n")

    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        for metric, value in record["metrics"].items():
            writer.writerow([metric, _csv_value(value)])

    summary = _read_summary(summary_path)
    summary["status"] = "metrics_recorded"
    split_summaries = summary.setdefault("metrics", {})
    split_summaries[normalized_split] = {
        "records": _record_count(jsonl_path),
        "latest": record,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    changed_paths = (
        "benchmark/summary.json",
        f"benchmark/{normalized_split}.metrics.jsonl",
        f"benchmark/{normalized_split}.metrics.csv",
    )
    _write_manifest_with_refreshed_checksums(artifact_path, manifest, changed_paths)
    write_sha256sums(artifact_path)
    return record
