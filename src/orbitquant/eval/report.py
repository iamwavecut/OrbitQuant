from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from orbitquant.artifacts import validate_orbitquant_artifact

_PAPER_TARGETS = {
    "black-forest-labs/FLUX.1-schnell",
    "Tongyi-MAI/Z-Image-Turbo",
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
}
_PERF_METRICS = {
    "generated_frames",
    "generated_samples",
    "peak_vram_bytes",
    "peak_vram_gb",
    "wall_time_seconds",
}


@dataclass(frozen=True)
class NativeEvalReportResult:
    report_path: Path
    table_paths: dict[str, Path]
    rows: list[dict[str, Any]]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _target_group(model_id: str) -> str:
    return "paper" if model_id in _PAPER_TARGETS else "extra"


def _metric_family(row: dict[str, Any]) -> str:
    suite = row.get("suite")
    target_policy = row.get("target_policy")
    if suite == "wan-native" or target_policy == "wan":
        return "video"
    return "image"


def _artifact_rows(artifact_dir: Path) -> list[dict[str, Any]]:
    validation = validate_orbitquant_artifact(artifact_dir)
    bits = f"W{validation['weight_bits']}A{validation['activation_bits']}"
    rows = []
    for split in ("original", "orbitquant"):
        metrics_path = artifact_dir / "benchmark" / f"{split}.metrics.jsonl"
        for record in _read_jsonl(metrics_path):
            metadata = dict(record.get("metadata", {}))
            row = {
                "artifact_dir": str(artifact_dir),
                "source_model_id": validation["source_model_id"],
                "source_revision": validation["source_revision"],
                "source_license": validation["source_license"],
                "target_group": _target_group(validation["source_model_id"]),
                "target_policy": validation["target_policy"],
                "bits": bits,
                "split": split,
                "suite": metadata.get("suite", ""),
                "seed": metadata.get("seed", ""),
                "metrics": dict(record.get("metrics", {})),
                "metadata": metadata,
            }
            rows.append(row)
    return rows


def _write_metric_table(path: Path, rows: list[dict[str, Any]], *, family: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "target_group",
                "model_id",
                "bits",
                "split",
                "suite",
                "metric",
                "value",
            ]
        )
        for row in rows:
            if _metric_family(row) != family:
                continue
            for metric, value in row["metrics"].items():
                if metric in _PERF_METRICS:
                    continue
                writer.writerow(
                    [
                        row["target_group"],
                        row["source_model_id"],
                        row["bits"],
                        row["split"],
                        row["suite"],
                        metric,
                        value,
                    ]
                )


def _write_perf_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "target_group",
                "model_id",
                "bits",
                "split",
                "suite",
                "metric",
                "value",
            ]
        )
        for row in rows:
            for metric, value in row["metrics"].items():
                if metric not in _PERF_METRICS:
                    continue
                writer.writerow(
                    [
                        row["target_group"],
                        row["source_model_id"],
                        row["bits"],
                        row["split"],
                        row["suite"],
                        metric,
                        value,
                    ]
                )


def _asset_entries(metadata: dict[str, Any]) -> list[tuple[str, str]]:
    entries = []
    output_path = metadata.get("output_path")
    if output_path:
        entries.append(("output", str(output_path)))
    contact_sheet_path = metadata.get("contact_sheet_path")
    if contact_sheet_path:
        entries.append(("contact_sheet", str(contact_sheet_path)))
    for asset_path in metadata.get("asset_paths", []):
        asset_type = "comparison" if "original_vs_orbitquant" in str(asset_path) else "asset"
        entries.append((asset_type, str(asset_path)))
    return entries


def _write_asset_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "target_group",
                "model_id",
                "bits",
                "split",
                "suite",
                "asset_type",
                "path",
            ]
        )
        for row in rows:
            metadata = row["metadata"]
            for asset_type, asset_path in _asset_entries(metadata):
                writer.writerow(
                    [
                        row["target_group"],
                        row["source_model_id"],
                        row["bits"],
                        row["split"],
                        row["suite"],
                        asset_type,
                        asset_path,
                    ]
                )


def _markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Group | Model | Bits | Split | Suite | Metrics |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        metrics = ", ".join(f"{key}={value}" for key, value in row["metrics"].items())
        lines.append(
            "| "
            + " | ".join(
                [
                    row["target_group"],
                    f"`{row['source_model_id']}`",
                    f"`{row['bits']}`",
                    row["split"],
                    row["suite"],
                    metrics,
                ]
            )
            + " |"
        )
    return lines


def generate_native_eval_report(
    artifact_dirs: list[str | Path],
    output_dir: str | Path,
    *,
    report_date: str | None = None,
) -> NativeEvalReportResult:
    if not artifact_dirs:
        raise ValueError("at least one artifact is required")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    tables_dir = output_path / "tables"
    rows: list[dict[str, Any]] = []
    for artifact_dir in artifact_dirs:
        rows.extend(_artifact_rows(Path(artifact_dir)))

    image_table = tables_dir / "image_geneval.csv"
    video_table = tables_dir / "video_vbench.csv"
    perf_table = tables_dir / "perf.csv"
    asset_table = tables_dir / "assets.csv"
    _write_metric_table(image_table, rows, family="image")
    _write_metric_table(video_table, rows, family="video")
    _write_perf_table(perf_table, rows)
    _write_asset_table(asset_table, rows)

    date_label = report_date or date.today().strftime("%Y%m%d")
    report_path = output_path / f"orbitquant-native-eval-{date_label}.md"
    paper_rows = [row for row in rows if row["target_group"] == "paper"]
    extra_rows = [row for row in rows if row["target_group"] == "extra"]
    report_lines = [
        "# OrbitQuant Native Eval",
        "",
        "## Paper Reproduction Targets",
        "",
        *(_markdown_table(paper_rows) if paper_rows else ["No paper target metrics recorded."]),
        "",
        "## Extra Targets",
        "",
        *(_markdown_table(extra_rows) if extra_rows else ["No extra target metrics recorded."]),
        "",
        "## Tables",
        "",
        f"- Image GenEval: `{image_table.relative_to(output_path).as_posix()}`",
        f"- Video VBench: `{video_table.relative_to(output_path).as_posix()}`",
        f"- Performance: `{perf_table.relative_to(output_path).as_posix()}`",
        f"- Generated Assets: `{asset_table.relative_to(output_path).as_posix()}`",
        "",
    ]
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return NativeEvalReportResult(
        report_path=report_path,
        table_paths={
            "image_geneval": image_table,
            "video_vbench": video_table,
            "perf": perf_table,
            "assets": asset_table,
        },
        rows=rows,
    )
