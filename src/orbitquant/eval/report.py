from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from orbitquant.artifacts import validate_orbitquant_artifact
from orbitquant.eval.native_settings import list_native_suites

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
_REQUIRED_METRICS_BY_SUITE = {
    "flux1-schnell-native": ("geneval_overall",),
    "z-image-native": ("geneval_overall",),
    "wan-native": (
        "vbench_imaging_quality",
        "vbench_aesthetic_quality",
        "vbench_motion_smoothness",
        "vbench_dynamic_degree",
        "vbench_background_consistency",
        "vbench_subject_consistency",
        "vbench_scene",
        "vbench_overall_consistency",
    ),
}
_SUITE_BY_MODEL_ID = {suite.model_id: suite for suite in list_native_suites()}


@dataclass(frozen=True)
class NativeEvalReportResult:
    report_path: Path
    table_paths: dict[str, Path]
    rows: list[dict[str, Any]]
    missing_required_metrics: list[dict[str, Any]]


@dataclass(frozen=True)
class _ArtifactEval:
    artifact_dir: Path
    validation: dict[str, Any]
    bits: str
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


def _artifact_eval(artifact_dir: Path) -> _ArtifactEval:
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
    return _ArtifactEval(artifact_dir=artifact_dir, validation=validation, bits=bits, rows=rows)


def _artifact_rows(artifact_dir: Path) -> list[dict[str, Any]]:
    return _artifact_eval(artifact_dir).rows


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


def _suite_names_for_artifact(artifact: _ArtifactEval) -> list[str]:
    row_suites = sorted({row["suite"] for row in artifact.rows if row["suite"]})
    if row_suites:
        return row_suites
    suite = _SUITE_BY_MODEL_ID.get(artifact.validation["source_model_id"])
    return [] if suite is None else [suite.name]


def _metrics_for_suite(
    rows: list[dict[str, Any]], *, split: str, suite: str, allow_blank_suite: bool
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for row in rows:
        row_suite = row["suite"]
        if row["split"] != split:
            continue
        if row_suite != suite and not (allow_blank_suite and not row_suite):
            continue
        metrics.update(row["metrics"])
    return metrics


def _missing_required_metrics(artifacts: list[_ArtifactEval]) -> list[dict[str, Any]]:
    missing = []
    for artifact in artifacts:
        allow_blank_suite = not any(row["suite"] for row in artifact.rows)
        for suite in _suite_names_for_artifact(artifact):
            required_metrics = _REQUIRED_METRICS_BY_SUITE.get(suite, ())
            if not required_metrics:
                continue
            for split in ("original", "orbitquant"):
                metrics = _metrics_for_suite(
                    artifact.rows,
                    split=split,
                    suite=suite,
                    allow_blank_suite=allow_blank_suite,
                )
                for metric in required_metrics:
                    if metric in metrics:
                        continue
                    missing.append(
                        {
                            "artifact_dir": str(artifact.artifact_dir),
                            "source_model_id": artifact.validation["source_model_id"],
                            "target_group": _target_group(
                                artifact.validation["source_model_id"]
                            ),
                            "bits": artifact.bits,
                            "split": split,
                            "suite": suite,
                            "metric": metric,
                        }
                    )
    return missing


def _write_missing_required_metric_table(
    path: Path, missing_metrics: list[dict[str, Any]]
) -> None:
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
                "artifact_dir",
            ]
        )
        for missing in missing_metrics:
            writer.writerow(
                [
                    missing["target_group"],
                    missing["source_model_id"],
                    missing["bits"],
                    missing["split"],
                    missing["suite"],
                    missing["metric"],
                    missing["artifact_dir"],
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


def _missing_markdown_table(missing_metrics: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Group | Model | Bits | Split | Suite | Missing Metric |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for missing in missing_metrics:
        lines.append(
            "| "
            + " | ".join(
                [
                    missing["target_group"],
                    f"`{missing['source_model_id']}`",
                    f"`{missing['bits']}`",
                    missing["split"],
                    missing["suite"],
                    missing["metric"],
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
    artifacts: list[_ArtifactEval] = []
    for artifact_dir in artifact_dirs:
        artifact = _artifact_eval(Path(artifact_dir))
        artifacts.append(artifact)
        rows.extend(artifact.rows)

    image_table = tables_dir / "image_geneval.csv"
    video_table = tables_dir / "video_vbench.csv"
    perf_table = tables_dir / "perf.csv"
    asset_table = tables_dir / "assets.csv"
    missing_table = tables_dir / "missing_required_metrics.csv"
    missing_required_metrics = _missing_required_metrics(artifacts)
    _write_metric_table(image_table, rows, family="image")
    _write_metric_table(video_table, rows, family="video")
    _write_perf_table(perf_table, rows)
    _write_asset_table(asset_table, rows)
    _write_missing_required_metric_table(missing_table, missing_required_metrics)

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
        "## Missing Required Metrics",
        "",
        *(
            _missing_markdown_table(missing_required_metrics)
            if missing_required_metrics
            else ["No required paper-target metrics are missing."]
        ),
        "",
        "## Tables",
        "",
        f"- Image GenEval: `{image_table.relative_to(output_path).as_posix()}`",
        f"- Video VBench: `{video_table.relative_to(output_path).as_posix()}`",
        f"- Performance: `{perf_table.relative_to(output_path).as_posix()}`",
        f"- Generated Assets: `{asset_table.relative_to(output_path).as_posix()}`",
        f"- Missing Required Metrics: `{missing_table.relative_to(output_path).as_posix()}`",
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
            "missing_required_metrics": missing_table,
        },
        rows=rows,
        missing_required_metrics=missing_required_metrics,
    )
