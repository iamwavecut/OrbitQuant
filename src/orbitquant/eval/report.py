from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from orbitquant.artifacts import validate_orbitquant_artifact
from orbitquant.eval.native_settings import list_native_suites

_IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png", ".webp"}
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_MAX_COMPARISON_MATRIX_ROWS = 10
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
_GENEVAL_REQUIRED_METRICS = (
    "geneval_overall",
    "geneval_per_task_single_object",
    "geneval_per_task_two_object",
    "geneval_per_task_counting",
    "geneval_per_task_colors",
    "geneval_per_task_position",
    "geneval_per_task_color_attr",
)
_REQUIRED_METRICS_BY_SUITE = {
    "flux1-schnell-native": _GENEVAL_REQUIRED_METRICS,
    "z-image-native": _GENEVAL_REQUIRED_METRICS,
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
    comparison_asset_paths: dict[str, Path]
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


def _safe_name(value: Any) -> str:
    return _SAFE_NAME_RE.sub("-", str(value)).strip("-") or "unknown"


def _artifact_relative_or_absolute_path(artifact_dir: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else artifact_dir / path


def _row_prompt_id(row: dict[str, Any]) -> str:
    prompt_record = row["metadata"].get("prompt_record")
    if isinstance(prompt_record, dict) and prompt_record.get("id") is not None:
        return str(prompt_record["id"])
    prompt = row["metadata"].get("prompt")
    if isinstance(prompt, str) and prompt:
        return _safe_name(prompt[:48])
    return "prompt"


def _visual_source_path(row: dict[str, Any]) -> tuple[Path, str] | None:
    artifact_dir = Path(row["artifact_dir"])
    metadata = row["metadata"]
    output_path = metadata.get("output_path")
    if output_path:
        path = _artifact_relative_or_absolute_path(artifact_dir, output_path)
        if path.suffix.lower() in _IMAGE_SUFFIXES and path.is_file():
            return path, "image"

    contact_sheet_path = metadata.get("contact_sheet_path")
    if contact_sheet_path:
        path = _artifact_relative_or_absolute_path(artifact_dir, contact_sheet_path)
        if path.suffix.lower() in _IMAGE_SUFFIXES and path.is_file():
            return path, "video"

    for asset_path in metadata.get("asset_paths", []):
        path = _artifact_relative_or_absolute_path(artifact_dir, asset_path)
        if (
            "contact_sheet" in path.name
            and path.suffix.lower() in _IMAGE_SUFFIXES
            and path.is_file()
        ):
            return path, "video"
    return None


def _comparison_column_label(row: dict[str, Any]) -> str:
    model_name = row["source_model_id"].rsplit("/", maxsplit=1)[-1]
    if row["split"] == "original":
        return f"{model_name}\nBF16 baseline"
    return f"{model_name}\n{row['bits']} OrbitQuant"


def _comparison_row_label(row_key: tuple[str, str, str]) -> str:
    suite, seed, prompt_id = row_key
    return f"{suite}\nseed {seed}\n{prompt_id}"


def _text_lines(value: str, *, max_chars: int = 26) -> list[str]:
    lines: list[str] = []
    for source_line in value.splitlines():
        words = source_line.split()
        current = ""
        for word in words:
            candidate = word if not current else f"{current} {word}"
            if len(candidate) <= max_chars:
                current = candidate
                continue
            if current:
                lines.append(current)
            current = word[:max_chars]
        if current:
            lines.append(current)
    return lines or [""]


def _draw_multiline_text(
    draw: Any,
    xy: tuple[int, int],
    text: str,
    *,
    fill: str = "black",
    max_chars: int = 26,
) -> None:
    x, y = xy
    for line in _text_lines(text, max_chars=max_chars):
        draw.text((x, y), line, fill=fill)
        y += 13


def _thumbnail(path: Path, *, size: tuple[int, int]) -> Any:
    from PIL import Image

    image = Image.open(path).convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    tile = Image.new("RGB", size, "white")
    tile.paste(
        image,
        ((size[0] - image.width) // 2, (size[1] - image.height) // 2),
    )
    return tile


def _write_comparison_matrix(
    path: Path,
    items: list[dict[str, Any]],
    *,
    title: str,
) -> Path | None:
    if not items:
        return None
    from PIL import Image, ImageDraw

    path.parent.mkdir(parents=True, exist_ok=True)
    row_keys = sorted({item["row_key"] for item in items})[:_MAX_COMPARISON_MATRIX_ROWS]
    col_keys = sorted({item["col_key"] for item in items})
    if not row_keys or not col_keys:
        return None

    by_cell = {(item["row_key"], item["col_key"]): item for item in items}
    row_label_width = 210
    col_width = 220
    title_height = 34
    header_height = 66
    row_height = 230
    tile_size = (col_width - 12, row_height - 16)
    width = row_label_width + col_width * len(col_keys)
    height = title_height + header_height + row_height * len(row_keys)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.rectangle((0, 0, width, title_height), fill="#111827")
    draw.text((8, 10), title, fill="white")
    header_bottom = title_height + header_height - 1
    draw.line((0, header_bottom, width, header_bottom), fill="#9ca3af")
    for col_index, col_key in enumerate(col_keys):
        x = row_label_width + col_index * col_width
        draw.rectangle(
            (x, title_height, x + col_width, title_height + header_height),
            fill="#f3f4f6",
        )
        _draw_multiline_text(
            draw,
            (x + 6, title_height + 8),
            str(col_key[1]),
            max_chars=24,
        )
        draw.line((x, title_height, x, height), fill="#d1d5db")
    for row_index, row_key in enumerate(row_keys):
        y = title_height + header_height + row_index * row_height
        draw.rectangle((0, y, row_label_width, y + row_height), fill="#f9fafb")
        _draw_multiline_text(
            draw,
            (8, y + 8),
            _comparison_row_label(row_key),
            max_chars=24,
        )
        draw.line((0, y, width, y), fill="#d1d5db")
        for col_index, col_key in enumerate(col_keys):
            item = by_cell.get((row_key, col_key))
            if item is None:
                continue
            x = row_label_width + col_index * col_width + 6
            tile = _thumbnail(item["path"], size=tile_size)
            sheet.paste(tile, (x, y + 8))
    sheet.save(path)
    return path


def _write_report_comparison_matrices(
    rows: list[dict[str, Any]], assets_dir: Path
) -> dict[str, Path]:
    items_by_media: dict[str, list[dict[str, Any]]] = {"image": [], "video": []}
    for row in rows:
        prompt_id = _row_prompt_id(row)
        if prompt_id.startswith("geneval-"):
            continue
        visual = _visual_source_path(row)
        if visual is None:
            continue
        path, media = visual
        row_key = (
            str(row["suite"]),
            str(row["seed"]),
            prompt_id,
        )
        column_label = _comparison_column_label(row)
        column_key = (
            row["target_group"],
            column_label,
            row["source_model_id"],
            row["bits"],
            0 if row["split"] == "original" else 1,
        )
        items_by_media[media].append(
            {
                "row_key": row_key,
                "col_key": column_key,
                "path": path,
            }
        )

    created: dict[str, Path] = {}
    for media, items in items_by_media.items():
        output_path = assets_dir / f"{media}_generation_comparison_matrix.webp"
        result = _write_comparison_matrix(
            output_path,
            items,
            title=f"OrbitQuant {media} generation comparison matrix",
        )
        if result is not None:
            created[media] = result
    return created


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
    assets_dir = output_path / "assets"
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
    comparison_asset_paths = _write_report_comparison_matrices(rows, assets_dir)

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
        "## Visual Comparison Matrices",
        "",
        *(
            [
                f"- {media.title()}: `{path.relative_to(output_path).as_posix()}`"
                for media, path in sorted(comparison_asset_paths.items())
            ]
            if comparison_asset_paths
            else ["No visual comparison matrices were generated."]
        ),
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
        comparison_asset_paths=comparison_asset_paths,
        rows=rows,
        missing_required_metrics=missing_required_metrics,
    )
