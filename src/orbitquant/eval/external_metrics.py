from __future__ import annotations

import json
from pathlib import Path
from typing import Any

VBENCH_CANONICAL_DIMENSIONS = (
    "imaging_quality",
    "aesthetic_quality",
    "motion_smoothness",
    "dynamic_degree",
    "background_consistency",
    "subject_consistency",
    "scene",
    "overall_consistency",
)
_SCORE_KEYS = ("score", "mean", "avg", "average", "value")
_DIMENSION_KEYS = ("dimension", "metric", "name", "task")


def _safe_key(value: str) -> str:
    return "_".join(part for part in value.lower().replace("-", "_").split("_") if part)


def summarize_geneval_results(results_jsonl: str | Path, output_json: str | Path) -> dict[str, Any]:
    path = Path(results_jsonl)
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"GenEval results file is empty: {path}")
    per_tag: dict[str, dict[str, int]] = {}
    correct_count = 0
    for record in records:
        tag = str(record.get("tag", "unknown"))
        correct = bool(record.get("correct", False))
        bucket = per_tag.setdefault(tag, {"correct": 0, "total": 0})
        bucket["total"] += 1
        if correct:
            bucket["correct"] += 1
            correct_count += 1
    summary = {
        "overall": correct_count / len(records),
        "records": len(records),
        "tags": {
            tag: {
                "score": values["correct"] / values["total"],
                "correct": values["correct"],
                "total": values["total"],
            }
            for tag, values in sorted(per_tag.items())
        },
    }
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _numeric_values(value: Any, *, prefix: tuple[str, ...] = ()) -> dict[str, float | int]:
    if isinstance(value, bool):
        return {}
    if isinstance(value, int | float):
        if not prefix:
            return {}
        return {"_".join(prefix): value}
    if isinstance(value, dict):
        found: dict[str, float | int] = {}
        for key, item in value.items():
            safe = _safe_key(str(key))
            if safe:
                found.update(_numeric_values(item, prefix=(*prefix, safe)))
        return found
    if isinstance(value, list):
        found = {}
        for index, item in enumerate(value):
            found.update(_numeric_values(item, prefix=(*prefix, str(index))))
        return found
    return {}


def _is_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int | float)


def _find_dimension_from_name(value: str) -> str | None:
    safe = _safe_key(value)
    for dimension in VBENCH_CANONICAL_DIMENSIONS:
        if dimension in safe:
            return dimension
    return None


def _first_named_score(value: Any) -> float | int | None:
    if _is_number(value):
        return value
    if isinstance(value, dict):
        for key in _SCORE_KEYS:
            item = value.get(key)
            if _is_number(item):
                return item
        for item in value.values():
            found = _first_named_score(item)
            if found is not None:
                return found
    if isinstance(value, list | tuple):
        for item in value:
            found = _first_named_score(item)
            if found is not None:
                return found
    return None


def _dimension_from_record(value: dict[str, Any]) -> str | None:
    for key in _DIMENSION_KEYS:
        item = value.get(key)
        if isinstance(item, str):
            dimension = _find_dimension_from_name(item)
            if dimension is not None:
                return dimension
    return None


def _collect_canonical_vbench_metrics(
    value: Any, *, context_dimension: str | None = None
) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {}
    if isinstance(value, dict):
        context_dimension = _dimension_from_record(value) or context_dimension
    if context_dimension is not None:
        score = _first_named_score(value)
        if score is not None:
            metrics[context_dimension] = score
    if isinstance(value, dict):
        for key, item in value.items():
            dimension = _find_dimension_from_name(str(key)) or context_dimension
            metrics.update(
                _collect_canonical_vbench_metrics(item, context_dimension=dimension)
            )
    if isinstance(value, list | tuple):
        for item in value:
            metrics.update(
                _collect_canonical_vbench_metrics(
                    item, context_dimension=context_dimension
                )
            )
    return metrics


def summarize_vbench_results(results_dir: str | Path, output_json: str | Path) -> dict[str, Any]:
    root = Path(results_dir)
    if not root.is_dir():
        raise ValueError(f"VBench results directory missing: {root}")
    canonical_metrics: dict[str, float | int] = {}
    metrics: dict[str, float | int] = {}
    source_files = []
    for path in sorted(root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        prefix = (_safe_key(path.stem),)
        context_dimension = _find_dimension_from_name(path.stem)
        canonical_metrics.update(
            _collect_canonical_vbench_metrics(
                payload, context_dimension=context_dimension
            )
        )
        numeric = _numeric_values(payload, prefix=prefix)
        if numeric:
            metrics.update(numeric)
            source_files.append(str(path))
    if not metrics:
        raise ValueError(f"no numeric VBench metrics found under {root}")
    ordered_canonical_metrics = {
        key: canonical_metrics[key]
        for key in VBENCH_CANONICAL_DIMENSIONS
        if key in canonical_metrics
    }
    summary = {
        "source_files": source_files,
        **ordered_canonical_metrics,
        "metrics": metrics,
    }
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
