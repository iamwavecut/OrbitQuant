from __future__ import annotations

import json
from pathlib import Path
from typing import Any


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


def summarize_vbench_results(results_dir: str | Path, output_json: str | Path) -> dict[str, Any]:
    root = Path(results_dir)
    if not root.is_dir():
        raise ValueError(f"VBench results directory missing: {root}")
    metrics: dict[str, float | int] = {}
    source_files = []
    for path in sorted(root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        prefix = (_safe_key(path.stem),)
        numeric = _numeric_values(payload, prefix=prefix)
        if numeric:
            metrics.update(numeric)
            source_files.append(str(path))
    if not metrics:
        raise ValueError(f"no numeric VBench metrics found under {root}")
    summary = {
        "source_files": source_files,
        "metrics": metrics,
    }
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
