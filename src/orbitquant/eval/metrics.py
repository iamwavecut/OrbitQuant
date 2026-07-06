from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _safe_metric_name(value: str) -> str:
    return "_".join(part for part in value.lower().replace("-", "_").split("_") if part)


def _flatten_numeric_metrics(
    value: Any, *, prefix_parts: tuple[str, ...] = ()
) -> dict[str, float | int]:
    if isinstance(value, bool):
        return {}
    if isinstance(value, int | float):
        if not prefix_parts:
            return {}
        return {"_".join(prefix_parts): value}
    if isinstance(value, dict):
        metrics: dict[str, float | int] = {}
        for key, item in value.items():
            safe_key = _safe_metric_name(str(key))
            if not safe_key:
                continue
            metrics.update(
                _flatten_numeric_metrics(item, prefix_parts=(*prefix_parts, safe_key))
            )
        return metrics
    return {}


def load_metric_json(
    path: str | Path, *, metric_prefix: str | None = None
) -> dict[str, float | int]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    prefix_parts = ()
    if metric_prefix:
        prefix_parts = (_safe_metric_name(metric_prefix),)
    metrics = _flatten_numeric_metrics(payload, prefix_parts=prefix_parts)
    if not metrics:
        raise ValueError("metrics JSON did not contain numeric metrics")
    return dict(sorted(metrics.items()))
