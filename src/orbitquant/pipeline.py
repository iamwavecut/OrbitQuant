from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from orbitquant.artifacts import (
    OrbitQuantManifest,
    load_orbitquant_artifact,
    save_orbitquant_artifact,
)
from orbitquant.config import OrbitQuantConfig
from orbitquant.modeling import QuantizationSummary, quantize_linear_modules


def _get_pipeline_component(pipeline: Any, component: str) -> torch.nn.Module:
    try:
        value = getattr(pipeline, component)
    except AttributeError as exc:
        raise ValueError(f"pipeline has no component {component!r}") from exc
    if not isinstance(value, torch.nn.Module):
        raise TypeError(f"pipeline component {component!r} is not a torch.nn.Module")
    return value


def quantize_pipeline(
    pipeline: Any,
    config: OrbitQuantConfig,
    *,
    component: str = "transformer",
) -> QuantizationSummary:
    target = _get_pipeline_component(pipeline, component)
    return quantize_linear_modules(target, config)


def save_quantized_pipeline_component(
    pipeline: Any,
    output_dir: str | Path,
    *,
    config: OrbitQuantConfig,
    component: str = "transformer",
    source_model_id: str,
    source_revision: str,
    source_license: str,
    summary: QuantizationSummary,
) -> OrbitQuantManifest:
    target = _get_pipeline_component(pipeline, component)
    return save_orbitquant_artifact(
        target,
        output_dir,
        config=config,
        source_model_id=source_model_id,
        source_revision=source_revision,
        source_license=source_license,
        summary=summary,
        component=component,
    )


def load_quantized_pipeline_component(
    pipeline: Any,
    artifact_dir: str | Path,
    *,
    component: str = "transformer",
    strict: bool = True,
) -> OrbitQuantManifest:
    target = _get_pipeline_component(pipeline, component)
    return load_orbitquant_artifact(target, artifact_dir, strict=strict)
