from __future__ import annotations

import json
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


def _validate_artifact_component(artifact_dir: str | Path, component: str) -> None:
    model_index_path = Path(artifact_dir) / "model_index.json"
    if not model_index_path.is_file():
        return
    payload = json.loads(model_index_path.read_text(encoding="utf-8"))
    artifact_component = payload.get("component")
    if artifact_component is None or artifact_component == component:
        return
    raise ValueError(
        "component mismatch: artifact was saved for "
        f"{artifact_component!r}, got {component!r}"
    )


def quantize_pipeline(
    pipeline: Any,
    config: OrbitQuantConfig,
    *,
    component: str = "transformer",
    quantization_device: str | torch.device | None = "auto",
    staging_mode: str = "streaming",
) -> QuantizationSummary:
    target = _get_pipeline_component(pipeline, component)
    return quantize_linear_modules(
        target,
        config,
        quantization_device=quantization_device,
        staging_mode=staging_mode,
    )


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
    validate_checksums: bool = True,
    device: str | torch.device | None = None,
) -> OrbitQuantManifest:
    _validate_artifact_component(artifact_dir, component)
    target = _get_pipeline_component(pipeline, component)
    return load_orbitquant_artifact(
        target,
        artifact_dir,
        strict=strict,
        validate_checksums=validate_checksums,
        device=device,
    )
