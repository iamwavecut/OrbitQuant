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
from orbitquant.quantizer import register_hf_quantizers


def _component_list(components: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(components, str):
        return [components]
    result = list(components)
    if not result:
        raise ValueError("components must not be empty")
    return result


def build_diffusers_pipeline_quantization_config(
    config: OrbitQuantConfig,
    *,
    components: str | list[str] | tuple[str, ...] = "transformer",
    granular: bool = True,
) -> Any:
    """Build a Diffusers PipelineQuantizationConfig for OrbitQuant components."""

    try:
        from diffusers.quantizers import PipelineQuantizationConfig
    except Exception as exc:
        raise ImportError(
            "build_diffusers_pipeline_quantization_config requires diffusers"
        ) from exc

    register_hf_quantizers()
    component_names = _component_list(components)
    if granular:
        return PipelineQuantizationConfig(
            quant_mapping={component: config for component in component_names}
        )
    return PipelineQuantizationConfig(
        quant_backend="orbitquant",
        quant_kwargs=config.to_dict(),
        components_to_quantize=component_names,
    )


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
    synchronize_per_module: bool = False,
) -> QuantizationSummary:
    target = _get_pipeline_component(pipeline, component)
    return quantize_linear_modules(
        target,
        config,
        quantization_device=quantization_device,
        staging_mode=staging_mode,
        synchronize_per_module=synchronize_per_module,
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
