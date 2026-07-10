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
from orbitquant.modeling import QuantizationSummary, quantize_model
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


def _artifact_model_index(artifact_dir: str | Path) -> dict[str, Any]:
    model_index_path = Path(artifact_dir) / "model_index.json"
    if not model_index_path.is_file():
        raise RuntimeError(f"required artifact file missing: {model_index_path}")
    return json.loads(model_index_path.read_text(encoding="utf-8"))


def _native_pipeline_class_name(source_model_id: str) -> str | None:
    from orbitquant.eval.native_settings import list_native_suites

    for suite in list_native_suites():
        if suite.model_id == source_model_id:
            return suite.pipeline
    return None


def _resolve_artifact_pipeline_cls(model_index: dict[str, Any], pipeline_cls: Any | None) -> Any:
    if pipeline_cls is not None:
        return pipeline_cls
    try:
        import diffusers
    except Exception as exc:
        raise ImportError(
            "load_quantized_pipeline_from_artifact requires diffusers or an explicit "
            "pipeline_cls"
        ) from exc

    source_model_id = model_index.get("source_model_id")
    if isinstance(source_model_id, str):
        native_pipeline_name = _native_pipeline_class_name(source_model_id)
        if native_pipeline_name is not None:
            native_pipeline_cls = getattr(diffusers, native_pipeline_name, None)
            if native_pipeline_cls is not None:
                return native_pipeline_cls
    return diffusers.DiffusionPipeline


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
    return quantize_model(
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
    runtime_mode: str | None = None,
    activation_kernel_backend: str | None = None,
) -> OrbitQuantManifest:
    _validate_artifact_component(artifact_dir, component)
    target = _get_pipeline_component(pipeline, component)
    return load_orbitquant_artifact(
        target,
        artifact_dir,
        strict=strict,
        validate_checksums=validate_checksums,
        device=device,
        runtime_mode=runtime_mode,
        activation_kernel_backend=activation_kernel_backend,
    )


def load_quantized_pipeline_from_artifact(
    artifact_dir: str | Path,
    *,
    pipeline_cls: Any | None = None,
    component: str | None = None,
    strict: bool = True,
    validate_checksums: bool = True,
    device: str | torch.device | None = None,
    runtime_mode: str | None = None,
    activation_kernel_backend: str | None = None,
    torch_dtype: torch.dtype | None = None,
    **from_pretrained_kwargs: Any,
) -> Any:
    """Load the source Diffusers pipeline and patch in an OrbitQuant component artifact."""

    artifact_path = Path(artifact_dir)
    model_index = _artifact_model_index(artifact_path)
    source_model_id = model_index.get("source_model_id")
    if not isinstance(source_model_id, str) or not source_model_id:
        raise RuntimeError("model_index.json is missing a non-empty source_model_id")

    resolved_component = component or model_index.get("component") or "transformer"
    _validate_artifact_component(artifact_path, resolved_component)

    pipeline_cls = _resolve_artifact_pipeline_cls(model_index, pipeline_cls)

    source_revision = model_index.get("source_revision")
    if (
        isinstance(source_revision, str)
        and source_revision
        and source_revision != "unknown"
        and "revision" not in from_pretrained_kwargs
    ):
        from_pretrained_kwargs["revision"] = source_revision
    if torch_dtype is not None and "torch_dtype" not in from_pretrained_kwargs:
        from_pretrained_kwargs["torch_dtype"] = torch_dtype

    pipeline = pipeline_cls.from_pretrained(source_model_id, **from_pretrained_kwargs)
    manifest = load_quantized_pipeline_component(
        pipeline,
        artifact_path,
        component=resolved_component,
        strict=strict,
        validate_checksums=validate_checksums,
        device=device,
        runtime_mode=runtime_mode,
        activation_kernel_backend=activation_kernel_backend,
    )
    pipeline.orbitquant_manifest = manifest
    pipeline.orbitquant_artifact_dir = str(artifact_path)
    if device is not None and hasattr(pipeline, "to"):
        pipeline = pipeline.to(device)
    return pipeline
