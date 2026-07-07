from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from orbitquant.config import OrbitQuantConfig
from orbitquant.eval.native_settings import NativeSuite
from orbitquant.modeling import QuantizationSummary, quantize_linear_modules

_BIT_SETTING_RE = re.compile(r"^W(?P<weight>\d+)A(?P<activation>\d+)$")
_TARGET_POLICY_BY_SUITE = {
    "flux2-native": "flux2",
    "flux1-schnell-native": "flux",
    "z-image-native": "z_image",
    "wan-native": "wan",
}


@dataclass(frozen=True)
class NativeGenerationResult:
    output_path: Path
    metadata_path: Path
    metadata: dict[str, Any]
    asset_paths: list[Path]


def _cuda_device_index(device: str | torch.device) -> int | None:
    torch_device = torch.device(device)
    if torch_device.type != "cuda" or not torch.cuda.is_available():
        return None
    return torch_device.index if torch_device.index is not None else torch.cuda.current_device()


def _reset_peak_vram(device: str | torch.device) -> None:
    device_index = _cuda_device_index(device)
    if device_index is None:
        return
    torch.cuda.reset_peak_memory_stats(device_index)


def _peak_vram_bytes(device: str | torch.device) -> int | None:
    device_index = _cuda_device_index(device)
    if device_index is None:
        return None
    return int(torch.cuda.max_memory_allocated(device_index))


def _runtime_device_metadata(device: str | torch.device) -> dict[str, Any]:
    torch_device = torch.device(device)
    payload: dict[str, Any] = {
        "requested_device": str(device),
        "resolved_device_type": torch_device.type,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_active": False,
    }
    device_index = _cuda_device_index(device)
    if device_index is None:
        return payload
    capability = torch.cuda.get_device_capability(device_index)
    payload.update(
        {
            "cuda_active": True,
            "cuda_device_index": int(device_index),
            "cuda_device_name": torch.cuda.get_device_name(device_index),
            "cuda_device_capability": [int(capability[0]), int(capability[1])],
            "cuda_memory_allocated_bytes": int(torch.cuda.memory_allocated(device_index)),
            "cuda_memory_reserved_bytes": int(torch.cuda.memory_reserved(device_index)),
            "cuda_peak_memory_allocated_bytes": int(
                torch.cuda.max_memory_allocated(device_index)
            ),
        }
    )
    return payload


def build_pipeline_kwargs(
    suite: NativeSuite,
    *,
    prompt: str,
    seed: int,
    device: str | torch.device,
) -> dict[str, Any]:
    generator_device = torch.device(device)
    if generator_device.type == "mps":
        # PyTorch MPS generators are still inconsistent across versions; CPU is
        # deterministic and accepted by Diffusers pipelines.
        generator_device = torch.device("cpu")
    kwargs: dict[str, Any] = {
        "prompt": prompt,
        "height": suite.height,
        "width": suite.width,
        "num_inference_steps": suite.steps,
        "guidance_scale": suite.guidance,
        "generator": torch.Generator(device=generator_device).manual_seed(seed),
    }
    if suite.frames is not None:
        kwargs["num_frames"] = suite.frames
    return kwargs


def parse_bit_setting(bit_setting: str) -> tuple[int, int]:
    match = _BIT_SETTING_RE.match(bit_setting)
    if match is None:
        raise ValueError("bit_setting must use the form W<weight_bits>A<activation_bits>")
    return int(match.group("weight")), int(match.group("activation"))


def build_quantization_config_for_suite(
    suite: NativeSuite,
    bit_setting: str,
    *,
    rotation_seed: int = 0,
    runtime_mode: str = "dequant_bf16",
    activation_kernel_backend: str = "auto",
) -> OrbitQuantConfig:
    normalized = bit_setting.upper()
    if normalized not in suite.bit_settings:
        raise ValueError(
            f"bit setting {normalized!r} is not listed for native suite {suite.name!r}"
        )
    weight_bits, activation_bits = parse_bit_setting(normalized)
    return OrbitQuantConfig(
        weight_bits=weight_bits,
        activation_bits=activation_bits,
        rotation_seed=rotation_seed,
        runtime_mode=runtime_mode,
        activation_kernel_backend=activation_kernel_backend,
        target_policy=_TARGET_POLICY_BY_SUITE.get(suite.name, "auto"),
    )


def target_policy_for_suite(suite: NativeSuite) -> str:
    return _TARGET_POLICY_BY_SUITE.get(suite.name, "auto")


def load_pipeline_for_suite(
    suite: NativeSuite,
    *,
    model_id: str | None = None,
    **from_pretrained_kwargs: Any,
) -> Any:
    try:
        import diffusers
    except Exception as exc:
        raise RuntimeError("diffusers is required to load native pipelines") from exc
    pipeline_cls = getattr(diffusers, suite.pipeline, None)
    if pipeline_cls is None:
        pipeline_cls = diffusers.DiffusionPipeline
    return pipeline_cls.from_pretrained(
        suite.model_id if model_id is None else model_id,
        **from_pretrained_kwargs,
    )


def load_component_skeleton_for_suite(
    suite: NativeSuite,
    *,
    component: str = "transformer",
    model_id: str | None = None,
    revision: str | None = None,
    local_files_only: bool = False,
) -> torch.nn.Module:
    if component != "transformer":
        raise ValueError("config-only skeleton loading is only defined for transformer components")
    if suite.transformer_class is None:
        raise ValueError(f"native suite {suite.name!r} does not define a transformer class")
    try:
        import diffusers
    except Exception as exc:
        raise RuntimeError("diffusers is required to load native component skeletons") from exc
    try:
        from accelerate import init_empty_weights
    except Exception as exc:
        raise RuntimeError("accelerate is required to load native component skeletons") from exc

    transformer_cls = getattr(diffusers, suite.transformer_class, None)
    if transformer_cls is None:
        raise RuntimeError(f"diffusers has no class {suite.transformer_class!r}")
    load_kwargs: dict[str, Any] = {
        "subfolder": component,
        "local_files_only": local_files_only,
    }
    if revision is not None:
        load_kwargs["revision"] = revision
    config = transformer_cls.load_config(
        suite.model_id if model_id is None else model_id,
        **load_kwargs,
    )
    with init_empty_weights():
        skeleton = transformer_cls.from_config(config)
    if not isinstance(skeleton, torch.nn.Module):
        raise TypeError(f"{suite.transformer_class} did not produce a torch.nn.Module")
    return skeleton


def apply_quantization_to_pipeline(
    pipeline: Any, suite: NativeSuite, config: OrbitQuantConfig
) -> QuantizationSummary:
    transformer = getattr(pipeline, "transformer", None)
    if transformer is None:
        raise ValueError(
            f"native suite {suite.name!r} expects a pipeline with a transformer component"
        )
    return quantize_linear_modules(transformer, config)


def output_path_for_suite(
    output_dir: str | Path,
    *,
    suite_name: str,
    seed: int,
    media_type: str,
    variant: str | None = None,
) -> Path:
    suffixes = {"image": ".png", "video": ".mp4"}
    try:
        suffix = suffixes[media_type]
    except KeyError as exc:
        raise ValueError(f"unknown media_type {media_type!r}") from exc
    variant_suffix = "" if variant is None else f"_{variant}"
    return Path(output_dir) / f"{suite_name}_seed{seed}{variant_suffix}{suffix}"


def _extract_image(output: Any) -> Any:
    images = getattr(output, "images", None)
    if not images:
        raise ValueError("pipeline output does not contain images")
    return images[0]


def extract_video_frames(output: Any) -> Any:
    frames = getattr(output, "frames", None)
    if frames is None:
        raise ValueError("pipeline output does not contain frames")
    if isinstance(frames, list | tuple) and len(frames) == 0:
        raise ValueError("pipeline output does not contain frames")
    return frames[0]


def _metadata_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".json")


def _contact_sheet_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_contact_sheet.webp")


def _video_contact_sheet_indices(frame_count: int, *, sample_count: int = 9) -> list[int]:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if frame_count <= sample_count:
        return list(range(frame_count))
    return sorted(
        {
            round(index * (frame_count - 1) / (sample_count - 1))
            for index in range(sample_count)
        }
    )


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_json_safe(item) for item in value]
    if hasattr(value, "to_dict"):
        return _json_safe(value.to_dict())
    return str(value)


def _scheduler_metadata(pipeline: Any) -> dict[str, Any] | None:
    scheduler = getattr(pipeline, "scheduler", None)
    if scheduler is None:
        return None
    config = getattr(scheduler, "config", None)
    return {
        "class": scheduler.__class__.__name__,
        "config": _json_safe(config),
    }


def _metadata_mismatch(key: str, expected: Any, actual: Any) -> str | None:
    if isinstance(expected, float):
        try:
            if abs(float(actual) - expected) <= 1e-9:
                return None
        except (TypeError, ValueError):
            pass
    elif actual == expected:
        return None
    return f"{key}: expected {expected!r}, got {actual!r}"


def validate_native_generation_output(
    output_path: str | Path,
    metadata_path: str | Path,
    suite: NativeSuite,
    *,
    seed: int,
    bit_setting: str,
    prompt: str | None = None,
    model_id: str | None = None,
) -> dict[str, Any]:
    output = Path(output_path)
    metadata_file = Path(metadata_path)
    if not output.is_file():
        raise RuntimeError(f"native generation output missing: {output}")
    if output.stat().st_size <= 0:
        raise RuntimeError(f"native generation output is empty: {output}")
    if not metadata_file.is_file():
        raise RuntimeError(f"native generation metadata missing: {metadata_file}")
    try:
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"native generation metadata is not valid JSON: {metadata_file}"
        ) from exc

    expected_model_id = suite.model_id if model_id is None else model_id
    mismatches = [
        mismatch
        for mismatch in (
            _metadata_mismatch("suite", suite.name, metadata.get("suite")),
            _metadata_mismatch("model_id", expected_model_id, metadata.get("model_id")),
            _metadata_mismatch("seed", seed, metadata.get("seed")),
            _metadata_mismatch("height", suite.height, metadata.get("height")),
            _metadata_mismatch("width", suite.width, metadata.get("width")),
            _metadata_mismatch("frames", suite.frames, metadata.get("frames")),
            _metadata_mismatch("export_fps", suite.export_fps, metadata.get("export_fps")),
            _metadata_mismatch("steps", suite.steps, metadata.get("steps")),
            _metadata_mismatch("guidance", suite.guidance, metadata.get("guidance")),
            None
            if prompt is None
            else _metadata_mismatch("prompt", prompt, metadata.get("prompt")),
        )
        if mismatch is not None
    ]

    if bit_setting == "original":
        if metadata.get("quantization") is not None:
            mismatches.append("quantization: expected None for original split")
    else:
        weight_bits, activation_bits = parse_bit_setting(bit_setting)
        quantization = metadata.get("quantization")
        config = quantization.get("config") if isinstance(quantization, Mapping) else None
        if not isinstance(config, Mapping):
            mismatches.append("quantization.config: expected quantized metadata")
        else:
            for key, expected in (
                ("weight_bits", weight_bits),
                ("activation_bits", activation_bits),
            ):
                mismatch = _metadata_mismatch(
                    f"quantization.config.{key}",
                    expected,
                    config.get(key),
                )
                if mismatch is not None:
                    mismatches.append(mismatch)

    if mismatches:
        raise RuntimeError(
            "native generation metadata mismatch: " + "; ".join(mismatches)
        )
    return {
        "valid": True,
        "output_path": str(output),
        "metadata_path": str(metadata_file),
        "suite": suite.name,
        "seed": seed,
        "bit_setting": bit_setting,
    }


def run_native_generation(
    pipeline: Any,
    suite: NativeSuite,
    *,
    prompt: str,
    seed: int,
    output_dir: str | Path,
    device: str | torch.device,
    quantization_config: OrbitQuantConfig | None = None,
    quantization_summary: QuantizationSummary | None = None,
    quantization_label: str | None = None,
    prewarm_metadata: dict[str, Any] | None = None,
    runtime_dtype: str | None = None,
    model_id: str | None = None,
) -> NativeGenerationResult:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    is_video = suite.frames is not None
    output_path = output_path_for_suite(
        output_root,
        suite_name=suite.name,
        seed=seed,
        media_type="video" if is_video else "image",
        variant=quantization_label,
    )
    kwargs = build_pipeline_kwargs(suite, prompt=prompt, seed=seed, device=device)
    _reset_peak_vram(device)
    started_at = time.perf_counter()
    output = pipeline(**kwargs)

    asset_paths: list[Path] = []
    contact_sheet_path: Path | None = None
    if is_video:
        try:
            from diffusers.utils import export_to_video
        except Exception as exc:
            raise RuntimeError(
                "diffusers video export utilities are required for video suites"
            ) from exc
        frames = extract_video_frames(output)
        export_kwargs = {} if suite.export_fps is None else {"fps": suite.export_fps}
        export_to_video(frames, str(output_path), **export_kwargs)
        from orbitquant.eval.assets import create_video_contact_sheet

        contact_sheet_path = _contact_sheet_path(output_path)
        create_video_contact_sheet(
            frames,
            contact_sheet_path,
            sample_indices=_video_contact_sheet_indices(len(frames)),
        )
        asset_paths.append(contact_sheet_path)
    else:
        image = _extract_image(output)
        image.save(output_path)
    wall_time_seconds = time.perf_counter() - started_at

    metadata_path = _metadata_path(output_path)
    metadata = {
        "suite": suite.name,
        "model_id": suite.model_id if model_id is None else model_id,
        "prompt": prompt,
        "seed": seed,
        "height": suite.height,
        "width": suite.width,
        "frames": suite.frames,
        "export_fps": suite.export_fps,
        "steps": suite.steps,
        "guidance": suite.guidance,
        "bit_settings": suite.bit_settings,
        "device": str(device),
        "runtime_device": _runtime_device_metadata(device),
        "dtype": runtime_dtype,
        "pipeline_class": pipeline.__class__.__name__,
        "scheduler": _scheduler_metadata(pipeline),
        "wall_time_seconds": wall_time_seconds,
        "peak_vram_bytes": _peak_vram_bytes(device),
        "contact_sheet_path": None
        if contact_sheet_path is None
        else str(contact_sheet_path),
        "quantization": None
        if quantization_config is None
        else {
            "config": quantization_config.to_dict(),
            "summary": None
            if quantization_summary is None
            else {
                "quantized_modules": quantization_summary.quantized_modules,
                "adaln_modules": quantization_summary.adaln_modules,
                "skipped_modules": quantization_summary.skipped_modules,
            },
            "prewarm": prewarm_metadata,
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return NativeGenerationResult(
        output_path=output_path,
        metadata_path=metadata_path,
        metadata=metadata,
        asset_paths=asset_paths,
    )
