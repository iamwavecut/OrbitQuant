from __future__ import annotations

import json
import re
import time
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
    if isinstance(frames, (list, tuple)) and len(frames) == 0:
        raise ValueError("pipeline output does not contain frames")
    return frames[0]


def _metadata_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".json")


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

    if is_video:
        try:
            from diffusers.utils import export_to_video
        except Exception as exc:
            raise RuntimeError(
                "diffusers video export utilities are required for video suites"
            ) from exc
        export_to_video(extract_video_frames(output), str(output_path))
    else:
        image = _extract_image(output)
        image.save(output_path)
    wall_time_seconds = time.perf_counter() - started_at

    metadata_path = _metadata_path(output_path)
    metadata = {
        "suite": suite.name,
        "model_id": suite.model_id,
        "prompt": prompt,
        "seed": seed,
        "height": suite.height,
        "width": suite.width,
        "frames": suite.frames,
        "steps": suite.steps,
        "guidance": suite.guidance,
        "bit_settings": suite.bit_settings,
        "wall_time_seconds": wall_time_seconds,
        "peak_vram_bytes": _peak_vram_bytes(device),
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
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return NativeGenerationResult(
        output_path=output_path,
        metadata_path=metadata_path,
        metadata=metadata,
    )
