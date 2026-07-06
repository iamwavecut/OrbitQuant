from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from orbitquant.eval.native_settings import NativeSuite


@dataclass(frozen=True)
class NativeGenerationResult:
    output_path: Path
    metadata_path: Path


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


def output_path_for_suite(
    output_dir: str | Path, *, suite_name: str, seed: int, media_type: str
) -> Path:
    suffixes = {"image": ".png", "video": ".mp4"}
    try:
        suffix = suffixes[media_type]
    except KeyError as exc:
        raise ValueError(f"unknown media_type {media_type!r}") from exc
    return Path(output_dir) / f"{suite_name}_seed{seed}{suffix}"


def _extract_image(output: Any) -> Any:
    images = getattr(output, "images", None)
    if not images:
        raise ValueError("pipeline output does not contain images")
    return images[0]


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
) -> NativeGenerationResult:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    is_video = suite.frames is not None
    output_path = output_path_for_suite(
        output_root, suite_name=suite.name, seed=seed, media_type="video" if is_video else "image"
    )
    kwargs = build_pipeline_kwargs(suite, prompt=prompt, seed=seed, device=device)
    output = pipeline(**kwargs)

    if is_video:
        try:
            from diffusers.utils import export_to_video
        except Exception as exc:
            raise RuntimeError(
                "diffusers video export utilities are required for video suites"
            ) from exc
        frames = getattr(output, "frames", None)
        if not frames:
            raise ValueError("pipeline output does not contain frames")
        export_to_video(frames[0], str(output_path))
    else:
        image = _extract_image(output)
        image.save(output_path)

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
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return NativeGenerationResult(output_path=output_path, metadata_path=metadata_path)
