from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from orbitquant import __version__
from orbitquant.artifacts import (
    create_artifact_image_comparisons,
    record_artifact_asset,
    record_artifact_metrics,
    refresh_artifact_checksums,
    repair_artifact_metadata,
    save_orbitquant_artifact,
    validate_artifact_policy_inventory,
    validate_orbitquant_artifact,
)
from orbitquant.benchmarks import benchmark_model_quantization, benchmark_orbit_linear
from orbitquant.config import OrbitQuantConfig
from orbitquant.eval import list_native_suites
from orbitquant.eval.external_export import export_geneval_artifact, export_vbench_artifact
from orbitquant.eval.external_metrics import (
    summarize_geneval_results,
    summarize_vbench_results,
)
from orbitquant.eval.external_plan import build_external_eval_plan, build_external_eval_script
from orbitquant.eval.metrics import load_metric_json
from orbitquant.eval.native_plan import build_native_eval_plan, build_native_run_script
from orbitquant.eval.native_runner import (
    apply_quantization_to_pipeline,
    build_pipeline_kwargs,
    build_quantization_config_for_suite,
    load_component_skeleton_for_suite,
    load_pipeline_for_suite,
    output_path_for_suite,
    run_native_generation,
    target_policy_for_suite,
    validate_native_generation_output,
)
from orbitquant.eval.native_settings import get_native_suite
from orbitquant.eval.prompts import (
    build_prompt_seed_jobs,
    default_prompt_payload,
    geneval_smoke_prompt_payload,
    load_geneval_prompt_payload,
    select_prompt_record,
)
from orbitquant.eval.report import generate_native_eval_report
from orbitquant.hub import (
    audit_hf_artifact_repos,
    cleanup_hf_artifact_reports,
    cleanup_hf_artifact_reports_matrix,
    fetch_hf_artifacts,
    inspect_model_metadata,
    render_hf_artifact_audit_markdown,
    repair_hf_artifact_metadata,
    repair_hf_artifact_metadata_matrix,
    repair_hf_native_smoke_proof,
    repair_hf_native_smoke_proof_matrix,
    upload_orbitquant_artifact,
)
from orbitquant.kernels import available_backends, backend_capabilities
from orbitquant.layers import OrbitQuantLinear
from orbitquant.modeling import (
    inspect_linear_module_policy,
    prewarm_quantized_linear_modules,
    quantize_linear_modules,
)
from orbitquant.pipeline import load_quantized_pipeline_component

_RUNTIME_MODE_CHOICES = [
    "auto_fused",
    "dequant_bf16",
    "debug_no_quant",
    "debug_no_activation_quant",
    "triton_packed_matmul",
    "native_packed_matmul",
]
_PACKED_MATMUL_RUNTIME_MODES = {
    "auto_fused",
    "triton_packed_matmul",
    "native_packed_matmul",
}


def _torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _pipeline_component(pipeline: Any, component: str) -> torch.nn.Module:
    try:
        value = getattr(pipeline, component)
    except AttributeError as exc:
        raise ValueError(f"pipeline has no component {component!r}") from exc
    if not isinstance(value, torch.nn.Module):
        raise TypeError(f"pipeline component {component!r} is not a torch.nn.Module")
    return value


def _prewarm_pipeline_component(
    pipeline: Any,
    component: str,
    *,
    device: str | torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    summary = prewarm_quantized_linear_modules(
        _pipeline_component(pipeline, component),
        device=device,
        dtype=dtype,
    )
    return {
        "orbitquant_modules": summary.orbitquant_modules,
        "adaln_modules": summary.adaln_modules,
        "total_modules": summary.total_modules,
        "elapsed_seconds": summary.elapsed_seconds,
        "device": summary.device,
        "dtype": summary.dtype,
    }


def _should_prewarm_quantized_weights(config: OrbitQuantConfig | None) -> bool:
    return config is None or config.runtime_mode not in _PACKED_MATMUL_RUNTIME_MODES


def _place_pipeline_for_generation(
    pipeline: Any,
    *,
    device: str,
    enable_model_cpu_offload: bool,
) -> None:
    if enable_model_cpu_offload:
        if not hasattr(pipeline, "enable_model_cpu_offload"):
            raise RuntimeError(
                "pipeline does not support enable_model_cpu_offload(); use a Diffusers "
                "pipeline with model CPU offload support, or omit "
                "--enable-model-cpu-offload and ensure the full pipeline fits on the device"
            )
        pipeline.enable_model_cpu_offload(device=device)
        return
    pipeline.to(device)


def _release_generation_pipeline(pipeline: Any, *, device: str) -> None:
    del pipeline
    gc.collect()
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch_device.type == "mps" and hasattr(torch, "mps"):
        empty_cache = getattr(torch.mps, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()


def _with_runtime_overrides(
    config: OrbitQuantConfig,
    *,
    runtime_mode: str | None,
    activation_kernel_backend: str | None,
) -> OrbitQuantConfig:
    payload = config.to_dict()
    if runtime_mode is not None:
        payload["runtime_mode"] = runtime_mode
    if activation_kernel_backend is not None:
        payload["activation_kernel_backend"] = activation_kernel_backend
    return OrbitQuantConfig.from_dict(payload)


def _comparison_image_source(result: Any, *, is_video: bool) -> Path:
    if is_video:
        if not result.asset_paths:
            raise RuntimeError("video comparison requires generated contact sheet assets")
        return result.asset_paths[0]
    return result.output_path


def _resolve_local_bundle_path(bundle_dir: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{label} is missing or is not a path string")
    path = Path(value)
    candidates = [path]
    if path.is_absolute():
        candidates.append(bundle_dir / path.name)
    else:
        candidates.append(bundle_dir / path)
        candidates.append(bundle_dir / path.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RuntimeError(f"{label} does not exist locally: {value}")


def _image_validation_stats(path: Path, *, blank_stddev_threshold: float) -> dict[str, Any]:
    from PIL import Image, ImageStat

    with Image.open(path) as image:
        rgb = image.convert("RGB")
        stat = ImageStat.Stat(rgb)
        mean = [float(value) for value in stat.mean]
        stddev = [float(value) for value in stat.stddev]
        extrema = [[int(low), int(high)] for low, high in rgb.getextrema()]
        blank_like = max(stddev) < blank_stddev_threshold
        return {
            "path": str(path),
            "size": [int(rgb.width), int(rgb.height)],
            "mode": rgb.mode,
            "mean": mean,
            "stddev": stddev,
            "extrema": extrema,
            "blank_like": blank_like,
            "blank_stddev_threshold": blank_stddev_threshold,
        }


def _validate_compare_native_bundle(
    bundle_dir: str | Path,
    *,
    blank_stddev_threshold: float = 1.0,
) -> dict[str, Any]:
    bundle_path = Path(bundle_dir)
    summary_path = bundle_path / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"comparison summary missing: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    suite = get_native_suite(str(summary["suite"]))
    bit_setting = str(summary["bit_setting"])
    seed = int(summary["seed"])
    prompt = str(summary["prompt"])
    model_id = str(summary["model_id"])
    frames = summary.get("frames")
    is_video = frames is not None

    split_payloads: dict[str, dict[str, Any]] = {}
    for split, expected_bit_setting in (
        ("original", "original"),
        ("orbitquant", bit_setting),
    ):
        split_summary = summary.get(split)
        if not isinstance(split_summary, dict):
            raise RuntimeError(f"{split} summary missing")
        output_path = _resolve_local_bundle_path(
            bundle_path,
            split_summary.get("output_path"),
            label=f"{split}.output_path",
        )
        metadata_path = _resolve_local_bundle_path(
            bundle_path,
            split_summary.get("metadata_path"),
            label=f"{split}.metadata_path",
        )
        validate_native_generation_output(
            output_path,
            metadata_path,
            suite,
            seed=seed,
            bit_setting=expected_bit_setting,
            prompt=prompt,
            model_id=model_id,
        )
        split_payloads[split] = {
            "output_path": str(output_path),
            "metadata_path": str(metadata_path),
            "wall_time_seconds": split_summary.get("wall_time_seconds"),
            "peak_vram_bytes": split_summary.get("peak_vram_bytes"),
        }
        if not is_video:
            split_payloads[split]["image"] = _image_validation_stats(
                output_path,
                blank_stddev_threshold=blank_stddev_threshold,
            )

    comparison_path = _resolve_local_bundle_path(
        bundle_path,
        summary.get("comparison_path"),
        label="comparison_path",
    )
    comparison_stats = _image_validation_stats(
        comparison_path,
        blank_stddev_threshold=blank_stddev_threshold,
    )
    if comparison_stats["blank_like"]:
        raise RuntimeError(f"comparison image looks blank: {comparison_path}")
    if not is_video:
        expected_size = [suite.width * 2, suite.height + 24]
        if comparison_stats["size"] != expected_size:
            raise RuntimeError(
                "comparison image size mismatch: "
                f"expected {expected_size}, got {comparison_stats['size']}"
            )
        for split in ("original", "orbitquant"):
            image_stats = split_payloads[split]["image"]
            if image_stats["size"] != [suite.width, suite.height]:
                raise RuntimeError(
                    f"{split} image size mismatch: expected "
                    f"{[suite.width, suite.height]}, got {image_stats['size']}"
                )
            if image_stats["blank_like"]:
                raise RuntimeError(f"{split} image looks blank: {image_stats['path']}")

    original_time = split_payloads["original"]["wall_time_seconds"]
    orbitquant_time = split_payloads["orbitquant"]["wall_time_seconds"]
    speed_ratio = None
    if (
        isinstance(original_time, int | float)
        and isinstance(orbitquant_time, int | float)
        and original_time > 0
    ):
        speed_ratio = float(orbitquant_time) / float(original_time)

    return {
        "valid": True,
        "bundle": str(bundle_path),
        "summary_path": str(summary_path),
        "suite": suite.name,
        "model_id": model_id,
        "bit_setting": bit_setting,
        "seed": seed,
        "prompt": prompt,
        "native_settings": {
            "width": suite.width,
            "height": suite.height,
            "frames": suite.frames,
            "steps": suite.steps,
            "guidance": suite.guidance,
        },
        "runtime_mode": summary.get("runtime_mode"),
        "activation_kernel_backend": summary.get("activation_kernel_backend"),
        "enable_model_cpu_offload": summary.get("enable_model_cpu_offload"),
        "available_backends": summary.get("available_backends"),
        "original": split_payloads["original"],
        "orbitquant": split_payloads["orbitquant"],
        "orbitquant_runtime": (
            summary.get("orbitquant", {}).get("runtime")
            if isinstance(summary.get("orbitquant"), dict)
            else None
        ),
        "comparison": comparison_stats,
        "speed_ratio_orbitquant_over_original": speed_ratio,
    }


def _orbitquant_runtime_summary(pipeline: Any, component: str) -> dict[str, Any]:
    target = _pipeline_component(pipeline, component)
    runtime_counts: dict[str, int] = {}
    activation_backend_counts: dict[str, int] = {}
    device_type_counts: dict[str, int] = {}
    unexecuted_modules: list[str] = []
    module_count = 0
    executed_count = 0
    for name, module in target.named_modules():
        if not isinstance(module, OrbitQuantLinear):
            continue
        module_count += 1
        runtime_mode = module.last_effective_runtime_mode
        if runtime_mode is None:
            unexecuted_modules.append(name)
            continue
        executed_count += 1
        runtime_counts[runtime_mode] = runtime_counts.get(runtime_mode, 0) + 1
        activation_backend = module.last_activation_kernel_backend
        activation_backend_key = "none" if activation_backend is None else activation_backend
        activation_backend_counts[activation_backend_key] = (
            activation_backend_counts.get(activation_backend_key, 0) + 1
        )
        device_type = module.last_forward_device_type or "unknown"
        device_type_counts[device_type] = device_type_counts.get(device_type, 0) + 1
    return {
        "orbitquant_linear_count": module_count,
        "executed_module_count": executed_count,
        "runtime_mode_counts": runtime_counts,
        "activation_kernel_backend_counts": activation_backend_counts,
        "forward_device_type_counts": device_type_counts,
        "unexecuted_module_count": len(unexecuted_modules),
        "unexecuted_module_sample": unexecuted_modules[:20],
    }


def _hf_artifact_audit_regressions(payload: dict[str, Any]) -> list[str]:
    repo_count = int(payload.get("repo_count") or 0)
    regressions: list[str] = []
    if repo_count <= 0:
        regressions.append("repo_count is zero")

    for key in (
        "existing_count",
        "artifact_ready_count",
        "native_smoke_ready_count",
        "metadata_complete_ready_count",
    ):
        if int(payload.get(key) or 0) != repo_count:
            regressions.append(f"{key}={payload.get(key, 0)} expected {repo_count}")

    if (
        payload.get("policy_inventory_root") is not None
        and int(payload.get("policy_inventory_ready_count") or 0) != repo_count
    ):
        regressions.append(
            "policy_inventory_ready_count="
            f"{payload.get('policy_inventory_ready_count', 0)} expected {repo_count}"
        )

    for key in (
        "policy_inventory_error_count",
        "manifest_warning_count",
        "metadata_missing_count",
        "remote_checksum_mismatch_count",
        "forbidden_file_count",
    ):
        if int(payload.get(key) or 0) != 0:
            regressions.append(f"{key}={payload.get(key, 0)} expected 0")
    return regressions


def _hf_artifact_audit_summary(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload.get("rows") or []
    summary_keys = (
        "namespace",
        "policy_inventory_root",
        "repo_count",
        "existing_count",
        "artifact_ready_count",
        "native_smoke_ready_count",
        "metadata_complete_ready_count",
        "policy_inventory_ready_count",
        "policy_inventory_error_count",
        "release_eval_applicable_count",
        "release_eval_not_applicable_count",
        "release_eval_ready_count",
        "missing_required_metric_count",
        "manifest_warning_count",
        "metadata_missing_count",
        "remote_checksum_mismatch_count",
        "readme_mismatch_count",
        "forbidden_file_count",
    )
    summary = {key: payload.get(key) for key in summary_keys if key in payload}
    summary["row_count"] = len(rows)
    summary["repos"] = [
        {
            "repo_id": row.get("repo_id"),
            "suite": row.get("suite"),
            "bit_setting": row.get("bit_setting"),
            "artifact_ready": row.get("artifact_ready"),
            "native_smoke_ready": row.get("native_smoke_ready"),
            "release_eval_applicable": row.get("release_eval_applicable"),
            "release_eval_ready": row.get("release_eval_ready"),
            "metadata_complete_ready": row.get("metadata_complete_ready"),
            "policy_inventory_ready": row.get("policy_inventory_ready"),
            "forbidden_file_count": row.get("forbidden_file_count"),
            "missing_required_metric_count": len(
                row.get("missing_required_metrics") or []
            ),
        }
        for row in rows
    ]
    return summary


def _parse_block_size(value: str) -> int | str:
    if value == "paper":
        return value
    try:
        return int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("block size must be 'paper' or an integer") from exc


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _path_is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _parse_seed_list(value: str) -> list[int]:
    try:
        seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be a comma-separated integer list") from exc
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def _load_generate_pack_prompt_payload(
    *,
    artifact_path: Path,
    suite: Any,
    prompt_pack: str,
    prompt_metadata_jsonl: str | None,
) -> dict[str, Any]:
    target_policy = target_policy_for_suite(suite)
    if prompt_metadata_jsonl is not None:
        if prompt_pack != "artifact":
            raise ValueError("--prompt-metadata-jsonl cannot be combined with --prompt-pack")
        if suite.frames is not None:
            raise ValueError("GenEval prompt metadata is only valid for image suites")
        return load_geneval_prompt_payload(
            prompt_metadata_jsonl,
            target_policy=target_policy,
        )
    if prompt_pack == "artifact":
        return json.loads((artifact_path / "prompts.json").read_text(encoding="utf-8"))
    if prompt_pack == "visual":
        return default_prompt_payload(target_policy)
    if prompt_pack == "geneval-smoke":
        if suite.frames is not None:
            raise ValueError("GenEval prompt packs are only valid for image suites")
        return geneval_smoke_prompt_payload(target_policy)
    raise ValueError(f"unknown prompt pack {prompt_pack!r}")


def _metadata_path_for_output(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".json")


def _expected_generation_output_path(
    output_dir: Path,
    *,
    suite: Any,
    seed: int,
    variant: str,
) -> Path:
    return output_path_for_suite(
        output_dir,
        suite_name=suite.name,
        seed=seed,
        media_type="video" if suite.frames is not None else "image",
        variant=variant,
    )


def _policy_inventory_summary(payload: dict[str, Any], output_path: Path) -> dict[str, Any]:
    return {
        "source_model_id": payload["source_model_id"],
        "source_revision": payload["source_revision"],
        "suite": payload["suite"],
        "component": payload["component"],
        "load_mode": payload["load_mode"],
        "pipeline_class": payload["pipeline_class"],
        "component_class": payload["component_class"],
        "target_policy": payload["target_policy"],
        "linear_module_count": payload["linear_module_count"],
        "action_counts": payload["action_counts"],
        "quantized_module_count": len(payload["quantized_modules"]),
        "adaln_module_count": len(payload["adaln_modules"]),
        "skipped_module_count": len(payload["skipped_modules"]),
        "output": str(output_path),
    }


def _record_generated_artifact(
    artifact_path: Path,
    result: Any,
    *,
    split: str,
    suite: Any,
    prompt: str,
    prompt_record: dict[str, Any] | None,
    seed: int,
    bit_setting: str | None,
    validate_checksums_enabled: bool = True,
    refresh_checksums_enabled: bool = True,
    create_comparisons_enabled: bool = True,
) -> tuple[dict[str, Any], list[str]]:
    if _path_is_relative_to(result.output_path, artifact_path):
        record_artifact_asset(
            artifact_path,
            result.output_path,
            validate_checksums_enabled=validate_checksums_enabled,
            refresh_checksums_enabled=refresh_checksums_enabled,
        )
    if _path_is_relative_to(result.metadata_path, artifact_path):
        record_artifact_asset(
            artifact_path,
            result.metadata_path,
            validate_checksums_enabled=validate_checksums_enabled,
            refresh_checksums_enabled=refresh_checksums_enabled,
        )
    for asset_path in result.asset_paths:
        if _path_is_relative_to(asset_path, artifact_path):
            record_artifact_asset(
                artifact_path,
                asset_path,
                validate_checksums_enabled=validate_checksums_enabled,
                refresh_checksums_enabled=refresh_checksums_enabled,
            )
    metrics = {
        "generated_samples": 1,
        "wall_time_seconds": result.metadata["wall_time_seconds"],
    }
    if suite.frames is not None:
        metrics["generated_frames"] = suite.frames
    if result.metadata["peak_vram_bytes"] is not None:
        metrics["peak_vram_bytes"] = result.metadata["peak_vram_bytes"]
    metrics_record = record_artifact_metrics(
        artifact_path,
        split=split,
        metrics=metrics,
        metadata={
            "suite": suite.name,
            "prompt": prompt,
            "prompt_record": prompt_record,
            "seed": seed,
            "height": suite.height,
            "width": suite.width,
            "frames": suite.frames,
            "export_fps": suite.export_fps,
            "steps": suite.steps,
            "guidance": suite.guidance,
            "bit_setting": bit_setting,
            "output_path": str(result.output_path),
            "metadata_path": str(result.metadata_path),
            "asset_paths": [str(asset_path) for asset_path in result.asset_paths],
            "device": result.metadata["device"],
            "runtime_device": result.metadata.get("runtime_device"),
            "dtype": result.metadata["dtype"],
            "pipeline_class": result.metadata["pipeline_class"],
            "scheduler": result.metadata["scheduler"],
        },
        validate_checksums_enabled=validate_checksums_enabled,
        refresh_checksums_enabled=refresh_checksums_enabled,
    )
    comparisons: list[str] = []
    if create_comparisons_enabled:
        comparisons = create_artifact_image_comparisons(
            artifact_path,
            validate_checksums_enabled=validate_checksums_enabled,
            refresh_checksums_enabled=refresh_checksums_enabled,
        )
    return metrics_record, comparisons


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="orbitquant")
    parser.add_argument("--version", action="store_true", help="print OrbitQuant version")
    subparsers = parser.add_subparsers(dest="command")

    inspect_parser = subparsers.add_parser("inspect", help="inspect Hugging Face model metadata")
    inspect_parser.add_argument("--model-id", required=True)
    inspect_parser.add_argument("--revision")
    inspect_policy_parser = subparsers.add_parser(
        "inspect-policy",
        help="load a Diffusers pipeline component and emit OrbitQuant policy inventory",
    )
    inspect_policy_parser.add_argument("--suite")
    inspect_policy_parser.add_argument("--model-id")
    inspect_policy_parser.add_argument("--revision")
    inspect_policy_parser.add_argument("--component", default="transformer")
    inspect_policy_parser.add_argument("--target-policy", default="auto")
    inspect_policy_parser.add_argument(
        "--load-mode",
        default="config",
        choices=["config", "pipeline"],
        help=(
            "config loads a component skeleton without model weights for native suites; "
            "pipeline loads the full Diffusers pipeline"
        ),
    )
    inspect_policy_parser.add_argument("--local-files-only", action="store_true")
    inspect_policy_parser.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"]
    )
    inspect_policy_parser.add_argument("--output")
    subparsers.add_parser("native-suites", help="list native eval suites")
    subparsers.add_parser("kernel-info", help="print activation kernel backend capabilities")
    kernel_bench_parser = subparsers.add_parser(
        "kernel-bench", help="benchmark OrbitQuantLinear kernel stages"
    )
    kernel_bench_parser.add_argument("--tokens", type=int, default=1024)
    kernel_bench_parser.add_argument("--in-features", type=int, default=3072)
    kernel_bench_parser.add_argument("--out-features", type=int, default=3072)
    kernel_bench_parser.add_argument("--weight-bits", type=int, default=4)
    kernel_bench_parser.add_argument("--activation-bits", type=int, default=4)
    kernel_bench_parser.add_argument("--block-size", type=_parse_block_size, default="paper")
    kernel_bench_parser.add_argument(
        "--activation-kernel-backend",
        default="auto",
        choices=["auto", "cpu", "mps", "triton_cuda"],
    )
    kernel_bench_parser.add_argument(
        "--runtime-mode",
        default="auto_fused",
        choices=_RUNTIME_MODE_CHOICES,
    )
    kernel_bench_parser.add_argument("--device", default="auto")
    kernel_bench_parser.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"]
    )
    kernel_bench_parser.add_argument("--warmup", type=int, default=5)
    kernel_bench_parser.add_argument("--iterations", type=int, default=20)
    kernel_bench_parser.add_argument("--seed", type=int, default=0)
    kernel_bench_parser.add_argument("--packed-matmul-block-m", type=int, default=32)
    kernel_bench_parser.add_argument("--packed-matmul-block-n", type=int, default=128)
    kernel_bench_parser.add_argument("--packed-matmul-block-k", type=int, default=64)
    kernel_bench_parser.add_argument("--packed-matmul-num-warps", type=int, default=8)

    quantize_bench_parser = subparsers.add_parser(
        "quantize-bench", help="benchmark full model quantization staging"
    )
    quantize_bench_parser.add_argument("--layers", type=int, default=4)
    quantize_bench_parser.add_argument("--in-features", type=int, default=3072)
    quantize_bench_parser.add_argument("--hidden-features", type=int)
    quantize_bench_parser.add_argument("--weight-bits", type=int, default=4)
    quantize_bench_parser.add_argument("--activation-bits", type=int, default=4)
    quantize_bench_parser.add_argument("--block-size", type=_parse_block_size, default="paper")
    quantize_bench_parser.add_argument("--source-device", default="cpu")
    quantize_bench_parser.add_argument("--quantization-device", default="auto")
    quantize_bench_parser.add_argument(
        "--staging-mode",
        default="component",
        choices=["streaming", "component"],
    )
    quantize_bench_parser.add_argument("--synchronize-per-module", action="store_true")
    quantize_bench_parser.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"]
    )
    quantize_bench_parser.add_argument("--seed", type=int, default=0)

    native_plan_parser = subparsers.add_parser(
        "native-plan", help="print native quant/eval job matrix"
    )
    native_plan_parser.add_argument("--suite", action="append")
    native_plan_parser.add_argument("--output-root", default="artifacts/native")
    native_plan_parser.add_argument("--seeds", type=_parse_seed_list, default=[0])
    native_plan_parser.add_argument(
        "--runtime-mode",
        default="auto_fused",
        choices=_RUNTIME_MODE_CHOICES,
    )

    external_eval_plan_parser = subparsers.add_parser(
        "external-eval-plan", help="print GenEval/VBench runner and metric import jobs"
    )
    external_eval_plan_parser.add_argument("--suite", action="append")
    external_eval_plan_parser.add_argument("--output-root", default="artifacts/native")
    external_eval_plan_parser.add_argument("--metrics-root", default="metrics/native")

    external_eval_script_parser = subparsers.add_parser(
        "external-eval-script", help="print a bash script for GenEval/VBench metric import"
    )
    external_eval_script_parser.add_argument("--suite", action="append")
    external_eval_script_parser.add_argument("--output-root", default="artifacts/native")
    external_eval_script_parser.add_argument("--metrics-root", default="metrics/native")
    external_eval_script_parser.add_argument("--report-output", default="reports/native")

    export_geneval_parser = subparsers.add_parser(
        "export-geneval", help="export generated artifact images to GenEval folder layout"
    )
    export_geneval_parser.add_argument("--artifact", required=True)
    export_geneval_parser.add_argument("--split", required=True, choices=["original", "orbitquant"])
    export_geneval_parser.add_argument("--output", required=True)

    export_vbench_parser = subparsers.add_parser(
        "export-vbench", help="export generated artifact videos to VBench custom input layout"
    )
    export_vbench_parser.add_argument("--artifact", required=True)
    export_vbench_parser.add_argument("--split", required=True, choices=["original", "orbitquant"])
    export_vbench_parser.add_argument("--output", required=True)
    export_vbench_parser.add_argument(
        "--link-mode",
        default="symlink",
        choices=["symlink", "hardlink", "copy"],
    )

    summarize_geneval_parser = subparsers.add_parser(
        "summarize-geneval-results", help="summarize GenEval results.jsonl to JSON metrics"
    )
    summarize_geneval_parser.add_argument("--results-jsonl", required=True)
    summarize_geneval_parser.add_argument("--output", required=True)

    summarize_vbench_parser = subparsers.add_parser(
        "summarize-vbench-results", help="summarize VBench JSON outputs to JSON metrics"
    )
    summarize_vbench_parser.add_argument("--results-dir", required=True)
    summarize_vbench_parser.add_argument("--output", required=True)

    native_script_parser = subparsers.add_parser(
        "native-script", help="print a bash script for the native quant/eval matrix"
    )
    native_script_parser.add_argument("--suite", action="append")
    native_script_parser.add_argument("--output-root", default="artifacts/native")
    native_script_parser.add_argument("--report-output", default="reports/native")
    native_script_parser.add_argument("--seeds", type=_parse_seed_list, default=[0])
    native_script_parser.add_argument("--prompt-limit", type=int)
    native_script_parser.add_argument(
        "--prompt-pack",
        default="artifact",
        choices=["artifact", "visual", "geneval-smoke"],
        help="prompt payload source passed to generate-pack",
    )
    native_script_parser.add_argument(
        "--prompt-metadata-jsonl",
        help="GenEval evaluation_metadata.jsonl file passed to generate-pack",
    )
    native_script_parser.add_argument("--device", default="cuda")
    native_script_parser.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"]
    )
    native_script_parser.add_argument(
        "--activation-kernel-backend",
        default="triton_cuda",
        choices=["auto", "cpu", "mps", "triton_cuda"],
    )
    native_script_parser.add_argument(
        "--runtime-mode",
        default="auto_fused",
        choices=_RUNTIME_MODE_CHOICES,
    )
    native_script_parser.add_argument(
        "--staging-mode",
        default="component",
        choices=["streaming", "component"],
    )
    native_script_parser.add_argument("--resume", action="store_true")

    quantize_parser = subparsers.add_parser("quantize", help="quantize a Diffusers component")
    quantize_parser.add_argument("--model-id")
    quantize_parser.add_argument("--suite")
    quantize_parser.add_argument("--revision")
    quantize_parser.add_argument("--output", required=True)
    quantize_parser.add_argument("--component", default="transformer")
    quantize_parser.add_argument("--target-policy", default="auto")
    quantize_parser.add_argument("--weight-bits", type=int, default=4)
    quantize_parser.add_argument("--activation-bits", type=int, default=4)
    quantize_parser.add_argument("--rotation-seed", type=int, default=0)
    quantize_parser.add_argument("--block-size", type=_parse_block_size, default="paper")
    quantize_parser.add_argument(
        "--runtime-mode",
        default="auto_fused",
        choices=_RUNTIME_MODE_CHOICES,
    )
    quantize_parser.add_argument(
        "--activation-kernel-backend",
        default="auto",
        choices=["auto", "cpu", "mps", "triton_cuda"],
    )
    quantize_parser.add_argument("--device", default="auto")
    quantize_parser.add_argument(
        "--staging-mode",
        default="streaming",
        choices=["streaming", "component"],
        help=(
            "streaming moves each target module to the quantization device just before "
            "replacement; component moves the full component first, which is preferred "
            "for large CUDA GPUs when VRAM allows it"
        ),
    )
    quantize_parser.add_argument(
        "--synchronize-per-module",
        action="store_true",
        help=(
            "synchronize the accelerator after each module replacement for debugging "
            "timings; the default only synchronizes at the end to avoid CPU wait "
            "between CUDA kernel launches"
        ),
    )
    quantize_parser.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"]
    )

    validate_parser = subparsers.add_parser(
        "validate-artifact", help="validate an OrbitQuant artifact"
    )
    validate_parser.add_argument("--artifact", required=True)
    validate_parser.add_argument(
        "--policy-inventory",
        help=(
            "optional inspect-policy JSON used to verify manifest module lists "
            "against the captured policy inventory"
        ),
    )
    validate_parser.add_argument("--runtime-mode", choices=_RUNTIME_MODE_CHOICES)

    repair_parser = subparsers.add_parser(
        "repair-artifact-metadata",
        help="refresh metadata-only artifact provenance and checksums",
    )
    repair_parser.add_argument("--artifact", required=True)
    repair_parser.add_argument("--quantization-device", required=True)
    repair_parser.add_argument("--weight-quantization-backend", required=True)
    repair_parser.add_argument(
        "--quantization-staging-mode",
        choices=["streaming", "component", "unknown"],
    )
    repair_parser.add_argument("--skip-tensor-validation", action="store_true")

    upload_parser = subparsers.add_parser(
        "upload-artifact", help="validate and upload an OrbitQuant artifact to HF Hub"
    )
    upload_parser.add_argument("--artifact", required=True)
    upload_parser.add_argument("--repo-id", required=True)
    upload_parser.add_argument("--revision")
    upload_parser.add_argument("--commit-message")
    upload_parser.add_argument("--public", action="store_true")
    upload_parser.add_argument("--no-create-repo", action="store_true")
    upload_parser.set_defaults(replace_repo_files=True)
    upload_parser.add_argument(
        "--replace-repo-files",
        action="store_true",
        default=True,
        help=(
            "replace existing remote repo files before uploading the compact artifact "
            "(default)"
        ),
    )
    upload_parser.add_argument("--skip-tensor-validation", action="store_true")
    upload_parser.add_argument(
        "--upload-profile",
        default="compact",
        choices=["compact"],
        help=(
            "compact stages a validated upload copy with final proof assets and "
            "without raw eval dumps or report logs"
        ),
    )
    upload_parser.add_argument(
        "--report-dir",
        action="append",
        help=(
            "native report directory whose final comparison matrices are promoted "
            "into assets/ when --upload-profile compact is used"
        ),
    )
    upload_parser.add_argument(
        "--staging-dir",
        help="optional directory for the compact staged upload copy; must be empty",
    )
    upload_parser.add_argument("--dry-run", action="store_true")

    audit_hf_parser = subparsers.add_parser(
        "audit-hf-artifacts", help="audit private/public HF OrbitQuant artifact repos"
    )
    audit_hf_parser.add_argument("--namespace", default="WaveCut")
    audit_hf_parser.add_argument("--suite", action="append")
    audit_hf_parser.add_argument("--revision")
    audit_hf_parser.add_argument(
        "--policy-inventory-root",
        help=(
            "optional directory containing <suite>-policy.json files; when set, "
            "remote manifests are checked against those inventories without "
            "downloading model.safetensors"
        ),
    )
    audit_hf_parser.add_argument("--output")
    audit_hf_parser.add_argument("--markdown-output")
    audit_hf_parser.add_argument(
        "--summary-only",
        action="store_true",
        help="print/write compact counts plus per-repo readiness instead of full row details",
    )
    audit_hf_parser.add_argument(
        "--fail-on-artifact-regression",
        action="store_true",
        help=(
            "return a non-zero exit code when compact artifact readiness, native "
            "smoke proof, metadata, policy inventory, checksums, or remote file "
            "hygiene regress; release metrics are intentionally ignored"
        ),
    )

    fetch_hf_parser = subparsers.add_parser(
        "fetch-hf-artifacts",
        help="download published HF OrbitQuant artifacts into the native artifact layout",
    )
    fetch_hf_parser.add_argument("--namespace", default="WaveCut")
    fetch_hf_parser.add_argument("--suite", action="append")
    fetch_hf_parser.add_argument("--output-root", default="artifacts/native")
    fetch_hf_parser.add_argument("--revision")
    fetch_hf_parser.add_argument("--no-resume", action="store_true")
    fetch_hf_parser.add_argument("--force-download", action="store_true")
    fetch_hf_parser.add_argument("--local-files-only", action="store_true")
    fetch_hf_parser.add_argument("--validate-checksums", action="store_true")
    fetch_hf_parser.add_argument("--validate-tensors", action="store_true")
    fetch_hf_parser.add_argument("--dry-run", action="store_true")
    fetch_hf_parser.add_argument("--no-stage-log", action="store_true")

    repair_hf_parser = subparsers.add_parser(
        "repair-hf-artifact-metadata",
        help="repair remote HF artifact metadata without reuploading large tensors",
    )
    repair_hf_parser.add_argument("--repo-id")
    repair_hf_parser.add_argument("--namespace", default="WaveCut")
    repair_hf_parser.add_argument("--suite", action="append")
    repair_hf_parser.add_argument("--revision")
    repair_hf_parser.add_argument("--commit-message")
    repair_hf_parser.add_argument("--quantization-device", required=True)
    repair_hf_parser.add_argument("--weight-quantization-backend", required=True)
    repair_hf_parser.add_argument(
        "--quantization-staging-mode",
        choices=["streaming", "component", "unknown"],
    )
    repair_hf_parser.add_argument("--dry-run", action="store_true")

    repair_native_smoke_parser = subparsers.add_parser(
        "repair-hf-native-smoke-proof",
        help="repair remote HF native smoke proof blocks without re-running generation",
    )
    repair_native_smoke_parser.add_argument("--repo-id")
    repair_native_smoke_parser.add_argument("--namespace", default="WaveCut")
    repair_native_smoke_parser.add_argument("--suite", action="append")
    repair_native_smoke_parser.add_argument("--native-smoke-backup-root")
    repair_native_smoke_parser.add_argument("--revision")
    repair_native_smoke_parser.add_argument("--commit-message")
    repair_native_smoke_parser.add_argument("--dry-run", action="store_true")

    cleanup_hf_parser = subparsers.add_parser(
        "cleanup-hf-artifact-reports",
        help=(
            "promote final comparison matrices and remove non-card assets/reports "
            "from remote HF OrbitQuant artifact repos"
        ),
    )
    cleanup_hf_parser.add_argument("--repo-id")
    cleanup_hf_parser.add_argument("--namespace", default="WaveCut")
    cleanup_hf_parser.add_argument("--suite", action="append")
    cleanup_hf_parser.add_argument("--revision")
    cleanup_hf_parser.add_argument("--commit-message")
    cleanup_hf_parser.add_argument("--dry-run", action="store_true")

    validate_generation_parser = subparsers.add_parser(
        "validate-generation", help="validate a native generation output and metadata pair"
    )
    validate_generation_parser.add_argument("--suite", required=True)
    validate_generation_parser.add_argument("--output", required=True)
    validate_generation_parser.add_argument("--metadata")
    validate_generation_parser.add_argument("--seed", type=int, required=True)
    validate_generation_parser.add_argument("--bit-setting", required=True)
    validate_generation_parser.add_argument("--prompt")
    validate_generation_parser.add_argument("--model-id")

    validate_comparison_parser = subparsers.add_parser(
        "validate-comparison",
        help="validate a local compare-native evidence bundle",
    )
    validate_comparison_parser.add_argument("--input", required=True)
    validate_comparison_parser.add_argument(
        "--blank-stddev-threshold",
        type=float,
        default=1.0,
        help="minimum max channel stddev before an image is treated as blank-like",
    )

    report_parser = subparsers.add_parser("report", help="write a native eval report")
    report_parser.add_argument("--artifact", action="append", required=True)
    report_parser.add_argument("--output", required=True)
    report_parser.add_argument("--date")
    report_parser.add_argument("--fail-on-missing-required", action="store_true")

    record_metrics_parser = subparsers.add_parser(
        "record-metrics", help="import external eval metrics into an artifact"
    )
    record_metrics_parser.add_argument("--artifact", required=True)
    record_metrics_parser.add_argument(
        "--split", required=True, choices=["original", "orbitquant"]
    )
    record_metrics_parser.add_argument("--metrics-json", required=True)
    record_metrics_parser.add_argument("--metric-prefix")
    record_metrics_parser.add_argument("--suite", required=True)
    record_metrics_parser.add_argument("--seed", type=int, required=True)
    record_metrics_parser.add_argument("--bit-setting", required=True)

    generate_parser = subparsers.add_parser("generate", help="run native generation suite")
    generate_parser.add_argument("--suite", required=True)
    generate_parser.add_argument("--prompt")
    generate_parser.add_argument("--prompt-id")
    generate_parser.add_argument("--prompt-index", type=int)
    generate_parser.add_argument("--output")
    generate_parser.add_argument("--artifact")
    generate_parser.add_argument("--component", default="transformer")
    generate_parser.add_argument(
        "--split", default="orbitquant", choices=["original", "orbitquant"]
    )
    generate_parser.add_argument("--seed", type=int, default=0)
    generate_parser.add_argument("--device", default="cuda")
    generate_parser.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"]
    )
    generate_parser.add_argument("--bit-setting", help="native OrbitQuant bit setting, e.g. W4A4")
    generate_parser.add_argument("--rotation-seed", type=int, default=0)
    generate_parser.add_argument(
        "--runtime-mode",
        default="auto_fused",
        choices=_RUNTIME_MODE_CHOICES,
    )
    generate_parser.add_argument(
        "--activation-kernel-backend",
        default="auto",
        choices=["auto", "cpu", "mps", "triton_cuda"],
    )
    generate_parser.add_argument(
        "--no-prewarm",
        action="store_true",
        help="do not prewarm quantized weight caches before generation",
    )
    generate_parser.add_argument(
        "--enable-model-cpu-offload",
        action="store_true",
        help=(
            "use Diffusers model CPU offload instead of moving the whole pipeline to "
            "the generation device"
        ),
    )
    generate_parser.add_argument("--dry-run", action="store_true")

    compare_native_parser = subparsers.add_parser(
        "compare-native",
        help="run original and OrbitQuant native generation into a local comparison bundle",
    )
    compare_native_parser.add_argument("--suite", required=True)
    compare_native_parser.add_argument("--artifact", required=True)
    compare_native_parser.add_argument(
        "--source-model",
        help="source model id or local directory used for loading the BF16 pipeline",
    )
    compare_native_parser.add_argument("--prompt")
    compare_native_parser.add_argument("--prompt-id")
    compare_native_parser.add_argument("--prompt-index", type=int)
    compare_native_parser.add_argument("--output", required=True)
    compare_native_parser.add_argument("--component", default="transformer")
    compare_native_parser.add_argument("--seed", type=int, default=0)
    compare_native_parser.add_argument("--device", default="cuda")
    compare_native_parser.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"]
    )
    compare_native_parser.add_argument(
        "--runtime-mode",
        default="auto_fused",
        choices=_RUNTIME_MODE_CHOICES,
        help="runtime override used for the OrbitQuant split",
    )
    compare_native_parser.add_argument(
        "--activation-kernel-backend",
        default="auto",
        choices=["auto", "cpu", "mps", "triton_cuda"],
    )
    compare_native_parser.add_argument(
        "--no-prewarm",
        action="store_true",
        help="do not prewarm quantized weight caches before the OrbitQuant split",
    )
    compare_native_parser.add_argument(
        "--enable-model-cpu-offload",
        action="store_true",
        help=(
            "use Diffusers model CPU offload instead of moving the whole pipeline to "
            "the generation device"
        ),
    )
    compare_native_parser.add_argument(
        "--skip-artifact-checksums",
        action="store_true",
        help="skip SHA256 validation while loading the local OrbitQuant artifact",
    )
    compare_native_parser.add_argument("--dry-run", action="store_true")

    generate_pack_parser = subparsers.add_parser(
        "generate-pack", help="run native generation for artifact prompt pack"
    )
    generate_pack_parser.add_argument("--suite", required=True)
    generate_pack_parser.add_argument("--artifact", required=True)
    generate_pack_parser.add_argument("--output")
    generate_pack_parser.add_argument("--component", default="transformer")
    generate_pack_parser.add_argument(
        "--split", default="orbitquant", choices=["original", "orbitquant"]
    )
    generate_pack_parser.add_argument("--prompt-id", action="append")
    generate_pack_parser.add_argument("--prompt-limit", type=int)
    generate_pack_parser.add_argument(
        "--prompt-pack",
        default="artifact",
        choices=["artifact", "visual", "geneval-smoke"],
        help="prompt payload source for generate-pack",
    )
    generate_pack_parser.add_argument(
        "--prompt-metadata-jsonl",
        help="GenEval evaluation_metadata.jsonl file to use instead of artifact prompts",
    )
    generate_pack_parser.add_argument(
        "--comparison-mode",
        default="auto",
        choices=["auto", "always", "never"],
        help=(
            "comparison sheet behavior for generate-pack; auto creates sheets once "
            "for artifact/visual packs and skips GenEval metadata packs"
        ),
    )
    generate_pack_parser.add_argument("--seeds", type=_parse_seed_list, default=[0])
    generate_pack_parser.add_argument("--device", default="cuda")
    generate_pack_parser.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"]
    )
    generate_pack_parser.add_argument("--resume-existing", action="store_true")
    generate_pack_parser.add_argument(
        "--no-prewarm",
        action="store_true",
        help="do not prewarm quantized weight caches before generation",
    )
    generate_pack_parser.add_argument(
        "--enable-model-cpu-offload",
        action="store_true",
        help=(
            "use Diffusers model CPU offload instead of moving the whole pipeline to "
            "the generation device"
        ),
    )
    generate_pack_parser.add_argument(
        "--skip-artifact-checksums",
        action="store_true",
        help=(
            "skip SHA256 validation during generation; run validate-artifact separately "
            "when auditing local or uploaded artifacts"
        ),
    )
    generate_pack_parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)
    if args.version:
        print(__version__)
        return 0
    if args.command == "inspect":
        print(json.dumps(inspect_model_metadata(args.model_id, revision=args.revision), indent=2))
        return 0
    if args.command == "inspect-policy":
        suite = None if args.suite is None else get_native_suite(args.suite)
        model_id = args.model_id
        if model_id is None:
            if suite is None:
                raise ValueError("inspect-policy requires --model-id unless --suite is provided")
            model_id = suite.model_id
        target_policy = args.target_policy
        if target_policy == "auto" and suite is not None:
            target_policy = target_policy_for_suite(suite)
        config = OrbitQuantConfig(target_policy=target_policy)
        if args.load_mode == "config":
            if suite is None:
                raise ValueError("inspect-policy --load-mode config requires --suite")
            component = load_component_skeleton_for_suite(
                suite,
                component=args.component,
                model_id=model_id,
                revision=args.revision,
                local_files_only=args.local_files_only,
            )
            pipeline_class = None
            component_class = type(component).__name__
        else:
            load_kwargs = {"torch_dtype": _torch_dtype(args.dtype)}
            if args.revision is not None:
                load_kwargs["revision"] = args.revision
            if args.local_files_only:
                load_kwargs["local_files_only"] = True
            if suite is None:
                from diffusers import DiffusionPipeline

                pipeline = DiffusionPipeline.from_pretrained(model_id, **load_kwargs)
            else:
                pipeline = load_pipeline_for_suite(suite, model_id=model_id, **load_kwargs)
            component = _pipeline_component(pipeline, args.component)
            pipeline_class = type(pipeline).__name__
            component_class = type(component).__name__
        payload = {
            "source_model_id": model_id,
            "source_revision": args.revision,
            "suite": None if suite is None else suite.name,
            "component": args.component,
            "load_mode": args.load_mode,
            "pipeline_class": pipeline_class,
            "component_class": component_class,
            **inspect_linear_module_policy(component, config),
        }
        if args.output is not None:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            payload = _policy_inventory_summary(payload, output_path)
        print(json.dumps(payload, indent=2))
        return 0
    if args.command == "native-suites":
        payload = [suite.__dict__ for suite in list_native_suites()]
        print(json.dumps(payload, indent=2))
        return 0
    if args.command == "kernel-info":
        print(json.dumps(backend_capabilities(), indent=2))
        return 0
    if args.command == "kernel-bench":
        print(
            json.dumps(
                benchmark_orbit_linear(
                    tokens=args.tokens,
                    in_features=args.in_features,
                    out_features=args.out_features,
                    weight_bits=args.weight_bits,
                    activation_bits=args.activation_bits,
                    block_size=args.block_size,
                    activation_kernel_backend=args.activation_kernel_backend,
                    runtime_mode=args.runtime_mode,
                    packed_matmul_block_m=args.packed_matmul_block_m,
                    packed_matmul_block_n=args.packed_matmul_block_n,
                    packed_matmul_block_k=args.packed_matmul_block_k,
                    packed_matmul_num_warps=args.packed_matmul_num_warps,
                    device=args.device,
                    dtype=_torch_dtype(args.dtype),
                    warmup=args.warmup,
                    iterations=args.iterations,
                    seed=args.seed,
                ),
                indent=2,
            )
        )
        return 0
    if args.command == "quantize-bench":
        print(
            json.dumps(
                benchmark_model_quantization(
                    layers=args.layers,
                    in_features=args.in_features,
                    hidden_features=args.hidden_features,
                    weight_bits=args.weight_bits,
                    activation_bits=args.activation_bits,
                    block_size=args.block_size,
                    source_device=args.source_device,
                    quantization_device=args.quantization_device,
                    staging_mode=args.staging_mode,
                    synchronize_per_module=args.synchronize_per_module,
                    dtype=_torch_dtype(args.dtype),
                    seed=args.seed,
                ),
                indent=2,
            )
        )
        return 0
    if args.command == "native-plan":
        suites = None
        if args.suite is not None:
            suites = [get_native_suite(name) for name in args.suite]
        print(
            json.dumps(
                build_native_eval_plan(
                    suites=suites,
                    output_root=args.output_root,
                    seeds=args.seeds,
                    runtime_mode=args.runtime_mode,
                ),
                indent=2,
            )
        )
        return 0
    if args.command == "external-eval-plan":
        suites = None
        if args.suite is not None:
            suites = [get_native_suite(name) for name in args.suite]
        print(
            json.dumps(
                build_external_eval_plan(
                    suites=suites,
                    output_root=args.output_root,
                    metrics_root=args.metrics_root,
                ),
                indent=2,
            )
        )
        return 0
    if args.command == "external-eval-script":
        suites = None
        if args.suite is not None:
            suites = [get_native_suite(name) for name in args.suite]
        print(
            build_external_eval_script(
                suites=suites,
                output_root=args.output_root,
                metrics_root=args.metrics_root,
                report_output_dir=args.report_output,
            )
        )
        return 0
    if args.command == "export-geneval":
        result = export_geneval_artifact(args.artifact, args.output, split=args.split)
        print(json.dumps(result.__dict__, indent=2))
        return 0
    if args.command == "export-vbench":
        result = export_vbench_artifact(
            args.artifact,
            args.output,
            split=args.split,
            link_mode=args.link_mode,
        )
        print(json.dumps(result.__dict__, indent=2))
        return 0
    if args.command == "summarize-geneval-results":
        print(
            json.dumps(
                summarize_geneval_results(args.results_jsonl, args.output),
                indent=2,
            )
        )
        return 0
    if args.command == "summarize-vbench-results":
        print(
            json.dumps(
                summarize_vbench_results(args.results_dir, args.output),
                indent=2,
            )
        )
        return 0
    if args.command == "native-script":
        suites = None
        if args.suite is not None:
            suites = [get_native_suite(name) for name in args.suite]
        print(
            build_native_run_script(
                suites=suites,
                output_root=args.output_root,
                report_output_dir=args.report_output,
                seeds=args.seeds,
                prompt_limit=args.prompt_limit,
                prompt_pack=args.prompt_pack,
                prompt_metadata_jsonl=args.prompt_metadata_jsonl,
                device=args.device,
                dtype=args.dtype,
                activation_kernel_backend=args.activation_kernel_backend,
                runtime_mode=args.runtime_mode,
                staging_mode=args.staging_mode,
                resume=args.resume,
            )
        )
        return 0
    if args.command == "quantize":
        command_started_at = time.perf_counter()
        device = _resolve_device(args.device)
        suite = None if args.suite is None else get_native_suite(args.suite)
        model_id = args.model_id if args.model_id is not None else None
        if model_id is None:
            if suite is None:
                raise ValueError("quantize requires --model-id unless --suite is provided")
            model_id = suite.model_id
        target_policy = args.target_policy
        if target_policy == "auto" and suite is not None:
            target_policy = target_policy_for_suite(suite)
        config = OrbitQuantConfig(
            weight_bits=args.weight_bits,
            activation_bits=args.activation_bits,
            target_policy=target_policy,
            rotation_seed=args.rotation_seed,
            block_size=args.block_size,
            runtime_mode=args.runtime_mode,
            activation_kernel_backend=args.activation_kernel_backend,
        )
        load_kwargs = {"torch_dtype": _torch_dtype(args.dtype)}
        if args.revision is not None:
            load_kwargs["revision"] = args.revision
        load_started_at = time.perf_counter()
        if suite is None:
            from diffusers import DiffusionPipeline

            pipeline = DiffusionPipeline.from_pretrained(model_id, **load_kwargs)
        else:
            pipeline = load_pipeline_for_suite(suite, model_id=model_id, **load_kwargs)
        load_elapsed_seconds = time.perf_counter() - load_started_at
        try:
            component = getattr(pipeline, args.component)
        except AttributeError as exc:
            raise ValueError(f"pipeline has no component {args.component!r}") from exc
        quantize_started_at = time.perf_counter()
        summary = quantize_linear_modules(
            component,
            config,
            quantization_device=device,
            staging_mode=args.staging_mode,
            synchronize_per_module=args.synchronize_per_module,
        )
        quantize_elapsed_seconds = time.perf_counter() - quantize_started_at
        metadata_started_at = time.perf_counter()
        metadata = inspect_model_metadata(model_id, revision=args.revision)
        metadata_elapsed_seconds = time.perf_counter() - metadata_started_at
        save_started_at = time.perf_counter()
        manifest = save_orbitquant_artifact(
            component,
            args.output,
            config=config,
            source_model_id=model_id,
            source_revision=metadata.get("sha") or args.revision or "unknown",
            source_license=metadata.get("license") or "unknown",
            summary=summary,
            component=args.component,
        )
        save_elapsed_seconds = time.perf_counter() - save_started_at
        print(
            json.dumps(
                {
                    "artifact_dir": args.output,
                    "component": args.component,
                    "source_model_id": model_id,
                    "source_revision": manifest.source_revision,
                    "source_license": manifest.source_license,
                    "quantization_device": summary.quantization_device,
                    "weight_quantization_backend": summary.weight_quantization_backend,
                    "quantization_staging_mode": summary.quantization_staging_mode,
                    "synchronize_per_module": summary.synchronize_per_module,
                    "load_elapsed_seconds": load_elapsed_seconds,
                    "quantization_elapsed_seconds": summary.elapsed_seconds,
                    "quantization_command_elapsed_seconds": quantize_elapsed_seconds,
                    "orbitquant_seconds": summary.orbitquant_seconds,
                    "adaln_seconds": summary.adaln_seconds,
                    "device_transfer_seconds": summary.device_transfer_seconds,
                    "module_device_transfer_count": summary.module_device_transfer_count,
                    "source_linear_device_counts": summary.source_linear_device_counts,
                    "quantized_buffer_device_counts": summary.quantized_buffer_device_counts,
                    "metadata_elapsed_seconds": metadata_elapsed_seconds,
                    "artifact_save_elapsed_seconds": save_elapsed_seconds,
                    "total_elapsed_seconds": time.perf_counter() - command_started_at,
                    "quantized_modules": summary.quantized_modules,
                    "adaln_modules": summary.adaln_modules,
                    "skipped_modules": summary.skipped_modules,
                }
            )
        )
        return 0
    if args.command == "validate-artifact":
        payload = validate_orbitquant_artifact(args.artifact)
        if args.runtime_mode is not None and payload["runtime_mode"] != args.runtime_mode:
            print(
                (
                    "artifact runtime_mode mismatch: "
                    f"expected {args.runtime_mode}, got {payload['runtime_mode']}"
                ),
                file=sys.stderr,
            )
            return 1
        if args.policy_inventory is not None:
            payload["policy_inventory_validation"] = validate_artifact_policy_inventory(
                args.artifact,
                args.policy_inventory,
            )
        print(json.dumps(payload))
        return 0
    if args.command == "repair-artifact-metadata":
        print(
            json.dumps(
                repair_artifact_metadata(
                    args.artifact,
                    quantization_device=args.quantization_device,
                    weight_quantization_backend=args.weight_quantization_backend,
                    quantization_staging_mode=args.quantization_staging_mode,
                    validate_tensors=not args.skip_tensor_validation,
                ),
                indent=2,
            )
        )
        return 0
    if args.command == "upload-artifact":
        print(
            json.dumps(
                upload_orbitquant_artifact(
                    args.artifact,
                    repo_id=args.repo_id,
                    private=not args.public,
                    create_repo=not args.no_create_repo,
                    revision=args.revision,
                    commit_message=args.commit_message,
                    replace_repo_files=args.replace_repo_files,
                    validate_tensors=not args.skip_tensor_validation,
                    upload_profile=args.upload_profile,
                    report_dirs=args.report_dir,
                    staging_dir=args.staging_dir,
                    dry_run=args.dry_run,
                ),
                indent=2,
            )
        )
        return 0
    if args.command == "audit-hf-artifacts":
        suites = None
        if args.suite is not None:
            suites = [get_native_suite(name) for name in args.suite]
        payload = audit_hf_artifact_repos(
            namespace=args.namespace,
            suites=suites,
            revision=args.revision,
            policy_inventory_root=args.policy_inventory_root,
        )
        rendered_payload = _hf_artifact_audit_summary(payload) if args.summary_only else payload
        rendered = json.dumps(rendered_payload, indent=2)
        if args.output is not None:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(rendered + "\n", encoding="utf-8")
        if args.markdown_output is not None:
            markdown_path = Path(args.markdown_output)
            markdown_path.parent.mkdir(parents=True, exist_ok=True)
            markdown_path.write_text(
                render_hf_artifact_audit_markdown(payload),
                encoding="utf-8",
            )
        print(rendered)
        if args.fail_on_artifact_regression:
            regressions = _hf_artifact_audit_regressions(payload)
            if regressions:
                print(
                    "HF artifact audit regressions: " + "; ".join(regressions),
                    file=sys.stderr,
                )
                return 1
        return 0
    if args.command == "fetch-hf-artifacts":
        suites = None
        if args.suite is not None:
            suites = [get_native_suite(name) for name in args.suite]

        def stage_logger(event: str, label: str) -> None:
            print(f"==== {event} {datetime.now().isoformat()} {label} ====", file=sys.stderr)

        print(
            json.dumps(
                fetch_hf_artifacts(
                    namespace=args.namespace,
                    suites=suites,
                    output_root=args.output_root,
                    revision=args.revision,
                    resume=not args.no_resume,
                    force_download=args.force_download,
                    local_files_only=args.local_files_only,
                    validate_checksums=args.validate_checksums,
                    validate_tensors=args.validate_tensors,
                    dry_run=args.dry_run,
                    stage_logger=None if args.no_stage_log or args.dry_run else stage_logger,
                ),
                indent=2,
            )
        )
        return 0
    if args.command == "repair-hf-artifact-metadata":
        suites = None
        if args.suite is not None:
            suites = [get_native_suite(name) for name in args.suite]
        if args.repo_id is not None and suites is not None:
            raise ValueError("--repo-id cannot be combined with --suite")
        if args.repo_id is not None:
            payload = repair_hf_artifact_metadata(
                repo_id=args.repo_id,
                quantization_device=args.quantization_device,
                weight_quantization_backend=args.weight_quantization_backend,
                quantization_staging_mode=args.quantization_staging_mode,
                revision=args.revision,
                commit_message=args.commit_message,
                dry_run=args.dry_run,
            )
        else:
            payload = repair_hf_artifact_metadata_matrix(
                namespace=args.namespace,
                suites=suites,
                quantization_device=args.quantization_device,
                weight_quantization_backend=args.weight_quantization_backend,
                quantization_staging_mode=args.quantization_staging_mode,
                revision=args.revision,
                commit_message=args.commit_message,
                dry_run=args.dry_run,
            )
        print(json.dumps(payload, indent=2))
        return 0
    if args.command == "repair-hf-native-smoke-proof":
        suites = None
        if args.suite is not None:
            suites = [get_native_suite(name) for name in args.suite]
        if args.repo_id is not None:
            if suites is None or len(suites) != 1:
                raise ValueError("--repo-id requires exactly one --suite")
            payload = repair_hf_native_smoke_proof(
                repo_id=args.repo_id,
                suite=suites[0],
                native_smoke_backup_root=args.native_smoke_backup_root,
                revision=args.revision,
                commit_message=args.commit_message,
                dry_run=args.dry_run,
            )
        else:
            payload = repair_hf_native_smoke_proof_matrix(
                namespace=args.namespace,
                suites=suites,
                native_smoke_backup_root=args.native_smoke_backup_root,
                revision=args.revision,
                commit_message=args.commit_message,
                dry_run=args.dry_run,
            )
        print(json.dumps(payload, indent=2))
        return 0
    if args.command == "cleanup-hf-artifact-reports":
        suites = None
        if args.suite is not None:
            suites = [get_native_suite(name) for name in args.suite]
        if args.repo_id is not None and suites is not None:
            raise ValueError("--repo-id cannot be combined with --suite")
        if args.repo_id is not None:
            payload = cleanup_hf_artifact_reports(
                repo_id=args.repo_id,
                revision=args.revision,
                commit_message=args.commit_message,
                dry_run=args.dry_run,
            )
        else:
            payload = cleanup_hf_artifact_reports_matrix(
                namespace=args.namespace,
                suites=suites,
                revision=args.revision,
                commit_message=args.commit_message,
                dry_run=args.dry_run,
            )
        print(json.dumps(payload, indent=2))
        return 0
    if args.command == "validate-generation":
        suite = get_native_suite(args.suite)
        output_path = Path(args.output)
        metadata_path = (
            _metadata_path_for_output(output_path)
            if args.metadata is None
            else Path(args.metadata)
        )
        print(
            json.dumps(
                validate_native_generation_output(
                    output_path,
                    metadata_path,
                    suite,
                    seed=args.seed,
                    bit_setting=args.bit_setting,
                    prompt=args.prompt,
                    model_id=args.model_id,
                )
            )
        )
        return 0
    if args.command == "validate-comparison":
        print(
            json.dumps(
                _validate_compare_native_bundle(
                    args.input,
                    blank_stddev_threshold=args.blank_stddev_threshold,
                ),
                indent=2,
            )
        )
        return 0
    if args.command == "report":
        result = generate_native_eval_report(
            args.artifact,
            args.output,
            report_date=args.date,
        )
        print(
            json.dumps(
                {
                    "report_path": str(result.report_path),
                    "tables": {key: str(value) for key, value in result.table_paths.items()},
                    "artifact_count": len(args.artifact),
                    "row_count": len(result.rows),
                    "missing_required_metric_count": len(result.missing_required_metrics),
                }
            )
        )
        if args.fail_on_missing_required and result.missing_required_metrics:
            return 1
        return 0
    if args.command == "record-metrics":
        metrics = load_metric_json(args.metrics_json, metric_prefix=args.metric_prefix)
        record = record_artifact_metrics(
            args.artifact,
            split=args.split,
            metrics=metrics,
            metadata={
                "suite": args.suite,
                "seed": args.seed,
                "bit_setting": args.bit_setting,
                "metrics_source": args.metrics_json,
            },
        )
        print(
            json.dumps(
                {
                    "artifact": args.artifact,
                    "split": args.split,
                    "metrics": record["metrics"],
                    "metadata": record["metadata"],
                }
            )
        )
        return 0
    if args.command == "compare-native":
        suite = get_native_suite(args.suite)
        artifact_path = Path(args.artifact)
        artifact_validation = validate_orbitquant_artifact(
            artifact_path,
            validate_checksums_enabled=not args.skip_artifact_checksums,
            validate_tensors=False,
        )
        artifact_config = OrbitQuantConfig.from_dict(
            json.loads((artifact_path / "quantization_config.json").read_text(encoding="utf-8"))
        )
        effective_config = _with_runtime_overrides(
            artifact_config,
            runtime_mode=args.runtime_mode,
            activation_kernel_backend=args.activation_kernel_backend,
        )
        bit_setting = (
            f"W{artifact_validation['weight_bits']}A"
            f"{artifact_validation['activation_bits']}"
        )
        model_id = artifact_validation["source_model_id"]
        source_model = model_id if args.source_model is None else args.source_model
        output_dir = Path(args.output)

        prompt = args.prompt
        prompt_record = None
        if args.prompt_id is not None or args.prompt_index is not None:
            if prompt is not None:
                raise ValueError(
                    "compare-native accepts either --prompt or a prompt selector, not both"
                )
            prompt_payload = json.loads(
                (artifact_path / "prompts.json").read_text(encoding="utf-8")
            )
            prompt_record = select_prompt_record(
                prompt_payload,
                prompt_id=args.prompt_id,
                prompt_index=args.prompt_index,
            )
            prompt = prompt_record["prompt"]
        if prompt is None:
            raise ValueError("compare-native requires --prompt or a prompt selector")

        pipeline_kwargs = build_pipeline_kwargs(
            suite, prompt=prompt, seed=args.seed, device=args.device
        )
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "suite": suite.__dict__,
                        "model_id": model_id,
                        "source_model": source_model,
                        "artifact": str(artifact_path),
                        "component": args.component,
                        "output": str(output_dir),
                        "prompt_record": prompt_record,
                        "seed": args.seed,
                        "device": args.device,
                        "dtype": args.dtype,
                        "enable_model_cpu_offload": args.enable_model_cpu_offload,
                        "bit_setting": bit_setting,
                        "runtime_mode": effective_config.runtime_mode,
                        "activation_kernel_backend": effective_config.activation_kernel_backend,
                        "pipeline_kwargs": {
                            key: value
                            for key, value in pipeline_kwargs.items()
                            if key != "generator"
                        },
                    },
                    indent=2,
                )
            )
            return 0

        output_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[orbitquant] compare-native original start suite={suite.name} seed={args.seed}",
            file=sys.stderr,
        )
        original_pipeline = load_pipeline_for_suite(
            suite,
            model_id=source_model,
            torch_dtype=_torch_dtype(args.dtype),
        )
        _place_pipeline_for_generation(
            original_pipeline,
            device=args.device,
            enable_model_cpu_offload=args.enable_model_cpu_offload,
        )
        original_result = run_native_generation(
            original_pipeline,
            suite,
            prompt=prompt,
            seed=args.seed,
            output_dir=output_dir,
            device=args.device,
            quantization_label="original",
            runtime_dtype=args.dtype,
            model_id=model_id,
        )
        validate_native_generation_output(
            original_result.output_path,
            original_result.metadata_path,
            suite,
            seed=args.seed,
            bit_setting="original",
            prompt=prompt,
            model_id=model_id,
        )
        _release_generation_pipeline(original_pipeline, device=args.device)
        original_pipeline = None
        print(
            f"[orbitquant] compare-native original done output={original_result.output_path}",
            file=sys.stderr,
        )

        print(
            f"[orbitquant] compare-native orbitquant start suite={suite.name} "
            f"seed={args.seed} bit_setting={bit_setting} runtime={effective_config.runtime_mode}",
            file=sys.stderr,
        )
        quantized_pipeline = load_pipeline_for_suite(
            suite,
            model_id=source_model,
            torch_dtype=_torch_dtype(args.dtype),
        )
        load_quantized_pipeline_component(
            quantized_pipeline,
            artifact_path,
            component=args.component,
            validate_checksums=not args.skip_artifact_checksums,
            runtime_mode=effective_config.runtime_mode,
            activation_kernel_backend=effective_config.activation_kernel_backend,
        )
        _place_pipeline_for_generation(
            quantized_pipeline,
            device=args.device,
            enable_model_cpu_offload=args.enable_model_cpu_offload,
        )
        prewarm_metadata = None
        if not args.no_prewarm and _should_prewarm_quantized_weights(effective_config):
            prewarm_metadata = _prewarm_pipeline_component(
                quantized_pipeline,
                args.component,
                device=args.device,
                dtype=_torch_dtype(args.dtype),
            )
        orbitquant_result = run_native_generation(
            quantized_pipeline,
            suite,
            prompt=prompt,
            seed=args.seed,
            output_dir=output_dir,
            device=args.device,
            quantization_config=effective_config,
            quantization_summary=None,
            quantization_label=bit_setting,
            prewarm_metadata=prewarm_metadata,
            runtime_dtype=args.dtype,
            model_id=model_id,
        )
        validate_native_generation_output(
            orbitquant_result.output_path,
            orbitquant_result.metadata_path,
            suite,
            seed=args.seed,
            bit_setting=bit_setting,
            prompt=prompt,
            model_id=model_id,
        )
        orbitquant_runtime = _orbitquant_runtime_summary(quantized_pipeline, args.component)
        _release_generation_pipeline(quantized_pipeline, device=args.device)
        quantized_pipeline = None
        print(
            f"[orbitquant] compare-native orbitquant done output={orbitquant_result.output_path}",
            file=sys.stderr,
        )

        comparison_path = (
            output_dir
            / f"{suite.name}_seed{args.seed}_{bit_setting}_original_vs_orbitquant.webp"
        )
        from orbitquant.eval.assets import create_image_comparison_sheet

        create_image_comparison_sheet(
            _comparison_image_source(original_result, is_video=suite.frames is not None),
            _comparison_image_source(orbitquant_result, is_video=suite.frames is not None),
            comparison_path,
            labels=("BF16", f"OrbitQuant {bit_setting}"),
        )
        summary_path = output_dir / "summary.json"
        summary = {
            "suite": suite.name,
            "model_id": model_id,
            "source_model": source_model,
            "artifact": str(artifact_path),
            "component": args.component,
            "prompt": prompt,
            "prompt_record": prompt_record,
            "seed": args.seed,
            "height": suite.height,
            "width": suite.width,
            "frames": suite.frames,
            "steps": suite.steps,
            "guidance": suite.guidance,
            "dtype": args.dtype,
            "device": args.device,
            "bit_setting": bit_setting,
            "runtime_mode": effective_config.runtime_mode,
            "activation_kernel_backend": effective_config.activation_kernel_backend,
            "enable_model_cpu_offload": args.enable_model_cpu_offload,
            "available_backends": available_backends(),
            "original": {
                "output_path": str(original_result.output_path),
                "metadata_path": str(original_result.metadata_path),
                "wall_time_seconds": original_result.metadata["wall_time_seconds"],
                "peak_vram_bytes": original_result.metadata["peak_vram_bytes"],
            },
            "orbitquant": {
                "output_path": str(orbitquant_result.output_path),
                "metadata_path": str(orbitquant_result.metadata_path),
                "wall_time_seconds": orbitquant_result.metadata["wall_time_seconds"],
                "peak_vram_bytes": orbitquant_result.metadata["peak_vram_bytes"],
                "prewarm": prewarm_metadata,
                "runtime": orbitquant_runtime,
            },
            "comparison_path": str(comparison_path),
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "summary_path": str(summary_path),
                    "comparison_path": str(comparison_path),
                    "original_output_path": str(original_result.output_path),
                    "orbitquant_output_path": str(orbitquant_result.output_path),
                    "runtime_mode": effective_config.runtime_mode,
                    "activation_kernel_backend": effective_config.activation_kernel_backend,
                }
            )
        )
        return 0
    if args.command == "generate-pack":
        suite = get_native_suite(args.suite)
        artifact_path = Path(args.artifact)
        artifact_validation = validate_orbitquant_artifact(
            artifact_path,
            validate_checksums_enabled=False,
            validate_tensors=False,
        )
        artifact_config = OrbitQuantConfig.from_dict(
            json.loads((artifact_path / "quantization_config.json").read_text(encoding="utf-8"))
        )
        artifact_bit_setting = (
            f"W{artifact_validation['weight_bits']}A"
            f"{artifact_validation['activation_bits']}"
        )
        bit_setting = artifact_bit_setting if args.split == "orbitquant" else "original"
        quantization_config = artifact_config if args.split == "orbitquant" else None
        output_dir = artifact_path / "assets" if args.output is None else Path(args.output)
        create_pack_comparisons = args.comparison_mode == "always" or (
            args.comparison_mode == "auto" and args.prompt_metadata_jsonl is None
        )
        prompt_payload = _load_generate_pack_prompt_payload(
            artifact_path=artifact_path,
            suite=suite,
            prompt_pack=args.prompt_pack,
            prompt_metadata_jsonl=args.prompt_metadata_jsonl,
        )
        jobs = build_prompt_seed_jobs(
            prompt_payload,
            seeds=args.seeds,
            prompt_ids=args.prompt_id,
            prompt_limit=args.prompt_limit,
        )
        comparison_keys = {
            (suite.name, int(job["seed"]), str(job["prompt_record"]["id"])) for job in jobs
        }
        pending_jobs = []
        skipped_outputs = []
        for job in jobs:
            prompt_record = job["prompt_record"]
            seed = int(job["seed"])
            variant = f"{bit_setting}_{prompt_record['id']}"
            expected_output_path = _expected_generation_output_path(
                output_dir, suite=suite, seed=seed, variant=variant
            )
            expected_metadata_path = _metadata_path_for_output(expected_output_path)
            if (
                args.resume_existing
                and expected_output_path.is_file()
                and expected_metadata_path.is_file()
            ):
                try:
                    validate_native_generation_output(
                        expected_output_path,
                        expected_metadata_path,
                        suite,
                        seed=seed,
                        bit_setting=bit_setting,
                        prompt=prompt_record["prompt"],
                        model_id=artifact_validation["source_model_id"],
                    )
                except RuntimeError:
                    pending_jobs.append(job)
                else:
                    skipped_outputs.append(str(expected_output_path))
            else:
                pending_jobs.append(job)
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "suite": suite.__dict__,
                        "model_id": artifact_validation["source_model_id"],
                        "artifact": str(artifact_path),
                        "component": args.component,
                        "output": str(output_dir),
                        "bit_setting": bit_setting,
                        "split": args.split,
                        "prompt_pack": prompt_payload.get("prompt_pack"),
                        "enable_model_cpu_offload": args.enable_model_cpu_offload,
                        "job_count": len(jobs),
                        "run_count": len(pending_jobs),
                        "skipped_count": len(skipped_outputs),
                        "skipped_outputs": skipped_outputs,
                        "jobs": jobs,
                    },
                    indent=2,
                )
            )
            return 0

        if not pending_jobs:
            artifact_comparisons: list[str] = []
            if create_pack_comparisons:
                artifact_comparisons = create_artifact_image_comparisons(
                    artifact_path,
                    comparison_keys=comparison_keys,
                    validate_checksums_enabled=not args.skip_artifact_checksums,
                    refresh_checksums_enabled=not args.skip_artifact_checksums,
                )
            checksum_refresh = None
            if args.skip_artifact_checksums:
                checksum_refresh = refresh_artifact_checksums(artifact_path)
            print(
                json.dumps(
                    {
                        "artifact": str(artifact_path),
                        "prompt_pack": prompt_payload.get("prompt_pack"),
                        "job_count": len(jobs),
                        "run_count": 0,
                        "skipped_count": len(skipped_outputs),
                        "skipped_outputs": skipped_outputs,
                        "artifact_comparisons": artifact_comparisons,
                        "checksum_refresh": checksum_refresh,
                        "outputs": [],
                    }
                )
            )
            return 0

        pipeline = load_pipeline_for_suite(
            suite,
            model_id=artifact_validation["source_model_id"],
            torch_dtype=_torch_dtype(args.dtype),
        )
        prewarm_metadata = None
        if args.split == "orbitquant":
            load_quantized_pipeline_component(
                pipeline,
                artifact_path,
                component=args.component,
                validate_checksums=not args.skip_artifact_checksums,
            )
        _place_pipeline_for_generation(
            pipeline,
            device=args.device,
            enable_model_cpu_offload=args.enable_model_cpu_offload,
        )
        if (
            args.split == "orbitquant"
            and not args.no_prewarm
            and _should_prewarm_quantized_weights(artifact_config)
        ):
            prewarm_metadata = _prewarm_pipeline_component(
                pipeline,
                args.component,
                device=args.device,
                dtype=_torch_dtype(args.dtype),
            )
        outputs = []
        for job in pending_jobs:
            prompt_record = job["prompt_record"]
            seed = int(job["seed"])
            variant = f"{bit_setting}_{prompt_record['id']}"
            result = run_native_generation(
                pipeline,
                suite,
                prompt=prompt_record["prompt"],
                seed=seed,
                output_dir=output_dir,
                device=args.device,
                quantization_config=quantization_config,
                quantization_summary=None,
                quantization_label=variant,
                prewarm_metadata=prewarm_metadata,
                runtime_dtype=args.dtype,
                model_id=artifact_validation["source_model_id"],
            )
            validate_native_generation_output(
                result.output_path,
                result.metadata_path,
                suite,
                seed=seed,
                bit_setting=bit_setting,
                prompt=prompt_record["prompt"],
                model_id=artifact_validation["source_model_id"],
            )
            metrics, comparisons = _record_generated_artifact(
                artifact_path,
                result,
                split=args.split,
                suite=suite,
                prompt=prompt_record["prompt"],
                prompt_record=prompt_record,
                seed=seed,
                bit_setting=bit_setting,
                validate_checksums_enabled=not args.skip_artifact_checksums,
                refresh_checksums_enabled=not args.skip_artifact_checksums,
                create_comparisons_enabled=False,
            )
            outputs.append(
                {
                    "output_path": str(result.output_path),
                    "metadata_path": str(result.metadata_path),
                    "metrics": metrics,
                    "comparisons": comparisons,
                    "prompt_record": prompt_record,
                    "seed": seed,
                }
            )
        artifact_comparisons: list[str] = []
        if create_pack_comparisons:
            artifact_comparisons = create_artifact_image_comparisons(
                artifact_path,
                comparison_keys=comparison_keys,
                validate_checksums_enabled=not args.skip_artifact_checksums,
                refresh_checksums_enabled=not args.skip_artifact_checksums,
            )
        checksum_refresh = None
        if args.skip_artifact_checksums:
            checksum_refresh = refresh_artifact_checksums(artifact_path)
        print(
            json.dumps(
                {
                    "artifact": str(artifact_path),
                    "prompt_pack": prompt_payload.get("prompt_pack"),
                    "job_count": len(jobs),
                    "run_count": len(outputs),
                    "skipped_count": len(skipped_outputs),
                    "skipped_outputs": skipped_outputs,
                    "artifact_comparisons": artifact_comparisons,
                    "checksum_refresh": checksum_refresh,
                    "outputs": outputs,
                }
            )
        )
        return 0
    if args.command == "generate":
        suite = get_native_suite(args.suite)
        artifact_path = None if args.artifact is None else Path(args.artifact)
        artifact_validation = None
        artifact_config = None
        if artifact_path is not None:
            artifact_validation = validate_orbitquant_artifact(artifact_path)
            artifact_config = OrbitQuantConfig.from_dict(
                json.loads(
                    (artifact_path / "quantization_config.json").read_text(
                        encoding="utf-8"
                    )
                )
            )
        if args.output is None:
            if artifact_path is None:
                raise ValueError("generate requires --output when --artifact is not provided")
            output_dir = artifact_path / "assets"
        else:
            output_dir = Path(args.output)

        quantization_config = None
        bit_setting = None
        if artifact_validation is not None:
            expected_bit_setting = (
                f"W{artifact_validation['weight_bits']}A"
                f"{artifact_validation['activation_bits']}"
            )
            if args.split == "original":
                if args.bit_setting is not None:
                    raise ValueError("original split does not accept --bit-setting")
                bit_setting = "original"
                quantization_config = None
            else:
                bit_setting = (
                    args.bit_setting.upper()
                    if args.bit_setting is not None
                    else expected_bit_setting
                )
                quantization_config = artifact_config
            if args.split == "orbitquant" and bit_setting != expected_bit_setting:
                raise ValueError(
                    f"artifact bit setting is {expected_bit_setting}, got {bit_setting}"
                )
            model_id = artifact_validation["source_model_id"]
        else:
            model_id = suite.model_id
        prompt = args.prompt
        prompt_record = None
        if args.prompt_id is not None or args.prompt_index is not None:
            if prompt is not None:
                raise ValueError("generate accepts either --prompt or a prompt selector, not both")
            if artifact_path is None:
                raise ValueError("prompt selection requires --artifact")
            prompt_payload = json.loads(
                (artifact_path / "prompts.json").read_text(encoding="utf-8")
            )
            prompt_record = select_prompt_record(
                prompt_payload,
                prompt_id=args.prompt_id,
                prompt_index=args.prompt_index,
            )
            prompt = prompt_record["prompt"]
        if prompt is None:
            raise ValueError(
                "generate requires --prompt unless a prompt selector is used with --artifact"
            )
        if artifact_validation is None and args.bit_setting is not None:
            bit_setting = args.bit_setting.upper()
            quantization_config = build_quantization_config_for_suite(
                suite,
                bit_setting,
                rotation_seed=args.rotation_seed,
                runtime_mode=args.runtime_mode,
                activation_kernel_backend=args.activation_kernel_backend,
            )
        kwargs = build_pipeline_kwargs(
            suite, prompt=prompt, seed=args.seed, device=args.device
        )
        if args.dry_run:
            payload = {
                "suite": suite.__dict__,
                "model_id": model_id,
                "artifact": None if artifact_path is None else str(artifact_path),
                "component": args.component,
                "output": str(output_dir),
                "prompt_record": prompt_record,
                "device": args.device,
                "dtype": args.dtype,
                "enable_model_cpu_offload": args.enable_model_cpu_offload,
                "quantization_config": None
                if quantization_config is None
                else quantization_config.to_dict(),
                "pipeline_kwargs": {
                    key: value
                    for key, value in kwargs.items()
                    if key != "generator"
                },
            }
            print(json.dumps(payload, indent=2))
            return 0

        pipeline = load_pipeline_for_suite(
            suite,
            model_id=model_id,
            torch_dtype=_torch_dtype(args.dtype),
        )
        quantization_summary = None
        prewarm_metadata = None
        if artifact_path is not None:
            if args.split == "orbitquant":
                load_quantized_pipeline_component(
                    pipeline,
                    artifact_path,
                    component=args.component,
                )
        elif quantization_config is not None:
            quantization_summary = apply_quantization_to_pipeline(
                pipeline, suite, quantization_config
            )
        _place_pipeline_for_generation(
            pipeline,
            device=args.device,
            enable_model_cpu_offload=args.enable_model_cpu_offload,
        )
        should_prewarm = (
            quantization_config is not None
            and not args.no_prewarm
            and _should_prewarm_quantized_weights(quantization_config)
            and (artifact_path is None or args.split == "orbitquant")
        )
        if should_prewarm:
            prewarm_metadata = _prewarm_pipeline_component(
                pipeline,
                args.component,
                device=args.device,
                dtype=_torch_dtype(args.dtype),
            )
        result = run_native_generation(
            pipeline,
            suite,
            prompt=prompt,
            seed=args.seed,
            output_dir=output_dir,
            device=args.device,
            quantization_config=quantization_config,
            quantization_summary=quantization_summary,
            quantization_label=bit_setting,
            prewarm_metadata=prewarm_metadata,
            runtime_dtype=args.dtype,
            model_id=model_id,
        )
        validate_native_generation_output(
            result.output_path,
            result.metadata_path,
            suite,
            seed=args.seed,
            bit_setting="original" if bit_setting is None else bit_setting,
            prompt=prompt,
            model_id=model_id,
        )
        artifact_metrics = None
        if artifact_path is not None:
            artifact_metrics, artifact_comparisons = _record_generated_artifact(
                artifact_path,
                result,
                split=args.split,
                suite=suite,
                prompt=prompt,
                prompt_record=prompt_record,
                seed=args.seed,
                bit_setting=bit_setting,
            )
        print(
            json.dumps(
                {
                    "output_path": str(result.output_path),
                    "metadata_path": str(result.metadata_path),
                    "artifact": None if artifact_path is None else str(artifact_path),
                    "artifact_metrics": artifact_metrics,
                    "artifact_comparisons": []
                    if artifact_path is None
                    else artifact_comparisons,
                }
            )
        )
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
