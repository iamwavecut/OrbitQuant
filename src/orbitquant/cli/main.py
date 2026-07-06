from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from orbitquant import __version__
from orbitquant.artifacts import (
    create_artifact_image_comparisons,
    record_artifact_asset,
    record_artifact_metrics,
    save_orbitquant_artifact,
    validate_orbitquant_artifact,
)
from orbitquant.config import OrbitQuantConfig
from orbitquant.eval import list_native_suites
from orbitquant.eval.metrics import load_metric_json
from orbitquant.eval.native_plan import build_native_eval_plan, build_native_run_script
from orbitquant.eval.native_runner import (
    apply_quantization_to_pipeline,
    build_pipeline_kwargs,
    build_quantization_config_for_suite,
    load_pipeline_for_suite,
    run_native_generation,
    target_policy_for_suite,
)
from orbitquant.eval.native_settings import get_native_suite
from orbitquant.eval.prompts import build_prompt_seed_jobs, select_prompt_record
from orbitquant.eval.report import generate_native_eval_report
from orbitquant.hub import inspect_model_metadata
from orbitquant.modeling import quantize_linear_modules
from orbitquant.pipeline import load_quantized_pipeline_component


def _torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


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
) -> tuple[dict[str, Any], list[str]]:
    if _path_is_relative_to(result.output_path, artifact_path):
        record_artifact_asset(artifact_path, result.output_path)
    if _path_is_relative_to(result.metadata_path, artifact_path):
        record_artifact_asset(artifact_path, result.metadata_path)
    for asset_path in result.asset_paths:
        if _path_is_relative_to(asset_path, artifact_path):
            record_artifact_asset(artifact_path, asset_path)
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
            "steps": suite.steps,
            "guidance": suite.guidance,
            "bit_setting": bit_setting,
            "output_path": str(result.output_path),
            "metadata_path": str(result.metadata_path),
            "asset_paths": [str(asset_path) for asset_path in result.asset_paths],
            "device": result.metadata["device"],
            "dtype": result.metadata["dtype"],
            "pipeline_class": result.metadata["pipeline_class"],
            "scheduler": result.metadata["scheduler"],
        },
    )
    return metrics_record, create_artifact_image_comparisons(artifact_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="orbitquant")
    parser.add_argument("--version", action="store_true", help="print OrbitQuant version")
    subparsers = parser.add_subparsers(dest="command")

    inspect_parser = subparsers.add_parser("inspect", help="inspect Hugging Face model metadata")
    inspect_parser.add_argument("--model-id", required=True)
    inspect_parser.add_argument("--revision")
    subparsers.add_parser("native-suites", help="list native eval suites")

    native_plan_parser = subparsers.add_parser(
        "native-plan", help="print native quant/eval job matrix"
    )
    native_plan_parser.add_argument("--suite", action="append")
    native_plan_parser.add_argument("--output-root", default="artifacts/native")
    native_plan_parser.add_argument("--seeds", type=_parse_seed_list, default=[0])

    native_script_parser = subparsers.add_parser(
        "native-script", help="print a bash script for the native quant/eval matrix"
    )
    native_script_parser.add_argument("--suite", action="append")
    native_script_parser.add_argument("--output-root", default="artifacts/native")
    native_script_parser.add_argument("--seeds", type=_parse_seed_list, default=[0])
    native_script_parser.add_argument("--prompt-limit", type=int)
    native_script_parser.add_argument("--device", default="cuda")
    native_script_parser.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"]
    )
    native_script_parser.add_argument(
        "--activation-kernel-backend",
        default="auto",
        choices=["auto", "cpu", "mps", "triton_cuda"],
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
        default="dequant_bf16",
        choices=["dequant_bf16", "debug_no_quant", "debug_no_activation_quant"],
    )
    quantize_parser.add_argument(
        "--activation-kernel-backend",
        default="auto",
        choices=["auto", "cpu", "mps", "triton_cuda"],
    )
    quantize_parser.add_argument("--device", default="auto")
    quantize_parser.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"]
    )

    validate_parser = subparsers.add_parser(
        "validate-artifact", help="validate an OrbitQuant artifact"
    )
    validate_parser.add_argument("--artifact", required=True)

    report_parser = subparsers.add_parser("report", help="write a native eval report")
    report_parser.add_argument("--artifact", action="append", required=True)
    report_parser.add_argument("--output", required=True)
    report_parser.add_argument("--date")

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
        default="dequant_bf16",
        choices=["dequant_bf16", "debug_no_quant", "debug_no_activation_quant"],
    )
    generate_parser.add_argument(
        "--activation-kernel-backend",
        default="auto",
        choices=["auto", "cpu", "mps", "triton_cuda"],
    )
    generate_parser.add_argument("--dry-run", action="store_true")

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
    generate_pack_parser.add_argument("--seeds", type=_parse_seed_list, default=[0])
    generate_pack_parser.add_argument("--device", default="cuda")
    generate_pack_parser.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"]
    )
    generate_pack_parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)
    if args.version:
        print(__version__)
        return 0
    if args.command == "inspect":
        print(json.dumps(inspect_model_metadata(args.model_id, revision=args.revision), indent=2))
        return 0
    if args.command == "native-suites":
        payload = [suite.__dict__ for suite in list_native_suites()]
        print(json.dumps(payload, indent=2))
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
                ),
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
                seeds=args.seeds,
                prompt_limit=args.prompt_limit,
                device=args.device,
                dtype=args.dtype,
                activation_kernel_backend=args.activation_kernel_backend,
                resume=args.resume,
            )
        )
        return 0
    if args.command == "quantize":
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
        if suite is None:
            from diffusers import DiffusionPipeline

            pipeline = DiffusionPipeline.from_pretrained(model_id, **load_kwargs)
        else:
            pipeline = load_pipeline_for_suite(suite, model_id=model_id, **load_kwargs)
        pipeline.to(device)
        try:
            component = getattr(pipeline, args.component)
        except AttributeError as exc:
            raise ValueError(f"pipeline has no component {args.component!r}") from exc
        summary = quantize_linear_modules(component, config)
        metadata = inspect_model_metadata(model_id, revision=args.revision)
        manifest = save_orbitquant_artifact(
            component,
            args.output,
            config=config,
            source_model_id=model_id,
            source_revision=metadata.get("sha") or args.revision or "unknown",
            source_license=metadata.get("license") or "unknown",
            summary=summary,
        )
        print(
            json.dumps(
                {
                    "artifact_dir": args.output,
                    "component": args.component,
                    "source_model_id": model_id,
                    "source_revision": manifest.source_revision,
                    "source_license": manifest.source_license,
                    "quantized_modules": summary.quantized_modules,
                    "adaln_modules": summary.adaln_modules,
                    "skipped_modules": summary.skipped_modules,
                }
            )
        )
        return 0
    if args.command == "validate-artifact":
        print(json.dumps(validate_orbitquant_artifact(args.artifact)))
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
                }
            )
        )
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
    if args.command == "generate-pack":
        suite = get_native_suite(args.suite)
        artifact_path = Path(args.artifact)
        artifact_validation = validate_orbitquant_artifact(artifact_path)
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
        prompt_payload = json.loads((artifact_path / "prompts.json").read_text(encoding="utf-8"))
        jobs = build_prompt_seed_jobs(
            prompt_payload,
            seeds=args.seeds,
            prompt_ids=args.prompt_id,
            prompt_limit=args.prompt_limit,
        )
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
                        "job_count": len(jobs),
                        "jobs": jobs,
                    },
                    indent=2,
                )
            )
            return 0

        pipeline = load_pipeline_for_suite(
            suite,
            model_id=artifact_validation["source_model_id"],
            torch_dtype=_torch_dtype(args.dtype),
        )
        pipeline.to(args.device)
        if args.split == "orbitquant":
            load_quantized_pipeline_component(
                pipeline,
                artifact_path,
                component=args.component,
            )
        outputs = []
        for job in jobs:
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
                runtime_dtype=args.dtype,
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
        print(
            json.dumps(
                {
                    "artifact": str(artifact_path),
                    "job_count": len(jobs),
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
        pipeline.to(args.device)
        quantization_summary = None
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
            runtime_dtype=args.dtype,
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
