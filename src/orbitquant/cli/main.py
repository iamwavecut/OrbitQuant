from __future__ import annotations

import argparse
import json

import torch

from orbitquant import __version__
from orbitquant.artifacts import save_orbitquant_artifact, validate_orbitquant_artifact
from orbitquant.config import OrbitQuantConfig
from orbitquant.eval import list_native_suites
from orbitquant.eval.native_runner import (
    apply_quantization_to_pipeline,
    build_pipeline_kwargs,
    build_quantization_config_for_suite,
    run_native_generation,
)
from orbitquant.eval.native_settings import get_native_suite
from orbitquant.hub import inspect_model_metadata
from orbitquant.modeling import quantize_linear_modules


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="orbitquant")
    parser.add_argument("--version", action="store_true", help="print OrbitQuant version")
    subparsers = parser.add_subparsers(dest="command")

    inspect_parser = subparsers.add_parser("inspect", help="inspect Hugging Face model metadata")
    inspect_parser.add_argument("--model-id", required=True)
    inspect_parser.add_argument("--revision")
    subparsers.add_parser("native-suites", help="list native eval suites")

    quantize_parser = subparsers.add_parser("quantize", help="quantize a Diffusers component")
    quantize_parser.add_argument("--model-id", required=True)
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

    generate_parser = subparsers.add_parser("generate", help="run native generation suite")
    generate_parser.add_argument("--suite", required=True)
    generate_parser.add_argument("--prompt", required=True)
    generate_parser.add_argument("--output", required=True)
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
    if args.command == "quantize":
        from diffusers import DiffusionPipeline

        device = _resolve_device(args.device)
        config = OrbitQuantConfig(
            weight_bits=args.weight_bits,
            activation_bits=args.activation_bits,
            target_policy=args.target_policy,
            rotation_seed=args.rotation_seed,
            block_size=args.block_size,
            runtime_mode=args.runtime_mode,
            activation_kernel_backend=args.activation_kernel_backend,
        )
        load_kwargs = {"torch_dtype": _torch_dtype(args.dtype)}
        if args.revision is not None:
            load_kwargs["revision"] = args.revision
        pipeline = DiffusionPipeline.from_pretrained(args.model_id, **load_kwargs)
        pipeline.to(device)
        try:
            component = getattr(pipeline, args.component)
        except AttributeError as exc:
            raise ValueError(f"pipeline has no component {args.component!r}") from exc
        summary = quantize_linear_modules(component, config)
        metadata = inspect_model_metadata(args.model_id, revision=args.revision)
        manifest = save_orbitquant_artifact(
            component,
            args.output,
            config=config,
            source_model_id=args.model_id,
            source_revision=metadata.get("sha") or args.revision or "unknown",
            source_license=metadata.get("license") or "unknown",
            summary=summary,
        )
        print(
            json.dumps(
                {
                    "artifact_dir": args.output,
                    "component": args.component,
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
    if args.command == "generate":
        suite = get_native_suite(args.suite)
        quantization_config = None
        bit_setting = None
        if args.bit_setting is not None:
            bit_setting = args.bit_setting.upper()
            quantization_config = build_quantization_config_for_suite(
                suite,
                bit_setting,
                rotation_seed=args.rotation_seed,
                runtime_mode=args.runtime_mode,
                activation_kernel_backend=args.activation_kernel_backend,
            )
        kwargs = build_pipeline_kwargs(
            suite, prompt=args.prompt, seed=args.seed, device=args.device
        )
        if args.dry_run:
            payload = {
                "suite": suite.__dict__,
                "output": args.output,
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

        from diffusers import DiffusionPipeline

        pipeline = DiffusionPipeline.from_pretrained(
            suite.model_id,
            torch_dtype=_torch_dtype(args.dtype),
        )
        pipeline.to(args.device)
        quantization_summary = None
        if quantization_config is not None:
            quantization_summary = apply_quantization_to_pipeline(
                pipeline, suite, quantization_config
            )
        result = run_native_generation(
            pipeline,
            suite,
            prompt=args.prompt,
            seed=args.seed,
            output_dir=args.output,
            device=args.device,
            quantization_config=quantization_config,
            quantization_summary=quantization_summary,
            quantization_label=bit_setting,
        )
        print(
            json.dumps(
                {"output_path": str(result.output_path), "metadata_path": str(result.metadata_path)}
            )
        )
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
