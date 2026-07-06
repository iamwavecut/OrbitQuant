from __future__ import annotations

import argparse
import json

import torch

from orbitquant import __version__
from orbitquant.eval import list_native_suites
from orbitquant.eval.native_runner import build_pipeline_kwargs, run_native_generation
from orbitquant.eval.native_settings import get_native_suite
from orbitquant.hub import inspect_model_metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="orbitquant")
    parser.add_argument("--version", action="store_true", help="print OrbitQuant version")
    subparsers = parser.add_subparsers(dest="command")

    inspect_parser = subparsers.add_parser("inspect", help="inspect Hugging Face model metadata")
    inspect_parser.add_argument("--model-id", required=True)
    inspect_parser.add_argument("--revision")
    subparsers.add_parser("native-suites", help="list native eval suites")
    generate_parser = subparsers.add_parser("generate", help="run native generation suite")
    generate_parser.add_argument("--suite", required=True)
    generate_parser.add_argument("--prompt", required=True)
    generate_parser.add_argument("--output", required=True)
    generate_parser.add_argument("--seed", type=int, default=0)
    generate_parser.add_argument("--device", default="cuda")
    generate_parser.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"]
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
    if args.command == "generate":
        suite = get_native_suite(args.suite)
        kwargs = build_pipeline_kwargs(
            suite, prompt=args.prompt, seed=args.seed, device=args.device
        )
        if args.dry_run:
            payload = {
                "suite": suite.__dict__,
                "output": args.output,
                "device": args.device,
                "dtype": args.dtype,
                "pipeline_kwargs": {
                    key: value
                    for key, value in kwargs.items()
                    if key != "generator"
                },
            }
            print(json.dumps(payload, indent=2))
            return 0

        from diffusers import DiffusionPipeline

        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[args.dtype]
        pipeline = DiffusionPipeline.from_pretrained(
            suite.model_id,
            torch_dtype=dtype,
        )
        pipeline.to(args.device)
        result = run_native_generation(
            pipeline,
            suite,
            prompt=args.prompt,
            seed=args.seed,
            output_dir=args.output,
            device=args.device,
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
