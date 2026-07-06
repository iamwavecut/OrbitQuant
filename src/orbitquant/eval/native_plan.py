from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

from orbitquant.eval.native_runner import target_policy_for_suite
from orbitquant.eval.native_settings import NativeSuite, list_native_suites


def _artifact_name(suite_name: str, bit_setting: str) -> str:
    return f"{suite_name}-{bit_setting.lower()}"


def build_native_eval_plan(
    *,
    suites: list[NativeSuite] | None = None,
    output_root: str | Path = "artifacts/native",
    seeds: list[int] | None = None,
) -> dict[str, Any]:
    selected_suites = list_native_suites() if suites is None else suites
    selected_seeds = [0] if seeds is None else seeds
    root = Path(output_root)
    jobs = []
    for suite in selected_suites:
        for bit_setting in suite.bit_settings:
            artifact_dir = root / _artifact_name(suite.name, bit_setting)
            for seed in selected_seeds:
                jobs.append(
                    {
                        "suite": suite.name,
                        "model_id": suite.model_id,
                        "pipeline": suite.pipeline,
                        "target_policy": target_policy_for_suite(suite),
                        "bit_setting": bit_setting,
                        "artifact_dir": str(artifact_dir),
                        "seed": seed,
                        "width": suite.width,
                        "height": suite.height,
                        "frames": suite.frames,
                        "steps": suite.steps,
                        "guidance": suite.guidance,
                        "metric": suite.metric,
                    }
                )
    return {"job_count": len(jobs), "jobs": jobs}


def _bits(bit_setting: str) -> tuple[str, str]:
    weight, activation = bit_setting.upper().split("A", maxsplit=1)
    return weight.removeprefix("W"), activation


def _cmd(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def build_native_run_script(
    *,
    suites: list[NativeSuite] | None = None,
    output_root: str | Path = "artifacts/native",
    seeds: list[int] | None = None,
    prompt_limit: int | None = None,
    device: str = "cuda",
    dtype: str = "bfloat16",
    activation_kernel_backend: str = "auto",
) -> str:
    selected_suites = list_native_suites() if suites is None else suites
    selected_seeds = [0] if seeds is None else seeds
    seed_arg = ",".join(str(seed) for seed in selected_seeds)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
    ]
    for suite in selected_suites:
        for bit_setting in suite.bit_settings:
            weight_bits, activation_bits = _bits(bit_setting)
            artifact_dir = str(Path(output_root) / _artifact_name(suite.name, bit_setting))
            lines.append(f"# {suite.name} {bit_setting}")
            lines.append(
                _cmd(
                    [
                        "orbitquant",
                        "quantize",
                        "--suite",
                        suite.name,
                        "--target-policy",
                        target_policy_for_suite(suite),
                        "--weight-bits",
                        weight_bits,
                        "--activation-bits",
                        activation_bits,
                        "--activation-kernel-backend",
                        activation_kernel_backend,
                        "--device",
                        device,
                        "--dtype",
                        dtype,
                        "--output",
                        artifact_dir,
                    ]
                )
            )
            lines.append(
                _cmd(["orbitquant", "validate-artifact", "--artifact", artifact_dir])
            )
            for split in ("original", "orbitquant"):
                command = [
                    "orbitquant",
                    "generate-pack",
                    "--suite",
                    suite.name,
                    "--artifact",
                    artifact_dir,
                    "--split",
                    split,
                    "--seeds",
                    seed_arg,
                    "--device",
                    device,
                    "--dtype",
                    dtype,
                ]
                if prompt_limit is not None:
                    command.extend(["--prompt-limit", str(prompt_limit)])
                lines.append(_cmd(command))
            lines.append(
                _cmd(["orbitquant", "validate-artifact", "--artifact", artifact_dir])
            )
            lines.append("")
    return "\n".join(lines)
