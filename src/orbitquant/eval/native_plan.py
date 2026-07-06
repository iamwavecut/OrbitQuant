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


def _preflight_lines(suites: list[NativeSuite]) -> list[str]:
    model_ids = sorted({suite.model_id for suite in suites})
    lines = [
        "# Preflight",
        "hf auth whoami",
        "hf env",
        "python - <<'PY'",
        "import importlib.metadata as metadata",
        "import shutil",
        "import sys",
        "import torch",
        "",
        "def version(package):",
        "    try:",
        "        return metadata.version(package)",
        "    except metadata.PackageNotFoundError:",
        "        return 'not-installed'",
        "",
        "print('python=' + sys.version.replace('\\n', ' '))",
        "print('torch=' + torch.__version__)",
        "print('diffusers=' + version('diffusers'))",
        "print('transformers=' + version('transformers'))",
        "print('accelerate=' + version('accelerate'))",
        "print('cuda_available=' + str(torch.cuda.is_available()))",
        "if not torch.cuda.is_available():",
        "    raise SystemExit('CUDA is required for native GPU evaluation')",
        "print('cuda_device=' + torch.cuda.get_device_name(0))",
        "print('disk_free_bytes=' + str(shutil.disk_usage('.').free))",
        "PY",
        "",
        "# Model access",
    ]
    lines.extend(
        _cmd(["hf", "models", "info", model_id, "--format", "json"]) + " >/dev/null"
        for model_id in model_ids
    )
    lines.append("")
    return lines


def build_native_run_script(
    *,
    suites: list[NativeSuite] | None = None,
    output_root: str | Path = "artifacts/native",
    report_output_dir: str | Path = "reports/native",
    seeds: list[int] | None = None,
    prompt_limit: int | None = None,
    device: str = "cuda",
    dtype: str = "bfloat16",
    activation_kernel_backend: str = "auto",
    resume: bool = False,
) -> str:
    selected_suites = list_native_suites() if suites is None else suites
    selected_seeds = [0] if seeds is None else seeds
    seed_arg = ",".join(str(seed) for seed in selected_seeds)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
    ]
    lines.extend(_preflight_lines(selected_suites))
    artifact_dirs = []
    for suite in selected_suites:
        for bit_setting in suite.bit_settings:
            weight_bits, activation_bits = _bits(bit_setting)
            artifact_dir = str(Path(output_root) / _artifact_name(suite.name, bit_setting))
            artifact_dirs.append(artifact_dir)
            lines.append(f"# {suite.name} {bit_setting}")
            quantize_command = _cmd(
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
            validate_command = _cmd(["orbitquant", "validate-artifact", "--artifact", artifact_dir])
            if resume:
                lines.extend(
                    [
                        f"if {validate_command} >/dev/null 2>&1; then",
                        f"echo 'Skipping existing valid artifact: {artifact_dir}'",
                        "else",
                        quantize_command,
                        "fi",
                    ]
                )
            else:
                lines.append(quantize_command)
            lines.append(
                validate_command
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
                if resume:
                    command.append("--resume-existing")
                lines.append(_cmd(command))
            lines.append(
                _cmd(["orbitquant", "validate-artifact", "--artifact", artifact_dir])
            )
            lines.append("")
    report_command = ["orbitquant", "report"]
    for artifact_dir in artifact_dirs:
        report_command.extend(["--artifact", artifact_dir])
    report_command.extend(["--output", str(report_output_dir)])
    lines.extend(["# Native report", _cmd(report_command), ""])
    return "\n".join(lines)
