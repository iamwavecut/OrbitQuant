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
    runtime_mode: str = "dequant_bf16",
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
                        "runtime_mode": runtime_mode,
                    }
                )
    return {"job_count": len(jobs), "jobs": jobs}


def _bits(bit_setting: str) -> tuple[str, str]:
    weight, activation = bit_setting.upper().split("A", maxsplit=1)
    return weight.removeprefix("W"), activation


def _cmd(parts: list[str | Path]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def _stage_lines(label: str, command: str) -> list[str]:
    return [
        _cmd(["stage_log", "START", label]),
        command,
        _cmd(["stage_log", "END", label]),
    ]


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


def _kernel_preflight_lines(
    suites: list[NativeSuite],
    *,
    device: str,
    dtype: str,
    activation_kernel_backend: str,
    runtime_mode: str,
) -> list[str]:
    lines = [
        "# Kernel preflight",
        "orbitquant kernel-info",
    ]
    unique_bit_settings = sorted(
        {bit_setting for suite in suites for bit_setting in suite.bit_settings}
    )
    for bit_setting in unique_bit_settings:
        weight_bits, activation_bits = _bits(bit_setting)
        lines.append(
            _cmd(
                [
                    "orbitquant",
                    "kernel-bench",
                    "--tokens",
                    "256",
                    "--in-features",
                    "3072",
                    "--out-features",
                    "3072",
                    "--weight-bits",
                    weight_bits,
                    "--activation-bits",
                    activation_bits,
                    "--activation-kernel-backend",
                    activation_kernel_backend,
                    "--runtime-mode",
                    runtime_mode,
                    "--device",
                    device,
                    "--dtype",
                    dtype,
                    "--warmup",
                    "1",
                    "--iterations",
                    "3",
                ]
            )
        )
    lines.append("")
    return lines


def _policy_inventory_lines(
    suites: list[NativeSuite],
    *,
    report_output_dir: str | Path,
    dtype: str,
) -> tuple[list[str], dict[str, str]]:
    inventory_dir = Path(report_output_dir) / "module-inventories"
    inventory_paths = {
        suite.name: str(inventory_dir / f"{suite.name}-policy.json")
        for suite in suites
    }
    lines = [
        "# Policy inventories",
        _cmd(["mkdir", "-p", str(inventory_dir)]),
    ]
    for suite in suites:
        lines.append(
            _cmd(
                [
                    "orbitquant",
                    "inspect-policy",
                    "--suite",
                    suite.name,
                    "--dtype",
                    dtype,
                    "--output",
                    inventory_paths[suite.name],
                ]
            )
        )
    lines.append("")
    return lines, inventory_paths


def build_native_run_script(
    *,
    suites: list[NativeSuite] | None = None,
    output_root: str | Path = "artifacts/native",
    report_output_dir: str | Path = "reports/native",
    seeds: list[int] | None = None,
    prompt_limit: int | None = None,
    prompt_pack: str = "artifact",
    prompt_metadata_jsonl: str | Path | None = None,
    device: str = "cuda",
    dtype: str = "bfloat16",
    activation_kernel_backend: str = "triton_cuda",
    runtime_mode: str = "dequant_bf16",
    staging_mode: str = "component",
    resume: bool = False,
) -> str:
    if prompt_metadata_jsonl is not None and prompt_pack != "artifact":
        raise ValueError("prompt_metadata_jsonl cannot be combined with a non-artifact prompt pack")
    selected_suites = list_native_suites() if suites is None else suites
    selected_seeds = [0] if seeds is None else seeds
    seed_arg = ",".join(str(seed) for seed in selected_seeds)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "stage_log() {",
        "  printf '==== %s %s %s ====\\n' \"$1\" \"$(date -Is)\" \"$2\"",
        "}",
        "",
    ]
    lines.append(_cmd(["stage_log", "START", "preflight"]))
    lines.extend(_preflight_lines(selected_suites))
    lines.append(_cmd(["stage_log", "END", "preflight"]))
    lines.append("")
    lines.append(_cmd(["stage_log", "START", "kernel preflight"]))
    lines.extend(
        _kernel_preflight_lines(
            selected_suites,
            device=device,
            dtype=dtype,
            activation_kernel_backend=activation_kernel_backend,
            runtime_mode=runtime_mode,
        )
    )
    lines.append(_cmd(["stage_log", "END", "kernel preflight"]))
    lines.append("")
    policy_lines, policy_inventory_paths = _policy_inventory_lines(
        selected_suites,
        report_output_dir=report_output_dir,
        dtype=dtype,
    )
    lines.append(_cmd(["stage_log", "START", "policy inventories"]))
    lines.extend(policy_lines)
    lines.append(_cmd(["stage_log", "END", "policy inventories"]))
    lines.append("")
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
                    "--runtime-mode",
                    runtime_mode,
                    "--device",
                    device,
                    "--staging-mode",
                    staging_mode,
                    "--dtype",
                    dtype,
                    "--output",
                    artifact_dir,
                ]
            )
            validate_command = _cmd(
                [
                    "orbitquant",
                    "validate-artifact",
                    "--artifact",
                    artifact_dir,
                    "--policy-inventory",
                    policy_inventory_paths[suite.name],
                ]
            )
            if resume:
                lines.extend(
                    [
                        _cmd(["stage_log", "START", f"{suite.name} {bit_setting} quantize"]),
                        f"if {validate_command} >/dev/null 2>&1; then",
                        f"echo 'Skipping existing valid artifact: {artifact_dir}'",
                        "else",
                        quantize_command,
                        "fi",
                        _cmd(["stage_log", "END", f"{suite.name} {bit_setting} quantize"]),
                    ]
                )
            else:
                lines.extend(_stage_lines(f"{suite.name} {bit_setting} quantize", quantize_command))
            lines.extend(
                _stage_lines(
                    f"{suite.name} {bit_setting} validate quantized artifact",
                    validate_command,
                )
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
                if prompt_metadata_jsonl is not None:
                    command.extend(["--prompt-metadata-jsonl", str(prompt_metadata_jsonl)])
                elif prompt_pack != "artifact":
                    command.extend(["--prompt-pack", prompt_pack])
                if prompt_limit is not None:
                    command.extend(["--prompt-limit", str(prompt_limit)])
                if resume:
                    command.append("--resume-existing")
                lines.extend(
                    _stage_lines(
                        f"{suite.name} {bit_setting} {split} generate-pack",
                        _cmd(command),
                    )
                )
            lines.extend(
                _stage_lines(
                    f"{suite.name} {bit_setting} validate generated artifact",
                    _cmd(["orbitquant", "validate-artifact", "--artifact", artifact_dir]),
                )
            )
            lines.append("")
    report_command = ["orbitquant", "report"]
    for artifact_dir in artifact_dirs:
        report_command.extend(["--artifact", artifact_dir])
    report_command.extend(["--output", str(report_output_dir)])
    lines.extend(
        [
            "# Native report",
            *_stage_lines("native eval report", _cmd(report_command)),
            "",
        ]
    )
    return "\n".join(lines)
