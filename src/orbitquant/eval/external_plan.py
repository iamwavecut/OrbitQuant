from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

from orbitquant.eval.native_settings import NativeSuite, list_native_suites


def _artifact_name(suite_name: str, bit_setting: str) -> str:
    return f"{suite_name}-{bit_setting.lower()}"


def _cmd(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def _metrics_json_path(metrics_root: Path, artifact_name: str, split: str, metric: str) -> Path:
    return metrics_root / f"{artifact_name}_{split}_{metric}.json"


def _eval_command(
    suite: NativeSuite,
    *,
    artifact_dir: Path,
    split: str,
    metrics_json: Path,
) -> str:
    if suite.metric == "geneval":
        return _cmd(
            [
                "geneval",
                "--metadata-dir",
                artifact_dir / "benchmark",
                "--split",
                split,
                "--output-json",
                metrics_json,
            ]
        )
    if suite.metric == "vbench":
        return _cmd(
            [
                "vbench",
                "--input-dir",
                artifact_dir / "assets",
                "--output-json",
                metrics_json,
            ]
        )
    raise ValueError(f"native suite {suite.name!r} has no external metric runner")


def _import_command(
    suite: NativeSuite,
    *,
    artifact_dir: Path,
    split: str,
    bit_setting: str,
    metrics_json: Path,
) -> str:
    return _cmd(
        [
            "orbitquant",
            "record-metrics",
            "--artifact",
            artifact_dir,
            "--split",
            split,
            "--metrics-json",
            metrics_json,
            "--metric-prefix",
            str(suite.metric),
            "--suite",
            suite.name,
            "--seed",
            "0",
            "--bit-setting",
            bit_setting,
        ]
    )


def build_external_eval_plan(
    suites: list[NativeSuite] | None = None,
    *,
    output_root: str | Path = "artifacts/native",
    metrics_root: str | Path = "metrics/native",
) -> dict[str, Any]:
    selected_suites = list_native_suites() if suites is None else suites
    output_path = Path(output_root)
    metrics_path = Path(metrics_root)
    jobs = []
    for suite in selected_suites:
        if suite.metric not in {"geneval", "vbench"}:
            continue
        for bit_setting in suite.bit_settings:
            artifact_name = _artifact_name(suite.name, bit_setting)
            artifact_dir = output_path / artifact_name
            for split in ("original", "orbitquant"):
                metrics_json = _metrics_json_path(
                    metrics_path, artifact_name, split, str(suite.metric)
                )
                jobs.append(
                    {
                        "suite": suite.name,
                        "model_id": suite.model_id,
                        "bit_setting": bit_setting,
                        "split": split,
                        "metric": suite.metric,
                        "artifact_dir": str(artifact_dir),
                        "metrics_json": str(metrics_json),
                        "eval_command": _eval_command(
                            suite,
                            artifact_dir=artifact_dir,
                            split=split,
                            metrics_json=metrics_json,
                        ),
                        "import_command": _import_command(
                            suite,
                            artifact_dir=artifact_dir,
                            split=split,
                            bit_setting=bit_setting,
                            metrics_json=metrics_json,
                        ),
                    }
                )
    return {"job_count": len(jobs), "jobs": jobs}


def build_external_eval_script(
    suites: list[NativeSuite] | None = None,
    *,
    output_root: str | Path = "artifacts/native",
    metrics_root: str | Path = "metrics/native",
    report_output_dir: str | Path = "reports/native",
) -> str:
    plan = build_external_eval_plan(
        suites=suites,
        output_root=output_root,
        metrics_root=metrics_root,
    )
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        _cmd(["mkdir", "-p", Path(metrics_root)]),
        "",
    ]
    artifact_dirs: list[str] = []
    for job in plan["jobs"]:
        artifact_dir = str(job["artifact_dir"])
        if artifact_dir not in artifact_dirs:
            artifact_dirs.append(artifact_dir)
        lines.extend(
            [
                f"# {job['suite']} {job['bit_setting']} {job['split']} {job['metric']}",
                _cmd(["mkdir", "-p", Path(str(job["metrics_json"])).parent]),
                str(job["eval_command"]),
                str(job["import_command"]),
                "",
            ]
        )
    if artifact_dirs:
        report_command = ["orbitquant", "report"]
        for artifact_dir in artifact_dirs:
            report_command.extend(["--artifact", artifact_dir])
        report_command.extend(["--output", str(report_output_dir)])
        lines.extend(["# Native report", _cmd(report_command), ""])
    return "\n".join(lines)
