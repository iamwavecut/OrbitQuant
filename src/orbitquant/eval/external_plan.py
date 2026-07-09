from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

from orbitquant.eval.native_settings import NativeSuite, list_native_suites

VBENCH_CUSTOM_INPUT_DIMENSIONS = (
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
    "scene",
    "overall_consistency",
)


def _artifact_name(suite_name: str, bit_setting: str) -> str:
    return f"{suite_name}-{bit_setting.lower()}"


def _cmd(parts: list[str | Path]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def _metrics_json_path(metrics_root: Path, artifact_name: str, split: str, metric: str) -> Path:
    return metrics_root / f"{artifact_name}_{split}_{metric}.json"


def _export_dir_path(metrics_root: Path, artifact_name: str, split: str, metric: str) -> Path:
    return metrics_root / "exports" / artifact_name / split / metric


def _geneval_results_jsonl_path(metrics_root: Path, artifact_name: str, split: str) -> Path:
    return metrics_root / f"{artifact_name}_{split}_geneval_results.jsonl"


def _vbench_results_dir_path(metrics_root: Path, artifact_name: str, split: str) -> Path:
    return metrics_root / f"{artifact_name}_{split}_vbench_output"


def _export_command(
    suite: NativeSuite,
    *,
    artifact_dir: Path,
    split: str,
    export_dir: Path,
) -> str:
    if suite.metric == "geneval":
        return _cmd(
            [
                "orbitquant",
                "export-geneval",
                "--artifact",
                artifact_dir,
                "--split",
                split,
                "--output",
                export_dir,
            ]
        )
    if suite.metric == "vbench":
        return _cmd(
            [
                "orbitquant",
                "export-vbench",
                "--artifact",
                artifact_dir,
                "--split",
                split,
                "--output",
                export_dir,
            ]
        )
    raise ValueError(f"native suite {suite.name!r} has no external metric export")


def _eval_command(
    suite: NativeSuite,
    *,
    export_dir: Path,
    metrics_root: Path,
    artifact_name: str,
    split: str,
) -> str:
    if suite.metric == "geneval":
        return " ".join(
            [
                "python",
                '"${GENEVAL_DIR}/evaluation/evaluate_images.py"',
                _cmd([export_dir]),
                "--outfile",
                _cmd([_geneval_results_jsonl_path(metrics_root, artifact_name, split)]),
                "--model-path",
                '"${GENEVAL_OBJECT_DETECTOR}"',
            ]
        )
    if suite.metric == "vbench":
        return _cmd(
            [
                "vbench",
                "evaluate",
                "--dimension",
                *VBENCH_CUSTOM_INPUT_DIMENSIONS,
                "--videos_path",
                export_dir,
                "--mode",
                "custom_input",
                "--prompt_file",
                export_dir / "vbench_prompts.json",
                "--output_path",
                _vbench_results_dir_path(metrics_root, artifact_name, split),
            ]
        )
    raise ValueError(f"native suite {suite.name!r} has no external metric runner")


def _summarize_command(
    suite: NativeSuite,
    *,
    metrics_root: Path,
    artifact_name: str,
    split: str,
    metrics_json: Path,
) -> str:
    if suite.metric == "geneval":
        return _cmd(
            [
                "orbitquant",
                "summarize-geneval-results",
                "--results-jsonl",
                _geneval_results_jsonl_path(metrics_root, artifact_name, split),
                "--output",
                metrics_json,
            ]
        )
    if suite.metric == "vbench":
        return _cmd(
            [
                "orbitquant",
                "summarize-vbench-results",
                "--results-dir",
                _vbench_results_dir_path(metrics_root, artifact_name, split),
                "--output",
                metrics_json,
            ]
        )
    raise ValueError(f"native suite {suite.name!r} has no external metric summarizer")


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


def _preflight_lines(plan: dict[str, Any]) -> list[str]:
    metrics = sorted({job["metric"] for job in plan["jobs"]})
    artifact_dirs = sorted({job["artifact_dir"] for job in plan["jobs"]})
    lines = ["# Preflight"]
    if "geneval" in metrics:
        lines.extend(
            [
                ": \"${GENEVAL_DIR:?set GENEVAL_DIR to the GenEval checkout}\"",
                (
                    ": \"${GENEVAL_OBJECT_DETECTOR:?set GENEVAL_OBJECT_DETECTOR "
                    "to the Mask2Former checkpoint directory}\""
                ),
                "if [ ! -f \"${GENEVAL_DIR}/evaluation/evaluate_images.py\" ]; then",
                "  echo 'missing GenEval evaluate_images.py under GENEVAL_DIR' >&2",
                "  exit 1",
                "fi",
                "if [ ! -d \"${GENEVAL_OBJECT_DETECTOR}\" ]; then",
                "  echo 'missing GenEval object detector directory' >&2",
                "  exit 1",
                "fi",
                "python - <<'PY'",
                "import importlib.util",
                (
                    "missing = [name for name in ('mmdet', 'open_clip') "
                    "if importlib.util.find_spec(name) is None]"
                ),
                "if missing:",
                (
                    "    raise SystemExit('missing GenEval Python dependencies: ' "
                    "+ ', '.join(missing))"
                ),
                "PY",
            ]
        )
    if "vbench" in metrics:
        lines.extend(
            [
                "if ! command -v vbench >/dev/null 2>&1; then",
                "  echo 'missing required VBench CLI: vbench' >&2",
                "  exit 127",
                "fi",
                (
                    "echo 'VBench custom_input dimensions: "
                    + " ".join(VBENCH_CUSTOM_INPUT_DIMENSIONS)
                    + "'"
                ),
            ]
        )
    for artifact_dir in artifact_dirs:
        lines.extend(
            [
                f"if [ ! -d {_cmd([artifact_dir])} ]; then",
                f"  echo 'missing artifact directory: {artifact_dir}' >&2",
                "  exit 1",
                "fi",
            ]
        )
    lines.append("")
    return lines


def _stage_lines(label: str, command: str) -> list[str]:
    return [
        _cmd(["stage_log", "START", label]),
        command,
        _cmd(["stage_log", "END", label]),
    ]


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
                metric = str(suite.metric)
                metrics_json = _metrics_json_path(metrics_path, artifact_name, split, metric)
                export_dir = _export_dir_path(metrics_path, artifact_name, split, metric)
                jobs.append(
                    {
                        "suite": suite.name,
                        "model_id": suite.model_id,
                        "bit_setting": bit_setting,
                        "split": split,
                        "metric": suite.metric,
                        "artifact_dir": str(artifact_dir),
                        "export_dir": str(export_dir),
                        "metrics_json": str(metrics_json),
                        "export_command": _export_command(
                            suite,
                            artifact_dir=artifact_dir,
                            split=split,
                            export_dir=export_dir,
                        ),
                        "eval_command": _eval_command(
                            suite,
                            export_dir=export_dir,
                            metrics_root=metrics_path,
                            artifact_name=artifact_name,
                            split=split,
                        ),
                        "summarize_command": _summarize_command(
                            suite,
                            metrics_root=metrics_path,
                            artifact_name=artifact_name,
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
        "stage_log() {",
        "  printf '==== %s %s %s ====\\n' \"$1\" \"$(date -Is)\" \"$2\"",
        "}",
        "",
    ]
    lines.extend(_stage_lines("setup metrics root", _cmd(["mkdir", "-p", Path(metrics_root)])))
    lines.append("")
    lines.append(_cmd(["stage_log", "START", "preflight"]))
    lines.extend(_preflight_lines(plan))
    lines.append(_cmd(["stage_log", "END", "preflight"]))
    lines.append("")
    artifact_dirs: list[str] = []
    for job in plan["jobs"]:
        artifact_dir = str(job["artifact_dir"])
        if artifact_dir not in artifact_dirs:
            artifact_dirs.append(artifact_dir)
        label_prefix = f"{job['suite']} {job['bit_setting']} {job['split']}"
        lines.extend(
            [
                f"# {job['suite']} {job['bit_setting']} {job['split']} {job['metric']}",
                _cmd(["mkdir", "-p", Path(str(job["metrics_json"])).parent]),
                _cmd(["mkdir", "-p", Path(str(job["export_dir"])).parent]),
            ]
        )
        lines.extend(
            _stage_lines(f"{label_prefix} export {job['metric']}", str(job["export_command"]))
        )
        lines.extend(
            _stage_lines(f"{label_prefix} evaluate {job['metric']}", str(job["eval_command"]))
        )
        lines.extend(
            _stage_lines(
                f"{label_prefix} summarize {job['metric']}",
                str(job["summarize_command"]),
            )
        )
        lines.extend(
            _stage_lines(f"{label_prefix} import {job['metric']}", str(job["import_command"]))
        )
        lines.append("")
    if artifact_dirs:
        report_command = ["orbitquant", "report"]
        for artifact_dir in artifact_dirs:
            report_command.extend(["--artifact", artifact_dir])
        report_command.extend(["--output", str(report_output_dir), "--fail-on-missing-required"])
        lines.extend(
            [
                "# Native report",
                *_stage_lines("native eval report", _cmd(report_command)),
                "",
            ]
        )
    return "\n".join(lines)
