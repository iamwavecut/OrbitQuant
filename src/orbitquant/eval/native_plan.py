from __future__ import annotations

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
