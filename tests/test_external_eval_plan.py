import shlex

from orbitquant.eval.external_plan import (
    VBENCH_CUSTOM_INPUT_DIMENSIONS,
    build_external_eval_plan,
    build_external_eval_script,
)
from orbitquant.eval.native_settings import get_native_suite


def test_build_external_eval_plan_emits_geneval_import_commands(tmp_path):
    plan = build_external_eval_plan(
        [get_native_suite("flux1-schnell-native")],
        output_root=tmp_path / "artifacts",
        metrics_root=tmp_path / "metrics",
    )

    assert plan["job_count"] == 8
    first = plan["jobs"][0]
    assert first["suite"] == "flux1-schnell-native"
    assert first["split"] == "original"
    assert first["metric"] == "geneval"
    assert first["artifact_dir"].endswith("flux1-schnell-native-w4a4")
    assert first["metrics_json"].endswith("flux1-schnell-native-w4a4_original_geneval.json")
    assert "orbitquant export-geneval" in first["export_command"]
    assert '"${GENEVAL_DIR}/evaluation/evaluate_images.py"' in first["eval_command"]
    assert "orbitquant summarize-geneval-results" in first["summarize_command"]
    assert "orbitquant record-metrics" in first["import_command"]
    assert "--metric-prefix geneval" in first["import_command"]
    assert "--split original" in first["import_command"]


def test_build_external_eval_plan_emits_vbench_import_commands(tmp_path):
    plan = build_external_eval_plan(
        [get_native_suite("wan-native")],
        output_root=tmp_path / "artifacts",
        metrics_root=tmp_path / "metrics",
    )

    assert plan["job_count"] == 4
    job = next(
        item
        for item in plan["jobs"]
        if item["bit_setting"] == "W4A6" and item["split"] == "orbitquant"
    )
    assert job["metric"] == "vbench"
    assert job["suite"] == "wan-native"
    assert job["metrics_json"].endswith("wan-native-w4a6_orbitquant_vbench.json")
    assert "orbitquant export-vbench" in job["export_command"]
    assert "vbench evaluate" in job["eval_command"]
    assert "--mode custom_input" in job["eval_command"]
    assert "scene" in job["eval_command"]
    assert "overall_consistency" in job["eval_command"]
    argv = shlex.split(job["eval_command"])
    dimension_values = argv[argv.index("--dimension") + 1 : argv.index("--videos_path")]
    assert tuple(dimension_values) == VBENCH_CUSTOM_INPUT_DIMENSIONS
    assert "orbitquant summarize-vbench-results" in job["summarize_command"]
    assert "--metric-prefix vbench" in job["import_command"]
    assert "--split orbitquant" in job["import_command"]


def test_build_external_eval_plan_skips_visual_only_extra_target(tmp_path):
    plan = build_external_eval_plan(
        [get_native_suite("flux2-native")],
        output_root=tmp_path / "artifacts",
        metrics_root=tmp_path / "metrics",
    )

    assert plan == {"job_count": 0, "jobs": []}


def test_build_external_eval_script_runs_vbench_import_and_report(tmp_path):
    script = build_external_eval_script(
        [get_native_suite("wan-native")],
        output_root=tmp_path / "artifacts",
        metrics_root=tmp_path / "metrics",
        report_output_dir=tmp_path / "reports",
    )

    assert script.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    assert "stage_log() {" in script
    assert "stage_log START 'setup metrics root'" in script
    assert "stage_log START preflight" in script
    assert "stage_log START 'wan-native W4A6 orbitquant evaluate vbench'" in script
    assert "stage_log END 'native eval report'" in script
    assert f"mkdir -p {tmp_path / 'metrics'}" in script
    assert "if ! command -v vbench >/dev/null 2>&1; then" in script
    assert f"if [ ! -d {tmp_path / 'artifacts' / 'wan-native-w4a6'} ]; then" in script
    assert f"if [ ! -d {tmp_path / 'artifacts' / 'wan-native-w4a4'} ]; then" in script
    assert script.count("orbitquant export-vbench") == 4
    assert script.count("vbench evaluate") == 4
    assert script.count("orbitquant summarize-vbench-results") == 4
    assert script.count("orbitquant record-metrics") == 4
    assert "--metric-prefix vbench" in script
    assert "--split original" in script
    assert "--split orbitquant" in script
    assert script.count("orbitquant report") == 1
    assert f"--artifact {tmp_path / 'artifacts' / 'wan-native-w4a6'}" in script
    assert f"--artifact {tmp_path / 'artifacts' / 'wan-native-w4a4'}" in script
    assert f"--output {tmp_path / 'reports'}" in script
    assert "--fail-on-missing-required" in script
    assert "range" not in script.lower()
