from orbitquant.eval.external_plan import build_external_eval_plan, build_external_eval_script
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
    assert "geneval" in first["eval_command"]
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
    assert "vbench" in job["eval_command"]
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
    assert f"mkdir -p {tmp_path / 'metrics'}" in script
    assert script.count("vbench --input-dir") == 4
    assert script.count("orbitquant record-metrics") == 4
    assert "--metric-prefix vbench" in script
    assert "--split original" in script
    assert "--split orbitquant" in script
    assert script.count("orbitquant report") == 1
    assert f"--artifact {tmp_path / 'artifacts' / 'wan-native-w4a6'}" in script
    assert f"--artifact {tmp_path / 'artifacts' / 'wan-native-w4a4'}" in script
    assert f"--output {tmp_path / 'reports'}" in script
    assert "range" not in script.lower()
