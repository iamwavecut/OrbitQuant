from orbitquant.eval.external_plan import build_external_eval_plan
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
