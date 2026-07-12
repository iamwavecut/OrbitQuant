import json
from types import SimpleNamespace

import pytest

import orbitquant.cli.main as cli_main
from orbitquant.cli.main import main


@pytest.mark.parametrize("runtime_mode", ["triton_packed_matmul", "native_packed_matmul"])
def test_cli_native_plan_lists_full_target_bit_matrix_without_range_smoke(
    capsys, tmp_path, runtime_mode
):
    assert (
        main(
            [
                "native-plan",
                "--output-root",
                str(tmp_path / "artifacts"),
                "--seeds",
                "0",
                "--runtime-mode",
                runtime_mode,
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    jobs = payload["jobs"]
    assert payload["job_count"] == 14
    assert {job["suite"] for job in jobs} == {
        "flux2-native",
        "flux1-schnell-native",
        "z-image-native",
        "wan-native",
    }
    assert sum(1 for job in jobs if job["suite"] == "wan-native") == 2
    assert {job["runtime_mode"] for job in jobs} == {runtime_mode}
    wan_w4a6 = next(
        job for job in jobs if job["suite"] == "wan-native" and job["bit_setting"] == "W4A6"
    )
    flux1_w3a3 = next(
        job
        for job in jobs
        if job["suite"] == "flux1-schnell-native" and job["bit_setting"] == "W3A3"
    )
    assert wan_w4a6["width"] == 832
    assert wan_w4a6["height"] == 480
    assert wan_w4a6["frames"] == 81
    assert wan_w4a6["metric"] == "vbench"
    assert flux1_w3a3["width"] == 1024
    assert flux1_w3a3["height"] == 1024
    assert flux1_w3a3["guidance"] == 0.0
    assert flux1_w3a3["artifact_dir"].endswith("flux1-schnell-native-w3a3")
    assert "range" not in json.dumps(payload).lower()


def test_cli_native_plan_defaults_to_auto_fused_runtime(capsys, tmp_path):
    assert (
        main(
            [
                "native-plan",
                "--output-root",
                str(tmp_path / "artifacts"),
                "--seeds",
                "0",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)

    assert payload["job_count"] == 14
    assert {job["runtime_mode"] for job in payload["jobs"]} == {"auto_fused"}


def test_cli_external_eval_plan_lists_metric_runner_import_commands(capsys, tmp_path):
    assert (
        main(
            [
                "external-eval-plan",
                "--suite",
                "wan-native",
                "--output-root",
                str(tmp_path / "artifacts"),
                "--metrics-root",
                str(tmp_path / "metrics"),
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["job_count"] == 4
    job = next(
        item
        for item in payload["jobs"]
        if item["bit_setting"] == "W4A6" and item["split"] == "orbitquant"
    )
    assert job["suite"] == "wan-native"
    assert job["metric"] == "vbench"
    assert job["artifact_dir"].endswith("wan-native-w4a6")
    assert job["metrics_json"].endswith("wan-native-w4a6_orbitquant_vbench.json")
    assert "orbitquant export-vbench" in job["export_command"]
    assert "vbench evaluate" in job["eval_command"]
    assert "orbitquant summarize-vbench-results" in job["summarize_command"]
    assert "orbitquant record-metrics" in job["import_command"]
    assert "--metric-prefix vbench" in job["import_command"]
    assert "--split orbitquant" in job["import_command"]
    assert "range" not in json.dumps(payload).lower()


def test_cli_external_eval_script_prints_metric_runner_script(capsys, tmp_path):
    assert (
        main(
            [
                "external-eval-script",
                "--suite",
                "wan-native",
                "--output-root",
                str(tmp_path / "artifacts"),
                "--metrics-root",
                str(tmp_path / "metrics"),
                "--report-output",
                str(tmp_path / "reports"),
            ]
        )
        == 0
    )

    script = capsys.readouterr().out
    assert script.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    assert "if ! command -v vbench >/dev/null 2>&1; then" in script
    assert f"if [ ! -d {tmp_path / 'artifacts' / 'wan-native-w4a6'} ]; then" in script
    assert "orbitquant export-vbench" in script
    assert "vbench evaluate" in script
    assert "orbitquant summarize-vbench-results" in script
    assert "orbitquant record-metrics" in script
    assert "--metric-prefix vbench" in script
    assert "orbitquant report" in script
    assert f"--output {tmp_path / 'reports'}" in script
    assert "--fail-on-missing-required" in script
    assert "range" not in script.lower()


def test_cli_external_eval_script_resume_skips_existing_metric_summaries(capsys, tmp_path):
    assert (
        main(
            [
                "external-eval-script",
                "--suite",
                "wan-native",
                "--output-root",
                str(tmp_path / "artifacts"),
                "--metrics-root",
                str(tmp_path / "metrics"),
                "--report-output",
                str(tmp_path / "reports"),
                "--resume",
            ]
        )
        == 0
    )

    script = capsys.readouterr().out
    metrics_json = tmp_path / "metrics" / "wan-native-w4a4_original_vbench.json"
    assert f"if [ -s {metrics_json} ]; then" in script
    assert "stage_log SKIP 'wan-native W4A4 original vbench metrics'" in script
    assert "fi\nstage_log START 'wan-native W4A4 original import vbench'" in script
    assert script.count("stage_log SKIP") == 4
    assert script.count("orbitquant record-metrics") == 4
    assert "--fail-on-missing-required" in script


def test_cli_report_can_fail_on_missing_required_metrics(capsys, monkeypatch, tmp_path):
    result = SimpleNamespace(
        report_path=tmp_path / "report.md",
        table_paths={},
        rows=[],
        missing_required_metrics=[{"metric": "geneval_overall"}],
    )
    monkeypatch.setattr(cli_main, "generate_native_eval_report", lambda *args, **kwargs: result)

    assert (
        main(["report", "--artifact", str(tmp_path / "artifact"), "--output", str(tmp_path)]) == 0
    )
    assert json.loads(capsys.readouterr().out)["missing_required_metric_count"] == 1

    assert (
        main(
            [
                "report",
                "--artifact",
                str(tmp_path / "artifact"),
                "--output",
                str(tmp_path),
                "--fail-on-missing-required",
            ]
        )
        == 1
    )
    assert json.loads(capsys.readouterr().out)["missing_required_metric_count"] == 1


def test_cli_native_script_groups_quantize_and_generate_pack_commands(capsys, tmp_path):
    assert (
        main(
            [
                "native-script",
                "--suite",
                "wan-native",
                "--output-root",
                str(tmp_path / "artifacts"),
                "--seeds",
                "0,1",
                "--prompt-limit",
                "1",
                "--device",
                "cuda",
                "--dtype",
                "bfloat16",
                "--activation-kernel-backend",
                "triton_cuda",
                "--runtime-mode",
                "triton_packed_matmul",
            ]
        )
        == 0
    )

    script = capsys.readouterr().out
    assert script.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    assert "stage_log() {" in script
    assert "stage_log START preflight" in script
    assert "stage_log START 'kernel preflight'" in script
    assert "stage_log START 'policy inventories'" in script
    assert "stage_log START 'wan-native W4A6 quantize'" in script
    assert "stage_log START 'wan-native W4A6 original generate-pack'" in script
    assert "stage_log END 'native eval report'" in script
    assert "hf auth whoami" in script
    assert "hf env" in script
    assert "torch.cuda.is_available()" in script
    assert "shutil.disk_usage" in script
    assert "hf models info Wan-AI/Wan2.1-T2V-1.3B-Diffusers --format json >/dev/null" in script
    assert "orbitquant kernel-info" in script
    assert script.count("orbitquant kernel-bench") == 2
    assert script.count("orbitquant inspect-policy") == 1
    assert "--output reports/native/module-inventories/wan-native-policy.json" in script
    assert "--tokens 256 --in-features 3072 --out-features 3072" in script
    assert script.count("orbitquant quantize") == 2
    assert "--suite wan-native" in script
    assert "--weight-bits 4 --activation-bits 6" in script
    assert "--weight-bits 4 --activation-bits 4" in script
    assert "--activation-kernel-backend triton_cuda" in script
    assert "--runtime-mode triton_packed_matmul" in script
    assert (
        "--policy-inventory reports/native/module-inventories/wan-native-policy.json "
        "--runtime-mode triton_packed_matmul"
    ) in script
    assert "--staging-mode component" in script
    assert script.count("\norbitquant generate-pack") == 4
    assert "--split original" in script
    assert "--split orbitquant" in script
    assert "--seeds 0,1" in script
    assert "--prompt-limit 1" in script
    assert script.count("orbitquant validate-artifact") == 4
    assert "--policy-inventory reports/native/module-inventories/wan-native-policy.json" in script
    assert script.count("orbitquant report") == 1
    assert f"--artifact {tmp_path / 'artifacts' / 'wan-native-w4a6'}" in script
    assert f"--artifact {tmp_path / 'artifacts' / 'wan-native-w4a4'}" in script
    assert "--output reports/native" in script
    assert "range" not in script.lower()


def test_cli_native_script_can_generate_geneval_metadata_pack(capsys, tmp_path):
    metadata_jsonl = tmp_path / "evaluation_metadata.jsonl"
    assert (
        main(
            [
                "native-script",
                "--suite",
                "flux1-schnell-native",
                "--output-root",
                str(tmp_path / "artifacts"),
                "--prompt-metadata-jsonl",
                str(metadata_jsonl),
                "--seeds",
                "0",
                "--prompt-limit",
                "2",
            ]
        )
        == 0
    )

    script = capsys.readouterr().out
    assert "--prompt-metadata-jsonl" in script
    assert str(metadata_jsonl) in script
    assert "--prompt-limit 2" in script
    assert "--prompt-pack" not in script
    assert "stage_log START 'flux1-schnell-native W4A4 original generate-pack'" in script
    assert "stage_log START 'flux1-schnell-native W4A4 orbitquant generate-pack'" in script
    assert script.count("\norbitquant generate-pack") == 8


def test_cli_native_script_resume_skips_valid_existing_artifacts(capsys, tmp_path):
    assert (
        main(
            [
                "native-script",
                "--suite",
                "flux2-native",
                "--output-root",
                str(tmp_path / "artifacts"),
                "--seeds",
                "0",
                "--resume",
            ]
        )
        == 0
    )

    script = capsys.readouterr().out
    artifact_dir = tmp_path / "artifacts" / "flux2-native-w4a4"
    inventory_path = "reports/native/module-inventories/flux2-native-policy.json"
    assert (
        f"if orbitquant validate-artifact --artifact {artifact_dir} "
        f"--policy-inventory {inventory_path} --runtime-mode auto_fused"
    ) in script
    assert f"echo 'Skipping existing valid artifact: {artifact_dir}'" in script
    assert "else\norbitquant quantize --suite flux2-native" in script
    assert "\nfi\nstage_log END 'flux2-native W4A4 quantize'" in script
    assert "stage_log START 'flux2-native W4A4 validate quantized artifact'" in script
    assert "\norbitquant generate-pack --suite flux2-native" in script
    assert "--resume-existing" in script
