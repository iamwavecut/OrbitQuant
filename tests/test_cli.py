import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

import orbitquant.cli.main as cli_main
from orbitquant.artifacts import save_orbitquant_artifact, validate_orbitquant_artifact
from orbitquant.artifacts.checksums import write_sha256sums_from_manifest
from orbitquant.cli.main import main
from orbitquant.config import OrbitQuantConfig
from orbitquant.modeling import quantize_linear_modules


def test_cli_version_prints_version(capsys):
    assert main(["--version"]) == 0

    output = capsys.readouterr().out
    assert "0.1.0" in output


def test_cli_native_suites_lists_no_range_smoke_settings(capsys):
    assert main(["native-suites"]) == 0

    output = capsys.readouterr().out
    assert "flux2-native" in output
    assert "wan-native" in output
    assert "range" not in output.lower()


def test_cli_kernel_info_reports_backend_capabilities(capsys, monkeypatch):
    monkeypatch.setattr(
        cli_main,
        "backend_capabilities",
        lambda: {
            "cpu": {
                "available": True,
                "claim_status": "reference_only",
                "optimized": False,
                "implemented_stage": None,
                "optimized_stage": None,
                "weight_dequant_optimized": False,
                "weight_pack_optimized": False,
                "weight_quant_optimized": False,
                "adaln_quant_optimized": False,
                "adaln_dequant_optimized": False,
                "hf_kernel_builder_compliant": False,
            },
            "mps": {
                "claim_status": "partial_optimized",
                "implementation": "torch_mps_compile_shader_codebook_rescale",
                "package_format": "torch.mps.compile_shader",
                "implemented_stage": "codebook_lookup_rescale,packed_weight_dequant",
                "optimized_stage": "codebook_lookup_rescale,packed_weight_dequant",
                "weight_dequant_optimized": True,
                "weight_pack_optimized": False,
                "weight_quant_optimized": False,
                "adaln_quant_optimized": False,
                "adaln_dequant_optimized": False,
                "full_fusion": False,
                "upstream_native_mps_op": False,
                "hf_kernel_builder_compliant": False,
            },
            "triton_cuda": {
                "claim_status": "partial_optimized",
                "implemented_stage": (
                    "activation_norm_rpbh_quant_rescale,packed_weight_dequant,"
                    "packed_weight_matmul,lowbit_pack,lowbit_unpack,"
                    "weight_rotation_fwht_quant_pack,"
                    "adaln_rtn_quant_pack,adaln_rtn_dequant"
                ),
                "optimized_stage": (
                    "activation_norm_rpbh_quant_rescale,packed_weight_dequant,"
                    "packed_weight_matmul,lowbit_pack,lowbit_unpack,"
                    "weight_rotation_fwht_quant_pack,"
                    "adaln_rtn_quant_pack,adaln_rtn_dequant"
                ),
                "implementation": "python_triton_orbitquant_pipeline",
                "package_format": "python_triton",
                "weight_dequant_optimized": True,
                "weight_pack_optimized": True,
                "weight_quant_optimized": True,
                "adaln_quant_optimized": True,
                "adaln_dequant_optimized": True,
                "full_fusion": False,
                "hf_kernel_builder_compliant": False,
            },
        },
    )

    assert main(["kernel-info"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["cpu"]["available"] is True
    assert payload["cpu"]["claim_status"] == "reference_only"
    assert payload["cpu"]["optimized"] is False
    assert payload["cpu"]["weight_dequant_optimized"] is False
    assert payload["cpu"]["weight_pack_optimized"] is False
    assert payload["cpu"]["weight_quant_optimized"] is False
    assert payload["cpu"]["adaln_quant_optimized"] is False
    assert payload["cpu"]["adaln_dequant_optimized"] is False
    assert payload["mps"]["implementation"] == "torch_mps_compile_shader_codebook_rescale"
    assert payload["mps"]["package_format"] == "torch.mps.compile_shader"
    assert payload["mps"]["claim_status"] == "partial_optimized"
    assert payload["mps"]["implemented_stage"] == "codebook_lookup_rescale,packed_weight_dequant"
    assert payload["mps"]["optimized_stage"] == "codebook_lookup_rescale,packed_weight_dequant"
    assert payload["mps"]["weight_dequant_optimized"] is True
    assert payload["mps"]["weight_pack_optimized"] is False
    assert payload["mps"]["weight_quant_optimized"] is False
    assert payload["mps"]["adaln_quant_optimized"] is False
    assert payload["mps"]["adaln_dequant_optimized"] is False
    assert payload["mps"]["full_fusion"] is False
    assert payload["mps"]["upstream_native_mps_op"] is False
    assert payload["mps"]["hf_kernel_builder_compliant"] is False
    expected_triton_stage = (
        "activation_norm_rpbh_quant_rescale,packed_weight_dequant,"
        "packed_weight_matmul,lowbit_pack,lowbit_unpack,"
        "weight_rotation_fwht_quant_pack,"
        "adaln_rtn_quant_pack,adaln_rtn_dequant"
    )
    assert payload["triton_cuda"]["implemented_stage"] == expected_triton_stage
    assert payload["triton_cuda"]["optimized_stage"] == expected_triton_stage
    assert payload["triton_cuda"]["implementation"] == "python_triton_orbitquant_pipeline"
    assert payload["triton_cuda"]["package_format"] == "python_triton"
    assert payload["triton_cuda"]["claim_status"] == "partial_optimized"
    assert payload["triton_cuda"]["weight_dequant_optimized"] is True
    assert payload["triton_cuda"]["weight_pack_optimized"] is True
    assert payload["triton_cuda"]["weight_quant_optimized"] is True
    assert payload["triton_cuda"]["adaln_quant_optimized"] is True
    assert payload["triton_cuda"]["adaln_dequant_optimized"] is True
    assert payload["triton_cuda"]["full_fusion"] is False
    assert payload["triton_cuda"]["hf_kernel_builder_compliant"] is False


def test_cli_kernel_bench_prints_stage_timings(capsys):
    assert (
        main(
            [
                "kernel-bench",
                "--tokens",
                "4",
                "--in-features",
                "16",
                "--out-features",
                "8",
                "--block-size",
                "8",
                "--activation-kernel-backend",
                "cpu",
                "--runtime-mode",
                "debug_no_activation_quant",
                "--device",
                "cpu",
                "--dtype",
                "float32",
                "--warmup",
                "0",
                "--iterations",
                "1",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["device"] == "cpu"
    assert payload["runtime_mode"] == "debug_no_activation_quant"
    assert payload["full_fusion"] is False
    assert payload["prewarm"]["total_modules"] == 1
    assert payload["timings_ms"]["weight_quantize_pack_cold_ms"] >= 0.0
    assert payload["timings_ms"]["weight_quantize_pack_hot_ms"] >= 0.0
    assert payload["timings_ms"]["forward_prewarmed_ms"] >= 0.0
    assert payload["selected_activation_kernel_backend"] == "cpu"
    assert payload["weight_quantization_backend"] == "torch_reference"
    assert payload["quantization_buffers"]["source_weight_device"] == "cpu"
    assert payload["quantization_buffers"]["source_weight_is_cuda"] is False
    assert payload["quantization_buffers"]["packed_weight_indices_device"] == "cpu"


def test_cli_kernel_bench_passes_packed_matmul_tile_options(monkeypatch, capsys):
    seen_kwargs = []

    def fake_benchmark_orbit_linear(**kwargs):
        seen_kwargs.append(kwargs)
        return {"ok": True}

    monkeypatch.setattr(cli_main, "benchmark_orbit_linear", fake_benchmark_orbit_linear)

    assert (
        main(
            [
                "kernel-bench",
                "--runtime-mode",
                "triton_packed_matmul",
                "--packed-matmul-block-m",
                "32",
                "--packed-matmul-block-n",
                "64",
                "--packed-matmul-block-k",
                "64",
                "--packed-matmul-num-warps",
                "8",
            ]
        )
        == 0
    )

    assert json.loads(capsys.readouterr().out) == {"ok": True}
    assert seen_kwargs[0]["runtime_mode"] == "triton_packed_matmul"
    assert seen_kwargs[0]["packed_matmul_block_m"] == 32
    assert seen_kwargs[0]["packed_matmul_block_n"] == 64
    assert seen_kwargs[0]["packed_matmul_block_k"] == 64
    assert seen_kwargs[0]["packed_matmul_num_warps"] == 8


def test_cli_quantize_bench_prints_full_model_staging_timings(capsys):
    assert (
        main(
            [
                "quantize-bench",
                "--layers",
                "1",
                "--in-features",
                "16",
                "--hidden-features",
                "32",
                "--block-size",
                "8",
                "--source-device",
                "cpu",
                "--quantization-device",
                "cpu",
                "--staging-mode",
                "component",
                "--dtype",
                "float32",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["source_device"] == "cpu"
    assert payload["quantization_device"] == "cpu"
    assert payload["staging_mode"] == "component"
    assert payload["synchronize_per_module"] is False
    assert payload["summary"]["quantization_staging_mode"] == "component"
    assert payload["summary"]["synchronize_per_module"] is False
    assert payload["summary"]["source_linear_device_counts"]["cpu"] == 7
    assert payload["summary"]["device_transfer_seconds"] >= 0.0
    assert payload["summary"]["quantized_modules"]


def test_cli_native_plan_lists_full_target_bit_matrix_without_range_smoke(capsys, tmp_path):
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
    jobs = payload["jobs"]
    assert payload["job_count"] == 14
    assert {job["suite"] for job in jobs} == {
        "flux2-native",
        "flux1-schnell-native",
        "z-image-native",
        "wan-native",
    }
    assert sum(1 for job in jobs if job["suite"] == "wan-native") == 2
    wan_w4a6 = next(
        job
        for job in jobs
        if job["suite"] == "wan-native" and job["bit_setting"] == "W4A6"
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


def test_cli_report_can_fail_on_missing_required_metrics(capsys, monkeypatch, tmp_path):
    result = SimpleNamespace(
        report_path=tmp_path / "report.md",
        table_paths={},
        rows=[],
        missing_required_metrics=[{"metric": "geneval_overall"}],
    )
    monkeypatch.setattr(cli_main, "generate_native_eval_report", lambda *args, **kwargs: result)

    assert (
        main(["report", "--artifact", str(tmp_path / "artifact"), "--output", str(tmp_path)])
        == 0
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
    assert "--staging-mode component" in script
    assert script.count("\norbitquant generate-pack") == 4
    assert "--split original" in script
    assert "--split orbitquant" in script
    assert "--seeds 0,1" in script
    assert "--prompt-limit 1" in script
    assert script.count("orbitquant validate-artifact") == 4
    assert (
        "--policy-inventory reports/native/module-inventories/wan-native-policy.json"
        in script
    )
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
        f"--policy-inventory {inventory_path}"
    ) in script
    assert f"echo 'Skipping existing valid artifact: {artifact_dir}'" in script
    assert "else\norbitquant quantize --suite flux2-native" in script
    assert "\nfi\nstage_log END 'flux2-native W4A4 quantize'" in script
    assert "stage_log START 'flux2-native W4A4 validate quantized artifact'" in script
    assert "\norbitquant generate-pack --suite flux2-native" in script
    assert "--resume-existing" in script


def test_cli_generate_requires_prompt_and_output():
    try:
        main(["generate", "--suite", "flux2-native"])
    except (SystemExit, ValueError) as exc:
        assert "prompt" in str(exc) or "output" in str(exc) or getattr(exc, "code", 0) != 0
    else:
        raise AssertionError("generate accepted missing prompt/output arguments")


def test_cli_record_metrics_imports_nested_json_into_artifact(capsys, tmp_path):
    model = torch.nn.Module()
    model.transformer_blocks = torch.nn.ModuleList(
        [torch.nn.ModuleDict({"attn": torch.nn.ModuleDict({"to_q": torch.nn.Linear(8, 8)})})]
    )
    config = OrbitQuantConfig(block_size=4, target_policy="flux")
    summary = quantize_linear_modules(model, config)
    save_orbitquant_artifact(
        model,
        tmp_path,
        config=config,
        source_model_id="black-forest-labs/FLUX.1-schnell",
        source_revision="abc123",
        source_license="apache-2.0",
        summary=summary,
    )
    metrics_path = tmp_path / "geneval.json"
    metrics_path.write_text(
        json.dumps(
            {
                "overall": 0.71,
                "per_task": {
                    "single_object": 0.92,
                    "counting": 0.47,
                },
                "notes": "ignored",
            }
        )
    )

    assert (
        main(
            [
                "record-metrics",
                "--artifact",
                str(tmp_path),
                "--split",
                "orbitquant",
                "--metrics-json",
                str(metrics_path),
                "--metric-prefix",
                "geneval",
                "--suite",
                "flux1-schnell-native",
                "--seed",
                "0",
                "--bit-setting",
                "W4A4",
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    record = json.loads((tmp_path / "benchmark" / "orbitquant.metrics.jsonl").read_text())
    csv_text = (tmp_path / "benchmark" / "orbitquant.metrics.csv").read_text()
    manifest = json.loads((tmp_path / "orbitquant_manifest.json").read_text())
    assert output["metrics"] == {
        "geneval_overall": 0.71,
        "geneval_per_task_counting": 0.47,
        "geneval_per_task_single_object": 0.92,
    }
    assert record["metrics"] == output["metrics"]
    assert record["metadata"] == {
        "suite": "flux1-schnell-native",
        "seed": 0,
        "bit_setting": "W4A4",
        "metrics_source": str(metrics_path),
    }
    assert "geneval_per_task_single_object,0.92" in csv_text
    assert "benchmark/orbitquant.metrics.jsonl" in manifest["checksums"]


def test_cli_generate_dry_run_prints_native_request(capsys, tmp_path):
    assert (
        main(
            [
                "generate",
                "--suite",
                "flux2-native",
                "--prompt",
                "A native prompt",
                "--output",
                str(tmp_path),
                "--seed",
                "9",
                "--device",
                "cpu",
                "--dry-run",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "black-forest-labs/FLUX.2-klein-4B" in output
    assert '"height": 1024' in output
    assert '"width": 1024' in output


def test_cli_generate_dry_run_prints_quantized_native_request(capsys, tmp_path):
    assert (
        main(
            [
                "generate",
                "--suite",
                "wan-native",
                "--prompt",
                "A native prompt",
                "--output",
                str(tmp_path),
                "--seed",
                "9",
                "--device",
                "cpu",
                "--bit-setting",
                "W4A6",
                "--activation-kernel-backend",
                "cpu",
                "--dry-run",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert '"height": 480' in output
    assert '"width": 832' in output
    assert '"weight_bits": 4' in output
    assert '"activation_bits": 6' in output
    assert '"activation_kernel_backend": "cpu"' in output
    assert '"target_policy": "wan"' in output


def test_cli_validate_generation_reports_valid_native_output(capsys, tmp_path):
    output_path = tmp_path / "flux2-native_seed5_W4A4_simple-object.png"
    Image.new("RGB", (16, 16), "red").save(output_path)
    output_path.with_suffix(".png.json").write_text(
        json.dumps(
            {
                "suite": "flux2-native",
                "model_id": "black-forest-labs/FLUX.2-klein-4B",
                "prompt": "A native image",
                "seed": 5,
                "height": 1024,
                "width": 1024,
                "frames": None,
                "steps": 4,
                "guidance": 1.0,
                "quantization": {
                    "config": {
                        "weight_bits": 4,
                        "activation_bits": 4,
                    }
                },
            }
        )
        + "\n"
    )

    assert (
        main(
            [
                "validate-generation",
                "--suite",
                "flux2-native",
                "--output",
                str(output_path),
                "--seed",
                "5",
                "--bit-setting",
                "W4A4",
                "--prompt",
                "A native image",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is True
    assert payload["suite"] == "flux2-native"
    assert payload["bit_setting"] == "W4A4"


def test_cli_generate_dry_run_uses_artifact_source_and_default_assets_output(capsys, tmp_path):
    model = torch.nn.Module()
    model.transformer_blocks = torch.nn.ModuleList(
        [torch.nn.ModuleDict({"attn": torch.nn.ModuleDict({"to_q": torch.nn.Linear(8, 8)})})]
    )
    config = OrbitQuantConfig(block_size=4, target_policy="generic_dit")
    summary = quantize_linear_modules(model, config)
    save_orbitquant_artifact(
        model,
        tmp_path,
        config=config,
        source_model_id="example/artifact-model",
        source_revision="abc123",
        source_license="apache-2.0",
        summary=summary,
    )

    assert (
        main(
            [
                "generate",
                "--suite",
                "flux2-native",
                "--prompt",
                "A native prompt",
                "--artifact",
                str(tmp_path),
                "--device",
                "cpu",
                "--dry-run",
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert output["model_id"] == "example/artifact-model"
    assert output["artifact"] == str(tmp_path)
    assert output["output"] == str(tmp_path / "assets")
    assert output["quantization_config"]["weight_bits"] == 4
    assert output["quantization_config"]["activation_bits"] == 4


def test_cli_generate_dry_run_selects_prompt_from_artifact_pack(capsys, tmp_path):
    model = torch.nn.Module()
    model.transformer_blocks = torch.nn.ModuleList(
        [torch.nn.ModuleDict({"attn": torch.nn.ModuleDict({"to_q": torch.nn.Linear(8, 8)})})]
    )
    config = OrbitQuantConfig(block_size=4, target_policy="generic_dit")
    summary = quantize_linear_modules(model, config)
    save_orbitquant_artifact(
        model,
        tmp_path,
        config=config,
        source_model_id="example/artifact-model",
        source_revision="abc123",
        source_license="apache-2.0",
        summary=summary,
    )

    assert (
        main(
            [
                "generate",
                "--suite",
                "flux2-native",
                "--artifact",
                str(tmp_path),
                "--prompt-id",
                "english-text-rendering",
                "--device",
                "cpu",
                "--dry-run",
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert output["prompt_record"]["id"] == "english-text-rendering"
    assert (
        output["pipeline_kwargs"]["prompt"]
        == 'A clean street sign with the exact text "ORBIT QUANT"'
    )


def test_cli_generate_with_artifact_loads_component_and_records_metrics(
    monkeypatch,
    capsys,
    tmp_path,
):
    class TinyPipeline:
        def __init__(self):
            self.transformer = torch.nn.Module()
            self.transformer.transformer_blocks = torch.nn.ModuleList(
                [
                    torch.nn.ModuleDict(
                        {"attn": torch.nn.ModuleDict({"to_q": torch.nn.Linear(8, 8)})}
                    )
                ]
            )
            self.device = None

        def to(self, device):
            self.device = device
            return self

        def __call__(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(images=[Image.new("RGB", (16, 16), "blue")])

    source = TinyPipeline()
    config = OrbitQuantConfig(block_size=4, target_policy="generic_dit")
    summary = quantize_linear_modules(source.transformer, config)
    save_orbitquant_artifact(
        source.transformer,
        tmp_path,
        config=config,
        source_model_id="example/artifact-model",
        source_revision="abc123",
        source_license="apache-2.0",
        summary=summary,
    )
    restored = TinyPipeline()

    class FakeDiffusionPipeline:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            assert model_id == "example/artifact-model"
            assert kwargs["torch_dtype"] is torch.float32
            return restored

    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(Flux2KleinPipeline=FakeDiffusionPipeline),
    )

    assert (
        main(
            [
                "generate",
                "--suite",
                "flux2-native",
                "--prompt",
                "A native prompt",
                "--artifact",
                str(tmp_path),
                "--seed",
                "3",
                "--device",
                "cpu",
                "--dtype",
                "float32",
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    record = json.loads((tmp_path / "benchmark" / "orbitquant.metrics.jsonl").read_text())
    assert output["artifact"] == str(tmp_path)
    assert output["output_path"].endswith("flux2-native_seed3_W4A4.png")
    assert (tmp_path / "assets" / "flux2-native_seed3_W4A4.png").is_file()
    assert record["split"] == "orbitquant"
    assert record["metrics"]["generated_samples"] == 1
    assert record["metrics"]["wall_time_seconds"] >= 0.0
    assert record["metadata"]["output_path"] == output["output_path"]
    assert record["metadata"]["device"] == "cpu"
    assert record["metadata"]["dtype"] == "float32"
    sample_metadata = json.loads(
        (tmp_path / "assets" / "flux2-native_seed3_W4A4.png.json").read_text()
    )
    assert sample_metadata["device"] == "cpu"
    assert sample_metadata["dtype"] == "float32"
    assert sample_metadata["pipeline_class"] == "TinyPipeline"
    manifest = json.loads((tmp_path / "orbitquant_manifest.json").read_text())
    assert "assets/flux2-native_seed3_W4A4.png" in manifest["checksums"]
    assert "assets/flux2-native_seed3_W4A4.png.json" in manifest["checksums"]
    assert restored.device == "cpu"


def test_cli_generate_with_artifact_original_split_skips_quantized_component(
    monkeypatch,
    capsys,
    tmp_path,
):
    class TinyPipeline:
        def __init__(self):
            self.transformer = torch.nn.Module()
            self.transformer.transformer_blocks = torch.nn.ModuleList(
                [
                    torch.nn.ModuleDict(
                        {"attn": torch.nn.ModuleDict({"to_q": torch.nn.Linear(8, 8)})}
                    )
                ]
            )

        def to(self, device):
            return self

        def __call__(self, **kwargs):
            return SimpleNamespace(images=[Image.new("RGB", (16, 16), "yellow")])

    source = TinyPipeline()
    config = OrbitQuantConfig(block_size=4, target_policy="generic_dit")
    summary = quantize_linear_modules(source.transformer, config)
    save_orbitquant_artifact(
        source.transformer,
        tmp_path,
        config=config,
        source_model_id="example/artifact-model",
        source_revision="abc123",
        source_license="apache-2.0",
        summary=summary,
    )

    class FakeDiffusionPipeline:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            return TinyPipeline()

    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(Flux2KleinPipeline=FakeDiffusionPipeline),
    )
    monkeypatch.setattr(
        cli_main,
        "load_quantized_pipeline_component",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("original split must not load quantized component")
        ),
    )

    assert (
        main(
            [
                "generate",
                "--suite",
                "flux2-native",
                "--prompt",
                "A native prompt",
                "--artifact",
                str(tmp_path),
                "--split",
                "original",
                "--seed",
                "3",
                "--device",
                "cpu",
                "--dtype",
                "float32",
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    record = json.loads((tmp_path / "benchmark" / "original.metrics.jsonl").read_text())
    assert output["output_path"].endswith("flux2-native_seed3_original.png")
    assert record["split"] == "original"
    assert record["metadata"]["bit_setting"] == "original"
    assert (tmp_path / "benchmark" / "orbitquant.metrics.jsonl").read_text() == ""


def test_cli_generate_creates_comparison_when_original_pair_exists(
    monkeypatch,
    capsys,
    tmp_path,
):
    class TinyPipeline:
        color = "red"

        def __init__(self):
            self.transformer = torch.nn.Module()
            self.transformer.transformer_blocks = torch.nn.ModuleList(
                [
                    torch.nn.ModuleDict(
                        {"attn": torch.nn.ModuleDict({"to_q": torch.nn.Linear(8, 8)})}
                    )
                ]
            )

        def to(self, device):
            return self

        def __call__(self, **kwargs):
            return SimpleNamespace(images=[Image.new("RGB", (16, 16), self.color)])

    source = TinyPipeline()
    config = OrbitQuantConfig(block_size=4, target_policy="generic_dit")
    summary = quantize_linear_modules(source.transformer, config)
    save_orbitquant_artifact(
        source.transformer,
        tmp_path,
        config=config,
        source_model_id="example/artifact-model",
        source_revision="abc123",
        source_license="apache-2.0",
        summary=summary,
    )

    class FakeDiffusionPipeline:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            return TinyPipeline()

    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(Flux2KleinPipeline=FakeDiffusionPipeline),
    )

    common_args = [
        "generate",
        "--suite",
        "flux2-native",
        "--prompt",
        "A native prompt",
        "--artifact",
        str(tmp_path),
        "--seed",
        "4",
        "--device",
        "cpu",
        "--dtype",
        "float32",
    ]
    assert main([*common_args, "--split", "original"]) == 0
    capsys.readouterr()
    TinyPipeline.color = "blue"
    assert main(common_args) == 0

    output = json.loads(capsys.readouterr().out)
    manifest = json.loads((tmp_path / "orbitquant_manifest.json").read_text())
    comparison = "assets/original_vs_orbitquant_flux2-native_seed4_W4A4_prompt.webp"
    assert comparison in manifest["checksums"]
    assert (tmp_path / comparison).is_file()
    assert output["artifact_comparisons"] == [comparison]


def test_cli_generate_with_video_artifact_records_contact_sheet_asset(
    monkeypatch,
    capsys,
    tmp_path,
):
    class TinyWanPipeline:
        def __init__(self):
            self.transformer = torch.nn.Module()
            self.transformer.transformer_blocks = torch.nn.ModuleList(
                [
                    torch.nn.ModuleDict(
                        {"attn": torch.nn.ModuleDict({"to_q": torch.nn.Linear(8, 8)})}
                    )
                ]
            )

        def to(self, device):
            self.device = device
            return self

        def __call__(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(frames=np.zeros((1, 2, 8, 8, 3), dtype=np.uint8))

    def fake_export_to_video(frames, path):
        with open(path, "wb") as output:
            output.write(b"fake mp4")

    source = TinyWanPipeline()
    config = OrbitQuantConfig(block_size=4, target_policy="generic_dit")
    summary = quantize_linear_modules(source.transformer, config)
    save_orbitquant_artifact(
        source.transformer,
        tmp_path,
        config=config,
        source_model_id="example/artifact-model",
        source_revision="abc123",
        source_license="apache-2.0",
        summary=summary,
    )
    restored = TinyWanPipeline()

    class FakeWanPipeline:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            assert model_id == "example/artifact-model"
            assert kwargs["torch_dtype"] is torch.float32
            return restored

    monkeypatch.setitem(sys.modules, "diffusers", SimpleNamespace(WanPipeline=FakeWanPipeline))
    monkeypatch.setitem(
        sys.modules,
        "diffusers.utils",
        SimpleNamespace(export_to_video=fake_export_to_video),
    )

    assert (
        main(
            [
                "generate",
                "--suite",
                "wan-native",
                "--prompt",
                "A native video prompt",
                "--artifact",
                str(tmp_path),
                "--seed",
                "2",
                "--device",
                "cpu",
                "--dtype",
                "float32",
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    manifest = json.loads((tmp_path / "orbitquant_manifest.json").read_text())
    assert output["output_path"].endswith("wan-native_seed2_W4A4.mp4")
    assert "assets/wan-native_seed2_W4A4.mp4" in manifest["checksums"]
    assert "assets/wan-native_seed2_W4A4.mp4.json" in manifest["checksums"]
    assert "assets/wan-native_seed2_W4A4_contact_sheet.webp" in manifest["checksums"]


def test_cli_generate_pack_dry_run_lists_prompt_seed_jobs(capsys, tmp_path):
    model = torch.nn.Module()
    model.transformer_blocks = torch.nn.ModuleList(
        [torch.nn.ModuleDict({"attn": torch.nn.ModuleDict({"to_q": torch.nn.Linear(8, 8)})})]
    )
    config = OrbitQuantConfig(block_size=4, target_policy="generic_dit")
    summary = quantize_linear_modules(model, config)
    save_orbitquant_artifact(
        model,
        tmp_path,
        config=config,
        source_model_id="example/artifact-model",
        source_revision="abc123",
        source_license="apache-2.0",
        summary=summary,
    )

    assert (
        main(
            [
                "generate-pack",
                "--suite",
                "flux2-native",
                "--artifact",
                str(tmp_path),
                "--prompt-limit",
                "2",
                "--seeds",
                "0,1",
                "--device",
                "cpu",
                "--dry-run",
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert output["job_count"] == 4
    assert output["jobs"][0]["prompt_record"]["id"] == "simple-object"
    assert output["jobs"][0]["seed"] == 0
    assert output["jobs"][2]["seed"] == 1
    assert output["output"] == str(tmp_path / "assets")


def test_cli_generate_pack_dry_run_accepts_geneval_smoke_prompt_pack(capsys, tmp_path):
    model = torch.nn.Module()
    model.transformer_blocks = torch.nn.ModuleList(
        [torch.nn.ModuleDict({"attn": torch.nn.ModuleDict({"to_q": torch.nn.Linear(8, 8)})})]
    )
    config = OrbitQuantConfig(block_size=4, target_policy="generic_dit")
    summary = quantize_linear_modules(model, config)
    save_orbitquant_artifact(
        model,
        tmp_path,
        config=config,
        source_model_id="example/artifact-model",
        source_revision="abc123",
        source_license="apache-2.0",
        summary=summary,
    )

    assert (
        main(
            [
                "generate-pack",
                "--suite",
                "flux1-schnell-native",
                "--artifact",
                str(tmp_path),
                "--prompt-pack",
                "geneval-smoke",
                "--prompt-limit",
                "2",
                "--seeds",
                "0",
                "--device",
                "cpu",
                "--dry-run",
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert output["prompt_pack"] == "geneval_smoke_v1"
    assert output["job_count"] == 2
    assert output["jobs"][0]["prompt_record"]["id"].startswith("geneval-00000-")
    assert output["jobs"][0]["prompt_record"]["geneval"]["tag"] == "single_object"


def test_cli_generate_pack_dry_run_accepts_geneval_metadata_jsonl(capsys, tmp_path):
    model = torch.nn.Module()
    model.transformer_blocks = torch.nn.ModuleList(
        [torch.nn.ModuleDict({"attn": torch.nn.ModuleDict({"to_q": torch.nn.Linear(8, 8)})})]
    )
    config = OrbitQuantConfig(block_size=4, target_policy="generic_dit")
    summary = quantize_linear_modules(model, config)
    save_orbitquant_artifact(
        model,
        tmp_path,
        config=config,
        source_model_id="example/artifact-model",
        source_revision="abc123",
        source_license="apache-2.0",
        summary=summary,
    )
    metadata_jsonl = tmp_path / "evaluation_metadata.jsonl"
    metadata_jsonl.write_text(
        json.dumps(
            {
                "tag": "single_object",
                "include": [{"class": "bench", "count": 1}],
                "prompt": "a photo of a bench",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert (
        main(
            [
                "generate-pack",
                "--suite",
                "z-image-native",
                "--artifact",
                str(tmp_path),
                "--prompt-metadata-jsonl",
                str(metadata_jsonl),
                "--seeds",
                "4",
                "--device",
                "cpu",
                "--dry-run",
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert output["prompt_pack"] == "geneval_metadata_jsonl"
    assert output["job_count"] == 1
    assert output["jobs"][0]["seed"] == 4
    assert output["jobs"][0]["prompt_record"]["geneval"]["include"] == [
        {"class": "bench", "count": 1}
    ]


def test_cli_generate_pack_runs_jobs_once_per_prompt_seed_and_records_artifacts(
    monkeypatch,
    capsys,
    tmp_path,
):
    class TinyPipeline:
        def __init__(self):
            self.transformer = torch.nn.Module()
            self.transformer.transformer_blocks = torch.nn.ModuleList(
                [
                    torch.nn.ModuleDict(
                        {"attn": torch.nn.ModuleDict({"to_q": torch.nn.Linear(8, 8)})}
                    )
                ]
            )
            self.calls = []

        def to(self, device):
            self.device = device
            return self

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(images=[Image.new("RGB", (16, 16), "green")])

    source = TinyPipeline()
    config = OrbitQuantConfig(block_size=4, target_policy="generic_dit")
    summary = quantize_linear_modules(source.transformer, config)
    save_orbitquant_artifact(
        source.transformer,
        tmp_path,
        config=config,
        source_model_id="example/artifact-model",
        source_revision="abc123",
        source_license="apache-2.0",
        summary=summary,
    )
    restored = TinyPipeline()

    class FakeDiffusionPipeline:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            assert model_id == "example/artifact-model"
            assert kwargs["torch_dtype"] is torch.float32
            return restored

    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(Flux2KleinPipeline=FakeDiffusionPipeline),
    )

    assert (
        main(
            [
                "generate-pack",
                "--suite",
                "flux2-native",
                "--artifact",
                str(tmp_path),
                "--prompt-id",
                "simple-object",
                "--prompt-id",
                "counting",
                "--seeds",
                "3",
                "--device",
                "cpu",
                "--dtype",
                "float32",
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    manifest = json.loads((tmp_path / "orbitquant_manifest.json").read_text())
    metrics_rows = (tmp_path / "benchmark" / "orbitquant.metrics.jsonl").read_text().splitlines()
    assert output["job_count"] == 2
    assert len(restored.calls) == 2
    assert "assets/flux2-native_seed3_W4A4_simple-object.png" in manifest["checksums"]
    assert "assets/flux2-native_seed3_W4A4_counting.png" in manifest["checksums"]
    assert len(metrics_rows) == 2
    assert json.loads(metrics_rows[0])["metadata"]["prompt_record"]["id"] == "simple-object"


def test_cli_generate_pack_skip_checksums_refreshes_artifact_once_at_end(
    monkeypatch,
    capsys,
    tmp_path,
):
    class TinyPipeline:
        def __init__(self):
            self.transformer = torch.nn.Module()
            self.transformer.transformer_blocks = torch.nn.ModuleList(
                [
                    torch.nn.ModuleDict(
                        {"attn": torch.nn.ModuleDict({"to_q": torch.nn.Linear(8, 8)})}
                    )
                ]
            )
            self.calls = []

        def to(self, device):
            return self

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(images=[Image.new("RGB", (16, 16), "green")])

    source = TinyPipeline()
    config = OrbitQuantConfig(block_size=4, target_policy="generic_dit")
    summary = quantize_linear_modules(source.transformer, config)
    save_orbitquant_artifact(
        source.transformer,
        tmp_path,
        config=config,
        source_model_id="example/artifact-model",
        source_revision="abc123",
        source_license="apache-2.0",
        summary=summary,
    )
    restored = TinyPipeline()

    class FakeDiffusionPipeline:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            assert model_id == "example/artifact-model"
            return restored

    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(Flux2KleinPipeline=FakeDiffusionPipeline),
    )

    assert (
        main(
            [
                "generate-pack",
                "--suite",
                "flux2-native",
                "--artifact",
                str(tmp_path),
                "--prompt-id",
                "simple-object",
                "--seeds",
                "3",
                "--resume-existing",
                "--skip-artifact-checksums",
                "--device",
                "cpu",
                "--dtype",
                "float32",
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    manifest = json.loads((tmp_path / "orbitquant_manifest.json").read_text())
    validation = validate_orbitquant_artifact(tmp_path)
    image_path = "assets/flux2-native_seed3_W4A4_simple-object.png"
    metadata_path = "assets/flux2-native_seed3_W4A4_simple-object.png.json"
    assert output["job_count"] == 1
    assert output["run_count"] == 1
    assert output["checksum_refresh"]["checksum_count"] == len(manifest["checksums"])
    assert validation["valid"] is True
    assert image_path in manifest["checksums"]
    assert metadata_path in manifest["checksums"]
    assert "benchmark/orbitquant.metrics.jsonl" in manifest["checksums"]
    assert output["artifact_comparisons"] == []
    assert output["outputs"][0]["comparisons"] == []


def test_cli_generate_pack_defers_comparison_creation_until_after_jobs(
    monkeypatch,
    capsys,
    tmp_path,
):
    class TinyPipeline:
        def __init__(self):
            self.transformer = torch.nn.Module()
            self.transformer.transformer_blocks = torch.nn.ModuleList(
                [
                    torch.nn.ModuleDict(
                        {"attn": torch.nn.ModuleDict({"to_q": torch.nn.Linear(8, 8)})}
                    )
                ]
            )

        def to(self, device):
            return self

        def __call__(self, **kwargs):
            return SimpleNamespace(images=[Image.new("RGB", (16, 16), "green")])

    source = TinyPipeline()
    config = OrbitQuantConfig(block_size=4, target_policy="generic_dit")
    summary = quantize_linear_modules(source.transformer, config)
    save_orbitquant_artifact(
        source.transformer,
        tmp_path,
        config=config,
        source_model_id="example/artifact-model",
        source_revision="abc123",
        source_license="apache-2.0",
        summary=summary,
    )

    class FakeDiffusionPipeline:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            assert model_id == "example/artifact-model"
            return TinyPipeline()

    comparison_calls = []

    def fake_create_comparisons(artifact_dir, **kwargs):
        comparison_calls.append({"artifact_dir": artifact_dir, "kwargs": kwargs})
        return ["assets/fake-comparison.webp"]

    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(Flux2KleinPipeline=FakeDiffusionPipeline),
    )
    monkeypatch.setattr(cli_main, "create_artifact_image_comparisons", fake_create_comparisons)

    assert (
        main(
            [
                "generate-pack",
                "--suite",
                "flux2-native",
                "--artifact",
                str(tmp_path),
                "--prompt-id",
                "simple-object",
                "--prompt-id",
                "counting",
                "--seeds",
                "3",
                "--device",
                "cpu",
                "--dtype",
                "float32",
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert len(comparison_calls) == 1
    assert comparison_calls[0]["kwargs"]["comparison_keys"] == {
        ("flux2-native", 3, "simple-object"),
        ("flux2-native", 3, "counting"),
    }
    assert output["run_count"] == 2
    assert output["artifact_comparisons"] == ["assets/fake-comparison.webp"]
    assert [item["comparisons"] for item in output["outputs"]] == [[], []]


def test_cli_generate_pack_prompt_metadata_disables_comparisons_by_default(
    monkeypatch,
    capsys,
    tmp_path,
):
    class TinyPipeline:
        def __init__(self):
            self.transformer = torch.nn.Module()
            self.transformer.transformer_blocks = torch.nn.ModuleList(
                [
                    torch.nn.ModuleDict(
                        {"attn": torch.nn.ModuleDict({"to_q": torch.nn.Linear(8, 8)})}
                    )
                ]
            )

        def to(self, device):
            return self

        def __call__(self, **kwargs):
            return SimpleNamespace(images=[Image.new("RGB", (16, 16), "green")])

    source = TinyPipeline()
    config = OrbitQuantConfig(block_size=4, target_policy="generic_dit")
    summary = quantize_linear_modules(source.transformer, config)
    save_orbitquant_artifact(
        source.transformer,
        tmp_path,
        config=config,
        source_model_id="example/artifact-model",
        source_revision="abc123",
        source_license="apache-2.0",
        summary=summary,
    )
    metadata_jsonl = tmp_path / "evaluation_metadata.jsonl"
    metadata_jsonl.write_text(
        json.dumps(
            {
                "tag": "single_object",
                "include": [{"class": "bench", "count": 1}],
                "prompt": "a photo of a bench",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class FakeDiffusionPipeline:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            assert model_id == "example/artifact-model"
            return TinyPipeline()

    def fail_create_comparisons(*args, **kwargs):
        raise AssertionError("GenEval metadata packs must not create comparison sheets by default")

    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(Flux2KleinPipeline=FakeDiffusionPipeline),
    )
    monkeypatch.setattr(cli_main, "create_artifact_image_comparisons", fail_create_comparisons)

    assert (
        main(
            [
                "generate-pack",
                "--suite",
                "flux2-native",
                "--artifact",
                str(tmp_path),
                "--prompt-metadata-jsonl",
                str(metadata_jsonl),
                "--seeds",
                "3",
                "--skip-artifact-checksums",
                "--device",
                "cpu",
                "--dtype",
                "float32",
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert output["run_count"] == 1
    assert output["artifact_comparisons"] == []
    assert output["outputs"][0]["comparisons"] == []


def test_cli_generate_pack_resume_existing_skips_completed_outputs(
    monkeypatch,
    capsys,
    tmp_path,
):
    model = torch.nn.Module()
    model.transformer_blocks = torch.nn.ModuleList(
        [torch.nn.ModuleDict({"attn": torch.nn.ModuleDict({"to_q": torch.nn.Linear(8, 8)})})]
    )
    config = OrbitQuantConfig(block_size=4, target_policy="generic_dit")
    summary = quantize_linear_modules(model, config)
    save_orbitquant_artifact(
        model,
        tmp_path,
        config=config,
        source_model_id="example/artifact-model",
        source_revision="abc123",
        source_license="apache-2.0",
        summary=summary,
    )
    output_path = tmp_path / "assets" / "flux2-native_seed3_W4A4_simple-object.png"
    Image.new("RGB", (16, 16), "green").save(output_path)
    output_path.with_suffix(".png.json").write_text(
        json.dumps(
            {
                "suite": "flux2-native",
                "model_id": "example/artifact-model",
                "prompt": (
                    "A red ceramic mug on a wooden desk, soft daylight, "
                    "shallow depth of field"
                ),
                "seed": 3,
                "height": 1024,
                "width": 1024,
                "frames": None,
                "steps": 4,
                "guidance": 1.0,
                "quantization": {
                    "config": {
                        "weight_bits": 4,
                        "activation_bits": 4,
                    }
                },
            }
        )
        + "\n"
    )
    monkeypatch.setattr(
        cli_main,
        "load_pipeline_for_suite",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("completed resume jobs must not load a pipeline")
        ),
    )

    assert (
        main(
            [
                "generate-pack",
                "--suite",
                "flux2-native",
                "--artifact",
                str(tmp_path),
                "--prompt-id",
                "simple-object",
                "--seeds",
                "3",
                "--resume-existing",
                "--device",
                "cpu",
                "--dtype",
                "float32",
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert output["job_count"] == 1
    assert output["run_count"] == 0
    assert output["skipped_count"] == 1
    assert output["skipped_outputs"] == [str(output_path)]


def test_cli_generate_pack_resume_existing_reruns_invalid_metadata(
    monkeypatch,
    capsys,
    tmp_path,
):
    class TinyPipeline:
        def __init__(self):
            self.transformer = torch.nn.Module()
            self.transformer.transformer_blocks = torch.nn.ModuleList(
                [
                    torch.nn.ModuleDict(
                        {"attn": torch.nn.ModuleDict({"to_q": torch.nn.Linear(8, 8)})}
                    )
                ]
            )
            self.calls = []

        def to(self, device):
            return self

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(images=[Image.new("RGB", (16, 16), "purple")])

    source = TinyPipeline()
    config = OrbitQuantConfig(block_size=4, target_policy="generic_dit")
    summary = quantize_linear_modules(source.transformer, config)
    save_orbitquant_artifact(
        source.transformer,
        tmp_path,
        config=config,
        source_model_id="example/artifact-model",
        source_revision="abc123",
        source_license="apache-2.0",
        summary=summary,
    )
    output_path = tmp_path / "assets" / "flux2-native_seed3_W4A4_simple-object.png"
    Image.new("RGB", (16, 16), "green").save(output_path)
    output_path.with_suffix(".png.json").write_text('{"status":"complete"}\n')
    restored = TinyPipeline()

    class FakeDiffusionPipeline:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            assert model_id == "example/artifact-model"
            return restored

    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(Flux2KleinPipeline=FakeDiffusionPipeline),
    )

    assert (
        main(
            [
                "generate-pack",
                "--suite",
                "flux2-native",
                "--artifact",
                str(tmp_path),
                "--prompt-id",
                "simple-object",
                "--seeds",
                "3",
                "--resume-existing",
                "--device",
                "cpu",
                "--dtype",
                "float32",
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert output["job_count"] == 1
    assert output["run_count"] == 1
    assert output["skipped_count"] == 0
    assert len(restored.calls) == 1


def test_cli_quantize_saves_transformer_component_artifact(monkeypatch, capsys, tmp_path):
    class TinyPipeline:
        def __init__(self):
            self.transformer = torch.nn.Module()
            self.transformer.transformer_blocks = torch.nn.ModuleList(
                [
                    torch.nn.ModuleDict(
                        {"attn": torch.nn.ModuleDict({"to_q": torch.nn.Linear(8, 8)})}
                    )
                ]
            )
            self.device = None

        def to(self, device):
            self.device = device
            return self

    pipeline = TinyPipeline()

    class FakeDiffusionPipeline:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            assert model_id == "example/model"
            assert kwargs["revision"] == "main"
            assert kwargs["torch_dtype"] is torch.float32
            return pipeline

    monkeypatch.setitem(
        sys.modules, "diffusers", SimpleNamespace(DiffusionPipeline=FakeDiffusionPipeline)
    )
    monkeypatch.setattr(
        cli_main,
        "inspect_model_metadata",
        lambda model_id, revision=None: {
            "sha": "abc123",
            "license": "apache-2.0",
        },
    )

    assert (
        main(
            [
                "quantize",
                "--model-id",
                "example/model",
                "--revision",
                "main",
                "--output",
                str(tmp_path),
                "--component",
                "transformer",
                "--target-policy",
                "generic_dit",
                "--weight-bits",
                "4",
                "--activation-bits",
                "4",
                "--block-size",
                "4",
                "--activation-kernel-backend",
                "cpu",
                "--device",
                "cpu",
                "--dtype",
                "float32",
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert output["artifact_dir"] == str(tmp_path)
    assert output["quantization_staging_mode"] == "streaming"
    assert output["synchronize_per_module"] is False
    assert output["device_transfer_seconds"] >= 0.0
    assert output["module_device_transfer_count"] >= 0
    assert output["source_linear_device_counts"]
    assert output["artifact_save_elapsed_seconds"] >= 0.0
    assert output["quantized_modules"] == ["transformer_blocks.0.attn.to_q"]
    assert (tmp_path / "model.safetensors").exists()
    assert (tmp_path / "orbitquant_manifest.json").exists()
    assert (tmp_path / "SHA256SUMS").exists()
    quantization_config = json.loads((tmp_path / "quantization_config.json").read_text())
    assert quantization_config["activation_kernel_backend"] == "cpu"


def test_cli_quantize_with_suite_uses_named_native_pipeline(monkeypatch, capsys, tmp_path):
    class TinyPipeline:
        def __init__(self):
            self.transformer = torch.nn.Module()
            self.transformer.transformer_blocks = torch.nn.ModuleList(
                [
                    torch.nn.ModuleDict(
                        {"attn": torch.nn.ModuleDict({"to_q": torch.nn.Linear(8, 8)})}
                    )
                ]
            )

        def to(self, device):
            return self

    class FakeFlux2KleinPipeline:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            assert model_id == "black-forest-labs/FLUX.2-klein-4B"
            assert kwargs["torch_dtype"] is torch.float32
            return TinyPipeline()

    class WrongDiffusionPipeline:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            raise AssertionError("suite-specific pipeline should be used")

    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(
            Flux2KleinPipeline=FakeFlux2KleinPipeline,
            DiffusionPipeline=WrongDiffusionPipeline,
        ),
    )
    monkeypatch.setattr(
        cli_main,
        "inspect_model_metadata",
        lambda model_id, revision=None: {
            "sha": "abc123",
            "license": "apache-2.0",
        },
    )

    assert (
        main(
            [
                "quantize",
                "--suite",
                "flux2-native",
                "--output",
                str(tmp_path),
                "--component",
                "transformer",
                "--weight-bits",
                "4",
                "--activation-bits",
                "4",
                "--block-size",
                "4",
                "--device",
                "cpu",
                "--dtype",
                "float32",
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    quantization_config = json.loads((tmp_path / "quantization_config.json").read_text())
    assert output["source_model_id"] == "black-forest-labs/FLUX.2-klein-4B"
    assert quantization_config["target_policy"] == "flux2"


def test_cli_inspect_policy_with_suite_writes_module_inventory(monkeypatch, capsys, tmp_path):
    calls = []

    class FakeFlux2Transformer2DModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = torch.nn.Module()
            self.double_stream_modulation_img = torch.nn.ModuleDict(
                {"linear": torch.nn.Linear(8, 16)}
            )
            self.transformer_blocks = torch.nn.ModuleList(
                [
                    torch.nn.ModuleDict(
                        {"attn": torch.nn.ModuleDict({"to_q": torch.nn.Linear(8, 8)})}
                    )
                ]
            )

        @classmethod
        def load_config(cls, model_id, **kwargs):
            assert model_id == "black-forest-labs/FLUX.2-klein-4B"
            assert kwargs["subfolder"] == "transformer"
            assert kwargs["local_files_only"] is False
            calls.append({"method": "load_config", "kwargs": kwargs})
            return {"hidden_size": 8}

        @classmethod
        def from_config(cls, config):
            assert config == {"hidden_size": 8}
            calls.append({"method": "from_config"})
            return cls()

    class WrongDiffusionPipeline:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            raise AssertionError("inspect-policy default must not load full pipeline weights")

    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(
            Flux2Transformer2DModel=FakeFlux2Transformer2DModel,
            DiffusionPipeline=WrongDiffusionPipeline,
        ),
    )
    output_path = tmp_path / "flux2-policy-inventory.json"

    assert (
        main(
            [
                "inspect-policy",
                "--suite",
                "flux2-native",
                "--dtype",
                "float32",
                "--output",
                str(output_path),
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    saved = json.loads(output_path.read_text())

    assert output["output"] == str(output_path)
    assert output["linear_module_count"] == 2
    assert output["quantized_module_count"] == 1
    assert output["adaln_module_count"] == 1
    assert output["skipped_module_count"] == 0
    assert "modules" not in output
    assert saved["source_model_id"] == "black-forest-labs/FLUX.2-klein-4B"
    assert saved["suite"] == "flux2-native"
    assert saved["component"] == "transformer"
    assert saved["load_mode"] == "config"
    assert saved["pipeline_class"] is None
    assert saved["component_class"] == "FakeFlux2Transformer2DModel"
    assert saved["target_policy"] == "flux2"
    assert saved["action_counts"] == {
        "orbitquant": 1,
        "adaln_int4_rtn": 1,
        "bf16_skip": 0,
    }
    assert saved["quantized_modules"] == ["transformer_blocks.0.attn.to_q"]
    assert saved["adaln_modules"] == ["double_stream_modulation_img.linear"]
    assert calls == [
        {
            "method": "load_config",
            "kwargs": {"subfolder": "transformer", "local_files_only": False},
        },
        {"method": "from_config"},
    ]


@pytest.mark.parametrize(
    ("suite", "transformer_class", "model_id", "target_policy", "module_family"),
    [
        (
            "flux2-native",
            "Flux2Transformer2DModel",
            "black-forest-labs/FLUX.2-klein-4B",
            "flux2",
            "flux",
        ),
        (
            "flux1-schnell-native",
            "FluxTransformer2DModel",
            "black-forest-labs/FLUX.1-schnell",
            "flux",
            "flux",
        ),
        (
            "z-image-native",
            "ZImageTransformer2DModel",
            "Tongyi-MAI/Z-Image-Turbo",
            "z_image",
            "z_image",
        ),
        (
            "wan-native",
            "WanTransformer3DModel",
            "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            "wan",
            "wan",
        ),
    ],
)
def test_cli_inspect_policy_config_mode_covers_all_native_target_suites(
    monkeypatch,
    capsys,
    tmp_path,
    suite,
    transformer_class,
    model_id,
    target_policy,
    module_family,
):
    calls = []

    def build_module(self):
        if module_family == "wan":
            self.blocks = torch.nn.ModuleList(
                [
                    torch.nn.ModuleDict(
                        {"attn1": torch.nn.ModuleDict({"to_q": torch.nn.Linear(8, 8)})}
                    )
                ]
            )
        elif module_family == "z_image":
            self.layers = torch.nn.ModuleList(
                [
                    torch.nn.ModuleDict(
                        {"attention": torch.nn.ModuleDict({"to_q": torch.nn.Linear(8, 8)})}
                    )
                ]
            )
        else:
            self.transformer_blocks = torch.nn.ModuleList(
                [
                    torch.nn.ModuleDict(
                        {"attn": torch.nn.ModuleDict({"to_q": torch.nn.Linear(8, 8)})}
                    )
                ]
            )

    def load_config(cls, requested_model_id, **kwargs):
        assert requested_model_id == model_id
        assert kwargs["subfolder"] == "transformer"
        calls.append(("load_config", transformer_class))
        return {"hidden_size": 8}

    def from_config(cls, config):
        assert config == {"hidden_size": 8}
        calls.append(("from_config", transformer_class))
        return cls()

    def init(self):
        torch.nn.Module.__init__(self)
        build_module(self)

    fake_transformer_cls = type(
        transformer_class,
        (torch.nn.Module,),
        {
            "__init__": init,
            "load_config": classmethod(load_config),
            "from_config": classmethod(from_config),
        },
    )

    class WrongDiffusionPipeline:
        @classmethod
        def from_pretrained(cls, requested_model_id, **kwargs):
            raise AssertionError("config inspect-policy must not load full pipeline weights")

    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(
            **{
                transformer_class: fake_transformer_cls,
                "DiffusionPipeline": WrongDiffusionPipeline,
            }
        ),
    )
    output_path = tmp_path / f"{suite}-policy-inventory.json"

    assert main(["inspect-policy", "--suite", suite, "--output", str(output_path)]) == 0

    output = json.loads(capsys.readouterr().out)
    saved = json.loads(output_path.read_text())

    assert output["output"] == str(output_path)
    assert saved["suite"] == suite
    assert saved["source_model_id"] == model_id
    assert saved["component"] == "transformer"
    assert saved["load_mode"] == "config"
    assert saved["pipeline_class"] is None
    assert saved["component_class"] == transformer_class
    assert saved["target_policy"] == target_policy
    assert saved["action_counts"]["orbitquant"] == 1
    assert saved["quantized_modules"]
    assert calls == [
        ("load_config", transformer_class),
        ("from_config", transformer_class),
    ]


def test_cli_validate_artifact_reports_valid_component_artifact(capsys, tmp_path):
    model = torch.nn.Module()
    model.transformer_blocks = torch.nn.ModuleList(
        [torch.nn.ModuleDict({"attn": torch.nn.ModuleDict({"to_q": torch.nn.Linear(8, 8)})})]
    )
    config = OrbitQuantConfig(block_size=4, target_policy="generic_dit")
    summary = quantize_linear_modules(model, config)
    save_orbitquant_artifact(
        model,
        tmp_path,
        config=config,
        source_model_id="example/model",
        source_revision="abc123",
        source_license="apache-2.0",
        summary=summary,
    )

    assert main(["validate-artifact", "--artifact", str(tmp_path)]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["valid"] is True
    assert output["source_model_id"] == "example/model"
    assert output["tensor_count"] > 0
    assert output["quantized_module_count"] == 1


def test_cli_validate_artifact_checks_policy_inventory(capsys, tmp_path):
    model = torch.nn.Module()
    model.transformer_blocks = torch.nn.ModuleList(
        [torch.nn.ModuleDict({"attn": torch.nn.ModuleDict({"to_q": torch.nn.Linear(8, 8)})})]
    )
    config = OrbitQuantConfig(block_size=4, target_policy="generic_dit")
    summary = quantize_linear_modules(model, config)
    save_orbitquant_artifact(
        model,
        tmp_path,
        config=config,
        source_model_id="example/model",
        source_revision="abc123",
        source_license="apache-2.0",
        summary=summary,
    )
    inventory_path = tmp_path / "policy-inventory.json"
    inventory_path.write_text(
        json.dumps(
            {
                "source_model_id": "example/model",
                "target_policy": "generic_dit",
                "component": "transformer",
                "load_mode": "config",
                "linear_module_count": 1,
                "action_counts": {
                    "orbitquant": 1,
                    "adaln_int4_rtn": 0,
                    "bf16_skip": 0,
                },
                "quantized_modules": summary.quantized_modules,
                "adaln_modules": summary.adaln_modules,
                "skipped_modules": summary.skipped_modules,
            }
        )
        + "\n"
    )

    assert (
        main(
            [
                "validate-artifact",
                "--artifact",
                str(tmp_path),
                "--policy-inventory",
                str(inventory_path),
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    policy_validation = output["policy_inventory_validation"]
    assert policy_validation["valid"] is True
    assert policy_validation["quantized_module_count"] == 1
    assert policy_validation["inventory_path"] == str(inventory_path)


def test_cli_validate_artifact_rejects_policy_inventory_mismatch(tmp_path):
    model = torch.nn.Module()
    model.transformer_blocks = torch.nn.ModuleList(
        [torch.nn.ModuleDict({"attn": torch.nn.ModuleDict({"to_q": torch.nn.Linear(8, 8)})})]
    )
    config = OrbitQuantConfig(block_size=4, target_policy="generic_dit")
    summary = quantize_linear_modules(model, config)
    save_orbitquant_artifact(
        model,
        tmp_path,
        config=config,
        source_model_id="example/model",
        source_revision="abc123",
        source_license="apache-2.0",
        summary=summary,
    )
    inventory_path = tmp_path / "policy-inventory.json"
    inventory_path.write_text(
        json.dumps(
            {
                "source_model_id": "example/model",
                "target_policy": "generic_dit",
                "quantized_modules": ["transformer_blocks.0.attn.to_k"],
                "adaln_modules": [],
                "skipped_modules": [],
            }
        )
        + "\n"
    )

    with pytest.raises(RuntimeError, match="policy inventory mismatch"):
        main(
            [
                "validate-artifact",
                "--artifact",
                str(tmp_path),
                "--policy-inventory",
                str(inventory_path),
            ]
        )


def test_cli_validate_artifact_rejects_policy_inventory_component_mismatch(tmp_path):
    model = torch.nn.Module()
    model.transformer_blocks = torch.nn.ModuleList(
        [torch.nn.ModuleDict({"attn": torch.nn.ModuleDict({"to_q": torch.nn.Linear(8, 8)})})]
    )
    config = OrbitQuantConfig(block_size=4, target_policy="generic_dit")
    summary = quantize_linear_modules(model, config)
    save_orbitquant_artifact(
        model,
        tmp_path,
        config=config,
        source_model_id="example/model",
        source_revision="abc123",
        source_license="apache-2.0",
        summary=summary,
        component="transformer",
    )
    inventory_path = tmp_path / "policy-inventory.json"
    inventory_path.write_text(
        json.dumps(
            {
                "source_model_id": "example/model",
                "target_policy": "generic_dit",
                "component": "denoiser",
                "quantized_modules": summary.quantized_modules,
                "adaln_modules": summary.adaln_modules,
                "skipped_modules": summary.skipped_modules,
            }
        )
        + "\n"
    )

    with pytest.raises(RuntimeError, match="component"):
        main(
            [
                "validate-artifact",
                "--artifact",
                str(tmp_path),
                "--policy-inventory",
                str(inventory_path),
            ]
        )


def test_cli_validate_artifact_rejects_manifest_policy_config_mismatch(tmp_path):
    model = torch.nn.Module()
    model.transformer_blocks = torch.nn.ModuleList(
        [torch.nn.ModuleDict({"attn": torch.nn.ModuleDict({"to_q": torch.nn.Linear(8, 8)})})]
    )
    config = OrbitQuantConfig(block_size=4, target_policy="generic_dit")
    summary = quantize_linear_modules(model, config)
    save_orbitquant_artifact(
        model,
        tmp_path,
        config=config,
        source_model_id="example/model",
        source_revision="abc123",
        source_license="apache-2.0",
        summary=summary,
    )
    manifest_path = tmp_path / "orbitquant_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["target_policy"] = "flux"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    write_sha256sums_from_manifest(tmp_path, manifest["checksums"])
    inventory_path = tmp_path / "policy-inventory.json"
    inventory_path.write_text(
        json.dumps(
            {
                "source_model_id": "example/model",
                "target_policy": "flux",
                "component": "transformer",
                "quantized_modules": summary.quantized_modules,
                "adaln_modules": summary.adaln_modules,
                "skipped_modules": summary.skipped_modules,
            }
        )
        + "\n"
    )

    with pytest.raises(RuntimeError, match="manifest_target_policy"):
        main(
            [
                "validate-artifact",
                "--artifact",
                str(tmp_path),
                "--policy-inventory",
                str(inventory_path),
            ]
        )


def test_cli_repair_artifact_metadata_wires_provenance_options(
    capsys, tmp_path, monkeypatch
):
    seen = {}

    def fake_repair_artifact_metadata(artifact, **kwargs):
        seen["artifact"] = artifact
        seen["kwargs"] = kwargs
        return {
            "artifact_dir": artifact,
            "updated": {
                "quantization_device": kwargs["quantization_device"],
                "weight_quantization_backend": kwargs["weight_quantization_backend"],
                "quantization_staging_mode": kwargs["quantization_staging_mode"],
            },
        }

    monkeypatch.setattr(cli_main, "repair_artifact_metadata", fake_repair_artifact_metadata)

    assert (
        main(
            [
                "repair-artifact-metadata",
                "--artifact",
                str(tmp_path),
                "--quantization-device",
                "cuda",
                "--weight-quantization-backend",
                "triton_cuda",
                "--quantization-staging-mode",
                "component",
                "--skip-tensor-validation",
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert output["updated"]["quantization_device"] == "cuda"
    assert output["updated"]["weight_quantization_backend"] == "triton_cuda"
    assert output["updated"]["quantization_staging_mode"] == "component"
    assert seen == {
        "artifact": str(tmp_path),
        "kwargs": {
            "quantization_device": "cuda",
            "weight_quantization_backend": "triton_cuda",
            "quantization_staging_mode": "component",
            "validate_tensors": False,
        },
    }


def test_cli_upload_artifact_wires_validation_and_hf_options(capsys, tmp_path, monkeypatch):
    seen = {}

    def fake_upload_artifact(artifact, **kwargs):
        seen["artifact"] = artifact
        seen["kwargs"] = kwargs
        return {
            "artifact_dir": artifact,
            "repo_id": kwargs["repo_id"],
            "private": kwargs["private"],
            "dry_run": kwargs["dry_run"],
            "validation": {"valid": True},
        }

    monkeypatch.setattr(cli_main, "upload_orbitquant_artifact", fake_upload_artifact)

    assert (
        main(
            [
                "upload-artifact",
                "--artifact",
                str(tmp_path),
                "--repo-id",
                "WaveCut/example-orbitquant",
                "--revision",
                "main",
                "--commit-message",
                "upload artifact",
                "--public",
                "--no-create-repo",
                "--replace-repo-files",
                "--skip-tensor-validation",
                "--upload-profile",
                "compact",
                "--report-dir",
                "/tmp/orbitquant-report",
                "--staging-dir",
                "/tmp/orbitquant-stage",
                "--dry-run",
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert output["repo_id"] == "WaveCut/example-orbitquant"
    assert output["private"] is False
    assert output["dry_run"] is True
    assert seen == {
        "artifact": str(tmp_path),
        "kwargs": {
            "repo_id": "WaveCut/example-orbitquant",
            "private": False,
            "create_repo": False,
            "revision": "main",
            "commit_message": "upload artifact",
            "replace_repo_files": True,
            "validate_tensors": False,
            "upload_profile": "compact",
            "report_dirs": ["/tmp/orbitquant-report"],
            "staging_dir": "/tmp/orbitquant-stage",
            "dry_run": True,
        },
    }


def test_cli_upload_artifact_defaults_to_compact_profile(capsys, tmp_path, monkeypatch):
    seen = {}

    def fake_upload_artifact(artifact, **kwargs):
        seen["artifact"] = artifact
        seen["kwargs"] = kwargs
        return {
            "artifact_dir": artifact,
            "repo_id": kwargs["repo_id"],
            "private": kwargs["private"],
            "dry_run": kwargs["dry_run"],
            "validation": {"valid": True},
        }

    monkeypatch.setattr(cli_main, "upload_orbitquant_artifact", fake_upload_artifact)

    assert (
        main(
            [
                "upload-artifact",
                "--artifact",
                str(tmp_path),
                "--repo-id",
                "WaveCut/example-orbitquant",
                "--dry-run",
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert output["repo_id"] == "WaveCut/example-orbitquant"
    assert seen["artifact"] == str(tmp_path)
    assert seen["kwargs"]["upload_profile"] == "compact"
    assert seen["kwargs"]["replace_repo_files"] is True
    assert seen["kwargs"]["report_dirs"] is None
    assert seen["kwargs"]["staging_dir"] is None


def test_cli_upload_artifact_can_disable_default_remote_file_replacement(
    capsys, tmp_path, monkeypatch
):
    seen = {}

    def fake_upload_artifact(artifact, **kwargs):
        seen["artifact"] = artifact
        seen["kwargs"] = kwargs
        return {
            "artifact_dir": artifact,
            "repo_id": kwargs["repo_id"],
            "private": kwargs["private"],
            "dry_run": kwargs["dry_run"],
            "validation": {"valid": True},
        }

    monkeypatch.setattr(cli_main, "upload_orbitquant_artifact", fake_upload_artifact)

    assert (
        main(
            [
                "upload-artifact",
                "--artifact",
                str(tmp_path),
                "--repo-id",
                "WaveCut/example-orbitquant",
                "--no-replace-repo-files",
                "--dry-run",
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert output["repo_id"] == "WaveCut/example-orbitquant"
    assert seen["kwargs"]["replace_repo_files"] is False


def test_cli_repair_hf_artifact_metadata_wires_single_repo_options(capsys, monkeypatch):
    seen = {}

    def fake_repair_hf_artifact_metadata(**kwargs):
        seen.update(kwargs)
        return {
            "repo_id": kwargs["repo_id"],
            "dry_run": kwargs["dry_run"],
            "changed_files": ["orbitquant_manifest.json"],
        }

    monkeypatch.setattr(
        cli_main, "repair_hf_artifact_metadata", fake_repair_hf_artifact_metadata
    )

    assert (
        main(
            [
                "repair-hf-artifact-metadata",
                "--repo-id",
                "WaveCut/example-orbitquant",
                "--revision",
                "main",
                "--commit-message",
                "repair metadata",
                "--quantization-device",
                "cuda",
                "--weight-quantization-backend",
                "triton_cuda",
                "--quantization-staging-mode",
                "component",
                "--dry-run",
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert output["repo_id"] == "WaveCut/example-orbitquant"
    assert output["dry_run"] is True
    assert seen == {
        "repo_id": "WaveCut/example-orbitquant",
        "quantization_device": "cuda",
        "weight_quantization_backend": "triton_cuda",
        "quantization_staging_mode": "component",
        "revision": "main",
        "commit_message": "repair metadata",
        "dry_run": True,
    }


def test_cli_repair_hf_native_smoke_proof_wires_single_repo_options(
    capsys, monkeypatch
):
    seen = {}

    def fake_repair_hf_native_smoke_proof(**kwargs):
        seen.update(kwargs)
        return {
            "repo_id": kwargs["repo_id"],
            "suite": kwargs["suite"].name,
            "dry_run": kwargs["dry_run"],
            "changed_files": ["benchmark/summary.json"],
        }

    monkeypatch.setattr(
        cli_main, "repair_hf_native_smoke_proof", fake_repair_hf_native_smoke_proof
    )

    assert (
        main(
            [
                "repair-hf-native-smoke-proof",
                "--repo-id",
                "WaveCut/example-orbitquant",
                "--suite",
                "flux2-native",
                "--revision",
                "main",
                "--commit-message",
                "repair native smoke proof",
                "--dry-run",
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert output["repo_id"] == "WaveCut/example-orbitquant"
    assert output["suite"] == "flux2-native"
    assert output["dry_run"] is True
    assert seen["repo_id"] == "WaveCut/example-orbitquant"
    assert seen["suite"].name == "flux2-native"
    assert seen["revision"] == "main"
    assert seen["commit_message"] == "repair native smoke proof"
    assert seen["dry_run"] is True


def test_cli_cleanup_hf_artifact_reports_wires_single_repo_options(
    capsys, monkeypatch
):
    seen = {}

    def fake_cleanup_hf_artifact_reports(**kwargs):
        seen.update(kwargs)
        return {
            "repo_id": kwargs["repo_id"],
            "revision": kwargs["revision"],
            "dry_run": kwargs["dry_run"],
            "report_file_count": 2,
            "promoted_assets": ["assets/image_generation_comparison_matrix.webp"],
            "delete_paths": ["reports"],
        }

    monkeypatch.setattr(
        cli_main, "cleanup_hf_artifact_reports", fake_cleanup_hf_artifact_reports
    )

    assert (
        main(
            [
                "cleanup-hf-artifact-reports",
                "--repo-id",
                "WaveCut/example-orbitquant",
                "--revision",
                "main",
                "--commit-message",
                "cleanup reports",
                "--dry-run",
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert output["repo_id"] == "WaveCut/example-orbitquant"
    assert output["promoted_assets"] == ["assets/image_generation_comparison_matrix.webp"]
    assert output["delete_paths"] == ["reports"]
    assert seen == {
        "repo_id": "WaveCut/example-orbitquant",
        "revision": "main",
        "commit_message": "cleanup reports",
        "dry_run": True,
    }


def test_cli_audit_hf_artifacts_writes_json_report(capsys, tmp_path, monkeypatch):
    seen = {}

    def fake_audit_hf_artifacts(*, namespace, suites, revision, policy_inventory_root):
        seen["namespace"] = namespace
        seen["suites"] = [suite.name for suite in suites]
        seen["revision"] = revision
        seen["policy_inventory_root"] = policy_inventory_root
        return {
            "namespace": namespace,
            "policy_inventory_root": policy_inventory_root,
            "repo_count": 1,
            "existing_count": 1,
            "artifact_ready_count": 1,
            "native_smoke_ready_count": 1,
            "release_eval_ready_count": 0,
            "rows": [{"repo_id": "WaveCut/example"}],
        }

    monkeypatch.setattr(cli_main, "audit_hf_artifact_repos", fake_audit_hf_artifacts)
    output_path = tmp_path / "reports" / "native" / "audit.json"
    markdown_output_path = tmp_path / "reports" / "native" / "audit.md"

    assert (
        main(
            [
                "audit-hf-artifacts",
                "--namespace",
                "WaveCut",
                "--suite",
                "flux2-native",
                "--revision",
                "main",
                "--policy-inventory-root",
                str(tmp_path / "inventories"),
                "--output",
                str(output_path),
                "--markdown-output",
                str(markdown_output_path),
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    written = json.loads(output_path.read_text())
    markdown = markdown_output_path.read_text()
    assert output_path.parent.is_dir()
    assert output == written
    assert output["repo_count"] == 1
    assert "# OrbitQuant HF Artifact Audit" in markdown
    assert "| `WaveCut/example` |" in markdown
    assert seen == {
        "namespace": "WaveCut",
        "suites": ["flux2-native"],
        "revision": "main",
        "policy_inventory_root": str(tmp_path / "inventories"),
    }


def test_cli_fetch_hf_artifacts_wires_suite_and_download_options(
    capsys, tmp_path, monkeypatch
):
    seen = {}

    def fake_fetch_hf_artifacts(**kwargs):
        seen.update(kwargs)
        if kwargs["stage_logger"] is not None:
            kwargs["stage_logger"]("START", "example fetch")
        return {
            "namespace": kwargs["namespace"],
            "output_root": str(kwargs["output_root"]),
            "repo_count": 1,
            "downloaded_count": 0,
            "skipped_existing_count": 0,
            "dry_run": kwargs["dry_run"],
            "rows": [{"repo_id": "WaveCut/example"}],
        }

    monkeypatch.setattr(cli_main, "fetch_hf_artifacts", fake_fetch_hf_artifacts)

    assert (
        main(
            [
                "fetch-hf-artifacts",
                "--namespace",
                "WaveCut",
                "--suite",
                "flux1-schnell-native",
                "--output-root",
                str(tmp_path / "artifacts"),
                "--revision",
                "main",
                "--no-resume",
                "--force-download",
                "--local-files-only",
                "--validate-checksums",
                "--validate-tensors",
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert output["repo_count"] == 1
    assert "example fetch" in captured.err
    assert seen["namespace"] == "WaveCut"
    assert [suite.name for suite in seen["suites"]] == ["flux1-schnell-native"]
    assert seen["output_root"] == str(tmp_path / "artifacts")
    assert seen["revision"] == "main"
    assert seen["resume"] is False
    assert seen["force_download"] is True
    assert seen["local_files_only"] is True
    assert seen["validate_checksums"] is True
    assert seen["validate_tensors"] is True
    assert seen["dry_run"] is False
    assert seen["stage_logger"] is not None


def test_cli_fetch_hf_artifacts_dry_run_suppresses_stage_log(capsys, monkeypatch):
    seen = {}

    def fake_fetch_hf_artifacts(**kwargs):
        seen.update(kwargs)
        return {
            "namespace": kwargs["namespace"],
            "output_root": str(kwargs["output_root"]),
            "repo_count": 0,
            "downloaded_count": 0,
            "skipped_existing_count": 0,
            "dry_run": kwargs["dry_run"],
            "rows": [],
        }

    monkeypatch.setattr(cli_main, "fetch_hf_artifacts", fake_fetch_hf_artifacts)

    assert main(["fetch-hf-artifacts", "--dry-run"]) == 0

    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert output["dry_run"] is True
    assert captured.err == ""
    assert seen["dry_run"] is True
    assert seen["stage_logger"] is None
