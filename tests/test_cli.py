import json
import sys
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

import orbitquant.cli.main as cli_main
from orbitquant.artifacts import save_orbitquant_artifact, validate_orbitquant_artifact
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
                "optimized": False,
                "weight_dequant_optimized": False,
                "weight_pack_optimized": False,
                "weight_quant_optimized": False,
                "adaln_quant_optimized": False,
                "adaln_dequant_optimized": False,
            },
            "mps": {
                "implementation": "metal_codebook_rescale",
                "optimized_stage": "codebook_lookup_rescale",
                "weight_dequant_optimized": True,
                "weight_pack_optimized": False,
                "weight_quant_optimized": False,
                "adaln_quant_optimized": False,
                "adaln_dequant_optimized": False,
                "full_fusion": False,
            },
            "triton_cuda": {
                "optimized_stage": (
                    "codebook_lookup_rescale,packed_weight_dequant,"
                    "lowbit_pack,weight_rotation_fwht_quant_pack,"
                    "adaln_rtn_quant_pack,adaln_rtn_dequant"
                ),
                "weight_dequant_optimized": True,
                "weight_pack_optimized": True,
                "weight_quant_optimized": True,
                "adaln_quant_optimized": True,
                "adaln_dequant_optimized": True,
                "full_fusion": False,
            },
        },
    )

    assert main(["kernel-info"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["cpu"]["available"] is True
    assert payload["cpu"]["optimized"] is False
    assert payload["cpu"]["weight_dequant_optimized"] is False
    assert payload["cpu"]["weight_pack_optimized"] is False
    assert payload["cpu"]["weight_quant_optimized"] is False
    assert payload["cpu"]["adaln_quant_optimized"] is False
    assert payload["cpu"]["adaln_dequant_optimized"] is False
    assert payload["mps"]["implementation"] == "metal_codebook_rescale"
    assert payload["mps"]["optimized_stage"] == "codebook_lookup_rescale"
    assert payload["mps"]["weight_dequant_optimized"] is True
    assert payload["mps"]["weight_pack_optimized"] is False
    assert payload["mps"]["weight_quant_optimized"] is False
    assert payload["mps"]["adaln_quant_optimized"] is False
    assert payload["mps"]["adaln_dequant_optimized"] is False
    assert payload["mps"]["full_fusion"] is False
    assert payload["triton_cuda"]["optimized_stage"] == (
        "codebook_lookup_rescale,packed_weight_dequant,"
        "lowbit_pack,weight_rotation_fwht_quant_pack,"
        "adaln_rtn_quant_pack,adaln_rtn_dequant"
    )
    assert payload["triton_cuda"]["weight_dequant_optimized"] is True
    assert payload["triton_cuda"]["weight_pack_optimized"] is True
    assert payload["triton_cuda"]["weight_quant_optimized"] is True
    assert payload["triton_cuda"]["adaln_quant_optimized"] is True
    assert payload["triton_cuda"]["adaln_dequant_optimized"] is True
    assert payload["triton_cuda"]["full_fusion"] is False


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
    assert "range" not in script.lower()


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
    assert "hf auth whoami" in script
    assert "hf env" in script
    assert "torch.cuda.is_available()" in script
    assert "shutil.disk_usage" in script
    assert "hf models info Wan-AI/Wan2.1-T2V-1.3B-Diffusers --format json >/dev/null" in script
    assert "orbitquant kernel-info" in script
    assert script.count("orbitquant kernel-bench") == 2
    assert "--tokens 256 --in-features 3072 --out-features 3072" in script
    assert script.count("orbitquant quantize") == 2
    assert "--suite wan-native" in script
    assert "--weight-bits 4 --activation-bits 6" in script
    assert "--weight-bits 4 --activation-bits 4" in script
    assert "--activation-kernel-backend triton_cuda" in script
    assert "--staging-mode component" in script
    assert script.count("orbitquant generate-pack") == 4
    assert "--split original" in script
    assert "--split orbitquant" in script
    assert "--seeds 0,1" in script
    assert "--prompt-limit 1" in script
    assert script.count("orbitquant validate-artifact") == 4
    assert script.count("orbitquant report") == 1
    assert f"--artifact {tmp_path / 'artifacts' / 'wan-native-w4a6'}" in script
    assert f"--artifact {tmp_path / 'artifacts' / 'wan-native-w4a4'}" in script
    assert "--output reports/native" in script
    assert "range" not in script.lower()


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
    assert f"if orbitquant validate-artifact --artifact {artifact_dir}" in script
    assert f"echo 'Skipping existing valid artifact: {artifact_dir}'" in script
    assert "else\norbitquant quantize --suite flux2-native" in script
    assert "\nfi\norbitquant validate-artifact --artifact" in script
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
            "dry_run": True,
        },
    }


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


def test_cli_audit_hf_artifacts_writes_json_report(capsys, tmp_path, monkeypatch):
    seen = {}

    def fake_audit_hf_artifacts(*, namespace, suites, revision):
        seen["namespace"] = namespace
        seen["suites"] = [suite.name for suite in suites]
        seen["revision"] = revision
        return {
            "namespace": namespace,
            "repo_count": 1,
            "existing_count": 1,
            "artifact_ready_count": 1,
            "native_smoke_ready_count": 1,
            "release_eval_ready_count": 0,
            "rows": [{"repo_id": "WaveCut/example"}],
        }

    monkeypatch.setattr(cli_main, "audit_hf_artifact_repos", fake_audit_hf_artifacts)
    output_path = tmp_path / "reports" / "native" / "audit.json"

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
                "--output",
                str(output_path),
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    written = json.loads(output_path.read_text())
    assert output_path.parent.is_dir()
    assert output == written
    assert output["repo_count"] == 1
    assert seen == {
        "namespace": "WaveCut",
        "suites": ["flux2-native"],
        "revision": "main",
    }
