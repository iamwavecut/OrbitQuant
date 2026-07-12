import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

import orbitquant.cli.main as cli_main
from orbitquant.artifacts import save_orbitquant_artifact
from orbitquant.cli.main import main
from orbitquant.config import OrbitQuantConfig
from orbitquant.eval.prompts import default_prompt_payload, select_prompt_record
from orbitquant.modeling import quantize_linear_modules


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

    report_dir = tmp_path / "reports"
    assert main(["report", "--artifact", str(tmp_path), "--output", str(report_dir)]) == 0

    report_output = json.loads(capsys.readouterr().out)
    missing_table = (report_dir / "tables" / "missing_required_metrics.csv").read_text()
    assert report_output["missing_required_metric_count"] == 11
    assert "orbitquant,flux1-schnell-native,geneval_overall" not in missing_table
    assert "orbitquant,flux1-schnell-native,geneval_per_task_single_object" not in (missing_table)
    assert "orbitquant,flux1-schnell-native,geneval_per_task_counting" not in (missing_table)
    assert "original,flux1-schnell-native,geneval_overall" in missing_table
    assert "orbitquant,flux1-schnell-native,geneval_per_task_color_attr" in missing_table


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
                "--enable-model-cpu-offload",
                "--dry-run",
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert output["model_id"] == "black-forest-labs/FLUX.2-klein-4B"
    assert output["pipeline_kwargs"]["height"] == 1024
    assert output["pipeline_kwargs"]["width"] == 1024
    assert output["enable_model_cpu_offload"] is True


@pytest.mark.parametrize("runtime_mode", ["triton_packed_matmul", "native_packed_matmul"])
def test_cli_generate_dry_run_prints_quantized_native_request(capsys, tmp_path, runtime_mode):
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
                "--runtime-mode",
                runtime_mode,
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
    assert f'"runtime_mode": "{runtime_mode}"' in output
    assert '"target_policy": "wan"' in output


def test_cli_generate_with_packed_runtime_on_the_fly_quantization_skips_dequant_prewarm(
    monkeypatch,
    capsys,
    tmp_path,
):
    class TinyPipeline:
        def __init__(self):
            self.transformer = torch.nn.Module()

        def to(self, device):
            return self

        def __call__(self, **kwargs):
            return SimpleNamespace(images=[Image.new("RGB", (16, 16), "blue")])

    def fake_apply_quantization(pipeline, suite, config):
        assert config.runtime_mode == "triton_packed_matmul"
        return SimpleNamespace(
            quantized_modules=["transformer_blocks.0.attn.to_q"],
            adaln_modules=[],
            skipped_modules=[],
        )

    def fail_prewarm(*args, **kwargs):
        raise AssertionError("packed runtime must not materialize dequant prewarm")

    monkeypatch.setattr(cli_main, "load_pipeline_for_suite", lambda *args, **kwargs: TinyPipeline())
    monkeypatch.setattr(cli_main, "apply_quantization_to_pipeline", fake_apply_quantization)
    monkeypatch.setattr(cli_main, "_prewarm_pipeline_component", fail_prewarm)

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
                "--bit-setting",
                "W4A4",
                "--runtime-mode",
                "triton_packed_matmul",
                "--device",
                "cpu",
                "--dtype",
                "float32",
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert output["output_path"].endswith("flux2-native_seed0_W4A4.png")


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
    expected_prompt = select_prompt_record(
        default_prompt_payload("flux2"), prompt_id="english-text-rendering"
    )["prompt"]
    assert output["pipeline_kwargs"]["prompt"] == expected_prompt


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


def test_cli_generate_with_packed_runtime_artifact_skips_dequant_prewarm(
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
            return SimpleNamespace(images=[Image.new("RGB", (16, 16), "blue")])

    source = TinyPipeline()
    config = OrbitQuantConfig(
        block_size=4,
        target_policy="generic_dit",
        runtime_mode="triton_packed_matmul",
    )
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

    def fail_prewarm(*args, **kwargs):
        raise AssertionError("packed runtime must not materialize dequant prewarm")

    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(Flux2KleinPipeline=FakeDiffusionPipeline),
    )
    monkeypatch.setattr(cli_main, "_prewarm_pipeline_component", fail_prewarm)

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
    assert output["output_path"].endswith("flux2-native_seed3_W4A4.png")


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
    assert not (tmp_path / "benchmark" / "orbitquant.metrics.jsonl").exists()


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

    export_calls = []

    def fake_export_to_video(frames, path, **kwargs):
        export_calls.append(kwargs)
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
    metrics_record = json.loads((tmp_path / "benchmark" / "orbitquant.metrics.jsonl").read_text())
    metadata = json.loads((tmp_path / "assets" / "wan-native_seed2_W4A4.mp4.json").read_text())
    assert output["output_path"].endswith("wan-native_seed2_W4A4.mp4")
    assert export_calls == [{"fps": 16}]
    assert metadata["export_fps"] == 16
    assert metrics_record["metadata"]["export_fps"] == 16
    assert "assets/wan-native_seed2_W4A4.mp4" in manifest["checksums"]
    assert "assets/wan-native_seed2_W4A4.mp4.json" in manifest["checksums"]
    assert "assets/wan-native_seed2_W4A4_contact_sheet.webp" in manifest["checksums"]
