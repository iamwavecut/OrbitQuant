import json
import sys
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

import orbitquant.cli.main as cli_main
from orbitquant.artifacts import save_orbitquant_artifact
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
