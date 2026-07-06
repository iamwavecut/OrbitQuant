import json
import sys
from types import SimpleNamespace

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


def test_cli_generate_requires_prompt_and_output():
    try:
        main(["generate", "--suite", "flux2-native"])
    except SystemExit as exc:
        assert exc.code != 0
    else:
        raise AssertionError("generate accepted missing prompt/output arguments")


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
        sys.modules, "diffusers", SimpleNamespace(DiffusionPipeline=FakeDiffusionPipeline)
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
    assert record["metadata"]["output_path"] == output["output_path"]
    assert restored.device == "cpu"


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
