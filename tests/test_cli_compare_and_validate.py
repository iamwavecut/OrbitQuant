import json
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

import orbitquant.cli.main as cli_main
from orbitquant.artifacts import save_orbitquant_artifact
from orbitquant.cli.main import main
from orbitquant.config import OrbitQuantConfig
from orbitquant.eval import get_native_suite
from orbitquant.modeling import quantize_linear_modules


def test_cli_compare_native_dry_run_prints_artifact_and_native_settings(capsys, tmp_path):
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

    assert (
        main(
            [
                "compare-native",
                "--suite",
                "flux2-native",
                "--artifact",
                str(tmp_path),
                "--source-model",
                str(tmp_path / "source-model"),
                "--prompt",
                "A native prompt",
                "--output",
                str(tmp_path / "comparison"),
                "--seed",
                "11",
                "--device",
                "cpu",
                "--dtype",
                "float32",
                "--runtime-mode",
                "triton_packed_matmul",
                "--activation-kernel-backend",
                "cpu",
                "--dry-run",
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert output["model_id"] == "example/artifact-model"
    assert output["source_model"] == str(tmp_path / "source-model")
    assert output["bit_setting"] == "W4A4"
    assert output["runtime_mode"] == "triton_packed_matmul"
    assert output["activation_kernel_backend"] == "cpu"
    assert output["pipeline_kwargs"]["height"] == 1024
    assert output["pipeline_kwargs"]["width"] == 1024


def test_cli_compare_native_runs_original_and_quantized_side_by_side(
    monkeypatch,
    capsys,
    tmp_path,
):
    monkeypatch.setattr(
        "orbitquant.layers._native_cpu_packed_matmul_load_error",
        lambda: RuntimeError("reference-path test"),
    )

    class TinyPipeline:
        def __init__(self, color="red"):
            self.color = color
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
            layer = self.transformer.transformer_blocks[0]["attn"]["to_q"]
            layer(torch.zeros(1, 1, 8))
            return SimpleNamespace(images=[Image.new("RGB", (16, 16), self.color)])

    source = TinyPipeline()
    config = OrbitQuantConfig(block_size=4, target_policy="generic_dit")
    summary = quantize_linear_modules(source.transformer, config)
    save_orbitquant_artifact(
        source.transformer,
        tmp_path / "artifact",
        config=config,
        source_model_id="example/artifact-model",
        source_revision="abc123",
        source_license="apache-2.0",
        summary=summary,
    )

    pipelines = iter([TinyPipeline("red"), TinyPipeline("blue")])

    def fake_load_pipeline_for_suite(suite, *, model_id=None, **kwargs):
        assert suite.name == "flux2-native"
        assert model_id == "example/artifact-model"
        assert kwargs["torch_dtype"] is torch.float32
        return next(pipelines)

    validation_calls = []

    def fake_validate_compare_native_bundle(bundle_dir):
        validation_calls.append(bundle_dir)
        return {
            "valid": True,
            "orbitquant_runtime": {
                "runtime_mode_counts": {
                    "dequant_bf16": 1,
                },
            },
        }

    monkeypatch.setattr(cli_main, "load_pipeline_for_suite", fake_load_pipeline_for_suite)
    monkeypatch.setattr(
        cli_main,
        "_validate_compare_native_bundle",
        fake_validate_compare_native_bundle,
    )

    assert (
        main(
            [
                "compare-native",
                "--suite",
                "flux2-native",
                "--artifact",
                str(tmp_path / "artifact"),
                "--prompt",
                "A native prompt",
                "--output",
                str(tmp_path / "comparison"),
                "--seed",
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
    summary = json.loads((tmp_path / "comparison" / "summary.json").read_text())
    comparison_path = (
        tmp_path / "comparison" / "flux2-native_seed4_W4A4_original_vs_orbitquant.webp"
    )
    assert output["summary_path"] == str(tmp_path / "comparison" / "summary.json")
    assert output["comparison_path"] == str(comparison_path)
    assert output["validation"]["valid"] is True
    assert output["validation"]["orbitquant_runtime"]["runtime_mode_counts"] == {"dequant_bf16": 1}
    assert validation_calls == [tmp_path / "comparison"]
    assert comparison_path.is_file()
    assert summary["original"]["output_path"].endswith("flux2-native_seed4_original.png")
    assert summary["orbitquant"]["output_path"].endswith("flux2-native_seed4_W4A4.png")
    assert summary["model_id"] == "example/artifact-model"
    assert summary["source_model"] == "example/artifact-model"
    assert summary["runtime_mode"] == "auto_fused"
    assert summary["enable_model_cpu_offload"] is False
    assert summary["available_backends"]["cpu"] is True
    assert summary["orbitquant"]["runtime"] == {
        "orbitquant_linear_count": 1,
        "executed_module_count": 1,
        "runtime_mode_counts": {"dequant_bf16": 1},
        "activation_kernel_backend_counts": {"cpu": 1},
        "forward_device_type_counts": {"cpu": 1},
        "unexecuted_module_count": 0,
        "unexecuted_module_sample": [],
    }
    assert not any((tmp_path / "artifact" / "assets").iterdir())


def test_cli_compare_native_can_skip_post_generation_validation(
    monkeypatch,
    capsys,
    tmp_path,
):
    class TinyPipeline:
        def __init__(self, color="red"):
            self.color = color
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
            layer = self.transformer.transformer_blocks[0]["attn"]["to_q"]
            layer(torch.zeros(1, 1, 8))
            return SimpleNamespace(images=[Image.new("RGB", (16, 16), self.color)])

    source = TinyPipeline()
    config = OrbitQuantConfig(block_size=4, target_policy="generic_dit")
    summary = quantize_linear_modules(source.transformer, config)
    save_orbitquant_artifact(
        source.transformer,
        tmp_path / "artifact",
        config=config,
        source_model_id="example/artifact-model",
        source_revision="abc123",
        source_license="apache-2.0",
        summary=summary,
    )
    pipelines = iter([TinyPipeline("red"), TinyPipeline("blue")])

    monkeypatch.setattr(
        cli_main,
        "load_pipeline_for_suite",
        lambda *args, **kwargs: next(pipelines),
    )
    monkeypatch.setattr(
        cli_main,
        "_validate_compare_native_bundle",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("post-generation validation should be skipped")
        ),
    )

    assert (
        main(
            [
                "compare-native",
                "--suite",
                "flux2-native",
                "--artifact",
                str(tmp_path / "artifact"),
                "--prompt",
                "A native prompt",
                "--output",
                str(tmp_path / "comparison"),
                "--device",
                "cpu",
                "--dtype",
                "float32",
                "--skip-comparison-validation",
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert output["validation"] is None
    assert (tmp_path / "comparison" / "summary.json").is_file()


def test_cli_validate_comparison_accepts_copied_runpod_bundle(capsys, tmp_path):
    suite = get_native_suite("flux2-native")
    original_path = tmp_path / "flux2-native_seed4_original.png"
    orbitquant_path = tmp_path / "flux2-native_seed4_W4A4.png"
    comparison_path = tmp_path / "flux2-native_seed4_W4A4_original_vs_orbitquant.webp"
    original_image = Image.effect_noise((suite.width, suite.height), 32).convert("RGB")
    orbitquant_image = Image.effect_noise((suite.width, suite.height), 32).convert("RGB")
    original_image.save(original_path)
    orbitquant_image.save(orbitquant_path)
    comparison_sheet = Image.new("RGB", (suite.width * 2, suite.height + 24), "white")
    comparison_sheet.paste(original_image, (0, 24))
    comparison_sheet.paste(orbitquant_image, (suite.width, 24))
    comparison_sheet.save(comparison_path)

    for path, bit_setting in (
        (original_path, "original"),
        (orbitquant_path, "W4A4"),
    ):
        metadata = {
            "suite": suite.name,
            "model_id": "example/artifact-model",
            "prompt": "A native prompt",
            "seed": 4,
            "height": suite.height,
            "width": suite.width,
            "frames": suite.frames,
            "export_fps": suite.export_fps,
            "steps": suite.steps,
            "guidance": suite.guidance,
            "quantization": None
            if bit_setting == "original"
            else {"config": {"weight_bits": 4, "activation_bits": 4}},
        }
        path.with_suffix(path.suffix + ".json").write_text(
            json.dumps(metadata) + "\n",
            encoding="utf-8",
        )

    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "suite": suite.name,
                "model_id": "example/artifact-model",
                "source_model": "/workspace/hf-models/example",
                "artifact": "/workspace/artifacts/example",
                "component": "transformer",
                "prompt": "A native prompt",
                "seed": 4,
                "height": suite.height,
                "width": suite.width,
                "frames": suite.frames,
                "steps": suite.steps,
                "guidance": suite.guidance,
                "dtype": "bfloat16",
                "device": "cuda",
                "bit_setting": "W4A4",
                "runtime_mode": "triton_packed_matmul",
                "activation_kernel_backend": "triton_cuda",
                "enable_model_cpu_offload": True,
                "available_backends": {"cpu": True, "mps": False, "triton_cuda": True},
                "original": {
                    "output_path": f"/workspace/run/{original_path.name}",
                    "metadata_path": f"/workspace/run/{original_path.name}.json",
                    "wall_time_seconds": 10.0,
                    "peak_vram_bytes": 1000,
                },
                "orbitquant": {
                    "output_path": f"/workspace/run/{orbitquant_path.name}",
                    "metadata_path": f"/workspace/run/{orbitquant_path.name}.json",
                    "wall_time_seconds": 15.0,
                    "peak_vram_bytes": 900,
                    "runtime": {
                        "runtime_mode_counts": {"triton_packed_matmul": 1},
                        "activation_kernel_backend_counts": {"triton_cuda": 1},
                    },
                },
                "comparison_path": f"/workspace/run/{comparison_path.name}",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert main(["validate-comparison", "--input", str(tmp_path)]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["valid"] is True
    assert output["speed_ratio_orbitquant_over_original"] == 1.5
    assert output["comparison"]["size"] == [suite.width * 2, suite.height + 24]
    assert output["orbitquant_runtime"]["runtime_mode_counts"] == {"triton_packed_matmul": 1}


def test_cli_validate_comparison_accepts_wan_video_contact_sheets(capsys, tmp_path):
    from orbitquant.eval.assets import create_image_comparison_sheet

    suite = get_native_suite("wan-native")
    original_path = tmp_path / "wan-native_seed4_original.mp4"
    orbitquant_path = tmp_path / "wan-native_seed4_W4A4.mp4"
    original_contact_sheet = tmp_path / "wan-native_seed4_original_contact_sheet.webp"
    orbitquant_contact_sheet = tmp_path / "wan-native_seed4_W4A4_contact_sheet.webp"
    comparison_path = tmp_path / "wan-native_seed4_W4A4_original_vs_orbitquant.webp"
    contact_sheet_size = (suite.width * 4, suite.height * 3)

    for path in (original_path, orbitquant_path):
        path.write_bytes(b"fake-mp4")
    for path, colors in (
        (original_contact_sheet, ((0, 0, 0), (255, 255, 255))),
        (orbitquant_contact_sheet, ((255, 0, 0), (0, 0, 255))),
    ):
        sheet = Image.new("RGB", contact_sheet_size, colors[0])
        sheet.paste(colors[1], (0, 0, contact_sheet_size[0] // 2, contact_sheet_size[1]))
        sheet.save(path)
    create_image_comparison_sheet(
        original_contact_sheet,
        orbitquant_contact_sheet,
        comparison_path,
        labels=("BF16", "OrbitQuant W4A4"),
    )

    for path, contact_sheet, bit_setting in (
        (original_path, original_contact_sheet, "original"),
        (orbitquant_path, orbitquant_contact_sheet, "W4A4"),
    ):
        metadata = {
            "suite": suite.name,
            "model_id": "example/wan-artifact-model",
            "prompt": "A native video prompt",
            "seed": 4,
            "height": suite.height,
            "width": suite.width,
            "frames": suite.frames,
            "export_fps": suite.export_fps,
            "steps": suite.steps,
            "guidance": suite.guidance,
            "contact_sheet_path": f"/workspace/run/{contact_sheet.name}",
            "quantization": None
            if bit_setting == "original"
            else {"config": {"weight_bits": 4, "activation_bits": 4}},
        }
        path.with_suffix(path.suffix + ".json").write_text(
            json.dumps(metadata) + "\n",
            encoding="utf-8",
        )

    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "suite": suite.name,
                "model_id": "example/wan-artifact-model",
                "source_model": "/workspace/hf-models/example-wan",
                "artifact": "/workspace/artifacts/example-wan",
                "component": "transformer",
                "prompt": "A native video prompt",
                "seed": 4,
                "height": suite.height,
                "width": suite.width,
                "frames": suite.frames,
                "steps": suite.steps,
                "guidance": suite.guidance,
                "dtype": "bfloat16",
                "device": "cuda",
                "bit_setting": "W4A4",
                "runtime_mode": "triton_packed_matmul",
                "activation_kernel_backend": "triton_cuda",
                "enable_model_cpu_offload": True,
                "available_backends": {"cpu": True, "mps": False, "triton_cuda": True},
                "original": {
                    "output_path": f"/workspace/run/{original_path.name}",
                    "metadata_path": f"/workspace/run/{original_path.name}.json",
                    "wall_time_seconds": 10.0,
                    "peak_vram_bytes": 1000,
                },
                "orbitquant": {
                    "output_path": f"/workspace/run/{orbitquant_path.name}",
                    "metadata_path": f"/workspace/run/{orbitquant_path.name}.json",
                    "wall_time_seconds": 14.0,
                    "peak_vram_bytes": 900,
                    "runtime": {
                        "runtime_mode_counts": {"triton_packed_matmul": 1},
                        "activation_kernel_backend_counts": {"triton_cuda": 1},
                    },
                },
                "comparison_path": f"/workspace/run/{comparison_path.name}",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert main(["validate-comparison", "--input", str(tmp_path)]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["valid"] is True
    assert output["native_settings"]["frames"] == 81
    assert output["original"]["contact_sheet"]["size"] == [
        contact_sheet_size[0],
        contact_sheet_size[1],
    ]
    assert output["orbitquant"]["contact_sheet"]["size"] == [
        contact_sheet_size[0],
        contact_sheet_size[1],
    ]
    assert output["comparison"]["size"] == [
        contact_sheet_size[0] * 2,
        contact_sheet_size[1] + 24,
    ]
    assert output["speed_ratio_orbitquant_over_original"] == 1.4


def test_cli_validate_comparison_rejects_blank_images(tmp_path):
    suite = get_native_suite("flux2-native")
    for name, quantization in (
        ("flux2-native_seed4_original.png", None),
        (
            "flux2-native_seed4_W4A4.png",
            {"config": {"weight_bits": 4, "activation_bits": 4}},
        ),
    ):
        path = tmp_path / name
        Image.new("RGB", (suite.width, suite.height), "black").save(path)
        path.with_suffix(path.suffix + ".json").write_text(
            json.dumps(
                {
                    "suite": suite.name,
                    "model_id": "example/artifact-model",
                    "prompt": "A native prompt",
                    "seed": 4,
                    "height": suite.height,
                    "width": suite.width,
                    "frames": suite.frames,
                    "export_fps": suite.export_fps,
                    "steps": suite.steps,
                    "guidance": suite.guidance,
                    "quantization": quantization,
                }
            )
            + "\n",
            encoding="utf-8",
        )
    Image.new("RGB", (suite.width * 2, suite.height + 24), "black").save(
        tmp_path / "flux2-native_seed4_W4A4_original_vs_orbitquant.webp"
    )
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "suite": suite.name,
                "model_id": "example/artifact-model",
                "prompt": "A native prompt",
                "seed": 4,
                "bit_setting": "W4A4",
                "original": {
                    "output_path": "/workspace/run/flux2-native_seed4_original.png",
                    "metadata_path": "/workspace/run/flux2-native_seed4_original.png.json",
                },
                "orbitquant": {
                    "output_path": "/workspace/run/flux2-native_seed4_W4A4.png",
                    "metadata_path": "/workspace/run/flux2-native_seed4_W4A4.png.json",
                },
                "comparison_path": (
                    "/workspace/run/flux2-native_seed4_W4A4_original_vs_orbitquant.webp"
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="comparison image looks blank"):
        main(["validate-comparison", "--input", str(tmp_path)])


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
