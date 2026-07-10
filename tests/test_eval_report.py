import json

import torch
from PIL import Image

from orbitquant.artifacts import record_artifact_metrics, save_orbitquant_artifact
from orbitquant.config import OrbitQuantConfig
from orbitquant.eval.report import _comparison_column_label, generate_native_eval_report
from orbitquant.modeling import quantize_linear_modules


class TinyReportModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList(
            [torch.nn.ModuleDict({"attn": torch.nn.ModuleDict({"to_q": torch.nn.Linear(8, 8)})})]
        )


def _write_artifact(path, *, source_model_id: str, target_policy: str):
    model = TinyReportModel()
    config = OrbitQuantConfig(block_size=4, target_policy=target_policy)
    summary = quantize_linear_modules(model, config)
    save_orbitquant_artifact(
        model,
        path,
        config=config,
        source_model_id=source_model_id,
        source_revision="abc123",
        source_license="apache-2.0",
        summary=summary,
    )
    original_path = path / "assets" / "flux2-native_seed1_original_simple-object.png"
    orbitquant_path = path / "assets" / "flux2-native_seed1_W4A4_simple-object.png"
    original_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), "red").save(original_path)
    Image.new("RGB", (16, 16), "blue").save(orbitquant_path)
    record_artifact_metrics(
        path,
        split="original",
        metrics={"geneval_overall": 0.74, "wall_time_seconds": 12.0},
        metadata={
            "suite": "flux2-native",
            "seed": 1,
            "output_path": str(original_path),
            "prompt_record": {"id": "simple-object"},
        },
    )
    record_artifact_metrics(
        path,
        split="orbitquant",
        metrics={"geneval_overall": 0.71, "wall_time_seconds": 8.5},
        metadata={
            "suite": "flux2-native",
            "seed": 1,
            "bit_setting": "W4A4",
            "output_path": str(orbitquant_path),
            "prompt_record": {"id": "simple-object"},
            "asset_paths": [
                str(path / "assets" / "original_vs_orbitquant_flux2-native_seed1_W4A4.webp")
            ],
        },
    )


def test_generate_native_eval_report_writes_markdown_and_tables(tmp_path):
    artifact_dir = tmp_path / "flux2-w4a4"
    report_dir = tmp_path / "reports"
    _write_artifact(
        artifact_dir,
        source_model_id="black-forest-labs/FLUX.2-klein-4B",
        target_policy="flux2",
    )

    result = generate_native_eval_report(
        [artifact_dir],
        report_dir,
        report_date="20260706",
    )

    assert result.report_path == report_dir / "orbitquant-native-eval-20260706.md"
    assert result.report_path.is_file()
    assert (report_dir / "tables" / "image_geneval.csv").is_file()
    assert (report_dir / "tables" / "video_vbench.csv").is_file()
    assert (report_dir / "tables" / "perf.csv").is_file()
    assert (report_dir / "tables" / "assets.csv").is_file()
    assert (report_dir / "tables" / "missing_required_metrics.csv").is_file()
    assert (report_dir / "assets" / "image_generation_comparison_matrix.webp").is_file()
    with Image.open(
        report_dir / "assets" / "image_generation_comparison_matrix.webp"
    ) as matrix:
        assert matrix.size == (2332, 1140)

    report = result.report_path.read_text()
    image_table = (report_dir / "tables" / "image_geneval.csv").read_text()
    perf_table = (report_dir / "tables" / "perf.csv").read_text()
    assets_table = (report_dir / "tables" / "assets.csv").read_text()
    missing_table = (report_dir / "tables" / "missing_required_metrics.csv").read_text()

    assert "Extra Targets" in report
    assert "black-forest-labs/FLUX.2-klein-4B" in report
    assert "W4A4" in report
    assert "geneval_overall" in report
    assert "Generated Assets" in report
    assert "Visual Comparison Matrices" in report
    assert "image_generation_comparison_matrix.webp" in report
    assert "No required paper-target metrics are missing." in report
    assert "0.71" in image_table
    assert "8.5" in perf_table
    assert "output" in assets_table
    assert "comparison" in assets_table
    assert "flux2-native_seed1_W4A4_simple-object.png" in assets_table
    assert "original_vs_orbitquant_flux2-native_seed1_W4A4.webp" in assets_table
    assert missing_table == "target_group,model_id,bits,split,suite,metric,artifact_dir\n"
    assert result.table_paths["assets"] == report_dir / "tables" / "assets.csv"
    assert result.comparison_asset_paths == {
        "image": report_dir / "assets" / "image_generation_comparison_matrix.webp"
    }
    assert (
        result.table_paths["missing_required_metrics"]
        == report_dir / "tables" / "missing_required_metrics.csv"
    )
    assert result.rows[0]["target_group"] == "extra"


def test_comparison_column_label_keeps_baseline_independent_from_bits():
    row = {
        "source_model_id": "black-forest-labs/FLUX.2-klein-4B",
        "bits": "W2A3",
        "split": "original",
    }

    assert _comparison_column_label(row) == "FLUX.2-klein-4B\nBF16 baseline"

    row["split"] = "orbitquant"

    assert _comparison_column_label(row) == "FLUX.2-klein-4B\nW2A3 OrbitQuant"


def test_generate_native_eval_report_validates_artifacts_before_reading_metrics(tmp_path):
    artifact_dir = tmp_path / "flux2-w4a4"
    _write_artifact(
        artifact_dir,
        source_model_id="black-forest-labs/FLUX.2-klein-4B",
        target_policy="flux2",
    )
    manifest_path = artifact_dir / "orbitquant_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["checksums"]["model.safetensors"] = "bad"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    try:
        generate_native_eval_report([artifact_dir], tmp_path / "reports")
    except RuntimeError as exc:
        assert "checksum mismatch for model.safetensors" in str(exc)
    else:
        raise AssertionError("generate_native_eval_report accepted a corrupted artifact")


def test_generate_native_eval_report_flags_missing_required_paper_metrics(tmp_path):
    artifact_dir = tmp_path / "flux1-w4a4"
    report_dir = tmp_path / "reports"
    _write_artifact(
        artifact_dir,
        source_model_id="black-forest-labs/FLUX.1-schnell",
        target_policy="flux",
    )
    record_artifact_metrics(
        artifact_dir,
        split="orbitquant",
        metrics={"wall_time_seconds": 8.5},
        metadata={"suite": "flux1-schnell-native", "seed": 1, "bit_setting": "W4A4"},
    )

    result = generate_native_eval_report(
        [artifact_dir],
        report_dir,
        report_date="20260706",
    )

    report = result.report_path.read_text()
    assert "Missing Required Metrics" in report
    assert "black-forest-labs/FLUX.1-schnell" in report
    assert "`W4A4`" in report
    assert "flux1-schnell-native" in report
    assert "geneval_overall" in report
    missing_table = (report_dir / "tables" / "missing_required_metrics.csv").read_text()
    assert "flux1-schnell-native,geneval_overall" in missing_table
    assert {
        "artifact_dir": str(artifact_dir),
        "source_model_id": "black-forest-labs/FLUX.1-schnell",
        "target_group": "paper",
        "bits": "W4A4",
        "split": "orbitquant",
        "suite": "flux1-schnell-native",
        "metric": "geneval_overall",
    } in result.missing_required_metrics
