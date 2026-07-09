import json

import torch
from PIL import Image

from orbitquant.artifacts import record_artifact_asset, record_artifact_metrics
from orbitquant.artifacts.writer import save_orbitquant_artifact
from orbitquant.config import OrbitQuantConfig
from orbitquant.eval.external_export import export_geneval_artifact, export_vbench_artifact
from orbitquant.eval.external_metrics import (
    summarize_geneval_results,
    summarize_vbench_results,
)
from orbitquant.eval.metrics import load_metric_json
from orbitquant.modeling import quantize_linear_modules


class TinyArtifactModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList(
            [torch.nn.ModuleDict({"attn": torch.nn.ModuleDict({"to_q": torch.nn.Linear(8, 8)})})]
        )


def _write_artifact(tmp_path):
    model = TinyArtifactModel()
    config = OrbitQuantConfig(block_size=4, target_policy="generic_dit")
    summary = quantize_linear_modules(model, config, quantization_device=None)
    save_orbitquant_artifact(
        model,
        tmp_path,
        config=config,
        source_model_id="example/model",
        source_revision="abc123",
        source_license="apache-2.0",
        summary=summary,
    )


def test_export_geneval_artifact_writes_upstream_folder_layout(tmp_path):
    _write_artifact(tmp_path)
    image_path = tmp_path / "assets" / "sample.png"
    Image.new("RGB", (8, 8), "red").save(image_path)
    record_artifact_asset(tmp_path, image_path, validate_checksums_enabled=False)
    record_artifact_metrics(
        tmp_path,
        split="orbitquant",
        metrics={"generated_samples": 1},
        metadata={
            "output_path": str(image_path),
            "prompt_record": {
                "id": "red-mug",
                "prompt": "a red mug",
                "geneval": {
                    "tag": "single_object",
                    "include": [{"class": "cup", "count": 1, "color": "red"}],
                    "exclude": [],
                },
            },
        },
        validate_checksums_enabled=False,
    )

    result = export_geneval_artifact(
        tmp_path,
        tmp_path / "geneval-export",
        split="orbitquant",
    )

    prompt_dir = tmp_path / "geneval-export" / "00000"
    metadata = json.loads((prompt_dir / "metadata.jsonl").read_text())
    assert result.sample_count == 1
    assert result.prompt_count == 1
    assert metadata["tag"] == "single_object"
    assert metadata["prompt"] == "a red mug"
    assert (prompt_dir / "samples" / "00000.png").is_file()
    assert (prompt_dir / "grid.png").is_file()
    assert (tmp_path / "geneval-export" / "orbitquant_geneval_export.json").is_file()


def test_export_geneval_artifact_fails_loud_when_no_geneval_records_exist(tmp_path):
    _write_artifact(tmp_path)
    image_path = tmp_path / "assets" / "sample.png"
    Image.new("RGB", (8, 8), "blue").save(image_path)
    record_artifact_metrics(
        tmp_path,
        split="orbitquant",
        metrics={"generated_samples": 1},
        metadata={
            "output_path": str(image_path),
            "prompt_record": {"id": "visual-only", "prompt": "a blue cube"},
        },
        validate_checksums_enabled=False,
    )

    try:
        export_geneval_artifact(tmp_path, tmp_path / "geneval-export", split="orbitquant")
    except ValueError as exc:
        assert "no GenEval-compatible image records" in str(exc)
    else:
        raise AssertionError("visual-only artifact was accepted as GenEval metadata")


def test_export_geneval_artifact_skips_visual_rows_in_mixed_artifact(tmp_path):
    _write_artifact(tmp_path)
    geneval_image_path = tmp_path / "assets" / "geneval-sample.png"
    Image.new("RGB", (8, 8), "red").save(geneval_image_path)
    record_artifact_metrics(
        tmp_path,
        split="orbitquant",
        metrics={"generated_samples": 1},
        metadata={
            "output_path": str(tmp_path / "assets" / "missing-visual-row.png"),
            "prompt_record": {"id": "visual-only", "prompt": "a missing visual row"},
        },
        validate_checksums_enabled=False,
    )
    record_artifact_metrics(
        tmp_path,
        split="orbitquant",
        metrics={"generated_samples": 1},
        metadata={
            "output_path": str(geneval_image_path),
            "prompt_record": {
                "id": "red-cup",
                "prompt": "a red cup",
                "geneval": {
                    "tag": "single_object",
                    "include": [{"class": "cup", "count": 1, "color": "red"}],
                    "exclude": [],
                },
            },
        },
        validate_checksums_enabled=False,
    )

    result = export_geneval_artifact(
        tmp_path,
        tmp_path / "geneval-export",
        split="orbitquant",
    )

    assert result.sample_count == 1
    assert result.prompt_count == 1
    assert (tmp_path / "geneval-export" / "00000" / "samples" / "00000.png").is_file()


def test_export_vbench_artifact_writes_video_folder_and_prompt_file(tmp_path):
    _write_artifact(tmp_path)
    video_path = tmp_path / "assets" / "sample.mp4"
    video_path.write_bytes(b"fake mp4")
    record_artifact_asset(tmp_path, video_path, validate_checksums_enabled=False)
    record_artifact_metrics(
        tmp_path,
        split="orbitquant",
        metrics={"generated_samples": 1},
        metadata={
            "output_path": str(video_path),
            "seed": 7,
            "prompt_record": {"id": "boat", "prompt": "a boat moving on water"},
        },
        validate_checksums_enabled=False,
    )

    result = export_vbench_artifact(
        tmp_path,
        tmp_path / "vbench-export",
        split="orbitquant",
        link_mode="copy",
    )

    exported_video = tmp_path / "vbench-export" / "00000_boat_seed7.mp4"
    prompt_file = tmp_path / "vbench-export" / "vbench_prompts.json"
    export_manifest = tmp_path / "vbench-export" / "orbitquant_vbench_export.json"
    prompts = json.loads(prompt_file.read_text())
    manifest = json.loads(export_manifest.read_text())
    assert result.sample_count == 1
    assert exported_video.read_bytes() == b"fake mp4"
    assert prompts["00000_boat_seed7.mp4"] == "a boat moving on water"
    assert manifest["videos"][0]["path"] == str(exported_video)


def test_external_metric_summarizers_write_numeric_json(tmp_path):
    geneval_results = tmp_path / "geneval.jsonl"
    geneval_results.write_text(
        "\n".join(
            [
                json.dumps({"tag": "single_object", "correct": True}),
                json.dumps({"tag": "single_object", "correct": False}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    geneval_summary = summarize_geneval_results(
        geneval_results,
        tmp_path / "geneval-summary.json",
    )
    imported_geneval_metrics = load_metric_json(
        tmp_path / "geneval-summary.json",
        metric_prefix="geneval",
    )
    assert geneval_summary["overall"] == 0.5
    assert geneval_summary["per_task"]["single_object"] == 0.5
    assert geneval_summary["tags"]["single_object"]["total"] == 2
    assert imported_geneval_metrics["geneval_overall"] == 0.5
    assert imported_geneval_metrics["geneval_per_task_single_object"] == 0.5

    vbench_dir = tmp_path / "vbench"
    vbench_dir.mkdir()
    (vbench_dir / "results.json").write_text(
        json.dumps({"subject_consistency": {"score": 0.75}}),
        encoding="utf-8",
    )
    (vbench_dir / "dynamic_degree.json").write_text(
        json.dumps({"score": 0.5}),
        encoding="utf-8",
    )
    (vbench_dir / "table.json").write_text(
        json.dumps({"dimension": "background_consistency", "score": 0.625}),
        encoding="utf-8",
    )
    summary_path = tmp_path / "vbench-summary.json"
    vbench_summary = summarize_vbench_results(
        vbench_dir,
        summary_path,
    )
    imported_metrics = load_metric_json(summary_path, metric_prefix="vbench")
    assert vbench_summary["metrics"]["results_subject_consistency_score"] == 0.75
    assert vbench_summary["subject_consistency"] == 0.75
    assert vbench_summary["dynamic_degree"] == 0.5
    assert vbench_summary["background_consistency"] == 0.625
    assert imported_metrics["vbench_subject_consistency"] == 0.75
    assert imported_metrics["vbench_dynamic_degree"] == 0.5
    assert imported_metrics["vbench_background_consistency"] == 0.625
