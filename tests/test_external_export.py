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


def test_export_geneval_artifact_fails_loud_for_visual_prompt_records(tmp_path):
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
        assert "not GenEval-compatible" in str(exc)
    else:
        raise AssertionError("visual prompt record was accepted as GenEval metadata")


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
    prompts = json.loads(prompt_file.read_text())
    assert result.sample_count == 1
    assert exported_video.read_bytes() == b"fake mp4"
    assert prompts[str(exported_video)] == "a boat moving on water"
    assert (tmp_path / "vbench-export" / "orbitquant_vbench_export.json").is_file()


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
    assert geneval_summary["overall"] == 0.5
    assert geneval_summary["tags"]["single_object"]["total"] == 2

    vbench_dir = tmp_path / "vbench"
    vbench_dir.mkdir()
    (vbench_dir / "results.json").write_text(
        json.dumps({"subject_consistency": {"score": 0.75}}),
        encoding="utf-8",
    )
    vbench_summary = summarize_vbench_results(
        vbench_dir,
        tmp_path / "vbench-summary.json",
    )
    assert vbench_summary["metrics"]["results_subject_consistency_score"] == 0.75
