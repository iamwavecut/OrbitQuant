import json

import torch
from safetensors.torch import load_file

from orbitquant.artifacts import (
    load_orbitquant_artifact,
    record_artifact_metrics,
    save_orbitquant_artifact,
    sha256_file,
    validate_orbitquant_artifact,
)
from orbitquant.config import OrbitQuantConfig
from orbitquant.layers import OrbitQuantLinear
from orbitquant.modeling import quantize_linear_modules


class TinyArtifactModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList(
            [torch.nn.ModuleDict({"attn": torch.nn.ModuleDict({"to_q": torch.nn.Linear(8, 8)})})]
        )


def test_save_orbitquant_artifact_writes_manifest_readme_weights_and_checksums(tmp_path):
    model = TinyArtifactModel()
    config = OrbitQuantConfig(block_size=4)
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

    manifest = json.loads((tmp_path / "orbitquant_manifest.json").read_text())
    tensors = load_file(tmp_path / "model.safetensors")

    assert manifest["artifact_format"] == "orbitquant-v1"
    assert manifest["source_model_id"] == "example/model"
    assert "README.md" in {path.name for path in tmp_path.iterdir()}
    assert "SHA256SUMS" in {path.name for path in tmp_path.iterdir()}
    assert "prompts.json" in {path.name for path in tmp_path.iterdir()}
    assert (tmp_path / "benchmark" / "summary.json").is_file()
    assert "orbitquant_codebooks.safetensors" in {path.name for path in tmp_path.iterdir()}
    assert "orbitquant_rotations.safetensors" in {path.name for path in tmp_path.iterdir()}
    assert manifest["checksums"]["model.safetensors"] == next(
        line.split()[0]
        for line in (tmp_path / "SHA256SUMS").read_text().splitlines()
        if line.endswith("  model.safetensors")
    )
    assert "orbitquant_codebooks.safetensors" in manifest["checksums"]
    assert "orbitquant_rotations.safetensors" in manifest["checksums"]
    assert "prompts.json" in manifest["checksums"]
    assert "benchmark/summary.json" in manifest["checksums"]
    assert (tmp_path / "benchmark" / "original.metrics.jsonl").is_file()
    assert (tmp_path / "benchmark" / "orbitquant.metrics.jsonl").is_file()
    assert (tmp_path / "benchmark" / "original.metrics.csv").is_file()
    assert (tmp_path / "benchmark" / "orbitquant.metrics.csv").is_file()
    assert (tmp_path / "assets" / ".gitkeep").is_file()
    assert "benchmark/original.metrics.jsonl" in manifest["checksums"]
    assert "benchmark/orbitquant.metrics.jsonl" in manifest["checksums"]
    assert "benchmark/original.metrics.csv" in manifest["checksums"]
    assert "benchmark/orbitquant.metrics.csv" in manifest["checksums"]
    assert "assets/.gitkeep" in manifest["checksums"]
    prompts = json.loads((tmp_path / "prompts.json").read_text())
    benchmark_summary = json.loads((tmp_path / "benchmark" / "summary.json").read_text())
    codebook_tensors = load_file(tmp_path / "orbitquant_codebooks.safetensors")
    rotation_tensors = load_file(tmp_path / "orbitquant_rotations.safetensors")
    assert any(name.endswith("packed_weight_indices") for name in tensors)
    assert prompts == {"prompts": []}
    assert benchmark_summary["status"] == "not_run"
    assert benchmark_summary["source_model_id"] == "example/model"
    assert (tmp_path / "benchmark" / "original.metrics.jsonl").read_text() == ""
    assert (tmp_path / "benchmark" / "orbitquant.metrics.jsonl").read_text() == ""
    assert (tmp_path / "benchmark" / "original.metrics.csv").read_text() == "metric,value\n"
    assert (tmp_path / "benchmark" / "orbitquant.metrics.csv").read_text() == "metric,value\n"
    assert any(name.endswith(".centroids") for name in codebook_tensors)
    assert any(name.endswith(".permutation") for name in rotation_tensors)


def test_load_orbitquant_artifact_restores_quantized_modules_into_matching_model(tmp_path):
    torch.manual_seed(0)
    source = TinyArtifactModel()
    config = OrbitQuantConfig(block_size=4)
    summary = quantize_linear_modules(source, config)

    save_orbitquant_artifact(
        source,
        tmp_path,
        config=config,
        source_model_id="example/model",
        source_revision="abc123",
        source_license="apache-2.0",
        summary=summary,
    )

    restored = TinyArtifactModel()
    manifest = load_orbitquant_artifact(restored, tmp_path)

    assert manifest.source_model_id == "example/model"
    assert isinstance(restored.transformer_blocks[0]["attn"]["to_q"], OrbitQuantLinear)
    restored_state = restored.state_dict()
    packed_key = "transformer_blocks.0.attn.to_q.packed_weight_indices"
    assert restored_state[packed_key].dtype == torch.uint8


def test_validate_orbitquant_artifact_reports_eval_ready_required_files(tmp_path):
    source = TinyArtifactModel()
    config = OrbitQuantConfig(block_size=4)
    summary = quantize_linear_modules(source, config)
    save_orbitquant_artifact(
        source,
        tmp_path,
        config=config,
        source_model_id="example/model",
        source_revision="abc123",
        source_license="apache-2.0",
        summary=summary,
    )

    result = validate_orbitquant_artifact(tmp_path)

    assert "benchmark/original.metrics.jsonl" in result["required_files"]
    assert "benchmark/orbitquant.metrics.jsonl" in result["required_files"]
    assert "benchmark/original.metrics.csv" in result["required_files"]
    assert "benchmark/orbitquant.metrics.csv" in result["required_files"]
    assert "assets/.gitkeep" in result["required_files"]


def test_record_artifact_metrics_keeps_manifest_and_sha256sums_valid(tmp_path):
    source = TinyArtifactModel()
    config = OrbitQuantConfig(block_size=4)
    summary = quantize_linear_modules(source, config)
    save_orbitquant_artifact(
        source,
        tmp_path,
        config=config,
        source_model_id="example/model",
        source_revision="abc123",
        source_license="apache-2.0",
        summary=summary,
    )

    record_artifact_metrics(
        tmp_path,
        split="orbitquant",
        metrics={"geneval_overall": 0.71, "peak_vram_gb": 42.5},
        metadata={"suite": "flux2-native", "seed": 5, "bit_setting": "W4A4"},
    )

    result = validate_orbitquant_artifact(tmp_path)
    manifest = json.loads((tmp_path / "orbitquant_manifest.json").read_text())
    summary_payload = json.loads((tmp_path / "benchmark" / "summary.json").read_text())
    jsonl_rows = (tmp_path / "benchmark" / "orbitquant.metrics.jsonl").read_text().splitlines()
    csv_rows = (tmp_path / "benchmark" / "orbitquant.metrics.csv").read_text().splitlines()
    sha_lines = (tmp_path / "SHA256SUMS").read_text().splitlines()

    assert result["valid"] is True
    assert len(jsonl_rows) == 1
    assert json.loads(jsonl_rows[0]) == {
        "split": "orbitquant",
        "metrics": {"geneval_overall": 0.71, "peak_vram_gb": 42.5},
        "metadata": {"suite": "flux2-native", "seed": 5, "bit_setting": "W4A4"},
    }
    assert csv_rows == [
        "metric,value",
        "geneval_overall,0.71",
        "peak_vram_gb,42.5",
    ]
    assert summary_payload["status"] == "metrics_recorded"
    assert summary_payload["metrics"]["orbitquant"]["records"] == 1
    assert summary_payload["metrics"]["orbitquant"]["latest"]["metadata"]["seed"] == 5
    assert manifest["checksums"]["benchmark/summary.json"] == sha256_file(
        tmp_path / "benchmark" / "summary.json"
    )
    assert manifest["checksums"]["benchmark/orbitquant.metrics.jsonl"] == sha256_file(
        tmp_path / "benchmark" / "orbitquant.metrics.jsonl"
    )
    assert manifest["checksums"]["benchmark/orbitquant.metrics.csv"] == sha256_file(
        tmp_path / "benchmark" / "orbitquant.metrics.csv"
    )
    assert any(line.endswith("  orbitquant_manifest.json") for line in sha_lines)
    assert any(
        line
        == f"{manifest['checksums']['benchmark/orbitquant.metrics.jsonl']}  "
        "benchmark/orbitquant.metrics.jsonl"
        for line in sha_lines
    )


def test_record_artifact_metrics_rejects_corrupted_artifact_before_refreshing_checksums(
    tmp_path,
):
    source = TinyArtifactModel()
    config = OrbitQuantConfig(block_size=4)
    summary = quantize_linear_modules(source, config)
    save_orbitquant_artifact(
        source,
        tmp_path,
        config=config,
        source_model_id="example/model",
        source_revision="abc123",
        source_license="apache-2.0",
        summary=summary,
    )
    original_manifest = json.loads((tmp_path / "orbitquant_manifest.json").read_text())
    with (tmp_path / "model.safetensors").open("ab") as handle:
        handle.write(b"corruption")

    try:
        record_artifact_metrics(
            tmp_path,
            split="orbitquant",
            metrics={"geneval_overall": 0.71},
        )
    except RuntimeError as exc:
        assert "checksum mismatch for model.safetensors" in str(exc)
    else:
        raise AssertionError("record_artifact_metrics refreshed a corrupted artifact")

    manifest = json.loads((tmp_path / "orbitquant_manifest.json").read_text())
    assert manifest == original_manifest
    assert (tmp_path / "benchmark" / "orbitquant.metrics.jsonl").read_text() == ""


def test_load_orbitquant_artifact_uses_prequantized_skeletons(tmp_path, monkeypatch):
    source = TinyArtifactModel()
    config = OrbitQuantConfig(block_size=4)
    summary = quantize_linear_modules(source, config)
    save_orbitquant_artifact(
        source,
        tmp_path,
        config=config,
        source_model_id="example/model",
        source_revision="abc123",
        source_license="apache-2.0",
        summary=summary,
    )

    def fail_from_linear(cls, *args, **kwargs):
        raise AssertionError("artifact load should not requantize source Linear weights")

    monkeypatch.setattr(OrbitQuantLinear, "from_linear", classmethod(fail_from_linear))

    restored = TinyArtifactModel()
    load_orbitquant_artifact(restored, tmp_path)

    assert isinstance(restored.transformer_blocks[0]["attn"]["to_q"], OrbitQuantLinear)


def test_load_orbitquant_artifact_rejects_checksum_mismatch(tmp_path):
    source = TinyArtifactModel()
    config = OrbitQuantConfig(block_size=4)
    summary = quantize_linear_modules(source, config)
    save_orbitquant_artifact(
        source,
        tmp_path,
        config=config,
        source_model_id="example/model",
        source_revision="abc123",
        source_license="apache-2.0",
        summary=summary,
    )
    with (tmp_path / "model.safetensors").open("ab") as handle:
        handle.write(b"corruption")

    try:
        load_orbitquant_artifact(TinyArtifactModel(), tmp_path)
    except RuntimeError as exc:
        assert "checksum mismatch" in str(exc)
    else:
        raise AssertionError("load_orbitquant_artifact accepted a corrupted artifact")


def test_load_orbitquant_artifact_rejects_missing_required_layout_file(tmp_path):
    source = TinyArtifactModel()
    config = OrbitQuantConfig(block_size=4)
    summary = quantize_linear_modules(source, config)
    save_orbitquant_artifact(
        source,
        tmp_path,
        config=config,
        source_model_id="example/model",
        source_revision="abc123",
        source_license="apache-2.0",
        summary=summary,
    )
    (tmp_path / "README.md").unlink()

    try:
        load_orbitquant_artifact(TinyArtifactModel(), tmp_path)
    except RuntimeError as exc:
        assert "required artifact file missing" in str(exc)
        assert "README.md" in str(exc)
    else:
        raise AssertionError("load_orbitquant_artifact accepted an incomplete artifact layout")


def test_validate_orbitquant_artifact_rejects_missing_required_layout_file(tmp_path):
    source = TinyArtifactModel()
    config = OrbitQuantConfig(block_size=4)
    summary = quantize_linear_modules(source, config)
    save_orbitquant_artifact(
        source,
        tmp_path,
        config=config,
        source_model_id="example/model",
        source_revision="abc123",
        source_license="apache-2.0",
        summary=summary,
    )
    (tmp_path / "README.md").unlink()

    try:
        validate_orbitquant_artifact(tmp_path)
    except RuntimeError as exc:
        assert "required artifact file missing" in str(exc)
        assert "README.md" in str(exc)
    else:
        raise AssertionError("validate_orbitquant_artifact accepted an incomplete artifact layout")
