import json

import torch
from safetensors.torch import load_file

from orbitquant.artifacts import load_orbitquant_artifact, save_orbitquant_artifact
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
    assert "orbitquant_codebooks.safetensors" in {path.name for path in tmp_path.iterdir()}
    assert "orbitquant_rotations.safetensors" in {path.name for path in tmp_path.iterdir()}
    assert manifest["checksums"]["model.safetensors"] == next(
        line.split()[0]
        for line in (tmp_path / "SHA256SUMS").read_text().splitlines()
        if line.endswith("  model.safetensors")
    )
    assert "orbitquant_codebooks.safetensors" in manifest["checksums"]
    assert "orbitquant_rotations.safetensors" in manifest["checksums"]
    codebook_tensors = load_file(tmp_path / "orbitquant_codebooks.safetensors")
    rotation_tensors = load_file(tmp_path / "orbitquant_rotations.safetensors")
    assert any(name.endswith("packed_weight_indices") for name in tensors)
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
