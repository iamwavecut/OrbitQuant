import json

import torch
from safetensors.torch import load_file

from orbitquant.artifacts import save_orbitquant_artifact
from orbitquant.config import OrbitQuantConfig
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
    assert any(name.endswith("packed_weight_indices") for name in tensors)
