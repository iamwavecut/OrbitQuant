from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

SCRIPT_PATH = Path("scripts/verify_hf_kernel_model_artifact.py")


def _load_script_module():
    spec = importlib.util.spec_from_file_location("verify_hf_kernel_model_artifact", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_model_kernel_verifier_targets_one_published_orbitquant_artifact_layer(tmp_path):
    verifier = _load_script_module()
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    module_name = "transformer_blocks.0.attn.to_q"
    manifest = {
        "source_model_id": "black-forest-labs/FLUX.2-klein-4B",
        "source_revision": "abc123",
        "weight_bits": 4,
        "activation_bits": 4,
        "quantized_modules": [module_name],
    }
    config = {
        "weight_bits": 4,
        "activation_bits": 4,
        "rotation_seed": 0,
        "block_size": 8,
        "target_policy": "flux2",
        "runtime_mode": "auto_fused",
        "activation_kernel_backend": "auto",
    }
    (artifact_dir / "orbitquant_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (artifact_dir / "quantization_config.json").write_text(json.dumps(config), encoding="utf-8")
    save_file(
        {
            f"{module_name}.packed_weight_indices": torch.zeros(2048, dtype=torch.uint8),
            f"{module_name}.row_norms": torch.ones(64, dtype=torch.bfloat16),
        },
        artifact_dir / "model.safetensors",
    )

    tensors = verifier._load_layer_tensors(artifact_dir, module_name)
    in_features, out_features = verifier._infer_features(
        packed_weight_indices=tensors["packed_weight_indices"],
        row_norms=tensors["row_norms"],
        bits=4,
    )
    storage = verifier._storage_payload(
        packed_weight_indices=tensors["packed_weight_indices"],
        row_norms=tensors["row_norms"],
        bias=tensors["bias"],
        bits=4,
        out_features=out_features,
        in_features=in_features,
        dtype=torch.float16,
    )

    assert verifier.DEFAULT_ARTIFACT_REPO == "WaveCut/FLUX.2-klein-4B-OrbitQuant-W4A4"
    assert verifier._select_module(manifest, None) == module_name
    assert in_features == 64
    assert out_features == 64
    assert storage["packed_weight_path_bytes"] < storage["materialized_weight_bytes"]


def test_model_kernel_verifier_avoids_full_pipeline_generation_path():
    source = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "load_native_packed_matmul_kernel" in source
    assert "OrbitQuantLinear" in source
    assert "snapshot_download" in source
    assert "DiffusionPipeline" not in source
    assert ".from_pretrained(" not in source
    assert ".generate(" not in source
