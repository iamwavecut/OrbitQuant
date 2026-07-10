import json

import torch
from PIL import Image
from safetensors.torch import load_file, save_file

from orbitquant.adaln import RTNInt4Linear
from orbitquant.artifacts import (
    create_artifact_image_comparisons,
    load_orbitquant_artifact,
    record_artifact_asset,
    record_artifact_metrics,
    refresh_artifact_checksums,
    repair_artifact_metadata,
    save_orbitquant_artifact,
    sha256_file,
    validate_artifact_policy_inventory,
    validate_orbitquant_artifact,
)
from orbitquant.artifacts.checksums import (
    read_sha256sums,
    validate_sha256sums,
    write_sha256sums,
    write_sha256sums_from_manifest,
)
from orbitquant.config import OrbitQuantConfig
from orbitquant.layers import OrbitQuantLinear
from orbitquant.modeling import inspect_linear_module_policy, quantize_linear_modules


class TinyArtifactModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList(
            [torch.nn.ModuleDict({"attn": torch.nn.ModuleDict({"to_q": torch.nn.Linear(8, 8)})})]
        )


class TinySharedCodebookArtifactModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {
                        "attn": torch.nn.ModuleDict(
                            {
                                "to_q": torch.nn.Linear(8, 8),
                                "to_k": torch.nn.Linear(8, 8),
                            }
                        )
                    }
                )
            ]
        )


class FluxTransformer2DArtifactModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {
                        "norm1": torch.nn.ModuleDict({"linear": torch.nn.Linear(8, 16)}),
                        "attn": torch.nn.ModuleDict({"to_q": torch.nn.Linear(8, 8)}),
                    }
                )
            ]
        )
        self.proj_out = torch.nn.Linear(8, 8)


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
    model_index = json.loads((tmp_path / "model_index.json").read_text())
    tensors = load_file(tmp_path / "model.safetensors")

    assert manifest["artifact_format"] == "orbitquant-v1"
    assert model_index["_class_name"] == "OrbitQuantComponentArtifact"
    assert model_index["quant_method"] == "orbitquant"
    assert model_index["source_model_id"] == "example/model"
    assert model_index["source_revision"] == "abc123"
    assert model_index["component"] == "transformer"
    assert model_index["weight_name"] == "model.safetensors"
    assert model_index["quantization_config"] == "quantization_config.json"
    assert model_index["manifest"] == "orbitquant_manifest.json"
    assert manifest["source_model_id"] == "example/model"
    assert manifest["quantization_device"] == summary.quantization_device
    assert manifest["weight_quantization_backend"] == summary.weight_quantization_backend
    assert manifest["quantization_staging_mode"] == summary.quantization_staging_mode
    assert manifest["adaln_group_size"] == 64
    assert manifest["adaln_policy"] == "int4_rtn_group64_bf16_activation"
    assert model_index["quantization_device"] == summary.quantization_device
    assert model_index["weight_quantization_backend"] == summary.weight_quantization_backend
    assert model_index["quantization_staging_mode"] == summary.quantization_staging_mode
    assert model_index["activation_eps"] == 1e-10
    assert model_index["codebook_version"] == 2
    assert "README.md" in {path.name for path in tmp_path.iterdir()}
    assert "model_index.json" in {path.name for path in tmp_path.iterdir()}
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
    assert "model_index.json" in manifest["checksums"]
    assert "prompts.json" in manifest["checksums"]
    assert "benchmark/summary.json" in manifest["checksums"]
    assert not (tmp_path / "benchmark" / "original.metrics.jsonl").exists()
    assert not (tmp_path / "benchmark" / "orbitquant.metrics.jsonl").exists()
    assert not (tmp_path / "benchmark" / "original.metrics.csv").exists()
    assert not (tmp_path / "benchmark" / "orbitquant.metrics.csv").exists()
    assert (tmp_path / "assets").is_dir()
    assert not (tmp_path / "assets" / ".gitkeep").exists()
    assert "benchmark/original.metrics.jsonl" not in manifest["checksums"]
    assert "benchmark/orbitquant.metrics.jsonl" not in manifest["checksums"]
    assert "benchmark/original.metrics.csv" not in manifest["checksums"]
    assert "benchmark/orbitquant.metrics.csv" not in manifest["checksums"]
    assert "assets/.gitkeep" not in manifest["checksums"]
    prompts = json.loads((tmp_path / "prompts.json").read_text())
    benchmark_summary = json.loads((tmp_path / "benchmark" / "summary.json").read_text())
    codebook_tensors = load_file(tmp_path / "orbitquant_codebooks.safetensors")
    rotation_tensors = load_file(tmp_path / "orbitquant_rotations.safetensors")
    assert any(name.endswith("packed_weight_indices") for name in tensors)
    assert prompts["prompt_pack"] == "image_visual_v2"
    assert prompts["media_type"] == "image"
    assert len(prompts["prompts"]) >= 10
    assert {item["id"] for item in prompts["prompts"]} >= {
        "simple-object",
        "two-object-composition",
        "counting",
        "color-binding",
        "spatial-relationship",
        "long-prompt",
        "english-text-rendering",
        "cyrillic-text-rendering",
        "style-heavy",
        "occlusion-reflection",
    }
    assert any("КВАНТОВАНИЕ" in item["prompt"] for item in prompts["prompts"])
    assert any("量子の軌道" in item["prompt"] for item in prompts["prompts"])
    assert any("量子轨道" in item["prompt"] for item in prompts["prompts"])
    assert benchmark_summary["status"] == "not_run"
    assert benchmark_summary["source_model_id"] == "example/model"
    assert benchmark_summary["codebook_version"] == 2
    assert benchmark_summary["quantization_device"] == summary.quantization_device
    assert benchmark_summary["weight_quantization_backend"] == summary.weight_quantization_backend
    assert benchmark_summary["quantization_staging_mode"] == summary.quantization_staging_mode
    assert benchmark_summary["synchronize_per_module"] == summary.synchronize_per_module
    assert benchmark_summary["device_transfer_seconds"] >= 0.0
    assert benchmark_summary["module_device_transfer_count"] >= 0
    assert benchmark_summary["source_linear_device_counts"]
    assert not (tmp_path / "benchmark" / "original.metrics.jsonl").exists()
    assert not (tmp_path / "benchmark" / "orbitquant.metrics.jsonl").exists()
    assert not (tmp_path / "benchmark" / "original.metrics.csv").exists()
    assert not (tmp_path / "benchmark" / "orbitquant.metrics.csv").exists()
    assert any(name.endswith(".centroids") for name in codebook_tensors)
    assert any(name.endswith(".permutation") for name in rotation_tensors)


def test_auto_policy_artifact_records_resolved_policy_and_loads_prequantized_flux_modules(
    tmp_path,
):
    torch.manual_seed(0)
    source = FluxTransformer2DArtifactModel()
    config = OrbitQuantConfig(block_size=4, target_policy="auto")
    summary = quantize_linear_modules(source, config)

    save_orbitquant_artifact(
        source,
        tmp_path,
        config=config,
        source_model_id="black-forest-labs/FLUX.1-schnell",
        source_revision="abc123",
        source_license="apache-2.0",
        summary=summary,
    )

    quantization_config = json.loads((tmp_path / "quantization_config.json").read_text())
    manifest = json.loads((tmp_path / "orbitquant_manifest.json").read_text())
    model_index = json.loads((tmp_path / "model_index.json").read_text())
    prompts = json.loads((tmp_path / "prompts.json").read_text())
    benchmark_summary = json.loads((tmp_path / "benchmark" / "summary.json").read_text())

    assert quantization_config["target_policy"] == "flux"
    assert manifest["target_policy"] == "flux"
    assert model_index["target_policy"] == "flux"
    assert prompts["target_policy"] == "flux"
    assert benchmark_summary["target_policy"] == "flux"
    assert manifest["adaln_modules"] == ["transformer_blocks.0.norm1.linear"]
    assert manifest["quantized_modules"] == ["transformer_blocks.0.attn.to_q"]

    inventory_path = tmp_path / "policy-inventory.json"
    inventory_payload = {
        "source_model_id": "black-forest-labs/FLUX.1-schnell",
        "source_revision": "abc123",
        "component": "transformer",
        "load_mode": "config",
        "pipeline_class": None,
        "component_class": "FluxTransformer2DArtifactModel",
        **inspect_linear_module_policy(FluxTransformer2DArtifactModel(), config),
    }
    inventory_path.write_text(json.dumps(inventory_payload, indent=2) + "\n")
    inventory_validation = validate_artifact_policy_inventory(tmp_path, inventory_path)

    assert inventory_validation["valid"] is True
    assert inventory_validation["target_policy"] == "flux"
    assert inventory_validation["quantized_module_count"] == 1
    assert inventory_validation["adaln_module_count"] == 1

    restored = FluxTransformer2DArtifactModel()
    loaded_manifest = load_orbitquant_artifact(restored, tmp_path)

    assert loaded_manifest.target_policy == "flux"
    assert isinstance(restored.transformer_blocks[0]["norm1"]["linear"], RTNInt4Linear)
    assert isinstance(restored.transformer_blocks[0]["attn"]["to_q"], OrbitQuantLinear)
    assert isinstance(restored.proj_out, torch.nn.Linear)


def test_saved_artifact_contains_no_activation_calibration_state_and_deduplicates_basis(
    tmp_path,
):
    model = TinySharedCodebookArtifactModel()
    config = OrbitQuantConfig(weight_bits=4, activation_bits=3, block_size=4)
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

    model_tensors = load_file(tmp_path / "model.safetensors")
    codebook_tensors = load_file(tmp_path / "orbitquant_codebooks.safetensors")
    rotation_tensors = load_file(tmp_path / "orbitquant_rotations.safetensors")

    assert set(model_tensors) == {
        "transformer_blocks.0.attn.to_k.bias",
        "transformer_blocks.0.attn.to_k.packed_weight_indices",
        "transformer_blocks.0.attn.to_k.row_norms",
        "transformer_blocks.0.attn.to_q.bias",
        "transformer_blocks.0.attn.to_q.packed_weight_indices",
        "transformer_blocks.0.attn.to_q.row_norms",
    }
    assert set(codebook_tensors) == {
        "dim8_bits3.boundaries",
        "dim8_bits3.centroids",
        "dim8_bits4.boundaries",
        "dim8_bits4.centroids",
    }
    assert set(rotation_tensors) == {
        "dim8_seed0_block4.inverse_permutation",
        "dim8_seed0_block4.normalization",
        "dim8_seed0_block4.permutation",
        "dim8_seed0_block4.signs",
    }


def test_sha256sums_ignore_huggingface_local_dir_cache_metadata(tmp_path):
    artifact_cache = tmp_path / ".cache" / "huggingface" / "download"
    artifact_cache.mkdir(parents=True)
    cache_metadata = artifact_cache / ".gitattributes.metadata"
    cache_metadata.write_text("transient hub metadata", encoding="utf-8")
    payload = tmp_path / "model.safetensors"
    payload.write_bytes(b"model")
    (tmp_path / ".gitattributes").write_text("*.safetensors filter=lfs\n", encoding="utf-8")

    write_sha256sums(tmp_path)
    entries = read_sha256sums(tmp_path / "SHA256SUMS")

    assert "model.safetensors" in entries
    assert ".gitattributes" not in entries
    assert ".cache/huggingface/download/.gitattributes.metadata" not in entries

    stale_sums = tmp_path / "SHA256SUMS"
    stale_sums.write_text(
        "\n".join(
            [
                f"{entries['model.safetensors']}  model.safetensors",
                "0" * 64 + "  .gitattributes",
                "0" * 64 + "  .cache/huggingface/download/.gitattributes.metadata",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    validated = validate_sha256sums(tmp_path)
    assert validated[".gitattributes"] == "0" * 64
    assert validated[".cache/huggingface/download/.gitattributes.metadata"] == "0" * 64


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
    config = OrbitQuantConfig(block_size=4, adaln_group_size=32)
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

    assert "benchmark/original.metrics.jsonl" not in result["required_files"]
    assert "benchmark/orbitquant.metrics.jsonl" not in result["required_files"]
    assert "benchmark/original.metrics.csv" not in result["required_files"]
    assert "benchmark/orbitquant.metrics.csv" not in result["required_files"]
    assert "assets/.gitkeep" not in result["required_files"]
    assert "model_index.json" in result["required_files"]
    assert result["checksum_validation"] == "checked"
    assert result["sha256sums_validation"] == "checked"
    assert result["sha256sums_entry_count"] > result["tensor_count"]
    assert result["tensor_validation"] == "checked"
    assert result["adaln_group_size"] == 32


def test_validate_orbitquant_artifact_can_skip_heavy_checksum_and_tensor_passes(tmp_path):
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

    result = validate_orbitquant_artifact(
        tmp_path,
        validate_checksums_enabled=False,
        validate_tensors=False,
    )

    assert result["valid"] is True
    assert result["checksum_validation"] == "skipped"
    assert result["sha256sums_validation"] == "skipped"
    assert result["tensor_validation"] == "skipped"


def test_validate_orbitquant_artifact_rejects_config_manifest_drift_without_checksums(
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
    config_path = tmp_path / "quantization_config.json"
    payload = json.loads(config_path.read_text())
    payload["runtime_mode"] = "debug_no_activation_quant"
    config_path.write_text(json.dumps(payload, indent=2) + "\n")

    try:
        validate_orbitquant_artifact(
            tmp_path,
            validate_checksums_enabled=False,
            validate_tensors=False,
        )
    except RuntimeError as exc:
        assert "quantization_config mismatch" in str(exc)
        assert "runtime_mode" in str(exc)
    else:
        raise AssertionError("validate_orbitquant_artifact accepted drifted metadata")


def test_repair_artifact_metadata_updates_provenance_and_checksums(tmp_path):
    source = TinyArtifactModel()
    config = OrbitQuantConfig(block_size=4)
    summary = quantize_linear_modules(source, config, quantization_device=None)
    save_orbitquant_artifact(
        source,
        tmp_path,
        config=config,
        source_model_id="example/model",
        source_revision="abc123",
        source_license="apache-2.0",
        summary=summary,
    )
    manifest_path = tmp_path / "orbitquant_manifest.json"
    model_index_path = tmp_path / "model_index.json"
    benchmark_path = tmp_path / "benchmark" / "summary.json"
    manifest_payload = json.loads(manifest_path.read_text())
    manifest_payload["quantization_device"] = None
    manifest_payload["weight_quantization_backend"] = None
    manifest_payload["quantization_staging_mode"] = None
    manifest_path.write_text(json.dumps(manifest_payload, indent=2) + "\n")
    model_index_payload = json.loads(model_index_path.read_text())
    model_index_payload.pop("quantization_device")
    model_index_payload.pop("weight_quantization_backend")
    model_index_payload.pop("quantization_staging_mode")
    model_index_path.write_text(json.dumps(model_index_payload, indent=2) + "\n")
    benchmark_payload = json.loads(benchmark_path.read_text())
    benchmark_payload.pop("quantization_device")
    benchmark_payload.pop("weight_quantization_backend")
    benchmark_payload.pop("quantization_staging_mode")
    benchmark_path.write_text(json.dumps(benchmark_payload, indent=2) + "\n")
    manifest_payload["checksums"]["model_index.json"] = sha256_file(model_index_path)
    manifest_payload["checksums"]["benchmark/summary.json"] = sha256_file(benchmark_path)
    manifest_path.write_text(json.dumps(manifest_payload, indent=2) + "\n")
    write_sha256sums_from_manifest(tmp_path, manifest_payload["checksums"])

    result = repair_artifact_metadata(
        tmp_path,
        quantization_device="cuda",
        weight_quantization_backend="triton_cuda",
        quantization_staging_mode="component",
        validate_tensors=False,
    )

    validation = validate_orbitquant_artifact(tmp_path, validate_tensors=False)
    manifest = json.loads(manifest_path.read_text())
    model_index = json.loads(model_index_path.read_text())
    benchmark = json.loads(benchmark_path.read_text())
    sha_entries = read_sha256sums(tmp_path / "SHA256SUMS")
    readme = (tmp_path / "README.md").read_text()
    assert result["before"]["quantization_device"] is None
    assert result["after"]["quantization_device"] == "cuda"
    assert validation["quantization_device"] == "cuda"
    assert validation["weight_quantization_backend"] == "triton_cuda"
    assert validation["quantization_staging_mode"] == "component"
    assert manifest["quantization_device"] == "cuda"
    assert manifest["weight_quantization_backend"] == "triton_cuda"
    assert manifest["quantization_staging_mode"] == "component"
    assert model_index["quantization_device"] == "cuda"
    assert model_index["weight_quantization_backend"] == "triton_cuda"
    assert model_index["quantization_staging_mode"] == "component"
    assert benchmark["quantization_device"] == "cuda"
    assert benchmark["weight_quantization_backend"] == "triton_cuda"
    assert benchmark["quantization_staging_mode"] == "component"
    assert sha_entries["orbitquant_manifest.json"] == sha256_file(manifest_path)
    assert sha_entries["README.md"] == sha256_file(tmp_path / "README.md")
    assert "- Quantization device: `cuda`" in readme
    assert "- Weight quantization backend: `triton_cuda`" in readme
    assert "Quantization staging" not in readme


def test_validate_orbitquant_artifact_rejects_corrupted_readme_checksum(tmp_path):
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
    with (tmp_path / "README.md").open("a", encoding="utf-8") as handle:
        handle.write("\ncorruption\n")

    try:
        validate_orbitquant_artifact(tmp_path)
    except RuntimeError as exc:
        assert "SHA256SUMS mismatch for README.md" in str(exc)
    else:
        raise AssertionError("validate_orbitquant_artifact accepted a corrupted README")


def test_validate_orbitquant_artifact_rejects_corrupted_manifest_checksum(tmp_path):
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
    manifest_path = tmp_path / "orbitquant_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["extra_ignored_field"] = "corruption"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    try:
        validate_orbitquant_artifact(tmp_path)
    except RuntimeError as exc:
        assert "SHA256SUMS mismatch for orbitquant_manifest.json" in str(exc)
    else:
        raise AssertionError("validate_orbitquant_artifact accepted a corrupted manifest")


def test_validate_orbitquant_artifact_rejects_model_index_manifest_mismatch(tmp_path):
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
    manifest_path = tmp_path / "orbitquant_manifest.json"
    model_index_path = tmp_path / "model_index.json"
    manifest = json.loads(manifest_path.read_text())
    model_index = json.loads(model_index_path.read_text())
    model_index["source_model_id"] = "example/other-model"
    model_index_path.write_text(json.dumps(model_index, indent=2) + "\n")
    manifest["checksums"]["model_index.json"] = sha256_file(model_index_path)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    write_sha256sums_from_manifest(tmp_path, manifest["checksums"])

    try:
        validate_orbitquant_artifact(tmp_path)
    except RuntimeError as exc:
        assert "model_index mismatch" in str(exc)
        assert "source_model_id" in str(exc)
    else:
        raise AssertionError("validate_orbitquant_artifact accepted mismatched model_index")


def test_validate_orbitquant_artifact_rejects_corrupted_codebook_semantics(tmp_path):
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
    codebook_path = tmp_path / "orbitquant_codebooks.safetensors"
    tensors = load_file(codebook_path)
    tensors["dim8_bits4.boundaries"] = torch.flip(tensors["dim8_bits4.boundaries"], dims=[0])
    save_file(tensors, codebook_path)
    manifest_path = tmp_path / "orbitquant_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["checksums"]["orbitquant_codebooks.safetensors"] = sha256_file(codebook_path)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    write_sha256sums_from_manifest(tmp_path, manifest["checksums"])

    try:
        validate_orbitquant_artifact(tmp_path)
    except RuntimeError as exc:
        assert "artifact codebook tensor mismatch" in str(exc)
    else:
        raise AssertionError("validate_orbitquant_artifact accepted a corrupted codebook")


def test_validate_orbitquant_artifact_rejects_wrong_versioned_codebook(tmp_path):
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
    codebook_path = tmp_path / "orbitquant_codebooks.safetensors"
    tensors = load_file(codebook_path)
    centroids = tensors["dim8_bits4.centroids"] * 0.99
    tensors["dim8_bits4.centroids"] = centroids
    tensors["dim8_bits4.boundaries"] = (centroids[:-1] + centroids[1:]) / 2
    save_file(tensors, codebook_path)
    manifest_path = tmp_path / "orbitquant_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["checksums"]["orbitquant_codebooks.safetensors"] = sha256_file(codebook_path)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    write_sha256sums_from_manifest(tmp_path, manifest["checksums"])

    try:
        validate_orbitquant_artifact(tmp_path)
    except RuntimeError as exc:
        assert "do not match codebook version 2" in str(exc)
    else:
        raise AssertionError("validator accepted centroids from a different algorithm")


def test_validate_orbitquant_artifact_rejects_corrupted_rotation_semantics(tmp_path):
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
    rotation_path = tmp_path / "orbitquant_rotations.safetensors"
    tensors = load_file(rotation_path)
    tensors["dim8_seed0_block4.signs"] = torch.zeros_like(tensors["dim8_seed0_block4.signs"])
    save_file(tensors, rotation_path)
    manifest_path = tmp_path / "orbitquant_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["checksums"]["orbitquant_rotations.safetensors"] = sha256_file(rotation_path)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    write_sha256sums_from_manifest(tmp_path, manifest["checksums"])

    try:
        validate_orbitquant_artifact(tmp_path)
    except RuntimeError as exc:
        assert "artifact rotation tensor mismatch" in str(exc)
    else:
        raise AssertionError("validate_orbitquant_artifact accepted a corrupted rotation")


def test_validate_orbitquant_artifact_rejects_valid_but_wrong_runtime_rotation(tmp_path):
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
    rotation_path = tmp_path / "orbitquant_rotations.safetensors"
    tensors = load_file(rotation_path)
    tensors["dim8_seed0_block4.signs"] = -tensors["dim8_seed0_block4.signs"]
    save_file(tensors, rotation_path)
    manifest_path = tmp_path / "orbitquant_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["checksums"]["orbitquant_rotations.safetensors"] = sha256_file(rotation_path)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    write_sha256sums_from_manifest(tmp_path, manifest["checksums"])

    try:
        validate_orbitquant_artifact(tmp_path)
    except RuntimeError as exc:
        assert "signs do not match runtime rotation" in str(exc)
    else:
        raise AssertionError("validator accepted a different valid RPBH draw")


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


def test_record_artifact_metrics_refreshes_model_card_release_metrics(tmp_path):
    source = TinyArtifactModel()
    config = OrbitQuantConfig(block_size=4, target_policy="flux")
    summary = quantize_linear_modules(source, config)
    save_orbitquant_artifact(
        source,
        tmp_path,
        config=config,
        source_model_id="black-forest-labs/FLUX.1-schnell",
        source_revision="abc123",
        source_license="apache-2.0",
        summary=summary,
    )

    record_artifact_metrics(
        tmp_path,
        split="original",
        metrics={
            "geneval_overall": 0.74,
            "geneval_per_task_single_object": 0.9,
        },
        metadata={"suite": "flux1-schnell-native", "seed": 0, "bit_setting": "BF16"},
    )
    record_artifact_metrics(
        tmp_path,
        split="orbitquant",
        metrics={
            "geneval_overall": 0.71,
            "geneval_per_task_single_object": 0.88,
        },
        metadata={"suite": "flux1-schnell-native", "seed": 0, "bit_setting": "W4A4"},
    )

    result = validate_orbitquant_artifact(tmp_path)
    readme = (tmp_path / "README.md").read_text()
    sha_entries = read_sha256sums(tmp_path / "SHA256SUMS")

    assert result["valid"] is True
    assert "Release-grade GenEval metrics: included below." in readme
    assert "Release-grade GenEval metrics: not included" not in readme
    assert "| `geneval_overall` | `0.74` | `0.71` |" in readme
    assert "| `geneval_per_task_single_object` | `0.9` | `0.88` |" in readme
    assert sha_entries["README.md"] == sha256_file(tmp_path / "README.md")


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
    assert not (tmp_path / "benchmark" / "orbitquant.metrics.jsonl").exists()


def test_record_artifact_metrics_can_skip_heavy_checksum_preflight(tmp_path):
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
    original_model_checksum = original_manifest["checksums"]["model.safetensors"]
    with (tmp_path / "model.safetensors").open("ab") as handle:
        handle.write(b"corruption")

    record_artifact_metrics(
        tmp_path,
        split="orbitquant",
        metrics={"generated_samples": 1},
        validate_checksums_enabled=False,
    )

    manifest = json.loads((tmp_path / "orbitquant_manifest.json").read_text())
    sha_lines = (tmp_path / "SHA256SUMS").read_text().splitlines()
    assert manifest["checksums"]["model.safetensors"] == original_model_checksum
    assert f"{original_model_checksum}  model.safetensors" in sha_lines
    try:
        validate_orbitquant_artifact(tmp_path)
    except RuntimeError as exc:
        assert "checksum mismatch for model.safetensors" in str(exc)
    else:
        raise AssertionError("strict artifact validation accepted a stale model checksum")


def test_deferred_artifact_refresh_rebuilds_manifest_and_sha256sums_once(tmp_path):
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
    asset_path = tmp_path / "assets" / "flux2-native_seed7_W4A4_simple-object.png"
    asset_path.write_bytes(b"fake image bytes")

    relative_asset = record_artifact_asset(
        tmp_path,
        asset_path,
        validate_checksums_enabled=False,
        refresh_checksums_enabled=False,
    )
    record_artifact_metrics(
        tmp_path,
        split="orbitquant",
        metrics={"generated_samples": 1},
        metadata={
            "suite": "flux2-native",
            "seed": 7,
            "bit_setting": "W4A4",
            "prompt_record": {"id": "simple-object"},
            "output_path": str(asset_path),
        },
        validate_checksums_enabled=False,
        refresh_checksums_enabled=False,
    )

    stale_manifest = json.loads((tmp_path / "orbitquant_manifest.json").read_text())
    stale_sha = read_sha256sums(tmp_path / "SHA256SUMS")
    assert relative_asset == "assets/flux2-native_seed7_W4A4_simple-object.png"
    assert relative_asset not in stale_manifest["checksums"]
    assert "benchmark/orbitquant.metrics.jsonl" not in stale_manifest["checksums"]
    assert "benchmark/orbitquant.metrics.jsonl" not in stale_sha

    result = refresh_artifact_checksums(tmp_path)

    validation = validate_orbitquant_artifact(tmp_path)
    manifest = json.loads((tmp_path / "orbitquant_manifest.json").read_text())
    sha_entries = read_sha256sums(tmp_path / "SHA256SUMS")
    assert result["checksum_count"] == len(manifest["checksums"])
    assert validation["valid"] is True
    assert manifest["checksums"][relative_asset] == sha256_file(asset_path)
    assert manifest["checksums"]["benchmark/orbitquant.metrics.jsonl"] == sha256_file(
        tmp_path / "benchmark" / "orbitquant.metrics.jsonl"
    )
    assert sha_entries[relative_asset] == manifest["checksums"][relative_asset]
    assert sha_entries["orbitquant_manifest.json"] == sha256_file(
        tmp_path / "orbitquant_manifest.json"
    )
    assert sha_entries["README.md"] == sha256_file(tmp_path / "README.md")


def test_record_artifact_asset_adds_asset_to_manifest_and_validation(tmp_path):
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
    asset_path = tmp_path / "assets" / "flux2-native_seed0_W4A4.png"
    asset_path.write_bytes(b"fake image bytes")

    relative_path = record_artifact_asset(tmp_path, asset_path)

    result = validate_orbitquant_artifact(tmp_path)
    manifest = json.loads((tmp_path / "orbitquant_manifest.json").read_text())
    assert relative_path == "assets/flux2-native_seed0_W4A4.png"
    assert result["checksums"][relative_path] == sha256_file(asset_path)
    assert manifest["checksums"][relative_path] == sha256_file(asset_path)
    assert (
        f"{manifest['checksums'][relative_path]}  assets/flux2-native_seed0_W4A4.png"
        in (tmp_path / "SHA256SUMS").read_text()
    )

    asset_path.write_bytes(b"corrupted")
    try:
        validate_orbitquant_artifact(tmp_path)
    except RuntimeError as exc:
        assert "checksum mismatch for assets/flux2-native_seed0_W4A4.png" in str(exc)
    else:
        raise AssertionError("validate_orbitquant_artifact accepted a corrupted asset")


def test_create_artifact_image_comparisons_pairs_original_and_orbitquant_outputs(tmp_path):
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
    original_path = tmp_path / "assets" / "flux2-native_seed3_original_simple-object.png"
    orbitquant_path = tmp_path / "assets" / "flux2-native_seed3_W4A4_simple-object.png"
    Image.new("RGB", (16, 16), "red").save(original_path)
    Image.new("RGB", (16, 16), "blue").save(orbitquant_path)
    record_artifact_asset(tmp_path, original_path)
    record_artifact_asset(tmp_path, orbitquant_path)
    record_artifact_metrics(
        tmp_path,
        split="original",
        metrics={"generated_samples": 1},
        metadata={
            "suite": "flux2-native",
            "seed": 3,
            "bit_setting": "original",
            "prompt_record": {"id": "simple-object"},
            "output_path": str(original_path),
        },
    )
    record_artifact_metrics(
        tmp_path,
        split="orbitquant",
        metrics={"generated_samples": 1},
        metadata={
            "suite": "flux2-native",
            "seed": 3,
            "bit_setting": "W4A4",
            "prompt_record": {"id": "simple-object"},
            "output_path": str(orbitquant_path),
        },
    )

    comparisons = create_artifact_image_comparisons(tmp_path)

    result = validate_orbitquant_artifact(tmp_path)
    assert comparisons == [
        "assets/original_vs_orbitquant_flux2-native_seed3_W4A4_simple-object.webp"
    ]
    comparison_path = tmp_path / comparisons[0]
    assert comparison_path.is_file()
    assert result["checksums"][comparisons[0]] == sha256_file(comparison_path)
    with Image.open(comparison_path) as sheet:
        assert sheet.size[0] == 32


def test_create_artifact_image_comparisons_can_filter_current_prompt_pack(tmp_path):
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
    for prompt_id in ["simple-object", "geneval-00000-single-object"]:
        original_path = tmp_path / "assets" / f"flux2-native_seed3_original_{prompt_id}.png"
        orbitquant_path = tmp_path / "assets" / f"flux2-native_seed3_W4A4_{prompt_id}.png"
        Image.new("RGB", (16, 16), "red").save(original_path)
        Image.new("RGB", (16, 16), "blue").save(orbitquant_path)
        record_artifact_asset(tmp_path, original_path)
        record_artifact_asset(tmp_path, orbitquant_path)
        record_artifact_metrics(
            tmp_path,
            split="original",
            metrics={"generated_samples": 1},
            metadata={
                "suite": "flux2-native",
                "seed": 3,
                "bit_setting": "original",
                "prompt_record": {"id": prompt_id},
                "output_path": str(original_path),
            },
        )
        record_artifact_metrics(
            tmp_path,
            split="orbitquant",
            metrics={"generated_samples": 1},
            metadata={
                "suite": "flux2-native",
                "seed": 3,
                "bit_setting": "W4A4",
                "prompt_record": {"id": prompt_id},
                "output_path": str(orbitquant_path),
            },
        )

    comparisons = create_artifact_image_comparisons(
        tmp_path,
        comparison_keys={("flux2-native", 3, "simple-object")},
    )

    assert comparisons == [
        "assets/original_vs_orbitquant_flux2-native_seed3_W4A4_simple-object.webp"
    ]
    assert (tmp_path / comparisons[0]).is_file()
    assert not (
        tmp_path
        / "assets"
        / "original_vs_orbitquant_flux2-native_seed3_W4A4_geneval-00000-single-object.webp"
    ).exists()


def test_create_artifact_image_comparisons_pairs_video_contact_sheets(tmp_path):
    source = TinyArtifactModel()
    config = OrbitQuantConfig(block_size=4, target_policy="generic_dit")
    summary = quantize_linear_modules(source, config)
    save_orbitquant_artifact(
        source,
        tmp_path,
        config=config,
        source_model_id="example/wan",
        source_revision="abc123",
        source_license="apache-2.0",
        summary=summary,
    )
    original_contact_sheet = tmp_path / "assets" / "wan-native_seed5_original_contact_sheet.webp"
    orbitquant_contact_sheet = tmp_path / "assets" / "wan-native_seed5_W4A4_contact_sheet.webp"
    Image.new("RGB", (16, 8), "red").save(original_contact_sheet)
    Image.new("RGB", (16, 8), "blue").save(orbitquant_contact_sheet)
    record_artifact_asset(tmp_path, original_contact_sheet)
    record_artifact_asset(tmp_path, orbitquant_contact_sheet)
    record_artifact_metrics(
        tmp_path,
        split="original",
        metrics={"generated_samples": 1, "generated_frames": 81},
        metadata={
            "suite": "wan-native",
            "seed": 5,
            "bit_setting": "original",
            "prompt_record": {"id": "simple-motion"},
            "output_path": str(tmp_path / "assets" / "wan-native_seed5_original.mp4"),
            "asset_paths": [str(original_contact_sheet)],
        },
    )
    record_artifact_metrics(
        tmp_path,
        split="orbitquant",
        metrics={"generated_samples": 1, "generated_frames": 81},
        metadata={
            "suite": "wan-native",
            "seed": 5,
            "bit_setting": "W4A4",
            "prompt_record": {"id": "simple-motion"},
            "output_path": str(tmp_path / "assets" / "wan-native_seed5_W4A4.mp4"),
            "asset_paths": [str(orbitquant_contact_sheet)],
        },
    )

    comparisons = create_artifact_image_comparisons(tmp_path)

    result = validate_orbitquant_artifact(tmp_path)
    assert comparisons == [
        "assets/original_vs_orbitquant_wan-native_seed5_W4A4_simple-motion.webp"
    ]
    comparison_path = tmp_path / comparisons[0]
    assert comparison_path.is_file()
    assert result["checksums"][comparisons[0]] == sha256_file(comparison_path)
    with Image.open(comparison_path) as sheet:
        assert sheet.size[0] == 32


def test_create_artifact_image_comparisons_skips_unpaired_outputs(tmp_path):
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
    original_path = tmp_path / "assets" / "flux2-native_seed3_original_simple-object.png"
    Image.new("RGB", (16, 16), "red").save(original_path)
    record_artifact_asset(tmp_path, original_path)
    record_artifact_metrics(
        tmp_path,
        split="original",
        metrics={"generated_samples": 1},
        metadata={
            "suite": "flux2-native",
            "seed": 3,
            "bit_setting": "original",
            "prompt_record": {"id": "simple-object"},
            "output_path": str(original_path),
        },
    )

    assert create_artifact_image_comparisons(tmp_path) == []


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


def test_load_orbitquant_artifact_rejects_sha256sums_mismatch(tmp_path):
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
    with (tmp_path / "README.md").open("a", encoding="utf-8") as handle:
        handle.write("\ncorruption\n")

    try:
        load_orbitquant_artifact(TinyArtifactModel(), tmp_path)
    except RuntimeError as exc:
        assert "SHA256SUMS mismatch for README.md" in str(exc)
    else:
        raise AssertionError("load_orbitquant_artifact accepted a corrupted SHA256SUMS target")


def test_load_orbitquant_artifact_can_skip_checksum_validation_for_trusted_local_runs(tmp_path):
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
    (tmp_path / "benchmark" / "summary.json").write_text('{"status":"changed"}\n')

    restored = TinyArtifactModel()
    manifest = load_orbitquant_artifact(restored, tmp_path, validate_checksums=False)

    assert manifest.source_model_id == "example/model"
    assert isinstance(restored.transformer_blocks[0]["attn"]["to_q"], OrbitQuantLinear)


def test_load_orbitquant_artifact_rejects_config_manifest_drift_without_checksums(
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
    config_path = tmp_path / "quantization_config.json"
    payload = json.loads(config_path.read_text())
    payload["activation_eps"] = 1e-4
    config_path.write_text(json.dumps(payload, indent=2) + "\n")

    try:
        load_orbitquant_artifact(
            TinyArtifactModel(),
            tmp_path,
            validate_checksums=False,
        )
    except RuntimeError as exc:
        assert "quantization_config mismatch" in str(exc)
        assert "activation_eps" in str(exc)
    else:
        raise AssertionError("load_orbitquant_artifact accepted drifted metadata")


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
