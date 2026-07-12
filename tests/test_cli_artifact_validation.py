import json

import pytest
import torch

import orbitquant.cli.main as cli_main
from orbitquant.artifacts import save_orbitquant_artifact
from orbitquant.artifacts.checksums import write_sha256sums_from_manifest
from orbitquant.cli.main import main
from orbitquant.config import OrbitQuantConfig
from orbitquant.modeling import quantize_linear_modules


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


@pytest.mark.parametrize("runtime_mode", ["triton_packed_matmul", "native_packed_matmul"])
def test_cli_validate_artifact_rejects_unexpected_runtime_mode(capsys, tmp_path, runtime_mode):
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

    assert (
        main(
            [
                "validate-artifact",
                "--artifact",
                str(tmp_path),
                "--runtime-mode",
                runtime_mode,
            ]
        )
        == 1
    )

    assert "runtime_mode mismatch" in capsys.readouterr().err


def test_cli_validate_artifact_checks_policy_inventory(capsys, tmp_path):
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
    inventory_path = tmp_path / "policy-inventory.json"
    inventory_path.write_text(
        json.dumps(
            {
                "source_model_id": "example/model",
                "target_policy": "generic_dit",
                "component": "transformer",
                "load_mode": "config",
                "linear_module_count": 1,
                "action_counts": {
                    "orbitquant": 1,
                    "adaln_int4_rtn": 0,
                    "bf16_skip": 0,
                },
                "quantized_modules": summary.quantized_modules,
                "adaln_modules": summary.adaln_modules,
                "skipped_modules": summary.skipped_modules,
            }
        )
        + "\n"
    )

    assert (
        main(
            [
                "validate-artifact",
                "--artifact",
                str(tmp_path),
                "--policy-inventory",
                str(inventory_path),
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    policy_validation = output["policy_inventory_validation"]
    assert policy_validation["valid"] is True
    assert policy_validation["quantized_module_count"] == 1
    assert policy_validation["inventory_path"] == str(inventory_path)


def test_cli_validate_artifact_rejects_policy_inventory_mismatch(tmp_path):
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
    inventory_path = tmp_path / "policy-inventory.json"
    inventory_path.write_text(
        json.dumps(
            {
                "source_model_id": "example/model",
                "target_policy": "generic_dit",
                "quantized_modules": ["transformer_blocks.0.attn.to_k"],
                "adaln_modules": [],
                "skipped_modules": [],
            }
        )
        + "\n"
    )

    with pytest.raises(RuntimeError, match="policy inventory mismatch"):
        main(
            [
                "validate-artifact",
                "--artifact",
                str(tmp_path),
                "--policy-inventory",
                str(inventory_path),
            ]
        )


def test_cli_validate_artifact_rejects_policy_inventory_component_mismatch(tmp_path):
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
        component="transformer",
    )
    inventory_path = tmp_path / "policy-inventory.json"
    inventory_path.write_text(
        json.dumps(
            {
                "source_model_id": "example/model",
                "target_policy": "generic_dit",
                "component": "denoiser",
                "quantized_modules": summary.quantized_modules,
                "adaln_modules": summary.adaln_modules,
                "skipped_modules": summary.skipped_modules,
            }
        )
        + "\n"
    )

    with pytest.raises(RuntimeError, match="component"):
        main(
            [
                "validate-artifact",
                "--artifact",
                str(tmp_path),
                "--policy-inventory",
                str(inventory_path),
            ]
        )


def test_cli_validate_artifact_rejects_manifest_policy_config_mismatch(tmp_path):
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
    manifest_path = tmp_path / "orbitquant_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["target_policy"] = "flux"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    write_sha256sums_from_manifest(tmp_path, manifest["checksums"])
    inventory_path = tmp_path / "policy-inventory.json"
    inventory_path.write_text(
        json.dumps(
            {
                "source_model_id": "example/model",
                "target_policy": "flux",
                "component": "transformer",
                "quantized_modules": summary.quantized_modules,
                "adaln_modules": summary.adaln_modules,
                "skipped_modules": summary.skipped_modules,
            }
        )
        + "\n"
    )

    with pytest.raises(RuntimeError, match="quantization_config mismatch.*target_policy"):
        main(
            [
                "validate-artifact",
                "--artifact",
                str(tmp_path),
                "--policy-inventory",
                str(inventory_path),
            ]
        )


def test_cli_repair_artifact_metadata_wires_provenance_options(capsys, tmp_path, monkeypatch):
    seen = {}

    def fake_repair_artifact_metadata(artifact, **kwargs):
        seen["artifact"] = artifact
        seen["kwargs"] = kwargs
        return {
            "artifact_dir": artifact,
            "updated": {
                "quantization_device": kwargs["quantization_device"],
                "weight_quantization_backend": kwargs["weight_quantization_backend"],
                "quantization_staging_mode": kwargs["quantization_staging_mode"],
            },
        }

    monkeypatch.setattr(cli_main, "repair_artifact_metadata", fake_repair_artifact_metadata)

    assert (
        main(
            [
                "repair-artifact-metadata",
                "--artifact",
                str(tmp_path),
                "--quantization-device",
                "cuda",
                "--weight-quantization-backend",
                "triton_cuda",
                "--quantization-staging-mode",
                "component",
                "--skip-tensor-validation",
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert output["updated"]["quantization_device"] == "cuda"
    assert output["updated"]["weight_quantization_backend"] == "triton_cuda"
    assert output["updated"]["quantization_staging_mode"] == "component"
    assert seen == {
        "artifact": str(tmp_path),
        "kwargs": {
            "quantization_device": "cuda",
            "weight_quantization_backend": "triton_cuda",
            "quantization_staging_mode": "component",
            "validate_tensors": False,
        },
    }
