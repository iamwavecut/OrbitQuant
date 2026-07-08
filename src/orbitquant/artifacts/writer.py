from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from orbitquant.artifacts.checksums import sha256_file, write_sha256sums
from orbitquant.artifacts.manifest import OrbitQuantManifest
from orbitquant.artifacts.model_card import render_model_card
from orbitquant.config import OrbitQuantConfig
from orbitquant.eval.prompts import default_prompt_payload
from orbitquant.layers import OrbitQuantLinear
from orbitquant.policies import resolve_target_policy


def _metadata_config(model: torch.nn.Module, config: OrbitQuantConfig) -> OrbitQuantConfig:
    resolved_policy = resolve_target_policy(model, config)
    if resolved_policy == config.target_policy:
        return config
    values = config.to_dict()
    values["target_policy"] = resolved_policy
    return OrbitQuantConfig.from_dict(values)


def _module_shapes(model: torch.nn.Module) -> dict[str, list[int]]:
    shapes: dict[str, list[int]] = {}
    for name, tensor in model.state_dict().items():
        shapes[name] = list(tensor.shape)
    return shapes


def _summary_list(summary: Any, field: str) -> list[str]:
    value = getattr(summary, field, [])
    return list(value)


def _codebook_tensors(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for module in model.modules():
        if not isinstance(module, OrbitQuantLinear):
            continue
        for codebook in (module.weight_codebook, module.activation_codebook):
            prefix = f"dim{codebook.dim}_bits{codebook.bits}"
            tensors[f"{prefix}.centroids"] = codebook.centroids.detach().cpu()
            tensors[f"{prefix}.boundaries"] = codebook.boundaries.detach().cpu()
    return tensors


def _rotation_tensors(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for module in model.modules():
        if not isinstance(module, OrbitQuantLinear):
            continue
        rotation = module.rotation
        prefix = f"dim{rotation.dim}_seed{rotation.seed}_block{rotation.block_size}"
        tensors[f"{prefix}.permutation"] = rotation.permutation.detach().cpu()
        tensors[f"{prefix}.inverse_permutation"] = rotation.inverse_permutation.detach().cpu()
        tensors[f"{prefix}.signs"] = rotation.signs.detach().cpu()
        tensors[f"{prefix}.normalization"] = torch.tensor(
            [rotation.normalization], dtype=torch.float32
        )
    return tensors


def _model_index_payload(
    *,
    config: OrbitQuantConfig,
    source_model_id: str,
    source_revision: str,
    source_license: str,
    component: str,
    quantization_device: str,
    weight_quantization_backend: str,
    quantization_staging_mode: str,
) -> dict[str, Any]:
    return {
        "_class_name": "OrbitQuantComponentArtifact",
        "artifact_format": "orbitquant-v1",
        "quant_method": "orbitquant",
        "source_model_id": source_model_id,
        "source_revision": source_revision,
        "source_license": source_license,
        "component": component,
        "weight_name": "model.safetensors",
        "quantization_config": "quantization_config.json",
        "manifest": "orbitquant_manifest.json",
        "codebooks": "orbitquant_codebooks.safetensors",
        "rotations": "orbitquant_rotations.safetensors",
        "weight_bits": config.weight_bits,
        "activation_bits": config.activation_bits,
        "target_policy": config.target_policy,
        "runtime_mode": config.runtime_mode,
        "activation_kernel_backend": config.activation_kernel_backend,
        "activation_eps": config.activation_eps,
        "quantization_device": quantization_device,
        "weight_quantization_backend": weight_quantization_backend,
        "quantization_staging_mode": quantization_staging_mode,
    }


def save_orbitquant_artifact(
    model: torch.nn.Module,
    output_dir: str | Path,
    *,
    config: OrbitQuantConfig,
    source_model_id: str,
    source_revision: str,
    source_license: str,
    summary: Any,
    component: str = "transformer",
) -> OrbitQuantManifest:
    config = _metadata_config(model, config)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tensor_path = output_path / "model.safetensors"
    state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    save_file(state_dict, tensor_path)
    codebook_path = output_path / "orbitquant_codebooks.safetensors"
    save_file(_codebook_tensors(model), codebook_path)
    rotation_path = output_path / "orbitquant_rotations.safetensors"
    save_file(_rotation_tensors(model), rotation_path)
    config_path = output_path / "quantization_config.json"
    config_path.write_text(json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8")
    model_index_path = output_path / "model_index.json"
    model_index_path.write_text(
        json.dumps(
            _model_index_payload(
                config=config,
                source_model_id=source_model_id,
                source_revision=source_revision,
                source_license=source_license,
                component=component,
                quantization_device=getattr(summary, "quantization_device", "unknown"),
                weight_quantization_backend=getattr(
                    summary, "weight_quantization_backend", "unknown"
                ),
                quantization_staging_mode=getattr(
                    summary, "quantization_staging_mode", "unknown"
                ),
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    prompts_path = output_path / "prompts.json"
    prompts_path.write_text(
        json.dumps(default_prompt_payload(config.target_policy), indent=2) + "\n",
        encoding="utf-8",
    )
    benchmark_path = output_path / "benchmark" / "summary.json"
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark_summary = {
        "status": "not_run",
        "source_model_id": source_model_id,
        "source_revision": source_revision,
        "weight_bits": config.weight_bits,
        "activation_bits": config.activation_bits,
        "target_policy": config.target_policy,
        "runtime_mode": config.runtime_mode,
        "activation_kernel_backend": config.activation_kernel_backend,
        "activation_eps": config.activation_eps,
        "quantization_device": getattr(summary, "quantization_device", "unknown"),
        "weight_quantization_backend": getattr(
            summary, "weight_quantization_backend", "unknown"
        ),
        "quantization_staging_mode": getattr(
            summary, "quantization_staging_mode", "unknown"
        ),
        "synchronize_per_module": getattr(summary, "synchronize_per_module", False),
        "quantization_elapsed_seconds": getattr(summary, "elapsed_seconds", 0.0),
        "orbitquant_seconds": getattr(summary, "orbitquant_seconds", 0.0),
        "adaln_seconds": getattr(summary, "adaln_seconds", 0.0),
        "device_transfer_seconds": getattr(summary, "device_transfer_seconds", 0.0),
        "module_device_transfer_count": getattr(
            summary, "module_device_transfer_count", 0
        ),
        "source_linear_device_counts": getattr(
            summary, "source_linear_device_counts", {}
        ),
        "quantized_buffer_device_counts": getattr(
            summary, "quantized_buffer_device_counts", {}
        ),
    }
    benchmark_path.write_text(json.dumps(benchmark_summary, indent=2) + "\n", encoding="utf-8")
    (output_path / "assets").mkdir(parents=True, exist_ok=True)

    skipped = _summary_list(summary, "skipped_modules")
    checksums = {
        "model.safetensors": sha256_file(tensor_path),
        "orbitquant_codebooks.safetensors": sha256_file(codebook_path),
        "orbitquant_rotations.safetensors": sha256_file(rotation_path),
        "quantization_config.json": sha256_file(config_path),
        "model_index.json": sha256_file(model_index_path),
        "prompts.json": sha256_file(prompts_path),
        "benchmark/summary.json": sha256_file(benchmark_path),
    }
    manifest = OrbitQuantManifest.from_config(
        config,
        source_model_id=source_model_id,
        source_revision=source_revision,
        source_license=source_license,
        quantized_modules=_summary_list(summary, "quantized_modules"),
        adaln_modules=_summary_list(summary, "adaln_modules"),
        skipped_modules=skipped,
        module_shapes=_module_shapes(model),
        checksums=checksums,
        quantization_device=getattr(summary, "quantization_device", "unknown"),
        weight_quantization_backend=getattr(
            summary, "weight_quantization_backend", "unknown"
        ),
        quantization_staging_mode=getattr(
            summary, "quantization_staging_mode", "unknown"
        ),
    )

    (output_path / "orbitquant_manifest.json").write_text(
        json.dumps(manifest.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    (output_path / "README.md").write_text(
        render_model_card(manifest, benchmark_summary=benchmark_summary),
        encoding="utf-8",
    )
    write_sha256sums(output_path)
    return manifest
