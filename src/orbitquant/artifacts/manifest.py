from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from orbitquant.config import OrbitQuantConfig


@dataclass(frozen=True)
class OrbitQuantManifest:
    source_model_id: str
    source_revision: str
    source_license: str
    weight_bits: int
    activation_bits: int
    rotation_seed: int
    block_size: int | str
    block_size_policy: str
    codebook_version: int
    target_policy: str
    runtime_mode: str
    activation_kernel_backend: str
    activation_eps: float = 1e-10
    adaln_group_size: int = 64
    quantization_device: str = "unknown"
    weight_quantization_backend: str = "unknown"
    quantization_staging_mode: str = "unknown"
    quantized_modules: list[str] = field(default_factory=list)
    adaln_modules: list[str] = field(default_factory=list)
    skipped_modules: list[str] = field(default_factory=list)
    module_shapes: dict[str, list[int]] = field(default_factory=dict)
    checksums: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_config(
        cls,
        config: OrbitQuantConfig,
        *,
        source_model_id: str,
        source_revision: str,
        source_license: str,
        quantized_modules: list[str],
        skipped_modules: list[str],
        adaln_modules: list[str] | None = None,
        module_shapes: dict[str, list[int]] | None = None,
        checksums: dict[str, str] | None = None,
        quantization_device: str = "unknown",
        weight_quantization_backend: str = "unknown",
        quantization_staging_mode: str = "unknown",
    ) -> OrbitQuantManifest:
        return cls(
            source_model_id=source_model_id,
            source_revision=source_revision,
            source_license=source_license,
            weight_bits=config.weight_bits,
            activation_bits=config.activation_bits,
            rotation_seed=config.rotation_seed,
            block_size=config.block_size,
            block_size_policy="largest_power_of_two_dividing_dim"
            if config.block_size == "paper"
            else "explicit",
            codebook_version=1,
            target_policy=config.target_policy,
            runtime_mode=config.runtime_mode,
            activation_kernel_backend=config.activation_kernel_backend,
            activation_eps=config.activation_eps,
            adaln_group_size=config.adaln_group_size,
            quantization_device=quantization_device,
            weight_quantization_backend=weight_quantization_backend,
            quantization_staging_mode=quantization_staging_mode,
            quantized_modules=list(quantized_modules),
            adaln_modules=[] if adaln_modules is None else list(adaln_modules),
            skipped_modules=list(skipped_modules),
            module_shapes={} if module_shapes is None else dict(module_shapes),
            checksums={} if checksums is None else dict(checksums),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_format": "orbitquant-v1",
            "source_model_id": self.source_model_id,
            "source_revision": self.source_revision,
            "source_license": self.source_license,
            "quant_method": "orbitquant",
            "paper": "https://arxiv.org/abs/2607.02461",
            "weight_bits": self.weight_bits,
            "activation_bits": self.activation_bits,
            "rotation": "rpbh",
            "rotation_seed": self.rotation_seed,
            "block_size": self.block_size,
            "block_size_policy": self.block_size_policy,
            "codebook": "lloyd_max",
            "codebook_version": self.codebook_version,
            "row_norm_dtype": "bfloat16",
            "runtime_mode": self.runtime_mode,
            "activation_kernel_backend": self.activation_kernel_backend,
            "activation_eps": self.activation_eps,
            "adaln_group_size": self.adaln_group_size,
            "quantization_device": self.quantization_device,
            "weight_quantization_backend": self.weight_quantization_backend,
            "quantization_staging_mode": self.quantization_staging_mode,
            "target_policy": self.target_policy,
            "adaln_policy": f"int4_rtn_group{self.adaln_group_size}_bf16_activation",
            "quantized_modules": self.quantized_modules,
            "adaln_modules": self.adaln_modules,
            "skipped_modules": self.skipped_modules,
            "module_shapes": self.module_shapes,
            "checksums": self.checksums,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OrbitQuantManifest:
        return cls(
            source_model_id=data["source_model_id"],
            source_revision=data["source_revision"],
            source_license=data["source_license"],
            weight_bits=int(data["weight_bits"]),
            activation_bits=int(data["activation_bits"]),
            rotation_seed=int(data.get("rotation_seed", 0)),
            block_size=data.get("block_size", "paper"),
            block_size_policy=data.get(
                "block_size_policy", "largest_power_of_two_dividing_dim"
            ),
            codebook_version=int(data.get("codebook_version", 1)),
            target_policy=data.get("target_policy", "auto"),
            runtime_mode=data.get("runtime_mode", "dequant_bf16"),
            activation_kernel_backend=data.get("activation_kernel_backend", "auto"),
            activation_eps=float(data.get("activation_eps", 1e-10)),
            adaln_group_size=int(data.get("adaln_group_size", 64)),
            quantization_device=data.get("quantization_device", "unknown"),
            weight_quantization_backend=data.get("weight_quantization_backend", "unknown"),
            quantization_staging_mode=data.get("quantization_staging_mode", "unknown"),
            quantized_modules=list(data.get("quantized_modules", [])),
            adaln_modules=list(data.get("adaln_modules", [])),
            skipped_modules=list(data.get("skipped_modules", [])),
            module_shapes=dict(data.get("module_shapes", {})),
            checksums=dict(data.get("checksums", {})),
        )
