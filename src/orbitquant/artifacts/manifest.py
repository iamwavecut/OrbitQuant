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
    target_policy: str
    runtime_mode: str
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
    ) -> OrbitQuantManifest:
        return cls(
            source_model_id=source_model_id,
            source_revision=source_revision,
            source_license=source_license,
            weight_bits=config.weight_bits,
            activation_bits=config.activation_bits,
            target_policy=config.target_policy,
            runtime_mode=config.runtime_mode,
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
            "codebook": "lloyd_max",
            "row_norm_dtype": "bfloat16",
            "runtime_mode": self.runtime_mode,
            "target_policy": self.target_policy,
            "adaln_policy": "int4_rtn_group64_bf16_activation",
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
            target_policy=data.get("target_policy", "auto"),
            runtime_mode=data.get("runtime_mode", "dequant_bf16"),
            quantized_modules=list(data.get("quantized_modules", [])),
            adaln_modules=list(data.get("adaln_modules", [])),
            skipped_modules=list(data.get("skipped_modules", [])),
            module_shapes=dict(data.get("module_shapes", {})),
            checksums=dict(data.get("checksums", {})),
        )
