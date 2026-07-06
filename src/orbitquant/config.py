from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

try:
    from diffusers.quantizers.quantization_config import QuantizationConfigMixin
except Exception:
    try:
        from transformers.utils.quantization_config import QuantizationConfigMixin
    except Exception:

        class QuantizationConfigMixin:  # type: ignore[no-redef]
            """Fallback used when Hugging Face quantization mixins are unavailable."""


_SUPPORTED_BITS = {2, 3, 4, 6, 8}
_SUPPORTED_RUNTIME_MODES = {"dequant_bf16", "debug_no_quant", "debug_no_activation_quant"}


@dataclass
class OrbitQuantConfig(QuantizationConfigMixin):
    """Serializable OrbitQuant configuration.

    The class intentionally stays independent from Diffusers/Transformers at the
    core layer so tests and artifact tools can run without importing those large
    packages. The HF adapter wraps this shape when those libraries are present.
    """

    weight_bits: int = 4
    activation_bits: int = 4
    quant_method: str = "orbitquant"
    rotation: str = "rpbh"
    rotation_seed: int = 0
    block_size: int | str = "paper"
    codebook: str = "lloyd_max"
    codebook_dtype: str = "float32"
    row_norm_dtype: str = "bfloat16"
    activation_norm_dtype: str = "float32"
    activation_eps: float = 1e-12
    weight_pack_dtype: str = "uint8"
    target_policy: str = "auto"
    adaln_policy: str = "int4_rtn"
    adaln_group_size: int = 64
    modules_to_not_convert: list[str] = field(default_factory=list)
    modules_dtype_dict: dict[str, list[str]] = field(default_factory=dict)
    artifact_format_version: int = 1
    runtime_mode: str = "dequant_bf16"

    def __post_init__(self) -> None:
        if self.weight_bits not in _SUPPORTED_BITS:
            raise ValueError(f"weight_bits must be one of {sorted(_SUPPORTED_BITS)}")
        if self.activation_bits not in _SUPPORTED_BITS:
            raise ValueError(f"activation_bits must be one of {sorted(_SUPPORTED_BITS)}")
        if self.quant_method != "orbitquant":
            raise ValueError("quant_method must be 'orbitquant'")
        if self.rotation != "rpbh":
            raise ValueError("only RPBH rotation is implemented")
        if self.codebook != "lloyd_max":
            raise ValueError("only Lloyd-Max codebooks are implemented")
        if self.runtime_mode not in _SUPPORTED_RUNTIME_MODES:
            raise ValueError(f"runtime_mode must be one of {sorted(_SUPPORTED_RUNTIME_MODES)}")
        if self.adaln_group_size <= 0:
            raise ValueError("adaln_group_size must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OrbitQuantConfig:
        values = dict(data)
        values.pop("_class_name", None)
        values.pop("_diffusers_version", None)
        return cls(**values)
