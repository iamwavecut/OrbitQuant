from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
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
_SUPPORTED_RUNTIME_MODES = {
    "auto_fused",
    "dequant_bf16",
    "debug_no_quant",
    "debug_no_activation_quant",
    "triton_packed_matmul",
    "native_packed_matmul",
}
_SUPPORTED_ACTIVATION_KERNEL_BACKENDS = {
    "auto",
    "cpu",
    "mps",
    "triton_cuda",
    "triton_rocm",
    "triton_xpu",
}
_SUPPORTED_TARGET_POLICIES = {
    "auto",
    "universal",
    "generic_dit",
    "flux",
    "flux2",
    "z_image",
    "wan",
}
_SUPPORTED_MODULE_DTYPES = {"bfloat16", "bf16", "float16", "fp16", "float32", "fp32"}
_SUPPORTED_CODEBOOK_DTYPES = {"float32", "fp32"}
_SUPPORTED_ROW_NORM_DTYPES = {"bfloat16", "bf16"}
_SUPPORTED_ACTIVATION_NORM_DTYPES = {"float32", "fp32"}
_SUPPORTED_WEIGHT_PACK_DTYPES = {"uint8"}
_SUPPORTED_ADALN_POLICIES = {"int4_rtn"}


class _QuantMethodName(str):
    @property
    def value(self) -> str:
        return str(self)


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
    codebook_version: int = 2
    codebook_dtype: str = "float32"
    row_norm_dtype: str = "bfloat16"
    activation_norm_dtype: str = "float32"
    activation_eps: float = 1e-10
    weight_pack_dtype: str = "uint8"
    target_policy: str = "auto"
    adaln_policy: str = "int4_rtn"
    adaln_group_size: int = 64
    modules_to_convert: list[str] = field(default_factory=list)
    modules_to_use_adaln: list[str] = field(default_factory=list)
    modules_to_not_convert: list[str] = field(default_factory=list)
    modules_dtype_dict: dict[str, list[str]] = field(default_factory=dict)
    artifact_format_version: int = 1
    runtime_mode: str = "auto_fused"
    activation_kernel_backend: str = "auto"
    packed_matmul_block_m: int = 64
    packed_matmul_block_n: int = 64
    packed_matmul_block_k: int = 128
    packed_matmul_num_warps: int = 4
    weight_row_tile_size: int = 256
    # Opt-in: keep a persistent INT8 copy of each W4A4 weight on CUDA (twice
    # the packed size) so the cuBLASLt path skips its per-forward decode.
    w4a4_int8_weight_cache: bool = False

    def __post_init__(self) -> None:
        if self.weight_bits not in _SUPPORTED_BITS:
            raise ValueError(f"weight_bits must be one of {sorted(_SUPPORTED_BITS)}")
        if self.activation_bits not in _SUPPORTED_BITS:
            raise ValueError(f"activation_bits must be one of {sorted(_SUPPORTED_BITS)}")
        if self.quant_method != "orbitquant":
            raise ValueError("quant_method must be 'orbitquant'")
        self.quant_method = _QuantMethodName(self.quant_method)
        if self.rotation != "rpbh":
            raise ValueError("only RPBH rotation is implemented")
        if self.codebook != "lloyd_max":
            raise ValueError("only Lloyd-Max codebooks are implemented")
        if self.codebook_version not in {1, 2}:
            raise ValueError("codebook_version must be 1 or 2")
        if self.codebook_dtype.lower() not in _SUPPORTED_CODEBOOK_DTYPES:
            raise ValueError("codebook_dtype must be 'float32'")
        if self.row_norm_dtype.lower() not in _SUPPORTED_ROW_NORM_DTYPES:
            raise ValueError("row_norm_dtype must be 'bfloat16'")
        if self.activation_norm_dtype.lower() not in _SUPPORTED_ACTIVATION_NORM_DTYPES:
            raise ValueError("activation_norm_dtype must be 'float32'")
        if self.weight_pack_dtype.lower() not in _SUPPORTED_WEIGHT_PACK_DTYPES:
            raise ValueError("weight_pack_dtype must be 'uint8'")
        if self.adaln_policy not in _SUPPORTED_ADALN_POLICIES:
            raise ValueError("adaln_policy must be 'int4_rtn'")
        if self.runtime_mode not in _SUPPORTED_RUNTIME_MODES:
            raise ValueError(f"runtime_mode must be one of {sorted(_SUPPORTED_RUNTIME_MODES)}")
        if self.activation_kernel_backend not in _SUPPORTED_ACTIVATION_KERNEL_BACKENDS:
            raise ValueError(
                "activation_kernel_backend must be one of "
                f"{sorted(_SUPPORTED_ACTIVATION_KERNEL_BACKENDS)}"
            )
        for field_name in (
            "packed_matmul_block_m",
            "packed_matmul_block_n",
            "packed_matmul_block_k",
            "packed_matmul_num_warps",
            "weight_row_tile_size",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")
        if self.target_policy not in _SUPPORTED_TARGET_POLICIES:
            raise ValueError(f"target_policy must be one of {sorted(_SUPPORTED_TARGET_POLICIES)}")
        if self.adaln_group_size <= 0:
            raise ValueError("adaln_group_size must be positive")
        if not isinstance(self.w4a4_int8_weight_cache, bool):
            raise ValueError("w4a4_int8_weight_cache must be a boolean")
        normalized_dtype_dict: dict[str, list[str]] = {}
        for dtype_name, module_names in self.modules_dtype_dict.items():
            normalized_dtype = dtype_name.lower()
            if normalized_dtype not in _SUPPORTED_MODULE_DTYPES:
                raise ValueError(
                    "modules_dtype_dict keys must be one of "
                    f"{sorted(_SUPPORTED_MODULE_DTYPES)}"
                )
            normalized_dtype_dict[normalized_dtype] = list(module_names)
        self.modules_dtype_dict = normalized_dtype_dict

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_diff_dict(self) -> dict[str, Any]:
        return self.to_dict()

    def to_json_string(self, use_diff: bool = True) -> str:
        payload = self.to_diff_dict() if use_diff else self.to_dict()
        return json.dumps(payload, indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path: str | Path, use_diff: bool = True) -> None:
        Path(json_file_path).write_text(self.to_json_string(use_diff=use_diff), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OrbitQuantConfig:
        values = dict(data)
        values.pop("_class_name", None)
        values.pop("_diffusers_version", None)
        # Artifacts produced before codebook versioning used the legacy v1
        # centroids. Preserve their packed-index interpretation on load.
        values.setdefault("codebook_version", 1)
        return cls(**values)
