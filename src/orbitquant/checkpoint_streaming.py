from __future__ import annotations

import tempfile
from dataclasses import dataclass
from math import prod
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from orbitquant.adaln import RTNInt4Linear
from orbitquant.config import OrbitQuantConfig
from orbitquant.layers import OrbitQuantLinear
from orbitquant.linear_adapters import linear_module_spec

_DTYPE_SIZES = {
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "F64": 8,
}


@dataclass(frozen=True)
class StreamingCheckpoint:
    directory: tempfile.TemporaryDirectory[str]
    packed_file: Path
    replaced_source_keys: tuple[str, ...]
    source_tensor_bytes: int


@dataclass(frozen=True)
class StreamingConversion:
    packed_state: dict[str, torch.Tensor]
    replaced_source_keys: tuple[str, ...]
    source_tensor_bytes: int


class _SafetensorsCanonicalRows:
    ndim = 2
    device = torch.device("cpu")

    def __init__(self, tensor_slice: Any, source_weight_layout: str) -> None:
        source_shape = tuple(tensor_slice.get_shape())
        self._slice = tensor_slice
        self._source_weight_layout = source_weight_layout
        self.shape = (
            source_shape
            if source_weight_layout == "out_in"
            else (source_shape[1], source_shape[0])
        )

    def __getitem__(self, rows: slice) -> torch.Tensor:
        if self._source_weight_layout == "out_in":
            return self._slice[rows, :]
        return self._slice[:, rows].transpose(0, 1)


def _tensor_index(
    checkpoint_files: list[Any],
) -> dict[str, tuple[str, tuple[int, ...], int]]:
    index: dict[str, tuple[str, tuple[int, ...], int]] = {}
    for checkpoint_file in checkpoint_files:
        path = str(checkpoint_file)
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():  # noqa: SIM118 - safe_open is not iterable
                tensor_slice = handle.get_slice(key)
                dtype = tensor_slice.get_dtype()
                index[key] = (
                    path,
                    tuple(tensor_slice.get_shape()),
                    _DTYPE_SIZES.get(dtype, 1),
                )
    return index


def _convert_streaming_checkpoint(
    model: torch.nn.Module,
    config: OrbitQuantConfig,
    checkpoint_files: list[Any],
    *,
    orbit_module_names: list[str],
    adaln_module_names: list[str],
    base_model_prefix: str,
    quantization_device: torch.device | None,
    apply_transformers_mapping: bool,
) -> StreamingConversion:
    """Convert selected safetensors rows with one source mapping open at a time."""

    tensor_index = _tensor_index(checkpoint_files)
    target_to_source = {key: key for key in tensor_index}
    if apply_transformers_mapping:
        from transformers.conversion_mapping import get_model_conversion_mapping
        from transformers.core_model_loading import (
            WeightConverter,
            WeightRenaming,
            rename_source_key,
        )

        model_conversions = get_model_conversion_mapping(model, hf_quantizer=None)
        renamings = [item for item in model_conversions if isinstance(item, WeightRenaming)]
        converters = [item for item in model_conversions if isinstance(item, WeightConverter)]
        meta_state_dict = model.state_dict()
        for source_key in tensor_index:
            target_key, _ = rename_source_key(
                source_key,
                renamings,
                converters,
                base_model_prefix=getattr(model, "base_model_prefix", ""),
                meta_state_dict=meta_state_dict,
            )
            target_to_source.setdefault(target_key, source_key)

    packed_state: dict[str, torch.Tensor] = {}
    replaced_source_keys: list[str] = []
    source_tensor_bytes = 0

    def source_key_for(module_name: str) -> str:
        candidates = [f"{module_name}.weight"]
        if base_model_prefix and module_name.startswith(f"{base_model_prefix}."):
            candidates.append(
                f"{module_name.removeprefix(f'{base_model_prefix}.')}.weight"
            )
        try:
            return next(
                target_to_source.get(key, key)
                for key in candidates
                if key in target_to_source or key in tensor_index
            )
        except StopIteration as exc:
            raise RuntimeError(
                f"OrbitQuant could not find a safetensors source for {module_name}.weight"
            ) from exc

    for module_name in orbit_module_names + adaln_module_names:
        module = model.get_submodule(module_name)
        source_key = source_key_for(module_name)
        checkpoint_file, source_shape, element_size = tensor_index[source_key]
        source_tensor_bytes += prod(source_shape) * element_size
        spec = linear_module_spec(module)
        if spec is None:
            raise TypeError(f"no OrbitQuant linear adapter registered for {type(module).__name__}")
        with torch.device("cpu"), safe_open(
            checkpoint_file, framework="pt", device="cpu"
        ) as handle:
            weight = _SafetensorsCanonicalRows(
                handle.get_slice(source_key),
                spec.weight_layout,
            )
            if module_name in orbit_module_names:
                quantized = OrbitQuantLinear.from_weight(
                    weight,
                    bias=None,
                    in_features=weight.shape[1],
                    out_features=weight.shape[0],
                    source_weight_layout=spec.weight_layout,
                    config=config,
                    module_name=module_name,
                    quantization_device=quantization_device,
                )
                if quantized.debug_weight is not None:
                    packed_state[f"{module_name}.debug_weight"] = quantized.debug_weight.detach()
                else:
                    packed_state[f"{module_name}.packed_weight_indices"] = (
                        quantized.packed_weight_indices.detach()
                    )
                    packed_state[f"{module_name}.row_norms"] = quantized.row_norms.detach()
            else:
                quantized = RTNInt4Linear.from_weight(
                    weight,
                    bias=None,
                    in_features=weight.shape[1],
                    out_features=weight.shape[0],
                    source_weight_layout=spec.weight_layout,
                    config=config,
                    module_name=module_name,
                    quantization_device=quantization_device,
                )
                packed_state[f"{module_name}.packed_weight"] = quantized.packed_weight.detach()
                packed_state[f"{module_name}.scales"] = quantized.scales.detach()
        replaced_source_keys.append(source_key)

    return StreamingConversion(
        packed_state=packed_state,
        replaced_source_keys=tuple(replaced_source_keys),
        source_tensor_bytes=source_tensor_bytes,
    )


def build_transformers_streaming_checkpoint(
    model: torch.nn.Module,
    config: OrbitQuantConfig,
    checkpoint_files: list[Any],
    *,
    orbit_module_names: list[str],
    adaln_module_names: list[str],
    base_model_prefix: str,
    quantization_device: torch.device | None,
) -> StreamingCheckpoint:
    conversion = _convert_streaming_checkpoint(
        model,
        config,
        checkpoint_files,
        orbit_module_names=orbit_module_names,
        adaln_module_names=adaln_module_names,
        base_model_prefix=base_model_prefix,
        quantization_device=quantization_device,
        apply_transformers_mapping=True,
    )
    directory = tempfile.TemporaryDirectory(prefix="orbitquant-streaming-")
    packed_file = Path(directory.name) / "orbitquant-streaming.safetensors"
    save_file(conversion.packed_state, packed_file, metadata={"format": "pt"})
    return StreamingCheckpoint(
        directory=directory,
        packed_file=packed_file,
        replaced_source_keys=conversion.replaced_source_keys,
        source_tensor_bytes=conversion.source_tensor_bytes,
    )


def build_diffusers_streaming_conversion(
    model: torch.nn.Module,
    config: OrbitQuantConfig,
    checkpoint_files: list[Any],
    *,
    orbit_module_names: list[str],
    adaln_module_names: list[str],
    quantization_device: torch.device | None,
) -> StreamingConversion:
    return _convert_streaming_checkpoint(
        model,
        config,
        checkpoint_files,
        orbit_module_names=orbit_module_names,
        adaln_module_names=adaln_module_names,
        base_model_prefix="",
        quantization_device=quantization_device,
        apply_transformers_mapping=False,
    )


__all__ = [
    "StreamingCheckpoint",
    "StreamingConversion",
    "build_diffusers_streaming_conversion",
    "build_transformers_streaming_checkpoint",
]
