from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
from transformers.core_model_loading import ConversionOps

from orbitquant.adaln import RTNInt4Linear
from orbitquant.layers import OrbitQuantLinear


class OrbitQuantWeightQuantize(ConversionOps):
    """Transformers v5 ConversionOps for full-precision Linear weights."""

    def __init__(self, hf_quantizer: Any) -> None:
        self.hf_quantizer = hf_quantizer

    @property
    def reverse_op(self) -> OrbitQuantWeightQuantize:
        return self

    def convert(
        self,
        input_dict: dict[str, Any],
        *,
        full_layer_name: str | None = None,
        model: torch.nn.Module | None = None,
        missing_keys: set[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        if full_layer_name is None or model is None:
            raise ValueError("OrbitQuant weight conversion requires full_layer_name and model")

        orbit_suffix = ".packed_weight_indices"
        rtn_suffix = ".packed_weight"
        if full_layer_name.endswith(orbit_suffix):
            module_name = full_layer_name[: -len(orbit_suffix)]
        elif full_layer_name.endswith(rtn_suffix):
            module_name = full_layer_name[: -len(rtn_suffix)]
        else:
            raise ValueError(f"expected OrbitQuant packed target, got {full_layer_name}")

        module = model.get_submodule(module_name)

        _, values = next(iter(input_dict.items()))
        weight = values[0] if isinstance(values, list) else values
        weight = self.hf_quantizer.move_tensor_for_quantization(weight)
        proxy = SimpleNamespace(
            in_features=module.in_features,
            out_features=module.out_features,
            weight=weight,
            bias=None,
        )

        results: dict[str, torch.Tensor] = {}
        if isinstance(module, OrbitQuantLinear):
            quantized = OrbitQuantLinear.from_linear(
                proxy,
                config=self.hf_quantizer.quantization_config,
                module_name=module_name,
            )
            if quantized.debug_weight is not None:
                debug_key = f"{module_name}.debug_weight"
                results[debug_key] = quantized.debug_weight
                if missing_keys is not None:
                    missing_keys.discard(debug_key)
            else:
                packed_key = f"{module_name}.packed_weight_indices"
                row_norms_key = f"{module_name}.row_norms"
                results[packed_key] = quantized.packed_weight_indices
                results[row_norms_key] = quantized.row_norms
                if missing_keys is not None:
                    missing_keys.discard(packed_key)
                    missing_keys.discard(row_norms_key)
            module.clear_dequantized_cache()
        elif isinstance(module, RTNInt4Linear):
            quantized_rtn = RTNInt4Linear.from_linear(
                proxy,
                config=self.hf_quantizer.quantization_config,
                module_name=module_name,
            )
            packed_key = f"{module_name}.packed_weight"
            scales_key = f"{module_name}.scales"
            results[packed_key] = quantized_rtn.packed_weight
            results[scales_key] = quantized_rtn.scales
            if missing_keys is not None:
                missing_keys.discard(packed_key)
                missing_keys.discard(scales_key)
        else:
            raise TypeError(
                f"expected OrbitQuantLinear or RTNInt4Linear at {module_name}, "
                f"got {type(module).__name__}"
            )
        return results
