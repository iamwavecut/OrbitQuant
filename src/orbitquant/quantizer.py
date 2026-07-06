from __future__ import annotations

from typing import Any

import torch

from orbitquant.adaln import RTNInt4Linear
from orbitquant.config import OrbitQuantConfig
from orbitquant.layers import OrbitQuantLinear
from orbitquant.modeling import (
    dequantize_quantized_linear_modules,
    prepare_prequantized_linear_modules,
    quantize_linear_modules,
)
from orbitquant.policies import classify_linear_modules

_ORBITQUANT_STATE_TENSORS = {"packed_weight_indices", "row_norms", "debug_weight", "bias"}
_RTN_INT4_STATE_TENSORS = {"packed_weight", "scales", "bias"}


def _hf_base_classes() -> tuple[type, ...]:
    bases: list[type] = []
    try:
        from diffusers.quantizers.base import DiffusersQuantizer

        bases.append(DiffusersQuantizer)
    except Exception:
        pass
    try:
        from transformers.quantizers import HfQuantizer

        bases.append(HfQuantizer)
    except Exception:
        pass
    return tuple(bases) or (object,)


def _module_and_tensor_name(model: Any, param_name: str) -> tuple[Any, str]:
    parts = param_name.split(".")
    module = model
    for part in parts[:-1]:
        if part.isdigit() and isinstance(module, (torch.nn.ModuleList, torch.nn.Sequential)):
            module = module[int(part)]
        elif isinstance(module, torch.nn.ModuleDict):
            module = module[part]
        else:
            module = getattr(module, part)
    return module, parts[-1]


def _is_prequantized_state_tensor(module: Any, tensor_name: str) -> bool:
    if isinstance(module, OrbitQuantLinear):
        return tensor_name in _ORBITQUANT_STATE_TENSORS and (
            tensor_name in module._parameters or tensor_name in module._buffers
        )
    if isinstance(module, RTNInt4Linear):
        return tensor_name in _RTN_INT4_STATE_TENSORS and (
            tensor_name in module._parameters or tensor_name in module._buffers
        )
    return False


def _move_like_loaded_tensor(
    value: torch.Tensor,
    *,
    target_device: Any,
    existing: torch.Tensor | None,
) -> torch.Tensor:
    dtype = None if existing is None else existing.dtype
    if target_device is None:
        return value.to(dtype=dtype) if dtype is not None else value
    return value.to(device=target_device, dtype=dtype)


class OrbitQuantizer(*_hf_base_classes()):
    """Small standalone HF-style quantizer adapter.

    When Diffusers/Transformers are installed, ``register_hf_quantizers`` also
    registers this class in their auto mappings. The methods stay intentionally
    conservative until the full pre-quantized safetensors loader lands.
    """

    requires_parameters_quantization = True
    requires_calibration = False
    required_packages = None

    use_keep_in_fp32_modules = True

    def __init__(
        self, quantization_config: OrbitQuantConfig | dict[str, Any], **kwargs: Any
    ) -> None:
        if isinstance(quantization_config, dict):
            quantization_config = OrbitQuantConfig.from_dict(quantization_config)
        modules_to_not_convert = kwargs.get("modules_to_not_convert", [])
        if self.__class__.__bases__ == (object,):
            self.quantization_config = quantization_config
            self.pre_quantized = kwargs.get("pre_quantized", True)
        else:
            super().__init__(quantization_config, **kwargs)
        if not hasattr(self, "modules_to_not_convert"):
            self.modules_to_not_convert = modules_to_not_convert
        self._transformers_postload_quantization = False

    def is_serializable(self, *args: Any, **kwargs: Any) -> bool:
        return True

    def _param_action(self, model: Any, param_name: str) -> str | None:
        if not param_name.endswith(".weight"):
            return None
        module_name = param_name.removesuffix(".weight")
        decisions = classify_linear_modules(model, self.quantization_config)
        decision = decisions.get(module_name)
        return None if decision is None else decision.action

    def param_needs_quantization(
        self, model: Any, param_name: str, *args: Any, **kwargs: Any
    ) -> bool:
        if self._transformers_postload_quantization:
            return False
        return self._param_action(model, param_name) in {"orbitquant", "adaln_int4_rtn"}

    def check_if_quantized_param(
        self,
        model: Any,
        param_value: Any,
        param_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> bool:
        if not self.pre_quantized:
            return False
        try:
            module, tensor_name = _module_and_tensor_name(model, param_name)
        except (AttributeError, IndexError, KeyError):
            return False
        return _is_prequantized_state_tensor(module, tensor_name)

    def check_quantized_param(self, *args: Any, **kwargs: Any) -> bool:
        return self.check_if_quantized_param(*args, **kwargs)

    def create_quantized_param(
        self,
        model: Any,
        param_value: torch.Tensor,
        param_name: str,
        target_device: Any,
        state_dict: dict[str, Any],
        unexpected_keys: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        if self.pre_quantized:
            module, tensor_name = _module_and_tensor_name(model, param_name)
            if not _is_prequantized_state_tensor(module, tensor_name):
                raise ValueError(f"{param_name} is not an OrbitQuant pre-quantized tensor")

            if tensor_name in module._parameters:
                old_value = module._parameters[tensor_name]
                loaded_value = _move_like_loaded_tensor(
                    param_value, target_device=target_device, existing=old_value
                )
                requires_grad = False if old_value is None else old_value.requires_grad
                module._parameters[tensor_name] = torch.nn.Parameter(
                    loaded_value, requires_grad=requires_grad
                )
            else:
                old_value = module._buffers[tensor_name]
                module._buffers[tensor_name] = _move_like_loaded_tensor(
                    param_value, target_device=target_device, existing=old_value
                )

            if isinstance(module, OrbitQuantLinear):
                module.clear_dequantized_cache()
            if unexpected_keys is not None and param_name in unexpected_keys:
                unexpected_keys.remove(param_name)
            return

        msg = (
            "OrbitQuant does not create quantized tensors during HF streaming "
            "weight loading from full precision tensors. Use pre_quantized=True "
            "to load packed OrbitQuant buffers into prepared module skeletons, "
            "or pre_quantized=False to load full precision weights and quantize "
            "after loading."
        )
        raise RuntimeError(msg)

    def check_quantized_param_shape(
        self,
        param_name: str,
        current_param: torch.Tensor,
        loaded_param: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> bool:
        if current_param.shape != loaded_param.shape:
            raise ValueError(
                f"Expected {param_name} to have shape {tuple(current_param.shape)}, "
                f"but loaded tensor has shape {tuple(loaded_param.shape)}."
            )
        return True

    def get_state_dict_and_metadata(
        self, state_or_model: Any, safe_serialization: bool = False
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if isinstance(state_or_model, torch.nn.Module):
            return state_or_model.state_dict(), {}
        return state_or_model, {}

    @property
    def is_trainable(self) -> bool:
        return False

    @property
    def is_compileable(self) -> bool:
        return True

    def _process_model_before_weight_loading(self, model: Any, *args: Any, **kwargs: Any) -> Any:
        model.quantization_config = self.quantization_config
        self._transformers_postload_quantization = (
            not self.pre_quantized and "checkpoint_files" in kwargs
        )
        if self.pre_quantized:
            prepare_prequantized_linear_modules(model, self.quantization_config)
        return model

    def _process_model_after_weight_loading(self, model: Any, *args: Any, **kwargs: Any) -> Any:
        if not self.pre_quantized:
            quantize_linear_modules(model, self.quantization_config)
        return model

    def _dequantize(self, model: Any) -> Any:
        return dequantize_quantized_linear_modules(model)


def _register_diffusers() -> bool:
    try:
        import diffusers.quantizers.auto as auto
    except Exception:
        return False
    auto.AUTO_QUANTIZER_MAPPING["orbitquant"] = OrbitQuantizer
    auto.AUTO_QUANTIZATION_CONFIG_MAPPING["orbitquant"] = OrbitQuantConfig
    return True


def _register_transformers() -> bool:
    try:
        import transformers.quantizers.auto as auto
    except Exception:
        return False
    auto.AUTO_QUANTIZER_MAPPING["orbitquant"] = OrbitQuantizer
    auto.AUTO_QUANTIZATION_CONFIG_MAPPING["orbitquant"] = OrbitQuantConfig
    return True


def register_hf_quantizers() -> dict[str, bool]:
    return {
        "diffusers": _register_diffusers(),
        "transformers": _register_transformers(),
    }
