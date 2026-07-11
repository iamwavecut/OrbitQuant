from __future__ import annotations

import re
from typing import Any

import torch

from orbitquant.adaln import RTNInt4Linear
from orbitquant.checkpoint_streaming import (
    StreamingCheckpoint,
    build_diffusers_streaming_conversion,
    build_transformers_streaming_checkpoint,
)
from orbitquant.config import OrbitQuantConfig
from orbitquant.layers import OrbitQuantLinear
from orbitquant.modeling import (
    _quantization_device,
    dequantize_quantized_linear_modules,
    prepare_prequantized_linear_modules,
    quantize_linear_modules,
)
from orbitquant.policies import classify_linear_modules

_ORBITQUANT_STATE_TENSORS = {"packed_weight_indices", "row_norms", "debug_weight", "bias"}
_RTN_INT4_STATE_TENSORS = {"packed_weight", "scales", "bias"}
_MISSING = object()


def _require_safetensors(checkpoint_files: list[Any], *, framework: str) -> None:
    unsupported = [str(path) for path in checkpoint_files if not str(path).endswith(".safetensors")]
    if unsupported:
        raise RuntimeError(
            f"OrbitQuant bounded-memory {framework} conversion requires safetensors "
            f"checkpoints; unsupported files: {unsupported}. Convert the source checkpoint "
            "to safetensors or load it without an OrbitQuant quantization_config and use the "
            "explicit post-load helper without a bounded-memory guarantee."
        )


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
        if part.isdigit() and isinstance(module, torch.nn.ModuleList | torch.nn.Sequential):
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


def _normalise_quantization_device(value: Any) -> torch.device | None:
    if value is None or value == "disk":
        return None
    if isinstance(value, int):
        value = f"cuda:{value}"
    return _quantization_device(value)


def _merge_unique(existing: list[str], extra: list[str]) -> list[str]:
    merged = list(existing)
    for item in extra:
        if item not in merged:
            merged.append(item)
    return merged


def _config_with_hf_overrides(
    config: OrbitQuantConfig,
    *,
    modules_to_not_convert: list[str],
    modules_dtype_dict: dict[str, list[str]],
) -> OrbitQuantConfig:
    if not modules_to_not_convert and not modules_dtype_dict:
        return config
    values = config.to_dict()
    values["modules_to_not_convert"] = _merge_unique(
        list(values.get("modules_to_not_convert", [])), modules_to_not_convert
    )
    merged_dtype_dict = {
        dtype_name: list(module_names)
        for dtype_name, module_names in values.get("modules_dtype_dict", {}).items()
    }
    for dtype_name, module_names in modules_dtype_dict.items():
        merged_dtype_dict[dtype_name] = _merge_unique(
            merged_dtype_dict.get(dtype_name, []), list(module_names)
        )
    values["modules_dtype_dict"] = merged_dtype_dict
    return OrbitQuantConfig.from_dict(values)


class OrbitQuantizer(*_hf_base_classes()):
    """Small standalone HF-style quantizer adapter.

    When Diffusers/Transformers are installed, ``register_hf_quantizers`` also
    registers this class in their auto mappings. The methods stay intentionally
    conservative around HF version-specific loading hooks.
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
        quantization_device_arg = kwargs.pop("quantization_device", _MISSING)
        quantization_device = _normalise_quantization_device(
            "auto" if quantization_device_arg is _MISSING else quantization_device_arg
        )
        modules_to_not_convert = list(kwargs.get("modules_to_not_convert") or [])
        modules_dtype_dict = {
            dtype_name: list(module_names)
            for dtype_name, module_names in (kwargs.get("modules_dtype_dict") or {}).items()
        }
        quantization_config = _config_with_hf_overrides(
            quantization_config,
            modules_to_not_convert=modules_to_not_convert,
            modules_dtype_dict=modules_dtype_dict,
        )
        if self.__class__.__bases__ == (object,):
            self.quantization_config = quantization_config
            self.pre_quantized = kwargs.get("pre_quantized", True)
        else:
            super().__init__(quantization_config, **kwargs)
        if not hasattr(self, "modules_to_not_convert"):
            self.modules_to_not_convert = modules_to_not_convert
        self._transformers_postload_quantization = False
        self._transformers_streaming_quantization = False
        self._transformers_orbit_module_names: list[str] = []
        self._transformers_adaln_module_names: list[str] = []
        self._transformers_base_model_prefix = ""
        self._transformers_preconverted_checkpoint = False
        self._transformers_streaming_checkpoint: StreamingCheckpoint | None = None
        self._diffusers_streaming_quantization = False
        self._diffusers_model: Any | None = None
        self._diffusers_loaded_keys: list[str] | None = None
        self._diffusers_checkpoint_files: list[Any] = []
        self._diffusers_packed_state: dict[str, torch.Tensor] | None = None
        self._diffusers_replaced_source_keys: set[str] = set()
        self.released_source_tensor_bytes = 0
        self.source_page_release_failures = 0
        self.quantization_device = quantization_device

    def release_source_tensor(self, tensor: torch.Tensor) -> None:
        if tensor.device.type == "cpu":
            self.released_source_tensor_bytes += tensor.numel() * tensor.element_size()

    def _quantization_device_from_kwargs(self, kwargs: dict[str, Any]) -> torch.device | None:
        explicit_device = _normalise_quantization_device(kwargs.get("quantization_device"))
        if explicit_device is not None:
            return explicit_device
        target_device = _normalise_quantization_device(kwargs.get("target_device"))
        if target_device is not None:
            return target_device
        return self.quantization_device

    def move_tensor_for_quantization(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.quantization_device is None:
            return tensor
        return tensor.to(device=self.quantization_device)

    def is_serializable(self, *args: Any, **kwargs: Any) -> bool:
        return True

    def validate_environment(self, *args: Any, **kwargs: Any) -> None:
        return None

    def update_torch_dtype(self, torch_dtype: torch.dtype | None) -> torch.dtype | None:
        return torch_dtype

    def adjust_target_dtype(self, torch_dtype: torch.dtype | None) -> torch.dtype | None:
        return torch_dtype

    def update_device_map(self, device_map: Any | None) -> Any | None:
        return device_map

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
        if not self.pre_quantized and not self._diffusers_streaming_quantization:
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
        if self.pre_quantized or self._diffusers_streaming_quantization:
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

    @property
    def supports_parallel_loading(self) -> bool:
        return self.pre_quantized

    def maybe_update_loaded_keys(
        self,
        loaded_keys: list[str],
        checkpoint_files: list[Any],
    ) -> list[str]:
        if self.pre_quantized:
            return loaded_keys
        _require_safetensors(checkpoint_files, framework="Diffusers")
        self._diffusers_loaded_keys = loaded_keys
        self._diffusers_checkpoint_files = list(checkpoint_files)
        return loaded_keys

    def _diffusers_streaming_key_mapping(self) -> dict[str, list[str]]:
        mapping: dict[str, list[str]] = {}
        for name in self._transformers_orbit_module_names:
            outputs = (
                [f"{name}.debug_weight"]
                if self.quantization_config.runtime_mode == "debug_no_quant"
                else [f"{name}.packed_weight_indices", f"{name}.row_norms"]
            )
            mapping[f"{name}.weight"] = outputs
        for name in self._transformers_adaln_module_names:
            mapping[f"{name}.weight"] = [f"{name}.packed_weight", f"{name}.scales"]
        return mapping

    def _update_diffusers_loaded_keys(self) -> None:
        if self._diffusers_loaded_keys is None:
            return
        mapping = self._diffusers_streaming_key_mapping()
        updated: list[str] = []
        for key in self._diffusers_loaded_keys:
            updated.extend(mapping.get(key, [key]))
        self._diffusers_loaded_keys[:] = updated

    def maybe_update_state_dict(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        if not self._diffusers_streaming_quantization:
            return state_dict
        if self._diffusers_model is None:
            raise RuntimeError("Diffusers streaming conversion has no prepared model skeleton")

        for source_key in self._diffusers_replaced_source_keys:
            state_dict.pop(source_key, None)
        if self._diffusers_packed_state is not None:
            state_dict.update(self._diffusers_packed_state)
            self._diffusers_packed_state = None
        return state_dict

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

    def get_weight_conversions(self) -> list[Any]:
        if (
            not self._transformers_streaming_quantization
            or self._transformers_preconverted_checkpoint
        ):
            return []
        from transformers.core_model_loading import WeightConverter

        from orbitquant.transformers_ops import OrbitQuantWeightQuantize

        def source_patterns(name: str) -> list[str]:
            patterns = [f"{name}.weight"]
            prefix = self._transformers_base_model_prefix
            if prefix and name.startswith(f"{prefix}."):
                patterns.append(f"{name.removeprefix(f'{prefix}.')}.weight")
            return patterns

        conversions = [
            WeightConverter(
                source_patterns=source_patterns(name),
                target_patterns=(
                    f"{name}.debug_weight"
                    if self.quantization_config.runtime_mode == "debug_no_quant"
                    else f"{name}.packed_weight_indices"
                ),
                operations=[OrbitQuantWeightQuantize(self)],
            )
            for name in self._transformers_orbit_module_names
        ]
        conversions.extend(
            WeightConverter(
                source_patterns=source_patterns(name),
                target_patterns=f"{name}.packed_weight",
                operations=[OrbitQuantWeightQuantize(self)],
            )
            for name in self._transformers_adaln_module_names
        )
        return conversions

    @property
    def is_trainable(self) -> bool:
        return False

    @property
    def is_compileable(self) -> bool:
        return True

    def _process_model_before_weight_loading(self, model: Any, *args: Any, **kwargs: Any) -> Any:
        model.quantization_config = self.quantization_config
        checkpoint_files = kwargs.get("checkpoint_files")
        self._transformers_postload_quantization = (
            not self.pre_quantized and checkpoint_files is not None
        )
        self._transformers_streaming_quantization = self._transformers_postload_quantization
        if self._transformers_streaming_quantization:
            _require_safetensors(checkpoint_files, framework="Transformers")
        self._diffusers_streaming_quantization = (
            not self.pre_quantized
            and not self._transformers_streaming_quantization
            and bool(self._diffusers_checkpoint_files)
        )
        self._transformers_orbit_module_names = []
        self._transformers_adaln_module_names = []
        self._transformers_base_model_prefix = str(getattr(model, "base_model_prefix", "") or "")
        if self._transformers_streaming_quantization or self._diffusers_streaming_quantization:
            decisions = classify_linear_modules(model, self.quantization_config)
            self._transformers_orbit_module_names = [
                name for name, decision in decisions.items() if decision.action == "orbitquant"
            ]
            self._transformers_adaln_module_names = [
                name for name, decision in decisions.items() if decision.action == "adaln_int4_rtn"
            ]
        if self._transformers_streaming_quantization:
            streaming_checkpoint = build_transformers_streaming_checkpoint(
                model,
                self.quantization_config,
                checkpoint_files,
                orbit_module_names=self._transformers_orbit_module_names,
                adaln_module_names=self._transformers_adaln_module_names,
                base_model_prefix=self._transformers_base_model_prefix,
                quantization_device=self.quantization_device,
            )
            checkpoint_files.append(str(streaming_checkpoint.packed_file))
            self._transformers_streaming_checkpoint = streaming_checkpoint
            self._transformers_preconverted_checkpoint = True
            self.released_source_tensor_bytes = streaming_checkpoint.source_tensor_bytes
            ignored_source_keys = set(
                getattr(model, "_keys_to_ignore_on_load_unexpected", None) or ()
            )
            ignored_source_keys.update(
                rf"^{re.escape(key)}$" for key in streaming_checkpoint.replaced_source_keys
            )
            for name in (
                self._transformers_orbit_module_names
                + self._transformers_adaln_module_names
            ):
                ignored_source_keys.add(rf"^{re.escape(f'{name}.weight')}$")
                if self._transformers_base_model_prefix:
                    ignored_source_keys.add(
                        rf"^{re.escape(f'{self._transformers_base_model_prefix}.{name}.weight')}$"
                    )
            model._keys_to_ignore_on_load_unexpected = ignored_source_keys
        if self._diffusers_streaming_quantization:
            conversion = build_diffusers_streaming_conversion(
                model,
                self.quantization_config,
                self._diffusers_checkpoint_files,
                orbit_module_names=self._transformers_orbit_module_names,
                adaln_module_names=self._transformers_adaln_module_names,
                quantization_device=self.quantization_device,
            )
            self._diffusers_packed_state = conversion.packed_state
            self._diffusers_replaced_source_keys = set(conversion.replaced_source_keys)
            self.released_source_tensor_bytes = conversion.source_tensor_bytes
        if (
            self.pre_quantized
            or self._transformers_streaming_quantization
            or self._diffusers_streaming_quantization
        ):
            prepare_prequantized_linear_modules(
                model,
                self.quantization_config,
            )
        if self._diffusers_streaming_quantization:
            self._diffusers_model = model
            self._update_diffusers_loaded_keys()
        return model

    def _process_model_after_weight_loading(self, model: Any, *args: Any, **kwargs: Any) -> Any:
        if (
            not self.pre_quantized
            and not self._transformers_streaming_quantization
            and not self._diffusers_streaming_quantization
        ):
            quantize_linear_modules(
                model,
                self.quantization_config,
                quantization_device=self._quantization_device_from_kwargs(kwargs),
            )
        if self._transformers_streaming_quantization:
            if hasattr(model, "_weight_conversions"):
                delattr(model, "_weight_conversions")
            self._transformers_postload_quantization = False
            self._transformers_streaming_quantization = False
            self._transformers_orbit_module_names = []
            self._transformers_adaln_module_names = []
            self._transformers_base_model_prefix = ""
            self._transformers_preconverted_checkpoint = False
            if self._transformers_streaming_checkpoint is not None:
                self._transformers_streaming_checkpoint.directory.cleanup()
                self._transformers_streaming_checkpoint = None
        if self._diffusers_streaming_quantization:
            self._diffusers_streaming_quantization = False
            self._diffusers_model = None
            self._diffusers_loaded_keys = None
            self._diffusers_checkpoint_files = []
            self._diffusers_packed_state = None
            self._diffusers_replaced_source_keys = set()
            self._transformers_orbit_module_names = []
            self._transformers_adaln_module_names = []
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
