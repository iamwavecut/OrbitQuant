from __future__ import annotations

from typing import Any

from orbitquant.config import OrbitQuantConfig
from orbitquant.modeling import prepare_prequantized_linear_modules, quantize_linear_modules
from orbitquant.policies import classify_linear_modules


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
        return self.param_needs_quantization(model, param_name)

    def check_quantized_param(self, *args: Any, **kwargs: Any) -> bool:
        return self.check_if_quantized_param(*args, **kwargs)

    def create_quantized_param(self, *args: Any, **kwargs: Any) -> None:
        msg = (
            "OrbitQuant does not create quantized tensors during HF streaming "
            "weight loading. Use pre_quantized=True to load packed OrbitQuant "
            "buffers into prepared module skeletons, or pre_quantized=False to "
            "load full precision weights and quantize after loading."
        )
        raise RuntimeError(msg)

    @property
    def is_trainable(self) -> bool:
        return False

    @property
    def is_compileable(self) -> bool:
        return True

    def _process_model_before_weight_loading(self, model: Any, *args: Any, **kwargs: Any) -> Any:
        model.quantization_config = self.quantization_config
        if self.pre_quantized:
            prepare_prequantized_linear_modules(model, self.quantization_config)
        return model

    def _process_model_after_weight_loading(self, model: Any, *args: Any, **kwargs: Any) -> Any:
        if not self.pre_quantized:
            quantize_linear_modules(model, self.quantization_config)
        return model

    def _dequantize(self, model: Any) -> Any:
        return model


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
