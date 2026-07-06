from __future__ import annotations

from typing import Any

from orbitquant.config import OrbitQuantConfig
from orbitquant.modeling import quantize_linear_modules


class OrbitQuantizer:
    """Small standalone HF-style quantizer adapter.

    When Diffusers/Transformers are installed, ``register_hf_quantizers`` also
    registers this class in their auto mappings. The methods stay intentionally
    conservative until the full pre-quantized safetensors loader lands.
    """

    requires_parameters_quantization = True
    requires_calibration = False
    required_packages = None

    def __init__(self, quantization_config: OrbitQuantConfig | dict[str, Any]) -> None:
        if isinstance(quantization_config, dict):
            quantization_config = OrbitQuantConfig.from_dict(quantization_config)
        self.quantization_config = quantization_config
        self.pre_quantized = False

    def is_serializable(self, *args: Any, **kwargs: Any) -> bool:
        return True

    @property
    def is_trainable(self) -> bool:
        return False

    @property
    def is_compileable(self) -> bool:
        return True

    def _process_model_before_weight_loading(self, model: Any, *args: Any, **kwargs: Any) -> Any:
        model.quantization_config = self.quantization_config
        return model

    def _process_model_after_weight_loading(self, model: Any, *args: Any, **kwargs: Any) -> Any:
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
