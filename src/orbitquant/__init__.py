"""Calibration-free OrbitQuant for transformer linear projections."""

from orbitquant.config import OrbitQuantConfig
from orbitquant.layers import OrbitQuantLinear
from orbitquant.linear_adapters import register_linear_adapter
from orbitquant.modeling import (
    inspect_linear_module_policy,
    prewarm_quantized_linear_modules,
    quantize_model,
)
from orbitquant.pipeline import (
    build_diffusers_pipeline_quantization_config,
    load_quantized_pipeline_component,
    load_quantized_pipeline_from_artifact,
    quantize_pipeline,
    save_quantized_pipeline_component,
)
from orbitquant.quantizer import register_hf_quantizers
from orbitquant.recipes import recipe

__version__ = "0.9.0"

register_hf_quantizers()

__all__ = [
    "OrbitQuantConfig",
    "OrbitQuantLinear",
    "__version__",
    "build_diffusers_pipeline_quantization_config",
    "load_quantized_pipeline_from_artifact",
    "load_quantized_pipeline_component",
    "inspect_linear_module_policy",
    "prewarm_quantized_linear_modules",
    "quantize_model",
    "quantize_pipeline",
    "recipe",
    "register_linear_adapter",
    "register_hf_quantizers",
    "save_quantized_pipeline_component",
]
