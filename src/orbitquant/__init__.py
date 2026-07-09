"""OrbitQuant clean-room implementation for diffusion transformer quantization."""

from orbitquant.config import OrbitQuantConfig
from orbitquant.layers import OrbitQuantLinear
from orbitquant.modeling import prewarm_quantized_linear_modules
from orbitquant.pipeline import (
    build_diffusers_pipeline_quantization_config,
    load_quantized_pipeline_component,
    load_quantized_pipeline_from_artifact,
    quantize_pipeline,
    save_quantized_pipeline_component,
)
from orbitquant.quantizer import register_hf_quantizers

__version__ = "0.1.3"

register_hf_quantizers()

__all__ = [
    "OrbitQuantConfig",
    "OrbitQuantLinear",
    "__version__",
    "build_diffusers_pipeline_quantization_config",
    "load_quantized_pipeline_from_artifact",
    "load_quantized_pipeline_component",
    "prewarm_quantized_linear_modules",
    "quantize_pipeline",
    "register_hf_quantizers",
    "save_quantized_pipeline_component",
]
