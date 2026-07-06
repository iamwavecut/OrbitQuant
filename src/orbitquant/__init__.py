"""OrbitQuant clean-room implementation for diffusion transformer quantization."""

from orbitquant.config import OrbitQuantConfig
from orbitquant.layers import OrbitQuantLinear
from orbitquant.quantizer import register_hf_quantizers

__version__ = "0.1.0"

register_hf_quantizers()

__all__ = ["OrbitQuantConfig", "OrbitQuantLinear", "__version__", "register_hf_quantizers"]
