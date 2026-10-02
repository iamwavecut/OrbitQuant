"""Fused transformer blocks for OrbitQuant checkpoints.

``fuse(transformer)`` regroups each block's OrbitQuant projections (Q|K|V, gate|up, ...) into
matrices that run as one INT8-surrogate GEMM with the following elementwise work in its
epilogue, and replaces the block forward accordingly; the transformer stays a drop-in module
for its Diffusers pipeline. Fused checkpoints store only the fused groups and load directly
into the fused form. The runtime needs the CUDA Triton backend; importing this package does
not, so the quantizer can recognize fused checkpoints anywhere.
"""

from importlib import import_module

_EXPORTS = {
    "FusedRuntime": "orbitquant.fused.runtime",
    "finalize": "orbitquant.fused.install",
    "fuse": "orbitquant.fused.install",
    "fuse_component_artifact": "orbitquant.fused.convert",
    "prepare_skeleton": "orbitquant.fused.install",
    "release": "orbitquant.fused.install",
    "save_pretrained": "orbitquant.fused.install",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value
