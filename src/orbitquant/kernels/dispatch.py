from __future__ import annotations

import importlib.util

import torch

from orbitquant.codebooks import LloydMaxCodebook
from orbitquant.functional import quantize_activations
from orbitquant.rotations import RPBHRotation


def _triton_available() -> bool:
    return importlib.util.find_spec("triton") is not None


def available_backends() -> dict[str, bool]:
    return {
        "cpu": True,
        "mps": bool(torch.backends.mps.is_available()),
        "triton_cuda": bool(torch.cuda.is_available() and _triton_available()),
    }


def select_backend(device: torch.device, *, requested: str = "auto") -> str:
    requested_device = torch.device(device)
    backends = available_backends()
    if requested == "auto":
        if requested_device.type == "cuda" and backends["triton_cuda"]:
            return "triton_cuda"
        if requested_device.type == "mps" and backends["mps"]:
            return "mps"
        return "cpu"
    if requested == "cpu":
        return "cpu"
    if requested == "triton_cuda":
        if not backends["triton_cuda"]:
            raise RuntimeError("CUDA/Triton activation kernel is not available in this environment")
        return "triton_cuda"
    if requested == "mps":
        if not backends["mps"]:
            raise RuntimeError("MPS activation kernel is not available in this environment")
        return "mps"
    raise ValueError(f"unknown OrbitQuant kernel backend {requested!r}")


def quantize_activations_kernel(
    x: torch.Tensor,
    *,
    rotation: RPBHRotation,
    codebook: LloydMaxCodebook,
    eps: float,
    backend: str = "auto",
) -> torch.Tensor:
    selected = select_backend(x.device, requested=backend)
    # The current CUDA and MPS paths use the reference PyTorch graph until
    # fused kernels are added. Dispatch is explicit so optimized kernels can
    # replace this branch without changing layer code or artifact format.
    if selected in {"cpu", "mps", "triton_cuda"}:
        return quantize_activations(x, rotation=rotation, codebook=codebook, eps=eps)
    raise AssertionError(f"unhandled backend {selected}")
