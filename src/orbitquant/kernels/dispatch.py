from __future__ import annotations

import importlib.util

import torch

from orbitquant.codebooks import LloydMaxCodebook
from orbitquant.functional import quantize_activations
from orbitquant.rotations import RPBHRotation

BackendAvailability = dict[str, bool]


def _triton_available() -> bool:
    return importlib.util.find_spec("triton") is not None


def available_backends() -> BackendAvailability:
    return {
        "cpu": True,
        "mps": bool(torch.backends.mps.is_available()),
        "triton_cuda": bool(torch.cuda.is_available() and _triton_available()),
    }


def select_backend(
    device: torch.device,
    *,
    requested: str = "auto",
    backends: BackendAvailability | None = None,
) -> str:
    requested_device = torch.device(device)
    available = available_backends() if backends is None else backends
    if requested == "auto":
        if requested_device.type == "cuda" and available["triton_cuda"]:
            return "triton_cuda"
        if requested_device.type == "mps" and available["mps"]:
            return "mps"
        return "cpu"
    if requested == "cpu":
        return "cpu"
    if requested == "triton_cuda":
        if not available["triton_cuda"]:
            raise RuntimeError("CUDA/Triton activation kernel is not available in this environment")
        return "triton_cuda"
    if requested == "mps":
        if not available["mps"]:
            raise RuntimeError("MPS activation kernel is not available in this environment")
        return "mps"
    raise ValueError(f"unknown OrbitQuant kernel backend {requested!r}")


def _reference_quantize_activations(
    x: torch.Tensor,
    *,
    rotation: RPBHRotation,
    codebook: LloydMaxCodebook,
    eps: float,
) -> torch.Tensor:
    return quantize_activations(x, rotation=rotation, codebook=codebook, eps=eps)


def _mps_quantize_activations(
    x: torch.Tensor,
    *,
    rotation: RPBHRotation,
    codebook: LloydMaxCodebook,
    eps: float,
) -> torch.Tensor:
    return _reference_quantize_activations(x, rotation=rotation, codebook=codebook, eps=eps)


def _triton_cuda_quantize_activations(
    x: torch.Tensor,
    *,
    rotation: RPBHRotation,
    codebook: LloydMaxCodebook,
    eps: float,
) -> torch.Tensor:
    return _reference_quantize_activations(x, rotation=rotation, codebook=codebook, eps=eps)


def quantize_activations_kernel(
    x: torch.Tensor,
    *,
    rotation: RPBHRotation,
    codebook: LloydMaxCodebook,
    eps: float,
    backend: str = "auto",
) -> torch.Tensor:
    selected = select_backend(x.device, requested=backend)
    if selected == "cpu":
        return _reference_quantize_activations(x, rotation=rotation, codebook=codebook, eps=eps)
    if selected == "mps":
        return _mps_quantize_activations(x, rotation=rotation, codebook=codebook, eps=eps)
    if selected == "triton_cuda":
        return _triton_cuda_quantize_activations(
            x, rotation=rotation, codebook=codebook, eps=eps
        )
    raise AssertionError(f"unhandled backend {selected}")
