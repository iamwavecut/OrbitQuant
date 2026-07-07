from __future__ import annotations

import importlib.util

import torch

from orbitquant.codebooks import LloydMaxCodebook
from orbitquant.functional import quantize_activations
from orbitquant.rotations import RPBHRotation

BackendAvailability = dict[str, bool]
BackendCapabilities = dict[str, dict[str, object]]


def _triton_available() -> bool:
    return importlib.util.find_spec("triton") is not None


def _mps_metal_available() -> bool:
    try:
        from orbitquant.kernels.mps import mps_metal_available
    except Exception:
        return False
    return mps_metal_available()


def available_backends() -> BackendAvailability:
    return {
        "cpu": True,
        "mps": bool(torch.backends.mps.is_available()),
        "triton_cuda": bool(torch.cuda.is_available() and _triton_available()),
    }


def backend_capabilities(backends: BackendAvailability | None = None) -> BackendCapabilities:
    available = available_backends() if backends is None else backends
    mps_optimized = bool(available["mps"] and _mps_metal_available())
    return {
        "cpu": {
            "available": bool(available["cpu"]),
            "optimized": False,
            "full_fusion": False,
            "optimized_stage": None,
            "weight_dequant_optimized": False,
            "weight_pack_optimized": False,
            "weight_quant_optimized": False,
            "adaln_quant_optimized": False,
            "adaln_dequant_optimized": False,
            "device_types": ["cpu"],
            "implementation": "torch_reference",
            "notes": "Correctness baseline using the reference PyTorch path.",
        },
        "mps": {
            "available": bool(available["mps"]),
            "optimized": mps_optimized,
            "full_fusion": False,
            "optimized_stage": "codebook_lookup_rescale" if mps_optimized else None,
            "weight_dequant_optimized": mps_optimized,
            "weight_pack_optimized": False,
            "weight_quant_optimized": False,
            "adaln_quant_optimized": False,
            "adaln_dequant_optimized": False,
            "device_types": ["mps"],
            "implementation": "metal_codebook_rescale" if mps_optimized else "torch_reference_mps",
            "notes": (
                "Norm and RPBH rotation still run in PyTorch; a Metal shader "
                "handles codebook lookup, norm rescale, and packed weight dequant."
                if mps_optimized
                else (
                    "Runs the reference PyTorch path on MPS tensors; native Metal "
                    "shader support is not available in this environment."
                )
            ),
        },
        "triton_cuda": {
            "available": bool(available["triton_cuda"]),
            "optimized": bool(available["triton_cuda"]),
            "full_fusion": False,
            "optimized_stage": (
                "activation_norm_rpbh_quant_rescale,packed_weight_dequant,"
                "lowbit_pack,weight_rotation_fwht_quant,"
                "adaln_rtn_quant_pack,adaln_rtn_dequant"
            ),
            "weight_dequant_optimized": bool(available["triton_cuda"]),
            "weight_pack_optimized": bool(available["triton_cuda"]),
            "weight_quant_optimized": bool(available["triton_cuda"]),
            "adaln_quant_optimized": bool(available["triton_cuda"]),
            "adaln_dequant_optimized": bool(available["triton_cuda"]),
            "device_types": ["cuda"],
            "implementation": "triton_codebook_rescale",
            "notes": (
                "Triton handles runtime activation norm, RPBH/FWHT rotation, codebook "
                "lookup/rescale, packed weight dequant, offline low-bit pack, offline "
                "weight RPBH/FWHT codebook indexing, and AdaLN INT4 RTN "
                "quantize/pack/dequant. Matmul is still the BF16 PyTorch linear path."
            ),
        },
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
    if not _mps_metal_available():
        return _reference_quantize_activations(x, rotation=rotation, codebook=codebook, eps=eps)
    from orbitquant.kernels.mps import quantize_rotated_activations_with_mps

    original_dtype = x.dtype
    work = x.to(torch.float32)
    norms = work.norm(dim=-1, keepdim=True).clamp_min(eps)
    rotated = rotation.apply_to_activations(work / norms)
    quantized = quantize_rotated_activations_with_mps(rotated, norms, codebook)
    return quantized.to(original_dtype)


def _triton_cuda_quantize_activations(
    x: torch.Tensor,
    *,
    rotation: RPBHRotation,
    codebook: LloydMaxCodebook,
    eps: float,
) -> torch.Tensor:
    from orbitquant.kernels.triton_cuda import quantize_activations_with_triton

    return quantize_activations_with_triton(x, rotation=rotation, codebook=codebook, eps=eps)


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
