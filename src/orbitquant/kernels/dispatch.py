from __future__ import annotations

import torch

from orbitquant.codebooks import LloydMaxCodebook
from orbitquant.functional import quantize_activations
from orbitquant.rotations import RPBHRotation
from orbitquant.rotations.fwht import fwht

BackendAvailability = dict[str, bool]
BackendCapabilities = dict[str, dict[str, object]]

_MPS_IMPLEMENTED_STAGE = "codebook_lookup_rescale,packed_weight_dequant"
_TRITON_CUDA_IMPLEMENTED_STAGE = (
    "activation_norm_rpbh_quant_rescale,packed_weight_dequant,"
    "packed_weight_matmul,lowbit_pack,lowbit_unpack,weight_rotation_fwht_quant_pack,"
    "adaln_rtn_quant_pack,adaln_rtn_dequant"
)


def _triton_available() -> bool:
    try:
        import triton  # noqa: F401
        import triton.language as tl  # noqa: F401
    except Exception:
        return False
    return True


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
            "claim_status": "reference_only",
            "optimized": False,
            "full_fusion": False,
            "implemented_stage": None,
            "optimized_stage": None,
            "weight_dequant_optimized": False,
            "weight_pack_optimized": False,
            "lowbit_unpack_optimized": False,
            "weight_quant_optimized": False,
            "adaln_quant_optimized": False,
            "adaln_dequant_optimized": False,
            "device_types": ["cpu"],
            "implementation": "torch_reference",
            "package_format": "torch_reference",
            "hf_kernel_builder_compliant": False,
            "notes": "Correctness baseline using the reference PyTorch path.",
        },
        "mps": {
            "available": bool(available["mps"]),
            "claim_status": "partial_optimized" if mps_optimized else "reference_only",
            "optimized": mps_optimized,
            "full_fusion": False,
            "implemented_stage": _MPS_IMPLEMENTED_STAGE,
            "optimized_stage": _MPS_IMPLEMENTED_STAGE if mps_optimized else None,
            "weight_dequant_optimized": mps_optimized,
            "weight_pack_optimized": False,
            "lowbit_unpack_optimized": False,
            "weight_quant_optimized": False,
            "adaln_quant_optimized": False,
            "adaln_dequant_optimized": False,
            "device_types": ["mps"],
            "implementation": (
                "torch_mps_compile_shader_codebook_rescale"
                if mps_optimized
                else "torch_reference_mps"
            ),
            "package_format": "torch.mps.compile_shader" if mps_optimized else "torch_reference",
            "upstream_native_mps_op": False,
            "hf_kernel_builder_compliant": False,
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
            "claim_status": "partial_optimized" if available["triton_cuda"] else "unavailable",
            "optimized": bool(available["triton_cuda"]),
            "full_fusion": False,
            "implemented_stage": _TRITON_CUDA_IMPLEMENTED_STAGE,
            "optimized_stage": _TRITON_CUDA_IMPLEMENTED_STAGE
            if available["triton_cuda"]
            else None,
            "weight_dequant_optimized": bool(available["triton_cuda"]),
            "weight_pack_optimized": bool(available["triton_cuda"]),
            "lowbit_unpack_optimized": bool(available["triton_cuda"]),
            "weight_quant_optimized": bool(available["triton_cuda"]),
            "adaln_quant_optimized": bool(available["triton_cuda"]),
            "adaln_dequant_optimized": bool(available["triton_cuda"]),
            "device_types": ["cuda"],
            "implementation": "python_triton_orbitquant_pipeline",
            "package_format": "python_triton",
            "hf_kernel_builder_compliant": False,
            "notes": (
                "Triton handles runtime activation norm, RPBH/FWHT rotation, codebook "
                "lookup/rescale, packed weight dequant, opt-in packed weight matmul, "
                "offline low-bit pack/unpack, offline weight RPBH/FWHT codebook indexing "
                "with direct low-bit packing, and AdaLN INT4 RTN quantize/pack/dequant. "
                "The default runtime mode still uses the BF16 PyTorch linear path."
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
    constant_tensors: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    if not _mps_metal_available():
        return _reference_quantize_activations(x, rotation=rotation, codebook=codebook, eps=eps)
    from orbitquant.kernels.mps import quantize_rotated_activations_with_mps

    original_dtype = x.dtype
    work = x.to(torch.float32)
    norms = work.norm(dim=-1, keepdim=True)
    unit = work / (norms + eps)
    if constant_tensors is None:
        rotated = rotation.apply_to_activations(unit)
    else:
        permutation = constant_tensors["permutation"].to(device=x.device)
        signs = constant_tensors["signs"].to(device=x.device, dtype=unit.dtype)
        rotated = unit.index_select(dim=-1, index=permutation)
        rotated = rotated * signs
        rotated = rotated.reshape(
            *rotated.shape[:-1], rotation.num_blocks, rotation.block_size
        )
        rotated = fwht(rotated) * rotation.normalization
        rotated = rotated.reshape(*unit.shape)
    quantized = quantize_rotated_activations_with_mps(
        rotated,
        norms,
        codebook,
        constant_tensors=constant_tensors,
    )
    return quantized.to(original_dtype)


def _triton_cuda_quantize_activations(
    x: torch.Tensor,
    *,
    rotation: RPBHRotation,
    codebook: LloydMaxCodebook,
    eps: float,
    constant_tensors: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    from orbitquant.kernels.triton_cuda import quantize_activations_with_triton

    return quantize_activations_with_triton(
        x,
        rotation=rotation,
        codebook=codebook,
        eps=eps,
        constant_tensors=constant_tensors,
    )


def quantize_activations_kernel(
    x: torch.Tensor,
    *,
    rotation: RPBHRotation,
    codebook: LloydMaxCodebook,
    eps: float,
    backend: str = "auto",
    constant_tensors: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    selected = select_backend(x.device, requested=backend)
    if selected == "cpu":
        return _reference_quantize_activations(x, rotation=rotation, codebook=codebook, eps=eps)
    if selected == "mps":
        return _mps_quantize_activations(
            x,
            rotation=rotation,
            codebook=codebook,
            eps=eps,
            constant_tensors=constant_tensors,
        )
    if selected == "triton_cuda":
        return _triton_cuda_quantize_activations(
            x,
            rotation=rotation,
            codebook=codebook,
            eps=eps,
            constant_tensors=constant_tensors,
        )
    raise AssertionError(f"unhandled backend {selected}")
