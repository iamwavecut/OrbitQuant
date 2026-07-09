from __future__ import annotations

import torch

from orbitquant.codebooks import LloydMaxCodebook
from orbitquant.functional import quantize_activations
from orbitquant.rotations import RPBHRotation

BackendAvailability = dict[str, bool]
BackendCapabilities = dict[str, dict[str, object]]

_MPS_SHADER_STAGE = "activation_norm_rpbh_quant_rescale,packed_weight_dequant"
_MPS_NATIVE_STAGE = "packed_weight_matmul"
_MPS_IMPLEMENTED_STAGE = f"{_MPS_SHADER_STAGE},{_MPS_NATIVE_STAGE}"
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


def _native_packed_matmul_available() -> bool:
    try:
        from orbitquant.kernels.native_packed_matmul import load_native_packed_matmul_kernel

        load_native_packed_matmul_kernel()
    except Exception:
        return False
    return True


def _join_stages(stages: list[str]) -> str | None:
    return ",".join(stages) if stages else None


def available_backends() -> BackendAvailability:
    return {
        "cpu": True,
        "mps": bool(torch.backends.mps.is_available()),
        "triton_cuda": bool(torch.cuda.is_available() and _triton_available()),
    }


def backend_capabilities(backends: BackendAvailability | None = None) -> BackendCapabilities:
    available = available_backends() if backends is None else backends
    mps_shader_optimized = bool(available["mps"] and _mps_metal_available())
    mps_native_matmul_optimized = bool(
        available["mps"] and _native_packed_matmul_available()
    )
    mps_optimized_stages = _join_stages(
        [
            *([_MPS_SHADER_STAGE] if mps_shader_optimized else []),
            *([_MPS_NATIVE_STAGE] if mps_native_matmul_optimized else []),
        ]
    )
    mps_optimized = mps_optimized_stages is not None
    if mps_shader_optimized and mps_native_matmul_optimized:
        mps_notes = (
            "Fused Metal shaders handle activation norm, RPBH/FWHT rotation, codebook "
            "lookup/rescale, and packed weight dequant. The native packed matmul "
            "package handles packed low-bit matmul."
        )
    elif mps_shader_optimized:
        mps_notes = (
            "Fused Metal shaders handle activation norm, RPBH/FWHT rotation, codebook "
            "lookup/rescale, and packed weight dequant. The native packed matmul "
            "package is not available in this environment."
        )
    elif mps_native_matmul_optimized:
        mps_notes = (
            "The native packed matmul package handles packed low-bit matmul. "
            "Activation quantization helpers use the reference PyTorch path in this "
            "environment."
        )
    else:
        mps_notes = (
            "Runs the reference PyTorch path on MPS tensors; native Metal shader "
            "support and the native packed matmul package are not available in this "
            "environment."
        )
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
            "optimized_stage": mps_optimized_stages,
            "weight_dequant_optimized": mps_shader_optimized,
            "weight_pack_optimized": False,
            "lowbit_unpack_optimized": False,
            "weight_quant_optimized": False,
            "adaln_quant_optimized": False,
            "adaln_dequant_optimized": False,
            "device_types": ["mps"],
            "implementation": (
                "torch_mps_compile_shader_fused_activation+native_packed_matmul"
                if mps_shader_optimized and mps_native_matmul_optimized
                else "torch_mps_compile_shader_fused_activation"
                if mps_shader_optimized
                else "native_packed_matmul"
                if mps_native_matmul_optimized
                else "torch_reference_mps"
            ),
            "package_format": (
                "torch.mps.compile_shader,native_kernel_package"
                if mps_shader_optimized and mps_native_matmul_optimized
                else "torch.mps.compile_shader"
                if mps_shader_optimized
                else "native_kernel_package"
                if mps_native_matmul_optimized
                else "torch_reference"
            ),
            "upstream_native_mps_op": False,
            "hf_kernel_builder_compliant": False,
            "notes": mps_notes,
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
                "lookup/rescale, packed weight dequant, packed weight matmul, "
                "offline low-bit pack/unpack, offline weight RPBH/FWHT codebook indexing "
                "with direct low-bit packing, and AdaLN INT4 RTN quantize/pack/dequant. "
                "The default auto_fused runtime prefers packed low-bit matmul when a "
                "native or Triton packed kernel is available; full-model speedup claims "
                "still require separate benchmark artifacts."
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
    from orbitquant.kernels.mps import quantize_activations_with_mps

    return quantize_activations_with_mps(
        x,
        rotation,
        codebook,
        eps=eps,
        constant_tensors=constant_tensors,
    )


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
