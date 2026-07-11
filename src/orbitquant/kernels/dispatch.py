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
_CPU_IMPLEMENTED_STAGE = (
    "activation_norm_rpbh_quant_rescale,packed_weight_matmul,adaln_rtn_packed_matmul"
)
_TRITON_CUDA_IMPLEMENTED_STAGE = (
    "activation_norm_rpbh_quant_rescale,packed_weight_dequant,"
    "packed_weight_matmul,lowbit_pack,lowbit_unpack,weight_rotation_fwht_quant_pack,"
    "adaln_rtn_quant_pack,adaln_rtn_dequant,adaln_rtn_packed_matmul"
)
_TRITON_ROCM_IMPLEMENTED_STAGE = _TRITON_CUDA_IMPLEMENTED_STAGE
_TRITON_XPU_IMPLEMENTED_STAGE = _TRITON_CUDA_IMPLEMENTED_STAGE


def _torch_uses_hip() -> bool:
    return bool(getattr(torch.version, "hip", None))


def _xpu_available() -> bool:
    xpu = getattr(torch, "xpu", None)
    return bool(xpu is not None and xpu.is_available())


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


def _native_cpu_packed_matmul_available() -> bool:
    try:
        from orbitquant.kernels.native_packed_matmul import (
            native_packed_matmul_device_available,
        )

        return native_packed_matmul_device_available("cpu")
    except Exception:
        return False


def _native_cpu_activation_available() -> bool:
    try:
        from orbitquant.kernels.native_packed_matmul import (
            native_cpu_activation_available,
        )

        return native_cpu_activation_available()
    except Exception:
        return False


def _native_cpu_adaln_available() -> bool:
    try:
        from orbitquant.kernels.native_packed_matmul import (
            native_cpu_adaln_available,
        )

        return native_cpu_adaln_available()
    except Exception:
        return False


def _join_stages(stages: list[str]) -> str | None:
    return ",".join(stages) if stages else None


def available_backends() -> BackendAvailability:
    triton_gpu_available = bool(torch.cuda.is_available() and _triton_available())
    triton_xpu_available = bool(_xpu_available() and _triton_available())
    uses_hip = _torch_uses_hip()
    return {
        "cpu": True,
        "mps": bool(torch.backends.mps.is_available()),
        "triton_cuda": bool(triton_gpu_available and not uses_hip),
        "triton_rocm": bool(triton_gpu_available and uses_hip),
        "triton_xpu": triton_xpu_available,
    }


def backend_capabilities(backends: BackendAvailability | None = None) -> BackendCapabilities:
    available = available_backends() if backends is None else backends
    triton_cuda_available = bool(available.get("triton_cuda", False))
    triton_rocm_available = bool(available.get("triton_rocm", False))
    triton_xpu_available = bool(available.get("triton_xpu", False))
    cpu_native_matmul_optimized = bool(available["cpu"] and _native_cpu_packed_matmul_available())
    cpu_native_activation_optimized = bool(available["cpu"] and _native_cpu_activation_available())
    cpu_native_adaln_optimized = bool(available["cpu"] and _native_cpu_adaln_available())
    cpu_optimized_stages = _join_stages(
        [
            *(["activation_norm_rpbh_quant_rescale"] if cpu_native_activation_optimized else []),
            *(["packed_weight_matmul"] if cpu_native_matmul_optimized else []),
            *(["adaln_rtn_packed_matmul"] if cpu_native_adaln_optimized else []),
        ]
    )
    cpu_implementation = (
        "native_exact_activation+native_exact_packed_matmul"
        if cpu_native_activation_optimized and cpu_native_matmul_optimized
        else "native_exact_packed_matmul+torch_activation_reference"
        if cpu_native_matmul_optimized
        else "native_exact_activation+reference_weight_matmul"
        if cpu_native_activation_optimized
        else "torch_reference"
    )
    if cpu_native_adaln_optimized:
        cpu_implementation += "+native_packed_adaln_int4"
    cpu_notes = []
    if cpu_native_activation_optimized:
        cpu_notes.append(
            "The native exact activation pipeline performs FP32 token norm, "
            "RPBH/FWHT, codebook assignment, and rescale."
        )
    else:
        cpu_notes.append("Activation quantization uses the reference PyTorch CPU path.")
    if cpu_native_matmul_optimized:
        cpu_notes.append(
            "Native exact packed matmul consumes low-bit weights without a full "
            "floating-point cache."
        )
    else:
        cpu_notes.append("The main linear matmul uses the reference PyTorch CPU path.")
    if cpu_native_adaln_optimized:
        cpu_notes.append(
            "Native packed INT4 AdaLN matmul consumes group-64 weights without a "
            "full floating-point cache."
        )
    else:
        cpu_notes.append("AdaLN uses the explicit BF16 dequantization path.")
    mps_shader_optimized = bool(available["mps"] and _mps_metal_available())
    mps_native_matmul_optimized = bool(available["mps"] and _native_packed_matmul_available())
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
            "claim_status": "partial_optimized"
            if cpu_optimized_stages is not None
            else "reference_only",
            "optimized": cpu_optimized_stages is not None,
            "full_fusion": False,
            "implemented_stage": _CPU_IMPLEMENTED_STAGE,
            "optimized_stage": cpu_optimized_stages,
            "weight_dequant_optimized": False,
            "weight_pack_optimized": False,
            "lowbit_unpack_optimized": False,
            "weight_quant_optimized": False,
            "adaln_quant_optimized": False,
            "adaln_dequant_optimized": cpu_native_adaln_optimized,
            "adaln_packed_matmul_optimized": cpu_native_adaln_optimized,
            "device_types": ["cpu"],
            "implementation": cpu_implementation,
            "package_format": "native_kernel_package_torch_stable_abi"
            if cpu_optimized_stages is not None
            else "torch_reference",
            "hf_kernel_builder_compliant": cpu_optimized_stages is not None,
            "notes": " ".join(cpu_notes),
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
            "adaln_packed_matmul_optimized": False,
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
            "available": triton_cuda_available,
            "claim_status": "partial_optimized" if triton_cuda_available else "unavailable",
            "optimized": triton_cuda_available,
            "full_fusion": False,
            "implemented_stage": _TRITON_CUDA_IMPLEMENTED_STAGE,
            "optimized_stage": _TRITON_CUDA_IMPLEMENTED_STAGE if triton_cuda_available else None,
            "weight_dequant_optimized": triton_cuda_available,
            "weight_pack_optimized": triton_cuda_available,
            "lowbit_unpack_optimized": triton_cuda_available,
            "weight_quant_optimized": triton_cuda_available,
            "adaln_quant_optimized": triton_cuda_available,
            "adaln_dequant_optimized": triton_cuda_available,
            "adaln_packed_matmul_optimized": triton_cuda_available,
            "device_types": ["cuda"],
            "implementation": "python_triton_orbitquant_pipeline",
            "package_format": "python_triton",
            "hf_kernel_builder_compliant": False,
            "notes": (
                "Triton handles runtime activation norm, RPBH/FWHT rotation, codebook "
                "lookup/rescale, packed weight dequant, packed weight matmul, "
                "offline low-bit pack/unpack, offline weight RPBH/FWHT codebook indexing "
                "with direct low-bit packing, and AdaLN INT4 RTN quantize/pack, "
                "dequant, and packed matmul. "
                "The default auto_fused runtime prefers packed low-bit matmul when a "
                "native or Triton packed kernel is available; full-model speedup claims "
                "still require separate benchmark artifacts."
            ),
        },
        "triton_rocm": {
            "available": triton_rocm_available,
            "claim_status": "experimental_unverified" if triton_rocm_available else "unavailable",
            "optimized": False,
            "full_fusion": False,
            "implemented_stage": _TRITON_ROCM_IMPLEMENTED_STAGE,
            "optimized_stage": None,
            "weight_dequant_optimized": False,
            "weight_pack_optimized": False,
            "lowbit_unpack_optimized": False,
            "weight_quant_optimized": False,
            "adaln_quant_optimized": False,
            "adaln_dequant_optimized": False,
            "adaln_packed_matmul_optimized": False,
            "device_types": ["cuda"],
            "implementation": "python_triton_orbitquant_pipeline_rocm_candidate",
            "package_format": "python_triton_rocm",
            "hf_kernel_builder_compliant": False,
            "notes": (
                "PyTorch exposes HIP tensors through the cuda device type. The candidate "
                "reuses the exact packed Triton pipeline without loading CUDA-native "
                "extensions. It remains experimental until correctness, memory, profiler, "
                "and performance evidence is recorded on supported AMD hardware."
            ),
        },
        "triton_xpu": {
            "available": triton_xpu_available,
            "claim_status": "experimental_unverified" if triton_xpu_available else "unavailable",
            "optimized": False,
            "full_fusion": False,
            "implemented_stage": _TRITON_XPU_IMPLEMENTED_STAGE,
            "optimized_stage": None,
            "weight_dequant_optimized": False,
            "weight_pack_optimized": False,
            "lowbit_unpack_optimized": False,
            "weight_quant_optimized": False,
            "adaln_quant_optimized": False,
            "adaln_dequant_optimized": False,
            "adaln_packed_matmul_optimized": False,
            "device_types": ["xpu"],
            "implementation": "python_triton_orbitquant_pipeline_xpu_candidate",
            "package_format": "python_triton_xpu",
            "hf_kernel_builder_compliant": False,
            "notes": (
                "The candidate reuses the exact packed Triton pipeline on torch.xpu. "
                "It remains explicit-only and experimental until correctness, memory, "
                "profiler, and performance evidence is recorded on supported Intel GPU "
                "hardware."
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
        if (
            requested_device.type == "cuda"
            and not _torch_uses_hip()
            and available.get("triton_cuda", False)
        ):
            return "triton_cuda"
        if requested_device.type == "mps" and available["mps"]:
            return "mps"
        return "cpu"
    if requested == "cpu":
        return "cpu"
    if requested == "triton_cuda":
        if not available.get("triton_cuda", False):
            raise RuntimeError("CUDA/Triton activation kernel is not available in this environment")
        return "triton_cuda"
    if requested == "triton_rocm":
        if not available.get("triton_rocm", False):
            raise RuntimeError("ROCm/Triton activation kernel is not available in this environment")
        return "triton_rocm"
    if requested == "triton_xpu":
        if not available.get("triton_xpu", False):
            raise RuntimeError("XPU/Triton activation kernel is not available in this environment")
        if requested_device.type != "xpu":
            raise RuntimeError("XPU/Triton activation kernel requires an XPU device")
        return "triton_xpu"
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


def _triton_quantize_activations(
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
    if selected in {"triton_cuda", "triton_rocm", "triton_xpu"}:
        return _triton_quantize_activations(
            x,
            rotation=rotation,
            codebook=codebook,
            eps=eps,
            constant_tensors=constant_tensors,
        )
    raise AssertionError(f"unhandled backend {selected}")
