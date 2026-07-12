from __future__ import annotations

from dataclasses import dataclass

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


_AVAILABLE_BACKENDS_CACHE: BackendAvailability | None = None


def clear_backend_availability_cache() -> None:
    """Reset the memoized backend availability (tests and diagnostics)."""
    global _AVAILABLE_BACKENDS_CACHE
    _AVAILABLE_BACKENDS_CACHE = None


def available_backends() -> BackendAvailability:
    # Probing availability imports triton and touches every accelerator
    # runtime; none of that changes within a process, and this runs on the
    # per-forward dispatch path, so memoize the result.
    global _AVAILABLE_BACKENDS_CACHE
    if _AVAILABLE_BACKENDS_CACHE is None:
        triton_available = _triton_available()
        triton_gpu_available = bool(torch.cuda.is_available() and triton_available)
        triton_xpu_available = bool(_xpu_available() and triton_available)
        uses_hip = _torch_uses_hip()
        _AVAILABLE_BACKENDS_CACHE = {
            "cpu": True,
            "mps": bool(torch.backends.mps.is_available()),
            "triton_cuda": bool(triton_gpu_available and not uses_hip),
            "triton_rocm": bool(triton_gpu_available and uses_hip),
            "triton_xpu": triton_xpu_available,
        }
    return dict(_AVAILABLE_BACKENDS_CACHE)


def _capability_entry(
    *,
    available: bool,
    claim_status: str,
    optimized: bool,
    implemented_stage: str,
    optimized_stage: str | None,
    device_types: list[str],
    implementation: str,
    package_format: str,
    hf_kernel_builder_compliant: bool,
    notes: str,
    weight_dequant_optimized: bool = False,
    weight_pack_optimized: bool = False,
    lowbit_unpack_optimized: bool = False,
    weight_quant_optimized: bool = False,
    adaln_quant_optimized: bool = False,
    adaln_dequant_optimized: bool = False,
    adaln_packed_matmul_optimized: bool = False,
    upstream_native_mps_op: bool | None = None,
) -> dict[str, object]:
    """Assemble one backend capability payload with the canonical key set and order."""
    entry: dict[str, object] = {
        "available": available,
        "claim_status": claim_status,
        "optimized": optimized,
        "full_fusion": False,
        "implemented_stage": implemented_stage,
        "optimized_stage": optimized_stage,
        "weight_dequant_optimized": weight_dequant_optimized,
        "weight_pack_optimized": weight_pack_optimized,
        "lowbit_unpack_optimized": lowbit_unpack_optimized,
        "weight_quant_optimized": weight_quant_optimized,
        "adaln_quant_optimized": adaln_quant_optimized,
        "adaln_dequant_optimized": adaln_dequant_optimized,
        "adaln_packed_matmul_optimized": adaln_packed_matmul_optimized,
        "device_types": device_types,
        "implementation": implementation,
        "package_format": package_format,
    }
    if upstream_native_mps_op is not None:
        entry["upstream_native_mps_op"] = upstream_native_mps_op
    entry["hf_kernel_builder_compliant"] = hf_kernel_builder_compliant
    entry["notes"] = notes
    return entry


# Per-stage CPU rows: (optimized stage name, note when native, note when reference).
# Row order matches the (activation, matmul, adaln) flag tuple in _cpu_capability.
_CPU_STAGE_TABLE: tuple[tuple[str, str, str], ...] = (
    (
        "activation_norm_rpbh_quant_rescale",
        "The native exact activation pipeline performs FP32 token norm, "
        "RPBH/FWHT, codebook assignment, and rescale.",
        "Activation quantization uses the reference PyTorch CPU path.",
    ),
    (
        "packed_weight_matmul",
        "Native exact packed matmul consumes low-bit weights without a full "
        "floating-point cache.",
        "The main linear matmul uses the reference PyTorch CPU path.",
    ),
    (
        "adaln_rtn_packed_matmul",
        "Native packed INT4 AdaLN matmul consumes group-64 weights without a "
        "full floating-point cache.",
        "AdaLN uses the explicit BF16 dequantization path.",
    ),
)

# (activation optimized, matmul optimized) -> implementation label.
_CPU_IMPLEMENTATIONS: dict[tuple[bool, bool], str] = {
    (True, True): "native_exact_activation+native_exact_packed_matmul",
    (True, False): "native_exact_activation+reference_weight_matmul",
    (False, True): "native_exact_packed_matmul+torch_activation_reference",
    (False, False): "torch_reference",
}


def _cpu_capability(available: bool) -> dict[str, object]:
    matmul_optimized = bool(available and _native_cpu_packed_matmul_available())
    activation_optimized = bool(available and _native_cpu_activation_available())
    adaln_optimized = bool(available and _native_cpu_adaln_available())
    stage_flags = (activation_optimized, matmul_optimized, adaln_optimized)
    optimized_stages = _join_stages(
        [
            stage
            for enabled, (stage, _, _) in zip(stage_flags, _CPU_STAGE_TABLE, strict=True)
            if enabled
        ]
    )
    optimized = optimized_stages is not None
    implementation = _CPU_IMPLEMENTATIONS[(activation_optimized, matmul_optimized)]
    if adaln_optimized:
        implementation += "+native_packed_adaln_int4"
    notes = " ".join(
        native_note if enabled else reference_note
        for enabled, (_, native_note, reference_note) in zip(
            stage_flags, _CPU_STAGE_TABLE, strict=True
        )
    )
    return _capability_entry(
        available=available,
        claim_status="partial_optimized" if optimized else "reference_only",
        optimized=optimized,
        implemented_stage=_CPU_IMPLEMENTED_STAGE,
        optimized_stage=optimized_stages,
        adaln_dequant_optimized=adaln_optimized,
        adaln_packed_matmul_optimized=adaln_optimized,
        device_types=["cpu"],
        implementation=implementation,
        package_format=(
            "native_kernel_package_torch_stable_abi" if optimized else "torch_reference"
        ),
        hf_kernel_builder_compliant=optimized,
        notes=notes,
    )


# (shader optimized, native packed matmul optimized)
# -> (implementation, package_format, notes).
_MPS_VARIANTS: dict[tuple[bool, bool], tuple[str, str, str]] = {
    (True, True): (
        "torch_mps_compile_shader_fused_activation+native_packed_matmul",
        "torch.mps.compile_shader,native_kernel_package",
        "Fused Metal shaders handle activation norm, RPBH/FWHT rotation, codebook "
        "lookup/rescale, and packed weight dequant. The native packed matmul "
        "package handles packed low-bit matmul.",
    ),
    (True, False): (
        "torch_mps_compile_shader_fused_activation",
        "torch.mps.compile_shader",
        "Fused Metal shaders handle activation norm, RPBH/FWHT rotation, codebook "
        "lookup/rescale, and packed weight dequant. The native packed matmul "
        "package is not available in this environment.",
    ),
    (False, True): (
        "native_packed_matmul",
        "native_kernel_package",
        "The native packed matmul package handles packed low-bit matmul. "
        "Activation quantization helpers use the reference PyTorch path in this "
        "environment.",
    ),
    (False, False): (
        "torch_reference_mps",
        "torch_reference",
        "Runs the reference PyTorch path on MPS tensors; native Metal shader "
        "support and the native packed matmul package are not available in this "
        "environment.",
    ),
}


def _mps_capability(available: bool) -> dict[str, object]:
    shader_optimized = bool(available and _mps_metal_available())
    native_matmul_optimized = bool(available and _native_packed_matmul_available())
    optimized_stages = _join_stages(
        [
            *([_MPS_SHADER_STAGE] if shader_optimized else []),
            *([_MPS_NATIVE_STAGE] if native_matmul_optimized else []),
        ]
    )
    optimized = optimized_stages is not None
    implementation, package_format, notes = _MPS_VARIANTS[
        (shader_optimized, native_matmul_optimized)
    ]
    return _capability_entry(
        available=available,
        claim_status="partial_optimized" if optimized else "reference_only",
        optimized=optimized,
        implemented_stage=_MPS_IMPLEMENTED_STAGE,
        optimized_stage=optimized_stages,
        weight_dequant_optimized=shader_optimized,
        device_types=["mps"],
        implementation=implementation,
        package_format=package_format,
        upstream_native_mps_op=False,
        hf_kernel_builder_compliant=False,
        notes=notes,
    )


@dataclass(frozen=True)
class _TritonBackendSpec:
    """Static description of one Triton backend flavor."""

    implemented_stage: str
    optimized_when_available: bool
    available_claim_status: str
    device_type: str
    implementation: str
    package_format: str
    notes: str


_TRITON_SPECS: dict[str, _TritonBackendSpec] = {
    "triton_cuda": _TritonBackendSpec(
        implemented_stage=_TRITON_CUDA_IMPLEMENTED_STAGE,
        optimized_when_available=True,
        available_claim_status="partial_optimized",
        device_type="cuda",
        implementation="python_triton_orbitquant_pipeline",
        package_format="python_triton",
        notes=(
            "Triton handles runtime activation norm, RPBH/FWHT rotation, codebook "
            "lookup/rescale, packed weight dequant, packed weight matmul, "
            "offline low-bit pack/unpack, offline weight RPBH/FWHT codebook indexing "
            "with direct low-bit packing, and AdaLN INT4 RTN quantize/pack, "
            "dequant, and packed matmul. "
            "The default auto_fused runtime prefers packed low-bit matmul when a "
            "native or Triton packed kernel is available; full-model speedup claims "
            "still require separate benchmark artifacts."
        ),
    ),
    "triton_rocm": _TritonBackendSpec(
        implemented_stage=_TRITON_ROCM_IMPLEMENTED_STAGE,
        optimized_when_available=False,
        available_claim_status="experimental_unverified",
        device_type="cuda",
        implementation="python_triton_orbitquant_pipeline_rocm_candidate",
        package_format="python_triton_rocm",
        notes=(
            "PyTorch exposes HIP tensors through the cuda device type. The candidate "
            "reuses the exact packed Triton pipeline without loading CUDA-native "
            "extensions. It remains experimental until correctness, memory, profiler, "
            "and performance evidence is recorded on supported AMD hardware."
        ),
    ),
    "triton_xpu": _TritonBackendSpec(
        implemented_stage=_TRITON_XPU_IMPLEMENTED_STAGE,
        optimized_when_available=False,
        available_claim_status="experimental_unverified",
        device_type="xpu",
        implementation="python_triton_orbitquant_pipeline_xpu_candidate",
        package_format="python_triton_xpu",
        notes=(
            "The candidate reuses the exact packed Triton pipeline on torch.xpu. "
            "It remains explicit-only and experimental until correctness, memory, "
            "profiler, and performance evidence is recorded on supported Intel GPU "
            "hardware."
        ),
    ),
}


def _triton_capability(spec: _TritonBackendSpec, available: bool) -> dict[str, object]:
    optimized = bool(available and spec.optimized_when_available)
    return _capability_entry(
        available=available,
        claim_status=spec.available_claim_status if available else "unavailable",
        optimized=optimized,
        implemented_stage=spec.implemented_stage,
        optimized_stage=spec.implemented_stage if optimized else None,
        weight_dequant_optimized=optimized,
        weight_pack_optimized=optimized,
        lowbit_unpack_optimized=optimized,
        weight_quant_optimized=optimized,
        adaln_quant_optimized=optimized,
        adaln_dequant_optimized=optimized,
        adaln_packed_matmul_optimized=optimized,
        device_types=[spec.device_type],
        implementation=spec.implementation,
        package_format=spec.package_format,
        hf_kernel_builder_compliant=False,
        notes=spec.notes,
    )


def backend_capabilities(backends: BackendAvailability | None = None) -> BackendCapabilities:
    available = available_backends() if backends is None else backends
    capabilities: BackendCapabilities = {
        "cpu": _cpu_capability(bool(available["cpu"])),
        "mps": _mps_capability(bool(available["mps"])),
    }
    for name, spec in _TRITON_SPECS.items():
        capabilities[name] = _triton_capability(spec, bool(available.get(name, False)))
    return capabilities


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
