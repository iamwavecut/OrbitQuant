from __future__ import annotations

import platform
import re
import sys
from typing import Any

import torch

from orbitquant.codebooks import LloydMaxCodebook

_KERNEL_REPO_ID = "WaveCut/orbitquant-packed-matmul"
_KERNEL_VERSION = 1
_NATIVE_KERNEL: Any | None = None


def _load_importable_packed_matmul_kernel() -> Any | None:
    try:
        import orbitquant_packed_matmul
    except Exception:
        return None
    return orbitquant_packed_matmul


def _runtime_variant_hint() -> str:
    cuda_version = torch.version.cuda
    accelerator = f"CUDA {cuda_version}" if cuda_version is not None else "non-CUDA"
    hint = (
        "Current runtime is "
        f"torch {torch.__version__}, {accelerator}, "
        f"{platform.system().lower()} {platform.machine()}. "
        "The built kernel variant must match this runtime."
    )
    torch_match = re.match(r"^(\d+)\.(\d+)", torch.__version__)
    if cuda_version is not None and sys.platform == "linux" and torch_match is not None:
        expected_variant = (
            f"torch{torch_match.group(1)}{torch_match.group(2)}-cxx11-"
            f"cu{cuda_version.replace('.', '')}-{platform.machine()}-linux"
        )
        hint = f"{hint} Expected kernel-builder CUDA variant: {expected_variant}."
    return hint


def _load_native_packed_matmul_kernel() -> Any:
    global _NATIVE_KERNEL
    if _NATIVE_KERNEL is not None:
        return _NATIVE_KERNEL

    _NATIVE_KERNEL = _load_importable_packed_matmul_kernel()
    if _NATIVE_KERNEL is not None:
        return _NATIVE_KERNEL

    try:
        from kernels import get_kernel
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "native_packed_matmul runtime requires either an importable "
            "orbitquant_packed_matmul kernel package or the Hugging Face kernels package. "
            "Install `kernels` and point LOCAL_KERNELS at a compatible "
            "built kernel variant directory containing metadata.json, make the "
            "compatible variant directory importable through PYTHONPATH, or use "
            f"runtime_mode='dequant_bf16'. {_runtime_variant_hint()}"
        ) from exc

    try:
        _NATIVE_KERNEL = get_kernel(
            _KERNEL_REPO_ID,
            version=_KERNEL_VERSION,
            trust_remote_code=True,
        )
    except Exception as exc:  # pragma: no cover - environment and Hub dependent
        raise RuntimeError(
            "native_packed_matmul runtime could not load "
            f"{_KERNEL_REPO_ID} version {_KERNEL_VERSION}. For local development, set "
            "LOCAL_KERNELS=WaveCut/orbitquant-packed-matmul=/absolute/path/to/a/"
            "built kernel variant directory that contains metadata.json before "
            "importing OrbitQuant; add that same variant directory to PYTHONPATH; "
            "or make a compatible orbitquant_packed_matmul package importable. "
            f"{_runtime_variant_hint()}"
        ) from exc
    return _NATIVE_KERNEL


def load_native_packed_matmul_kernel() -> Any:
    return _load_native_packed_matmul_kernel()


def native_packed_matmul_device_available(device_type: str) -> bool:
    if device_type not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"unknown device type {device_type!r}")
    kernel = _load_native_packed_matmul_kernel()
    supports_device = getattr(kernel, "supports_device", None)
    if callable(supports_device):
        return bool(supports_device(device_type))
    # Kernel variants built before CPU support did not expose capability
    # introspection. They were CUDA- or Metal-only, never CPU variants.
    return device_type in {"cuda", "mps"}


def native_cpu_activation_available() -> bool:
    kernel = _load_native_packed_matmul_kernel()
    supports_cpu_activation = getattr(kernel, "supports_cpu_activation", None)
    return callable(supports_cpu_activation) and bool(supports_cpu_activation())


def native_cpu_adaln_available() -> bool:
    kernel = _load_native_packed_matmul_kernel()
    supports_cpu_adaln = getattr(kernel, "supports_cpu_adaln", None)
    return callable(supports_cpu_adaln) and bool(supports_cpu_adaln())


def quantize_activations_with_native_cpu_kernel(
    x: torch.Tensor,
    permutation: torch.Tensor,
    signs: torch.Tensor,
    centroids: torch.Tensor,
    boundaries: torch.Tensor,
    *,
    eps: float,
    inv_sqrt_block: float,
    block_size: int,
) -> torch.Tensor:
    if x.device.type != "cpu":
        raise RuntimeError("native CPU activation quantization requires CPU tensors")
    kernel = _load_native_packed_matmul_kernel()
    operation = getattr(kernel, "quantize_activations_cpu", None)
    if not callable(operation) or not native_cpu_activation_available():
        raise RuntimeError(
            "the loaded native packed matmul package does not provide the CPU "
            "activation pipeline. Build a current CPU variant or use the reference "
            "activation backend."
        )
    return operation(
        x,
        permutation,
        signs,
        centroids,
        boundaries,
        eps=eps,
        inv_sqrt_block=inv_sqrt_block,
        block_size=block_size,
    )


def matmul_packed_adaln_int4_with_native_cpu_kernel(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    scales: torch.Tensor,
    *,
    out_features: int,
    in_features: int,
    group_size: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if x.device.type != "cpu":
        raise RuntimeError("native packed AdaLN requires CPU tensors")
    kernel = _load_native_packed_matmul_kernel()
    operation = getattr(kernel, "matmul_packed_adaln_int4_cpu", None)
    if not callable(operation) or not native_cpu_adaln_available():
        raise RuntimeError(
            "the loaded native packed matmul package does not provide the CPU "
            "AdaLN INT4 group kernel. Build a current CPU variant or use "
            "runtime_mode='dequant_bf16'."
        )
    return operation(
        x,
        packed_weight,
        scales,
        out_features=out_features,
        in_features=in_features,
        group_size=group_size,
        bias=bias,
    )


def native_packed_w4a4_available() -> bool:
    kernel = _load_native_packed_matmul_kernel()
    return callable(getattr(kernel, "matmul_packed_w4a4_int8", None))


def native_packed_w4_activation_available() -> bool:
    kernel = _load_native_packed_matmul_kernel()
    return callable(getattr(kernel, "quantize_activations_packed_w4", None))


def native_int8_activation_available() -> bool:
    kernel = _load_native_packed_matmul_kernel()
    return callable(getattr(kernel, "quantize_activations_int8", None))


def matmul_packed_weight_with_native_kernel(
    x: torch.Tensor,
    packed_weight_indices: torch.Tensor,
    row_norms: torch.Tensor,
    codebook: LloydMaxCodebook | torch.Tensor,
    *,
    bits: int,
    out_features: int,
    in_features: int,
    bias: torch.Tensor | None = None,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 128,
) -> torch.Tensor:
    if x.device.type not in {"cpu", "cuda", "mps"}:
        raise RuntimeError(
            "native_packed_matmul runtime requires CPU, CUDA, or MPS input tensors; "
            f"got {x.device.type}."
        )
    if x.shape[-1] != in_features:
        raise ValueError(f"expected input last dimension {in_features}, got {x.shape[-1]}")

    kernel = _load_native_packed_matmul_kernel()
    if not native_packed_matmul_device_available(x.device.type):
        raise RuntimeError(
            "the loaded native packed matmul package does not contain a "
            f"{x.device.type.upper()} backend. Install or build a compatible variant, "
            "or use runtime_mode='dequant_bf16'. "
            f"{_runtime_variant_hint()}"
        )
    centroids = codebook if isinstance(codebook, torch.Tensor) else codebook.centroids
    return kernel.matmul_packed_weight(
        x,
        packed_weight_indices,
        row_norms,
        centroids,
        bits=bits,
        out_features=out_features,
        in_features=in_features,
        bias=bias,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
    )


def matmul_packed_w4a4_int8_with_native_kernel(
    packed_activations: torch.Tensor,
    packed_weight_indices: torch.Tensor,
    token_norms: torch.Tensor,
    row_norms: torch.Tensor,
    activation_codes: torch.Tensor,
    weight_codes: torch.Tensor,
    *,
    activation_scale: float,
    weight_scale: float,
    out_features: int,
    in_features: int,
    bias: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.bfloat16,
    tile_m: int = 128,
    tile_n: int = 128,
    async_packed: bool = True,
    weight_k_major: bool = False,
) -> torch.Tensor:
    if not packed_activations.is_cuda:
        raise RuntimeError("native packed W4A4 INT8 matmul requires CUDA tensors")
    kernel = _load_native_packed_matmul_kernel()
    operation = getattr(kernel, "matmul_packed_w4a4_int8", None)
    if not callable(operation):
        raise RuntimeError(
            "the loaded native packed matmul package does not provide matmul_packed_w4a4_int8"
        )
    return operation(
        packed_activations,
        packed_weight_indices,
        token_norms,
        row_norms,
        activation_codes,
        weight_codes,
        activation_scale=activation_scale,
        weight_scale=weight_scale,
        out_features=out_features,
        in_features=in_features,
        bias=bias,
        output_dtype=output_dtype,
        tile_m=tile_m,
        tile_n=tile_n,
        async_packed=async_packed,
        weight_k_major=weight_k_major,
    )


def quantize_activations_packed_w4_with_native_kernel(
    x: torch.Tensor,
    permutation: torch.Tensor,
    signs: torch.Tensor,
    boundaries: torch.Tensor,
    *,
    eps: float,
    inv_sqrt_block: float,
    threads: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not x.is_cuda:
        raise RuntimeError("native packed W4 activation quantization requires CUDA tensors")
    kernel = _load_native_packed_matmul_kernel()
    operation = getattr(kernel, "quantize_activations_packed_w4", None)
    if not callable(operation):
        raise RuntimeError(
            "the loaded native packed matmul package does not provide "
            "quantize_activations_packed_w4"
        )
    return operation(
        x,
        permutation,
        signs,
        boundaries,
        eps=eps,
        inv_sqrt_block=inv_sqrt_block,
        threads=threads,
    )


def quantize_activations_int8_with_native_kernel(
    x: torch.Tensor,
    permutation: torch.Tensor,
    signs: torch.Tensor,
    boundaries: torch.Tensor,
    codes: torch.Tensor,
    *,
    eps: float,
    inv_sqrt_block: float,
    threads: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not x.is_cuda:
        raise RuntimeError("native INT8 activation quantization requires CUDA tensors")
    kernel = _load_native_packed_matmul_kernel()
    operation = getattr(kernel, "quantize_activations_int8", None)
    if not callable(operation):
        raise RuntimeError(
            "the loaded native packed matmul package does not provide "
            "quantize_activations_int8"
        )
    return operation(
        x,
        permutation,
        signs,
        boundaries,
        codes,
        eps=eps,
        inv_sqrt_block=inv_sqrt_block,
        threads=threads,
    )
