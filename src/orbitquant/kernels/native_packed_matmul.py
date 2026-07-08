from __future__ import annotations

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


def _load_native_packed_matmul_kernel() -> Any:
    global _NATIVE_KERNEL
    if _NATIVE_KERNEL is not None:
        return _NATIVE_KERNEL

    try:
        from kernels import get_kernel
    except Exception as exc:  # pragma: no cover - optional dependency
        _NATIVE_KERNEL = _load_importable_packed_matmul_kernel()
        if _NATIVE_KERNEL is not None:
            return _NATIVE_KERNEL
        raise RuntimeError(
            "native_packed_matmul runtime requires either an importable "
            "orbitquant_packed_matmul kernel package or the Hugging Face kernels package. "
            "Install a compatible OrbitQuant native kernel build, install `kernels`, "
            "or use runtime_mode='dequant_bf16'."
        ) from exc

    try:
        _NATIVE_KERNEL = get_kernel(_KERNEL_REPO_ID, version=_KERNEL_VERSION)
    except Exception as exc:  # pragma: no cover - environment and Hub dependent
        _NATIVE_KERNEL = _load_importable_packed_matmul_kernel()
        if _NATIVE_KERNEL is not None:
            return _NATIVE_KERNEL
        raise RuntimeError(
            "native_packed_matmul runtime could not load "
            f"{_KERNEL_REPO_ID} version {_KERNEL_VERSION}. For local development, set "
            "LOCAL_KERNELS=WaveCut/orbitquant-packed-matmul=/absolute/path/to/"
            "native-kernels/orbitquant-packed-matmul before importing OrbitQuant, "
            "or make a compatible orbitquant_packed_matmul package importable."
        ) from exc
    return _NATIVE_KERNEL


def load_native_packed_matmul_kernel() -> Any:
    return _load_native_packed_matmul_kernel()


def matmul_packed_weight_with_native_kernel(
    x: torch.Tensor,
    packed_weight_indices: torch.Tensor,
    row_norms: torch.Tensor,
    codebook: LloydMaxCodebook,
    *,
    bits: int,
    out_features: int,
    in_features: int,
    bias: torch.Tensor | None = None,
    block_m: int = 32,
    block_n: int = 64,
    block_k: int = 64,
) -> torch.Tensor:
    if x.device.type not in {"cuda", "mps"}:
        raise RuntimeError(
            "native_packed_matmul runtime requires CUDA or MPS input tensors; "
            f"got {x.device.type}."
        )
    if x.shape[-1] != in_features:
        raise ValueError(f"expected input last dimension {in_features}, got {x.shape[-1]}")

    kernel = _load_native_packed_matmul_kernel()
    return kernel.matmul_packed_weight(
        x,
        packed_weight_indices,
        row_norms,
        codebook.centroids,
        bits=bits,
        out_features=out_features,
        in_features=in_features,
        bias=bias,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
    )
