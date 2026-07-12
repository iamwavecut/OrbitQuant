from __future__ import annotations

import torch

from ._ops import ops

__all__ = [
    "matmul_packed_w4a4_int8",
    "matmul_packed_adaln_int4_cpu",
    "matmul_packed_weight",
    "quantize_activations_cpu",
    "quantize_activations_int8",
    "quantize_activations_packed_w4",
    "supports_cpu_activation",
    "supports_cpu_adaln",
    "supports_device",
]


def supports_device(device_type: str) -> bool:
    dispatch_keys = {
        "cpu": "CPU",
        "cuda": "CUDA",
        "mps": "MPS",
    }
    try:
        dispatch_key = dispatch_keys[device_type]
    except KeyError as exc:
        raise ValueError(f"unknown device type {device_type!r}") from exc
    return bool(
        torch._C._dispatch_has_kernel_for_dispatch_key(  # noqa: SLF001
            ops.matmul_packed_weight._qualified_op_name,  # noqa: SLF001
            dispatch_key,
        )
    )


def supports_cpu_activation() -> bool:
    try:
        operation = ops.quantize_activations_cpu
        qualified_name = operation._qualified_op_name  # noqa: SLF001
    except (AttributeError, RuntimeError):
        return False
    return bool(
        torch._C._dispatch_has_kernel_for_dispatch_key(  # noqa: SLF001
            qualified_name,
            "CPU",
        )
    )


def supports_cpu_adaln() -> bool:
    try:
        operation = ops.matmul_packed_adaln_int4_cpu
        qualified_name = operation._qualified_op_name  # noqa: SLF001
    except (AttributeError, RuntimeError):
        return False
    return bool(
        torch._C._dispatch_has_kernel_for_dispatch_key(  # noqa: SLF001
            qualified_name,
            "CPU",
        )
    )


def quantize_activations_cpu(
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
    if x.dtype not in {torch.float32, torch.float16, torch.bfloat16}:
        raise ValueError("x must be float32, float16, or bfloat16")
    if block_size <= 0 or block_size & (block_size - 1):
        raise ValueError("block_size must be a positive power of two")
    dim = x.shape[-1]
    if dim % block_size != 0:
        raise ValueError("block_size must divide the input dimension")

    original_shape = x.shape
    values = x.contiguous().reshape(-1, dim)
    out = torch.empty_like(values)
    permutation_values = permutation.to(device="cpu", dtype=torch.int64).contiguous()
    sign_values = signs.to(device="cpu", dtype=torch.int8).contiguous()
    centroid_values = centroids.to(device="cpu", dtype=torch.float32).contiguous()
    boundary_values = boundaries.to(device="cpu", dtype=torch.float32).contiguous()
    ops.quantize_activations_cpu(
        out,
        values,
        permutation_values,
        sign_values,
        centroid_values,
        boundary_values,
        eps,
        inv_sqrt_block,
        block_size,
    )
    return out.reshape(original_shape)


def matmul_packed_weight(
    x: torch.Tensor,
    packed_weight_indices: torch.Tensor,
    row_norms: torch.Tensor,
    centroids: torch.Tensor,
    *,
    bits: int,
    out_features: int,
    in_features: int,
    bias: torch.Tensor | None = None,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 128,
) -> torch.Tensor:
    if bits <= 0 or bits > 8:
        raise ValueError("bits must be in [1, 8]")
    if x.shape[-1] != in_features:
        raise ValueError(f"expected input last dimension {in_features}, got {x.shape[-1]}")
    if block_m <= 0 or block_n <= 0 or block_k <= 0:
        raise ValueError("packed matmul tile sizes must be positive")

    original_shape = x.shape
    x_2d = x.contiguous().reshape(-1, in_features)
    out = torch.empty((x_2d.shape[0], out_features), device=x.device, dtype=x.dtype)
    packed = packed_weight_indices.to(device=x.device, dtype=torch.uint8).contiguous()
    auxiliary_dtype = torch.bfloat16 if x.device.type == "cuda" else torch.float32
    norms = row_norms.to(device=x.device, dtype=auxiliary_dtype).contiguous()
    centroid_values = centroids.to(device=x.device, dtype=torch.float32).contiguous()
    if bias is None:
        bias_values = torch.empty((1,), device=x.device, dtype=torch.float32)
        has_bias = False
    else:
        bias_dtype = x.dtype if x.device.type == "cuda" else torch.float32
        bias_values = bias.to(device=x.device, dtype=bias_dtype).contiguous()
        has_bias = True

    ops.matmul_packed_weight(
        out,
        x_2d,
        packed,
        norms,
        centroid_values,
        bias_values,
        has_bias,
        bits,
        out_features,
        in_features,
        block_m,
        block_n,
        block_k,
    )
    return out.reshape(*original_shape[:-1], out_features)


def matmul_packed_adaln_int4_cpu(
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
    if x.shape[-1] != in_features:
        raise ValueError(f"expected input last dimension {in_features}, got {x.shape[-1]}")
    if group_size <= 0:
        raise ValueError("group_size must be positive")

    original_shape = x.shape
    x_2d = x.to(dtype=torch.bfloat16).contiguous().reshape(-1, in_features)
    out = torch.empty((x_2d.shape[0], out_features), dtype=torch.bfloat16)
    packed = packed_weight.to(device="cpu", dtype=torch.uint8).contiguous()
    scale_values = scales.to(device="cpu", dtype=torch.float32).contiguous()
    if bias is None:
        bias_values = torch.empty((1,), dtype=torch.float32)
        has_bias = False
    else:
        bias_values = (
            bias.to(device="cpu", dtype=torch.bfloat16).to(dtype=torch.float32).contiguous()
        )
        has_bias = True
    ops.matmul_packed_adaln_int4_cpu(
        out,
        x_2d,
        packed,
        scale_values,
        bias_values,
        has_bias,
        out_features,
        in_features,
        group_size,
    )
    return out.reshape(*original_shape[:-1], out_features)


def matmul_packed_w4a4_int8(
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
    async_packed: bool = False,
    weight_k_major: bool = False,
) -> torch.Tensor:
    if not packed_activations.is_cuda:
        raise RuntimeError("packed W4A4 INT8 matmul requires CUDA tensors")
    if in_features <= 0 or in_features % 64 != 0:
        raise ValueError("in_features must be positive and divisible by 64")
    if packed_activations.shape[-1] != in_features // 2:
        raise ValueError(
            f"expected packed activation last dimension {in_features // 2}, "
            f"got {packed_activations.shape[-1]}"
        )
    if output_dtype not in {torch.float16, torch.bfloat16}:
        raise ValueError("output_dtype must be float16 or bfloat16")
    if (tile_m, tile_n) not in {(128, 128), (256, 128), (128, 256)}:
        raise ValueError("tile must be 128x128, 256x128, or 128x256")

    original_shape = packed_activations.shape
    activations = (
        packed_activations.to(dtype=torch.uint8).contiguous().reshape(-1, in_features // 2)
    )
    weights = packed_weight_indices.to(device=activations.device, dtype=torch.uint8).contiguous()
    norms = token_norms.to(device=activations.device, dtype=torch.float32).contiguous()
    weight_norms = row_norms.to(device=activations.device, dtype=torch.bfloat16).contiguous()
    activation_code_values = activation_codes.to(
        device=activations.device, dtype=torch.int8
    ).contiguous()
    weight_code_values = weight_codes.to(device=activations.device, dtype=torch.int8).contiguous()
    out = torch.empty(
        (activations.shape[0], out_features),
        device=activations.device,
        dtype=output_dtype,
    )
    if bias is None:
        bias_values = out
        has_bias = False
    else:
        bias_values = bias.to(device=activations.device, dtype=output_dtype).contiguous()
        has_bias = True

    ops.matmul_packed_w4a4_int8(
        out,
        activations,
        weights,
        norms,
        weight_norms,
        activation_code_values,
        weight_code_values,
        bias_values,
        has_bias,
        activation_scale,
        weight_scale,
        out_features,
        in_features,
        tile_m,
        tile_n,
        async_packed,
        weight_k_major,
    )
    return out.reshape(*original_shape[:-1], out_features)


def quantize_activations_packed_w4(
    x: torch.Tensor,
    permutation: torch.Tensor,
    signs: torch.Tensor,
    boundaries: torch.Tensor,
    *,
    eps: float = 1e-12,
    inv_sqrt_block: float,
    threads: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not x.is_cuda:
        raise RuntimeError("native packed W4 activation quantization requires CUDA tensors")
    if x.dtype not in {torch.float16, torch.bfloat16}:
        raise ValueError("x must be float16 or bfloat16")
    dim = x.shape[-1]
    if dim not in {512, 1024, 2048, 4096, 8192, 16384}:
        raise ValueError(
            "native packed W4 activation quantization supports dimensions "
            "512, 1024, 2048, 4096, 8192, and 16384"
        )
    if threads not in {128, 256, 512}:
        raise ValueError("threads must be 128, 256, or 512")
    if permutation.numel() != dim or signs.numel() != dim:
        raise ValueError("permutation and signs must match the input dimension")
    if boundaries.numel() != 15:
        raise ValueError("boundaries must contain 15 values")

    original_shape = x.shape
    values = x.contiguous().reshape(-1, dim)
    packed = torch.empty((values.shape[0], dim // 2), device=x.device, dtype=torch.uint8)
    norms = torch.empty(values.shape[0], device=x.device, dtype=torch.float32)
    permutation_dtype = torch.int32 if permutation.dtype == torch.int32 else torch.int64
    permutation_values = permutation.to(device=x.device, dtype=permutation_dtype).contiguous()
    sign_values = signs.to(device=x.device, dtype=torch.int8).contiguous()
    boundary_values = boundaries.to(device=x.device, dtype=torch.float32).contiguous()
    ops.quantize_activations_packed_w4(
        packed,
        norms,
        values,
        permutation_values,
        sign_values,
        boundary_values,
        eps,
        inv_sqrt_block,
        threads,
    )
    return (
        packed.reshape(*original_shape[:-1], dim // 2),
        norms.reshape(original_shape[:-1]),
    )


def quantize_activations_int8(
    x: torch.Tensor,
    permutation: torch.Tensor,
    signs: torch.Tensor,
    boundaries: torch.Tensor,
    codes: torch.Tensor,
    *,
    eps: float = 1e-12,
    inv_sqrt_block: float,
    threads: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not x.is_cuda:
        raise RuntimeError("native INT8 activation quantization requires CUDA tensors")
    if x.dtype not in {torch.float16, torch.bfloat16}:
        raise ValueError("x must be float16 or bfloat16")
    dim = x.shape[-1]
    if dim not in {512, 1024, 2048, 4096, 8192, 12288, 16384}:
        raise ValueError(
            "native INT8 activation quantization supports dimensions "
            "512, 1024, 2048, 4096, 8192, 12288, and 16384"
        )
    if threads not in {128, 256, 512}:
        raise ValueError("threads must be 128, 256, or 512")
    if permutation.numel() != dim or signs.numel() != dim:
        raise ValueError("permutation and signs must match the input dimension")
    if boundaries.numel() != 15:
        raise ValueError("boundaries must contain 15 values")
    if codes.numel() != 16:
        raise ValueError("codes must contain 16 values")

    original_shape = x.shape
    values = x.contiguous().reshape(-1, dim)
    quantized = torch.empty(values.shape, device=x.device, dtype=torch.int8)
    norms = torch.empty(values.shape[0], device=x.device, dtype=torch.float32)
    permutation_dtype = torch.int32 if permutation.dtype == torch.int32 else torch.int64
    permutation_values = permutation.to(device=x.device, dtype=permutation_dtype).contiguous()
    sign_values = signs.to(device=x.device, dtype=torch.int8).contiguous()
    boundary_values = boundaries.to(device=x.device, dtype=torch.float32).contiguous()
    code_values = codes.to(device=x.device, dtype=torch.int8).contiguous()
    ops.quantize_activations_int8(
        quantized,
        norms,
        values,
        permutation_values,
        sign_values,
        boundary_values,
        code_values,
        eps,
        inv_sqrt_block,
        threads,
    )
    return (
        quantized.reshape(*original_shape[:-1], dim),
        norms.reshape(original_shape[:-1]),
    )
