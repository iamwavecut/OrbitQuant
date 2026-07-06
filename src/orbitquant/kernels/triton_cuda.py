from __future__ import annotations

import torch

from orbitquant.codebooks import LloydMaxCodebook
from orbitquant.rotations import RPBHRotation


def _load_triton():
    try:
        import triton
        import triton.language as tl
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Triton is required for the triton_cuda backend") from exc
    return triton, tl


triton, tl = _load_triton()


@triton.jit
def _codebook_rescale_kernel(
    rotated_ptr,
    norms_ptr,
    centroids_ptr,
    boundaries_ptr,
    output_ptr,
    total: tl.constexpr,
    dim: tl.constexpr,
    block_size: tl.constexpr,
    levels: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < total
    values = tl.load(rotated_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    indices = tl.zeros((block_size,), dtype=tl.int32)
    for idx in tl.static_range(0, levels - 1):
        boundary = tl.load(boundaries_ptr + idx)
        indices += (values > boundary).to(tl.int32)
    centroids = tl.load(centroids_ptr + indices)
    rows = offsets // dim
    norms = tl.load(norms_ptr + rows, mask=mask, other=0.0).to(tl.float32)
    tl.store(output_ptr + offsets, centroids * norms, mask=mask)


@triton.jit
def _dequantize_packed_weight_kernel(
    packed_ptr,
    row_norms_ptr,
    centroids_ptr,
    output_ptr,
    total: tl.constexpr,
    in_features: tl.constexpr,
    bits: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < total
    bit_starts = offsets * bits
    byte_indices = bit_starts // 8
    bit_offsets = bit_starts % 8
    low = tl.load(packed_ptr + byte_indices, mask=mask, other=0).to(tl.uint32)
    needs_high = bit_offsets + bits > 8
    high = tl.load(packed_ptr + byte_indices + 1, mask=mask & needs_high, other=0).to(
        tl.uint32
    )
    raw = low | (high << 8)
    indices = (raw >> bit_offsets) & ((1 << bits) - 1)
    rows = offsets // in_features
    norms = tl.load(row_norms_ptr + rows, mask=mask, other=0.0).to(tl.float32)
    centroids = tl.load(centroids_ptr + indices)
    tl.store(output_ptr + offsets, centroids * norms, mask=mask)


@triton.jit
def _pack_lowbit_kernel(
    values_ptr,
    packed_ptr,
    packed_length: tl.constexpr,
    value_count: tl.constexpr,
    bits: tl.constexpr,
    block_size: tl.constexpr,
):
    byte_offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = byte_offsets < packed_length
    packed = tl.zeros((block_size,), dtype=tl.uint32)
    byte_bit_starts = byte_offsets * 8
    for bit_id in tl.static_range(0, 8):
        global_bit = byte_bit_starts + bit_id
        value_offsets = global_bit // bits
        value_bit_offsets = global_bit % bits
        values = tl.load(
            values_ptr + value_offsets,
            mask=mask & (value_offsets < value_count),
            other=0,
        )
        source_bits = (values.to(tl.uint32) >> value_bit_offsets) & 1
        packed |= source_bits << bit_id
    tl.store(packed_ptr + byte_offsets, packed.to(tl.uint8), mask=mask)


@triton.jit
def _permute_sign_weight_kernel(
    weight_ptr,
    permutation_ptr,
    signs_ptr,
    work_ptr,
    total: tl.constexpr,
    in_features: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < total
    rows = offsets // in_features
    cols = offsets % in_features
    source_cols = tl.load(permutation_ptr + cols, mask=mask, other=0).to(tl.int64)
    signs = tl.load(signs_ptr + cols, mask=mask, other=1).to(tl.float32)
    values = tl.load(
        weight_ptr + rows * in_features + source_cols,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    tl.store(work_ptr + offsets, values * signs, mask=mask)


@triton.jit
def _fwht_stage_weight_kernel(
    work_ptr,
    total_pairs: tl.constexpr,
    in_features: tl.constexpr,
    num_blocks: tl.constexpr,
    orbit_block_size: tl.constexpr,
    stage_width: tl.constexpr,
    block_size: tl.constexpr,
):
    pair_offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = pair_offsets < total_pairs
    pairs_per_orbit_block: tl.constexpr = orbit_block_size // 2
    row_block = pair_offsets // pairs_per_orbit_block
    pair_in_orbit_block = pair_offsets % pairs_per_orbit_block
    rows = row_block // num_blocks
    orbit_blocks = row_block % num_blocks
    groups = pair_in_orbit_block // stage_width
    inner = pair_in_orbit_block % stage_width
    left_cols = orbit_blocks * orbit_block_size + groups * (stage_width * 2) + inner
    right_cols = left_cols + stage_width
    left_offsets = rows * in_features + left_cols
    right_offsets = rows * in_features + right_cols
    left = tl.load(work_ptr + left_offsets, mask=mask, other=0.0).to(tl.float32)
    right = tl.load(work_ptr + right_offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(work_ptr + left_offsets, left + right, mask=mask)
    tl.store(work_ptr + right_offsets, left - right, mask=mask)


@triton.jit
def _quantize_rotated_weight_indices_kernel(
    work_ptr,
    row_norms_ptr,
    boundaries_ptr,
    indices_ptr,
    total: tl.constexpr,
    in_features: tl.constexpr,
    levels: tl.constexpr,
    inv_sqrt_block: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < total
    rows = offsets // in_features
    norms = tl.load(row_norms_ptr + rows, mask=mask, other=1.0).to(tl.float32)
    values = tl.load(work_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    values = values * inv_sqrt_block / norms
    indices = tl.zeros((block_size,), dtype=tl.int32)
    for idx in tl.static_range(0, levels - 1):
        boundary = tl.load(boundaries_ptr + idx)
        indices += (values > boundary).to(tl.int32)
    tl.store(indices_ptr + offsets, indices.to(tl.uint8), mask=mask)


def quantize_rotated_activations_with_triton(
    rotated: torch.Tensor,
    norms: torch.Tensor,
    codebook: LloydMaxCodebook,
) -> torch.Tensor:
    if not rotated.is_cuda:
        raise RuntimeError("triton_cuda backend requires CUDA tensors")
    triton, _ = _load_triton()
    rotated_contiguous = rotated.contiguous()
    flat = rotated_contiguous.reshape(-1)
    row_norms = norms.contiguous().reshape(-1).to(device=rotated.device, dtype=torch.float32)
    centroids = codebook.centroids.to(device=rotated.device, dtype=torch.float32).contiguous()
    boundaries = codebook.boundaries.to(device=rotated.device, dtype=torch.float32).contiguous()
    output = torch.empty_like(flat, dtype=torch.float32)
    block_size = 256
    grid = (triton.cdiv(flat.numel(), block_size),)
    _codebook_rescale_kernel[grid](
        flat,
        row_norms,
        centroids,
        boundaries,
        output,
        total=flat.numel(),
        dim=rotated.shape[-1],
        block_size=block_size,
        levels=centroids.numel(),
    )
    return output.reshape_as(rotated_contiguous)


def dequantize_packed_weight_with_triton(
    packed_weight_indices: torch.Tensor,
    row_norms: torch.Tensor,
    codebook: LloydMaxCodebook,
    *,
    bits: int,
    out_features: int,
    in_features: int,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("triton_cuda backend requires CUDA tensors")
    triton, _ = _load_triton()
    target_device = torch.device(device)
    if target_device.type != "cuda":
        raise RuntimeError("triton_cuda weight dequant requires a CUDA device")
    total = out_features * in_features
    packed = packed_weight_indices.to(device=target_device, dtype=torch.uint8).contiguous()
    norms = row_norms.to(device=target_device, dtype=torch.float32).contiguous()
    centroids = codebook.centroids.to(device=target_device, dtype=torch.float32).contiguous()
    output = torch.empty(total, device=target_device, dtype=torch.float32)
    if total == 0:
        return output.reshape(out_features, in_features)

    block_size = 256
    grid = (triton.cdiv(total, block_size),)
    _dequantize_packed_weight_kernel[grid](
        packed,
        norms,
        centroids,
        output,
        total=total,
        in_features=in_features,
        bits=bits,
        block_size=block_size,
    )
    return output.reshape(out_features, in_features)


def pack_lowbit_with_triton(values: torch.Tensor, *, bits: int) -> torch.Tensor:
    if bits <= 0 or bits > 8:
        raise ValueError("bits must be in [1, 8]")
    if not values.is_cuda:
        raise RuntimeError("triton_cuda pack requires CUDA tensors")
    max_value = (1 << bits) - 1
    flat_raw = values.detach().flatten()
    if flat_raw.numel() and (int(flat_raw.min()) < 0 or int(flat_raw.max()) > max_value):
        raise ValueError(f"all values must fit in {bits} bits")
    flat = flat_raw.to(dtype=torch.uint8).contiguous()
    packed_length = (flat.numel() * bits + 7) // 8
    packed = torch.empty(packed_length, device=flat.device, dtype=torch.uint8)
    if packed_length == 0:
        return packed

    triton, _ = _load_triton()
    block_size = 256
    grid = (triton.cdiv(packed_length, block_size),)
    _pack_lowbit_kernel[grid](
        flat,
        packed,
        packed_length=packed_length,
        value_count=flat.numel(),
        bits=bits,
        block_size=block_size,
    )
    return packed


def quantize_weight_indices_with_triton(
    weight: torch.Tensor,
    row_norms: torch.Tensor,
    *,
    rotation: RPBHRotation,
    codebook: LloydMaxCodebook,
) -> torch.Tensor:
    if not weight.is_cuda:
        raise RuntimeError("triton_cuda weight quantization requires CUDA tensors")
    if weight.ndim != 2:
        raise ValueError("weight must be a rank-2 tensor")
    if weight.shape[-1] != rotation.dim:
        raise ValueError(f"expected in_features {rotation.dim}, got {weight.shape[-1]}")

    triton, _ = _load_triton()
    out_features, in_features = weight.shape
    total = out_features * in_features
    weight_fp32 = weight.to(dtype=torch.float32).contiguous()
    norms = row_norms.to(device=weight.device, dtype=torch.float32).contiguous()
    permutation = rotation.permutation.to(device=weight.device, dtype=torch.int64).contiguous()
    signs = rotation.signs.to(device=weight.device, dtype=torch.int8).contiguous()
    boundaries = codebook.boundaries.to(device=weight.device, dtype=torch.float32).contiguous()
    work = torch.empty(total, device=weight.device, dtype=torch.float32)
    indices = torch.empty(total, device=weight.device, dtype=torch.uint8)
    if total == 0:
        return indices.reshape(out_features, in_features)

    element_block_size = 256
    _permute_sign_weight_kernel[(triton.cdiv(total, element_block_size),)](
        weight_fp32,
        permutation,
        signs,
        work,
        total=total,
        in_features=in_features,
        block_size=element_block_size,
    )

    orbit_block_size = int(rotation.block_size)
    num_blocks = int(rotation.num_blocks)
    total_pairs = out_features * num_blocks * (orbit_block_size // 2)
    stage_width = 1
    while stage_width < orbit_block_size:
        _fwht_stage_weight_kernel[(triton.cdiv(total_pairs, element_block_size),)](
            work,
            total_pairs=total_pairs,
            in_features=in_features,
            num_blocks=num_blocks,
            orbit_block_size=orbit_block_size,
            stage_width=stage_width,
            block_size=element_block_size,
        )
        stage_width *= 2

    _quantize_rotated_weight_indices_kernel[(triton.cdiv(total, element_block_size),)](
        work,
        norms,
        boundaries,
        indices,
        total=total,
        in_features=in_features,
        levels=codebook.centroids.numel(),
        inv_sqrt_block=float(rotation.normalization),
        block_size=element_block_size,
    )
    return indices.reshape(out_features, in_features)
