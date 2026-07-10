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

_WEIGHT_QUANT_CONSTANT_CACHE: dict[tuple[object, ...], torch.Tensor] = {}
_INT8_CENTROID_SURROGATE_CACHE: dict[tuple[float, ...], tuple[torch.Tensor, float]] = {}


def _normalize_cuda_device(device: torch.device | str) -> torch.device:
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and torch_device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return torch_device


def clear_triton_constant_cache() -> None:
    _WEIGHT_QUANT_CONSTANT_CACHE.clear()


def _cached_cuda_tensor(
    key: tuple[object, ...],
    source: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    device = _normalize_cuda_device(device)
    cached = _WEIGHT_QUANT_CONSTANT_CACHE.get(key)
    if cached is not None and cached.device == device and cached.dtype == dtype:
        return cached
    tensor = source.to(device=device, dtype=dtype).contiguous()
    _WEIGHT_QUANT_CONSTANT_CACHE[key] = tensor
    return tensor


def _weight_quantization_constants(
    *,
    rotation: RPBHRotation,
    codebook: LloydMaxCodebook,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = _normalize_cuda_device(device)
    device_key = str(device)
    rotation_key = (int(rotation.dim), int(rotation.seed), int(rotation.block_size), device_key)
    codebook_key = (
        int(codebook.dim),
        int(codebook.bits),
        int(codebook.algorithm_version),
        device_key,
    )
    permutation = _cached_cuda_tensor(
        ("rotation_permutation", *rotation_key),
        rotation.permutation,
        device=device,
        dtype=torch.int64,
    )
    signs = _cached_cuda_tensor(
        ("rotation_signs", *rotation_key),
        rotation.signs,
        device=device,
        dtype=torch.int8,
    )
    boundaries = _cached_cuda_tensor(
        ("codebook_boundaries", *codebook_key),
        codebook.boundaries,
        device=device,
        dtype=torch.float32,
    )
    return permutation, signs, boundaries


@triton.jit
def _row_norm_kernel(
    input_ptr,
    norms_ptr,
    rows: tl.constexpr,
    dim: tl.constexpr,
    eps: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, block_size)
    mask = (row < rows) & (offsets < dim)
    values = tl.load(input_ptr + row * dim + offsets, mask=mask, other=0.0).to(tl.float32)
    squared_sum = tl.sum(values * values, axis=0)
    norm = tl.sqrt(squared_sum)
    tl.store(norms_ptr + row, norm, mask=row < rows)


@triton.jit
def _permute_sign_normalize_activation_kernel(
    input_ptr,
    norms_ptr,
    permutation_ptr,
    signs_ptr,
    work_ptr,
    total: tl.constexpr,
    dim: tl.constexpr,
    eps: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < total
    rows = offsets // dim
    cols = offsets % dim
    source_cols = tl.load(permutation_ptr + cols, mask=mask, other=0).to(tl.int64)
    signs = tl.load(signs_ptr + cols, mask=mask, other=1).to(tl.float32)
    norms = tl.load(norms_ptr + rows, mask=mask, other=1.0).to(tl.float32)
    denom = norms + eps
    values = tl.load(input_ptr + rows * dim + source_cols, mask=mask, other=0.0).to(
        tl.float32
    )
    tl.store(work_ptr + offsets, values * signs / denom, mask=mask)


@triton.jit
def _quantize_activation_work_rescale_kernel(
    work_ptr,
    norms_ptr,
    centroids_ptr,
    boundaries_ptr,
    output_ptr,
    total: tl.constexpr,
    dim: tl.constexpr,
    levels: tl.constexpr,
    inv_sqrt_block: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < total
    values = tl.load(work_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    values = values * inv_sqrt_block
    indices = tl.zeros((block_size,), dtype=tl.int32)
    for idx in tl.static_range(0, levels - 1):
        boundary = tl.load(boundaries_ptr + idx)
        indices += (values > boundary).to(tl.int32)
    centroids = tl.load(centroids_ptr + indices)
    rows = offsets // dim
    norms = tl.load(norms_ptr + rows, mask=mask, other=0.0).to(tl.float32)
    tl.store(output_ptr + offsets, centroids * norms, mask=mask)


@triton.jit
def _fwht_local_stage(values, block_size: tl.constexpr, stage_width: tl.constexpr):
    butterfly = tl.reshape(
        values,
        (block_size // (stage_width * 2), 2, stage_width),
    )
    butterfly = tl.permute(butterfly, (0, 2, 1))
    left, right = tl.split(butterfly)
    butterfly = tl.join(left + right, left - right)
    butterfly = tl.permute(butterfly, (0, 2, 1))
    return tl.reshape(butterfly, (block_size,))


@triton.jit
def _fused_rpbh_quantize_activation_kernel(
    input_ptr,
    norms_ptr,
    permutation_ptr,
    signs_ptr,
    centroids_ptr,
    boundaries_ptr,
    output_ptr,
    rows: tl.constexpr,
    dim: tl.constexpr,
    num_blocks: tl.constexpr,
    orbit_block_size: tl.constexpr,
    fwht_stages: tl.constexpr,
    levels: tl.constexpr,
    eps: tl.constexpr,
    inv_sqrt_block: tl.constexpr,
):
    row_block = tl.program_id(0)
    row = row_block // num_blocks
    orbit_block = row_block % num_blocks
    local_cols = tl.arange(0, orbit_block_size)
    cols = orbit_block * orbit_block_size + local_cols
    mask = (row < rows) & (cols < dim)
    source_cols = tl.load(permutation_ptr + cols, mask=mask, other=0).to(tl.int64)
    signs = tl.load(signs_ptr + cols, mask=mask, other=1).to(tl.float32)
    norm = tl.load(norms_ptr + row, mask=row < rows, other=0.0).to(tl.float32)
    denominator = norm + eps
    values = tl.load(
        input_ptr + row * dim + source_cols,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    values = values * signs / denominator

    if fwht_stages > 0:
        values = _fwht_local_stage(values, orbit_block_size, 1)
    if fwht_stages > 1:
        values = _fwht_local_stage(values, orbit_block_size, 2)
    if fwht_stages > 2:
        values = _fwht_local_stage(values, orbit_block_size, 4)
    if fwht_stages > 3:
        values = _fwht_local_stage(values, orbit_block_size, 8)
    if fwht_stages > 4:
        values = _fwht_local_stage(values, orbit_block_size, 16)
    if fwht_stages > 5:
        values = _fwht_local_stage(values, orbit_block_size, 32)
    if fwht_stages > 6:
        values = _fwht_local_stage(values, orbit_block_size, 64)
    if fwht_stages > 7:
        values = _fwht_local_stage(values, orbit_block_size, 128)
    if fwht_stages > 8:
        values = _fwht_local_stage(values, orbit_block_size, 256)
    if fwht_stages > 9:
        values = _fwht_local_stage(values, orbit_block_size, 512)
    if fwht_stages > 10:
        values = _fwht_local_stage(values, orbit_block_size, 1024)
    if fwht_stages > 11:
        values = _fwht_local_stage(values, orbit_block_size, 2048)

    values *= inv_sqrt_block
    indices = tl.zeros((orbit_block_size,), dtype=tl.int32)
    for index in tl.static_range(0, levels - 1):
        boundary = tl.load(boundaries_ptr + index)
        indices += (values > boundary).to(tl.int32)
    quantized = tl.load(centroids_ptr + indices)
    tl.store(output_ptr + row * dim + cols, quantized * norm, mask=mask)


@triton.jit
def _fused_rpbh_quantize_activation_pack_w4_kernel(
    input_ptr,
    norms_ptr,
    permutation_ptr,
    signs_ptr,
    boundaries_ptr,
    output_ptr,
    rows: tl.constexpr,
    dim: tl.constexpr,
    num_blocks: tl.constexpr,
    orbit_block_size: tl.constexpr,
    fwht_stages: tl.constexpr,
    levels: tl.constexpr,
    eps: tl.constexpr,
    inv_sqrt_block: tl.constexpr,
):
    row_block = tl.program_id(0)
    row = row_block // num_blocks
    orbit_block = row_block % num_blocks
    local_cols = tl.arange(0, orbit_block_size)
    cols = orbit_block * orbit_block_size + local_cols
    mask = (row < rows) & (cols < dim)
    source_cols = tl.load(permutation_ptr + cols, mask=mask, other=0).to(tl.int64)
    signs = tl.load(signs_ptr + cols, mask=mask, other=1).to(tl.float32)
    norm = tl.load(norms_ptr + row, mask=row < rows, other=0.0).to(tl.float32)
    values = tl.load(
        input_ptr + row * dim + source_cols,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    values = values * signs / (norm + eps)

    if fwht_stages > 0:
        values = _fwht_local_stage(values, orbit_block_size, 1)
    if fwht_stages > 1:
        values = _fwht_local_stage(values, orbit_block_size, 2)
    if fwht_stages > 2:
        values = _fwht_local_stage(values, orbit_block_size, 4)
    if fwht_stages > 3:
        values = _fwht_local_stage(values, orbit_block_size, 8)
    if fwht_stages > 4:
        values = _fwht_local_stage(values, orbit_block_size, 16)
    if fwht_stages > 5:
        values = _fwht_local_stage(values, orbit_block_size, 32)
    if fwht_stages > 6:
        values = _fwht_local_stage(values, orbit_block_size, 64)
    if fwht_stages > 7:
        values = _fwht_local_stage(values, orbit_block_size, 128)
    if fwht_stages > 8:
        values = _fwht_local_stage(values, orbit_block_size, 256)
    if fwht_stages > 9:
        values = _fwht_local_stage(values, orbit_block_size, 512)

    values *= inv_sqrt_block
    indices = tl.zeros((orbit_block_size,), dtype=tl.int32)
    for index in tl.static_range(0, levels - 1):
        boundary = tl.load(boundaries_ptr + index)
        indices += (values > boundary).to(tl.int32)
    pairs = tl.reshape(indices, (orbit_block_size // 2, 2))
    low, high = tl.split(pairs)
    packed = low | (high << 4)
    local_bytes = tl.arange(0, orbit_block_size // 2)
    output_offsets = (
        row * (dim // 2)
        + orbit_block * (orbit_block_size // 2)
        + local_bytes
    )
    tl.store(output_ptr + output_offsets, packed.to(tl.uint8), mask=row < rows)


@triton.jit
def _fused_rpbh_fwht_1024_chunk_kernel(
    input_ptr,
    norms_ptr,
    permutation_ptr,
    signs_ptr,
    work_ptr,
    rows: tl.constexpr,
    dim: tl.constexpr,
    num_blocks: tl.constexpr,
    orbit_block_size: tl.constexpr,
    chunks_per_orbit_block: tl.constexpr,
    eps: tl.constexpr,
):
    program = tl.program_id(0)
    chunk = program % chunks_per_orbit_block
    row_block = program // chunks_per_orbit_block
    row = row_block // num_blocks
    orbit_block = row_block % num_blocks
    local_cols = tl.arange(0, 1024)
    cols = orbit_block * orbit_block_size + chunk * 1024 + local_cols
    mask = (row < rows) & (cols < dim)
    source_cols = tl.load(permutation_ptr + cols, mask=mask, other=0).to(tl.int64)
    signs = tl.load(signs_ptr + cols, mask=mask, other=1).to(tl.float32)
    norm = tl.load(norms_ptr + row, mask=row < rows, other=0.0).to(tl.float32)
    values = tl.load(
        input_ptr + row * dim + source_cols,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    values = values * signs / (norm + eps)
    values = _fwht_local_stage(values, 1024, 1)
    values = _fwht_local_stage(values, 1024, 2)
    values = _fwht_local_stage(values, 1024, 4)
    values = _fwht_local_stage(values, 1024, 8)
    values = _fwht_local_stage(values, 1024, 16)
    values = _fwht_local_stage(values, 1024, 32)
    values = _fwht_local_stage(values, 1024, 64)
    values = _fwht_local_stage(values, 1024, 128)
    values = _fwht_local_stage(values, 1024, 256)
    values = _fwht_local_stage(values, 1024, 512)
    tl.store(work_ptr + row * dim + cols, values, mask=mask)


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
def _quantize_activation_work_pack_w4_kernel(
    work_ptr,
    boundaries_ptr,
    output_ptr,
    packed_total: tl.constexpr,
    levels: tl.constexpr,
    inv_sqrt_block: tl.constexpr,
    block_size: tl.constexpr,
):
    byte_offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = byte_offsets < packed_total
    value_offsets = byte_offsets * 2
    low_values = tl.load(work_ptr + value_offsets, mask=mask, other=0.0).to(tl.float32)
    high_values = tl.load(work_ptr + value_offsets + 1, mask=mask, other=0.0).to(
        tl.float32
    )
    low_values *= inv_sqrt_block
    high_values *= inv_sqrt_block
    low_indices = tl.zeros((block_size,), dtype=tl.int32)
    high_indices = tl.zeros((block_size,), dtype=tl.int32)
    for index in tl.static_range(0, levels - 1):
        boundary = tl.load(boundaries_ptr + index)
        low_indices += (low_values > boundary).to(tl.int32)
        high_indices += (high_values > boundary).to(tl.int32)
    packed = low_indices | (high_indices << 4)
    tl.store(output_ptr + byte_offsets, packed.to(tl.uint8), mask=mask)


@triton.jit
def _fwht_stage_kernel(
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
def _fwht_two_stage_kernel(
    work_ptr,
    total_quads: tl.constexpr,
    in_features: tl.constexpr,
    num_blocks: tl.constexpr,
    orbit_block_size: tl.constexpr,
    stage_width: tl.constexpr,
    block_size: tl.constexpr,
):
    quad_offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = quad_offsets < total_quads
    quads_per_orbit_block: tl.constexpr = orbit_block_size // 4
    row_block = quad_offsets // quads_per_orbit_block
    quad_in_orbit_block = quad_offsets % quads_per_orbit_block
    rows = row_block // num_blocks
    orbit_blocks = row_block % num_blocks
    groups = quad_in_orbit_block // stage_width
    inner = quad_in_orbit_block % stage_width
    base_cols = orbit_blocks * orbit_block_size + groups * (stage_width * 4) + inner
    offsets0 = rows * in_features + base_cols
    offsets1 = offsets0 + stage_width
    offsets2 = offsets0 + stage_width * 2
    offsets3 = offsets0 + stage_width * 3
    value0 = tl.load(work_ptr + offsets0, mask=mask, other=0.0).to(tl.float32)
    value1 = tl.load(work_ptr + offsets1, mask=mask, other=0.0).to(tl.float32)
    value2 = tl.load(work_ptr + offsets2, mask=mask, other=0.0).to(tl.float32)
    value3 = tl.load(work_ptr + offsets3, mask=mask, other=0.0).to(tl.float32)
    sum01 = value0 + value1
    diff01 = value0 - value1
    sum23 = value2 + value3
    diff23 = value2 - value3
    tl.store(work_ptr + offsets0, sum01 + sum23, mask=mask)
    tl.store(work_ptr + offsets1, diff01 + diff23, mask=mask)
    tl.store(work_ptr + offsets2, sum01 - sum23, mask=mask)
    tl.store(work_ptr + offsets3, diff01 - diff23, mask=mask)


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
def _matmul_packed_weight_kernel(
    input_ptr,
    packed_ptr,
    row_norms_ptr,
    centroids_ptr,
    bias_ptr,
    output_ptr,
    rows: tl.constexpr,
    out_features: tl.constexpr,
    in_features: tl.constexpr,
    bits: tl.constexpr,
    has_bias: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    offs_k = tl.arange(0, block_k)
    accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)

    for k_start in range(0, in_features, block_k):
        k = k_start + offs_k
        input_values = tl.load(
            input_ptr + offs_m[:, None] * in_features + k[None, :],
            mask=(offs_m[:, None] < rows) & (k[None, :] < in_features),
            other=0.0,
        )

        value_offsets = offs_n[None, :] * in_features + k[:, None]
        weight_mask = (offs_n[None, :] < out_features) & (k[:, None] < in_features)
        bit_starts = value_offsets * bits
        byte_indices = bit_starts // 8
        bit_offsets = bit_starts % 8
        low = tl.load(packed_ptr + byte_indices, mask=weight_mask, other=0).to(tl.uint32)
        needs_high = bit_offsets + bits > 8
        high = tl.load(
            packed_ptr + byte_indices + 1,
            mask=weight_mask & needs_high,
            other=0,
        ).to(tl.uint32)
        raw = low | (high << 8)
        indices = (raw >> bit_offsets) & ((1 << bits) - 1)
        centroids = tl.load(centroids_ptr + indices, mask=weight_mask, other=0.0).to(
            tl.float32
        )
        norms = tl.load(row_norms_ptr + offs_n, mask=offs_n < out_features, other=0.0).to(
            tl.float32
        )
        weights = (centroids * norms[None, :]).to(input_values.dtype)
        accumulator += tl.dot(input_values, weights)

    if has_bias:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < out_features, other=0.0).to(tl.float32)
        accumulator += bias[None, :]

    tl.store(
        output_ptr + offs_m[:, None] * out_features + offs_n[None, :],
        accumulator,
        mask=(offs_m[:, None] < rows) & (offs_n[None, :] < out_features),
    )


@triton.jit
def _matmul_packed_weight_w4_kernel(
    input_ptr,
    packed_ptr,
    row_norms_ptr,
    centroids_ptr,
    bias_ptr,
    output_ptr,
    rows: tl.constexpr,
    out_features: tl.constexpr,
    in_features: tl.constexpr,
    has_bias: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k_bytes: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    offs_k_bytes = tl.arange(0, block_k_bytes)
    accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)
    packed_row_stride: tl.constexpr = in_features // 2

    for k_byte_start in range(0, packed_row_stride, block_k_bytes):
        k_bytes = k_byte_start + offs_k_bytes
        k0 = k_bytes * 2
        k1 = k0 + 1
        k_mask = k_bytes < packed_row_stride
        input_values0 = tl.load(
            input_ptr + offs_m[:, None] * in_features + k0[None, :],
            mask=(offs_m[:, None] < rows) & k_mask[None, :],
            other=0.0,
        )
        input_values1 = tl.load(
            input_ptr + offs_m[:, None] * in_features + k1[None, :],
            mask=(offs_m[:, None] < rows) & k_mask[None, :],
            other=0.0,
        )

        packed_offsets = offs_n[None, :] * packed_row_stride + k_bytes[:, None]
        weight_mask = (offs_n[None, :] < out_features) & k_mask[:, None]
        packed_values = tl.load(packed_ptr + packed_offsets, mask=weight_mask, other=0).to(
            tl.uint32
        )
        indices0 = packed_values & 15
        indices1 = (packed_values >> 4) & 15
        centroids0 = tl.load(centroids_ptr + indices0, mask=weight_mask, other=0.0).to(
            tl.float32
        )
        centroids1 = tl.load(centroids_ptr + indices1, mask=weight_mask, other=0.0).to(
            tl.float32
        )
        norms = tl.load(row_norms_ptr + offs_n, mask=offs_n < out_features, other=0.0).to(
            tl.float32
        )
        weights0 = (centroids0 * norms[None, :]).to(input_values0.dtype)
        weights1 = (centroids1 * norms[None, :]).to(input_values1.dtype)
        accumulator += tl.dot(input_values0, weights0)
        accumulator += tl.dot(input_values1, weights1)

    if has_bias:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < out_features, other=0.0).to(tl.float32)
        accumulator += bias[None, :]

    tl.store(
        output_ptr + offs_m[:, None] * out_features + offs_n[None, :],
        accumulator,
        mask=(offs_m[:, None] < rows) & (offs_n[None, :] < out_features),
    )


@triton.jit
def _decode_packed_w4_codes_kernel(
    packed_ptr,
    codes_ptr,
    output_ptr,
    total: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < total
    packed = tl.load(packed_ptr + offsets // 2, mask=mask, other=0).to(tl.uint32)
    shifts = (offsets & 1) * 4
    indices = (packed >> shifts) & 15
    values = tl.load(codes_ptr + indices, mask=mask, other=0).to(tl.int8)
    tl.store(output_ptr + offsets, values, mask=mask)


@triton.jit
def _scale_int32_matmul_output_chunk_kernel(
    accumulator_ptr,
    token_norms_ptr,
    row_norms_ptr,
    bias_ptr,
    output_ptr,
    output_col_start,
    rows: tl.constexpr,
    chunk_out_features: tl.constexpr,
    out_features: tl.constexpr,
    combined_scale: tl.constexpr,
    has_bias: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    offs_m = tl.program_id(0) * block_m + tl.arange(0, block_m)
    offs_n = tl.program_id(1) * block_n + tl.arange(0, block_n)
    mask = (offs_m[:, None] < rows) & (offs_n[None, :] < chunk_out_features)
    accumulator_offsets = offs_m[:, None] * chunk_out_features + offs_n[None, :]
    output_columns = output_col_start + offs_n
    output_offsets = offs_m[:, None] * out_features + output_columns[None, :]
    accumulator = tl.load(
        accumulator_ptr + accumulator_offsets, mask=mask, other=0
    ).to(tl.float32)
    token_norms = tl.load(
        token_norms_ptr + offs_m, mask=offs_m < rows, other=0.0
    ).to(tl.float32)
    row_norms = tl.load(
        row_norms_ptr + output_columns,
        mask=output_columns < out_features,
        other=0.0,
    ).to(tl.float32)
    output = accumulator * token_norms[:, None] * row_norms[None, :] * combined_scale
    if has_bias:
        bias = tl.load(
            bias_ptr + output_columns,
            mask=output_columns < out_features,
            other=0.0,
        ).to(tl.float32)
        output += bias[None, :]
    tl.store(output_ptr + output_offsets, output, mask=mask)


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
def _unpack_lowbit_kernel(
    packed_ptr,
    values_ptr,
    length: tl.constexpr,
    bits: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < length
    bit_starts = offsets * bits
    byte_indices = bit_starts // 8
    bit_offsets = bit_starts % 8
    low = tl.load(packed_ptr + byte_indices, mask=mask, other=0).to(tl.uint32)
    needs_high = bit_offsets + bits > 8
    high = tl.load(packed_ptr + byte_indices + 1, mask=mask & needs_high, other=0).to(
        tl.uint32
    )
    raw = low | (high << 8)
    values = (raw >> bit_offsets) & ((1 << bits) - 1)
    tl.store(values_ptr + offsets, values.to(tl.uint8), mask=mask)


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
    eps: tl.constexpr,
    inv_sqrt_block: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < total
    rows = offsets // in_features
    norms = tl.load(row_norms_ptr + rows, mask=mask, other=1.0).to(tl.float32)
    values = tl.load(work_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    values = values * inv_sqrt_block / tl.maximum(norms, eps)
    indices = tl.zeros((block_size,), dtype=tl.int32)
    for idx in tl.static_range(0, levels - 1):
        boundary = tl.load(boundaries_ptr + idx)
        indices += (values > boundary).to(tl.int32)
    tl.store(indices_ptr + offsets, indices.to(tl.uint8), mask=mask)


@triton.jit
def _quantize_rotated_weight_pack_kernel(
    work_ptr,
    row_norms_ptr,
    boundaries_ptr,
    packed_ptr,
    packed_length: tl.constexpr,
    value_count: tl.constexpr,
    in_features: tl.constexpr,
    bits: tl.constexpr,
    levels: tl.constexpr,
    eps: tl.constexpr,
    inv_sqrt_block: tl.constexpr,
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
        valid = mask & (value_offsets < value_count)
        rows = value_offsets // in_features
        norms = tl.load(row_norms_ptr + rows, mask=valid, other=1.0).to(tl.float32)
        values = tl.load(work_ptr + value_offsets, mask=valid, other=0.0).to(tl.float32)
        values = values * inv_sqrt_block / tl.maximum(norms, eps)
        indices = tl.zeros((block_size,), dtype=tl.int32)
        for idx in tl.static_range(0, levels - 1):
            boundary = tl.load(boundaries_ptr + idx)
            indices += (values > boundary).to(tl.int32)
        source_bits = (indices.to(tl.uint32) >> value_bit_offsets) & 1
        packed |= source_bits << bit_id
    tl.store(packed_ptr + byte_offsets, packed.to(tl.uint8), mask=mask)


@triton.jit
def _adaln_group_scales_kernel(
    weight_ptr,
    scales_ptr,
    out_features: tl.constexpr,
    in_features: tl.constexpr,
    num_groups: tl.constexpr,
    group_size: tl.constexpr,
    block_size: tl.constexpr,
):
    group_offsets = tl.program_id(0)
    rows = group_offsets // num_groups
    groups = group_offsets % num_groups
    offsets = tl.arange(0, block_size)
    cols = groups * group_size + offsets
    mask = (rows < out_features) & (offsets < group_size) & (cols < in_features)
    values = tl.load(
        weight_ptr + rows * in_features + cols,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    max_abs = tl.max(tl.abs(values), axis=0)
    max_abs = tl.maximum(max_abs, 1.0e-12)
    tl.store(scales_ptr + rows * num_groups + groups, max_abs / 7.0)


@triton.jit
def _adaln_quantize_pack_int4_kernel(
    weight_ptr,
    scales_ptr,
    packed_ptr,
    packed_length: tl.constexpr,
    total_values: tl.constexpr,
    in_features: tl.constexpr,
    padded_in_features: tl.constexpr,
    num_groups: tl.constexpr,
    group_size: tl.constexpr,
    block_size: tl.constexpr,
):
    byte_offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = byte_offsets < packed_length
    value_offsets0 = byte_offsets * 2
    value_offsets1 = value_offsets0 + 1

    rows0 = value_offsets0 // padded_in_features
    cols0 = value_offsets0 % padded_in_features
    groups0 = cols0 // group_size
    valid0 = mask & (value_offsets0 < total_values)
    weight_valid0 = valid0 & (cols0 < in_features)
    values0 = tl.load(
        weight_ptr + rows0 * in_features + cols0,
        mask=weight_valid0,
        other=0.0,
    ).to(tl.float32)
    scales0 = tl.load(
        scales_ptr + rows0 * num_groups + groups0,
        mask=valid0,
        other=1.0,
    ).to(tl.float32)
    scaled0 = values0 / scales0
    rounded0 = tl.inline_asm_elementwise(
        "cvt.rni.s32.f32 $0, $1;",
        "=r,f",
        [scaled0],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )
    clamped0 = tl.minimum(tl.maximum(rounded0, -8), 7) + 8

    rows1 = value_offsets1 // padded_in_features
    cols1 = value_offsets1 % padded_in_features
    groups1 = cols1 // group_size
    valid1 = mask & (value_offsets1 < total_values)
    weight_valid1 = valid1 & (cols1 < in_features)
    values1 = tl.load(
        weight_ptr + rows1 * in_features + cols1,
        mask=weight_valid1,
        other=0.0,
    ).to(tl.float32)
    scales1 = tl.load(
        scales_ptr + rows1 * num_groups + groups1,
        mask=valid1,
        other=1.0,
    ).to(tl.float32)
    scaled1 = values1 / scales1
    rounded1 = tl.inline_asm_elementwise(
        "cvt.rni.s32.f32 $0, $1;",
        "=r,f",
        [scaled1],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )
    clamped1 = tl.minimum(tl.maximum(rounded1, -8), 7) + 8
    clamped1 = tl.where(valid1, clamped1, 0)

    packed = clamped0.to(tl.uint32) | (clamped1.to(tl.uint32) << 4)
    tl.store(packed_ptr + byte_offsets, packed.to(tl.uint8), mask=mask)


@triton.jit
def _dequantize_adaln_weight_kernel(
    packed_ptr,
    scales_ptr,
    output_ptr,
    total: tl.constexpr,
    in_features: tl.constexpr,
    padded_in_features: tl.constexpr,
    num_groups: tl.constexpr,
    group_size: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < total
    rows = offsets // in_features
    cols = offsets % in_features
    groups = cols // group_size
    value_offsets = rows * padded_in_features + cols
    byte_offsets = value_offsets // 2
    shifts = (value_offsets % 2) * 4
    packed = tl.load(packed_ptr + byte_offsets, mask=mask, other=0).to(tl.uint32)
    unsigned = (packed >> shifts) & 15
    signed = unsigned.to(tl.int32) - 8
    scales = tl.load(scales_ptr + rows * num_groups + groups, mask=mask, other=0.0).to(
        tl.float32
    )
    tl.store(output_ptr + offsets, signed.to(tl.float32) * scales, mask=mask)


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


def quantize_activations_with_triton(
    x: torch.Tensor,
    *,
    rotation: RPBHRotation,
    codebook: LloydMaxCodebook,
    eps: float,
    constant_tensors: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    if not x.is_cuda:
        raise RuntimeError("triton_cuda activation quantization requires CUDA tensors")
    if x.shape[-1] != rotation.dim:
        raise ValueError(f"expected last dimension {rotation.dim}, got {x.shape[-1]}")

    triton, _ = _load_triton()
    original_shape = x.shape
    dim = int(rotation.dim)
    input_contiguous = x.contiguous().reshape(-1, dim)
    rows = input_contiguous.shape[0]
    total = rows * dim
    output = torch.empty_like(input_contiguous)
    if total == 0:
        return output.reshape(original_shape)

    norms = torch.empty(rows, device=x.device, dtype=torch.float32)
    constants = {} if constant_tensors is None else constant_tensors
    permutation_source = constants.get("permutation", rotation.permutation)
    signs_source = constants.get("signs", rotation.signs)
    centroids_source = constants.get("centroids", codebook.centroids)
    boundaries_source = constants.get("boundaries", codebook.boundaries)
    permutation = permutation_source.to(device=x.device, dtype=torch.int64).contiguous()
    signs = signs_source.to(device=x.device, dtype=torch.int8).contiguous()
    centroids = centroids_source.to(device=x.device, dtype=torch.float32).contiguous()
    boundaries = boundaries_source.to(device=x.device, dtype=torch.float32).contiguous()

    norm_block_size = triton.next_power_of_2(dim)
    _row_norm_kernel[(rows,)](
        input_contiguous,
        norms,
        rows=rows,
        dim=dim,
        eps=float(eps),
        block_size=norm_block_size,
        num_warps=8,
    )

    orbit_block_size = int(rotation.block_size)
    num_blocks = int(rotation.num_blocks)
    if orbit_block_size <= 1024:
        fwht_stages = orbit_block_size.bit_length() - 1
        _fused_rpbh_quantize_activation_kernel[(rows * num_blocks,)](
            input_contiguous,
            norms,
            permutation,
            signs,
            centroids,
            boundaries,
            output,
            rows=rows,
            dim=dim,
            num_blocks=num_blocks,
            orbit_block_size=orbit_block_size,
            fwht_stages=fwht_stages,
            levels=centroids.numel(),
            eps=float(eps),
            inv_sqrt_block=float(rotation.normalization),
            num_warps=8 if orbit_block_size >= 512 else 4,
        )
        return output.reshape(original_shape)

    element_block_size = 256
    work = torch.empty(total, device=x.device, dtype=torch.float32)
    chunks_per_orbit_block = orbit_block_size // 1024
    _fused_rpbh_fwht_1024_chunk_kernel[
        (rows * num_blocks * chunks_per_orbit_block,)
    ](
        input_contiguous,
        norms,
        permutation,
        signs,
        work,
        rows=rows,
        dim=dim,
        num_blocks=num_blocks,
        orbit_block_size=orbit_block_size,
        chunks_per_orbit_block=chunks_per_orbit_block,
        eps=float(eps),
        num_warps=8,
    )

    stage_width = 1024
    while stage_width * 2 < orbit_block_size:
        total_quads = rows * num_blocks * (orbit_block_size // 4)
        _fwht_two_stage_kernel[(triton.cdiv(total_quads, element_block_size),)](
            work,
            total_quads=total_quads,
            in_features=dim,
            num_blocks=num_blocks,
            orbit_block_size=orbit_block_size,
            stage_width=stage_width,
            block_size=element_block_size,
        )
        stage_width *= 4
    if stage_width < orbit_block_size:
        total_pairs = rows * num_blocks * (orbit_block_size // 2)
        _fwht_stage_kernel[(triton.cdiv(total_pairs, element_block_size),)](
            work,
            total_pairs=total_pairs,
            in_features=dim,
            num_blocks=num_blocks,
            orbit_block_size=orbit_block_size,
            stage_width=stage_width,
            block_size=element_block_size,
        )

    _quantize_activation_work_rescale_kernel[
        (triton.cdiv(total, element_block_size),)
    ](
        work,
        norms,
        centroids,
        boundaries,
        output,
        total=total,
        dim=dim,
        levels=centroids.numel(),
        inv_sqrt_block=float(rotation.normalization),
        block_size=element_block_size,
    )
    return output.reshape(original_shape)


def quantize_activations_packed_w4_with_triton(
    x: torch.Tensor,
    *,
    rotation: RPBHRotation,
    codebook: LloydMaxCodebook,
    eps: float,
    constant_tensors: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize RPBH activations directly to packed W4 indices and token norms."""
    if not x.is_cuda:
        raise RuntimeError("triton_cuda packed activation quantization requires CUDA tensors")
    if codebook.bits != 4:
        raise ValueError("packed W4 activation quantization requires a 4-bit codebook")
    if x.shape[-1] != rotation.dim:
        raise ValueError(f"expected last dimension {rotation.dim}, got {x.shape[-1]}")
    if rotation.dim % 2 != 0 or rotation.block_size % 2 != 0:
        raise ValueError("packed W4 activation quantization requires even dimensions")

    triton, _ = _load_triton()
    original_shape = x.shape
    dim = int(rotation.dim)
    input_contiguous = x.contiguous().reshape(-1, dim)
    rows = input_contiguous.shape[0]
    total = rows * dim
    packed = torch.empty(rows * (dim // 2), device=x.device, dtype=torch.uint8)
    norms = torch.empty(rows, device=x.device, dtype=torch.float32)
    if total == 0:
        return packed.reshape(*original_shape[:-1], dim // 2), norms.reshape(
            original_shape[:-1]
        )

    constants = {} if constant_tensors is None else constant_tensors
    permutation = constants.get("permutation", rotation.permutation).to(
        device=x.device, dtype=torch.int64
    ).contiguous()
    signs = constants.get("signs", rotation.signs).to(
        device=x.device, dtype=torch.int8
    ).contiguous()
    boundaries = constants.get("boundaries", codebook.boundaries).to(
        device=x.device, dtype=torch.float32
    ).contiguous()

    norm_block_size = triton.next_power_of_2(dim)
    _row_norm_kernel[(rows,)](
        input_contiguous,
        norms,
        rows=rows,
        dim=dim,
        eps=float(eps),
        block_size=norm_block_size,
        num_warps=8,
    )

    orbit_block_size = int(rotation.block_size)
    num_blocks = int(rotation.num_blocks)
    if orbit_block_size <= 1024:
        fwht_stages = orbit_block_size.bit_length() - 1
        _fused_rpbh_quantize_activation_pack_w4_kernel[(rows * num_blocks,)](
            input_contiguous,
            norms,
            permutation,
            signs,
            boundaries,
            packed,
            rows=rows,
            dim=dim,
            num_blocks=num_blocks,
            orbit_block_size=orbit_block_size,
            fwht_stages=fwht_stages,
            levels=codebook.centroids.numel(),
            eps=float(eps),
            inv_sqrt_block=float(rotation.normalization),
            num_warps=8 if orbit_block_size >= 512 else 4,
        )
        return packed.reshape(*original_shape[:-1], dim // 2), norms.reshape(
            original_shape[:-1]
        )

    element_block_size = 256
    work = torch.empty(total, device=x.device, dtype=torch.float32)
    chunks_per_orbit_block = orbit_block_size // 1024
    _fused_rpbh_fwht_1024_chunk_kernel[
        (rows * num_blocks * chunks_per_orbit_block,)
    ](
        input_contiguous,
        norms,
        permutation,
        signs,
        work,
        rows=rows,
        dim=dim,
        num_blocks=num_blocks,
        orbit_block_size=orbit_block_size,
        chunks_per_orbit_block=chunks_per_orbit_block,
        eps=float(eps),
        num_warps=8,
    )
    stage_width = 1024
    while stage_width * 2 < orbit_block_size:
        total_quads = rows * num_blocks * (orbit_block_size // 4)
        _fwht_two_stage_kernel[(triton.cdiv(total_quads, element_block_size),)](
            work,
            total_quads=total_quads,
            in_features=dim,
            num_blocks=num_blocks,
            orbit_block_size=orbit_block_size,
            stage_width=stage_width,
            block_size=element_block_size,
        )
        stage_width *= 4
    if stage_width < orbit_block_size:
        total_pairs = rows * num_blocks * (orbit_block_size // 2)
        _fwht_stage_kernel[(triton.cdiv(total_pairs, element_block_size),)](
            work,
            total_pairs=total_pairs,
            in_features=dim,
            num_blocks=num_blocks,
            orbit_block_size=orbit_block_size,
            stage_width=stage_width,
            block_size=element_block_size,
        )
    _quantize_activation_work_pack_w4_kernel[
        (triton.cdiv(packed.numel(), element_block_size),)
    ](
        work,
        boundaries,
        packed,
        packed_total=packed.numel(),
        levels=codebook.centroids.numel(),
        inv_sqrt_block=float(rotation.normalization),
        block_size=element_block_size,
    )
    return packed.reshape(*original_shape[:-1], dim // 2), norms.reshape(
        original_shape[:-1]
    )


def quantize_adaln_weight_with_triton(
    weight: torch.Tensor,
    *,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not weight.is_cuda:
        raise RuntimeError("triton_cuda AdaLN RTN quantization requires CUDA tensors")
    if weight.ndim != 2:
        raise ValueError("weight must be a rank-2 tensor")
    if group_size <= 0:
        raise ValueError("group_size must be positive")

    triton, _ = _load_triton()
    weight_fp32 = weight.to(dtype=torch.float32).contiguous()
    out_features, in_features = weight_fp32.shape
    num_groups = triton.cdiv(in_features, group_size)
    padded_in_features = num_groups * group_size
    total_values = out_features * padded_in_features
    packed_length = (total_values + 1) // 2
    scales = torch.empty(out_features, num_groups, device=weight.device, dtype=torch.float32)
    packed = torch.empty(packed_length, device=weight.device, dtype=torch.uint8)
    if out_features == 0 or total_values == 0:
        return packed, scales

    scale_block_size = triton.next_power_of_2(group_size)
    _adaln_group_scales_kernel[(out_features * num_groups,)](
        weight_fp32,
        scales,
        out_features=out_features,
        in_features=in_features,
        num_groups=num_groups,
        group_size=group_size,
        block_size=scale_block_size,
    )

    pack_block_size = 256
    _adaln_quantize_pack_int4_kernel[(triton.cdiv(packed_length, pack_block_size),)](
        weight_fp32,
        scales,
        packed,
        packed_length=packed_length,
        total_values=total_values,
        in_features=in_features,
        padded_in_features=padded_in_features,
        num_groups=num_groups,
        group_size=group_size,
        block_size=pack_block_size,
    )
    return packed, scales


def dequantize_adaln_weight_with_triton(
    packed_weight: torch.Tensor,
    scales: torch.Tensor,
    *,
    out_features: int,
    in_features: int,
    group_size: int,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("triton_cuda backend requires CUDA tensors")
    target_device = torch.device(device)
    if target_device.type != "cuda":
        raise RuntimeError("triton_cuda AdaLN dequant requires a CUDA device")
    if group_size <= 0:
        raise ValueError("group_size must be positive")

    triton, _ = _load_triton()
    num_groups = triton.cdiv(in_features, group_size)
    padded_in_features = num_groups * group_size
    total = out_features * in_features
    packed = packed_weight.to(device=target_device, dtype=torch.uint8).contiguous()
    scales_fp32 = scales.to(device=target_device, dtype=torch.float32).contiguous()
    output = torch.empty(total, device=target_device, dtype=torch.float32)
    if total == 0:
        return output.reshape(out_features, in_features)

    block_size = 256
    _dequantize_adaln_weight_kernel[(triton.cdiv(total, block_size),)](
        packed,
        scales_fp32,
        output,
        total=total,
        in_features=in_features,
        padded_in_features=padded_in_features,
        num_groups=num_groups,
        group_size=group_size,
        block_size=block_size,
    )
    return output.reshape(out_features, in_features)


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


def matmul_packed_weight_with_triton(
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
    num_warps: int = 4,
) -> torch.Tensor:
    if bits <= 0 or bits > 8:
        raise ValueError("bits must be in [1, 8]")
    if block_m <= 0 or block_n <= 0 or block_k <= 0 or num_warps <= 0:
        raise ValueError("packed matmul tile sizes and num_warps must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("triton_cuda backend requires CUDA tensors")
    if not x.is_cuda:
        raise RuntimeError("triton_cuda packed matmul requires CUDA input tensors")
    if x.shape[-1] != in_features:
        raise ValueError(f"expected input last dimension {in_features}, got {x.shape[-1]}")

    triton, _ = _load_triton()
    original_shape = x.shape
    input_contiguous = x.contiguous().reshape(-1, in_features)
    rows = input_contiguous.shape[0]
    output = torch.empty((rows, out_features), device=x.device, dtype=x.dtype)
    if rows == 0 or out_features == 0:
        return output.reshape(*original_shape[:-1], out_features)

    packed = packed_weight_indices.to(device=x.device, dtype=torch.uint8).contiguous()
    norms = row_norms.to(device=x.device).contiguous()
    centroid_values = codebook if isinstance(codebook, torch.Tensor) else codebook.centroids
    centroids = centroid_values.to(device=x.device, dtype=torch.float32).contiguous()
    if bias is None:
        bias_tensor = output
        has_bias = False
    else:
        bias_tensor = bias.to(device=x.device).contiguous()
        has_bias = True

    effective_block_m = min(block_m, 16 if rows < 16 else triton.next_power_of_2(rows))
    grid = (triton.cdiv(rows, effective_block_m), triton.cdiv(out_features, block_n))
    if bits == 4 and in_features % 2 == 0 and block_k >= 2:
        block_k_bytes = max(1, block_k // 2)
        _matmul_packed_weight_w4_kernel[grid](
            input_contiguous,
            packed,
            norms,
            centroids,
            bias_tensor,
            output,
            rows=rows,
            out_features=out_features,
            in_features=in_features,
            has_bias=has_bias,
            block_m=effective_block_m,
            block_n=block_n,
            block_k_bytes=block_k_bytes,
            num_warps=num_warps,
        )
    else:
        _matmul_packed_weight_kernel[grid](
            input_contiguous,
            packed,
            norms,
            centroids,
            bias_tensor,
            output,
            rows=rows,
            out_features=out_features,
            in_features=in_features,
            bits=bits,
            has_bias=has_bias,
            block_m=effective_block_m,
            block_n=block_n,
            block_k=block_k,
            num_warps=num_warps,
        )
    return output.reshape(*original_shape[:-1], out_features)


def fit_int8_centroid_surrogate(centroids: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Fit a symmetric INT8 scale without changing the 4-bit centroid indices."""
    values = centroids.detach().to(device="cpu", dtype=torch.float64).flatten()
    if values.numel() != 16:
        raise ValueError("the W4A4 INT8 surrogate requires exactly 16 centroids")
    cache_key = tuple(float(value) for value in values)
    cached = _INT8_CENTROID_SURROGATE_CACHE.get(cache_key)
    if cached is not None:
        return cached[0].clone(), cached[1]
    max_abs = float(values.abs().max())
    if max_abs == 0.0:
        result = (torch.zeros(16, dtype=torch.int8), 1.0)
        _INT8_CENTROID_SURROGATE_CACHE[cache_key] = result
        return result[0].clone(), result[1]
    base_scale = max_abs / 127.0
    candidate_scales = base_scale * torch.linspace(0.5, 2.0, 3001, dtype=torch.float64)
    candidate_codes = torch.round(values[None, :] / candidate_scales[:, None]).clamp(
        -127, 127
    )
    errors = (
        candidate_codes * candidate_scales[:, None] - values[None, :]
    ).square().mean(dim=1)
    best = int(errors.argmin())
    result = (candidate_codes[best].to(torch.int8), float(candidate_scales[best]))
    _INT8_CENTROID_SURROGATE_CACHE[cache_key] = result
    return result[0].clone(), result[1]


def matmul_packed_w4a4_with_int_mm(
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
    epilogue_block_m: int = 32,
    epilogue_block_n: int = 256,
    chunk_out_features: int | None = None,
    activations_are_int8: bool = False,
) -> torch.Tensor:
    """Decode active W4/A4 tiles to INT8 and use PyTorch's CUTLASS GEMM."""
    if not packed_activations.is_cuda:
        raise RuntimeError("packed W4A4 int_mm requires CUDA tensors")
    if output_dtype not in {torch.float16, torch.bfloat16}:
        raise ValueError("output_dtype must be float16 or bfloat16")
    if in_features <= 0 or in_features % 16 != 0:
        raise ValueError("in_features must be positive and divisible by 16")
    if out_features <= 0 or out_features % 16 != 0:
        raise ValueError("out_features must be positive and divisible by 16")
    if min(epilogue_block_m, epilogue_block_n) <= 0:
        raise ValueError("epilogue tile sizes must be positive")
    if chunk_out_features is not None and (
        chunk_out_features <= 0 or chunk_out_features % 16 != 0
    ):
        raise ValueError("chunk_out_features must be positive and divisible by 16")

    triton, _ = _load_triton()
    original_shape = packed_activations.shape
    packed_k = in_features // 2
    if activations_are_int8:
        if packed_activations.dtype != torch.int8:
            raise ValueError("INT8 activations must have dtype int8")
        if original_shape[-1] != in_features:
            raise ValueError(
                f"expected INT8 activation last dimension {in_features}, "
                f"got {original_shape[-1]}"
            )
        activation_values = packed_activations.contiguous().reshape(-1, in_features)
        activations = None
        rows = activation_values.shape[0]
    else:
        if original_shape[-1] != packed_k:
            raise ValueError(
                f"expected packed activation last dimension {packed_k}, "
                f"got {original_shape[-1]}"
            )
        activations = (
            packed_activations.to(dtype=torch.uint8).contiguous().reshape(-1, packed_k)
        )
        rows = activations.shape[0]
        activation_values = torch.empty(
            (rows, in_features), device=activations.device, dtype=torch.int8
        )
    device = packed_activations.device
    expected_weight_bytes = out_features * packed_k
    if packed_weight_indices.numel() != expected_weight_bytes:
        raise ValueError(
            f"expected {expected_weight_bytes} packed weight bytes, "
            f"got {packed_weight_indices.numel()}"
        )
    if rows == 0:
        return torch.empty(
            (*original_shape[:-1], out_features),
            device=device,
            dtype=output_dtype,
        )

    weights = packed_weight_indices.to(
        device=device, dtype=torch.uint8
    ).contiguous().flatten()
    activation_code_values = activation_codes.to(
        device=device, dtype=torch.int8
    ).contiguous()
    weight_code_values = weight_codes.to(
        device=device, dtype=torch.int8
    ).contiguous()
    if activation_code_values.numel() != 16 or weight_code_values.numel() != 16:
        raise ValueError("packed W4A4 int_mm requires 16 activation and weight codes")

    token_norm_values = token_norms.to(
        device=device, dtype=torch.float32
    ).contiguous().reshape(-1)
    row_norm_values = row_norms.to(
        device=device, dtype=torch.float32
    ).contiguous().reshape(-1)
    if token_norm_values.numel() != rows:
        raise ValueError("token_norms must contain one value per activation row")
    if row_norm_values.numel() != out_features:
        raise ValueError("row_norms must contain one value per output feature")

    decode_block_size = 256
    if not activations_are_int8:
        activation_total = rows * in_features
        _decode_packed_w4_codes_kernel[
            (triton.cdiv(activation_total, decode_block_size),)
        ](
            activations,
            activation_code_values,
            activation_values,
            total=activation_total,
            block_size=decode_block_size,
        )

    output = torch.empty(
        (rows, out_features), device=device, dtype=output_dtype
    )
    if bias is None:
        bias_values = output
        has_bias = False
    else:
        bias_values = bias.to(device=device, dtype=output_dtype).contiguous()
        has_bias = True
    combined_scale = float(activation_scale * weight_scale)

    chunk_size = min(chunk_out_features or out_features, out_features)
    for output_col_start in range(0, out_features, chunk_size):
        current_chunk = min(chunk_size, out_features - output_col_start)
        packed_offset = output_col_start * packed_k
        packed_count = current_chunk * packed_k
        packed_weight_chunk = weights.narrow(0, packed_offset, packed_count)
        weight_storage = torch.empty(
            (current_chunk, in_features),
            device=device,
            dtype=torch.int8,
        )
        weight_total = current_chunk * in_features
        _decode_packed_w4_codes_kernel[
            (triton.cdiv(weight_total, decode_block_size),)
        ](
            packed_weight_chunk,
            weight_code_values,
            weight_storage,
            total=weight_total,
            block_size=decode_block_size,
        )
        accumulator = torch._int_mm(activation_values, weight_storage.t())
        del weight_storage
        epilogue_grid = (
            triton.cdiv(rows, epilogue_block_m),
            triton.cdiv(current_chunk, epilogue_block_n),
        )
        _scale_int32_matmul_output_chunk_kernel[epilogue_grid](
            accumulator,
            token_norm_values,
            row_norm_values,
            bias_values,
            output,
            output_col_start,
            rows=rows,
            chunk_out_features=current_chunk,
            out_features=out_features,
            combined_scale=combined_scale,
            has_bias=has_bias,
            block_m=epilogue_block_m,
            block_n=epilogue_block_n,
            num_warps=4,
        )
        del accumulator
    del activation_values
    return output.reshape(*original_shape[:-1], out_features)


def matmul_int8_activations_packed_w4_with_int_mm(
    int8_activations: torch.Tensor,
    packed_weight_indices: torch.Tensor,
    token_norms: torch.Tensor,
    row_norms: torch.Tensor,
    weight_codes: torch.Tensor,
    *,
    activation_scale: float,
    weight_scale: float,
    out_features: int,
    in_features: int,
    bias: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.bfloat16,
    epilogue_block_m: int = 32,
    epilogue_block_n: int = 256,
    chunk_out_features: int | None = None,
) -> torch.Tensor:
    return matmul_packed_w4a4_with_int_mm(
        int8_activations,
        packed_weight_indices,
        token_norms,
        row_norms,
        weight_codes,
        weight_codes,
        activation_scale=activation_scale,
        weight_scale=weight_scale,
        out_features=out_features,
        in_features=in_features,
        bias=bias,
        output_dtype=output_dtype,
        epilogue_block_m=epilogue_block_m,
        epilogue_block_n=epilogue_block_n,
        chunk_out_features=chunk_out_features,
        activations_are_int8=True,
    )


def pack_lowbit_with_triton(
    values: torch.Tensor, *, bits: int, validate: bool = True
) -> torch.Tensor:
    if bits <= 0 or bits > 8:
        raise ValueError("bits must be in [1, 8]")
    if not values.is_cuda:
        raise RuntimeError("triton_cuda pack requires CUDA tensors")
    max_value = (1 << bits) - 1
    flat_raw = values.detach().flatten()
    if validate and flat_raw.numel() and bool(
        torch.any((flat_raw < 0) | (flat_raw > max_value)).item()
    ):
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


def unpack_lowbit_with_triton(packed: torch.Tensor, *, bits: int, length: int) -> torch.Tensor:
    if bits <= 0 or bits > 8:
        raise ValueError("bits must be in [1, 8]")
    if length < 0:
        raise ValueError("length must be non-negative")
    if not packed.is_cuda:
        raise RuntimeError("triton_cuda unpack requires CUDA tensors")

    triton, _ = _load_triton()
    packed_contiguous = packed.detach().to(dtype=torch.uint8).contiguous().flatten()
    values = torch.empty(length, device=packed.device, dtype=torch.uint8)
    if length == 0:
        return values

    block_size = 256
    grid = (triton.cdiv(length, block_size),)
    _unpack_lowbit_kernel[grid](
        packed_contiguous,
        values,
        length=length,
        bits=bits,
        block_size=block_size,
    )
    return values


def row_norms_with_triton(weight: torch.Tensor, *, eps: float) -> torch.Tensor:
    if not weight.is_cuda:
        raise RuntimeError("triton_cuda row norm requires CUDA tensors")
    if weight.ndim != 2:
        raise ValueError("weight must be a rank-2 tensor")
    if not weight.is_floating_point():
        raise TypeError("weight must be a floating point tensor")

    triton, _ = _load_triton()
    out_features, in_features = weight.shape
    norms = torch.empty(out_features, device=weight.device, dtype=torch.float32)
    if out_features == 0:
        return norms

    weight_contiguous = weight.contiguous()
    norm_block_size = triton.next_power_of_2(in_features)
    _row_norm_kernel[(out_features,)](
        weight_contiguous,
        norms,
        rows=out_features,
        dim=in_features,
        eps=float(eps),
        block_size=norm_block_size,
        num_warps=8,
    )
    return norms


def quantize_weight_indices_with_triton(
    weight: torch.Tensor,
    row_norms: torch.Tensor,
    *,
    rotation: RPBHRotation,
    codebook: LloydMaxCodebook,
    eps: float = 1e-10,
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
    weight_contiguous = weight.contiguous()
    norms = row_norms.to(device=weight.device, dtype=torch.float32).contiguous()
    permutation, signs, boundaries = _weight_quantization_constants(
        rotation=rotation,
        codebook=codebook,
        device=weight.device,
    )
    work = torch.empty(total, device=weight.device, dtype=torch.float32)
    indices = torch.empty(total, device=weight.device, dtype=torch.uint8)
    if total == 0:
        return indices.reshape(out_features, in_features)

    element_block_size = 256
    _permute_sign_weight_kernel[(triton.cdiv(total, element_block_size),)](
        weight_contiguous,
        permutation,
        signs,
        work,
        total=total,
        in_features=in_features,
        block_size=element_block_size,
    )

    orbit_block_size = int(rotation.block_size)
    num_blocks = int(rotation.num_blocks)
    stage_width = 1
    while stage_width * 2 < orbit_block_size:
        total_quads = out_features * num_blocks * (orbit_block_size // 4)
        _fwht_two_stage_kernel[(triton.cdiv(total_quads, element_block_size),)](
            work,
            total_quads=total_quads,
            in_features=in_features,
            num_blocks=num_blocks,
            orbit_block_size=orbit_block_size,
            stage_width=stage_width,
            block_size=element_block_size,
        )
        stage_width *= 4
    if stage_width < orbit_block_size:
        total_pairs = out_features * num_blocks * (orbit_block_size // 2)
        _fwht_stage_weight_kernel[(triton.cdiv(total_pairs, element_block_size),)](
            work,
            total_pairs=total_pairs,
            in_features=in_features,
            num_blocks=num_blocks,
            orbit_block_size=orbit_block_size,
            stage_width=stage_width,
            block_size=element_block_size,
        )

    _quantize_rotated_weight_indices_kernel[(triton.cdiv(total, element_block_size),)](
        work,
        norms,
        boundaries,
        indices,
        total=total,
        in_features=in_features,
        levels=codebook.centroids.numel(),
        eps=float(eps),
        inv_sqrt_block=float(rotation.normalization),
        block_size=element_block_size,
    )
    return indices.reshape(out_features, in_features)


def quantize_weight_packed_with_triton(
    weight: torch.Tensor,
    row_norms: torch.Tensor,
    *,
    rotation: RPBHRotation,
    codebook: LloydMaxCodebook,
    bits: int,
    eps: float = 1e-10,
) -> torch.Tensor:
    if bits <= 0 or bits > 8:
        raise ValueError("bits must be in [1, 8]")
    if not weight.is_cuda:
        raise RuntimeError("triton_cuda packed weight quantization requires CUDA tensors")
    if weight.ndim != 2:
        raise ValueError("weight must be a rank-2 tensor")
    if weight.shape[-1] != rotation.dim:
        raise ValueError(f"expected in_features {rotation.dim}, got {weight.shape[-1]}")

    triton, _ = _load_triton()
    out_features, in_features = weight.shape
    total = out_features * in_features
    packed_length = (total * bits + 7) // 8
    packed = torch.empty(packed_length, device=weight.device, dtype=torch.uint8)
    if total == 0:
        return packed

    weight_contiguous = weight.contiguous()
    norms = row_norms.to(device=weight.device, dtype=torch.float32).contiguous()
    permutation, signs, boundaries = _weight_quantization_constants(
        rotation=rotation,
        codebook=codebook,
        device=weight.device,
    )
    work = torch.empty(total, device=weight.device, dtype=torch.float32)

    element_block_size = 256
    _permute_sign_weight_kernel[(triton.cdiv(total, element_block_size),)](
        weight_contiguous,
        permutation,
        signs,
        work,
        total=total,
        in_features=in_features,
        block_size=element_block_size,
    )

    orbit_block_size = int(rotation.block_size)
    num_blocks = int(rotation.num_blocks)
    stage_width = 1
    while stage_width * 2 < orbit_block_size:
        total_quads = out_features * num_blocks * (orbit_block_size // 4)
        _fwht_two_stage_kernel[(triton.cdiv(total_quads, element_block_size),)](
            work,
            total_quads=total_quads,
            in_features=in_features,
            num_blocks=num_blocks,
            orbit_block_size=orbit_block_size,
            stage_width=stage_width,
            block_size=element_block_size,
        )
        stage_width *= 4
    if stage_width < orbit_block_size:
        total_pairs = out_features * num_blocks * (orbit_block_size // 2)
        _fwht_stage_weight_kernel[(triton.cdiv(total_pairs, element_block_size),)](
            work,
            total_pairs=total_pairs,
            in_features=in_features,
            num_blocks=num_blocks,
            orbit_block_size=orbit_block_size,
            stage_width=stage_width,
            block_size=element_block_size,
        )

    _quantize_rotated_weight_pack_kernel[(triton.cdiv(packed_length, element_block_size),)](
        work,
        norms,
        boundaries,
        packed,
        packed_length=packed_length,
        value_count=total,
        in_features=in_features,
        bits=bits,
        levels=codebook.centroids.numel(),
        eps=float(eps),
        inv_sqrt_block=float(rotation.normalization),
        block_size=element_block_size,
    )
    return packed
