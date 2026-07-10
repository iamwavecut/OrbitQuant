from __future__ import annotations

from functools import lru_cache

import torch

from orbitquant.codebooks import LloydMaxCodebook
from orbitquant.rotations import RPBHRotation

_MAX_FUSED_RPBH_BLOCK_SIZE = 4096

_MPS_KERNEL_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

constant uint orbitquant_activation_threads = 256;
constant uint orbitquant_wide_activation_threads = 512;
constant uint orbitquant_max_rpbh_block_size = 4096;

template <typename scalar_t>
inline void orbitquant_row_norm_impl(
    const device scalar_t* input,
    device float* norms,
    constant int& rows,
    constant int& dim,
    threadgroup float* scratch,
    uint row,
    uint thread_index,
    uint thread_count) {
  if (row >= uint(rows)) {
    return;
  }

  float sum = 0.0f;
  const uint row_offset = row * uint(dim);
  for (uint col = thread_index; col < uint(dim); col += thread_count) {
    const float value = float(input[row_offset + col]);
    sum += value * value;
  }
  scratch[thread_index] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = thread_count >> 1; stride > 0; stride >>= 1) {
    if (thread_index < stride) {
      scratch[thread_index] += scratch[thread_index + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (thread_index == 0) {
    norms[row] = sqrt(scratch[0]);
  }
}

template <typename scalar_t>
inline void orbitquant_fused_activation_impl(
    const device scalar_t* input,
    const device long* permutation,
    const device char* signs,
    const device float* centroids,
    const device float* boundaries,
    const device float* norms,
    device scalar_t* output,
    constant int& rows,
    constant int& dim,
    constant int& block_size,
    constant int& num_blocks,
    constant int& levels,
    constant float& normalization,
    constant float& eps,
    threadgroup float* values,
    uint group,
    uint thread_index,
    uint thread_count) {
  const uint row = group / uint(num_blocks);
  if (row >= uint(rows)) {
    return;
  }
  const uint block = group - row * uint(num_blocks);
  const uint block_start = block * uint(block_size);
  const uint row_offset = row * uint(dim);
  const float norm = norms[row];
  const float inverse_norm = 1.0f / (norm + eps);

  for (uint col = thread_index; col < uint(block_size); col += thread_count) {
    const uint rotated_col = block_start + col;
    const uint source_col = uint(permutation[rotated_col]);
    values[col] = float(input[row_offset + source_col]) *
        float(signs[rotated_col]) * inverse_norm;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = 1; stride < uint(block_size); stride <<= 1) {
    const uint pairs = uint(block_size) >> 1;
    for (uint pair = thread_index; pair < pairs; pair += thread_count) {
      const uint pair_group = pair / stride;
      const uint pair_offset = pair - pair_group * stride;
      const uint left = pair_group * (stride << 1) + pair_offset;
      const uint right = left + stride;
      const float left_value = values[left];
      const float right_value = values[right];
      values[left] = left_value + right_value;
      values[right] = left_value - right_value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (uint col = thread_index; col < uint(block_size); col += thread_count) {
    const float value = values[col] * normalization;
    int index = 0;
    for (int boundary = 0; boundary < levels - 1; ++boundary) {
      index += value > boundaries[boundary];
    }
    output[row_offset + block_start + col] = scalar_t(centroids[index] * norm);
  }
}

#define ORBITQUANT_ROW_NORM_KERNEL(NAME, SCALAR_T, THREADS)                  \
kernel void NAME(                                                            \
    const device SCALAR_T* input [[buffer(0)]],                              \
    device float* norms [[buffer(1)]],                                        \
    constant int& rows [[buffer(2)]],                                         \
    constant int& dim [[buffer(3)]],                                          \
    uint3 group_id [[threadgroup_position_in_grid]],                          \
    uint thread_index [[thread_index_in_threadgroup]],                        \
    uint3 group_size [[threads_per_threadgroup]]) {                           \
  threadgroup float scratch[THREADS];                                         \
  orbitquant_row_norm_impl(                                                   \
      input, norms, rows, dim, scratch, group_id.x, thread_index, group_size.x); \
}

ORBITQUANT_ROW_NORM_KERNEL(
    orbitquant_row_norm_float, float, orbitquant_activation_threads)
ORBITQUANT_ROW_NORM_KERNEL(
    orbitquant_row_norm_half, half, orbitquant_activation_threads)
ORBITQUANT_ROW_NORM_KERNEL(
    orbitquant_row_norm_bfloat16, bfloat, orbitquant_activation_threads)
ORBITQUANT_ROW_NORM_KERNEL(
    orbitquant_row_norm_float_wide, float, orbitquant_wide_activation_threads)
ORBITQUANT_ROW_NORM_KERNEL(
    orbitquant_row_norm_half_wide, half, orbitquant_wide_activation_threads)
ORBITQUANT_ROW_NORM_KERNEL(
    orbitquant_row_norm_bfloat16_wide, bfloat, orbitquant_wide_activation_threads)

#undef ORBITQUANT_ROW_NORM_KERNEL

#define ORBITQUANT_FUSED_ACTIVATION_KERNEL(NAME, SCALAR_T)                  \
kernel void NAME(                                                            \
    const device SCALAR_T* input [[buffer(0)]],                              \
    const device long* permutation [[buffer(1)]],                            \
    const device char* signs [[buffer(2)]],                                  \
    const device float* centroids [[buffer(3)]],                             \
    const device float* boundaries [[buffer(4)]],                            \
    const device float* norms [[buffer(5)]],                                 \
    device SCALAR_T* output [[buffer(6)]],                                   \
    constant int& rows [[buffer(7)]],                                        \
    constant int& dim [[buffer(8)]],                                         \
    constant int& block_size [[buffer(9)]],                                  \
    constant int& num_blocks [[buffer(10)]],                                 \
    constant int& levels [[buffer(11)]],                                     \
    constant float& normalization [[buffer(12)]],                            \
    constant float& eps [[buffer(13)]],                                      \
    uint3 group_id [[threadgroup_position_in_grid]],                         \
    uint thread_index [[thread_index_in_threadgroup]],                       \
    uint3 group_size [[threads_per_threadgroup]]) {                          \
  threadgroup float values[orbitquant_max_rpbh_block_size];                  \
  orbitquant_fused_activation_impl(                                          \
      input, permutation, signs, centroids, boundaries, norms, output, rows, \
      dim, block_size, num_blocks, levels, normalization, eps, values,       \
      group_id.x, thread_index, group_size.x);                               \
}

ORBITQUANT_FUSED_ACTIVATION_KERNEL(orbitquant_fused_activation_float, float)
ORBITQUANT_FUSED_ACTIVATION_KERNEL(orbitquant_fused_activation_half, half)
ORBITQUANT_FUSED_ACTIVATION_KERNEL(
    orbitquant_fused_activation_bfloat16, bfloat)

#undef ORBITQUANT_FUSED_ACTIVATION_KERNEL

kernel void orbitquant_codebook_rescale(
    const device float* rotated [[buffer(0)]],
    const device float* norms [[buffer(1)]],
    const device float* centroids [[buffer(2)]],
    const device float* boundaries [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant int& total [[buffer(5)]],
    constant int& dim [[buffer(6)]],
    constant int& levels [[buffer(7)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid >= uint(total)) {
    return;
  }
  float value = rotated[tid];
  int index = 0;
  for (int idx = 0; idx < levels - 1; ++idx) {
    if (value > boundaries[idx]) {
      index += 1;
    }
  }
  uint row = tid / uint(dim);
  output[tid] = centroids[index] * norms[row];
}

kernel void orbitquant_dequantize_packed_weight(
    const device uchar* packed [[buffer(0)]],
    const device float* row_norms [[buffer(1)]],
    const device float* centroids [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant int& total [[buffer(4)]],
    constant int& in_features [[buffer(5)]],
    constant int& bits [[buffer(6)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid >= uint(total)) {
    return;
  }

  uint bit_start = tid * uint(bits);
  uint byte_index = bit_start >> 3;
  uint bit_offset = bit_start & 7;
  uint raw = uint(packed[byte_index]);
  if (bit_offset + uint(bits) > 8) {
    raw |= uint(packed[byte_index + 1]) << 8;
  }
  uint mask = (1u << uint(bits)) - 1u;
  uint index = (raw >> bit_offset) & mask;
  uint row = tid / uint(in_features);
  output[tid] = centroids[index] * row_norms[row];
}
"""


def mps_metal_available() -> bool:
    return bool(torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader"))


@lru_cache(maxsize=1)
def _mps_shader():
    if not mps_metal_available():
        raise RuntimeError("MPS Metal shader backend is not available in this environment")
    return torch.mps.compile_shader(_MPS_KERNEL_SOURCE)


def _activation_kernel_names(dtype: torch.dtype) -> tuple[str, str, str]:
    names = {
        torch.float32: (
            "orbitquant_row_norm_float",
            "orbitquant_row_norm_float_wide",
            "orbitquant_fused_activation_float",
        ),
        torch.float16: (
            "orbitquant_row_norm_half",
            "orbitquant_row_norm_half_wide",
            "orbitquant_fused_activation_half",
        ),
        torch.bfloat16: (
            "orbitquant_row_norm_bfloat16",
            "orbitquant_row_norm_bfloat16_wide",
            "orbitquant_fused_activation_bfloat16",
        ),
    }
    try:
        return names[dtype]
    except KeyError as exc:
        raise TypeError(
            "MPS fused activation quantization supports float32, float16, and bfloat16; "
            f"got {dtype}"
        ) from exc


def quantize_activations_with_mps(
    x: torch.Tensor,
    rotation: RPBHRotation,
    codebook: LloydMaxCodebook,
    *,
    eps: float,
    constant_tensors: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    if x.device.type != "mps":
        raise RuntimeError("mps backend requires MPS tensors")
    if x.shape[-1] != rotation.dim:
        raise ValueError(f"expected last dimension {rotation.dim}, got {x.shape[-1]}")
    if rotation.block_size > _MAX_FUSED_RPBH_BLOCK_SIZE:
        raise RuntimeError(
            "MPS fused activation quantization supports RPBH block sizes up to "
            f"{_MAX_FUSED_RPBH_BLOCK_SIZE}; got {rotation.block_size}. "
            "Choose a smaller block_size or use activation_kernel_backend='cpu' "
            "as a reference path."
        )

    contiguous = x.contiguous()
    flat = contiguous.reshape(-1, rotation.dim)
    rows = flat.shape[0]
    if rows == 0:
        return torch.empty_like(contiguous)

    if constant_tensors is None:
        permutation = rotation.permutation.to(device=x.device, dtype=torch.int64)
        signs = rotation.signs.to(device=x.device, dtype=torch.int8)
        centroids = codebook.centroids.to(device=x.device, dtype=torch.float32)
        boundaries = codebook.boundaries.to(device=x.device, dtype=torch.float32)
    else:
        permutation = constant_tensors["permutation"].to(device=x.device, dtype=torch.int64)
        signs = constant_tensors["signs"].to(device=x.device, dtype=torch.int8)
        centroids = constant_tensors["centroids"].to(device=x.device, dtype=torch.float32)
        boundaries = constant_tensors["boundaries"].to(device=x.device, dtype=torch.float32)

    permutation = permutation.contiguous()
    signs = signs.contiguous()
    centroids = centroids.contiguous()
    boundaries = boundaries.contiguous()
    norms = torch.empty(rows, device=x.device, dtype=torch.float32)
    output = torch.empty_like(flat)
    norm_name, wide_norm_name, activation_name = _activation_kernel_names(x.dtype)
    shader = _mps_shader()
    activation_threads = (
        512 if rotation.block_size == 4096 and rotation.num_blocks == 1 else 256
    )

    getattr(shader, wide_norm_name if activation_threads == 512 else norm_name)(
        flat,
        norms,
        rows,
        rotation.dim,
        threads=[rows * activation_threads, 1, 1],
        group_size=[activation_threads, 1, 1],
    )
    groups = rows * rotation.num_blocks
    getattr(shader, activation_name)(
        flat,
        permutation,
        signs,
        centroids,
        boundaries,
        norms,
        output,
        rows,
        rotation.dim,
        rotation.block_size,
        rotation.num_blocks,
        centroids.numel(),
        rotation.normalization,
        eps,
        threads=[groups * activation_threads, 1, 1],
        group_size=[activation_threads, 1, 1],
    )
    return output.reshape_as(contiguous)


def quantize_rotated_activations_with_mps(
    rotated: torch.Tensor,
    norms: torch.Tensor,
    codebook: LloydMaxCodebook,
    *,
    constant_tensors: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    if rotated.device.type != "mps":
        raise RuntimeError("mps backend requires MPS tensors")
    rotated_contiguous = rotated.to(torch.float32).contiguous()
    flat = rotated_contiguous.reshape(-1)
    if flat.numel() == 0:
        return torch.empty_like(rotated_contiguous, dtype=torch.float32)

    row_norms = norms.contiguous().reshape(-1).to(device=rotated.device, dtype=torch.float32)
    if constant_tensors is None:
        centroids = codebook.centroids.to(device=rotated.device, dtype=torch.float32)
        boundaries = codebook.boundaries.to(device=rotated.device, dtype=torch.float32)
    else:
        centroids = constant_tensors["centroids"].to(device=rotated.device, dtype=torch.float32)
        boundaries = constant_tensors["boundaries"].to(device=rotated.device, dtype=torch.float32)
    centroids = centroids.contiguous()
    boundaries = boundaries.contiguous()
    output = torch.empty_like(flat, dtype=torch.float32)

    shader = _mps_shader()
    shader.orbitquant_codebook_rescale(
        flat,
        row_norms,
        centroids,
        boundaries,
        output,
        flat.numel(),
        rotated.shape[-1],
        centroids.numel(),
        threads=[flat.numel(), 1, 1],
        group_size=[min(flat.numel(), 256), 1, 1],
    )
    return output.reshape_as(rotated_contiguous)


def dequantize_packed_weight_with_mps(
    packed_weight_indices: torch.Tensor,
    row_norms: torch.Tensor,
    codebook: LloydMaxCodebook,
    *,
    bits: int,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    total = out_features * in_features
    if total == 0:
        return torch.empty(out_features, in_features, device="mps", dtype=torch.float32)

    packed = packed_weight_indices.to(device="mps", dtype=torch.uint8).contiguous()
    norms = row_norms.to(device="mps", dtype=torch.float32).contiguous()
    centroids = codebook.centroids.to(device="mps", dtype=torch.float32).contiguous()
    output = torch.empty(total, device="mps", dtype=torch.float32)

    shader = _mps_shader()
    shader.orbitquant_dequantize_packed_weight(
        packed,
        norms,
        centroids,
        output,
        total,
        in_features,
        bits,
        threads=[total, 1, 1],
        group_size=[min(total, 256), 1, 1],
    )
    return output.reshape(out_features, in_features)
