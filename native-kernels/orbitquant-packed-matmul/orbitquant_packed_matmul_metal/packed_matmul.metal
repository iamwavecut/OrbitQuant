#include <metal_stdlib>

using namespace metal;

struct PackedMatmulParams {
  long rows;
  long out_features;
  long in_features;
  long bits;
  long block_k;
  int has_bias;
};

inline float bf16_to_float(ushort value) {
  return as_type<float>(uint(value) << 16);
}

inline ushort float_to_bf16(float value) {
  const uint bits = as_type<uint>(value);
  const uint lsb = (bits >> 16) & 1u;
  return ushort((bits + 0x7fffu + lsb) >> 16);
}

template <typename scalar_t>
inline void packed_matmul_tiled_value(
    device scalar_t *out,
    device const scalar_t *x,
    device const uchar *packed_weight_indices,
    device const float *row_norms,
    device const float *centroids,
    device const float *bias,
    constant PackedMatmulParams &params,
    threadgroup float *shared,
    uint2 tid,
    uint2 local_tid,
    uint2 threads_per_group) {
  const long col = tid.x;
  const long row = tid.y;
  const bool output_valid = row < params.rows && col < params.out_features;
  const long block_k = params.block_k;
  threadgroup float *x_tile = shared;
  threadgroup float *w_tile = shared + long(threads_per_group.y) * block_k;
  const long local_col = local_tid.x;
  const long local_row = local_tid.y;
  const long thread_linear = local_row * long(threads_per_group.x) + local_col;
  const long thread_count = long(threads_per_group.x) * long(threads_per_group.y);

  const uint mask = (1u << uint(params.bits)) - 1u;
  float acc = output_valid && params.has_bias != 0 ? bias[col] : 0.0f;

  for (long k_start = 0; k_start < params.in_features; k_start += block_k) {
    const long x_tile_values = long(threads_per_group.y) * block_k;
    for (long offset = thread_linear; offset < x_tile_values; offset += thread_count) {
      const long tile_row = offset / block_k;
      const long tile_k = offset - tile_row * block_k;
      const long global_row = long(tid.y) - local_row + tile_row;
      const long global_k = k_start + tile_k;
      float value = 0.0f;
      if (global_row < params.rows && global_k < params.in_features) {
        value = float(x[global_row * params.in_features + global_k]);
      }
      x_tile[offset] = value;
    }

    const long w_tile_values = block_k * long(threads_per_group.x);
    for (long offset = thread_linear; offset < w_tile_values; offset += thread_count) {
      const long tile_k = offset / long(threads_per_group.x);
      const long tile_col = offset - tile_k * long(threads_per_group.x);
      const long global_k = k_start + tile_k;
      const long global_col = long(tid.x) - local_col + tile_col;
      float value = 0.0f;
      if (global_col < params.out_features && global_k < params.in_features) {
        const long value_offset = global_col * params.in_features + global_k;
        const long bit_start = value_offset * params.bits;
        const long byte_index = bit_start >> 3;
        const long bit_offset = bit_start & 7;
        uint raw = packed_weight_indices[byte_index];
        if (bit_offset + params.bits > 8) {
          raw |= uint(packed_weight_indices[byte_index + 1]) << 8;
        }
        const uint index = (raw >> uint(bit_offset)) & mask;
        value = row_norms[global_col] * centroids[index];
      }
      w_tile[offset] = value;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (output_valid) {
      for (long tile_k = 0; tile_k < block_k && k_start + tile_k < params.in_features;
           ++tile_k) {
        acc += x_tile[local_row * block_k + tile_k] *
            w_tile[tile_k * long(threads_per_group.x) + local_col];
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (output_valid) {
    out[row * params.out_features + col] = scalar_t(acc);
  }
}

inline void packed_matmul_tiled_value_bfloat16(
    device ushort *out,
    device const ushort *x,
    device const uchar *packed_weight_indices,
    device const float *row_norms,
    device const float *centroids,
    device const float *bias,
    constant PackedMatmulParams &params,
    threadgroup float *shared,
    uint2 tid,
    uint2 local_tid,
    uint2 threads_per_group) {
  const long col = tid.x;
  const long row = tid.y;
  const bool output_valid = row < params.rows && col < params.out_features;
  const long block_k = params.block_k;
  threadgroup float *x_tile = shared;
  threadgroup float *w_tile = shared + long(threads_per_group.y) * block_k;
  const long local_col = local_tid.x;
  const long local_row = local_tid.y;
  const long thread_linear = local_row * long(threads_per_group.x) + local_col;
  const long thread_count = long(threads_per_group.x) * long(threads_per_group.y);

  const uint mask = (1u << uint(params.bits)) - 1u;
  float acc = output_valid && params.has_bias != 0 ? bias[col] : 0.0f;

  for (long k_start = 0; k_start < params.in_features; k_start += block_k) {
    const long x_tile_values = long(threads_per_group.y) * block_k;
    for (long offset = thread_linear; offset < x_tile_values; offset += thread_count) {
      const long tile_row = offset / block_k;
      const long tile_k = offset - tile_row * block_k;
      const long global_row = long(tid.y) - local_row + tile_row;
      const long global_k = k_start + tile_k;
      float value = 0.0f;
      if (global_row < params.rows && global_k < params.in_features) {
        value = bf16_to_float(x[global_row * params.in_features + global_k]);
      }
      x_tile[offset] = value;
    }

    const long w_tile_values = block_k * long(threads_per_group.x);
    for (long offset = thread_linear; offset < w_tile_values; offset += thread_count) {
      const long tile_k = offset / long(threads_per_group.x);
      const long tile_col = offset - tile_k * long(threads_per_group.x);
      const long global_k = k_start + tile_k;
      const long global_col = long(tid.x) - local_col + tile_col;
      float value = 0.0f;
      if (global_col < params.out_features && global_k < params.in_features) {
        const long value_offset = global_col * params.in_features + global_k;
        const long bit_start = value_offset * params.bits;
        const long byte_index = bit_start >> 3;
        const long bit_offset = bit_start & 7;
        uint raw = packed_weight_indices[byte_index];
        if (bit_offset + params.bits > 8) {
          raw |= uint(packed_weight_indices[byte_index + 1]) << 8;
        }
        const uint index = (raw >> uint(bit_offset)) & mask;
        value = row_norms[global_col] * centroids[index];
      }
      w_tile[offset] = value;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (output_valid) {
      for (long tile_k = 0; tile_k < block_k && k_start + tile_k < params.in_features;
           ++tile_k) {
        acc += x_tile[local_row * block_k + tile_k] *
            w_tile[tile_k * long(threads_per_group.x) + local_col];
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (output_valid) {
    out[row * params.out_features + col] = float_to_bf16(acc);
  }
}

kernel void packed_matmul_forward_float(
    device float *out [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device const uchar *packed_weight_indices [[buffer(2)]],
    device const float *row_norms [[buffer(3)]],
    device const float *centroids [[buffer(4)]],
    device const float *bias [[buffer(5)]],
    constant PackedMatmulParams &params [[buffer(6)]],
    threadgroup float *shared [[threadgroup(0)]],
    uint2 tid [[thread_position_in_grid]],
    uint2 local_tid [[thread_position_in_threadgroup]],
    uint2 threads_per_group [[threads_per_threadgroup]]) {
  packed_matmul_tiled_value(
      out,
      x,
      packed_weight_indices,
      row_norms,
      centroids,
      bias,
      params,
      shared,
      tid,
      local_tid,
      threads_per_group);
}

kernel void packed_matmul_forward_half(
    device half *out [[buffer(0)]],
    device const half *x [[buffer(1)]],
    device const uchar *packed_weight_indices [[buffer(2)]],
    device const float *row_norms [[buffer(3)]],
    device const float *centroids [[buffer(4)]],
    device const float *bias [[buffer(5)]],
    constant PackedMatmulParams &params [[buffer(6)]],
    threadgroup float *shared [[threadgroup(0)]],
    uint2 tid [[thread_position_in_grid]],
    uint2 local_tid [[thread_position_in_threadgroup]],
    uint2 threads_per_group [[threads_per_threadgroup]]) {
  packed_matmul_tiled_value(
      out,
      x,
      packed_weight_indices,
      row_norms,
      centroids,
      bias,
      params,
      shared,
      tid,
      local_tid,
      threads_per_group);
}

kernel void packed_matmul_forward_bfloat16(
    device ushort *out [[buffer(0)]],
    device const ushort *x [[buffer(1)]],
    device const uchar *packed_weight_indices [[buffer(2)]],
    device const float *row_norms [[buffer(3)]],
    device const float *centroids [[buffer(4)]],
    device const float *bias [[buffer(5)]],
    constant PackedMatmulParams &params [[buffer(6)]],
    threadgroup float *shared [[threadgroup(0)]],
    uint2 tid [[thread_position_in_grid]],
    uint2 local_tid [[thread_position_in_threadgroup]],
    uint2 threads_per_group [[threads_per_threadgroup]]) {
  packed_matmul_tiled_value_bfloat16(
      out,
      x,
      packed_weight_indices,
      row_norms,
      centroids,
      bias,
      params,
      shared,
      tid,
      local_tid,
      threads_per_group);
}
