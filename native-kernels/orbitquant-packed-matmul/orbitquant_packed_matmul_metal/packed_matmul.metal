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

constant uint packed_small_cols = 4;

template <typename scalar_t>
inline void packed_matmul_small_rows_value(
    device scalar_t *out,
    device const scalar_t *x,
    device const uchar *packed_weight_indices,
    device const float *row_norms,
    device const float *centroids,
    device const float *bias,
    constant PackedMatmulParams &params,
    uint2 group_id,
    ushort lane) {
  const long row = long(group_id.y);
  const long col_start = long(group_id.x) * packed_small_cols;
  const uint bits = uint(params.bits);
  const uint mask = (1u << bits) - 1u;
  float accumulators[packed_small_cols] = {0.0f, 0.0f, 0.0f, 0.0f};
  float norms[packed_small_cols];

#pragma clang loop unroll(full)
  for (uint col_offset = 0; col_offset < packed_small_cols; ++col_offset) {
    const long col = col_start + long(col_offset);
    norms[col_offset] = col < params.out_features ? row_norms[col] : 0.0f;
  }

  for (long k = long(lane); k < params.in_features; k += 32) {
    const float x_value = float(x[row * params.in_features + k]);
#pragma clang loop unroll(full)
    for (uint col_offset = 0; col_offset < packed_small_cols; ++col_offset) {
      const long col = col_start + long(col_offset);
      if (col >= params.out_features) {
        continue;
      }
      const long value_offset = col * params.in_features + k;
      const long bit_start = value_offset * params.bits;
      const long byte_index = bit_start >> 3;
      const uint bit_offset = uint(bit_start & 7);
      uint raw = packed_weight_indices[byte_index];
      if (bit_offset + bits > 8) {
        raw |= uint(packed_weight_indices[byte_index + 1]) << 8;
      }
      const uint index = (raw >> bit_offset) & mask;
      accumulators[col_offset] += x_value * norms[col_offset] * centroids[index];
    }
  }

#pragma clang loop unroll(full)
  for (uint col_offset = 0; col_offset < packed_small_cols; ++col_offset) {
    const float value = simd_sum(accumulators[col_offset]);
    const long col = col_start + long(col_offset);
    if (lane == 0 && col < params.out_features) {
      out[row * params.out_features + col] =
          scalar_t(value + (params.has_bias != 0 ? bias[col] : 0.0f));
    }
  }
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

constant uint packed_mma_tile = 32;
constant uint packed_mma_padded_k = 40;

struct alignas(16) PackedMMAReadVector {
  uchar values[16];
};

struct alignas(1) PackedMMARead3 {
  uchar values[3];
};

template <typename scalar_t>
inline void packed_mma_fragment(
    thread float2 &output,
    thread vec<scalar_t, 2> &lhs,
    thread vec<scalar_t, 2> &rhs,
    thread float2 &accumulator) {
  simdgroup_matrix<float, 8, 8> output_matrix;
  simdgroup_matrix<scalar_t, 8, 8> lhs_matrix;
  simdgroup_matrix<scalar_t, 8, 8> rhs_matrix;
  simdgroup_matrix<float, 8, 8> accumulator_matrix;
  reinterpret_cast<thread vec<scalar_t, 2> &>(lhs_matrix.thread_elements()) = lhs;
  reinterpret_cast<thread vec<scalar_t, 2> &>(rhs_matrix.thread_elements()) = rhs;
  reinterpret_cast<thread float2 &>(accumulator_matrix.thread_elements()) = accumulator;
  simdgroup_multiply_accumulate(
      output_matrix, lhs_matrix, rhs_matrix, accumulator_matrix);
  output = reinterpret_cast<thread float2 &>(output_matrix.thread_elements());
}

template <typename scalar_t, uint Bits>
inline void decode_packed_mma_weight_segment(
    threadgroup scalar_t *destination,
    device const uchar *packed_weight_indices,
    device const float *row_norms,
    device const float *centroids,
    long global_col,
    long global_k,
    long in_features) {
  constexpr uint values = 8;
  constexpr uint byte_count = Bits;
  constexpr uint mask = (1u << Bits) - 1u;
  const long value_offset = global_col * in_features + global_k;
  const long byte_index = (value_offset * Bits) >> 3;
  const float norm = row_norms[global_col];

  if (Bits == 2) {
    const ushort packed = *reinterpret_cast<device const ushort *>(
        packed_weight_indices + byte_index);
#pragma clang loop unroll(full)
    for (uint idx = 0; idx < values; ++idx) {
      destination[idx] =
          scalar_t(norm * centroids[(uint(packed) >> (idx * 2)) & 3u]);
    }
    return;
  }

  if (Bits == 3) {
    const PackedMMARead3 bytes = *reinterpret_cast<device const PackedMMARead3 *>(
        packed_weight_indices + byte_index);
    const uint packed =
        uint(bytes.values[0]) | (uint(bytes.values[1]) << 8) |
        (uint(bytes.values[2]) << 16);
#pragma clang loop unroll(full)
    for (uint idx = 0; idx < values; ++idx) {
      destination[idx] =
          scalar_t(norm * centroids[(packed >> (idx * 3)) & 7u]);
    }
    return;
  }

  if (Bits == 4) {
    const uchar4 packed = *reinterpret_cast<device const uchar4 *>(
        packed_weight_indices + byte_index);
#pragma clang loop unroll(full)
    for (uint idx = 0; idx < 4; ++idx) {
      const uchar value = packed[idx];
      destination[idx * 2] = scalar_t(norm * centroids[uint(value) & 15u]);
      destination[idx * 2 + 1] =
          scalar_t(norm * centroids[(uint(value) >> 4) & 15u]);
    }
    return;
  }

  uchar packed[byte_count];
#pragma clang loop unroll(full)
  for (uint idx = 0; idx < byte_count; ++idx) {
    packed[idx] = packed_weight_indices[byte_index + idx];
  }
#pragma clang loop unroll(full)
  for (uint idx = 0; idx < values; ++idx) {
    const uint bit_start = idx * Bits;
    const uint source_byte = bit_start >> 3;
    const uint shift = bit_start & 7u;
    uint raw = uint(packed[source_byte]);
    if (shift + Bits > 8u) {
      raw |= uint(packed[source_byte + 1]) << 8;
    }
    destination[idx] = scalar_t(norm * centroids[(raw >> shift) & mask]);
  }
}

template <typename scalar_t, uint Bits>
inline void packed_matmul_padded_mma_value(
    device scalar_t *out,
    device const scalar_t *x,
    device const uchar *packed_weight_indices,
    device const float *row_norms,
    device const float *centroids,
    device const float *bias,
    constant PackedMatmulParams &params,
    threadgroup uchar *shared,
    uint2 group_id,
    uint thread_index,
    ushort simdgroup_id,
    ushort lane) {
  threadgroup scalar_t *x_tile = reinterpret_cast<threadgroup scalar_t *>(shared);
  threadgroup scalar_t *weight_tile = x_tile + packed_mma_tile * packed_mma_padded_k;
  const long row_start = long(group_id.y) * packed_mma_tile;
  const long col_start = long(group_id.x) * packed_mma_tile;
  const ushort load_row = ushort(thread_index / 4);
  const ushort load_k = ushort(thread_index % 4) * 8;
  const ushort quad = lane / 4;
  const ushort fragment_row = (quad & 4) + ((lane / 2) % 4);
  const ushort fragment_col = (quad & 2) * 2 + (lane % 2) * 2;
  const ushort simd_row = 8 * (simdgroup_id / 2);
  const ushort simd_col = 8 * (simdgroup_id % 2);

  float2 accumulators[2][2];
#pragma clang loop unroll(full)
  for (ushort matrix_m = 0; matrix_m < 2; ++matrix_m) {
#pragma clang loop unroll(full)
    for (ushort matrix_n = 0; matrix_n < 2; ++matrix_n) {
      accumulators[matrix_m][matrix_n] = float2(0.0f);
    }
  }

  for (long k_start = 0; k_start < params.in_features; k_start += packed_mma_tile) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup scalar_t *x_destination =
        x_tile + load_row * packed_mma_padded_k + load_k;
    if (row_start + long(load_row) < params.rows) {
      *reinterpret_cast<threadgroup PackedMMAReadVector *>(x_destination) =
          *reinterpret_cast<device const PackedMMAReadVector *>(
              x + (row_start + long(load_row)) * params.in_features +
              k_start + long(load_k));
    } else {
#pragma clang loop unroll(full)
      for (ushort offset = 0; offset < 8; ++offset) {
        x_destination[offset] = scalar_t(0.0f);
      }
    }
    decode_packed_mma_weight_segment<scalar_t, Bits>(
        weight_tile + load_row * packed_mma_padded_k + load_k,
        packed_weight_indices,
        row_norms,
        centroids,
        col_start + long(load_row),
        k_start + long(load_k),
        params.in_features);
    threadgroup_barrier(mem_flags::mem_threadgroup);

#pragma clang loop unroll(full)
    for (ushort matrix_k = 0; matrix_k < packed_mma_tile; matrix_k += 8) {
      vec<scalar_t, 2> x_fragments[2];
      vec<scalar_t, 2> weight_fragments[2];
      simdgroup_barrier(mem_flags::mem_none);
#pragma clang loop unroll(full)
      for (ushort matrix_m = 0; matrix_m < 2; ++matrix_m) {
        threadgroup const scalar_t *source =
            x_tile + (simd_row + fragment_row + matrix_m * 16) *
                packed_mma_padded_k +
            matrix_k + fragment_col;
        x_fragments[matrix_m] = vec<scalar_t, 2>(source[0], source[1]);
      }
      simdgroup_barrier(mem_flags::mem_none);
#pragma clang loop unroll(full)
      for (ushort matrix_n = 0; matrix_n < 2; ++matrix_n) {
        threadgroup const scalar_t *source =
            weight_tile + (simd_col + fragment_col + matrix_n * 16) *
                packed_mma_padded_k +
            matrix_k + fragment_row;
        weight_fragments[matrix_n] =
            vec<scalar_t, 2>(source[0], source[packed_mma_padded_k]);
      }
      simdgroup_barrier(mem_flags::mem_none);
#pragma clang loop unroll(full)
      for (ushort matrix_m = 0; matrix_m < 2; ++matrix_m) {
#pragma clang loop unroll(full)
        for (ushort matrix_n = 0; matrix_n < 2; ++matrix_n) {
          packed_mma_fragment(
              accumulators[matrix_m][matrix_n],
              x_fragments[matrix_m],
              weight_fragments[matrix_n],
              accumulators[matrix_m][matrix_n]);
        }
      }
    }
  }

#pragma clang loop unroll(full)
  for (ushort matrix_m = 0; matrix_m < 2; ++matrix_m) {
#pragma clang loop unroll(full)
    for (ushort matrix_n = 0; matrix_n < 2; ++matrix_n) {
      const long global_row =
          row_start + long(simd_row + fragment_row + matrix_m * 16);
      const long global_col =
          col_start + long(simd_col + fragment_col + matrix_n * 16);
      float2 values = accumulators[matrix_m][matrix_n];
      if (params.has_bias != 0) {
        values += float2(bias[global_col], bias[global_col + 1]);
      }
      if (global_row < params.rows) {
        out[global_row * params.out_features + global_col] = scalar_t(values[0]);
        out[global_row * params.out_features + global_col + 1] = scalar_t(values[1]);
      }
    }
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
    threadgroup uchar *shared [[threadgroup(0)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    ushort simdgroup_id [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
#define ORBITQUANT_PACKED_MMA_HALF(BITS_VALUE)                              \
  packed_matmul_padded_mma_value<half, BITS_VALUE>(                         \
      out, x, packed_weight_indices, row_norms, centroids, bias, params,    \
      shared, group_id, thread_index, simdgroup_id, lane)
  switch (params.bits) {
    case 2:
      ORBITQUANT_PACKED_MMA_HALF(2);
      break;
    case 3:
      ORBITQUANT_PACKED_MMA_HALF(3);
      break;
    case 4:
      ORBITQUANT_PACKED_MMA_HALF(4);
      break;
    case 6:
      ORBITQUANT_PACKED_MMA_HALF(6);
      break;
  }
#undef ORBITQUANT_PACKED_MMA_HALF
}

kernel void packed_matmul_forward_bfloat16(
    device bfloat *out [[buffer(0)]],
    device const bfloat *x [[buffer(1)]],
    device const uchar *packed_weight_indices [[buffer(2)]],
    device const float *row_norms [[buffer(3)]],
    device const float *centroids [[buffer(4)]],
    device const float *bias [[buffer(5)]],
    constant PackedMatmulParams &params [[buffer(6)]],
    threadgroup uchar *shared [[threadgroup(0)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    ushort simdgroup_id [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
#define ORBITQUANT_PACKED_MMA_BFLOAT16(BITS_VALUE)                          \
  packed_matmul_padded_mma_value<bfloat, BITS_VALUE>(                       \
      out, x, packed_weight_indices, row_norms, centroids, bias, params,    \
      shared, group_id, thread_index, simdgroup_id, lane)
  switch (params.bits) {
    case 2:
      ORBITQUANT_PACKED_MMA_BFLOAT16(2);
      break;
    case 3:
      ORBITQUANT_PACKED_MMA_BFLOAT16(3);
      break;
    case 4:
      ORBITQUANT_PACKED_MMA_BFLOAT16(4);
      break;
    case 6:
      ORBITQUANT_PACKED_MMA_BFLOAT16(6);
      break;
  }
#undef ORBITQUANT_PACKED_MMA_BFLOAT16
}

kernel void packed_matmul_forward_half_scalar(
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

kernel void packed_matmul_forward_bfloat16_scalar(
    device bfloat *out [[buffer(0)]],
    device const bfloat *x [[buffer(1)]],
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

#define ORBITQUANT_PACKED_SMALL_ROWS_KERNEL(NAME, TYPE)                    \
  kernel void NAME(                                                       \
      device TYPE *out [[buffer(0)]],                                     \
      device const TYPE *x [[buffer(1)]],                                 \
      device const uchar *packed_weight_indices [[buffer(2)]],            \
      device const float *row_norms [[buffer(3)]],                         \
      device const float *centroids [[buffer(4)]],                         \
      device const float *bias [[buffer(5)]],                              \
      constant PackedMatmulParams &params [[buffer(6)]],                   \
      uint2 group_id [[threadgroup_position_in_grid]],                     \
      ushort lane [[thread_index_in_simdgroup]]) {                         \
    packed_matmul_small_rows_value(                                       \
        out, x, packed_weight_indices, row_norms, centroids, bias, params, \
        group_id, lane);                                                   \
  }

ORBITQUANT_PACKED_SMALL_ROWS_KERNEL(packed_matmul_forward_float_small_rows, float)
ORBITQUANT_PACKED_SMALL_ROWS_KERNEL(packed_matmul_forward_half_small_rows, half)
ORBITQUANT_PACKED_SMALL_ROWS_KERNEL(packed_matmul_forward_bfloat16_small_rows, bfloat)

#undef ORBITQUANT_PACKED_SMALL_ROWS_KERNEL
