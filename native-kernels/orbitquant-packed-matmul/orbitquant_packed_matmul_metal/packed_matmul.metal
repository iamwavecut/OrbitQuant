#include <metal_stdlib>

using namespace metal;

struct PackedMatmulParams {
  long rows;
  long out_features;
  long in_features;
  long bits;
  int has_bias;
};

template <typename scalar_t>
inline void packed_matmul_value(
    device scalar_t *out,
    device const scalar_t *x,
    device const uchar *packed_weight_indices,
    device const float *row_norms,
    device const float *centroids,
    device const float *bias,
    constant PackedMatmulParams &params,
    uint2 tid) {
  const long col = tid.x;
  const long row = tid.y;
  if (row >= params.rows || col >= params.out_features) {
    return;
  }

  const uint mask = (1u << uint(params.bits)) - 1u;
  float acc = 0.0f;
  const float row_norm = row_norms[col];
  for (long k = 0; k < params.in_features; ++k) {
    const long value_offset = col * params.in_features + k;
    const long bit_start = value_offset * params.bits;
    const long byte_index = bit_start >> 3;
    const long bit_offset = bit_start & 7;
    uint raw = packed_weight_indices[byte_index];
    if (bit_offset + params.bits > 8) {
      raw |= uint(packed_weight_indices[byte_index + 1]) << 8;
    }
    const uint index = (raw >> uint(bit_offset)) & mask;
    acc += float(x[row * params.in_features + k]) * centroids[index];
  }
  float value = acc * row_norm;
  if (params.has_bias != 0) {
    value += bias[col];
  }
  out[row * params.out_features + col] = scalar_t(value);
}

kernel void packed_matmul_forward_float(
    device float *out [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device const uchar *packed_weight_indices [[buffer(2)]],
    device const float *row_norms [[buffer(3)]],
    device const float *centroids [[buffer(4)]],
    device const float *bias [[buffer(5)]],
    constant PackedMatmulParams &params [[buffer(6)]],
    uint2 tid [[thread_position_in_grid]]) {
  packed_matmul_value(out, x, packed_weight_indices, row_norms, centroids, bias, params, tid);
}

kernel void packed_matmul_forward_half(
    device half *out [[buffer(0)]],
    device const half *x [[buffer(1)]],
    device const uchar *packed_weight_indices [[buffer(2)]],
    device const float *row_norms [[buffer(3)]],
    device const float *centroids [[buffer(4)]],
    device const float *bias [[buffer(5)]],
    constant PackedMatmulParams &params [[buffer(6)]],
    uint2 tid [[thread_position_in_grid]]) {
  packed_matmul_value(out, x, packed_weight_indices, row_norms, centroids, bias, params, tid);
}
