#include "packed_matmul_cpu.h"

#include <torch/headeronly/util/BFloat16.h>
#include <torch/headeronly/util/Half.h>

#include <cstddef>

namespace orbitquant::cpu {
namespace {

template <typename scalar_t>
inline float load_scalar(void const *data, std::int64_t offset) {
  return static_cast<float>(static_cast<scalar_t const *>(data)[offset]);
}

template <typename scalar_t>
inline void store_scalar(void *data, std::int64_t offset, float value) {
  static_cast<scalar_t *>(data)[offset] = scalar_t(value);
}

template <>
inline float load_scalar<float>(void const *data, std::int64_t offset) {
  return static_cast<float const *>(data)[offset];
}

template <>
inline void store_scalar<float>(void *data, std::int64_t offset, float value) {
  static_cast<float *>(data)[offset] = value;
}

inline std::uint32_t unpack_index(
    std::uint8_t const *packed,
    std::int64_t value_offset,
    std::int64_t bits) {
  const std::int64_t bit_start = value_offset * bits;
  const std::int64_t byte_index = bit_start >> 3;
  const unsigned bit_offset = static_cast<unsigned>(bit_start & 7);
  std::uint32_t raw = packed[byte_index];
  if (bit_offset + static_cast<unsigned>(bits) > 8) {
    raw |= static_cast<std::uint32_t>(packed[byte_index + 1]) << 8;
  }
  return (raw >> bit_offset) & ((1u << static_cast<unsigned>(bits)) - 1u);
}

template <typename scalar_t>
void packed_matmul_scalar_typed(
    PackedMatmulArgs const &args,
    std::int64_t out_start,
    std::int64_t out_end) {
  for (std::int64_t out_col = out_start; out_col < out_end; ++out_col) {
    const float row_norm = args.row_norms[out_col];
    const std::int64_t weight_row_offset = out_col * args.in_features;
    for (std::int64_t row = 0; row < args.rows; ++row) {
      const std::int64_t input_row_offset = row * args.in_features;
      float accumulator = 0.0f;
      for (std::int64_t k = 0; k < args.in_features; ++k) {
        const std::uint32_t index = unpack_index(
            args.packed_weight_indices,
            weight_row_offset + k,
            args.bits);
        accumulator += load_scalar<scalar_t>(args.x, input_row_offset + k) *
            args.centroids[index];
      }
      accumulator *= row_norm;
      if (args.has_bias) {
        accumulator += args.bias[out_col];
      }
      store_scalar<scalar_t>(
          args.out,
          row * args.out_features + out_col,
          accumulator);
    }
  }
}

}  // namespace

void packed_matmul_scalar_range(
    PackedMatmulArgs const &args,
    std::int64_t out_start,
    std::int64_t out_end) {
  switch (args.scalar_kind) {
    case ScalarKind::Float32:
      packed_matmul_scalar_typed<float>(args, out_start, out_end);
      return;
    case ScalarKind::Float16:
      packed_matmul_scalar_typed<c10::Half>(args, out_start, out_end);
      return;
    case ScalarKind::BFloat16:
      packed_matmul_scalar_typed<c10::BFloat16>(args, out_start, out_end);
      return;
  }
}

}  // namespace orbitquant::cpu
