#pragma once

#include <cstdint>

namespace orbitquant::cpu {

enum class ScalarKind : std::uint8_t {
  Float32,
  Float16,
  BFloat16,
};

struct PackedMatmulArgs {
  void *out;
  void const *x;
  std::uint8_t const *packed_weight_indices;
  float const *row_norms;
  float const *centroids;
  float const *bias;
  bool has_bias;
  ScalarKind scalar_kind;
  std::int64_t rows;
  std::int64_t out_features;
  std::int64_t in_features;
  std::int64_t bits;
};

using PackedMatmulRangeFn = void (*)(
    PackedMatmulArgs const &args,
    std::int64_t out_start,
    std::int64_t out_end);

void packed_matmul_scalar_range(
    PackedMatmulArgs const &args,
    std::int64_t out_start,
    std::int64_t out_end);

bool packed_matmul_neon_available();

void packed_matmul_neon_range(
    PackedMatmulArgs const &args,
    std::int64_t out_start,
    std::int64_t out_end);

bool packed_matmul_x86_avx2_available();

void packed_matmul_x86_avx2_range(
    PackedMatmulArgs const &args,
    std::int64_t out_start,
    std::int64_t out_end);

bool packed_matmul_x86_avx512_available();

void packed_matmul_x86_avx512_range(
    PackedMatmulArgs const &args,
    std::int64_t out_start,
    std::int64_t out_end);

}  // namespace orbitquant::cpu
