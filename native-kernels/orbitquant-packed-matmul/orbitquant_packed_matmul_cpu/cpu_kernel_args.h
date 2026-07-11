#pragma once

#include "packed_matmul_cpu.h"

#include <cstdint>

namespace orbitquant::cpu {

enum class ActivationIsa : std::uint8_t {
  Portable,
  Neon,
  Avx2,
  Avx512,
};

struct ActivationArgs {
  void *out;
  void const *x;
  std::int64_t const *permutation;
  std::int8_t const *signs;
  float const *centroids;
  float const *boundaries;
  ScalarKind scalar_kind;
  ActivationIsa isa;
  std::int64_t rows;
  std::int64_t dim;
  std::int64_t boundary_count;
  std::int64_t block_size;
  float eps;
  float inv_sqrt_block;
};

void activation_fwht_msvc_avx2(float *values, std::int64_t block_size);

float activation_squared_norm_msvc_avx2(
    void const *data,
    ScalarKind scalar_kind,
    std::int64_t offset,
    std::int64_t dim);

void activation_quantize_lookup_msvc_avx2(
    ActivationArgs const &args,
    float const *scratch,
    std::int64_t output_offset,
    float norm);

struct AdalnArgs {
  void *out;
  void const *x;
  std::uint8_t const *packed_weight;
  float const *scales;
  float const *bias;
  bool has_bias;
  std::int64_t rows;
  std::int64_t out_features;
  std::int64_t in_features;
  std::int64_t group_size;
  std::int64_t num_groups;
  std::int64_t padded_in_features;
};

using AdalnRangeFn = void (*)(
    AdalnArgs const &args,
    std::int64_t out_start,
    std::int64_t out_end);

void packed_adaln_msvc_avx2_range(
    AdalnArgs const &args,
    std::int64_t out_start,
    std::int64_t out_end);

}  // namespace orbitquant::cpu
