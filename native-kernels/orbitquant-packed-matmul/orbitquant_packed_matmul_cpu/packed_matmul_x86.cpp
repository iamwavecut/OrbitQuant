#include "packed_matmul_cpu.h"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif

#include <torch/headeronly/util/BFloat16.h>
#include <torch/headeronly/util/Half.h>

#include <cstring>
#include <cstdint>
#include <type_traits>

#if defined(_MSC_VER)
#define ORBITQUANT_TARGET_AVX2
#define ORBITQUANT_NOINLINE __declspec(noinline)
#else
#define ORBITQUANT_TARGET_AVX2 __attribute__((target("avx2,fma,f16c")))
#define ORBITQUANT_NOINLINE __attribute__((noinline))
#endif

namespace orbitquant::cpu {
namespace {

ORBITQUANT_TARGET_AVX2 inline float horizontal_sum(__m256 value) {
  const __m128 halves =
      _mm_add_ps(_mm256_castps256_ps128(value), _mm256_extractf128_ps(value, 1));
  const __m128 pairs = _mm_hadd_ps(halves, halves);
  return _mm_cvtss_f32(_mm_hadd_ps(pairs, pairs));
}

ORBITQUANT_TARGET_AVX2 inline __m256 load_float8(
    void const *data,
    std::int64_t offset) {
  return _mm256_loadu_ps(static_cast<float const *>(data) + offset);
}

ORBITQUANT_TARGET_AVX2 inline __m256 load_half8(
    void const *data,
    std::int64_t offset) {
  const auto *source = static_cast<std::uint16_t const *>(data) + offset;
  const __m128i packed =
      _mm_loadu_si128(reinterpret_cast<__m128i const *>(source));
  return _mm256_cvtph_ps(packed);
}

ORBITQUANT_TARGET_AVX2 inline __m256 load_bfloat8(
    void const *data,
    std::int64_t offset) {
  const auto *source = static_cast<std::uint16_t const *>(data) + offset;
  const __m128i packed =
      _mm_loadu_si128(reinterpret_cast<__m128i const *>(source));
  const __m256i widened = _mm256_cvtepu16_epi32(packed);
  return _mm256_castsi256_ps(_mm256_slli_epi32(widened, 16));
}

template <typename scalar_t>
inline void store_value(void *data, std::int64_t offset, float value) {
  static_cast<scalar_t *>(data)[offset] = scalar_t(value);
}

template <>
inline void store_value<float>(void *data, std::int64_t offset, float value) {
  static_cast<float *>(data)[offset] = value;
}

template <
    typename scalar_t,
    __m256 (*load8)(void const *, std::int64_t),
    int row_tile>
ORBITQUANT_TARGET_AVX2 inline void packed_matmul_avx2_w4_rows(
    PackedMatmulArgs const &args,
    std::uint8_t const *packed_row,
    std::int64_t out_col,
    std::int64_t row_start) {
  __m256 accumulators[row_tile];
#pragma clang loop unroll(full)
  for (int row = 0; row < row_tile; ++row) {
    accumulators[row] = _mm256_setzero_ps();
  }
  const __m256 centroid_lut_low = _mm256_loadu_ps(args.centroids);
  const __m256 centroid_lut_high = _mm256_loadu_ps(args.centroids + 8);
  const __m128i nibble_mask = _mm_set1_epi8(15);
  const __m256i low_table_limit = _mm256_set1_epi32(7);

  std::int64_t k = 0;
  for (; k + 8 <= args.in_features; k += 8) {
    const std::int64_t byte_offset = k / 2;
    std::int32_t packed;
    std::memcpy(&packed, packed_row + byte_offset, sizeof(packed));
    const __m128i bytes = _mm_cvtsi32_si128(packed);
    const __m128i low = _mm_and_si128(bytes, nibble_mask);
    const __m128i high = _mm_and_si128(
        _mm_srli_epi16(bytes, 4),
        nibble_mask);
    const __m256i indices =
        _mm256_cvtepu8_epi32(_mm_unpacklo_epi8(low, high));
    const __m256 low_weights =
        _mm256_permutevar8x32_ps(centroid_lut_low, indices);
    const __m256 high_weights =
        _mm256_permutevar8x32_ps(centroid_lut_high, indices);
    const __m256 weight = _mm256_blendv_ps(
        low_weights,
        high_weights,
        _mm256_castsi256_ps(_mm256_cmpgt_epi32(indices, low_table_limit)));
#pragma clang loop unroll(full)
    for (int row = 0; row < row_tile; ++row) {
      const std::int64_t input_offset =
          (row_start + row) * args.in_features + k;
      accumulators[row] = _mm256_fmadd_ps(
          load8(args.x, input_offset),
          weight,
          accumulators[row]);
    }
  }

  const float row_norm = args.row_norms[out_col];
#pragma clang loop unroll(full)
  for (int row = 0; row < row_tile; ++row) {
    const std::int64_t input_row_offset =
        (row_start + row) * args.in_features;
    float accumulator = horizontal_sum(accumulators[row]);
    for (std::int64_t tail = k; tail < args.in_features; ++tail) {
      const std::uint8_t packed = packed_row[tail / 2];
      const std::uint8_t index =
          (tail & 1) == 0 ? packed & 15u : (packed >> 4) & 15u;
      if constexpr (std::is_same_v<scalar_t, float>) {
        accumulator +=
            static_cast<float const *>(args.x)[input_row_offset + tail] *
            args.centroids[index];
      } else {
        accumulator += static_cast<float>(
                           static_cast<scalar_t const *>(
                               args.x)[input_row_offset + tail]) *
            args.centroids[index];
      }
    }
    accumulator *= row_norm;
    if (args.has_bias) {
      accumulator += args.bias[out_col];
    }
    store_value<scalar_t>(
        args.out,
        (row_start + row) * args.out_features + out_col,
        accumulator);
  }
}

template <typename scalar_t>
bool use_verified_amd_cezanne_row_tile(PackedMatmulArgs const &args) {
  if constexpr (!std::is_same_v<scalar_t, c10::BFloat16>) {
    return false;
  }
  const bool tuned_dimension = args.in_features == 1536 ||
      args.in_features == 1920 || args.in_features == 3072;
  if (args.rows < 16 || !tuned_dimension) {
    return false;
  }
  static const bool verified_cpu = [] {
    unsigned int eax = 0;
    unsigned int ebx = 0;
    unsigned int ecx = 0;
    unsigned int edx = 0;
#if defined(_MSC_VER)
    int registers[4]{};
    __cpuid(registers, 0);
    eax = static_cast<unsigned int>(registers[0]);
    ebx = static_cast<unsigned int>(registers[1]);
    ecx = static_cast<unsigned int>(registers[2]);
    edx = static_cast<unsigned int>(registers[3]);
    if (ebx != 0x68747541u || edx != 0x69746e65u ||
        ecx != 0x444d4163u) {
      return false;
    }
    __cpuid(registers, 1);
    eax = static_cast<unsigned int>(registers[0]);
#else
    // CPUID vendor registers spell "AuthenticAMD" in EBX, EDX, ECX order.
    if (!__get_cpuid(0, &eax, &ebx, &ecx, &edx) ||
        ebx != 0x68747541u || edx != 0x69746e65u || ecx != 0x444d4163u ||
        !__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
      return false;
    }
#endif
    const unsigned int base_family = (eax >> 8) & 0xfu;
    const unsigned int base_model = (eax >> 4) & 0xfu;
    const unsigned int family = base_family == 0xfu
        ? base_family + ((eax >> 20) & 0xffu)
        : base_family;
    const unsigned int model = (base_family == 0x6u || base_family == 0xfu)
        ? base_model + (((eax >> 16) & 0xfu) << 4)
        : base_model;
    // Family 19h/model 50h is the measured Ryzen 5 5600G configuration.
    return family == 0x19u && model == 0x50u;
  }();
  return verified_cpu;
}

template <
    typename scalar_t,
    __m256 (*load8)(void const *, std::int64_t),
    int primary_row_tile>
ORBITQUANT_TARGET_AVX2 ORBITQUANT_NOINLINE void packed_matmul_avx2_w4_typed(
    PackedMatmulArgs const &args,
    std::int64_t out_start,
    std::int64_t out_end) {
  static_assert(primary_row_tile == 8 || primary_row_tile == 16);
  const std::int64_t packed_row_bytes = args.in_features / 2;
  for (std::int64_t out_col = out_start; out_col < out_end; ++out_col) {
    const auto *packed_row =
        args.packed_weight_indices + out_col * packed_row_bytes;
    std::int64_t row = 0;
    for (; row + primary_row_tile <= args.rows; row += primary_row_tile) {
      packed_matmul_avx2_w4_rows<scalar_t, load8, primary_row_tile>(
          args, packed_row, out_col, row);
    }
    if (row + 8 <= args.rows) {
      packed_matmul_avx2_w4_rows<scalar_t, load8, 8>(
          args, packed_row, out_col, row);
      row += 8;
    }
    if (row + 4 <= args.rows) {
      packed_matmul_avx2_w4_rows<scalar_t, load8, 4>(
          args, packed_row, out_col, row);
      row += 4;
    }
    switch (args.rows - row) {
      case 3:
        packed_matmul_avx2_w4_rows<scalar_t, load8, 3>(
            args, packed_row, out_col, row);
        break;
      case 2:
        packed_matmul_avx2_w4_rows<scalar_t, load8, 2>(
            args, packed_row, out_col, row);
        break;
      case 1:
        packed_matmul_avx2_w4_rows<scalar_t, load8, 1>(
            args, packed_row, out_col, row);
        break;
      default:
        break;
    }
  }
}

}  // namespace

void packed_matmul_x86_avx2_range(
    PackedMatmulArgs const &args,
    std::int64_t out_start,
    std::int64_t out_end) {
  if (!packed_matmul_x86_avx2_available() || args.bits != 4 ||
      args.in_features % 2 != 0) {
    packed_matmul_scalar_range(args, out_start, out_end);
    return;
  }
  switch (args.scalar_kind) {
    case ScalarKind::Float32:
      packed_matmul_avx2_w4_typed<float, load_float8, 8>(
          args, out_start, out_end);
      return;
    case ScalarKind::Float16:
      packed_matmul_avx2_w4_typed<c10::Half, load_half8, 8>(
          args, out_start, out_end);
      return;
    case ScalarKind::BFloat16:
      if (use_verified_amd_cezanne_row_tile<c10::BFloat16>(args)) {
        packed_matmul_avx2_w4_typed<c10::BFloat16, load_bfloat8, 16>(
            args, out_start, out_end);
      } else {
        packed_matmul_avx2_w4_typed<c10::BFloat16, load_bfloat8, 8>(
            args, out_start, out_end);
      }
      return;
  }
}

}  // namespace orbitquant::cpu

#else

namespace orbitquant::cpu {

void packed_matmul_x86_avx2_range(
    PackedMatmulArgs const &args,
    std::int64_t out_start,
    std::int64_t out_end) {
  packed_matmul_scalar_range(args, out_start, out_end);
}

}  // namespace orbitquant::cpu

#endif
