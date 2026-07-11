#include "packed_matmul_cpu.h"

#if (defined(__x86_64__) || defined(_M_X64)) && !defined(_MSC_VER)
#include <cpuid.h>
#include <immintrin.h>

#include <torch/headeronly/util/BFloat16.h>
#include <torch/headeronly/util/Half.h>

#include <cstring>
#include <cstdint>
#include <type_traits>

#define ORBITQUANT_TARGET_AVX512 \
  __attribute__((target("avx512f,avx512dq,avx512bw,avx512vl,fma,f16c")))
#define ORBITQUANT_TARGET_AVX512_BF16 \
  __attribute__((target( \
      "avx512f,avx512dq,avx512bw,avx512vl,avx512bf16,fma,f16c")))
#define ORBITQUANT_HAS_AVX512_BF16_INTRINSICS 1
#define ORBITQUANT_NOINLINE __attribute__((noinline))
#define ORBITQUANT_ALWAYS_INLINE __attribute__((always_inline))

namespace orbitquant::cpu {
namespace {

ORBITQUANT_TARGET_AVX512 inline float horizontal_sum(__m512 value) {
  return _mm512_reduce_add_ps(value);
}

ORBITQUANT_TARGET_AVX512 inline __m512 load_float16(
    void const *data,
    std::int64_t offset) {
  return _mm512_loadu_ps(static_cast<float const *>(data) + offset);
}

ORBITQUANT_TARGET_AVX512 inline __m512 load_half16(
    void const *data,
    std::int64_t offset) {
  const auto *source = static_cast<std::uint16_t const *>(data) + offset;
  const __m256i packed =
      _mm256_loadu_si256(reinterpret_cast<__m256i const *>(source));
  return _mm512_cvtph_ps(packed);
}

ORBITQUANT_TARGET_AVX512 inline __m512 load_bfloat16(
    void const *data,
    std::int64_t offset) {
  const auto *source = static_cast<std::uint16_t const *>(data) + offset;
  const __m256i packed =
      _mm256_loadu_si256(reinterpret_cast<__m256i const *>(source));
  const __m512i widened = _mm512_cvtepu16_epi32(packed);
  return _mm512_castsi512_ps(_mm512_slli_epi32(widened, 16));
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
    __m512 (*load16)(void const *, std::int64_t),
    int row_tile>
ORBITQUANT_TARGET_AVX512 inline void packed_matmul_avx512_w4_rows(
    PackedMatmulArgs const &args,
    std::uint8_t const *packed_row,
    std::int64_t out_col,
    std::int64_t row_start) {
  __m512 accumulators[row_tile];
#pragma clang loop unroll(full)
  for (int row = 0; row < row_tile; ++row) {
    accumulators[row] = _mm512_setzero_ps();
  }
  const __m512 centroid_lut = _mm512_loadu_ps(args.centroids);
  const __m128i nibble_mask = _mm_set1_epi8(15);

  std::int64_t k = 0;
  for (; k + 16 <= args.in_features; k += 16) {
    const std::int64_t byte_offset = k / 2;
    std::int64_t packed;
    std::memcpy(&packed, packed_row + byte_offset, sizeof(packed));
    const __m128i bytes = _mm_cvtsi64_si128(packed);
    const __m128i low = _mm_and_si128(bytes, nibble_mask);
    const __m128i high = _mm_and_si128(
        _mm_srli_epi16(bytes, 4),
        nibble_mask);
    const __m512i indices =
        _mm512_cvtepu8_epi32(_mm_unpacklo_epi8(low, high));
    const __m512 weight = _mm512_permutexvar_ps(indices, centroid_lut);
#pragma clang loop unroll(full)
    for (int row = 0; row < row_tile; ++row) {
      const std::int64_t input_offset =
          (row_start + row) * args.in_features + k;
      accumulators[row] = _mm512_fmadd_ps(
          load16(args.x, input_offset),
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

template <typename scalar_t, __m512 (*load16)(void const *, std::int64_t)>
ORBITQUANT_TARGET_AVX512 ORBITQUANT_NOINLINE void packed_matmul_avx512_w4_typed(
    PackedMatmulArgs const &args,
    std::int64_t out_start,
    std::int64_t out_end) {
  constexpr int kPrimaryRowTile = 8;
  const std::int64_t packed_row_bytes = args.in_features / 2;
  for (std::int64_t out_col = out_start; out_col < out_end; ++out_col) {
    const auto *packed_row =
        args.packed_weight_indices + out_col * packed_row_bytes;
    std::int64_t row = 0;
    for (; row + kPrimaryRowTile <= args.rows; row += kPrimaryRowTile) {
      packed_matmul_avx512_w4_rows<scalar_t, load16, kPrimaryRowTile>(
          args, packed_row, out_col, row);
    }
    if (row + 8 <= args.rows) {
      packed_matmul_avx512_w4_rows<scalar_t, load16, 8>(
          args, packed_row, out_col, row);
      row += 8;
    }
    if (row + 4 <= args.rows) {
      packed_matmul_avx512_w4_rows<scalar_t, load16, 4>(
          args, packed_row, out_col, row);
      row += 4;
    }
    switch (args.rows - row) {
      case 3:
        packed_matmul_avx512_w4_rows<scalar_t, load16, 3>(
            args, packed_row, out_col, row);
        break;
      case 2:
        packed_matmul_avx512_w4_rows<scalar_t, load16, 2>(
            args, packed_row, out_col, row);
        break;
      case 1:
        packed_matmul_avx512_w4_rows<scalar_t, load16, 1>(
            args, packed_row, out_col, row);
        break;
      default:
        break;
    }
  }
}

#if defined(ORBITQUANT_HAS_AVX512_BF16_INTRINSICS)
template <int row_tile>
ORBITQUANT_TARGET_AVX512_BF16 ORBITQUANT_ALWAYS_INLINE inline void
accumulate_bf16_w4_chunk(
    PackedMatmulArgs const &args,
    std::uint8_t const *packed_row,
    std::int64_t row_start,
    std::int64_t k,
    __m512i centroid_lut_words,
    __m128i nibble_mask,
    __m512 (&accumulators)[row_tile]) {
  const std::int64_t byte_offset = k / 2;
  const __m128i bytes = _mm_loadu_si128(
      reinterpret_cast<__m128i const *>(packed_row + byte_offset));
  const __m128i low = _mm_and_si128(bytes, nibble_mask);
  const __m128i high = _mm_and_si128(
      _mm_srli_epi16(bytes, 4),
      nibble_mask);
  const __m256i packed_indices = _mm256_set_m128i(
      _mm_unpackhi_epi8(low, high),
      _mm_unpacklo_epi8(low, high));
  const __m512i indices = _mm512_cvtepu8_epi16(packed_indices);
  const __m512bh weights = (__m512bh)_mm512_permutexvar_epi16(
      indices,
      centroid_lut_words);
#pragma clang loop unroll(full)
  for (int row = 0; row < row_tile; ++row) {
    const std::int64_t input_offset =
        (row_start + row) * args.in_features + k;
    const __m512bh activations = (__m512bh)_mm512_loadu_si512(
        static_cast<std::uint16_t const *>(args.x) + input_offset);
    accumulators[row] =
        _mm512_dpbf16_ps(accumulators[row], activations, weights);
  }
}

template <int row_tile, bool aligned_k, bool unroll_k>
ORBITQUANT_TARGET_AVX512_BF16 inline void packed_matmul_avx512_bf16_w4_rows(
    PackedMatmulArgs const &args,
    std::uint8_t const *packed_row,
    std::int64_t out_col,
    std::int64_t row_start) {
  __m512 accumulators[row_tile];
#pragma clang loop unroll(full)
  for (int row = 0; row < row_tile; ++row) {
    accumulators[row] = _mm512_setzero_ps();
  }
  const float row_norm = args.row_norms[out_col];
  const __m512 centroid_lut = _mm512_mul_ps(
      _mm512_loadu_ps(args.centroids),
      _mm512_set1_ps(row_norm));
  const __m512i centroid_lut_words = (__m512i)_mm512_cvtne2ps_pbh(
      centroid_lut,
      centroid_lut);
  const __m128i nibble_mask = _mm_set1_epi8(15);

  std::int64_t k = 0;
  if constexpr (unroll_k) {
#if defined(__clang__)
#pragma clang loop unroll_count(2)
#elif defined(__GNUC__)
#pragma GCC unroll 2
#endif
    for (; k + 32 <= args.in_features; k += 32) {
      accumulate_bf16_w4_chunk<row_tile>(
          args,
          packed_row,
          row_start,
          k,
          centroid_lut_words,
          nibble_mask,
          accumulators);
    }
  } else {
    for (; k + 32 <= args.in_features; k += 32) {
      accumulate_bf16_w4_chunk<row_tile>(
          args,
          packed_row,
          row_start,
          k,
          centroid_lut_words,
          nibble_mask,
          accumulators);
    }
  }

#pragma clang loop unroll(full)
  for (int row = 0; row < row_tile; ++row) {
    const std::int64_t input_row_offset =
        (row_start + row) * args.in_features;
    float accumulator = horizontal_sum(accumulators[row]);
    if constexpr (!aligned_k) {
      for (std::int64_t tail = k; tail < args.in_features; ++tail) {
        const std::uint8_t packed = packed_row[tail / 2];
        const std::uint8_t index =
            (tail & 1) == 0 ? packed & 15u : (packed >> 4) & 15u;
        const float activation = static_cast<float>(
            static_cast<c10::BFloat16 const *>(args.x)[input_row_offset + tail]);
        const float weight = static_cast<float>(
            c10::BFloat16(args.centroids[index] * row_norm));
        accumulator += activation * weight;
      }
    }
    if (args.has_bias) {
      accumulator += args.bias[out_col];
    }
    store_value<c10::BFloat16>(
        args.out,
        (row_start + row) * args.out_features + out_col,
        accumulator);
  }
}

template <int primary_row_tile, bool aligned_k, bool unroll_k>
ORBITQUANT_TARGET_AVX512_BF16 ORBITQUANT_NOINLINE void
packed_matmul_avx512_bf16_w4_typed(
    PackedMatmulArgs const &args,
    std::int64_t out_start,
    std::int64_t out_end) {
  static_assert(primary_row_tile == 4 || primary_row_tile == 8);
  const std::int64_t packed_row_bytes = args.in_features / 2;
  for (std::int64_t out_col = out_start; out_col < out_end; ++out_col) {
    const auto *packed_row =
        args.packed_weight_indices + out_col * packed_row_bytes;
    std::int64_t row = 0;
    for (; row + primary_row_tile <= args.rows; row += primary_row_tile) {
      packed_matmul_avx512_bf16_w4_rows<primary_row_tile, aligned_k, unroll_k>(
          args, packed_row, out_col, row);
    }
    if (row + 4 <= args.rows) {
      packed_matmul_avx512_bf16_w4_rows<4, aligned_k, unroll_k>(
          args, packed_row, out_col, row);
      row += 4;
    }
    switch (args.rows - row) {
      case 3:
        packed_matmul_avx512_bf16_w4_rows<3, aligned_k, unroll_k>(
            args, packed_row, out_col, row);
        break;
      case 2:
        packed_matmul_avx512_bf16_w4_rows<2, aligned_k, unroll_k>(
            args, packed_row, out_col, row);
        break;
      case 1:
        packed_matmul_avx512_bf16_w4_rows<1, aligned_k, unroll_k>(
            args, packed_row, out_col, row);
        break;
      default:
        break;
    }
  }
}
#endif

bool runtime_has_avx512() {
#if defined(_MSC_VER)
  int registers[4]{};
  __cpuid(registers, 1);
  const bool osxsave = (registers[2] & (1 << 27)) != 0;
  const bool avx = (registers[2] & (1 << 28)) != 0;
  const bool fma = (registers[2] & (1 << 12)) != 0;
  const bool f16c = (registers[2] & (1 << 29)) != 0;
  if (!osxsave || !avx || !fma || !f16c || (_xgetbv(0) & 0xE6) != 0xE6) {
    return false;
  }
  __cpuidex(registers, 7, 0);
  constexpr unsigned required_ebx =
      (1u << 16) | (1u << 17) | (1u << 30) | (1u << 31);
  return (static_cast<unsigned>(registers[1]) & required_ebx) == required_ebx;
#else
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx512f") &&
      __builtin_cpu_supports("avx512dq") &&
      __builtin_cpu_supports("avx512bw") &&
      __builtin_cpu_supports("avx512vl") &&
      __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c");
#endif
}

bool runtime_has_avx512_bf16() {
#if defined(ORBITQUANT_HAS_AVX512_BF16_INTRINSICS)
  __builtin_cpu_init();
  return runtime_has_avx512() && __builtin_cpu_supports("avx512bf16");
#else
  return false;
#endif
}

bool runtime_has_verified_amd_bf16_tuning() {
  static const bool available = [] {
    unsigned int eax = 0;
    unsigned int ebx = 0;
    unsigned int ecx = 0;
    unsigned int edx = 0;
    // CPUID vendor registers spell "AuthenticAMD" in EBX, EDX, ECX order.
    if (!__get_cpuid(0, &eax, &ebx, &ecx, &edx) ||
        ebx != 0x68747541u || edx != 0x69746e65u || ecx != 0x444d4163u ||
        !__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
      return false;
    }
    const unsigned int base_family = (eax >> 8) & 0xfu;
    const unsigned int base_model = (eax >> 4) & 0xfu;
    const unsigned int family = base_family == 0xfu
        ? base_family + ((eax >> 20) & 0xffu)
        : base_family;
    const unsigned int model = (base_family == 0x6u || base_family == 0xfu)
        ? base_model + (((eax >> 16) & 0xfu) << 4)
        : base_model;
    // Family 19h/model 61h is the measured EPYC 4564P configuration.
    return family == 0x19u && model == 0x61u;
  }();
  return available;
}

}  // namespace

bool packed_matmul_x86_avx512_available() {
  static const bool available = runtime_has_avx512();
  return available;
}

void packed_matmul_x86_avx512_range(
    PackedMatmulArgs const &args,
    std::int64_t out_start,
    std::int64_t out_end) {
  if (!packed_matmul_x86_avx512_available() || args.bits != 4 ||
      args.in_features % 2 != 0) {
    packed_matmul_scalar_range(args, out_start, out_end);
    return;
  }
  switch (args.scalar_kind) {
    case ScalarKind::Float32:
      packed_matmul_avx512_w4_typed<float, load_float16>(
          args, out_start, out_end);
      return;
    case ScalarKind::Float16:
      packed_matmul_avx512_w4_typed<c10::Half, load_half16>(
          args, out_start, out_end);
      return;
    case ScalarKind::BFloat16:
#if defined(ORBITQUANT_HAS_AVX512_BF16_INTRINSICS)
      if (runtime_has_avx512_bf16()) {
        const bool tuned_dimension = args.in_features == 1536 ||
            args.in_features == 1920 || args.in_features == 3072;
        const bool use_tuned_shape =
            runtime_has_verified_amd_bf16_tuning() && args.rows >= 16 &&
            tuned_dimension;
        if (args.in_features % 32 == 0) {
          if (use_tuned_shape) {
            packed_matmul_avx512_bf16_w4_typed<4, true, true>(
                args, out_start, out_end);
          } else {
            packed_matmul_avx512_bf16_w4_typed<8, true, false>(
                args, out_start, out_end);
          }
        } else {
          packed_matmul_avx512_bf16_w4_typed<8, false, false>(
              args, out_start, out_end);
        }
        return;
      }
#endif
      packed_matmul_avx512_w4_typed<c10::BFloat16, load_bfloat16>(
          args, out_start, out_end);
      return;
  }
}

}  // namespace orbitquant::cpu

#else

namespace orbitquant::cpu {

bool packed_matmul_x86_avx512_available() {
  return false;
}

void packed_matmul_x86_avx512_range(
    PackedMatmulArgs const &args,
    std::int64_t out_start,
    std::int64_t out_end) {
  packed_matmul_scalar_range(args, out_start, out_end);
}

}  // namespace orbitquant::cpu

#endif
