#include "cpu_threads.h"
#include "cpu_kernel_args.h"
#include "packed_matmul_cpu.h"
#include "../torch-ext/torch_binding.h"

#include <torch/headeronly/core/DeviceType.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>
#include <torch/headeronly/util/BFloat16.h>

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#endif

#if (defined(__x86_64__) || defined(_M_X64)) && !defined(_MSC_VER)
#include <immintrin.h>
#define ORBITQUANT_TARGET_AVX2 __attribute__((target("avx2,fma,f16c")))
#define ORBITQUANT_TARGET_AVX512 \
  __attribute__((target("avx512f,avx512dq,avx512bw,avx512vl,fma,f16c")))
#define ORBITQUANT_TARGET_AVX512_BF16 \
  __attribute__((target( \
      "avx512f,avx512dq,avx512bw,avx512vl,avx512bf16,fma,f16c")))
#define ORBITQUANT_HAS_AVX512_BF16_INTRINSICS 1
#define ORBITQUANT_NOINLINE __attribute__((noinline))
#endif

namespace {

using orbitquant::cpu::AdalnArgs;
using orbitquant::cpu::AdalnRangeFn;

inline float load_bfloat(
    void const *data,
    std::int64_t offset) {
  return static_cast<float>(
      static_cast<c10::BFloat16 const *>(data)[offset]);
}

inline void store_bfloat(
    void *data,
    std::int64_t offset,
    float value) {
  static_cast<c10::BFloat16 *>(data)[offset] = c10::BFloat16(value);
}

inline std::uint8_t unpack_adaln_index(
    std::uint8_t const *packed,
    std::int64_t flat_index) {
  const std::uint8_t byte = packed[flat_index / 2];
  return (flat_index & 1) == 0 ? byte & 15u : (byte >> 4) & 15u;
}

inline float dequantized_adaln_value(
    std::uint8_t index,
    float scale) {
  return static_cast<float>(
      c10::BFloat16((static_cast<int>(index) - 8) * scale));
}

inline void fill_group_lut(float *lut, float scale) {
  for (int index = 0; index < 16; ++index) {
    lut[index] = dequantized_adaln_value(
        static_cast<std::uint8_t>(index),
        scale);
  }
}

void packed_adaln_scalar_range(
    AdalnArgs const &args,
    std::int64_t out_start,
    std::int64_t out_end) {
  for (std::int64_t out_col = out_start; out_col < out_end; ++out_col) {
    for (std::int64_t row = 0; row < args.rows; ++row) {
      float accumulator = 0.0f;
      const std::int64_t input_offset = row * args.in_features;
      for (std::int64_t k = 0; k < args.in_features; ++k) {
        const std::int64_t weight_offset =
            out_col * args.padded_in_features + k;
        const std::uint8_t index =
            unpack_adaln_index(args.packed_weight, weight_offset);
        const float scale =
            args.scales[out_col * args.num_groups + k / args.group_size];
        accumulator += load_bfloat(args.x, input_offset + k) *
            dequantized_adaln_value(index, scale);
      }
      if (args.has_bias) {
        accumulator += args.bias[out_col];
      }
      store_bfloat(
          args.out,
          row * args.out_features + out_col,
          accumulator);
    }
  }
}

#if defined(__aarch64__) || defined(_M_ARM64)
inline float32x4_t load_bfloat4(
    void const *data,
    std::int64_t offset) {
  const uint16x4_t raw = vld1_u16(
      static_cast<std::uint16_t const *>(data) + offset);
  return vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(raw), 16));
}

template <int row_tile>
inline void packed_adaln_neon_rows(
    AdalnArgs const &args,
    std::uint8_t const *packed_row,
    float const *scale_row,
    std::int64_t out_col,
    std::int64_t row_start) {
  float32x4_t accumulators[row_tile];
  float tails[row_tile]{};
#pragma clang loop unroll(full)
  for (int row = 0; row < row_tile; ++row) {
    accumulators[row] = vdupq_n_f32(0.0f);
  }

  for (std::int64_t group = 0; group < args.num_groups; ++group) {
    float lut[16];
    fill_group_lut(lut, scale_row[group]);
    const std::int64_t group_start = group * args.group_size;
    const std::int64_t group_end = std::min(
        args.in_features,
        group_start + args.group_size);
    std::int64_t k = group_start;
    for (; k + 4 <= group_end; k += 4) {
      float weights[4];
#pragma clang loop unroll(full)
      for (int lane = 0; lane < 4; ++lane) {
        weights[lane] = lut[unpack_adaln_index(packed_row, k + lane)];
      }
      const float32x4_t weight = vld1q_f32(weights);
#pragma clang loop unroll(full)
      for (int row = 0; row < row_tile; ++row) {
        const std::int64_t input_offset =
            (row_start + row) * args.in_features + k;
        accumulators[row] = vfmaq_f32(
            accumulators[row],
            load_bfloat4(args.x, input_offset),
            weight);
      }
    }
    for (; k < group_end; ++k) {
      const float weight = lut[unpack_adaln_index(packed_row, k)];
#pragma clang loop unroll(full)
      for (int row = 0; row < row_tile; ++row) {
        tails[row] += load_bfloat(
            args.x,
            (row_start + row) * args.in_features + k) * weight;
      }
    }
  }

#pragma clang loop unroll(full)
  for (int row = 0; row < row_tile; ++row) {
    float accumulator = vaddvq_f32(accumulators[row]) + tails[row];
    if (args.has_bias) {
      accumulator += args.bias[out_col];
    }
    store_bfloat(
        args.out,
        (row_start + row) * args.out_features + out_col,
        accumulator);
  }
}

void packed_adaln_neon_range(
    AdalnArgs const &args,
    std::int64_t out_start,
    std::int64_t out_end) {
  if (args.group_size % 4 != 0 || args.padded_in_features % 2 != 0) {
    packed_adaln_scalar_range(args, out_start, out_end);
    return;
  }
  constexpr int kPrimaryRowTile = 8;
  const std::int64_t packed_row_bytes = args.padded_in_features / 2;
  for (std::int64_t out_col = out_start; out_col < out_end; ++out_col) {
    const auto *packed_row =
        args.packed_weight + out_col * packed_row_bytes;
    const auto *scale_row =
        args.scales + out_col * args.num_groups;
    std::int64_t row = 0;
    for (; row + kPrimaryRowTile <= args.rows; row += kPrimaryRowTile) {
      packed_adaln_neon_rows<kPrimaryRowTile>(
          args, packed_row, scale_row, out_col, row);
    }
    if (row + 4 <= args.rows) {
      packed_adaln_neon_rows<4>(
          args, packed_row, scale_row, out_col, row);
      row += 4;
    }
    switch (args.rows - row) {
      case 3:
        packed_adaln_neon_rows<3>(
            args, packed_row, scale_row, out_col, row);
        break;
      case 2:
        packed_adaln_neon_rows<2>(
            args, packed_row, scale_row, out_col, row);
        break;
      case 1:
        packed_adaln_neon_rows<1>(
            args, packed_row, scale_row, out_col, row);
        break;
      default:
        break;
    }
  }
}
#else
void packed_adaln_neon_range(
    AdalnArgs const &args,
    std::int64_t out_start,
    std::int64_t out_end) {
  packed_adaln_scalar_range(args, out_start, out_end);
}
#endif

#if (defined(__x86_64__) || defined(_M_X64)) && !defined(_MSC_VER)
ORBITQUANT_TARGET_AVX2 inline __m256 load_bfloat8(
    void const *data,
    std::int64_t offset) {
  const auto *source =
      static_cast<std::uint16_t const *>(data) + offset;
  const __m128i packed =
      _mm_loadu_si128(reinterpret_cast<__m128i const *>(source));
  return _mm256_castsi256_ps(
      _mm256_slli_epi32(_mm256_cvtepu16_epi32(packed), 16));
}

ORBITQUANT_TARGET_AVX512 inline __m512 load_bfloat16(
    void const *data,
    std::int64_t offset) {
  const auto *source =
      static_cast<std::uint16_t const *>(data) + offset;
  const __m256i packed =
      _mm256_loadu_si256(reinterpret_cast<__m256i const *>(source));
  return _mm512_castsi512_ps(
      _mm512_slli_epi32(_mm512_cvtepu16_epi32(packed), 16));
}

ORBITQUANT_TARGET_AVX2 inline float horizontal_sum(__m256 value) {
  const __m128 halves = _mm_add_ps(
      _mm256_castps256_ps128(value),
      _mm256_extractf128_ps(value, 1));
  const __m128 pairs = _mm_hadd_ps(halves, halves);
  return _mm_cvtss_f32(_mm_hadd_ps(pairs, pairs));
}

ORBITQUANT_TARGET_AVX512 inline float horizontal_sum(__m512 value) {
  return _mm512_reduce_add_ps(value);
}

template <int row_tile>
ORBITQUANT_TARGET_AVX2 inline void packed_adaln_avx2_rows(
    AdalnArgs const &args,
    std::uint8_t const *packed_row,
    float const *scale_row,
    std::int64_t out_col,
    std::int64_t row_start) {
  __m256 accumulators[row_tile];
  float tails[row_tile]{};
#pragma clang loop unroll(full)
  for (int row = 0; row < row_tile; ++row) {
    accumulators[row] = _mm256_setzero_ps();
  }
  const __m128i nibble_mask = _mm_set1_epi8(15);
  const __m256i low_table_limit = _mm256_set1_epi32(7);

  for (std::int64_t group = 0; group < args.num_groups; ++group) {
    float lut[16];
    fill_group_lut(lut, scale_row[group]);
    const __m256 lut_low = _mm256_loadu_ps(lut);
    const __m256 lut_high = _mm256_loadu_ps(lut + 8);
    const std::int64_t group_start = group * args.group_size;
    const std::int64_t group_end = std::min(
        args.in_features,
        group_start + args.group_size);
    std::int64_t k = group_start;
    for (; k + 8 <= group_end; k += 8) {
      std::int32_t packed;
      std::memcpy(&packed, packed_row + k / 2, sizeof(packed));
      const __m128i bytes = _mm_cvtsi32_si128(packed);
      const __m128i low = _mm_and_si128(bytes, nibble_mask);
      const __m128i high = _mm_and_si128(
          _mm_srli_epi16(bytes, 4),
          nibble_mask);
      const __m256i indices =
          _mm256_cvtepu8_epi32(_mm_unpacklo_epi8(low, high));
      const __m256 weight = _mm256_blendv_ps(
          _mm256_permutevar8x32_ps(lut_low, indices),
          _mm256_permutevar8x32_ps(lut_high, indices),
          _mm256_castsi256_ps(
              _mm256_cmpgt_epi32(indices, low_table_limit)));
#pragma clang loop unroll(full)
      for (int row = 0; row < row_tile; ++row) {
        const std::int64_t input_offset =
            (row_start + row) * args.in_features + k;
        accumulators[row] = _mm256_fmadd_ps(
            load_bfloat8(args.x, input_offset),
            weight,
            accumulators[row]);
      }
    }
    for (; k < group_end; ++k) {
      const float weight = lut[unpack_adaln_index(packed_row, k)];
#pragma clang loop unroll(full)
      for (int row = 0; row < row_tile; ++row) {
        tails[row] += load_bfloat(
            args.x,
            (row_start + row) * args.in_features + k) * weight;
      }
    }
  }

#pragma clang loop unroll(full)
  for (int row = 0; row < row_tile; ++row) {
    float accumulator = horizontal_sum(accumulators[row]) + tails[row];
    if (args.has_bias) {
      accumulator += args.bias[out_col];
    }
    store_bfloat(
        args.out,
        (row_start + row) * args.out_features + out_col,
        accumulator);
  }
}

template <int row_tile>
ORBITQUANT_TARGET_AVX512 inline void packed_adaln_avx512_rows(
    AdalnArgs const &args,
    std::uint8_t const *packed_row,
    float const *scale_row,
    std::int64_t out_col,
    std::int64_t row_start) {
  __m512 accumulators[row_tile];
  float tails[row_tile]{};
#pragma clang loop unroll(full)
  for (int row = 0; row < row_tile; ++row) {
    accumulators[row] = _mm512_setzero_ps();
  }
  const __m128i nibble_mask = _mm_set1_epi8(15);

  for (std::int64_t group = 0; group < args.num_groups; ++group) {
    float lut[16];
    fill_group_lut(lut, scale_row[group]);
    const __m512 centroid_lut = _mm512_loadu_ps(lut);
    const std::int64_t group_start = group * args.group_size;
    const std::int64_t group_end = std::min(
        args.in_features,
        group_start + args.group_size);
    std::int64_t k = group_start;
    for (; k + 16 <= group_end; k += 16) {
      std::int64_t packed;
      std::memcpy(&packed, packed_row + k / 2, sizeof(packed));
      const __m128i bytes = _mm_cvtsi64_si128(packed);
      const __m128i low = _mm_and_si128(bytes, nibble_mask);
      const __m128i high = _mm_and_si128(
          _mm_srli_epi16(bytes, 4),
          nibble_mask);
      const __m512i indices =
          _mm512_cvtepu8_epi32(_mm_unpacklo_epi8(low, high));
      const __m512 weight =
          _mm512_permutexvar_ps(indices, centroid_lut);
#pragma clang loop unroll(full)
      for (int row = 0; row < row_tile; ++row) {
        const std::int64_t input_offset =
            (row_start + row) * args.in_features + k;
        accumulators[row] = _mm512_fmadd_ps(
            load_bfloat16(args.x, input_offset),
            weight,
            accumulators[row]);
      }
    }
    for (; k < group_end; ++k) {
      const float weight = lut[unpack_adaln_index(packed_row, k)];
#pragma clang loop unroll(full)
      for (int row = 0; row < row_tile; ++row) {
        tails[row] += load_bfloat(
            args.x,
            (row_start + row) * args.in_features + k) * weight;
      }
    }
  }

#pragma clang loop unroll(full)
  for (int row = 0; row < row_tile; ++row) {
    float accumulator = horizontal_sum(accumulators[row]) + tails[row];
    if (args.has_bias) {
      accumulator += args.bias[out_col];
    }
    store_bfloat(
        args.out,
        (row_start + row) * args.out_features + out_col,
        accumulator);
  }
}

#if defined(ORBITQUANT_HAS_AVX512_BF16_INTRINSICS)
template <bool aligned_k>
ORBITQUANT_TARGET_AVX512_BF16 inline __m512bh load_bfloat32(
    std::uint16_t const *data,
    __mmask32 valid_mask) {
  if constexpr (aligned_k) {
    return (__m512bh)_mm512_loadu_si512(data);
  }
  return (__m512bh)_mm512_maskz_loadu_epi16(valid_mask, data);
}

template <int row_tile, bool aligned_k>
ORBITQUANT_TARGET_AVX512_BF16 inline void packed_adaln_avx512_bf16_rows(
    AdalnArgs const &args,
    std::uint8_t const *packed_row,
    float const *scale_row,
    std::int64_t out_col,
    std::int64_t row_start) {
  static_assert(row_tile >= 1 && row_tile <= 8);
  __m512 accumulator0 = _mm512_setzero_ps();
  __m512 accumulator1 = _mm512_setzero_ps();
  __m512 accumulator2 = _mm512_setzero_ps();
  __m512 accumulator3 = _mm512_setzero_ps();
  __m512 accumulator4 = _mm512_setzero_ps();
  __m512 accumulator5 = _mm512_setzero_ps();
  __m512 accumulator6 = _mm512_setzero_ps();
  __m512 accumulator7 = _mm512_setzero_ps();
  const __m128i nibble_mask = _mm_set1_epi8(15);
  const __m512 signed_codes = _mm512_setr_ps(
      -8.0f, -7.0f, -6.0f, -5.0f,
      -4.0f, -3.0f, -2.0f, -1.0f,
      0.0f, 1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f, 7.0f);
  const auto *input_base =
      static_cast<std::uint16_t const *>(args.x) +
      row_start * args.in_features;
  const std::int64_t input_stride = args.in_features;

  for (std::int64_t group = 0; group < args.num_groups; ++group) {
    const __m512 scaled_codes = _mm512_mul_ps(
        signed_codes,
        _mm512_set1_ps(scale_row[group]));
    const __m512i lut_words = (__m512i)_mm512_cvtne2ps_pbh(
        scaled_codes,
        scaled_codes);
    const std::int64_t group_start = group * args.group_size;
    for (std::int64_t offset = 0; offset < args.group_size; offset += 32) {
      const std::int64_t k = group_start + offset;
      __mmask32 valid_mask = static_cast<__mmask32>(0xffffffffu);
      if constexpr (!aligned_k) {
        const std::int64_t valid = std::max<std::int64_t>(
            0,
            std::min<std::int64_t>(32, args.in_features - k));
        if (valid == 0) {
          continue;
        }
        valid_mask = valid == 32
            ? static_cast<__mmask32>(0xffffffffu)
            : static_cast<__mmask32>((1u << valid) - 1u);
      }
      const __m128i bytes = _mm_loadu_si128(
          reinterpret_cast<__m128i const *>(packed_row + k / 2));
      const __m128i low = _mm_and_si128(bytes, nibble_mask);
      const __m128i high = _mm_and_si128(
          _mm_srli_epi16(bytes, 4),
          nibble_mask);
      const __m256i packed_indices = _mm256_set_m128i(
          _mm_unpackhi_epi8(low, high),
          _mm_unpacklo_epi8(low, high));
      const __m512i indices =
          _mm512_cvtepu8_epi16(packed_indices);
      const __m512bh weights = (__m512bh)_mm512_permutexvar_epi16(
          indices,
          lut_words);
      accumulator0 = _mm512_dpbf16_ps(
          accumulator0,
          load_bfloat32<aligned_k>(input_base + k, valid_mask),
          weights);
      if constexpr (row_tile > 1) {
        accumulator1 = _mm512_dpbf16_ps(
            accumulator1,
            load_bfloat32<aligned_k>(
                input_base + input_stride + k,
                valid_mask),
            weights);
      }
      if constexpr (row_tile > 2) {
        accumulator2 = _mm512_dpbf16_ps(
            accumulator2,
            load_bfloat32<aligned_k>(
                input_base + 2 * input_stride + k,
                valid_mask),
            weights);
      }
      if constexpr (row_tile > 3) {
        accumulator3 = _mm512_dpbf16_ps(
            accumulator3,
            load_bfloat32<aligned_k>(
                input_base + 3 * input_stride + k,
                valid_mask),
            weights);
      }
      if constexpr (row_tile > 4) {
        accumulator4 = _mm512_dpbf16_ps(
            accumulator4,
            load_bfloat32<aligned_k>(
                input_base + 4 * input_stride + k,
                valid_mask),
            weights);
      }
      if constexpr (row_tile > 5) {
        accumulator5 = _mm512_dpbf16_ps(
            accumulator5,
            load_bfloat32<aligned_k>(
                input_base + 5 * input_stride + k,
                valid_mask),
            weights);
      }
      if constexpr (row_tile > 6) {
        accumulator6 = _mm512_dpbf16_ps(
            accumulator6,
            load_bfloat32<aligned_k>(
                input_base + 6 * input_stride + k,
                valid_mask),
            weights);
      }
      if constexpr (row_tile > 7) {
        accumulator7 = _mm512_dpbf16_ps(
            accumulator7,
            load_bfloat32<aligned_k>(
                input_base + 7 * input_stride + k,
                valid_mask),
            weights);
      }
    }
  }

  const float bias = args.has_bias ? args.bias[out_col] : 0.0f;
  const std::int64_t output_offset =
      row_start * args.out_features + out_col;
  store_bfloat(
      args.out,
      output_offset,
      horizontal_sum(accumulator0) + bias);
  if constexpr (row_tile > 1) {
    store_bfloat(
        args.out,
        output_offset + args.out_features,
        horizontal_sum(accumulator1) + bias);
  }
  if constexpr (row_tile > 2) {
    store_bfloat(
        args.out,
        output_offset + 2 * args.out_features,
        horizontal_sum(accumulator2) + bias);
  }
  if constexpr (row_tile > 3) {
    store_bfloat(
        args.out,
        output_offset + 3 * args.out_features,
        horizontal_sum(accumulator3) + bias);
  }
  if constexpr (row_tile > 4) {
    store_bfloat(
        args.out,
        output_offset + 4 * args.out_features,
        horizontal_sum(accumulator4) + bias);
  }
  if constexpr (row_tile > 5) {
    store_bfloat(
        args.out,
        output_offset + 5 * args.out_features,
        horizontal_sum(accumulator5) + bias);
  }
  if constexpr (row_tile > 6) {
    store_bfloat(
        args.out,
        output_offset + 6 * args.out_features,
        horizontal_sum(accumulator6) + bias);
  }
  if constexpr (row_tile > 7) {
    store_bfloat(
        args.out,
        output_offset + 7 * args.out_features,
        horizontal_sum(accumulator7) + bias);
  }
}

bool adaln_avx512_bf16_available() {
  __builtin_cpu_init();
  return orbitquant::cpu::packed_matmul_x86_avx512_available() &&
      __builtin_cpu_supports("avx512bf16");
}
#else
bool adaln_avx512_bf16_available() {
  return false;
}
#endif

template <
    void (*rows8)(AdalnArgs const &, std::uint8_t const *, float const *, std::int64_t, std::int64_t),
    void (*rows4)(AdalnArgs const &, std::uint8_t const *, float const *, std::int64_t, std::int64_t),
    void (*rows3)(AdalnArgs const &, std::uint8_t const *, float const *, std::int64_t, std::int64_t),
    void (*rows2)(AdalnArgs const &, std::uint8_t const *, float const *, std::int64_t, std::int64_t),
    void (*rows1)(AdalnArgs const &, std::uint8_t const *, float const *, std::int64_t, std::int64_t)>
void packed_adaln_x86_tiled_range(
    AdalnArgs const &args,
    std::int64_t out_start,
    std::int64_t out_end) {
  const std::int64_t packed_row_bytes = args.padded_in_features / 2;
  for (std::int64_t out_col = out_start; out_col < out_end; ++out_col) {
    const auto *packed_row =
        args.packed_weight + out_col * packed_row_bytes;
    const auto *scale_row =
        args.scales + out_col * args.num_groups;
    std::int64_t row = 0;
    for (; row + 8 <= args.rows; row += 8) {
      rows8(args, packed_row, scale_row, out_col, row);
    }
    if (row + 4 <= args.rows) {
      rows4(args, packed_row, scale_row, out_col, row);
      row += 4;
    }
    switch (args.rows - row) {
      case 3:
        rows3(args, packed_row, scale_row, out_col, row);
        break;
      case 2:
        rows2(args, packed_row, scale_row, out_col, row);
        break;
      case 1:
        rows1(args, packed_row, scale_row, out_col, row);
        break;
      default:
        break;
    }
  }
}

void packed_adaln_avx2_range(
    AdalnArgs const &args,
    std::int64_t out_start,
    std::int64_t out_end) {
  if (args.group_size % 8 != 0 || args.padded_in_features % 2 != 0) {
    packed_adaln_scalar_range(args, out_start, out_end);
    return;
  }
  packed_adaln_x86_tiled_range<
      packed_adaln_avx2_rows<8>,
      packed_adaln_avx2_rows<4>,
      packed_adaln_avx2_rows<3>,
      packed_adaln_avx2_rows<2>,
      packed_adaln_avx2_rows<1>>(args, out_start, out_end);
}

void packed_adaln_avx512_range(
    AdalnArgs const &args,
    std::int64_t out_start,
    std::int64_t out_end) {
#if defined(ORBITQUANT_HAS_AVX512_BF16_INTRINSICS)
  if (args.group_size % 32 == 0 &&
      args.padded_in_features % 2 == 0 &&
      adaln_avx512_bf16_available()) {
    if (args.in_features % args.group_size == 0) {
      packed_adaln_x86_tiled_range<
          packed_adaln_avx512_bf16_rows<8, true>,
          packed_adaln_avx512_bf16_rows<4, true>,
          packed_adaln_avx512_bf16_rows<3, true>,
          packed_adaln_avx512_bf16_rows<2, true>,
          packed_adaln_avx512_bf16_rows<1, true>>(args, out_start, out_end);
      return;
    }
    packed_adaln_x86_tiled_range<
        packed_adaln_avx512_bf16_rows<8, false>,
        packed_adaln_avx512_bf16_rows<4, false>,
        packed_adaln_avx512_bf16_rows<3, false>,
        packed_adaln_avx512_bf16_rows<2, false>,
        packed_adaln_avx512_bf16_rows<1, false>>(args, out_start, out_end);
    return;
  }
#endif
  if (args.group_size % 16 != 0 || args.padded_in_features % 2 != 0) {
    packed_adaln_scalar_range(args, out_start, out_end);
    return;
  }
  packed_adaln_x86_tiled_range<
      packed_adaln_avx512_rows<8>,
      packed_adaln_avx512_rows<4>,
      packed_adaln_avx512_rows<3>,
      packed_adaln_avx512_rows<2>,
      packed_adaln_avx512_rows<1>>(args, out_start, out_end);
}
#else
void packed_adaln_avx2_range(
    AdalnArgs const &args,
    std::int64_t out_start,
    std::int64_t out_end) {
  packed_adaln_scalar_range(args, out_start, out_end);
}

void packed_adaln_avx512_range(
    AdalnArgs const &args,
    std::int64_t out_start,
    std::int64_t out_end) {
  packed_adaln_scalar_range(args, out_start, out_end);
}
#endif

AdalnRangeFn select_adaln_kernel() {
  const char *requested = std::getenv("ORBITQUANT_CPU_ISA");
  if (requested == nullptr || std::strcmp(requested, "auto") == 0) {
    if (orbitquant::cpu::packed_matmul_x86_avx512_available()) {
      return packed_adaln_avx512_range;
    }
    if (orbitquant::cpu::packed_matmul_x86_avx2_available()) {
#if defined(_MSC_VER) && defined(_M_X64)
      return orbitquant::cpu::packed_adaln_msvc_avx2_range;
#else
      return packed_adaln_avx2_range;
#endif
    }
    if (orbitquant::cpu::packed_matmul_neon_available()) {
      return packed_adaln_neon_range;
    }
    return packed_adaln_scalar_range;
  }
  if (std::strcmp(requested, "scalar") == 0) {
    return packed_adaln_scalar_range;
  }
  if (std::strcmp(requested, "avx2") == 0) {
    STD_TORCH_CHECK(
        orbitquant::cpu::packed_matmul_x86_avx2_available(),
        "ORBITQUANT_CPU_ISA=avx2 requested AVX2/FMA/F16C on an unsupported CPU");
#if defined(_MSC_VER) && defined(_M_X64)
    return orbitquant::cpu::packed_adaln_msvc_avx2_range;
#else
    return packed_adaln_avx2_range;
#endif
  }
  if (std::strcmp(requested, "avx512") == 0) {
    STD_TORCH_CHECK(
        orbitquant::cpu::packed_matmul_x86_avx512_available(),
        "ORBITQUANT_CPU_ISA=avx512 requested AVX-512F/DQ/BW/VL on an unsupported CPU");
    return packed_adaln_avx512_range;
  }
  if (std::strcmp(requested, "neon") == 0) {
    STD_TORCH_CHECK(
        orbitquant::cpu::packed_matmul_neon_available(),
        "ORBITQUANT_CPU_ISA=neon requested NEON on an unsupported CPU");
    return packed_adaln_neon_range;
  }
  STD_TORCH_CHECK(
      false,
      "ORBITQUANT_CPU_ISA must be auto, scalar, avx2, avx512, or neon");
  return packed_adaln_scalar_range;
}

void parallel_packed_adaln(
    AdalnArgs const &args,
    AdalnRangeFn function) {
  const std::int64_t arithmetic =
      args.rows * args.out_features * args.in_features;
  const int max_threads = orbitquant::cpu::requested_threads();
  const int threads = arithmetic < 1'000'000
      ? 1
      : std::max<int>(
            1,
            std::min<std::int64_t>(
                max_threads,
                args.out_features / 16));
  if (threads == 1) {
    function(args, 0, args.out_features);
    return;
  }

  std::vector<std::thread> workers;
  workers.reserve(threads);
  const std::int64_t columns_per_thread =
      (args.out_features + threads - 1) / threads;
  for (int thread = 0; thread < threads; ++thread) {
    const std::int64_t start = thread * columns_per_thread;
    const std::int64_t end = std::min(
        args.out_features,
        start + columns_per_thread);
    if (start >= end) {
      break;
    }
    workers.emplace_back([&args, function, start, end] {
      function(args, start, end);
    });
  }
  for (auto &worker : workers) {
    worker.join();
  }
}

}  // namespace

void matmul_packed_adaln_int4_cpu(
    OrbitQuantTensor &out,
    OrbitQuantTensor const &x,
    OrbitQuantTensor const &packed_weight,
    OrbitQuantTensor const &scales,
    OrbitQuantTensor const &bias,
    bool has_bias,
    int64_t out_features,
    int64_t in_features,
    int64_t group_size) {
  using torch::headeronly::DeviceType;
  using torch::headeronly::ScalarType;

  STD_TORCH_CHECK(x.device().type() == DeviceType::CPU, "x must be a CPU tensor");
  STD_TORCH_CHECK(out.device().type() == DeviceType::CPU, "out must be a CPU tensor");
  STD_TORCH_CHECK(
      packed_weight.device().type() == DeviceType::CPU,
      "packed_weight must be a CPU tensor");
  STD_TORCH_CHECK(
      scales.device().type() == DeviceType::CPU,
      "scales must be a CPU tensor");
  STD_TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  STD_TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  STD_TORCH_CHECK(packed_weight.is_contiguous(), "packed_weight must be contiguous");
  STD_TORCH_CHECK(scales.is_contiguous(), "scales must be contiguous");
  STD_TORCH_CHECK(x.scalar_type() == ScalarType::BFloat16, "x must be bfloat16");
  STD_TORCH_CHECK(out.scalar_type() == ScalarType::BFloat16, "out must be bfloat16");
  STD_TORCH_CHECK(
      packed_weight.scalar_type() == ScalarType::Byte,
      "packed_weight must be uint8");
  STD_TORCH_CHECK(scales.scalar_type() == ScalarType::Float, "scales must be float32");
  STD_TORCH_CHECK(x.dim() == 2, "x must be rank 2");
  STD_TORCH_CHECK(out.dim() == 2, "out must be rank 2");
  STD_TORCH_CHECK(group_size > 0, "group_size must be positive");
  STD_TORCH_CHECK(x.size(1) == in_features, "x has an unexpected input dimension");
  STD_TORCH_CHECK(out.size(0) == x.size(0), "out has an unexpected row count");
  STD_TORCH_CHECK(out.size(1) == out_features, "out has an unexpected output dimension");
  const int64_t num_groups = (in_features + group_size - 1) / group_size;
  const int64_t padded_in_features = num_groups * group_size;
  const int64_t packed_values = out_features * padded_in_features;
  STD_TORCH_CHECK(
      packed_weight.numel() >= (packed_values + 1) / 2,
      "packed_weight is too short");
  STD_TORCH_CHECK(
      scales.numel() == out_features * num_groups,
      "scales must match out_features and num_groups");
  if (has_bias) {
    STD_TORCH_CHECK(bias.device().type() == DeviceType::CPU, "bias must be a CPU tensor");
    STD_TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    STD_TORCH_CHECK(bias.scalar_type() == ScalarType::Float, "bias must be float32");
    STD_TORCH_CHECK(bias.numel() == out_features, "bias must match out_features");
  }
  if (x.numel() == 0 || out_features == 0) {
    return;
  }

  const AdalnArgs args{
      out.mutable_data_ptr(),
      x.const_data_ptr(),
      packed_weight.const_data_ptr<std::uint8_t>(),
      scales.const_data_ptr<float>(),
      has_bias ? bias.const_data_ptr<float>() : nullptr,
      has_bias,
      x.size(0),
      out_features,
      in_features,
      group_size,
      num_groups,
      padded_in_features,
  };
  parallel_packed_adaln(args, select_adaln_kernel());
}
