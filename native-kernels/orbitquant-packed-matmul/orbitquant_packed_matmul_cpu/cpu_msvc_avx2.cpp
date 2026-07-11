#include "cpu_kernel_args.h"

#if defined(_MSC_VER) && defined(_M_X64)
#include <immintrin.h>

#include <torch/headeronly/util/BFloat16.h>
#include <torch/headeronly/util/Half.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace orbitquant::cpu {
namespace {

template <typename scalar_t>
inline float load_scalar(void const *data, std::int64_t offset) {
  return static_cast<float>(static_cast<scalar_t const *>(data)[offset]);
}

template <>
inline float load_scalar<float>(void const *data, std::int64_t offset) {
  return static_cast<float const *>(data)[offset];
}

template <typename scalar_t>
inline void store_scalar(void *data, std::int64_t offset, float value) {
  static_cast<scalar_t *>(data)[offset] = scalar_t(value);
}

template <>
inline void store_scalar<float>(void *data, std::int64_t offset, float value) {
  static_cast<float *>(data)[offset] = value;
}

template <typename scalar_t>
float squared_norm_avx2(
    void const *data,
    std::int64_t offset,
    std::int64_t dim) {
  __m256 accumulator = _mm256_setzero_ps();
  std::int64_t index = 0;
  for (; index + 8 <= dim; index += 8) {
    __m256 values;
    if constexpr (std::is_same_v<scalar_t, float>) {
      values = _mm256_loadu_ps(
          static_cast<float const *>(data) + offset + index);
    } else if constexpr (std::is_same_v<scalar_t, c10::Half>) {
      const auto *source =
          static_cast<std::uint16_t const *>(data) + offset + index;
      values = _mm256_cvtph_ps(
          _mm_loadu_si128(reinterpret_cast<__m128i const *>(source)));
    } else {
      const auto *source =
          static_cast<std::uint16_t const *>(data) + offset + index;
      const __m128i packed =
          _mm_loadu_si128(reinterpret_cast<__m128i const *>(source));
      values = _mm256_castsi256_ps(
          _mm256_slli_epi32(_mm256_cvtepu16_epi32(packed), 16));
    }
    accumulator = _mm256_fmadd_ps(values, values, accumulator);
  }
  const __m128 halves = _mm_add_ps(
      _mm256_castps256_ps128(accumulator),
      _mm256_extractf128_ps(accumulator, 1));
  const __m128 pairs = _mm_hadd_ps(halves, halves);
  float result = _mm_cvtss_f32(_mm_hadd_ps(pairs, pairs));
  for (; index < dim; ++index) {
    const float value = load_scalar<scalar_t>(data, offset + index);
    result += value * value;
  }
  return result;
}

inline std::int64_t nearest_centroid(
    float value,
    float const *boundaries,
    std::int64_t boundary_count) {
  std::int64_t low = 0;
  std::int64_t high = boundary_count;
  while (low < high) {
    const std::int64_t middle = low + (high - low) / 2;
    if (value <= boundaries[middle]) {
      high = middle;
    } else {
      low = middle + 1;
    }
  }
  return low;
}

inline __m128i float8_to_bfloat8(__m256 values) {
  const __m256i bits = _mm256_castps_si256(values);
  const __m256i absolute_bits =
      _mm256_and_si256(bits, _mm256_set1_epi32(0x7fffffff));
  const __m256i nan_mask =
      _mm256_cmpgt_epi32(absolute_bits, _mm256_set1_epi32(0x7f800000));
  const __m256i rounding = _mm256_add_epi32(
      _mm256_set1_epi32(0x7fff),
      _mm256_and_si256(_mm256_srli_epi32(bits, 16), _mm256_set1_epi32(1)));
  const __m256i upper = _mm256_srli_epi32(
      _mm256_add_epi32(bits, rounding),
      16);
  __m128i packed = _mm_packus_epi32(
      _mm256_castsi256_si128(upper),
      _mm256_extracti128_si256(upper, 1));
  const __m128i packed_nan_mask = _mm_packs_epi32(
      _mm256_castsi256_si128(nan_mask),
      _mm256_extracti128_si256(nan_mask, 1));
  return _mm_blendv_epi8(
      packed,
      _mm_set1_epi16(0x7fc0),
      packed_nan_mask);
}

template <typename scalar_t>
void quantize_lookup_avx2(
    ActivationArgs const &args,
    float const *scratch,
    std::int64_t output_offset,
    float norm) {
  const __m256 inverse_sqrt_block = _mm256_set1_ps(args.inv_sqrt_block);
  const __m256 output_norm = _mm256_set1_ps(norm);
  const __m256i ones = _mm256_set1_epi32(1);
  std::int64_t index = 0;
  for (; index + 8 <= args.dim; index += 8) {
    const __m256 direction = _mm256_mul_ps(
        _mm256_loadu_ps(scratch + index), inverse_sqrt_block);
    __m256i centroid_indices = _mm256_setzero_si256();
    for (std::int64_t boundary = 0; boundary < args.boundary_count; ++boundary) {
      const __m256 comparison = _mm256_cmp_ps(
          direction,
          _mm256_set1_ps(args.boundaries[boundary]),
          _CMP_NLE_UQ);
      centroid_indices = _mm256_add_epi32(
          centroid_indices,
          _mm256_and_si256(_mm256_castps_si256(comparison), ones));
    }
    const __m256 output = _mm256_mul_ps(
        _mm256_i32gather_ps(args.centroids, centroid_indices, 4),
        output_norm);
    if constexpr (std::is_same_v<scalar_t, float>) {
      _mm256_storeu_ps(
          static_cast<float *>(args.out) + output_offset + index,
          output);
    } else if constexpr (std::is_same_v<scalar_t, c10::Half>) {
      _mm_storeu_si128(
          reinterpret_cast<__m128i *>(
              static_cast<std::uint16_t *>(args.out) + output_offset + index),
          _mm256_cvtps_ph(output, _MM_FROUND_TO_NEAREST_INT));
    } else {
      _mm_storeu_si128(
          reinterpret_cast<__m128i *>(
              static_cast<std::uint16_t *>(args.out) + output_offset + index),
          float8_to_bfloat8(output));
    }
  }
  for (; index < args.dim; ++index) {
    const float direction = scratch[index] * args.inv_sqrt_block;
    const std::int64_t centroid_index = nearest_centroid(
        direction, args.boundaries, args.boundary_count);
    store_scalar<scalar_t>(
        args.out,
        output_offset + index,
        args.centroids[centroid_index] * norm);
  }
}

inline float load_bfloat(void const *data, std::int64_t offset) {
  return static_cast<float>(
      static_cast<c10::BFloat16 const *>(data)[offset]);
}

inline void store_bfloat(void *data, std::int64_t offset, float value) {
  static_cast<c10::BFloat16 *>(data)[offset] = c10::BFloat16(value);
}

inline std::uint8_t unpack_adaln_index(
    std::uint8_t const *packed,
    std::int64_t flat_index) {
  const std::uint8_t byte = packed[flat_index / 2];
  return (flat_index & 1) == 0 ? byte & 15u : (byte >> 4) & 15u;
}

inline float dequantized_adaln_value(std::uint8_t index, float scale) {
  return static_cast<float>(
      c10::BFloat16((static_cast<int>(index) - 8) * scale));
}

inline void fill_group_lut(float *lut, float scale) {
  for (int index = 0; index < 16; ++index) {
    lut[index] = dequantized_adaln_value(
        static_cast<std::uint8_t>(index), scale);
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

inline __m256 load_bfloat8(void const *data, std::int64_t offset) {
  const auto *source =
      static_cast<std::uint16_t const *>(data) + offset;
  const __m128i packed =
      _mm_loadu_si128(reinterpret_cast<__m128i const *>(source));
  return _mm256_castsi256_ps(
      _mm256_slli_epi32(_mm256_cvtepu16_epi32(packed), 16));
}

inline float horizontal_sum(__m256 value) {
  const __m128 halves = _mm_add_ps(
      _mm256_castps256_ps128(value),
      _mm256_extractf128_ps(value, 1));
  const __m128 pairs = _mm_hadd_ps(halves, halves);
  return _mm_cvtss_f32(_mm_hadd_ps(pairs, pairs));
}

template <int row_tile>
inline void packed_adaln_avx2_rows(
    AdalnArgs const &args,
    std::uint8_t const *packed_row,
    float const *scale_row,
    std::int64_t out_col,
    std::int64_t row_start) {
  __m256 accumulators[row_tile];
  float tails[row_tile]{};
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
          _mm_srli_epi16(bytes, 4), nibble_mask);
      const __m256i indices =
          _mm256_cvtepu8_epi32(_mm_unpacklo_epi8(low, high));
      const __m256 weight = _mm256_blendv_ps(
          _mm256_permutevar8x32_ps(lut_low, indices),
          _mm256_permutevar8x32_ps(lut_high, indices),
          _mm256_castsi256_ps(
              _mm256_cmpgt_epi32(indices, low_table_limit)));
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
      for (int row = 0; row < row_tile; ++row) {
        tails[row] += load_bfloat(
            args.x,
            (row_start + row) * args.in_features + k) * weight;
      }
    }
  }

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

template <
    void (*rows8)(AdalnArgs const &, std::uint8_t const *, float const *, std::int64_t, std::int64_t),
    void (*rows4)(AdalnArgs const &, std::uint8_t const *, float const *, std::int64_t, std::int64_t),
    void (*rows3)(AdalnArgs const &, std::uint8_t const *, float const *, std::int64_t, std::int64_t),
    void (*rows2)(AdalnArgs const &, std::uint8_t const *, float const *, std::int64_t, std::int64_t),
    void (*rows1)(AdalnArgs const &, std::uint8_t const *, float const *, std::int64_t, std::int64_t)>
void packed_adaln_tiled_range(
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

}  // namespace

void activation_fwht_msvc_avx2(float *values, std::int64_t block_size) {
  for (std::int64_t half = 1; half < block_size; half *= 2) {
    for (std::int64_t base = 0; base < block_size; base += 2 * half) {
      std::int64_t offset = 0;
      for (; offset + 8 <= half; offset += 8) {
        const __m256 left = _mm256_loadu_ps(values + base + offset);
        const __m256 right =
            _mm256_loadu_ps(values + base + half + offset);
        _mm256_storeu_ps(values + base + offset, _mm256_add_ps(left, right));
        _mm256_storeu_ps(
            values + base + half + offset,
            _mm256_sub_ps(left, right));
      }
      for (; offset < half; ++offset) {
        const float left = values[base + offset];
        const float right = values[base + half + offset];
        values[base + offset] = left + right;
        values[base + half + offset] = left - right;
      }
    }
  }
}

float activation_squared_norm_msvc_avx2(
    void const *data,
    ScalarKind scalar_kind,
    std::int64_t offset,
    std::int64_t dim) {
  switch (scalar_kind) {
    case ScalarKind::Float32:
      return squared_norm_avx2<float>(data, offset, dim);
    case ScalarKind::Float16:
      return squared_norm_avx2<c10::Half>(data, offset, dim);
    case ScalarKind::BFloat16:
      return squared_norm_avx2<c10::BFloat16>(data, offset, dim);
  }
  return 0.0f;
}

void activation_quantize_lookup_msvc_avx2(
    ActivationArgs const &args,
    float const *scratch,
    std::int64_t output_offset,
    float norm) {
  switch (args.scalar_kind) {
    case ScalarKind::Float32:
      quantize_lookup_avx2<float>(args, scratch, output_offset, norm);
      return;
    case ScalarKind::Float16:
      quantize_lookup_avx2<c10::Half>(args, scratch, output_offset, norm);
      return;
    case ScalarKind::BFloat16:
      quantize_lookup_avx2<c10::BFloat16>(
          args, scratch, output_offset, norm);
      return;
  }
}

void packed_adaln_msvc_avx2_range(
    AdalnArgs const &args,
    std::int64_t out_start,
    std::int64_t out_end) {
  if (args.group_size % 8 != 0 || args.padded_in_features % 2 != 0) {
    packed_adaln_scalar_range(args, out_start, out_end);
    return;
  }
  packed_adaln_tiled_range<
      packed_adaln_avx2_rows<8>,
      packed_adaln_avx2_rows<4>,
      packed_adaln_avx2_rows<3>,
      packed_adaln_avx2_rows<2>,
      packed_adaln_avx2_rows<1>>(args, out_start, out_end);
}

}  // namespace orbitquant::cpu
#endif
