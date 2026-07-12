#include "cpu_pool.h"
#include "cpu_threads.h"
#include "cpu_kernel_args.h"
#include "packed_matmul_cpu.h"
#include "../torch-ext/torch_binding.h"

#include <torch/headeronly/core/DeviceType.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>
#include <torch/headeronly/util/BFloat16.h>
#include <torch/headeronly/util/Half.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#endif

#if (defined(__x86_64__) || defined(_M_X64)) && !defined(_MSC_VER)
#include <immintrin.h>
#define ORBITQUANT_TARGET_AVX2 __attribute__((target("avx2,fma,f16c")))
#define ORBITQUANT_TARGET_AVX512 \
  __attribute__((target("avx512f,avx512dq,avx512bw,avx512vl,fma,f16c")))
#endif

namespace {

using orbitquant::cpu::ActivationArgs;
using orbitquant::cpu::ActivationIsa;

ActivationIsa select_activation_isa() {
  const char *requested = std::getenv("ORBITQUANT_CPU_ISA");
  if (requested == nullptr || std::strcmp(requested, "auto") == 0) {
    if (orbitquant::cpu::packed_matmul_x86_avx512_available()) {
      return ActivationIsa::Avx512;
    }
    if (orbitquant::cpu::packed_matmul_x86_avx2_available()) {
      return ActivationIsa::Avx2;
    }
    if (orbitquant::cpu::packed_matmul_neon_available()) {
      return ActivationIsa::Neon;
    }
    return ActivationIsa::Portable;
  }
  if (std::strcmp(requested, "scalar") == 0) {
    return ActivationIsa::Portable;
  }
  if (std::strcmp(requested, "avx2") == 0) {
    STD_TORCH_CHECK(
        orbitquant::cpu::packed_matmul_x86_avx2_available(),
        "ORBITQUANT_CPU_ISA=avx2 requested AVX2/FMA/F16C on an unsupported CPU");
    return ActivationIsa::Avx2;
  }
  if (std::strcmp(requested, "avx512") == 0) {
    STD_TORCH_CHECK(
        orbitquant::cpu::packed_matmul_x86_avx512_available(),
        "ORBITQUANT_CPU_ISA=avx512 requested AVX-512F/DQ/BW/VL on an unsupported CPU");
    return ActivationIsa::Avx512;
  }
  if (std::strcmp(requested, "neon") == 0) {
    STD_TORCH_CHECK(
        orbitquant::cpu::packed_matmul_neon_available(),
        "ORBITQUANT_CPU_ISA=neon requested NEON on an unsupported CPU");
    return ActivationIsa::Neon;
  }
  STD_TORCH_CHECK(
      false,
      "ORBITQUANT_CPU_ISA must be auto, scalar, avx2, avx512, or neon");
  return ActivationIsa::Portable;
}

orbitquant::cpu::ScalarKind activation_scalar_kind(
    OrbitQuantTensor const &tensor) {
  using torch::headeronly::ScalarType;
  switch (tensor.scalar_type()) {
    case ScalarType::Float:
      return orbitquant::cpu::ScalarKind::Float32;
    case ScalarType::Half:
      return orbitquant::cpu::ScalarKind::Float16;
    case ScalarType::BFloat16:
      return orbitquant::cpu::ScalarKind::BFloat16;
    default:
      STD_TORCH_CHECK(
          false,
          "CPU activation quantization supports float32, float16, and bfloat16 inputs");
  }
  return orbitquant::cpu::ScalarKind::Float32;
}

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

inline void fwht_block_portable(float *values, std::int64_t block_size) {
  for (std::int64_t half = 1; half < block_size; half *= 2) {
    for (std::int64_t base = 0; base < block_size; base += 2 * half) {
      for (std::int64_t offset = 0; offset < half; ++offset) {
        const float left = values[base + offset];
        const float right = values[base + half + offset];
        values[base + offset] = left + right;
        values[base + half + offset] = left - right;
      }
    }
  }
}

#if defined(__aarch64__) || defined(_M_ARM64)
inline void fwht_block_neon(float *values, std::int64_t block_size) {
  if (block_size < 4) {
    fwht_block_portable(values, block_size);
    return;
  }
  const uint32x4_t odd_lanes = {0u, 0xFFFFFFFFu, 0u, 0xFFFFFFFFu};
  const uint32x4_t high_lanes = {0u, 0u, 0xFFFFFFFFu, 0xFFFFFFFFu};
  for (std::int64_t base = 0; base < block_size; base += 4) {
    float32x4_t value = vld1q_f32(values + base);
    float32x4_t swapped = vrev64q_f32(value);
    value = vbslq_f32(
        odd_lanes, vsubq_f32(swapped, value), vaddq_f32(value, swapped));
    swapped = vextq_f32(value, value, 2);
    value = vbslq_f32(
        high_lanes, vsubq_f32(swapped, value), vaddq_f32(value, swapped));
    vst1q_f32(values + base, value);
  }
  for (std::int64_t half = 4; half < block_size; half *= 2) {
    for (std::int64_t base = 0; base < block_size; base += 2 * half) {
      for (std::int64_t offset = 0; offset + 4 <= half; offset += 4) {
        const float32x4_t left = vld1q_f32(values + base + offset);
        const float32x4_t right = vld1q_f32(values + base + half + offset);
        vst1q_f32(values + base + offset, vaddq_f32(left, right));
        vst1q_f32(values + base + half + offset, vsubq_f32(left, right));
      }
    }
  }
}
#endif

#if (defined(__x86_64__) || defined(_M_X64)) && !defined(_MSC_VER)
ORBITQUANT_TARGET_AVX2 void fwht_block_avx2(
    float *values,
    std::int64_t block_size) {
  if (block_size < 8) {
    fwht_block_portable(values, block_size);
    return;
  }
  for (std::int64_t base = 0; base < block_size; base += 8) {
    __m256 value = _mm256_loadu_ps(values + base);
    __m256 swapped = _mm256_permute_ps(value, 0xB1);
    value = _mm256_blend_ps(
        _mm256_add_ps(value, swapped), _mm256_sub_ps(swapped, value), 0xAA);
    swapped = _mm256_permute_ps(value, 0x4E);
    value = _mm256_blend_ps(
        _mm256_add_ps(value, swapped), _mm256_sub_ps(swapped, value), 0xCC);
    swapped = _mm256_permute2f128_ps(value, value, 0x01);
    value = _mm256_blend_ps(
        _mm256_add_ps(value, swapped), _mm256_sub_ps(swapped, value), 0xF0);
    _mm256_storeu_ps(values + base, value);
  }
  for (std::int64_t half = 8; half < block_size; half *= 2) {
    for (std::int64_t base = 0; base < block_size; base += 2 * half) {
      for (std::int64_t offset = 0; offset + 8 <= half; offset += 8) {
        const __m256 left = _mm256_loadu_ps(values + base + offset);
        const __m256 right =
            _mm256_loadu_ps(values + base + half + offset);
        _mm256_storeu_ps(values + base + offset, _mm256_add_ps(left, right));
        _mm256_storeu_ps(
            values + base + half + offset,
            _mm256_sub_ps(left, right));
      }
    }
  }
}

ORBITQUANT_TARGET_AVX512 void fwht_block_avx512(
    float *values,
    std::int64_t block_size) {
  if (block_size < 16) {
    fwht_block_portable(values, block_size);
    return;
  }
  // Butterfly widths 1-8 stay inside one 16-float register: swap the paired
  // lanes with shuffles and blend the +/- results, one pass over memory
  // instead of four.
  for (std::int64_t base = 0; base < block_size; base += 16) {
    __m512 value = _mm512_loadu_ps(values + base);
    __m512 swapped = _mm512_permute_ps(value, 0xB1);
    value = _mm512_mask_blend_ps(
        0xAAAA, _mm512_add_ps(value, swapped), _mm512_sub_ps(swapped, value));
    swapped = _mm512_permute_ps(value, 0x4E);
    value = _mm512_mask_blend_ps(
        0xCCCC, _mm512_add_ps(value, swapped), _mm512_sub_ps(swapped, value));
    swapped = _mm512_shuffle_f32x4(value, value, 0xB1);
    value = _mm512_mask_blend_ps(
        0xF0F0, _mm512_add_ps(value, swapped), _mm512_sub_ps(swapped, value));
    swapped = _mm512_shuffle_f32x4(value, value, 0x4E);
    value = _mm512_mask_blend_ps(
        0xFF00, _mm512_add_ps(value, swapped), _mm512_sub_ps(swapped, value));
    _mm512_storeu_ps(values + base, value);
  }
  for (std::int64_t half = 16; half < block_size; half *= 2) {
    for (std::int64_t base = 0; base < block_size; base += 2 * half) {
      for (std::int64_t offset = 0; offset + 16 <= half; offset += 16) {
        const __m512 left = _mm512_loadu_ps(values + base + offset);
        const __m512 right =
            _mm512_loadu_ps(values + base + half + offset);
        _mm512_storeu_ps(values + base + offset, _mm512_add_ps(left, right));
        _mm512_storeu_ps(
            values + base + half + offset,
            _mm512_sub_ps(left, right));
      }
    }
  }
}
#endif

inline void fwht_block(
    float *values,
    std::int64_t block_size,
    ActivationIsa isa) {
#if defined(__x86_64__) || defined(_M_X64)
  if (isa == ActivationIsa::Avx512) {
#if defined(_MSC_VER)
    orbitquant::cpu::activation_fwht_msvc_avx2(values, block_size);
#else
    fwht_block_avx512(values, block_size);
#endif
    return;
  }
  if (isa == ActivationIsa::Avx2) {
#if defined(_MSC_VER)
    orbitquant::cpu::activation_fwht_msvc_avx2(values, block_size);
#else
    fwht_block_avx2(values, block_size);
#endif
    return;
  }
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
  if (isa == ActivationIsa::Neon) {
    fwht_block_neon(values, block_size);
    return;
  }
#endif
  fwht_block_portable(values, block_size);
}

template <typename scalar_t>
float squared_norm_scalar(void const *data, std::int64_t offset, std::int64_t dim) {
  float result = 0.0f;
  for (std::int64_t index = 0; index < dim; ++index) {
    const float value = load_scalar<scalar_t>(data, offset + index);
    result += value * value;
  }
  return result;
}

#if defined(__aarch64__) || defined(_M_ARM64)
template <typename scalar_t>
float squared_norm_neon(void const *data, std::int64_t offset, std::int64_t dim) {
  float32x4_t accumulator = vdupq_n_f32(0.0f);
  std::int64_t index = 0;
  for (; index + 4 <= dim; index += 4) {
    float32x4_t values;
    if constexpr (std::is_same_v<scalar_t, float>) {
      values = vld1q_f32(static_cast<float const *>(data) + offset + index);
    } else if constexpr (std::is_same_v<scalar_t, c10::Half>) {
      const auto *source = reinterpret_cast<float16_t const *>(
          static_cast<std::uint16_t const *>(data) + offset + index);
      values = vcvt_f32_f16(vld1_f16(source));
    } else {
      const uint16x4_t raw = vld1_u16(
          static_cast<std::uint16_t const *>(data) + offset + index);
      values = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(raw), 16));
    }
    accumulator = vfmaq_f32(accumulator, values, values);
  }
  float result = vaddvq_f32(accumulator);
  for (; index < dim; ++index) {
    const float value = load_scalar<scalar_t>(data, offset + index);
    result += value * value;
  }
  return result;
}
#endif

#if (defined(__x86_64__) || defined(_M_X64)) && !defined(_MSC_VER)
template <typename scalar_t>
ORBITQUANT_TARGET_AVX2 float squared_norm_avx2(
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

template <typename scalar_t>
ORBITQUANT_TARGET_AVX512 float squared_norm_avx512(
    void const *data,
    std::int64_t offset,
    std::int64_t dim) {
  __m512 accumulator = _mm512_setzero_ps();
  std::int64_t index = 0;
  for (; index + 16 <= dim; index += 16) {
    __m512 values;
    if constexpr (std::is_same_v<scalar_t, float>) {
      values = _mm512_loadu_ps(
          static_cast<float const *>(data) + offset + index);
    } else if constexpr (std::is_same_v<scalar_t, c10::Half>) {
      const auto *source =
          static_cast<std::uint16_t const *>(data) + offset + index;
      values = _mm512_cvtph_ps(
          _mm256_loadu_si256(reinterpret_cast<__m256i const *>(source)));
    } else {
      const auto *source =
          static_cast<std::uint16_t const *>(data) + offset + index;
      const __m256i packed =
          _mm256_loadu_si256(reinterpret_cast<__m256i const *>(source));
      values = _mm512_castsi512_ps(
          _mm512_slli_epi32(_mm512_cvtepu16_epi32(packed), 16));
    }
    accumulator = _mm512_fmadd_ps(values, values, accumulator);
  }
  float result = _mm512_reduce_add_ps(accumulator);
  for (; index < dim; ++index) {
    const float value = load_scalar<scalar_t>(data, offset + index);
    result += value * value;
  }
  return result;
}
#endif

template <typename scalar_t>
float squared_norm(
    void const *data,
    std::int64_t offset,
    std::int64_t dim,
    ActivationIsa isa) {
#if defined(__x86_64__) || defined(_M_X64)
  if (isa == ActivationIsa::Avx512) {
#if defined(_MSC_VER)
    constexpr auto scalar_kind = std::is_same_v<scalar_t, float>
        ? orbitquant::cpu::ScalarKind::Float32
        : std::is_same_v<scalar_t, c10::Half>
        ? orbitquant::cpu::ScalarKind::Float16
        : orbitquant::cpu::ScalarKind::BFloat16;
    return orbitquant::cpu::activation_squared_norm_msvc_avx2(
        data, scalar_kind, offset, dim);
#else
    return squared_norm_avx512<scalar_t>(data, offset, dim);
#endif
  }
  if (isa == ActivationIsa::Avx2) {
#if defined(_MSC_VER)
    constexpr auto scalar_kind = std::is_same_v<scalar_t, float>
        ? orbitquant::cpu::ScalarKind::Float32
        : std::is_same_v<scalar_t, c10::Half>
        ? orbitquant::cpu::ScalarKind::Float16
        : orbitquant::cpu::ScalarKind::BFloat16;
    return orbitquant::cpu::activation_squared_norm_msvc_avx2(
        data, scalar_kind, offset, dim);
#else
    return squared_norm_avx2<scalar_t>(data, offset, dim);
#endif
  }
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
  if (isa == ActivationIsa::Neon) {
    return squared_norm_neon<scalar_t>(data, offset, dim);
  }
#endif
  return squared_norm_scalar<scalar_t>(data, offset, dim);
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

template <typename scalar_t>
void quantize_lookup_scalar(
    ActivationArgs const &args,
    float const *scratch,
    std::int64_t output_offset,
    float norm,
    std::int64_t start = 0) {
  for (std::int64_t index = start; index < args.dim; ++index) {
    const float direction = scratch[index] * args.inv_sqrt_block;
    const std::int64_t centroid_index =
        nearest_centroid(direction, args.boundaries, args.boundary_count);
    store_scalar<scalar_t>(
        args.out,
        output_offset + index,
        args.centroids[centroid_index] * norm);
  }
}

#if (defined(__x86_64__) || defined(_M_X64)) && !defined(_MSC_VER)
ORBITQUANT_TARGET_AVX2 inline __m128i float8_to_bfloat8(__m256 values) {
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
  packed = _mm_blendv_epi8(
      packed,
      _mm_set1_epi16(0x7fc0),
      packed_nan_mask);
  return packed;
}

ORBITQUANT_TARGET_AVX512 inline __m256i float16_to_bfloat16(__m512 values) {
  const __m512i bits = _mm512_castps_si512(values);
  const __m512i absolute_bits =
      _mm512_and_si512(bits, _mm512_set1_epi32(0x7fffffff));
  const __mmask16 nan_mask = _mm512_cmp_epu32_mask(
      absolute_bits,
      _mm512_set1_epi32(0x7f800000),
      _MM_CMPINT_GT);
  const __m512i rounding = _mm512_add_epi32(
      _mm512_set1_epi32(0x7fff),
      _mm512_and_si512(_mm512_srli_epi32(bits, 16), _mm512_set1_epi32(1)));
  const __m512i upper = _mm512_srli_epi32(
      _mm512_add_epi32(bits, rounding),
      16);
  return _mm256_mask_mov_epi16(
      _mm512_cvtepi32_epi16(upper),
      nan_mask,
      _mm256_set1_epi16(0x7fc0));
}

template <typename scalar_t>
ORBITQUANT_TARGET_AVX2 void quantize_lookup_avx2(
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
        _mm256_i32gather_ps(args.centroids, centroid_indices, 4), output_norm);
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
  quantize_lookup_scalar<scalar_t>(args, scratch, output_offset, norm, index);
}

template <typename scalar_t>
ORBITQUANT_TARGET_AVX512 void quantize_lookup_avx512(
    ActivationArgs const &args,
    float const *scratch,
    std::int64_t output_offset,
    float norm) {
  const __m512 inverse_sqrt_block = _mm512_set1_ps(args.inv_sqrt_block);
  const __m512 output_norm = _mm512_set1_ps(norm);
  const __m512i ones = _mm512_set1_epi32(1);
  std::int64_t index = 0;
  for (; index + 16 <= args.dim; index += 16) {
    const __m512 direction = _mm512_mul_ps(
        _mm512_loadu_ps(scratch + index), inverse_sqrt_block);
    __m512i centroid_indices = _mm512_setzero_si512();
    for (std::int64_t boundary = 0; boundary < args.boundary_count; ++boundary) {
      const __mmask16 comparison = _mm512_cmp_ps_mask(
          direction,
          _mm512_set1_ps(args.boundaries[boundary]),
          _CMP_NLE_UQ);
      centroid_indices = _mm512_mask_add_epi32(
          centroid_indices,
          comparison,
          centroid_indices,
          ones);
    }
    const __m512 output = _mm512_mul_ps(
        _mm512_i32gather_ps(centroid_indices, args.centroids, 4),
        output_norm);
    if constexpr (std::is_same_v<scalar_t, float>) {
      _mm512_storeu_ps(
          static_cast<float *>(args.out) + output_offset + index,
          output);
    } else if constexpr (std::is_same_v<scalar_t, c10::Half>) {
      _mm256_storeu_si256(
          reinterpret_cast<__m256i *>(
              static_cast<std::uint16_t *>(args.out) + output_offset + index),
          _mm512_cvtps_ph(output, _MM_FROUND_TO_NEAREST_INT));
    } else {
      _mm256_storeu_si256(
          reinterpret_cast<__m256i *>(
              static_cast<std::uint16_t *>(args.out) + output_offset + index),
          float16_to_bfloat16(output));
    }
  }
  quantize_lookup_scalar<scalar_t>(args, scratch, output_offset, norm, index);
}
#endif

template <typename scalar_t>
void quantize_lookup(
    ActivationArgs const &args,
    float const *scratch,
    std::int64_t output_offset,
    float norm,
    ActivationIsa isa) {
#if defined(__x86_64__) || defined(_M_X64)
  if (isa == ActivationIsa::Avx512) {
#if defined(_MSC_VER)
    orbitquant::cpu::activation_quantize_lookup_msvc_avx2(
        args, scratch, output_offset, norm);
#else
    quantize_lookup_avx512<scalar_t>(args, scratch, output_offset, norm);
#endif
    return;
  }
  if (isa == ActivationIsa::Avx2) {
#if defined(_MSC_VER)
    orbitquant::cpu::activation_quantize_lookup_msvc_avx2(
        args, scratch, output_offset, norm);
#else
    quantize_lookup_avx2<scalar_t>(args, scratch, output_offset, norm);
#endif
    return;
  }
#endif
  quantize_lookup_scalar<scalar_t>(args, scratch, output_offset, norm);
}

template <typename scalar_t, typename index_t>
void quantize_activation_range(
    ActivationArgs const &args,
    index_t const *permutation,
    std::int64_t row_start,
    std::int64_t row_end) {
  thread_local std::vector<float> scratch;
  if (scratch.size() < static_cast<std::size_t>(args.dim)) {
    scratch.resize(static_cast<std::size_t>(args.dim));
  }
  for (std::int64_t row = row_start; row < row_end; ++row) {
    const std::int64_t input_offset = row * args.dim;
    const float norm_squared =
        squared_norm<scalar_t>(args.x, input_offset, args.dim, args.isa);
    const float norm = std::sqrt(norm_squared);
    const float inverse_norm = 1.0f / (norm + args.eps);
    for (std::int64_t index = 0; index < args.dim; ++index) {
      const float value = load_scalar<scalar_t>(
          args.x,
          input_offset + static_cast<std::int64_t>(permutation[index]));
      scratch[index] = value * static_cast<float>(args.signs[index]) * inverse_norm;
    }
    for (std::int64_t block = 0; block < args.dim; block += args.block_size) {
      fwht_block(scratch.data() + block, args.block_size, args.isa);
    }
    quantize_lookup<scalar_t>(
        args,
        scratch.data(),
        input_offset,
        norm,
        args.isa);
  }
}

template <typename index_t>
void quantize_activation_dispatch_indexed(
    ActivationArgs const &args,
    index_t const *permutation,
    std::int64_t row_start,
    std::int64_t row_end) {
  switch (args.scalar_kind) {
    case orbitquant::cpu::ScalarKind::Float32:
      quantize_activation_range<float>(args, permutation, row_start, row_end);
      return;
    case orbitquant::cpu::ScalarKind::Float16:
      quantize_activation_range<c10::Half>(args, permutation, row_start, row_end);
      return;
    case orbitquant::cpu::ScalarKind::BFloat16:
      quantize_activation_range<c10::BFloat16>(args, permutation, row_start, row_end);
      return;
  }
}

void quantize_activation_dispatch(
    ActivationArgs const &args,
    std::int64_t row_start,
    std::int64_t row_end) {
  if (args.permutation_i32 != nullptr) {
    quantize_activation_dispatch_indexed(
        args, args.permutation_i32, row_start, row_end);
    return;
  }
  quantize_activation_dispatch_indexed(args, args.permutation, row_start, row_end);
}

void parallel_quantize_activations(ActivationArgs const &args) {
  const std::int64_t values = args.rows * args.dim;
  const int threads = values < 16'384
      ? 1
      : std::max<int>(
            1,
            std::min<std::int64_t>(
                orbitquant::cpu::requested_threads(),
                args.rows));
  if (threads == 1) {
    quantize_activation_dispatch(args, 0, args.rows);
    return;
  }

  std::vector<std::pair<std::int64_t, std::int64_t>> ranges;
  ranges.reserve(threads);
  const std::int64_t rows_per_thread = (args.rows + threads - 1) / threads;
  for (int thread = 0; thread < threads; ++thread) {
    const std::int64_t start = thread * rows_per_thread;
    const std::int64_t end = std::min(args.rows, start + rows_per_thread);
    if (start >= end) {
      break;
    }
    ranges.emplace_back(start, end);
  }
  orbitquant::cpu::run_ranges(
      ranges,
      [&args](std::int64_t start, std::int64_t end) {
        quantize_activation_dispatch(args, start, end);
      });
}

}  // namespace

void quantize_activations_cpu(
    OrbitQuantTensor &out,
    OrbitQuantTensor const &x,
    OrbitQuantTensor const &permutation,
    OrbitQuantTensor const &signs,
    OrbitQuantTensor const &centroids,
    OrbitQuantTensor const &boundaries,
    double eps,
    double inv_sqrt_block,
    int64_t block_size) {
  using torch::headeronly::DeviceType;
  using torch::headeronly::ScalarType;

  STD_TORCH_CHECK(x.device().type() == DeviceType::CPU, "x must be a CPU tensor");
  STD_TORCH_CHECK(out.device().type() == DeviceType::CPU, "out must be a CPU tensor");
  STD_TORCH_CHECK(
      permutation.device().type() == DeviceType::CPU,
      "permutation must be a CPU tensor");
  STD_TORCH_CHECK(signs.device().type() == DeviceType::CPU, "signs must be a CPU tensor");
  STD_TORCH_CHECK(
      centroids.device().type() == DeviceType::CPU,
      "centroids must be a CPU tensor");
  STD_TORCH_CHECK(
      boundaries.device().type() == DeviceType::CPU,
      "boundaries must be a CPU tensor");
  STD_TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  STD_TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  STD_TORCH_CHECK(permutation.is_contiguous(), "permutation must be contiguous");
  STD_TORCH_CHECK(signs.is_contiguous(), "signs must be contiguous");
  STD_TORCH_CHECK(centroids.is_contiguous(), "centroids must be contiguous");
  STD_TORCH_CHECK(boundaries.is_contiguous(), "boundaries must be contiguous");
  STD_TORCH_CHECK(out.scalar_type() == x.scalar_type(), "out dtype must match x dtype");
  STD_TORCH_CHECK(
      permutation.scalar_type() == ScalarType::Long ||
          permutation.scalar_type() == ScalarType::Int,
      "permutation must be int32 or int64");
  STD_TORCH_CHECK(signs.scalar_type() == ScalarType::Char, "signs must be int8");
  STD_TORCH_CHECK(centroids.scalar_type() == ScalarType::Float, "centroids must be float32");
  STD_TORCH_CHECK(boundaries.scalar_type() == ScalarType::Float, "boundaries must be float32");
  STD_TORCH_CHECK(x.dim() == 2, "x must be rank 2");
  STD_TORCH_CHECK(out.dim() == 2, "out must be rank 2");
  STD_TORCH_CHECK(out.size(0) == x.size(0), "out row count must match x");
  STD_TORCH_CHECK(out.size(1) == x.size(1), "out dimension must match x");
  const int64_t dim = x.size(1);
  STD_TORCH_CHECK(permutation.numel() == dim, "permutation must match the input dimension");
  STD_TORCH_CHECK(signs.numel() == dim, "signs must match the input dimension");
  STD_TORCH_CHECK(
      centroids.numel() == boundaries.numel() + 1,
      "centroids must contain exactly one more value than boundaries");
  STD_TORCH_CHECK(centroids.numel() >= 2, "at least two centroids are required");
  STD_TORCH_CHECK(block_size > 0, "block_size must be positive");
  STD_TORCH_CHECK(
      (block_size & (block_size - 1)) == 0,
      "block_size must be a power of two");
  STD_TORCH_CHECK(dim % block_size == 0, "block_size must divide the input dimension");
  STD_TORCH_CHECK(eps >= 0.0, "eps must be non-negative");
  STD_TORCH_CHECK(inv_sqrt_block > 0.0, "inv_sqrt_block must be positive");

  const bool int32_permutation = permutation.scalar_type() == ScalarType::Int;
  const std::int64_t *permutation_values =
      int32_permutation ? nullptr : permutation.const_data_ptr<std::int64_t>();
  const std::int32_t *permutation_values_i32 =
      int32_permutation ? permutation.const_data_ptr<std::int32_t>() : nullptr;
  const auto *sign_values = signs.const_data_ptr<std::int8_t>();
  const auto permutation_at = [&](std::int64_t index) -> std::int64_t {
    return int32_permutation
        ? static_cast<std::int64_t>(permutation_values_i32[index])
        : permutation_values[index];
  };
  // The permutation/sign buffers are immutable module constants, so cache a
  // fingerprint of validated buffers instead of re-scanning them per forward.
  struct ValidatedEntry {
    void const *permutation;
    void const *signs;
    std::int64_t dim;
    std::int64_t head;
    std::int64_t tail;
  };
  static std::mutex validated_mutex;
  static std::array<ValidatedEntry, 16> validated_entries{};
  static std::size_t validated_cursor = 0;
  const ValidatedEntry candidate{
      int32_permutation ? static_cast<void const *>(permutation_values_i32)
                        : static_cast<void const *>(permutation_values),
      sign_values,
      dim,
      permutation_at(0),
      permutation_at(dim - 1)};
  bool already_validated = false;
  {
    std::lock_guard<std::mutex> lock(validated_mutex);
    for (auto const &entry : validated_entries) {
      if (entry.permutation == candidate.permutation &&
          entry.signs == candidate.signs && entry.dim == candidate.dim &&
          entry.head == candidate.head && entry.tail == candidate.tail) {
        already_validated = true;
        break;
      }
    }
  }
  if (!already_validated) {
    for (int64_t index = 0; index < dim; ++index) {
      const std::int64_t source_index = permutation_at(index);
      STD_TORCH_CHECK(
          source_index >= 0 && source_index < dim,
          "permutation contains an out-of-range index");
      STD_TORCH_CHECK(
          sign_values[index] == -1 || sign_values[index] == 1,
          "signs must contain only -1 and 1");
    }
    std::lock_guard<std::mutex> lock(validated_mutex);
    validated_entries[validated_cursor % validated_entries.size()] = candidate;
    ++validated_cursor;
  }
  if (x.numel() == 0) {
    return;
  }

  const ActivationArgs args{
      out.mutable_data_ptr(),
      x.const_data_ptr(),
      permutation_values,
      permutation_values_i32,
      sign_values,
      centroids.const_data_ptr<float>(),
      boundaries.const_data_ptr<float>(),
      activation_scalar_kind(x),
      select_activation_isa(),
      x.size(0),
      dim,
      boundaries.numel(),
      block_size,
      static_cast<float>(eps),
      static_cast<float>(inv_sqrt_block),
  };
  parallel_quantize_activations(args);
}
