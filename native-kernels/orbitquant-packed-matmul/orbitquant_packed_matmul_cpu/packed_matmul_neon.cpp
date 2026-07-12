#include "packed_matmul_cpu.h"

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>

#include <torch/headeronly/util/BFloat16.h>
#include <torch/headeronly/util/Half.h>

#include <type_traits>
#include <vector>

namespace orbitquant::cpu {
namespace {

inline float horizontal_sum(float32x4_t value) {
#if defined(__aarch64__)
  return vaddvq_f32(value);
#else
  const float32x2_t pair = vadd_f32(vget_low_f32(value), vget_high_f32(value));
  return vget_lane_f32(vpadd_f32(pair, pair), 0);
#endif
}

inline float32x4_t load_float4(void const *data, std::int64_t offset) {
  return vld1q_f32(static_cast<float const *>(data) + offset);
}

inline float32x4_t load_half4(void const *data, std::int64_t offset) {
  const auto *source = reinterpret_cast<float16_t const *>(
      static_cast<std::uint16_t const *>(data) + offset);
  return vcvt_f32_f16(vld1_f16(source));
}

inline float32x4_t load_bfloat4(void const *data, std::int64_t offset) {
  const uint16x4_t raw = vld1_u16(
      static_cast<std::uint16_t const *>(data) + offset);
  return vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(raw), 16));
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
    float32x4_t (*load4)(void const *, std::int64_t),
    int row_tile>
inline void packed_matmul_neon_w4_rows(
    PackedMatmulArgs const &args,
    std::uint8_t const *packed_row,
    std::int64_t out_col,
    std::int64_t row_start) {
  float32x4_t accumulator0[row_tile];
  float32x4_t accumulator1[row_tile];
#pragma clang loop unroll(full)
  for (int row = 0; row < row_tile; ++row) {
    accumulator0[row] = vdupq_n_f32(0.0f);
    accumulator1[row] = vdupq_n_f32(0.0f);
  }

  std::int64_t k = 0;
  for (; k + 8 <= args.in_features; k += 8) {
    const std::int64_t byte_offset = k / 2;
    float weights0[4];
    float weights1[4];
#pragma clang loop unroll(full)
    for (int pair = 0; pair < 4; ++pair) {
      const std::uint8_t packed = packed_row[byte_offset + pair];
      const int value = pair * 2;
      if (value < 4) {
        weights0[value] = args.centroids[packed & 15u];
        weights0[value + 1] = args.centroids[(packed >> 4) & 15u];
      } else {
        weights1[value - 4] = args.centroids[packed & 15u];
        weights1[value - 3] = args.centroids[(packed >> 4) & 15u];
      }
    }
    const float32x4_t weight0 = vld1q_f32(weights0);
    const float32x4_t weight1 = vld1q_f32(weights1);
#pragma clang loop unroll(full)
    for (int row = 0; row < row_tile; ++row) {
      const std::int64_t input_offset =
          (row_start + row) * args.in_features + k;
      accumulator0[row] =
          vfmaq_f32(accumulator0[row], load4(args.x, input_offset), weight0);
      accumulator1[row] = vfmaq_f32(
          accumulator1[row], load4(args.x, input_offset + 4), weight1);
    }
  }

  const float row_norm = args.row_norms[out_col];
#pragma clang loop unroll(full)
  for (int row = 0; row < row_tile; ++row) {
    const std::int64_t input_row_offset =
        (row_start + row) * args.in_features;
    float accumulator =
        horizontal_sum(vaddq_f32(accumulator0[row], accumulator1[row]));
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

template <int Bits>
inline std::uint32_t unpack_index_generic(
    std::uint8_t const *packed_row,
    std::int64_t value_index) {
  const std::int64_t bit_start = value_index * Bits;
  const std::int64_t byte_index = bit_start >> 3;
  const unsigned bit_offset = static_cast<unsigned>(bit_start & 7);
  std::uint32_t raw = packed_row[byte_index];
  if (bit_offset + static_cast<unsigned>(Bits) > 8) {
    raw |= static_cast<std::uint32_t>(packed_row[byte_index + 1]) << 8;
  }
  return (raw >> bit_offset) & ((1u << Bits) - 1u);
}

template <
    typename scalar_t,
    float32x4_t (*load4)(void const *, std::int64_t),
    int Bits,
    int row_tile>
inline void packed_matmul_neon_lowbit_rows(
    PackedMatmulArgs const &args,
    std::uint8_t const *packed_row,
    std::int64_t out_col,
    std::int64_t row_start) {
  float32x4_t accumulator0[row_tile];
  float32x4_t accumulator1[row_tile];
#pragma clang loop unroll(full)
  for (int row = 0; row < row_tile; ++row) {
    accumulator0[row] = vdupq_n_f32(0.0f);
    accumulator1[row] = vdupq_n_f32(0.0f);
  }

  std::int64_t k = 0;
  for (; k + 8 <= args.in_features; k += 8) {
    float weights0[4];
    float weights1[4];
#pragma clang loop unroll(full)
    for (int value = 0; value < 4; ++value) {
      weights0[value] =
          args.centroids[unpack_index_generic<Bits>(packed_row, k + value)];
      weights1[value] =
          args.centroids[unpack_index_generic<Bits>(packed_row, k + 4 + value)];
    }
    const float32x4_t weight0 = vld1q_f32(weights0);
    const float32x4_t weight1 = vld1q_f32(weights1);
#pragma clang loop unroll(full)
    for (int row = 0; row < row_tile; ++row) {
      const std::int64_t input_offset =
          (row_start + row) * args.in_features + k;
      accumulator0[row] =
          vfmaq_f32(accumulator0[row], load4(args.x, input_offset), weight0);
      accumulator1[row] = vfmaq_f32(
          accumulator1[row], load4(args.x, input_offset + 4), weight1);
    }
  }

  const float row_norm = args.row_norms[out_col];
#pragma clang loop unroll(full)
  for (int row = 0; row < row_tile; ++row) {
    const std::int64_t input_row_offset =
        (row_start + row) * args.in_features;
    float accumulator =
        horizontal_sum(vaddq_f32(accumulator0[row], accumulator1[row]));
    for (std::int64_t tail = k; tail < args.in_features; ++tail) {
      const std::uint32_t index =
          unpack_index_generic<Bits>(packed_row, tail);
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

template <typename scalar_t, float32x4_t (*load4)(void const *, std::int64_t), int row_tile>
inline void packed_matmul_neon_buffered_rows(
    PackedMatmulArgs const &args,
    float const *decoded_row,
    std::int64_t out_col,
    std::int64_t row_start) {
  float32x4_t accumulator0[row_tile];
  float32x4_t accumulator1[row_tile];
#pragma clang loop unroll(full)
  for (int row = 0; row < row_tile; ++row) {
    accumulator0[row] = vdupq_n_f32(0.0f);
    accumulator1[row] = vdupq_n_f32(0.0f);
  }

  std::int64_t k = 0;
  for (; k + 8 <= args.in_features; k += 8) {
    const float32x4_t weight0 = vld1q_f32(decoded_row + k);
    const float32x4_t weight1 = vld1q_f32(decoded_row + k + 4);
#pragma clang loop unroll(full)
    for (int row = 0; row < row_tile; ++row) {
      const std::int64_t input_offset =
          (row_start + row) * args.in_features + k;
      accumulator0[row] =
          vfmaq_f32(accumulator0[row], load4(args.x, input_offset), weight0);
      accumulator1[row] = vfmaq_f32(
          accumulator1[row], load4(args.x, input_offset + 4), weight1);
    }
  }

  const float row_norm = args.row_norms[out_col];
#pragma clang loop unroll(full)
  for (int row = 0; row < row_tile; ++row) {
    const std::int64_t input_row_offset =
        (row_start + row) * args.in_features;
    float accumulator =
        horizontal_sum(vaddq_f32(accumulator0[row], accumulator1[row]));
    for (std::int64_t tail = k; tail < args.in_features; ++tail) {
      if constexpr (std::is_same_v<scalar_t, float>) {
        accumulator +=
            static_cast<float const *>(args.x)[input_row_offset + tail] *
            decoded_row[tail];
      } else {
        accumulator += static_cast<float>(
                           static_cast<scalar_t const *>(
                               args.x)[input_row_offset + tail]) *
            decoded_row[tail];
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

template <
    typename scalar_t,
    float32x4_t (*load4)(void const *, std::int64_t),
    int Bits>
void packed_matmul_neon_lowbit_typed(
    PackedMatmulArgs const &args,
    std::int64_t out_start,
    std::int64_t out_end) {
  const std::int64_t packed_row_bytes = args.in_features * Bits / 8;
  // The NEON decode goes through scalar table lookups, so expanding the
  // column once per >=16-row call amortizes the costliest stage.
  const bool use_decoded_buffer = args.rows >= 16;
  thread_local std::vector<float> decoded_row_storage;
  if (use_decoded_buffer &&
      decoded_row_storage.size() < static_cast<std::size_t>(args.in_features)) {
    decoded_row_storage.resize(static_cast<std::size_t>(args.in_features));
  }
  for (std::int64_t out_col = out_start; out_col < out_end; ++out_col) {
    const auto *packed_row =
        args.packed_weight_indices + out_col * packed_row_bytes;
    if (use_decoded_buffer) {
      float *decoded_row = decoded_row_storage.data();
      for (std::int64_t k = 0; k < args.in_features; ++k) {
        decoded_row[k] =
            args.centroids[unpack_index_generic<Bits>(packed_row, k)];
      }
      std::int64_t row = 0;
      for (; row + 8 <= args.rows; row += 8) {
        packed_matmul_neon_buffered_rows<scalar_t, load4, 8>(
            args, decoded_row, out_col, row);
      }
      for (; row + 4 <= args.rows; row += 4) {
        packed_matmul_neon_buffered_rows<scalar_t, load4, 4>(
            args, decoded_row, out_col, row);
      }
      switch (args.rows - row) {
        case 3:
          packed_matmul_neon_buffered_rows<scalar_t, load4, 3>(
              args, decoded_row, out_col, row);
          break;
        case 2:
          packed_matmul_neon_buffered_rows<scalar_t, load4, 2>(
              args, decoded_row, out_col, row);
          break;
        case 1:
          packed_matmul_neon_buffered_rows<scalar_t, load4, 1>(
              args, decoded_row, out_col, row);
          break;
        default:
          break;
      }
      continue;
    }
    std::int64_t row = 0;
    for (; row + 8 <= args.rows; row += 8) {
      packed_matmul_neon_lowbit_rows<scalar_t, load4, Bits, 8>(
          args, packed_row, out_col, row);
    }
    for (; row + 4 <= args.rows; row += 4) {
      packed_matmul_neon_lowbit_rows<scalar_t, load4, Bits, 4>(
          args, packed_row, out_col, row);
    }
    switch (args.rows - row) {
      case 3:
        packed_matmul_neon_lowbit_rows<scalar_t, load4, Bits, 3>(
            args, packed_row, out_col, row);
        break;
      case 2:
        packed_matmul_neon_lowbit_rows<scalar_t, load4, Bits, 2>(
            args, packed_row, out_col, row);
        break;
      case 1:
        packed_matmul_neon_lowbit_rows<scalar_t, load4, Bits, 1>(
            args, packed_row, out_col, row);
        break;
      default:
        break;
    }
  }
}

template <int Bits>
void packed_matmul_neon_lowbit_dispatch(
    PackedMatmulArgs const &args,
    std::int64_t out_start,
    std::int64_t out_end) {
  switch (args.scalar_kind) {
    case ScalarKind::Float32:
      packed_matmul_neon_lowbit_typed<float, load_float4, Bits>(
          args, out_start, out_end);
      return;
    case ScalarKind::Float16:
      packed_matmul_neon_lowbit_typed<c10::Half, load_half4, Bits>(
          args, out_start, out_end);
      return;
    case ScalarKind::BFloat16:
      packed_matmul_neon_lowbit_typed<c10::BFloat16, load_bfloat4, Bits>(
          args, out_start, out_end);
      return;
  }
}

template <typename scalar_t, float32x4_t (*load4)(void const *, std::int64_t)>
void packed_matmul_neon_w4_typed(
    PackedMatmulArgs const &args,
    std::int64_t out_start,
    std::int64_t out_end) {
  constexpr int kPrimaryRowTile = 8;
  const std::int64_t packed_row_bytes = args.in_features / 2;
  for (std::int64_t out_col = out_start; out_col < out_end; ++out_col) {
    const auto *packed_row =
        args.packed_weight_indices + out_col * packed_row_bytes;
    std::int64_t row = 0;
    if constexpr (kPrimaryRowTile == 8) {
      for (; row + 8 <= args.rows; row += 8) {
        packed_matmul_neon_w4_rows<scalar_t, load4, 8>(
            args, packed_row, out_col, row);
      }
    }
    for (; row + 4 <= args.rows; row += 4) {
      packed_matmul_neon_w4_rows<scalar_t, load4, 4>(
          args, packed_row, out_col, row);
    }
    switch (args.rows - row) {
      case 3:
        packed_matmul_neon_w4_rows<scalar_t, load4, 3>(
            args, packed_row, out_col, row);
        break;
      case 2:
        packed_matmul_neon_w4_rows<scalar_t, load4, 2>(
            args, packed_row, out_col, row);
        break;
      case 1:
        packed_matmul_neon_w4_rows<scalar_t, load4, 1>(
            args, packed_row, out_col, row);
        break;
      default:
        break;
    }
  }
}

}  // namespace

bool packed_matmul_neon_available() {
  return true;
}

void packed_matmul_neon_range(
    PackedMatmulArgs const &args,
    std::int64_t out_start,
    std::int64_t out_end) {
  if (args.bits == 4 && args.in_features % 2 == 0 && args.rows >= 16) {
    // The buffered generic path decodes each column once, which beats the
    // per-tile scalar decode as soon as several row tiles reuse it.
    packed_matmul_neon_lowbit_dispatch<4>(args, out_start, out_end);
    return;
  }
  if (args.bits == 2 && args.in_features % 4 == 0) {
    packed_matmul_neon_lowbit_dispatch<2>(args, out_start, out_end);
    return;
  }
  if (args.bits == 3 && args.in_features % 8 == 0) {
    packed_matmul_neon_lowbit_dispatch<3>(args, out_start, out_end);
    return;
  }
  if (args.bits == 6 && args.in_features % 4 == 0) {
    packed_matmul_neon_lowbit_dispatch<6>(args, out_start, out_end);
    return;
  }
  if (args.bits != 4 || args.in_features % 2 != 0) {
    packed_matmul_scalar_range(args, out_start, out_end);
    return;
  }
  switch (args.scalar_kind) {
    case ScalarKind::Float32:
      packed_matmul_neon_w4_typed<float, load_float4>(args, out_start, out_end);
      return;
    case ScalarKind::Float16:
      packed_matmul_neon_w4_typed<c10::Half, load_half4>(args, out_start, out_end);
      return;
    case ScalarKind::BFloat16:
      packed_matmul_neon_w4_typed<c10::BFloat16, load_bfloat4>(
          args, out_start, out_end);
      return;
  }
}

}  // namespace orbitquant::cpu

#else

namespace orbitquant::cpu {

bool packed_matmul_neon_available() {
  return false;
}

void packed_matmul_neon_range(
    PackedMatmulArgs const &args,
    std::int64_t out_start,
    std::int64_t out_end) {
  packed_matmul_scalar_range(args, out_start, out_end);
}

}  // namespace orbitquant::cpu

#endif
