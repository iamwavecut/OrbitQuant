#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "../torch-ext/torch_binding.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <algorithm>
#include <cstdint>

using namespace nvcuda;

__device__ __forceinline__ uint32_t unpack_lowbit_index(
    uint8_t const *__restrict__ packed_weight_indices,
    int64_t value_offset,
    int64_t bits,
    uint32_t mask) {
  const int64_t bit_start = value_offset * bits;
  const int64_t byte_index = bit_start >> 3;
  const int64_t bit_offset = bit_start & 7;
  uint32_t raw = packed_weight_indices[byte_index];
  if (bit_offset + bits > 8) {
    raw |= static_cast<uint32_t>(packed_weight_indices[byte_index + 1]) << 8;
  }
  return (raw >> bit_offset) & mask;
}

template <int Bits>
__device__ __forceinline__ uint32_t unpack_lowbit_index_const(
    uint8_t const *__restrict__ packed_weight_indices,
    int64_t value_offset) {
  constexpr uint32_t mask = (1u << Bits) - 1u;
  const int64_t bit_start = value_offset * Bits;
  const int64_t byte_index = bit_start >> 3;
  const int64_t bit_offset = bit_start & 7;
  uint32_t raw = packed_weight_indices[byte_index];
  if (bit_offset + Bits > 8) {
    raw |= static_cast<uint32_t>(packed_weight_indices[byte_index + 1]) << 8;
  }
  return (raw >> bit_offset) & mask;
}

template <int Bits>
__global__ void orbitquant_packed_matmul_wmma_bf16_kernel(
    c10::BFloat16 *__restrict__ out,
    c10::BFloat16 const *__restrict__ x,
    uint8_t const *__restrict__ packed_weight_indices,
    float const *__restrict__ row_norms,
    float const *__restrict__ centroids,
    float const *__restrict__ bias,
    bool has_bias,
    int64_t rows,
    int64_t out_features,
    int64_t in_features) {
  constexpr int tile = 16;
  constexpr int col_tiles = 4;
  constexpr int warps_per_block = 8;
  constexpr int rows_per_block = tile * warps_per_block;
  __shared__ __nv_bfloat16 x_tile[warps_per_block * tile * tile];
  __shared__ __nv_bfloat16 w_tile[col_tiles * tile * tile];
  __shared__ float acc_tile[warps_per_block * col_tiles * tile * tile];

  const int warp_id = threadIdx.x / warpSize;
  const int lane = threadIdx.x & (warpSize - 1);
  const int64_t row_start = blockIdx.y * rows_per_block + warp_id * tile;
  const int64_t col_start = blockIdx.x * tile * col_tiles;
  __nv_bfloat16 *warp_x_tile = x_tile + warp_id * tile * tile;

  wmma::fragment<wmma::matrix_a, tile, tile, tile, __nv_bfloat16, wmma::row_major> a_frag;
  wmma::fragment<wmma::accumulator, tile, tile, tile, float> c_frag[col_tiles];
  for (int col_tile = 0; col_tile < col_tiles; ++col_tile) {
    wmma::fill_fragment(c_frag[col_tile], 0.0f);
  }

  for (int64_t k_start = 0; k_start < in_features; k_start += tile) {
    for (int offset = lane; offset < tile * tile; offset += warpSize) {
      const int local_row = offset / tile;
      const int local_k = offset - local_row * tile;
      const int64_t global_row = row_start + local_row;
      const int64_t global_k = k_start + local_k;
      float value = 0.0f;
      if (global_row < rows && global_k < in_features) {
        value = static_cast<float>(x[global_row * in_features + global_k]);
      }
      warp_x_tile[offset] = __float2bfloat16(value);
    }
    for (int load_col_tile = warp_id; load_col_tile < col_tiles;
         load_col_tile += warps_per_block) {
      __nv_bfloat16 *warp_w_tile = w_tile + load_col_tile * tile * tile;
      const int64_t tile_col_start = col_start + load_col_tile * tile;
      for (int offset = lane; offset < tile * tile; offset += warpSize) {
        const int local_k = offset / tile;
        const int local_col = offset - local_k * tile;
        const int64_t global_k = k_start + local_k;
        const int64_t global_col = tile_col_start + local_col;
        float value = 0.0f;
        if (global_col < out_features && global_k < in_features) {
          const int64_t value_offset = global_col * in_features + global_k;
          const uint32_t index = unpack_lowbit_index_const<Bits>(
              packed_weight_indices, value_offset);
          value = row_norms[global_col] * centroids[index];
        }
        warp_w_tile[offset] = __float2bfloat16(value);
      }
    }
    __syncthreads();

    wmma::load_matrix_sync(a_frag, warp_x_tile, tile);
    for (int col_tile = 0; col_tile < col_tiles; ++col_tile) {
      wmma::fragment<wmma::matrix_b, tile, tile, tile, __nv_bfloat16, wmma::row_major>
          b_frag;
      wmma::load_matrix_sync(b_frag, w_tile + col_tile * tile * tile, tile);
      wmma::mma_sync(c_frag[col_tile], a_frag, b_frag, c_frag[col_tile]);
    }
    __syncthreads();
  }

  for (int col_tile = 0; col_tile < col_tiles; ++col_tile) {
    float *warp_acc_tile = acc_tile + (warp_id * col_tiles + col_tile) * tile * tile;
    wmma::store_matrix_sync(warp_acc_tile, c_frag[col_tile], tile, wmma::mem_row_major);
    __syncwarp();
    for (int offset = lane; offset < tile * tile; offset += warpSize) {
      const int local_row = offset / tile;
      const int local_col = offset - local_row * tile;
      const int64_t global_row = row_start + local_row;
      const int64_t global_col = col_start + col_tile * tile + local_col;
      if (global_row < rows && global_col < out_features) {
        float value = warp_acc_tile[offset];
        if (has_bias) {
          value += bias[global_col];
        }
        out[global_row * out_features + global_col] = static_cast<c10::BFloat16>(value);
      }
    }
  }
}

template <int Bits>
__global__ void orbitquant_packed_matmul_wmma_half_kernel(
    c10::Half *__restrict__ out,
    c10::Half const *__restrict__ x,
    uint8_t const *__restrict__ packed_weight_indices,
    float const *__restrict__ row_norms,
    float const *__restrict__ centroids,
    float const *__restrict__ bias,
    bool has_bias,
    int64_t rows,
    int64_t out_features,
    int64_t in_features) {
  constexpr int tile = 16;
  constexpr int col_tiles = 4;
  constexpr int warps_per_block = 8;
  constexpr int rows_per_block = tile * warps_per_block;
  __shared__ half x_tile[warps_per_block * tile * tile];
  __shared__ half w_tile[col_tiles * tile * tile];
  __shared__ float acc_tile[warps_per_block * col_tiles * tile * tile];

  const int warp_id = threadIdx.x / warpSize;
  const int lane = threadIdx.x & (warpSize - 1);
  const int64_t row_start = blockIdx.y * rows_per_block + warp_id * tile;
  const int64_t col_start = blockIdx.x * tile * col_tiles;
  half *warp_x_tile = x_tile + warp_id * tile * tile;

  wmma::fragment<wmma::matrix_a, tile, tile, tile, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::accumulator, tile, tile, tile, float> c_frag[col_tiles];
  for (int col_tile = 0; col_tile < col_tiles; ++col_tile) {
    wmma::fill_fragment(c_frag[col_tile], 0.0f);
  }

  for (int64_t k_start = 0; k_start < in_features; k_start += tile) {
    for (int offset = lane; offset < tile * tile; offset += warpSize) {
      const int local_row = offset / tile;
      const int local_k = offset - local_row * tile;
      const int64_t global_row = row_start + local_row;
      const int64_t global_k = k_start + local_k;
      float value = 0.0f;
      if (global_row < rows && global_k < in_features) {
        value = static_cast<float>(x[global_row * in_features + global_k]);
      }
      warp_x_tile[offset] = __float2half(value);
    }
    for (int load_col_tile = warp_id; load_col_tile < col_tiles;
         load_col_tile += warps_per_block) {
      half *warp_w_tile = w_tile + load_col_tile * tile * tile;
      const int64_t tile_col_start = col_start + load_col_tile * tile;
      for (int offset = lane; offset < tile * tile; offset += warpSize) {
        const int local_k = offset / tile;
        const int local_col = offset - local_k * tile;
        const int64_t global_k = k_start + local_k;
        const int64_t global_col = tile_col_start + local_col;
        float value = 0.0f;
        if (global_col < out_features && global_k < in_features) {
          const int64_t value_offset = global_col * in_features + global_k;
          const uint32_t index = unpack_lowbit_index_const<Bits>(
              packed_weight_indices, value_offset);
          value = row_norms[global_col] * centroids[index];
        }
        warp_w_tile[offset] = __float2half(value);
      }
    }
    __syncthreads();

    wmma::load_matrix_sync(a_frag, warp_x_tile, tile);
    for (int col_tile = 0; col_tile < col_tiles; ++col_tile) {
      wmma::fragment<wmma::matrix_b, tile, tile, tile, half, wmma::row_major> b_frag;
      wmma::load_matrix_sync(b_frag, w_tile + col_tile * tile * tile, tile);
      wmma::mma_sync(c_frag[col_tile], a_frag, b_frag, c_frag[col_tile]);
    }
    __syncthreads();
  }

  for (int col_tile = 0; col_tile < col_tiles; ++col_tile) {
    float *warp_acc_tile = acc_tile + (warp_id * col_tiles + col_tile) * tile * tile;
    wmma::store_matrix_sync(warp_acc_tile, c_frag[col_tile], tile, wmma::mem_row_major);
    __syncwarp();
    for (int offset = lane; offset < tile * tile; offset += warpSize) {
      const int local_row = offset / tile;
      const int local_col = offset - local_row * tile;
      const int64_t global_row = row_start + local_row;
      const int64_t global_col = col_start + col_tile * tile + local_col;
      if (global_row < rows && global_col < out_features) {
        float value = warp_acc_tile[offset];
        if (has_bias) {
          value += bias[global_col];
        }
        out[global_row * out_features + global_col] = static_cast<c10::Half>(value);
      }
    }
  }
}

template <typename scalar_t>
__global__ void orbitquant_packed_matmul_tiled_kernel(
    scalar_t *__restrict__ out,
    scalar_t const *__restrict__ x,
    uint8_t const *__restrict__ packed_weight_indices,
    float const *__restrict__ row_norms,
    float const *__restrict__ centroids,
    float const *__restrict__ bias,
    bool has_bias,
    int64_t rows,
    int64_t out_features,
    int64_t in_features,
    int64_t bits,
    int64_t block_k) {
  extern __shared__ float shared[];
  float *x_tile = shared;
  float *w_tile = shared + blockDim.y * block_k;

  const int64_t row = blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t col = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t local_row = threadIdx.y;
  const int64_t local_col = threadIdx.x;
  const int64_t thread_linear = threadIdx.y * blockDim.x + threadIdx.x;
  const int64_t thread_count = blockDim.x * blockDim.y;

  const uint32_t mask = (1u << bits) - 1u;
  const bool output_valid = row < rows && col < out_features;
  float acc = output_valid && has_bias ? bias[col] : 0.0f;

  for (int64_t k_start = 0; k_start < in_features; k_start += block_k) {
    const int64_t x_tile_values = blockDim.y * block_k;
    for (int64_t offset = thread_linear; offset < x_tile_values; offset += thread_count) {
      const int64_t tile_row = offset / block_k;
      const int64_t tile_k = offset - tile_row * block_k;
      const int64_t global_row = blockIdx.y * blockDim.y + tile_row;
      const int64_t global_k = k_start + tile_k;
      float value = 0.0f;
      if (global_row < rows && global_k < in_features) {
        value = static_cast<float>(x[global_row * in_features + global_k]);
      }
      x_tile[offset] = value;
    }

    const int64_t w_tile_values = block_k * blockDim.x;
    for (int64_t offset = thread_linear; offset < w_tile_values; offset += thread_count) {
      const int64_t tile_k = offset / blockDim.x;
      const int64_t tile_col = offset - tile_k * blockDim.x;
      const int64_t global_k = k_start + tile_k;
      const int64_t global_col = blockIdx.x * blockDim.x + tile_col;
      float value = 0.0f;
      if (global_col < out_features && global_k < in_features) {
        const int64_t value_offset = global_col * in_features + global_k;
        const uint32_t index =
            unpack_lowbit_index(packed_weight_indices, value_offset, bits, mask);
        value = row_norms[global_col] * centroids[index];
      }
      w_tile[offset] = value;
    }
    __syncthreads();

    if (output_valid) {
      for (int64_t tile_k = 0; tile_k < block_k; ++tile_k) {
        acc += x_tile[local_row * block_k + tile_k] * w_tile[tile_k * blockDim.x + local_col];
      }
    }
    __syncthreads();
  }

  if (output_valid) {
    out[row * out_features + col] = static_cast<scalar_t>(acc);
  }
}

void matmul_packed_weight(
    torch::Tensor &out,
    torch::Tensor const &x,
    torch::Tensor const &packed_weight_indices,
    torch::Tensor const &row_norms,
    torch::Tensor const &centroids,
    torch::Tensor const &bias,
    bool has_bias,
    int64_t bits,
    int64_t out_features,
    int64_t in_features,
    int64_t block_m,
    int64_t block_n,
    int64_t block_k) {
  TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(out.device().is_cuda(), "out must be a CUDA tensor");
  TORCH_CHECK(packed_weight_indices.device().is_cuda(), "packed weights must be CUDA tensors");
  TORCH_CHECK(row_norms.device().is_cuda(), "row norms must be CUDA tensors");
  TORCH_CHECK(centroids.device().is_cuda(), "centroids must be CUDA tensors");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(packed_weight_indices.is_contiguous(), "packed weights must be contiguous");
  TORCH_CHECK(row_norms.is_contiguous(), "row norms must be contiguous");
  TORCH_CHECK(centroids.is_contiguous(), "centroids must be contiguous");
  TORCH_CHECK(packed_weight_indices.scalar_type() == torch::kUInt8, "packed weights must be uint8");
  TORCH_CHECK(row_norms.scalar_type() == torch::kFloat, "row_norms must be float32");
  TORCH_CHECK(centroids.scalar_type() == torch::kFloat, "centroids must be float32");
  TORCH_CHECK(x.dim() == 2, "x must be rank 2");
  TORCH_CHECK(out.dim() == 2, "out must be rank 2");
  TORCH_CHECK(out.scalar_type() == x.scalar_type(), "out dtype must match x dtype");
  TORCH_CHECK(bits > 0 && bits <= 8, "bits must be in [1, 8]");
  TORCH_CHECK(block_m > 0 && block_n > 0 && block_k > 0, "tile sizes must be positive");
  TORCH_CHECK(x.size(1) == in_features, "x has an unexpected input dimension");
  TORCH_CHECK(out.size(0) == x.size(0), "out has an unexpected row count");
  TORCH_CHECK(out.size(1) == out_features, "out has an unexpected output dimension");
  TORCH_CHECK(row_norms.numel() == out_features, "row_norms must match out_features");
  TORCH_CHECK(centroids.numel() >= (1LL << bits), "centroids must contain 2**bits values");
  const int64_t packed_bytes = (out_features * in_features * bits + 7) / 8;
  TORCH_CHECK(packed_weight_indices.numel() >= packed_bytes, "packed weights are too short");
  if (has_bias) {
    TORCH_CHECK(bias.device().is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(bias.scalar_type() == torch::kFloat, "bias must be float32");
    TORCH_CHECK(bias.numel() == out_features, "bias must match out_features");
  }
  if (x.numel() == 0 || out_features == 0) {
    return;
  }

  const int threads_n = static_cast<int>(std::min<int64_t>(std::max<int64_t>(block_n, 1), 64));
  const int threads_m =
      static_cast<int>(std::min<int64_t>(std::max<int64_t>(block_m, 1), 1024 / threads_n));
  const int tile_k = static_cast<int>(std::min<int64_t>(std::max<int64_t>(block_k, 1), 128));
  const dim3 block(threads_n, threads_m);
  const dim3 grid(
      (out_features + threads_n - 1) / threads_n,
      (x.size(0) + threads_m - 1) / threads_m);
  const size_t shared_bytes = static_cast<size_t>(threads_m * tile_k + tile_k * threads_n) *
      sizeof(float);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (x.scalar_type() == at::kBFloat16) {
    constexpr int tile = 16;
    constexpr int col_tiles = 4;
    constexpr int rows_per_block = tile * 8;
    const dim3 wmma_block(256);
    const dim3 wmma_grid(
        (out_features + tile * col_tiles - 1) / (tile * col_tiles),
        (x.size(0) + rows_per_block - 1) / rows_per_block);
#define ORBITQUANT_LAUNCH_BF16(BITS_VALUE)                                             \
  orbitquant_packed_matmul_wmma_bf16_kernel<BITS_VALUE><<<wmma_grid, wmma_block, 0,    \
                                                         stream>>>(                    \
      reinterpret_cast<c10::BFloat16 *>(out.data_ptr()),                                \
      reinterpret_cast<c10::BFloat16 const *>(x.data_ptr()),                            \
      packed_weight_indices.data_ptr<uint8_t>(),                                        \
      row_norms.data_ptr<float>(),                                                      \
      centroids.data_ptr<float>(),                                                      \
      has_bias ? bias.data_ptr<float>() : nullptr,                                      \
      has_bias,                                                                         \
      x.size(0),                                                                        \
      out_features,                                                                     \
      in_features)
    switch (bits) {
      case 1:
        ORBITQUANT_LAUNCH_BF16(1);
        break;
      case 2:
        ORBITQUANT_LAUNCH_BF16(2);
        break;
      case 3:
        ORBITQUANT_LAUNCH_BF16(3);
        break;
      case 4:
        ORBITQUANT_LAUNCH_BF16(4);
        break;
      case 5:
        ORBITQUANT_LAUNCH_BF16(5);
        break;
      case 6:
        ORBITQUANT_LAUNCH_BF16(6);
        break;
      case 7:
        ORBITQUANT_LAUNCH_BF16(7);
        break;
      case 8:
        ORBITQUANT_LAUNCH_BF16(8);
        break;
    }
#undef ORBITQUANT_LAUNCH_BF16
    return;
  }

  if (x.scalar_type() == at::kHalf) {
    constexpr int tile = 16;
    constexpr int col_tiles = 4;
    constexpr int rows_per_block = tile * 8;
    const dim3 wmma_block(256);
    const dim3 wmma_grid(
        (out_features + tile * col_tiles - 1) / (tile * col_tiles),
        (x.size(0) + rows_per_block - 1) / rows_per_block);
#define ORBITQUANT_LAUNCH_HALF(BITS_VALUE)                                             \
  orbitquant_packed_matmul_wmma_half_kernel<BITS_VALUE><<<wmma_grid, wmma_block, 0,    \
                                                         stream>>>(                    \
      reinterpret_cast<c10::Half *>(out.data_ptr()),                                    \
      reinterpret_cast<c10::Half const *>(x.data_ptr()),                                \
      packed_weight_indices.data_ptr<uint8_t>(),                                        \
      row_norms.data_ptr<float>(),                                                      \
      centroids.data_ptr<float>(),                                                      \
      has_bias ? bias.data_ptr<float>() : nullptr,                                      \
      has_bias,                                                                         \
      x.size(0),                                                                        \
      out_features,                                                                     \
      in_features)
    switch (bits) {
      case 1:
        ORBITQUANT_LAUNCH_HALF(1);
        break;
      case 2:
        ORBITQUANT_LAUNCH_HALF(2);
        break;
      case 3:
        ORBITQUANT_LAUNCH_HALF(3);
        break;
      case 4:
        ORBITQUANT_LAUNCH_HALF(4);
        break;
      case 5:
        ORBITQUANT_LAUNCH_HALF(5);
        break;
      case 6:
        ORBITQUANT_LAUNCH_HALF(6);
        break;
      case 7:
        ORBITQUANT_LAUNCH_HALF(7);
        break;
      case 8:
        ORBITQUANT_LAUNCH_HALF(8);
        break;
    }
#undef ORBITQUANT_LAUNCH_HALF
    return;
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, x.scalar_type(), "orbitquant_packed_matmul_cuda", [&] {
        orbitquant_packed_matmul_tiled_kernel<scalar_t><<<grid, block, shared_bytes, stream>>>(
            out.data_ptr<scalar_t>(),
            x.data_ptr<scalar_t>(),
            packed_weight_indices.data_ptr<uint8_t>(),
            row_norms.data_ptr<float>(),
            centroids.data_ptr<float>(),
            has_bias ? bias.data_ptr<float>() : nullptr,
            has_bias,
            x.size(0),
            out_features,
            in_features,
            bits,
            tile_k);
      });
}
