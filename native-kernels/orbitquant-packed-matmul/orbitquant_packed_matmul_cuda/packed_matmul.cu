#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "../torch-ext/torch_binding.h"

#include <algorithm>
#include <cstdint>

template <typename scalar_t>
__global__ void orbitquant_packed_matmul_kernel(
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
    int64_t bits) {
  const int64_t row = blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows || col >= out_features) {
    return;
  }

  const uint32_t mask = (1u << bits) - 1u;
  float acc = has_bias ? bias[col] : 0.0f;
  const float row_norm = row_norms[col];
  for (int64_t k = 0; k < in_features; ++k) {
    const int64_t value_offset = col * in_features + k;
    const int64_t bit_start = value_offset * bits;
    const int64_t byte_index = bit_start >> 3;
    const int64_t bit_offset = bit_start & 7;
    uint32_t raw = packed_weight_indices[byte_index];
    if (bit_offset + bits > 8) {
      raw |= static_cast<uint32_t>(packed_weight_indices[byte_index + 1]) << 8;
    }
    const uint32_t index = (raw >> bit_offset) & mask;
    const float weight = row_norm * centroids[index];
    acc += static_cast<float>(x[row * in_features + k]) * weight;
  }

  out[row * out_features + col] = static_cast<scalar_t>(acc);
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
  TORCH_CHECK(x.dim() == 2, "x must be rank 2");
  TORCH_CHECK(out.dim() == 2, "out must be rank 2");
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
    TORCH_CHECK(bias.numel() == out_features, "bias must match out_features");
  }
  if (x.numel() == 0 || out_features == 0) {
    return;
  }

  const int threads_n = static_cast<int>(std::min<int64_t>(std::max<int64_t>(block_n, 1), 32));
  const int threads_m =
      static_cast<int>(std::min<int64_t>(std::max<int64_t>(block_m, 1), 1024 / threads_n));
  const dim3 block(threads_n, threads_m);
  const dim3 grid(
      (out_features + threads_n - 1) / threads_n,
      (x.size(0) + threads_m - 1) / threads_m);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, x.scalar_type(), "orbitquant_packed_matmul_cuda", [&] {
        orbitquant_packed_matmul_kernel<scalar_t><<<grid, block, 0, stream>>>(
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
            bits);
      });
}
