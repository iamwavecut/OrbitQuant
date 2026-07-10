#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def(
      "matmul_packed_weight(Tensor! out, Tensor x, Tensor packed_weight_indices, "
      "Tensor row_norms, Tensor centroids, Tensor bias, bool has_bias, int bits, "
      "int out_features, int in_features, int block_m, int block_n, int block_k) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("matmul_packed_weight", torch::kCUDA, &matmul_packed_weight);
  ops.def(
      "matmul_packed_w4a4_int8(Tensor! out, Tensor packed_activations, "
      "Tensor packed_weight_indices, Tensor token_norms, Tensor row_norms, "
      "Tensor activation_codes, Tensor weight_codes, Tensor bias, bool has_bias, "
      "float activation_scale, float weight_scale, int out_features, "
      "int in_features, int tile_m, int tile_n, bool async_packed, "
      "bool weight_k_major) -> ()");
  ops.impl("matmul_packed_w4a4_int8", torch::kCUDA, &matmul_packed_w4a4_int8);
  ops.def(
      "quantize_activations_packed_w4(Tensor! packed_out, Tensor! norms_out, "
      "Tensor x, Tensor permutation, Tensor signs, Tensor boundaries, float eps, "
      "float inv_sqrt_block, int threads) -> ()");
  ops.impl(
      "quantize_activations_packed_w4",
      torch::kCUDA,
      &quantize_activations_packed_w4);
  ops.def(
      "quantize_activations_int8(Tensor! int8_out, Tensor! norms_out, Tensor x, "
      "Tensor permutation, Tensor signs, Tensor boundaries, Tensor codes, float eps, "
      "float inv_sqrt_block, int threads) -> ()");
  ops.impl(
      "quantize_activations_int8",
      torch::kCUDA,
      &quantize_activations_int8);
#elif defined(METAL_KERNEL)
  ops.impl("matmul_packed_weight", torch::kMPS, &matmul_packed_weight);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
