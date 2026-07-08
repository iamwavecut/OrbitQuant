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
#elif defined(METAL_KERNEL)
  ops.impl("matmul_packed_weight", torch::kMPS, &matmul_packed_weight);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
