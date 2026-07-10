#pragma once

#include <torch/torch.h>

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
    int64_t block_k);

#if defined(CUDA_KERNEL)
void matmul_packed_w4a4_int8(
    torch::Tensor &out,
    torch::Tensor const &packed_activations,
    torch::Tensor const &packed_weight_indices,
    torch::Tensor const &token_norms,
    torch::Tensor const &row_norms,
    torch::Tensor const &activation_codes,
    torch::Tensor const &weight_codes,
    torch::Tensor const &bias,
    bool has_bias,
    double activation_scale,
    double weight_scale,
    int64_t out_features,
    int64_t in_features,
    int64_t tile_m,
    int64_t tile_n,
    bool async_packed,
    bool weight_k_major);

void quantize_activations_packed_w4(
    torch::Tensor &packed_out,
    torch::Tensor &norms_out,
    torch::Tensor const &x,
    torch::Tensor const &permutation,
    torch::Tensor const &signs,
    torch::Tensor const &boundaries,
    double eps,
    double inv_sqrt_block,
    int64_t threads);

void quantize_activations_int8(
    torch::Tensor &int8_out,
    torch::Tensor &norms_out,
    torch::Tensor const &x,
    torch::Tensor const &permutation,
    torch::Tensor const &signs,
    torch::Tensor const &boundaries,
    torch::Tensor const &codes,
    double eps,
    double inv_sqrt_block,
    int64_t threads);
#endif
