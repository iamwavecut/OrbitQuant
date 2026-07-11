#pragma once

#if defined(CPU_KERNEL)
#include <torch/csrc/stable/tensor.h>
using OrbitQuantTensor = torch::stable::Tensor;
#else
#include <torch/torch.h>
using OrbitQuantTensor = torch::Tensor;
#endif

void matmul_packed_weight(
    OrbitQuantTensor &out,
    OrbitQuantTensor const &x,
    OrbitQuantTensor const &packed_weight_indices,
    OrbitQuantTensor const &row_norms,
    OrbitQuantTensor const &centroids,
    OrbitQuantTensor const &bias,
    bool has_bias,
    int64_t bits,
    int64_t out_features,
    int64_t in_features,
    int64_t block_m,
    int64_t block_n,
    int64_t block_k);

#if defined(CPU_KERNEL)
void quantize_activations_cpu(
    OrbitQuantTensor &out,
    OrbitQuantTensor const &x,
    OrbitQuantTensor const &permutation,
    OrbitQuantTensor const &signs,
    OrbitQuantTensor const &centroids,
    OrbitQuantTensor const &boundaries,
    double eps,
    double inv_sqrt_block,
    int64_t block_size);

void matmul_packed_adaln_int4_cpu(
    OrbitQuantTensor &out,
    OrbitQuantTensor const &x,
    OrbitQuantTensor const &packed_weight,
    OrbitQuantTensor const &scales,
    OrbitQuantTensor const &bias,
    bool has_bias,
    int64_t out_features,
    int64_t in_features,
    int64_t group_size);
#endif

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
