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
