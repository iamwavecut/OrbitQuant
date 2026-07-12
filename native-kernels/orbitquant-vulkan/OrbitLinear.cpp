#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/DispatchNode.h>
#include <executorch/backends/vulkan/runtime/graph/ops/PrepackNode.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

#include <cmath>
#include <cstdint>
#include <string>

namespace vkcompute {
namespace {

constexpr uint32_t kNormWorkers = 64u;
constexpr uint32_t kRpbhWorkers = 64u;
constexpr uint32_t kMatmulPrefillLocalX = 16u;
constexpr uint32_t kMatmulPrefillLocalY = 4u;
constexpr uint32_t kMatmulPrefillOutputTileM = 8u;
constexpr uint32_t kMatmulPrefillOutputTileN = 2u;

ValueRef prepack_buffer(ComputeGraph& graph, const ValueRef data) {
  return prepack_standard(
      graph, data, utils::kBuffer, utils::kWidthPacked, false);
}

ValueRef prepack_w4_transposed(
    ComputeGraph& graph,
    const ValueRef data,
    const int64_t N,
    const int64_t K) {
  const int64_t words_per_row = K / 8;
  const ValueRef packed = graph.add_tensor(
      {N * words_per_row},
      vkapi::kInt,
      utils::kBuffer,
      utils::kWidthPacked);
  const utils::uvec3 global_workgroup_size = {
      utils::safe_downcast<uint32_t>(N),
      utils::safe_downcast<uint32_t>(words_per_row),
      1u};
  const utils::ivec4 problem_sizes = {
      0,
      utils::safe_downcast<int32_t>(N),
      utils::safe_downcast<int32_t>(K),
      0};
  graph.prepack_nodes().emplace_back(new PrepackNode(
      graph,
      VK_KERNEL_FROM_STR("orbitquant_pack_w4_transposed"),
      global_workgroup_size,
      {8u, 8u, 1u},
      data,
      packed,
      {},
      {},
      {PushConstantDataInfo(&problem_sizes, sizeof(problem_sizes))}));
  return packed;
}

std::string dtype_shader_name(
    std::string kernel_name,
    const vkapi::ScalarType dtype) {
  add_dtype_suffix(kernel_name, dtype);
  return kernel_name;
}

void check_float_constant(ComputeGraph& graph, const ValueRef ref) {
  VK_CHECK_COND(graph.dtype_of(ref) == vkapi::kFloat);
}

int64_t logical_numel(ComputeGraph& graph, const ValueRef ref) {
  int64_t numel = 1;
  for (const int64_t size : graph.sizes_of(ref)) {
    numel *= size;
  }
  return numel;
}

} // namespace

void orbitquant_linear_w4a4_exact(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  int32_t idx = 0;
  const ValueRef input = args.at(idx++);
  const ValueRef packed_weight_data = args.at(idx++);
  const ValueRef row_norms_data = args.at(idx++);
  const ValueRef permutation_data = args.at(idx++);
  const ValueRef signs_data = args.at(idx++);
  const ValueRef activation_boundaries_data = args.at(idx++);
  const ValueRef pair_lut_data = args.at(idx++);
  const ValueRef block_size_ref = args.at(idx++);
  const ValueRef eps_ref = args.at(idx++);
  const ValueRef bias_data = args.at(idx++);
  const ValueRef output = args.at(idx);

  const vkapi::ScalarType input_dtype = graph.dtype_of(input);
  VK_CHECK_COND(
      input_dtype == vkapi::kFloat || input_dtype == vkapi::kHalf,
      "OrbitQuant Vulkan W4A4 requires float32 or float16 activations");
  VK_CHECK_COND(graph.dtype_of(output) == input_dtype);
  VK_CHECK_COND(graph.dtype_of(packed_weight_data) == vkapi::kByte);
  VK_CHECK_COND(graph.dtype_of(permutation_data) == vkapi::kInt);
  VK_CHECK_COND(graph.dtype_of(signs_data) == vkapi::kInt);
  check_float_constant(graph, row_norms_data);
  check_float_constant(graph, activation_boundaries_data);
  check_float_constant(graph, pair_lut_data);

  const std::vector<int64_t> input_sizes = graph.sizes_of(input);
  VK_CHECK_COND(!input_sizes.empty());
  const int64_t K = input_sizes.back();
  const int64_t M = graph.numel_of(input) / K;
  const int64_t N = logical_numel(graph, row_norms_data);
  const int64_t block_size = graph.extract_scalar<int64_t>(block_size_ref);
  const float eps = graph.extract_scalar<float>(eps_ref);

  VK_CHECK_COND(K > 0 && M > 0 && N > 0);
  VK_CHECK_COND(K % 8 == 0, "OrbitQuant Vulkan W4A4 requires K divisible by 8");
  VK_CHECK_COND(
      block_size >= 8 && block_size <= 4096 &&
          (block_size & (block_size - 1)) == 0 && K % block_size == 0,
      "OrbitQuant Vulkan W4A4 requires a power-of-two block_size in "
      "[8, 4096] dividing K");
  VK_CHECK_COND(
      logical_numel(graph, packed_weight_data) == N * K / 2,
      "packed W4 weight length does not match N*K/2");
  VK_CHECK_COND(logical_numel(graph, permutation_data) == K);
  VK_CHECK_COND(logical_numel(graph, signs_data) == K);
  VK_CHECK_COND(logical_numel(graph, activation_boundaries_data) == 15);
  VK_CHECK_COND(logical_numel(graph, pair_lut_data) == 256);
  VK_CHECK_COND(graph.numel_of(output) == M * N);

  const ValueRef packed_weight =
      prepack_w4_transposed(graph, packed_weight_data, N, K);
  const ValueRef row_norms = prepack_buffer(graph, row_norms_data);
  const ValueRef permutation = prepack_buffer(graph, permutation_data);
  const ValueRef signs = prepack_buffer(graph, signs_data);
  const ValueRef activation_boundaries =
      prepack_buffer(graph, activation_boundaries_data);
  const ValueRef pair_lut = prepack_buffer(graph, pair_lut_data);

  TmpTensor token_norms(
      &graph, {M}, vkapi::kFloat, utils::kBuffer, utils::kWidthPacked);
  TmpTensor packed_activations(
      &graph,
      {M * K / 8},
      vkapi::kInt,
      utils::kBuffer,
      utils::kWidthPacked);

  const utils::ivec4 problem_sizes = {
      utils::safe_downcast<int32_t>(M),
      utils::safe_downcast<int32_t>(N),
      utils::safe_downcast<int32_t>(K),
      utils::safe_downcast<int32_t>(block_size)};

  graph.execute_nodes().emplace_back(new DispatchNode(
      graph,
      VK_KERNEL_FROM_STR(dtype_shader_name("orbitquant_token_norm", input_dtype)),
      {kNormWorkers, utils::safe_downcast<uint32_t>(M), 1u},
      {kNormWorkers, 1u, 1u},
      {{token_norms.vref, vkapi::kWrite}, {input, vkapi::kRead}},
      {},
      {PushConstantDataInfo(&problem_sizes, sizeof(problem_sizes))}));

  const float inv_sqrt_block = 1.0f / std::sqrt(static_cast<float>(block_size));
  std::string rpbh_kernel =
      "orbitquant_rpbh_w4_b" + std::to_string(block_size);
  graph.execute_nodes().emplace_back(new DispatchNode(
      graph,
      VK_KERNEL_FROM_STR(dtype_shader_name(rpbh_kernel, input_dtype)),
      {utils::safe_downcast<uint32_t>(K / block_size) * kRpbhWorkers,
       utils::safe_downcast<uint32_t>(M),
       1u},
      {kRpbhWorkers, 1u, 1u},
      {{packed_activations.vref, vkapi::kWrite},
       {{input,
         token_norms.vref,
         permutation,
         signs,
         activation_boundaries},
        vkapi::kRead}},
      {},
      {PushConstantDataInfo(&problem_sizes, sizeof(problem_sizes)),
       PushConstantDataInfo(&eps, sizeof(eps)),
       PushConstantDataInfo(&inv_sqrt_block, sizeof(inv_sqrt_block))}));

  TmpTensor dummy_bias(
      &graph, {}, vkapi::kFloat, utils::kBuffer, utils::kWidthPacked);
  ValueRef bias = dummy_bias.vref;
  int32_t apply_bias = 0;
  if (graph.val_is_not_none(bias_data)) {
    check_float_constant(graph, bias_data);
    VK_CHECK_COND(logical_numel(graph, bias_data) == N);
    bias = prepack_buffer(graph, bias_data);
    apply_bias = 1;
  }

  std::string matmul_kernel = "orbitquant_w4a4_lut";
  utils::uvec3 matmul_global = {
      utils::safe_downcast<uint32_t>(N),
      utils::safe_downcast<uint32_t>(M),
      1u};
  utils::uvec3 matmul_local = {64u, 1u, 1u};
  if (M >= 8) {
    matmul_kernel = "orbitquant_w4a4_lut_tiled_t8";
    matmul_global = {
        utils::safe_downcast<uint32_t>(
            (N + kMatmulPrefillOutputTileN - 1u) /
            kMatmulPrefillOutputTileN),
        utils::safe_downcast<uint32_t>(
            (M + kMatmulPrefillOutputTileM - 1u) /
            kMatmulPrefillOutputTileM),
        1u};
    matmul_local = {kMatmulPrefillLocalX, kMatmulPrefillLocalY, 1u};
  }

  graph.execute_nodes().emplace_back(new DispatchNode(
      graph,
      VK_KERNEL_FROM_STR(dtype_shader_name(matmul_kernel, input_dtype)),
      matmul_global,
      matmul_local,
      {{output, vkapi::kWrite},
       {{packed_activations.vref,
         packed_weight,
         token_norms.vref,
         row_norms,
         pair_lut,
         bias},
        vkapi::kRead}},
      {},
      {PushConstantDataInfo(&problem_sizes, sizeof(problem_sizes)),
       PushConstantDataInfo(&apply_bias, sizeof(apply_bias))}));
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(
      orbitquant_vulkan.linear_w4a4_exact.default,
      orbitquant_linear_w4a4_exact);
}

} // namespace vkcompute
