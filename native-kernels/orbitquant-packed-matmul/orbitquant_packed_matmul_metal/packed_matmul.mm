#include "../torch-ext/torch_binding.h"

#include <torch/mps.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#ifdef EMBEDDED_METALLIB_HEADER
#include EMBEDDED_METALLIB_HEADER
#else
#error "EMBEDDED_METALLIB_HEADER not defined"
#endif

#include <algorithm>
#include <cstdint>
#include <mutex>

struct PackedMatmulParams {
  int64_t rows;
  int64_t out_features;
  int64_t in_features;
  int64_t bits;
  int64_t block_k;
  int32_t has_bias;
};

static inline id<MTLBuffer> getMTLBufferStorage(const torch::Tensor &tensor) {
  return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

struct PackedMatmulPipelineCache {
  id<MTLDevice> device;
  id<MTLComputePipelineState> float_pipeline;
  id<MTLComputePipelineState> half_pipeline;
  id<MTLComputePipelineState> bfloat16_pipeline;
  id<MTLComputePipelineState> half_scalar_pipeline;
  id<MTLComputePipelineState> bfloat16_scalar_pipeline;
  id<MTLComputePipelineState> float_small_rows_pipeline;
  id<MTLComputePipelineState> half_small_rows_pipeline;
  id<MTLComputePipelineState> bfloat16_small_rows_pipeline;
};

static id<MTLComputePipelineState> create_pipeline(
    id<MTLDevice> device,
    id<MTLLibrary> library,
    char const *kernel_name) {
  NSError *error = nil;
  id<MTLFunction> function =
      [library newFunctionWithName:[NSString stringWithUTF8String:kernel_name]];
  TORCH_CHECK(function, "Failed to create Metal function ", kernel_name);
  id<MTLComputePipelineState> pipeline =
      [device newComputePipelineStateWithFunction:function error:&error];
  TORCH_CHECK(pipeline, "Failed to create Metal pipeline ", kernel_name, ": ",
              error.localizedDescription.UTF8String);
  return pipeline;
}

static PackedMatmulPipelineCache &packed_matmul_pipeline_cache() {
  static std::once_flag once;
  static PackedMatmulPipelineCache cache{};
  std::call_once(once, [] {
    @autoreleasepool {
      NSError *error = nil;
      cache.device = MTLCreateSystemDefaultDevice();
      TORCH_CHECK(cache.device, "Failed to create Metal device");
      id<MTLLibrary> library =
          EMBEDDED_METALLIB_NAMESPACE::createLibrary(cache.device, &error);
      TORCH_CHECK(library, "Failed to create Metal library: ",
                  error.localizedDescription.UTF8String);
      cache.float_pipeline =
          create_pipeline(cache.device, library, "packed_matmul_forward_float");
      cache.half_pipeline =
          create_pipeline(cache.device, library, "packed_matmul_forward_half");
      cache.bfloat16_pipeline =
          create_pipeline(cache.device, library, "packed_matmul_forward_bfloat16");
      cache.half_scalar_pipeline =
          create_pipeline(cache.device, library, "packed_matmul_forward_half_scalar");
      cache.bfloat16_scalar_pipeline =
          create_pipeline(cache.device, library, "packed_matmul_forward_bfloat16_scalar");
      cache.float_small_rows_pipeline =
          create_pipeline(cache.device, library, "packed_matmul_forward_float_small_rows");
      cache.half_small_rows_pipeline =
          create_pipeline(cache.device, library, "packed_matmul_forward_half_small_rows");
      cache.bfloat16_small_rows_pipeline = create_pipeline(
          cache.device, library, "packed_matmul_forward_bfloat16_small_rows");
    }
  });
  return cache;
}

static id<MTLComputePipelineState> select_packed_matmul_pipeline(
    PackedMatmulPipelineCache &cache,
    c10::ScalarType dtype,
    bool use_padded_mma,
    bool use_small_rows) {
  if (use_small_rows) {
    if (dtype == torch::kFloat) {
      return cache.float_small_rows_pipeline;
    }
    if (dtype == torch::kHalf) {
      return cache.half_small_rows_pipeline;
    }
    return cache.bfloat16_small_rows_pipeline;
  }
  if (dtype == torch::kFloat) {
    return cache.float_pipeline;
  }
  if (dtype == torch::kHalf) {
    return use_padded_mma ? cache.half_pipeline : cache.half_scalar_pipeline;
  }
  return use_padded_mma ? cache.bfloat16_pipeline : cache.bfloat16_scalar_pipeline;
}

static void dispatch_packed_matmul_kernel(
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
  @autoreleasepool {
    const bool use_padded_mma =
        x.scalar_type() != torch::kFloat && x.size(0) > 1 &&
        out_features % 32 == 0 && in_features % 32 == 0 &&
        (bits == 2 || bits == 3 || bits == 4 || bits == 6);
    const bool use_small_rows = x.size(0) <= 8 && !use_padded_mma;
    PackedMatmulPipelineCache &cache = packed_matmul_pipeline_cache();
    id<MTLComputePipelineState> pipeline =
        select_packed_matmul_pipeline(
            cache, x.scalar_type(), use_padded_mma, use_small_rows);

    id<MTLCommandBuffer> command_buffer = torch::mps::get_command_buffer();
    TORCH_CHECK(command_buffer, "Failed to retrieve MPS command buffer");
    dispatch_queue_t queue = torch::mps::get_dispatch_queue();

    PackedMatmulParams params{
        x.size(0),
        out_features,
        in_features,
        bits,
        block_k,
        has_bias ? 1 : 0,
    };

    dispatch_sync(queue, ^() {
      id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
      TORCH_CHECK(encoder, "Failed to create Metal compute encoder");
      [encoder setComputePipelineState:pipeline];
      [encoder setBuffer:getMTLBufferStorage(out)
                  offset:out.storage_offset() * out.element_size()
                 atIndex:0];
      [encoder setBuffer:getMTLBufferStorage(x)
                  offset:x.storage_offset() * x.element_size()
                 atIndex:1];
      [encoder setBuffer:getMTLBufferStorage(packed_weight_indices)
                  offset:packed_weight_indices.storage_offset() * packed_weight_indices.element_size()
                 atIndex:2];
      [encoder setBuffer:getMTLBufferStorage(row_norms)
                  offset:row_norms.storage_offset() * row_norms.element_size()
                 atIndex:3];
      [encoder setBuffer:getMTLBufferStorage(centroids)
                  offset:centroids.storage_offset() * centroids.element_size()
                 atIndex:4];
      [encoder setBuffer:getMTLBufferStorage(bias)
                  offset:bias.storage_offset() * bias.element_size()
                 atIndex:5];
      [encoder setBytes:&params length:sizeof(params) atIndex:6];

      if (use_small_rows) {
        constexpr NSUInteger channels_per_threadgroup = 4;
        constexpr NSUInteger threads = 32;
        MTLSize threadgroups = MTLSizeMake(
            (out_features + channels_per_threadgroup - 1) /
                channels_per_threadgroup,
            x.size(0),
            1);
        MTLSize threadgroup_size = MTLSizeMake(threads, 1, 1);
        [encoder dispatchThreadgroups:threadgroups
            threadsPerThreadgroup:threadgroup_size];
      } else if (!use_padded_mma) {
        const NSUInteger threads_x =
            std::min<int64_t>(std::max<int64_t>(block_n, 1), 32);
        const NSUInteger threads_y = std::min<int64_t>(
            x.size(0), std::min<int64_t>(std::max<int64_t>(block_m, 1), 32));
        const NSUInteger tile_k =
            std::min<int64_t>(std::max<int64_t>(block_k, 1), 128);
        const NSUInteger shared_bytes =
            (threads_y * tile_k + tile_k * threads_x) * sizeof(float);
        [encoder setThreadgroupMemoryLength:shared_bytes atIndex:0];
        MTLSize grid_size = MTLSizeMake(out_features, x.size(0), 1);
        MTLSize threadgroup_size = MTLSizeMake(threads_x, threads_y, 1);
        [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
      } else {
        constexpr NSUInteger tile_m = 32;
        constexpr NSUInteger tile_n = 32;
        constexpr NSUInteger padded_k = 40;
        constexpr NSUInteger threads = 128;
        const NSUInteger scalar_bytes = x.element_size();
        const NSUInteger shared_bytes =
            2 * tile_m * padded_k * scalar_bytes;
        TORCH_CHECK(
            threads <= pipeline.maxTotalThreadsPerThreadgroup,
            "Metal packed matmul pipeline supports only ",
            pipeline.maxTotalThreadsPerThreadgroup,
            " threads per threadgroup, but ",
            threads,
            " are required");
        [encoder setThreadgroupMemoryLength:shared_bytes atIndex:0];
        MTLSize threadgroups = MTLSizeMake(
            (out_features + tile_n - 1) / tile_n,
            (x.size(0) + tile_m - 1) / tile_m,
            1);
        MTLSize threadgroup_size = MTLSizeMake(threads, 1, 1);
        [encoder dispatchThreadgroups:threadgroups
            threadsPerThreadgroup:threadgroup_size];
      }
      [encoder endEncoding];
      torch::mps::commit();
    });
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
  TORCH_CHECK(x.device().is_mps(), "x must be an MPS tensor");
  TORCH_CHECK(out.device().is_mps(), "out must be an MPS tensor");
  TORCH_CHECK(packed_weight_indices.device().is_mps(), "packed weights must be MPS tensors");
  TORCH_CHECK(row_norms.device().is_mps(), "row norms must be MPS tensors");
  TORCH_CHECK(centroids.device().is_mps(), "centroids must be MPS tensors");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(packed_weight_indices.is_contiguous(), "packed weights must be contiguous");
  TORCH_CHECK(row_norms.is_contiguous(), "row norms must be contiguous");
  TORCH_CHECK(centroids.is_contiguous(), "centroids must be contiguous");
  TORCH_CHECK(packed_weight_indices.scalar_type() == torch::kUInt8, "packed weights must be uint8");
  TORCH_CHECK(x.scalar_type() == torch::kFloat || x.scalar_type() == torch::kHalf ||
                  x.scalar_type() == torch::kBFloat16,
              "Metal packed matmul supports float32, float16, and bfloat16 inputs");
  TORCH_CHECK(out.scalar_type() == x.scalar_type(), "out dtype must match x dtype");
  TORCH_CHECK(row_norms.scalar_type() == torch::kFloat, "row_norms must be float32");
  TORCH_CHECK(centroids.scalar_type() == torch::kFloat, "centroids must be float32");
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
    TORCH_CHECK(bias.device().is_mps(), "bias must be an MPS tensor");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(bias.scalar_type() == torch::kFloat, "bias must be float32");
    TORCH_CHECK(bias.numel() == out_features, "bias must match out_features");
  }
  if (x.numel() == 0 || out_features == 0) {
    return;
  }
  dispatch_packed_matmul_kernel(
      out,
      x,
      packed_weight_indices,
      row_norms,
      centroids,
      bias,
      has_bias,
      bits,
      out_features,
      in_features,
      block_m,
      block_n,
      block_k);
}
