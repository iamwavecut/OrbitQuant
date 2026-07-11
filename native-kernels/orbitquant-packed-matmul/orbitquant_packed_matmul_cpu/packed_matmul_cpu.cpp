#include "cpu_threads.h"
#include "packed_matmul_cpu.h"
#include "../torch-ext/torch_binding.h"

#include <torch/headeronly/core/DeviceType.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

namespace {

orbitquant::cpu::ScalarKind scalar_kind(OrbitQuantTensor const &tensor) {
  using torch::headeronly::ScalarType;
  switch (tensor.scalar_type()) {
    case ScalarType::Float:
      return orbitquant::cpu::ScalarKind::Float32;
    case ScalarType::Half:
      return orbitquant::cpu::ScalarKind::Float16;
    case ScalarType::BFloat16:
      return orbitquant::cpu::ScalarKind::BFloat16;
    default:
      STD_TORCH_CHECK(
          false,
          "CPU packed matmul supports float32, float16, and bfloat16 inputs");
  }
  return orbitquant::cpu::ScalarKind::Float32;
}

void parallel_packed_matmul(
    orbitquant::cpu::PackedMatmulArgs const &args,
    orbitquant::cpu::PackedMatmulRangeFn function) {
  const std::int64_t arithmetic =
      args.rows * args.out_features * args.in_features;
  const int max_threads = orbitquant::cpu::requested_threads();
  const int threads = arithmetic < 1'000'000
      ? 1
      : std::max<int>(
            1,
            std::min<std::int64_t>(max_threads, args.out_features / 16));
  if (threads == 1) {
    function(args, 0, args.out_features);
    return;
  }

  std::vector<std::thread> workers;
  workers.reserve(threads);
  const std::int64_t columns_per_thread =
      (args.out_features + threads - 1) / threads;
  for (int thread = 0; thread < threads; ++thread) {
    const std::int64_t start = thread * columns_per_thread;
    const std::int64_t end = std::min(
        args.out_features,
        start + columns_per_thread);
    if (start >= end) {
      break;
    }
    workers.emplace_back([&args, function, start, end] {
      function(args, start, end);
    });
  }
  for (auto &worker : workers) {
    worker.join();
  }
}

orbitquant::cpu::PackedMatmulRangeFn select_packed_matmul() {
  const char *requested = std::getenv("ORBITQUANT_CPU_ISA");
  if (requested == nullptr || std::strcmp(requested, "auto") == 0) {
    if (orbitquant::cpu::packed_matmul_x86_avx512_available()) {
      return orbitquant::cpu::packed_matmul_x86_avx512_range;
    }
    if (orbitquant::cpu::packed_matmul_x86_avx2_available()) {
      return orbitquant::cpu::packed_matmul_x86_avx2_range;
    }
    if (orbitquant::cpu::packed_matmul_neon_available()) {
      return orbitquant::cpu::packed_matmul_neon_range;
    }
    return orbitquant::cpu::packed_matmul_scalar_range;
  }
  if (std::strcmp(requested, "scalar") == 0) {
    return orbitquant::cpu::packed_matmul_scalar_range;
  }
  if (std::strcmp(requested, "avx2") == 0) {
    STD_TORCH_CHECK(
        orbitquant::cpu::packed_matmul_x86_avx2_available(),
        "ORBITQUANT_CPU_ISA=avx2 requested AVX2/FMA/F16C on an unsupported CPU");
    return orbitquant::cpu::packed_matmul_x86_avx2_range;
  }
  if (std::strcmp(requested, "avx512") == 0) {
    STD_TORCH_CHECK(
        orbitquant::cpu::packed_matmul_x86_avx512_available(),
        "ORBITQUANT_CPU_ISA=avx512 requested AVX-512F/DQ/BW/VL on an unsupported CPU");
    return orbitquant::cpu::packed_matmul_x86_avx512_range;
  }
  if (std::strcmp(requested, "neon") == 0) {
    STD_TORCH_CHECK(
        orbitquant::cpu::packed_matmul_neon_available(),
        "ORBITQUANT_CPU_ISA=neon requested NEON on an unsupported CPU");
    return orbitquant::cpu::packed_matmul_neon_range;
  }
  STD_TORCH_CHECK(
      false,
      "ORBITQUANT_CPU_ISA must be auto, scalar, avx2, avx512, or neon");
  return orbitquant::cpu::packed_matmul_scalar_range;
}

}  // namespace

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
    int64_t block_k) {
  using torch::headeronly::DeviceType;
  using torch::headeronly::ScalarType;

  STD_TORCH_CHECK(x.device().type() == DeviceType::CPU, "x must be a CPU tensor");
  STD_TORCH_CHECK(out.device().type() == DeviceType::CPU, "out must be a CPU tensor");
  STD_TORCH_CHECK(
      packed_weight_indices.device().type() == DeviceType::CPU,
      "packed weights must be CPU tensors");
  STD_TORCH_CHECK(
      row_norms.device().type() == DeviceType::CPU,
      "row norms must be CPU tensors");
  STD_TORCH_CHECK(
      centroids.device().type() == DeviceType::CPU,
      "centroids must be CPU tensors");
  STD_TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  STD_TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  STD_TORCH_CHECK(
      packed_weight_indices.is_contiguous(), "packed weights must be contiguous");
  STD_TORCH_CHECK(row_norms.is_contiguous(), "row norms must be contiguous");
  STD_TORCH_CHECK(centroids.is_contiguous(), "centroids must be contiguous");
  STD_TORCH_CHECK(
      packed_weight_indices.scalar_type() == ScalarType::Byte,
      "packed weights must be uint8");
  STD_TORCH_CHECK(
      row_norms.scalar_type() == ScalarType::Float,
      "row_norms must be float32");
  STD_TORCH_CHECK(
      centroids.scalar_type() == ScalarType::Float,
      "centroids must be float32");
  STD_TORCH_CHECK(out.scalar_type() == x.scalar_type(), "out dtype must match x dtype");
  STD_TORCH_CHECK(x.dim() == 2, "x must be rank 2");
  STD_TORCH_CHECK(out.dim() == 2, "out must be rank 2");
  STD_TORCH_CHECK(bits > 0 && bits <= 8, "bits must be in [1, 8]");
  STD_TORCH_CHECK(block_m > 0 && block_n > 0 && block_k > 0, "tile sizes must be positive");
  STD_TORCH_CHECK(x.size(1) == in_features, "x has an unexpected input dimension");
  STD_TORCH_CHECK(out.size(0) == x.size(0), "out has an unexpected row count");
  STD_TORCH_CHECK(out.size(1) == out_features, "out has an unexpected output dimension");
  STD_TORCH_CHECK(row_norms.numel() == out_features, "row_norms must match out_features");
  STD_TORCH_CHECK(centroids.numel() >= (1LL << bits), "centroids are too short");
  const int64_t packed_bytes = (out_features * in_features * bits + 7) / 8;
  STD_TORCH_CHECK(
      packed_weight_indices.numel() >= packed_bytes,
      "packed weights are too short");
  if (has_bias) {
    STD_TORCH_CHECK(
        bias.device().type() == DeviceType::CPU,
        "bias must be a CPU tensor");
    STD_TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    STD_TORCH_CHECK(
        bias.scalar_type() == ScalarType::Float,
        "bias must be float32");
    STD_TORCH_CHECK(bias.numel() == out_features, "bias must match out_features");
  }
  if (x.numel() == 0 || out_features == 0) {
    return;
  }

  orbitquant::cpu::PackedMatmulArgs args{
      out.mutable_data_ptr(),
      x.const_data_ptr(),
      packed_weight_indices.const_data_ptr<std::uint8_t>(),
      row_norms.const_data_ptr<float>(),
      centroids.const_data_ptr<float>(),
      has_bias ? bias.const_data_ptr<float>() : nullptr,
      has_bias,
      scalar_kind(x),
      x.size(0),
      out_features,
      in_features,
      bits,
  };
  parallel_packed_matmul(args, select_packed_matmul());
}
