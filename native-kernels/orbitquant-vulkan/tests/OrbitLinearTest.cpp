#include "utils.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace executorch::vulkan::prototyping;

namespace {

struct TestShape {
  int32_t rows;
  int32_t outputs;
  int32_t input_dim;
  int32_t block_size;
  const char* name;
};

constexpr TestShape kSmallShape = {3, 7, 24, 8, "small"};
constexpr TestShape kDit1536Shape = {2, 9, 1536, 512, "dit1536"};
constexpr TestShape kTiledPartialShape = {9, 17, 24, 8, "tiled_partial"};
constexpr TestShape kDit1536TiledShape = {8, 9, 1536, 512, "dit1536_tiled"};
constexpr TestShape kDit1536DecodeShape = {
    1, 1536, 1536, 512, "dit1536_decode"};
constexpr TestShape kDit1536PrefillShape = {
    32, 1536, 1536, 512, "dit1536_prefill"};
constexpr TestShape kDit3072PrefillShape = {
    32, 3072, 3072, 1024, "dit3072_prefill"};

bool profile_enabled() {
  const char* value = std::getenv("ORBITQUANT_VULKAN_PROFILE");
  return value != nullptr && std::string(value) == "1";
}

std::string profile_case_filter() {
  const char* value = std::getenv("ORBITQUANT_VULKAN_PROFILE_CASE");
  return value == nullptr ? std::string() : std::string(value);
}

float percentile(std::vector<float> values, const float quantile) {
  if (values.empty()) {
    return 0.0f;
  }
  std::sort(values.begin(), values.end());
  const size_t index = static_cast<size_t>(
      std::ceil(quantile * static_cast<float>(values.size())) - 1.0f);
  return values.at(std::min(index, values.size() - 1));
}

void print_profile_percentiles(const TestResult& result) {
  for (const BenchmarkResult& benchmark : result.get_results()) {
    std::cout << "ORBITQUANT_PROFILE case=" << benchmark.get_kernel_name()
              << " median_us="
              << percentile(benchmark.get_iter_timings(), 0.5f)
              << " p95_us="
              << percentile(benchmark.get_iter_timings(), 0.95f)
              << std::endl;
    for (const ShaderTiming& shader : benchmark.get_shader_timings()) {
      std::cout << "ORBITQUANT_PROFILE case=" << benchmark.get_kernel_name()
                << " shader=" << shader.shader_name << " median_us="
                << percentile(shader.iter_timings_us, 0.5f) << " p95_us="
                << percentile(shader.iter_timings_us, 0.95f) << std::endl;
    }
  }
}

ValueSpec tensor_spec(
    const std::vector<int64_t>& sizes,
    const vkapi::ScalarType dtype,
    const bool constant) {
  ValueSpec spec(
      sizes,
      dtype,
      utils::kBuffer,
      utils::kWidthPacked,
      DataGenType::ZEROS);
  if (constant) {
    spec.set_constant(true);
  } else {
    spec.ensure_data_generated();
  }
  return spec;
}

TestCase make_test_case(
    const TestShape shape,
    const vkapi::ScalarType input_dtype,
    const bool has_bias) {
  const std::string dtype_name =
      input_dtype == vkapi::kHalf ? "f16" : "f32";
  TestCase test_case(
      std::string(shape.name) + "_" + dtype_name +
      (has_bias ? "_bias" : "_no_bias"));
  test_case.set_operator_name("orbitquant_vulkan.linear_w4a4_exact.default");
  const float tolerance = input_dtype == vkapi::kHalf ? 8.0e-3f : 2.0e-4f;
  test_case.set_abs_tolerance(tolerance);
  test_case.set_rel_tolerance(tolerance);
  test_case.set_op_invocations_per_execute(1);

  ValueSpec input =
      tensor_spec({shape.rows, shape.input_dim}, input_dtype, false);
  const int32_t first_nonzero_row = shape.rows == 1 ? 0 : 1;
  for (int32_t row = first_nonzero_row; row < shape.rows; ++row) {
    for (int32_t k = 0; k < shape.input_dim; ++k) {
      const float value = std::sin(float(row * 19 + k * 7) * 0.13f);
      const int32_t index = row * shape.input_dim + k;
      if (input_dtype == vkapi::kHalf) {
        input.get_half_data()[index] = float_to_half(value);
      } else {
        input.get_float_data()[index] = value;
      }
    }
  }

  ValueSpec packed_weight =
      tensor_spec({shape.outputs, shape.input_dim / 2}, vkapi::kByte, true);
  for (int32_t n = 0; n < shape.outputs; ++n) {
    for (int32_t k = 0; k < shape.input_dim; k += 2) {
      const uint8_t low = static_cast<uint8_t>((n * 3 + k * 5) & 15);
      const uint8_t high =
          static_cast<uint8_t>((n * 3 + (k + 1) * 5) & 15);
      packed_weight.get_uint8_data()[n * (shape.input_dim / 2) + k / 2] =
          static_cast<uint8_t>(low | (high << 4));
    }
  }

  ValueSpec row_norms = tensor_spec({shape.outputs}, vkapi::kFloat, true);
  for (int32_t n = 0; n < shape.outputs; ++n) {
    row_norms.get_float_data()[n] = 0.55f + float(n % 17) * 0.007f;
  }

  ValueSpec permutation = tensor_spec({shape.input_dim}, vkapi::kInt, true);
  ValueSpec signs = tensor_spec({shape.input_dim}, vkapi::kInt, true);
  for (int32_t k = 0; k < shape.input_dim; ++k) {
    permutation.get_int32_data()[k] = (k * 5) % shape.input_dim;
    signs.get_int32_data()[k] = (k % 3 == 0) ? -1 : 1;
  }

  std::vector<float> activation_centroids(16);
  std::vector<float> weight_centroids(16);
  for (int32_t index = 0; index < 16; ++index) {
    activation_centroids[index] = -0.75f + float(index) * 0.1f;
    weight_centroids[index] = -0.9f + float(index) * 0.12f;
  }

  ValueSpec boundaries = tensor_spec({15}, vkapi::kFloat, true);
  for (int32_t index = 0; index < 15; ++index) {
    boundaries.get_float_data()[index] =
        0.5f * (activation_centroids[index] + activation_centroids[index + 1]);
  }

  ValueSpec pair_lut = tensor_spec({256}, vkapi::kFloat, true);
  for (int32_t activation = 0; activation < 16; ++activation) {
    for (int32_t weight = 0; weight < 16; ++weight) {
      pair_lut.get_float_data()[activation * 16 + weight] =
          activation_centroids[activation] * weight_centroids[weight];
    }
  }

  ValueSpec bias = tensor_spec({shape.outputs}, vkapi::kFloat, true);
  if (has_bias) {
    for (int32_t n = 0; n < shape.outputs; ++n) {
      bias.get_float_data()[n] = float(n - 3) * 0.025f;
    }
  } else {
    bias.set_none(true);
  }

  test_case.add_input_spec(input);
  test_case.add_input_spec(packed_weight);
  test_case.add_input_spec(row_norms);
  test_case.add_input_spec(permutation);
  test_case.add_input_spec(signs);
  test_case.add_input_spec(boundaries);
  test_case.add_input_spec(pair_lut);
  test_case.add_input_spec(ValueSpec(shape.block_size));
  test_case.add_input_spec(ValueSpec(1.0e-10f));
  test_case.add_input_spec(bias);
  test_case.add_output_spec(
      tensor_spec({shape.rows, shape.outputs}, input_dtype, false));
  return test_case;
}

TestCase make_fp16_linear_test_case(const TestShape shape) {
  TestCase test_case(std::string("fp16_linear_") + shape.name);
  test_case.set_operator_name("aten.linear.default");
  test_case.set_abs_tolerance(2.0e-2f);
  test_case.set_rel_tolerance(2.0e-2f);
  test_case.set_op_invocations_per_execute(1);

  ValueSpec input =
      tensor_spec({shape.rows, shape.input_dim}, vkapi::kHalf, false);
  for (int32_t row = 0; row < shape.rows; ++row) {
    for (int32_t k = 0; k < shape.input_dim; ++k) {
      const float value = std::sin(float(row * 19 + k * 7) * 0.013f);
      input.get_half_data()[row * shape.input_dim + k] =
          float_to_half(value);
    }
  }

  ValueSpec weight = tensor_spec(
      {shape.outputs, shape.input_dim}, vkapi::kHalf, true);
  for (int32_t n = 0; n < shape.outputs; ++n) {
    for (int32_t k = 0; k < shape.input_dim; ++k) {
      const float value =
          std::sin(float(n * 13 + k * 11) * 0.017f) * 0.01f;
      weight.get_half_data()[n * shape.input_dim + k] =
          float_to_half(value);
    }
  }

  ValueSpec bias = tensor_spec({shape.outputs}, vkapi::kHalf, true);
  for (int32_t n = 0; n < shape.outputs; ++n) {
    bias.get_half_data()[n] = float_to_half(float(n % 17 - 8) * 0.001f);
  }

  test_case.add_input_spec(input);
  test_case.add_input_spec(weight);
  test_case.add_input_spec(bias);
  test_case.add_output_spec(
      tensor_spec({shape.rows, shape.outputs}, vkapi::kHalf, false));
  return test_case;
}

std::vector<TestCase> generate_test_cases() {
  std::vector<TestCase> test_cases = {
      make_test_case(kSmallShape, vkapi::kFloat, false),
      make_test_case(kSmallShape, vkapi::kFloat, true),
      make_test_case(kSmallShape, vkapi::kHalf, false),
      make_test_case(kSmallShape, vkapi::kHalf, true),
      make_test_case(kDit1536Shape, vkapi::kFloat, false),
      make_test_case(kTiledPartialShape, vkapi::kHalf, true),
      make_test_case(kDit1536TiledShape, vkapi::kHalf, true),
  };
  if (profile_enabled()) {
    const std::string filter = profile_case_filter();
    if (!filter.empty()) {
      test_cases.clear();
      if (filter == "packed_decode1536") {
        test_cases.emplace_back(
            make_test_case(kDit1536DecodeShape, vkapi::kHalf, true));
      } else if (filter == "packed_prefill1536") {
        test_cases.emplace_back(
            make_test_case(kDit1536PrefillShape, vkapi::kHalf, true));
      } else if (filter == "packed_prefill3072") {
        test_cases.emplace_back(
            make_test_case(kDit3072PrefillShape, vkapi::kHalf, true));
      } else if (filter == "fp16_decode1536") {
        test_cases.emplace_back(
            make_fp16_linear_test_case(kDit1536DecodeShape));
      } else if (filter == "fp16_prefill1536") {
        test_cases.emplace_back(
            make_fp16_linear_test_case(kDit1536PrefillShape));
      } else if (filter == "fp16_prefill3072") {
        test_cases.emplace_back(
            make_fp16_linear_test_case(kDit3072PrefillShape));
      } else {
        throw std::invalid_argument(
            "unknown ORBITQUANT_VULKAN_PROFILE_CASE=" + filter);
      }
      return test_cases;
    }
    test_cases.emplace_back(
        make_test_case(kDit1536DecodeShape, vkapi::kHalf, true));
    test_cases.emplace_back(
        make_test_case(kDit1536PrefillShape, vkapi::kHalf, true));
    test_cases.emplace_back(
        make_test_case(kDit3072PrefillShape, vkapi::kHalf, true));
    test_cases.emplace_back(make_fp16_linear_test_case(kDit1536DecodeShape));
    test_cases.emplace_back(make_fp16_linear_test_case(kDit1536PrefillShape));
    test_cases.emplace_back(make_fp16_linear_test_case(kDit3072PrefillShape));
  }
  return test_cases;
}

void fwht(std::vector<float>& values, const int32_t block_size) {
  for (int32_t span = 1; span < block_size; span *= 2) {
    for (int32_t base = 0; base < block_size; base += span * 2) {
      for (int32_t offset = 0; offset < span; ++offset) {
        const float a = values[base + offset];
        const float b = values[base + span + offset];
        values[base + offset] = a + b;
        values[base + span + offset] = a - b;
      }
    }
  }
}

void reference_compute(TestCase& test_case) {
  if (test_case.operator_name() == "aten.linear.default") {
    const ValueSpec& input = test_case.inputs().at(0);
    const ValueSpec& weight = test_case.inputs().at(1);
    const ValueSpec& bias = test_case.inputs().at(2);
    const int32_t M =
        static_cast<int32_t>(input.get_tensor_sizes().at(0));
    const int32_t K =
        static_cast<int32_t>(input.get_tensor_sizes().at(1));
    const int32_t N =
        static_cast<int32_t>(weight.get_tensor_sizes().at(0));
    auto& output = test_case.outputs().at(0).get_ref_float_data();
    output.assign(M * N, 0.0f);
    for (int32_t m = 0; m < M; ++m) {
      for (int32_t n = 0; n < N; ++n) {
        float accumulator = half_to_float(bias.get_half_data().at(n));
        for (int32_t k = 0; k < K; ++k) {
          accumulator +=
              half_to_float(input.get_half_data().at(m * K + k)) *
              half_to_float(weight.get_half_data().at(n * K + k));
        }
        output[m * N + n] = accumulator;
      }
    }
    return;
  }

  const ValueSpec& input_spec = test_case.inputs().at(0);
  const auto& packed_weight = test_case.inputs().at(1).get_uint8_data();
  const auto& row_norms = test_case.inputs().at(2).get_float_data();
  const auto& permutation = test_case.inputs().at(3).get_int32_data();
  const auto& signs = test_case.inputs().at(4).get_int32_data();
  const auto& boundaries = test_case.inputs().at(5).get_float_data();
  const auto& pair_lut = test_case.inputs().at(6).get_float_data();
  const ValueSpec& bias_spec = test_case.inputs().at(9);
  const auto& bias = bias_spec.get_float_data();
  const int32_t block_size = test_case.inputs().at(7).get_int_value();
  const int32_t M = static_cast<int32_t>(input_spec.get_tensor_sizes().at(0));
  const int32_t K = static_cast<int32_t>(input_spec.get_tensor_sizes().at(1));
  const int32_t N = static_cast<int32_t>(row_norms.size());
  auto& output = test_case.outputs().at(0).get_ref_float_data();
  output.assign(M * N, 0.0f);

  const auto input_value = [&input_spec](const int32_t index) {
    if (input_spec.dtype == vkapi::kHalf) {
      return half_to_float(input_spec.get_half_data().at(index));
    }
    return input_spec.get_float_data().at(index);
  };

  std::vector<uint8_t> activation_codes(M * K);
  std::vector<float> token_norms(M);
  for (int32_t row = 0; row < M; ++row) {
    float squared_norm = 0.0f;
    for (int32_t k = 0; k < K; ++k) {
      const float value = input_value(row * K + k);
      squared_norm += value * value;
    }
    token_norms[row] = std::sqrt(squared_norm);
    for (int32_t block = 0; block < K / block_size; ++block) {
      std::vector<float> values(block_size);
      for (int32_t k = 0; k < block_size; ++k) {
        const int32_t rotated_index = block * block_size + k;
        values[k] =
            input_value(row * K + permutation[rotated_index]) *
            signs[rotated_index] /
            (token_norms[row] + 1.0e-10f);
      }
      fwht(values, block_size);
      for (int32_t k = 0; k < block_size; ++k) {
        const float value = values[k] / std::sqrt(float(block_size));
        uint8_t code = 0;
        for (const float boundary : boundaries) {
          code += value > boundary ? 1 : 0;
        }
        activation_codes[row * K + block * block_size + k] = code;
      }
    }
  }

  for (int32_t row = 0; row < M; ++row) {
    for (int32_t n = 0; n < N; ++n) {
      float accumulator = 0.0f;
      for (int32_t k = 0; k < K; ++k) {
        const uint8_t packed = packed_weight[n * (K / 2) + k / 2];
        const uint8_t weight_code =
            (k & 1) == 0 ? packed & 15 : (packed >> 4) & 15;
        accumulator += pair_lut[
            activation_codes[row * K + k] * 16 + weight_code];
      }
      float value = accumulator * token_norms[row] * row_norms[n];
      if (!bias_spec.is_none()) {
        value += bias[n];
      }
      output[row * N + n] = value;
    }
  }
}

int64_t flop_count(const TestCase& test_case) {
  const auto& input_sizes = test_case.inputs().at(0).get_tensor_sizes();
  const int64_t outputs = test_case.inputs().at(2).numel();
  return int64_t{2} * input_sizes.at(0) * outputs * input_sizes.at(1);
}

} // namespace

int main() {
  const bool profile = profile_enabled();
  set_print_output(false);
  set_print_latencies(profile);
  set_use_gpu_timestamps(true);
  try {
    api::context()->initialize_querypool();
  } catch (const std::exception& error) {
    std::cerr << "Failed to initialize Vulkan: " << error.what() << std::endl;
    return 1;
  }

  const TestResult result = execute_test_cases(
      generate_test_cases,
      flop_count,
      "OrbitQuant exact W4A4",
      profile ? 10 : 1,
      profile ? 31 : 3,
      reference_compute);
  if (profile) {
    result.print_detailed_results();
    print_profile_percentiles(result);
  }
  return result.get_correctness_passed() ? 0 : 1;
}
