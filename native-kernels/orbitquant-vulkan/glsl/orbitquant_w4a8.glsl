#version 450 core

${define_required_extensions("buffer", DTYPE)}

#define PRECISION ${PRECISION}
#define T ${buffer_scalar_type(DTYPE)}

${define_active_storage_type("buffer")}

layout(std430) buffer;

// out[m, n] = scale[m] * row_norm[n] * sum_k code[m, k] * centroid[w[n, k]] (+ bias[n]),
// with FP32 accumulation of the signed INT8 activation codes.
${layout_declare_tensor(B, "w", "t_output", DTYPE, "buffer")}
${layout_declare_tensor(B, "r", "t_activations", "int", "buffer")}
${layout_declare_tensor(B, "r", "t_weight", "uint", "buffer")}
${layout_declare_tensor(B, "r", "t_scales", "float", "buffer")}
${layout_declare_tensor(B, "r", "t_row_norms", "float", "buffer")}
${layout_declare_tensor(B, "r", "t_centroids", "float", "buffer")}
${layout_declare_tensor(B, "r", "t_bias", "float", "buffer")}

layout(push_constant) uniform restrict Block {
  ivec4 problem_sizes;
  int apply_bias;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const uint n = gl_GlobalInvocationID.x;
  const uint m = gl_GlobalInvocationID.y;
  const uint M = uint(problem_sizes.x);
  const uint N = uint(problem_sizes.y);
  const uint K = uint(problem_sizes.z);
  if (m >= M || n >= N) {
    return;
  }

  float centroids[16];
  for (uint index = 0u; index < 16u; ++index) {
    centroids[index] = t_centroids[index];
  }

  float accumulator = 0.0;
  const uint weight_words = K >> 3u;
  for (uint word = 0u; word < weight_words; ++word) {
    const uint weight_codes = t_weight[word * N + n];
    const int low = t_activations[(word * 2u) * M + m];
    const int high = t_activations[(word * 2u + 1u) * M + m];
    for (uint nibble = 0u; nibble < 8u; ++nibble) {
      const int activations = nibble < 4u ? low : high;
      const int code = bitfieldExtract(activations, int((nibble & 3u) * 8u), 8);
      accumulator +=
          float(code) * centroids[(weight_codes >> (nibble * 4u)) & 15u];
    }
  }

  float value = accumulator * t_scales[m] * t_row_norms[n];
  if (apply_bias != 0) {
    value += t_bias[n];
  }
  t_output[m * N + n] = T(value);
}
