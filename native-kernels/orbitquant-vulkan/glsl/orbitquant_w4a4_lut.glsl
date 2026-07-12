#version 450 core

${define_required_extensions("buffer", DTYPE)}
#extension GL_EXT_control_flow_attributes : require

#define PRECISION ${PRECISION}
#define T ${buffer_scalar_type(DTYPE)}

${define_active_storage_type("buffer")}

#define TILE_M 1
#define TILE_N 1

layout(std430) buffer;

${layout_declare_tensor(B, "w", "t_output", DTYPE, "buffer")}
${layout_declare_tensor(B, "r", "t_activations", "int", "buffer")}
${layout_declare_tensor(B, "r", "t_weight", "uint", "buffer")}
${layout_declare_tensor(B, "r", "t_token_norms", "float", "buffer")}
${layout_declare_tensor(B, "r", "t_row_norms", "float", "buffer")}
${layout_declare_tensor(B, "r", "t_pair_lut", "float", "buffer")}
${layout_declare_tensor(B, "r", "t_bias", "float", "buffer")}

layout(push_constant) uniform restrict Block {
  ivec4 problem_sizes;
  int apply_bias;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const uint n_base = gl_GlobalInvocationID.x * TILE_N;
  const uint m_base = gl_GlobalInvocationID.y * TILE_M;
  const uint M = uint(problem_sizes.x);
  const uint N = uint(problem_sizes.y);
  const uint K = uint(problem_sizes.z);
  if (m_base >= M || n_base >= N) {
    return;
  }

  const uint words_per_row = K >> 3u;
  float accumulators[TILE_M][TILE_N];
  [[unroll]] for (uint mi = 0u; mi < TILE_M; ++mi) {
    [[unroll]] for (uint ni = 0u; ni < TILE_N; ++ni) {
      accumulators[mi][ni] = 0.0;
    }
  }

  for (uint word = 0u; word < words_per_row; ++word) {
    uint activation_codes[TILE_M];
    uint weight_codes[TILE_N];
    [[unroll]] for (uint mi = 0u; mi < TILE_M; ++mi) {
      const uint m = min(m_base + mi, M - 1u);
      activation_codes[mi] =
          uint(t_activations[word * M + m]);
    }
    [[unroll]] for (uint ni = 0u; ni < TILE_N; ++ni) {
      const uint n = min(n_base + ni, N - 1u);
      weight_codes[ni] = t_weight[word * N + n];
    }

    for (uint nibble = 0u; nibble < 8u; ++nibble) {
      const uint shift = nibble * 4u;
      [[unroll]] for (uint mi = 0u; mi < TILE_M; ++mi) {
        const uint activation_code =
            (activation_codes[mi] >> shift) & 15u;
        [[unroll]] for (uint ni = 0u; ni < TILE_N; ++ni) {
          const uint weight_code = (weight_codes[ni] >> shift) & 15u;
          accumulators[mi][ni] +=
              t_pair_lut[activation_code * 16u + weight_code];
        }
      }
    }
  }

  [[unroll]] for (uint mi = 0u; mi < TILE_M; ++mi) {
    const uint m = m_base + mi;
    if (m >= M) {
      continue;
    }
    [[unroll]] for (uint ni = 0u; ni < TILE_N; ++ni) {
      const uint n = n_base + ni;
      if (n >= N) {
        continue;
      }
      float value =
          accumulators[mi][ni] * t_token_norms[m] * t_row_norms[n];
      if (apply_bias != 0) {
        value += t_bias[n];
      }
      t_output[m * N + n] = T(value);
    }
  }
}
