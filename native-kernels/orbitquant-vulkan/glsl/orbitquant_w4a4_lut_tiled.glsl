#version 450 core

${define_required_extensions("buffer", DTYPE)}
#extension GL_EXT_control_flow_attributes : require

#define PRECISION ${PRECISION}
#define T ${buffer_scalar_type(DTYPE)}
#define K_TILE_WORDS ${K_TILE_WORDS}
#define USE_SHARED_LUT ${USE_SHARED_LUT}
#define TILE_M 8u
#define TILE_N 2u
#define WORKGROUP_M 32u
#define WORKGROUP_N 32u
#define WORKGROUP_SIZE 64u

${define_active_storage_type("buffer")}

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

shared uint activation_tile[K_TILE_WORDS][WORKGROUP_M];
shared uint weight_tile[K_TILE_WORDS][WORKGROUP_N];
#if USE_SHARED_LUT
shared float pair_lut_tile[256];
#endif

void main() {
  const uint M = uint(problem_sizes.x);
  const uint N = uint(problem_sizes.y);
  const uint K = uint(problem_sizes.z);
  const uint words_per_row = K >> 3u;
  const uint lane = gl_LocalInvocationID.y * 16u + gl_LocalInvocationID.x;
  const uint group_m_base = gl_WorkGroupID.y * WORKGROUP_M;
  const uint group_n_base = gl_WorkGroupID.x * WORKGROUP_N;
  const uint m_base = group_m_base + gl_LocalInvocationID.y * TILE_M;
  const uint n_base = group_n_base + gl_LocalInvocationID.x * TILE_N;

#if USE_SHARED_LUT
  for (uint index = lane; index < 256u; index += WORKGROUP_SIZE) {
    pair_lut_tile[index] = t_pair_lut[index];
  }
  memoryBarrierShared();
  barrier();
#endif

  float accumulators[TILE_M][TILE_N];
  [[unroll]] for (uint mi = 0u; mi < TILE_M; ++mi) {
    [[unroll]] for (uint ni = 0u; ni < TILE_N; ++ni) {
      accumulators[mi][ni] = 0.0;
    }
  }

  for (uint word_base = 0u; word_base < words_per_row;
       word_base += K_TILE_WORDS) {
    for (uint index = lane; index < K_TILE_WORDS * WORKGROUP_M;
         index += WORKGROUP_SIZE) {
      const uint tile_word = index / WORKGROUP_M;
      const uint row = index - tile_word * WORKGROUP_M;
      const uint word = word_base + tile_word;
      const uint m = group_m_base + row;
      activation_tile[tile_word][row] =
          word < words_per_row && m < M
          ? uint(t_activations[word * M + m])
          : 0u;
      const uint n = group_n_base + row;
      weight_tile[tile_word][row] =
          word < words_per_row && n < N ? t_weight[word * N + n] : 0u;
    }
    memoryBarrierShared();
    barrier();

    [[unroll]] for (uint tile_word = 0u; tile_word < K_TILE_WORDS;
                    ++tile_word) {
      if (word_base + tile_word >= words_per_row) {
        continue;
      }
      uint activation_codes[TILE_M];
      uint weight_codes[TILE_N];
      [[unroll]] for (uint mi = 0u; mi < TILE_M; ++mi) {
        activation_codes[mi] = activation_tile[tile_word]
            [gl_LocalInvocationID.y * TILE_M + mi];
      }
      [[unroll]] for (uint ni = 0u; ni < TILE_N; ++ni) {
        weight_codes[ni] =
            weight_tile[tile_word][gl_LocalInvocationID.x * TILE_N + ni];
      }
      [[unroll]] for (uint nibble = 0u; nibble < 8u; ++nibble) {
        const uint shift = nibble * 4u;
        [[unroll]] for (uint mi = 0u; mi < TILE_M; ++mi) {
          const uint activation_code =
              (activation_codes[mi] >> shift) & 15u;
          [[unroll]] for (uint ni = 0u; ni < TILE_N; ++ni) {
            const uint weight_code = (weight_codes[ni] >> shift) & 15u;
            const uint lut_index = activation_code * 16u + weight_code;
#if USE_SHARED_LUT
            accumulators[mi][ni] += pair_lut_tile[lut_index];
#else
            accumulators[mi][ni] += t_pair_lut[lut_index];
#endif
          }
        }
      }
    }
    memoryBarrierShared();
    barrier();
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
