#version 450 core

${define_required_extensions("buffer", DTYPE)}

#define PRECISION ${PRECISION}
#define T ${buffer_scalar_type(DTYPE)}
#define BLOCK_SIZE ${BLOCK_SIZE}

${define_active_storage_type("buffer")}

layout(std430) buffer;

${layout_declare_tensor(B, "w", "t_packed", "int", "buffer")}
${layout_declare_tensor(B, "r", "t_input", DTYPE, "buffer")}
${layout_declare_tensor(B, "r", "t_norms", "float", "buffer")}
${layout_declare_tensor(B, "r", "t_permutation", "int", "buffer")}
${layout_declare_tensor(B, "r", "t_signs", "int", "buffer")}
${layout_declare_tensor(B, "r", "t_boundaries", "float", "buffer")}

layout(push_constant) uniform restrict Block {
  ivec4 problem_sizes;
  float eps;
  float inv_sqrt_block;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

shared float values[BLOCK_SIZE];

void main() {
  const uint block = gl_WorkGroupID.x;
  const uint row = gl_WorkGroupID.y;
  const uint lane = gl_LocalInvocationID.x;
  const uint rows = uint(problem_sizes.x);
  const uint K = uint(problem_sizes.z);
  const uint block_size = uint(problem_sizes.w);
  if (row >= rows || block_size != uint(BLOCK_SIZE)) {
    return;
  }

  const uint block_offset = block * block_size;
  const float denominator = t_norms[row] + eps;
  for (uint i = lane; i < block_size; i += 64u) {
    const uint rotated_index = block_offset + i;
    const uint source_index = uint(t_permutation[rotated_index]);
    const float sign = float(t_signs[rotated_index]);
    values[i] = float(t_input[row * K + source_index]) * sign / denominator;
  }
  memoryBarrierShared();
  barrier();

  for (uint span = 1u; span < block_size; span <<= 1u) {
    const uint butterflies = block_size >> 1u;
    for (uint i = lane; i < butterflies; i += 64u) {
      const uint group = i / span;
      const uint offset = i - group * span;
      const uint left = group * (span << 1u) + offset;
      const uint right = left + span;
      const float a = values[left];
      const float b = values[right];
      values[left] = a + b;
      values[right] = a - b;
    }
    memoryBarrierShared();
    barrier();
  }

  const uint words_per_block = block_size >> 3u;
  const uint words_per_row = K >> 3u;
  for (uint word = lane; word < words_per_block; word += 64u) {
    uint packed = 0u;
    for (uint nibble = 0u; nibble < 8u; ++nibble) {
      const float value = values[word * 8u + nibble] * inv_sqrt_block;
      uint code = 0u;
      for (uint boundary = 0u; boundary < 15u; ++boundary) {
        code += value > t_boundaries[boundary] ? 1u : 0u;
      }
      packed |= code << (nibble * 4u);
    }
    const uint output_word = block * words_per_block + word;
    t_packed[output_word * rows + row] = int(packed);
  }
}
