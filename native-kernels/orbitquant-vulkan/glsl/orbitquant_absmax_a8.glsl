#version 450 core

${define_required_extensions("buffer", "int")}
${define_active_storage_type("buffer")}

#define PRECISION highp

layout(std430) buffer;

// Per-token absmax INT8 of the rotated values: scale = max(absmax, 1e-12) / 127 and
// code = clamp(roundEven(value / scale), -127, 127), four codes per word, word-major.
${layout_declare_tensor(B, "w", "t_packed", "int", "buffer")}
${layout_declare_tensor(B, "w", "t_scales", "float", "buffer")}
${layout_declare_tensor(B, "r", "t_rotated", "float", "buffer")}
${layout_declare_tensor(B, "r", "t_block_absmax", "float", "buffer")}

layout(push_constant) uniform restrict Block {
  ivec4 problem_sizes;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

shared float maxima[64];

void main() {
  const uint row = gl_GlobalInvocationID.y;
  const uint lane = gl_LocalInvocationID.x;
  const uint rows = uint(problem_sizes.x);
  const uint K = uint(problem_sizes.z);
  const uint blocks = K / uint(problem_sizes.w);

  float largest = 0.0;
  if (row < rows) {
    for (uint block = lane; block < blocks; block += 64u) {
      largest = max(largest, t_block_absmax[row * blocks + block]);
    }
  }
  maxima[lane] = largest;
  memoryBarrierShared();
  barrier();

  for (uint stride = 32u; stride > 0u; stride >>= 1u) {
    if (lane < stride) {
      maxima[lane] = max(maxima[lane], maxima[lane + stride]);
    }
    memoryBarrierShared();
    barrier();
  }
  if (row >= rows) {
    return;
  }

  const float scale = max(maxima[0], 1.0e-12) / 127.0;
  if (lane == 0u) {
    t_scales[row] = scale;
  }
  const uint words = K >> 2u;
  for (uint word = lane; word < words; word += 64u) {
    uint packed = 0u;
    for (uint byte = 0u; byte < 4u; ++byte) {
      const float value = t_rotated[row * K + word * 4u + byte];
      const int code = int(clamp(roundEven(value / scale), -127.0, 127.0));
      packed |= (uint(code) & 255u) << (byte * 8u);
    }
    t_packed[word * rows + row] = int(packed);
  }
}
