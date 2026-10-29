#version 450 core

${define_required_extensions("buffer", DTYPE)}

#define PRECISION ${PRECISION}
#define T ${buffer_scalar_type(DTYPE)}
#define BLOCK_SIZE ${BLOCK_SIZE}

${define_active_storage_type("buffer")}

layout(std430) buffer;

// One RPBH block of one token: the rotated FP32 values and the block's absmax.
// Absmax INT8 activations keep the token scale, so the input is not normalized.
${layout_declare_tensor(B, "w", "t_rotated", "float", "buffer")}
${layout_declare_tensor(B, "w", "t_block_absmax", "float", "buffer")}
${layout_declare_tensor(B, "r", "t_input", DTYPE, "buffer")}
${layout_declare_tensor(B, "r", "t_permutation", "int", "buffer")}
${layout_declare_tensor(B, "r", "t_signs", "int", "buffer")}

layout(push_constant) uniform restrict Block {
  ivec4 problem_sizes;
  float inv_sqrt_block;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

shared float values[BLOCK_SIZE];
shared float maxima[64];

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
  for (uint i = lane; i < block_size; i += 64u) {
    const uint rotated_index = block_offset + i;
    const uint source_index = uint(t_permutation[rotated_index]);
    const float sign = float(t_signs[rotated_index]);
    values[i] = float(t_input[row * K + source_index]) * sign;
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

  float largest = 0.0;
  for (uint i = lane; i < block_size; i += 64u) {
    const float value = values[i] * inv_sqrt_block;
    t_rotated[row * K + block_offset + i] = value;
    largest = max(largest, abs(value));
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

  if (lane == 0u) {
    t_block_absmax[row * (K / block_size) + block] = maxima[0];
  }
}
