#version 450 core

${define_required_extensions("buffer", DTYPE)}

#define PRECISION ${PRECISION}
#define T ${buffer_scalar_type(DTYPE)}

${define_active_storage_type("buffer")}

layout(std430) buffer;

${layout_declare_tensor(B, "w", "t_norms", "float", "buffer")}
${layout_declare_tensor(B, "r", "t_input", DTYPE, "buffer")}

layout(push_constant) uniform restrict Block {
  ivec4 problem_sizes;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

shared float sums[64];

void main() {
  const uint row = gl_GlobalInvocationID.y;
  const uint lane = gl_LocalInvocationID.x;
  const uint rows = uint(problem_sizes.x);
  const uint K = uint(problem_sizes.z);

  float sum = 0.0;
  if (row < rows) {
    const uint row_offset = row * K;
    for (uint k = lane; k < K; k += 64u) {
      const float value = float(t_input[row_offset + k]);
      sum += value * value;
    }
  }
  sums[lane] = sum;
  memoryBarrierShared();
  barrier();

  for (uint stride = 32u; stride > 0u; stride >>= 1u) {
    if (lane < stride) {
      sums[lane] += sums[lane + stride];
    }
    memoryBarrierShared();
    barrier();
  }

  if (lane == 0u && row < rows) {
    t_norms[row] = sqrt(sums[0]);
  }
}
