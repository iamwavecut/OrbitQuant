#version 450 core

${define_required_extensions("buffer", "int")}
${define_active_storage_type("buffer")}

#define PRECISION highp

layout(std430) buffer;

${layout_declare_tensor(B, "w", "t_output", "int", "buffer")}
${layout_declare_tensor(B, "r", "t_input", "uint", "buffer")}

layout(push_constant) uniform restrict Block {
  ivec4 problem_sizes;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const uint n = gl_GlobalInvocationID.x;
  const uint word = gl_GlobalInvocationID.y;
  const uint N = uint(problem_sizes.y);
  const uint words_per_row = uint(problem_sizes.z) >> 3u;
  if (n >= N || word >= words_per_row) {
    return;
  }
  t_output[word * N + n] = int(t_input[n * words_per_row + word]);
}
