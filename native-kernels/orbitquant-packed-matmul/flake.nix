{
  description = "Flake for OrbitQuant packed matmul kernel";

  inputs = {
    kernel-builder.url = "github:huggingface/kernels";
  };

  outputs = { self, kernel-builder }:
    kernel-builder.lib.genKernelFlakeOutputs {
      inherit self;
      path = ./.;
    };
}
