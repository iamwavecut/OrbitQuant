# OrbitQuant 0.1.1 Release Notes

OrbitQuant 0.1.1 is a patch release for the optimized packed-matmul runtime
defaults.

## Changes

- CUDA/Triton packed low-bit matmul now defaults to tile
  `block_m=32`, `block_n=128`, `block_k=64`, `num_warps=8`.
- The same default is exposed through `OrbitQuantConfig`, the CLI
  `kernel-bench` command, and the native/Triton packed matmul wrappers.

## Verification

- CPU/unit CI passed on GitHub for commit
  `5f49867d39d8998b1cff7c981c313507cb07b4c5`.
- A CUDA smoke on RTX 4090 with Torch 2.9.1+cu128 verified that fresh `main`
  uses `runtime_mode="auto_fused"`, selects `triton_cuda`, keeps packed
  tensors on CUDA, and reports `packed_matmul_tile.block_n == 128`.

## Claim Boundary

This release does not claim full-model speedup. Current CUDA/Triton packed
matmul remains memory-path evidence until broader model-level benchmarks show a
throughput win. Use `runtime_mode="dequant_bf16"` for reference/debug behavior.
