# OrbitQuant 0.1.6 Release Notes

OrbitQuant 0.1.6 aligns generated codebooks and activation normalization with
the OrbitQuant paper and improves compact Hugging Face artifact publication.

## Install

```bash
pip install "orbitquant[hf]==0.1.6"
```

For optimized packed-weight inference dependencies:

```bash
pip install "orbitquant[kernels]==0.1.6"
```

## Changes

- New artifacts use converged Lloyd-Max codebook version 2 derived from the
  exact beta marginal of unit-sphere coordinates.
- Activation normalization follows the paper's `x / (norm(x) + 1e-10)` rule
  in reference, Triton CUDA, and Metal paths.
- Existing version 1 artifacts remain loadable with their original codebooks.
- Compact uploads can consume a validated `compare-native` bundle, publish one
  canonical comparison matrix, and retain paired native-smoke evidence without
  uploading raw generated images or videos.

## Runtime

`runtime_mode="auto_fused"` remains the default. CUDA prefers the importable
native packed-matmul package and otherwise uses Triton; MPS uses the native
Metal package. `runtime_mode="dequant_bf16"` remains an explicit reference and
debug mode.

The package does not claim a universal throughput gain. Backend and model
performance depend on shape, device, framework version, and offload policy.
