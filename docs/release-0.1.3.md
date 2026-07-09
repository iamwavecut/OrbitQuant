# OrbitQuant 0.1.3 Release Notes

OrbitQuant 0.1.3 is a packaging patch for cross-platform kernel extras.

## Changes

- The `kernels` extra remains the default install target for optimized
  packed-weight inference support:

  ```bash
  pip install "orbitquant[kernels]==0.1.3"
  ```

- `triton>=3.5` is now constrained to Linux installs of the `kernels` extra.
  This keeps CUDA/Triton support available on Linux while allowing macOS/MPS
  users to install `orbitquant[kernels]` for Hugging Face `kernels` package
  support without trying to resolve an unavailable Triton wheel.
- The package version, CLI version, and `orbitquant.__version__` are aligned at
  `0.1.3`.

## Claim Boundary

This release does not change quantization math, artifact format, model cards,
runtime dispatch policy, or kernel implementations. It only fixes optional
dependency resolution for platforms where Triton is not published.
