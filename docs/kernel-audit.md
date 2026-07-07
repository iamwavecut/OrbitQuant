# OrbitQuant Kernel Claim Boundary

This document defines what this repository can claim about kernel acceleration
for the current artifact format and runtime modes.

## Current Backends

| Backend | Status | Implemented path | Release claim boundary |
| --- | --- | --- | --- |
| CPU | Reference-only | PyTorch reference activation quantization, weight dequantization, and linear matmul. | Correctness baseline only. Do not claim optimized CPU kernels or CPU speedup. |
| MPS/Metal | Partial optimized | Inline Metal shader for codebook lookup/rescale and packed weight dequantization. | PyTorch still handles norm, RPBH rotation, and `F.linear`; do not claim full MPS fusion. |
| CUDA/Triton | Partial optimized | Triton activation norm/RPBH/FWHT/codebook/rescale, packed weight dequant, low-bit pack/unpack, offline weight quantization, AdaLN RTN quant/dequant, and opt-in packed matmul. | Default runtime remains `dequant_bf16`; do not claim default low-bit tensor-core speedup or full activation-plus-matmul fusion. |
| ROCm | Unsupported | No implementation in this tree. | Exclude from release acceleration claims unless implemented and verified. |
| XPU | Unsupported | No implementation in this tree. | Exclude from release acceleration claims unless implemented and verified. |

## Verification Gates

- `orbitquant kernel-info` prints machine-readable backend capabilities and
  `claim_status` values. `implemented_stage` describes code present in the
  package; `optimized_stage` is populated only when that backend is active in
  the current environment.
- `scripts/run_cuda_kernel_checks.sh` is the CUDA/Triton correctness and
  benchmark gate for GPU hosts.
- `scripts/run_mps_kernel_checks.sh` is the MPS/Metal correctness and smoke
  benchmark gate for Apple Silicon hosts.
- Full-model speedup claims require backend-specific benchmark artifacts from
  the target model class and native settings. Synthetic kernel benchmarks are
  useful diagnostics, not release evidence for FLUX, Z-Image, or Wan throughput.

## Packaging Boundary

The current CUDA path is implemented with Python Triton kernels. It is not a
Hugging Face Kernels Hub `kernel-builder` package and must not be described as
ABI3 kernel-builder compliant until that packaging path exists and passes the
kernel-builder checks. `orbitquant kernel-info` therefore reports
`hf_kernel_builder_compliant=false` for the current CUDA path.

The current MPS path uses `torch.mps.compile_shader` for local Metal shaders.
It is not an upstream PyTorch native MPS operator implementation, so
`orbitquant kernel-info` reports `upstream_native_mps_op=false`.
