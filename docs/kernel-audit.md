# OrbitQuant Kernel Claim Boundary

This document defines what this repository can claim about kernel acceleration
for the current artifact format and runtime modes.

## Current Backends

| Backend | Status | Implemented path | Release claim boundary |
| --- | --- | --- | --- |
| CPU | Reference-only | PyTorch reference activation quantization, weight dequantization, and linear matmul. | Correctness baseline only. Do not claim optimized CPU kernels or CPU speedup. |
| MPS/Metal | Partial optimized | Native packed matmul package when importable, plus inline Metal shader coverage for codebook lookup/rescale and packed weight dequantization. | `auto_fused` requires native packed matmul on MPS; do not claim full activation-plus-matmul fusion without model benchmark artifacts. |
| CUDA/Triton | Partial optimized | Native packed matmul package when importable, Triton activation norm/RPBH/FWHT/codebook/rescale, packed weight dequant, low-bit pack/unpack, offline weight quantization, AdaLN RTN quant/dequant, and packed matmul. | `auto_fused` prefers native packed matmul then Triton packed matmul on CUDA; do not claim full-model speedup or full activation-plus-matmul fusion without benchmark artifacts. |
| ROCm | Unsupported | No implementation in this tree. | Exclude from release acceleration claims unless implemented and verified. |
| XPU | Unsupported | No implementation in this tree. | Exclude from release acceleration claims unless implemented and verified. |

## Verification Gates

- `orbitquant kernel-info` prints machine-readable backend capabilities and
  `claim_status` values. `implemented_stage` describes code present in the
  package; `optimized_stage` is populated only when that backend is active in
  the current environment.
- `scripts/run_cuda_kernel_checks.sh` is the CUDA correctness and benchmark
  gate for GPU hosts. By default it also runs
  `native-kernels/orbitquant-packed-matmul` kernel-builder CI so the native
  packed matmul package is validated together with the Python/Triton CUDA path.
  The gate loads that native package through Hugging Face `kernels` and
  benchmarks `native_packed_matmul` explicitly, matching the `auto_fused`
  runtime priority.
- `scripts/run_mps_kernel_checks.sh` is the MPS/Metal correctness and smoke
  benchmark gate for Apple Silicon hosts.
- Full-model speedup claims require backend-specific benchmark artifacts from
  the target model class and native settings. Synthetic kernel benchmarks are
  useful diagnostics, not release evidence for FLUX, Z-Image, or Wan throughput.

## Current Verification Evidence

- MPS/Metal partial gate passed locally on 2026-07-08T15:58Z with
  `PYTHON_BIN="$(uv python find)" scripts/run_mps_kernel_checks.sh`. The run
  verified Torch 2.12.1 MPS availability, `torch.mps.compile_shader`, MPS
  kernel tests, `orbitquant kernel-info`, native `WaveCut/orbitquant-packed-matmul`
  loading through Hugging Face `kernels` via `LOCAL_KERNELS`, `auto_fused`
  benchmark execution, and explicit `runtime_mode="native_packed_matmul"`
  benchmark execution.
- The native packed matmul package passed local kernel-builder CI on
  2026-07-08T16:31Z with
  `nix --option sandbox relaxed run .#ci-test -L`. The run verified
  kernel-builder layout hooks, macOS 15/Python ABI 3.9 compatibility,
  get-kernel loading, and 17 package tests for the Metal build.
- A private Hugging Face repo exists at
  `WaveCut/orbitquant-packed-matmul`, but Kernel Hub publication is not yet
  approved for the account. `nix --option sandbox relaxed --option max-jobs 1
  --option cores 1 run .#build-and-upload -L` built the three Metal variants
  for commit `a4d927c` and then failed at upload with the Kernel Hub approval
  error. The approval request draft is
  [kernel-hub-approval-request.md](kernel-hub-approval-request.md). Do not treat
  the native package as remotely loadable through `get_kernel` until that
  approval is granted and upload verification passes.
- CUDA/Triton must still be verified on a CUDA host with
  `scripts/run_cuda_kernel_checks.sh` before the overall kernel audit release
  gate can be closed.

## Packaging Boundary

The current CUDA OrbitQuant pipeline path is implemented with Python Triton
kernels. It is not itself a Hugging Face Kernels Hub `kernel-builder` package
and must not be described as ABI3 kernel-builder compliant.
`orbitquant kernel-info` therefore reports `hf_kernel_builder_compliant=false`
for the `triton_cuda` backend.

The `native_packed_matmul` runtime uses the separate
`native-kernels/orbitquant-packed-matmul` package. That package is configured
for `kernel-builder`, targets CUDA and Metal, uses ABI3-safe
`TORCH_LIBRARY_EXPAND`/`REGISTER_EXTENSION` bindings, and has its own package
tests. Remote loading through Hugging Face `kernels.get_kernel` requires
Kernel Hub publish approval for `WaveCut/orbitquant-packed-matmul`; until then,
release tests must use `LOCAL_KERNELS` or an importable local package. It should
not be used as evidence that the Python Triton backend is fully fused or
kernel-builder compliant.

The current MPS path uses `torch.mps.compile_shader` for local Metal shaders.
It is not an upstream PyTorch native MPS operator implementation, so
`orbitquant kernel-info` reports `upstream_native_mps_op=false`.
