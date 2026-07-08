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
  gate for GPU hosts. By default it builds the exact
  `native-kernels/orbitquant-packed-matmul` kernel-builder redistributable
  variant matching the runtime Torch/CUDA/platform tuple, loads that native
  package through Hugging Face `kernels` via `LOCAL_KERNELS`, runs the native
  package tests, and benchmarks `native_packed_matmul` explicitly, matching the
  `auto_fused` runtime priority. If the current kernel-builder matrix has no
  matching variant, the gate fails explicitly instead of loading an
  incompatible local build.
- `scripts/runpod_ssh_health.sh` is the preflight for RunPod basic SSH hosts.
  It checks actual SSH authentication and remote command execution with
  `ssh -F /dev/null -tt`, ignoring local SSH config and ControlMaster state.
  Use it before starting the CUDA gate when the host comes from a RunPod
  Connect-tab SSH command.
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
- The native packed matmul package passed local kernel-builder CI again on
  2026-07-08T16:59Z with
  `nix --option sandbox relaxed run .#ci-test -L` after adding kernel
  `upstream`/`source` metadata. The run verified kernel-builder layout hooks,
  macOS 15/Python ABI 3.9 compatibility, get-kernel loading, and 17 package
  tests for the Metal build.
- A private Hugging Face source snapshot exists at
  `WaveCut/orbitquant-packed-matmul` commit
  `6821e4cd5ff1894994d7137c1d861660cfeed1c8`, refreshed on
  2026-07-08T18:00Z after adding CUDA launch-error checks, but Kernel Hub
  publication is not yet approved for the account. On 2026-07-08T17:02Z,
  `nix --option sandbox relaxed run .#build-and-copy -L` built and copied the
  three Metal variants, and
  `nix --option sandbox relaxed run .#build-and-upload -L` found those variants
  before failing only at the Hugging Face permission check. The approval
  request is open as
  `https://huggingface.co/spaces/kernels-community/README/discussions/15`.
  A follow-up comment on 2026-07-08T18:03Z points reviewers to refreshed
  source snapshot `6821e4cd5ff1894994d7137c1d861660cfeed1c8` and source
  archive SHA256
  `77aef6caa1bbdbbd77e2cbf5003423073e001191d008473c957795d7bed03651`.
  Re-running `nix --option sandbox relaxed run .#build-and-upload -L` on
  2026-07-08T18:12Z at OrbitQuant commit `956842a` rebuilt the three Metal
  variants, passed ABI/get-kernel build checks, and still stopped at the same
  Kernel Hub publish permission error.
  The submitted request text is
  [kernel-hub-approval-request.md](kernel-hub-approval-request.md). Do not treat
  the native package as remotely loadable through `get_kernel` until that
  approval is granted and upload verification passes.
- The MPS native package path has smoke benchmark evidence from the matching
  `torch212-metal-aarch64-darwin` variant: W4 512x1024x1024 float16 at
  `0.00764581459807232` seconds/iteration over 20 iterations, and W4
  512x3072x3072 float16 at `0.10189520000712946` seconds/iteration over
  10 iterations.
- The OrbitQuant native loader was smoke-tested through `LOCAL_KERNELS` on
  2026-07-08T17:10Z. With Torch 2.12.1 it selected
  `build/torch212-metal-aarch64-darwin`, ran `matmul_packed_weight` on MPS, and
  produced a finite float16 output tensor.
- CUDA/Triton partial gate passed on 2026-07-08T19:31Z at OrbitQuant commit
  `301d836` on a RunPod secure-cloud RTX 4090 host with Torch 2.9.1+cu128,
  CUDA 12.8, Triton 3.5.1, and driver 570.211.01. The run used
  `ORBITQUANT_RUN_NATIVE_KERNEL_PACKAGE_CI=0` and completed CUDA kernel tests,
  `orbitquant kernel-info`, `auto_fused` CUDA `kernel-bench`, and CUDA
  `quantize-bench` with exit 0. This verifies the Python/Triton CUDA path and
  packed-weight CUDA runtime fallback behavior, not the separate native CUDA
  kernel-builder package.
- Native CUDA `native_packed_matmul` still needs a compatible loadable variant.
  A locally built `build/torch29-cxx11-cu130-x86_64-linux` variant was copied
  to the same CUDA 12.8 host and failed before execution with
  `ImportError: libcudart.so.13`, proving that artifact is a CUDA 13 build and
  cannot close the CUDA 12.8 native-package gate. A CUDA 12.8-compatible
  kernel-builder variant or approved Hugging Face Kernel Hub upload is still
  required before claiming native CUDA package coverage. Current local checks
  build exact `redistributable.<runtime-variant>` outputs instead of selecting
  ignored `build/` artifacts. The current HF `kernel-builder` matrix exports
  `torch211-cxx11-cu128-x86_64-linux`, but not `torch29-cxx11-cu128-x86_64-linux`;
  `kernels` rejects CUDA variants newer than the runtime CUDA minor version.
  Therefore the existing RunPod image with Torch 2.9.1+cu128 can keep serving
  Triton/eval work, but it cannot close the native CUDA package gate. Closing
  that gate requires a runtime with an exported compatible variant, such as
  Torch 2.11+cu128, or an approved Kernel Hub upload with a compatible build.

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
