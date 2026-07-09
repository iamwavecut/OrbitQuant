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
  gate for GPU hosts. It first tries to load a prebuilt
  `native_packed_matmul` package through Hugging Face `kernels`,
  `LOCAL_KERNELS`, or an importable package. If no compatible prebuilt package
  is available, it fails before starting a source build unless
  `ORBITQUANT_ALLOW_NATIVE_KERNEL_BUILD=1` is set. With that explicit opt-in,
  it builds the exact
  `native-kernels/orbitquant-packed-matmul` kernel-builder redistributable
  variant matching the runtime Torch/CUDA/platform tuple, runs the native
  package tests, and benchmarks `native_packed_matmul` explicitly, matching the
  `auto_fused` runtime priority. This keeps paid GPU hosts from silently
  entering uncached CUDA/NCCL source builds. If neither a compatible prebuilt
  package nor a matching kernel-builder variant is available, the gate fails
  explicitly instead of loading an incompatible build.
- `scripts/runpod_ssh_health.sh` is the preflight for RunPod basic SSH hosts.
  It checks actual SSH authentication and remote shell execution with
  `ssh -F /dev/null -tt`, ignoring local SSH config and ControlMaster state.
  The probe feeds commands through stdin because RunPod basic SSH proxies can
  require a PTY while ignoring remote command arguments.
  Use it before starting the CUDA gate when the host comes from a RunPod
  Connect-tab SSH command.
- `scripts/run_mps_kernel_checks.sh` is the MPS/Metal correctness and smoke
  benchmark gate for Apple Silicon hosts. By default it requires the native
  packed matmul package. Set `ORBITQUANT_RUN_NATIVE_KERNEL_PACKAGE_CI=0` only
  to verify the inline Metal shader stages without closing the native packed
  matmul package gate.
- `scripts/verify_hf_kernel_model_artifact.py` is a model-artifact verification
  script for reviewers. It downloads one published OrbitQuant artifact by
  default, restores one real packed `OrbitQuantLinear` transformer projection,
  runs it through `runtime_mode="native_packed_matmul"`, compares against
  `dequant_bf16`, and prints finite-output, allclose, error, and weight-side
  storage accounting JSON. It intentionally avoids full Diffusers pipeline
  loading and image/video generation.
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
- A public Hugging Face source snapshot exists at
  `WaveCut/orbitquant-packed-matmul` commit
  `cb0ceb1a4d070556c52cfba691aba3f6647c246b`. It contains the tracked
  `native-kernels/orbitquant-packed-matmul` source package with no generated
  `build/`, local `.venv/`, `__pycache__/`, binary extension, or benchmark
  output files. The PyPI `orbitquant-0.1.0.tar.gz` source distribution also
  contains this kernel source under
  `orbitquant-0.1.0/native-kernels/orbitquant-packed-matmul/`, with SHA256
  `6abedb769b32c8d70f2763278e106346319d628d85ed7469549faa5020ab1a89`.
  Kernel Hub publication is not yet approved for the account. On
  2026-07-08T17:02Z,
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
  A second follow-up comment on 2026-07-09T11:56Z points reviewers to the
  public source snapshot, checked commit
  `b050a89d6e6f52098c73d904a85011231f77485c`, public PyPI source
  distribution URL, and SHA256.
  A third follow-up comment on 2026-07-09T12:22Z points reviewers to source
  snapshot `c34d9851cde2cf098589927a7b0bed85d65426af`, whose benchmark reports
  both `predequantized_f_linear_seconds_per_iter` and
  `dequantize_then_f_linear_seconds_per_iter`. That comment explicitly
  clarifies that the current MPS native packed matmul path is not throughput
  proof for large matrices: local W4 512x1024x1024 fp16 measured about
  `0.045x` versus dequantize-then-F.linear, and W4 512x3072x3072 fp16 measured
  about `0.044x` versus dequantize-then-F.linear. Treat those MPS numbers as
  correctness and memory-path evidence only, not performance evidence.
  A fourth follow-up comment on 2026-07-09T12:27Z answered the model-scope
  question: the kernel is intended for OrbitQuant-converted diffusion
  transformer backbones with packed `OrbitQuantLinear` layers, currently FLUX.1
  Schnell, FLUX.2 Klein, Z-Image-Turbo, and Wan2.1-T2V-1.3B-Diffusers; it is
  not a drop-in kernel for arbitrary unquantized models or skipped components
  such as text encoders, VAEs, embeddings, timestep MLPs, or final projection
  heads.
  The public source snapshot was updated again on 2026-07-09T12:39Z to commit
  `cb0ceb1a4d070556c52cfba691aba3f6647c246b`; the benchmark now also reports
  `packed_weight_path_bytes`, `materialized_weight_bytes`, and
  `packed_weight_path_vs_materialized_weight_ratio` so kernel review can
  distinguish weight-side storage savings from throughput claims.
  A fifth follow-up comment on 2026-07-09T12:41Z points reviewers to this
  snapshot and repeats that the storage fields are memory-path accounting, not
  large-matrix throughput proof.
  After reviewer asked for model-level verification scripts,
  `scripts/verify_hf_kernel_model_artifact.py` was added to verify the default
  `WaveCut/FLUX.2-klein-4B-OrbitQuant-W4A4` artifact at one restored packed
  transformer projection without running full generation.
  A sixth follow-up comment on 2026-07-09T12:50Z points reviewers to commit
  `f42d2dc19897adde62ec3ebb33e4ce748255dd54` and includes MPS/CUDA
  `LOCAL_KERNELS` commands for running that verifier.
  On 2026-07-09T12:57Z, that verifier passed locally on Apple Silicon MPS using
  the `torch212-metal-aarch64-darwin` local kernel variant and the published
  `WaveCut/FLUX.2-klein-4B-OrbitQuant-W4A4` artifact. It restored
  `transformer_blocks.0.attn.to_q` with shape 3072x3072, ran
  `runtime_mode="native_packed_matmul"` for one token, compared against
  `dequant_bf16`, and reported `finite=true`, `allclose_to_dequant_bf16=true`,
  `max_abs_error_vs_dequant_bf16=0.001953125`, and
  `packed_weight_path_vs_materialized_weight_ratio=0.2503289116753472`.
  A seventh follow-up comment on 2026-07-09T12:58Z posted the verifier command
  and JSON result summary to discussion 15.
  On 2026-07-09T13:05Z, the same published artifact layer was re-verified with
  `runtime_mode="auto_fused"` and the same local
  `torch212-metal-aarch64-darwin` package. It again reported `finite=true`,
  `allclose_to_dequant_bf16=true`,
  `max_abs_error_vs_dequant_bf16=0.001953125`, and
  `packed_weight_path_vs_materialized_weight_ratio=0.2503289116753472`.
  This proves that default optimized dispatch reaches the native packed matmul
  path for a real published artifact when the local Metal package is available.
  An eighth follow-up comment on 2026-07-09T13:05Z posted the `auto_fused`
  command and JSON result summary to discussion 15.
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
- A public-package CUDA smoke passed on 2026-07-09T13:15Z on the active RunPod
  `orbitquant-cuda-gate-4090` pod (`ofz7pyxcw6vlzm`) with Torch 2.9.1+cu128,
  CUDA 12.8, driver 580.159.04, and an NVIDIA GeForce RTX 4090. It installed
  `orbitquant[kernels]==0.1.0` from PyPI into a temporary `/tmp` venv using
  `--system-site-packages`, ran `orbitquant kernel-info`, and ran
  `orbitquant kernel-bench --device cuda --dtype float16 --runtime-mode
  auto_fused --tokens 16 --in-features 128 --out-features 128 --warmup 1
  --iterations 2`. The benchmark selected `triton_cuda` for both activation
  kernels and weight quantization, kept packed weight indices and row norms on
  `cuda:0`, reported `forward_prewarmed_ms=0.14182400703430176`, and removed
  the temporary venv after completion. This verifies the published PyPI package
  CUDA/Triton `auto_fused` path, not the separate native CUDA Kernel Hub
  package.
- After a reviewer asked for actual numbers, a model-like CUDA microbenchmark
  was run on the same pod on 2026-07-09T13:21Z with
  `orbitquant[kernels]==0.1.0` from PyPI, `tokens=512`, `in_features=3072`,
  `out_features=3072`, W4A4, float16, warmup 2, and 5 iterations. The
  `auto_fused` Triton path reported `forward_prewarmed_ms=0.6518784046173096`,
  `forward_cold_ms=0.6581952095031738`, and `peak_memory_bytes=69293568`.
  The explicit `dequant_bf16` reference reported
  `forward_prewarmed_ms=0.13742079734802246`,
  `forward_cold_ms=0.20090880393981933`, and
  `peak_memory_bytes=115025408`. These numbers were posted to discussion 15 on
  2026-07-09T13:22Z with the explicit caveat that current CUDA/Triton
  `auto_fused` is memory-path evidence, not a throughput win on this RTX 4090
  microbenchmark.
- A follow-up tile sweep on the same RTX 4090, PyPI package, and
  512x3072x3072 W4A4 float16 benchmark found the best tested Triton packed
  matmul tile at `block_m=32`, `block_n=128`, `block_k=64`, `num_warps=8`.
  A focused 20-iteration confirmation measured the previous default
  `block_n=64` at `forward_prewarmed_ms=0.6374400138854981` and the selected
  `block_n=128` default at `forward_prewarmed_ms=0.596992015838623`, both with
  `peak_memory_bytes=69293568`. This updates the package default tile for the
  packed matmul path, but it remains a local CUDA/Triton microbenchmark result
  and does not change the no-throughput-win claim boundary above.
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
- On 2026-07-09, `scripts/runpod_ssh_health.sh ssh
  ofz7pyxcw6vlzm-6441163d@ssh.runpod.io -i ~/.ssh/id_ed25519` passed against
  the active RTX 4090 pod after switching the probe to stdin-fed PTY execution.
  The same session confirmed that direct Kernel Hub publication is still
  blocked: `HfApi.create_repo(..., repo_type="kernel")` returned `403
  Forbidden: Kernel repository creation is restricted`. An uncached
  kernel-builder attempt for `torch212-cxx11-cu130-x86_64-linux` was stopped
  after it began compiling the CUDA/NCCL stack from source; this is not the
  release path for paid evaluation pods. Use an approved Kernel Hub upload or a
  pre-cached builder environment for native CUDA package closure.
- On 2026-07-09, the MPS shader-only gate passed locally with
  `ORBITQUANT_RUN_NATIVE_KERNEL_PACKAGE_CI=0` and tiny benchmark dimensions.
  The run verified Torch 2.12.1 MPS availability, `torch.mps.compile_shader`,
  MPS/backend capability tests, `orbitquant kernel-info`, and an MPS
  `runtime_mode="dequant_bf16"` benchmark. `optimized_stage` was
  `codebook_lookup_rescale,packed_weight_dequant`; native packed matmul load
  and benchmark stages were explicitly skipped.
- On 2026-07-09, a prebuilt-only native loader check still found no loadable
  CUDA/Metal Kernel Hub artifact: `repo_info(..., repo_type="kernel")`
  returned 404. After the storage-footprint benchmark update, the public source
  snapshot model repo resolved to commit
  `cb0ceb1a4d070556c52cfba691aba3f6647c246b`. The native loader therefore
  still requires `LOCAL_KERNELS`, an importable package, or Kernel Hub approval.

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
