# OrbitQuant Kernel Support

OrbitQuant uses packed low-bit weights directly in optimized runtime modes. The
reference path materializes a floating-point weight matrix and is available only
when explicitly selected.

The runtime contract below reflects the current source tree. Benchmark tables
retain the exact software and hardware of each recorded run rather than
implying that those versions are current installation requirements. Binary
platform support is only marked verified where build, install, and forward
evidence exists.

## Runtime Contract

`runtime_mode="auto_fused"` is the default.

| Device | Default dispatch | Required support |
| --- | --- | --- |
| CUDA | Native activation kernel plus packed W4A4 tensor-core path; native or Triton packed fallback | A matching native package for the fastest path; Triton for the CUTLASS epilogue and generic packed fallback |
| MPS | Native packed matmul | An importable local Metal package |
| CPU | Native exact activation, packed low-bit matmul, and packed INT4 AdaLN when the CPU variant is importable; reference fallback otherwise | A matching native CPU package for packed execution |

CUDA and MPS raise an actionable error when no packed backend is available.
They do not silently fall back to full weight dequantization. CPU keeps a
compatibility fallback when the optional native package is absent. Use
`runtime_mode="dequant_bf16"` explicitly for compatibility, debugging, or
numerical comparison.

Other explicit modes are `native_packed_matmul`, `triton_packed_matmul`,
`debug_no_quant`, and `debug_no_activation_quant`.

## Backend Status

| Backend | Status | Implemented path |
| --- | --- | --- |
| CUDA | Optimized packed inference | Native RPBH/quantization, chunked packed-weight decode plus CUTLASS INT8 matmul, direct packed CUDA MMA fallback, and generic Triton packed fallback |
| MPS/Metal | Optimized packed inference | Native Metal packed matmul and Metal activation quantization stages |
| CPU | Hardware-verified source builds; Windows x86_64 ABI3 wheel verified in CI but not yet published | Runtime ISA dispatch across scalar, AVX2/FMA, AVX-512/BF16, and ARM64 NEON; exact packed activation, packed low-bit matmul, and packed INT4 group-64 AdaLN |
| Vulkan | Unverified | No runtime backend |
| ROCm | Unsupported | No release backend |
| XPU | Unsupported | No release backend |

## Local Native Package

Kernel Hub publication is not required. Build the native package locally from
`native-kernels/orbitquant-packed-matmul`:

```bash
cd native-kernels/orbitquant-packed-matmul
nix --option sandbox relaxed run .#build-and-copy -L
```

Expose the matching generated variant directly:

```bash
export PYTHONPATH="$PWD/build/<matching-variant>:$PYTHONPATH"
```

For a fast machine-local CUDA build without Nix:

```bash
cargo install --git https://github.com/huggingface/kernels hf-kernel-builder
cd native-kernels/orbitquant-packed-matmul
kernel-builder check-config .
kernel-builder create-pyproject -f .
TORCH_CUDA_ARCH_LIST="8.9" CUDACXX=/usr/local/cuda/bin/nvcc \
  python setup.py build_kernel
export PYTHONPATH="$PWD/build/<matching-cuda-variant>:$PYTHONPATH"
```

For a locally built Metal variant that remains loadable on macOS 15 and newer:

```bash
cargo install --git https://github.com/huggingface/kernels hf-kernel-builder
cd native-kernels/orbitquant-packed-matmul
kernel-builder check-config .
kernel-builder create-pyproject -f .
MACOSX_DEPLOYMENT_TARGET=15.0 \
  CMAKE_ARGS="-DCMAKE_OSX_DEPLOYMENT_TARGET=15.0" \
  python setup.py build_kernel
kernel-builder check-abi --macos 15.0 --python-abi 3.9 .
export PYTHONPATH="$PWD/build/<matching-metal-variant>:$PYTHONPATH"
```

For PyTorch 2.9 CUDA workloads, enable expandable allocator segments before the
Python process starts to minimize reserved/NVML memory:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python generate.py
```

The generated `setup.py` and CMake files come from `kernel-builder` and must not
be committed. This development build targets the current host toolchain. Use the
Nix build for a redistributable variant and run `kernel-builder check-abi` before
distributing it. A local Ubuntu 24.04 build can be ABI3 at the Python boundary
while still depending on a GLIBC version newer than `manylinux_2_28`.

To load the local build through Hugging Face `kernels` instead of importing it
through `PYTHONPATH`, map the kernel repository to the same generated variant
directory containing `metadata.json`:

```bash
export LOCAL_KERNELS="WaveCut/orbitquant-packed-matmul=$PWD/build/<matching-variant>"
```

The variant must match the active Torch, CUDA or Metal, platform, and C++ ABI
tuple. OrbitQuant rejects incompatible packages instead of loading them.

## Verified Devices And Shapes

The native CPU package was built and executed on an AMD EPYC 4564P (Zen 4)
with GCC 13.3 and Torch 2.13.0+cpu. The hosted cpuset exposed logical CPUs
`13,29`, which are the two SMT threads of one physical core. These results are
single-physical-core ISA and latency evidence, not a multi-core scaling claim.
Runtime dispatch exercised scalar, AVX2/FMA, and AVX-512 paths against the same
independent BF16 reference. The AVX-512 group-64 path uses `vpermw` for INT4
lookup and `vdpbf16ps` accumulation; annotated disassembly confirmed that the
hot rows=4 and rows=8 loops retain accumulators in ZMM registers without a full
weight decode buffer or stack-spilled accumulator tile.

For an exact W4A4 projection with 32 activation rows and dimension 1536, the
native activation stage measured 0.1484 ms median, packed matmul measured
1.0528 ms, and the full native layer measured 1.2081 ms. Explicit
`dequant_bf16` measured 1.5836 ms and source BF16 `F.linear` measured 0.5479 ms.
The packed and explicit-reference outputs had relative RMSE `1.203e-7` and
maximum error `1.526e-5`. Packed weights and row norms occupied 1,179,648 bytes
versus 4,718,592 bytes for BF16 weights, and the native layer retained no
dequantized-weight cache.

A clean ABI3 wheel rebuild was also profiled on an AMD EPYC 9654 (Zen 4) with
GCC 13.3 and Torch 2.12.1+cpu. Its hosted cpuset exposed logical CPUs `36,132`,
the two SMT threads of one physical core. At 32 rows and dimension 1536, exact
W4A4 packed matmul measured 1.4928 ms hot median and 1.5151 ms p95. Explicit
dequantization plus `F.linear` measured 6.6373 ms median, so the packed path was
4.51x faster while using 1,182,784 bytes instead of 4,718,592 bytes for the
weight-side representation. A resident BF16 `F.linear` was faster at 0.7823 ms
but retained the complete dense weight. Relative RMSE was `6.546e-6` and maximum
error was `0.03125`. Disassembly contained the selected AVX-512/BF16 `vpermw`
and `vdpbf16ps` instructions. The Ubuntu 24.04 development wheel required
`manylinux_2_38`; it is local evidence, not a distributable Linux release
artifact.

AdaLN uses an exact signed INT4 group-64 lookup with BF16 scales and
activations. The realistic modulation shape below is 3072 input channels and
18432 output channels. Timings are 21 post-warmup calls with
`ORBITQUANT_CPU_THREADS=1`; `auto_fused` selected the AVX-512/BF16 path.

| Rows | Packed median | Packed p95 | Resident BF16 median | Packed vs resident BF16 | Relative RMSE | Max error |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.4828 ms | 2.7575 ms | 6.9140 ms | 2.78x | 1.01e-6 | 0.001953 |
| 4 | 3.0471 ms | 3.0726 ms | 6.3742 ms | 2.09x | 2.50e-5 | 0.0625 |
| 32 | 24.4159 ms | 24.4693 ms | 15.9194 ms | 0.65x | 2.32e-5 | 0.125 |

The packed AdaLN weight is 28,311,552 bytes and its BF16 scales are 1,769,472
bytes, compared with 113,246,208 bytes for the BF16 weight. A native-only
process increased resident memory by 1.4 MB on its first call and by another
4.3 MB across 500 calls; it did not allocate the 113 MB dense weight. The
rows=32 result documents the crossover where oneDNN's resident BF16 GEMM is
faster; AdaLN conditioning normally uses batch-sized row counts rather than
spatial-token row counts.

The same x86_64 stable-ABI 2.11 binary built against Torch 2.13 loaded and ran
under Torch 2.12.1. This verifies the current LibTorch Stable ABI boundary for
those two runtimes; it does not make the wheel independent of the platform
GLIBC, C++ runtime, or CPU ISA. A separate Windows Server 2022 build on an AMD
EPYC 7763 used Python 3.12.10, Torch 2.12.1, CMake 3.31.6, Ninja Multi-Config,
and MSVC 19.44. It produced a 147,326-byte
`orbitquant_packed_matmul-0.1.0-cp39-abi3-win_amd64.whl` with
`Requires-Dist: torch>=2.11`. A clean environment passed 73 native-package
tests with 33 accelerator-only skips; OrbitQuant integration and the independent
paper oracle passed another 33 tests with two CUDA/Triton skips. The suite
forced both scalar and AVX2 dispatch on the x86_64 runner. This verifies the
Windows x86_64 build, installation, packed forward, and artifact contract; the
wheel is not published yet, and Windows Server 2025/Visual Studio 2026 remains
unverified. A release-compatible Linux wheel also remains pending.

The ARM64 native CPU path was separately built on an Apple M2 Max. At 32 rows
and dimension 1536, the full native W4A4 layer measured about 1.20 ms versus
1.84 ms for explicit BF16 dequantization; the packed matmul itself was about
0.81 ms versus 0.79 ms for a resident dense matmul. Scalar and NEON paths were
both exercised against the independent reference. These CPU measurements are
separate from the preferred Metal GPU path on Apple Silicon.

CUDA native-package verification passed on an NVIDIA RTX PRO 4500 Blackwell
with Torch 2.13.0+cu130 using the
`torch213-cxx11-cu130-x86_64-linux` ABI3 variant. Native-resolution generation
used `native_packed_matmul` for every OrbitQuant linear in all release model
families:

| Model | Packed OrbitQuant linears |
| --- | ---: |
| FLUX.2 Klein | 100/100 |
| FLUX.1-schnell | 418/418 |
| Z-Image-Turbo | 238/238 |
| Wan2.1-T2V-1.3B | 300/300 |

The activation path used Triton CUDA. No tested layer reached full-weight
dequantization or `F.linear` fallback in optimized mode.

OrbitQuant 0.3.1 MPS verification passed on an Apple M2 Max with Torch 2.12.1.
It covered the native Metal package, inline shader stages, `auto_fused`
dispatch, and a real 3072x3072 projection restored from the published FLUX.2
W4A4 artifact. The current native package suite passed 74 tests on the MPS
host; 23 CUDA-only cases were skipped. The macOS 15 deployment-target build
passed `kernel-builder check-abi` for the Python 3.9 stable ABI, and the packed
and reference outputs were finite and numerically close.

The Metal package also passed an ABI3 build matrix for Torch 2.11, 2.12, and
2.13. A quantized tiny GPT-2 run exercised all eight wrapped projections during
prefill and cached decode through `native_packed_matmul`, with finite outputs.

The CUDA W4A4 stack released in OrbitQuant 0.3.0 was built and tested on an
NVIDIA L40S (`sm_89`) with Torch 2.9.1+cu128 and CUDA 12.8. That recorded native
package suite passed 49 CUDA tests; the ten skipped cases were Metal-only.
Coverage includes W2/W3/W4/W6 generic packed matmul, FP16/BF16, bias and
no-bias paths, partial output tiles, direct packed W4A4 MMA, native packed-A4
activation quantization, and native INT8 activation quantization for full-block
dimensions and the 12288/4096 blocked RPBH case. OrbitQuant 0.3.1 changes the
Metal path; the CUDA implementation measured in this section is unchanged.

For W4A4 on compute capability 8.0 or newer, the selected path is:

1. A native CUDA launch computes token norms, applies RPBH/FWHT, selects the
   fixed Lloyd-Max bins, and emits INT8 surrogate codes.
2. Packed row-major W4 indices are decoded one bounded output-channel chunk at
   a time; the complete floating-point weight matrix is never materialized.
3. `torch._int_mm` dispatches the INT8 matrix product to CUTLASS tensor cores.
4. A Triton epilogue applies token norms, BF16 row norms, both surrogate scales,
   and bias.

The direct packed CUDA MMA implementation remains available for unsupported
CUTLASS shapes. It includes asynchronous packed loads and SM89-specific tile
selection. The checkpoint keeps the original row-major four-bit payload; no
repacked duplicate weights are stored.

The selected production dispatch was also profiled on an NVIDIA GeForce RTX
4090 (`sm_89`) with Torch 2.9.1+cu128 and CUDA 12.8. For a representative
FLUX.2 fused-input projection with 4608 activation rows, 4096 input channels,
and 36864 output channels, ten post-warmup calls measured 4.317 ms median,
4.352 ms mean, and 0.867 GB peak allocated memory. The output was finite and
the dispatch reported `native_packed_matmul` with
`native_cuda_int8_surrogate` activation quantization.

Nsight Systems attributed 77.8% of GPU kernel time to the CUTLASS INT8 GEMM,
10.0% to the fused scale/norm/bias epilogue, 8.2% to bounded packed-W4 decode,
and 4.0% to native token norm, RPBH/FWHT, and codebook assignment. The same
shape measured 4.809 ms median on the L40S. Nsight Compute performance counters
were unavailable on the hosted 4090 because the provider disabled GPU counter
access (`ERR_NVGPUCTRPERM`); the Systems trace and CUDA event timings do not
depend on those counters.

A full FLUX.2 Klein 9B W4A4 pipeline exercised all 396 packed projections
across the transformer and Qwen3 text encoder. The controlled native run used
1024x1024 output, four steps, guidance 1.0, seed 0, and ten identical prompts:

| Runtime | Load | Hot mean | Hot median | CUDA allocated peak | CUDA reserved peak | NVML peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SDNQ UINT4 | 5.918 s | 2.0885 s | 2.0875 s | 14.844 GB | 16.377 GB | 17.564 GB |
| OrbitQuant W4A4 | 2.543 s | 2.0907 s | 2.0920 s | 13.942 GB | 14.544 GB | 15.731 GB |

OrbitQuant was within 0.11% of SDNQ hot mean while using 0.902 GB less CUDA
allocated memory and 1.833 GB less reserved/NVML memory. Every projection
reported `native_cuda_int8_surrogate`; no full-weight dequantization path was
entered. The ten deterministic outputs were finite and matched the separately
validated cumulative W4A4 run byte for byte.

Measured W4 BF16 operator latency for `in_features=768` and
`out_features=2304`:

| Rows | Packed CUDA | Resident BF16 `F.linear` | Materialize + `F.linear` | Packed vs materialize |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0300 ms | 0.0176 ms | 0.0863 ms | 2.88x |
| 3 | 0.0311 ms | 0.0193 ms | 0.0883 ms | 2.84x |
| 8 | 0.0339 ms | 0.0211 ms | 0.0929 ms | 2.74x |
| 9 | 0.0330 ms | 0.0208 ms | 0.0929 ms | 2.82x |
| 15 | 0.0328 ms | 0.0206 ms | 0.0936 ms | 2.85x |
| 16 | 0.0336 ms | 0.0210 ms | 0.0935 ms | 2.78x |
| 31 | 0.0336 ms | 0.0235 ms | 0.0915 ms | 2.73x |
| 64 | 0.0310 ms | 0.0233 ms | 0.0936 ms | 3.02x |
| 512 | 0.0499 ms | 0.0211 ms | 0.0889 ms | 1.78x |

Rows 1-8 use a warp packed-matvec. Rows 9 and above use zero-padded
WMMA/MMA tiles when the dtype and input dimension permit them. CUDA reads BF16
row norms and FP16/BF16 bias directly, so optimized OrbitQuant inference does
not create per-forward FP32 copies of those tensors.

The full W4A4 `OrbitQuantLinear` path, including Triton activation norm, RPBH,
codebook quantization, and native packed matmul, measured 0.1652 ms at one token,
0.1630 ms at 16 tokens, and 0.1644 ms at 512 tokens. The corresponding
prewarmed `dequant_bf16` path measured 0.1519 ms, 0.1574 ms, and 0.1576 ms while
retaining a full BF16 weight. Peak allocated memory was 14.0/14.1/18.0 MB for
packed execution versus 28.2/28.2/29.8 MB for the reference path.

Measured operator latency on an Apple M2 Max with Torch 2.12.1, FP16
activations, W4 packed weights, `in_features=768`, and
`out_features=2304`:

| Rows | Packed Metal | Resident FP16 `F.linear` | Materialize + `F.linear` | Packed vs materialize |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0470 ms | 0.0411 ms | 0.2310 ms | 4.92x |
| 2 | 0.0439 ms | 0.0422 ms | 0.2371 ms | 5.40x |
| 3 | 0.0451 ms | 0.0404 ms | 0.2348 ms | 5.20x |
| 8 | 0.0420 ms | 0.0503 ms | 0.2512 ms | 5.97x |
| 9 | 0.0446 ms | 0.0501 ms | 0.2422 ms | 5.43x |
| 16 | 0.0459 ms | 0.0581 ms | 0.2512 ms | 5.47x |
| 31 | 0.0428 ms | 0.0659 ms | 0.2623 ms | 6.12x |

One-row projections use the SIMD-group packed matvec. Aligned FP16/BF16
projections with two or more rows use the padded matrix path; unsupported or
unaligned shapes retain the generic packed path. All paths consume packed
indices directly and do not allocate a full floating-point weight matrix.

For this shape, packed indices, row norms, and centroids occupy 25.26% of the
FP16 materialized weight size. A permanently resident pre-dequantized
`F.linear` can remain faster at one to three rows, at the cost of retaining the
full FP16 weight; packed execution is faster in the measured 8-31 row cases.

For a full 4096-coordinate RPBH block with constants resident on MPS, the
fused activation stage uses a 512-thread group:

| Rows | 256 threads | 512 threads | Speedup |
| ---: | ---: | ---: | ---: |
| 1 | 0.0752 ms | 0.0562 ms | 1.34x |
| 8 | 0.0797 ms | 0.0551 ms | 1.45x |
| 64 | 0.0839 ms | 0.0693 ms | 1.21x |
| 512 | 0.3229 ms | 0.2548 ms | 1.27x |
| 4096 | 2.0754 ms | 1.7717 ms | 1.17x |

Smaller or multi-block RPBH dimensions retain the 256-thread path.

The Triton CUDA fallback passed on an NVIDIA B200 with Torch 2.8.0+cu128 and
Triton 3.7.1. For the same 1x768 by 2304x768 W4 shape, activation quantization
took 0.0581 ms and the prewarmed packed matmul forward took 0.1234 ms. A
pre-dequantized `F.linear` took 0.0134 ms but requires the full floating-point
weight to remain resident; this comparison is a memory/latency trade-off, not
a packed-kernel speedup claim.

Run the backend gates with:

```bash
scripts/run_cuda_kernel_checks.sh
PYTHON_BIN="$(uv python find)" scripts/run_mps_kernel_checks.sh
```

Verify a published artifact projection with:

```bash
python scripts/verify_hf_kernel_model_artifact.py \
  --repo-id WaveCut/FLUX.2-klein-4B-OrbitQuant-W4A4 \
  --runtime-mode native_packed_matmul
```

## Performance Claims

Packed execution reduces weight-side materialization and runtime memory for the
validated image pipelines. On the controlled L40S FLUX.2 Klein 9B comparison,
the optimized W4A4 path reached practical SDNQ hot-generation parity with lower
allocated, reserved, and NVML memory. Throughput still depends on model shapes,
device, Torch, offload policy, and backend. Wan with CPU offload did not show a
throughput or peak-memory improvement in the recorded native run. OrbitQuant
does not claim a universal speedup.

Synthetic operator benchmarks are diagnostics. Results above compare packed
execution with both weight materialization plus `F.linear` and, where stated,
a permanently resident pre-dequantized weight. Model-level performance claims
must use native model settings and report the reference configuration beside
the packed configuration.
