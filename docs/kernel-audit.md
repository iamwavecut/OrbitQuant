# OrbitQuant Kernel Support

OrbitQuant uses packed low-bit weights directly in optimized runtime modes. The
reference path materializes a floating-point weight matrix and is available only
when explicitly selected.

## Runtime Contract

`runtime_mode="auto_fused"` is the default.

| Device | Default dispatch | Required support |
| --- | --- | --- |
| CUDA | Native packed matmul, then Triton packed matmul | An importable local native package or Triton |
| MPS | Native packed matmul | An importable local Metal package |
| CPU | Reference matmul | PyTorch |

CUDA and MPS raise an actionable error when no packed backend is available.
They do not silently fall back to full weight dequantization. Use
`runtime_mode="dequant_bf16"` explicitly for compatibility, debugging, or
numerical comparison.

Other explicit modes are `native_packed_matmul`, `triton_packed_matmul`,
`debug_no_quant`, and `debug_no_activation_quant`.

## Backend Status

| Backend | Status | Implemented path |
| --- | --- | --- |
| CUDA | Optimized packed inference | Native CUDA packed matmul; Triton activation norm, RPBH/FWHT, codebook lookup, rescale, and packed matmul fallback |
| MPS/Metal | Optimized packed inference | Native Metal packed matmul and Metal activation quantization stages |
| CPU | Reference | PyTorch activation quantization, weight dequantization, and linear matmul |
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

The generated `setup.py` and CMake files come from `kernel-builder` and must not
be committed. This development build targets the current host toolchain. Use the
Nix build for a redistributable variant and run `kernel-builder check-abi` before
distributing it. A local Ubuntu 24.04 build can be ABI3 at the Python boundary
while still depending on a GLIBC version newer than `manylinux_2_28`.

Alternatively, point Hugging Face `kernels` at the local source tree:

```bash
export LOCAL_KERNELS="$PWD"
```

The variant must match the active Torch, CUDA or Metal, platform, and C++ ABI
tuple. OrbitQuant rejects incompatible packages instead of loading them.

## Verified Devices And Shapes

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

MPS verification passed on Apple Silicon with Torch 2.12.1. It covered the
native Metal package, inline shader stages, `auto_fused` dispatch, and a real
3072x3072 projection restored from the published FLUX.2 W4A4 artifact. The
packed and reference outputs were finite and numerically close.

The Metal package also passed an ABI3 build matrix for Torch 2.11, 2.12, and
2.13. A quantized tiny GPT-2 run exercised all eight wrapped projections during
prefill and cached decode through `native_packed_matmul`, with finite outputs.

The native CUDA package was also built and tested on an NVIDIA RTX 4090
(`sm_89`) with Torch 2.9.1+cu128, CUDA 12.8, and kernel-builder 0.17.0-dev0.
All 34 applicable package tests passed for W2/W3/W4/W6, FP16/BF16, bias and
no-bias paths, partial output tiles, short rows, and tensor-core rows.

The same local package build passed its 34 CUDA tests on an NVIDIA A40
(`sm_86`) with Torch 2.9.1+cu128 and CUDA 12.8. A full FLUX.2 Klein 9B W4A4
pipeline exercised 396 packed linears across the transformer and Qwen3 text
encoder. Native 1024x1024 generation at four steps used
`native_packed_matmul` for every packed linear and the Triton activation path;
no full weight matrix was materialized.

Representative A40 W4A4 layer timings from that pipeline are:

| Projection | Rows | Shape | RPBH + activation quantization | Native packed matmul | Full layer |
| --- | ---: | ---: | ---: | ---: | ---: |
| Double-stream Q | 4096 | 4096 -> 4096 | 0.698 ms | 2.607 ms | 3.279 ms |
| Single-stream fused input | 4608 | 4096 -> 36864 | 0.770 ms | 25.962 ms | 26.677 ms |
| Single-stream output | 4608 | 16384 -> 4096 | 4.257 ms | 11.244 ms | 15.514 ms |

The corresponding full-pipeline hot-generation mean was 7.418 seconds with a
17.66 GB NVML peak. On the same A40 and prompt pack, BF16 measured 3.911
seconds and 40.83 GB, while SDNQ UINT4 measured 3.546 seconds and 17.55 GB.
These results establish the memory reduction and the remaining throughput gap;
they do not support a model-level speedup claim for OrbitQuant on A40.

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

| Rows | Packed Metal | Materialize weight + `F.linear` | Packed speedup |
| ---: | ---: | ---: | ---: |
| 1 | 0.0508 ms | 0.2181 ms | 4.29x |
| 3 | 0.0879 ms | 0.2188 ms | 2.49x |
| 8 | 0.1698 ms | 0.2230 ms | 1.31x |
| 9 | 0.0368 ms | 0.2264 ms | 6.15x |
| 16 | 0.0485 ms | 0.2581 ms | 5.33x |
| 31 | 0.0568 ms | 0.2635 ms | 4.64x |

Rows 1-8 use a SIMD-group packed matvec. Larger and partial tiles use the
padded matrix path. Both paths consume packed indices directly and do not
allocate a full floating-point weight matrix.

For this shape, packed indices, row norms, and centroids occupy 25.26% of the
FP16 materialized weight size. A permanently resident pre-dequantized
`F.linear` remains faster on these short rows, at the cost of retaining the
full FP16 weight.

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
validated image pipelines. Throughput depends on model shapes, device, Torch,
offload policy, and backend. Wan with CPU offload did not show a throughput or
peak-memory improvement in the recorded native run. OrbitQuant does not claim a universal speedup.

Synthetic operator benchmarks are diagnostics. Results above compare packed
execution with both weight materialization plus `F.linear` and, where stated,
a permanently resident pre-dequantized weight. Model-level performance claims
must use native model settings and report the reference configuration beside
the packed configuration.
