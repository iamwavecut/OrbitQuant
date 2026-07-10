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

Alternatively, point Hugging Face `kernels` at the local source tree:

```bash
export LOCAL_KERNELS="$PWD"
```

The variant must match the active Torch, CUDA or Metal, platform, and C++ ABI
tuple. OrbitQuant rejects incompatible packages instead of loading them.

## Verification

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

Synthetic operator benchmarks are diagnostics. Model-level performance claims
must use native model settings and report the reference configuration beside
the packed configuration.
