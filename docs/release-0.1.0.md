# OrbitQuant 0.1.0 Release Notes

OrbitQuant 0.1.0 is the first public package release candidate for
calibration-free OrbitQuant post-training quantization of image and video
diffusion transformers.

## Package Scope

- Python package: `orbitquant`.
- License: Apache-2.0.
- Python support: 3.11 and newer.
- Primary integrations: Hugging Face Diffusers, Hugging Face Transformers, and
  ComfyUI-OrbitQuant.
- Artifact format: compact transformer-component artifacts with
  `model.safetensors`, `quantization_config.json`, `orbitquant_manifest.json`,
  shared codebook/rotation sidecars, checksums, and model-card comparison
  assets.

The package does not include source model weights or generated validation media.
Published model repositories carry compact OrbitQuant artifacts separately from
the Python source distribution.

## Implemented Quantization

- Data-agnostic RPBH rotations.
- Lloyd-Max scalar codebooks keyed by input dimension and bit width.
- Weight and activation quantization for transformer-block linear projections.
- INT4 RTN weight-only handling for AdaLN modulation projections.
- Model policies for:
  - `black-forest-labs/FLUX.2-klein-4B`
  - `black-forest-labs/FLUX.1-schnell`
  - `Tongyi-MAI/Z-Image-Turbo`
  - `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`

Text encoders, VAE modules, embeddings, timestep MLPs, and final projection
heads remain in source precision by default.

## Runtime Modes

`OrbitQuantConfig` defaults to `runtime_mode="auto_fused"`.

- CUDA: `auto_fused` prefers the native packed low-bit matmul package, then the
  Triton packed matmul path.
- MPS: `auto_fused` requires the native Metal packed matmul package.
- CPU: reference path only.
- `runtime_mode="dequant_bf16"` is the explicit compatibility/debug reference
  mode and materializes dequantized weights before PyTorch matmul.

`auto_fused` does not silently fall back to full BF16/FP16 weight
materialization on CUDA or MPS when packed kernels are unavailable.

Install the optional kernel runtime dependencies on hosts that should use
optimized CUDA or Hub-published native packed matmul kernels:

```bash
pip install "orbitquant[kernels]"
```

The `kernels` extra installs the Hugging Face `kernels` loader and Triton
dependency used by the CUDA runtime path. CPU reference runs do not require
this extra.

## Artifact Targets

Paper-aligned targets:

- FLUX.1-schnell: 1024x1024, 4 steps, guidance 0.0, W4A4/W3A3/W2A4/W2A3.
- Z-Image-Turbo: 1024x1024, 10 steps, guidance 0.0, W4A4/W3A3/W2A4/W2A3.
- Wan2.1 T2V 1.3B: 832x480, 81 frames, 50 steps, guidance 5.0, W4A6/W4A4.

Additional target:

- FLUX.2 Klein: 1024x1024, 4 steps, guidance 1.0, W4A4/W3A3/W2A4/W2A3.

FLUX.2 Klein artifacts are not claimed as OrbitQuant paper reproduction
artifacts.

## Verification Commands

Core package:

```bash
uv run pytest
uv run ruff check .
uv run --with build python -m build
uv run --with twine python -m twine check dist/*
```

Paper methodology and policy invariants:

```bash
scripts/run_paper_methodology_checks.sh
```

Hugging Face compatibility:

```bash
scripts/run_hf_compat_checks.sh --mode all
```

Published artifact hygiene:

```bash
orbitquant audit-hf-artifacts \
  --namespace WaveCut \
  --policy-inventory-root reports/native/module-inventories \
  --fail-on-artifact-regression
```

Kernel gates:

```bash
scripts/run_mps_kernel_checks.sh
scripts/run_cuda_kernel_checks.sh
```

The CUDA kernel gate requires a CUDA host. The MPS gate requires Apple Silicon
with the PyTorch MPS backend.

## Claim Boundaries

- No calibration dataset, prompt statistics, timestep ranges, or generated image
  statistics are used to construct the quantizer.
- Compact artifacts can be native-smoke validated without full release metrics.
- GenEval and VBench numbers are release evidence for paper metric claims and
  are not implied by compact artifact readiness.
- CPU is a correctness reference path only.
- ROCm and XPU are not implemented backends in this release.
- Full-model speedup claims require backend-specific benchmark artifacts for the
  target model and native settings.

## Source Distribution

Expected PyPI artifacts:

- `orbitquant-0.1.0.tar.gz`
- `orbitquant-0.1.0-py3-none-any.whl`

The source distribution contains package code, tests, documentation, and native
kernel source. It must not contain generated `build/`, `.venv/`, `__pycache__/`,
raw benchmark output, local model weights, or generated media.

## Native Kernel Package

The native packed matmul package lives under
`native-kernels/orbitquant-packed-matmul`. It is tracked separately from the
Python wheel and is intended for Hugging Face Kernel Hub publication as
`WaveCut/orbitquant-packed-matmul` version 1.

Until Kernel Hub publication is available, local validation can use
`LOCAL_KERNELS` or an importable local package build.
