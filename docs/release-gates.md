# OrbitQuant Release Gates

This page describes the current release state and the evidence required for its
public claims.

## Release State

| Surface | Status |
| --- | --- |
| GitHub repository | Public, `main` |
| Python package | `orbitquant==0.1.6` on PyPI |
| GitHub release | `v0.1.6` |
| Default runtime | `auto_fused` |
| Hugging Face artifacts | 14 public compact model repositories |
| ComfyUI integration | Public `ComfyUI-OrbitQuant`; PyPI `comfyui-orbitquant==0.1.3` |

## Methodology

- The implementation is audited against arXiv 2607.02461v1.
- New artifacts use converged Lloyd-Max codebook version 2 from the exact
  unit-sphere coordinate marginal.
- Activation normalization is `x / (norm(x) + 1e-10)` in reference, Triton,
  and Metal paths.
- RPBH weight folding and online activation rotation use the same deterministic
  rotation state.
- AdaLN modulation projections use group-64 INT4 RTN with floating-point
  activations where required by model policy.
- Text encoders, VAEs, embeddings, timestep MLPs, and final projection heads
  remain outside the OrbitQuant transformer-linear path.

The detailed conformance matrix is in
[paper-methodology-audit.md](paper-methodology-audit.md).

## Kernel Gate

`runtime_mode="auto_fused"` is the default for optimized inference.

- CUDA optimized mode uses native packed matmul when available and Triton
  packed matmul otherwise.
- MPS optimized mode requires the local native Metal package.
- CUDA and MPS do not silently materialize a full dequantized weight matrix in
  `auto_fused` mode.
- CPU remains a reference backend.
- ROCm and XPU are not release backends.
- Native packages can be built and loaded locally without Kernel Hub.

See [kernel-audit.md](kernel-audit.md) for build commands, verified hardware,
model coverage, and performance claim boundaries.

## Artifact Gate

The canonical inventory contains:

| Family | Configurations |
| --- | --- |
| FLUX.2 Klein | W4A4, W3A3, W2A4, W2A3 |
| FLUX.1-schnell | W4A4, W3A3, W2A4, W2A3 |
| Z-Image-Turbo | W4A4, W3A3, W2A4, W2A3 |
| Wan2.1-T2V-1.3B-Diffusers | W4A6, W4A4 |

All 14 repositories must pass:

- compact artifact and checksum validation;
- source revision and license metadata validation;
- quantized, AdaLN, and skipped-module policy inventory validation;
- native-settings finite-output smoke validation;
- one canonical BF16-versus-OrbitQuant comparison matrix;
- model-card usage instructions;
- rejection of raw generated images, videos, temporary files, and local logs.

The machine-readable audit is:

```bash
uv run orbitquant audit-hf-artifacts \
  --namespace WaveCut \
  --policy-inventory-root reports/native/module-inventories \
  --summary-only \
  --fail-on-artifact-regression
```

FLUX.1-schnell, Z-Image-Turbo, and Wan2.1 are paper-aligned targets. FLUX.2
Klein is an additional target using a paper-style native protocol.

## Framework Compatibility

The release gate covers both published and development framework lines:

```bash
scripts/run_hf_compat_checks.sh --mode all
```

The current matrix passes Diffusers 0.39.0 with Transformers 5.13.0 and the
corresponding development checkouts used by CI. The adapter retains explicit
fallback helpers for framework versions whose quantizer registration surfaces
differ.

## Quality Claim Boundary

Native BF16-versus-quantized generations exist for every configuration and are
shown in model cards. The release does not publish GenEval or VBench scores and
does not claim reproduction of the paper's metric tables. Missing metric cells
therefore block only metric and ranking claims, not artifact availability or
native smoke status.

## Required Checks

```bash
uv run pytest -q
uv run ruff check .
scripts/run_paper_methodology_checks.sh
scripts/run_hf_compat_checks.sh --mode all
uv build
uvx twine check dist/*
```

A release is complete only when the public PyPI package installs in a clean
environment, the Git tag resolves to the intended `main` commit, all public
artifact checks pass, and the repository worktree is clean.
