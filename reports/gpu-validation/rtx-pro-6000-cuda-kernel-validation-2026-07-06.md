# RTX PRO 6000 CUDA Kernel Validation - 2026-07-06

This is a kernel-validation checkpoint, not a release-grade native image/video
quality report.

## RunPod Search

- Target GPU: `NVIDIA RTX PRO 6000`.
- User constraint: do not wait more than 3 minutes for a single pod to
  initialize.
- Image used for the successful pod:
  `runpod/pytorch:1.0.3-cu1300-torch291-ubuntu2404`.
- Network volume: none.
- Container disk: 80 GB local disk.

Stuck-init attempts:

- `twp41vshqmm0cf`: RTX PRO 6000 Blackwell Server Edition, official image,
  deleted after readiness did not complete within the 3-minute limit.

Successful pod:

- Pod: `6q11sjcpfjryc9`.
- GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition.
- Driver: `570.211.01`.
- VRAM: `97887 MiB`.
- Pod was deleted after evidence was copied back.

## Runtime

```text
torch 2.9.1+cu130
cuda 13.0
triton 3.5.1
numpy 2.5.1
cuda_available True
device NVIDIA RTX PRO 6000 Blackwell Workstation Edition
```

## Kernel Capability

`orbitquant kernel-info` reported:

- `triton_cuda.available`: `true`.
- `triton_cuda.optimized`: `true`.
- `triton_cuda.optimized_stage`: `codebook_lookup_rescale`.
- `triton_cuda.weight_dequant_optimized`: `true`.
- `triton_cuda.full_fusion`: `false`.

This validates the current partial CUDA path:

- activation norm remains PyTorch,
- RPBH rotation remains PyTorch,
- activation codebook lookup/rescale uses Triton,
- packed weight dequantization uses Triton,
- matmul is still the PyTorch BF16/FP32 path.

## Verification

Remote targeted kernel tests:

```bash
pytest tests/test_kernels.py -q -rs
```

Result:

```text
.......ssssss..... [100%]
```

Only MPS-specific tests were skipped.

Remote full test suite:

```bash
pytest -q -rs
```

Result:

```text
139 passed, 10 skipped
```

Skips were expected:

- MPS-only tests on a CUDA host,
- optional Diffusers/Transformers integration tests because the GPU validation
  venv installed `.[dev,eval]`, not `.[hf]`.

Remote lint:

```bash
ruff check .
```

Result:

```text
All checks passed!
```

## Raw Evidence

Raw command output is stored next to this file:

- `reports/gpu-validation/rtx-pro-6000-cuda-kernel-validation-2026-07-06.txt`
