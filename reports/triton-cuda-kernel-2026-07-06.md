# Triton CUDA Kernel Checkpoint - 2026-07-06

## Environment

- RunPod pod: `5yhnllitz76nht`
- GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition
- VRAM: 97887 MiB
- Driver: 580.126.16
- PyTorch in venv: 2.12.1+cu130
- CUDA runtime reported by PyTorch: 13.0
- Triton in venv: 3.7.1

## Scope

This checkpoint adds the first real `triton_cuda` activation backend increment:

- Norm computation remains PyTorch.
- RPBH rotation remains PyTorch.
- Codebook bucket lookup and norm rescale now run through a Triton kernel.
- Full fused norm+RPBH+lookup remains future work.

## Verification

Remote pod:

```bash
uv run pytest
uv run ruff check .
```

Result:

```text
70 passed
All checks passed!
```

Local Mac:

```bash
uv run pytest
uv run ruff check .
uv run git diff --check
```

Result:

```text
69 passed, 1 skipped
All checks passed!
```

The skipped local test is the GPU-only Triton backend test.

## Microbenchmark

Quick CUDA event timing on the RunPod RTX PRO 6000 pod, comparing the PyTorch
reference activation quantization path with `backend="triton_cuda"`.

| tokens | dim | block | reference ms | triton ms | speedup | max diff |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 3072 | 1024 | 0.3592 | 0.3279 | 1.0953x | 0.0 |
| 2048 | 3072 | 1024 | 0.5263 | 0.4707 | 1.1181x | 0.0 |
| 512 | 7680 | 512 | 0.3859 | 0.3511 | 1.0991x | 0.0 |

These numbers only measure the current partial Triton path. They should not be
reported as final optimized OrbitQuant latency.
