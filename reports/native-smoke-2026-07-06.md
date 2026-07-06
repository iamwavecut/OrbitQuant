# OrbitQuant Native Smoke Checkpoint - 2026-07-06

This is a checkpoint report, not a release-grade GenEval/VBench report.

## Environment

- GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97,887 MiB VRAM.
- Runpod pod: `yu1wzrff35p95p`, deleted after artifacts were copied back.
- Torch: `2.8.0+cu128`.
- Diffusers: `0.39.0`.
- Transformers: `5.13.0`.
- OrbitQuant branch: `orbitquant-mvp`.
- Latest checkpoint commit at the end of the run: `2d0eadf`.
- Remote verification on pod: `47 passed`; `ruff check .` clean.
- Local verification: `47 passed`; `ruff check .` clean.

## Prompt

Image prompt:

```text
A small red cube on a white table, studio lighting, sharp focus
```

Video prompt:

```text
A small red cube rotating slowly on a white table, smooth camera, studio lighting
```

These are native smoke prompts only. They are not a substitute for GenEval or
VBench.

## Image Native Smoke

All image runs used native 1024x1024 output.

| Suite | Model | Steps | Guidance | Runs |
| --- | --- | ---: | ---: | --- |
| `flux2-native` | `black-forest-labs/FLUX.2-klein-4B` | 4 | 1.0 | BF16, W4A4, W3A3, W2A4, W2A3 |
| `flux1-schnell-native` | `black-forest-labs/FLUX.1-schnell` | 4 | 0.0 | BF16, W4A4, W3A3, W2A4, W2A3 |
| `z-image-native` | `Tongyi-MAI/Z-Image-Turbo` | 10 | 0.0 | BF16, W4A4, W3A3, W2A4, W2A3 |

Local artifacts:

- `artifacts/native-smoke/flux2/`
- `artifacts/native-smoke/flux1/`
- `artifacts/native-smoke/zimage/`

Observed outcome:

- FLUX.2 and FLUX.1 produced finite, nonblank images for all tested bit settings.
- Z-Image produced finite, nonblank images for all tested bit settings.
- Z-Image W2A3 visibly degraded on this smoke prompt, as expected from the risk notes.

## Video Native Smoke

All Wan runs used native 832x480 video, 81 frames, 50 steps, CFG 5.0.

| Suite | Model | Runs |
| --- | --- | --- |
| `wan-native` | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | BF16, W4A6, W4A4 |

Local artifacts:

- `artifacts/native-smoke/wan/wan-native_seed0.mp4`
- `artifacts/native-smoke/wan/wan-native_seed0_W4A6.mp4`
- `artifacts/native-smoke/wan/wan-native_seed0_W4A4.mp4`

Local `ffprobe` sanity:

| File | Width | Height | Frames | Duration |
| --- | ---: | ---: | ---: | ---: |
| `wan-native_seed0.mp4` | 832 | 480 | 81 | 8.1s |
| `wan-native_seed0_W4A6.mp4` | 832 | 480 | 81 | 8.1s |
| `wan-native_seed0_W4A4.mp4` | 832 | 480 | 81 | 8.1s |

## Bugs Found By Native Runs

- Low-bit packing was originally Python-loop based and stalled real FLUX
  quantization. Fixed by vectorizing pack/unpack in chunks.
- `OrbitQuantLinear` originally unpacked/dequantized weights on every forward.
  Fixed by adding runtime-only dequantized weight cache keyed by device and dtype.
- Generic policy incorrectly quantized Z-Image `t_embedder.mlp.0`, which broke
  code that reads `.weight.dtype`. Fixed by making timestep/embed modules hard
  skips before block/MLP matching.
- Video runner treated numpy frame batches as boolean. Fixed by using explicit
  frame extraction.
- Wan export required eval dependencies. Installed `imageio` and
  `imageio-ffmpeg` on the pod; the package already exposes them via `.[eval]`.

## Not Yet Done

- GenEval metrics for FLUX.1-schnell and Z-Image-Turbo.
- VBench metrics for Wan.
- Multi-prompt visual comparison pack and contact sheets.
- Compact pre-quantized full pipeline artifacts for every bit setting.
- Fused CUDA/Triton kernels; current optimized path is still PyTorch graph plus
  cached BF16 dequantized weights.
- MPS/Metal optimized path.
- ComfyUI repository and nodes.
