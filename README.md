# OrbitQuant

Clean-room implementation of OrbitQuant for calibration-free post-training
quantization of image and video diffusion transformers.

OrbitQuant is based on
[OrbitQuant: Data-Agnostic Quantization for Image and Video Diffusion Transformers](https://arxiv.org/abs/2607.02461).
The library targets Hugging Face Diffusers pipelines and stores compact
transformer-component artifacts that can be patched back into the original
pipeline.

This repository is still pre-release. The artifact format, public API, and
kernel backends are being hardened against native FLUX, Z-Image, and Wan
generation workloads.

## What It Provides

- Data-agnostic RPBH rotation and Lloyd-Max codebook quantization.
- Weight and activation quantization for diffusion transformer linear modules.
- INT4 RTN weight-only handling for AdaLN modulation projections.
- Compact `safetensors` artifacts with manifest, codebooks, rotations, and
  checksums.
- Diffusers helper APIs for quantizing, saving, loading, and validating
  transformer-component artifacts.
- Native image/video generation helpers for producing BF16-vs-OrbitQuant
  comparison assets.

Text encoders, VAE, embeddings, timestep MLPs, and final projection heads are
left in source precision by default.

## Install

```bash
pip install -e ".[hf,eval,dev]"
```

For runtime-only use from a checked-out package:

```bash
pip install -e ".[hf]"
```

On managed CUDA images that already ship a vendor-matched PyTorch/Triton stack,
keep the image-provided `torch` instead of resolving a replacement wheel. For
kernel validation on those hosts, use:

```bash
scripts/run_cuda_kernel_checks.sh
```

The script creates a `--system-site-packages` venv, installs OrbitQuant without
dependencies, installs only the lightweight test/runtime packages that are
missing, and emits `REMOTE_STAGE` markers around each long step.

## Load A Published Artifact

Published OrbitQuant model repos are component artifacts. Load the source
Diffusers pipeline first, download the OrbitQuant artifact, then patch the
pipeline component:

```python
import torch
from diffusers import DiffusionPipeline
from huggingface_hub import snapshot_download
from orbitquant import load_quantized_pipeline_component

base_model = "black-forest-labs/FLUX.1-schnell"
artifact_id = "WaveCut/FLUX.1-schnell-OrbitQuant-W4A4"

artifact_dir = snapshot_download(artifact_id, repo_type="model")
pipe = DiffusionPipeline.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16,
)
load_quantized_pipeline_component(
    pipe,
    artifact_dir,
    component="transformer",
    device="cuda",
)
pipe.to("cuda")

result = pipe(
    prompt="A clean product photo of a red ceramic mug on a wooden desk",
    num_inference_steps=4,
    guidance_scale=0.0,
)
```

Use the model-specific Diffusers class when available. The published artifact
cards include full code-only examples for the matching pipeline and native
generation settings.

## Quantize A Pipeline Component

```python
import torch
from diffusers import DiffusionPipeline
from orbitquant import (
    OrbitQuantConfig,
    quantize_pipeline,
    save_quantized_pipeline_component,
)

pipe = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-4B",
    torch_dtype=torch.bfloat16,
)

config = OrbitQuantConfig(
    weight_bits=4,
    activation_bits=4,
    target_policy="flux2",
    activation_kernel_backend="triton_cuda",
)
summary = quantize_pipeline(
    pipe,
    config,
    component="transformer",
    quantization_device="cuda",
    staging_mode="component",
)
save_quantized_pipeline_component(
    pipe,
    "./artifacts/flux2-klein-w4a4",
    config=config,
    component="transformer",
    source_model_id="black-forest-labs/FLUX.2-klein-4B",
    source_revision="resolved-revision",
    source_license="apache-2.0",
    summary=summary,
)
```

Equivalent CLI:

```bash
orbitquant quantize \
  --model-id black-forest-labs/FLUX.2-klein-4B \
  --component transformer \
  --target-policy flux2 \
  --weight-bits 4 \
  --activation-bits 4 \
  --activation-kernel-backend triton_cuda \
  --device cuda \
  --staging-mode component \
  --output ./artifacts/flux2-klein-w4a4

orbitquant validate-artifact --artifact ./artifacts/flux2-klein-w4a4
```

## Native Targets

The current target matrix follows the agreed native settings:

| Suite | Source model | Pipeline class | Native setting | Bit settings |
| --- | --- | --- | --- | --- |
| `flux2-native` | `black-forest-labs/FLUX.2-klein-4B` | `Flux2KleinPipeline` | 1024x1024, 4 steps, guidance 1.0 | W4A4, W3A3, W2A4, W2A3 |
| `flux1-schnell-native` | `black-forest-labs/FLUX.1-schnell` | `FluxPipeline` | 1024x1024, 4 steps, guidance 0.0 | W4A4, W3A3, W2A4, W2A3 |
| `z-image-native` | `Tongyi-MAI/Z-Image-Turbo` | `ZImagePipeline` | 1024x1024, 10 steps, guidance 0.0 | W4A4, W3A3, W2A4, W2A3 |
| `wan-native` | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | `WanPipeline` | 832x480, 81 frames, 50 steps, guidance 5.0 | W4A6, W4A4 |

Small range smoke generations are not used as quality evidence. User-facing
comparison assets are generated at the native settings above.

## Comparison Assets

Artifacts can include BF16-vs-OrbitQuant visual comparisons under `assets/`.
When these files are present and recorded in the manifest, the generated
Hugging Face model card embeds them directly:

- `assets/original_vs_orbitquant_*.webp`
- `assets/*generation_comparison_matrix.webp`
- Wan contact-sheet comparisons for video artifacts

Use `orbitquant upload-artifact --upload-profile compact` for model repos. The
compact profile promotes final report comparison matrices into `assets/` and
omits `reports/` logs and raw eval dumps from the uploaded repository. Local raw
logs and intermediate eval outputs should stay outside the model repo. Model
cards should describe the artifact, show how to use it, state the source
model/provenance, and display final comparison assets.

## Artifact Layout

Each artifact is intentionally inspectable without executing code:

- `model.safetensors`: packed OrbitQuant and INT4 module tensors.
- `quantization_config.json`: serialized `OrbitQuantConfig`.
- `orbitquant_manifest.json`: source provenance, policies, module lists, and
  checksums.
- `orbitquant_codebooks.safetensors`: Lloyd-Max centroids and boundaries.
- `orbitquant_rotations.safetensors`: deterministic RPBH rotation metadata.
- `benchmark/*.jsonl` and `benchmark/*.csv`: imported metrics and native
  generation metadata.
- `assets/`: selected final comparison images, contact sheets, and videos.
- `SHA256SUMS`: checksums for artifact files.

## Kernels

The default correctness runtime still uses BF16 PyTorch matmul after low-bit
dequantization, so disk compression and runtime VRAM/latency are reported
separately. The CUDA/Triton backend currently covers the heavy OrbitQuant
stages around that matmul:

- runtime activation norm, RPBH/FWHT rotation, codebook lookup, and rescale;
- packed weight dequantization;
- low-bit pack/unpack helpers;
- offline weight RPBH/FWHT codebook indexing with direct low-bit packing;
- AdaLN INT4 RTN quantize/pack/dequant.

Run `orbitquant kernel-info` to inspect the active backend and
`scripts/run_cuda_kernel_checks.sh` to run the CUDA kernel test and benchmark
gate on a GPU host. Fully fused low-bit matmul is not enabled in the current
runtime mode.

## License

Code in this repository is Apache-2.0. Quantized artifacts also record the
source model license in `orbitquant_manifest.json` and in the generated model
card.
