# OrbitQuant

Clean-room implementation of OrbitQuant for calibration-free post-training
quantization of image and video diffusion transformers.

OrbitQuant is based on
[OrbitQuant: Data-Agnostic Quantization for Image and Video Diffusion Transformers](https://arxiv.org/abs/2607.02461).
The library targets Hugging Face Diffusers pipelines and stores compact
transformer-component artifacts that can be patched back into the original
pipeline.

The repository contains the Python package, quantization code, artifact tools,
and validation helpers. Release-ready model repositories must contain compact
artifacts, usage instructions, provenance, native validation summaries, and the
final comparison matrices embedded by the model card.

## What It Provides

- Data-agnostic RPBH rotation and Lloyd-Max codebook quantization.
- Weight and activation quantization for diffusion transformer linear modules.
- INT4 RTN weight-only handling for AdaLN modulation projections.
- Compact `safetensors` artifacts with manifest, codebooks, rotations, and
  checksums.
- Diffusers helper APIs for quantizing, saving, loading, and validating
  transformer-component artifacts.
- Native image/video generation helpers for producing final BF16-vs-OrbitQuant
  comparison assets.

Text encoders, VAE, embeddings, timestep MLPs, and final projection heads are
left in source precision by default.

## Install

Install the package with the Hugging Face runtime dependencies:

```bash
pip install "orbitquant[hf]"
```

The development branch can also be installed directly from GitHub:

```bash
pip install "orbitquant[hf] @ git+https://github.com/iamwavecut/OrbitQuant.git"
```

For local development from a checkout:

```bash
pip install -e ".[hf,eval,dev]"
```

For CUDA validation on hosts that already ship a vendor-matched PyTorch/Triton
stack, keep the image-provided `torch` instead of resolving a replacement wheel:

```bash
scripts/run_cuda_kernel_checks.sh
```

The script creates a `--system-site-packages` venv, installs OrbitQuant without
dependencies, installs only the lightweight test/runtime packages that are
missing, and runs the CUDA kernel validation gate.

## Load A Published Artifact

Published OrbitQuant model repos are component artifacts. Load the source
Diffusers pipeline and patch the quantized component from the artifact:

```python
import torch
from huggingface_hub import snapshot_download
from orbitquant import load_quantized_pipeline_from_artifact

artifact_id = "WaveCut/FLUX.1-schnell-OrbitQuant-W4A4"

artifact_dir = snapshot_download(artifact_id, repo_type="model")
pipe = load_quantized_pipeline_from_artifact(
    artifact_dir,
    torch_dtype=torch.bfloat16,
    runtime_mode="auto_fused",
)
pipe.enable_model_cpu_offload(device="cuda")

result = pipe(
    prompt="A clean product photo of a red ceramic mug on a wooden desk",
    num_inference_steps=4,
    guidance_scale=0.0,
)
```

The helper reads `model_index.json`, uses the model-specific Diffusers pipeline
class for the supported native targets when available, loads the recorded source
pipeline revision, and patches the artifact's recorded component. To control
the source pipeline class or load steps directly, use the lower-level component
loader:

```python
import torch
from diffusers import FluxPipeline
from huggingface_hub import snapshot_download
from orbitquant import load_quantized_pipeline_component

base_model = "black-forest-labs/FLUX.1-schnell"
artifact_id = "WaveCut/FLUX.1-schnell-OrbitQuant-W4A4"

artifact_dir = snapshot_download(artifact_id, repo_type="model")
pipe = FluxPipeline.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16,
)
load_quantized_pipeline_component(
    pipe,
    artifact_dir,
    component="transformer",
)
pipe.enable_model_cpu_offload(device="cuda")

result = pipe(
    prompt="A clean product photo of a red ceramic mug on a wooden desk",
    num_inference_steps=4,
    guidance_scale=0.0,
)
```

Use the model-specific Diffusers class when available. Published artifact cards
include code-only examples for the matching pipeline and native generation
settings.

These diffusion artifacts are not standalone `transformers.AutoModel` repos.
OrbitQuant integrates with Hugging Face configuration and quantization
mechanisms, but the published FLUX, Z-Image, and Wan artifacts are Diffusers
transformer-component artifacts and should be loaded through Diffusers.

## Hugging Face Native Loaders

For `transformers.PreTrainedModel` or `diffusers.ModelMixin` classes that own
their transformer linears directly, importing `orbitquant` registers the
`orbitquant` backend with the installed Hugging Face libraries:

```python
import torch
import orbitquant
from orbitquant import OrbitQuantConfig
from transformers import AutoModel

config = OrbitQuantConfig(
    weight_bits=4,
    activation_bits=4,
    target_policy="generic_dit",
)

model = AutoModel.from_pretrained(
    "./source-pretrained-model",
    torch_dtype=torch.bfloat16,
    quantization_config=config,
)
model.save_pretrained("./source-pretrained-model-orbitquant-w4a4")

restored = AutoModel.from_pretrained("./source-pretrained-model-orbitquant-w4a4")
```

This path is for Hugging Face-native model repositories. Published FLUX,
Z-Image, and Wan artifacts remain Diffusers pipeline-component artifacts and
use the component loader shown above.

## Runtime Modes

`OrbitQuantConfig` defaults to `runtime_mode="auto_fused"`. On CUDA this tries
the native packed low-bit matmul kernel first, then the Triton packed matmul
kernel. On MPS it requires the native Metal packed low-bit matmul kernel. On CPU
it uses the reference path.

Install the optional Hugging Face Kernels loader when using Hub-published native
kernels:

```bash
pip install "orbitquant[kernels]"
```

For local native-kernel builds, either add the matching
`native-kernels/orbitquant-packed-matmul/build/torch*-<backend>-<platform>`
directory to `PYTHONPATH`, or set `LOCAL_KERNELS` to that same built variant
directory before importing OrbitQuant:

```bash
# CUDA/Linux example.
export LOCAL_KERNELS="WaveCut/orbitquant-packed-matmul=/path/to/native-kernels/orbitquant-packed-matmul/build/torch29-cxx11-cu128-x86_64-linux"

# Metal/macOS example.
export LOCAL_KERNELS="WaveCut/orbitquant-packed-matmul=/path/to/native-kernels/orbitquant-packed-matmul/build/torch212-metal-aarch64-darwin"
```

`LOCAL_KERNELS` must point at a built variant directory that contains
`metadata.json`, not at the native-kernel source package root. Native-kernel
load errors include the current Torch/CUDA/platform runtime and, on CUDA/Linux,
the expected kernel-builder variant name.

CUDA and MPS `auto_fused` inference does not silently fall back to materializing
the full dequantized weight matrix. If a packed kernel is unavailable,
OrbitQuant raises an error that names the missing backend and points to the
explicit reference mode:

```python
from orbitquant import OrbitQuantConfig

config = OrbitQuantConfig(
    weight_bits=4,
    activation_bits=4,
    runtime_mode="dequant_bf16",  # compatibility/debug reference path
)
```

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

To use Diffusers' native pipeline-level quantization entrypoint, build a
`PipelineQuantizationConfig` for the component that should be quantized:

```python
import torch
from diffusers import DiffusionPipeline
from orbitquant import OrbitQuantConfig, build_diffusers_pipeline_quantization_config

config = OrbitQuantConfig(
    weight_bits=4,
    activation_bits=4,
    target_policy="flux2",
)
pipeline_quant_config = build_diffusers_pipeline_quantization_config(
    config,
    components="transformer",
)

pipe = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-4B",
    torch_dtype=torch.bfloat16,
    quantization_config=pipeline_quant_config,
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

For native generation on GPUs that cannot hold the full source pipeline at
once, add `--enable-model-cpu-offload` to `orbitquant generate` or
`orbitquant generate-pack`. The flag uses Diffusers model CPU offload instead
of moving the entire pipeline to the generation device.

## Inspect A Policy Inventory

Use `inspect-policy` to produce a JSON inventory of every linear module in a
pipeline component and the action selected by the current OrbitQuant policy.
For native suites this uses the transformer config and an empty-weight skeleton
by default, so it does not download or instantiate full model weights. This is
the lightweight audit artifact used to verify model-specific coverage before
quantization or generation:

```bash
orbitquant inspect-policy \
  --suite flux2-native \
  --dtype bfloat16 \
  --output ./reports/inventories/flux2-klein-policy.json
```

Use the saved inventory as a release check for a compact artifact manifest:

```bash
orbitquant validate-artifact \
  --artifact ./artifacts/flux2-klein-w4a4 \
  --policy-inventory ./reports/inventories/flux2-klein-policy.json
```

`orbitquant native-script` includes the same policy-inventory gate for every
artifact it quantizes: it writes suite inventories under
`reports/native/module-inventories/` and passes the matching inventory to each
`validate-artifact` command.

Published Hub repositories can be checked without downloading tensor weights:

```bash
orbitquant audit-hf-artifacts \
  --namespace WaveCut \
  --policy-inventory-root ./reports/native/module-inventories \
  --fail-on-artifact-regression \
  --output ./reports/native/hf-artifact-audit.json \
  --markdown-output ./reports/native/hf-artifact-audit.md
```

## Release Target Settings

Paper-aligned artifacts use these native target settings:

| Suite | Source model | Pipeline class | Native setting | Bit settings |
| --- | --- | --- | --- | --- |
| `flux1-schnell-native` | `black-forest-labs/FLUX.1-schnell` | `FluxPipeline` | 1024x1024, 4 steps, guidance 0.0 | W4A4, W3A3, W2A4, W2A3 |
| `z-image-native` | `Tongyi-MAI/Z-Image-Turbo` | `ZImagePipeline` | 1024x1024, 10 steps, guidance 0.0 | W4A4, W3A3, W2A4, W2A3 |
| `wan-native` | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | `WanPipeline` | 832x480, 81 frames, 50 steps, guidance 5.0 | W4A6, W4A4 |

Extra target artifacts use the same native-validation rules, but are not paper
reproduction targets:

| Suite | Source model | Pipeline class | Native setting | Bit settings |
| --- | --- | --- | --- | --- |
| `flux2-native` | `black-forest-labs/FLUX.2-klein-4B` | `Flux2KleinPipeline` | 1024x1024, 4 steps, guidance 1.0 | W4A4, W3A3, W2A4, W2A3 |

User-facing comparison assets are generated at the native settings above. Small
low-resolution checks are not accepted as published quality evidence.

## Release Metrics

Full GenEval and VBench runs are release evidence for paper reproduction,
metric-table, or leaderboard-style claims. Compact artifact readiness is tracked
separately through native comparison assets, native validation evidence,
manifests, and checksums.

For release-grade metric claims, import metrics from the upstream metric
runners. For image paper targets, first fetch the published artifacts into the
native local layout:

```bash
orbitquant fetch-hf-artifacts \
  --suite flux1-schnell-native \
  --output-root ./artifacts/native
```

Then generate native samples with the upstream GenEval metadata file:

```bash
orbitquant native-script \
  --suite flux1-schnell-native \
  --prompt-metadata-jsonl /path/to/GenEval/evaluation_metadata.jsonl \
  --output-root ./artifacts/native \
  --resume > run-native-flux1-geneval.sh
```

Then export generated samples, run the external metric runner, summarize
results, and import the metrics back into the artifacts:

```bash
orbitquant external-eval-script \
  --suite flux1-schnell-native \
  --output-root ./artifacts/native \
  --metrics-root ./metrics/native \
  --report-output ./reports/native > run-flux1-geneval-metrics.sh
```

For Wan, use the same `external-eval-script` path with `--suite wan-native` to
run VBench custom-input dimensions against the native 832x480, 81-frame videos.

Final publication gates are tracked in
[docs/release-gates.md](docs/release-gates.md).

## Comparison Assets

Artifacts can include a final BF16-vs-OrbitQuant comparison matrix under
`assets/`. When this file is present and recorded in the manifest, the generated
Hugging Face model card embeds it directly:

- `assets/*_generation_comparison_matrix.webp`

`orbitquant upload-artifact` uses the compact upload profile by default. The
compact profile promotes final comparison matrices into `assets/` and uploads
only the compact artifact files required for use, validation, and the model
card. Existing remote files are replaced by default so stale raw assets from
older uploads do not remain in the model repository. Model cards describe the
artifact, show how to use it, state source provenance, and display final
comparison matrices.

## Artifact Layout

Each artifact is intentionally inspectable without executing code:

- `model.safetensors`: packed OrbitQuant and INT4 module tensors.
- `quantization_config.json`: serialized `OrbitQuantConfig`.
- `orbitquant_manifest.json`: source provenance, policies, module lists, and
  checksums.
- `orbitquant_codebooks.safetensors`: Lloyd-Max centroids and boundaries.
- `orbitquant_rotations.safetensors`: deterministic RPBH rotation metadata.
- `benchmark/summary.json`: compact validation and imported-metric summary.
- Local validation outputs may include raw `benchmark/*.jsonl` and
  `benchmark/*.csv` records; compact published artifacts omit those raw files.
- `assets/`: final comparison matrices embedded by the model card.
- `SHA256SUMS`: checksums for artifact files.

## Kernels

The default runtime is `auto_fused`, which uses packed low-bit matmul on CUDA
or MPS when the matching kernel package is available. The explicit
`runtime_mode="dequant_bf16"` compatibility path materializes dequantized
weights before BF16 PyTorch matmul. Disk compression and runtime VRAM/latency
are reported separately. Kernel support is backend-specific:

- CPU is a correctness reference path only and does not claim optimized CPU
  kernels.
- MPS/Metal uses the native packed low-bit matmul package in `auto_fused` mode
  when it is importable. Lower-level Metal helpers also cover codebook
  lookup/rescale and packed weight dequantization.
- CUDA/Triton is partially optimized: Triton covers runtime activation norm,
  RPBH/FWHT rotation, codebook lookup/rescale, packed weight dequantization,
  low-bit pack/unpack, offline weight RPBH/FWHT codebook indexing with direct
  low-bit packing, AdaLN INT4 RTN quantize/pack/dequant, and packed-weight
  matmul via `runtime_mode="triton_packed_matmul"`.
- `runtime_mode="native_packed_matmul"` explicitly selects the native packed
  matmul package for CUDA/MPS when available.
- ROCm and XPU are not implemented backends in this repository.

Run `orbitquant kernel-info` to inspect backend capabilities. In that output,
`implemented_stage` is the code present in the package, while `optimized_stage`
is populated only when the backend is active in the current environment.
`scripts/run_cuda_kernel_checks.sh` runs the CUDA kernel test and benchmark gate
on a GPU host. It validates the native `orbitquant-packed-matmul` kernel package
with kernel-builder, loads that local build through Hugging Face `kernels`, and
benchmarks `native_packed_matmul` explicitly. Use
`scripts/run_mps_kernel_checks.sh` for the equivalent short MPS/Metal gate on
Apple Silicon. Full-model speedup claims still require backend-specific
benchmark artifacts for the target model and native settings. See
[docs/kernel-audit.md](docs/kernel-audit.md) for the release claim boundary.

To verify that a published OrbitQuant model artifact executes through the
native packed matmul runtime without running full image/video generation:

```bash
LOCAL_KERNELS="WaveCut/orbitquant-packed-matmul=/path/to/build/torch212-metal-aarch64-darwin" \
  uv run python scripts/verify_hf_kernel_model_artifact.py --device mps
```

The default artifact is `WaveCut/FLUX.2-klein-4B-OrbitQuant-W4A4`. The script
loads one quantized transformer projection from the artifact, runs it with
`runtime_mode="native_packed_matmul"`, compares it with `dequant_bf16`, and
prints JSON with finite-output, allclose, error, and packed-vs-materialized
weight storage fields.

## License

Code in this repository is Apache-2.0. Quantized artifacts also record the
source model license in `orbitquant_manifest.json` and in the generated model
card.
