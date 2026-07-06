# OrbitQuant

Pre-release clean-room implementation of OrbitQuant for diffusion transformer
post-training quantization.

OrbitQuant is based on the paper
[OrbitQuant: Data-Agnostic Quantization for Image and Video Diffusion Transformers](https://arxiv.org/abs/2607.02461).
The package is intended to provide Hugging Face Diffusers/Transformers adapters,
compact quantized artifacts, and native-resolution evaluation scripts.

This repository is currently private and experimental. Do not treat the current
runtime as fully optimized low-bit inference until the full CUDA/MPS fusion work
lands.

## Initial Scope

- Calibration-free RPBH + Lloyd-Max weight/activation quantization.
- Transformer-only quantization for diffusion pipelines.
- BF16 text encoders, VAE, embeddings, timestep MLP, and final heads.
- AdaLN modulation projections as INT4 RTN weight-only by default.
- Native eval settings for FLUX.2 Klein, FLUX.1-schnell, Z-Image-Turbo, and
  Wan2.1-T2V-1.3B.

## CLI

Create a compact quantized transformer component artifact:

```bash
orbitquant quantize \
  --model-id black-forest-labs/FLUX.2-klein-4B \
  --component transformer \
  --target-policy flux2 \
  --weight-bits 4 \
  --activation-bits 4 \
  --activation-kernel-backend auto \
  --output ./artifacts/flux2-klein-w4a4
```

Run a native-resolution generation check:

```bash
orbitquant generate \
  --suite flux2-native \
  --prompt "A small red cube on a white table" \
  --output ./artifacts/native-smoke/flux2 \
  --bit-setting W4A4 \
  --activation-kernel-backend auto
```

Validate an artifact before publishing or moving it:

```bash
orbitquant validate-artifact --artifact ./artifacts/flux2-klein-w4a4
```

Prepare a native GPU run on an RTX PRO 6000 96GB pod:

```bash
hf auth whoami
orbitquant native-plan --output-root ./artifacts/native --seeds 0
orbitquant native-script \
  --output-root ./artifacts/native \
  --seeds 0 \
  --device cuda \
  --dtype bfloat16 \
  --activation-kernel-backend triton_cuda \
  --resume \
  > run-native.sh
bash run-native.sh
```

`native-script` emits a preflight block (`hf auth whoami`, `hf env`, CUDA,
package-version, disk, and `hf models info` access checks) before quantizing and
running native `generate-pack` jobs. The generated matrix uses the native
settings in this repository: FLUX/Z-Image at 1024x1024 and Wan at 832x480,
81 frames, 50 steps, CFG 5.0. It does not create range smoke jobs. With
`--resume`, the script skips quantization for existing valid artifacts and adds
`generate-pack --resume-existing` so completed sample outputs are not regenerated.
The script finishes with `orbitquant report`, writing Markdown/CSV readiness
outputs under `reports/native` by default.

Print the external GenEval/VBench jobs needed after native samples are created:

```bash
orbitquant external-eval-plan \
  --output-root ./artifacts/native \
  --metrics-root ./metrics/native
```

Validate a generated native sample and its metadata after copying artifacts back
from a pod:

```bash
orbitquant validate-generation \
  --suite flux2-native \
  --output ./artifacts/native/flux2-native-w4a4/assets/flux2-native_seed0_W4A4_simple-object.png \
  --seed 0 \
  --bit-setting W4A4
```

After external GenEval or VBench runs finish, import their JSON metrics into the
artifact so reports and checksums stay consistent:

```bash
orbitquant record-metrics \
  --artifact ./artifacts/native/flux1-schnell-native-w4a4 \
  --split orbitquant \
  --metrics-json ./geneval-flux1-w4a4.json \
  --metric-prefix geneval \
  --suite flux1-schnell-native \
  --seed 0 \
  --bit-setting W4A4
```

Native reports also write `tables/missing_required_metrics.csv` and print
`missing_required_metric_count` from `orbitquant report`. For paper targets,
this flags missing GenEval/VBench fields separately from completed metric rows.

## Python API

```python
import torch
from diffusers import DiffusionPipeline
from orbitquant import (
    OrbitQuantConfig,
    load_quantized_pipeline_component,
    quantize_pipeline,
    save_quantized_pipeline_component,
)

pipe = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-4B",
    torch_dtype=torch.bfloat16,
)
config = OrbitQuantConfig(weight_bits=4, activation_bits=4, target_policy="flux2")
summary = quantize_pipeline(pipe, config, component="transformer")
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

restored_pipe = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-4B",
    torch_dtype=torch.bfloat16,
)
load_quantized_pipeline_component(
    restored_pipe,
    "./artifacts/flux2-klein-w4a4",
    component="transformer",
)
```

Current artifacts include:

- `model.safetensors`: packed quantized module state.
- `quantization_config.json`: serialized `OrbitQuantConfig`.
- `orbitquant_manifest.json`: source provenance, quantization policy, tensor
  shapes, runtime settings, and checksums.
- `orbitquant_codebooks.safetensors`: Lloyd-Max centroids and boundaries for
  the dimensions and bit-widths used by the artifact.
- `orbitquant_rotations.safetensors`: deterministic RPBH permutation, inverse
  permutation, signs, and normalization tensors.
- `prompts.json`: prompt/eval prompt container, initially empty until native
  generation or metric runs populate it.
- `benchmark/summary.json`: benchmark/eval status summary, initially marked
  `not_run` for newly quantized artifacts.
- `SHA256SUMS`: checksums for all artifact files.

`activation_kernel_backend` accepts `auto`, `cpu`, `mps`, and `triton_cuda`.
The current `triton_cuda` path keeps norm and RPBH rotation in PyTorch but uses
real Triton kernels for codebook lookup, norm rescale, and packed weight
dequantization. On Apple Silicon, `mps` uses Metal shaders for the same
lookup/rescale and packed weight dequant stages when `torch.mps.compile_shader`
is available. Otherwise it falls back to the reference PyTorch path on MPS
tensors. Full fused norm+RPBH+lookup+matmul kernels are still separate work.
The reference PyTorch path remains the correctness baseline.

Inspect backend availability and optimization status on the current machine:

```bash
orbitquant kernel-info
```

`kernel-info` reports whether `mps` is using `metal_codebook_rescale` or the
`torch_reference_mps` fallback on the current machine. It reports `triton_cuda`
as partial optimization because activation norm/RPBH and matmul are not fused
today. The `weight_dequant_optimized` field records whether packed weight
dequantization avoids the CPU unpack path for that backend.

## License

The code in this repository is Apache-2.0. Quantized model artifacts must record
and respect the license of their source model.
