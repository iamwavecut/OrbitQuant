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
  --activation-kernel-backend triton_cuda \
  --device cuda \
  --output ./artifacts/flux2-klein-w4a4
```

Run a native-resolution generation check:

```bash
orbitquant generate \
  --suite flux2-native \
  --prompt "A small red cube on a white table" \
  --output ./artifacts/native-smoke/flux2 \
  --bit-setting W4A4 \
  --activation-kernel-backend triton_cuda
```

Validate an artifact before publishing or moving it:

```bash
orbitquant validate-artifact --artifact ./artifacts/flux2-klein-w4a4
```

Upload a validated artifact to a private Hugging Face model repo:

```bash
orbitquant upload-artifact \
  --artifact ./artifacts/flux2-klein-w4a4 \
  --repo-id WaveCut/FLUX.2-klein-4B-OrbitQuant-W4A4 \
  --replace-repo-files
```

`upload-artifact` validates required files, checksums, `SHA256SUMS`, and tensor
shapes before upload. It creates private model repos by default; use `--public`
only after the artifact and native eval report are ready for release. For very
large artifacts that were already deep-validated locally, `--skip-tensor-validation`
keeps checksum validation but avoids loading all safetensors before upload.

Audit the expected private Hugging Face artifact matrix without downloading
large weight files:

```bash
orbitquant audit-hf-artifacts \
  --namespace WaveCut \
  --output ./reports/native/hf-artifact-audit.json
```

The audit checks expected repo names from the native suite matrix, required
artifact files, manifest/model-bit consistency, native smoke metadata, and
paper-target required metrics. It reports `artifact_ready`,
`native_smoke_ready`, and `release_eval_ready` separately so visual smoke
artifacts are not confused with GenEval/VBench completion.

Repair metadata-only provenance on an existing local artifact without rewriting
packed weights:

```bash
orbitquant repair-artifact-metadata \
  --artifact ./artifacts/flux2-klein-w4a4 \
  --quantization-device cuda \
  --weight-quantization-backend triton_cuda
```

This refreshes `orbitquant_manifest.json`, `model_index.json`,
`benchmark/summary.json`, `README.md`, and `SHA256SUMS`. Use it for schema or
provenance fixes before re-running `upload-artifact`.

Repair the same provenance fields directly in existing Hugging Face artifact
repos without downloading or reuploading `model.safetensors`:

```bash
orbitquant repair-hf-artifact-metadata \
  --namespace WaveCut \
  --quantization-device cuda \
  --weight-quantization-backend triton_cuda \
  --dry-run
```

Pass `--repo-id WaveCut/...` to repair one repo, or omit it to repair the native
suite matrix. The command creates one small metadata-only commit per changed repo
and preserves existing checksum entries for large unchanged files.

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
running native `generate-pack` jobs. It also emits a kernel preflight block with
`orbitquant kernel-info` and short `orbitquant kernel-bench` runs for the unique
bit settings in the matrix. This makes first-use CUDA/Triton compilation
visible before the long quantization jobs and keeps JIT CPU time separate from
hot kernel timings. The generated matrix uses the native
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

Print the matching executable script when the external metric tools are already
installed in the eval environment:

```bash
orbitquant external-eval-script \
  --output-root ./artifacts/native \
  --metrics-root ./metrics/native \
  --report-output ./reports/native \
  > run-external-eval.sh
```

For GenEval, set:

```bash
export GENEVAL_DIR=/path/to/geneval
export GENEVAL_OBJECT_DETECTOR=/path/to/mask2former/checkpoints
```

The generated script runs `orbitquant export-geneval` to create the upstream
`00000/metadata.jsonl` and `00000/samples/*.png` folder layout, then calls
`${GENEVAL_DIR}/evaluation/evaluate_images.py`, summarizes the resulting
JSONL with `orbitquant summarize-geneval-results`, and records the normalized
JSON metrics into the artifact. GenEval export requires prompt records with
GenEval metadata (`tag`, `include`, and optional `exclude`); the visual prompt
pack fails loudly instead of pretending to be a GenEval suite.

Generate GenEval-compatible native samples with the upstream GenEval metadata
JSONL before running the external script:

```bash
curl -L \
  https://raw.githubusercontent.com/djghosh13/geneval/main/prompts/evaluation_metadata.jsonl \
  -o ./metrics/native/geneval-evaluation_metadata.jsonl

orbitquant generate-pack \
  --suite flux1-schnell-native \
  --artifact ./artifacts/native/flux1-schnell-native-w4a4 \
  --prompt-metadata-jsonl ./metrics/native/geneval-evaluation_metadata.jsonl \
  --seeds 0 \
  --device cuda \
  --dtype bfloat16 \
  --resume-existing
```

For a quick schema-only dry run, use `--prompt-pack geneval-smoke`; do not use
that smoke pack for release metrics.

For VBench, the generated script runs `orbitquant export-vbench`, then
`vbench evaluate --mode custom_input`, then `orbitquant summarize-vbench-results`.
The custom-input path covers the VBench dimensions supported by upstream for
arbitrary videos: subject consistency, background consistency, motion smoothness,
dynamic degree, aesthetic quality, and imaging quality. Paper-style standard
VBench dimensions such as scene and overall consistency require generating the
official VBench prompt suite, not only exporting the visual comparison pack.

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
summary = quantize_pipeline(
    pipe,
    config,
    component="transformer",
    quantization_device="cuda",
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

restored_pipe = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-4B",
    torch_dtype=torch.bfloat16,
)
load_quantized_pipeline_component(
    restored_pipe,
    "./artifacts/flux2-klein-w4a4",
    component="transformer",
    device="cuda",
)
```

`quantization_device` controls where full-precision weights are rotated,
codebook-indexed, and low-bit packed during artifact creation. It is deliberately
not serialized into `OrbitQuantConfig`; artifacts record the quantization method,
bits, policy, runtime mode, and kernel backend, while the device belongs to the
current machine/run. Passing `quantization_device="cuda"` or `"mps"` fails loudly
if that accelerator is unavailable.

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

Lloyd-Max codebooks are data-agnostic and cached outside the current Python
process after first generation. By default the cache is written under
`$XDG_CACHE_HOME/orbitquant/codebooks` or `~/.cache/orbitquant/codebooks`.
Set `ORBITQUANT_CODEBOOK_CACHE_DIR` to choose another cache directory, or
`ORBITQUANT_DISABLE_CODEBOOK_DISK_CACHE=1` to force regeneration. This avoids
repeating the CPU-only offline codebook solve for every new quantization run.

`activation_kernel_backend` accepts `auto`, `cpu`, `mps`, and `triton_cuda`.
The current `triton_cuda` path uses real Triton kernels for runtime activation
norm calculation, RPBH/FWHT rotation, codebook lookup, norm rescale, packed
weight dequantization, offline low-bit pack/unpack, and offline weight
RPBH/FWHT-to-codebook indexing with direct packed-byte output during artifact
creation. It also uses Triton for
AdaLN INT4 RTN group scale calculation, quantize+pack, and runtime dequantization
instead of round-tripping through CPU unpack/pack. CUDA low-bit unpack also stays
on CUDA and the CUDA pack path fails loudly if Triton is unavailable, instead of
silently falling back to the CPU packer. On Apple Silicon, `mps`
uses Metal shaders for runtime lookup/rescale and packed weight dequant stages
when `torch.mps.compile_shader` is available. Otherwise it falls back to the
reference PyTorch path on MPS tensors. Full fused activation+packed-weight matmul
kernels are still separate work. The reference PyTorch path remains the
correctness baseline.

Inspect backend availability and optimization status on the current machine:

```bash
orbitquant kernel-info
```

Benchmark the current `OrbitQuantLinear` kernel stages on the active machine:

```bash
orbitquant kernel-bench \
  --tokens 256 \
  --in-features 3072 \
  --out-features 3072 \
  --weight-bits 4 \
  --activation-bits 4 \
  --activation-kernel-backend triton_cuda \
  --device cuda \
  --dtype bfloat16
```

`kernel-info` reports whether `mps` is using `metal_codebook_rescale` or the
`torch_reference_mps` fallback on the current machine. It reports `triton_cuda`
as partial optimization because matmul is still the BF16 PyTorch linear path.
`kernel-bench` reports both `weight_quantize_pack_cold_ms` and
`weight_quantize_pack_hot_ms`: cold includes first-use backend compilation, while
hot measures the already compiled quantize+pack path. On Triton/CUDA, cold JIT
compilation can be CPU-heavy and may show little GPU activity in provider UIs;
the hot timing and `quantization_buffers` devices are the checks that the real
weight-index and low-bit packing work stays on CUDA.
The `weight_dequant_optimized` field records whether packed weight
dequantization avoids the CPU unpack path for that backend. The
`weight_pack_optimized` field records whether artifact creation can pack low-bit
weight indices without a CPU round-trip. The `lowbit_unpack_optimized` field
records whether direct low-bit unpack stays on the accelerator. The
`weight_quant_optimized` field records whether artifact creation can rotate/FWHT
weights and map them to Lloyd-Max codebook indices on the accelerator. The
`adaln_quant_optimized` and `adaln_dequant_optimized` fields report whether the
AdaLN INT4 RTN path avoids CPU quantize/pack and runtime unpack/dequant for that
backend.

## License

The code in this repository is Apache-2.0. Quantized model artifacts must record
and respect the license of their source model.
