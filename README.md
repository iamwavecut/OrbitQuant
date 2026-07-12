# OrbitQuant

OrbitQuant is a calibration-free post-training quantizer for transformer linear
projections. It implements the method from
[OrbitQuant: Data-Agnostic Quantization for Image and Video Diffusion Transformers](https://arxiv.org/abs/2607.02461)
and exposes it through Hugging Face Transformers, Diffusers, and a direct
PyTorch API.

The implementation is clean-room and Apache-2.0 licensed.

## Features

- Automatic coverage of registered linear-compatible modules in transformer
  backbones, independent of model class or modality.
- Built-in support for `torch.nn.Linear` and Hugging Face `Conv1D` projections.
- Public adapters for custom `F.linear`-equivalent module types and transposed
  weight layouts.
- RPBH rotation, exact unit-sphere Lloyd-Max codebooks, packed 2/3/4/6/8-bit
  weights, and online activation quantization without calibration data.
- Model-specific policies for paper-sensitive AdaLN and output-layer handling.
- Compact `safetensors` artifacts and Hugging Face `save_pretrained()` /
  `from_pretrained()` integration.
- Packed-weight CUDA, Triton, and Metal inference paths that avoid a full
  dequantized weight matrix.
- `torch.compile(fullgraph=True)` support: quantized forwards run behind a
  registered custom op with a fake implementation, so compiled models match
  eager output exactly. The forward is also CUDA-Graph-capturable;
  `torch.compile(mode="reduce-overhead")` removes the per-launch overhead that
  dominates small decode batches (measured 1.6-5.5x at 1-32 rows on
  A40/RTX 4090).

Embeddings, timestep modules, task heads, and common final projections are
kept in source precision by default. Every automatic decision is available as
a machine-readable inventory before quantization.

## Install

```bash
pip install "orbitquant[hf]"
```

Install the CUDA/Triton and local-kernel loader dependencies with:

```bash
pip install "orbitquant[hf,kernels]"
```

## Quantize A Transformers Model

Importing `orbitquant` registers the backend with supported Transformers and
Diffusers versions. The default `target_policy="auto"` selects a known
paper policy where applicable and otherwise uses the universal policy.

```python
import orbitquant
from transformers import AutoModel
from orbitquant import OrbitQuantConfig

model_id = "google/vit-base-patch16-224"
model = AutoModel.from_pretrained(
    model_id,
    quantization_config=OrbitQuantConfig(target_policy="auto"),
    low_cpu_mem_usage=True,
)
```

This is checkpoint-level conversion, not post-load module replacement. For each
target weight OrbitQuant reads aligned row tiles from safetensors, closes that
source mapping, and feeds a temporary packed shard back to the normal
Transformers loader. Peak model-side memory is bounded by packed resident state,
one source row tile, quantization workspace, intentionally skipped state, and
runtime overhead. The full source transformer is never resident at once.

Save an offline packed artifact without loading the source model again:

```python
model.save_pretrained("./model-orbitquant-w4a4", max_shard_size="2GB")
```

Load the packed artifact after importing the backend:

```python
import orbitquant
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "./model-orbitquant-w4a4",
    device_map="auto",
)
```

Named recipes are `w4a4`, `w3a3`, `w2a4`, `w2a3`, and `w4a6`. They create a
normal `OrbitQuantConfig`, so every field can be overridden.

Bounded on-the-fly conversion requires safetensors and the Transformers 5
weight-conversion APIs. Local paths and Hub model IDs use the normal
`revision`, `variant`, token/auth, `device_map`, `max_memory`, CPU, and disk
offload arguments. Pickle/`.bin` checkpoints fail with an actionable error;
use the explicit post-load helper only when accepting that it has no
checkpoint-level memory guarantee.

## Inspect Coverage

Inspect a model before replacing modules:

```python
from orbitquant import inspect_linear_module_policy, recipe
from transformers import AutoModel

model = AutoModel.from_pretrained("your-org/your-transformer")
report = inspect_linear_module_policy(model, recipe("w4a4"))

print(report["action_counts"])
print(report["quantized_modules"])
print(report["skipped_modules"])
print(report["unsupported_linear_modules"])
```

The universal policy quantizes every registered linear-compatible module except
known embeddings, timestep modules, task/output heads, and explicit skips. It
does not depend on names such as `layers`, `blocks`, or a particular model
class.

Use `modules_to_convert` as an allowlist and define AdaLN/skips with exact names,
substrings, or glob patterns:

```python
from orbitquant import OrbitQuantConfig

config = OrbitQuantConfig(
    modules_to_convert=["backbone.*.projection"],
    modules_to_use_adaln=["backbone.*.modulation"],
    modules_to_not_convert=["*.sensitive_output"],
)
```

Explicit dtype overrides remain available through `modules_dtype_dict`.

## Quantize An Instantiated Module

For ordinary PyTorch models or frameworks that do not use Hugging Face loading
hooks, or as an explicit compatibility fallback:

```python
from orbitquant import quantize_model, recipe

summary = quantize_model(
    model,
    recipe("w4a4"),
    quantization_device="cuda",
)
print(summary.quantized_modules)
```

Here `staging_mode="streaming"` limits device staging per module only. The
source model is already loaded, so this helper does not make a bounded host-RAM
claim.

The replacement supports arbitrary leading dimensions and treats the final
dimension as `in_features`, including sequence, image-token, and video-token
layouts.

## Custom Linear Modules

Register a module whose forward operation is equivalent to `F.linear`. The
adapter describes only its source weight layout and feature attributes:

```python
from orbitquant import register_linear_adapter

register_linear_adapter(
    MyLinear,
    weight_layout="in_out",
    in_features_attr="input_size",
    out_features_attr="output_size",
)
```

OrbitQuant stores every replacement in canonical `[out_features, in_features]`
layout. Modules with additional routing, tensor-parallel communication, sparse
expert selection, or non-linear forward semantics need an architecture-aware
adapter; the inspection report lists unregistered linear candidates instead of
silently replacing them.

## Diffusers

Quantize the denoiser while loading an arbitrary compatible pipeline:

```python
import torch
import orbitquant
from diffusers import DiffusionPipeline
from orbitquant import (
    OrbitQuantConfig,
    build_diffusers_pipeline_quantization_config,
)

model_id = "black-forest-labs/FLUX.1-schnell"
qconfig = build_diffusers_pipeline_quantization_config(
    OrbitQuantConfig(target_policy="auto"),
    components="transformer",
)
pipe = DiffusionPipeline.from_pretrained(
    model_id,
    quantization_config=qconfig,
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()
```

Use sequential offload without an OrbitQuant-specific hook or preparation step:

```python
pipe = DiffusionPipeline.from_pretrained(
    model_id,
    quantization_config=qconfig,
    torch_dtype=torch.bfloat16,
)
pipe.enable_sequential_cpu_offload()
```

Save the packed pipeline for later prequantized loading:

```python
pipe.save_pretrained("./pipeline-orbitquant", safe_serialization=True)
```

The default component is only `transformer`; text encoders require an explicit
component opt-in. Diffusers uses the same safetensors row-sliced conversion and
normal `PipelineQuantizationConfig` loading path. Unknown architectures receive
structural `target_policy="auto"` coverage, not a quality guarantee; inspect the
policy and validate output quality before publishing an artifact.

Published FLUX, Z-Image, and Wan repositories are compact Diffusers component
artifacts. Their model cards contain the matching pipeline code and validation
evidence.

Load a published component artifact together with its recorded source pipeline:

```python
import torch
from huggingface_hub import snapshot_download
from orbitquant import load_quantized_pipeline_from_artifact

artifact_dir = snapshot_download(
    "WaveCut/FLUX.1-schnell-OrbitQuant-W4A4",
    repo_type="model",
)
pipe = load_quantized_pipeline_from_artifact(
    artifact_dir,
    torch_dtype=torch.bfloat16,
    runtime_mode="auto_fused",
)
```

## Reproduce Load Memory

The memory harness runs ordinary, on-the-fly, and prequantized loads in separate
processes and reports RSS separately from mmap virtual size:

```bash
python scripts/measure_streaming_load_memory.py \
  --framework transformers \
  --model-id google/vit-base-patch16-224 \
  --prequantized-model-id ./vit-base-orbitquant \
  --torch-dtype bfloat16 \
  --output ./memory-results.json
```

One CPU/MPS host run with Torch 2.12.1 measured:

| Load | Peak RSS | Virtual/mmap peak | Resident state | Disk artifact | Wall time |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 source | 705.7 MiB | 416.3 GiB | 164.8 MiB | 330.3 MiB | 1.57 s |
| On-the-fly OrbitQuant | 436.6 MiB | 416.4 GiB | 43.4 MiB | 330.3 MiB source | 3.01 s |
| Prequantized | 331.3 MiB | 415.9 GiB | 43.4 MiB | 43.5 MiB | 0.14 s |

The streaming run processed 324.0 MiB of selected source tensors with a 9.0
MiB largest source tensor, retained no full dequantized cache, and reported no
source-release failure. CUDA allocated/reserved and NVML fields are emitted
when CUDA and NVML are available; they were unavailable for this Mac run.

## Packed Runtime

`runtime_mode="auto_fused"` is the default:

| Device | Dispatch |
| --- | --- |
| CUDA | Native activation kernel plus packed W4A4 tensor-core path; native or Triton packed fallback |
| MPS | Native packed Metal package |
| CPU | Native exact activation, packed low-bit matmul, and packed INT4 AdaLN when the CPU package is importable; PyTorch reference fallback otherwise |

CUDA and MPS do not silently materialize a full BF16/FP16 weight matrix in
`auto_fused`. If no packed backend is available, the error includes the missing
backend and installation guidance. CPU retains a compatibility fallback when
the optional native package is absent.

End-to-end FLUX.2 Klein 9B numbers for the released package are recorded in
[docs/flux2-klein-9b-sdnq-vs-orbitquant.md](docs/flux2-klein-9b-sdnq-vs-orbitquant.md)
(0.5.0 on an NVIDIA A40: 4.00 s hot median at 1024x1024/4 steps with
13.43 GB peak CUDA allocated memory).

Use the explicit reference path for compatibility or numerical debugging:

```python
config = orbitquant.recipe("w4a4", runtime_mode="dequant_bf16")
```

On CUDA compute capability 8.0 or newer, the W4A4 fast path fuses token norm,
RPBH/FWHT, and codebook assignment in the native package, decodes only a bounded
output-channel chunk of packed weights to INT8, and uses the Torch CUTLASS
tensor-core matmul. It never materializes the full BF16/FP16 weight matrix.
The existing direct packed CUDA MMA kernel remains the fallback for compatible
W4A4 shapes when CUTLASS INT8 matmul is unavailable.

The optimized CUDA path maps the fixed Lloyd-Max centroids to a symmetric INT8
surrogate plus one scalar per codebook. Packed checkpoint indices and artifact
size are unchanged. Use `dequant_bf16` when exact Lloyd-Max centroid evaluation
is required.

Build the ABI3 native package locally without Kernel Hub:

```bash
cd native-kernels/orbitquant-packed-matmul
nix --option sandbox relaxed run .#build-and-copy -L
export PYTHONPATH="$PWD/build/<matching-torch-backend-platform-variant>:$PYTHONPATH"
```

PyTorch 2.9 CUDA users can reduce allocator reservation during native diffusion
inference by setting the allocator before Python starts:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python generate.py
```

CUDA and Metal variants must match the Torch/backend tuple. The CPU variant
targets the LibTorch Stable ABI 2.11, but remains specific to its operating
system, C++ runtime, and architecture. See
[`docs/kernel-audit.md`](docs/kernel-audit.md) for tested shapes, benchmark
methodology, and local package verification.

## Validated Architecture Coverage

The integration suite instantiates and inventories encoder-only, decoder-only,
encoder-decoder, causal LM, and vision transformer families:

| Family | Projection type |
| --- | --- |
| BERT | `torch.nn.Linear` |
| GPT-2 | Hugging Face `Conv1D` with transposed source weights |
| Llama | `torch.nn.Linear`, including GQA projections |
| T5 | encoder and decoder `torch.nn.Linear` projections |
| ViT | vision transformer `torch.nn.Linear` projections |

The paper-aligned release artifacts remain FLUX.1-schnell, Z-Image-Turbo, and
Wan 2.1 T2V. FLUX.2 Klein is an additional validated diffusion target.

Architecture coverage means the model can be discovered, quantized, executed,
saved, and restored without model-name-specific code. It does not guarantee a
quality-preserving bit setting. OrbitQuant was evaluated in the paper on image
and video diffusion transformers; language and classification models can be
more sensitive, and their quality must be measured before publishing a
checkpoint. The library exposes module overrides for such recipes but does not
silently substitute a different quantization algorithm.

## Method Conformance

The implementation follows the paper's shared data-agnostic basis:

- RPBH permutation, Rademacher signs, block FWHT, and orthonormal scaling.
- Offline folded weight rotation with BF16 row norms and quantized unit
  directions.
- Online per-token norm, normalized activation rotation, nearest-centroid
  quantization, and rescaling.
- One fixed codebook per `(input dimension, bit width, algorithm version)` and
  no prompt, timestep, or calibration statistics.
- INT4 group-64 RTN for model policies that identify dynamic AdaLN projections.

The detailed requirement matrix is in
[`docs/paper-methodology-audit.md`](docs/paper-methodology-audit.md).

## Citation

If you use OrbitQuant in research, cite the original paper describing the
method:

> Donghyun Lee, Jitesh Chavan, Duy Nguyen, Sam Huang, Liming Jiang,
> Priyadarshini Panda, Timo Mertens, and Saurabh Shukla. "OrbitQuant:
> Data-Agnostic Quantization for Image and Video Diffusion Transformers."
> arXiv:2607.02461, 2026. https://doi.org/10.48550/arXiv.2607.02461

```bibtex
@misc{lee2026orbitquant,
  title         = {OrbitQuant: Data-Agnostic Quantization for Image and Video Diffusion Transformers},
  author        = {Lee, Donghyun and Chavan, Jitesh and Nguyen, Duy and Huang, Sam and Jiang, Liming and Panda, Priyadarshini and Mertens, Timo and Shukla, Saurabh},
  year          = {2026},
  eprint        = {2607.02461},
  archiveprefix = {arXiv},
  primaryclass  = {cs.CV},
  doi           = {10.48550/arXiv.2607.02461},
  url           = {https://arxiv.org/abs/2607.02461},
}
```

## Development

```bash
uv sync --extra hf --extra dev
uv run pytest -q
uv run ruff check .
```

## License

OrbitQuant is licensed under Apache-2.0. Model artifacts retain the license and
provenance of their source checkpoints.
