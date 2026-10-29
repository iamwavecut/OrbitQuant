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
- Optional GPTQ rounding (`orbitquant.gptq`): calibrated codes in the same codebook and
  artifact format, for models that lose too much with nearest rounding.
- Optional quality knobs outside the paper recipe: a second RPBH round for narrow
  Hadamard blocks, per-token absmax INT8 activations (`recipe("w4a8")`), MSE-fitted
  AdaLN scales, a Lloyd-Max gain correction, and BF16 side columns for massive
  activation channels (see [Quality options](#quality-options-opt-in)).
- Compact `safetensors` artifacts and Hugging Face `save_pretrained()` /
  `from_pretrained()` integration.
- Packed-weight CUDA, Triton, and Metal inference paths that avoid a full
  dequantized weight matrix.
- Fused transformer checkpoints for Krea 2, FLUX.2, Ideogram 4, Qwen-Image 2.1,
  MiniMax-H3, Boogu-Image, FLUX.3 Klein, MOSS-Image and Fibo 2.0 (`orbitquant.fused`): each block runs
  grouped INT8 GEMMs with norm/modulation prologues and SwiGLU/gated-residual
  epilogues plus an INT8 attention kernel, and the checkpoint stores only the
  fused groups.
- `torch.compile(fullgraph=True)` support: quantized forwards run behind a
  registered custom op with a fake implementation, so compiled models match
  eager output exactly. The forward is also CUDA-Graph-capturable;
  `torch.compile(mode="reduce-overhead")` removes the per-launch overhead that
  dominates small decode batches.

Embeddings, timestep modules, task heads, and common final projections are
kept in source precision by default. Every automatic decision is available as
a machine-readable inventory before quantization.

## INT8 output heads (opt-in)

Output heads stay in source precision by default. For memory-bound decode the head can be
the largest per-token read (a 150k-row BF16 vocabulary head is 0.6 GB per step), so
OrbitQuant also offers an explicit per-row INT8 head:

```python
from orbitquant import quantize_output_heads

quantize_output_heads(model, names=("lm_head",))   # in-place Int8RowLinear replacement
```

`Int8RowLinear` stores per-row absmax INT8 weights plus FP32 scales (2x smaller than BF16),
quantizes activations per token at forward time, and serves 1..8-row decode with the native
`matmul_int8_rows` DP4A kernel (CUDA package 1.0.5+), falling back to `torch._int_mm` for
larger batches or when the native package is unavailable. All paths share the epilogue
`float(sum) * (x_scale * w_scale)`, so results are identical across them. Head quantization
changes logits at the INT8 rounding level (~0.4% relative); measure sampled outputs before
adopting it for a model.

## GPTQ rounding (opt-in calibration)

Plain OrbitQuant rounds every rotated weight to its nearest Lloyd-Max centroid and uses no
data. At 4 bits some models lose too much this way. `orbitquant.gptq` keeps the codebook,
rotation, bit width, BF16 row-norm scale and packed layout, and chooses the codes against the
Hessian `X^T X` of each projection's rotated calibration inputs: columns are rounded one at a
time (largest Hessian diagonal first) and each rounding error is compensated in the columns not
yet rounded. The result is an ordinary OrbitQuant artifact that loads and runs through every
runtime path unchanged.

```python
from orbitquant import quantize_model
from orbitquant.gptq import HessianCollector, load_rounded_codes, round_groups

groups = [["blocks.0.attn.q", "blocks.0.attn.k", "blocks.0.attn.v"], ["blocks.0.attn.o"]]
with HessianCollector(model, groups, config) as hessians:  # BF16 model on the GPU
    for batch in calibration:
        model(**batch)
codes = round_groups(model, groups, hessians, config)
quantize_model(model, config)
load_rounded_codes(model, codes)
```

- Projections in one group read the same input and share one Hessian.
- `update_state_dict(state, codes)` rewrites the tensors of an existing artifact's state dict.
- For a large model, collect and round a few blocks at a time. Groups of the same shape round
  together on the GPU (`batch`, `memory_budget`). `sequential=True` writes the rounded weights
  back, so later blocks are calibrated on rounded inputs.
- Hessians and updates run in full FP32 (TF32 is disabled inside). A Hessian that does not
  factorize gets 10x more damping, up to six times; `RoundedCodes.damping` reports the value
  used, and `rtn_error` / `gptq_error` report the Hessian-weighted output error of nearest and
  GPTQ rounding.

On the MOSS-Image 4-bit transformer, calibrated on 10 prompts at 12 sampling steps, the mean
LPIPS of 8 generated images against BF16 fell from 0.371 (nearest rounding) to 0.209; 6-bit
nearest rounding gives 0.166.

## Quality options (opt-in)

Every option below defaults to the paper recipe and is stored in the artifact config.

| Field | Values | Effect | Runtime cost |
| --- | --- | --- | --- |
| `rotation_rounds` | `1` (paper), `2`, `"auto"` | An independent first RPBH round in front of the paper round. The paper block `h` is the largest power of two dividing `d`: `d = 3360` gets `h = 32`, `d = 3840` and `d = 8960` get `h = 256`, so one round spreads an outlier over few coordinates. `"auto"` adds the round where `h < 512`. | One extra RPBH pass per activation, done in PyTorch: on an RTX 4090 it adds 0.7-0.8 ms per call at 256 tokens and 2.6-3.4 ms at 4096 tokens for `d = 3072`/`3840` (shared across Q/K/V of one input); not available on the fused block runtime. |
| `activation_quantizer` | `"lloyd_max"` (paper), `"absmax_int8"` | With `activation_bits=8`, per-token absmax INT8 of the rotated input, the numerics of the fused runtime's 8-bit activations. `recipe("w4a8")` selects it. No activation codebook is built or stored: exact 8-bit Lloyd-Max tables take minutes per width and do not converge for `d >= 7680`, so `"lloyd_max"` A8 fails on such inputs. | CUDA runs an INT8 GEMM against the packed W2/W4 weights, the same GEMM class as W4A4. |
| `adaln_scale` | `"absmax"` (paper implementation), `"mse"` | Per INT4 group, the BF16 scale with the lowest squared error from `absmax x [0.6, 1.15] / {7, 7.5, 8}`. | None: offline only, same packed format. Not recommended: it lowers the weight error but raised the AdaLN output error on 59 of 60 CogVideoX-2B projections (median 10x) and 17 of 32 Z-Image projections. |
| `output_gain_correction` | `False`, `True` | Divides the BF16 row norms by `E<u,Q(u)>` of the weight and activation codebooks (`0.9905` per side at 4 bits, `0.8826` at 2). This helps outputs whose weight rows and inputs are aligned and hurts uncorrelated ones; compare generated samples before adopting it. | None. |
| `outlier_channels` | `{module name: [channels]}` | Input channels kept out of the quantizer and applied as a BF16 side GEMM over their weight columns. `orbitquant.outliers.find_outlier_channels(model, run)` finds channels above `ratio x` the median token RMS from calibration forwards. | A `k`-column BF16 GEMM per projection; not available on the fused block runtime or in checkpoint-streaming conversion. |

`outlier_channels` uses calibration forwards; the other options use no data.

Measured on Z-Image-Turbo (1024², 10 steps, 10 prompts x 2 seeds, LPIPS against BF16 on an
RTX 4090, 2026-10-29): W4A4 0.434, W4A4 + `rotation_rounds="auto"` 0.344 (better on 18 of 20
pairs), W4A8 absmax 0.375, W4A8 absmax + `"auto"` 0.327, W3A3 0.535, W3A3 + `"auto"` 0.501.
`output_gain_correction`, `adaln_scale="mse"` and `outlier_channels` changed LPIPS by less than
the pair-to-pair spread (95% intervals include zero). `outlier_channels` cut the geometric-mean
output error of the 34 Z-Image FFN `w2` projections (input peak at a median 35000x the token
RMS) by 67%.

## Install

```bash
pip install "orbitquant[hf]"
```

PyTorch's Linux wheels supply the matching Triton build used by the CUDA
fallback. Do not install or upgrade Triton independently: doing so can force a
different PyTorch/CUDA stack. The legacy `kernels` extra remains accepted for
command compatibility, so this is also valid:

```bash
pip install "orbitquant[hf,kernels]"
```

The optimized native kernel package (`orbitquant_packed_matmul`) is provisioned
automatically the first time a packed runtime path needs it. How it works: the
kernel binary must match the runtime exactly — torch minor and CUDA version
for CUDA builds, torch stable ABI (any torch>=2.11) for CPU builds, and the OS
and architecture everywhere — so OrbitQuant derives that variant name from the
running process and resolves it in order: an installed
`orbitquant_packed_matmul` package, a locally built variant (`LOCAL_KERNELS`),
the on-disk cache, and finally the matching prebuilt variant wheel from this
repository's `kernels-v1` GitHub release (checksum-verified against the
release manifest, then cached). Everything happens once per process at model
load, never inside the forward path, and any failure falls back to the Triton
(CUDA) or reference paths with an actionable message. Provision explicitly
(or offline) with:

```bash
orbitquant kernels-install          # download the matching prebuilt variant
orbitquant kernels-install --build  # or compile from bundled sources (needs a toolchain)
orbitquant kernels-status           # inspect what the resolver would do
```

Set `ORBITQUANT_KERNELS_AUTOFETCH=0` to forbid downloads,
`ORBITQUANT_KERNELS_AUTOBUILD=1` to allow automatic source builds, and
`ORBITQUANT_KERNELS_CACHE` to relocate the cache (default
`~/.cache/orbitquant/kernels`). Runtimes without a matching variant fall back
to Triton (CUDA) or the reference paths; see
[`docs/kernel-audit.md`](docs/kernel-audit.md) for the full contract.

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

Pre-quantized Diffusers artifacts default their remaining compute modules to
BF16 when no dtype is supplied. This keeps activations on the packed W4A4 CUDA
path instead of the much slower FP32 compatibility path. An explicit dtype is
always honored, and on-the-fly source conversion keeps the source dtype.

Named recipes are `w4a4`, `w3a3`, `w2a4`, `w2a3`, `w4a6`, and `w4a8` (absmax INT8
activations). They create a normal `OrbitQuantConfig`, so every field can be
overridden.

For an unknown architecture resolved to `target_policy="universal"`, W2 uses
mixed-bit protection by default: out-of-block and the first/last four indexed
blocks use W4, separate interior Q/K projections remain W2, and the other
interior projections use W3. `inspect_linear_module_policy()` reports the
effective `weight_bits` for every module. Set
`lowbit_interior_protection=False` to retain the legacy uniform-W2 interior;
known paper-model policies are unchanged unless protection is explicitly
forced with `lowbit_interior_protection=True`.

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
class. A linear inside an AdaLN norm (`AdaLayerNorm*`, `*NormZero*`,
`*AdaNorm`) uses the INT4 RTN AdaLN path, and a linear inside a timestep, text
or patch embedder stays BF16, whatever the attribute is called. Configs saved
before this rule carry no `target_policy_version` and load as version 1, the
name-token classification they were quantized with.

Qwen2.5-VL is supported by the universal policy, including W4A6 text and vision
attention/FFN projections. Token embeddings, the language output head and the
vision patch embedding remain unquantized. Packed loading is tested with both
text-only and image-token forwards.

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

`Kandinsky6Transformer3DModel` resolves to `target_policy="kandinsky6"`.
It quantizes attention and feed-forward projections in both text branches and
the joint video/audio blocks, including cross-modal attention. Timestep and
AdaLN modulation projections stay FP32, matching Diffusers' precision policy;
input embeddings and final video/audio projections retain their source precision.
This requires a Diffusers version that provides the Kandinsky 6 classes.
See [Kandinsky 6 inference](docs/kandinsky6-runtime.md) for opt-in attention,
pointwise, projection reuse and normalization optimizations.

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

## CUDA Decode

Version 0.9.8 includes native 1.0.5 register-codebook GEMV for SM89 (Ada)
and SM120 GPUs.
For row-major W4A4 inputs with one to eight rows, input width 1024–16384,
and at least 2048 output features, it decodes arbitrary 16-entry signed INT8
codebooks with register byte permutations. Two rows share each decoded weight
vector. The kernel needs no shared lookup table or unpacked weight cache.
Other devices and shapes keep the existing native dispatch.

On RTX PRO 4500 Blackwell with Torch 2.10.0+cu128, interleaved full YuE2
songs measured 22.61 → 22.16 s at CFG 1 and 26.04 → 24.57 s at CFG 1.5
versus native 1.0.3. Each setting used one warmup followed by two timed runs
per kernel, with INT8 KV cache, fused RMS quantization, eight ODE steps, and
FP16 VAE. All ten PCM outputs matched their corresponding control song.
These are warm-generation measurements on one GPU, not a universal gain or
a fresh-process memory comparison.

The Python package and native kernel are separate installations. Managed
caches are partitioned by the minimum native release (currently 1.0.5), so an
upgrade does not silently reuse an older managed binary. Older cache directories
are preserved. Downloaded wheels must satisfy that version floor and match the
release checksum; a current cached variant remains usable offline. Provision the
new variant before taking an upgraded installation offline.

Explicitly installed packages and `LOCAL_KERNELS` retain priority. Upgrade an
explicit installation with the matching wheel from the
[`kernels-v1` release](https://github.com/iamwavecut/OrbitQuant/releases/tag/kernels-v1),
or build the bundled source with kernel-builder. `orbitquant kernels-status`
reports the native version floor and managed cache selection.
`ORBITQUANT_W4A4_DISABLE_GEMV=1` selects the Tensor Core fallback.

## Reproduce Load Memory

Version 0.9.4 fixes a model-lifetime leak in the registry used by `torch.compile`.
Earlier versions registered every `OrbitQuantLinear` and `RTNInt4Linear` in a
strong-reference dictionary, so discarded layers remained alive even without
compilation. The registry now holds weak values and uses non-reusable handles;
live compiled models keep working, while stale handles fail explicitly.
This changes ownership, not quantization or kernel arithmetic.


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
[docs/flux2-klein-9b-sdnq-vs-orbitquant.md](docs/flux2-klein-9b-sdnq-vs-orbitquant.md):
on a single RTX PRO 6000, OrbitQuant 0.5.0's hot median (1.164 s at
1024x1024/4 steps) is 3.2% faster than unquantized BF16 and 8.6% faster than
SDNQ UINT4, with 12.51 GB peak CUDA allocated memory versus BF16's 34.72 GB.

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

## Fused Transformers

Version 0.11 runs the transformer blocks of supported diffusion models fused:
the projections that read one input (attention Q|K|V, SwiGLU gate|up, the two
streams of a joint attention) form one weight matrix and one INT8-surrogate
GEMM, the RMSNorm/LayerNorm and AdaLN modulation in front of them run in the
activation-quantization prologue, and SwiGLU, sigmoid gates and gated residual
updates run in the GEMM epilogue. Q/K normalization and RoPE are one kernel,
and attention uses an INT8 Q.K^T / FP16-accumulated P.V kernel (the
SageAttention v1 scheme); blocks whose Q/K RMSNorm scales one channel far above
the rest keep BF16 attention.

A fused checkpoint stores only the fused groups and records the layout in its
`quantization_config`, so the usual loader returns the fused transformer:

```python
import torch
import orbitquant  # registers the loader
from diffusers import Krea2Pipeline

pipe = Krea2Pipeline.from_pretrained(
    "WaveCut/Krea-2-Turbo-OrbitQuant-W4A4", torch_dtype=torch.bfloat16
).to("cuda")
```

Fuse a per-projection checkpoint on the host and save it:

```python
import torch
from diffusers import Flux2Transformer2DModel
from orbitquant import fused

transformer = Flux2Transformer2DModel.from_pretrained(
    model_dir, subfolder="transformer", torch_dtype=torch.bfloat16
)
fused.fuse(transformer)  # attention="flash" keeps BF16 attention everywhere
fused.save_pretrained(transformer, "fused/transformer")
```

`orbitquant.fused.fuse_component_artifact(model, source_dir, output_dir)` does
the same for an `orbitquant-v1` component artifact, and
`load_orbitquant_artifact` loads the result fused.

| Family | Transformer class | Fused blocks |
| --- | --- | --- |
| Krea 2 | `Krea2Transformer2DModel` | Q\|K\|V\|gate, gated output, SwiGLU; W8A8 down projections; the text fusion runs once per prompt and batch-1 forwards skip the padded prompt rows |
| FLUX.2 | `Flux2Transformer2DModel` | double-stream (per-stream Q\|K\|V, gated residuals) and single-stream blocks; reference-image KV cache |
| Ideogram 4 | `Ideogram4Transformer2DModel` | sandwich-norm blocks with row-indexed modulation |
| Qwen-Image 2.1 | `QwenImage21Transformer2DModel` | block-causal prefill and KV-cached decode |
| MiniMax-H3 | `MiniMaxH3Transformer3DModel` | per-row (timestep, modality) AdaLN tables, partial rotate-half RoPE |
| Boogu-Image | `BooguImageTransformer2DModel` | refiner, single- and double-stream blocks; 8-bit activations |
| FLUX.3 Klein | `flux3.dit.DiT` (Black Forest Labs' `flux3` package) | per-mode and single-stream blocks: Q\|K\|V, SwiGLU, gated residuals; one sample at a time |
| MOSS-Image | `MOSSImageTransformer` (the model's `pipeline_moss_image.py`) | joint image/text blocks: RMSNorm and shared AdaLN prologues, Q\|K\|V, SiTU gate\|up, gated residuals; one sample at a time |
| Fibo 2.0 | `BriaFibo2Transformer2DModel` | Z-Image blocks with per-token (noisy/clean) modulation tables, sandwich post-norms and tanh gates as row kernels; gated cross-attention of the five injection blocks; Perceiver layers; SwiGLU halves padded to whole interleave chunks; 8-bit activations; one sample at a time |

Groups take the activation width of their projections unless the layout
overrides it: `fuse(..., activation_bits={"qkv": 8})` gives a group 8-bit
activations (per-token absmax INT8 of the rotated input). Boogu-Image runs all
its groups this way because its 3360 channels only allow a 32-wide rotation
block; the GEMMs stay INT8, so the wider codes cost almost no speed. Low-bit
artifacts with mixed weight widths (boundary and interior protection) fuse into
row segments of one GEMM.

Components without a block family can still run their OrbitQuant projections
on the fused runtime: `fused.fuse_linears(model)` replaces every loaded
`OrbitQuantLinear` with a `FusedLinear` (one INT8 GEMM over the projection's
packed rows) and quantizes its input per token, by default with 8-bit absmax
activations. Text encoders that run once per prompt use it:

```python
import orbitquant
from orbitquant import fused
from transformers import AutoModel

encoder = AutoModel.from_pretrained(text_encoder_dir, device_map={"": "cuda"})
fused.fuse_linears(encoder)  # activation_bits=4 keeps the Lloyd-Max codes
```

The fused runtime needs CUDA and Triton. Run inference under `torch.no_grad()`
or `torch.inference_mode()`. Fused weights are frozen parameters, so model CPU
offload and Diffusers block-level group offloading (including streamed
`use_stream=True` prefetching) move them with their blocks; leaf-level group
offloading does not, because a fused block reads its groups directly instead of
calling them. Stream without `record_stream`: recorded streams keep every
streamed block reserved until the compute stream has used it, which grew
MiniMax-H3's GPU memory to its 12 GiB cap (5.1 GiB without).
`orbitquant.runtime.krea2` from 0.10 remains for existing deployments.

## Validated Architecture Coverage

The integration suite instantiates and inventories encoder-only, decoder-only,
encoder-decoder, causal LM, and vision transformer families:

| Family | Projection type |
| --- | --- |
| BERT | `torch.nn.Linear` |
| GPT-2 | Hugging Face `Conv1D` with transposed source weights |
| GPTBigCode | `torch.nn.Linear`, including multi-query attention |
| Llama | `torch.nn.Linear`, including GQA projections |
| T5 | encoder and decoder `torch.nn.Linear` projections |
| ViT | vision transformer `torch.nn.Linear` projections |

The paper-aligned release artifacts remain FLUX.1-schnell, Z-Image-Turbo, and
Wan 2.1 T2V. FLUX.2 Klein is an additional validated diffusion target.

Architecture coverage means the model can be discovered, quantized, executed,
saved, and restored through the same public quantization API. It does not guarantee a
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

GPTQ rounding (`orbitquant.gptq`) is not part of the paper's method: it uses calibration data
to choose codes within the same codebook and format. The default quantizer never calls it.
The [quality options](#quality-options-opt-in) are also outside the paper recipe; their
defaults keep it.

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
