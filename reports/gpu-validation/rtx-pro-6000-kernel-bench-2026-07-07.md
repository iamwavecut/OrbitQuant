# RTX PRO 6000 Kernel Bench - 2026-07-07

This is a CUDA kernel validation and micro-benchmark checkpoint. It is not a
release-grade native FLUX/Z-Image/Wan quality report.

## Scope

- Device: NVIDIA RTX PRO 6000 Blackwell Server Edition.
- Runtime: official RunPod PyTorch/CUDA image.
- Repository commit under test: `d60a7bf`.
- Benchmark target: one `OrbitQuantLinear` shaped like a large DiT projection.
- Shape: `tokens=256`, `in_features=3072`, `out_features=3072`.
- Quantization: W4A4, `block_size=1024`, BF16 runtime.
- Backend: `activation_kernel_backend=triton_cuda`.
- Runtime mode: `dequant_bf16`.
- `full_fusion=false`: matmul is still PyTorch BF16 linear.

## Why This Checkpoint Exists

During full-model quantization, CPU activity is expected for model download,
JSON/manifest generation, checksum calculation, safetensors serialization, and
the first Lloyd-Max codebook solve. It is not acceptable for CUDA weight
rotation, FWHT, codebook indexing, low-bit pack/unpack, or runtime dequantization
to silently fall back to CPU when a CUDA quantization device is requested.

Commit `d60a7bf` tightens that boundary:

- `pack_lowbit(cuda)` now requires the Triton CUDA backend and fails loudly if it
  is unavailable.
- `unpack_lowbit(cuda)` now uses a Triton kernel and returns a CUDA tensor.
- Trusted internal weight/codebook index packing skips redundant range
  validation, avoiding a per-matrix GPU-to-CPU scalar sync.
- `kernel-info` now exposes `lowbit_unpack_optimized`.

## Commands

Local verification before remote CUDA:

```bash
uv run pytest -q
uv run ruff check .
```

Remote targeted verification on the RTX PRO 6000 pod:

```bash
.venv/bin/ruff check \
  src/orbitquant/packing/bitpack.py \
  src/orbitquant/kernels/triton_cuda.py \
  src/orbitquant/kernels/dispatch.py \
  src/orbitquant/layers.py \
  src/orbitquant/adaln.py \
  tests/test_bitpack.py \
  tests/test_kernels.py

.venv/bin/pytest tests/test_bitpack.py tests/test_kernels.py tests/test_orbit_linear.py -q -rs
```

CUDA smoke:

```bash
.venv/bin/python - <<'PY'
import torch
from orbitquant import OrbitQuantConfig, prewarm_quantized_linear_modules
from orbitquant.modeling import quantize_linear_modules
from orbitquant.packing import pack_lowbit, unpack_lowbit

values = (torch.arange(4099, device="cuda", dtype=torch.uint8) * 3) % 16
packed = pack_lowbit(values, bits=4, validate=False)
unpacked = unpack_lowbit(packed, bits=4, length=values.numel())
print("lowbit packed", packed.device, packed.dtype, tuple(packed.shape))
print("lowbit unpacked", unpacked.device, unpacked.dtype, torch.equal(unpacked.cpu(), values.cpu()))

class TinyDiT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([
            torch.nn.ModuleDict({
                "attn": torch.nn.Linear(128, 256, device="cuda", dtype=torch.bfloat16)
            })
        ])

model = TinyDiT()
config = OrbitQuantConfig(
    weight_bits=4,
    activation_bits=4,
    target_policy="generic_dit",
    activation_kernel_backend="triton_cuda",
)
summary = quantize_linear_modules(model, config, quantization_device="cuda")
module = model.transformer_blocks[0]["attn"]
prewarm = prewarm_quantized_linear_modules(model, device="cuda", dtype=torch.bfloat16)
print("quantized", summary.quantized_modules)
print("buffers", module.packed_weight_indices.device, module.row_norms.device)
print("cache", module._dequantized_weight_cache.device, module._dequantized_weight_cache.dtype)
print("prewarm total", prewarm.total_modules, prewarm.device, prewarm.dtype)
PY
```

Benchmark:

```bash
.venv/bin/orbitquant kernel-bench \
  --tokens 256 \
  --in-features 3072 \
  --out-features 3072 \
  --weight-bits 4 \
  --activation-bits 4 \
  --activation-kernel-backend triton_cuda \
  --device cuda \
  --dtype bfloat16 \
  --warmup 3 \
  --iterations 10
```

## Verification Result

Targeted CUDA tests passed. Only MPS tests were skipped on the CUDA host.

Explicit smoke output:

```text
lowbit packed cuda:0 torch.uint8 (2050,)
lowbit unpacked cuda:0 torch.uint8 True
quantized ['transformer_blocks.0.attn']
buffers cuda:0 cuda:0
cache cuda:0 torch.bfloat16
prewarm total 1 cuda bfloat16
```

`kernel-info`/benchmark capability output included:

```text
triton_cuda.available: true
triton_cuda.optimized: true
triton_cuda.full_fusion: false
triton_cuda.optimized_stage: activation_norm_rpbh_quant_rescale,packed_weight_dequant,lowbit_pack,lowbit_unpack,weight_rotation_fwht_quant,adaln_rtn_quant_pack,adaln_rtn_dequant
triton_cuda.weight_dequant_optimized: true
triton_cuda.weight_pack_optimized: true
triton_cuda.lowbit_unpack_optimized: true
triton_cuda.weight_quant_optimized: true
triton_cuda.adaln_quant_optimized: true
triton_cuda.adaln_dequant_optimized: true
```

## Benchmark Table

Baseline `4386243` used the first Triton CUDA benchmark before the two-stage
FWHT launch reduction. Current `d60a7bf` includes the two-stage FWHT kernel and
CUDA low-bit unpack.

| Metric | `4386243` ms | `d60a7bf` ms | Speedup |
| --- | ---: | ---: | ---: |
| `torch_linear_ms` | 0.031206 | 0.031616 | 0.99x |
| `activation_quant_ms` | 0.541158 | 0.171034 | 3.16x |
| `weight_dequant_cold_ms` | 0.181562 | 0.076832 | 2.36x |
| `weight_dequant_cached_ms` | 0.002691 | 0.001088 | 2.47x |
| `forward_cold_ms` | 0.866368 | 0.296230 | 2.92x |
| `forward_prewarmed_ms` | 0.629738 | 0.213651 | 2.95x |

Current raw benchmark metadata:

```json
{
  "device": "cuda",
  "device_name": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
  "dtype": "bfloat16",
  "tokens": 256,
  "in_features": 3072,
  "out_features": 3072,
  "weight_bits": 4,
  "activation_bits": 4,
  "block_size": 1024,
  "activation_kernel_backend": "triton_cuda",
  "runtime_mode": "dequant_bf16",
  "full_fusion": false,
  "prewarm": {
    "orbitquant_modules": 1,
    "adaln_modules": 0,
    "total_modules": 1,
    "elapsed_seconds": 0.0022144890390336514,
    "device": "cuda",
    "dtype": "bfloat16"
  },
  "timings_ms": {
    "torch_linear_ms": 0.031615999341011045,
    "activation_quant_ms": 0.17103359699249268,
    "weight_dequant_cold_ms": 0.07683200240135193,
    "weight_dequant_cached_ms": 0.0010879999957978726,
    "forward_cold_ms": 0.296230411529541,
    "forward_prewarmed_ms": 0.2136512041091919
  },
  "peak_memory_bytes": 112740352
}
```

## CUDA Weight Quant Hardening Follow-Up

After the first `weight_rotation_fwht_quant_pack` checkpoint, the weight
quantization path was hardened further because full-model quantization still
showed too much CPU activity for a CUDA run.

The follow-up working tree after `869670d` adds:

- cached `RPBHRotation` objects keyed by `(dim, seed, block_size)`;
- a Triton CUDA row-norm kernel for rank-2 weights;
- direct BF16/FP16/FP32 weight input to the Triton quantize-pack kernel instead
  of materializing a full FP32 copy before the kernel launch;
- cached CUDA constant tensors for weight quantization permutation, signs, and
  codebook boundaries.

Local verification:

```bash
uv run pytest -q
uv run ruff check .
```

Remote RTX PRO 6000 targeted verification:

```bash
PYTHONPATH=/workspace/OrbitQuant-head/src \
  /workspace/OrbitQuant/.venv/bin/python -m pytest \
  tests/test_kernels.py \
  tests/test_orbit_linear.py \
  tests/test_model_quantization.py \
  -q
```

Result: targeted CUDA tests passed. MPS-only tests were skipped on the CUDA host.

Post-hardening micro-benchmark on the same 3072x3072 BF16 shape:

```json
{
  "selected_activation_kernel_backend": "triton_cuda",
  "weight_quantization_backend": "triton_cuda",
  "timings_ms": {
    "weight_quantize_pack_cold_ms": 1155.5595703125,
    "weight_quantize_pack_hot_ms": 0.23645439147949218,
    "torch_linear_ms": 0.03475199937820435,
    "activation_quant_ms": 0.15082240104675293,
    "weight_dequant_cold_ms": 0.07378559708595275,
    "weight_dequant_cached_ms": 0.0017791999503970145,
    "forward_cold_ms": 0.2544640064239502,
    "forward_prewarmed_ms": 0.16335999965667725
  },
  "quantization_buffers": {
    "source_weight_device": "cuda:0",
    "source_weight_is_cuda": true,
    "packed_weight_indices_device": "cuda:0",
    "row_norms_device": "cuda:0",
    "packed_weight_indices_is_cuda": true,
    "row_norms_is_cuda": true
  },
  "peak_memory_bytes": 111879680,
  "full_fusion": false
}
```

The larger cold time is first-use Triton JIT compilation, including the new
row-norm kernel. The hot quantize-pack path is the relevant steady-state CUDA
measurement and is now `0.236 ms` for this shape.

To verify that a sustained hot loop is visible to NVIDIA telemetry, the same
pod ran repeated hot `OrbitQuantLinear.from_linear(...)` calls while sampling
`nvidia-smi dmon`:

```text
# gpu    pwr  gtemp  mtemp     sm    mem    enc    dec    jpg    ofa   mclk   pclk     fb   bar1   ccpm
# Idx      W      C      C      %      %      %      %      %      %    MHz    MHz     MB     MB     MB
    0    218     43      -     95      0      0      0      0      0  12481   2325    725    725      0
    0    411     45      -     95      0      0      0      0      0  12481   2370    725    725      0
    0    365     44      -     95      0      0      0      0      0  12481   2415    725    725      0
    0    362     45      -     95      0      0      0      0      0  12481   2422    725    725      0
    0    365     46      -     96      0      0      0      0      0  12481   2422    725    725      0
    0    366     47      -     96      0      0      0      0      0  12481   2422    725    725      0
HOT_LOOP_COUNT=39542
```

This explains why short single-layer quantization may not light up a web UI GPU
badge: a single hot kernel sequence is sub-millisecond and easy for coarse UI
sampling to miss. Sustained hot quantization is visible as 95-96% SM utilization.

## Full-Model Quantization Staging Follow-Up

Commit `a9ef83d` addresses the remaining CPU-heavy full-model quantization
symptom. The previous full replacement loop synchronized CUDA after each module
replacement. That made CUDA quantization correct, but it also forced Python/CPU
to wait between many short kernel sequences and could make provider-side 1s GPU
sampling look idle.

The production path now:

- avoids per-module CUDA synchronization by default;
- synchronizes once at the end of `quantize_linear_modules`;
- records `synchronize_per_module=false` in CLI output and artifact
  `benchmark/summary.json`;
- keeps `--synchronize-per-module` as an explicit debug timing mode;
- adds `--staging-mode component` for large-GPU native quantization scripts;
- adds `orbitquant quantize-bench` to measure source-device staging separately
  from OrbitQuant/AdaLN compute.

Local verification before the pod run:

```bash
uv run pytest -q
uv run ruff check .
git diff --check
```

Remote RTX PRO 6000 targeted verification:

```bash
PYTHONPATH=/workspace/OrbitQuant-git/src \
  /workspace/OrbitQuant/.venv/bin/python -m pytest \
  tests/test_model_quantization.py \
  tests/test_cli.py::test_cli_quantize_bench_prints_full_model_staging_timings \
  -q
```

Result: `12 passed`.

Synthetic DiT-like full replacement loop, source on CPU and quantization on
CUDA, `layers=2`, `in_features=3072`, `hidden_features=9216`, W4A4:

```text
streaming {
  wall_seconds: 0.4387,
  synchronize_per_module: false,
  transfer_seconds: 0.0576,
  transfer_count: 12,
  orbitquant_seconds: 0.3706,
  adaln_seconds: 0.0094,
  source_devices: {'cpu': 13},
  buffer_devices: {'cuda:0': 36},
  quantized_modules: 10,
  adaln_modules: 2,
  peak_memory_gb: 0.265
}
component {
  wall_seconds: 0.4584,
  synchronize_per_module: false,
  transfer_seconds: 0.0709,
  transfer_count: 1,
  orbitquant_seconds: 0.3759,
  adaln_seconds: 0.0095,
  source_devices: {'cpu': 13},
  buffer_devices: {'cuda:0': 36},
  quantized_modules: 10,
  adaln_modules: 2,
  peak_memory_gb: 0.519
}
```

For comparison, the same small synthetic case before removing per-module sync
was `2.1643s` in streaming mode with 12 transfer operations. The new default
cuts that CPU-wait-heavy path to `0.4387s`.

A heavier CUDA-resident synthetic case, `layers=8`, `in_features=4096`,
`hidden_features=12288`, W4A4, shows why a single full quantize may still be too
short for coarse UI sampling:

```text
{
  wall_seconds: 0.426,
  device: 'NVIDIA RTX PRO 6000 Blackwell Server Edition',
  sync_per_module: false,
  transfer_seconds: 0.001,
  transfer_count: 1,
  orbitquant_seconds: 0.380,
  adaln_seconds: 0.011,
  source_devices: {'cuda:0': 49},
  buffer_devices: {'cuda:0': 144},
  quantized_modules: 40,
  adaln_modules: 8,
  peak_memory_gb: 3.138
}
```

The same heavy case repeated for 15 seconds is visible to `nvidia-smi dmon` as
real GPU work:

```text
loop_count=262 elapsed=15.030 last_wall_seconds=0.0513
# gpu    pwr  gtemp  mtemp     sm    mem    enc    dec    jpg    ofa   mclk   pclk     fb   bar1   ccpm
# Idx      W      C      C      %      %      %      %      %      %    MHz    MHz     MB     MB     MB
    0    158     40      -     97     87      0      0      0      0  12481   2310   3783   3783      0
    0    406     42      -     97     87      0      0      0      0  12481   2310   3783   3783      0
    0    408     44      -     97     87      0      0      0      0  12481   2310   3783   3783      0
    0    389     44      -     97     87      0      0      0      0  12481   2407   3783   3783      0
    0    382     45      -     95     86      0      0      0      0  12481   2422   3783   3783      0
    0    338     45      -     97     87      0      0      0      0  12481   2370   3783   3783      0
    0    376     46      -     95     87      0      0      0      0  12481   2422   3783   3783      0
    0    373     47      -     95     87      0      0      0      0  12481   2422   3783   3783      0
    0    374     49      -     95     87      0      0      0      0  12481   2422   3783   3783      0
    0    376     50      -     95     86      0      0      0      0  12481   2422   3783   3783      0
```

Conclusion for the GPU-badge concern: a single optimized quantization pass can
now complete too quickly for provider UI sampling, but sustained quantization
lights the RTX PRO 6000 at 95-97% SM and 86-88% memory utilization. The
remaining full-model CPU work is mostly orchestration, model loading, artifact
serialization, checksums, and future fused-kernel packaging work, not a silent
CPU fallback for the implemented OrbitQuant CUDA stages.

## Native Generation Artifact CPU Follow-Up

Commits `c7efe7d` and `93b6b82` address the artifact side of the CPU-load
concern for native generation packs. The previous
`generate-pack --skip-artifact-checksums` path skipped strict checksum
validation, but still refreshed manifest/SHA256/README metadata after every
generated image/video sample. It also ran the comparison-sheet builder after
every sample, repeatedly scanning accumulated metrics and assets. On
GenEval-scale runs this could make the host CPU and disk look busy even when
CUDA generation was active.

The production path now:

- records generated assets and metric rows without per-sample manifest/SHA
  refresh when `--skip-artifact-checksums` is set;
- performs one final `refresh_artifact_checksums()` pass at the end of
  `generate-pack`;
- returns a `checksum_refresh` summary in CLI JSON output;
- defers comparison-sheet creation until after the pack loop;
- skips comparison sheets by default for GenEval metadata packs;
- keeps strict artifact validation passing after the final refresh;
- stages OrbitQuant rotation/codebook constants on the same real device as the
  quantized buffers, while preserving CPU constants for Diffusers meta-tensor
  skeleton load.

Local verification:

```bash
uv run pytest -q
uv run ruff check .
git diff --check
```

Targeted tests added for this checkpoint:

```bash
uv run pytest \
  tests/test_artifact_writer.py::test_deferred_artifact_refresh_rebuilds_manifest_and_sha256sums_once \
  tests/test_cli.py::test_cli_generate_pack_skip_checksums_refreshes_artifact_once_at_end \
  tests/test_cli.py::test_cli_generate_pack_defers_comparison_creation_until_after_jobs \
  tests/test_cli.py::test_cli_generate_pack_prompt_metadata_disables_comparisons_by_default \
  tests/test_diffusers_modelmixin_integration.py::test_diffusers_modelmixin_save_pretrained_round_trips_pre_quantized_model \
  -q
```

During the already-running FLUX.1-schnell W4A4 native GenEval generation on the
RTX PRO 6000 pod, the quantized split showed real GPU load:

```text
split: orbitquant
original.metrics.jsonl: 554 lines
orbitquant.metrics.jsonl: 5 lines at sample time
GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
VRAM: 43807 / 97887 MiB
GPU utilization: 99 %
power: 508.12 W
```

That run was intentionally not interrupted; it is still using the previous
checkout. The checksum refresh improvement applies to subsequent native
generation runs after the pod checkout is updated to `c7efe7d` or newer.

The running job was then resumed again at `93b6b82` to pick up deferred
comparison generation. After the second resume, the same FLUX.1-schnell W4A4
GenEval artifact progressed from 90 to 157 quantized metric rows in the first
2:56, including pipeline reload and resume checks, with the RTX PRO 6000 showing
99% GPU utilization and 515 W at the sample point:

```text
git: 93b6b82
split: orbitquant
orbitquant.metrics.jsonl: 90 -> 157 lines
GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
VRAM: 44079 / 97887 MiB
GPU utilization: 99 %
power: 515.67 W
```

This confirms that the previous long idle phases were not an acceptable
quantization behavior; they were artifact bookkeeping in the batch generation
loop.

## Remaining Kernel Work

The current CUDA path is no longer a CPU fallback path for the quantization
stages listed above. It is still not the final kernel story:

- fused low-bit matmul is not implemented yet,
- activation norm + RPBH + codebook lookup can be fused more aggressively,
- the current Triton kernels are not yet packaged as Hugging Face Kernel Hub
  ABI3/kernel-builder artifacts,
- end-to-end native image/video generation timing still needs to be measured
  separately from this micro-benchmark.
