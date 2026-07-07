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

## Remaining Kernel Work

The current CUDA path is no longer a CPU fallback path for the quantization
stages listed above. It is still not the final kernel story:

- fused low-bit matmul is not implemented yet,
- activation norm + RPBH + codebook lookup can be fused more aggressively,
- the current Triton kernels are not yet packaged as Hugging Face Kernel Hub
  ABI3/kernel-builder artifacts,
- end-to-end native image/video generation timing still needs to be measured
  separately from this micro-benchmark.
