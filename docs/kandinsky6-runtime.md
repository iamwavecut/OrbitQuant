# Kandinsky 6 inference

Requires OrbitQuant 0.12.0 or later. The ready-to-run
[generation example](../examples/kandinsky6_generate.py) loads the published
two-component artifact and supports `fast`, `low-memory` and `exact` profiles.
See the [model card](https://huggingface.co/WaveCut/Kandinsky-6.0-Lite-distill-5s-OrbitQuant-W4A4)
for the validated dependency set and video examples.

```bash
python examples/kandinsky6_generate.py --profile fast --prompt "A red fox in a snowy forest."
python examples/kandinsky6_generate.py --profile low-memory --prompt "A red fox in a snowy forest."
python examples/kandinsky6_generate.py --profile low-memory --width 576 --height 320 --allocator-cap-gib 7 --prompt "An acoustic guitarist."
```

Use `--warmup` to discard one complete generation before timing the saved video.
The printed memory value is Torch allocation, not total device memory.

The Kandinsky 6 runtime helpers work on a loaded quantized transformer. They do
not change checkpoint weights or the checkpoint format. Install them before
`enable_model_cpu_offload()` and run the pipeline under `torch.inference_mode()`.

```python
import torch
from orbitquant.runtime.kandinsky6 import (
    install_kandinsky6_attention,
    install_kandinsky6_pointwise,
    install_kandinsky6_projection_reuse,
    install_kandinsky6_norm_fusion,
    install_kandinsky6_qk_norm,
)
from orbitquant.runtime.kandinsky6_video import (
    install_kandinsky6_video_io,
    install_kandinsky6_video_norm,
)

# `pipe` is a loaded Kandinsky6TI2VAPipeline with OrbitQuant projections.
attention = install_kandinsky6_attention(pipe.transformer)
pointwise = install_kandinsky6_pointwise(pipe.transformer)
projections = install_kandinsky6_projection_reuse(pipe.transformer)
normalization = install_kandinsky6_norm_fusion(pipe.transformer)
qk_normalization = install_kandinsky6_qk_norm(pipe.transformer)
video_io = install_kandinsky6_video_io(pipe)
video_norm = install_kandinsky6_video_norm(pipe)

pipe.vae.enable_tiling(
    tile_sample_min_height=272,
    tile_sample_min_width=320,
    tile_sample_stride_height=240,
    tile_sample_stride_width=288,
)
pipe.enable_model_cpu_offload()
with torch.inference_mode():
    output = pipe(
        prompt="A red fox walks through a snowy pine forest.",
        width=864,
        height=480,
        num_frames=121,
        num_inference_steps=10,
        guidance_scale=1.0,
        generator=torch.Generator(device="cuda").manual_seed(61000),
    )
```

The helpers have different numerical contracts:

| Helper | Behavior |
| --- | --- |
| Pointwise | Fuses FP32 modulation, residual addition and rotary arithmetic, preserving eager rounding. |
| Projection reuse | Prepares a shared input once for consecutive compatible Q/K/V or K/V projections, including inference tensors. |
| Attention | Uses INT8 Q/K and FP16 P/V attention for eligible long video self-attention. This changes numerical results. |
| Normalization | Fuses LayerNorm and modulation. Reduction rounding can differ from PyTorch and can change generated output. |
| Q/K normalization | Processes eight RMSNorm rows per CUDA program on Torch 2.12. Other Torch versions use their native RMSNorm. |
| Video I/O | Accelerates replicated VAE padding and packs RGB bytes before transfer to the CPU. Preserves decoded values and PIL pixels. |
| Video normalization | Fuses large BF16 GroupNorm and SiLU calls. Statistic reduction can differ at rounding boundaries. |

To retain the original attention and normalization arithmetic, install only
pointwise, projection reuse and video I/O. Each helper is opt-in. Validate generated video
and audio for the prompts and checkpoint used by the application. Keep the default
VAE tile settings when testing exact equivalence against the original pipeline.

Attention requires a CUDA GPU with compute capability 8.0 or later, batch size
one, head width 64, and at least 4,096 tokens. Masked attention, cross-attention,
gradient-enabled calls and custom attention backends retain their original
implementation. Blocks whose Q/K normalization scales exceed the runtime guard
also retain their original attention. Inspect `attention.installed` and
`attention.skipped` to see the selection.

Projection reuse is bounded by one helper call. Different quantization signatures
prepare separate inputs. Custom projection forwards and hooks disable reuse.
CPU and gradient-enabled pointwise and normalization calls use eager arithmetic.

## Resolution and video decoding

The tested Diffusers pipeline accepts positive widths and heights divisible by
16, including 576×320. Use a frame count of `4n + 1`, such as 121. Lower resolution
reduces video tokens and changes generation quality. It is a separate inference
configuration, not an optimization measured at the same output resolution.

VAE tile dimensions and overlap affect decoder work, memory and boundary
differences. Configure them with `pipe.vae.enable_tiling(...)`. Keep total
device memory, including CUDA context and allocator reservation, separate from
`torch.cuda.max_memory_allocated()`. A smaller PyTorch allocation alone does not
establish compatibility with an 8 GiB card.

The example uses 32-pixel spatial overlap. This reduces redundant convolution
work but changes decoded pixels relative to the default overlap. Temporal tiling
keeps its original settings.

For a smaller memory budget, release completed stages before VAE decoding.
Install this wrapper before CPU offload. It uses Diffusers' public hook cleanup
method to offload completed components and release unused allocator blocks:

```python
original_decode = pipe.vae.decode

def decode_after_reclaim(*args, **kwargs):
    pipe.maybe_free_model_hooks()
    return original_decode(*args, **kwargs)

pipe.vae.decode = decode_after_reclaim
```

For a capped measurement, set the allocator limit before loading the pipeline:

```python
device = torch.cuda.current_device()
total_bytes = torch.cuda.get_device_properties(device).total_memory
torch.cuda.set_per_process_memory_fraction(7.25 * 1024**3 / total_bytes, device)
```

This cap controls PyTorch's allocator. CUDA runtime allocations remain outside
the cap. Measure total device memory separately. A tighter cap can select slower
convolution algorithms or raise an out-of-memory error. The measured 320p profile
uses the same tile geometry with a 7.0 GiB allocator cap.

## Restoration and compatibility

Remove CPU offload hooks before restoration. Restore helpers in reverse order:

```python
pipe.remove_all_hooks()
video_norm.restore()
video_io.restore()
qk_normalization.restore()
normalization.restore()
projections.restore()
pointwise.restore()
attention.restore()
```

If the decode wrapper was installed, also restore `pipe.vae.decode = original_decode`.

Installation is local to the model and does not replace Diffusers module globals.
Restoration preserves subsequent user replacements. Install after loading weights
so that the attention guard reads the loaded normalization scales.

The block implementations were validated against Diffusers commit
`d961a388fd02e4db38d17350c8dd9b8abe642e05`. They require a Diffusers version that
provides Kandinsky 6 classes. Check compatibility before upgrading Diffusers,
especially when enabling normalization fusion, which supplies block forwards.
The CUDA tests use Torch 2.12.1+cu130 and an RTX 3090. The Q/K normalization
helper preserves that Torch version's FP32 epsilon default for BF16 RMSNorm.
