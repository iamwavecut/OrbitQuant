# FLUX.2 Klein 9B: OrbitQuant W4A4 vs SDNQ UINT4

This report compares OrbitQuant and SDNQ on the same FLUX.2 Klein 9B source,
hardware, prompts and generation settings. It covers checkpoint size, load and
generation latency, VRAM, energy, paired visual output and the measured OrbitQuant
kernel path.

## Checkpoints

| Variant | Repository | Revision | Quantized components |
| --- | --- | --- | --- |
| BF16 | [`black-forest-labs/FLUX.2-klein-9B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) | `92196c8e11f7b6cf2b7493e037d8c5345c559216` | None |
| SDNQ UINT4 | [`WaveCut/FLUX.2-klein-9B-SDNQ-uint4-static`](https://huggingface.co/WaveCut/FLUX.2-klein-9B-SDNQ-uint4-static) | `ed71b3f19ce640e88b66a2a743aabb8a613adeac` | Transformer and Qwen3 text encoder |
| OrbitQuant W4A4 | [`WaveCut/FLUX.2-klein-9B-OrbitQuant-W4A4`](https://huggingface.co/WaveCut/FLUX.2-klein-9B-OrbitQuant-W4A4) | `ee3a38f7767ae199818d746c840be0f1837887bf` | Transformer and Qwen3 text encoder |

The public SDNQ checkpoint was selected instead of the AI Farm LoRA/heretic-text-encoder
variant so that quantizer choice is the main experimental difference. SDNQ uses UINT4
weights with BF16 activations. OrbitQuant uses 4-bit codebook weights and 4-bit codebook
activations.

OrbitQuant quantized 144 transformer projections and 252 text-encoder projections. Three
transformer modulation projections use INT4 RTN with BF16 activations, and the text
encoder `lm_head` remains BF16. Quantizing the text encoder is a deliberate universal
adapter extension for this comparison; the OrbitQuant paper leaves text encoders in BF16.

## Protocol

- GPU: NVIDIA A40 48 GB (`sm_86`)
- Torch: 2.9.1+cu128
- CUDA: 12.8
- Diffusers: 0.39.0
- Transformers: 5.13.0
- OrbitQuant: 0.2.2
- SDNQ: 0.1.8
- Arithmetic: BF16
- CPU offload: disabled
- Output: 1024x1024
- Steps: 4
- Guidance: 1.0
- Seed: 0
- Batch size: 1
- Ten identical prompts per variant
- Each variant loaded and ran in a separate process

The prompt pack stresses micro-detail, exact counting, nested spatial composition,
fictional authorial style, abstract material separation, English fine print, Russian,
Japanese and Chinese typography, reflections, occlusion and a dense panoramic scene.

## Artifact Size

The table counts model weight payloads only, excluding cards and comparison images.

| Variant | Transformer | Text encoder | VAE | Total weights |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 18.157 GB | 16.382 GB | 0.168 GB | 34.707 GB |
| SDNQ UINT4 | 5.616 GB | 6.397 GB | 0.168 GB | 12.181 GB |
| OrbitQuant W4A4 | 4.705 GB | 5.966 GB | 0.168 GB | 10.839 GB |

OrbitQuant's weight payload is 68.8% smaller than BF16 and 11.0% smaller than the
controlled SDNQ artifact. The complete loadable OrbitQuant pipeline before its model-card
matrix is 10.85 GB.

## Runtime

The native OrbitQuant row used a locally built ABI3 CUDA package from
`native-kernels/orbitquant-packed-matmul`; Kernel Hub was not involved. All 396 packed
projections selected `native_packed_matmul`, while activation norm, RPBH, codebook lookup
and rescale used Triton CUDA.

| Variant | Load | Cold image | Hot mean | Hot median | Hot p95 | Encode median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 8.877 s | 4.597 s | 3.911 s | 3.912 s | 3.931 s | 0.110 s |
| SDNQ UINT4 | 37.642 s | 25.421 s | 3.546 s | 3.545 s | 3.576 s | 0.151 s |
| OrbitQuant W4A4, native CUDA | 25.229 s | 9.617 s | 7.418 s | 7.415 s | 7.503 s | 0.239 s |
| OrbitQuant W4A4, Triton fallback | 19.325 s | 30.721 s | 19.309 s | 19.307 s | 19.365 s | 0.588 s |

| Variant | NVML peak | CUDA allocated peak | Mean GPU util. | Mean power | Peak power | Energy / 10 images |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 40.83 GB | 37.31 GB | 96.87% | 279.08 W | 307.28 W | 3.035 Wh |
| SDNQ UINT4 | 17.55 GB | 14.86 GB | 97.19% | 274.03 W | 302.01 W | 2.704 Wh |
| OrbitQuant W4A4, native CUDA | 17.66 GB | 13.96 GB | 99.56% | 289.66 W | 306.37 W | 5.979 Wh |
| OrbitQuant W4A4, Triton fallback | 17.94 GB | 14.34 GB | 99.82% | 287.03 W | 302.90 W | 15.406 Wh |

Native OrbitQuant used 56.7% less peak NVML memory than BF16 and approximately the same
peak as SDNQ. It was 1.90x slower than BF16 and 2.09x slower than SDNQ in hot generation
on this A40. The result supports a memory-efficiency claim, not an end-to-end speedup claim.
The native package was materially faster than the Triton packed-matmul fallback.

Representative W4A4 layer timings after CUDA kernel tuning:

| Projection | Rows | Shape | Activation path | Native packed matmul | Full layer |
| --- | ---: | ---: | ---: | ---: | ---: |
| Double-stream Q | 4096 | 4096 -> 4096 | 0.698 ms | 2.607 ms | 3.279 ms |
| Single-stream fused input | 4608 | 4096 -> 36864 | 0.770 ms | 25.962 ms | 26.677 ms |
| Single-stream output | 4608 | 16384 -> 4096 | 4.257 ms | 11.244 ms | 15.514 ms |

The wide fused projections remain the primary throughput limitation. Local native-package
build and verification instructions are in [the kernel audit](kernel-audit.md#local-native-package).

## Paired Visual Comparison

The matrix uses full 1024x1024 tiles and WebP quality 95. Every column uses the same prompt,
seed and pipeline settings.

![BF16, SDNQ UINT4 and OrbitQuant W4A4 across ten difficult prompts](assets/flux2-klein-9b-sdnq-vs-orbitquant.webp)

## Visual Assessment

- **No collapse:** all thirty outputs are finite, coherent and detailed. OrbitQuant did not
  produce blank, noisy or structurally broken images.
- **Micro-detail and materials:** all three variants preserve gears, filigree, architectural
  interiors, paper grain, metal, resin and reflected surfaces. OrbitQuant remains competitive
  with BF16 and SDNQ in these cases.
- **Dense composition:** all variants retain foreground/background separation and the main
  hierarchy in the architectural cutaway and orbital-banquet prompts. Individual requested
  objects move or disappear because quantization changes the denoising trajectory.
- **Counting:** none of the variants reliably renders exactly nine performers or every exact
  repeated motif. This is a base-model limitation in the tested setting rather than an
  OrbitQuant-only collapse.
- **English typography:** SDNQ is strongest on the small four-line specification table.
  OrbitQuant preserves the headline and subtitle but misspells or truncates some fine print.
- **Russian typography:** all variants render the large headline, subtitle and archive stamp
  well; small contents text contains errors in every column.
- **Japanese and Chinese typography:** visual glyph quality is plausible, but exact requested
  strings are not reliably reproduced by any variant.
- **Trajectory fidelity:** SDNQ often stays visually closer to BF16. OrbitQuant also quantizes
  activations and therefore produces larger compositional changes at the same seed.

This assessment is subjective and paired. It demonstrates non-collapse and exposes concrete
failure modes; it is not a substitute for GenEval or another task-specific objective metric.

## Result

OrbitQuant produces the smallest of the two complete 4-bit pipelines and reaches SDNQ's
runtime-memory class while additionally quantizing activations without calibration data. On
the tested A40, SDNQ remains substantially faster and more energy-efficient, and it performs
better on the most demanding small English typography example. OrbitQuant's strongest result
is compact, calibration-free W4A4 inference with preserved complex visual structure; further
packed-GEMM work is required before making a speed claim.

Machine-readable metrics and the exact ten prompts are included with the
[OrbitQuant checkpoint](https://huggingface.co/WaveCut/FLUX.2-klein-9B-OrbitQuant-W4A4).
