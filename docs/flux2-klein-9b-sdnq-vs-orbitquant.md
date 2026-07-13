# FLUX.2 Klein 9B: OrbitQuant W4A4 vs SDNQ UINT4

This report compares OrbitQuant and SDNQ on the same FLUX.2 Klein 9B source,
prompts and generation settings. Runtime rows use the same RTX PRO 6000 host
and software environment. It covers checkpoint size, load and generation latency,
VRAM, paired visual output and the measured OrbitQuant kernel path.

## Checkpoints

| Variant | Repository | Measured revision | Quantized components |
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

The revisions above pin the weight snapshots used by the experiment. Later
OrbitQuant Hub commits update only its model card and comparison matrix; the
measured transformer, text-encoder, and VAE payloads are unchanged.

## Protocol

- GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition 96 GB (`sm_120`), one host, one session
- Torch: 2.9.1+cu128
- CUDA: 12.8
- Diffusers: 0.39.0
- Transformers: 5.13.0
- OrbitQuant: 0.5.0 (released package, locally built sm_120 native kernel)
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
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

These are the measured software versions, not current installation minimums.

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

All three variants ran back-to-back on the same card, one separate process
per variant, with the pinned checkpoint revisions above. The OrbitQuant rows
used a locally built ABI3 CUDA package from
`native-kernels/orbitquant-packed-matmul`; Kernel Hub was not involved. All
396 packed projections selected `native_packed_matmul` and the optimized
W4A4 path.

| Variant | Load | Cold image | Hot mean | Hot median | Hot NVML peak | CUDA allocated peak | CUDA reserved peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 (no quantization) | 3.06 s | 1.54 s | 1.2014 s | 1.2019 s | 36.40 GB | 34.72 GB | 35.12 GB |
| SDNQ UINT4 | 3.02 s | 5.81 s | 1.2735 s | 1.2730 s | 15.42 GB | 13.75 GB | 14.14 GB |
| OrbitQuant W4A4 | 1.95 s | 4.96 s | 1.1633 s | 1.1640 s | 14.17 GB | 12.51 GB | 12.89 GB |

OrbitQuant's hot median is 3.2% faster than unquantized BF16 and 8.6% faster
than SDNQ, it loads fastest of the three, and it uses 22.2 GB less peak NVML
memory than BF16 and 1.25 GB less than SDNQ. The checkpoint's
`benchmark/summary.json` carries the machine-readable results.

The selected CUDA path performs native token norm, RPBH/FWHT and codebook-bin
selection, emits an INT8 surrogate of the 4-bit activation codebook, decodes
only a bounded output-channel chunk of the packed W4 weights, calls the
CUTLASS-backed `torch._int_mm`, and applies norms, scales and bias in a Triton
epilogue. It does not materialize a full BF16/FP16 weight matrix. The direct
packed CUDA MMA implementation remains the fallback for unsupported shapes.

The optimized path adds a small runtime approximation beyond the paper
equation: fixed Lloyd-Max centroids are represented by symmetric INT8 codes and
one scalar per codebook. Packed indices and the checkpoint are unchanged;
`runtime_mode="dequant_bf16"` remains the exact-centroid reference. Local build
and verification instructions are in [the kernel audit](kernel-audit.md#local-native-package).

## Paired Visual Comparison

The matrix uses the complete ten-prompt stress pack with full 1024x1024 tiles
and WebP quality 95. BF16 is the full-precision reference from the controlled
visual run; the SDNQ and OrbitQuant columns use the recorded benchmark
outputs. Every row uses the same prompt, seed, resolution, step count and
guidance.

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
- **English typography:** OrbitQuant is strongest on this row: it preserves the headline,
  subtitle and all four specification lines. SDNQ preserves the headline and three table
  lines but omits or corrupts some requested text.
- **Russian typography:** all variants render the large headline, subtitle and archive stamp
  well; small contents text contains errors in every column.
- **Japanese and Chinese typography:** visual glyph quality is plausible, but exact requested
  strings are not reliably reproduced by any variant.
- **Trajectory fidelity:** both quantizers change the denoising trajectory at the same seed;
  neither remains consistently closer to BF16 across all ten prompts.

This assessment is subjective and paired. It demonstrates non-collapse and exposes concrete
failure modes; it is not a substitute for GenEval or another task-specific objective metric.

## Result

OrbitQuant produces the smaller complete 4-bit pipeline, additionally quantizes
activations without calibration data, and outruns both SDNQ and unquantized BF16 on
the tested RTX PRO 6000 while using materially less runtime memory. The visual matrix
shows preserved complex structure without collapse, with OrbitQuant producing
the strongest English fine-print result in this prompt pack. This is a controlled
result for FLUX.2 Klein 9B, not a universal speed claim for every model or GPU.

Machine-readable metrics and the exact ten prompts are included with the
[OrbitQuant checkpoint](https://huggingface.co/WaveCut/FLUX.2-klein-9B-OrbitQuant-W4A4).
