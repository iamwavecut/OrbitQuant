# OrbitQuant Paper Methodology Audit

Paper revision: arXiv 2607.02461v1.

This audit defines the method-conformance and claim boundaries for OrbitQuant.
Run `scripts/run_paper_methodology_checks.sh` to verify the implementation,
model policies, artifact basis metadata, and native settings. Native artifact
proof and release-grade GenEval/VBench metrics are separate release gates.

Paper source:

- arXiv abstract: https://arxiv.org/abs/2607.02461
- arXiv HTML, version 1: https://arxiv.org/html/2607.02461v1

This document compares the paper methodology against the implementation in this
repository. Claims about paper reproduction, metrics, and kernel acceleration
are limited to the evidence recorded here.

## Scope

Audited paper areas:

- Section 3.2: fixed Lloyd-Max codebook over the post-rotation coordinate
  marginal.
- Section 4.1: shared rotated normalized basis.
- Section 4.2: offline weight rotation, row norm storage, and direction
  quantization.
- Section 4.3: online activation normalization, rotation, quantization, and
  rescale.
- Section 4.4: randomized permuted block-Hadamard rotation.
- Section 4.5: data-agnostic codebook construction.
- Appendix B.1: native generation settings.
- Appendix B.2: quantized and skipped layer policy.
- Appendix C and Section 5.5 only where they affect release metric and runtime
  claims.

Status legend:

- `Pass`: current code and tests provide direct evidence.
- `Partial`: implementation exists, but release evidence is incomplete or the
  claim must be narrower.
- `Blocked for claim`: do not make the specific release claim until the listed
  evidence exists.

## Requirement Matrix

| Paper requirement | Current status | Evidence | Notes |
| --- | --- | --- | --- |
| No calibration data, prompt statistics, timestep ranges, or generated-image statistics are used to construct the quantizer. | Pass | `src/orbitquant/codebooks/lloyd_max.py`, `src/orbitquant/rotations/rpbh.py`, `src/orbitquant/functional.py`, `src/orbitquant/layers.py`, `tests/test_kernels.py`, `tests/test_orbit_linear.py`, `tests/test_artifact_writer.py` | Codebooks depend on `(dim, bits, algorithm_version)`. Rotations depend on `(dim, seed, block_size)`. Activations use only runtime token norms, and serialized layer/artifact state excludes activation calibration/range tensors. |
| One converged Lloyd-Max scalar codebook is built offline per input dimension and bit width. | Pass for codebook version 2 | `get_codebook(dim, bits, algorithm_version)` in `src/orbitquant/codebooks/lloyd_max.py`; `tests/test_codebooks.py`; `tests/test_orbit_linear.py`; `tests/test_artifact_writer.py` | Version 2 evaluates the exact beta marginal and iterates to the Lloyd-Max centroid condition. Version 1 remains read-only compatibility for existing packed indices. Persistent cache keys include algorithm version, dimension, bits, checksum, and structural validation. |
| Lloyd-Max target distribution is the coordinate marginal of a random unit vector in dimension `d`. | Pass | `_coordinate_density()` in `src/orbitquant/codebooks/lloyd_max.py` | The implementation uses the paper density up to normalization: `(1 - z^2)^((d - 3) / 2)` on `[-1, 1]`, then normalizes numerically. |
| Quantization uses nearest fixed centroids, with no zero-point, learned scale, per-channel range, or timestep/prompt range. | Pass | `LloydMaxCodebook.quantize_indices()` and `quantize()` in `src/orbitquant/codebooks/lloyd_max.py` | `torch.bucketize` against midpoint boundaries is equivalent to nearest-centroid lookup for sorted Lloyd-Max centroids. |
| RPBH uses uniform random permutation, Rademacher signs, per-block Walsh-Hadamard transform, and `1 / sqrt(block_size)` normalization. | Pass | `src/orbitquant/rotations/rpbh.py`, `src/orbitquant/rotations/fwht.py`, `tests/test_rpbh.py` | The implementation applies permutation first, then signs, then block FWHT and normalization. |
| RPBH stores compact permutation/sign metadata, not dense rotation matrices. | Pass | `src/orbitquant/rotations/rpbh.py`, `src/orbitquant/artifacts/writer.py` | Artifact rotation tensors are permutation, inverse permutation, signs, and normalization metadata. |
| Compact artifact sidecar files identify the exact runtime basis. | Pass | `src/orbitquant/artifacts/validator.py`, `tests/test_artifact_writer.py` | Validation checks shapes and semantics, then compares centroids and boundaries with the declared codebook version and compares permutation/sign tensors with the runtime RPBH draw. Checksums alone are not treated as sufficient evidence. |
| Default paper block-size policy is the largest power of two dividing the input dimension. | Pass | `RPBHRotation.__post_init__()` in `src/orbitquant/rotations/rpbh.py`; `tests/test_rpbh.py` | Degenerate dimensions warn and fall back to signs/permutation only. |
| Weight rotation is folded offline so activations and weights share the same basis. | Pass | `OrbitQuantLinear.from_linear()` in `src/orbitquant/layers.py`; `tests/test_rpbh.py`; `tests/test_orbit_linear.py` | For PyTorch `linear(x, W, b)`, the code stores `W @ R` and computes `(x @ R) @ (W @ R).T + b`. |
| No inverse rotation is used in runtime quantized forward. | Pass | `OrbitQuantLinear.forward()` in `src/orbitquant/layers.py` | Inverse rotation appears only in `_dequantize()` conversion back to ordinary linear modules. |
| Weight rows are split into row norm plus unit direction; row norm is BF16. | Pass | `OrbitQuantLinear.from_linear()` in `src/orbitquant/layers.py`, `tests/test_orbit_linear.py` | Raw row norms are stored as BF16 buffers. The epsilon guard is used only for division, so zero rows dequantize back to zero rows instead of receiving an epsilon-scaled codebook value. |
| Weight direction coordinates are quantized with the Lloyd-Max codebook and packed into low-bit indices. | Pass | `src/orbitquant/layers.py`, `src/orbitquant/packing/bitpack.py`, `tests/test_bitpack.py`, `tests/test_kernels.py` | Bit packing covers 2, 3, 4, and 6 bit paths. |
| Runtime activations compute per-token norm `s`, normalize by `s + ε`, apply RPBH, nearest-centroid quantize, and rescale by the raw token norm `s`, with `ε = 1e-10` by default. | Pass | `src/orbitquant/config.py`, `src/orbitquant/functional.py`, `src/orbitquant/kernels/dispatch.py`, `src/orbitquant/kernels/triton_cuda.py`, `src/orbitquant/kernels/mps.py`, `tests/test_config.py`, `tests/test_orbit_linear.py`, `tests/test_kernels.py` | This follows Algorithm 1 exactly. Arbitrary leading dimensions are preserved, zero tokens remain zero after rescaling, and CPU, Triton/CUDA, and Metal/MPS paths use the same denominator. Manifests record the actual epsilon. |
| The only input-dependent runtime scalar is the per-token norm. | Pass | `src/orbitquant/functional.py`, `tests/test_kernels.py`, `tests/test_orbit_linear.py` | Codebook, rotation, centroids, boundaries, signs, and permutation are fixed after construction; persistent layer state is limited to packed weight indices, row norms, and optional bias. |
| AdaLN modulation projections use INT4 weight-only RTN with group size 64 and BF16 activations. | Pass | `src/orbitquant/adaln.py`, `src/orbitquant/config.py`, `src/orbitquant/artifacts/manifest.py`, `tests/test_adaln_rtn.py` | AdaLN wrappers do not call OrbitQuant activation rotation. Default `adaln_group_size` is 64, and artifacts record the actual group size so non-default artifacts are labeled. |
| Transformer-block linear projections are quantized through OrbitQuant. | Pass | `src/orbitquant/policies/generic_dit.py`, `src/orbitquant/linear_adapters.py`, `tests/test_target_policies.py`, `tests/test_universal_transformers.py` | Paper targets retain exact model policies. Unknown architectures use the universal policy over every registered linear-compatible module, subject to explicit boundary skips and user allowlists. |
| Embeddings, timestep MLPs, final projection/unpatchify heads, text encoders, VAE, scheduler, safety/image processors remain unquantized by default. | Pass for configured transformer components | `src/orbitquant/policies/generic_dit.py`, `tests/test_target_policies.py`; inventory summary below | Text encoders and VAE are outside the transformer component and are not passed into the default quantization helper. Artifact manifests still need per-artifact cross-checks before final publication. |
| Native settings match paper for FLUX.1-schnell, Z-Image-Turbo, and Wan 2.1-1.3B. | Pass for encoded settings | `src/orbitquant/eval/native_settings.py`, `README.md`, `src/orbitquant/artifacts/model_card.py` | Native artifact-readiness evidence is separate from release-grade metric tables. Full metric runs are required before metric-table or paper-reproduction claims. |
| FLUX.2 Klein is separated from paper-reproduction targets. | Pass | `src/orbitquant/eval/native_settings.py`, `src/orbitquant/artifacts/model_card.py` | It is treated as an additional target using paper-style native settings. |
| Runtime acceleration claims match implemented kernels. | Pass for the measured FLUX.2 Klein 9B W4A4 configuration; partial across other models | `src/orbitquant/kernels/dispatch.py`, `src/orbitquant/kernels/triton_cuda.py`, `src/orbitquant/kernels/mps.py`, `tests/test_kernels.py`, `tests/test_orbit_linear.py`, `docs/kernel-audit.md` | Default `auto_fused` requires packed low-bit matmul on CUDA/MPS and fails loudly when kernels are missing. CUDA and MPS avoid full floating-point weight materialization. The CUDA INT8-surrogate fast path is an explicitly documented runtime approximation; exact Lloyd-Max centroid evaluation remains available through `dequant_bf16`. |
| Release-grade GenEval/VBench metrics are available for paper target claims. | Blocked for metric claims | `src/orbitquant/hub.py`, `src/orbitquant/eval/` | Missing metrics block only paper metric/reproduction claims. |

## Model Policy Evidence

Paper-target policy coverage remains pattern-based and scoped to Diffusers
transformer components. The separate `universal` policy is structural: it
quantizes every registered linear-compatible module except embeddings,
timestep modules, task/output heads, and explicit skips. Built-in adapters cover
`torch.nn.Linear` and Hugging Face `Conv1D`; custom modules must register their
weight layout and feature attributes. Inventory summaries are derived from config-based
`orbitquant inspect-policy --suite ... --load-mode config --dtype bfloat16`
outputs. Raw inventory JSON is audit evidence, not package or model artifact
content.

| Model family | Quantized patterns | AdaLN/INT4 patterns | Default skips | Evidence |
| --- | --- | --- | --- | --- |
| FLUX.1 | Transformer block attention `to_q`, `to_k`, `to_v`, `to_out`, joint text projections `add_*`, `to_add_out`, FFN modules, single-block projections. | `norm1.linear`, `norm1_context.linear`, single-block `norm.linear`. | `time_text_embed`, embedders, `norm_out`, final `proj_out`, text encoders, VAE. | `src/orbitquant/policies/generic_dit.py`; `tests/test_target_policies.py` instantiates `FluxTransformer2DModel`. |
| FLUX.2 Klein | Double-stream attention, text-conditioning projections, FFN `linear_in` and `linear_out`, fused single-stream `to_qkv_mlp_proj`, `to_out`. | `double_stream_modulation_img`, `double_stream_modulation_txt`, `single_stream_modulation`. | `time_guidance_embed`, embedders, `norm_out`, final `proj_out`, text encoders, VAE. | `src/orbitquant/policies/generic_dit.py`; `tests/test_target_policies.py` instantiates `Flux2Transformer2DModel`. |
| Z-Image-Turbo | `noise_refiner`, `context_refiner`, and `layers` attention projections and FFN `w1`, `w2`, `w3`. | Refiner and main-layer `adaLN_modulation`. | `all_x_embedder`, `t_embedder`, `cap_embedder`, final layer and final AdaLN modulation. | `src/orbitquant/policies/generic_dit.py`; `tests/test_target_policies.py` instantiates `ZImageTransformer2DModel`. |
| Wan 2.1 | `blocks.*.attn1`, `blocks.*.attn2`, and `blocks.*.ffn` projections. | None expected for Wan 2.1-1.3B. | `condition_embedder`, time/text embedders, final `proj_out`, text encoder, VAE. | `src/orbitquant/policies/generic_dit.py`; `tests/test_target_policies.py` instantiates `WanTransformer3DModel`. |

The universal integration suite additionally covers BERT, GPT-2 `Conv1D`,
Llama, T5 encoder-decoder, and ViT module inventories. It verifies quantize,
forward, `save_pretrained()`, and packed `from_pretrained()` restoration. This
is architecture compatibility evidence, not a quality claim for a particular
bit setting.

Inventory summary:

| Suite | Component class | Linear modules | OrbitQuant | AdaLN INT4 | BF16 skip |
| --- | --- | ---: | ---: | ---: | ---: |
| `flux2-native` | `Flux2Transformer2DModel` | 109 | 100 | 3 | 6 |
| `flux1-schnell-native` | `FluxTransformer2DModel` | 502 | 418 | 76 | 8 |
| `z-image-native` | `ZImageTransformer2DModel` | 276 | 238 | 32 | 6 |
| `wan-native` | `WanTransformer3DModel` | 306 | 300 | 0 | 6 |

Artifact validation compares manifest `quantized_modules`, `adaln_modules`, and
`skipped_modules` against the policy inventory. The paper gate pins exact
module-list hashes in addition to aggregate counts, so a count-preserving swap
from a paper projection to an unrelated module fails the gate.

## Native Setting Provenance

Paper-aligned generation settings are encoded in
`src/orbitquant/eval/native_settings.py`. These settings define native
artifact-readiness runs and the input generation scripts for later external
metrics; they do not by themselves claim GenEval or VBench scores.

| Target | Paper source | Encoded suite | Encoded setting | Claim boundary |
| --- | --- | --- | --- | --- |
| FLUX.1-schnell | Appendix B.1 generation settings; Section 5.1 and supplementary low-bit settings | `flux1-schnell-native` | 1024x1024, 4 steps, guidance 0.0, W4A4/W3A3/W2A4/W2A3 | Paper target; GenEval metrics required before reproduction-score claims. |
| Z-Image-Turbo | Appendix B.1 generation settings; Section 5.1 and supplementary low-bit settings | `z-image-native` | 1024x1024, 10 steps, guidance 0.0, W4A4/W3A3/W2A4/W2A3 | Paper target; GenEval metrics required before reproduction-score claims. |
| Wan 2.1-1.3B | Appendix B.1 generation settings; Section 5.1 video bit settings | `wan-native` | 832x480, 81 frames, 50 steps, guidance 5.0, W4A6/W4A4 | Paper target; VBench metrics required before reproduction-score claims. |
| FLUX.2 Klein | Additional non-paper target | `flux2-native` | 1024x1024, 4 steps, guidance 1.0, W4A4/W3A3/W2A4/W2A3 | Extra target using the same native-validation discipline; not a paper reproduction target. |

## Kernel And Runtime Evidence

| Backend | Status | Evidence | Claim boundary |
| --- | --- | --- | --- |
| CPU | Pass as reference | `src/orbitquant/kernels/dispatch.py`, `src/orbitquant/functional.py` | Correctness baseline only; no optimized CPU kernel claim. |
| CUDA/native/Triton | Pass for optimized W4A4 and packed fallback | `native-kernels/orbitquant-packed-matmul`, `src/orbitquant/kernels/native_packed_matmul.py`, `src/orbitquant/kernels/triton_cuda.py`, `tests/test_native_packed_matmul.py`, `tests/test_kernels.py`, `tests/test_orbit_linear.py`, `docs/kernel-audit.md` | The selected W4A4 path fuses norm/RPBH/FWHT/codebook assignment to INT8 surrogate activations, decodes bounded W4 chunks, uses CUTLASS INT8 matmul, and applies a fused epilogue. Direct packed CUDA MMA and generic Triton packed matmul remain fallbacks. No path selected by `auto_fused` materializes a full BF16/FP16 weight matrix. |
| MPS/Metal | Pass for native packed inference | `src/orbitquant/kernels/mps.py`, `src/orbitquant/kernels/dispatch.py`, `tests/test_kernels.py`, `tests/test_orbit_linear.py`, `docs/kernel-audit.md` | A fused Metal shader performs activation norm, RPBH/FWHT, codebook lookup, and rescale. The native package performs packed matmul for generic leading dimensions, including short decode rows and partial matrix tiles, without full weight materialization. Offline weight and AdaLN quantization remain reference paths on MPS. |
| ROCm | Blocked for backend claim | No implementation in current tree | Do not claim ROCm optimization. |
| XPU | Blocked for backend claim | No implementation in current tree | Do not claim XPU optimization. |

## Acceleration Claim Boundary

- `runtime_mode="auto_fused"` is the default optimized policy. It avoids silent
  CUDA/MPS fallback to full dequantized BF16 weight materialization. Explicit
  `runtime_mode="dequant_bf16"` remains the compatibility/debug reference path.
- The CUDA W4A4 tensor-core path retains the paper's packed nearest-centroid
  indices but approximates each fixed Lloyd-Max centroid with a symmetric INT8
  code and one scalar per codebook. Measured codebook relative RMSE is
  0.21-0.28% for the FLUX.2 dimensions. This extra approximation is not part of
  the paper equation and is never presented as the exact reference path.

## Native Eval And Claim Policy

Native artifact readiness is separate from full GenEval or VBench scoring.
Those runs prove metric-table claims. The required artifact-readiness evidence
is:

- Ten-prompt native-resolution BF16-vs-OrbitQuant comparison matrix for image
  artifacts, including dense composition, fine detail, artistic style, and
  Latin, Russian, Japanese, and Chinese typography stress cases.
- Same prompt and seed for BF16 and OrbitQuant.
- Native settings from `src/orbitquant/eval/native_settings.py`.
- Finite, nonblank output checks.
- Compact artifact validation, checksums, manifest, and load test.

Published comparison matrices and aggregate compact metrics are not enough to
reconstruct paired proof after the fact. `native_smoke` readiness requires a
proof block derived from raw local records before upload; recovered proof claims
from compact summaries are rejected by the HF artifact audit.

Full metric runs are required only before saying that an artifact reproduces
the paper's GenEval or VBench numbers.

Automatic coverage outside the paper models does not imply that W4A4 or another
profile preserves task quality. Real GPT-2 and DeiT checks confirm that loading,
rotation folding, quantization, execution, and serialization work, while their
low-bit outputs are substantially more sensitive than the validated diffusion
targets. OrbitQuant therefore requires per-model quality validation and does
not silently substitute a different quantization method.

## Deviations And Limitations

| Item | Status | Rationale |
| --- | --- | --- |
| Explicit `dequant_bf16` runtime uses dequantized BF16 matmul. | Accepted reference path | It is kept for compatibility and debugging. Do not claim it as low-bit fused inference. |
| Zero weight rows use an epsilon guard for direction quantization. | Accepted implementation guard | The paper defines weight directions as `w' / ||w'||` for nonzero rows. The implementation divides by `max(||w'||, ε)` only when choosing codebook indices, stores the raw BF16 row norm, and dequantizes zero rows back to exactly zero. |
| The optimized CUDA W4A4 path evaluates an INT8 surrogate of each Lloyd-Max codebook. | Documented runtime deviation | Packed nearest-centroid indices, row norms, token norms, and the artifact remain unchanged. The surrogate adds 0.21-0.28% codebook relative RMSE for the measured FLUX.2 dimensions. `dequant_bf16` evaluates the stored Lloyd-Max centroids directly and is the exact methodology reference. |
| Full-model speedup is configuration-specific. | Accepted claim boundary | FLUX.2 Klein 9B on L40S reached practical SDNQ hot-generation parity with lower memory. That result does not establish universal speedup for other models, shapes, GPUs, or offload policies. |
| The paper's block-size enumeration omits `h=256`, although its stated largest-power-of-two-divisor rule gives `h=256` for Z-Image `d=3840` and Wan `d=8960` projections. | Paper inconsistency | The implementation follows the formal rule. The selected target dimensions produce `h` in `{256, 512, 1024, 2048, 4096}`. |
| Published checkpoints use converged Lloyd-Max codebook version 2 and `activation_eps=1e-10`. | Pass | All 14 canonical FLUX.2, FLUX.1-schnell, Z-Image-Turbo, and Wan2.1 artifacts were regenerated and validated. Legacy version 1 artifacts remain loadable by the library but are no longer the published release checkpoints. |
| Full config-derived inventories are audit artifacts, not committed source files. | Accepted artifact hygiene choice | Inventory summaries are recorded above; raw JSON may remain unpublished to avoid turning the repository into an artifact store. |
| Release-grade GenEval/VBench metrics are required only for metric claims. | Accepted claim boundary | Missing full metrics block paper metric/reproduction claims only. |
| ROCm and XPU kernels are not implemented. | Backend claim blocker | The release must either implement and verify them or explicitly exclude them. |

GenEval/VBench scores are not claimed without their corresponding external
metric runs. Any changed checkpoint must pass manifest, policy-inventory,
checksum, and native-output validation before publication.
