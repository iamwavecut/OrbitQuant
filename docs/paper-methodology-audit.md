# OrbitQuant Paper Methodology Audit

Status date: 2026-07-07.

Paper source:

- arXiv abstract: https://arxiv.org/abs/2607.02461
- arXiv HTML, version 1: https://arxiv.org/html/2607.02461v1

This document compares the paper methodology against the current implementation
in this repository. It is not a development log. It is a release gate: claims
about paper reproduction, metrics, and kernel acceleration must be limited to
the items proven here or in later dated audit updates.

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
| No calibration data, prompt statistics, timestep ranges, or generated-image statistics are used to construct the quantizer. | Pass | `src/orbitquant/codebooks/lloyd_max.py`, `src/orbitquant/rotations/rpbh.py`, `src/orbitquant/functional.py`, `src/orbitquant/layers.py` | Codebooks depend on `(dim, bits, algorithm_version)`. Rotations depend on `(dim, seed, block_size)`. Activations use only runtime token norms. |
| One Lloyd-Max scalar codebook is built offline per input dimension and bit width. | Pass | `get_codebook(dim, bits)` in `src/orbitquant/codebooks/lloyd_max.py`; `tests/test_codebooks.py` | Persistent cache keys include algorithm version, dimension, and bits. No layer or prompt identifier enters the codebook. |
| Lloyd-Max target distribution is the coordinate marginal of a random unit vector in dimension `d`. | Pass | `_coordinate_density()` in `src/orbitquant/codebooks/lloyd_max.py` | The implementation uses the paper density up to normalization: `(1 - z^2)^((d - 3) / 2)` on `[-1, 1]`, then normalizes numerically. |
| Quantization uses nearest fixed centroids, with no zero-point, learned scale, per-channel range, or timestep/prompt range. | Pass | `LloydMaxCodebook.quantize_indices()` and `quantize()` in `src/orbitquant/codebooks/lloyd_max.py` | `torch.bucketize` against midpoint boundaries is equivalent to nearest-centroid lookup for sorted Lloyd-Max centroids. |
| RPBH uses uniform random permutation, Rademacher signs, per-block Walsh-Hadamard transform, and `1 / sqrt(block_size)` normalization. | Pass | `src/orbitquant/rotations/rpbh.py`, `src/orbitquant/rotations/fwht.py`, `tests/test_rpbh.py` | The implementation applies permutation first, then signs, then block FWHT and normalization. |
| RPBH stores compact permutation/sign metadata, not dense rotation matrices. | Pass | `src/orbitquant/rotations/rpbh.py`, `src/orbitquant/artifacts/writer.py` | Artifact rotation tensors are permutation, inverse permutation, signs, and normalization metadata. |
| Default paper block-size policy is the largest power of two dividing the input dimension. | Pass | `RPBHRotation.__post_init__()` in `src/orbitquant/rotations/rpbh.py`; `tests/test_rpbh.py` | Degenerate dimensions warn and fall back to signs/permutation only. |
| Weight rotation is folded offline so activations and weights share the same basis. | Pass | `OrbitQuantLinear.from_linear()` in `src/orbitquant/layers.py`; `tests/test_rpbh.py`; `tests/test_orbit_linear.py` | For PyTorch `linear(x, W, b)`, the code stores `W @ R` and computes `(x @ R) @ (W @ R).T + b`. |
| No inverse rotation is used in runtime quantized forward. | Pass | `OrbitQuantLinear.forward()` in `src/orbitquant/layers.py` | Inverse rotation appears only in `_dequantize()` conversion back to ordinary linear modules. |
| Weight rows are split into row norm plus unit direction; row norm is BF16. | Pass | `OrbitQuantLinear.from_linear()` in `src/orbitquant/layers.py` | Row norms are computed in FP32-compatible paths and stored as BF16 buffers. |
| Weight direction coordinates are quantized with the Lloyd-Max codebook and packed into low-bit indices. | Pass | `src/orbitquant/layers.py`, `src/orbitquant/packing/bitpack.py`, `tests/test_bitpack.py`, `tests/test_kernels.py` | Bit packing covers 2, 3, 4, and 6 bit paths. |
| Runtime activations compute per-token norm, normalize, apply RPBH, nearest-centroid quantize, and rescale by the token norm. | Pass | `src/orbitquant/functional.py`, `src/orbitquant/kernels/dispatch.py`, `src/orbitquant/kernels/triton_cuda.py`, `tests/test_orbit_linear.py`, `tests/test_kernels.py` | Arbitrary leading dimensions are preserved; the final feature dimension is the rotation dimension. |
| The only input-dependent runtime scalar is the per-token norm. | Pass | `src/orbitquant/functional.py` | Codebook, rotation, centroids, boundaries, signs, and permutation are fixed after construction. |
| AdaLN modulation projections use INT4 weight-only RTN with group size 64 and BF16 activations. | Pass | `src/orbitquant/adaln.py`, `src/orbitquant/config.py`, `src/orbitquant/artifacts/manifest.py`, `tests/test_adaln_rtn.py` | AdaLN wrappers do not call OrbitQuant activation rotation. Default `adaln_group_size` is 64, and artifacts record the actual group size so non-default artifacts are labeled. |
| Transformer-block linear projections are quantized through OrbitQuant. | Pass for configured transformer components | `src/orbitquant/policies/generic_dit.py`, `tests/test_target_policies.py`; local inventory summary below | Current Diffusers transformer configs are covered for FLUX.1, FLUX.2, Z-Image, and Wan. Artifact manifests still need per-artifact cross-checks before final publication. |
| Embeddings, timestep MLPs, final projection/unpatchify heads, text encoders, VAE, scheduler, safety/image processors remain unquantized by default. | Pass for configured transformer components | `src/orbitquant/policies/generic_dit.py`, `tests/test_target_policies.py`; local inventory summary below | Text encoders and VAE are outside the transformer component and are not passed into the default quantization helper. Artifact manifests still need per-artifact cross-checks before final publication. |
| Native settings match paper for FLUX.1-schnell, Z-Image-Turbo, and Wan 2.1-1.3B. | Pass for encoded settings | `src/orbitquant/eval/native_settings.py`, `README.md`, `src/orbitquant/artifacts/model_card.py` | Full metric runs are not required for development, but are required before metric-table or paper-reproduction claims. |
| FLUX.2 Klein is separated from paper-reproduction targets. | Pass | `src/orbitquant/eval/native_settings.py`, `src/orbitquant/artifacts/model_card.py`, `docs/release-gates.md` | It is treated as an additional target using paper-style native settings. |
| Runtime acceleration claims match implemented kernels. | Partial | `src/orbitquant/kernels/dispatch.py`, `src/orbitquant/kernels/triton_cuda.py`, `src/orbitquant/kernels/mps.py`, `tests/test_kernels.py` | CUDA/Triton covers several quant/dequant stages and optional packed matmul. Default runtime remains BF16 PyTorch matmul after dequantization. |
| Release-grade GenEval/VBench metrics are available for paper target claims. | Blocked for metric claims | `src/orbitquant/hub.py`, `docs/release-gates.md` | Full GenEval/VBench is optional during development. Missing metrics block only paper metric/reproduction claims. |

## Model Policy Evidence

Current policy coverage is pattern-based and intentionally scoped to Diffusers
transformer components. Full config-derived inventories were generated locally
under ignored `reports/native/module-inventories/` on 2026-07-07 using
`orbitquant inspect-policy --suite ... --load-mode config --dtype bfloat16
--output ...`; raw inventories are kept out of Git because they are audit
artifacts, not package source.

| Model family | Quantized patterns | AdaLN/INT4 patterns | Default skips | Evidence |
| --- | --- | --- | --- | --- |
| FLUX.1 | Transformer block attention `to_q`, `to_k`, `to_v`, `to_out`, joint text projections `add_*`, `to_add_out`, FFN modules, single-block projections. | `norm1.linear`, `norm1_context.linear`, single-block `norm.linear`. | `time_text_embed`, embedders, `norm_out`, final `proj_out`, text encoders, VAE. | `src/orbitquant/policies/generic_dit.py`; `tests/test_target_policies.py` instantiates `FluxTransformer2DModel`. |
| FLUX.2 Klein | Double-stream attention, text-conditioning projections, FFN `linear_in` and `linear_out`, fused single-stream `to_qkv_mlp_proj`, `to_out`. | `double_stream_modulation_img`, `double_stream_modulation_txt`, `single_stream_modulation`. | `time_guidance_embed`, embedders, `norm_out`, final `proj_out`, text encoders, VAE. | `src/orbitquant/policies/generic_dit.py`; `tests/test_target_policies.py` instantiates `Flux2Transformer2DModel`. |
| Z-Image-Turbo | `noise_refiner`, `context_refiner`, and `layers` attention projections and FFN `w1`, `w2`, `w3`. | Refiner and main-layer `adaLN_modulation`. | `all_x_embedder`, `t_embedder`, `cap_embedder`, final layer and final AdaLN modulation. | `src/orbitquant/policies/generic_dit.py`; `tests/test_target_policies.py` instantiates `ZImageTransformer2DModel`. |
| Wan 2.1 | `blocks.*.attn1`, `blocks.*.attn2`, and `blocks.*.ffn` projections. | None expected for Wan 2.1-1.3B. | `condition_embedder`, time/text embedders, final `proj_out`, text encoder, VAE. | `src/orbitquant/policies/generic_dit.py`; `tests/test_target_policies.py` instantiates `WanTransformer3DModel`. |

Inventory summary:

| Suite | Component class | Linear modules | OrbitQuant | AdaLN INT4 | BF16 skip |
| --- | --- | ---: | ---: | ---: | ---: |
| `flux2-native` | `Flux2Transformer2DModel` | 109 | 100 | 3 | 6 |
| `flux1-schnell-native` | `FluxTransformer2DModel` | 502 | 418 | 76 | 8 |
| `z-image-native` | `ZImageTransformer2DModel` | 276 | 238 | 32 | 6 |
| `wan-native` | `WanTransformer3DModel` | 306 | 300 | 0 | 6 |

Required before final publication:

- Verify the manifest `quantized_modules`, `adaln_modules`, and
  `skipped_modules` lists for each published artifact against the corresponding
  inventory summary with `orbitquant validate-artifact --policy-inventory`.
- Treat access failures for gated models as model-access blockers only, not as
  method blockers.

## Kernel And Runtime Evidence

| Backend | Status | Evidence | Claim boundary |
| --- | --- | --- | --- |
| CPU | Pass as reference | `src/orbitquant/kernels/dispatch.py`, `src/orbitquant/functional.py` | Correctness baseline only; no optimized CPU kernel claim. |
| CUDA/Triton | Partial optimized path | `src/orbitquant/kernels/triton_cuda.py`, `tests/test_kernels.py`, `tests/test_orbit_linear.py` | Covers activation norm/RPBH/lookup/rescale, packed weight dequant, low-bit pack/unpack, offline weight quantization, AdaLN RTN quant/dequant, and opt-in packed matmul. Default `dequant_bf16` still uses PyTorch BF16 matmul. |
| MPS/Metal | Partial optimized path | `src/orbitquant/kernels/mps.py`, `src/orbitquant/kernels/dispatch.py` | Metal handles codebook lookup/rescale and packed weight dequant. Norm and RPBH rotation still use PyTorch. |
| ROCm | Blocked for backend claim | No implementation in current tree | Do not claim ROCm optimization. |
| XPU | Blocked for backend claim | No implementation in current tree | Do not claim XPU optimization. |

Known kernel follow-up:

- `runtime_mode="triton_packed_matmul"` is CUDA-only and now fails before
  activation quantization when the input tensor is non-CUDA. It remains an
  opt-in experimental path until full-model CUDA benchmarks are complete.

## Native Eval And Claim Policy

The current development path does not require a full GenEval or VBench run.
Those runs are expensive and only prove metric-table claims. The required
artifact-readiness evidence is:

- Native-resolution BF16-vs-OrbitQuant comparison matrix.
- Same prompt and seed for BF16 and OrbitQuant.
- Native settings from `src/orbitquant/eval/native_settings.py`.
- Finite, nonblank output checks.
- Compact artifact validation, checksums, manifest, and load test.

Full metric runs are required only before saying that an artifact reproduces
the paper's GenEval or VBench numbers.

## Deviations And Limitations

| Item | Status | Rationale |
| --- | --- | --- |
| Default runtime uses dequantized BF16 matmul. | Accepted limitation | The paper's latency/memory analysis also evaluates fake quantization with BF16 matmul. Do not claim realized native low-bit tensor-core speedup for this default path. |
| Optional `triton_packed_matmul` exists but is not the default. | Accepted experimental path | It materially advances the kernel objective, but needs more full-model CUDA benchmarking before broad acceleration claims. |
| Full config-derived inventories are local ignored audit artifacts, not committed source files. | Accepted artifact hygiene choice | Inventory summaries are recorded above; raw JSON remains under ignored `reports/` to avoid turning the repository into an artifact store. |
| Release-grade GenEval/VBench metrics are not mandatory during development. | Accepted claim boundary | Missing full metrics block paper metric/reproduction claims only. |
| ROCm and XPU kernels are not implemented. | Backend claim blocker | The release must either implement and verify them or explicitly exclude them. |

## Next Audit Actions

1. Cross-check each published artifact manifest against the matching inventory
   summary before public release.
2. Run the targeted native comparison pack for each published artifact when
   refreshing cards, without promoting full GenEval/VBench to a development
   blocker.
3. Run full GenEval/VBench only before paper-reproduction or metric-table
   claims.
4. Complete backend-specific release notes for CUDA, CPU, MPS/Metal, ROCm, and
   XPU before public release.
