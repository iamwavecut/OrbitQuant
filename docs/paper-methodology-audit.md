# OrbitQuant Paper Methodology Audit Plan

This is the required conformance audit before a public release claim. The audit
source is arXiv `2607.02461v1`:

- https://arxiv.org/abs/2607.02461
- https://arxiv.org/html/2607.02461v1

The audit must compare the paper methodology against the current implementation,
not against design intent or old notes. Every item below needs a dated evidence
entry in the final audit report.

## Scope

- Methodology sections: 3.1, 3.2, 4.1 through 4.5.
- Layer policy and generation settings: Appendix B.1 and B.2.
- Low-bit and video settings: Appendix C.1 and C.2 where they affect release
  claims.
- Kernel and runtime claims: only claim acceleration for paths that are actually
  implemented, selected, benchmarked, and used by the artifact/runtime mode.

## Core Method Checks

### Calibration-Free Boundary

- Verify no calibration dataset, prompt set, timestep sample, or generated image
  statistics are used to construct codebooks, rotations, activation scales, or
  per-layer ranges.
- Inspect:
  - `src/orbitquant/codebooks/`
  - `src/orbitquant/rotations/`
  - `src/orbitquant/functional.py`
  - `src/orbitquant/layers.py`
  - `src/orbitquant/modeling.py`
- Evidence:
  - Static code notes proving codebooks depend only on input dimension, bit
    width, and codebook version.
  - Tests proving repeated prompts/timesteps do not update quantization state.

### RPBH Rotation

- Verify the implementation matches randomized permuted block-Hadamard:
  uniform random permutation first, per-block Rademacher signs, block FWHT, and
  `1 / sqrt(block_size)` normalization.
- Verify default block size is the largest power-of-two divisor of the input
  dimension, with explicit behavior for degenerate dimensions.
- Verify storage is compact permutation/sign metadata, not dense rotation
  matrices.
- Verify determinism for `(dimension, seed, block_size)` and distinct rotations
  for distinct seeds.
- Inspect:
  - `src/orbitquant/rotations/rpbh.py`
  - `src/orbitquant/rotations/fwht.py`
  - `tests/test_rpbh.py`
- Evidence:
  - Orthogonality and norm-preservation tests.
  - Weight-folding identity tests for PyTorch linear convention.
  - A short implementation note on exact multiplication order.

### Weight Folding Algebra

- Paper requirement: for PyTorch `linear(x, W, b)`, store rotated weights so
  the activation forward rotation and weight rotation cancel in the matrix
  product, with no inverse rotation at runtime.
- Verify the implementation uses the correct convention for `x @ W.T + b`.
- Verify `debug/no quantization` mode matches the original linear within
  floating point tolerance.
- Inspect:
  - `src/orbitquant/functional.py`
  - `src/orbitquant/layers.py`
  - `tests/test_orbit_linear.py`
  - `tests/test_rpbh.py`
- Evidence:
  - Direct random-matrix identity tests.
  - Saved artifact metadata showing rotated packed weights only.

### Lloyd-Max Codebook

- Verify the codebook is one fixed Lloyd-Max scalar codebook per
  `(input_dimension, bit_width, codebook_version)`.
- Verify the target marginal is the coordinate distribution of a random unit
  vector in the input dimension, not empirical activation ranges.
- Verify nearest-centroid lookup has no zero-point, learned scale, calibration
  range, or timestep/prompt dependence.
- Inspect:
  - `src/orbitquant/codebooks/lloyd_max.py`
  - `src/orbitquant/codebooks/cache.py`
  - `tests/test_codebooks.py`
- Evidence:
  - Symmetry, sorting, determinism, and monotonic MSE tests.
  - Numeric audit for dimensions used by FLUX.1, FLUX.2, Z-Image, and Wan.

### Weight Quantization

- Paper requirement: rotate each weight row into the shared basis, split into
  BF16 row norm plus unit direction, quantize coordinates with the Lloyd-Max
  codebook, and store low-bit codebook indices.
- Verify:
  - Quantization compute uses FP32 where needed for stability.
  - Row norms are stored as BF16 or explicitly documented if not.
  - Bias handling stays source precision and is not folded incorrectly.
  - Packed indices are bit-exact for 2, 3, 4, and 6 bit paths where supported.
- Inspect:
  - `src/orbitquant/functional.py`
  - `src/orbitquant/layers.py`
  - `src/orbitquant/packing/bitpack.py`
  - `src/orbitquant/kernels/`
  - `tests/test_bitpack.py`
  - `tests/test_orbit_linear.py`
  - `tests/test_kernels.py`

### Activation Quantization

- Paper requirement: at runtime compute per-token norm, normalize, apply RPBH,
  nearest-centroid quantize, rescale by token norm, then feed the rotated
  quantized activation to the rotated quantized weight.
- Verify:
  - Activation shapes with arbitrary leading dimensions flatten only over token
    rows and preserve the final feature dimension.
  - The only input-dependent scalar is the token norm.
  - No inverse rotation is used.
  - Padding/zero norm behavior is controlled by the configured epsilon.
- Inspect:
  - `src/orbitquant/functional.py`
  - `src/orbitquant/layers.py`
  - `src/orbitquant/kernels/`
  - `tests/test_orbit_linear.py`
  - `tests/test_kernels.py`

### AdaLN Treatment

- Paper requirement: AdaLN modulation projections use INT4 weight-only RTN with
  group size 64 and BF16 activations; Wan 2.1-1.3B has no AdaLN modulation.
- Verify:
  - AdaLN modules are not routed through OrbitQuant W/A activation rotation.
  - Group-size and INT4 ranges are correct.
  - FLUX and Z-Image policies classify modulation modules consistently.
  - Wan policy does not invent AdaLN modules.
- Inspect:
  - `src/orbitquant/rtn.py`
  - `src/orbitquant/policies/`
  - `tests/test_adaln_rtn.py`
  - policy tests for FLUX.1, FLUX.2, Z-Image, and Wan.

## Layer Policy Checks

- Quantize every transformer-block linear projection in the OrbitQuant path:
  image/text Q, K, V, output projections, FFN projections, joint-attention text
  path projections, and Wan cross-attention projections.
- Skip or keep BF16:
  embeddings, timestep MLP, final unpatchify/projection head, text encoders,
  VAE, scheduler, safety/image processors.
- Produce a full module inventory for each target:
  - `black-forest-labs/FLUX.2-klein-4B`
  - `black-forest-labs/FLUX.1-schnell`
  - `Tongyi-MAI/Z-Image-Turbo`
  - `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`
- Evidence:
  - `orbitquant inspect` output saved locally.
  - Artifact manifests with quantized, AdaLN, and skipped module lists.
  - A reviewer-readable diff table that maps each paper layer category to
    concrete module name patterns per model.

## Native Eval And Claims

- FLUX.1-schnell: 1024x1024, 4 steps, guidance 0.0, GenEval.
- Z-Image-Turbo: 1024x1024, 10 steps, guidance 0.0, GenEval.
- Wan 2.1-1.3B: 832x480, 81 frames, 50 steps, CFG 5.0, VBench.
- FLUX.2 Klein is an extra target, not a paper reproduction target.
- Release cards must not claim release-grade metrics until GenEval/VBench
  metrics are imported into artifacts and audited.

## Kernel Audit

- CUDA/Triton:
  - Verify runtime activation norm, RPBH/FWHT, codebook lookup/rescale,
    packed-weight dequant, low-bit pack/unpack, weight quantization, and AdaLN
    INT4 paths on an NVIDIA GPU.
  - Benchmark cold compile separately from hot path.
  - State clearly when `runtime_mode=dequant_bf16` still uses BF16 PyTorch
    matmul after optimized quant/dequant stages.
- CPU:
  - Verify reference correctness and document that it is not optimized unless a
    dedicated CPU kernel exists.
- Metal/MPS, ROCm, XPU:
  - Either provide tested optimized paths or explicit unsupported/not-yet
    optimized notes. Do not imply cross-platform optimization without evidence.

## Final Audit Output

The final audit report must include:

- Source paper revision and access date.
- Requirement-by-requirement pass/fail table.
- File and test evidence for each pass.
- Exact deviations from the paper, with rationale and whether each deviation is
  acceptable for release.
- Open blockers, especially methodology mismatches, missing native metrics, or
  unverified kernel backends.
