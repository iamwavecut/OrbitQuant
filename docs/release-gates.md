# OrbitQuant Release Gates

Use this checklist as the final acceptance gate before announcing or cutting a
release. Each item must have a dated artifact, verification output, published
URL, or signed-off audit note.

- [ ] Kernel audit and benchmarks match the advertised backend claim boundary:
  CUDA/Triton partial optimized, Metal/MPS partial optimized, CPU
  reference-only, and ROCm/XPU explicitly unsupported unless implemented later.
  The current backend claim boundary is [kernel-audit.md](kernel-audit.md).
  Current partial evidence: MPS/Metal passed
  `scripts/run_mps_kernel_checks.sh` on 2026-07-08T15:58Z, including native
  packed matmul load and explicit `runtime_mode="native_packed_matmul"`
  benchmark execution. CUDA/Triton remains pending on a CUDA host.
- [x] Final paper conformance audit is complete against arXiv 2607.02461, with
  documented deviations, implementation notes, and evidence that accepted
  deviations are intentional. The required audit checklist is
  [paper-methodology-audit.md](paper-methodology-audit.md), and the lightweight
  invariant gate is `scripts/run_paper_methodology_checks.sh`.
  Evidence: passed on 2026-07-08T15:49Z against arXiv 2607.02461v1, including
  codebook, RPBH, OrbitQuantLinear, AdaLN INT4, target-policy, native-setting,
  and config-inventory checks for FLUX.2 Klein, FLUX.1-schnell, Z-Image-Turbo,
  and Wan2.1.
- [ ] Release wording separates the paper-aligned subset from extra targets:
  FLUX.1-schnell, Z-Image-Turbo, and Wan2.1 are paper targets; FLUX.2 Klein is
  an additional target unless the paper scope is expanded.
- [x] Native artifact validation is complete for every advertised release
  artifact. At minimum this includes native-resolution BF16-vs-OrbitQuant
  comparison assets, load validation, finite-output checks, manifests, and
  checksums. Published compact artifacts must include a `native_smoke` proof
  block in `benchmark/summary.json`; raw generation records remain local-only.
  Evidence: `orbitquant audit-hf-artifacts --namespace WaveCut
  --policy-inventory-root reports/native/module-inventories
  --fail-on-artifact-regression` passed on 2026-07-08T16:00Z for 14/14
  private artifact repos. It reported 14/14 artifact-ready, 14/14 native-smoke
  ready, 14/14 metadata-complete, zero manifest warnings, zero missing
  metadata, zero remote checksum mismatches, zero README mismatches, and zero
  forbidden remote files.
- [x] Full-model module classification inventories are captured for FLUX.2
  Klein, FLUX.1-schnell, Z-Image-Turbo, and Wan2.1. Raw inventory JSON may
  remain unpublished, but each published artifact manifest must be
  cross-checked against the captured summary for quantized, AdaLN INT4, and
  skipped modules.
  Evidence: `scripts/run_paper_methodology_checks.sh` produced and hash-checked
  config-mode transformer inventories for all four native suites. The same
  2026-07-08T16:00Z HF artifact audit cross-checked those inventories against
  all 14 published private artifact manifests with `policy_inventory_ready=14`
  and `policy_inventory_error_count=0`.
- [x] Compatibility is verified against the latest published releases and dev
  branches of Diffusers and Transformers with
  `scripts/run_hf_compat_checks.sh --mode all`, using the current Torch base.
  This gate uses registration, pipeline quantization config, and mini
  integration tests; it does not download model weights or generate samples.
  Evidence: passed on 2026-07-08T15:47Z with Torch 2.12.1, current/release
  Diffusers 0.39.0 and Transformers 5.13.0, and dev Diffusers 0.40.0.dev0 plus
  Transformers 5.14.0.dev0.
- [ ] Checkpoint and model repositories are published with artifact-focused
  model cards, complete file manifests, checksums where applicable, and native
  comparison assets for the advertised targets. Cards must describe the
  artifact and usage, not host logs, raw eval dumps, or terminal transcripts.
  HF artifact audits must report `metadata_complete_ready` for every released
  artifact, proving quantization device, weight quantization backend, and
  staging mode provenance are present.
- [ ] The GitHub repository is public, tagged, and includes the release docs,
  license, source distribution expectations, and reproducible verification
  commands.
- [ ] The PyPI package is built and checked with `python -m build` and
  `python -m twine check dist/*`, then uploaded with `python -m twine upload
  dist/*`. Credentials may require a user-provided PyPI token or browser
  action before upload can complete.
- [x] ComfyUI compatibility is verified after the relevant schema stabilizes,
  including load, graph execution, and artifact metadata behavior.
  Evidence: ComfyUI-OrbitQuant commit `1d73b36` passed `uv run pytest -q`
  and `uv run ruff check .` on 2026-07-08T15:54Z. The package smoke covers
  legacy node mappings, V3 entrypoint/schema/delegation, real OrbitQuant
  artifact load, inspector-to-loader node graph behavior, metadata propagation,
  and finite forward execution through the restored `OrbitQuantLinear`.
- [ ] Release-grade metrics are complete before making paper reproduction or
  metric-table claims. Image paper-target artifacts then include GenEval
  overall and per-task scores; Wan artifacts then include all required VBench
  dimensions. Missing release metrics block only those metric/reproduction
  claims; compact artifacts without those metrics must present native comparison
  status instead of paper-reproduction metric claims.
