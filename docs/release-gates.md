# OrbitQuant Release Gates

Use this checklist as the final acceptance gate before announcing or cutting a
release. Each item must have a dated artifact, command transcript, published
URL, or signed-off audit note.

- [ ] Final paper conformance audit is complete against arXiv 2607.02461, with
  documented deviations, implementation notes, and evidence that accepted
  deviations are intentional. The required audit checklist is
  [paper-methodology-audit.md](paper-methodology-audit.md).
- [ ] Release wording separates the paper-aligned subset from extra targets:
  FLUX.1-schnell, Z-Image-Turbo, and Wan2.1 are paper targets; FLUX.2 Klein is
  an additional target unless the paper scope is expanded.
- [ ] Native artifact validation is complete for every advertised release
  artifact. At minimum this includes native-resolution BF16-vs-OrbitQuant
  comparison assets, load validation, finite-output checks, manifests, and
  checksums. Published compact artifacts must include a `native_smoke` proof
  block in `benchmark/summary.json`; raw generation records remain local-only.
- [ ] Release-grade metrics are complete before making paper reproduction or
  metric-table claims. Image paper-target artifacts then include GenEval
  overall and per-task scores; Wan artifacts then include all required VBench
  dimensions. Missing release metrics block only those metric/reproduction
  claims, not ordinary compact artifact development, artifact cleanup, kernel
  work, or model-card refreshes.
- [ ] Full-model module classification inventories are captured for FLUX.2
  Klein, FLUX.1-schnell, Z-Image-Turbo, and Wan2.1. Raw inventory JSON can stay
  local under ignored `reports/`, but each published artifact manifest must be
  cross-checked against the captured summary for quantized, AdaLN INT4, and
  skipped modules.
- [ ] Kernel audit and benchmarks match the advertised backend claim boundary:
  CUDA/Triton partial optimized, Metal/MPS partial optimized, CPU
  reference-only, and ROCm/XPU explicitly unsupported unless implemented later.
  The current backend claim boundary is [kernel-audit.md](kernel-audit.md).
- [ ] Compatibility is verified against the latest published releases and dev
  branches of Diffusers and Transformers.
- [ ] ComfyUI compatibility is verified after the relevant schema stabilizes,
  including load, graph execution, and artifact metadata behavior.
- [ ] Checkpoint and model repositories are published with artifact-focused
  model cards, complete file manifests, checksums where applicable, and native
  comparison assets for the advertised targets. Cards must describe the
  artifact and usage, not host logs, raw eval dumps, or terminal transcripts.
- [ ] The GitHub repository is public, tagged, and includes the release docs,
  license, source distribution expectations, and reproducible verification
  commands.
- [ ] The PyPI package is built and checked with `python -m build` and
  `python -m twine check dist/*`, then uploaded with `python -m twine upload
  dist/*`. Credentials may require a user-provided PyPI token or browser
  action before upload can complete.
