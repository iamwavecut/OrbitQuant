# OrbitQuant Release Gates

Use this checklist as the final acceptance gate before announcing or cutting a
release. Each item must have a dated artifact, command transcript, published
URL, or signed-off audit note.

- [ ] Final paper conformance audit is complete against arXiv 2607.02461, with
  documented deviations, implementation notes, and evidence that accepted
  deviations are intentional.
- [ ] Release wording separates the paper-aligned subset from extra targets:
  FLUX.1-schnell, Z-Image-Turbo, and Wan2.1 are paper targets; FLUX.2 Klein is
  an additional target unless the paper scope is expanded.
- [ ] Native metrics are complete for every advertised release artifact. Image
  artifacts include GenEval overall and per-task scores; Wan artifacts include
  all required VBench dimensions. Any nonzero missing-required-metric count is
  a release blocker.
- [ ] Full-model module classification inventories are captured for FLUX.2
  Klein, FLUX.1-schnell, Z-Image-Turbo, and Wan2.1, proving quantized, AdaLN
  INT4, and skipped modules match the paper policy.
- [ ] Optimized kernel audit and benchmarks are complete for the supported or
  targeted backends: CUDA/Triton, CPU, Metal/MPS, ROCm, and XPU. Unsupported
  backends have an explicit exclusion note instead of a silent gap.
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
