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
  benchmark execution. After adding kernel `upstream`/`source` metadata,
  `native-kernels/orbitquant-packed-matmul` passed
  `nix --option sandbox relaxed run .#ci-test -L` locally again on
  2026-07-08T16:59Z at OrbitQuant commit `5cc7d30`, including kernel-builder
  layout hooks, ABI compatibility for macOS 15/Python ABI 3.9, get-kernel
  loading, and 17 package tests.
  A private `WaveCut/orbitquant-packed-matmul` repo exists on Hugging Face, and
  its commit `6821e4cd5ff1894994d7137c1d861660cfeed1c8` contains the
  reviewable source package from `native-kernels/orbitquant-packed-matmul`,
  refreshed on 2026-07-08T18:00Z after adding CUDA launch-error checks,
  without generated `build/`, local `.venv/`, `__pycache__/`, or benchmark
  output directories. This source snapshot is review evidence only; the final
  loadable Kernel Hub artifact must be a `kernel`-type repository with
  `build/` variants uploaded by `kernel-builder build-and-upload`. Hugging Face
  `kernel-builder` currently asks publishers to request access through a
  discussion at
  `https://huggingface.co/spaces/kernels-community/README/discussions/new`.
  On 2026-07-08T17:02Z, `nix --option sandbox relaxed run .#build-and-copy -L`
  passed locally and copied 3 Metal build variants; the following
  `nix --option sandbox relaxed run .#build-and-upload -L` found those 3
  variants and stopped only at the Hugging Face permission error. The request
  was submitted as
  `https://huggingface.co/spaces/kernels-community/README/discussions/15`;
  a follow-up comment on 2026-07-08T18:03Z refreshed the review source snapshot
  to `6821e4cd5ff1894994d7137c1d861660cfeed1c8` and source archive SHA256
  `77aef6caa1bbdbbd77e2cbf5003423073e001191d008473c957795d7bed03651`.
  Re-running `nix --option sandbox relaxed run .#build-and-upload -L` on
  2026-07-08T18:12Z at OrbitQuant commit `956842a` rebuilt the three Metal
  variants, passed ABI/get-kernel build checks, and still stopped at the same
  Kernel Hub publish permission error.
  Approval remains pending. As of 2026-07-08T17:15Z, the linked
  `WaveCut/orbitquant-packed-matmul` source snapshot repo is still private; if
  reviewers cannot access it, provide a tracked source archive in the
  discussion or get explicit approval to make only that source-only kernel repo
  public. The request text is
  [kernel-hub-approval-request.md](kernel-hub-approval-request.md).
  The request includes local MPS smoke benchmark numbers from the matching
  `torch212-metal-aarch64-darwin` variant: W4 512x1024x1024 float16 at
  `0.00764581459807232` seconds/iteration over 20 iterations, and W4
  512x3072x3072 float16 at `0.10189520000712946` seconds/iteration over
  10 iterations. On 2026-07-08T17:10Z, the OrbitQuant native loader was also
  smoke-tested through `LOCAL_KERNELS`; with Torch 2.12.1 it selected
  `build/torch212-metal-aarch64-darwin`, ran `matmul_packed_weight` on MPS, and
  produced a finite float16 output tensor.
  CUDA/Triton partial gate passed on 2026-07-08T19:31Z at OrbitQuant commit
  `301d836` on a RunPod secure-cloud RTX 4090 host with Torch 2.9.1+cu128,
  CUDA 12.8, Triton 3.5.1, and driver 570.211.01. The run completed CUDA
  kernel tests, `orbitquant kernel-info`, `auto_fused` CUDA `kernel-bench`,
  and CUDA `quantize-bench` with exit 0 using
  `ORBITQUANT_RUN_NATIVE_KERNEL_PACKAGE_CI=0`. Native CUDA
  `native_packed_matmul` remains open: the available local
  `build/torch29-cxx11-cu130-x86_64-linux` variant failed to load on that CUDA
  12.8 host with `ImportError: libcudart.so.13`, so a CUDA 12.8-compatible
  kernel-builder variant or approved Hugging Face Kernel Hub upload is still
  required before closing this gate.
- [x] Final paper conformance audit is complete against arXiv 2607.02461, with
  documented deviations, implementation notes, and evidence that accepted
  deviations are intentional. The required audit checklist is
  [paper-methodology-audit.md](paper-methodology-audit.md), and the lightweight
  invariant gate is `scripts/run_paper_methodology_checks.sh`.
  Evidence: passed on 2026-07-08T15:49Z against arXiv 2607.02461v1, including
  codebook, RPBH, OrbitQuantLinear, AdaLN INT4, target-policy, native-setting,
  and config-inventory checks for FLUX.2 Klein, FLUX.1-schnell, Z-Image-Turbo,
  and Wan2.1.
- [x] Release wording separates the paper-aligned subset from extra targets:
  FLUX.1-schnell, Z-Image-Turbo, and Wan2.1 are paper targets; FLUX.2 Klein is
  an additional target unless the paper scope is expanded.
  Evidence: `README.md` separates paper-aligned artifacts from extra target
  artifacts, `src/orbitquant/artifacts/model_card.py` renders FLUX.2 Klein as
  `extra target; not an OrbitQuant paper reproduction model`, and
  `tests/test_readme.py` plus `tests/test_model_card.py` guard this wording.
  The 2026-07-08T16:00Z HF artifact audit reported `readme_mismatch_count=0`
  across all 14 published private artifact repos.
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
  forbidden remote files. The same audit was re-run on 2026-07-08T17:27Z and
  again reported 14/14 artifact-ready, 14/14 native-smoke-ready, 14/14
  metadata-complete, 14/14 policy-inventory-ready, zero forbidden files, zero
  remote checksum mismatches, and zero README mismatches.
- [x] Full-model module classification inventories are captured for FLUX.2
  Klein, FLUX.1-schnell, Z-Image-Turbo, and Wan2.1. Raw inventory JSON may
  remain unpublished, but each published artifact manifest must be
  cross-checked against the captured summary for quantized, AdaLN INT4, and
  skipped modules.
  Evidence: `scripts/run_paper_methodology_checks.sh` produced and hash-checked
  config-mode transformer inventories for all four native suites. The same
  2026-07-08T16:00Z HF artifact audit cross-checked those inventories against
  all 14 published private artifact manifests with `policy_inventory_ready=14`
  and `policy_inventory_error_count=0`. The 2026-07-08T17:27Z re-check again
  reported `policy_inventory_ready_count=14` and
  `policy_inventory_error_count=0`.
- [x] Compatibility is verified against the latest published releases and dev
  branches of Diffusers and Transformers with
  `scripts/run_hf_compat_checks.sh --mode all`, using the current Torch base.
  This gate uses registration, pipeline quantization config, and mini
  integration tests; it does not download model weights or generate samples.
  Evidence: passed on 2026-07-08T15:47Z with Torch 2.12.1, current/release
  Diffusers 0.39.0 and Transformers 5.13.0, and dev Diffusers 0.40.0.dev0 plus
  Transformers 5.14.0.dev0.
- [x] Checkpoint and model repositories are published with artifact-focused
  model cards, complete file manifests, checksums where applicable, and native
  comparison assets for the advertised targets. Cards must describe the
  artifact and usage, not host logs, raw eval dumps, or terminal transcripts.
  HF artifact audits must report `metadata_complete_ready` for every released
  artifact, proving quantization device, weight quantization backend, and
  staging mode provenance are present.
  Evidence: the 2026-07-08T16:00Z HF artifact audit passed with
  `repo_count=14`, `artifact_ready_count=14`, `metadata_complete_ready_count=14`,
  `native_smoke_ready_count=14`, `remote_checksum_mismatch_count=0`,
  `readme_mismatch_count=0`, and `forbidden_file_count=0`. The
  2026-07-08T17:27Z re-check reported the same zero-regression counts and
  `policy_inventory_ready_count=14`.
- [ ] The GitHub repository is public, tagged, and includes the release docs,
  license, source distribution expectations, and reproducible verification
  commands. Release-note content for the first public package release candidate
  is prepared in [release-0.1.0.md](release-0.1.0.md); repository visibility and
  the release tag remain pending explicit approval.
- [ ] The PyPI package is built and checked with `python -m build` and
  `python -m twine check dist/*`, then uploaded with `python -m twine upload
  dist/*`. Credentials may require a user-provided PyPI token or browser
  action before upload can complete.
  Current evidence: local build/check/smoke passed on 2026-07-08T16:06Z.
  `uv run --with build python -m build` produced
  `orbitquant-0.1.0.tar.gz` and `orbitquant-0.1.0-py3-none-any.whl`;
  `uv run --with twine python -m twine check dist/*` passed for both files;
  installing the wheel in a fresh venv, importing `orbitquant`, and running
  `orbitquant --version` returned `0.1.0`. Re-checked on 2026-07-08T17:27Z
  using a temporary build output directory: `twine check` passed for both
  artifacts, fresh wheel install/import succeeded, `OrbitQuantConfig()` defaulted
  to `runtime_mode="auto_fused"`, and `orbitquant --version` returned `0.1.0`.
  Re-checked on 2026-07-08T18:07Z after the native kernel source refresh:
  `tests/test_distribution.py` verifies that the wheel target contains only the
  Python runtime package while the source distribution keeps the tracked native
  kernel source and excludes generated `build/`, `.venv/`, `__pycache__/`,
  `.pyc`, `.so`, local artifact, and report paths. A fresh build in
  `/tmp/orbitquant-build-verify-20260708T180719Z` produced both expected
  artifacts and `twine check` passed. GitHub CI for OrbitQuant commit
  `f0c4855` passed on 2026-07-08T18:42Z, including package build, metadata
  check, and wheel smoke test after adding the RunPod SSH health preflight.
  Upload remains pending.
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
