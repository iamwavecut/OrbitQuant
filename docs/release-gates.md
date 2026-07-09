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
  loading, and package tests.
  A public `WaveCut/orbitquant-packed-matmul` source snapshot repo exists on
  Hugging Face, and the checked commit
  `cb0ceb1a4d070556c52cfba691aba3f6647c246b` contains the reviewable source
  package from `native-kernels/orbitquant-packed-matmul`, without generated
  `build/`, local `.venv/`, `__pycache__/`, binary extension, or benchmark
  output files. The PyPI `orbitquant-0.1.0.tar.gz` source distribution also
  contains the same tracked kernel source under
  `orbitquant-0.1.0/native-kernels/orbitquant-packed-matmul/`, with SHA256
  `6abedb769b32c8d70f2763278e106346319d628d85ed7469549faa5020ab1a89`.
  This source snapshot is review evidence only; the final loadable Kernel Hub
  artifact must be a `kernel`-type repository with `build/` variants uploaded
  by `kernel-builder build-and-upload`. Hugging Face `kernel-builder`
  currently asks publishers to request access through a discussion at
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
  Approval remains pending for the loadable `kernel`-type repository. As of
  2026-07-09T11:54Z, the linked `WaveCut/orbitquant-packed-matmul` source
  snapshot repo is public, and a live Hugging Face check across the source
  snapshot plus all 14 canonical OrbitQuant model artifact repos reported
  `private_count=0`. The request text and reviewer follow-up notes are in
  [kernel-hub-approval-request.md](kernel-hub-approval-request.md). A
  follow-up comment was posted to discussion 15 on 2026-07-09T11:56Z with the
  public source snapshot URL, checked commit, PyPI source distribution URL, and
  SHA256. Another follow-up comment was posted on 2026-07-09T12:22Z after
  updating `benchmarks/benchmark.py` to report both predequantized-F.linear and
  dequantize-then-F.linear baselines; it clarified that current local MPS
  native packed matmul numbers are correctness and memory-path evidence only,
  not throughput proof.
  The public source snapshot was updated again on 2026-07-09T12:39Z to commit
  `cb0ceb1a4d070556c52cfba691aba3f6647c246b`; the benchmark now reports
  `packed_weight_path_bytes`, `materialized_weight_bytes`, and
  `packed_weight_path_vs_materialized_weight_ratio` to make weight-side storage
  accounting explicit. A follow-up comment was posted to discussion 15 on
  2026-07-09T12:41Z with the new snapshot URL and the same non-throughput
  claim boundary.
  A reviewer then asked for a way to try one model with these kernels.
  `scripts/verify_hf_kernel_model_artifact.py` was added to verify the default
  `WaveCut/FLUX.2-klein-4B-OrbitQuant-W4A4` artifact at one restored packed
  transformer projection using `runtime_mode="native_packed_matmul"`, with
  `dequant_bf16` comparison and storage accounting, without full generation.
  A follow-up comment was posted on 2026-07-09T12:50Z with commit
  `f42d2dc19897adde62ec3ebb33e4ce748255dd54` and MPS/CUDA `LOCAL_KERNELS`
  example commands for the verifier.
  The verifier then passed locally on 2026-07-09T12:57Z using the
  `torch212-metal-aarch64-darwin` local kernel variant and the published
  `WaveCut/FLUX.2-klein-4B-OrbitQuant-W4A4` artifact. It verified
  `transformer_blocks.0.attn.to_q` (3072x3072) through
  `runtime_mode="native_packed_matmul"` against `dequant_bf16` with
  `finite=true`, `allclose_to_dequant_bf16=true`,
  `max_abs_error_vs_dequant_bf16=0.001953125`, and
  `packed_weight_path_vs_materialized_weight_ratio=0.2503289116753472`.
  A follow-up comment was posted to discussion 15 on 2026-07-09T12:58Z with
  the verifier command and JSON result summary.
  The same published artifact layer was re-verified on 2026-07-09T13:05Z with
  `runtime_mode="auto_fused"` and the same local
  `torch212-metal-aarch64-darwin` package. It again reported `finite=true`,
  `allclose_to_dequant_bf16=true`,
  `max_abs_error_vs_dequant_bf16=0.001953125`, and
  `packed_weight_path_vs_materialized_weight_ratio=0.2503289116753472`,
  proving the default optimized dispatch reaches the native packed matmul path
  for this real published artifact when the local Metal kernel package is
  available. Another follow-up comment was posted to discussion 15 on
  2026-07-09T13:05Z with the `auto_fused` command and JSON result summary.
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
  `ORBITQUANT_RUN_NATIVE_KERNEL_PACKAGE_CI=0`. A public-package CUDA smoke
  passed on 2026-07-09T13:15Z on the active RunPod
  `orbitquant-cuda-gate-4090` pod (`ofz7pyxcw6vlzm`) with Torch 2.9.1+cu128,
  CUDA 12.8, driver 580.159.04, and an NVIDIA GeForce RTX 4090. The smoke
  installed `orbitquant[kernels]==0.1.0` from PyPI into a temporary `/tmp` venv
  using `--system-site-packages`, ran `orbitquant kernel-info`, and ran
  `orbitquant kernel-bench --device cuda --dtype float16 --runtime-mode
  auto_fused --tokens 16 --in-features 128 --out-features 128 --warmup 1
  --iterations 2`. The benchmark selected `triton_cuda` for both activation
  kernels and weight quantization, kept packed weight indices and row norms on
  `cuda:0`, reported `forward_prewarmed_ms=0.14182400703430176`, and removed
  the temporary venv after completion. This verifies the published PyPI package
  CUDA/Triton `auto_fused` path, not the separate native CUDA Kernel Hub
  package. After a reviewer asked for actual numbers, a model-like CUDA
  microbenchmark was run on the same pod on 2026-07-09T13:21Z with
  `orbitquant[kernels]==0.1.0` from PyPI, `tokens=512`, `in_features=3072`,
  `out_features=3072`, W4A4, float16, warmup 2, and 5 iterations. The
  `auto_fused` Triton path reported `forward_prewarmed_ms=0.6518784046173096`,
  `forward_cold_ms=0.6581952095031738`, and `peak_memory_bytes=69293568`.
  The explicit `dequant_bf16` reference reported
  `forward_prewarmed_ms=0.13742079734802246`,
  `forward_cold_ms=0.20090880393981933`, and
  `peak_memory_bytes=115025408`. These numbers were posted to discussion 15 on
  2026-07-09T13:22Z with the explicit caveat that current CUDA/Triton
  `auto_fused` is memory-path evidence, not a throughput win on this RTX 4090
  microbenchmark. A follow-up tile sweep on the same RTX 4090, PyPI package,
  and 512x3072x3072 W4A4 float16 benchmark found the best tested Triton packed
  matmul tile at `block_m=32`, `block_n=128`, `block_k=64`, `num_warps=8`.
  A focused 20-iteration confirmation measured the previous default
  `block_n=64` at `forward_prewarmed_ms=0.6374400138854981` and the selected
  `block_n=128` default at `forward_prewarmed_ms=0.596992015838623`, both with
  `peak_memory_bytes=69293568`. This updates the package default tile for the
  packed matmul path, but it remains a local CUDA/Triton microbenchmark result
  and does not change the no-throughput-win claim boundary. A post-publication
  CUDA smoke for `orbitquant[kernels]==0.1.1` passed on 2026-07-09T14:07Z on
  the same active RunPod RTX 4090 with Torch 2.9.1+cu128, CUDA 12.8, driver
  580.159.04, 512x3072x3072 W4A4 float16, warmup 5, and 20 measured
  iterations. It verified the published package default
  `OrbitQuantConfig().runtime_mode == "auto_fused"` and
  `packed_matmul_block_n == 128`. The `auto_fused` run selected
  `triton_cuda`, used tile `{block_m=32, block_n=128, block_k=64,
  num_warps=8}`, kept packed weight indices and row norms on `cuda:0`,
  reported `forward_prewarmed_ms=0.5946400165557861`,
  `forward_cold_ms=0.6080512046813965`, and
  `peak_memory_bytes=69293568`. The explicit `dequant_bf16` reference reported
  `forward_prewarmed_ms=0.12743680477142333`,
  `forward_cold_ms=0.1994752049446106`, and
  `peak_memory_bytes=115025408`. The packed path saved 45,731,840 peak-memory
  bytes in this isolated benchmark but remained 4.666x slower than
  `dequant_bf16`, so the no-throughput-win claim boundary remains unchanged.
  Native CUDA
  artifact-layer verification on the same active RunPod RTX 4090 passed on
  2026-07-09T13:54Z using the published
  `WaveCut/FLUX.2-klein-4B-OrbitQuant-W4A4` artifact, source model
  `black-forest-labs/FLUX.2-klein-4B` revision
  `e7b7dc27f91deacad38e78976d1f2b499d76a294`, Torch 2.9.1+cu128, CUDA 12.8,
  driver 580.159.04, `runtime_mode="auto_fused"`,
  `activation_kernel_backend="triton_cuda"`, float16, W4A4, and 512 tokens. It
  restored `transformer_blocks.0.attn.to_q` (3072x3072), reported
  `finite=true`, `allclose_to_dequant_bf16=true`,
  `max_abs_error_vs_dequant_bf16=0.015625`,
  `packed_weight_path_bytes=4724800`, `materialized_weight_bytes=18874368`,
  `packed_weight_path_vs_materialized_weight_ratio=0.2503289116753472`, and
  `peak_memory_bytes=87756800`. Timings were
  `auto_fused_forward_first_ms=477.2464599609375`,
  `auto_fused_forward_prewarmed_ms=0.6343167781829834`,
  `dequant_bf16_forward_first_ms=103.32876586914062`, and
  `dequant_bf16_forward_prewarmed_ms=0.1256432056427002` with warmup 5 and 20
  measured iterations. This is actual published model artifact layer evidence
  for the CUDA/Triton path; it is not a full image-generation benchmark and
  still does not prove a throughput win. A follow-up comment with these actual
  model artifact numbers was posted to discussion 15 on 2026-07-09T14:00Z.
  Native CUDA
  `native_packed_matmul` remains open: the available local
  `build/torch29-cxx11-cu130-x86_64-linux` variant failed to load on that CUDA
  12.8 host with `ImportError: libcudart.so.13`, so a CUDA 12.8-compatible
  kernel-builder variant or approved Hugging Face Kernel Hub upload is still
  required before closing this gate. The CUDA gate now builds and loads the
  exact `redistributable.<runtime-variant>` path instead of selecting ignored
  local `build/` artifacts. The current HF `kernel-builder` matrix exports
  `torch211-cxx11-cu128-x86_64-linux`, but not
  `torch29-cxx11-cu128-x86_64-linux`; `kernels` rejects CUDA variants newer
  than the runtime CUDA minor version. The existing Torch 2.9.1+cu128 RunPod
  can still be used for Triton/eval work, but native CUDA package closure needs
  a runtime with an exported compatible variant, such as Torch 2.11+cu128, or
  an approved Kernel Hub upload with a compatible build. The CUDA gate is now
  prebuilt-first and build-safe: after a failed native prebuilt load,
  `scripts/run_cuda_kernel_checks.sh` will not start a kernel-builder/Nix
  source build unless `ORBITQUANT_ALLOW_NATIVE_KERNEL_BUILD=1` is set. A
  2026-07-09 prebuilt-only loader check still returned 404 for the Kernel Hub
  repo while the public source snapshot model repo resolved to commit
  `cb0ceb1a4d070556c52cfba691aba3f6647c246b`. A 2026-07-09T15:06Z live
  refresh confirmed discussion 15 remains open, the public source snapshot
  model repo still resolves to commit
  `cb0ceb1a4d070556c52cfba691aba3f6647c246b`, and
  `repo_info("WaveCut/orbitquant-packed-matmul", repo_type="kernel")` still
  returns 404. The loadable Kernel Hub package therefore remains an external
  approval gate, not a completed release artifact.
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
  The 2026-07-09T11:44Z HF artifact audit reported `readme_mismatch_count=0`
  across all 14 public artifact repos.
- [x] Native artifact validation is complete for every advertised release
  artifact. At minimum this includes native-resolution BF16-vs-OrbitQuant
  comparison assets, load validation, finite-output checks, manifests, and
  checksums. Published compact artifacts must include a `native_smoke` proof
  block in `benchmark/summary.json`; raw generation records remain local-only.
  Evidence: `orbitquant audit-hf-artifacts --namespace WaveCut
  --policy-inventory-root reports/native/module-inventories
  --fail-on-artifact-regression` passed on 2026-07-09T11:44Z for 14/14
  public artifact repos. It reported 14/14 artifact-ready, 14/14 native-smoke
  ready, 14/14 metadata-complete, zero manifest warnings, zero missing
  metadata, zero remote checksum mismatches, zero README mismatches, and zero
  forbidden remote files. It also reported `public_count=14`,
  `private_count=0`, 14/14 policy-inventory-ready, and 144 missing
  release-grade metrics. A 2026-07-09T15:06Z rerun with `--summary-only` and
  the same policy inventory root again passed `--fail-on-artifact-regression`
  with 14/14 artifact-ready, native-smoke-ready, metadata-complete, and
  policy-inventory-ready; it reported `release_eval_applicable_count=10`,
  `release_eval_ready_count=0`, and `missing_required_metric_count=144`.
  A 2026-07-09T18:00Z live rerun with
  `--policy-inventory-root reports/paper-methodology/module-inventories`
  again passed `--summary-only --fail-on-artifact-regression`, reporting
  14/14 artifact-ready, native-smoke-ready, metadata-complete, and
  policy-inventory-ready with `readme_mismatch_count=0`,
  `forbidden_file_count=0`, `remote_checksum_mismatch_count=0`, and the same
  144 missing release-grade metrics.
- [x] Full-model module classification inventories are captured for FLUX.2
  Klein, FLUX.1-schnell, Z-Image-Turbo, and Wan2.1. Raw inventory JSON may
  remain unpublished, but each published artifact manifest must be
  cross-checked against the captured summary for quantized, AdaLN INT4, and
  skipped modules.
  Evidence: `scripts/run_paper_methodology_checks.sh` produced and hash-checked
  config-mode transformer inventories for all four native suites. The same
  2026-07-09T11:44Z HF artifact audit cross-checked those inventories against
  all 14 public artifact manifests with `policy_inventory_ready_count=14` and
  `policy_inventory_error_count=0`.
- [x] Compatibility is verified against the latest published releases and dev
  branches of Diffusers and Transformers with
  `scripts/run_hf_compat_checks.sh --mode all`, using the current Torch base.
  This gate uses registration, pipeline quantization config, and mini
  integration tests; it does not download model weights or generate samples.
  Evidence: passed on 2026-07-09T17:18Z with Torch 2.12.1, current/release
  Diffusers 0.39.0 and Transformers 5.13.0, and dev Diffusers 0.40.0.dev0
  at `208704a` plus Transformers 5.14.0.dev0 at `0bc3554`.
- [x] Checkpoint and model repositories are published with artifact-focused
  model cards, complete file manifests, checksums where applicable, and native
  comparison assets for the advertised targets. Cards must describe the
  artifact and usage, not host logs, raw eval dumps, or terminal transcripts.
  HF artifact audits must report `metadata_complete_ready` for every released
  artifact, proving quantization device, weight quantization backend, and
  staging mode provenance are present.
  Evidence: the 2026-07-09T11:44Z HF artifact audit passed with
  `repo_count=14`, `artifact_ready_count=14`, `metadata_complete_ready_count=14`,
  `native_smoke_ready_count=14`, `remote_checksum_mismatch_count=0`,
  `readme_mismatch_count=0`, `forbidden_file_count=0`, `public_count=14`, and
  `private_count=0`. A 2026-07-09T15:06Z `--summary-only` rerun preserved the
  same compact artifact counts and still reported `forbidden_file_count=0`,
  `remote_checksum_mismatch_count=0`, and `readme_mismatch_count=0`. A direct
  2026-07-09T18:00Z README/assets scan across all 14 canonical artifact repos
  found no stale `orbitquant==0.1.3`, `orbitquant>=0.1.3`,
  `orbitquant[kernels]==0.1.3`, `orbitquant[kernels]>=0.1.3`, or
  `comfyui-orbitquant==0.1.2` install instructions, confirmed
  `pip install "orbitquant[hf]"` and `runtime_mode="auto_fused"` were present
  in every card, and found no raw remote assets outside the final
  `assets/*_generation_comparison_matrix.webp` card assets.
- [x] The GitHub repository is public, tagged, released, and includes the
  release docs, license, source distribution expectations, and reproducible
  verification commands.
  Release-note content for the first public package release candidate is
  [release-0.1.0.md](release-0.1.0.md). The exact manual tag/release
  sequence is recorded in [publication-checklist.md](publication-checklist.md).
  GitHub CI for release-readiness
  commit `0c0f63a` passed as run `29016554734` on 2026-07-09, including HF
  integration tests, full pytest, package build, `twine check`, and wheel
  smoke. A live GitHub check on 2026-07-09T11:54Z reported
  `iamwavecut/OrbitQuant` as `PUBLIC`, Apache-2.0 licensed, with homepage
  `https://pypi.org/project/orbitquant/`. Git tag `v0.1.0` was created on
  2026-07-09 and resolves to commit
  `ce5c232a8bf9b450c7d94eeae07445317c98b1d0`, matching the GitHub Actions
  PyPI publish run head SHA. GitHub Release
  `https://github.com/iamwavecut/OrbitQuant/releases/tag/v0.1.0` was published
  on 2026-07-09T12:10Z with exact PyPI-matching assets:
  `orbitquant-0.1.0.tar.gz` SHA256
  `6abedb769b32c8d70f2763278e106346319d628d85ed7469549faa5020ab1a89` and
  `orbitquant-0.1.0-py3-none-any.whl` SHA256
  `dfbfa80ff79132457b6918d69f8ae9d8961ea3d898487d105a0e74da906eeaaa`.
  Patch release notes for the packed-matmul default update are
  [release-0.1.1.md](release-0.1.1.md). GitHub CI for commit `7d797ba`
  passed as run `29022202797`, tag `v0.1.1` resolves to that commit, and
  GitHub Release `https://github.com/iamwavecut/OrbitQuant/releases/tag/v0.1.1`
  was published on 2026-07-09 with exact PyPI-matching assets:
  `orbitquant-0.1.1.tar.gz` SHA256
  `5972b6cfd1d89653fb9ac17668f72818c61a0e3cc5ea1cdd46e59d54405dc1ff` and
  `orbitquant-0.1.1-py3-none-any.whl` SHA256
  `b829c5df00093e697872ca24104e6ef38dbe7e7d70b2c2e33560bfaec224cde1`.
  Patch release notes for compact artifact audit output are
  [release-0.1.2.md](release-0.1.2.md). GitHub CI for commit `f18feb9`
  passed as run `29025114082`, tag `v0.1.2` points at commit
  `f18feb95f7595965f046b3455997d6ce9b8e8e4a`, and GitHub Release
  `https://github.com/iamwavecut/OrbitQuant/releases/tag/v0.1.2` was published
  on 2026-07-09 with exact PyPI-matching assets:
  `orbitquant-0.1.2.tar.gz` SHA256
  `4afc15589fe713345dfa47be30dc078d943484a1f4bbd4861a413532a6ec8377` and
  `orbitquant-0.1.2-py3-none-any.whl` SHA256
  `3bdddebaa46f60307ed50e2ebf4b7ff4fef7817845b512ecfbb6fbf8ba71c91c`.
  Patch release notes for the cross-platform kernel-extra dependency fix are
  [release-0.1.3.md](release-0.1.3.md). GitHub CI for commit `02166ec`
  passed as run `29034308427`, tag `v0.1.3` points at commit
  `02166ec6bdcd8920b0012a1ff930dce8dd976fdb`, and GitHub Release
  `https://github.com/iamwavecut/OrbitQuant/releases/tag/v0.1.3` was published
  on 2026-07-09 with exact PyPI-matching assets:
  `orbitquant-0.1.3.tar.gz` SHA256
  `008f949641d00df46f840580c424b9e9fad0d853b4345d454a0b8042c61f3366` and
  `orbitquant-0.1.3-py3-none-any.whl` SHA256
  `181e9b532a07312ec47b54091d651b5a62d5aefd32c88d40830bd8529a0fdc53`.
  Patch release notes for the external GenEval/VBench metric import fixes are
  [release-0.1.4.md](release-0.1.4.md). GitHub CI for commit `a7d28d9`
  passed as run `29037952922`, tag `v0.1.4` points at commit
  `a7d28d96ff47d5ae72121bbefc3aab30ca732b42`, and GitHub Release
  `https://github.com/iamwavecut/OrbitQuant/releases/tag/v0.1.4` was published
  on 2026-07-09 with exact PyPI-matching assets:
  `orbitquant-0.1.4.tar.gz` SHA256
  `7db4168bb2e9c3b838af0a932a96ca505d7bb91475239545d3e3a4c8130e07f4` and
  `orbitquant-0.1.4-py3-none-any.whl` SHA256
  `641a4d74a811023ff6cc39ae9680eb1393eb5e2b5dd390d586ff087ac1533af3`.
- [x] The PyPI package is published as `orbitquant==0.1.4`.
  Evidence: commit `ce5c232` added the manual
  `.github/workflows/publish-pypi.yml` Trusted Publishing workflow. A PyPI
  pending publisher was registered for project `orbitquant`, owner
  `iamwavecut`, repository `OrbitQuant`, workflow `publish-pypi.yml`, and
  environment `pypi`. GitHub Actions run `29015072821` completed successfully on
  2026-07-09, including full pytest, `ruff check`, package build, `twine check`,
  wheel smoke, OIDC publication, and PyPI digital attestations. The PyPI JSON
  API reports version `0.1.0` with `orbitquant-0.1.0.tar.gz` and
  `orbitquant-0.1.0-py3-none-any.whl`; `python -m pip index versions
  orbitquant` reports `0.1.0`; a fresh venv installed `orbitquant==0.1.0` from
  PyPI and verified `orbitquant.__version__ == "0.1.0"`,
  `OrbitQuantConfig().runtime_mode == "auto_fused"`, and
  `orbitquant --version == "0.1.0"`.
  Patch publish run `29022356757` completed successfully on 2026-07-09 from
  head SHA `7d797baf6b41e2f67dca662d148173034f3738d9`, including full pytest,
  `ruff check`, package build, `twine check`, wheel smoke, OIDC publication,
  and PyPI digital attestations. The PyPI JSON API reports version `0.1.1` with
  `orbitquant-0.1.1.tar.gz` SHA256
  `5972b6cfd1d89653fb9ac17668f72818c61a0e3cc5ea1cdd46e59d54405dc1ff` and
  `orbitquant-0.1.1-py3-none-any.whl` SHA256
  `b829c5df00093e697872ca24104e6ef38dbe7e7d70b2c2e33560bfaec224cde1`; a fresh
  venv installed `orbitquant==0.1.1` from PyPI and verified
  `orbitquant.__version__ == "0.1.1"`,
  `OrbitQuantConfig().runtime_mode == "auto_fused"`,
  `OrbitQuantConfig().packed_matmul_block_n == 128`, and
  `orbitquant --version == "0.1.1"`.
  Patch publish run `29025225397` completed successfully on 2026-07-09 from
  head SHA `f18feb95f7595965f046b3455997d6ce9b8e8e4a`, including full pytest,
  `ruff check`, package build, `twine check`, wheel smoke, OIDC publication,
  and PyPI digital attestations. The PyPI JSON API reports version `0.1.2` with
  `orbitquant-0.1.2.tar.gz` SHA256
  `4afc15589fe713345dfa47be30dc078d943484a1f4bbd4861a413532a6ec8377` and
  `orbitquant-0.1.2-py3-none-any.whl` SHA256
  `3bdddebaa46f60307ed50e2ebf4b7ff4fef7817845b512ecfbb6fbf8ba71c91c`; a fresh
  venv installed `orbitquant==0.1.2` from PyPI and verified
  `orbitquant.__version__ == "0.1.2"`,
  `OrbitQuantConfig().runtime_mode == "auto_fused"`,
  `orbitquant --version == "0.1.2"`, and `orbitquant audit-hf-artifacts --help`
  includes `--summary-only`.
  Patch publish run `29034405916` completed successfully on 2026-07-09 from
  head SHA `02166ec6bdcd8920b0012a1ff930dce8dd976fdb`, including full pytest,
  `ruff check`, package build, `twine check`, wheel smoke, OIDC publication,
  and PyPI digital attestations. The PyPI JSON API reports version `0.1.3` with
  `orbitquant-0.1.3.tar.gz` SHA256
  `008f949641d00df46f840580c424b9e9fad0d853b4345d454a0b8042c61f3366` and
  `orbitquant-0.1.3-py3-none-any.whl` SHA256
  `181e9b532a07312ec47b54091d651b5a62d5aefd32c88d40830bd8529a0fdc53`; a fresh
  venv installed `orbitquant[kernels]==0.1.3` from PyPI on macOS and verified
  `orbitquant.__version__ == "0.1.3"`, installed `kernels==0.16.0`, and did
  not install Triton because the `triton>=3.5` extra is Linux-only.
  Patch publish run `29038063969` completed successfully on 2026-07-09 from
  head SHA `a7d28d96ff47d5ae72121bbefc3aab30ca732b42`, including full pytest,
  `ruff check`, package build, `twine check`, wheel smoke, OIDC publication,
  and PyPI digital attestations. The PyPI version-specific JSON API reports
  version `0.1.4` with `orbitquant-0.1.4.tar.gz` SHA256
  `7db4168bb2e9c3b838af0a932a96ca505d7bb91475239545d3e3a4c8130e07f4` and
  `orbitquant-0.1.4-py3-none-any.whl` SHA256
  `641a4d74a811023ff6cc39ae9680eb1393eb5e2b5dd390d586ff087ac1533af3`; a fresh
  venv installed `orbitquant==0.1.4` from PyPI with `uv pip install --refresh`
  and verified `orbitquant.__version__ == "0.1.4"` and
  `orbitquant --version == "0.1.4"`. A 2026-07-09T18:04Z fresh PyPI install
  smoke verified `orbitquant[hf]==0.1.4` imports OrbitQuant 0.1.4 with
  Diffusers 0.39.0, Transformers 5.13.0, Accelerate 1.14.0, and default
  `OrbitQuantConfig().runtime_mode == "auto_fused"`. The same smoke verified
  `orbitquant[kernels]==0.1.4` installs `kernels==0.16.0`, keeps
  `runtime_mode == "auto_fused"`, and does not install Triton on macOS because
  the `triton>=3.5` extra is Linux-only.
- [x] ComfyUI compatibility is verified after the relevant schema stabilizes,
  including load, graph execution, artifact metadata behavior, and kernel extra
  install guidance for the default `auto_fused` runtime.
  Evidence: ComfyUI-OrbitQuant commit `3f2ea7a` passed GitHub CI run
  `28977790874` on 2026-07-08T21:42Z. Local checks also passed
  `uv run pytest -q`, `uv run ruff check .`, package build, and
  `twine check`. The package smoke covers legacy node mappings,
  V3 entrypoint/schema/delegation, real OrbitQuant artifact load,
  inspector-to-loader node graph behavior, metadata propagation, and finite
  forward execution through the restored `OrbitQuantLinear`. The
  public-readiness commit `97d7efc` replaced the OrbitQuant dependency with the
  PyPI release constraint `orbitquant>=0.1.0`, switched the README clone command
  to HTTPS, restored the full Apache-2.0 license text, passed local `pytest`,
  `ruff`, package build, and `twine check`, and passed GitHub CI run
  `29020564152` on 2026-07-09. A live GitHub check then reported
  `iamwavecut/ComfyUI-OrbitQuant` as `PUBLIC`, Apache-2.0 licensed, with
  default branch `main`. Follow-up commit `85527ee` required
  `orbitquant>=0.1.1` after the packed-matmul default update and passed
  GitHub CI run `29022774661`. Commit `4832d4a` requires
  `orbitquant>=0.1.2`, updates the README install commands for both
  `orbitquant>=0.1.2` and `orbitquant[kernels]>=0.1.2`, refreshes `uv.lock`
  to the PyPI `orbitquant` 0.1.2 release hashes, and passed GitHub CI run
  `29027011708` on 2026-07-09. GitHub Release
  `https://github.com/iamwavecut/ComfyUI-OrbitQuant/releases/tag/v0.1.0` was
  published on 2026-07-09T15:12Z from commit
  `4832d4addb39681e82f38e07627a2bb682e4332d` with assets
  `comfyui_orbitquant-0.1.0.tar.gz` SHA256
  `630628c56e5ed35626cd7cca6749c51056cb78a0041b7c1268cf0b5e995d28c0` and
  `comfyui_orbitquant-0.1.0-py3-none-any.whl` SHA256
  `0c774c20a6759bea18d5d02b598035c3446a72ee4efaa0be6c1f325f4b3e928b`.
  Before release, local `uv run pytest -q`, `uv run ruff check .`, package
  build, and `twine check` passed for the node pack. Follow-up commit
  `ca62b7a` added the manual `.github/workflows/publish-pypi.yml` Trusted
  Publishing workflow for a future `comfyui-orbitquant` PyPI release. GitHub
  reported both `CI` and `Publish PyPI` workflows as active, and ComfyUI CI run
  `29028857599` passed on 2026-07-09 with node-pack tests, lint, package
  build, package metadata check, and wheel smoke. The GitHub `pypi`
  environment was created on 2026-07-09T15:21Z. After PyPI password
  confirmation, the pending publisher for `comfyui-orbitquant` was registered
  and workflow run `29029534070` published `comfyui-orbitquant==0.1.0` through
  Trusted Publishing on 2026-07-09. The PyPI version JSON for `0.1.0` reports
  `comfyui_orbitquant-0.1.0-py3-none-any.whl` SHA256
  `0c774c20a6759bea18d5d02b598035c3446a72ee4efaa0be6c1f325f4b3e928b` and
  `comfyui_orbitquant-0.1.0.tar.gz` SHA256
  `9a2c1e23e7c7674bb8fb7c350473c9ad636c952c5495b71d805a91996a00b51f`. A fresh
  venv installed `comfyui-orbitquant==0.1.0` from PyPI and imported
  `comfyui_orbitquant.NODE_CLASS_MAPPINGS`.
  Current commit `7e5fc4c` requires `orbitquant>=0.1.3`, exposes the
  `comfyui-orbitquant[kernels]` extra through `orbitquant[kernels]>=0.1.3`,
  refreshes `uv.lock` to the PyPI OrbitQuant 0.1.3 release, and passed GitHub
  CI run `29034705959` on 2026-07-09. Publish run `29034761024` published
  `comfyui-orbitquant==0.1.2` from head SHA
  `7e5fc4c0f74dbcb341b755b2b53cb0edf40cb311`. GitHub Release
  `https://github.com/iamwavecut/ComfyUI-OrbitQuant/releases/tag/v0.1.2` was
  published on 2026-07-09 with exact PyPI-matching assets:
  `comfyui_orbitquant-0.1.2.tar.gz` SHA256
  `48ea3909f96620baba3f2acf52532b15330fc89cb847557f13da915969f4e42b` and
  `comfyui_orbitquant-0.1.2-py3-none-any.whl` SHA256
  `2fbcf8a4fecb7d4384d2461001fa33b291a02720d20cf465e7e2669ad145fd83`. A fresh
  venv installed `comfyui-orbitquant[kernels]==0.1.2` from PyPI on macOS and
  verified `comfyui-orbitquant==0.1.2`, `orbitquant==0.1.3`,
  `kernels==0.16.0`, no Triton install, and exported FLUX, Z-Image, Wan,
  generic component-loader, and artifact-inspector nodes.
  Current commit `435baab` requires `orbitquant>=0.1.4`, exposes the
  `comfyui-orbitquant[kernels]` extra through `orbitquant[kernels]>=0.1.4`,
  refreshes `uv.lock` to the PyPI OrbitQuant 0.1.4 release, and passed GitHub
  CI run `29038618250` on 2026-07-09. Publish run `29038664476` published
  `comfyui-orbitquant==0.1.3` from head SHA
  `435baabea32f3f50bdf7ef4eee719e1fd4b82c12`. GitHub Release
  `https://github.com/iamwavecut/ComfyUI-OrbitQuant/releases/tag/v0.1.3` was
  published on 2026-07-09 with exact PyPI-matching assets:
  `comfyui_orbitquant-0.1.3.tar.gz` SHA256
  `202400246f267bfb4b97974207a5d482b71b80db9122cca60134f4355fc42f19` and
  `comfyui_orbitquant-0.1.3-py3-none-any.whl` SHA256
  `453c72d95e5dafee5408969225809ad278eb577a9386728b7f5bdcef918de1d1`. A fresh
  venv installed `comfyui-orbitquant==0.1.3` from PyPI and verified
  `comfyui-orbitquant==0.1.3`, `orbitquant==0.1.4`, and exported FLUX,
  Z-Image, Wan, generic component-loader, and artifact-inspector nodes. A
  2026-07-09T18:04Z fresh PyPI install smoke also verified
  `comfyui-orbitquant[kernels]==0.1.3` resolves to `orbitquant==0.1.4`,
  installs `kernels==0.16.0`, does not install Triton on macOS, and exports
  the same five ComfyUI node classes.
- [x] MPS shader-only gate can be run independently from the native packed
  matmul package gate. Evidence: on 2026-07-09,
  `ORBITQUANT_RUN_NATIVE_KERNEL_PACKAGE_CI=0` with tiny benchmark dimensions
  passed `scripts/run_mps_kernel_checks.sh`, including MPS/backend capability
  tests, `orbitquant kernel-info`, and an MPS `runtime_mode="dequant_bf16"`
  benchmark. The reported optimized stages were limited to
  `codebook_lookup_rescale,packed_weight_dequant`; native packed matmul stages
  were explicitly skipped.
- [x] Native run orchestration is ready for GPU execution without range smoke
  substitutions. Evidence: a 2026-07-09T18:21Z dry-run check invoked
  `uv run orbitquant native-plan` and `uv run orbitquant native-script` with
  `--runtime-mode auto_fused`, `--device cuda`, `--dtype bfloat16`,
  `--staging-mode streaming`, `--prompt-pack artifact`, and `--resume`. The
  generated native plan contained 14 jobs: W4A4/W3A3/W2A4/W2A3 for
  `flux2-native`, `flux1-schnell-native`, and `z-image-native`, plus
  W4A6/W4A4 for `wan-native`. The generated script included stage logging for
  preflight, kernel preflight, policy inventories, quantization, original and
  OrbitQuant native generation packs, artifact validation, and report
  generation. It contained 14 `orbitquant quantize` commands, 28
  `orbitquant generate-pack` commands, 42 `orbitquant validate-artifact`
  mentions including resume guards, 5 `orbitquant kernel-bench` commands, and
  4 `orbitquant inspect-policy` commands; the generated script contained no
  range-smoke path.
- [ ] Release-grade metrics are complete before making paper reproduction or
  metric-table claims. Image paper-target artifacts then include GenEval
  overall and per-task scores; Wan artifacts then include all required VBench
  dimensions. Missing release metrics block only those metric/reproduction
  claims; compact artifacts without those metrics must present native
  comparison status instead of paper-reproduction metric claims. A
  2026-07-09T18:12Z dry-run readiness check generated the external eval plan
  and script without running GenEval or VBench. It verified 20 metric jobs:
  8 `flux1-schnell-native` GenEval jobs, 8 `z-image-native` GenEval jobs, and
  4 `wan-native` VBench jobs. The job matrix covered BF16/original and
  OrbitQuant splits for W4A4/W3A3/W2A4/W2A3 image artifacts and W4A6/W4A4 Wan
  artifacts. It also verified the generated VBench commands pass all required
  dimensions as separate `--dimension` values:
  `imaging_quality`, `aesthetic_quality`, `motion_smoothness`,
  `dynamic_degree`, `background_consistency`, `subject_consistency`, `scene`,
  and `overall_consistency`.
