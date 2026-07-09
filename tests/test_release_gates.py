from pathlib import Path


def test_release_gates_document_final_acceptance_checklist():
    release_gates = Path("docs/release-gates.md").read_text(encoding="utf-8")

    assert "final acceptance gate" in release_gates
    assert "verification output" in release_gates
    assert "arXiv 2607.02461" in release_gates
    assert "`scripts/run_paper_methodology_checks.sh`" in release_gates
    assert "- [x] Final paper conformance audit" in release_gates
    assert "passed on 2026-07-08T15:49Z against arXiv 2607.02461v1" in release_gates
    assert "- [x] Release wording separates" in release_gates
    assert "paper-aligned subset" in release_gates
    assert "FLUX.2 Klein is\n  an additional target" in release_gates
    assert "README.md` separates paper-aligned artifacts from extra target" in (
        release_gates
    )
    assert "extra target; not an OrbitQuant paper reproduction model" in release_gates
    assert "`tests/test_readme.py` plus `tests/test_model_card.py`" in release_gates
    assert "`readme_mismatch_count=0`" in release_gates
    assert "- [x] Native artifact validation" in release_gates
    assert "native-resolution BF16-vs-OrbitQuant\n  comparison assets" in release_gates
    assert "finite-output checks" in release_gates
    assert "`native_smoke` proof\n  block in `benchmark/summary.json`" in release_gates
    assert "raw generation records remain local-only" in release_gates
    assert "audit-hf-artifacts --namespace WaveCut" in release_gates
    assert "--fail-on-artifact-regression` passed on 2026-07-09T11:44Z" in (
        release_gates
    )
    assert "14/14 artifact-ready" in release_gates
    assert "14/14 native-smoke\n  ready" in release_gates
    assert "zero remote checksum mismatches" in release_gates
    assert "zero\n  forbidden remote files" in release_gates
    assert "`public_count=14`" in release_gates
    assert "`private_count=0`" in release_gates
    assert "14/14 policy-inventory-ready" in release_gates
    assert "A 2026-07-09T15:06Z rerun with `--summary-only`" in release_gates
    assert "again passed `--fail-on-artifact-regression`" in release_gates
    assert (
        "14/14 artifact-ready, native-smoke-ready, metadata-complete, and\n"
        "  policy-inventory-ready"
    ) in release_gates
    assert "`release_eval_applicable_count=10`" in release_gates
    assert "`release_eval_ready_count=0`" in release_gates
    assert "`missing_required_metric_count=144`" in release_gates
    assert "paper reproduction or\n  metric-table claims" in release_gates
    assert "GenEval\n  overall and per-task scores" in release_gates
    assert "all required VBench\n  dimensions" in release_gates
    assert "Missing release metrics block only those metric/reproduction\n  claims" in release_gates
    assert "compact artifacts without those metrics must present native comparison" in (
        release_gates
    )
    assert "status instead of paper-reproduction metric claims" in release_gates
    assert "- [x] Full-model module classification inventories" in release_gates
    assert "Raw inventory JSON may\n  remain unpublished" in release_gates
    assert "`scripts/run_paper_methodology_checks.sh` produced and hash-checked" in (
        release_gates
    )
    assert "`policy_inventory_ready_count=14`" in release_gates
    assert "policy_inventory_error_count=0" in release_gates
    assert "`policy_inventory_ready_count=14`" in release_gates
    assert "144 missing\n  release-grade metrics" in release_gates
    assert "same compact artifact counts and still reported `forbidden_file_count=0`" in (
        release_gates
    )
    assert "`remote_checksum_mismatch_count=0`, and `readme_mismatch_count=0`" in (
        release_gates
    )
    assert "CUDA/Triton partial optimized" in release_gates
    assert "Metal/MPS partial optimized" in release_gates
    assert "CPU\n  reference-only" in release_gates
    assert "ROCm/XPU explicitly unsupported" in release_gates
    assert "[kernel-audit.md](kernel-audit.md)" in release_gates
    assert "MPS/Metal passed\n  `scripts/run_mps_kernel_checks.sh`" in release_gates
    assert "explicit `runtime_mode=\"native_packed_matmul\"`\n  benchmark execution" in (
        release_gates
    )
    assert "After adding kernel `upstream`/`source` metadata" in release_gates
    assert "2026-07-08T16:59Z at OrbitQuant commit `5cc7d30`" in release_gates
    assert "get-kernel\n  loading, and package tests" in release_gates
    assert "reviewable source\n  package from `native-kernels/orbitquant-packed-matmul`" in (
        release_gates
    )
    assert "without generated\n  `build/`, local `.venv/`, `__pycache__/`" in release_gates
    assert "loadable Kernel Hub\n  artifact must be a `kernel`-type repository" in (
        release_gates
    )
    assert "kernels-community/README/discussions/new" in release_gates
    assert "2026-07-08T17:02Z" in release_gates
    assert "`nix --option sandbox relaxed run .#build-and-copy -L`" in release_gates
    assert "passed locally and copied 3 Metal build variants" in release_gates
    assert "`nix --option sandbox relaxed run .#build-and-upload -L` found those 3" in (
        release_gates
    )
    assert "stopped only at the Hugging Face permission error" in release_gates
    assert "kernels-community/README/discussions/15" in release_gates
    assert "Approval remains pending" in release_gates
    assert "2026-07-08T18:12Z at OrbitQuant commit `956842a`" in release_gates
    assert "still stopped at the same\n  Kernel Hub publish permission error" in (
        release_gates
    )
    assert "2026-07-09T11:54Z" in release_gates
    assert "source\n  snapshot repo is public" in release_gates
    assert "`cb0ceb1a4d070556c52cfba691aba3f6647c246b`" in release_gates
    assert "`packed_weight_path_vs_materialized_weight_ratio`" in release_gates
    assert "PyPI `orbitquant-0.1.0.tar.gz` source distribution" in release_gates
    assert (
        "source\n  snapshot plus all 14 canonical OrbitQuant model artifact repos "
        "reported\n  `private_count=0`"
    ) in (
        release_gates
    )
    assert "follow-up comment was posted to discussion 15 on 2026-07-09T11:56Z" in (
        release_gates
    )
    assert "follow-up comment was posted on 2026-07-09T12:22Z" in release_gates
    assert "correctness and memory-path evidence only" in release_gates
    assert "matching\n  `torch212-metal-aarch64-darwin` variant" in release_gates
    assert "W4 512x1024x1024 float16" in release_gates
    assert "0.00764581459807232" in release_gates
    assert "W4\n  512x3072x3072 float16" in release_gates
    assert "0.10189520000712946" in release_gates
    assert "2026-07-08T17:10Z" in release_gates
    assert "`LOCAL_KERNELS`; with Torch 2.12.1 it selected" in release_gates
    assert "`build/torch212-metal-aarch64-darwin`" in release_gates
    assert "finite float16 output tensor" in release_gates
    assert "CUDA/Triton partial gate passed on 2026-07-08T19:31Z" in release_gates
    assert "OrbitQuant commit\n  `301d836`" in release_gates
    assert "Torch 2.9.1+cu128" in release_gates
    assert "ORBITQUANT_RUN_NATIVE_KERNEL_PACKAGE_CI=0" in release_gates
    assert "public-package CUDA smoke\n  passed on 2026-07-09T13:15Z" in (
        release_gates
    )
    assert "`orbitquant[kernels]==0.1.0` from PyPI" in release_gates
    assert "`forward_prewarmed_ms=0.14182400703430176`" in release_gates
    assert "published PyPI package\n  CUDA/Triton `auto_fused` path" in (
        release_gates
    )
    assert "model-like CUDA\n  microbenchmark was run" in release_gates
    assert "`tokens=512`, `in_features=3072`" in release_gates
    assert "`forward_prewarmed_ms=0.6518784046173096`" in release_gates
    assert "`forward_prewarmed_ms=0.13742079734802246`" in release_gates
    assert "not a throughput win on this RTX 4090\n  microbenchmark" in (
        release_gates
    )
    assert "follow-up tile sweep on the same RTX 4090" in release_gates
    assert "`block_n=128` default" in release_gates
    assert "`forward_prewarmed_ms=0.6374400138854981`" in release_gates
    assert "`forward_prewarmed_ms=0.596992015838623`" in release_gates
    assert "`peak_memory_bytes=69293568`" in release_gates
    assert "post-publication\n  CUDA smoke for `orbitquant[kernels]==0.1.1`" in (
        release_gates
    )
    assert "`packed_matmul_block_n == 128`" in release_gates
    assert "`forward_prewarmed_ms=0.5946400165557861`" in release_gates
    assert "`forward_prewarmed_ms=0.12743680477142333`" in release_gates
    assert "45,731,840 peak-memory\n  bytes" in release_gates
    assert "4.666x slower than\n  `dequant_bf16`" in release_gates
    assert "CUDA\n  artifact-layer verification on the same active RunPod RTX 4090" in (
        release_gates
    )
    assert "published\n  `WaveCut/FLUX.2-klein-4B-OrbitQuant-W4A4` artifact" in (
        release_gates
    )
    assert "`activation_kernel_backend=\"triton_cuda\"`" in release_gates
    assert "`max_abs_error_vs_dequant_bf16=0.015625`" in release_gates
    assert "`auto_fused_forward_prewarmed_ms=0.6343167781829834`" in (
        release_gates
    )
    assert "`dequant_bf16_forward_prewarmed_ms=0.1256432056427002`" in (
        release_gates
    )
    assert "actual published model artifact layer evidence" in release_gates
    assert "posted to discussion 15 on 2026-07-09T14:00Z" in release_gates
    assert "Native CUDA\n  `native_packed_matmul` remains open" in release_gates
    assert "`ImportError: libcudart.so.13`" in release_gates
    assert "`torch211-cxx11-cu128-x86_64-linux`" in release_gates
    assert "`torch29-cxx11-cu128-x86_64-linux`" in release_gates
    assert "runtime with an exported compatible variant" in release_gates
    assert (
        "latest published releases and dev\n  branches of Diffusers and Transformers"
        in release_gates
    )
    assert "- [x] Compatibility is verified" in release_gates
    assert "passed on 2026-07-09T17:18Z" in release_gates
    assert "Diffusers 0.40.0.dev0" in release_gates
    assert "`208704a`" in release_gates
    assert "Transformers 5.14.0.dev0" in release_gates
    assert "`0bc3554`" in release_gates
    assert "- [x] ComfyUI compatibility is verified" in release_gates
    assert "kernel extra\n  install guidance for the default `auto_fused` runtime" in (
        release_gates
    )
    assert "ComfyUI-OrbitQuant commit `3f2ea7a`" in release_gates
    assert "GitHub CI run\n  `28977790874`" in release_gates
    assert "public-readiness commit `97d7efc`" in release_gates
    assert "`orbitquant>=0.1.0`" in release_gates
    assert "passed GitHub CI run\n  `29020564152`" in release_gates
    assert "Follow-up commit `85527ee` required\n  `orbitquant>=0.1.1`" in (
        release_gates
    )
    assert "GitHub CI run `29022774661`" in release_gates
    assert "Commit `4832d4a` requires\n  `orbitquant>=0.1.2`" in (
        release_gates
    )
    assert "`orbitquant[kernels]>=0.1.2`" in release_gates
    assert "refreshes `uv.lock`\n  to the PyPI `orbitquant` 0.1.2 release hashes" in (
        release_gates
    )
    assert "passed GitHub CI run\n  `29027011708`" in release_gates
    assert "Current commit `7e5fc4c` requires `orbitquant>=0.1.3`" in (
        release_gates
    )
    assert "`comfyui-orbitquant[kernels]` extra through `orbitquant[kernels]>=0.1.3`" in (
        release_gates
    )
    assert "CI run `29034705959`" in release_gates
    assert "Publish run `29034761024` published\n  `comfyui-orbitquant==0.1.2`" in (
        release_gates
    )
    assert "7e5fc4c0f74dbcb341b755b2b53cb0edf40cb311" in release_gates
    assert "https://github.com/iamwavecut/ComfyUI-OrbitQuant/releases/tag/v0.1.2" in (
        release_gates
    )
    assert "`comfyui_orbitquant-0.1.2.tar.gz` SHA256" in release_gates
    assert "48ea3909f96620baba3f2acf52532b15330fc89cb847557f13da915969f4e42b" in (
        release_gates
    )
    assert "`comfyui_orbitquant-0.1.2-py3-none-any.whl` SHA256" in (
        release_gates
    )
    assert "2fbcf8a4fecb7d4384d2461001fa33b291a02720d20cf465e7e2669ad145fd83" in (
        release_gates
    )
    assert "`comfyui-orbitquant==0.1.2`, `orbitquant==0.1.3`" in release_gates
    assert "generic component-loader, and artifact-inspector nodes" in release_gates
    assert "`iamwavecut/ComfyUI-OrbitQuant` as `PUBLIC`" in release_gates
    assert "`uv run pytest -q`, `uv run ruff check .`, package build, and\n  `twine check`" in (
        release_gates
    )
    assert "legacy node mappings" in release_gates
    assert "V3 entrypoint/schema/delegation" in release_gates
    assert "inspector-to-loader node graph behavior" in release_gates
    assert "finite\n  forward execution through the restored `OrbitQuantLinear`" in (
        release_gates
    )
    assert "- [x] Checkpoint and model repositories are published" in release_gates
    assert "artifact-focused\n  model cards" in release_gates
    assert "native\n  comparison assets" in release_gates
    assert "`metadata_complete_ready`" in release_gates
    assert (
        "quantization device, weight quantization backend, and\n  staging mode provenance"
        in release_gates
    )
    assert "`repo_count=14`" in release_gates
    assert "`artifact_ready_count=14`" in release_gates
    assert "`metadata_complete_ready_count=14`" in release_gates
    assert "`native_smoke_ready_count=14`" in release_gates
    assert "`remote_checksum_mismatch_count=0`" in release_gates
    assert "`forbidden_file_count=0`" in release_gates
    assert "`public_count=14`" in release_gates
    assert "`private_count=0`" in release_gates
    assert "all 14 public artifact repos" in release_gates
    assert "all 14 public artifact manifests" in release_gates
    assert "not host logs, raw eval dumps, or terminal transcripts" in release_gates
    assert "- [x] The GitHub repository is public, tagged, released" in release_gates
    assert "[release-0.1.0.md](release-0.1.0.md)" in release_gates
    assert "[publication-checklist.md](publication-checklist.md)" in release_gates
    assert "release-readiness\n  commit `0c0f63a` passed as run `29016554734`" in (
        release_gates
    )
    assert "`iamwavecut/OrbitQuant` as `PUBLIC`" in release_gates
    assert "https://pypi.org/project/orbitquant/" in release_gates
    assert "Git tag `v0.1.0` was created" in release_gates
    assert "ce5c232a8bf9b450c7d94eeae07445317c98b1d0" in release_gates
    assert "https://github.com/iamwavecut/OrbitQuant/releases/tag/v0.1.0" in (
        release_gates
    )
    assert "6abedb769b32c8d70f2763278e106346319d628d85ed7469549faa5020ab1a89" in (
        release_gates
    )
    assert "dfbfa80ff79132457b6918d69f8ae9d8961ea3d898487d105a0e74da906eeaaa" in (
        release_gates
    )
    assert "[release-0.1.1.md](release-0.1.1.md)" in release_gates
    assert "commit `7d797ba`\n  passed as run `29022202797`" in release_gates
    assert "https://github.com/iamwavecut/OrbitQuant/releases/tag/v0.1.1" in (
        release_gates
    )
    assert "`orbitquant-0.1.1.tar.gz` SHA256" in release_gates
    assert "5972b6cfd1d89653fb9ac17668f72818c61a0e3cc5ea1cdd46e59d54405dc1ff" in (
        release_gates
    )
    assert "`orbitquant-0.1.1-py3-none-any.whl` SHA256" in release_gates
    assert "b829c5df00093e697872ca24104e6ef38dbe7e7d70b2c2e33560bfaec224cde1" in (
        release_gates
    )
    assert "[release-0.1.2.md](release-0.1.2.md)" in release_gates
    assert "commit `f18feb9`\n  passed as run `29025114082`" in release_gates
    assert "tag `v0.1.2` points at commit" in release_gates
    assert "f18feb95f7595965f046b3455997d6ce9b8e8e4a" in release_gates
    assert "https://github.com/iamwavecut/OrbitQuant/releases/tag/v0.1.2" in (
        release_gates
    )
    assert "`orbitquant-0.1.2.tar.gz` SHA256" in release_gates
    assert "4afc15589fe713345dfa47be30dc078d943484a1f4bbd4861a413532a6ec8377" in (
        release_gates
    )
    assert "`orbitquant-0.1.2-py3-none-any.whl` SHA256" in release_gates
    assert "3bdddebaa46f60307ed50e2ebf4b7ff4fef7817845b512ecfbb6fbf8ba71c91c" in (
        release_gates
    )
    assert "[release-0.1.3.md](release-0.1.3.md)" in release_gates
    assert "commit `02166ec`\n  passed as run `29034308427`" in release_gates
    assert "tag `v0.1.3` points at commit" in release_gates
    assert "02166ec6bdcd8920b0012a1ff930dce8dd976fdb" in release_gates
    assert "https://github.com/iamwavecut/OrbitQuant/releases/tag/v0.1.3" in (
        release_gates
    )
    assert "`orbitquant-0.1.3.tar.gz` SHA256" in release_gates
    assert "008f949641d00df46f840580c424b9e9fad0d853b4345d454a0b8042c61f3366" in (
        release_gates
    )
    assert "`orbitquant-0.1.3-py3-none-any.whl` SHA256" in release_gates
    assert "181e9b532a07312ec47b54091d651b5a62d5aefd32c88d40830bd8529a0fdc53" in (
        release_gates
    )
    assert (
        "including HF\n  integration tests, full pytest, package build, "
        "`twine check`, and wheel"
    ) in (
        release_gates
    )
    assert "- [x] The PyPI package is published as `orbitquant==0.1.3`" in (
        release_gates
    )
    assert ".github/workflows/publish-pypi.yml" in release_gates
    assert "PyPI\n  pending publisher was registered" in release_gates
    assert "GitHub Actions run `29015072821` completed successfully" in release_gates
    assert "full pytest, `ruff check`, package build, `twine check`,\n  wheel smoke" in (
        release_gates
    )
    assert "PyPI digital attestations" in release_gates
    assert "`orbitquant-0.1.0.tar.gz`" in release_gates
    assert "`orbitquant-0.1.0-py3-none-any.whl`" in release_gates
    assert "Patch publish run `29022356757` completed successfully" in release_gates
    assert "head SHA `7d797baf6b41e2f67dca662d148173034f3738d9`" in release_gates
    assert "PyPI JSON API reports version `0.1.1`" in release_gates
    assert "`orbitquant-0.1.1.tar.gz` SHA256" in release_gates
    assert "`orbitquant-0.1.1-py3-none-any.whl` SHA256" in release_gates
    assert "Patch publish run `29025225397` completed successfully" in release_gates
    assert "PyPI JSON API reports version `0.1.2`" in release_gates
    assert "`orbitquant-0.1.2.tar.gz` SHA256" in release_gates
    assert "`orbitquant-0.1.2-py3-none-any.whl` SHA256" in release_gates
    assert "Patch publish run `29034405916` completed successfully" in release_gates
    assert "head SHA `02166ec6bdcd8920b0012a1ff930dce8dd976fdb`" in (
        release_gates
    )
    assert "PyPI JSON API reports version `0.1.3`" in release_gates
    assert "`orbitquant-0.1.3.tar.gz` SHA256" in release_gates
    assert "`orbitquant-0.1.3-py3-none-any.whl` SHA256" in release_gates
    assert "python -m pip index versions\n  orbitquant` reports `0.1.0`" in (
        release_gates
    )
    assert '`OrbitQuantConfig().runtime_mode == "auto_fused"`' in release_gates
    assert '`orbitquant --version == "0.1.0"`' in release_gates
    assert '`OrbitQuantConfig().packed_matmul_block_n == 128`' in release_gates
    assert '`orbitquant --version == "0.1.1"`' in release_gates
    assert '`orbitquant.__version__ == "0.1.2"`' in release_gates
    assert '`orbitquant --version == "0.1.2"`' in release_gates
    assert '`orbitquant.__version__ == "0.1.3"`' in release_gates
    assert "`kernels==0.16.0`" in release_gates
    assert "`triton>=3.5` extra is Linux-only" in release_gates
    assert "`orbitquant audit-hf-artifacts --help`\n  includes `--summary-only`" in (
        release_gates
    )
    assert "Upload remains pending" not in release_gates
    assert "command transcript" not in release_gates
    assert "local under ignored" not in release_gates
    assert "chronology" not in release_gates.lower()


def test_publication_checklist_contains_gated_release_commands():
    checklist = Path("docs/publication-checklist.md").read_text(encoding="utf-8")

    assert "OrbitQuant 0.1.0 Publication Checklist" in checklist
    assert "PyPI `orbitquant==0.1.0` is published" in checklist
    assert "the GitHub repository is public" in checklist
    assert "GitHub Release `v0.1.0` is\npublished" in checklist
    assert "Trusted Publishing is configured" in checklist
    assert "git status --short --branch" in checklist
    assert "git fetch origin main --tags" in checklist
    assert 'test "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)"' in checklist
    assert (
        'test "$(git rev-list -n 1 v0.1.0)" = '
        '"ce5c232a8bf9b450c7d94eeae07445317c98b1d0"'
    ) in checklist
    assert "gh repo view iamwavecut/OrbitQuant --json" in checklist
    assert "uv run pytest -q" in checklist
    assert "uv run ruff check ." in checklist
    assert "uv run --with build python -m build" in checklist
    assert "uv run --with twine python -m twine check dist/*" in checklist
    assert "uv run --with /Users/Shared/src/github.com/iamwavecut/OrbitQuant/dist/" in (
        checklist
    )
    assert "OrbitQuantConfig().runtime_mode" in checklist
    assert "https://pypi.org/pypi/orbitquant/json" in checklist
    assert "Repository visibility is already public" in checklist
    assert "tag points to the PyPI\npublication workflow head SHA" in checklist
    assert (
        'test "$(gh repo view iamwavecut/OrbitQuant --json isPrivate '
        '--jq .isPrivate)" = "false"'
    ) in checklist
    assert (
        "git tag -a v0.1.0 ce5c232a8bf9b450c7d94eeae07445317c98b1d0"
        in checklist
    )
    assert "git push origin v0.1.0" in checklist
    assert "gh release create v0.1.0" in checklist
    assert "--verify-tag" in checklist
    assert "--notes-file docs/release-0.1.0.md" in checklist
    assert "gh workflow run publish-pypi.yml" in checklist
    assert "gh run watch 29015072821" in checklist
    assert "orbitquant-0.1.0.tar.gz" in checklist
    assert "orbitquant-0.1.0-py3-none-any.whl" in checklist
    assert "gh release view v0.1.0" in checklist
    assert "python -m pip index versions orbitquant" in checklist
    assert "orbitquant-publication-ok" in checklist
    assert "RunPod" not in checklist
    assert "chronology" not in checklist.lower()


def test_release_gates_keep_current_priority_order():
    release_gates = Path("docs/release-gates.md").read_text(encoding="utf-8")

    expected_order = [
        "Kernel audit and benchmarks",
        "Final paper conformance audit",
        "Release wording separates",
        "Native artifact validation",
        "Full-model module classification inventories",
        "Compatibility is verified against",
        "Checkpoint and model repositories",
        "The GitHub repository is public",
        "The PyPI package is published",
        "ComfyUI compatibility is verified",
        "Release-grade metrics are complete",
    ]
    positions = [release_gates.index(token) for token in expected_order]

    assert positions == sorted(positions)


def test_release_notes_are_artifact_focused_and_reproducible():
    release_notes = Path("docs/release-0.1.0.md").read_text(encoding="utf-8")

    assert "OrbitQuant 0.1.0 Release Notes" in release_notes
    assert "Package Scope" in release_notes
    assert "Implemented Quantization" in release_notes
    assert 'runtime_mode="auto_fused"' in release_notes
    assert "runtime_mode=\"dequant_bf16\"" in release_notes
    assert 'pip install "orbitquant[kernels]"' in release_notes
    assert "Hugging Face `kernels` loader and Triton" in release_notes
    assert "CPU reference runs do not require\nthis extra" in release_notes
    assert "Paper-aligned targets" in release_notes
    assert "FLUX.1-schnell" in release_notes
    assert "Z-Image-Turbo" in release_notes
    assert "Wan2.1 T2V 1.3B" in release_notes
    assert "FLUX.2 Klein artifacts are not claimed" in release_notes
    assert "scripts/run_paper_methodology_checks.sh" in release_notes
    assert "scripts/run_hf_compat_checks.sh --mode all" in release_notes
    assert "audit-hf-artifacts" in release_notes
    assert "scripts/run_mps_kernel_checks.sh" in release_notes
    assert "scripts/run_cuda_kernel_checks.sh" in release_notes
    assert "GenEval and VBench numbers are release evidence" in release_notes
    assert "ROCm and XPU are not implemented" in release_notes
    assert "orbitquant-0.1.0.tar.gz" in release_notes
    assert "orbitquant-0.1.0-py3-none-any.whl" in release_notes
    assert "must not contain generated `build/`, `.venv/`, `__pycache__/`" in (
        release_notes
    )
    assert "WaveCut/orbitquant-packed-matmul" in release_notes
    assert "`LOCAL_KERNELS` must point\n" in release_notes
    assert "expected kernel-builder variant" in release_notes
    assert "RunPod" not in release_notes
    assert "discussion" not in release_notes.lower()
    assert "chronology" not in release_notes.lower()


def test_release_011_notes_include_pypi_cuda_smoke_without_speedup_claims():
    release_notes = Path("docs/release-0.1.1.md").read_text(encoding="utf-8")

    assert "OrbitQuant 0.1.1 Release Notes" in release_notes
    assert "`block_m=32`, `block_n=128`, `block_k=64`, `num_warps=8`" in (
        release_notes
    )
    assert "`orbitquant[kernels]==0.1.1` from PyPI" in release_notes
    assert "`runtime_mode=\"auto_fused\"`" in release_notes
    assert "`packed_matmul_block_n == 128`" in release_notes
    assert "selected\n  `triton_cuda`" in release_notes
    assert "finite benchmark output" in release_notes
    assert "does not claim full-model speedup" in release_notes


def test_release_012_notes_document_compact_audit_only():
    release_notes = Path("docs/release-0.1.2.md").read_text(encoding="utf-8")

    assert "OrbitQuant 0.1.2 Release Notes" in release_notes
    assert "`orbitquant audit-hf-artifacts` now accepts `--summary-only`" in (
        release_notes
    )
    assert "--fail-on-artifact-regression" in release_notes
    assert "does not weaken the artifact-readiness gate" in release_notes
    assert "does not change quantization math" in release_notes
    assert "runtime kernel\n" in release_notes
    assert "model cards" in release_notes
    assert "RunPod" not in release_notes
    assert "chronology" not in release_notes.lower()


def test_release_013_notes_document_platform_kernel_extra_fix_only():
    release_notes = Path("docs/release-0.1.3.md").read_text(encoding="utf-8")

    assert "OrbitQuant 0.1.3 Release Notes" in release_notes
    assert "cross-platform kernel extras" in release_notes
    assert 'pip install "orbitquant[kernels]==0.1.3"' in release_notes
    assert "`triton>=3.5` is now constrained to Linux" in release_notes
    assert "macOS/MPS\n  users" in release_notes
    assert "`orbitquant.__version__`" in release_notes
    assert "does not change quantization math" in release_notes
    assert "runtime dispatch policy" in release_notes
    assert "kernel implementations" in release_notes
    assert "performance" not in release_notes.lower()
    assert "speedup" not in release_notes.lower()
    assert "RunPod" not in release_notes
    assert "chronology" not in release_notes.lower()


def test_release_014_notes_document_external_metric_import_fixes_only():
    release_notes = Path("docs/release-0.1.4.md").read_text(encoding="utf-8")

    assert "OrbitQuant 0.1.4 Release Notes" in release_notes
    assert "external GenEval and VBench\nmetric import" in release_notes
    assert 'pip install "orbitquant[hf]==0.1.4"' in release_notes
    assert 'pip install "orbitquant[kernels]==0.1.4"' in release_notes
    assert "`geneval_overall`" in release_notes
    assert "average over task scores" in release_notes
    assert "`geneval_image_accuracy`" in release_notes
    assert "`geneval_prompt_accuracy`" in release_notes
    assert "VBench external eval commands" in release_notes
    assert "separate CLI\n  arguments" in release_notes
    assert "exported video filenames" in release_notes
    assert "does not change quantization math" in release_notes
    assert "runtime\ndispatch" in release_notes
    assert "Release-grade\nGenEval/VBench runs remain separate" in release_notes
    assert "RunPod" not in release_notes
    assert "chronology" not in release_notes.lower()


def test_kernel_audit_documents_backend_claim_boundaries():
    kernel_audit = Path("docs/kernel-audit.md").read_text(encoding="utf-8")

    assert "CPU | Reference-only" in kernel_audit
    assert "MPS/Metal | Partial optimized" in kernel_audit
    assert "CUDA/Triton | Partial optimized" in kernel_audit
    assert "ROCm | Unsupported" in kernel_audit
    assert "XPU | Unsupported" in kernel_audit
    assert "`auto_fused` prefers native packed matmul then Triton packed matmul" in (
        kernel_audit
    )
    assert "full activation-plus-matmul fusion" in kernel_audit
    assert "`claim_status` values" in kernel_audit
    assert "not itself a Hugging Face Kernels Hub `kernel-builder` package" in kernel_audit
    assert "kernel-builder CI" in kernel_audit
    assert "scripts/runpod_ssh_health.sh" in kernel_audit
    assert "ssh -F /dev/null -tt" in kernel_audit
    assert "ignoring local SSH config and ControlMaster state" in kernel_audit
    assert "scripts/verify_hf_kernel_model_artifact.py" in kernel_audit
    assert "intentionally avoids full Diffusers pipeline" in kernel_audit
    assert "`native_packed_matmul` runtime uses the separate" in kernel_audit
    assert "targets CUDA and Metal" in kernel_audit
    assert "Current Verification Evidence" in kernel_audit
    assert "scripts/run_mps_kernel_checks.sh" in kernel_audit
    assert "native `WaveCut/orbitquant-packed-matmul`\n  loading" in kernel_audit
    assert "explicit `runtime_mode=\"native_packed_matmul\"`\n  benchmark execution" in (
        kernel_audit
    )
    assert "2026-07-08T16:59Z" in kernel_audit
    assert "cb0ceb1a4d070556c52cfba691aba3f6647c246b" in kernel_audit
    assert "PyPI `orbitquant-0.1.0.tar.gz` source distribution" in kernel_audit
    assert "6abedb769b32c8d70f2763278e106346319d628d85ed7469549faa5020ab1a89" in (
        kernel_audit
    )
    assert "6821e4cd5ff1894994d7137c1d861660cfeed1c8" in kernel_audit
    assert "kernels-community/README/discussions/15" in kernel_audit
    assert "follow-up comment on 2026-07-08T18:03Z" in kernel_audit
    assert "second follow-up comment on 2026-07-09T11:56Z" in kernel_audit
    assert "third follow-up comment on 2026-07-09T12:22Z" in kernel_audit
    assert "`predequantized_f_linear_seconds_per_iter`" in kernel_audit
    assert "`dequantize_then_f_linear_seconds_per_iter`" in kernel_audit
    assert "`0.045x` versus dequantize-then-F.linear" in kernel_audit
    assert "`0.044x` versus dequantize-then-F.linear" in kernel_audit
    assert "fourth follow-up comment on 2026-07-09T12:27Z" in kernel_audit
    assert "OrbitQuant-converted diffusion\n  transformer backbones" in kernel_audit
    assert "not a drop-in kernel for arbitrary unquantized models" in kernel_audit
    assert "updated again on 2026-07-09T12:39Z" in kernel_audit
    assert "`packed_weight_path_bytes`" in kernel_audit
    assert "`packed_weight_path_vs_materialized_weight_ratio`" in kernel_audit
    assert "fifth follow-up comment on 2026-07-09T12:41Z" in kernel_audit
    assert "After reviewer asked for model-level verification scripts" in kernel_audit
    assert "WaveCut/FLUX.2-klein-4B-OrbitQuant-W4A4" in kernel_audit
    assert "sixth follow-up comment on 2026-07-09T12:50Z" in kernel_audit
    assert "f42d2dc19897adde62ec3ebb33e4ce748255dd54" in kernel_audit
    assert "On 2026-07-09T12:57Z" in kernel_audit
    assert "`max_abs_error_vs_dequant_bf16=0.001953125`" in kernel_audit
    assert "`packed_weight_path_vs_materialized_weight_ratio=0.2503289116753472`" in (
        kernel_audit
    )
    assert "seventh follow-up comment on 2026-07-09T12:58Z" in kernel_audit
    assert "On 2026-07-09T13:05Z" in kernel_audit
    assert "`runtime_mode=\"auto_fused\"`" in kernel_audit
    assert "default optimized dispatch reaches the native packed matmul" in (
        kernel_audit
    )
    assert "eighth follow-up comment on 2026-07-09T13:05Z" in kernel_audit
    assert "2026-07-08T18:12Z at OrbitQuant commit `956842a`" in kernel_audit
    assert "still stopped at the same\n  Kernel Hub publish permission error" in (
        kernel_audit
    )
    assert "W4 512x1024x1024 float16" in kernel_audit
    assert "0.10189520000712946" in kernel_audit
    assert "build/torch212-metal-aarch64-darwin" in kernel_audit
    assert "finite float16 output tensor" in kernel_audit
    assert "[kernel-hub-approval-request.md]" in kernel_audit
    assert "CUDA/Triton partial gate passed on 2026-07-08T19:31Z" in kernel_audit
    assert "Torch 2.9.1+cu128" in kernel_audit
    assert "ORBITQUANT_RUN_NATIVE_KERNEL_PACKAGE_CI=0" in kernel_audit
    assert "public-package CUDA smoke passed on 2026-07-09T13:15Z" in (
        kernel_audit
    )
    assert "`orbitquant[kernels]==0.1.0` from PyPI" in kernel_audit
    assert "`forward_prewarmed_ms=0.14182400703430176`" in kernel_audit
    assert "published PyPI package\n  CUDA/Triton `auto_fused` path" in (
        kernel_audit
    )
    assert "model-like CUDA microbenchmark\n  was run" in kernel_audit
    assert "`tokens=512`, `in_features=3072`" in kernel_audit
    assert "`forward_prewarmed_ms=0.6518784046173096`" in kernel_audit
    assert "`forward_prewarmed_ms=0.13742079734802246`" in kernel_audit
    assert "not a throughput win on this RTX 4090\n  microbenchmark" in (
        kernel_audit
    )
    assert "follow-up tile sweep on the same RTX 4090" in kernel_audit
    assert "`block_n=128` default" in kernel_audit
    assert "`forward_prewarmed_ms=0.6374400138854981`" in kernel_audit
    assert "`forward_prewarmed_ms=0.596992015838623`" in kernel_audit
    assert "`peak_memory_bytes=69293568`" in kernel_audit
    assert "post-publication CUDA smoke for `orbitquant[kernels]==0.1.1`" in (
        kernel_audit
    )
    assert "`packed_matmul_block_n == 128`" in kernel_audit
    assert "`forward_prewarmed_ms=0.5946400165557861`" in kernel_audit
    assert "`forward_prewarmed_ms=0.12743680477142333`" in kernel_audit
    assert "favors `dequant_bf16` for throughput by about 4.666x" in kernel_audit
    assert "After reviewer asked to run on an actual model" in kernel_audit
    assert "CUDA artifact-layer\n  verification passed" in kernel_audit
    assert "installed OrbitQuant from GitHub `main`" in kernel_audit
    assert "512 float16 tokens" in kernel_audit
    assert "`activation_kernel_backend=\"triton_cuda\"`" in kernel_audit
    assert "`max_abs_error_vs_dequant_bf16=0.015625`" in kernel_audit
    assert "`auto_fused_forward_first_ms=477.2464599609375`" in kernel_audit
    assert "`auto_fused_forward_prewarmed_ms=0.6343167781829834`" in (
        kernel_audit
    )
    assert "`dequant_bf16_forward_prewarmed_ms=0.1256432056427002`" in (
        kernel_audit
    )
    assert "actual\n  published artifact layer execution" in kernel_audit
    assert "posted to\n  discussion 15 on 2026-07-09T14:00Z" in kernel_audit
    assert "Native CUDA `native_packed_matmul` still needs" in kernel_audit
    assert "`ImportError: libcudart.so.13`" in kernel_audit
    assert "`torch211-cxx11-cu128-x86_64-linux`" in kernel_audit
    assert "`torch29-cxx11-cu128-x86_64-linux`" in kernel_audit
    assert "Torch 2.11+cu128" in kernel_audit
    assert "a4d927c" not in kernel_audit


def test_kernel_hub_approval_request_contains_required_review_fields():
    request = Path("docs/kernel-hub-approval-request.md").read_text(encoding="utf-8")

    assert "Request Kernel Hub publish access" in request
    assert "kernels-community/README" in request
    assert "huggingface.co/spaces/kernels-community/README/discussions/new" in request
    assert "huggingface.co/spaces/kernels-community/README/discussions/15" in request
    assert "Follow-up comment" in request
    assert "On 2026-07-08T18:03Z" in request
    assert "Reviewer follow-up" in request
    assert "On 2026-07-09T07:29Z" in request
    assert "`sayakpaul` reported" in request
    assert "On 2026-07-09T09:59Z" in request
    assert "without changing repository visibility" in request
    assert "public source-only snapshot repo or a source archive" in request
    assert "MPS native\npacked matmul smoke benchmark numbers" in request
    assert "native CUDA package numbers pending" in request
    assert "Source visibility follow-up" in request
    assert "As of 2026-07-09T11:54Z" in request
    assert "is public as a\nsource snapshot repo" in request
    assert "c34d9851cde2cf098589927a7b0bed85d65426af" in request
    assert "public PyPI source distribution" in request
    assert "On 2026-07-09T11:56Z" in request
    assert "posted a follow-up comment in discussion 15" in request
    assert "On 2026-07-09T12:22Z" in request
    assert "updating the benchmark source" in request
    assert "`0.045x` versus" in request
    assert "`0.044x`" in request
    assert "On 2026-07-09T12:27Z" in request
    assert "answered the model-scope question" in request
    assert "OrbitQuant-converted\n" in request
    assert "FLUX.1-schnell" in request
    assert "FLUX.2 Klein" in request
    assert "Z-Image-Turbo" in request
    assert "Wan2.1-T2V-1.3B-Diffusers" in request
    assert "On 2026-07-09T12:39Z" in request
    assert "packed weight storage fields" in request
    assert "`packed_weight_path_vs_materialized_weight_ratio`" in request
    assert "On 2026-07-09T12:41Z" in request
    assert "the new fields are weight-side storage accounting" in request
    assert "On 2026-07-09T12:42Z" in request
    assert "asked for a way to try optimizing one target\nmodel" in request
    assert "scripts/verify_hf_kernel_model_artifact.py" in request
    assert "On 2026-07-09T12:50Z" in request
    assert "MPS and CUDA `LOCAL_KERNELS`\nexample commands" in request
    assert "On 2026-07-09T12:57Z" in request
    assert "`transformer_blocks.0.attn.to_q` (3072x3072)" in request
    assert "`allclose_to_dequant_bf16=true`" in request
    assert "On 2026-07-09T12:58Z" in request
    assert "posted that verifier command and JSON result\nsummary" in request
    assert "On 2026-07-09T13:05Z" in request
    assert "`runtime_mode=\"auto_fused\"`" in request
    assert "default optimized dispatch path" in request
    assert "posted that `auto_fused` verifier command and\nJSON result summary" in (
        request
    )
    assert "On 2026-07-09T13:08Z" in request
    assert "asked for actual reported numbers" in request
    assert "On 2026-07-09T13:22Z" in request
    assert "`forward_prewarmed_ms=0.6518784046173096`" in request
    assert "`forward_prewarmed_ms=0.13742079734802246`" in request
    assert "memory-path evidence, not a throughput win" in request
    assert "On 2026-07-09T13:54Z" in request
    assert "asked to run on an actual model" in request
    assert "published\n`WaveCut/FLUX.2-klein-4B-OrbitQuant-W4A4` artifact" in request
    assert "`activation_kernel_backend=\"triton_cuda\"`" in request
    assert "`max_abs_error_vs_dequant_bf16=0.015625`" in request
    assert "`peak_memory_bytes=87756800`" in request
    assert "`auto_fused_forward_prewarmed_ms=0.6343167781829834`" in request
    assert "actual\npublished model artifact layer execution" in request
    assert "On 2026-07-09T14:00Z" in request
    assert "posted those actual model artifact numbers" in request
    assert "6abedb769b32c8d70f2763278e106346319d628d85ed7469549faa5020ab1a89" in (
        request
    )
    assert "locally prepared and checked on 2026-07-08T18:00Z" in request
    assert "77aef6caa1bbdbbd77e2cbf5003423073e001191d008473c957795d7bed03651" in (
        request
    )
    assert "21 tar entries" in request
    assert "binary `.so`, or benchmark output files" in request
    assert "WaveCut/orbitquant-packed-matmul" in request
    assert "native-kernels/orbitquant-packed-matmul" in request
    assert "https://github.com/iamwavecut/OrbitQuant" in request
    assert "Review source snapshot:" in request
    assert "cb0ceb1a4d070556c52cfba691aba3f6647c246b" in request
    assert "f7eb3fa912caa27ad682c7ea1757f580a2751a01" not in request
    assert "Apache-2.0" in request
    assert "Review-ready source package" in request
    assert "huggingface.co/WaveCut/orbitquant-packed-matmul/commit" in request
    assert "git archive --format=tar" in request
    assert "HEAD:native-kernels/orbitquant-packed-matmul" in request
    assert "orbitquant-packed-matmul-source.tar" in request
    assert "torch-ext/torch_binding.cpp" in request
    assert "orbitquant_packed_matmul_cuda/packed_matmul.cu" in request
    assert "orbitquant_packed_matmul_metal/packed_matmul.metal" in request
    assert "Do not attach generated `build/`, local `.venv/`, `__pycache__/`" in (
        request
    )
    assert "packed low-bit matrix multiplication" in request
    assert "Hugging Face Diffusers" in request
    assert "Hugging Face Transformers" in request
    assert "ComfyUI-OrbitQuant" in request
    assert "TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops)" in request
    assert "Python ABI 3.9" in request
    assert "`build-and-copy` currently builds 3 local Metal variants" in request
    assert "publish access is needed" in request
    assert "torch212-metal-aarch64-darwin" in request
    assert "0.00764581459807232" in request
    assert "0.10189520000712946" in request
    assert "MPS native packed matmul is currently not throughput-competitive" in request
    assert "CUDA native package benchmark evidence is pending" in request
    assert "uploaded as a `kernel`-type repository" in request
    assert "trust_remote_code=True" in request


def test_paper_methodology_audit_uses_claim_boundary_language():
    audit = Path("docs/paper-methodology-audit.md").read_text(encoding="utf-8")

    assert "Verification: `scripts/run_paper_methodology_checks.sh` passed" in audit
    assert "2026-07-08T15:49Z against arXiv 2607.02461v1" in audit
    assert "release-grade GenEval/VBench metric claims remain separate" in audit
    assert "Native artifact readiness is separate from full GenEval or VBench scoring" in (
        audit
    )
    assert "Pending Evidence For Acceleration Claims" in audit
    assert "Raw inventory JSON is audit evidence" in audit
    assert "not a development log" in audit
    for forbidden in (
        "Full GenEval/VBench is optional during development",
        "The current development path",
        "Known kernel follow-up",
        "development blocker",
        "local ignored audit artifacts",
    ):
        assert forbidden not in audit
