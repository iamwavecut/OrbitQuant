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
    assert "--fail-on-artifact-regression` passed on 2026-07-08T16:00Z" in (
        release_gates
    )
    assert "14/14 artifact-ready" in release_gates
    assert "14/14 native-smoke\n  ready" in release_gates
    assert "zero remote checksum mismatches" in release_gates
    assert "zero\n  forbidden remote files" in release_gates
    assert "re-run on 2026-07-08T17:27Z" in release_gates
    assert "14/14 policy-inventory-ready" in release_gates
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
    assert "policy_inventory_ready=14" in release_gates
    assert "policy_inventory_error_count=0" in release_gates
    assert "`policy_inventory_ready_count=14`" in release_gates
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
    assert "get-kernel\n  loading, and 17 package tests" in release_gates
    assert "reviewable source package from `native-kernels/orbitquant-packed-matmul`" in (
        release_gates
    )
    assert "without generated `build/`, local `.venv/`, `__pycache__/`" in release_gates
    assert "loadable Kernel Hub artifact must be a `kernel`-type repository" in (
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
    assert "approval remains pending" in release_gates
    assert "2026-07-08T17:15Z" in release_gates
    assert "source snapshot repo is still private" in release_gates
    assert "tracked source archive" in release_gates
    assert "source-only kernel repo\n  public" in release_gates
    assert "matching\n  `torch212-metal-aarch64-darwin` variant" in release_gates
    assert "W4 512x1024x1024 float16" in release_gates
    assert "0.00764581459807232" in release_gates
    assert "W4\n  512x3072x3072 float16" in release_gates
    assert "0.10189520000712946" in release_gates
    assert "2026-07-08T17:10Z" in release_gates
    assert "`LOCAL_KERNELS`; with Torch 2.12.1 it selected" in release_gates
    assert "`build/torch212-metal-aarch64-darwin`" in release_gates
    assert "finite float16 output tensor" in release_gates
    assert "CUDA/Triton remains pending on a CUDA host" in release_gates
    assert (
        "latest published releases and dev\n  branches of Diffusers and Transformers"
        in release_gates
    )
    assert "- [x] Compatibility is verified" in release_gates
    assert "passed on 2026-07-08T15:47Z" in release_gates
    assert "Diffusers 0.40.0.dev0" in release_gates
    assert "Transformers 5.14.0.dev0" in release_gates
    assert "- [x] ComfyUI compatibility is verified" in release_gates
    assert "ComfyUI-OrbitQuant commit `1d73b36`" in release_gates
    assert "passed `uv run pytest -q`\n  and `uv run ruff check .`" in release_gates
    assert "legacy node mappings" in release_gates
    assert "V3 entrypoint/schema/delegation" in release_gates
    assert "inspector-to-loader node graph behavior" in release_gates
    assert "finite forward execution through the restored `OrbitQuantLinear`" in (
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
    assert "zero-regression counts" in release_gates
    assert "not host logs, raw eval dumps, or terminal transcripts" in release_gates
    assert "The GitHub repository is public" in release_gates
    assert "[release-0.1.0.md](release-0.1.0.md)" in release_gates
    assert "repository visibility and\n  the release tag remain pending explicit approval" in (
        release_gates
    )
    assert "python -m build" in release_gates
    assert "python -m twine check dist/*" in release_gates
    assert "python -m twine upload\n  dist/*" in release_gates
    assert "PyPI token or browser\n  action" in release_gates
    assert "local build/check/smoke passed on 2026-07-08T16:06Z" in release_gates
    assert "`orbitquant-0.1.0.tar.gz`" in release_gates
    assert "`orbitquant-0.1.0-py3-none-any.whl`" in release_gates
    assert "returned `0.1.0`" in release_gates
    assert "Re-checked on 2026-07-08T17:27Z" in release_gates
    assert "using a temporary build output directory" in release_gates
    assert "`OrbitQuantConfig()` defaulted\n  to `runtime_mode=\"auto_fused\"`" in (
        release_gates
    )
    assert "Upload remains pending" in release_gates
    assert "command transcript" not in release_gates
    assert "local under ignored" not in release_gates
    assert "chronology" not in release_gates.lower()


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
        "The PyPI package is built",
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
    assert "RunPod" not in release_notes
    assert "discussion" not in release_notes.lower()
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
    assert "`native_packed_matmul` runtime uses the separate" in kernel_audit
    assert "targets CUDA and Metal" in kernel_audit
    assert "Current Verification Evidence" in kernel_audit
    assert "scripts/run_mps_kernel_checks.sh" in kernel_audit
    assert "native `WaveCut/orbitquant-packed-matmul`\n  loading" in kernel_audit
    assert "explicit `runtime_mode=\"native_packed_matmul\"`\n  benchmark execution" in (
        kernel_audit
    )
    assert "2026-07-08T16:59Z" in kernel_audit
    assert "062b934389dce9242e0a9185ed469cc3170e3e73" in kernel_audit
    assert "kernels-community/README/discussions/15" in kernel_audit
    assert "W4 512x1024x1024 float16" in kernel_audit
    assert "0.10189520000712946" in kernel_audit
    assert "build/torch212-metal-aarch64-darwin" in kernel_audit
    assert "finite float16 output tensor" in kernel_audit
    assert "[kernel-hub-approval-request.md]" in kernel_audit
    assert "CUDA/Triton must still be verified on a CUDA host" in kernel_audit
    assert "a4d927c" not in kernel_audit


def test_kernel_hub_approval_request_contains_required_review_fields():
    request = Path("docs/kernel-hub-approval-request.md").read_text(encoding="utf-8")

    assert "Request Kernel Hub publish access" in request
    assert "kernels-community/README" in request
    assert "huggingface.co/spaces/kernels-community/README/discussions/new" in request
    assert "huggingface.co/spaces/kernels-community/README/discussions/15" in request
    assert "Source visibility follow-up" in request
    assert "still a private\nsource snapshot repo" in request
    assert "make only this source-only kernel repo public" in request
    assert "locally prepared and checked on 2026-07-08T17:39Z" in request
    assert "9dcd6896a4d9e259d29d17589e230ce1ed7aec2cf2de715a43e47ba55edb37a7" in (
        request
    )
    assert "21 tar entries" in request
    assert "binary `.so`, or benchmark output files" in request
    assert "WaveCut/orbitquant-packed-matmul" in request
    assert "native-kernels/orbitquant-packed-matmul" in request
    assert "https://github.com/iamwavecut/OrbitQuant" in request
    assert "Review source snapshot:" in request
    assert "062b934389dce9242e0a9185ed469cc3170e3e73" in request
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
    assert "CUDA host benchmark evidence is pending" in request
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
