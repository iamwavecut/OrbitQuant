from pathlib import Path


def test_release_gates_document_final_acceptance_checklist():
    release_gates = Path("docs/release-gates.md").read_text(encoding="utf-8")

    assert "final acceptance gate" in release_gates
    assert "verification output" in release_gates
    assert "arXiv 2607.02461" in release_gates
    assert "`scripts/run_paper_methodology_checks.sh`" in release_gates
    assert "- [x] Final paper conformance audit" in release_gates
    assert "passed on 2026-07-08T15:49Z against arXiv 2607.02461v1" in release_gates
    assert "paper-aligned subset" in release_gates
    assert "FLUX.2 Klein is\n  an additional target" in release_gates
    assert "native-resolution BF16-vs-OrbitQuant\n  comparison assets" in release_gates
    assert "finite-output checks" in release_gates
    assert "`native_smoke` proof\n  block in `benchmark/summary.json`" in release_gates
    assert "raw generation records remain local-only" in release_gates
    assert "paper reproduction or\n  metric-table claims" in release_gates
    assert "GenEval\n  overall and per-task scores" in release_gates
    assert "all required VBench\n  dimensions" in release_gates
    assert "Missing release metrics block only those metric/reproduction\n  claims" in release_gates
    assert "compact artifacts without those metrics must present native comparison" in (
        release_gates
    )
    assert "status instead of paper-reproduction metric claims" in release_gates
    assert "Full-model module classification inventories" in release_gates
    assert "Raw inventory JSON may\n  remain unpublished" in release_gates
    assert "CUDA/Triton partial optimized" in release_gates
    assert "Metal/MPS partial optimized" in release_gates
    assert "CPU\n  reference-only" in release_gates
    assert "ROCm/XPU explicitly unsupported" in release_gates
    assert "[kernel-audit.md](kernel-audit.md)" in release_gates
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
    assert "artifact-focused\n  model cards" in release_gates
    assert "native\n  comparison assets" in release_gates
    assert "`metadata_complete_ready`" in release_gates
    assert (
        "quantization device, weight quantization backend, and\n  staging mode provenance"
        in release_gates
    )
    assert "not host logs, raw eval dumps, or terminal transcripts" in release_gates
    assert "The GitHub repository is public" in release_gates
    assert "python -m build" in release_gates
    assert "python -m twine check dist/*" in release_gates
    assert "python -m twine upload\n  dist/*" in release_gates
    assert "PyPI token or browser\n  action" in release_gates
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
