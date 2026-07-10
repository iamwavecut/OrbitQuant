from pathlib import Path


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_release_gates_document_current_public_contract():
    release_gates = _read("docs/release-gates.md")

    for required in (
        "orbitquant==0.1.6",
        "`runtime_mode=\"auto_fused\"`",
        "14 public compact model repositories",
        "arXiv 2607.02461v1",
        "codebook version 2",
        "x / (norm(x) + 1e-10)",
        "scripts/run_paper_methodology_checks.sh",
        "scripts/run_hf_compat_checks.sh --mode all",
        "--fail-on-artifact-regression",
        "FLUX.1-schnell",
        "FLUX.2 Klein",
        "Z-Image-Turbo",
        "Wan2.1-T2V-1.3B-Diffusers",
        "does not publish GenEval or VBench scores",
    ):
        assert required in release_gates

    assert "raw generated images, videos, temporary files" in release_gates
    assert "Kernel Hub" in release_gates
    assert "discussion" not in release_gates.lower()
    assert "On 2026-" not in release_gates


def test_publication_checklist_is_reusable_and_verifies_public_install():
    checklist = _read("docs/publication-checklist.md")

    for required in (
        'export VERSION="0.1.6"',
        "test -z \"$(git status --porcelain)\"",
        "uv run pytest -q",
        "uv run ruff check .",
        "uv build",
        "uvx twine check dist/*",
        'gh workflow run publish-pypi.yml --ref main -f version="$VERSION"',
        "UV_NO_CACHE=1 uv pip install",
        "--index-url https://pypi.org/simple",
        "Attach the exact files served by PyPI",
        "gh release create",
        "audit-hf-artifacts",
    ):
        assert required in checklist

    assert "RunPod" not in checklist
    assert "discussion" not in checklist.lower()
    assert "gh run watch 290" not in checklist


def test_release_notes_are_artifact_focused():
    release_notes = _read("docs/release-0.1.0.md")

    assert "OrbitQuant 0.1.0 Release Notes" in release_notes
    assert "Implemented Quantization" in release_notes
    assert 'runtime_mode="auto_fused"' in release_notes
    assert 'runtime_mode="dequant_bf16"' in release_notes
    assert "Paper-aligned targets" in release_notes
    assert "ROCm and XPU are not implemented" in release_notes
    assert "RunPod" not in release_notes
    assert "discussion" not in release_notes.lower()


def test_intermediate_release_notes_keep_their_user_facing_scope():
    notes_011 = _read("docs/release-0.1.1.md")
    notes_012 = _read("docs/release-0.1.2.md")
    notes_013 = _read("docs/release-0.1.3.md")
    notes_014 = _read("docs/release-0.1.4.md")

    assert "does not claim full-model speedup" in notes_011
    assert "does not weaken the artifact-readiness gate" in notes_012
    assert 'pip install "orbitquant[kernels]==0.1.3"' in notes_013
    assert "external GenEval and VBench\nmetric import" in notes_014
    for notes in (notes_011, notes_012, notes_013, notes_014):
        assert "RunPod" not in notes


def test_release_016_notes_describe_the_published_runtime_and_math():
    release_notes = _read("docs/release-0.1.6.md")

    for required in (
        "OrbitQuant 0.1.6 Release Notes",
        'pip install "orbitquant[hf]==0.1.6"',
        "Lloyd-Max codebook version 2",
        "x / (norm(x) + 1e-10)",
        '`runtime_mode="auto_fused"` remains the default',
        '`runtime_mode="dequant_bf16"` remains an explicit reference',
        "does not claim a universal throughput gain",
    ):
        assert required in release_notes


def test_kernel_audit_documents_dispatch_and_claim_boundaries():
    kernel_audit = _read("docs/kernel-audit.md")

    for required in (
        '`runtime_mode="auto_fused"` is the default',
        '`runtime_mode="dequant_bf16"` explicitly',
        "CUDA | Optimized packed inference",
        "MPS/Metal | Optimized packed inference",
        "CPU | Reference",
        "ROCm | Unsupported",
        "XPU | Unsupported",
        "build-and-copy",
        "PYTHONPATH",
        "LOCAL_KERNELS",
        "RTX PRO 4500 Blackwell",
        "100/100",
        "418/418",
        "238/238",
        "300/300",
        "scripts/run_cuda_kernel_checks.sh",
        "scripts/run_mps_kernel_checks.sh",
        "does not claim a universal speedup",
    ):
        assert required in kernel_audit

    assert "discussion" not in kernel_audit.lower()
    assert "On 2026-" not in kernel_audit


def test_kernel_hub_approval_chronicle_is_not_repository_documentation():
    assert not Path("docs/kernel-hub-approval-request.md").exists()


def test_paper_methodology_audit_uses_claim_boundary_language():
    audit = _read("docs/paper-methodology-audit.md")

    assert "Paper revision: arXiv 2607.02461v1" in audit
    assert "Run `scripts/run_paper_methodology_checks.sh`" in audit
    assert "release-grade GenEval/VBench metrics are separate release gates" in audit
    assert "Native artifact readiness is separate from full GenEval or VBench scoring" in audit
    assert "Pending Evidence For Acceleration Claims" in audit
    assert "Raw inventory JSON is audit evidence" in audit
    assert "not a development log" in audit
    assert "normalize by `s + ε`" in audit
    assert "Pass for codebook version 2" in audit
    assert "Published checkpoints use converged Lloyd-Max codebook version 2" in audit
    assert "paper's block-size enumeration omits `h=256`" in audit

    for forbidden in (
        "Full GenEval/VBench is optional during development",
        "The current development path",
        "Known kernel follow-up",
        "development blocker",
        "local ignored audit artifacts",
    ):
        assert forbidden not in audit
