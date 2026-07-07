from pathlib import Path


def test_release_gates_document_final_acceptance_checklist():
    release_gates = Path("docs/release-gates.md").read_text(encoding="utf-8")

    assert "final acceptance gate" in release_gates
    assert "arXiv 2607.02461" in release_gates
    assert "paper-aligned subset" in release_gates
    assert "FLUX.2 Klein is\n  an additional target" in release_gates
    assert "native-resolution BF16-vs-OrbitQuant\n  comparison assets" in release_gates
    assert "finite-output checks" in release_gates
    assert "paper reproduction or\n  metric-table claims" in release_gates
    assert "GenEval\n  overall and per-task scores" in release_gates
    assert "all required VBench\n  dimensions" in release_gates
    assert "Missing release metrics block only those metric/reproduction\n  claims" in release_gates
    assert "Full-model module classification inventories" in release_gates
    assert "CUDA/Triton, CPU, Metal/MPS, ROCm, and XPU" in release_gates
    assert (
        "latest published releases and dev\n  branches of Diffusers and Transformers"
        in release_gates
    )
    assert "ComfyUI compatibility is verified after the relevant schema stabilizes" in release_gates
    assert "artifact-focused\n  model cards" in release_gates
    assert "native\n  comparison assets" in release_gates
    assert "not host logs, raw eval dumps, or terminal transcripts" in release_gates
    assert "The GitHub repository is public" in release_gates
    assert "python -m build" in release_gates
    assert "python -m twine check dist/*" in release_gates
    assert "python -m twine upload\n  dist/*" in release_gates
    assert "PyPI token or browser\n  action" in release_gates
    assert "chronology" not in release_gates.lower()
