from pathlib import Path


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_readme_documents_the_universal_public_contract():
    readme = _read("README.md")
    normalized = " ".join(readme.split())

    assert "Automatic coverage" in readme
    assert "register_linear_adapter" in readme
    assert "inspect_linear_module_policy" in readme
    assert "quantize_model" in readme
    assert "save_pretrained" in readme
    assert "snapshot_download" in readme
    assert 'runtime_mode="auto_fused"' in readme
    assert 'runtime_mode="dequant_bf16"' in readme
    assert "does not guarantee a quality-preserving bit setting" in normalized


def test_documentation_is_product_focused_instead_of_a_release_chronicle():
    docs = sorted(path.name for path in Path("docs").glob("*.md"))

    assert docs == ["kernel-audit.md", "paper-methodology-audit.md"]
    assert "## Runtime Contract" in _read("docs/kernel-audit.md")
    assert "## Requirement Matrix" in _read("docs/paper-methodology-audit.md")


def test_methodology_gate_references_only_live_test_files():
    script = _read("scripts/run_paper_methodology_checks.sh")

    assert "test_release_gates.py" not in script
    assert "test_readme.py" not in script
