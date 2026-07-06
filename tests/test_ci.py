from pathlib import Path


def test_github_actions_cpu_unit_workflow_exists():
    workflow = Path(".github/workflows/ci.yml")

    assert workflow.is_file()
    text = workflow.read_text(encoding="utf-8")
    assert "uses: actions/checkout@v7.0.0" in text
    assert "uses: actions/setup-python@v6.3.0" in text
    assert "uses: astral-sh/setup-uv@v8.3.0" in text
    assert "uv sync --extra dev --extra hf --extra eval" in text
    assert "uv run ruff check ." in text
    assert "uv run pytest" in text
    assert "uv build" in text
    assert "uv pip install --python" in text
    assert "dist/orbitquant-0.1.0-py3-none-any.whl" in text
    assert "orbitquant --version" in text
