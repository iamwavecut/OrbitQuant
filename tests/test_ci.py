from pathlib import Path


def test_github_actions_cpu_unit_workflow_exists():
    workflow = Path(".github/workflows/ci.yml")

    assert workflow.is_file()
    text = workflow.read_text(encoding="utf-8")
    assert "uv sync --extra dev --extra hf --extra eval" in text
    assert "uv run ruff check ." in text
    assert "uv run pytest" in text
