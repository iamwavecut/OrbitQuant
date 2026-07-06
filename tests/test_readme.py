from pathlib import Path


def test_readme_documents_native_gpu_pod_workflow():
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "RTX PRO 6000" in readme
    assert "orbitquant native-plan" in readme
    assert "orbitquant native-script" in readme
    assert "hf auth whoami" in readme
    assert "orbitquant record-metrics" in readme
