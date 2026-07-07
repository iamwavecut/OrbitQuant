from pathlib import Path


def test_readme_documents_component_artifact_usage():
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "snapshot_download" in readme
    assert "load_quantized_pipeline_component" in readme
    assert "Published OrbitQuant model repos are component artifacts" in readme
    assert "DiffusionPipeline.from_pretrained" in readme
    assert "orbitquant quantize" in readme
    assert "orbitquant validate-artifact" in readme
    assert "Comparison Assets" in readme
    assert "assets/*_generation_comparison_matrix.webp" in readme
    assert "original_vs_orbitquant" not in readme
    assert "contact sheet" not in readme.lower()
    assert "Pipeline class" in readme
    assert "Flux2KleinPipeline" in readme
    assert "FluxPipeline" in readme
    assert "ZImagePipeline" in readme
    assert "WanPipeline" in readme
    assert "832x480, 81 frames, 50 steps, guidance 5.0" in readme
    assert "`orbitquant upload-artifact` uses the compact upload profile by default" in readme
    assert "uploads\nonly the compact artifact files required for use" in readme
    assert "These diffusion artifacts are not standalone `transformers.AutoModel` repos" in readme
    assert 'quantization_device="cuda"' in readme
    assert "not used\nas quality evidence" in readme
    assert "## Release Metrics" in readme
    assert "orbitquant fetch-hf-artifacts" in readme
    assert "orbitquant native-script" in readme
    assert "--prompt-metadata-jsonl /path/to/GenEval/evaluation_metadata.jsonl" in readme
    assert "orbitquant external-eval-script" in readme
    assert "[docs/release-gates.md](docs/release-gates.md)" in readme
    assert "stage_log START/END" not in readme
    assert "RunPod" not in readme
    assert "REMOTE_STAGE" not in readme
