from pathlib import Path


def test_readme_documents_component_artifact_usage():
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "snapshot_download" in readme
    assert 'pip install "orbitquant[hf]"' in readme
    assert (
        'pip install "orbitquant[hf] @ git+https://github.com/iamwavecut/OrbitQuant.git"'
        in readme
    )
    assert 'pip install -e ".[hf,eval,dev]"' in readme
    assert "load_quantized_pipeline_component" in readme
    assert "Published OrbitQuant model repos are component artifacts" in readme
    assert "DiffusionPipeline.from_pretrained" in readme
    assert "build_diffusers_pipeline_quantization_config" in readme
    assert "`PipelineQuantizationConfig`" in readme
    assert "orbitquant quantize" in readme
    assert "orbitquant inspect-policy" in readme
    assert "empty-weight skeleton" in readme
    assert "orbitquant validate-artifact" in readme
    assert "orbitquant audit-hf-artifacts" in readme
    assert "--policy-inventory-root ./reports/native/module-inventories" in readme
    assert "Comparison Assets" in readme
    assert "assets/*_generation_comparison_matrix.webp" in readme
    assert "original_vs_orbitquant" not in readme
    assert "contact sheet" not in readme.lower()
    assert "Pipeline class" in readme
    assert "Paper-aligned artifacts use these native target settings" in readme
    assert "Extra target artifacts use the same native-validation rules" in readme
    assert "Flux2KleinPipeline" in readme
    assert "FluxPipeline" in readme
    assert "ZImagePipeline" in readme
    assert "WanPipeline" in readme
    assert "832x480, 81 frames, 50 steps, guidance 5.0" in readme
    assert "`orbitquant upload-artifact` uses the compact upload profile by default" in readme
    assert "uploads\nonly the compact artifact files required for use" in readme
    assert "Existing remote files are replaced by default" in readme
    assert "These diffusion artifacts are not standalone `transformers.AutoModel` repos" in readme
    assert "## Hugging Face Native Loaders" in readme
    assert "from transformers import AutoModel" in readme
    assert "quantization_config=config" in readme
    assert 'model.save_pretrained("./source-pretrained-model-orbitquant-w4a4")' in readme
    assert "This path is for Hugging Face-native model repositories" in readme
    assert 'quantization_device="cuda"' in readme
    assert "not accepted as published quality evidence" in readme
    assert "Local validation outputs may include raw `benchmark/*.jsonl`" in readme
    assert "compact published artifacts omit those raw files" in readme
    assert "outside the current release claim boundary" in " ".join(readme.split())
    assert "## Release Metrics" in readme
    assert "Full GenEval and VBench runs are release evidence" in readme
    assert "Compact artifact readiness is tracked\nseparately" in readme
    assert "native validation evidence" in readme
    assert "orbitquant fetch-hf-artifacts" in readme
    assert "orbitquant native-script" in readme
    assert "--prompt-metadata-jsonl /path/to/GenEval/evaluation_metadata.jsonl" in readme
    assert "orbitquant external-eval-script" in readme
    assert "[docs/release-gates.md](docs/release-gates.md)" in readme
    assert "CPU is a correctness reference path only" in readme
    assert "MPS/Metal is partially optimized" in readme
    assert "CUDA/Triton is partially optimized" in readme
    assert "ROCm and XPU are not implemented backends" in readme
    assert "scripts/run_mps_kernel_checks.sh" in readme
    assert "[docs/kernel-audit.md](docs/kernel-audit.md)" in readme
    assert "stage_log START/END" not in readme
    assert "RunPod" not in readme
    assert "REMOTE_STAGE" not in readme
    assert "agreed native settings" not in readme
    assert "Local working artifacts" not in readme
    assert "future work" not in readme
