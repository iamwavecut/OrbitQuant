from orbitquant.artifacts import OrbitQuantManifest, render_model_card
from orbitquant.config import OrbitQuantConfig


def _manifest_for_model(
    model_id: str,
    bits: tuple[int, int] = (4, 4),
    *,
    checksums: dict[str, str] | None = None,
    quantization_staging_mode: str = "unknown",
) -> OrbitQuantManifest:
    config = OrbitQuantConfig(
        weight_bits=bits[0],
        activation_bits=bits[1],
        target_policy="auto",
    )
    return OrbitQuantManifest.from_config(
        config,
        source_model_id=model_id,
        source_revision="abc123",
        source_license="unknown",
        quantized_modules=["transformer_blocks.0.attn.to_q"],
        skipped_modules=["text_encoder"],
        checksums=checksums
        or {
            "assets/image_generation_comparison_matrix.webp": "2" * 64,
            "assets/original_vs_orbitquant_seed0.webp": "0" * 64,
            (
                "reports/native/sample-report/assets/"
                "image_generation_comparison_matrix.webp"
            ): "1" * 64,
        },
        quantization_staging_mode=quantization_staging_mode,
    )


def test_model_card_renders_rotation_and_codebook_metadata():
    config = OrbitQuantConfig(
        weight_bits=3,
        activation_bits=3,
        target_policy="z_image",
        rotation_seed=9,
        block_size="paper",
    )
    manifest = OrbitQuantManifest.from_config(
        config,
        source_model_id="Tongyi-MAI/Z-Image-Turbo",
        source_revision="abc123",
        source_license="unknown",
        quantized_modules=["transformer_blocks.0.attn.to_q"],
        skipped_modules=["text_encoder"],
        checksums={
            "assets/image_generation_comparison_matrix.webp": "0" * 64,
        },
    )

    card = render_model_card(manifest)

    assert "snapshot_download" in card
    assert "load_quantized_pipeline_from_artifact" in card
    assert "pipe.enable_model_cpu_offload(device=\"cuda\")" in card
    assert "    device=\"cuda\"," not in card
    assert "load_quantized_pipeline_component" not in card
    assert "component=\"transformer\"" not in card
    assert "with the quantized component patched in" in card
    assert "not used as a standalone Diffusers pipeline repository" in card
    assert "- Rotation seed: `9`" in card
    assert "- Block size: `paper`" in card
    assert "- Block size policy: `largest_power_of_two_dividing_dim`" in card
    assert "- Codebook version: `1`" in card
    assert "- Target policy: `z_image`" in card
    assert "- Quantized transformer modules: `1`" in card
    assert "## Native Settings" in card
    assert "## Validation Status" in card
    assert "Release-grade GenEval metrics: not included in this artifact." in card
    assert "## Visual Comparison" in card
    assert (
        "![assets/image_generation_comparison_matrix.webp]"
        "(assets/image_generation_comparison_matrix.webp)"
    ) in card


def test_model_card_renders_native_validation_evidence_without_raw_records():
    manifest = _manifest_for_model("black-forest-labs/FLUX.1-schnell")
    benchmark_summary = {
        "published_summary": "compact",
        "raw_generation_records": "local-only",
        "native_smoke": {
            "proof_format": "orbitquant-native-smoke-v1",
            "comparison_asset_path": "assets/image_generation_comparison_matrix.webp",
            "paired_prompt_seed_count": 2,
            "paired_prompt_seed_keys": [
                ["flux1-schnell-native", "0", "simple-object"],
                ["flux1-schnell-native", "1", "counting"],
            ],
            "splits": {
                "original": {
                    "generated_samples": 2,
                    "generated_frames": 0,
                    "nonempty_output_count": 2,
                },
                "orbitquant": {
                    "generated_samples": 2,
                    "generated_frames": 0,
                    "nonempty_output_count": 2,
                },
            },
        },
    }

    card = render_model_card(manifest, benchmark_summary=benchmark_summary)

    assert "## Native Validation Evidence" in card
    assert "| Comparison matrix | `assets/image_generation_comparison_matrix.webp` |" in card
    assert "| Paired prompt/seed count | `2` |" in card
    assert "| BF16 source generated samples | `2` |" in card
    assert "| OrbitQuant nonempty outputs | `2` |" in card
    assert "Detailed per-sample generation records are retained outside" in card
    assert "paired_prompt_seed_keys" not in card
    assert "original.metrics.jsonl" not in card


def test_model_card_renders_imported_geneval_release_metrics():
    manifest = _manifest_for_model("black-forest-labs/FLUX.1-schnell")
    benchmark_summary = {
        "metrics": {
            "original": {
                "latest": {
                    "metrics": {
                        "geneval_overall": 0.74,
                        "geneval_per_task_single_object": 0.9,
                        "wall_time_seconds": 12.0,
                    }
                }
            },
            "orbitquant": {
                "latest": {
                    "metrics": {
                        "geneval_overall": 0.71,
                        "geneval_per_task_single_object": 0.88,
                        "wall_time_seconds": 8.5,
                    }
                }
            },
        }
    }

    card = render_model_card(manifest, benchmark_summary=benchmark_summary)

    assert "Release-grade GenEval metrics: included below." in card
    assert "Release-grade GenEval metrics: not included" not in card
    assert "### Release-Grade GenEval Metrics" in card
    assert "| `geneval_overall` | `0.74` | `0.71` |" in card
    assert "| `geneval_per_task_single_object` | `0.9` | `0.88` |" in card
    assert "wall_time_seconds" not in card


def test_model_card_renders_imported_vbench_release_metrics():
    manifest = _manifest_for_model("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", bits=(4, 6))
    benchmark_summary = {
        "metrics": {
            "original": {
                "latest": {
                    "metrics": {
                        "vbench_subject_consistency": 0.82,
                        "vbench_overall_consistency": 0.76,
                    }
                }
            },
            "orbitquant": {
                "latest": {
                    "metrics": {
                        "vbench_subject_consistency": 0.8,
                        "vbench_overall_consistency": 0.74,
                    }
                }
            },
        }
    }

    card = render_model_card(manifest, benchmark_summary=benchmark_summary)

    assert "Release-grade VBench metrics: included below." in card
    assert "Release-grade VBench metrics: not included" not in card
    assert "### Release-Grade VBench Metrics" in card
    assert "| `vbench_subject_consistency` | `0.82` | `0.8` |" in card
    assert "| `vbench_overall_consistency` | `0.76` | `0.74` |" in card


def test_model_card_contains_install_command_not_workflow_log_language():
    card = render_model_card(_manifest_for_model("black-forest-labs/FLUX.1-schnell"))

    assert 'pip install "orbitquant[hf]"' in card
    assert "git+https://github.com/iamwavecut/OrbitQuant.git" not in card
    for forbidden in (
        "reports/",
        "terminal.log",
        "run.jsonl",
        "stage_log",
        "runner logs",
        "RunPod",
        "REMOTE_STAGE",
    ):
        assert forbidden not in card


def test_model_card_embeds_only_promoted_comparison_matrix_assets():
    card = render_model_card(_manifest_for_model("black-forest-labs/FLUX.1-schnell"))

    matrix = "assets/image_generation_comparison_matrix.webp"
    single_sample = "assets/original_vs_orbitquant_seed0.webp"

    assert f"![{matrix}]({matrix})" in card
    assert f"![{single_sample}]({single_sample})" not in card
    assert "reports/native" not in card


def test_model_card_embeds_only_one_comparison_matrix_asset():
    card = render_model_card(
        _manifest_for_model(
            "black-forest-labs/FLUX.1-schnell",
            checksums={
                "assets/image_generation_comparison_matrix.webp": "0" * 64,
                "assets/video_generation_comparison_matrix.webp": "1" * 64,
            },
        )
    )

    assert (
        "![assets/image_generation_comparison_matrix.webp]"
        "(assets/image_generation_comparison_matrix.webp)"
    ) in card
    assert "![assets/video_generation_comparison_matrix.webp]" not in card


def test_model_card_reports_non_default_adaln_group_size():
    config = OrbitQuantConfig(
        weight_bits=4,
        activation_bits=4,
        target_policy="flux",
        adaln_group_size=32,
    )
    manifest = OrbitQuantManifest.from_config(
        config,
        source_model_id="black-forest-labs/FLUX.1-schnell",
        source_revision="abc123",
        source_license="apache-2.0",
        quantized_modules=["transformer_blocks.0.attn.to_q"],
        adaln_modules=["transformer_blocks.0.norm1.linear"],
        skipped_modules=["proj_out"],
        checksums={"assets/image_generation_comparison_matrix.webp": "0" * 64},
    )

    card = render_model_card(manifest)

    assert "- AdaLN policy: `int4_rtn_group32_bf16_activation`" in card
    assert "- AdaLN group size: `32`" in card
    assert "- AdaLN group-size note: non-paper-default setting." in card


def test_model_card_ignores_contact_sheets_for_published_artifacts():
    contact_sheet = "assets/flux1_schnell_contact_sheet.webp"
    single_sample = "assets/original_vs_orbitquant_seed0.webp"
    card = render_model_card(
        _manifest_for_model(
            "black-forest-labs/FLUX.1-schnell",
            checksums={
                single_sample: "0" * 64,
                contact_sheet: "1" * 64,
            },
        )
    )

    assert "Validation status: comparison asset missing" in card
    assert "This artifact does not include a generation comparison matrix." in card
    assert f"![{contact_sheet}]({contact_sheet})" not in card
    assert f"![{single_sample}]({single_sample})" not in card


def test_model_card_marks_missing_promoted_comparison_assets_without_logs():
    card = render_model_card(
        _manifest_for_model(
            "black-forest-labs/FLUX.1-schnell",
            checksums={
                "assets/terminal.log": "0" * 64,
                "assets/run.jsonl": "1" * 64,
                "assets/nested/debug_generation_comparison_matrix.webp": "2" * 64,
                "reports/native/stage_log/image_generation_comparison_matrix.webp": "2" * 64,
            },
            quantization_staging_mode="REMOTE_STAGE",
        )
    )

    assert "Validation status: comparison asset missing" in card
    assert "does not include a generation comparison matrix" in card
    for forbidden in (
        "reports/",
        "terminal.log",
        "run.jsonl",
        "debug_generation_comparison_matrix.webp",
        "stage_log",
        "runner logs",
        "RunPod",
        "REMOTE_STAGE",
    ):
        assert forbidden not in card


def test_model_card_uses_flux2_native_code_example():
    card = render_model_card(_manifest_for_model("black-forest-labs/FLUX.2-klein-4B"))

    assert "load_quantized_pipeline_from_artifact" in card
    assert "Flux2KleinPipeline.from_pretrained" not in card
    assert "height=1024" in card
    assert "width=1024" in card
    assert "num_inference_steps=4" in card
    assert "guidance_scale=1.0" in card
    assert "| Pipeline | `Flux2KleinPipeline` |" in card
    assert "| Resolution | `1024x1024` |" in card
    assert "| Guidance scale | `1.0` |" in card
    assert "| Scope | extra target; not an OrbitQuant paper reproduction model |" in card
    assert "Release-grade paper metrics: not applicable to this extra target." in card


def test_model_card_uses_flux1_schnell_native_code_example():
    card = render_model_card(_manifest_for_model("black-forest-labs/FLUX.1-schnell"))

    assert "load_quantized_pipeline_from_artifact" in card
    assert "FluxPipeline.from_pretrained" not in card
    assert "height=1024" in card
    assert "width=1024" in card
    assert "num_inference_steps=4" in card
    assert "guidance_scale=0.0" in card
    assert "| Pipeline | `FluxPipeline` |" in card
    assert "| Inference steps | `4` |" in card
    assert "| Guidance scale | `0.0` |" in card
    assert "| Scope | paper image target |" in card
    assert "Release-grade GenEval metrics: not included in this artifact." in card


def test_model_card_uses_z_image_native_code_example():
    card = render_model_card(_manifest_for_model("Tongyi-MAI/Z-Image-Turbo"))

    assert "load_quantized_pipeline_from_artifact" in card
    assert "ZImagePipeline.from_pretrained" not in card
    assert "height=1024" in card
    assert "width=1024" in card
    assert "num_inference_steps=10" in card
    assert "guidance_scale=0.0" in card
    assert "| Pipeline | `ZImagePipeline` |" in card
    assert "| Inference steps | `10` |" in card
    assert "| Guidance scale | `0.0` |" in card
    assert "| Scope | paper image target |" in card
    assert "Release-grade GenEval metrics: not included in this artifact." in card


def test_model_card_uses_wan_native_code_example():
    card = render_model_card(
        _manifest_for_model("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", bits=(4, 6))
    )

    assert "from diffusers.utils import export_to_video" in card
    assert "load_quantized_pipeline_from_artifact" in card
    assert "WanPipeline.from_pretrained" not in card
    assert "height=480" in card
    assert "width=832" in card
    assert "num_frames=81" in card
    assert "num_inference_steps=50" in card
    assert "guidance_scale=5.0" in card
    assert "| Pipeline | `WanPipeline` |" in card
    assert "| Resolution | `832x480` |" in card
    assert "| Frames | `81` |" in card
    assert "| Export FPS | `16` |" in card
    assert "| Scope | paper video target |" in card
    assert "Release-grade VBench metrics: not included in this artifact." in card
