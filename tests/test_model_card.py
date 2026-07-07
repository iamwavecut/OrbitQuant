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
    assert "load_quantized_pipeline_component" in card
    assert "component=\"transformer\"" in card
    assert "not used as a standalone Diffusers pipeline repository" in card
    assert "- Rotation seed: `9`" in card
    assert "- Block size: `paper`" in card
    assert "- Block size policy: `largest_power_of_two_dividing_dim`" in card
    assert "- Codebook version: `1`" in card
    assert "- Target policy: `z_image`" in card
    assert "- Quantized transformer modules: `1`" in card
    assert "## Native Settings" in card
    assert "## Visual Comparison" in card
    assert (
        "![assets/image_generation_comparison_matrix.webp]"
        "(assets/image_generation_comparison_matrix.webp)"
    ) in card


def test_model_card_contains_install_command_not_workflow_log_language():
    card = render_model_card(_manifest_for_model("black-forest-labs/FLUX.1-schnell"))

    assert "pip install git+https://github.com/iamwavecut/OrbitQuant.git" in card
    assert "diffusers" in card
    assert "transformers" in card
    assert "accelerate" in card
    for forbidden in (
        "reports/",
        "terminal.log",
        "run.jsonl",
        "stage_log",
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

    assert "Release readiness: blocked" in card
    assert f"![{contact_sheet}]({contact_sheet})" not in card
    assert f"![{single_sample}]({single_sample})" not in card


def test_model_card_marks_release_blocked_without_promoted_comparison_assets():
    card = render_model_card(
        _manifest_for_model(
            "black-forest-labs/FLUX.1-schnell",
            checksums={
                "assets/terminal.log": "0" * 64,
                "assets/run.jsonl": "1" * 64,
                "reports/native/stage_log/image_generation_comparison_matrix.webp": "2" * 64,
            },
            quantization_staging_mode="REMOTE_STAGE",
        )
    )

    assert "Release readiness: blocked" in card
    assert "does not include a generation comparison matrix" in card
    for forbidden in (
        "reports/",
        "terminal.log",
        "run.jsonl",
        "stage_log",
        "RunPod",
        "REMOTE_STAGE",
    ):
        assert forbidden not in card


def test_model_card_uses_flux2_native_code_example():
    card = render_model_card(_manifest_for_model("black-forest-labs/FLUX.2-klein-4B"))

    assert "from diffusers import Flux2KleinPipeline" in card
    assert "Flux2KleinPipeline.from_pretrained" in card
    assert "height=1024" in card
    assert "width=1024" in card
    assert "num_inference_steps=4" in card
    assert "guidance_scale=1.0" in card
    assert "| Pipeline | `Flux2KleinPipeline` |" in card
    assert "| Resolution | `1024x1024` |" in card
    assert "| Guidance scale | `1.0` |" in card
    assert "| Scope | extra target; not an OrbitQuant paper reproduction model |" in card


def test_model_card_uses_flux1_schnell_native_code_example():
    card = render_model_card(_manifest_for_model("black-forest-labs/FLUX.1-schnell"))

    assert "from diffusers import FluxPipeline" in card
    assert "FluxPipeline.from_pretrained" in card
    assert "height=1024" in card
    assert "width=1024" in card
    assert "num_inference_steps=4" in card
    assert "guidance_scale=0.0" in card
    assert "| Pipeline | `FluxPipeline` |" in card
    assert "| Inference steps | `4` |" in card
    assert "| Guidance scale | `0.0` |" in card
    assert "| Scope | paper image target |" in card


def test_model_card_uses_z_image_native_code_example():
    card = render_model_card(_manifest_for_model("Tongyi-MAI/Z-Image-Turbo"))

    assert "from diffusers import ZImagePipeline" in card
    assert "ZImagePipeline.from_pretrained" in card
    assert "height=1024" in card
    assert "width=1024" in card
    assert "num_inference_steps=10" in card
    assert "guidance_scale=0.0" in card
    assert "| Pipeline | `ZImagePipeline` |" in card
    assert "| Inference steps | `10` |" in card
    assert "| Guidance scale | `0.0` |" in card
    assert "| Scope | paper image target |" in card


def test_model_card_uses_wan_native_code_example():
    card = render_model_card(
        _manifest_for_model("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", bits=(4, 6))
    )

    assert "from diffusers import WanPipeline" in card
    assert "from diffusers.utils import export_to_video" in card
    assert "WanPipeline.from_pretrained" in card
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
