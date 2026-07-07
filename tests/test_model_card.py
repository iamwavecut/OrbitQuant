from orbitquant.artifacts import OrbitQuantManifest, render_model_card
from orbitquant.config import OrbitQuantConfig


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
            "assets/original_vs_orbitquant_z-image-native_seed0_W3A3_simple-object.webp": (
                "0" * 64
            )
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
    assert "## Visual Comparison" in card
    assert (
        "![assets/original_vs_orbitquant_z-image-native_seed0_W3A3_simple-object.webp]"
        "(assets/original_vs_orbitquant_z-image-native_seed0_W3A3_simple-object.webp)"
    ) in card
