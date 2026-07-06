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
    )

    card = render_model_card(manifest)

    assert "- Rotation seed: `9`" in card
    assert "- Block size: `paper`" in card
    assert "- Block size policy: `largest_power_of_two_dividing_dim`" in card
    assert "- Codebook version: `1`" in card
    assert "- Target policy: `z_image`" in card
