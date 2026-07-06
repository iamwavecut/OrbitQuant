from orbitquant.artifacts import OrbitQuantManifest
from orbitquant.config import OrbitQuantConfig


def test_manifest_records_source_and_quantization_settings():
    config = OrbitQuantConfig(weight_bits=4, activation_bits=4, target_policy="flux2")
    manifest = OrbitQuantManifest.from_config(
        config,
        source_model_id="black-forest-labs/FLUX.2-klein-4B",
        source_revision="abc123",
        source_license="apache-2.0",
        quantized_modules=["transformer_blocks.0.attn.to_q"],
        skipped_modules=["text_encoder"],
    )

    data = manifest.to_dict()

    assert data["artifact_format"] == "orbitquant-v1"
    assert data["source_model_id"] == "black-forest-labs/FLUX.2-klein-4B"
    assert data["source_revision"] == "abc123"
    assert data["source_license"] == "apache-2.0"
    assert data["weight_bits"] == 4
    assert data["activation_bits"] == 4
    assert data["quantized_modules"] == ["transformer_blocks.0.attn.to_q"]
