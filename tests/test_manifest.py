from orbitquant.artifacts import OrbitQuantManifest
from orbitquant.config import OrbitQuantConfig


def test_manifest_records_source_and_quantization_settings():
    config = OrbitQuantConfig(
        weight_bits=4,
        activation_bits=4,
        target_policy="flux2",
        rotation_seed=17,
        block_size=128,
    )
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
    assert data["rotation"] == "rpbh"
    assert data["rotation_seed"] == 17
    assert data["block_size"] == 128
    assert data["block_size_policy"] == "explicit"
    assert data["codebook"] == "lloyd_max"
    assert data["codebook_version"] == 2
    assert data["activation_eps"] == 1e-10
    assert data["quantized_modules"] == ["transformer_blocks.0.attn.to_q"]

    restored = OrbitQuantManifest.from_dict(data)
    assert restored.rotation_seed == 17
    assert restored.block_size == 128
    assert restored.block_size_policy == "explicit"
    assert restored.codebook_version == 2
    assert restored.activation_eps == 1e-10


def test_manifest_records_paper_block_size_policy():
    config = OrbitQuantConfig(block_size="paper")
    manifest = OrbitQuantManifest.from_config(
        config,
        source_model_id="example/model",
        source_revision="abc123",
        source_license="apache-2.0",
        quantized_modules=[],
        skipped_modules=[],
    )

    data = manifest.to_dict()

    assert data["block_size"] == "paper"
    assert data["block_size_policy"] == "largest_power_of_two_dividing_dim"


def test_manifest_reads_legacy_payload_without_activation_epsilon():
    data = {
        "source_model_id": "example/model",
        "source_revision": "abc123",
        "source_license": "apache-2.0",
        "weight_bits": 4,
        "activation_bits": 4,
    }

    manifest = OrbitQuantManifest.from_dict(data)

    assert manifest.activation_eps == 1e-10
