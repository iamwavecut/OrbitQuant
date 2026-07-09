import torch

from orbitquant.artifacts.manifest import OrbitQuantManifest
from orbitquant.config import OrbitQuantConfig
from orbitquant.functional import quantize_activations
from orbitquant.layers import OrbitQuantLinear
from orbitquant.rotations import RPBHRotation


def test_layers_with_same_input_dimension_share_orbit_state():
    config = OrbitQuantConfig(weight_bits=4, activation_bits=4, rotation_seed=17)

    first = OrbitQuantLinear.from_linear(
        torch.nn.Linear(16, 7),
        config=config,
        module_name="transformer_blocks.0.attn.to_q",
    )
    second = OrbitQuantLinear.from_linear(
        torch.nn.Linear(16, 5),
        config=config,
        module_name="transformer_blocks.1.attn.to_v",
    )
    different_dim = OrbitQuantLinear.from_linear(
        torch.nn.Linear(32, 5),
        config=config,
        module_name="transformer_blocks.2.attn.to_k",
    )

    assert first.rotation is second.rotation
    assert first.weight_codebook is second.weight_codebook
    assert first.activation_codebook is second.activation_codebook
    assert different_dim.rotation is not first.rotation
    assert different_dim.weight_codebook is not first.weight_codebook
    assert different_dim.activation_codebook is not first.activation_codebook


def test_activation_quantization_has_no_batch_range_dependency():
    torch.manual_seed(31)
    x = torch.randn(3, 16).clamp(-2, 2)
    outlier = torch.full((1, 16), 10000.0)
    config = OrbitQuantConfig(weight_bits=4, activation_bits=4, rotation_seed=3)
    layer = OrbitQuantLinear.from_linear(
        torch.nn.Linear(16, 8),
        config=config,
        module_name="transformer_blocks.0.attn.to_q",
    )

    baseline = quantize_activations(
        x,
        rotation=layer.rotation,
        codebook=layer.activation_codebook,
        eps=config.activation_eps,
    )
    with_outlier = quantize_activations(
        torch.cat((x, outlier), dim=0),
        rotation=layer.rotation,
        codebook=layer.activation_codebook,
        eps=config.activation_eps,
    )

    assert torch.equal(with_outlier[: x.shape[0]], baseline)


def test_manifest_records_data_agnostic_quantization_without_calibration_state():
    config = OrbitQuantConfig(
        weight_bits=3,
        activation_bits=3,
        target_policy="flux2",
        rotation_seed=11,
    )
    manifest = OrbitQuantManifest.from_config(
        config,
        source_model_id="black-forest-labs/FLUX.2-klein-4B",
        source_revision="test-revision",
        source_license="Apache-2.0",
        quantized_modules=["transformer_blocks.0.attn.to_q"],
        adaln_modules=["double_stream_modulation_img.linear"],
        skipped_modules=["proj_out"],
        module_shapes={"transformer_blocks.0.attn.to_q.packed_weight_indices": [128]},
        checksums={},
        quantization_device="cpu",
        weight_quantization_backend="torch_reference",
        quantization_staging_mode="component",
    )

    payload = manifest.to_dict()
    forbidden_terms = ("calibration", "calibrate", "activation_range", "prompt_range")

    assert payload["codebook"] == "lloyd_max"
    assert payload["rotation"] == "rpbh"
    assert payload["block_size_policy"] == "largest_power_of_two_dividing_dim"
    assert payload["adaln_policy"] == "int4_rtn_group64_bf16_activation"
    assert not any(term in key for key in payload for term in forbidden_terms)
    assert not any(
        term in str(value).lower()
        for value in payload.values()
        for term in forbidden_terms
    )


def test_paper_block_rule_covers_all_target_projection_dimensions():
    expected = {
        1536: 512,
        3072: 1024,
        3840: 256,
        8960: 256,
        9216: 1024,
        10240: 2048,
        12288: 4096,
        15360: 1024,
    }

    actual = {
        dim: RPBHRotation(dim=dim, block_size="paper").block_size for dim in expected
    }

    assert actual == expected
