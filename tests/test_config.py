import json

from orbitquant.config import OrbitQuantConfig


def test_orbit_quant_config_round_trips_to_dict():
    config = OrbitQuantConfig(
        weight_bits=3,
        activation_bits=3,
        target_policy="flux",
        activation_kernel_backend="cpu",
        modules_to_not_convert=["text_encoder"],
    )

    restored = OrbitQuantConfig.from_dict(config.to_dict())

    assert restored.weight_bits == 3
    assert restored.activation_bits == 3
    assert restored.target_policy == "flux"
    assert restored.activation_kernel_backend == "cpu"
    assert restored.modules_to_not_convert == ["text_encoder"]
    assert restored.quant_method == "orbitquant"


def test_orbit_quant_config_supports_hf_json_helpers():
    config = OrbitQuantConfig(weight_bits=3, activation_bits=3, block_size=8)

    payload = json.loads(config.to_json_string())
    restored = OrbitQuantConfig.from_dict(payload)

    assert payload["quant_method"] == "orbitquant"
    assert payload["weight_bits"] == 3
    assert payload["activation_bits"] == 3
    assert restored.weight_bits == 3
    assert restored.activation_bits == 3


def test_orbit_quant_config_default_epsilon_matches_paper_equation():
    config = OrbitQuantConfig()

    assert config.activation_eps == 1e-10


def test_orbit_quant_config_defaults_to_auto_fused_runtime_mode():
    config = OrbitQuantConfig()

    assert config.runtime_mode == "auto_fused"
    assert config.packed_matmul_block_n == 128


def test_orbit_quant_config_accepts_auto_fused_runtime_mode():
    config = OrbitQuantConfig(runtime_mode="auto_fused")

    assert config.runtime_mode == "auto_fused"


def test_orbit_quant_config_accepts_triton_packed_matmul_runtime_mode():
    config = OrbitQuantConfig(runtime_mode="triton_packed_matmul")

    assert config.runtime_mode == "triton_packed_matmul"


def test_orbit_quant_config_accepts_native_packed_matmul_runtime_mode():
    config = OrbitQuantConfig(runtime_mode="native_packed_matmul")

    assert config.runtime_mode == "native_packed_matmul"


def test_orbit_quant_config_round_trips_packed_matmul_tile_config():
    config = OrbitQuantConfig(
        packed_matmul_block_m=32,
        packed_matmul_block_n=64,
        packed_matmul_block_k=64,
        packed_matmul_num_warps=8,
    )

    restored = OrbitQuantConfig.from_dict(config.to_dict())

    assert restored.packed_matmul_block_m == 32
    assert restored.packed_matmul_block_n == 64
    assert restored.packed_matmul_block_k == 64
    assert restored.packed_matmul_num_warps == 8


def test_orbit_quant_config_rejects_invalid_bits():
    try:
        OrbitQuantConfig(weight_bits=1, activation_bits=4)
    except ValueError as exc:
        assert "weight_bits" in str(exc)
    else:
        raise AssertionError("invalid weight_bits were accepted")


def test_orbit_quant_config_rejects_invalid_activation_kernel_backend():
    try:
        OrbitQuantConfig(activation_kernel_backend="cuda")
    except ValueError as exc:
        assert "activation_kernel_backend" in str(exc)
    else:
        raise AssertionError("invalid activation_kernel_backend was accepted")


def test_orbit_quant_config_rejects_invalid_module_dtype_override():
    try:
        OrbitQuantConfig(modules_dtype_dict={"int8": ["transformer_blocks.0.attn.to_q"]})
    except ValueError as exc:
        assert "modules_dtype_dict" in str(exc)
    else:
        raise AssertionError("invalid modules_dtype_dict dtype was accepted")


def test_orbit_quant_config_rejects_unknown_target_policy():
    try:
        OrbitQuantConfig(target_policy="flxu")
    except ValueError as exc:
        assert "target_policy" in str(exc)
    else:
        raise AssertionError("unknown target_policy was accepted")


def test_orbit_quant_config_rejects_unimplemented_serialized_knobs():
    invalid_kwargs = [
        {"row_norm_dtype": "float32"},
        {"activation_norm_dtype": "bfloat16"},
        {"codebook_dtype": "bfloat16"},
        {"weight_pack_dtype": "int32"},
        {"adaln_policy": "orbitquant"},
    ]

    for kwargs in invalid_kwargs:
        try:
            OrbitQuantConfig(**kwargs)
        except ValueError as exc:
            assert next(iter(kwargs)) in str(exc)
        else:
            raise AssertionError(f"unsupported OrbitQuantConfig kwargs accepted: {kwargs}")
