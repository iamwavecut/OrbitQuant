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
