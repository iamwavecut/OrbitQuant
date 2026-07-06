from orbitquant.config import OrbitQuantConfig


def test_orbit_quant_config_round_trips_to_dict():
    config = OrbitQuantConfig(
        weight_bits=3,
        activation_bits=3,
        target_policy="flux",
        modules_to_not_convert=["text_encoder"],
    )

    restored = OrbitQuantConfig.from_dict(config.to_dict())

    assert restored.weight_bits == 3
    assert restored.activation_bits == 3
    assert restored.target_policy == "flux"
    assert restored.modules_to_not_convert == ["text_encoder"]
    assert restored.quant_method == "orbitquant"


def test_orbit_quant_config_rejects_invalid_bits():
    try:
        OrbitQuantConfig(weight_bits=1, activation_bits=4)
    except ValueError as exc:
        assert "weight_bits" in str(exc)
    else:
        raise AssertionError("invalid weight_bits were accepted")
