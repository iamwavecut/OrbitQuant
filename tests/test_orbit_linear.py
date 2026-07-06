import torch

from orbitquant.config import OrbitQuantConfig
from orbitquant.layers import OrbitQuantLinear


def test_orbit_linear_debug_rotation_matches_source_linear():
    torch.manual_seed(0)
    source = torch.nn.Linear(16, 7)
    x = torch.randn(2, 5, 16)
    config = OrbitQuantConfig(
        weight_bits=4,
        activation_bits=4,
        runtime_mode="debug_no_quant",
        rotation_seed=11,
        block_size=8,
    )

    quantized = OrbitQuantLinear.from_linear(source, config=config, module_name="block.attn.to_q")

    expected = source(x)
    actual = quantized(x)

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_orbit_linear_quantized_forward_is_finite_and_shape_preserving():
    torch.manual_seed(1)
    source = torch.nn.Linear(16, 7)
    x = torch.randn(2, 5, 16)
    config = OrbitQuantConfig(weight_bits=4, activation_bits=4, rotation_seed=11, block_size=8)

    quantized = OrbitQuantLinear.from_linear(source, config=config, module_name="block.ff.linear")
    actual = quantized(x)

    assert actual.shape == (2, 5, 7)
    assert torch.isfinite(actual).all()
    assert not any(parameter.requires_grad for parameter in quantized.parameters())
