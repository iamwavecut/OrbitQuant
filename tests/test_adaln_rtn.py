import torch

from orbitquant.adaln import RTNInt4Linear
from orbitquant.config import OrbitQuantConfig


def test_int4_rtn_linear_preserves_shape_and_freezes_parameters():
    torch.manual_seed(0)
    source = torch.nn.Linear(17, 9)
    x = torch.randn(2, 3, 17)
    config = OrbitQuantConfig(adaln_group_size=8)

    quantized = RTNInt4Linear.from_linear(source, config=config, module_name="block.modulation")
    actual = quantized(x)

    assert actual.shape == (2, 3, 9)
    assert torch.isfinite(actual).all()
    assert not any(parameter.requires_grad for parameter in quantized.parameters())
    assert quantized.group_size == 8


def test_int4_rtn_rejects_non_positive_group_size():
    source = torch.nn.Linear(16, 8)
    config = OrbitQuantConfig(adaln_group_size=1)

    quantized = RTNInt4Linear.from_linear(source, config=config, module_name="block.modulation")

    assert quantized.packed_weight.dtype == torch.uint8
