import pytest
import torch

from orbitquant.adaln import RTNInt4Linear
from orbitquant.config import OrbitQuantConfig
from orbitquant.layers import OrbitQuantLinear


def _build_layers() -> tuple[OrbitQuantLinear, RTNInt4Linear]:
    torch.manual_seed(0)
    config = OrbitQuantConfig(
        weight_bits=4,
        activation_bits=4,
        block_size=8,
        runtime_mode="dequant_bf16",
    )
    orbit = OrbitQuantLinear.from_linear(
        torch.nn.Linear(32, 24, bias=True),
        config=config,
        module_name="orbit",
    )
    adaln = RTNInt4Linear.from_linear(
        torch.nn.Linear(32, 16, bias=True),
        config=config,
        module_name="adaln",
    )
    return orbit, adaln


def test_orbit_linear_compiles_fullgraph_and_matches_eager():
    orbit, _ = _build_layers()
    x = torch.randn(3, 32)
    eager = orbit(x)

    compiled = torch.compile(orbit, fullgraph=True)
    compiled_out = compiled(x)

    assert compiled_out.shape == eager.shape
    assert compiled_out.dtype == eager.dtype
    assert torch.equal(compiled_out, eager)


def test_adaln_linear_compiles_fullgraph_and_matches_eager():
    _, adaln = _build_layers()
    x = torch.randn(3, 32)
    eager = adaln(x)

    compiled = torch.compile(adaln, fullgraph=True)
    compiled_out = compiled(x)

    assert compiled_out.shape == eager.shape
    assert compiled_out.dtype == torch.bfloat16
    assert torch.equal(compiled_out, eager)


def test_compiled_stack_of_quantized_layers_matches_eager():
    orbit, adaln = _build_layers()

    class Stack(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.orbit = orbit
            self.adaln = adaln

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.adaln(x).float().sum() + self.orbit(x).float().sum()

    model = Stack()
    x = torch.randn(2, 32)
    eager = model(x)
    compiled = torch.compile(model, fullgraph=True)
    out = compiled(x)
    assert torch.allclose(out, eager)


def test_compiled_handle_survives_repeated_calls():
    orbit, _ = _build_layers()
    compiled = torch.compile(orbit, fullgraph=True)
    x = torch.randn(4, 32)
    first = compiled(x)
    second = compiled(x)
    assert torch.equal(first, second)


@pytest.mark.parametrize("rows", [1, 5])
def test_compiled_orbit_linear_handles_leading_batch_dims(rows):
    orbit, _ = _build_layers()
    compiled = torch.compile(orbit, fullgraph=True)
    x = torch.randn(rows, 2, 32)
    assert torch.equal(compiled(x), orbit(x))
