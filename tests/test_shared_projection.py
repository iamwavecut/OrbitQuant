import pytest
import torch

from orbitquant import layers
from orbitquant.config import OrbitQuantConfig


def _layers():
    return [
        layers.OrbitQuantLinear.from_linear(
            torch.nn.Linear(128, 64), config=OrbitQuantConfig(), module_name=f"qkv.{i}"
        ).eval()
        for i in range(3)
    ]


def test_inference_reuse_is_bounded_and_refreshes_after_mutation(monkeypatch):
    projections = _layers()
    calls = []

    def forward(self, x):
        def prepare():
            calls.append(1)
            return x.clone(), x.sum()

        (prepared, _), _ = layers._get_or_create_shared_activation_quantization(
            x, ("test",), prepare
        )
        return prepared.clone()

    monkeypatch.setattr(layers.OrbitQuantLinear, "_forward_impl", forward)
    layers._clear_shared_activation_cache()
    with torch.inference_mode():
        x = torch.ones(1, 128)
        result = layers.project_shared_input(x, *projections)
        assert len(calls) == 1
        assert all(torch.equal(y, x) for y in result)
        assert getattr(layers._SHARED_ACTIVATION_CACHE, "projection_input", None) is None
        x.add_(1)
        result = layers.project_shared_input(x, *projections)
        assert len(calls) == 2
        assert all(torch.equal(y, x) for y in result)
        projections[0](x)
        projections[1](x)
        assert len(calls) == 4


def test_hooks_disable_inference_reuse(monkeypatch):
    projections = _layers()
    calls = []

    def forward(self, x):
        def prepare():
            calls.append(1)
            return x.clone(), x.sum()

        return layers._get_or_create_shared_activation_quantization(x, (), prepare)[0][0]

    monkeypatch.setattr(layers.OrbitQuantLinear, "_forward_impl", forward)
    handle = projections[1].register_forward_pre_hook(lambda module, args: args[0].add_(1))
    with torch.inference_mode():
        outputs = layers.project_shared_input(torch.ones(1, 128), *projections)
    handle.remove()
    assert len(calls) == 3
    assert torch.equal(outputs[0], torch.ones(1, 128))
    assert torch.equal(outputs[1], torch.full((1, 128), 2.0))


def test_projection_exception_clears_scope(monkeypatch):
    def fail(self, x):
        raise RuntimeError("projection failed")

    projections = _layers()
    monkeypatch.setattr(layers.OrbitQuantLinear, "_forward_impl", fail)
    with torch.inference_mode(), pytest.raises(RuntimeError, match="projection failed"):
        layers.project_shared_input(torch.ones(1, 128), *projections)
    assert getattr(layers._SHARED_ACTIVATION_CACHE, "projection_input", None) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_real_cuda_projection_reuse_is_bit_identical():
    projections = [layer.cuda() for layer in _layers()]
    with torch.inference_mode():
        x = torch.randn(1, 257, 128, device="cuda", dtype=torch.bfloat16)
        expected = tuple(layer(x) for layer in projections)
        actual = layers.project_shared_input(x, *projections)
        assert [layer.last_activation_cache_hit for layer in projections] == [False, True, True]
        assert all(torch.equal(a, b) for a, b in zip(expected, actual, strict=True))


def test_incompatible_quantization_signatures_do_not_share(monkeypatch):
    projections = _layers()
    calls = []

    def forward(self, x):
        def prepare():
            calls.append(self.module_name)
            return x.clone(), x.sum()

        return layers._get_or_create_shared_activation_quantization(
            x, (self.module_name,), prepare
        )[0][0]

    monkeypatch.setattr(layers.OrbitQuantLinear, "_forward_impl", forward)
    with torch.inference_mode():
        layers.project_shared_input(torch.ones(1, 128), *projections)
    assert len(calls) == 3
