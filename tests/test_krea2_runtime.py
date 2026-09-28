import types

import pytest
import torch

pytest.importorskip("triton")


def _quantized(in_features, out_features, seed):
    from orbitquant import OrbitQuantConfig
    from orbitquant.layers import OrbitQuantLinear

    torch.manual_seed(seed)
    linear = torch.nn.Linear(in_features, out_features, bias=False, dtype=torch.bfloat16)
    return OrbitQuantLinear.from_linear(linear, config=OrbitQuantConfig(), module_name=f"l{seed}")


def _block(seed, dim=256, heads=2, kv_heads=1, head_dim=128, hidden=512):
    attn = types.SimpleNamespace(
        to_q=_quantized(dim, heads * head_dim, seed),
        to_k=_quantized(dim, kv_heads * head_dim, seed + 1),
        to_v=_quantized(dim, kv_heads * head_dim, seed + 2),
        to_gate=_quantized(dim, heads * head_dim, seed + 3),
        to_out=[_quantized(heads * head_dim, dim, seed + 4)],
        num_heads=heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        norm_q=types.SimpleNamespace(weight=torch.randn(head_dim) * 0.1),
        norm_k=types.SimpleNamespace(weight=torch.randn(head_dim) * 0.1),
    )
    down = torch.nn.Linear(hidden, dim, bias=False, dtype=torch.bfloat16)
    ff = types.SimpleNamespace(
        gate=_quantized(dim, hidden, seed + 5), up=_quantized(dim, hidden, seed + 6), down=down
    )
    return types.SimpleNamespace(attn=attn, ff=ff, norm1=types.SimpleNamespace(eps=1e-6))


def _transformer():
    return types.SimpleNamespace(transformer_blocks=[_block(10), _block(20)])


def test_fused_weights_round_trip_through_a_mapped_file(tmp_path):
    from orbitquant.runtime import krea2

    built = krea2.FusedKrea2Blocks(_transformer())
    path = tmp_path / "fused.safetensors"
    krea2.save_fused(built, path)

    target = _transformer()
    loaded = krea2.install(target, fused_path=path)

    assert target.fused_blocks is loaded
    assert isinstance(target.transformer_blocks[0].attn.to_q, torch.nn.Identity)
    expected, actual = built.state_dict(), loaded.state_dict()
    assert expected.keys() == actual.keys()
    for key in expected:
        assert torch.equal(expected[key], actual[key]), key
    for a, b in zip(built.fused, loaded.fused, strict=True):
        assert (a.qkvg.alpha, a.out.alpha, a.gateup.alpha) == (
            b.qkvg.alpha,
            b.out.alpha,
            b.gateup.alpha,
        )


def test_fused_file_must_match_the_down_mode(tmp_path):
    from orbitquant.runtime import krea2

    path = tmp_path / "fused.safetensors"
    krea2.save_fused(krea2.FusedKrea2Blocks(_transformer(), down="int8"), path)
    with pytest.raises(ValueError, match="down"):
        krea2.install(_transformer(), down="bf16", fused_path=path)
