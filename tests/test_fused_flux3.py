import types

import pytest
import torch
from torch import nn

pytest.importorskip("triton")
from orbitquant.fused.families import adapter_for_class, adapter_for_family, flux3  # noqa: E402


def test_family_is_registered_for_the_native_transformer_class():
    assert adapter_for_family("flux3") is flux3
    assert adapter_for_class("DiT") is flux3


def _stub_model(modes=("image", "image_cond"), depth=2, single=3):
    def block():
        return nn.Module()

    model = nn.Module()
    model.content_mode_blocks = nn.ModuleDict(
        {mode: nn.ModuleList([block() for _ in range(depth)]) for mode in modes}
    )
    model.txt_mode_blocks = nn.ModuleList([block() for _ in range(depth)])
    model.single_blocks = nn.ModuleList([block() for _ in range(single)])
    return model


def test_every_block_of_every_mode_is_listed_once():
    paths = [path for path, _ in flux3.blocks(_stub_model())]
    assert paths == [
        "content_mode_blocks.image.0",
        "content_mode_blocks.image.1",
        "content_mode_blocks.image_cond.0",
        "content_mode_blocks.image_cond.1",
        "txt_mode_blocks.0",
        "txt_mode_blocks.1",
        "single_blocks.0",
        "single_blocks.1",
        "single_blocks.2",
    ]


def test_groups_split_mlp_in_into_gate_and_up_halves():
    block = nn.Module()
    block.mlp_out = nn.Linear(9216, 3072, bias=False)
    specs = flux3.groups("block", block)
    assert list(specs) == ["qkv", "mlp", "attn_out", "mlp_out"]
    gate, up = specs["mlp"].sources
    assert (gate.path, gate.start, gate.stop) == ("mlp_in", 0, 9216)
    assert (up.path, up.start, up.stop) == ("mlp_in", 9216, 18432)
    assert specs["mlp"].swiglu and not specs["qkv"].swiglu


def _reference_rope(pos, dim, theta):
    scale = torch.arange(0, dim, 2, dtype=torch.float64) / dim
    out = torch.einsum("...n,d->...nd", pos, 1.0 / (theta**scale))
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    return out.reshape(*out.shape[:-1], 2, 2).float()


def test_rotation_matrices_become_per_channel_cos_and_sin():
    positions = torch.arange(5, dtype=torch.float32)
    ids = torch.stack([torch.zeros(5), positions, positions * 0.5, positions], dim=-1)
    matrices = torch.cat([_reference_rope(ids[..., i], 32, 10000) for i in range(4)], dim=-3)
    freqs = matrices.unsqueeze(0).unsqueeze(0)
    runtime = types.SimpleNamespace(__dict__={})
    cos, sin = flux3._rope(runtime, freqs)
    assert cos.shape == sin.shape == (5, 128)
    assert torch.equal(cos[:, 0::2], cos[:, 1::2])
    assert torch.equal(sin[:, 0::2], sin[:, 1::2])
    assert torch.equal(cos[:, ::2], matrices[..., 0, 0])
    assert torch.equal(sin[:, ::2], matrices[..., 1, 0])
    assert flux3._rope(runtime, freqs)[0] is cos  # cached per forward


def test_modulation_is_contiguous_bf16_chunks():
    class ModeModulation(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(8, 24, bias=False)

    model = nn.Module()
    model.mod = ModeModulation()
    flux3.install(model)
    shift, scale, gate = model.mod.forward(torch.randn(5, 8))
    for chunk in (shift, scale, gate):
        assert chunk.dtype == torch.bfloat16 and chunk.is_contiguous() and chunk.shape == (5, 8)


def test_multiple_samples_are_rejected():
    block = nn.Module()
    with pytest.raises(NotImplementedError, match="one sample"):
        flux3._block(block, torch.zeros(2, 4), None, [(0, 1), (1, 2)], None)


def _installed_modulation():
    class ModeModulation(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(8, 24, bias=False)

    model = nn.Module()
    model.mod = ModeModulation()
    flux3.install(model)
    return model.mod


def test_uniform_mode_computes_one_row_expanded_over_the_tokens():
    module = _installed_modulation()
    vec = torch.randn(1, 8).expand(6, 8).clone()
    chunks = module.forward(vec)
    projected = module.lin(torch.nn.functional.silu(vec[:1])).expand(6, 24)
    reference = [c.to(torch.bfloat16) for c in projected.chunk(3, -1)]
    for chunk, ref in zip(chunks, reference, strict=True):
        assert chunk.shape == (6, 8) and chunk.stride(0) == 0
        assert torch.equal(chunk, ref)
        # The single-stream concatenation of slices materializes one row per token.
        assert torch.equal(torch.cat([chunk[0:4], chunk[4:6]]), ref)


def test_varying_rows_keep_one_modulation_row_per_token():
    module = _installed_modulation()
    shift, _, _ = module.forward(torch.randn(5, 8))
    assert shift.is_contiguous() and shift.stride(0) == 8


def test_token_index_is_cached_per_length_and_kind():
    runtime = types.SimpleNamespace(__dict__={})
    zeros = flux3._index(runtime, 4, 1, torch.device("cpu"))
    tokens = flux3._index(runtime, 4, 4, torch.device("cpu"))
    assert torch.equal(zeros, torch.zeros(4, dtype=torch.int32))
    assert torch.equal(tokens, torch.arange(4, dtype=torch.int32))
    assert flux3._index(runtime, 4, 1, torch.device("cpu")) is zeros

