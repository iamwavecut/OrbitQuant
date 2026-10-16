import types

import pytest
import torch
import torch.nn.functional as F
from torch import nn

pytest.importorskip("triton")
from orbitquant.fused.families import (  # noqa: E402
    adapter_for_class,
    adapter_for_family,
    moss_image,
)
from orbitquant.kernels import available_backends  # noqa: E402


def _require_cuda():
    if not torch.cuda.is_available() or not available_backends()["triton_cuda"]:
        pytest.skip("CUDA/Triton backend is not available")


# A copy of MOSS-Image's block (pipeline_moss_image.py), the eager reference of the fused forward.
class _RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        return y.to(x.dtype) * self.weight


def _situ(hidden, gate):
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(hidden / 25.0))


def _rotate(x, rope):
    sin, cos = rope.chunk(2, -1)
    sin = torch.cat((sin, sin), -1).unsqueeze(-2)
    cos = torch.cat((cos, cos), -1).unsqueeze(-2)
    a, b = x.chunk(2, -1)
    return x * cos + torch.cat((-b, a), -1) * sin


def _attention(q, k, v, rope, heads):
    shape = (*q.shape[:2], heads, q.shape[-1] // heads)
    q = _rotate(q.reshape(shape), rope).transpose(1, 2)
    k = _rotate(k.reshape(shape), rope).transpose(1, 2)
    v = v.reshape(shape).transpose(1, 2)
    return F.scaled_dot_product_attention(q, k, v).transpose(1, 2).flatten(2)


class _Block(nn.Module):
    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = _RMSNorm(dim), _RMSNorm(dim)
        self.q_norm, self.k_norm = _RMSNorm(dim // heads), _RMSNorm(dim // heads)
        self.proj = nn.ModuleDict(
            {name: nn.Linear(dim, dim, bias=False) for name in ("q", "k", "v", "o")}
        )
        self.proj["ffn_up_gate"] = nn.Linear(dim, 2 * ffn_dim, bias=False)
        self.proj["ffn_down"] = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x, rope, modulation):
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation.unbind(1)
        y = self.norm1(x) * (1 + scale_a[:, None]) + shift_a[:, None]
        q = self.q_norm(self.proj["q"](y).unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(self.proj["k"](y).unflatten(-1, (self.heads, -1))).flatten(2)
        x = (
            x
            + self.proj["o"](_attention(q, k, self.proj["v"](y), rope, self.heads))
            * gate_a[:, None]
        )
        y = self.norm2(x) * (1 + scale_f[:, None]) + shift_f[:, None]
        hidden, gate = self.proj["ffn_up_gate"](y).chunk(2, -1)
        return x + self.proj["ffn_down"](_situ(hidden, gate)) * gate_f[:, None]


def _rope_table(seq, bands=16, device="cpu"):
    """The model's ``[1, L, 128]`` sin | cos table (4 axes of 16 bands), cast to bf16 like the
    model."""
    positions = torch.randn(seq, 4, device=device) * 20
    freqs = 1 / (10000 ** (torch.arange(bands, device=device, dtype=torch.float32) / bands))
    phase = positions.unsqueeze(-1) * freqs
    return torch.cat((phase.sin(), phase.cos()), -2).flatten(-2).unsqueeze(0).to(torch.bfloat16)


def test_family_is_registered_for_the_transformer_class():
    assert adapter_for_family("moss_image") is moss_image
    assert adapter_for_class("MOSSImageTransformer") is moss_image


def test_blocks_and_groups():
    model = nn.Module()
    model.layers = nn.ModuleList([_Block(256, 512, 2) for _ in range(3)])
    assert [p for p, _ in moss_image.blocks(model)] == ["layers.0", "layers.1", "layers.2"]
    specs = moss_image.groups("block", model.layers[0])
    assert list(specs) == ["qkv", "mlp", "attn_out", "mlp_out"]
    gate, up = specs["mlp"].sources
    # ffn_up_gate rows are [hidden | gate]: the epilogue's gate input is the second half.
    assert (gate.path, gate.start, gate.stop) == ("proj.ffn_up_gate", 512, 1024)
    assert (up.path, up.start, up.stop) == ("proj.ffn_up_gate", 0, 512)
    assert specs["mlp"].swiglu and not specs["qkv"].swiglu


def test_rope_table_becomes_per_channel_cos_and_sin():
    rope = _rope_table(5)
    runtime = types.SimpleNamespace(__dict__={})
    cos, sin = moss_image._rope(runtime, rope)
    assert cos.shape == sin.shape == (5, 128)
    assert torch.equal(cos[:, :64], cos[:, 64:]) and torch.equal(sin[:, :64], sin[:, 64:])
    assert torch.equal(sin[:, :64], rope[0, :, :64].float())
    assert torch.equal(cos[:, :64], rope[0, :, 64:].float())
    assert moss_image._rope(runtime, rope)[0] is cos


def test_multiple_samples_are_rejected():
    with pytest.raises(NotImplementedError, match="one sample"):
        moss_image._block(nn.Module(), torch.zeros(2, 4, 8), None, None)


def test_situ_epilogue_matches_eager_rounding():
    _require_cuda()
    from orbitquant.kernels import triton_int8_gemm as gemm

    torch.manual_seed(3)
    rows, k, n = 300, 512, 1024
    a = torch.randint(-127, 128, (rows, k), device="cuda", dtype=torch.int8)
    b = torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8)
    a_s = torch.rand(rows, device="cuda") * 1e-3 + 1e-4
    b_s = torch.rand(n, device="cuda") + 0.5
    alpha = 0.37
    y = ((((a.float() @ b.float().T) * a_s[:, None]) * b_s[None, :]) * alpha).to(torch.bfloat16)
    half = gemm.SWIGLU_INTERLEAVE
    chunks = y.reshape(rows, -1, 2, half)
    gate, up = chunks[:, :, 0].reshape(rows, -1), chunks[:, :, 1].reshape(rows, -1)
    expected = _situ(up, gate)
    actual = gemm.matmul_int8_scaled_with_triton(
        a, b, a_s, b_s, alpha=alpha, epilogue=gemm.EPILOGUE_SITU
    )
    assert actual.shape == expected.shape
    # libdevice and torch tanh agree to an ulp of fp32, so a rare bf16 tie can round apart.
    mismatch = (actual != expected).float().mean().item()
    assert mismatch < 1e-3
    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=1e-2)


def _quantized_block(dim=256, ffn=512, heads=2):
    from orbitquant import OrbitQuantConfig
    from orbitquant.layers import OrbitQuantLinear

    torch.manual_seed(0)
    block = _Block(dim, ffn, heads).to(torch.bfloat16)
    with torch.no_grad():
        for norm in (block.norm1, block.norm2, block.q_norm, block.k_norm):
            norm.weight.copy_(torch.rand_like(norm.weight) * 0.5 + 0.75)
    reference = _Block(dim, ffn, heads).to(torch.bfloat16)
    reference.load_state_dict(block.state_dict())
    config = OrbitQuantConfig()
    for name, linear in list(block.proj.items()):
        block.proj[name] = OrbitQuantLinear.from_linear(
            linear, config=config, module_name=f"proj.{name}"
        )
    model = nn.Module()
    model.layers = nn.ModuleList([block])
    model.quantization_config = config
    type(model).__name__  # noqa: B018
    return model, reference


def test_fused_block_tracks_the_eager_block():
    _require_cuda()
    from orbitquant.fused.groups import PackedGroup
    from orbitquant.fused.install import _attach
    from orbitquant.fused.runtime import FusedRuntime

    model, reference = _quantized_block()
    model, reference = model.cuda(), reference.cuda()
    block = model.layers[0]
    runtime = FusedRuntime.from_config(model.quantization_config, attention="sdpa")
    modules = nn.ModuleDict(
        {
            name: PackedGroup.from_sources(
                block, list(spec.sources), swiglu=spec.swiglu, activation_bits=8
            )
            for name, spec in moss_image.groups("block", block).items()
        }
    )
    moss_image.prepare(block, "block")
    _attach(model, block, moss_image, "block", modules, runtime)
    seq, dim = 300, 256
    x = torch.randn(1, seq, dim, device="cuda", dtype=torch.bfloat16)
    rope = _rope_table(seq, device="cuda")
    modulation = (torch.randn(1, 6, dim, device="cuda") * 0.3).to(torch.bfloat16)
    with torch.no_grad():
        expected = reference(x, rope, modulation)
        actual = block(x, rope, modulation)
    assert actual.shape == expected.shape and actual.dtype == torch.bfloat16
    update, ref_update = (actual - x).float().flatten(), (expected - x).float().flatten()
    assert F.cosine_similarity(update, ref_update, dim=0) > 0.98


def test_fuse_linears_replaces_projections_with_int8_gemms():
    _require_cuda()
    from orbitquant import OrbitQuantConfig
    from orbitquant.fused import FusedLinear, fuse_linears, release
    from orbitquant.layers import OrbitQuantLinear

    torch.manual_seed(1)
    config = OrbitQuantConfig()
    dense = nn.Sequential(
        nn.Linear(4096, 1024, bias=False), nn.SiLU(), nn.Linear(1024, 4096, bias=True)
    ).to(torch.bfloat16)
    model = nn.Sequential(*[m for m in dense])
    for index in (0, 2):
        model[index] = OrbitQuantLinear.from_linear(
            dense[index], config=config, module_name=str(index)
        )
    model = model.cuda()
    dense = dense.cuda()
    model.quantization_config = config
    assert fuse_linears(model) == 2
    assert isinstance(model[0], FusedLinear) and isinstance(model[2], FusedLinear)
    x = torch.randn(2, 37, 4096, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        actual, expected = model(x), dense(x)
    assert actual.shape == expected.shape == (2, 37, 4096)
    assert F.cosine_similarity(actual.float().flatten(), expected.float().flatten(), dim=0) > 0.98
    release(model)
