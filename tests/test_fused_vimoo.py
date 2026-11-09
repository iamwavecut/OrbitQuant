import types

import pytest
import torch
import torch.nn.functional as F
from torch import nn

pytest.importorskip("triton")
from orbitquant.fused.families import adapter_for_class, adapter_for_family, vimoo  # noqa: E402
from orbitquant.kernels import available_backends  # noqa: E402


def _require_cuda():
    if not torch.cuda.is_available() or not available_backends()["triton_cuda"]:
        pytest.skip("CUDA/Triton backend is not available")


def _vimoo_module():
    return pytest.importorskip(
        "diffusers.models.transformers.transformer_vimoo",
        reason="Vimoo transformer needs the Vimoo-AI diffusers fork",
    )


def test_family_is_registered_for_the_transformer_class():
    assert adapter_for_family("vimoo") is vimoo
    assert adapter_for_class("VimooTransformer3DModel") is vimoo


def test_blocks_follow_the_numerical_profile():
    model = nn.Module()
    model.double_blocks = nn.ModuleList([nn.Module() for _ in range(3)])
    model.numerical_profile = "lite"
    assert vimoo.blocks(model) == [(f"double_blocks.{i}", "lite") for i in range(3)]
    model.numerical_profile = "standard"
    assert [k for _, k in vimoo.blocks(model)] == ["standard"] * 3


def test_groups_cover_both_streams():
    standard = vimoo.groups("standard", None)
    lite = vimoo.groups("lite", None)
    assert list(standard) == list(lite) == list(vimoo.GROUPS)
    assert [s.path for s in standard["img_qkv"].sources] == ["attn.to_q", "attn.to_k", "attn.to_v"]
    assert [s.path for s in standard["txt_qkv"].sources] == [
        "attn.add_q_proj",
        "attn.add_k_proj",
        "attn.add_v_proj",
    ]
    assert standard["img_ff_in"].sources[0].path == "img_mlp.net.0.proj"
    assert standard["txt_ff_out"].sources[0].path == "txt_mlp.net.2"
    assert lite["img_out"].sources[0].path == "attn.to_out.0.projection"
    assert lite["txt_ff_in"].sources[0].path == "txt_mlp.in_proj.projection"
    assert not any(spec.swiglu for spec in standard.values())
    assert set(vimoo.ACTIVATION_BITS) == set(vimoo.GROUPS)
    assert set(vimoo.ACTIVATION_BITS.values()) == {8}


def test_valid_text_must_be_a_prefix():
    runtime = types.SimpleNamespace(__dict__={})
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])
    assert vimoo._valid_text(runtime, mask, 2, 5) == [3, 2]
    assert vimoo._valid_text(runtime, mask, 2, 5) == [3, 2]
    with pytest.raises(ValueError, match="valid text tokens first"):
        vimoo._valid_text(types.SimpleNamespace(__dict__={}), torch.tensor([[1, 0, 1]]), 1, 3)
    assert vimoo._valid_text(runtime, None, 2, 7) == [7, 7]


def test_gelu_erf_epilogue_matches_eager():
    _require_cuda()
    from orbitquant.kernels import triton_int8_gemm as gemm

    torch.manual_seed(5)
    rows, k, n = 200, 256, 384
    a = torch.randint(-127, 128, (rows, k), device="cuda", dtype=torch.int8)
    b = torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8)
    a_s = torch.rand(rows, device="cuda") * 1e-3 + 1e-4
    b_s = torch.rand(n, device="cuda") + 0.5
    bias = torch.randn(n, device="cuda")
    y = (((a.float() @ b.float().T) * a_s[:, None]) * b_s[None, :] + bias).to(torch.bfloat16)
    expected = F.gelu(y)
    actual = gemm.matmul_int8_scaled_with_triton(
        a, b, a_s, b_s, bias=bias, epilogue=gemm.EPILOGUE_GELU_ERF
    )
    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=1e-2)


def _block(mod, lite, dim=256, heads=2):
    """A Vimoo block with random weights: Lite (FP32 rank-16 modulation, exact GELU, separate output
    biases) or standard (BF16 rank-64 modulation with FP32 bias, tanh GELU)."""
    torch.manual_seed(0)
    if lite:
        block = mod.VimooTransformerBlock(
            dim, heads, block_modulation_rank=16, fp32_modulation=True, compact=True
        )
    else:
        block = mod.VimooTransformerBlock(
            dim, heads, block_modulation_rank=64, mixed_precision_modulation=True
        )
    with torch.no_grad():
        for name, p in block.named_parameters():
            if "weight" in name and p.dim() == 2:
                p.copy_(torch.randn_like(p) / p.shape[1] ** 0.5)
            elif "norm" in name:
                p.copy_(torch.rand_like(p) * 0.5 + 0.75)
            else:
                p.copy_(torch.randn_like(p) * 0.05)
    block = block.to(torch.bfloat16)
    if lite:
        for m in (block.img_mod, block.txt_mod):
            m.float()
    return block


def _quantize(block, kind):
    from orbitquant import OrbitQuantConfig
    from orbitquant.fused.groups import _resolve
    from orbitquant.layers import OrbitQuantLinear

    config = OrbitQuantConfig(activation_bits=8)
    for spec in vimoo.groups(kind, block).values():
        for source in spec.sources:
            parent_path, _, child = source.path.rpartition(".")
            parent = _resolve(block, parent_path)
            linear = _resolve(block, source.path)
            q = OrbitQuantLinear.from_linear(linear, config=config, module_name=source.path)
            if child.isdigit():
                parent[int(child)] = q
            else:
                setattr(parent, child, q)
    return config


@pytest.mark.parametrize("lite", [False, True])
def test_fused_block_tracks_the_eager_block(lite):
    _require_cuda()
    mod = _vimoo_module()
    from orbitquant.fused.groups import PackedGroup
    from orbitquant.fused.install import _attach
    from orbitquant.fused.runtime import FusedRuntime

    kind = "lite" if lite else "standard"
    dim, heads = 256, 2
    block = _block(mod, lite, dim, heads).cuda()
    config = _quantize(block, kind)
    model = nn.Module()
    model.double_blocks = nn.ModuleList([block])
    batch, img, txt, valid = 2, 300, 40, (25, 31)
    # Column-major video stream, as the standard (direct) patch embedding hands it over.
    x = torch.randn(batch, dim, img, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    c = torch.randn(batch, txt, dim, device="cuda", dtype=torch.bfloat16)
    mask = torch.zeros(batch, txt, dtype=torch.int64, device="cuda")
    for b, n in enumerate(valid):
        mask[b, :n] = 1
    rank = 16 if lite else 64
    vec = torch.randn(batch, rank, device="cuda", dtype=torch.float32 if lite else torch.bfloat16)
    angles = torch.rand(img, dim // heads // 2, device="cuda") * 6.28
    freqs = (angles.cos().repeat_interleave(2, -1), angles.sin().repeat_interleave(2, -1))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ex_img, ex_txt = block(x, c, vec, freqs, mask, None, None, torch.bfloat16)
    runtime = FusedRuntime.from_config(config, attention="sdpa")
    modules = nn.ModuleDict(
        {
            name: PackedGroup.from_sources(block, list(spec.sources), activation_bits=8)
            for name, spec in vimoo.groups(kind, block).items()
        }
    )
    vimoo.prepare(block, kind)
    _attach(model, block, vimoo, kind, modules, runtime)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ac_img, ac_txt = block(x, c, vec, freqs, mask, None, None, torch.bfloat16)
    assert ac_img.shape == ex_img.shape and ac_txt.shape == ex_txt.shape
    upd, ref = (ac_img - x).float().flatten(), (ex_img - x).float().flatten()
    assert F.cosine_similarity(upd, ref, dim=0) > 0.98
    for b, n in enumerate(valid):
        upd = (ac_txt[b, :n] - c[b, :n]).float().flatten()
        ref = (ex_txt[b, :n] - c[b, :n]).float().flatten()
        assert F.cosine_similarity(upd, ref, dim=0) > 0.98
        # Padded text rows leave the fused block unchanged.
        assert torch.equal(ac_txt[b, n:], c[b, n:].to(ac_txt.dtype))
