"""FLUX.3 Klein (Black Forest Labs' native ``flux3.dit.DiT``) per-mode and single-stream blocks.

Every block is one parallel attention + SwiGLU block: LayerNorm and AdaLN modulation run in the
activation-quantization prologue, Q|K|V and the SwiGLU gate|up rows are two GEMMs over one set of
codes, Q/K RMSNorm and rotary embedding are one kernel, and both output projections add their
gated residual in the GEMM epilogue. The blocks run one sample at a time, as the distilled model
does. The modulation of a mode is computed once per forward, on one row when every token of the
mode carries the same time embedding (the sampler fills a mode's timesteps with one value).
"""

from __future__ import annotations

import types

import torch
import torch.nn.functional as F

from orbitquant.fused.families import int8_attention_allowed, specs
from orbitquant.fused.groups import RowSource
from orbitquant.kernels import triton_activation as act
from orbitquant.kernels import triton_int8_gemm as gemm

FAMILY = "flux3"
MODEL_CLASSES = ("DiT",)
EPS = 1e-6


def blocks(model):
    paths = [
        f"content_mode_blocks.{mode}.{i}"
        for mode in model.content_mode_blocks
        for i in range(len(model.content_mode_blocks[mode]))
    ]
    paths += [f"txt_mode_blocks.{i}" for i in range(len(model.txt_mode_blocks))]
    paths += [f"single_blocks.{i}" for i in range(len(model.single_blocks))]
    return [(path, "block") for path in paths]


def groups(kind, block):
    hidden = block.mlp_out.in_features
    return specs(
        qkv=["q_proj", "k_proj", "v_proj"],
        mlp=(
            [RowSource("mlp_in", 0, hidden), RowSource("mlp_in", hidden, 2 * hidden)],
            {"swiglu": True},
        ),
        attn_out=["attn_out"],
        mlp_out=["mlp_out"],
    )


def prepare(block, kind):
    norm = block.norm
    block.oq_int8_attention = int8_attention_allowed(norm.query_norm.scale, norm.key_norm.scale)


def forward(kind):
    return _block


def install(model):
    """Per-mode modulation returns contiguous bf16 shift, scale and gate once per forward."""
    for module in model.modules():
        if type(module).__name__ == "ModeModulation":
            module.forward = types.MethodType(_modulation, module)


def _modulation(self, vec):
    # One host sync per mode, before the blocks run. A uniform mode costs one row, returned
    # expanded over the tokens: the single-stream concatenation still sees one row per token.
    rows = vec.shape[0]
    uniform = rows > 1 and torch.equal(vec, vec[:1].expand_as(vec))
    chunks = self.lin(F.silu(vec[:1] if uniform else vec)).chunk(3, dim=-1)
    chunks = [c.to(torch.bfloat16).contiguous() for c in chunks]
    return tuple(c.expand(rows, -1) if uniform else c for c in chunks)


def _index(runtime, seq, rows, device):
    """Modulation row of every token: all zeros for a one-row modulation, else the token index."""
    cache = runtime.__dict__.setdefault("flux3_index", {})
    key = (seq, rows == 1, device)
    index = cache.get(key)
    if index is None:
        make = torch.zeros if rows == 1 else torch.arange
        index = cache[key] = make(seq, device=device, dtype=torch.int32)
    return index


def _rope(runtime, freqs):
    """Rotation matrices ``[1, 1, L, 64, 2, 2]`` as per-channel cos and sin ``[L, 128]``."""
    cached = runtime.__dict__.get("flux3_rope")
    if cached is not None and cached[0] is freqs:
        return cached[1], cached[2]
    matrices = freqs[0, 0]
    cos = matrices[..., 0, 0].repeat_interleave(2, dim=-1).contiguous()
    sin = matrices[..., 1, 0].repeat_interleave(2, dim=-1).contiguous()
    runtime.__dict__["flux3_rope"] = (freqs, cos, sin)
    return cos, sin


def _block(self, x, freqs, bounds, mod):
    if len(bounds) != 1:
        raise NotImplementedError("fused FLUX.3 blocks run one sample at a time")
    start, end = bounds[0]
    if start == end:
        return x  # the phantom token of an empty mode; its result is never read
    runtime, fused = self.oq_runtime, self.oq_fused
    shift, scale, gate = mod
    if scale.stride(0) == 0:  # one modulation row expanded over the tokens
        shift, scale, gate = shift[:1], scale[:1], gate[:1]
    h = x.to(torch.bfloat16).clone()
    seq, dim = h.shape
    index = _index(runtime, seq, shift.shape[0], h.device)
    source = runtime.input(
        h,
        dim,
        norm=act.NORM_LAYER,
        mod=act.MOD_SHIFT_SCALE,
        mod_scale=scale,
        mod_shift=shift,
        mod_index=index,
        rms_eps=EPS,
    )
    inner = fused.qkv.out_features // 3
    heads = self.heads
    head_dim = inner // heads
    qkv = runtime.linear(fused.qkv, source)
    mid = runtime.linear(fused.mlp, source, epilogue=gemm.EPILOGUE_SWIGLU)
    cos, sin = _rope(runtime, freqs)
    query = torch.empty((1, seq, heads, head_dim), device=h.device, dtype=torch.bfloat16)
    key = torch.empty_like(query)
    for src, weight, out in (
        (qkv[:, :inner], self.norm.query_norm.scale, query),
        (qkv[:, inner : 2 * inner], self.norm.key_norm.scale, key),
    ):
        act.qk_norm_rope_with_triton(
            src,
            heads,
            head_dim,
            weight,
            cos,
            sin,
            EPS,
            weight_mode=act.QK_WEIGHT_TORCH,
            style=act.ROPE_INTERLEAVED,
            out=out,
        )
    value = qkv[:, 2 * inner :].unflatten(-1, (heads, head_dim))[None]
    attention = runtime.attention(query, key, value, int8_allowed=self.oq_int8_attention)
    attention = attention.reshape(seq, inner)
    runtime.linear(
        fused.attn_out,
        runtime.input(attention, inner),
        epilogue=gemm.EPILOGUE_RESIDUAL_GATE,
        residual=h,
        gate=gate,
        gate_index=index,
        out=h,
    )
    runtime.linear(
        fused.mlp_out,
        runtime.input(mid, mid.shape[1]),
        epilogue=gemm.EPILOGUE_RESIDUAL_GATE,
        residual=h,
        gate=gate,
        gate_index=index,
        out=h,
    )
    return h

