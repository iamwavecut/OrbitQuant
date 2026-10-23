"""Fibo 2.0 (Bria ``BriaFibo2Transformer2DModel``) blocks.

The noise refiner and the 44 main blocks are Z-Image blocks: RMSNorm x (1 + scale) prologue, one
Q|K|V GEMM, Q/K RMSNorm with interleaved complex RoPE in one kernel, sandwich post-norms with
tanh gates as row kernels and a SwiGLU feed-forward. The main blocks modulate per token: gist
and context tokens take the clean (t = 0) rows, image tokens the noisy rows, so the prologue and
the post-norm kernels read their modulation from a two-row table by token index. Five injection
blocks add a gated cross-attention to their own Qwen3-VL bundle (7680 channels) between the
attention and the feed-forward. The two Perceiver layers turn the first bundle into the gist
tokens with plain residuals.

Every group takes 8-bit activations (per-token absmax INT8 of the rotated input): the
feed-forward output projection reads 9536 channels, whose RPBH block is 64 wide, too narrow to
spread outliers for 4-bit codes; the GEMMs are INT8 either way.
"""

from __future__ import annotations

import torch

from orbitquant.fused.families import int8_attention_allowed, specs
from orbitquant.kernels import triton_activation as act
from orbitquant.kernels import triton_int8_gemm as gemm

FAMILY = "fibo2"
MODEL_CLASSES = ("BriaFibo2Transformer2DModel",)

_BLOCK_GROUPS = ("qkv", "attn_out", "ff_in", "ff_out")
_CROSS_GROUPS = ("xq", "xkv", "xout")
ACTIVATION_BITS = dict.fromkeys(_BLOCK_GROUPS + _CROSS_GROUPS, 8)


def blocks(model):
    paths = [(f"noise_refiner.{i}", "block") for i in range(len(model.noise_refiner))]
    paths += [(f"perceiver.layers.{i}", "perceiver") for i in range(len(model.perceiver.layers))]
    for i, block in enumerate(model.layers):
        paths.append((f"layers.{i}", "inject" if block.cross_attention is not None else "block"))
    return paths


def _cross_groups(prefix="cross_attention"):
    return dict(
        xq=[f"{prefix}.to_q"],
        xkv=[f"{prefix}.to_k", f"{prefix}.to_v"],
        xout=[f"{prefix}.to_out.0"],
    )


def groups(kind, block):
    if kind == "perceiver":
        return specs(
            **_cross_groups(),
            qkv=["self_attention.to_q", "self_attention.to_k", "self_attention.to_v"],
            attn_out=["self_attention.to_out.0"],
            ff_in=(["ffn.w1", "ffn.w3"], {"swiglu": True}),
            ff_out=["ffn.w2"],
        )
    out = dict(
        qkv=["attention.to_q", "attention.to_k", "attention.to_v"],
        attn_out=["attention.to_out.0"],
        ff_in=(["feed_forward.w1", "feed_forward.w3"], {"swiglu": True}),
        ff_out=["feed_forward.w2"],
    )
    if kind == "inject":
        out.update(_cross_groups())
    return specs(**out)


def prepare(block, kind):
    attn = block.self_attention if kind == "perceiver" else block.attention
    block.oq_int8_attention = int8_attention_allowed(attn.norm_q.weight, attn.norm_k.weight)


def forward(kind):
    return _perceiver if kind == "perceiver" else _block


def _cached(runtime, name, key, build, slots=2):
    """A small per-forward cache keyed by tensor identity (rope tables, token indices): the
    tensors are shared by every block of one transformer call."""
    cache = runtime.__dict__.setdefault(name, [])
    for entry in cache:
        if entry[0] is key:
            return entry[1]
    value = build()
    cache.insert(0, (key, value))
    del cache[slots:]
    return value


def _rope(runtime, freqs_cis, rows):
    """Complex ``[1, seq, head_dim / 2]`` frequencies -> interleaved fp32 cos/sin rows."""

    def build():
        freqs = freqs_cis.reshape(-1, freqs_cis.shape[-1])
        return (
            freqs.real.float().repeat_interleave(2, -1).contiguous(),
            freqs.imag.float().repeat_interleave(2, -1).contiguous(),
        )

    cos, sin = _cached(runtime, "fibo2_rope", freqs_cis, build)
    return cos[:rows], sin[:rows]


def _token_index(runtime, noise_mask, sample):
    """Row ``i`` of the modulation tables for every token: 0 noisy, 1 clean."""
    if noise_mask is None:
        return None
    index = _cached(runtime, "fibo2_index", noise_mask, lambda: (noise_mask == 0).to(torch.int32))
    return index[sample].contiguous()


def _modulation(block, adaln):
    """``(scale_msa, gate_msa, scale_mlp, gate_mlp)`` rows of one timestep embedding, with the
    eager ``1 +`` and ``tanh`` applied."""
    scale_msa, gate_msa, scale_mlp, gate_mlp = block.adaLN_modulation(adaln).chunk(4, dim=1)
    return 1.0 + scale_msa, gate_msa.tanh(), 1.0 + scale_mlp, gate_mlp.tanh()


def _tables(block, adaln_input, noise_mask, adaln_noisy, adaln_clean, sample):
    """Modulation tables ``[modes, dim]`` of one sample: one row (global) or noisy | clean."""
    if noise_mask is None:
        rows = _modulation(block, adaln_input[sample : sample + 1])
    else:
        noisy = _modulation(block, adaln_noisy[sample : sample + 1])
        clean = _modulation(block, adaln_clean[sample : sample + 1])
        rows = [torch.cat(pair) for pair in zip(noisy, clean, strict=True)]
    return [r.to(torch.bfloat16).contiguous() for r in rows]


def _valid(mask, rows):
    """Indices of the real keys of one sample's ``[seq]`` padding mask (``None``: all real)."""
    if mask is None:
        return None
    valid = mask.reshape(-1)[:rows].bool()
    return None if bool(valid.all()) else valid


def _heads(src, norm, heads, cos=None, sin=None):
    return act.qk_norm_rope_with_triton(
        src,
        heads,
        src.shape[1] // heads,
        norm.weight,
        cos,
        sin,
        norm.eps,
        weight_mode=act.QK_WEIGHT_DIFFUSERS,
        style=act.ROPE_INTERLEAVED,
    )


def _attend(block, attn, query_src, kv_src, inner, *, cos=None, sin=None, valid=None, int8=True):
    """Q/K RMSNorm (+ RoPE), attention; ``query_src`` holds Q, ``kv_src`` K | V."""
    runtime = block.oq_runtime
    heads = attn.heads
    head_dim = inner // heads
    query = _heads(query_src, attn.norm_q, heads, cos, sin)
    key = _heads(kv_src[:, :inner], attn.norm_k, heads, cos, sin)
    value = kv_src[:, inner : 2 * inner].unflatten(-1, (heads, head_dim))[None]
    if valid is not None:
        key, value = key[:, valid], value[:, valid]
    out = runtime.attention(query, key, value, int8_allowed=int8 and block.oq_int8_attention)
    return out.reshape(query.shape[1], inner)


def _self_attention(block, attn, fused, source, cos, sin, valid):
    runtime = block.oq_runtime
    qkv = runtime.linear(fused.qkv, source)
    inner = qkv.shape[1] // 3
    return _attend(
        block, attn, qkv[:, :inner], qkv[:, inner:], inner, cos=cos, sin=sin, valid=valid
    )


def _cross_attention(block, attn, fused, x, norm, text, text_valid):
    """``attn(norm(x), text)`` of one sample, before its output projection."""
    runtime = block.oq_runtime
    dim = x.shape[1]
    query = runtime.linear(
        fused.xq,
        runtime.input(x, dim, norm=act.NORM_RMS, norm_weight=norm.weight, rms_eps=norm.eps),
    )
    kv = runtime.linear(fused.xkv, runtime.input(text, text.shape[1]))
    return _attend(block, attn, query, kv, query.shape[1], valid=text_valid, int8=False)


def _swiglu(runtime, fused, source):
    """SwiGLU hidden state as the input of ``ff_out``. The hidden width (9536) is not a whole
    number of interleave chunks: the gated GEMM pads it with zero columns, which the strided
    view leaves out."""
    mid = runtime.linear(fused.ff_in, source, epilogue=gemm.EPILOGUE_SWIGLU)
    width = fused.ff_out.in_features
    return runtime.input(mid[:, :width], width)


def _block(
    self,
    x,
    attn_mask,
    freqs_cis,
    adaln_input=None,
    noise_mask=None,
    adaln_noisy=None,
    adaln_clean=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
):
    runtime, fused = self.oq_runtime, self.oq_fused
    batch, seq, dim = x.shape
    cos, sin = _rope(runtime, freqs_cis, seq)
    out = torch.empty_like(x, dtype=torch.bfloat16)
    for sample in range(batch):
        h = out[sample]
        h.copy_(x[sample])
        index = _token_index(runtime, noise_mask, sample)
        scale_msa, gate_msa, scale_mlp, gate_mlp = _tables(
            self, adaln_input, noise_mask, adaln_noisy, adaln_clean, sample
        )
        prologue = dict(norm=act.NORM_RMS, mod=act.MOD_SCALE, mod_index=index)
        norm = self.attention_norm1
        source = runtime.input(
            h, dim, norm_weight=norm.weight, rms_eps=norm.eps, mod_scale=scale_msa, **prologue
        )
        valid = _valid(None if attn_mask is None else attn_mask[sample], seq)
        att = _self_attention(self, self.attention, fused, source, cos, sin, valid)
        att_out = runtime.linear(fused.attn_out, runtime.input(att, att.shape[1]))
        norm = self.attention_norm2
        act.postnorm_residual_with_triton(
            h, att_out, norm.weight, gate_msa, norm.eps, gate_index=index
        )
        if self.cross_attention is not None:
            text = encoder_hidden_states[sample]
            text_valid = _valid(
                None if encoder_attention_mask is None else encoder_attention_mask[sample],
                text.shape[0],
            )
            cross = _cross_attention(
                self, self.cross_attention, fused, h, self.cross_attn_norm1, text, text_valid
            )
            cross_out = runtime.linear(fused.xout, runtime.input(cross, cross.shape[1]))
            gate = self.cross_attn_gate.tanh().to(torch.bfloat16).expand(dim)
            norm = self.cross_attn_norm2
            act.postnorm_residual_with_triton(h, cross_out, norm.weight, gate, norm.eps)
        norm = self.ffn_norm1
        source = runtime.input(
            h, dim, norm_weight=norm.weight, rms_eps=norm.eps, mod_scale=scale_mlp, **prologue
        )
        ff_out = runtime.linear(fused.ff_out, _swiglu(runtime, fused, source))
        norm = self.ffn_norm2
        act.postnorm_residual_with_triton(
            h, ff_out, norm.weight, gate_mlp, norm.eps, gate_index=index
        )
    return out


def _perceiver(
    self, latents, encoder_hidden_states, encoder_attention_mask=None, attention_mask=None
):
    runtime, fused = self.oq_runtime, self.oq_fused
    batch, rows, dim = latents.shape
    out = torch.empty_like(latents, dtype=torch.bfloat16)
    for sample in range(batch):
        h = out[sample]
        h.copy_(latents[sample])
        text = encoder_hidden_states[sample]
        text_valid = _valid(
            None if encoder_attention_mask is None else encoder_attention_mask[sample],
            text.shape[0],
        )
        cross = _cross_attention(
            self, self.cross_attention, fused, h, self.cross_attn_norm, text, text_valid
        )
        h.add_(runtime.linear(fused.xout, runtime.input(cross, cross.shape[1])))
        norm = self.self_attn_norm
        source = runtime.input(h, dim, norm=act.NORM_RMS, norm_weight=norm.weight, rms_eps=norm.eps)
        valid = _valid(None if attention_mask is None else attention_mask[sample], rows)
        att = _self_attention(self, self.self_attention, fused, source, None, None, valid)
        h.add_(runtime.linear(fused.attn_out, runtime.input(att, att.shape[1])))
        norm = self.ffn_norm
        source = runtime.input(h, dim, norm=act.NORM_RMS, norm_weight=norm.weight, rms_eps=norm.eps)
        h.add_(runtime.linear(fused.ff_out, _swiglu(runtime, fused, source)))
    return out
