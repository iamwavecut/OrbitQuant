"""Ideogram 4 (``Ideogram4Transformer2DModel``) blocks.

Sandwich-norm block: RMSNorm x (1 + scale) prologue, one Q|K|V GEMM, Q/K RMSNorm + rotate-half
MRoPE on 256-wide heads, attention, then ``h + tanh(gate) * RMSNorm(out)`` as a row kernel (the
post-norm needs the whole row, so it cannot ride in a GEMM epilogue); the SwiGLU feed-forward
mirrors it. The AdaLN modulation projection keeps its own module.
"""

from __future__ import annotations

import torch

from orbitquant.fused.families import int8_attention_allowed, specs
from orbitquant.kernels import triton_activation as act
from orbitquant.kernels import triton_int8_gemm as gemm

FAMILY = "ideogram4"
MODEL_CLASSES = ("Ideogram4Transformer2DModel",)


def blocks(model):
    return [(f"layers.{i}", "main") for i in range(len(model.layers))]


def groups(kind, block):
    return specs(
        qkv=["attention.to_q", "attention.to_k", "attention.to_v"],
        out=["attention.to_out.0"],
        ff_in=(["feed_forward.w1", "feed_forward.w3"], {"swiglu": True}),
        ff_out=["feed_forward.w2"],
    )


def prepare(block, kind):
    attn = block.attention
    block.oq_int8_attention = int8_attention_allowed(attn.norm_q.weight, attn.norm_k.weight)


def forward(kind):
    return _forward


def _per_row(t, seq):
    """``[1, rows, dim]`` modulation -> (table, index) addressing it per token."""
    t = t.reshape(-1, t.shape[-1])
    if t.shape[0] == 1:
        return t.reshape(-1), None
    if t.shape[0] != seq:
        raise ValueError(f"modulation has {t.shape[0]} rows for {seq} tokens")
    return t.contiguous(), torch.arange(seq, device=t.device, dtype=torch.int32)


def _forward(self, hidden_states, attention_mask, image_rotary_emb, adaln_input):
    if hidden_states.shape[0] != 1:
        return torch.cat(
            [
                _forward(
                    self,
                    hidden_states[i : i + 1],
                    None if attention_mask is None else attention_mask[i : i + 1],
                    tuple(t[i : i + 1] for t in image_rotary_emb),
                    adaln_input[i : i + 1],
                )
                for i in range(hidden_states.shape[0])
            ]
        )
    runtime, fused, attn = self.oq_runtime, self.oq_fused, self.attention
    _, seq, dim = hidden_states.shape
    mod = self.adaln_modulation(adaln_input)
    scale_msa, gate_msa, scale_mlp, gate_mlp = mod.chunk(4, dim=-1)
    gate_msa, gate_mlp = torch.tanh(gate_msa), torch.tanh(gate_mlp)
    scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp
    scale_msa, index = _per_row(scale_msa, seq)
    gate_msa, _ = _per_row(gate_msa, seq)
    scale_mlp, _ = _per_row(scale_mlp, seq)
    gate_mlp, _ = _per_row(gate_mlp, seq)
    eps = float(self.attention_norm1.eps)
    h = hidden_states[0].clone()

    source = runtime.input(
        h,
        dim,
        norm=act.NORM_RMS,
        mod=act.MOD_SCALE,
        norm_weight=self.attention_norm1.weight,
        mod_scale=scale_msa,
        mod_index=index,
        rms_eps=eps,
    )
    qkv = runtime.linear(fused.qkv, source)
    heads, head_dim = attn.num_heads, attn.head_dim
    cos, sin = (t.reshape(seq, -1) for t in image_rotary_emb)
    query, key = (
        act.qk_norm_rope_with_triton(
            qkv[:, offset : offset + dim],
            heads,
            head_dim,
            norm.weight,
            cos,
            sin,
            norm.eps,
            weight_mode=act.QK_WEIGHT_DIFFUSERS,
            style=act.ROPE_HALF,
            rope_bf16=True,
        )
        for offset, norm in ((0, attn.norm_q), (dim, attn.norm_k))
    )
    value = qkv[:, 2 * dim :].unflatten(-1, (heads, head_dim))[None]
    single_segment = attention_mask is None or bool(attention_mask.all())
    if single_segment:
        att = runtime.attention(query, key, value, int8_allowed=self.oq_int8_attention)
    else:
        att = torch.nn.functional.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask=attention_mask,
        ).transpose(1, 2)
    out = runtime.linear(fused.out, runtime.input(att.reshape(seq, dim), dim))
    act.postnorm_residual_with_triton(
        h, out, self.attention_norm2.weight, gate_msa, eps, norm=act.NORM_RMS, gate_index=index
    )

    source = runtime.input(
        h,
        dim,
        norm=act.NORM_RMS,
        mod=act.MOD_SCALE,
        norm_weight=self.ffn_norm1.weight,
        mod_scale=scale_mlp,
        mod_index=index,
        rms_eps=eps,
    )
    mid = runtime.linear(fused.ff_in, source, epilogue=gemm.EPILOGUE_SWIGLU)
    out = runtime.linear(fused.ff_out, runtime.input(mid, mid.shape[1]))
    act.postnorm_residual_with_triton(
        h, out, self.ffn_norm2.weight, gate_mlp, eps, norm=act.NORM_RMS, gate_index=index
    )
    return h[None]
