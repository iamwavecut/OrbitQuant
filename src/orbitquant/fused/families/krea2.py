"""Krea 2 Turbo (``Krea2Transformer2DModel``) main blocks.

Per block: one Q|K|V|gate GEMM with the sigmoid gate in its epilogue, Q/K RMSNorm + interleaved
RoPE, attention, the gated output projection and the SwiGLU feed-forward with their residual
updates in GEMM epilogues. The protected BF16 ``ff.down`` runs as per-row INT8 (W8A8).
"""

from __future__ import annotations

import torch

from orbitquant.fused.families import int8_attention_allowed, specs
from orbitquant.kernels import triton_activation as act
from orbitquant.kernels import triton_int8_gemm as gemm

FAMILY = "krea2"
MODEL_CLASSES = ("Krea2Transformer2DModel",)


def blocks(model):
    return [(f"transformer_blocks.{i}", "main") for i in range(len(model.transformer_blocks))]


def groups(kind, block):
    return specs(
        qkvg=["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_gate"],
        out=["attn.to_out.0"],
        gateup=(["ff.gate", "ff.up"], {"swiglu": True}),
        down=["ff.down"],
    )


def prepare(block, kind):
    # Krea 2 stores its Q/K RMSNorm weights zero-centered.
    attn = block.attn
    block.oq_int8_attention = int8_attention_allowed(
        attn.norm_q.weight, attn.norm_k.weight, offset=1.0
    )


def forward(kind):
    return _forward


def _forward(self, hidden_states, temb, image_rotary_emb, attention_mask=None):
    if hidden_states.shape[0] != 1:
        return torch.cat(
            [
                _forward(
                    self,
                    hidden_states[i : i + 1],
                    temb[i : i + 1],
                    image_rotary_emb,
                    (None if attention_mask is None else attention_mask[i : i + 1]),
                )
                for i in range(hidden_states.shape[0])
            ]
        )
    runtime, fused, attn = self.oq_runtime, self.oq_fused, self.attn
    _, seq, dim = hidden_states.shape
    h = hidden_states.reshape(seq, dim)
    modulation = temb.reshape(-1).unflatten(-1, (6, -1)) + self.scale_shift_table
    prescale, preshift, pregate, postscale, postshift, postgate = (
        t.reshape(dim).contiguous() for t in modulation.unbind(-2)
    )
    eps = float(self.norm1.eps)

    source = runtime.input(
        h,
        dim,
        norm=act.NORM_RMS_ONE_PLUS,
        mod=act.MOD_SHIFT_SCALE,
        norm_weight=self.norm1.weight,
        mod_scale=prescale,
        mod_shift=preshift,
        rms_eps=eps,
    )
    qkvg = runtime.linear(
        fused.qkvg,
        source,
        epilogue=gemm.EPILOGUE_SIGMOID_TAIL,
        sig_from=fused.qkvg.out_features - dim,
    )
    heads, kv_heads, head_dim = attn.num_heads, attn.num_kv_heads, attn.head_dim
    qw, kw = heads * head_dim, kv_heads * head_dim
    cos, sin = image_rotary_emb
    query = act.qk_norm_rope_with_triton(
        qkvg[:, :qw], heads, head_dim, attn.norm_q.weight, cos, sin, attn.norm_q.eps
    )
    key = act.qk_norm_rope_with_triton(
        qkvg[:, qw : qw + kw], kv_heads, head_dim, attn.norm_k.weight, cos, sin, attn.norm_k.eps
    )
    value = qkvg[:, qw + kw : qw + 2 * kw].unflatten(-1, (kv_heads, head_dim))[None]
    if attention_mask is not None and not bool(attention_mask.all()):
        valid = attention_mask.reshape(-1).bool()
        key, value = key[:, valid], value[:, valid]
    att = runtime.attention(query, key, value, int8_allowed=self.oq_int8_attention).reshape(
        seq, dim
    )

    source = runtime.input(att, dim, prologue=act.PROLOGUE_MULTIPLY, x2=qkvg[:, qw + 2 * kw :])
    h = runtime.linear(
        fused.out, source, epilogue=gemm.EPILOGUE_RESIDUAL_GATE, residual=h, gate=pregate, out=h
    )
    source = runtime.input(
        h,
        dim,
        norm=act.NORM_RMS_ONE_PLUS,
        mod=act.MOD_SHIFT_SCALE,
        norm_weight=self.norm2.weight,
        mod_scale=postscale,
        mod_shift=postshift,
        rms_eps=eps,
    )
    mid = runtime.linear(fused.gateup, source, epilogue=gemm.EPILOGUE_SWIGLU)
    h = runtime.linear(
        fused.down,
        runtime.input(mid, mid.shape[1]),
        epilogue=gemm.EPILOGUE_RESIDUAL_GATE,
        residual=h,
        gate=postgate,
        out=h,
    )
    return h.reshape(1, seq, dim)
