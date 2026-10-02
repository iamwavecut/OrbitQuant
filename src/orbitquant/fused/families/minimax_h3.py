"""MiniMax-H3 (``MiniMaxH3Transformer3DModel``) main blocks.

One packed video/audio/text sequence; the AdaLN modulation of every row is a row of a
(timestep, modality) table. The prologues and residual epilogues read that table through the
per-row index directly instead of materializing ``[seq, hidden]`` modulation tensors. Q/K get
RMSNorm and a rotate-half RoPE on the leading 96 channels of each 128-wide head. The token
refiner blocks (text only, once per forward) keep their modules.

Padding rows form their own attention document (the model's mask pairs live rows with live rows
and padding rows with padding rows); the two documents run as separate mask-free attentions.
"""

from __future__ import annotations

import torch

from orbitquant.fused.families import int8_attention_allowed, specs
from orbitquant.fused.groups import RowSource
from orbitquant.kernels import triton_activation as act
from orbitquant.kernels import triton_int8_gemm as gemm

FAMILY = "minimax_h3"
MODEL_CLASSES = ("MiniMaxH3Transformer3DModel",)


def blocks(model):
    return [(f"transformer_blocks.{i}", "main") for i in range(len(model.transformer_blocks))]


def groups(kind, block):
    ffn = block.ff.net[2].in_features
    return specs(
        qkv=["attn.to_q", "attn.to_k", "attn.to_v"],
        out=["attn.to_out.0"],
        # diffusers' SwiGLU computes first_half * silu(second_half): the gate is the second half.
        ff_in=(
            [RowSource("ff.net.0.proj", ffn, 2 * ffn), RowSource("ff.net.0.proj", 0, ffn)],
            {"swiglu": True},
        ),
        ff_out=["ff.net.2"],
    )


def prepare(block, kind):
    attn = block.attn
    block.oq_int8_attention = int8_attention_allowed(attn.norm_q.weight, attn.norm_k.weight)


def forward(kind):
    return _forward


def _forward(self, hidden_states, temb, adaln_indices, rotary_emb, attention_mask=None):
    if hidden_states.shape[0] != 1:
        return torch.cat(
            [
                _forward(
                    self, hidden_states[i : i + 1], temb, adaln_indices, rotary_emb, attention_mask
                )
                for i in range(hidden_states.shape[0])
            ]
        )
    runtime, fused, attn = self.oq_runtime, self.oq_fused, self.attn
    _, seq, dim = hidden_states.shape
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
        t.contiguous() for t in self.adaln_proj(temb)
    )
    index = adaln_indices.to(torch.int32).contiguous()
    h = hidden_states[0].clone()
    source = runtime.input(
        h,
        dim,
        norm=act.NORM_RMS_TORCH,
        mod=act.MOD_SHIFT_SCALE,
        norm_weight=self.norm1.weight,
        mod_scale=scale_msa,
        mod_shift=shift_msa,
        mod_index=index,
        rms_eps=float(self.norm1.eps),
    )
    qkv = runtime.linear(fused.qkv, source)
    heads, head_dim = attn.heads, attn.head_dim
    inner = heads * head_dim
    cos, sin = rotary_emb
    query, key = (
        act.qk_norm_rope_with_triton(
            qkv[:, offset : offset + inner],
            heads,
            head_dim,
            norm.weight,
            cos,
            sin,
            norm.eps,
            weight_mode=act.QK_WEIGHT_TORCH,
            style=act.ROPE_HALF,
            rope_bf16=True,
        )
        for offset, norm in ((0, attn.norm_q), (inner, attn.norm_k))
    )
    value = qkv[:, 2 * inner :].unflatten(-1, (heads, head_dim))[None]
    if attention_mask is None:
        att = runtime.attention(query, key, value, int8_allowed=self.oq_int8_attention)
    else:
        att = torch.empty_like(query)
        first_document = attention_mask.reshape(seq, seq)[0]
        for rows in (first_document, ~first_document):
            rows = rows.nonzero().flatten()
            if rows.numel():
                att[:, rows] = runtime.attention(
                    query[:, rows],
                    key[:, rows],
                    value[:, rows],
                    int8_allowed=self.oq_int8_attention,
                )
    runtime.linear(
        fused.out,
        runtime.input(att.reshape(seq, inner), inner),
        epilogue=gemm.EPILOGUE_RESIDUAL_GATE,
        residual=h,
        gate=gate_msa,
        gate_index=index,
        out=h,
    )
    source = runtime.input(
        h,
        dim,
        norm=act.NORM_RMS_TORCH,
        mod=act.MOD_SHIFT_SCALE,
        norm_weight=self.norm2.weight,
        mod_scale=scale_mlp,
        mod_shift=shift_mlp,
        mod_index=index,
        rms_eps=float(self.norm2.eps),
    )
    mid = runtime.linear(fused.ff_in, source, epilogue=gemm.EPILOGUE_SWIGLU)
    runtime.linear(
        fused.ff_out,
        runtime.input(mid, mid.shape[1]),
        epilogue=gemm.EPILOGUE_RESIDUAL_GATE,
        residual=h,
        gate=gate_mlp,
        gate_index=index,
        out=h,
    )
    return h[None]
