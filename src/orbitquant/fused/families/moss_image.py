"""MOSS-Image (OpenMOSS ``MOSSImageTransformer``) joint image and text blocks.

Every block is a sequential attention + SiTU feed-forward block over the concatenated image and
text tokens. Both RMSNorms and the shared AdaLN modulation (one row per sample, the same for all
42 blocks) run in the activation-quantization prologue; Q|K|V is one GEMM, Q/K RMSNorm and the
rotate-half rotary embedding are one kernel, the gate|up rows of ``ffn_up_gate`` are one GEMM
with the SiTU epilogue, and both output projections add their gated residual in the GEMM
epilogue. The transformer evaluates its samples one at a time, so the blocks see one sample.
"""

from __future__ import annotations

import torch

from orbitquant.fused.families import int8_attention_allowed, specs
from orbitquant.fused.groups import RowSource
from orbitquant.kernels import triton_activation as act
from orbitquant.kernels import triton_int8_gemm as gemm

FAMILY = "moss_image"
MODEL_CLASSES = ("MOSSImageTransformer",)
EPS = 1e-6


def blocks(model):
    return [(f"layers.{i}", "block") for i in range(len(model.layers))]


def groups(kind, block):
    # ffn_up_gate rows are [hidden | gate]; the gated-unit epilogue reads (gate, up) pairs.
    ffn = block.proj["ffn_down"].in_features
    return specs(
        qkv=["proj.q", "proj.k", "proj.v"],
        mlp=(
            [RowSource("proj.ffn_up_gate", ffn, 2 * ffn), RowSource("proj.ffn_up_gate", 0, ffn)],
            {"swiglu": True},
        ),
        attn_out=["proj.o"],
        mlp_out=["proj.ffn_down"],
    )


def prepare(block, kind):
    block.oq_int8_attention = int8_attention_allowed(block.q_norm.weight, block.k_norm.weight)


def forward(kind):
    return _block


def _rope(runtime, rope):
    """The ``[1, L, 128]`` bf16 ``sin | cos`` table of 64 frequencies as per-channel ``[L, 128]``
    cos and sin."""
    cached = runtime.__dict__.get("moss_image_rope")
    if cached is not None and cached[0] is rope:
        return cached[1], cached[2]
    sin, cos = rope[0].float().chunk(2, dim=-1)
    cos = torch.cat((cos, cos), dim=-1).contiguous()
    sin = torch.cat((sin, sin), dim=-1).contiguous()
    runtime.__dict__["moss_image_rope"] = (rope, cos, sin)
    return cos, sin


def _block(self, x, rope, modulation):
    if x.shape[0] != 1:
        raise NotImplementedError("fused MOSS-Image blocks run one sample at a time")
    runtime, fused = self.oq_runtime, self.oq_fused
    shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = (
        m.reshape(-1).contiguous() for m in modulation.to(torch.bfloat16).unbind(1)
    )
    h = x[0].to(torch.bfloat16).clone()
    seq, dim = h.shape
    source = runtime.input(
        h,
        dim,
        norm=act.NORM_RMS,
        norm_weight=self.norm1.weight,
        mod=act.MOD_SHIFT_SCALE,
        mod_scale=scale_a,
        mod_shift=shift_a,
        rms_eps=EPS,
    )
    inner = fused.qkv.out_features // 3
    heads = self.heads
    head_dim = inner // heads
    qkv = runtime.linear(fused.qkv, source)
    cos, sin = _rope(runtime, rope)
    query = torch.empty((1, seq, heads, head_dim), device=h.device, dtype=torch.bfloat16)
    key = torch.empty_like(query)
    for src, weight, out in (
        (qkv[:, :inner], self.q_norm.weight, query),
        (qkv[:, inner : 2 * inner], self.k_norm.weight, key),
    ):
        act.qk_norm_rope_with_triton(
            src,
            heads,
            head_dim,
            weight,
            cos,
            sin,
            EPS,
            weight_mode=act.QK_WEIGHT_DIFFUSERS,
            style=act.ROPE_HALF,
            rope_bf16=True,
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
        gate=gate_a,
        out=h,
    )
    source = runtime.input(
        h,
        dim,
        norm=act.NORM_RMS,
        norm_weight=self.norm2.weight,
        mod=act.MOD_SHIFT_SCALE,
        mod_scale=scale_f,
        mod_shift=shift_f,
        rms_eps=EPS,
    )
    mid = runtime.linear(fused.mlp, source, epilogue=gemm.EPILOGUE_SITU)
    runtime.linear(
        fused.mlp_out,
        runtime.input(mid, mid.shape[1]),
        epilogue=gemm.EPILOGUE_RESIDUAL_GATE,
        residual=h,
        gate=gate_f,
        out=h,
    )
    return h[None]
