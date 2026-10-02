"""Qwen-Image 2.1 (``QwenImage21Transformer2DModel``) single-stream blocks.

Per block: LayerNorm x (1 + scale) prologue, one Q|K|V GEMM, Q/K RMSNorm + interleaved complex
RoPE, attention over the block-causal joint sequence, the tanh-gated output projection and the
SwiGLU feed-forward with their residual updates in GEMM epilogues. The shared modulation holds
one row per sample plus, with ``causal_condition``, a ``t = 0`` row for the text and condition
tokens; the prologues and gates address it per token.

Attention follows ``QwenImage21AttnProcessor``: in the prefill every text segment is causal over
its own keys after full access to everything before it (SDPA with an explicit mask; text is
short), every image segment attends to all keys up to its end, and the target attends to all
keys; the decode reuses the cached post-RoPE prefix keys and values. Padded text keys
(``key_valid``) are dropped rather than masked. The checkpoint keeps some ``img_mlp.out``
projections in BF16 for quality; they stay BF16 here too.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from orbitquant.fused.families import int8_attention_allowed, specs
from orbitquant.kernels import triton_activation as act
from orbitquant.kernels import triton_int8_gemm as gemm

FAMILY = "qwenimage21"
MODEL_CLASSES = ("QwenImage21Transformer2DModel",)
DENSE = "bf16"


def blocks(model):
    return [(f"transformer_blocks.{i}", "main") for i in range(len(model.transformer_blocks))]


def groups(kind, block):
    return specs(
        qkv=["attn.to_q", "attn.to_k", "attn.to_v"],
        out=["attn.to_out.0"],
        ff_in=(["img_mlp.gate_layer", "img_mlp.proj"], {"swiglu": True}),
        ff_out=["img_mlp.out"],
    )


def prepare(block, kind):
    attn = block.attn
    block.oq_int8_attention = int8_attention_allowed(attn.norm_q.weight, attn.norm_k.weight)


def forward(kind):
    return _forward


def _rows(params, target_token_mask, sample):
    """Modulation table of one sample and the per-token row index into it (or None)."""
    if target_token_mask is None:
        return params[sample : sample + 1].contiguous(), None
    table = torch.cat([params[sample : sample + 1], params[-1:]]).contiguous()
    index = torch.where(target_token_mask, 0, 1).to(torch.int32)
    return table, index


def _rope_tables(freqs_cis):
    """Complex ``[seq, head_dim / 2]`` frequencies -> interleaved fp32 cos/sin rows."""
    return (
        freqs_cis.real.float().repeat_interleave(2, -1).contiguous(),
        freqs_cis.imag.float().repeat_interleave(2, -1).contiguous(),
    )


def _attend(runtime, block, query, key, value, valid):
    if valid is not None and not bool(valid.all()):
        key, value = key[:, valid], value[:, valid]
    return runtime.attention(query, key, value, int8_allowed=block.oq_int8_attention)


def _text_segment(query, key, value, start, valid):
    """Causal over the segment's own keys after full access to the keys before it."""
    rows = query.shape[1]
    mask = torch.ones(rows, start + rows, dtype=torch.bool, device=query.device)
    mask[:, start:] = torch.tril(mask[:, start:])
    if valid is not None:
        mask = mask & valid[None, : start + rows]
    out = F.scaled_dot_product_attention(
        query.transpose(1, 2),
        key[:, : start + rows].transpose(1, 2),
        value[:, : start + rows].transpose(1, 2),
        attn_mask=mask[None, None],
    )
    return out.transpose(1, 2)


def _forward(
    self,
    hidden_states,
    modulation,
    rotary_emb=None,
    attention_mask=None,
    target_token_mask=None,
    layer_cache=None,
    kv_cache_mode=None,
    cache_write_slice=None,
    segments=None,
    key_valid=None,
):
    batch = hidden_states.shape[0]
    if batch != 1 and layer_cache is not None:
        raise NotImplementedError("fused Qwen-Image 2.1 blocks cache one sample at a time")
    if key_valid is None and isinstance(attention_mask, torch.Tensor):
        # The decode passes key validity as a [batch, 1, 1, keys] mask instead.
        key_valid = attention_mask.reshape(batch, -1).bool()
    outs = []
    for sample in range(batch):
        valid = None if key_valid is None else key_valid[sample]
        outs.append(
            _sample(
                self,
                hidden_states[sample],
                modulation,
                sample,
                rotary_emb,
                target_token_mask,
                layer_cache,
                kv_cache_mode,
                cache_write_slice,
                segments,
                valid,
            )
        )
    return torch.stack(outs)


def _sample(
    self,
    hidden,
    modulation,
    sample,
    rotary_emb,
    target_token_mask,
    layer_cache,
    kv_cache_mode,
    cache_write_slice,
    segments,
    valid,
):
    runtime, fused, attn = self.oq_runtime, self.oq_fused, self.attn
    seq, dim = hidden.shape
    mod1, mod2 = modulation.chunk(2, dim=-1)
    scale1, gate1 = mod1.chunk(2, dim=-1)
    scale2, gate2 = mod2.chunk(2, dim=-1)
    scale1, index = _rows(scale1, target_token_mask, sample)
    scale2, _ = _rows(scale2, target_token_mask, sample)
    gate1, _ = _rows(torch.tanh(gate1), target_token_mask, sample)
    gate2, _ = _rows(torch.tanh(gate2), target_token_mask, sample)
    h = hidden.clone()

    def modulated(x, norm, scale):
        return runtime.input(
            x,
            dim,
            norm=act.NORM_LAYER,
            mod=act.MOD_ONE_PLUS_SCALE,
            mod_scale=scale,
            mod_index=index,
            rms_eps=norm.eps,
        )

    qkv = runtime.linear(fused.qkv, modulated(h, self.img_norm1, scale1))
    inner, heads = attn.inner_dim, attn.heads
    head_dim = inner // heads
    query = torch.empty((1, seq, heads, head_dim), device=h.device, dtype=torch.bfloat16)
    key = torch.empty_like(query)
    cos, sin = _rope_tables(rotary_emb) if rotary_emb is not None else (None, None)
    for offset, norm, out in ((0, attn.norm_q, query), (inner, attn.norm_k, key)):
        act.qk_norm_rope_with_triton(
            qkv[:, offset : offset + inner],
            heads,
            head_dim,
            norm.weight,
            cos,
            sin,
            norm.eps,
            weight_mode=act.QK_WEIGHT_DIFFUSERS,
            style=act.ROPE_INTERLEAVED,
            out=out,
        )
    value = qkv[:, 2 * inner :].unflatten(-1, (heads, head_dim))[None]
    if layer_cache is not None and kv_cache_mode == "extract" and cache_write_slice is not None:
        layer_cache.store(key[:, cache_write_slice].clone(), value[:, cache_write_slice].clone())
    elif layer_cache is not None and kv_cache_mode == "cached":
        cached_key, cached_value = layer_cache.get()
        key = torch.cat([cached_key, key], dim=1)
        value = torch.cat([cached_value, value], dim=1)

    if segments is None:
        att = _attend(runtime, self, query, key, value, valid)
    else:
        prefix = segments[-1][1] if segments else 0
        att = torch.empty_like(query)
        for start, end, is_text in segments:
            if is_text:
                att[:, start:end] = _text_segment(query[:, start:end], key, value, start, valid)
            else:
                part_valid = None if valid is None else valid[:end]
                att[:, start:end] = _attend(
                    runtime, self, query[:, start:end], key[:, :end], value[:, :end], part_valid
                )
        att[:, prefix:] = _attend(runtime, self, query[:, prefix:], key, value, valid)

    runtime.linear(
        fused.out,
        runtime.input(att.reshape(seq, inner), inner),
        epilogue=gemm.EPILOGUE_RESIDUAL_GATE,
        residual=h,
        gate=gate1,
        gate_index=index,
        out=h,
    )
    mid = runtime.linear(
        fused.ff_in, modulated(h, self.img_norm2, scale2), epilogue=gemm.EPILOGUE_SWIGLU
    )
    runtime.linear(
        fused.ff_out,
        runtime.input(mid, mid.shape[1]),
        epilogue=gemm.EPILOGUE_RESIDUAL_GATE,
        residual=h,
        gate=gate2,
        gate_index=index,
        out=h,
    )
    return h
