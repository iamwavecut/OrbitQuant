"""FLUX.2 (``Flux2Transformer2DModel``) double- and single-stream blocks.

Double-stream block, per stream: LayerNorm + AdaLN prologue, one Q|K|V GEMM, Q/K RMSNorm +
interleaved RoPE into a joint text|image sequence, the gated output projection and the SwiGLU
feed-forward with their residual updates in GEMM epilogues.

Single-stream block: the fused ``to_qkv_mlp_proj`` splits into a Q|K|V group and a SwiGLU
group whose activation lands next to the attention output, so ``to_out`` reads both from one
buffer.

The reference-image KV cache modes keep their attention pattern (diffusers'
``_flux2_kv_causal_attention``) on SDPA.
"""

from __future__ import annotations

import torch

from orbitquant.fused.families import int8_attention_allowed, specs
from orbitquant.fused.groups import RowSource
from orbitquant.kernels import triton_activation as act
from orbitquant.kernels import triton_int8_gemm as gemm

FAMILY = "flux2"
MODEL_CLASSES = ("Flux2Transformer2DModel",)


def blocks(model):
    return [(f"transformer_blocks.{i}", "double") for i in range(len(model.transformer_blocks))] + [
        (f"single_transformer_blocks.{i}", "single")
        for i in range(len(model.single_transformer_blocks))
    ]


def groups(kind, block):
    if kind == "double":
        hidden = block.ff.linear_out.in_features
        return specs(
            img_qkv=["attn.to_q", "attn.to_k", "attn.to_v"],
            txt_qkv=["attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj"],
            img_out=["attn.to_out.0"],
            txt_out=["attn.to_add_out"],
            img_ff_in=(
                [
                    RowSource("ff.linear_in", 0, hidden),
                    RowSource("ff.linear_in", hidden, 2 * hidden),
                ],
                {"swiglu": True},
            ),
            img_ff_out=["ff.linear_out"],
            txt_ff_in=(
                [
                    RowSource("ff_context.linear_in", 0, hidden),
                    RowSource("ff_context.linear_in", hidden, 2 * hidden),
                ],
                {"swiglu": True},
            ),
            txt_ff_out=["ff_context.linear_out"],
        )
    attn = block.attn
    inner, hidden = attn.inner_dim, attn.mlp_hidden_dim
    return specs(
        qkv=[RowSource("attn.to_qkv_mlp_proj", 0, 3 * inner)],
        mlp=(
            [
                RowSource("attn.to_qkv_mlp_proj", 3 * inner, 3 * inner + hidden),
                RowSource("attn.to_qkv_mlp_proj", 3 * inner + hidden, 3 * inner + 2 * hidden),
            ],
            {"swiglu": True},
        ),
        out=["attn.to_out"],
    )


def prepare(block, kind):
    attn = block.attn
    weights = [attn.norm_q.weight, attn.norm_k.weight]
    if kind == "double":
        weights += [attn.norm_added_q.weight, attn.norm_added_k.weight]
    block.oq_int8_attention = int8_attention_allowed(*weights)


def forward(kind):
    return _double if kind == "double" else _single


def _modulation(mod, sets):
    """``Flux2Modulation.split`` as flat ``[rows, dim]`` tensors plus a row index (or None)."""
    if mod.ndim == 2:
        mod = mod.unsqueeze(1)
    chunks = torch.chunk(mod, 3 * sets, dim=-1)
    rows = mod.shape[1]
    out = []
    for i in range(sets):
        shift, scale, gate = (c.reshape(rows, -1).contiguous() for c in chunks[3 * i : 3 * i + 3])
        out.append((shift, scale, gate))
    return out, rows


def _row_index(rows, seq, device):
    if rows == 1:
        return None
    if rows != seq:
        raise ValueError(f"modulation has {rows} rows for {seq} tokens")
    return torch.arange(seq, device=device, dtype=torch.int32)


def _ln_input(runtime, x, dim, shift, scale, index, eps):
    return runtime.input(
        x,
        dim,
        norm=act.NORM_LAYER,
        mod=act.MOD_SHIFT_SCALE,
        mod_scale=scale if index is not None else scale.reshape(-1),
        mod_shift=shift if index is not None else shift.reshape(-1),
        mod_index=index,
        rms_eps=eps,
    )


def _gate(gate, index):
    return (gate, index) if index is not None else (gate.reshape(-1), None)


def _qk(attn, qkv, rows, query, key, cos, sin, added):
    """Q/K RMSNorm + RoPE of ``qkv`` into rows ``rows`` of ``[1, seq, heads, head_dim]``."""
    inner, heads, head_dim = attn.inner_dim, attn.heads, attn.head_dim
    norm_q = attn.norm_added_q if added else attn.norm_q
    norm_k = attn.norm_added_k if added else attn.norm_k
    for src, norm, out in (
        (qkv[:, :inner], norm_q, query),
        (qkv[:, inner : 2 * inner], norm_k, key),
    ):
        act.qk_norm_rope_with_triton(
            src,
            heads,
            head_dim,
            norm.weight,
            cos,
            sin,
            norm.eps,
            weight_mode=act.QK_WEIGHT_TORCH,
            style=act.ROPE_INTERLEAVED,
            out=out[:, rows],
        )


def _kv_cache_attention(query, key, value, kwargs, num_txt):
    from diffusers.models.transformers.transformer_flux2 import _flux2_kv_causal_attention

    mode, cache = kwargs.get("kv_cache_mode"), kwargs.get("kv_cache")
    num_ref = int(kwargs.get("num_ref_tokens", 0) or 0)
    if mode == "extract" and cache is not None and num_ref > 0:
        cache.store(
            key[:, num_txt : num_txt + num_ref].clone(),
            value[:, num_txt : num_txt + num_ref].clone(),
        )
    if mode == "extract" and num_ref > 0:
        return _flux2_kv_causal_attention(query, key, value, num_txt, num_ref)
    return _flux2_kv_causal_attention(query, key, value, num_txt, 0, kv_cache=cache)


def _attention(self, query, key, value, kwargs, num_txt):
    runtime = self.oq_runtime
    if kwargs and kwargs.get("kv_cache_mode") is not None:
        return _kv_cache_attention(query, key, value, kwargs, num_txt)
    return runtime.attention(query, key, value, int8_allowed=self.oq_int8_attention)


def _double(
    self,
    hidden_states,
    encoder_hidden_states,
    temb_mod_img,
    temb_mod_txt,
    image_rotary_emb=None,
    joint_attention_kwargs=None,
):
    if hidden_states.shape[0] != 1:
        outs = [
            _double(
                self,
                hidden_states[i : i + 1],
                encoder_hidden_states[i : i + 1],
                temb_mod_img[i : i + 1],
                temb_mod_txt[i : i + 1],
                image_rotary_emb,
                joint_attention_kwargs,
            )
            for i in range(hidden_states.shape[0])
        ]
        return torch.cat([o[0] for o in outs]), torch.cat([o[1] for o in outs])
    runtime, fused, attn = self.oq_runtime, self.oq_fused, self.attn
    kwargs = joint_attention_kwargs or {}
    img, txt = hidden_states[0], encoder_hidden_states[0]
    s_img, s_txt, dim = img.shape[0], txt.shape[0], img.shape[1]
    seq = s_txt + s_img
    eps = float(self.norm1.eps)
    (i_msa, i_mlp), i_rows = _modulation(temb_mod_img, 2)
    (t_msa, t_mlp), t_rows = _modulation(temb_mod_txt, 2)
    i_index = _row_index(i_rows, s_img, img.device)
    t_index = _row_index(t_rows, s_txt, img.device)

    inner, heads, head_dim = attn.inner_dim, attn.heads, attn.head_dim
    qkv = torch.empty((seq, 3 * inner), device=img.device, dtype=torch.bfloat16)
    runtime.linear(
        fused.txt_qkv,
        _ln_input(runtime, txt, dim, t_msa[0], t_msa[1], t_index, eps),
        out=qkv[:s_txt],
    )
    runtime.linear(
        fused.img_qkv,
        _ln_input(runtime, img, dim, i_msa[0], i_msa[1], i_index, eps),
        out=qkv[s_txt:],
    )
    cos, sin = image_rotary_emb
    query = torch.empty((1, seq, heads, head_dim), device=img.device, dtype=torch.bfloat16)
    key = torch.empty_like(query)
    _qk(attn, qkv[:s_txt], slice(0, s_txt), query, key, cos[:s_txt], sin[:s_txt], True)
    _qk(attn, qkv[s_txt:], slice(s_txt, seq), query, key, cos[s_txt:], sin[s_txt:], False)
    value = qkv[:, 2 * inner :].unflatten(-1, (heads, head_dim))[None]
    att = _attention(self, query, key, value, kwargs, s_txt).reshape(seq, inner)

    gate, index = _gate(i_msa[2], i_index)
    img = img.clone()
    runtime.linear(
        fused.img_out,
        runtime.input(att[s_txt:], inner),
        epilogue=gemm.EPILOGUE_RESIDUAL_GATE,
        residual=img,
        gate=gate,
        gate_index=index,
        out=img,
    )
    shift, scale, gate = i_mlp
    mid = runtime.linear(
        fused.img_ff_in,
        _ln_input(runtime, img, dim, shift, scale, i_index, eps),
        epilogue=gemm.EPILOGUE_SWIGLU,
    )
    gate, index = _gate(gate, i_index)
    runtime.linear(
        fused.img_ff_out,
        runtime.input(mid, mid.shape[1]),
        epilogue=gemm.EPILOGUE_RESIDUAL_GATE,
        residual=img,
        gate=gate,
        gate_index=index,
        out=img,
    )

    gate, index = _gate(t_msa[2], t_index)
    txt = txt.clone()
    runtime.linear(
        fused.txt_out,
        runtime.input(att[:s_txt], inner),
        epilogue=gemm.EPILOGUE_RESIDUAL_GATE,
        residual=txt,
        gate=gate,
        gate_index=index,
        out=txt,
    )
    shift, scale, gate = t_mlp
    mid = runtime.linear(
        fused.txt_ff_in,
        _ln_input(runtime, txt, dim, shift, scale, t_index, eps),
        epilogue=gemm.EPILOGUE_SWIGLU,
    )
    gate, index = _gate(gate, t_index)
    runtime.linear(
        fused.txt_ff_out,
        runtime.input(mid, mid.shape[1]),
        epilogue=gemm.EPILOGUE_RESIDUAL_GATE,
        residual=txt,
        gate=gate,
        gate_index=index,
        out=txt,
    )
    return txt[None], img[None]


def _single(
    self,
    hidden_states,
    encoder_hidden_states,
    temb_mod,
    image_rotary_emb=None,
    joint_attention_kwargs=None,
    split_hidden_states=False,
    text_seq_len=None,
):
    if encoder_hidden_states is not None:
        text_seq_len = encoder_hidden_states.shape[1]
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
    if hidden_states.shape[0] != 1:
        out = torch.cat(
            [
                _single(
                    self,
                    hidden_states[i : i + 1],
                    None,
                    temb_mod[i : i + 1],
                    image_rotary_emb,
                    joint_attention_kwargs,
                )
                for i in range(hidden_states.shape[0])
            ]
        )
    else:
        out = _single_one(self, hidden_states, temb_mod, image_rotary_emb, joint_attention_kwargs)
    if split_hidden_states:
        return out[:, :text_seq_len], out[:, text_seq_len:]
    return out


def _single_one(self, hidden_states, temb_mod, image_rotary_emb, joint_attention_kwargs):
    runtime, fused, attn = self.oq_runtime, self.oq_fused, self.attn
    kwargs = joint_attention_kwargs or {}
    h = hidden_states[0].clone()
    seq, dim = h.shape
    eps = float(self.norm.eps)
    ((shift, scale, gate),), rows = _modulation(temb_mod, 1)
    index = _row_index(rows, seq, h.device)
    source = _ln_input(runtime, h, dim, shift, scale, index, eps)
    inner, heads, head_dim = attn.inner_dim, attn.heads, attn.head_dim
    qkv = runtime.linear(fused.qkv, source)
    joined = torch.empty((seq, inner + attn.mlp_hidden_dim), device=h.device, dtype=torch.bfloat16)
    runtime.linear(fused.mlp, source, epilogue=gemm.EPILOGUE_SWIGLU, out=joined, col_offset=inner)
    cos, sin = image_rotary_emb
    query = torch.empty((1, seq, heads, head_dim), device=h.device, dtype=torch.bfloat16)
    key = torch.empty_like(query)
    _qk(attn, qkv, slice(0, seq), query, key, cos, sin, False)
    value = qkv[:, 2 * inner :].unflatten(-1, (heads, head_dim))[None]
    num_txt = int(kwargs.get("num_txt_tokens", 0) or 0)
    joined[:, :inner] = _attention(self, query, key, value, kwargs, num_txt).reshape(seq, inner)
    gate, gate_index = _gate(gate, index)
    runtime.linear(
        fused.out,
        runtime.input(joined, joined.shape[1]),
        epilogue=gemm.EPILOGUE_RESIDUAL_GATE,
        residual=h,
        gate=gate,
        gate_index=gate_index,
        out=h,
    )
    return h[None]
