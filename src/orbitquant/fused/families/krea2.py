"""Krea 2 Turbo (``Krea2Transformer2DModel``) main blocks.

Per block: one Q|K|V|gate GEMM with the sigmoid gate in its epilogue, Q/K RMSNorm + interleaved
RoPE, attention, the gated output projection and the SwiGLU feed-forward with their residual
updates in GEMM epilogues. The protected BF16 ``ff.down`` runs as per-row INT8 (W8A8).
"""

from __future__ import annotations

import weakref

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


def install(model):
    _run_on_valid_text_rows(model)
    _cache_text_fusion(model)


def _run_on_valid_text_rows(model):
    """The pipeline pads the prompt to its maximum length and masks the padding. Padded rows
    get no attention weight and never reach the output, so a batch-1 forward runs on the valid
    text rows only and needs no mask: no per-block gather and scatter of the padded sequence,
    and RoPE and the text projection cover fewer rows. The trimmed tensors of the last two
    prompts are reused (a CFG pipeline alternates two), which keeps the text-fusion cache warm."""
    run = model.forward
    entries = []

    def same(held, tensors):
        return all(
            ref() is t and version == t._version
            for (ref, version), t in zip(held, tensors, strict=True)
        )

    def trimmed(encoder_hidden_states, mask, position_ids):
        tensors = (encoder_hidden_states, mask, position_ids)
        for held, out in entries:
            if same(held, tensors):
                return out
        text_len = encoder_hidden_states.shape[1]
        rows = mask.reshape(-1).nonzero().squeeze(1)
        out = (
            encoder_hidden_states[:, rows],
            torch.cat([position_ids[rows], position_ids[text_len:]]),
        )
        entries.insert(0, (tuple((weakref.ref(t), t._version) for t in tensors), out))
        del entries[2:]
        return out

    def forward(
        hidden_states,
        encoder_hidden_states,
        timestep,
        position_ids,
        encoder_attention_mask=None,
        attention_kwargs=None,
        return_dict=True,
    ):
        if (
            encoder_attention_mask is not None
            and hidden_states.shape[0] == 1
            and not torch.is_grad_enabled()
        ):
            encoder_hidden_states, position_ids = trimmed(
                encoder_hidden_states, encoder_attention_mask, position_ids
            )
            encoder_attention_mask = None
        return run(
            hidden_states,
            encoder_hidden_states,
            timestep,
            position_ids,
            encoder_attention_mask=encoder_attention_mask,
            attention_kwargs=attention_kwargs,
            return_dict=return_dict,
        )

    model.forward = forward


def _cache_text_fusion(model):
    """The text fusion stack does not depend on the timestep: run it once per prompt instead of
    in every denoising step. The last two results are kept (a CFG pipeline alternates two
    prompts), each for the same embeddings and mask tensor objects at the same versions, so a new
    prompt always recomputes."""
    text_fusion = model.text_fusion
    compute = text_fusion.forward
    entries = []

    def stamp(tensor):
        return None if tensor is None else (weakref.ref(tensor), tensor._version)

    def matches(held, tensor):
        if held is None:
            return tensor is None
        ref, version = held
        return tensor is not None and ref() is tensor and tensor._version == version

    def forward(encoder_hidden_states, attention_mask=None):
        if torch.is_grad_enabled():
            return compute(encoder_hidden_states, attention_mask=attention_mask)
        for hidden, mask, out in entries:
            if matches(hidden, encoder_hidden_states) and matches(mask, attention_mask):
                return out
        out = compute(encoder_hidden_states, attention_mask=attention_mask)
        entries.insert(0, (stamp(encoder_hidden_states), stamp(attention_mask), out))
        del entries[2:]
        return out

    text_fusion.forward = forward


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
    _, seq, dim = hidden_states.shape
    h = hidden_states.reshape(seq, dim)
    rows = _valid_rows(attention_mask)
    if rows is not None:
        # Padded text rows are never attended to and the model drops them at the output, so
        # only the valid rows run; the padded ones keep their input values.
        cos, sin = image_rotary_emb
        h[rows] = _block(self, h[rows], temb, (cos[rows], sin[rows]))
        return h.reshape(1, seq, dim)
    return _block(self, h, temb, image_rotary_emb).reshape(1, seq, dim)


# The last mask seen: (weak reference, version, valid rows).
_LAST_MASK: list = [None]


def _valid_rows(attention_mask):
    """Indices of the valid rows, or None when every row is valid. Every block of a forward
    gets the same mask tensor, so the host synchronization of finding them happens once."""
    if attention_mask is None:
        return None
    held = _LAST_MASK[0]
    if held is not None and held[0]() is attention_mask and held[1] == attention_mask._version:
        return held[2]
    mask = attention_mask.reshape(-1)
    rows = None if bool(mask.all()) else mask.nonzero().squeeze(1)
    _LAST_MASK[0] = (weakref.ref(attention_mask), attention_mask._version, rows)
    return rows


def _block(self, h, temb, image_rotary_emb):
    """One block over all-valid rows ``h`` ([rows, dim], updated in place)."""
    runtime, fused, attn = self.oq_runtime, self.oq_fused, self.attn
    seq, dim = h.shape
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
    return h
