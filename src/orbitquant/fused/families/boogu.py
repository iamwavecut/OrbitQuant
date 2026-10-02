"""Boogu-Image (``BooguImageTransformer2DModel``) blocks.

Refiner and single-stream blocks share the Lumina recipe: RMSNorm x (1 + scale) prologue, one
Q|K|V GEMM for grouped-query attention (28/7 heads of 120 channels, zero-padded to 128 for the
INT8 kernel), interleaved complex RoPE, sandwich post-norms with tanh gates as row kernels and a
SwiGLU feed-forward. Double-stream blocks run a joint instruction|image attention with
per-stream Q|K|V and output projections, then an image self-attention; their MLP inputs are
modulated norms of the block input, not of the attention-updated streams.

Attention masks only mark the padding of batched packed sequences. Each sample runs alone, so
padded keys are dropped instead and attention stays mask-free.

Every group takes 8-bit activations (per-token absmax INT8 of the rotated input): at 3360
channels the RPBH block is 32 wide, too narrow to spread outliers for 4-bit codes. On the Q|K|V
inputs the 4-bit error turns into knit-like texture across the image; with 8 bits everywhere the
images stay closer to BF16 than the SDNQ W4A8 checkpoint, at the speed of 4-bit codes (the GEMMs
are INT8 either way).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from orbitquant.fused.families import int8_attention_allowed, specs
from orbitquant.kernels import triton_activation as act
from orbitquant.kernels import triton_int8_gemm as gemm

FAMILY = "boogu"
MODEL_CLASSES = ("BooguImageTransformer2DModel",)

_SINGLE_LISTS = ("noise_refiner", "ref_image_refiner", "context_refiner", "single_stream_layers")
_JOINT = "img_instruct_attn.processor"
ACTIVATION_BITS = dict.fromkeys(
    (
        "qkv",
        "out",
        "ff_in",
        "ff_out",
        "joint_img_qkv",
        "joint_txt_qkv",
        "joint_img_out",
        "joint_txt_out",
        "joint_out",
        "self_qkv",
        "self_out",
        "img_ff_in",
        "img_ff_out",
        "txt_ff_in",
        "txt_ff_out",
    ),
    8,
)


def blocks(model):
    paths = [
        (f"{name}.{i}", "single")
        for name in _SINGLE_LISTS
        for i in range(len(getattr(model, name, ())))
    ]
    return paths + [
        (f"double_stream_layers.{i}", "double") for i in range(len(model.double_stream_layers))
    ]


def groups(kind, block):
    if kind == "single":
        return specs(
            qkv=["attn.to_q", "attn.to_k", "attn.to_v"],
            out=["attn.to_out.0"],
            ff_in=(["feed_forward.linear_1", "feed_forward.linear_3"], {"swiglu": True}),
            ff_out=["feed_forward.linear_2"],
        )
    return specs(
        joint_img_qkv=[f"{_JOINT}.img_to_q", f"{_JOINT}.img_to_k", f"{_JOINT}.img_to_v"],
        joint_txt_qkv=[
            f"{_JOINT}.instruct_to_q",
            f"{_JOINT}.instruct_to_k",
            f"{_JOINT}.instruct_to_v",
        ],
        joint_img_out=[f"{_JOINT}.img_out"],
        joint_txt_out=[f"{_JOINT}.instruct_out"],
        joint_out=["img_instruct_attn.to_out.0"],
        self_qkv=["img_self_attn.to_q", "img_self_attn.to_k", "img_self_attn.to_v"],
        self_out=["img_self_attn.to_out.0"],
        img_ff_in=(["img_feed_forward.linear_1", "img_feed_forward.linear_3"], {"swiglu": True}),
        img_ff_out=["img_feed_forward.linear_2"],
        txt_ff_in=(
            ["instruct_feed_forward.linear_1", "instruct_feed_forward.linear_3"],
            {"swiglu": True},
        ),
        txt_ff_out=["instruct_feed_forward.linear_2"],
    )


def prepare(block, kind):
    attns = [block.attn] if kind == "single" else [block.img_instruct_attn, block.img_self_attn]
    weights = [w for attn in attns for w in (attn.norm_q.weight, attn.norm_k.weight)]
    block.oq_int8_attention = int8_attention_allowed(*weights)


def forward(kind):
    return _single if kind == "single" else _double


def _modulation(norm_zero, temb):
    """The four ``[dim]`` rows a ``LuminaRMSNormZero`` projects from the timestep embedding."""
    emb = norm_zero.linear(F.silu(temb))
    return [t.reshape(-1).contiguous() for t in emb.chunk(4, dim=1)]


def _rope_tables(freqs_cis, rows):
    """Complex ``[1, seq, head_dim / 2]`` frequencies -> interleaved fp32 cos/sin rows."""
    freqs = freqs_cis.reshape(-1, freqs_cis.shape[-1])[:rows]
    return (
        freqs.real.float().repeat_interleave(2, -1).contiguous(),
        freqs.imag.float().repeat_interleave(2, -1).contiguous(),
    )


def _valid_keys(mask, rows):
    if mask is None:
        return None
    if mask.dim() != 2:
        raise ValueError("fused Boogu blocks take [batch, seq] padding masks")
    valid = mask.reshape(-1)[:rows].bool()
    return None if bool(valid.all()) else valid


class _Heads:
    """Query/key/value buffers of one GQA attention, heads zero-padded to a power of two."""

    def __init__(self, attn, qkv_width, rows, device):
        self.attn = attn
        self.heads = attn.heads
        self.head_dim = attn.norm_q.weight.shape[0]
        self.kv_heads = (qkv_width - self.heads * self.head_dim) // (2 * self.head_dim)
        padded = 1 << (self.head_dim - 1).bit_length()
        kv_shape = (1, rows, self.kv_heads, padded)
        self.query = torch.empty((1, rows, self.heads, padded), device=device, dtype=torch.bfloat16)
        self.key = torch.empty(kv_shape, device=device, dtype=torch.bfloat16)
        self.value = torch.zeros(kv_shape, device=device, dtype=torch.bfloat16)

    def fill(self, qkv, cos, sin, rows):
        """RMS-normed, rotated Q/K and V of ``qkv`` rows into buffer rows ``rows``."""
        attn, head_dim = self.attn, self.head_dim
        q_width, kv_width = self.heads * head_dim, self.kv_heads * head_dim
        for src, heads, norm, out in (
            (qkv[:, :q_width], self.heads, attn.norm_q, self.query),
            (qkv[:, q_width : q_width + kv_width], self.kv_heads, attn.norm_k, self.key),
        ):
            act.qk_norm_rope_with_triton(
                src,
                heads,
                head_dim,
                norm.weight,
                cos,
                sin,
                norm.eps,
                weight_mode=act.QK_WEIGHT_DIFFUSERS,
                style=act.ROPE_INTERLEAVED,
                out=out[:, rows],
                out_dim=out.shape[-1],
            )
        self.value[0, rows, :, :head_dim] = qkv[:, q_width + kv_width :].unflatten(
            -1, (self.kv_heads, head_dim)
        )

    def attend(self, block, valid=None):
        key, value = self.key, self.value
        if valid is not None:
            key, value = key[:, valid], value[:, valid]
        out = block.oq_runtime.attention(
            self.query,
            key,
            value,
            int8_allowed=block.oq_int8_attention,
            sm_scale=float(self.attn.scale),
        )
        return out[0, :, :, : self.head_dim].reshape(out.shape[1], self.heads * self.head_dim)


def _self_attention(block, attn, qkv, freqs_cis, mask):
    rows = qkv.shape[0]
    heads = _Heads(attn, qkv.shape[1], rows, qkv.device)
    cos, sin = _rope_tables(freqs_cis, rows)
    heads.fill(qkv, cos, sin, slice(0, rows))
    return heads.attend(block, _valid_keys(mask, rows))


def _gated(h, x, norm, gate):
    act.postnorm_residual_with_triton(h, x, norm.weight, gate, norm.eps, norm=act.NORM_RMS_TORCH)


def _ffn(runtime, h, x, ff_in, ff_out, norm_in, norm_out, gate, mod=None):
    source = runtime.input(
        x,
        x.shape[1],
        norm=act.NORM_RMS_TORCH,
        norm_weight=norm_in.weight,
        rms_eps=norm_in.eps,
        **(mod or {}),
    )
    mid = runtime.linear(ff_in, source, epilogue=gemm.EPILOGUE_SWIGLU)
    _gated(h, runtime.linear(ff_out, runtime.input(mid, mid.shape[1])), norm_out, gate)


def _single(self, hidden_states, attention_mask, image_rotary_emb, temb=None):
    if hidden_states.shape[0] != 1:
        return torch.cat(
            [
                _single(
                    self,
                    hidden_states[i : i + 1],
                    None if attention_mask is None else attention_mask[i : i + 1],
                    image_rotary_emb[i : i + 1],
                    None if temb is None else temb[i : i + 1],
                )
                for i in range(hidden_states.shape[0])
            ]
        )
    runtime, fused = self.oq_runtime, self.oq_fused
    dim = hidden_states.shape[2]
    h = hidden_states[0].clone()
    if self.modulation:
        scale_msa, gate_msa, scale_mlp, gate_mlp = _modulation(self.norm1, temb)
        gate_msa, gate_mlp = torch.tanh(gate_msa), torch.tanh(gate_mlp)
        norm1 = self.norm1.norm
        attn_mod = dict(mod=act.MOD_ONE_PLUS_SCALE, mod_scale=scale_msa)
        mlp_mod = dict(mod=act.MOD_ONE_PLUS_SCALE, mod_scale=scale_mlp)
    else:
        gate_msa = gate_mlp = torch.ones(dim, device=h.device, dtype=h.dtype)
        norm1, attn_mod, mlp_mod = self.norm1, {}, {}
    source = runtime.input(
        h, dim, norm=act.NORM_RMS_TORCH, norm_weight=norm1.weight, rms_eps=norm1.eps, **attn_mod
    )
    qkv = runtime.linear(fused.qkv, source)
    att = _self_attention(self, self.attn, qkv, image_rotary_emb, attention_mask)
    _gated(h, runtime.linear(fused.out, runtime.input(att, att.shape[1])), self.norm2, gate_msa)
    _ffn(
        runtime, h, h, fused.ff_in, fused.ff_out, self.ffn_norm1, self.ffn_norm2, gate_mlp, mlp_mod
    )
    return h[None]


def _double(
    self,
    img_hidden_states,
    instruct_hidden_states,
    img_attention_mask,
    joint_attention_mask,
    image_rotary_emb,
    rotary_emb,
    temb=None,
    encoder_seq_lengths=None,
    seq_lengths=None,
):
    if not self.modulation:
        raise NotImplementedError("fused Boogu double-stream blocks are modulated")
    if img_hidden_states.shape[0] != 1:
        outs = [
            _double(
                self,
                img_hidden_states[i : i + 1],
                instruct_hidden_states[i : i + 1],
                None if img_attention_mask is None else img_attention_mask[i : i + 1],
                None if joint_attention_mask is None else joint_attention_mask[i : i + 1],
                image_rotary_emb[i : i + 1],
                rotary_emb[i : i + 1],
                temb[i : i + 1],
                [encoder_seq_lengths[i]],
                [seq_lengths[i]],
            )
            for i in range(img_hidden_states.shape[0])
        ]
        return torch.cat([o[0] for o in outs]), torch.cat([o[1] for o in outs])
    runtime, fused = self.oq_runtime, self.oq_fused
    l_txt, l_all = int(encoder_seq_lengths[0]), int(seq_lengths[0])
    l_img = l_all - l_txt
    img_in, txt_in = img_hidden_states[0, :l_img], instruct_hidden_states[0, :l_txt]
    dim = img_in.shape[1]

    i_scale1, i_gate_msa, i_scale_mlp, i_gate_mlp = _modulation(self.img_norm1, temb)
    i_scale2, i_shift_mlp, _, _ = _modulation(self.img_norm2, temb)
    i_scale3, i_gate_self, _, _ = _modulation(self.img_norm3, temb)
    t_scale1, t_gate_msa, t_scale_mlp, t_gate_mlp = _modulation(self.instruct_norm1, temb)
    t_scale2, t_shift_mlp, _, _ = _modulation(self.instruct_norm2, temb)

    def modulated(x, norm_zero, scale):
        return runtime.input(
            x,
            dim,
            norm=act.NORM_RMS_TORCH,
            mod=act.MOD_ONE_PLUS_SCALE,
            norm_weight=norm_zero.norm.weight,
            mod_scale=scale,
            rms_eps=norm_zero.norm.eps,
        )

    def mlp_input(x, norm_zero, scale, scale_mlp, shift_mlp):
        # Two norms in a row (RMSNormZero, then ffn_norm1): the first runs as a prologue-only
        # pass, the shift-scale between them with the eager rounding of each step.
        return (1 + scale_mlp) * modulated(x, norm_zero, scale).dense() + shift_mlp

    joint_attn = self.img_instruct_attn
    txt_qkv = runtime.linear(fused.joint_txt_qkv, modulated(txt_in, self.instruct_norm1, t_scale1))
    img_qkv = runtime.linear(fused.joint_img_qkv, modulated(img_in, self.img_norm1, i_scale1))
    heads = _Heads(joint_attn, img_qkv.shape[1], l_all, img_in.device)
    cos, sin = _rope_tables(rotary_emb, l_all)
    heads.fill(txt_qkv, cos[:l_txt], sin[:l_txt], slice(0, l_txt))
    heads.fill(img_qkv, cos[l_txt:], sin[l_txt:], slice(l_txt, l_all))
    att = heads.attend(self, _valid_keys(joint_attention_mask, l_all))
    merged = torch.empty((l_all, dim), device=img_in.device, dtype=torch.bfloat16)
    runtime.linear(fused.joint_txt_out, runtime.input(att[:l_txt], dim), out=merged[:l_txt])
    runtime.linear(fused.joint_img_out, runtime.input(att[l_txt:], dim), out=merged[l_txt:])
    joint = runtime.linear(fused.joint_out, runtime.input(merged, dim))

    self_qkv = runtime.linear(fused.self_qkv, modulated(img_in, self.img_norm3, i_scale3))
    self_att = _self_attention(
        self, self.img_self_attn, self_qkv, image_rotary_emb, img_attention_mask
    )
    self_out = runtime.linear(fused.self_out, runtime.input(self_att, dim))

    img_mlp = mlp_input(img_in, self.img_norm2, i_scale2, i_scale_mlp, i_shift_mlp)
    txt_mlp = mlp_input(txt_in, self.instruct_norm2, t_scale2, t_scale_mlp, t_shift_mlp)

    img = img_in.clone()
    _gated(img, joint[l_txt:], self.img_attn_norm, torch.tanh(i_gate_msa))
    _gated(img, self_out, self.img_self_attn_norm, torch.tanh(i_gate_self))
    _ffn(
        runtime,
        img,
        img_mlp,
        fused.img_ff_in,
        fused.img_ff_out,
        self.img_ffn_norm1,
        self.img_ffn_norm2,
        torch.tanh(i_gate_mlp),
    )
    txt = txt_in.clone()
    _gated(txt, joint[:l_txt], self.instruct_attn_norm, torch.tanh(t_gate_msa))
    _ffn(
        runtime,
        txt,
        txt_mlp,
        fused.txt_ff_in,
        fused.txt_ff_out,
        self.instruct_ffn_norm1,
        self.instruct_ffn_norm2,
        torch.tanh(t_gate_mlp),
    )

    img_out, txt_out = img_hidden_states.clone(), instruct_hidden_states.clone()
    img_out[0, :l_img] = img
    txt_out[0, :l_txt] = txt
    return img_out, txt_out
