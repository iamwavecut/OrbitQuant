"""Kandinsky 6 block forwards with fused LayerNorm and FP32 modulation.

Adapted from Diffusers transformer_kandinsky6.py at d961a388fd02e4db38d17350c8dd9b8abe642e05
(Apache-2.0). The fused reduction can differ from PyTorch at rounding boundaries.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _norm_modulate(x, norm, scale, shift):
    if x.device.type == "cuda" and not torch.is_grad_enabled():
        from orbitquant.kernels.triton_kandinsky6 import norm_modulate

        return norm_modulate(x, norm, scale, shift)
    from diffusers.models.transformers.transformer_kandinsky6 import apply_scale_shift

    return apply_scale_shift(norm(x.float()), x, scale, shift)


def _gate_sum(x, out, gate):
    if x.device.type == "cuda" and not torch.is_grad_enabled():
        from orbitquant.kernels.triton_kandinsky6 import gate_sum

        return gate_sum(x, out, gate)
    from diffusers.models.transformers.transformer_kandinsky6 import apply_gate_sum

    return apply_gate_sum(x, out, gate)


def video_out_forward(self, visual_embed: Tensor, time_embed: Tensor) -> Tensor:
    shift, scale = torch.chunk(self.modulation(time_embed), 2, dim=-1)
    condition_shape = (scale.shape[0],) + (1,) * (visual_embed.ndim - 2) + (scale.shape[-1],)
    x = _norm_modulate(
        visual_embed, self.norm, scale.reshape(condition_shape), shift.reshape(condition_shape)
    )
    x = self.out_layer(x)
    batch, duration, height, width = x.shape[:4]
    p_t, p_h, p_w = self.patch_size
    return (
        x.view(batch, duration, height, width, -1, p_t, p_h, p_w)
        .permute(0, 1, 5, 2, 6, 3, 7, 4)
        .flatten(1, 2)
        .flatten(2, 3)
        .flatten(3, 4)
    )


def audio_out_forward(self, audio_embed: Tensor, time_embed: Tensor) -> Tensor:
    shift, scale = torch.chunk(self.modulation(time_embed), 2, dim=-1)
    x = _norm_modulate(audio_embed, self.norm, scale, shift)
    x = self.norm(x)
    return self.out_layer(x)


def encoder_forward(
    self, x: Tensor, time_embed: Tensor, rope: Tensor, attn_mask: Tensor | None = None
) -> Tensor:
    sa_params, ff_params = torch.chunk(self.text_modulation(time_embed), 2, dim=-1)
    shift, scale, gate = torch.chunk(sa_params, 3, dim=-1)
    x = _gate_sum(
        x,
        self.attn(
            _norm_modulate(x, self.attn_norm, scale, shift), rotary_emb=rope, attn_mask=attn_mask
        ),
        gate,
    )
    shift, scale, gate = torch.chunk(ff_params, 3, dim=-1)
    return _gate_sum(
        x, self.feed_forward(_norm_modulate(x, self.feed_forward_norm, scale, shift)), gate
    )


def decoder_forward(
    self,
    vis: Tensor | None,
    aud: Tensor | None,
    text_v: Tensor,
    text_a: Tensor,
    time_embed: tuple[Tensor, Tensor],
    vis_rope: Tensor | None,
    aud_rope: Tensor | None,
    attn_mask=None,
) -> tuple[Tensor | None, Tensor | None]:
    t_v, t_a = time_embed
    if vis is not None:
        sa_p, ca_p, ff_p = torch.chunk(self.video_dec_block.visual_modulation(t_v), 3, dim=-1)
        shift, scale, gate = torch.chunk(sa_p, 3, dim=-1)
        vis = _gate_sum(
            vis,
            self.video_dec_block.self_attention(
                _norm_modulate(vis, self.video_dec_block.self_attention_norm, scale, shift),
                rotary_emb=vis_rope,
            ),
            gate,
        ).type_as(vis)
        shift, scale, gate_v = torch.chunk(ca_p, 3, dim=-1)
        vis_pre_ca = _norm_modulate(vis, self.video_dec_block.cross_attention_norm, scale, shift)
        vis_out_t = self.video_dec_block.cross_attention(
            vis_pre_ca, encoder_hidden_states=text_v, attn_mask=attn_mask
        )
    if aud is not None:
        sa_p, ca_p, ff_p_a = torch.chunk(self.audio_dec_block.visual_modulation(t_a), 3, dim=-1)
        shift, scale, gate = torch.chunk(sa_p, 3, dim=-1)
        aud = _gate_sum(
            aud,
            self.audio_dec_block.self_attention(
                _norm_modulate(aud, self.audio_dec_block.self_attention_norm, scale, shift),
                rotary_emb=aud_rope,
            ),
            gate,
        ).type_as(aud)
        shift, scale, gate_a = torch.chunk(ca_p, 3, dim=-1)
        aud_pre_ca = _norm_modulate(aud, self.audio_dec_block.cross_attention_norm, scale, shift)
        aud_out_t = self.audio_dec_block.cross_attention(
            aud_pre_ca, encoder_hidden_states=text_a, attn_mask=attn_mask
        )
        aud = _gate_sum(aud, aud_out_t, gate_a).type_as(aud)
        if vis is not None:
            t_va_mod = t_a if not self.fix_modulation else t_v
            t_av_mod = t_v if not self.fix_modulation else t_a
            va_params = self.va_modulation(t_va_mod)
            av_params = self.av_modulation(t_av_mod)
            if self.cross_gates:
                va_shift, va_scale, va_gate = torch.split(
                    va_params, [self.model_dim, self.model_dim, self.model_dim_a], dim=-1
                )
                av_shift, av_scale, av_gate = torch.split(
                    av_params, [self.model_dim_a, self.model_dim_a, self.model_dim], dim=-1
                )
            else:
                va_shift, va_scale, va_gate = torch.chunk(va_params, 3, dim=-1)
                av_shift, av_scale, av_gate = torch.chunk(av_params, 3, dim=-1)
            vis = _gate_sum(vis, vis_out_t, gate_v).type_as(vis)
            vis_for_va = _norm_modulate(vis, self.va_normalization, va_scale, va_shift)
            aud_for_av = _norm_modulate(aud, self.av_normalization, av_scale, av_shift)
            rq_v = vis_rope if self.ca_rope else None
            rk_a = aud_rope if self.ca_rope else None
            vis_from_aud = self.va_cross_attention(
                vis_for_va, encoder_hidden_states=aud_pre_ca, rope_q=rq_v, rope_kv=rk_a
            )
            aud_from_vis = self.av_cross_attention(
                aud_for_av, encoder_hidden_states=vis_pre_ca, rope_q=rk_a, rope_kv=rq_v
            )
            vis = _gate_sum(
                vis, vis_from_aud, va_gate if not self.cross_gates else av_gate
            ).type_as(vis)
            aud = _gate_sum(
                aud, aud_from_vis, av_gate if not self.cross_gates else va_gate
            ).type_as(aud)
    elif vis is not None:
        vis = _gate_sum(vis, vis_out_t, gate_v).type_as(vis)
    if vis is not None:
        shift, scale, gate = torch.chunk(ff_p, 3, dim=-1)
        vis = _gate_sum(
            vis,
            self.video_dec_block.feed_forward(
                _norm_modulate(vis, self.video_dec_block.feed_forward_norm, scale, shift)
            ),
            gate,
        ).type_as(vis)
    if aud is not None:
        shift, scale, gate = torch.chunk(ff_p_a, 3, dim=-1)
        aud = _gate_sum(
            aud,
            self.audio_dec_block.feed_forward(
                _norm_modulate(aud, self.audio_dec_block.feed_forward_norm, scale, shift)
            ),
            gate,
        ).type_as(aud)
    return (vis, aud)
