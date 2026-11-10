"""MiniMax-H3 (``MiniMaxH3Transformer3DModel``) and MiniMax-H3-Image
(``MiniMaxH3ImageTransformer2DModel``) main blocks.

One packed video/audio/text sequence; the AdaLN modulation of every row is a row of a
(timestep, modality) table. The prologues and residual epilogues read that table through the
per-row index directly instead of materializing ``[seq, hidden]`` modulation tensors. Q/K get
RMSNorm and a rotate-half RoPE on the leading 96 channels of each 128-wide head. The token
refiner blocks (text only, once per forward) keep their modules.

MiniMax-H3-Image samples a fixed grid of six timesteps (plus one noise level for reference rows),
and its AdaLN projections read nothing but the timestep embedding. ``tabulate_modulation`` stores
their outputs for those timesteps (``modulation.json`` lists them) and removes the projections:
13.0 of the 33.1 billion parameters. ``install`` then runs the model forward with table lookups;
a timestep outside the table raises an error.

Padding rows form their own attention document (the model's mask pairs live rows with live rows
and padding rows with padding rows); the two documents run as separate mask-free attentions.
"""

from __future__ import annotations

import types

import torch
import torch.nn.functional as F
from torch import nn

from orbitquant.fused.families import int8_attention_allowed, specs
from orbitquant.fused.groups import RowSource
from orbitquant.kernels import triton_activation as act
from orbitquant.kernels import triton_int8_gemm as gemm

FAMILY = "minimax_h3"
MODEL_CLASSES = ("MiniMaxH3Transformer3DModel", "MiniMaxH3ImageTransformer2DModel")
# Rows per timestep in the AdaLN tables: video/image, text, audio.
MODALITIES = 3


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


class ModulationTable(nn.Module):
    """The outputs of one block's ``adaln_proj`` for a fixed set of timesteps.

    ``table`` holds ``adaln_proj.linear(silu(temb))`` reshaped to ``(timesteps * MODALITIES,
    6 * hidden)``, the row layout the block indexes with ``timestep_index * MODALITIES + tag``.
    ``forward`` takes the table row of each distinct timestep of the call."""

    def __init__(self, timesteps: int, width: int, dtype=torch.bfloat16, device=None):
        super().__init__()
        table = torch.empty(timesteps * MODALITIES, width, dtype=dtype, device=device)
        self.register_buffer("table", table)

    def forward(self, rows: torch.Tensor) -> tuple[torch.Tensor, ...]:
        tags = torch.arange(MODALITIES, device=rows.device)
        index = (rows[:, None] * MODALITIES + tags).reshape(-1)
        return self.table.index_select(0, index).chunk(6, dim=-1)


class ModulationKeys(nn.Module):
    """The tabulated timesteps and the output norm's shift/scale for each of them."""

    def __init__(self, timesteps: int, hidden: int, dtype=torch.bfloat16, device=None):
        super().__init__()
        keys = torch.empty(timesteps, dtype=torch.float32, device=device)
        self.register_buffer("timesteps", keys)
        output = torch.empty(timesteps, 2 * hidden, dtype=dtype, device=device)
        self.register_buffer("output", output)

    def rows(self, timestep: torch.Tensor) -> torch.Tensor:
        """Table row of every timestep (one host sync)."""
        timestep = timestep.to(device=self.timesteps.device, dtype=torch.float32)
        match = timestep[:, None] == self.timesteps[None]
        if not bool(match.any(dim=1).all()):
            raise ValueError(
                f"timesteps {timestep.tolist()} are not all in the modulation table "
                f"{self.timesteps.tolist()}; this checkpoint supports the six-step sampling grid "
                "of MiniMax-H3-Image only"
            )
        return match.to(torch.int32).argmax(dim=1)


def add_modulation_tables(model, timesteps: int, device=None) -> None:
    """Replace the AdaLN projections and the timestep MLP by empty tables (before a tabulated
    checkpoint loads)."""
    hidden = model.config.hidden_size
    dtype = model.transformer_blocks[0].adaln_proj.linear.weight.dtype
    if dtype not in (torch.bfloat16, torch.float16, torch.float32):
        dtype = torch.bfloat16
    for block in model.transformer_blocks:
        block.adaln_proj = ModulationTable(timesteps, 6 * hidden, dtype=dtype, device=device)
    model.oq_modulation = ModulationKeys(timesteps, hidden, dtype=dtype, device=device)
    del model.time_embedder
    del model.time_proj
    del model.norm_out.linear


@torch.no_grad()
def tabulate_modulation(model, timesteps) -> list[float]:
    """Compute the AdaLN tables of ``timesteps`` (unscaled, as the pipeline passes them) from the
    loaded weights and remove the projections. Each timestep is projected on its own, as a
    text-to-image call projects its one timestep."""
    from diffusers.models.modeling_utils import get_parameter_dtype

    keys = sorted({float(torch.tensor(t, dtype=torch.float32)) for t in timesteps})
    device = model.transformer_blocks[0].adaln_proj.linear.weight.device
    embedder_dtype = get_parameter_dtype(model.time_embedder)
    tables = [[] for _ in model.transformer_blocks]
    output = []
    for t in keys:
        timestep = torch.tensor([t], dtype=torch.float32, device=device)
        temb = model.time_embedder(model.time_proj(timestep).to(embedder_dtype))
        for i, block in enumerate(model.transformer_blocks):
            shift_scale = block.adaln_proj(temb)
            tables[i].append(torch.cat(shift_scale, dim=-1))
        linear = model.norm_out.linear
        output.append(linear(F.silu(temb).to(linear.weight.dtype)))
    dtype = tables[0][0].dtype
    add_modulation_tables(model, len(keys), device=device)
    for block, rows in zip(model.transformer_blocks, tables, strict=True):
        block.adaln_proj.table.copy_(torch.cat(rows, dim=0))
    model.oq_modulation.timesteps.copy_(torch.tensor(keys, dtype=torch.float32))
    model.oq_modulation.output.copy_(torch.cat(output, dim=0).to(dtype))
    install(model)
    return keys


def install(model):
    image_model = type(model).__name__ == "MiniMaxH3ImageTransformer2DModel"
    if image_model and hasattr(model, "oq_modulation"):
        model.forward = types.MethodType(_image_forward, model)


def _image_forward(
    self,
    hidden_states,
    encoder_hidden_states,
    timestep,
    timestep_indices,
    token_tags,
    position_ids,
    image_indices,
    text_indices,
    pdd_step_index,
    return_dict=True,
):
    """``MiniMaxH3ImageTransformer2DModel.forward`` with tabulated modulation."""
    from diffusers.models.modeling_outputs import Transformer2DModelOutput
    from diffusers.models.modeling_utils import get_parameter_dtype

    rotary_emb = self.rope(position_ids)
    image_embeds = self.proj_in(hidden_states.to(get_parameter_dtype(self.proj_in)))
    text_embeds = self.token_refiner(
        self.context_embedder(encoder_hidden_states.to(get_parameter_dtype(self.context_embedder)))
    )
    sequence_length = position_ids.shape[0]
    batch, width = text_embeds.shape[0], text_embeds.shape[-1]
    hidden_states = text_embeds.new_zeros((batch, sequence_length, width))
    hidden_states = hidden_states.index_copy(1, text_indices, text_embeds)
    hidden_states = hidden_states.index_copy(1, image_indices, image_embeds.to(text_embeds.dtype))

    rows = self.oq_modulation.rows(timestep)
    adaln_indices = timestep_indices * MODALITIES + token_tags
    for block in self.transformer_blocks:
        hidden_states = block(hidden_states, rows, adaln_indices, rotary_emb)

    proj_out = self.pdd_proj_out[pdd_step_index]
    shift, scale = self.oq_modulation.output.index_select(0, rows).chunk(2, dim=-1)
    hidden_states = self.norm_out.norm(hidden_states)
    scale = scale.index_select(0, timestep_indices)
    shift = shift.index_select(0, timestep_indices)
    hidden_states = hidden_states * (1.0 + scale) + shift
    output = proj_out(hidden_states.to(get_parameter_dtype(proj_out)))
    output = output.index_select(1, image_indices)
    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)
