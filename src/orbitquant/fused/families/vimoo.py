"""Vimoo 1.0 (``VimooTransformer3DModel``) dual-stream MMDiT blocks.

Each of the 40 blocks keeps separate video and text streams that meet in one joint attention.
Per stream: LayerNorm (no affine) and the ``(1 + scale) * x + shift`` modulation run in the
activation-quantization prologue, Q|K|V is one GEMM, per-head Q/K RMSNorm and (video only) the
interleaved rotary embedding are one kernel, and both output projections and the feed-forward
output add their gated residual in the GEMM epilogue. The feed-forward input GEMM applies GELU
in its epilogue: tanh-approximated for standard checkpoints, exact for Lite ones.

The transformer moves the valid text tokens of every sample to the front of the text stream
(Qwen, glyph and image tokens; padding last). Padding is a key-masked suffix that no valid token
reads, so the fused blocks process only the valid prefix: the joint keys and values are one
contiguous ``[video | valid text]`` row range, and padded text rows leave the block unchanged.

Lite checkpoints (``numerical_profile="lite"``) keep the residual streams in the dtype the
transformer hands over (FP32 when conditioning arrives in FP32) and store the output and
feed-forward biases next to bias-free projections; the epilogues add them. For the observed
first frame of image-to-video sampling the video tokens of that frame read the second row of
two-row modulation tables.

Every group takes 8-bit activations (per-token absmax INT8 of the rotated input); the GEMMs are
INT8 either way. The transformer runs one sample at a time through the fused blocks.
"""

from __future__ import annotations

import torch

from orbitquant.fused.families import int8_attention_allowed, specs
from orbitquant.kernels import triton_activation as act
from orbitquant.kernels import triton_int8_gemm as gemm

FAMILY = "vimoo"
MODEL_CLASSES = ("VimooTransformer3DModel",)

STREAMS = ("img", "txt")
GROUPS = tuple(f"{s}_{g}" for s in STREAMS for g in ("qkv", "out", "ff_in", "ff_out"))
ACTIVATION_BITS = dict.fromkeys(GROUPS, 8)
EPS = 1e-6
# Feed-forward rows per pass: bounds the [rows, 4 * hidden] intermediate of long videos.
FF_CHUNK_ROWS = 32768


def _kind(model) -> str:
    profile = getattr(model, "numerical_profile", None)
    if profile is None:
        profile = getattr(getattr(model, "config", None), "numerical_profile", "standard")
    return "lite" if profile == "lite" else "standard"


def blocks(model):
    kind = _kind(model)
    return [(f"double_blocks.{i}", kind) for i in range(len(model.double_blocks))]


def _paths(kind):
    """Projection paths (relative to the block) of each stream: Q, K, V, out, ff_in, ff_out."""
    if kind == "lite":
        return {
            "img": (
                ["attn.to_q", "attn.to_k", "attn.to_v"],
                "attn.to_out.0.projection",
                "img_mlp.in_proj.projection",
                "img_mlp.out_proj.projection",
            ),
            "txt": (
                ["attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj"],
                "attn.to_add_out.projection",
                "txt_mlp.in_proj.projection",
                "txt_mlp.out_proj.projection",
            ),
        }
    return {
        "img": (
            ["attn.to_q", "attn.to_k", "attn.to_v"],
            "attn.to_out.0",
            "img_mlp.net.0.proj",
            "img_mlp.net.2",
        ),
        "txt": (
            ["attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj"],
            "attn.to_add_out",
            "txt_mlp.net.0.proj",
            "txt_mlp.net.2",
        ),
    }


def groups(kind, block):
    out = {}
    for stream, (qkv, attn_out, ff_in, ff_out) in _paths(kind).items():
        out[f"{stream}_qkv"] = qkv
        out[f"{stream}_out"] = [attn_out]
        out[f"{stream}_ff_in"] = [ff_in]
        out[f"{stream}_ff_out"] = [ff_out]
    return specs(**out)


def prepare(block, kind):
    attn = block.attn
    block.oq_int8_attention = int8_attention_allowed(
        attn.norm_qk.query_weight,
        attn.norm_qk.key_weight,
        attn.norm_added_qk.query_weight,
        attn.norm_added_qk.key_weight,
    )


def forward(kind):
    return _lite if kind == "lite" else _standard


def _cached(runtime, name, key, build, slots=2):
    """A small per-forward cache keyed by tensor identity: the rope table, the text mask and the
    observed-frame mask are shared by every block of one transformer call."""
    cache = runtime.__dict__.setdefault(name, [])
    for entry in cache:
        if entry[0] is key:
            return entry[1]
    value = build()
    cache.insert(0, (key, value))
    del cache[slots:]
    return value


def _valid_text(runtime, text_mask, batch, rows):
    """Valid text rows of every sample. They must be a prefix (the transformer sorts them first)."""
    if text_mask is None:
        return [rows] * batch

    def build():
        mask = text_mask.reshape(batch, -1)[:, :rows].bool()
        counts = mask.sum(dim=1)
        prefix = torch.arange(rows, device=mask.device)[None, :] < counts[:, None]
        if not bool(torch.equal(mask, prefix)):
            raise ValueError("fused Vimoo blocks need the valid text tokens first")
        return [int(c) for c in counts.tolist()]

    return _cached(runtime, "vimoo_text", text_mask, build)


def _rope(runtime, freqs_cis, rows):
    cos, sin = freqs_cis

    def build():
        return (
            cos.reshape(-1, cos.shape[-1]).float().contiguous(),
            sin.reshape(-1, sin.shape[-1]).float().contiguous(),
        )

    cos, sin = _cached(runtime, "vimoo_rope", freqs_cis[0], build)
    if cos.shape[0] != rows:
        raise ValueError(f"rotary table has {cos.shape[0]} rows for {rows} video tokens")
    return cos, sin


def _observed_index(runtime, observed_mask, rows):
    """Row 1 of the video modulation tables for the observed first-frame tokens, row 0 else."""
    if observed_mask is None:
        return None

    def build():
        return observed_mask.reshape(-1)[:rows].to(torch.int32).contiguous()

    return _cached(runtime, "vimoo_observed", observed_mask, build)


def _mods(chunks, sample, observed):
    """The six modulation vectors of one sample: ``[dim]`` rows, or ``[2, dim]`` tables
    (unobserved | observed) when the video stream has an observed first frame."""
    if observed:
        return [torch.stack((c[0, sample], c[1, sample])).contiguous() for c in chunks]
    return [c[sample].contiguous() for c in chunks]


def _qk(src, norm_weight, heads, head_dim, weight_mode, out, cos=None, sin=None):
    act.qk_norm_rope_with_triton(
        src,
        heads,
        head_dim,
        norm_weight,
        cos,
        sin,
        EPS,
        weight_mode=weight_mode,
        style=act.ROPE_INTERLEAVED,
        out=out,
    )


def _bias(module):
    """The separate bias of a Lite deferred-bias projection."""
    return getattr(module, "bias", None)


def _stream_update(self, runtime, fused, stream, h, att, mods, index, ff_epilogue, biases):
    """Output projection and feed-forward of one stream, both added to ``h`` in place."""
    shift2, scale2, gate1, gate2 = mods
    dim = h.shape[1]
    out_bias, ff_in_bias, ff_out_bias = biases
    runtime.linear(
        fused[f"{stream}_out"],
        runtime.input(att, att.shape[1]),
        epilogue=gemm.EPILOGUE_RESIDUAL_GATE,
        residual=h,
        gate=gate1,
        gate_index=index,
        out=h,
        bias=out_bias,
    )
    for start in range(0, h.shape[0], FF_CHUNK_ROWS):
        rows = h[start : start + FF_CHUNK_ROWS]
        rows_index = None if index is None else index[start : start + FF_CHUNK_ROWS]
        source = runtime.input(
            rows,
            dim,
            norm=act.NORM_LAYER,
            mod=act.MOD_SHIFT_SCALE,
            mod_scale=scale2,
            mod_shift=shift2,
            mod_index=rows_index,
            rms_eps=EPS,
        )
        mid = runtime.linear(
            fused[f"{stream}_ff_in"], source, epilogue=ff_epilogue, bias=ff_in_bias
        )
        runtime.linear(
            fused[f"{stream}_ff_out"],
            runtime.input(mid, mid.shape[1]),
            epilogue=gemm.EPILOGUE_RESIDUAL_GATE,
            residual=rows,
            gate=gate2,
            gate_index=rows_index,
            out=rows,
            bias=ff_out_bias,
        )


def _joint_attention(self, runtime, fused, h_img, h_txt, mods, index, rope, qk_mode):
    """Joint attention of one sample over ``[video | valid text]`` rows, before the output
    projections: ``[rows, dim]`` bf16."""
    attn = self.attn
    img_rows, dim = h_img.shape
    rows = img_rows + h_txt.shape[0]
    heads = attn.heads
    head_dim = dim // heads
    (i_shift, i_scale), (t_shift, t_scale) = mods
    qkv = torch.empty((rows, 3 * dim), device=h_img.device, dtype=torch.bfloat16)
    for stream, h, shift, scale, idx, start in (
        ("img", h_img, i_shift, i_scale, index, 0),
        ("txt", h_txt, t_shift, t_scale, None, img_rows),
    ):
        if h.shape[0] == 0:
            continue
        source = runtime.input(
            h,
            dim,
            norm=act.NORM_LAYER,
            mod=act.MOD_SHIFT_SCALE,
            mod_scale=scale,
            mod_shift=shift,
            mod_index=idx,
            rms_eps=EPS,
        )
        runtime.linear(fused[f"{stream}_qkv"], source, out=qkv[start : start + h.shape[0]])
    query = torch.empty((1, rows, heads, head_dim), device=qkv.device, dtype=torch.bfloat16)
    key = torch.empty_like(query)
    cos, sin = rope
    for norm, start, stop, table in (
        (attn.norm_qk, 0, img_rows, (cos, sin)),
        (attn.norm_added_qk, img_rows, rows, (None, None)),
    ):
        if stop == start:
            continue
        _qk(
            qkv[start:stop, :dim],
            norm.query_weight,
            heads,
            head_dim,
            qk_mode,
            query[:, start:stop],
            *table,
        )
        _qk(
            qkv[start:stop, dim : 2 * dim],
            norm.key_weight,
            heads,
            head_dim,
            qk_mode,
            key[:, start:stop],
            *table,
        )
    value = qkv[:, 2 * dim :].unflatten(-1, (heads, head_dim))[None]
    out = runtime.attention(query, key, value, int8_allowed=self.oq_int8_attention)
    return out.reshape(rows, dim)


def _run(
    self,
    hidden_states,
    encoder_hidden_states,
    vec,
    freqs_cis,
    text_mask,
    observed_image_mask,
    compute_dtype,
    *,
    lite,
):
    runtime, fused = self.oq_runtime, self.oq_fused
    batch, img_rows, _ = hidden_states.shape
    txt_rows = encoder_hidden_states.shape[1]
    observed = observed_image_mask is not None
    if compute_dtype is None:
        compute_dtype = hidden_states.dtype
    # The modulation dtypes of the eager block.
    modulation_dtype = compute_dtype if lite else None
    mixed = getattr(self, "mixed_precision_modulation", False)
    img_dtype = hidden_states.dtype if mixed else modulation_dtype
    txt_dtype = encoder_hidden_states.dtype if mixed else modulation_dtype
    img_mod = self.img_mod(vec, dtype=img_dtype).chunk(6, dim=-1)
    txt_mod = self.txt_mod(vec[0] if observed else vec, dtype=txt_dtype).chunk(6, dim=-1)
    qk_mode = act.QK_WEIGHT_TORCH if lite else act.QK_WEIGHT_DIFFUSERS
    ff_epilogue = gemm.EPILOGUE_GELU_ERF if lite else gemm.EPILOGUE_GELU_TANH
    valid = _valid_text(runtime, text_mask, batch, txt_rows)
    rope = _rope(runtime, freqs_cis, img_rows)
    index = _observed_index(runtime, observed_image_mask, img_rows)
    if lite:
        attn = self.attn
        img_biases = (
            _bias(attn.to_out[0]),
            _bias(self.img_mlp.in_proj),
            _bias(self.img_mlp.out_proj),
        )
        txt_biases = (
            _bias(attn.to_add_out),
            _bias(self.txt_mlp.in_proj),
            _bias(self.txt_mlp.out_proj),
        )
    else:
        img_biases = txt_biases = (None, None, None)

    # Row-major copies: the standard patch embedding hands over a transposed (column-major) video
    # stream, and the fused kernels update the rows in place.
    img_out = hidden_states.clone(memory_format=torch.contiguous_format)
    txt_out = encoder_hidden_states.clone(memory_format=torch.contiguous_format)
    for sample in range(batch):
        n_txt = valid[sample]
        h_img = img_out[sample]
        h_txt = txt_out[sample, :n_txt]
        i_shift1, i_scale1, i_gate1, i_shift2, i_scale2, i_gate2 = _mods(img_mod, sample, observed)
        t_shift1, t_scale1, t_gate1, t_shift2, t_scale2, t_gate2 = _mods(txt_mod, sample, False)
        att = _joint_attention(
            self,
            runtime,
            fused,
            h_img,
            h_txt,
            ((i_shift1, i_scale1), (t_shift1, t_scale1)),
            index,
            rope,
            qk_mode,
        )
        _stream_update(
            self,
            runtime,
            fused,
            "img",
            h_img,
            att[:img_rows],
            (i_shift2, i_scale2, i_gate1, i_gate2),
            index,
            ff_epilogue,
            img_biases,
        )
        if n_txt:
            _stream_update(
                self,
                runtime,
                fused,
                "txt",
                h_txt,
                att[img_rows:],
                (t_shift2, t_scale2, t_gate1, t_gate2),
                None,
                ff_epilogue,
                txt_biases,
            )
        del att
    return img_out, txt_out


def _standard(
    self,
    hidden_states,
    encoder_hidden_states,
    vec,
    freqs_cis,
    text_mask=None,
    attention_plan=None,
    attention_kwargs=None,
    compute_dtype=None,
    observed_image_mask=None,
):
    del attention_plan, attention_kwargs
    if observed_image_mask is not None:
        raise ValueError("observed first-frame modulation needs a Lite checkpoint")
    return _run(
        self,
        hidden_states,
        encoder_hidden_states,
        vec,
        freqs_cis,
        text_mask,
        None,
        compute_dtype,
        lite=False,
    )


def _lite(
    self,
    hidden_states,
    encoder_hidden_states,
    vec,
    freqs_cis,
    text_mask=None,
    attention_plan=None,
    attention_kwargs=None,
    compute_dtype=None,
    observed_image_mask=None,
):
    del attention_plan, attention_kwargs
    return _run(
        self,
        hidden_states,
        encoder_hidden_states,
        vec,
        freqs_cis,
        text_mask,
        observed_image_mask,
        compute_dtype,
        lite=True,
    )
