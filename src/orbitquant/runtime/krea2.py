"""Fused inference runtime for Krea 2 with OrbitQuant W4A4 weights.

``install(transformer)`` regroups the projection weights of every ``Krea2TransformerBlock``
(Q|K|V|gate, out, gate/up interleaved for the SwiGLU epilogue, down) into a ``fused_blocks``
submodule and replaces the original projection modules with placeholders. ``Krea2FastRunner``
then runs a ``diffusers.Krea2Pipeline`` with:

* text encoding without the 512-token padding (right-padded to a bucket; the encoder is
  causal, so the valid outputs do not change) and without the unused vision tower,
* a compacted DiT sequence (padded text lanes dropped) so attention needs no mask and runs
  natively grouped-query in Flash or cuDNN SDPA, or in the INT8 Q.K^T / FP16 P.V kernel
  (``attention="int8"``, ``orbitquant.kernels.triton_attention``),
* text fusion, text projection and rotary tables computed once per image,
* per block: one activation quantization per shared input fused with its RMSNorm/modulation
  or gate product, one grouped INT8 GEMM for Q|K|V|gate with the sigmoid gate in its
  epilogue, a Q/K RMSNorm + RoPE kernel, and the SwiGLU and the gated residual updates fused
  into GEMM epilogues. The W4 weights of each group are decoded to INT8 right before their
  GEMM; ``prefetch=True`` runs the next decode on a side stream instead.

``down="int8"`` stores the down projections as per-row INT8 (W8A8, ``Int8RowLinear``
numerics) instead of the source BF16, which is 3x faster on GeForce cards and halves their
memory; ``down="bf16"`` keeps them as shipped.

Building the fused buffers copies the block weights into new host memory. ``save_fused``
writes them to a safetensors file once; ``install(..., fused_path=...)`` then maps that file
instead, so the fused weights stay reclaimable page cache like the rest of the checkpoint.
"""

from __future__ import annotations

import os
import types
from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from orbitquant.int8_head import quantize_int8_rows
from orbitquant.kernels import triton_activation as act
from orbitquant.kernels import triton_int8_gemm as gemm
from orbitquant.kernels.triton_attention import int8_attention_with_triton
from orbitquant.kernels.triton_cuda import fit_int8_centroid_surrogate
from orbitquant.layers import OrbitQuantLinear

__all__ = ["FusedKrea2Blocks", "Krea2FastRunner", "install", "save_fused"]

FUSED_FORMAT = "orbitquant-krea2-fused"
FUSED_FORMAT_VERSION = "1"

PROMPT_PREFIX = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n"
)
PROMPT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
PROMPT_PREFIX_TOKENS = 34
PROMPT_SUFFIX_TOKENS = 5
MAX_SEQUENCE_LENGTH = 512
TEXT_BUCKET = 64
DISTILLED_SHIFT_MU = 1.15

# INT8 Q/K loses the small channels of heads whose Q/K RMSNorm scales one channel far above
# the rest (Krea 2 block 0 scales a channel 36x the median); those blocks keep BF16 attention.
INT8_ATTENTION_MAX_NORM_SPREAD = 3.0

_ATTENTION_BACKENDS = {
    "flash": SDPBackend.FLASH_ATTENTION,
    "cudnn": SDPBackend.CUDNN_ATTENTION,
    "int8": None,
}


def _codes_and_alpha(layer: OrbitQuantLinear) -> tuple[torch.Tensor, float]:
    _, act_scale = fit_int8_centroid_surrogate(layer.activation_codebook.centroids)
    codes, w_scale = fit_int8_centroid_surrogate(layer.weight_codebook.centroids)
    return codes, float(act_scale * w_scale)


class FusedGroup(nn.Module):
    """Row-concatenated packed W4 weights of projections that read the same activation."""

    def __init__(
        self,
        packed: torch.Tensor,
        row_norms: torch.Tensor,
        codes: torch.Tensor,
        alpha: float,
        in_features: int,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.alpha = float(alpha)
        self.register_buffer("packed", packed)
        self.register_buffer("row_norms", row_norms)
        self.register_buffer("codes", codes.to(torch.int8))
        self.out_features = int(packed.shape[0])

    @classmethod
    def from_layers(
        cls, layers: list[OrbitQuantLinear], interleave: int | None = None
    ) -> FusedGroup:
        codes, alpha = _group_constants(layers)
        packed = [layer.packed_weight_indices.reshape(layer.out_features, -1) for layer in layers]
        norms = [layer.row_norms.float().reshape(-1) for layer in layers]
        if interleave is not None:
            (gate_p, up_p), (gate_n, up_n) = packed, norms
            packed, norms = [], []
            for start in range(0, gate_p.shape[0], interleave):
                end = start + interleave
                packed += [gate_p[start:end], up_p[start:end]]
                norms += [gate_n[start:end], up_n[start:end]]
        return cls(
            torch.cat(packed, dim=0).contiguous(),
            torch.cat(norms, dim=0).contiguous(),
            codes,
            alpha,
            layers[0].in_features,
        )

    @classmethod
    def from_stored(
        cls, layers: list[OrbitQuantLinear], packed: torch.Tensor, row_norms: torch.Tensor
    ) -> FusedGroup:
        codes, alpha = _group_constants(layers)
        rows = sum(layer.out_features for layer in layers)
        expected = (rows, layers[0].in_features // 2)
        if tuple(packed.shape) != expected or tuple(row_norms.shape) != (rows,):
            raise ValueError(
                f"stored fused group {tuple(packed.shape)} does not match the layers {expected}"
            )
        return cls(packed, row_norms, codes, alpha, layers[0].in_features)

    def decode(self, buffer: torch.Tensor) -> torch.Tensor:
        view = buffer[: self.out_features * self.in_features].view(
            self.out_features, self.in_features
        )
        return act.decode_w4_to_int8_with_triton(
            self.packed, self.codes, self.out_features, self.in_features, out=view
        )


def _group_constants(layers: list[OrbitQuantLinear]) -> tuple[torch.Tensor, float]:
    codes, alpha = _codes_and_alpha(layers[0])
    for layer in layers:
        other_codes, other_alpha = _codes_and_alpha(layer)
        if (
            not isinstance(layer, OrbitQuantLinear)
            or layer.weight_bits != 4
            or layer.bias is not None
            or other_alpha != alpha
            or not torch.equal(other_codes, codes)
        ):
            raise ValueError("fused groups need bias-free W4 OrbitQuant layers sharing codebooks")
    return codes, alpha


class Int8Rows(nn.Module):
    """Per-row INT8 weights of a W8A8 projection (``Int8RowLinear`` numerics)."""

    def __init__(self, q: torch.Tensor, scales: torch.Tensor):
        super().__init__()
        self.register_buffer("q", q)
        self.register_buffer("scales", scales)

    @classmethod
    def from_weight(cls, weight: torch.Tensor) -> Int8Rows:
        return cls(*quantize_int8_rows(weight))


class Bf16Weight(nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.register_buffer("weight", weight.detach())


_GROUP_LAYERS = {
    "qkvg": lambda attn, ff: [attn.to_q, attn.to_k, attn.to_v, attn.to_gate],
    "out": lambda attn, ff: [attn.to_out[0]],
    "gateup": lambda attn, ff: [ff.gate, ff.up],
}


class FusedKrea2Block(nn.Module):
    def __init__(self, block, down: str, stored: Callable[[str], torch.Tensor] | None = None):
        super().__init__()
        attn, ff = block.attn, block.ff
        for name, layers_of in _GROUP_LAYERS.items():
            layers = layers_of(attn, ff)
            interleave = gemm.SWIGLU_INTERLEAVE if name == "gateup" else None
            if stored is None:
                group = FusedGroup.from_layers(layers, interleave=interleave)
            else:
                group = FusedGroup.from_stored(
                    layers, stored(f"{name}.packed"), stored(f"{name}.row_norms")
                )
            setattr(self, name, group)
        weight = ff.down.weight
        if down == "int8":
            if stored is None:
                self.down = Int8Rows.from_weight(weight)
            else:
                self.down = Int8Rows(stored("down.q"), stored("down.scales"))
                if self.down.q.shape != weight.shape:
                    raise ValueError("stored down projection does not match the layer")
        elif down == "bf16":
            self.down = Bf16Weight(weight if stored is None else stored("down.weight"))
        else:
            raise ValueError(f"unknown down projection mode {down!r}")
        self.q_width = attn.num_heads * attn.head_dim
        self.kv_width = attn.num_kv_heads * attn.head_dim
        scales = torch.cat([attn.norm_q.weight, attn.norm_k.weight]).float().add(1.0).abs()
        self.int8_attention = bool(scales.max() <= INT8_ATTENTION_MAX_NORM_SPREAD * scales.median())


class FusedKrea2Blocks(nn.Module):
    """Fused weights and runtime state for all transformer blocks of one Krea 2 DiT."""

    def __init__(
        self,
        transformer,
        *,
        down: str = "int8",
        attention: str = "flash",
        prefetch: bool = False,
        stored: Callable[[str], torch.Tensor] | None = None,
    ):
        super().__init__()
        if attention not in _ATTENTION_BACKENDS:
            raise ValueError(f"unknown attention {attention!r}")
        blocks = transformer.transformer_blocks
        first = blocks[0].attn.to_q
        # Plain attribute, not a submodule: only the RPBH/codebook constants are needed.
        self.__dict__["act_spec"] = types.SimpleNamespace(
            rotation=first.rotation, codebook=first.activation_codebook, eps=first.activation_eps
        )
        self.down_mode = down
        self.attention = _ATTENTION_BACKENDS[attention]
        self.prefetch = prefetch
        self.rms_eps = float(blocks[0].norm1.eps)
        self.fused = nn.ModuleList(
            [
                FusedKrea2Block(
                    block,
                    down,
                    None if stored is None else (lambda key, i=index: stored(f"blocks.{i}.{key}")),
                )
                for index, block in enumerate(blocks)
            ]
        )
        self.max_decode = (
            max(max(f.qkvg.out_features, f.gateup.out_features) for f in self.fused)
            * first.in_features
        )
        self.order = [
            (i, name) for i in range(len(self.fused)) for name in ("qkvg", "out", "gateup")
        ]
        self._reset_runtime()

    def _reset_runtime(self):
        self._quantizer = None
        self._buffers_ = None
        self._stream = None
        self._pending = {}
        self._free = [None, None]
        self._cursor = 0

    def _runtime(self, device):
        if self._quantizer is None or self._buffers_[0].device != device:
            spec = self.act_spec
            self._quantizer = act.ActivationQuantizer(
                spec.rotation, spec.codebook, spec.eps, device
            )
            count = 2 if self.prefetch else 1
            self._buffers_ = [
                torch.empty(self.max_decode, device=device, dtype=torch.int8) for _ in range(count)
            ]
            self._stream = torch.cuda.Stream(device=device) if self.prefetch else None
            self._pending = {}
            self._free = [None, None]
            self._cursor = 0
        return self._quantizer

    def release(self):
        """Drop runtime buffers (decode double buffer, side stream) before offloading."""
        if self._stream is not None:
            torch.cuda.current_stream().wait_stream(self._stream)
        self._reset_runtime()

    def _issue(self, position: int):
        index, name = self.order[position % len(self.order)]
        slot = position % 2
        group = getattr(self.fused[index], name)
        with torch.cuda.stream(self._stream):
            if self._free[slot] is not None:
                self._stream.wait_event(self._free[slot])
            view = group.decode(self._buffers_[slot])
            ready = torch.cuda.Event()
            ready.record(self._stream)
        self._pending[position] = (view, ready)

    def _weight(self, index: int, name: str) -> torch.Tensor:
        group = getattr(self.fused[index], name)
        if not self.prefetch:
            return group.decode(self._buffers_[0])
        position = self._cursor
        expected = self.order[position % len(self.order)]
        if expected != (index, name):
            raise RuntimeError(f"decode order mismatch: expected {expected}, got {(index, name)}")
        if position not in self._pending:
            self._issue(position)
        view, ready = self._pending.pop(position)
        torch.cuda.current_stream().wait_event(ready)
        self._cursor += 1
        return view

    def _consumed(self):
        # The main stream has queued the GEMM that reads the last weight: the next decode may
        # start into the other buffer once the GEMM before it has finished.
        if not self.prefetch:
            return
        done = torch.cuda.Event()
        done.record(torch.cuda.current_stream())
        self._free[(self._cursor - 1) % 2] = done
        self._issue(self._cursor)

    def forward_block(self, block, index: int, hidden: torch.Tensor, temb_mod: torch.Tensor, rope):
        fused = self.fused[index]
        quantize = self._runtime(hidden.device)
        attn = block.attn
        batch, seq, dim = hidden.shape
        if batch != 1:
            raise ValueError("the fused Krea 2 runtime runs one image at a time")
        h = hidden.reshape(seq, dim)
        modulation = temb_mod.unflatten(-1, (6, -1)) + block.scale_shift_table
        prescale, preshift, pregate, postscale, postshift, postgate = (
            t.reshape(dim).contiguous() for t in modulation.unbind(-2)
        )

        codes, norms = quantize(
            h,
            prologue=act.PROLOGUE_NORM_MODULATE,
            norm_weight=block.norm1.weight,
            mod_scale=prescale,
            mod_shift=preshift,
            rms_eps=self.rms_eps,
        )
        group = fused.qkvg
        qkvg = gemm.matmul_int8_scaled_with_triton(
            codes,
            self._weight(index, "qkvg"),
            norms,
            group.row_norms,
            alpha=group.alpha,
            epilogue=gemm.EPILOGUE_SIGMOID_TAIL,
            sig_from=group.out_features - dim,
        )
        self._consumed()
        qw, kw = fused.q_width, fused.kv_width
        cos, sin = rope
        query = act.qk_norm_rope_with_triton(
            qkvg[:, :qw],
            attn.num_heads,
            attn.head_dim,
            attn.norm_q.weight,
            cos,
            sin,
            attn.norm_q.eps,
        )
        key = act.qk_norm_rope_with_triton(
            qkvg[:, qw : qw + kw],
            attn.num_kv_heads,
            attn.head_dim,
            attn.norm_k.weight,
            cos,
            sin,
            attn.norm_k.eps,
        )
        value = qkvg[:, qw + kw : qw + 2 * kw].unflatten(-1, (attn.num_kv_heads, attn.head_dim))[
            None
        ]
        if self.attention is None and fused.int8_attention:
            att = int8_attention_with_triton(query, key, value).reshape(seq, dim)
        else:
            with sdpa_kernel(self.attention or SDPBackend.FLASH_ATTENTION):
                att = F.scaled_dot_product_attention(
                    query.transpose(1, 2),
                    key.transpose(1, 2),
                    value.transpose(1, 2),
                    enable_gqa=attn.num_heads != attn.num_kv_heads,
                )
            att = att.transpose(1, 2).reshape(seq, dim)

        codes, norms = quantize(att, prologue=act.PROLOGUE_MULTIPLY, x2=qkvg[:, qw + 2 * kw :])
        group = fused.out
        h = gemm.matmul_int8_scaled_with_triton(
            codes,
            self._weight(index, "out"),
            norms,
            group.row_norms,
            alpha=group.alpha,
            epilogue=gemm.EPILOGUE_RESIDUAL_GATE,
            residual=h,
            gate=pregate,
            out=h,
        )
        self._consumed()

        codes, norms = quantize(
            h,
            prologue=act.PROLOGUE_NORM_MODULATE,
            norm_weight=block.norm2.weight,
            mod_scale=postscale,
            mod_shift=postshift,
            rms_eps=self.rms_eps,
        )
        group = fused.gateup
        mid = gemm.matmul_int8_scaled_with_triton(
            codes,
            self._weight(index, "gateup"),
            norms,
            group.row_norms,
            alpha=group.alpha,
            epilogue=gemm.EPILOGUE_SWIGLU,
        )
        self._consumed()
        if self.down_mode == "int8":
            codes, scales = act.quantize_rows_int8_with_triton(mid)
            h = gemm.matmul_int8_scaled_with_triton(
                codes,
                fused.down.q,
                scales,
                fused.down.scales,
                scale_mode=1,
                epilogue=gemm.EPILOGUE_RESIDUAL_GATE,
                residual=h,
                gate=postgate,
                out=h,
            )
        else:
            h = h + postgate * F.linear(mid, fused.down.weight)
        return h.reshape(batch, seq, dim)


def install(
    transformer,
    *,
    down: str = "int8",
    attention: str = "flash",
    prefetch: bool = False,
    fused_path: str | os.PathLike | None = None,
) -> FusedKrea2Blocks:
    """Move the block projections of ``transformer`` into a ``fused_blocks`` submodule.

    Call it while the weights are on the host; the original projection modules are replaced
    by placeholders. Without ``fused_path`` the grouped buffers are built from the source
    tensors; with it they are the memory-mapped tensors of a ``save_fused`` file.
    """
    stored = None
    if fused_path is not None:
        stored = _open_fused(fused_path, down, len(transformer.transformer_blocks))
    fused = FusedKrea2Blocks(
        transformer, down=down, attention=attention, prefetch=prefetch, stored=stored
    )
    for block in transformer.transformer_blocks:
        attn, ff = block.attn, block.ff
        attn.to_q = attn.to_k = attn.to_v = attn.to_gate = nn.Identity()
        attn.to_out[0] = nn.Identity()
        ff.gate = ff.up = ff.down = nn.Identity()
    transformer.fused_blocks = fused
    return fused


def save_fused(fused: FusedKrea2Blocks, path: str | os.PathLike) -> None:
    """Write the fused block weights to a safetensors file ``install`` can map."""
    tensors = {
        f"blocks.{index}.{name}": tensor.contiguous()
        for index, block in enumerate(fused.fused)
        for name, tensor in block.state_dict().items()
        if not name.endswith(".codes")
    }
    metadata = {
        "format": FUSED_FORMAT,
        "format_version": FUSED_FORMAT_VERSION,
        "down": fused.down_mode,
        "blocks": str(len(fused.fused)),
        "swiglu_interleave": str(gemm.SWIGLU_INTERLEAVE),
    }
    save_file(tensors, str(path), metadata=metadata)


def _open_fused(path, down: str, blocks: int) -> Callable[[str], torch.Tensor]:
    handle = safe_open(str(path), framework="pt", device="cpu")
    metadata = handle.metadata() or {}
    expected = {
        "format": FUSED_FORMAT,
        "format_version": FUSED_FORMAT_VERSION,
        "down": down,
        "blocks": str(blocks),
        "swiglu_interleave": str(gemm.SWIGLU_INTERLEAVE),
    }
    mismatched = {k: metadata.get(k) for k, v in expected.items() if metadata.get(k) != v}
    if mismatched:
        raise ValueError(f"{path} is not a matching fused Krea 2 file: {mismatched}")
    return handle.get_tensor


class Krea2FastRunner:
    """Text encode, fused denoise and decode for a ``diffusers.Krea2Pipeline`` (batch 1, no CFG).

    Components are moved between devices by the caller; the runner only computes on the
    device of the module it calls.
    """

    def __init__(self, pipe, fused: FusedKrea2Blocks):
        self.pipe = pipe
        self.fused = fused
        if getattr(pipe.text_encoder, "visual", None) is not None:
            pipe.text_encoder.visual = None

    @torch.no_grad()
    def encode(self, prompt: str, device: torch.device) -> torch.Tensor:
        """Selected encoder hidden states of the valid tokens ``[1, tokens, layers, dim]``."""
        pipe = self.pipe
        tokens = pipe.tokenizer(
            [PROMPT_PREFIX + prompt],
            truncation=True,
            max_length=MAX_SEQUENCE_LENGTH + PROMPT_PREFIX_TOKENS - PROMPT_SUFFIX_TOKENS,
            return_tensors="pt",
        ).input_ids
        suffix = pipe.tokenizer([PROMPT_SUFFIX], return_tensors="pt").input_ids
        input_ids = torch.cat([tokens, suffix], dim=1)
        length = input_ids.shape[1]
        # Right padding keeps kernel shapes stable across prompts; the encoder is causal, so
        # tokens after the last valid one cannot change the valid outputs.
        bucket = -(-length // TEXT_BUCKET) * TEXT_BUCKET
        pad_id = pipe.tokenizer.pad_token_id if pipe.tokenizer.pad_token_id is not None else 0
        padded = torch.full((1, bucket), pad_id, dtype=input_ids.dtype)
        padded[:, :length] = input_ids
        attention_mask = torch.zeros((1, bucket), dtype=torch.bool)
        attention_mask[:, :length] = True
        position_ids = torch.arange(bucket, device=device)[None].expand(3, 1, -1)
        outputs = pipe.text_encoder(
            input_ids=padded.to(device),
            attention_mask=attention_mask.to(device),
            position_ids=position_ids,
            output_hidden_states=True,
        )
        hidden = torch.stack(
            [outputs.hidden_states[i] for i in pipe.text_encoder_select_layers], dim=2
        )
        return hidden[:, PROMPT_PREFIX_TOKENS:length]

    @torch.no_grad()
    def denoise(
        self,
        text_states: torch.Tensor,
        *,
        width: int,
        height: int,
        steps: int = 8,
        generator: torch.Generator | None = None,
        callback=None,
    ) -> torch.Tensor:
        """Packed latents after ``steps`` Euler steps of the distilled schedule (``mu = 1.15``)."""
        from diffusers.pipelines.krea2.pipeline_krea2 import retrieve_timesteps

        pipe = self.pipe
        tr = pipe.transformer
        device = text_states.device
        text_len = text_states.shape[1]
        text = tr.txt_in(tr.text_fusion(text_states, attention_mask=None))
        channels = tr.config.in_channels // (pipe.patch_size**2)
        latents = pipe.prepare_latents(
            1, channels, height, width, text_states.dtype, device, generator
        )
        factor = pipe.vae_scale_factor * pipe.patch_size
        rope = tr.rotary_emb(
            pipe.prepare_position_ids(text_len, height // factor, width // factor, device)
        )
        sigmas = np.linspace(1.0, 1 / steps, steps)
        timesteps, _ = retrieve_timesteps(
            pipe.scheduler, steps, device, sigmas=sigmas, mu=DISTILLED_SHIFT_MU
        )
        pipe.scheduler.set_begin_index(0)
        try:
            for index, t in enumerate(timesteps):
                timestep = (
                    (t / pipe.scheduler.config.num_train_timesteps).expand(1).to(latents.dtype)
                )
                temb = tr.time_embed(timestep, dtype=latents.dtype)
                temb_mod = tr.time_mod_proj(F.gelu(temb, approximate="tanh"))
                hidden = torch.cat([text, tr.img_in(latents)], dim=1)
                for block_index, block in enumerate(tr.transformer_blocks):
                    hidden = self.fused.forward_block(block, block_index, hidden, temb_mod, rope)
                velocity = tr.final_layer(hidden[:, text_len:], temb)
                latents = pipe.scheduler.step(velocity, t, latents, return_dict=False)[0]
                if callback is not None:
                    callback(index)
        finally:
            self.fused.release()
        return latents

    @torch.no_grad()
    def decode(self, latents: torch.Tensor, *, width: int, height: int):
        pipe = self.pipe
        latents = pipe._unpack_latents(latents, height, width).to(pipe.vae.dtype)
        z_dim = pipe.vae.config.z_dim
        mean = torch.tensor(pipe.vae.config.latents_mean).view(1, z_dim, 1, 1, 1).to(latents)
        inv_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(1, z_dim, 1, 1, 1).to(
            latents
        )
        image = pipe.vae.decode(latents / inv_std + mean, return_dict=False)[0][:, :, 0]
        return pipe.image_processor.postprocess(image, output_type="pil")[0]
