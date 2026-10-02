"""Per-device state and the primitive operations fused blocks are written with.

A ``FusedRuntime`` belongs to one transformer. It owns the INT8 workspace the packed groups
decode into right before their GEMM, one activation quantizer per input width, and the
attention dispatch. Block forwards call:

* ``input(x, width, ...)``: the prologue (norm, modulation, product) of a block input; its
  RPBH/Lloyd-Max codes are computed once per activation width and shared by every group that
  reads the same input;
* ``linear(group, source, ...)``: one fused GEMM with its epilogue;
* ``attention(q, k, v, ...)``: INT8 Q.K^T / FP16 P.V where the block allows it, else SDPA.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from orbitquant.codebooks import get_codebook
from orbitquant.fused.groups import Bf16Group, Int8Group, PackedGroup
from orbitquant.kernels import triton_activation as act
from orbitquant.kernels import triton_int8_gemm as gemm
from orbitquant.kernels.triton_attention import int8_attention_with_triton
from orbitquant.rotations import get_rpbh_rotation

ATTENTION_BACKENDS = ("int8", "flash", "cudnn", "sdpa")


class FusedInput:
    """One block input prepared for its consumers: INT8 codes per activation width for packed
    groups, the bf16 prologue output for INT8-row and BF16 groups (each computed on first use)."""

    __slots__ = ("_runtime", "_x", "_width", "_kwargs", "_codes", "_dense")

    def __init__(self, runtime, x, width, kwargs):
        self._runtime, self._x, self._width, self._kwargs = runtime, x, width, kwargs
        self._codes: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._dense = None

    def codes(self, bits: int):
        codes = self._codes.get(bits)
        if codes is None:
            quantizer = self._runtime.quantizer(self._width, bits, self._x.device)
            codes = self._codes[bits] = quantizer(self._x, **self._kwargs)
        return codes

    def dense(self):
        if self._dense is None:
            if not self._kwargs:
                self._dense = self._x.reshape(-1, self._width)
            else:
                # The prologue does not depend on the activation width.
                bits = next(iter(self._codes), 4)
                quantizer = self._runtime.quantizer(self._width, bits, self._x.device)
                self._dense, _ = quantizer(self._x, prologue_only=True, **self._kwargs)
        return self._dense


class FusedRuntime:
    def __init__(
        self,
        *,
        rotation_seed: int,
        block_size,
        codebook_version: int,
        activation_eps: float,
        attention: str = "int8",
    ):
        if attention not in ATTENTION_BACKENDS:
            raise ValueError(f"unknown attention backend {attention!r}")
        self.rotation_seed = rotation_seed
        self.block_size = block_size
        self.codebook_version = codebook_version
        self.activation_eps = activation_eps
        self.attention_backend = attention
        self._quantizers: dict[tuple, act.ActivationQuantizer] = {}
        self._workspace: torch.Tensor | None = None

    @classmethod
    def from_config(cls, config, *, attention: str = "int8") -> FusedRuntime:
        return cls(
            rotation_seed=config.rotation_seed,
            block_size=config.block_size,
            codebook_version=config.codebook_version,
            activation_eps=config.activation_eps,
            attention=attention,
        )

    def quantizer(self, width: int, bits: int, device: torch.device) -> act.ActivationQuantizer:
        key = (width, bits, str(device))
        quantizer = self._quantizers.get(key)
        if quantizer is None:
            rotation = get_rpbh_rotation(
                dim=width, seed=self.rotation_seed, block_size=self.block_size
            )
            # 8-bit activations are per-token absmax INT8 of the rotated input, no codebook.
            codebook = None if bits == 8 else get_codebook(width, bits, self.codebook_version)
            quantizer = act.ActivationQuantizer(rotation, codebook, self.activation_eps, device)
            self._quantizers[key] = quantizer
        return quantizer

    def release(self) -> None:
        """Drop device buffers before the weights leave the GPU."""
        self._quantizers.clear()
        self._workspace = None

    def _decode(self, group: PackedGroup) -> torch.Tensor:
        need = group.out_features * group.in_features
        device = group.packed.device
        if (
            self._workspace is None
            or self._workspace.numel() < need
            or (self._workspace.device != device)
        ):
            self._workspace = torch.empty(need, device=device, dtype=torch.int8)
        return group.decode(self._workspace)

    def input(self, x: torch.Tensor, width: int, **prologue) -> FusedInput:
        return FusedInput(self, x, width, prologue)

    def linear(
        self,
        group,
        source: FusedInput,
        *,
        epilogue: int = gemm.EPILOGUE_PLAIN,
        out: torch.Tensor | None = None,
        col_offset: int = 0,
        residual: torch.Tensor | None = None,
        gate: torch.Tensor | None = None,
        gate_index: torch.Tensor | None = None,
        sig_from: int = 0,
    ) -> torch.Tensor:
        common = dict(
            bias=group.bias,
            out=out,
            col_offset=col_offset,
            epilogue=epilogue,
            residual=residual,
            gate=gate,
            gate_index=gate_index,
            sig_from=sig_from,
        )
        if isinstance(group, PackedGroup):
            codes, norms = source.codes(group.activation_bits)
            return gemm.matmul_int8_scaled_with_triton(
                codes, self._decode(group), norms, group.row_scales(), alpha=group.alpha, **common
            )
        if isinstance(group, Int8Group):
            codes, scales = act.quantize_rows_int8_with_triton(source.dense())
            return gemm.matmul_int8_scaled_with_triton(
                codes, group.q, scales, group.scales, scale_mode=1, **common
            )
        if isinstance(group, Bf16Group):
            return _dense_epilogue(F.linear(source.dense(), group.weight, group.bias), **common)
        raise TypeError(f"unknown fused group {type(group).__name__}")

    def attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        int8_allowed: bool = True,
        sm_scale: float | None = None,
    ) -> torch.Tensor:
        """``[1, S, H, D]`` query, ``[1, S, Hkv, D]`` key/value -> ``[1, S, H, D]``."""
        if self.attention_backend == "int8" and int8_allowed:
            return int8_attention_with_triton(query, key, value, sm_scale=sm_scale)
        backend = {
            "cudnn": SDPBackend.CUDNN_ATTENTION,
            "sdpa": None,
        }.get(self.attention_backend, SDPBackend.FLASH_ATTENTION)
        kwargs = dict(scale=sm_scale, enable_gqa=query.shape[2] != key.shape[2])
        q, k, v = (t.transpose(1, 2) for t in (query, key, value))
        if backend is None:
            out = F.scaled_dot_product_attention(q, k, v, **kwargs)
        else:
            with sdpa_kernel(backend):
                out = F.scaled_dot_product_attention(q, k, v, **kwargs)
        return out.transpose(1, 2)


def _dense_epilogue(
    y,
    *,
    bias,
    out,
    col_offset,
    epilogue,
    residual,
    gate,
    gate_index,
    sig_from,
):
    """Eager equivalents of the GEMM epilogues for BF16 groups (bias is already applied)."""
    del bias
    if epilogue == gemm.EPILOGUE_SWIGLU:
        rows = y.shape[0]
        chunks = y.reshape(rows, -1, 2, gemm.SWIGLU_INTERLEAVE)
        g, u = chunks[:, :, 0].reshape(rows, -1), chunks[:, :, 1].reshape(rows, -1)
        y = F.silu(g) * u
    elif epilogue == gemm.EPILOGUE_GELU_TANH:
        y = F.gelu(y, approximate="tanh")
    elif epilogue == gemm.EPILOGUE_SIGMOID_TAIL:
        y = torch.cat([y[:, :sig_from], torch.sigmoid(y[:, sig_from:])], dim=1)
    elif epilogue == gemm.EPILOGUE_RESIDUAL_GATE:
        g = gate[gate_index] if gate_index is not None else gate
        y = residual[:, col_offset : col_offset + y.shape[1]] + g * y
    if out is None:
        return y
    out[:, col_offset : col_offset + y.shape[1]] = y
    return out


def default_sm_scale(head_dim: int) -> float:
    return 1.0 / math.sqrt(head_dim)
