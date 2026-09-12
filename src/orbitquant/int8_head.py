"""INT8 row-quantized output heads.

OrbitQuant keeps output heads (``lm_head`` and similar) in source precision by default,
because their logits feed sampling directly. For memory-bound decode the head is often the
single largest BF16 read per token (a 150k-row vocabulary head is 0.6 GB per step). This
module offers an explicit, opt-in per-row INT8 representation:

* weights: per-row absmax INT8 (``weight_int8 [N, K]`` + ``scales [N]`` FP32),
* activations: per-token absmax INT8 at forward time,
* decode (1..8 rows): the native ``matmul_int8_rows`` DP4A kernel when the CUDA package is
  available, otherwise a cuBLASLt ``torch._int_mm`` fallback with padded rows,
* larger batches: ``torch._int_mm`` (CUDA) or a dequantized float matmul.

The INT8 GEMM epilogue is ``float(sum) * (x_scale * w_scale)`` in every path, and every path
derives ``scale = absmax / 127`` with a true IEEE division (the native kernel uses
``__fdiv_rn``; the PyTorch paths divide by a same-device tensor, because dividing by a Python
scalar makes PyTorch's CUDA kernel multiply by a reciprocal, which can be 1 ulp off). The native
kernel and the ``_int_mm`` fallback therefore agree bit for bit.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

__all__ = ["Int8RowLinear", "absmax_int8_scales", "quantize_int8_rows", "quantize_output_heads"]


def absmax_int8_scales(values: torch.Tensor) -> torch.Tensor:
    """Per-row ``absmax / 127`` in float32 with a true IEEE division (see module docstring)."""
    amax = values.to(torch.float32).abs().amax(-1).clamp_min(1e-12)
    return amax / amax.new_full((), 127.0)


def quantize_int8_rows(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row absmax INT8 quantization: returns ``(weight_int8 [N, K], scales [N] float32)``."""
    if weight.ndim != 2:
        raise ValueError("weight must be a rank-2 [out_features, in_features] tensor")
    values = weight.detach().to(torch.float32)
    scales = absmax_int8_scales(values)
    rows = (values / scales[:, None]).round().clamp(-127, 127).to(torch.int8)
    return rows.contiguous(), scales.contiguous()


def _native_int8_rows():
    try:
        from orbitquant.kernels.native_packed_matmul import load_native_packed_matmul_kernel

        kernel = load_native_packed_matmul_kernel()
    except Exception:  # pragma: no cover - depends on the provisioned binary
        return None
    if not hasattr(kernel, "matmul_int8_rows") or not hasattr(kernel, "quantize_rows_int8"):
        return None
    return kernel


class Int8RowLinear(nn.Module):
    """``nn.Linear`` replacement holding per-row INT8 weights; forward returns ``x.dtype``."""

    def __init__(
        self,
        weight_int8: torch.Tensor,
        scales: torch.Tensor,
        bias: torch.Tensor | None = None,
        *,
        module_name: str = "",
    ) -> None:
        super().__init__()
        if weight_int8.dtype != torch.int8 or weight_int8.ndim != 2:
            raise ValueError("weight_int8 must be an INT8 [out_features, in_features] tensor")
        if scales.numel() != weight_int8.shape[0]:
            raise ValueError("scales must have one entry per output row")
        self.out_features, self.in_features = (int(s) for s in weight_int8.shape)
        if self.in_features % 4:
            raise ValueError("in_features must be a multiple of 4")
        self.register_buffer("weight_int8", weight_int8.contiguous())
        self.register_buffer("scales", scales.to(torch.float32).contiguous())
        if bias is not None:
            self.register_buffer("bias", bias.detach().clone())
        else:
            self.bias = None
        self.module_name = module_name
        self.last_path: str | None = None

    @classmethod
    def from_linear(cls, linear: nn.Module, *, module_name: str = "") -> Int8RowLinear:
        weight = getattr(linear, "weight", None)
        if weight is None or weight.ndim != 2:
            raise TypeError("from_linear expects a module with a rank-2 weight (nn.Linear layout)")
        rows, scales = quantize_int8_rows(weight)
        bias = getattr(linear, "bias", None)
        return cls(rows, scales, None if bias is None else bias.detach(), module_name=module_name)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )

    def _int_mm_pad(self) -> tuple[torch.Tensor, int]:
        pad = (-self.out_features) % 8
        if not pad:
            return self.weight_int8, 0
        cached = getattr(self, "_padded_weight", None)
        if cached is None or cached.device != self.weight_int8.device:
            zeros = torch.zeros(
                (pad, self.in_features), dtype=torch.int8, device=self.weight_int8.device
            )
            cached = torch.cat([self.weight_int8, zeros], 0)
            self._padded_weight = cached
        return cached, pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lead = x.shape[:-1]
        x2 = x.reshape(-1, self.in_features)
        rows = x2.shape[0]
        out_dtype = x.dtype if x.dtype in {torch.bfloat16, torch.float16} else torch.bfloat16
        if x.device.type == "cuda":
            xf = x2.to(torch.float32)
            x_scales = absmax_int8_scales(xf)
            kernel = None
            if 1 <= rows <= 8:
                kernel = _native_int8_rows()
            if kernel is not None and x2.dtype == torch.bfloat16:
                x_int8, x_scales = kernel.quantize_rows_int8(x2.contiguous())
                bias = None if self.bias is None else self.bias.to(out_dtype)
                out = kernel.matmul_int8_rows(
                    x_int8,
                    x_scales,
                    self.weight_int8,
                    self.scales,
                    bias=bias,
                    output_dtype=out_dtype,
                )
                self.last_path = "native_int8_rows"
                return out.reshape(*lead, self.out_features)
            x_int8 = (xf / x_scales[:, None]).round().clamp(-127, 127).to(torch.int8)
            if callable(getattr(torch, "_int_mm", None)):
                weight, pad = self._int_mm_pad()
                pad_rows = max(32, rows)
                if pad_rows != rows:
                    zeros = torch.zeros(
                        (pad_rows - rows, self.in_features), dtype=torch.int8, device=x.device
                    )
                    x_int8 = torch.cat([x_int8, zeros], 0)
                acc = torch._int_mm(x_int8, weight.t())[:rows, : self.out_features]
                out = acc.to(torch.float32) * (x_scales[:, None] * self.scales[None, :])
                self.last_path = "int_mm"
            else:  # pragma: no cover - very old torch
                dequantized = self.weight_int8.float() * self.scales[:, None]
                out = (x_int8.float() * x_scales[:, None]) @ dequantized.t()
                self.last_path = "float_fallback"
        else:
            xf = x2.to(torch.float32)
            x_scales = absmax_int8_scales(xf)
            x_int8 = (xf / x_scales[:, None]).round().clamp(-127, 127)
            acc = x_int8 @ (self.weight_int8.to(torch.float32)).t()
            out = acc * (x_scales[:, None] * self.scales[None, :])
            self.last_path = "cpu_float"
        if self.bias is not None:
            out = out + self.bias.to(torch.float32)
        return out.to(out_dtype).reshape(*lead, self.out_features)


def quantize_output_heads(
    model: nn.Module, names: Iterable[str] = ("lm_head",)
) -> dict[str, Int8RowLinear]:
    """Replace the named ``nn.Linear`` heads of ``model`` by :class:`Int8RowLinear` in place."""
    replaced: dict[str, Int8RowLinear] = {}
    for name in names:
        parent_name, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        original = getattr(parent, leaf)
        head = Int8RowLinear.from_linear(original, module_name=name)
        setattr(parent, leaf, head)
        replaced[name] = head
    return replaced
