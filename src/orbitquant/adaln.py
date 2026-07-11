from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from orbitquant.config import OrbitQuantConfig
from orbitquant.linear_adapters import canonical_linear_weight, linear_module_spec
from orbitquant.packing import pack_lowbit, unpack_lowbit


def _packed_length(value_count: int, bits: int) -> int:
    return (value_count * bits + 7) // 8


def _quantize_adaln_weight_reference(
    weight: torch.Tensor,
    *,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_groups = (weight.shape[1] + group_size - 1) // group_size
    padded = torch.zeros(
        weight.shape[0],
        num_groups * group_size,
        dtype=torch.float32,
        device=weight.device,
    )
    padded[:, : weight.shape[1]] = weight.to(torch.float32)
    grouped = padded.reshape(weight.shape[0], num_groups, group_size)
    scales = grouped.abs().amax(dim=-1).clamp_min(1e-12) / 7.0
    quantized_signed = torch.round(grouped / scales[..., None]).clamp(-8, 7).to(torch.int16)
    quantized_unsigned = (quantized_signed + 8).to(torch.uint8)
    packed = pack_lowbit(quantized_unsigned.flatten(), bits=4, validate=False)
    return packed, scales


def _quantize_adaln_weight(
    weight: torch.Tensor,
    *,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if weight.is_cuda:
        try:
            from orbitquant.kernels.triton_cuda import quantize_adaln_weight_with_triton
        except Exception:
            pass
        else:
            return quantize_adaln_weight_with_triton(weight, group_size=group_size)
    return _quantize_adaln_weight_reference(weight, group_size=group_size)


class RTNInt4Linear(nn.Module):
    """Symmetric INT4 round-to-nearest linear used for AdaLN modulation weights."""

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        group_size: int,
        runtime_mode: str,
        module_name: str,
        source_weight_layout: str,
        packed_weight: torch.Tensor,
        scales: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.runtime_mode = runtime_mode
        self.module_name = module_name
        self.source_weight_layout = source_weight_layout
        self.num_groups = (in_features + group_size - 1) // group_size
        self.register_buffer("packed_weight", packed_weight)
        self.register_buffer("scales", scales)
        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(bias.detach().clone(), requires_grad=False)
        self._dequantized_weight_cache: torch.Tensor | None = None
        self._dequantized_weight_cache_key: tuple[str, torch.dtype] | None = None
        self.last_effective_runtime_mode: str | None = None

    def clear_dequantized_cache(self) -> None:
        self._dequantized_weight_cache = None
        self._dequantized_weight_cache_key = None

    @classmethod
    def from_linear(
        cls, layer: nn.Module, *, config: OrbitQuantConfig, module_name: str
    ) -> RTNInt4Linear:
        spec = linear_module_spec(layer)
        if spec is None:
            raise TypeError(f"no OrbitQuant linear adapter registered for {type(layer).__name__}")
        group_size = config.adaln_group_size
        weight = canonical_linear_weight(layer).detach().to(torch.float32)
        packed, scales = _quantize_adaln_weight(weight, group_size=group_size)
        source_bias = getattr(layer, "bias", None)
        bias = None if source_bias is None else source_bias.detach()
        return cls(
            in_features=spec.in_features,
            out_features=spec.out_features,
            group_size=group_size,
            runtime_mode=config.runtime_mode,
            module_name=module_name,
            source_weight_layout=spec.weight_layout,
            packed_weight=packed,
            scales=scales.to(torch.bfloat16),
            bias=bias,
        )

    @classmethod
    def empty_from_linear(
        cls, layer: nn.Module, *, config: OrbitQuantConfig, module_name: str
    ) -> RTNInt4Linear:
        spec = linear_module_spec(layer)
        if spec is None:
            raise TypeError(f"no OrbitQuant linear adapter registered for {type(layer).__name__}")
        group_size = config.adaln_group_size
        num_groups = (spec.in_features + group_size - 1) // group_size
        packed = torch.empty(
            _packed_length(spec.out_features * num_groups * group_size, 4),
            dtype=torch.uint8,
            device=layer.weight.device,
        )
        scales = torch.empty(
            spec.out_features, num_groups, dtype=torch.bfloat16, device=layer.weight.device
        )
        source_bias = getattr(layer, "bias", None)
        bias = None
        if source_bias is not None:
            bias = torch.zeros(
                spec.out_features, dtype=source_bias.dtype, device=source_bias.device
            )
        return cls(
            in_features=spec.in_features,
            out_features=spec.out_features,
            group_size=group_size,
            runtime_mode=config.runtime_mode,
            module_name=module_name,
            source_weight_layout=spec.weight_layout,
            packed_weight=packed,
            scales=scales,
            bias=bias,
        )

    def _dequantize_weight(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        cache_key = (str(device), dtype)
        if (
            self._dequantized_weight_cache is not None
            and self._dequantized_weight_cache_key == cache_key
        ):
            return self._dequantized_weight_cache

        if device.type == "cuda":
            try:
                from orbitquant.kernels.triton_cuda import dequantize_adaln_weight_with_triton
            except Exception:
                dequantize_adaln_weight_with_triton = None
            if dequantize_adaln_weight_with_triton is not None:
                weight = dequantize_adaln_weight_with_triton(
                    self.packed_weight,
                    self.scales,
                    out_features=self.out_features,
                    in_features=self.in_features,
                    group_size=self.group_size,
                    device=device,
                )
                dequantized = weight.to(dtype=dtype)
                self._dequantized_weight_cache = dequantized.detach()
                self._dequantized_weight_cache_key = cache_key
                return dequantized

        total = self.out_features * self.num_groups * self.group_size
        unsigned = unpack_lowbit(self.packed_weight, bits=4, length=total).to(device=device)
        signed = unsigned.to(torch.int16).sub(8).to(torch.float32)
        grouped = signed.reshape(self.out_features, self.num_groups, self.group_size)
        scales = self.scales.to(device=device, dtype=torch.float32)
        weight = (grouped * scales[..., None]).reshape(
            self.out_features, self.num_groups * self.group_size
        )
        weight = weight[:, : self.in_features]
        dequantized = weight.to(device=device, dtype=dtype)
        self._dequantized_weight_cache = dequantized.detach()
        self._dequantized_weight_cache_key = cache_key
        return dequantized

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activation = x.to(torch.bfloat16)
        if x.device.type == "cpu" and self.runtime_mode in {
            "auto_fused",
            "native_packed_matmul",
        }:
            try:
                from orbitquant.kernels.native_packed_matmul import (
                    matmul_packed_adaln_int4_with_native_cpu_kernel,
                    native_cpu_adaln_available,
                )

                native_available = native_cpu_adaln_available()
            except Exception as exc:
                if self.runtime_mode == "native_packed_matmul":
                    raise RuntimeError(
                        "native_packed_matmul AdaLN on CPU requires a current "
                        "orbitquant_packed_matmul CPU package with the INT4 group "
                        "kernel; install/build it or use runtime_mode='dequant_bf16'."
                    ) from exc
                native_available = False
            if native_available:
                self.last_effective_runtime_mode = "native_packed_adaln_int4"
                return matmul_packed_adaln_int4_with_native_cpu_kernel(
                    activation,
                    self.packed_weight,
                    self.scales,
                    out_features=self.out_features,
                    in_features=self.in_features,
                    group_size=self.group_size,
                    bias=self.bias,
                )
            if self.runtime_mode == "native_packed_matmul":
                raise RuntimeError(
                    "the loaded orbitquant_packed_matmul CPU package has no packed "
                    "AdaLN INT4 kernel; build a current variant or use "
                    "runtime_mode='dequant_bf16'."
                )

        self.last_effective_runtime_mode = "dequant_bf16"
        weight = self._dequantize_weight(device=x.device, dtype=torch.bfloat16)
        bias = None if self.bias is None else self.bias.to(device=x.device, dtype=torch.bfloat16)
        return F.linear(activation, weight, bias)
