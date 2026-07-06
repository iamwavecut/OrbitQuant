from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from orbitquant.config import OrbitQuantConfig
from orbitquant.packing import pack_lowbit, unpack_lowbit


def _packed_length(value_count: int, bits: int) -> int:
    return (value_count * bits + 7) // 8


class RTNInt4Linear(nn.Module):
    """Symmetric INT4 round-to-nearest linear used for AdaLN modulation weights."""

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        group_size: int,
        module_name: str,
        packed_weight: torch.Tensor,
        scales: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.module_name = module_name
        self.num_groups = (in_features + group_size - 1) // group_size
        self.register_buffer("packed_weight", packed_weight)
        self.register_buffer("scales", scales)
        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(bias.detach().clone(), requires_grad=False)

    @classmethod
    def from_linear(
        cls, layer: nn.Linear, *, config: OrbitQuantConfig, module_name: str
    ) -> RTNInt4Linear:
        group_size = config.adaln_group_size
        weight = layer.weight.detach().to(torch.float32)
        num_groups = (layer.in_features + group_size - 1) // group_size
        padded = torch.zeros(layer.out_features, num_groups * group_size, dtype=torch.float32)
        padded[:, : layer.in_features] = weight
        grouped = padded.reshape(layer.out_features, num_groups, group_size)
        scales = grouped.abs().amax(dim=-1).clamp_min(1e-12) / 7.0
        quantized_signed = torch.round(grouped / scales[..., None]).clamp(-8, 7).to(torch.int16)
        quantized_unsigned = (quantized_signed + 8).to(torch.uint8)
        packed = pack_lowbit(quantized_unsigned.flatten(), bits=4)
        bias = None if layer.bias is None else layer.bias.detach()
        return cls(
            in_features=layer.in_features,
            out_features=layer.out_features,
            group_size=group_size,
            module_name=module_name,
            packed_weight=packed,
            scales=scales.to(torch.bfloat16),
            bias=bias,
        )

    @classmethod
    def empty_from_linear(
        cls, layer: nn.Linear, *, config: OrbitQuantConfig, module_name: str
    ) -> RTNInt4Linear:
        group_size = config.adaln_group_size
        num_groups = (layer.in_features + group_size - 1) // group_size
        packed = torch.empty(
            _packed_length(layer.out_features * num_groups * group_size, 4),
            dtype=torch.uint8,
            device=layer.weight.device,
        )
        scales = torch.empty(
            layer.out_features, num_groups, dtype=torch.bfloat16, device=layer.weight.device
        )
        bias = None
        if layer.bias is not None:
            bias = torch.zeros(layer.out_features, dtype=layer.bias.dtype, device=layer.bias.device)
        return cls(
            in_features=layer.in_features,
            out_features=layer.out_features,
            group_size=group_size,
            module_name=module_name,
            packed_weight=packed,
            scales=scales,
            bias=bias,
        )

    def _dequantize_weight(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        total = self.out_features * self.num_groups * self.group_size
        unsigned = unpack_lowbit(self.packed_weight, bits=4, length=total)
        signed = unsigned.to(torch.int16).sub(8).to(torch.float32)
        grouped = signed.reshape(self.out_features, self.num_groups, self.group_size)
        scales = self.scales.to(device="cpu", dtype=torch.float32)
        weight = (grouped * scales[..., None]).reshape(
            self.out_features, self.num_groups * self.group_size
        )
        weight = weight[:, : self.in_features]
        return weight.to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self._dequantize_weight(device=x.device, dtype=x.dtype)
        bias = None if self.bias is None else self.bias.to(device=x.device, dtype=x.dtype)
        return F.linear(x, weight, bias)
