from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from orbitquant.config import OrbitQuantConfig
from orbitquant.linear_adapters import canonical_linear_weight, linear_module_spec
from orbitquant.packing import pack_lowbit, unpack_lowbit
from orbitquant.streaming import accelerate_hook_offloads, iter_aligned_row_tiles


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


def _quantize_adaln_weight_bounded(
    weight: torch.Tensor,
    *,
    group_size: int,
    row_tile_size: int,
    quantization_device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if weight.ndim != 2:
        raise ValueError("weight must be a matrix")
    out_features, in_features = weight.shape
    num_groups = (in_features + group_size - 1) // group_size
    padded_values_per_row = num_groups * group_size
    output_device = weight.device
    work_device = output_device if quantization_device is None else quantization_device
    packed = torch.empty(
        _packed_length(out_features * padded_values_per_row, 4),
        dtype=torch.uint8,
        device=output_device,
    )
    scales_output = torch.empty(
        out_features,
        num_groups,
        dtype=torch.bfloat16,
        device=output_device,
    )
    packed_offset = 0
    for row_start, row_end in iter_aligned_row_tiles(
        out_features,
        padded_values_per_row,
        4,
        row_tile_size,
    ):
        tile = weight[row_start:row_end].to(device=work_device)
        packed_tile, scales = _quantize_adaln_weight(tile, group_size=group_size)
        packed_end = packed_offset + packed_tile.numel()
        packed[packed_offset:packed_end].copy_(packed_tile.to(device=output_device))
        scales_output[row_start:row_end].copy_(
            scales.to(device=output_device, dtype=torch.bfloat16)
        )
        packed_offset = packed_end
    if packed_offset != packed.numel():
        raise RuntimeError(
            f"row-tiled packing wrote {packed_offset} bytes, expected {packed.numel()}"
        )
    return packed, scales_output


class RTNInt4Linear(nn.Module):
    """Symmetric INT4 round-to-nearest linear used for AdaLN modulation weights."""

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name == "_hf_hook" and accelerate_hook_offloads(self):
            self.clear_dequantized_cache()

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        group_size: int,
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
        self.module_name = module_name
        self.source_weight_layout = source_weight_layout
        self.num_groups = (in_features + group_size - 1) // group_size
        self.packed_weight = nn.Parameter(packed_weight, requires_grad=False)
        self.scales = nn.Parameter(scales, requires_grad=False)
        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(bias.detach().clone(), requires_grad=False)
        self._dequantized_weight_cache: torch.Tensor | None = None
        self._dequantized_weight_cache_key: tuple[str, torch.dtype] | None = None

    def clear_dequantized_cache(self) -> None:
        self._dequantized_weight_cache = None
        self._dequantized_weight_cache_key = None

    def _remember_dequantized_weight(
        self,
        weight: torch.Tensor,
        cache_key: tuple[str, torch.dtype],
    ) -> torch.Tensor:
        if not accelerate_hook_offloads(self):
            self._dequantized_weight_cache = weight.detach()
            self._dequantized_weight_cache_key = cache_key
        return weight

    @classmethod
    def from_linear(
        cls, layer: nn.Module, *, config: OrbitQuantConfig, module_name: str
    ) -> RTNInt4Linear:
        spec = linear_module_spec(layer)
        if spec is None:
            raise TypeError(f"no OrbitQuant linear adapter registered for {type(layer).__name__}")
        weight = canonical_linear_weight(layer).detach()
        source_bias = getattr(layer, "bias", None)
        bias = None if source_bias is None else source_bias.detach()
        return cls.from_weight(
            weight,
            bias=bias,
            in_features=spec.in_features,
            out_features=spec.out_features,
            source_weight_layout=spec.weight_layout,
            config=config,
            module_name=module_name,
        )

    @classmethod
    def from_weight(
        cls,
        weight: torch.Tensor,
        *,
        bias: torch.Tensor | None,
        in_features: int,
        out_features: int,
        source_weight_layout: str,
        config: OrbitQuantConfig,
        module_name: str,
        quantization_device: torch.device | None = None,
    ) -> RTNInt4Linear:
        expected_shape = (out_features, in_features)
        if tuple(weight.shape) != expected_shape:
            raise ValueError(
                f"expected canonical weight shape {expected_shape}, got {tuple(weight.shape)}"
            )
        group_size = config.adaln_group_size
        packed, scales = _quantize_adaln_weight_bounded(
            weight,
            group_size=group_size,
            row_tile_size=config.weight_row_tile_size,
            quantization_device=quantization_device,
        )
        return cls(
            in_features=in_features,
            out_features=out_features,
            group_size=group_size,
            module_name=module_name,
            source_weight_layout=source_weight_layout,
            packed_weight=packed,
            scales=scales,
            bias=bias,
        )

    @classmethod
    def empty_from_linear(
        cls,
        layer: nn.Module,
        *,
        config: OrbitQuantConfig,
        module_name: str,
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
            module_name=module_name,
            source_weight_layout=spec.weight_layout,
            packed_weight=packed,
            scales=scales,
            bias=bias,
        )

    def _dequantize_weight(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        cache_key = (str(device), dtype)
        if (
            not accelerate_hook_offloads(self)
            and self._dequantized_weight_cache is not None
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
                return self._remember_dequantized_weight(dequantized, cache_key)

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
        return self._remember_dequantized_weight(dequantized, cache_key)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activation = x.to(torch.bfloat16)
        weight = self._dequantize_weight(device=x.device, dtype=torch.bfloat16)
        bias = None if self.bias is None else self.bias.to(device=x.device, dtype=torch.bfloat16)
        return F.linear(activation, weight, bias)
