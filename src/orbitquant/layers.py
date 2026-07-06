from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from orbitquant.codebooks import get_codebook
from orbitquant.config import OrbitQuantConfig
from orbitquant.functional import quantize_activations
from orbitquant.packing import pack_lowbit, unpack_lowbit
from orbitquant.rotations import RPBHRotation


class OrbitQuantLinear(nn.Module):
    """Linear layer with OrbitQuant-packed rotated weights.

    The v1 runtime dequantizes weights before BF16/FP32 matmul. This validates
    the paper's quantization path and artifact shape before fused kernels land.
    """

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        config: OrbitQuantConfig,
        module_name: str,
        bias: torch.Tensor | None,
        packed_weight_indices: torch.Tensor | None,
        row_norms: torch.Tensor | None,
        debug_weight: torch.Tensor | None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_bits = config.weight_bits
        self.activation_bits = config.activation_bits
        self.runtime_mode = config.runtime_mode
        self.module_name = module_name
        self.activation_eps = config.activation_eps
        self.rotation = RPBHRotation(
            dim=in_features, seed=config.rotation_seed, block_size=config.block_size
        )
        self.weight_codebook = get_codebook(in_features, config.weight_bits)
        self.activation_codebook = get_codebook(in_features, config.activation_bits)

        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(bias.detach().clone(), requires_grad=False)

        if packed_weight_indices is not None:
            self.register_buffer("packed_weight_indices", packed_weight_indices)
        else:
            self.packed_weight_indices = None
        if row_norms is not None:
            self.register_buffer("row_norms", row_norms)
        else:
            self.row_norms = None
        if debug_weight is not None:
            self.register_buffer("debug_weight", debug_weight)
        else:
            self.debug_weight = None

    @classmethod
    def from_linear(
        cls,
        layer: nn.Linear,
        *,
        config: OrbitQuantConfig,
        module_name: str,
    ) -> OrbitQuantLinear:
        weight = layer.weight.detach().to(torch.float32)
        bias = None if layer.bias is None else layer.bias.detach()
        rotation = RPBHRotation(
            dim=layer.in_features, seed=config.rotation_seed, block_size=config.block_size
        )
        rotated_weight = rotation.apply_to_weight(weight)

        if config.runtime_mode == "debug_no_quant":
            return cls(
                in_features=layer.in_features,
                out_features=layer.out_features,
                config=config,
                module_name=module_name,
                bias=bias,
                packed_weight_indices=None,
                row_norms=None,
                debug_weight=rotated_weight,
            )

        row_norms = rotated_weight.norm(dim=-1).clamp_min(config.activation_eps)
        unit_weight = rotated_weight / row_norms[:, None]
        codebook = get_codebook(layer.in_features, config.weight_bits)
        weight_indices = codebook.quantize_indices(unit_weight)
        packed = pack_lowbit(weight_indices, bits=config.weight_bits)

        return cls(
            in_features=layer.in_features,
            out_features=layer.out_features,
            config=config,
            module_name=module_name,
            bias=bias,
            packed_weight_indices=packed,
            row_norms=row_norms.to(torch.bfloat16),
            debug_weight=None,
        )

    def _dequantize_weight(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.debug_weight is not None:
            return self.debug_weight.to(device=device, dtype=dtype)
        if self.packed_weight_indices is None or self.row_norms is None:
            raise RuntimeError("OrbitQuantLinear is missing quantized weight buffers")

        flat = unpack_lowbit(
            self.packed_weight_indices,
            bits=self.weight_bits,
            length=self.out_features * self.in_features,
        ).to(device=device, dtype=torch.long)
        indices = flat.reshape(self.out_features, self.in_features)
        centroids = self.weight_codebook.centroids.to(device=device, dtype=torch.float32)
        row_norms = self.row_norms.to(device=device, dtype=torch.float32)
        weight = row_norms[:, None] * centroids[indices]
        return weight.to(dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.runtime_mode == "debug_no_quant":
            rotated_x = self.rotation.apply_to_activations(x.to(torch.float32)).to(x.dtype)
        elif self.runtime_mode == "debug_no_activation_quant":
            work = x.to(torch.float32)
            norms = work.norm(dim=-1, keepdim=True).clamp_min(self.activation_eps)
            rotated_x = (self.rotation.apply_to_activations(work / norms) * norms).to(x.dtype)
        else:
            rotated_x = quantize_activations(
                x,
                rotation=self.rotation,
                codebook=self.activation_codebook,
                eps=self.activation_eps,
            )

        weight = self._dequantize_weight(device=x.device, dtype=rotated_x.dtype)
        bias = None if self.bias is None else self.bias.to(device=x.device, dtype=rotated_x.dtype)
        return F.linear(rotated_x, weight, bias)
