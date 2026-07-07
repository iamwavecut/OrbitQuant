from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from orbitquant.codebooks import get_codebook
from orbitquant.config import OrbitQuantConfig
from orbitquant.kernels import quantize_activations_kernel
from orbitquant.packing import pack_lowbit, unpack_lowbit
from orbitquant.rotations import RPBHRotation, get_rpbh_rotation


def _packed_length(value_count: int, bits: int) -> int:
    return (value_count * bits + 7) // 8


def _first_available_device(*tensors: torch.Tensor | None) -> torch.device | None:
    for tensor in tensors:
        if tensor is not None and tensor.device.type != "meta":
            return tensor.device
    return None


def _clone_constant(tensor: torch.Tensor, *, device: torch.device | None) -> torch.Tensor:
    constant = tensor.detach().clone()
    if device is not None:
        constant = constant.to(device=device)
    return constant


def _quantize_weight_indices(
    weight: torch.Tensor,
    row_norms: torch.Tensor,
    *,
    rotation: RPBHRotation,
    codebook,
) -> torch.Tensor:
    if weight.is_cuda:
        from orbitquant.kernels.triton_cuda import quantize_weight_indices_with_triton

        return quantize_weight_indices_with_triton(
            weight,
            row_norms,
            rotation=rotation,
            codebook=codebook,
        )

    rotated_weight = rotation.apply_to_weight(weight)
    unit_weight = rotated_weight / row_norms[:, None]
    return codebook.quantize_indices(unit_weight)


def _quantize_weight_pack(
    weight: torch.Tensor,
    row_norms: torch.Tensor,
    *,
    rotation: RPBHRotation,
    codebook,
    bits: int,
) -> torch.Tensor:
    if weight.is_cuda:
        from orbitquant.kernels.triton_cuda import quantize_weight_packed_with_triton

        return quantize_weight_packed_with_triton(
            weight,
            row_norms,
            rotation=rotation,
            codebook=codebook,
            bits=bits,
        )

    weight_indices = _quantize_weight_indices(
        weight,
        row_norms,
        rotation=rotation,
        codebook=codebook,
    )
    return pack_lowbit(weight_indices, bits=bits, validate=False)


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
        self.activation_kernel_backend = config.activation_kernel_backend
        self.module_name = module_name
        self.activation_eps = config.activation_eps
        self.rotation = get_rpbh_rotation(
            dim=in_features, seed=config.rotation_seed, block_size=config.block_size
        )
        self.weight_codebook = get_codebook(in_features, config.weight_bits)
        self.activation_codebook = get_codebook(in_features, config.activation_bits)
        constant_device = _first_available_device(
            packed_weight_indices,
            row_norms,
            debug_weight,
            bias,
        )
        self.register_buffer(
            "_rotation_permutation",
            _clone_constant(self.rotation.permutation, device=constant_device),
            persistent=False,
        )
        self.register_buffer(
            "_rotation_signs",
            _clone_constant(self.rotation.signs, device=constant_device),
            persistent=False,
        )
        self.register_buffer(
            "_activation_codebook_centroids",
            _clone_constant(self.activation_codebook.centroids, device=constant_device),
            persistent=False,
        )
        self.register_buffer(
            "_activation_codebook_boundaries",
            _clone_constant(self.activation_codebook.boundaries, device=constant_device),
            persistent=False,
        )

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
        self._dequantized_weight_cache: torch.Tensor | None = None
        self._dequantized_weight_cache_key: tuple[str, torch.dtype] | None = None

    @classmethod
    def from_linear(
        cls,
        layer: nn.Linear,
        *,
        config: OrbitQuantConfig,
        module_name: str,
    ) -> OrbitQuantLinear:
        source_weight = layer.weight.detach()
        bias = None if layer.bias is None else layer.bias.detach()
        rotation = get_rpbh_rotation(
            dim=layer.in_features, seed=config.rotation_seed, block_size=config.block_size
        )

        if config.runtime_mode == "debug_no_quant":
            weight = source_weight.to(torch.float32)
            rotated_weight = rotation.apply_to_weight(weight)
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

        if source_weight.is_cuda:
            from orbitquant.kernels.triton_cuda import row_norms_with_triton

            row_norms = row_norms_with_triton(source_weight, eps=config.activation_eps)
            quantization_weight = source_weight
        else:
            quantization_weight = source_weight.to(torch.float32)
            row_norms = quantization_weight.norm(dim=-1).clamp_min(config.activation_eps)
        codebook = get_codebook(layer.in_features, config.weight_bits)
        packed = _quantize_weight_pack(
            quantization_weight,
            row_norms,
            rotation=rotation,
            codebook=codebook,
            bits=config.weight_bits,
        )

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

    @classmethod
    def empty_from_linear(
        cls,
        layer: nn.Linear,
        *,
        config: OrbitQuantConfig,
        module_name: str,
    ) -> OrbitQuantLinear:
        bias = None
        if layer.bias is not None:
            bias = torch.zeros(layer.out_features, dtype=layer.bias.dtype, device=layer.bias.device)
        debug_weight = None
        packed_weight_indices = None
        row_norms = None
        if config.runtime_mode == "debug_no_quant":
            debug_weight = torch.empty(
                layer.out_features,
                layer.in_features,
                dtype=layer.weight.dtype,
                device=layer.weight.device,
            )
        else:
            packed_weight_indices = torch.empty(
                _packed_length(layer.out_features * layer.in_features, config.weight_bits),
                dtype=torch.uint8,
                device=layer.weight.device,
            )
            row_norms = torch.empty(
                layer.out_features, dtype=torch.bfloat16, device=layer.weight.device
            )
        return cls(
            in_features=layer.in_features,
            out_features=layer.out_features,
            config=config,
            module_name=module_name,
            bias=bias,
            packed_weight_indices=packed_weight_indices,
            row_norms=row_norms,
            debug_weight=debug_weight,
        )

    def clear_dequantized_cache(self) -> None:
        self._dequantized_weight_cache = None
        self._dequantized_weight_cache_key = None

    def _constant_buffer(
        self,
        name: str,
        *,
        device: torch.device,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        tensor = getattr(self, name)
        target_dtype = tensor.dtype if dtype is None else dtype
        if tensor.device != device or tensor.dtype != target_dtype:
            tensor = tensor.to(device=device, dtype=target_dtype)
            setattr(self, name, tensor)
        return tensor

    def _activation_kernel_constant_tensors(self, device: torch.device) -> dict[str, torch.Tensor]:
        return {
            "permutation": self._constant_buffer(
                "_rotation_permutation", device=device, dtype=torch.int64
            ),
            "signs": self._constant_buffer("_rotation_signs", device=device, dtype=torch.int8),
            "centroids": self._constant_buffer(
                "_activation_codebook_centroids", device=device, dtype=torch.float32
            ),
            "boundaries": self._constant_buffer(
                "_activation_codebook_boundaries", device=device, dtype=torch.float32
            ),
        }

    def _dequantize_weight(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        cache_key = (str(device), dtype)
        if (
            self._dequantized_weight_cache is not None
            and self._dequantized_weight_cache_key == cache_key
        ):
            return self._dequantized_weight_cache

        if self.debug_weight is not None:
            weight = self.debug_weight.to(device=device, dtype=dtype)
            self._dequantized_weight_cache = weight.detach()
            self._dequantized_weight_cache_key = cache_key
            return weight
        if self.packed_weight_indices is None or self.row_norms is None:
            raise RuntimeError("OrbitQuantLinear is missing quantized weight buffers")

        if device.type == "mps":
            try:
                from orbitquant.kernels.mps import (
                    dequantize_packed_weight_with_mps,
                    mps_metal_available,
                )
            except Exception:
                mps_metal_available = None
            if mps_metal_available is not None and mps_metal_available():
                weight = dequantize_packed_weight_with_mps(
                    self.packed_weight_indices,
                    self.row_norms,
                    self.weight_codebook,
                    bits=self.weight_bits,
                    out_features=self.out_features,
                    in_features=self.in_features,
                )
                dequantized = weight.to(dtype=dtype)
                self._dequantized_weight_cache = dequantized.detach()
                self._dequantized_weight_cache_key = cache_key
                return dequantized
        if device.type == "cuda":
            try:
                from orbitquant.kernels.triton_cuda import dequantize_packed_weight_with_triton
            except Exception:
                dequantize_packed_weight_with_triton = None
            if dequantize_packed_weight_with_triton is not None:
                weight = dequantize_packed_weight_with_triton(
                    self.packed_weight_indices,
                    self.row_norms,
                    self.weight_codebook,
                    bits=self.weight_bits,
                    out_features=self.out_features,
                    in_features=self.in_features,
                    device=device,
                )
                dequantized = weight.to(dtype=dtype)
                self._dequantized_weight_cache = dequantized.detach()
                self._dequantized_weight_cache_key = cache_key
                return dequantized

        flat = unpack_lowbit(
            self.packed_weight_indices,
            bits=self.weight_bits,
            length=self.out_features * self.in_features,
        ).to(device=device, dtype=torch.long)
        indices = flat.reshape(self.out_features, self.in_features)
        centroids = self.weight_codebook.centroids.to(device=device, dtype=torch.float32)
        row_norms = self.row_norms.to(device=device, dtype=torch.float32)
        weight = row_norms[:, None] * centroids[indices]
        dequantized = weight.to(dtype=dtype)
        self._dequantized_weight_cache = dequantized.detach()
        self._dequantized_weight_cache_key = cache_key
        return dequantized

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.runtime_mode == "debug_no_quant":
            rotated_x = self.rotation.apply_to_activations(x.to(torch.float32)).to(x.dtype)
        elif self.runtime_mode == "debug_no_activation_quant":
            work = x.to(torch.float32)
            norms = work.norm(dim=-1, keepdim=True).clamp_min(self.activation_eps)
            rotated_x = (self.rotation.apply_to_activations(work / norms) * norms).to(x.dtype)
        else:
            rotated_x = quantize_activations_kernel(
                x,
                rotation=self.rotation,
                codebook=self.activation_codebook,
                eps=self.activation_eps,
                backend=self.activation_kernel_backend,
                constant_tensors=self._activation_kernel_constant_tensors(x.device),
            )

        bias = None if self.bias is None else self.bias.to(device=x.device, dtype=rotated_x.dtype)
        if self.runtime_mode == "triton_packed_matmul":
            if self.packed_weight_indices is None or self.row_norms is None:
                raise RuntimeError("OrbitQuantLinear is missing quantized weight buffers")
            try:
                from orbitquant.kernels.triton_cuda import matmul_packed_weight_with_triton
            except Exception as exc:
                raise RuntimeError(
                    "triton_packed_matmul runtime requires the Triton CUDA backend"
                ) from exc
            return matmul_packed_weight_with_triton(
                rotated_x,
                self.packed_weight_indices,
                self.row_norms,
                self.weight_codebook,
                bits=self.weight_bits,
                out_features=self.out_features,
                in_features=self.in_features,
                bias=bias,
            )

        weight = self._dequantize_weight(device=x.device, dtype=rotated_x.dtype)
        return F.linear(rotated_x, weight, bias)
