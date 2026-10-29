from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from orbitquant.packing import unpack_lowbit
from orbitquant.rotations.fwht import fwht

if TYPE_CHECKING:
    from orbitquant.layers import OrbitQuantLinear

# A dense Hadamard is pre_block_size**2 FP32 values per layer.
_MAX_PRE_ROTATION_BLOCK = 1024


def _linear_w4a4_exact_reference(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    row_norms: torch.Tensor,
    permutation: torch.Tensor,
    signs: torch.Tensor,
    activation_boundaries: torch.Tensor,
    pair_lut: torch.Tensor,
    block_size: int,
    eps: float,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    in_features = x.shape[-1]
    out_features = row_norms.numel()
    work = x.to(dtype=torch.float32)
    token_norms = torch.linalg.vector_norm(work, dim=-1, keepdim=True)
    rotated = work.index_select(-1, permutation.to(dtype=torch.int64))
    rotated = rotated * signs.to(dtype=torch.float32)
    rotated = rotated / (token_norms + eps)
    rotated = rotated.reshape(*rotated.shape[:-1], in_features // block_size, block_size)
    rotated = fwht(rotated) * (block_size**-0.5)
    rotated = rotated.reshape(*work.shape)
    activation_indices = torch.bucketize(rotated, activation_boundaries)

    weight_indices = unpack_lowbit(
        packed_weight,
        bits=4,
        length=out_features * in_features,
    ).to(dtype=torch.int64)
    weight_indices = weight_indices.reshape(out_features, in_features)
    flat_activation_indices = activation_indices.reshape(-1, in_features)
    flat_token_norms = token_norms.reshape(-1)
    output = torch.empty(
        (flat_activation_indices.shape[0], out_features),
        device=x.device,
        dtype=torch.float32,
    )
    for out_feature in range(out_features):
        lut_indices = flat_activation_indices * 16 + weight_indices[out_feature]
        values = pair_lut[lut_indices].sum(dim=-1)
        values = values * flat_token_norms * row_norms[out_feature]
        if bias is not None:
            values = values + bias[out_feature]
        output[:, out_feature] = values
    output = output.reshape(*x.shape[:-1], out_features)
    return output.to(dtype=x.dtype)


@torch.library.custom_op("orbitquant_vulkan::linear_w4a4_exact", mutates_args=())
def linear_w4a4_exact(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    row_norms: torch.Tensor,
    permutation: torch.Tensor,
    signs: torch.Tensor,
    activation_boundaries: torch.Tensor,
    pair_lut: torch.Tensor,
    block_size: int,
    eps: float,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return _linear_w4a4_exact_reference(
        x,
        packed_weight,
        row_norms,
        permutation,
        signs,
        activation_boundaries,
        pair_lut,
        block_size,
        eps,
        bias,
    )


@linear_w4a4_exact.register_fake
def _linear_w4a4_exact_fake(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    row_norms: torch.Tensor,
    permutation: torch.Tensor,
    signs: torch.Tensor,
    activation_boundaries: torch.Tensor,
    pair_lut: torch.Tensor,
    block_size: int,
    eps: float,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    del (
        packed_weight,
        permutation,
        signs,
        activation_boundaries,
        pair_lut,
        block_size,
        eps,
        bias,
    )
    return x.new_empty((*x.shape[:-1], row_norms.numel()))


def _linear_w4a8_absmax_reference(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    row_norms: torch.Tensor,
    permutation: torch.Tensor,
    signs: torch.Tensor,
    weight_centroids: torch.Tensor,
    block_size: int,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    # The numerics of OrbitQuantLinear's absmax_int8 path: per-token absmax INT8 of the
    # rotated input (no token normalization), then the INT8 codes against W4 centroids.
    in_features = x.shape[-1]
    out_features = row_norms.numel()
    work = x.to(dtype=torch.float32)
    rotated = work.index_select(-1, permutation.to(dtype=torch.int64))
    rotated = rotated * signs.to(dtype=torch.float32)
    rotated = rotated.reshape(*rotated.shape[:-1], in_features // block_size, block_size)
    rotated = fwht(rotated) * (block_size**-0.5)
    rotated = rotated.reshape(-1, in_features)
    scale = rotated.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / 127.0
    codes = torch.round(rotated / scale).clamp_(-127.0, 127.0)

    weight_indices = unpack_lowbit(
        packed_weight,
        bits=4,
        length=out_features * in_features,
    ).to(dtype=torch.int64)
    weight = weight_centroids.to(dtype=torch.float32)[
        weight_indices.reshape(out_features, in_features)
    ]
    output = (codes @ weight.T) * scale * row_norms.to(dtype=torch.float32)
    if bias is not None:
        output = output + bias.to(dtype=torch.float32)
    output = output.reshape(*x.shape[:-1], out_features)
    return output.to(dtype=x.dtype)


@torch.library.custom_op("orbitquant_vulkan::linear_w4a8_absmax", mutates_args=())
def linear_w4a8_absmax(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    row_norms: torch.Tensor,
    permutation: torch.Tensor,
    signs: torch.Tensor,
    weight_centroids: torch.Tensor,
    block_size: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return _linear_w4a8_absmax_reference(
        x,
        packed_weight,
        row_norms,
        permutation,
        signs,
        weight_centroids,
        block_size,
        bias,
    )


@linear_w4a8_absmax.register_fake
def _linear_w4a8_absmax_fake(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    row_norms: torch.Tensor,
    permutation: torch.Tensor,
    signs: torch.Tensor,
    weight_centroids: torch.Tensor,
    block_size: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    del packed_weight, permutation, signs, weight_centroids, block_size, bias
    return x.new_empty((*x.shape[:-1], row_norms.numel()))


class _ExecuTorchVulkanLinear(nn.Module):
    """Packed W4 weights, the RPBH tables and the optional first RPBH round, shared by the
    export-only Vulkan linears."""

    def __init__(self, layer: OrbitQuantLinear) -> None:
        super().__init__()
        if layer.weight_bits != 4:
            raise ValueError("ExecuTorch Vulkan currently implements W4 weights only")
        if layer.outlier_channels is not None:
            raise ValueError("ExecuTorch Vulkan quantizes every input channel; no outlier_channels")
        pre_rotation = layer.pre_rotation
        if pre_rotation is not None and pre_rotation.block_size > _MAX_PRE_ROTATION_BLOCK:
            raise ValueError(
                "ExecuTorch Vulkan applies the rotation_rounds pre-rotation as a dense Hadamard "
                f"matmul, up to {_MAX_PRE_ROTATION_BLOCK}-wide blocks"
            )
        if layer.packed_weight_indices is None or layer.row_norms is None:
            raise ValueError("OrbitQuantLinear is missing packed weights or row norms")
        if layer.in_features % 8 != 0:
            raise ValueError("ExecuTorch Vulkan requires in_features divisible by 8")
        block_size = int(layer.rotation.block_size)
        if block_size < 8 or block_size > 4096:
            raise ValueError("ExecuTorch Vulkan currently requires an RPBH block size in [8, 4096]")
        if block_size & (block_size - 1) or layer.in_features % block_size != 0:
            raise ValueError(
                "ExecuTorch Vulkan requires a power-of-two RPBH block size dividing in_features"
            )

        self.in_features = layer.in_features
        self.out_features = layer.out_features
        self.block_size = block_size
        self.register_buffer(
            "packed_weight",
            layer.packed_weight_indices.detach().to(device="cpu", dtype=torch.uint8).contiguous(),
        )
        self.register_buffer(
            "row_norms",
            layer.row_norms.detach().to(device="cpu", dtype=torch.float32).contiguous(),
        )
        self.register_buffer(
            "permutation",
            layer.rotation.permutation.detach().to(device="cpu", dtype=torch.int32).contiguous(),
        )
        self.register_buffer(
            "signs",
            layer.rotation.signs.detach().to(device="cpu", dtype=torch.int32).contiguous(),
        )
        # The first round of a two-round RPBH runs as ordinary exportable ops in front of the
        # custom op: a gather, the signs and one Hadamard matmul per block, in FP32 like
        # RPBHRotation.apply_to_activations.
        self.pre_block_size = 0 if pre_rotation is None else int(pre_rotation.block_size)
        if pre_rotation is None:
            for name in ("pre_permutation", "pre_signs", "pre_hadamard"):
                self.register_buffer(name, None)
        else:
            self.register_buffer(
                "pre_permutation",
                pre_rotation.permutation.detach().to(device="cpu", dtype=torch.int64).contiguous(),
            )
            self.register_buffer(
                "pre_signs",
                pre_rotation.signs.detach().to(device="cpu", dtype=torch.float32).contiguous(),
            )
            hadamard = fwht(torch.eye(self.pre_block_size, dtype=torch.float32))
            self.register_buffer(
                "pre_hadamard", (hadamard * pre_rotation.normalization).contiguous()
            )
        if layer.bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer(
                "bias",
                layer.bias.detach().to(device="cpu", dtype=torch.float32).contiguous(),
            )

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype not in {torch.float16, torch.float32}:
            raise TypeError(
                "ExecuTorch Vulkan accepts float16 or float32 activations; "
                f"got {x.dtype}. Convert the exported model explicitly."
            )
        if x.shape[-1] != self.in_features:
            raise ValueError(f"expected last dimension {self.in_features}, got {x.shape[-1]}")
        if self.pre_hadamard is None:
            return x
        rotated = x.to(torch.float32).index_select(-1, self.pre_permutation) * self.pre_signs
        rotated = rotated.reshape(
            *x.shape[:-1], self.in_features // self.pre_block_size, self.pre_block_size
        )
        return torch.matmul(rotated, self.pre_hadamard).reshape(x.shape).to(x.dtype)


class ExecuTorchVulkanW4A4Linear(_ExecuTorchVulkanLinear):
    """Export-only exact W4A4 OrbitLinear for the ExecuTorch Vulkan delegate."""

    def __init__(self, layer: OrbitQuantLinear) -> None:
        if (
            layer.weight_bits != 4
            or layer.activation_bits != 4
            or layer.activation_quantizer != "lloyd_max"
        ):
            raise ValueError(
                "ExecuTorchVulkanW4A4Linear implements exact W4A4 with Lloyd-Max activations"
            )
        super().__init__(layer)
        self.eps = float(layer.activation_eps)
        activation_centroids = layer.activation_codebook.centroids.detach().to(
            device="cpu", dtype=torch.float32
        )
        weight_centroids = layer.weight_codebook.centroids.detach().to(
            device="cpu", dtype=torch.float32
        )
        self.register_buffer(
            "activation_boundaries",
            layer.activation_codebook.boundaries.detach()
            .to(device="cpu", dtype=torch.float32)
            .contiguous(),
        )
        self.register_buffer(
            "pair_lut",
            (activation_centroids[:, None] * weight_centroids[None, :]).flatten().contiguous(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._prepare_input(x)
        return linear_w4a4_exact(
            x,
            self.packed_weight,
            self.row_norms,
            self.permutation,
            self.signs,
            self.activation_boundaries,
            self.pair_lut,
            self.block_size,
            self.eps,
            self.bias,
        )


class ExecuTorchVulkanW4A8Linear(_ExecuTorchVulkanLinear):
    """Export-only W4A8 OrbitLinear with per-token absmax INT8 activations for the ExecuTorch
    Vulkan delegate."""

    def __init__(self, layer: OrbitQuantLinear) -> None:
        if (
            layer.weight_bits != 4
            or layer.activation_bits != 8
            or layer.activation_quantizer != "absmax_int8"
        ):
            raise ValueError(
                "ExecuTorchVulkanW4A8Linear implements W4A8 with absmax INT8 activations"
            )
        super().__init__(layer)
        self.register_buffer(
            "weight_centroids",
            layer.weight_codebook.centroids.detach()
            .to(device="cpu", dtype=torch.float32)
            .contiguous(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._prepare_input(x)
        return linear_w4a8_absmax(
            x,
            self.packed_weight,
            self.row_norms,
            self.permutation,
            self.signs,
            self.weight_centroids,
            self.block_size,
            self.bias,
        )


def executorch_vulkan_linear(layer: OrbitQuantLinear) -> _ExecuTorchVulkanLinear:
    """The export-only Vulkan module for one loaded OrbitQuantLinear: exact W4A4 with
    Lloyd-Max activations, or W4A8 with absmax INT8 activations."""

    if layer.activation_bits == 8 and layer.activation_quantizer == "absmax_int8":
        return ExecuTorchVulkanW4A8Linear(layer)
    if layer.activation_bits == 4 and layer.activation_quantizer == "lloyd_max":
        return ExecuTorchVulkanW4A4Linear(layer)
    raise ValueError(
        "ExecuTorch Vulkan implements W4A4 with Lloyd-Max activations and W4A8 with absmax "
        f"INT8 activations; got A{layer.activation_bits} {layer.activation_quantizer}"
    )


def prepare_executorch_vulkan_w4a4_model(model: nn.Module) -> nn.Module:
    """Replace loaded OrbitQuant W4A4 and W4A8 linears with export-only Vulkan modules."""

    from orbitquant.layers import OrbitQuantLinear

    if isinstance(model, OrbitQuantLinear):
        return executorch_vulkan_linear(model)

    def replace_children(module: nn.Module, prefix: str) -> None:
        for child_name, child in list(module.named_children()):
            qualified_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, OrbitQuantLinear):
                try:
                    replacement = executorch_vulkan_linear(child)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"cannot prepare {qualified_name!r} for ExecuTorch Vulkan: {exc}"
                    ) from exc
                module._modules[child_name] = replacement
            else:
                replace_children(child, qualified_name)

    replace_children(model, "")
    return model


def _register_executorch_vulkan_op(name: str) -> object:
    try:
        from executorch.backends.vulkan import utils
        from executorch.backends.vulkan.op_registry import (
            OpFeatures,
            update_features,
            vulkan_supported_ops,
        )
        from executorch.exir.dialects._ops import ops as exir_ops
    except ImportError as exc:
        raise RuntimeError(
            "ExecuTorch Vulkan export requires an ExecuTorch source build containing "
            "desktop Vulkan support (upstream after 2026-07-01)."
        ) from exc

    edge_op = getattr(exir_ops.edge.orbitquant_vulkan, name).default
    if edge_op not in vulkan_supported_ops:

        @update_features(edge_op)
        def register_features() -> OpFeatures:
            return OpFeatures(
                inputs_storage=utils.CONTIGUOUS_BUFFER,
                inputs_dtypes=utils.ALL_T,
                outputs_dtypes=utils.FP_T,
                supports_highdim=True,
                supports_prepacking=True,
            )

    return edge_op


def register_executorch_vulkan_w4a4() -> object:
    """Register the exact W4A4 custom op with the current ExecuTorch Vulkan partitioner."""

    return _register_executorch_vulkan_op("linear_w4a4_exact")


def register_executorch_vulkan_w4a8() -> object:
    """Register the absmax W4A8 custom op with the current ExecuTorch Vulkan partitioner."""

    return _register_executorch_vulkan_op("linear_w4a8_absmax")


__all__ = [
    "ExecuTorchVulkanW4A4Linear",
    "ExecuTorchVulkanW4A8Linear",
    "executorch_vulkan_linear",
    "linear_w4a4_exact",
    "linear_w4a8_absmax",
    "prepare_executorch_vulkan_w4a4_model",
    "register_executorch_vulkan_w4a4",
    "register_executorch_vulkan_w4a8",
]
