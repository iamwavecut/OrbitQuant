from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from orbitquant.packing import unpack_lowbit
from orbitquant.rotations.fwht import fwht

if TYPE_CHECKING:
    from orbitquant.layers import OrbitQuantLinear


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


class ExecuTorchVulkanW4A4Linear(nn.Module):
    """Export-only exact W4A4 OrbitLinear for the ExecuTorch Vulkan delegate."""

    def __init__(self, layer: OrbitQuantLinear) -> None:
        super().__init__()
        if layer.weight_bits != 4 or layer.activation_bits != 4:
            raise ValueError("ExecuTorch Vulkan currently implements exact W4A4 only")
        if layer.packed_weight_indices is None or layer.row_norms is None:
            raise ValueError("OrbitQuantLinear is missing packed weights or row norms")
        if layer.in_features % 8 != 0:
            raise ValueError("ExecuTorch Vulkan W4A4 requires in_features divisible by 8")
        block_size = int(layer.rotation.block_size)
        if block_size < 8 or block_size > 4096:
            raise ValueError(
                "ExecuTorch Vulkan W4A4 currently requires an RPBH block size in [8, 4096]"
            )
        if block_size & (block_size - 1) or layer.in_features % block_size != 0:
            raise ValueError(
                "ExecuTorch Vulkan W4A4 requires a power-of-two RPBH block size "
                "dividing in_features"
            )

        self.in_features = layer.in_features
        self.out_features = layer.out_features
        self.block_size = block_size
        self.eps = float(layer.activation_eps)
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
        if layer.bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer(
                "bias",
                layer.bias.detach().to(device="cpu", dtype=torch.float32).contiguous(),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype not in {torch.float16, torch.float32}:
            raise TypeError(
                "ExecuTorch Vulkan accepts float16 or float32 activations; "
                f"got {x.dtype}. Convert the exported model explicitly."
            )
        if x.shape[-1] != self.in_features:
            raise ValueError(f"expected last dimension {self.in_features}, got {x.shape[-1]}")
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


def prepare_executorch_vulkan_w4a4_model(model: nn.Module) -> nn.Module:
    """Replace loaded OrbitQuant W4A4 linears with export-only Vulkan modules."""

    from orbitquant.layers import OrbitQuantLinear

    if isinstance(model, OrbitQuantLinear):
        return ExecuTorchVulkanW4A4Linear(model)

    def replace_children(module: nn.Module, prefix: str) -> None:
        for child_name, child in list(module.named_children()):
            qualified_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, OrbitQuantLinear):
                try:
                    replacement = ExecuTorchVulkanW4A4Linear(child)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"cannot prepare {qualified_name!r} for ExecuTorch Vulkan: {exc}"
                    ) from exc
                module._modules[child_name] = replacement
            else:
                replace_children(child, qualified_name)

    replace_children(model, "")
    return model


def register_executorch_vulkan_w4a4() -> object:
    """Register the custom op with the current ExecuTorch Vulkan partitioner."""

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

    edge_op = exir_ops.edge.orbitquant_vulkan.linear_w4a4_exact.default
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


__all__ = [
    "ExecuTorchVulkanW4A4Linear",
    "linear_w4a4_exact",
    "prepare_executorch_vulkan_w4a4_model",
    "register_executorch_vulkan_w4a4",
]
