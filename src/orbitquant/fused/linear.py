"""Single OrbitQuant projections as fused INT8 GEMMs, for models without a block family.

``fuse_linears(model)`` swaps every loaded ``OrbitQuantLinear`` for a ``FusedLinear``: the
projection's packed rows become one ``PackedGroup`` that decodes into the shared INT8 workspace
right before its GEMM, and its input is quantized per token by the fused activation kernels.
With ``activation_bits=8`` (the default) the rotated input becomes INT8 with one absmax scale
per token instead of the projection's Lloyd-Max codes; the GEMM is INT8 either way, so the
wider activations cost no GEMM time. Use it for components that run a few times per
generation, such as text encoders; block families also fuse norms, modulation and epilogues.
"""

from __future__ import annotations

import torch
from torch import nn

from orbitquant.fused.groups import PackedGroup, RowSource
from orbitquant.fused.runtime import FusedRuntime
from orbitquant.layers import OrbitQuantLinear


class FusedLinear(nn.Module):
    """One packed projection on the fused runtime; bf16 output of the input's shape."""

    def __init__(self, group: PackedGroup, runtime: FusedRuntime):
        super().__init__()
        self.group = group
        self.in_features = group.in_features
        self.out_features = group.out_features
        self.__dict__["oq_runtime"] = runtime

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rows = x.reshape(-1, self.in_features)
        if rows.dtype != torch.bfloat16 or not rows.is_contiguous():
            rows = rows.to(torch.bfloat16).contiguous()
        runtime = self.oq_runtime
        out = runtime.linear(self.group, runtime.input(rows, self.in_features))
        return out.reshape(*x.shape[:-1], self.out_features)


def fuse_linears(model: nn.Module, *, activation_bits: int = 8, config=None) -> int:
    """Replace the loaded ``OrbitQuantLinear`` modules of ``model`` in place; returns how many.

    ``config`` is the OrbitQuant config when the model does not carry one. Call
    ``orbitquant.fused.release(model)`` before moving the weights off the GPU."""
    from orbitquant.fused.install import _quant_config

    runtime = FusedRuntime.from_config(_quant_config(model, config), attention="sdpa")
    count = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, OrbitQuantLinear) or module.packed_weight_indices is None:
            continue
        holder = nn.Module()
        holder.proj = module
        group = PackedGroup.from_sources(
            holder, [RowSource("proj")], activation_bits=activation_bits
        )
        parent_path, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        fused = FusedLinear(group, runtime)
        if child.isdigit():
            parent[int(child)] = fused
        else:
            setattr(parent, child, fused)
        count += 1
    model.__dict__["oq_runtime"] = runtime
    return count
