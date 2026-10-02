"""Fused projection weights: projections that read the same activation, stored as one matrix.

* ``PackedGroup``: OrbitQuant low-bit rows (W2/W3/W4) of several projections concatenated, run
  as one INT8-surrogate GEMM. Rows destined for a SwiGLU epilogue interleave gate and up chunks.
  Projections of different weight widths (low-bit boundary and interior protection) become row
  segments that decode with their own codebooks; the per-row scale absorbs the codebook scale
  difference, so a mixed group is still one GEMM.
* ``Int8Group``: per-row INT8 weights (W8A8, ``Int8RowLinear`` numerics).
* ``Bf16Group``: source-precision weights kept as shipped.

Groups are built from the layers of a loaded model (``from_sources``) or created empty with the
same shapes (``empty``) so a checkpoint saved from them loads straight into the buffers.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from orbitquant.codebooks import get_codebook
from orbitquant.int8_head import quantize_int8_rows
from orbitquant.layers import OrbitQuantLinear

SWIGLU_INTERLEAVE = 128


@dataclass(frozen=True)
class RowSource:
    """Rows ``[start, stop)`` of the projection at ``path`` (``stop=None``: to the end)."""

    path: str
    start: int = 0
    stop: int | None = None


def _resolve(root: nn.Module, path: str) -> nn.Module:
    module = root
    for part in path.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def _rows(source: RowSource, layer: nn.Module) -> tuple[int, int]:
    out_features = layer.out_features
    stop = out_features if source.stop is None else source.stop
    if not 0 <= source.start < stop <= out_features:
        raise ValueError(f"rows [{source.start}, {stop}) outside {source.path} ({out_features})")
    return source.start, stop


def _interleave(gate: torch.Tensor, up: torch.Tensor, chunk: int) -> torch.Tensor:
    if gate.shape != up.shape or gate.shape[0] % chunk:
        raise ValueError(f"SwiGLU halves need equal row counts divisible by {chunk}")
    pieces = []
    for start in range(0, gate.shape[0], chunk):
        pieces += [gate[start : start + chunk], up[start : start + chunk]]
    return torch.cat(pieces, dim=0)


def surrogate_constants(
    in_features: int, weight_bits: int, activation_bits: int, codebook_version: int
) -> tuple[torch.Tensor, float]:
    """INT8 surrogate codes of the weight codebook and ``alpha = a_scale * w_scale`` (8-bit
    activations carry their absolute scale per token: ``a_scale = 1``)."""
    from orbitquant.kernels.triton_cuda import fit_int8_centroid_surrogate

    act_scale = 1.0
    if activation_bits != 8:
        _, act_scale = fit_int8_centroid_surrogate(
            get_codebook(in_features, activation_bits, codebook_version).centroids
        )
    codes, w_scale = fit_int8_centroid_surrogate(
        get_codebook(in_features, weight_bits, codebook_version).centroids
    )
    return codes, float(act_scale * w_scale)


def _segments(sizes_and_bits) -> tuple[tuple[int, int], ...]:
    """Consecutive ``(rows, bits)`` pieces merged into runs of one weight width."""
    runs: list[list[int]] = []
    for rows, bits in sizes_and_bits:
        if runs and runs[-1][1] == bits:
            runs[-1][0] += rows
        else:
            runs.append([rows, bits])
    return tuple((rows, bits) for rows, bits in runs)


def _register_weights(module: nn.Module, **tensors: torch.Tensor | None) -> None:
    # Frozen parameters, not buffers: diffusers' streamed group offloading moves only the
    # parameters of an offloaded module back to the host.
    for name, tensor in tensors.items():
        module.register_parameter(
            name, None if tensor is None else nn.Parameter(tensor, requires_grad=False)
        )


class PackedGroup(nn.Module):
    kind = "packed"

    def __init__(
        self,
        packed: torch.Tensor,
        row_norms: torch.Tensor,
        bias: torch.Tensor | None,
        *,
        in_features: int,
        segments,
        activation_bits: int,
        codebook_version: int,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.segments = tuple((int(rows), int(bits)) for rows, bits in segments)
        self.out_features = sum(rows for rows, _ in self.segments)
        self.activation_bits = int(activation_bits)
        self.codebook_version = int(codebook_version)
        if tuple(row_norms.shape) != (self.out_features,):
            raise ValueError(f"{tuple(row_norms.shape)} row norms for {self.out_features} rows")
        if tuple(packed.shape) != self.packed_shape(self.segments, self.in_features):
            raise ValueError(f"packed weights {tuple(packed.shape)} do not match {self.segments}")
        _register_weights(self, packed=packed, row_norms=row_norms, bias=bias)
        constants = {
            bits: surrogate_constants(
                self.in_features, bits, self.activation_bits, self.codebook_version
            )
            for bits in dict.fromkeys(bits for _, bits in self.segments)
        }
        for bits, (codes, _) in constants.items():
            self.register_buffer(f"codes_w{bits}", codes, persistent=False)
        self.alpha = constants[self.segments[0][1]][1]
        factor = None
        if len(constants) > 1:
            factor = torch.cat(
                [
                    torch.full((rows,), constants[bits][1] / self.alpha, dtype=torch.float32)
                    for rows, bits in self.segments
                ]
            )
        self.register_buffer("scale_factor", factor, persistent=False)

    @staticmethod
    def packed_shape(segments, in_features: int) -> tuple[int, ...]:
        """``[rows, row_bytes]`` for one weight width, a flat byte run for mixed widths."""
        if len(segments) == 1:
            rows, bits = segments[0]
            return (rows, in_features * bits // 8)
        return (sum(rows * in_features * bits // 8 for rows, bits in segments),)

    @property
    def weight_bits(self) -> int | None:
        return self.segments[0][1] if len(self.segments) == 1 else None

    @classmethod
    def empty(
        cls,
        segments,
        in_features: int,
        *,
        activation_bits: int,
        codebook_version: int,
        bias: bool = False,
        device: torch.device | str = "meta",
    ) -> PackedGroup:
        segments = tuple(segments)
        rows = sum(count for count, _ in segments)
        return cls(
            torch.empty(cls.packed_shape(segments, in_features), dtype=torch.uint8, device=device),
            torch.empty(rows, dtype=torch.float32, device=device),
            torch.empty(rows, dtype=torch.bfloat16, device=device) if bias else None,
            in_features=in_features,
            segments=segments,
            activation_bits=activation_bits,
            codebook_version=codebook_version,
        )

    @staticmethod
    def source_segments(layers, sources, *, swiglu: bool = False) -> tuple[tuple[int, int], ...]:
        """Row segments of a group built from ``layers`` (loaded or skeleton projections)."""
        sizes = []
        for source, layer in zip(sources, layers, strict=True):
            start, stop = _rows(source, layer)
            sizes.append((stop - start, int(layer.weight_bits)))
        if swiglu:
            if len(sizes) != 2:
                raise ValueError("a SwiGLU group takes exactly a gate and an up source")
            if sizes[0][1] != sizes[1][1]:
                raise ValueError("the gate and up halves of a SwiGLU group need one weight width")
        return _segments(sizes)

    @classmethod
    def from_sources(
        cls,
        root: nn.Module,
        sources: list[RowSource],
        *,
        swiglu: bool = False,
        activation_bits: int | None = None,
    ) -> PackedGroup:
        """Concatenate the rows of OrbitQuant projections; with ``swiglu`` the two sources are
        the gate and up halves, interleaved in ``SWIGLU_INTERLEAVE`` row chunks.
        ``activation_bits`` overrides the projections' activation width (the weights do not
        depend on it)."""
        layers = [_resolve(root, source.path) for source in sources]
        first = layers[0]

        def signature(layer):
            return (
                layer.in_features,
                layer.activation_bits,
                layer.weight_codebook.algorithm_version,
                layer.rotation,
            )

        for layer in layers:
            if not isinstance(layer, OrbitQuantLinear) or layer.packed_weight_indices is None:
                raise ValueError("packed groups need loaded OrbitQuant projections")
            if signature(layer) != signature(first):
                raise ValueError("projections of one group must share width and rotation")
        segments = cls.source_segments(layers, sources, swiglu=swiglu)
        packed, norms, biases = [], [], []
        has_bias = [layer.bias is not None for layer in layers]
        if any(has_bias) and not all(has_bias):
            raise ValueError("either every projection of a group has a bias or none has")
        for source, layer in zip(sources, layers, strict=True):
            start, stop = _rows(source, layer)
            row_bytes = layer.in_features * layer.weight_bits // 8
            rows = layer.packed_weight_indices.reshape(layer.out_features, row_bytes)
            packed.append(rows[start:stop])
            norms.append(layer.row_norms.float().reshape(-1)[start:stop])
            if layer.bias is not None:
                biases.append(layer.bias.to(torch.bfloat16).reshape(-1)[start:stop])
        if swiglu:
            packed = [_interleave(packed[0], packed[1], SWIGLU_INTERLEAVE)]
            norms = [_interleave(norms[0][:, None], norms[1][:, None], SWIGLU_INTERLEAVE)[:, 0]]
            if biases:
                biases = [
                    _interleave(biases[0][:, None], biases[1][:, None], SWIGLU_INTERLEAVE)[:, 0]
                ]
        if len(segments) > 1:
            packed = [rows.reshape(-1) for rows in packed]
        return cls(
            torch.cat(packed, dim=0).contiguous(),
            torch.cat(norms, dim=0).contiguous(),
            torch.cat(biases, dim=0).contiguous() if biases else None,
            in_features=first.in_features,
            segments=segments,
            activation_bits=activation_bits or first.activation_bits,
            codebook_version=first.weight_codebook.algorithm_version,
        )

    def row_scales(self) -> torch.Tensor:
        """Per-row GEMM scale relative to ``alpha`` (the row norms, rescaled for segments whose
        codebook surrogate has a different scale than the first one)."""
        if self.scale_factor is None:
            return self.row_norms
        return self.row_norms * self.scale_factor.to(self.row_norms.device)

    def decode(self, workspace: torch.Tensor) -> torch.Tensor:
        from orbitquant.kernels.triton_activation import decode_lowbit_to_int8_with_triton

        view = workspace[: self.out_features * self.in_features].view(
            self.out_features, self.in_features
        )
        flat = self.packed.reshape(-1)
        row = offset = 0
        for rows, bits in self.segments:
            nbytes = rows * self.in_features * bits // 8
            codes = getattr(self, f"codes_w{bits}")
            decode_lowbit_to_int8_with_triton(
                flat[offset : offset + nbytes],
                codes.to(flat.device),
                bits,
                rows,
                self.in_features,
                out=view[row : row + rows],
            )
            row += rows
            offset += nbytes
        return view


class Int8Group(nn.Module):
    kind = "int8"

    def __init__(self, q: torch.Tensor, scales: torch.Tensor, bias: torch.Tensor | None):
        super().__init__()
        self.out_features, self.in_features = (int(v) for v in q.shape)
        _register_weights(self, q=q, scales=scales, bias=bias)

    @classmethod
    def empty(cls, rows: int, in_features: int, *, bias: bool = False, device="meta") -> Int8Group:
        return cls(
            torch.empty(rows, in_features, dtype=torch.int8, device=device),
            torch.empty(rows, dtype=torch.float32, device=device),
            torch.empty(rows, dtype=torch.bfloat16, device=device) if bias else None,
        )

    @classmethod
    def from_sources(
        cls,
        root: nn.Module,
        sources: list[RowSource],
        *,
        swiglu: bool = False,
        activation_bits: int | None = None,
    ) -> Int8Group:
        weights, biases = [], []
        for source in sources:
            layer = _resolve(root, source.path)
            weight = _dense_weight(layer)
            start, stop = _rows(source, layer)
            weights.append(weight[start:stop])
            if getattr(layer, "bias", None) is not None:
                biases.append(layer.bias.to(torch.bfloat16).reshape(-1)[start:stop])
        if swiglu:
            weights = [_interleave(weights[0], weights[1], SWIGLU_INTERLEAVE)]
            if biases:
                biases = [
                    _interleave(biases[0][:, None], biases[1][:, None], SWIGLU_INTERLEAVE)[:, 0]
                ]
        q, scales = quantize_int8_rows(torch.cat(weights, dim=0))
        return cls(q, scales, torch.cat(biases).contiguous() if biases else None)


class Bf16Group(nn.Module):
    kind = "bf16"

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None):
        super().__init__()
        self.out_features, self.in_features = (int(v) for v in weight.shape)
        _register_weights(self, weight=weight, bias=bias)

    @classmethod
    def empty(cls, rows: int, in_features: int, *, bias: bool = False, device="meta") -> Bf16Group:
        return cls(
            torch.empty(rows, in_features, dtype=torch.bfloat16, device=device),
            torch.empty(rows, dtype=torch.bfloat16, device=device) if bias else None,
        )

    @classmethod
    def from_sources(
        cls,
        root: nn.Module,
        sources: list[RowSource],
        *,
        swiglu: bool = False,
        activation_bits: int | None = None,
    ) -> Bf16Group:
        weights, biases = [], []
        for source in sources:
            layer = _resolve(root, source.path)
            start, stop = _rows(source, layer)
            weights.append(_dense_weight(layer)[start:stop].to(torch.bfloat16))
            if getattr(layer, "bias", None) is not None:
                biases.append(layer.bias.to(torch.bfloat16).reshape(-1)[start:stop])
        if swiglu:
            weights = [_interleave(weights[0], weights[1], SWIGLU_INTERLEAVE)]
            if biases:
                biases = [
                    _interleave(biases[0][:, None], biases[1][:, None], SWIGLU_INTERLEAVE)[:, 0]
                ]
        return cls(
            torch.cat(weights, dim=0).contiguous(),
            torch.cat(biases).contiguous() if biases else None,
        )


def _dense_weight(layer: nn.Module) -> torch.Tensor:
    """Source-precision weight of a projection the artifact kept unquantized."""
    if isinstance(layer, OrbitQuantLinear):
        raise ValueError("an OrbitQuant projection belongs in a packed group")
    weight = getattr(layer, "weight", None)
    if weight is None:
        raise ValueError(f"{type(layer).__name__} has no dense weight")
    return weight.detach()


GROUP_TYPES = {cls.kind: cls for cls in (PackedGroup, Int8Group, Bf16Group)}
