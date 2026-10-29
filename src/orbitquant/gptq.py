"""Optional calibration: GPTQ rounding of OrbitQuant weight codes.

Plain OrbitQuant rounds every rotated, row-normalized weight to its nearest Lloyd-Max centroid
and needs no data. GPTQ keeps the codebook, rotation, bit width, row-norm scale and packed layout,
and chooses the codes column by column against the Hessian ``X^T X`` of the projection's rotated
calibration inputs: the rounding error of each column is compensated in the columns not yet
rounded. The packed result loads and runs through every existing runtime path unchanged.

Flow, with the BF16 model on the GPU::

    groups = [["blocks.0.attn.q", "blocks.0.attn.k", "blocks.0.attn.v"], ["blocks.0.attn.o"]]
    with HessianCollector(model, groups, config) as hessians:
        for batch in calibration:
            model(**batch)
    codes = round_groups(model, groups, hessians, config)
    quantize_model(model, config)
    load_rounded_codes(model, codes)

Projections in one group read the same input and share one Hessian. For a model too large for all
Hessians at once, collect and round a few blocks at a time; ``sequential=True`` writes the rounded
weights back into the BF16 model so that later blocks are calibrated on rounded inputs.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch
from torch import nn

from orbitquant.codebooks.lloyd_max import get_codebook
from orbitquant.config import OrbitQuantConfig
from orbitquant.packing import pack_lowbit
from orbitquant.rotations.rpbh import get_pre_rotation, get_rpbh_rotation

__all__ = [
    "HessianCollector",
    "RoundedCodes",
    "gptq_round",
    "load_rounded_codes",
    "round_groups",
    "update_state_dict",
]


@dataclass
class RoundedCodes:
    """Packed codes and BF16 row norms of one projection, with the Hessian-weighted relative
    output error of nearest rounding (``rtn_error``) and of GPTQ rounding (``gptq_error``) and
    the relative damping its Hessian needed."""

    packed: torch.Tensor
    row_norms: torch.Tensor
    bits: int
    rtn_error: float
    gptq_error: float
    damping: float = 0.0


@contextlib.contextmanager
def _full_fp32():
    """Hessians and GPTQ updates need full float32 matmuls: with TF32 a few large-norm tokens
    leave rounding noise above the damping in the weak directions, and Cholesky fails."""
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def _inverse_factor(H: torch.Tensor, damp: float, attempts: int = 6):
    """Upper Cholesky factor of the damped inverse Hessian per group, and the damping it took:
    a group that is not positive definite at ``damp`` retries with ten times the damping."""
    eye = torch.eye(H.shape[-1], device=H.device)
    mean = torch.diagonal(H, dim1=1, dim2=2).mean(dim=1)
    level = torch.full_like(mean, damp)
    for _ in range(attempts):
        L, info = torch.linalg.cholesky_ex(H + (level * mean)[:, None, None] * eye)
        if not info.any():
            U, info = torch.linalg.cholesky_ex(torch.cholesky_inverse(L), upper=True)
            if not info.any():
                return U, level
        level = torch.where(info != 0, level * 10, level)
    raise torch.linalg.LinAlgError(f"Hessian not positive definite at damping {level.tolist()}")


def _weighted_error(weight: torch.Tensor, approx: torch.Tensor, hessian: torch.Tensor) -> float:
    diff = weight - approx
    with _full_fp32():
        return (torch.sum((diff @ hessian) * diff) / torch.sum((weight @ hessian) * weight)).item()


def gptq_round(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    *,
    bits: int,
    codebook_version: int = 3,
    damp: float = 0.01,
    block: int = 128,
    act_order: bool = True,
    eps: float = 1e-10,
    return_damping: bool = False,
):
    """Codes ``[rows, cols]`` (uint8) and row scales ``[rows]`` (BF16-representable float32) of a
    rotated weight against the Hessian of its rotated input; with ``return_damping`` also the
    relative damping each Hessian needed (``damp`` or a tenfold multiple of it).

    ``weight`` is the rotated weight in float32. The row scale is the BF16-rounded row norm, as in
    nearest rounding, so only the codes differ. A leading batch dimension (``[G, rows, cols]``
    weights with ``[G, cols, cols]`` Hessians) rounds G same-shape projections in one pass, which
    keeps the GPU busy: the column loop launches the same number of kernels for any G."""
    with _full_fp32():
        return _gptq_round(
            weight, hessian, bits, codebook_version, damp, block, act_order, eps, return_damping
        )


def _gptq_round(
    weight, hessian, bits, codebook_version, damp, block, act_order, eps, return_damping
):
    batched = weight.ndim == 3
    W = (weight if batched else weight[None]).clone()
    H = (hessian if batched else hessian[None]).to(torch.float32).clone()
    groups, rows, cols = W.shape
    if H.shape != (groups, cols, cols):
        raise ValueError("weight must be [(G,) rows, cols] and hessian [(G,) cols, cols]")
    codebook = get_codebook(cols, bits, codebook_version)
    centroids = codebook.centroids.to(W.device)
    boundaries = codebook.boundaries.to(W.device)
    eye = torch.eye(cols, device=H.device)
    diag = torch.diagonal(H, dim1=1, dim2=2)
    dead = diag == 0
    H = H + eye * dead[:, None, :]
    W = W.masked_fill(dead[:, None, :], 0)
    scale = W.new_empty(groups, rows)
    scale[:] = (
        (weight if batched else weight[None]).norm(dim=2).to(torch.bfloat16).to(torch.float32)
    )
    scale = scale.clamp_min(eps)
    if act_order:
        perm = torch.argsort(torch.diagonal(H, dim1=1, dim2=2), dim=1, descending=True)
    else:
        perm = torch.arange(cols, device=W.device).expand(groups, cols)
    W = torch.gather(W, 2, perm[:, None, :].expand(groups, rows, cols))
    H = torch.gather(
        torch.gather(H, 1, perm[:, :, None].expand(groups, cols, cols)),
        2,
        perm[:, None, :].expand(groups, cols, cols),
    )
    Hinv, used_damp = _inverse_factor(H, damp)
    codes = torch.zeros(groups, rows, cols, dtype=torch.uint8, device=W.device)
    for start in range(0, cols, block):
        stop = min(start + block, cols)
        W1 = W[:, :, start:stop].clone()
        E1 = torch.zeros_like(W1)
        Hinv1 = Hinv[:, start:stop, start:stop]
        for i in range(stop - start):
            column = W1[:, :, i]
            index = torch.bucketize(column / scale, boundaries)
            codes[:, :, start + i] = index.to(torch.uint8)
            err = (column - centroids[index] * scale) / Hinv1[:, i, i][:, None]
            W1[:, :, i:] -= err[:, :, None] * Hinv1[:, i, i:][:, None, :]
            E1[:, :, i] = err
        W[:, :, stop:] -= E1 @ Hinv[:, start:stop, stop:]
    out = torch.empty_like(codes)
    out.scatter_(2, perm[:, None, :].expand(groups, rows, cols), codes)
    result = (out, scale, used_damp) if batched else (out[0], scale[0], used_damp[0])
    return result if return_damping else result[:2]


class HessianCollector:
    """Accumulates ``X^T X`` of the rotated input of the first projection of every group while the
    context is open. Index it by any projection name of a group."""

    def __init__(self, model: nn.Module, groups: list[list[str]], config: OrbitQuantConfig):
        self.model = model
        self.groups = [list(group) for group in groups]
        self.config = config
        self._hessians: dict[str, torch.Tensor] = {}
        self._owner: dict[str, str] = {}
        self._handles = []
        self.tokens: dict[str, int] = {}

    def __enter__(self) -> HessianCollector:
        for group in self.groups:
            head = group[0]
            module = self.model.get_submodule(head)
            dim = module.in_features
            rotation = get_rpbh_rotation(
                dim=dim, seed=self.config.rotation_seed, block_size=self.config.block_size
            )
            pre_rotation = get_pre_rotation(
                dim,
                seed=self.config.rotation_seed,
                block_size=self.config.block_size,
                rotation_rounds=self.config.rotation_rounds,
            )
            outliers = _group_outlier_channels(self.config, group)
            device = module.weight.device
            self._hessians[head] = torch.zeros(dim, dim, dtype=torch.float32, device=device)
            self.tokens[head] = 0
            for name in group:
                self._owner[name] = head

            def hook(
                _module,
                args,
                head=head,
                dim=dim,
                rotation=rotation,
                pre_rotation=pre_rotation,
                outliers=outliers,
            ):
                x = args[0].reshape(-1, dim).to(torch.float32)
                if outliers is not None:
                    # The quantized path sees the input with its outlier channels zeroed.
                    x = x.index_fill(1, torch.tensor(outliers, device=x.device), 0)
                if pre_rotation is not None:
                    x = pre_rotation.apply_to_activations(x)
                x = rotation.apply_to_activations(x)
                with _full_fp32():
                    self._hessians[head].addmm_(x.T, x)
                self.tokens[head] += x.shape[0]

            self._handles.append(module.register_forward_pre_hook(hook))
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __getitem__(self, name: str) -> torch.Tensor:
        return self._hessians[self._owner[name]]

    def release(self, name: str) -> None:
        """Free the Hessian of ``name``'s group."""
        self._hessians.pop(self._owner[name], None)


def _group_outlier_channels(config: OrbitQuantConfig, group: list[str]) -> list[int] | None:
    """Outlier channels of a group sharing one input; every member must keep the same ones."""
    channel_sets = {tuple(config.outlier_channels.get(name, ())) for name in group}
    if len(channel_sets) != 1:
        raise ValueError(f"group {group} mixes outlier_channels {sorted(channel_sets)}")
    channels = channel_sets.pop()
    return list(channels) or None


def _quantized_columns(weight: torch.Tensor, outliers: list[int] | None) -> torch.Tensor:
    if outliers is None:
        return weight
    return weight.index_fill(1, torch.tensor(outliers, device=weight.device), 0)


def _to_quantization_basis(rotation, pre_rotation, weight: torch.Tensor) -> torch.Tensor:
    if pre_rotation is not None:
        weight = pre_rotation.apply_to_weight(weight)
    return rotation.apply_to_weight(weight)


def _from_quantization_basis(rotation, pre_rotation, weight: torch.Tensor) -> torch.Tensor:
    weight = rotation.apply_inverse_to_weight(weight)
    if pre_rotation is not None:
        weight = pre_rotation.apply_inverse_to_weight(weight)
    return weight


def round_groups(
    model: nn.Module,
    groups: list[list[str]],
    hessians: HessianCollector,
    config: OrbitQuantConfig,
    *,
    module_bits: dict[str, int] | None = None,
    damp: float = 0.01,
    block: int = 128,
    act_order: bool = True,
    sequential: bool = False,
    release: bool = True,
    batch: int = 8,
    memory_budget: int = 6 << 30,
) -> dict[str, RoundedCodes]:
    """GPTQ codes of every projection in ``groups`` from the BF16 weights of ``model``.

    ``module_bits`` overrides ``config.weight_bits`` per projection name (one width per group).
    Up to ``batch`` groups of the same shape and width round together in one pass, fewer when
    their float32 work tensors would exceed ``memory_budget`` bytes.
    With ``sequential`` the BF16 weights are replaced by their rounded values."""
    module_bits = module_bits or {}
    prepared = []
    for group in groups:
        modules = [model.get_submodule(name) for name in group]
        bits = {module_bits.get(name, config.weight_bits) for name in group}
        if len(bits) != 1:
            raise ValueError(f"group {group} mixes weight widths {sorted(bits)}")
        sizes = [m.out_features for m in modules]
        prepared.append((group, modules, sizes, bits.pop(), sum(sizes), modules[0].in_features))
    buckets: dict[tuple[int, int, int], list] = {}
    for item in prepared:
        buckets.setdefault((item[3], item[4], item[5]), []).append(item)
    result: dict[str, RoundedCodes] = {}
    for (bits, rows, dim), items in buckets.items():
        rotation = get_rpbh_rotation(
            dim=dim, seed=config.rotation_seed, block_size=config.block_size
        )
        pre_rotation = get_pre_rotation(
            dim,
            seed=config.rotation_seed,
            block_size=config.block_size,
            rotation_rounds=config.rotation_rounds,
        )

        codebook = get_codebook(dim, bits, config.codebook_version)
        # Per group: about five [cols, cols] (Hessian copies, Cholesky factors) and eight
        # [rows, cols] float32 work tensors; the batch also stays within half the free GPU memory.
        per_group = 4 * dim * (5 * dim + 8 * rows)
        budget = memory_budget
        if torch.cuda.is_available() and items[0][1][0].weight.is_cuda:
            budget = min(budget, torch.cuda.mem_get_info(items[0][1][0].weight.device)[0] // 2)
        size = max(1, min(batch, budget // per_group))
        for first in range(0, len(items), size):
            chunk = items[first : first + size]
            H = torch.stack([hessians[item[0][0]] for item in chunk])
            weight = torch.stack(
                [
                    _to_quantization_basis(
                        rotation,
                        pre_rotation,
                        torch.cat(
                            [
                                _quantized_columns(
                                    m.weight.detach().to(device=H.device, dtype=torch.float32),
                                    _group_outlier_channels(config, item[0]),
                                )
                                for m in item[1]
                            ]
                        )
                    )
                    for item in chunk
                ]
            )
            codes, scale, used = gptq_round(
                weight,
                H,
                bits=bits,
                codebook_version=config.codebook_version,
                damp=damp,
                block=block,
                act_order=act_order,
                eps=config.activation_eps,
                return_damping=True,
            )
            centroids = codebook.centroids.to(weight.device)
            boundaries = codebook.boundaries.to(weight.device)
            for g, (group, modules, sizes, _, _, _) in enumerate(chunk):
                rounded = centroids[codes[g].long()] * scale[g, :, None]
                nearest = centroids[torch.bucketize(weight[g] / scale[g, :, None], boundaries)]
                rtn_error = _weighted_error(weight[g], nearest * scale[g, :, None], H[g])
                gptq_error = _weighted_error(weight[g], rounded, H[g])
                del nearest
                for name, part, part_scale in zip(
                    group, codes[g].split(sizes), scale[g].split(sizes), strict=True
                ):
                    result[name] = RoundedCodes(
                        packed=pack_lowbit(part.contiguous(), bits=bits),
                        row_norms=part_scale.to(torch.bfloat16),
                        bits=bits,
                        rtn_error=rtn_error,
                        gptq_error=gptq_error,
                        damping=float(used[g]),
                    )
                if sequential:
                    original = _from_quantization_basis(rotation, pre_rotation, rounded)
                    outliers = _group_outlier_channels(config, group)
                    for module, part in zip(modules, original.split(sizes), strict=True):
                        if outliers is not None:
                            # Outlier columns stay at their source values (BF16 side GEMM).
                            index = torch.tensor(outliers, device=part.device)
                            source = module.weight.detach().to(part.device, part.dtype)
                            part = part.index_copy(1, index, source.index_select(1, index))
                        module.weight.data = part.to(
                            device=module.weight.device, dtype=module.weight.dtype
                        )
                if release:
                    hessians.release(group[0])
                del rounded
            del H, weight, codes, scale
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return result


def load_rounded_codes(model: nn.Module, codes: dict[str, RoundedCodes]) -> None:
    """Copy rounded codes into the ``OrbitQuantLinear`` modules of a quantized ``model``."""
    from orbitquant.layers import OrbitQuantLinear

    for name, rounded in codes.items():
        module = model.get_submodule(name)
        if not isinstance(module, OrbitQuantLinear) or module.packed_weight_indices is None:
            raise TypeError(f"{name} is not a packed OrbitQuantLinear")
        if module.weight_bits != rounded.bits:
            raise ValueError(f"{name}: {module.weight_bits}-bit module, {rounded.bits}-bit codes")
        target = module.packed_weight_indices
        if target.numel() != rounded.packed.numel():
            raise ValueError(f"{name}: packed size {rounded.packed.numel()} != {target.numel()}")
        target.data.copy_(rounded.packed.reshape(target.shape).to(target.device))
        row_norms = rounded.row_norms.to(module.row_norms.device, torch.float32)
        module.row_norms.data.copy_((row_norms * module.output_gain).to(module.row_norms.dtype))
        module.clear_dequantized_cache()


def update_state_dict(state: dict[str, torch.Tensor], codes: dict[str, RoundedCodes]) -> None:
    """Write rounded codes into a quantized state dict (``<name>.packed_weight_indices`` and
    ``<name>.row_norms``), keeping each tensor's shape and dtype."""
    for name, rounded in codes.items():
        packed_key, norms_key = f"{name}.packed_weight_indices", f"{name}.row_norms"
        packed = state[packed_key]
        if packed.numel() != rounded.packed.numel():
            raise ValueError(f"{name}: packed size {rounded.packed.numel()} != {packed.numel()}")
        state[packed_key] = rounded.packed.reshape(packed.shape).to("cpu")
        state[norms_key] = rounded.row_norms.to("cpu", state[norms_key].dtype)
