"""Massive-activation channels for ``OrbitQuantConfig.outlier_channels``.

OrbitQuant normalizes each token by its norm before the RPBH rotation. One input channel at
hundreds of times the token RMS takes most of that norm, so after the rotation every other
channel falls into a few central codebook levels. Keeping such channels out of the quantizer
(a BF16 side GEMM over their weight columns) restores the error of the remaining channels.
The channels are found from real forwards, so this step needs calibration inputs.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import nn

from orbitquant.config import OrbitQuantConfig


def find_outlier_channels(
    model: nn.Module,
    run: Callable[[], object],
    *,
    config: OrbitQuantConfig | None = None,
    ratio: float = 32.0,
    max_channels: int = 4,
) -> dict[str, list[int]]:
    """Input channels whose peak magnitude exceeds ``ratio`` x the median token RMS.

    ``model`` is the unquantized model; ``run()`` performs the calibration forwards. Only the
    projections that ``config`` quantizes with OrbitQuant are inspected; each keeps at most
    ``max_channels`` channels, the largest first. The result is a value for
    ``OrbitQuantConfig.outlier_channels``.
    """
    from orbitquant.modeling import classify_linear_modules_for_quantization

    if ratio <= 1.0:
        raise ValueError("ratio must be greater than 1")
    if max_channels <= 0:
        raise ValueError("max_channels must be positive")
    config = OrbitQuantConfig() if config is None else config
    decisions = classify_linear_modules_for_quantization(model, config)
    names = [name for name, decision in decisions.items() if decision.action == "orbitquant"]
    peaks: dict[str, torch.Tensor] = {}
    token_rms: dict[str, list[torch.Tensor]] = {name: [] for name in names}
    handles = []

    def hook(module, args, name):
        x = args[0].detach().reshape(-1, args[0].shape[-1]).to(torch.float32)
        peak = x.abs().amax(dim=0)
        previous = peaks.get(name)
        peaks[name] = peak if previous is None else torch.maximum(previous, peak)
        token_rms[name].append(x.norm(dim=-1).median() / math.sqrt(x.shape[-1]))

    for name in names:
        module = model.get_submodule(name)
        handles.append(
            module.register_forward_pre_hook(
                lambda module, args, name=name: hook(module, args, name)
            )
        )
    try:
        with torch.no_grad():
            run()
    finally:
        for handle in handles:
            handle.remove()

    found: dict[str, list[int]] = {}
    for name in names:
        if name not in peaks:
            continue
        typical = torch.stack(token_rms[name]).median()
        peak = peaks[name]
        candidates = torch.nonzero(peak > ratio * typical).flatten()
        if candidates.numel() == 0:
            continue
        order = torch.argsort(peak[candidates], descending=True)[:max_channels]
        found[name] = sorted(int(channel) for channel in candidates[order].tolist())
    return found


__all__ = ["find_outlier_channels"]
