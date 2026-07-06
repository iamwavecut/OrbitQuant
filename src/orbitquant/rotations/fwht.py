from __future__ import annotations

import torch


def is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def fwht(x: torch.Tensor) -> torch.Tensor:
    """Apply an unnormalized Walsh-Hadamard transform over the last dimension."""

    n = x.shape[-1]
    if not is_power_of_two(n):
        raise ValueError("FWHT last dimension must be a power of two")

    original_shape = x.shape
    y = x.contiguous().reshape(-1, n)
    h = 1
    while h < n:
        y = y.reshape(-1, n // (2 * h), 2, h)
        left = y[:, :, 0, :]
        right = y[:, :, 1, :]
        y = torch.cat((left + right, left - right), dim=-1).reshape(-1, n)
        h *= 2
    return y.reshape(original_shape)
