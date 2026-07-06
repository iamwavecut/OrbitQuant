from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import torch

from orbitquant.rotations.fwht import fwht, is_power_of_two


def _largest_power_of_two_divisor(value: int) -> int:
    divisor = 1
    while value % (divisor * 2) == 0:
        divisor *= 2
    return divisor


@dataclass(frozen=True)
class RPBHRotation:
    """Randomized permuted block-Hadamard rotation.

    For row vectors, ``apply_to_activations(x)`` computes ``x @ R``. Applying
    the same operation to a linear weight matrix row-wise gives ``W @ R``.
    """

    dim: int
    seed: int = 0
    block_size: int | str = "paper"

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError("dim must be positive")

        block_size = self.block_size
        if block_size == "paper":
            block_size = _largest_power_of_two_divisor(self.dim)
            if block_size == 1:
                warnings.warn(
                    "RPBH paper block-size policy selected block_size=1; "
                    "rotation degenerates to signs/permutation only.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        if not isinstance(block_size, int):
            raise TypeError("block_size must be an int or 'paper'")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if not is_power_of_two(block_size):
            raise ValueError("block_size must be a power of two")
        if self.dim % block_size != 0:
            raise ValueError("block_size must divide dim")

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(self.seed))
        permutation = torch.randperm(self.dim, generator=generator, dtype=torch.long)
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(self.dim, dtype=torch.long)
        signs = torch.randint(0, 2, (self.dim,), generator=generator, dtype=torch.int8)
        signs = signs.mul(2).sub(1)

        object.__setattr__(self, "block_size", block_size)
        object.__setattr__(self, "num_blocks", self.dim // block_size)
        object.__setattr__(self, "permutation", permutation)
        object.__setattr__(self, "inverse_permutation", inverse)
        object.__setattr__(self, "signs", signs)
        object.__setattr__(self, "normalization", 1.0 / math.sqrt(block_size))

    def apply_to_activations(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.dim:
            raise ValueError(f"expected last dimension {self.dim}, got {x.shape[-1]}")

        permutation = self.permutation.to(device=x.device)
        signs = self.signs.to(device=x.device, dtype=x.dtype)
        y = x.index_select(dim=-1, index=permutation)
        y = y * signs
        y = y.reshape(*y.shape[:-1], self.num_blocks, self.block_size)
        y = fwht(y) * self.normalization
        return y.reshape(*x.shape)

    def apply_to_weight(self, weight: torch.Tensor) -> torch.Tensor:
        if weight.ndim != 2:
            raise ValueError("weight must be a rank-2 tensor")
        if weight.shape[-1] != self.dim:
            raise ValueError(f"expected in_features {self.dim}, got {weight.shape[-1]}")
        return self.apply_to_activations(weight)

    def apply_inverse_to_activations(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.dim:
            raise ValueError(f"expected last dimension {self.dim}, got {x.shape[-1]}")

        signs = self.signs.to(device=x.device, dtype=x.dtype)
        inverse_permutation = self.inverse_permutation.to(device=x.device)
        y = x.reshape(*x.shape[:-1], self.num_blocks, self.block_size)
        y = fwht(y) * self.normalization
        y = y.reshape(*x.shape)
        y = y * signs
        return y.index_select(dim=-1, index=inverse_permutation)

    def apply_inverse_to_weight(self, weight: torch.Tensor) -> torch.Tensor:
        if weight.ndim != 2:
            raise ValueError("weight must be a rank-2 tensor")
        if weight.shape[-1] != self.dim:
            raise ValueError(f"expected in_features {self.dim}, got {weight.shape[-1]}")
        return self.apply_inverse_to_activations(weight)
