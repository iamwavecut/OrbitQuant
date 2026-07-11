from __future__ import annotations

from collections.abc import Iterator
from math import gcd
from typing import Any


def iter_aligned_row_tiles(
    row_count: int,
    values_per_row: int,
    bits: int,
    max_rows: int,
) -> Iterator[tuple[int, int]]:
    """Yield row tiles whose non-final packed payloads end on byte boundaries."""

    if row_count < 0:
        raise ValueError("row_count must be non-negative")
    if values_per_row <= 0:
        raise ValueError("values_per_row must be positive")
    if bits <= 0 or bits > 8:
        raise ValueError("bits must be in [1, 8]")
    if max_rows <= 0:
        raise ValueError("max_rows must be positive")

    row_alignment = 8 // gcd(8, values_per_row * bits)
    tile_rows = max(row_alignment, max_rows - (max_rows % row_alignment))
    for start in range(0, row_count, tile_rows):
        yield start, min(start + tile_rows, row_count)


def accelerate_hook_offloads(module: Any) -> bool:
    hook = getattr(module, "_hf_hook", None)
    if hook is None:
        return False
    if bool(getattr(hook, "offload", False)):
        return True
    return any(
        bool(getattr(child_hook, "offload", False))
        for child_hook in getattr(hook, "hooks", ())
    )


__all__ = [
    "accelerate_hook_offloads",
    "iter_aligned_row_tiles",
]
