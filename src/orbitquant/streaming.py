from __future__ import annotations

import ctypes
import mmap
import os
from collections.abc import Iterator
from math import gcd
from typing import Any

import torch


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


def release_cpu_tensor_pages(tensor: torch.Tensor) -> bool:
    """Drop clean CPU pages after the checkpoint tensor's final use."""

    if tensor.device.type != "cpu" or tensor.numel() == 0:
        return False
    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    start = int(tensor.data_ptr())
    aligned_start = start - (start % page_size)
    end = start + tensor.numel() * tensor.element_size()
    aligned_length = ((end - aligned_start + page_size - 1) // page_size) * page_size
    madvise = ctypes.CDLL(None, use_errno=True).madvise
    madvise.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
    madvise.restype = ctypes.c_int
    advice = int(getattr(mmap, "MADV_DONTNEED", 4))
    return madvise(ctypes.c_void_p(aligned_start), aligned_length, advice) == 0


__all__ = [
    "accelerate_hook_offloads",
    "iter_aligned_row_tiles",
    "release_cpu_tensor_pages",
]
