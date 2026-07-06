from __future__ import annotations

import torch


def pack_lowbit(values: torch.Tensor, bits: int) -> torch.Tensor:
    if bits <= 0 or bits > 8:
        raise ValueError("bits must be in [1, 8]")
    values_cpu = values.detach().to(device="cpu", dtype=torch.uint8).flatten()
    max_value = (1 << bits) - 1
    if values_cpu.numel() and int(values_cpu.max()) > max_value:
        raise ValueError(f"all values must fit in {bits} bits")

    total_bits = values_cpu.numel() * bits
    packed = torch.zeros((total_bits + 7) // 8, dtype=torch.uint8)
    bit_offset = 0
    for value in values_cpu.tolist():
        remaining = bits
        source_shift = 0
        while remaining:
            byte_idx = bit_offset // 8
            bit_idx = bit_offset % 8
            take = min(remaining, 8 - bit_idx)
            mask = (1 << take) - 1
            packed[byte_idx] |= ((value >> source_shift) & mask) << bit_idx
            bit_offset += take
            source_shift += take
            remaining -= take
    return packed


def unpack_lowbit(packed: torch.Tensor, bits: int, length: int) -> torch.Tensor:
    if bits <= 0 or bits > 8:
        raise ValueError("bits must be in [1, 8]")
    if length < 0:
        raise ValueError("length must be non-negative")

    packed_cpu = packed.detach().to(device="cpu", dtype=torch.uint8).flatten()
    values = torch.zeros(length, dtype=torch.uint8)
    bit_offset = 0
    for idx in range(length):
        value = 0
        written = 0
        remaining = bits
        while remaining:
            byte_idx = bit_offset // 8
            bit_idx = bit_offset % 8
            take = min(remaining, 8 - bit_idx)
            mask = (1 << take) - 1
            value |= int((int(packed_cpu[byte_idx]) >> bit_idx) & mask) << written
            bit_offset += take
            written += take
            remaining -= take
        values[idx] = value
    return values
