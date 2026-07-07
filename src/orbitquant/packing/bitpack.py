from __future__ import annotations

import torch

_CHUNK_VALUES = 4_000_000


def _pack_lowbit_cpu(values: torch.Tensor, bits: int, *, validate: bool) -> torch.Tensor:
    max_value = (1 << bits) - 1
    values_cpu_raw = values.detach().to(device="cpu").flatten()
    if validate and values_cpu_raw.numel() and (
        int(values_cpu_raw.min()) < 0 or int(values_cpu_raw.max()) > max_value
    ):
        raise ValueError(f"all values must fit in {bits} bits")
    values_cpu = values_cpu_raw.to(dtype=torch.uint8)

    total_bits = values_cpu.numel() * bits
    packed = torch.zeros((total_bits + 7) // 8, dtype=torch.int16)
    bit_ids = torch.arange(bits, dtype=torch.int64)

    for start in range(0, values_cpu.numel(), _CHUNK_VALUES):
        end = min(start + _CHUNK_VALUES, values_cpu.numel())
        chunk = values_cpu[start:end].to(dtype=torch.int16)
        bit_offsets = torch.arange(start, end, dtype=torch.int64)[:, None] * bits + bit_ids
        byte_offsets = (bit_offsets // 8).flatten()
        shifts_in_byte = (bit_offsets % 8).to(dtype=torch.int16)
        source_bits = ((chunk[:, None] >> bit_ids.to(dtype=torch.int16)) & 1) << shifts_in_byte
        packed.scatter_add_(0, byte_offsets, source_bits.flatten())

    return packed.to(dtype=torch.uint8)


def pack_lowbit(values: torch.Tensor, bits: int, *, validate: bool = True) -> torch.Tensor:
    if bits <= 0 or bits > 8:
        raise ValueError("bits must be in [1, 8]")
    if values.is_cuda:
        try:
            from orbitquant.kernels.triton_cuda import pack_lowbit_with_triton
        except Exception as exc:
            raise RuntimeError("CUDA low-bit pack requires the Triton CUDA backend") from exc
        else:
            return pack_lowbit_with_triton(values, bits=bits, validate=validate)

    return _pack_lowbit_cpu(values, bits, validate=validate)


def unpack_lowbit(packed: torch.Tensor, bits: int, length: int) -> torch.Tensor:
    if bits <= 0 or bits > 8:
        raise ValueError("bits must be in [1, 8]")
    if length < 0:
        raise ValueError("length must be non-negative")
    if packed.is_cuda:
        try:
            from orbitquant.kernels.triton_cuda import unpack_lowbit_with_triton
        except Exception as exc:
            raise RuntimeError("CUDA low-bit unpack requires the Triton CUDA backend") from exc
        else:
            return unpack_lowbit_with_triton(packed, bits=bits, length=length)

    packed_cpu = packed.detach().to(device="cpu", dtype=torch.uint8).flatten()
    values = torch.zeros(length, dtype=torch.uint8)
    bit_ids = torch.arange(bits, dtype=torch.int64)

    for start in range(0, length, _CHUNK_VALUES):
        end = min(start + _CHUNK_VALUES, length)
        bit_offsets = torch.arange(start, end, dtype=torch.int64)[:, None] * bits + bit_ids
        byte_offsets = bit_offsets // 8
        shifts_in_byte = bit_offsets % 8
        packed_bits = (packed_cpu[byte_offsets] >> shifts_in_byte.to(dtype=torch.uint8)) & 1
        value_bits = packed_bits.to(dtype=torch.int16) << bit_ids.to(dtype=torch.int16)
        values[start:end] = value_bits.sum(dim=1).to(dtype=torch.uint8)
    return values
