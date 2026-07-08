from __future__ import annotations

import torch
from orbitquant_packed_matmul import matmul_packed_weight


def _pack(values: torch.Tensor, bits: int) -> torch.Tensor:
    flat = values.detach().to(device="cpu", dtype=torch.uint8).flatten()
    packed = torch.zeros((flat.numel() * bits + 7) // 8, dtype=torch.uint8)
    for value_index, value in enumerate(flat.tolist()):
        bit_start = value_index * bits
        byte_index = bit_start // 8
        shift = bit_start % 8
        packed[byte_index] |= (value << shift) & 0xFF
        if shift + bits > 8:
            packed[byte_index + 1] |= value >> (8 - shift)
    return packed


device = "cuda" if torch.cuda.is_available() else "mps"
bits = 4
rows = 8
in_features = 16
out_features = 6
x = torch.randn(rows, in_features, device=device, dtype=torch.float16)
indices = torch.arange(out_features * in_features, dtype=torch.uint8).reshape(
    out_features, in_features
) % (2**bits)
packed = _pack(indices, bits).to(device)
row_norms = torch.linspace(0.5, 1.5, out_features, device=device)
centroids = torch.linspace(-1.0, 1.0, 2**bits, device=device)

out = matmul_packed_weight(
    x,
    packed,
    row_norms,
    centroids,
    bits=bits,
    out_features=out_features,
    in_features=in_features,
)
print(out.shape)
