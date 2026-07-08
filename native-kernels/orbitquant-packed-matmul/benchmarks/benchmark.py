from __future__ import annotations

import argparse
import time

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cuda", "mps"], default="cuda")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--in-features", type=int, default=3072)
    parser.add_argument("--out-features", type=int, default=3072)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    dtype = torch.float16 if args.device == "mps" else torch.bfloat16
    x = torch.randn(args.rows, args.in_features, device=args.device, dtype=dtype)
    indices = torch.randint(
        0,
        2**args.bits,
        (args.out_features, args.in_features),
        dtype=torch.uint8,
    )
    packed = _pack(indices, args.bits).to(args.device)
    row_norms = torch.ones(args.out_features, device=args.device)
    centroids = torch.linspace(-1.0, 1.0, 2**args.bits, device=args.device)

    for _ in range(3):
        matmul_packed_weight(
            x,
            packed,
            row_norms,
            centroids,
            bits=args.bits,
            out_features=args.out_features,
            in_features=args.in_features,
        )
    if args.device == "cuda":
        torch.cuda.synchronize()
    elif args.device == "mps":
        torch.mps.synchronize()

    start = time.perf_counter()
    for _ in range(args.iters):
        matmul_packed_weight(
            x,
            packed,
            row_norms,
            centroids,
            bits=args.bits,
            out_features=args.out_features,
            in_features=args.in_features,
        )
    if args.device == "cuda":
        torch.cuda.synchronize()
    elif args.device == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - start
    print({"seconds_per_iter": elapsed / args.iters, "iters": args.iters})


if __name__ == "__main__":
    main()
