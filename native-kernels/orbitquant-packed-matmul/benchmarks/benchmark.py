from __future__ import annotations

import argparse
import json
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


def _synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def _time_call(device: str, iters: int, fn) -> float:
    _synchronize(device)
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    _synchronize(device)
    return (time.perf_counter() - start) / iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cuda", "mps"], default="cuda")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--in-features", type=int, default=3072)
    parser.add_argument("--out-features", type=int, default=3072)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--with-bias", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
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
    bias = (
        torch.randn(args.out_features, device=args.device, dtype=dtype)
        if args.with_bias
        else None
    )

    indices_device = indices.long().to(args.device)

    def materialize_reference_weight() -> torch.Tensor:
        return (row_norms[:, None] * centroids[indices_device]).to(dtype)

    reference_weight = materialize_reference_weight()
    packed_weight_indices_bytes = packed.numel() * packed.element_size()
    row_norms_bytes = row_norms.numel() * row_norms.element_size()
    centroid_bytes = centroids.numel() * centroids.element_size()
    packed_weight_path_bytes = packed_weight_indices_bytes + row_norms_bytes + centroid_bytes
    materialized_weight_bytes = reference_weight.numel() * reference_weight.element_size()

    def packed_call() -> torch.Tensor:
        return matmul_packed_weight(
            x,
            packed,
            row_norms,
            centroids,
            bits=args.bits,
            out_features=args.out_features,
            in_features=args.in_features,
            bias=bias,
        )

    def predequantized_linear_call() -> torch.Tensor:
        return torch.nn.functional.linear(x, reference_weight, bias)

    def dequantize_then_linear_call() -> torch.Tensor:
        return torch.nn.functional.linear(x, materialize_reference_weight(), bias)

    for _ in range(args.warmup):
        packed_call()
        predequantized_linear_call()
        dequantize_then_linear_call()
    packed_seconds = _time_call(args.device, args.iters, packed_call)
    predequantized_linear_seconds = _time_call(
        args.device,
        args.iters,
        predequantized_linear_call,
    )
    dequantize_then_linear_seconds = _time_call(
        args.device,
        args.iters,
        dequantize_then_linear_call,
    )

    packed_output = packed_call()
    reference_output = predequantized_linear_call()
    _synchronize(args.device)
    max_abs_error = (packed_output.float() - reference_output.float()).abs().max().item()

    payload = {
        "device": args.device,
        "device_name": torch.cuda.get_device_name(0) if args.device == "cuda" else "mps",
        "dtype": str(dtype).replace("torch.", ""),
        "bits": args.bits,
        "rows": args.rows,
        "in_features": args.in_features,
        "out_features": args.out_features,
        "iters": args.iters,
        "warmup": args.warmup,
        "with_bias": args.with_bias,
        "packed_seconds_per_iter": packed_seconds,
        "predequantized_f_linear_seconds_per_iter": predequantized_linear_seconds,
        "dequantize_then_f_linear_seconds_per_iter": dequantize_then_linear_seconds,
        "packed_weight_indices_bytes": packed_weight_indices_bytes,
        "row_norms_bytes": row_norms_bytes,
        "centroid_bytes": centroid_bytes,
        "packed_weight_path_bytes": packed_weight_path_bytes,
        "materialized_weight_bytes": materialized_weight_bytes,
        "packed_weight_path_vs_materialized_weight_ratio": packed_weight_path_bytes
        / materialized_weight_bytes
        if materialized_weight_bytes > 0
        else None,
        "packed_vs_predequantized_f_linear_speedup": predequantized_linear_seconds
        / packed_seconds
        if packed_seconds > 0
        else None,
        "packed_vs_dequantize_then_f_linear_speedup": dequantize_then_linear_seconds
        / packed_seconds
        if packed_seconds > 0
        else None,
        "reference_seconds_per_iter": predequantized_linear_seconds,
        "packed_vs_reference_speedup": predequantized_linear_seconds / packed_seconds
        if packed_seconds > 0
        else None,
        "max_abs_error": max_abs_error,
        "reference": (
            "predequantized PyTorch F.linear over a materialized dequantized "
            "weight matrix"
        ),
        "dequantize_reference": (
            "materialize the dequantized weight matrix, then call PyTorch F.linear"
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
