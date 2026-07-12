from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
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


def _time_call(device: str, fn) -> float:
    _synchronize(device)
    start = time.perf_counter_ns()
    fn()
    _synchronize(device)
    return (time.perf_counter_ns() - start) / 1_000_000_000


def _time_distribution(device: str, iters: int, fn) -> dict[str, float]:
    samples = []
    for _ in range(iters):
        samples.append(_time_call(device, fn))
    samples.sort()
    return {
        "mean": statistics.fmean(samples),
        "median": statistics.median(samples),
        "p95": samples[min(len(samples) - 1, int(len(samples) * 0.95))],
    }


def _parse_rows(raw: str) -> list[int]:
    rows_values = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        rows = int(chunk)
        if rows <= 0:
            raise argparse.ArgumentTypeError("--rows values must be positive")
        rows_values.append(rows)
    if not rows_values:
        raise argparse.ArgumentTypeError("--rows must list at least one row count")
    return rows_values


_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _benchmark_rows(args, rows: int, dtype: torch.dtype, weights) -> dict:
    packed, row_norms, centroids, indices_device = weights
    x = torch.randn(rows, args.in_features, device=args.device, dtype=dtype)
    bias = (
        torch.randn(args.out_features, device=args.device, dtype=dtype)
        if args.with_bias
        else None
    )

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

    packed_first_call_seconds = _time_call(args.device, packed_call)
    predequantized_first_call_seconds = _time_call(args.device, predequantized_linear_call)
    dequantize_then_first_call_seconds = _time_call(args.device, dequantize_then_linear_call)

    for _ in range(args.warmup):
        packed_call()
        predequantized_linear_call()
        dequantize_then_linear_call()
    packed_distribution = _time_distribution(args.device, args.iters, packed_call)
    predequantized_distribution = _time_distribution(
        args.device,
        args.iters,
        predequantized_linear_call,
    )
    dequantize_then_distribution = _time_distribution(
        args.device,
        args.iters,
        dequantize_then_linear_call,
    )
    # Headline numbers are hot-loop medians; the mean is retained alongside the
    # median/p95 so noisy first-iteration outliers cannot skew comparisons.
    packed_seconds = packed_distribution["median"]
    predequantized_linear_seconds = predequantized_distribution["median"]
    dequantize_then_linear_seconds = dequantize_then_distribution["median"]

    packed_output = packed_call()
    reference_output = predequantized_linear_call()
    _synchronize(args.device)
    error = packed_output.float() - reference_output.float()
    max_abs_error = error.abs().max().item()
    rmse = error.square().mean().sqrt().item()
    reference_rms = reference_output.float().square().mean().sqrt().item()
    relative_rmse = rmse / max(reference_rms, 1e-12)

    return {
        "device": args.device,
        "device_name": (
            torch.cuda.get_device_name(0)
            if args.device == "cuda"
            else "mps"
            if args.device == "mps"
            else f"{platform.processor() or platform.machine()} "
            f"({torch.backends.cpu.get_cpu_capability()})"
        ),
        "dtype": str(dtype).replace("torch.", ""),
        "bits": args.bits,
        "rows": rows,
        "in_features": args.in_features,
        "out_features": args.out_features,
        "iters": args.iters,
        "warmup": args.warmup,
        "threads": (
            os.environ.get("ORBITQUANT_CPU_THREADS", "runtime default")
            if args.device == "cpu"
            else None
        ),
        "torch_threads": torch.get_num_threads() if args.device == "cpu" else None,
        "with_bias": args.with_bias,
        "packed_seconds_per_iter": packed_seconds,
        "packed_first_call_seconds": packed_first_call_seconds,
        "packed_hot_mean_seconds": packed_distribution["mean"],
        "packed_hot_median_seconds": packed_distribution["median"],
        "packed_hot_p95_seconds": packed_distribution["p95"],
        "predequantized_f_linear_seconds_per_iter": predequantized_linear_seconds,
        "predequantized_first_call_seconds": predequantized_first_call_seconds,
        "predequantized_hot_mean_seconds": predequantized_distribution["mean"],
        "predequantized_hot_median_seconds": predequantized_distribution["median"],
        "predequantized_hot_p95_seconds": predequantized_distribution["p95"],
        "dequantize_then_f_linear_seconds_per_iter": dequantize_then_linear_seconds,
        "dequantize_then_first_call_seconds": dequantize_then_first_call_seconds,
        "dequantize_then_hot_mean_seconds": dequantize_then_distribution["mean"],
        "dequantize_then_hot_median_seconds": dequantize_then_distribution["median"],
        "dequantize_then_hot_p95_seconds": dequantize_then_distribution["p95"],
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
        "rmse": rmse,
        "relative_rmse": relative_rmse,
        "timing_headline": "hot-loop median seconds per iteration",
        "reference": (
            "predequantized PyTorch F.linear over a materialized dequantized "
            "weight matrix"
        ),
        "dequantize_reference": (
            "materialize the dequantized weight matrix, then call PyTorch F.linear"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cuda")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument(
        "--rows",
        type=_parse_rows,
        default=[1, 8, 512, 4096],
        help="comma-separated row counts to sweep (default covers decode-bound "
        "small batches and GEMM-bound large batches)",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", *sorted(_DTYPES)],
        default="auto",
        help="activation dtype; auto picks float16 on mps and bfloat16 elsewhere",
    )
    parser.add_argument("--in-features", type=int, default=3072)
    parser.add_argument("--out-features", type=int, default=3072)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--with-bias", action="store_true")
    args = parser.parse_args()

    if args.threads < 0:
        parser.error("--threads must be non-negative")
    if args.iters <= 0 or args.warmup < 0:
        parser.error("--iters must be positive and --warmup must be non-negative")
    if args.device == "cpu" and args.threads > 0:
        os.environ["ORBITQUANT_CPU_THREADS"] = str(args.threads)
        torch.set_num_threads(args.threads)

    torch.manual_seed(args.seed)
    if args.dtype == "auto":
        dtype = torch.float16 if args.device == "mps" else torch.bfloat16
    else:
        dtype = _DTYPES[args.dtype]
    indices = torch.randint(
        0,
        2**args.bits,
        (args.out_features, args.in_features),
        dtype=torch.uint8,
    )
    packed = _pack(indices, args.bits).to(args.device)
    row_norms = torch.ones(args.out_features, device=args.device, dtype=torch.bfloat16)
    centroids = torch.linspace(-1.0, 1.0, 2**args.bits, device=args.device)
    indices_device = indices.long().to(args.device)
    weights = (packed, row_norms, centroids, indices_device)

    payloads = [_benchmark_rows(args, rows, dtype, weights) for rows in args.rows]
    if len(payloads) == 1:
        print(json.dumps(payloads[0], indent=2, sort_keys=True))
    else:
        print(json.dumps(payloads, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
