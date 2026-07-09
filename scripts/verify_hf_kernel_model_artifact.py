#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

from orbitquant.config import OrbitQuantConfig
from orbitquant.layers import OrbitQuantLinear

DEFAULT_ARTIFACT_REPO = "WaveCut/FLUX.2-klein-4B-OrbitQuant-W4A4"


def _stage(message: str) -> None:
    print(f"VERIFY_STAGE {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {message}")


def _device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError(
            "no CUDA or MPS device is available; native_packed_matmul verification "
            "requires a GPU/MPS runtime"
        )
    device = torch.device(name)
    if device.type not in {"cuda", "mps"}:
        raise ValueError("device must be 'auto', 'cuda', or 'mps'")
    return device


def _dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "auto":
        return torch.float16 if device.type == "mps" else torch.bfloat16
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype {name!r}")


def _download_artifact(repo_id: str, local_dir: str | None) -> Path:
    allow_patterns = [
        "model.safetensors",
        "orbitquant_manifest.json",
        "quantization_config.json",
    ]
    path = snapshot_download(
        repo_id,
        local_dir=local_dir,
        allow_patterns=allow_patterns,
    )
    return Path(path)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _select_module(manifest: dict[str, Any], requested: str | None) -> str:
    if requested is not None:
        if requested not in manifest["quantized_modules"]:
            raise ValueError(f"module {requested!r} is not in manifest quantized_modules")
        return requested
    return manifest["quantized_modules"][0]


def _load_layer_tensors(artifact_dir: Path, module_name: str) -> dict[str, torch.Tensor | None]:
    packed_key = f"{module_name}.packed_weight_indices"
    row_norm_key = f"{module_name}.row_norms"
    bias_key = f"{module_name}.bias"
    with safe_open(artifact_dir / "model.safetensors", framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        missing = sorted({packed_key, row_norm_key} - keys)
        if missing:
            raise RuntimeError(f"artifact is missing layer tensors: {missing}")
        return {
            "packed_weight_indices": handle.get_tensor(packed_key),
            "row_norms": handle.get_tensor(row_norm_key),
            "bias": handle.get_tensor(bias_key) if bias_key in keys else None,
        }


def _infer_features(
    *,
    packed_weight_indices: torch.Tensor,
    row_norms: torch.Tensor,
    bits: int,
) -> tuple[int, int]:
    out_features = int(row_norms.numel())
    value_capacity = (int(packed_weight_indices.numel()) * 8) // bits
    in_features = value_capacity // out_features
    expected_bytes = (out_features * in_features * bits + 7) // 8
    if expected_bytes != int(packed_weight_indices.numel()):
        raise RuntimeError(
            "could not infer a dense Linear shape from packed tensor length: "
            f"out_features={out_features}, inferred_in_features={in_features}, "
            f"packed_bytes={packed_weight_indices.numel()}, expected_bytes={expected_bytes}"
        )
    return in_features, out_features


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _time_once_ms(fn, *, device: torch.device) -> tuple[torch.Tensor, float]:
    _synchronize(device)
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        end.record()
        end.synchronize()
        return result, float(start.elapsed_time(end))

    started_at = time.perf_counter()
    result = fn()
    _synchronize(device)
    return result, (time.perf_counter() - started_at) * 1000.0


def _mean_time_ms(fn, *, device: torch.device, warmup: int, iterations: int) -> float:
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    for _ in range(warmup):
        fn()
    _synchronize(device)

    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        return float(start.elapsed_time(end) / iterations)

    started_at = time.perf_counter()
    for _ in range(iterations):
        fn()
    _synchronize(device)
    return (time.perf_counter() - started_at) * 1000.0 / iterations


def _storage_payload(
    *,
    packed_weight_indices: torch.Tensor,
    row_norms: torch.Tensor,
    bias: torch.Tensor | None,
    bits: int,
    out_features: int,
    in_features: int,
    dtype: torch.dtype,
) -> dict[str, int | float]:
    packed_bytes = int(packed_weight_indices.numel() * packed_weight_indices.element_size())
    row_norm_bytes = int(row_norms.numel() * row_norms.element_size())
    bias_bytes = 0 if bias is None else int(bias.numel() * bias.element_size())
    centroid_bytes = int((2**bits) * torch.empty((), dtype=torch.float32).element_size())
    packed_path_bytes = packed_bytes + row_norm_bytes + centroid_bytes + bias_bytes
    materialized_weight_bytes = int(
        out_features * in_features * torch.empty((), dtype=dtype).element_size()
    )
    return {
        "packed_weight_indices_bytes": packed_bytes,
        "row_norms_bytes": row_norm_bytes,
        "centroid_bytes": centroid_bytes,
        "bias_bytes": bias_bytes,
        "packed_weight_path_bytes": packed_path_bytes,
        "materialized_weight_bytes": materialized_weight_bytes,
        "packed_weight_path_vs_materialized_weight_ratio": (
            packed_path_bytes / materialized_weight_bytes
            if materialized_weight_bytes > 0
            else 0.0
        ),
    }


def _make_layer(
    *,
    config_payload: dict[str, Any],
    module_name: str,
    in_features: int,
    out_features: int,
    tensors: dict[str, torch.Tensor | None],
    runtime_mode: str,
    activation_kernel_backend: str,
) -> OrbitQuantLinear:
    config_payload = dict(config_payload)
    config_payload["runtime_mode"] = runtime_mode
    config_payload["activation_kernel_backend"] = activation_kernel_backend
    config = OrbitQuantConfig.from_dict(config_payload)
    return OrbitQuantLinear(
        in_features=in_features,
        out_features=out_features,
        config=config,
        module_name=module_name,
        bias=tensors["bias"],
        packed_weight_indices=tensors["packed_weight_indices"],
        row_norms=tensors["row_norms"],
        debug_weight=None,
    ).eval()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that one published OrbitQuant model artifact layer can execute "
            "through the native packed matmul kernel."
        )
    )
    parser.add_argument("--artifact-repo", default=DEFAULT_ARTIFACT_REPO)
    parser.add_argument("--local-dir", default=None)
    parser.add_argument("--module-name", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps"])
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
    )
    parser.add_argument(
        "--runtime-mode",
        default="native_packed_matmul",
        choices=["native_packed_matmul", "auto_fused"],
    )
    parser.add_argument("--activation-kernel-backend", default="auto")
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=8e-2)
    parser.add_argument("--rtol", type=float, default=8e-2)
    args = parser.parse_args()

    target_device = _device(args.device)
    target_dtype = _dtype(args.dtype, target_device)
    _stage(f"device device={target_device} dtype={target_dtype}")

    if args.runtime_mode == "native_packed_matmul":
        _stage("native-kernel-load-start")
        from orbitquant.kernels.native_packed_matmul import load_native_packed_matmul_kernel

        kernel = load_native_packed_matmul_kernel()
        if not hasattr(kernel, "matmul_packed_weight"):
            raise RuntimeError("native kernel package is missing matmul_packed_weight")
        _stage(f"native-kernel-load-done kernel={kernel}")

    _stage(f"artifact-download-start repo={args.artifact_repo}")
    artifact_dir = _download_artifact(args.artifact_repo, args.local_dir)
    _stage(f"artifact-download-done path={artifact_dir}")

    manifest = _load_json(artifact_dir / "orbitquant_manifest.json")
    config_payload = _load_json(artifact_dir / "quantization_config.json")
    module_name = _select_module(manifest, args.module_name)
    tensors = _load_layer_tensors(artifact_dir, module_name)
    packed = tensors["packed_weight_indices"]
    row_norms = tensors["row_norms"]
    if packed is None or row_norms is None:
        raise AssertionError("required layer tensors were unexpectedly None")

    in_features, out_features = _infer_features(
        packed_weight_indices=packed,
        row_norms=row_norms,
        bits=int(manifest["weight_bits"]),
    )
    _stage(
        "layer-selected "
        f"module={module_name} in_features={in_features} out_features={out_features}"
    )

    packed_layer = _make_layer(
        config_payload=config_payload,
        module_name=module_name,
        in_features=in_features,
        out_features=out_features,
        tensors=tensors,
        runtime_mode=args.runtime_mode,
        activation_kernel_backend=args.activation_kernel_backend,
    )
    reference_layer = _make_layer(
        config_payload=config_payload,
        module_name=module_name,
        in_features=in_features,
        out_features=out_features,
        tensors=tensors,
        runtime_mode="dequant_bf16",
        activation_kernel_backend=args.activation_kernel_backend,
    )

    torch.manual_seed(args.seed)
    x = torch.randn(args.tokens, in_features, device=target_device, dtype=target_dtype)
    packed_layer = packed_layer.to(target_device)
    reference_layer = reference_layer.to(target_device)

    if target_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target_device)

    def packed_forward() -> torch.Tensor:
        return packed_layer(x)

    def reference_forward() -> torch.Tensor:
        return reference_layer(x)

    _stage("forward-start")
    with torch.inference_mode():
        packed_output, packed_forward_first_ms = _time_once_ms(
            packed_forward, device=target_device
        )
        reference_output, reference_forward_first_ms = _time_once_ms(
            reference_forward, device=target_device
        )
        packed_forward_prewarmed_ms = _mean_time_ms(
            packed_forward,
            device=target_device,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        reference_forward_prewarmed_ms = _mean_time_ms(
            reference_forward,
            device=target_device,
            warmup=args.warmup,
            iterations=args.iterations,
        )
    _synchronize(target_device)
    _stage("forward-done")

    max_abs_error = float((packed_output.float() - reference_output.float()).abs().max().item())
    finite = bool(torch.isfinite(packed_output).all().item())
    close = bool(
        torch.allclose(
            packed_output.float(),
            reference_output.float(),
            atol=args.atol,
            rtol=args.rtol,
        )
    )
    if not finite:
        raise RuntimeError("native packed output contains non-finite values")
    if not close:
        raise RuntimeError(
            "native packed output differs from dequant_bf16 reference: "
            f"max_abs_error={max_abs_error}, atol={args.atol}, rtol={args.rtol}"
        )

    payload: dict[str, Any] = {
        "artifact_repo": args.artifact_repo,
        "source_model_id": manifest["source_model_id"],
        "source_revision": manifest["source_revision"],
        "module_name": module_name,
        "runtime_mode": args.runtime_mode,
        "activation_kernel_backend": args.activation_kernel_backend,
        "device": str(target_device),
        "dtype": str(target_dtype).replace("torch.", ""),
        "tokens": args.tokens,
        "weight_bits": int(manifest["weight_bits"]),
        "activation_bits": int(manifest["activation_bits"]),
        "in_features": in_features,
        "out_features": out_features,
        "output_shape": list(packed_output.shape),
        "finite": finite,
        "allclose_to_dequant_bf16": close,
        "max_abs_error_vs_dequant_bf16": max_abs_error,
        "timings_ms": {
            f"{args.runtime_mode}_forward_first_ms": packed_forward_first_ms,
            "dequant_bf16_forward_first_ms": reference_forward_first_ms,
            f"{args.runtime_mode}_forward_prewarmed_ms": packed_forward_prewarmed_ms,
            "dequant_bf16_forward_prewarmed_ms": reference_forward_prewarmed_ms,
            "warmup": args.warmup,
            "iterations": args.iterations,
        },
        "peak_memory_bytes": (
            int(torch.cuda.max_memory_allocated(target_device))
            if target_device.type == "cuda"
            else None
        ),
        **_storage_payload(
            packed_weight_indices=packed,
            row_norms=row_norms,
            bias=tensors["bias"],
            bits=int(manifest["weight_bits"]),
            out_features=out_features,
            in_features=in_features,
            dtype=target_dtype,
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
