from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import torch

from orbitquant.config import OrbitQuantConfig
from orbitquant.kernels import backend_capabilities, quantize_activations_kernel, select_backend
from orbitquant.layers import OrbitQuantLinear
from orbitquant.modeling import (
    QuantizationPrewarmSummary,
    prewarm_quantized_linear_modules,
    quantize_linear_modules,
)


def _resolve_device(device: str | torch.device) -> torch.device:
    requested = str(device)
    if requested != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif device.type == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


def _mean_time_ms(
    fn: Callable[[], torch.Tensor],
    *,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> float:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")

    for _ in range(warmup):
        fn()
    _synchronize(device)

    if device.type == "cuda" and torch.cuda.is_available():
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


def _time_once_ms(fn: Callable[[], Any], *, device: torch.device) -> tuple[Any, float]:
    _synchronize(device)
    if device.type == "cuda" and torch.cuda.is_available():
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


def _device_name(device: torch.device) -> str:
    if device.type == "cuda" and torch.cuda.is_available():
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return "mps"
    return "cpu"


def _peak_memory_bytes(device: torch.device) -> int | None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    return int(torch.cuda.max_memory_allocated(device))


def benchmark_orbit_linear(
    *,
    tokens: int = 1024,
    in_features: int = 3072,
    out_features: int = 3072,
    weight_bits: int = 4,
    activation_bits: int = 4,
    block_size: int | str = "paper",
    activation_kernel_backend: str = "auto",
    runtime_mode: str = "dequant_bf16",
    packed_matmul_block_m: int = 32,
    packed_matmul_block_n: int = 64,
    packed_matmul_block_k: int = 64,
    packed_matmul_num_warps: int = 8,
    device: str | torch.device = "auto",
    dtype: torch.dtype = torch.bfloat16,
    warmup: int = 5,
    iterations: int = 20,
    seed: int = 0,
) -> dict[str, Any]:
    if tokens <= 0:
        raise ValueError("tokens must be positive")
    if in_features <= 0 or out_features <= 0:
        raise ValueError("features must be positive")

    target_device = _resolve_device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested but CUDA is not available")
    if target_device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS benchmark requested but MPS is not available")

    torch.manual_seed(seed)
    if target_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(target_device)

    config = OrbitQuantConfig(
        weight_bits=weight_bits,
        activation_bits=activation_bits,
        block_size=block_size,
        activation_kernel_backend=activation_kernel_backend,
        runtime_mode=runtime_mode,
        packed_matmul_block_m=packed_matmul_block_m,
        packed_matmul_block_n=packed_matmul_block_n,
        packed_matmul_block_k=packed_matmul_block_k,
        packed_matmul_num_warps=packed_matmul_num_warps,
    )
    source = torch.nn.Linear(
        in_features,
        out_features,
        device=target_device,
        dtype=dtype,
    )
    source.requires_grad_(False)
    def quantize_source_linear() -> OrbitQuantLinear:
        return OrbitQuantLinear.from_linear(
            source,
            config=config,
            module_name="benchmark.linear",
        )

    quantized, weight_quantize_pack_cold_ms = _time_once_ms(
        quantize_source_linear,
        device=target_device,
    )
    quantized.to(target_device)
    x = torch.randn(tokens, in_features, device=target_device, dtype=dtype)

    if runtime_mode == "triton_packed_matmul":
        prewarm = QuantizationPrewarmSummary(
            orbitquant_modules=0,
            adaln_modules=0,
            total_modules=0,
            elapsed_seconds=0.0,
            device=str(target_device),
            dtype=str(dtype).removeprefix("torch."),
        )
    else:
        prewarm = prewarm_quantized_linear_modules(
            quantized,
            device=target_device,
            dtype=dtype,
        )

    def source_forward() -> torch.Tensor:
        return source(x)

    def activation_quant() -> torch.Tensor:
        return quantize_activations_kernel(
            x,
            rotation=quantized.rotation,
            codebook=quantized.activation_codebook,
            eps=quantized.activation_eps,
            backend=quantized.activation_kernel_backend,
            constant_tensors=quantized._activation_kernel_constant_tensors(target_device),
        )

    def weight_quantize_pack_hot() -> torch.Tensor:
        hot_quantized = quantize_source_linear()
        if hot_quantized.packed_weight_indices is not None:
            return hot_quantized.packed_weight_indices
        if hot_quantized.debug_weight is not None:
            return hot_quantized.debug_weight
        raise RuntimeError("benchmark quantized layer has no quantized weight tensor")

    def cold_weight_dequant() -> torch.Tensor:
        quantized.clear_dequantized_cache()
        return quantized._dequantize_weight(device=target_device, dtype=dtype)

    def cached_weight_dequant() -> torch.Tensor:
        return quantized._dequantize_weight(device=target_device, dtype=dtype)

    def forward_cold() -> torch.Tensor:
        quantized.clear_dequantized_cache()
        return quantized(x)

    def forward_prewarmed() -> torch.Tensor:
        return quantized(x)

    timings = {
        "weight_quantize_pack_cold_ms": weight_quantize_pack_cold_ms,
        "weight_quantize_pack_hot_ms": _mean_time_ms(
            weight_quantize_pack_hot,
            device=target_device,
            warmup=max(0, min(warmup, 2)),
            iterations=max(1, min(iterations, 5)),
        ),
        "torch_linear_ms": _mean_time_ms(
            source_forward,
            device=target_device,
            warmup=warmup,
            iterations=iterations,
        ),
        "activation_quant_ms": _mean_time_ms(
            activation_quant,
            device=target_device,
            warmup=warmup,
            iterations=iterations,
        ),
    }
    if runtime_mode == "triton_packed_matmul":
        timings["weight_dequant_cold_ms"] = None
        timings["weight_dequant_cached_ms"] = None
    else:
        timings["weight_dequant_cold_ms"] = _mean_time_ms(
            cold_weight_dequant,
            device=target_device,
            warmup=max(1, min(warmup, 3)),
            iterations=max(1, min(iterations, 5)),
        )
        prewarm_quantized_linear_modules(quantized, device=target_device, dtype=dtype)
        timings["weight_dequant_cached_ms"] = _mean_time_ms(
            cached_weight_dequant,
            device=target_device,
            warmup=warmup,
            iterations=iterations,
        )
    timings["forward_cold_ms"] = _mean_time_ms(
        forward_cold,
        device=target_device,
        warmup=max(1, min(warmup, 3)),
        iterations=max(1, min(iterations, 5)),
    )
    if runtime_mode != "triton_packed_matmul":
        prewarm_quantized_linear_modules(quantized, device=target_device, dtype=dtype)
    timings["forward_prewarmed_ms"] = _mean_time_ms(
        forward_prewarmed,
        device=target_device,
        warmup=warmup,
        iterations=iterations,
    )

    return {
        "device": str(target_device),
        "device_name": _device_name(target_device),
        "dtype": str(dtype).removeprefix("torch."),
        "tokens": tokens,
        "in_features": in_features,
        "out_features": out_features,
        "weight_bits": weight_bits,
        "activation_bits": activation_bits,
        "block_size": quantized.rotation.block_size,
        "activation_kernel_backend": activation_kernel_backend,
        "selected_activation_kernel_backend": select_backend(
            target_device, requested=activation_kernel_backend
        ),
        "weight_quantization_backend": (
            "triton_cuda"
            if target_device.type == "cuda" and backend_capabilities()["triton_cuda"]["available"]
            else "torch_reference"
        ),
        "runtime_mode": config.runtime_mode,
        "packed_matmul_tile": {
            "block_m": config.packed_matmul_block_m,
            "block_n": config.packed_matmul_block_n,
            "block_k": config.packed_matmul_block_k,
            "num_warps": config.packed_matmul_num_warps,
        },
        "full_fusion": False,
        "prewarm": prewarm.__dict__,
        "timings_ms": timings,
        "peak_memory_bytes": _peak_memory_bytes(target_device),
        "quantization_buffers": {
            "source_weight_device": str(source.weight.device),
            "source_weight_is_cuda": bool(source.weight.is_cuda),
            "packed_weight_indices_device": (
                None
                if quantized.packed_weight_indices is None
                else str(quantized.packed_weight_indices.device)
            ),
            "row_norms_device": (
                None if quantized.row_norms is None else str(quantized.row_norms.device)
            ),
            "debug_weight_device": (
                None if quantized.debug_weight is None else str(quantized.debug_weight.device)
            ),
            "packed_weight_indices_is_cuda": (
                None
                if quantized.packed_weight_indices is None
                else bool(quantized.packed_weight_indices.is_cuda)
            ),
            "row_norms_is_cuda": (
                None if quantized.row_norms is None else bool(quantized.row_norms.is_cuda)
            ),
        },
        "backend_capabilities": backend_capabilities(),
        "notes": (
            "weight_quantize_pack_cold_ms includes first-use backend compilation "
            "where applicable. On Triton/CUDA this can be CPU-heavy and show low "
            "GPU utilization; weight_quantize_pack_hot_ms measures the already "
            "compiled CUDA path. In dequant_bf16 mode, forward_prewarmed_ms uses "
            "OrbitQuant activation kernels plus cached dequantized weights and "
            "PyTorch linear. In triton_packed_matmul mode, forward_prewarmed_ms "
            "uses the opt-in packed-weight matmul path instead of the cached "
            "dequantized-weight path."
        ),
    }


class _QuantizeBenchModel(torch.nn.Module):
    def __init__(
        self,
        *,
        layers: int,
        in_features: int,
        hidden_features: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        blocks = []
        for _ in range(layers):
            blocks.append(
                torch.nn.ModuleDict(
                    {
                        "attn": torch.nn.ModuleDict(
                            {
                                "to_q": torch.nn.Linear(
                                    in_features, in_features, device=device, dtype=dtype
                                ),
                                "to_k": torch.nn.Linear(
                                    in_features, in_features, device=device, dtype=dtype
                                ),
                                "to_v": torch.nn.Linear(
                                    in_features, in_features, device=device, dtype=dtype
                                ),
                            }
                        ),
                        "ff": torch.nn.ModuleDict(
                            {
                                "linear_in": torch.nn.Linear(
                                    in_features, hidden_features, device=device, dtype=dtype
                                ),
                                "linear_out": torch.nn.Linear(
                                    hidden_features, in_features, device=device, dtype=dtype
                                ),
                            }
                        ),
                        "modulation": torch.nn.Linear(
                            in_features, in_features * 2, device=device, dtype=dtype
                        ),
                    }
                )
            )
        self.transformer_blocks = torch.nn.ModuleList(blocks)
        self.proj_out = torch.nn.Linear(in_features, in_features, device=device, dtype=dtype)


def benchmark_model_quantization(
    *,
    layers: int = 4,
    in_features: int = 3072,
    hidden_features: int | None = None,
    weight_bits: int = 4,
    activation_bits: int = 4,
    block_size: int | str = "paper",
    source_device: str | torch.device = "cpu",
    quantization_device: str | torch.device = "auto",
    staging_mode: str = "component",
    synchronize_per_module: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 0,
) -> dict[str, Any]:
    if layers <= 0:
        raise ValueError("layers must be positive")
    if in_features <= 0:
        raise ValueError("in_features must be positive")
    if hidden_features is None:
        hidden_features = in_features * 3
    if hidden_features <= 0:
        raise ValueError("hidden_features must be positive")

    source = _resolve_device(source_device)
    target = _resolve_device(quantization_device)
    if source.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA source device requested but CUDA is not available")
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA quantization device requested but CUDA is not available")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if target.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target)

    model = _QuantizeBenchModel(
        layers=layers,
        in_features=in_features,
        hidden_features=hidden_features,
        device=source,
        dtype=dtype,
    )
    model.requires_grad_(False)
    config = OrbitQuantConfig(
        weight_bits=weight_bits,
        activation_bits=activation_bits,
        block_size=block_size,
        target_policy="flux2",
        activation_kernel_backend="triton_cuda" if target.type == "cuda" else "auto",
        runtime_mode="dequant_bf16",
    )

    _synchronize(target)
    started_at = time.perf_counter()
    summary = quantize_linear_modules(
        model,
        config,
        quantization_device=target,
        staging_mode=staging_mode,
        synchronize_per_module=synchronize_per_module,
    )
    _synchronize(target)
    wall_seconds = time.perf_counter() - started_at

    return {
        "source_device": str(source),
        "quantization_device": str(target),
        "device_name": _device_name(target),
        "dtype": str(dtype).removeprefix("torch."),
        "layers": layers,
        "in_features": in_features,
        "hidden_features": hidden_features,
        "weight_bits": weight_bits,
        "activation_bits": activation_bits,
        "block_size": block_size,
        "staging_mode": staging_mode,
        "synchronize_per_module": synchronize_per_module,
        "wall_seconds": wall_seconds,
        "summary": summary.__dict__,
        "peak_memory_bytes": _peak_memory_bytes(target),
        "notes": (
            "This benchmark measures the full model replacement loop, including "
            "host-to-device staging when source_device is cpu and quantization_device "
            "is cuda. The OrbitQuant/AdaLN compute fields exclude explicit transfer "
            "time; device_transfer_seconds is reported separately."
        ),
    }
