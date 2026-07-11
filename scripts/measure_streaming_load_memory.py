#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import psutil

_RESULT_PREFIX = "ORBITQUANT_LOAD_RESULT="


def _dtype(name: str):
    import torch

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _tensor_bytes(tensor: Any) -> int:
    if tensor is None or getattr(tensor, "device", None) is None:
        return 0
    if tensor.device.type == "meta":
        return 0
    return int(tensor.numel() * tensor.element_size())


def _module_metrics(module: Any) -> dict[str, int]:
    state_bytes = sum(_tensor_bytes(tensor) for tensor in module.state_dict().values())
    cache_bytes = sum(
        _tensor_bytes(getattr(child, "_dequantized_weight_cache", None))
        for child in module.modules()
    )
    quantizer = getattr(module, "hf_quantizer", None)
    return {
        "resident_state_bytes": state_bytes,
        "full_dequantized_cache_bytes": cache_bytes,
        "streamed_source_tensor_bytes": int(
            getattr(quantizer, "released_source_tensor_bytes", 0)
        ),
        "source_page_release_failures": int(
            getattr(quantizer, "source_page_release_failures", 0)
        ),
    }


def _worker(spec: dict[str, Any]) -> dict[str, Any]:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    framework = spec["framework"]
    mode = spec["mode"]
    model_id = spec["model_id"]
    if mode != "baseline":
        import orbitquant  # noqa: F401

    common: dict[str, Any] = {
        "revision": spec.get("revision"),
        "variant": spec.get("variant"),
        "local_files_only": spec.get("local_files_only", False),
        "trust_remote_code": spec.get("trust_remote_code", False),
        "torch_dtype": _dtype(spec["torch_dtype"]),
    }
    common = {key: value for key, value in common.items() if value is not None}

    started_at = time.perf_counter()
    if framework == "transformers":
        from transformers import AutoModel

        if mode == "streaming":
            from orbitquant import OrbitQuantConfig

            common["quantization_config"] = OrbitQuantConfig(
                target_policy=spec["target_policy"],
                weight_row_tile_size=spec["weight_row_tile_size"],
            )
            common["low_cpu_mem_usage"] = True
            common["use_safetensors"] = True
        model = AutoModel.from_pretrained(model_id, **common)
        metrics = _module_metrics(model)
    else:
        from diffusers import DiffusionPipeline

        if mode == "streaming":
            from orbitquant import (
                OrbitQuantConfig,
                build_diffusers_pipeline_quantization_config,
            )

            common["quantization_config"] = build_diffusers_pipeline_quantization_config(
                OrbitQuantConfig(
                    target_policy=spec["target_policy"],
                    weight_row_tile_size=spec["weight_row_tile_size"],
                ),
                components=spec["component"],
            )
            common["use_safetensors"] = True
        model = DiffusionPipeline.from_pretrained(model_id, **common)
        modules = [
            value
            for value in model.components.values()
            if isinstance(value, torch.nn.Module)
        ]
        metrics = {
            "resident_state_bytes": sum(
                _module_metrics(item)["resident_state_bytes"] for item in modules
            ),
            "full_dequantized_cache_bytes": sum(
                _module_metrics(item)["full_dequantized_cache_bytes"] for item in modules
            ),
            "streamed_source_tensor_bytes": sum(
                _module_metrics(item)["streamed_source_tensor_bytes"] for item in modules
            ),
            "source_page_release_failures": sum(
                _module_metrics(item)["source_page_release_failures"] for item in modules
            ),
        }

    metrics["load_wall_seconds"] = time.perf_counter() - started_at
    metrics["torch_cuda_allocated_peak_bytes"] = (
        int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
    )
    metrics["torch_cuda_reserved_peak_bytes"] = (
        int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None
    )
    return metrics


class _NvmlSampler:
    def __init__(self) -> None:
        self.error: str | None = None
        self._nvml = None
        self._handles: list[Any] = []
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handles = [
                pynvml.nvmlDeviceGetHandleByIndex(index)
                for index in range(pynvml.nvmlDeviceGetCount())
            ]
        except Exception as exc:
            self.error = str(exc)

    def sample(self, pid: int) -> int | None:
        if self._nvml is None:
            return None
        total = 0
        for handle in self._handles:
            process_lists = []
            for name in (
                "nvmlDeviceGetComputeRunningProcesses",
                "nvmlDeviceGetGraphicsRunningProcesses",
            ):
                try:
                    process_lists.append(getattr(self._nvml, name)(handle))
                except Exception:
                    continue
            for processes in process_lists:
                total += sum(
                    int(process.usedGpuMemory)
                    for process in processes
                    if process.pid == pid and process.usedGpuMemory is not None
                )
        return total


def _run_clean_worker(spec: dict[str, Any], poll_interval: float) -> dict[str, Any]:
    command = [sys.executable, str(Path(__file__).resolve()), "--worker-spec", json.dumps(spec)]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    observed = psutil.Process(process.pid)
    peak_rss = 0
    peak_vms = 0
    peak_nvml: int | None = None
    nvml = _NvmlSampler()
    while process.poll() is None:
        try:
            memory = observed.memory_info()
            peak_rss = max(peak_rss, int(memory.rss))
            peak_vms = max(peak_vms, int(memory.vms))
            sample = nvml.sample(process.pid)
            if sample is not None:
                peak_nvml = max(peak_nvml or 0, sample)
        except psutil.NoSuchProcess:
            break
        time.sleep(poll_interval)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        raise RuntimeError(
            f"{spec['mode']} worker failed with exit {process.returncode}:\n{stderr}"
        )
    result_lines = [line for line in stdout.splitlines() if line.startswith(_RESULT_PREFIX)]
    if len(result_lines) != 1:
        raise RuntimeError(
            f"worker emitted no unique result line; stdout:\n{stdout}\nstderr:\n{stderr}"
        )
    result = json.loads(result_lines[0].removeprefix(_RESULT_PREFIX))
    result.update(
        {
            "cpu_rss_peak_bytes": peak_rss,
            "virtual_memory_peak_bytes": peak_vms,
            "nvml_process_peak_bytes": peak_nvml,
            "nvml_unavailable_reason": nvml.error,
        }
    )
    return result


def _resolve_local_snapshot(model_id: str, revision: str | None) -> Path | None:
    path = Path(model_id).expanduser()
    if path.exists():
        return path.resolve()
    try:
        from huggingface_hub import snapshot_download

        return Path(
            snapshot_download(
                model_id,
                revision=revision,
                local_files_only=True,
            )
        )
    except Exception:
        return None


def _disk_size(path: Path | None) -> int | None:
    if path is None:
        return None
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _safetensors_stats(path: Path | None) -> dict[str, int | None]:
    if path is None:
        return {"source_tensor_bytes": None, "largest_source_tensor_bytes": None}
    from safetensors import safe_open

    dtype_sizes = {
        "BOOL": 1,
        "F8_E4M3": 1,
        "F8_E5M2": 1,
        "I8": 1,
        "U8": 1,
        "BF16": 2,
        "F16": 2,
        "I16": 2,
        "U16": 2,
        "F32": 4,
        "I32": 4,
        "U32": 4,
        "F64": 8,
        "I64": 8,
        "U64": 8,
    }

    total = 0
    largest = 0
    for checkpoint in path.rglob("*.safetensors"):
        with safe_open(checkpoint, framework="pt", device="cpu") as handle:
            for key in handle.keys():  # noqa: SIM118 - safe_open is not iterable
                tensor_slice = handle.get_slice(key)
                element_size = dtype_sizes[tensor_slice.get_dtype()]
                size = element_size
                for dimension in tensor_slice.get_shape():
                    size *= dimension
                total += size
                largest = max(largest, size)
    return {
        "source_tensor_bytes": total,
        "largest_source_tensor_bytes": largest,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare clean-process baseline, streaming, and prequantized load peaks."
    )
    parser.add_argument("--framework", choices=("transformers", "diffusers"))
    parser.add_argument("--model-id")
    parser.add_argument("--prequantized-model-id")
    parser.add_argument("--revision")
    parser.add_argument("--variant")
    parser.add_argument("--component", default="transformer")
    parser.add_argument(
        "--torch-dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--target-policy", default="auto")
    parser.add_argument("--weight-row-tile-size", type=int, default=256)
    parser.add_argument("--poll-interval", type=float, default=0.01)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker-spec", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.worker_spec is not None:
        print(_RESULT_PREFIX + json.dumps(_worker(json.loads(args.worker_spec)), sort_keys=True))
        return
    if not args.framework or not args.model_id or not args.prequantized_model_id:
        raise SystemExit("--framework, --model-id, and --prequantized-model-id are required")
    if args.weight_row_tile_size <= 0 or args.poll_interval <= 0:
        raise SystemExit("tile size and poll interval must be positive")

    common = {
        "framework": args.framework,
        "revision": args.revision,
        "variant": args.variant,
        "component": args.component,
        "torch_dtype": args.torch_dtype,
        "target_policy": args.target_policy,
        "weight_row_tile_size": args.weight_row_tile_size,
        "local_files_only": args.local_files_only,
        "trust_remote_code": args.trust_remote_code,
    }
    results = {}
    for mode, model_id in (
        ("baseline", args.model_id),
        ("streaming", args.model_id),
        ("prequantized", args.prequantized_model_id),
    ):
        results[mode] = _run_clean_worker(
            {**common, "mode": mode, "model_id": model_id},
            args.poll_interval,
        )
        snapshot = _resolve_local_snapshot(model_id, args.revision)
        results[mode]["disk_artifact_bytes"] = _disk_size(snapshot)
        results[mode].update(_safetensors_stats(snapshot))

    results["comparison"] = {
        "streaming_rss_reduction_vs_baseline_bytes": (
            results["baseline"]["cpu_rss_peak_bytes"]
            - results["streaming"]["cpu_rss_peak_bytes"]
        ),
        "prequantized_rss_reduction_vs_baseline_bytes": (
            results["baseline"]["cpu_rss_peak_bytes"]
            - results["prequantized"]["cpu_rss_peak_bytes"]
        ),
        "rss_and_virtual_memory_reported_separately": True,
    }
    payload = json.dumps(results, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
