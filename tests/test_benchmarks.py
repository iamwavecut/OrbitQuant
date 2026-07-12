import torch

import orbitquant.benchmarks as benchmarks_module
from orbitquant.benchmarks import _weight_quantization_backend_label, benchmark_orbit_linear
from orbitquant.layers import OrbitQuantLinear


def test_benchmark_orbit_linear_reports_stage_timings_on_cpu():
    result = benchmark_orbit_linear(
        tokens=4,
        in_features=16,
        out_features=8,
        weight_bits=4,
        activation_bits=4,
        block_size=8,
        activation_kernel_backend="cpu",
        device="cpu",
        dtype=torch.float32,
        warmup=0,
        iterations=1,
        seed=0,
    )

    assert result["device"] == "cpu"
    assert result["dtype"] == "float32"
    assert result["full_fusion"] is False
    assert result["prewarm"]["total_modules"] == 1
    assert result["prewarm"]["device"] == "cpu"
    assert result["prewarm"]["dtype"] == "float32"
    for key in (
        "weight_quantize_pack_cold_ms",
        "weight_quantize_pack_hot_ms",
        "torch_linear_ms",
        "activation_quant_ms",
        "weight_dequant_cold_ms",
        "weight_dequant_cached_ms",
        "forward_cold_ms",
        "forward_prewarmed_ms",
    ):
        assert result["timings_ms"][key] >= 0.0
    assert result["quantization_buffers"]["packed_weight_indices_device"] == "cpu"
    assert result["quantization_buffers"]["row_norms_device"] == "cpu"
    assert result["quantization_buffers"]["source_weight_device_type"] == "cpu"
    assert result["quantization_buffers"]["packed_weight_indices_device_type"] == "cpu"
    assert result["quantization_buffers"]["row_norms_device_type"] == "cpu"
    assert result["quantization_buffers"]["packed_weight_indices_is_cuda"] is False
    assert result["quantization_buffers"]["row_norms_is_cuda"] is False


def test_benchmark_orbit_linear_accepts_runtime_mode_override_on_cpu():
    result = benchmark_orbit_linear(
        tokens=4,
        in_features=16,
        out_features=8,
        block_size=8,
        activation_kernel_backend="cpu",
        runtime_mode="debug_no_activation_quant",
        device="cpu",
        dtype=torch.float32,
        warmup=0,
        iterations=1,
    )

    assert result["runtime_mode"] == "debug_no_activation_quant"


def test_benchmark_orbit_linear_reports_tuned_packed_matmul_default_tile_on_cpu():
    result = benchmark_orbit_linear(
        tokens=4,
        in_features=16,
        out_features=8,
        block_size=8,
        activation_kernel_backend="cpu",
        runtime_mode="debug_no_activation_quant",
        device="cpu",
        dtype=torch.float32,
        warmup=0,
        iterations=1,
    )

    assert result["packed_matmul_tile"] == {
        "block_m": 64,
        "block_n": 64,
        "block_k": 128,
        "num_warps": 4,
    }


def test_benchmark_orbit_linear_treats_native_packed_runtime_as_cacheless(monkeypatch):
    def fail_prewarm(*args, **kwargs):
        raise AssertionError("packed runtime must not materialize dequant prewarm")

    def fail_dequant(*args, **kwargs):
        raise AssertionError("packed runtime benchmark must not dequantize weights")

    def fake_forward(self, x):
        return torch.zeros(*x.shape[:-1], self.out_features, device=x.device, dtype=x.dtype)

    monkeypatch.setattr(benchmarks_module, "prewarm_quantized_linear_modules", fail_prewarm)
    monkeypatch.setattr(OrbitQuantLinear, "_dequantize_weight", fail_dequant)
    monkeypatch.setattr(OrbitQuantLinear, "forward", fake_forward)

    result = benchmark_orbit_linear(
        tokens=4,
        in_features=16,
        out_features=8,
        block_size=8,
        activation_kernel_backend="cpu",
        runtime_mode="native_packed_matmul",
        device="cpu",
        dtype=torch.float32,
        warmup=0,
        iterations=1,
    )

    assert result["runtime_mode"] == "native_packed_matmul"
    assert result["prewarm"]["total_modules"] == 0
    assert result["timings_ms"]["weight_dequant_cold_ms"] is None
    assert result["timings_ms"]["weight_dequant_cached_ms"] is None


def test_benchmark_weight_backend_label_separates_reference_mps_from_cpu():
    assert _weight_quantization_backend_label(torch.device("cpu")) == "torch_reference"
    assert _weight_quantization_backend_label(torch.device("mps")) == "torch_reference_mps"


def test_benchmark_weight_backend_label_reports_rocm_on_hip(monkeypatch):
    monkeypatch.setattr(benchmarks_module.torch.version, "hip", "7.2", raising=False)
    monkeypatch.setattr(
        benchmarks_module,
        "backend_capabilities",
        lambda: {"triton_rocm": {"available": True}},
    )

    assert _weight_quantization_backend_label(torch.device("cuda")) == "triton_rocm"


def test_benchmark_weight_backend_label_reports_explicit_xpu(monkeypatch):
    monkeypatch.setattr(
        benchmarks_module,
        "backend_capabilities",
        lambda: {"triton_xpu": {"available": True}},
    )

    assert _weight_quantization_backend_label(torch.device("xpu")) == "triton_xpu"
