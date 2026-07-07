import torch

from orbitquant.benchmarks import benchmark_orbit_linear


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
    assert result["quantization_buffers"]["packed_weight_indices_is_cuda"] is False
    assert result["quantization_buffers"]["row_norms_is_cuda"] is False
