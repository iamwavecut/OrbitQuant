import torch

import orbitquant.modeling as modeling_module
from orbitquant import OrbitQuantConfig, OrbitQuantLinear
from orbitquant.adaln import RTNInt4Linear
from orbitquant.modeling import (
    inspect_linear_module_policy,
    prewarm_quantized_linear_modules,
    quantize_linear_modules,
)


class TinyPipelineTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {
                        "attn": torch.nn.ModuleDict({"to_q": torch.nn.Linear(16, 16)}),
                        "modulation": torch.nn.Linear(16, 32),
                    }
                )
            ]
        )
        self.proj_out = torch.nn.Linear(16, 16)


def test_quantize_linear_modules_replaces_orbit_and_adaln_targets_only():
    model = TinyPipelineTransformer()
    config = OrbitQuantConfig(block_size=8)

    summary = quantize_linear_modules(model, config)

    assert isinstance(model.transformer_blocks[0]["attn"]["to_q"], OrbitQuantLinear)
    assert isinstance(model.transformer_blocks[0]["modulation"], RTNInt4Linear)
    assert isinstance(model.proj_out, torch.nn.Linear)
    assert summary.quantized_modules == ["transformer_blocks.0.attn.to_q"]
    assert summary.adaln_modules == ["transformer_blocks.0.modulation"]
    assert summary.skipped_modules == ["proj_out"]
    assert summary.quantization_device == ("cuda" if torch.cuda.is_available() else "cpu")
    assert summary.weight_quantization_backend in {"torch_reference", "triton_cuda"}
    assert summary.quantization_staging_mode == "streaming"
    assert summary.synchronize_per_module is False
    assert summary.elapsed_seconds >= 0.0
    assert summary.orbitquant_seconds >= 0.0
    assert summary.adaln_seconds >= 0.0
    assert summary.device_transfer_seconds >= 0.0
    assert summary.module_device_transfer_count >= 0
    assert summary.source_linear_device_counts
    assert summary.quantized_buffer_device_counts


def test_inspect_linear_module_policy_reports_inventory_without_mutating_model():
    model = TinyPipelineTransformer()
    config = OrbitQuantConfig(block_size=8)

    inventory = inspect_linear_module_policy(model, config)

    assert inventory["target_policy"] == "auto"
    assert inventory["linear_module_count"] == 3
    assert inventory["action_counts"] == {
        "orbitquant": 1,
        "adaln_int4_rtn": 1,
        "bf16_skip": 1,
    }
    assert inventory["quantized_modules"] == ["transformer_blocks.0.attn.to_q"]
    assert inventory["adaln_modules"] == ["transformer_blocks.0.modulation"]
    assert inventory["skipped_modules"] == ["proj_out"]
    assert inventory["modules"][0] == {
        "name": "transformer_blocks.0.attn.to_q",
        "action": "orbitquant",
        "reason": "transformer block linear",
        "dtype": None,
        "in_features": 16,
        "out_features": 16,
        "bias": True,
        "weight_dtype": "float32",
        "weight_device": "cpu",
    }
    assert isinstance(model.transformer_blocks[0]["attn"]["to_q"], torch.nn.Linear)
    assert isinstance(model.transformer_blocks[0]["modulation"], torch.nn.Linear)


def test_quantize_linear_modules_keeps_dtype_overridden_modules_unquantized():
    model = TinyPipelineTransformer()
    config = OrbitQuantConfig(
        block_size=8,
        modules_dtype_dict={"float16": ["transformer_blocks.0.attn.to_q"]},
    )

    summary = quantize_linear_modules(model, config)

    overridden = model.transformer_blocks[0]["attn"]["to_q"]
    assert isinstance(overridden, torch.nn.Linear)
    assert overridden.weight.dtype is torch.float16
    assert summary.quantized_modules == []
    assert summary.adaln_modules == ["transformer_blocks.0.modulation"]
    assert "transformer_blocks.0.attn.to_q" in summary.skipped_modules


def test_quantize_linear_modules_fails_loud_for_unavailable_cuda_device(monkeypatch):
    model = TinyPipelineTransformer()
    config = OrbitQuantConfig(block_size=8)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    try:
        quantize_linear_modules(model, config, quantization_device="cuda")
    except RuntimeError as exc:
        assert "CUDA quantization device requested" in str(exc)
    else:
        raise AssertionError("unavailable CUDA quantization device was accepted")


def test_quantize_linear_modules_fails_loud_for_cuda_without_triton(monkeypatch):
    model = TinyPipelineTransformer()
    config = OrbitQuantConfig(block_size=8)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        modeling_module,
        "available_backends",
        lambda: {"cpu": True, "mps": False, "triton_cuda": False},
    )

    try:
        quantize_linear_modules(model, config, quantization_device="cuda")
    except RuntimeError as exc:
        assert "Triton CUDA backend" in str(exc)
    else:
        raise AssertionError("CUDA quantization without Triton was accepted")


def test_auto_quantization_device_prefers_cuda_and_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert modeling_module._quantization_device("auto") == torch.device("cuda")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert modeling_module._quantization_device("auto") == torch.device("cpu")


def test_quantize_linear_modules_preserve_device_mode_records_reference_backend():
    model = TinyPipelineTransformer()
    config = OrbitQuantConfig(block_size=8)

    summary = quantize_linear_modules(model, config, quantization_device=None)

    assert summary.quantization_device == "preserve"
    assert summary.weight_quantization_backend == "module_device"


def test_quantize_linear_modules_component_staging_moves_model_once():
    model = TinyPipelineTransformer()
    config = OrbitQuantConfig(block_size=8)

    summary = quantize_linear_modules(
        model,
        config,
        quantization_device="cpu",
        staging_mode="component",
    )

    assert summary.quantization_staging_mode == "component"
    assert summary.synchronize_per_module is False
    assert summary.source_linear_device_counts == {"cpu": 3}
    assert summary.module_device_transfer_count == 0
    assert isinstance(model.transformer_blocks[0]["attn"]["to_q"], OrbitQuantLinear)


def test_quantize_linear_modules_rejects_unknown_staging_mode():
    model = TinyPipelineTransformer()
    config = OrbitQuantConfig(block_size=8)

    try:
        quantize_linear_modules(model, config, staging_mode="elevator")
    except ValueError as exc:
        assert "staging_mode" in str(exc)
    else:
        raise AssertionError("unknown quantization staging mode was accepted")


def test_quantize_linear_modules_records_debug_synchronize_per_module():
    model = TinyPipelineTransformer()
    config = OrbitQuantConfig(block_size=8)

    summary = quantize_linear_modules(
        model,
        config,
        quantization_device="cpu",
        synchronize_per_module=True,
    )

    assert summary.synchronize_per_module is True
    assert isinstance(model.transformer_blocks[0]["attn"]["to_q"], OrbitQuantLinear)


def test_prewarm_quantized_linear_modules_materializes_weight_caches():
    model = TinyPipelineTransformer()
    config = OrbitQuantConfig(block_size=8)
    quantize_linear_modules(model, config)

    orbit_layer = model.transformer_blocks[0]["attn"]["to_q"]
    adaln_layer = model.transformer_blocks[0]["modulation"]
    assert isinstance(orbit_layer, OrbitQuantLinear)
    assert isinstance(adaln_layer, RTNInt4Linear)
    assert orbit_layer._dequantized_weight_cache is None
    assert adaln_layer._dequantized_weight_cache is None

    summary = prewarm_quantized_linear_modules(model, device="cpu", dtype=torch.float32)

    assert summary.orbitquant_modules == 1
    assert summary.adaln_modules == 1
    assert summary.total_modules == 2
    assert summary.device == "cpu"
    assert summary.dtype == "float32"
    assert summary.elapsed_seconds >= 0.0
    assert orbit_layer._dequantized_weight_cache is not None
    assert adaln_layer._dequantized_weight_cache is not None
    assert orbit_layer._dequantized_weight_cache.dtype is torch.float32
    assert adaln_layer._dequantized_weight_cache.dtype is torch.float32


def test_prewarm_quantized_linear_modules_reports_empty_model_without_side_effects():
    model = torch.nn.Sequential(torch.nn.Linear(4, 4))

    summary = prewarm_quantized_linear_modules(model, device="cpu", dtype=torch.float32)

    assert summary.orbitquant_modules == 0
    assert summary.adaln_modules == 0
    assert summary.total_modules == 0
    assert summary.device == "cpu"
    assert summary.dtype == "float32"
