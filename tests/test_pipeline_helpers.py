import json
import sys
from types import SimpleNamespace

import pytest
import torch

from orbitquant import OrbitQuantConfig
from orbitquant.layers import OrbitQuantLinear
from orbitquant.pipeline import (
    build_diffusers_pipeline_quantization_config,
    load_quantized_pipeline_component,
    load_quantized_pipeline_from_artifact,
    quantize_pipeline,
    save_quantized_pipeline_component,
)


class TinyPipeline:
    def __init__(self):
        self.transformer = torch.nn.Module()
        self.transformer.transformer_blocks = torch.nn.ModuleList(
            [torch.nn.ModuleDict({"attn": torch.nn.ModuleDict({"to_q": torch.nn.Linear(8, 8)})})]
        )
        self.denoiser = torch.nn.Module()
        self.denoiser.transformer_blocks = torch.nn.ModuleList(
            [torch.nn.ModuleDict({"attn": torch.nn.ModuleDict({"to_q": torch.nn.Linear(8, 8)})})]
        )


def test_quantize_pipeline_quantizes_named_component():
    pipeline = TinyPipeline()
    config = OrbitQuantConfig(block_size=4, target_policy="generic_dit")

    summary = quantize_pipeline(pipeline, config, component="transformer")

    assert summary.quantized_modules == ["transformer_blocks.0.attn.to_q"]
    assert isinstance(pipeline.transformer.transformer_blocks[0]["attn"]["to_q"], OrbitQuantLinear)


def test_build_diffusers_pipeline_quantization_config_uses_component_mapping():
    pytest.importorskip("diffusers")
    config = OrbitQuantConfig(
        weight_bits=3,
        activation_bits=3,
        block_size=4,
        target_policy="generic_dit",
    )

    pipeline_config = build_diffusers_pipeline_quantization_config(
        config, components=["transformer", "denoiser"]
    )

    assert pipeline_config.is_granular is True
    assert set(pipeline_config.quant_mapping) == {"transformer", "denoiser"}
    resolved = pipeline_config._resolve_quant_config(
        is_diffusers=True, module_name="transformer"
    )
    restored = OrbitQuantConfig.from_dict(resolved.to_dict())

    assert resolved is config
    assert restored.quant_method == "orbitquant"
    assert restored.weight_bits == 3
    assert restored.activation_bits == 3
    assert pipeline_config._resolve_quant_config(
        is_diffusers=True, module_name="text_encoder"
    ) is None


def test_build_diffusers_pipeline_quantization_config_can_use_backend_mode():
    pytest.importorskip("diffusers")
    config = OrbitQuantConfig(block_size=4, target_policy="generic_dit")

    pipeline_config = build_diffusers_pipeline_quantization_config(
        config, components="transformer", granular=False
    )
    resolved = pipeline_config._resolve_quant_config(
        is_diffusers=True, module_name="transformer"
    )

    assert pipeline_config.is_granular is False
    assert pipeline_config.quant_backend == "orbitquant"
    assert pipeline_config.components_to_quantize == ["transformer"]
    assert resolved.to_dict() == config.to_dict()
    assert pipeline_config._resolve_quant_config(
        is_diffusers=True, module_name="text_encoder"
    ) is None


def test_build_diffusers_pipeline_quantization_config_rejects_empty_components():
    pytest.importorskip("diffusers")
    config = OrbitQuantConfig(block_size=4)

    try:
        build_diffusers_pipeline_quantization_config(config, components=[])
    except ValueError as exc:
        assert "components" in str(exc)
    else:
        raise AssertionError("empty Diffusers pipeline component list was accepted")


def test_save_quantized_pipeline_component_writes_artifact(tmp_path):
    pipeline = TinyPipeline()
    config = OrbitQuantConfig(block_size=4, target_policy="generic_dit")
    summary = quantize_pipeline(pipeline, config, component="transformer")

    manifest = save_quantized_pipeline_component(
        pipeline,
        tmp_path,
        config=config,
        component="transformer",
        source_model_id="example/model",
        source_revision="abc123",
        source_license="apache-2.0",
        summary=summary,
    )

    assert manifest.source_model_id == "example/model"
    assert (tmp_path / "model.safetensors").is_file()
    assert (tmp_path / "orbitquant_manifest.json").is_file()


def test_save_quantized_pipeline_component_records_component_in_model_index(tmp_path):
    pipeline = TinyPipeline()
    config = OrbitQuantConfig(block_size=4, target_policy="generic_dit")
    summary = quantize_pipeline(pipeline, config, component="denoiser")

    save_quantized_pipeline_component(
        pipeline,
        tmp_path,
        config=config,
        component="denoiser",
        source_model_id="example/model",
        source_revision="abc123",
        source_license="apache-2.0",
        summary=summary,
    )

    model_index = json.loads((tmp_path / "model_index.json").read_text())
    assert model_index["component"] == "denoiser"


def test_load_quantized_pipeline_component_restores_saved_component_artifact(tmp_path):
    source_pipeline = TinyPipeline()
    config = OrbitQuantConfig(block_size=4, target_policy="generic_dit")
    summary = quantize_pipeline(source_pipeline, config, component="transformer")
    save_quantized_pipeline_component(
        source_pipeline,
        tmp_path,
        config=config,
        component="transformer",
        source_model_id="example/model",
        source_revision="abc123",
        source_license="apache-2.0",
        summary=summary,
    )

    restored_pipeline = TinyPipeline()
    manifest = load_quantized_pipeline_component(
        restored_pipeline, tmp_path, component="transformer"
    )

    restored_layer = restored_pipeline.transformer.transformer_blocks[0]["attn"]["to_q"]
    assert manifest.source_model_id == "example/model"
    assert isinstance(restored_layer, OrbitQuantLinear)
    assert torch.isfinite(restored_layer(torch.randn(1, 2, 8))).all()


def test_load_quantized_pipeline_component_applies_runtime_override_without_rewriting_manifest(
    tmp_path,
):
    source_pipeline = TinyPipeline()
    config = OrbitQuantConfig(
        block_size=4,
        target_policy="generic_dit",
        runtime_mode="dequant_bf16",
        activation_kernel_backend="cpu",
    )
    summary = quantize_pipeline(source_pipeline, config, component="transformer")
    save_quantized_pipeline_component(
        source_pipeline,
        tmp_path,
        config=config,
        component="transformer",
        source_model_id="example/model",
        source_revision="abc123",
        source_license="apache-2.0",
        summary=summary,
    )

    restored_pipeline = TinyPipeline()
    manifest = load_quantized_pipeline_component(
        restored_pipeline,
        tmp_path,
        component="transformer",
        runtime_mode="debug_no_activation_quant",
        activation_kernel_backend="auto",
    )

    restored_layer = restored_pipeline.transformer.transformer_blocks[0]["attn"]["to_q"]
    assert manifest.runtime_mode == "dequant_bf16"
    assert isinstance(restored_layer, OrbitQuantLinear)
    assert restored_layer.runtime_mode == "debug_no_activation_quant"
    assert restored_layer.activation_kernel_backend == "auto"


def test_load_quantized_pipeline_from_artifact_loads_source_pipeline_and_component(tmp_path):
    source_pipeline = TinyPipeline()
    config = OrbitQuantConfig(block_size=4, target_policy="generic_dit")
    summary = quantize_pipeline(source_pipeline, config, component="transformer")
    save_quantized_pipeline_component(
        source_pipeline,
        tmp_path,
        config=config,
        component="transformer",
        source_model_id="example/model",
        source_revision="abc123",
        source_license="apache-2.0",
        summary=summary,
    )
    calls = []

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls.append((model_id, kwargs))
            return TinyPipeline()

    pipeline = load_quantized_pipeline_from_artifact(
        tmp_path,
        pipeline_cls=FakePipeline,
        torch_dtype=torch.float32,
        device="cpu",
    )

    restored_layer = pipeline.transformer.transformer_blocks[0]["attn"]["to_q"]
    assert calls == [
        (
            "example/model",
            {
                "revision": "abc123",
                "torch_dtype": torch.float32,
            },
        )
    ]
    assert isinstance(restored_layer, OrbitQuantLinear)
    assert pipeline.orbitquant_manifest.source_model_id == "example/model"
    assert pipeline.orbitquant_artifact_dir == str(tmp_path)
    assert torch.isfinite(restored_layer(torch.randn(1, 2, 8))).all()


def test_load_quantized_pipeline_from_artifact_passes_runtime_override_to_component(tmp_path):
    source_pipeline = TinyPipeline()
    config = OrbitQuantConfig(
        block_size=4,
        target_policy="generic_dit",
        runtime_mode="dequant_bf16",
        activation_kernel_backend="cpu",
    )
    summary = quantize_pipeline(source_pipeline, config, component="transformer")
    save_quantized_pipeline_component(
        source_pipeline,
        tmp_path,
        config=config,
        component="transformer",
        source_model_id="example/model",
        source_revision="abc123",
        source_license="apache-2.0",
        summary=summary,
    )

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            return TinyPipeline()

    pipeline = load_quantized_pipeline_from_artifact(
        tmp_path,
        pipeline_cls=FakePipeline,
        runtime_mode="debug_no_activation_quant",
        activation_kernel_backend="auto",
    )

    restored_layer = pipeline.transformer.transformer_blocks[0]["attn"]["to_q"]
    assert isinstance(restored_layer, OrbitQuantLinear)
    assert restored_layer.runtime_mode == "debug_no_activation_quant"
    assert restored_layer.activation_kernel_backend == "auto"


def test_load_quantized_pipeline_from_artifact_uses_native_pipeline_class(
    monkeypatch,
    tmp_path,
):
    source_pipeline = TinyPipeline()
    config = OrbitQuantConfig(block_size=4, target_policy="generic_dit")
    summary = quantize_pipeline(source_pipeline, config, component="transformer")
    save_quantized_pipeline_component(
        source_pipeline,
        tmp_path,
        config=config,
        component="transformer",
        source_model_id="black-forest-labs/FLUX.1-schnell",
        source_revision="abc123",
        source_license="apache-2.0",
        summary=summary,
    )
    calls = []

    class FakeFluxPipeline:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls.append(("flux", model_id, kwargs))
            return TinyPipeline()

    class FakeDiffusionPipeline:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls.append(("generic", model_id, kwargs))
            return TinyPipeline()

    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(
            FluxPipeline=FakeFluxPipeline,
            DiffusionPipeline=FakeDiffusionPipeline,
        ),
    )

    pipeline = load_quantized_pipeline_from_artifact(tmp_path)

    assert calls == [("flux", "black-forest-labs/FLUX.1-schnell", {"revision": "abc123"})]
    assert isinstance(pipeline.transformer.transformer_blocks[0]["attn"]["to_q"], OrbitQuantLinear)


def test_load_quantized_pipeline_from_artifact_falls_back_to_generic_pipeline(
    monkeypatch,
    tmp_path,
):
    source_pipeline = TinyPipeline()
    config = OrbitQuantConfig(block_size=4, target_policy="generic_dit")
    summary = quantize_pipeline(source_pipeline, config, component="transformer")
    save_quantized_pipeline_component(
        source_pipeline,
        tmp_path,
        config=config,
        component="transformer",
        source_model_id="black-forest-labs/FLUX.1-schnell",
        source_revision="abc123",
        source_license="apache-2.0",
        summary=summary,
    )
    calls = []

    class FakeDiffusionPipeline:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls.append((model_id, kwargs))
            return TinyPipeline()

    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(DiffusionPipeline=FakeDiffusionPipeline),
    )

    pipeline = load_quantized_pipeline_from_artifact(tmp_path)

    assert calls == [("black-forest-labs/FLUX.1-schnell", {"revision": "abc123"})]
    assert isinstance(pipeline.transformer.transformer_blocks[0]["attn"]["to_q"], OrbitQuantLinear)


def test_load_quantized_pipeline_from_artifact_respects_explicit_revision(tmp_path):
    source_pipeline = TinyPipeline()
    config = OrbitQuantConfig(block_size=4, target_policy="generic_dit")
    summary = quantize_pipeline(source_pipeline, config, component="transformer")
    save_quantized_pipeline_component(
        source_pipeline,
        tmp_path,
        config=config,
        component="transformer",
        source_model_id="example/model",
        source_revision="abc123",
        source_license="apache-2.0",
        summary=summary,
    )
    calls = []

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls.append((model_id, kwargs))
            return TinyPipeline()

    load_quantized_pipeline_from_artifact(
        tmp_path,
        pipeline_cls=FakePipeline,
        revision="main",
    )

    assert calls == [("example/model", {"revision": "main"})]


def test_load_quantized_pipeline_from_artifact_rejects_component_mismatch(tmp_path):
    source_pipeline = TinyPipeline()
    config = OrbitQuantConfig(block_size=4, target_policy="generic_dit")
    summary = quantize_pipeline(source_pipeline, config, component="denoiser")
    save_quantized_pipeline_component(
        source_pipeline,
        tmp_path,
        config=config,
        component="denoiser",
        source_model_id="example/model",
        source_revision="abc123",
        source_license="apache-2.0",
        summary=summary,
    )

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            raise AssertionError("component mismatch should fail before source load")

    with pytest.raises(ValueError, match="component mismatch"):
        load_quantized_pipeline_from_artifact(
            tmp_path,
            pipeline_cls=FakePipeline,
            component="transformer",
        )


def test_load_quantized_pipeline_component_rejects_component_mismatch(tmp_path):
    source_pipeline = TinyPipeline()
    config = OrbitQuantConfig(block_size=4, target_policy="generic_dit")
    summary = quantize_pipeline(source_pipeline, config, component="denoiser")
    save_quantized_pipeline_component(
        source_pipeline,
        tmp_path,
        config=config,
        component="denoiser",
        source_model_id="example/model",
        source_revision="abc123",
        source_license="apache-2.0",
        summary=summary,
    )

    restored_pipeline = TinyPipeline()
    try:
        load_quantized_pipeline_component(
            restored_pipeline, tmp_path, component="transformer"
        )
    except ValueError as exc:
        assert "component mismatch" in str(exc)
        assert "denoiser" in str(exc)
        assert "transformer" in str(exc)
    else:
        raise AssertionError("component mismatch was accepted")


def test_quantize_pipeline_fails_loud_for_missing_component():
    pipeline = TinyPipeline()
    config = OrbitQuantConfig(block_size=4)

    try:
        quantize_pipeline(pipeline, config, component="unet")
    except ValueError as exc:
        assert "pipeline has no component" in str(exc)
        assert "unet" in str(exc)
    else:
        raise AssertionError("quantize_pipeline accepted a missing pipeline component")


def test_load_quantized_pipeline_component_fails_loud_for_missing_component(tmp_path):
    pipeline = TinyPipeline()

    try:
        load_quantized_pipeline_component(pipeline, tmp_path, component="unet")
    except ValueError as exc:
        assert "pipeline has no component" in str(exc)
        assert "unet" in str(exc)
    else:
        raise AssertionError("load_quantized_pipeline_component accepted a missing component")
