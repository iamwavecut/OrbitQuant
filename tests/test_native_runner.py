import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from orbitquant.eval import get_native_suite
from orbitquant.eval.native_runner import (
    apply_quantization_to_pipeline,
    build_pipeline_kwargs,
    build_quantization_config_for_suite,
    extract_video_frames,
    load_pipeline_for_suite,
    output_path_for_suite,
    parse_bit_setting,
    run_native_generation,
)
from orbitquant.layers import OrbitQuantLinear


class FakeImageOutput:
    def __init__(self):
        self.images = [Image.new("RGB", (16, 16), "red")]


class FakeVideoOutput:
    def __init__(self):
        self.frames = np.zeros((1, 2, 8, 8, 3), dtype=np.uint8)


class FakeImagePipeline:
    def __init__(self):
        self.kwargs = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return FakeImageOutput()


class FakeTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([nn.Linear(4, 4)])


class FakeQuantPipeline:
    def __init__(self):
        self.transformer = FakeTransformer()


def test_build_pipeline_kwargs_uses_flux2_native_resolution_and_seed():
    suite = get_native_suite("flux2-native")
    kwargs = build_pipeline_kwargs(suite, prompt="A sign that says hello", seed=17, device="cpu")

    assert kwargs["prompt"] == "A sign that says hello"
    assert kwargs["height"] == 1024
    assert kwargs["width"] == 1024
    assert kwargs["num_inference_steps"] == 4
    assert kwargs["guidance_scale"] == 1.0
    assert "frames" not in kwargs


def test_build_pipeline_kwargs_uses_wan_native_video_settings():
    suite = get_native_suite("wan-native")
    kwargs = build_pipeline_kwargs(suite, prompt="A slow camera pan", seed=4, device="cpu")

    assert kwargs["height"] == 480
    assert kwargs["width"] == 832
    assert kwargs["num_frames"] == 81
    assert kwargs["num_inference_steps"] == 50
    assert kwargs["guidance_scale"] == 5.0


def test_output_path_for_suite_uses_native_suite_name_and_media_extension(tmp_path):
    image_path = output_path_for_suite(
        tmp_path, suite_name="flux2-native", seed=0, media_type="image"
    )
    video_path = output_path_for_suite(
        Path(tmp_path), suite_name="wan-native", seed=0, media_type="video"
    )

    assert image_path.name == "flux2-native_seed0.png"
    assert video_path.name == "wan-native_seed0.mp4"


def test_output_path_for_suite_includes_quantization_variant(tmp_path):
    image_path = output_path_for_suite(
        tmp_path, suite_name="flux2-native", seed=0, media_type="image", variant="W4A4"
    )

    assert image_path.name == "flux2-native_seed0_W4A4.png"


def test_parse_bit_setting_uses_weight_and_activation_bits():
    assert parse_bit_setting("W4A6") == (4, 6)


def test_build_quantization_config_for_suite_rejects_unsupported_native_bit_setting():
    suite = get_native_suite("flux2-native")

    with pytest.raises(ValueError, match="not listed"):
        build_quantization_config_for_suite(suite, "W8A8")


def test_load_pipeline_for_suite_uses_named_diffusers_pipeline_class(monkeypatch):
    suite = get_native_suite("flux2-native")

    class FakeFlux2KleinPipeline:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            assert model_id == "black-forest-labs/FLUX.2-klein-4B"
            assert kwargs["torch_dtype"] is torch.float32
            return cls()

    class WrongDiffusionPipeline:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            raise AssertionError("suite-specific pipeline should be used")

    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(
            Flux2KleinPipeline=FakeFlux2KleinPipeline,
            DiffusionPipeline=WrongDiffusionPipeline,
        ),
    )

    pipeline = load_pipeline_for_suite(suite, torch_dtype=torch.float32)

    assert isinstance(pipeline, FakeFlux2KleinPipeline)


def test_load_pipeline_for_suite_falls_back_to_diffusion_pipeline(monkeypatch):
    suite = get_native_suite("z-image-native")

    class FakeDiffusionPipeline:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            assert model_id == "Tongyi-MAI/Z-Image-Turbo"
            return cls()

    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(DiffusionPipeline=FakeDiffusionPipeline),
    )

    pipeline = load_pipeline_for_suite(suite, torch_dtype=torch.float32)

    assert isinstance(pipeline, FakeDiffusionPipeline)


def test_apply_quantization_to_pipeline_targets_transformer_component():
    pipeline = FakeQuantPipeline()
    suite = get_native_suite("flux2-native")
    config = build_quantization_config_for_suite(suite, "W4A4")

    summary = apply_quantization_to_pipeline(pipeline, suite, config)

    assert summary.quantized_modules == ["transformer_blocks.0"]
    assert isinstance(pipeline.transformer.transformer_blocks[0], OrbitQuantLinear)


def test_extract_video_frames_accepts_numpy_frame_batches():
    frames = extract_video_frames(FakeVideoOutput())

    assert frames.shape == (2, 8, 8, 3)


def test_run_native_generation_saves_video_contact_sheet_and_metadata(
    monkeypatch, tmp_path
):
    def fake_export_to_video(frames, path):
        Path(path).write_bytes(b"fake mp4")

    monkeypatch.setitem(
        sys.modules,
        "diffusers.utils",
        SimpleNamespace(export_to_video=fake_export_to_video),
    )
    suite = get_native_suite("wan-native")

    result = run_native_generation(
        lambda **kwargs: FakeVideoOutput(),
        suite,
        prompt="A native video",
        seed=7,
        output_dir=tmp_path,
        device="cpu",
        quantization_label="W4A4",
    )

    contact_sheet_path = tmp_path / "wan-native_seed7_W4A4_contact_sheet.webp"
    metadata = json.loads(result.metadata_path.read_text())
    assert result.output_path == tmp_path / "wan-native_seed7_W4A4.mp4"
    assert result.output_path.is_file()
    assert contact_sheet_path.is_file()
    assert metadata["contact_sheet_path"] == str(contact_sheet_path)
    assert result.asset_paths == [contact_sheet_path]
    with Image.open(contact_sheet_path) as sheet:
        assert sheet.size == (32, 8)


def test_run_native_generation_saves_image_and_metadata(tmp_path):
    pipeline = FakeImagePipeline()
    suite = get_native_suite("flux2-native")

    result = run_native_generation(
        pipeline,
        suite,
        prompt="A native image",
        seed=5,
        output_dir=tmp_path,
        device="cpu",
    )

    assert result.output_path.exists()
    assert result.metadata_path.exists()
    metadata = json.loads(result.metadata_path.read_text())
    assert result.metadata["wall_time_seconds"] >= 0.0
    assert metadata["wall_time_seconds"] == result.metadata["wall_time_seconds"]
    assert metadata["peak_vram_bytes"] is None
    assert pipeline.kwargs["height"] == 1024
    assert pipeline.kwargs["width"] == 1024
