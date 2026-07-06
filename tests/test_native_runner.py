from pathlib import Path

from PIL import Image

from orbitquant.eval import get_native_suite
from orbitquant.eval.native_runner import (
    build_pipeline_kwargs,
    output_path_for_suite,
    run_native_generation,
)


class FakeImageOutput:
    def __init__(self):
        self.images = [Image.new("RGB", (16, 16), "red")]


class FakeImagePipeline:
    def __init__(self):
        self.kwargs = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return FakeImageOutput()


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
    assert pipeline.kwargs["height"] == 1024
    assert pipeline.kwargs["width"] == 1024
