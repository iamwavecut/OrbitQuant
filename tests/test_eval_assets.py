import numpy as np
from PIL import Image

from orbitquant.eval.assets import create_image_comparison_sheet, create_video_contact_sheet


def test_create_image_comparison_sheet_writes_side_by_side_webp(tmp_path):
    original_path = tmp_path / "original.png"
    quantized_path = tmp_path / "orbitquant.png"
    output_path = tmp_path / "comparison.webp"
    Image.new("RGB", (32, 16), "red").save(original_path)
    Image.new("RGB", (32, 16), "blue").save(quantized_path)

    result = create_image_comparison_sheet(
        original_path,
        quantized_path,
        output_path,
        labels=("BF16", "OrbitQuant"),
    )

    assert result == output_path
    assert output_path.is_file()
    with Image.open(output_path) as sheet:
        assert sheet.size[0] == 64
        assert sheet.size[1] > 16


def test_create_video_contact_sheet_samples_fixed_indices(tmp_path):
    frames = [
        np.full((12, 16, 3), fill_value=value, dtype=np.uint8)
        for value in (0, 60, 120, 180)
    ]
    output_path = tmp_path / "wan_contact_sheet.webp"

    result = create_video_contact_sheet(
        frames,
        output_path,
        sample_indices=[0, 2, 3],
        columns=3,
    )

    assert result == output_path
    assert output_path.is_file()
    with Image.open(output_path) as sheet:
        assert sheet.size == (48, 12)


def test_create_video_contact_sheet_accepts_float_frames_from_video_pipelines(tmp_path):
    frames = [
        np.full((12, 16, 3), fill_value=value, dtype=np.float32)
        for value in (0.0, 0.5, 1.0)
    ]
    output_path = tmp_path / "wan_float_contact_sheet.webp"

    result = create_video_contact_sheet(
        frames,
        output_path,
        sample_indices=[0, 1, 2],
        columns=3,
    )

    assert result == output_path
    with Image.open(output_path) as sheet:
        assert sheet.size == (48, 12)
        assert sheet.mode == "RGB"
