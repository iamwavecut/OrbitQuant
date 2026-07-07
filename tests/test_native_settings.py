from orbitquant.eval import get_native_suite


def test_flux2_native_suite_uses_symmetric_megapixel_resolution():
    suite = get_native_suite("flux2-native")

    assert suite.model_id == "black-forest-labs/FLUX.2-klein-4B"
    assert suite.width == 1024
    assert suite.height == 1024
    assert suite.steps == 4
    assert suite.guidance == 1.0
    assert suite.bit_settings == ["W4A4", "W3A3", "W2A4", "W2A3"]


def test_flux1_schnell_native_suite_uses_paper_image_settings():
    suite = get_native_suite("flux1-schnell-native")

    assert suite.model_id == "black-forest-labs/FLUX.1-schnell"
    assert suite.width == 1024
    assert suite.height == 1024
    assert suite.steps == 4
    assert suite.guidance == 0.0
    assert suite.bit_settings == ["W4A4", "W3A3", "W2A4", "W2A3"]
    assert suite.metric == "geneval"


def test_z_image_native_suite_uses_paper_image_settings():
    suite = get_native_suite("z-image-native")

    assert suite.model_id == "Tongyi-MAI/Z-Image-Turbo"
    assert suite.width == 1024
    assert suite.height == 1024
    assert suite.steps == 10
    assert suite.guidance == 0.0
    assert suite.bit_settings == ["W4A4", "W3A3", "W2A4", "W2A3"]
    assert suite.metric == "geneval"


def test_wan_native_suite_uses_paper_video_settings():
    suite = get_native_suite("wan-native")

    assert suite.model_id == "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    assert suite.width == 832
    assert suite.height == 480
    assert suite.frames == 81
    assert suite.export_fps == 16
    assert suite.steps == 50
    assert suite.guidance == 5.0
    assert suite.bit_settings == ["W4A6", "W4A4"]
