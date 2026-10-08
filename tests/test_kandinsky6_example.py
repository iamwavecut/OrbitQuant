"""Check public runner input validation and offload ordering without downloading weights."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "k6_example", Path(__file__).parents[1] / "examples/kandinsky6_generate.py"
)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


@pytest.mark.parametrize("arguments", [
    ["--width", "0"], ["--height", "321"], ["--frames", "120"],
    ["--allocator-cap-gib", "nan"], ["--allocator-cap-gib", "-1"],
])
def test_invalid_shapes_and_memory_limits_fail_before_model_loading(arguments):
    with pytest.raises(SystemExit):
        example.parse_args(["--prompt", "fox", *arguments])


def test_320p_arguments_are_supported():
    args = example.parse_args(["--prompt", "fox", "--width", "576", "--height", "320"])
    assert (args.width, args.height, args.frames) == (576, 320, 121)


@pytest.mark.parametrize("profile", ["fast", "low-memory", "exact"])
def test_profiles_install_before_offload_and_reclaim_before_decode(monkeypatch, profile):
    import orbitquant.runtime.kandinsky6 as runtime
    import orbitquant.runtime.kandinsky6_video as video

    calls = []
    for module in (runtime, video):
        for name in dir(module):
            if name.startswith("install_kandinsky6_"):
                monkeypatch.setattr(module, name, lambda obj, name=name: calls.append(name))
    vae = SimpleNamespace(
        enable_tiling=lambda **kw: calls.append(("tiles", kw)),
        decode=lambda value: calls.append(("decode", value)) or value,
    )
    pipe = SimpleNamespace(
        transformer=object(), vae=vae,
        enable_model_cpu_offload=lambda: calls.append("offload"),
        maybe_free_model_hooks=lambda: calls.append("reclaim"),
    )
    assert example.configure_pipeline(pipe, profile) is pipe
    assert calls[-1] == "offload"
    assert vae.decode("latent") == "latent"
    if profile == "low-memory":
        assert calls[-2:] == ["reclaim", ("decode", "latent")]
    else:
        assert "reclaim" not in calls
    if profile == "exact":
        assert ("tiles", {}) in calls
        assert "install_kandinsky6_attention" not in calls
        assert "install_kandinsky6_norm_fusion" not in calls
        assert "install_kandinsky6_video_norm" not in calls
