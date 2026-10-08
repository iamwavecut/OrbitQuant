from types import SimpleNamespace

import numpy as np
import pytest
import torch

from orbitquant.runtime.kandinsky6_video import (
    install_kandinsky6_video_io,
    install_kandinsky6_video_norm,
)

vae_module = pytest.importorskip("diffusers.models.autoencoders.autoencoder_kl_hunyuan_video")
from diffusers.video_processor import VideoProcessor  # noqa: E402


def _pipe():
    vae = torch.nn.Sequential(vae_module.HunyuanVideoCausalConv3d(3, 3))
    return SimpleNamespace(vae=vae, video_processor=VideoProcessor())


def test_video_installation_preserves_cpu_gradients_and_restores():
    pipe = _pipe()
    conv = pipe.vae[0]
    original = conv.forward
    x = torch.randn(1, 3, 5, 7, 9, requires_grad=True)
    expected = conv(x)
    expected_grad = torch.autograd.grad(expected.sum(), x)[0]
    report = install_kandinsky6_video_io(pipe)
    assert report.padding_modules == ["0"] and report.rgb_conversion
    assert install_kandinsky6_video_io(pipe) is report
    actual = conv(x)
    assert torch.equal(expected, actual)
    assert torch.equal(expected_grad, torch.autograd.grad(actual.sum(), x)[0])
    report.restore()
    report.restore()
    assert conv.forward == original


def test_video_installation_preserves_custom_forward_and_processor():
    pipe = _pipe()
    original_conv = pipe.vae[0].forward

    def custom_conv(x):
        return original_conv(x)

    pipe.vae[0].forward = custom_conv

    def custom_processor(video, output_type="np", **kwargs):
        return video

    pipe.video_processor.postprocess_video = custom_processor
    report = install_kandinsky6_video_io(pipe)
    assert not report.padding_modules and not report.rgb_conversion
    assert pipe.vae[0].forward is custom_conv
    assert pipe.video_processor.postprocess_video is custom_processor


def test_video_norm_preserves_cpu_forward_and_gradients():
    block = vae_module.HunyuanVideoResnetBlockCausal3D(32, 32).eval()
    pipe = SimpleNamespace(vae=torch.nn.Sequential(block))
    x = torch.randn(1, 32, 3, 5, 7, requires_grad=True)
    expected = block(x)
    expected_grad = torch.autograd.grad(expected.sum(), x)[0]
    original = block.forward
    report = install_kandinsky6_video_norm(pipe)
    assert report.normalization_modules == ["0"]
    assert install_kandinsky6_video_norm(pipe) is report
    actual = block(x)
    assert torch.equal(actual, expected)
    assert torch.equal(torch.autograd.grad(actual.sum(), x)[0], expected_grad)
    report.restore()
    assert block.forward == original


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_video_group_norm_matches_eager_affine_and_silu_rounding():
    from orbitquant.kernels.triton_video import group_norm_silu

    with torch.inference_mode():
        torch.manual_seed(91)
        for shape in [(1, 128, 5, 67, 79), (2, 64, 7, 31, 33)]:
            x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
            norm = torch.nn.GroupNorm(32, shape[1], eps=1e-6).to(device="cuda", dtype=x.dtype)
            norm.weight.normal_(1, 0.3)
            norm.bias.normal_(0, 0.2)
            expected = torch.nn.functional.silu(norm(x))
            actual = group_norm_silu(x, norm)
            relative = (actual.float() - expected.float()).norm() / expected.float().norm()
            assert torch.isfinite(actual).all() and relative < 0.0005
            # A hook must execute, and its changed output must reach SiLU.
            handle = norm.register_forward_hook(lambda module, args, output: output + 1)
            expected = torch.nn.functional.silu(norm(x))
            assert torch.equal(group_norm_silu(x, norm), expected)
            handle.remove()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_video_padding_and_rgb_pixels_match_reference_on_cuda(dtype):
    from orbitquant.kernels.triton_video import replicate_video_padding

    pipe = _pipe()
    processor = pipe.video_processor
    original = processor.postprocess_video
    report = install_kandinsky6_video_io(pipe)
    with torch.inference_mode():
        x = torch.randn(2, 3, 5, 17, 29, device="cuda", dtype=dtype) * 2
        for padding in [(1, 2, 3, 0, 4, 1), (0, 0, 0, 0, 0, 0)]:
            assert torch.equal(
                replicate_video_padding(x, padding),
                torch.nn.functional.pad(x, padding, mode="replicate"),
            )
        expected = original(x, "pil")
        actual = processor.postprocess_video(x, "pil")
        assert all(
            np.array_equal(np.asarray(a), np.asarray(b))
            for aa, bb in zip(expected, actual, strict=True)
            for a, b in zip(aa, bb, strict=True)
        )
        for mode in ("np", "pt"):
            expected, actual = original(x, mode), processor.postprocess_video(x, mode)
            assert (
                np.array_equal(expected, actual) if mode == "np" else torch.equal(expected, actual)
            )
        # A denormalization override must retain the original implementation.
        expected = original(x, "pil", do_denormalize=[False] * 5)
        actual = processor.postprocess_video(x, "pil", do_denormalize=[False] * 5)
        assert np.array_equal(np.asarray(expected), np.asarray(actual))
    report.restore()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_rgb_conversion_preserves_all_finite_bfloat16_rounding_boundaries():
    pipe = _pipe()
    original = pipe.video_processor.postprocess_video
    install_kandinsky6_video_io(pipe)
    with torch.inference_mode():
        bits = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16)
        values = bits.view(torch.bfloat16)
        values = values[torch.isfinite(values)].float().clamp(-3, 3).bfloat16().cuda()
        x = values[None, None, None, None, :].expand(1, 3, 1, 1, -1).contiguous()
        expected = original(x, "pil")[0][0]
        actual = pipe.video_processor.postprocess_video(x, "pil")[0][0]
        assert np.array_equal(np.asarray(expected), np.asarray(actual))
