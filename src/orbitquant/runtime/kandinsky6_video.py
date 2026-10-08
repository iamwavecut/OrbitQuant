"""Exact padding and PIL conversion for Kandinsky 6 video inference."""

from __future__ import annotations

import types
from dataclasses import dataclass, field
from typing import Any

import torch


def _causal_conv_forward(self, hidden_states):
    if (
        hidden_states.device.type == "cuda"
        and not torch.is_grad_enabled()
        and self.pad_mode == "replicate"
    ):
        from orbitquant.kernels.triton_video import replicate_video_padding

        padded = replicate_video_padding(hidden_states, self.time_causal_padding)
    else:
        padded = torch.nn.functional.pad(
            hidden_states, self.time_causal_padding, mode=self.pad_mode
        )
    return self.conv(padded)


def _postprocessor(original):
    def postprocess(self, video, output_type="np", **kwargs):
        if (
            output_type == "pil"
            and not kwargs
            and self.config.do_normalize
            and isinstance(video, torch.Tensor)
            and video.is_cuda
            and video.ndim == 5
            and video.shape[1] == 3
            and video.is_contiguous()
            and 0 < video.numel() < 2**31
            and not torch.is_grad_enabled()
            and video.dtype in (torch.bfloat16, torch.float16, torch.float32)
        ):
            from PIL import Image

            from orbitquant.kernels.triton_video import video_to_rgb8

            arrays = video_to_rgb8(video).cpu().numpy()
            return [[Image.fromarray(frame) for frame in batch] for batch in arrays]
        return original(video, output_type, **kwargs)

    return postprocess


@dataclass
class Kandinsky6VideoInstallation:
    """Reversible video padding and output conversion replacements."""

    padding_modules: list[str] = field(default_factory=list)
    normalization_modules: list[str] = field(default_factory=list)
    rgb_conversion: bool = False
    _entries: list[tuple[Any, str, Any, Any]] = field(default_factory=list, repr=False)
    _pipeline: Any = field(default=None, repr=False)
    _attribute: str = field(default="_orbitquant_kandinsky6_video", repr=False)

    def restore(self):
        for owner, name, original, replacement in reversed(self._entries):
            if getattr(owner, name) is replacement:
                setattr(owner, name, original)
        self._entries.clear()
        if getattr(self._pipeline, self._attribute, None) is self:
            delattr(self._pipeline, self._attribute)


def install_kandinsky6_video_io(pipe) -> Kandinsky6VideoInstallation:
    """Accelerate replicated VAE padding and default PIL output conversion.

    Install before offload hooks. No tile settings, weights, float output modes
    or dtype rounding rules change. Custom forwards and processors stay intact.
    """
    existing = getattr(pipe, "_orbitquant_kandinsky6_video", None)
    if existing is not None:
        return existing
    from diffusers.models.autoencoders.autoencoder_kl_hunyuan_video import HunyuanVideoCausalConv3d
    from diffusers.video_processor import VideoProcessor

    report = Kandinsky6VideoInstallation(_pipeline=pipe)
    for name, module in pipe.vae.named_modules():
        if (
            type(module) is HunyuanVideoCausalConv3d
            and module.pad_mode == "replicate"
            and getattr(module.forward, "__func__", None) is HunyuanVideoCausalConv3d.forward
        ):
            replacement = types.MethodType(_causal_conv_forward, module)
            report._entries.append((module, "forward", module.forward, replacement))
            module.forward = replacement
            report.padding_modules.append(name)
    processor = pipe.video_processor
    if type(processor) is VideoProcessor and not any(
        name in processor.__dict__
        for name in (
            "postprocess_video",
            "postprocess",
            "numpy_to_pil",
            "pt_to_numpy",
            "denormalize",
            "_denormalize_conditionally",
        )
    ):
        replacement = types.MethodType(_postprocessor(processor.postprocess_video), processor)
        report._entries.append(
            (processor, "postprocess_video", processor.postprocess_video, replacement)
        )
        processor.postprocess_video = replacement
        report.rgb_conversion = True
    pipe._orbitquant_kandinsky6_video = report
    return report


def _resnet_forward(original):
    def forward(self, hidden_states):
        if (
            self.training
            or torch.is_grad_enabled()
            or hidden_states.device.type != "cuda"
            or any(
                m._forward_hooks or m._forward_pre_hooks
                for m in (self.norm1, self.norm2, self.nonlinearity)
            )
        ):
            return original(hidden_states)
        from orbitquant.kernels.triton_video import group_norm_silu

        hidden_states = hidden_states.contiguous()
        residual = hidden_states
        hidden_states = self.conv1(group_norm_silu(hidden_states, self.norm1))
        hidden_states = self.conv2(self.dropout(group_norm_silu(hidden_states, self.norm2)))
        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)
        return hidden_states + residual

    return forward


def install_kandinsky6_video_norm(pipe) -> Kandinsky6VideoInstallation:
    """Fuse large decoder GroupNorm and SiLU calls before installing offload hooks.

    Parallel reduction can change rounding. This is separate from exact video
    I/O optimization and must be enabled explicitly. Small tensors, gradients,
    training and custom blocks retain eager execution.
    """
    attribute = "_orbitquant_kandinsky6_video_norm"
    existing = getattr(pipe, attribute, None)
    if existing is not None:
        return existing
    from diffusers.models.autoencoders.autoencoder_kl_hunyuan_video import (
        HunyuanVideoResnetBlockCausal3D,
    )

    report = Kandinsky6VideoInstallation(_pipeline=pipe, _attribute=attribute)
    for name, module in pipe.vae.named_modules():
        if (
            type(module) is HunyuanVideoResnetBlockCausal3D
            and type(module.nonlinearity) is torch.nn.SiLU
            and "forward" not in module.nonlinearity.__dict__
            and getattr(module.forward, "__func__", None) is HunyuanVideoResnetBlockCausal3D.forward
        ):
            replacement = types.MethodType(_resnet_forward(module.forward), module)
            report._entries.append((module, "forward", module.forward, replacement))
            module.forward = replacement
            report.normalization_modules.append(name)
    setattr(pipe, attribute, report)
    return report
