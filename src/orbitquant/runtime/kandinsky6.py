"""Opt-in long-sequence attention for Kandinsky 6 OrbitQuant inference.

The processor keeps the checkpoint's projections, normalization and rotary
embeddings. Only eligible video self-attention uses OrbitQuant's INT8 QK /
FP16 PV kernel. Masked, short, training and unsupported calls delegate to the
original Diffusers processor. Installation does not alter checkpoint weights.
"""

from __future__ import annotations

import copy
import types
from dataclasses import dataclass, field
from typing import Any

import torch

from orbitquant.fused.families import int8_attention_allowed

_INSTALLATION = "_orbitquant_kandinsky6_attention"


def _project_qkv(attn, hidden_states, encoder_hidden_states=None):
    from orbitquant.layers import project_shared_input

    if encoder_hidden_states is None:
        return project_shared_input(hidden_states, attn.to_query, attn.to_key, attn.to_value)
    query = attn.to_query(hidden_states)
    key, value = project_shared_input(encoder_hidden_states, attn.to_key, attn.to_value)
    return query, key, value


def _projection_attention(
    self,
    attn,
    hidden_states,
    encoder_hidden_states=None,
    rotary_emb=None,
    rotary_emb_kv=None,
    attn_mask=None,
):
    from diffusers.models.transformers.transformer_kandinsky6 import (
        apply_rotary,
        dispatch_attention_fn,
    )

    query, key, value = (
        x.unflatten(-1, (attn.num_heads, -1))
        for x in _project_qkv(attn, hidden_states, encoder_hidden_states)
    )
    query, key = attn.query_norm(query), attn.key_norm(key)
    rotary = getattr(self, "_pointwise_rotary", apply_rotary)
    if rotary_emb is not None:
        query = rotary(query, rotary_emb).to(query.dtype)
    if rotary_emb_kv is not None:
        key = rotary(key, rotary_emb_kv).to(key.dtype)
    output = dispatch_attention_fn(
        query,
        key,
        value,
        attn_mask=attn_mask,
        backend=self._attention_backend,
        parallel_config=self._parallel_config,
    )
    return attn.out_layer(output.flatten(-2, -1))


class Kandinsky6Int8AttnProcessor:
    """Preserve K6 processor semantics around the accelerated attention call."""

    def __init__(self, original: Any, *, min_tokens: int = 4096):
        self.original = original
        self.min_tokens = min_tokens
        self._attention_backend = getattr(original, "_attention_backend", None)
        self._parallel_config = getattr(original, "_parallel_config", None)

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        rotary_emb=None,
        rotary_emb_kv=None,
        attn_mask=None,
    ):
        eligible = (
            not attn.training
            and not torch.is_grad_enabled()
            and hidden_states.ndim == 3
            and hidden_states.shape[0] == 1
            and hidden_states.shape[1] >= self.min_tokens
            and hidden_states.device.type == "cuda"
            and hidden_states.dtype in (torch.float16, torch.bfloat16)
            and encoder_hidden_states is None
            and attn_mask is None
            and self._attention_backend is None
            and self._parallel_config is None
            and attn.to_query.out_features // attn.num_heads == 64
        )
        if eligible:
            eligible = torch.cuda.get_device_capability(hidden_states.device)[0] >= 8
        if not eligible:
            fallback = self.original
            if (
                fallback._attention_backend != self._attention_backend
                or fallback._parallel_config is not self._parallel_config
            ):
                fallback = copy.copy(fallback)
                fallback._attention_backend = self._attention_backend
                fallback._parallel_config = self._parallel_config
            return fallback(
                attn,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                rotary_emb=rotary_emb,
                rotary_emb_kv=rotary_emb_kv,
                attn_mask=attn_mask,
            )

        from diffusers.models.transformers.transformer_kandinsky6 import apply_rotary

        from orbitquant.kernels.triton_attention import int8_attention_with_triton

        query, key, value = (
            x.unflatten(-1, (attn.num_heads, -1))
            for x in (
                _project_qkv(attn, hidden_states)
                if getattr(self, "_shared_projections", False)
                else tuple(p(hidden_states) for p in (attn.to_query, attn.to_key, attn.to_value))
            )
        )
        query, key = attn.query_norm(query), attn.key_norm(key)
        apply_rotary = getattr(self, "_pointwise_rotary", apply_rotary)
        if rotary_emb is not None:
            query = apply_rotary(query, rotary_emb).to(query.dtype)
        if rotary_emb_kv is not None:
            key = apply_rotary(key, rotary_emb_kv).to(key.dtype)
        output = int8_attention_with_triton(query, key, value)
        return attn.out_layer(output.flatten(-2, -1))


@dataclass
class Kandinsky6AttentionInstallation:
    """Installation report and reversible processor replacement handle."""

    min_tokens: int
    installed: list[str] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)
    _entries: list[tuple[Any, Any, Any]] = field(default_factory=list, repr=False)
    _model: Any = field(default=None, repr=False)

    def restore(self) -> None:
        """Restore our processors, preserving subsequent user replacements."""
        for module, original, replacement in self._entries:
            if module.processor is replacement:
                module.set_processor(original)
        self._entries.clear()
        if getattr(self._model, _INSTALLATION, None) is self:
            delattr(self._model, _INSTALLATION)


def install_kandinsky6_attention(
    model, *, min_tokens: int = 4096
) -> Kandinsky6AttentionInstallation:
    """Accelerate eligible video self-attention; return a report with ``restore``.

    Call after loading checkpoint weights. Blocks with uneven Q/K RMSNorm
    scales retain their original processor to avoid losing small channels in
    INT8 quantization. Custom attention processors/backends remain untouched.
    Repeated installation with the same options returns the existing handle.
    """
    if not isinstance(min_tokens, int) or isinstance(min_tokens, bool) or min_tokens < 1:
        raise ValueError("min_tokens must be a positive integer")
    existing = getattr(model, _INSTALLATION, None)
    if existing is not None:
        if existing.min_tokens != min_tokens:
            raise ValueError("restore the existing installation before changing min_tokens")
        return existing
    if not hasattr(model, "visual_transformer_blocks"):
        raise ValueError("expected a Kandinsky6 transformer with visual_transformer_blocks")

    from diffusers.models.transformers.transformer_kandinsky6 import Kandinsky6AttnProcessor

    report = Kandinsky6AttentionInstallation(min_tokens=min_tokens, _model=model)
    for index, block in enumerate(model.visual_transformer_blocks):
        name = f"visual_transformer_blocks.{index}.video_dec_block.self_attention"
        attn = block.video_dec_block.self_attention
        original = attn.processor
        if type(original) is not Kandinsky6AttnProcessor:
            report.skipped[name] = "custom processor"
        elif original._attention_backend is not None or original._parallel_config is not None:
            report.skipped[name] = "custom attention backend or parallel configuration"
        elif not int8_attention_allowed(attn.query_norm.weight, attn.key_norm.weight):
            report.skipped[name] = "Q/K normalization scale spread exceeds INT8 guard"
        else:
            replacement = Kandinsky6Int8AttnProcessor(original, min_tokens=min_tokens)
            attn.set_processor(replacement)
            report._entries.append((attn, original, replacement))
            report.installed.append(name)
    setattr(model, _INSTALLATION, report)
    return report


def _lazy_pointwise(name, original):
    def helper(*args, **kwargs):
        if not torch.is_grad_enabled() and args[0].device.type == "cuda":
            from orbitquant.kernels import triton_kandinsky6

            return getattr(triton_kandinsky6, name)(*args, **kwargs)
        return original(*args, **kwargs)

    return helper


def _pointwise_function(function):
    """Give one upstream function private helper bindings; keep its Python code."""
    namespace = function.__globals__.copy()
    for original, replacement in (
        ("apply_gate_sum", "gate_sum"),
        ("apply_scale_shift", "scale_shift"),
        ("apply_rotary", "rotary"),
    ):
        if original in namespace:
            namespace[original] = _lazy_pointwise(replacement, namespace[original])
    clone = types.FunctionType(
        function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__
    )
    clone.__kwdefaults__ = function.__kwdefaults__
    clone.__annotations__ = function.__annotations__.copy()
    return clone


class _PointwiseProcessor:
    def __init__(self, original, call):
        self.__dict__.update(original.__dict__)
        self._call = types.MethodType(call, self)

    def __call__(self, *args, **kwargs):
        if getattr(self, "_shared_projections", False):
            return _projection_attention(self, *args, **kwargs)
        return self._call(*args, **kwargs)


@dataclass
class Kandinsky6PointwiseInstallation:
    """Reversible helper bindings confined to one model's modules."""

    blocks: list[str] = field(default_factory=list)
    processors: list[str] = field(default_factory=list)
    _forwards: list[tuple[Any, Any, Any]] = field(default_factory=list, repr=False)
    _processors: list[tuple[Any, Any, Any]] = field(default_factory=list, repr=False)
    _attention: list[Any] = field(default_factory=list, repr=False)
    _model: Any = field(default=None, repr=False)

    def restore(self):
        for module, original, replacement in self._forwards:
            if module.forward is replacement:
                module.forward = original
        for module, original, replacement in self._processors:
            if module.processor is replacement:
                module.set_processor(original)
        for processor in self._attention:
            if hasattr(processor, "_pointwise_rotary"):
                del processor._pointwise_rotary
        self._forwards.clear()
        self._processors.clear()
        self._attention.clear()
        if getattr(self._model, "_orbitquant_kandinsky6_pointwise", None) is self:
            del self._model._orbitquant_kandinsky6_pointwise


def install_kandinsky6_pointwise(model) -> Kandinsky6PointwiseInstallation:
    """Fuse FP32 helper arithmetic without changing upstream module globals.

    Install after ``install_kandinsky6_attention`` and before CPU offload hooks.
    CPU and gradient-enabled calls retain eager arithmetic. Unsupported tensor
    layouts also fall back. No weights or public checkpoint format change.
    """
    existing = getattr(model, "_orbitquant_kandinsky6_pointwise", None)
    if existing is not None:
        return existing
    if not hasattr(model, "visual_transformer_blocks"):
        raise ValueError("expected a Kandinsky6 transformer with visual_transformer_blocks")
    from diffusers.models.transformers import transformer_kandinsky6 as k6

    report = Kandinsky6PointwiseInstallation(_model=model)
    functions = {}
    processor_call = _pointwise_function(k6.Kandinsky6AttnProcessor.__call__)
    for name, module in model.named_modules():
        function = getattr(module.forward, "__func__", None)
        if (
            function is not None
            and function.__module__ == k6.__name__
            and {"apply_gate_sum", "apply_scale_shift"}.intersection(function.__code__.co_names)
        ):
            if function not in functions:
                functions[function] = _pointwise_function(function)
            replacement = types.MethodType(functions[function], module)
            report._forwards.append((module, module.forward, replacement))
            module.forward = replacement
            report.blocks.append(name)
        processor = getattr(module, "processor", None)
        if isinstance(processor, Kandinsky6Int8AttnProcessor):
            processor._pointwise_rotary = _lazy_pointwise("rotary", k6.apply_rotary)
            report._attention.append(processor)
            report.processors.append(name)
        elif type(processor) is k6.Kandinsky6AttnProcessor:
            replacement = _PointwiseProcessor(processor, processor_call)
            replacement._pointwise_rotary = _lazy_pointwise("rotary", k6.apply_rotary)
            module.set_processor(replacement)
            report._processors.append((module, processor, replacement))
            report.processors.append(name)
    model._orbitquant_kandinsky6_pointwise = report
    return report


@dataclass
class Kandinsky6ProjectionInstallation:
    """Scoped Q/K/V reuse without checkpoint or global cache changes."""

    installed: list[str] = field(default_factory=list)
    _processors: list[tuple[Any, Any, Any]] = field(default_factory=list, repr=False)
    _flags: list[Any] = field(default_factory=list, repr=False)
    _model: Any = field(default=None, repr=False)

    def restore(self):
        for module, original, replacement in self._processors:
            if module.processor is replacement:
                module.set_processor(original)
        for processor in self._flags:
            processor._shared_projections = False
        self._processors.clear()
        self._flags.clear()
        if getattr(self._model, "_orbitquant_kandinsky6_projections", None) is self:
            del self._model._orbitquant_kandinsky6_projections


def install_kandinsky6_projection_reuse(model) -> Kandinsky6ProjectionInstallation:
    """Prepare each attention input once for Q/K/V or cross-attention K/V.

    Install after attention and pointwise optimizations. Restore in reverse order.
    Custom processors remain untouched. The helper supports inference tensors and
    clears the prepared input as soon as the consecutive projections return.
    """
    existing = getattr(model, "_orbitquant_kandinsky6_projections", None)
    if existing is not None:
        return existing
    from diffusers.models.transformers.transformer_kandinsky6 import Kandinsky6AttnProcessor

    report = Kandinsky6ProjectionInstallation(_model=model)
    for name, module in model.named_modules():
        processor = getattr(module, "processor", None)
        if isinstance(processor, (Kandinsky6Int8AttnProcessor, _PointwiseProcessor)):
            processor._shared_projections = True
            report._flags.append(processor)
        elif type(processor) is Kandinsky6AttnProcessor:
            replacement = _PointwiseProcessor(processor, _projection_attention)
            replacement._shared_projections = True
            module.set_processor(replacement)
            report._processors.append((module, processor, replacement))
        else:
            continue
        report.installed.append(name)
    model._orbitquant_kandinsky6_projections = report
    return report


@dataclass
class Kandinsky6NormInstallation:
    """Reversible block forwards with fused normalization and modulation."""

    installed: list[str] = field(default_factory=list)
    _entries: list[tuple[Any, Any, Any]] = field(default_factory=list, repr=False)
    _model: Any = field(default=None, repr=False)
    _attribute: str = field(default="_orbitquant_kandinsky6_norm", repr=False)

    def restore(self):
        for module, original, replacement in self._entries:
            if module.forward is replacement:
                module.forward = original
        self._entries.clear()
        if getattr(self._model, self._attribute, None) is self:
            delattr(self._model, self._attribute)


def install_kandinsky6_norm_fusion(model) -> Kandinsky6NormInstallation:
    """Fuse LayerNorm and modulation without changing weights or step count.

    Install after other K6 optimizations and before offload hooks. This is an
    opt-in numerical change: the reduction can differ at rounding boundaries.
    CPU and gradient-enabled calls use the original arithmetic. Remove offload
    hooks before restoring installations, in reverse installation order.
    """
    existing = getattr(model, "_orbitquant_kandinsky6_norm", None)
    if existing is not None:
        return existing
    from diffusers.models.transformers import transformer_kandinsky6 as k6

    from orbitquant.runtime import kandinsky6_blocks as blocks

    implementations = {
        k6.Kandinsky6TransformerEncoderBlock: blocks.encoder_forward,
        k6.Kandinsky6FusedTransformerDecoderBlock: blocks.decoder_forward,
        k6.Kandinsky6OutLayer: blocks.video_out_forward,
        k6.Kandinsky6OutLayerAudio: blocks.audio_out_forward,
    }
    report = Kandinsky6NormInstallation(_model=model)
    for name, module in model.named_modules():
        function = implementations.get(type(module))
        if function is None:
            continue
        forward = getattr(module.forward, "__func__", None)
        if forward is None or forward.__module__ != k6.__name__:
            continue
        replacement = types.MethodType(function, module)
        report._entries.append((module, module.forward, replacement))
        module.forward = replacement
        report.installed.append(name)
    model._orbitquant_kandinsky6_norm = report
    return report


def _qk_norm_forward(original):
    def forward(self, x):
        if (
            not torch.is_grad_enabled()
            and x.device.type == "cuda"
            and x.dtype == torch.bfloat16
            and x.ndim == 4
            and x.shape[0] == 1
            and x.shape[-1] == 64
            and x.numel() > 0
            and x.is_contiguous()
            and self.weight is not None
            and self.weight.device == x.device
            and self.weight.dtype == x.dtype
            and self.weight.is_contiguous()
        ):
            from orbitquant.kernels.triton_kandinsky6 import qk_rms_norm

            # Torch 2.12 uses opmath epsilon for RMSNorm(None), including BF16 input.
            eps = torch.finfo(torch.float32).eps if self.eps is None else self.eps
            return qk_rms_norm(x, self.weight, eps)
        return original(x)

    return forward


def install_kandinsky6_qk_norm(model) -> Kandinsky6NormInstallation:
    """Accelerate BF16 Q/K RMSNorm on the validated Torch 2.12 runtime.

    Install before offload hooks. Module hooks remain active. Other Torch
    versions keep their native implementation, including their epsilon default.
    CPU, gradients, custom forwards and other head widths retain eager calls.
    """
    attribute = "_orbitquant_kandinsky6_qk_norm"
    existing = getattr(model, attribute, None)
    if existing is not None:
        return existing
    from packaging.version import Version

    report = Kandinsky6NormInstallation(_model=model, _attribute=attribute)
    if Version(torch.__version__.split("+", 1)[0]).release[:2] == (2, 12):
        for name, module in model.named_modules():
            if (
                name.rsplit(".", 1)[-1] in ("query_norm", "key_norm")
                and type(module) is torch.nn.RMSNorm
                and tuple(module.normalized_shape) == (64,)
                and getattr(module.forward, "__func__", None) is torch.nn.RMSNorm.forward
            ):
                replacement = types.MethodType(_qk_norm_forward(module.forward), module)
                report._entries.append((module, module.forward, replacement))
                module.forward = replacement
                report.installed.append(name)
    setattr(model, attribute, report)
    return report
