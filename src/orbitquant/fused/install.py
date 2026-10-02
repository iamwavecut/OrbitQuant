"""Turn a loaded OrbitQuant transformer into its fused form, or prepare one to load as fused.

``fuse(model)`` regroups the projections of every block into ``block.oq_fused`` (a
``ModuleDict`` of groups), replaces the source projections with ``nn.Identity`` and swaps in the
family's block forward. ``model.oq_fused_layout`` then describes the result; saved next to the
weights it lets ``prepare_skeleton`` rebuild the same empty groups before a checkpoint loads,
so a fused checkpoint never holds the unfused projections.

A group's type and row segments follow from its source projections: OrbitQuant projections form
a packed group (segmented by weight width), projections the artifact kept dense become INT8 rows
or stay BF16 as the layout's ``dense`` mode says. The quantization config already determines the
projection types and widths, so the skeleton rebuilds the same groups from it. The layout's
``activation_bits`` maps group names to an activation width other than the config's (8-bit
activations for groups whose inputs 4-bit codes cannot carry).
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import torch
from torch import nn

from orbitquant.fused.families import adapter_for_class, adapter_for_family
from orbitquant.fused.groups import GROUP_TYPES, Bf16Group, Int8Group, PackedGroup, _resolve
from orbitquant.fused.runtime import FusedRuntime
from orbitquant.layers import OrbitQuantLinear

LAYOUT_VERSION = 1


def _set(root: nn.Module, path: str, module: nn.Module) -> None:
    parent_path, _, name = path.rpartition(".")
    parent = _resolve(root, parent_path) if parent_path else root
    if name.isdigit():
        parent[int(name)] = module
    else:
        setattr(parent, name, module)


DENSE_MODES = ("int8", "bf16")


def _group_kind(block: nn.Module, spec, dense: str) -> str:
    packed = [isinstance(_resolve(block, s.path), OrbitQuantLinear) for s in spec.sources]
    if all(packed):
        return PackedGroup.kind
    if any(packed):
        raise ValueError("a group mixes OrbitQuant and dense projections")
    mode = spec.dense or dense
    if mode not in DENSE_MODES:
        raise ValueError(f"unknown dense group mode {mode!r}")
    return Int8Group.kind if mode == "int8" else Bf16Group.kind


def _quant_config(model, config=None):
    """The OrbitQuant config: given, on the model, on its HF quantizer or in its config."""
    if config is None:
        config = getattr(model, "quantization_config", None)
    if config is None:
        config = getattr(getattr(model, "hf_quantizer", None), "quantization_config", None)
    if config is None:
        model_config = getattr(model, "config", None)
        payload = getattr(model_config, "quantization_config", None)
        if payload is None and isinstance(model_config, dict):
            payload = model_config.get("quantization_config")
        if isinstance(payload, dict):
            from orbitquant.config import OrbitQuantConfig

            config = OrbitQuantConfig.from_dict(payload)
        else:
            config = payload
    if config is None:
        raise ValueError("the model carries no OrbitQuant quantization_config")
    return config


def _attach(model, block, adapter, kind, group_modules, runtime):
    specs = adapter.groups(kind, block)
    for spec in specs.values():
        for source in spec.sources:
            _set(block, source.path, nn.Identity())
    block.oq_fused = group_modules
    block.__dict__["oq_runtime"] = runtime
    block.oq_kind = kind
    block.forward = types.MethodType(adapter.forward(kind), block)


def fuse(
    model: nn.Module,
    *,
    dense: str | None = None,
    attention: str = "int8",
    activation_bits: dict[str, int] | None = None,
    config=None,
) -> dict:
    """Fuse a loaded OrbitQuant transformer in place (weights on the host or the GPU).

    ``dense`` picks INT8 rows or BF16 for projections the artifact kept unquantized (default:
    the family's ``DENSE``, else INT8); ``activation_bits`` maps group names to their activation
    width (default: the family's ``ACTIVATION_BITS``, else the config's); ``config`` is the
    OrbitQuant config when the model does not carry one (``orbitquant-v1`` component
    artifacts)."""
    adapter = adapter_for_class(type(model).__name__)
    dense = dense or getattr(adapter, "DENSE", "int8")
    if dense not in DENSE_MODES:
        raise ValueError(f"unknown dense group mode {dense!r}")
    widths = dict(
        getattr(adapter, "ACTIVATION_BITS", {}) if activation_bits is None else activation_bits
    )
    runtime = FusedRuntime.from_config(_quant_config(model, config), attention=attention)
    for path, kind in adapter.blocks(model):
        block = _resolve(model, path)
        modules = nn.ModuleDict()
        for name, spec in adapter.groups(kind, block).items():
            group_cls = GROUP_TYPES[_group_kind(block, spec, dense)]
            modules[name] = group_cls.from_sources(
                block, list(spec.sources), swiglu=spec.swiglu, activation_bits=widths.get(name)
            )
        prepare = getattr(adapter, "prepare", None)
        if prepare is not None:
            prepare(block, kind)
        _attach(model, block, adapter, kind, modules, runtime)
    layout = {"family": adapter.FAMILY, "version": LAYOUT_VERSION, "dense": dense}
    if widths:
        layout["activation_bits"] = widths
    model.oq_fused_layout = layout
    model.__dict__["oq_runtime"] = runtime
    _install_model(model, adapter)
    return layout


def _install_model(model: nn.Module, adapter) -> None:
    install = getattr(adapter, "install", None)
    if install is not None:
        install(model)


def _empty_group(block, spec, group_kind: str, config, device, activation_bits=None):
    layers = [_resolve(block, source.path) for source in spec.sources]
    first = layers[0]
    bias = getattr(first, "bias", None) is not None
    if group_kind == PackedGroup.kind:
        return PackedGroup.empty(
            PackedGroup.source_segments(layers, spec.sources, swiglu=spec.swiglu),
            first.in_features,
            activation_bits=activation_bits or first.activation_bits,
            codebook_version=config.codebook_version,
            bias=bias,
            device=device,
        )
    rows = 0
    for source, layer in zip(spec.sources, layers, strict=True):
        stop = layer.out_features if source.stop is None else source.stop
        rows += stop - source.start
    return GROUP_TYPES[group_kind].empty(rows, first.in_features, bias=bias, device=device)


def prepare_skeleton(
    model: nn.Module, layout: dict, *, attention: str = "int8", config=None
) -> None:
    """Install empty fused groups (and placeholders for their sources) before a fused
    checkpoint loads. Call after the quantized projections were turned into OrbitQuant
    skeletons; ``finalize`` completes the blocks once the weights are in."""
    if int(layout.get("version", 0)) != LAYOUT_VERSION:
        raise ValueError(f"unsupported fused layout version {layout.get('version')!r}")
    adapter = adapter_for_family(layout["family"])
    config = _quant_config(model, config)
    dense = layout.get("dense", "int8")
    widths = layout.get("activation_bits", {})
    runtime = FusedRuntime.from_config(config, attention=attention)
    for path, kind in adapter.blocks(model):
        block = _resolve(model, path)
        specs = adapter.groups(kind, block)
        reference = _resolve(block, next(iter(specs.values())).sources[0].path)
        tensors = [*reference.buffers(), *reference.parameters()]
        device = tensors[0].device if tensors else torch.device("meta")
        modules = nn.ModuleDict(
            {
                name: _empty_group(
                    block, spec, _group_kind(block, spec, dense), config, device, widths.get(name)
                )
                for name, spec in specs.items()
            }
        )
        _attach(model, block, adapter, kind, modules, runtime)
    model.oq_fused_layout = layout
    model.__dict__["oq_runtime"] = runtime
    _install_model(model, adapter)


def finalize(model: nn.Module) -> None:
    """Derive the per-block settings that depend on loaded weights (e.g. INT8 attention)."""
    layout = model.oq_fused_layout
    adapter = adapter_for_family(layout["family"])
    prepare = getattr(adapter, "prepare", None)
    if prepare is None:
        return
    for path, kind in adapter.blocks(model):
        prepare(_resolve(model, path), kind)


def release(model: nn.Module) -> None:
    """Drop the runtime's device buffers (call before moving the weights off the GPU)."""
    runtime = model.__dict__.get("oq_runtime")
    if runtime is not None:
        runtime.release()


def is_fused_tensor_name(name: str) -> bool:
    return ".oq_fused." in f".{name}"


def save_pretrained(model: nn.Module, directory, **kwargs) -> None:
    """``model.save_pretrained`` for a fused transformer, recording the layout in the
    ``quantization_config`` of ``config.json`` so ``from_pretrained`` loads it fused."""
    layout = model.oq_fused_layout
    config = _quant_config(model)
    config.fused_layout = layout
    model.save_pretrained(directory, **kwargs)
    config_path = Path(directory) / "config.json"
    data = json.loads(config_path.read_text())
    quantization = data.get("quantization_config")
    if quantization is None:
        quantization = config.to_dict()
        data["quantization_config"] = quantization
    if quantization.get("fused_layout") != layout:
        quantization["fused_layout"] = layout
        config_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
