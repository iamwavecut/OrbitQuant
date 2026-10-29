from __future__ import annotations

import logging
import os
import threading
import weakref
from collections.abc import Callable
from itertools import count

import torch
from torch import nn
from torch.nn import functional as F

from orbitquant.codebooks import get_codebook
from orbitquant.codebooks.lloyd_max import codebook_retained_energy
from orbitquant.config import OrbitQuantConfig
from orbitquant.kernels import quantize_activations_kernel, select_backend
from orbitquant.linear_adapters import canonical_linear_weight, linear_module_spec
from orbitquant.packing import pack_lowbit, unpack_lowbit
from orbitquant.rotations import RPBHRotation, get_pre_rotation, get_rpbh_rotation
from orbitquant.streaming import accelerate_hook_offloads, iter_aligned_row_tiles

_PACKED_MATMUL_PROBE_MISSING = object()

# Measured crossover on RTX 4090-class hardware: below this row count the
# per-forward INT8 weight decode dominates torch._int_mm, so the fused Triton
# kernel that reads packed nibbles directly wins.
# Measured crossover vs the int_mm path with the tuned tile table (RTX 4090,
# 2026-07): the fused kernel stays ahead through 2048 rows.
_W4A4_FUSED_MAX_ROWS = 2048

logger = logging.getLogger("orbitquant")

# One warning per distinct degraded layer shape per process; auto_fused
# degradations repeat on every forward otherwise.
_AUTO_FUSED_DEGRADATION_WARNED: set[tuple[int, int, int, str]] = set()

# torch.compile support: the runtime dispatch below is Python-heavy, so the
# whole quantized forward is exposed to Dynamo as one opaque custom op. The
# registry maps a stable integer handle (burned into the traced graph) back to
# the live module without owning its lifetime. A strong registry prevents
# finalizers from ever running and retains discarded model weights.
_COMPILE_REGISTRY: weakref.WeakValueDictionary[int, object] = weakref.WeakValueDictionary()
_COMPILE_HANDLE_IDS = count(1)


_SHARED_ACTIVATION_CACHE = threading.local()


def _clear_shared_activation_cache() -> None:
    """Drop the current thread's last prepared activation."""

    _SHARED_ACTIVATION_CACHE.entry = None
    _SHARED_ACTIVATION_CACHE.pre_rotation_entry = None
    _SHARED_ACTIVATION_CACHE.outlier_entry = None


def _get_or_create_shared_activation_quantization(
    x: torch.Tensor,
    signature: tuple[object, ...],
    factory: Callable[[], tuple[torch.Tensor, torch.Tensor]],
    *,
    slot: str = "entry",
) -> tuple[tuple[torch.Tensor, torch.Tensor], bool]:
    """Reuse an unchanged activation across consecutive quantized projections.

    Attention Q/K/V (and gate) and SwiGLU gate/up projections receive the same tensor object
    and use the same RPBH/codebook transform. Keeping only the most recent prepared activation
    makes that common pattern cheap without an unbounded cache or pointer-based aliasing.
    """

    if torch.is_grad_enabled():
        setattr(_SHARED_ACTIVATION_CACHE, slot, None)
        return factory(), False
    try:
        version = x._version
    except RuntimeError:
        # Inference tensors hide their version counter, so an in-place mutation between two
        # arbitrary projections could not be detected. The bounded projection helper owns
        # the complete sequence of pure forwards and clears its cache before returning.
        if getattr(_SHARED_ACTIVATION_CACHE, "projection_input", None) is not x:
            setattr(_SHARED_ACTIVATION_CACHE, slot, None)
            return factory(), False
        version = None
    metadata = (tuple(x.shape), tuple(x.stride()), x.storage_offset(), x.dtype, x.device, version)
    entry = getattr(_SHARED_ACTIVATION_CACHE, slot, None)
    if entry is not None:
        input_ref, cached_signature, cached_metadata, prepared = entry
        if input_ref() is x and cached_signature == signature and cached_metadata == metadata:
            return prepared, True
    prepared = factory()
    setattr(_SHARED_ACTIVATION_CACHE, slot, (weakref.ref(x), signature, metadata, prepared))
    return prepared, False


def project_shared_input(x: torch.Tensor, *projections: nn.Module) -> tuple[torch.Tensor, ...]:
    """Run pure OrbitQuant projections with one prepared input, including inference tensors.

    The cache lasts only for this call. Different quantization signatures still prepare
    separate inputs. Gradients, custom forwards and module hooks retain ordinary calls.
    """
    pure = not torch.is_grad_enabled() and not torch.compiler.is_compiling() and all(
        type(layer) is OrbitQuantLinear
        and "forward" not in layer.__dict__
        and "_forward_impl" not in layer.__dict__
        and not layer._forward_pre_hooks
        and not layer._forward_hooks
        and layer._compiled_call_impl is None
        for layer in projections
    )
    # Global hooks can also modify an input between projections.
    hooks = torch.nn.modules.module
    pure = pure and not hooks._global_forward_pre_hooks and not hooks._global_forward_hooks
    if not pure:
        return tuple(layer(x) for layer in projections)
    previous_input = getattr(_SHARED_ACTIVATION_CACHE, "projection_input", None)
    previous_entry = getattr(_SHARED_ACTIVATION_CACHE, "entry", None)
    previous_pre_rotation = getattr(_SHARED_ACTIVATION_CACHE, "pre_rotation_entry", None)
    previous_outlier = getattr(_SHARED_ACTIVATION_CACHE, "outlier_entry", None)
    _SHARED_ACTIVATION_CACHE.projection_input = x
    _clear_shared_activation_cache()
    try:
        return tuple(layer(x) for layer in projections)
    finally:
        _SHARED_ACTIVATION_CACHE.projection_input = previous_input
        _SHARED_ACTIVATION_CACHE.entry = previous_entry
        _SHARED_ACTIVATION_CACHE.pre_rotation_entry = previous_pre_rotation
        _SHARED_ACTIVATION_CACHE.outlier_entry = previous_outlier


def _compile_registry_lookup(handle: int):
    module = _COMPILE_REGISTRY.get(handle)
    if module is None:
        raise RuntimeError(
            "orbitquant compiled-forward handle is stale; rebuild the module "
            "before calling the compiled graph"
        )
    return module


def _compile_registry_register(module) -> int:
    # Never reuse a stale traced handle when Python reuses an object address.
    handle = next(_COMPILE_HANDLE_IDS)
    _COMPILE_REGISTRY[handle] = module
    return handle


@torch.library.custom_op("orbitquant::packed_linear_forward", mutates_args=())
def _packed_linear_forward_op(
    x: torch.Tensor, handle: int, out_features: int, bf16_output: bool
) -> torch.Tensor:
    module = _compile_registry_lookup(handle)
    return module._forward_impl(x)


@_packed_linear_forward_op.register_fake
def _packed_linear_forward_fake(
    x: torch.Tensor, handle: int, out_features: int, bf16_output: bool
) -> torch.Tensor:
    dtype = torch.bfloat16 if bf16_output else x.dtype
    return x.new_empty((*x.shape[:-1], out_features), dtype=dtype)
_NATIVE_PACKED_MATMUL_LOAD_ERROR: object | Exception | None = _PACKED_MATMUL_PROBE_MISSING
_NATIVE_CPU_PACKED_MATMUL_LOAD_ERROR: object | Exception | None = (
    _PACKED_MATMUL_PROBE_MISSING
)
_TRITON_PACKED_MATMUL_IMPORT_ERROR: object | Exception | None = _PACKED_MATMUL_PROBE_MISSING


def _torch_uses_hip() -> bool:
    return bool(getattr(torch.version, "hip", None))


def _packed_length(value_count: int, bits: int) -> int:
    return (value_count * bits + 7) // 8


def _first_available_device(*tensors: torch.Tensor | None) -> torch.device | None:
    for tensor in tensors:
        if tensor is not None and tensor.device.type != "meta":
            return tensor.device
    return None


def _clone_constant(tensor: torch.Tensor, *, device: torch.device | None) -> torch.Tensor:
    constant = tensor.detach().clone()
    if device is not None:
        constant = constant.to(device=device)
    return constant


def _quantize_weight_indices(
    weight: torch.Tensor,
    row_norms: torch.Tensor,
    *,
    rotation: RPBHRotation,
    codebook,
    eps: float,
) -> torch.Tensor:
    if weight.device.type in {"cuda", "xpu"}:
        from orbitquant.kernels.triton_cuda import quantize_weight_indices_with_triton

        return quantize_weight_indices_with_triton(
            weight,
            row_norms,
            rotation=rotation,
            codebook=codebook,
            eps=eps,
        )

    rotated_weight = rotation.apply_to_weight(weight)
    # RPBH is norm-preserving, so rotating first and dividing by the original
    # row norm is equivalent to rotating the unit direction from the paper.
    unit_weight = rotated_weight / row_norms.clamp_min(eps)[:, None]
    return codebook.quantize_indices(unit_weight)


def _quantize_weight_pack(
    weight: torch.Tensor,
    row_norms: torch.Tensor,
    *,
    rotation: RPBHRotation,
    codebook,
    bits: int,
    eps: float,
) -> torch.Tensor:
    if weight.device.type in {"cuda", "xpu"}:
        from orbitquant.kernels.triton_cuda import quantize_weight_packed_with_triton

        return quantize_weight_packed_with_triton(
            weight,
            row_norms,
            rotation=rotation,
            codebook=codebook,
            bits=bits,
            eps=eps,
        )

    weight_indices = _quantize_weight_indices(
        weight,
        row_norms,
        rotation=rotation,
        codebook=codebook,
        eps=eps,
    )
    return pack_lowbit(weight_indices, bits=bits, validate=False)


def _quantize_weight_bounded(
    weight: torch.Tensor,
    *,
    rotation: RPBHRotation,
    codebook,
    bits: int,
    eps: float,
    row_tile_size: int,
    quantization_device: torch.device | None = None,
    pre_rotation: RPBHRotation | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a matrix with workspace bounded by one aligned row tile.

    ``pre_rotation`` is the first round of a two-round RPBH; it is folded into each
    row tile before the paper rotation."""

    if weight.ndim != 2:
        raise ValueError("weight must be a matrix")
    out_features, in_features = weight.shape
    output_device = weight.device
    work_device = output_device if quantization_device is None else quantization_device
    packed = torch.empty(
        _packed_length(out_features * in_features, bits),
        dtype=torch.uint8,
        device=output_device,
    )
    row_norms_output = torch.empty(
        out_features,
        dtype=torch.bfloat16,
        device=output_device,
    )
    packed_offset = 0

    for row_start, row_end in iter_aligned_row_tiles(
        out_features,
        in_features,
        bits,
        row_tile_size,
    ):
        tile = weight[row_start:row_end].to(device=work_device)
        if pre_rotation is not None:
            tile = pre_rotation.apply_to_weight(tile.to(torch.float32))
        if tile.device.type in {"cuda", "xpu"}:
            from orbitquant.kernels.triton_cuda import row_norms_with_triton

            row_norms = row_norms_with_triton(tile, eps=eps)
            quantization_weight = tile
        else:
            quantization_weight = tile.to(torch.float32)
            row_norms = quantization_weight.norm(dim=-1)
        packed_tile = _quantize_weight_pack(
            quantization_weight,
            row_norms,
            rotation=rotation,
            codebook=codebook,
            bits=bits,
            eps=eps,
        )
        packed_end = packed_offset + packed_tile.numel()
        packed[packed_offset:packed_end].copy_(packed_tile.to(device=output_device))
        row_norms_output[row_start:row_end].copy_(
            row_norms.to(device=output_device, dtype=torch.bfloat16)
        )
        packed_offset = packed_end

    if packed_offset != packed.numel():
        raise RuntimeError(
            f"row-tiled packing wrote {packed_offset} bytes, expected {packed.numel()}"
        )
    return packed, row_norms_output


def _rotate_weight_bounded(
    weight: torch.Tensor,
    *,
    rotation: RPBHRotation,
    row_tile_size: int,
    quantization_device: torch.device | None = None,
    pre_rotation: RPBHRotation | None = None,
) -> torch.Tensor:
    output_device = weight.device
    work_device = output_device if quantization_device is None else quantization_device
    rotated = torch.empty(weight.shape, dtype=torch.float32, device=output_device)
    for row_start, row_end in iter_aligned_row_tiles(
        weight.shape[0],
        weight.shape[1],
        8,
        row_tile_size,
    ):
        tile = weight[row_start:row_end].to(device=work_device, dtype=torch.float32)
        if pre_rotation is not None:
            tile = pre_rotation.apply_to_weight(tile)
        rotated[row_start:row_end].copy_(
            rotation.apply_to_weight(tile).to(device=output_device)
        )
    return rotated


def _outlier_tensors(
    config: OrbitQuantConfig, module_name: str, in_features: int, *, device: torch.device
) -> torch.Tensor | None:
    channels = config.outlier_channels.get(module_name)
    if not channels:
        return None
    if channels[-1] >= in_features:
        raise ValueError(
            f"{module_name}: outlier channel {channels[-1]} is outside {in_features} inputs"
        )
    return torch.tensor(channels, dtype=torch.long, device=device)


def output_gain(config: OrbitQuantConfig, in_features: int) -> float:
    """Row-norm factor of ``output_gain_correction``: 1 / (E<u,Q_w(u)> * E<u,Q_a(u)>)."""
    if not config.output_gain_correction:
        return 1.0
    retained = codebook_retained_energy(
        get_codebook(in_features, config.weight_bits, config.codebook_version)
    )
    if config.activation_quantizer == "lloyd_max":
        retained *= codebook_retained_energy(
            get_codebook(in_features, config.activation_bits, config.codebook_version)
        )
    return 1.0 / retained


def _native_packed_matmul_load_error() -> Exception | None:
    global _NATIVE_PACKED_MATMUL_LOAD_ERROR
    if _NATIVE_PACKED_MATMUL_LOAD_ERROR is not _PACKED_MATMUL_PROBE_MISSING:
        return _NATIVE_PACKED_MATMUL_LOAD_ERROR
    try:
        from orbitquant.kernels.native_packed_matmul import load_native_packed_matmul_kernel

        load_native_packed_matmul_kernel()
    except Exception as exc:
        _NATIVE_PACKED_MATMUL_LOAD_ERROR = exc
        return exc
    _NATIVE_PACKED_MATMUL_LOAD_ERROR = None
    return None


def _triton_packed_matmul_import_error() -> Exception | None:
    global _TRITON_PACKED_MATMUL_IMPORT_ERROR
    if _TRITON_PACKED_MATMUL_IMPORT_ERROR is not _PACKED_MATMUL_PROBE_MISSING:
        return _TRITON_PACKED_MATMUL_IMPORT_ERROR
    try:
        from orbitquant.kernels.triton_cuda import matmul_packed_weight_with_triton  # noqa: F401
    except Exception as exc:
        _TRITON_PACKED_MATMUL_IMPORT_ERROR = exc
        return exc
    _TRITON_PACKED_MATMUL_IMPORT_ERROR = None
    return None


def _native_cpu_packed_matmul_load_error() -> Exception | None:
    global _NATIVE_CPU_PACKED_MATMUL_LOAD_ERROR
    if _NATIVE_CPU_PACKED_MATMUL_LOAD_ERROR is not _PACKED_MATMUL_PROBE_MISSING:
        return _NATIVE_CPU_PACKED_MATMUL_LOAD_ERROR
    try:
        from orbitquant.kernels.native_packed_matmul import (
            native_packed_matmul_device_available,
        )

        if not native_packed_matmul_device_available("cpu"):
            raise RuntimeError("the loaded native package has no CPU backend")
    except Exception as exc:
        _NATIVE_CPU_PACKED_MATMUL_LOAD_ERROR = exc
        return exc
    _NATIVE_CPU_PACKED_MATMUL_LOAD_ERROR = None
    return None


def _clear_packed_matmul_probe_cache() -> None:
    global _NATIVE_CPU_PACKED_MATMUL_LOAD_ERROR
    global _NATIVE_PACKED_MATMUL_LOAD_ERROR, _TRITON_PACKED_MATMUL_IMPORT_ERROR
    _NATIVE_PACKED_MATMUL_LOAD_ERROR = _PACKED_MATMUL_PROBE_MISSING
    _NATIVE_CPU_PACKED_MATMUL_LOAD_ERROR = _PACKED_MATMUL_PROBE_MISSING
    _TRITON_PACKED_MATMUL_IMPORT_ERROR = _PACKED_MATMUL_PROBE_MISSING


class OrbitQuantLinear(nn.Module):
    """Linear layer with OrbitQuant-packed rotated weights.

    The default runtime uses packed low-bit matmul on CPU/CUDA/MPS when the
    native or Triton kernels are available. The explicit ``dequant_bf16`` mode
    keeps the compatibility/debug reference path.
    """

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name == "_hf_hook" and accelerate_hook_offloads(self):
            self.clear_dequantized_cache()

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        config: OrbitQuantConfig,
        module_name: str,
        source_weight_layout: str,
        bias: torch.Tensor | None,
        packed_weight_indices: torch.Tensor | None,
        row_norms: torch.Tensor | None,
        debug_weight: torch.Tensor | None,
        outlier_channels: torch.Tensor | None = None,
        outlier_weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_bits = config.weight_bits
        self.activation_bits = config.activation_bits
        self.runtime_mode = config.runtime_mode
        self.w4a4_int8_weight_cache = config.w4a4_int8_weight_cache
        self.activation_kernel_backend = config.activation_kernel_backend
        self.packed_matmul_block_m = config.packed_matmul_block_m
        self.packed_matmul_block_n = config.packed_matmul_block_n
        self.packed_matmul_block_k = config.packed_matmul_block_k
        self.packed_matmul_num_warps = config.packed_matmul_num_warps
        self.module_name = module_name
        self.source_weight_layout = source_weight_layout
        self.activation_eps = config.activation_eps
        self.activation_quantizer = config.activation_quantizer
        self.output_gain = output_gain(config, in_features)
        self.rotation = get_rpbh_rotation(
            dim=in_features, seed=config.rotation_seed, block_size=config.block_size
        )
        self.rotation_rounds = config.rotation_rounds
        self.pre_rotation = get_pre_rotation(
            in_features,
            seed=config.rotation_seed,
            block_size=config.block_size,
            rotation_rounds=config.rotation_rounds,
        )
        self.weight_codebook = get_codebook(
            in_features, config.weight_bits, config.codebook_version
        )
        # Absmax INT8 activations use no Lloyd-Max table; exact 8-bit tables also stop
        # converging for wide inputs (d >= 7680), so none is built for them.
        self.activation_codebook = (
            None
            if self.activation_quantizer == "absmax_int8"
            else get_codebook(in_features, config.activation_bits, config.codebook_version)
        )
        constant_device = _first_available_device(
            packed_weight_indices,
            row_norms,
            debug_weight,
            bias,
        )
        self.register_buffer(
            "_rotation_permutation",
            _clone_constant(
                self.rotation.permutation.to(torch.int32), device=constant_device
            ),
            persistent=False,
        )
        self.register_buffer(
            "_rotation_signs",
            _clone_constant(self.rotation.signs, device=constant_device),
            persistent=False,
        )
        self.register_buffer(
            "_pre_rotation_permutation",
            None
            if self.pre_rotation is None
            else _clone_constant(self.pre_rotation.permutation, device=constant_device),
            persistent=False,
        )
        self.register_buffer(
            "_pre_rotation_signs",
            None
            if self.pre_rotation is None
            else _clone_constant(self.pre_rotation.signs, device=constant_device),
            persistent=False,
        )
        self.register_buffer(
            "_activation_codebook_centroids",
            None
            if self.activation_codebook is None
            else _clone_constant(self.activation_codebook.centroids, device=constant_device),
            persistent=False,
        )
        self.register_buffer(
            "_activation_codebook_boundaries",
            None
            if self.activation_codebook is None
            else _clone_constant(self.activation_codebook.boundaries, device=constant_device),
            persistent=False,
        )
        self.register_buffer(
            "_weight_codebook_centroids",
            _clone_constant(self.weight_codebook.centroids, device=constant_device),
            persistent=False,
        )

        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(bias.detach().clone(), requires_grad=False)

        if packed_weight_indices is not None:
            self.packed_weight_indices = nn.Parameter(
                packed_weight_indices, requires_grad=False
            )
        else:
            self.packed_weight_indices = None
        if row_norms is not None:
            self.row_norms = nn.Parameter(row_norms, requires_grad=False)
        else:
            self.row_norms = None
        if debug_weight is not None:
            self.register_buffer("debug_weight", debug_weight)
        else:
            self.debug_weight = None
        # Input channels kept out of the quantizer (config.outlier_channels): their
        # columns of the source weight, a BF16 side GEMM on the raw channels.
        if (outlier_channels is None) != (outlier_weight is None):
            raise ValueError("outlier_channels and outlier_weight go together")
        self.register_buffer("outlier_channels", outlier_channels)
        self._outlier_signature = (
            None if outlier_channels is None else tuple(outlier_channels.tolist())
        )
        if outlier_weight is None:
            self.register_parameter("outlier_weight", None)
        else:
            self.outlier_weight = nn.Parameter(outlier_weight, requires_grad=False)
        self._dequantized_weight_cache: torch.Tensor | None = None
        self._dequantized_weight_cache_key: tuple[str, torch.dtype] | None = None
        self._int8_surrogate_cache: tuple[str, torch.Tensor, float, torch.Tensor, float] | None = (
            None
        )
        self._runtime_probe_cache: dict[str, bool] = {}
        self._bias_cache: tuple[tuple[str, torch.dtype], torch.Tensor] | None = None
        self._int8_weight_cache: tuple[str, torch.Tensor] | None = None
        self._absmax_quantizer: tuple[str, object] | None = None
        self._compile_handle = _compile_registry_register(self)
        self.last_effective_runtime_mode: str | None = None
        self.last_activation_kernel_backend: str | None = None
        self.last_forward_device_type: str | None = None
        self.last_native_w4a4_enabled = False
        self.last_native_w4_activation_enabled = False
        self.last_native_int8_activation_enabled = False
        self.last_activation_cache_hit = False
        self._derived_constants_valid = True

    @classmethod
    def from_linear(
        cls,
        layer: nn.Module,
        *,
        config: OrbitQuantConfig,
        module_name: str,
    ) -> OrbitQuantLinear:
        spec = linear_module_spec(layer)
        if spec is None:
            raise TypeError(f"no OrbitQuant linear adapter registered for {type(layer).__name__}")
        source_weight = canonical_linear_weight(layer).detach()
        source_bias = getattr(layer, "bias", None)
        bias = None if source_bias is None else source_bias.detach()
        return cls.from_weight(
            source_weight,
            bias=bias,
            in_features=spec.in_features,
            out_features=spec.out_features,
            source_weight_layout=spec.weight_layout,
            config=config,
            module_name=module_name,
        )

    @classmethod
    def from_weight(
        cls,
        weight: torch.Tensor,
        *,
        bias: torch.Tensor | None,
        in_features: int,
        out_features: int,
        source_weight_layout: str,
        config: OrbitQuantConfig,
        module_name: str,
        quantization_device: torch.device | None = None,
    ) -> OrbitQuantLinear:
        expected_shape = (out_features, in_features)
        if tuple(weight.shape) != expected_shape:
            raise ValueError(
                f"expected canonical weight shape {expected_shape}, got {tuple(weight.shape)}"
            )
        rotation = get_rpbh_rotation(
            dim=in_features, seed=config.rotation_seed, block_size=config.block_size
        )
        pre_rotation = get_pre_rotation(
            in_features,
            seed=config.rotation_seed,
            block_size=config.block_size,
            rotation_rounds=config.rotation_rounds,
        )
        outliers = _outlier_tensors(config, module_name, in_features, device=weight.device)
        outlier_weight = None
        if outliers is not None:
            outlier_weight = weight.index_select(1, outliers).to(torch.bfloat16).contiguous()
            weight = weight.index_fill(1, outliers, 0)

        if config.runtime_mode == "debug_no_quant":
            rotated_weight = _rotate_weight_bounded(
                weight,
                rotation=rotation,
                row_tile_size=config.weight_row_tile_size,
                quantization_device=quantization_device,
                pre_rotation=pre_rotation,
            )
            return cls(
                in_features=in_features,
                out_features=out_features,
                config=config,
                module_name=module_name,
                source_weight_layout=source_weight_layout,
                bias=bias,
                packed_weight_indices=None,
                row_norms=None,
                debug_weight=rotated_weight,
                outlier_channels=outliers,
                outlier_weight=outlier_weight,
            )

        codebook = get_codebook(in_features, config.weight_bits, config.codebook_version)
        packed, row_norms = _quantize_weight_bounded(
            weight,
            rotation=rotation,
            codebook=codebook,
            bits=config.weight_bits,
            eps=config.activation_eps,
            row_tile_size=config.weight_row_tile_size,
            quantization_device=quantization_device,
            pre_rotation=pre_rotation,
        )
        gain = output_gain(config, in_features)
        if gain != 1.0:
            row_norms = (row_norms.to(torch.float32) * gain).to(row_norms.dtype)

        return cls(
            in_features=in_features,
            out_features=out_features,
            config=config,
            module_name=module_name,
            source_weight_layout=source_weight_layout,
            bias=bias,
            packed_weight_indices=packed,
            row_norms=row_norms,
            debug_weight=None,
            outlier_channels=outliers,
            outlier_weight=outlier_weight,
        )

    @classmethod
    def empty_from_linear(
        cls,
        layer: nn.Module,
        *,
        config: OrbitQuantConfig,
        module_name: str,
    ) -> OrbitQuantLinear:
        spec = linear_module_spec(layer)
        if spec is None:
            raise TypeError(f"no OrbitQuant linear adapter registered for {type(layer).__name__}")
        source_bias = getattr(layer, "bias", None)
        bias = None
        if source_bias is not None:
            bias = torch.zeros(
                spec.out_features, dtype=source_bias.dtype, device=source_bias.device
            )
        debug_weight = None
        packed_weight_indices = None
        row_norms = None
        if config.runtime_mode == "debug_no_quant":
            debug_weight = torch.empty(
                spec.out_features,
                spec.in_features,
                dtype=layer.weight.dtype,
                device=layer.weight.device,
            )
        else:
            packed_weight_indices = torch.empty(
                _packed_length(spec.out_features * spec.in_features, config.weight_bits),
                dtype=torch.uint8,
                device=layer.weight.device,
            )
            row_norms = torch.empty(
                spec.out_features, dtype=torch.bfloat16, device=layer.weight.device
            )
        outliers = _outlier_tensors(
            config, module_name, spec.in_features, device=layer.weight.device
        )
        outlier_weight = None
        if outliers is not None:
            outlier_weight = torch.empty(
                spec.out_features,
                outliers.numel(),
                dtype=torch.bfloat16,
                device=layer.weight.device,
            )
        empty = cls(
            in_features=spec.in_features,
            out_features=spec.out_features,
            config=config,
            module_name=module_name,
            source_weight_layout=spec.weight_layout,
            bias=bias,
            packed_weight_indices=packed_weight_indices,
            row_norms=row_norms,
            debug_weight=debug_weight,
            outlier_channels=outliers,
            outlier_weight=outlier_weight,
        )
        # Hugging Face streaming loaders may replace non-persistent buffers while
        # moving an empty module skeleton out of meta initialization. Rebuild these
        # deterministic RPBH/codebook constants on the first real forward.
        empty._derived_constants_valid = False
        return empty

    def clear_dequantized_cache(self) -> None:
        self._dequantized_weight_cache = None
        self._dequantized_weight_cache_key = None

    def _remember_dequantized_weight(
        self,
        weight: torch.Tensor,
        cache_key: tuple[str, torch.dtype],
    ) -> torch.Tensor:
        if not accelerate_hook_offloads(self):
            self._dequantized_weight_cache = weight.detach()
            self._dequantized_weight_cache_key = cache_key
        return weight

    def _apply(self, fn, recurse: bool = True):
        result = super()._apply(fn, recurse=recurse)
        self._derived_constants_valid = False
        self._int8_surrogate_cache = None
        self._runtime_probe_cache = {}
        self._bias_cache = None
        self._int8_weight_cache = None
        self.clear_dequantized_cache()
        return result

    def _int8_weight_for(self, device: torch.device) -> torch.Tensor | None:
        """Opt-in persistent INT8 surrogate weight (skips per-forward decode)."""
        if not self.w4a4_int8_weight_cache or accelerate_hook_offloads(self):
            return None
        key = str(device)
        if self._int8_weight_cache is not None and self._int8_weight_cache[0] == key:
            return self._int8_weight_cache[1]
        from orbitquant.kernels.triton_cuda import decode_packed_w4_weight_to_int8

        _, _, weight_codes, _ = self._int8_surrogate_constants(device)
        decoded = decode_packed_w4_weight_to_int8(
            self.packed_weight_indices,
            weight_codes,
            out_features=self.out_features,
            in_features=self.in_features,
        )
        self._int8_weight_cache = (key, decoded)
        return decoded

    def _cached_runtime_probe(self, key: str, probe) -> bool:
        cached = self._runtime_probe_cache.get(key)
        if cached is None:
            cached = bool(probe())
            self._runtime_probe_cache[key] = cached
        return cached

    def _bias_for(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
        if self.bias is None:
            return None
        if self.bias.device == device and self.bias.dtype == dtype:
            return self.bias
        key = (str(device), dtype)
        if self._bias_cache is not None and self._bias_cache[0] == key:
            return self._bias_cache[1]
        value = self.bias.to(device=device, dtype=dtype)
        self._bias_cache = (key, value)
        return value

    def _ensure_derived_constants(self, device: torch.device) -> None:
        if self._derived_constants_valid and self._rotation_permutation.device == device:
            return
        constants = (
            ("_rotation_permutation", self.rotation.permutation, torch.int32),
            ("_rotation_signs", self.rotation.signs, torch.int8),
            (
                "_weight_codebook_centroids",
                self.weight_codebook.centroids,
                torch.float32,
            ),
        )
        if self.activation_codebook is not None:
            constants += (
                (
                    "_activation_codebook_centroids",
                    self.activation_codebook.centroids,
                    torch.float32,
                ),
                (
                    "_activation_codebook_boundaries",
                    self.activation_codebook.boundaries,
                    torch.float32,
                ),
            )
        if self.pre_rotation is not None:
            constants += (
                ("_pre_rotation_permutation", self.pre_rotation.permutation, torch.long),
                ("_pre_rotation_signs", self.pre_rotation.signs, torch.int8),
            )
        for name, source, dtype in constants:
            self._replace_registered_buffer_data(
                name,
                source.to(device=device, dtype=dtype).clone(),
            )
        self._derived_constants_valid = True

    def _replace_registered_buffer_data(
        self,
        name: str,
        value: torch.Tensor,
    ) -> torch.Tensor:
        buffer = self._buffers.get(name)
        if buffer is None:
            raise RuntimeError(f"{name} is not a registered OrbitQuant buffer")
        buffer.data = value
        return buffer

    def _constant_buffer(
        self,
        name: str,
        *,
        device: torch.device,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        tensor = getattr(self, name)
        target_dtype = tensor.dtype if dtype is None else dtype
        if tensor.device != device or tensor.dtype != target_dtype:
            tensor = self._replace_registered_buffer_data(
                name,
                tensor.to(device=device, dtype=target_dtype),
            )
        return tensor

    def _activation_kernel_constant_tensors(self, device: torch.device) -> dict[str, torch.Tensor]:
        return {
            "permutation": self._constant_buffer(
                "_rotation_permutation", device=device, dtype=torch.int32
            ),
            "signs": self._constant_buffer("_rotation_signs", device=device, dtype=torch.int8),
            "centroids": self._constant_buffer(
                "_activation_codebook_centroids", device=device, dtype=torch.float32
            ),
            "boundaries": self._constant_buffer(
                "_activation_codebook_boundaries", device=device, dtype=torch.float32
            ),
        }

    def _validate_triton_packed_matmul_input(self, x: torch.Tensor) -> None:
        if x.device.type not in {"cuda", "xpu"}:
            raise RuntimeError(
                "triton_packed_matmul runtime requires CUDA, HIP, or XPU input tensors; "
                f"got {x.device.type}."
            )

    def _validate_native_packed_matmul_input(self, x: torch.Tensor) -> None:
        if x.device.type == "cuda" and _torch_uses_hip():
            raise RuntimeError(
                "native_packed_matmul currently provides CUDA, not HIP, accelerator "
                "kernels. Use runtime_mode='auto_fused' or 'triton_packed_matmul' on "
                "ROCm, or runtime_mode='dequant_bf16' for the explicit reference path."
            )
        if x.device.type not in {"cpu", "cuda", "mps"}:
            raise RuntimeError(
                "native_packed_matmul runtime requires CPU, CUDA, or MPS input tensors; "
                f"got {x.device.type}."
            )

    def _fused_w2a4_available(self, x: torch.Tensor) -> bool:
        if (
            x.device.type != "cuda"
            or _torch_uses_hip()
            or x.dtype not in {torch.bfloat16, torch.float16}
            or self.weight_bits != 2
            or self.activation_bits != 4
            or self.in_features % 64 != 0
            or self.out_features % 16 != 0
            or self.packed_weight_indices is None
            or self.row_norms is None
        ):
            return False
        rows = x.numel() // self.in_features
        if not (32 <= rows < _W4A4_FUSED_MAX_ROWS):
            return False

        def probe() -> bool:
            try:
                from orbitquant.kernels.triton_cuda import _load_triton

                _load_triton()
            except Exception:
                return False
            return True

        return self._cached_runtime_probe("triton_fused_lowbit", probe)

    def _native_w4a4_available(self, x: torch.Tensor) -> bool:
        if (
            x.device.type != "cuda"
            or _torch_uses_hip()
            or x.dtype not in {torch.bfloat16, torch.float16}
            or self.weight_bits != 4
            or self.activation_bits != 4
            or self.in_features % 64 != 0
        ):
            return False
        def probe() -> bool:
            try:
                from orbitquant.kernels.native_packed_matmul import (
                    native_packed_w4a4_available,
                )

                if not native_packed_w4a4_available():
                    return False
                from orbitquant.kernels.triton_cuda import (  # noqa: F401
                    quantize_activations_packed_w4_with_triton,
                )
            except (ImportError, RuntimeError):
                return False
            return True

        return self._cached_runtime_probe("native_w4a4_kernels", probe)

    def _native_cpu_activation_available(self, x: torch.Tensor) -> bool:
        if x.device.type != "cpu":
            return False

        def probe() -> bool:
            try:
                from orbitquant.kernels.native_packed_matmul import (
                    native_cpu_activation_available,
                )

                return native_cpu_activation_available()
            except (ImportError, RuntimeError):
                return False

        return self._cached_runtime_probe("native_cpu_activation", probe)

    def _native_w4_activation_available(self, x: torch.Tensor) -> bool:
        if (
            not x.is_cuda
            or _torch_uses_hip()
            or self.rotation.block_size != self.in_features
            or self.in_features not in {512, 1024, 2048, 4096, 8192, 16384}
        ):
            return False

        def probe() -> bool:
            try:
                from orbitquant.kernels.native_packed_matmul import (
                    native_packed_w4_activation_available,
                )

                return native_packed_w4_activation_available()
            except (ImportError, RuntimeError):
                return False

        return self._cached_runtime_probe("native_w4_activation", probe)

    def _native_int8_activation_available(self, x: torch.Tensor) -> bool:
        supported_rotation = (
            self.rotation.block_size == self.in_features
            and self.in_features in {512, 1024, 2048, 4096, 8192, 16384}
        ) or (self.in_features, self.rotation.block_size) == (12288, 4096)
        if not x.is_cuda or _torch_uses_hip() or not supported_rotation:
            return False

        def probe() -> bool:
            try:
                from orbitquant.kernels.native_packed_matmul import (
                    native_int8_activation_available,
                )

                return native_int8_activation_available()
            except (ImportError, RuntimeError):
                return False

        return self._cached_runtime_probe("native_int8_activation", probe)

    def _int8_surrogate_constants(
        self, device: torch.device
    ) -> tuple[torch.Tensor, float, torch.Tensor, float]:
        device_key = str(device)
        if self._int8_surrogate_cache is not None:
            cached_device, activation_codes, activation_scale, weight_codes, weight_scale = (
                self._int8_surrogate_cache
            )
            if cached_device == device_key:
                return activation_codes, activation_scale, weight_codes, weight_scale

        from orbitquant.kernels.triton_cuda import fit_int8_centroid_surrogate

        if self.activation_quantizer == "absmax_int8":
            # Absmax INT8 activations carry their own per-token scale; only the
            # weight code table needs a surrogate (an 8-bit codebook has 256 levels).
            activation_codes, activation_scale = torch.empty(0, dtype=torch.int8), 1.0
        else:
            activation_codes, activation_scale = fit_int8_centroid_surrogate(
                self.activation_codebook.centroids
            )
        weight_codes, weight_scale = fit_int8_centroid_surrogate(self.weight_codebook.centroids)
        activation_codes = activation_codes.to(device=device).contiguous()
        weight_codes = weight_codes.to(device=device).contiguous()
        self._int8_surrogate_cache = (
            device_key,
            activation_codes,
            activation_scale,
            weight_codes,
            weight_scale,
        )
        return activation_codes, activation_scale, weight_codes, weight_scale

    @staticmethod
    def _native_w4a4_tile(
        *, rows: int, out_features: int, capability: tuple[int, int]
    ) -> tuple[int, int]:
        if capability == (8, 9):
            if rows >= 256:
                return 256, 128
            if out_features >= 256:
                return 128, 256
        return 128, 128

    def _auto_fused_unavailable_error(
        self,
        *,
        device_type: str,
        native_error: Exception | None,
        triton_error: Exception | None = None,
    ) -> RuntimeError:
        reference_hint = "Set runtime_mode='dequant_bf16' to use the reference/debug path."
        native_hint = (
            "Install the Hugging Face `kernels` package with access to "
            "WaveCut/orbitquant-packed-matmul version 1, set LOCAL_KERNELS to a "
            "compatible built kernel variant directory containing metadata.json, "
            "or make the `orbitquant_packed_matmul` package importable."
        )
        if device_type == "cuda":
            if _torch_uses_hip():
                triton_hint = (
                    "Install the ROCm-compatible PyTorch Triton package to use "
                    "triton_packed_matmul."
                )
                return RuntimeError(
                    "auto_fused runtime requires packed low-bit matmul on ROCm and will "
                    "not silently materialize a full dequantized weight matrix. "
                    f"triton_packed_matmul failed: {triton_error}. {triton_hint} "
                    f"{reference_hint}"
                )
            triton_hint = "Install a CUDA-compatible Triton stack to use triton_packed_matmul."
            return RuntimeError(
                "auto_fused runtime requires packed low-bit matmul on CUDA and will not "
                "silently materialize a full dequantized weight matrix. Tried "
                f"native_packed_matmul ({native_error}) and triton_packed_matmul "
                f"({triton_error}). {native_hint} {triton_hint} {reference_hint}"
            )
        if device_type == "mps":
            return RuntimeError(
                "auto_fused runtime requires the native Metal packed low-bit matmul "
                "kernel on MPS and will not silently materialize a full dequantized "
                f"weight matrix. native_packed_matmul failed: {native_error}. "
                f"{native_hint} {reference_hint}"
            )
        if device_type == "xpu":
            return RuntimeError(
                "auto_fused XPU dispatch remains experimental until real Intel GPU "
                "correctness, memory, profiler, and performance proof is recorded. "
                "For explicit validation, set runtime_mode='triton_packed_matmul' and "
                "activation_kernel_backend='triton_xpu'. "
                f"{reference_hint}"
            )
        return RuntimeError(
            f"auto_fused runtime does not support device type {device_type!r}. {reference_hint}"
        )

    def _transient_dequant_allowed(self) -> bool:
        if os.environ.get("ORBITQUANT_STRICT_PACKED", "").strip() in {"1", "true", "yes"}:
            return False
        max_mb_raw = os.environ.get("ORBITQUANT_TRANSIENT_DEQUANT_MAX_MB", "512")
        try:
            max_bytes = float(max_mb_raw) * 1024 * 1024
        except ValueError:
            max_bytes = 512 * 1024 * 1024
        return self.out_features * self.in_features * 2 <= max_bytes

    def _warn_auto_fused_degraded(
        self, *, target: str, native_error: Exception | None
    ) -> None:
        key = (self.in_features, self.out_features, self.weight_bits, target)
        if key in _AUTO_FUSED_DEGRADATION_WARNED:
            return
        _AUTO_FUSED_DEGRADATION_WARNED.add(key)
        logger.warning(
            "auto_fused could not use the optimized packed kernels for a "
            "W%dA%d %dx%d layer and fell back to %s. Native kernels were "
            "unavailable (%s). Run `orbitquant kernels-install` (or "
            "`--build`) to provision them; `orbitquant kernels-status` "
            "explains the resolution. Set ORBITQUANT_STRICT_PACKED=1 to "
            "forbid the transient dequant fallback.",
            self.weight_bits,
            self.activation_bits,
            self.out_features,
            self.in_features,
            target,
            native_error,
        )

    def _resolve_auto_fused_runtime(self, x: torch.Tensor) -> str:
        device_type = x.device.type
        if device_type == "cpu":
            if _native_cpu_packed_matmul_load_error() is None:
                return "native_packed_matmul"
            return "dequant_bf16"
        if device_type == "cuda":
            if _torch_uses_hip():
                triton_error = _triton_packed_matmul_import_error()
                if triton_error is not None:
                    raise self._auto_fused_unavailable_error(
                        device_type=device_type,
                        native_error=None,
                        triton_error=triton_error,
                    )
                raise RuntimeError(
                    "auto_fused ROCm dispatch remains experimental until real AMD "
                    "hardware correctness, memory, and performance proof is recorded. "
                    "For explicit validation, set runtime_mode='triton_packed_matmul' "
                    "and activation_kernel_backend='triton_rocm'. Set "
                    "runtime_mode='dequant_bf16' for the reference/debug path."
                )
            native_error = _native_packed_matmul_load_error()
            if native_error is None:
                return "native_packed_matmul"
            # Without the native package the only packed CUDA path is the
            # generic Triton GEMM, which re-decodes the packed weight for
            # every row tile: measured 5.3x slower than BF16 end to end on a
            # 4608-wide DiT. A per-forward transient dequant (peak overhead of
            # exactly one BF16 weight matrix, nothing cached) restores the
            # cuBLAS floor, so prefer it for layers inside the size budget.
            if self._transient_dequant_allowed():
                self._warn_auto_fused_degraded(
                    target="a transient per-forward dequant (BF16-floor speed)",
                    native_error=native_error,
                )
                return "dequant_transient"
            triton_error = _triton_packed_matmul_import_error()
            if triton_error is None:
                self._warn_auto_fused_degraded(
                    target="the generic Triton packed GEMM (slow on large row counts)",
                    native_error=native_error,
                )
                return "triton_packed_matmul"
            raise self._auto_fused_unavailable_error(
                device_type=device_type,
                native_error=native_error,
                triton_error=triton_error,
            )
        if device_type == "mps":
            native_error = _native_packed_matmul_load_error()
            if native_error is None:
                return "native_packed_matmul"
            raise self._auto_fused_unavailable_error(
                device_type=device_type,
                native_error=native_error,
            )
        if device_type == "xpu":
            raise self._auto_fused_unavailable_error(
                device_type=device_type,
                native_error=None,
            )
        raise self._auto_fused_unavailable_error(
            device_type=device_type,
            native_error=None,
        )

    def _dequantize_weight(
        self, *, device: torch.device, dtype: torch.dtype, remember: bool = True
    ) -> torch.Tensor:
        cache_key = (str(device), dtype)
        if (
            not accelerate_hook_offloads(self)
            and self._dequantized_weight_cache is not None
            and self._dequantized_weight_cache_key == cache_key
        ):
            return self._dequantized_weight_cache

        def _finish(weight: torch.Tensor) -> torch.Tensor:
            if remember:
                return self._remember_dequantized_weight(weight, cache_key)
            return weight

        if self.debug_weight is not None:
            weight = self.debug_weight.to(device=device, dtype=dtype)
            return _finish(weight)
        if self.packed_weight_indices is None or self.row_norms is None:
            raise RuntimeError("OrbitQuantLinear is missing quantized weight buffers")

        if device.type == "mps":
            try:
                from orbitquant.kernels.mps import (
                    dequantize_packed_weight_with_mps,
                    mps_metal_available,
                )
            except Exception:
                mps_metal_available = None
            if mps_metal_available is not None and mps_metal_available():
                weight = dequantize_packed_weight_with_mps(
                    self.packed_weight_indices,
                    self.row_norms,
                    self.weight_codebook,
                    bits=self.weight_bits,
                    out_features=self.out_features,
                    in_features=self.in_features,
                )
                dequantized = weight.to(dtype=dtype)
                return _finish(dequantized)
        if device.type in {"cuda", "xpu"}:
            try:
                from orbitquant.kernels.triton_cuda import dequantize_packed_weight_with_triton
            except Exception:
                dequantize_packed_weight_with_triton = None
            if dequantize_packed_weight_with_triton is not None:
                weight = dequantize_packed_weight_with_triton(
                    self.packed_weight_indices,
                    self.row_norms,
                    self.weight_codebook,
                    bits=self.weight_bits,
                    out_features=self.out_features,
                    in_features=self.in_features,
                    device=device,
                )
                dequantized = weight.to(dtype=dtype)
                return _finish(dequantized)

        flat = unpack_lowbit(
            self.packed_weight_indices,
            bits=self.weight_bits,
            length=self.out_features * self.in_features,
        ).to(device=device, dtype=torch.long)
        indices = flat.reshape(self.out_features, self.in_features)
        centroids = self.weight_codebook.centroids.to(device=device, dtype=torch.float32)
        row_norms = self.row_norms.to(device=device, dtype=torch.float32)
        weight = row_norms[:, None] * centroids[indices]
        dequantized = weight.to(dtype=dtype)
        return _finish(dequantized)

    def _absmax_int8_activations(self, x: torch.Tensor) -> torch.Tensor:
        """Rotated input as per-token absmax INT8 values, the fused runtime's 8-bit numerics."""

        def factory() -> torch.Tensor:
            permutation = self._constant_buffer(
                "_rotation_permutation", device=x.device, dtype=torch.int32
            )
            signs = self._constant_buffer("_rotation_signs", device=x.device, dtype=torch.int8)
            rotated = self.rotation.apply_to_activations(
                x.to(torch.float32), permutation=permutation.to(torch.long), signs=signs
            )
            scale = rotated.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / 127.0
            codes = torch.round(rotated / scale).clamp_(-127.0, 127.0)
            return (codes * scale).to(x.dtype)

        rotated, self.last_activation_cache_hit = _get_or_create_shared_activation_quantization(
            x, ("absmax_int8", id(self.rotation)), factory
        )
        return rotated

    def _absmax_int8_cuda_available(self, x: torch.Tensor, runtime_mode: str) -> bool:
        if runtime_mode not in {"native_packed_matmul", "triton_packed_matmul"}:
            return False
        if not x.is_cuda or _torch_uses_hip() or x.dtype != torch.bfloat16:
            return False
        if self.weight_bits not in (2, 4) or self.packed_weight_indices is None:
            return False
        if self.in_features % 64 != 0 or self.out_features % 16 != 0:
            return False
        if self.rotation.block_size > 4096 and self.in_features not in (8192, 16384):
            return False

        def probe() -> bool:
            try:
                from orbitquant.kernels import triton_activation  # noqa: F401
            except (ImportError, RuntimeError):
                return False
            return True

        return self._cached_runtime_probe("triton_absmax_int8", probe)

    def _absmax_int8_cuda_forward(self, x: torch.Tensor) -> torch.Tensor:
        """INT8 GEMM of absmax INT8 activations against the packed W2/W4 weight codes."""
        from orbitquant.kernels import triton_activation
        from orbitquant.kernels.triton_cuda import (
            matmul_int8_activations_packed_lowbit_fused_with_triton,
            matmul_int8_activations_packed_w4_with_int_mm,
        )

        device_key = str(x.device)
        if self._absmax_quantizer is None or self._absmax_quantizer[0] != device_key:
            quantizer = triton_activation.ActivationQuantizer(
                self.rotation, None, self.activation_eps, x.device
            )
            self._absmax_quantizer = (device_key, quantizer)
        quantizer = self._absmax_quantizer[1]
        rows_x = x.reshape(-1, self.in_features)
        (codes, token_scales), self.last_activation_cache_hit = (
            _get_or_create_shared_activation_quantization(
                x, ("triton_absmax_int8", id(self.rotation)), lambda: quantizer(rows_x)
            )
        )
        _, _, weight_codes, weight_scale = self._int8_surrogate_constants(x.device)
        bias = self._bias_for(x.device, x.dtype)
        self.last_activation_kernel_backend = "triton_cuda_absmax_int8"
        self.last_native_int8_activation_enabled = True
        rows = rows_x.shape[0]
        if self.weight_bits == 2 or rows < _W4A4_FUSED_MAX_ROWS:
            output = matmul_int8_activations_packed_lowbit_fused_with_triton(
                codes,
                self.packed_weight_indices,
                token_scales,
                self.row_norms,
                weight_codes,
                weight_bits=self.weight_bits,
                activation_scale=1.0,
                weight_scale=weight_scale,
                out_features=self.out_features,
                in_features=self.in_features,
                bias=bias,
                output_dtype=x.dtype,
            )
        else:
            output = matmul_int8_activations_packed_w4_with_int_mm(
                codes,
                self.packed_weight_indices,
                token_scales,
                self.row_norms,
                weight_codes,
                activation_scale=1.0,
                weight_scale=weight_scale,
                out_features=self.out_features,
                in_features=self.in_features,
                bias=bias,
                output_dtype=x.dtype,
                chunk_out_features=(
                    2048 if rows >= 2048 and self.out_features >= 8192 else None
                ),
                decoded_weight=self._int8_weight_for(x.device),
            )
        return output.reshape(*x.shape[:-1], self.out_features)

    def _pre_rotate(self, x: torch.Tensor) -> torch.Tensor:
        """First RPBH round of a two-round rotation, shared across projections of one input."""
        rotation = self.pre_rotation

        def factory() -> torch.Tensor:
            rotated = rotation.apply_to_activations(
                x.to(torch.float32),
                permutation=self._pre_rotation_permutation,
                signs=self._pre_rotation_signs,
            )
            return rotated.to(x.dtype)

        rotated, _ = _get_or_create_shared_activation_quantization(
            x, ("pre_rotation", id(rotation)), factory, slot="pre_rotation_entry"
        )
        return rotated

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.compiler.is_compiling():
            return torch.ops.orbitquant.packed_linear_forward(
                x, self._compile_handle, self.out_features, False
            )
        return self._forward_impl(x)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        if self.outlier_channels is None:
            return self._forward_quantized(x)
        channels = self.outlier_channels.to(device=x.device)
        side = F.linear(
            x.index_select(-1, channels),
            self.outlier_weight.to(device=x.device, dtype=x.dtype),
        )

        def factory() -> torch.Tensor:
            return x.index_fill(-1, channels, 0)

        # Q/K/V of one input keep the same channels out; split the input once.
        main, _ = _get_or_create_shared_activation_quantization(
            x, ("outlier_split", self._outlier_signature), factory, slot="outlier_entry"
        )
        return self._forward_quantized(main) + side

    def _forward_quantized(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_derived_constants(x.device)
        if self.pre_rotation is not None:
            x = self._pre_rotate(x)
        runtime_mode = (
            self._resolve_auto_fused_runtime(x)
            if self.runtime_mode == "auto_fused"
            else self.runtime_mode
        )
        self.last_effective_runtime_mode = runtime_mode
        self.last_forward_device_type = x.device.type
        self.last_native_w4a4_enabled = False
        self.last_native_w4_activation_enabled = False
        self.last_native_int8_activation_enabled = False
        self.last_activation_cache_hit = False

        if runtime_mode == "triton_packed_matmul":
            self._validate_triton_packed_matmul_input(x)
        elif runtime_mode == "native_packed_matmul":
            self._validate_native_packed_matmul_input(x)

        if runtime_mode == "native_packed_matmul" and self._fused_w2a4_available(x):
            activation_constants = self._activation_kernel_constant_tensors(x.device)
            activation_codes, activation_scale, weight_codes, weight_scale = (
                self._int8_surrogate_constants(x.device)
            )
            bias = self._bias_for(x.device, x.dtype)
            if self._native_int8_activation_available(x):
                from orbitquant.kernels.native_packed_matmul import (
                    quantize_activations_int8_with_native_kernel,
                )
                from orbitquant.kernels.triton_cuda import (
                    matmul_int8_activations_packed_lowbit_fused_with_triton,
                )

                capability = torch.cuda.get_device_capability(x.device)
                int8_x, token_norms = quantize_activations_int8_with_native_kernel(
                    x,
                    activation_constants["permutation"],
                    activation_constants["signs"],
                    activation_constants["boundaries"],
                    activation_codes,
                    eps=self.activation_eps,
                    inv_sqrt_block=self.rotation.normalization,
                    threads=512 if capability == (8, 9) else 256,
                )
                self.last_activation_kernel_backend = "native_cuda_int8_surrogate"
                self.last_native_int8_activation_enabled = True
                return matmul_int8_activations_packed_lowbit_fused_with_triton(
                    int8_x,
                    self.packed_weight_indices,
                    token_norms,
                    self.row_norms,
                    weight_codes,
                    weight_bits=2,
                    activation_scale=activation_scale,
                    weight_scale=weight_scale,
                    out_features=self.out_features,
                    in_features=self.in_features,
                    bias=bias,
                    output_dtype=x.dtype,
                )
            from orbitquant.kernels.triton_cuda import (
                matmul_packed_w2a4_fused_with_triton,
                quantize_activations_packed_w4_with_triton,
            )

            packed_x, token_norms = quantize_activations_packed_w4_with_triton(
                x,
                rotation=self.rotation,
                codebook=self.activation_codebook,
                eps=self.activation_eps,
                constant_tensors=activation_constants,
            )
            self.last_activation_kernel_backend = "triton_cuda_packed_w4"
            return matmul_packed_w2a4_fused_with_triton(
                packed_x,
                self.packed_weight_indices,
                token_norms,
                self.row_norms,
                activation_codes,
                weight_codes,
                activation_scale=activation_scale,
                weight_scale=weight_scale,
                out_features=self.out_features,
                in_features=self.in_features,
                bias=bias,
                output_dtype=x.dtype,
            )

        if runtime_mode == "native_packed_matmul" and self._native_w4a4_available(x):
            if self.packed_weight_indices is None or self.row_norms is None:
                raise RuntimeError("OrbitQuantLinear is missing quantized weight buffers")
            capability = torch.cuda.get_device_capability(x.device)
            activation_constants = self._activation_kernel_constant_tensors(x.device)
            activation_codes, activation_scale, weight_codes, weight_scale = (
                self._int8_surrogate_constants(x.device)
            )
            rows = x.numel() // self.in_features
            # cuBLASLt INT8 GEMM needs padded rows below 32; the fused native
            # tile kernel reads the packed weights directly and wins there.
            use_cutlass_tn = (
                capability[0] >= 8
                and rows >= 32
                and self.out_features % 16 == 0
                and callable(getattr(torch, "_int_mm", None))
            )
            int8_weight = self._int8_weight_for(x.device) if use_cutlass_tn else None
            # Mid-size row counts pay more for the per-forward INT8 weight
            # decode than the GEMM itself; the fused Triton kernel reads the
            # packed nibbles directly there.
            use_fused_triton = (
                use_cutlass_tn and int8_weight is None and rows < _W4A4_FUSED_MAX_ROWS
            )
            int8_x: torch.Tensor | None = None
            if use_cutlass_tn and self._native_int8_activation_available(x):
                from orbitquant.kernels.native_packed_matmul import (
                    quantize_activations_int8_with_native_kernel,
                )

                threads = 512 if capability == (8, 9) else 256
                (int8_x, token_norms), self.last_activation_cache_hit = (
                    _get_or_create_shared_activation_quantization(
                        x,
                        ("native_cuda_int8_surrogate", id(self.rotation),
                         id(self.activation_codebook), self.activation_eps, threads),
                        lambda: quantize_activations_int8_with_native_kernel(
                            x,
                            activation_constants["permutation"],
                            activation_constants["signs"],
                            activation_constants["boundaries"],
                            activation_codes,
                            eps=self.activation_eps,
                            inv_sqrt_block=self.rotation.normalization,
                            threads=threads,
                        ),
                    )
                )
                self.last_activation_kernel_backend = "native_cuda_int8_surrogate"
                self.last_native_w4_activation_enabled = True
                self.last_native_int8_activation_enabled = True
            elif use_cutlass_tn and self.rotation.block_size <= 4096:
                from orbitquant.kernels.triton_cuda import quantize_activations_int8_with_triton

                (int8_x, token_norms), self.last_activation_cache_hit = (
                    _get_or_create_shared_activation_quantization(
                        x,
                        ("triton_cuda_int8_surrogate", id(self.rotation),
                         id(self.activation_codebook), self.activation_eps),
                        lambda: quantize_activations_int8_with_triton(
                            x,
                            rotation=self.rotation,
                            codebook=self.activation_codebook,
                            activation_codes=activation_codes,
                            eps=self.activation_eps,
                            constant_tensors=activation_constants,
                        ),
                    )
                )
                self.last_activation_kernel_backend = "triton_cuda_int8_surrogate"
                self.last_native_w4_activation_enabled = True
                self.last_native_int8_activation_enabled = True
            elif self._native_w4_activation_available(x):
                from orbitquant.kernels.native_packed_matmul import (
                    quantize_activations_packed_w4_with_native_kernel,
                )

                packed_x, token_norms = quantize_activations_packed_w4_with_native_kernel(
                    x,
                    activation_constants["permutation"],
                    activation_constants["signs"],
                    activation_constants["boundaries"],
                    eps=self.activation_eps,
                    inv_sqrt_block=self.rotation.normalization,
                    threads=512 if capability == (8, 9) else 256,
                )
                self.last_activation_kernel_backend = "native_cuda_packed_w4"
                self.last_native_w4_activation_enabled = True
            else:
                from orbitquant.kernels.triton_cuda import (
                    quantize_activations_packed_w4_with_triton,
                )

                packed_x, token_norms = quantize_activations_packed_w4_with_triton(
                    x,
                    rotation=self.rotation,
                    codebook=self.activation_codebook,
                    eps=self.activation_eps,
                    constant_tensors=activation_constants,
                )
                self.last_activation_kernel_backend = "triton_cuda_packed_w4"
            bias = self._bias_for(x.device, x.dtype)
            self.last_native_w4a4_enabled = True
            if use_fused_triton:
                if int8_x is not None:
                    from orbitquant.kernels.triton_cuda import (
                        matmul_int8_activations_packed_lowbit_fused_with_triton,
                    )

                    return matmul_int8_activations_packed_lowbit_fused_with_triton(
                        int8_x,
                        self.packed_weight_indices,
                        token_norms,
                        self.row_norms,
                        weight_codes,
                        weight_bits=4,
                        activation_scale=activation_scale,
                        weight_scale=weight_scale,
                        out_features=self.out_features,
                        in_features=self.in_features,
                        bias=bias,
                        output_dtype=x.dtype,
                    )
                from orbitquant.kernels.triton_cuda import (
                    matmul_packed_w4a4_fused_with_triton,
                )

                return matmul_packed_w4a4_fused_with_triton(
                    packed_x,
                    self.packed_weight_indices,
                    token_norms,
                    self.row_norms,
                    activation_codes,
                    weight_codes,
                    activation_scale=activation_scale,
                    weight_scale=weight_scale,
                    out_features=self.out_features,
                    in_features=self.in_features,
                    bias=bias,
                    output_dtype=x.dtype,
                )
            if use_cutlass_tn:
                # Chunking bounds the transient INT8 weight but re-reads the
                # activations per chunk; it only measured faster for large row
                # counts against wide projections.
                if rows >= 2048 and self.out_features >= 8192:
                    cutlass_chunk_out_features: int | None = 2048
                else:
                    cutlass_chunk_out_features = None
                if int8_x is not None:
                    from orbitquant.kernels.triton_cuda import (
                        matmul_int8_activations_packed_w4_with_int_mm,
                    )

                    return matmul_int8_activations_packed_w4_with_int_mm(
                        int8_x,
                        self.packed_weight_indices,
                        token_norms,
                        self.row_norms,
                        weight_codes,
                        activation_scale=activation_scale,
                        weight_scale=weight_scale,
                        out_features=self.out_features,
                        in_features=self.in_features,
                        bias=bias,
                        output_dtype=x.dtype,
                        chunk_out_features=cutlass_chunk_out_features,
                        decoded_weight=int8_weight,
                    )
                from orbitquant.kernels.triton_cuda import (
                    matmul_packed_w4a4_with_int_mm,
                )

                return matmul_packed_w4a4_with_int_mm(
                    packed_x,
                    self.packed_weight_indices,
                    token_norms,
                    self.row_norms,
                    activation_codes,
                    weight_codes,
                    activation_scale=activation_scale,
                    weight_scale=weight_scale,
                    out_features=self.out_features,
                    in_features=self.in_features,
                    bias=bias,
                    output_dtype=x.dtype,
                    chunk_out_features=cutlass_chunk_out_features,
                    decoded_weight=int8_weight,
                )
            from orbitquant.kernels.native_packed_matmul import (
                matmul_packed_w4a4_int8_with_native_kernel,
            )

            tile_m, tile_n = self._native_w4a4_tile(
                rows=rows,
                out_features=self.out_features,
                capability=capability,
            )
            return matmul_packed_w4a4_int8_with_native_kernel(
                packed_x,
                self.packed_weight_indices,
                token_norms,
                self.row_norms,
                activation_codes,
                weight_codes,
                activation_scale=activation_scale,
                weight_scale=weight_scale,
                out_features=self.out_features,
                in_features=self.in_features,
                bias=bias,
                output_dtype=x.dtype,
                tile_m=tile_m,
                tile_n=tile_n,
                async_packed=capability[0] >= 8,
                weight_k_major=False,
            )

        absmax_int8 = self.activation_quantizer == "absmax_int8" and runtime_mode not in {
            "debug_no_quant",
            "debug_no_activation_quant",
        }
        if absmax_int8 and self._absmax_int8_cuda_available(x, runtime_mode):
            return self._absmax_int8_cuda_forward(x)

        if absmax_int8:
            self.last_activation_kernel_backend = "torch_absmax_int8"
            rotated_x = self._absmax_int8_activations(x)
        elif runtime_mode == "debug_no_quant":
            self.last_activation_kernel_backend = None
            rotated_x = self.rotation.apply_to_activations(x.to(torch.float32)).to(x.dtype)
        elif runtime_mode == "debug_no_activation_quant":
            self.last_activation_kernel_backend = None
            work = x.to(torch.float32)
            norms = work.norm(dim=-1, keepdim=True)
            rotated_x = (
                self.rotation.apply_to_activations(work / (norms + self.activation_eps)) * norms
            ).to(x.dtype)
        elif runtime_mode == "native_packed_matmul" and self._native_cpu_activation_available(x):
            from orbitquant.kernels.native_packed_matmul import (
                quantize_activations_with_native_cpu_kernel,
            )

            activation_constants = self._activation_kernel_constant_tensors(x.device)
            self.last_activation_kernel_backend = "native_cpu"
            rotated_x = quantize_activations_with_native_cpu_kernel(
                x,
                activation_constants["permutation"],
                activation_constants["signs"],
                activation_constants["centroids"],
                activation_constants["boundaries"],
                eps=self.activation_eps,
                inv_sqrt_block=self.rotation.normalization,
                block_size=self.rotation.block_size,
            )
        else:
            self.last_activation_kernel_backend = select_backend(
                x.device, requested=self.activation_kernel_backend
            )
            rotated_x = quantize_activations_kernel(
                x,
                rotation=self.rotation,
                codebook=self.activation_codebook,
                eps=self.activation_eps,
                backend=self.activation_kernel_backend,
                constant_tensors=self._activation_kernel_constant_tensors(x.device),
            )

        bias = self._bias_for(x.device, rotated_x.dtype)
        if runtime_mode == "triton_packed_matmul":
            if self.packed_weight_indices is None or self.row_norms is None:
                raise RuntimeError("OrbitQuantLinear is missing quantized weight buffers")
            try:
                from orbitquant.kernels.triton_cuda import matmul_packed_weight_with_triton
            except Exception as exc:
                backend = (
                    "XPU"
                    if x.device.type == "xpu"
                    else "ROCm"
                    if _torch_uses_hip()
                    else "CUDA"
                )
                raise RuntimeError(
                    f"triton_packed_matmul runtime requires the Triton {backend} backend"
                ) from exc
            return matmul_packed_weight_with_triton(
                rotated_x,
                self.packed_weight_indices,
                self.row_norms,
                self._constant_buffer(
                    "_weight_codebook_centroids", device=x.device, dtype=torch.float32
                ),
                bits=self.weight_bits,
                out_features=self.out_features,
                in_features=self.in_features,
                bias=bias,
                block_m=self.packed_matmul_block_m,
                block_n=self.packed_matmul_block_n,
                block_k=self.packed_matmul_block_k,
                num_warps=self.packed_matmul_num_warps,
            )
        if runtime_mode == "native_packed_matmul":
            if self.packed_weight_indices is None or self.row_norms is None:
                raise RuntimeError("OrbitQuantLinear is missing quantized weight buffers")
            from orbitquant.kernels.native_packed_matmul import (
                matmul_packed_weight_with_native_kernel,
            )

            return matmul_packed_weight_with_native_kernel(
                rotated_x,
                self.packed_weight_indices,
                self.row_norms,
                self._constant_buffer(
                    "_weight_codebook_centroids", device=x.device, dtype=torch.float32
                ),
                bits=self.weight_bits,
                out_features=self.out_features,
                in_features=self.in_features,
                bias=bias,
                block_m=self.packed_matmul_block_m,
                block_n=self.packed_matmul_block_n,
                block_k=self.packed_matmul_block_k,
            )

        weight = self._dequantize_weight(
            device=x.device,
            dtype=rotated_x.dtype,
            remember=runtime_mode != "dequant_transient",
        )
        return F.linear(rotated_x, weight, bias)
