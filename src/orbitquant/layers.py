from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from orbitquant.codebooks import get_codebook
from orbitquant.config import OrbitQuantConfig
from orbitquant.kernels import quantize_activations_kernel, select_backend
from orbitquant.linear_adapters import canonical_linear_weight, linear_module_spec
from orbitquant.packing import pack_lowbit, unpack_lowbit
from orbitquant.rotations import RPBHRotation, get_rpbh_rotation

_PACKED_MATMUL_PROBE_MISSING = object()
_NATIVE_PACKED_MATMUL_LOAD_ERROR: object | Exception | None = _PACKED_MATMUL_PROBE_MISSING
_TRITON_PACKED_MATMUL_IMPORT_ERROR: object | Exception | None = _PACKED_MATMUL_PROBE_MISSING


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
    if weight.is_cuda:
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
    if weight.is_cuda:
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


def _clear_packed_matmul_probe_cache() -> None:
    global _NATIVE_PACKED_MATMUL_LOAD_ERROR, _TRITON_PACKED_MATMUL_IMPORT_ERROR
    _NATIVE_PACKED_MATMUL_LOAD_ERROR = _PACKED_MATMUL_PROBE_MISSING
    _TRITON_PACKED_MATMUL_IMPORT_ERROR = _PACKED_MATMUL_PROBE_MISSING


class OrbitQuantLinear(nn.Module):
    """Linear layer with OrbitQuant-packed rotated weights.

    The default runtime uses packed low-bit matmul on CUDA/MPS when the native
    or Triton kernels are available. The explicit ``dequant_bf16`` mode keeps
    the compatibility/debug reference path.
    """

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
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_bits = config.weight_bits
        self.activation_bits = config.activation_bits
        self.runtime_mode = config.runtime_mode
        self.activation_kernel_backend = config.activation_kernel_backend
        self.packed_matmul_block_m = config.packed_matmul_block_m
        self.packed_matmul_block_n = config.packed_matmul_block_n
        self.packed_matmul_block_k = config.packed_matmul_block_k
        self.packed_matmul_num_warps = config.packed_matmul_num_warps
        self.module_name = module_name
        self.source_weight_layout = source_weight_layout
        self.activation_eps = config.activation_eps
        self.rotation = get_rpbh_rotation(
            dim=in_features, seed=config.rotation_seed, block_size=config.block_size
        )
        self.weight_codebook = get_codebook(
            in_features, config.weight_bits, config.codebook_version
        )
        self.activation_codebook = get_codebook(
            in_features, config.activation_bits, config.codebook_version
        )
        constant_device = _first_available_device(
            packed_weight_indices,
            row_norms,
            debug_weight,
            bias,
        )
        self.register_buffer(
            "_rotation_permutation",
            _clone_constant(self.rotation.permutation, device=constant_device),
            persistent=False,
        )
        self.register_buffer(
            "_rotation_signs",
            _clone_constant(self.rotation.signs, device=constant_device),
            persistent=False,
        )
        self.register_buffer(
            "_activation_codebook_centroids",
            _clone_constant(self.activation_codebook.centroids, device=constant_device),
            persistent=False,
        )
        self.register_buffer(
            "_activation_codebook_boundaries",
            _clone_constant(self.activation_codebook.boundaries, device=constant_device),
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
            self.register_buffer("packed_weight_indices", packed_weight_indices)
        else:
            self.packed_weight_indices = None
        if row_norms is not None:
            self.register_buffer("row_norms", row_norms)
        else:
            self.row_norms = None
        if debug_weight is not None:
            self.register_buffer("debug_weight", debug_weight)
        else:
            self.debug_weight = None
        self._dequantized_weight_cache: torch.Tensor | None = None
        self._dequantized_weight_cache_key: tuple[str, torch.dtype] | None = None
        self._int8_surrogate_cache: tuple[str, torch.Tensor, float, torch.Tensor, float] | None = (
            None
        )
        self.last_effective_runtime_mode: str | None = None
        self.last_activation_kernel_backend: str | None = None
        self.last_forward_device_type: str | None = None
        self.last_native_w4a4_enabled = False
        self.last_native_w4_activation_enabled = False
        self.last_native_int8_activation_enabled = False
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
        rotation = get_rpbh_rotation(
            dim=spec.in_features, seed=config.rotation_seed, block_size=config.block_size
        )

        if config.runtime_mode == "debug_no_quant":
            weight = source_weight.to(torch.float32)
            rotated_weight = rotation.apply_to_weight(weight)
            return cls(
                in_features=spec.in_features,
                out_features=spec.out_features,
                config=config,
                module_name=module_name,
                source_weight_layout=spec.weight_layout,
                bias=bias,
                packed_weight_indices=None,
                row_norms=None,
                debug_weight=rotated_weight,
            )

        if source_weight.is_cuda:
            from orbitquant.kernels.triton_cuda import row_norms_with_triton

            row_norms = row_norms_with_triton(source_weight, eps=config.activation_eps)
            quantization_weight = source_weight
        else:
            quantization_weight = source_weight.to(torch.float32)
            row_norms = quantization_weight.norm(dim=-1)
        codebook = get_codebook(spec.in_features, config.weight_bits, config.codebook_version)
        packed = _quantize_weight_pack(
            quantization_weight,
            row_norms,
            rotation=rotation,
            codebook=codebook,
            bits=config.weight_bits,
            eps=config.activation_eps,
        )

        return cls(
            in_features=spec.in_features,
            out_features=spec.out_features,
            config=config,
            module_name=module_name,
            source_weight_layout=spec.weight_layout,
            bias=bias,
            packed_weight_indices=packed,
            row_norms=row_norms.to(torch.bfloat16),
            debug_weight=None,
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
        return cls(
            in_features=spec.in_features,
            out_features=spec.out_features,
            config=config,
            module_name=module_name,
            source_weight_layout=spec.weight_layout,
            bias=bias,
            packed_weight_indices=packed_weight_indices,
            row_norms=row_norms,
            debug_weight=debug_weight,
        )

    def clear_dequantized_cache(self) -> None:
        self._dequantized_weight_cache = None
        self._dequantized_weight_cache_key = None

    def _apply(self, fn, recurse: bool = True):
        result = super()._apply(fn, recurse=recurse)
        self._derived_constants_valid = False
        self._int8_surrogate_cache = None
        self.clear_dequantized_cache()
        return result

    def _ensure_derived_constants(self, device: torch.device) -> None:
        if self._derived_constants_valid and self._rotation_permutation.device == device:
            return
        constants = (
            ("_rotation_permutation", self.rotation.permutation, torch.int64),
            ("_rotation_signs", self.rotation.signs, torch.int8),
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
            (
                "_weight_codebook_centroids",
                self.weight_codebook.centroids,
                torch.float32,
            ),
        )
        for name, source, dtype in constants:
            setattr(self, name, source.to(device=device, dtype=dtype).clone())
        self._derived_constants_valid = True

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
            tensor = tensor.to(device=device, dtype=target_dtype)
            setattr(self, name, tensor)
        return tensor

    def _activation_kernel_constant_tensors(self, device: torch.device) -> dict[str, torch.Tensor]:
        return {
            "permutation": self._constant_buffer(
                "_rotation_permutation", device=device, dtype=torch.int64
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
        if x.device.type != "cuda":
            raise RuntimeError(
                f"triton_packed_matmul runtime requires CUDA input tensors; got {x.device.type}."
            )

    def _validate_native_packed_matmul_input(self, x: torch.Tensor) -> None:
        if x.device.type not in {"cuda", "mps"}:
            raise RuntimeError(
                "native_packed_matmul runtime requires CUDA or MPS input tensors; "
                f"got {x.device.type}."
            )

    def _native_w4a4_available(self, x: torch.Tensor) -> bool:
        if (
            x.device.type != "cuda"
            or x.dtype not in {torch.bfloat16, torch.float16}
            or self.weight_bits != 4
            or self.activation_bits != 4
            or self.in_features % 64 != 0
        ):
            return False
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

    def _native_w4_activation_available(self, x: torch.Tensor) -> bool:
        if (
            not x.is_cuda
            or self.rotation.block_size != self.in_features
            or self.in_features not in {512, 1024, 2048, 4096, 8192, 16384}
        ):
            return False
        try:
            from orbitquant.kernels.native_packed_matmul import (
                native_packed_w4_activation_available,
            )

            return native_packed_w4_activation_available()
        except (ImportError, RuntimeError):
            return False

    def _native_int8_activation_available(self, x: torch.Tensor) -> bool:
        supported_rotation = (
            self.rotation.block_size == self.in_features
            and self.in_features in {512, 1024, 2048, 4096, 8192, 16384}
        ) or (self.in_features, self.rotation.block_size) == (12288, 4096)
        if not x.is_cuda or not supported_rotation:
            return False
        try:
            from orbitquant.kernels.native_packed_matmul import (
                native_int8_activation_available,
            )

            return native_int8_activation_available()
        except (ImportError, RuntimeError):
            return False

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
        return RuntimeError(
            f"auto_fused runtime does not support device type {device_type!r}. {reference_hint}"
        )

    def _resolve_auto_fused_runtime(self, x: torch.Tensor) -> str:
        device_type = x.device.type
        if device_type == "cpu":
            return "dequant_bf16"
        if device_type == "cuda":
            native_error = _native_packed_matmul_load_error()
            if native_error is None:
                return "native_packed_matmul"
            triton_error = _triton_packed_matmul_import_error()
            if triton_error is None:
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
        raise self._auto_fused_unavailable_error(
            device_type=device_type,
            native_error=None,
        )

    def _dequantize_weight(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        cache_key = (str(device), dtype)
        if (
            self._dequantized_weight_cache is not None
            and self._dequantized_weight_cache_key == cache_key
        ):
            return self._dequantized_weight_cache

        if self.debug_weight is not None:
            weight = self.debug_weight.to(device=device, dtype=dtype)
            self._dequantized_weight_cache = weight.detach()
            self._dequantized_weight_cache_key = cache_key
            return weight
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
                self._dequantized_weight_cache = dequantized.detach()
                self._dequantized_weight_cache_key = cache_key
                return dequantized
        if device.type == "cuda":
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
                self._dequantized_weight_cache = dequantized.detach()
                self._dequantized_weight_cache_key = cache_key
                return dequantized

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
        self._dequantized_weight_cache = dequantized.detach()
        self._dequantized_weight_cache_key = cache_key
        return dequantized

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_derived_constants(x.device)
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

        if runtime_mode == "triton_packed_matmul":
            self._validate_triton_packed_matmul_input(x)
        elif runtime_mode == "native_packed_matmul":
            self._validate_native_packed_matmul_input(x)

        if runtime_mode == "native_packed_matmul" and self._native_w4a4_available(x):
            if self.packed_weight_indices is None or self.row_norms is None:
                raise RuntimeError("OrbitQuantLinear is missing quantized weight buffers")
            capability = torch.cuda.get_device_capability(x.device)
            activation_constants = self._activation_kernel_constant_tensors(x.device)
            activation_codes, activation_scale, weight_codes, weight_scale = (
                self._int8_surrogate_constants(x.device)
            )
            use_cutlass_tn = (
                capability[0] >= 8
                and self.out_features % 16 == 0
                and callable(getattr(torch, "_int_mm", None))
            )
            int8_x: torch.Tensor | None = None
            if use_cutlass_tn and self._native_int8_activation_available(x):
                from orbitquant.kernels.native_packed_matmul import (
                    quantize_activations_int8_with_native_kernel,
                )

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
            bias = None if self.bias is None else self.bias.to(device=x.device, dtype=x.dtype)
            self.last_native_w4a4_enabled = True
            if use_cutlass_tn:
                if self.in_features >= 8192 and self.out_features <= 4096:
                    cutlass_chunk_out_features = 4096
                else:
                    cutlass_chunk_out_features = 2048
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
                )
            from orbitquant.kernels.native_packed_matmul import (
                matmul_packed_w4a4_int8_with_native_kernel,
            )

            rows = x.numel() // self.in_features
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

        if runtime_mode == "debug_no_quant":
            self.last_activation_kernel_backend = None
            rotated_x = self.rotation.apply_to_activations(x.to(torch.float32)).to(x.dtype)
        elif runtime_mode == "debug_no_activation_quant":
            self.last_activation_kernel_backend = None
            work = x.to(torch.float32)
            norms = work.norm(dim=-1, keepdim=True)
            rotated_x = (
                self.rotation.apply_to_activations(work / (norms + self.activation_eps)) * norms
            ).to(x.dtype)
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

        bias = None if self.bias is None else self.bias.to(device=x.device, dtype=rotated_x.dtype)
        if runtime_mode == "triton_packed_matmul":
            if self.packed_weight_indices is None or self.row_norms is None:
                raise RuntimeError("OrbitQuantLinear is missing quantized weight buffers")
            try:
                from orbitquant.kernels.triton_cuda import matmul_packed_weight_with_triton
            except Exception as exc:
                raise RuntimeError(
                    "triton_packed_matmul runtime requires the Triton CUDA backend"
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

        weight = self._dequantize_weight(device=x.device, dtype=rotated_x.dtype)
        return F.linear(rotated_x, weight, bias)
