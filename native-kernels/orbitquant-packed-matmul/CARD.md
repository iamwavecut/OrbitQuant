# OrbitQuant Packed Matmul

Packed low-bit matrix multiplication kernel for OrbitQuant inference.

This kernel consumes OrbitQuant packed weight indices, per-row norms, and Lloyd-Max
centroids directly, avoiding a full BF16/FP16 dequantized weight cache before the
linear projection.

## API

```python
import torch
from orbitquant_packed_matmul import matmul_packed_weight

out = matmul_packed_weight(
    x,
    packed_weight_indices,
    row_norms,
    centroids,
    bits=4,
    out_features=3072,
    in_features=3072,
    bias=bias,
)
```

Inputs:

- `x`: contiguous or reshapeable tensor with shape `[..., in_features]`.
- `packed_weight_indices`: `uint8` low-bit packed row-major codebook indices.
- `row_norms`: row norms with shape `[out_features]`; CUDA consumes the
  artifact's `bfloat16` values directly, while Metal uses `float32` internally.
- `centroids`: `float32` Lloyd-Max centroids with shape `[2**bits]`.
- `bias`: optional projection bias.

`x` may be `float32`, `float16`, or `bfloat16`. The output has shape
`[..., out_features]` and the same dtype as `x`.

## Build And Test

```bash
nix --option sandbox relaxed --option max-jobs 1 --option cores 8 \
  run .#build-and-copy -L
nix --option sandbox relaxed --option max-jobs 1 --option cores 8 \
  run .#ci-test -L
```

The build produces ABI3 Hugging Face Kernels artifacts under `build/` for the
supported backend variants on the current platform. On macOS, `sandbox relaxed`
or enabled Nix sandboxing is required by `kernel-builder`. The commands build
local files only; they do not upload to Kernel Hub.

For a faster CUDA-only development build on a machine with a matching Torch and
CUDA toolchain:

```bash
cargo install --git https://github.com/huggingface/kernels hf-kernel-builder
kernel-builder check-config .
kernel-builder create-pyproject -f .
TORCH_CUDA_ARCH_LIST="8.9" CUDACXX=/usr/local/cuda/bin/nvcc \
  python setup.py build_kernel
```

This generated project is for local testing and must not be committed or
distributed without a successful `kernel-builder check-abi`. Use the Nix build
for redistributable variants.

For direct local imports, add the matching `build/torch*-<backend>-<platform>`
directory to `PYTHONPATH`; the `torch*` variant must match the runtime PyTorch
version:

```bash
export PYTHONPATH="/path/to/build/torch212-metal-aarch64-darwin:$PYTHONPATH"
python -c "import orbitquant_packed_matmul; print(orbitquant_packed_matmul)"
```

OrbitQuant detects that importable package before trying any Hub loader. For
Hugging Face `kernels` local loading instead, set `LOCAL_KERNELS` to the same
built variant directory containing `metadata.json`, not to the source package
root:

```bash
export LOCAL_KERNELS="WaveCut/orbitquant-packed-matmul=/path/to/build/torch212-metal-aarch64-darwin"
```

## Benchmark

The benchmark reports two PyTorch references:

- `predequantized_f_linear_seconds_per_iter`: `torch.nn.functional.linear`
  over a full dequantized weight matrix that was materialized before timing.
- `dequantize_then_f_linear_seconds_per_iter`: materialize the full
  dequantized weight matrix inside each timed iteration, then call
  `torch.nn.functional.linear`.

```bash
PYTHONPATH=/path/to/build/torch212-metal-aarch64-darwin \
python benchmarks/benchmark.py \
  --device mps \
  --bits 4 \
  --rows 512 \
  --in-features 3072 \
  --out-features 3072 \
  --iters 20
```

The script prints JSON with `packed_seconds_per_iter`,
`predequantized_f_linear_seconds_per_iter`,
`dequantize_then_f_linear_seconds_per_iter`,
`packed_vs_predequantized_f_linear_speedup`,
`packed_vs_dequantize_then_f_linear_speedup`, compatibility aliases
`reference_seconds_per_iter` and `packed_vs_reference_speedup`, and
`max_abs_error`.

It also reports storage accounting for the packed weight path:
`packed_weight_indices_bytes`, `row_norms_bytes`, `centroid_bytes`,
`packed_weight_path_bytes`, `materialized_weight_bytes`, and
`packed_weight_path_vs_materialized_weight_ratio`. These values describe only
the weight-side storage used by this operator; they are not end-to-end model
VRAM measurements.
