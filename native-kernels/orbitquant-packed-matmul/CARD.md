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
- `row_norms`: `float32` row norms with shape `[out_features]`.
- `centroids`: `float32` Lloyd-Max centroids with shape `[2**bits]`.
- `bias`: optional projection bias.

The output has shape `[..., out_features]` and the same dtype as `x`.

## Build And Test

```bash
nix --option sandbox relaxed run .#build-and-copy -L
nix --option sandbox relaxed run .#ci-test -L
```

The build produces ABI3 Hugging Face Kernels artifacts under `build/` for the
supported backend variants on the current platform. On macOS, `sandbox relaxed`
or enabled Nix sandboxing is required by `kernel-builder`.
