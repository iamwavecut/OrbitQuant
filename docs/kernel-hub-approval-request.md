# Kernel Hub Approval Request

Use this as the source text for the Hugging Face `kernels-community/README`
discussion requested by `kernel-builder build-and-upload`:

`https://huggingface.co/spaces/kernels-community/README/discussions/new`

Before posting, ensure the referenced source URI is visible to the reviewer or
attach the kernel source directly.

## Title

Request Kernel Hub publish access for `WaveCut/orbitquant-packed-matmul`

## Body

We would like Kernel Hub publish access to publish
`WaveCut/orbitquant-packed-matmul` as a Hugging Face Kernel Hub package.

Repository:

- Kernel Hub repo id: `WaveCut/orbitquant-packed-matmul`
- Source package path: `native-kernels/orbitquant-packed-matmul`
- Source repository: `https://github.com/iamwavecut/OrbitQuant`
- Review source snapshot: `https://huggingface.co/WaveCut/orbitquant-packed-matmul/commit/f7eb3fa912caa27ad682c7ea1757f580a2751a01`
- License: Apache-2.0

Review-ready source package:

The private Kernel Hub repo already contains the reviewable tracked source
snapshot at:

`https://huggingface.co/WaveCut/orbitquant-packed-matmul/commit/062b934389dce9242e0a9185ed469cc3170e3e73`

If the source repository is still private when this request is posted, attach a
source archive generated from the Git-tracked kernel package path:

```bash
git archive --format=tar \
  --prefix=orbitquant-packed-matmul/ \
  HEAD:native-kernels/orbitquant-packed-matmul \
  > /tmp/orbitquant-packed-matmul-source.tar
```

The archive should contain only the tracked source and test files:

- `build.toml`
- `CARD.md`
- `example.py`
- `flake.nix`
- `flake.lock`
- `benchmarks/benchmark.py`
- `tests/__init__.py`
- `tests/test_packed_matmul.py`
- `torch-ext/orbitquant_packed_matmul/__init__.py`
- `torch-ext/torch_binding.cpp`
- `torch-ext/torch_binding.h`
- `orbitquant_packed_matmul_cuda/packed_matmul.cu`
- `orbitquant_packed_matmul_metal/packed_matmul.mm`
- `orbitquant_packed_matmul_metal/packed_matmul.metal`

Do not attach generated `build/`, local `.venv/`, `__pycache__/`, or benchmark
output directories as source material.

Purpose:

`orbitquant-packed-matmul` implements packed low-bit matrix multiplication for
OrbitQuant inference. It consumes packed OrbitQuant weight codebook indices,
per-row norms, Lloyd-Max centroids, and optional bias directly, avoiding a full
BF16/FP16 dequantized weight matrix before the linear projection.

Ecosystem fit:

- Primary library: `orbitquant`
- Target integrations: Hugging Face Diffusers, Hugging Face Transformers, and
  ComfyUI-OrbitQuant
- Target model families: image and video diffusion transformers, including
  FLUX.2 Klein, FLUX.1-schnell, Z-Image-Turbo, and Wan2.1 T2V
- Runtime role: default `auto_fused` path prefers this native packed matmul
  package on CUDA/MPS, with explicit `dequant_bf16` available as reference mode

Kernel-builder compliance:

- Uses `build.toml` and `kernel-builder`
- Uses ABI3-safe `TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops)` bindings
- Does not use pybind11, `torch/extension.h`, `setup.py`, or hardcoded
  `torch.ops` namespaces
- Package tests cover 2/3/4/6-bit packed matmul with and without bias

Current build and verification status:

- Local kernel-builder CI passed with
  `nix --option sandbox relaxed run .#ci-test -L`
- Local Metal builds passed ABI compatibility checks for Python ABI 3.9
- Local MPS OrbitQuant gate passed with `runtime_mode="native_packed_matmul"`
  through `LOCAL_KERNELS`
- `build-and-copy` currently builds 3 local Metal variants
- `build-and-upload` finds those 3 variants and then stops at Kernel Hub
  publish permission, which is why publish access is needed

Benchmark status:

- MPS smoke benchmarks are available from the OrbitQuant kernel gate.
- CUDA host benchmark evidence is pending a stable CUDA host. We can provide
  CUDA benchmark output after the Kernel Hub approval process if required.

Requested approval:

Please enable Kernel Hub publish access for the `WaveCut` publisher namespace
so this package can be uploaded as a `kernel`-type repository and loaded with:

```python
from kernels import get_kernel

kernel = get_kernel(
    "WaveCut/orbitquant-packed-matmul",
    version=1,
    trust_remote_code=True,
)
```
