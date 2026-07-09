# Kernel Hub Approval Request

Use this as the source text for the Hugging Face `kernels-community/README`
discussion requested by `kernel-builder build-and-upload`:

`https://huggingface.co/spaces/kernels-community/README/discussions/new`

Submitted discussion:

`https://huggingface.co/spaces/kernels-community/README/discussions/15`

Follow-up comment:

On 2026-07-08T18:03Z, the discussion was updated with source snapshot
`6821e4cd5ff1894994d7137c1d861660cfeed1c8` and source archive SHA256
`77aef6caa1bbdbbd77e2cbf5003423073e001191d008473c957795d7bed03651`.

Reviewer follow-up:

On 2026-07-09T07:29Z, `sayakpaul` reported that the source snapshot repo URL
was not visible and asked for performance numbers. On 2026-07-09T09:59Z,
`WaveCut` replied without changing repository visibility: the source snapshot
repo is still private, the tracked source archive SHA256 is
`77aef6caa1bbdbbd77e2cbf5003423073e001191d008473c957795d7bed03651`, and the
reply offered either a public source-only snapshot repo or a source archive
through the preferred review channel. The reply also restated the MPS native
packed matmul smoke benchmark numbers and the CUDA/Triton partial gate evidence,
while keeping native CUDA package numbers pending.

Source visibility follow-up:

As of 2026-07-09T11:54Z, `WaveCut/orbitquant-packed-matmul` is public as a
source snapshot repo:

`https://huggingface.co/WaveCut/orbitquant-packed-matmul`

The live checked commit is
`cb0ceb1a4d070556c52cfba691aba3f6647c246b`. Its file list contains only the
tracked source/test files and no generated `build/`, local `.venv/`,
`__pycache__/`, binary `.so`, or benchmark output files.

The public PyPI source distribution also contains the reviewable kernel source:

- PyPI project: `https://pypi.org/project/orbitquant/0.1.0/`
- Source distribution:
  `https://files.pythonhosted.org/packages/c8/41/1df33fe61ff5638a0b9385bbe0493353b62cfd8f39dd1ee7481487f92ed0/orbitquant-0.1.0.tar.gz`
- SHA256:
  `6abedb769b32c8d70f2763278e106346319d628d85ed7469549faa5020ab1a89`
- Kernel path inside the archive:
  `orbitquant-0.1.0/native-kernels/orbitquant-packed-matmul/`

The older source archive was locally prepared and checked on 2026-07-08T18:00Z:
`/tmp/orbitquant-packed-matmul-source.tar`, SHA256
`77aef6caa1bbdbbd77e2cbf5003423073e001191d008473c957795d7bed03651`,
21 tar entries, and no generated `build/`, local `.venv/`, `__pycache__/`,
binary `.so`, or benchmark output files.

On 2026-07-09T11:56Z, `WaveCut` posted a follow-up comment in discussion 15
with the public source snapshot URL, checked commit, PyPI source distribution
URL, SHA256, and the same claim boundary: MPS native packed-matmul smoke
numbers plus CUDA/Triton gate evidence are available, while native CUDA
`native_packed_matmul` package numbers remain pending a compatible Kernel
Hub/CUDA build path.

On 2026-07-09T12:22Z, `WaveCut` posted another follow-up comment after
updating the benchmark source to snapshot
`c34d9851cde2cf098589927a7b0bed85d65426af`. The updated benchmark reports both
`predequantized_f_linear_seconds_per_iter` and
`dequantize_then_f_linear_seconds_per_iter`. The comment explicitly says the
current MPS native packed-matmul path is not throughput proof for large
matrices: local W4 512x1024x1024 fp16 measured about `0.045x` versus
dequantize-then-F.linear, and W4 512x3072x3072 fp16 measured about `0.044x`
versus dequantize-then-F.linear. The current CUDA source uses WMMA for
FP16/BF16 low-bit modes, but native CUDA package benchmarks remain pending a
compatible Kernel Hub/CUDA build path.

On 2026-07-09T12:27Z, `WaveCut` answered the model-scope question. The kernel
is model-agnostic at the operator level, but intended for OrbitQuant-converted
diffusion transformer backbones with packed `OrbitQuantLinear` layers. Current
target families are FLUX.1-schnell, FLUX.2 Klein, Z-Image-Turbo, and
Wan2.1-T2V-1.3B-Diffusers. The kernel is not intended for arbitrary
unquantized models, text encoders, VAEs, embeddings, timestep MLPs, or final
projection heads.

On 2026-07-09T12:39Z, the public source snapshot was updated to commit
`cb0ceb1a4d070556c52cfba691aba3f6647c246b`. The benchmark JSON now reports
packed weight storage fields:
`packed_weight_indices_bytes`, `row_norms_bytes`, `centroid_bytes`,
`packed_weight_path_bytes`, `materialized_weight_bytes`, and
`packed_weight_path_vs_materialized_weight_ratio`.

On 2026-07-09T12:41Z, `WaveCut` posted a follow-up comment in discussion 15
with the `cb0ceb1a4d070556c52cfba691aba3f6647c246b` source snapshot URL and
the same claim boundary: the new fields are weight-side storage accounting,
not throughput proof; native CUDA package benchmark evidence remains pending a
compatible Kernel Hub/CUDA build path.

On 2026-07-09T12:42Z, `sayakpaul` asked for a way to try optimizing one target
model with these kernels. The repository now includes
`scripts/verify_hf_kernel_model_artifact.py`, which verifies one published
OrbitQuant artifact layer through `runtime_mode="native_packed_matmul"` and
compares it with `dequant_bf16` without loading the full Diffusers pipeline or
running image/video generation.

On 2026-07-09T12:50Z, `WaveCut` replied with the verification script commit
`f42d2dc19897adde62ec3ebb33e4ce748255dd54`, MPS and CUDA `LOCAL_KERNELS`
example commands, and the expected behavior when no loadable native kernel is
available.

On 2026-07-09T12:57Z, the verifier passed locally on Apple Silicon MPS with
the `torch212-metal-aarch64-darwin` local kernel variant and the published
`WaveCut/FLUX.2-klein-4B-OrbitQuant-W4A4` artifact. It verified
`transformer_blocks.0.attn.to_q` (3072x3072) with
`runtime_mode="native_packed_matmul"` against `dequant_bf16` and reported
`finite=true`, `allclose_to_dequant_bf16=true`,
`max_abs_error_vs_dequant_bf16=0.001953125`, and
`packed_weight_path_vs_materialized_weight_ratio=0.2503289116753472`.

On 2026-07-09T12:58Z, `WaveCut` posted that verifier command and JSON result
summary in discussion 15.

## Title

Request Kernel Hub publish access for `WaveCut/orbitquant-packed-matmul`

## Body

We would like Kernel Hub publish access to publish
`WaveCut/orbitquant-packed-matmul` as a Hugging Face Kernel Hub package.

Repository:

- Kernel Hub repo id: `WaveCut/orbitquant-packed-matmul`
- Source package path: `native-kernels/orbitquant-packed-matmul`
- Source repository: `https://github.com/iamwavecut/OrbitQuant`
- Review source snapshot: `https://huggingface.co/WaveCut/orbitquant-packed-matmul/commit/cb0ceb1a4d070556c52cfba691aba3f6647c246b`
- License: Apache-2.0

Review-ready source package:

The public source snapshot repo contains the reviewable tracked source snapshot
at:

`https://huggingface.co/WaveCut/orbitquant-packed-matmul/commit/cb0ceb1a4d070556c52cfba691aba3f6647c246b`

If a source archive is requested instead, generate it from the Git-tracked
kernel package path:

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

- MPS smoke benchmarks from the local `torch212-metal-aarch64-darwin` package
  variant on Apple Silicon with Torch 2.12.1:
  - W4, 512 rows, 1024 input features, 1024 output features, float16:
    `0.00764581459807232` seconds/iteration over 20 iterations.
  - W4, 512 rows, 3072 input features, 3072 output features, float16:
    `0.10189520000712946` seconds/iteration over 10 iterations.
- MPS native packed matmul is currently not throughput-competitive with the
  PyTorch baselines on the large local shapes above; treat those values as
  correctness and memory-path evidence only.
- CUDA native package benchmark evidence is pending a compatible CUDA
  kernel-builder or Kernel Hub build path.

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
