# OrbitQuant Kernel Support

OrbitQuant uses packed low-bit weights directly in optimized runtime modes. The
reference path materializes a floating-point weight matrix and is available only
when explicitly selected.

The runtime contract below reflects the current source tree. Benchmark tables
retain the exact software and hardware of each recorded run rather than
implying that those versions are current installation requirements. Binary
platform support is only marked verified where build, install, and forward
evidence exists.

## Runtime Contract

`runtime_mode="auto_fused"` is the default.

| Device | Default dispatch | Required support |
| --- | --- | --- |
| CUDA | Native activation kernel plus packed W4A4 tensor-core path; native or Triton packed fallback | A matching native package for the fastest path; Triton for the CUTLASS epilogue and generic packed fallback |
| MPS | Native packed matmul | An importable local Metal package |
| CPU | Native exact activation, packed low-bit matmul, and packed INT4 AdaLN when the CPU variant is importable; reference fallback otherwise | A matching native CPU package for packed execution |

CUDA and MPS raise an actionable error when no packed backend is available.
They do not silently fall back to full weight dequantization. CPU keeps a
compatibility fallback when the optional native package is absent. Use
`runtime_mode="dequant_bf16"` explicitly for compatibility, debugging, or
numerical comparison.

Other explicit modes are `native_packed_matmul`, `triton_packed_matmul`,
`debug_no_quant`, and `debug_no_activation_quant`.

## Backend Status

| Backend | Status | Implemented path |
| --- | --- | --- |
| CUDA | Optimized packed inference | Native RPBH/quantization, chunked packed-weight decode plus CUTLASS INT8 matmul, direct packed CUDA MMA fallback, and generic Triton packed fallback |
| MPS/Metal | Optimized packed inference | Native Metal packed matmul and Metal activation quantization stages |
| CPU | Hardware-verified native source and CI wheel builds; wheel publication pending | Runtime ISA dispatch across scalar, AVX2/FMA, AVX-512/BF16, and ARM64 NEON; exact packed activation, packed low-bit matmul, and packed INT4 group-64 AdaLN |
| Vulkan | Experimental source path; hardware-verified on AMD Cezanne/RADV Linux, Windows unverified | ExecuTorch whole-graph delegate adapter with exact packed W4A4 activation and matmul shaders |
| ROCm | Experimental source candidate; supported-hardware proof pending | Exact Triton activation, low-bit pack/unpack, packed matmul, weight quantization, and packed INT4 AdaLN; explicit-only |
| XPU | Experimental source candidate; Intel hardware proof pending | Exact Triton activation, low-bit pack/unpack, packed matmul, weight quantization, and packed INT4 AdaLN on `torch.xpu`; explicit-only |

## Local Native Package

Kernel Hub publication is not required. Build the native package locally from
`native-kernels/orbitquant-packed-matmul`:

```bash
cd native-kernels/orbitquant-packed-matmul
nix --option sandbox relaxed run .#build-and-copy -L
```

Expose the matching generated variant directly:

```bash
export PYTHONPATH="$PWD/build/<matching-variant>:$PYTHONPATH"
```

For a fast machine-local CUDA build without Nix:

```bash
cargo install --git https://github.com/huggingface/kernels hf-kernel-builder
cd native-kernels/orbitquant-packed-matmul
kernel-builder check-config .
kernel-builder create-pyproject -f .
TORCH_CUDA_ARCH_LIST="8.9" CUDACXX=/usr/local/cuda/bin/nvcc \
  python setup.py build_kernel
export PYTHONPATH="$PWD/build/<matching-cuda-variant>:$PYTHONPATH"
```

For a locally built Metal variant that remains loadable on macOS 15 and newer:

```bash
cargo install --git https://github.com/huggingface/kernels hf-kernel-builder
cd native-kernels/orbitquant-packed-matmul
kernel-builder check-config .
kernel-builder create-pyproject -f .
MACOSX_DEPLOYMENT_TARGET=15.0 \
  CMAKE_ARGS="-DCMAKE_OSX_DEPLOYMENT_TARGET=15.0" \
  python setup.py build_kernel
kernel-builder check-abi --macos 15.0 --python-abi 3.9 .
export PYTHONPATH="$PWD/build/<matching-metal-variant>:$PYTHONPATH"
```

For PyTorch 2.9 CUDA workloads, enable expandable allocator segments before the
Python process starts to minimize reserved/NVML memory:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python generate.py
```

The generated `setup.py` and CMake files come from `kernel-builder` and must not
be committed. This development build targets the current host toolchain. Use the
Nix build for a redistributable variant and run `kernel-builder check-abi` before
distributing it. The Linux wheel CI builds inside `manylinux_2_28`, repairs the
result with `auditwheel`, and runs strict `abi3audit`; an ordinary local Ubuntu
build can still depend on a newer GLIBC.

To load the local build through Hugging Face `kernels` instead of importing it
through `PYTHONPATH`, map the kernel repository to the same generated variant
directory containing `metadata.json`:

```bash
export LOCAL_KERNELS="WaveCut/orbitquant-packed-matmul=$PWD/build/<matching-variant>"
```

The variant must match the active Torch, CUDA or Metal, platform, and C++ ABI
tuple. OrbitQuant rejects incompatible packages instead of loading them.

### Experimental Vulkan source build

The Vulkan path targets the ExecuTorch whole-graph delegate because desktop
PyTorch does not expose a Vulkan tensor backend. It accepts exported FP16 or
FP32 activations, keeps W4 weights and activation indices packed, and uses a
256-entry exact pair-product LUT. It does not create an FP16/BF16 weight
matrix. The source path currently implements W4A4 only and is not selected by
`auto_fused`.

Desktop Vulkan support requires an ExecuTorch source revision containing the
June/July 2026 desktop delegate changes. The source below was verified at
`a6d812a082df57898b8608f56c867140cc9da32c`. Build it with a current LunarG
Vulkan SDK; the system `glslc` shipped by older Linux distributions may not
compile the upstream shaders.

```bash
export EXECUTORCH_SRC=/path/to/executorch
export EXECUTORCH_BUILD=/path/to/executorch-build
export EXECUTORCH_INSTALL=/path/to/executorch-install
export GLSLC=/path/to/vulkan-sdk/bin/glslc

cmake -S "$EXECUTORCH_SRC" -B "$EXECUTORCH_BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$EXECUTORCH_INSTALL" \
  -DPYTHON_EXECUTABLE="$(command -v python3)" \
  -DEXECUTORCH_BUILD_VULKAN=ON \
  -DEXECUTORCH_BUILD_PORTABLE_OPS=ON \
  -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=OFF \
  -DEXECUTORCH_BUILD_TESTS=OFF \
  -DEXECUTORCH_BUILD_PYBIND=OFF \
  -DEXECUTORCH_BUILD_XNNPACK=OFF \
  -DGLSLC_PATH="$GLSLC"
cmake --build "$EXECUTORCH_BUILD" --target install --parallel

cmake -S native-kernels/orbitquant-vulkan \
  -B .local-artifacts/orbitquant-vulkan-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$EXECUTORCH_INSTALL" \
  -DPYTHON_EXECUTABLE="$(command -v python3)" \
  -DORBITQUANT_EXECUTORCH_ROOT="$EXECUTORCH_SRC" \
  -DGLSLC_PATH="$GLSLC" \
  -DORBITQUANT_VULKAN_BUILD_TESTS=ON \
  -DORBITQUANT_VULKAN_BUILD_PTE_RUNNER=ON
cmake --build .local-artifacts/orbitquant-vulkan-build \
  --target orbitquant_vulkan_test orbitquant_vulkan_pte_runner --parallel
.local-artifacts/orbitquant-vulkan-build/orbitquant_vulkan_test
```

`prepare_executorch_vulkan_w4a4_model()` converts already-loaded
`OrbitQuantLinear` modules without requantizing or changing the artifact.
`ExecuTorchVulkanW4A4Linear` and
`register_executorch_vulkan_w4a4()` in
`orbitquant.kernels.executorch_vulkan` provide the Python export boundary. The
native package supplies token norm, RPBH/FWHT plus exact code assignment, and
direct packed pair-LUT matmul dispatches.

The same build and an exported `.pte` were executed on an AMD Ryzen 5 5600G
Cezanne integrated Radeon through Mesa 25.2.8 RADV, Vulkan 1.4.318, ExecuTorch
1.4.0 source commit `a6d812a082df57898b8608f56c867140cc9da32c`, and Vulkan
SDK 1.4.350.1. The device uses wave64. It exposes
`VK_KHR_shader_integer_dot_product`, but reports no accelerated integer-dot
properties and no cooperative-matrix support, so the selected exact path uses
a shared 256-entry pair-product LUT rather than claiming unavailable matrix
hardware.

The finite shader search compared scalar output decode, transposed packed
weights, 4x4, 8x1, 8x2, and 16x1 tiles, shared versus global LUTs, and K tiles
of 4 through 32. The selected prefill shader uses an 8x2 output tile, a 16x4
workgroup, an eight-coordinate K tile, coalesced transposed packed loads, and
3,072 bytes of LDS. RADV reported 32 SGPRs, 48 VGPRs, and no register spills.
Matmul accounts for 92.6% of the 1536 prefill median and 95.8% of the 3072
prefill median; token norm and RPBH/FWHT are not the remaining performance
bottleneck.

A same-host ABBA control also compared wave64 subgroup shuffle against the
selected LDS tile. Each process used 10 warmups and 31 timed iterations. The
subgroup matmul medians were 583.48 versus 532.08 µs at 32x1536 and 2072.44
versus 1859.50 µs at 32x3072, 9.66% and 11.45% slower respectively. RADV
reported 48 VGPRs and no spills for both variants; the subgroup shader reduced
code size and static instruction count but not measured latency, so runtime
dispatch retains the LDS path.

| Rows | Projection | Exact packed median | Exact packed p95 | Resident FP16 median | Packed versus resident FP16 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1536x1536 | 0.1598 ms | 0.1634 ms | 0.2660 ms | 1.66x faster |
| 32 | 1536x1536 | 0.5782 ms | 0.5818 ms | 0.5084 ms | 1.14x slower |
| 32 | 3072x3072 | 1.9480 ms | 1.9659 ms | 1.5318 ms | 1.27x slower |

Relative to the initial direct-LUT implementation, the selected shader reduced
the corresponding medians from 0.3174, 3.0317, and 12.0223 ms. The packed
path is therefore useful for decode on this device, while its prefill path
remains experimental rather than being presented as a universal speedup.

Seven FP32/FP16, bias/no-bias, partial-tile, and realistic-shape C++ cases
passed an independent host reference with a matching Khronos validation layer.
The realistic exported graph contained one Vulkan delegate and produced 12,288
values with maximum absolute error `2.682e-6` and relative RMSE `7.588e-7`.
Its `.pte` was 1,208,960 bytes and ran through the standalone ExecuTorch runner
on the Radeon device.

For a 3072x3072 projection, packed W4 indices occupy 4,718,592 bytes versus
18,874,368 bytes for FP16 weights. Separate clean processes reached 72,920 KiB
and 92,060 KiB maximum RSS respectively; measured GTT growth was 14,770,176
bytes for packed execution and 43,327,488 bytes for resident FP16. First/cached
packed prepack times were 2.925/0.725 ms at 1536 and 10.454/2.273 ms at 3072,
versus 5.645/2.101 ms and 17.969/8.692 ms for FP16 upload. No full floating
weight is created by the packed delegate.

The earlier llvmpipe run remains build and validation evidence only. Windows
Vulkan and other AMD architectures are unverified, the path is W4A4-only, and
it is not selected by `auto_fused`; these limitations keep Vulkan experimental.

Windows x86_64 source portability is covered by the path-filtered
`Windows Vulkan build` workflow. At OrbitQuant commit `f41ae5e`, run
`29174664556` used MSVC 19.51, CMake 4.3, shaderc 2026.2, and the pinned
ExecuTorch commit above to compile the packed shader library, the 9,336,320-byte
hardware test executable, and the 11,616,768-byte PTE runner. The hosted runner
did not expose a Vulkan GPU, so this is build evidence rather than Windows
runtime, correctness, memory, or performance evidence; Windows Vulkan remains
unverified until the same executables run on an actual AMD device.

### Experimental ROCm and XPU source candidates

ROCm and Intel XPU share the exact Triton kernel surface with CUDA where the
upstream compiler supports the operations. PyTorch exposes HIP tensors through
the `cuda` device type, so OrbitQuant checks `torch.version.hip` before loading
or dispatching CUDA-only native kernels. Intel uses the `xpu` device type. Both
candidates remain explicit-only and report `experimental_unverified`; neither
is selected by `auto_fused`.

Use `runtime_mode="triton_packed_matmul"` with
`activation_kernel_backend="triton_rocm"` or `"triton_xpu"` only for validation
with a matching official PyTorch and Triton stack. The optimized call consumes
packed weights directly; `runtime_mode="dequant_bf16"` remains the explicit
floating-point reference. Missing compiler or device support raises an
actionable error instead of silently materializing a full floating-point
weight.

The Cezanne Radeon used for the Vulkan measurements is not ROCm release
evidence: `gfx90c` is outside the current official PyTorch/ROCm support matrix.
ROCm status therefore requires a repeat of the independent oracle, memory
probe, profiler, and realistic-shape benchmark on supported AMD hardware. XPU
requires the same evidence on a real Intel GPU before its status can change.

## Verified Devices And Shapes

The native CPU package was built and executed on two independent Secure Cloud
placements of an AMD EPYC 4564P (Zen 4), using GCC 13.x and Torch 2.12.1 or
2.13.0+cpu. Each hosted cpuset exposed the two SMT threads of one physical
core. These results are single-physical-core ISA and latency evidence, not a
multi-core scaling claim. Runtime dispatch exercised scalar, AVX2/FMA, and
AVX-512/BF16 paths against the same independent reference.

The repeat ISA comparison below pins one hardware thread, leaves its SMT
sibling idle, uses 10 warmups and 31 timed iterations, and reports the packed
matmul portion for 32 BF16 rows. It demonstrates the benefit of the dispatched
hardware primitive rather than treating the scalar fallback as the optimized
implementation.

| Dimension | Scalar | AVX2/FMA | AVX-512/BF16 | AVX-512 versus AVX2 |
| ---: | ---: | ---: | ---: | ---: |
| 1536 | 62.4821 ms | 2.5246 ms | 0.9792 ms | 2.58x |
| 3072 | 249.2694 ms | 9.9002 ms | 3.8213 ms | 2.59x |

For BF16 W4 projections with at least 16 rows, runtime dispatch uses a four-row
accumulator tile and a two-chunk K-loop unroll on the verified AMD family
19h/model 61h processor for input dimensions 1536, 1920, and 3072. Other
processors, shorter row counts, unaligned dimensions, and larger projections
retain the eight-row non-unrolled path. The finite search compared 4, 8, 12,
and 16 row tiles plus unrolled and non-unrolled K loops with one physical core
pinned. The final same-host ABBA repeat below uses Torch 2.12.1+cpu, GCC 13.3,
20 warmups, two seconds of frequency stabilization, one hardware thread pinned
with its SMT sibling idle, and 101 timed iterations per leg.

| Rows | Dimension | Eight-row median | Selected median | Selected p95 | Median improvement |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 1536 | 1.0372 ms | 0.9864 ms | 0.9935 ms | 5.15% |
| 32 | 1920 | 1.6140 ms | 1.5278 ms | 1.5340 ms | 5.64% |
| 32 | 3072 | 4.0210 ms | 3.8480 ms | 3.8631 ms | 4.50% |

Dimension 3840 selected the unchanged eight-row path and repeated within 0.24%
of the baseline. No 9216 speedup is claimed because the repeated distributions
were noisier and dispatch remains on the generic path. At dimension 1536, the
complete activation-plus-matmul path measured 1.1230 ms median: 0.1463 ms for
activation norm, RPBH/FWHT, and code assignment, and 0.9737 ms for packed
matmul. At dimension 3072, the corresponding medians were 4.1220, 0.2764, and
3.8332 ms. A resident BF16 linear measured 0.5400 and 2.1544 ms respectively;
packed dispatch preserves the 4x weight-memory reduction rather than claiming
to beat a permanently dense weight at this row count.

Native sampling put 88.3% of hot-loop samples in the selected
`<4, true, true>` specialization. Against the generic eight-row baseline,
annotated disassembly reduced the hot body from 896 to 703 instructions,
increased `vdpbf16ps` sites from 8 to 30, and reduced ZMM stack references from
37 to zero. The provider exposed `perf_event_paranoid=4`, so hardware PMU
counters were unavailable; the evidence uses wall-clock distributions, native
stack sampling, and ISA disassembly instead. Two threads pinned to the same
physical core were 6.8% and 8.3% slower at dimensions 1536 and 3072, so one
worker per physical core is selected. This does not establish multi-core or
NUMA scaling.

A native-only 32x9216 memory probe constructed the 42,467,328-byte packed
payload directly, without an unpacked index tensor or reference weight. The
first packed call increased process peak RSS by 2,129,920 bytes and 20 calls
increased it by 3,440,640 bytes; a materialized BF16 weight for the same
projection would require 169,869,312 bytes. The optimized call therefore
retained bounded output and allocator scratch instead of a full floating-point
weight.

W2, W3, and W6 packed matmuls dispatch to dedicated SIMD decoders on every
tier: AVX-512 covers all three widths, AVX2 covers W2 (`vpermps` lookup) and
W6 (`vgatherdps` lookup) while W3 stays scalar there, and NEON covers all
three with vectorized FMA over scalar table decode. Row starts must be
byte-aligned (`in_features % 4 == 0` for W2/W6, `% 8 == 0` for W3); other
shapes keep the scalar fallback. Multi-threaded execution uses a lazily
created persistent worker pool instead of per-call thread spawning, with
outputs bitwise identical to single-threaded runs. When at least 16 activation
rows are present, every SIMD tier decodes each packed column once into a
thread-local buffer and reuses it across the row tile instead of re-decoding
per row. Measured with six unpinned worker threads, 32 BF16 rows, 21 timed
iterations (hot-loop medians), and square projections on an AMD EPYC 9354
(Zen 4) Secure Cloud cpuset, and with one thread on an Apple M2 Max
aarch64-linux container:

| Bits | Dimension | Scalar (Zen 4) | AVX2/FMA | AVX-512 | Scalar (M2) | NEON |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| W2 | 3072 / 1536 | 81.76 ms | 5.04 ms | 4.96 ms | 91.66 ms | 7.27 ms |
| W3 | 3072 / 1536 | 82.10 ms | scalar | 7.04 ms | 92.32 ms | 7.49 ms |
| W6 | 3072 / 1536 | 83.98 ms | 8.69 ms | 5.49 ms | 92.31 ms | 8.15 ms |
| W4 | 3072 / 1536 | 82.35 ms | 6.87 ms | 2.53 ms | 91.63 ms | 7.30 ms |

Zen 4 columns use dimension 3072 and the M2 columns use dimension 1536. A
higher-clock EPYC 9684X cpuset measured the same AVX-512 shapes at
1.92/2.86/2.03 ms (W2/W3/W6); absolute numbers track the host, the tier
ordering does not.

The AVX2-only tier was separately profiled on an AMD Ryzen 5 5600G Cezanne
(family 19h/model 50h), GCC 13.3, and Torch 2.11.0. Runtime dispatch uses a
16-row primary accumulator tile with 8-row and 4-row tails only for BF16 W4,
at least 16 rows, dimensions 1536, 1920, or 3072, and that exact verified
CPUID. Every other AVX2 processor and shape retains the generic eight-row
path. The exact Cezanne guard prevents a machine-specific result from changing
the portable AVX2 default.

The finite same-host search compared 4, 8, 10, 12, and 16-row tiles,
`vgatherdps` versus two-register `vpermps` lookup, K-loop unrolling, software
prefetch, LTO, and `-mtune=znver3`. Gather lookup, four-row tiling, and software
prefetch were materially slower; unrolling and LTO did not improve both tested
dimensions. The selected 16/8/4 topology amortizes packed decode across more
activation rows without changing the artifact or numerical path. Disassembly
confirmed AVX2 `vpermps` lookup and FMA accumulation. PMU counters were
unavailable because the host sets `perf_event_paranoid=4`.

The final ABBA repeat pinned one logical CPU with its SMT sibling idle and used
10 warmups plus 101 timed calls per leg. Values below average the two medians;
the selected p95 is the slower of its two legs.

| Rows | Dimension | Eight-row median | Selected median | Selected p95 | Resident BF16 median | Median improvement |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 1536 | 3.9535 ms | 3.5735 ms | 3.6396 ms | 10.1302 ms | 9.61% |
| 32 | 1920 | 6.1341 ms | 5.5369 ms | 5.6402 ms | 15.6631 ms | 9.73% |
| 32 | 3072 | 15.5206 ms | 14.8918 ms | 15.4647 ms | 39.7343 ms | 4.05% |

Activation plus matmul measured 3.7806, 5.7961, and 15.2280 ms at those
dimensions, improvements of 9.24%, 9.74%, and 4.91%. Matmul accounted for
94.5%, 95.5%, and 97.8% of the selected full-pipeline medians, identifying
packed decode/LUT/FMA as the remaining bottleneck rather than activation norm
or RPBH/FWHT. Relative RMSE against the independently materialized BF16
reference was 0.002315, 0.002292, and 0.002250; the optimized and baseline
AVX2 paths had the same error.

Using the six physical cores, with the container cpuset restricted to one
hardware thread per core, improved packed-matmul medians by 4.43x, 4.87x, and
5.33x. Twelve SMT workers were slower than six physical-core workers on every
shape. The host also serves production workloads, so the measured p95 includes
their scheduling interference rather than representing an isolated-server
tail-latency claim.

| Dimension | One thread | Six physical cores | Six-core p95 | Twelve SMT threads |
| ---: | ---: | ---: | ---: | ---: |
| 1536 | 3.5735 ms | 0.8063 ms | 1.4194 ms | 1.0395 ms |
| 1920 | 5.5369 ms | 1.1366 ms | 2.1565 ms | 1.2310 ms |
| 3072 | 14.8918 ms | 2.7934 ms | 5.5808 ms | 3.9207 ms |

Independent hardware checks covered exact activation RPBH/FWHT and exhaustive
centroid assignment, packed W2/W3/W4/W6 with bias, realistic 16/24/32/33-row
W4 shapes, and packed INT4 group-64 AdaLN. A native-only 32x9216 probe used a
42,467,328-byte packed payload and 18,432 bytes of row norms. Its peak RSS grew
by 3,796,992 bytes across 20 calls, versus 169,869,312 bytes for a full BF16
weight, so the Cezanne specialization retains bounded scratch and does not
materialize a floating-point weight matrix.

For an exact W4A4 projection with 32 activation rows and dimension 1536, the
native activation stage measured 0.1484 ms median, packed matmul measured
1.0528 ms, and the full native layer measured 1.2081 ms. Explicit
`dequant_bf16` measured 1.5836 ms and source BF16 `F.linear` measured 0.5479 ms.
The packed and explicit-reference outputs had relative RMSE `1.203e-7` and
maximum error `1.526e-5`. Packed weights and row norms occupied 1,179,648 bytes
versus 4,718,592 bytes for BF16 weights, and the native layer retained no
dequantized-weight cache.

A second x86_64 profile used a Secure Cloud AMD EPYC 9654 host with GCC 13.3,
Torch 2.12.1+cpu, and the two SMT threads `36,132` of one physical core. For a
W4A4 packed matmul with 32 rows and a 1536x1536 projection, the hot median was
1.4928 ms, p95 was 1.5151 ms, and first call was 1.9775 ms. Materializing the
weight and then calling `F.linear` measured 6.6373 ms median and 7.0199 ms p95,
so the packed path was 4.51x faster than that explicit reference. A resident
BF16 `F.linear` remained faster at 0.7823 ms; the packed kernel is selected to
retain the memory advantage, not to claim superiority over a permanently dense
weight. The packed payload, row norms, and centroids occupied 1,182,784 bytes
versus 4,718,592 bytes for BF16, with relative RMSE `6.546e-6` and maximum error
`0.03125`. Disassembly confirmed `vpermw` lookup and `vdpbf16ps` accumulation.

AdaLN uses an exact signed INT4 group-64 lookup with BF16 scales and
activations. The realistic modulation shape below is 3072 input channels and
18432 output channels. Timings are 21 post-warmup calls with
`ORBITQUANT_CPU_THREADS=1`; `auto_fused` selected the AVX-512/BF16 path.

| Rows | Packed median | Packed p95 | Resident BF16 median | Packed vs resident BF16 | Relative RMSE | Max error |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.4828 ms | 2.7575 ms | 6.9140 ms | 2.78x | 1.01e-6 | 0.001953 |
| 4 | 3.0471 ms | 3.0726 ms | 6.3742 ms | 2.09x | 2.50e-5 | 0.0625 |
| 32 | 24.4159 ms | 24.4693 ms | 15.9194 ms | 0.65x | 2.32e-5 | 0.125 |

The packed AdaLN weight is 28,311,552 bytes and its BF16 scales are 1,769,472
bytes, compared with 113,246,208 bytes for the BF16 weight. A native-only
process increased resident memory by 1.4 MB on its first call and by another
4.3 MB across 500 calls; it did not allocate the 113 MB dense weight. The
rows=32 result documents the crossover where oneDNN's resident BF16 GEMM is
faster; AdaLN conditioning normally uses batch-sized row counts rather than
spatial-token row counts.

The same x86_64 stable-ABI 2.11 binary built against Torch 2.13 loaded and ran
under Torch 2.12.1. This verifies the current LibTorch Stable ABI boundary for
those two runtimes; it does not make the wheel independent of the platform
GLIBC, C++ runtime, or CPU ISA.

CI run `29162460269` separately verified installable Python 3.9 ABI3 wheels on
Linux and Windows at source commit `769c9b8`. The 101,455-byte Linux wheel has
both `manylinux_2_24_x86_64` and `manylinux_2_28_x86_64` tags, passed strict
`abi3audit` with a computed Python 3.9 baseline, and has SHA-256
`5458d49de7cb6b15db7f88313d96cb34827dea2b3d4cc21a39c9574f3c928b16`.
Inside the manylinux container, 73 native tests passed and 33 CUDA/MPS tests
were skipped; a separate clean environment with OrbitQuant 0.4.0 and Torch
2.12.1+cpu passed 33 integration/oracle tests with two CUDA/Triton skips.

The 52,805-byte Windows wheel has the `cp39-abi3-win_amd64` tag and SHA-256
`832aa3e89836c3ec8de44a0ec3bd160fe90e162d8a89c7cdca76c657df2e6728`.
It was built with MSVC 14.51 on Windows Server 2025 and executed on an AMD EPYC
9V74 runner with two cores and four logical processors. Its clean native suite
passed 73 tests with 33 CUDA/MPS skips, and the OrbitQuant integration/oracle
suite passed 33 tests with two CUDA/Triton skips. Both wheels include only the
extension, Python loader, and metadata rather than bundled LibTorch libraries.
They are verified CI artifacts, not yet published platform packages.

The ARM64 native CPU path was separately built on an Apple M2 Max. At 32 rows
and dimension 1536, the full native W4A4 layer measured about 1.20 ms versus
1.84 ms for explicit BF16 dequantization; the packed matmul itself was about
0.81 ms versus 0.79 ms for a resident dense matmul. Scalar and NEON paths were
both exercised against the independent reference. These CPU measurements are
separate from the preferred Metal GPU path on Apple Silicon.

CUDA native-package verification passed on an NVIDIA RTX PRO 4500 Blackwell
with Torch 2.13.0+cu130 using the
`torch213-cxx11-cu130-x86_64-linux` ABI3 variant. Native-resolution generation
used `native_packed_matmul` for every OrbitQuant linear in all release model
families:

| Model | Packed OrbitQuant linears |
| --- | ---: |
| FLUX.2 Klein | 100/100 |
| FLUX.1-schnell | 418/418 |
| Z-Image-Turbo | 238/238 |
| Wan2.1-T2V-1.3B | 300/300 |

The activation path used Triton CUDA. No tested layer reached full-weight
dequantization or `F.linear` fallback in optimized mode.

OrbitQuant 0.3.1 MPS verification passed on an Apple M2 Max with Torch 2.12.1.
It covered the native Metal package, inline shader stages, `auto_fused`
dispatch, and a real 3072x3072 projection restored from the published FLUX.2
W4A4 artifact. The current native package suite passed 74 tests on the MPS
host; 23 CUDA-only cases were skipped. The macOS 15 deployment-target build
passed `kernel-builder check-abi` for the Python 3.9 stable ABI, and the packed
and reference outputs were finite and numerically close.

The Metal package also passed an ABI3 build matrix for Torch 2.11, 2.12, and
2.13. A quantized tiny GPT-2 run exercised all eight wrapped projections during
prefill and cached decode through `native_packed_matmul`, with finite outputs.

The CUDA W4A4 stack released in OrbitQuant 0.3.0 was built and tested on an
NVIDIA L40S (`sm_89`) with Torch 2.9.1+cu128 and CUDA 12.8. That recorded native
package suite passed 49 CUDA tests; the ten skipped cases were Metal-only.
Coverage includes W2/W3/W4/W6 generic packed matmul, FP16/BF16, bias and
no-bias paths, partial output tiles, direct packed W4A4 MMA, native packed-A4
activation quantization, and native INT8 activation quantization for full-block
dimensions and the 12288/4096 blocked RPBH case. OrbitQuant 0.3.1 changes the
Metal path; the CUDA implementation measured in this section is unchanged.

For W4A4 on compute capability 8.0 or newer, the selected path is:

1. A native CUDA launch computes token norms, applies RPBH/FWHT, selects the
   fixed Lloyd-Max bins, and emits INT8 surrogate codes.
2. Packed row-major W4 indices are decoded one bounded output-channel chunk at
   a time; the complete floating-point weight matrix is never materialized.
3. `torch._int_mm` dispatches the INT8 matrix product to CUTLASS tensor cores.
4. A Triton epilogue applies token norms, BF16 row norms, both surrogate scales,
   and bias.

The direct packed CUDA MMA implementation remains available for unsupported
CUTLASS shapes. It includes asynchronous packed loads and SM89-specific tile
selection. The checkpoint keeps the original row-major four-bit payload; no
repacked duplicate weights are stored.

The selected production dispatch was also profiled on an NVIDIA GeForce RTX
4090 (`sm_89`) with Torch 2.9.1+cu128 and CUDA 12.8. For a representative
FLUX.2 fused-input projection with 4608 activation rows, 4096 input channels,
and 36864 output channels, ten post-warmup calls measured 4.317 ms median,
4.352 ms mean, and 0.867 GB peak allocated memory. The output was finite and
the dispatch reported `native_packed_matmul` with
`native_cuda_int8_surrogate` activation quantization.

Nsight Systems attributed 77.8% of GPU kernel time to the CUTLASS INT8 GEMM,
10.0% to the fused scale/norm/bias epilogue, 8.2% to bounded packed-W4 decode,
and 4.0% to native token norm, RPBH/FWHT, and codebook assignment. The same
shape measured 4.809 ms median on the L40S. Nsight Compute performance counters
were unavailable on the hosted 4090 because the provider disabled GPU counter
access (`ERR_NVGPUCTRPERM`); the Systems trace and CUDA event timings do not
depend on those counters.

A full FLUX.2 Klein 9B W4A4 pipeline exercised all 396 packed projections
across the transformer and Qwen3 text encoder. The controlled native run used
1024x1024 output, four steps, guidance 1.0, seed 0, and ten identical prompts:

| Runtime | Load | Hot mean | Hot median | CUDA allocated peak | CUDA reserved peak | NVML peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SDNQ UINT4 | 5.918 s | 2.0885 s | 2.0875 s | 14.844 GB | 16.377 GB | 17.564 GB |
| OrbitQuant W4A4 | 2.543 s | 2.0907 s | 2.0920 s | 13.942 GB | 14.544 GB | 15.731 GB |

OrbitQuant was within 0.11% of SDNQ hot mean while using 0.902 GB less CUDA
allocated memory and 1.833 GB less reserved/NVML memory. Every projection
reported `native_cuda_int8_surrogate`; no full-weight dequantization path was
entered. The ten deterministic outputs were finite and matched the separately
validated cumulative W4A4 run byte for byte.

Measured W4 BF16 operator latency for `in_features=768` and
`out_features=2304`:

| Rows | Packed CUDA | Resident BF16 `F.linear` | Materialize + `F.linear` | Packed vs materialize |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0300 ms | 0.0176 ms | 0.0863 ms | 2.88x |
| 3 | 0.0311 ms | 0.0193 ms | 0.0883 ms | 2.84x |
| 8 | 0.0339 ms | 0.0211 ms | 0.0929 ms | 2.74x |
| 9 | 0.0330 ms | 0.0208 ms | 0.0929 ms | 2.82x |
| 15 | 0.0328 ms | 0.0206 ms | 0.0936 ms | 2.85x |
| 16 | 0.0336 ms | 0.0210 ms | 0.0935 ms | 2.78x |
| 31 | 0.0336 ms | 0.0235 ms | 0.0915 ms | 2.73x |
| 64 | 0.0310 ms | 0.0233 ms | 0.0936 ms | 3.02x |
| 512 | 0.0499 ms | 0.0211 ms | 0.0889 ms | 1.78x |

Rows 1-8 use a warp packed-matvec. Rows 9 and above use zero-padded
WMMA/MMA tiles when the dtype and input dimension permit them. CUDA reads BF16
row norms and FP16/BF16 bias directly, so optimized OrbitQuant inference does
not create per-forward FP32 copies of those tensors.

The full W4A4 `OrbitQuantLinear` path, including Triton activation norm, RPBH,
codebook quantization, and native packed matmul, measured 0.1652 ms at one token,
0.1630 ms at 16 tokens, and 0.1644 ms at 512 tokens. The corresponding
prewarmed `dequant_bf16` path measured 0.1519 ms, 0.1574 ms, and 0.1576 ms while
retaining a full BF16 weight. Peak allocated memory was 14.0/14.1/18.0 MB for
packed execution versus 28.2/28.2/29.8 MB for the reference path.

Measured operator latency on an Apple M2 Max with Torch 2.12.1, FP16
activations, W4 packed weights, `in_features=768`, and
`out_features=2304`:

| Rows | Packed Metal | Resident FP16 `F.linear` | Materialize + `F.linear` | Packed vs materialize |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0470 ms | 0.0411 ms | 0.2310 ms | 4.92x |
| 2 | 0.0439 ms | 0.0422 ms | 0.2371 ms | 5.40x |
| 3 | 0.0451 ms | 0.0404 ms | 0.2348 ms | 5.20x |
| 8 | 0.0420 ms | 0.0503 ms | 0.2512 ms | 5.97x |
| 9 | 0.0446 ms | 0.0501 ms | 0.2422 ms | 5.43x |
| 16 | 0.0459 ms | 0.0581 ms | 0.2512 ms | 5.47x |
| 31 | 0.0428 ms | 0.0659 ms | 0.2623 ms | 6.12x |

One-row projections use the SIMD-group packed matvec. Aligned FP16/BF16
projections with two or more rows use the padded matrix path; unsupported or
unaligned shapes retain the generic packed path. All paths consume packed
indices directly and do not allocate a full floating-point weight matrix.

For this shape, packed indices, row norms, and centroids occupy 25.26% of the
FP16 materialized weight size. A permanently resident pre-dequantized
`F.linear` can remain faster at one to three rows, at the cost of retaining the
full FP16 weight; packed execution is faster in the measured 8-31 row cases.

For a full 4096-coordinate RPBH block with constants resident on MPS, the
fused activation stage uses a 512-thread group:

| Rows | 256 threads | 512 threads | Speedup |
| ---: | ---: | ---: | ---: |
| 1 | 0.0752 ms | 0.0562 ms | 1.34x |
| 8 | 0.0797 ms | 0.0551 ms | 1.45x |
| 64 | 0.0839 ms | 0.0693 ms | 1.21x |
| 512 | 0.3229 ms | 0.2548 ms | 1.27x |
| 4096 | 2.0754 ms | 1.7717 ms | 1.17x |

Smaller or multi-block RPBH dimensions retain the 256-thread path.

The Triton CUDA fallback passed on an NVIDIA B200 with Torch 2.8.0+cu128 and
Triton 3.7.1. For the same 1x768 by 2304x768 W4 shape, activation quantization
took 0.0581 ms and the prewarmed packed matmul forward took 0.1234 ms. A
pre-dequantized `F.linear` took 0.0134 ms but requires the full floating-point
weight to remain resident; this comparison is a memory/latency trade-off, not
a packed-kernel speedup claim.

Run the backend gates with:

```bash
scripts/run_cuda_kernel_checks.sh
PYTHON_BIN="$(uv python find)" scripts/run_mps_kernel_checks.sh
```

Verify a published artifact projection with:

```bash
python scripts/verify_hf_kernel_model_artifact.py \
  --repo-id WaveCut/FLUX.2-klein-4B-OrbitQuant-W4A4 \
  --runtime-mode native_packed_matmul
```

## Performance Claims

Packed execution reduces weight-side materialization and runtime memory for the
validated image pipelines. On the controlled L40S FLUX.2 Klein 9B comparison,
the optimized W4A4 path reached practical SDNQ hot-generation parity with lower
allocated, reserved, and NVML memory. Throughput still depends on model shapes,
device, Torch, offload policy, and backend. Wan with CPU offload did not show a
throughput or peak-memory improvement in the recorded native run. OrbitQuant
does not claim a universal speedup.

Synthetic operator benchmarks are diagnostics. Results above compare packed
execution with both weight materialization plus `F.linear` and, where stated,
a permanently resident pre-dequantized weight. Model-level performance claims
must use native model settings and report the reference configuration beside
the packed configuration.
