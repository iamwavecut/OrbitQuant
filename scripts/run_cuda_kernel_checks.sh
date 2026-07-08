#!/usr/bin/env bash
set -euo pipefail

stage() {
  printf 'REMOTE_STAGE %s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${ORBITQUANT_VENV:-"$ROOT_DIR/.venv"}"

TOKENS="${ORBITQUANT_BENCH_TOKENS:-512}"
IN_FEATURES="${ORBITQUANT_BENCH_IN_FEATURES:-3072}"
OUT_FEATURES="${ORBITQUANT_BENCH_OUT_FEATURES:-3072}"
HIDDEN_FEATURES="${ORBITQUANT_BENCH_HIDDEN_FEATURES:-9216}"
BLOCK_SIZE="${ORBITQUANT_BENCH_BLOCK_SIZE:-1024}"
WARMUP="${ORBITQUANT_BENCH_WARMUP:-2}"
ITERATIONS="${ORBITQUANT_BENCH_ITERATIONS:-5}"
PACKED_MATMUL_BLOCK_M="${ORBITQUANT_PACKED_MATMUL_BLOCK_M:-32}"
PACKED_MATMUL_BLOCK_N="${ORBITQUANT_PACKED_MATMUL_BLOCK_N:-64}"
PACKED_MATMUL_BLOCK_K="${ORBITQUANT_PACKED_MATMUL_BLOCK_K:-64}"
PACKED_MATMUL_NUM_WARPS="${ORBITQUANT_PACKED_MATMUL_NUM_WARPS:-8}"
RUN_NATIVE_KERNEL_PACKAGE_CI="${ORBITQUANT_RUN_NATIVE_KERNEL_PACKAGE_CI:-1}"

cd "$ROOT_DIR"

stage env-start
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$PYTHON_BIN" -m venv --system-site-packages "$VENV_DIR"
fi

# Keep managed CUDA images on their vendor-matched torch/triton wheels. The
# project itself is installed without dependencies; only lightweight support
# packages are added to the venv.
# shellcheck disable=SC1091
. "$VENV_DIR/bin/activate"

if command -v uv >/dev/null 2>&1; then
  uv pip install hatchling numpy safetensors huggingface_hub packaging tqdm pytest pytest-xdist ruff "kernels>=0.16"
  uv pip install --no-deps -e .
else
  python -m pip install hatchling numpy safetensors huggingface_hub packaging tqdm pytest pytest-xdist ruff "kernels>=0.16"
  python -m pip install --no-deps -e .
fi

python - <<'PY'
import torch
import triton

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in this Python environment")

print("python-env-ok")
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("triton", triton.__version__)
print("device", torch.cuda.get_device_name(0))
PY
stage env-done

stage kernel-tests-start
pytest tests/test_kernels.py tests/test_adaln_rtn.py tests/test_orbit_linear.py -q
stage kernel-tests-done

stage kernel-info-start
orbitquant kernel-info
python - <<'PY'
from orbitquant.kernels import backend_capabilities

capability = backend_capabilities()["triton_cuda"]
required_stages = {
    "activation_norm_rpbh_quant_rescale",
    "packed_weight_dequant",
    "packed_weight_matmul",
    "lowbit_pack",
    "lowbit_unpack",
    "weight_rotation_fwht_quant_pack",
    "adaln_rtn_quant_pack",
    "adaln_rtn_dequant",
}
stages = set(str(capability["optimized_stage"] or "").split(","))
missing = sorted(required_stages - stages)
if capability["claim_status"] != "partial_optimized":
    raise SystemExit(f"unexpected triton_cuda claim_status: {capability['claim_status']}")
if not capability["optimized"]:
    raise SystemExit("triton_cuda backend is not optimized in this environment")
if capability["full_fusion"]:
    raise SystemExit("triton_cuda should not claim full_fusion")
if capability["hf_kernel_builder_compliant"]:
    raise SystemExit("triton_cuda should not claim HF kernel-builder compliance")
if missing:
    raise SystemExit(f"triton_cuda optimized_stage missing: {missing}")
print("triton-cuda-kernel-contract-ok")
PY
stage kernel-info-done

stage kernel-bench-start
orbitquant kernel-bench \
  --tokens "$TOKENS" \
  --in-features "$IN_FEATURES" \
  --out-features "$OUT_FEATURES" \
  --weight-bits 4 \
  --activation-bits 4 \
  --block-size "$BLOCK_SIZE" \
  --activation-kernel-backend triton_cuda \
  --device cuda \
  --dtype bfloat16 \
  --warmup "$WARMUP" \
  --iterations "$ITERATIONS"
stage kernel-bench-done

stage quantize-bench-start
orbitquant quantize-bench \
  --layers 2 \
  --in-features "$IN_FEATURES" \
  --hidden-features "$HIDDEN_FEATURES" \
  --weight-bits 4 \
  --activation-bits 4 \
  --block-size "$BLOCK_SIZE" \
  --source-device cpu \
  --quantization-device cuda \
  --staging-mode component \
  --dtype bfloat16
stage quantize-bench-done

if [[ "$RUN_NATIVE_KERNEL_PACKAGE_CI" == "1" ]]; then
  stage native-kernel-package-ci-start
  if ! command -v nix >/dev/null 2>&1; then
    printf '%s\n' \
      "native kernel package CI requires nix for kernel-builder." \
      "Install nix, use a GPU image with nix, or set ORBITQUANT_RUN_NATIVE_KERNEL_PACKAGE_CI=0 only when a compatible native packed-matmul kernel is already loadable through Hugging Face kernels or an importable package." \
      >&2
    exit 1
  fi
  (
    cd native-kernels/orbitquant-packed-matmul
    nix --option sandbox relaxed run .#ci-test -L
  )
  export LOCAL_KERNELS="WaveCut/orbitquant-packed-matmul=$ROOT_DIR/native-kernels/orbitquant-packed-matmul"
  stage native-packed-matmul-load-start
  python - <<'PY'
from orbitquant.kernels.native_packed_matmul import load_native_packed_matmul_kernel

kernel = load_native_packed_matmul_kernel()
if not hasattr(kernel, "matmul_packed_weight"):
    raise SystemExit("native packed matmul kernel is missing matmul_packed_weight")
print("native-packed-matmul-kernel-ok", kernel)
PY
  stage native-packed-matmul-load-done
  stage native-kernel-package-ci-done
fi

stage native-packed-matmul-bench-start
orbitquant kernel-bench \
  --tokens "$TOKENS" \
  --in-features "$IN_FEATURES" \
  --out-features "$OUT_FEATURES" \
  --weight-bits 4 \
  --activation-bits 4 \
  --block-size "$BLOCK_SIZE" \
  --activation-kernel-backend triton_cuda \
  --runtime-mode native_packed_matmul \
  --packed-matmul-block-m "$PACKED_MATMUL_BLOCK_M" \
  --packed-matmul-block-n "$PACKED_MATMUL_BLOCK_N" \
  --packed-matmul-block-k "$PACKED_MATMUL_BLOCK_K" \
  --packed-matmul-num-warps "$PACKED_MATMUL_NUM_WARPS" \
  --device cuda \
  --dtype bfloat16 \
  --warmup "$WARMUP" \
  --iterations "$ITERATIONS"
stage native-packed-matmul-bench-done
