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
  uv pip install hatchling numpy safetensors huggingface_hub packaging tqdm pytest pytest-xdist ruff
  uv pip install --no-deps -e .
else
  python -m pip install hatchling numpy safetensors huggingface_hub packaging tqdm pytest pytest-xdist ruff
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
pytest tests/test_kernels.py tests/test_adaln_rtn.py -q
stage kernel-tests-done

stage kernel-info-start
orbitquant kernel-info
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

stage packed-matmul-bench-start
orbitquant kernel-bench \
  --tokens "$TOKENS" \
  --in-features "$IN_FEATURES" \
  --out-features "$OUT_FEATURES" \
  --weight-bits 4 \
  --activation-bits 4 \
  --block-size "$BLOCK_SIZE" \
  --activation-kernel-backend triton_cuda \
  --runtime-mode triton_packed_matmul \
  --packed-matmul-block-m "$PACKED_MATMUL_BLOCK_M" \
  --packed-matmul-block-n "$PACKED_MATMUL_BLOCK_N" \
  --packed-matmul-block-k "$PACKED_MATMUL_BLOCK_K" \
  --packed-matmul-num-warps "$PACKED_MATMUL_NUM_WARPS" \
  --device cuda \
  --dtype bfloat16 \
  --warmup "$WARMUP" \
  --iterations "$ITERATIONS"
stage packed-matmul-bench-done

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
