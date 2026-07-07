#!/usr/bin/env bash
set -euo pipefail

stage() {
  printf 'REMOTE_STAGE %s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${ORBITQUANT_VENV:-"$ROOT_DIR/.venv"}"

TOKENS="${ORBITQUANT_BENCH_TOKENS:-64}"
IN_FEATURES="${ORBITQUANT_BENCH_IN_FEATURES:-1024}"
OUT_FEATURES="${ORBITQUANT_BENCH_OUT_FEATURES:-1024}"
BLOCK_SIZE="${ORBITQUANT_BENCH_BLOCK_SIZE:-1024}"
WARMUP="${ORBITQUANT_BENCH_WARMUP:-1}"
ITERATIONS="${ORBITQUANT_BENCH_ITERATIONS:-3}"

cd "$ROOT_DIR"

stage env-start
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$PYTHON_BIN" -m venv --system-site-packages "$VENV_DIR"
fi

# shellcheck disable=SC1091
. "$VENV_DIR/bin/activate"

if command -v uv >/dev/null 2>&1; then
  uv pip install hatchling numpy safetensors huggingface_hub packaging tqdm pytest ruff
  uv pip install --no-deps -e .
else
  python -m pip install hatchling numpy safetensors huggingface_hub packaging tqdm pytest ruff
  python -m pip install --no-deps -e .
fi

python - <<'PY'
import torch

if not torch.backends.mps.is_available():
    raise SystemExit("MPS is not available in this Python environment")

print("python-env-ok")
print("torch", torch.__version__)
print("mps", torch.backends.mps.is_available())
print("compile_shader", hasattr(torch.mps, "compile_shader"))
PY
stage env-done

stage kernel-tests-start
pytest tests/test_kernels.py tests/test_orbit_linear.py -q -k 'mps or backend_capabilities'
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
  --activation-kernel-backend mps \
  --device mps \
  --dtype float32 \
  --warmup "$WARMUP" \
  --iterations "$ITERATIONS"
stage kernel-bench-done
