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
RUN_NATIVE_KERNEL_PACKAGE_CI="${ORBITQUANT_RUN_NATIVE_KERNEL_PACKAGE_CI:-1}"
NATIVE_KERNEL_REPO_ID="WaveCut/orbitquant-packed-matmul"
NATIVE_KERNEL_SOURCE_DIR="$ROOT_DIR/native-kernels/orbitquant-packed-matmul"

native_kernel_local_variant_dir() {
  python - "$NATIVE_KERNEL_SOURCE_DIR" <<'PY'
import platform
import re
import sys
from pathlib import Path

import torch

source_dir = Path(sys.argv[1])
torch_match = re.match(r"^(\d+)\.(\d+)", torch.__version__)
if torch_match is None:
    raise SystemExit(f"could not parse torch version from {torch.__version__!r}")
if sys.platform != "darwin":
    raise SystemExit(f"MPS kernel package CI expects macOS, got {sys.platform}")
machine = platform.machine()
if machine == "arm64":
    machine = "aarch64"
variant = (
    f"torch{torch_match.group(1)}{torch_match.group(2)}-metal-"
    f"{machine}-darwin"
)
variant_dir = source_dir / "build" / variant
metadata_path = variant_dir / "metadata.json"
if not metadata_path.is_file():
    raise SystemExit(f"native kernel variant metadata is missing: {metadata_path}")
print(variant_dir)
PY
}

cd "$ROOT_DIR"

stage env-start
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$PYTHON_BIN" -m venv --system-site-packages "$VENV_DIR"
fi

# shellcheck disable=SC1091
. "$VENV_DIR/bin/activate"

if command -v uv >/dev/null 2>&1; then
  uv pip install hatchling numpy safetensors huggingface_hub packaging tqdm pytest ruff "kernels>=0.16"
  uv pip install --no-deps -e .
else
  python -m pip install hatchling numpy safetensors huggingface_hub packaging tqdm pytest ruff "kernels>=0.16"
  python -m pip install --no-deps -e .
fi

python - <<'PY'
import torch

if not torch.backends.mps.is_available():
    raise SystemExit("MPS is not available in this Python environment")
if not hasattr(torch.mps, "compile_shader"):
    raise SystemExit("MPS Metal compile_shader is not available in this Python environment")

print("python-env-ok")
print("torch", torch.__version__)
print("mps", torch.backends.mps.is_available())
print("compile_shader", hasattr(torch.mps, "compile_shader"))
PY
stage env-done

if [[ "$RUN_NATIVE_KERNEL_PACKAGE_CI" == "1" && -z "${LOCAL_KERNELS:-}" ]]; then
  set +e
  native_kernel_variant_dir="$(native_kernel_local_variant_dir)"
  native_kernel_variant_status=$?
  set -e
  if [[ "$native_kernel_variant_status" -eq 0 ]]; then
    export LOCAL_KERNELS="$NATIVE_KERNEL_REPO_ID=$native_kernel_variant_dir"
    stage "native-packed-matmul-local-variant-selected variant=$(basename "$native_kernel_variant_dir")"
  else
    stage "native-packed-matmul-local-variant-missing status=$native_kernel_variant_status"
    printf '%s\n' \
      "native packed matmul local variant was not found for this MPS runtime." \
      "Make a compatible orbitquant_packed_matmul package importable, set LOCAL_KERNELS=$NATIVE_KERNEL_REPO_ID=/absolute/path/to/a/built/variant containing metadata.json, build the matching kernel-builder variant, or set ORBITQUANT_RUN_NATIVE_KERNEL_PACKAGE_CI=0 to run only the inline Metal shader checks." \
      >&2
  fi
fi

if [[ "$RUN_NATIVE_KERNEL_PACKAGE_CI" == "1" ]]; then
  stage native-packed-matmul-load-start
  python - <<'PY'
from orbitquant.kernels.native_packed_matmul import load_native_packed_matmul_kernel

kernel = load_native_packed_matmul_kernel()
if not hasattr(kernel, "matmul_packed_weight"):
    raise SystemExit("native packed matmul kernel is missing matmul_packed_weight")
print("native-packed-matmul-kernel-ok", kernel)
PY
  stage native-packed-matmul-load-done
else
  stage native-packed-matmul-load-skipped
fi

stage kernel-tests-start
pytest tests/test_kernels.py tests/test_orbit_linear.py -q -k 'mps or backend_capabilities'
stage kernel-tests-done

stage kernel-info-start
orbitquant kernel-info
python - <<'PY'
import os

from orbitquant.kernels import backend_capabilities

capability = backend_capabilities()["mps"]
required_stages = {"activation_norm_rpbh_quant_rescale", "packed_weight_dequant"}
if os.environ.get("ORBITQUANT_RUN_NATIVE_KERNEL_PACKAGE_CI", "1") == "1":
    required_stages.add("packed_weight_matmul")
stages = set(str(capability["optimized_stage"] or "").split(","))
missing = sorted(required_stages - stages)
if capability["claim_status"] != "partial_optimized":
    raise SystemExit(f"unexpected mps claim_status: {capability['claim_status']}")
if not capability["optimized"]:
    raise SystemExit("mps backend is not optimized in this environment")
if capability["full_fusion"]:
    raise SystemExit("mps should not claim full_fusion")
if capability["hf_kernel_builder_compliant"]:
    raise SystemExit("mps should not claim HF kernel-builder compliance")
if capability["upstream_native_mps_op"]:
    raise SystemExit("mps should not claim an upstream native op")
if missing:
    raise SystemExit(f"mps optimized_stage missing: {missing}")
print("mps-kernel-contract-ok")
PY
stage kernel-info-done

BENCH_RUNTIME_ARGS=()
if [[ "$RUN_NATIVE_KERNEL_PACKAGE_CI" != "1" ]]; then
  BENCH_RUNTIME_ARGS=(--runtime-mode dequant_bf16)
fi

stage kernel-bench-start
orbitquant kernel-bench \
  --tokens "$TOKENS" \
  --in-features "$IN_FEATURES" \
  --out-features "$OUT_FEATURES" \
  --weight-bits 4 \
  --activation-bits 4 \
  --block-size "$BLOCK_SIZE" \
  --activation-kernel-backend mps \
  ${BENCH_RUNTIME_ARGS[@]+"${BENCH_RUNTIME_ARGS[@]}"} \
  --device mps \
  --dtype bfloat16 \
  --warmup "$WARMUP" \
  --iterations "$ITERATIONS"
stage kernel-bench-done

if [[ "$RUN_NATIVE_KERNEL_PACKAGE_CI" == "1" ]]; then
  stage native-packed-matmul-bench-start
  orbitquant kernel-bench \
    --tokens "$TOKENS" \
    --in-features "$IN_FEATURES" \
    --out-features "$OUT_FEATURES" \
    --weight-bits 4 \
    --activation-bits 4 \
    --block-size "$BLOCK_SIZE" \
    --activation-kernel-backend mps \
    --runtime-mode native_packed_matmul \
    --device mps \
    --dtype bfloat16 \
    --warmup "$WARMUP" \
    --iterations "$ITERATIONS"
  stage native-packed-matmul-bench-done
else
  stage native-packed-matmul-bench-skipped
fi
