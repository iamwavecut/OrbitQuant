#!/usr/bin/env bash
set -euo pipefail

stage() {
  printf 'HF_COMPAT_STAGE %s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

usage() {
  cat <<'EOF'
Usage: scripts/run_hf_compat_checks.sh [--mode current|release|dev|all]

Runs the lightweight Hugging Face compatibility gate without downloading model
weights or generating samples.

Modes:
  current  Use the active project environment.
  release  Create an HF overlay venv with published Diffusers/Transformers releases.
  dev      Create an HF overlay venv with Diffusers/Transformers from GitHub main.
  all      Run current, release, then dev.
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_ROOT="${ORBITQUANT_HF_COMPAT_VENV_ROOT:-"$ROOT_DIR/.venvs/hf-compat"}"
MODE="current"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$MODE" != "current" && "$MODE" != "release" && "$MODE" != "dev" && "$MODE" != "all" ]]; then
  echo "Unsupported mode: $MODE" >&2
  usage >&2
  exit 2
fi

DEFAULT_TESTS=(
  "tests/test_quantizer_adapter.py"
  "tests/test_pipeline_helpers.py"
  "tests/test_diffusers_modelmixin_integration.py"
  "tests/test_transformers_pretrained_integration.py"
)

compat_tests=("${DEFAULT_TESTS[@]}")
if [[ -n "${ORBITQUANT_HF_COMPAT_TESTS:-}" ]]; then
  # shellcheck disable=SC2206
  compat_tests=(${ORBITQUANT_HF_COMPAT_TESTS})
fi

run_python_compat_check() {
  "$@" <<'PY'
import importlib.metadata as metadata
import json

import torch

import orbitquant
from orbitquant import (
    OrbitQuantConfig,
    build_diffusers_pipeline_quantization_config,
)
from orbitquant.quantizer import OrbitQuantConfig as AdapterConfig
from orbitquant.quantizer import OrbitQuantizer, register_hf_quantizers

versions = {}
for package in (
    "orbitquant",
    "torch",
    "diffusers",
    "transformers",
    "accelerate",
    "huggingface_hub",
):
    try:
        versions[package] = metadata.version(package)
    except metadata.PackageNotFoundError:
        versions[package] = "not-installed"

print(json.dumps({"hf_compat_versions": versions}, sort_keys=True))

result = register_hf_quantizers()
assert result["diffusers"], "Diffusers quantizer registration failed"
assert result["transformers"], "Transformers quantizer registration failed"

import diffusers.quantizers.auto as diffusers_auto
import transformers.quantizers.auto as transformers_auto

assert diffusers_auto.AUTO_QUANTIZER_MAPPING["orbitquant"] is OrbitQuantizer
assert diffusers_auto.AUTO_QUANTIZATION_CONFIG_MAPPING["orbitquant"] is AdapterConfig
assert transformers_auto.AUTO_QUANTIZER_MAPPING["orbitquant"] is OrbitQuantizer
assert transformers_auto.AUTO_QUANTIZATION_CONFIG_MAPPING["orbitquant"] is AdapterConfig

config = OrbitQuantConfig(
    weight_bits=4,
    activation_bits=4,
    block_size=4,
    target_policy="generic_dit",
)

granular = build_diffusers_pipeline_quantization_config(
    config,
    components=["transformer", "denoiser"],
)
assert granular.is_granular is True
assert set(granular.quant_mapping) == {"transformer", "denoiser"}
resolved = granular._resolve_quant_config(is_diffusers=True, module_name="transformer")
assert resolved.to_dict() == config.to_dict()
assert granular._resolve_quant_config(is_diffusers=True, module_name="text_encoder") is None

backend = build_diffusers_pipeline_quantization_config(
    config,
    components="transformer",
    granular=False,
)
assert backend.is_granular is False
assert backend.quant_backend == "orbitquant"
assert backend.components_to_quantize == ["transformer"]
assert backend._resolve_quant_config(
    is_diffusers=True,
    module_name="transformer",
).to_dict() == config.to_dict()

quantizer = OrbitQuantizer(config)
assert quantizer.requires_parameters_quantization is True
assert quantizer.requires_calibration is False
assert quantizer.is_serializable() is True
assert quantizer.update_torch_dtype(torch.bfloat16) is torch.bfloat16

print("hf-compat-contract-ok")
PY
}

install_with_python() {
  local python_bin="$1"
  shift
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$python_bin" "$@"
  else
    "$python_bin" -m pip install "$@"
  fi
}

run_current() {
  stage current-start
  cd "$ROOT_DIR"
  if command -v uv >/dev/null 2>&1; then
    stage current-contract-start
    run_python_compat_check uv run python
    stage current-contract-done
    stage current-tests-start
    uv run pytest -q "${compat_tests[@]}"
    stage current-tests-done
  else
    stage current-contract-start
    run_python_compat_check python
    stage current-contract-done
    stage current-tests-start
    pytest -q "${compat_tests[@]}"
    stage current-tests-done
  fi
  stage current-done
}

prepare_venv() {
  local mode="$1"
  local venv_dir="$VENV_ROOT/$mode"
  local python_path="$venv_dir/bin/python"

  stage "$mode-env-start"
  mkdir -p "$VENV_ROOT"
  if [[ ! -x "$python_path" ]]; then
    # Reuse the current Torch base so release/dev framework compatibility can be
    # checked without repeatedly downloading large backend wheels.
    "$PYTHON_BIN" -m venv --system-site-packages "$venv_dir"
  fi

  if [[ "$mode" == "release" ]]; then
    install_with_python \
      "$python_path" \
      "diffusers>=0.35" \
      "transformers>=4.53" \
      "accelerate>=1.8" \
      "huggingface_hub>=0.33" \
      "safetensors>=0.5" \
      "packaging>=24.0" \
      "numpy>=2.0" \
      "pytest>=8.4"
  elif [[ "$mode" == "dev" ]]; then
    install_with_python \
      "$python_path" \
      "diffusers @ git+https://github.com/huggingface/diffusers.git" \
      "transformers @ git+https://github.com/huggingface/transformers.git" \
      "accelerate>=1.8" \
      "huggingface_hub>=0.33" \
      "safetensors>=0.5" \
      "packaging>=24.0" \
      "numpy>=2.0" \
      "pytest>=8.4"
  else
    echo "prepare_venv received unsupported mode: $mode" >&2
    exit 2
  fi

  install_with_python "$python_path" --no-deps -e "$ROOT_DIR"
  stage "$mode-env-done"
  PREPARED_PYTHON_PATH="$python_path"
}

run_isolated_mode() {
  local mode="$1"

  stage "$mode-start"
  PREPARED_PYTHON_PATH=""
  prepare_venv "$mode"
  cd "$ROOT_DIR"
  stage "$mode-contract-start"
  run_python_compat_check "$PREPARED_PYTHON_PATH"
  stage "$mode-contract-done"
  stage "$mode-tests-start"
  "$PREPARED_PYTHON_PATH" -m pytest -q "${compat_tests[@]}"
  stage "$mode-tests-done"
  stage "$mode-done"
}

case "$MODE" in
  current)
    run_current
    ;;
  release)
    run_isolated_mode release
    ;;
  dev)
    run_isolated_mode dev
    ;;
  all)
    run_current
    run_isolated_mode release
    run_isolated_mode dev
    ;;
esac
