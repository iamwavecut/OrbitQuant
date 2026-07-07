#!/usr/bin/env bash
set -euo pipefail

stage() {
  printf 'PAPER_METHOD_STAGE %s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

usage() {
  cat <<'EOF'
Usage: scripts/run_paper_methodology_checks.sh

Runs the lightweight OrbitQuant paper-methodology gate. The gate checks the
math, artifact, native-settings, and model-policy invariants that support the
methodology audit. It does not run GenEval, VBench, model generation, or weight
downloads.

Environment:
  ORBITQUANT_PAPER_AUDIT_DIR     Output directory for local audit JSON files.
  ORBITQUANT_PAPER_AUDIT_SUITES  Space-separated native suites to inventory.
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORT_DIR="${ORBITQUANT_PAPER_AUDIT_DIR:-"$ROOT_DIR/reports/paper-methodology"}"
SUITES="${ORBITQUANT_PAPER_AUDIT_SUITES:-flux2-native flux1-schnell-native z-image-native wan-native}"

while [[ $# -gt 0 ]]; do
  case "$1" in
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

if command -v uv >/dev/null 2>&1; then
  RUN=(uv run)
  PYTHON=(uv run python)
else
  RUN=()
  PYTHON=("${PYTHON_BIN:-python3}")
fi

METHODOLOGY_TESTS=(
  "tests/test_codebooks.py"
  "tests/test_rpbh.py"
  "tests/test_bitpack.py"
  "tests/test_orbit_linear.py"
  "tests/test_adaln_rtn.py"
  "tests/test_config.py"
  "tests/test_manifest.py"
  "tests/test_artifact_writer.py"
  "tests/test_native_settings.py"
  "tests/test_target_policies.py"
  "tests/test_kernels.py"
  "tests/test_release_gates.py"
)

cd "$ROOT_DIR"

stage tests-start
"${RUN[@]}" pytest -q "${METHODOLOGY_TESTS[@]}"
stage tests-done

INVENTORY_DIR="$REPORT_DIR/module-inventories"
SUMMARY_PATH="$REPORT_DIR/paper-methodology-summary.json"
mkdir -p "$INVENTORY_DIR"

stage inventories-start
for suite in $SUITES; do
  "${RUN[@]}" orbitquant inspect-policy \
    --suite "$suite" \
    --load-mode config \
    --dtype bfloat16 \
    --output "$INVENTORY_DIR/$suite-policy.json"
done
stage inventories-done

stage inventory-contract-start
"${PYTHON[@]}" - "$INVENTORY_DIR" "$SUMMARY_PATH" $SUITES <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

inventory_dir = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
suites = sys.argv[3:]

expected_component_classes = {
    "flux2-native": "Flux2Transformer2DModel",
    "flux1-schnell-native": "FluxTransformer2DModel",
    "z-image-native": "ZImageTransformer2DModel",
    "wan-native": "WanTransformer3DModel",
}
expected_policies = {
    "flux2-native": "flux2",
    "flux1-schnell-native": "flux",
    "z-image-native": "z_image",
    "wan-native": "wan",
}

summary = {"suites": {}, "claim_boundary": "config_inventory_only"}
for suite in suites:
    path = inventory_dir / f"{suite}-policy.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    action_counts = data["action_counts"]
    total_actions = sum(action_counts.values())
    quantized_module_count = len(data["quantized_modules"])
    adaln_module_count = len(data["adaln_modules"])
    skipped_module_count = len(data["skipped_modules"])

    if data["suite"] != suite:
        raise SystemExit(f"{path}: suite mismatch")
    if data["load_mode"] != "config":
        raise SystemExit(f"{path}: expected config load mode")
    if data["component"] != "transformer":
        raise SystemExit(f"{path}: expected transformer component")
    if data["component_class"] != expected_component_classes[suite]:
        raise SystemExit(f"{path}: unexpected component class {data['component_class']!r}")
    if data["target_policy"] != expected_policies[suite]:
        raise SystemExit(f"{path}: unexpected target policy {data['target_policy']!r}")
    if data["linear_module_count"] != total_actions:
        raise SystemExit(f"{path}: action counts do not cover all linears")
    if quantized_module_count != action_counts["orbitquant"]:
        raise SystemExit(f"{path}: quantized module list/count mismatch")
    if adaln_module_count != action_counts["adaln_int4_rtn"]:
        raise SystemExit(f"{path}: AdaLN module list/count mismatch")
    if skipped_module_count != action_counts["bf16_skip"]:
        raise SystemExit(f"{path}: skipped module list/count mismatch")
    if quantized_module_count <= 0:
        raise SystemExit(f"{path}: no OrbitQuant modules selected")
    if skipped_module_count <= 0:
        raise SystemExit(f"{path}: no BF16 skips selected")
    if suite == "wan-native" and adaln_module_count != 0:
        raise SystemExit(f"{path}: Wan should not classify AdaLN INT4 modules")
    if suite != "wan-native" and adaln_module_count <= 0:
        raise SystemExit(f"{path}: expected AdaLN INT4 modules")

    summary["suites"][suite] = {
        "component_class": data["component_class"],
        "linear_module_count": data["linear_module_count"],
        "action_counts": action_counts,
    }

summary_path.parent.mkdir(parents=True, exist_ok=True)
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, sort_keys=True))
PY
stage inventory-contract-done

stage done
