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
  "tests/test_paper_methodology.py"
  "tests/test_documentation.py"
  "tests/test_linear_adapters.py"
  "tests/test_universal_transformers.py"
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
import hashlib
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
expected_counts = {
    "flux2-native": {
        "linear_module_count": 109,
        "orbitquant": 100,
        "adaln_int4_rtn": 3,
        "bf16_skip": 6,
    },
    "flux1-schnell-native": {
        "linear_module_count": 502,
        "orbitquant": 418,
        "adaln_int4_rtn": 76,
        "bf16_skip": 8,
    },
    "z-image-native": {
        "linear_module_count": 276,
        "orbitquant": 238,
        "adaln_int4_rtn": 32,
        "bf16_skip": 6,
    },
    "wan-native": {
        "linear_module_count": 306,
        "orbitquant": 300,
        "adaln_int4_rtn": 0,
        "bf16_skip": 6,
    },
}
expected_module_list_hashes = {
    "flux2-native": {
        "quantized_modules": "6d79a8f00fa1a15e9ec2800d4568860853347f998eddb03c623c4be62482bbfb",
        "adaln_modules": "e1605750dfcc3e7d6821ff951ff94077380c02dc84f55ead620f22c3fae58473",
        "skipped_modules": "c3834abafcaf954a69e31ad64cb98436aeaf16ff36a5f6e382933b63bfb06f54",
    },
    "flux1-schnell-native": {
        "quantized_modules": "7327a220588353bbaf8b5697326b278da40c8c79f617479fc55e55c6966f2563",
        "adaln_modules": "d8f0d8f0716341fc7b618c0cd87e2ef1f8ec699d1071f47922150b54a4d497bd",
        "skipped_modules": "3980cd68701b61ea2bd8af052f78014c7f6d4d63a16012664917ffca46b1f0e3",
    },
    "z-image-native": {
        "quantized_modules": "3d43a43b2cd1369e10d3bc88bca11ef9960d8189d101dedaadfd79242539b526",
        "adaln_modules": "e9cb615912b15212a065255dc35d08559b4277f2e381287107dbe35e6e4e4a59",
        "skipped_modules": "e09dcfc87e2f64c3f620c6381b865c95594f80d31a73e3f8faea16a07f056a1a",
    },
    "wan-native": {
        "quantized_modules": "3110e1ac320e60bc85f1e0556a982c222af93e604a624bce24ff1cd6efa69c00",
        "adaln_modules": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
        "skipped_modules": "f0a09fe0938a413a1321bf49bb9de450f3232ca4b77e989c1c5b037e0ff8fec5",
    },
}

def module_list_hash(values):
    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()

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
    suite_expected_counts = expected_counts[suite]
    if data["linear_module_count"] != suite_expected_counts["linear_module_count"]:
        raise SystemExit(
            f"{path}: expected {suite_expected_counts['linear_module_count']} linears, "
            f"got {data['linear_module_count']}"
        )
    for action, expected_count in suite_expected_counts.items():
        if action == "linear_module_count":
            continue
        actual_count = action_counts.get(action, 0)
        if actual_count != expected_count:
            raise SystemExit(
                f"{path}: expected {expected_count} {action} modules, got {actual_count}"
            )
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
    for list_name, expected_hash in expected_module_list_hashes[suite].items():
        actual_hash = module_list_hash(data.get(list_name, []))
        if actual_hash != expected_hash:
            raise SystemExit(
                f"{path}: {list_name} changed; expected sha256 {expected_hash}, "
                f"got {actual_hash}"
            )
    if suite == "wan-native" and adaln_module_count != 0:
        raise SystemExit(f"{path}: Wan should not classify AdaLN INT4 modules")
    if suite != "wan-native" and adaln_module_count <= 0:
        raise SystemExit(f"{path}: expected AdaLN INT4 modules")

    summary["suites"][suite] = {
        "component_class": data["component_class"],
        "linear_module_count": data["linear_module_count"],
        "action_counts": action_counts,
        "module_list_hashes": {
            key: module_list_hash(data[key])
            for key in ("quantized_modules", "adaln_modules", "skipped_modules")
        },
    }

summary_path.parent.mkdir(parents=True, exist_ok=True)
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, sort_keys=True))
PY
stage inventory-contract-done

stage done
