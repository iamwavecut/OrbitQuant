# OrbitQuant 0.1.2 Release Notes

OrbitQuant 0.1.2 is a patch release for compact artifact-audit reporting.

## Changes

- `orbitquant audit-hf-artifacts` now accepts `--summary-only`.
- The default audit output remains unchanged. With `--summary-only`, the command
  prints and writes aggregate counts plus per-repository readiness fields,
  omitting the full remote row payload.
- `--fail-on-artifact-regression` still evaluates the full audit payload, so
  compact output does not weaken the artifact-readiness gate.

## Usage

```bash
orbitquant audit-hf-artifacts \
  --namespace WaveCut \
  --policy-inventory-root ./reports/native/module-inventories \
  --summary-only \
  --fail-on-artifact-regression
```

## Claim Boundary

This release does not change quantization math, artifact format, runtime kernel
dispatch, or model cards. It only adds a compact reporting mode for the Hugging
Face artifact audit CLI.
