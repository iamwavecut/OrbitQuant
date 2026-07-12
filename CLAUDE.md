# CLAUDE.md

Guidance for coding agents working in this repository (OrbitQuant: calibration-free
post-training quantizer for transformer linear projections).

## Environment

```bash
uv sync --extra hf --extra dev
```

- Tests: `uv run pytest -ra`
- Lint: `uv run ruff check .` (line-length 100)

## Formatting

- NEVER mass-reformat. `ruff format .` is forbidden: the tree has historical
  formatting drift and a blanket reformat produces unreviewable diffs.
- Format only the lines you touch.

## Gates Before Claiming Done

Run all of these and confirm they pass:

```bash
uv run ruff check .
uv run pytest -ra
scripts/run_paper_methodology_checks.sh
scripts/run_hf_compat_checks.sh --mode all
```

Backend gates (run on matching hardware only):

```bash
scripts/run_cuda_kernel_checks.sh                          # CUDA host
PYTHON_BIN="$(uv python find)" scripts/run_mps_kernel_checks.sh   # Apple Silicon
```

The MPS gate requires bash >= 4; on macOS use `/opt/homebrew/bin/bash`.

## Native Kernels

The native package lives in `native-kernels/orbitquant-packed-matmul` and uses the
Hugging Face kernel-builder, pinned to rev
`d43de01d0b43285d8e5061ca4380c2bd1c40ae3b` (see `.github/workflows/ci.yml`).

Local build:

```bash
cd native-kernels/orbitquant-packed-matmul
kernel-builder create-pyproject -f .
CMAKE_ARGS=-DGPU_LANG=CPU python setup.py build_kernel
```

or via Nix:

```bash
nix --option sandbox relaxed run .#build-and-copy -L
```

Build artifacts inside the kernel directory are GENERATED — never commit them:
`CMakeLists.txt`, `cmake/`, `compat.py`, `metadata-*.json`, `pyproject.toml`,
`setup.py`, `build/`, `torch-ext/registration.h`,
`torch-ext/orbitquant_packed_matmul/_ops.py`, and `uv.lock` inside the kernel
directory. Delete them before running the repo test suite: two ABI3 lint tests
scan the kernel directory and fail on leftover generated files.

## Documentation Policy

- Docs describe the current artifact and user contract only. No development
  chronicles, no raw logs.
- The source of truth for backend status is
  `orbitquant.kernels.dispatch.backend_capabilities()`; the audited backend
  contract is `docs/kernel-audit.md`.

## Hard Boundaries

- Hugging Face Kernel Hub publication is OUT OF SCOPE. Never post there.
- Do not commit, push, or publish anything without explicit operator instruction.
