# OrbitQuant 0.1.0 Publication Checklist

This checklist is for the final public repository, GitHub release, and PyPI
package publication step. Do not run the publication commands until repository
visibility, tag creation, GitHub release creation, and PyPI upload are
explicitly approved.

## Preconditions

- `main` is clean and matches `origin/main`.
- The release notes in `docs/release-0.1.0.md` match the package version.
- The release gates in `docs/release-gates.md` are current.
- The build artifacts are freshly produced from the release commit.
- PyPI upload credentials are available as a token and are not written to disk,
  committed, printed, or pasted into logs.

## Preflight

```bash
git status --short --branch
git fetch origin main --tags
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)"
test "$(python - <<'PY'
import tomllib
print(tomllib.loads(open("pyproject.toml", "rb").read().decode())["project"]["version"])
PY
)" = "0.1.0"
test "$(git tag --list v0.1.0)" = ""
gh repo view iamwavecut/OrbitQuant --json nameWithOwner,visibility,isPrivate,defaultBranchRef
```

## Verification

```bash
rm -rf dist
uv run pytest -q
uv run ruff check .
uv run --with build python -m build
uv run --with twine python -m twine check dist/*
cd /tmp
uv run --with /Users/Shared/src/github.com/iamwavecut/OrbitQuant/dist/orbitquant-0.1.0-py3-none-any.whl \
  python - <<'PY'
import json
import subprocess
import sys

import orbitquant
from orbitquant import OrbitQuantConfig

config = OrbitQuantConfig()
payload = json.loads(config.to_json_string())
assert orbitquant.__version__ == "0.1.0"
assert config.runtime_mode == "auto_fused"
assert payload["runtime_mode"] == "auto_fused"
assert subprocess.check_output(
    [sys.executable, "-m", "orbitquant.cli.main", "--version"],
    text=True,
).strip() == "0.1.0"
PY
```

## Publish GitHub

Run only after explicit approval:

```bash
cd /Users/Shared/src/github.com/iamwavecut/OrbitQuant
gh repo edit iamwavecut/OrbitQuant \
  --visibility public \
  --accept-visibility-change-consequences
git tag -a v0.1.0 -m "OrbitQuant 0.1.0"
git push origin v0.1.0
gh release create v0.1.0 \
  dist/orbitquant-0.1.0.tar.gz \
  dist/orbitquant-0.1.0-py3-none-any.whl \
  --repo iamwavecut/OrbitQuant \
  --verify-tag \
  --title "OrbitQuant 0.1.0" \
  --notes-file docs/release-0.1.0.md
```

## Publish PyPI

Run only after explicit approval and with a PyPI API token available in the
environment:

```bash
cd /Users/Shared/src/github.com/iamwavecut/OrbitQuant
TWINE_USERNAME=__token__ \
TWINE_PASSWORD="$PYPI_API_TOKEN" \
uv run --with twine python -m twine upload dist/*
```

## Post-Publication Checks

```bash
gh repo view iamwavecut/OrbitQuant --json visibility,isPrivate,url,defaultBranchRef
gh release view v0.1.0 --repo iamwavecut/OrbitQuant
python -m pip index versions orbitquant
python -m pip install --upgrade orbitquant
python - <<'PY'
import orbitquant
from orbitquant import OrbitQuantConfig

assert orbitquant.__version__ == "0.1.0"
assert OrbitQuantConfig().runtime_mode == "auto_fused"
print("orbitquant-publication-ok")
PY
```
