# OrbitQuant 0.1.0 Publication Checklist

This checklist is for the public repository, GitHub release, and PyPI package
publication step. PyPI `orbitquant==0.1.0` is already published through Trusted
Publishing, and the GitHub repository is already public. Tag creation and
GitHub release creation remain gated by explicit approval.

## Preconditions

- `main` is clean and matches `origin/main`.
- The release notes in `docs/release-0.1.0.md` match the package version.
- The release gates in `docs/release-gates.md` are current.
- The build artifacts are freshly produced from the release commit.
- PyPI Trusted Publishing is configured for
  `iamwavecut/OrbitQuant/.github/workflows/publish-pypi.yml` with environment
  `pypi`.

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
python - <<'PY'
import json
import urllib.request

with urllib.request.urlopen("https://pypi.org/pypi/orbitquant/json", timeout=20) as response:
    payload = json.load(response)

assert payload["info"]["version"] == "0.1.0"
filenames = {file["filename"] for file in payload["releases"]["0.1.0"]}
assert "orbitquant-0.1.0.tar.gz" in filenames
assert "orbitquant-0.1.0-py3-none-any.whl" in filenames
PY
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

## Publish GitHub Release

Run only after explicit approval. Repository visibility is already public, so
this section only creates the version tag and GitHub release:

```bash
cd /Users/Shared/src/github.com/iamwavecut/OrbitQuant
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

PyPI `orbitquant==0.1.0` was published with GitHub Actions Trusted Publishing,
not a local token:

```bash
cd /Users/Shared/src/github.com/iamwavecut/OrbitQuant
gh workflow run publish-pypi.yml --repo iamwavecut/OrbitQuant --ref main -f version=0.1.0
gh run watch 29015072821 --repo iamwavecut/OrbitQuant --exit-status
```

## Post-Publication Checks

```bash
gh repo view iamwavecut/OrbitQuant --json visibility,isPrivate,url,defaultBranchRef
test "$(gh repo view iamwavecut/OrbitQuant --json isPrivate --jq .isPrivate)" = "false"
gh release view v0.1.0 --repo iamwavecut/OrbitQuant
python - <<'PY'
import json
import urllib.request

with urllib.request.urlopen("https://pypi.org/pypi/orbitquant/json", timeout=20) as response:
    payload = json.load(response)

assert payload["info"]["version"] == "0.1.0"
filenames = {file["filename"] for file in payload["releases"]["0.1.0"]}
assert "orbitquant-0.1.0.tar.gz" in filenames
assert "orbitquant-0.1.0-py3-none-any.whl" in filenames
PY
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
