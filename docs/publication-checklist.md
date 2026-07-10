# OrbitQuant Publication Checklist

Use this checklist to publish a tagged Python release from `main`.

## Preconditions

- `main` is clean and matches `origin/main`.
- `pyproject.toml`, `src/orbitquant/__init__.py`, `uv.lock`, release notes, and
  the PyPI workflow input use the same version.
- [release-gates.md](release-gates.md) is satisfied.
- PyPI Trusted Publishing is configured for
  `iamwavecut/OrbitQuant/.github/workflows/publish-pypi.yml` and environment
  `pypi`.

Set the release version once:

```bash
export VERSION="0.1.6"
export TAG="v${VERSION}"
```

## Verify Source

```bash
git fetch origin main --tags
test -z "$(git status --porcelain)"
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)"
test "$(python - <<'PY'
import tomllib
with open('pyproject.toml', 'rb') as file:
    print(tomllib.load(file)['project']['version'])
PY
)" = "$VERSION"
uv run pytest -q
uv run ruff check .
scripts/run_paper_methodology_checks.sh
scripts/run_hf_compat_checks.sh --mode all
```

## Build

```bash
rm -rf dist
uv build
uvx twine check dist/*
```

Install the wheel into a clean environment and verify the public defaults:

```bash
rm -rf /tmp/orbitquant-release-venv
uv venv /tmp/orbitquant-release-venv
uv pip install --python /tmp/orbitquant-release-venv/bin/python dist/*.whl
/tmp/orbitquant-release-venv/bin/python - <<'PY'
import orbitquant
from orbitquant import OrbitQuantConfig

config = OrbitQuantConfig()
assert config.runtime_mode == 'auto_fused'
assert config.codebook_version == 2
print(orbitquant.__version__)
PY
```

## Publish PyPI

Always pass the version explicitly:

```bash
gh workflow run publish-pypi.yml --ref main -f version="$VERSION"
RUN_ID="$(gh run list --workflow publish-pypi.yml --limit 1 --json databaseId --jq '.[0].databaseId')"
gh run watch "$RUN_ID" --exit-status
```

Verify the public package from a clean environment with the cache disabled:

```bash
rm -rf /tmp/orbitquant-pypi-venv
uv venv /tmp/orbitquant-pypi-venv
UV_NO_CACHE=1 uv pip install \
  --python /tmp/orbitquant-pypi-venv/bin/python \
  --index-url https://pypi.org/simple \
  "orbitquant==$VERSION"
/tmp/orbitquant-pypi-venv/bin/python -c \
  "import orbitquant; assert orbitquant.__version__ == '$VERSION'"
```

## Publish GitHub Release

Attach the exact files served by PyPI, not a second local build. Verify their
SHA256 digests against the PyPI JSON response before upload.

```bash
git tag -a "$TAG" -m "OrbitQuant $VERSION" HEAD
git push origin "$TAG"
gh release create "$TAG" /tmp/orbitquant-pypi-"$VERSION"/* \
  --verify-tag \
  --title "OrbitQuant $VERSION" \
  --notes-file "docs/release-$VERSION.md"
gh release view "$TAG"
```

## Final Audit

```bash
gh repo view iamwavecut/OrbitQuant --json visibility,url,defaultBranchRef
uv run orbitquant audit-hf-artifacts \
  --namespace WaveCut \
  --policy-inventory-root reports/native/module-inventories \
  --summary-only \
  --fail-on-artifact-regression
test -z "$(git status --porcelain)"
```
