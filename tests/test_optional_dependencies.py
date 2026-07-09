import subprocess
import sys
import tomllib
from pathlib import Path

from orbitquant import __version__


def test_kernels_extra_keeps_triton_linux_only_for_mps_installs():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    kernels_extra = pyproject["project"]["optional-dependencies"]["kernels"]

    assert "kernels>=0.16" in kernels_extra
    assert "triton>=3.5; platform_system == 'Linux'" in kernels_extra
    assert "triton>=3.5" not in kernels_extra


def test_core_import_does_not_require_pillow():
    script = """
import importlib.abc
import sys

class BlockPillow(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "PIL" or fullname.startswith("PIL."):
            raise ImportError("Pillow is intentionally blocked")
        return None

sys.meta_path.insert(0, BlockPillow())
import orbitquant
print(orbitquant.__version__)
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert __version__ in result.stdout
