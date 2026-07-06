import subprocess
import sys


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
    assert "0.1.0" in result.stdout
