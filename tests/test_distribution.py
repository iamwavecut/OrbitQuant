from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path


def _tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [Path(line) for line in result.stdout.splitlines() if line]


def test_wheel_distribution_contains_only_the_python_runtime_package() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == [
        "src/orbitquant"
    ]


def test_source_distribution_keeps_kernel_source_without_generated_artifacts() -> None:
    tracked = _tracked_files()
    tracked_strings = {path.as_posix() for path in tracked}

    required_source_paths = {
        "native-kernels/orbitquant-packed-matmul/build.toml",
        "native-kernels/orbitquant-packed-matmul/CARD.md",
        "native-kernels/orbitquant-packed-matmul/flake.nix",
        "native-kernels/orbitquant-packed-matmul/flake.lock",
        "native-kernels/orbitquant-packed-matmul/benchmarks/benchmark.py",
        "native-kernels/orbitquant-packed-matmul/tests/test_packed_matmul.py",
        "native-kernels/orbitquant-packed-matmul/torch-ext/torch_binding.cpp",
        "native-kernels/orbitquant-packed-matmul/torch-ext/torch_binding.h",
        "native-kernels/orbitquant-packed-matmul/torch-ext/orbitquant_packed_matmul/__init__.py",
        "native-kernels/orbitquant-packed-matmul/orbitquant_packed_matmul_cuda/packed_matmul.cu",
        "native-kernels/orbitquant-packed-matmul/orbitquant_packed_matmul_metal/packed_matmul.mm",
        "native-kernels/orbitquant-packed-matmul/orbitquant_packed_matmul_metal/packed_matmul.metal",
    }
    assert required_source_paths <= tracked_strings

    forbidden_parts = {"build", ".venv", "__pycache__"}
    forbidden_suffixes = {".pyc", ".so"}
    forbidden_roots = {"artifacts", "reports", ".local-artifacts"}
    violations = [
        path.as_posix()
        for path in tracked
        if forbidden_parts.intersection(path.parts)
        or path.suffix in forbidden_suffixes
        or (path.parts and path.parts[0] in forbidden_roots)
    ]

    assert violations == []
