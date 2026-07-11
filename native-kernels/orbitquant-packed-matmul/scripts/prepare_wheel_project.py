from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare kernel-builder output for a platform wheel build."
    )
    parser.add_argument("project", type=Path)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()

    pyproject = args.project / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")

    old_version = 'version = "0.1.0"'
    if text.count(old_version) != 1:
        raise RuntimeError("generated pyproject must contain one stub version")
    text = text.replace(old_version, f'version = "{args.version}"', 1)

    requires_python = 'requires-python = ">=3.9"'
    if text.count(requires_python) != 1:
        raise RuntimeError("generated pyproject must contain one Python requirement")
    text = text.replace(
        requires_python,
        f'{requires_python}\ndependencies = ["torch>=2.11"]',
        1,
    )
    pyproject.write_text(text, encoding="utf-8")

    setup = args.project / "setup.py"
    setup_text = setup.read_text(encoding="utf-8")
    ninja_path = 'ninja_executable_path = Path(ninja.BIN_DIR) / "ninja"'
    if setup_text.count(ninja_path) != 1:
        raise RuntimeError("generated setup must contain one Ninja executable path")
    setup_text = setup_text.replace(
        ninja_path,
        'ninja_executable_path = Path(ninja.BIN_DIR) / '
        '("ninja.exe" if os.name == "nt" else "ninja")',
        1,
    )
    for cache_tool in ("sccache", "ccache"):
        availability = f'return which("{cache_tool}") is not None'
        if setup_text.count(availability) != 1:
            raise RuntimeError(
                f"generated setup must contain one {cache_tool} availability check"
            )
        setup_text = setup_text.replace(
            availability,
            f'return os.name != "nt" and which("{cache_tool}") is not None',
            1,
        )
    setup.write_text(setup_text, encoding="utf-8")


if __name__ == "__main__":
    main()
