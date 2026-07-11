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

    cmake_args = '    if "CMAKE_ARGS" in os.environ:\n'
    if setup_text.count(cmake_args) != 1:
        raise RuntimeError("generated setup must contain one CMake arguments block")
    setup_text = setup_text.replace(
        cmake_args,
        '    if "CMAKE_MAKE_PROGRAM" in os.environ:\n'
        "        cmake_args.append(\n"
        '            f"-DCMAKE_MAKE_PROGRAM:FILEPATH='
        "{os.environ['CMAKE_MAKE_PROGRAM']}" + '"\n'
        "        )\n\n" + cmake_args,
        1,
    )

    ccache_condition = "    elif is_ccache_available():\n"
    if setup_text.count(ccache_condition) != 1:
        raise RuntimeError("generated setup must contain one ccache condition")
    setup_text = setup_text.replace(
        ccache_condition,
        '    elif is_ccache_available() and sys.platform != "win32":\n',
        1,
    )

    windows_move = '        if sys.platform == "win32":\n'
    if setup_text.count(windows_move) != 1:
        raise RuntimeError("generated setup must contain one Windows output move")
    setup_text = setup_text.replace(
        windows_move,
        '        if sys.platform == "win32" and (extdir / cfg).is_dir():\n',
        1,
    )
    setup.write_text(setup_text, encoding="utf-8")


if __name__ == "__main__":
    main()
