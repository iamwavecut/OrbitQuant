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

    cmake = args.project / "CMakeLists.txt"
    cmake_text = cmake.read_text(encoding="utf-8")
    for required in (False, True):
        marker = " REQUIRED" if required else ""
        development = (
            f"find_package(Python3{marker} COMPONENTS Development "
            "Development.SABIModule Interpreter)"
        )
        if cmake_text.count(development) != 1:
            raise RuntimeError(
                "generated CMake must contain one Python development lookup"
            )
        cmake_text = cmake_text.replace(
            development,
            f"find_package(Python3{marker} COMPONENTS Development.SABIModule Interpreter)",
            1,
        )
    cmake.write_text(cmake_text, encoding="utf-8")

    setup = args.project / "setup.py"
    setup_text = setup.read_text(encoding="utf-8")
    shutil_import = "from shutil import which, move\n"
    if setup_text.count(shutil_import) != 1:
        raise RuntimeError("generated setup must contain one shutil import")
    setup_text = setup_text.replace(
        shutil_import,
        "from shutil import copy2, move, which\n",
        1,
    )
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
    cmake_args_hook = (
        '    if "CMAKE_ARGS" in os.environ:\n'
        '        cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") '
        "if item]\n"
    )
    if setup_text.count(cmake_args_hook) != 1:
        raise RuntimeError("generated setup must contain one CMAKE_ARGS hook")
    setup_text = setup_text.replace(
        cmake_args_hook,
        cmake_args_hook
        + '    cmake_make_program = os.environ.get("ORBITQUANT_CMAKE_MAKE_PROGRAM")\n'
        + "    if cmake_make_program:\n"
        + '        cmake_args.append(f"-DCMAKE_MAKE_PROGRAM:FILEPATH={cmake_make_program}")\n',
        1,
    )
    build_temp = "        build_temp = Path(self.build_temp) / ext.name"
    if setup_text.count(build_temp) != 1:
        raise RuntimeError("generated setup must contain one extension build temp")
    setup_text = setup_text.replace(
        build_temp,
        '        build_temp_root = os.environ.get("ORBITQUANT_BUILD_TEMP", '
        "self.build_temp)\n"
        "        build_temp = (Path(build_temp_root) / ext.name).resolve()",
        1,
    )
    windows_multi_config = (
        '        if sys.platform == "win32":\n'
        "            # Move the dylib one folder up for discovery."
    )
    if setup_text.count(windows_multi_config) != 1:
        raise RuntimeError("generated setup must contain one Windows output move")
    setup_text = setup_text.replace(
        windows_multi_config,
        '        if sys.platform == "win32" and (extdir / cfg).is_dir():\n'
        "            # Move the dylib one folder up for discovery.",
        1,
    )
    build_call = (
        "        subprocess.run(\n"
        '            ["cmake", "--build", str(build_temp), *build_args], '
        "cwd=build_temp, check=True\n"
        "        )\n"
    )
    if setup_text.count(build_call) != 1:
        raise RuntimeError("generated setup must contain one wheel CMake build call")
    setup_text = setup_text.replace(
        build_call,
        build_call
        + "\n"
        + '        package_name = ext.name.split(".", 1)[0]\n'
        + "        generated_ops = (\n"
        + '            Path(ext.sourcedir) / "torch-ext" / package_name / "_ops.py"\n'
        + "        )\n"
        + '        copy2(generated_ops, extdir / "_ops.py")\n',
        1,
    )
    zip_safe = "    zip_safe=False,\n"
    if setup_text.count(zip_safe) != 1:
        raise RuntimeError("generated setup must contain one zip-safe option")
    setup_text = setup_text.replace(
        zip_safe,
        '    options={"bdist_wheel": {"py_limited_api": "cp39"}},\n' + zip_safe,
        1,
    )
    setup.write_text(setup_text, encoding="utf-8")


if __name__ == "__main__":
    main()
