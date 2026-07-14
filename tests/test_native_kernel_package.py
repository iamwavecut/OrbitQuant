from __future__ import annotations

import ast
import re
import subprocess
import sys
import tomllib
from pathlib import Path

KERNEL_ROOT = Path("native-kernels/orbitquant-packed-matmul")
_GENERATED_DIRS = {".venv", "__pycache__", "build"}


def _kernel_source_files() -> list[Path]:
    return [
        path
        for path in KERNEL_ROOT.rglob("*")
        if path.is_file() and _GENERATED_DIRS.isdisjoint(path.relative_to(KERNEL_ROOT).parts)
    ]


def test_kernel_builder_manifest_targets_cpu_cuda_and_metal() -> None:
    config = tomllib.loads((KERNEL_ROOT / "build.toml").read_text(encoding="utf-8"))

    assert config["general"]["name"] == "orbitquant-packed-matmul"
    assert config["general"]["edition"] == 5
    assert config["general"]["license"] == "Apache-2.0"
    assert config["general"]["backends"] == ["cpu", "cuda", "metal"]
    assert config["general"]["hub"]["repo-id"] == "WaveCut/orbitquant-packed-matmul"
    assert config["torch"]["src"] == [
        "torch-ext/torch_binding.cpp",
        "torch-ext/torch_binding.h",
    ]
    kernels = config["kernel"]
    assert config["torch"]["stable-abi"] == {"cpu": "2.11"}
    assert kernels["packed_matmul_cpu"]["backend"] == "cpu"
    assert kernels["packed_matmul_cpu"]["depends"] == ["torch"]
    assert kernels["packed_matmul_cpu"]["src"] == [
        "orbitquant_packed_matmul_cpu/cpu_isa.cpp",
        "orbitquant_packed_matmul_cpu/cpu_kernel_args.h",
        "orbitquant_packed_matmul_cpu/cpu_pool.cpp",
        "orbitquant_packed_matmul_cpu/cpu_pool.h",
        "orbitquant_packed_matmul_cpu/cpu_threads.cpp",
        "orbitquant_packed_matmul_cpu/cpu_threads.h",
        "orbitquant_packed_matmul_cpu/packed_adaln_cpu.cpp",
        "orbitquant_packed_matmul_cpu/packed_matmul_cpu.cpp",
        "orbitquant_packed_matmul_cpu/packed_matmul_cpu.h",
        "orbitquant_packed_matmul_cpu/packed_matmul_scalar.cpp",
        "orbitquant_packed_matmul_cpu/packed_matmul_neon.cpp",
        "orbitquant_packed_matmul_cpu/packed_matmul_x86_avx512.cpp",
        "orbitquant_packed_matmul_cpu/quantize_activations_cpu.cpp",
    ]
    assert kernels["packed_matmul_cpu_x86_avx2"]["backend"] == "cpu"
    assert kernels["packed_matmul_cpu_x86_avx2"]["depends"] == ["torch"]
    assert kernels["packed_matmul_cpu_x86_avx2"]["cxx-flags"] == [
        "$<$<CXX_COMPILER_ID:MSVC>:/arch:AVX2>"
    ]
    assert kernels["packed_matmul_cpu_x86_avx2"]["src"] == [
        "orbitquant_packed_matmul_cpu/cpu_msvc_avx2.cpp",
        "orbitquant_packed_matmul_cpu/packed_matmul_x86.cpp",
    ]
    assert kernels["packed_matmul_cuda"]["backend"] == "cuda"
    assert kernels["packed_matmul_cuda"]["depends"] == ["torch"]
    assert kernels["packed_matmul_metal"]["backend"] == "metal"
    assert kernels["packed_matmul_metal"]["depends"] == ["torch"]


def test_wheel_project_preparation_keeps_windows_ninja_executable() -> None:
    script = (KERNEL_ROOT / "scripts/prepare_wheel_project.py").read_text(
        encoding="utf-8"
    )

    assert '"--torch-requirement"' in script
    assert 'default="torch>=2.11"' in script
    assert 'ninja_executable_path = Path(ninja.BIN_DIR)' in script
    assert 'which("ninja")' not in script
    assert '"ninja.exe" if os.name == "nt" else "ninja"' in script
    assert 'os.name != "nt" and which("{cache_tool}") is not None' in script
    assert 'os.environ.get("ORBITQUANT_CMAKE_MAKE_PROGRAM")' in script
    assert 'cmake_args.append(f"-DCMAKE_MAKE_PROGRAM:FILEPATH=' in script


def test_wheel_project_preparation_injects_build_tool_argv_hook(tmp_path) -> None:
    project = tmp_path / "kernel"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        '[project]\nversion = "0.1.0"\nrequires-python = ">=3.9"\n',
        encoding="utf-8",
    )
    (project / "CMakeLists.txt").write_text(
        "find_package(Python3 COMPONENTS Development Development.SABIModule Interpreter)\n"
        "find_package(Python3 REQUIRED COMPONENTS Development "
        "Development.SABIModule Interpreter)\n",
        encoding="utf-8",
    )
    (project / "setup.py").write_text(
        """import os
from pathlib import Path
from shutil import which, move

def is_sccache_available() -> bool:
    return which("sccache") is not None

def is_ccache_available() -> bool:
    return which("ccache") is not None

def make_args():
    cmake_args = []
    if "CMAKE_ARGS" in os.environ:
        cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]
    ninja_executable_path = Path(ninja.BIN_DIR) / "ninja"
    return cmake_args, ninja_executable_path

class Build:
    def build_extension(self, ext):
        build_temp = Path(self.build_temp) / ext.name
        extdir = Path("output")
        cfg = "Release"
        build_args = []
        subprocess.run(
            ["cmake", "--build", str(build_temp), *build_args], cwd=build_temp, check=True
        )
        if sys.platform == "win32":
            # Move the dylib one folder up for discovery.
            for filename in os.listdir(extdir / cfg):
                move(extdir / cfg / filename, extdir / filename)
        return build_temp

setup(
    zip_safe=False,
)
""",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            KERNEL_ROOT / "scripts/prepare_wheel_project.py",
            project,
            "--version",
            "0.4.0",
        ],
        check=True,
    )

    prepared_setup = (project / "setup.py").read_text(encoding="utf-8")
    ast.parse(prepared_setup)
    assert 'os.environ.get("ORBITQUANT_CMAKE_MAKE_PROGRAM")' in prepared_setup
    assert 'os.environ.get("ORBITQUANT_BUILD_TEMP", self.build_temp)' in prepared_setup
    assert "build_temp = (Path(build_temp_root) / ext.name).resolve()" in prepared_setup
    assert 'sys.platform == "win32" and (extdir / cfg).is_dir()' in prepared_setup
    assert 'return os.name != "nt" and which("ccache") is not None' in prepared_setup
    assert 'Path(ninja.BIN_DIR) / ("ninja.exe" if os.name == "nt" else "ninja")' in prepared_setup
    assert 'copy2(generated_ops, extdir / "_ops.py")' in prepared_setup
    assert 'options={"bdist_wheel": {"py_limited_api": "cp39"}}' in prepared_setup
    prepared_project = tomllib.loads(
        (project / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert prepared_project["project"]["version"] == "0.4.0"
    assert prepared_project["project"]["dependencies"] == ["torch>=2.11"]


def test_wheel_project_preparation_pins_custom_torch_requirement(tmp_path) -> None:
    project = tmp_path / "kernel"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        '[project]\nversion = "0.1.0"\nrequires-python = ">=3.9"\n',
        encoding="utf-8",
    )
    (project / "CMakeLists.txt").write_text(
        "find_package(Python3 COMPONENTS Development Development.SABIModule Interpreter)\n"
        "find_package(Python3 REQUIRED COMPONENTS Development "
        "Development.SABIModule Interpreter)\n",
        encoding="utf-8",
    )
    setup_template = (
        "import os\n"
        "from pathlib import Path\n"
        "from shutil import which, move\n"
        "\n"
        "def is_sccache_available() -> bool:\n"
        '    return which("sccache") is not None\n'
        "\n"
        "def is_ccache_available() -> bool:\n"
        '    return which("ccache") is not None\n'
        "\n"
        "def make_args():\n"
        "    cmake_args = []\n"
        '    if "CMAKE_ARGS" in os.environ:\n'
        '        cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") '
        "if item]\n"
        '    ninja_executable_path = Path(ninja.BIN_DIR) / "ninja"\n'
        "    return cmake_args, ninja_executable_path\n"
        "\n"
        "class Build:\n"
        "    def build_extension(self, ext):\n"
        "        build_temp = Path(self.build_temp) / ext.name\n"
        '        extdir = Path("output")\n'
        '        cfg = "Release"\n'
        "        build_args = []\n"
        "        subprocess.run(\n"
        '            ["cmake", "--build", str(build_temp), *build_args], '
        "cwd=build_temp, check=True\n"
        "        )\n"
        '        if sys.platform == "win32":\n'
        "            # Move the dylib one folder up for discovery.\n"
        "            for filename in os.listdir(extdir / cfg):\n"
        "                move(extdir / cfg / filename, extdir / filename)\n"
        "        return build_temp\n"
        "\n"
        "setup(\n"
        "    zip_safe=False,\n"
        ")\n"
    )
    (project / "setup.py").write_text(setup_template, encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            KERNEL_ROOT / "scripts/prepare_wheel_project.py",
            project,
            "--version",
            "1.0.0+cu128torch29",
            "--torch-requirement",
            "torch>=2.9,<2.10",
        ],
        check=True,
    )

    prepared_project = tomllib.loads(
        (project / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert prepared_project["project"]["version"] == "1.0.0+cu128torch29"
    assert prepared_project["project"]["dependencies"] == ["torch>=2.9,<2.10"]
    prepared_cmake = (project / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "COMPONENTS Development.SABIModule Interpreter" in prepared_cmake
    assert "COMPONENTS Development Development.SABIModule" not in prepared_cmake


def test_kernel_builder_binding_uses_abi3_safe_registration_pattern() -> None:
    binding = (KERNEL_ROOT / "torch-ext/torch_binding.cpp").read_text(encoding="utf-8")
    header = (KERNEL_ROOT / "torch-ext/torch_binding.h").read_text(encoding="utf-8")

    assert "TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops)" in binding
    assert "STABLE_TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops)" in binding
    assert "STABLE_TORCH_LIBRARY_IMPL_EXPAND" in binding
    assert "TORCH_BOX(&matmul_packed_weight)" in binding
    assert "TORCH_BOX(&matmul_packed_adaln_int4_cpu)" in binding
    assert "REGISTER_EXTENSION(TORCH_EXTENSION_NAME)" in binding
    assert "torch/torch.h" in header
    assert "torch/csrc/stable/tensor.h" in header

    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (KERNEL_ROOT / "torch-ext").rglob("*")
        if path.suffix in {".cpp", ".h", ".py"}
    )
    banned = [
        "pybind11",
        "PYBIND11",
        "torch/extension.h",
        "PYBIND11_MODULE",
        "py::",
        "TORCH_LIBRARY(",
        "TORCH_LIBRARY_FRAGMENT",
        "TORCH_LIBRARY_IMPL(",
        "PyInit",
        "torch.ops.",
        "import orbitquant",
        "from orbitquant",
    ]
    for pattern in banned:
        assert pattern not in combined


def test_kernel_builder_package_has_no_setuptools_extension_path() -> None:
    assert not (KERNEL_ROOT / "setup.py").exists()
    combined = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in _kernel_source_files()
    )
    banned = [
        "CUDAExtension",
        "BuildExtension",
        "cpp_extension",
        "load_inline",
        "torch.utils.cpp_extension",
    ]
    for pattern in banned:
        assert pattern not in combined


def test_kernel_python_wrapper_imports_only_torch_and_relative_ops() -> None:
    wrapper = KERNEL_ROOT / "torch-ext/orbitquant_packed_matmul/__init__.py"
    tree = ast.parse(wrapper.read_text(encoding="utf-8"))

    imports: list[ast.Import | ast.ImportFrom] = [
        node for node in ast.walk(tree) if isinstance(node, ast.Import | ast.ImportFrom)
    ]
    for node in imports:
        if isinstance(node, ast.Import):
            assert [alias.name for alias in node.names] == ["torch"]
        else:
            assert node.level > 0 or node.module == "__future__"
            assert node.module in {"__future__", "_ops"}


def test_kernel_sources_expose_the_current_runtime_contract() -> None:
    public_api = (
        KERNEL_ROOT / "torch-ext/orbitquant_packed_matmul/__init__.py"
    ).read_text(encoding="utf-8")
    cuda_source = (KERNEL_ROOT / "orbitquant_packed_matmul_cuda/packed_matmul.cu").read_text(
        encoding="utf-8"
    )
    metal_source = (KERNEL_ROOT / "orbitquant_packed_matmul_metal/packed_matmul.metal").read_text(
        encoding="utf-8"
    )
    cpu_source = (KERNEL_ROOT / "orbitquant_packed_matmul_cpu/packed_matmul_cpu.cpp").read_text(
        encoding="utf-8"
    )
    cpu_isa_source = (
        KERNEL_ROOT / "orbitquant_packed_matmul_cpu/cpu_isa.cpp"
    ).read_text(encoding="utf-8")
    cpu_msvc_avx2_source = (
        KERNEL_ROOT / "orbitquant_packed_matmul_cpu/cpu_msvc_avx2.cpp"
    ).read_text(encoding="utf-8")
    cpu_threads_source = (
        KERNEL_ROOT / "orbitquant_packed_matmul_cpu/cpu_threads.cpp"
    ).read_text(encoding="utf-8")
    cpu_neon_source = (
        KERNEL_ROOT / "orbitquant_packed_matmul_cpu/packed_matmul_neon.cpp"
    ).read_text(encoding="utf-8")
    cpu_avx2_source = (
        KERNEL_ROOT / "orbitquant_packed_matmul_cpu/packed_matmul_x86.cpp"
    ).read_text(encoding="utf-8")
    cpu_avx512_source = (
        KERNEL_ROOT / "orbitquant_packed_matmul_cpu/packed_matmul_x86_avx512.cpp"
    ).read_text(encoding="utf-8")
    cpu_activation_source = (
        KERNEL_ROOT / "orbitquant_packed_matmul_cpu/quantize_activations_cpu.cpp"
    ).read_text(encoding="utf-8")
    cpu_adaln_source = (
        KERNEL_ROOT / "orbitquant_packed_matmul_cpu/packed_adaln_cpu.cpp"
    ).read_text(encoding="utf-8")

    for token in (
        "packed_weight_indices",
        "row_norms",
        "centroids",
        "bits",
        "out_features",
        "in_features",
        "block_m",
        "block_n",
        "block_k",
    ):
        assert token in public_api
    assert re.search(r"byte_index\s*=\s*bit_start\s*>>\s*3", cuda_source)
    assert re.search(r"bit_offset\s*=\s*bit_start\s*&\s*7", cuda_source)
    assert "packed_weight_indices.scalar_type() == torch::kUInt8" in cuda_source
    assert "row_norms.scalar_type() == torch::kBFloat16" in cuda_source
    assert "centroids.scalar_type() == torch::kFloat" in cuda_source
    assert "bias.scalar_type() == x.scalar_type()" in cuda_source
    assert 'auxiliary_dtype = torch.bfloat16 if x.device.type == "cuda"' in public_api
    assert "packed_matmul_forward_float" in metal_source
    assert "packed_matmul_forward_half" in metal_source
    assert "packed_matmul_forward_bfloat16" in metal_source
    assert "packed_matmul_padded_mma_value<bfloat" in metal_source
    assert "scalar_t(acc)" in metal_source
    assert "threadgroup float *shared" in metal_source
    assert "threadgroup_barrier(mem_flags::mem_threadgroup)" in metal_source
    assert "params.block_k" in metal_source
    assert "torch::headeronly::ScalarType" in cpu_source
    assert "packed_matmul_neon_available" in cpu_source
    assert "vfmaq_f32" in cpu_neon_source
    assert "packed_matmul_scalar_range" in cpu_neon_source
    assert "ORBITQUANT_CPU_ISA" in cpu_source
    assert "packed_matmul_x86_avx2_available" in cpu_source
    assert "_mm256_permutevar8x32_ps" in cpu_avx2_source
    assert "runtime_has_avx2_fma_f16c" in cpu_isa_source
    assert "__cpuidex" in cpu_isa_source
    assert "activation_fwht_msvc_avx2" in cpu_msvc_avx2_source
    assert "packed_adaln_msvc_avx2_range" in cpu_msvc_avx2_source
    assert "_mm512_permutexvar_ps" in cpu_avx512_source
    assert "_mm512_permutexvar_epi16" in cpu_avx512_source
    assert "_mm512_dpbf16_ps" in cpu_avx512_source
    assert '__builtin_cpu_supports("avx512bf16")' in cpu_avx512_source
    assert "kPrimaryRowTile = 8" in cpu_avx512_source
    assert "quantize_lookup_avx2" in cpu_activation_source
    assert "quantize_lookup_avx512" in cpu_activation_source
    assert "_mm512_mask_add_epi32" in cpu_activation_source
    assert "select_activation_isa" in cpu_activation_source
    assert "packed_adaln_scalar_range" in cpu_adaln_source
    assert "packed_adaln_neon_range" in cpu_adaln_source
    assert "packed_adaln_avx2_range" in cpu_adaln_source
    assert "packed_adaln_avx512_range" in cpu_adaln_source
    assert "_mm512_dpbf16_ps" in cpu_adaln_source
    assert "sched_getaffinity" in cpu_threads_source
    assert "physical_package_id" in cpu_threads_source
    assert "hw.perflevel0.physicalcpu" in cpu_threads_source


def test_cuda_kernel_uses_wmma_for_supported_bf16_and_fp16_low_bit_modes() -> None:
    cuda_source = (KERNEL_ROOT / "orbitquant_packed_matmul_cuda/packed_matmul.cu").read_text(
        encoding="utf-8"
    )

    assert "orbitquant_packed_matmul_wmma_bf16_kernel" in cuda_source
    assert "orbitquant_packed_matmul_wmma_half_kernel" in cuda_source
    assert "wmma::mma_sync" in cuda_source
    for bits in (2, 3, 4, 6, 8):
        assert f"ORBITQUANT_LAUNCH_BF16({bits});" in cuda_source
        assert f"ORBITQUANT_LAUNCH_HALF({bits});" in cuda_source


def test_metal_host_source_matches_cuda_packed_weight_contract() -> None:
    metal_host = (
        KERNEL_ROOT / "orbitquant_packed_matmul_metal/packed_matmul.mm"
    ).read_text(encoding="utf-8")

    assert "#include <torch/mps.h>" in metal_host
    assert "torch::mps::get_command_buffer()" in metal_host
    assert "torch::mps::get_dispatch_queue()" in metal_host
    assert "torch::mps::commit()" in metal_host
    assert "setThreadgroupMemoryLength" in metal_host
    assert "packed_matmul_forward_bfloat16" in metal_host
    assert "x.scalar_type() == torch::kBFloat16" in metal_host
    assert "packed_weight_indices.scalar_type() == torch::kUInt8" in metal_host
    assert "row_norms.scalar_type() == torch::kFloat" in metal_host
    assert "centroids.scalar_type() == torch::kFloat" in metal_host
    assert "bias.scalar_type() == torch::kFloat" in metal_host


def test_kernel_package_pytest_marks_runtime_test_for_kernel_builder_ci() -> None:
    test_source = (KERNEL_ROOT / "tests/test_packed_matmul.py").read_text(encoding="utf-8")

    assert "@pytest.mark.kernels_ci" in test_source


def test_kernel_benchmark_reports_reference_comparison() -> None:
    benchmark_source = (KERNEL_ROOT / "benchmarks/benchmark.py").read_text(encoding="utf-8")

    assert "torch.nn.functional.linear" in benchmark_source
    assert "reference_weight" in benchmark_source
    assert "materialize_reference_weight" in benchmark_source
    assert "packed_seconds_per_iter" in benchmark_source
    assert 'choices=["cpu", "cuda", "mps"]' in benchmark_source
    assert "packed_first_call_seconds" in benchmark_source
    assert "packed_hot_median_seconds" in benchmark_source
    assert "packed_hot_p95_seconds" in benchmark_source
    assert "predequantized_f_linear_seconds_per_iter" in benchmark_source
    assert "predequantized_hot_median_seconds" in benchmark_source
    assert "predequantized_hot_p95_seconds" in benchmark_source
    assert "dequantize_then_f_linear_seconds_per_iter" in benchmark_source
    assert "dequantize_then_hot_median_seconds" in benchmark_source
    assert "dequantize_then_hot_p95_seconds" in benchmark_source
    assert "packed_weight_indices_bytes" in benchmark_source
    assert "row_norms_bytes" in benchmark_source
    assert "centroid_bytes" in benchmark_source
    assert "packed_weight_path_bytes" in benchmark_source
    assert "materialized_weight_bytes" in benchmark_source
    assert "packed_weight_path_vs_materialized_weight_ratio" in benchmark_source
    assert "packed_vs_predequantized_f_linear_speedup" in benchmark_source
    assert "packed_vs_dequantize_then_f_linear_speedup" in benchmark_source
    assert "reference_seconds_per_iter" in benchmark_source
    assert "packed_vs_reference_speedup" in benchmark_source
    assert "max_abs_error" in benchmark_source
    assert "relative_rmse" in benchmark_source
