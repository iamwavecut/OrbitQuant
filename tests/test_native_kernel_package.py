from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path

KERNEL_ROOT = Path("native-kernels/orbitquant-packed-matmul")


def test_kernel_builder_manifest_targets_cuda_and_metal() -> None:
    config = tomllib.loads((KERNEL_ROOT / "build.toml").read_text(encoding="utf-8"))

    assert config["general"]["name"] == "orbitquant-packed-matmul"
    assert config["general"]["edition"] == 5
    assert config["general"]["license"] == "Apache-2.0"
    assert config["general"]["backends"] == ["cuda", "metal"]
    assert config["general"]["hub"]["repo-id"] == "WaveCut/orbitquant-packed-matmul"
    assert config["torch"]["src"] == [
        "torch-ext/torch_binding.cpp",
        "torch-ext/torch_binding.h",
    ]
    kernels = config["kernel"]
    assert kernels["packed_matmul_cuda"]["backend"] == "cuda"
    assert kernels["packed_matmul_cuda"]["depends"] == ["torch"]
    assert kernels["packed_matmul_metal"]["backend"] == "metal"
    assert kernels["packed_matmul_metal"]["depends"] == ["torch"]


def test_kernel_builder_binding_uses_abi3_safe_registration_pattern() -> None:
    binding = (KERNEL_ROOT / "torch-ext/torch_binding.cpp").read_text(encoding="utf-8")
    header = (KERNEL_ROOT / "torch-ext/torch_binding.h").read_text(encoding="utf-8")

    assert "TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops)" in binding
    assert "REGISTER_EXTENSION(TORCH_EXTENSION_NAME)" in binding
    assert "torch/torch.h" in header

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
        "TORCH_LIBRARY_IMPL",
        "PyInit",
        "torch.ops.",
        "import orbitquant",
        "from orbitquant",
    ]
    for pattern in banned:
        assert pattern not in combined


def test_kernel_builder_package_has_no_setuptools_extension_path() -> None:
    combined = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in KERNEL_ROOT.rglob("*")
        if path.is_file()
    )
    banned = [
        "setup.py",
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
    assert "row_norms.scalar_type() == torch::kFloat" in cuda_source
    assert "centroids.scalar_type() == torch::kFloat" in cuda_source
    assert "bias.scalar_type() == torch::kFloat" in cuda_source
    assert "packed_matmul_forward_float" in metal_source
    assert "packed_matmul_forward_half" in metal_source


def test_metal_host_source_matches_cuda_packed_weight_contract() -> None:
    metal_host = (
        KERNEL_ROOT / "orbitquant_packed_matmul_metal/packed_matmul.mm"
    ).read_text(encoding="utf-8")

    assert "#include <torch/mps.h>" in metal_host
    assert "torch::mps::get_command_buffer()" in metal_host
    assert "torch::mps::get_dispatch_queue()" in metal_host
    assert "torch::mps::commit()" in metal_host
    assert "packed_weight_indices.scalar_type() == torch::kUInt8" in metal_host
    assert "row_norms.scalar_type() == torch::kFloat" in metal_host
    assert "centroids.scalar_type() == torch::kFloat" in metal_host
    assert "bias.scalar_type() == torch::kFloat" in metal_host


def test_kernel_package_pytest_marks_runtime_test_for_kernel_builder_ci() -> None:
    test_source = (KERNEL_ROOT / "tests/test_packed_matmul.py").read_text(encoding="utf-8")

    assert "@pytest.mark.kernels_ci" in test_source
