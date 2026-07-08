import os
import re
import subprocess
import tomllib
from pathlib import Path


def _tracked_native_kernel_files(package_root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", str(package_root)],
        check=True,
        capture_output=True,
        text=True,
    )
    return [Path(line) for line in result.stdout.splitlines() if line]


def test_github_actions_cpu_unit_workflow_exists():
    workflow = Path(".github/workflows/ci.yml")

    assert workflow.is_file()
    text = workflow.read_text(encoding="utf-8")
    assert "uses: actions/checkout@v7.0.0" in text
    assert "uses: actions/setup-python@v6.3.0" in text
    assert "uses: astral-sh/setup-uv@v8.3.0" in text
    assert "uv sync --extra dev --extra hf --extra eval" in text
    assert "uv run ruff check ." in text
    assert "HF integration tests" in text
    assert "scripts/run_hf_compat_checks.sh --mode current" in text
    assert "uv run pytest" in text
    assert "uv run --with build python -m build" in text
    assert "uv run --with twine python -m twine check dist/*" in text
    assert "uv pip install --python" in text
    assert "dist/*.whl" in text
    assert "orbitquant-0.1.0-py3-none-any.whl" not in text
    assert "orbitquant --version" in text


def test_kernel_check_scripts_are_executable_and_stage_logged():
    for script_name in ("run_cuda_kernel_checks.sh", "run_mps_kernel_checks.sh"):
        script = Path("scripts") / script_name
        assert script.is_file()
        assert os.access(script, os.X_OK)

        text = script.read_text(encoding="utf-8")
        assert "REMOTE_STAGE" in text
        assert "orbitquant kernel-info" in text

    cuda_script = Path("scripts/run_cuda_kernel_checks.sh").read_text(encoding="utf-8")
    assert "triton-cuda-kernel-contract-ok" in cuda_script
    assert "native-kernel-package-ci-start" in cuda_script
    assert "nix --option sandbox relaxed run .#ci-test -L" in cuda_script
    assert "kernels>=0.16" in cuda_script
    assert "LOCAL_KERNELS=\"WaveCut/orbitquant-packed-matmul=" in cuda_script
    assert "native-packed-matmul-kernel-ok" in cuda_script
    assert "--runtime-mode native_packed_matmul" in cuda_script
    assert "activation_norm_rpbh_quant_rescale" in cuda_script
    assert "packed_weight_matmul" in cuda_script
    assert "hf_kernel_builder_compliant" in cuda_script
    assert cuda_script.index("kernel-tests-start") < cuda_script.index(
        "native-kernel-package-ci-start"
    )
    assert cuda_script.index("kernel-info-start") < cuda_script.index(
        "native-kernel-package-ci-start"
    )
    assert cuda_script.index("kernel-bench-start") < cuda_script.index(
        "native-kernel-package-ci-start"
    )
    assert cuda_script.index("native-kernel-package-ci-done") < cuda_script.index(
        "native-packed-matmul-bench-start"
    )

    mps_script = Path("scripts/run_mps_kernel_checks.sh").read_text(encoding="utf-8")
    assert "MPS Metal compile_shader is not available" in mps_script
    assert "mps-kernel-contract-ok" in mps_script
    assert "codebook_lookup_rescale" in mps_script
    assert "upstream_native_mps_op" in mps_script
    assert "kernels>=0.16" in mps_script
    assert "native-packed-matmul-kernel-ok" in mps_script
    assert "--runtime-mode native_packed_matmul" in mps_script


def test_native_packed_matmul_kernel_package_stays_kernel_builder_abi3_compliant():
    package_root = Path("native-kernels/orbitquant-packed-matmul")
    tracked_files = _tracked_native_kernel_files(package_root)

    forbidden_patterns = {
        "pybind11": re.compile(r"pybind11|PYBIND11|py::"),
        "torch extension header": re.compile(r"torch/extension\.h"),
        "hardcoded torch op namespace": re.compile(
            r"TORCH_LIBRARY\(|TORCH_LIBRARY_FRAGMENT|TORCH_LIBRARY_IMPL|PyInit"
        ),
        "setuptools extension build": re.compile(
            r"CUDAExtension|BuildExtension|cpp_extension\.load|load_inline"
        ),
        "hardcoded Python torch.ops lookup": re.compile(r"torch\.ops\."),
    }
    violations: list[str] = []
    for path in tracked_files:
        if path.suffix not in {".cpp", ".h", ".cu", ".mm", ".metal", ".py"}:
            continue
        text = path.read_text(encoding="utf-8")
        for label, pattern in forbidden_patterns.items():
            if pattern.search(text):
                violations.append(f"{path}: {label}")

    assert not (package_root / "setup.py").exists()
    assert not (package_root / "pyproject.toml").exists()
    assert not violations

    build_config = tomllib.loads((package_root / "build.toml").read_text(encoding="utf-8"))
    assert build_config["general"]["name"] == "orbitquant-packed-matmul"
    assert "_" not in build_config["general"]["name"]
    assert build_config["general"]["license"] == "Apache-2.0"
    assert build_config["general"]["backends"] == ["cuda", "metal"]
    assert build_config["general"]["upstream"] == "https://github.com/iamwavecut/OrbitQuant"
    assert (
        build_config["general"]["source"]
        == "https://huggingface.co/WaveCut/orbitquant-packed-matmul"
    )

    binding = (package_root / "torch-ext/torch_binding.cpp").read_text(encoding="utf-8")
    assert "#include <torch/library.h>" in binding
    assert "#include \"registration.h\"" in binding
    assert "TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops)" in binding
    assert "REGISTER_EXTENSION(TORCH_EXTENSION_NAME)" in binding

    cuda_source = (
        package_root / "orbitquant_packed_matmul_cuda/packed_matmul.cu"
    ).read_text(encoding="utf-8")
    assert "#include <c10/cuda/CUDAException.h>" in cuda_source
    assert cuda_source.count("C10_CUDA_KERNEL_LAUNCH_CHECK();") == 3

    package_api = (
        package_root / "torch-ext/orbitquant_packed_matmul/__init__.py"
    ).read_text(encoding="utf-8")
    assert "from ._ops import ops" in package_api
    assert "from orbitquant_packed_matmul" not in package_api


def test_hf_compatibility_script_is_executable_and_release_dev_aware():
    script = Path("scripts/run_hf_compat_checks.sh")

    assert script.is_file()
    assert os.access(script, os.X_OK)

    text = script.read_text(encoding="utf-8")
    assert "HF_COMPAT_STAGE" in text
    assert "--mode current|release|dev|all" in text
    assert "git+https://github.com/huggingface/diffusers.git" in text
    assert "git+https://github.com/huggingface/transformers.git" in text
    assert "build_diffusers_pipeline_quantization_config" in text
    assert "tests/test_quantizer_adapter.py" in text
    assert "tests/test_pipeline_helpers.py" in text
    assert "tests/test_diffusers_modelmixin_integration.py" in text
    assert "tests/test_transformers_pretrained_integration.py" in text


def test_paper_methodology_script_is_executable_and_scope_limited():
    script = Path("scripts/run_paper_methodology_checks.sh")

    assert script.is_file()
    assert os.access(script, os.X_OK)

    text = script.read_text(encoding="utf-8")
    assert "PAPER_METHOD_STAGE" in text
    assert "tests/test_codebooks.py" in text
    assert "tests/test_rpbh.py" in text
    assert "tests/test_orbit_linear.py" in text
    assert "tests/test_adaln_rtn.py" in text
    assert "tests/test_target_policies.py" in text
    assert "orbitquant inspect-policy" in text
    assert "flux1-schnell-native" in text
    assert "z-image-native" in text
    assert "wan-native" in text
    assert "GenEval" in text
    assert "VBench" in text
    assert "model generation" in text


def test_pyproject_has_release_package_metadata():
    payload = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    project = payload["project"]

    assert project["license"] == "Apache-2.0"
    assert "diffusion" in project["keywords"]
    assert "quantization" in project["keywords"]
    assert "License :: OSI Approved :: Apache Software License" in project["classifiers"]
    assert "Topic :: Scientific/Engineering :: Artificial Intelligence" in (
        project["classifiers"]
    )
    assert project["urls"]["Repository"] == "https://github.com/iamwavecut/OrbitQuant"
    assert project["urls"]["Paper"] == "https://arxiv.org/abs/2607.02461"
