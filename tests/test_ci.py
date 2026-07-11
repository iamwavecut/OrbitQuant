import os
import re
import subprocess
import tomllib
from pathlib import Path

import orbitquant


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
    assert "windows-native-cpu:" in text
    assert "runs-on: windows-2025" in text
    assert "--rev d43de01d0b43285d8e5061ca4380c2bd1c40ae3b" in text
    assert "--debug" in text
    assert "prepare_wheel_project.py" in text
    assert "uses: actions/cache/restore@v6.1.0" in text
    assert "uses: actions/cache/save@v6.1.0" in text
    assert "vswhere.exe" in text
    assert "VsDevCmd.bat" in text
    assert "Get-Command nmake.exe -ErrorAction Stop" in text
    assert 'CMAKE_GENERATOR = "NMake Makefiles"' in text
    assert "Get-CimInstance Win32_Processor" in text
    assert "bdist_wheel --py-limited-api=cp39" in text
    assert '"-cp39-abi3-win_amd64\\.whl$"' in text
    assert "torch==2.12.1 pytest" in text
    assert "-m kernels_ci" in text


def test_package_version_matches_import_version():
    with Path("pyproject.toml").open("rb") as file:
        project = tomllib.load(file)["project"]

    assert project["version"] == orbitquant.__version__


def test_github_actions_pypi_publish_workflow_uses_trusted_publishing():
    workflow = Path(".github/workflows/publish-pypi.yml")

    assert workflow.is_file()
    text = workflow.read_text(encoding="utf-8")
    assert "workflow_dispatch" in text
    assert "version:" in text
    assert "test \"$version\" = \"${{ inputs.version }}\"" in text
    assert "uv run pytest" in text
    assert "uv run ruff check ." in text
    assert "uv run --with build python -m build" in text
    assert "uv run --with twine python -m twine check dist/*" in text
    assert "uses: actions/upload-artifact@v6.0.0" in text
    assert "uses: actions/download-artifact@v7.0.0" in text
    assert "environment:" in text
    assert "name: pypi" in text
    assert "id-token: write" in text
    assert "uses: pypa/gh-action-pypi-publish@release/v1" in text
    assert "TWINE_PASSWORD" not in text
    assert "PYPI_API_TOKEN" not in text


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
    assert "native-kernel-runtime-variant expected=$native_kernel_variant" in cuda_script
    assert "native-kernel-package-prebuilt-load-start" in cuda_script
    assert "native-kernel-package-prebuilt-load-done" in cuda_script
    assert "native-kernel-package-prebuilt-load-unavailable" in cuda_script
    assert "native_kernel_runtime_variant_name" in cuda_script
    assert "native_kernel_build_variant_dir" in cuda_script
    assert "ensure_native_kernel_source_git" in cuda_script
    assert "native-kernel-package-source-git-init-start" in cuda_script
    assert "Prepare source archive for kernel-builder" in cuda_script
    assert (
        'nix --extra-experimental-features "nix-command flakes" '
        '--option sandbox relaxed build --no-link --json ".#redistributable.$variant"'
        in cuda_script
    )
    assert "kernels>=0.16" in cuda_script
    assert "triton>=3.5" in cuda_script
    assert "ORBITQUANT_ALLOW_NATIVE_KERNEL_BUILD" in cuda_script
    assert "uncached CUDA builds can compile the CUDA/NCCL stack" in cuda_script
    assert "will not start a kernel-builder/Nix source build" in cuda_script
    assert "LOCAL_KERNELS=\"$NATIVE_KERNEL_REPO_ID=$native_kernel_variant_dir\"" in cuda_script
    assert (
        'nix --extra-experimental-features "nix-command flakes" '
        "--option sandbox relaxed eval --json .#variants"
        in cuda_script
    )
    assert "native kernel runtime variant is not exported by this flake" in cuda_script
    assert "torch.version.cuda" in cuda_script
    assert "expected_variant" in cuda_script
    assert "native-packed-matmul-variant-ok" in cuda_script
    assert "native-kernel-package-tests-start" in cuda_script
    assert "tests/test_packed_matmul.py" in cuda_script
    assert "native-packed-matmul-bench-skipped-no-local-kernel" in cuda_script
    assert "native-packed-matmul-kernel-ok" in cuda_script
    assert "NATIVE_PACKED_MATMUL_READY=1" in cuda_script
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
    assert cuda_script.index("ALLOW_NATIVE_KERNEL_BUILD") < cuda_script.index(
        "command -v nix"
    )
    assert cuda_script.index("ALLOW_NATIVE_KERNEL_BUILD") < cuda_script.index(
        "native-kernel-package-build-start"
    )
    assert cuda_script.index("native-kernel-package-ci-done") < cuda_script.index(
        "native-packed-matmul-bench-start"
    )

    mps_script = Path("scripts/run_mps_kernel_checks.sh").read_text(encoding="utf-8")
    assert "MPS Metal compile_shader is not available" in mps_script
    assert "mps-kernel-contract-ok" in mps_script
    assert "activation_norm_rpbh_quant_rescale" in mps_script
    assert "--dtype bfloat16" in mps_script
    assert "upstream_native_mps_op" in mps_script
    assert "kernels>=0.16" in mps_script
    assert "ORBITQUANT_RUN_NATIVE_KERNEL_PACKAGE_CI" in mps_script
    assert "native_kernel_local_variant_dir" in mps_script
    assert 'if machine == "arm64":' in mps_script
    assert 'machine = "aarch64"' in mps_script
    assert "native-packed-matmul-local-variant-selected" in mps_script
    assert "native-packed-matmul-local-variant-missing" in mps_script
    assert "metadata.json" in mps_script
    assert "LOCAL_KERNELS=\"$NATIVE_KERNEL_REPO_ID=$native_kernel_variant_dir\"" in mps_script
    assert "native-packed-matmul-kernel-ok" in mps_script
    assert "native-packed-matmul-load-skipped" in mps_script
    assert "native-packed-matmul-bench-skipped" in mps_script
    assert "BENCH_RUNTIME_ARGS=(--runtime-mode dequant_bf16)" in mps_script
    assert "--runtime-mode native_packed_matmul" in mps_script


def test_runpod_ssh_health_script_uses_sterile_ssh_probe():
    script = Path("scripts/runpod_ssh_health.sh")

    assert script.is_file()
    assert os.access(script, os.X_OK)

    text = script.read_text(encoding="utf-8")
    assert "It does not\nquery, create, stop, or modify pods." in text
    assert "-F /dev/null" in text
    assert "-tt" in text
    assert "-o IdentitiesOnly=yes" in text
    assert "-o ControlMaster=no" in text
    assert "-o ControlPath=none" in text
    assert "-o PreferredAuthentications=publickey" in text
    assert "-o PasswordAuthentication=no" in text
    assert "-o KbdInteractiveAuthentication=no" in text
    assert "__RUNPOD_SSH_HEALTH_OK__" in text
    assert "RUNPOD_SSH_HEALTH_READY" in text
    assert "feeds the health command through stdin" in text
    assert "runpodctl pod" not in text


def test_runpod_ssh_health_script_executes_clean_probe_with_web_ui_command(tmp_path):
    fake_ssh = tmp_path / "ssh"
    fake_ssh.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$@\" > \"$RUNPOD_FAKE_SSH_ARGS\"\n"
        "cat > \"$RUNPOD_FAKE_SSH_STDIN\"\n"
        "printf '__RUNPOD_SSH_HEALTH_OK__\\nfake-host\\n'\n",
        encoding="utf-8",
    )
    fake_ssh.chmod(0o755)
    key_path = tmp_path / "id_ed25519"
    key_path.write_text("fake-key", encoding="utf-8")
    args_path = tmp_path / "ssh-args.txt"
    stdin_path = tmp_path / "ssh-stdin.txt"
    env = os.environ.copy()
    env["PATH"] = f"{tmp_path}:{env['PATH']}"
    env["RUNPOD_FAKE_SSH_ARGS"] = str(args_path)
    env["RUNPOD_FAKE_SSH_STDIN"] = str(stdin_path)

    result = subprocess.run(
        [
            "scripts/runpod_ssh_health.sh",
            "ssh",
            "pod-user@ssh.runpod.io",
            "-i",
            str(key_path),
            "-p",
            "45678",
        ],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    ssh_args = args_path.read_text(encoding="utf-8").splitlines()
    ssh_stdin = stdin_path.read_text(encoding="utf-8")
    assert "RUNPOD_SSH_HEALTH_READY target=pod-user@ssh.runpod.io" in result.stdout
    assert ssh_args[:3] == ["-F", "/dev/null", "-tt"]
    assert "-o" in ssh_args
    assert "IdentitiesOnly=yes" in ssh_args
    assert "ControlMaster=no" in ssh_args
    assert "ControlPath=none" in ssh_args
    assert "PreferredAuthentications=publickey" in ssh_args
    assert "-i" in ssh_args
    assert str(key_path) in ssh_args
    assert "-p" in ssh_args
    assert "45678" in ssh_args
    assert "pod-user@ssh.runpod.io" in ssh_args
    assert not any("__RUNPOD_SSH_HEALTH_OK__" in arg for arg in ssh_args)
    assert "__RUNPOD_SSH_HEALTH_OK__" in ssh_stdin
    assert ssh_stdin.rstrip().endswith("exit")


def test_runpod_ssh_health_script_accepts_runpod_pty_control_sequences(tmp_path):
    fake_ssh = tmp_path / "ssh"
    fake_ssh.write_text(
        "#!/usr/bin/env bash\n"
        "cat > /dev/null\n"
        "printf '\\033[?2004l__RUNPOD_SSH_HEALTH_OK__\\nfake-host\\n'\n",
        encoding="utf-8",
    )
    fake_ssh.chmod(0o755)
    key_path = tmp_path / "id_ed25519"
    key_path.write_text("fake-key", encoding="utf-8")
    env = os.environ.copy()
    env["PATH"] = f"{tmp_path}:{env['PATH']}"

    result = subprocess.run(
        [
            "scripts/runpod_ssh_health.sh",
            "pod-user@ssh.runpod.io",
            str(key_path),
        ],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert "RUNPOD_SSH_HEALTH_READY target=pod-user@ssh.runpod.io" in result.stdout


def test_native_packed_matmul_kernel_package_stays_kernel_builder_abi3_compliant():
    package_root = Path("native-kernels/orbitquant-packed-matmul")
    tracked_files = _tracked_native_kernel_files(package_root)

    forbidden_patterns = {
        "pybind11": re.compile(r"pybind11|PYBIND11|py::"),
        "torch extension header": re.compile(r"torch/extension\.h"),
        "hardcoded torch op namespace": re.compile(
            r"TORCH_LIBRARY\(|TORCH_LIBRARY_FRAGMENT|"
            r"(?<!STABLE_)TORCH_LIBRARY_IMPL|PyInit"
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
    assert build_config["general"]["backends"] == ["cpu", "cuda", "metal"]
    assert build_config["torch"]["stable-abi"] == {"cpu": "2.11"}
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
    assert cuda_source.count("C10_CUDA_KERNEL_LAUNCH_CHECK();") == 9
    assert "orbitquant_packed_matmul_small_rows_kernel" in cuda_source
    assert "orbitquant_rpbh_quantize_pack_w4_kernel" in cuda_source
    assert "orbitquant_rpbh_quantize_int8_kernel" in cuda_source

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
    assert "tests/test_paper_reference_oracle.py" in text
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
