import builtins
import sys
from types import SimpleNamespace

import pytest
import torch

import orbitquant.kernels.native_packed_matmul as native_module
from orbitquant.codebooks import get_codebook


def test_native_packed_matmul_rejects_cpu_before_loading_kernel():
    codebook = get_codebook(dim=16, bits=4)

    with pytest.raises(RuntimeError, match="requires CUDA or MPS input tensors"):
        native_module.matmul_packed_weight_with_native_kernel(
            torch.randn(2, 16),
            torch.empty(56, dtype=torch.uint8),
            torch.ones(7, dtype=torch.bfloat16),
            codebook,
            bits=4,
            out_features=7,
            in_features=16,
        )


def test_native_packed_matmul_loader_prefers_importable_package_before_hf_kernel(
    monkeypatch,
):
    calls = []
    fake_kernel = SimpleNamespace(matmul_packed_weight=lambda *args, **kwargs: None)
    fake_direct_kernel = SimpleNamespace(matmul_packed_weight=lambda *args, **kwargs: None)

    def fake_get_kernel(repo_id, *, version, trust_remote_code):
        calls.append((repo_id, version, trust_remote_code))
        return fake_kernel

    monkeypatch.setattr(native_module, "_NATIVE_KERNEL", None)
    monkeypatch.setitem(sys.modules, "orbitquant_packed_matmul", fake_direct_kernel)
    monkeypatch.setitem(sys.modules, "kernels", SimpleNamespace(get_kernel=fake_get_kernel))

    assert native_module._load_native_packed_matmul_kernel() is fake_direct_kernel
    assert calls == []


def test_native_packed_matmul_loader_accepts_kernel_builder_variant_layout(
    tmp_path,
    monkeypatch,
):
    previous_package = sys.modules.pop("orbitquant_packed_matmul", None)
    previous_variant = sys.modules.pop("orbitquant_packed_matmul_variant_test", None)
    variant = tmp_path / "torch999-cxx11-cu999-x86_64-linux"
    package_dir = variant / "orbitquant_packed_matmul"
    package_dir.mkdir(parents=True)
    (variant / "metadata.json").write_text('{"name": "orbitquant-packed-matmul"}')
    (variant / "__init__.py").write_text(
        "def matmul_packed_weight(*args, **kwargs):\n"
        "    return ('variant-root', args, kwargs)\n",
        encoding="utf-8",
    )
    (package_dir / "__init__.py").write_text(
        "import importlib.util\n"
        "import sys\n"
        "from pathlib import Path\n"
        "module_name = 'orbitquant_packed_matmul_variant_test'\n"
        "spec = importlib.util.spec_from_file_location(\n"
        "    module_name, Path(__file__).parent.parent / '__init__.py'\n"
        ")\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "sys.modules[module_name] = module\n"
        "spec.loader.exec_module(module)\n"
        "globals().update(vars(module))\n",
        encoding="utf-8",
    )

    def fail_get_kernel(*args, **kwargs):
        raise AssertionError("Hub kernel should not be probed for an importable variant")

    monkeypatch.setattr(native_module, "_NATIVE_KERNEL", None)
    monkeypatch.syspath_prepend(str(variant))
    monkeypatch.setitem(sys.modules, "kernels", SimpleNamespace(get_kernel=fail_get_kernel))

    try:
        kernel = native_module._load_native_packed_matmul_kernel()

        assert kernel.matmul_packed_weight("x") == ("variant-root", ("x",), {})
    finally:
        native_module._NATIVE_KERNEL = None
        sys.modules.pop("orbitquant_packed_matmul", None)
        sys.modules.pop("orbitquant_packed_matmul_variant_test", None)
        if previous_package is not None:
            sys.modules["orbitquant_packed_matmul"] = previous_package
        if previous_variant is not None:
            sys.modules["orbitquant_packed_matmul_variant_test"] = previous_variant


def test_native_packed_matmul_loader_uses_versioned_hf_kernel_when_import_missing(
    monkeypatch,
):
    calls = []
    fake_kernel = SimpleNamespace(matmul_packed_weight=lambda *args, **kwargs: None)
    real_import = builtins.__import__

    def fake_get_kernel(repo_id, *, version, trust_remote_code):
        calls.append((repo_id, version, trust_remote_code))
        return fake_kernel

    def fake_import(name, *args, **kwargs):
        if name == "orbitquant_packed_matmul":
            raise ImportError("missing importable native package")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(native_module, "_NATIVE_KERNEL", None)
    monkeypatch.delitem(sys.modules, "orbitquant_packed_matmul", raising=False)
    monkeypatch.setitem(sys.modules, "kernels", SimpleNamespace(get_kernel=fake_get_kernel))
    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert native_module._load_native_packed_matmul_kernel() is fake_kernel
    assert calls == [("WaveCut/orbitquant-packed-matmul", 1, True)]


def test_native_packed_matmul_loader_falls_back_to_importable_package_without_hf_kernels(
    monkeypatch,
):
    fake_kernel = SimpleNamespace(matmul_packed_weight=lambda *args, **kwargs: None)

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "kernels":
            raise ImportError("missing Hugging Face kernels package")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(native_module, "_NATIVE_KERNEL", None)
    monkeypatch.setitem(sys.modules, "orbitquant_packed_matmul", fake_kernel)
    monkeypatch.delitem(sys.modules, "kernels", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert native_module._load_native_packed_matmul_kernel() is fake_kernel


def test_native_packed_matmul_loader_does_not_probe_hf_when_importable_package_exists(
    monkeypatch,
):
    fake_kernel = SimpleNamespace(matmul_packed_weight=lambda *args, **kwargs: None)

    def fail_get_kernel(*args, **kwargs):
        raise AssertionError("Hub kernel should not be probed")

    monkeypatch.setattr(native_module, "_NATIVE_KERNEL", None)
    monkeypatch.setitem(sys.modules, "orbitquant_packed_matmul", fake_kernel)
    monkeypatch.setitem(sys.modules, "kernels", SimpleNamespace(get_kernel=fail_get_kernel))

    assert native_module._load_native_packed_matmul_kernel() is fake_kernel


def test_native_packed_matmul_loader_reports_missing_optional_dependency(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in {"orbitquant_packed_matmul", "kernels"}:
            raise ImportError("missing kernels package")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(native_module, "_NATIVE_KERNEL", None)
    monkeypatch.delitem(sys.modules, "orbitquant_packed_matmul", raising=False)
    monkeypatch.delitem(sys.modules, "kernels", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="orbitquant_packed_matmul") as exc_info:
        native_module._load_native_packed_matmul_kernel()

    message = str(exc_info.value)
    assert "built kernel variant directory" in message
    assert "metadata.json" in message
    assert "PYTHONPATH" in message
    assert "Current runtime is torch" in message
    assert "The built kernel variant must match this runtime." in message
    if torch.version.cuda is not None and sys.platform == "linux":
        assert "Expected kernel-builder CUDA variant:" in message
