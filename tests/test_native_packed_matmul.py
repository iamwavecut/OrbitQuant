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


def test_native_packed_matmul_loader_uses_versioned_hf_kernel(monkeypatch):
    calls = []
    fake_kernel = SimpleNamespace(matmul_packed_weight=lambda *args, **kwargs: None)
    fake_direct_kernel = SimpleNamespace(matmul_packed_weight=lambda *args, **kwargs: None)

    def fake_get_kernel(repo_id, *, version, trust_remote_code):
        calls.append((repo_id, version, trust_remote_code))
        return fake_kernel

    monkeypatch.setattr(native_module, "_NATIVE_KERNEL", None)
    monkeypatch.setitem(sys.modules, "orbitquant_packed_matmul", fake_direct_kernel)
    monkeypatch.setitem(sys.modules, "kernels", SimpleNamespace(get_kernel=fake_get_kernel))

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


def test_native_packed_matmul_loader_falls_back_to_importable_package_after_hf_error(
    monkeypatch,
):
    fake_kernel = SimpleNamespace(matmul_packed_weight=lambda *args, **kwargs: None)

    def fail_get_kernel(*args, **kwargs):
        raise RuntimeError("Hub kernel unavailable")

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

    with pytest.raises(RuntimeError, match="orbitquant_packed_matmul"):
        native_module._load_native_packed_matmul_kernel()
