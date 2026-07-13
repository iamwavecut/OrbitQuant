import builtins
import sys
from types import SimpleNamespace

import pytest
import torch

import orbitquant.kernels.native_packed_matmul as native_module
from orbitquant.codebooks import get_codebook


def test_native_packed_matmul_rejects_variant_without_cpu_backend(monkeypatch):
    codebook = get_codebook(dim=16, bits=4)
    fake_kernel = SimpleNamespace(
        matmul_packed_weight=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("unsupported CPU variant must not be invoked")
        ),
        supports_device=lambda device_type: device_type == "mps",
    )
    monkeypatch.setattr(native_module, "_NATIVE_KERNEL", fake_kernel)

    with pytest.raises(RuntimeError, match="does not contain a CPU backend"):
        native_module.matmul_packed_weight_with_native_kernel(
            torch.randn(2, 16),
            torch.empty(56, dtype=torch.uint8),
            torch.ones(7, dtype=torch.bfloat16),
            codebook,
            bits=4,
            out_features=7,
            in_features=16,
        )


def test_native_device_capability_treats_legacy_variants_as_accelerator_only(monkeypatch):
    fake_legacy_kernel = SimpleNamespace(matmul_packed_weight=lambda *args, **kwargs: None)
    monkeypatch.setattr(native_module, "_NATIVE_KERNEL", fake_legacy_kernel)

    assert native_module.native_packed_matmul_device_available("cpu") is False
    assert native_module.native_packed_matmul_device_available("mps") is True


def test_native_cpu_adaln_availability_requires_capability_operation(monkeypatch):
    fake_kernel = SimpleNamespace(
        supports_cpu_adaln=lambda: True,
        matmul_packed_adaln_int4_cpu=lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(native_module, "_NATIVE_KERNEL", fake_kernel)

    assert native_module.native_cpu_adaln_available() is True


def test_native_cpu_adaln_bridge_rejects_legacy_variant(monkeypatch):
    fake_kernel = SimpleNamespace(matmul_packed_adaln_int4_cpu=lambda *args, **kwargs: None)
    monkeypatch.setattr(native_module, "_NATIVE_KERNEL", fake_kernel)

    with pytest.raises(RuntimeError, match="does not provide the CPU AdaLN"):
        native_module.matmul_packed_adaln_int4_with_native_cpu_kernel(
            torch.randn(2, 8, dtype=torch.bfloat16),
            torch.empty(32, dtype=torch.uint8),
            torch.ones(4, 1, dtype=torch.bfloat16),
            out_features=4,
            in_features=8,
            group_size=8,
        )


def test_native_packed_w4_activation_rejects_cpu_before_loading_kernel():
    with pytest.raises(RuntimeError, match="requires CUDA tensors"):
        native_module.quantize_activations_packed_w4_with_native_kernel(
            torch.randn(2, 512),
            torch.arange(512),
            torch.ones(512, dtype=torch.int8),
            torch.linspace(-1.0, 1.0, 15),
            eps=1e-12,
            inv_sqrt_block=512**-0.5,
        )


def test_native_packed_w4_activation_availability_checks_operation(monkeypatch):
    fake_kernel = SimpleNamespace(quantize_activations_packed_w4=lambda *args, **kwargs: None)
    monkeypatch.setattr(native_module, "_NATIVE_KERNEL", fake_kernel)

    assert native_module.native_packed_w4_activation_available()


def test_native_int8_activation_rejects_cpu_before_loading_kernel():
    with pytest.raises(RuntimeError, match="requires CUDA tensors"):
        native_module.quantize_activations_int8_with_native_kernel(
            torch.randn(2, 512),
            torch.arange(512),
            torch.ones(512, dtype=torch.int8),
            torch.linspace(-1.0, 1.0, 15),
            torch.arange(-8, 8, dtype=torch.int8),
            eps=1e-12,
            inv_sqrt_block=512**-0.5,
        )


def test_native_int8_activation_availability_checks_operation(monkeypatch):
    fake_kernel = SimpleNamespace(quantize_activations_int8=lambda *args, **kwargs: None)
    monkeypatch.setattr(native_module, "_NATIVE_KERNEL", fake_kernel)

    assert native_module.native_int8_activation_available()


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


def test_native_packed_matmul_loader_provisions_when_import_missing(monkeypatch):
    fake_kernel = SimpleNamespace(matmul_packed_weight=lambda *args, **kwargs: None)
    real_import = builtins.__import__
    provisioned = []

    def fake_import(name, *args, **kwargs):
        if name == "orbitquant_packed_matmul" and not provisioned:
            raise ImportError("missing importable native package")
        return real_import(name, *args, **kwargs)

    def fake_provision():
        provisioned.append(True)
        monkeypatch.setitem(sys.modules, "orbitquant_packed_matmul", fake_kernel)
        return SimpleNamespace(sys_path_entry="/tmp/orbitquant-variant", detail="test")

    monkeypatch.setattr(native_module, "_NATIVE_KERNEL", None)
    monkeypatch.delitem(sys.modules, "orbitquant_packed_matmul", raising=False)
    monkeypatch.setattr(
        native_module.provision, "provision_native_kernel_package", fake_provision
    )
    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert native_module._load_native_packed_matmul_kernel() is fake_kernel
    assert provisioned == [True]


def test_native_packed_matmul_loader_uses_importable_package_without_provisioning(
    monkeypatch,
):
    fake_kernel = SimpleNamespace(matmul_packed_weight=lambda *args, **kwargs: None)

    def fail_provision(*args, **kwargs):
        raise AssertionError("provisioning must not run when the package is importable")

    monkeypatch.setattr(native_module, "_NATIVE_KERNEL", None)
    monkeypatch.setitem(sys.modules, "orbitquant_packed_matmul", fake_kernel)
    monkeypatch.setattr(
        native_module.provision, "provision_native_kernel_package", fail_provision
    )

    assert native_module._load_native_packed_matmul_kernel() is fake_kernel


def test_native_packed_matmul_loader_reports_provisioning_failure(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "orbitquant_packed_matmul":
            raise ImportError("missing importable native package")
        return real_import(name, *args, **kwargs)

    def unavailable_provision():
        return SimpleNamespace(sys_path_entry=None, detail="offline for the test")

    monkeypatch.setattr(native_module, "_NATIVE_KERNEL", None)
    monkeypatch.delitem(sys.modules, "orbitquant_packed_matmul", raising=False)
    monkeypatch.setattr(
        native_module.provision, "provision_native_kernel_package", unavailable_provision
    )
    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="orbitquant_packed_matmul") as exc_info:
        native_module._load_native_packed_matmul_kernel()

    message = str(exc_info.value)
    assert "pip install orbitquant-packed-matmul" in message
    assert "kernels-install" in message
    assert "built kernel variant directory" in message
    assert "offline for the test" in message
    assert "Current runtime is torch" in message
    assert "The built kernel variant must match this runtime." in message
    if torch.version.cuda is not None and sys.platform == "linux":
        assert "Expected kernel-builder CUDA variant:" in message
