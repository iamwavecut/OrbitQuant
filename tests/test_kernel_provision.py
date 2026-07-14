import hashlib
import importlib
import io
import json
import sys
import zipfile
from pathlib import Path

import pytest

from orbitquant.kernels import provision


@pytest.fixture(autouse=True)
def _reset_provision_state(monkeypatch, tmp_path):
    provision._reset_provision_memo_for_tests()
    # Earlier tests may leave a fake orbitquant_packed_matmul in sys.modules
    # (importlib.util.find_spec consults it first), so isolate module state.
    sys.modules.pop("orbitquant_packed_matmul", None)
    importlib.invalidate_caches()
    monkeypatch.setenv("ORBITQUANT_KERNELS_CACHE", str(tmp_path / "kernels-cache"))
    monkeypatch.delenv("LOCAL_KERNELS", raising=False)
    monkeypatch.delenv("ORBITQUANT_KERNELS_AUTOFETCH", raising=False)
    monkeypatch.delenv("ORBITQUANT_KERNELS_AUTOBUILD", raising=False)
    monkeypatch.delenv("ORBITQUANT_KERNELS_RELEASE_BASE", raising=False)
    original_sys_path = list(sys.path)
    yield
    sys.path[:] = original_sys_path
    sys.modules.pop("orbitquant_packed_matmul", None)
    importlib.invalidate_caches()
    provision._reset_provision_memo_for_tests()


def _patch_runtime(
    monkeypatch,
    *,
    torch_version="2.12.1",
    cuda=None,
    platform_name="linux",
    machine="x86_64",
):
    monkeypatch.setattr(provision.torch, "__version__", torch_version)
    monkeypatch.setattr(provision.torch.version, "cuda", cuda)
    monkeypatch.setattr(provision.sys, "platform", platform_name)
    monkeypatch.setattr(provision.platform, "machine", lambda: machine)


def test_cuda_variant_name_linux(monkeypatch):
    _patch_runtime(monkeypatch, torch_version="2.9.1", cuda="12.8")
    assert provision.cuda_variant_name() == "torch29-cxx11-cu128-x86_64-linux"


def test_cuda_variant_name_windows_has_no_cxx11_segment(monkeypatch):
    _patch_runtime(
        monkeypatch, torch_version="2.13.0", cuda="13.0", platform_name="win32", machine="AMD64"
    )
    assert provision.cuda_variant_name() == "torch213-cu130-x86_64-windows"


def test_metal_variant_name(monkeypatch):
    _patch_runtime(monkeypatch, torch_version="2.12.1", platform_name="darwin", machine="arm64")
    assert provision.metal_variant_name() == "torch212-metal-aarch64-darwin"


def test_cpu_variant_name_uses_stable_abi_prefix(monkeypatch):
    _patch_runtime(monkeypatch, torch_version="2.12.1")
    assert provision.cpu_variant_name() == "torch-stable-abi211-cpu-x86_64-linux"


def test_cpu_variant_name_requires_stable_abi_torch(monkeypatch):
    _patch_runtime(monkeypatch, torch_version="2.9.1")
    assert provision.cpu_variant_name() is None


def test_candidate_order_prefers_accelerator(monkeypatch):
    _patch_runtime(monkeypatch, torch_version="2.12.1", cuda="12.6")
    assert provision.candidate_variant_names() == (
        "torch212-cxx11-cu126-x86_64-linux",
        "torch-stable-abi211-cpu-x86_64-linux",
    )


def _make_variant_dir(root: Path, variant: str, *, marker=True) -> Path:
    variant_dir = root / variant
    package_dir = variant_dir / provision.KERNEL_PACKAGE_NAME
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text(
        "PROVISION_TEST_SENTINEL = True\n", encoding="utf-8"
    )
    if marker:
        (variant_dir / provision._PROVISION_MARKER).write_text("{}", encoding="utf-8")
    return variant_dir


def test_provision_uses_cached_variant(monkeypatch):
    _patch_runtime(monkeypatch, torch_version="2.12.1")
    cache_root = provision.kernels_cache_root() / f"v{provision.KERNEL_VERSION}" / "prebuilt"
    variant = provision.cpu_variant_name()
    variant_dir = _make_variant_dir(cache_root, variant)

    report = provision.provision_native_kernel_package(allow_fetch=False, allow_build=False)

    assert report.source == "cache"
    assert report.variant == variant
    assert report.sys_path_entry == str(variant_dir)
    assert str(variant_dir) in sys.path


def test_provision_ignores_cache_without_marker(monkeypatch):
    _patch_runtime(monkeypatch, torch_version="2.12.1")
    cache_root = provision.kernels_cache_root() / f"v{provision.KERNEL_VERSION}" / "prebuilt"
    _make_variant_dir(cache_root, provision.cpu_variant_name(), marker=False)

    report = provision.provision_native_kernel_package(allow_fetch=False, allow_build=False)

    assert report.source == "unavailable"


def test_provision_uses_local_kernels_env(monkeypatch, tmp_path):
    _patch_runtime(monkeypatch, torch_version="2.12.1")
    local_dir = _make_variant_dir(tmp_path, "torch212-cxx11-cu126-x86_64-linux", marker=False)
    monkeypatch.setenv("LOCAL_KERNELS", f"{provision.KERNEL_REPO_ID}={local_dir}")

    report = provision.provision_native_kernel_package(allow_fetch=False, allow_build=False)

    assert report.source == "local-kernels"
    assert report.sys_path_entry == str(local_dir)


def test_provision_memoizes_negative_result(monkeypatch):
    _patch_runtime(monkeypatch, torch_version="2.12.1")
    calls = []

    def counting_fetch(variant):
        calls.append(variant)
        return None, "offline"

    monkeypatch.setattr(provision, "_fetch_prebuilt_variant", counting_fetch)
    first = provision.provision_native_kernel_package(allow_build=False)
    second = provision.provision_native_kernel_package(allow_build=False)

    assert first.source == "unavailable"
    assert second is first
    assert len(calls) == 1


def _wheel_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as wheel:
        wheel.writestr(
            f"{provision.KERNEL_PACKAGE_NAME}/__init__.py",
            "PROVISION_TEST_SENTINEL = True\n",
        )
        wheel.writestr("orbitquant_packed_matmul-1.0.dist-info/METADATA", "stub")
    return buffer.getvalue()


def _serve_release(monkeypatch, variant: str, wheel_payload: bytes, *, sha256=None):
    filename = f"{variant}.whl"
    manifest = {
        "kernel_version": provision.KERNEL_VERSION,
        "variants": {
            variant: {
                "filename": filename,
                "sha256": sha256 or hashlib.sha256(wheel_payload).hexdigest(),
            }
        },
    }

    def fake_http_get(url, timeout):
        if url.endswith(provision._MANIFEST_FILENAME):
            return json.dumps(manifest).encode("utf-8")
        if url.endswith(filename):
            return wheel_payload
        raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr(provision, "_http_get", fake_http_get)


def test_provision_fetches_release_wheel(monkeypatch):
    _patch_runtime(monkeypatch, torch_version="2.12.1")
    variant = provision.cpu_variant_name()
    _serve_release(monkeypatch, variant, _wheel_bytes())

    report = provision.provision_native_kernel_package(allow_build=False)

    assert report.source == "release"
    assert report.variant == variant
    variant_dir = Path(report.sys_path_entry)
    assert (variant_dir / provision.KERNEL_PACKAGE_NAME / "__init__.py").is_file()
    assert (variant_dir / provision._PROVISION_MARKER).is_file()

    provision._reset_provision_memo_for_tests()
    sys.path.remove(str(variant_dir))
    importlib.invalidate_caches()
    monkeypatch.setattr(
        provision, "_http_get", lambda url, timeout: (_ for _ in ()).throw(OSError("offline"))
    )
    cached = provision.provision_native_kernel_package(allow_build=False)
    assert cached.source == "cache"
    assert cached.variant == variant


def test_provision_rejects_checksum_mismatch(monkeypatch):
    _patch_runtime(monkeypatch, torch_version="2.12.1")
    variant = provision.cpu_variant_name()
    _serve_release(monkeypatch, variant, _wheel_bytes(), sha256="0" * 64)

    report = provision.provision_native_kernel_package(allow_build=False)

    assert report.source == "unavailable"
    assert "checksum mismatch" in report.detail
    cache_dir = (
        provision.kernels_cache_root() / f"v{provision.KERNEL_VERSION}" / "prebuilt" / variant
    )
    assert not cache_dir.exists()


def test_provision_respects_autofetch_disable(monkeypatch):
    _patch_runtime(monkeypatch, torch_version="2.12.1")
    monkeypatch.setenv("ORBITQUANT_KERNELS_AUTOFETCH", "0")

    def unexpected_fetch(variant):
        raise AssertionError("fetch must not run when autofetch is disabled")

    monkeypatch.setattr(provision, "_fetch_prebuilt_variant", unexpected_fetch)
    report = provision.provision_native_kernel_package(allow_build=False)

    assert report.source == "unavailable"
    assert "AUTOFETCH=0" in report.detail


def test_extract_wheel_rejects_traversal(tmp_path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as wheel:
        wheel.writestr("../escape.py", "payload")
    wheel_path = tmp_path / "bad.whl"
    wheel_path.write_bytes(buffer.getvalue())

    with pytest.raises(RuntimeError, match="unsafe wheel member"):
        provision._extract_wheel(wheel_path, tmp_path / "out")


def test_provision_status_reports_without_side_effects(monkeypatch):
    _patch_runtime(monkeypatch, torch_version="2.12.1", cuda="12.8")
    status = provision.provision_status()

    assert status["kernel_version"] == provision.KERNEL_VERSION
    assert status["candidate_variants"] == [
        "torch212-cxx11-cu128-x86_64-linux",
        "torch-stable-abi211-cpu-x86_64-linux",
    ]
    assert status["autofetch"] is True
    assert status["autobuild"] is False
    # A repository checkout resolves to native-kernels/orbitquant-packed-matmul,
    # an installed wheel resolves to the bundled orbitquant/_kernel_src copy.
    assert Path(status["bundled_sources"]).name in {"orbitquant-packed-matmul", "_kernel_src"}


def test_bundled_source_root_prefers_repo_checkout():
    root = provision._bundled_kernel_source_root()
    assert root is not None
    assert (root / "torch-ext" / "torch_binding.cpp").is_file()
