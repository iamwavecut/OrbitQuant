"""Runtime provisioning of the native ``orbitquant_packed_matmul`` package.

The native kernel package is resolved in this order:

1. an already importable ``orbitquant_packed_matmul`` package (for example the
   PyPI CPU wheel or a manually installed wheel),
2. a kernel variant directory named by the ``LOCAL_KERNELS`` environment
   variable (``<repo-id>=<variant-dir>``; the same contract the repository
   check scripts use),
3. a previously provisioned variant in the local cache,
4. a prebuilt variant wheel downloaded from the OrbitQuant GitHub release
   (checksum-verified against the release manifest),
5. an opt-in local JIT build from the bundled kernel sources.

Environment contract:

- ``ORBITQUANT_KERNELS_AUTOFETCH=0`` disables the release download step.
- ``ORBITQUANT_KERNELS_AUTOBUILD=1`` enables the local JIT build step.
- ``ORBITQUANT_KERNELS_CACHE`` overrides the cache directory
  (default ``~/.cache/orbitquant/kernels``).
- ``ORBITQUANT_KERNELS_RELEASE_BASE`` overrides the release download base URL.
- ``LOCAL_KERNELS`` points at locally built variant directories.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import re
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

KERNEL_VERSION = 1
KERNEL_PACKAGE_NAME = "orbitquant_packed_matmul"
KERNEL_REPO_ID = "WaveCut/orbitquant-packed-matmul"
_RELEASE_TAG = f"kernels-v{KERNEL_VERSION}"
_DEFAULT_RELEASE_BASE = (
    f"https://github.com/iamwavecut/OrbitQuant/releases/download/{_RELEASE_TAG}"
)
_MANIFEST_FILENAME = "manifest.json"
_PROVISION_MARKER = ".orbitquant-provisioned.json"
_MANIFEST_TIMEOUT_SECONDS = 15
_WHEEL_TIMEOUT_SECONDS = 180
_STABLE_ABI_TORCH_MINIMUM = (2, 11)

_MEMOIZED_REPORT: KernelProvisionReport | None = None


@dataclass(frozen=True)
class KernelProvisionReport:
    requested_variants: tuple[str, ...]
    variant: str | None
    source: str
    sys_path_entry: str | None
    detail: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["requested_variants"] = list(self.requested_variants)
        return payload


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def _torch_version_tuple() -> tuple[int, int] | None:
    match = re.match(r"^(\d+)\.(\d+)", torch.__version__)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _torch_tag() -> str | None:
    parsed = _torch_version_tuple()
    if parsed is None:
        return None
    return f"torch{parsed[0]}{parsed[1]}"


def _normalized_machine() -> str:
    machine = platform.machine().lower()
    if machine in {"arm64", "aarch64"}:
        return "aarch64"
    if machine in {"x86_64", "amd64"}:
        return "x86_64"
    return machine


def _os_name() -> str | None:
    if sys.platform.startswith("linux"):
        return "linux"
    if sys.platform == "darwin":
        return "darwin"
    if sys.platform == "win32":
        return "windows"
    return None


def cuda_variant_name() -> str | None:
    cuda_version = torch.version.cuda
    torch_tag = _torch_tag()
    os_name = _os_name()
    if cuda_version is None or torch_tag is None or os_name in {None, "darwin"}:
        return None
    cuda_tag = "cu" + "".join(cuda_version.split(".")[:2])
    machine = _normalized_machine()
    if os_name == "linux":
        return f"{torch_tag}-cxx11-{cuda_tag}-{machine}-{os_name}"
    return f"{torch_tag}-{cuda_tag}-{machine}-{os_name}"


def metal_variant_name() -> str | None:
    torch_tag = _torch_tag()
    if torch_tag is None or _os_name() != "darwin":
        return None
    return f"{torch_tag}-metal-{_normalized_machine()}-darwin"


def cpu_variant_name() -> str | None:
    parsed = _torch_version_tuple()
    os_name = _os_name()
    if parsed is None or os_name is None or parsed < _STABLE_ABI_TORCH_MINIMUM:
        return None
    abi_tag = "".join(str(part) for part in _STABLE_ABI_TORCH_MINIMUM)
    return f"torch-stable-abi{abi_tag}-cpu-{_normalized_machine()}-{os_name}"


def candidate_variant_names() -> tuple[str, ...]:
    candidates = [cuda_variant_name(), metal_variant_name(), cpu_variant_name()]
    return tuple(name for name in candidates if name is not None)


def kernels_cache_root() -> Path:
    override = os.environ.get("ORBITQUANT_KERNELS_CACHE")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "orbitquant" / "kernels"


def _release_base_url() -> str:
    return os.environ.get(
        "ORBITQUANT_KERNELS_RELEASE_BASE", _DEFAULT_RELEASE_BASE
    ).rstrip("/")


def _variant_dir_is_valid(variant_dir: Path) -> bool:
    package_init = variant_dir / KERNEL_PACKAGE_NAME / "__init__.py"
    return package_init.is_file() and (variant_dir / _PROVISION_MARKER).is_file()


def _attach_to_sys_path(variant_dir: Path) -> str:
    entry = str(variant_dir)
    if entry not in sys.path:
        sys.path.insert(0, entry)
    return entry


def _kernel_package_importable() -> bool:
    return importlib.util.find_spec(KERNEL_PACKAGE_NAME) is not None


def _local_kernels_variant_dirs() -> list[Path]:
    raw = os.environ.get("LOCAL_KERNELS", "")
    directories: list[Path] = []
    for item in raw.split(","):
        item = item.strip()
        if not item or "=" not in item:
            continue
        repo_id, _, path_text = item.partition("=")
        if repo_id.strip() != KERNEL_REPO_ID:
            continue
        path = Path(path_text.strip()).expanduser()
        if (path / KERNEL_PACKAGE_NAME / "__init__.py").is_file():
            directories.append(path)
    return directories


def _cached_variant_dirs(variant: str) -> list[Path]:
    root = kernels_cache_root() / f"v{KERNEL_VERSION}"
    return [root / "prebuilt" / variant, root / "jit" / variant]


def _write_provision_marker(variant_dir: Path, payload: dict[str, Any]) -> None:
    marker = variant_dir / _PROVISION_MARKER
    marker.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _http_get(url: str, timeout: float) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "orbitquant-kernels"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return response.read()


def _fetch_release_manifest() -> dict[str, Any] | None:
    url = f"{_release_base_url()}/{_MANIFEST_FILENAME}"
    try:
        raw = _http_get(url, timeout=_MANIFEST_TIMEOUT_SECONDS)
    except (urllib.error.URLError, OSError, ValueError):
        return None
    try:
        manifest = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(manifest, dict) or not isinstance(manifest.get("variants"), dict):
        return None
    return manifest


def _extract_wheel(wheel_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(wheel_path) as wheel:
        for info in wheel.infolist():
            name = info.filename
            if name.startswith(("/", "\\")) or ".." in Path(name).parts:
                raise RuntimeError(f"refusing to extract unsafe wheel member {name!r}")
        wheel.extractall(destination)


def _fetch_prebuilt_variant(variant: str) -> tuple[Path | None, str]:
    manifest = _fetch_release_manifest()
    if manifest is None:
        return None, f"release manifest unavailable at {_release_base_url()}"
    entry = manifest["variants"].get(variant)
    if not isinstance(entry, dict):
        available = ", ".join(sorted(manifest["variants"])) or "none"
        return None, (
            f"variant {variant} is not published on {_release_base_url()} "
            f"(published: {available})"
        )
    filename = entry.get("filename")
    expected_sha256 = entry.get("sha256")
    if not isinstance(filename, str) or not isinstance(expected_sha256, str):
        return None, f"release manifest entry for {variant} is malformed"

    try:
        payload = _http_get(
            f"{_release_base_url()}/{filename}", timeout=_WHEEL_TIMEOUT_SECONDS
        )
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return None, f"failed to download {filename}: {exc}"

    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != expected_sha256:
        return None, (
            f"checksum mismatch for {filename}: expected {expected_sha256}, "
            f"got {actual_sha256}"
        )

    final_dir = kernels_cache_root() / f"v{KERNEL_VERSION}" / "prebuilt" / variant
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_root = tempfile.mkdtemp(prefix=f"{variant}.", dir=final_dir.parent)
    try:
        staging_dir = Path(staging_root)
        wheel_path = staging_dir / filename
        wheel_path.write_bytes(payload)
        extract_dir = staging_dir / "contents"
        extract_dir.mkdir()
        _extract_wheel(wheel_path, extract_dir)
        if not (extract_dir / KERNEL_PACKAGE_NAME / "__init__.py").is_file():
            return None, f"{filename} does not contain the {KERNEL_PACKAGE_NAME} package"
        _write_provision_marker(
            extract_dir,
            {
                "variant": variant,
                "source": "release",
                "filename": filename,
                "sha256": actual_sha256,
                "kernel_version": KERNEL_VERSION,
            },
        )
        if final_dir.exists():
            shutil.rmtree(final_dir)
        os.replace(extract_dir, final_dir)
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
    return final_dir, f"downloaded {filename}"


def _bundled_kernel_source_root() -> Path | None:
    repo_root = Path(__file__).resolve().parents[3]
    checkout = repo_root / "native-kernels" / "orbitquant-packed-matmul"
    if (checkout / "torch-ext" / "torch_binding.cpp").is_file():
        return checkout
    bundled = Path(__file__).resolve().parents[1] / "_kernel_src"
    if (bundled / "torch-ext" / "torch_binding.cpp").is_file():
        return bundled
    return None


def _jit_backend() -> str:
    if torch.version.cuda is not None:
        return "cuda"
    # The Metal shader library is embedded at kernel-builder build time, so the
    # JIT fallback cannot produce the metal variant; the CPU variant is still
    # a valid build target on macOS.
    return "cpu"


def _jit_variant_name(backend: str) -> str | None:
    if backend == "cuda":
        return cuda_variant_name()
    return cpu_variant_name()


def _maybe_set_cuda_arch_list() -> None:
    if os.environ.get("TORCH_CUDA_ARCH_LIST") or not torch.cuda.is_available():
        return
    supported = [
        int(arch.split("_")[1])
        for arch in torch.cuda.get_arch_list()
        if arch.startswith("sm_")
    ]
    if not supported:
        return
    maximum = max(divmod(arch, 10) for arch in supported)
    architectures: list[str] = []
    for index in range(torch.cuda.device_count()):
        capability = min(maximum, torch.cuda.get_device_capability(index))
        text = f"{capability[0]}.{capability[1]}"
        if text not in architectures:
            architectures.append(text)
    if architectures:
        os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(sorted(architectures))


def _jit_sources(source_root: Path, backend: str) -> list[str]:
    sources = [source_root / "torch-ext" / "torch_binding.cpp"]
    if backend == "cuda":
        sources.append(
            source_root / "orbitquant_packed_matmul_cuda" / "packed_matmul.cu"
        )
    else:
        cpu_dir = source_root / "orbitquant_packed_matmul_cpu"
        sources.extend(sorted(cpu_dir.glob("*.cpp")))
    return [str(path) for path in sources]


def _generated_ops_module(native_module_name: str) -> str:
    return (
        "import sys\n"
        "\n"
        "import torch\n"
        "\n"
        f'if "{native_module_name}" not in sys.modules:\n'
        f"    from . import {native_module_name}  # noqa: F401\n"
        f'ops = getattr(torch.ops, "{native_module_name}")\n'
        "\n"
        "\n"
        "def add_op_namespace_prefix(op_name: str):\n"
        f'    return f"{native_module_name}::{{op_name}}"\n'
    )


def build_native_kernel_package_jit() -> Path:
    """Build the native kernel package from bundled sources into the cache."""

    backend = _jit_backend()
    parsed = _torch_version_tuple()
    if backend == "cpu" and (parsed is None or parsed < _STABLE_ABI_TORCH_MINIMUM):
        minimum = ".".join(str(part) for part in _STABLE_ABI_TORCH_MINIMUM)
        raise RuntimeError(
            f"the CPU JIT build requires torch>={minimum} "
            "(the CPU kernel binds through the torch stable ABI headers); "
            f"found torch {torch.__version__}."
        )
    variant = _jit_variant_name(backend)
    if variant is None:
        raise RuntimeError(
            "could not derive a kernel variant name for this runtime "
            f"(torch {torch.__version__}, platform {sys.platform})."
        )
    source_root = _bundled_kernel_source_root()
    if source_root is None:
        raise RuntimeError(
            "the bundled native kernel sources are not available; reinstall "
            "orbitquant or run from a repository checkout."
        )

    from torch.utils import cpp_extension

    _maybe_set_cuda_arch_list()
    native_module_name = f"_orbitquant_packed_matmul_jit_{backend}"
    define = "-DCUDA_KERNEL" if backend == "cuda" else "-DCPU_KERNEL"
    include_paths = [
        str(Path(__file__).resolve().parent / "_jit_support"),
        str(source_root / "torch-ext"),
        str(source_root / f"orbitquant_packed_matmul_{backend}"),
    ]
    extra_cflags = [define] if sys.platform == "win32" else ["-O3", define]
    module = cpp_extension.load(
        name=native_module_name,
        sources=_jit_sources(source_root, backend),
        extra_include_paths=include_paths,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=["-O3", "-lineinfo", define],
        verbose=_env_flag("ORBITQUANT_KERNELS_BUILD_VERBOSE", False),
    )

    built_library = Path(module.__file__)
    variant_dir = kernels_cache_root() / f"v{KERNEL_VERSION}" / "jit" / variant
    package_dir = variant_dir / KERNEL_PACKAGE_NAME
    package_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        source_root / "torch-ext" / KERNEL_PACKAGE_NAME / "__init__.py",
        package_dir / "__init__.py",
    )
    (package_dir / "_ops.py").write_text(
        _generated_ops_module(native_module_name), encoding="utf-8"
    )
    shutil.copy2(built_library, package_dir / built_library.name)
    (variant_dir / "metadata.json").write_text(
        json.dumps(
            {
                "name": "orbitquant-packed-matmul",
                "id": native_module_name,
                "version": KERNEL_VERSION,
                "backend": {"type": backend},
                "provenance": {"builder": "orbitquant-jit", "torch": torch.__version__},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_provision_marker(
        variant_dir,
        {
            "variant": variant,
            "source": "jit",
            "backend": backend,
            "torch": torch.__version__,
            "kernel_version": KERNEL_VERSION,
        },
    )
    return variant_dir


def _report(
    requested: tuple[str, ...],
    variant: str | None,
    source: str,
    sys_path_entry: str | None,
    detail: str,
) -> KernelProvisionReport:
    return KernelProvisionReport(
        requested_variants=requested,
        variant=variant,
        source=source,
        sys_path_entry=sys_path_entry,
        detail=detail,
    )


def provision_native_kernel_package(
    *,
    allow_fetch: bool | None = None,
    allow_build: bool | None = None,
) -> KernelProvisionReport:
    """Resolve the native kernel package, mutating ``sys.path`` when needed."""

    global _MEMOIZED_REPORT
    if _MEMOIZED_REPORT is not None:
        return _MEMOIZED_REPORT

    fetch_enabled = (
        _env_flag("ORBITQUANT_KERNELS_AUTOFETCH", True)
        if allow_fetch is None
        else allow_fetch
    )
    build_enabled = (
        _env_flag("ORBITQUANT_KERNELS_AUTOBUILD", False)
        if allow_build is None
        else allow_build
    )
    requested = candidate_variant_names()
    failures: list[str] = []

    if _kernel_package_importable():
        report = _report(
            requested,
            None,
            "already-importable",
            None,
            f"{KERNEL_PACKAGE_NAME} is already importable",
        )
        _MEMOIZED_REPORT = report
        return report

    for local_dir in _local_kernels_variant_dirs():
        entry = _attach_to_sys_path(local_dir)
        report = _report(
            requested,
            local_dir.name,
            "local-kernels",
            entry,
            f"attached LOCAL_KERNELS directory {local_dir}",
        )
        _MEMOIZED_REPORT = report
        return report

    for variant in requested:
        for cached_dir in _cached_variant_dirs(variant):
            if _variant_dir_is_valid(cached_dir):
                entry = _attach_to_sys_path(cached_dir)
                report = _report(
                    requested,
                    variant,
                    "cache",
                    entry,
                    f"attached cached variant {cached_dir}",
                )
                _MEMOIZED_REPORT = report
                return report

    if fetch_enabled:
        for variant in requested:
            fetched_dir, detail = _fetch_prebuilt_variant(variant)
            if fetched_dir is not None:
                entry = _attach_to_sys_path(fetched_dir)
                report = _report(requested, variant, "release", entry, detail)
                _MEMOIZED_REPORT = report
                return report
            failures.append(detail)
    else:
        failures.append("release download disabled (ORBITQUANT_KERNELS_AUTOFETCH=0)")

    if build_enabled:
        try:
            built_dir = build_native_kernel_package_jit()
        except Exception as exc:  # noqa: BLE001 - reported through the provision detail
            failures.append(f"JIT build failed: {exc}")
        else:
            entry = _attach_to_sys_path(built_dir)
            report = _report(
                requested,
                built_dir.name,
                "jit",
                entry,
                f"built {built_dir}",
            )
            _MEMOIZED_REPORT = report
            return report
    else:
        failures.append("JIT build disabled (set ORBITQUANT_KERNELS_AUTOBUILD=1 to enable)")

    report = _report(
        requested,
        None,
        "unavailable",
        None,
        "; ".join(failures) if failures else "no provisioning path succeeded",
    )
    _MEMOIZED_REPORT = report
    return report


def provision_status() -> dict[str, Any]:
    """Describe provisioning state without mutating ``sys.path`` or the cache."""

    requested = candidate_variant_names()
    cached: dict[str, str] = {}
    for variant in requested:
        for cached_dir in _cached_variant_dirs(variant):
            if _variant_dir_is_valid(cached_dir):
                cached[variant] = str(cached_dir)
                break
    return {
        "kernel_version": KERNEL_VERSION,
        "package": KERNEL_PACKAGE_NAME,
        "importable": _kernel_package_importable(),
        "candidate_variants": list(requested),
        "cache_root": str(kernels_cache_root()),
        "cached_variants": cached,
        "local_kernels": [str(path) for path in _local_kernels_variant_dirs()],
        "release_base": _release_base_url(),
        "autofetch": _env_flag("ORBITQUANT_KERNELS_AUTOFETCH", True),
        "autobuild": _env_flag("ORBITQUANT_KERNELS_AUTOBUILD", False),
        "bundled_sources": str(_bundled_kernel_source_root() or ""),
    }


def _reset_provision_memo_for_tests() -> None:
    global _MEMOIZED_REPORT
    _MEMOIZED_REPORT = None
