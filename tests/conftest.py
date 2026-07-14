import os
import tempfile

import pytest

import orbitquant.kernels.dispatch as _dispatch_module

# The test run must never download kernel variants from the GitHub release:
# native probes would otherwise provision a real package mid-suite (network
# flakiness plus an importable orbitquant_packed_matmul leaking into every
# later test). Provision tests mock the transport explicitly. The cache is
# also redirected so the suite neither reads nor pollutes the user-level
# ~/.cache/orbitquant/kernels directory.
os.environ.setdefault("ORBITQUANT_KERNELS_AUTOFETCH", "0")
os.environ.setdefault(
    "ORBITQUANT_KERNELS_CACHE", tempfile.mkdtemp(prefix="orbitquant-test-kernels-")
)


@pytest.fixture(autouse=True)
def _clear_backend_availability_cache():
    """Backend availability is memoized per process; tests monkeypatch the
    underlying probes, so reset the cache around every test."""
    _dispatch_module.clear_backend_availability_cache()
    yield
    _dispatch_module.clear_backend_availability_cache()
