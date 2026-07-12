import pytest

import orbitquant.kernels.dispatch as _dispatch_module


@pytest.fixture(autouse=True)
def _clear_backend_availability_cache():
    """Backend availability is memoized per process; tests monkeypatch the
    underlying probes, so reset the cache around every test."""
    _dispatch_module.clear_backend_availability_cache()
    yield
    _dispatch_module.clear_backend_availability_cache()
