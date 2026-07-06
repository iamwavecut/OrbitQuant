class OrbitQuantError(RuntimeError):
    """Base error for OrbitQuant failures."""


class OrbitQuantConfigError(ValueError):
    """Raised when an OrbitQuant config is inconsistent."""
