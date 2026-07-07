from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_sha256sums(root: str | Path, output: str | Path | None = None) -> Path:
    root_path = Path(root)
    output_path = root_path / "SHA256SUMS" if output is None else Path(output)
    lines: list[str] = []
    for path in sorted(root_path.rglob("*")):
        if not path.is_file() or path == output_path:
            continue
        rel = path.relative_to(root_path)
        lines.append(f"{sha256_file(path)}  {rel.as_posix()}")
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return output_path


def read_sha256sums(path: str | Path) -> dict[str, str]:
    path = Path(path)
    if not path.is_file():
        return {}
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, _, relative_path = line.partition("  ")
        if digest and relative_path:
            entries[relative_path] = digest
    return entries


def write_sha256sums_from_manifest(
    root: str | Path,
    checksums: dict[str, str],
    output: str | Path | None = None,
) -> Path:
    root_path = Path(root)
    output_path = root_path / "SHA256SUMS" if output is None else Path(output)
    entries = read_sha256sums(output_path)
    entries.update(checksums)

    manifest_path = root_path / "orbitquant_manifest.json"
    if manifest_path.is_file():
        entries["orbitquant_manifest.json"] = sha256_file(manifest_path)
    readme_path = root_path / "README.md"
    if readme_path.is_file() and "README.md" not in entries:
        entries["README.md"] = sha256_file(readme_path)

    entries = {
        relative_path: digest
        for relative_path, digest in entries.items()
        if relative_path != "SHA256SUMS" and (root_path / relative_path).is_file()
    }
    output_path.write_text(
        "\n".join(f"{digest}  {relative_path}" for relative_path, digest in sorted(entries.items()))
        + ("\n" if entries else ""),
        encoding="utf-8",
    )
    return output_path


def validate_checksums(root: str | Path, checksums: dict[str, str]) -> None:
    root_path = Path(root)
    for relative_path, expected in checksums.items():
        path = root_path / relative_path
        if not path.is_file():
            raise RuntimeError(f"artifact checksum target missing: {relative_path}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"artifact checksum mismatch for {relative_path}: "
                f"expected {expected}, got {actual}"
            )


def validate_sha256sums(
    root: str | Path,
    *,
    required_paths: tuple[str, ...] = (),
    sha256sums_path: str | Path | None = None,
) -> dict[str, str]:
    root_path = Path(root)
    sums_path = root_path / "SHA256SUMS" if sha256sums_path is None else Path(sha256sums_path)
    entries = read_sha256sums(sums_path)
    if not entries:
        raise RuntimeError("SHA256SUMS is empty or missing")
    missing_entries = sorted(set(required_paths) - set(entries))
    if missing_entries:
        raise RuntimeError(f"SHA256SUMS missing entries: {missing_entries}")
    for relative_path, expected in entries.items():
        if relative_path == "SHA256SUMS":
            raise RuntimeError("SHA256SUMS must not include itself")
        path = root_path / relative_path
        if not path.is_file():
            raise RuntimeError(f"SHA256SUMS target missing: {relative_path}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"SHA256SUMS mismatch for {relative_path}: expected {expected}, got {actual}"
            )
    return entries
