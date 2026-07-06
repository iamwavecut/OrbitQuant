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
