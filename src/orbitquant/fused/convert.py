"""Fused copies of published ``orbitquant-v1`` component artifacts.

``fuse_component_artifact`` rewrites such an artifact so that ``model.safetensors`` holds the
fused groups instead of the per-projection tensors and ``quantization_config.json`` carries the
``fused_layout``; every other file is copied. The manifest keeps the projection list and the
per-module widths the loader rebuilds the groups from. Diffusers-format components need no
helper: load the transformer with ``from_pretrained``, ``fuse`` it and save it with
``orbitquant.fused.save_pretrained``.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path

from safetensors.torch import save_file
from torch import nn

from orbitquant.artifacts.checksums import (
    is_ignored_artifact_relative_path,
    sha256_file,
    write_sha256sums,
)
from orbitquant.artifacts.loader import load_orbitquant_artifact
from orbitquant.artifacts.manifest import OrbitQuantManifest
from orbitquant.config import OrbitQuantConfig
from orbitquant.fused.install import fuse

_REWRITTEN = ("model.safetensors", "quantization_config.json", "orbitquant_manifest.json")


def fuse_component_artifact(
    model: nn.Module,
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    dense: str | None = None,
    validate_checksums: bool = True,
) -> OrbitQuantManifest:
    """Load the artifact at ``source_dir`` into ``model`` (the unquantized component skeleton),
    fuse it and write the fused artifact to ``output_dir``. Rewrite the model card and refresh
    ``SHA256SUMS`` (``write_sha256sums``) after editing it."""
    source, output = Path(source_dir), Path(output_dir)
    if output.resolve() == source.resolve():
        raise ValueError("write the fused artifact to a new directory")
    config = OrbitQuantConfig.from_dict(
        json.loads((source / "quantization_config.json").read_text(encoding="utf-8"))
    )
    if config.fused_layout:
        raise ValueError(f"{source} is already fused")
    manifest = load_orbitquant_artifact(model, source, validate_checksums=validate_checksums)
    config.fused_layout = fuse(model, dense=dense, config=config)

    output.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source).as_posix()
        if (
            path.is_dir()
            or relative in _REWRITTEN
            or relative == "SHA256SUMS"
            or is_ignored_artifact_relative_path(relative)
        ):
            continue
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)

    state = {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}
    save_file(state, output / "model.safetensors")
    (output / "quantization_config.json").write_text(
        json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    fused_manifest = dataclasses.replace(
        manifest,
        module_shapes={name: list(tensor.shape) for name, tensor in state.items()},
        checksums={name: sha256_file(output / name) for name in manifest.checksums},
    )
    (output / "orbitquant_manifest.json").write_text(
        json.dumps(fused_manifest.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    write_sha256sums(output)
    return fused_manifest
