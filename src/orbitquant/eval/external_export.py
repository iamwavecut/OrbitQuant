from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png", ".webp"}
_VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm"}


@dataclass(frozen=True)
class ExternalExportResult:
    artifact_dir: str
    split: str
    output_dir: str
    sample_count: int
    prompt_count: int
    manifest_path: str


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _artifact_path(artifact_path: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else artifact_path / path


def _record_output_path(artifact_path: Path, record: dict[str, Any]) -> Path | None:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        return None
    output_path = metadata.get("output_path")
    if not output_path:
        return None
    return _artifact_path(artifact_path, output_path)


def _prompt_record(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        return {}
    prompt_record = metadata.get("prompt_record")
    return dict(prompt_record) if isinstance(prompt_record, dict) else {}


def _prompt_text(record: dict[str, Any]) -> str | None:
    prompt_record = _prompt_record(record)
    if isinstance(prompt_record.get("prompt"), str):
        return str(prompt_record["prompt"])
    metadata = record.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("prompt"), str):
        return str(metadata["prompt"])
    return None


def _geneval_metadata(record: dict[str, Any]) -> dict[str, Any]:
    prompt_record = _prompt_record(record)
    source = prompt_record.get("geneval", prompt_record)
    if not isinstance(source, dict):
        source = {}
    prompt = _prompt_text(record)
    metadata = {
        "tag": source.get("tag"),
        "prompt": source.get("prompt", prompt),
        "include": source.get("include"),
        "exclude": source.get("exclude", []),
    }
    missing = [
        key
        for key in ("tag", "prompt", "include")
        if metadata.get(key) in (None, "")
    ]
    if missing:
        prompt_id = prompt_record.get("id", "<unknown>")
        raise ValueError(
            "record is not GenEval-compatible: prompt_record "
            f"{prompt_id!r} is missing {', '.join(missing)}"
        )
    if not isinstance(metadata["include"], list):
        raise ValueError("GenEval metadata 'include' must be a list")
    if not isinstance(metadata["exclude"], list):
        raise ValueError("GenEval metadata 'exclude' must be a list")
    return metadata


def _copy_png(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() == ".png":
        shutil.copy2(source, destination)
        return
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("Pillow is required to convert GenEval samples to PNG") from exc
    with Image.open(source) as image:
        image.convert("RGB").save(destination)


def _write_image_grid(sample_paths: list[Path], output_path: Path) -> None:
    if not sample_paths:
        return
    try:
        from PIL import Image
    except Exception:
        return
    images = [Image.open(path).convert("RGB") for path in sample_paths]
    try:
        width = max(image.width for image in images)
        height = max(image.height for image in images)
        grid = Image.new("RGB", (width * len(images), height), "white")
        for index, image in enumerate(images):
            grid.paste(image, (index * width, 0))
        grid.save(output_path)
    finally:
        for image in images:
            image.close()


def export_geneval_artifact(
    artifact_dir: str | Path,
    output_dir: str | Path,
    *,
    split: str,
) -> ExternalExportResult:
    artifact_path = Path(artifact_dir)
    output_path = Path(output_dir)
    records = _read_jsonl(artifact_path / "benchmark" / f"{split}.metrics.jsonl")
    groups: dict[str, dict[str, Any]] = {}
    for record in records:
        source_path = _record_output_path(artifact_path, record)
        if source_path is None or source_path.suffix.lower() not in _IMAGE_SUFFIXES:
            continue
        if not source_path.is_file():
            raise FileNotFoundError(f"generated image missing: {source_path}")
        metadata = _geneval_metadata(record)
        key = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        group = groups.setdefault(key, {"metadata": metadata, "sources": []})
        group["sources"].append(source_path)

    if not groups:
        raise ValueError(
            f"no GenEval-compatible image records found in {artifact_path} split {split!r}"
        )

    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)
    exported_prompts = []
    sample_count = 0
    for prompt_index, group in enumerate(groups.values()):
        prompt_dir = output_path / f"{prompt_index:05d}"
        samples_dir = prompt_dir / "samples"
        samples_dir.mkdir(parents=True)
        metadata = group["metadata"]
        (prompt_dir / "metadata.jsonl").write_text(
            json.dumps(metadata, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        sample_paths = []
        for sample_index, source_path in enumerate(group["sources"]):
            sample_path = samples_dir / f"{sample_index:05d}.png"
            _copy_png(source_path, sample_path)
            sample_paths.append(sample_path)
            sample_count += 1
        _write_image_grid(sample_paths, prompt_dir / "grid.png")
        exported_prompts.append(
            {
                "index": prompt_index,
                "metadata": metadata,
                "sample_count": len(sample_paths),
            }
        )

    manifest_path = output_path / "orbitquant_geneval_export.json"
    manifest_path.write_text(
        json.dumps(
            {
                "artifact_dir": str(artifact_path),
                "split": split,
                "format": "geneval-image-folder",
                "prompt_count": len(exported_prompts),
                "sample_count": sample_count,
                "prompts": exported_prompts,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return ExternalExportResult(
        artifact_dir=str(artifact_path),
        split=split,
        output_dir=str(output_path),
        sample_count=sample_count,
        prompt_count=len(exported_prompts),
        manifest_path=str(manifest_path),
    )


def _video_source_path(artifact_path: Path, record: dict[str, Any]) -> Path | None:
    output_path = _record_output_path(artifact_path, record)
    if output_path is not None and output_path.suffix.lower() in _VIDEO_SUFFIXES:
        return output_path
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        return None
    for value in metadata.get("asset_paths", []):
        path = _artifact_path(artifact_path, value)
        if path.suffix.lower() in _VIDEO_SUFFIXES:
            return path
    return None


def _link_or_copy(source: Path, destination: Path, *, mode: str) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if mode == "copy":
        shutil.copy2(source, destination)
        return "copy"
    if mode == "hardlink":
        try:
            os.link(source, destination)
            return "hardlink"
        except OSError:
            shutil.copy2(source, destination)
            return "copy"
    if mode == "symlink":
        try:
            destination.symlink_to(source)
            return "symlink"
        except OSError:
            shutil.copy2(source, destination)
            return "copy"
    raise ValueError("mode must be one of: symlink, hardlink, copy")


def export_vbench_artifact(
    artifact_dir: str | Path,
    output_dir: str | Path,
    *,
    split: str,
    link_mode: str = "symlink",
) -> ExternalExportResult:
    artifact_path = Path(artifact_dir)
    output_path = Path(output_dir)
    records = _read_jsonl(artifact_path / "benchmark" / f"{split}.metrics.jsonl")
    exports = []
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)
    for index, record in enumerate(records):
        source_path = _video_source_path(artifact_path, record)
        if source_path is None:
            continue
        if not source_path.is_file():
            raise FileNotFoundError(f"generated video missing: {source_path}")
        prompt_record = _prompt_record(record)
        prompt_id = str(prompt_record.get("id") or f"sample-{index:05d}")
        record_metadata = record.get("metadata")
        seed = record_metadata.get("seed", index) if isinstance(record_metadata, dict) else index
        destination = output_path / f"{index:05d}_{prompt_id}_seed{seed}.mp4"
        materialization = _link_or_copy(source_path, destination, mode=link_mode)
        exports.append(
            {
                "source": str(source_path),
                "path": str(destination),
                "materialization": materialization,
                "prompt": _prompt_text(record),
                "prompt_record": prompt_record,
            }
        )

    if not exports:
        raise ValueError(f"no video records found in {artifact_path} split {split!r}")

    prompt_file_path = output_path / "vbench_prompts.json"
    prompt_file_path.write_text(
        json.dumps({item["path"]: item.get("prompt") for item in exports}, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest_path = output_path / "orbitquant_vbench_export.json"
    manifest_path.write_text(
        json.dumps(
            {
                "artifact_dir": str(artifact_path),
                "split": split,
                "format": "vbench-custom-input-folder",
                "sample_count": len(exports),
                "prompt_count": len(exports),
                "prompt_file": str(prompt_file_path),
                "videos": exports,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return ExternalExportResult(
        artifact_dir=str(artifact_path),
        split=split,
        output_dir=str(output_path),
        sample_count=len(exports),
        prompt_count=len(exports),
        manifest_path=str(manifest_path),
    )
