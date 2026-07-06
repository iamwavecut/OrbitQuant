from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _as_rgb_image(value: str | Path | Image.Image | np.ndarray) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, np.ndarray):
        return Image.fromarray(value).convert("RGB")
    return Image.open(value).convert("RGB")


def create_image_comparison_sheet(
    original_path: str | Path,
    orbitquant_path: str | Path,
    output_path: str | Path,
    *,
    labels: tuple[str, str] | None = ("BF16", "OrbitQuant"),
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    original = _as_rgb_image(original_path)
    orbitquant = _as_rgb_image(orbitquant_path)
    if orbitquant.size != original.size:
        orbitquant = orbitquant.resize(original.size, Image.Resampling.LANCZOS)

    label_height = 24 if labels is not None else 0
    sheet = Image.new(
        "RGB",
        (original.width + orbitquant.width, original.height + label_height),
        "white",
    )
    if labels is not None:
        draw = ImageDraw.Draw(sheet)
        draw.text((4, 5), labels[0], fill="black")
        draw.text((original.width + 4, 5), labels[1], fill="black")
    sheet.paste(original, (0, label_height))
    sheet.paste(orbitquant, (original.width, label_height))
    sheet.save(output)
    return output


def create_video_contact_sheet(
    frames: Sequence[Image.Image | np.ndarray],
    output_path: str | Path,
    *,
    sample_indices: Sequence[int] | None = None,
    columns: int = 4,
) -> Path:
    if columns <= 0:
        raise ValueError("columns must be positive")
    if len(frames) == 0:
        raise ValueError("frames must not be empty")
    indices = list(sample_indices) if sample_indices is not None else list(range(len(frames)))
    if not indices:
        raise ValueError("sample_indices must not be empty")

    selected = []
    for index in indices:
        if index < 0 or index >= len(frames):
            raise ValueError(f"frame index out of range: {index}")
        selected.append(_as_rgb_image(frames[index]))

    tile_width, tile_height = selected[0].size
    rows = math.ceil(len(selected) / columns)
    sheet = Image.new("RGB", (tile_width * columns, tile_height * rows), "black")
    for position, frame in enumerate(selected):
        if frame.size != (tile_width, tile_height):
            frame = frame.resize((tile_width, tile_height), Image.Resampling.LANCZOS)
        x = (position % columns) * tile_width
        y = (position // columns) * tile_height
        sheet.paste(frame, (x, y))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    return output
