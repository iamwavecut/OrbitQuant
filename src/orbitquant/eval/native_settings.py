from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NativeSuite:
    name: str
    model_id: str
    pipeline: str
    width: int
    height: int
    steps: int
    guidance: float
    bit_settings: list[str]
    frames: int | None = None
    metric: str | None = None
    note: str = ""


_SUITES = {
    "flux2-native": NativeSuite(
        name="flux2-native",
        model_id="black-forest-labs/FLUX.2-klein-4B",
        pipeline="Flux2KleinPipeline",
        width=1024,
        height=1024,
        steps=4,
        guidance=1.0,
        bit_settings=["W4A4", "W3A3", "W2A4", "W2A3"],
        metric="visual+optional-geneval",
        note="Extra target; not an OrbitQuant paper reproduction model.",
    ),
    "flux1-schnell-native": NativeSuite(
        name="flux1-schnell-native",
        model_id="black-forest-labs/FLUX.1-schnell",
        pipeline="FluxPipeline",
        width=1024,
        height=1024,
        steps=4,
        guidance=0.0,
        bit_settings=["W4A4", "W3A3", "W2A4", "W2A3"],
        metric="geneval",
    ),
    "z-image-native": NativeSuite(
        name="z-image-native",
        model_id="Tongyi-MAI/Z-Image-Turbo",
        pipeline="ZImagePipeline",
        width=1024,
        height=1024,
        steps=10,
        guidance=0.0,
        bit_settings=["W4A4", "W3A3", "W2A4", "W2A3"],
        metric="geneval",
        note="Paper-aligned 10-step setting; record actual scheduler forward count.",
    ),
    "wan-native": NativeSuite(
        name="wan-native",
        model_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        pipeline="WanPipeline",
        width=832,
        height=480,
        frames=81,
        steps=50,
        guidance=5.0,
        bit_settings=["W4A6", "W4A4"],
        metric="vbench",
    ),
}


def get_native_suite(name: str) -> NativeSuite:
    try:
        return _SUITES[name]
    except KeyError as exc:
        raise KeyError(f"unknown native suite {name!r}") from exc


def list_native_suites() -> list[NativeSuite]:
    return list(_SUITES.values())
