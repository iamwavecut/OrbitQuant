from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

IMAGE_VISUAL_PROMPTS = [
    {
        "id": "simple-object",
        "title": "01 Fine-detail astrolabe",
        "category": "fine_detail",
        "prompt": (
            "A museum-grade macro photograph of a single ornate brass astronomical clock, "
            "covered with interlocking gears, engraved constellations, enamel moon phases, "
            "hair-thin hands, tiny screws, worn gilt edges, and dust caught in the mechanisms; "
            "dramatic Rembrandt lighting, black velvet background, extreme material detail, "
            "large-format photography, every cog mechanically coherent"
        ),
    },
    {
        "id": "two-object-composition",
        "title": "02 Layered character composition",
        "category": "layered_character_composition",
        "prompt": (
            "A lacquered white android and an elderly watchmaker facing each other across a "
            "crowded Art Nouveau workshop, jointly repairing a mechanical hummingbird; the "
            "android is on the left, the watchmaker on the right, the bird centered between "
            "them, hundreds of tools and clock parts in the midground, rain and a tram visible "
            "through the window, coherent mirror reflections, cinematic tungsten and cyan light, "
            "intricate faces and hands, editorial realism"
        ),
    },
    {
        "id": "counting",
        "title": "03 Exact counting and choreography",
        "category": "counting_and_choreography",
        "prompt": (
            "Exactly seven masked dancers performing on seven separate illuminated platforms "
            "inside a vast flooded opera house, no extra dancers; alternating crimson and ivory "
            "costumes from left to right, each dancer in a distinct pose, balconies reflected in "
            "the water, floating candles, volumetric stage haze, baroque theatrical photography, "
            "sharp foreground and readable background architecture"
        ),
    },
    {
        "id": "color-binding",
        "title": "04 Dense color and object binding",
        "category": "dense_color_binding",
        "prompt": (
            "An elaborate surreal fashion tableau with three models: the left model wears a "
            "cobalt-blue coat and holds a yellow glass sphere, the center model wears a saffron "
            "dress and holds a green ceramic pyramid, the right model wears an emerald suit and "
            "holds a red velvet cube; rococo greenhouse, rare orchids, patterned tile floor, "
            "prismatic sunlight, magazine-cover precision, preserve every color-object pairing"
        ),
    },
    {
        "id": "spatial-relationship",
        "title": "05 Nested spatial relationships",
        "category": "nested_spatial_relationships",
        "prompt": (
            "A meticulous cutaway diorama of a vertical city: a glass greenhouse sits directly "
            "above a silver subway car, a red fox stands inside the greenhouse beneath a hanging "
            "moon lamp, a violinist waits below the subway platform, and a yellow airship passes "
            "behind the entire structure; isometric perspective, architectural-section drawing "
            "mixed with photoreal materials, dozens of tiny rooms, stairs and people, clean depth"
        ),
    },
    {
        "id": "long-prompt",
        "title": "06 Cinematic night-market panorama",
        "category": "cinematic_panorama",
        "prompt": (
            "A sweeping cinematic panorama of a rain-soaked floating night market at blue hour: "
            "in the foreground a chef plates translucent dumplings under a red silk canopy; in "
            "the midground children chase paper lanterns across narrow bridges while merchants "
            "unload exotic fruit from wooden boats; in the background a terraced megacity rises "
            "through mist beneath a storm, with hundreds of warm windows, wet reflections, steam, "
            "umbrellas, ropes and signage; deep focus, anamorphic highlights, realistic faces, "
            "coherent perspective, painterly color grading with documentary-level detail"
        ),
    },
    {
        "id": "english-text-rendering",
        "title": "07 Editorial Latin typography",
        "category": "latin_typography",
        "prompt": (
            "A sophisticated Swiss International Style exhibition poster photographed behind "
            "slightly reflective museum glass, with the exact large headline \"ORBIT QUANT\" and "
            "the exact smaller subtitle \"DATA WITHOUT CALIBRATION\"; strict modular grid, red, "
            "black and white screenprint, tiny registration marks, embossed paper fibers, sharp "
            "letterforms, dramatic gallery shadows, no other text"
        ),
    },
    {
        "id": "cyrillic-text-rendering",
        "title": "08 Russian Constructivist typography",
        "category": "russian_typography",
        "prompt": (
            "A richly detailed Russian Constructivist science-fiction poster with the exact "
            "Cyrillic headline \"КВАНТОВАЯ ОРБИТА\", the exact subtitle \"МОСКВА 2049\", and a "
            "small exact stamp \"КВАНТОВАНИЕ\"; diagonal red and black geometry, cream paper, "
            "cosmonaut portrait, orbital diagrams, halftone grain, folded corners, layered ink, "
            "museum archival photograph, all letters crisp and correctly ordered"
        ),
    },
    {
        "id": "style-heavy",
        "title": "09 Japanese typography and mixed style",
        "category": "japanese_typography",
        "prompt": (
            "An elaborate Japanese art magazine cover combining Edo woodblock printing with a "
            "futuristic Tokyo skyline, with the exact vertical title \"量子の軌道\" and the exact "
            "subtitle \"東京の未来\"; giant indigo waves curl around glass towers, red-crowned "
            "cranes cross a gold moon, tiny pedestrians and trains fill the lower streets, visible "
            "washi fibers, layered spot colors, precise Japanese glyphs, balanced editorial layout"
        ),
    },
    {
        "id": "occlusion-reflection",
        "title": "10 Chinese typography, reflection, occlusion",
        "category": "chinese_typography_reflection_occlusion",
        "prompt": (
            "A luxurious Chinese retro-futurist department-store window at night with the exact "
            "gold title \"量子轨道\" and the exact red subtitle \"未来之城\"; a curved chrome "
            "robot "
            "is partly occluded by peonies and blue-and-white porcelain, the calligraphy and neon "
            "street must appear coherently reflected across its body, multiple glass layers, silk "
            "textures, passing bicycles, cinematic rain, fine product-photography detail"
        ),
    },
]

VIDEO_VISUAL_PROMPTS = [
    {
        "id": "simple-motion",
        "category": "simple_motion",
        "prompt": "A small boat moving steadily across calm water, consistent background",
    },
    {
        "id": "subject-consistency",
        "category": "subject_consistency",
        "prompt": (
            "A person in a red coat walking across a snowy field while the camera stays fixed"
        ),
    },
    {
        "id": "camera-movement",
        "category": "camera_movement",
        "prompt": "A slow camera pan across a quiet mountain lake at sunrise",
    },
    {
        "id": "scene-consistency",
        "category": "scene_consistency",
        "prompt": "A red sports car driving through rain at night, reflections on the road",
    },
    {
        "id": "text-rendering",
        "category": "text_rendering",
        "prompt": 'A neon sign with the exact text "ORBIT" while the camera slowly moves forward',
    },
]

GENEVAL_SMOKE_METADATA = [
    {
        "tag": "single_object",
        "include": [{"class": "bicycle", "count": 1}],
        "prompt": "a photo of a bicycle",
    },
    {
        "tag": "two_object",
        "include": [{"class": "chair", "count": 1}, {"class": "backpack", "count": 1}],
        "prompt": "a photo of a chair and a backpack",
    },
    {
        "tag": "counting",
        "include": [{"class": "cup", "count": 2}],
        "exclude": [{"class": "cup", "count": 3}],
        "prompt": "a photo of two cups",
    },
    {
        "tag": "colors",
        "include": [{"class": "bus", "count": 1, "color": "red"}],
        "prompt": "a photo of a red bus",
    },
    {
        "tag": "position",
        "include": [
            {"class": "couch", "count": 1},
            {"class": "cat", "count": 1, "position": ["left of", 0]},
        ],
        "prompt": "a photo of a cat left of a couch",
    },
    {
        "tag": "color_attr",
        "include": [
            {"class": "bottle", "count": 1, "color": "green"},
            {"class": "banana", "count": 1, "color": "yellow"},
        ],
        "prompt": "a photo of a green bottle and a yellow banana",
    },
]

IMAGE_PROMPTS = [item["prompt"] for item in IMAGE_VISUAL_PROMPTS]
VIDEO_PROMPTS = [item["prompt"] for item in VIDEO_VISUAL_PROMPTS]


def default_prompt_payload(target_policy: str) -> dict[str, Any]:
    normalized_policy = target_policy.lower()
    if normalized_policy == "wan":
        return {
            "prompt_pack": "video_visual_v1",
            "media_type": "video",
            "target_policy": target_policy,
            "prompts": deepcopy(VIDEO_VISUAL_PROMPTS),
        }
    return {
        "prompt_pack": "image_visual_v2",
        "media_type": "image",
        "target_policy": target_policy,
        "prompts": deepcopy(IMAGE_VISUAL_PROMPTS),
    }


def _geneval_prompt_record(record: dict[str, Any], index: int) -> dict[str, Any]:
    tag = record.get("tag")
    prompt = record.get("prompt")
    include = record.get("include")
    if not isinstance(tag, str) or not tag:
        raise ValueError(f"GenEval metadata line {index + 1} is missing string tag")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError(f"GenEval metadata line {index + 1} is missing string prompt")
    if not isinstance(include, list) or not include:
        raise ValueError(f"GenEval metadata line {index + 1} is missing include list")
    geneval = {
        "tag": tag,
        "prompt": prompt,
        "include": deepcopy(include),
    }
    if "exclude" in record:
        exclude = record["exclude"]
        if not isinstance(exclude, list):
            raise ValueError(f"GenEval metadata line {index + 1} has non-list exclude")
        geneval["exclude"] = deepcopy(exclude)
    return {
        "id": f"geneval-{index:05d}-{tag.replace('_', '-')}",
        "category": tag,
        "prompt": prompt,
        "geneval": geneval,
    }


def geneval_prompt_payload(
    records: list[dict[str, Any]],
    *,
    target_policy: str,
    prompt_pack: str = "geneval_metadata_jsonl",
) -> dict[str, Any]:
    if not records:
        raise ValueError("GenEval prompt metadata is empty")
    return {
        "prompt_pack": prompt_pack,
        "media_type": "image",
        "target_policy": target_policy,
        "prompts": [
            _geneval_prompt_record(record, index) for index, record in enumerate(records)
        ],
    }


def geneval_smoke_prompt_payload(target_policy: str) -> dict[str, Any]:
    return geneval_prompt_payload(
        deepcopy(GENEVAL_SMOKE_METADATA),
        target_policy=target_policy,
        prompt_pack="geneval_smoke_v1",
    )


def load_geneval_prompt_payload(
    metadata_jsonl: str | Path,
    *,
    target_policy: str,
) -> dict[str, Any]:
    path = Path(metadata_jsonl)
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"GenEval metadata line {line_number} is not valid JSON"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(f"GenEval metadata line {line_number} is not an object")
            records.append(record)
    return geneval_prompt_payload(records, target_policy=target_policy)


def select_prompt_record(
    payload: dict[str, Any],
    *,
    prompt_id: str | None = None,
    prompt_index: int | None = None,
) -> dict[str, Any]:
    if (prompt_id is None) == (prompt_index is None):
        raise ValueError("provide exactly one prompt selector")
    prompts = list(payload.get("prompts", []))
    if prompt_id is not None:
        for record in prompts:
            if record.get("id") == prompt_id:
                return deepcopy(record)
        raise ValueError(f"prompt id not found: {prompt_id}")
    assert prompt_index is not None
    if prompt_index < 0 or prompt_index >= len(prompts):
        raise ValueError(
            f"prompt index out of range: {prompt_index}; available prompts: {len(prompts)}"
        )
    return deepcopy(prompts[prompt_index])


def build_prompt_seed_jobs(
    payload: dict[str, Any],
    *,
    seeds: list[int],
    prompt_ids: list[str] | None = None,
    prompt_limit: int | None = None,
) -> list[dict[str, Any]]:
    if not seeds:
        raise ValueError("at least one seed is required")
    if prompt_limit is not None and prompt_limit <= 0:
        raise ValueError("prompt_limit must be positive")

    prompts = list(payload.get("prompts", []))
    if prompt_ids is not None:
        selected = []
        for prompt_id in prompt_ids:
            selected.append(select_prompt_record(payload, prompt_id=prompt_id))
        prompts = selected
    else:
        prompts = deepcopy(prompts)
    if prompt_limit is not None:
        prompts = prompts[:prompt_limit]
    if not prompts:
        raise ValueError("prompt selection produced no jobs")

    jobs = []
    for seed in seeds:
        for prompt_record in prompts:
            jobs.append({"seed": seed, "prompt_record": deepcopy(prompt_record)})
    return jobs
