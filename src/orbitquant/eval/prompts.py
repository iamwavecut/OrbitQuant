from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

IMAGE_VISUAL_PROMPTS = [
    {
        "id": "simple-object",
        "category": "simple_object",
        "prompt": "A red ceramic mug on a wooden desk, soft daylight, shallow depth of field",
    },
    {
        "id": "two-object-composition",
        "category": "two_object_composition",
        "prompt": "Two robots playing chess in a quiet library, one silver and one matte black",
    },
    {
        "id": "counting",
        "category": "counting",
        "prompt": "Five glass marbles arranged in a straight line on white paper",
    },
    {
        "id": "color-binding",
        "category": "color_binding",
        "prompt": "A blue cube beside a yellow sphere and a green pyramid on a gray table",
    },
    {
        "id": "spatial-relationship",
        "category": "spatial_relationship",
        "prompt": "A blue cube to the left of a yellow sphere, both centered in the frame",
    },
    {
        "id": "long-prompt",
        "category": "long_prompt",
        "prompt": (
            "A precise editorial photo of a compact robotics workbench with labeled tools, "
            "a small oscilloscope, coiled cables, a half-assembled drone, and a warm desk lamp; "
            "the scene should remain readable without clutter"
        ),
    },
    {
        "id": "english-text-rendering",
        "category": "text_rendering",
        "prompt": 'A clean street sign with the exact text "ORBIT QUANT"',
    },
    {
        "id": "cyrillic-text-rendering",
        "category": "text_rendering",
        "prompt": 'A handwritten note in Cyrillic that says "КВАНТОВАНИЕ"',
    },
    {
        "id": "style-heavy",
        "category": "style",
        "prompt": (
            "A crowded train platform at sunset, documentary photo style, high dynamic range, "
            "natural grain, realistic faces, detailed architecture"
        ),
    },
    {
        "id": "occlusion-reflection",
        "category": "occlusion_reflection",
        "prompt": (
            "A reflective chrome teapot partly hidden behind orange flowers, with the room "
            "visible as a coherent reflection on the metal"
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
        "prompt_pack": "image_visual_v1",
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
