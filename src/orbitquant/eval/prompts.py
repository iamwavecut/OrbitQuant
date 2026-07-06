from __future__ import annotations

from copy import deepcopy
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
