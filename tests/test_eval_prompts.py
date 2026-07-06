from orbitquant.eval.prompts import default_prompt_payload


def test_default_prompt_payload_uses_image_visual_pack_for_image_policies():
    payload = default_prompt_payload("flux2")

    assert payload["prompt_pack"] == "image_visual_v1"
    assert payload["media_type"] == "image"
    assert len(payload["prompts"]) >= 10
    assert {item["id"] for item in payload["prompts"]} >= {
        "simple-object",
        "two-object-composition",
        "counting",
        "color-binding",
        "spatial-relationship",
        "long-prompt",
        "english-text-rendering",
        "cyrillic-text-rendering",
        "style-heavy",
        "occlusion-reflection",
    }


def test_default_prompt_payload_uses_video_visual_pack_for_wan_policy():
    payload = default_prompt_payload("wan")

    assert payload["prompt_pack"] == "video_visual_v1"
    assert payload["media_type"] == "video"
    assert len(payload["prompts"]) >= 5
    assert {item["id"] for item in payload["prompts"]} >= {
        "simple-motion",
        "subject-consistency",
        "camera-movement",
        "scene-consistency",
        "text-rendering",
    }
