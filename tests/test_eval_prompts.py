import json

from orbitquant.eval.prompts import (
    default_prompt_payload,
    geneval_smoke_prompt_payload,
    load_geneval_prompt_payload,
)


def test_default_prompt_payload_uses_image_visual_pack_for_image_policies():
    payload = default_prompt_payload("flux2")

    assert payload["prompt_pack"] == "image_visual_v2"
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
    assert all(item.get("title") for item in payload["prompts"])
    assert min(len(item["prompt"]) for item in payload["prompts"]) >= 350
    prompts = "\n".join(item["prompt"] for item in payload["prompts"])
    assert "КВАНТОВАЯ ОРБИТА" in prompts
    assert "量子の軌道" in prompts
    assert "量子轨道" in prompts


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


def test_geneval_smoke_prompt_payload_contains_upstream_metadata_shape():
    payload = geneval_smoke_prompt_payload("flux")

    assert payload["prompt_pack"] == "geneval_smoke_v1"
    assert payload["media_type"] == "image"
    assert {item["category"] for item in payload["prompts"]} == {
        "single_object",
        "two_object",
        "counting",
        "colors",
        "position",
        "color_attr",
    }
    first = payload["prompts"][0]
    assert first["id"].startswith("geneval-00000-")
    assert first["geneval"]["tag"] == first["category"]
    assert first["geneval"]["prompt"] == first["prompt"]
    assert isinstance(first["geneval"]["include"], list)


def test_load_geneval_prompt_payload_reads_evaluation_metadata_jsonl(tmp_path):
    metadata_jsonl = tmp_path / "evaluation_metadata.jsonl"
    metadata_jsonl.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "tag": "single_object",
                        "include": [{"class": "bench", "count": 1}],
                        "prompt": "a photo of a bench",
                    }
                ),
                json.dumps(
                    {
                        "tag": "counting",
                        "include": [{"class": "clock", "count": 2}],
                        "exclude": [{"class": "clock", "count": 3}],
                        "prompt": "a photo of two clocks",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    payload = load_geneval_prompt_payload(metadata_jsonl, target_policy="z_image")

    assert payload["prompt_pack"] == "geneval_metadata_jsonl"
    assert payload["target_policy"] == "z_image"
    assert [item["id"] for item in payload["prompts"]] == [
        "geneval-00000-single-object",
        "geneval-00001-counting",
    ]
    assert payload["prompts"][1]["geneval"]["exclude"] == [
        {"class": "clock", "count": 3}
    ]
