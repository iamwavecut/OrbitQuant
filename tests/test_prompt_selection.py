import pytest

from orbitquant.eval.prompts import default_prompt_payload, select_prompt_record


def test_select_prompt_record_uses_prompt_id():
    payload = default_prompt_payload("flux2")

    record = select_prompt_record(payload, prompt_id="english-text-rendering")

    assert record["id"] == "english-text-rendering"
    assert "ORBIT QUANT" in record["prompt"]


def test_select_prompt_record_uses_zero_based_prompt_index():
    payload = default_prompt_payload("wan")

    record = select_prompt_record(payload, prompt_index=0)

    assert record["id"] == "simple-motion"


def test_select_prompt_record_rejects_ambiguous_selector():
    payload = default_prompt_payload("flux2")

    with pytest.raises(ValueError, match="one prompt selector"):
        select_prompt_record(payload, prompt_id="simple-object", prompt_index=0)
