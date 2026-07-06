from __future__ import annotations

from dataclasses import asdict
from typing import Any

from huggingface_hub import HfApi

from orbitquant.eval import list_native_suites


def inspect_model_metadata(model_id: str, *, revision: str | None = None) -> dict[str, Any]:
    info = HfApi().model_info(model_id, revision=revision)
    card_data = getattr(info, "card_data", None)
    license_name = None
    if card_data is not None:
        if isinstance(card_data, dict):
            license_name = card_data.get("license")
        else:
            license_name = getattr(card_data, "license", None)

    native_suite = None
    for suite in list_native_suites():
        if suite.model_id == model_id:
            native_suite = asdict(suite)
            break

    return {
        "model_id": model_id,
        "sha": info.sha,
        "private": info.private,
        "gated": info.gated,
        "license": license_name,
        "tags": sorted(info.tags or []),
        "native_suite": native_suite,
    }
