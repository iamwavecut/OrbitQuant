import json

from hub_helpers import _legacy_compact_summary_without_native_smoke, _native_smoke_summary

import orbitquant.hub as hub_module
from orbitquant.eval.native_settings import NativeSuite


def test_native_smoke_expected_settings_include_video_export_fps_only_when_defined():
    image_suite = NativeSuite(
        name="flux2-native",
        model_id="black-forest-labs/FLUX.2-klein-4B",
        pipeline="Flux2KleinPipeline",
        width=1024,
        height=1024,
        steps=4,
        guidance=1.0,
        bit_settings=["W4A4"],
    )
    video_suite = NativeSuite(
        name="wan-native",
        model_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        pipeline="WanPipeline",
        width=832,
        height=480,
        frames=81,
        export_fps=16,
        steps=50,
        guidance=5.0,
        bit_settings=["W4A4"],
    )

    assert "export_fps" not in hub_module._native_smoke_expected_settings(image_suite)
    assert hub_module._native_smoke_expected_settings(video_suite)["export_fps"] == 16


def test_recover_native_smoke_proof_from_compact_summary_requires_raw_pair_evidence():
    suite = NativeSuite(
        name="flux2-native",
        model_id="black-forest-labs/FLUX.2-klein-4B",
        pipeline="Flux2KleinPipeline",
        width=1024,
        height=1024,
        steps=4,
        guidance=1.0,
        bit_settings=["W4A4"],
    )
    summary = json.loads(_legacy_compact_summary_without_native_smoke(suite))
    file_names = {"assets/image_generation_comparison_matrix.webp"}

    proof, reason = hub_module._recover_native_smoke_proof_from_compact_summary(
        summary,
        suite=suite,
        file_names=file_names,
    )

    assert proof is None
    assert reason == "raw_paired_native_smoke_evidence_missing"


def test_native_smoke_proof_status_rejects_recovered_pair_claims():
    suite = NativeSuite(
        name="flux2-native",
        model_id="black-forest-labs/FLUX.2-klein-4B",
        pipeline="Flux2KleinPipeline",
        width=1024,
        height=1024,
        steps=4,
        guidance=1.0,
        bit_settings=["W4A4"],
    )
    summary = json.loads(_native_smoke_summary(suite))
    summary["native_smoke"]["proof_source"] = (
        "recovered_from_compact_summary_and_published_comparison_matrix"
    )

    status = hub_module._native_smoke_proof_status(
        summary,
        suite=suite,
        file_names={"assets/image_generation_comparison_matrix.webp"},
    )

    assert status["ready"] is False
    assert "native_smoke.raw_paired_native_smoke_evidence" in status["missing"]


def test_recover_native_smoke_proof_requires_uploaded_comparison_asset():
    suite = NativeSuite(
        name="flux2-native",
        model_id="black-forest-labs/FLUX.2-klein-4B",
        pipeline="Flux2KleinPipeline",
        width=1024,
        height=1024,
        steps=4,
        guidance=1.0,
        bit_settings=["W4A4"],
    )
    summary = json.loads(_legacy_compact_summary_without_native_smoke(suite))
    summary.pop("raw_generation_records")

    proof, reason = hub_module._recover_native_smoke_proof_from_compact_summary(
        summary,
        suite=suite,
        file_names=set(),
    )

    assert proof is None
    assert reason == "comparison_asset_missing"


def test_recover_native_smoke_proof_rejects_missing_generated_samples():
    suite = NativeSuite(
        name="flux2-native",
        model_id="black-forest-labs/FLUX.2-klein-4B",
        pipeline="Flux2KleinPipeline",
        width=1024,
        height=1024,
        steps=4,
        guidance=1.0,
        bit_settings=["W4A4"],
    )
    summary = json.loads(_legacy_compact_summary_without_native_smoke(suite, generated_samples=0))
    summary.pop("raw_generation_records")

    proof, reason = hub_module._recover_native_smoke_proof_from_compact_summary(
        summary,
        suite=suite,
        file_names={"assets/image_generation_comparison_matrix.webp"},
    )

    assert proof is None
    assert reason == "original.generated_samples_missing"


def test_recover_native_smoke_proof_rejects_video_without_generated_frames():
    suite = NativeSuite(
        name="wan-native",
        model_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        pipeline="WanPipeline",
        width=832,
        height=480,
        frames=81,
        steps=50,
        guidance=5.0,
        bit_settings=["W4A4"],
    )
    summary = json.loads(_legacy_compact_summary_without_native_smoke(suite))
    summary.pop("raw_generation_records")

    proof, reason = hub_module._recover_native_smoke_proof_from_compact_summary(
        summary,
        suite=suite,
        file_names={"assets/video_generation_comparison_matrix.webp"},
    )

    assert proof is None
    assert reason == "original.generated_frames_insufficient"
