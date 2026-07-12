import json
from pathlib import Path

import pytest
from hub_helpers import FakeHfApi, _write_artifact

import orbitquant.hub as hub_module
from orbitquant.artifacts import record_artifact_metrics, refresh_artifact_checksums
from orbitquant.artifacts.checksums import read_sha256sums
from orbitquant.hub import stage_compact_upload_artifact, upload_orbitquant_artifact


def test_upload_orbitquant_artifact_dry_run_validates_without_hub_calls(tmp_path):
    _write_artifact(tmp_path)
    fake_api = FakeHfApi()

    result = upload_orbitquant_artifact(
        tmp_path,
        repo_id="WaveCut/example-orbitquant",
        dry_run=True,
        api=fake_api,
    )

    assert result["dry_run"] is True
    assert result["repo_id"] == "WaveCut/example-orbitquant"
    assert result["private"] is True
    assert result["validation"]["valid"] is True
    assert result["upload"] is None
    assert fake_api.create_repo_calls == []
    assert fake_api.upload_folder_calls == []
    assert fake_api.model_info_calls == []


def test_stage_compact_upload_artifact_omits_raw_eval_reports_and_promotes_matrices(
    tmp_path,
):
    _write_artifact(tmp_path)
    raw_eval_asset = tmp_path / "assets" / "flux2-native_seed0_W4A4_geneval-00000.png"
    raw_eval_metadata = raw_eval_asset.with_suffix(".png.json")
    visual_asset = tmp_path / "assets" / "flux2-native_seed0_W4A4_simple-object.png"
    comparison_asset = (
        tmp_path / "assets" / "original_vs_orbitquant_flux2-native_seed0_W4A4_simple-object.webp"
    )
    raw_video_asset = tmp_path / "assets" / "wan-native_seed0_W4A4_simple-motion.mp4"
    raw_video_metadata = raw_video_asset.with_suffix(".mp4.json")
    raw_eval_asset.write_bytes(b"raw eval image")
    raw_eval_metadata.write_text('{"prompt_id":"geneval-00000"}\n', encoding="utf-8")
    visual_asset.write_bytes(b"visual image")
    comparison_asset.write_bytes(b"comparison")
    raw_video_asset.write_bytes(b"raw video")
    raw_video_metadata.write_text('{"prompt_id":"simple-motion"}\n', encoding="utf-8")
    report_dir = tmp_path.parent / f"{tmp_path.name}-report"
    report_matrix = report_dir / "assets" / "image_generation_comparison_matrix.webp"
    report_matrix.parent.mkdir(parents=True)
    report_matrix.write_bytes(b"report matrix")
    report_markdown = report_dir / "orbitquant-native-eval.md"
    report_markdown.write_text("# local report log\n", encoding="utf-8")
    report_table = report_dir / "tables" / "perf.csv"
    report_table.parent.mkdir(parents=True)
    report_table.write_text("metric,value\n", encoding="utf-8")
    in_artifact_report = (
        tmp_path
        / "reports"
        / "native"
        / "flux2-w4a4"
        / "assets"
        / "image_generation_comparison_matrix.webp"
    )
    in_artifact_report.parent.mkdir(parents=True)
    in_artifact_report.write_bytes(b"in artifact report matrix")
    in_artifact_markdown = in_artifact_report.parent.parent / "orbitquant-native-eval.md"
    in_artifact_markdown.write_text("# local artifact report log\n", encoding="utf-8")
    refresh_artifact_checksums(tmp_path)

    stage_dir = tmp_path.parent / f"{tmp_path.name}-stage"
    result = stage_compact_upload_artifact(
        tmp_path,
        stage_dir,
        report_dirs=[report_dir],
        validate_tensors=False,
    )

    staged_checksums = read_sha256sums(stage_dir / "SHA256SUMS")
    assert result["validation"]["valid"] is True
    assert result["omitted_raw_eval_asset_count"] == 6
    assert result["omitted_report_file_count"] == 2
    assert result["copied_report_asset_count"] == 1
    assert not (stage_dir / raw_eval_asset.relative_to(tmp_path)).exists()
    assert not (stage_dir / raw_eval_metadata.relative_to(tmp_path)).exists()
    assert not (stage_dir / visual_asset.relative_to(tmp_path)).exists()
    assert not (stage_dir / comparison_asset.relative_to(tmp_path)).exists()
    assert not (stage_dir / raw_video_asset.relative_to(tmp_path)).exists()
    assert not (stage_dir / raw_video_metadata.relative_to(tmp_path)).exists()
    assert not (stage_dir / in_artifact_report.relative_to(tmp_path)).exists()
    assert not (stage_dir / in_artifact_markdown.relative_to(tmp_path)).exists()
    staged_report = stage_dir / "assets" / "image_generation_comparison_matrix.webp"
    staged_collision_report = (
        stage_dir / "assets" / f"{report_dir.name}_image_generation_comparison_matrix.webp"
    )
    assert staged_report.is_file()
    assert staged_report.read_bytes() == b"in artifact report matrix"
    assert not staged_collision_report.exists()
    assert raw_eval_asset.relative_to(tmp_path).as_posix() not in staged_checksums
    assert visual_asset.relative_to(tmp_path).as_posix() not in staged_checksums
    assert comparison_asset.relative_to(tmp_path).as_posix() not in staged_checksums
    assert raw_video_asset.relative_to(tmp_path).as_posix() not in staged_checksums
    assert not any(path.startswith("reports/") for path in staged_checksums)
    assert staged_report.relative_to(stage_dir).as_posix() in staged_checksums


def test_stage_compact_upload_artifact_keeps_one_direct_comparison_matrix(tmp_path):
    _write_artifact(tmp_path)
    image_matrix = tmp_path / "assets" / "image_generation_comparison_matrix.webp"
    video_matrix = tmp_path / "assets" / "video_generation_comparison_matrix.webp"
    image_matrix.write_bytes(b"image matrix")
    video_matrix.write_bytes(b"video matrix")
    report_matrix = (
        tmp_path
        / "reports"
        / "native"
        / "flux2-w4a4"
        / "assets"
        / "image_generation_comparison_matrix.webp"
    )
    report_matrix.parent.mkdir(parents=True)
    report_matrix.write_bytes(b"report matrix")
    refresh_artifact_checksums(tmp_path)

    stage_dir = tmp_path.parent / f"{tmp_path.name}-stage"
    result = stage_compact_upload_artifact(
        tmp_path,
        stage_dir,
        validate_tensors=False,
    )

    staged_checksums = read_sha256sums(stage_dir / "SHA256SUMS")
    assert result["copied_report_asset_count"] == 0
    assert (stage_dir / "assets" / "image_generation_comparison_matrix.webp").read_bytes() == (
        b"image matrix"
    )
    assert not (stage_dir / "assets" / "video_generation_comparison_matrix.webp").exists()
    assert "assets/image_generation_comparison_matrix.webp" in staged_checksums
    assert "assets/video_generation_comparison_matrix.webp" not in staged_checksums


def test_stage_compact_upload_artifact_enforces_asset_allowlist(tmp_path):
    _write_artifact(tmp_path)
    allowed_matrix = tmp_path / "assets" / "image_generation_comparison_matrix.webp"
    extra_matrix = tmp_path / "assets" / "video_generation_comparison_matrix.webp"
    raw_png = tmp_path / "assets" / "flux2-native_seed0_W4A4_simple-object.png"
    raw_png_sidecar = raw_png.with_suffix(".png.json")
    raw_webp = tmp_path / "assets" / "original_vs_orbitquant_seed0.webp"
    contact_sheet = tmp_path / "assets" / "wan_contact_sheet.webp"
    raw_video = tmp_path / "assets" / "wan-native_seed0_W4A4_motion.mp4"
    raw_video_sidecar = raw_video.with_suffix(".mp4.json")
    nested_raw = tmp_path / "assets" / "nested" / "debug_frame.png"
    nested_matrix = tmp_path / "assets" / "nested" / "debug_generation_comparison_matrix.webp"
    unexpected_file = tmp_path / "debug.txt"
    ignored_cache = tmp_path / ".cache" / "upload.tmp"
    for path, payload in (
        (allowed_matrix, b"matrix"),
        (extra_matrix, b"video matrix"),
        (raw_png, b"png"),
        (raw_png_sidecar, b"json"),
        (raw_webp, b"webp"),
        (contact_sheet, b"contact"),
        (raw_video, b"mp4"),
        (raw_video_sidecar, b"mp4 json"),
        (nested_raw, b"nested"),
        (nested_matrix, b"nested matrix"),
        (unexpected_file, b"debug"),
        (ignored_cache, b"ignored"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    refresh_artifact_checksums(tmp_path)

    stage_dir = tmp_path.parent / f"{tmp_path.name}-asset-allowlist-stage"
    result = stage_compact_upload_artifact(tmp_path, stage_dir, validate_tensors=False)

    staged_files = {
        path.relative_to(stage_dir).as_posix() for path in stage_dir.rglob("*") if path.is_file()
    }
    expected_files = set(hub_module._REQUIRED_ARTIFACT_FILES) | {
        "assets/image_generation_comparison_matrix.webp"
    }
    assert staged_files == expected_files

    staged_manifest = json.loads((stage_dir / "orbitquant_manifest.json").read_text())
    manifest_checksum_files = set(staged_manifest["checksums"])
    sha256sum_files = set(read_sha256sums(stage_dir / "SHA256SUMS"))
    assert manifest_checksum_files == staged_files - {
        "README.md",
        "SHA256SUMS",
        "orbitquant_manifest.json",
    }
    assert sha256sum_files == staged_files - {"SHA256SUMS"}

    staged_summary = json.loads((stage_dir / "benchmark" / "summary.json").read_text())
    assert staged_summary["published_summary"] == "compact"
    assert staged_summary["raw_generation_records"] == "local-only"
    assert not any(path.endswith(".png") for path in staged_files)
    assert not any(path.endswith(".png.json") for path in staged_files)
    assert not any(path.endswith(".mp4") for path in staged_files)
    assert "assets/video_generation_comparison_matrix.webp" not in staged_files
    assert "assets/original_vs_orbitquant_seed0.webp" not in staged_files
    assert "assets/wan_contact_sheet.webp" not in staged_files
    assert "assets/nested/debug_frame.png" not in staged_files
    assert "assets/nested/debug_generation_comparison_matrix.webp" not in staged_files
    assert "debug.txt" not in staged_files
    assert ".cache/upload.tmp" not in staged_files
    assert result["omitted_unexpected_file_count"] == 1
    assert result["omitted_unexpected_files"] == ["debug.txt"]
    assert set(result["omitted_raw_eval_assets"]) >= {
        "assets/video_generation_comparison_matrix.webp",
        "assets/flux2-native_seed0_W4A4_simple-object.png",
        "assets/flux2-native_seed0_W4A4_simple-object.png.json",
        "assets/original_vs_orbitquant_seed0.webp",
        "assets/wan_contact_sheet.webp",
        "assets/wan-native_seed0_W4A4_motion.mp4",
        "assets/wan-native_seed0_W4A4_motion.mp4.json",
        "assets/nested/debug_frame.png",
        "assets/nested/debug_generation_comparison_matrix.webp",
    }


def test_stage_compact_upload_artifact_writes_native_smoke_proof(tmp_path):
    _write_artifact(tmp_path)
    matrix = tmp_path / "assets" / "image_generation_comparison_matrix.webp"
    original_output = tmp_path / "assets" / "flux2-native_seed0_original_simple-object.png"
    orbitquant_output = tmp_path / "assets" / "flux2-native_seed0_W4A4_simple-object.png"
    for path, payload in (
        (matrix, b"matrix"),
        (original_output, b"original image"),
        (orbitquant_output, b"orbitquant image"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    for split, output, bit_setting in (
        ("original", original_output, "original"),
        ("orbitquant", orbitquant_output, "W4A4"),
    ):
        record_artifact_metrics(
            tmp_path,
            split=split,
            metrics={"generated_samples": 1},
            metadata={
                "suite": "flux2-native",
                "prompt": "A native prompt",
                "prompt_record": {"id": "simple-object"},
                "seed": 0,
                "height": 1024,
                "width": 1024,
                "frames": None,
                "steps": 4,
                "guidance": 1.0,
                "bit_setting": bit_setting,
                "output_path": str(output),
            },
            validate_checksums_enabled=False,
            refresh_checksums_enabled=False,
        )
    refresh_artifact_checksums(tmp_path)

    stage_dir = tmp_path.parent / f"{tmp_path.name}-native-smoke-stage"
    stage_compact_upload_artifact(tmp_path, stage_dir, validate_tensors=False)

    staged_summary = json.loads((stage_dir / "benchmark" / "summary.json").read_text())
    staged_readme = (stage_dir / "README.md").read_text(encoding="utf-8")
    assert staged_summary["raw_generation_records"] == "local-only"
    assert "latest" not in staged_summary["metrics"]["original"]
    assert "## Native Validation Evidence" in staged_readme
    assert "| Comparison matrix | `assets/image_generation_comparison_matrix.webp` |" in (
        staged_readme
    )
    assert "| Paired prompt/seed count | `1` |" in staged_readme
    assert "| BF16 source nonempty outputs | `1` |" in staged_readme
    assert "| OrbitQuant nonempty outputs | `1` |" in staged_readme
    assert "original.metrics.jsonl" not in staged_readme
    assert staged_summary["native_smoke"] == {
        "proof_format": "orbitquant-native-smoke-v1",
        "comparison_asset_path": "assets/image_generation_comparison_matrix.webp",
        "paired_prompt_seed_count": 1,
        "paired_prompt_seed_keys": [["flux2-native", "0", "simple-object"]],
        "splits": {
            "original": {
                "records": 1,
                "generated_samples": 1,
                "generated_frames": 0,
                "nonempty_output_count": 1,
                "seeds": ["0"],
                "prompt_ids": ["simple-object"],
                "pair_keys": [["flux2-native", "0", "simple-object"]],
                "native_settings": [
                    {
                        "suite": "flux2-native",
                        "height": 1024,
                        "width": 1024,
                        "frames": None,
                        "steps": 4,
                        "guidance": 1.0,
                    }
                ],
            },
            "orbitquant": {
                "records": 1,
                "generated_samples": 1,
                "generated_frames": 0,
                "nonempty_output_count": 1,
                "seeds": ["0"],
                "prompt_ids": ["simple-object"],
                "pair_keys": [["flux2-native", "0", "simple-object"]],
                "native_settings": [
                    {
                        "suite": "flux2-native",
                        "height": 1024,
                        "width": 1024,
                        "frames": None,
                        "steps": 4,
                        "guidance": 1.0,
                    }
                ],
            },
        },
    }


def test_stage_compact_upload_artifact_promotes_compare_native_bundle(tmp_path):
    _write_artifact(tmp_path)
    bundle_dir = tmp_path.parent / f"{tmp_path.name}-compare-native"
    bundle_dir.mkdir()
    comparison = bundle_dir / "flux2-native_seed0_W4A4_original_vs_orbitquant.webp"
    original = bundle_dir / "flux2-native_seed0_original.png"
    orbitquant = bundle_dir / "flux2-native_seed0_W4A4.png"
    for path, payload in (
        (comparison, b"comparison"),
        (original, b"original"),
        (orbitquant, b"orbitquant"),
    ):
        path.write_bytes(payload)
    metadata = {
        "suite": "flux2-native",
        "prompt": "A native prompt",
        "prompt_record": {"id": "simple-object"},
        "seed": 0,
        "height": 1024,
        "width": 1024,
        "frames": None,
        "steps": 4,
        "guidance": 1.0,
    }
    for output in (original, orbitquant):
        output.with_suffix(output.suffix + ".json").write_text(
            json.dumps(metadata) + "\n", encoding="utf-8"
        )
    (bundle_dir / "summary.json").write_text(
        json.dumps(
            {
                **metadata,
                "model_id": "example/model",
                "bit_setting": "W4A4",
                "comparison_path": "/workspace/run/" + comparison.name,
                "original": {
                    "output_path": "/workspace/run/" + original.name,
                    "metadata_path": "/workspace/run/" + original.name + ".json",
                },
                "orbitquant": {
                    "output_path": "/workspace/run/" + orbitquant.name,
                    "metadata_path": "/workspace/run/" + orbitquant.name + ".json",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    stage_dir = tmp_path.parent / f"{tmp_path.name}-compare-native-stage"
    result = stage_compact_upload_artifact(
        tmp_path,
        stage_dir,
        report_dirs=[bundle_dir],
        validate_tensors=False,
    )

    matrix_path = stage_dir / "assets" / "image_generation_comparison_matrix.webp"
    summary = json.loads((stage_dir / "benchmark" / "summary.json").read_text())
    proof = summary["native_smoke"]
    assert result["copied_report_assets"] == ["assets/image_generation_comparison_matrix.webp"]
    assert matrix_path.read_bytes() == b"comparison"
    assert proof["proof_source"] == "local_compare_native_bundle"
    assert proof["comparison_asset_path"] == ("assets/image_generation_comparison_matrix.webp")
    assert proof["paired_prompt_seed_keys"] == [["flux2-native", "0", "simple-object"]]
    assert proof["splits"]["original"]["nonempty_output_count"] == 1
    assert proof["splits"]["orbitquant"]["nonempty_output_count"] == 1
    assert not (stage_dir / original.name).exists()
    assert not (stage_dir / orbitquant.name).exists()


def test_upload_orbitquant_artifact_rejects_raw_full_upload_profile(tmp_path):
    _write_artifact(tmp_path)
    fake_api = FakeHfApi()

    with pytest.raises(ValueError, match="unsupported upload profile: full"):
        upload_orbitquant_artifact(
            tmp_path,
            repo_id="WaveCut/example-orbitquant",
            private=False,
            revision="main",
            commit_message="upload test artifact",
            replace_repo_files=True,
            validate_tensors=False,
            upload_profile="full",
            api=fake_api,
        )

    assert fake_api.create_repo_calls == []
    assert fake_api.upload_folder_calls == []


def test_upload_orbitquant_artifact_can_upload_compact_staged_copy(tmp_path):
    _write_artifact(tmp_path)
    raw_eval_asset = tmp_path / "assets" / "flux2-native_seed0_W4A4_geneval-00000.png"
    visual_asset = tmp_path / "assets" / "flux2-native_seed0_W4A4_simple-object.png"
    raw_eval_asset.write_bytes(b"raw eval image")
    visual_asset.write_bytes(b"visual image")
    refresh_artifact_checksums(tmp_path)
    stage_dir = tmp_path.parent / f"{tmp_path.name}-upload-stage"
    fake_api = FakeHfApi()

    result = upload_orbitquant_artifact(
        tmp_path,
        repo_id="WaveCut/example-orbitquant",
        upload_profile="compact",
        staging_dir=stage_dir,
        validate_tensors=False,
        api=fake_api,
    )

    assert result["upload_profile"] == "compact"
    assert result["staging"]["enabled"] is True
    assert result["staging"]["omitted_raw_eval_asset_count"] == 2
    assert len(fake_api.upload_folder_calls) == 1
    assert fake_api.upload_folder_calls[0]["folder_path"] == str(stage_dir)
    assert not (stage_dir / raw_eval_asset.relative_to(tmp_path)).exists()
    assert not (stage_dir / visual_asset.relative_to(tmp_path)).exists()
    assert not (stage_dir / "benchmark" / "original.metrics.jsonl").exists()
    assert (stage_dir / "benchmark" / "summary.json").is_file()
    staged_summary = json.loads((stage_dir / "benchmark" / "summary.json").read_text())
    assert staged_summary["published_summary"] == "compact"
    assert staged_summary["raw_generation_records"] == "local-only"
    assert result["validation"]["valid"] is True
    assert result["upload"]["commit_oid"] == "uploaded-sha"


def test_upload_orbitquant_artifact_defaults_to_compact_staged_copy(tmp_path):
    _write_artifact(tmp_path)
    raw_eval_asset = tmp_path / "assets" / "flux2-native_seed0_W4A4_geneval-00000.png"
    visual_asset = tmp_path / "assets" / "flux2-native_seed0_W4A4_simple-object.png"
    raw_eval_asset.write_bytes(b"raw eval image")
    visual_asset.write_bytes(b"visual image")
    refresh_artifact_checksums(tmp_path)
    fake_api = FakeHfApi()

    result = upload_orbitquant_artifact(
        tmp_path,
        repo_id="WaveCut/example-orbitquant",
        validate_tensors=False,
        api=fake_api,
    )

    assert result["upload_profile"] == "compact"
    assert result["replace_repo_files"] is True
    assert result["upload_kwargs"]["delete_patterns"] == "*"
    assert result["staging"]["enabled"] is True
    assert result["staging"]["omitted_raw_eval_asset_count"] == 2
    assert set(result["staging"]["omitted_raw_eval_assets"]) == {
        "assets/flux2-native_seed0_W4A4_geneval-00000.png",
        "assets/flux2-native_seed0_W4A4_simple-object.png",
    }
    upload_path = Path(fake_api.upload_folder_calls[0]["folder_path"])
    assert upload_path != tmp_path


def test_upload_orbitquant_artifact_rejects_remote_file_replacement_opt_out(tmp_path):
    _write_artifact(tmp_path)
    fake_api = FakeHfApi()

    with pytest.raises(ValueError, match="must replace remote files"):
        upload_orbitquant_artifact(
            tmp_path,
            repo_id="WaveCut/example-orbitquant",
            replace_repo_files=False,
            validate_tensors=False,
            api=fake_api,
        )

    assert fake_api.create_repo_calls == []
    assert fake_api.upload_folder_calls == []
    assert fake_api.model_info_calls == []


def test_upload_orbitquant_artifact_rejects_invalid_artifact_before_hub_calls(tmp_path):
    fake_api = FakeHfApi()

    with pytest.raises(RuntimeError, match="required artifact file missing"):
        upload_orbitquant_artifact(
            tmp_path,
            repo_id="WaveCut/bad-artifact",
            validate_tensors=False,
            api=fake_api,
        )

    assert fake_api.create_repo_calls == []
    assert fake_api.upload_folder_calls == []
    assert fake_api.model_info_calls == []
