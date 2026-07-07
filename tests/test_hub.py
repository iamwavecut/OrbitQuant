import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from huggingface_hub import CommitOperationAdd, CommitOperationDelete

import orbitquant.hub as hub_module
from orbitquant.artifacts import (
    record_artifact_metrics,
    refresh_artifact_checksums,
    save_orbitquant_artifact,
)
from orbitquant.artifacts.checksums import read_sha256sums, write_sha256sums
from orbitquant.config import OrbitQuantConfig
from orbitquant.eval.native_settings import NativeSuite
from orbitquant.hub import (
    audit_hf_artifact_repos,
    cleanup_hf_artifact_reports,
    cleanup_hf_artifact_reports_matrix,
    fetch_hf_artifacts,
    render_hf_artifact_audit_markdown,
    repair_hf_artifact_metadata,
    repair_hf_artifact_metadata_matrix,
    repair_hf_native_smoke_proof,
    repair_hf_native_smoke_proof_matrix,
    stage_compact_upload_artifact,
    upload_orbitquant_artifact,
)
from orbitquant.modeling import quantize_linear_modules


class TinyHubArtifactModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList(
            [torch.nn.ModuleDict({"attn": torch.nn.ModuleDict({"to_q": torch.nn.Linear(8, 8)})})]
        )


class FakeHfApi:
    def __init__(self):
        self.create_repo_calls = []
        self.upload_folder_calls = []
        self.model_info_calls = []

    def create_repo(self, **kwargs):
        self.create_repo_calls.append(kwargs)
        return f"https://huggingface.co/{kwargs['repo_id']}"

    def upload_folder(self, **kwargs):
        self.upload_folder_calls.append(kwargs)
        return SimpleNamespace(
            oid="uploaded-sha",
            commit_url=f"https://huggingface.co/{kwargs['repo_id']}/commit/uploaded-sha",
            pr_url=None,
        )

    def model_info(self, repo_id, *, revision=None):
        self.model_info_calls.append({"repo_id": repo_id, "revision": revision})
        return SimpleNamespace(sha=revision or "main-sha", private=True, gated=False)


class FakeAuditHfApi:
    def __init__(self, siblings_by_repo):
        self.siblings_by_repo = siblings_by_repo
        self.model_info_calls = []

    def model_info(self, repo_id, *, revision=None, files_metadata=False):
        self.model_info_calls.append(
            {"repo_id": repo_id, "revision": revision, "files_metadata": files_metadata}
        )
        if repo_id not in self.siblings_by_repo:
            raise RuntimeError("missing repo")
        siblings = []
        for name, metadata in self.siblings_by_repo[repo_id].items():
            if isinstance(metadata, dict):
                lfs_sha256 = metadata.get("lfs_sha256")
                lfs = (
                    SimpleNamespace(sha256=lfs_sha256, size=metadata.get("size"))
                    if lfs_sha256 is not None
                    else None
                )
                siblings.append(
                    SimpleNamespace(rfilename=name, size=metadata.get("size"), lfs=lfs)
                )
            else:
                siblings.append(SimpleNamespace(rfilename=name, size=metadata, lfs=None))
        return SimpleNamespace(
            sha="remote-sha",
            private=True,
            gated=False,
            siblings=siblings,
        )


class FakeCommitHfApi:
    def __init__(self):
        self.create_commit_calls = []

    def create_commit(self, **kwargs):
        self.create_commit_calls.append(kwargs)
        return SimpleNamespace(
            oid="repair-sha",
            commit_url=f"https://huggingface.co/{kwargs['repo_id']}/commit/repair-sha",
            pr_url=None,
        )


class FakeCleanupHfApi(FakeCommitHfApi):
    def __init__(self, artifact_dir):
        super().__init__()
        self.artifact_dir = artifact_dir
        self.model_info_calls = []

    def model_info(self, repo_id, *, revision=None, files_metadata=False):
        self.model_info_calls.append(
            {"repo_id": repo_id, "revision": revision, "files_metadata": files_metadata}
        )
        siblings = [
            SimpleNamespace(rfilename=path.relative_to(self.artifact_dir).as_posix(), size=1)
            for path in sorted(self.artifact_dir.rglob("*"))
            if path.is_file()
        ]
        return SimpleNamespace(sha="remote-sha", private=True, gated=False, siblings=siblings)


def _write_artifact(tmp_path):
    model = TinyHubArtifactModel()
    config = OrbitQuantConfig(block_size=4, target_policy="generic_dit")
    summary = quantize_linear_modules(model, config, quantization_device=None)
    return save_orbitquant_artifact(
        model,
        tmp_path,
        config=config,
        source_model_id="example/model",
        source_revision="abc123",
        source_license="apache-2.0",
        summary=summary,
    )


def _required_remote_files():
    return {name: 1 for name in hub_module._REQUIRED_ARTIFACT_FILES}


def _remote_file_map(repo_id, artifact_dir):
    return {
        (repo_id, "orbitquant_manifest.json"): artifact_dir / "orbitquant_manifest.json",
        (repo_id, "model_index.json"): artifact_dir / "model_index.json",
        (repo_id, "benchmark/summary.json"): artifact_dir / "benchmark" / "summary.json",
        (repo_id, "quantization_config.json"): artifact_dir / "quantization_config.json",
        (repo_id, "README.md"): artifact_dir / "README.md",
        (repo_id, "SHA256SUMS"): artifact_dir / "SHA256SUMS",
    }


def _remote_path(artifact_dir, filename):
    return artifact_dir / filename


def _write_remote_model_index(
    path,
    *,
    activation_eps=None,
    quantization_device="unknown",
    weight_quantization_backend="unknown",
    quantization_staging_mode="unknown",
):
    payload = {
        "_class_name": "OrbitQuantArtifact",
        "quantization_device": quantization_device,
        "weight_quantization_backend": weight_quantization_backend,
        "quantization_staging_mode": quantization_staging_mode,
    }
    if activation_eps is not None:
        payload["activation_eps"] = activation_eps
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _audit_file_map(
    repo_id,
    *,
    manifest_path,
    sha256sums_path,
    summary_path,
    model_index_path,
    quantization_config_path=None,
):
    file_map = {
        (repo_id, "orbitquant_manifest.json"): manifest_path,
        (repo_id, "model_index.json"): model_index_path,
        (repo_id, "SHA256SUMS"): sha256sums_path,
        (repo_id, "benchmark/summary.json"): summary_path,
    }
    if quantization_config_path is not None:
        file_map[(repo_id, "quantization_config.json")] = quantization_config_path
    return file_map


def _native_smoke_summary(
    suite,
    *,
    release_metrics=None,
    activation_eps=None,
    quantization_device="unknown",
    weight_quantization_backend="unknown",
    quantization_staging_mode="unknown",
):
    native_settings = {
        "suite": suite.name,
        "height": suite.height,
        "width": suite.width,
        "frames": suite.frames,
        "steps": suite.steps,
        "guidance": suite.guidance,
    }
    generated_frames = 0 if suite.frames is None else suite.frames
    metrics = {"generated_samples": 1}
    if generated_frames:
        metrics["generated_frames"] = generated_frames
    for key, value in (release_metrics or {}).items():
        metrics[key] = value
    split_proof = {
        "records": 1,
        "generated_samples": 1,
        "generated_frames": generated_frames,
        "nonempty_output_count": 1,
        "seeds": ["0"],
        "prompt_ids": ["simple-object"],
        "pair_keys": [[suite.name, "0", "simple-object"]],
        "native_settings": [native_settings],
    }
    payload = {
        "published_summary": "compact",
        "raw_generation_records": "local-only",
        "quantization_device": quantization_device,
        "weight_quantization_backend": weight_quantization_backend,
        "quantization_staging_mode": quantization_staging_mode,
        "metrics": {
            "original": {"records": 1, "latest_metrics": metrics},
            "orbitquant": {"records": 1, "latest_metrics": metrics},
        },
        "native_smoke": {
            "proof_format": "orbitquant-native-smoke-v1",
            "comparison_asset_path": "assets/image_generation_comparison_matrix.webp",
            "paired_prompt_seed_count": 1,
            "paired_prompt_seed_keys": [[suite.name, "0", "simple-object"]],
            "splits": {
                "original": split_proof,
                "orbitquant": split_proof,
            },
        },
    }
    if activation_eps is not None:
        payload["activation_eps"] = activation_eps
    return json.dumps(payload, indent=2)


def _legacy_compact_summary_without_native_smoke(
    suite,
    *,
    generated_samples=1,
    generated_frames=0,
):
    metrics = {"generated_samples": generated_samples}
    if generated_frames:
        metrics["generated_frames"] = generated_frames
    return json.dumps(
        {
            "published_summary": "compact",
            "raw_generation_records": "local-only",
            "activation_eps": 1e-10,
            "quantization_device": "cuda",
            "weight_quantization_backend": "triton_cuda",
            "quantization_staging_mode": "component",
            "metrics": {
                "original": {"records": generated_samples, "latest_metrics": metrics},
                "orbitquant": {"records": generated_samples, "latest_metrics": metrics},
            },
        },
        indent=2,
    )


def _expected_missing_geneval_metrics():
    return [
        {"split": split, "metric": metric}
        for metric in hub_module._GENEVAL_REQUIRED_METRICS
        for split in ("original", "orbitquant")
    ]


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
    summary = json.loads(
        _legacy_compact_summary_without_native_smoke(suite, generated_samples=0)
    )
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
        tmp_path
        / "assets"
        / "original_vs_orbitquant_flux2-native_seed0_W4A4_simple-object.webp"
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
    assert result["omitted_raw_eval_asset_count"] == 10
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
        (unexpected_file, b"debug"),
        (ignored_cache, b"ignored"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    refresh_artifact_checksums(tmp_path)

    stage_dir = tmp_path.parent / f"{tmp_path.name}-asset-allowlist-stage"
    result = stage_compact_upload_artifact(tmp_path, stage_dir, validate_tensors=False)

    staged_files = {
        path.relative_to(stage_dir).as_posix()
        for path in stage_dir.rglob("*")
        if path.is_file()
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
    assert "## Native Validation Proof" in staged_readme
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


def test_repair_hf_artifact_metadata_dry_run_preserves_large_file_checksum(
    tmp_path,
    monkeypatch,
):
    repo_id = "WaveCut/example-orbitquant"
    _write_artifact(tmp_path)
    old_checksums = read_sha256sums(tmp_path / "SHA256SUMS")

    def fake_download(repo, filename, **kwargs):
        return str(_remote_file_map(repo_id, tmp_path)[(repo, filename)])

    monkeypatch.setattr(hub_module, "hf_hub_download", fake_download)
    fake_api = FakeCommitHfApi()

    result = repair_hf_artifact_metadata(
        repo_id=repo_id,
        quantization_device="cuda",
        weight_quantization_backend="triton_cuda",
        quantization_staging_mode="component",
        dry_run=True,
        api=fake_api,
    )

    assert result["dry_run"] is True
    assert result["updated"]["quantization_staging_mode"] == "component"
    assert result["commit"] is None
    assert "model.safetensors" in result["preserved_checksum_entries"]
    assert old_checksums["model.safetensors"]
    assert fake_api.create_commit_calls == []


def test_repair_hf_artifact_metadata_commits_only_metadata_files_and_sha256sums(
    tmp_path,
    monkeypatch,
):
    repo_id = "WaveCut/example-orbitquant"
    _write_artifact(tmp_path)
    stale_cache_metadata = tmp_path / ".cache" / "huggingface" / "download" / "README.md.metadata"
    stale_cache_metadata.parent.mkdir(parents=True)
    stale_cache_metadata.write_text("transient hub metadata", encoding="utf-8")
    with (tmp_path / "SHA256SUMS").open("a", encoding="utf-8") as handle:
        handle.write(
            "0" * 64 + "  .cache/huggingface/download/README.md.metadata\n"
        )
    old_checksums = read_sha256sums(tmp_path / "SHA256SUMS")

    def fake_download(repo, filename, **kwargs):
        return str(_remote_file_map(repo_id, tmp_path)[(repo, filename)])

    monkeypatch.setattr(hub_module, "hf_hub_download", fake_download)
    fake_api = FakeCommitHfApi()

    result = repair_hf_artifact_metadata(
        repo_id=repo_id,
        quantization_device="cuda",
        weight_quantization_backend="triton_cuda",
        quantization_staging_mode="component",
        revision="main",
        commit_message="repair metadata",
        api=fake_api,
    )

    assert result["commit"]["commit_oid"] == "repair-sha"
    assert len(fake_api.create_commit_calls) == 1
    commit_call = fake_api.create_commit_calls[0]
    assert commit_call["repo_id"] == repo_id
    assert commit_call["repo_type"] == "model"
    assert commit_call["revision"] == "main"
    assert commit_call["commit_message"] == "repair metadata"
    operation_paths = {operation.path_in_repo for operation in commit_call["operations"]}
    assert operation_paths == {
        "orbitquant_manifest.json",
        "model_index.json",
        "benchmark/summary.json",
        "README.md",
        "SHA256SUMS",
    }
    assert "model.safetensors" not in operation_paths
    sha_operation = next(
        operation
        for operation in commit_call["operations"]
        if operation.path_in_repo == "SHA256SUMS"
    )
    sha_entries = hub_module._parse_sha256sums_bytes(sha_operation.path_or_fileobj)
    assert sha_entries["model.safetensors"] == old_checksums["model.safetensors"]
    assert sha_entries["orbitquant_manifest.json"] != old_checksums["orbitquant_manifest.json"]
    assert ".cache/huggingface/download/README.md.metadata" not in sha_entries
    manifest_operation = next(
        operation
        for operation in commit_call["operations"]
        if operation.path_in_repo == "orbitquant_manifest.json"
    )
    assert (
        b'"quantization_staging_mode": "component"'
        in manifest_operation.path_or_fileobj
    )
    assert b'"activation_eps": 1e-10' in manifest_operation.path_or_fileobj


def test_cleanup_hf_artifact_reports_promotes_matrices_and_deletes_report_folder(
    tmp_path,
    monkeypatch,
):
    repo_id = "WaveCut/example-orbitquant"
    _write_artifact(tmp_path)
    raw_asset = tmp_path / "assets" / "flux2-native_seed0_W4A4_simple-object.png"
    raw_sidecar = raw_asset.with_suffix(".png.json")
    raw_asset.write_bytes(b"raw generated image")
    raw_sidecar.write_text('{"prompt_id":"simple-object"}\n', encoding="utf-8")
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
    report_markdown = report_matrix.parent.parent / "orbitquant-native-eval.md"
    report_markdown.write_text("# local report log\n", encoding="utf-8")
    refresh_artifact_checksums(tmp_path)

    def fake_download(repo, filename, **kwargs):
        return str(_remote_path(tmp_path, filename))

    monkeypatch.setattr(hub_module, "hf_hub_download", fake_download)
    fake_api = FakeCleanupHfApi(tmp_path)

    result = cleanup_hf_artifact_reports(
        repo_id=repo_id,
        commit_message="cleanup reports",
        api=fake_api,
    )

    assert result["report_file_count"] == 2
    assert result["report_matrix_count"] == 1
    assert result["forbidden_file_count"] == 8
    assert result["promoted_assets"] == ["assets/image_generation_comparison_matrix.webp"]
    assert result["delete_paths"] == [
        "assets/flux2-native_seed0_W4A4_simple-object.png",
        "assets/flux2-native_seed0_W4A4_simple-object.png.json",
        "benchmark/orbitquant.metrics.csv",
        "benchmark/orbitquant.metrics.jsonl",
        "benchmark/original.metrics.csv",
        "benchmark/original.metrics.jsonl",
        "reports",
    ]
    assert result["commit"]["commit_oid"] == "repair-sha"
    assert len(fake_api.create_commit_calls) == 1

    commit_call = fake_api.create_commit_calls[0]
    assert commit_call["commit_message"] == "cleanup reports"
    operations = commit_call["operations"]
    added = {
        operation.path_in_repo: operation.path_or_fileobj
        for operation in operations
        if isinstance(operation, CommitOperationAdd)
    }
    deleted = [
        operation
        for operation in operations
        if isinstance(operation, CommitOperationDelete)
    ]
    assert "model.safetensors" not in added
    assert added["assets/image_generation_comparison_matrix.webp"] == b"report matrix"
    assert len(deleted) == 7
    deleted_paths = {operation.path_in_repo: operation.is_folder for operation in deleted}
    assert deleted_paths == {
        "assets/flux2-native_seed0_W4A4_simple-object.png": False,
        "assets/flux2-native_seed0_W4A4_simple-object.png.json": False,
        "benchmark/orbitquant.metrics.csv": False,
        "benchmark/orbitquant.metrics.jsonl": False,
        "benchmark/original.metrics.csv": False,
        "benchmark/original.metrics.jsonl": False,
        "reports": True,
    }

    next_manifest = json.loads(added["orbitquant_manifest.json"].decode("utf-8"))
    next_summary = json.loads(added["benchmark/summary.json"].decode("utf-8"))
    next_readme = added["README.md"].decode("utf-8")
    next_sha = added["SHA256SUMS"].decode("utf-8")
    assert "assets/image_generation_comparison_matrix.webp" in next_manifest["checksums"]
    assert "benchmark/summary.json" in next_manifest["checksums"]
    assert not any(path.endswith(".metrics.jsonl") for path in next_manifest["checksums"])
    assert not any(path.endswith(".metrics.csv") for path in next_manifest["checksums"])
    assert not any(path.startswith("reports/") for path in next_manifest["checksums"])
    assert next_summary["published_summary"] == "compact"
    assert next_summary["raw_generation_records"] == "local-only"
    assert "assets/image_generation_comparison_matrix.webp" in next_readme
    assert "## Validation Status" in next_readme
    assert "reports/native" not in next_readme
    assert "assets/image_generation_comparison_matrix.webp" in next_sha
    assert "benchmark/original.metrics.jsonl" not in next_sha
    assert "reports/native" not in next_sha


def test_cleanup_hf_artifact_reports_deletes_extra_comparison_matrix_assets(
    tmp_path,
    monkeypatch,
):
    repo_id = "WaveCut/example-orbitquant"
    _write_artifact(tmp_path)
    image_matrix = tmp_path / "assets" / "image_generation_comparison_matrix.webp"
    video_matrix = tmp_path / "assets" / "video_generation_comparison_matrix.webp"
    image_matrix.write_bytes(b"image matrix")
    video_matrix.write_bytes(b"video matrix")
    refresh_artifact_checksums(tmp_path)

    def fake_download(repo, filename, **kwargs):
        return str(_remote_path(tmp_path, filename))

    monkeypatch.setattr(hub_module, "hf_hub_download", fake_download)
    fake_api = FakeCleanupHfApi(tmp_path)

    result = cleanup_hf_artifact_reports(
        repo_id=repo_id,
        commit_message="cleanup extra matrix",
        api=fake_api,
    )

    assert result["promoted_assets"] == []
    assert result["forbidden_files"] == [
        "assets/video_generation_comparison_matrix.webp",
        "benchmark/orbitquant.metrics.csv",
        "benchmark/orbitquant.metrics.jsonl",
        "benchmark/original.metrics.csv",
        "benchmark/original.metrics.jsonl",
    ]
    assert "assets/video_generation_comparison_matrix.webp" in result["delete_paths"]
    commit_call = fake_api.create_commit_calls[0]
    operations = commit_call["operations"]
    added = {
        operation.path_in_repo: operation.path_or_fileobj
        for operation in operations
        if isinstance(operation, CommitOperationAdd)
    }
    deleted_paths = {
        operation.path_in_repo
        for operation in operations
        if isinstance(operation, CommitOperationDelete)
    }
    assert "assets/video_generation_comparison_matrix.webp" in deleted_paths
    next_manifest = json.loads(added["orbitquant_manifest.json"].decode("utf-8"))
    next_readme = added.get("README.md", (tmp_path / "README.md").read_bytes()).decode("utf-8")
    assert "assets/image_generation_comparison_matrix.webp" in next_manifest["checksums"]
    assert "assets/video_generation_comparison_matrix.webp" not in next_manifest["checksums"]
    assert "assets/image_generation_comparison_matrix.webp" in next_readme
    assert "assets/video_generation_comparison_matrix.webp" not in next_readme


def test_cleanup_hf_artifact_reports_matrix_cleans_expected_suite_repo(
    tmp_path,
    monkeypatch,
):
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
    repo_id = "WaveCut/FLUX.2-klein-4B-OrbitQuant-W4A4"
    _write_artifact(tmp_path)
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

    def fake_download(repo, filename, **kwargs):
        return str(_remote_path(tmp_path, filename))

    monkeypatch.setattr(hub_module, "hf_hub_download", fake_download)
    fake_api = FakeCleanupHfApi(tmp_path)

    result = cleanup_hf_artifact_reports_matrix(
        suites=[suite],
        dry_run=True,
        api=fake_api,
    )

    assert result["repo_count"] == 1
    assert result["changed_repo_count"] == 1
    assert result["error_count"] == 0
    assert result["report_file_count"] == 1
    assert result["promoted_asset_count"] == 1
    assert result["rows"][0]["repo_id"] == repo_id
    assert result["rows"][0]["suite"] == "flux2-native"
    assert result["rows"][0]["bit_setting"] == "W4A4"


def test_repair_hf_artifact_metadata_matrix_repairs_expected_suite_repo(
    tmp_path,
    monkeypatch,
):
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
    repo_id = "WaveCut/FLUX.2-klein-4B-OrbitQuant-W4A4"
    _write_artifact(tmp_path)

    def fake_download(repo, filename, **kwargs):
        return str(_remote_file_map(repo_id, tmp_path)[(repo, filename)])

    monkeypatch.setattr(hub_module, "hf_hub_download", fake_download)
    fake_api = FakeCommitHfApi()

    result = repair_hf_artifact_metadata_matrix(
        suites=[suite],
        quantization_device="cuda",
        weight_quantization_backend="triton_cuda",
        quantization_staging_mode="component",
        dry_run=True,
        api=fake_api,
    )

    assert result["repo_count"] == 1
    assert result["error_count"] == 0
    assert result["rows"][0]["repo_id"] == repo_id
    assert result["rows"][0]["suite"] == "flux2-native"
    assert result["rows"][0]["bit_setting"] == "W4A4"


def test_repair_hf_native_smoke_proof_skips_compact_summary_without_raw_pair_evidence(
    tmp_path,
    monkeypatch,
):
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
    repo_id = "WaveCut/FLUX.2-klein-4B-OrbitQuant-W4A4"
    _write_artifact(tmp_path)
    matrix = tmp_path / "assets" / "image_generation_comparison_matrix.webp"
    matrix.parent.mkdir(parents=True, exist_ok=True)
    matrix.write_bytes(b"matrix")
    summary_path = tmp_path / "benchmark" / "summary.json"
    summary_path.write_text(
        _legacy_compact_summary_without_native_smoke(suite),
        encoding="utf-8",
    )
    refresh_artifact_checksums(tmp_path)

    def fake_download(repo, filename, **kwargs):
        return str(_remote_file_map(repo_id, tmp_path)[(repo, filename)])

    monkeypatch.setattr(hub_module, "hf_hub_download", fake_download)
    fake_api = FakeCleanupHfApi(tmp_path)

    result = repair_hf_native_smoke_proof(
        repo_id=repo_id,
        suite=suite,
        revision="main",
        commit_message="repair native smoke proof",
        api=fake_api,
    )

    assert result["commit"] is None
    assert result["repair_skipped_reason"] == "raw_paired_native_smoke_evidence_missing"
    assert result["changed_files"] == []
    assert fake_api.create_commit_calls == []


def test_repair_hf_native_smoke_proof_refreshes_stale_readme_when_proof_exists(
    tmp_path,
    monkeypatch,
):
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
    repo_id = "WaveCut/FLUX.2-klein-4B-OrbitQuant-W4A4"
    _write_artifact(tmp_path)
    matrix = tmp_path / "assets" / "image_generation_comparison_matrix.webp"
    matrix.parent.mkdir(parents=True, exist_ok=True)
    matrix.write_bytes(b"matrix")
    summary_path = tmp_path / "benchmark" / "summary.json"
    summary_path.write_text(_native_smoke_summary(suite), encoding="utf-8")
    refresh_artifact_checksums(tmp_path)
    manifest = hub_module.OrbitQuantManifest.from_dict(
        json.loads((tmp_path / "orbitquant_manifest.json").read_text())
    )
    (tmp_path / "README.md").write_text(hub_module.render_model_card(manifest), encoding="utf-8")
    write_sha256sums(tmp_path)

    def fake_download(repo, filename, **kwargs):
        return str(_remote_file_map(repo_id, tmp_path)[(repo, filename)])

    monkeypatch.setattr(hub_module, "hf_hub_download", fake_download)
    fake_api = FakeCleanupHfApi(tmp_path)

    result = repair_hf_native_smoke_proof(
        repo_id=repo_id,
        suite=suite,
        revision="main",
        api=fake_api,
    )

    assert result["existing_native_smoke_ready"] is True
    assert result["changed_files"] == ["README.md", "SHA256SUMS"]
    operation_by_path = {
        operation.path_in_repo: operation.path_or_fileobj
        for operation in fake_api.create_commit_calls[0]["operations"]
    }
    repaired_readme = operation_by_path["README.md"].decode("utf-8")
    assert "## Native Validation Proof" in repaired_readme
    assert "| Paired prompt/seed count | `1` |" in repaired_readme


def test_repair_hf_native_smoke_proof_skips_when_video_frames_are_insufficient(
    tmp_path,
    monkeypatch,
):
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
    repo_id = "WaveCut/Wan2.1-T2V-1.3B-Diffusers-OrbitQuant-W4A4"
    _write_artifact(tmp_path)
    matrix = tmp_path / "assets" / "video_generation_comparison_matrix.webp"
    matrix.parent.mkdir(parents=True, exist_ok=True)
    matrix.write_bytes(b"matrix")
    (tmp_path / "benchmark" / "summary.json").write_text(
        _legacy_compact_summary_without_native_smoke(
            suite,
            generated_samples=1,
            generated_frames=1,
        ),
        encoding="utf-8",
    )
    refresh_artifact_checksums(tmp_path)

    def fake_download(repo, filename, **kwargs):
        return str(_remote_file_map(repo_id, tmp_path)[(repo, filename)])

    monkeypatch.setattr(hub_module, "hf_hub_download", fake_download)
    fake_api = FakeCleanupHfApi(tmp_path)

    result = repair_hf_native_smoke_proof(
        repo_id=repo_id,
        suite=suite,
        dry_run=True,
        api=fake_api,
    )

    assert result["commit"] is None
    assert result["changed_files"] == []
    assert result["repair_skipped_reason"] == "raw_paired_native_smoke_evidence_missing"
    assert fake_api.create_commit_calls == []


def test_repair_hf_native_smoke_proof_matrix_skips_compact_summary_without_raw_pair_evidence(
    tmp_path,
    monkeypatch,
):
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
    repo_id = "WaveCut/FLUX.2-klein-4B-OrbitQuant-W4A4"
    _write_artifact(tmp_path)
    matrix = tmp_path / "assets" / "image_generation_comparison_matrix.webp"
    matrix.parent.mkdir(parents=True, exist_ok=True)
    matrix.write_bytes(b"matrix")
    (tmp_path / "benchmark" / "summary.json").write_text(
        _legacy_compact_summary_without_native_smoke(suite),
        encoding="utf-8",
    )
    refresh_artifact_checksums(tmp_path)

    def fake_download(repo, filename, **kwargs):
        return str(_remote_file_map(repo_id, tmp_path)[(repo, filename)])

    monkeypatch.setattr(hub_module, "hf_hub_download", fake_download)
    fake_api = FakeCleanupHfApi(tmp_path)

    result = repair_hf_native_smoke_proof_matrix(
        suites=[suite],
        dry_run=True,
        api=fake_api,
    )

    assert result["repo_count"] == 1
    assert result["changed_repo_count"] == 0
    assert result["skipped_repo_count"] == 1
    assert result["error_count"] == 0
    assert result["rows"][0]["repo_id"] == repo_id
    assert result["rows"][0]["suite"] == "flux2-native"
    assert result["rows"][0]["bit_setting"] == "W4A4"
    assert result["rows"][0]["repair_skipped_reason"] == (
        "raw_paired_native_smoke_evidence_missing"
    )


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
    assert result["staging"]["omitted_raw_eval_asset_count"] == 6
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
    assert result["staging"]["omitted_raw_eval_asset_count"] == 6
    assert set(result["staging"]["omitted_raw_eval_assets"]) == {
        "assets/flux2-native_seed0_W4A4_geneval-00000.png",
        "assets/flux2-native_seed0_W4A4_simple-object.png",
        "benchmark/orbitquant.metrics.csv",
        "benchmark/orbitquant.metrics.jsonl",
        "benchmark/original.metrics.csv",
        "benchmark/original.metrics.jsonl",
    }
    upload_path = Path(fake_api.upload_folder_calls[0]["folder_path"])
    assert upload_path != tmp_path


def test_upload_orbitquant_artifact_can_opt_out_of_remote_file_replacement(tmp_path):
    _write_artifact(tmp_path)
    fake_api = FakeHfApi()

    result = upload_orbitquant_artifact(
        tmp_path,
        repo_id="WaveCut/example-orbitquant",
        replace_repo_files=False,
        validate_tensors=False,
        api=fake_api,
    )

    assert result["replace_repo_files"] is False
    assert result["upload_kwargs"]["delete_patterns"] is None
    assert fake_api.upload_folder_calls[0]["delete_patterns"] is None


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


def test_fetch_hf_artifacts_dry_run_reports_native_artifact_layout(tmp_path):
    suite = NativeSuite(
        name="flux1-schnell-native",
        model_id="black-forest-labs/FLUX.1-schnell",
        pipeline="FluxPipeline",
        width=1024,
        height=1024,
        steps=4,
        guidance=0.0,
        bit_settings=["W4A4", "W3A3"],
        metric="geneval",
    )

    result = fetch_hf_artifacts(
        namespace="WaveCut",
        suites=[suite],
        output_root=tmp_path / "artifacts",
        dry_run=True,
    )

    assert result["repo_count"] == 2
    assert result["downloaded_count"] == 0
    assert result["dry_run"] is True
    assert result["rows"][0]["repo_id"] == "WaveCut/FLUX.1-schnell-OrbitQuant-W4A4"
    assert result["rows"][0]["artifact_dir"].endswith("flux1-schnell-native-w4a4")
    assert result["rows"][1]["repo_id"] == "WaveCut/FLUX.1-schnell-OrbitQuant-W3A3"
    assert result["rows"][1]["artifact_dir"].endswith("flux1-schnell-native-w3a3")


def test_fetch_hf_artifacts_downloads_and_validates_artifact(
    tmp_path,
    monkeypatch,
):
    source_dir = tmp_path / "source"
    _write_artifact(source_dir)
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
    calls = []
    stages = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        destination = Path(kwargs["local_dir"])
        shutil.copytree(source_dir, destination, dirs_exist_ok=True)
        return str(destination)

    monkeypatch.setattr(hub_module, "snapshot_download", fake_snapshot_download)

    result = fetch_hf_artifacts(
        namespace="WaveCut",
        suites=[suite],
        output_root=tmp_path / "artifacts",
        revision="main",
        stage_logger=lambda event, label: stages.append((event, label)),
    )

    artifact_dir = tmp_path / "artifacts" / "flux2-native-w4a4"
    assert result["downloaded_count"] == 1
    assert result["skipped_existing_count"] == 0
    assert result["rows"][0]["artifact_dir"] == str(artifact_dir)
    assert result["rows"][0]["validation"]["valid"] is True
    assert calls == [
        {
            "repo_id": "WaveCut/FLUX.2-klein-4B-OrbitQuant-W4A4",
            "repo_type": "model",
            "revision": "main",
            "local_dir": artifact_dir,
            "force_download": False,
            "local_files_only": False,
        }
    ]
    assert stages == [
        ("START", "flux2-native W4A4 fetch WaveCut/FLUX.2-klein-4B-OrbitQuant-W4A4"),
        ("END", "flux2-native W4A4 fetch WaveCut/FLUX.2-klein-4B-OrbitQuant-W4A4"),
    ]


def test_fetch_hf_artifacts_resume_skips_valid_existing_artifact(tmp_path, monkeypatch):
    artifact_dir = tmp_path / "artifacts" / "flux2-native-w4a4"
    _write_artifact(artifact_dir)
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

    def fail_snapshot_download(**kwargs):
        raise AssertionError("resume should not download a valid existing artifact")

    monkeypatch.setattr(hub_module, "snapshot_download", fail_snapshot_download)

    result = fetch_hf_artifacts(
        namespace="WaveCut",
        suites=[suite],
        output_root=tmp_path / "artifacts",
    )

    assert result["downloaded_count"] == 0
    assert result["skipped_existing_count"] == 1
    assert result["rows"][0]["skipped_existing"] is True
    assert result["rows"][0]["validation"]["valid"] is True


def test_fetch_hf_artifacts_logs_error_stage_on_download_failure(tmp_path, monkeypatch):
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
    stages = []

    def fail_snapshot_download(**kwargs):
        raise RuntimeError("download failed")

    monkeypatch.setattr(hub_module, "snapshot_download", fail_snapshot_download)

    with pytest.raises(RuntimeError, match="download failed"):
        fetch_hf_artifacts(
            namespace="WaveCut",
            suites=[suite],
            output_root=tmp_path / "artifacts",
            stage_logger=lambda event, label: stages.append((event, label)),
        )

    assert stages == [
        ("START", "flux2-native W4A4 fetch WaveCut/FLUX.2-klein-4B-OrbitQuant-W4A4"),
        ("ERROR", "flux2-native W4A4 fetch WaveCut/FLUX.2-klein-4B-OrbitQuant-W4A4"),
    ]


def test_audit_hf_artifact_repos_flags_native_smoke_ready_but_missing_release_metrics(
    tmp_path,
    monkeypatch,
):
    suite = NativeSuite(
        name="flux1-schnell-native",
        model_id="black-forest-labs/FLUX.1-schnell",
        pipeline="FluxPipeline",
        width=1024,
        height=1024,
        steps=4,
        guidance=0.0,
        bit_settings=["W4A4"],
        metric="geneval",
    )
    repo_id = "WaveCut/FLUX.1-schnell-OrbitQuant-W4A4"
    siblings = _required_remote_files()
    siblings.update(
        {
            "model.safetensors": {"size": 123, "lfs_sha256": "a" * 64},
            "assets/image_generation_comparison_matrix.webp": 10,
        }
    )
    api = FakeAuditHfApi({repo_id: siblings})
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        """{
          "source_model_id": "black-forest-labs/FLUX.1-schnell",
          "weight_bits": 4,
          "activation_bits": 4,
          "target_policy": "flux",
          "quantized_modules": ["block.attn.to_q"],
          "adaln_modules": ["block.modulation"],
          "checksums": {
            "model.safetensors": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
          }
        }"""
    )
    sha256sums_path = tmp_path / "SHA256SUMS"
    sha256sums_path.write_text(
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa  model.safetensors\n"
    )
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(_native_smoke_summary(suite), encoding="utf-8")
    model_index_path = tmp_path / "model_index.json"
    _write_remote_model_index(model_index_path)
    file_map = _audit_file_map(
        repo_id,
        manifest_path=manifest_path,
        sha256sums_path=sha256sums_path,
        summary_path=summary_path,
        model_index_path=model_index_path,
    )

    def fake_download(repo, filename, **kwargs):
        return str(file_map[(repo, filename)])

    monkeypatch.setattr(hub_module, "hf_hub_download", fake_download)

    result = audit_hf_artifact_repos(suites=[suite], api=api)

    assert result["repo_count"] == 1
    assert result["existing_count"] == 1
    assert result["artifact_ready_count"] == 1
    assert result["native_smoke_ready_count"] == 1
    assert result["release_eval_applicable_count"] == 1
    assert result["release_eval_not_applicable_count"] == 0
    assert result["release_eval_ready_count"] == 0
    assert result["forbidden_file_count"] == 0
    assert result["missing_required_metric_count"] == len(_expected_missing_geneval_metrics())
    row = result["rows"][0]
    assert row["repo_id"] == repo_id
    assert row["artifact_ready"] is True
    assert row["native_smoke_ready"] is True
    assert row["release_eval_applicable"] is True
    assert row["release_eval_ready"] is False
    assert row["manifest_warnings"] == [
        "quantization_device_missing",
        "weight_quantization_backend_missing",
    ]
    assert row["metadata_complete_ready"] is False
    assert row["metadata_missing"] == [
        "manifest.activation_eps_missing",
        "manifest.quantization_device_missing",
        "manifest.weight_quantization_backend_missing",
        "manifest.quantization_staging_mode_missing",
        "model_index.activation_eps_missing",
        "model_index.quantization_device_missing",
        "model_index.weight_quantization_backend_missing",
        "model_index.quantization_staging_mode_missing",
        "benchmark_summary.activation_eps_missing",
        "benchmark_summary.quantization_device_missing",
        "benchmark_summary.weight_quantization_backend_missing",
        "benchmark_summary.quantization_staging_mode_missing",
    ]
    assert row["remote_checksum_mismatches"] == []
    assert row["missing_required_metrics"] == _expected_missing_geneval_metrics()
    assert row["native_smoke_proof_ready"] is True
    assert row["native_smoke_missing_evidence"] == []


def test_audit_hf_artifact_repos_marks_complete_metadata_when_provenance_matches(
    tmp_path,
    monkeypatch,
):
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
    repo_id = "WaveCut/FLUX.2-klein-4B-OrbitQuant-W4A4"
    siblings = _required_remote_files()
    siblings.update(
        {
            "model.safetensors": {"size": 123, "lfs_sha256": "a" * 64},
            "assets/image_generation_comparison_matrix.webp": 10,
        }
    )
    api = FakeAuditHfApi({repo_id: siblings})
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        """{
          "source_model_id": "black-forest-labs/FLUX.2-klein-4B",
          "weight_bits": 4,
          "activation_bits": 4,
          "target_policy": "flux2",
          "activation_eps": 1e-10,
          "quantization_device": "cuda",
          "weight_quantization_backend": "triton_cuda",
          "quantization_staging_mode": "component",
          "quantized_modules": ["block.attn.to_q"],
          "adaln_modules": [],
          "checksums": {
            "model.safetensors": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
          }
        }"""
    )
    sha256sums_path = tmp_path / "SHA256SUMS"
    sha256sums_path.write_text(
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa  model.safetensors\n"
    )
    model_index_path = tmp_path / "model_index.json"
    _write_remote_model_index(
        model_index_path,
        activation_eps=1e-10,
        quantization_device="cuda",
        weight_quantization_backend="triton_cuda",
        quantization_staging_mode="component",
    )
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        _native_smoke_summary(
            suite,
            activation_eps=1e-10,
            quantization_device="cuda",
            weight_quantization_backend="triton_cuda",
            quantization_staging_mode="component",
        ),
        encoding="utf-8",
    )
    file_map = _audit_file_map(
        repo_id,
        manifest_path=manifest_path,
        sha256sums_path=sha256sums_path,
        summary_path=summary_path,
        model_index_path=model_index_path,
    )

    def fake_download(repo, filename, **kwargs):
        return str(file_map[(repo, filename)])

    monkeypatch.setattr(hub_module, "hf_hub_download", fake_download)

    result = audit_hf_artifact_repos(suites=[suite], api=api)

    row = result["rows"][0]
    assert result["artifact_ready_count"] == 1
    assert result["metadata_complete_ready_count"] == 1
    assert result["metadata_missing_count"] == 0
    assert row["metadata_complete_ready"] is True
    assert row["metadata_missing"] == []


def test_audit_hf_artifact_repos_validates_policy_inventory_without_tensor_download(
    tmp_path,
    monkeypatch,
):
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
    repo_id = "WaveCut/FLUX.2-klein-4B-OrbitQuant-W4A4"
    siblings = _required_remote_files()
    siblings.update(
        {
            "model.safetensors": {"size": 123, "lfs_sha256": "a" * 64},
            "assets/image_generation_comparison_matrix.webp": 10,
        }
    )
    api = FakeAuditHfApi({repo_id: siblings})
    config = OrbitQuantConfig(target_policy="flux2")
    quantized_modules = ["block.attn.to_q"]
    adaln_modules = ["block.modulation"]
    skipped_modules = ["proj_out"]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "source_model_id": suite.model_id,
                "source_revision": "remote-sha",
                "source_license": "apache-2.0",
                "weight_bits": 4,
                "activation_bits": 4,
                "target_policy": "flux2",
                "quantized_modules": quantized_modules,
                "adaln_modules": adaln_modules,
                "skipped_modules": skipped_modules,
                "checksums": {
                    "model.safetensors": "a" * 64,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    quantization_config_path = tmp_path / "quantization_config.json"
    quantization_config_path.write_text(
        json.dumps(config.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    model_index_path = tmp_path / "model_index.json"
    model_index_path.write_text(
        json.dumps({"component": "transformer"}, indent=2) + "\n",
        encoding="utf-8",
    )
    sha256sums_path = tmp_path / "SHA256SUMS"
    sha256sums_path.write_text(f"{'a' * 64}  model.safetensors\n", encoding="utf-8")
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(_native_smoke_summary(suite), encoding="utf-8")
    inventory_root = tmp_path / "inventories"
    inventory_root.mkdir()
    (inventory_root / "flux2-native-policy.json").write_text(
        json.dumps(
            {
                "source_model_id": suite.model_id,
                "target_policy": "flux2",
                "component": "transformer",
                "load_mode": "config",
                "linear_module_count": 3,
                "action_counts": {
                    "orbitquant": 1,
                    "adaln_int4_rtn": 1,
                    "bf16_skip": 1,
                },
                "quantized_modules": quantized_modules,
                "adaln_modules": adaln_modules,
                "skipped_modules": skipped_modules,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    file_map = _audit_file_map(
        repo_id,
        manifest_path=manifest_path,
        sha256sums_path=sha256sums_path,
        summary_path=summary_path,
        model_index_path=model_index_path,
        quantization_config_path=quantization_config_path,
    )
    downloaded_files = []

    def fake_download(repo, filename, **kwargs):
        downloaded_files.append(filename)
        if filename == "model.safetensors":
            raise AssertionError("metadata-only audit must not download tensors")
        return str(file_map[(repo, filename)])

    monkeypatch.setattr(hub_module, "hf_hub_download", fake_download)

    result = audit_hf_artifact_repos(
        suites=[suite],
        api=api,
        policy_inventory_root=inventory_root,
    )

    row = result["rows"][0]
    assert result["policy_inventory_ready_count"] == 1
    assert result["policy_inventory_error_count"] == 0
    assert result["artifact_ready_count"] == 1
    assert row["policy_inventory_ready"] is True
    assert row["policy_inventory_error"] is None
    assert row["policy_inventory_validation"]["inventory_path"] == str(
        inventory_root / "flux2-native-policy.json"
    )
    assert "quantization_config.json" in downloaded_files
    assert "model.safetensors" not in downloaded_files


def test_audit_hf_artifact_repos_requires_native_smoke_proof_block(
    tmp_path,
    monkeypatch,
):
    suite = NativeSuite(
        name="flux1-schnell-native",
        model_id="black-forest-labs/FLUX.1-schnell",
        pipeline="FluxPipeline",
        width=1024,
        height=1024,
        steps=4,
        guidance=0.0,
        bit_settings=["W4A4"],
        metric="geneval",
    )
    repo_id = "WaveCut/FLUX.1-schnell-OrbitQuant-W4A4"
    siblings = _required_remote_files()
    siblings.update(
        {
            "model.safetensors": {"size": 123, "lfs_sha256": "a" * 64},
            "assets/image_generation_comparison_matrix.webp": 10,
        }
    )
    api = FakeAuditHfApi({repo_id: siblings})
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        """{
          "source_model_id": "black-forest-labs/FLUX.1-schnell",
          "weight_bits": 4,
          "activation_bits": 4,
          "target_policy": "flux",
          "quantized_modules": ["block.attn.to_q"],
          "adaln_modules": ["block.modulation"],
          "checksums": {
            "model.safetensors": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
          }
        }"""
    )
    sha256sums_path = tmp_path / "SHA256SUMS"
    sha256sums_path.write_text(
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa  model.safetensors\n"
    )
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        """{
          "metrics": {
            "original": {"records": 1, "latest_metrics": {"generated_samples": 1}},
            "orbitquant": {"records": 1, "latest_metrics": {"generated_samples": 1}}
          }
        }"""
    )
    model_index_path = tmp_path / "model_index.json"
    _write_remote_model_index(model_index_path)
    file_map = _audit_file_map(
        repo_id,
        manifest_path=manifest_path,
        sha256sums_path=sha256sums_path,
        summary_path=summary_path,
        model_index_path=model_index_path,
    )

    def fake_download(repo, filename, **kwargs):
        return str(file_map[(repo, filename)])

    monkeypatch.setattr(hub_module, "hf_hub_download", fake_download)

    result = audit_hf_artifact_repos(suites=[suite], api=api)

    row = result["rows"][0]
    assert result["artifact_ready_count"] == 1
    assert result["native_smoke_ready_count"] == 0
    assert result["release_eval_ready_count"] == 0
    assert row["artifact_ready"] is True
    assert row["native_smoke_ready"] is False
    assert row["native_smoke_proof_ready"] is False
    assert row["native_smoke_missing_evidence"] == ["native_smoke_missing"]


def test_audit_hf_artifact_repos_requires_geneval_per_task_metrics(
    tmp_path,
    monkeypatch,
):
    suite = NativeSuite(
        name="z-image-native",
        model_id="Tongyi-MAI/Z-Image-Turbo",
        pipeline="ZImagePipeline",
        width=1024,
        height=1024,
        steps=10,
        guidance=0.0,
        bit_settings=["W4A4"],
        metric="geneval",
    )
    repo_id = "WaveCut/Z-Image-Turbo-OrbitQuant-W4A4"
    siblings = _required_remote_files()
    siblings.update(
        {
            "model.safetensors": {"size": 123, "lfs_sha256": "a" * 64},
            "assets/image_generation_comparison_matrix.webp": 10,
        }
    )
    api = FakeAuditHfApi({repo_id: siblings})
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        """{
          "source_model_id": "Tongyi-MAI/Z-Image-Turbo",
          "weight_bits": 4,
          "activation_bits": 4,
          "target_policy": "z_image",
          "quantized_modules": ["block.attn.to_q"],
          "adaln_modules": ["block.modulation"],
          "checksums": {
            "model.safetensors": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
          }
        }"""
    )
    sha256sums_path = tmp_path / "SHA256SUMS"
    sha256sums_path.write_text(
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa  model.safetensors\n"
    )
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        _native_smoke_summary(suite, release_metrics={"geneval_overall": 0.71}),
        encoding="utf-8",
    )
    model_index_path = tmp_path / "model_index.json"
    _write_remote_model_index(model_index_path)
    file_map = _audit_file_map(
        repo_id,
        manifest_path=manifest_path,
        sha256sums_path=sha256sums_path,
        summary_path=summary_path,
        model_index_path=model_index_path,
    )

    def fake_download(repo, filename, **kwargs):
        return str(file_map[(repo, filename)])

    monkeypatch.setattr(hub_module, "hf_hub_download", fake_download)

    result = audit_hf_artifact_repos(suites=[suite], api=api)

    row = result["rows"][0]
    assert result["artifact_ready_count"] == 1
    assert result["native_smoke_ready_count"] == 1
    assert result["release_eval_ready_count"] == 0
    assert result["missing_required_metric_count"] == 12
    assert row["release_eval_ready"] is False
    assert row["missing_required_metrics"] == [
        {"split": split, "metric": metric}
        for metric in hub_module._GENEVAL_REQUIRED_METRICS
        if metric != "geneval_overall"
        for split in ("original", "orbitquant")
    ]


def test_audit_hf_artifact_repos_flags_extra_comparison_matrix_assets(
    tmp_path,
    monkeypatch,
):
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
    repo_id = "WaveCut/FLUX.2-klein-4B-OrbitQuant-W4A4"
    siblings = _required_remote_files()
    siblings.update(
        {
            "model.safetensors": {"size": 123, "lfs_sha256": "a" * 64},
            "assets/image_generation_comparison_matrix.webp": 10,
            "assets/video_generation_comparison_matrix.webp": 10,
        }
    )
    api = FakeAuditHfApi({repo_id: siblings})
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        """{
          "source_model_id": "black-forest-labs/FLUX.2-klein-4B",
          "weight_bits": 4,
          "activation_bits": 4,
          "target_policy": "flux2",
          "quantized_modules": ["block.attn.to_q"],
          "adaln_modules": ["block.modulation"],
          "checksums": {
            "model.safetensors": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
          }
        }"""
    )
    sha256sums_path = tmp_path / "SHA256SUMS"
    sha256sums_path.write_text(
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa  model.safetensors\n"
    )
    summary_path = tmp_path / "summary.json"
    summary_path.write_text('{"metrics": {}}\n')
    model_index_path = tmp_path / "model_index.json"
    _write_remote_model_index(model_index_path)
    file_map = _audit_file_map(
        repo_id,
        manifest_path=manifest_path,
        sha256sums_path=sha256sums_path,
        summary_path=summary_path,
        model_index_path=model_index_path,
    )

    def fake_download(repo, filename, **kwargs):
        return str(file_map[(repo, filename)])

    monkeypatch.setattr(hub_module, "hf_hub_download", fake_download)

    result = audit_hf_artifact_repos(suites=[suite], api=api)

    assert result["artifact_ready_count"] == 0
    assert result["forbidden_file_count"] == 1
    row = result["rows"][0]
    assert row["artifact_ready"] is False
    assert row["asset_count"] == 2
    assert row["forbidden_files"] == ["assets/video_generation_comparison_matrix.webp"]


def test_audit_hf_artifact_repos_rejects_lfs_checksum_mismatch(tmp_path, monkeypatch):
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
    repo_id = "WaveCut/FLUX.2-klein-4B-OrbitQuant-W4A4"
    siblings = _required_remote_files()
    siblings["model.safetensors"] = {"size": 123, "lfs_sha256": "b" * 64}
    api = FakeAuditHfApi({repo_id: siblings})
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        """{
          "source_model_id": "black-forest-labs/FLUX.2-klein-4B",
          "weight_bits": 4,
          "activation_bits": 4,
          "target_policy": "flux2",
          "quantized_modules": ["block.attn.to_q"],
          "adaln_modules": [],
          "checksums": {
            "model.safetensors": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
          }
        }"""
    )
    sha256sums_path = tmp_path / "SHA256SUMS"
    sha256sums_path.write_text(
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa  model.safetensors\n"
    )
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(_native_smoke_summary(suite), encoding="utf-8")
    model_index_path = tmp_path / "model_index.json"
    _write_remote_model_index(model_index_path)
    file_map = _audit_file_map(
        repo_id,
        manifest_path=manifest_path,
        sha256sums_path=sha256sums_path,
        summary_path=summary_path,
        model_index_path=model_index_path,
    )

    def fake_download(repo, filename, **kwargs):
        return str(file_map[(repo, filename)])

    monkeypatch.setattr(hub_module, "hf_hub_download", fake_download)

    result = audit_hf_artifact_repos(suites=[suite], api=api)

    row = result["rows"][0]
    assert result["artifact_ready_count"] == 0
    assert result["remote_checksum_mismatch_count"] == 2
    assert result["release_eval_applicable_count"] == 0
    assert row["release_eval_applicable"] is False
    assert row["artifact_ready"] is False
    assert row["remote_checksum_mismatches"] == [
        (
            "manifest/LFS mismatch for model.safetensors: expected "
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa, got "
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        ),
        (
            "SHA256SUMS/LFS mismatch for model.safetensors: expected "
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa, got "
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        ),
    ]


def test_render_hf_artifact_audit_markdown_reports_checksum_mismatch_count():
    markdown = render_hf_artifact_audit_markdown(
        {
            "namespace": "WaveCut",
            "repo_count": 1,
            "existing_count": 1,
            "artifact_ready_count": 0,
            "native_smoke_ready_count": 0,
            "release_eval_applicable_count": 1,
            "release_eval_not_applicable_count": 0,
            "release_eval_ready_count": 0,
            "missing_required_metric_count": 0,
            "manifest_warning_count": 0,
            "remote_checksum_mismatch_count": 2,
            "rows": [],
        }
    )

    assert "- Remote checksum mismatches: 2" in markdown


def test_audit_hf_artifact_repos_marks_visual_only_extra_target_not_applicable(
    tmp_path, monkeypatch
):
    suite = NativeSuite(
        name="flux2-native",
        model_id="black-forest-labs/FLUX.2-klein-4B",
        pipeline="Flux2KleinPipeline",
        width=1024,
        height=1024,
        steps=4,
        guidance=1.0,
        bit_settings=["W4A4"],
        metric="visual+optional-geneval",
    )
    repo_id = "WaveCut/FLUX.2-klein-4B-OrbitQuant-W4A4"
    siblings = _required_remote_files()
    siblings.update(
        {
            "model.safetensors": {"size": 123, "lfs_sha256": "a" * 64},
            "assets/image_generation_comparison_matrix.webp": 10,
        }
    )
    api = FakeAuditHfApi({repo_id: siblings})
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        """{
          "source_model_id": "black-forest-labs/FLUX.2-klein-4B",
          "weight_bits": 4,
          "activation_bits": 4,
          "target_policy": "flux2",
          "quantized_modules": ["block.attn.to_q"],
          "adaln_modules": [],
          "checksums": {
            "model.safetensors": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
          }
        }"""
    )
    sha256sums_path = tmp_path / "SHA256SUMS"
    sha256sums_path.write_text(
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa  model.safetensors\n"
    )
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(_native_smoke_summary(suite), encoding="utf-8")
    model_index_path = tmp_path / "model_index.json"
    _write_remote_model_index(model_index_path)
    file_map = _audit_file_map(
        repo_id,
        manifest_path=manifest_path,
        sha256sums_path=sha256sums_path,
        summary_path=summary_path,
        model_index_path=model_index_path,
    )

    def fake_download(repo, filename, **kwargs):
        return str(file_map[(repo, filename)])

    monkeypatch.setattr(hub_module, "hf_hub_download", fake_download)

    result = audit_hf_artifact_repos(suites=[suite], api=api)

    row = result["rows"][0]
    assert result["artifact_ready_count"] == 1
    assert result["native_smoke_ready_count"] == 1
    assert result["release_eval_applicable_count"] == 0
    assert result["release_eval_not_applicable_count"] == 1
    assert result["release_eval_ready_count"] == 0
    assert result["forbidden_file_count"] == 0
    assert row["release_eval_applicable"] is False
    assert row["release_eval_ready"] is False
    assert row["missing_required_metrics"] == []


def test_audit_hf_artifact_repos_flags_forbidden_remote_assets(tmp_path, monkeypatch):
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
    repo_id = "WaveCut/FLUX.2-klein-4B-OrbitQuant-W4A4"
    siblings = _required_remote_files()
    siblings.update(
        {
            "model.safetensors": {"size": 123, "lfs_sha256": "a" * 64},
            "assets/image_generation_comparison_matrix.webp": 10,
            "assets/flux2-native_seed0_W4A4_simple-object.png": 10,
        }
    )
    api = FakeAuditHfApi({repo_id: siblings})
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        """{
          "source_model_id": "black-forest-labs/FLUX.2-klein-4B",
          "weight_bits": 4,
          "activation_bits": 4,
          "target_policy": "flux2",
          "quantized_modules": ["block.attn.to_q"],
          "adaln_modules": [],
          "checksums": {
            "model.safetensors": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
          }
        }"""
    )
    sha256sums_path = tmp_path / "SHA256SUMS"
    sha256sums_path.write_text(
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa  model.safetensors\n"
    )
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        """{
          "metrics": {
            "original": {"records": 1, "latest_metrics": {"generated_samples": 1}},
            "orbitquant": {"records": 1, "latest_metrics": {"generated_samples": 1}}
          }
        }"""
    )
    model_index_path = tmp_path / "model_index.json"
    _write_remote_model_index(model_index_path)
    file_map = _audit_file_map(
        repo_id,
        manifest_path=manifest_path,
        sha256sums_path=sha256sums_path,
        summary_path=summary_path,
        model_index_path=model_index_path,
    )

    def fake_download(repo, filename, **kwargs):
        return str(file_map[(repo, filename)])

    monkeypatch.setattr(hub_module, "hf_hub_download", fake_download)

    result = audit_hf_artifact_repos(suites=[suite], api=api)
    row = result["rows"][0]

    assert result["forbidden_file_count"] == 1
    assert result["artifact_ready_count"] == 0
    assert row["artifact_ready"] is False
    assert row["forbidden_files"] == ["assets/flux2-native_seed0_W4A4_simple-object.png"]


def test_audit_hf_artifact_repos_reports_missing_repo_without_downloading():
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
    api = FakeAuditHfApi({})

    result = audit_hf_artifact_repos(suites=[suite], api=api)

    assert result["repo_count"] == 1
    assert result["existing_count"] == 0
    assert result["rows"][0]["exists"] is False
    assert "missing repo" in result["rows"][0]["error"]


def test_render_hf_artifact_audit_markdown_summarizes_ready_and_metric_gaps():
    payload = {
        "namespace": "WaveCut",
        "repo_count": 2,
        "existing_count": 2,
        "artifact_ready_count": 2,
        "metadata_complete_ready_count": 1,
        "native_smoke_ready_count": 2,
        "release_eval_applicable_count": 1,
        "release_eval_not_applicable_count": 1,
        "release_eval_ready_count": 0,
        "missing_required_metric_count": 2,
        "manifest_warning_count": 0,
        "metadata_missing_count": 3,
        "rows": [
            {
                "suite": "flux2-native",
                "bit_setting": "W4A4",
                "repo_id": "WaveCut/FLUX.2-klein-4B-OrbitQuant-W4A4",
                "private": True,
                "artifact_ready": True,
                "metadata_complete_ready": True,
                "native_smoke_ready": True,
                "release_eval_applicable": False,
                "release_eval_ready": False,
                "sha": "abcdef1234567890",
                "missing_required_metrics": [],
            },
            {
                "suite": "flux1-schnell-native",
                "bit_setting": "W4A4",
                "repo_id": "WaveCut/FLUX.1-schnell-OrbitQuant-W4A4",
                "private": True,
                "artifact_ready": True,
                "metadata_complete_ready": False,
                "native_smoke_ready": True,
                "release_eval_applicable": True,
                "release_eval_ready": False,
                "sha": "123456abcdef7890",
                "missing_required_metrics": [
                    {"split": "original", "metric": "geneval_overall"},
                    {"split": "orbitquant", "metric": "geneval_overall"},
                ],
            },
        ],
    }

    markdown = render_hf_artifact_audit_markdown(payload)

    assert "# OrbitQuant HF Artifact Audit" in markdown
    assert "- Metadata complete: 1 / 2" in markdown
    assert "- Policy inventory ready: not checked" in markdown
    assert "- Release eval applicable: 1 / 2" in markdown
    assert "- Release eval ready: 0 / 1" in markdown
    assert "- Missing release metrics: 2" in markdown
    assert "- Metadata missing fields: 3" in markdown
    assert "## Readiness Semantics" in markdown
    assert "no forbidden raw files" in markdown
    assert "it is not a GenEval or VBench result" in markdown
    assert "Metadata complete means activation normalization epsilon" in markdown
    assert "paper metric or reproduction claims" in markdown
    assert "Missing required metrics" not in markdown
    assert (
        "| flux2-native | W4A4 | `WaveCut/FLUX.2-klein-4B-OrbitQuant-W4A4` | "
        "yes | yes | yes | yes | n/a | n/a | abcdef123456 |  |  |"
    ) in markdown
    assert (
        "| flux1-schnell-native | W4A4 | `WaveCut/FLUX.1-schnell-OrbitQuant-W4A4` | "
        "yes | yes | no | yes | n/a | no | 123456abcdef | 2 release metrics missing |  |"
    ) in markdown
    assert (
        "`WaveCut/FLUX.1-schnell-OrbitQuant-W4A4`: "
        "orbitquant:geneval_overall, original:geneval_overall"
    ) in markdown
