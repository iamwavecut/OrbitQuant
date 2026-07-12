import json

from hub_helpers import (
    FakeCleanupHfApi,
    FakeCommitHfApi,
    _legacy_compact_summary_without_native_smoke,
    _native_smoke_summary,
    _remote_file_map,
    _write_artifact,
    _write_native_smoke_backup,
    make_fake_download,
)
from huggingface_hub import CommitOperationAdd, CommitOperationDelete

import orbitquant.hub as hub_module
from orbitquant.artifacts import refresh_artifact_checksums
from orbitquant.artifacts.checksums import read_sha256sums, write_sha256sums
from orbitquant.eval.native_settings import NativeSuite
from orbitquant.hub import (
    cleanup_hf_artifact_reports,
    cleanup_hf_artifact_reports_matrix,
    repair_hf_artifact_metadata,
    repair_hf_artifact_metadata_matrix,
    repair_hf_native_smoke_proof,
    repair_hf_native_smoke_proof_matrix,
)


def test_repair_hf_artifact_metadata_dry_run_preserves_large_file_checksum(
    tmp_path,
    monkeypatch,
):
    repo_id = "WaveCut/example-orbitquant"
    _write_artifact(tmp_path)
    old_checksums = read_sha256sums(tmp_path / "SHA256SUMS")

    monkeypatch.setattr(
        hub_module, "hf_hub_download", make_fake_download(_remote_file_map(repo_id, tmp_path))
    )
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
        handle.write("0" * 64 + "  .cache/huggingface/download/README.md.metadata\n")
    old_checksums = read_sha256sums(tmp_path / "SHA256SUMS")

    monkeypatch.setattr(
        hub_module, "hf_hub_download", make_fake_download(_remote_file_map(repo_id, tmp_path))
    )
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
    assert b'"quantization_staging_mode": "component"' in manifest_operation.path_or_fileobj
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

    monkeypatch.setattr(hub_module, "hf_hub_download", make_fake_download(artifact_dir=tmp_path))
    fake_api = FakeCleanupHfApi(tmp_path)

    result = cleanup_hf_artifact_reports(
        repo_id=repo_id,
        commit_message="cleanup reports",
        api=fake_api,
    )

    assert result["report_file_count"] == 2
    assert result["report_matrix_count"] == 1
    assert result["forbidden_file_count"] == 4
    assert result["promoted_assets"] == ["assets/image_generation_comparison_matrix.webp"]
    assert result["delete_paths"] == [
        "assets/flux2-native_seed0_W4A4_simple-object.png",
        "assets/flux2-native_seed0_W4A4_simple-object.png.json",
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
        operation for operation in operations if isinstance(operation, CommitOperationDelete)
    ]
    assert "model.safetensors" not in added
    assert added["assets/image_generation_comparison_matrix.webp"] == b"report matrix"
    assert len(deleted) == 3
    deleted_paths = {operation.path_in_repo: operation.is_folder for operation in deleted}
    assert deleted_paths == {
        "assets/flux2-native_seed0_W4A4_simple-object.png": False,
        "assets/flux2-native_seed0_W4A4_simple-object.png.json": False,
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

    monkeypatch.setattr(hub_module, "hf_hub_download", make_fake_download(artifact_dir=tmp_path))
    fake_api = FakeCleanupHfApi(tmp_path)

    result = cleanup_hf_artifact_reports(
        repo_id=repo_id,
        commit_message="cleanup extra matrix",
        api=fake_api,
    )

    assert result["promoted_assets"] == []
    assert result["forbidden_files"] == [
        "assets/video_generation_comparison_matrix.webp",
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

    monkeypatch.setattr(hub_module, "hf_hub_download", make_fake_download(artifact_dir=tmp_path))
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

    monkeypatch.setattr(
        hub_module, "hf_hub_download", make_fake_download(_remote_file_map(repo_id, tmp_path))
    )
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

    monkeypatch.setattr(
        hub_module, "hf_hub_download", make_fake_download(_remote_file_map(repo_id, tmp_path))
    )
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


def test_repair_hf_native_smoke_proof_removes_recovered_pair_claim(
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
    summary = json.loads(_native_smoke_summary(suite))
    summary["native_smoke"]["proof_source"] = (
        "recovered_from_compact_summary_and_published_comparison_matrix"
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    refresh_artifact_checksums(tmp_path)

    monkeypatch.setattr(
        hub_module, "hf_hub_download", make_fake_download(_remote_file_map(repo_id, tmp_path))
    )
    fake_api = FakeCleanupHfApi(tmp_path)

    result = repair_hf_native_smoke_proof(
        repo_id=repo_id,
        suite=suite,
        revision="main",
        api=fake_api,
    )

    assert result["commit"]["commit_oid"] == "repair-sha"
    assert result["removed_invalid_native_smoke"] is True
    assert result["changed_files"] == [
        "benchmark/summary.json",
        "orbitquant_manifest.json",
        "README.md",
        "SHA256SUMS",
    ]
    operation_by_path = {
        operation.path_in_repo: operation.path_or_fileobj
        for operation in fake_api.create_commit_calls[0]["operations"]
    }
    repaired_summary = json.loads(operation_by_path["benchmark/summary.json"])
    repaired_readme = operation_by_path["README.md"].decode("utf-8")
    repaired_manifest = json.loads(operation_by_path["orbitquant_manifest.json"])
    sha_entries = hub_module._parse_sha256sums_bytes(operation_by_path["SHA256SUMS"])

    assert "native_smoke" not in repaired_summary
    assert "## Native Validation Proof" not in repaired_readme
    assert (
        repaired_manifest["checksums"]["benchmark/summary.json"]
        == sha_entries["benchmark/summary.json"]
    )
    assert sha_entries["README.md"] == hub_module._sha256_bytes(operation_by_path["README.md"])


def test_repair_hf_native_smoke_proof_restores_from_local_backup_without_raw_upload(
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
    backup_root = tmp_path / "native-backup"
    _write_native_smoke_backup(
        backup_root,
        repo_id=repo_id,
        suite=suite,
        bit_setting="W4A4",
    )

    monkeypatch.setattr(
        hub_module, "hf_hub_download", make_fake_download(_remote_file_map(repo_id, tmp_path))
    )
    fake_api = FakeCleanupHfApi(tmp_path)

    result = repair_hf_native_smoke_proof(
        repo_id=repo_id,
        suite=suite,
        native_smoke_backup_root=backup_root,
        revision="main",
        commit_message="restore native smoke proof",
        api=fake_api,
    )

    assert result["commit"]["commit_oid"] == "repair-sha"
    assert result["backup_skipped_reason"] is None
    assert result["proof_source"] == "local_paired_native_smoke_backup"
    assert result["changed_files"] == [
        "benchmark/summary.json",
        "orbitquant_manifest.json",
        "README.md",
        "SHA256SUMS",
    ]
    operation_by_path = {
        operation.path_in_repo: operation.path_or_fileobj
        for operation in fake_api.create_commit_calls[0]["operations"]
    }
    assert sorted(operation_by_path) == sorted(result["changed_files"])

    repaired_summary = json.loads(operation_by_path["benchmark/summary.json"])
    proof = repaired_summary["native_smoke"]
    assert proof["proof_source"] == "local_paired_native_smoke_backup"
    assert proof["comparison_asset_path"] == "assets/image_generation_comparison_matrix.webp"
    assert proof["paired_prompt_seed_count"] == 2
    assert proof["paired_prompt_seed_keys"] == [
        ["flux2-native", "0", "counting"],
        ["flux2-native", "0", "simple-object"],
    ]
    assert proof["splits"]["original"]["generated_samples"] == 2
    assert proof["splits"]["orbitquant"]["generated_samples"] == 2
    assert proof["splits"]["original"]["nonempty_output_count"] == 2
    assert proof["splits"]["orbitquant"]["nonempty_output_count"] == 2
    assert proof["splits"]["original"]["native_settings"] == [
        hub_module._native_smoke_expected_settings(suite)
    ]
    assert "## Native Validation Evidence" in operation_by_path["README.md"].decode("utf-8")


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

    monkeypatch.setattr(
        hub_module, "hf_hub_download", make_fake_download(_remote_file_map(repo_id, tmp_path))
    )
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
    assert "## Native Validation Evidence" in repaired_readme
    assert "| Paired prompt/seed count | `1` |" in repaired_readme


def test_repair_hf_native_smoke_proof_normalizes_legacy_video_export_fps(
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
        export_fps=16,
        steps=50,
        guidance=5.0,
        bit_settings=["W4A4"],
    )
    repo_id = "WaveCut/Wan2.1-T2V-1.3B-Diffusers-OrbitQuant-W4A4"
    _write_artifact(tmp_path)
    matrix = tmp_path / "assets" / "video_generation_comparison_matrix.webp"
    matrix.parent.mkdir(parents=True, exist_ok=True)
    matrix.write_bytes(b"matrix")
    summary_path = tmp_path / "benchmark" / "summary.json"
    summary = json.loads(_native_smoke_summary(suite))
    summary["native_smoke"]["comparison_asset_path"] = (
        "assets/video_generation_comparison_matrix.webp"
    )
    summary["native_smoke"]["proof_source"] = "local_paired_native_smoke_backup"
    for split in ("original", "orbitquant"):
        del summary["native_smoke"]["splits"][split]["native_settings"][0]["export_fps"]
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    refresh_artifact_checksums(tmp_path)

    monkeypatch.setattr(
        hub_module, "hf_hub_download", make_fake_download(_remote_file_map(repo_id, tmp_path))
    )
    fake_api = FakeCleanupHfApi(tmp_path)

    result = repair_hf_native_smoke_proof(
        repo_id=repo_id,
        suite=suite,
        revision="main",
        api=fake_api,
    )

    assert result["existing_native_smoke_ready"] is False
    assert result["normalized_native_smoke_settings"] is True
    assert set(result["changed_files"]) >= {
        "benchmark/summary.json",
        "orbitquant_manifest.json",
        "SHA256SUMS",
    }
    assert "model.safetensors" not in result["changed_files"]
    operation_by_path = {
        operation.path_in_repo: operation.path_or_fileobj
        for operation in fake_api.create_commit_calls[0]["operations"]
    }
    repaired_summary = json.loads(operation_by_path["benchmark/summary.json"])
    expected_settings = hub_module._native_smoke_expected_settings(suite)
    for split in ("original", "orbitquant"):
        assert repaired_summary["native_smoke"]["splits"][split]["native_settings"] == [
            expected_settings
        ]
    status = hub_module._native_smoke_proof_status(
        repaired_summary,
        suite=suite,
        file_names={"assets/video_generation_comparison_matrix.webp"},
    )
    assert status["ready"] is True


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

    monkeypatch.setattr(
        hub_module, "hf_hub_download", make_fake_download(_remote_file_map(repo_id, tmp_path))
    )
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

    monkeypatch.setattr(
        hub_module, "hf_hub_download", make_fake_download(_remote_file_map(repo_id, tmp_path))
    )
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


def test_repair_hf_native_smoke_proof_matrix_uses_local_backup(
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
    backup_root = tmp_path / "native-backup"
    _write_native_smoke_backup(
        backup_root,
        repo_id=repo_id,
        suite=suite,
        bit_setting="W4A4",
        prompt_ids=("simple-object",),
    )

    monkeypatch.setattr(
        hub_module, "hf_hub_download", make_fake_download(_remote_file_map(repo_id, tmp_path))
    )
    fake_api = FakeCleanupHfApi(tmp_path)

    result = repair_hf_native_smoke_proof_matrix(
        suites=[suite],
        native_smoke_backup_root=backup_root,
        dry_run=True,
        api=fake_api,
    )

    assert result["repo_count"] == 1
    assert result["changed_repo_count"] == 1
    assert result["skipped_repo_count"] == 0
    assert result["error_count"] == 0
    assert result["rows"][0]["repo_id"] == repo_id
    assert result["rows"][0]["proof_source"] == "local_paired_native_smoke_backup"
    assert result["rows"][0]["backup_skipped_reason"] is None
    assert result["rows"][0]["changed_files"] == [
        "benchmark/summary.json",
        "orbitquant_manifest.json",
        "README.md",
        "SHA256SUMS",
    ]
