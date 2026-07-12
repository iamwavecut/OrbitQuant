import json

from hub_helpers import (
    FakeAuditHfApi,
    _audit_file_map,
    _expected_missing_geneval_metrics,
    _native_smoke_summary,
    _required_remote_files,
    _write_remote_model_index,
    make_fake_download,
)

import orbitquant.hub as hub_module
from orbitquant.config import OrbitQuantConfig
from orbitquant.eval.native_settings import NativeSuite
from orbitquant.hub import audit_hf_artifact_repos, render_hf_artifact_audit_markdown


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

    monkeypatch.setattr(hub_module, "hf_hub_download", make_fake_download(file_map))

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

    monkeypatch.setattr(hub_module, "hf_hub_download", make_fake_download(file_map))

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

    monkeypatch.setattr(hub_module, "hf_hub_download", make_fake_download(file_map))

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

    monkeypatch.setattr(hub_module, "hf_hub_download", make_fake_download(file_map))

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

    monkeypatch.setattr(hub_module, "hf_hub_download", make_fake_download(file_map))

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

    monkeypatch.setattr(hub_module, "hf_hub_download", make_fake_download(file_map))

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


def test_audit_hf_artifact_repos_flags_stale_remote_readme(tmp_path, monkeypatch):
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
    readme_path = tmp_path / "README.md"
    readme_path.write_text(
        "# Stale card\n\nRunPod logs and reports/native/debug output.\n",
        encoding="utf-8",
    )
    file_map = _audit_file_map(
        repo_id,
        manifest_path=manifest_path,
        sha256sums_path=sha256sums_path,
        summary_path=summary_path,
        model_index_path=model_index_path,
        readme_path=readme_path,
    )

    monkeypatch.setattr(hub_module, "hf_hub_download", make_fake_download(file_map))

    result = audit_hf_artifact_repos(suites=[suite], api=api)
    row = result["rows"][0]

    assert result["artifact_ready_count"] == 0
    assert result["readme_mismatch_count"] == 1
    assert row["artifact_ready"] is False
    assert row["readme_mismatches"] == ["README.md does not match generated OrbitQuant model card"]


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

    monkeypatch.setattr(hub_module, "hf_hub_download", make_fake_download(file_map))

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
            "assets/nested/debug_generation_comparison_matrix.webp": 10,
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

    monkeypatch.setattr(hub_module, "hf_hub_download", make_fake_download(file_map))

    result = audit_hf_artifact_repos(suites=[suite], api=api)
    row = result["rows"][0]

    assert result["forbidden_file_count"] == 2
    assert result["artifact_ready_count"] == 0
    assert row["artifact_ready"] is False
    assert row["forbidden_files"] == [
        "assets/flux2-native_seed0_W4A4_simple-object.png",
        "assets/nested/debug_generation_comparison_matrix.webp",
    ]


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
