import json

import pytest

import orbitquant.cli.main as cli_main
from orbitquant.cli.main import main


def test_cli_upload_artifact_wires_validation_and_hf_options(capsys, tmp_path, monkeypatch):
    seen = {}

    def fake_upload_artifact(artifact, **kwargs):
        seen["artifact"] = artifact
        seen["kwargs"] = kwargs
        return {
            "artifact_dir": artifact,
            "repo_id": kwargs["repo_id"],
            "private": kwargs["private"],
            "dry_run": kwargs["dry_run"],
            "validation": {"valid": True},
        }

    monkeypatch.setattr(cli_main, "upload_orbitquant_artifact", fake_upload_artifact)

    assert (
        main(
            [
                "upload-artifact",
                "--artifact",
                str(tmp_path),
                "--repo-id",
                "WaveCut/example-orbitquant",
                "--revision",
                "main",
                "--commit-message",
                "upload artifact",
                "--public",
                "--no-create-repo",
                "--replace-repo-files",
                "--skip-tensor-validation",
                "--upload-profile",
                "compact",
                "--report-dir",
                "/tmp/orbitquant-report",
                "--staging-dir",
                "/tmp/orbitquant-stage",
                "--dry-run",
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert output["repo_id"] == "WaveCut/example-orbitquant"
    assert output["private"] is False
    assert output["dry_run"] is True
    assert seen == {
        "artifact": str(tmp_path),
        "kwargs": {
            "repo_id": "WaveCut/example-orbitquant",
            "private": False,
            "create_repo": False,
            "revision": "main",
            "commit_message": "upload artifact",
            "replace_repo_files": True,
            "validate_tensors": False,
            "upload_profile": "compact",
            "report_dirs": ["/tmp/orbitquant-report"],
            "staging_dir": "/tmp/orbitquant-stage",
            "dry_run": True,
        },
    }


def test_cli_upload_artifact_defaults_to_compact_profile(capsys, tmp_path, monkeypatch):
    seen = {}

    def fake_upload_artifact(artifact, **kwargs):
        seen["artifact"] = artifact
        seen["kwargs"] = kwargs
        return {
            "artifact_dir": artifact,
            "repo_id": kwargs["repo_id"],
            "private": kwargs["private"],
            "dry_run": kwargs["dry_run"],
            "validation": {"valid": True},
        }

    monkeypatch.setattr(cli_main, "upload_orbitquant_artifact", fake_upload_artifact)

    assert (
        main(
            [
                "upload-artifact",
                "--artifact",
                str(tmp_path),
                "--repo-id",
                "WaveCut/example-orbitquant",
                "--dry-run",
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert output["repo_id"] == "WaveCut/example-orbitquant"
    assert seen["artifact"] == str(tmp_path)
    assert seen["kwargs"]["upload_profile"] == "compact"
    assert seen["kwargs"]["replace_repo_files"] is True
    assert seen["kwargs"]["report_dirs"] is None
    assert seen["kwargs"]["staging_dir"] is None


def test_cli_upload_artifact_rejects_disabling_remote_file_replacement(
    capsys, tmp_path, monkeypatch
):
    seen = {}

    def fake_upload_artifact(artifact, **kwargs):
        seen["artifact"] = artifact
        seen["kwargs"] = kwargs
        return {
            "artifact_dir": artifact,
            "repo_id": kwargs["repo_id"],
            "private": kwargs["private"],
            "dry_run": kwargs["dry_run"],
            "validation": {"valid": True},
        }

    monkeypatch.setattr(cli_main, "upload_orbitquant_artifact", fake_upload_artifact)

    with pytest.raises(SystemExit):
        main(
            [
                "upload-artifact",
                "--artifact",
                str(tmp_path),
                "--repo-id",
                "WaveCut/example-orbitquant",
                "--no-replace-repo-files",
                "--dry-run",
            ]
        )

    assert seen == {}
    assert "unrecognized arguments: --no-replace-repo-files" in capsys.readouterr().err


def test_cli_repair_hf_artifact_metadata_wires_single_repo_options(capsys, monkeypatch):
    seen = {}

    def fake_repair_hf_artifact_metadata(**kwargs):
        seen.update(kwargs)
        return {
            "repo_id": kwargs["repo_id"],
            "dry_run": kwargs["dry_run"],
            "changed_files": ["orbitquant_manifest.json"],
        }

    monkeypatch.setattr(cli_main, "repair_hf_artifact_metadata", fake_repair_hf_artifact_metadata)

    assert (
        main(
            [
                "repair-hf-artifact-metadata",
                "--repo-id",
                "WaveCut/example-orbitquant",
                "--revision",
                "main",
                "--commit-message",
                "repair metadata",
                "--quantization-device",
                "cuda",
                "--weight-quantization-backend",
                "triton_cuda",
                "--quantization-staging-mode",
                "component",
                "--dry-run",
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert output["repo_id"] == "WaveCut/example-orbitquant"
    assert output["dry_run"] is True
    assert seen == {
        "repo_id": "WaveCut/example-orbitquant",
        "quantization_device": "cuda",
        "weight_quantization_backend": "triton_cuda",
        "quantization_staging_mode": "component",
        "revision": "main",
        "commit_message": "repair metadata",
        "dry_run": True,
    }


def test_cli_repair_hf_native_smoke_proof_wires_single_repo_options(capsys, monkeypatch):
    seen = {}

    def fake_repair_hf_native_smoke_proof(**kwargs):
        seen.update(kwargs)
        return {
            "repo_id": kwargs["repo_id"],
            "suite": kwargs["suite"].name,
            "dry_run": kwargs["dry_run"],
            "changed_files": ["benchmark/summary.json"],
        }

    monkeypatch.setattr(cli_main, "repair_hf_native_smoke_proof", fake_repair_hf_native_smoke_proof)

    assert (
        main(
            [
                "repair-hf-native-smoke-proof",
                "--repo-id",
                "WaveCut/example-orbitquant",
                "--suite",
                "flux2-native",
                "--revision",
                "main",
                "--commit-message",
                "repair native smoke proof",
                "--dry-run",
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert output["repo_id"] == "WaveCut/example-orbitquant"
    assert output["suite"] == "flux2-native"
    assert output["dry_run"] is True
    assert seen["repo_id"] == "WaveCut/example-orbitquant"
    assert seen["suite"].name == "flux2-native"
    assert seen["revision"] == "main"
    assert seen["commit_message"] == "repair native smoke proof"
    assert seen["dry_run"] is True


def test_cli_cleanup_hf_artifact_reports_wires_single_repo_options(capsys, monkeypatch):
    seen = {}

    def fake_cleanup_hf_artifact_reports(**kwargs):
        seen.update(kwargs)
        return {
            "repo_id": kwargs["repo_id"],
            "revision": kwargs["revision"],
            "dry_run": kwargs["dry_run"],
            "report_file_count": 2,
            "promoted_assets": ["assets/image_generation_comparison_matrix.webp"],
            "delete_paths": ["reports"],
        }

    monkeypatch.setattr(cli_main, "cleanup_hf_artifact_reports", fake_cleanup_hf_artifact_reports)

    assert (
        main(
            [
                "cleanup-hf-artifact-reports",
                "--repo-id",
                "WaveCut/example-orbitquant",
                "--revision",
                "main",
                "--commit-message",
                "cleanup reports",
                "--dry-run",
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert output["repo_id"] == "WaveCut/example-orbitquant"
    assert output["promoted_assets"] == ["assets/image_generation_comparison_matrix.webp"]
    assert output["delete_paths"] == ["reports"]
    assert seen == {
        "repo_id": "WaveCut/example-orbitquant",
        "revision": "main",
        "commit_message": "cleanup reports",
        "dry_run": True,
    }


def test_cli_audit_hf_artifacts_writes_json_report(capsys, tmp_path, monkeypatch):
    seen = {}

    def fake_audit_hf_artifacts(*, namespace, suites, revision, policy_inventory_root):
        seen["namespace"] = namespace
        seen["suites"] = [suite.name for suite in suites]
        seen["revision"] = revision
        seen["policy_inventory_root"] = policy_inventory_root
        return {
            "namespace": namespace,
            "policy_inventory_root": policy_inventory_root,
            "repo_count": 1,
            "existing_count": 1,
            "artifact_ready_count": 1,
            "native_smoke_ready_count": 1,
            "release_eval_ready_count": 0,
            "rows": [{"repo_id": "WaveCut/example"}],
        }

    monkeypatch.setattr(cli_main, "audit_hf_artifact_repos", fake_audit_hf_artifacts)
    output_path = tmp_path / "reports" / "native" / "audit.json"
    markdown_output_path = tmp_path / "reports" / "native" / "audit.md"

    assert (
        main(
            [
                "audit-hf-artifacts",
                "--namespace",
                "WaveCut",
                "--suite",
                "flux2-native",
                "--revision",
                "main",
                "--policy-inventory-root",
                str(tmp_path / "inventories"),
                "--output",
                str(output_path),
                "--markdown-output",
                str(markdown_output_path),
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    written = json.loads(output_path.read_text())
    markdown = markdown_output_path.read_text()
    assert output_path.parent.is_dir()
    assert output == written
    assert output["repo_count"] == 1
    assert "# OrbitQuant HF Artifact Audit" in markdown
    assert "| `WaveCut/example` |" in markdown
    assert seen == {
        "namespace": "WaveCut",
        "suites": ["flux2-native"],
        "revision": "main",
        "policy_inventory_root": str(tmp_path / "inventories"),
    }


def test_cli_audit_hf_artifacts_summary_only_omits_full_rows(capsys, tmp_path, monkeypatch):
    def fake_audit_hf_artifacts(*, namespace, suites, revision, policy_inventory_root):
        return {
            "namespace": namespace,
            "policy_inventory_root": policy_inventory_root,
            "repo_count": 1,
            "existing_count": 1,
            "artifact_ready_count": 1,
            "native_smoke_ready_count": 1,
            "metadata_complete_ready_count": 1,
            "policy_inventory_ready_count": 1,
            "policy_inventory_error_count": 0,
            "release_eval_applicable_count": 1,
            "release_eval_ready_count": 0,
            "missing_required_metric_count": 2,
            "manifest_warning_count": 0,
            "metadata_missing_count": 0,
            "remote_checksum_mismatch_count": 0,
            "readme_mismatch_count": 0,
            "forbidden_file_count": 0,
            "rows": [
                {
                    "repo_id": "WaveCut/example",
                    "suite": "flux1-schnell-native",
                    "bit_setting": "W4A4",
                    "artifact_ready": True,
                    "native_smoke_ready": True,
                    "release_eval_applicable": True,
                    "release_eval_ready": False,
                    "metadata_complete_ready": True,
                    "policy_inventory_ready": True,
                    "forbidden_file_count": 0,
                    "missing_required_metrics": [
                        {"split": "original", "metric": "geneval_overall"},
                        {"split": "orbitquant", "metric": "geneval_overall"},
                    ],
                    "large_remote_row_detail": {"not": "printed"},
                }
            ],
        }

    monkeypatch.setattr(cli_main, "audit_hf_artifact_repos", fake_audit_hf_artifacts)
    output_path = tmp_path / "audit-summary.json"

    assert (
        main(
            [
                "audit-hf-artifacts",
                "--summary-only",
                "--output",
                str(output_path),
                "--fail-on-artifact-regression",
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    written = json.loads(output_path.read_text())
    assert output == written
    assert "rows" not in output
    assert output["repo_count"] == 1
    assert output["row_count"] == 1
    assert output["repos"] == [
        {
            "repo_id": "WaveCut/example",
            "suite": "flux1-schnell-native",
            "bit_setting": "W4A4",
            "artifact_ready": True,
            "native_smoke_ready": True,
            "release_eval_applicable": True,
            "release_eval_ready": False,
            "metadata_complete_ready": True,
            "policy_inventory_ready": True,
            "forbidden_file_count": 0,
            "missing_required_metric_count": 2,
        }
    ]
    assert "large_remote_row_detail" not in json.dumps(output)


def test_cli_audit_hf_artifacts_artifact_regression_gate_ignores_release_metrics(
    capsys, tmp_path, monkeypatch
):
    def fake_audit_hf_artifacts(*, namespace, suites, revision, policy_inventory_root):
        return {
            "namespace": namespace,
            "policy_inventory_root": policy_inventory_root,
            "repo_count": 1,
            "existing_count": 1,
            "artifact_ready_count": 1,
            "native_smoke_ready_count": 1,
            "metadata_complete_ready_count": 1,
            "policy_inventory_ready_count": 1,
            "policy_inventory_error_count": 0,
            "release_eval_ready_count": 0,
            "missing_required_metric_count": 14,
            "manifest_warning_count": 0,
            "metadata_missing_count": 0,
            "remote_checksum_mismatch_count": 0,
            "forbidden_file_count": 0,
            "rows": [{"repo_id": "WaveCut/example"}],
        }

    monkeypatch.setattr(cli_main, "audit_hf_artifact_repos", fake_audit_hf_artifacts)

    assert (
        main(
            [
                "audit-hf-artifacts",
                "--policy-inventory-root",
                str(tmp_path / "inventories"),
                "--fail-on-artifact-regression",
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert output["missing_required_metric_count"] == 14


def test_cli_audit_hf_artifacts_artifact_regression_gate_fails_on_remote_hygiene(
    capsys, tmp_path, monkeypatch
):
    def fake_audit_hf_artifacts(*, namespace, suites, revision, policy_inventory_root):
        return {
            "namespace": namespace,
            "policy_inventory_root": policy_inventory_root,
            "repo_count": 1,
            "existing_count": 1,
            "artifact_ready_count": 1,
            "native_smoke_ready_count": 1,
            "metadata_complete_ready_count": 1,
            "policy_inventory_ready_count": 1,
            "policy_inventory_error_count": 0,
            "release_eval_ready_count": 0,
            "missing_required_metric_count": 0,
            "manifest_warning_count": 0,
            "metadata_missing_count": 0,
            "remote_checksum_mismatch_count": 0,
            "forbidden_file_count": 1,
            "rows": [{"repo_id": "WaveCut/example"}],
        }

    monkeypatch.setattr(cli_main, "audit_hf_artifact_repos", fake_audit_hf_artifacts)

    assert (
        main(
            [
                "audit-hf-artifacts",
                "--policy-inventory-root",
                str(tmp_path / "inventories"),
                "--fail-on-artifact-regression",
            ]
        )
        == 1
    )

    captured = capsys.readouterr()
    assert json.loads(captured.out)["forbidden_file_count"] == 1
    assert "forbidden_file_count=1 expected 0" in captured.err


def test_cli_fetch_hf_artifacts_wires_suite_and_download_options(capsys, tmp_path, monkeypatch):
    seen = {}

    def fake_fetch_hf_artifacts(**kwargs):
        seen.update(kwargs)
        if kwargs["stage_logger"] is not None:
            kwargs["stage_logger"]("START", "example fetch")
        return {
            "namespace": kwargs["namespace"],
            "output_root": str(kwargs["output_root"]),
            "repo_count": 1,
            "downloaded_count": 0,
            "skipped_existing_count": 0,
            "dry_run": kwargs["dry_run"],
            "rows": [{"repo_id": "WaveCut/example"}],
        }

    monkeypatch.setattr(cli_main, "fetch_hf_artifacts", fake_fetch_hf_artifacts)

    assert (
        main(
            [
                "fetch-hf-artifacts",
                "--namespace",
                "WaveCut",
                "--suite",
                "flux1-schnell-native",
                "--output-root",
                str(tmp_path / "artifacts"),
                "--revision",
                "main",
                "--no-resume",
                "--force-download",
                "--local-files-only",
                "--validate-checksums",
                "--validate-tensors",
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert output["repo_count"] == 1
    assert "example fetch" in captured.err
    assert seen["namespace"] == "WaveCut"
    assert [suite.name for suite in seen["suites"]] == ["flux1-schnell-native"]
    assert seen["output_root"] == str(tmp_path / "artifacts")
    assert seen["revision"] == "main"
    assert seen["resume"] is False
    assert seen["force_download"] is True
    assert seen["local_files_only"] is True
    assert seen["validate_checksums"] is True
    assert seen["validate_tensors"] is True
    assert seen["dry_run"] is False
    assert seen["stage_logger"] is not None


def test_cli_fetch_hf_artifacts_dry_run_suppresses_stage_log(capsys, monkeypatch):
    seen = {}

    def fake_fetch_hf_artifacts(**kwargs):
        seen.update(kwargs)
        return {
            "namespace": kwargs["namespace"],
            "output_root": str(kwargs["output_root"]),
            "repo_count": 0,
            "downloaded_count": 0,
            "skipped_existing_count": 0,
            "dry_run": kwargs["dry_run"],
            "rows": [],
        }

    monkeypatch.setattr(cli_main, "fetch_hf_artifacts", fake_fetch_hf_artifacts)

    assert main(["fetch-hf-artifacts", "--dry-run"]) == 0

    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert output["dry_run"] is True
    assert captured.err == ""
    assert seen["dry_run"] is True
    assert seen["stage_logger"] is None
