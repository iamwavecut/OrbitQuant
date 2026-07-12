import shutil
from pathlib import Path

import pytest
from hub_helpers import _write_artifact

import orbitquant.hub as hub_module
from orbitquant.eval.native_settings import NativeSuite
from orbitquant.hub import fetch_hf_artifacts


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
