from types import SimpleNamespace

import pytest
import torch

import orbitquant.hub as hub_module
from orbitquant.artifacts import save_orbitquant_artifact
from orbitquant.artifacts.checksums import read_sha256sums
from orbitquant.config import OrbitQuantConfig
from orbitquant.eval.native_settings import NativeSuite
from orbitquant.hub import (
    audit_hf_artifact_repos,
    repair_hf_artifact_metadata,
    repair_hf_artifact_metadata_matrix,
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
        siblings = [
            SimpleNamespace(rfilename=name, size=size)
            for name, size in self.siblings_by_repo[repo_id].items()
        ]
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
        dry_run=True,
        api=fake_api,
    )

    assert result["dry_run"] is True
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
    old_checksums = read_sha256sums(tmp_path / "SHA256SUMS")

    def fake_download(repo, filename, **kwargs):
        return str(_remote_file_map(repo_id, tmp_path)[(repo, filename)])

    monkeypatch.setattr(hub_module, "hf_hub_download", fake_download)
    fake_api = FakeCommitHfApi()

    result = repair_hf_artifact_metadata(
        repo_id=repo_id,
        quantization_device="cuda",
        weight_quantization_backend="triton_cuda",
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
        dry_run=True,
        api=fake_api,
    )

    assert result["repo_count"] == 1
    assert result["error_count"] == 0
    assert result["rows"][0]["repo_id"] == repo_id
    assert result["rows"][0]["suite"] == "flux2-native"
    assert result["rows"][0]["bit_setting"] == "W4A4"


def test_upload_orbitquant_artifact_creates_uploads_and_audits_model_repo(tmp_path):
    _write_artifact(tmp_path)
    fake_api = FakeHfApi()

    result = upload_orbitquant_artifact(
        tmp_path,
        repo_id="WaveCut/example-orbitquant",
        private=False,
        revision="main",
        commit_message="upload test artifact",
        replace_repo_files=True,
        validate_tensors=False,
        api=fake_api,
    )

    assert fake_api.create_repo_calls == [
        {
            "repo_id": "WaveCut/example-orbitquant",
            "repo_type": "model",
            "private": False,
            "exist_ok": True,
        }
    ]
    assert len(fake_api.upload_folder_calls) == 1
    upload_call = fake_api.upload_folder_calls[0]
    assert upload_call["repo_id"] == "WaveCut/example-orbitquant"
    assert upload_call["repo_type"] == "model"
    assert upload_call["folder_path"] == str(tmp_path)
    assert upload_call["revision"] == "main"
    assert upload_call["commit_message"] == "upload test artifact"
    assert upload_call["delete_patterns"] == "*"
    assert result["upload"]["commit_oid"] == "uploaded-sha"
    assert result["upload"]["commit_url"].endswith("/commit/uploaded-sha")
    assert result["uploaded_repo"] == {
        "repo_id": "WaveCut/example-orbitquant",
        "sha": "uploaded-sha",
        "private": True,
        "gated": False,
    }
    assert fake_api.model_info_calls == [
        {"repo_id": "WaveCut/example-orbitquant", "revision": "uploaded-sha"}
    ]
    assert result["validation"]["tensor_validation"] == "skipped"


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
            "model.safetensors": 123,
            "assets/original.png": 10,
            "assets/orbitquant.png": 10,
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
          "adaln_modules": ["block.modulation"]
        }"""
    )
    original_metrics_path = tmp_path / "original.metrics.jsonl"
    original_metrics_path.write_text('{"metrics":{"generated_samples":1}}\n')
    orbitquant_metrics_path = tmp_path / "orbitquant.metrics.jsonl"
    orbitquant_metrics_path.write_text('{"metrics":{"generated_samples":1}}\n')
    file_map = {
        (repo_id, "orbitquant_manifest.json"): manifest_path,
        (repo_id, "benchmark/original.metrics.jsonl"): original_metrics_path,
        (repo_id, "benchmark/orbitquant.metrics.jsonl"): orbitquant_metrics_path,
    }

    def fake_download(repo, filename, **kwargs):
        return str(file_map[(repo, filename)])

    monkeypatch.setattr(hub_module, "hf_hub_download", fake_download)

    result = audit_hf_artifact_repos(suites=[suite], api=api)

    assert result["repo_count"] == 1
    assert result["existing_count"] == 1
    assert result["artifact_ready_count"] == 1
    assert result["native_smoke_ready_count"] == 1
    assert result["release_eval_ready_count"] == 0
    assert result["missing_required_metric_count"] == 2
    row = result["rows"][0]
    assert row["repo_id"] == repo_id
    assert row["artifact_ready"] is True
    assert row["native_smoke_ready"] is True
    assert row["release_eval_ready"] is False
    assert row["manifest_warnings"] == [
        "quantization_device_missing",
        "weight_quantization_backend_missing",
    ]
    assert row["missing_required_metrics"] == [
        {"split": "original", "metric": "geneval_overall"},
        {"split": "orbitquant", "metric": "geneval_overall"},
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
