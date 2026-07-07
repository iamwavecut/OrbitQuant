from types import SimpleNamespace

import pytest
import torch

import orbitquant.hub as hub_module
from orbitquant.artifacts import save_orbitquant_artifact
from orbitquant.config import OrbitQuantConfig
from orbitquant.eval.native_settings import NativeSuite
from orbitquant.hub import audit_hf_artifact_repos, upload_orbitquant_artifact
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
