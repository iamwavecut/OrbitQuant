from types import SimpleNamespace

import pytest
import torch

from orbitquant.artifacts import save_orbitquant_artifact
from orbitquant.config import OrbitQuantConfig
from orbitquant.hub import upload_orbitquant_artifact
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
