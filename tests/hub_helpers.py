"""Shared Hugging Face fakes and artifact helpers for the test_hub_* modules."""

import json
from pathlib import Path
from types import SimpleNamespace

import torch

import orbitquant.hub as hub_module
from orbitquant.artifacts import save_orbitquant_artifact
from orbitquant.config import OrbitQuantConfig
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
                siblings.append(SimpleNamespace(rfilename=name, size=metadata.get("size"), lfs=lfs))
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


def make_fake_download(file_map=None, *, artifact_dir=None):
    """Build an ``hf_hub_download`` replacement for monkeypatching.

    Resolves ``(repo_id, filename)`` keys through ``file_map`` when given,
    otherwise resolves ``filename`` directly inside ``artifact_dir``.
    """
    if (file_map is None) == (artifact_dir is None):
        raise ValueError("provide exactly one of file_map or artifact_dir")

    def fake_download(repo, filename, **kwargs):
        if file_map is not None:
            return str(file_map[(repo, filename)])
        return str(_remote_path(artifact_dir, filename))

    return fake_download


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
    readme_path=None,
    quantization_config_path=None,
):
    if readme_path is None:
        readme_path = Path(manifest_path).parent / "README.md"
        manifest_payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        manifest_payload.setdefault("source_revision", "unknown")
        manifest_payload.setdefault("source_license", "unknown")
        manifest = hub_module.OrbitQuantManifest.from_dict(manifest_payload)
        benchmark_summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
        Path(readme_path).write_text(
            hub_module.render_model_card(manifest, benchmark_summary=benchmark_summary),
            encoding="utf-8",
        )
    file_map = {
        (repo_id, "orbitquant_manifest.json"): manifest_path,
        (repo_id, "model_index.json"): model_index_path,
        (repo_id, "SHA256SUMS"): sha256sums_path,
        (repo_id, "benchmark/summary.json"): summary_path,
        (repo_id, "README.md"): readme_path,
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
    native_settings = hub_module._native_smoke_expected_settings(suite)
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


def _write_native_smoke_backup(
    backup_root,
    *,
    repo_id,
    suite,
    bit_setting,
    prompt_ids=("simple-object", "counting"),
):
    assets = backup_root / repo_id.replace("/", "__") / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    extension = "mp4" if suite.frames is not None else "png"
    for prompt_id in prompt_ids:
        for split_token, quantization in (
            ("original", None),
            (
                bit_setting,
                {
                    "config": {
                        "weight_bits": int(bit_setting.split("A", maxsplit=1)[0][1:]),
                        "activation_bits": int(bit_setting.split("A", maxsplit=1)[1]),
                    }
                },
            ),
        ):
            media_path = assets / f"{suite.name}_seed0_{split_token}_{prompt_id}.{extension}"
            media_path.write_bytes(b"not empty")
            metadata = {
                "suite": suite.name,
                "model_id": suite.model_id,
                "prompt": f"prompt for {prompt_id}",
                "seed": 0,
                "height": suite.height,
                "width": suite.width,
                "frames": suite.frames,
                "steps": suite.steps,
                "guidance": suite.guidance,
                "quantization": quantization,
            }
            if suite.export_fps is not None:
                metadata["export_fps"] = suite.export_fps
            (media_path.with_suffix(media_path.suffix + ".json")).write_text(
                json.dumps(metadata, indent=2) + "\n",
                encoding="utf-8",
            )


def _expected_missing_geneval_metrics():
    return [
        {"split": split, "metric": metric}
        for metric in hub_module._GENEVAL_REQUIRED_METRICS
        for split in ("original", "orbitquant")
    ]
