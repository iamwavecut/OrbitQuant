from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi

from orbitquant.artifacts import validate_orbitquant_artifact
from orbitquant.eval import list_native_suites

_DEFAULT_IGNORE_PATTERNS = [
    ".DS_Store",
    "*/.DS_Store",
    "__pycache__/*",
    "*/__pycache__/*",
    ".pytest_cache/*",
]


def inspect_model_metadata(model_id: str, *, revision: str | None = None) -> dict[str, Any]:
    info = HfApi().model_info(model_id, revision=revision)
    card_data = getattr(info, "card_data", None)
    license_name = None
    if card_data is not None:
        if isinstance(card_data, dict):
            license_name = card_data.get("license")
        else:
            license_name = getattr(card_data, "license", None)

    native_suite = None
    for suite in list_native_suites():
        if suite.model_id == model_id:
            native_suite = asdict(suite)
            break

    return {
        "model_id": model_id,
        "sha": info.sha,
        "private": info.private,
        "gated": info.gated,
        "license": license_name,
        "tags": sorted(info.tags or []),
        "native_suite": native_suite,
    }


def _commit_info_payload(commit_info: Any) -> dict[str, Any]:
    return {
        "commit_oid": getattr(commit_info, "oid", None),
        "commit_url": str(getattr(commit_info, "commit_url", "")) or None,
        "pr_url": str(getattr(commit_info, "pr_url", "")) or None,
    }


def upload_orbitquant_artifact(
    artifact_dir: str | Path,
    *,
    repo_id: str,
    private: bool = True,
    create_repo: bool = True,
    revision: str | None = None,
    commit_message: str | None = None,
    replace_repo_files: bool = False,
    validate_tensors: bool = True,
    dry_run: bool = False,
    api: HfApi | None = None,
) -> dict[str, Any]:
    """Validate and upload an OrbitQuant artifact directory to a HF model repo."""

    artifact_path = Path(artifact_dir)
    validation = validate_orbitquant_artifact(
        artifact_path,
        validate_checksums_enabled=True,
        validate_tensors=validate_tensors,
    )
    upload_kwargs = {
        "repo_id": repo_id,
        "repo_type": "model",
        "folder_path": str(artifact_path),
        "revision": revision,
        "commit_message": commit_message or "Upload OrbitQuant artifact",
        "ignore_patterns": list(_DEFAULT_IGNORE_PATTERNS),
        "delete_patterns": "*" if replace_repo_files else None,
    }
    result: dict[str, Any] = {
        "artifact_dir": str(artifact_path),
        "repo_id": repo_id,
        "repo_type": "model",
        "private": private,
        "revision": revision,
        "create_repo": create_repo,
        "replace_repo_files": replace_repo_files,
        "dry_run": dry_run,
        "validation": validation,
        "upload": None,
        "uploaded_repo": None,
        "upload_kwargs": {
            key: value for key, value in upload_kwargs.items() if key != "folder_path"
        },
    }
    if dry_run:
        return result

    api = HfApi() if api is None else api
    if create_repo:
        repo_url = api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=private,
            exist_ok=True,
        )
        result["created_repo_url"] = str(repo_url)
    commit_info = api.upload_folder(**upload_kwargs)
    upload_payload = _commit_info_payload(commit_info)
    result["upload"] = upload_payload

    audit_revision = upload_payload["commit_oid"] or revision
    uploaded_info = api.model_info(repo_id, revision=audit_revision)
    result["uploaded_repo"] = {
        "repo_id": repo_id,
        "sha": uploaded_info.sha,
        "private": uploaded_info.private,
        "gated": uploaded_info.gated,
    }
    return result
