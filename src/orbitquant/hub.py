from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

from orbitquant.artifacts import validate_orbitquant_artifact
from orbitquant.artifacts.checksums import is_ignored_artifact_relative_path
from orbitquant.artifacts.manifest import OrbitQuantManifest
from orbitquant.artifacts.model_card import render_model_card
from orbitquant.eval import list_native_suites
from orbitquant.eval.native_runner import target_policy_for_suite
from orbitquant.eval.native_settings import NativeSuite

_DEFAULT_IGNORE_PATTERNS = [
    ".DS_Store",
    "*/.DS_Store",
    ".gitattributes",
    "*/.gitattributes",
    ".gitignore",
    "*/.gitignore",
    ".cache/*",
    ".cache/**",
    "*/.cache/*",
    "*/.cache/**",
    "__pycache__/*",
    "*/__pycache__/*",
    ".pytest_cache/*",
]
_REQUIRED_ARTIFACT_FILES = (
    "README.md",
    "SHA256SUMS",
    "model_index.json",
    "model.safetensors",
    "quantization_config.json",
    "orbitquant_manifest.json",
    "orbitquant_codebooks.safetensors",
    "orbitquant_rotations.safetensors",
    "prompts.json",
    "benchmark/summary.json",
    "benchmark/original.metrics.jsonl",
    "benchmark/orbitquant.metrics.jsonl",
    "benchmark/original.metrics.csv",
    "benchmark/orbitquant.metrics.csv",
    "assets/.gitkeep",
)
_REQUIRED_METRICS_BY_SUITE = {
    "flux1-schnell-native": ("geneval_overall",),
    "z-image-native": ("geneval_overall",),
    "wan-native": (
        "vbench_imaging_quality",
        "vbench_aesthetic_quality",
        "vbench_motion_smoothness",
        "vbench_dynamic_degree",
        "vbench_background_consistency",
        "vbench_subject_consistency",
        "vbench_scene",
        "vbench_overall_consistency",
    ),
}


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


def default_artifact_repo_id(namespace: str, suite: NativeSuite, bit_setting: str) -> str:
    model_name = suite.model_id.rsplit("/", maxsplit=1)[-1]
    return f"{namespace}/{model_name}-OrbitQuant-{bit_setting.upper()}"


def _read_remote_json(repo_id: str, filename: str, *, revision: str | None = None) -> Any:
    path = hf_hub_download(repo_id, filename, repo_type="model", revision=revision)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_remote_bytes(repo_id: str, filename: str, *, revision: str | None = None) -> bytes:
    path = hf_hub_download(repo_id, filename, repo_type="model", revision=revision)
    return Path(path).read_bytes()


def _read_remote_jsonl(
    repo_id: str, filename: str, *, revision: str | None = None
) -> list[dict]:
    path = hf_hub_download(repo_id, filename, repo_type="model", revision=revision)
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _metrics_by_split(
    repo_id: str,
    file_names: set[str],
    *,
    revision: str | None,
) -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {"original": {}, "orbitquant": {}}
    for split in payload:
        filename = f"benchmark/{split}.metrics.jsonl"
        if filename not in file_names:
            continue
        for record in _read_remote_jsonl(repo_id, filename, revision=revision):
            payload[split].update(record.get("metrics", {}))
    return payload


def _manifest_warnings(manifest: dict[str, Any]) -> list[str]:
    warnings = []
    if not manifest:
        return ["manifest_missing"]
    if manifest.get("quantization_device") in (None, "", "unknown"):
        warnings.append("quantization_device_missing")
    if manifest.get("weight_quantization_backend") in (None, "", "unknown"):
        warnings.append("weight_quantization_backend_missing")
    return warnings


def _manifest_mismatches(
    manifest: dict[str, Any],
    *,
    suite: NativeSuite,
    bit_setting: str,
) -> list[str]:
    if not manifest:
        return ["manifest_missing"]
    weight_bits, activation_bits = bit_setting.upper().removeprefix("W").split("A", maxsplit=1)
    expected = {
        "source_model_id": suite.model_id,
        "weight_bits": int(weight_bits),
        "activation_bits": int(activation_bits),
        "target_policy": target_policy_for_suite(suite),
    }
    mismatches = []
    for key, value in expected.items():
        if manifest.get(key) != value:
            mismatches.append(f"{key}: expected {value!r}, got {manifest.get(key)!r}")
    return mismatches


def _missing_required_metrics(
    metrics: dict[str, dict[str, Any]], *, suite_name: str
) -> list[dict[str, str]]:
    missing = []
    for metric in _REQUIRED_METRICS_BY_SUITE.get(suite_name, ()):
        for split in ("original", "orbitquant"):
            if metric not in metrics.get(split, {}):
                missing.append({"split": split, "metric": metric})
    return missing


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2) + "\n").encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256sums_bytes(entries: dict[str, str]) -> bytes:
    filtered = {
        relative_path: digest
        for relative_path, digest in entries.items()
        if relative_path != "SHA256SUMS"
        and not is_ignored_artifact_relative_path(relative_path)
    }
    return (
        "\n".join(
            f"{digest}  {relative_path}" for relative_path, digest in sorted(filtered.items())
        )
        + ("\n" if filtered else "")
    ).encode("utf-8")


def _parse_sha256sums_bytes(payload: bytes) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in payload.decode("utf-8").splitlines():
        if not line.strip():
            continue
        digest, _, relative_path = line.partition("  ")
        if digest and relative_path:
            entries[relative_path] = digest
    return entries


def audit_hf_artifact_repos(
    *,
    namespace: str = "WaveCut",
    suites: list[NativeSuite] | None = None,
    revision: str | None = None,
    api: HfApi | None = None,
) -> dict[str, Any]:
    selected_suites = list_native_suites() if suites is None else suites
    api = HfApi() if api is None else api
    rows = []
    for suite in selected_suites:
        for bit_setting in suite.bit_settings:
            repo_id = default_artifact_repo_id(namespace, suite, bit_setting)
            row: dict[str, Any] = {
                "suite": suite.name,
                "bit_setting": bit_setting,
                "repo_id": repo_id,
                "exists": False,
                "artifact_ready": False,
                "native_smoke_ready": False,
                "release_eval_ready": False,
            }
            try:
                info = api.model_info(repo_id, revision=revision, files_metadata=True)
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {str(exc)}"
                rows.append(row)
                continue

            siblings = list(info.siblings or [])
            files = {sibling.rfilename: getattr(sibling, "size", None) for sibling in siblings}
            file_names = set(files)
            required_missing = [
                filename for filename in _REQUIRED_ARTIFACT_FILES if filename not in file_names
            ]
            manifest = {}
            manifest_error = None
            if "orbitquant_manifest.json" in file_names:
                try:
                    manifest = _read_remote_json(
                        repo_id, "orbitquant_manifest.json", revision=revision
                    )
                except Exception as exc:
                    manifest_error = f"{type(exc).__name__}: {str(exc)}"
            metrics = _metrics_by_split(repo_id, file_names, revision=revision)
            missing_metrics = _missing_required_metrics(metrics, suite_name=suite.name)
            manifest_mismatches = _manifest_mismatches(
                manifest,
                suite=suite,
                bit_setting=bit_setting,
            )
            generated_splits = sorted(
                split for split, values in metrics.items() if values.get("generated_samples")
            )
            row.update(
                {
                    "exists": True,
                    "sha": info.sha,
                    "private": info.private,
                    "gated": info.gated,
                    "file_count": len(file_names),
                    "model_size": files.get("model.safetensors"),
                    "asset_count": sum(
                        1
                        for filename in file_names
                        if filename.startswith("assets/") and filename != "assets/.gitkeep"
                    ),
                    "required_missing": required_missing,
                    "manifest_error": manifest_error,
                    "manifest_mismatches": manifest_mismatches,
                    "manifest_warnings": _manifest_warnings(manifest),
                    "quantized_modules": len(manifest.get("quantized_modules") or []),
                    "adaln_modules": len(manifest.get("adaln_modules") or []),
                    "metrics_by_split": metrics,
                    "generated_splits": generated_splits,
                    "missing_required_metrics": missing_metrics,
                }
            )
            row["artifact_ready"] = (
                not required_missing
                and files.get("model.safetensors") is not None
                and not manifest_error
                and not manifest_mismatches
            )
            row["native_smoke_ready"] = (
                row["artifact_ready"]
                and row["asset_count"] > 0
                and set(generated_splits) == {"original", "orbitquant"}
            )
            row["release_eval_ready"] = row["native_smoke_ready"] and not missing_metrics
            rows.append(row)
    return {
        "namespace": namespace,
        "repo_count": len(rows),
        "existing_count": sum(1 for row in rows if row["exists"]),
        "artifact_ready_count": sum(1 for row in rows if row["artifact_ready"]),
        "native_smoke_ready_count": sum(1 for row in rows if row["native_smoke_ready"]),
        "release_eval_ready_count": sum(1 for row in rows if row["release_eval_ready"]),
        "missing_required_metric_count": sum(
            len(row.get("missing_required_metrics", [])) for row in rows
        ),
        "manifest_warning_count": sum(len(row.get("manifest_warnings", [])) for row in rows),
        "rows": rows,
    }


def repair_hf_artifact_metadata(
    *,
    repo_id: str,
    quantization_device: str,
    weight_quantization_backend: str,
    revision: str | None = None,
    commit_message: str | None = None,
    dry_run: bool = False,
    api: HfApi | None = None,
) -> dict[str, Any]:
    """Repair remote metadata files without downloading or reuploading large tensors."""

    manifest_bytes = _read_remote_bytes(repo_id, "orbitquant_manifest.json", revision=revision)
    model_index_bytes = _read_remote_bytes(repo_id, "model_index.json", revision=revision)
    benchmark_bytes = _read_remote_bytes(repo_id, "benchmark/summary.json", revision=revision)
    config_bytes = _read_remote_bytes(repo_id, "quantization_config.json", revision=revision)
    readme_bytes = _read_remote_bytes(repo_id, "README.md", revision=revision)
    sha256sums_bytes = _read_remote_bytes(repo_id, "SHA256SUMS", revision=revision)

    manifest = OrbitQuantManifest.from_dict(json.loads(manifest_bytes.decode("utf-8")))
    model_index = json.loads(model_index_bytes.decode("utf-8"))
    benchmark_summary = json.loads(benchmark_bytes.decode("utf-8"))

    before = {
        "quantization_device": manifest.quantization_device,
        "weight_quantization_backend": manifest.weight_quantization_backend,
    }
    repaired_manifest = OrbitQuantManifest(
        source_model_id=manifest.source_model_id,
        source_revision=manifest.source_revision,
        source_license=manifest.source_license,
        weight_bits=manifest.weight_bits,
        activation_bits=manifest.activation_bits,
        rotation_seed=manifest.rotation_seed,
        block_size=manifest.block_size,
        block_size_policy=manifest.block_size_policy,
        codebook_version=manifest.codebook_version,
        target_policy=manifest.target_policy,
        runtime_mode=manifest.runtime_mode,
        activation_kernel_backend=manifest.activation_kernel_backend,
        quantization_device=quantization_device,
        weight_quantization_backend=weight_quantization_backend,
        quantized_modules=manifest.quantized_modules,
        adaln_modules=manifest.adaln_modules,
        skipped_modules=manifest.skipped_modules,
        module_shapes=manifest.module_shapes,
        checksums={
            relative_path: digest
            for relative_path, digest in manifest.checksums.items()
            if not is_ignored_artifact_relative_path(relative_path)
        },
    )

    model_index["quantization_device"] = quantization_device
    model_index["weight_quantization_backend"] = weight_quantization_backend
    benchmark_summary["quantization_device"] = quantization_device
    benchmark_summary["weight_quantization_backend"] = weight_quantization_backend

    next_model_index_bytes = _json_bytes(model_index)
    next_benchmark_bytes = _json_bytes(benchmark_summary)
    repaired_manifest.checksums.update(
        {
            "model_index.json": _sha256_bytes(next_model_index_bytes),
            "benchmark/summary.json": _sha256_bytes(next_benchmark_bytes),
            "quantization_config.json": _sha256_bytes(config_bytes),
        }
    )
    next_manifest_bytes = _json_bytes(repaired_manifest.to_dict())
    next_readme_bytes = render_model_card(repaired_manifest).encode("utf-8")

    sha_entries = _parse_sha256sums_bytes(sha256sums_bytes)
    sha_entries.update(
        {
            "orbitquant_manifest.json": _sha256_bytes(next_manifest_bytes),
            "model_index.json": _sha256_bytes(next_model_index_bytes),
            "benchmark/summary.json": _sha256_bytes(next_benchmark_bytes),
            "quantization_config.json": _sha256_bytes(config_bytes),
            "README.md": _sha256_bytes(next_readme_bytes),
        }
    )
    sha_entries.pop("SHA256SUMS", None)
    next_sha256sums_bytes = _sha256sums_bytes(sha_entries)

    file_payloads = {
        "orbitquant_manifest.json": next_manifest_bytes,
        "model_index.json": next_model_index_bytes,
        "benchmark/summary.json": next_benchmark_bytes,
        "README.md": next_readme_bytes,
        "SHA256SUMS": next_sha256sums_bytes,
    }
    original_payloads = {
        "orbitquant_manifest.json": manifest_bytes,
        "model_index.json": model_index_bytes,
        "benchmark/summary.json": benchmark_bytes,
        "README.md": readme_bytes,
        "SHA256SUMS": sha256sums_bytes,
    }
    changed_files = [
        filename
        for filename, payload in file_payloads.items()
        if original_payloads.get(filename) != payload
    ]
    result: dict[str, Any] = {
        "repo_id": repo_id,
        "repo_type": "model",
        "revision": revision,
        "dry_run": dry_run,
        "before": before,
        "updated": {
            "quantization_device": quantization_device,
            "weight_quantization_backend": weight_quantization_backend,
        },
        "changed_files": changed_files,
        "preserved_checksum_entries": sorted(
            relative_path for relative_path in sha_entries if relative_path not in file_payloads
        ),
        "commit": None,
    }
    if dry_run or not changed_files:
        return result

    api = HfApi() if api is None else api
    operations = [
        CommitOperationAdd(path_in_repo=filename, path_or_fileobj=file_payloads[filename])
        for filename in changed_files
    ]
    commit_info = api.create_commit(
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        operations=operations,
        commit_message=commit_message or "Repair OrbitQuant artifact metadata",
    )
    result["commit"] = _commit_info_payload(commit_info)
    return result


def repair_hf_artifact_metadata_matrix(
    *,
    namespace: str = "WaveCut",
    suites: list[NativeSuite] | None = None,
    quantization_device: str,
    weight_quantization_backend: str,
    revision: str | None = None,
    commit_message: str | None = None,
    dry_run: bool = False,
    api: HfApi | None = None,
) -> dict[str, Any]:
    selected_suites = list_native_suites() if suites is None else suites
    api = HfApi() if api is None else api
    rows = []
    for suite in selected_suites:
        for bit_setting in suite.bit_settings:
            repo_id = default_artifact_repo_id(namespace, suite, bit_setting)
            try:
                row = repair_hf_artifact_metadata(
                    repo_id=repo_id,
                    quantization_device=quantization_device,
                    weight_quantization_backend=weight_quantization_backend,
                    revision=revision,
                    commit_message=commit_message,
                    dry_run=dry_run,
                    api=api,
                )
            except Exception as exc:
                row = {
                    "repo_id": repo_id,
                    "suite": suite.name,
                    "bit_setting": bit_setting,
                    "dry_run": dry_run,
                    "error": f"{type(exc).__name__}: {str(exc)}",
                }
            else:
                row["suite"] = suite.name
                row["bit_setting"] = bit_setting
            rows.append(row)
    return {
        "namespace": namespace,
        "repo_count": len(rows),
        "dry_run": dry_run,
        "changed_repo_count": sum(1 for row in rows if row.get("changed_files")),
        "error_count": sum(1 for row in rows if row.get("error")),
        "rows": rows,
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
