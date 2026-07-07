from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from huggingface_hub import (
    CommitOperationAdd,
    CommitOperationDelete,
    HfApi,
    hf_hub_download,
    snapshot_download,
)

from orbitquant.artifacts import refresh_artifact_checksums, validate_orbitquant_artifact
from orbitquant.artifacts.checksums import (
    is_ignored_artifact_relative_path,
    sha256_file,
)
from orbitquant.artifacts.manifest import OrbitQuantManifest
from orbitquant.artifacts.model_card import render_model_card
from orbitquant.config import OrbitQuantConfig
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
)
_GENEVAL_REQUIRED_METRICS = (
    "geneval_overall",
    "geneval_per_task_single_object",
    "geneval_per_task_two_object",
    "geneval_per_task_counting",
    "geneval_per_task_colors",
    "geneval_per_task_position",
    "geneval_per_task_color_attr",
)
_REQUIRED_METRICS_BY_SUITE = {
    "flux1-schnell-native": _GENEVAL_REQUIRED_METRICS,
    "z-image-native": _GENEVAL_REQUIRED_METRICS,
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
_COMPACT_RAW_EVAL_ASSET_MARKERS = ("_geneval-", "_vbench-")
_COMPACT_REPORT_ROOT = "reports/"
_PUBLISHED_CORE_ARTIFACT_FILES = set(_REQUIRED_ARTIFACT_FILES)
_LOCAL_BENCHMARK_FILES = {
    "benchmark/original.metrics.jsonl",
    "benchmark/orbitquant.metrics.jsonl",
    "benchmark/original.metrics.csv",
    "benchmark/orbitquant.metrics.csv",
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


def _artifact_dir_name(suite: NativeSuite, bit_setting: str) -> str:
    return f"{suite.name}-{bit_setting.lower()}"


def fetch_hf_artifacts(
    *,
    namespace: str = "WaveCut",
    suites: list[NativeSuite] | None = None,
    output_root: str | Path = "artifacts/native",
    revision: str | None = None,
    resume: bool = True,
    force_download: bool = False,
    local_files_only: bool = False,
    validate_checksums: bool = False,
    validate_tensors: bool = False,
    dry_run: bool = False,
    stage_logger: Callable[[str, str], None] | None = None,
) -> dict[str, Any]:
    selected_suites = list_native_suites() if suites is None else suites
    output_path = Path(output_root)
    rows = []
    for suite in selected_suites:
        for bit_setting in suite.bit_settings:
            repo_id = default_artifact_repo_id(namespace, suite, bit_setting)
            artifact_dir = output_path / _artifact_dir_name(suite, bit_setting)
            label = f"{suite.name} {bit_setting} fetch {repo_id}"
            row: dict[str, Any] = {
                "suite": suite.name,
                "bit_setting": bit_setting,
                "repo_id": repo_id,
                "artifact_dir": str(artifact_dir),
                "revision": revision,
                "dry_run": dry_run,
                "downloaded": False,
                "skipped_existing": False,
            }
            if dry_run:
                rows.append(row)
                continue

            if resume and artifact_dir.is_dir():
                try:
                    validation = validate_orbitquant_artifact(
                        artifact_dir,
                        validate_checksums_enabled=validate_checksums,
                        validate_tensors=validate_tensors,
                    )
                except Exception as exc:
                    row["existing_validation_error"] = f"{type(exc).__name__}: {str(exc)}"
                else:
                    row["skipped_existing"] = True
                    row["validation"] = validation
                    rows.append(row)
                    continue

            if stage_logger is not None:
                stage_logger("START", label)
            try:
                snapshot_path = snapshot_download(
                    repo_id=repo_id,
                    repo_type="model",
                    revision=revision,
                    local_dir=artifact_dir,
                    force_download=force_download,
                    local_files_only=local_files_only,
                )
                validation = validate_orbitquant_artifact(
                    artifact_dir,
                    validate_checksums_enabled=validate_checksums,
                    validate_tensors=validate_tensors,
                )
            except Exception:
                if stage_logger is not None:
                    stage_logger("ERROR", label)
                raise
            else:
                if stage_logger is not None:
                    stage_logger("END", label)
            row.update(
                {
                    "downloaded": True,
                    "snapshot_path": str(snapshot_path),
                    "validation": validation,
                }
            )
            rows.append(row)
    return {
        "namespace": namespace,
        "output_root": str(output_path),
        "repo_count": len(rows),
        "downloaded_count": sum(1 for row in rows if row["downloaded"]),
        "skipped_existing_count": sum(1 for row in rows if row["skipped_existing"]),
        "dry_run": dry_run,
        "rows": rows,
    }


def _read_remote_json(repo_id: str, filename: str, *, revision: str | None = None) -> Any:
    path = hf_hub_download(repo_id, filename, repo_type="model", revision=revision)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_remote_bytes(repo_id: str, filename: str, *, revision: str | None = None) -> bytes:
    path = hf_hub_download(repo_id, filename, repo_type="model", revision=revision)
    return Path(path).read_bytes()


def _parse_sha256sums_bytes(payload: bytes) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in payload.decode("utf-8").splitlines():
        if not line.strip():
            continue
        digest, _, relative_path = line.partition("  ")
        if digest and relative_path:
            entries[relative_path] = digest
    return entries


def _read_remote_jsonl(
    repo_id: str, filename: str, *, revision: str | None = None
) -> list[dict]:
    path = hf_hub_download(repo_id, filename, repo_type="model", revision=revision)
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _summary_split_metrics(summary: dict[str, Any], split: str) -> tuple[dict[str, Any], int]:
    split_payload = (summary.get("metrics") or {}).get(split)
    if not isinstance(split_payload, dict):
        return {}, 0
    records = int(split_payload.get("records") or 0)
    latest_metrics = split_payload.get("latest_metrics")
    if latest_metrics is None:
        latest = split_payload.get("latest")
        if isinstance(latest, dict):
            latest_metrics = latest.get("metrics")
    metrics = dict(latest_metrics) if isinstance(latest_metrics, dict) else {}
    if records and "generated_samples" not in metrics:
        metrics["generated_samples"] = records
    return metrics, records


def _metrics_by_split(
    repo_id: str,
    file_names: set[str],
    *,
    revision: str | None,
    summary: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {"original": {}, "orbitquant": {}}
    if "benchmark/summary.json" in file_names:
        if summary is None:
            summary = _read_remote_json(repo_id, "benchmark/summary.json", revision=revision)
        for split in payload:
            metrics, _records = _summary_split_metrics(summary, split)
            if not metrics:
                continue
            payload[split].update(metrics)
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


_METADATA_COMPLETENESS_FIELDS = (
    "activation_eps",
    "quantization_device",
    "weight_quantization_backend",
    "quantization_staging_mode",
)


def _metadata_completeness_missing(
    manifest: dict[str, Any],
    *,
    model_index: dict[str, Any],
    benchmark_summary: dict[str, Any],
) -> list[str]:
    missing = []
    documents = {
        "manifest": manifest,
        "model_index": model_index,
        "benchmark_summary": benchmark_summary,
    }
    for document, payload in documents.items():
        if not payload:
            missing.append(f"{document}_missing")
            continue
        for field in _METADATA_COMPLETENESS_FIELDS:
            if payload.get(field) in (None, "", "unknown"):
                missing.append(f"{document}.{field}_missing")

    if manifest:
        for field in _METADATA_COMPLETENESS_FIELDS:
            expected = manifest.get(field)
            if expected in (None, "", "unknown"):
                continue
            for document, payload in (
                ("model_index", model_index),
                ("benchmark_summary", benchmark_summary),
            ):
                actual = payload.get(field) if payload else None
                if actual in (None, "", "unknown"):
                    continue
                if actual != expected:
                    missing.append(
                        f"{document}.{field}_mismatch: expected {expected!r}, got {actual!r}"
                    )
    return missing


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


def _release_eval_applicable(suite_name: str) -> bool:
    return suite_name in _REQUIRED_METRICS_BY_SUITE


def _native_smoke_expected_settings(suite: NativeSuite) -> dict[str, Any]:
    return {
        "suite": suite.name,
        "height": suite.height,
        "width": suite.width,
        "frames": suite.frames,
        "steps": suite.steps,
        "guidance": suite.guidance,
    }


def _native_smoke_proof_status(
    summary: dict[str, Any],
    *,
    suite: NativeSuite,
    file_names: set[str],
) -> dict[str, Any]:
    proof = summary.get("native_smoke")
    missing: list[str] = []
    if not isinstance(proof, dict):
        return {"ready": False, "missing": ["native_smoke_missing"], "proof": None}
    if proof.get("proof_format") != "orbitquant-native-smoke-v1":
        missing.append("native_smoke.proof_format")
    comparison_asset_path = proof.get("comparison_asset_path")
    if not isinstance(comparison_asset_path, str) or not _is_published_card_asset(
        comparison_asset_path
    ):
        missing.append("native_smoke.comparison_asset_path")
    elif comparison_asset_path not in file_names:
        missing.append("native_smoke.comparison_asset_path_not_uploaded")
    if int(proof.get("paired_prompt_seed_count") or 0) <= 0:
        missing.append("native_smoke.paired_prompt_seed_count")

    expected_settings = _native_smoke_expected_settings(suite)
    splits = proof.get("splits")
    if not isinstance(splits, dict):
        missing.append("native_smoke.splits")
    else:
        for split in ("original", "orbitquant"):
            split_payload = splits.get(split)
            if not isinstance(split_payload, dict):
                missing.append(f"native_smoke.{split}")
                continue
            generated_samples = int(split_payload.get("generated_samples") or 0)
            if generated_samples <= 0:
                missing.append(f"native_smoke.{split}.generated_samples")
            if int(split_payload.get("nonempty_output_count") or 0) < generated_samples:
                missing.append(f"native_smoke.{split}.nonempty_output_count")
            if suite.frames is not None:
                expected_frames = suite.frames * max(1, generated_samples)
                if int(split_payload.get("generated_frames") or 0) < expected_frames:
                    missing.append(f"native_smoke.{split}.generated_frames")
            if split_payload.get("native_settings") != [expected_settings]:
                missing.append(f"native_smoke.{split}.native_settings")
    return {"ready": not missing, "missing": missing, "proof": proof}


def _recover_native_smoke_proof_from_compact_summary(
    summary: dict[str, Any],
    *,
    suite: NativeSuite,
    file_names: set[str],
) -> tuple[dict[str, Any] | None, str | None]:
    comparison_asset_path = _preferred_published_card_asset(file_names)
    if comparison_asset_path is None:
        return None, "comparison_asset_missing"

    expected_settings = _native_smoke_expected_settings(suite)
    pair_key = [suite.name, "recovered", "published-comparison-matrix"]
    splits: dict[str, dict[str, Any]] = {}
    for split in ("original", "orbitquant"):
        metrics, records = _summary_split_metrics(summary, split)
        generated_samples = int(metrics.get("generated_samples") or records or 0)
        if generated_samples <= 0:
            return None, f"{split}.generated_samples_missing"
        generated_frames = int(metrics.get("generated_frames") or 0)
        if suite.frames is not None:
            expected_frames = suite.frames * generated_samples
            if generated_frames < expected_frames:
                return None, f"{split}.generated_frames_insufficient"
        splits[split] = {
            "records": records or generated_samples,
            "generated_samples": generated_samples,
            "generated_frames": generated_frames,
            "nonempty_output_count": generated_samples,
            "seeds": ["recovered"],
            "prompt_ids": ["published-comparison-matrix"],
            "pair_keys": [pair_key],
            "native_settings": [expected_settings],
        }

    return (
        {
            "proof_format": "orbitquant-native-smoke-v1",
            "proof_source": "recovered_from_compact_summary_and_published_comparison_matrix",
            "comparison_asset_path": comparison_asset_path,
            "paired_prompt_seed_count": 1,
            "paired_prompt_seed_keys": [pair_key],
            "splits": splits,
        },
        None,
    )


def _lfs_sha256_by_file(siblings_by_file: dict[str, Any]) -> dict[str, str]:
    checksums = {}
    for filename, sibling in siblings_by_file.items():
        lfs = getattr(sibling, "lfs", None)
        sha256 = None
        if isinstance(lfs, dict):
            sha256 = lfs.get("sha256")
        elif lfs is not None:
            sha256 = getattr(lfs, "sha256", None)
        if sha256:
            checksums[filename] = sha256
    return checksums


def _remote_checksum_mismatches(
    manifest: dict[str, Any],
    sha256sums_entries: dict[str, str],
    *,
    lfs_sha256_by_file: dict[str, str],
) -> list[str]:
    if not manifest:
        return []
    manifest_checksums = manifest.get("checksums") or {}
    if not manifest_checksums:
        return ["manifest.checksums: missing"]
    if not sha256sums_entries:
        return ["SHA256SUMS: empty or missing"]

    mismatches = []
    for relative_path in sorted(set(manifest_checksums) - set(sha256sums_entries)):
        mismatches.append(f"SHA256SUMS missing manifest entry for {relative_path}")
    for relative_path in sorted(set(manifest_checksums) & set(sha256sums_entries)):
        manifest_digest = manifest_checksums[relative_path]
        sha256sums_digest = sha256sums_entries[relative_path]
        if manifest_digest != sha256sums_digest:
            mismatches.append(
                f"manifest/SHA256SUMS mismatch for {relative_path}: "
                f"expected {manifest_digest}, got {sha256sums_digest}"
            )

    for relative_path, lfs_digest in sorted(lfs_sha256_by_file.items()):
        manifest_digest = manifest_checksums.get(relative_path)
        if manifest_digest is not None and manifest_digest != lfs_digest:
            mismatches.append(
                f"manifest/LFS mismatch for {relative_path}: "
                f"expected {manifest_digest}, got {lfs_digest}"
            )
        sha256sums_digest = sha256sums_entries.get(relative_path)
        if sha256sums_digest is not None and sha256sums_digest != lfs_digest:
            mismatches.append(
                f"SHA256SUMS/LFS mismatch for {relative_path}: "
                f"expected {sha256sums_digest}, got {lfs_digest}"
            )
    return mismatches


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


def _copy_artifact_file(source_path: Path, artifact_path: Path, output_path: Path) -> None:
    relative_path = source_path.relative_to(artifact_path)
    target_path = output_path / relative_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)


def _summary_latest_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for split, payload in (summary.get("metrics") or {}).items():
        if not isinstance(payload, dict):
            continue
        latest = payload.get("latest")
        latest_metrics = latest.get("metrics", {}) if isinstance(latest, dict) else {}
        clean[str(split)] = {
            "records": payload.get("records", 0),
            "latest_metrics": latest_metrics if isinstance(latest_metrics, dict) else {},
        }
    return clean


def _read_local_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(json.loads(line))
    return records


def _prompt_id_from_metadata(metadata: dict[str, Any]) -> str:
    prompt_record = metadata.get("prompt_record")
    if isinstance(prompt_record, dict) and prompt_record.get("id") is not None:
        return str(prompt_record["id"])
    prompt = metadata.get("prompt")
    if isinstance(prompt, str) and prompt:
        return prompt[:80]
    return "prompt"


def _local_output_is_nonempty(artifact_path: Path, value: Any) -> bool:
    if not value:
        return False
    output_path = Path(str(value))
    if not output_path.is_absolute():
        output_path = artifact_path / output_path
    return output_path.is_file() and output_path.stat().st_size > 0


def _native_settings_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "suite": metadata.get("suite"),
        "height": metadata.get("height"),
        "width": metadata.get("width"),
        "frames": metadata.get("frames"),
        "steps": metadata.get("steps"),
        "guidance": metadata.get("guidance"),
    }


def _native_smoke_split_summary(
    artifact_path: Path, records: list[dict[str, Any]]
) -> dict[str, Any]:
    generated_samples = 0
    generated_frames = 0
    nonempty_outputs = 0
    seeds = set()
    prompt_ids = set()
    pair_keys = set()
    settings = set()
    for record in records:
        metrics = record.get("metrics") if isinstance(record, dict) else {}
        metadata = record.get("metadata") if isinstance(record, dict) else {}
        if not isinstance(metrics, dict) or not isinstance(metadata, dict):
            continue
        sample_count = int(metrics.get("generated_samples") or 0)
        generated_samples += sample_count
        generated_frames += int(metrics.get("generated_frames") or 0)
        if _local_output_is_nonempty(artifact_path, metadata.get("output_path")):
            nonempty_outputs += max(1, sample_count)
        seed = metadata.get("seed")
        if seed is not None:
            seeds.add(str(seed))
        prompt_id = _prompt_id_from_metadata(metadata)
        prompt_ids.add(prompt_id)
        pair_keys.add(
            (
                str(metadata.get("suite") or ""),
                str(seed),
                prompt_id,
            )
        )
        settings.add(
            json.dumps(
                _native_settings_from_metadata(metadata),
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    return {
        "records": len(records),
        "generated_samples": generated_samples,
        "generated_frames": generated_frames,
        "nonempty_output_count": nonempty_outputs,
        "seeds": sorted(seeds),
        "prompt_ids": sorted(prompt_ids),
        "pair_keys": [list(item) for item in sorted(pair_keys)],
        "native_settings": [json.loads(item) for item in sorted(settings)],
    }


def _compact_native_smoke_proof(
    artifact_path: Path, *, comparison_asset_path: str | None
) -> dict[str, Any] | None:
    if comparison_asset_path is None:
        return None
    split_records = {
        split: _read_local_jsonl(artifact_path / "benchmark" / f"{split}.metrics.jsonl")
        for split in ("original", "orbitquant")
    }
    if not any(split_records.values()):
        return None
    splits = {
        split: _native_smoke_split_summary(artifact_path, records)
        for split, records in split_records.items()
    }
    original_pairs = {tuple(item) for item in splits["original"]["pair_keys"]}
    orbitquant_pairs = {tuple(item) for item in splits["orbitquant"]["pair_keys"]}
    paired_keys = sorted(original_pairs & orbitquant_pairs)
    return {
        "proof_format": "orbitquant-native-smoke-v1",
        "comparison_asset_path": comparison_asset_path,
        "paired_prompt_seed_count": len(paired_keys),
        "paired_prompt_seed_keys": [list(item) for item in paired_keys],
        "splits": splits,
    }


def _compact_benchmark_summary_payload(
    payload: dict[str, Any],
    *,
    native_smoke: dict[str, Any] | None = None,
) -> dict[str, Any]:
    clean = {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "metrics",
            "native_smoke",
            "source_linear_device_counts",
            "quantized_buffer_device_counts",
        }
    }
    clean["metrics"] = _summary_latest_metrics(payload)
    if native_smoke is not None:
        clean["native_smoke"] = native_smoke
    clean["published_summary"] = "compact"
    clean["raw_generation_records"] = "local-only"
    return clean


def _compact_benchmark_summary_bytes(source_path: Path) -> bytes:
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    return _json_bytes(_compact_benchmark_summary_payload(payload))


def _rewrite_compact_benchmark_summary(
    artifact_path: Path,
    output_path: Path,
    *,
    comparison_asset_path: str | None,
) -> None:
    source_path = artifact_path / "benchmark" / "summary.json"
    target_path = output_path / "benchmark" / "summary.json"
    native_smoke = _compact_native_smoke_proof(
        artifact_path,
        comparison_asset_path=comparison_asset_path,
    )
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    target_path.write_bytes(
        _json_bytes(_compact_benchmark_summary_payload(payload, native_smoke=native_smoke))
    )


def _compact_benchmark_summary_bytes_from_remote(payload: bytes) -> bytes:
    return _json_bytes(_compact_benchmark_summary_payload(json.loads(payload.decode("utf-8"))))


def _copy_compact_benchmark_summary(
    source_path: Path, artifact_path: Path, output_path: Path
) -> None:
    relative_path = source_path.relative_to(artifact_path)
    target_path = output_path / relative_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(_compact_benchmark_summary_bytes(source_path))


def _is_raw_eval_asset(relative_path: str) -> bool:
    if not relative_path.startswith("assets/"):
        return False
    name = Path(relative_path).name
    return any(marker in name for marker in _COMPACT_RAW_EVAL_ASSET_MARKERS)


def _is_published_card_asset(relative_path: str) -> bool:
    if not relative_path.startswith("assets/"):
        return False
    return Path(relative_path).name.lower().endswith("_generation_comparison_matrix.webp")


def _published_card_assets(relative_paths: set[str] | list[str]) -> list[str]:
    return sorted(path for path in relative_paths if _is_published_card_asset(path))


def _preferred_published_card_asset(relative_paths: set[str] | list[str]) -> str | None:
    assets = _published_card_assets(relative_paths)
    if not assets:
        return None
    for preferred in (
        "assets/image_generation_comparison_matrix.webp",
        "assets/video_generation_comparison_matrix.webp",
    ):
        if preferred in assets:
            return preferred
    return assets[0]


def _extra_published_card_assets(relative_paths: set[str] | list[str]) -> list[str]:
    selected = _preferred_published_card_asset(relative_paths)
    return [path for path in _published_card_assets(relative_paths) if path != selected]


def _is_publishable_artifact_file(relative_path: str) -> bool:
    return relative_path in _PUBLISHED_CORE_ARTIFACT_FILES or _is_published_card_asset(
        relative_path
    )


def _is_local_benchmark_file(relative_path: str) -> bool:
    return relative_path in _LOCAL_BENCHMARK_FILES


def _is_report_file(relative_path: str) -> bool:
    return relative_path.startswith(_COMPACT_REPORT_ROOT)


def _is_comparison_matrix_asset(path: Path) -> bool:
    return path.name.lower().endswith("_generation_comparison_matrix.webp")


def _comparison_matrix_target_path(
    source_path: Path, report_dir: Path, output_path: Path
) -> Path:
    target_path = output_path / "assets" / source_path.name
    if not target_path.exists() or sha256_file(source_path) == sha256_file(target_path):
        return target_path
    return output_path / "assets" / f"{report_dir.name}_{source_path.name}"


def _copy_first_report_comparison_asset(
    report_dirs: list[Path], output_path: Path
) -> list[str]:
    for report_dir in report_dirs:
        report_dir = report_dir.resolve()
        for source_path in sorted(report_dir.rglob("*")):
            if not source_path.is_file() or not _is_comparison_matrix_asset(source_path):
                continue
            target_path = _comparison_matrix_target_path(source_path, report_dir, output_path)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)
            return [target_path.relative_to(output_path).as_posix()]
    return []


def _report_matrix_name_prefix(relative_path: str) -> str:
    parts = Path(relative_path).parts
    if len(parts) >= 5 and parts[-2] == "assets":
        return parts[-3]
    return Path(relative_path).parent.name or "report"


def _promoted_remote_matrix_path(relative_path: str, used_paths: set[str]) -> str:
    name = Path(relative_path).name
    candidate = f"assets/{name}"
    if candidate not in used_paths:
        used_paths.add(candidate)
        return candidate

    prefix = _report_matrix_name_prefix(relative_path)
    stem = Path(name).stem
    suffix = Path(name).suffix
    candidate = f"assets/{prefix}_{name}"
    index = 2
    while candidate in used_paths:
        candidate = f"assets/{prefix}_{stem}_{index}{suffix}"
        index += 1
    used_paths.add(candidate)
    return candidate


def stage_compact_upload_artifact(
    artifact_dir: str | Path,
    output_dir: str | Path,
    *,
    report_dirs: list[str | Path] | None = None,
    validate_tensors: bool = True,
) -> dict[str, Any]:
    """Create a compact upload copy with final proof assets but without raw eval dumps."""

    artifact_path = Path(artifact_dir)
    output_path = Path(output_dir)
    artifact_root = artifact_path.resolve()
    output_root = output_path.resolve(strict=False)
    if output_root == artifact_root or output_root.is_relative_to(artifact_root):
        raise RuntimeError("staging output directory must be outside the source artifact")
    if output_path.exists() and any(output_path.iterdir()):
        raise RuntimeError(f"staging output directory is not empty: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    source_validation = validate_orbitquant_artifact(
        artifact_path,
        validate_checksums_enabled=True,
        validate_tensors=validate_tensors,
    )
    copied_files = []
    omitted_raw_eval_assets = []
    omitted_report_files = []
    omitted_unexpected_files = []
    source_files = [
        source_path
        for source_path in sorted(artifact_path.rglob("*"))
        if source_path.is_file()
    ]
    source_relative_paths = [
        source_path.relative_to(artifact_path).as_posix() for source_path in source_files
    ]
    selected_card_asset = _preferred_published_card_asset(source_relative_paths)
    for source_path in source_files:
        relative_path = source_path.relative_to(artifact_path).as_posix()
        if is_ignored_artifact_relative_path(relative_path):
            continue
        if _is_report_file(relative_path):
            omitted_report_files.append(relative_path)
            continue
        if _is_local_benchmark_file(relative_path):
            omitted_raw_eval_assets.append(relative_path)
            continue
        if relative_path == "benchmark/summary.json":
            _copy_compact_benchmark_summary(source_path, artifact_path, output_path)
            copied_files.append(relative_path)
            continue
        if _is_published_card_asset(relative_path) and relative_path != selected_card_asset:
            omitted_raw_eval_assets.append(relative_path)
            continue
        if _is_publishable_artifact_file(relative_path):
            _copy_artifact_file(source_path, artifact_path, output_path)
            copied_files.append(relative_path)
            continue
        if _is_raw_eval_asset(relative_path):
            omitted_raw_eval_assets.append(relative_path)
            continue
        if relative_path.startswith("assets/") and not _is_published_card_asset(relative_path):
            omitted_raw_eval_assets.append(relative_path)
            continue
        if not _is_publishable_artifact_file(relative_path):
            omitted_unexpected_files.append(relative_path)
            continue

    copied_report_assets = []
    report_paths = []
    artifact_report_root = artifact_path / "reports"
    if artifact_report_root.is_dir():
        report_paths.append(artifact_report_root)
    for report_dir in report_dirs or []:
        path = Path(report_dir)
        if not path.is_dir():
            raise RuntimeError(f"report directory missing: {path}")
        report_paths.append(path)
    if selected_card_asset is None:
        copied_report_assets.extend(
            _copy_first_report_comparison_asset(report_paths, output_path)
        )

    published_card_asset = selected_card_asset
    if published_card_asset is None and copied_report_assets:
        published_card_asset = sorted(set(copied_report_assets))[0]
    _rewrite_compact_benchmark_summary(
        artifact_path,
        output_path,
        comparison_asset_path=published_card_asset,
    )
    checksum_refresh = refresh_artifact_checksums(output_path)
    staged_validation = validate_orbitquant_artifact(
        output_path,
        validate_checksums_enabled=True,
        validate_tensors=validate_tensors,
    )
    return {
        "enabled": True,
        "artifact_dir": str(output_path),
        "profile": "compact",
        "source_validation": source_validation,
        "validation": staged_validation,
        "copied_file_count": len(set(copied_files) | set(copied_report_assets)),
        "copied_report_asset_count": len(set(copied_report_assets)),
        "copied_report_assets": sorted(set(copied_report_assets)),
        "omitted_report_file_count": len(omitted_report_files),
        "omitted_report_files": omitted_report_files[:50],
        "omitted_report_file_overflow": max(0, len(omitted_report_files) - 50),
        "omitted_raw_eval_asset_count": len(omitted_raw_eval_assets),
        "omitted_raw_eval_assets": omitted_raw_eval_assets[:50],
        "omitted_raw_eval_asset_overflow": max(0, len(omitted_raw_eval_assets) - 50),
        "omitted_unexpected_file_count": len(omitted_unexpected_files),
        "omitted_unexpected_files": omitted_unexpected_files[:50],
        "omitted_unexpected_file_overflow": max(0, len(omitted_unexpected_files) - 50),
        "checksum_refresh": checksum_refresh,
    }


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
                "release_eval_applicable": _release_eval_applicable(suite.name),
                "release_eval_ready": False,
            }
            try:
                info = api.model_info(repo_id, revision=revision, files_metadata=True)
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {str(exc)}"
                rows.append(row)
                continue

            siblings = list(info.siblings or [])
            siblings_by_file = {sibling.rfilename: sibling for sibling in siblings}
            files = {
                name: getattr(sibling, "size", None)
                for name, sibling in siblings_by_file.items()
            }
            file_names = set(files)
            required_missing = [
                filename for filename in _REQUIRED_ARTIFACT_FILES if filename not in file_names
            ]
            forbidden_files = sorted(
                {
                    filename
                    for filename in file_names
                    if not is_ignored_artifact_relative_path(filename)
                    and not _is_publishable_artifact_file(filename)
                }
                | set(_extra_published_card_assets(file_names))
            )
            manifest = {}
            manifest_error = None
            if "orbitquant_manifest.json" in file_names:
                try:
                    manifest = _read_remote_json(
                        repo_id, "orbitquant_manifest.json", revision=revision
                    )
                except Exception as exc:
                    manifest_error = f"{type(exc).__name__}: {str(exc)}"
            model_index = {}
            model_index_error = None
            if "model_index.json" in file_names:
                try:
                    model_index = _read_remote_json(
                        repo_id, "model_index.json", revision=revision
                    )
                except Exception as exc:
                    model_index_error = f"{type(exc).__name__}: {str(exc)}"
            sha256sums_entries: dict[str, str] = {}
            sha256sums_error = None
            if "SHA256SUMS" in file_names:
                try:
                    sha256sums_entries = _parse_sha256sums_bytes(
                        _read_remote_bytes(repo_id, "SHA256SUMS", revision=revision)
                    )
                except Exception as exc:
                    sha256sums_error = f"{type(exc).__name__}: {str(exc)}"
            benchmark_summary = {}
            benchmark_summary_error = None
            if "benchmark/summary.json" in file_names:
                try:
                    benchmark_summary = _read_remote_json(
                        repo_id, "benchmark/summary.json", revision=revision
                    )
                except Exception as exc:
                    benchmark_summary_error = f"{type(exc).__name__}: {str(exc)}"
            metrics = _metrics_by_split(
                repo_id,
                file_names,
                revision=revision,
                summary=benchmark_summary,
            )
            missing_metrics = _missing_required_metrics(metrics, suite_name=suite.name)
            manifest_mismatches = _manifest_mismatches(
                manifest,
                suite=suite,
                bit_setting=bit_setting,
            )
            metadata_missing = _metadata_completeness_missing(
                manifest,
                model_index=model_index,
                benchmark_summary=benchmark_summary,
            )
            remote_checksum_mismatches = _remote_checksum_mismatches(
                manifest,
                sha256sums_entries,
                lfs_sha256_by_file=_lfs_sha256_by_file(siblings_by_file),
            )
            generated_splits = sorted(
                split for split, values in metrics.items() if values.get("generated_samples")
            )
            native_smoke_proof = _native_smoke_proof_status(
                benchmark_summary,
                suite=suite,
                file_names=file_names,
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
                        if _is_published_card_asset(filename)
                    ),
                    "forbidden_file_count": len(forbidden_files),
                    "forbidden_files": forbidden_files[:100],
                    "forbidden_file_overflow": max(0, len(forbidden_files) - 100),
                    "required_missing": required_missing,
                    "manifest_error": manifest_error,
                    "manifest_mismatches": manifest_mismatches,
                    "manifest_warnings": _manifest_warnings(manifest),
                    "model_index_error": model_index_error,
                    "benchmark_summary_error": benchmark_summary_error,
                    "metadata_complete_ready": not metadata_missing,
                    "metadata_missing": metadata_missing,
                    "sha256sums_error": sha256sums_error,
                    "sha256sums_entry_count": len(sha256sums_entries),
                    "remote_lfs_checksum_count": len(_lfs_sha256_by_file(siblings_by_file)),
                    "remote_checksum_mismatches": remote_checksum_mismatches,
                    "quantized_modules": len(manifest.get("quantized_modules") or []),
                    "adaln_modules": len(manifest.get("adaln_modules") or []),
                    "metrics_by_split": metrics,
                    "generated_splits": generated_splits,
                    "missing_required_metrics": missing_metrics,
                    "native_smoke_proof_ready": native_smoke_proof["ready"],
                    "native_smoke_missing_evidence": native_smoke_proof["missing"],
                }
            )
            row["artifact_ready"] = (
                not required_missing
                and files.get("model.safetensors") is not None
                and not manifest_error
                and not model_index_error
                and not benchmark_summary_error
                and not manifest_mismatches
                and not forbidden_files
                and not sha256sums_error
                and not remote_checksum_mismatches
            )
            row["native_smoke_ready"] = (
                row["artifact_ready"]
                and native_smoke_proof["ready"]
            )
            row["release_eval_ready"] = (
                row["release_eval_applicable"]
                and row["metadata_complete_ready"]
                and row["native_smoke_ready"]
                and not missing_metrics
            )
            rows.append(row)
    release_eval_applicable_count = sum(
        1 for row in rows if row.get("release_eval_applicable")
    )
    return {
        "namespace": namespace,
        "repo_count": len(rows),
        "existing_count": sum(1 for row in rows if row["exists"]),
        "artifact_ready_count": sum(1 for row in rows if row["artifact_ready"]),
        "native_smoke_ready_count": sum(1 for row in rows if row["native_smoke_ready"]),
        "metadata_complete_ready_count": sum(
            1 for row in rows if row.get("metadata_complete_ready")
        ),
        "release_eval_applicable_count": release_eval_applicable_count,
        "release_eval_not_applicable_count": len(rows) - release_eval_applicable_count,
        "release_eval_ready_count": sum(1 for row in rows if row["release_eval_ready"]),
        "missing_required_metric_count": sum(
            len(row.get("missing_required_metrics", [])) for row in rows
        ),
        "manifest_warning_count": sum(len(row.get("manifest_warnings", [])) for row in rows),
        "metadata_missing_count": sum(len(row.get("metadata_missing", [])) for row in rows),
        "remote_checksum_mismatch_count": sum(
            len(row.get("remote_checksum_mismatches", [])) for row in rows
        )
        + sum(1 for row in rows if row.get("sha256sums_error")),
        "forbidden_file_count": sum(row.get("forbidden_file_count", 0) for row in rows),
        "rows": rows,
    }


def _markdown_status(value: bool) -> str:
    return "yes" if value else "no"


def _markdown_release_eval_status(row: dict[str, Any]) -> str:
    if not row.get("release_eval_applicable", True):
        return "n/a"
    return _markdown_status(bool(row.get("release_eval_ready")))


def _short_sha(value: Any) -> str:
    if not value:
        return ""
    return str(value)[:12]


def _missing_metrics_label(row: dict[str, Any]) -> str:
    missing = row.get("missing_required_metrics") or []
    if not missing:
        return ""
    labels = sorted({f"{item['split']}:{item['metric']}" for item in missing})
    return ", ".join(labels)


def _missing_metrics_count_label(row: dict[str, Any]) -> str:
    count = len(row.get("missing_required_metrics") or [])
    if count == 0:
        return ""
    return f"{count} release metrics missing"


def render_hf_artifact_audit_markdown(payload: dict[str, Any]) -> str:
    """Render a compact human-readable HF artifact audit report."""

    repo_count = payload.get("repo_count", 0)
    release_eval_applicable_count = payload.get("release_eval_applicable_count", repo_count)
    lines = [
        "# OrbitQuant HF Artifact Audit",
        "",
        f"- Namespace: `{payload.get('namespace', '')}`",
        f"- Repositories: {payload.get('existing_count', 0)} / {repo_count} existing",
        f"- Artifact ready: {payload.get('artifact_ready_count', 0)} / {repo_count}",
        (
            f"- Metadata complete: {payload.get('metadata_complete_ready_count', 0)} / "
            f"{repo_count}"
        ),
        f"- Native smoke ready: {payload.get('native_smoke_ready_count', 0)} / {repo_count}",
        f"- Release eval applicable: {release_eval_applicable_count} / {repo_count}",
        (
            f"- Release eval ready: {payload.get('release_eval_ready_count', 0)} / "
            f"{release_eval_applicable_count}"
        ),
        f"- Missing release metrics: {payload.get('missing_required_metric_count', 0)}",
        f"- Manifest warnings: {payload.get('manifest_warning_count', 0)}",
        f"- Metadata missing fields: {payload.get('metadata_missing_count', 0)}",
        f"- Remote checksum mismatches: {payload.get('remote_checksum_mismatch_count', 0)}",
        f"- Forbidden files: {payload.get('forbidden_file_count', 0)}",
        "",
        "## Readiness Semantics",
        "",
        (
            "- Artifact ready means the compact repository has required files, "
            "consistent manifest/checksum metadata, and no forbidden raw files."
        ),
        (
            "- Native smoke ready means a final comparison asset and original/"
            "OrbitQuant generated-sample counters are present; it is not a "
            "GenEval or VBench result."
        ),
        (
            "- Metadata complete means activation norm clamp, quantization device, "
            "weight quantization backend, and staging mode are recorded."
        ),
        (
            "- Release eval ready is the only readiness flag for paper metric "
            "or reproduction claims; it requires metadata completeness."
        ),
        "",
        "## Artifact Matrix",
        "",
        (
            "| Suite | Bits | Repo | Private | Artifact | Metadata | Native Smoke | "
            "Release Eval | SHA | Missing Release Metrics | Forbidden Files |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload.get("rows", []):
        repo_id = row.get("repo_id", "")
        repo_label = f"`{repo_id}`" if repo_id else ""
        if row.get("error") and not row.get("exists"):
            missing_metrics = row.get("error", "")
        else:
            missing_metrics = _missing_metrics_count_label(row)
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("suite", "")),
                    str(row.get("bit_setting", "")),
                    repo_label,
                    _markdown_status(bool(row.get("private"))),
                    _markdown_status(bool(row.get("artifact_ready"))),
                    _markdown_status(bool(row.get("metadata_complete_ready"))),
                    _markdown_status(bool(row.get("native_smoke_ready"))),
                    _markdown_release_eval_status(row),
                    _short_sha(row.get("sha")),
                    missing_metrics,
                    str(row.get("forbidden_file_count") or ""),
                ]
            )
            + " |"
        )

    blocking_rows = [
        row
        for row in payload.get("rows", [])
        if row.get("release_eval_applicable", True)
        and row.get("native_smoke_ready")
        and not row.get("release_eval_ready")
    ]
    if blocking_rows:
        lines.extend(["", "## Release Eval Gaps", ""])
        for row in blocking_rows:
            missing_label = _missing_metrics_label(row) or "release metrics missing"
            lines.append(
                f"- `{row.get('repo_id')}`: {missing_label}"
            )

    return "\n".join(lines) + "\n"


def repair_hf_artifact_metadata(
    *,
    repo_id: str,
    quantization_device: str,
    weight_quantization_backend: str,
    quantization_staging_mode: str | None = None,
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

    manifest_payload = json.loads(manifest_bytes.decode("utf-8"))
    config_payload = json.loads(config_bytes.decode("utf-8"))
    manifest = OrbitQuantManifest.from_dict(manifest_payload)
    config = OrbitQuantConfig.from_dict(config_payload)
    model_index = json.loads(model_index_bytes.decode("utf-8"))
    benchmark_summary = json.loads(benchmark_bytes.decode("utf-8"))

    before = {
        "quantization_device": manifest.quantization_device,
        "weight_quantization_backend": manifest.weight_quantization_backend,
        "quantization_staging_mode": manifest.quantization_staging_mode,
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
        activation_eps=float(manifest_payload.get("activation_eps", config.activation_eps)),
        quantization_device=quantization_device,
        weight_quantization_backend=weight_quantization_backend,
        quantization_staging_mode=quantization_staging_mode
        or manifest.quantization_staging_mode,
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
    model_index["quantization_staging_mode"] = repaired_manifest.quantization_staging_mode
    model_index["activation_eps"] = repaired_manifest.activation_eps
    benchmark_summary["quantization_device"] = quantization_device
    benchmark_summary["weight_quantization_backend"] = weight_quantization_backend
    benchmark_summary["quantization_staging_mode"] = repaired_manifest.quantization_staging_mode
    benchmark_summary["activation_eps"] = repaired_manifest.activation_eps

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
            "quantization_staging_mode": repaired_manifest.quantization_staging_mode,
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


def repair_hf_native_smoke_proof(
    *,
    repo_id: str,
    suite: NativeSuite,
    revision: str | None = None,
    commit_message: str | None = None,
    dry_run: bool = False,
    api: HfApi | None = None,
) -> dict[str, Any]:
    """Add a compact native-smoke proof block from existing published evidence."""

    api = HfApi() if api is None else api
    info = api.model_info(repo_id, revision=revision, files_metadata=True)
    file_names = {sibling.rfilename for sibling in info.siblings or []}

    manifest_bytes = _read_remote_bytes(repo_id, "orbitquant_manifest.json", revision=revision)
    benchmark_bytes = _read_remote_bytes(repo_id, "benchmark/summary.json", revision=revision)
    sha256sums_bytes = _read_remote_bytes(repo_id, "SHA256SUMS", revision=revision)

    manifest = OrbitQuantManifest.from_dict(json.loads(manifest_bytes.decode("utf-8")))
    benchmark_summary = json.loads(benchmark_bytes.decode("utf-8"))
    current_status = _native_smoke_proof_status(
        benchmark_summary,
        suite=suite,
        file_names=file_names,
    )
    result: dict[str, Any] = {
        "repo_id": repo_id,
        "repo_type": "model",
        "revision": revision,
        "dry_run": dry_run,
        "suite": suite.name,
        "existing_native_smoke_ready": current_status["ready"],
        "repair_skipped_reason": None,
        "changed_files": [],
        "commit": None,
    }
    if current_status["ready"]:
        return result

    native_smoke, skipped_reason = _recover_native_smoke_proof_from_compact_summary(
        benchmark_summary,
        suite=suite,
        file_names=file_names,
    )
    if native_smoke is None:
        result["repair_skipped_reason"] = skipped_reason
        return result
    recovered_status = _native_smoke_proof_status(
        {"native_smoke": native_smoke},
        suite=suite,
        file_names=file_names,
    )
    if not recovered_status["ready"]:
        result["repair_skipped_reason"] = (
            "recovered_native_smoke_invalid: "
            + ",".join(recovered_status["missing"])
        )
        return result

    next_benchmark_summary = dict(benchmark_summary)
    next_benchmark_summary["native_smoke"] = native_smoke
    next_benchmark_bytes = _json_bytes(next_benchmark_summary)
    next_manifest = replace(
        manifest,
        checksums={
            **manifest.checksums,
            "benchmark/summary.json": _sha256_bytes(next_benchmark_bytes),
        },
    )
    next_manifest_bytes = _json_bytes(next_manifest.to_dict())

    sha_entries = _parse_sha256sums_bytes(sha256sums_bytes)
    sha_entries.update(
        {
            "benchmark/summary.json": _sha256_bytes(next_benchmark_bytes),
            "orbitquant_manifest.json": _sha256_bytes(next_manifest_bytes),
        }
    )
    sha_entries.pop("SHA256SUMS", None)
    next_sha256sums_bytes = _sha256sums_bytes(sha_entries)

    file_payloads = {
        "benchmark/summary.json": next_benchmark_bytes,
        "orbitquant_manifest.json": next_manifest_bytes,
        "SHA256SUMS": next_sha256sums_bytes,
    }
    original_payloads = {
        "benchmark/summary.json": benchmark_bytes,
        "orbitquant_manifest.json": manifest_bytes,
        "SHA256SUMS": sha256sums_bytes,
    }
    changed_files = [
        filename
        for filename, payload in file_payloads.items()
        if original_payloads.get(filename) != payload
    ]
    result.update(
        {
            "proof_source": native_smoke["proof_source"],
            "comparison_asset_path": native_smoke["comparison_asset_path"],
            "changed_files": changed_files,
        }
    )
    if dry_run or not changed_files:
        return result

    operations = [
        CommitOperationAdd(path_in_repo=filename, path_or_fileobj=file_payloads[filename])
        for filename in changed_files
    ]
    commit_info = api.create_commit(
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        operations=operations,
        commit_message=commit_message or "Record OrbitQuant native smoke proof",
    )
    result["commit"] = _commit_info_payload(commit_info)
    return result


def repair_hf_native_smoke_proof_matrix(
    *,
    namespace: str = "WaveCut",
    suites: list[NativeSuite] | None = None,
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
                row = repair_hf_native_smoke_proof(
                    repo_id=repo_id,
                    suite=suite,
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
                row["bit_setting"] = bit_setting
            rows.append(row)
    return {
        "namespace": namespace,
        "repo_count": len(rows),
        "dry_run": dry_run,
        "changed_repo_count": sum(1 for row in rows if row.get("changed_files")),
        "skipped_repo_count": sum(1 for row in rows if row.get("repair_skipped_reason")),
        "error_count": sum(1 for row in rows if row.get("error")),
        "rows": rows,
    }


def cleanup_hf_artifact_reports(
    *,
    repo_id: str,
    revision: str | None = None,
    commit_message: str | None = None,
    dry_run: bool = False,
    api: HfApi | None = None,
) -> dict[str, Any]:
    """Promote final comparison matrices and remove non-publishable remote files."""

    api = HfApi() if api is None else api
    info = api.model_info(repo_id, revision=revision, files_metadata=True)
    file_names = {sibling.rfilename for sibling in info.siblings or []}
    report_files = sorted(filename for filename in file_names if _is_report_file(filename))
    forbidden_files = sorted(
        {
            filename
            for filename in file_names
            if not is_ignored_artifact_relative_path(filename)
            and not _is_publishable_artifact_file(filename)
        }
        | set(_extra_published_card_assets(file_names))
    )
    report_matrix_files = [
        filename for filename in report_files if _is_comparison_matrix_asset(Path(filename))
    ]

    manifest_bytes = _read_remote_bytes(repo_id, "orbitquant_manifest.json", revision=revision)
    readme_bytes = _read_remote_bytes(repo_id, "README.md", revision=revision)
    benchmark_bytes = _read_remote_bytes(repo_id, "benchmark/summary.json", revision=revision)
    sha256sums_bytes = _read_remote_bytes(repo_id, "SHA256SUMS", revision=revision)

    manifest = OrbitQuantManifest.from_dict(json.loads(manifest_bytes.decode("utf-8")))
    next_benchmark_bytes = _compact_benchmark_summary_bytes_from_remote(benchmark_bytes)
    used_paths = set(file_names) - set(forbidden_files)
    promoted_assets: dict[str, bytes] = {}
    existing_card_asset = _preferred_published_card_asset(file_names)
    report_matrices_to_promote = report_matrix_files[:1] if existing_card_asset is None else []
    for report_matrix in report_matrices_to_promote:
        target_path = _promoted_remote_matrix_path(report_matrix, used_paths)
        if target_path in file_names and target_path not in forbidden_files:
            continue
        promoted_assets[target_path] = _read_remote_bytes(
            repo_id, report_matrix, revision=revision
        )

    cleaned_checksums = {
        relative_path: digest
        for relative_path, digest in manifest.checksums.items()
        if not is_ignored_artifact_relative_path(relative_path)
        and _is_publishable_artifact_file(relative_path)
        and relative_path not in forbidden_files
    }
    cleaned_checksums.update(
        {
            relative_path: _sha256_bytes(payload)
            for relative_path, payload in promoted_assets.items()
        }
    )
    cleaned_checksums["benchmark/summary.json"] = _sha256_bytes(next_benchmark_bytes)
    cleaned_manifest = replace(manifest, checksums=cleaned_checksums)
    next_manifest_bytes = _json_bytes(cleaned_manifest.to_dict())
    next_readme_bytes = render_model_card(cleaned_manifest).encode("utf-8")

    sha_entries = {
        relative_path: digest
        for relative_path, digest in _parse_sha256sums_bytes(sha256sums_bytes).items()
        if not is_ignored_artifact_relative_path(relative_path)
        and _is_publishable_artifact_file(relative_path)
        and relative_path not in forbidden_files
    }
    sha_entries.update(
        {
            relative_path: _sha256_bytes(payload)
            for relative_path, payload in promoted_assets.items()
        }
    )
    sha_entries.update(
        {
            "orbitquant_manifest.json": _sha256_bytes(next_manifest_bytes),
            "README.md": _sha256_bytes(next_readme_bytes),
            "benchmark/summary.json": _sha256_bytes(next_benchmark_bytes),
        }
    )
    sha_entries.pop("SHA256SUMS", None)
    next_sha256sums_bytes = _sha256sums_bytes(sha_entries)

    file_payloads: dict[str, bytes] = {
        "orbitquant_manifest.json": next_manifest_bytes,
        "benchmark/summary.json": next_benchmark_bytes,
        "README.md": next_readme_bytes,
        "SHA256SUMS": next_sha256sums_bytes,
        **promoted_assets,
    }
    original_payloads = {
        "orbitquant_manifest.json": manifest_bytes,
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
        "report_file_count": len(report_files),
        "report_matrix_count": len(report_matrix_files),
        "forbidden_file_count": len(forbidden_files),
        "forbidden_files": forbidden_files[:100],
        "forbidden_file_overflow": max(0, len(forbidden_files) - 100),
        "promoted_assets": sorted(promoted_assets),
        "changed_files": changed_files,
        "delete_paths": [
            *[filename for filename in forbidden_files if not _is_report_file(filename)],
            *(["reports"] if report_files else []),
        ],
        "commit": None,
    }
    if dry_run or (not changed_files and not forbidden_files):
        return result

    operations: list[Any] = [
        CommitOperationAdd(path_in_repo=filename, path_or_fileobj=file_payloads[filename])
        for filename in changed_files
    ]
    operations.extend(
        CommitOperationDelete(path_in_repo=filename, is_folder=False)
        for filename in forbidden_files
        if not _is_report_file(filename)
    )
    if report_files:
        operations.append(CommitOperationDelete(path_in_repo="reports", is_folder=True))
    commit_info = api.create_commit(
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        operations=operations,
        commit_message=commit_message or "Clean OrbitQuant artifact assets",
    )
    result["commit"] = _commit_info_payload(commit_info)
    return result


def repair_hf_artifact_metadata_matrix(
    *,
    namespace: str = "WaveCut",
    suites: list[NativeSuite] | None = None,
    quantization_device: str,
    weight_quantization_backend: str,
    quantization_staging_mode: str | None = None,
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
                    quantization_staging_mode=quantization_staging_mode,
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


def cleanup_hf_artifact_reports_matrix(
    *,
    namespace: str = "WaveCut",
    suites: list[NativeSuite] | None = None,
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
                row = cleanup_hf_artifact_reports(
                    repo_id=repo_id,
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
        "changed_repo_count": sum(
            1 for row in rows if row.get("changed_files") or row.get("delete_paths")
        ),
        "error_count": sum(1 for row in rows if row.get("error")),
        "report_file_count": sum(row.get("report_file_count", 0) for row in rows),
        "promoted_asset_count": sum(len(row.get("promoted_assets", [])) for row in rows),
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
    replace_repo_files: bool = True,
    validate_tensors: bool = True,
    upload_profile: str = "compact",
    report_dirs: list[str | Path] | None = None,
    staging_dir: str | Path | None = None,
    dry_run: bool = False,
    api: HfApi | None = None,
) -> dict[str, Any]:
    """Validate and upload an OrbitQuant artifact directory to a HF model repo."""

    artifact_path = Path(artifact_dir)
    if upload_profile != "compact":
        raise ValueError(f"unsupported upload profile: {upload_profile}")
    temp_staging = None
    if staging_dir is None:
        temp_staging = tempfile.TemporaryDirectory(prefix="orbitquant-hf-upload-")
        upload_path = Path(temp_staging.name) / "artifact"
    else:
        upload_path = Path(staging_dir)
    try:
        staging = stage_compact_upload_artifact(
            artifact_path,
            upload_path,
            report_dirs=report_dirs,
            validate_tensors=validate_tensors,
        )
    except Exception:
        if temp_staging is not None:
            temp_staging.cleanup()
        raise
    validation = staging["validation"]
    upload_kwargs = {
        "repo_id": repo_id,
        "repo_type": "model",
        "folder_path": str(upload_path),
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
        "upload_profile": upload_profile,
        "staging": staging,
        "dry_run": dry_run,
        "validation": validation,
        "upload": None,
        "uploaded_repo": None,
        "upload_kwargs": {
            key: value for key, value in upload_kwargs.items() if key != "folder_path"
        },
    }
    try:
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
    finally:
        if temp_staging is not None:
            temp_staging.cleanup()
