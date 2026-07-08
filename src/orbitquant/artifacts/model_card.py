from __future__ import annotations

from typing import Any

from orbitquant.artifacts.manifest import OrbitQuantManifest
from orbitquant.eval.native_settings import NativeSuite, list_native_suites


def _comparison_assets(checksums: dict[str, str]) -> list[str]:
    assets = []
    for path in checksums:
        if not path.startswith("assets/"):
            continue
        name = path.rsplit("/", maxsplit=1)[-1].lower()
        if name.endswith("_generation_comparison_matrix.webp"):
            assets.append(path)
    assets = sorted(assets)
    for preferred in (
        "assets/image_generation_comparison_matrix.webp",
        "assets/video_generation_comparison_matrix.webp",
    ):
        if preferred in assets:
            return [preferred]
    return assets[:1]


def _artifact_slug(model_id: str, bits: str) -> str:
    return f"{model_id.rsplit('/', maxsplit=1)[-1]}-OrbitQuant-{bits}"


def _native_suite_for_source_model(source_model_id: str) -> NativeSuite | None:
    for suite in list_native_suites():
        if suite.model_id == source_model_id:
            return suite
    return None


def _install_snippet() -> str:
    return "\n".join(
        [
            "```bash",
            "pip install \"orbitquant[hf]\"",
            "```",
        ]
    )


def _usage_snippet(source_model_id: str, bits: str) -> str:
    placeholder_repo = f"WaveCut/{_artifact_slug(source_model_id, bits)}"
    suite = _native_suite_for_source_model(source_model_id)
    lines = [
        "```python",
        "import torch",
    ]
    if suite is not None and suite.frames is not None:
        lines.append("from diffusers.utils import export_to_video")
    lines.extend(
        [
            "from huggingface_hub import snapshot_download",
            "from orbitquant import load_quantized_pipeline_from_artifact",
            "",
            f'artifact_id = "{placeholder_repo}"',
            "",
            "artifact_dir = snapshot_download(artifact_id, repo_type=\"model\")",
            "pipe = load_quantized_pipeline_from_artifact(",
            "    artifact_dir,",
            "    torch_dtype=torch.bfloat16,",
            "    device=\"cuda\",",
            ")",
            "",
        ]
    )
    if suite is None:
        lines.extend(
            [
                "result = pipe(",
                "    prompt=\"A precise product photo of a red ceramic mug on a wooden desk\",",
                ")",
                "```",
            ]
        )
        return "\n".join(lines)

    if suite.frames is None:
        output_name = {
            "black-forest-labs/FLUX.2-klein-4B": "flux2-klein-orbitquant.png",
            "black-forest-labs/FLUX.1-schnell": "flux1-schnell-orbitquant.png",
            "Tongyi-MAI/Z-Image-Turbo": "z-image-orbitquant.png",
        }.get(source_model_id, "orbitquant.png")
        lines.extend(
            [
                "image = pipe(",
                "    prompt=\"A precise product photo of a red ceramic mug on a wooden desk\",",
                f"    height={suite.height},",
                f"    width={suite.width},",
                f"    num_inference_steps={suite.steps},",
                f"    guidance_scale={suite.guidance},",
                ").images[0]",
                f'image.save("{output_name}")',
                "```",
            ]
        )
        return "\n".join(lines)

    export_fps = suite.export_fps or 16
    lines.extend(
        [
            "frames = pipe(",
            "    prompt=\"A cinematic shot of a small robot walking through a neon market\",",
            f"    height={suite.height},",
            f"    width={suite.width},",
            f"    num_frames={suite.frames},",
            f"    num_inference_steps={suite.steps},",
            f"    guidance_scale={suite.guidance},",
            ").frames[0]",
            f'export_to_video(frames, "wan-orbitquant.mp4", fps={export_fps})',
            "```",
        ]
    )
    return "\n".join(lines)


def _native_settings_section(source_model_id: str) -> list[str]:
    suite = _native_suite_for_source_model(source_model_id)
    if suite is None:
        return []
    output = "video" if suite.frames is not None else "image"
    if suite.note.startswith("Extra target"):
        scope = (suite.note[:1].lower() + suite.note[1:]).rstrip(".")
    elif suite.metric == "vbench":
        scope = "paper video target"
    else:
        scope = "paper image target"
    rows = [
        ("Pipeline", f"`{suite.pipeline}`"),
        ("Resolution", f"`{suite.width}x{suite.height}`"),
    ]
    if suite.frames is not None:
        rows.append(("Frames", f"`{suite.frames}`"))
    rows.extend(
        [
            ("Inference steps", f"`{suite.steps}`"),
            ("Guidance scale", f"`{suite.guidance}`"),
        ]
    )
    if suite.export_fps is not None:
        rows.append(("Export FPS", f"`{suite.export_fps}`"))
    rows.extend(
        [
            ("Output", output),
            ("Scope", scope),
        ]
    )

    lines = [
        "## Native Settings",
        "",
        "Use these settings when comparing this artifact against the BF16 source "
        "model or the visual assets below:",
        "",
        "| Setting | Value |",
        "| --- | --- |",
    ]
    lines.extend(f"| {name} | {value} |" for name, value in rows)
    lines.append("")
    return lines


def _validation_status_section(source_model_id: str) -> list[str]:
    release_metric = {
        "black-forest-labs/FLUX.1-schnell": "GenEval",
        "Tongyi-MAI/Z-Image-Turbo": "GenEval",
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers": "VBench",
    }.get(source_model_id)
    release_line = (
        f"- Release-grade {release_metric} metrics: not included in this artifact."
        if release_metric
        else "- Release-grade paper metrics: not applicable to this extra target."
    )
    return [
        "## Validation Status",
        "",
        "- Native BF16-vs-OrbitQuant comparison: included when the visual matrix "
        "below is present.",
        release_line,
        "- The model card reports artifact-level validation status only.",
        "",
    ]


def _native_validation_proof_section(
    benchmark_summary: dict[str, Any] | None,
) -> list[str]:
    if not isinstance(benchmark_summary, dict):
        return []
    proof = benchmark_summary.get("native_smoke")
    if not isinstance(proof, dict):
        return []

    rows: list[tuple[str, str]] = []
    comparison_asset_path = proof.get("comparison_asset_path")
    if isinstance(comparison_asset_path, str) and comparison_asset_path:
        rows.append(("Comparison matrix", f"`{comparison_asset_path}`"))

    paired_count = proof.get("paired_prompt_seed_count")
    if paired_count is not None:
        rows.append(("Paired prompt/seed count", f"`{paired_count}`"))

    splits = proof.get("splits")
    if isinstance(splits, dict):
        for split, label in (
            ("original", "BF16 source"),
            ("orbitquant", "OrbitQuant"),
        ):
            split_payload = splits.get(split)
            if not isinstance(split_payload, dict):
                continue
            for key, name in (
                ("generated_samples", "generated samples"),
                ("generated_frames", "generated frames"),
                ("nonempty_output_count", "nonempty outputs"),
            ):
                value = split_payload.get(key)
                if value is not None:
                    rows.append((f"{label} {name}", f"`{value}`"))

    if not rows:
        return []

    lines = [
        "## Native Validation Evidence",
        "",
        "The compact benchmark summary records native BF16-vs-OrbitQuant "
        "evidence for the comparison matrix below. Detailed per-sample generation "
        "records are retained outside this compact artifact.",
        "",
        "| Evidence | Value |",
        "| --- | --- |",
    ]
    lines.extend(f"| {name} | {value} |" for name, value in rows)
    lines.append("")
    return lines


def render_model_card(
    manifest: OrbitQuantManifest,
    *,
    benchmark_summary: dict[str, Any] | None = None,
) -> str:
    data = manifest.to_dict()
    bits = f"W{data['weight_bits']}A{data['activation_bits']}"
    comparison_assets = _comparison_assets(data["checksums"])
    native_settings_lines = _native_settings_section(data["source_model_id"])
    validation_status_lines = _validation_status_section(data["source_model_id"])
    native_validation_proof_lines = _native_validation_proof_section(benchmark_summary)
    adaln_group_size = int(data.get("adaln_group_size", 64))
    adaln_default_note = (
        "- AdaLN group-size note: paper default."
        if adaln_group_size == 64
        else "- AdaLN group-size note: non-paper-default setting."
    )
    comparison_lines = []
    if comparison_assets:
        comparison_lines.extend(
            [
                "## Visual Comparison",
                "",
                "The following assets are stored in this artifact and compare the BF16 "
                "base generation against the OrbitQuant generation with the same prompt "
                "and seed.",
                "",
            ]
        )
        for path in comparison_assets:
            comparison_lines.append(f"![{path}]({path})")
            comparison_lines.append("")
    else:
        comparison_lines.extend(
            [
                "## Visual Comparison",
                "",
                "Validation status: comparison asset missing. This artifact does "
                "not include a generation comparison matrix.",
                "",
            ]
        )

    return "\n".join(
        [
            "---",
            f"base_model: {data['source_model_id']}",
            f"license: {data['source_license']}",
            "tags:",
            "- orbitquant",
            "- quantized",
            "- diffusers",
            "- diffusion-transformer",
            "---",
            "",
            f"# {data['source_model_id']} OrbitQuant {bits}",
            "",
            "This repository contains a compact OrbitQuant transformer-component "
            "artifact for the source Diffusers model listed above. It is intended "
            "to be loaded into the original pipeline, not used as a standalone "
            "Diffusers pipeline repository.",
            "",
            "OrbitQuant is a calibration-free post-training quantization method "
            "for image and video diffusion transformers. This artifact keeps the "
            "text encoders, VAE, embeddings, timestep MLP, and final heads in the "
            "source precision by default and replaces the transformer linear "
            "projections with OrbitQuant modules.",
            "",
            "## Usage",
            "",
            "Install OrbitQuant and the Hugging Face runtime dependencies:",
            "",
            _install_snippet(),
            "",
            "Download this model repository as an OrbitQuant artifact, then load "
            "the source Diffusers pipeline with the quantized component patched in:",
            "",
            _usage_snippet(data["source_model_id"], bits),
            "",
            *native_settings_lines,
            *validation_status_lines,
            *native_validation_proof_lines,
            "## Quantization",
            "",
            f"- Method: `{data['quant_method']}`",
            f"- Bits: `{bits}`",
            f"- Runtime mode: `{data['runtime_mode']}`",
            f"- Activation kernel backend: `{data['activation_kernel_backend']}`",
            f"- Activation normalization epsilon: `{data['activation_eps']}`",
            f"- Quantization device: `{data['quantization_device']}`",
            f"- Weight quantization backend: `{data['weight_quantization_backend']}`",
            f"- Target policy: `{data['target_policy']}`",
            f"- AdaLN policy: `{data['adaln_policy']}`",
            f"- AdaLN group size: `{adaln_group_size}`",
            adaln_default_note,
            f"- Rotation: `{data['rotation']}`",
            f"- Rotation seed: `{data['rotation_seed']}`",
            f"- Block size: `{data['block_size']}`",
            f"- Block size policy: `{data['block_size_policy']}`",
            f"- Codebook: `{data['codebook']}`",
            f"- Codebook version: `{data['codebook_version']}`",
            f"- Quantized transformer modules: `{len(data['quantized_modules'])}`",
            f"- AdaLN INT4 modules: `{len(data['adaln_modules'])}`",
            f"- Skipped modules: `{len(data['skipped_modules'])}`",
            "- Calibration data: none",
            "- Text encoders and VAE: left in source precision by default",
            "",
            *comparison_lines,
            "## Source",
            "",
            f"- Model: `{data['source_model_id']}`",
            f"- Revision: `{data['source_revision']}`",
            f"- Source license: `{data['source_license']}`",
            "- OrbitQuant paper: https://arxiv.org/abs/2607.02461",
            "",
            "## Artifact Files",
            "",
            "- `model.safetensors`: packed OrbitQuant/INT4 module tensors.",
            "- `quantization_config.json`: serialized OrbitQuant runtime settings.",
            "- `orbitquant_manifest.json`: source provenance, policies, module lists, "
            "and checksums.",
            "- `orbitquant_codebooks.safetensors`: Lloyd-Max codebooks.",
            "- `orbitquant_rotations.safetensors`: deterministic RPBH rotation metadata.",
            "",
            "## Limitations",
            "",
            "- This is a transformer-component artifact; load it into the source "
            "pipeline as shown above.",
            "- Runtime mode may dequantize packed weights before BF16 matmul. Disk "
            "artifacts are compact, while runtime VRAM depends on the selected "
            "backend.",
            "- Quality depends on the source model and bit setting. Very low-bit "
            "settings can degrade prompt following or visual detail.",
            "",
        ]
    )
