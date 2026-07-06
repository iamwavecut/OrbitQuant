from __future__ import annotations

from orbitquant.artifacts.manifest import OrbitQuantManifest


def render_model_card(manifest: OrbitQuantManifest) -> str:
    data = manifest.to_dict()
    bits = f"W{data['weight_bits']}A{data['activation_bits']}"
    return "\n".join(
        [
            "---",
            f"base_model: {data['source_model_id']}",
            "tags:",
            "- orbitquant",
            "- quantized",
            "- diffusers",
            "---",
            "",
            f"# {data['source_model_id']} OrbitQuant {bits}",
            "",
            "This is an OrbitQuant artifact generated from the source model listed above.",
            "",
            "## Quantization",
            "",
            f"- Method: `{data['quant_method']}`",
            f"- Bits: `{bits}`",
            f"- Runtime mode: `{data['runtime_mode']}`",
            f"- Activation kernel backend: `{data['activation_kernel_backend']}`",
            f"- Rotation: `{data['rotation']}`",
            f"- Codebook: `{data['codebook']}`",
            "- Calibration data: none",
            "- Text encoders and VAE: left in source precision by default",
            "",
            "## Source",
            "",
            f"- Model: `{data['source_model_id']}`",
            f"- Revision: `{data['source_revision']}`",
            f"- Source license: `{data['source_license']}`",
            "- OrbitQuant paper: https://arxiv.org/abs/2607.02461",
            "",
            "## Limitations",
            "",
            "The initial runtime may dequantize packed weights before BF16 matmul. "
            "Disk artifacts are compact, but optimized low-bit CUDA/MPS kernels are separate work.",
            "",
        ]
    )
