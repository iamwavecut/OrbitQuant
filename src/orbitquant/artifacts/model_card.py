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
            f"- Quantization device: `{data['quantization_device']}`",
            f"- Weight quantization backend: `{data['weight_quantization_backend']}`",
            f"- Target policy: `{data['target_policy']}`",
            f"- Rotation: `{data['rotation']}`",
            f"- Rotation seed: `{data['rotation_seed']}`",
            f"- Block size: `{data['block_size']}`",
            f"- Block size policy: `{data['block_size_policy']}`",
            f"- Codebook: `{data['codebook']}`",
            f"- Codebook version: `{data['codebook_version']}`",
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
            "Disk artifacts are compact; current CUDA/MPS kernels optimize selected "
            "activation codebook lookup/rescale stages and packed weight dequantization; "
            "full fused low-bit kernels are separate work.",
            "",
        ]
    )
