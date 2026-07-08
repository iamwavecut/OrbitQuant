import torch

from orbitquant.config import OrbitQuantConfig
from orbitquant.eval import get_native_suite
from orbitquant.layers import OrbitQuantLinear
from orbitquant.policies import classify_linear_modules


class PaperPolicyToyDenoiser(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {
                        "attn": torch.nn.ModuleDict(
                            {
                                "to_q": torch.nn.Linear(16, 16),
                                "to_k": torch.nn.Linear(16, 16),
                                "to_v": torch.nn.Linear(16, 16),
                                "to_out": torch.nn.Sequential(torch.nn.Linear(16, 16)),
                            }
                        ),
                        "ff": torch.nn.ModuleDict(
                            {
                                "linear_in": torch.nn.Linear(16, 64),
                                "linear_out": torch.nn.Linear(64, 16),
                            }
                        ),
                        "norm1": torch.nn.ModuleDict({"linear": torch.nn.Linear(16, 96)}),
                    }
                )
            ]
        )
        self.x_embedder = torch.nn.Linear(8, 16)
        self.timestep_embedder = torch.nn.Linear(16, 16)
        self.text_encoder = torch.nn.ModuleDict({"proj": torch.nn.Linear(16, 16)})
        self.vae = torch.nn.ModuleDict({"proj": torch.nn.Linear(16, 16)})
        self.proj_out = torch.nn.Linear(16, 8)


def test_paper_defaults_are_calibration_free_orbitquant_settings():
    config = OrbitQuantConfig()

    assert config.rotation == "rpbh"
    assert config.block_size == "paper"
    assert config.codebook == "lloyd_max"
    assert config.row_norm_dtype == "bfloat16"
    assert config.activation_norm_dtype == "float32"
    assert config.adaln_policy == "int4_rtn"
    assert config.adaln_group_size == 64


def test_paper_native_suites_keep_agreed_image_and_video_settings():
    expected = {
        "flux1-schnell-native": {
            "model_id": "black-forest-labs/FLUX.1-schnell",
            "size": (1024, 1024),
            "steps": 4,
            "guidance": 0.0,
            "bits": ["W4A4", "W3A3", "W2A4", "W2A3"],
            "metric": "geneval",
        },
        "z-image-native": {
            "model_id": "Tongyi-MAI/Z-Image-Turbo",
            "size": (1024, 1024),
            "steps": 10,
            "guidance": 0.0,
            "bits": ["W4A4", "W3A3", "W2A4", "W2A3"],
            "metric": "geneval",
        },
        "wan-native": {
            "model_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            "size": (832, 480),
            "frames": 81,
            "steps": 50,
            "guidance": 5.0,
            "bits": ["W4A6", "W4A4"],
            "metric": "vbench",
        },
    }

    for suite_name, suite_expected in expected.items():
        suite = get_native_suite(suite_name)
        assert suite.model_id == suite_expected["model_id"]
        assert (suite.width, suite.height) == suite_expected["size"]
        assert suite.steps == suite_expected["steps"]
        assert suite.guidance == suite_expected["guidance"]
        assert suite.bit_settings == suite_expected["bits"]
        assert suite.metric == suite_expected["metric"]
        if "frames" in suite_expected:
            assert suite.frames == suite_expected["frames"]


def test_paper_layer_policy_quantizes_projections_and_skips_non_denoiser_modules():
    decisions = classify_linear_modules(
        PaperPolicyToyDenoiser(), OrbitQuantConfig(target_policy="flux")
    )

    for name in (
        "transformer_blocks.0.attn.to_q",
        "transformer_blocks.0.attn.to_k",
        "transformer_blocks.0.attn.to_v",
        "transformer_blocks.0.attn.to_out.0",
        "transformer_blocks.0.ff.linear_in",
        "transformer_blocks.0.ff.linear_out",
    ):
        assert decisions[name].action == "orbitquant"

    assert decisions["transformer_blocks.0.norm1.linear"].action == "adaln_int4_rtn"
    assert decisions["x_embedder"].action == "bf16_skip"
    assert decisions["timestep_embedder"].action == "bf16_skip"
    assert decisions["text_encoder.proj"].action == "bf16_skip"
    assert decisions["vae.proj"].action == "bf16_skip"
    assert decisions["proj_out"].action == "bf16_skip"


def test_paper_activation_path_stores_no_calibration_statistics():
    torch.manual_seed(0)
    source = torch.nn.Linear(16, 8)
    quantized = OrbitQuantLinear.from_linear(
        source,
        config=OrbitQuantConfig(weight_bits=4, activation_bits=4, block_size=8),
        module_name="transformer_blocks.0.attn.to_q",
    )

    assert set(quantized.state_dict()) == {"bias", "packed_weight_indices", "row_norms"}
