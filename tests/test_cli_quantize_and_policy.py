import json
import sys
from types import SimpleNamespace

import pytest
import torch

import orbitquant.cli.main as cli_main
from orbitquant.cli.main import main


def test_cli_quantize_saves_transformer_component_artifact(monkeypatch, capsys, tmp_path):
    class TinyPipeline:
        def __init__(self):
            self.transformer = torch.nn.Module()
            self.transformer.transformer_blocks = torch.nn.ModuleList(
                [
                    torch.nn.ModuleDict(
                        {"attn": torch.nn.ModuleDict({"to_q": torch.nn.Linear(8, 8)})}
                    )
                ]
            )
            self.device = None

        def to(self, device):
            self.device = device
            return self

    pipeline = TinyPipeline()

    class FakeDiffusionPipeline:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            assert model_id == "example/model"
            assert kwargs["revision"] == "main"
            assert kwargs["torch_dtype"] is torch.float32
            return pipeline

    monkeypatch.setitem(
        sys.modules, "diffusers", SimpleNamespace(DiffusionPipeline=FakeDiffusionPipeline)
    )
    monkeypatch.setattr(
        cli_main,
        "inspect_model_metadata",
        lambda model_id, revision=None: {
            "sha": "abc123",
            "license": "apache-2.0",
        },
    )

    assert (
        main(
            [
                "quantize",
                "--model-id",
                "example/model",
                "--revision",
                "main",
                "--output",
                str(tmp_path),
                "--component",
                "transformer",
                "--target-policy",
                "generic_dit",
                "--weight-bits",
                "4",
                "--activation-bits",
                "4",
                "--block-size",
                "4",
                "--activation-kernel-backend",
                "cpu",
                "--device",
                "cpu",
                "--dtype",
                "float32",
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert output["artifact_dir"] == str(tmp_path)
    assert output["quantization_staging_mode"] == "streaming"
    assert output["synchronize_per_module"] is False
    assert output["device_transfer_seconds"] >= 0.0
    assert output["module_device_transfer_count"] >= 0
    assert output["source_linear_device_counts"]
    assert output["artifact_save_elapsed_seconds"] >= 0.0
    assert output["quantized_modules"] == ["transformer_blocks.0.attn.to_q"]
    assert (tmp_path / "model.safetensors").exists()
    assert (tmp_path / "orbitquant_manifest.json").exists()
    assert (tmp_path / "SHA256SUMS").exists()
    quantization_config = json.loads((tmp_path / "quantization_config.json").read_text())
    assert quantization_config["activation_kernel_backend"] == "cpu"


def test_cli_quantize_with_suite_uses_named_native_pipeline(monkeypatch, capsys, tmp_path):
    class TinyPipeline:
        def __init__(self):
            self.transformer = torch.nn.Module()
            self.transformer.transformer_blocks = torch.nn.ModuleList(
                [
                    torch.nn.ModuleDict(
                        {"attn": torch.nn.ModuleDict({"to_q": torch.nn.Linear(8, 8)})}
                    )
                ]
            )

        def to(self, device):
            return self

    class FakeFlux2KleinPipeline:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            assert model_id == "black-forest-labs/FLUX.2-klein-4B"
            assert kwargs["torch_dtype"] is torch.float32
            return TinyPipeline()

    class WrongDiffusionPipeline:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            raise AssertionError("suite-specific pipeline should be used")

    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(
            Flux2KleinPipeline=FakeFlux2KleinPipeline,
            DiffusionPipeline=WrongDiffusionPipeline,
        ),
    )
    monkeypatch.setattr(
        cli_main,
        "inspect_model_metadata",
        lambda model_id, revision=None: {
            "sha": "abc123",
            "license": "apache-2.0",
        },
    )

    assert (
        main(
            [
                "quantize",
                "--suite",
                "flux2-native",
                "--output",
                str(tmp_path),
                "--component",
                "transformer",
                "--weight-bits",
                "4",
                "--activation-bits",
                "4",
                "--block-size",
                "4",
                "--device",
                "cpu",
                "--dtype",
                "float32",
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    quantization_config = json.loads((tmp_path / "quantization_config.json").read_text())
    assert output["source_model_id"] == "black-forest-labs/FLUX.2-klein-4B"
    assert quantization_config["target_policy"] == "flux2"


def test_cli_inspect_policy_with_suite_writes_module_inventory(monkeypatch, capsys, tmp_path):
    calls = []

    class FakeFlux2Transformer2DModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = torch.nn.Module()
            self.double_stream_modulation_img = torch.nn.ModuleDict(
                {"linear": torch.nn.Linear(8, 16)}
            )
            self.transformer_blocks = torch.nn.ModuleList(
                [
                    torch.nn.ModuleDict(
                        {"attn": torch.nn.ModuleDict({"to_q": torch.nn.Linear(8, 8)})}
                    )
                ]
            )

        @classmethod
        def load_config(cls, model_id, **kwargs):
            assert model_id == "black-forest-labs/FLUX.2-klein-4B"
            assert kwargs["subfolder"] == "transformer"
            assert kwargs["local_files_only"] is False
            calls.append({"method": "load_config", "kwargs": kwargs})
            return {"hidden_size": 8}

        @classmethod
        def from_config(cls, config):
            assert config == {"hidden_size": 8}
            calls.append({"method": "from_config"})
            return cls()

    class WrongDiffusionPipeline:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            raise AssertionError("inspect-policy default must not load full pipeline weights")

    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(
            Flux2Transformer2DModel=FakeFlux2Transformer2DModel,
            DiffusionPipeline=WrongDiffusionPipeline,
        ),
    )
    output_path = tmp_path / "flux2-policy-inventory.json"

    assert (
        main(
            [
                "inspect-policy",
                "--suite",
                "flux2-native",
                "--dtype",
                "float32",
                "--output",
                str(output_path),
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    saved = json.loads(output_path.read_text())

    assert output["output"] == str(output_path)
    assert output["linear_module_count"] == 2
    assert output["quantized_module_count"] == 1
    assert output["adaln_module_count"] == 1
    assert output["skipped_module_count"] == 0
    assert "modules" not in output
    assert saved["source_model_id"] == "black-forest-labs/FLUX.2-klein-4B"
    assert saved["suite"] == "flux2-native"
    assert saved["component"] == "transformer"
    assert saved["load_mode"] == "config"
    assert saved["pipeline_class"] is None
    assert saved["component_class"] == "FakeFlux2Transformer2DModel"
    assert saved["target_policy"] == "flux2"
    assert saved["action_counts"] == {
        "orbitquant": 1,
        "adaln_int4_rtn": 1,
        "bf16_skip": 0,
    }
    assert saved["quantized_modules"] == ["transformer_blocks.0.attn.to_q"]
    assert saved["adaln_modules"] == ["double_stream_modulation_img.linear"]
    assert calls == [
        {
            "method": "load_config",
            "kwargs": {"subfolder": "transformer", "local_files_only": False},
        },
        {"method": "from_config"},
    ]


@pytest.mark.parametrize(
    ("suite", "transformer_class", "model_id", "target_policy", "module_family"),
    [
        (
            "flux2-native",
            "Flux2Transformer2DModel",
            "black-forest-labs/FLUX.2-klein-4B",
            "flux2",
            "flux",
        ),
        (
            "flux1-schnell-native",
            "FluxTransformer2DModel",
            "black-forest-labs/FLUX.1-schnell",
            "flux",
            "flux",
        ),
        (
            "z-image-native",
            "ZImageTransformer2DModel",
            "Tongyi-MAI/Z-Image-Turbo",
            "z_image",
            "z_image",
        ),
        (
            "wan-native",
            "WanTransformer3DModel",
            "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            "wan",
            "wan",
        ),
    ],
)
def test_cli_inspect_policy_config_mode_covers_all_native_target_suites(
    monkeypatch,
    capsys,
    tmp_path,
    suite,
    transformer_class,
    model_id,
    target_policy,
    module_family,
):
    calls = []

    def build_module(self):
        if module_family == "wan":
            self.blocks = torch.nn.ModuleList(
                [
                    torch.nn.ModuleDict(
                        {"attn1": torch.nn.ModuleDict({"to_q": torch.nn.Linear(8, 8)})}
                    )
                ]
            )
        elif module_family == "z_image":
            self.layers = torch.nn.ModuleList(
                [
                    torch.nn.ModuleDict(
                        {"attention": torch.nn.ModuleDict({"to_q": torch.nn.Linear(8, 8)})}
                    )
                ]
            )
        else:
            self.transformer_blocks = torch.nn.ModuleList(
                [
                    torch.nn.ModuleDict(
                        {"attn": torch.nn.ModuleDict({"to_q": torch.nn.Linear(8, 8)})}
                    )
                ]
            )

    def load_config(cls, requested_model_id, **kwargs):
        assert requested_model_id == model_id
        assert kwargs["subfolder"] == "transformer"
        calls.append(("load_config", transformer_class))
        return {"hidden_size": 8}

    def from_config(cls, config):
        assert config == {"hidden_size": 8}
        calls.append(("from_config", transformer_class))
        return cls()

    def init(self):
        torch.nn.Module.__init__(self)
        build_module(self)

    fake_transformer_cls = type(
        transformer_class,
        (torch.nn.Module,),
        {
            "__init__": init,
            "load_config": classmethod(load_config),
            "from_config": classmethod(from_config),
        },
    )

    class WrongDiffusionPipeline:
        @classmethod
        def from_pretrained(cls, requested_model_id, **kwargs):
            raise AssertionError("config inspect-policy must not load full pipeline weights")

    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(
            **{
                transformer_class: fake_transformer_cls,
                "DiffusionPipeline": WrongDiffusionPipeline,
            }
        ),
    )
    output_path = tmp_path / f"{suite}-policy-inventory.json"

    assert main(["inspect-policy", "--suite", suite, "--output", str(output_path)]) == 0

    output = json.loads(capsys.readouterr().out)
    saved = json.loads(output_path.read_text())

    assert output["output"] == str(output_path)
    assert saved["suite"] == suite
    assert saved["source_model_id"] == model_id
    assert saved["component"] == "transformer"
    assert saved["load_mode"] == "config"
    assert saved["pipeline_class"] is None
    assert saved["component_class"] == transformer_class
    assert saved["target_policy"] == target_policy
    assert saved["action_counts"]["orbitquant"] == 1
    assert saved["quantized_modules"]
    assert calls == [
        ("load_config", transformer_class),
        ("from_config", transformer_class),
    ]
