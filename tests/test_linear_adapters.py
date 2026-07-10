from __future__ import annotations

import torch

from orbitquant import OrbitQuantConfig, OrbitQuantLinear, register_linear_adapter
from orbitquant.adaln import RTNInt4Linear
from orbitquant.modeling import inspect_linear_module_policy, quantize_model


class CustomTransposedLinear(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weight = torch.nn.Parameter(torch.randn(input_size, output_size))
        self.bias = torch.nn.Parameter(torch.randn(output_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.addmm(self.bias, x.reshape(-1, self.input_size), self.weight).reshape(
            *x.shape[:-1], self.output_size
        )


class CustomTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.stages = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {
                        "projection": CustomTransposedLinear(8, 8),
                        "conditioning": torch.nn.Linear(8, 16),
                    }
                )
            ]
        )
        self.auxiliary = torch.nn.Linear(8, 8)
        self.output_projection = torch.nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stages[0]["projection"](x)


def test_custom_linear_adapter_and_explicit_recipe_overrides():
    register_linear_adapter(
        CustomTransposedLinear,
        weight_layout="in_out",
        in_features_attr="input_size",
        out_features_attr="output_size",
    )
    model = CustomTransformer()
    config = OrbitQuantConfig(
        block_size=8,
        runtime_mode="dequant_bf16",
        activation_kernel_backend="cpu",
        modules_to_convert=["stages.*.projection"],
        modules_to_use_adaln=["stages.*.conditioning"],
    )
    report = inspect_linear_module_policy(model, config)

    assert report["unsupported_linear_module_count"] == 0
    assert report["modules"][0]["adapter"] == "CustomTransposedLinear"
    assert report["quantized_modules"] == ["stages.0.projection"]
    assert report["adaln_modules"] == ["stages.0.conditioning"]
    assert report["skipped_modules"] == ["auxiliary", "output_projection"]

    quantize_model(model, config, quantization_device="cpu")

    assert isinstance(model.stages[0]["projection"], OrbitQuantLinear)
    assert model.stages[0]["projection"].source_weight_layout == "in_out"
    assert isinstance(model.stages[0]["conditioning"], RTNInt4Linear)
    assert isinstance(model.auxiliary, torch.nn.Linear)
    assert isinstance(model.output_projection, torch.nn.Linear)
    assert torch.isfinite(model(torch.randn(2, 3, 8))).all()


class UnregisteredLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.in_features = 8
        self.out_features = 8
        self.weight = torch.nn.Parameter(torch.randn(8, 8))


class UnfamiliarTransformerNames(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_mixer = torch.nn.Linear(8, 8)
        self.token_router = torch.nn.Linear(8, 8)
        self.lm_head = torch.nn.Linear(8, 8)


def test_inspection_reports_unregistered_linear_candidates():
    model = torch.nn.ModuleDict({"blocks": torch.nn.ModuleList([UnregisteredLinear()])})

    report = inspect_linear_module_policy(model, OrbitQuantConfig(block_size=8))

    assert report["unsupported_linear_module_count"] == 1
    assert report["unsupported_linear_modules"] == [
        {
            "name": "blocks.0",
            "module_type": "UnregisteredLinear",
            "weight_shape": [8, 8],
        }
    ]


def test_universal_policy_does_not_depend_on_known_block_names():
    report = inspect_linear_module_policy(
        UnfamiliarTransformerNames(), OrbitQuantConfig(block_size=8)
    )

    assert report["quantized_modules"] == ["feature_mixer", "token_router"]
    assert report["skipped_modules"] == ["lm_head"]
    assert report["unclassified_modules"] == []
