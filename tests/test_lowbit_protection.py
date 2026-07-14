import torch

from orbitquant.artifacts.manifest import OrbitQuantManifest
from orbitquant.config import OrbitQuantConfig
from orbitquant.layers import OrbitQuantLinear
from orbitquant.modeling import quantize_model
from orbitquant.policies.generic_dit import PolicyDecision
from orbitquant.policies.lowbit_protection import (
    apply_lowbit_boundary_protection,
    resolve_protected_block_count,
)


def _decisions(names):
    return {name: PolicyDecision(name, "orbitquant", "test") for name in names}


def _block_names(count=10):
    names = []
    for index in range(count):
        names.append(f"layers.{index}.attention.to_q")
        names.append(f"layers.{index}.feed_forward.up")
    names.append("input_proj")
    names.append("llm_cond_proj")
    return names


def test_protection_upgrades_boundary_and_out_of_block_modules():
    config = OrbitQuantConfig(weight_bits=2, activation_bits=4, lowbit_protected_blocks=2)
    decisions = apply_lowbit_boundary_protection(_decisions(_block_names()), config)

    assert decisions["layers.0.attention.to_q"].weight_bits == 4
    assert decisions["layers.1.feed_forward.up"].weight_bits == 4
    assert decisions["layers.8.attention.to_q"].weight_bits == 4
    assert decisions["layers.9.feed_forward.up"].weight_bits == 4
    assert decisions["layers.5.attention.to_q"].weight_bits is None
    assert decisions["input_proj"].weight_bits == 4
    assert decisions["llm_cond_proj"].weight_bits == 4


def test_protection_is_inert_for_three_bit_and_above():
    config = OrbitQuantConfig(weight_bits=3, activation_bits=3)
    decisions = apply_lowbit_boundary_protection(_decisions(_block_names()), config)
    assert all(d.weight_bits is None for d in decisions.values())
    assert resolve_protected_block_count(config) == 0


def test_protection_disabled_explicitly():
    config = OrbitQuantConfig(weight_bits=2, activation_bits=4, lowbit_boundary_protection=0)
    decisions = apply_lowbit_boundary_protection(_decisions(_block_names()), config)
    assert all(d.weight_bits is None for d in decisions.values())


def test_manifest_round_trips_module_bits():
    manifest = OrbitQuantManifest.from_config(
        OrbitQuantConfig(weight_bits=2, activation_bits=4),
        source_model_id="example/model",
        source_revision="rev",
        source_license="apache-2.0",
        quantized_modules=["layers.0.attn"],
        skipped_modules=[],
        module_bits={"layers.0.attn": 4},
    )
    payload = manifest.to_dict()
    assert payload["module_bits"] == {"layers.0.attn": 4}
    restored = OrbitQuantManifest.from_dict(payload)
    assert restored.module_bits == {"layers.0.attn": 4}

    uniform = OrbitQuantManifest.from_config(
        OrbitQuantConfig(),
        source_model_id="example/model",
        source_revision="rev",
        source_license="apache-2.0",
        quantized_modules=[],
        skipped_modules=[],
    )
    assert "module_bits" not in uniform.to_dict()
    assert OrbitQuantManifest.from_dict(uniform.to_dict()).module_bits == {}


class _TinyDiT(torch.nn.Module):
    def __init__(self, blocks=6):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {
                        "attention": torch.nn.ModuleDict(
                            {"to_q": torch.nn.Linear(16, 16)}
                        )
                    }
                )
                for _ in range(blocks)
            ]
        )


def test_quantize_model_applies_mixed_bits_end_to_end():
    torch.manual_seed(0)
    model = _TinyDiT()
    config = OrbitQuantConfig(
        weight_bits=2,
        activation_bits=4,
        block_size=8,
        target_policy="universal",
        lowbit_protected_blocks=1,
        runtime_mode="dequant_bf16",
    )
    quantize_model(model, config, quantization_device=None)

    first = model.layers[0]["attention"]["to_q"]
    middle = model.layers[3]["attention"]["to_q"]
    last = model.layers[5]["attention"]["to_q"]
    assert isinstance(first, OrbitQuantLinear)
    assert first.weight_bits == 4
    assert middle.weight_bits == 2
    assert last.weight_bits == 4
