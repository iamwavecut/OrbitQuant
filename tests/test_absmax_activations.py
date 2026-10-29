import pytest
import torch
import torch.nn.functional as F

from orbitquant import recipe
from orbitquant.artifacts import (
    load_orbitquant_artifact,
    save_orbitquant_artifact,
    validate_orbitquant_artifact,
)
from orbitquant.config import OrbitQuantConfig
from orbitquant.layers import OrbitQuantLinear
from orbitquant.modeling import quantize_linear_modules


def test_absmax_activations_need_eight_bits():
    with pytest.raises(ValueError, match="activation_bits=8"):
        OrbitQuantConfig(activation_bits=4, activation_quantizer="absmax_int8")
    with pytest.raises(ValueError, match="activation_quantizer"):
        OrbitQuantConfig(activation_quantizer="minmax")


def test_w4a8_recipe_uses_absmax_int8_activations():
    config = recipe("w4a8")

    assert (config.weight_bits, config.activation_bits) == (4, 8)
    assert config.activation_quantizer == "absmax_int8"
    assert recipe("w4a4").activation_quantizer == "lloyd_max"


def test_reference_absmax_path_is_per_token_int8_of_the_rotated_input():
    torch.manual_seed(0)
    linear = torch.nn.Linear(128, 32)
    x = torch.randn(2, 5, 128)
    config = OrbitQuantConfig(
        weight_bits=4,
        activation_bits=8,
        activation_quantizer="absmax_int8",
        runtime_mode="dequant_bf16",
    )
    layer = OrbitQuantLinear.from_linear(linear, config=config, module_name="proj")

    rotated = layer.rotation.apply_to_activations(x)
    scale = rotated.abs().amax(dim=-1, keepdim=True) / 127.0
    quantized = torch.round(rotated / scale).clamp(-127, 127) * scale
    weight = layer._dequantize_weight(device=x.device, dtype=torch.float32, remember=False)
    expected = F.linear(quantized, weight, linear.bias)

    with torch.no_grad():
        output = layer(x)

    assert layer.last_activation_kernel_backend == "torch_absmax_int8"
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)


def test_absmax_layers_build_no_activation_codebook(monkeypatch):
    # Exact 8-bit Lloyd-Max tables take minutes and stop converging at d >= 7680;
    # absmax activations never read one, so none may be requested.
    import orbitquant.layers as layers

    requested = []
    original = layers.get_codebook

    def recording(dim, bits, *args):
        requested.append(bits)
        return original(dim, bits, *args)

    monkeypatch.setattr(layers, "get_codebook", recording)
    linear = torch.nn.Linear(256, 32).requires_grad_(False)
    config = recipe("w4a8", runtime_mode="dequant_bf16", output_gain_correction=True)
    layer = OrbitQuantLinear.from_linear(linear, config=config, module_name="proj")

    assert layer.activation_codebook is None
    assert 8 not in requested
    with torch.no_grad():
        assert torch.isfinite(layer(torch.randn(4, 256))).all()


def test_absmax_int8_activations_are_closer_than_four_bit_codes():
    torch.manual_seed(0)
    linear = torch.nn.Linear(256, 64, bias=False).requires_grad_(False)
    x = torch.randn(32, 256) * torch.linspace(0.2, 3.0, 256)
    exact = linear(x)

    def relative_error(**fields):
        config = OrbitQuantConfig(weight_bits=8, runtime_mode="dequant_bf16", **fields)
        layer = OrbitQuantLinear.from_linear(linear, config=config, module_name="proj")
        return float((layer(x) - exact).pow(2).sum() / exact.pow(2).sum())

    four_bit = relative_error(activation_bits=4)
    absmax = relative_error(activation_bits=8, activation_quantizer="absmax_int8")

    assert absmax < 0.1 * four_bit


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS required")
def test_absmax_int8_runs_on_mps_with_reference_numerics():
    try:
        import orbitquant_packed_matmul
    except Exception:
        pytest.skip("native packed matmul kernel package is not importable")
    if not orbitquant_packed_matmul.supports_device("mps"):
        pytest.skip("the importable native packed matmul variant has no MPS backend")
    torch.manual_seed(0)
    linear = torch.nn.Linear(256, 64)
    x = torch.randn(4, 256)
    config = OrbitQuantConfig(
        weight_bits=4,
        activation_bits=8,
        activation_quantizer="absmax_int8",
        runtime_mode="native_packed_matmul",
    )
    reference = OrbitQuantLinear.from_linear(
        linear,
        config=OrbitQuantConfig(
            weight_bits=4,
            activation_bits=8,
            activation_quantizer="absmax_int8",
            runtime_mode="dequant_bf16",
        ),
        module_name="proj",
    )
    layer = OrbitQuantLinear.from_linear(linear, config=config, module_name="proj").to("mps")

    with torch.no_grad():
        expected = reference(x)
        output = layer(x.to("mps")).cpu()

    torch.testing.assert_close(output, expected, rtol=2e-3, atol=2e-3)


class _ArtifactModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList(
            [torch.nn.ModuleDict({"attn": torch.nn.ModuleDict({"to_q": torch.nn.Linear(64, 32)})})]
        )


def test_absmax_artifact_records_and_restores_the_activation_quantizer(tmp_path):
    torch.manual_seed(0)
    source = _ArtifactModel()
    config = recipe("w4a8", target_policy="auto", runtime_mode="dequant_bf16")
    summary = quantize_linear_modules(source, config)
    save_orbitquant_artifact(
        source,
        tmp_path,
        config=config,
        source_model_id="example/model",
        source_revision="abc123",
        source_license="apache-2.0",
        summary=summary,
    )

    validate_orbitquant_artifact(tmp_path)
    restored = _ArtifactModel()
    manifest = load_orbitquant_artifact(restored, tmp_path)

    assert manifest.activation_quantizer == "absmax_int8"
    layer = restored.transformer_blocks[0]["attn"]["to_q"]
    assert layer.activation_quantizer == "absmax_int8"
    x = torch.randn(3, 64)
    with torch.no_grad():
        torch.testing.assert_close(layer(x), source.transformer_blocks[0]["attn"]["to_q"](x))


def _cuda_triton_available():
    if not torch.cuda.is_available() or torch.version.hip is not None:
        return False
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(not _cuda_triton_available(), reason="CUDA with Triton required")
@pytest.mark.parametrize("weight_bits", [2, 4])
@pytest.mark.parametrize("rows", [7, 4096])
def test_cuda_absmax_int8_gemm_matches_the_reference(weight_bits, rows):
    torch.manual_seed(0)
    linear = torch.nn.Linear(1024, 512).to(torch.bfloat16)
    x = torch.randn(rows, 1024, dtype=torch.bfloat16)
    fields = dict(weight_bits=weight_bits, activation_bits=8, activation_quantizer="absmax_int8")
    reference = OrbitQuantLinear.from_linear(
        linear, config=OrbitQuantConfig(runtime_mode="dequant_bf16", **fields), module_name="p"
    )
    layer = OrbitQuantLinear.from_linear(
        linear, config=OrbitQuantConfig(runtime_mode="triton_packed_matmul", **fields),
        module_name="p",
    ).to("cuda")

    with torch.no_grad():
        expected = reference(x.float())
        output = layer(x.to("cuda")).float().cpu()

    assert layer.last_activation_kernel_backend == "triton_cuda_absmax_int8"
    # The GEMM uses the INT8 surrogate of the weight centroids (docs/kernel-audit.md).
    error = (output - expected).pow(2).sum() / expected.pow(2).sum()
    assert float(error) < 2e-3
