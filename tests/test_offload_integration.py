import pytest
import torch

from orbitquant import OrbitQuantConfig, build_diffusers_pipeline_quantization_config
from orbitquant.layers import OrbitQuantLinear

diffusers = pytest.importorskip("diffusers")
configuration_utils = pytest.importorskip("diffusers.configuration_utils")
pytest.importorskip("accelerate")


class TinyOffloadTransformer(diffusers.ModelMixin, diffusers.ConfigMixin):
    config_name = "config.json"

    @configuration_utils.register_to_config
    def __init__(self, hidden_size: int = 16):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {
                        "attn": torch.nn.ModuleDict(
                            {"to_q": torch.nn.Linear(hidden_size, hidden_size)}
                        )
                    }
                )
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transformer_blocks[0]["attn"]["to_q"](x)


# Make the synthetic component discoverable through the same public Diffusers
# library lookup used by real pipeline components.
TinyOffloadTransformer.__module__ = "diffusers"
diffusers.TinyOffloadTransformer = TinyOffloadTransformer


class TinyOffloadPipeline(diffusers.DiffusionPipeline):
    model_cpu_offload_seq = "transformer"

    def __init__(self, transformer: TinyOffloadTransformer):
        super().__init__()
        self.register_modules(transformer=transformer)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        output = self.transformer(x)
        self.maybe_free_model_hooks()
        return output


def _streaming_pipeline(tmp_path):
    source = TinyOffloadPipeline(TinyOffloadTransformer())
    source.save_pretrained(tmp_path)
    quantization_config = build_diffusers_pipeline_quantization_config(
        OrbitQuantConfig(
            block_size=8,
            runtime_mode="debug_no_activation_quant",
            activation_kernel_backend="cpu",
            weight_row_tile_size=2,
        ),
        components="transformer",
    )
    return TinyOffloadPipeline.from_pretrained(
        tmp_path,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
    )


def _quantized_linear(pipe):
    return pipe.transformer.transformer_blocks[0]["attn"]["to_q"]


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is not available")
def test_streaming_pipeline_model_cpu_offload_repeats_without_pinned_weight_cache(tmp_path):
    pipe = _streaming_pipeline(tmp_path)
    assert isinstance(_quantized_linear(pipe), OrbitQuantLinear)
    pipe.enable_model_cpu_offload(device="mps")

    for _ in range(2):
        output = pipe(torch.randn(2, 3, 16, dtype=torch.bfloat16))
        linear = _quantized_linear(pipe)
        assert torch.isfinite(output).all()
        assert linear.packed_weight_indices.device.type == "cpu"
        assert linear._dequantized_weight_cache is None


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is not available")
def test_streaming_pipeline_sequential_cpu_offload_repeats_and_offloads_packed_state(tmp_path):
    pipe = _streaming_pipeline(tmp_path)
    assert isinstance(_quantized_linear(pipe), OrbitQuantLinear)
    pipe.enable_sequential_cpu_offload(device="mps")

    for _ in range(2):
        output = pipe(torch.randn(2, 3, 16, dtype=torch.bfloat16))
        linear = _quantized_linear(pipe)
        assert torch.isfinite(output).all()
        assert linear.packed_weight_indices.device.type == "meta"
        assert linear.row_norms.device.type == "meta"
        assert linear._dequantized_weight_cache is None
