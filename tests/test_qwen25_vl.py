"""Exercise Qwen2.5-VL W4A6 loading with text and real vision-token inputs."""

import pytest
import torch

from orbitquant import OrbitQuantConfig, quantize_model
from orbitquant.layers import OrbitQuantLinear

transformers = pytest.importorskip("transformers")


def _model():
    config = transformers.Qwen2_5_VLConfig(
        text_config={
            "vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
            "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
            "max_position_embeddings": 64, "bos_token_id": 1, "eos_token_id": 2,
            "rope_parameters": {"rope_type": "default", "mrope_section": [1, 1, 2]},
        },
        vision_config={
            "depth": 2, "hidden_size": 32, "intermediate_size": 64, "num_heads": 4,
            "out_hidden_size": 32, "patch_size": 2, "temporal_patch_size": 2,
            "spatial_merge_size": 2, "window_size": 8, "fullatt_block_indexes": [0, 1],
        },
        image_token_id=62, video_token_id=63, vision_start_token_id=61, vision_end_token_id=60,
    )
    return transformers.Qwen2_5_VLForConditionalGeneration(config).eval()


@pytest.mark.parametrize("with_image", [False, True])
@pytest.mark.parametrize("conversion", ["streaming", "postload"])
def test_qwen25_vl_w4a6_conversion_and_packed_reload(tmp_path, with_image, conversion):
    source = _model()
    source.save_pretrained(tmp_path / "source")
    config = OrbitQuantConfig(
        weight_bits=4, activation_bits=6, block_size=8,
        runtime_mode="dequant_bf16", activation_kernel_backend="cpu",
    )
    if conversion == "streaming":
        quantized, info = type(source).from_pretrained(
            tmp_path / "source", quantization_config=config, output_loading_info=True,
        )
        assert not info["missing_keys"] and not info["unexpected_keys"]
    else:
        quantized = source
        quantize_model(quantized, config, quantization_device="cpu")
    assert isinstance(quantized.model.language_model.layers[0].self_attn.q_proj, OrbitQuantLinear)
    assert isinstance(quantized.model.visual.blocks[0].attn.qkv, OrbitQuantLinear)
    assert isinstance(quantized.model.language_model.embed_tokens, torch.nn.Embedding)
    assert isinstance(quantized.lm_head, torch.nn.Linear)
    if with_image:
        inputs = {"input_ids": torch.tensor([[61, 62, 62, 62, 62, 60, 1, 2]]),
                  "pixel_values": torch.randn(16, 24),
                  "image_grid_thw": torch.tensor([[1, 4, 4]])}
    else:
        inputs = {"input_ids": torch.tensor([[1, 3, 4, 2]])}
    inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
    with torch.no_grad():
        expected = quantized(**inputs, output_hidden_states=True, use_cache=False)
    assert torch.isfinite(expected.logits).all()
    assert torch.isfinite(expected.hidden_states[-1]).all()
    quantized.save_pretrained(tmp_path / "packed", max_shard_size="30KB")
    restored, info = type(source).from_pretrained(tmp_path / "packed", output_loading_info=True)
    assert not info["missing_keys"] and not info["unexpected_keys"]
    with torch.no_grad():
        actual = restored(**inputs, output_hidden_states=True, use_cache=False)
    torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)
    torch.testing.assert_close(actual.hidden_states[-1], expected.hidden_states[-1], rtol=0, atol=0)
