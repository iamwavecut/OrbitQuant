from __future__ import annotations

from collections import Counter

import pytest
import torch
from safetensors.torch import load_file, save_file

from orbitquant import OrbitQuantConfig, OrbitQuantLinear, inspect_linear_module_policy

transformers = pytest.importorskip("transformers")


def _tiny_models():
    return {
        "bert": transformers.BertModel(
            transformers.BertConfig(
                hidden_size=32,
                num_hidden_layers=1,
                num_attention_heads=4,
                intermediate_size=64,
                vocab_size=64,
            )
        ),
        "gpt2": transformers.GPT2LMHeadModel(
            transformers.GPT2Config(
                n_embd=32,
                n_layer=1,
                n_head=4,
                n_positions=16,
                n_ctx=16,
                vocab_size=64,
                bos_token_id=1,
                eos_token_id=2,
            )
        ),
        "llama": transformers.LlamaForCausalLM(
            transformers.LlamaConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                vocab_size=64,
            )
        ),
        "t5": transformers.T5ForConditionalGeneration(
            transformers.T5Config(
                d_model=32,
                d_ff=64,
                num_layers=1,
                num_decoder_layers=1,
                num_heads=4,
                d_kv=8,
                vocab_size=64,
            )
        ),
        "vit": transformers.ViTForImageClassification(
            transformers.ViTConfig(
                hidden_size=32,
                num_hidden_layers=1,
                num_attention_heads=4,
                intermediate_size=64,
                image_size=16,
                patch_size=8,
                num_channels=3,
                num_labels=3,
            )
        ),
    }


@pytest.mark.parametrize(
    ("model_name", "quantized_count", "adapter_counts"),
    [
        ("bert", 6, {"Linear": 7}),
        ("gpt2", 4, {"transformers.Conv1D": 4, "Linear": 1}),
        ("llama", 7, {"Linear": 8}),
        ("t5", 16, {"Linear": 17}),
        ("vit", 6, {"Linear": 7}),
    ],
)
def test_universal_policy_covers_transformer_families_without_model_recipes(
    model_name,
    quantized_count,
    adapter_counts,
):
    report = inspect_linear_module_policy(_tiny_models()[model_name], OrbitQuantConfig())

    assert report["target_policy"] == "universal"
    assert report["action_counts"]["orbitquant"] == quantized_count
    assert report["unsupported_linear_module_count"] == 0
    assert report["unclassified_modules"] == []
    assert Counter(item["adapter"] for item in report["modules"]) == adapter_counts


def test_gpt2_conv1d_quantize_save_and_prequantized_restore(tmp_path):
    source_dir = tmp_path / "source"
    quantized_dir = tmp_path / "quantized"
    model = _tiny_models()["gpt2"]
    model.save_pretrained(source_dir)
    config = OrbitQuantConfig(
        block_size=8,
        runtime_mode="dequant_bf16",
        activation_kernel_backend="cpu",
    )

    quantized, loading_info = transformers.GPT2LMHeadModel.from_pretrained(
        source_dir,
        quantization_config=config,
        output_loading_info=True,
    )
    input_ids = torch.tensor([[1, 2, 3]])
    logits = quantized(input_ids=input_ids).logits

    assert not loading_info["missing_keys"]
    assert not loading_info["unexpected_keys"]
    assert isinstance(quantized.transformer.h[0].attn.c_attn, OrbitQuantLinear)
    assert quantized.transformer.h[0].attn.c_attn.source_weight_layout == "in_out"
    assert isinstance(quantized.lm_head, torch.nn.Linear)
    assert torch.isfinite(logits).all()

    quantized.save_pretrained(quantized_dir)
    restored, restored_info = transformers.GPT2LMHeadModel.from_pretrained(
        quantized_dir,
        output_loading_info=True,
    )

    assert not restored_info["missing_keys"]
    assert not restored_info["unexpected_keys"]
    assert isinstance(restored.transformer.h[0].attn.c_attn, OrbitQuantLinear)
    torch.testing.assert_close(restored(input_ids=input_ids).logits, logits)


def test_gpt2_streaming_quantization_accepts_base_model_prefix_checkpoint_keys(tmp_path):
    source_dir = tmp_path / "source"
    model = _tiny_models()["gpt2"]
    model.save_pretrained(source_dir)
    checkpoint_path = source_dir / "model.safetensors"
    checkpoint = load_file(checkpoint_path)
    checkpoint = {
        key.removeprefix("transformer.") if key.startswith("transformer.h.") else key: value
        for key, value in checkpoint.items()
    }
    save_file(checkpoint, checkpoint_path, metadata={"format": "pt"})
    config = OrbitQuantConfig(
        block_size=8,
        runtime_mode="dequant_bf16",
        activation_kernel_backend="cpu",
    )

    quantized, loading_info = transformers.GPT2LMHeadModel.from_pretrained(
        source_dir,
        quantization_config=config,
        output_loading_info=True,
    )

    assert not loading_info["missing_keys"]
    assert not loading_info["unexpected_keys"]
    assert isinstance(quantized.transformer.h[0].attn.c_attn, OrbitQuantLinear)
    assert torch.isfinite(quantized(input_ids=torch.tensor([[1, 2, 3]])).logits).all()


def test_gpt2_debug_no_quant_streaming_loads_rotated_weights(tmp_path):
    source_dir = tmp_path / "source"
    model = _tiny_models()["gpt2"].eval()
    input_ids = torch.tensor([[1, 2, 3]])
    with torch.inference_mode():
        expected = model(input_ids=input_ids).logits
    model.save_pretrained(source_dir)
    config = OrbitQuantConfig(
        block_size=8,
        runtime_mode="debug_no_quant",
        activation_kernel_backend="cpu",
    )

    quantized, loading_info = transformers.GPT2LMHeadModel.from_pretrained(
        source_dir,
        quantization_config=config,
        output_loading_info=True,
    )

    assert not loading_info["missing_keys"]
    assert not loading_info["unexpected_keys"]
    with torch.inference_mode():
        actual = quantized(input_ids=input_ids).logits
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
