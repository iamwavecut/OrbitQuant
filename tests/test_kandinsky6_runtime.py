import pytest
import torch

from orbitquant.runtime.kandinsky6 import (
    Kandinsky6Int8AttnProcessor,
    install_kandinsky6_attention,
    install_kandinsky6_pointwise,
)

k6 = pytest.importorskip("diffusers.models.transformers.transformer_kandinsky6")


def _model():
    model = torch.nn.Module()
    model.visual_transformer_blocks = torch.nn.ModuleList()
    for _ in range(3):
        block = torch.nn.Module()
        block.video_dec_block = torch.nn.Module()
        block.video_dec_block.self_attention = k6.Kandinsky6Attention(128, 64)
        model.visual_transformer_blocks.append(block)
    return model.eval()


def _attention(model, index=0):
    return model.visual_transformer_blocks[index].video_dec_block.self_attention


def test_install_skips_outlier_norms_and_preserves_custom_processors():
    model = _model()
    with torch.no_grad():
        _attention(model, 1).query_norm.weight[0] = 10

    class CustomProcessor(k6.Kandinsky6AttnProcessor):
        pass

    custom = CustomProcessor()
    _attention(model, 2).set_processor(custom)
    original = _attention(model).processor
    report = install_kandinsky6_attention(model)
    assert len(report.installed) == 1
    assert len(report.skipped) == 2
    assert isinstance(_attention(model).processor, Kandinsky6Int8AttnProcessor)
    assert _attention(model, 2).processor is custom
    assert install_kandinsky6_attention(model) is report
    with pytest.raises(ValueError, match="restore"):
        install_kandinsky6_attention(model, min_tokens=128)
    report.restore()
    report.restore()
    assert _attention(model).processor is original
    assert _attention(model, 2).processor is custom


def test_restore_keeps_a_later_user_replacement():
    model = _model()
    report = install_kandinsky6_attention(model)
    replacement = k6.Kandinsky6AttnProcessor()
    _attention(model).set_processor(replacement)
    report.restore()
    assert _attention(model).processor is replacement


def test_cpu_fallback_preserves_outputs_and_gradients():
    torch.manual_seed(7)
    model = _model()
    attn = _attention(model)
    x = torch.randn(2, 11, 128, requires_grad=True)
    reference = attn(x)
    expected_gradient = torch.autograd.grad(reference.square().sum(), x)[0]
    install_kandinsky6_attention(model, min_tokens=1)
    output = attn(x)
    gradient = torch.autograd.grad(output.square().sum(), x)[0]
    assert torch.equal(reference, output)
    assert torch.equal(expected_gradient, gradient)


@pytest.mark.parametrize("min_tokens", [0, -1, 1.5, True])
def test_rejects_invalid_token_threshold(min_tokens):
    with pytest.raises(ValueError, match="positive integer"):
        install_kandinsky6_attention(_model(), min_tokens=min_tokens)


def test_custom_backend_is_not_replaced():
    model = _model()
    attn = _attention(model)
    attn.processor._attention_backend = "_native_flash"
    original = attn.processor
    report = install_kandinsky6_attention(model)
    assert len(report.installed) == 2
    assert attn.processor is original


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_cuda_fast_attention_and_masked_fallback(dtype):
    torch.manual_seed(7)
    model = _model().to(device="cuda", dtype=dtype)
    attn = _attention(model)
    x = torch.randn(1, 257, 128, device="cuda", dtype=dtype)
    original = attn.processor
    with torch.inference_mode():
        reference = attn(x)
        install_kandinsky6_attention(model, min_tokens=1)
        output = attn(x)
        relative = (reference.float() - output.float()).norm() / reference.float().norm()
        assert torch.isfinite(output).all()
        assert relative < 0.025
        mask = torch.ones(1, 1, 257, 257, device="cuda", dtype=torch.bool)
        mask[..., -17:] = False
        masked_reference = original(attn, x, attn_mask=mask)
        masked_output = attn(x, attn_mask=mask)
        assert torch.equal(masked_reference, masked_output)


def test_pointwise_installation_is_local_and_restorable():
    model = _model()
    model.text_block = k6.Kandinsky6TransformerEncoderBlock(128, 128, 256, 64).eval()
    x = torch.randn(1, 11, 128)
    time = torch.randn(1, 128)
    rope = torch.eye(2).expand(11, 1, 32, 2, 2).contiguous()
    original_helper = k6.apply_gate_sum
    original_forward = model.text_block.forward
    original_processor = model.text_block.attn.processor
    with torch.inference_mode():
        reference = model.text_block(x, time, rope)
        report = install_kandinsky6_pointwise(model)
        assert install_kandinsky6_pointwise(model) is report
        assert report.blocks == ["text_block"]
        assert k6.apply_gate_sum is original_helper
        assert torch.equal(model.text_block(x, time, rope), reference)
        report.restore()
        assert model.text_block.forward == original_forward
        assert model.text_block.attn.processor is original_processor


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("width", [896, 1792])
def test_pointwise_cuda_preserves_fp32_arithmetic_and_final_rounding(width):
    from orbitquant.kernels import triton_kandinsky6 as fused

    torch.manual_seed(19)
    with torch.inference_mode():
        x = torch.randn(1, 257, width, device="cuda", dtype=torch.bfloat16)
        normalized = torch.randn(x.shape, device="cuda", dtype=torch.float32)
        scale = torch.randn(1, width, device="cuda", dtype=torch.float32)
        shift = torch.randn_like(scale)
        assert torch.equal(
            fused.scale_shift(normalized, x, scale, shift),
            k6.apply_scale_shift(normalized, x, scale, shift),
        )
        out = torch.randn_like(x)
        assert torch.equal(fused.gate_sum(x, out, scale), k6.apply_gate_sum(x, out, scale))
        query = x.unflatten(-1, (width // 64, 64))
        angles = torch.randn(257, 32, device="cuda")
        cos, sin = angles.cos(), angles.sin()
        rope = torch.stack([cos, -sin, sin, cos], -1).reshape(257, 1, 32, 2, 2)
        assert torch.equal(fused.rotary(query, rope), k6.apply_rotary(query, rope))


def test_projection_reuse_matches_stock_self_and_cross_attention():
    from orbitquant.runtime.kandinsky6 import install_kandinsky6_projection_reuse

    model = _model()
    attn = _attention(model)
    x = torch.randn(1, 11, 128)
    text = torch.randn(1, 7, 128)
    mask = torch.tensor([[True, True, True, False, False, False, False]])
    with torch.inference_mode():
        expected_self = attn(x)
        expected_cross = attn(x, encoder_hidden_states=text, attn_mask=mask)
        original = attn.processor
        report = install_kandinsky6_projection_reuse(model)
        assert install_kandinsky6_projection_reuse(model) is report
        assert len(report.installed) == 3
        assert torch.equal(attn(x), expected_self)
        assert torch.equal(attn(x, encoder_hidden_states=text, attn_mask=mask), expected_cross)
        report.restore()
        assert attn.processor is original


@pytest.mark.parametrize(
    "cross_gates,fix_modulation,audio",
    [(False, False, True), (True, False, True), (True, True, True), (True, True, False)],
)
def test_norm_fusion_preserves_decoder_contract_on_cpu(cross_gates, fix_modulation, audio):
    from orbitquant.runtime.kandinsky6 import install_kandinsky6_norm_fusion

    audio_dim = 64 if cross_gates else 128
    audio_time = 64 if fix_modulation else 128
    block = k6.Kandinsky6FusedTransformerDecoderBlock(
        128,
        128,
        256,
        64,
        audio_dim,
        audio_time,
        audio_dim * 2,
        64,
        ca_rope=True,
        cross_gates=cross_gates,
        fix_modulation=fix_modulation,
    ).eval()
    model = torch.nn.Module()
    model.visual_transformer_blocks = torch.nn.ModuleList([block])
    vis = torch.randn(1, 17, 128, requires_grad=True)
    aud = torch.randn(1, 7, audio_dim) if audio else None
    text_v, text_a = torch.randn(1, 3, 128), torch.randn(1, 3, audio_dim)
    times = (torch.randn(1, 128), torch.randn(1, audio_time))
    rope_v = torch.eye(2).expand(17, 1, 32, 2, 2).contiguous()
    rope_a = torch.eye(2).expand(7, 1, 32, 2, 2).contiguous()
    args = (vis, aud, text_v, text_a, times, rope_v, rope_a)
    expected = block(*args)
    expected_grad = torch.autograd.grad(expected[0].square().sum(), vis)[0]
    original = block.forward
    report = install_kandinsky6_norm_fusion(model)
    assert install_kandinsky6_norm_fusion(model) is report
    actual = block(*args)
    actual_grad = torch.autograd.grad(actual[0].square().sum(), vis)[0]
    assert torch.equal(actual[0], expected[0])
    assert (actual[1] is None) if not audio else torch.equal(actual[1], expected[1])
    assert torch.equal(actual_grad, expected_grad)
    report.restore()
    assert block.forward == original


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("width", [896, 1792])
def test_norm_modulation_cuda_error_is_bounded(width):
    from orbitquant.kernels.triton_kandinsky6 import norm_modulate

    with torch.inference_mode():
        x = torch.randn(1, 257, width, device="cuda", dtype=torch.bfloat16)
        scale = torch.randn(1, width, device="cuda", dtype=torch.float32)
        shift = torch.randn_like(scale)
        norm = torch.nn.LayerNorm(width, elementwise_affine=False).cuda()
        expected = k6.apply_scale_shift(norm(x.float()), x, scale, shift)
        actual = norm_modulate(x, norm, scale, shift)
        assert actual.dtype == expected.dtype
        assert torch.isfinite(actual).all()
        relative = (actual.float() - expected.float()).norm() / expected.float().norm()
        assert relative < 1e-4


def test_qk_norm_preserves_cpu_gradients_hooks_and_restoration():
    from orbitquant.runtime.kandinsky6 import install_kandinsky6_qk_norm

    model = _model()
    norm = _attention(model).query_norm
    original = norm.forward
    x = torch.randn(1, 17, 2, 64, requires_grad=True)
    expected = norm(x)
    gradient = torch.autograd.grad(expected.sum(), x)[0]
    report = install_kandinsky6_qk_norm(model)
    assert install_kandinsky6_qk_norm(model) is report
    actual = norm(x)
    assert torch.equal(actual, expected)
    assert torch.equal(torch.autograd.grad(actual.sum(), x)[0], gradient)
    report.restore()
    assert norm.forward == original


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("eps", [None, 1e-6])
def test_qk_norm_matches_torch_cuda_and_preserves_module_hooks(eps):
    from packaging.version import Version

    from orbitquant.runtime.kandinsky6 import install_kandinsky6_qk_norm

    if Version(torch.__version__.split("+", 1)[0]).release[:2] != (2, 12):
        pytest.skip("Torch 2.12 numerical contract")
    model = _model().to(device="cuda", dtype=torch.bfloat16)
    norm = _attention(model).query_norm
    norm.eps = eps
    with torch.inference_mode():
        torch.manual_seed(99)
        norm.weight.normal_(1, 0.2)
        x = torch.randn(1, 257, 2, 64, device="cuda", dtype=torch.bfloat16)
        expected = norm(x)
        report = install_kandinsky6_qk_norm(model)
        assert len(report.installed) == 6
        assert torch.equal(norm(x), expected)
        handle = norm.register_forward_hook(lambda module, args, output: output + 1)
        assert torch.equal(norm(x), expected + 1)
        handle.remove()
        report.restore()


def test_qk_norm_leaves_other_torch_versions_untouched(monkeypatch):
    from orbitquant.runtime.kandinsky6 import install_kandinsky6_qk_norm

    model = _model()
    original = _attention(model).query_norm.forward
    monkeypatch.setattr(torch, "__version__", "2.11.0")
    report = install_kandinsky6_qk_norm(model)
    assert not report.installed
    assert _attention(model).query_norm.forward == original
    report.restore()
