import pytest
import torch

from orbitquant.kernels import available_backends


def _require_cuda():
    if not torch.cuda.is_available() or not available_backends()["triton_cuda"]:
        pytest.skip("CUDA/Triton backend is not available")


def _layer(in_features, out_features, *, bias=False):
    from orbitquant import OrbitQuantConfig
    from orbitquant.layers import OrbitQuantLinear

    torch.manual_seed(0)
    linear = torch.nn.Linear(in_features, out_features, bias=bias, dtype=torch.bfloat16)
    return OrbitQuantLinear.from_linear(
        linear, config=OrbitQuantConfig(), module_name="probe"
    ).cuda()


def _int_mm_reference(a, b):
    rows = a.shape[0]
    padded = torch.zeros(
        (max(32, -(-rows // 32) * 32), a.shape[1]), device=a.device, dtype=torch.int8
    )
    padded[:rows] = a
    return torch._int_mm(padded, b.t())[:rows]


@pytest.mark.parametrize("rows", [1, 33, 2053])
def test_fused_epilogue_gemm_is_bit_identical_to_int_mm_path(rows):
    _require_cuda()
    from orbitquant.kernels import triton_cuda as tc

    layer = _layer(1024, 4096, bias=True)
    x = torch.randn(rows, 1024, device="cuda", dtype=torch.bfloat16)
    act_codes, act_scale, w_codes, w_scale = layer._int8_surrogate_constants(x.device)
    packed, norms = tc.quantize_activations_packed_w4_with_triton(
        x, rotation=layer.rotation, codebook=layer.activation_codebook, eps=layer.activation_eps
    )
    common = dict(
        activation_scale=act_scale,
        weight_scale=w_scale,
        out_features=4096,
        in_features=1024,
        bias=layer.bias.to(torch.bfloat16),
        output_dtype=torch.bfloat16,
        chunk_out_features=2048,
    )
    args = (packed, layer.packed_weight_indices, norms, layer.row_norms, act_codes, w_codes)
    old = tc.matmul_packed_w4a4_with_int_mm(*args, fused_epilogue=False, **common)
    new = tc.matmul_packed_w4a4_with_int_mm(*args, fused_epilogue=True, **common)
    assert torch.equal(old, new)


@pytest.mark.parametrize("dim", [1024, 3072, 6144])
def test_direct_int8_activation_codes_match_packed_codes(dim):
    _require_cuda()
    from orbitquant.kernels import triton_cuda as tc

    layer = _layer(dim, 256)
    x = torch.randn(77, dim, device="cuda", dtype=torch.bfloat16)
    act_codes, _, _, _ = layer._int8_surrogate_constants(x.device)
    packed, norms = tc.quantize_activations_packed_w4_with_triton(
        x, rotation=layer.rotation, codebook=layer.activation_codebook, eps=layer.activation_eps
    )
    direct, direct_norms = tc.quantize_activations_int8_with_triton(
        x,
        rotation=layer.rotation,
        codebook=layer.activation_codebook,
        activation_codes=act_codes,
        eps=layer.activation_eps,
    )
    codes = act_codes.cuda()
    decoded = torch.stack([codes[(packed & 15).long()], codes[(packed >> 4).long()]], -1)
    assert torch.equal(direct, decoded.reshape(77, dim))
    assert torch.equal(direct_norms, norms)


def _gemm_inputs(rows, k, n):
    torch.manual_seed(1)
    a = torch.randint(-127, 128, (rows, k), device="cuda", dtype=torch.int8)
    b = torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8)
    a_s = torch.rand(rows, device="cuda") * 1e-3 + 1e-4
    b_s = torch.rand(n, device="cuda") + 0.5
    return a, b, a_s, b_s


def test_swiglu_epilogue_matches_eager_rounding():
    _require_cuda()
    from orbitquant.kernels import triton_int8_gemm as gemm

    a, b, a_s, b_s = _gemm_inputs(300, 512, 1024)
    alpha = 0.37
    y = (((_int_mm_reference(a, b).float() * a_s[:, None]) * b_s[None, :]) * alpha).to(
        torch.bfloat16
    )
    half = gemm.SWIGLU_INTERLEAVE
    chunks = y.reshape(300, -1, 2, half)
    gate, up = chunks[:, :, 0].reshape(300, -1), chunks[:, :, 1].reshape(300, -1)
    expected = torch.nn.functional.silu(gate) * up
    actual = gemm.matmul_int8_scaled_with_triton(
        a, b, a_s, b_s, alpha=alpha, epilogue=gemm.EPILOGUE_SWIGLU
    )
    assert torch.equal(actual, expected)


def test_residual_gate_and_sigmoid_tail_epilogues_match_eager_rounding():
    _require_cuda()
    from orbitquant.kernels import triton_int8_gemm as gemm

    a, b, a_s, b_s = _gemm_inputs(129, 512, 768)
    y = (((_int_mm_reference(a, b).float() * a_s[:, None]) * b_s[None, :]) * 0.5).to(torch.bfloat16)
    residual = torch.randn(129, 768, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn(768, device="cuda", dtype=torch.bfloat16)
    expected = residual + gate * y
    actual = gemm.matmul_int8_scaled_with_triton(
        a,
        b,
        a_s,
        b_s,
        alpha=0.5,
        epilogue=gemm.EPILOGUE_RESIDUAL_GATE,
        residual=residual.clone(),
        gate=gate,
    )
    assert torch.equal(actual, expected)

    # Gate logits of real layers stay far from the range where the sigmoid is subnormal, which
    # Triton flushes to zero and torch does not.
    alpha = 0.01
    y = (((_int_mm_reference(a, b).float() * a_s[:, None]) * b_s[None, :]) * alpha).to(
        torch.bfloat16
    )
    tail = gemm.matmul_int8_scaled_with_triton(
        a, b, a_s, b_s, alpha=alpha, epilogue=gemm.EPILOGUE_SIGMOID_TAIL, sig_from=512
    )
    assert torch.equal(tail[:, :512], y[:, :512])
    assert torch.equal(tail[:, 512:], torch.sigmoid(y[:, 512:]))


def test_in_place_residual_gate_survives_autotuning():
    _require_cuda()
    from orbitquant.kernels import triton_int8_gemm as gemm

    kernel = gemm._int8_scaled_gemm
    if not hasattr(kernel, "cache") or not hasattr(kernel, "cache_results"):
        pytest.skip("Triton autotuner internals differ")
    a, b, a_s, b_s = _gemm_inputs(1500, 384, 768)
    residual = torch.randn(1500, 768, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn(768, device="cuda", dtype=torch.bfloat16) * 0.1

    def call():
        h = residual.clone()
        return gemm.matmul_int8_scaled_with_triton(
            a, b, a_s, b_s, epilogue=gemm.EPILOGUE_RESIDUAL_GATE, residual=h, gate=gate, out=h
        )

    saved = kernel.cache_results
    kernel.cache.clear()
    kernel.cache_results = False
    try:
        tuned = call()
    finally:
        kernel.cache_results = saved
    assert torch.equal(tuned, call())


def test_rows_int8_matches_int8_row_linear():
    _require_cuda()
    from orbitquant.int8_head import Int8RowLinear
    from orbitquant.kernels import triton_activation as act
    from orbitquant.kernels import triton_int8_gemm as gemm

    torch.manual_seed(2)
    linear = torch.nn.Linear(2048, 512, bias=False, dtype=torch.bfloat16)
    head = Int8RowLinear.from_linear(linear).cuda()
    x = torch.randn(300, 2048, device="cuda", dtype=torch.bfloat16)
    expected = head(x)
    codes, scales = act.quantize_rows_int8_with_triton(x)
    actual = gemm.matmul_int8_scaled_with_triton(
        codes, head.weight_int8, scales, head.scales, scale_mode=1
    )
    assert torch.equal(actual, expected)


def test_qk_norm_rope_matches_diffusers():
    _require_cuda()
    diffusers_embeddings = pytest.importorskip("diffusers.models.embeddings")
    from orbitquant.kernels import triton_activation as act

    torch.manual_seed(3)
    rows, heads, head_dim = 97, 4, 128
    src = torch.randn(rows, 3 * heads * head_dim, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(head_dim, device="cuda") * 0.1
    angles = torch.rand(rows, head_dim // 2, device="cuda", dtype=torch.float64) * 6.28
    cos = angles.cos().repeat_interleave(2, dim=1).float()
    sin = angles.sin().repeat_interleave(2, dim=1).float()
    view = src[:, : heads * head_dim]
    q = view.unflatten(-1, (heads, head_dim))[None]
    normed = torch.nn.functional.rms_norm(q.float(), (head_dim,), weight=weight + 1.0, eps=1e-5)
    expected = diffusers_embeddings.apply_rotary_emb(
        normed.to(torch.bfloat16), (cos, sin), sequence_dim=1
    )
    actual = act.qk_norm_rope_with_triton(view, heads, head_dim, weight, cos, sin, 1e-5)
    # The RMS reduction order differs from torch's kernel; everything after it is exact.
    assert (actual.float() - expected.float()).abs().max() <= 2**-6 * expected.float().abs().max()


def _fake_quantized_blocks(x, block, multiplier, mean=None):
    x = (x.float() - (0 if mean is None else mean)) * multiplier
    out = torch.empty_like(x)
    for start in range(0, x.shape[0], block):
        chunk = x[start : start + block]
        scale = chunk.abs().amax(dim=(0, 2), keepdim=True).clamp_min(1e-20) / 127
        codes = chunk / scale
        out[start : start + block] = (
            torch.where(codes >= 0, codes + 0.5, codes - 0.5).trunc() * scale
        )
    return out


@pytest.mark.parametrize("seq", [64, 1000, 4133])
def test_int8_attention_matches_its_quantization_and_fp32(seq):
    _require_cuda()
    from orbitquant.kernels import triton_attention as attention

    torch.manual_seed(4)
    query = torch.randn(1, seq, 8, 128, device="cuda", dtype=torch.bfloat16) * 2
    # A shared per-channel offset in K is what the mean subtraction is for.
    key = torch.randn(1, seq, 2, 128, device="cuda", dtype=torch.bfloat16) * 2 + 3
    value = torch.randn(1, seq, 2, 128, device="cuda", dtype=torch.bfloat16)
    out = attention.int8_attention_with_triton(query, key, value).float()

    q = _fake_quantized_blocks(query[0], attention.BLOCK_Q, 128**-0.5 * attention.LOG2E)
    k = _fake_quantized_blocks(key[0], attention.BLOCK_KV, 1.0, key[0].float().mean(dim=0))
    v = value[0].to(torch.float16).float()
    logits = torch.einsum("shd,thd->hst", q, k.repeat_interleave(4, dim=1))
    weights = torch.exp2(logits - logits.amax(dim=-1, keepdim=True))
    weights = weights / weights.sum(dim=-1, keepdim=True)
    emulated = torch.einsum("hst,thd->shd", weights, v.repeat_interleave(4, dim=1))[None]
    exact = torch.nn.functional.scaled_dot_product_attention(
        *(t.transpose(1, 2).float() for t in (query, key, value)), enable_gqa=True
    ).transpose(1, 2)

    def rel(a, b):
        return ((a - b).norm() / b.norm()).item()

    # Everything but the FP16 P.V tiles is emulated exactly.
    assert rel(out, emulated) < 5e-3
    # INT8 Q/K itself keeps these peaked scores within a few percent.
    assert rel(out, exact) < 5e-2
