import pytest
import torch
import torch.nn.functional as F

from orbitquant.kernels import available_backends


def _require_cuda():
    if not torch.cuda.is_available() or not available_backends()["triton_cuda"]:
        pytest.skip("CUDA/Triton backend is not available")


def _quantizer(dim, bits=4):
    from orbitquant.codebooks import get_codebook
    from orbitquant.kernels import triton_activation as act
    from orbitquant.rotations import get_rpbh_rotation

    rotation = get_rpbh_rotation(dim=dim, seed=0, block_size="paper")
    return act.ActivationQuantizer(
        rotation, get_codebook(dim, bits, 2), 1e-10, torch.device("cuda")
    )


def _bf(t):
    return t.to(torch.bfloat16)


@pytest.mark.parametrize("norm", ["layer", "rms", "rms_torch"])
@pytest.mark.parametrize("indexed", [False, True])
def test_norm_modulation_prologues_match_eager(norm, indexed):
    _require_cuda()
    from orbitquant.kernels import triton_activation as act

    torch.manual_seed(0)
    rows, dim = 77, 1024
    x = torch.randn(rows, dim, device="cuda", dtype=torch.bfloat16)
    weight = _bf(torch.randn(dim, device="cuda") * 0.1 + 1.0)
    table_rows = 3 if indexed else 1
    scale = _bf(torch.randn(table_rows, dim, device="cuda") * 0.1)
    shift = _bf(torch.randn(table_rows, dim, device="cuda") * 0.1)
    index = (torch.arange(rows, device="cuda") % 3).to(torch.int32) if indexed else None
    if norm == "layer":
        normed = F.layer_norm(x, (dim,), eps=1e-6)
        mode, eps = act.NORM_LAYER, 1e-6
    elif norm == "rms":
        inv = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-5)
        normed = _bf(x * inv) * weight
        mode, eps = act.NORM_RMS, 1e-5
    else:
        normed = F.rms_norm(x, (dim,), weight, eps=1e-5)
        mode, eps = act.NORM_RMS_TORCH, 1e-5
    s = scale[index.long()] if indexed else scale
    t = shift[index.long()] if indexed else shift
    expected = (1 + s) * normed + t
    quantizer = _quantizer(dim)
    actual, _ = quantizer(
        x,
        norm=mode,
        mod=act.MOD_SHIFT_SCALE,
        norm_weight=weight,
        mod_scale=scale if indexed else scale.reshape(-1),
        mod_shift=shift if indexed else shift.reshape(-1),
        mod_index=index,
        rms_eps=eps,
        prologue_only=True,
    )
    diff = (actual.float() - expected.float()).abs()
    # One bf16 ulp where the reduction order differs from the eager kernels.
    assert diff.max().item() <= 2**-6 * expected.float().abs().max().item()
    assert (diff > 0).float().mean().item() < 0.01


@pytest.mark.parametrize(
    ("style", "head_dim", "rotary", "out_dim"),
    [
        ("interleaved", 128, 128, 128),
        ("half", 256, 256, 256),
        ("half", 128, 96, 128),
        ("half", 120, 120, 128),
    ],
)
def test_qk_norm_rope_styles_match_eager(style, head_dim, rotary, out_dim):
    _require_cuda()
    from orbitquant.kernels import triton_activation as act

    torch.manual_seed(1)
    rows, heads = 65, 3
    src = torch.randn(rows, heads * head_dim + 16, device="cuda", dtype=torch.bfloat16)
    weight = _bf(torch.randn(head_dim, device="cuda") * 0.1 + 1.0)
    angles = torch.rand(rows, rotary // 2, device="cuda") * 6.0
    if style == "interleaved":
        cos = angles.cos().repeat_interleave(2, -1)
        sin = angles.sin().repeat_interleave(2, -1)
    else:
        cos = torch.cat([angles.cos(), angles.cos()], -1)
        sin = torch.cat([angles.sin(), angles.sin()], -1)
    x = src[:, : heads * head_dim].unflatten(-1, (heads, head_dim))
    normed = F.rms_norm(x, (head_dim,), weight, eps=1e-6)
    rot, rest = normed[..., :rotary].float(), normed[..., rotary:]
    c, s = cos[:, None], sin[:, None]
    if style == "interleaved":
        pairs = rot.reshape(*rot.shape[:-1], -1, 2)
        rotated = torch.stack([-pairs[..., 1], pairs[..., 0]], -1).flatten(-2)
        out = rot * c + rotated * s
    else:
        half = rotary // 2
        rotated = torch.cat([-rot[..., half:], rot[..., :half]], -1)
        out = rot * c + rotated * s
    expected = torch.cat([out.to(torch.bfloat16), rest], -1)
    actual = act.qk_norm_rope_with_triton(
        src,
        heads,
        head_dim,
        weight,
        cos,
        sin,
        1e-6,
        weight_mode=act.QK_WEIGHT_TORCH,
        style=act.ROPE_INTERLEAVED if style == "interleaved" else act.ROPE_HALF,
        out_dim=out_dim,
    )[0]
    assert actual.shape == (rows, heads, out_dim)
    torch.testing.assert_close(
        actual[..., :head_dim].float(), expected.float(), atol=2e-2, rtol=1e-2
    )
    if out_dim > head_dim:
        assert torch.count_nonzero(actual[..., head_dim:]) == 0


def test_postnorm_residual_matches_eager():
    _require_cuda()
    from orbitquant.kernels import triton_activation as act

    torch.manual_seed(2)
    rows, dim = 50, 768
    res = torch.randn(rows, dim, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(rows, dim, device="cuda", dtype=torch.bfloat16)
    weight = _bf(torch.randn(dim, device="cuda") * 0.1 + 1.0)
    gate = torch.tanh(torch.randn(dim, device="cuda")).to(torch.bfloat16)
    inv = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-5)
    expected = res + gate * (_bf(x * inv) * weight)
    actual = act.postnorm_residual_with_triton(res.clone(), x, weight, gate, 1e-5)
    torch.testing.assert_close(actual.float(), expected.float(), atol=3e-2, rtol=1e-2)


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_lowbit_decode_matches_unpack(bits):
    _require_cuda()
    from orbitquant.kernels import triton_activation as act
    from orbitquant.packing import pack_lowbit, unpack_lowbit

    torch.manual_seed(3)
    out_features, in_features = 96, 512
    indices = torch.randint(0, 1 << bits, (out_features * in_features,), dtype=torch.uint8)
    packed = pack_lowbit(indices, bits=bits).cuda()
    codes = torch.randint(-127, 128, (1 << bits,), dtype=torch.int8, device="cuda")
    decoded = act.decode_lowbit_to_int8_with_triton(packed, codes, bits, out_features, in_features)
    expected = codes[unpack_lowbit(packed.cpu(), bits=bits, length=indices.numel()).long().cuda()]
    assert torch.equal(decoded.reshape(-1), expected)


def test_gelu_and_indexed_gate_epilogues_match_eager():
    _require_cuda()
    from orbitquant.kernels import triton_int8_gemm as gemm

    torch.manual_seed(4)
    rows, k, n = 70, 256, 384
    a = torch.randint(-127, 128, (rows, k), device="cuda", dtype=torch.int8)
    b = torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8)
    a_s = torch.rand(rows, device="cuda") * 1e-3 + 1e-4
    b_s = torch.rand(n, device="cuda") + 0.5
    y = ((a.float() @ b.float().T) * a_s[:, None] * b_s[None, :] * 0.5).to(torch.bfloat16)
    gelu = gemm.matmul_int8_scaled_with_triton(
        a, b, a_s, b_s, alpha=0.5, epilogue=gemm.EPILOGUE_GELU_TANH
    )
    torch.testing.assert_close(
        gelu.float(), F.gelu(y.float(), approximate="tanh"), atol=2e-2, rtol=2e-2
    )
    table = torch.randn(4, n, device="cuda", dtype=torch.bfloat16)
    index = (torch.arange(rows, device="cuda") % 4).to(torch.int32)
    residual = torch.randn(rows, n, device="cuda", dtype=torch.bfloat16)
    out = gemm.matmul_int8_scaled_with_triton(
        a,
        b,
        a_s,
        b_s,
        alpha=0.5,
        epilogue=gemm.EPILOGUE_RESIDUAL_GATE,
        residual=residual.clone(),
        gate=table,
        gate_index=index,
    )
    expected = residual + table[index.long()] * y
    torch.testing.assert_close(out.float(), expected.float(), atol=5e-2, rtol=2e-2)


@pytest.mark.parametrize(("head_dim", "kv_len"), [(128, 300), (256, 700)])
def test_int8_attention_handles_wide_heads_and_separate_kv(head_dim, kv_len):
    _require_cuda()
    from orbitquant.kernels.triton_attention import int8_attention_with_triton

    torch.manual_seed(5)
    q = torch.randn(1, 513, 4, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, kv_len, 2, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, kv_len, 2, head_dim, device="cuda", dtype=torch.bfloat16)
    out = int8_attention_with_triton(q, k, v)
    ref = F.scaled_dot_product_attention(
        q.transpose(1, 2).float(),
        k.transpose(1, 2).float(),
        v.transpose(1, 2).float(),
        enable_gqa=True,
    ).transpose(1, 2)
    error = (out.float() - ref).norm() / ref.norm()
    assert error.item() < 5e-2


def _projection(name, rows, bits, dim=1024):
    from orbitquant import OrbitQuantConfig
    from orbitquant.layers import OrbitQuantLinear

    linear = torch.nn.Linear(dim, rows, bias=False, dtype=torch.bfloat16)
    config = OrbitQuantConfig(weight_bits=bits, activation_bits=4, codebook_version=2)
    return OrbitQuantLinear.from_linear(linear, config=config, module_name=name).cuda()


def test_mixed_width_group_is_one_gemm_matching_its_projections():
    _require_cuda()
    from orbitquant.fused.groups import PackedGroup, RowSource
    from orbitquant.fused.runtime import FusedRuntime

    torch.manual_seed(6)
    root = torch.nn.Module()
    root.q, root.k, root.v = (
        _projection("q", 256, 2),
        _projection("k", 128, 2),
        _projection("v", 128, 3),
    )
    sources = [RowSource("q"), RowSource("k"), RowSource("v")]
    group = PackedGroup.from_sources(root, sources).cuda()
    assert group.segments == ((384, 2), (128, 3))
    runtime = FusedRuntime(
        rotation_seed=0, block_size="paper", codebook_version=2, activation_eps=1e-10
    )
    x = torch.randn(64, 1024, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        fused = runtime.linear(group, runtime.input(x, 1024))
        reference = torch.cat([root.q(x), root.k(x), root.v(x)], dim=1)
    error = (fused.float() - reference.float()).norm() / reference.float().norm()
    assert error.item() < 2e-2

    skeleton = PackedGroup.empty(
        PackedGroup.source_segments([root.q, root.k, root.v], sources),
        1024,
        activation_bits=4,
        codebook_version=2,
        device="meta",
    )
    skeleton.load_state_dict(group.state_dict(), assign=True)
    skeleton.cuda()
    with torch.no_grad():
        again = runtime.linear(skeleton, runtime.input(x, 1024))
    assert torch.equal(again, fused)


@pytest.mark.parametrize("dim", [3360, 13568, 6144, 8192, 16384])
def test_absmax_int8_activations_match_rotated_reference(dim):
    _require_cuda()
    from orbitquant.kernels import triton_activation as act
    from orbitquant.rotations import get_rpbh_rotation

    torch.manual_seed(7)
    rotation = get_rpbh_rotation(dim=dim, seed=0, block_size="paper")
    quantizer = act.ActivationQuantizer(rotation, None, 1e-10, torch.device("cuda"))
    x = torch.randn(37, dim, device="cuda", dtype=torch.bfloat16)
    x[:, 5] *= 40  # an outlier channel
    codes, scales = quantizer(x)
    rotated = rotation.apply_to_activations(x.float())
    expected_scales = rotated.abs().amax(-1) / 127
    torch.testing.assert_close(scales, expected_scales, rtol=1e-5, atol=0)
    expected = torch.round(rotated / expected_scales[:, None]).clamp(-127, 127)
    assert (codes.float() - expected).abs().max().item() <= 1
    assert (codes.float() != expected).float().mean().item() < 1e-3


@pytest.mark.parametrize("dim", [3360, 8192, 16384])
def test_lloyd_max_activation_codes_match_the_rotated_reference(dim):
    _require_cuda()
    from orbitquant.codebooks import get_codebook
    from orbitquant.kernels import triton_activation as act
    from orbitquant.rotations import get_rpbh_rotation

    torch.manual_seed(8)
    rotation = get_rpbh_rotation(dim=dim, seed=0, block_size="paper")
    codebook = get_codebook(dim, 4, 2)
    quantizer = act.ActivationQuantizer(rotation, codebook, 1e-10, torch.device("cuda"))
    x = torch.randn(9, dim, device="cuda", dtype=torch.bfloat16)
    codes, _ = quantizer(x)
    rotated = rotation.apply_to_activations(x.float())
    unit = rotated / (rotated.norm(dim=-1, keepdim=True) + 1e-10)
    index = torch.bucketize(unit, codebook.boundaries.to("cuda", torch.float32))
    expected = quantizer.codes[index]
    assert (codes != expected).float().mean().item() < 1e-3
