from __future__ import annotations

import pytest
import torch
from orbitquant_packed_matmul import matmul_int8_rows, quantize_rows_int8, supports_device


def _absmax_scale(values):
    # tensor divisor => true IEEE division; a Python-scalar divisor makes torch's CUDA kernel
    # multiply by a reciprocal, which is 1 ulp off the kernel's __fdiv_rn in some rows
    amax = values.float().abs().amax(-1).clamp_min(1e-12)
    return amax / amax.new_full((), 127.0)


def _int8_rows(weight):
    scale = _absmax_scale(weight)
    rows = (weight.float() / scale[:, None]).round().clamp(-127, 127).to(torch.int8)
    return rows.contiguous(), scale.contiguous()


@pytest.mark.kernels_ci
@pytest.mark.parametrize("rows", [1, 2, 3, 8])
@pytest.mark.parametrize(
    "n,k", [(129, 64), (2048, 2048), (32769, 2048), (151644, 2048), (777, 4096)]
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("bias_enabled", [False, True])
def test_int8_rows_matches_int_mm(rows, n, k, dtype, bias_enabled):
    if not torch.cuda.is_available() or not supports_device("cuda"):
        pytest.skip("CUDA kernel required")
    torch.manual_seed(rows * 31 + n + k)
    x = torch.randn(rows, k, device="cuda").bfloat16()
    w = torch.randn(n, k, device="cuda").bfloat16()
    w8, ws = _int8_rows(w)
    x8, xs = quantize_rows_int8(x)
    xf = x.float()
    xs_ref = _absmax_scale(xf)
    torch.testing.assert_close(xs, xs_ref, rtol=0, atol=0)
    x8_ref = (xf / xs_ref[:, None]).round().clamp(-127, 127).to(torch.int8)
    assert (x8 != x8_ref).sum().item() <= rows * 2  # rintf vs round on exact ties only
    pad_rows = 32
    xpad = torch.zeros((pad_rows, k), device="cuda", dtype=torch.int8)
    xpad[:rows] = x8
    npad = (-n) % 8
    wpad = w8
    if npad:
        wpad = torch.cat([w8, torch.zeros((npad, k), device="cuda", dtype=torch.int8)], 0)
    acc = torch._int_mm(xpad, wpad.t())[:rows, :n]
    bias = torch.randn(n, device="cuda").to(dtype) if bias_enabled else None
    ref = acc.float() * (xs[:, None] * ws[None, :])
    if bias is not None:
        ref = ref + bias.float()
    ref = ref.to(dtype)
    out = matmul_int8_rows(x8, xs, w8, ws, bias=bias, output_dtype=dtype)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


@pytest.mark.kernels_ci
def test_int8_rows_graph_capture_and_contract():
    if not torch.cuda.is_available() or not supports_device("cuda"):
        pytest.skip("CUDA kernel required")
    torch.manual_seed(5)
    x = torch.randn(2, 2048, device="cuda").bfloat16()
    w8, ws = _int8_rows(torch.randn(4096, 2048, device="cuda").bfloat16())

    def run():
        x8, xs = quantize_rows_int8(x)
        return matmul_int8_rows(x8, xs, w8, ws)

    ref = run()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    nine = torch.zeros(9, 2048, device="cuda", dtype=torch.int8)
    with pytest.raises(RuntimeError):
        matmul_int8_rows(nine, torch.ones(9, device="cuda"), w8, ws)
    with pytest.raises(RuntimeError):
        quantize_rows_int8(torch.randn(1, 2046, device="cuda").bfloat16())
