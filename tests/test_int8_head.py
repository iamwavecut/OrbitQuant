from __future__ import annotations

import pytest
import torch
from torch import nn

from orbitquant import (
    Int8RowLinear,
    absmax_int8_scales,
    quantize_int8_rows,
    quantize_output_heads,
)


def test_absmax_int8_scales_is_true_division():
    values = torch.randn(64, 2048)
    scales = absmax_int8_scales(values)
    ref = (values.abs().amax(-1).clamp_min(1e-12).double() / 127.0).float()
    torch.testing.assert_close(scales, ref, rtol=0, atol=0)
    assert scales.dtype == torch.float32 and scales.shape == (64,)


def test_quantize_int8_rows_shapes_and_range():
    weight = torch.randn(37, 64)
    rows, scales = quantize_int8_rows(weight)
    assert rows.dtype == torch.int8 and rows.shape == (37, 64) and scales.shape == (37,)
    assert rows.abs().max().item() <= 127
    recon = rows.float() * scales[:, None]
    step = (weight.abs().amax(-1) / 127.0).max().item()
    assert (recon - weight).abs().max().item() <= step * 0.5 + 1e-6


@pytest.mark.parametrize("rows", [1, 3, 40])
def test_int8_head_cpu_close_to_linear(rows):
    torch.manual_seed(1)
    linear = nn.Linear(64, 200)
    head = Int8RowLinear.from_linear(linear, module_name="lm_head")
    x = torch.randn(rows, 64)
    ref = linear(x)
    out = head(x)
    assert out.dtype == torch.bfloat16 and out.shape == ref.shape
    err = (out.float() - ref).abs().max().item()
    assert err < 0.05 * ref.abs().max().item()
    assert head.last_path == "cpu_float"


def test_quantize_output_heads_replaces_in_place():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Linear(16, 16)
            self.lm_head = nn.Linear(16, 50, bias=False)

    model = Tiny()
    replaced = quantize_output_heads(model, ("lm_head",))
    assert isinstance(model.lm_head, Int8RowLinear) and "lm_head" in replaced
    assert model.lm_head.bias is None and model.lm_head.out_features == 50


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("rows", [1, 2, 8, 33])
def test_int8_head_cuda_paths_agree(rows):
    torch.manual_seed(7)
    linear = nn.Linear(2048, 4104, bias=True).cuda().bfloat16()
    head = Int8RowLinear.from_linear(linear, module_name="lm_head").cuda()
    x = torch.randn(rows, 2048, device="cuda").bfloat16()
    out = head(x)
    assert out.dtype == torch.bfloat16 and out.shape == (rows, 4104)
    # explicit _int_mm reference with the same epilogue formula
    xf = x.float()
    xs = absmax_int8_scales(xf)
    x8 = (xf / xs[:, None]).round().clamp(-127, 127).to(torch.int8)
    pad = torch.zeros((max(32, rows), 2048), device="cuda", dtype=torch.int8)
    pad[:rows] = x8
    acc = torch._int_mm(pad, head.weight_int8.t())[:rows]
    scaled = acc.float() * (xs[:, None] * head.scales[None, :]) + head.bias.float()
    ref = scaled.to(torch.bfloat16)
    if head.last_path == "native_int8_rows":
        torch.testing.assert_close(out, ref, rtol=0, atol=0)
    else:
        torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
    ref_linear = linear(x).float()
    assert (out.float() - ref_linear).abs().max().item() < 0.05 * ref_linear.abs().max().item()
