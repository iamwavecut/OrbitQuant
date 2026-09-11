from __future__ import annotations

import pytest
import torch
from orbitquant_packed_matmul import matmul_packed_w4a4_int8, supports_device


@pytest.mark.kernels_ci
@pytest.mark.parametrize(
    "rows,n,k",
    [
        (1, 129, 64),
        (8, 3, 64),
        (1, 129, 512),
        (8, 129, 1024),
        (2, 1024, 2048),
        (8, 2048, 6144),
        (1, 257, 16384),
        (9, 129, 128),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "offset,bias_enabled,k_major",
    [(0, False, False), (1, True, False), (3, False, False), (0, True, True)],
)
def test_decode_against_integer_oracle(rows, n, k, dtype, offset, bias_enabled, k_major):
    if not torch.cuda.is_available() or not supports_device("cuda"):
        pytest.skip("CUDA kernel required")
    torch.manual_seed(42)
    x = torch.randint(0, 256, (rows * k // 2 + offset,), dtype=torch.uint8)[offset:]
    w = torch.randint(0, 256, (n * k // 2 + offset,), dtype=torch.uint8)[offset:]
    # Exercise every code, including signed INT8 extremes and mixed nibble signs.
    ac = torch.tensor(
        [-128, 127, -1, 0, 1, -64, 64, -32, 32, -16, 16, -8, 8, -4, 4, 2], dtype=torch.int8
    )
    wc = ac.flip(0)

    def unpack(values, count):
        values = values.reshape(count, k // 2)
        return torch.stack((values & 15, values >> 4), -1).reshape(count, k).long()

    sums = ac.long()[unpack(x, rows)] @ wc.long()[unpack(w, n)].T

    # Transfer whole storage to retain the deliberately unaligned device view.
    def device_slice(values):
        storage = torch.zeros(values.numel() + offset, dtype=torch.uint8)
        storage[offset:] = values
        return storage.cuda()[offset:]

    xd = device_slice(x).reshape(rows, k // 2)
    wd = device_slice(w)
    if k_major:
        wd = wd.reshape(n, k // 2).T.contiguous()
    xn = torch.rand(rows, device="cuda")
    wn = torch.rand(n, device="cuda", dtype=torch.bfloat16)
    bias = torch.rand(n, device="cuda", dtype=dtype) if bias_enabled else None
    # Binary-exact scales isolate the integer accumulation and norm epilogue.
    expected = sums.cuda().float() * (xn[:, None] * wn.float()[None, :] * 0.001953125)
    if bias is not None:
        # CUDA fuses the accumulator scaling and bias addition into one FMA.
        scale = xn[:, None] * wn.float()[None, :] * 0.001953125
        expected = (sums.cuda().float().double() * scale.double() + bias.double()).float()
    expected = expected.to(dtype)
    args = (xd, wd, xn, wn, ac.cuda(), wc.cuda())
    kwargs = dict(
        activation_scale=0.03125,
        weight_scale=0.0625,
        out_features=n,
        in_features=k,
        bias=bias,
        output_dtype=dtype,
        weight_k_major=k_major,
    )
    torch.testing.assert_close(matmul_packed_w4a4_int8(*args, **kwargs), expected, rtol=0, atol=0)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            matmul_packed_w4a4_int8(*args, **kwargs)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = matmul_packed_w4a4_int8(*args, **kwargs)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, expected, rtol=0, atol=0)
