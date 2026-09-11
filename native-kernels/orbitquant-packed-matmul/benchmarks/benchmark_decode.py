import hashlib
import json
import os
import statistics
from pathlib import Path

import orbitquant_packed_matmul as kernel
import torch

results = []
torch.manual_seed(0)
for m, n, k in [
    (1, 1024, 2048),
    (1, 2048, 2048),
    (1, 4096, 2048),
    (1, 12288, 2048),
    (1, 2048, 6144),
    (2, 4096, 4096),
    (8, 8192, 8192),
    (16, 2048, 2048),
]:
    x = torch.randint(256, (m, k // 2), dtype=torch.uint8, device="cuda")
    w = torch.randint(256, (n * k // 2,), dtype=torch.uint8, device="cuda")
    xn = torch.rand(m, device="cuda")
    wn = torch.rand(n, device="cuda", dtype=torch.bfloat16)
    ac = torch.randint(-127, 128, (16,), device="cuda", dtype=torch.int8)
    wc = ac.flip(0).contiguous()

    def f(x=x, w=w, xn=xn, wn=wn, ac=ac, wc=wc, n=n, k=k):
        return kernel.matmul_packed_w4a4_int8(
            x,
            w,
            xn,
            wn,
            ac,
            wc,
            activation_scale=0.03,
            weight_scale=0.04,
            out_features=n,
            in_features=k,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(5):
            out = f()
    torch.cuda.current_stream().wait_stream(stream)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(16):
            out = f()
    samples = []
    for _repeat in range(5):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(50):
            g.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / 800)
    results.append(
        dict(
            m=m,
            n=n,
            k=k,
            ms=statistics.median(samples),
            samples=samples,
            sha256=hashlib.sha256(out.cpu().view(torch.uint8).numpy().tobytes()).hexdigest(),
        )
    )
    print(results[-1], flush=True)
name = "legacy" if os.getenv("ORBITQUANT_W4A4_DISABLE_GEMV") == "1" else "gemv"
Path("orbitquant-" + name + ".json").write_text(json.dumps(results, indent=2))
