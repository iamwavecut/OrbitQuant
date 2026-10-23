import copy

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("triton")
diffusers = pytest.importorskip("diffusers")
if not hasattr(diffusers, "BriaFibo2Transformer2DModel"):
    pytest.skip("diffusers without Fibo 2.0", allow_module_level=True)

from orbitquant.fused.families import adapter_for_class, adapter_for_family, fibo2  # noqa: E402
from orbitquant.fused.groups import swiglu_half_rows  # noqa: E402
from orbitquant.kernels import available_backends  # noqa: E402

# Head width 128 matches the RoPE axes (32 + 48 + 48); 7 heads make the SwiGLU width 2368, which is
# not a whole number of 128-row interleave chunks, as in the real model (9536).
CONFIG = dict(
    dim=896,
    n_heads=7,
    n_layers=3,
    n_refiner_layers=1,
    cap_feat_dim=768,
    injection_layer_ids=(1,),
    perceiver_num_layers=1,
    min_num_gist_tokens=64,
    max_num_gist_tokens=128,
    gist_step=64,
    gist_min_text_len=16,
    gist_max_text_len=64,
)
TARGETS = [
    "noise_refiner.*.attention.to_*",
    "noise_refiner.*.feed_forward.w*",
    "layers.*.attention.to_*",
    "layers.*.feed_forward.w*",
    "layers.*.cross_attention.to_*",
    "perceiver.layers.*.cross_attention.to_*",
    "perceiver.layers.*.self_attention.to_*",
    "perceiver.layers.*.ffn.w*",
]


def _require_cuda():
    if not torch.cuda.is_available() or not available_backends()["triton_cuda"]:
        pytest.skip("CUDA/Triton backend is not available")


def _model():
    torch.manual_seed(0)
    model = diffusers.BriaFibo2Transformer2DModel(**CONFIG).to(torch.bfloat16)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("cross_attn_gate"):
                p.fill_(0.5)
            elif "norm" in name:
                p.copy_(torch.rand_like(p) * 0.5 + 0.75)
            elif "adaLN_modulation" in name:
                p.mul_(0.05)
    return model.eval()


def _inputs(edit):
    g = torch.Generator("cuda").manual_seed(1)
    latents = torch.randn(1, 32, 16, 16, device="cuda", generator=g).to(torch.bfloat16)
    text = torch.randn(1, 6, 40, 768, device="cuda", generator=g).to(torch.bfloat16)
    context = (
        [torch.randn(1, 32, 16, 12, device="cuda", generator=g).to(torch.bfloat16)]
        if edit
        else None
    )
    return dict(
        hidden_states=latents,
        timestep=torch.tensor([0.4], device="cuda"),
        encoder_hidden_states=text,
        context_latents=context,
        return_dict=False,
    )


def _cos(a, b):
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def test_family_is_registered_for_the_transformer_class():
    assert adapter_for_family("fibo2") is fibo2
    assert adapter_for_class("BriaFibo2Transformer2DModel") is fibo2


def test_blocks_and_groups():
    model = _model()
    blocks = dict(fibo2.blocks(model))
    assert blocks == {
        "noise_refiner.0": "block",
        "perceiver.layers.0": "perceiver",
        "layers.0": "block",
        "layers.1": "inject",
        "layers.2": "block",
    }
    assert list(fibo2.groups("block", model.layers[0])) == ["qkv", "attn_out", "ff_in", "ff_out"]
    inject = fibo2.groups("inject", model.layers[1])
    assert [s.path for s in inject["xkv"].sources] == [
        "cross_attention.to_k",
        "cross_attention.to_v",
    ]
    perceiver = fibo2.groups("perceiver", model.perceiver.layers[0])
    assert perceiver["ff_in"].swiglu and [s.path for s in perceiver["ff_in"].sources] == [
        "ffn.w1",
        "ffn.w3",
    ]


def test_swiglu_halves_pad_to_whole_chunks():
    assert swiglu_half_rows(2368) == 2432
    assert swiglu_half_rows(9536) == 9600
    assert swiglu_half_rows(1024) == 1024


@pytest.mark.parametrize("edit", [False, True], ids=["t2i", "edit"])
def test_fused_bf16_groups_match_the_eager_model(edit):
    _require_cuda()
    from orbitquant import OrbitQuantConfig
    from orbitquant.fused import fuse

    model = _model().cuda()
    reference = copy.deepcopy(model)
    fuse(model, dense="bf16", config=OrbitQuantConfig(), attention="sdpa")
    with torch.inference_mode():
        expected = reference(**_inputs(edit))[0]
        actual = model(**_inputs(edit))[0]
    assert actual.shape == expected.shape
    assert _cos(actual, expected) > 0.999


def test_fused_w4a8_model_tracks_the_eager_model():
    _require_cuda()
    from orbitquant import OrbitQuantConfig, quantize_model
    from orbitquant.fused import fuse

    model = _model().cuda()
    reference = copy.deepcopy(model)
    config = OrbitQuantConfig(target_policy="universal", modules_to_convert=TARGETS)
    summary = quantize_model(model, config, quantization_device="cuda")
    assert len(summary.quantized_modules) == 7 * 4 + 4 + 11
    fuse(model, config=config, attention="sdpa")
    assert model.layers[0].oq_fused.ff_in.out_features == 2 * 2432
    with torch.inference_mode():
        expected = reference(**_inputs(True))[0]
        actual = model(**_inputs(True))[0]
    assert _cos(actual, expected) > 0.95
