"""MiniMax-H3-Image: tabulated AdaLN modulation reproduces the original transformer forward."""

import pytest
import torch

diffusers = pytest.importorskip("diffusers")
if not hasattr(diffusers, "MiniMaxH3ImageTransformer2DModel"):
    pytest.skip("diffusers without MiniMaxH3ImageTransformer2DModel", allow_module_level=True)

from orbitquant.fused.families import minimax_h3 as family  # noqa: E402

TIMESTEPS = [0.0, 0.25, 0.5, 0.999]


def _model():
    torch.manual_seed(0)
    return diffusers.MiniMaxH3ImageTransformer2DModel(
        num_attention_heads=2,
        attention_head_dim=32,
        hidden_size=48,
        num_layers=2,
        num_refiner_layers=1,
        ffn_dim=64,
        in_channels=4,
        text_dim=16,
        freq_dim=32,
        time_embed_hidden_dim=48,
        time_embed_dim=24,
        rope_freq_dim=4,
        num_pdd_heads=2,
    ).eval()


def _inputs(timestep):
    image_rows, text_rows = 8, 3
    seq = image_rows + text_rows
    grid = torch.meshgrid(torch.arange(2), torch.arange(4), indexing="ij")
    grid = torch.stack(grid, -1).reshape(-1, 2)
    positions = torch.zeros(seq, 3, dtype=torch.long)
    positions[text_rows:, 1:] = grid
    positions[:text_rows, 0] = torch.arange(text_rows)
    return dict(
        hidden_states=torch.randn(1, image_rows, 16),
        encoder_hidden_states=torch.randn(1, text_rows, 16),
        timestep=torch.tensor(timestep, dtype=torch.float32),
        timestep_indices=torch.tensor([0] * text_rows + [len(timestep) - 1] * image_rows),
        token_tags=torch.tensor([1] * text_rows + [0] * image_rows),
        position_ids=positions,
        image_indices=torch.arange(text_rows, seq),
        text_indices=torch.arange(text_rows),
        pdd_step_index=1,
        return_dict=False,
    )


@torch.no_grad()
def test_tabulated_forward_matches_original():
    model = _model()
    calls = [_inputs([0.25]), _inputs([0.5, 0.999])]
    reference = [model(**kw)[0] for kw in calls]
    keys = family.tabulate_modulation(model, TIMESTEPS)
    assert keys == torch.tensor(sorted(TIMESTEPS), dtype=torch.float32).tolist()
    assert not hasattr(model, "time_embedder")
    assert all(isinstance(b.adaln_proj, family.ModulationTable) for b in model.transformer_blocks)
    for kw, ref in zip(calls, reference, strict=True):
        torch.testing.assert_close(model(**kw)[0], ref, rtol=1e-5, atol=1e-5)


@torch.no_grad()
def test_tables_load_into_empty_skeleton():
    model = _model()
    family.tabulate_modulation(model, TIMESTEPS)
    state = model.state_dict()
    fresh = _model()
    family.add_modulation_tables(fresh, len(TIMESTEPS))
    fresh.load_state_dict(state, strict=True)
    family.install(fresh)
    kw = _inputs([0.5])
    torch.testing.assert_close(fresh(**kw)[0], model(**kw)[0])


@torch.no_grad()
def test_unknown_timestep_raises():
    model = _model()
    family.tabulate_modulation(model, TIMESTEPS)
    with pytest.raises(ValueError, match="modulation table"):
        model(**_inputs([0.3]))
