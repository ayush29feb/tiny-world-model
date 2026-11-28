"""Tests for the tokenizer, the DiT and flow matching."""

import itertools

import pytest

torch = pytest.importorskip("torch")

from games.dino import clip_stream  # noqa: E402
from tiny_world_model.dit import DiT, DiTConfig  # noqa: E402
from tiny_world_model.flow import flow_matching_loss, sample  # noqa: E402
from tiny_world_model.tokenizer import (  # noqa: E402
    Patching,
    patchify,
    to_bool,
    to_signed,
    unpatchify,
)

# A small model on short clips, so the tests run fast on CPU.
TINY = DiTConfig(frames=4, dim=32, depth=2, heads=2)


def real_clips(n: int, frames: int) -> torch.Tensor:
    clips = itertools.islice(clip_stream(seed=0, clip_len=frames - 1, num_bots=8), n)
    return torch.stack([torch.from_numpy(c.frames) for c in clips])


# --- tokenizer -------------------------------------------------------------------


def test_patchify_shapes():
    tokens = patchify(torch.zeros(2, 32, 48, 128))
    assert tokens.shape == (2, 16 * 6 * 16, 128)


def test_patchify_round_trip_is_exact_on_game_clips():
    clips = real_clips(3, frames=32)
    tokens = patchify(to_signed(clips))
    assert torch.equal(to_bool(unpatchify(tokens, 32, 48, 128)), clips)


def test_each_pixel_lands_in_the_right_token():
    clips = torch.zeros(1, 32, 48, 128)
    clips[0, 5, 20, 100] = 1  # frame 5, row 20, column 100
    tokens = patchify(clips)
    # frame 5 -> time slot 2 (offset 1); row 20 -> row slot 2 (offset 4);
    # column 100 -> column slot 12 (offset 4)
    token = (2 * 6 + 2) * 16 + 12
    value = (1 * 8 + 4) * 8 + 4
    assert tokens[0, token, value] == 1
    assert tokens.sum() == 1


def test_clip_size_must_divide_into_patches():
    with pytest.raises(ValueError, match="frames=33"):
        patchify(torch.zeros(1, 33, 48, 128))


def test_signed_round_trip():
    frames = real_clips(1, frames=4)
    signed = to_signed(frames)
    assert set(signed.unique().tolist()) == {-1.0, 1.0}
    assert torch.equal(to_bool(signed), frames)


def test_other_patch_sizes_round_trip():
    patching = Patching(patch_t=1, patch_h=4, patch_w=4)
    x = torch.randn(2, 3, 48, 128)
    tokens = patchify(x, patching)
    assert tokens.shape == (2, 3 * 12 * 32, 16)
    assert torch.equal(unpatchify(tokens, 3, 48, 128, patching), x)


# --- model -----------------------------------------------------------------------


def test_default_model_size():
    model = DiT()
    assert model.config.num_tokens == 1536
    assert 2_000_000 < model.num_params() < 3_000_000


def test_model_output_matches_input_and_starts_at_zero():
    model = DiT(TINY)
    x = torch.randn(2, TINY.num_tokens, 128)
    v = model(x, torch.rand(2))
    assert v.shape == x.shape
    # adaLN-Zero: an untrained model predicts zero velocity.
    assert torch.equal(v, torch.zeros_like(v))


def test_model_rejects_the_wrong_clip_size():
    with pytest.raises(ValueError, match="expected"):
        DiT(TINY)(torch.randn(1, 7, 128), torch.rand(1))


def test_loss_backpropagates_to_every_parameter():
    model = DiT(TINY)
    loss = flow_matching_loss(model, real_clips(2, frames=4))
    loss.backward()
    assert torch.isfinite(loss)
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"no gradient for {missing}"


def test_loss_checks_clip_shape():
    with pytest.raises(ValueError, match="model expects"):
        flow_matching_loss(DiT(TINY), real_clips(1, frames=6))


def test_sample_shape_and_type():
    clips = sample(DiT(TINY), num_clips=2, steps=3)
    assert clips.shape == (2, 4, 48, 128)
    assert clips.dtype == torch.bool


def test_training_reduces_the_loss():
    torch.manual_seed(0)
    model = DiT(TINY)
    clips = real_clips(4, frames=4)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    def avg_loss():
        with torch.no_grad():
            return sum(flow_matching_loss(model, clips).item() for _ in range(10)) / 10

    before = avg_loss()
    for _ in range(60):
        opt.zero_grad()
        flow_matching_loss(model, clips).backward()
        opt.step()
    assert avg_loss() < 0.7 * before
