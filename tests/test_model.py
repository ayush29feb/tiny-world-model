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


# --- schedule and validation loss -------------------------------------------------

from tiny_world_model.flow import (  # noqa: E402
    VALIDATION_TS,
    Schedule,
    sampling_times,
    shift_t,
    validation_loss,
)


def test_shift_moves_times_toward_noise_and_keeps_the_ends():
    t = torch.tensor([0.0, 0.5, 1.0])
    assert torch.allclose(shift_t(t, 1.0), t)
    assert torch.allclose(shift_t(t, 3.0), torch.tensor([0.0, 0.25, 1.0]))
    grid = torch.linspace(0, 1, 101)
    assert torch.all(shift_t(grid, 3.0) <= grid + 1e-6)
    assert torch.all(torch.diff(shift_t(grid, 3.0)) > 0), "shift must keep the order"


@pytest.mark.parametrize(
    "schedule, median, heavy_noise",
    [
        (Schedule("uniform", shift=1.0), 0.50, 0.30),
        (Schedule("logit_normal", shift=1.0), 0.50, 0.20),
        (Schedule(), 0.25, 0.60),  # the default: logit-normal shifted by 3
    ],
)
def test_schedule_concentrates_training_where_expected(schedule, median, heavy_noise):
    t = schedule.sample(200_000, "cpu", generator=torch.Generator().manual_seed(0))
    assert t.min() > 0 and t.max() < 1
    assert t.median().item() == pytest.approx(median, abs=0.01)
    assert (t < 0.3).float().mean().item() == pytest.approx(heavy_noise, abs=0.01)


def test_unknown_schedule_is_rejected():
    with pytest.raises(ValueError, match="unknown dist"):
        Schedule("cosine").sample(4, "cpu")


def test_sampling_times_run_from_noise_to_data():
    times = sampling_times(20, 3.0)
    assert times[0] == 0 and times[-1] == 1 and len(times) == 21
    assert torch.all(torch.diff(times) > 0)
    assert (times < 0.25).sum() > (times > 0.75).sum(), "more steps at high noise"


def test_validation_loss_is_deterministic():
    torch.manual_seed(0)
    model = DiT(TINY)
    for p in model.out.parameters():
        p.data.normal_(0, 0.1)  # make the output non-zero
    frames = real_clips(2, frames=4)
    noise = torch.randn(2, TINY.num_tokens, 128)
    first = validation_loss(model, frames, noise)
    torch.randn(100)  # disturb the global RNG: the result must not depend on it
    assert validation_loss(model, frames, noise) == first
    assert len(VALIDATION_TS) == 8 and VALIDATION_TS[0] == 0.0625


def test_untrained_validation_loss_is_the_velocity_size():
    # An untrained model predicts zero, so the loss is the mean of (x1 - x0)^2 = 1 + 1.
    frames = real_clips(2, frames=4)
    noise = torch.randn(2, TINY.num_tokens, 128, generator=torch.Generator().manual_seed(0))
    assert validation_loss(DiT(TINY), frames, noise) == pytest.approx(2.0, abs=0.05)


def test_sampling_with_a_shift_still_gives_bool_clips():
    clips = sample(DiT(TINY), num_clips=2, steps=4, shift=3.0)
    assert clips.shape == (2, 4, 48, 128) and clips.dtype == torch.bool
