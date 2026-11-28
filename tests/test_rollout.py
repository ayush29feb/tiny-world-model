"""Tests for generating clips on the fly, with and without a DataLoader."""

import itertools

import numpy as np
import pytest

from games.dino import Bot, clip_stream


def take(stream, n):
    return list(itertools.islice(stream, n))


def test_clip_shapes_and_types():
    clip = next(clip_stream(seed=0, clip_len=16, num_bots=8))
    assert clip.frames.shape == (17, 48, 128)
    assert clip.frames.dtype == bool
    assert clip.actions.shape == (16,)
    assert clip.actions.dtype == np.int64
    assert set(clip.actions.tolist()) <= {0, 1, 2}
    assert clip.done.shape == (17,)


def test_stream_is_reproducible_from_its_seed():
    a, b = take(clip_stream(seed=3, num_bots=8), 20), take(clip_stream(seed=3, num_bots=8), 20)
    for x, y in zip(a, b, strict=True):
        assert np.array_equal(x.frames, y.frames)
        assert np.array_equal(x.actions, y.actions)


def test_different_seeds_give_different_clips():
    a, b = next(clip_stream(seed=1, num_bots=8)), next(clip_stream(seed=2, num_bots=8))
    assert not np.array_equal(a.frames, b.frames)


def test_a_bots_clips_continue_where_the_last_one_ended():
    bot = Bot(np.random.default_rng(0), max_noise=0.0, max_episode_frames=3000)
    first, second = bot.record(8), bot.record(8)
    assert not first.done[-1]
    assert np.array_equal(first.frames[-1], second.frames[0])
    assert second.start_tick == first.start_tick + 8


def test_clips_start_at_random_points_in_the_game():
    clips = take(clip_stream(seed=0, max_episode_frames=3000), 64)
    ticks = np.array([c.start_tick for c in clips])
    # Even the very first batch covers early, middle and late game.
    assert (ticks < 1000).any()
    assert ((ticks >= 1000) & (ticks < 2000)).any()
    assert (ticks >= 2000).any()


def test_consecutive_samples_come_from_different_games():
    clips = take(clip_stream(seed=0, num_bots=64), 64)
    follow_on = sum(
        np.array_equal(a.frames[-1], b.frames[0]) for a, b in itertools.pairwise(clips)
    )
    # Picking the same bot twice in a row happens about 1 time in 64.
    assert follow_on <= 4, f"{follow_on} of 63 neighbouring clips are from the same game"


def test_frames_freeze_after_game_over():
    for clip in take(clip_stream(seed=0, num_bots=8, max_noise=1.0), 200):
        if clip.done.any():
            crash = int(np.argmax(clip.done))
            assert clip.done[crash:].all(), "done must stay True once set"
            for frame in clip.frames[crash + 1 :]:
                assert np.array_equal(frame, clip.frames[crash])
            return
    pytest.fail("no crash in 200 clips of random play")


def test_default_data_covers_the_whole_game_and_includes_crashes():
    clips = take(clip_stream(seed=0, clip_len=32), 1500)
    ticks = np.array([c.start_tick for c in clips])
    assert (ticks >= 1500).mean() > 0.25, "too few clips from late in the game"
    assert (ticks < 500).mean() < 0.4, "too many clips from the slow opening"
    assert sum(bool(c.done.any()) for c in clips) >= 5, "no crashes to learn from"


def test_noisier_bots_crash_more():
    calm = take(clip_stream(seed=0, num_bots=8, max_noise=0.01), 300)
    noisy = take(clip_stream(seed=0, num_bots=8, max_noise=0.5), 300)
    crashes = [sum(bool(c.done.any()) for c in clips) for clips in (calm, noisy)]
    assert crashes[1] > crashes[0]


# --- PyTorch DataLoader ----------------------------------------------------------

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader  # noqa: E402

from games.dino.dataset import DinoClips  # noqa: E402


def test_loader_batches():
    loader = DataLoader(DinoClips(seed=0, clip_len=8, num_bots=4), batch_size=4)
    batch = next(iter(loader))
    assert batch["frames"].shape == (4, 9, 48, 128)
    assert batch["frames"].dtype == torch.bool
    assert batch["actions"].shape == (4, 8)
    assert batch["done"].shape == (4, 9)
    assert batch["start_tick"].shape == (4,)


def test_iterating_again_replays_the_same_clips():
    loader = DataLoader(DinoClips(seed=5, clip_len=8, num_bots=4), batch_size=4)
    assert torch.equal(next(iter(loader))["frames"], next(iter(loader))["frames"])


def test_each_worker_plays_its_own_games():
    loader = DataLoader(
        DinoClips(seed=0, clip_len=8, num_bots=4), batch_size=4, num_workers=2
    )
    # With two workers, batches alternate between them: 0, 1, 0, 1, ...
    from_worker_0, from_worker_1 = take(iter(loader), 2)
    for a in from_worker_0["frames"]:
        for b in from_worker_1["frames"]:
            assert not torch.equal(a, b), "two workers produced the same clip"
