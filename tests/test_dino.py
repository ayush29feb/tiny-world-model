"""Tests for the dino game engine. No pygame needed."""

import numpy as np
import pytest

from games.bitmap import blit, draw_text, overlaps, sprite, text_width
from games.dino import Action, DinoGame, Obstacle, scripted_action
from games.dino.game import BIRD_HIGH, BIRD_LOW


def only(game: DinoGame, obstacle: Obstacle) -> None:
    """Replace the world's obstacles with one, and stop new ones spawning."""
    game.obstacles = [obstacle]
    game._spawn_timer = 10**9


# --- frames --------------------------------------------------------------------


def test_frame_is_a_binary_bitmap():
    frame = DinoGame(seed=0).reset()
    assert frame.shape == (48, 128)
    assert frame.dtype == bool


def test_frame_packs_to_bits_and_back():
    frame = DinoGame(seed=0).step(Action.NOOP)
    packed = np.packbits(frame)
    assert packed.nbytes == 48 * 128 // 8
    restored = np.unpackbits(packed)[: frame.size].reshape(frame.shape).astype(bool)
    assert np.array_equal(restored, frame)


def test_same_seed_and_actions_give_identical_frames():
    actions = np.random.default_rng(1).integers(0, 3, 400)
    a, b = DinoGame(seed=7), DinoGame(seed=7)
    for act in actions:
        assert np.array_equal(a.step(act), b.step(act))


def test_different_seeds_give_different_games():
    a, b = DinoGame(seed=1), DinoGame(seed=2)
    frames_a = [a.step(Action.NOOP) for _ in range(80)]
    frames_b = [b.step(Action.NOOP) for _ in range(80)]
    assert not all(np.array_equal(x, y) for x, y in zip(frames_a, frames_b, strict=True))


def test_score_is_drawn_top_right():
    game = DinoGame(seed=0)
    for _ in range(40):
        frame = game.step(Action.NOOP)
    assert game.score > 0
    text = f"{game.score:05d}"
    expected = np.zeros_like(frame)
    draw_text(expected, text, frame.shape[1] - 1 - text_width(text), 1)
    region = (slice(1, 6), slice(frame.shape[1] - 1 - text_width(text), frame.shape[1] - 1))
    assert np.array_equal(frame[region], expected[region])


# --- game over -----------------------------------------------------------------


def test_never_jumping_ends_the_game_with_game_over_text():
    game = DinoGame(seed=0)
    frame = game.reset()
    for _ in range(500):
        frame = game.step(Action.NOOP)
        if game.done:
            break
    assert game.done

    expected = np.zeros_like(frame)
    draw_text(expected, "GAME OVER", (128 - text_width("GAME OVER", 2)) // 2, 12, scale=2)
    assert np.array_equal(frame & expected, expected), "GAME OVER missing from frame"


def test_frame_is_frozen_after_game_over():
    game = DinoGame(seed=0)
    while not game.done:
        game.step(Action.NOOP)
    score, last = game.score, game.render()
    for action in Action:
        assert np.array_equal(game.step(action), last)
    assert game.score == score


def test_reset_starts_a_fresh_game():
    game = DinoGame(seed=0)
    while not game.done:
        game.step(Action.NOOP)
    game.reset()
    assert not game.done
    assert game.score == 0
    assert game.obstacles == []


# --- obstacles -----------------------------------------------------------------


@pytest.mark.parametrize("variant", [0, 1, 2])
def test_rock_kills_a_standing_dino_and_a_jump_clears_it(variant):
    for jump in (False, True):
        game = DinoGame(seed=0)
        only(game, Obstacle("rock", variant, x=game.config.dino_x + 11 + 8))
        game.step(Action.JUMP if jump else Action.NOOP)
        for _ in range(40):
            game.step(Action.NOOP)
        assert game.done is not jump


def test_ducking_clears_a_high_bird_but_standing_does_not():
    for duck in (False, True):
        game = DinoGame(seed=0)
        only(game, Obstacle("bird", 0, x=game.config.dino_x + 20, altitude=BIRD_HIGH))
        for _ in range(30):
            game.step(Action.DUCK if duck else Action.NOOP)
        assert game.done is not duck


def test_low_bird_must_be_jumped():
    for action in (Action.NOOP, Action.DUCK):
        game = DinoGame(seed=0)
        only(game, Obstacle("bird", 0, x=game.config.dino_x + 20, altitude=BIRD_LOW))
        for _ in range(30):
            game.step(action)
        assert game.done, f"{action.name} should not clear a low bird"


def test_dino_cannot_double_jump():
    game = DinoGame(seed=0)
    game.step(Action.JUMP)
    peak = 0.0
    for _ in range(40):
        game.step(Action.JUMP)
        peak = max(peak, game.altitude)
    assert peak < 20


@pytest.mark.parametrize("seed", range(5))
def test_a_scripted_player_survives_rocks_and_birds(seed):
    """Every obstacle the game spawns, at every speed, can be cleared."""
    game = DinoGame(seed=seed)
    seen = set()
    for _ in range(3000):
        game.step(scripted_action(game))
        seen |= {(o.kind, o.altitude) for o in game.obstacles}
        assert not game.done, f"bot died at frame {game.tick}, score {game.score}"
    assert {("rock", 0), ("bird", BIRD_LOW), ("bird", BIRD_HIGH)} <= seen
    assert game.speed == game.config.max_speed


# --- bitmap helpers ------------------------------------------------------------


def test_blit_clips_at_the_edges():
    canvas = np.zeros((4, 4), dtype=bool)
    blit(canvas, np.ones((3, 3), dtype=bool), 2, -1)
    assert canvas.sum() == 4  # only the 2x2 corner lands on the canvas


def test_overlaps_is_pixel_perfect():
    diagonal = sprite("#.\n.#")
    assert overlaps(diagonal, 0, 0, diagonal, 0, 0)
    # Bounding boxes overlap here, but no on pixels coincide.
    assert not overlaps(diagonal, 0, 0, sprite(".#\n#."), 0, 0)
