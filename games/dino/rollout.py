"""Clips of the dino game, grabbed from bots that are always playing.

Picture a room of bots, each playing its own game. Asking for a clip picks one bot
at random and records what happens on its screen for the next `clip_len` frames.
Because the bots are all at different points in their games, every clip starts
at a random moment: early, slow games, fast late ones, and crashes.

Pure numpy. `games.dino.dataset` wraps this for a PyTorch DataLoader.

A clip is `clip_len` actions and the `clip_len + 1` frames around them:
frame[t] --action[t]--> frame[t + 1].
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from games.dino import sprites
from games.dino.game import BIRD_HIGH, Action, DinoGame

DINO_WIDTH = sprites.DINO_STAND.shape[1]


def scripted_action(game: DinoGame) -> Action:
    """A simple player that never dies: jump rocks and low birds, duck high birds."""
    c = game.config
    ahead = [o for o in game.obstacles if o.x + o.sprite(0).shape[1] > c.dino_x]
    if not ahead:
        return Action.NOOP
    nearest = min(ahead, key=lambda o: o.x)
    gap = nearest.x - (c.dino_x + DINO_WIDTH)
    if nearest.kind == "bird" and nearest.altitude == BIRD_HIGH:
        return Action.DUCK if gap < game.speed * 6 else Action.NOOP
    return Action.JUMP if gap <= game.speed * 4 else Action.NOOP


@dataclass(frozen=True)
class Clip:
    frames: np.ndarray  # (clip_len + 1, height, width) bool
    actions: np.ndarray  # (clip_len,) int64
    done: np.ndarray  # (clip_len + 1,) bool, True once the game is over
    start_tick: int  # how many frames into its game the clip starts


class Bot:
    """One bot playing one game after another.

    Each game gets a noise level between 0 and `max_noise`: the chance, each frame,
    that the bot presses a random key instead of the right one. Low noise gives
    long runs up to top speed; high noise gives early crashes.

    A game ends when the dino crashes or after `max_episode_frames`, so a bot that
    makes no mistakes cannot stay in one game forever.
    """

    def __init__(self, rng: np.random.Generator, max_noise: float, max_episode_frames: int):
        self.rng = rng
        self.max_noise = max_noise
        self.max_episode_frames = max_episode_frames
        self.game = DinoGame()
        self._new_game()

    def _new_game(self) -> None:
        self.frame = self.game.reset(seed=int(self.rng.integers(2**63)))
        self.noise = self.rng.uniform(0, self.max_noise)

    def _act(self) -> Action:
        if self.rng.random() < self.noise:
            return Action(int(self.rng.integers(len(Action))))
        return scripted_action(self.game)

    def _game_over(self) -> bool:
        return self.game.done or self.game.tick >= self.max_episode_frames

    def fast_forward(self, frames: int) -> None:
        """Play without recording, starting new games as old ones end."""
        for _ in range(frames):
            if self._game_over():
                self._new_game()
            self.frame = self.game.step(self._act())

    def record(self, clip_len: int) -> Clip:
        """Record the next `clip_len` frames of play.

        If the dino crashes partway, the clip is filled out with the frozen
        game-over frame, so the data also shows that nothing moves after a crash.
        The next clip from this bot then starts a new game.
        """
        if self._game_over():
            self._new_game()
        start_tick = self.game.tick
        frames, actions, done = [self.frame], [], [self.game.done]
        for _ in range(clip_len):
            action = self._act()
            self.frame = self.game.step(action)
            frames.append(self.frame)
            actions.append(action)
            done.append(self.game.done)
        return Clip(
            frames=np.stack(frames),
            actions=np.array(actions, dtype=np.int64),
            done=np.array(done),
            start_tick=start_tick,
        )


def clip_stream(
    seed: int,
    clip_len: int = 32,
    num_bots: int = 64,
    max_noise: float = 0.02,
    max_episode_frames: int = 3000,
) -> Iterator[Clip]:
    """Yield clips forever, each from a randomly chosen bot. Deterministic from `seed`.

    `max_noise` trades crashes against coverage of the whole game. Every random key
    press risks a crash, and bots that crash a lot keep restarting, so their clips
    pile up in the slow opening. Measured over 3000 clips:

        max_noise   clips from late game (>1500 frames)   clips with a crash
          0.3                    15%                              6.2%
          0.05                   22%                              2.5%
          0.02                   35%                              1.2%

    At startup every bot fast-forwards a random number of frames, so the bots are
    spread across all points in a game from the very first clip. That costs about
    num_bots * max_episode_frames / 2 frames of play once (a second or two).
    """
    rng = np.random.default_rng(seed)
    bots = [Bot(rng, max_noise, max_episode_frames) for _ in range(num_bots)]
    for bot in bots:
        bot.fast_forward(int(rng.integers(max_episode_frames)))

    while True:
        yield bots[int(rng.integers(num_bots))].record(clip_len)
