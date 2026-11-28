"""The dino runner game, as a headless engine that renders binary bitmaps.

The engine has no display and no input handling: you call `step(action)` and get
back the next frame as a 2D bool array. That makes it easy to record games as
(frame, action) sequences. Given the same seed and the same actions, a game
always produces exactly the same frames.

    game = DinoGame(seed=0)
    frame = game.reset()            # (48, 128) bool array
    frame = game.step(Action.JUMP)
    game.done, game.score
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from games.bitmap import blit, draw_text, overlaps, text_width
from games.dino import sprites


class Action(IntEnum):
    NOOP = 0
    JUMP = 1
    DUCK = 2  # duck on the ground; fall faster in the air


@dataclass(frozen=True)
class DinoConfig:
    width: int = 128
    height: int = 48
    ground_y: int = 42  # row of the ground line; sprites rest on the row above it
    dino_x: int = 10

    start_speed: float = 2.0  # pixels per frame
    max_speed: float = 4.0
    acceleration: float = 0.001  # speed gained per frame

    jump_velocity: float = 3.6  # peak height is about v^2 / 2g = 18 pixels
    gravity: float = 0.36
    fast_fall: float = 3.0  # gravity multiplier when ducking in the air

    score_per_pixel: float = 0.125
    bird_min_score: int = 100  # birds only appear once the score reaches this
    bird_chance: float = 0.35
    first_obstacle_frame: int = 45
    min_gap_frames: int = 28  # frames between obstacle spawns
    max_gap_frames: int = 64
    ground_dots: int = 22


# How far above the ground each bird flies, measured to the bottom of its sprite.
BIRD_LOW = 2  # must be jumped over
BIRD_HIGH = 8  # clears a ducking dino by one pixel, but hits a standing one


@dataclass
class Obstacle:
    kind: str  # "rock" or "bird"
    variant: int  # index into sprites.ROCKS; unused for birds
    x: float  # left edge, in pixels
    altitude: int = 0  # pixels between the ground and the bottom of the sprite

    def sprite(self, tick: int) -> np.ndarray:
        if self.kind == "bird":
            return sprites.BIRD[(tick // 6) % 2]
        return sprites.ROCKS[self.variant]


class DinoGame:
    def __init__(self, seed: int | None = None, config: DinoConfig | None = None) -> None:
        self.config = config or DinoConfig()
        self.rng = np.random.default_rng(seed)
        self.reset()

    # ------------------------------------------------------------------ api

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Start a new game. Passing a seed makes this game reproducible."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        c = self.config

        self.tick = 0
        self.done = False
        self.distance = 0.0
        self.speed = c.start_speed

        self.altitude = 0.0  # dino's height above the ground
        self.velocity = 0.0
        self.ducking = False

        self.obstacles: list[Obstacle] = []
        self._spawn_timer = c.first_obstacle_frame

        dot_rows = np.arange(c.ground_y + 2, c.height)
        self._dots = np.stack(
            [
                self.rng.uniform(0, c.width, c.ground_dots),
                self.rng.choice(dot_rows, c.ground_dots),
            ],
            axis=1,
        )
        return self.render()

    @property
    def score(self) -> int:
        return int(self.distance * self.config.score_per_pixel)

    @property
    def airborne(self) -> bool:
        return self.altitude > 0 or self.velocity > 0

    def step(self, action: int | Action) -> np.ndarray:
        """Advance one frame and return it. After game over, the frame is frozen."""
        if self.done:
            return self.render()
        action = Action(action)
        c = self.config

        self._move_dino(action)

        self.speed = min(c.max_speed, c.start_speed + c.acceleration * self.tick)
        self.distance += self.speed
        for ob in self.obstacles:
            ob.x -= self.speed
        self.obstacles = [ob for ob in self.obstacles if ob.x + ob.sprite(0).shape[1] > 0]

        self._spawn_timer -= 1
        if self._spawn_timer <= 0:
            self._spawn()

        self._dots[:, 0] -= self.speed
        gone = self._dots[:, 0] < 0
        self._dots[gone, 0] += c.width + self.rng.uniform(0, 8, gone.sum())

        self.tick += 1
        self.done = self._collided()
        return self.render()

    def render(self) -> np.ndarray:
        """Draw the current state as a (height, width) bool array."""
        c = self.config
        frame = np.zeros((c.height, c.width), dtype=bool)

        frame[c.ground_y, :] = True
        cols = self._dots[:, 0].astype(int)
        visible = cols < c.width
        frame[self._dots[visible, 1].astype(int), cols[visible]] = True

        for spr, x, y in self._placements():
            blit(frame, spr, x, y)

        score = f"{min(self.score, 99999):05d}"
        draw_text(frame, score, c.width - 1 - text_width(score), 1)

        if self.done:
            message = "GAME OVER"
            draw_text(frame, message, (c.width - text_width(message, 2)) // 2, 12, scale=2)
        return frame

    # ------------------------------------------------------------ internals

    def _move_dino(self, action: Action) -> None:
        c = self.config
        if action == Action.JUMP and not self.airborne:
            self.velocity = c.jump_velocity
        if self.airborne:
            self.altitude += self.velocity
            self.velocity -= c.gravity * (c.fast_fall if action == Action.DUCK else 1.0)
            if self.altitude <= 0:
                self.altitude = 0.0
                self.velocity = 0.0
        self.ducking = action == Action.DUCK and not self.airborne

    def _dino_sprite(self) -> np.ndarray:
        if self.ducking:
            return sprites.DINO_DUCK[(self.tick // 4) % 2]
        if self.airborne or self.done:
            return sprites.DINO_STAND
        return sprites.DINO_RUN[(self.tick // 4) % 2]

    def _placements(self) -> list[tuple[np.ndarray, int, int]]:
        """(sprite, x, y) for everything that can collide, dino first.

        Rendering and collision both use this, so what you see is exactly what
        collides.
        """
        rest = self.config.ground_y - 1  # bottom row of a sprite standing on the ground
        dino = self._dino_sprite()
        placed = [(dino, self.config.dino_x, rest - round(self.altitude) - dino.shape[0] + 1)]
        for ob in self.obstacles:
            spr = ob.sprite(self.tick)
            placed.append((spr, round(ob.x), rest - ob.altitude - spr.shape[0] + 1))
        return placed

    def _collided(self) -> bool:
        (dino, dx, dy), *others = self._placements()
        return any(overlaps(dino, dx, dy, spr, x, y) for spr, x, y in others)

    def _spawn(self) -> None:
        c = self.config
        if self.score >= c.bird_min_score and self.rng.random() < c.bird_chance:
            altitude = BIRD_LOW if self.rng.random() < 0.5 else BIRD_HIGH
            self.obstacles.append(Obstacle("bird", 0, float(c.width), altitude))
        else:
            variant = int(self.rng.integers(len(sprites.ROCKS)))
            self.obstacles.append(Obstacle("rock", variant, float(c.width)))
        self._spawn_timer = int(self.rng.integers(c.min_gap_frames, c.max_gap_frames + 1))
