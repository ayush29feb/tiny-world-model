"""A procedurally generated bouncing-ball dataset.

Deliberately synthetic: it needs no download, so a Colab runtime is productive
the moment the repo is cloned, and the GPU path is easy to verify.
"""

from __future__ import annotations

import numpy as np


def make_rollout(
    steps: int = 32,
    size: int = 32,
    radius: float = 3.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Render one rollout of a ball bouncing in a box.

    Returns float32 array of shape (steps, 1, size, size) in [0, 1].
    """
    rng = rng or np.random.default_rng()

    pos = rng.uniform(radius, size - radius, size=2)
    vel = rng.uniform(-1.5, 1.5, size=2)
    # Avoid near-stationary rollouts, which carry no dynamics to learn.
    speed = float(np.linalg.norm(vel))
    if speed < 0.5:
        vel = vel / max(speed, 1e-6) * 0.5

    yy, xx = np.mgrid[0:size, 0:size]
    frames = np.zeros((steps, 1, size, size), dtype=np.float32)

    for t in range(steps):
        dist = np.sqrt((xx - pos[0]) ** 2 + (yy - pos[1]) ** 2)
        # Soft edge keeps the target differentiable-friendly rather than binary.
        frames[t, 0] = np.clip(radius - dist + 0.5, 0.0, 1.0)

        pos = pos + vel
        for axis in (0, 1):
            if pos[axis] < radius:
                pos[axis] = 2 * radius - pos[axis]
                vel[axis] *= -1
            elif pos[axis] > size - radius:
                pos[axis] = 2 * (size - radius) - pos[axis]
                vel[axis] *= -1

    return frames


class BouncingBallDataset:
    """Map-style dataset of (context_frames, next_frame) pairs.

    Implements __len__/__getitem__ so it plugs straight into torch DataLoader
    without importing torch here.
    """

    def __init__(
        self,
        n_rollouts: int = 512,
        steps: int = 32,
        size: int = 32,
        context: int = 4,
        seed: int = 0,
    ) -> None:
        if steps <= context:
            raise ValueError(f"steps ({steps}) must exceed context ({context})")

        self.context = context
        self.size = size
        rng = np.random.default_rng(seed)
        self.rollouts = np.stack(
            [make_rollout(steps=steps, size=size, rng=rng) for _ in range(n_rollouts)]
        )
        self.per_rollout = steps - context

    def __len__(self) -> int:
        return len(self.rollouts) * self.per_rollout

    def __getitem__(self, idx: int):
        r, t = divmod(idx, self.per_rollout)
        rollout = self.rollouts[r]
        # (context, 1, H, W) -> (context, H, W): channel dim carries time.
        ctx = rollout[t : t + self.context, 0]
        target = rollout[t + self.context, 0:1]
        return ctx.copy(), target.copy()
