"""Flow matching: the noise-level schedule, the training loss, a fixed validation
loss, and the sampler.

Conventions (rectified flow):
    x0 ~ N(0, I)            noise, at t = 0
    x1 = clip as +-1         data, at t = 1
    x_t = (1 - t) x0 + t x1  a straight line between them
    v   = x1 - x0            the velocity the model learns to predict

Training draws t from a logit-normal distribution (most steps at middle noise
levels), then shifts it toward the noise end. Sampling follows the predicted
velocity from t = 0 to 1 on a time grid with the same shift, then thresholds at 0
to get bool pixels back.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from tiny_world_model.dit import DiT
from tiny_world_model.tokenizer import patchify, to_bool, to_signed, unpatchify


def shift_t(t: torch.Tensor, shift: float) -> torch.Tensor:
    """Move flow times toward the noise end (t = 0) by `shift` (1 = no change).

    This is SD3's resolution shift, written for t = 0 being noise. With noise level
    s = 1 - t it maps s -> shift * s / (1 + (shift - 1) * s). With shift 3, t = 0.5
    becomes 0.25. The ends stay put: 0 -> 0 and 1 -> 1.
    """
    noise = 1 - t
    return 1 - shift * noise / (1 + (shift - 1) * noise)


@dataclass(frozen=True)
class Schedule:
    """Which noise levels training spends its steps on.

    `dist` is "logit_normal" (t = sigmoid(N(mean, std))) or "uniform". The result is
    then moved toward noise by `shift`. The defaults, logit-normal(0, 1) shifted by
    3, put the median at t = 0.25 and about 60% of steps below t = 0.3. Clips are
    large and very redundant, so their layout survives a lot of noise and is decided
    at high noise levels; the shift gives those levels more training.
    """

    dist: str = "logit_normal"
    mean: float = 0.0
    std: float = 1.0
    shift: float = 3.0

    def sample(self, n: int, device, generator: torch.Generator | None = None) -> torch.Tensor:
        if self.dist == "logit_normal":
            z = torch.randn(n, device=device, generator=generator)
            t = torch.sigmoid(self.mean + self.std * z)
        elif self.dist == "uniform":
            t = torch.rand(n, device=device, generator=generator)
        else:
            raise ValueError(f"unknown dist {self.dist!r}; use 'logit_normal' or 'uniform'")
        return shift_t(t, self.shift)


def _check_clip_shape(model: DiT, frames: torch.Tensor) -> None:
    c = model.config
    if tuple(frames.shape[1:]) != (c.frames, c.height, c.width):
        raise ValueError(
            f"clips are {tuple(frames.shape[1:])}, model expects {(c.frames, c.height, c.width)}"
        )


def _velocity_loss(model: DiT, x1: torch.Tensor, x0: torch.Tensor, t: torch.Tensor):
    xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
    return F.mse_loss(model(xt, t), x1 - x0)


def flow_matching_loss(
    model: DiT, frames: torch.Tensor, schedule: Schedule = Schedule()
) -> torch.Tensor:
    """Mean squared error between predicted and true velocity, at random noise levels.

    frames: (B, T, H, W) bool clips matching the model's DiTConfig. Dataset clips
    have one more frame than actions, so DinoClips(clip_len=31) gives 32 frames.

    The value depends on `schedule`: noisier levels have a higher loss, so training
    losses under different schedules can't be compared. Use `validation_loss` for that.
    """
    _check_clip_shape(model, frames)
    x1 = patchify(to_signed(frames), model.config.patching)
    x0 = torch.randn_like(x1)
    t = schedule.sample(x1.shape[0], x1.device)
    return _velocity_loss(model, x1, x0, t)


VALIDATION_TS = tuple((i + 0.5) / 8 for i in range(8))  # 0.0625 ... 0.9375, evenly spaced


@torch.no_grad()
def validation_loss(
    model: DiT, frames: torch.Tensor, noise: torch.Tensor, ts: tuple[float, ...] = VALIDATION_TS
) -> float:
    """The velocity loss averaged over a fixed, evenly spaced set of noise levels.

    With the same `frames` and `noise` every time, this is deterministic and does not
    depend on the training schedule, so it compares checkpoints and runs fairly.
    `noise` must have the shape of patchify(frames): (B, num_tokens, patch_dim).
    """
    _check_clip_shape(model, frames)
    x1 = patchify(to_signed(frames), model.config.patching)
    total = 0.0
    for t in ts:
        tt = torch.full((x1.shape[0],), t, device=x1.device)
        total += _velocity_loss(model, x1, noise, tt).item()
    return total / len(ts)


def sampling_times(steps: int, shift: float, device=None) -> torch.Tensor:
    """steps + 1 times from 0 (noise) to 1 (data), spaced evenly then shifted."""
    return shift_t(torch.linspace(0, 1, steps + 1, device=device), shift)


@torch.no_grad()
def sample(
    model: DiT,
    num_clips: int,
    steps: int = 50,
    generator: torch.Generator | None = None,
    threshold: bool = True,
    shift: float = 3.0,
) -> torch.Tensor:
    """Generate clips from noise with Euler steps.

    `shift` should match training: it spends more of the steps at high noise, where
    the layout is decided. Returns (num_clips, T, H, W): bool pixels, or the raw
    +-1-ish values if `threshold` is False (useful for seeing how confident it is).
    """
    c = model.config
    device = next(model.parameters()).device
    x = torch.randn(
        num_clips, c.num_tokens, c.patching.patch_dim, generator=generator, device=device
    )
    times = sampling_times(steps, shift, device)
    for t, t_next in zip(times[:-1], times[1:], strict=True):
        x = x + (t_next - t) * model(x, t.expand(num_clips))
    clips = unpatchify(x, c.frames, c.height, c.width, c.patching)
    return to_bool(clips) if threshold else clips
