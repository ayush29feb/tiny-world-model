"""Flow matching: the training loss and the sampler.

Conventions (rectified flow):
    x0 ~ N(0, I)            noise, at t = 0
    x1 = clip as +-1         data, at t = 1
    x_t = (1 - t) x0 + t x1  a straight line between them
    v   = x1 - x0            the velocity the model learns to predict

Sampling starts from noise and follows the predicted velocity from t = 0 to 1,
then thresholds at 0 to get bool pixels back.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tiny_world_model.dit import DiT
from tiny_world_model.tokenizer import patchify, to_bool, to_signed, unpatchify


def flow_matching_loss(model: DiT, frames: torch.Tensor) -> torch.Tensor:
    """Mean squared error between predicted and true velocity.

    frames: (B, T, H, W) bool clips matching the model's DiTConfig. Dataset clips
    have one more frame than actions, so DinoClips(clip_len=31) gives 32 frames.
    """
    c = model.config
    if tuple(frames.shape[1:]) != (c.frames, c.height, c.width):
        raise ValueError(
            f"clips are {tuple(frames.shape[1:])}, model expects {(c.frames, c.height, c.width)}"
        )
    x1 = patchify(to_signed(frames), c.patching)
    x0 = torch.randn_like(x1)
    t = torch.rand(x1.shape[0], device=x1.device)
    xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
    return F.mse_loss(model(xt, t), x1 - x0)


@torch.no_grad()
def sample(
    model: DiT,
    num_clips: int,
    steps: int = 50,
    generator: torch.Generator | None = None,
    threshold: bool = True,
) -> torch.Tensor:
    """Generate clips from noise with Euler steps.

    Returns (num_clips, T, H, W): bool pixels, or the raw +-1-ish values if
    `threshold` is False (useful for seeing how confident the model is).
    """
    c = model.config
    device = next(model.parameters()).device
    x = torch.randn(
        num_clips, c.num_tokens, c.patching.patch_dim, generator=generator, device=device
    )
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((num_clips,), i * dt, device=device)
        x = x + dt * model(x, t)
    clips = unpatchify(x, c.frames, c.height, c.width, c.patching)
    return to_bool(clips) if threshold else clips
