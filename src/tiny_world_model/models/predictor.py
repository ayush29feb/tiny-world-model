"""A small conv next-frame predictor."""

from __future__ import annotations

import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.body(x))


class TinyWorldModel(nn.Module):
    """Predicts frame t+1 from a stack of `context` previous frames.

    Context frames are fed as input channels, so the model sees short-horizon
    dynamics (position and velocity) without any recurrence.
    """

    def __init__(self, context: int = 4, width: int = 64, depth: int = 4) -> None:
        super().__init__()
        self.context = context
        self.stem = nn.Sequential(
            nn.Conv2d(context, width, 3, padding=1),
            nn.GroupNorm(8, width),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(*[ResBlock(width) for _ in range(depth)])
        self.head = nn.Conv2d(width, 1, 3, padding=1)

    def forward(self, context_frames: torch.Tensor) -> torch.Tensor:
        """context_frames: (B, context, H, W) -> logits (B, 1, H, W)."""
        if context_frames.shape[1] != self.context:
            raise ValueError(
                f"expected {self.context} context frames, got {context_frames.shape[1]}"
            )
        h = self.stem(context_frames)
        h = self.blocks(h)
        return self.head(h)

    @torch.no_grad()
    def rollout(self, context_frames: torch.Tensor, steps: int = 16) -> torch.Tensor:
        """Autoregressively imagine `steps` future frames. Returns (B, steps, H, W)."""
        self.eval()
        window = context_frames.clone()
        out = []
        for _ in range(steps):
            nxt = torch.sigmoid(self(window))  # (B, 1, H, W)
            out.append(nxt)
            window = torch.cat([window[:, 1:], nxt], dim=1)
        return torch.cat(out, dim=1)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
