"""A small diffusion transformer (DiT) over clip tokens.

Input: noisy tokens (B, N, patch_dim) and the flow time t in [0, 1].
Output: a predicted velocity for every token, same shape as the input.

Follows DiT (Peebles & Xie, 2023): tokens are projected to the model width,
given positions, and run through transformer blocks whose LayerNorms are
modulated by the time step ("adaLN-Zero"). Attention is full: every token
attends to every other token in the clip, across space and time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from tiny_world_model.tokenizer import Patching


@dataclass(frozen=True)
class DiTConfig:
    # Clip shape the model is built for. Positions are learned, so it only works
    # on clips of exactly this size.
    frames: int = 32
    height: int = 48
    width: int = 128
    patching: Patching = field(default_factory=Patching)

    # Model size. `dim` is the width each token is carried at inside the model.
    dim: int = 128
    depth: int = 8
    heads: int = 4
    mlp_ratio: int = 4

    @property
    def grid(self) -> tuple[int, int, int]:
        return self.patching.grid(self.frames, self.height, self.width)

    @property
    def num_tokens(self) -> int:
        gt, gh, gw = self.grid
        return gt * gh * gw


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale[:, None]) + shift[:, None]


class TimeEmbedding(nn.Module):
    """Flow time t in [0, 1] -> a vector of size dim."""

    def __init__(self, dim: int, frequencies: int = 256) -> None:
        super().__init__()
        self.frequencies = frequencies
        self.mlp = nn.Sequential(nn.Linear(frequencies, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.frequencies // 2
        freqs = torch.exp(-math.log(10_000) * torch.arange(half, device=t.device) / half)
        angles = (t[:, None] * 1000) * freqs[None]
        return self.mlp(torch.cat([angles.cos(), angles.sin()], dim=-1))


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: int) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim),
        )
        # Six vectors from the time embedding: shift, scale and gate for each sub-layer.
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.modulation(c).chunk(6, dim=-1)

        b, n, d = x.shape
        h = modulate(self.norm1(x), shift1, scale1)
        q, k, v = self.qkv(h).reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        h = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, d)
        x = x + gate1[:, None] * self.proj(h)

        h = modulate(self.norm2(x), shift2, scale2)
        return x + gate2[:, None] * self.mlp(h)


class DiT(nn.Module):
    def __init__(self, config: DiTConfig = DiTConfig()) -> None:
        super().__init__()
        self.config = config
        c = config
        gt, gh, gw = c.grid

        self.embed = nn.Linear(c.patching.patch_dim, c.dim)  # 128 values -> dim
        # Positions are learned separately for time, row and column, then added.
        self.pos_t = nn.Parameter(torch.randn(gt, c.dim) * 0.02)
        self.pos_h = nn.Parameter(torch.randn(gh, c.dim) * 0.02)
        self.pos_w = nn.Parameter(torch.randn(gw, c.dim) * 0.02)
        self.time = TimeEmbedding(c.dim)

        self.blocks = nn.ModuleList(Block(c.dim, c.heads, c.mlp_ratio) for _ in range(c.depth))

        self.final_norm = nn.LayerNorm(c.dim, elementwise_affine=False, eps=1e-6)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(c.dim, 2 * c.dim))
        self.out = nn.Linear(c.dim, c.patching.patch_dim)  # dim -> 128 values

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        # adaLN-Zero: every block starts as the identity and the output starts at
        # zero, which makes early training stable.
        for block in self.blocks:
            nn.init.zeros_(block.modulation[-1].weight)
            nn.init.zeros_(block.modulation[-1].bias)
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def positions(self) -> torch.Tensor:
        """(num_tokens, dim), in the same time-row-column order as patchify."""
        pos = self.pos_t[:, None, None] + self.pos_h[None, :, None] + self.pos_w[None, None, :]
        return pos.reshape(-1, self.config.dim)

    def forward(self, tokens: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """tokens: (B, N, patch_dim) noisy clip; t: (B,) flow time. Returns velocity."""
        if tokens.shape[1] != self.config.num_tokens:
            raise ValueError(
                f"expected {self.config.num_tokens} tokens, got {tokens.shape[1]}; "
                "the model is built for one clip size (see DiTConfig)"
            )
        x = self.embed(tokens) + self.positions()
        c = self.time(t)
        for block in self.blocks:
            x = block(x, c)
        shift, scale = self.final_modulation(c).chunk(2, dim=-1)
        return self.out(modulate(self.final_norm(x), shift, scale))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
