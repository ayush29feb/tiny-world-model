"""Turn clips of binary frames into patch tokens, and back again, losslessly.

A clip of shape (B, T, H, W) is cut into non-overlapping tubelets of
`patch_t` frames x `patch_h` rows x `patch_w` columns. Each tubelet is flattened
into one token of patch_t * patch_h * patch_w values. With the default 2x8x8
tubelets, a 32-frame 48x128 clip becomes 16 x 6 x 16 = 1,536 tokens of 128 values.

Tokens are ordered time-major, then row, then column. Inside a token, values are
ordered frame, then row, then column.

There is no learned compression here: unpatchify(patchify(x)) == x exactly. The
model's own Linear layer maps each 128-value token to its working width.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Patching:
    patch_t: int = 2
    patch_h: int = 8
    patch_w: int = 8

    @property
    def patch_dim(self) -> int:
        """Values per token."""
        return self.patch_t * self.patch_h * self.patch_w

    def grid(self, frames: int, height: int, width: int) -> tuple[int, int, int]:
        """Number of tokens along (time, rows, columns) for a clip of this size."""
        for name, size, patch in (
            ("frames", frames, self.patch_t),
            ("height", height, self.patch_h),
            ("width", width, self.patch_w),
        ):
            if size % patch:
                raise ValueError(f"{name}={size} is not divisible by its patch size {patch}")
        return frames // self.patch_t, height // self.patch_h, width // self.patch_w


def patchify(clips: torch.Tensor, patching: Patching = Patching()) -> torch.Tensor:
    """(B, T, H, W) -> (B, num_tokens, patch_dim)."""
    b, t, h, w = clips.shape
    gt, gh, gw = patching.grid(t, h, w)
    pt, ph, pw = patching.patch_t, patching.patch_h, patching.patch_w
    x = clips.reshape(b, gt, pt, gh, ph, gw, pw)
    x = x.permute(0, 1, 3, 5, 2, 4, 6)  # (B, gt, gh, gw, pt, ph, pw)
    return x.reshape(b, gt * gh * gw, pt * ph * pw)


def unpatchify(
    tokens: torch.Tensor, frames: int, height: int, width: int, patching: Patching = Patching()
) -> torch.Tensor:
    """(B, num_tokens, patch_dim) -> (B, T, H, W). The exact inverse of patchify."""
    b = tokens.shape[0]
    gt, gh, gw = patching.grid(frames, height, width)
    pt, ph, pw = patching.patch_t, patching.patch_h, patching.patch_w
    x = tokens.reshape(b, gt, gh, gw, pt, ph, pw)
    x = x.permute(0, 1, 4, 2, 5, 3, 6)  # (B, gt, pt, gh, ph, gw, pw)
    return x.reshape(b, frames, height, width)


def to_signed(frames: torch.Tensor) -> torch.Tensor:
    """bool pixels -> float -1 (off) / +1 (on), the form flow matching works on."""
    return frames.float() * 2 - 1


def to_bool(x: torch.Tensor) -> torch.Tensor:
    """Continuous model output -> bool pixels, thresholding at 0."""
    return x > 0
