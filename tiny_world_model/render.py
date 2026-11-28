"""Render bool clips as an animated GIF grid, in the Chrome dino's colours. Needs Pillow."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

FOREGROUND = (83, 83, 83)
BACKGROUND = (247, 247, 247)
GAP = (210, 210, 210)


def clips_to_gif(clips, cols: int = 4, scale: int = 2, fps: int = 15, pad: int = 4) -> bytes:
    """clips: (N, T, H, W) bool (tensor or array) -> GIF bytes of an N-clip grid."""
    clips = np.asarray(clips.cpu() if hasattr(clips, "cpu") else clips, dtype=bool)
    n, t, h, w = clips.shape
    rows = -(-n // cols)
    th, tw = h * scale, w * scale
    width = cols * tw + (cols + 1) * pad
    height = rows * th + (rows + 1) * pad

    frames = []
    for f in range(t):
        canvas = np.empty((height, width, 3), dtype=np.uint8)
        canvas[:] = GAP
        for i in range(n):
            r, c = divmod(i, cols)
            y, x = pad + r * (th + pad), pad + c * (tw + pad)
            tile = np.where(clips[i, f][..., None], FOREGROUND, BACKGROUND).astype(np.uint8)
            canvas[y : y + th, x : x + tw] = tile.repeat(scale, 0).repeat(scale, 1)
        frames.append(Image.fromarray(canvas))

    buf = io.BytesIO()
    frames[0].save(
        buf, format="GIF", save_all=True, append_images=frames[1:],
        duration=int(1000 / fps), loop=0,
    )
    return buf.getvalue()
