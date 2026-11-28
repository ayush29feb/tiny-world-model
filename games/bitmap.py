"""Helpers for drawing into binary bitmaps, shared by all the games.

A bitmap is a 2D numpy array of dtype bool, indexed [row, column], with True
meaning the pixel is on. Sprites are bitmaps too, written as ASCII art.
"""

from __future__ import annotations

import textwrap

import numpy as np


def sprite(art: str) -> np.ndarray:
    """Parse ASCII art into a bool array: '#' is on, anything else is off.

    Rows shorter than the widest row are padded with off pixels on the right.
    """
    rows = textwrap.dedent(art).strip("\n").splitlines()
    width = max(len(row) for row in rows)
    return np.array([[ch == "#" for ch in row.ljust(width)] for row in rows], dtype=bool)


def blit(canvas: np.ndarray, spr: np.ndarray, x: int, y: int) -> None:
    """OR a sprite onto the canvas with its top-left corner at (x, y).

    Parts of the sprite that fall outside the canvas are clipped, so sprites
    can slide in from and out past the edges.
    """
    h, w = spr.shape
    rows, cols = canvas.shape
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, cols), min(y + h, rows)
    if x0 >= x1 or y0 >= y1:
        return
    canvas[y0:y1, x0:x1] |= spr[y0 - y : y1 - y, x0 - x : x1 - x]


def overlaps(a: np.ndarray, ax: int, ay: int, b: np.ndarray, bx: int, by: int) -> bool:
    """True if two placed sprites share at least one on pixel (pixel-perfect)."""
    x0, y0 = max(ax, bx), max(ay, by)
    x1 = min(ax + a.shape[1], bx + b.shape[1])
    y1 = min(ay + a.shape[0], by + b.shape[0])
    if x0 >= x1 or y0 >= y1:
        return False
    a_part = a[y0 - ay : y1 - ay, x0 - ax : x1 - ax]
    b_part = b[y0 - by : y1 - by, x0 - bx : x1 - bx]
    return bool(np.any(a_part & b_part))


# A 3x5 pixel font. Only the characters the games need so far.
_FONT_ART = {
    "0": ["###", "#.#", "#.#", "#.#", "###"],
    "1": [".#.", "##.", ".#.", ".#.", "###"],
    "2": ["###", "..#", "###", "#..", "###"],
    "3": ["###", "..#", "###", "..#", "###"],
    "4": ["#.#", "#.#", "###", "..#", "..#"],
    "5": ["###", "#..", "###", "..#", "###"],
    "6": ["###", "#..", "###", "#.#", "###"],
    "7": ["###", "..#", "..#", "..#", "..#"],
    "8": ["###", "#.#", "###", "#.#", "###"],
    "9": ["###", "#.#", "###", "..#", "###"],
    "A": [".#.", "#.#", "###", "#.#", "#.#"],
    "E": ["###", "#..", "##.", "#..", "###"],
    "G": ["###", "#..", "#.#", "#.#", "###"],
    "M": ["#.#", "###", "###", "#.#", "#.#"],
    "O": [".#.", "#.#", "#.#", "#.#", ".#."],
    "R": ["##.", "#.#", "##.", "#.#", "#.#"],
    "V": ["#.#", "#.#", "#.#", "#.#", ".#."],
    " ": ["...", "...", "...", "...", "..."],
}
FONT = {ch: sprite("\n".join(rows)) for ch, rows in _FONT_ART.items()}
GLYPH_W, GLYPH_H = 3, 5


def text_width(text: str, scale: int = 1) -> int:
    """Width in pixels of `text`, with one pixel of space between glyphs."""
    if not text:
        return 0
    return (len(text) * (GLYPH_W + 1) - 1) * scale


def draw_text(canvas: np.ndarray, text: str, x: int, y: int, scale: int = 1) -> None:
    """Draw `text` with its top-left corner at (x, y). `scale` enlarges each pixel."""
    for i, ch in enumerate(text.upper()):
        if ch not in FONT:
            raise KeyError(f"no glyph for {ch!r}; add it to games/bitmap.py")
        glyph = FONT[ch]
        if scale > 1:
            glyph = glyph.repeat(scale, axis=0).repeat(scale, axis=1)
        blit(canvas, glyph, x + i * (GLYPH_W + 1) * scale, y)
