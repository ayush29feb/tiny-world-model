"""Chrome-style dino runner rendered as binary bitmaps.

The engine and rollout code need only numpy. `games.dino.dataset` (PyTorch) and
`games.dino.play` (pygame) are imported separately so neither is required here.
"""

from games.dino.game import Action, DinoConfig, DinoGame, Obstacle
from games.dino.rollout import Bot, Clip, clip_stream, scripted_action

__all__ = [
    "Action",
    "Bot",
    "Clip",
    "DinoConfig",
    "DinoGame",
    "Obstacle",
    "clip_stream",
    "scripted_action",
]
