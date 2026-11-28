"""PyTorch dataset that plays the dino game inside DataLoader workers.

Nothing is stored on disk. Every worker keeps its own pool of bots playing the
game, and each sample is a clip grabbed from a random bot at whatever point its
game has reached. See games/dino/rollout.py.

    loader = DataLoader(DinoClips(seed=0), batch_size=64, num_workers=2)
    for batch in itertools.islice(loader, num_steps):
        batch["frames"]   # (64, clip_len + 1, 48, 128) bool
        batch["actions"]  # (64, clip_len) int64, values from games.dino.Action
        batch["done"]     # (64, clip_len + 1) bool
        batch["start_tick"]  # (64,) how far into its game each clip starts

The stream never ends, so count training steps rather than epochs. Iterating the
same dataset again replays the same clips, which makes a fixed validation set
easy: use a different seed, e.g. DinoClips(seed=1), and take the first N batches.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from games.dino.rollout import clip_stream


class DinoClips(IterableDataset):
    def __init__(
        self,
        seed: int = 0,
        clip_len: int = 32,
        num_bots: int = 64,
        max_noise: float = 0.02,
        max_episode_frames: int = 3000,
    ) -> None:
        super().__init__()
        self.seed = seed
        self.clip_len = clip_len
        self.num_bots = num_bots
        self.max_noise = max_noise
        self.max_episode_frames = max_episode_frames

    def __iter__(self):
        # Each worker needs its own stream, or every worker would play the exact
        # same games and every batch would hold num_workers copies of each clip.
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        stream_seed = int(np.random.SeedSequence([self.seed, worker_id]).generate_state(1)[0])

        for clip in clip_stream(
            stream_seed, self.clip_len, self.num_bots, self.max_noise, self.max_episode_frames
        ):
            yield {
                "frames": torch.from_numpy(clip.frames),
                "actions": torch.from_numpy(clip.actions),
                "done": torch.from_numpy(clip.done),
                "start_tick": clip.start_tick,
            }
