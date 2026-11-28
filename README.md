# tiny-world-model

Tiny pixel games whose every frame is a binary bitmap, so games can be recorded as
frame sequences and used to train a world model.

## Dino runner

A black-and-white version of Chrome's offline dino game. Jump rocks, duck or jump
birds, and the score counts up in the top right. Hitting anything ends the game and
draws GAME OVER.

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install -e ".[play,train,notebook,dev]"
python -m games.dino.play
```

**Controls:** Space or Up to jump, hold Down to duck (or to fall faster mid-jump),
Space to restart after game over, Esc or Q to quit.

### The state is a bitmap

The engine is headless: `step(action)` returns the next frame as a `(48, 128)` numpy
array of `bool`. Everything is drawn into it — dino, obstacles, ground, score, and the
GAME OVER text — so a frame is the complete visible state of the game.

```python
from games.dino import Action, DinoGame

game = DinoGame(seed=0)
frame = game.reset()                # (48, 128) bool
while not game.done:
    frame = game.step(Action.JUMP)  # NOOP, JUMP or DUCK
print(game.score)
```

- **Deterministic:** the same seed and the same actions always give identical frames.
- **Pixel-perfect collisions:** the game ends when an on pixel of the dino overlaps an on
  pixel of an obstacle, using the same placements it draws, so what you see is exactly
  what collides.
- **Frozen after game over:** further steps return the same frame, so a recording of
  fixed length is easy.
- **Compact:** `np.packbits(frame)` stores a frame in 768 bytes.

### Obstacles

| Obstacle | How to clear it |
|---|---|
| Rock (small, large, double) | Jump |
| Low bird | Jump |
| High bird | Duck, or jump |

Birds appear once the score reaches 100. The game speeds up from 2 to 4 pixels per
frame over the first ~67 seconds. All tuning lives in `DinoConfig`
(`games/dino/game.py`).

## Training data, generated on the fly

Nothing is recorded to disk. `DinoClips` is a PyTorch `IterableDataset` that plays
the game inside each DataLoader worker and hands out clips as training asks for them.

```python
import itertools
from torch.utils.data import DataLoader
from games.dino.dataset import DinoClips

loader = DataLoader(DinoClips(seed=0, clip_len=32), batch_size=64, num_workers=2)
for batch in itertools.islice(loader, num_steps):
    batch["frames"]      # (64, 33, 48, 128) bool: frame[t] --action[t]--> frame[t+1]
    batch["actions"]     # (64, 32) int64: 0 NOOP, 1 JUMP, 2 DUCK
    batch["done"]        # (64, 33) bool: True from the crash onward
    batch["start_tick"]  # (64,) how far into its game each clip starts
```

**How clips are made.** Each worker runs a pool of 64 bots, each playing its own game.
A clip picks a random bot and records its next 32 frames. The bots are spread across
every point in a game, so a batch mixes slow openings, fast late games, and crashes,
and neighbouring samples come from different games. A bot is a scripted player that
never makes a mistake on its own; each game it gets a small chance per frame of
pressing a random key instead, which is what makes it sometimes crash.

**Throughput** (batches of 64 clips of 32 frames, on an Apple M-series laptop):

| Workers | Clips/s | Frames/s |
|---|---|---|
| 0 | 1,400 | 48,000 |
| 2 (free Colab has 2 cores) | 2,700 | 90,000 |
| 4 | 4,500 | 150,000 |

Each worker spends a few seconds at startup fast-forwarding its bots to random points
in their games.

**Things to know:**

- The stream never ends: count training steps, not epochs.
- Iterating the same dataset again replays the same clips. For a fixed validation set,
  use a different seed, such as `DinoClips(seed=1)`, and take its first N batches.
- `max_noise` trades crashes against coverage of the whole game (see the docstring in
  `games/dino/rollout.py` for measurements). The default, 0.02, gives about 35% of clips
  from late game and a crash in about 1 clip in 80.
- On macOS, a script that uses `num_workers > 0` needs an `if __name__ == "__main__":`
  guard, because workers start fresh Python processes that re-import it. Colab and
  Linux don't need this.

## Generating clips with flow matching

`tiny_world_model/` is a small diffusion transformer (DiT, 2.5M parameters) trained
with flow matching to generate 32-frame clips from noise. Clips are cut losslessly into
1,536 tokens of 2 frames × 8 × 8 pixels; there is no VAE.

Train it on a Colab GPU:

```
https://colab.research.google.com/github/ayush29feb/tiny-world-model/blob/main/notebooks/train_dino_dit.ipynb
```

Set **Runtime → Change runtime type → T4 GPU** and run the cells. The training cell
shows a live loss plot and, every 500 steps, clips generated from noise. Checkpoints
and those GIFs are saved to Google Drive, and re-running the cell resumes.

Or from Python:

```python
from tiny_world_model.train import TrainConfig, train
ema = train(TrainConfig(steps=20_000, checkpoint_dir="runs/dit128"))
```

## Layout

```
games/
  bitmap.py        sprite parsing, blitting, pixel collision, 3x5 font (shared)
  dino/
    sprites.py     ASCII-art sprites; edit these to change how things look
    game.py        headless engine: DinoGame, Action, DinoConfig
    rollout.py     scripted bots and the clip generator (numpy only)
    dataset.py     DinoClips, a PyTorch IterableDataset over the generator
    play.py        pygame window for playing it yourself
tiny_world_model/  tokenizer, DiT, flow matching, training loop, live monitor
notebooks/         Colab training notebook
tests/             engine, data, model and training tests; no pygame needed
```

```bash
pytest -q
ruff check .
```
