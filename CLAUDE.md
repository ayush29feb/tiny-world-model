# tiny-world-model

Tiny pixel games whose frames are binary bitmaps, used to train a world model: given
past frames and the player's key presses, predict the next frame. Training data is
generated on the fly by bots playing the games inside PyTorch DataLoader workers.
Nothing is recorded to disk. Training will run on Colab GPUs.

**Keep this file current.** When the game, the data format or the data mix changes,
update the relevant section here: shapes, dtypes, measured numbers and gotchas.

## Layout

```
games/
  bitmap.py        sprite parsing, blitting, pixel-perfect collision, 3x5 font (shared)
  dino/
    sprites.py     ASCII-art sprites ('#' on, '.' off)
    game.py        headless engine: DinoGame, Action, DinoConfig
    rollout.py     scripted bots and the clip generator (numpy only)
    dataset.py     DinoClips, a PyTorch IterableDataset (the only torch import)
    play.py        pygame window for a human (the only pygame import)
tests/             engine and data tests; no pygame needed
```

## Rules for game code

- **A frame is a 2D `bool` numpy array, `[row, column]`, True = on.** Everything visible,
  including the score and any text, is drawn into it. Never add state that matters to the
  game but is not in the frame without saying so.
- **Engines are headless and pure numpy.** `step(action)` returns the next frame. pygame
  and torch stay out of engine code so the engine runs anywhere.
- **Deterministic from a seed.** All randomness goes through the game's own
  `np.random.Generator`. Never use `random` or the global numpy RNG.
- **Render and collide from the same placements**, so collisions match what is drawn.
- **After game over, `step` returns the same frame forever** and the score stops.
- Shared drawing helpers go in `games/bitmap.py`; game-specific code goes in
  `games/<name>/`. The font only has the glyphs used so far (digits and the letters
  in GAME OVER); add glyphs to `_FONT_ART` as needed.

## The dino game

- Frame is **48 × 128**. Ground line is row 42; sprites rest on row 41; scrolling
  ground dots are on rows 44–47. Score is 5 digits in the top-right (rows 1–5).
  GAME OVER is drawn 2× size, centered, on the crash frame.
- Dino: 11 × 12 standing, 15 × 7 ducking, fixed at x = 10.
- Actions: `NOOP = 0`, `JUMP = 1` (only works on the ground), `DUCK = 2` (on the
  ground it ducks; in the air it triples gravity to fall faster).
- Obstacles: three rock shapes (jump them) and a bird at two heights. `BIRD_LOW`
  (altitude 2) must be jumped; `BIRD_HIGH` (altitude 8) clears a ducking dino by one
  pixel but hits a standing one. Birds appear from score 100.
- Speed goes from 2 to 4 pixels per frame, reaching the top after 2000 frames.
  Peak jump height is about 18 pixels. All tuning is in `DinoConfig`.

## Training data

`DinoClips` runs a pool of 64 bots per DataLoader worker, each playing its own game.
Each sample picks a random bot and records its next `clip_len` frames, so clips start
at random points in a game and neighbouring samples come from different games. At
startup each bot fast-forwards a random number of frames (a few seconds per worker).

### A batch

`DataLoader(DinoClips(seed=0, clip_len=32), batch_size=64)` yields a dict:

| key | shape | dtype | size | meaning |
|---|---|---|---|---|
| `frames` | `(64, 33, 48, 128)` | `bool` | 13.0 MB | batch × time × height × width |
| `actions` | `(64, 32)` | `int64` | 16 KB | key pressed between frame t and t+1 |
| `done` | `(64, 33)` | `bool` | 2 KB | True from the crash onward |
| `start_tick` | `(64,)` | `int64` | 0.5 KB | frames into its game the clip starts |

`frames[:, t]` + `actions[:, t]` → `frames[:, t+1]`, so the usual split is
inputs `frames[:, :-1]` with `actions`, and targets `frames[:, 1:]`.

### Things that matter when training on it

- **Only ~4.8% of pixels are on.** Predicting an all-empty frame scores ~95% per-pixel
  accuracy. Don't trust plain accuracy; weight on-pixels or score changed pixels.
- **Convert to float on the GPU.** Send the `bool` frames to the device first, then call
  `.float()`. As `float32` a batch is 52 MB instead of 13 MB.
- **Many actions have no effect.** The bot holds JUMP through a whole jump, but only the
  press on the ground does anything. The model has to learn that.
- **`start_tick` is for checking the data, not a model input.**
- About 1 clip in 80 contains a crash at the default noise. A batch of 16 often has none.
- The stream never ends: count steps, not epochs. Re-iterating the same dataset replays
  the same clips; use another seed (e.g. `DinoClips(seed=1)`) for a fixed validation set.

### The data mix: `max_noise`

Each bot game gets a noise level in `[0, max_noise]`: the per-frame chance of pressing a
random key instead of the scripted one. Random presses cause crashes, and crash-prone
bots keep restarting, so clips pile up in the slow opening. Measured over 3000 clips:

| `max_noise` | clips from late game (>1500 frames) | clips with a crash |
|---|---|---|
| 0.3 | 15% | 6.2% |
| 0.05 | 22% | 2.5% |
| **0.02 (default)** | **35%** | **1.2%** |

After changing bots, noise or game speed, re-measure the `start_tick` spread and crash
rate. It is easy to end up with data that is mostly the first 500 frames.

### Throughput

Batches of 64 clips of 32 frames, measured on an Apple M-series laptop:

| DataLoader workers | clips/s | frames/s |
|---|---|---|
| 0 | 1,400 | 48,000 |
| 2 (free Colab has 2 cores) | 2,700 | 90,000 |
| 4 | 4,500 | 150,000 |

Not yet measured on Colab, whose CPUs are likely slower.

### Gotchas

- **Each DataLoader worker must get its own seed** (see `DinoClips.__iter__`), or every
  worker plays identical games and batches fill with duplicates.
  `test_each_worker_plays_its_own_games` guards this.
- **On macOS, scripts using `num_workers > 0` need `if __name__ == "__main__":`.** Workers
  start fresh Python processes that re-import the script, and code piped in on stdin
  can't be re-imported. Linux and Colab start workers differently and don't need it.

## Commands

```bash
source .venv/bin/activate
uv pip install -e ".[play,train,dev]"
pytest -q
ruff check .
python -m games.dino.play
```

## Colab

Keep `torch` out of the hard dependencies (it lives in the `train` extra), since Colab
ships its own CUDA build and installing a second one breaks the runtime.
