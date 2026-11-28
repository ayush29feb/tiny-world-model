# tiny-world-model

Tiny pixel games whose frames are binary bitmaps, used to train generative video
models of them. The current model is a flow-matching diffusion transformer that
generates 32-frame dino clips from noise. Training data is generated on the fly by
bots playing the games inside PyTorch DataLoader workers; nothing is recorded to
disk. Training runs on Colab GPUs (`notebooks/train_dino_dit.ipynb`).

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
tiny_world_model/  the model (all torch)
  tokenizer.py     patchify / unpatchify clips into 2x8x8 tubelet tokens (lossless)
  dit.py           DiT: the diffusion transformer, DiTConfig
  flow.py          flow-matching loss and Euler sampler
  train.py         training loop: TrainConfig, train(), checkpoints, EMA
  render.py        clips -> animated GIF grid
  notebook.py      LiveMonitor: live loss plot + generated clips in Colab/Jupyter
notebooks/
  train_dino_dit.ipynb   Colab: clone, install, train with live display
tests/             engine, data, model and training tests; no pygame needed
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

## The model

### Tokens

A 32-frame clip (B, 32, 48, 128) is cut into **2 × 8 × 8 tubelets**: 2 frames × 8 rows
× 8 columns = **128 values per token**, and a 16 × 6 × 16 grid = **1,536 tokens per
clip**, ordered time, row, column. There is no VAE: the clips are already about the
size of one Stable Diffusion latent, and a lossy encoder would blur exactly the pixels
the game rules depend on (the one-pixel gap under the high bird, the 3×5 score
digits). `unpatchify(patchify(x)) == x` exactly.

Dataset clips have one more frame than actions, so the model's 32 frames come from
`DinoClips(clip_len=31)`. `TrainConfig` does this automatically.

### DiT

`DiTConfig()` defaults: **width 128, 8 blocks, 4 heads, 2.49M parameters.** Each
128-value token goes through `Linear(128 → dim)`, gets learned positions (separate
time, row and column tables, added together) and full attention over all 1,536
tokens. Blocks are conditioned on the flow time with adaLN-Zero, so an untrained
model outputs exactly zero. Positions are learned, so a model only works on the clip
size it was built for.

Width is a setting, kept separate from the patch size. Width 128 was chosen to start
because at 1,536 tokens most compute is attention between tokens: width 256 costs
about 2.7× more per layer, not 4×. To cut compute, reduce tokens (factorized
space/time attention, fewer frames) before reducing width. A 128 vs 256 comparison
has not been run yet.

### Flow matching (`flow.py`)

- Pixels are mapped to **−1 (off) / +1 (on)**: `to_signed`. Generated values are
  thresholded at 0: `to_bool`.
- **t = 0 is noise, t = 1 is data.** `x_t = (1 − t)·noise + t·data`, the model predicts
  velocity `v = data − noise`, loss is MSE. t is sampled uniformly.
- Sampling: 50 Euler steps from t = 0 to 1, then threshold.

### Training (`train.py`)

AdamW, lr 3e-4 with 500 warmup steps then constant, grad clip 1.0, batch 32.
**Samples always come from an EMA of the weights (decay 0.999)**, drawn from the same
fixed noise each time so progress is comparable. fp16 autocast + GradScaler on CUDA
(the T4 has no fast bf16); full precision elsewhere. With `checkpoint_dir` set, it saves
every `checkpoint_every` steps (write, then rename) and resumes automatically; the data
seed is offset by the resume step so resumed runs don't replay the same clips.

### Measured speeds

| What | Where | Speed |
|---|---|---|
| Data generation | 1 CPU core | ~1,400 clips/s |
| Training, width 128 | Apple M-series GPU (MPS) | **~13 clips/s**, flat for batch 8–32, fp32/bf16/fp16 alike |
| Training, width 128 | Colab T4 | not measured yet |

On the Mac the model is ~100× slower than the data and fully compute-bound, so it's
only good for checking that things run. Train on Colab.

### Gotchas

- **Don't judge samples by "% of pixels matching".** ~95% of pixels are background, so
  an empty frame already matches ~95%. Look at the samples, or measure how many
  *on* pixels are reproduced.
- **Early in training the model learns the layout first:** the ground line, the dots
  under it and digit-like marks in the score corner. The dino and obstacles come later.
  After 510 steps on 8 clips (MPS) there was layout but no dino. That is expected, not
  a bug.
- The first `clips_per_sec` reading in each run is low: it includes the DataLoader
  workers fast-forwarding their bots at startup.
- Outside a notebook, `IPython.display.display(..., display_id=True)` returns `None`.
  `LiveMonitor` creates `DisplayHandle`s directly so it also runs in tests.

## Commands

```bash
source .venv/bin/activate
uv pip install -e ".[play,train,notebook,dev]"
pytest -q
ruff check .
python -m games.dino.play
```

## Colab

Open `notebooks/train_dino_dit.ipynb` in Colab (it clones the public repo, so pushed
code is what runs there). It installs with plain `pip install -e .`, no extras: keep
`torch` out of the hard dependencies (it lives in the `train` extra), since Colab ships
its own CUDA build and installing a second one breaks the runtime. Colab already has
Pillow, matplotlib and IPython. Checkpoints and sample GIFs go to Google Drive.
