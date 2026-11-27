# tiny-world-model

A small next-frame predictor, set up so the same code runs on a laptop (Claude Code) and on
a Google Colab GPU, with GitHub as the only thing in between.

## How the loop works

```
  laptop / Claude Code                GitHub                    Colab
  ────────────────────                ──────                    ─────
  edit src/, run pytest   ──push──>   main   ──pull (cell 2)──>  pip install -e .
                                                                 train on GPU
                          <─────── checkpoints via Drive ───────
```

Code lives in `src/`, not in notebooks. Notebooks only orchestrate — that keeps diffs
reviewable, makes the logic testable in CI, and avoids merge conflicts in JSON.

## Local setup

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install -e ".[local,dev]"
pytest -q
```

Tests are CPU-only and take seconds; run them before pushing to Colab.

A short smoke run on your own machine (MPS on Apple silicon):

```bash
python scripts/train.py --epochs 1 --n-rollouts 32 --num-workers 0
```

## Colab

Open the notebook on a GPU runtime and run it top to bottom — the repo is public, so the
clone needs no token:

```
https://colab.research.google.com/github/ayush29feb/tiny-world-model/blob/main/notebooks/colab_bootstrap.ipynb
```

Set **Runtime → Change runtime type → T4 GPU** first, or cell 1 will report no GPU.

### Syncing after you edit locally

Commit and push, then re-run **cell 2** (git pull) in Colab. Because the package is
installed with `pip install -e .` and the notebook uses `%autoreload 2`, the new code is
picked up without restarting the runtime or reinstalling.

Restart the runtime only when `pyproject.toml` dependencies change.

### Checkpoints survive disconnects

`artifacts_dir()` resolves in this order:

1. `$TWM_ARTIFACTS` if set
2. `/content/drive/MyDrive/tiny-world-model/artifacts` when Drive is mounted on Colab
3. `./artifacts` locally

Training writes `checkpoint.pt` after every epoch and `train()` resumes from it by default,
so a dropped Colab session costs you at most one epoch.

## Layout

```
src/tiny_world_model/
  env.py              device + path resolution (laptop vs Colab)
  data/bouncing.py    synthetic dataset, no download needed
  models/predictor.py conv next-frame predictor
  training/train.py   loop with checkpoint/resume + AMP
scripts/train.py      CLI entrypoint
scripts/gpu_check.py  runtime preflight: device, AMP, throughput
notebooks/            Colab orchestration only
tests/                fast CPU tests, run in CI
```

## Checking a runtime

```bash
python scripts/gpu_check.py                # whatever device is available
python scripts/gpu_check.py --require-gpu  # fail if CUDA is missing
python scripts/gpu_check.py --device cpu   # force a device
```

Checks the accelerator is real, that mixed precision actually engages, that a training
step reduces the loss, and reports throughput and peak GPU memory. Exits non-zero on
failure. Cell 5 of the Colab notebook runs it with `--require-gpu`.

## Notebook hygiene

Committed notebook outputs create noisy diffs and can leak data. Strip them automatically:

```bash
uv pip install nbstripout
nbstripout --install --attributes .gitattributes
```
