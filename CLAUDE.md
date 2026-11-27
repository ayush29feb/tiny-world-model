# tiny-world-model

A tiny next-frame world model. Developed locally, trained on Google Colab GPUs, with
GitHub as the sync point.

## Architecture rule

**All logic lives in `src/tiny_world_model/`. Notebooks only orchestrate.**

Notebooks are for calling into the package, plotting, and inspecting results. When a cell
grows past a few lines of real logic, move it into the package and import it. This keeps
the code testable in CI and keeps git diffs readable.

## Dual-environment constraint

Every change must work both on a laptop (CPU/MPS) and on a Colab GPU:

- Never hardcode `"cuda"` — call `resolve_device()` from `tiny_world_model.env`.
- Never hardcode output paths — call `artifacts_dir()`, which handles Drive on Colab.
- Keep `torch` out of the hard dependencies in `pyproject.toml`. Colab ships its own CUDA
  build, and installing a second one breaks the runtime. It belongs in the `local` extra.
- Guard CUDA-only features (AMP, `pin_memory`, `GradScaler`) on the resolved device.

## Commands

```bash
source .venv/bin/activate
pytest -q                                              # fast, CPU-only
ruff check .
python scripts/train.py --epochs 1 --n-rollouts 32 --num-workers 0   # local smoke run
```

Run `pytest` before committing. Colab iterations are slow, so bugs should be caught locally.

## Checkpoints

`train()` resumes from `artifacts/<run_name>/checkpoint.pt` by default. Checkpoints are
saved write-then-rename so a killed runtime cannot corrupt them. Pass `resume=False` to
start fresh. Checkpoints are gitignored — they move through Drive, not git.

## Conventions

- Type hints on public functions; `from __future__ import annotations` at the top.
- Raise on invalid shapes/config early rather than letting broadcasting hide a bug.
- New dataset or model: add a shape test to `tests/test_smoke.py`.
