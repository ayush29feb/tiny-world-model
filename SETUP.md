# Setup

Done — the repo is live and public at
https://github.com/ayush29feb/tiny-world-model, and CI is green.

Nothing needs configuring to train on Colab. Because the repo is public, the clone
needs no token and no Colab↔GitHub authorization.

## Open it on a GPU

```
https://colab.research.google.com/github/ayush29feb/tiny-world-model/blob/main/notebooks/colab_bootstrap.ipynb
```

Then **Runtime → Change runtime type → T4 GPU** and run the cells top to bottom.

## Local development

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install -e ".[local,dev]"
pytest -q
```

## The loop

| Where | What |
|---|---|
| Laptop (Claude Code) | Edit `src/`, `pytest -q`, commit, push |
| Colab | Re-run **cell 2** (git pull), then your training cell |

`%autoreload 2` plus the editable install means no runtime restart and no reinstall —
only when `pyproject.toml` dependencies change.

Checkpoints go to Drive (cell 6), so training resumes where it left off even if the
runtime disconnects.

## If you make the repo private later

Two things change:

1. Add a fine-grained token (Contents: **Read-only**) as a Colab secret named
   `GITHUB_TOKEN` via the 🔑 icon, with *Notebook access* on, and build the clone URL as
   `https://{token}@github.com/{REPO}.git`.
2. Opening the notebook by URL needs Colab authorized against GitHub:
   File → Open notebook → GitHub → tick **Include private repos**.

Never paste a token into a cell — cell contents are saved into the `.ipynb` and pushed.
Read it with `google.colab.userdata.get("GITHUB_TOKEN")` and scrub it from any git output.

## Pushing from Colab

Not set up, deliberately. Anonymous HTTPS can clone a public repo but cannot push to it.
Treat Colab as read-only for code: experiment in cells, make the real edit locally.
