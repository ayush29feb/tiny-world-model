# Setup steps you need to do

Everything in the repo is done and committed. Three things need your accounts.

## 1. Create the GitHub repo and push

The `gh` CLI isn't installed, so either install it:

```bash
brew install gh && gh auth login
```

then from this folder:

```bash
gh repo create tiny-world-model --private --source=. --remote=origin --push
```

Or create the repo by hand at https://github.com/new (name it `tiny-world-model`,
**don't** add a README or .gitignore — this folder already has them) and then:

```bash
git remote add origin https://github.com/ayush29feb/tiny-world-model.git && git push -u origin main
```

## 2. If the repo is private: make a token for Colab

Colab can't see a private repo without credentials.

1. Go to https://github.com/settings/personal-access-tokens/new (fine-grained token).
2. **Repository access** → Only select repositories → `tiny-world-model`.
3. **Permissions** → Repository permissions → Contents → **Read-only**.
   Read-only is enough to clone and pull. Only grant Read and write if you intend to
   commit from inside Colab.
4. Set a short expiry (90 days is plenty) and create it. Copy the token.

Then in Colab: click the 🔑 key icon in the left sidebar → **Add new secret** →
name it exactly `GITHUB_TOKEN`, paste the value, and turn on **Notebook access**.

The notebook reads it with `userdata.get("GITHUB_TOKEN")` and scrubs it from any git
output, so it won't end up in a saved cell. Never paste the token into a cell directly —
that *does* get saved into the notebook and pushed to GitHub.

## 3. Open the notebook in Colab

Once pushed, this URL opens it directly (replace the username):

```
https://colab.research.google.com/github/ayush29feb/tiny-world-model/blob/main/notebooks/colab_bootstrap.ipynb
```

For a private repo, Colab needs GitHub authorization to open it that way:
File → Open notebook → GitHub tab → tick **Include private repos**. Alternatively,
open a blank notebook and run cell 2's clone code — the token handles the rest.

Then: **Runtime → Change runtime type → T4 GPU**, and edit `REPO` in cell 2 to your
username. Run the cells top to bottom.

## Day-to-day after that

| Where | What |
|---|---|
| Laptop (Claude Code) | Edit `src/`, run `pytest -q`, commit, push |
| Colab | Re-run **cell 2** (git pull) and your training cell |

`%autoreload 2` plus the editable install means you don't restart the runtime or
reinstall — only when `pyproject.toml` dependencies change.

Checkpoints land in Drive (cell 5), so training resumes where it left off even if the
runtime disconnects.
