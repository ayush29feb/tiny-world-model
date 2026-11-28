"""Tests for the training loop, GIF rendering and the live notebook monitor."""

import io

import pytest

torch = pytest.importorskip("torch")

from tiny_world_model.dit import DiTConfig  # noqa: E402
from tiny_world_model.render import clips_to_gif  # noqa: E402
from tiny_world_model.train import TrainConfig, train  # noqa: E402

TINY = DiTConfig(frames=4, dim=32, depth=2, heads=2)


class Recorder:
    def __init__(self):
        self.logs, self.samples = [], []

    def on_log(self, step, metrics):
        self.logs.append((step, metrics))

    def on_samples(self, step, clips):
        self.samples.append((step, clips))


def tiny_config(**overrides):
    base = dict(
        model=TINY, steps=6, batch_size=2, num_workers=0, device="cpu", warmup_steps=2,
        log_every=2, sample_every=3, sample_clips=2, sample_steps=2,
    )
    base.update(overrides)
    return TrainConfig(**base)


def test_logs_and_samples_arrive_on_schedule():
    rec = Recorder()
    train(tiny_config(), monitor=rec)
    assert [s for s, _ in rec.logs] == [2, 4, 6]
    assert [s for s, _ in rec.samples] == [3, 6]
    for _, m in rec.logs:
        assert m["loss"] > 0 and m["clips_per_sec"] > 0 and m["lr"] > 0
    clips = rec.samples[-1][1]
    assert clips.shape == (2, 4, 48, 128) and clips.dtype == torch.bool


def test_warmup_ramps_the_learning_rate():
    rec = Recorder()
    train(tiny_config(steps=4, log_every=1, warmup_steps=4, lr=1e-3), monitor=rec)
    lrs = [m["lr"] for _, m in rec.logs]
    assert lrs == pytest.approx([2.5e-4, 5e-4, 7.5e-4, 1e-3])


def test_ema_weights_trail_the_trained_weights():
    ema = train(tiny_config(ema_decay=0.5), monitor=Recorder())
    assert any(p.abs().sum() > 0 for p in ema.out.parameters()), "EMA never moved"


def test_checkpoint_resume(tmp_path):
    first = Recorder()
    train(tiny_config(steps=4, checkpoint_dir=str(tmp_path), checkpoint_every=2), monitor=first)
    assert (tmp_path / "checkpoint.pt").exists()
    assert torch.load(tmp_path / "checkpoint.pt", weights_only=False)["step"] == 4

    second = Recorder()
    train(tiny_config(steps=6, checkpoint_dir=str(tmp_path), checkpoint_every=2), monitor=second)
    assert [s for s, _ in second.logs] == [6], "should resume at step 4, not start over"
    assert torch.load(tmp_path / "checkpoint.pt", weights_only=False)["step"] == 6


def test_gif_has_one_frame_per_clip_frame():
    from PIL import Image

    clips = torch.rand(5, 4, 48, 128) < 0.1
    img = Image.open(io.BytesIO(clips_to_gif(clips, cols=4, scale=2)))
    assert img.format == "GIF"
    assert img.n_frames == 4
    assert img.size == (4 * 256 + 5 * 4, 2 * 96 + 3 * 4)  # 4 columns, 2 rows


def test_live_monitor_runs_outside_a_notebook(tmp_path, capsys):
    pytest.importorskip("IPython")
    pytest.importorskip("matplotlib")
    from tiny_world_model.notebook import LiveMonitor

    monitor = LiveMonitor(save_dir=str(tmp_path))
    for step in (25, 50, 75):
        monitor.on_log(step, {"loss": 1.0 / step, "clips_per_sec": 300.0, "lr": 3e-4})
    monitor.on_samples(75, torch.rand(2, 4, 48, 128) < 0.1)
    assert (tmp_path / "samples_0000075.gif").exists()
    assert monitor._plot_png().startswith(b"\x89PNG")


# --- EMA warmup ------------------------------------------------------------------


def test_ema_decay_warms_up_to_its_maximum():
    from tiny_world_model.train import ema_decay_at

    decays = [ema_decay_at(n, 0.999) for n in (0, 10, 100, 1000, 100_000)]
    assert decays == sorted(decays), "decay should only rise"
    assert decays[0] < 0.2
    assert decays[-1] == 0.999
    # The share of the untrained model left in the EMA after 1,000 updates.
    leftover = 1.0
    for n in range(1000):
        leftover *= ema_decay_at(n, 0.999)
    assert leftover < 1e-3  # with a fixed 0.999 it would be 0.999**1000, about 37%


def _same_weights(a, b):
    return all(torch.equal(x, y) for x, y in zip(a.values(), b.values(), strict=True))


def test_checkpoint_records_ema_updates(tmp_path):
    cfg = tiny_config(steps=4, checkpoint_dir=str(tmp_path), checkpoint_every=2)
    train(cfg, monitor=Recorder())
    assert torch.load(tmp_path / "checkpoint.pt", weights_only=False)["ema_updates"] == 4


def test_resuming_a_pre_warmup_checkpoint_restarts_the_ema(tmp_path):
    cfg = tiny_config(steps=4, checkpoint_dir=str(tmp_path), checkpoint_every=2)
    train(cfg, monitor=Recorder())
    path = tmp_path / "checkpoint.pt"
    state = torch.load(path, weights_only=False)
    # Make it look like a checkpoint saved before the fix: no update count, and an
    # EMA that is nothing like the trained weights.
    del state["ema_updates"]
    state["ema"] = {k: torch.zeros_like(v) for k, v in state["ema"].items()}
    torch.save(state, path)

    ema = train(cfg, monitor=Recorder())  # already at step 4: resumes and trains 0 steps
    assert _same_weights(ema.state_dict(), state["model"])


def test_resuming_a_new_checkpoint_keeps_its_ema(tmp_path):
    cfg = tiny_config(steps=4, checkpoint_dir=str(tmp_path), checkpoint_every=2)
    train(cfg, monitor=Recorder())
    state = torch.load(tmp_path / "checkpoint.pt", weights_only=False)
    ema = train(cfg, monitor=Recorder())
    assert _same_weights(ema.state_dict(), state["ema"])
    assert not _same_weights(ema.state_dict(), state["model"])


def test_reset_ema_option_restarts_it(tmp_path):
    cfg = tiny_config(steps=4, checkpoint_dir=str(tmp_path), checkpoint_every=2)
    train(cfg, monitor=Recorder())
    state = torch.load(tmp_path / "checkpoint.pt", weights_only=False)
    reset = tiny_config(steps=4, checkpoint_dir=str(tmp_path), reset_ema=True)
    ema = train(reset, monitor=Recorder())
    assert _same_weights(ema.state_dict(), state["model"])
