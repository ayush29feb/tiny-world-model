"""Fast CPU-only tests. These must pass before you bother spinning up a GPU."""

import numpy as np
import pytest

from tiny_world_model.data import BouncingBallDataset, make_rollout
from tiny_world_model.env import device_info, resolve_device

torch = pytest.importorskip("torch")

from tiny_world_model.models import TinyWorldModel


def test_rollout_shape_and_range():
    frames = make_rollout(steps=8, size=16)
    assert frames.shape == (8, 1, 16, 16)
    assert frames.dtype == np.float32
    assert frames.min() >= 0.0 and frames.max() <= 1.0


def test_rollout_has_motion():
    frames = make_rollout(steps=8, size=32, rng=np.random.default_rng(0))
    assert not np.allclose(frames[0], frames[-1]), "ball never moved"


def test_dataset_pairs():
    ds = BouncingBallDataset(n_rollouts=2, steps=10, size=16, context=3)
    assert len(ds) == 2 * (10 - 3)
    ctx, target = ds[0]
    assert ctx.shape == (3, 16, 16)
    assert target.shape == (1, 16, 16)


def test_dataset_rejects_short_rollouts():
    with pytest.raises(ValueError):
        BouncingBallDataset(steps=4, context=4)


def test_model_forward():
    model = TinyWorldModel(context=4, width=16, depth=1)
    out = model(torch.zeros(2, 4, 32, 32))
    assert out.shape == (2, 1, 32, 32)


def test_model_rejects_wrong_context():
    model = TinyWorldModel(context=4, width=16, depth=1)
    with pytest.raises(ValueError):
        model(torch.zeros(2, 2, 32, 32))


def test_autoregressive_rollout():
    model = TinyWorldModel(context=4, width=16, depth=1)
    imagined = model.rollout(torch.zeros(1, 4, 32, 32), steps=5)
    assert imagined.shape == (1, 5, 32, 32)
    assert torch.all((imagined >= 0) & (imagined <= 1))


def test_device_resolves():
    assert resolve_device() in {"cpu", "cuda", "mps"}
    assert resolve_device("cpu") == "cpu"
    assert "device" in str(device_info())
