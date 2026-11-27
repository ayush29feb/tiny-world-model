"""Training loop with checkpoint/resume, so a dropped Colab session is cheap."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from tiny_world_model.data import BouncingBallDataset
from tiny_world_model.env import artifacts_dir, resolve_device
from tiny_world_model.models import TinyWorldModel


@dataclass
class TrainConfig:
    epochs: int = 5
    batch_size: int = 64
    lr: float = 3e-4
    context: int = 4
    width: int = 64
    depth: int = 4
    n_rollouts: int = 512
    steps: int = 32
    size: int = 32
    seed: int = 0
    num_workers: int = 2
    device: str | None = None
    run_name: str = "default"
    # Colab GPUs (T4 and newer) get a solid speedup from mixed precision.
    amp: bool = True
    log_every: int = 50


def _ckpt_path(run_name: str) -> Path:
    return artifacts_dir() / run_name / "checkpoint.pt"


def save_checkpoint(path: Path, model, optimizer, epoch: int, cfg: TrainConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pt.tmp")
    # Write-then-rename: a runtime killed mid-save leaves the old ckpt intact.
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "config": asdict(cfg),
        },
        tmp,
    )
    tmp.replace(path)


def load_checkpoint(path: Path, model, optimizer=None, device: str = "cpu") -> int:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt.get("epoch", 0))


def train(cfg: TrainConfig | None = None, resume: bool = True) -> dict:
    cfg = cfg or TrainConfig()
    device = resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)

    dataset = BouncingBallDataset(
        n_rollouts=cfg.n_rollouts,
        steps=cfg.steps,
        size=cfg.size,
        context=cfg.context,
        seed=cfg.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )

    model = TinyWorldModel(context=cfg.context, width=cfg.width, depth=cfg.depth).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    use_amp = cfg.amp and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    ckpt = _ckpt_path(cfg.run_name)
    start_epoch = 0
    if resume and ckpt.exists():
        start_epoch = load_checkpoint(ckpt, model, optimizer, device=device)
        print(f"resumed from {ckpt} at epoch {start_epoch}")

    print(f"device={device} params={model.num_params():,} batches/epoch={len(loader)}")

    history: list[dict] = []
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        running, seen = 0.0, 0
        t0 = time.time()
        bar = tqdm(loader, desc=f"epoch {epoch + 1}/{cfg.epochs}", leave=False)

        for ctx, target in bar:
            ctx = ctx.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(ctx)
                loss = loss_fn(pred, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += loss.item() * ctx.size(0)
            seen += ctx.size(0)
            bar.set_postfix(loss=f"{running / max(seen, 1):.5f}")

        epoch_loss = running / max(seen, 1)
        entry = {"epoch": epoch + 1, "loss": epoch_loss, "seconds": time.time() - t0}
        history.append(entry)
        print(f"epoch {epoch + 1}: loss={epoch_loss:.5f} ({entry['seconds']:.1f}s)")

        save_checkpoint(ckpt, model, optimizer, epoch + 1, cfg)

    metrics = {"device": device, "params": model.num_params(), "history": history}
    (ckpt.parent / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
