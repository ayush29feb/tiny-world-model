"""Training loop for the flow-matching DiT on clips generated on the fly.

    from tiny_world_model.train import TrainConfig, train
    train(TrainConfig(steps=20_000, checkpoint_dir="/content/drive/MyDrive/twm/run1"))

Every `log_every` steps it reports the loss and throughput, and every
`sample_every` steps it generates clips from noise with the EMA weights. Pass
`monitor=` an object with `on_log(step, metrics)` and `on_samples(step, clips)`
methods to receive them (see tiny_world_model.notebook.LiveMonitor for Colab).

With `checkpoint_dir` set, it saves a checkpoint every `checkpoint_every` steps
and resumes from it automatically, so a dropped Colab session only loses the
steps since the last save.
"""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from games.dino.dataset import DinoClips
from tiny_world_model.dit import DiT, DiTConfig
from tiny_world_model.flow import flow_matching_loss, sample


@dataclass
class TrainConfig:
    model: DiTConfig = field(default_factory=DiTConfig)
    steps: int = 20_000
    batch_size: int = 32
    lr: float = 3e-4
    warmup_steps: int = 500
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    ema_decay: float = 0.999  # samples come from an exponential moving average of weights
    amp: bool = True  # fp16 mixed precision on CUDA; ignored elsewhere

    data_seed: int = 0
    num_workers: int = 2

    log_every: int = 50
    sample_every: int = 1000
    sample_clips: int = 8
    sample_steps: int = 50
    sample_seed: int = 1234  # the same noise every time, so you can watch samples improve

    checkpoint_dir: str | None = None
    checkpoint_every: int = 1000
    device: str | None = None


class PrintMonitor:
    """Default monitor: print a line per log, and a note when samples are made."""

    def on_log(self, step: int, metrics: dict) -> None:
        print(
            f"step {step:6d}  loss {metrics['loss']:.4f}  "
            f"{metrics['clips_per_sec']:.0f} clips/s  lr {metrics['lr']:.2e}",
            flush=True,
        )

    def on_samples(self, step: int, clips: torch.Tensor) -> None:
        print(f"step {step:6d}  generated {len(clips)} clips", flush=True)


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warmup, then constant."""
    return cfg.lr * min(1.0, (step + 1) / max(cfg.warmup_steps, 1))


@torch.no_grad()
def _update_ema(ema: DiT, model: DiT, decay: float) -> None:
    for e, p in zip(ema.parameters(), model.parameters(), strict=True):
        e.lerp_(p, 1 - decay)


def _checkpoint_path(cfg: TrainConfig) -> Path | None:
    return Path(cfg.checkpoint_dir) / "checkpoint.pt" if cfg.checkpoint_dir else None


def save_checkpoint(path: Path, step: int, model, ema, opt, scaler, cfg: TrainConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    # Write, then rename: a runtime killed mid-save leaves the previous checkpoint intact.
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "opt": opt.state_dict(),
            "scaler": scaler.state_dict(),
            "config": asdict(cfg),
        },
        tmp,
    )
    tmp.replace(path)


def train(cfg: TrainConfig | None = None, monitor=None) -> DiT:
    """Train and return the EMA model."""
    cfg = cfg or TrainConfig()
    monitor = monitor or PrintMonitor()
    device = cfg.device or default_device()
    use_amp = cfg.amp and device == "cuda"

    model = DiT(cfg.model).to(device)
    ema = copy.deepcopy(model).requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start = 0
    ckpt = _checkpoint_path(cfg)
    if ckpt is not None and ckpt.exists():
        state = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        opt.load_state_dict(state["opt"])
        scaler.load_state_dict(state["scaler"])
        start = state["step"]
        print(f"resumed from {ckpt} at step {start}", flush=True)

    # Clips have one more frame than actions; clip_len = frames - 1 gives the model's size.
    # Offsetting the seed by the resume step avoids replaying clips already trained on.
    dataset = DinoClips(seed=cfg.data_seed + start, clip_len=cfg.model.frames - 1)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=device == "cuda",
        persistent_workers=cfg.num_workers > 0,
    )
    batches = iter(loader)

    print(
        f"device={device}  amp={use_amp}  params={model.num_params():,}  "
        f"tokens/clip={cfg.model.num_tokens}  batch={cfg.batch_size}",
        flush=True,
    )

    model.train()
    window_loss, window_steps, window_clips = 0.0, 0, 0
    window_start, paused = time.perf_counter(), 0.0
    for step in range(start, cfg.steps):
        frames = next(batches)["frames"].to(device, non_blocking=True)

        for group in opt.param_groups:
            group["lr"] = _lr_at(step, cfg)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            loss = flow_matching_loss(model, frames)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(opt)
        scaler.update()
        _update_ema(ema, model, cfg.ema_decay)

        window_loss += loss.item()
        window_steps += 1
        window_clips += frames.shape[0]
        done = step + 1

        if done % cfg.log_every == 0 or done == cfg.steps:
            elapsed = time.perf_counter() - window_start - paused  # training time only
            monitor.on_log(
                done,
                {
                    "loss": window_loss / window_steps,
                    "clips_per_sec": window_clips / max(elapsed, 1e-9),
                    "lr": opt.param_groups[0]["lr"],
                },
            )
            window_loss, window_steps, window_clips = 0.0, 0, 0
            window_start, paused = time.perf_counter(), 0.0

        if done % cfg.sample_every == 0 or done == cfg.steps:
            sampling_start = time.perf_counter()
            ema.eval()
            gen = torch.Generator(device=device).manual_seed(cfg.sample_seed)
            clips = sample(ema, cfg.sample_clips, steps=cfg.sample_steps, generator=gen)
            monitor.on_samples(done, clips.cpu())
            paused += time.perf_counter() - sampling_start

        if ckpt is not None and (done % cfg.checkpoint_every == 0 or done == cfg.steps):
            save_checkpoint(ckpt, done, model, ema, opt, scaler, cfg)

    return ema
