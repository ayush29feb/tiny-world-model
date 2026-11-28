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
import itertools
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from games.dino import clip_stream
from games.dino.dataset import DinoClips
from tiny_world_model.dit import DiT, DiTConfig
from tiny_world_model.flow import Schedule, flow_matching_loss, sample, validation_loss


@dataclass
class TrainConfig:
    model: DiTConfig = field(default_factory=DiTConfig)
    steps: int = 20_000
    batch_size: int = 64  # width-128 matmuls are small; bigger batches keep a GPU busier
    lr: float = 6e-4
    warmup_steps: int = 500
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    ema_decay: float = 0.999  # samples come from an exponential moving average of weights
    reset_ema: bool = False  # on resume, restart the EMA from the trained weights
    amp: bool = True  # fp16 mixed precision on CUDA; ignored elsewhere
    compile: bool = False  # torch.compile the model: faster steps, a few minutes to compile

    # Which noise levels training uses; see tiny_world_model.flow.Schedule.
    schedule: Schedule = field(default_factory=Schedule)

    data_seed: int = 0
    num_workers: int = 2

    log_every: int = 50
    sample_every: int = 1000
    sample_clips: int = 8
    sample_steps: int = 50
    sample_seed: int = 1234  # the same noise every time, so you can watch samples improve
    sample_shift: float | None = None  # shift of the sampling time grid; None = schedule's

    # Validation loss on fixed clips, fixed noise and evenly spaced noise levels. Unlike
    # the training loss it doesn't depend on the schedule, so runs can be compared.
    val_every: int = 250  # 0 turns it off
    val_clips: int = 32
    val_seed: int = 1  # a different seed from training data

    checkpoint_dir: str | None = None
    checkpoint_every: int = 1000
    device: str | None = None


class PrintMonitor:
    """Default monitor: print a line per log, and a note when samples are made."""

    def on_log(self, step: int, metrics: dict) -> None:
        gpu = ""
        if "gpu_util" in metrics:
            gpu += f"  GPU busy {metrics['gpu_util']:.0f}%"
        if "gpu_mem_gb" in metrics:
            gpu += f"  {metrics['gpu_mem_gb']:.1f} GB"
        if "val_loss" in metrics:
            gpu += f"  val {metrics['val_loss']:.4f} (EMA {metrics['val_loss_ema']:.4f})"
        print(
            f"step {step:6d}  loss {metrics['loss']:.4f}  "
            f"{metrics['clips_per_sec']:.0f} clips/s  lr {metrics['lr']:.2e}{gpu}",
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


def ema_decay_at(updates: int, max_decay: float) -> float:
    """EMA decay after `updates` updates: low at first, rising to `max_decay`.

    A fixed 0.999 from step 0 keeps 0.999^n of the untrained model, which is still
    37% after 1,000 steps. That model's output layer starts at zero, so the EMA
    predicts velocities that are far too small and samples stay close to the starting
    noise. Warming up the decay makes the EMA follow training closely at first.
    """
    return min(max_decay, (1 + updates) / (10 + updates))


@torch.no_grad()
def _update_ema(ema: DiT, model: DiT, decay: float) -> None:
    # One batched operation over all weight tensors, not one GPU launch per tensor.
    torch._foreach_lerp_(list(ema.parameters()), list(model.parameters()), 1 - decay)


def gpu_stats(device: str) -> dict:
    """How busy the GPU was recently (nvidia-smi) and peak memory PyTorch allocated."""
    if device != "cuda":
        return {}
    stats = {"gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9}
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2, check=True,
        ).stdout
        stats["gpu_util"] = float(out.split()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        pass  # no nvidia-smi: report memory only
    return stats


def _checkpoint_path(cfg: TrainConfig) -> Path | None:
    return Path(cfg.checkpoint_dir) / "checkpoint.pt" if cfg.checkpoint_dir else None


def save_checkpoint(
    path: Path, step: int, model, ema, opt, scaler, cfg: TrainConfig, ema_updates: int
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    # Write, then rename: a runtime killed mid-save leaves the previous checkpoint intact.
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "ema_updates": ema_updates,
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
    # Fused AdamW on CUDA runs the step as one kernel, and checks fp16 overflow on the
    # GPU instead of GradScaler reading it back to the CPU every step.
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, fused=device == "cuda"
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start, ema_updates = 0, 0
    ckpt = _checkpoint_path(cfg)
    if ckpt is not None and ckpt.exists():
        state = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        scaler.load_state_dict(state["scaler"])
        start = state["step"]
        # Checkpoints from before the EMA warmup have no "ema_updates", and their EMA
        # still holds a large share of the untrained model. Restart it from the
        # trained weights instead.
        if cfg.reset_ema or "ema_updates" not in state:
            ema.load_state_dict(state["model"])
            why = "asked to" if cfg.reset_ema else "checkpoint predates EMA warmup"
            print(f"resumed from {ckpt} at step {start}; EMA restarted ({why})", flush=True)
        else:
            ema.load_state_dict(state["ema"])
            ema_updates = state["ema_updates"]
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

    # A fixed validation set: clips from a different seed, and fixed noise to go with them.
    val_frames = val_noise = None
    if cfg.val_every > 0:
        clips = itertools.islice(
            clip_stream(cfg.val_seed, clip_len=cfg.model.frames - 1), cfg.val_clips
        )
        val_frames = torch.stack([torch.from_numpy(c.frames) for c in clips]).to(device)
        gen = torch.Generator(device=device).manual_seed(cfg.val_seed)
        val_noise = torch.randn(
            cfg.val_clips, cfg.model.num_tokens, cfg.model.patching.patch_dim,
            generator=gen, device=device,
        )

    # Compile a wrapper and keep `model` itself for checkpoints, so saved weights don't
    # get the wrapper's "_orig_mod." prefix and load either way.
    step_model = torch.compile(model) if cfg.compile else model

    print(
        f"device={device}  amp={use_amp}  params={model.num_params():,}  "
        f"tokens/clip={cfg.model.num_tokens}  batch={cfg.batch_size}  compile={cfg.compile}",
        flush=True,
    )

    model.train()
    # The running loss stays on the device and is read only when logging, so the CPU
    # never waits for the GPU between steps and can queue work ahead.
    window_loss = torch.zeros((), device=device)
    window_steps, window_clips = 0, 0
    window_start, paused = time.perf_counter(), 0.0
    for step in range(start, cfg.steps):
        frames = next(batches)["frames"].to(device, non_blocking=True)

        for group in opt.param_groups:
            group["lr"] = _lr_at(step, cfg)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            loss = flow_matching_loss(step_model, frames, cfg.schedule)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(opt)
        scaler.update()
        _update_ema(ema, model, ema_decay_at(ema_updates, cfg.ema_decay))
        ema_updates += 1

        window_loss += loss.detach()
        window_steps += 1
        window_clips += frames.shape[0]
        done = step + 1

        val = {}
        if cfg.val_every > 0 and (done % cfg.val_every == 0 or done == cfg.steps):
            val_start = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                model.eval()
                val["val_loss"] = validation_loss(model, val_frames, val_noise)
                model.train()
                ema.eval()
                val["val_loss_ema"] = validation_loss(ema, val_frames, val_noise)
            paused += time.perf_counter() - val_start

        if done % cfg.log_every == 0 or done == cfg.steps or val:
            mean_loss = window_loss.item() / window_steps  # waits for queued GPU work
            elapsed = time.perf_counter() - window_start - paused  # training time only
            monitor.on_log(
                done,
                {
                    "loss": mean_loss,
                    "clips_per_sec": window_clips / max(elapsed, 1e-9),
                    "lr": opt.param_groups[0]["lr"],
                    **gpu_stats(device),
                    **val,
                },
            )
            window_loss = torch.zeros((), device=device)
            window_steps, window_clips = 0, 0
            window_start, paused = time.perf_counter(), 0.0

        if done % cfg.sample_every == 0 or done == cfg.steps:
            sampling_start = time.perf_counter()
            ema.eval()
            gen = torch.Generator(device=device).manual_seed(cfg.sample_seed)
            shift = cfg.schedule.shift if cfg.sample_shift is None else cfg.sample_shift
            clips = sample(
                ema, cfg.sample_clips, steps=cfg.sample_steps, generator=gen, shift=shift
            )
            monitor.on_samples(done, clips.cpu())
            paused += time.perf_counter() - sampling_start

        if ckpt is not None and (done % cfg.checkpoint_every == 0 or done == cfg.steps):
            save_checkpoint(ckpt, done, model, ema, opt, scaler, cfg, ema_updates)

    return ema
