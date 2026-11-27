#!/usr/bin/env python3
"""Preflight check for a training runtime.

Answers, in one command: is the accelerator real, does mixed precision actually
engage, does a training step reduce the loss, and how fast is this machine?

    python scripts/gpu_check.py                  # whatever device is available
    python scripts/gpu_check.py --require-gpu    # fail if CUDA is missing

The --require-gpu form is what you want in Colab: it turns "I forgot to switch
the runtime to a GPU" from a slow mystery into an immediate, named failure.
Exits non-zero if any check fails, so it works in CI too.
"""

from __future__ import annotations

import argparse
import sys
import time

import torch
from torch import nn

from tiny_world_model.data import BouncingBallDataset
from tiny_world_model.env import artifacts_dir, device_info, resolve_device
from tiny_world_model.models import TinyWorldModel

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"


class Checks:
    """Collects check results and prints them as they happen."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def add(self, status: str, name: str, detail: str = "") -> None:
        self.rows.append((status, name, detail))
        mark = {PASS: "  ok ", FAIL: "FAIL ", WARN: "warn "}[status]
        print(f"{mark} {name}" + (f" — {detail}" if detail else ""), flush=True)

    @property
    def failed(self) -> int:
        return sum(1 for s, _, _ in self.rows if s == FAIL)

    def summary(self) -> str:
        n = len(self.rows)
        warns = sum(1 for s, _, _ in self.rows if s == WARN)
        if self.failed:
            return f"{self.failed} of {n} checks FAILED ({warns} warnings)"
        return f"all {n} checks passed ({warns} warnings)"


def check_device(checks: Checks, require_gpu: bool, prefer: str | None = None) -> str:
    device = resolve_device(prefer)

    if device == "cuda":
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        checks.add(PASS, "accelerator", f"CUDA — {name}, {total:.1f} GB")
    elif require_gpu:
        checks.add(
            FAIL,
            "accelerator",
            f"--require-gpu was passed but device is '{device}'. "
            "In Colab: Runtime -> Change runtime type -> T4 GPU, then rerun.",
        )
    elif device == "mps":
        checks.add(WARN, "accelerator", "Apple MPS — fine locally, no CUDA features")
    else:
        checks.add(WARN, "accelerator", "CPU only — training will be slow")

    return device


def check_tensor_ops(checks: Checks, device: str) -> None:
    """A matmul that actually lands on the device. Catches a broken CUDA install."""
    try:
        a = torch.randn(256, 256, device=device)
        result = (a @ a).sum().item()
        if result != result:  # NaN
            checks.add(FAIL, "tensor ops", "matmul produced NaN")
        else:
            checks.add(PASS, "tensor ops", f"256x256 matmul on {device}")
    except Exception as exc:  # noqa: BLE001 - surfacing any backend failure is the point
        checks.add(FAIL, "tensor ops", f"{type(exc).__name__}: {exc}")


def check_amp(checks: Checks, device: str) -> bool:
    """Confirm autocast actually changes the compute dtype, rather than silently no-op.

    A passing AMP check is the difference between a 2-3x speedup and wondering
    why the GPU is no faster than the CPU.
    """
    if device != "cuda":
        checks.add(WARN, "mixed precision", f"skipped — not available on {device}")
        return False

    try:
        with torch.amp.autocast("cuda"):
            out = torch.randn(64, 64, device="cuda") @ torch.randn(64, 64, device="cuda")
        if out.dtype in (torch.float16, torch.bfloat16):
            checks.add(PASS, "mixed precision", f"autocast active, dtype={out.dtype}")
            return True
        checks.add(WARN, "mixed precision", f"autocast gave {out.dtype}, expected fp16/bf16")
        return False
    except Exception as exc:  # noqa: BLE001
        checks.add(FAIL, "mixed precision", f"{type(exc).__name__}: {exc}")
        return False


def check_training_step(checks: Checks, device: str, steps: int, batch_size: int) -> None:
    """Run real steps and assert the loss moves. Catches a model that cannot learn."""
    torch.manual_seed(0)
    dataset = BouncingBallDataset(n_rollouts=32, steps=16, size=32, context=4, seed=0)
    model = TinyWorldModel(context=4, width=32, depth=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    idx = torch.randint(0, len(dataset), (steps, batch_size))
    losses: list[float] = []

    for step in range(steps):
        ctx = torch.stack([torch.from_numpy(dataset[int(i)][0]) for i in idx[step]]).to(device)
        target = torch.stack([torch.from_numpy(dataset[int(i)][1]) for i in idx[step]]).to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            loss = loss_fn(model(ctx), target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.item())

    first, last = losses[0], losses[-1]
    if any(loss != loss for loss in losses):
        checks.add(FAIL, "training step", "loss became NaN")
    elif last < first:
        checks.add(PASS, "training step", f"loss {first:.4f} -> {last:.4f} over {steps} steps")
    else:
        checks.add(FAIL, "training step", f"loss did not fall: {first:.4f} -> {last:.4f}")


def check_throughput(checks: Checks, device: str, batch_size: int) -> None:
    """Measure steady-state samples/sec, excluding warmup."""
    torch.manual_seed(0)
    model = TinyWorldModel(context=4, width=64, depth=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    ctx = torch.randn(batch_size, 4, 32, 32, device=device)
    target = torch.rand(batch_size, 1, 32, 32, device=device)

    def one_step() -> None:
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            loss = loss_fn(model(ctx), target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    for _ in range(5):  # warmup: cuDNN autotuning and lazy init
        one_step()
    if device == "cuda":
        torch.cuda.synchronize()

    measured, t0 = 20, time.perf_counter()
    for _ in range(measured):
        one_step()
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    per_sec = measured * batch_size / elapsed
    ms = elapsed / measured * 1000
    checks.add(PASS, "throughput", f"{per_sec:,.0f} samples/s ({ms:.1f} ms/step @ bs={batch_size})")

    if device == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1e9
        checks.add(PASS, "gpu memory", f"peak {peak:.2f} GB allocated")


def check_checkpoint_roundtrip(checks: Checks, device: str) -> None:
    """Save and reload weights. This is what a Colab disconnect depends on."""
    from tiny_world_model.training.train import TrainConfig, load_checkpoint, save_checkpoint

    path = artifacts_dir() / "_gpu_check" / "checkpoint.pt"
    try:
        model = TinyWorldModel(context=4, width=16, depth=1).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        save_checkpoint(path, model, optimizer, epoch=7, cfg=TrainConfig())

        reloaded = TinyWorldModel(context=4, width=16, depth=1).to(device)
        epoch = load_checkpoint(path, reloaded, device=device)

        before, after = model.state_dict(), reloaded.state_dict()
        if before.keys() != after.keys():
            checks.add(FAIL, "checkpoint roundtrip", "state_dict keys differ after reload")
            return
        mismatched = [k for k in before if not torch.equal(before[k].cpu(), after[k].cpu())]
        if not mismatched and epoch == 7:
            checks.add(PASS, "checkpoint roundtrip", f"{len(before)} tensors match, epoch={epoch}")
        else:
            detail = f"epoch={epoch}"
            if mismatched:
                detail += f", {len(mismatched)} tensor(s) differ: {mismatched[:3]}"
            checks.add(FAIL, "checkpoint roundtrip", detail)
    except Exception as exc:  # noqa: BLE001
        checks.add(FAIL, "checkpoint roundtrip", f"{type(exc).__name__}: {exc}")
    finally:
        if path.exists():
            path.unlink()
            path.parent.rmdir()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="fail instead of warn when CUDA is unavailable",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="force a device (cuda/mps/cpu) instead of auto-detecting",
    )
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    print(device_info(), end="\n\n")
    if args.device:
        print(f"(device forced to '{args.device}' via --device)\n")

    checks = Checks()
    device = check_device(checks, args.require_gpu, args.device)

    # Without working tensor ops the rest is noise, so stop early.
    check_tensor_ops(checks, device)
    if checks.failed:
        print(f"\n{checks.summary()}")
        return 1

    check_amp(checks, device)
    check_training_step(checks, device, args.steps, args.batch_size)
    check_throughput(checks, device, args.batch_size)
    check_checkpoint_roundtrip(checks, device)

    print(f"\n{checks.summary()}")
    return 1 if checks.failed else 0


if __name__ == "__main__":
    sys.exit(main())
