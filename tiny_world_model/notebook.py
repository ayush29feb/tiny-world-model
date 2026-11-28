"""Live training display for Colab / Jupyter: a loss plot and generated clips that
update in place while training runs.

    from tiny_world_model.notebook import LiveMonitor
    train(cfg, monitor=LiveMonitor(save_dir=cfg.checkpoint_dir))

Needs IPython, matplotlib and Pillow, which Colab already has.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np

from tiny_world_model.render import clips_to_gif


class LiveMonitor:
    def __init__(self, save_dir: str | None = None, smoothing: float = 0.6) -> None:
        from IPython.display import HTML, DisplayHandle

        self.save_dir = Path(save_dir) if save_dir else None
        self.smoothing = smoothing
        self.steps: list[int] = []
        self.losses: list[float] = []
        self.speeds: list[float] = []
        # Explicit handles, so each output updates in place. (display(display_id=True)
        # returns None outside a notebook, which would break running this in tests.)
        self.plot, self.samples = DisplayHandle(), DisplayHandle()
        self.plot.display(HTML("<i>Loss plot appears after the first log.</i>"))
        self.samples.display(HTML("<i>Generated clips appear here.</i>"))

    def on_log(self, step: int, metrics: dict) -> None:
        from IPython.display import Image

        self.steps.append(step)
        self.losses.append(metrics["loss"])
        self.speeds.append(metrics["clips_per_sec"])
        self.plot.update(Image(data=self._plot_png(), format="png"))

    def on_samples(self, step: int, clips) -> None:
        from IPython.display import HTML

        gif = clips_to_gif(clips)
        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            (self.save_dir / f"samples_{step:07d}.gif").write_bytes(gif)
        encoded = base64.b64encode(gif).decode()
        on = float(np.asarray(clips).mean())
        self.samples.update(
            HTML(
                f"<b>Generated from noise at step {step:,}</b> (EMA weights, same noise "
                f"each time). Pixels on: <b>{on:.1%}</b> (real clips: about 5%; "
                f"near 50% means still noise)<br><img src='data:image/gif;base64,{encoded}'>"
            )
        )

    def _smoothed(self) -> list[float]:
        out, value = [], None
        for x in self.losses:
            value = x if value is None else self.smoothing * value + (1 - self.smoothing) * x
            out.append(value)
        return out

    def _plot_png(self) -> bytes:
        from matplotlib.figure import Figure  # no pyplot: nothing is auto-displayed

        fig = Figure(figsize=(11, 3.2), dpi=100)
        loss_ax, speed_ax = fig.subplots(1, 2, gridspec_kw={"width_ratios": [2, 1]})

        loss_ax.plot(self.steps, self.losses, color="#c8c8c8", lw=1, label="loss")
        loss_ax.plot(self.steps, self._smoothed(), color="#2a6fdb", lw=2, label="smoothed")
        loss_ax.set_yscale("log")
        loss_ax.set_xlabel("step")
        loss_ax.set_title(f"flow-matching loss: {self.losses[-1]:.4f} at step {self.steps[-1]:,}")
        loss_ax.grid(alpha=0.3, which="both")
        loss_ax.legend(frameon=False)

        speed_ax.plot(self.steps, self.speeds, color="#2a6fdb", lw=1.5)
        speed_ax.set_ylim(bottom=0)
        speed_ax.set_xlabel("step")
        speed_ax.set_title(f"training speed: {self.speeds[-1]:.0f} clips/s")
        speed_ax.grid(alpha=0.3)

        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        return buf.getvalue()
