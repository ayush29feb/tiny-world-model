"""Runtime/environment helpers so the same code runs on a laptop and on Colab."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


def in_colab() -> bool:
    """True when running inside a Google Colab runtime."""
    try:
        import google.colab  # noqa: F401
    except ImportError:
        return False
    return True


def resolve_device(prefer: str | None = None) -> str:
    """Pick the best available torch device.

    Order: explicit `prefer` -> CUDA (Colab GPU) -> MPS (Apple silicon) -> CPU.
    """
    if prefer:
        return prefer

    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def repo_root() -> Path:
    """Repo root, found by walking up to the directory holding pyproject.toml."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def artifacts_dir() -> Path:
    """Where checkpoints and samples go.

    Colab runtimes are ephemeral, so prefer Google Drive when it is mounted.
    Override anywhere with TWM_ARTIFACTS.
    """
    override = os.environ.get("TWM_ARTIFACTS")
    if override:
        path = Path(override)
    elif in_colab() and Path("/content/drive/MyDrive").exists():
        path = Path("/content/drive/MyDrive/tiny-world-model/artifacts")
    else:
        path = repo_root() / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _nvidia_smi() -> str | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


@dataclass
class EnvReport:
    python: str
    platform: str
    colab: bool
    device: str
    torch_version: str | None = None
    cuda_version: str | None = None
    gpu: str | None = None
    artifacts: str = field(default="")

    def __str__(self) -> str:
        lines = [
            f"python        {self.python}",
            f"platform      {self.platform}",
            f"colab         {self.colab}",
            f"torch         {self.torch_version or 'not installed'}",
            f"cuda          {self.cuda_version or '-'}",
            f"device        {self.device}",
            f"gpu           {self.gpu or '-'}",
            f"artifacts     {self.artifacts}",
        ]
        return "\n".join(lines)


def device_info() -> EnvReport:
    """One-call sanity check. Print this at the top of every Colab session."""
    torch_version = cuda_version = None
    try:
        import torch

        torch_version = torch.__version__
        cuda_version = torch.version.cuda
    except ImportError:
        pass

    return EnvReport(
        python=platform.python_version(),
        platform=platform.platform(),
        colab=in_colab(),
        device=resolve_device(),
        torch_version=torch_version,
        cuda_version=cuda_version,
        gpu=_nvidia_smi(),
        artifacts=str(artifacts_dir()),
    )
