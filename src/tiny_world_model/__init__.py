"""Tiny world model: a small next-frame/next-state predictor."""

__version__ = "0.1.0"

from tiny_world_model.env import device_info, resolve_device

__all__ = ["__version__", "device_info", "resolve_device"]
