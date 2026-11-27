#!/usr/bin/env python3
"""CLI entrypoint: python scripts/train.py --epochs 10 --run-name baseline"""

from __future__ import annotations

import argparse
from dataclasses import fields

from tiny_world_model.env import device_info
from tiny_world_model.training import TrainConfig, train


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    defaults = TrainConfig()
    for f in fields(TrainConfig):
        flag = "--" + f.name.replace("_", "-")
        current = getattr(defaults, f.name)
        if f.type == "bool":
            p.add_argument(flag, action="store_true", default=current)
            p.add_argument(f"--no-{f.name.replace('_', '-')}", dest=f.name, action="store_false")
        elif f.name == "device":
            p.add_argument(flag, type=str, default=None)
        else:
            p.add_argument(flag, type=type(current), default=current)
    p.add_argument("--no-resume", dest="resume", action="store_false", default=True)
    return p


def main() -> None:
    args = build_parser().parse_args()
    print(device_info(), end="\n\n")
    cfg_kwargs = {f.name: getattr(args, f.name) for f in fields(TrainConfig)}
    train(TrainConfig(**cfg_kwargs), resume=args.resume)


if __name__ == "__main__":
    main()
