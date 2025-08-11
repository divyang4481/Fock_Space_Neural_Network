"""Placeholder training script."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training entry point")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        f"Training placeholder with config={args.config} device={args.device} seed={args.seed}"
    )


if __name__ == "__main__":
    main()
