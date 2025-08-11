"""Download datasets used in FSNN experiments."""
from __future__ import annotations

from pathlib import Path

from fsnn import get_dataloaders


def main() -> None:
    root = Path("./data")
    root.mkdir(exist_ok=True)
    for name in ["mnist", "adult"]:
        print(f"Preparing dataset {name}...")
        get_dataloaders(name, root)


if __name__ == "__main__":
    main()
