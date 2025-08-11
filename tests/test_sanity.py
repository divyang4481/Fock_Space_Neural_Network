"""Basic sanity tests for repository scaffolding."""

from pathlib import Path
import sys

# Ensure src/ is on the path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_import() -> None:
    import fsnn  # noqa: F401


def test_placeholder() -> None:
    assert True
