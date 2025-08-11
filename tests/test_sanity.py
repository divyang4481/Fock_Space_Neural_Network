"""Basic sanity tests for FSNN components."""
from pathlib import Path
import sys

# Ensure src/ is on the path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_import() -> None:
    import fsnn  # noqa: F401


def test_dynamic_linear_grow_prune() -> None:
    from fsnn.core import DynamicLinear
    import torch

    layer = DynamicLinear(4, 4)
    layer.grow(2)
    assert layer.out_features == 6
    x = torch.randn(1, 4)
    y = layer(x)
    assert y.shape == (1, 6)
    stats = layer.prune([0, 1])
    assert stats.removed == 2
    assert layer.out_features == 4
