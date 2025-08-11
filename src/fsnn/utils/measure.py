"""Utility functions for FLOPs and latency measurements."""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Callable, Iterable

import torch
from fvcore.nn import FlopCountAnalysis


def count_flops(model: torch.nn.Module, inputs: torch.Tensor) -> int:
    """Return total FLOPs for ``model`` with given ``inputs``."""
    model.eval()
    with torch.no_grad():
        flops = FlopCountAnalysis(model, inputs)
        return int(flops.total())


def measure_latency(
    func: Callable[[], torch.Tensor], *, repeats: int = 50, warmup: int = 10
) -> float:
    """Measure average latency (in seconds) of ``func`` on current device."""
    for _ in range(warmup):
        func()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    for _ in range(repeats):
        func()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end = time.perf_counter()
    return (end - start) / repeats


__all__ = ["count_flops", "measure_latency"]
