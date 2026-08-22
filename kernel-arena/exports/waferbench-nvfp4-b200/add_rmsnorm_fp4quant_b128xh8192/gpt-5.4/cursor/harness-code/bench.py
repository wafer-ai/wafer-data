"""Sustained GPU kernel benchmarking utilities.

Follows ThunderKittens 2.0 benchmarking convention:
- CUDA events: 2 events wrapping all reps (not per-rep)
- 500 warmup iterations (power-steady state)
- 100 profiling iterations back-to-back (no intermediate sync)
- L2 cache eviction via input group cycling
- 500ms thermal cooldown between kernels
- Bitwise-identical random inputs across all kernels
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import torch

_REAL_RECORD = torch.cuda.Event.record
_REAL_ELAPSED = torch.cuda.Event.elapsed_time


def _check_invariants() -> None:
    assert torch.cuda.Event.record is _REAL_RECORD, "Event.record was overridden"
    assert torch.cuda.Event.elapsed_time is _REAL_ELAPSED, (
        "Event.elapsed_time was overridden"
    )


@dataclass
class BenchResult:
    avg_us: float

    def summary(self, width: int = 30) -> str:
        return f"avg={self.avg_us:>9.1f} us"


class CyclingCallable:
    """Cycles through multiple input-group callables to naturally evict L2."""
    __slots__ = ("_fns", "_n", "_idx")

    def __init__(self, fns: list[Callable]) -> None:
        assert len(fns) >= 1
        self._fns = fns
        self._n = len(fns)
        self._idx = 0

    def __call__(self):
        fn = self._fns[self._idx % self._n]
        self._idx += 1
        return fn()

    def reset(self) -> None:
        self._idx = 0


def compute_num_input_groups(input_bytes: int, max_groups: int = 256) -> int:
    props = torch.cuda.get_device_properties(0)
    l2_size = props.L2_cache_size
    if input_bytes >= l2_size * 3:
        return 1
    n = int(l2_size * 3 / input_bytes) + 1
    return min(n, max_groups)


def _reset_if_cycling(fn: Callable) -> None:
    if isinstance(fn, CyclingCallable):
        fn.reset()


def bench_sustained(
    fns: dict[str, Callable],
    *,
    warmup: int = 500,
    rep: int = 100,
    cooldown_s: float = 0.5,
) -> dict[str, BenchResult]:
    """Sustained benchmark: 2 CUDA events wrapping all reps (TK 2.0 convention)."""
    _check_invariants()
    results: dict[str, BenchResult] = {}

    for name, fn in fns.items():
        _reset_if_cycling(fn)

        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(rep):
            fn()
        end.record()
        torch.cuda.synchronize()

        total_ms = start.elapsed_time(end)
        avg_us = total_ms * 1000.0 / rep
        results[name] = BenchResult(avg_us=avg_us)

        time.sleep(cooldown_s)

    return results


def print_results(label: str, results: dict[str, BenchResult], width: int = 35) -> None:
    print(f"\n  {label}")
    print(f"  {'─' * 80}")
    for name, r in results.items():
        print(f"  {name:>{width}s}: {r.summary()}")
