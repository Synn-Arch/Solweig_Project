# SPDX-License-Identifier: GPL-3.0-only
"""T13 — dispatch cost model: measured-only backend selection (TASKS
T13, DESIGN 13.6). Torch-free; stdlib only.

Auto dispatch is a COST decision, never a correctness one. Every verdict
comes from RECORDED wall-clock measurements — there is no availability
probing, no heuristic, and no guessing:

* ``select(key)`` compares the minimum observed CPU cost against the
  minimum observed CUDA cost for that key; an unmeasured (or
  half-measured) key raises ``UnmeasuredCostError``;
* the tiny-edit rule falls out of the measurements: when the CPU side
  measures faster (typical for small edits where staging + launch dwarf
  the math), cpu is selected — and a CPU-selected execution must make
  ZERO CUDA device allocations (the monotonic sw_alloc counter is the
  fence; tests/ultrafast/test_cuda_dispatch_cost.py pins it);
* construction is restricted to the certified profile: a legacy runtime
  is refused outright (``DispatchError``), so the automatic path can
  never schedule an uncertified kernel.

This module owns no policy thresholds and performs no CUDA work itself;
T15 wires it into the runtime dispatcher.
"""
from __future__ import annotations

import sys
import time

__all__ = [
    "DispatchError", "UnmeasuredCostError", "Dispatcher",
]

_BACKENDS = ("cpu", "cuda")


class DispatchError(RuntimeError):
    pass


class UnmeasuredCostError(DispatchError):
    """Selection requested for a key lacking a measurement on at least
    one backend."""


def _host():
    host = sys.modules.get("sw_cuda_host")
    if host is None:
        raise DispatchError(
            "native/cuda/host.py must be loaded first (as 'sw_cuda_host')")
    return host


class Dispatcher:
    """Measured-only cpu/cuda selection for ONE certified runtime."""

    MEASURE_REPS = 5

    def __init__(self, rt) -> None:
        host = _host()
        if getattr(rt, "profile_id", None) != host.CANONICAL_CUDA_V1:
            raise DispatchError(
                f"auto dispatch is restricted to the certified profile "
                f"{host.CANONICAL_CUDA_V1!r}; refusing "
                f"{getattr(rt, 'profile_id', None)!r} (use the kernels "
                f"manually for other profiles)")
        self._rt = rt
        self._obs: dict[tuple, dict[str, list[float]]] = {}

    # -- observation ------------------------------------------------------------

    def observe(self, key, backend: str, seconds: float) -> None:
        if backend not in _BACKENDS:
            raise DispatchError(f"unknown backend {backend!r}")
        seconds = float(seconds)
        if seconds < 0:
            raise DispatchError("negative cost observation")
        self._obs.setdefault(key, {b: [] for b in _BACKENDS})[backend].append(
            seconds)

    def costs(self, key) -> dict:
        """Minimum observed seconds per backend (None = unmeasured)."""
        rec = self._obs.get(key)
        if rec is None:
            return {b: None for b in _BACKENDS}
        return {b: (min(v) if v else None) for b, v in rec.items()}

    # -- selection ----------------------------------------------------------------

    def select(self, key) -> str:
        c = self.costs(key)
        if c["cpu"] is None or c["cuda"] is None:
            raise UnmeasuredCostError(
                f"key {key!r} has no measured cpu/cuda pair "
                f"(cpu={'measured' if c['cpu'] is not None else 'missing'}, "
                f"cuda={'measured' if c['cuda'] is not None else 'missing'}); "
                f"measure() before select()")
        return "cpu" if c["cpu"] <= c["cuda"] else "cuda"

    def measure(self, key, cpu_fn, gpu_fn, reps: int = MEASURE_REPS) -> dict:
        """Time both callables (one warmup + ``reps`` timed each, minimum
        recorded) and return the observed costs."""
        cpu_fn()  # warmup (imports, allocators, numba compile...)
        gpu_fn()
        for _ in range(int(reps)):
            t0 = time.perf_counter()
            cpu_fn()
            self.observe(key, "cpu", time.perf_counter() - t0)
            t0 = time.perf_counter()
            gpu_fn()
            self.observe(key, "cuda", time.perf_counter() - t0)
        return self.costs(key)

    def run(self, key, cpu_fn, gpu_fn):
        """Select on the recorded measurements, then execute the chosen
        side exactly once."""
        backend = self.select(key)
        return cpu_fn() if backend == "cpu" else gpu_fn()
