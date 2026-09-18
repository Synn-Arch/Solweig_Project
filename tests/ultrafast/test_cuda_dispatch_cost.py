# SPDX-License-Identifier: GPL-3.0-only
"""T13 GATE 3 — dispatch cost model: measured-only backend selection
(TASKS T13, DESIGN 13.6).

Auto dispatch is a COST decision, never a correctness one, and it is
restricted to the SAME certified profile (canonical_cuda_v1): a
legacy-profile runtime must be refused construction outright, so no
uncertified kernel can ever be picked by the automatic path.

Three contracts:

* measured-only — selection compares RECORDED measurements; an
  unmeasured (or half-measured) key raises UnmeasuredCostError instead
  of guessing (no availability probing, no heuristics);
* tiny-edit rule — when the CPU side is measured faster, the CPU is
  selected, and the CPU-selected execution must make ZERO CUDA device
  allocations (RED witness 6: 'CUDA allocation for a CPU-selected
  request');
* min-of-observations — repeated measurements select on the minimum
  (steady-state cost, not one cold outlier).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ULTRA_DIR = Path(__file__).resolve().parent
if str(ULTRA_DIR) not in sys.path:
    sys.path.insert(0, str(ULTRA_DIR))

from sw_cuda_harness import (  # noqa: E402
    load_host_module,
    require_canonical_runtime,
    require_runtime,
)

REPO_ROOT = ULTRA_DIR.parents[1]
NATIVE_CUDA = REPO_ROOT / "native" / "cuda"

_T13_MODULES: dict[str, object] = {}


def load_t13(name: str):
    """Load a native/cuda T13 module ONCE, sharing the harness's host
    module copy (single CudaRuntime identity)."""
    if name in _T13_MODULES:
        return _T13_MODULES[name]
    host = load_host_module()
    sys.modules.setdefault("sw_cuda_host", host)
    reg = f"sw_cuda_{name}"
    mod = sys.modules.get(reg)
    if mod is None:
        spec = importlib.util.spec_from_file_location(
            reg, NATIVE_CUDA / f"{name}.py")
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[reg] = mod
        spec.loader.exec_module(mod)
    _T13_MODULES[name] = mod
    return mod


@pytest.fixture(scope="module")
def rt():
    return require_canonical_runtime()


@pytest.fixture(scope="module")
def host():
    return load_host_module()


@pytest.fixture(scope="module")
def dispatch_cost():
    return load_t13("dispatch_cost")


@pytest.fixture(scope="module")
def residency():
    return load_t13("residency")


# ---------------------------------------------------------------------------
# profile restriction
# ---------------------------------------------------------------------------


class TestProfileRestriction:
    def test_legacy_profile_refused(self, host, dispatch_cost):
        rt_leg = require_runtime(host.LEGACY_CUDA_V1)
        with pytest.raises(dispatch_cost.DispatchError, match="certified"):
            dispatch_cost.Dispatcher(rt_leg)

    def test_constructor_performs_no_device_allocation(self, rt, residency,
                                                       dispatch_cost):
        before = residency.device_alloc_count()
        dispatch_cost.Dispatcher(rt)
        assert residency.device_alloc_count() == before


# ---------------------------------------------------------------------------
# measured-only selection
# ---------------------------------------------------------------------------


class TestMeasuredOnlySelection:
    def test_select_uses_measured_fastest_backend(self, rt, dispatch_cost):
        d = dispatch_cost.Dispatcher(rt)
        d.observe("tiny", "cpu", 1.0e-5)
        d.observe("tiny", "cuda", 5.0e-2)
        assert d.select("tiny") == "cpu"
        d.observe("big", "cpu", 5.0e-2)
        d.observe("big", "cuda", 1.0e-4)
        assert d.select("big") == "cuda"

    def test_unmeasured_key_refused(self, rt, dispatch_cost):
        d = dispatch_cost.Dispatcher(rt)
        with pytest.raises(dispatch_cost.UnmeasuredCostError):
            d.select("never-measured")

    def test_half_measured_key_refused(self, rt, dispatch_cost):
        d = dispatch_cost.Dispatcher(rt)
        d.observe("half", "cpu", 1.0e-5)
        with pytest.raises(dispatch_cost.UnmeasuredCostError):
            d.select("half")

    def test_selection_uses_min_of_observations(self, rt, dispatch_cost):
        d = dispatch_cost.Dispatcher(rt)
        # one cold outlier must not flip the verdict
        for s in (0.5, 0.4, 0.6):
            d.observe("k", "cpu", s)
        d.observe("k", "cuda", 0.45)
        assert d.select("k") == "cpu"
        d.observe("k", "cuda", 0.05)
        assert d.select("k") == "cuda"

    def test_costs_reported_per_key(self, rt, dispatch_cost):
        d = dispatch_cost.Dispatcher(rt)
        d.observe("k", "cpu", 0.25)
        d.observe("k", "cuda", 0.5)
        c = d.costs("k")
        assert c["cpu"] == pytest.approx(0.25)
        assert c["cuda"] == pytest.approx(0.5)

    def test_measure_records_both_sides(self, rt, dispatch_cost):
        d = dispatch_cost.Dispatcher(rt)
        calls = {"cpu": 0, "cuda": 0}

        def cpu_fn():
            calls["cpu"] += 1
            return sum(range(64))

        def gpu_fn():
            calls["cuda"] += 1
            return 2016

        got = d.measure("probe", cpu_fn, gpu_fn)
        assert got["cpu"] > 0.0 and got["cuda"] > 0.0
        assert calls["cpu"] >= 1 and calls["cuda"] >= 1
        assert d.select("probe") in ("cpu", "cuda")


# ---------------------------------------------------------------------------
# RED witness 6: CUDA allocation for a CPU-selected request
# ---------------------------------------------------------------------------


class TestCpuSelectedZeroCudaAlloc:
    def test_cpu_selected_run_allocates_nothing_on_device(self, rt, host,
                                                          residency,
                                                          dispatch_cost):
        """Tiny edit rule: for a small workload the CPU path measures
        faster (kernel launch + staging dwarf the math), so the
        dispatcher must pick cpu — and the CPU execution must not touch
        the device AT ALL (zero sw_alloc calls)."""
        tiny = [np.arange(16, dtype=np.float32)] * 2

        def cpu_fn():
            return float(np.add(tiny[0], tiny[1]).sum())

        def gpu_fn():
            add, _sub, _mul, _div = host.run_arith(rt, tiny[0], tiny[1])
            return float(np.asarray(add).sum())

        d = dispatch_cost.Dispatcher(rt)
        d.measure("tiny-arith", cpu_fn, gpu_fn)
        assert d.select("tiny-arith") == "cpu", (
            f"tiny workload must select cpu, costs={d.costs('tiny-arith')}"
        )
        before = residency.device_alloc_count()
        d.run("tiny-arith", cpu_fn, gpu_fn)
        after = residency.device_alloc_count()
        assert after == before, (
            f"CPU-selected request made {after - before} CUDA allocations"
        )

    def test_gpu_selected_run_executes_gpu_side(self, rt, host, residency,
                                                dispatch_cost):
        """When the recorded measurements favour cuda, run() must execute
        the GPU callable — verified by the monotonic alloc counter MOVING
        (the mirror image of the zero-alloc witness). The selection input
        here is recorded observations; which real workloads flip the
        verdict is the benchmark's question (bench_cuda_t13.py), not this
        gate's — on this host even 2048^2 np.exp beats kernel+transfers,
        so an 'honest' forced-flip workload does not exist at this size."""
        ran = {"gpu": False, "cpu": False}

        def cpu_fn():
            ran["cpu"] = True
            return 1

        def gpu_fn():
            ran["gpu"] = True
            x = np.arange(64, dtype=np.float32)
            add, _s, _m, _dv = host.run_arith(rt, x, x)
            return int(np.asarray(add).sum())

        d = dispatch_cost.Dispatcher(rt)
        d.observe("flip", "cpu", 1.0)
        d.observe("flip", "cuda", 1.0e-4)
        assert d.select("flip") == "cuda"
        before = residency.device_alloc_count()
        got = d.run("flip", cpu_fn, gpu_fn)
        after = residency.device_alloc_count()
        assert ran["gpu"] and not ran["cpu"]
        assert after > before, "cuda-selected run must touch the device"
        assert got == int(np.arange(64, dtype=np.float32).sum() * 2)
