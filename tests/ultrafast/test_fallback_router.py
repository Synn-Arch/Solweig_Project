# SPDX-License-Identifier: GPL-3.0-only
"""T11 fallback ROUTER gates: dense vs sparse vs full-domain decided by
cost/shape (measured, never assumed), the memory bound on huge work
lists (split/spill, not crash), and the dispatch seam that names the
torch-free numba backend explicitly.

RED witnesses pinned here:
  * a building (geometry) edit is NEVER served warm — the anchor veto
    holds even when classify_anchor accepted the candidate;
  * the lower-density lane is never forced: an anchor is bypassed when
    the MEASURED warm cost loses to the cold full solve;
  * a huge collect list never plans an unbounded in-memory result: the
    batches each fit the memory budget (1e6 steps stays a plan);
  * an unknown backend preference is BLOCKED, never guessed.
"""
from __future__ import annotations

import dataclasses

import pytest

from solweig_core import dispatch as core_dispatch
from solweig_core import request as core_request
from solweig_core.numba_cpu import fallback_router as fr
from solweig_core.numba_cpu import thermal
from solweig_core.status import DispatchStatus, RefusalReason

MUTATION_ANCHORS = {
    "fallback_router.route_fallback.geometry_veto":
        "test_geometry_edit_routes_regenerate_even_with_anchor",
    "fallback_router.route_fallback.warm_cost_gate":
        "test_warm_anchor_bypassed_when_measured_slower",
    "fallback_router._batch_collect_ts.memory_bound":
        "test_huge_collect_list_is_batched_not_oom",
    "fallback_router.route_fallback.window_validation":
        "test_malformed_requests_are_typed_errors",
}

COSTS = fr.MeasuredCosts(
    full_ms_per_step=100.0, sparse_ms_per_step=50.0, output_bytes_per_t=1000)


def anchor(resume_step: int) -> thermal.AnchorDecision:
    return thermal.AnchorDecision(
        selected=thermal.CheckpointCandidate(
            scene_revision=1, next_step=resume_step, thermal=True,
            input_fingerprint={}),
        resume_step=resume_step, invalidation="warm", reason=None)


def cold(reason="no checkpoint available — cold replay from step 0"):
    return thermal.AnchorDecision(
        selected=None, resume_step=0, invalidation="none", reason=reason)


# ---------------------------------------------------------------------------
# decision table
# ---------------------------------------------------------------------------

def test_geometry_edit_routes_regenerate_even_with_anchor() -> None:
    """RED witness: building edit + a fully-applicable anchor still
    REGENERATES (capture rebuild + cold full solve) — a warm start across
    a geometry edit is forbidden; the missing met/paint overlay is served
    by the regeneration, never by stale planes."""
    dec = fr.route_fallback(
        geometry_changed=True, r0=0, n_window=24,
        anchor_decision=anchor(12), costs=COSTS)
    assert dec.route == fr.ROUTE_REGENERATE
    assert dec.resume_step == 0
    assert dec.steps_to_advance == 24
    assert "anchor vetoed=True" in dec.reason


def test_met_edit_with_anchor_routes_warm_when_cheaper() -> None:
    dec = fr.route_fallback(
        geometry_changed=False, r0=12, n_window=24,
        anchor_decision=anchor(12), costs=COSTS)
    assert dec.route == fr.ROUTE_WARM_SPARSE
    assert dec.resume_step == 12
    assert dec.steps_to_advance == 12
    assert dec.est_cost_ms == pytest.approx(12 * 50.0)
    assert "measured cheaper" in dec.reason


def test_warm_anchor_bypassed_when_measured_slower() -> None:
    """RED witness (inverse direction): the sparse lane is NOT forced —
    when the measured warm cost loses to the cold full solve (e.g. an
    anchor at step 0 with a slower replay engine), routing goes cold
    full and the decision table records the veto."""
    slow_sparse = fr.MeasuredCosts(
        full_ms_per_step=100.0, sparse_ms_per_step=250.0,
        output_bytes_per_t=1000)
    dec = fr.route_fallback(
        geometry_changed=False, r0=0, n_window=24,
        anchor_decision=anchor(0), costs=slow_sparse)
    assert dec.route == fr.ROUTE_COLD_FULL
    assert dec.decision_table["warm_veto"].startswith("measured cost vetoes")
    assert dec.decision_table["warm_cost_ms"] == pytest.approx(24 * 250.0)
    assert dec.decision_table["cold_cost_ms"] == pytest.approx(24 * 100.0)


def test_no_anchor_routes_cold_full_with_recorded_reason() -> None:
    dec = fr.route_fallback(
        geometry_changed=False, r0=5, n_window=24,
        anchor_decision=cold("geometry-history mismatch at 'composed_scene'"),
        costs=COSTS)
    assert dec.route == fr.ROUTE_COLD_FULL
    assert "geometry-history" in dec.reason


def test_anchor_covering_window_advances_zero_steps() -> None:
    dec = fr.route_fallback(
        geometry_changed=False, r0=24, n_window=24,
        anchor_decision=anchor(24), costs=COSTS)
    assert dec.route == fr.ROUTE_WARM_SPARSE
    assert dec.steps_to_advance == 0
    assert dec.est_cost_ms == 0.0


# ---------------------------------------------------------------------------
# memory bound: huge work lists split, never OOM
# ---------------------------------------------------------------------------

def test_huge_collect_list_is_batched_not_oom() -> None:
    """1e6 requested timesteps under a 10 KiB budget with 1000 B/ts
    outputs -> ~103-ts batches; the router returns a PLAN (no output
    allocation ever happens here), so the request cannot OOM the solve
    step that consumes batches one at a time."""
    n = 10 ** 6
    dec = fr.route_fallback(
        geometry_changed=False, r0=0, n_window=n,
        anchor_decision=cold(), collect_ts=range(n),
        memory_budget_bytes=100 * 1024, costs=COSTS)
    batches = dec.collect_batches
    total = sum(len(b) for b in batches)
    assert total == n
    max_per_batch = (100 * 1024) // 1000
    for b in batches:
        assert len(b) <= max_per_batch
    # contiguity: the batch sequence is the sorted collect list split
    flat = [ts for b in batches for ts in b]
    assert flat == sorted(flat)
    assert flat[0] == 0 and flat[-1] == n - 1


def test_oversized_single_timestep_forms_its_own_batch() -> None:
    dec = fr.route_fallback(
        geometry_changed=False, r0=0, n_window=3,
        anchor_decision=cold(), collect_ts=[0, 1, 2],
        memory_budget_bytes=1, costs=COSTS)  # budget < one ts
    assert dec.collect_batches == ((0,), (1,), (2,))


def test_default_collect_is_the_whole_window() -> None:
    dec = fr.route_fallback(
        geometry_changed=False, r0=0, n_window=4,
        anchor_decision=cold(), costs=COSTS,
        memory_budget_bytes=10 ** 9)
    assert dec.collect_batches == ((0, 1, 2, 3),)


# ---------------------------------------------------------------------------
# typed errors: malformed requests are never silently normalized
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs,needle", [
    (dict(geometry_changed=False, r0=-1, n_window=24), "r0"),
    (dict(geometry_changed=False, r0=25, n_window=24), "r0"),
    (dict(geometry_changed=False, r0=0, n_window=0), "n_window"),
    (dict(geometry_changed=False, r0=0, n_window=4, collect_ts=[4]),
     "outside the window"),
    (dict(geometry_changed=False, r0=0, n_window=4,
          memory_budget_bytes=0), "memory_budget"),
])
def test_malformed_requests_are_typed_errors(kwargs, needle) -> None:
    with pytest.raises(fr.RouterError, match=needle):
        fr.route_fallback(anchor_decision=cold(), costs=COSTS, **kwargs)


def test_set_measured_costs_refuses_unmeasured() -> None:
    with pytest.raises(fr.RouterError, match="unmeasured"):
        fr.set_measured_costs(dataclasses.replace(
            COSTS, full_ms_per_step=0.0))
    # restore the module default table for the other tests in this run
    fr.set_measured_costs(COSTS)


# ---------------------------------------------------------------------------
# dispatch seam: the numba full-solve backend is named explicitly
# ---------------------------------------------------------------------------

def make_request(device="cpu", **overrides):
    kwargs = dict(
        logical_domain_id="site:test:100x80",
        rows=100,
        cols=80,
        origin_x_m=1000.0,
        origin_y_m=2000.0,
        pixel_size_m=2.0,
        read_window=core_request.PhysicalWindow(0, 100, 0, 80),
        write_window=core_request.PhysicalWindow(10, 20, 10, 20),
        time=core_request.TimeCoverage(0, None, 24),
        profile_id="canonical_cpu_v1",
        device=device,
    )
    kwargs.update(overrides)
    return core_request.SolveRequestView(**kwargs)


class TestDispatchSeam:
    def test_default_resolution_unchanged_legacy(self) -> None:
        result = core_dispatch.resolve(make_request())
        assert result.ready
        assert result.plan.backend == "legacy-torch-cpu"

    def test_numba_full_solve_backend_resolvable(self) -> None:
        result = core_dispatch.resolve(make_request(),
                                       prefer_backend="numba-full-solve")
        assert result.ready
        assert result.plan.backend == "numba-full-solve"
        assert result.plan.fallback_reason is None

    def test_unknown_preference_blocked_not_guessed(self) -> None:
        result = core_dispatch.resolve(make_request(),
                                       prefer_backend="numba-cuda-magic")
        assert result.status is DispatchStatus.BLOCKED
        assert result.plan is None
        assert result.refusal.reason is RefusalReason.UNKNOWN_BACKEND

    def test_cuda_preference_never_retargets_cpu(self) -> None:
        result = core_dispatch.resolve(make_request(device="cuda"),
                                       prefer_backend="numba-full-solve")
        assert result.status is DispatchStatus.BLOCKED
        assert result.refusal.reason is RefusalReason.CUDA_UNAVAILABLE
