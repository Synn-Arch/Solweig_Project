# SPDX-License-Identifier: GPL-3.0-only
"""Routing policy wave: subscriber-gated chases, exact-lane demotion
escape, idle once-daily reconciliation, honest ETAs, routing metrics.

Pins the adjudicated policy (routing-design report; live incident: five
dead-session full-tile chases starved a live user ~50 min):

* **R1** — the exact lane mints a chase only for a workspace somebody is
  watching (SSE subscriber) or that already owns a nonterminal exact job.
  A gated-out workspace keeps its recoverable epochs and gains durable
  reconciliation debt instead. A hub-less runner stays ungated (legacy
  unit-harness behavior).
* **R2** — exact-lane solves escape windowed mode only at a 0.95 dirty
  fraction, threaded to BOTH coupled sites (worker mode choice and
  planner safety policy); legacy ``/edits`` jobs keep the 0.30 default.
  A reconciliation-marked job routes a forced full.
* **R4** — a quiet workspace (no subscribers, sustained past the grace,
  empty queue) with a recoverable revision gap mints ONE full-tile
  reconcile per day; a superseded attempt consumes nothing; only a
  completion owns the day; durable debt survives a restart.
* **R5** — the job body carries an honest ETA (``eta_seconds`` +
  ``eta_basis``) from the scenario's own completed-job history: p80
  history, disclosed 480 s static fallback for fulls, or null when
  unknown — never a guess.
* **R6** — published metrics lift the worker's TRUE pre-mode dirty
  fraction and ``demoted_by_fraction`` verdict, so a full-mode job can
  always say why it is full.

All written failing-first against the pre-wave tree.
"""

from __future__ import annotations

import logging
import queue
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from solweig_gpu.server import executor_bridge
from solweig_gpu.server import routes_jobs
from solweig_gpu.server.executor_bridge import (
    EXACT_LANE_FULL_RECOMPUTE_FRACTION,
    RECONCILE_FULL_RECOMPUTE_FRACTION,
)
from solweig_gpu.server.jobs import (
    JobRunner,
    RunnerContext,
    SiteRegistry,
    SolveRequest,
    SolveResult,
    job_eta_fields,
)
from solweig_gpu.server.store import Store, _now_utc

from tests.test_incremental_worker import science_site  # noqa: F401 (fixture)
from tests.test_realtime_epochs import (
    WS,
    make_store,
    make_workspace,
    op_item,
)
from tests.test_realtime_r2_exact_lane import (
    SseSession,
    _native_veg_add_item,
    _post_native_ops,
    _quiesce_workspace,
    close_epoch,
    exact_jobs,
    job_rows,
)
from tests.test_server_api import SITE_ID, FakeSolver, make_app, make_site_cache
from tests.test_server_universal_edits import family_site  # noqa: F401 (fixture)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class CountingHub:
    """BroadcastHub stand-in whose subscriber count the test controls."""

    def __init__(self, count: int = 0) -> None:
        self.count = count

    def subscriber_count(self, workspace_id: str | None = None) -> int:
        return self.count


def make_runner(
    tmp_path: Path,
    store: Store,
    broadcast_hub: Any = None,
    *,
    reconcile_idle_grace_s: float = 300.0,
) -> JobRunner:
    """A JobRunner wired like the app's, worker NOT started (unit seams)."""
    context = RunnerContext(
        store=store,
        sites=SiteRegistry({SITE_ID: {"cache_dir": make_site_cache(tmp_path)}}),
        state_root=tmp_path / "state",
    )
    return JobRunner(
        context,
        solver_factory=lambda ctx: (lambda request, progress: None),
        coalescing_window_ms=0.0,
        broadcast_hub=broadcast_hub,
        reconcile_idle_grace_s=reconcile_idle_grace_s,
    )


def today_utc() -> str:
    return _now_utc()[:10]


def _drain_queue(runner: JobRunner) -> None:
    """Empty the harness runner's dispatch queue (its worker thread never
    started) so the next idle scan sees a quiet runner."""
    while True:
        try:
            runner._queue.get_nowait()
        except queue.Empty:
            break


# ---------------------------------------------------------------------------
# R1: subscription-gated chase minting
# ---------------------------------------------------------------------------


def test_r1_subscribed_workspace_mints_chase(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    make_workspace(store, WS)
    hub = CountingHub(count=1)
    runner = make_runner(tmp_path, store, hub)

    store.append_operations(WS, [op_item("op_r1_1")])
    close_epoch(store, epoch_id=0)

    runner._exact_lane_tick()
    pending = [row for row in exact_jobs(store, WS) if row["status"] == "queued"]
    assert len(pending) == 1, job_rows(store, WS)
    assert store.reconcile_state(WS) == (None, False), (
        "a watched chase must never mint reconciliation debt"
    )


def test_r1_unwatched_workspace_defers_and_owes(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    make_workspace(store, WS)
    hub = CountingHub(count=0)
    runner = make_runner(tmp_path, store, hub)

    store.append_operations(WS, [op_item("op_r1_1")])
    close_epoch(store, epoch_id=0)

    runner._exact_lane_tick()
    assert exact_jobs(store, WS) == [], "a dead session must not mint a chase"
    # The epochs stay recoverable — nothing was settled.
    assert {e.workspace_id for e in store.recoverable_epochs()} == {WS}
    # The revision gap became durable reconciliation debt.
    assert store.reconcile_state(WS) == ("1970-01-01", True)
    assert store.workspaces_owing_reconcile() == [WS]


def test_r1_live_exact_job_keeps_lane_alive(tmp_path: Path) -> None:
    """Zero subscribers but a nonterminal exact job exists: the lane keeps
    chasing (the job's owner is owed the newer revision)."""
    store = make_store(tmp_path)
    make_workspace(store, WS)
    hub = CountingHub(count=0)
    runner = make_runner(tmp_path, store, hub)

    store.append_operations(WS, [op_item("op_r1_1")])
    close_epoch(store, epoch_id=0)
    # An audience-free first mint cannot happen under the gate, so seed the
    # nonterminal exact job the way a subscribed era would have left it.
    with store._write() as conn:
        first = Store._insert_job_locked(
            conn,
            WS,
            target_scene_version=1,
            request={"exact_lane": {"target_revision": 1}},
            edit_watermark=0,
        )
    store.mark_job_running(first)

    # A second epoch commits: the RUNNING exact job is the audience.
    store.append_operations(WS, [op_item("op_r1_2")])
    store.mark_epoch_status(WS, 1, "closed")
    committed = store.commit_epoch_reduction(
        WS, 1, families={"vegetation_geometry": {"objects": {}}}
    )
    assert committed is not None
    runner._exact_lane_tick()

    statuses = sorted(
        (row["status"], row["request"]["exact_lane"]["target_revision"])
        for row in exact_jobs(store, WS)
    )
    assert statuses == [("queued", 2), ("running", 1)], statuses
    assert store.reconcile_state(WS) == (None, False), (
        "a live exact job means no deferral debt"
    )


def test_r1_hubless_runner_stays_ungated(tmp_path: Path) -> None:
    """No hub (legacy unit harness / hub-free deployment): legacy behavior."""
    store = make_store(tmp_path)
    make_workspace(store, WS)
    runner = make_runner(tmp_path, store, broadcast_hub=None)

    store.append_operations(WS, [op_item("op_r1_1")])
    close_epoch(store, epoch_id=0)
    runner._exact_lane_tick()
    assert len(exact_jobs(store, WS)) == 1
    assert store.reconcile_state(WS) == (None, False)


def test_r1_subscriber_arrival_discharges_gap_with_local_completion(
    tmp_path: Path,
) -> None:
    """A gated-out gap is chased when a subscriber arrives, and a LOCAL
    completion discharges the debt WITHOUT consuming the daily full."""
    store = make_store(tmp_path)
    make_workspace(store, WS)
    hub = CountingHub(count=0)
    runner = make_runner(tmp_path, store, hub)

    store.publish_result(
        WS, 0, manifest={"checksum": "c0", "scene_version": 0}, payload=b"p0", exact=True
    )
    store.append_operations(WS, [op_item("op_r1_1")])
    close_epoch(store, epoch_id=0)
    runner._exact_lane_tick()
    assert store.reconcile_state(WS)[1] is True

    # The subscriber arrives: the gate opens (the gated tick stamped no
    # churn-guard entry, so the same revision is mintable).
    hub.count = 1
    runner._exact_lane_tick()
    pending = [row for row in exact_jobs(store, WS) if row["status"] == "queued"]
    assert len(pending) == 1, "a subscribed workspace must chase its debt"

    # LOCAL completion: debt discharged, daily cap untouched.
    job_id = pending[0]["job_id"]
    store.mark_job_running(job_id)
    runner._finalize(
        store.require_job(job_id),
        store.require_scenario(WS),
        SolveResult(status="no-op", scene_version=1),
        duration_ms=1.0,
    )
    assert store.require_job(job_id).status == "complete"
    last_completed, owed = store.reconcile_state(WS)
    assert owed is False
    # The sentinel date survives (the row exists), but the cap is judged
    # against TODAY: a local settle consumes nothing.
    assert last_completed != today_utc(), "a local settle must not consume the daily full"


# ---------------------------------------------------------------------------
# R2: exact-lane demotion escape at both coupled sites
# ---------------------------------------------------------------------------


def _request(exact_lane: dict[str, Any] | None) -> SolveRequest:
    return SolveRequest(
        job_id="job_r2",
        scenario_id=WS,
        site_id=SITE_ID,
        target_scene_version=1,
        edit_watermark=0,
        trees=(),
        events=(),
        requested={},
        grid={"rows": 8, "cols": 8, "pixel_size_m": 2.0, "origin_x_m": 0.0,
              "origin_y_m": 0.0, "time_steps": 1},
        exact_lane=exact_lane,
    )


def test_r2_bridge_fraction_for_each_lane() -> None:
    """Legacy jobs keep the worker/planner default; exact lane escapes at
    0.95; a reconcile marker routes a forced full."""
    assert executor_bridge._exact_lane_recompute_fraction(_request(None)) is None
    assert executor_bridge._exact_lane_recompute_fraction(
        _request({"target_revision": 1, "consumed_revision": 0,
                  "consumed_seq": 0, "span_hi": 1})
    ) == EXACT_LANE_FULL_RECOMPUTE_FRACTION == 0.95
    assert executor_bridge._exact_lane_recompute_fraction(
        _request({"target_revision": 1, "consumed_revision": 0,
                  "consumed_seq": 0, "span_hi": 1, "reconcile": True})
    ) == RECONCILE_FULL_RECOMPUTE_FRACTION


@pytest.mark.scientific
def test_r2_executor_threads_both_coupled_sites(science_site) -> None:
    """``PlanExecutor(full_recompute_fraction=...)`` moves the WORKER's mode
    choice and the PLANNER's safety policy together (the _reconcile_scope
    contract), without leaking the routing override into the adapter-facing
    site-adjusted influence config."""
    from tests.test_incremental_worker import DATE_STR, _state_layer
    from solweig_gpu.incremental.executor import PlanExecutor

    layer = _state_layer(science_site, "baseline")
    common = dict(
        cache=science_site.cache_a,
        layer=_state_layer(science_site, "baseline"),
        site_dir=science_site.site_dir,
        results_root=science_site.results_root("r2_exec"),
        selected_date_str=DATE_STR,
    )
    assert layer is not common["layer"]  # one TreeLayer per executor

    default_executor = PlanExecutor(**common)
    assert default_executor._worker.influence_config.full_recompute_fraction == 0.30
    assert default_executor._planner._policy.full_recompute_fraction == 0.30

    exact_executor = PlanExecutor(
        **{**common, "results_root": science_site.results_root("r2_exec_exact")},
        full_recompute_fraction=EXACT_LANE_FULL_RECOMPUTE_FRACTION,
    )
    assert (
        exact_executor._worker.influence_config.full_recompute_fraction == 0.95
    )
    assert exact_executor._planner._policy.full_recompute_fraction == 0.95
    # The adapter-facing config keeps the site-adjusted PHYSICS derivation:
    # sun floor and shadow length identical to the default executor's. (The
    # routing fraction rides the replace()d copy but is inert on that path —
    # the adapter's dirty windows read only the sun/shadow fields, and the
    # mode choice takes its fraction as a separate worker-side kwarg.)
    for field in (
        "minimum_direct_sun_altitude_deg",
        "lowest_sky_patch_altitude_deg",
        "maximum_shadow_length_m",
        "safety_margin_m",
    ):
        assert getattr(exact_executor._influence_config, field) == getattr(
            default_executor._influence_config, field
        ), field


def test_r2_planner_036_dirty_stays_local_at_095() -> None:
    """The planner site of the coupled pair: a 36%-of-tile window — well
    past the legacy 0.30 demotion — stays WINDOWS under the exact-lane
    policy, and still demotes FULL at the legacy default."""
    from solweig_gpu.incremental.edit_types import SpatialScope
    from solweig_gpu.incremental.geometry import RasterWindow
    from solweig_gpu.incremental.planner import ConservativeSafetyPolicy

    from tests.test_incremental_edit_engine import (
        initial_state,
        make_planner,
        scopes_by_node,
        veg_delta,
        veg_edit,
    )

    big_window = RasterWindow(0, 300, 0, 300)
    legacy = make_planner().plan(
        [veg_edit(delta=veg_delta(windows=(big_window,)))], initial_state()
    )
    assert any(scope is SpatialScope.FULL for scope in scopes_by_node(legacy).values()), (
        "the legacy 0.30 default must still demote a 36% dirty window"
    )
    assert any("demoted to full" in r for r in legacy.fallback_reasons)

    exact = make_planner(
        policy=ConservativeSafetyPolicy(
            full_recompute_fraction=EXACT_LANE_FULL_RECOMPUTE_FRACTION
        )
    ).plan([veg_edit(delta=veg_delta(windows=(big_window,)))], initial_state())
    assert all(
        scope is not SpatialScope.FULL for scope in scopes_by_node(exact).values()
    ), "the exact lane must keep a 36% dirty window incremental (0.95 escape)"
    assert not any("demoted to full" in r for r in exact.fallback_reasons)


@pytest.mark.scientific
def test_r2_worker_single_tree_stays_local_at_095(science_site) -> None:
    """The worker site of the coupled pair (real solve): an ordinary
    small-window edit that a 0.02 threshold full-tiles stays LOCAL under
    the exact-lane 0.95 threshold, and both legs carry the R6 routing
    diagnostics (TRUE pre-mode dirty fraction + demotion verdict)."""
    from solweig_gpu.incremental.geometry import InfluenceConfig

    from tests.test_incremental_worker import _local_worker, _state_layer

    legacy = _local_worker(
        science_site,
        _state_layer(science_site, "add"),
        science_site.results_root("r2_worker_legacy"),
        config=InfluenceConfig(full_recompute_fraction=0.02),
    )
    legacy.bump_scene_revision()
    outcome = legacy.run()
    assert outcome.published and outcome.mode == "full"
    assert outcome.diagnostics["demoted_by_fraction"] is True, outcome.diagnostics
    assert 0.0 < outcome.diagnostics["dirty_fraction"] < 1.0

    exact = _local_worker(
        science_site,
        _state_layer(science_site, "add"),
        science_site.results_root("r2_worker_exact"),
        config=InfluenceConfig(
            full_recompute_fraction=EXACT_LANE_FULL_RECOMPUTE_FRACTION
        ),
    )
    exact.bump_scene_revision()
    outcome = exact.run()
    assert outcome.published and outcome.mode == "local", (
        "the exact lane must keep an ordinary edit incremental"
    )
    assert outcome.diagnostics["demoted_by_fraction"] is False, outcome.diagnostics
    assert 0.0 < outcome.diagnostics["dirty_fraction"] < 1.0


# ---------------------------------------------------------------------------
# R4: idle once-daily reconciliation full
# ---------------------------------------------------------------------------


def _grace_zero_runner(tmp_path: Path, store: Store, hub: CountingHub) -> JobRunner:
    return make_runner(tmp_path, store, hub, reconcile_idle_grace_s=0.0)


def test_r4_idle_reconcile_mints_one_marked_full(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    make_workspace(store, WS)
    hub = CountingHub(count=0)
    runner = _grace_zero_runner(tmp_path, store, hub)

    store.append_operations(WS, [op_item("op_r4_1")])
    close_epoch(store, epoch_id=0)
    runner._exact_lane_tick()  # gated: debt recorded
    assert store.reconcile_state(WS)[1] is True

    # First idle scan starts the emptiness clock; the second is eligible
    # (grace 0). Clock values chosen past the 15 s scan throttle.
    runner._idle_reconcile_tick(now_monotonic=100.0)
    runner._idle_reconcile_tick(now_monotonic=200.0)

    pending = [row for row in exact_jobs(store, WS) if row["status"] == "queued"]
    assert len(pending) == 1, job_rows(store, WS)
    marker = pending[0]["request"]["exact_lane"]
    assert marker.get("reconcile") is True, marker


def test_r4_completion_consumes_cap_and_supersession_does_not(
    tmp_path: Path,
) -> None:
    store = make_store(tmp_path)
    make_workspace(store, WS)
    hub = CountingHub(count=0)
    runner = _grace_zero_runner(tmp_path, store, hub)

    store.publish_result(
        WS, 0, manifest={"checksum": "c0", "scene_version": 0}, payload=b"p0", exact=True
    )
    store.append_operations(WS, [op_item("op_r4_1")])
    close_epoch(store, epoch_id=0)
    runner._exact_lane_tick()  # gated: debt recorded
    assert store.reconcile_state(WS) == ("1970-01-01", True)

    # A superseded reconcile attempt: settles nothing, consumes nothing.
    with store._write() as conn:
        superseded = Store._insert_job_locked(
            conn,
            WS,
            target_scene_version=1,
            request={"exact_lane": {"target_revision": 1, "reconcile": True}},
            edit_watermark=0,
        )
    store.mark_job_running(superseded)
    assert store.finish_job(superseded, "superseded")
    assert store.reconcile_state(WS) == ("1970-01-01", True), (
        "a superseded attempt must leave the debt standing"
    )

    # The idle tick re-mints after its throttle — same day, still owed.
    runner._idle_reconcile_tick(now_monotonic=100.0)
    runner._idle_reconcile_tick(now_monotonic=10_000.0)
    pending = [row for row in exact_jobs(store, WS) if row["status"] == "queued"]
    assert len(pending) == 1, "a superseded attempt must not consume the daily cap"
    assert pending[0]["request"]["exact_lane"].get("reconcile") is True

    # COMPLETION owns the day: reconcile state records today, debt gone.
    job_id = pending[0]["job_id"]
    store.mark_job_running(job_id)
    runner._finalize(
        store.require_job(job_id),
        store.require_scenario(WS),
        SolveResult(status="no-op", scene_version=1),
        duration_ms=1.0,
    )
    assert store.require_job(job_id).status == "complete"
    assert store.reconcile_state(WS) == (today_utc(), False)

    # Re-arm the workspace: a FRESH revision gap (new operation + epoch,
    # still no subscriber -> the gate defers again) puts it back in the
    # idle scan's candidate set while today's completion stands. Without
    # this the cap assertions below are vacuous — a discharged workspace
    # (no recoverable epochs, no debt) is never scanned at all, so even a
    # cap-less runner mints nothing.
    store.append_operations(WS, [op_item("op_r4_2")])
    close_epoch(store, epoch_id=1)
    runner._exact_lane_tick()  # gated: same-day debt re-accrues (date kept)
    assert store.reconcile_state(WS) == (today_utc(), True), (
        "re-arm failed: the workspace must re-enter the scan owing debt"
    )

    # The harness runner's worker thread never started, so the earlier
    # mint's queue entry never drains — pull it (a quiet runner by
    # definition) so the final ticks actually reach the scan instead of
    # bouncing off the idle guard's non-empty-queue check.
    while True:
        try:
            runner._queue.get_nowait()
        except queue.Empty:
            break

    # Same-day cap: no second reconcile mint even deep past every backoff,
    # with the debt STANDING (candidate set non-empty). Mutation-checked:
    # disabling the `last_completed == today` early-return in
    # _idle_reconcile_workspace mints a reconcile job at revision 2 and
    # FAILS the assertions below (review finding: the prior pin was
    # vacuous — probed separately, the product cap held).
    runner._idle_reconcile_tick(now_monotonic=100_000.0)
    runner._idle_reconcile_tick(now_monotonic=100_200.0)
    queued = [
        row
        for row in exact_jobs(store, WS)
        if row["status"] == "queued" and row["request"]["exact_lane"].get("reconcile")
    ]
    assert queued == [], (
        "the once-daily cap must hold even with re-armed same-day debt"
    )
    assert store.reconcile_state(WS) == (today_utc(), True), (
        "the held cap must leave the debt standing for tomorrow's scan"
    )


def test_r4_watched_workspace_never_idle_reconciles(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    make_workspace(store, WS)
    hub = CountingHub(count=1)
    runner = _grace_zero_runner(tmp_path, store, hub)

    store.append_operations(WS, [op_item("op_r4_1")])
    close_epoch(store, epoch_id=0)
    runner._exact_lane_tick()  # subscribed: normal chase, no debt

    runner._idle_reconcile_tick(now_monotonic=100.0)
    runner._idle_reconcile_tick(now_monotonic=10_000.0)
    reconcile_jobs = [
        row
        for row in exact_jobs(store, WS)
        if row["request"]["exact_lane"].get("reconcile")
    ]
    assert reconcile_jobs == [], "a watched workspace must never idle-reconcile"


def test_r4_grace_not_elapsed_defers(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    make_workspace(store, WS)
    hub = CountingHub(count=0)
    runner = make_runner(tmp_path, store, hub, reconcile_idle_grace_s=300.0)

    store.append_operations(WS, [op_item("op_r4_1")])
    close_epoch(store, epoch_id=0)
    runner._exact_lane_tick()

    runner._idle_reconcile_tick(now_monotonic=100.0)
    runner._idle_reconcile_tick(now_monotonic=200.0)  # 100 s << 300 s grace
    assert exact_jobs(store, WS) == [], "emptiness shorter than the grace defers"


def test_r4_deep_patch_chain_arms_without_grace(tmp_path: Path) -> None:
    """Secondary trigger: >50 revisions since the last completed full,
    still while quiet — eligible without waiting out the grace."""
    store = make_store(tmp_path)
    make_workspace(store, WS)
    hub = CountingHub(count=0)
    runner = make_runner(tmp_path, store, hub, reconcile_idle_grace_s=1e9)

    store.append_operations(WS, [op_item("op_r4_1")])
    close_epoch(store, epoch_id=0)
    # A completed FULL result at version 1 (the chain anchor), then a deep
    # canonical chain manufactured as the durable state the heuristic reads.
    with store._write() as conn:
        anchor = Store._insert_job_locked(
            conn, WS, target_scene_version=1, request={}, edit_watermark=0
        )
    store.mark_job_running(anchor)
    store.finish_job(anchor, "complete", mode="full", result_scene_version=1)
    assert store.last_full_result_version(WS) == 1
    with store._write() as conn:
        conn.execute(
            "UPDATE scenarios SET scene_version = 60 WHERE scenario_id = ?", (WS,)
        )

    runner._idle_reconcile_tick(now_monotonic=100.0)
    runner._idle_reconcile_tick(now_monotonic=200.0)
    pending = [row for row in exact_jobs(store, WS) if row["status"] == "queued"]
    assert len(pending) == 1, job_rows(store, WS)
    assert pending[0]["request"]["exact_lane"].get("reconcile") is True


def test_r4_debt_survives_restart(tmp_path: Path) -> None:
    """The owed flag is durable: a fresh Store on the same database still
    reports the debt (the idle scan re-discovers the workspace)."""
    store = make_store(tmp_path)
    make_workspace(store, WS)
    hub = CountingHub(count=0)
    runner = make_runner(tmp_path, store, hub)

    store.append_operations(WS, [op_item("op_r4_1")])
    close_epoch(store, epoch_id=0)
    runner._exact_lane_tick()
    assert store.reconcile_state(WS)[1] is True

    reopened = Store(store.db_path, results_root=store.results_root)
    try:
        assert reopened.reconcile_state(WS)[1] is True
        assert reopened.workspaces_owing_reconcile() == [WS]
    finally:
        reopened.close()


# ---------------------------------------------------------------------------
# R5: honest ETA on the job body
# ---------------------------------------------------------------------------


def _finish_with_metrics(
    store: Store, scenario_id: str, *, mode: str, duration_ms: float,
    window_fraction: float | None, version: int,
) -> None:
    with store._write() as conn:
        job_id = Store._insert_job_locked(
            conn, scenario_id, target_scene_version=version, request={},
            edit_watermark=0,
        )
    store.mark_job_running(job_id)
    store.finish_job(
        job_id,
        "complete",
        mode=mode,
        metrics={
            "mode": mode,
            "duration_ms": duration_ms,
            **({"window_fraction": window_fraction} if window_fraction is not None else {}),
        },
        result_scene_version=version,
    )


def test_r5_eta_full_static_then_history(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    make_workspace(store, WS)

    # No history: disclosed static fallback for fulls.
    assert store.estimate_solve_seconds(WS, "full") == (480, "static")

    # Three fulls (100/200/300 s): p80 = 300 s.
    for version, seconds in enumerate((100.0, 200.0, 300.0), start=1):
        _finish_with_metrics(
            store, WS, mode="full", duration_ms=seconds * 1000.0,
            window_fraction=None, version=version,
        )
    assert store.estimate_solve_seconds(WS, "full") == (300, "history")


def test_r5_eta_local_bucket_rescale_and_null(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    make_workspace(store, WS)

    # No local history: honestly null, never a guess.
    assert store.estimate_solve_seconds(WS, "local", window_fraction=0.2) == (None, None)

    # Three locals at wf=0.3 (60/60/120 s) rescaled toward smaller windows.
    for version, seconds in enumerate((60.0, 60.0, 120.0), start=1):
        _finish_with_metrics(
            store, WS, mode="local", duration_ms=seconds * 1000.0,
            window_fraction=0.3, version=version,
        )
    # wf=0.15 is inside the ±0.2 bucket: rescaled p80 = 60 s.
    assert store.estimate_solve_seconds(WS, "local", window_fraction=0.15) == (
        60,
        "history",
    )
    # No nearby bucket (target far from every sample): null.
    assert store.estimate_solve_seconds(WS, "local", window_fraction=0.9) == (
        None,
        None,
    )


def test_r5_job_body_carries_eta_fields(tmp_path: Path) -> None:
    """The route seam: queued + running bodies carry ``eta_seconds`` /
    ``eta_basis``; a reconcile-queued job estimates as a full; terminal
    bodies never carry them."""
    app = make_app(tmp_path, FakeSolver())
    with TestClient(app) as client:
        created = client.post(
            "/api/v1/scenarios",
            json={"site_id": SITE_ID, "name": "eta", "initial_state": "baseline"},
        )
        assert created.status_code == 201, created.text
        workspace = created.json()["scenario_id"]
        store = app.state.context.store

        # A queued reconcile job with no history: static full estimate.
        with store._write() as conn:
            reconcile_job = Store._insert_job_locked(
                conn,
                workspace,
                target_scene_version=1,
                request={"exact_lane": {"target_revision": 1, "reconcile": True}},
                edit_watermark=0,
            )
        body = client.get(f"/api/v1/jobs/{reconcile_job}").json()
        assert body["eta_seconds"] == 480, body
        assert body["eta_basis"] == "static", body

        # A plain queued job: unknown routing, honest null.
        with store._write() as conn:
            plain_job = Store._insert_job_locked(
                conn, workspace, target_scene_version=2, request={}, edit_watermark=0
            )
        body = client.get(f"/api/v1/jobs/{plain_job}").json()
        assert body["eta_seconds"] is None and body["eta_basis"] is None, body

        # A running local job with history: bucket-rescaled estimate. The
        # window spans 5 of 32 rows (~0.156 of the tile), inside the ±0.2
        # bucket around the 0.3 samples -> rescaled p80 = 60 s.
        for version, seconds in enumerate((60.0, 60.0, 120.0), start=10):
            _finish_with_metrics(
                store, workspace, mode="local", duration_ms=seconds * 1000.0,
                window_fraction=0.3, version=version,
            )
        with store._write() as conn:
            running_job = Store._insert_job_locked(
                conn, workspace, target_scene_version=3, request={}, edit_watermark=0
            )
        store.mark_job_running(running_job)
        store.update_job_progress(
            running_job,
            stage="time_loop",
            progress={"completed_time_steps": 1, "total_time_steps": 4},
            mode="local",
            window={"row_start": 0, "row_stop": 5, "col_start": 0, "col_stop": 48},
        )
        job = store.require_job(running_job)
        assert job_eta_fields(store, app.state.context.sites, job) == (60, "history")

        # Terminal bodies never carry ETA fields.
        store.finish_job(running_job, "cancelled", error={"code": "cancelled"})
        body = client.get(f"/api/v1/jobs/{running_job}").json()
        assert "eta_seconds" not in body and "eta_basis" not in body, body


# ---------------------------------------------------------------------------
# R6: routing observability in published metrics
# ---------------------------------------------------------------------------


@pytest.mark.scientific
def test_r6_bridge_lifts_routing_metrics(tmp_path: Path, family_site) -> None:
    """End to end: an exact-lane solve's PUBLISHED metrics carry the
    worker's TRUE pre-mode dirty fraction and the demotion verdict —
    persisted on the job row by materialize_result, so a full-mode job can
    always say WHY it is full. A single tree on the small family tile
    legitimately dirties the whole tile (its shadow reach spans it): an
    HONEST full — dirty_fraction 1.0 with demoted_by_fraction False —
    which is exactly the case the silent 0.30 demotion used to hide."""
    from solweig_gpu.server.realtime.epochs import EpochScheduler
    from solweig_gpu.server.realtime.scheduler import FastLaneScheduler
    from solweig_gpu.server.realtime.telemetry import TelemetryRegistry

    from tests.test_server_universal_edits import (
        SITE_ID as FAMILY_SITE,
        _make_client,
        _scenario_ready,
    )

    # Isolated telemetry + no fast-lane thread (the r2 E2E precedent):
    # create_app's default factories register into the module-shared
    # singleton, leaking into test_realtime_telemetry later in the session.
    def isolated_scheduler(store, hub):
        return EpochScheduler(store, hub=hub, telemetry=TelemetryRegistry())

    def isolated_fast_lane(store, hub):
        return FastLaneScheduler(store, hub=hub, telemetry=TelemetryRegistry())

    with _make_client(
        tmp_path,
        family_site,
        epoch_scheduler_factory=isolated_scheduler,
        fast_lane_factory=isolated_fast_lane,
        start_fast_lane=False,
    ) as client:
        workspace = _scenario_ready(client)
        watch = SseSession(client, workspace)
        store = client.app.state.context.store
        _post_native_ops(
            client,
            workspace,
            [
                _native_veg_add_item(
                    client,
                    FAMILY_SITE,
                    "r6_veg_1",
                    "tree-r6-1",
                    u=0.5,
                    v=0.5,
                )
            ],
        )
        body = _quiesce_workspace(client, workspace, timeout=120.0)
        assert body["workspace_revision"] >= 1

        jobs = exact_jobs(store, workspace)
        final_job = jobs[-1]
        assert final_job["status"] == "complete", jobs
        metrics = final_job["metrics"] or {}
        assert metrics.get("mode") == "full", metrics
        assert metrics.get("dirty_fraction") == 1.0, metrics
        assert metrics.get("demoted_by_fraction") is False, metrics


# ---------------------------------------------------------------------------
# Review follow-ups on the routing policy wave: the cancel route's ETA
# contract, the reconcile retry ceiling, and typed-refusal debt hygiene
# ---------------------------------------------------------------------------


def test_review_cancel_route_consults_eta_on_every_job_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cancel route's two bodies (the fresh cancel and the idempotent
    replay) render through the same eta-wired path as GET /jobs/{id}:
    every ``_job_body`` call site consults :func:`job_eta_fields`, so the
    response shape cannot drift between the routes serving one job row."""
    app = make_app(tmp_path, FakeSolver())
    consulted: list[str] = []
    real = routes_jobs.job_eta_fields

    def spy(store, sites, job):
        consulted.append(job.job_id)
        return real(store, sites, job)

    monkeypatch.setattr(routes_jobs, "job_eta_fields", spy)
    with TestClient(app) as client:
        created = client.post(
            "/api/v1/scenarios",
            json={
                "site_id": SITE_ID,
                "name": "cancel-eta",
                "initial_state": "baseline",
            },
        )
        assert created.status_code == 201, created.text
        workspace = created.json()["scenario_id"]
        store = app.state.context.store
        with store._write() as conn:
            queued = Store._insert_job_locked(
                conn,
                workspace,
                target_scene_version=1,
                request={"exact_lane": {"target_revision": 1, "reconcile": True}},
                edit_watermark=0,
            )

        fresh = client.post(f"/api/v1/jobs/{queued}/cancel")
        assert fresh.status_code == 200, fresh.text
        body = fresh.json()
        assert body["status"] == "cancelled", body

        replay = client.post(f"/api/v1/jobs/{queued}/cancel")
        assert replay.status_code == 200, replay.text
        assert replay.json() == body, "the idempotent replay must be byte-stable"

        # The cancel responses render the same row GET serves, identically:
        # a cancelled job is terminal, so no eta keys anywhere (the contract
        # every route must agree on).
        assert body == client.get(f"/api/v1/jobs/{queued}").json()

    assert consulted == [queued, queued, queued], (
        "both cancel-path bodies must render through job_eta_fields "
        f"(before the GET's own consultation): saw {consulted}"
    )


def test_review_reconcile_retry_ceiling_bounded_per_day(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A crash-class poison reconcile must not loop a full solve roughly
    continuously: absent a completion, at most five reconcile mints run
    per workspace per UTC day, the hold is logged exactly once (not per
    scan), and the durable debt stands for the next day's scan."""
    store = make_store(tmp_path)
    make_workspace(store, WS)
    hub = CountingHub(count=0)
    runner = _grace_zero_runner(tmp_path, store, hub)

    store.append_operations(WS, [op_item("op_ceiling_1")])
    close_epoch(store, epoch_id=0)
    runner._exact_lane_tick()  # gated: debt recorded
    assert store.reconcile_state(WS)[1] is True

    with caplog.at_level(logging.INFO, logger="solweig_gpu.server.jobs"):
        runner._idle_reconcile_tick(now_monotonic=100.0)  # emptiness clock
        now = 100.0
        for _ in range(8):  # crash-loop opportunities, deep past every backoff
            now += 1_000.0
            runner._idle_reconcile_tick(now_monotonic=now)
            _drain_queue(runner)
            # The crash-class failure shape: every minted attempt dies
            # superseded (never finalized, epochs stay recoverable), so the
            # next scan re-mints — exactly the loop the ceiling bounds.
            for row in exact_jobs(store, WS):
                if row["status"] in ("queued", "running"):
                    store.mark_job_running(row["job_id"])
                    assert store.finish_job(row["job_id"], "superseded")

    reconcile_rows = [
        row
        for row in exact_jobs(store, WS)
        if row["request"]["exact_lane"].get("reconcile")
    ]
    assert len(reconcile_rows) == 5, (
        "the daily attempt ceiling must bound the crash-class retry loop: "
        f"{len(reconcile_rows)} mints ran"
    )
    assert store.reconcile_state(WS)[1] is True, (
        "the held ceiling must leave the debt standing for tomorrow's scan"
    )
    holds = [r for r in caplog.records if "attempt ceiling" in r.message]
    assert len(holds) == 1, [r.message for r in caplog.records]


def test_review_reconcile_ceiling_ledger_resets_on_completion(
    tmp_path: Path,
) -> None:
    """A COMPLETED reconcile consumes the day AND clears the in-process
    attempt ledger: the completion proved the workspace is not poisoned,
    so the next day's attempts start from zero."""
    store = make_store(tmp_path)
    make_workspace(store, WS)
    hub = CountingHub(count=0)
    runner = _grace_zero_runner(tmp_path, store, hub)

    store.publish_result(
        WS, 0, manifest={"checksum": "c0", "scene_version": 0}, payload=b"p0", exact=True
    )
    store.append_operations(WS, [op_item("op_ceiling_reset_1")])
    close_epoch(store, epoch_id=0)
    runner._exact_lane_tick()  # gated: debt recorded

    runner._idle_reconcile_tick(now_monotonic=100.0)  # emptiness clock
    # Attempt 1 dies superseded (ledger 1); attempt 2 completes.
    runner._idle_reconcile_tick(now_monotonic=1_000.0)
    _drain_queue(runner)
    first = [row for row in exact_jobs(store, WS) if row["status"] == "queued"][0]
    store.mark_job_running(first["job_id"])
    assert store.finish_job(first["job_id"], "superseded")

    runner._idle_reconcile_tick(now_monotonic=2_000.0)
    _drain_queue(runner)
    second = [row for row in exact_jobs(store, WS) if row["status"] == "queued"][0]
    assert second["request"]["exact_lane"].get("reconcile") is True
    store.mark_job_running(second["job_id"])

    runner._finalize(
        store.require_job(second["job_id"]),
        store.require_scenario(WS),
        SolveResult(status="no-op", scene_version=1),
        duration_ms=1.0,
    )
    assert store.require_job(second["job_id"]).status == "complete"
    assert store.reconcile_state(WS) == (today_utc(), False)
    assert WS not in runner._reconcile_failures, (
        "a completion must reset the attempt ledger for the next day"
    )


def test_review_typed_refusal_clears_stale_reconcile_debt(tmp_path: Path) -> None:
    """An exact-lane job failing with a typed refusal (``edit_rejected``)
    consumes its epochs but never reaches ``mark_reconcile_settlement`` —
    without the explicit clear, owed=1 lingers forever with no recoverable
    gap behind it (permanent candidate-set noise). The refusal settles the
    span, so the debt's purpose (folding the revision gap) is moot."""
    store = make_store(tmp_path)
    make_workspace(store, WS)
    hub = CountingHub(count=0)
    runner = make_runner(tmp_path, store, hub)

    store.append_operations(WS, [op_item("op_refuse_1")])
    close_epoch(store, epoch_id=0)
    runner._exact_lane_tick()  # gated: durable debt
    assert store.reconcile_state(WS) == ("1970-01-01", True)

    hub.count = 1  # the subscriber arrives: the gated gap is chased
    runner._exact_lane_tick()
    pending = [row for row in exact_jobs(store, WS) if row["status"] == "queued"]
    assert len(pending) == 1, job_rows(store, WS)

    store.mark_job_running(pending[0]["job_id"])
    runner._finalize(
        store.require_job(pending[0]["job_id"]),
        store.require_scenario(WS),
        SolveResult(
            status="failed",
            scene_version=1,
            error={"code": "edit_rejected", "message": "unfoldable verb"},
        ),
        duration_ms=1.0,
    )
    assert store.require_job(pending[0]["job_id"]).status == "failed"
    # The refusal consumed the span's epochs: no recoverable gap remains...
    assert {e.workspace_id for e in store.recoverable_epochs()} == set()
    # ...so the owed flag must not linger as idle-scan noise.
    assert store.reconcile_state(WS) == ("1970-01-01", False)
    assert store.workspaces_owing_reconcile() == []


def test_review_crash_class_failure_keeps_reconcile_debt(tmp_path: Path) -> None:
    """The crash-class counterpart (the review red line): an UNTYPED
    solver failure settles no epochs and keeps the debt — the retry is
    wanted, so only the typed refusal may clear it."""
    store = make_store(tmp_path)
    # The workspace pins a site the registry can actually serve: _execute
    # resolves the site geometry before the solver call.
    store.create_scenario(site_id=SITE_ID, name="rt", scenario_id=WS)
    hub = CountingHub(count=0)
    context = RunnerContext(
        store=store,
        sites=SiteRegistry({SITE_ID: {"cache_dir": make_site_cache(tmp_path)}}),
        state_root=tmp_path / "state",
    )

    def poison(request, progress):
        raise RuntimeError("poison request")

    runner = JobRunner(
        context,
        solver_factory=lambda ctx: poison,
        coalescing_window_ms=0.0,
        broadcast_hub=hub,
        reconcile_idle_grace_s=0.0,
    )
    runner._solver = poison  # what start() would assign

    store.append_operations(WS, [op_item("op_crash_1")])
    close_epoch(store, epoch_id=0)
    runner._exact_lane_tick()  # gated: debt recorded
    hub.count = 1
    runner._exact_lane_tick()
    pending = [row for row in exact_jobs(store, WS) if row["status"] == "queued"]
    assert len(pending) == 1, job_rows(store, WS)

    runner._execute(store.require_job(pending[0]["job_id"]), store.require_scenario(WS))
    job = store.require_job(pending[0]["job_id"])
    assert job.status == "failed" and job.error["code"] == "job_failed", job.error
    assert {e.workspace_id for e in store.recoverable_epochs()} == {WS}, (
        "a crash-class failure must leave the epochs recoverable for the retry"
    )
    assert store.reconcile_state(WS) == ("1970-01-01", True)
