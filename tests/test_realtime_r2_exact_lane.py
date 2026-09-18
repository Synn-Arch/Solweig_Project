# SPDX-License-Identifier: GPL-3.0-only
"""R2a exact lane v2: revision-chasing queue, epoch consumption, legacy
funnel (failing-first).

Pins the mission contracts (``docs/incremental_design_tool/
realtime_collaboration/epoch_scheduler.md`` "Exact queue" +
``compute_architecture.md`` "Two-lane scheduling" + the r2a packet scope):

* **Compaction** — per workspace at most ONE running + ONE latest-pending
  exact job. Jobs target the canonical ``workspace_revision``
  (``scenarios.scene_version``), never an edit batch; a pending target
  older than the newest demand is dropped (superseded) and re-created at
  the current revision.
* **Consumption** — the exact lane derives its batch from the epoch plane:
  every non-terminal epoch with an assigned revision below the published
  exact result's version advances to ``exact_targeted`` (and leaves the
  recoverable set). A LEGACY job completing must never consume epochs.
* **Zero loss** — the exact-lane solve folds exactly the realtime-native
  operations of the span ``(consumed_seq, span_hi]``, each exactly once,
  onto the canonical baseline at the consumed revision; anything else is
  a tripwire (``RuntimeError``), never a silent partial fold.
* **Legacy funnel** — on a workspace that already carries realtime
  operations, BOTH legacy edit paths (``/edits`` tree edits and
  ``/edits/universal`` family edits) additionally land as realtime
  operations (actor ``legacy_edits``) so the epoch plane sees the whole
  collaborative history. Pure-legacy workspaces are untouched: no
  operations, no epochs, no extra version bumps. Funnelled operations
  carry ``base_revision`` ADVISORY (the exact_session If-Match equality
  gate stays in the route, exactly as before) and are skipped by the
  exact lane's command derivation (their scene effects come from the
  authoritative ``edit_events`` replay in the same job — no double
  apply).

All written failing-first against base e1e7d8e (r1 waves merged, no r2a
behavior).
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pytest
from fastapi.testclient import TestClient

from solweig_gpu.server import executor_bridge
from solweig_gpu.server import jobs as jobs_module
from solweig_gpu.server.executor_bridge import make_universal_dispatch_solver
from solweig_gpu.server.realtime.epochs import EpochScheduler
from solweig_gpu.server.realtime.scheduler import FastLaneScheduler
from solweig_gpu.server.realtime.telemetry import TelemetryRegistry
from solweig_gpu.server.jobs import (
    JobRunner,
    RunnerContext,
    SiteRegistry,
    SolveRequest,
    SolveResult,
)
from solweig_gpu.server.patch_codec import decode_payload
from solweig_gpu.server.store import Store

from tests.test_realtime_epochs import (
    WINDOW_S,
    WS,
    make_rt_app,
    make_store,
    make_workspace,
    op_item,
    rt_post,
)
from tests.test_server_api import (
    SITE_ID,
    FakeSolver,
    make_site_cache,
    make_tree,
)
from tests.test_server_universal_edits import family_site

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

#: actor_id marking funnelled legacy edits (r2a). Reference through the
#: module (not a from-import) so the RED run at base fails the TESTS, not
#: collection.
LEGACY_ACTOR = "legacy_edits"


def make_runner(
    tmp_path: Path, store: Store, broadcast_hub: Any = None
) -> JobRunner:
    """A JobRunner wired like the app's, with its worker NOT started.

    Unit tests drive the exact-lane tick / finalize paths directly on the
    calling thread, so every assertion below is deterministic (no worker
    races). The solver never runs for these paths.
    """
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
    )


def job_rows(store: Store, scenario_id: str) -> list[dict[str, Any]]:
    """Every job row of one scenario (ascending), with parsed payloads."""
    conn = sqlite3.connect(store.db_path)
    try:
        rows = conn.execute(
            "SELECT job_id, status, target_scene_version, request_json, "
            "metrics_json, error_json FROM jobs WHERE scenario_id = ? "
            "ORDER BY rowid",
            (scenario_id,),
        ).fetchall()
    finally:
        conn.close()
    return [
        {
            "job_id": row[0],
            "status": row[1],
            "target": int(row[2]),
            "request": json.loads(row[3]) if row[3] else {},
            "metrics": json.loads(row[4]) if row[4] else None,
            "error": json.loads(row[5]) if row[5] else None,
        }
        for row in rows
    ]


def exact_jobs(store: Store, scenario_id: str) -> list[dict[str, Any]]:
    return [row for row in job_rows(store, scenario_id) if row["request"].get("exact_lane")]


def close_epoch(store: Store, *, epoch_id: int = 0) -> int:
    """Assign one epoch's reduction directly (the r1 close commit)."""
    store.mark_epoch_status(WS, epoch_id, "closed")
    committed = store.commit_epoch_reduction(
        WS, epoch_id, families={"vegetation_geometry": {"objects": {}}}
    )
    assert committed is not None and committed.first_time
    return committed.workspace_revision


def poll_until(predicate, *, timeout: float = 60.0, interval: float = 0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    return None


@contextmanager
def sse_subscribed(client: TestClient, workspace: str):
    """Hold one watcher for ``workspace`` (routing policy R1 audience).

    The exact lane's chase minting is subscriber-gated: a workspace
    nobody is watching defers its recoverable revisions to the idle
    reconcile instead of piling solves ahead of live users. The gate
    reads the broadcast hub's subscriber count, so a test that expects
    exact-lane catch-up registers a subscription directly at the hub
    seam — neither TestClient nor httpx's ASGITransport can hold an
    infinite SSE stream open (both run the app to completion and
    deadlock; see the SSEConnection precedent in
    test_realtime_epochs), and the real studio client's audience is
    exactly one registered subscription.
    """
    subscription = client.app.state.broadcast_hub.subscribe(workspace)
    try:
        yield subscription
    finally:
        subscription.close()


class SseSession:
    """Same audience modeling as :func:`sse_subscribed`, held open
    explicitly until ``close()`` — for test bodies where a wrapping
    ``with`` block would swallow the rest of the function."""

    def __init__(self, client: TestClient, workspace: str) -> None:
        self._subscription = client.app.state.broadcast_hub.subscribe(workspace)

    def close(self) -> None:
        self._subscription.close()


# ---------------------------------------------------------------------------
# Compaction: one running + one latest-pending, targeting workspace_revision
# ---------------------------------------------------------------------------


def test_tick_without_epochs_creates_nothing(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    make_workspace(store, WS)
    runner = make_runner(tmp_path, store)
    runner._exact_lane_tick()
    # ``create_scenario`` queues a baseline job (the legacy lane's); the
    # tick must add no exact job and never touch it.
    assert exact_jobs(store, WS) == [], "fresh scenario must not gain exact jobs"
    baseline = job_rows(store, WS)
    assert [row["status"] for row in baseline] == ["queued"]


def test_tick_creates_single_pending_job_targeting_workspace_revision(
    tmp_path: Path,
) -> None:
    store = make_store(tmp_path)
    make_workspace(store, WS)
    store.append_operations(WS, [op_item("op_1"), op_item("op_2")])
    revision = close_epoch(store)
    assert revision == 1

    make_runner(tmp_path, store)._exact_lane_tick()

    pending = [row for row in exact_jobs(store, WS) if row["status"] == "queued"]
    assert len(pending) == 1, job_rows(store, WS)
    marker = pending[0]["request"]["exact_lane"]
    assert marker["target_revision"] == 1 == store.require_scenario(WS).scene_version
    scenario = store.require_scenario(WS)
    assert pending[0]["request"] is not None
    # The exact job inherits the whole unconsumed edit window (superseded
    # legacy jobs included) exactly like a legacy superseding job would.
    conn = sqlite3.connect(store.db_path)
    try:
        row = conn.execute(
            "SELECT edit_watermark, base_edit_watermark FROM jobs WHERE job_id = ?",
            (pending[0]["job_id"],),
        ).fetchone()
    finally:
        conn.close()
    assert int(row[0]) == scenario.edit_sequence
    assert int(row[1]) == scenario.acked_sequence


def test_newer_revision_replaces_pending_target(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    make_workspace(store, WS)
    runner = make_runner(tmp_path, store)
    store.append_operations(WS, [op_item("op_1")])
    close_epoch(store, epoch_id=0)
    runner._exact_lane_tick()
    first = exact_jobs(store, WS)
    assert len(first) == 1 and first[0]["status"] == "queued"

    # A second epoch closes: the canonical revision moved to 2.
    store.append_operations(WS, [op_item("op_2")])
    store.mark_epoch_status(WS, 1, "closed")
    store.commit_epoch_reduction(
        WS, 1, families={"vegetation_geometry": {"objects": {}}}
    )
    runner._exact_lane_tick()

    jobs = exact_jobs(store, WS)
    assert [row["status"] for row in jobs] == ["superseded", "queued"], jobs
    assert jobs[1]["request"]["exact_lane"]["target_revision"] == 2
    queued = [row for row in job_rows(store, WS) if row["status"] == "queued"]
    assert len(queued) == 1, "one latest-pending exact job per workspace"


def test_running_exact_job_not_duplicated_newer_revision_may_queue(
    tmp_path: Path,
) -> None:
    store = make_store(tmp_path)
    make_workspace(store, WS)
    runner = make_runner(tmp_path, store)
    store.append_operations(WS, [op_item("op_1")])
    close_epoch(store, epoch_id=0)
    runner._exact_lane_tick()
    running = exact_jobs(store, WS)[0]
    store.mark_job_running(running["job_id"])

    # Same revision: single-flight — no second exact job while one runs.
    runner._exact_lane_tick()
    assert len(exact_jobs(store, WS)) == 1

    # Newer revision: ONE pending job at the new target; the running job
    # keeps running (its stale completion is discarded by the finalize
    # guard, never published).
    store.append_operations(WS, [op_item("op_2")])
    store.mark_epoch_status(WS, 1, "closed")
    store.commit_epoch_reduction(
        WS, 1, families={"vegetation_geometry": {"objects": {}}}
    )
    runner._exact_lane_tick()
    statuses = sorted(
        (row["status"], row["request"]["exact_lane"]["target_revision"])
        for row in exact_jobs(store, WS)
    )
    assert statuses == [("queued", 2), ("running", 1)], statuses


def test_failed_exact_attempt_does_not_churn(tmp_path: Path) -> None:
    """A typed refusal must not restart on every tick (bounded attempts).

    One attempt per canonical revision; a NEWER revision re-arms it.
    """
    store = make_store(tmp_path)
    make_workspace(store, WS)
    runner = make_runner(tmp_path, store)
    store.append_operations(WS, [op_item("op_1")])
    close_epoch(store, epoch_id=0)
    runner._exact_lane_tick()
    job = exact_jobs(store, WS)[0]
    store.mark_job_running(job["job_id"])
    assert store.finish_job(
        job["job_id"], "failed", error={"code": "engine_refused", "message": "x"}
    )

    runner._exact_lane_tick()
    runner._exact_lane_tick()
    assert len(exact_jobs(store, WS)) == 1, "failed attempt must not respawn"

    # Newer demand re-arms the lane.
    store.append_operations(WS, [op_item("op_2")])
    store.mark_epoch_status(WS, 1, "closed")
    store.commit_epoch_reduction(
        WS, 1, families={"vegetation_geometry": {"objects": {}}}
    )
    runner._exact_lane_tick()
    assert [row["status"] for row in exact_jobs(store, WS)] == ["failed", "queued"]


# ---------------------------------------------------------------------------
# Revision-chasing supersession (the scan extension, jobs.py:749)
# ---------------------------------------------------------------------------


def test_supersede_scan_aborts_stale_exact_job_before_solve(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    make_workspace(store, WS)
    runner = make_runner(tmp_path, store)
    with store._write() as conn:
        job_id = Store._insert_job_locked(
            conn,
            WS,
            target_scene_version=1,
            request={"exact_lane": {"target_revision": 1}},
            edit_watermark=0,
        )
    # The canonical revision advances past the job's target (later epochs
    # committed): the queued job must abort BEFORE its solve is paid for.
    store.append_operations(WS, [op_item("op_2")])
    store.mark_epoch_status(WS, 0, "closed")
    store.commit_epoch_reduction(
        WS, 0, families={"vegetation_geometry": {"objects": {}}}
    )
    store.append_operations(WS, [op_item("op_3")])
    store.mark_epoch_status(WS, 1, "closed")
    store.commit_epoch_reduction(
        WS, 1, families={"vegetation_geometry": {"objects": {}}}
    )
    assert store.require_scenario(WS).scene_version == 2

    assert runner._supersede_scan(job_id) is True
    job = store.get_job(job_id)
    assert job is not None and job.status == "superseded"


# ---------------------------------------------------------------------------
# Consumption: exact_targeted at exact-lane completion only
# ---------------------------------------------------------------------------


def test_exact_settle_broadcasts_named_exact_revision_event(tmp_path: Path) -> None:
    """verify-contract must-fix: the studio's verify bridge (badge → Exact,
    exact-result fetch) rides the NAMED ``exact_revision`` SSE event —
    which nothing emitted server-side (``broadcast_exact`` had zero call
    sites; only the ``exact_revision`` FIELD on fast frames reconciled
    the badge). The exact lane's settle must emit it — but only when a
    result actually EXISTS at the settled revision: failed/superseded
    settles advance no exact state and must never claim exactness.
    """
    from tests.test_realtime_epochs import epoch_of

    class RecordingHub:
        """BroadcastHub stand-in recording exact events only."""

        def __init__(self) -> None:
            self.exact: list[tuple[str, dict[str, Any]]] = []

        def broadcast_exact(self, workspace_id: str, event: dict[str, Any]) -> None:
            self.exact.append((workspace_id, dict(event)))

    store = make_store(tmp_path)
    make_workspace(store, WS)
    hub = RecordingHub()
    runner = make_runner(tmp_path, store, broadcast_hub=hub)

    # A published result at version 0 so the no-op re-serve path can
    # re-publish it at the target revision (the settle under test).
    store.publish_result(
        WS,
        0,
        manifest={"checksum": "c0", "scene_version": 0},
        payload=b"payload-0",
        exact=True,
    )
    store.append_operations(WS, [op_item("op_1")])
    close_epoch(store, epoch_id=0)

    with store._write() as conn:
        exact_job = Store._insert_job_locked(
            conn,
            WS,
            target_scene_version=1,
            request={"exact_lane": {"target_revision": 1}},
            edit_watermark=0,
        )
    store.mark_job_running(exact_job)
    runner._finalize(
        store.require_job(exact_job),
        store.require_scenario(WS),
        SolveResult(status="no-op", scene_version=1),
        duration_ms=1.0,
    )
    assert store.require_job(exact_job).status == "complete"
    assert epoch_of(store, WS, 0).status == "exact_targeted"
    assert len(hub.exact) == 1, hub.exact
    workspace_id, event = hub.exact[0]
    assert workspace_id == WS
    assert event["exact_revision"] == 1
    assert event["workspace_revision"] == 1

    # A FAILED settle (deterministic edit rejection) advances no result:
    # no second event, no over-claimed exactness.
    store.append_operations(WS, [op_item("op_2")])
    store.mark_epoch_status(WS, 1, "closed")
    committed = store.commit_epoch_reduction(
        WS, 1, families={"vegetation_geometry": {"objects": {}}}
    )
    assert committed is not None
    with store._write() as conn:
        refuse_job = Store._insert_job_locked(
            conn,
            WS,
            target_scene_version=committed.workspace_revision,
            request={"exact_lane": {"target_revision": committed.workspace_revision}},
            edit_watermark=0,
        )
    store.mark_job_running(refuse_job)
    runner._finalize(
        store.require_job(refuse_job),
        store.require_scenario(WS),
        SolveResult(
            status="failed",
            scene_version=committed.workspace_revision,
            error={"code": "edit_rejected", "message": "unfoldable verb"},
        ),
        duration_ms=1.0,
    )
    assert store.require_job(refuse_job).status == "failed"
    assert len(hub.exact) == 1


def test_exact_settle_broadcasts_on_published_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The PRIMARY published path emits the named exact_revision event too
    (the settle test above drives only the no-op re-serve branch)."""
    from tests.test_realtime_epochs import epoch_of

    class RecordingHub:
        def __init__(self) -> None:
            self.exact: list[tuple[str, dict[str, Any]]] = []

        def broadcast_exact(self, workspace_id: str, event: dict[str, Any]) -> None:
            self.exact.append((workspace_id, dict(event)))

    store = make_store(tmp_path)
    make_workspace(store, WS)
    hub = RecordingHub()
    runner = make_runner(tmp_path, store, broadcast_hub=hub)

    store.append_operations(WS, [op_item("op_1")])
    close_epoch(store, epoch_id=0)
    manifest = {"checksum": "c1", "scene_version": 1, "metrics": {"mode": "test"}}
    monkeypatch.setattr(
        jobs_module,
        "materialize_result",
        lambda context, scenario, result, duration_ms: (manifest, b"payload-1"),
    )

    with store._write() as conn:
        job_id = Store._insert_job_locked(
            conn,
            WS,
            target_scene_version=1,
            request={"exact_lane": {"target_revision": 1}},
            edit_watermark=0,
        )
    store.mark_job_running(job_id)
    runner._finalize(
        store.require_job(job_id),
        store.require_scenario(WS),
        SolveResult(status="published", scene_version=1),
        duration_ms=1.0,
    )
    assert store.require_job(job_id).status == "complete"
    assert epoch_of(store, WS, 0).status == "exact_targeted"
    assert hub.exact == [(WS, {"workspace_id": WS, "exact_revision": 1, "workspace_revision": 1})]


def test_exact_settle_reemits_identical_event_on_crash_window_republish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Crash-window re-drive (ResultAlreadyPublished with matching bytes)
    must RE-EMIT the identical exact_revision event — idempotent by
    content, exactly the promise _broadcast_exact_revision's docstring
    makes for the publish-then-crash-before-finish window."""
    from tests.test_realtime_epochs import epoch_of

    class RecordingHub:
        def __init__(self) -> None:
            self.exact: list[tuple[str, dict[str, Any]]] = []

        def broadcast_exact(self, workspace_id: str, event: dict[str, Any]) -> None:
            self.exact.append((workspace_id, dict(event)))

    store = make_store(tmp_path)
    make_workspace(store, WS)
    hub = RecordingHub()
    runner = make_runner(tmp_path, store, broadcast_hub=hub)

    store.append_operations(WS, [op_item("op_1")])
    close_epoch(store, epoch_id=0)
    # The result is ALREADY durable at v1 (the first run published, then
    # crashed before finish_job); the re-run reproduces identical bytes.
    store.publish_result(
        WS,
        1,
        manifest={"checksum": "c1", "scene_version": 1},
        payload=b"payload-1",
        exact=True,
    )
    manifest = {"checksum": "c1", "scene_version": 1, "metrics": {"mode": "test"}}
    monkeypatch.setattr(
        jobs_module,
        "materialize_result",
        lambda context, scenario, result, duration_ms: (manifest, b"payload-1"),
    )

    with store._write() as conn:
        job_id = Store._insert_job_locked(
            conn,
            WS,
            target_scene_version=1,
            request={"exact_lane": {"target_revision": 1}},
            edit_watermark=0,
        )
    store.mark_job_running(job_id)
    runner._finalize(
        store.require_job(job_id),
        store.require_scenario(WS),
        SolveResult(status="published", scene_version=1),
        duration_ms=1.0,
    )
    assert store.require_job(job_id).status == "complete"
    assert epoch_of(store, WS, 0).status == "exact_targeted"
    assert hub.exact == [(WS, {"workspace_id": WS, "exact_revision": 1, "workspace_revision": 1})], hub.exact


def test_exact_completion_consumes_epochs_legacy_completion_does_not(
    tmp_path: Path,
) -> None:
    store = make_store(tmp_path)
    make_workspace(store, WS)
    runner = make_runner(tmp_path, store)

    # A published result at version 0 for the no-op republish path — it
    # must land while the scenario is still AT version 0 (publishing a
    # stale version is refused by contract).
    store.publish_result(
        WS,
        0,
        manifest={"checksum": "c0", "scene_version": 0},
        payload=b"payload-0",
        exact=True,
    )
    store.append_operations(WS, [op_item("op_1")])
    close_epoch(store, epoch_id=0)

    with store._write() as conn:
        exact_job = Store._insert_job_locked(
            conn,
            WS,
            target_scene_version=1,
            request={"exact_lane": {"target_revision": 1}},
            edit_watermark=0,
        )
    store.mark_job_running(exact_job)
    scenario = store.require_scenario(WS)
    runner._finalize(
        store.require_job(exact_job),
        scenario,
        SolveResult(status="no-op", scene_version=1),
        duration_ms=1.0,
    )
    assert store.require_job(exact_job).status == "complete"
    from tests.test_realtime_epochs import epoch_of

    assert epoch_of(store, WS, 0).status == "exact_targeted"
    assert store.require_scenario(WS).exact_result_version == 1
    # Consumed epochs leave the recoverable set.
    assert {e.workspace_id for e in store.recoverable_epochs()} == set()

    # A second epoch + a LEGACY job completing at its version: epochs are
    # the exact lane's to consume, and a legacy publish never does.
    store.append_operations(WS, [op_item("op_2")])
    store.mark_epoch_status(WS, 1, "closed")
    store.commit_epoch_reduction(
        WS, 1, families={"vegetation_geometry": {"objects": {}}}
    )
    with store._write() as conn:
        legacy_job = Store._insert_job_locked(
            conn,
            WS,
            target_scene_version=2,
            request={"time_indices": [0]},
            edit_watermark=0,
        )
    store.mark_job_running(legacy_job)
    runner._finalize(
        store.require_job(legacy_job),
        store.require_scenario(WS),
        SolveResult(status="no-op", scene_version=2),
        duration_ms=1.0,
    )
    assert store.require_job(legacy_job).status == "complete"
    assert epoch_of(store, WS, 1).status == "reducing", (
        "a legacy job completing must never consume epochs"
    )


# ---------------------------------------------------------------------------
# Zero-loss span derivation (bridge fence)
# ---------------------------------------------------------------------------


def test_span_derivation_missing_op_raises_zero_loss_fence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = make_store(tmp_path)
    make_workspace(store, WS)
    store.append_operations(WS, [op_item("op_1"), op_item("op_2")])
    close_epoch(store, epoch_id=0)
    runner = make_runner(tmp_path, store)
    solve = make_universal_dispatch_solver(lambda ctx: None)(runner.context)

    # Capture the REAL unbound method BEFORE the monkeypatch replaces the
    # class attribute (calling Store.operations_since after patching would
    # recurse into the stub — the r1 OneShotCommitFailure precedent).
    real_operations_since = Store.operations_since

    def lying_operations_since(inner_store, workspace_id: str, sequence: int):
        # Drops server_sequence 1 of the span (1, 2]: a partial fold must
        # refuse, never silently publish half the epoch.
        return [
            record
            for record in real_operations_since(inner_store, workspace_id, sequence)
            if record.server_sequence != 1
        ]

    monkeypatch.setattr(Store, "operations_since", lying_operations_since)
    request = SolveRequest(
        job_id="job_fence",
        scenario_id=WS,
        site_id=SITE_ID,
        target_scene_version=1,
        edit_watermark=0,
        trees=(),
        events=(),
        requested={},
        grid={"rows": 8, "cols": 8, "pixel_size_m": 2.0, "origin_x_m": 0.0,
              "origin_y_m": 0.0, "time_steps": 1},
        exact_lane={
            "target_revision": 1,
            "consumed_revision": 0,
            "consumed_seq": 0,
            "span_hi": 2,
        },
    )
    with pytest.raises(RuntimeError, match="zero-loss"):
        solve(request, lambda *args, **kwargs: None)


# ---------------------------------------------------------------------------
# Legacy funnel: /edits and /edits/universal land as realtime operations
# ---------------------------------------------------------------------------


@pytest.fixture()
def rt_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """App with a fake-clock epoch scheduler driven manually (no thread).

    Mirrors ``tests/test_realtime_epochs.rt_app`` (a local fixture —
    pytest fixtures do not travel through plain imports).
    """
    from tests.test_realtime_epochs import FakeClock

    clock = FakeClock()
    monkeypatch.setattr("solweig_gpu.server.store._now_utc", clock)
    app = make_rt_app(tmp_path, clock=clock)
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/scenarios",
            json={"site_id": SITE_ID, "name": "rt", "initial_state": "baseline"},
        )
        assert response.status_code == 201
        yield client, app, response.json()["scenario_id"], clock


def rt_op_with_gate(client: TestClient, workspace: str) -> None:
    """Open the collaborative gate: one realtime-native operation."""
    posted = rt_post(
        client,
        workspace,
        [
            {
                "operation_id": "gate_op_1",
                "client_sequence": 1,
                "base_revision": 0,
                "source_family": "vegetation_geometry",
                "entity_id": "tree-gate",
                "verb": "add",
                "payload": {"values": {"height_m": 9.0}},
            }
        ],
    )
    assert posted.status_code == 200, posted.text


def test_legacy_tree_edits_funnel_into_operation_log(rt_app) -> None:
    client, app, workspace, _clock = rt_app
    store = app.state.context.store
    rt_op_with_gate(client, workspace)

    tree = make_tree("tree_legacy_1", u=0.5, v=0.4)
    response = client.post(
        f"/api/v1/scenarios/{workspace}/edits",
        json={
            "base_scene_version": 0,
            "edits": [{"operation": "add", "tree": tree}],
            "requested_result": {"time_indices": [1]},
        },
        headers={"Idempotency-Key": "funnel-tree-1"},
    )
    assert response.status_code == 202, response.text
    scenario = store.require_scenario(workspace)
    assert scenario.scene_version == 1, "the legacy bump semantics are unchanged"

    ops = store.operations_since(workspace, 0)
    funnelled = [op for op in ops if op.actor_id == jobs_module.LEGACY_ACTOR_ID]
    assert len(funnelled) == 1, [op.operation_id for op in ops]
    op = funnelled[0]
    assert op.source_family == "vegetation_geometry"
    assert op.entity_id == "tree_legacy_1"
    assert op.verb == "add"
    assert op.server_sequence == 2, "the op log sees the whole collaborative history"
    # Adapter-side decomposition: API UV -> world metres, diameter -> radius.
    assert op.payload["values"] == {
        "x_m": 1048.0,
        "y_m": 1974.4,
        "height_m": 18.0,
        "canopy_radius_m": 5.5,
        "trunk_ratio": 0.25,
    }
    # base_revision rides ADVISORY (the route's If-Match equality gate is
    # untouched): the funnel never refuses on it and the store never gates.
    assert op.base_revision == 0

    # Retrying the SAME submit (idempotent redelivery) must not add a
    # second operation: the funnel's operation ids are deterministic.
    app.state.runner.submit(response.json()["job_id"])
    assert len([o for o in store.operations_since(workspace, 0)
                if o.actor_id == jobs_module.LEGACY_ACTOR_ID]) == 1


def test_universal_edits_funnel_into_operation_log(rt_app) -> None:
    client, app, workspace, _clock = rt_app
    store = app.state.context.store
    rt_op_with_gate(client, workspace)

    response = client.post(
        f"/api/v1/scenarios/{workspace}/edits/universal",
        json={
            "base_scene_version": 0,
            "edits": [
                {
                    "adapter": "meteorological_forcing",
                    "operation": "update_time_row",
                    "time_index": 1,
                    "values": {"air_temperature": 30.0},
                }
            ],
        },
        headers={"Idempotency-Key": "funnel-met-1"},
    )
    assert response.status_code == 202, response.text

    funnelled = [
        op
        for op in store.operations_since(workspace, 0)
        if op.actor_id == jobs_module.LEGACY_ACTOR_ID
    ]
    assert len(funnelled) == 1
    op = funnelled[0]
    assert op.source_family == "meteorological_forcing"
    assert op.verb == "set"
    assert op.payload["values"] == {"air_temperature": 30.0}
    assert op.payload["time_index"] == 1


def test_pure_legacy_workspace_is_untouched(tmp_path: Path) -> None:
    """No realtime history => no funnel, no epochs, no extra bumps."""
    from tests.test_server_api import make_app

    # Isolated telemetry: make_app wires create_app's DEFAULT scheduler
    # factories, whose fast-lane construction registers standard metrics
    # into the module-shared singleton at create time (app.py
    # _default_fast_lane) — this file runs before test_realtime_telemetry,
    # whose singleton-purity pin would otherwise fail (merge-interaction
    # of the r2a-fix default-wiring test with r2b's default lane wiring;
    # same hazard class the helpers above already isolate).
    def isolated_scheduler(store, hub):
        return EpochScheduler(store, hub=hub, telemetry=TelemetryRegistry())

    def isolated_fast_lane(store, hub):
        return FastLaneScheduler(store, hub=hub, telemetry=TelemetryRegistry())

    app = make_app(
        tmp_path,
        FakeSolver(),
        epoch_scheduler_factory=isolated_scheduler,
        fast_lane_factory=isolated_fast_lane,
    )
    with TestClient(app) as client:
        store = app.state.context.store
        created = client.post(
            "/api/v1/scenarios",
            json={"site_id": SITE_ID, "name": "legacy", "initial_state": "baseline"},
        )
        workspace = created.json()["scenario_id"]
        tree = make_tree("tree_pure_1")
        response = client.post(
            f"/api/v1/scenarios/{workspace}/edits",
            json={
                "base_scene_version": 0,
                "edits": [{"operation": "add", "tree": tree}],
                "requested_result": {"time_indices": [1]},
            },
            headers={"Idempotency-Key": "pure-legacy-1"},
        )
        assert response.status_code == 202, response.text

        assert store.operations_since(workspace, 0) == []
        assert store.epoch_records(workspace) == []
        scenario = store.require_scenario(workspace)
        assert scenario.scene_version == 1
        runner = app.state.runner
        runner._exact_lane_tick()
        assert exact_jobs(store, workspace) == []


# ---------------------------------------------------------------------------
# End to end: the exact lane publishes and consumes (real prepared site)
# ---------------------------------------------------------------------------


def test_exact_lane_publishes_and_consumes_epochs(
    tmp_path: Path, family_site
) -> None:
    from tests.test_server_universal_edits import (
        SITE_ID as FAMILY_SITE,
        _make_client,
        _scenario_ready,
    )

    # Isolated telemetry: create_app's default scheduler factory injects
    # the module-singleton registry, whose publish observations would leak
    # into tests that (correctly) assume a fresh singleton later in the
    # session (test_realtime_telemetry). r2b adds the same hazard for the
    # default fast-lane factory — isolate BOTH planes and keep the lane
    # thread off (this test asserts deterministic job/SSE sequences).
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
        with sse_subscribed(client, workspace):
            store = client.app.state.context.store
            geometry = dict(client.app.state.context.sites.geometry(FAMILY_SITE))
            center_x = float(geometry["origin_x_m"]) + 0.5 * int(geometry["cols"]) * float(
                geometry["pixel_size_m"]
            )
            center_y = float(geometry["origin_y_m"]) - 0.5 * int(geometry["rows"]) * float(
                geometry["pixel_size_m"]
            )

            posted = client.post(
                f"/api/v1/workspaces/{workspace}/operations",
                json={
                    "actor_id": "actor_a",
                    "operations": [
                        {
                            "operation_id": "exact_e2e_1",
                            "client_sequence": 1,
                            "base_revision": 0,
                            "source_family": "vegetation_geometry",
                            "entity_id": "tree-exact-1",
                            "verb": "add",
                            "payload": {
                                "values": {
                                    "x_m": center_x,
                                    "y_m": center_y,
                                    "height_m": 12.0,
                                    "canopy_radius_m": 5.5,
                                    "trunk_ratio": 0.25,
                                }
                            },
                        }
                    ],
                },
            )
            assert posted.status_code == 200, posted.text

            # The epoch closes (~one window), the tick targets the new
            # revision, the exact job solves and publishes. Poll the revision
            # triple on the workspace catch-up read: exact_revision catching
            # up to workspace_revision IS the contract.
            def caught_up() -> bool:
                body = client.get(
                    f"/api/v1/workspaces/{workspace}/operations"
                ).json()
                return (
                    body["workspace_revision"] >= 1
                    and body["exact_revision"] == body["workspace_revision"]
                )

            assert poll_until(caught_up, timeout=120.0), "exact lane never published"

            jobs = exact_jobs(store, workspace)
            assert jobs and jobs[-1]["status"] == "complete", jobs
            metrics = jobs[-1]["metrics"] or {}
            lane = metrics.get("exact_lane")
            assert lane is not None, metrics
            assert lane["span_native_ops"] == 1
            assert lane["target_revision"] >= 1
            assert lane["legacy_ops_skipped"] == 0
            epochs = store.epoch_records(workspace)
            assert [epoch.status for epoch in epochs] == ["exact_targeted"]
            assert {e.workspace_id for e in store.recoverable_epochs()} == set()


def test_compose_scatters_sparse_extra_patch_at_declared_timesteps(
    tmp_path: Path,
) -> None:
    """Merge regression (r3a ae0bccb into r2a): a sparse extra patch's
    DECLARED ``time_indices`` — never the dense block offset — select the
    destination timestep in ``_compose_current_state``'s extra-patch loop,
    the compose path the r2a exact lane shares with the legacy executor
    route. A dense-offset mis-scatter would land the changed t=3 plane at
    t=0 and leave the changed timestep stale (r3a review C1).
    """
    from solweig_gpu.incremental.geometry import RasterWindow
    from solweig_gpu.server.patch_codec import build_manifest, encode_payload

    store = make_store(tmp_path)
    make_workspace(store, WS)
    runner = make_runner(tmp_path, store)
    grid = {"rows": 8, "cols": 8, "time_steps": 4}

    # Durable base: a dense full-window v0 result whose utci plane at each
    # timestep carries that timestep as its sentinel value.
    base = np.stack(
        [np.full((8, 8), float(t), dtype=np.float32) for t in range(4)]
    )
    meta, payload, checksum = encode_payload({"utci": base}, ["utci"])
    manifest = build_manifest(
        scenario_id=WS,
        scene_version=0,
        window=RasterWindow(0, 8, 0, 8),
        time_indices=range(4),
        variables=meta,
        payload_url="unused",
        checksum=checksum,
        model_version="test",
        site_cache_version="test",
        site_id=SITE_ID,
    )
    store.publish_result(WS, 0, manifest=manifest, payload=payload, exact=True)

    # Sparse extra patch (r3a met fast path shape): covers ONLY global t=3.
    class _SparseExtra:
        write_window = RasterWindow(0, 8, 0, 8)
        time_indices = (3,)
        n_time_steps = 1  # dense-prefix fallback length
        arrays = {"utci": np.full((1, 8, 8), 99.0, dtype=np.float32)}

    state = jobs_module._compose_current_state(
        runner.context,
        WS,
        grid,
        ("utci",),
        RasterWindow(0, 8, 0, 8),
        1,
        extra_patches=[_SparseExtra()],
    )
    assert np.all(state["utci"][3] == 99.0), (
        "the sparse patch's plane must land at its declared timestep t=3"
    )
    for t in (0, 1, 2):
        assert np.all(state["utci"][t] == float(t)), (
            f"unchanged t={t} must keep the composed base plane — a "
            "dense-offset scatter would have overwritten t=0"
        )


# ---------------------------------------------------------------------------
# r2a-fix remediation (non-author review findings) — failing-first at 1ee0fd1
#
# BLOCKING-1a  pre-gate legacy objects must be known to the fold baseline
#              (gate-open bootstrap), so a native delete of a tree created
#              via legacy /edits folds to a REAL delta.
# BLOCKING-1b  a native delete/replace/move of an id the fold can never know
#              refuses LOUDLY (typed edit_rejected + lane metrics) and
#              SETTLES the span — never a silent no-delta, never a wedge.
# BLOCKING-2   a met-only (unmapped-family) span is a legitimate no-op: the
#              epoch settles via the no-op republish path and exact_revision
#              advances instead of wedging behind a refused batch.
# MEDIUM-1     skipped families are disclosed IN THE PUBLISHED MANIFEST's
#              limitations (published path and no-op republish path), not
#              job metrics only.
# LOW-1        the no-op finalize branch settles epochs even when the
#              republish fails (ResultAlreadyPublished stays idempotent).
# LOW-4(3/4)   the per-op epoch-revision fence and the canonical-baseline
#              fence (both pre-existing tripwires) get direct coverage.
# ---------------------------------------------------------------------------


def legacy_tree_event(
    store: Store, workspace: str, tree: Mapping[str, Any], *, sequence: int = 1
) -> None:
    """One PRE-GATE legacy ``/edits`` add, committed directly.

    The gate-open bootstrap folds the durable legacy ledger, so the unit
    seams only need the rows: one ``edit_events`` entry (never funnelled —
    no realtime operations existed when it was submitted) plus the
    authoritative ``scenario_trees`` row the executor seeds from.
    """
    tree_json = json.dumps(tree)
    with store._write() as conn:
        conn.execute(
            "INSERT INTO edit_events (scenario_id, sequence, event_id, "
            "base_scene_version, operation, tree_id, new_tree_json, submitted_at) "
            "VALUES (?, ?, ?, 0, 'add', ?, ?, ?)",
            (
                workspace,
                sequence,
                f"evt-{sequence}",
                tree["tree_id"],
                tree_json,
                "2026-09-04T00:00:00.000Z",
            ),
        )
        conn.execute(
            "INSERT INTO scenario_trees (scenario_id, tree_id, tree_json) "
            "VALUES (?, ?, ?)",
            (workspace, tree["tree_id"], tree_json),
        )


def native_delete_item(
    operation_id: str, entity_id: str, *, client_sequence: int = 1
) -> dict[str, Any]:
    return op_item(
        operation_id,
        entity_id=entity_id,
        verb="delete",
        payload={"values": {}},
        client_sequence=client_sequence,
    )


def exact_request(**overrides: Any) -> SolveRequest:
    """A minimal exact-lane :class:`SolveRequest` against the cheap site."""
    fields: dict[str, Any] = dict(
        job_id="job_exact_unit",
        scenario_id=WS,
        site_id=SITE_ID,
        target_scene_version=1,
        edit_watermark=0,
        trees=(),
        events=(),
        requested={},
        grid={
            "rows": 8,
            "cols": 8,
            "pixel_size_m": 2.0,
            "origin_x_m": 0.0,
            "origin_y_m": 0.0,
            "time_steps": 1,
        },
        exact_lane={
            "target_revision": 1,
            "consumed_revision": 0,
            "consumed_seq": 0,
            "span_hi": 2,
        },
    )
    fields.update(overrides)
    return SolveRequest(**fields)


def _capture_commands_stub(monkeypatch: pytest.MonkeyPatch, captured: dict) -> None:
    """Stub the executor solve: record the exact-lane factory's commands."""

    def fake_solve_with_executor(
        context, request, progress, *, extra_commands_factory=None, **kwargs
    ):
        executor = SimpleNamespace(scene_revision=0)
        captured["commands"] = (
            list(extra_commands_factory(executor, ("utci",), (0,)))
            if extra_commands_factory is not None
            else []
        )
        return SolveResult(
            status="no-op",
            scene_version=request.target_scene_version,
            metrics={"mode": "no-op"},
        )

    monkeypatch.setattr(executor_bridge, "_solve_with_executor", fake_solve_with_executor)


def _forbid_executor_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the executor solve to BLOW: the fence must refuse before it."""

    def exploding(context, request, progress, *, extra_commands_factory=None, **kwargs):
        raise AssertionError(
            "the unfoldable-native-op fence must refuse BEFORE any executor "
            "solve work runs"
        )

    monkeypatch.setattr(executor_bridge, "_solve_with_executor", exploding)


def _running_exact_job(store: Store, target: int) -> str:
    with store._write() as conn:
        job_id = Store._insert_job_locked(
            conn,
            WS,
            target_scene_version=target,
            request={"exact_lane": {"target_revision": target}},
            edit_watermark=0,
        )
    store.mark_job_running(job_id)
    return job_id


def test_pregate_legacy_objects_known_to_exact_lane_fold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """BLOCKING-1a unit pin: the gate-open bootstrap baseline carries the
    pre-gate legacy scene, so a native delete of a legacy-created tree is a
    real fold delta (a delete command onto the executor), never a silent
    delete-of-unknown no-op."""
    store = make_store(tmp_path)
    make_workspace(store, WS)
    runner = make_runner(tmp_path, store)
    # Pre-gate legacy world: one tree via /edits, acked, no realtime ops yet.
    legacy_tree_event(store, WS, make_tree("tree_pregate", u=0.5, v=0.5))
    # The collaborative era opens: native delete of that tree + a native add
    # (the add keeps the executor batch non-empty on a green run).
    store.append_operations(
        WS,
        [
            native_delete_item("op_del_pregate", "tree_pregate"),
            op_item(
                "op_add_new",
                entity_id="tree_new",
                client_sequence=2,
                payload={"values": {"x_m": 3.0, "y_m": 4.0, "height_m": 9.0}},
            ),
        ],
    )
    close_epoch(store, epoch_id=0)

    captured: dict[str, Any] = {}
    _capture_commands_stub(monkeypatch, captured)
    solve = make_universal_dispatch_solver(lambda ctx: None)(runner.context)
    result = solve(exact_request(), lambda *args, **kwargs: None)

    commands = captured.get("commands", [])
    summary = [(c.adapter_id, c.operation) for c in commands]
    deletes = [
        c
        for c in commands
        if c.adapter_id == "vegetation_geometry" and c.operation == "delete"
    ]
    assert deletes, (
        "the native delete of the pre-gate legacy tree vanished from the "
        f"fold (commands: {summary}) — the published result would keep the "
        "deleted tree's shadow"
    )
    assert deletes[0].old_state is not None
    assert deletes[0].old_state.get("tree_id") == "tree_pregate"
    adds = [
        c
        for c in commands
        if c.adapter_id == "vegetation_geometry" and c.operation == "add"
    ]
    assert adds and adds[0].new_state.get("tree_id") == "tree_new", summary
    assert result.error is None

    # The bootstrap is PINNED durably (revision 0) and deterministic.
    pinned = store.canonical_state_at(WS, 0)
    assert pinned is not None, "gate-open bootstrap must pin a rev-0 canonical row"
    objects = pinned.families["vegetation_geometry"]["objects"]
    assert "tree_pregate" in objects, pinned.families["vegetation_geometry"]


def test_unfoldable_native_delete_refuses_loudly_and_settles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """BLOCKING-1b: a native delete of an id no fold baseline can ever know
    (no legacy history at all) must refuse with a TYPED error + lane metric
    (never silently fold to no delta), and the refusal must SETTLE the span
    so the exact lane advances instead of wedging behind the churn guard."""
    from tests.test_realtime_epochs import epoch_of

    store = make_store(tmp_path)
    make_workspace(store, WS)
    runner = make_runner(tmp_path, store)
    store.append_operations(WS, [native_delete_item("op_del_ghost", "tree_ghost")])
    close_epoch(store, epoch_id=0)

    _forbid_executor_stub(monkeypatch)
    solve = make_universal_dispatch_solver(lambda ctx: None)(runner.context)
    result = solve(
        exact_request(exact_lane={
            "target_revision": 1,
            "consumed_revision": 0,
            "consumed_seq": 0,
            "span_hi": 1,
        }),
        lambda *args, **kwargs: None,
    )

    assert result.error is not None, "an unfoldable accepted op must never pass"
    assert result.error["code"] == "edit_rejected", result.error
    lane = (result.metrics or {}).get("exact_lane") or {}
    assert lane.get("unfoldable_native_ops") == 1, lane
    assert "tree_ghost" in json.dumps(result.error), result.error

    # Finalize: loud FAILED job, epochs SETTLED (exact_targeted), lane not
    # wedged at 'reducing' behind the churn guard.
    job_id = _running_exact_job(store, 1)
    runner._finalize(
        store.require_job(job_id),
        store.require_scenario(WS),
        result,
        duration_ms=1.0,
    )
    assert store.require_job(job_id).status == "failed"
    assert "tree_ghost" in json.dumps(store.require_job(job_id).error)
    assert epoch_of(store, WS, 0).status == "exact_targeted"
    assert {e.workspace_id for e in store.recoverable_epochs()} == set()


def test_noop_finalize_without_prior_result_settles_epochs(tmp_path: Path) -> None:
    """LOW-1: a no-op exact finalize with NO earlier result to re-serve
    fails the job but still settles the epochs — an unsettled epoch plus
    the churn guard would wedge the lane at that revision forever."""
    from tests.test_realtime_epochs import epoch_of

    store = make_store(tmp_path)
    make_workspace(store, WS)
    runner = make_runner(tmp_path, store)
    store.append_operations(WS, [op_item("op_1")])
    close_epoch(store, epoch_id=0)
    job_id = _running_exact_job(store, 1)

    runner._finalize(
        store.require_job(job_id),
        store.require_scenario(WS),
        SolveResult(status="no-op", scene_version=1),
        duration_ms=1.0,
    )
    assert store.require_job(job_id).status == "failed"
    assert epoch_of(store, WS, 0).status == "exact_targeted", (
        "no-op failure must still consume epochs (LOW-1)"
    )
    assert {e.workspace_id for e in store.recoverable_epochs()} == set()


def test_noop_republish_failure_settles_epochs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """LOW-1: when the no-op republish itself throws, the epochs still
    settle before the job's terminal finish (settle-then-finish)."""
    from solweig_gpu.server.store import StoreError

    from tests.test_realtime_epochs import epoch_of

    store = make_store(tmp_path)
    make_workspace(store, WS)
    runner = make_runner(tmp_path, store)
    store.publish_result(
        WS,
        0,
        manifest={"checksum": "c0", "scene_version": 0},
        payload=b"payload-0",
        exact=True,
    )
    store.append_operations(WS, [op_item("op_1")])
    close_epoch(store, epoch_id=0)
    job_id = _running_exact_job(store, 1)

    def exploding_republish(*args: Any, **kwargs: Any) -> None:
        raise StoreError("boom: republish path failed")

    monkeypatch.setattr(Store, "republish_result_at_version", exploding_republish)
    runner._finalize(
        store.require_job(job_id),
        store.require_scenario(WS),
        SolveResult(status="no-op", scene_version=1),
        duration_ms=1.0,
    )
    assert store.require_job(job_id).status == "superseded"
    assert epoch_of(store, WS, 0).status == "exact_targeted", (
        "a failing no-op republish must still consume epochs (LOW-1)"
    )
    assert {e.workspace_id for e in store.recoverable_epochs()} == set()


def test_per_op_epoch_revision_fence_trips(tmp_path: Path) -> None:
    """LOW-4(3): an operation whose epoch revision falls outside the span
    ``(consumed_revision, target_revision]`` trips the per-op zero-loss
    fence (RuntimeError, never a partial fold)."""
    store = make_store(tmp_path)
    make_workspace(store, WS)
    store.append_operations(WS, [op_item("op_1")])
    close_epoch(store, epoch_id=0)  # revision 1
    runner = make_runner(tmp_path, store)
    solve = make_universal_dispatch_solver(lambda ctx: None)(runner.context)
    # consumed_revision == the epoch's own revision: the span (1, 1] is
    # empty for it, so folding op_1 would double-apply a consumed epoch.
    request = exact_request(
        exact_lane={
            "target_revision": 1,
            "consumed_revision": 1,
            "consumed_seq": 0,
            "span_hi": 1,
        }
    )
    with pytest.raises(RuntimeError, match="belongs to epoch"):
        solve(request, lambda *args, **kwargs: None)


def test_missing_canonical_baseline_fence_trips(tmp_path: Path) -> None:
    """LOW-4(4): a consumed revision whose canonical state row is missing
    trips the canonical-baseline zero-loss fence (RuntimeError)."""
    store = make_store(tmp_path)
    make_workspace(store, WS)
    store.append_operations(WS, [op_item("op_1")])
    close_epoch(store, epoch_id=0)  # revision 1, canonical row at revision 1
    # A second epoch (revision 2) so the span op passes the per-op fence
    # and the fold reaches baseline resolution.
    store.append_operations(WS, [op_item("op_2")])
    store.mark_epoch_status(WS, 1, "closed")
    store.commit_epoch_reduction(
        WS, 1, families={"vegetation_geometry": {"objects": {}}}
    )
    with store._write() as conn:
        conn.execute(
            "DELETE FROM realtime_canonical_state WHERE workspace_id = ? "
            "AND workspace_revision = 1",
            (WS,),
        )
    runner = make_runner(tmp_path, store)
    solve = make_universal_dispatch_solver(lambda ctx: None)(runner.context)
    request = exact_request(
        target_scene_version=2,
        exact_lane={
            "target_revision": 2,
            "consumed_revision": 1,
            "consumed_seq": 1,
            "span_hi": 2,
        },
    )
    with pytest.raises(RuntimeError, match="canonical state at revision"):
        solve(request, lambda *args, **kwargs: None)


# -- End to end (real prepared site): bitwise + settle contracts --------------


def _arrays_at(
    client: TestClient, scenario_id: str, version: int
) -> dict[str, np.ndarray]:
    manifest = client.get(f"/api/v1/scenarios/{scenario_id}/results/{version}").json()
    payload = client.get(
        f"/api/v1/scenarios/{scenario_id}/results/{version}/payload"
    ).content
    return decode_payload(manifest, payload)


def _wait_exact_caught_up(
    client: TestClient, scenario_id: str, timeout: float = 120.0
):
    def caught() -> Any:
        body = client.get(f"/api/v1/scenarios/{scenario_id}").json()
        if body["exact_result_version"] == body["scene_version"]:
            return body["exact_result_version"]
        return None

    return poll_until(caught, timeout=timeout)


def _quiesce_workspace(
    client: TestClient, workspace: str, *, timeout: float = 60.0
) -> dict[str, Any]:
    """Poll the workspace catch-up until exact_revision == workspace_revision
    and stays there (the exact lane is drained)."""
    deadline = time.monotonic() + timeout
    stable = 0
    last = None
    while time.monotonic() < deadline:
        body = client.get(f"/api/v1/workspaces/{workspace}/operations").json()
        key = (body["workspace_revision"], body["exact_revision"])
        if key == last and body["exact_revision"] == body["workspace_revision"]:
            stable += 1
            if stable >= 6:
                return body
        else:
            stable, last = 0, key
        time.sleep(0.25)
    pytest.fail(f"workspace {workspace} never quiesced at {last}")


def _assert_nan_aware_bitwise_equal(
    actual: Mapping[str, np.ndarray], reference: Mapping[str, np.ndarray]
) -> None:
    """NaN-aware bitwise oracle (NaN != NaN is not divergence when BOTH
    lanes paint it — class-2 landcover legitimately NaNs utci)."""
    assert sorted(actual) == sorted(reference), (sorted(actual), sorted(reference))
    for name in sorted(actual):
        a, b = np.asarray(actual[name]), np.asarray(reference[name])
        assert a.shape == b.shape, (name, a.shape, b.shape)
        nan_mismatch = np.isnan(a) != np.isnan(b)
        value_diff = (a != b) & ~np.isnan(a) & ~np.isnan(b)
        if nan_mismatch.any() or value_diff.any():
            pytest.fail(
                f"variable {name}: diverged from the reference (nan mismatches "
                f"{int(nan_mismatch.sum())}, value diffs {int(value_diff.sum())})"
            )


def _native_veg_add_item(
    client: TestClient,
    site_id: str,
    operation_id: str,
    entity: str,
    *,
    u: float,
    v: float,
    client_sequence: int = 1,
) -> dict[str, Any]:
    from solweig_gpu.server.store import uv_to_world

    geometry = dict(client.app.state.context.sites.geometry(site_id))
    x_m, y_m = uv_to_world(
        u,
        v,
        rows=int(geometry["rows"]),
        cols=int(geometry["cols"]),
        pixel_size_m=float(geometry["pixel_size_m"]),
        origin_x_m=float(geometry["origin_x_m"]),
        origin_y_m=float(geometry["origin_y_m"]),
    )
    return {
        "operation_id": operation_id,
        "client_sequence": client_sequence,
        "base_revision": 0,
        "source_family": "vegetation_geometry",
        "entity_id": entity,
        "verb": "add",
        "payload": {
            "values": {
                "x_m": x_m,
                "y_m": y_m,
                "height_m": 13.0,
                "canopy_radius_m": 4.0,
                "trunk_ratio": 0.25,
            }
        },
    }


def _native_met_set_item(
    operation_id: str, *, temperature: float, client_sequence: int = 1
) -> dict[str, Any]:
    return {
        "operation_id": operation_id,
        "client_sequence": client_sequence,
        "base_revision": 0,
        "source_family": "meteorological_forcing",
        "entity_id": None,
        "verb": "set",
        "payload": {
            "values": {"air_temperature": temperature},
            "time_index": 1,
        },
    }


def _post_native_ops(client: TestClient, workspace: str, operations: list) -> None:
    # The route owns actor_id (request body) and stamps received_at itself;
    # the store-direct helpers carry both for append_operations, so strip
    # them here rather than maintaining two item builders.
    sanitized = [
        {key: value for key, value in op.items() if key not in ("actor_id", "received_at")}
        for op in operations
    ]
    posted = client.post(
        f"/api/v1/workspaces/{workspace}/operations",
        json={"actor_id": "r2a_fix", "operations": sanitized},
    )
    assert posted.status_code == 200, posted.text


def _isolated_scheduler_factory(store, hub):
    return EpochScheduler(store, hub=hub, telemetry=TelemetryRegistry())


@pytest.mark.scientific
def test_pregate_mixed_usage_bitwise_parity_vs_pure_legacy(
    tmp_path: Path, family_site
) -> None:
    """BLOCKING-1a bitwise pin: a tree created via LEGACY /edits BEFORE any
    realtime op, then natively deleted (+ a native add) in the collaborative
    era, must publish the SAME arrays as the identical scene built through
    the pure-legacy routes (at base the exact result kept the deleted
    tree's shadow)."""
    from tests.test_server_universal_edits import (
        SITE_ID as FAMILY_SITE,
        _make_client,
        _scenario_ready,
    )

    with _make_client(
        tmp_path, family_site, epoch_scheduler_factory=_isolated_scheduler_factory
    ) as client:
        # -- MIXED: pre-gate legacy add, then native delete + native add ----
        mixed = _scenario_ready(client)
        mixed_watch = SseSession(client, mixed)
        response = client.post(
            f"/api/v1/scenarios/{mixed}/edits",
            json={
                "base_scene_version": 0,
                "edits": [
                    {
                        "operation": "add",
                        "tree": make_tree(
                            "tree_pregate",
                            u=0.5,
                            v=0.5,
                            height_m=16.0,
                            canopy_diameter_m=11.0,
                        ),
                    }
                ],
            },
            headers={"Idempotency-Key": "parity-pregate-add"},
        )
        assert response.status_code == 202, response.text
        with_tree_version = _wait_exact_caught_up(client, mixed)
        assert with_tree_version is not None, "legacy pre-gate add never published"
        with_tree = _arrays_at(client, mixed, int(with_tree_version))

        _post_native_ops(
            client,
            mixed,
            [
                native_delete_item("parity_del_pregate", "tree_pregate"),
                _native_veg_add_item(
                    client,
                    FAMILY_SITE,
                    "parity_add_new",
                    "tree_new",
                    u=0.4,
                    v=0.6,
                    client_sequence=2,
                ),
            ],
        )
        body = _quiesce_workspace(client, mixed, timeout=120.0)
        final = _arrays_at(client, mixed, int(body["exact_revision"]))

        # The native delete MUST have changed the scene: at base the exact
        # result was bitwise the WITH-TREE scene (the delete was lost).
        changed = False
        for name in sorted(final):
            a, b = np.asarray(final[name]), np.asarray(with_tree[name])
            diff = (a != b) & ~np.isnan(a) & ~np.isnan(b)
            if int(diff.sum()) > 0:
                changed = True
        assert changed, (
            "the native delete of the pre-gate tree changed nothing — the "
            "published exact result still carries the deleted tree's shadow"
        )

        # -- PURE: the identical scene via legacy routes only ---------------
        pure = _scenario_ready(client)
        pure_watch = SseSession(client, pure)
        response = client.post(
            f"/api/v1/scenarios/{pure}/edits",
            json={
                "base_scene_version": 0,
                "edits": [
                    {
                        "operation": "add",
                        "tree": make_tree(
                            "tree_pregate",
                            u=0.5,
                            v=0.5,
                            height_m=16.0,
                            canopy_diameter_m=11.0,
                        ),
                    }
                ],
            },
            headers={"Idempotency-Key": "parity-pure-add"},
        )
        assert response.status_code == 202, response.text
        base_version = _wait_exact_caught_up(client, pure)
        assert base_version is not None
        response = client.post(
            f"/api/v1/scenarios/{pure}/edits",
            json={
                "base_scene_version": int(base_version),
                "edits": [
                    {"operation": "delete", "tree_id": "tree_pregate"},
                    {
                        "operation": "add",
                        "tree": make_tree(
                            "tree_new",
                            u=0.4,
                            v=0.6,
                            height_m=13.0,
                            canopy_diameter_m=8.0,
                        ),
                    },
                ],
            },
            headers={"Idempotency-Key": "parity-pure-del-add"},
        )
        assert response.status_code == 202, response.text
        pure_version = _wait_exact_caught_up(client, pure)
        assert pure_version is not None
        reference = _arrays_at(client, pure, int(pure_version))

        _assert_nan_aware_bitwise_equal(final, reference)


@pytest.mark.scientific
def test_met_only_epoch_settles_via_noop_republish(tmp_path: Path, family_site) -> None:
    """BLOCKING-2 pin: a met-only native op epoch (zero executor commands)
    settles through the no-op republish path — exact_revision advances, the
    epoch reaches exact_targeted, the job completes (mode no-op), and the
    REPUBLISHED manifest discloses the skipped family (MEDIUM-1)."""
    from tests.test_server_universal_edits import _make_client, _scenario_ready

    with _make_client(
        tmp_path, family_site, epoch_scheduler_factory=_isolated_scheduler_factory
    ) as client:
        workspace = _scenario_ready(client)
        watch = SseSession(client, workspace)
        store = client.app.state.context.store
        _post_native_ops(
            client,
            workspace,
            [_native_met_set_item("met_only_1", temperature=45.0)],
        )

        body = _quiesce_workspace(client, workspace, timeout=60.0)
        assert body["workspace_revision"] >= 1

        jobs = exact_jobs(store, workspace)
        final_job = jobs[-1]
        assert final_job["status"] == "complete", jobs
        metrics = final_job["metrics"] or {}
        assert metrics.get("mode") == "no-op", metrics
        lane = metrics.get("exact_lane") or {}
        assert lane.get("families_skipped") == ["meteorological_forcing"], lane

        epochs = store.epoch_records(workspace)
        assert epochs and all(
            epoch.status == "exact_targeted" for epoch in epochs
        ), [epoch.status for epoch in epochs]
        assert {e.workspace_id for e in store.recoverable_epochs()} == set()

        manifest = client.get(
            f"/api/v1/scenarios/{workspace}/results/{body['exact_revision']}"
        ).json()
        limitations = [str(item) for item in manifest.get("limitations", [])]
        assert any("meteorological_forcing" in item for item in limitations), (
            limitations,
            "MEDIUM-1: the republished manifest must disclose the skipped family",
        )


@pytest.mark.scientific
def test_unmapped_family_disclosed_in_published_manifest(
    tmp_path: Path, family_site
) -> None:
    """MEDIUM-1 published-path pin: a span mixing a mapped family (veg add)
    with an unmapped one (met set) publishes a real result whose MANIFEST
    limitations carry the skip disclosure (at base: metrics+log only)."""
    from tests.test_server_universal_edits import (
        SITE_ID as FAMILY_SITE,
        _make_client,
        _scenario_ready,
    )

    with _make_client(
        tmp_path, family_site, epoch_scheduler_factory=_isolated_scheduler_factory
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
                    "mix_veg_1",
                    "tree-mix-1",
                    u=0.5,
                    v=0.5,
                ),
                _native_met_set_item("mix_met_1", temperature=40.0,
                                     client_sequence=2),
            ],
        )
        body = _quiesce_workspace(client, workspace, timeout=120.0)

        jobs = exact_jobs(store, workspace)
        final_job = jobs[-1]
        assert final_job["status"] == "complete", jobs
        metrics = final_job["metrics"] or {}
        lane = metrics.get("exact_lane") or {}
        assert lane.get("families_skipped") == ["meteorological_forcing"], lane

        manifest = client.get(
            f"/api/v1/scenarios/{workspace}/results/{body['exact_revision']}"
        ).json()
        limitations = [str(item) for item in manifest.get("limitations", [])]
        assert any("meteorological_forcing" in item for item in limitations), (
            limitations,
            "MEDIUM-1: the published manifest must disclose the skipped family",
        )
