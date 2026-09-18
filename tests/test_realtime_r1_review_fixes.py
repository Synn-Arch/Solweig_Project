# SPDX-License-Identifier: GPL-3.0-only
"""R1 review remediation regressions (r1-epochs-review, APPROVE-WITH-NITS).

M1 — transient failure between ``mark_epoch_status('closed')`` and
``commit_epoch_reduction`` must not strand an epoch's operations outside
every canonical revision until restart: the live tick re-drives
closed-but-unassigned epochs, OLDEST FIRST, before closing later epochs
(no epoch-id/revision inversion).

M2 — a concurrent commit that lands between this close's baseline read and
its commit must be refused (typed ``EpochBaselineStale``) and retried on a
fresh baseline, so the latest canonical state never silently loses the
other epoch's effects.

L1 — an SSE subscriber whose snapshot revision went stale between the
route's scenario read and ``hub.subscribe`` receives a corrective snapshot
frame before live frames.

T2 — the zero-loss/ordering fences raise (tripwires survive ``python -O``)
instead of bare ``assert``.

All written failing-first against HEAD 1d14751.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from solweig_gpu.server.realtime.broadcast import BroadcastHub
from solweig_gpu.server.realtime.epochs import EpochScheduler
from solweig_gpu.server.realtime.reducer import DeterministicEpochReducer
from solweig_gpu.server.realtime.telemetry import TelemetryRegistry
from solweig_gpu.server.store import Store
from solweig_gpu.server.realtime import types as rt_types

from tests.test_realtime_epochs import (
    FakeClock,
    RecordingHub,
    SSEConnection,
    SSEReader,
    WINDOW_S,
    WS,
    close_window,
    epoch_of,
    make_rt_app,
    make_scheduler,
    make_store,
    make_workspace,
    op_item,
)
from tests.test_server_api import SITE_ID

# ---------------------------------------------------------------------------
# M1: stranded closed-unassigned epochs are re-driven by the LIVE cadence
# ---------------------------------------------------------------------------


class OneShotCommitFailure:
    """Store.commit_epoch_reduction stub that fails exactly once."""

    def __init__(self, store: Store) -> None:
        self._store = store
        # Capture the REAL unbound method BEFORE the monkeypatch replaces
        # the class attribute — calling type(store).commit_epoch_reduction
        # after patching would recurse into this stub.
        self._real = type(store).commit_epoch_reduction
        self.failed_once = False
        self.calls = 0

    def __call__(self, *args: Any, **kwargs: Any):
        self.calls += 1
        if not self.failed_once:
            self.failed_once = True
            raise RuntimeError("simulated transient sqlite I/O error")
        return self._real(self._store, *args, **kwargs)


def test_transient_failure_between_mark_and_commit_is_redriven_next_tick(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = make_store(tmp_path)
    make_workspace(store)
    clock = FakeClock()
    monkeypatch.setattr("solweig_gpu.server.store._now_utc", clock)
    scheduler, hub, _ = make_scheduler(store, clock=clock)

    stub = OneShotCommitFailure(store)
    monkeypatch.setattr(Store, "commit_epoch_reduction", stub)

    store.append_operations(WS, [op_item("op_1")])
    close_window(scheduler, clock)  # tick 1: commit raises (transient)

    epoch = epoch_of(store, WS, 0)
    assert epoch.status == "closed"  # marked, then the commit died
    assert epoch.workspace_revision is None
    assert store.require_scenario(WS).scene_version == 0
    assert hub.canonical == []

    close_window(scheduler, clock)  # tick 2: MUST re-drive (no restart)

    epoch = epoch_of(store, WS, 0)
    assert epoch.workspace_revision == 1, "stranded epoch must self-heal"
    assert epoch.status == "reducing"
    assert store.require_scenario(WS).scene_version == 1
    assert [event["operations"][0]["operation_id"] for event in
            [body for _, body in hub.canonical]] == ["op_1"]


def test_stranded_epoch_redrive_precedes_later_epoch_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No epoch-id/revision inversion: the older stranded epoch gets the
    smaller revision even when a younger epoch closes in the same tick."""
    store = make_store(tmp_path)
    make_workspace(store)
    clock = FakeClock()
    monkeypatch.setattr("solweig_gpu.server.store._now_utc", clock)
    scheduler, hub, _ = make_scheduler(store, clock=clock)

    stub = OneShotCommitFailure(store)
    monkeypatch.setattr(Store, "commit_epoch_reduction", stub)

    store.append_operations(WS, [op_item("op_a")])
    close_window(scheduler, clock)  # epoch 0 strands (commit failed once)

    stub.failed_once = True  # subsequent commits succeed normally
    store.append_operations(WS, [op_item("op_b")])  # lands in epoch 1
    close_window(scheduler, clock)  # one tick, TWO due epochs

    revisions = [epoch.workspace_revision for epoch in store.epoch_records(WS)]
    assert revisions == [1, 2], (
        f"epoch-id order must match revision order, got {revisions}"
    )
    ids_per_epoch = [
        [op["operation_id"] for op in body["operations"]]
        for _, body in hub.canonical
    ]
    assert ids_per_epoch == [["op_a"], ["op_b"]]


# ---------------------------------------------------------------------------
# M2: stale-baseline commit refused + retried (no silent state loss)
# ---------------------------------------------------------------------------


def test_stale_baseline_commit_refused_and_retried_on_fresh_baseline(
    tmp_path: Path,
) -> None:
    """A concurrent driver commits epoch 1 between this close's baseline
    read and its commit: the guard refuses the stale fold and the scheduler
    retries on the fresh baseline, so BOTH epochs' ops survive in the
    latest canonical state."""
    store = make_store(tmp_path)
    make_workspace(store)
    scheduler, hub, _ = make_scheduler(store)

    # Two epochs, both marked closed (simulating two racing drivers).
    # Epoch 0 must be non-open BEFORE op_b is appended, else the store
    # assigns both operations to the same open epoch and epoch 1 never
    # exists.
    store.append_operations(WS, [op_item("op_a")])
    store.mark_epoch_status(WS, 0, "closed")
    store.append_operations(WS, [op_item("op_b")])  # opens epoch 1
    store.mark_epoch_status(WS, 1, "closed")

    inner = DeterministicEpochReducer()

    class ConcurrentDriver(DeterministicEpochReducer):
        """reduce_epoch() is the close pipeline's stall point: epoch 1's
        driver commits HERE, after epoch 0 already read its baseline."""

        def reduce_epoch(self, baseline, operations):
            if not getattr(self, "_drove", False):
                self._drove = True
                scheduler.close_epoch(WS, 1)  # the other driver wins the race
            return inner.reduce_epoch(baseline, operations)

    scheduler._reducer = ConcurrentDriver()

    scheduler.close_epoch(WS, 0)

    latest = store.latest_canonical_state(WS)
    veg = latest.families["vegetation_geometry"]
    tree_ids = sorted(veg["objects"])
    assert tree_ids == ["tree-op_a", "tree-op_b"], (
        f"latest canonical state lost a concurrent epoch's effects: {tree_ids}"
    )
    revisions = [epoch.workspace_revision for epoch in store.epoch_records(WS)]
    assert revisions == [2, 1]  # epoch 1 committed first; epoch 0 retried after


# ---------------------------------------------------------------------------
# L1: SSE subscribe TOCTOU corrective snapshot
# ---------------------------------------------------------------------------


def test_subscribe_advancing_revision_gets_corrective_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An epoch closing between the route's scenario read and
    hub.subscribe leaves the snapshot stale: the subscriber must receive a
    corrective canonical_revision frame carrying the true revision."""

    clock = FakeClock()
    monkeypatch.setattr("solweig_gpu.server.store._now_utc", clock)
    app = make_rt_app(tmp_path, clock=clock)
    real_subscribe = BroadcastHub.subscribe

    def racing_subscribe(self: BroadcastHub, workspace_id: str, snapshot=None):
        # Simulate the close landing mid-route: bump the durable revision
        # AFTER the route built its snapshot but BEFORE registration.
        store = app.state.context.store
        store.append_operations(workspace_id, [op_item("op_toe")])
        app.state.epoch_scheduler.close_epoch(workspace_id, 0)
        return real_subscribe(self, workspace_id, snapshot=snapshot)

    monkeypatch.setattr(BroadcastHub, "subscribe", racing_subscribe)

    async def scenario() -> None:
        from fastapi.testclient import TestClient

        # Keep the TestClient lifespan OPEN through the SSE phase: exiting
        # the context runs app shutdown, which closes the store the SSE
        # route (driven directly below) still needs.
        with TestClient(app) as client:
            created = client.post(
                "/api/v1/scenarios",
                json={"site_id": SITE_ID, "name": "rt", "initial_state": "baseline"},
            ).json()
            workspace = created["scenario_id"]
            async with SSEConnection(
                app, f"/api/v1/workspaces/{workspace}/events"
            ) as conn:
                reader = SSEReader(conn.lines())
                name, first = await reader.next_event()
                assert name == "canonical_revision" and first["snapshot"] is True
                # Stale snapshot said revision 0; the corrective frame must
                # carry the post-close truth.
                name, second = await reader.next_event(timeout=5.0)
                assert name == "canonical_revision"
                assert second["workspace_revision"] == 1, (
                    f"subscriber stranded on stale revision: {second}"
                )

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# T2: fences raise (tripwires), not bare assert
# ---------------------------------------------------------------------------


def test_unsorted_epoch_operations_raise_runtime_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = make_store(tmp_path)
    make_workspace(store)
    scheduler, _, _ = make_scheduler(store)

    records = store.append_operations(WS, [op_item("op_1")])
    shuffled = [records[0], records[0]]  # duplicate = not strictly ordered

    monkeypatch.setattr(
        Store, "operations_for_epoch", lambda *a, **k: shuffled
    )
    store.mark_epoch_status(WS, 0, "closed")
    with pytest.raises(RuntimeError, match="out of order"):
        scheduler.close_epoch(WS, 0)


def test_zero_loss_fence_raises_runtime_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = make_store(tmp_path)
    make_workspace(store)
    hub = RecordingHub()
    scheduler = EpochScheduler(
        store, hub=hub, telemetry=TelemetryRegistry()
    )
    store.append_operations(WS, [op_item("op_1")])
    store.mark_epoch_status(WS, 0, "closed")

    from solweig_gpu.server.realtime import operations as rt

    real_body = rt.operation_body  # pre-patch capture; the stub must not recurse

    def lying_body(record: Any) -> dict[str, Any]:
        body = real_body(record)
        body["operation_id"] = "ghost"  # renderer "loses" the real op
        return body

    monkeypatch.setattr(rt, "operation_body", lying_body)
    with pytest.raises(RuntimeError, match="zero-loss fence"):
        scheduler.close_epoch(WS, 0)
