# SPDX-License-Identifier: GPL-3.0-only
"""R1 epochs wave: the 100 ms epoch scheduler, canonical revision
assignment, and SSE broadcast (failing-first).

Pins the mission contracts
(``docs/incremental_design_tool/realtime_collaboration/epoch_scheduler.md`` +
``collaborative_state.md``):

* epochs close on the scheduler tick cadence; ops arriving after close land
  in the NEXT epoch; epoch ids stay contiguous;
* every durably accepted operation lands in EXACTLY ONE canonical revision
  (zero loss across interleaved multi-actor submits);
* the revision bump is atomic with canonical-state persistence and
  idempotent under re-drive / restart recovery;
* the reducer threads canonical state across epochs (epoch N+1's baseline is
  epoch N's persisted state);
* the SSE route + catch-up endpoint satisfy the merged client wire contract
  (``studio/realtime_client.mjs``).

Scheduler unit tests inject a fake wall clock THROUGH THE STORE (patching
``store._now_utc``) so ``realtime_epochs.opened_at`` / operation
``accepted_at`` stamps advance deterministically with the scheduler's
injected ``utc_now``.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from solweig_gpu.server.app import create_app
from solweig_gpu.server.realtime.broadcast import BroadcastHub
from solweig_gpu.server.realtime.epochs import (
    DEFAULT_HEARTBEAT_INTERVAL_S,
    DEFAULT_TICK_MS,
    EPOCH_SCHEDULER_THREAD_NAME,
    EpochScheduler,
)
from solweig_gpu.server.realtime.scheduler import FastLaneScheduler
from solweig_gpu.server.realtime.telemetry import (
    METRIC_EPOCH_WAIT_MS,
    METRIC_EPOCHS_CLOSED_TOTAL,
    METRIC_PUBLISH_MS,
    METRIC_REDUCE_MS,
    TelemetryRegistry,
)
from solweig_gpu.server.realtime.types import ConflictKind
from solweig_gpu.server.store import (
    LATEST_SCHEMA_VERSION,
    Store,
)

from tests.test_server_api import SITE_ID, FakeSolver, make_site_cache

WS = "ws_1"
WINDOW_S = 0.1  # the 100 ms micro-epoch window (realtime_contract.yaml)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeClock:
    """Wall clock in the store's exact ``_now_utc`` format."""

    def __init__(self) -> None:
        self.now = datetime.now(timezone.utc)

    def __call__(self) -> str:
        return self.now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


class RecordingHub:
    """BroadcastHub stand-in that records canonical events for assertions."""

    def __init__(self) -> None:
        self.canonical: list[tuple[str, dict[str, Any]]] = []
        self.heartbeats = 0

    def broadcast_canonical(self, workspace_id: str, event: dict[str, Any]) -> None:
        self.canonical.append((workspace_id, dict(event)))

    def send_heartbeats(self) -> int:
        self.heartbeats += 1
        return 0


def make_store(tmp_path: Path) -> Store:
    return Store(tmp_path / "store.sqlite3", results_root=tmp_path / "scenarios")


def make_workspace(store: Store, workspace_id: str = WS) -> str:
    store.create_scenario(site_id="site", name="rt", scenario_id=workspace_id)
    return workspace_id


def op_item(operation_id: str, **overrides: Any) -> dict[str, Any]:
    item: dict[str, Any] = {
        "operation_id": operation_id,
        "actor_id": "actor_a",
        "client_sequence": 1,
        "base_revision": 0,
        "source_family": "vegetation_geometry",
        "entity_id": f"tree-{operation_id}",
        "verb": "add",
        "payload": {"values": {"x_m": 1.0, "y_m": 2.0, "height_m": 8.0}},
        "received_at": "2026-09-04T00:00:00.000Z",
    }
    item.update(overrides)
    return item


def veg_add(operation_id: str, entity: str, *, actor: str = "actor_a") -> dict[str, Any]:
    return op_item(
        operation_id,
        actor_id=actor,
        entity_id=entity,
        payload={"values": {"x_m": 1.0, "y_m": 2.0, "height_m": 8.0}},
    )


def make_scheduler(
    store: Store,
    hub: RecordingHub | None = None,
    clock: FakeClock | None = None,
    *,
    telemetry: TelemetryRegistry | None = None,
    tick_ms: float = 100.0,
) -> tuple[EpochScheduler, RecordingHub, FakeClock]:
    hub = hub if hub is not None else RecordingHub()
    clock = clock if clock is not None else FakeClock()
    scheduler = EpochScheduler(
        store,
        hub=hub,
        telemetry=telemetry if telemetry is not None else TelemetryRegistry(),
        tick_ms=tick_ms,
        utc_now=clock,
    )
    return scheduler, hub, clock


@pytest.fixture()
def clocked_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Store, FakeClock]:
    """A store whose durable stamps follow the injected fake clock."""
    clock = FakeClock()
    monkeypatch.setattr("solweig_gpu.server.store._now_utc", clock)
    store = make_store(tmp_path)
    make_workspace(store)
    return store, clock


def epoch_of(store: Store, workspace_id: str, epoch_id: int):
    return next(
        record
        for record in store.epoch_records(workspace_id)
        if record.epoch_id == epoch_id
    )


def close_window(scheduler: EpochScheduler, clock: FakeClock) -> dict[str, Any] | None:
    """Advance one full epoch window and run one scheduler tick."""
    clock.advance(WINDOW_S)
    scheduler.run_once()
    return None


# ---------------------------------------------------------------------------
# Constants / migration
# ---------------------------------------------------------------------------


def test_scheduler_constants_pin_the_contract() -> None:
    # realtime_contract.yaml: micro_epoch_ms 100; heartbeat default 5 s.
    assert DEFAULT_TICK_MS == 100.0
    assert DEFAULT_HEARTBEAT_INTERVAL_S == 5.0
    assert EPOCH_SCHEDULER_THREAD_NAME == "solweig-epoch-scheduler"


def test_migration_v8_adds_canonical_state_table(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    try:
        assert store.schema_version == LATEST_SCHEMA_VERSION == 9
        conn = sqlite3.connect(store.db_path)
        try:
            columns = {
                row[1]
                for row in conn.execute("PRAGMA table_info(realtime_canonical_state)")
            }
            assert columns == {
                "workspace_id",
                "workspace_revision",
                "state_json",
                "created_at",
            }
            # PK (workspace_id, workspace_revision): duplicate revision refused.
            conn.execute(
                "INSERT INTO realtime_canonical_state (workspace_id, "
                "workspace_revision, state_json, created_at) VALUES "
                "('w', 1, '{}', 't')"
            )
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO realtime_canonical_state (workspace_id, "
                    "workspace_revision, state_json, created_at) VALUES "
                    "('w', 1, '{}', 't')"
                )
            conn.rollback()
        finally:
            conn.close()
        # Migration is idempotent: reopening the same DB keeps version 9.
        store.close()
        reopened = Store(store.db_path, results_root=store.results_root)
        try:
            assert reopened.schema_version == 9
        finally:
            reopened.close()
    finally:
        # store.close() may have run inside the try; closing twice is a no-op
        # for this test's lifetime.
        pass


def test_store_epoch_queries_for_the_scheduler(clocked_store) -> None:
    store, _clock = clocked_store
    store.append_operations(WS, [op_item("op_1")])
    other = make_workspace(store, "ws_2")
    store.append_operations(other, [op_item("op_other", entity_id="tree-x")])
    assert sorted(store.open_epoch_workspaces()) == [WS, "ws_2"]
    recoverable = {(e.workspace_id, e.epoch_id) for e in store.recoverable_epochs()}
    assert recoverable == {(WS, 0), ("ws_2", 0)}
    # Terminal epochs leave the recoverable set.
    store.mark_epoch_status(WS, 0, "fast_published", workspace_revision=1)
    assert store.get_epoch(WS, 0).status == "fast_published"
    assert {e.workspace_id for e in store.recoverable_epochs()} == {"ws_2"}
    assert store.get_epoch(WS, 99) is None


def test_commit_epoch_reduction_is_atomic_and_idempotent(clocked_store) -> None:
    store, clock = clocked_store
    store.append_operations(WS, [op_item("op_1"), op_item("op_2")])
    families = {"vegetation_geometry": {"objects": {"tree-1": {"height_m": 8.0}}}}
    committed = store.commit_epoch_reduction(WS, 0, families=families)
    assert committed is not None
    assert (committed.workspace_revision, committed.first_time) == (1, True)
    # scene_version (== workspace_revision) bumped by exactly one, the epoch
    # carries the revision, and the canonical state row is durable.
    assert store.require_scenario(WS).scene_version == 1
    assert epoch_of(store, WS, 0).workspace_revision == 1
    state = store.latest_canonical_state(WS)
    assert state is not None
    assert state.workspace_revision == 1
    assert state.families["vegetation_geometry"]["objects"]["tree-1"] == {
        "height_m": 8.0
    }
    assert store.canonical_state_at(WS, 1) == state

    # Re-drive with DIFFERENT recomputed families: the revision never bumps
    # again and the ORIGINAL state bytes win (a re-fold from a newer baseline
    # must not rewrite history).
    again = store.commit_epoch_reduction(
        WS, 0, families={"vegetation_geometry": {"objects": {}}}
    )
    assert (again.workspace_revision, again.first_time) == (1, False)
    assert store.require_scenario(WS).scene_version == 1
    assert store.latest_canonical_state(WS) == state

    rows = sqlite3.connect(store.db_path).execute(
        "SELECT COUNT(*) FROM realtime_canonical_state WHERE workspace_id = ?",
        (WS,),
    ).fetchone()[0]
    assert rows == 1

    # Unknown epoch: typed None, no side effects.
    assert store.commit_epoch_reduction(WS, 42, families={}) is None


# ---------------------------------------------------------------------------
# Epoch close cadence + zero loss
# ---------------------------------------------------------------------------


def test_epoch_closes_on_first_tick_past_the_window(clocked_store) -> None:
    store, clock = clocked_store
    scheduler, hub, _ = make_scheduler(store, clock=clock)
    store.append_operations(WS, [op_item("op_1")])

    clock.advance(WINDOW_S - 0.001)  # 99 ms: not due yet
    scheduler.run_once()
    assert epoch_of(store, WS, 0).status == "open"
    assert hub.canonical == []

    clock.advance(0.001)  # exactly the 100 ms window boundary
    scheduler.run_once()
    epoch = epoch_of(store, WS, 0)
    assert epoch.status == "reducing"  # frozen vocabulary: no 'reduced' status
    assert epoch.workspace_revision == 1
    assert epoch.closed_at is not None
    assert store.require_scenario(WS).scene_version == 1
    assert len(hub.canonical) == 1


def test_ops_after_close_land_in_next_epoch_contiguous_ids(clocked_store) -> None:
    store, clock = clocked_store
    scheduler, hub, _ = make_scheduler(store, clock=clock)
    store.append_operations(WS, [op_item("early_1"), op_item("early_2")])
    close_window(scheduler, clock)

    late = store.append_operations(WS, [op_item("late_1"), op_item("late_2")])
    assert [record.epoch_id for record in late] == [1, 1]
    assert [record.server_sequence for record in late] == [3, 4]
    close_window(scheduler, clock)

    ids = [record.epoch_id for record in store.epoch_records(WS)]
    assert ids == [0, 1]  # contiguous
    assert [e.workspace_revision for e in store.epoch_records(WS)] == [1, 2]
    assert store.require_scenario(WS).scene_version == 2
    # Each epoch's broadcast carried exactly its own ops.
    first_ops = hub.canonical[0][1]["operations"]
    second_ops = hub.canonical[1][1]["operations"]
    assert [op["operation_id"] for op in first_ops] == ["early_1", "early_2"]
    assert [op["operation_id"] for op in second_ops] == ["late_1", "late_2"]


def test_zero_loss_three_actors_three_epochs(clocked_store) -> None:
    store, clock = clocked_store
    scheduler, hub, _ = make_scheduler(store, clock=clock)
    actors = ("alice", "bob", "carol")
    expected: list[str] = []
    revisions: list[int] = []
    for round_ in range(3):
        batch = [
            op_item(f"{actor}_r{round_}_{step}", actor_id=actor, entity_id=f"tree-{actor}-{step}")
            for actor in actors
            for step in range(2)
        ]
        records = store.append_operations(WS, batch)
        assert all(record.epoch_id == round_ for record in records)
        expected.extend(record.operation_id for record in records)
        close_window(scheduler, clock)
        revisions.append(hub.canonical[-1][1]["workspace_revision"])

    # Monotone +1 per non-empty epoch; scene_version advanced by 3.
    assert revisions == [1, 2, 3]
    assert store.require_scenario(WS).scene_version == 3
    # ZERO LOSS: every accepted operation appears in exactly ONE epoch's
    # broadcast payload (union == all, pairwise disjoint).
    per_epoch = [event["operations"] for _, event in hub.canonical]
    seen: list[str] = []
    for operations in per_epoch:
        ids = [op["operation_id"] for op in operations]
        assert not (set(ids) & set(seen)), "operation broadcast twice"
        seen.extend(ids)
    assert sorted(seen) == sorted(expected)
    # Rendered ops carry the identity triple the client's ack fence reads.
    for operations in per_epoch:
        for op in operations:
            assert {"operation_id", "server_sequence", "epoch_id"} <= set(op)


def test_duplicate_only_batch_after_close_never_bumps(clocked_store) -> None:
    store, clock = clocked_store
    scheduler, hub, _ = make_scheduler(store, clock=clock)
    store.append_operations(WS, [op_item("op_1")])
    close_window(scheduler, clock)
    assert store.require_scenario(WS).scene_version == 1
    broadcasts = len(hub.canonical)

    # A pure retry batch: duplicates never create rows, never open an epoch.
    store.append_operations(WS, [op_item("op_1", client_sequence=99)])
    close_window(scheduler, clock)
    assert [e.epoch_id for e in store.epoch_records(WS)] == [0]
    assert store.require_scenario(WS).scene_version == 1
    assert len(hub.canonical) == broadcasts


def test_defensive_empty_open_epoch_closes_without_bump(clocked_store) -> None:
    """r1-ops never creates an op-less epoch row, but if one exists the
    scheduler must close it WITHOUT assigning a revision."""
    store, clock = clocked_store
    scheduler, hub, _ = make_scheduler(store, clock=clock)
    conn = sqlite3.connect(store.db_path)
    try:
        opened_at = clock()
        conn.execute(
            "INSERT INTO realtime_epochs (workspace_id, epoch_id, status, "
            "opened_at) VALUES (?, 0, 'open', ?)",
            (WS, opened_at),
        )
        conn.commit()
    finally:
        conn.close()
    close_window(scheduler, clock)
    epoch = epoch_of(store, WS, 0)
    assert epoch.status == "closed"
    assert epoch.workspace_revision is None
    assert store.require_scenario(WS).scene_version == 0
    assert store.latest_canonical_state(WS) is None
    assert hub.canonical == []


# ---------------------------------------------------------------------------
# Idempotent re-drive + restart recovery
# ---------------------------------------------------------------------------


def test_close_pipeline_twice_single_bump_single_state_row(clocked_store) -> None:
    store, clock = clocked_store
    scheduler, hub, _ = make_scheduler(store, clock=clock)
    store.append_operations(WS, [op_item("op_1")])
    clock.advance(WINDOW_S)

    first = scheduler.close_epoch(WS, 0)
    second = scheduler.close_epoch(WS, 0)

    assert first is not None and second is not None
    assert first["workspace_revision"] == second["workspace_revision"] == 1
    assert store.require_scenario(WS).scene_version == 1
    rows = sqlite3.connect(store.db_path).execute(
        "SELECT COUNT(*) FROM realtime_canonical_state WHERE workspace_id = ?",
        (WS,),
    ).fetchone()[0]
    assert rows == 1
    # The re-drive RE-BROADCASTS (idempotent publish) but never re-bumps.
    assert len(hub.canonical) == 2
    assert hub.canonical[0][1]["operations"] == hub.canonical[1][1]["operations"]


def test_restart_recovery_assigns_revision_exactly_once(clocked_store) -> None:
    store, clock = clocked_store
    # Simulated crash between the status marks: epoch stuck 'closed' with
    # ops and no revision.
    store.append_operations(WS, [op_item("op_1"), op_item("op_2")])
    store.mark_epoch_status(WS, 0, "closed")

    scheduler2, hub2, _ = make_scheduler(store, clock=clock)
    driven = scheduler2.recover()
    assert driven == [(WS, 0)]
    epoch = epoch_of(store, WS, 0)
    assert epoch.status == "reducing"
    assert epoch.workspace_revision == 1
    assert store.require_scenario(WS).scene_version == 1
    assert len(hub2.canonical) == 1

    # A second restart re-drives (re-broadcast) but assigns nothing new.
    scheduler3, hub3, _ = make_scheduler(store, clock=clock)
    scheduler3.recover()
    assert store.require_scenario(WS).scene_version == 1
    assert len(hub3.canonical) == 1  # subscribers may have missed the event
    assert epoch_of(store, WS, 0).workspace_revision == 1


def test_restart_recovery_reducing_with_revision_only_rebroadcasts(
    clocked_store,
) -> None:
    store, clock = clocked_store
    scheduler, hub, _ = make_scheduler(store, clock=clock)
    store.append_operations(WS, [op_item("op_1")])
    close_window(scheduler, clock)
    assert store.require_scenario(WS).scene_version == 1
    # Crash after the durable commit, before R2 advanced the epoch.

    scheduler2, hub2, _ = make_scheduler(store, clock=clock)
    assert scheduler2.recover() == [(WS, 0)]
    assert store.require_scenario(WS).scene_version == 1  # no second bump
    rows = sqlite3.connect(store.db_path).execute(
        "SELECT COUNT(*) FROM realtime_canonical_state WHERE workspace_id = ?",
        (WS,),
    ).fetchone()[0]
    assert rows == 1
    assert len(hub2.canonical) == 1  # but the event was re-published
    assert hub2.canonical[0][1]["workspace_revision"] == 1


def test_recovery_leaves_young_open_epochs_to_normal_cadence(
    clocked_store,
) -> None:
    store, clock = clocked_store
    scheduler, hub, _ = make_scheduler(store, clock=clock)
    store.append_operations(WS, [op_item("op_1")])  # fresh open epoch
    assert scheduler.recover() == []
    assert epoch_of(store, WS, 0).status == "open"
    assert hub.canonical == []
    close_window(scheduler, clock)  # the tick, not recovery, closes it
    assert epoch_of(store, WS, 0).workspace_revision == 1


# ---------------------------------------------------------------------------
# Reducer integration: cross-epoch state threading
# ---------------------------------------------------------------------------


def test_multi_epoch_state_threading_and_cross_epoch_conflict(clocked_store) -> None:
    store, clock = clocked_store
    scheduler, hub, _ = make_scheduler(store, clock=clock)

    # Epoch 0: add tree-cross.
    store.append_operations(WS, [veg_add("add_1", "tree-cross")])
    close_window(scheduler, clock)
    state1 = store.latest_canonical_state(WS)
    assert state1 is not None and state1.workspace_revision == 1
    assert state1.families["vegetation_geometry"]["objects"]["tree-cross"][
        "height_m"
    ] == 8.0

    # Epoch 1: delete it — the baseline IS epoch 0's persisted state.
    store.append_operations(
        WS,
        [op_item("del_1", entity_id="tree-cross", verb="delete", payload={"reason": "x"})],
    )
    close_window(scheduler, clock)
    state2 = store.latest_canonical_state(WS)
    assert state2 is not None and state2.workspace_revision == 2
    veg = state2.families["vegetation_geometry"]
    assert veg["objects"] == {}
    assert veg["tombstones"] == {"tree-cross": 0}

    # Epoch 2: re-add over the tombstone -> GENERATION_RECREATED, and the new
    # generation folds from the tombstoned baseline.
    store.append_operations(WS, [veg_add("readd_1", "tree-cross", actor="bob")])
    close_window(scheduler, clock)
    state3 = store.latest_canonical_state(WS)
    assert state3.workspace_revision == 3
    recreated = state3.families["vegetation_geometry"]["objects"]["tree-cross"]
    assert recreated["generation"] == 1

    event = hub.canonical[-1][1]
    kinds = [conflict["kind"] for conflict in event["conflicts"]]
    assert ConflictKind.GENERATION_RECREATED in kinds
    recreation = next(
        conflict
        for conflict in event["conflicts"]
        if conflict["kind"] == ConflictKind.GENERATION_RECREATED
    )
    assert recreation["entity_id"] == "tree-cross"
    assert recreation["winner_operation_id"] == "readd_1"
    assert recreation["detail"]["resurrected"] is True


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------


def test_telemetry_stages_recorded_per_close(clocked_store) -> None:
    store, clock = clocked_store
    registry = TelemetryRegistry()
    scheduler, hub, _ = make_scheduler(store, clock=clock, telemetry=registry)
    store.append_operations(WS, [op_item("op_1"), op_item("op_2")])
    close_window(scheduler, clock)
    # Second window carries its own op (r1-ops never opens op-less epochs,
    # so every closed epoch here is a real close).
    store.append_operations(WS, [op_item("op_3")])
    close_window(scheduler, clock)

    snapshot = registry.snapshot()
    assert snapshot[METRIC_EPOCH_WAIT_MS][""]["count"] >= 1
    # The wait is the oldest accepted_at -> close gap: one 100 ms window.
    wait = snapshot[METRIC_EPOCH_WAIT_MS][""]["last"]
    assert 90.0 <= wait <= 110.0
    assert snapshot[METRIC_REDUCE_MS][""]["count"] >= 1
    assert snapshot[METRIC_PUBLISH_MS][""]["count"] >= 1
    assert snapshot[METRIC_EPOCHS_CLOSED_TOTAL][""]["value"] >= 2


# ---------------------------------------------------------------------------
# Scheduler lifecycle (thread hygiene)
# ---------------------------------------------------------------------------


def scheduler_threads() -> list[threading.Thread]:
    return [
        thread
        for thread in threading.enumerate()
        if thread.name == EPOCH_SCHEDULER_THREAD_NAME
    ]


def wait_for_scheduler_exit(timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not scheduler_threads():
            return True
        time.sleep(0.02)
    return not scheduler_threads()


def test_stop_joins_promptly_and_two_app_cycles_leave_no_thread(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Isolate the module-level telemetry singleton: this test deliberately
    # exercises the DEFAULT app wiring (whose scheduler registers standard
    # metrics on the shared registry), but suite-order pollution of
    # default_registry() breaks test_realtime_telemetry's count assertions.
    monkeypatch.setattr(
        "solweig_gpu.server.app.default_registry",
        lambda: TelemetryRegistry(),
    )

    # Cycle 1: the DEFAULT app wiring (real scheduler thread, real clock).
    app = create_app(
        state_root=tmp_path / "state1",
        sites={SITE_ID: {"cache_dir": make_site_cache(tmp_path)}},
        solver_factory=lambda context: FakeSolver(),
        start_worker=False,
    )
    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200
        assert scheduler_threads(), "scheduler thread should be live in-app"
        scheduler = app.state.epoch_scheduler
        assert scheduler.running
        started = time.monotonic()
        scheduler.stop(timeout=2.0)
        assert time.monotonic() - started < 2.0, "stop must join promptly"
    assert wait_for_scheduler_exit(), "no scheduler thread may outlive the app"

    # Cycle 2: same dance on a fresh app (suite-wide leak guard).
    app2 = create_app(
        state_root=tmp_path / "state2",
        sites={SITE_ID: {"cache_dir": make_site_cache(tmp_path)}},
        solver_factory=lambda context: FakeSolver(),
        start_worker=False,
    )
    with TestClient(app2) as client2:
        assert client2.get("/healthz").status_code == 200
    assert wait_for_scheduler_exit(), "second cycle leaked a scheduler thread"


# ---------------------------------------------------------------------------
# App / route surface: SSE + catch-up + acks alias
# ---------------------------------------------------------------------------


def make_rt_app(
    tmp_path: Path,
    *,
    clock: FakeClock,
    start_epoch_scheduler: bool = False,
    **kwargs: Any,
):
    kwargs.setdefault("requests_per_minute_per_ip", None)
    kwargs.setdefault("edits_per_minute", None)

    def scheduler_factory(store, hub):
        return EpochScheduler(store, hub=hub, utc_now=clock, tick_ms=100.0)

    # r2b: keep the fast lane OFF in these (fake-clock, manually driven)
    # apps — a live lane would advance fast_revision and interleave
    # fast_revision SSE frames nondeterministically with the frame-order
    # assertions below. tests/test_realtime_r2_fast_lane.py owns the lane.
    def fast_lane_factory(store, hub):
        return FastLaneScheduler(store, hub=hub, utc_now=clock)

    kwargs.setdefault("start_fast_lane", False)
    return create_app(
        state_root=tmp_path / "state",
        sites={SITE_ID: {"cache_dir": make_site_cache(tmp_path)}},
        solver_factory=lambda context: FakeSolver(),
        coalescing_window_ms=20.0,
        start_worker=True,
        epoch_scheduler_factory=scheduler_factory,
        start_epoch_scheduler=start_epoch_scheduler,
        fast_lane_factory=fast_lane_factory,
        **kwargs,
    )


@pytest.fixture()
def rt_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """App with a fake-clock epoch scheduler driven manually (no thread)."""
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


class SSEReader:
    """Async SSE frame reader over an async line iterator."""

    def __init__(self, lines: Any) -> None:
        self._lines = lines
        self._buffer: list[str] = []

    async def next_event(self, timeout: float = 5.0) -> tuple[str, dict[str, Any]]:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            while "" in self._buffer:
                end = self._buffer.index("")  # blank line closes a frame
                frame, self._buffer = self._buffer[:end], self._buffer[end + 1 :]
                parsed = self._parse(frame)
                if parsed is not None:
                    return parsed
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise AssertionError(
                    f"timed out waiting for an SSE event; buffer={self._buffer!r}"
                )
            try:
                line = await asyncio.wait_for(self._lines.__anext__(), remaining)
            except StopAsyncIteration:
                raise AssertionError(
                    f"SSE stream ended early; buffer={self._buffer!r}"
                ) from None
            except TimeoutError:
                raise AssertionError(
                    f"SSE stream stalled; buffer={self._buffer!r}"
                ) from None
            self._buffer.append(line)

    @staticmethod
    def _parse(frame: list[str]) -> tuple[str, dict[str, Any]] | None:
        name = None
        data: list[str] = []
        for line in frame:
            if line.startswith("event:"):
                name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data.append(line[len("data:") :].strip())
        if name is None:
            return None  # comment/keep-alive block
        return name, json.loads("\n".join(data))


class SSEConnection:
    """Incremental SSE consumer driving the app's raw ASGI interface.

    Neither the installed starlette ``TestClient`` nor ``httpx``'s
    ``ASGITransport`` stream incrementally (both run the app to completion
    and buffer the body), so an infinite SSE response deadlocks through
    them. This connection runs ``app(scope, receive, send)`` as a task,
    pushes each ``http.response.body`` chunk onto a queue the test reads
    live, and turns ``close()`` into an ASGI ``http.disconnect`` — exactly
    what a real client going away does.
    """

    _DONE = object()

    def __init__(self, app: Any, path: str) -> None:
        self._app = app
        self._path = path
        self._chunks: asyncio.Queue = asyncio.Queue()
        self._disconnect = asyncio.Event()
        self._started = asyncio.Event()
        self._request_sent = False
        self._task: asyncio.Task | None = None
        self.status: int | None = None
        self.headers: dict[str, str] = {}

    async def __aenter__(self) -> "SSEConnection":
        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": self._path,
            "raw_path": self._path.encode(),
            "query_string": b"",
            "headers": [(b"accept", b"text/event-stream")],
            "server": ("testserver", 80),
            "client": ("127.0.0.1", 123),
            "root_path": "",
        }
        self._task = asyncio.create_task(self._app(scope, self._receive, self._send))
        await asyncio.wait_for(self._started.wait(), 5.0)
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    async def _receive(self) -> dict[str, Any]:
        if not self._request_sent:
            self._request_sent = True
            return {"type": "http.request", "body": b"", "more_body": False}
        await self._disconnect.wait()
        return {"type": "http.disconnect"}

    async def _send(self, message: dict[str, Any]) -> None:
        kind = message["type"]
        if kind == "http.response.start":
            self.status = int(message["status"])
            self.headers = {
                key.decode(): value.decode()
                for key, value in message.get("headers", [])
            }
            self._started.set()
        elif kind == "http.response.body":
            body = message.get("body", b"")
            if body:
                self._chunks.put_nowait(body)
            if not message.get("more_body", False):
                self._chunks.put_nowait(self._DONE)

    async def lines(self):
        buffer = b""
        while True:
            item = await self._chunks.get()
            if item is self._DONE:
                return
            buffer += item
            while b"\n" in buffer:
                line, _, buffer = buffer.partition(b"\n")
                yield line.decode()

    async def close(self) -> None:
        """Disconnect the client and let the app finish cleanly."""
        self._disconnect.set()
        task = self._task
        if task is None:
            return
        try:
            await asyncio.wait_for(task, 5.0)
        except TimeoutError:  # pragma: no cover - app ignored disconnect
            task.cancel()


def rt_post(
    client: TestClient, workspace: str, operations: list[dict[str, Any]], actor: str = "actor_a"
):
    return client.post(
        f"/api/v1/workspaces/{workspace}/operations",
        json={"actor_id": actor, "operations": operations},
    )


def route_item(operation_id: str, **overrides: Any) -> dict[str, Any]:
    item: dict[str, Any] = {
        "operation_id": operation_id,
        "client_sequence": 1,
        "base_revision": 0,
        "source_family": "vegetation_geometry",
        "entity_id": f"tree-{operation_id}",
        "verb": "add",
        "payload": {"values": {"height_m": 9.0}},
    }
    item.update(overrides)
    return item


def test_sse_snapshot_then_canonical_frame(rt_app) -> None:
    client, app, workspace, clock = rt_app
    scheduler = app.state.epoch_scheduler
    hub: BroadcastHub = app.state.broadcast_hub

    async def scenario() -> None:
        async with SSEConnection(app, f"/api/v1/workspaces/{workspace}/events") as conn:
            assert conn.status == 200
            assert conn.headers["content-type"].startswith("text/event-stream")
            reader = SSEReader(conn.lines())

            # Late joiner: the FIRST frame is a state-snapshot
            # canonical_revision (revisions from the scenario row, EMPTY
            # operations — the catch-up GET owns history).
            name, snapshot = await reader.next_event()
            assert name == "canonical_revision"
            assert snapshot["operations"] == []
            assert snapshot["workspace_revision"] == 0
            assert snapshot["fast_revision"] == 0
            assert snapshot["exact_revision"] == 0
            assert snapshot["snapshot"] is True

            # A live submit, then the epoch close pushes the event. (The
            # TestClient call blocks this loop briefly; the SSE task stays
            # parked on its queue, and the broadcast lands afterwards.)
            posted = rt_post(client, workspace, [route_item("op_sse_1")])
            assert posted.status_code == 200
            clock.advance(WINDOW_S)
            scheduler.run_once()

            name, event = await reader.next_event()
            assert name == "canonical_revision"
            assert event["workspace_revision"] == 1
            assert event["epoch_id"] == 0
            operations = event["operations"]
            assert len(operations) == 1
            assert operations[0]["operation_id"] == "op_sse_1"
            assert operations[0]["server_sequence"] == 1
            assert operations[0]["epoch_id"] == 0
            # Revision triple + invariant on every canonical event.
            assert event["fast_revision"] == 0
            assert event["exact_revision"] == 0
            assert (
                event["exact_revision"]
                <= event["fast_revision"]
                <= event["workspace_revision"]
            )

            # Heartbeat frames (stall detection) flow on the same stream.
            hub.send_heartbeats()
            beat_name, _beat = await reader.next_event()
            assert beat_name == "heartbeat"

        # Disconnect cleaned the hub: no subscriber leak.
        deadline = asyncio.get_running_loop().time() + 5.0
        while hub.subscriber_count(workspace) and (
            asyncio.get_running_loop().time() < deadline
        ):
            await asyncio.sleep(0.02)
        assert hub.subscriber_count(workspace) == 0

    asyncio.run(scenario())


def test_catch_up_endpoint_serves_exact_set_since_watermark(rt_app) -> None:
    client, app, workspace, clock = rt_app
    scheduler = app.state.epoch_scheduler
    for round_ in range(2):
        rt_post(
            client,
            workspace,
            [route_item(f"op_{round_}_1"), route_item(f"op_{round_}_2")],
        )
        clock.advance(WINDOW_S)
        scheduler.run_once()

    catch_up = client.get(
        f"/api/v1/workspaces/{workspace}/operations",
        params={"since_server_sequence": 2},
    )
    assert catch_up.status_code == 200
    body = catch_up.json()
    assert body["workspace_id"] == workspace
    sequences = [op["server_sequence"] for op in body["operations"]]
    assert sequences == [3, 4]  # exact set, ascending
    assert body["workspace_revision"] == 2
    assert body["fast_revision"] == 0
    assert body["exact_revision"] == 0

    # Default watermark serves the full durable log.
    everything = client.get(f"/api/v1/workspaces/{workspace}/operations").json()
    assert [op["server_sequence"] for op in everything["operations"]] == [1, 2, 3, 4]
    # Unknown workspace: typed 404.
    missing = client.get("/api/v1/workspaces/scn_missing/operations")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "scenario_not_found"


def test_post_operations_response_carries_acks_alias(rt_app) -> None:
    client, _app, workspace, _clock = rt_app
    response = rt_post(client, workspace, [route_item("op_ack_1")])
    assert response.status_code == 200
    body = response.json()
    # The merged client reads ``acks ?? operations`` — without the alias
    # every submit stalls until epoch close.
    assert body["acks"] == body["accepted"]
    assert [item["operation_id"] for item in body["acks"]] == ["op_ack_1"]


def test_sse_route_unknown_workspace_404(rt_app) -> None:
    client, _app, _workspace, _clock = rt_app
    with client.stream("GET", "/api/v1/workspaces/scn_missing/events") as stream:
        assert stream.status_code == 404


def test_real_scheduler_thread_closes_epochs_end_to_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full default wiring (real thread, real clock): a submitted operation
    becomes a canonical revision broadcast within a few epoch windows."""
    # Default wiring registers standard metrics on the module-level shared
    # registry; isolate it so the epoch this test closes does not pollute
    # test_realtime_telemetry's count assertions in combined runs.
    monkeypatch.setattr(
        "solweig_gpu.server.app.default_registry",
        lambda: TelemetryRegistry(),
    )
    app = create_app(
        state_root=tmp_path / "state",
        sites={SITE_ID: {"cache_dir": make_site_cache(tmp_path)}},
        solver_factory=lambda context: FakeSolver(),
        coalescing_window_ms=20.0,
        start_worker=True,
        requests_per_minute_per_ip=None,
        edits_per_minute=None,
    )
    with TestClient(app) as client:
        created = client.post(
            "/api/v1/scenarios",
            json={"site_id": SITE_ID, "name": "rt", "initial_state": "baseline"},
        )
        workspace = created.json()["scenario_id"]

        async def scenario() -> None:
            async with SSEConnection(
                app, f"/api/v1/workspaces/{workspace}/events"
            ) as conn:
                reader = SSEReader(conn.lines())
                name, _snapshot = await reader.next_event()
                assert name == "canonical_revision"
                # r2b review F1 re-pin: over several epochs (each iteration
                # re-opens the commit->broadcast window the fast lane's
                # 100 ms discovery scan races), canonical_revision(R) MUST
                # reach subscribers before fast_revision(R).
                for revision in (1, 2, 3, 4, 5):
                    posted = rt_post(
                        client,
                        workspace,
                        [
                            route_item(
                                f"op_live_{revision}",
                                client_sequence=revision,
                                base_revision=revision - 1,
                            )
                        ],
                    )
                    assert posted.status_code == 200
                    canonical_seen_at = None
                    fast_seen_at = None
                    arrivals = 0
                    while canonical_seen_at is None or fast_seen_at is None:
                        name, event = await reader.next_event(timeout=10.0)
                        arrivals += 1
                        if (
                            name == "canonical_revision"
                            and event.get("workspace_revision") == revision
                        ):
                            canonical_seen_at = arrivals
                            if revision == 1:
                                assert (
                                    event["operations"][0]["operation_id"]
                                    == "op_live_1"
                                )
                        if (
                            name == "fast_revision"
                            and event.get("workspace_revision") == revision
                        ):
                            fast_seen_at = arrivals
                        assert arrivals <= 8, "ordering frames never arrived"
                    assert canonical_seen_at is not None
                    assert fast_seen_at is not None
                    assert (
                        canonical_seen_at < fast_seen_at
                    ), f"fast_revision({revision}) preceded canonical_revision({revision})"

        asyncio.run(scenario())
    assert wait_for_scheduler_exit()
