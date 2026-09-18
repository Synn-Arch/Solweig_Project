# SPDX-License-Identifier: GPL-3.0-only
"""R1 durable operation plane: idempotent append, epoch assignment, routes.

Written failing-first against the mission contract
(``docs/incremental_design_tool/realtime_collaboration/collaborative_state.md``
+ ``epoch_scheduler.md``): every client operation is an idempotent append to
a per-workspace contiguous log, assigned to the workspace's open epoch;
``base_revision`` is advisory and never rejects; the log is append-only.

Store-level tests exercise :class:`solweig_gpu.server.store.Store` directly;
the route tests drive one full app through ``TestClient`` with the synthetic
site-cache fixture and a fake solver (same pattern as ``test_server_api``).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from solweig_gpu.server.app import create_app
from solweig_gpu.server.realtime import operations as rt
from solweig_gpu.server.realtime.broadcast import BroadcastHub
from solweig_gpu.server.realtime.epochs import EpochScheduler
from solweig_gpu.server.realtime.scheduler import FastLaneScheduler
from solweig_gpu.server.realtime.telemetry import TelemetryRegistry
from solweig_gpu.server.store import (
    LATEST_SCHEMA_VERSION,
    OperationFingerprintConflict,
    ScenarioNotFound,
    Store,
)

from tests.test_server_api import SITE_ID, FakeSolver, make_site_cache


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def make_store(tmp_path: Path) -> Store:
    return Store(tmp_path / "store.sqlite3", results_root=tmp_path / "scenarios")


def make_workspace(store: Store, workspace_id: str = "ws_1") -> str:
    store.create_scenario(site_id="site", name="rt", scenario_id=workspace_id)
    return workspace_id


def op_item(operation_id: str, **overrides: Any) -> dict[str, Any]:
    item: dict[str, Any] = {
        "operation_id": operation_id,
        "actor_id": "actor_a",
        "client_sequence": 1,
        "base_revision": 0,
        "source_family": "building_geometry",
        "entity_id": "b1",
        "verb": "add",
        "payload": {"footprint_uv": [[0.1, 0.1], [0.2, 0.2]], "height_m": 12.0},
        "received_at": "2026-09-04T00:00:00.000Z",
    }
    item.update(overrides)
    return item


def make_rt_app(tmp_path: Path, solver: FakeSolver, **kwargs: Any):
    kwargs.setdefault("requests_per_minute_per_ip", None)
    kwargs.setdefault("edits_per_minute", None)

    # Isolate the epoch scheduler's telemetry: the default app wiring
    # registers standard metrics on the module-level shared registry, and a
    # real-thread epoch close inside these route tests would pollute
    # test_realtime_telemetry's singleton-count assertions in combined runs.
    def scheduler_factory(store, hub: BroadcastHub):
        return EpochScheduler(store, hub=hub, telemetry=TelemetryRegistry())

    # Isolate the fast lane the same way (r2b): private telemetry AND no
    # daemon thread — these route tests drive the plane synchronously, and
    # a live fast lane would advance fast_revision / epoch statuses (and
    # enqueue fast_revision SSE frames) nondeterministically between the
    # route call and its assertions. r2b's own tests exercise the lane.
    def fast_lane_factory(store, hub: BroadcastHub):
        return FastLaneScheduler(store, hub=hub, telemetry=TelemetryRegistry())

    kwargs.setdefault("start_fast_lane", False)
    return create_app(
        state_root=tmp_path / "state",
        sites={SITE_ID: {"cache_dir": make_site_cache(tmp_path)}},
        solver_factory=lambda context: solver,
        coalescing_window_ms=20.0,
        start_worker=True,
        epoch_scheduler_factory=scheduler_factory,
        fast_lane_factory=fast_lane_factory,
        **kwargs,
    )


@pytest.fixture()
def rt_workspace(tmp_path: Path) -> tuple[TestClient, str]:
    """One live app + one scenario row (the workspace) to operate on."""
    app = make_rt_app(tmp_path, FakeSolver())
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/scenarios",
            json={"site_id": SITE_ID, "name": "rt", "initial_state": "baseline"},
        )
        assert response.status_code == 201
        yield client, response.json()["scenario_id"]


def rt_workspace_with_budget(tmp_path: Path, **kwargs: Any):
    """Same as the fixture but with non-default app knobs (rate limits)."""
    app = make_rt_app(tmp_path, FakeSolver(), **kwargs)
    client = TestClient(app)
    client.__enter__()
    response = client.post(
        "/api/v1/scenarios",
        json={"site_id": SITE_ID, "name": "rt", "initial_state": "baseline"},
    )
    assert response.status_code == 201
    return client, response.json()["scenario_id"]


def rt_post(
    client: TestClient,
    workspace: str,
    operations: list[dict[str, Any]],
    actor: str = "actor_a",
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
        "source_family": "building_geometry",
        "entity_id": "b1",
        "verb": "add",
        "payload": {"height_m": 12.0},
    }
    item.update(overrides)
    return item


# ---------------------------------------------------------------------------
# Store: migration + append semantics
# ---------------------------------------------------------------------------


def test_migration_v7_creates_operation_log_epoch_and_revision_columns(
    tmp_path: Path,
) -> None:
    store = make_store(tmp_path)
    try:
        assert store.schema_version == LATEST_SCHEMA_VERSION == 9
        conn = sqlite3.connect(store.db_path)
        try:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            assert {"realtime_operations", "realtime_epochs"} <= tables
            op_columns = {
                row[1]
                for row in conn.execute("PRAGMA table_info(realtime_operations)")
            }
            assert op_columns == {
                "workspace_id",
                "server_sequence",
                "epoch_id",
                "operation_id",
                "actor_id",
                "client_sequence",
                "base_revision",
                "source_family",
                "entity_id",
                "verb",
                "payload_json",
                "received_at",
                "accepted_at",
            }
            # operation_id is UNIQUE across the whole table (global
            # idempotency): the column constraint's auto-index reports
            # origin 'u' in PRAGMA index_list, and the epoch-read index
            # exists alongside it.
            # (seq, name, unique, origin, partial)
            index_rows = list(
                conn.execute("PRAGMA index_list(realtime_operations)")
            )
            unique_indexes = [row for row in index_rows if row[3] == "u"]
            assert unique_indexes, index_rows
            for row in unique_indexes:
                columns = [
                    info[2]
                    for info in conn.execute("PRAGMA index_info(%s)" % row[1])
                ]
                assert columns == ["operation_id"], (row[1], columns)
            assert any(
                row[1] == "idx_realtime_operations_epoch" for row in index_rows
            )
            # Functional uniqueness: a second row with the same operation_id
            # is refused by the table itself, whatever the code does.
            def raw_insert(sequence: int) -> None:
                conn.execute(
                    "INSERT INTO realtime_operations (workspace_id, "
                    "server_sequence, epoch_id, operation_id, actor_id, "
                    "source_family, verb, payload_json, received_at, "
                    "accepted_at) VALUES (?, ?, 0, 'dup', 'a', "
                    "'building_geometry', 'add', '{}', 't', 't')",
                    ("w", sequence),
                )

            raw_insert(1)
            with pytest.raises(sqlite3.IntegrityError):
                raw_insert(2)
            scenario_columns = {
                row[1]
                for row in conn.execute("PRAGMA table_info(scenarios)")
            }
            assert {"fast_revision", "exact_base_revision"} <= scenario_columns
        finally:
            conn.close()
    finally:
        store.close()


def test_revision_triple_defaults_and_record_fields(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    try:
        make_workspace(store)
        record = store.get_scenario("ws_1")
        assert record is not None
        # scene_version IS workspace_revision semantically (not renamed);
        # the two new triple members default to 0 on a fresh row.
        assert record.scene_version == 0
        assert record.fast_revision == 0
        assert record.exact_base_revision == 0
    finally:
        store.close()


def test_append_assigns_contiguous_sequence_and_open_epoch_zero(
    tmp_path: Path,
) -> None:
    store = make_store(tmp_path)
    try:
        make_workspace(store)
        records = store.append_operations(
            "ws_1",
            [op_item("op_1"), op_item("op_2"), op_item("op_3")],
        )
        assert [r.server_sequence for r in records] == [1, 2, 3]
        assert all(r.epoch_id == 0 for r in records)
        assert all(r.duplicate is False for r in records)
        epoch = store.epoch_records("ws_1")[0]
        assert (epoch.epoch_id, epoch.status) == (0, "open")
        assert (epoch.first_sequence, epoch.last_sequence) == (1, 3)
    finally:
        store.close()


def test_duplicate_operation_id_same_content_returns_existing(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    try:
        make_workspace(store)
        first = store.append_operations("ws_1", [op_item("op_1")])[0]
        # Genuinely-retried operation: same operation-defining content
        # (family, entity, verb, payload) — even with a different
        # client_sequence/base_revision metadata claim.
        again = store.append_operations(
            "ws_1",
            [op_item("op_1", client_sequence=99, base_revision=7)],
        )[0]
        assert again.duplicate is True
        assert again.server_sequence == first.server_sequence
        assert again.epoch_id == first.epoch_id
        assert again.accepted_at == first.accepted_at
        # Exactly one row: never a second accepted effect.
        ops = store.operations_since("ws_1", 0)
        assert len(ops) == 1
    finally:
        store.close()


def test_duplicate_operation_id_mismatched_content_is_typed_conflict(
    tmp_path: Path,
) -> None:
    store = make_store(tmp_path)
    try:
        make_workspace(store)
        store.append_operations("ws_1", [op_item("op_1")])
        with pytest.raises(OperationFingerprintConflict):
            store.append_operations(
                "ws_1", [op_item("op_1", payload={"height_m": 99.0})]
            )
        # The batch is all-or-nothing: a batch whose LATER item conflicts
        # must not leave the earlier new item durably appended.
        with pytest.raises(OperationFingerprintConflict):
            store.append_operations(
                "ws_1",
                [op_item("op_new"), op_item("op_1", verb="delete", payload={})],
            )
        assert len(store.operations_since("ws_1", 0)) == 1
    finally:
        store.close()


def test_interleaved_multi_actor_appends_stay_contiguous(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    try:
        make_workspace(store)
        expected = 0
        for round_ in range(5):
            batch = [
                op_item(f"a_{round_}_{step}", actor_id="actor_a", client_sequence=round_)
                for step in range(2)
            ] + [
                op_item(f"b_{round_}_{step}", actor_id="actor_b", client_sequence=round_)
                for step in range(3)
            ]
            records = store.append_operations("ws_1", batch)
            expected += len(batch)
            assert [r.server_sequence for r in records] == list(
                range(expected - len(batch) + 1, expected + 1)
            )
        all_ops = store.operations_since("ws_1", 0)
        assert [r.server_sequence for r in all_ops] == list(range(1, expected + 1))
    finally:
        store.close()


def test_threaded_appends_stay_contiguous(tmp_path: Path) -> None:
    """Real cross-thread interleaving: one shared connection + RLock (the
    store.py single-conn discipline R0's queue cartography mapped) must
    still hand out a gap-free per-workspace sequence run."""
    import threading

    store = make_store(tmp_path)
    try:
        make_workspace(store)
        errors: list[BaseException] = []
        barrier = threading.Barrier(4)

        def worker(worker_id: int) -> None:
            try:
                barrier.wait()
                for step in range(10):
                    store.append_operations(
                        "ws_1",
                        [
                            op_item(
                                f"t{worker_id}_{step}",
                                actor_id=f"actor_{worker_id}",
                                client_sequence=step,
                            )
                        ],
                    )
            except BaseException as error:  # pragma: no cover - failure path
                errors.append(error)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert not errors
        all_ops = store.operations_since("ws_1", 0)
        assert [op.server_sequence for op in all_ops] == list(range(1, 41))
    finally:
        store.close()


def test_epoch_assignment_new_epoch_after_terminal_mark(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    try:
        make_workspace(store)
        store.append_operations("ws_1", [op_item("op_1"), op_item("op_2")])
        # R2 owns closure semantics; the primitive it builds on must record
        # status, closed_at, and the revision assigned at close.
        assert (
            store.mark_epoch_status("ws_1", 0, "fast_published", workspace_revision=4)
            is True
        )
        epoch = store.epoch_records("ws_1")[0]
        assert epoch.status == "fast_published"
        assert epoch.workspace_revision == 4
        assert epoch.closed_at is not None
        # Ops continue: a new epoch opens, sequences stay contiguous.
        records = store.append_operations("ws_1", [op_item("op_3")])
        assert records[0].epoch_id == 1
        assert records[0].server_sequence == 3
        epochs = {e.epoch_id: e for e in store.epoch_records("ws_1")}
        assert epochs[0].status == "fast_published"
        assert epochs[1].status == "open"
        assert (epochs[1].first_sequence, epochs[1].last_sequence) == (3, 3)
        assert store.open_epochs("ws_1") == [epochs[1]]
    finally:
        store.close()


def test_mark_epoch_status_validates_status_and_missing_row(tmp_path: Path) -> None:
    from solweig_gpu.server.store import StoreError

    store = make_store(tmp_path)
    try:
        make_workspace(store)
        with pytest.raises(StoreError):
            store.mark_epoch_status("ws_1", 0, "exploded")
        assert store.mark_epoch_status("ws_1", 99, "closed") is False
    finally:
        store.close()


def test_append_requires_known_workspace(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    try:
        with pytest.raises(ScenarioNotFound):
            store.append_operations("ws_missing", [op_item("op_1")])
    finally:
        store.close()


def test_restart_recovery_sequences_continue_nothing_autocloses(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "store.sqlite3"
    store = Store(db_path, results_root=tmp_path / "scenarios")
    make_workspace(store)
    store.append_operations("ws_1", [op_item("op_1"), op_item("op_2")])
    store.mark_epoch_status("ws_1", 0, "reducing")
    store.close()

    reopened = Store(db_path, results_root=tmp_path / "scenarios")
    try:
        # Nothing auto-closed at reopen: the reducing epoch stays exactly as
        # recorded (R2 recovery owns closure) and stays queryable.
        records = reopened.epoch_records("ws_1")
        assert [e.status for e in records] == ["reducing"]
        # A non-open epoch means the next append lazily opens epoch 1 and
        # the sequence continues from the durable high-water mark.
        ops = reopened.append_operations("ws_1", [op_item("op_3")])
        assert ops[0].server_sequence == 3
        assert ops[0].epoch_id == 1
        assert reopened.get_operation("ws_1", "op_1").server_sequence == 1
        assert reopened.operations_for_epoch("ws_1", 0)[0].operation_id == "op_1"
    finally:
        reopened.close()


def test_operations_for_epoch_and_since_catchup(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    try:
        make_workspace(store)
        store.append_operations("ws_1", [op_item(f"op_{i}") for i in range(1, 4)])
        store.mark_epoch_status("ws_1", 0, "closed")
        store.append_operations("ws_1", [op_item(f"op_{i}") for i in range(4, 7)])
        epoch0 = store.operations_for_epoch("ws_1", 0)
        epoch1 = store.operations_for_epoch("ws_1", 1)
        assert [o.operation_id for o in epoch0] == ["op_1", "op_2", "op_3"]
        assert [o.operation_id for o in epoch1] == ["op_4", "op_5", "op_6"]
        catch_up = store.operations_since("ws_1", 3)
        assert [o.server_sequence for o in catch_up] == [4, 5, 6]
        assert catch_up[0].payload == op_item("op_4")["payload"]
    finally:
        store.close()


def test_operation_log_is_append_only_not_pruned_by_retention(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    try:
        make_workspace(store)
        store.append_operations("ws_1", [op_item(f"op_{i}") for i in range(1, 6)])
        deleted = store.sweep_retention(idempotency_ttl_hours=0.0, event_tail=0)
        # The operation audit must never appear in the retention sweep (the
        # v5/v6 bug precedent: pruning a ledger the routing logic consults).
        assert "realtime_operations" not in deleted
        assert len(store.operations_since("ws_1", 0)) == 5
    finally:
        store.close()


def test_duplicate_within_one_batch_dedupes(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    try:
        make_workspace(store)
        records = store.append_operations(
            "ws_1", [op_item("op_1"), op_item("op_1")]
        )
        assert [r.duplicate for r in records] == [False, True]
        assert len(store.operations_since("ws_1", 0)) == 1
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Transport validation (realtime.operations)
# ---------------------------------------------------------------------------


def test_source_family_vocabulary_matches_frozen_types() -> None:
    assert rt.SOURCE_FAMILIES == {
        "building_geometry",
        "vegetation_geometry",
        "landcover_surface",
        "meteorological_forcing",
        "model_receptor_parameters",
        "selected_date_time",
        "output_view",
    }
    assert rt.MAX_OPERATIONS_PER_BATCH == 128


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


def test_route_accepts_operations_with_revision_counters(
    rt_workspace: tuple[TestClient, str],
) -> None:
    client, workspace = rt_workspace
    response = rt_post(
        client,
        workspace,
        [
            route_item("op_1"),
            route_item(
                "op_2", source_family="output_view", verb="select", entity_id=None
            ),
        ],
    )
    assert response.status_code == 200
    body = response.json()
    assert body["workspace_id"] == workspace
    assert [item["operation_id"] for item in body["accepted"]] == ["op_1", "op_2"]
    assert [item["server_sequence"] for item in body["accepted"]] == [1, 2]
    assert [item["epoch_id"] for item in body["accepted"]] == [0, 0]
    assert all(item["duplicate"] is False for item in body["accepted"])
    # The revision triple (workspace_revision == scene_version semantically)
    # rides the acceptance ack.
    assert body["workspace_revision"] == 0
    assert body["fast_revision"] == 0
    assert body["exact_revision"] == 0
    assert body["exact_base_revision"] == 0
    assert body["received_at"] <= body["accepted_at"]

    lookup = client.get(f"/api/v1/workspaces/{workspace}/operations/op_1")
    assert lookup.status_code == 200
    stored = lookup.json()
    assert stored["server_sequence"] == 1
    assert stored["source_family"] == "building_geometry"
    assert stored["payload"] == {"height_m": 12.0}

    epochs = client.get(f"/api/v1/workspaces/{workspace}/epochs").json()
    assert [e["epoch_id"] for e in epochs["epochs"]] == [0]
    assert epochs["epochs"][0]["status"] == "open"


def test_route_idempotent_retry_and_mismatched_409(
    rt_workspace: tuple[TestClient, str],
) -> None:
    client, workspace = rt_workspace
    first = rt_post(client, workspace, [route_item("op_1")]).json()
    retry = rt_post(client, workspace, [route_item("op_1")])
    assert retry.status_code == 200
    body = retry.json()
    assert body["accepted"][0]["duplicate"] is True
    assert body["accepted"][0]["server_sequence"] == first["accepted"][0]["server_sequence"]

    conflict = rt_post(
        client, workspace, [route_item("op_1", payload={"height_m": 3.0})]
    )
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "operation_id_reused"


def test_route_unknown_family_and_verb_typed_400(
    rt_workspace: tuple[TestClient, str],
) -> None:
    client, workspace = rt_workspace
    unknown_family = rt_post(
        client, workspace, [route_item("op_1", source_family="dem")]
    )
    assert unknown_family.status_code == 400
    assert unknown_family.json()["error"]["code"] == "unknown_source_family"

    unknown_verb = rt_post(client, workspace, [route_item("op_2", verb="explode")])
    assert unknown_verb.status_code == 400
    assert unknown_verb.json()["error"]["code"] == "unknown_operation_verb"

    empty_payload = rt_post(client, workspace, [route_item("op_3", payload={})])
    assert empty_payload.status_code == 400
    assert empty_payload.json()["error"]["code"] == "invalid_operation_payload"


def test_route_batch_cap_and_refusal_shapes(
    rt_workspace: tuple[TestClient, str],
) -> None:
    client, workspace = rt_workspace
    at_cap = rt_post(
        client,
        workspace,
        [route_item(f"cap_{i}") for i in range(rt.MAX_OPERATIONS_PER_BATCH)],
    )
    assert at_cap.status_code == 200
    assert len(at_cap.json()["accepted"]) == rt.MAX_OPERATIONS_PER_BATCH

    over_cap = rt_post(
        client,
        workspace,
        [
            route_item(f"over_{i}")
            for i in range(rt.MAX_OPERATIONS_PER_BATCH + 1)
        ],
    )
    assert over_cap.status_code == 400
    assert over_cap.json()["error"]["code"] == "invalid_request"

    empty = client.post(
        f"/api/v1/workspaces/{workspace}/operations",
        json={"actor_id": "actor_a", "operations": []},
    )
    assert empty.status_code == 400

    unknown_ws = rt_post(client, "scn_missing", [route_item("op_x")])
    assert unknown_ws.status_code == 404
    assert unknown_ws.json()["error"]["code"] == "scenario_not_found"

    missing_op = client.get(f"/api/v1/workspaces/{workspace}/operations/nope")
    assert missing_op.status_code == 404
    assert missing_op.json()["error"]["code"] == "operation_not_found"


def test_route_advisory_base_revision_never_rejects(
    rt_workspace: tuple[TestClient, str],
) -> None:
    client, workspace = rt_workspace
    # Wildly stale and future base revisions are both ACCEPTED: multi-writer
    # semantics never gate on base_revision equality.
    stale = rt_post(client, workspace, [route_item("stale", base_revision=987)])
    assert stale.status_code == 200
    ahead = rt_post(client, workspace, [route_item("ahead", base_revision=424242)])
    assert ahead.status_code == 200
    none_base = rt_post(client, workspace, [route_item("nb", base_revision=None)])
    assert none_base.status_code == 200
    stored = client.get(
        f"/api/v1/workspaces/{workspace}/operations/stale"
    ).json()
    assert stored["base_revision"] == 987


def test_route_reuses_rate_limit_guard(tmp_path: Path) -> None:
    client, workspace = rt_workspace_with_budget(tmp_path, edits_per_minute=2)
    try:
        assert rt_post(client, workspace, [route_item("op_1")]).status_code == 200
        assert rt_post(client, workspace, [route_item("op_2")]).status_code == 200
        limited = rt_post(client, workspace, [route_item("op_3")])
        assert limited.status_code == 429
        assert limited.json()["error"]["code"] == "rate_limited"
    finally:
        client.close()
