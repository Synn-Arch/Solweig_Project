# SPDX-License-Identifier: GPL-3.0-only
"""R8 chaos invariants: the cheap CI fences for the load/chaos/soak matrix.

The full matrix (admission flood at concurrency, ``os._exit`` crash seams
inside the real commit paths, one-hour soak with RSS slope) lives in
``docs/incremental_design_tool/realtime_collaboration/design/proofs/
r8_load_chaos_soak.py`` -- it needs real subprocesses and minutes of wall
time, so it does not run in CI. These tests pin the same INVARIANTS at
in-process scale, cheap enough for every push:

* **reject BEFORE acceptance** -- a batch past the admission envelope is
  refused with a typed 429 and NONE of its operations ever reach the
  durable log (no phantom accepts); the app keeps serving (R8 chaos case
  ``accept_queue_overflow``).
* **duplicates + stale bases fold deterministically** -- the same op
  sequence driven into two workspaces produces byte-identical final
  canonical state; duplicate replays add no durable rows; a reused
  operation id with DIFFERENT content is a typed 409 (R8 chaos case
  ``duplicate_stale_determinism``).
* **the epoch plane stays deadline-bounded** -- under the real scheduler
  thread, an accepted operation reaches a canonical revision within a
  bounded multiple of the coalescing window (the R8 load decomposition
  showed epoch WAIT dominates visibility; reduce/publish are sub-ms).
* **invalid scientific content is a typed refusal that SETTLES the lane**
  -- a landcover paint of a class outside the adapter vocabulary passes
  transport (it is structurally valid), then the exact lane refuses it
  typed, consumes the epoch, and advances instead of wedging (discovered
  by the R8 smoke run; the same loss model as unfoldable deletes).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from solweig_gpu.server.realtime.admission import AdmissionRejected
from solweig_gpu.server.store import Store

from tests.test_realtime_operations import (
    make_rt_app,
    rt_post,
    route_item,
)
from tests.test_realtime_r2_exact_lane import SseSession
from tests.test_server_api import SITE_ID, FakeSolver
from tests.test_server_universal_edits import family_site


def _view_item(operation_id: str, **overrides: Any) -> dict[str, Any]:
    item = route_item(
        operation_id,
        source_family="output_view",
        verb="select",
        entity_id=None,
        payload={"values": {"timestep": 0, "layer": "utci"}},
    )
    item.update(overrides)
    return item


def _make_client(tmp_path: Path) -> TestClient:
    app = make_rt_app(tmp_path, FakeSolver())
    client = TestClient(app)
    client.__enter__()
    return client


def _workspace(client: TestClient, name: str = "r8") -> str:
    response = client.post(
        "/api/v1/scenarios",
        json={"site_id": SITE_ID, "name": name, "initial_state": "baseline"},
    )
    assert response.status_code == 201, response.text
    return response.json()["scenario_id"]


# ---------------------------------------------------------------------------
# Invariant: reject BEFORE acceptance (admission overflow, zero phantoms)
# ---------------------------------------------------------------------------


def test_admission_overflow_rejects_before_durable_append(
    tmp_path: Path,
) -> None:
    client = _make_client(tmp_path)
    try:
        workspace = _workspace(client)
        store: Store = client.app.state.context.store

        # One maximal batch per sliding second is the envelope's own
        # contract (max_operations_per_epoch == max_accepted_per_second
        # == 128); the SECOND maximal batch inside the same second is the
        # overflow, refused BEFORE the durable append.
        first = [_view_item(f"ok_{i}") for i in range(128)]
        response = rt_post(client, workspace, first)
        assert response.status_code == 200, response.text
        assert len(response.json()["accepted"]) == 128

        overflow = [_view_item(f"over_{i}") for i in range(128)]
        response = rt_post(client, workspace, overflow)
        assert response.status_code == 429, response.text
        body = response.json()
        assert body["error"]["code"] == "fast_lane_overloaded"
        # the envelope details merge flat into the error object
        assert "max_accepted_operations_per_second" in body["error"]

        # Zero phantoms: nothing the gate rejected is in the durable log.
        durable = {
            op.operation_id for op in store.operations_since(workspace, 0)
        }
        assert durable == {item["operation_id"] for item in first}, (
            "a rejected operation reached the durable log -- "
            "reject-before-acceptance violated"
        )

        # The app is still live and still admits after the sliding second
        # empties (nothing stays wedged).
        deadline = time.monotonic() + 5.0
        admitted = False
        while time.monotonic() < deadline:
            response = rt_post(client, workspace, [_view_item("after_window")])
            if response.status_code == 200:
                admitted = True
                break
            time.sleep(0.25)
        assert admitted, "admission never recovered after the sliding second"
        assert client.get("/health/ready").status_code == 200
    finally:
        # __exit__ (not .close()): this starlette's close() does NOT run
        # the lifespan shutdown, so the scheduler thread (and its store)
        # would outlive the test and trip later suites' no-leak fences.
        client.__exit__(None, None, None)


def test_admission_rejection_raises_typed_before_store_touch(
    tmp_path: Path,
) -> None:
    """The gate raises AdmissionRejected (429) from ``check`` itself."""
    from solweig_gpu.server.realtime.admission import AdmissionEnvelope

    envelope = AdmissionEnvelope()
    assert envelope.max_operations_per_epoch == 128
    with pytest.raises(AdmissionRejected):
        # over-batch directly against the gate (no app needed)
        from solweig_gpu.server.realtime.admission import FastAdmission

        FastAdmission(envelope=envelope).check(
            "ws", [{"source_family": "output_view"}] * 129,
            received_at="2026-09-05T00:00:00.000Z",
        )


# ---------------------------------------------------------------------------
# Invariant: duplicates + stale bases fold deterministically
# ---------------------------------------------------------------------------

#: A mixed deterministic sequence: view/veg/met/landcover + stale bases.
_SEQUENCE: tuple[tuple[str, dict[str, Any]], ...] = tuple(
    (
        f"seq_{index}",
        route_item(
            f"seq_{index}",
            source_family=family,
            verb=verb,
            entity_id=entity,
            payload=payload,
            base_revision=(index % 7),  # deliberately stale/divergent bases
            client_sequence=index,
        ),
    )
    for index, (family, verb, entity, payload) in enumerate(
        [
            ("output_view", "select", None, {"values": {"timestep": 0}}),
            (
                "vegetation_geometry",
                "add",
                "tree-d1",
                {"values": {"x_m": 1020.0, "y_m": 1980.0, "height_m": 9.0,
                            "canopy_radius_m": 3.0, "trunk_ratio": 0.25}},
            ),
            ("meteorological_forcing", "set", None,
             {"values": {"air_temperature": 31.0}, "time_index": 1}),
            (
                "landcover_surface",
                "paint",
                None,
                {"window": {"row_start": 10, "row_stop": 14,
                            "col_start": 10, "col_stop": 14}, "class": 5},
            ),
            (
                "vegetation_geometry",
                "replace",
                "tree-d1",
                {"values": {"height_m": 12.0, "canopy_radius_m": 4.0}},
            ),
            ("output_view", "select", None,
             {"values": {"timestep": 2, "layer": "tmrt"}}),
            (
                "vegetation_geometry",
                "delete",
                "tree-d1",
                {"values": {}},
            ),
            (
                "vegetation_geometry",
                "add",
                "tree-d2",
                {"values": {"x_m": 1040.0, "y_m": 1960.0, "height_m": 7.0,
                            "canopy_radius_m": 2.5, "trunk_ratio": 0.25}},
            ),
            ("meteorological_forcing", "set", None,
             {"values": {"air_temperature": 27.5}, "time_index": 2}),
            (
                "landcover_surface",
                "paint",
                None,
                {"window": {"row_start": 40, "row_stop": 44,
                            "col_start": 40, "col_stop": 44}, "class": 2},
            ),
        ]
    )
)


def _drive_sequence(client: TestClient, workspace: str, prefix: str) -> int:
    """Post the deterministic sequence; return durable unique-op count."""
    acked = 0
    for _, item in _SEQUENCE:
        body = dict(item)
        body["operation_id"] = f"{prefix}-{item['operation_id']}"
        response = rt_post(client, workspace, [body])
        assert response.status_code == 200, response.text
        acked += 1
    return acked


def _await_fully_folded(client: TestClient, workspace: str, n_ops: int) -> None:
    """Wait until every accepted op sits in a REVISED (non-open) epoch.

    Comparing canonical state before all epochs folded would race the
    close loop; the fold-complete condition is: n_ops durable rows, no
    open epoch, and every epoch carrying ops has its revision.
    """
    store: Store = client.app.state.context.store
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        ops = store.operations_since(workspace, 0)
        epochs = store.epoch_records(workspace)
        folded = (
            len(ops) >= n_ops
            and all(epoch.status != "open" for epoch in epochs)
            and all(
                epoch.workspace_revision is not None
                for epoch in epochs
                if (epoch.first_sequence is not None)
            )
        )
        if folded:
            return
        time.sleep(0.05)
    pytest.fail(
        f"workspace {workspace} never folded {n_ops} ops into canonical state"
    )


def test_duplicate_stale_sequence_folds_identically_across_workspaces(
    tmp_path: Path,
) -> None:
    client = _make_client(tmp_path)
    try:
        first = _workspace(client, "r8_a")
        second = _workspace(client, "r8_b")
        store: Store = client.app.state.context.store

        acked = _drive_sequence(client, first, "a")

        # duplicate replay: same content => 200 + duplicate=True, NO new row
        replay = dict(_SEQUENCE[1][1])
        replay["operation_id"] = f"a-{_SEQUENCE[1][1]['operation_id']}"
        response = rt_post(client, first, [replay])
        assert response.status_code == 200, response.text
        assert response.json()["accepted"][0]["duplicate"] is True

        # reused id with DIFFERENT content: typed 409, no durable effect
        mutated = dict(replay)
        mutated["payload"] = {"values": {"height_m": 99.0}}
        conflict = rt_post(client, first, [mutated])
        assert conflict.status_code == 409, conflict.text
        assert conflict.json()["error"]["code"] == "operation_id_reused"

        # the SAME sequence into the second workspace (identical content;
        # only the id prefix differs and canonical state carries no ids)
        _drive_sequence(client, second, "b")

        _await_fully_folded(client, first, acked)
        _await_fully_folded(client, second, acked)

        # exactly-one-revision per op: duplicates/conflicts added no rows
        assert len(store.operations_since(first, 0)) == acked
        assert len(store.operations_since(second, 0)) == acked

        # deterministic fold: byte-identical final canonical state
        left = store.latest_canonical_state(first)
        right = store.latest_canonical_state(second)
        assert left is not None and right is not None
        assert left.families == right.families, (
            "identical op sequences produced different canonical state "
            f"across workspaces\nleft:  {left.families}\nright: {right.families}"
        )
    finally:
        # __exit__ (not .close()): this starlette's close() does NOT run
        # the lifespan shutdown, so the scheduler thread (and its store)
        # would outlive the test and trip later suites' no-leak fences.
        client.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# Invariant: the epoch plane stays deadline-bounded (real scheduler thread)
# ---------------------------------------------------------------------------


def test_accepted_op_reaches_canonical_revision_within_bounded_windows(
    tmp_path: Path,
) -> None:
    """The R8 decomposition: visibility latency IS epoch wait (reduce and
    publish are sub-millisecond), so the fence here is that the epoch
    plane actually closes -- an accepted op reaches a canonical revision
    within a bound far above 2x window + close work, under the REAL
    scheduler thread. Catches a wedged/starved close loop, not jitter."""
    client = _make_client(tmp_path)
    try:
        workspace = _workspace(client)
        bound_s = 2.0  # >> 2x20ms window + close work; a wedged lane blows it

        for attempt in range(5):
            ack = rt_post(client, workspace, [_view_item(f"latency_{attempt}")])
            assert ack.status_code == 200, ack.text
            started = time.monotonic()
            seen = False
            while time.monotonic() - started < bound_s:
                body = client.get(
                    f"/api/v1/workspaces/{workspace}/operations"
                ).json()
                if body["workspace_revision"] >= attempt + 1:
                    seen = True
                    break
                time.sleep(0.005)
            assert seen, (
                f"op latency_{attempt} never reached a canonical revision "
                f"within {bound_s}s (revision stuck below {attempt + 1})"
            )
    finally:
        # __exit__ (not .close()): this starlette's close() does NOT run
        # the lifespan shutdown, so the scheduler thread (and its store)
        # would outlive the test and trip later suites' no-leak fences.
        client.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# Invariant: invalid scientific content refuses typed and SETTLES the lane
# (discovered by the R8 smoke run: landcover classes 3/4 are outside the
# adapter vocabulary; transport admits them, the exact lane refuses them)
# ---------------------------------------------------------------------------


@pytest.mark.scientific
def test_invalid_landcover_class_typed_refusal_settles_exact_lane(
    tmp_path: Path, family_site
) -> None:
    from tests.test_server_universal_edits import _make_client, _scenario_ready

    with _make_client(tmp_path, family_site) as client:
        workspace = _scenario_ready(client)
        # Audience (routing policy R1): the exact lane's chase minting is
        # subscriber-gated, and this test expects the lane to run — hold
        # the SSE subscription a real studio client opens on connect.
        watch = SseSession(client, workspace)
        store: Store = client.app.state.context.store
        posted = rt_post(
            client,
            workspace,
            [
                route_item(
                    "r8_invalid_paint",
                    source_family="landcover_surface",
                    verb="paint",
                    entity_id=None,
                    payload={
                        "window": {
                            "row_start": 30,
                            "row_stop": 34,
                            "col_start": 30,
                            "col_stop": 34,
                        },
                        "class": 3,  # outside the adapter vocabulary
                    },
                )
            ],
        )
        assert posted.status_code == 200, posted.text

        deadline = time.monotonic() + 120.0
        outcome: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            epochs = store.epoch_records(workspace)
            jobs = [
                row
                for row in _job_rows(store, workspace)
                if row["request"].get("exact_lane")
            ]
            if jobs and jobs[-1]["status"] in ("failed", "complete"):
                outcome = {
                    "epochs": [epoch.status for epoch in epochs],
                    "job": jobs[-1],
                }
                break
            time.sleep(0.25)
        assert outcome is not None, "exact lane never settled the span"

        job = outcome["job"]
        assert job["status"] == "failed"
        assert job["error"]["code"] == "edit_rejected"
        assert "land-cover" in job["error"]["message"]
        # SETTLED: every epoch consumed; the lane advances, never wedges.
        assert outcome["epochs"], "no epochs recorded"
        assert all(
            status == "exact_targeted" for status in outcome["epochs"]
        ), outcome["epochs"]

        # ...and the lane is not churn-blocked: a FOLLOW-UP valid op still
        # folds, settles (exact_revision catches the canonical revision),
        # and consumes its epoch.
        valid = rt_post(
            client,
            workspace,
            [
                _view_item("r8_valid_after_refusal", client_sequence=2)
            ],
        )
        assert valid.status_code == 200, valid.text
        deadline = time.monotonic() + 120.0
        while time.monotonic() < deadline:
            body = client.get(
                f"/api/v1/workspaces/{workspace}/operations"
            ).json()
            if (
                body["workspace_revision"] >= 2
                and body["exact_revision"] == body["workspace_revision"]
            ):
                break
            time.sleep(0.25)
        else:
            pytest.fail("lane wedged after the typed refusal")


def _job_rows(store: Store, scenario_id: str) -> list[dict[str, Any]]:
    import json
    import sqlite3

    conn = sqlite3.connect(store.db_path)
    try:
        rows = conn.execute(
            "SELECT job_id, status, request_json, error_json FROM jobs "
            "WHERE scenario_id = ? ORDER BY rowid",
            (scenario_id,),
        ).fetchall()
    finally:
        conn.close()
    return [
        {
            "job_id": row[0],
            "status": row[1],
            "request": json.loads(row[2]) if row[2] else {},
            "error": json.loads(row[3]) if row[3] else None,
        }
        for row in rows
    ]
