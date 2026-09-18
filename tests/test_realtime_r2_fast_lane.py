# SPDX-License-Identifier: GPL-3.0-only
"""R2b fast lane: deadline-isolated scheduler thread + admission control
(failing-first).

Pins the mission contracts (``docs/incremental_design_tool/
realtime_collaboration/epoch_scheduler.md`` "Fast queue" + "Backpressure" +
"Deadline budgeting"; ``service_level_contract.md`` "Admission envelope" +
"Degradation ladder"; ``realtime_contract.yaml`` queue model
``fast: earliest_deadline_with_fairness``):

* **Deadline isolation** — the fast lane runs on its own thread and plans
  from the epoch-final canonical state. It never waits behind exact work
  and never blocks the epoch close pipeline (the epochs.py hook is
  enqueue-and-return).
* **Admission before acceptance** — predicted fast cost + reserved capacity
  + the 1,000 ms envelope (ingest 100 / epoch wait <=100 / reduce+plan 100
  / compute 550 / encode+publish 150). Outside the envelope the operation
  batch is REJECTED (429 workspace burst / 503 server-wide) with retry
  metadata BEFORE the durable append — once accepted, never dropped.
* **Result classes v1** — ``fast_exact`` (view/cached switch),
  ``visual_pending`` (default when no fast kernel can meet the deadline);
  ``fast_qualified`` stays reserved (R3/R7) and must NEVER appear on the
  wire this wave. The class rides the ``fast_revision`` SSE payload.
* **Revision model** — the lane advances ``scenarios.fast_revision``
  monotonically, never past ``workspace_revision``, keeping
  ``exact_revision <= fast_revision <= workspace_revision``.
* **Exact-lane non-disturbance** — the fast lane advances the epoch row at
  most ``reducing -> fast_planned``; it does NOT mark ``fast_published``
  (terminal semantics are shared with the exact lane's consumption math in
  r2a jobs.py — see the r2b handoff vocabulary-gap note) and an epoch the
  fast lane touched remains consumable by the exact lane.

All written failing-first against base 03d58e2 (r2a merged, no r2b
behavior).
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from solweig_gpu.server.app import create_app
from solweig_gpu.server.realtime.admission import (
    DEFAULT_FAST_DEADLINE_BUDGET_MS,
    STAGE_COMPUTE_MS,
    STAGE_ENCODE_PUBLISH_MS,
    STAGE_EPOCH_WAIT_MS,
    STAGE_INGEST_MS,
    STAGE_REDUCE_PLAN_MS,
    AdmissionEnvelope,
    AdmissionRejected,
    FastAdmission,
    predict_epoch_compute_ms,
)
from solweig_gpu.server.realtime.broadcast import BroadcastHub
from solweig_gpu.server.realtime.epochs import EpochScheduler
from solweig_gpu.server.realtime.scheduler import (
    DEFAULT_FAST_DEADLINE_MS,
    FAST_LANE_THREAD_NAME,
    FastLaneScheduler,
    ViewCacheKernel,
)
from solweig_gpu.server.realtime.telemetry import (
    METRIC_FAST_COMPUTE_MS,
    METRIC_RESULT_CLASS_TOTAL,
    TelemetryRegistry,
)
from solweig_gpu.server.store import Store

from tests.test_realtime_epochs import (
    WINDOW_S,
    WS,
    FakeClock,
    SSEConnection,
    SSEReader,
    make_store,
    make_workspace,
    op_item,
    rt_post,
    route_item,
)
from tests.test_server_api import SITE_ID, FakeSolver, make_site_cache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeMonotonic:
    """Controllable monotonic clock (durations/telemetry assertions)."""

    def __init__(self, start_ns: int = 1_000_000_000) -> None:
        self.now_ns = start_ns

    def __call__(self) -> int:
        return self.now_ns

    def advance_ms(self, ms: float) -> None:
        self.now_ns += int(ms * 1_000_000)


class RecordingHub:
    """BroadcastHub stand-in recording fast/canonical events."""

    def __init__(self) -> None:
        self.fast: list[tuple[str, dict[str, Any]]] = []
        self.canonical: list[tuple[str, dict[str, Any]]] = []

    def broadcast_fast(self, workspace_id: str, event: dict[str, Any]) -> None:
        self.fast.append((workspace_id, dict(event)))

    def broadcast_canonical(self, workspace_id: str, event: dict[str, Any]) -> None:
        self.canonical.append((workspace_id, dict(event)))


class StubKernel:
    """Injected fast kernel with a declared predicted cost + fixed class."""

    name = "stub"

    def __init__(self, *, predicted_ms: float = 10.0, result_class: str = "fast_exact"):
        self.predicted_ms = float(predicted_ms)
        self.result_class = result_class
        self.predicted_calls: list[int] = []
        self.run_calls: list[int] = []

    def predicted_cost_ms(self, plan: Any) -> float:
        self.predicted_calls.append(plan.workspace_revision)
        return self.predicted_ms

    def run(self, plan: Any) -> Any:
        self.run_calls.append(plan.workspace_revision)
        from solweig_gpu.server.realtime.scheduler import FastKernelResult

        return FastKernelResult(
            result_class=self.result_class,  # type: ignore[arg-type]
            payload={"kernel": self.name},
        )


def make_fast_lane(
    store: Store,
    hub: RecordingHub | None = None,
    clock: FakeClock | None = None,
    mono: FakeMonotonic | None = None,
    *,
    kernel: Any = None,
    deadline_ms: float = 1000.0,
    **kwargs: Any,
) -> tuple[FastLaneScheduler, RecordingHub, FakeClock, FakeMonotonic]:
    hub = hub if hub is not None else RecordingHub()
    clock = clock if clock is not None else FakeClock()
    mono = mono if mono is not None else FakeMonotonic()
    lane = FastLaneScheduler(
        store,
        hub=hub,
        telemetry=TelemetryRegistry(),
        kernel=kernel,
        deadline_ms=deadline_ms,
        utc_now=clock,
        monotonic_ns=mono,
        **kwargs,
    )
    return lane, hub, clock, mono


def close_epoch_with_ops(
    store: Store,
    scheduler: EpochScheduler,
    clock: FakeClock,
    operations: list[dict[str, Any]],
    workspace: str = WS,
) -> int:
    """Append ops, close the epoch, return the assigned revision."""
    store.append_operations(workspace, operations)
    clock.advance(WINDOW_S)
    scheduler.run_once()
    epochs = store.epoch_records(workspace)
    assigned = [e.workspace_revision for e in epochs if e.workspace_revision]
    assert assigned, "test setup: the epoch must commit"
    return int(assigned[-1])


def publish_due(lane: FastLaneScheduler, *, passes: int = 2):
    """Run lane passes until due work publishes.

    Pass 1 is the scan-discovery grace (review F1): scan-discovered
    revisions defer one pass so the fast frame never precedes the
    canonical frame. Hook-woken work publishes on pass 1; this helper
    covers both without racing.
    """
    published: list[tuple[str, int]] = []
    for _ in range(passes):
        published = lane.run_once()
        if published:
            return published
    return published


def scenario_of(store: Store, workspace: str = WS):
    return store.require_scenario(workspace)


def make_rt_app(tmp_path: Path, **kwargs: Any):
    """App with PRIVATE-telemetry realtime plane (test isolation rule)."""

    def scheduler_factory(store, hub: BroadcastHub):
        return EpochScheduler(store, hub=hub, telemetry=TelemetryRegistry())

    def fast_lane_factory(store, hub: BroadcastHub):
        return FastLaneScheduler(store, hub=hub, telemetry=TelemetryRegistry())

    kwargs.setdefault("requests_per_minute_per_ip", None)
    kwargs.setdefault("edits_per_minute", None)
    kwargs.setdefault("start_epoch_scheduler", False)
    kwargs.setdefault("start_fast_lane", False)
    return create_app(
        state_root=tmp_path / "state",
        sites={SITE_ID: {"cache_dir": make_site_cache(tmp_path)}},
        solver_factory=lambda context: FakeSolver(),
        coalescing_window_ms=20.0,
        start_worker=True,
        epoch_scheduler_factory=scheduler_factory,
        fast_lane_factory=fast_lane_factory,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Contract pins
# ---------------------------------------------------------------------------


def test_fast_lane_constants_pin_the_contract() -> None:
    # epoch_scheduler.md deadline budgeting table (1000 ms target).
    assert STAGE_INGEST_MS == 100.0
    assert STAGE_EPOCH_WAIT_MS <= 100.0
    assert STAGE_REDUCE_PLAN_MS == 100.0
    assert STAGE_COMPUTE_MS == 550.0
    assert STAGE_ENCODE_PUBLISH_MS == 150.0
    assert (
        STAGE_INGEST_MS
        + STAGE_EPOCH_WAIT_MS
        + STAGE_REDUCE_PLAN_MS
        + STAGE_COMPUTE_MS
        + STAGE_ENCODE_PUBLISH_MS
        == DEFAULT_FAST_DEADLINE_BUDGET_MS
        == 1000.0
    )
    assert DEFAULT_FAST_DEADLINE_MS == 1000.0
    assert FAST_LANE_THREAD_NAME == "solweig-fast-lane"


def test_envelope_defaults_match_contract_required_fields() -> None:
    # realtime_contract.yaml admission_envelope_required_fields.
    envelope = AdmissionEnvelope()
    for field in (
        "max_operations_per_epoch",
        "max_accepted_operations_per_second",
        "max_changed_cells_per_epoch",
        "max_geometry_objects_touched_per_epoch",
        "max_payload_bytes_per_operation",
        "max_active_workspaces",
        "fast_compute_budget_ms_per_epoch",
    ):
        assert hasattr(envelope, field), field
    assert envelope.max_operations_per_epoch >= 1


# ---------------------------------------------------------------------------
# Admission (unit)
# ---------------------------------------------------------------------------


def test_admission_accepts_batch_inside_envelope(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    make_workspace(store)
    clock = FakeClock()
    admission = FastAdmission(utc_now=clock)
    received_at = clock()
    clock.advance(0.005)  # 5 ms of ingest
    # A small mixed batch is comfortably inside the envelope.
    admission.check(
        WS,
        [
            op_item("adm_ok_1", source_family="output_view", verb="select",
                    entity_id=None, payload={"view": "utci"}),
            op_item("adm_ok_2"),
        ],
        received_at=received_at,
    )


def test_admission_rejects_predicted_overrun_before_acceptance() -> None:
    clock = FakeClock()
    admission = FastAdmission(
        envelope=AdmissionEnvelope(fast_compute_budget_ms_per_epoch=200.0),
        utc_now=clock,
    )
    received_at = clock()
    clock.advance(0.005)
    # Building operations book the heaviest family base (220 ms) plus a
    # per-op margin — over a 200 ms commissioned compute budget.
    heavy = {
        "source_family": "building_geometry",
        "entity_id": "b1",
        "payload": {"height_m": 12.0},
    }
    with pytest.raises(AdmissionRejected) as excinfo:
        admission.check(
            WS,
            [
                op_item("adm_big_1", **heavy),
                op_item("adm_big_2", **heavy),
                op_item("adm_big_3", **heavy),
            ],
            received_at=received_at,
        )
    error = excinfo.value
    assert error.status == 429
    assert error.code == "fast_lane_overloaded"
    details = error.details
    assert details["retry_after_ms"] > 0
    assert details["deadline_budget_ms"] == 1000.0
    assert details["predicted_fast_ms"] > details["compute_available_ms"]
    assert details["advice"], "backpressure must tell the client what to reduce"


def test_admission_counts_elapsed_ingest_against_the_envelope() -> None:
    clock = FakeClock()
    admission = FastAdmission(utc_now=clock)
    received_at = clock()
    # 700 ms already spent in ingest: the fixed stages (epoch wait + reduce
    # + publish = 350 ms) alone exceed what remains of the 1,000 ms budget,
    # so ANY operation with a nonzero predicted cost is outside envelope.
    clock.advance(0.700)
    with pytest.raises(AdmissionRejected) as excinfo:
        admission.check(
            WS, [op_item("adm_late_1")], received_at=received_at
        )
    assert excinfo.value.status == 429
    assert excinfo.value.details["elapsed_ingest_ms"] >= 400.0


def test_admission_rejects_batches_over_max_operations_per_epoch() -> None:
    clock = FakeClock()
    admission = FastAdmission(
        envelope=AdmissionEnvelope(max_operations_per_epoch=4), utc_now=clock
    )
    received_at = clock()
    with pytest.raises(AdmissionRejected) as excinfo:
        admission.check(
            WS,
            [op_item(f"adm_cnt_{i}") for i in range(5)],
            received_at=received_at,
        )
    assert excinfo.value.status == 429
    assert excinfo.value.details["max_operations_per_epoch"] == 4


def test_admission_rate_budget_rejects_bursts_with_retry_hint() -> None:
    clock = FakeClock()
    admission = FastAdmission(
        envelope=AdmissionEnvelope(max_accepted_operations_per_second=3.0),
        utc_now=clock,
    )
    received_at = clock()
    admission.check(WS, [op_item("adm_r_1"), op_item("adm_r_2")], received_at=received_at)
    with pytest.raises(AdmissionRejected) as excinfo:
        admission.check(
            WS, [op_item("adm_r_3"), op_item("adm_r_4")], received_at=received_at
        )
    error = excinfo.value
    assert error.status == 429
    assert error.details["retry_after_ms"] > 0


def test_admission_rate_window_frees_with_time() -> None:
    clock = FakeClock()
    admission = FastAdmission(
        envelope=AdmissionEnvelope(max_accepted_operations_per_second=3.0),
        utc_now=clock,
    )
    received_at = clock()
    admission.check(WS, [op_item("adm_w_1"), op_item("adm_w_2")], received_at=received_at)
    with pytest.raises(AdmissionRejected):
        admission.check(WS, [op_item("adm_w_3"), op_item("adm_w_4")], received_at=received_at)
    clock.advance(1.5)  # the 1 s window drained
    admission.check(WS, [op_item("adm_w_5"), op_item("adm_w_6")], received_at=clock())


def test_admission_geometry_objects_touched_bound() -> None:
    clock = FakeClock()
    admission = FastAdmission(
        envelope=AdmissionEnvelope(
            max_geometry_objects_touched_per_epoch=2,
            # keep the other budgets permissive so this is the binding one
            max_accepted_operations_per_second=10_000.0,
        ),
        utc_now=clock,
    )
    received_at = clock()
    with pytest.raises(AdmissionRejected) as excinfo:
        admission.check(
            WS,
            [
                op_item("adm_g_1"),
                op_item("adm_g_2", source_family="building_geometry"),
                op_item("adm_g_3", source_family="building_geometry"),
            ],
            received_at=received_at,
        )
    assert excinfo.value.details["max_geometry_objects_touched_per_epoch"] == 2


def test_admission_changed_cells_bound() -> None:
    clock = FakeClock()
    admission = FastAdmission(
        envelope=AdmissionEnvelope(
            max_changed_cells_per_epoch=100,
            max_accepted_operations_per_second=10_000.0,
        ),
        utc_now=clock,
    )
    received_at = clock()
    paint = op_item(
        "adm_paint_1",
        source_family="landcover_surface",
        verb="paint",
        entity_id=None,
        payload={"window": {"row_start": 0, "row_end": 10, "col_start": 0, "col_end": 9}},
    )
    with pytest.raises(AdmissionRejected) as excinfo:
        admission.check(WS, [paint], received_at=received_at)
    assert excinfo.value.details["max_changed_cells_per_epoch"] == 100


def test_admission_payload_size_bound() -> None:
    clock = FakeClock()
    admission = FastAdmission(
        envelope=AdmissionEnvelope(
            max_payload_bytes_per_operation=64,
            max_accepted_operations_per_second=10_000.0,
        ),
        utc_now=clock,
    )
    received_at = clock()
    big = op_item(
        "adm_bytes_1",
        payload={"blob": "x" * 512},
    )
    with pytest.raises(AdmissionRejected) as excinfo:
        admission.check(WS, [big], received_at=received_at)
    assert excinfo.value.details["max_payload_bytes_per_operation"] == 64


def test_admission_server_wide_saturation_is_503() -> None:
    clock = FakeClock()

    class Saturated:
        def reserved_ms(self, workspace_id: str) -> float:
            return 0.0

        def saturated(self) -> bool:
            return True

    admission = FastAdmission(capacity=Saturated(), utc_now=clock)
    with pytest.raises(AdmissionRejected) as excinfo:
        admission.check(WS, [op_item("adm_sat_1")], received_at=clock())
    error = excinfo.value
    assert error.status == 503
    assert error.code == "fast_lane_unavailable"
    assert error.details["retry_after_ms"] > 0


def test_admission_active_workspace_bound_is_503() -> None:
    clock = FakeClock()
    admission = FastAdmission(
        envelope=AdmissionEnvelope(
            max_active_workspaces=1, max_accepted_operations_per_second=10_000.0
        ),
        utc_now=clock,
    )
    admission.check(WS, [op_item("adm_ws_1")], received_at=clock())
    with pytest.raises(AdmissionRejected) as excinfo:
        admission.check(
            "ws_2", [op_item("adm_ws_2")], received_at=clock()
        )
    assert excinfo.value.status == 503


class StaticCapacity:
    """Capacity reporter with a fixed reserved cost."""

    def __init__(self, reserved: float) -> None:
        self.reserved = reserved

    def reserved_ms(self, workspace_id: str) -> float:
        return self.reserved

    def saturated(self) -> bool:
        return False


def test_admission_reserved_capacity_reduces_headroom() -> None:
    clock = FakeClock()
    # 500 ms already reserved ahead of this batch: one vegetation op
    # (180 ms booked) no longer fits the 550 ms compute budget; a cheap
    # view op still does.
    admission = FastAdmission(capacity=StaticCapacity(500.0), utc_now=clock)
    with pytest.raises(AdmissionRejected):
        admission.check(WS, [op_item("adm_res_1")], received_at=clock())
    admission.check(
        WS,
        [
            op_item(
                "adm_res_2",
                source_family="output_view",
                verb="select",
                entity_id=None,
                payload={"view": "utci"},
            )
        ],
        received_at=clock(),
    )


def test_admission_rejections_are_counted() -> None:
    clock = FakeClock()
    admission = FastAdmission(
        envelope=AdmissionEnvelope(max_operations_per_epoch=2), utc_now=clock
    )
    try:
        admission.check(
            WS,
            [op_item("adm_c_1"), op_item("adm_c_2"), op_item("adm_c_3")],
            received_at=clock(),
        )
    except AdmissionRejected:
        pass
    assert admission.rejections_total >= 1


# ---------------------------------------------------------------------------
# Admission (route: reject BEFORE durable acceptance)
# ---------------------------------------------------------------------------


def test_route_admission_rejects_before_durable_acceptance(
    tmp_path: Path,
) -> None:
    app = make_rt_app(
        tmp_path,
        admission_envelope=AdmissionEnvelope(
            max_operations_per_epoch=2, max_accepted_operations_per_second=10_000.0
        ),
    )
    with TestClient(app) as client:
        created = client.post(
            "/api/v1/scenarios",
            json={"site_id": SITE_ID, "name": "rt", "initial_state": "baseline"},
        )
        assert created.status_code == 201
        workspace = created.json()["scenario_id"]

        response = rt_post(client, workspace, [route_item(f"rej_{i}") for i in range(3)])
        assert response.status_code == 429
        error = response.json()["error"]
        assert error["code"] == "fast_lane_overloaded"
        assert error["retry_after_ms"] > 0
        assert error["max_operations_per_epoch"] == 2

        # REJECT-BEFORE-ACCEPT: nothing landed in the durable operation
        # log, and no epoch was opened for the rejected batch.
        ops = client.get(f"/api/v1/workspaces/{workspace}/operations").json()
        assert ops["operations"] == []
        epochs = client.get(f"/api/v1/workspaces/{workspace}/epochs").json()
        assert epochs["epochs"] == []


def test_route_admission_accepts_inside_envelope(tmp_path: Path) -> None:
    app = make_rt_app(tmp_path)
    with TestClient(app) as client:
        created = client.post(
            "/api/v1/scenarios",
            json={"site_id": SITE_ID, "name": "rt", "initial_state": "baseline"},
        )
        workspace = created.json()["scenario_id"]
        response = rt_post(client, workspace, [route_item("ok_op_1")])
        assert response.status_code == 200
        assert [item["operation_id"] for item in response.json()["accepted"]] == [
            "ok_op_1"
        ]


# ---------------------------------------------------------------------------
# Fast lane: result classes + revision advance
# ---------------------------------------------------------------------------


def test_view_only_epoch_publishes_fast_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, clock = _clocked_store(tmp_path, monkeypatch)
    scheduler = EpochScheduler(store, hub=RecordingHub(), utc_now=clock)
    lane, hub, _, _ = make_fast_lane(store, clock=clock)

    store.append_operations(
        WS,
        [
            op_item(
                "fast_v_1",
                source_family="output_view",
                verb="select",
                entity_id=None,
                payload={"view": "utci"},
            )
        ],
    )
    clock.advance(WINDOW_S)
    scheduler.run_once()
    publish_due(lane)

    scenario = scenario_of(store)
    assert scenario.fast_revision == 1
    assert scenario.exact_result_version == 0
    # revision triple invariant on the row: exact <= fast <= workspace.
    assert (
        scenario.exact_result_version <= scenario.fast_revision <= scenario.scene_version
    )
    assert len(hub.fast) == 1
    _, event = hub.fast[0]
    assert event["result_class"] == "fast_exact"
    assert event["fast_revision"] == 1
    assert event["workspace_revision"] == 1
    assert event["epoch_id"] == 0
    assert "exact_revision" in event and "exact_base_revision" in event


def test_science_epoch_publishes_visual_pending_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, clock = _clocked_store(tmp_path, monkeypatch)
    scheduler = EpochScheduler(store, hub=RecordingHub(), utc_now=clock)
    lane, hub, _, _ = make_fast_lane(store, clock=clock)

    store.append_operations(WS, [op_item("fast_s_1")])
    clock.advance(WINDOW_S)
    scheduler.run_once()
    publish_due(lane)

    scenario = scenario_of(store)
    assert scenario.fast_revision == 1
    _, event = hub.fast[0]
    assert event["result_class"] == "visual_pending"
    assert event["fast_revision"] == 1


def test_exact_covered_revision_publishes_fast_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, clock = _clocked_store(tmp_path, monkeypatch)
    scheduler = EpochScheduler(store, hub=RecordingHub(), utc_now=clock)
    lane, hub, _, _ = make_fast_lane(store, clock=clock)

    store.append_operations(WS, [op_item("fast_e_1")])
    clock.advance(WINDOW_S)
    scheduler.run_once()
    # The exact lane already reconciled revision 1 (cache switch class).
    with store._write() as conn:
        conn.execute(
            "UPDATE scenarios SET exact_result_version = 1 WHERE scenario_id = ?",
            (WS,),
        )
    publish_due(lane)

    _, event = hub.fast[0]
    assert event["result_class"] == "fast_exact"
    assert event["exact_revision"] == 1
    assert event["fast_revision"] == 1


def test_fast_revision_monotonic_and_bounded_by_workspace_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, clock = _clocked_store(tmp_path, monkeypatch)
    scheduler = EpochScheduler(store, hub=RecordingHub(), utc_now=clock)
    lane, hub, _, _ = make_fast_lane(store, clock=clock)

    for index in range(3):
        store.append_operations(WS, [op_item(f"fast_m_{index}")])
        clock.advance(WINDOW_S)
        scheduler.run_once()
        publish_due(lane)
        scenario = scenario_of(store)
        assert scenario.fast_revision == scenario.scene_version
        assert (
            scenario.exact_result_version
            <= scenario.fast_revision
            <= scenario.scene_version
        )
    assert [event["fast_revision"] for _, event in hub.fast] == [1, 2, 3]


def test_lagging_lane_supersedes_to_newest_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, clock = _clocked_store(tmp_path, monkeypatch)
    scheduler = EpochScheduler(store, hub=RecordingHub(), utc_now=clock)
    lane, hub, _, _ = make_fast_lane(store, clock=clock)

    store.append_operations(WS, [op_item("fast_l_1")])
    clock.advance(WINDOW_S)
    scheduler.run_once()
    store.append_operations(WS, [op_item("fast_l_2")])
    clock.advance(WINDOW_S)
    scheduler.run_once()
    # Both epochs committed, lane never ran: ONE catch-up publish at the
    # newest epoch-final canonical state subsumes the older revision.
    publish_due(lane)

    scenario = scenario_of(store)
    assert scenario.fast_revision == 2
    assert len(hub.fast) == 1
    assert hub.fast[0][1]["fast_revision"] == 2
    assert hub.fast[0][1]["epoch_id"] == 1


def test_no_due_work_is_a_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, clock = _clocked_store(tmp_path, monkeypatch)
    lane, hub, _, _ = make_fast_lane(store, clock=clock)
    assert lane.run_once() == []
    assert hub.fast == []
    assert scenario_of(store).fast_revision == 0


# ---------------------------------------------------------------------------
# Deadline semantics (injected fake clocks)
# ---------------------------------------------------------------------------


def test_overdue_prediction_downgrades_without_running_kernel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, clock = _clocked_store(tmp_path, monkeypatch)
    scheduler = EpochScheduler(store, hub=RecordingHub(), utc_now=clock)
    kernel = StubKernel(predicted_ms=800.0, result_class="fast_exact")
    lane, hub, _, _ = make_fast_lane(store, clock=clock, kernel=kernel)

    store.append_operations(WS, [op_item("fast_d_1")])
    clock.advance(WINDOW_S)
    scheduler.run_once()
    # 950 ms of the 1,000 ms deadline already spent: the 800 ms kernel
    # cannot finish — publish the LOWER class before the deadline instead
    # of running it (degradation ladder, never a missed deadline).
    clock.advance(0.950)
    publish_due(lane)

    assert kernel.predicted_calls, "the cost prediction gates the kernel"
    assert kernel.run_calls == [], "an unmeetable kernel must not run"
    _, event = hub.fast[0]
    assert event["result_class"] == "visual_pending"
    assert scenario_of(store).fast_revision == 1


def test_kernel_inside_deadline_publishes_its_class(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, clock = _clocked_store(tmp_path, monkeypatch)
    scheduler = EpochScheduler(store, hub=RecordingHub(), utc_now=clock)
    kernel = StubKernel(predicted_ms=50.0, result_class="fast_exact")
    lane, hub, _, mono = make_fast_lane(store, clock=clock, kernel=kernel)

    store.append_operations(WS, [op_item("fast_k_1")])
    clock.advance(WINDOW_S)
    scheduler.run_once()
    mono.advance_ms(3.0)  # kernel + publish wall time
    publish_due(lane)

    assert kernel.run_calls == [1]
    _, event = hub.fast[0]
    assert event["result_class"] == "fast_exact"
    # telemetry standard names observed.
    names = lane_telemetry_names(lane)
    assert METRIC_FAST_COMPUTE_MS in names
    assert METRIC_RESULT_CLASS_TOTAL in names


def test_fast_qualified_is_reserved_and_never_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, clock = _clocked_store(tmp_path, monkeypatch)
    scheduler = EpochScheduler(store, hub=RecordingHub(), utc_now=clock)
    kernel = StubKernel(predicted_ms=10.0, result_class="fast_qualified")
    lane, hub, _, _ = make_fast_lane(store, clock=clock, kernel=kernel)

    store.append_operations(WS, [op_item("fast_q_1")])
    clock.advance(WINDOW_S)
    scheduler.run_once()
    publish_due(lane)

    _, event = hub.fast[0]
    assert event["result_class"] == "visual_pending", (
        "fast_qualified is reserved for R3/R7; the lane must downgrade"
    )
    assert all(
        body["result_class"] != "fast_qualified" for _, body in hub.fast
    ), "the reserved class must never ride the wire this wave"


# ---------------------------------------------------------------------------
# Queue semantics: deadline order, bounds, wake
# ---------------------------------------------------------------------------


def test_deadline_ordered_processing_across_workspaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, clock = _clocked_store(tmp_path, monkeypatch)
    make_workspace(store, "ws_late")
    scheduler = EpochScheduler(store, hub=RecordingHub(), utc_now=clock)
    kernel = StubKernel(predicted_ms=5.0)
    lane, hub, _, _ = make_fast_lane(store, clock=clock, kernel=kernel)

    # ws_late's operation is accepted FIRST (earlier deadline); WS's is a
    # full window later. The lane must serve the earlier deadline first.
    store.append_operations("ws_late", [op_item("late_1")])
    clock.advance(WINDOW_S / 2)
    store.append_operations(WS, [op_item("early_1")])
    clock.advance(WINDOW_S)
    scheduler.run_once()  # both epochs closed
    publish_due(lane)

    assert kernel.run_calls == [1, 1], "both revisions processed"
    order = [workspace for workspace, _ in hub.fast]
    assert order[0] == "ws_late", "earlier absolute deadline must be first"


def test_pending_discovery_is_bounded_and_self_heals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, clock = _clocked_store(tmp_path, monkeypatch)
    lane, _, _, _ = make_fast_lane(
        store, clock=clock, max_pending_workspaces=2
    )
    for index in range(10):
        lane.on_epoch_closed(f"ws_{index}", index, {"workspace_revision": index})
    assert len(lane.pending_workspaces()) <= 2
    # The scan (run_once) re-derives everything from durable state, so a
    # dropped wake cannot lose work — that is the self-heal.


def test_hook_entry_is_enqueue_and_return_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, clock = _clocked_store(tmp_path, monkeypatch)
    scheduler = EpochScheduler(store, hub=RecordingHub(), utc_now=clock)
    lane, hub, _, _ = make_fast_lane(store, clock=clock)
    scheduler.attach_after_close(lane.on_epoch_closed)

    store.append_operations(WS, [op_item("fast_hook_1")])
    clock.advance(WINDOW_S)
    scheduler.run_once()

    # The tail hook fired (waking the lane) but must NOT have run any fast
    # work inline: nothing published until the lane's own pass.
    assert hub.fast == []
    assert scenario_of(store).fast_revision == 0
    publish_due(lane)
    assert scenario_of(store).fast_revision == 1


# ---------------------------------------------------------------------------
# Epoch statuses: fast_planned advance + exact-lane non-disturbance
# ---------------------------------------------------------------------------


def test_epoch_row_advances_to_fast_planned_not_fast_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, clock = _clocked_store(tmp_path, monkeypatch)
    scheduler = EpochScheduler(store, hub=RecordingHub(), utc_now=clock)
    lane, _, _, _ = make_fast_lane(store, clock=clock)

    store.append_operations(WS, [op_item("fast_p_1")])
    clock.advance(WINDOW_S)
    scheduler.run_once()
    publish_due(lane)

    epoch = store.epoch_records(WS)[0]
    assert epoch.status == "fast_planned"
    # NOT fast_published: that status is TERMINAL and r2a's exact lane
    # counts terminal epochs as consumed — the fast side must not settle
    # the exact side's books. Fast completion lives in fast_revision.
    assert store.get_epoch(WS, 0).status != "fast_published"


def test_fast_publish_does_not_disturb_exact_consumption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.test_realtime_r2_exact_lane import make_runner

    store, clock = _clocked_store(tmp_path, monkeypatch)
    scheduler = EpochScheduler(store, hub=RecordingHub(), utc_now=clock)
    lane, _, _, _ = make_fast_lane(store, clock=clock)

    store.append_operations(WS, [op_item("fast_x_1")])
    clock.advance(WINDOW_S)
    scheduler.run_once()
    publish_due(lane)  # fast side published first (sub-second)

    # The epoch the fast lane touched is still consumable by the exact
    # lane: recoverable, and the exact tick still spawns its job.
    statuses = {e.epoch_id: e.status for e in store.epoch_records(WS)}
    assert statuses[0] == "fast_planned"
    assert store.recoverable_epochs(), "fast_planned must stay recoverable"

    runner = make_runner(tmp_path, store)
    runner._exact_lane_tick()
    exact_jobs = [
        job
        for job in store.nonterminal_jobs()
        if (job.request or {}).get("exact_lane")
    ]
    assert exact_jobs, "the exact lane must still target the revision"

    # And exact completion settles the epoch exactly as before.
    runner._consume_exact_epochs(WS, 1)
    assert store.get_epoch(WS, 0).status == "exact_targeted"


# ---------------------------------------------------------------------------
# epochs.py tail hook
# ---------------------------------------------------------------------------


def test_close_pipeline_fires_tail_hook_after_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, clock = _clocked_store(tmp_path, monkeypatch)
    hub = RecordingHub()
    calls: list[tuple[str, int, int]] = []  # (ws, epoch, canonical_seen)

    def after_close(workspace_id: str, epoch_id: int, event: dict[str, Any]) -> None:
        calls.append((workspace_id, epoch_id, len(hub.canonical)))

    scheduler = EpochScheduler(store, hub=hub, utc_now=clock)
    scheduler.attach_after_close(after_close)

    store.append_operations(WS, [op_item("hook_1")])
    clock.advance(WINDOW_S)
    event = scheduler.run_once()

    assert calls == [(WS, 0, 1)], "hook fires after the canonical broadcast"
    assert event  # the close pipeline result is unchanged


def test_attachable_post_construction_and_none_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, clock = _clocked_store(tmp_path, monkeypatch)
    scheduler = EpochScheduler(store, hub=RecordingHub(), utc_now=clock)
    scheduler.run_once()  # no hook attached: no error
    seen: list[str] = []
    scheduler.attach_after_close(
        lambda ws, epoch_id, event: seen.append(ws)
    )
    store.append_operations(WS, [op_item("hook_2")])
    clock.advance(WINDOW_S)
    scheduler.run_once()
    assert seen == [WS]


def test_hook_failure_never_breaks_close_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, clock = _clocked_store(tmp_path, monkeypatch)
    hub = RecordingHub()

    def bad_hook(workspace_id: str, epoch_id: int, event: dict[str, Any]) -> None:
        raise RuntimeError("hook exploded")

    scheduler = EpochScheduler(store, hub=hub, utc_now=clock)
    scheduler.attach_after_close(bad_hook)

    store.append_operations(WS, [op_item("hook_3")])
    clock.advance(WINDOW_S)
    event = scheduler.run_once()

    assert event is not None, "the close pipeline survives a broken hook"
    assert scenario_of(store).scene_version == 1
    assert len(hub.canonical) == 1


# ---------------------------------------------------------------------------
# Thread lifecycle (real threads)
# ---------------------------------------------------------------------------


def test_fast_lane_thread_start_stop_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, clock = _clocked_store(tmp_path, monkeypatch)
    lane, _, _, _ = make_fast_lane(store, clock=clock)
    assert not lane.running
    lane.start()
    assert lane.running
    thread = lane._thread
    assert thread is not None and thread.name == FAST_LANE_THREAD_NAME
    lane.start()  # idempotent: no second thread, same loop
    assert lane.running and lane._thread is thread
    assert lane.stop(timeout=2.0) is True
    assert not lane.running
    assert lane.stop(timeout=2.0) is True  # idempotent stop


def test_fast_lane_thread_publishes_after_hook_wake(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, clock = _clocked_store(tmp_path, monkeypatch)
    scheduler = EpochScheduler(store, hub=RecordingHub(), utc_now=clock)
    lane, hub, _, _ = make_fast_lane(store, clock=clock)
    scheduler.attach_after_close(lane.on_epoch_closed)

    started_at = time.monotonic()
    lane.start()
    try:
        store.append_operations(WS, [op_item("fast_thr_1")])
        clock.advance(WINDOW_S)
        scheduler.run_once()  # tail hook wakes the lane thread

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if scenario_of(store).fast_revision >= 1:
                break
            time.sleep(0.01)
        elapsed_ms = (time.monotonic() - started_at) * 1000.0
        assert scenario_of(store).fast_revision == 1
        assert hub.fast, "the publication must broadcast"
        assert elapsed_ms < 5000.0
    finally:
        assert lane.stop(timeout=2.0) is True


# ---------------------------------------------------------------------------
# App wiring: fast lane + admission + /metrics
# ---------------------------------------------------------------------------


def test_create_app_wires_default_fast_lane_and_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Default wiring registers standard metrics on the module-level shared
    # singleton; isolate it (precedent: test_realtime_epochs lifecycle
    # tests) so this app cycle cannot pollute test_realtime_telemetry.
    monkeypatch.setattr(
        "solweig_gpu.server.app.default_registry", lambda: TelemetryRegistry()
    )
    app = create_app(
        state_root=tmp_path / "state",
        sites={SITE_ID: {"cache_dir": make_site_cache(tmp_path)}},
        solver_factory=lambda context: FakeSolver(),
        start_worker=False,
        start_epoch_scheduler=False,
    )
    from solweig_gpu.server.realtime.admission import FastAdmission as FA

    assert isinstance(app.state.fast_lane, FastLaneScheduler)
    assert isinstance(app.state.fast_admission, FA)
    with TestClient(app):
        assert app.state.fast_lane.running, "lifespan starts the fast lane"
    assert not app.state.fast_lane.running, "lifespan stops the fast lane"


def test_create_app_start_fast_lane_false(tmp_path: Path) -> None:
    app = make_rt_app(tmp_path)  # helpers inject start_fast_lane=False
    with TestClient(app) as client:  # noqa: SIM117 - readability
        assert not app.state.fast_lane.running


def test_metrics_includes_realtime_telemetry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # This test drives REAL epoch close + fast publish through the DEFAULT
    # wiring; the shared singleton must stay clean for other suites. One
    # shared private registry for the whole patch so the app's schedulers
    # AND the /metrics route (default_registry() per request) agree.
    registry = TelemetryRegistry()
    monkeypatch.setattr(
        "solweig_gpu.server.app.default_registry", lambda: registry
    )
    app = create_app(
        state_root=tmp_path / "state",
        sites={SITE_ID: {"cache_dir": make_site_cache(tmp_path)}},
        solver_factory=lambda context: FakeSolver(),
        start_worker=False,
        start_epoch_scheduler=False,
    )
    scheduler: EpochScheduler = app.state.epoch_scheduler
    lane: FastLaneScheduler = app.state.fast_lane
    with TestClient(app) as client:
        created = client.post(
            "/api/v1/scenarios",
            json={"site_id": SITE_ID, "name": "rt", "initial_state": "baseline"},
        )
        workspace = created.json()["scenario_id"]
        posted = rt_post(client, workspace, [route_item("m_op_1")])
        assert posted.status_code == 200
        # Drive both planes manually (deterministic), then scrape.
        scheduler.close_epoch(workspace, 0)
        lane.run_once()

        metrics = client.get("/metrics")
        assert metrics.status_code == 200
        body = metrics.json()
        assert body["scenarios"] == 1
        assert "realtime" in body, "r1-epochs flagged gap: /metrics wiring"
        names = set(body["realtime"])
        assert METRIC_RESULT_CLASS_TOTAL in names
        assert METRIC_FAST_COMPUTE_MS in names


def test_admission_gate_reachable_through_default_app(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The default admission uses the default envelope; one whole-raster
    # land-cover paint (600x600 = 360,000 cells > the 100,000-cell bound,
    # and ~1.4 s of booked compute) must 429 BEFORE the durable append.
    monkeypatch.setattr(
        "solweig_gpu.server.app.default_registry", lambda: TelemetryRegistry()
    )
    app = create_app(
        state_root=tmp_path / "state",
        sites={SITE_ID: {"cache_dir": make_site_cache(tmp_path)}},
        solver_factory=lambda context: FakeSolver(),
        start_worker=False,
        start_epoch_scheduler=False,
    )
    with TestClient(app) as client:
        created = client.post(
            "/api/v1/scenarios",
            json={"site_id": SITE_ID, "name": "rt", "initial_state": "baseline"},
        )
        workspace = created.json()["scenario_id"]
        paint = route_item(
            "def_paint_1",
            source_family="landcover_surface",
            verb="paint",
            entity_id=None,
            payload={
                "window": {
                    "row_start": 0,
                    "row_end": 599,
                    "col_start": 0,
                    "col_end": 599,
                },
                "value": 3,
            },
        )
        response = rt_post(client, workspace, [paint])
        assert response.status_code == 429
        error = response.json()["error"]
        assert error["code"] == "fast_lane_overloaded"
        ops = client.get(f"/api/v1/workspaces/{workspace}/operations").json()
        assert ops["operations"] == []


# ---------------------------------------------------------------------------
# SSE wire shape
# ---------------------------------------------------------------------------


def test_fast_revision_sse_event_on_the_stream(tmp_path: Path) -> None:
    app = make_rt_app(tmp_path)
    with TestClient(app) as client:
        created = client.post(
            "/api/v1/scenarios",
            json={"site_id": SITE_ID, "name": "rt", "initial_state": "baseline"},
        )
        workspace = created.json()["scenario_id"]
        posted = rt_post(client, workspace, [route_item("sse_fast_1")])
        assert posted.status_code == 200

        scheduler: EpochScheduler = app.state.epoch_scheduler
        lane: FastLaneScheduler = app.state.fast_lane

        async def scenario() -> None:
            async with SSEConnection(
                app, f"/api/v1/workspaces/{workspace}/events"
            ) as conn:
                reader = SSEReader(conn.lines())
                name, _snapshot = await reader.next_event()

                scheduler.close_epoch(workspace, 0)
                lane.run_once()

                # canonical_revision first (epoch close), then the fast
                # lane's fast_revision frame carrying the result class.
                name, event = await reader.next_event()
                assert name == "canonical_revision"
                name, fast_event = await reader.next_event()
                assert name == "fast_revision"
                assert fast_event["result_class"] == "visual_pending"
                assert fast_event["fast_revision"] == 1
                assert fast_event["workspace_revision"] == 1
                assert (
                    fast_event["exact_revision"]
                    <= fast_event["fast_revision"]
                    <= fast_event["workspace_revision"]
                )

        asyncio.run(asyncio.wait_for(scenario(), 10.0))


# ---------------------------------------------------------------------------
# r2b review remediation (F1 ordering grace, F2 reservation parity,
# F4 boot re-broadcast phantom)
# ---------------------------------------------------------------------------


def test_scan_discovery_defers_one_pass_before_publishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F1 (r2b review): the discovery scan can observe a revision between
    the epoch pipeline's durable commit and its canonical broadcast
    submission; a scan-discovered revision therefore waits ONE lane pass
    (grace) so the fast frame can never reach subscribers before the
    canonical frame for the same revision. Hook-woken work is already
    post-broadcast and pays no grace."""
    store, clock = _clocked_store(tmp_path, monkeypatch)
    scheduler = EpochScheduler(store, hub=RecordingHub(), utc_now=clock)
    lane, hub, _, _ = make_fast_lane(store, clock=clock)

    store.append_operations(WS, [op_item("grace_1")])
    clock.advance(WINDOW_S)
    scheduler.run_once()  # epoch committed; the lane was NEVER woken

    # First scan pass: due but deferred (grace) — reserved, not published.
    assert lane.run_once() == []
    assert lane.pending_workspaces() == [WS]
    assert hub.fast == []

    # Second pass: grace spent, the revision publishes.
    assert lane.run_once() == [(WS, 0)]
    assert scenario_of(store).fast_revision == 1
    assert hub.fast[0][1]["workspace_revision"] == 1
    assert lane.latest_publication(WS) is not None


def test_hook_woken_revision_publishes_on_the_first_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F1 corollary: the grace belongs to the SCAN path only. The tail
    hook fires after the canonical broadcast was submitted, so
    hook-woken work publishes on the very next pass (sub-tick latency)."""
    store, clock = _clocked_store(tmp_path, monkeypatch)
    epoch_hub = RecordingHub()
    scheduler = EpochScheduler(store, hub=epoch_hub, utc_now=clock)
    lane, hub, _, _ = make_fast_lane(store, clock=clock)

    store.append_operations(WS, [op_item("grace_h_1")])
    clock.advance(WINDOW_S)
    assert scheduler.run_once(), "test setup: the epoch must close"
    # Re-deliver the hook exactly as epochs.py would, post-broadcast: the
    # canonical event IS the close event handed to the tail hook.
    _, close_event = epoch_hub.canonical[-1]
    lane.on_epoch_closed(WS, 0, close_event)
    assert lane.run_once() == [(WS, 0)]
    assert hub.fast[0][1]["workspace_revision"] == 1


def test_reserved_ms_books_payload_costs_like_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F2 (r2b review): reserved_ms must use the SAME epoch-coalesced
    formula as admission INCLUDING land-cover window cell counts — a
    payload-stripped reservation under-books by the cell-derived cost and
    admission then offers compute the lane has already promised."""
    monkeypatch.setattr(
        "solweig_gpu.server.realtime.scheduler._SCAN_CADENCE_MS", 60_000.0
    )
    store, clock = _clocked_store(tmp_path, monkeypatch)
    scheduler = EpochScheduler(store, hub=RecordingHub(), utc_now=clock)
    lane, _, _, _ = make_fast_lane(store, clock=clock)

    paint_payload = {
        "window": {"row_start": 0, "row_end": 199, "col_start": 0, "col_end": 199},
        "value": 3,
    }
    store.append_operations(
        WS,
        [
            op_item(
                "reserve_paint_1",
                source_family="landcover_surface",
                verb="paint",
                entity_id=None,
                payload=paint_payload,
            )
        ],
    )
    clock.advance(WINDOW_S)
    scheduler.run_once()

    lane.start()  # recovery scan discovers the epoch (grace: not yet published)
    try:
        expected = predict_epoch_compute_ms(
            [
                {
                    "source_family": "landcover_surface",
                    "payload": paint_payload,
                }
            ]
        )
        stripped = predict_epoch_compute_ms(
            [{"source_family": "landcover_surface"}]
        )
        # 40,000 window cells dominate the family base; the margin is per-op.
        assert expected == 200 * 200 * 0.004 + 2.0
        assert expected > stripped, "payload MUST move the booking"
        assert lane.pending_workspaces() == [WS]
        assert lane.reserved_ms(WS) == expected
    finally:
        lane.stop()


def test_boot_rebroadcast_adds_no_phantom_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F4 (r2b review): after boot recovery publishes a revision, the
    epoch plane's own re-drive re-broadcasts the same close (tail hook).
    The hook must skip revisions at or below the workspace's last fast
    publication — otherwise a pending entry that no scan ever matches
    (fast_revision is already caught up) is never popped and reserves
    compute until the next epoch."""
    monkeypatch.setattr(
        "solweig_gpu.server.realtime.scheduler._SCAN_CADENCE_MS", 60_000.0
    )
    store, clock = _clocked_store(tmp_path, monkeypatch)
    scheduler = EpochScheduler(store, hub=RecordingHub(), utc_now=clock)
    lane, _, _, _ = make_fast_lane(store, clock=clock)

    store.append_operations(WS, [op_item("phantom_1")])
    clock.advance(WINDOW_S)
    scheduler.run_once()

    lane.start()
    try:
        # Publish the due revision explicitly (recovery's grace pass first).
        lane.run_once()
        publication = lane.latest_publication(WS)
        assert publication is not None
        assert publication.workspace_revision == 1
        assert lane.pending_workspaces() == []

        # The epoch plane re-drives the SAME close after restart: the hook
        # must not re-enqueue published work.
        operations = store.operations_for_epoch(WS, publication.epoch_id)
        rebroadcast = {
            "workspace_revision": 1,
            "epoch_id": publication.epoch_id,
            "operations": [
                {"operation_id": op.operation_id, "accepted_at": op.accepted_at}
                for op in operations
            ],
        }
        lane.on_epoch_closed(WS, publication.epoch_id, rebroadcast)

        assert lane.pending_workspaces() == [], "phantom pending entry"
        assert lane.reserved_ms(WS) == 0.0, "phantom reservation"
    finally:
        lane.stop()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _clocked_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Store, FakeClock]:
    clock = FakeClock()
    monkeypatch.setattr("solweig_gpu.server.store._now_utc", clock)
    store = make_store(tmp_path)
    make_workspace(store)
    return store, clock


def lane_telemetry_names(lane: FastLaneScheduler) -> set[str]:
    registry = lane._telemetry
    assert registry is not None
    return set(registry.snapshot())
