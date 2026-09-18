# SPDX-License-Identifier: GPL-3.0-only
"""R7 fast-qualified lane: compensated vegetation epochs, un-reserved with
structural qualifier fences (failing-first).

Pins the R7 mission contracts (``docs/incremental_design_tool/
realtime_collaboration/compensation_and_exactness.md`` +
``service_level_contract.md`` "Degradation ladder" +
``realtime_contract.yaml`` ``result_classes``):

* **Un-reserved class** — a kernel may return ``fast_qualified`` ONLY with a
  COMPLETE anchored qualifier payload. The scheduler structurally validates
  the payload; a ``fast_qualified`` result without complete qualifier
  metadata is a DEFECT and downgrades to ``visual_pending`` loudly (typed
  telemetry counter + payload disclosure). "Never reduce precision
  silently" is enforced structurally, not by convention.
* **Anchored delta** — a qualified payload is a DELTA against the last
  exact planes: it carries ``exact_base_revision`` plus what base was used
  (``base_planes``). Never standalone planes.
* **Supersession fence** — a qualified frame must never survive as
  authoritative after the exact lane publishes a result covering that
  revision: the next lane pass re-publishes the frame as ``fast_exact``
  (superseded), once.
* **Priority discipline** — compensation is the LAST resort: epochs served
  by cheaper structural paths (view-only → ``fast_exact``; met-only epochs
  served exactly by the exact-lane fast paths) are NOT compensated.
* **Deadline discipline unchanged** — a kernel whose measured-honest
  predicted cost cannot fit the remaining budget still never runs.

Written failing-first against base b08b22d (fast_qualified reserved: the
lane downgrades the class unconditionally, no qualifier contract, no
supersession fence, no compensated kernel).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from solweig_gpu.server.app import create_app
from solweig_gpu.server.realtime.broadcast import BroadcastHub
from solweig_gpu.server.realtime.epochs import EpochScheduler
from solweig_gpu.server.realtime.scheduler import (
    FastKernelResult,
    FastLaneScheduler,
    FastPlan,
    ViewCacheKernel,
)
from solweig_gpu.server.realtime.telemetry import TelemetryRegistry
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
from tests.test_realtime_r2_fast_lane import (
    RecordingHub,
    _clocked_store,
    make_fast_lane,
    publish_due,
    scenario_of,
)
from tests.test_server_api import SITE_ID, FakeSolver, make_site_cache

# ---------------------------------------------------------------------------
# Fixtures: a complete anchored qualifier payload (the structural contract)
# ---------------------------------------------------------------------------


def complete_qualified_payload(**overrides: Any) -> dict[str, Any]:
    """A fast_qualified payload carrying EVERY required qualifier field.

    The schema (validated structurally by the lane, kernel-agnostic):

    * ``qualifier.model_version``   — non-empty version of the compensation
      model that produced the delta.
    * ``qualifier.form``            — the model form (e.g. geometric
      overlay); pins how consumers must interpret the delta.
    * ``qualifier.domain``          — qualification domain: families +
      variables covered.
    * ``qualifier.error_evidence``  — measured error bound from a HELD-OUT
      calibration split: per-variable metrics, ``n_holdout``, and an
      evidence pointer to the calibration run.
    * ``qualifier.reconciliation``  — how exact reconciliation happens.
    * ``exact_base_revision``       — the ANCHOR: exact revision whose
      planes the delta is computed against.
    * ``base_planes``               — WHAT base was used (result version +
      variables), never implicit.
    """
    payload: dict[str, Any] = {
        "kernel": "qualified_stub",
        "qualifier": {
            "model_version": "veg-shadow-comp-v1",
            "form": "geometric_overlay",
            "domain": {
                "families": ["vegetation_geometry"],
                "variables": ["shadow"],
            },
            "error_evidence": {
                "metrics": {
                    "shadow_iou": {"p50": 0.93, "p95": 0.88, "max": 0.81},
                },
                "n_holdout": 4,
                "evidence": "/tmp/r7_proof/holdout.json",
                "calibration_run_id": "r7-holdout-site500",
            },
            "reconciliation": "exact lane targets this revision; delta overlay discarded on arrival",
        },
        "exact_base_revision": 0,
        "base_planes": {
            "result_version": 0,
            "variables": ["shadow"],
            "source": "exact lane result",
        },
        "delta": {"trees": 1, "time_steps": 24},
    }
    payload.update(overrides)
    return payload


class QualifiedStubKernel:
    """Injected kernel returning fast_qualified with a COMPLETE payload."""

    name = "qualified_stub"

    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        self.payload = payload if payload is not None else complete_qualified_payload()
        self.run_calls: list[int] = []
        self.predicted_ms = 10.0

    def predicted_cost_ms(self, plan: Any) -> float:
        return self.predicted_ms

    def run(self, plan: Any) -> FastKernelResult:
        self.run_calls.append(plan.workspace_revision)
        return FastKernelResult(result_class="fast_qualified", payload=self.payload)


def drive_epoch(
    store: Store,
    scheduler: EpochScheduler,
    clock: FakeClock,
    operations: list[dict[str, Any]],
    workspace: str = WS,
) -> None:
    """Append ops, close the epoch, commit the canonical revision."""
    store.append_operations(workspace, operations)
    clock.advance(WINDOW_S)
    scheduler.run_once()


def publish_exact_stub(
    store: Store, workspace: str = WS, variables: tuple[str, ...] = ("shadow",)
) -> int:
    """Publish a stub EXACT result for the current scene version.

    Realistic manifest shape (patch_codec): ``variables`` is a list of
    variable-meta mappings. Sets ``exact_result_version`` like the exact
    lane does. Returns the published version.
    """
    scenario = store.require_scenario(workspace)
    version = int(scenario.scene_version)
    store.publish_result(
        workspace,
        version,
        manifest={
            "variables": [
                {"name": name, "dtype": "float32", "shape": [3, 64, 64]}
                for name in variables
            ],
            "checksum": "stub",
        },
        payload=b"stub-exact-bytes",
    )
    return version


# ---------------------------------------------------------------------------
# Un-reserve: a COMPLETE qualified payload publishes
# ---------------------------------------------------------------------------


def test_complete_qualifier_publishes_fast_qualified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, clock = _clocked_store(tmp_path, monkeypatch)
    scheduler = EpochScheduler(store, hub=RecordingHub(), utc_now=clock)
    kernel = QualifiedStubKernel()
    lane, hub, _, _ = make_fast_lane(store, clock=clock, kernel=kernel)

    drive_epoch(store, scheduler, clock, [op_item("fq_ok_1")])
    publish_due(lane)

    assert kernel.run_calls == [1], "an eligible kernel must run"
    assert len(hub.fast) == 1
    _, event = hub.fast[0]
    assert event["result_class"] == "fast_qualified"
    assert event["payload"]["qualifier"]["model_version"] == "veg-shadow-comp-v1"
    assert event["fast_revision"] == 1
    # Revision triple invariant unchanged by the new class.
    scenario = scenario_of(store)
    assert (
        scenario.exact_result_version
        <= scenario.fast_revision
        <= scenario.scene_version
    )


def test_qualified_publication_is_anchored_to_exact_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, clock = _clocked_store(tmp_path, monkeypatch)
    scheduler = EpochScheduler(store, hub=RecordingHub(), utc_now=clock)
    # Epoch 1 reconciled exactly (anchor candidate); epoch 2 is the
    # compensated vegetation epoch anchored to revision 1.
    drive_epoch(store, scheduler, clock, [op_item("fq_anchor_1")])
    publish_exact_stub(store)
    payload = complete_qualified_payload(
        exact_base_revision=1,
        base_planes={"result_version": 1, "variables": ["shadow"], "source": "exact lane result"},
    )
    kernel = QualifiedStubKernel(payload=payload)
    lane, hub, _, _ = make_fast_lane(store, clock=clock, kernel=kernel)
    drive_epoch(store, scheduler, clock, [op_item("fq_anchor_2")])
    publish_due(lane)

    _, event = hub.fast[0]
    assert event["result_class"] == "fast_qualified"
    assert event["workspace_revision"] == 2
    # The anchor rides the wire: base <= exact <= fast <= workspace.
    assert event["exact_base_revision"] == 1
    assert (
        event["exact_base_revision"]
        <= event["exact_revision"]
        <= event["fast_revision"]
        <= event["workspace_revision"]
    )
    assert event["payload"]["base_planes"]["result_version"] == 1


def test_qualifier_acceptance_leaves_view_epochs_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """View-only epochs keep their structural fast_exact path; the new
    class only ever ADDS a lane, never steals the cheaper one."""
    store, clock = _clocked_store(tmp_path, monkeypatch)
    scheduler = EpochScheduler(store, hub=RecordingHub(), utc_now=clock)
    lane, hub, _, _ = make_fast_lane(store, clock=clock)  # default kernel

    store.append_operations(
        WS,
        [
            op_item(
                "fq_view_1",
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

    _, event = hub.fast[0]
    assert event["result_class"] == "fast_exact"


# ---------------------------------------------------------------------------
# Structural downgrade: EVERY missing/malformed qualifier field
# ---------------------------------------------------------------------------

_MISSING_FIELD_CASES: list[tuple[str, Any, str]] = [
    ("missing_qualifier", lambda p: p.pop("qualifier"), "missing_qualifier"),
    ("invalid_qualifier", lambda p: p.update(qualifier="not-a-mapping"), "invalid_qualifier"),
    ("missing_model_version", lambda p: p["qualifier"].pop("model_version"), "missing_model_version"),
    ("blank_model_version", lambda p: p["qualifier"].update(model_version="  "), "missing_model_version"),
    ("missing_form", lambda p: p["qualifier"].pop("form"), "missing_form"),
    ("missing_domain", lambda p: p["qualifier"].pop("domain"), "missing_domain"),
    ("missing_domain_variables", lambda p: p["qualifier"]["domain"].pop("variables"), "missing_domain_variables"),
    ("missing_domain_families", lambda p: p["qualifier"]["domain"].pop("families"), "missing_domain"),
    ("missing_error_evidence", lambda p: p["qualifier"].pop("error_evidence"), "missing_error_evidence"),
    ("empty_metrics", lambda p: p["qualifier"]["error_evidence"].update(metrics={}), "invalid_error_evidence"),
    ("zero_holdout", lambda p: p["qualifier"]["error_evidence"].update(n_holdout=0), "invalid_error_evidence"),
    ("missing_evidence_pointer", lambda p: p["qualifier"]["error_evidence"].pop("evidence"), "invalid_error_evidence"),
    ("missing_reconciliation", lambda p: p["qualifier"].pop("reconciliation"), "missing_reconciliation"),
    ("missing_anchor", lambda p: p.pop("exact_base_revision"), "missing_anchor"),
    ("negative_anchor", lambda p: p.update(exact_base_revision=-1), "invalid_anchor"),
    ("missing_base_planes", lambda p: p.pop("base_planes"), "missing_base_planes"),
]


@pytest.mark.parametrize(
    "case_id,mutate,expected_defect",
    _MISSING_FIELD_CASES,
    ids=[case[0] for case in _MISSING_FIELD_CASES],
)
def test_incomplete_qualifier_downgrades_to_visual_pending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case_id: str,
    mutate: Any,
    expected_defect: str,
) -> None:
    store, clock = _clocked_store(tmp_path, monkeypatch)
    scheduler = EpochScheduler(store, hub=RecordingHub(), utc_now=clock)
    payload = complete_qualified_payload()
    mutate(payload)
    kernel = QualifiedStubKernel(payload=payload)
    lane, hub, _, _ = make_fast_lane(store, clock=clock, kernel=kernel)

    drive_epoch(store, scheduler, clock, [op_item(f"fq_bad_{case_id}")])
    publish_due(lane)

    assert len(hub.fast) == 1
    _, event = hub.fast[0]
    assert event["result_class"] == "visual_pending", (
        f"a fast_qualified payload with defect {expected_defect} must downgrade"
    )
    assert all(
        body["result_class"] != "fast_qualified" for _, body in hub.fast
    ), "an unqualified payload must never ride the wire as fast_qualified"
    # The downgrade is LOUD: the defect rides the payload.
    reasons = event["payload"].get("downgrade_reasons", [])
    assert expected_defect in reasons, f"payload must disclose defect {expected_defect}"


def test_qualifier_downgrade_is_counted_in_telemetry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from solweig_gpu.server.realtime.telemetry import (
        METRIC_FAST_QUALIFIED_DOWNGRADED_TOTAL,
    )

    store, clock = _clocked_store(tmp_path, monkeypatch)
    scheduler = EpochScheduler(store, hub=RecordingHub(), utc_now=clock)
    payload = complete_qualified_payload()
    payload.pop("qualifier")
    registry = TelemetryRegistry()
    hub = RecordingHub()
    lane = FastLaneScheduler(
        store,
        hub=hub,
        telemetry=registry,
        kernel=QualifiedStubKernel(payload=payload),
        utc_now=clock,
    )

    drive_epoch(store, scheduler, clock, [op_item("fq_tel_1")])
    publish_due(lane)

    assert hub.fast[0][1]["result_class"] == "visual_pending"
    snapshot = registry.snapshot()
    assert METRIC_FAST_QUALIFIED_DOWNGRADED_TOTAL in snapshot, (
        "qualifier downgrades need a dedicated typed counter"
    )
    series = snapshot[METRIC_FAST_QUALIFIED_DOWNGRADED_TOTAL]
    assert any("missing_qualifier" in key for key in series), series


def test_reserved_downgrade_reason_is_now_qualifier_not_class(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bare fast_qualified (no qualifier at all) still downgrades — the
    r2b reserved-class fence narrows to the qualifier-completeness fence."""
    store, clock = _clocked_store(tmp_path, monkeypatch)
    scheduler = EpochScheduler(store, hub=RecordingHub(), utc_now=clock)
    kernel = QualifiedStubKernel(payload={"kernel": "bare"})
    lane, hub, _, _ = make_fast_lane(store, clock=clock, kernel=kernel)

    drive_epoch(store, scheduler, clock, [op_item("fq_bare_1")])
    publish_due(lane)

    _, event = hub.fast[0]
    assert event["result_class"] == "visual_pending"
    assert "missing_qualifier" in event["payload"]["downgrade_reasons"]


# ---------------------------------------------------------------------------
# Supersession fence: qualified never survives exact publication
# ---------------------------------------------------------------------------


def test_supersession_fence_republishes_after_exact_advance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, clock = _clocked_store(tmp_path, monkeypatch)
    scheduler = EpochScheduler(store, hub=RecordingHub(), utc_now=clock)
    lane, hub, _, _ = make_fast_lane(store, clock=clock, kernel=QualifiedStubKernel())

    drive_epoch(store, scheduler, clock, [op_item("fq_sup_1")])
    publish_due(lane)  # qualified at revision 1, anchored to exact base 0
    assert hub.fast[-1][1]["result_class"] == "fast_qualified"

    # The exact lane reconciles revision 1 (direct SQL: the r2a precedent).
    with store._write() as conn:
        conn.execute(
            "UPDATE scenarios SET exact_result_version = 1 WHERE scenario_id = ?",
            (WS,),
        )

    published = lane.run_once()
    assert published == [(WS, 0)], "the supersession pass reports the work"
    event = hub.fast[-1][1]
    assert event["result_class"] == "fast_exact"
    assert event["fast_revision"] == 1  # no revision churn: class supersession only
    assert event["workspace_revision"] == 1
    assert event["payload"]["superseded_qualified"] is True

    # Idempotent: the fence fires exactly once per stale frame.
    assert lane.run_once() == []
    assert len(hub.fast) == 2


def test_no_supersession_when_exact_below_qualified_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exact caught up to revision 1 while the qualified frame represents
    revision 2: the frame is still the newest view — no supersession."""
    store, clock = _clocked_store(tmp_path, monkeypatch)
    scheduler = EpochScheduler(store, hub=RecordingHub(), utc_now=clock)
    drive_epoch(store, scheduler, clock, [op_item("fq_nosup_1")])
    publish_exact_stub(store)
    lane, hub, _, _ = make_fast_lane(
        store,
        clock=clock,
        kernel=QualifiedStubKernel(
            payload=complete_qualified_payload(exact_base_revision=1)
        ),
    )
    drive_epoch(store, scheduler, clock, [op_item("fq_nosup_2")])
    publish_due(lane)
    assert hub.fast[-1][1]["result_class"] == "fast_qualified"
    assert hub.fast[-1][1]["workspace_revision"] == 2

    assert lane.run_once() == []
    assert len(hub.fast) == 1, "exact at 1 does not cover revision 2"


# ---------------------------------------------------------------------------
# Deadline discipline: an unmeetable qualified kernel never runs
# ---------------------------------------------------------------------------


def test_overdue_qualified_kernel_refused_before_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, clock = _clocked_store(tmp_path, monkeypatch)
    scheduler = EpochScheduler(store, hub=RecordingHub(), utc_now=clock)
    kernel = QualifiedStubKernel()
    kernel.predicted_ms = 800.0
    lane, hub, _, _ = make_fast_lane(store, clock=clock, kernel=kernel)

    drive_epoch(store, scheduler, clock, [op_item("fq_dl_1")])
    clock.advance(0.950)  # 950 of the 1,000 ms budget spent
    publish_due(lane)

    assert kernel.run_calls == [], "an unmeetable kernel must not run"
    _, event = hub.fast[0]
    assert event["result_class"] == "visual_pending"
    assert event["payload"]["downgraded"] is True


# ---------------------------------------------------------------------------
# SSE wire: the class (and its qualifier) ride the stream
# ---------------------------------------------------------------------------


def _qualified_rt_app(tmp_path: Path) -> Any:
    def scheduler_factory(store: Store, hub: BroadcastHub) -> EpochScheduler:
        return EpochScheduler(store, hub=hub, telemetry=TelemetryRegistry())

    def fast_lane_factory(store: Store, hub: BroadcastHub) -> FastLaneScheduler:
        return FastLaneScheduler(
            store,
            hub=hub,
            telemetry=TelemetryRegistry(),
            kernel=QualifiedStubKernel(),
        )

    return create_app(
        state_root=tmp_path / "state",
        sites={SITE_ID: {"cache_dir": make_site_cache(tmp_path)}},
        solver_factory=lambda context: FakeSolver(),
        coalescing_window_ms=20.0,
        start_worker=True,
        requests_per_minute_per_ip=None,
        edits_per_minute=None,
        start_epoch_scheduler=False,
        start_fast_lane=False,
        epoch_scheduler_factory=scheduler_factory,
        fast_lane_factory=fast_lane_factory,
    )


def test_sse_stream_carries_qualified_class(tmp_path: Path) -> None:
    app = _qualified_rt_app(tmp_path)
    with TestClient(app) as client:
        created = client.post(
            "/api/v1/scenarios",
            json={"site_id": SITE_ID, "name": "rt", "initial_state": "baseline"},
        )
        assert created.status_code == 201
        workspace = created.json()["scenario_id"]
        posted = rt_post(client, workspace, [route_item("fq_sse_1")])
        assert posted.status_code == 200

        scheduler: EpochScheduler = app.state.epoch_scheduler
        lane: FastLaneScheduler = app.state.fast_lane

        async def scenario() -> None:
            async with SSEConnection(
                app, f"/api/v1/workspaces/{workspace}/events"
            ) as conn:
                reader = SSEReader(conn.lines())
                await reader.next_event()  # initial snapshot

                scheduler.close_epoch(workspace, 0)
                lane.run_once()

                await reader.next_event()  # canonical_revision
                name, fast_event = await reader.next_event()
                assert name == "fast_revision"
                assert fast_event["result_class"] == "fast_qualified"
                qualifier = fast_event["payload"]["qualifier"]
                assert qualifier["model_version"]
                assert qualifier["error_evidence"]["n_holdout"] >= 1
                assert (
                    fast_event["exact_base_revision"]
                    <= fast_event["exact_revision"]
                    <= fast_event["fast_revision"]
                    <= fast_event["workspace_revision"]
                )

        asyncio.run(asyncio.wait_for(scenario(), 10.0))


# ---------------------------------------------------------------------------
# The compensated vegetation kernel (priority discipline + anchoring)
# ---------------------------------------------------------------------------


class FakeSites:
    """Minimal site source: grid geometry + deterministic sun positions."""

    def geometry(self, site_id: str) -> dict[str, Any]:
        # North-up grid (RasterGrid convention): origin is the outer
        # upper-left corner, rows increase southward — northing spans
        # [origin_y - 128, origin_y] so test trees at y_m <= 64 are on-grid.
        return {
            "rows": 64,
            "cols": 64,
            "pixel_size_m": 2.0,
            "origin_x_m": 0.0,
            "origin_y_m": 128.0,
            "time_steps": 3,
        }

    def sun_positions(self, site_id: str) -> list[tuple[float, float]]:
        return [(45.0, 180.0), (25.0, 210.0), (-8.0, 300.0)]


def veg_plan(store: Store, *, families: tuple[str, ...]) -> FastPlan:
    """A plan over the LATEST canonical state (the lane's planning input)."""
    canonical = store.latest_canonical_state(WS)
    now = datetime.now(timezone.utc)
    return FastPlan(
        workspace_id=WS,
        epoch_id=1,
        workspace_revision=int(canonical.workspace_revision) if canonical else 1,
        canonical_revision=int(canonical.workspace_revision) if canonical else 1,
        deadline_at=now + timedelta(seconds=1.0),
        now=now,
        families=families,
        operation_count=1,
        canonical_state=dict(canonical.families) if canonical else {},
    )


def compensated_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Store, FakeClock, EpochScheduler]:
    """A store anchored on a published exact shadow result at revision 1."""
    store, clock = _clocked_store(tmp_path, monkeypatch)
    scheduler = EpochScheduler(store, hub=RecordingHub(), utc_now=clock)
    drive_epoch(store, scheduler, clock, [op_item("fq_k_base")])
    publish_exact_stub(store)
    return store, clock, scheduler


def make_kernel(store: Store) -> Any:
    from solweig_gpu.server.realtime.compensated import (
        VegetationCompensationKernel,
    )

    return VegetationCompensationKernel(store=store, sites=FakeSites())


def test_kernel_publishes_qualified_for_pure_vegetation_epoch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from solweig_gpu.server.realtime.qualification import qualifier_defects

    store, clock, scheduler = compensated_setup(tmp_path, monkeypatch)
    kernel = make_kernel(store)
    drive_epoch(
        store,
        scheduler,
        clock,
        [
            op_item(
                "fq_k_add",
                entity_id="tree-B",
                payload={
                    "values": {
                        "x_m": 40.0,
                        "y_m": 40.0,
                        "height_m": 10.0,
                        "canopy_radius_m": 5.5,
                        "trunk_ratio": 0.25,
                    }
                },
            )
        ],
    )
    plan = veg_plan(store, families=("vegetation_geometry",))

    assert kernel.eligible(plan) is True
    assert kernel.predicted_cost_ms(plan) < 500.0, "must fit the 550 ms compute budget"
    result = kernel.run(plan)

    assert result.result_class == "fast_qualified"
    assert qualifier_defects(dict(result.payload)) == ()
    payload = dict(result.payload)
    assert payload["exact_base_revision"] == 1, "anchored to the exact result"
    assert payload["base_planes"]["result_version"] == 1
    coverage = payload["coverage"]
    assert coverage, "sun-up timesteps must carry predicted change cells"
    assert all(step["changed_cells"] > 0 for step in coverage)
    assert len(coverage) == 2, "only sun-up timesteps (2 of 3 here)"


def test_kernel_refuses_without_exact_base_planes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No published exact shadow result -> NO anchor -> honest refusal."""
    store, clock = _clocked_store(tmp_path, monkeypatch)
    scheduler = EpochScheduler(store, hub=RecordingHub(), utc_now=clock)
    kernel = make_kernel(store)
    drive_epoch(store, scheduler, clock, [op_item("fq_k_nobase")])
    plan = veg_plan(store, families=("vegetation_geometry",))

    result = kernel.run(plan)
    assert result.result_class == "visual_pending"
    assert result.payload.get("reason") == "no_exact_base"
    assert "planes" not in result.payload, "never standalone planes"


def test_kernel_eligibility_priority_discipline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Compensation is the LAST resort: met-only epochs (served exactly by
    the exact-lane fast paths) and view epochs (fast_exact) are NOT
    compensated."""
    store, clock, scheduler = compensated_setup(tmp_path, monkeypatch)
    kernel = make_kernel(store)

    met_plan = veg_plan(store, families=("meteorological_forcing",))
    view_plan = veg_plan(store, families=("output_view",))
    mixed_plan = veg_plan(
        store, families=("vegetation_geometry", "meteorological_forcing")
    )

    assert kernel.eligible(met_plan) is False
    assert kernel.eligible(view_plan) is False
    assert kernel.eligible(mixed_plan) is False, (
        "a met+vegetation epoch is served by the exact lane's met paths, "
        "not by vegetation compensation"
    )
    met_result = kernel.run(met_plan)
    assert met_result.result_class == "visual_pending"
    assert met_result.payload.get("reason") == "ineligible_families"

    # View epochs keep the structural fast_exact path.
    assert ViewCacheKernel().classify(view_plan) == "fast_exact"


def test_kernel_ignores_physically_identical_trees(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An epoch whose vegetation state equals the anchor state produces NO
    delta — the kernel must not fabricate coverage."""
    store, clock, scheduler = compensated_setup(tmp_path, monkeypatch)
    kernel = make_kernel(store)
    # Re-add the SAME tree the anchor already contains (generation bump,
    # identical fields): physically no change.
    base_state = store.canonical_state_at(WS, 1)
    base_tree = next(iter(base_state.families["vegetation_geometry"]["objects"]))
    drive_epoch(
        store,
        scheduler,
        clock,
        [
            op_item(
                "fq_k_same",
                entity_id=base_tree,
                payload={"values": {"x_m": 1.0, "y_m": 2.0, "height_m": 8.0}},
            )
        ],
    )
    plan = veg_plan(store, families=("vegetation_geometry",))
    result = kernel.run(plan)
    assert result.result_class == "visual_pending"
    assert result.payload.get("reason") == "no_vegetation_delta"


# ---------------------------------------------------------------------------
# The qualification validator (unit)
# ---------------------------------------------------------------------------


def test_qualifier_defects_validator_contract() -> None:
    from solweig_gpu.server.realtime.qualification import qualifier_defects

    assert qualifier_defects(complete_qualified_payload()) == ()
    assert "missing_qualifier" in qualifier_defects({"kernel": "bare"})
    assert "missing_anchor" in qualifier_defects(complete_qualified_payload(exact_base_revision=None))
    assert "invalid_anchor" in qualifier_defects(complete_qualified_payload(exact_base_revision=-3))
    # A non-qualified payload is not the validator's problem (class first).
    assert qualifier_defects({}) == ("missing_qualifier",)
