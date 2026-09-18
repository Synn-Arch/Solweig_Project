# SPDX-License-Identifier: GPL-3.0-only
"""R8b wiring + admission fences: contract-promised knobs and unfenced paths.

Three fences from the R8 review (r5a-review2, N3/N4):

* **epoch tick wiring** (N3) -- service_level_contract.md promises an epoch
  duration "100 ms nominal; configurable 50-200 ms", but ``create_app``
  exposed no path to the scheduler's ``tick_ms`` (the default factory
  always built ``DEFAULT_TICK_MS``). These tests are RED on that tree: a
  50 ms config must reach the REAL scheduler AND materially change close
  cadence, and out-of-domain values must refuse loudly (49/200.5 ms are
  outside the contract's own domain).
* **batch-size-exceeded admission** (N4) -- a batch larger than
  ``max_operations_per_epoch`` is refused 429 typed with envelope details
  BEFORE the durable append (zero phantom rows), and the gate recovers.
* **capacity-saturated admission** (N4) -- a saturated fast lane refuses
  503 ``fast_lane_unavailable`` with a retry hint BEFORE the durable
  append, and the app keeps serving (saturation is load-shedding, not
  death).

Both admission paths existed before R8b but had no fences: mutating them
out (deleting the ``len(items)`` check / the ``saturated()`` branch) left
every suite green. The fence proof is mutation-run: see the R8b report.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from solweig_gpu.server.app import create_app
from solweig_gpu.server.realtime.admission import (
    AdmissionEnvelope,
    AdmissionRejected,
    FastAdmission,
    MIN_RETRY_AFTER_MS,
)
from solweig_gpu.server.realtime.broadcast import BroadcastHub
from solweig_gpu.server.realtime.epochs import EpochScheduler
from solweig_gpu.server.realtime.scheduler import FastLaneScheduler
from solweig_gpu.server.realtime.telemetry import TelemetryRegistry

from tests.test_realtime_operations import make_rt_app, rt_post, route_item
from tests.test_server_api import SITE_ID, FakeSolver, make_site_cache


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


def _workspace(client: TestClient, name: str = "r8b") -> str:
    response = client.post(
        "/api/v1/scenarios",
        json={"site_id": SITE_ID, "name": name, "initial_state": "baseline"},
    )
    assert response.status_code == 201, response.text
    return response.json()["scenario_id"]


def _durable_op_ids(state_root: Path, workspace: str) -> set[str]:
    conn = sqlite3.connect(state_root / "state" / "store.sqlite3")
    try:
        return {
            row[0]
            for row in conn.execute(
                "SELECT operation_id FROM realtime_operations WHERE workspace_id=?",
                (workspace,),
            )
        }
    finally:
        conn.close()


def _quiet_app(
    tmp_path: Path,
    fast_lane_factory: Any = None,
    **kwargs: Any,
):
    """create_app with the REAL default epoch scheduler, isolated telemetry.

    The default epoch scheduler registers standard metrics on the shared
    module registry; a real-thread close here would pollute
    test_realtime_telemetry's singleton count assertions in combined runs,
    so the registry lookup itself is swapped for a private one (the wiring
    under test — ``create_app`` -> default factory -> ``EpochScheduler`` —
    is fully real).
    """
    import solweig_gpu.server.app as app_module

    lane_factory = fast_lane_factory or (
        lambda store, hub: FastLaneScheduler(
            store, hub=hub, telemetry=TelemetryRegistry()
        )
    )
    real_registry = app_module.default_registry

    def _private_registry():
        return TelemetryRegistry()

    app_module.default_registry = _private_registry
    try:
        app = create_app(
            state_root=tmp_path / "state",
            sites={SITE_ID: {"cache_dir": make_site_cache(tmp_path)}},
            solver_factory=lambda context: FakeSolver(),
            requests_per_minute_per_ip=None,
            edits_per_minute=None,
            start_worker=False,
            start_fast_lane=False,
            fast_lane_factory=lane_factory,
            **kwargs,
        )
    finally:
        app_module.default_registry = real_registry
    return app


# ---------------------------------------------------------------------------
# N3: epoch tick wiring (RED on pre-R8b main: create_app has no knob)
# ---------------------------------------------------------------------------


def test_epoch_tick_ms_config_reaches_real_scheduler(tmp_path: Path) -> None:
    app = _quiet_app(tmp_path / "t50", epoch_tick_ms=50.0)
    scheduler = app.state.epoch_scheduler
    assert isinstance(scheduler, EpochScheduler)
    assert scheduler.tick_ms == 50.0
    assert scheduler._window_s() == pytest.approx(0.05)

    default_app = _quiet_app(tmp_path / "tdef")
    assert default_app.state.epoch_scheduler.tick_ms == 100.0

    boundary_low = _quiet_app(tmp_path / "tblow", epoch_tick_ms=50.0)
    assert boundary_low.state.epoch_scheduler.tick_ms == 50.0
    boundary_high = _quiet_app(tmp_path / "tbhigh", epoch_tick_ms=200.0)
    assert boundary_high.state.epoch_scheduler.tick_ms == 200.0


def test_epoch_tick_ms_out_of_domain_refuses_loudly(tmp_path: Path) -> None:
    for bad in (49.0, 200.5, 0.0, -100.0, 10000.0):
        with pytest.raises(ValueError) as excinfo:
            _quiet_app(tmp_path / f"bad{int(bad * 10)}", epoch_tick_ms=bad)
        message = str(excinfo.value)
        assert "epoch_tick_ms" in message
        assert "50" in message and "200" in message, (
            "the refusal must name the contract's own domain (50-200 ms)"
        )


def test_epoch_tick_ms_changes_scheduler_cadence(tmp_path: Path) -> None:
    """50 ms config must change CLOSE CADENCE, not just a config field.

    Floor logic: an epoch closes on the first tick where its age >= the
    window, so closed-epoch rows have ``closed_at - opened_at`` uniform in
    [W, 2W) on a quiet host: mean ~75 ms at W=50 vs ~150 ms at W=100.
    """

    def _mean_close_age_ms(root: Path, tick_ms: float) -> float:
        app = _quiet_app(root, epoch_tick_ms=tick_ms, start_epoch_scheduler=True)
        with TestClient(app) as client:
            workspace = _workspace(client, f"cadence-{int(tick_ms)}")
            for index in range(8):
                response = rt_post(
                    client, workspace, [_view_item(f"cad-{tick_ms}-{index:03d}")]
                )
                assert response.status_code == 200, response.text
                time.sleep(0.03)
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                conn = sqlite3.connect(root / "state" / "store.sqlite3")
                try:
                    rows = conn.execute(
                        "SELECT opened_at, closed_at FROM realtime_epochs "
                        "WHERE workspace_id=? AND closed_at IS NOT NULL",
                        (workspace,),
                    ).fetchall()
                finally:
                    conn.close()
                if rows:
                    break
                time.sleep(0.05)
        assert rows, "no closed epochs were observed"
        from solweig_gpu.server.realtime.epochs import _parse_stamp

        ages_ms = [
            (_parse_stamp(closed) - _parse_stamp(opened)).total_seconds() * 1000.0
            for opened, closed in rows
        ]
        return sum(ages_ms) / len(ages_ms)

    mean_50 = _mean_close_age_ms(tmp_path / "c50", 50.0)
    mean_100 = _mean_close_age_ms(tmp_path / "c100", 100.0)

    assert mean_50 < 105.0, (
        f"a 50 ms tick must close epochs near the 50 ms floor, got {mean_50:.1f} ms"
    )
    assert mean_100 >= 100.0, (
        f"the 100 ms default must hold its own floor, got {mean_100:.1f} ms"
    )
    assert mean_100 - mean_50 >= 40.0, (
        "the config value must materially change scheduler cadence: "
        f"100 ms mean {mean_100:.1f} vs 50 ms mean {mean_50:.1f}"
    )


# ---------------------------------------------------------------------------
# N4: admission fences (paths existed; fences are new and mutation-proven)
# ---------------------------------------------------------------------------


def test_batch_size_exceeded_rejects_typed_before_durable_append(
    tmp_path: Path,
) -> None:
    app = make_rt_app(
        tmp_path,
        FakeSolver(),
        admission_envelope=AdmissionEnvelope(max_operations_per_epoch=4),
    )
    with TestClient(app) as client:
        workspace = _workspace(client, "batchcap")

        oversized = [
            _view_item(f"batchcap-{index:03d}") for index in range(5)
        ]
        response = rt_post(client, workspace, oversized)
        assert response.status_code == 429, response.text
        error = response.json()["error"]
        assert error["code"] == "fast_lane_overloaded"
        assert error["max_operations_per_epoch"] == 4
        assert "retry_after_ms" in error
        assert isinstance(error["advice"], list) and error["advice"]

        # Reject BEFORE acceptance: none of the refused batch's operations
        # may exist in the durable log (zero phantom accepts).
        assert _durable_op_ids(tmp_path, workspace) == set()

        # Recovery: an in-envelope batch is accepted immediately after —
        # the gate sheds load, it does not wedge.
        ok = rt_post(client, workspace, [_view_item("batchcap-ok-000")])
        assert ok.status_code == 200, ok.text
        assert _durable_op_ids(tmp_path, workspace) == {"batchcap-ok-000"}


def test_capacity_saturated_rejects_503_typed_before_durable_append(
    tmp_path: Path,
) -> None:
    class _SaturatedLane:
        """CapacityReporter stub: the fast lane's queue is full."""

        def reserved_ms(self, workspace_id: str) -> float:
            return 0.0

        def saturated(self) -> bool:
            return True

        # app wiring touchpoints (never exercised: the lane never starts
        # and the scheduler is not driven in this test)
        def on_epoch_closed(self, *args: Any, **kwargs: Any) -> None:
            return None

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

    # Unit: the gate maps saturation to the typed 503 with a retry hint.
    admission = FastAdmission(capacity=_SaturatedLane())
    with pytest.raises(AdmissionRejected) as excinfo:
        admission.check(
            "ws_sat", [_view_item("sat-000")], received_at="2026-01-01T00:00:00.000Z"
        )
    assert excinfo.value.status == 503
    assert excinfo.value.code == "fast_lane_unavailable"
    assert excinfo.value.details["retry_after_ms"] >= 2 * MIN_RETRY_AFTER_MS

    # Route: refuse BEFORE the durable append, server stays alive.
    root = tmp_path / "sat-route"
    app = _quiet_app(root, fast_lane_factory=lambda store, hub: _SaturatedLane())
    with TestClient(app) as client:
        workspace = _workspace(client, "sat")
        response = rt_post(client, workspace, [_view_item("sat-route-000")])
        assert response.status_code == 503, response.text
        error = response.json()["error"]
        assert error["code"] == "fast_lane_unavailable"
        assert error["retry_after_ms"] >= 2 * MIN_RETRY_AFTER_MS
        assert _durable_op_ids(root, workspace) == set()
        health = client.get("/healthz")
        assert health.status_code == 200, (
            "saturation is load-shedding, not death: the app keeps serving"
        )
