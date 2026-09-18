# SPDX-License-Identifier: GPL-3.0-only
"""T15 selected-time streaming (``SOLWEIG_RT_SELECTED_TIME_STREAMING``).

The SEPARATE flag: every behavior in this module is additive and gated on
the flag; the flag-off contract is pinned byte-identical (no manifest
keys, no guard, no endpoint, no stream parameter).

Coverage model (DESIGN §11.4/§15): a publication produced under a
client's requested time subset is a REQUEST-CUT record — its uncovered
times were computed and then discarded, so no later composition may
silently treat them as unchanged. Such a record discloses
``time_coverage.complete=false`` (witness: full-coverage false claim),
the composition guard refuses histories with unhealed request-cut gaps
(witness: accepted operation loss — the follow-up solve heals by
recomputing full coverage instead of publishing silently-wrong planes),
and the coverage surface (endpoint + ``selected_time`` SSE frames)
reports exactly the times the server can serve bitwise-correctly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from solweig_gpu.server import patch_codec
from solweig_gpu.server.jobs import (
    RunnerContext,
    SolveResult,
    _compose_current_state,
    materialize_result,
)
from solweig_gpu.server.realtime import selected_time
from solweig_gpu.server.store import Store

from tests.test_incremental_worker import science_site  # noqa: F401  (fixture)
from tests.test_server_api import SITE_ID, TIME_STEPS, make_site_cache

FLAG = selected_time.ENV_FLAG_NAME


# ---------------------------------------------------------------------------
# The flag itself
# ---------------------------------------------------------------------------


def test_flag_defaults_off_and_env_enables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(FLAG, raising=False)
    assert selected_time.enabled() is False
    for value in ("1", "true", "True", "YES", "on"):
        monkeypatch.setenv(FLAG, value)
        assert selected_time.enabled() is True, value
    for value in ("", "0", "false", "maybe"):
        monkeypatch.setenv(FLAG, value)
        assert selected_time.enabled() is False, value


# ---------------------------------------------------------------------------
# Manifest disclosure (witness: full-coverage false claim)
# ---------------------------------------------------------------------------


def _result_context(tmp_path: Path) -> RunnerContext:
    return RunnerContext(
        store=Store(tmp_path / "store.sqlite3", results_root=tmp_path / "scenarios"),
        sites=_FakeSites(make_site_cache(tmp_path)),
        state_root=tmp_path / "state",
    )


class _FakeSites:
    """SiteRegistry stand-in exposing geometry + manifest for SITE_ID."""

    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = cache_dir

    def geometry(self, site_id: str) -> dict[str, int | float]:
        return {"rows": 32, "cols": 48, "time_steps": TIME_STEPS}

    def manifest(self, site_id: str) -> "_FakeSites.Manifest":
        return _FakeSites.Manifest()

    def site_cache_version(self, site_id: str) -> str:
        return "test-cache"

    class Manifest:
        model_version = "test-model"


def _subset_result() -> SolveResult:
    arrays = {
        "utci": np.full((1, 4, 4), 30.0, dtype=np.float32),
        "tmrt": np.full((1, 4, 4), 50.0, dtype=np.float32),
    }
    from solweig_gpu.incremental.geometry import RasterWindow

    return SolveResult(
        status="published",
        scene_version=1,
        mode="local",
        window=RasterWindow(0, 4, 0, 4),
        time_indices=(1,),
        variables=("utci", "tmrt"),
        arrays=arrays,
        requested_time_subset=True,
    )


def test_request_cut_result_discloses_incomplete_time_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(FLAG, "1")
    context = _result_context(tmp_path)
    context.store.create_scenario(site_id=SITE_ID, name="t15", scenario_id="ws_cut")
    scenario = context.store.require_scenario("ws_cut")
    manifest, _payload = materialize_result(
        context, scenario, _subset_result(), duration_ms=1.0
    )
    block = manifest.get("time_coverage")
    assert block is not None, "a request-cut publication must disclose its coverage"
    assert block["complete"] is False
    assert block["time_indices"] == [1]
    assert block["time_steps"] == TIME_STEPS


def test_full_result_and_flag_off_keep_the_manifest_byte_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flag-off pin: no ``time_coverage`` key ever, and a full-coverage
    result never grows the block even with the flag on (the block exists
    to mark REQUEST-CUT records, not to restate full ones)."""
    from solweig_gpu.incremental.geometry import RasterWindow

    context = _result_context(tmp_path)
    context.store.create_scenario(site_id=SITE_ID, name="t15", scenario_id="ws_full")
    scenario = context.store.require_scenario("ws_full")
    full = SolveResult(
        status="published",
        scene_version=1,
        mode="full",
        window=RasterWindow(0, 32, 0, 48),
        time_indices=tuple(range(TIME_STEPS)),
        variables=("utci",),
        arrays={"utci": np.zeros((TIME_STEPS, 32, 48), dtype=np.float32)},
    )
    monkeypatch.delenv(FLAG, raising=False)
    manifest, _ = materialize_result(context, scenario, full, duration_ms=1.0)
    assert "time_coverage" not in manifest
    monkeypatch.setenv(FLAG, "1")
    manifest_on, _ = materialize_result(context, scenario, full, duration_ms=1.0)
    assert "time_coverage" not in manifest_on
    # The subset result with the flag OFF must not grow the key either.
    monkeypatch.delenv(FLAG, raising=False)
    subset = _subset_result()
    manifest_off, _ = materialize_result(context, scenario, subset, duration_ms=1.0)
    assert "time_coverage" not in manifest_off


# ---------------------------------------------------------------------------
# The composition guard (witness: accepted operation loss)
# ---------------------------------------------------------------------------


def _publish_record(
    store: Store,
    workspace: str,
    version: int,
    *,
    times: tuple[int, ...],
    window: dict | None = None,
    request_cut: bool,
    variables: tuple[str, ...] = ("utci",),
) -> None:
    from solweig_gpu.incremental.geometry import RasterWindow

    window_dict = window or {
        "row_start": 0,
        "row_stop": 32,
        "col_start": 0,
        "col_stop": 48,
    }
    rows = int(window_dict["row_stop"]) - int(window_dict["row_start"])
    cols = int(window_dict["col_stop"]) - int(window_dict["col_start"])
    arrays = {
        name: np.full((len(times), rows, cols), 1.0 + version, dtype=np.float32)
        for name in variables
    }
    meta, payload, checksum = patch_codec.encode_payload(arrays, list(variables))
    manifest = patch_codec.build_manifest(
        scenario_id=workspace,
        scene_version=version,
        window=RasterWindow(
            int(window_dict["row_start"]),
            int(window_dict["row_stop"]),
            int(window_dict["col_start"]),
            int(window_dict["col_stop"]),
        ),
        time_indices=times,
        variables=meta,
        payload_url=f"/api/v1/scenarios/{workspace}/results/{version}/payload",
        checksum=checksum,
        model_version="test-model",
        site_cache_version="test-cache",
    )
    if request_cut:
        manifest["time_coverage"] = selected_time.disclosure_block(
            list(times), TIME_STEPS
        )
    # publish_result fences result version == scenario.scene_version; a
    # synthetic multi-version history advances the scenario row directly.
    with store._write() as conn:
        conn.execute(
            "UPDATE scenarios SET scene_version = ? WHERE scenario_id = ?",
            (int(version), workspace),
        )
    store.publish_result(
        workspace, version, manifest=manifest, payload=payload, exact=True
    )


def _compose(tmp_path: Path, records: list[dict], variables=("utci",), tag="g") -> dict:
    """Build a store carrying ``records`` (oldest first) and compose v_last."""
    from solweig_gpu.incremental.geometry import RasterWindow

    root = tmp_path / f"compose-{tag}"
    store = Store(root / "store.sqlite3", results_root=root / "scenarios")
    store.create_scenario(site_id=SITE_ID, name="t15", scenario_id="ws_g")
    for record in records:
        _publish_record(store, "ws_g", **record)
    context = RunnerContext(
        store=store,
        sites=_FakeSites(make_site_cache(tmp_path)),
        state_root=root / "state",
    )
    grid = {"time_steps": TIME_STEPS, "rows": 32, "cols": 48}
    return _compose_current_state(
        context,
        "ws_g",
        grid,
        variables,
        RasterWindow(0, 32, 0, 48),
        records[-1]["version"],
    )


def test_guard_refuses_a_history_with_an_unhealed_request_cut(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RED witness (accepted operation loss): composing a history whose
    latest revision 1 was REQUEST-CUT to t=1 (nothing after it healed)
    would silently keep revision 1's uncovered times at their stale
    values — the accepted edit's effect at those times is dropped from
    every later composition. With the flag on, the guard refuses with a
    typed gap naming the missing times; the solve path then heals by
    recomputing full coverage."""
    monkeypatch.setenv(FLAG, "1")
    records = [
        {"version": 0, "times": tuple(range(TIME_STEPS)), "request_cut": False},
        {"version": 1, "times": (1,), "request_cut": True},
    ]
    with pytest.raises(selected_time.TimeCoverageGap) as gap:
        _compose(tmp_path, records)
    assert sorted(gap.value.missing_times) == [t for t in range(TIME_STEPS) if t != 1]
    assert gap.value.variables == ("utci",)


def test_guard_heals_after_a_full_window_full_time_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(FLAG, "1")
    records = [
        {"version": 0, "times": tuple(range(TIME_STEPS)), "request_cut": False},
        {"version": 1, "times": (1,), "request_cut": True},
        # The heal: a later record covering the FULL window and every time
        # step recomputed the scene, including revision 1's discarded times.
        {"version": 2, "times": tuple(range(TIME_STEPS)), "request_cut": False},
        {
            "version": 3,
            "times": tuple(range(TIME_STEPS)),
            "request_cut": False,
            "window": {"row_start": 0, "row_stop": 4, "col_start": 0, "col_stop": 4},
        },
    ]
    # Version 2 is full-window/full-time (heals); version 3 is a later
    # local record that must compose normally afterwards.
    state = _compose(tmp_path, records[:3], tag="heal3")
    assert state["utci"].shape[0] == TIME_STEPS
    state4 = _compose(tmp_path, records, tag="heal4")
    assert state4 is not None


def test_partial_window_full_time_record_does_not_heal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only a FULL-WINDOW record heals: a local record covering every time
    step but a quarter of the site leaves the cut revision's lost times
    broken everywhere else."""
    monkeypatch.setenv(FLAG, "1")
    quarter = {"row_start": 0, "row_stop": 16, "col_start": 0, "col_stop": 24}
    records = [
        {"version": 0, "times": tuple(range(TIME_STEPS)), "request_cut": False},
        {"version": 1, "times": (1,), "request_cut": True},
        {
            "version": 2,
            "times": tuple(range(TIME_STEPS)),
            "request_cut": False,
            "window": quarter,
        },
    ]
    with pytest.raises(selected_time.TimeCoverageGap):
        _compose(tmp_path, records)


def test_flag_off_composes_the_same_history_without_a_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flag-off pin: the identical history composes exactly as before
    (legacy semantics preserved; the guard is the flag's addition)."""
    monkeypatch.delenv(FLAG, raising=False)
    records = [
        {"version": 0, "times": tuple(range(TIME_STEPS)), "request_cut": True},
    ]
    state = _compose(tmp_path, records)
    assert state["utci"].shape == (TIME_STEPS, 32, 48)


def test_native_sparse_records_are_not_request_cut(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker-native sparse patch (the r3a met fast path: its time
    indices ARE its exact change set, no ``time_coverage`` block) is safe
    and must never trip the guard — only request-cut records do."""
    monkeypatch.setenv(FLAG, "1")
    records = [
        {"version": 0, "times": tuple(range(TIME_STEPS)), "request_cut": False},
        {"version": 1, "times": (1,), "request_cut": False},
    ]
    state = _compose(tmp_path, records)
    assert state["utci"].shape[0] == TIME_STEPS


# ---------------------------------------------------------------------------
# Coverage report + streaming surface
# ---------------------------------------------------------------------------


def _coverage_store(tmp_path: Path) -> Store:
    store = Store(tmp_path / "store.sqlite3", results_root=tmp_path / "scenarios")
    store.create_scenario(site_id=SITE_ID, name="t15", scenario_id="ws_c")
    _publish_record(store, "ws_c", 0, times=tuple(range(TIME_STEPS)), request_cut=False)
    _publish_record(store, "ws_c", 1, times=(1,), request_cut=True)
    return store


def test_coverage_report_names_missing_times(tmp_path: Path) -> None:
    store = _coverage_store(tmp_path)
    report = selected_time.coverage_report(store, "ws_c", ("utci",), TIME_STEPS)
    assert report["time_steps"] == TIME_STEPS
    per_var = report["variables"]["utci"]
    assert per_var["missing_times"] == [t for t in range(TIME_STEPS) if t != 1]
    assert per_var["covered_times"] == [1]
    assert report["complete"] is False


def test_coverage_report_complete_when_no_request_cuts(tmp_path: Path) -> None:
    store = Store(tmp_path / "store.sqlite3", results_root=tmp_path / "scenarios")
    store.create_scenario(site_id=SITE_ID, name="t15", scenario_id="ws_cc")
    _publish_record(
        store, "ws_cc", 0, times=tuple(range(TIME_STEPS)), request_cut=False
    )
    _publish_record(store, "ws_cc", 1, times=(2,), request_cut=False)
    report = selected_time.coverage_report(store, "ws_cc", ("utci",), TIME_STEPS)
    assert report["complete"] is True
    assert report["variables"]["utci"]["missing_times"] == []
    assert sorted(report["variables"]["utci"]["covered_times"]) == list(
        range(TIME_STEPS)
    )


def test_coverage_report_carries_state_age_and_supersession_count(
    tmp_path: Path,
) -> None:
    """Exit-criterion introspection: the newest state's age and the
    workspace's supersession count ride the same report."""
    import sqlite3

    store = Store(tmp_path / "store.sqlite3", results_root=tmp_path / "scenarios")
    store.create_scenario(site_id=SITE_ID, name="t15", scenario_id="ws_age")
    _publish_record(store, "ws_age", 0, times=tuple(range(TIME_STEPS)), request_cut=False)
    report = selected_time.coverage_report(store, "ws_age", ("utci",), TIME_STEPS)
    assert report["latest_result_age_s"] is not None
    assert report["latest_result_age_s"] >= 0.0
    assert report["superseded_jobs"] == 0
    conn = sqlite3.connect(store.db_path)
    try:
        conn.execute(
            "UPDATE jobs SET status = 'superseded' "
            "WHERE scenario_id = 'ws_age' AND status != 'superseded'"
        )
        conn.commit()
    finally:
        conn.close()
    report = selected_time.coverage_report(store, "ws_age", ("utci",), TIME_STEPS)
    assert report["superseded_jobs"] >= 1


def test_initial_frame_splits_requested_into_covered_and_missing() -> None:
    report = {
        "workspace_id": "ws_f",
        "exact_revision": 3,
        "workspace_revision": 5,
        "complete": False,
        "latest_result_age_s": 1.25,
        "variables": {
            "utci": {"missing_times": [0, 2], "covered_times": [1]},
            "tmrt": {"missing_times": [], "covered_times": [0, 1, 2]},
        },
    }
    frame = selected_time.initial_frame(report, (0, 1))
    assert frame["requested"] == [0, 1]
    assert frame["covered"] == [1]
    assert frame["missing"] == [0]
    assert frame["complete"] is False
    assert frame["exact_revision"] == 3
    assert frame["workspace_revision"] == 5
    assert frame["latest_result_age_s"] == 1.25
    assert frame["snapshot"] is True


# ---------------------------------------------------------------------------
# HTTP surface: the flag-gated endpoint + stream parameter
# ---------------------------------------------------------------------------


def _app_client(tmp_path: Path):
    from tests.test_server_api import FakeSolver, make_app

    app = make_app(tmp_path, FakeSolver(), start_worker=False)
    return TestClient(app)


def _workspace(client: TestClient) -> str:
    response = client.post(
        "/api/v1/scenarios", json={"site_id": SITE_ID, "name": "t15"}
    )
    assert response.status_code == 201, response.text
    return response.json()["scenario_id"]


def test_time_coverage_endpoint_flag_off_404s_naming_the_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(FLAG, raising=False)
    client = _app_client(tmp_path)
    ws = _workspace(client)
    response = client.get(f"/api/v1/workspaces/{ws}/time-coverage")
    assert response.status_code == 404
    error = response.json()["error"]
    assert error["code"] == "selected_time_streaming_disabled"
    assert FLAG in error["message"]


def test_time_coverage_endpoint_reports_gaps_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(FLAG, "1")
    client = _app_client(tmp_path)
    ws = _workspace(client)
    store = client.app.state.context.store
    # Scenario creation already publishes the site-cache baseline (full
    # coverage) at its own version; publish above it.
    base = max(store.result_versions(ws), default=-1) + 1
    _publish_record(store, ws, base, times=tuple(range(TIME_STEPS)), request_cut=False)
    _publish_record(store, ws, base + 1, times=(1,), request_cut=True)
    response = client.get(f"/api/v1/workspaces/{ws}/time-coverage")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["complete"] is False
    assert body["variables"]["utci"]["missing_times"] == [
        t for t in range(TIME_STEPS) if t != 1
    ]
    assert body["exact_revision"] == base + 1
    assert body["workspace_revision"] == base + 1
    assert body["latest_result_age_s"] is not None
    assert body["superseded_jobs"] == 0


def test_events_time_indices_param_flag_off_400s_naming_the_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(FLAG, raising=False)
    client = _app_client(tmp_path)
    ws = _workspace(client)
    response = client.get(
        f"/api/v1/workspaces/{ws}/events", params={"time_indices": "1,2"}
    )
    assert response.status_code == 400
    error = response.json()["error"]
    assert FLAG in error["message"]


def test_events_time_indices_validates_against_the_site_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(FLAG, "1")
    client = _app_client(tmp_path)
    ws = _workspace(client)
    for bad in ("abc", "99", "-1", ","):
        response = client.get(
            f"/api/v1/workspaces/{ws}/events", params={"time_indices": bad}
        )
        assert response.status_code == 400, (bad, response.text)
        assert response.json()["error"]["code"] == "invalid_request"


def test_flag_on_wires_demand_tracker_and_epoch_floor_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(FLAG, "1")
    client = _app_client(tmp_path)
    demand = getattr(client.app.state, "selected_time_demand", None)
    assert isinstance(demand, selected_time.SelectedTimeDemand)
    scheduler = client.app.state.epoch_scheduler
    # No demand: the configured cadence stands.
    assert scheduler._window_s() == scheduler.tick_ms / 1000.0
    demand.register("ws_1")
    try:
        # Pending selected-time demand: the window collapses to the
        # contract's 50 ms latency floor (clamped domain).
        assert scheduler._window_s() == 50.0 / 1000.0
    finally:
        demand.release("ws_1")
    assert scheduler._window_s() == scheduler.tick_ms / 1000.0


def test_flag_off_leaves_no_demand_tracker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(FLAG, raising=False)
    client = _app_client(tmp_path)
    assert getattr(client.app.state, "selected_time_demand", None) is None
    scheduler = client.app.state.epoch_scheduler
    assert scheduler._window_s() == scheduler.tick_ms / 1000.0


def test_selected_time_demand_is_presence_only_per_workspace() -> None:
    demand = selected_time.SelectedTimeDemand()
    assert demand.pending() is False
    demand.register("a")
    demand.register("a")
    demand.register("b")
    assert demand.pending() is True
    assert demand.workspaces() == ["a", "b"]
    demand.release("a")
    assert demand.workspaces() == ["a", "b"]
    demand.release("a")
    assert demand.workspaces() == ["b"]
    demand.release("a")  # extra release is safe (idempotent floor)
    demand.release("b")
    assert demand.pending() is False


# ---------------------------------------------------------------------------
# Adaptive epoch window: the hint is clamped into the contract domain
# ---------------------------------------------------------------------------


def test_epoch_window_hint_clamps_into_the_contract_domain(tmp_path: Path) -> None:
    from solweig_gpu.server.realtime.epochs import EpochScheduler

    store = Store(tmp_path / "store.sqlite3", results_root=tmp_path / "scenarios")
    store.create_scenario(site_id=SITE_ID, name="t15", scenario_id="ws_h")
    base = EpochScheduler(store, tick_ms=100.0)
    assert base._window_s() == 0.1
    too_small = EpochScheduler(store, tick_ms=100.0, window_hint_s=lambda: 0.001)
    assert too_small._window_s() == 0.05
    too_big = EpochScheduler(store, tick_ms=100.0, window_hint_s=lambda: 5.0)
    assert too_big._window_s() == 0.2
    none_hint = EpochScheduler(store, tick_ms=100.0, window_hint_s=lambda: None)
    assert none_hint._window_s() == 0.1
    broken = EpochScheduler(store, tick_ms=100.0, window_hint_s=lambda: 1 / 0)
    assert broken._window_s() == 0.1  # a raising hint must not break cadence


def test_selected_time_sse_frame_uses_the_shared_publish_fence() -> None:
    from solweig_gpu.server.realtime.broadcast import BroadcastHub, format_sse

    hub = BroadcastHub()
    subscription = hub.subscribe("ws_s")
    hub.broadcast_selected_time("ws_s", {"complete": True, "requested": [1]})
    frame = subscription._queue.get_nowait()
    assert frame.startswith("event: selected_time\ndata: ")
    assert json.loads(frame.split("data: ", 1)[1].strip()) == {
        "complete": True,
        "requested": [1],
    }


# ---------------------------------------------------------------------------
# The exit differential (scientific): a request-cut publication followed by
# another edit composes to the FRESH full-day oracle, bitwise.
# ---------------------------------------------------------------------------


@pytest.mark.scientific
def test_request_cut_then_follow_up_edit_heals_to_the_fresh_oracle(
    science_site, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE T15 exit gate (witness: accepted operation loss). Sequence: an
    edit served under a strict requested time subset publishes a
    REQUEST-CUT revision (its uncovered times computed then discarded);
    a follow-up edit on the same scenario must NOT compose silently over
    that hole — it recomputes FULL coverage and the final served state is
    raw-bit equal to a fresh full-day solve of the same final scene."""
    from solweig_gpu.server.app import create_app
    from tests.test_incremental_perf_wave1 import (
        _seed_baseline_results,
        _world_to_uv,
    )
    from tests.test_server_api import wait_for_job

    monkeypatch.setenv(FLAG, "1")
    cache = science_site.cache_a
    cache_dir = Path(cache._cache_dir)
    grid = science_site.grid
    _seed_baseline_results(cache_dir, int(cache.time_steps), grid.rows, grid.cols)
    u, v = _world_to_uv(science_site, science_site.add_position_m)
    tree_one = {
        "tree_id": "t15-tree-one",
        "component_type": "broad_canopy",
        "u": u,
        "v": v,
        "height_m": 6.0,
        "canopy_diameter_m": 10.0,
    }
    tree_two = {
        "tree_id": "t15-tree-two",
        "component_type": "broad_canopy",
        "u": u,
        "v": v,
        "height_m": 9.0,
        "canopy_diameter_m": 8.0,
    }

    app = create_app(
        state_root=tmp_path / "state",
        sites={
            "science": {
                "cache_dir": cache_dir,
                "site_dir": science_site.site_dir,
                "selected_date_str": "2024-06-20",
            }
        },
        coalescing_window_ms=50.0,
        requests_per_minute_per_ip=None,
        edits_per_minute=None,
    )
    with TestClient(app) as client:
        store = app.state.context.store

        # (A) request-cut revision 1 ...
        created = client.post("/api/v1/scenarios", json={"site_id": "science"})
        assert created.status_code == 201, created.text
        cut_ws = created.json()["scenario_id"]
        submitted = client.post(
            f"/api/v1/scenarios/{cut_ws}/edits",
            json={
                "base_scene_version": 0,
                "edits": [
                    {"operation": "add", "tree": tree_one},
                ],
                "requested_result": {
                    "time_indices": [12],
                    "variables": ["utci", "tmrt"],
                },
            },
            headers={"Idempotency-Key": "t15-cut"},
        )
        assert submitted.status_code == 202, submitted.text
        cut_detail = wait_for_job(client, submitted.json()["job_id"], timeout=300.0)
        assert cut_detail["status"] == "complete", cut_detail
        cut_record = store.get_result(cut_ws, 1)
        assert cut_record is not None
        assert cut_record.manifest["time_indices"] == [12]
        assert cut_record.manifest["time_coverage"]["complete"] is False

        # ... then the follow-up edit whose compose must refuse the hole.
        follow = client.post(
            f"/api/v1/scenarios/{cut_ws}/edits",
            json={
                "base_scene_version": 1,
                "edits": [
                    {"operation": "add", "tree": tree_two},
                ],
                "requested_result": {"variables": ["utci", "tmrt"]},
            },
            headers={"Idempotency-Key": "t15-follow"},
        )
        assert follow.status_code == 202, follow.text
        follow_detail = wait_for_job(client, follow.json()["job_id"], timeout=300.0)
        assert follow_detail["status"] == "complete", follow_detail
        healed = store.get_result(cut_ws, 2)
        assert healed is not None
        assert list(healed.manifest["time_indices"]) == list(
            range(int(cache.time_steps))
        )
        assert "time_coverage" not in healed.manifest
        assert healed.manifest["metrics"].get("fallback_reason") == (
            "time_coverage_heal"
        )
        arrays_healed = patch_codec.decode_payload(
            healed.manifest, healed.payload_bytes()
        )

        # (B) the fresh oracle: both edits, full day, fresh scenario.
        created = client.post("/api/v1/scenarios", json={"site_id": "science"})
        assert created.status_code == 201, created.text
        oracle_ws = created.json()["scenario_id"]
        submitted = client.post(
            f"/api/v1/scenarios/{oracle_ws}/edits",
            json={
                "base_scene_version": 0,
                "edits": [
                    {"operation": "add", "tree": tree_one},
                    {"operation": "add", "tree": tree_two},
                ],
                "requested_result": {"variables": ["utci", "tmrt"]},
            },
            headers={"Idempotency-Key": "t15-oracle"},
        )
        assert submitted.status_code == 202, submitted.text
        oracle_detail = wait_for_job(client, submitted.json()["job_id"], timeout=300.0)
        assert oracle_detail["status"] == "complete", oracle_detail
        oracle = store.get_result(oracle_ws, 1)
        assert oracle is not None
        arrays_oracle = patch_codec.decode_payload(
            oracle.manifest, oracle.payload_bytes()
        )

    # The oracle job served its write WINDOW (the seeded baseline makes
    # both scenarios non-bootstrap, so the fresh solve is a local-window
    # exact result); the healed publication recomputed the full tile.
    # Parity is asserted inside the oracle's window — every cell the
    # oracle can speak for must be raw-bit equal.
    oracle_window = oracle.manifest["window"]
    r0, r1 = int(oracle_window["row_start"]), int(oracle_window["row_stop"])
    c0, c1 = int(oracle_window["col_start"]), int(oracle_window["col_stop"])
    for name in ("utci", "tmrt"):
        healed_window = arrays_healed[name][:, r0:r1, c0:c1]
        assert healed_window.shape == arrays_oracle[name].shape, name
        assert np.array_equal(
            healed_window, arrays_oracle[name], equal_nan=True
        ), (
            f"{name}: the healed composition is not raw-bit equal to the "
            "fresh full-day oracle of the same final scene"
        )
