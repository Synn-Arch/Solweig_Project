# SPDX-License-Identifier: GPL-3.0-only
"""HTTP end-to-end: the universal edit transport against the REAL executor (u-d4).

``POST /api/v1/scenarios/{id}/edits/universal`` carries the four integrated
families (meteorology, landcover, buildings, model parameters) onto the
executor path. These tests run the REAL solver stack — the tiny prepared
site, a REAL full-domain baseline published through the site cache, and the
default dispatch solver — and prove:

* per-family E2E products match the full-domain oracle through the
  sanctioned materialization seams (``run_full_tile(overlay=...)``,
  ``run_full_tile(model_parameters=...)``, ``run_full_tile(
  landcover_overlay=...)``) — bitwise, never a relaxed tolerance;
* batch semantics: one request = one job = one version bump; mixed
  family batches mark every edited source node in the executed plan;
* typed refusals: the landcover water fence (class 7) relays as a 422
  naming the fence, vegetation is refused with its named transport,
  unknown adapters are refused against the capability document;
* transport discipline: Idempotency-Key replay, optimistic If-Match /
  base-scene-version conflicts, view-only output_view items enqueue
  ZERO scientific jobs, tree edits still travel POST /edits and replay
  onto the executor AFTER a family edit (snapshot restore + pending
  tree-event replay), and reset voids the family overlays (fresh
  rebuild) exactly like the baseline re-publish contract.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_incremental_worker import (  # noqa: E402
    DATE_STR,
    TINY_EPSG,
    TINY_ORIGIN,
    TINY_PIXEL,
    TINY_TREE,
    _build_cache,
    _compute_baseline_svf,
    _make_prepared_site,
)
from test_server_api import make_tree, wait_for_job  # noqa: E402

from solweig_gpu.incremental.edit_types import SiteContext
from solweig_gpu.incremental.geometry import RasterGrid, RasterWindow
from solweig_gpu.incremental import landcover_overlay_from_deltas, overlay_from_deltas
from solweig_gpu.incremental.solver import load_site_forcing, run_full_tile
from solweig_gpu.incremental.trees import TreeLayer
from solweig_gpu.server import patch_codec, universal
from solweig_gpu.server.app import create_app
from solweig_gpu.server.jobs import api_tree_to_spec
from solweig_gpu.server.realtime.scheduler import FastLaneScheduler
from solweig_gpu.server.realtime.epochs import EpochScheduler
from solweig_gpu.server.realtime.telemetry import TelemetryRegistry

SITE_ID = "tiny"
VARIABLES = ("utci", "tmrt")
#: A paint footprint inside the prepared site's grass region (class 5),
#: matching the lc-int suite's proven footprint.
PAINT_WINDOW = RasterWindow(40, 48, 40, 48)
#: An added building footprint: rows 60:72, cols 60:72 in map coordinates.
NEW_BLOCK = [
    [1120.0, 1856.0],
    [1144.0, 1856.0],
    [1144.0, 1880.0],
    [1120.0, 1880.0],
]


# ---------------------------------------------------------------------------
# One real site + REAL baseline per module (full-domain SOLWEIG run)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def family_site(tmp_path_factory) -> SimpleNamespace:
    """A REAL-SVF prepared site whose baseline_results are a REAL full run.

    The site carries genuine SVF artifacts (``_compute_baseline_svf`` —
    windowed solves read the shadow cubes, so synthetic patch-count
    fixtures would break the anisotropic-sky loop), and the baseline
    arrays come from the standard full-domain path: they are the
    composition base for every family job AND the reference the oracles
    are built against.
    """
    root = tmp_path_factory.mktemp("ud4_family_site")
    grid, site = _make_prepared_site(
        root,
        rows=128,
        cols=128,
        pixel=TINY_PIXEL,
        origin=TINY_ORIGIN,
        epsg=TINY_EPSG,
        base_trees=(TINY_TREE,),
        met_hours=range(10, 13),
    )
    _compute_baseline_svf(site)
    cache_dir = root / "cache"
    cache = _build_cache(
        site,
        cache_dir,
        met_path=site / "metfiles" / f"metfile_0_0_{DATE_STR}.txt",
        site_id=SITE_ID,
    )
    baseline = _full_run(cache, site, grid, root / "baseline_scratch")
    baseline_dir = cache_dir / "baseline_results"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    for name in VARIABLES:
        np.save(baseline_dir / f"{name}.f32.npy", baseline[name])
    (baseline_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "variables": list(VARIABLES),
                "time_steps": int(cache.time_steps),
                "source": "baseline",
            }
        )
    )
    return SimpleNamespace(
        root=root,
        grid=grid,
        site=site,
        cache=cache,
        cache_dir=cache_dir,
        baseline=baseline,
    )


def _full_run(
    cache,
    site: Path,
    grid: RasterGrid,
    scratch: Path,
    *,
    overlay=None,
    model_parameters=None,
    landcover_overlay=None,
    layer: TreeLayer | None = None,
) -> dict[str, np.ndarray]:
    """The full-domain oracle through the sanctioned materialization seams."""
    forcing = load_site_forcing(
        cache, site_dir=site, selected_date_str=DATE_STR, overlay=overlay
    )
    return run_full_tile(
        cache,
        layer if layer is not None else TreeLayer(cache.tree_base, grid),
        forcing=forcing,
        site_dir=site,
        scratch_dir=scratch,
        requested_variables=VARIABLES,
        model_parameters=model_parameters,
        landcover_overlay=landcover_overlay,
    )


def _make_client(tmp_path: Path, family_site, *, extra_sites=None, **kwargs):
    sites = {
        SITE_ID: {
            "cache_dir": family_site.cache_dir,
            "site_dir": family_site.site,
            "selected_date_str": DATE_STR,
        }
    }
    sites.update(extra_sites or {})
    kwargs.setdefault("requests_per_minute_per_ip", None)
    kwargs.setdefault("edits_per_minute", None)
    kwargs.setdefault("coalescing_window_ms", 20.0)
    # Isolated telemetry: create_app's DEFAULT scheduler factories register
    # standard metrics into the module-shared singleton at construction
    # time (app.py _default_epoch_scheduler/_default_fast_lane). This
    # helper's apps are also imported by realtime-block test files that run
    # BEFORE test_realtime_telemetry, whose singleton-purity pin would
    # otherwise fail (merge interaction of r2a-fix default-wiring tests
    # with r2b's default lane wiring). Private registries reproduce the
    # pre-r2b base behavior exactly (base registered nothing at
    # construction); callers may still override either factory.
    kwargs.setdefault(
        "epoch_scheduler_factory",
        lambda store, hub: EpochScheduler(
            store, hub=hub, telemetry=TelemetryRegistry()
        ),
    )
    kwargs.setdefault(
        "fast_lane_factory",
        lambda store, hub: FastLaneScheduler(
            store, hub=hub, telemetry=TelemetryRegistry()
        ),
    )
    app = create_app(state_root=tmp_path / "state", sites=sites, **kwargs)
    return TestClient(app)


@pytest.fixture()
def client(tmp_path: Path, family_site):
    with _make_client(tmp_path, family_site) as test_client:
        yield test_client


def _scenario_ready(
    client: TestClient, site_id: str = SITE_ID, *, wait_exact: bool = True
) -> str:
    response = client.post(
        "/api/v1/scenarios",
        json={"site_id": site_id, "name": "ud4", "initial_state": "baseline"},
    )
    assert response.status_code == 201, response.text
    scenario_id = response.json()["scenario_id"]
    if not wait_exact:
        return scenario_id
    for _ in range(500):
        body = client.get(f"/api/v1/scenarios/{scenario_id}").json()
        if body["exact_result_version"] == body["scene_version"]:
            return scenario_id
        time.sleep(0.01)
    pytest.fail("scenario baseline never published")


def _universal(
    client: TestClient,
    scenario_id: str,
    edits: list[dict],
    base: int,
    *,
    key: str | None = None,
    headers: dict | None = None,
):
    payload = {"base_scene_version": base, "edits": edits}
    request_headers = {"Idempotency-Key": key or f"u-{base}-{len(edits)}-{time.time()}"}
    request_headers.update(headers or {})
    return client.post(
        f"/api/v1/scenarios/{scenario_id}/edits/universal",
        json=payload,
        headers=request_headers,
    )


_JOB_STATUSES = ("queued", "running", "complete", "superseded", "failed", "cancelled")


def _job_count(client: TestClient) -> int:
    jobs = client.get("/metrics").json()["jobs"]
    return sum(int(jobs.get(status, 0)) for status in _JOB_STATUSES)


def _served_arrays(client: TestClient, job: dict) -> dict[str, np.ndarray]:
    manifest = client.get(job["result_manifest_url"]).json()
    payload = client.get(manifest["payload_url"])
    assert payload.status_code == 200, payload.text
    return patch_codec.decode_payload(manifest, payload.content)


def _validated_delta(family_site, item: dict):
    """The item's adapter-validated delta (the SAME mapping the route runs)."""
    command = universal.command_from_item(
        universal.UniversalEditItem(**item),
        scenario_id="oracle",
        scene_revision=0,
        edit_id="oracle-0",
    )
    assert command is not None
    adapter = universal.family_registry_adapter(item["adapter"])
    context = SiteContext(
        site_id=SITE_ID,
        grid=family_site.grid,
        scene_revision=0,
        available_times=tuple(range(family_site.cache.time_steps)),
    )
    return adapter.validate(command, context).delta


def _assert_bitwise(actual: np.ndarray, expected: np.ndarray, note: str) -> None:
    assert actual.shape == expected.shape, f"{note}: shape drift"
    assert np.array_equal(actual, expected, equal_nan=True), (
        f"{note}: not bitwise equal (never a relaxed tolerance)"
    )


def _grid_info(family_site) -> dict:
    cache = family_site.cache
    return {
        "rows": cache.rows,
        "cols": cache.cols,
        "pixel_size_m": cache.pixel_size_m,
        "origin_x_m": family_site.grid.origin_x_m,
        "origin_y_m": family_site.grid.origin_y_m,
        "time_steps": cache.time_steps,
    }


# ---------------------------------------------------------------------------
# Per-family differentials (bitwise vs the full-domain oracle)
# ---------------------------------------------------------------------------


@pytest.mark.scientific
class TestPerFamilyEndToEnd:
    def test_meteorology_edit_matches_oracle_bitwise(
        self, client: TestClient, family_site
    ) -> None:
        scenario_id = _scenario_ready(client)
        edits = [
            {
                "adapter": "meteorological_forcing",
                "operation": "update_time_row",
                "time_index": 1,
                "values": {"air_temperature": 30.0},
            }
        ]
        response = _universal(client, scenario_id, edits, base=0, key="met-1")
        assert response.status_code == 202, response.text
        body = response.json()
        assert body["transport"] == "universal-edits-v1"
        assert body["status_url"].endswith(f"/jobs/{body['job_id']}")
        assert response.headers["ETag"] == '"scene-version-1"'

        job = wait_for_job(client, body["job_id"])
        assert job["status"] == "complete"
        assert job["mode"] == "full"  # forcing edits are site-global
        assert job["metrics"]["window_fraction"] == 1.0
        assert job["metrics"]["transport"] == "executor"

        # The executed plan is surfaced with the family source node marked
        # changed and the geometry caches reused.
        plan = job["impact_plan"]
        assert plan["schema_version"] == 1
        stages = {node["node"]: node["stage"] for node in plan["nodes"]}
        assert stages["meteorology"] == "changed"
        assert "reused" in stages.values()
        assert plan["routing"]["mode"] == "full"
        assert plan["realized"]["published_variables"]

        scenario = client.get(f"/api/v1/scenarios/{scenario_id}").json()
        assert scenario["scene_version"] == 1
        assert scenario["exact_result_version"] == 1
        assert scenario["status"] == "exact"

        delta = _validated_delta(
            family_site,
            {
                "adapter": "meteorological_forcing",
                "operation": "update_time_row",
                "time_index": 1,
                "values": {"air_temperature": 30.0},
            },
        )
        oracle = _full_run(
            family_site.cache,
            family_site.site,
            family_site.grid,
            family_site.root / "oracle_met",
            overlay=overlay_from_deltas([delta]),
        )
        served = _served_arrays(client, job)
        for name in VARIABLES:
            _assert_bitwise(served[name], oracle[name], f"{name}: met transport")
        # The edited timestep actually moved.
        assert not np.array_equal(
            served["utci"][1], family_site.baseline["utci"][1], equal_nan=True
        )

    def test_model_parameters_edit_matches_oracle_bitwise(
        self, client: TestClient, family_site
    ) -> None:
        scenario_id = _scenario_ready(client)
        edits = [
            {
                "adapter": "model_receptor_parameters",
                "operation": "update",
                "values": {"albedo_b": 0.35},
            }
        ]
        response = _universal(client, scenario_id, edits, base=0, key="params-1")
        assert response.status_code == 202, response.text
        job = wait_for_job(client, response.json()["job_id"])
        assert job["status"] == "complete"
        assert job["mode"] == "full"  # receptor parameters are site-global
        stages = {node["node"]: node["stage"] for node in job["impact_plan"]["nodes"]}
        assert stages["model_parameters"] == "changed"

        oracle = _full_run(
            family_site.cache,
            family_site.site,
            family_site.grid,
            family_site.root / "oracle_params",
            model_parameters={"albedo_b": 0.35},
        )
        served = _served_arrays(client, job)
        for name in VARIABLES:
            _assert_bitwise(served[name], oracle[name], f"{name}: params transport")
        assert not np.array_equal(
            served["tmrt"], family_site.baseline["tmrt"], equal_nan=True
        )

    def test_landcover_paint_matches_oracle_inside_window(
        self, client: TestClient, family_site
    ) -> None:
        scenario_id = _scenario_ready(client)
        window = {
            "row_start": PAINT_WINDOW.row_start,
            "row_stop": PAINT_WINDOW.row_stop,
            "col_start": PAINT_WINDOW.col_start,
            "col_stop": PAINT_WINDOW.col_stop,
        }
        edits = [
            {
                "adapter": "landcover_surface",
                "operation": "paint",
                "target": window,
                "values": {"class": 2},
            }
        ]
        response = _universal(client, scenario_id, edits, base=0, key="lc-1")
        assert response.status_code == 202, response.text
        job = wait_for_job(client, response.json()["job_id"])
        assert job["status"] == "complete"
        assert job["mode"] == "local"  # a paint routes windowed
        assert job["metrics"]["window_fraction"] < 1.0
        assert job["window"]["row_start"] <= PAINT_WINDOW.row_start
        assert job["window"]["row_stop"] >= PAINT_WINDOW.row_stop
        stages = {node["node"]: node["stage"] for node in job["impact_plan"]["nodes"]}
        assert stages["landcover"] == "changed"

        delta = _validated_delta(
            family_site,
            {
                "adapter": "landcover_surface",
                "operation": "paint",
                "target": window,
                "values": {"class": 2},
            },
        )
        oracle = _full_run(
            family_site.cache,
            family_site.site,
            family_site.grid,
            family_site.root / "oracle_lc",
            landcover_overlay=landcover_overlay_from_deltas([delta]),
        )
        served = _served_arrays(client, job)
        rows = slice(job["window"]["row_start"], job["window"]["row_stop"])
        cols = slice(job["window"]["col_start"], job["window"]["col_stop"])
        for name in VARIABLES:
            # The served payload carries the write window only (patch
            # semantics); the oracle is sliced to the same footprint.
            _assert_bitwise(
                served[name],
                oracle[name][:, rows, cols],
                f"{name}: paint inside write window",
            )
        # The painted cells changed.
        paint_rows = slice(
            PAINT_WINDOW.row_start - rows.start, PAINT_WINDOW.row_stop - rows.start
        )
        paint_cols = slice(
            PAINT_WINDOW.col_start - cols.start, PAINT_WINDOW.col_stop - cols.start
        )
        assert not np.array_equal(
            served["utci"][:, paint_rows, paint_cols],
            family_site.baseline["utci"][:, rows, cols][:, paint_rows, paint_cols],
            equal_nan=True,
        )

    def test_building_add_routes_full_and_marks_building_dsm(
        self, client: TestClient, family_site
    ) -> None:
        """Building edits are structural here: the massing chain's physics
        differential lives in test_incremental_building_integration; the
        transport asserts routing, plan attribution, and publication."""
        scenario_id = _scenario_ready(client)
        edits = [
            {
                "adapter": "building_geometry",
                "operation": "add",
                "target": "block-a",
                "values": {"footprint_m": NEW_BLOCK, "height_m": 12.0},
            }
        ]
        response = _universal(client, scenario_id, edits, base=0, key="bldg-1")
        assert response.status_code == 202, response.text
        job = wait_for_job(client, response.json()["job_id"], timeout=60.0)
        assert job["status"] == "complete"
        assert job["mode"] == "full"  # geometry edits invalidate the caches
        stages = {node["node"]: node["stage"] for node in job["impact_plan"]["nodes"]}
        assert stages["building_dsm"] == "changed"
        served = _served_arrays(client, job)
        for name in VARIABLES:
            assert served[name].shape == family_site.baseline[name].shape
            assert served[name].dtype == np.float32
        # The new block changes the microclimate around it.
        block_rows = slice(58, 74)
        block_cols = slice(58, 74)
        assert not np.array_equal(
            served["utci"][:, block_rows, block_cols],
            family_site.baseline["utci"][:, block_rows, block_cols],
            equal_nan=True,
        )

    def test_building_then_met_edit_keeps_the_building_bitwise(
        self, client: TestClient, family_site
    ) -> None:
        """u-d4c, user-facing: a met edit AFTER a published building edit
        keeps the building. The served products match a twin full-domain
        oracle (an independently-run ``regenerate_building_batch`` carrying
        the building AND the met overlay) bitwise — the pre-u-d4c engine
        served the met-only baseline scene here, silently reverting the
        massing."""
        from solweig_gpu.incremental.adapters.building import (
            massing_edits_from_deltas,
        )
        from solweig_gpu.incremental.regenerate import regenerate_building_batch

        building = [
            {
                "adapter": "building_geometry",
                "operation": "add",
                "target": "block-a",
                "values": {"footprint_m": NEW_BLOCK, "height_m": 12.0},
            }
        ]
        response = _universal(client, scenario_id := _scenario_ready(client), building, base=0, key="bldg-a")
        assert response.status_code == 202, response.text
        assert wait_for_job(
            client, response.json()["job_id"], timeout=60.0
        )["status"] == "complete"

        met = [
            {
                "adapter": "meteorological_forcing",
                "operation": "update_time_row",
                "time_index": 1,
                "values": {"air_temperature": 30.0},
            }
        ]
        response = _universal(client, scenario_id, met, base=1, key="met-after-bldg")
        assert response.status_code == 202, response.text
        job = wait_for_job(client, response.json()["job_id"], timeout=60.0)
        assert job["status"] == "complete"
        # Chain-routed with the accumulated fold (u-d4c): full tile through
        # the regeneration chain, never the baseline-bound worker.
        assert job["mode"] == "full"
        assert job["metrics"]["transport"] == "executor"

        # The twin oracle: the SAME adapter-validated building delta the
        # route committed, plus the met overlay, through one regeneration.
        building_delta = _validated_delta(family_site, building[0])
        met_delta = _validated_delta(family_site, met[0])
        twin = regenerate_building_batch(
            edits=massing_edits_from_deltas((building_delta,)),
            baseline_site_dir=family_site.site,
            baseline_cache=family_site.cache,
            scenario_root=family_site.root / "twin_bldg_met" / "scenario",
            selected_date_str=DATE_STR,
            forcing_overlay=overlay_from_deltas([met_delta]),
        )
        served = _served_arrays(client, job)
        for name in VARIABLES:
            _assert_bitwise(served[name], twin.outputs[name], f"{name}: building->met")

    def test_mixed_family_batch_is_one_job_marking_both_sources(
        self, client: TestClient, family_site
    ) -> None:
        scenario_id = _scenario_ready(client)
        edits = [
            {
                "adapter": "meteorological_forcing",
                "operation": "update_time_row",
                "time_index": 2,
                "values": {"humidity": 55.0},
            },
            {
                "adapter": "model_receptor_parameters",
                "operation": "update",
                "values": {"albedo_b": 0.30},
            },
        ]
        response = _universal(client, scenario_id, edits, base=0, key="mixed-1")
        assert response.status_code == 202, response.text
        body = response.json()
        jobs_before = _job_count(client)
        job = wait_for_job(client, body["job_id"])
        # ONE job for the whole batch.
        assert _job_count(client) == jobs_before
        scenario = client.get(f"/api/v1/scenarios/{scenario_id}").json()
        assert scenario["scene_version"] == 1  # ONE version bump
        stages = {node["node"]: node["stage"] for node in job["impact_plan"]["nodes"]}
        assert stages["meteorology"] == "changed"
        assert stages["model_parameters"] == "changed"

        met_delta = _validated_delta(
            family_site,
            {
                "adapter": "meteorological_forcing",
                "operation": "update_time_row",
                "time_index": 2,
                "values": {"humidity": 55.0},
            },
        )
        oracle = _full_run(
            family_site.cache,
            family_site.site,
            family_site.grid,
            family_site.root / "oracle_mixed",
            overlay=overlay_from_deltas([met_delta]),
            model_parameters={"albedo_b": 0.30},
        )
        served = _served_arrays(client, job)
        for name in VARIABLES:
            _assert_bitwise(served[name], oracle[name], f"{name}: mixed batch")


# ---------------------------------------------------------------------------
# Typed refusals (validators are the single source of truth)
# ---------------------------------------------------------------------------


class TestTypedRefusals:
    def test_landcover_water_fence_relays_as_422_naming_the_fence(
        self, client: TestClient
    ) -> None:
        scenario_id = _scenario_ready(client)
        jobs_before = _job_count(client)
        edits = [
            {
                "adapter": "landcover_surface",
                "operation": "paint",
                "target": {
                    "row_start": 8,
                    "row_stop": 16,
                    "col_start": 6,
                    "col_stop": 14,
                },
                "values": {"class": 7},  # water: fenced by lead ruling
            }
        ]
        response = _universal(client, scenario_id, edits, base=0, key="water-1")
        assert response.status_code == 422, response.text
        error = response.json()["error"]
        assert error["code"] == "invalid_edit_state"
        # The adapter's fence text, relayed verbatim (never restated).
        assert "class 7 (water) is fenced" in error["message"]
        assert "lead ruling" in error["message"]
        assert error["adapter"] == "landcover_surface"
        assert error["operation"] == "paint"
        # Refused before any store mutation: no job, no version bump.
        assert _job_count(client) == jobs_before
        assert client.get(f"/api/v1/scenarios/{scenario_id}").json()[
            "scene_version"
        ] == 0

    def test_vegetation_is_refused_with_its_named_transport(
        self, client: TestClient
    ) -> None:
        scenario_id = _scenario_ready(client)
        jobs_before = _job_count(client)
        edits = [
            {
                "adapter": "vegetation_geometry",
                "operation": "add",
                "values": {"x_m": 1.0, "y_m": 1.0, "height_m": 3.0},
            }
        ]
        response = _universal(client, scenario_id, edits, base=0, key="veg-1")
        assert response.status_code == 400, response.text
        error = response.json()["error"]
        assert error["code"] == "unsupported_transport"
        assert "tree-edits-v1" in error["message"]
        assert _job_count(client) == jobs_before

    def test_unknown_adapter_is_refused_against_capabilities(
        self, client: TestClient
    ) -> None:
        scenario_id = _scenario_ready(client)
        edits = [
            {"adapter": "terrain_dem", "operation": "sculpt", "values": {"delta": 1.0}}
        ]
        response = _universal(client, scenario_id, edits, base=0, key="terrain-1")
        assert response.status_code == 400, response.text
        error = response.json()["error"]
        assert error["code"] == "unsupported_transport"
        assert "capabilities" in error["message"]

    def test_building_edit_requires_an_id_target(self, client: TestClient) -> None:
        scenario_id = _scenario_ready(client)
        edits = [
            {
                "adapter": "building_geometry",
                "operation": "add",
                "values": {"footprint_m": NEW_BLOCK, "height_m": 12.0},
            }
        ]
        response = _universal(client, scenario_id, edits, base=0, key="bldg-notarget")
        assert response.status_code == 400, response.text
        assert response.json()["error"]["field"] == "edits[0].target"


# ---------------------------------------------------------------------------
# Transport discipline: idempotency, optimistic versioning, views
# ---------------------------------------------------------------------------


class TestTransportDiscipline:
    def test_idempotent_replay_returns_the_stored_response(
        self, client: TestClient
    ) -> None:
        scenario_id = _scenario_ready(client)
        edits = [
            {
                "adapter": "model_receptor_parameters",
                "operation": "update",
                "values": {"albedo_b": 0.25},
            }
        ]
        first = _universal(client, scenario_id, edits, base=0, key="replay-1")
        assert first.status_code == 202, first.text
        second = _universal(client, scenario_id, edits, base=0, key="replay-1")
        assert second.status_code == 202, second.text
        assert second.json() == first.json()
        assert second.headers["ETag"] == first.headers["ETag"]
        # One job, one version bump — the replay never re-enqueued.
        wait_for_job(client, first.json()["job_id"])
        scenario = client.get(f"/api/v1/scenarios/{scenario_id}").json()
        assert scenario["scene_version"] == 1

    def test_idempotency_key_reuse_with_different_body_conflicts(
        self, client: TestClient
    ) -> None:
        scenario_id = _scenario_ready(client)
        first_edits = [
            {
                "adapter": "model_receptor_parameters",
                "operation": "update",
                "values": {"albedo_b": 0.25},
            }
        ]
        other_edits = [
            {
                "adapter": "model_receptor_parameters",
                "operation": "update",
                "values": {"albedo_b": 0.45},
            }
        ]
        assert (
            _universal(client, scenario_id, first_edits, base=0, key="reuse-1").status_code
            == 202
        )
        response = _universal(client, scenario_id, other_edits, base=0, key="reuse-1")
        assert response.status_code == 409, response.text
        assert response.json()["error"]["code"] == "idempotency_key_reused"

    def test_stale_base_version_conflicts(self, client: TestClient) -> None:
        scenario_id = _scenario_ready(client)
        edits = [
            {
                "adapter": "meteorological_forcing",
                "operation": "update_time_row",
                "time_index": 0,
                "values": {"air_temperature": 25.0},
            }
        ]
        first = _universal(client, scenario_id, edits, base=0, key="stale-1")
        wait_for_job(client, first.json()["job_id"])
        stale = _universal(client, scenario_id, edits, base=0, key="stale-2")
        assert stale.status_code == 409, stale.text
        assert stale.json()["error"]["code"] == "scene_version_conflict"

    def test_if_match_and_base_disagreement_is_rejected(
        self, client: TestClient
    ) -> None:
        scenario_id = _scenario_ready(client)
        edits = [
            {
                "adapter": "meteorological_forcing",
                "operation": "update_time_row",
                "time_index": 0,
                "values": {"air_temperature": 25.0},
            }
        ]
        response = _universal(
            client,
            scenario_id,
            edits,
            base=0,
            key="ifmatch-1",
            headers={"If-Match": '"scene-version-5"'},
        )
        assert response.status_code == 400, response.text
        assert response.json()["error"]["code"] == "invalid_request"

    def test_if_match_precondition_replays_after_stale(
        self, client: TestClient
    ) -> None:
        scenario_id = _scenario_ready(client)
        edits = [
            {
                "adapter": "meteorological_forcing",
                "operation": "update_time_row",
                "time_index": 0,
                "values": {"air_temperature": 25.0},
            }
        ]
        first = _universal(
            client,
            scenario_id,
            edits,
            base=0,
            key="ifmatch-ok",
            headers={"If-Match": '"scene-version-0"'},
        )
        assert first.status_code == 202, first.text
        wait_for_job(client, first.json()["job_id"])
        stale = _universal(
            client,
            scenario_id,
            edits,
            base=0,
            key="ifmatch-ok-2",
            headers={"If-Match": '"scene-version-0"'},
        )
        assert stale.status_code == 409, stale.text
        assert stale.json()["error"]["code"] == "scene_version_conflict"


# ---------------------------------------------------------------------------
# View-only guarantee (UEDIT-007 holds on the universal transport too)
# ---------------------------------------------------------------------------


class TestViewOnlyItems:
    def test_output_view_item_answers_view_only_zero_jobs(
        self, client: TestClient, family_site
    ) -> None:
        scenario_id = _scenario_ready(client)
        # Publish a family edit first so a non-baseline result exists.
        edits = [
            {
                "adapter": "meteorological_forcing",
                "operation": "update_time_row",
                "time_index": 1,
                "values": {"air_temperature": 30.0},
            }
        ]
        job = wait_for_job(
            client, _universal(client, scenario_id, edits, base=0, key="view-pre").json()["job_id"]
        )
        assert job["status"] == "complete"

        jobs_before = _job_count(client)
        version_before = client.get(f"/api/v1/scenarios/{scenario_id}").json()[
            "scene_version"
        ]
        view_edits = [
            {
                "adapter": "output_view",
                "operation": "select_layer",
                "values": {"layer": "utci", "time_index": 1},
            }
        ]
        response = _universal(client, scenario_id, view_edits, base=1, key="view-1")
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["layer"]["name"] == "utci"
        assert payload["zero_scientific_jobs"] is True
        assert payload["job_enqueued"] is False
        # ZERO scientific jobs and no scene mutation.
        assert _job_count(client) == jobs_before
        assert (
            client.get(f"/api/v1/scenarios/{scenario_id}").json()["scene_version"]
            == version_before
        )

    def test_output_view_item_must_be_alone_in_the_batch(
        self, client: TestClient
    ) -> None:
        scenario_id = _scenario_ready(client)
        edits = [
            {
                "adapter": "output_view",
                "operation": "select_layer",
                "values": {"layer": "utci", "time_index": 0},
            },
            {
                "adapter": "model_receptor_parameters",
                "operation": "update",
                "values": {"albedo_b": 0.25},
            },
        ]
        response = _universal(client, scenario_id, edits, base=0, key="view-mixed")
        assert response.status_code == 400, response.text
        assert response.json()["error"]["code"] == "invalid_family_batch"

    def test_unknown_view_operation_is_rejected(self, client: TestClient) -> None:
        scenario_id = _scenario_ready(client)
        edits = [
            {
                "adapter": "output_view",
                "operation": "select_layers",  # not in the registry
                "values": {"layer": "utci", "time_index": 0},
            }
        ]
        response = _universal(client, scenario_id, edits, base=0, key="view-bad")
        assert response.status_code == 400, response.text
        assert response.json()["error"]["code"] == "invalid_request"


# ---------------------------------------------------------------------------
# Ledger interplay: tree edits, reset, and rebuild failures
# ---------------------------------------------------------------------------


@pytest.mark.scientific
class TestLedgerInterplay:
    def test_tree_edit_after_family_edit_replays_on_the_executor(
        self, client: TestClient, family_site
    ) -> None:
        """The second family-scenario job exercises the snapshot restore:
        the first job snapshotted at sequence 1, the tree edit arrives as
        ledger sequence 2 and replays as a vegetation command ON TOP of
        the restored family state."""
        scenario_id = _scenario_ready(client)
        met_edits = [
            {
                "adapter": "meteorological_forcing",
                "operation": "update_time_row",
                "time_index": 1,
                "values": {"air_temperature": 30.0},
            }
        ]
        first = _universal(client, scenario_id, met_edits, base=0, key="mix-1")
        wait_for_job(client, first.json()["job_id"])

        tree = make_tree("tree_xx", u=0.42, v=0.48)
        tree_response = client.post(
            f"/api/v1/scenarios/{scenario_id}/edits",
            json={
                "base_scene_version": 1,
                "edits": [{"operation": "add", "tree": tree}],
                "requested_result": {"time_indices": [1], "variables": ["utci", "tmrt"]},
            },
            headers={"Idempotency-Key": "tree-after-family"},
        )
        assert tree_response.status_code == 202, tree_response.text
        job = wait_for_job(client, tree_response.json()["job_id"])
        assert job["status"] == "complete"
        assert job["metrics"]["transport"] == "executor"
        scenario = client.get(f"/api/v1/scenarios/{scenario_id}").json()
        assert scenario["scene_version"] == 2
        assert scenario["exact_result_version"] == 2
        assert len(scenario["trees"]) == 1  # tree authority stays on /edits

        # The served union equals the full-domain oracle over the SAME
        # met overlay and tree, inside the tree job's write window.
        met_delta = _validated_delta(
            family_site,
            {
                "adapter": "meteorological_forcing",
                "operation": "update_time_row",
                "time_index": 1,
                "values": {"air_temperature": 30.0},
            },
        )
        layer = TreeLayer(family_site.cache.tree_base, family_site.grid)
        layer.add_tree(api_tree_to_spec(tree, _grid_info(family_site)))
        oracle = _full_run(
            family_site.cache,
            family_site.site,
            family_site.grid,
            family_site.root / "oracle_tree_after",
            overlay=overlay_from_deltas([met_delta]),
            layer=layer,
        )
        served = _served_arrays(client, job)
        window = job["window"]
        rows = slice(window["row_start"], window["row_stop"])
        cols = slice(window["col_start"], window["col_stop"])
        for name in VARIABLES:
            # The tree payload is window-only at the requested time index
            # [1]: served plane 0 IS the t=1 slice the oracle compares
            # against, sliced to the same write window.
            _assert_bitwise(
                served[name][0],
                oracle[name][1][rows, cols],
                f"{name}: tree-after-family inside write window",
            )

    def test_reset_after_family_edit_republishes_the_baseline(
        self, client: TestClient, family_site
    ) -> None:
        scenario_id = _scenario_ready(client)
        edits = [
            {
                "adapter": "meteorological_forcing",
                "operation": "update_time_row",
                "time_index": 1,
                "values": {"air_temperature": 30.0},
            }
        ]
        first = _universal(client, scenario_id, edits, base=0, key="reset-pre")
        wait_for_job(client, first.json()["job_id"])
        edited = client.get(f"/api/v1/scenarios/{scenario_id}").json()
        assert edited["scene_version"] == 1

        reset = client.post(
            f"/api/v1/scenarios/{scenario_id}/reset",
            json={"base_scene_version": 1},
            headers={"Idempotency-Key": "reset-1"},
        )
        assert reset.status_code == 200, reset.text
        scenario = client.get(f"/api/v1/scenarios/{scenario_id}").json()
        assert scenario["trees"] == []
        for _ in range(500):
            scenario = client.get(f"/api/v1/scenarios/{scenario_id}").json()
            if scenario["exact_result_version"] == scenario["scene_version"]:
                break
            time.sleep(0.01)
        assert scenario["exact_result_version"] == scenario["scene_version"]

        # A family edit AFTER the reset rebuilds fresh (no stale overlay):
        # bitwise the same oracle as the very first edit on the scenario.
        again = _universal(client, scenario_id, edits, base=scenario["scene_version"], key="reset-post")
        job = wait_for_job(client, again.json()["job_id"])
        assert job["status"] == "complete"
        met_delta = _validated_delta(
            family_site,
            {
                "adapter": "meteorological_forcing",
                "operation": "update_time_row",
                "time_index": 1,
                "values": {"air_temperature": 30.0},
            },
        )
        oracle = _full_run(
            family_site.cache,
            family_site.site,
            family_site.grid,
            family_site.root / "oracle_reset",
            overlay=overlay_from_deltas([met_delta]),
        )
        served = _served_arrays(client, job)
        for name in VARIABLES:
            _assert_bitwise(served[name], oracle[name], f"{name}: post-reset rebuild")

    def test_family_edit_on_site_without_stored_baseline_bootstraps(
        self, tmp_path: Path, family_site
    ) -> None:
        """A site with no stored baseline outputs still converges: scenario
        creation enqueues a baseline full-solve job (the legacy worker path,
        tree-only ledger), it publishes v0, and the family edit composes on
        top — bitwise the same oracle as the stored-baseline deployment.
        (The bridge's typed ``scenario_state_unrecoverable`` refusal guards
        the no-published-result corner defensively; the public API always
        bootstraps through the baseline job first.)"""
        cache_copy = tmp_path / "cache_nobase"
        cache_nobase = _build_cache(
            family_site.site,
            cache_copy,
            met_path=family_site.site / "metfiles" / f"metfile_0_0_{DATE_STR}.txt",
            site_id="nobase",
        )
        assert not (cache_copy / "baseline_results").exists()
        assert cache_nobase.site_id == "nobase"
        sites = {
            "nobase": {
                "cache_dir": cache_copy,
                "site_dir": family_site.site,
                "selected_date_str": DATE_STR,
            }
        }
        with _make_client(tmp_path, family_site, extra_sites=sites) as client:
            scenario_id = _scenario_ready(client, site_id="nobase", wait_exact=False)
            body = client.get(f"/api/v1/scenarios/{scenario_id}").json()
            assert body["status"] == "refining"  # baseline job in flight
            edits = [
                {
                    "adapter": "meteorological_forcing",
                    "operation": "update_time_row",
                    "time_index": 1,
                    "values": {"air_temperature": 30.0},
                }
            ]
            response = _universal(client, scenario_id, edits, base=0, key="nobase-1")
            assert response.status_code == 202, response.text
            job = wait_for_job(
                client, response.json()["job_id"], timeout=60.0
            )
            assert job["status"] == "complete"
            for _ in range(500):
                scenario = client.get(f"/api/v1/scenarios/{scenario_id}").json()
                if scenario["exact_result_version"] == scenario["scene_version"]:
                    break
                time.sleep(0.01)
            assert scenario["exact_result_version"] == 1

            met_delta = _validated_delta(
                family_site,
                {
                    "adapter": "meteorological_forcing",
                    "operation": "update_time_row",
                    "time_index": 1,
                    "values": {"air_temperature": 30.0},
                },
            )
            oracle = _full_run(
                family_site.cache,
                family_site.site,
                family_site.grid,
                family_site.root / "oracle_nobase",
                overlay=overlay_from_deltas([met_delta]),
            )
            served = _served_arrays(client, job)
            for name in VARIABLES:
                _assert_bitwise(
                    served[name], oracle[name], f"{name}: no-baseline bootstrap"
                )


# ---------------------------------------------------------------------------
# Retention durability (u-e1 F1): the routing signal must survive sweeps
# ---------------------------------------------------------------------------


@pytest.mark.scientific
class TestRetentionDurableRouting:
    def test_tree_edit_after_retention_prune_keeps_family_state_bitwise(
        self, client: TestClient, family_site
    ) -> None:
        """u-e1 F1 regression: once ledger retention prunes the scenario's
        family events past the audit tail, a later TREE edit must still
        route the executor and publish the ACCUMULATED family state.

        On the unfixed code the routing predicate read the retention-pruned
        ledger (``any(event.family)`` over ``store.list_events``), so the
        pruned scenario silently re-routed onto the legacy baseline-bound
        solver and published a window WITHOUT the meteorological overlay —
        bitwise-equal to the wrong (no-met) twin while the correct twin
        diverged across the whole window. The sweep below uses the
        PRODUCTION delete predicate (``sequence <= acked_sequence - tail``)
        with the tail compressed to 0; only the volume differs.
        """
        scenario_id = _scenario_ready(client)
        met_edits = [
            {
                "adapter": "meteorological_forcing",
                "operation": "update_time_row",
                "time_index": 1,
                "values": {"air_temperature": 30.0},
            }
        ]
        first = _universal(client, scenario_id, met_edits, base=0, key="ret-1")
        assert first.status_code == 202, first.text
        job1 = wait_for_job(client, first.json()["job_id"])
        assert job1["status"] == "complete"
        assert job1["metrics"]["transport"] == "executor"

        # Retention sweep with production delete semantics.
        store = client.app.state.context.store
        assert [e for e in store.list_events(scenario_id) if e.get("family")]
        deleted = store.sweep_retention(event_tail=0)
        assert deleted["edit_events"] >= 1
        assert not [
            e for e in store.list_events(scenario_id) if e.get("family")
        ], "family events survived the sweep — the repro is invalid"

        # A TREE edit through the tree transport: WHICH solver runs now?
        tree = make_tree("tree_ret", u=0.42, v=0.48)
        tree_response = client.post(
            f"/api/v1/scenarios/{scenario_id}/edits",
            json={
                "base_scene_version": 1,
                "edits": [{"operation": "add", "tree": tree}],
                "requested_result": {"time_indices": [1], "variables": ["utci", "tmrt"]},
            },
            headers={"Idempotency-Key": "ret-tree"},
        )
        assert tree_response.status_code == 202, tree_response.text
        job2 = wait_for_job(client, tree_response.json()["job_id"])
        assert job2["status"] == "complete"
        assert job2["metrics"]["transport"] == "executor", (
            "the pruned scenario re-routed onto the legacy solver (u-e1 F1)"
        )

        # Twin oracles: the same tree with and without the met overlay.
        met_delta = _validated_delta(family_site, met_edits[0])
        layer = TreeLayer(family_site.cache.tree_base, family_site.grid)
        layer.add_tree(api_tree_to_spec(tree, _grid_info(family_site)))
        with_met = _full_run(
            family_site.cache,
            family_site.site,
            family_site.grid,
            family_site.root / "oracle_ret_met",
            overlay=overlay_from_deltas([met_delta]),
            layer=layer,
        )
        without_met = _full_run(
            family_site.cache,
            family_site.site,
            family_site.grid,
            family_site.root / "oracle_ret_bare",
            layer=layer,
        )
        served = _served_arrays(client, job2)
        window = job2["window"]
        rows = slice(window["row_start"], window["row_stop"])
        cols = slice(window["col_start"], window["col_stop"])
        for name in VARIABLES:
            got = served[name][0]
            # The correct twin, bitwise — the accumulated family state is
            # still carried inside the tree job's write window.
            _assert_bitwise(
                got,
                with_met[name][1][rows, cols],
                f"{name}: tree-after-prune vs WITH-met twin",
            )
            # Non-vacuous: the wrong (no-met) twin — what the legacy
            # re-route published — must DIFFER inside the window.
            wrong = without_met[name][1][rows, cols]
            assert not np.array_equal(
                got, wrong, equal_nan=True
            ), f"{name}: the twins agree — the met edit has no in-window effect, so the differential is vacuous"


# ---------------------------------------------------------------------------
# Typed failure envelopes (u-e1 F2/F4): corrupt snapshots and engine
# refusals must surface typed, never a generic job_failed
# ---------------------------------------------------------------------------


@pytest.mark.scientific
class TestTypedFailureEnvelopes:
    def test_torn_snapshot_fails_typed_idempotently_and_reset_heals(
        self, client: TestClient, family_site
    ) -> None:
        """u-e1 F2 regression: a corrupt CURRENT-generation executor
        snapshot wedges every later family job — the refusal must carry
        the bridge's typed ``scenario_state_unrecoverable`` envelope (its
        own docstring vocabulary), fail identically on repeat, and heal
        after reset. On the unfixed code ScenarioStateError escaped the
        bridge's catch list and surfaced as a generic
        ``{"code": "job_failed"}``.
        """
        scenario_id = _scenario_ready(client)
        first = _universal(client, scenario_id, [_MET_T1_30], base=0, key="torn-1")
        assert wait_for_job(client, first.json()["job_id"])["status"] == "complete"
        second = _universal(
            client, scenario_id, [_paint(PAINT_WINDOW)], base=1, key="torn-2"
        )
        assert wait_for_job(client, second.json()["job_id"])["status"] == "complete"

        # Corrupt EVERY snapshot the next job could trust or adopt: the
        # sidecar-named generation AND the staged pending (a corrupt OLD
        # generation self-heals via pending rotation, so both must go).
        state_dir = (
            client.app.state.context.store.results_root
            / scenario_id
            / "executor-state"
        )
        targets = sorted(state_dir.rglob("scenario-state.json"))
        assert targets, "no executor snapshots on disk — the repro is invalid"
        for path in targets:
            original = path.read_bytes()
            path.write_bytes(original[: len(original) // 2])

        third = _universal(client, scenario_id, [_MET_T0_25], base=2, key="torn-3")
        assert third.status_code == 202, third.text
        job3 = _wait_terminal(client, third.json()["job_id"])
        assert job3["status"] != "complete"
        assert job3["error"]["code"] == "scenario_state_unrecoverable", job3["error"]

        # Idempotent failure: the same deterministic corruption, the same
        # typed refusal — no envelope drift between retries.
        fourth = _universal(client, scenario_id, [_MET_T0_25], base=3, key="torn-4")
        assert fourth.status_code == 202, fourth.text
        job4 = _wait_terminal(client, fourth.json()["job_id"])
        assert job4["status"] != "complete"
        assert job4["error"]["code"] == "scenario_state_unrecoverable", job4["error"]

        # Reset heals: the restore gate (covered <= last_reset) bypasses
        # the corrupt snapshot and the family path publishes again.
        current = client.get(f"/api/v1/scenarios/{scenario_id}").json()
        reset = client.post(
            f"/api/v1/scenarios/{scenario_id}/reset",
            json={"base_scene_version": current["scene_version"]},
            headers={"Idempotency-Key": "torn-reset"},
        )
        assert reset.status_code == 200, reset.text
        for _ in range(500):
            body = client.get(f"/api/v1/scenarios/{scenario_id}").json()
            if body["exact_result_version"] == body["scene_version"]:
                break
            time.sleep(0.01)
        fifth = _universal(
            client, scenario_id, [_MET_T1_30], body["scene_version"], key="torn-5"
        )
        assert fifth.status_code == 202, fifth.text
        job5 = wait_for_job(client, fifth.json()["job_id"])
        assert job5["status"] == "complete"
        assert job5["metrics"]["transport"] == "executor"

    def test_executor_error_surfaces_typed(
        self, client: TestClient, monkeypatch
    ) -> None:
        """u-e1 F4: a deterministic ExecutorError (inconsistent worker
        state, uncovered scope) must surface as a typed engine refusal,
        not an untyped job_failed."""
        from solweig_gpu.incremental.executor import ExecutorError, PlanExecutor

        def refuse(self, commands):
            raise ExecutorError("probe: nothing pending for this batch")

        monkeypatch.setattr(PlanExecutor, "execute", refuse)
        scenario_id = _scenario_ready(client)
        response = _universal(
            client, scenario_id, [_MET_T1_30], base=0, key="f4-executor"
        )
        assert response.status_code == 202, response.text
        job = _wait_terminal(client, response.json()["job_id"])
        assert job["status"] != "complete"
        assert job["error"]["code"] == "engine_refused", job["error"]
        assert "nothing pending" in job["error"]["message"]

    def test_solver_input_error_surfaces_typed(
        self, client: TestClient, monkeypatch
    ) -> None:
        """u-e1 F4: a deterministic SolverInputError (the solver refusing
        its own inputs mid-batch) must surface as a typed engine refusal
        too — the same envelope vocabulary, a distinct failure source."""
        from solweig_gpu.incremental.executor import PlanExecutor
        from solweig_gpu.incremental.solver import SolverInputError

        def refuse(self, commands):
            raise SolverInputError("probe: unsafe local job input")

        monkeypatch.setattr(PlanExecutor, "execute", refuse)
        scenario_id = _scenario_ready(client)
        response = _universal(
            client, scenario_id, [_MET_T1_30], base=0, key="f4-solver"
        )
        assert response.status_code == 202, response.text
        job = _wait_terminal(client, response.json()["job_id"])
        assert job["status"] != "complete"
        assert job["error"]["code"] == "engine_refused", job["error"]
        assert "unsafe local job input" in job["error"]["message"]


# ---------------------------------------------------------------------------
# Bootstrap shape fence (u-e1 F5): a mis-built stored baseline must be a
# typed refusal, never a silent NaN tail
# ---------------------------------------------------------------------------


class TestBootstrapShapeFence:
    def test_wrong_shape_stored_baseline_is_refused_typed(self, tmp_path: Path) -> None:
        """u-e1 F5: ``_bootstrap_state`` assigned stored baseline arrays with
        ``full[: array.shape[0]] = array`` and NO shape validation — a
        mis-built cache (here: one timestep short) padded the composition
        with a silent NaN tail instead of refusing."""
        from solweig_gpu.server.executor_bridge import _Unrecoverable, _bootstrap_state

        baseline_dir = tmp_path / "baseline_results"
        baseline_dir.mkdir()
        # utci one time step short of the site's series; tmrt correct —
        # the corruption must be caught per-array, not by accident.
        np.save(baseline_dir / "utci.f32.npy", np.zeros((2, 8, 8), np.float32))
        np.save(baseline_dir / "tmrt.f32.npy", np.zeros((3, 8, 8), np.float32))
        context = SimpleNamespace(
            sites=SimpleNamespace(
                config=lambda site_id: SimpleNamespace(
                    cache_dir=tmp_path, site_dir=None
                )
            ),
            store=SimpleNamespace(results_root=tmp_path / "results"),
        )
        cache = SimpleNamespace(time_steps=3)
        grid = RasterGrid(rows=8, cols=8, pixel_size_m=2.0, origin_x_m=0.0, origin_y_m=0.0)
        request = SimpleNamespace(
            site_id="tiny", scenario_id="scn_probe", trees=(), grid=grid
        )
        with pytest.raises(_Unrecoverable, match="utci"):
            _bootstrap_state(
                context, request, cache, grid, ("utci", "tmrt"), ()
            )


# ---------------------------------------------------------------------------
# Transport grammar fences (u-e2 M1/M2/M3): the transport relays verbatim
# and refuses ambiguity typed — it never rewrites, coerces, or drops
# ---------------------------------------------------------------------------


class TestTransportGrammarFences:
    """u-e2 M1: an undeclared operation must surface as a typed 4xx
    naming the operation, for EVERY family. At base, landcover/met/
    building escaped as AdapterRegistryError (HTTP 500) and params was
    silently rewritten to ``update`` so the typo never surfaced at all."""

    def test_undeclared_operation_refused_typed_per_family(
        self, client: TestClient
    ) -> None:
        scenario_id = _scenario_ready(client)
        paint_target = {
            "row_start": 40,
            "row_stop": 48,
            "col_start": 40,
            "col_stop": 48,
        }
        cases = [
            ("landcover_surface", "sculpt", {"class": 5}, paint_target),
            ("meteorological_forcing", "update", {"air_temperature": 30.0}, None),
            (
                "building_geometry",
                "grow",
                {"height_m": 12.0, "footprint_m": NEW_BLOCK},
                "bldg-new",
            ),
            # The params typo is the sharpest probe: at base the transport
            # rewrote ANY non-reset op to "update", so "updaet" was
            # silently accepted as a parameter update.
            ("model_receptor_parameters", "updaet", {"albedo_b": 0.3}, None),
        ]
        for number, (adapter, operation, values, target) in enumerate(cases):
            edits = [
                {
                    "adapter": adapter,
                    "operation": operation,
                    "values": values,
                    "target": target,
                }
            ]
            response = _universal(
                client, scenario_id, edits, base=0, key=f"m1-{number}"
            )
            assert response.status_code == 400, (operation, response.text)
            error = response.json()["error"]
            assert error["code"] == "invalid_request", (operation, error)
            assert "does not support operation" in error["message"], error
            assert operation in error["message"], error
            assert error["field"] == "edits[0].operation", error

    def test_values_time_index_accepts_only_json_integers(
        self, client: TestClient
    ) -> None:
        """u-e2 M2: ``values.time_index`` is the DANGEROUS coercion site —
        the transport's ``int()`` ran before the met adapter's timestep
        fence saw the original value, so ``5.9`` became a silent edit at
        timestep 5 and ``True`` became timestep 1."""
        scenario_id = _scenario_ready(client)
        for number, bad in enumerate([5.9, True, "07", "abc", [5]]):
            edits = [
                {
                    "adapter": "meteorological_forcing",
                    "operation": "update_time_row",
                    "values": {"time_index": bad, "air_temperature": 30.0},
                }
            ]
            response = _universal(
                client, scenario_id, edits, base=0, key=f"m2-{number}"
            )
            assert response.status_code == 400, (bad, response.text)
            error = response.json()["error"]
            assert error["code"] == "invalid_request", (bad, error)
            assert error["field"] == "edits[0].values.time_index", (bad, error)
            assert "integer" in error["message"], (bad, error)
        # A real JSON integer still travels: the edit enqueues.
        response = _universal(
            client,
            scenario_id,
            [
                {
                    "adapter": "meteorological_forcing",
                    "operation": "update_time_row",
                    "values": {"time_index": 1, "air_temperature": 30.0},
                }
            ],
            base=0,
            key="m2-ok",
        )
        assert response.status_code == 202, response.text

    def test_item_time_index_accepts_only_json_integers(
        self, client: TestClient
    ) -> None:
        """u-e3-review NIT-2 (item level of u-e2 M2): pydantic lax mode
        coerced ``True``->1, ``5.0``->5 and ``"07"``->7 at the MODEL
        boundary before the transport's strict fence ever saw the original
        value — ``True`` was a silent wrong-timestep edit, the exact class
        the values-level fence exists to kill."""
        scenario_id = _scenario_ready(client)
        for number, bad in enumerate([5.0, True, "07"]):
            edits = [
                {
                    "adapter": "meteorological_forcing",
                    "operation": "update_time_row",
                    "values": {"air_temperature": 30.0},
                    "time_index": bad,
                }
            ]
            response = _universal(
                client, scenario_id, edits, base=0, key=f"m2-item-{number}"
            )
            # Model-boundary refusal: the app's RequestValidationError
            # envelope (400 invalid_request naming the field), not a silent
            # coercion and never a 500.
            assert response.status_code == 400, (bad, response.text)
            error = response.json()["error"]
            assert error["code"] == "invalid_request"
            assert error["field"].endswith("time_index"), error["field"]

    def test_met_time_index_ambiguity_refused(
        self, client: TestClient
    ) -> None:
        """u-e2 M3: two time_index sources, or a time_index alongside a
        range, were silently resolved by dropping one of them — the
        client could not know which timestep the edit landed on."""
        scenario_id = _scenario_ready(client)
        duplicate = [
            {
                "adapter": "meteorological_forcing",
                "operation": "update_time_row",
                "time_index": 0,
                "values": {"time_index": 1, "air_temperature": 30.0},
            }
        ]
        response = _universal(client, scenario_id, duplicate, base=0, key="m3-dup")
        assert response.status_code == 400, response.text
        error = response.json()["error"]
        assert error["code"] == "invalid_request", error
        assert "time_index" in error["message"], error

        stray = [
            {
                "adapter": "meteorological_forcing",
                "operation": "update_range",
                "time_index": 1,
                "values": {
                    "time_start": 0,
                    "time_stop": 1,
                    "values": {"air_temperature": 30.0},
                },
            }
        ]
        response = _universal(client, scenario_id, stray, base=0, key="m3-stray")
        assert response.status_code == 400, response.text
        error = response.json()["error"]
        assert error["code"] == "invalid_request", error
        assert "time_index" in error["message"], error

        # Without the stray, the range edit is unambiguous and travels.
        clean = [
            {
                "adapter": "meteorological_forcing",
                "operation": "update_range",
                "values": {
                    "time_start": 0,
                    "time_stop": 1,
                    "values": {"air_temperature": 30.0},
                },
            }
        ]
        response = _universal(client, scenario_id, clean, base=0, key="m3-range")
        assert response.status_code == 202, response.text

    def test_building_id_in_values_must_match_target(
        self, client: TestClient
    ) -> None:
        """u-e2 M3: ``values.building_id``/``old_values.building_id``
        silently overrode the edit target — a request aimed at tower-A
        could mutate tower-B's state payload."""
        scenario_id = _scenario_ready(client)
        overrides = [
            # values.building_id overrides the target in new_state:
            {
                "adapter": "building_geometry",
                "operation": "add",
                "target": "bldg-new",
                "values": {
                    "building_id": "bldg-other",
                    "height_m": 12.0,
                    "footprint_m": NEW_BLOCK,
                },
            },
            # old_values.building_id overrides the target in old_state:
            {
                "adapter": "building_geometry",
                "operation": "update",
                "target": "bldg-new",
                "values": {"height_m": 14.0},
                "old_values": {
                    "building_id": "bldg-other",
                    "height_m": 12.0,
                },
            },
        ]
        for number, edit in enumerate(overrides):
            response = _universal(
                client, scenario_id, [edit], base=0, key=f"m3-bldg-{number}"
            )
            assert response.status_code == 400, (number, response.text)
            error = response.json()["error"]
            assert error["code"] == "invalid_request", (number, error)
            assert "building_id" in error["message"], (number, error)
            expected_field = (
                "edits[0].old_values" if edit.get("old_values") else "edits[0].values"
            )
            assert error["field"] == expected_field, (number, error)

    def test_landcover_paint_unknown_values_key_refused(
        self, client: TestClient
    ) -> None:
        """u-e2 M3: the paint grammar is exactly ``values={"class"}``;
        at base any other key was silently dropped — a typo'd class field
        would paint the DEFAULT class instead of refusing."""
        scenario_id = _scenario_ready(client)
        edits = [
            {
                "adapter": "landcover_surface",
                "operation": "paint",
                "target": {
                    "row_start": 40,
                    "row_stop": 48,
                    "col_start": 40,
                    "col_stop": 48,
                },
                "values": {"class": 5, "water_please": 7},
            }
        ]
        response = _universal(client, scenario_id, edits, base=0, key="m3-lc")
        assert response.status_code == 400, response.text
        error = response.json()["error"]
        assert error["code"] == "invalid_request", error
        assert "water_please" in error["message"], error
        assert error["field"] == "edits[0].values", error

    def test_output_view_values_time_index_strict(
        self, client: TestClient, family_site
    ) -> None:
        """u-e2 M2 (view path): the view-only branch repeated the same
        coercion (``int(values.get("time_index") or 0)``) — "07" served
        plane 7 and a missing key silently meant 0."""
        scenario_id = _scenario_ready(client)
        first = _universal(client, scenario_id, [_MET_T1_30], base=0, key="m2-view-pre")
        wait_for_job(client, first.json()["job_id"])
        bad = [
            {
                "adapter": "output_view",
                "operation": "select_layer",
                "values": {"layer": "utci", "time_index": "07"},
            }
        ]
        response = _universal(client, scenario_id, bad, base=1, key="m2-view-bad")
        assert response.status_code == 400, response.text
        error = response.json()["error"]
        assert error["code"] == "invalid_request", error
        assert "time_index" in error["message"], error
        # A JSON integer still serves the view.
        good = [
            {
                "adapter": "output_view",
                "operation": "select_layer",
                "values": {"layer": "utci", "time_index": 1},
            }
        ]
        response = _universal(client, scenario_id, good, base=1, key="m2-view-ok")
        assert response.status_code == 200, response.text
        assert response.json()["zero_scientific_jobs"] is True


# ---------------------------------------------------------------------------
# Discarded-job convergence (u-d4 remediation H1): R1/R2/R3 interleavings
# ---------------------------------------------------------------------------

_TERMINAL_STATUSES = ("complete", "superseded", "failed", "cancelled")


class _Gate:
    """One-shot fault seam: block the runner thread inside the solver."""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def wait_entered(self, timeout: float = 120.0) -> None:
        assert self.entered.wait(timeout), "solver never reached the fault seam"

    def open(self) -> None:
        self.release.set()


def _gate_load_patch(monkeypatch, gate: _Gate) -> None:
    """Block the bridge AFTER the executor published internally, BEFORE
    any durable scenario-state staging — the seam where a concurrent
    commit / cancel must still converge (R1, R3)."""
    from solweig_gpu.server import executor_bridge

    original = executor_bridge.load_patch

    def gated(path):
        if not gate.entered.is_set():
            gate.entered.set()
            assert gate.release.wait(180.0), "test never released the gate"
        return original(path)

    monkeypatch.setattr(executor_bridge, "load_patch", gated)


def _wait_terminal(client: TestClient, job_id: str, timeout: float = 180.0) -> dict:
    deadline = time.monotonic() + timeout
    last: dict = {}
    while time.monotonic() < deadline:
        last = client.get(f"/api/v1/jobs/{job_id}").json()
        if last["status"] in _TERMINAL_STATUSES:
            return last
        time.sleep(0.02)
    pytest.fail(f"job {job_id} never reached a terminal status; last: {last}")


def _wait_fault_fired(client: TestClient, job_id: str, timeout: float = 180.0) -> None:
    """The injected store fault fired: the job row is stuck ``running``
    while the runner itself has gone idle (the exception was logged and
    the loop moved on — exactly the durable residue a process crash at
    the same seam leaves behind)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = client.get(f"/api/v1/jobs/{job_id}").json()["status"]
        if status in _TERMINAL_STATUSES:
            pytest.fail(f"job {job_id} reached {status} before the fault fired")
        worker = client.get("/health/worker").json()
        if status == "running" and worker.get("idle"):
            return
        time.sleep(0.05)
    pytest.fail("injected fault never fired (job never ran / runner never idled)")


def _restartable_app(family_site, state_root: Path):
    app = create_app(
        state_root=state_root,
        sites={
            SITE_ID: {
                "cache_dir": family_site.cache_dir,
                "site_dir": family_site.site,
                "selected_date_str": DATE_STR,
            }
        },
        requests_per_minute_per_ip=None,
        edits_per_minute=None,
        coalescing_window_ms=20.0,
    )
    return app, TestClient(app)


_MET_T1_30 = {
    "adapter": "meteorological_forcing",
    "operation": "update_time_row",
    "time_index": 1,
    "values": {"air_temperature": 30.0},
}
_MET_T0_25 = {
    "adapter": "meteorological_forcing",
    "operation": "update_time_row",
    "time_index": 0,
    "values": {"air_temperature": 25.0},
}


def _paint(window: RasterWindow, *, klass: int = 2) -> dict:
    return {
        "adapter": "landcover_surface",
        "operation": "paint",
        "target": {
            "row_start": window.row_start,
            "row_stop": window.row_stop,
            "col_start": window.col_start,
            "col_stop": window.col_stop,
        },
        "values": {"class": klass},
    }


@pytest.mark.scientific
class TestDiscardedJobConvergence:
    """The interleavings serialization hid (u-d4 remediation H1): a family
    job's result can be discarded (concurrent edit, cancel, crash) AFTER
    its solve consumed the ledger window — the durable executor state
    must NOT have advanced past such a job, so the NEXT job replays its
    events and published results converge exactly."""

    def test_R1_concurrent_edit_discard_replays_on_the_next_job(
        self, client: TestClient, family_site, monkeypatch
    ) -> None:
        """No wait_for_job serialization: edit 2 commits while edit 1's
        job is still solving. Job 1's result is stale at finalize and
        discarded; job 2 must replay BOTH windowed paints (the discarded
        edit's window included) — a follow-up that skipped the discarded
        paint would drop its cells from the published composition."""
        scenario_id = _scenario_ready(client)
        paint_a = RasterWindow(40, 48, 40, 48)
        paint_b = RasterWindow(72, 80, 8, 16)  # disjoint from A
        gate = _Gate()
        _gate_load_patch(monkeypatch, gate)
        first = _universal(client, scenario_id, [_paint(paint_a)], base=0, key="r1-1")
        assert first.status_code == 202, first.text
        job1_id = first.json()["job_id"]
        gate.wait_entered()  # job 1 is mid-solve, past its executor batch

        # A concurrent commit lands while job 1 solves (base = v1: edit 1
        # is already committed to the ledger).
        second = _universal(client, scenario_id, [_paint(paint_b)], base=1, key="r1-2")
        assert second.status_code == 202, second.text
        gate.open()

        job1 = _wait_terminal(client, job1_id)
        assert job1["status"] == "superseded", job1
        job2 = wait_for_job(client, second.json()["job_id"], timeout=180.0)
        assert job2["status"] == "complete", job2
        assert job2["metrics"]["transport"] == "executor"

        scenario = client.get(f"/api/v1/scenarios/{scenario_id}").json()
        assert scenario["scene_version"] == 2
        assert scenario["exact_result_version"] == 2  # caught up exactly

        # Job 2's write window covers BOTH paints (it replayed the
        # discarded job 1's window); code that advanced durable coverage
        # past the discarded job publishes paint B's window alone.
        window = job2["window"]
        assert window["row_start"] <= min(paint_a.row_start, paint_b.row_start)
        assert window["row_stop"] >= max(paint_a.row_stop, paint_b.row_stop)
        assert window["col_start"] <= min(paint_a.col_start, paint_b.col_start)
        assert window["col_stop"] >= max(paint_a.col_stop, paint_b.col_stop)

        delta_a = _validated_delta(family_site, _paint(paint_a))
        delta_b = _validated_delta(family_site, _paint(paint_b))
        oracle = _full_run(
            family_site.cache,
            family_site.site,
            family_site.grid,
            family_site.root / "oracle_r1_both",
            landcover_overlay=landcover_overlay_from_deltas([delta_a, delta_b]),
        )
        served = _served_arrays(client, job2)
        rows = slice(window["row_start"], window["row_stop"])
        cols = slice(window["col_start"], window["col_stop"])
        # The payload carries the union write window, all times (patch
        # semantics); the oracle is sliced to the same footprint.
        for name in VARIABLES:
            _assert_bitwise(
                served[name],
                oracle[name][:, rows, cols],
                f"{name}: R1 replay of discarded paint inside write window",
            )

    def test_R2_crash_before_publish_recovers_after_restart(
        self, tmp_path: Path, family_site, monkeypatch
    ) -> None:
        """Fault at the store publish (the crash-before-publish seam):
        the durable row is left ``running`` with the solve already
        staged. A restarted runner requeues it; the re-run must IGNORE
        its own staged state, recompute, and publish — not find an empty
        window and fail with a misleading edit_rejected."""
        state_root = tmp_path / "state"
        app1, client1 = _restartable_app(family_site, state_root)
        with client1:
            scenario_id = _scenario_ready(client1)
            response = _universal(client1, scenario_id, [_MET_T1_30], base=0, key="r2a-1")
            job_id = response.json()["job_id"]

            store = app1.state.context.store
            original = store.publish_result
            fired = {"n": 0}

            def exploding_publish(*args, **kwargs):
                if kwargs.get("job_id") == job_id:
                    fired["n"] += 1
                    if fired["n"] == 1:
                        raise RuntimeError("injected crash before publish")
                return original(*args, **kwargs)

            monkeypatch.setattr(store, "publish_result", exploding_publish)
            _wait_fault_fired(client1, job_id)
            # Crash residue: the durable row is stuck running.
            assert client1.get(f"/api/v1/jobs/{job_id}").json()["status"] == "running"

        # Restart over the SAME durable state: the recovered runner
        # requeues the running row (no fault in the fresh store object).
        _, client2 = _restartable_app(family_site, state_root)
        with client2:
            job = wait_for_job(client2, job_id, timeout=180.0)
            assert job["status"] == "complete", job
            assert job["metrics"]["transport"] == "executor"
            scenario = client2.get(f"/api/v1/scenarios/{scenario_id}").json()
            assert scenario["exact_result_version"] == 1

            delta = _validated_delta(family_site, _MET_T1_30)
            oracle = _full_run(
                family_site.cache,
                family_site.site,
                family_site.grid,
                family_site.root / "oracle_r2a",
                overlay=overlay_from_deltas([delta]),
            )
            served = _served_arrays(client2, job)
            for name in VARIABLES:
                _assert_bitwise(
                    served[name], oracle[name], f"{name}: R2a restart recovery"
                )

            # The durable protocol converged lazily, by design: the
            # recovered job's staged snapshot sits in pending/ under its
            # own job id, and the NEXT family job adopts it (the sidecar
            # flips to that generation) before replaying its own window.
            state_dir = state_root / "scenarios" / scenario_id / "executor-state"
            staged = json.loads(
                (state_dir / "pending" / "pending.json").read_text()
            )
            assert staged == {"job_id": job_id, "covered_sequence": 1}

            third = _universal(client2, scenario_id, [_MET_T0_25], base=1, key="r2a-3")
            assert third.status_code == 202, third.text
            job3 = wait_for_job(client2, third.json()["job_id"], timeout=180.0)
            assert job3["status"] == "complete", job3

            coverage = json.loads((state_dir / "ledger-coverage.json").read_text())
            assert coverage["covered_sequence"] == 1
            assert coverage["generation"] == f"gen-{job_id}"
            # The adopted snapshot is what job 3 replayed from: its result
            # reflects BOTH edits bitwise.
            delta_2 = _validated_delta(family_site, _MET_T0_25)
            oracle_both = _full_run(
                family_site.cache,
                family_site.site,
                family_site.grid,
                family_site.root / "oracle_r2a_both",
                overlay=overlay_from_deltas([delta, delta_2]),
            )
            served3 = _served_arrays(client2, job3)
            for name in VARIABLES:
                _assert_bitwise(
                    served3[name],
                    oracle_both[name],
                    f"{name}: R2a adoption by the next family job",
                )

    def test_R2_crash_after_publish_completes_idempotently(
        self, tmp_path: Path, family_site, monkeypatch
    ) -> None:
        """Fault AFTER the publish committed (the crash-between-publish-
        and-completion seam): the result is durable, the row is stuck
        ``running``. The restarted re-run recomputes deterministically,
        hits ResultAlreadyPublished, and the checksum-verified idempotent
        completion path finishes the job."""
        state_root = tmp_path / "state"
        app1, client1 = _restartable_app(family_site, state_root)
        with client1:
            scenario_id = _scenario_ready(client1)
            response = _universal(client1, scenario_id, [_MET_T1_30], base=0, key="r2b-1")
            job_id = response.json()["job_id"]

            store = app1.state.context.store
            original = store.finish_job
            fired = {"n": 0}

            def exploding_finish(job_id_arg, *args, **kwargs):
                if job_id_arg == job_id:
                    fired["n"] += 1
                    if fired["n"] == 1:
                        raise RuntimeError("injected crash after publish")
                return original(job_id_arg, *args, **kwargs)

            monkeypatch.setattr(store, "finish_job", exploding_finish)
            _wait_fault_fired(client1, job_id)
            # The result itself IS durable even though the row is stuck.
            assert client1.get(f"/api/v1/jobs/{job_id}").json()["status"] == "running"
            assert (
                client1.get(f"/api/v1/scenarios/{scenario_id}").json()[
                    "exact_result_version"
                ]
                == 1
            )

        _, client2 = _restartable_app(family_site, state_root)
        with client2:
            job = wait_for_job(client2, job_id, timeout=180.0)
            assert job["status"] == "complete", job
            scenario = client2.get(f"/api/v1/scenarios/{scenario_id}").json()
            assert scenario["exact_result_version"] == 1

            delta = _validated_delta(family_site, _MET_T1_30)
            oracle = _full_run(
                family_site.cache,
                family_site.site,
                family_site.grid,
                family_site.root / "oracle_r2b",
                overlay=overlay_from_deltas([delta]),
            )
            served = _served_arrays(client2, job)
            for name in VARIABLES:
                _assert_bitwise(
                    served[name], oracle[name], f"{name}: R2b idempotent completion"
                )

    def test_R3_cancel_in_the_guard_to_publish_window_replays(
        self, client: TestClient, family_site, monkeypatch
    ) -> None:
        """Cancel while the solver is finishing (after its batch, before
        finalize's liveness check): the result is discarded by contract
        ("the next edit supersedes the cancelled work") — so the next
        edit MUST replay the cancelled paint, and its published window
        must cover BOTH paints."""
        scenario_id = _scenario_ready(client)
        paint_a = RasterWindow(40, 48, 40, 48)
        paint_b = RasterWindow(72, 80, 8, 16)  # disjoint from A
        gate = _Gate()
        _gate_load_patch(monkeypatch, gate)
        first = _universal(client, scenario_id, [_paint(paint_a)], base=0, key="r3-1")
        assert first.status_code == 202, first.text
        job1_id = first.json()["job_id"]
        gate.wait_entered()

        cancel = client.post(f"/api/v1/jobs/{job1_id}/cancel")
        assert cancel.status_code == 200, cancel.text
        assert cancel.json()["status"] == "cancelled"
        gate.open()

        job1 = _wait_terminal(client, job1_id)
        assert job1["status"] == "cancelled", job1

        # The next edit (base = v1: the cancelled edit is committed to
        # the ledger) must supersede the cancelled work, not skip it.
        second = _universal(client, scenario_id, [_paint(paint_b)], base=1, key="r3-2")
        assert second.status_code == 202, second.text
        job2 = wait_for_job(client, second.json()["job_id"], timeout=180.0)
        assert job2["status"] == "complete", job2
        assert job2["metrics"]["transport"] == "executor"
        scenario = client.get(f"/api/v1/scenarios/{scenario_id}").json()
        assert scenario["exact_result_version"] == 2

        # Job 2's write window covers BOTH paints (it replayed the
        # cancelled one); code that consumed the cancelled edit
        # publishes paint B's window alone.
        window = job2["window"]
        assert window["row_start"] <= min(paint_a.row_start, paint_b.row_start)
        assert window["row_stop"] >= max(paint_a.row_stop, paint_b.row_stop)
        assert window["col_start"] <= min(paint_a.col_start, paint_b.col_start)
        assert window["col_stop"] >= max(paint_a.col_stop, paint_b.col_stop)

        delta_a = _validated_delta(family_site, _paint(paint_a))
        delta_b = _validated_delta(family_site, _paint(paint_b))
        oracle = _full_run(
            family_site.cache,
            family_site.site,
            family_site.grid,
            family_site.root / "oracle_r3_both",
            landcover_overlay=landcover_overlay_from_deltas([delta_a, delta_b]),
        )
        served = _served_arrays(client, job2)
        rows = slice(window["row_start"], window["row_stop"])
        cols = slice(window["col_start"], window["col_stop"])
        # The payload carries the union write window, all times (patch
        # semantics); the oracle is sliced to the same footprint.
        for name in VARIABLES:
            _assert_bitwise(
                served[name],
                oracle[name][:, rows, cols],
                f"{name}: R3 replay of cancelled paint inside write window",
            )

    def test_building_then_tree_edit_restores_snapshot_and_routes_executor(
        self, client: TestClient, family_site
    ) -> None:
        """A building edit (full-tile chain) snapshots; a LATER tree edit
        routes through the executor again (the family scenario never
        falls back to the legacy solver). u-d4c: with the building fold
        accumulated, the tree batch rides the regeneration chain with the
        building, so the served window is bitwise the building+tree twin
        oracle — never the baseline+tree scene that silently reverted the
        massing."""
        from solweig_gpu.incremental.adapters.building import (
            massing_edits_from_deltas,
        )
        from solweig_gpu.incremental.regenerate import regenerate_building_batch

        scenario_id = _scenario_ready(client)
        building = [
            {
                "adapter": "building_geometry",
                "operation": "add",
                "target": "block-a",
                "values": {"footprint_m": NEW_BLOCK, "height_m": 12.0},
            }
        ]
        first = _universal(client, scenario_id, building, base=0, key="snap-bld-1")
        job1 = wait_for_job(client, first.json()["job_id"], timeout=180.0)
        assert job1["status"] == "complete", job1
        assert job1["mode"] == "full"

        tree = make_tree("tree_far", u=0.12, v=0.15)
        tree_response = client.post(
            f"/api/v1/scenarios/{scenario_id}/edits",
            json={
                "base_scene_version": 1,
                "edits": [{"operation": "add", "tree": tree}],
                "requested_result": {"time_indices": [1], "variables": ["utci", "tmrt"]},
            },
            headers={"Idempotency-Key": "tree-after-building"},
        )
        assert tree_response.status_code == 202, tree_response.text
        job2 = wait_for_job(client, tree_response.json()["job_id"], timeout=180.0)
        assert job2["status"] == "complete", job2
        assert job2["metrics"]["transport"] == "executor"
        assert job2["mode"] == "full"  # chain-routed with the fold (u-d4c)

        # The twin oracle: the SAME building delta the route committed,
        # with the tree riding the layer, through one regeneration.
        layer = TreeLayer(family_site.cache.tree_base, family_site.grid)
        layer.add_tree(api_tree_to_spec(tree, _grid_info(family_site)))
        twin = regenerate_building_batch(
            edits=massing_edits_from_deltas(
                (_validated_delta(family_site, building[0]),)
            ),
            baseline_site_dir=family_site.site,
            baseline_cache=family_site.cache,
            scenario_root=(
                family_site.root / "twin_tree_after_building" / "scenario"
            ),
            selected_date_str=DATE_STR,
            tree_layer=layer,
        )
        served = _served_arrays(client, job2)
        window = job2["window"]
        rows = slice(window["row_start"], window["row_stop"])
        cols = slice(window["col_start"], window["col_stop"])
        # The tree payload is window-sliced at the requested time index [1].
        for name in VARIABLES:
            _assert_bitwise(
                served[name][0],
                twin.outputs[name][1][rows, cols],
                f"{name}: tree-after-building inside write window",
            )


@pytest.mark.scientific
class TestBuildingLedgerReplay:
    """u-d4c remediation F2: BUILDING events replay through the REAL
    bridge fresh-rebuild path (executor_bridge._build_executor /
    _pending_commands -> universal.command_from_item's building_geometry
    branch) — the deterministic-rebuild unit test re-derived the commands
    on a test-local executor, so this drives the actual HTTP bridge."""

    def test_R4_snapshot_loss_replays_the_building_event_bitwise(
        self,
        client: TestClient,
        monkeypatch,
        tmp_path: Path,
    ) -> None:
        """A published building edit, then executor-state loss (staged
        snapshot + coverage sidecar + generations gone): the next family
        job rebuilds the executor FRESH from the ledger, replays the
        BUILDING event through the real command assembly, re-forms the
        SAME fold, routes the chain, and serves arrays bitwise-identical
        to an uninterrupted twin scenario."""
        import shutil

        from solweig_gpu.server import executor_bridge
        from solweig_gpu.server import universal as universal_mod

        building = [
            {
                "adapter": "building_geometry",
                "operation": "add",
                "target": "block-a",
                "values": {"footprint_m": NEW_BLOCK, "height_m": 12.0},
            }
        ]
        met = [
            {
                "adapter": "meteorological_forcing",
                "operation": "update_time_row",
                "time_index": 1,
                "values": {"air_temperature": 30.0},
            }
        ]

        # 1) The scenario publishes a building edit; its fold is durable
        #    in the job's staged snapshot.
        scenario_id = _scenario_ready(client)
        first = _universal(client, scenario_id, building, base=0, key="r4-bld-1")
        assert first.status_code == 202, first.text
        job1 = wait_for_job(client, first.json()["job_id"], timeout=180.0)
        assert job1["status"] == "complete", job1
        assert job1["mode"] == "full"
        state_dir = (
            tmp_path / "state" / "scenarios" / scenario_id / "executor-state"
        )
        original_fold = json.loads(
            (state_dir / "pending" / "scenario-state.json").read_text()
        )["massing_edits"]
        assert [item["building_id"] for item in original_fold] == ["block-a"]
        assert original_fold[0]["after"]["height_m"] == 12.0

        # 2) The uninterrupted twin: the SAME building edit then a met
        #    edit, with no state loss — the reference the rebuild must
        #    match bitwise.
        twin_id = _scenario_ready(client)
        twin_bld = _universal(client, twin_id, building, base=0, key="r4-twin-1")
        assert (
            wait_for_job(client, twin_bld.json()["job_id"], timeout=180.0)[
                "status"
            ]
            == "complete"
        )
        twin_met = _universal(client, twin_id, met, base=1, key="r4-twin-2")
        twin_job = wait_for_job(client, twin_met.json()["job_id"], timeout=180.0)
        assert twin_job["status"] == "complete", twin_job
        assert twin_job["metrics"]["transport"] == "executor"

        # 3) STATE LOSS on the primary: the executor-state tree vanishes.
        #    The bridge's own defensive read treats a sidecar naming a
        #    missing snapshot (here: no sidecar at all) as NO coverage
        #    and rebuilds fresh from the ledger.
        shutil.rmtree(state_dir)

        # Spies on the real path: the rebuild must NOT restore, and the
        # BUILDING event must travel the REAL command assembly (the
        # universal transport's building_geometry branch).
        restores: list = []
        original_restore = executor_bridge.restore_into_executor

        def spy_restore(*args, **kwargs):
            restores.append(args)
            return original_restore(*args, **kwargs)

        monkeypatch.setattr(executor_bridge, "restore_into_executor", spy_restore)

        assembled: list = []
        original_assemble = universal_mod.command_from_item

        def spy_assemble(item, **kwargs):
            assembled.append((item.adapter, item.operation))
            return original_assemble(item, **kwargs)

        monkeypatch.setattr(universal_mod, "command_from_item", spy_assemble)

        # 4) The next family job (met) rebuilds FRESH: the pending window
        #    replays the BUILDING event as a command, the fold re-forms,
        #    and the batch routes the regeneration chain.
        second = _universal(client, scenario_id, met, base=1, key="r4-met-2")
        assert second.status_code == 202, second.text
        job2 = wait_for_job(client, second.json()["job_id"], timeout=180.0)
        assert job2["status"] == "complete", job2
        assert job2["metrics"]["transport"] == "executor"
        assert job2["mode"] == "full"  # chain-routed: the rebuilt fold routed it

        assert restores == [], "the state-lost scenario must rebuild fresh"
        assert ("building_geometry", "add") in assembled, (
            "the BUILDING ledger event never traveled the real "
            "command_from_item assembly"
        )

        # 5) The rebuilt executor's fold equals the original's.
        rebuilt_fold = json.loads(
            (state_dir / "pending" / "scenario-state.json").read_text()
        )["massing_edits"]
        assert rebuilt_fold == original_fold

        # 6) Bitwise-identical served arrays vs the uninterrupted twin.
        served = _served_arrays(client, job2)
        twin_served = _served_arrays(client, twin_job)
        for name in VARIABLES:
            _assert_bitwise(
                served[name],
                twin_served[name],
                f"{name}: R4 fresh rebuild vs uninterrupted twin",
            )


class TestStoreSupersessionWatch:
    """Fast unit coverage for the store-bound checkpoint (H1/R1): while
    the store's scene version is unchanged it is transparent (reports
    the executor's internal revision); any store commit trips it."""

    def _watch(self):
        from solweig_gpu.server.executor_bridge import _StoreSupersessionWatch

        store = SimpleNamespace(version=5)
        store.current_scene_version = lambda scenario_id: store.version
        executor = SimpleNamespace(scene_revision=7)
        watch = _StoreSupersessionWatch(store, "scen")
        watch.bind(executor)
        return watch, store

    def test_transparent_while_store_version_unchanged(self) -> None:
        watch, _ = self._watch()
        assert not watch.tripped()
        assert watch() == 7  # the executor's internal revision, verbatim

    def test_any_store_commit_trips_the_checkpoint(self) -> None:
        watch, store = self._watch()
        store.version = 6  # an edit (or a reset) committed mid-solve
        assert watch.tripped()
        # A revision no supersession target can match: every cooperative
        # checkpoint aborts the batch.
        assert watch() == -1

    def test_executor_passes_the_provider_to_its_worker(self, tmp_path: Path) -> None:
        """The PlanExecutor wires the override through to the worker's
        supersession checkpoints (``None`` keeps the internal default)."""
        from solweig_gpu.incremental.executor import PlanExecutor
        from tests.test_incremental_worker import _make_tiny_site

        grid, site = _make_tiny_site(tmp_path / "site")
        cache = _build_cache(
            site,
            tmp_path / "cache",
            met_path=site / "metfiles" / f"metfile_0_0_{DATE_STR}.txt",
            site_id="tiny",
        )
        provider = lambda: 42  # noqa: E731
        executor = PlanExecutor(
            cache=cache,
            layer=TreeLayer(cache.tree_base, grid),
            site_dir=site,
            results_root=tmp_path / "results",
            selected_date_str=DATE_STR,
            revision_provider=provider,
        )
        assert executor._worker._revision_provider is provider
        assert executor._worker._revision_provider() == 42

    def test_executor_default_provider_reports_internal_revision(
        self, tmp_path: Path
    ) -> None:
        from solweig_gpu.incremental.executor import PlanExecutor
        from tests.test_incremental_worker import _make_tiny_site

        grid, site = _make_tiny_site(tmp_path / "site")
        cache = _build_cache(
            site,
            tmp_path / "cache",
            met_path=site / "metfiles" / f"metfile_0_0_{DATE_STR}.txt",
            site_id="tiny",
        )
        executor = PlanExecutor(
            cache=cache,
            layer=TreeLayer(cache.tree_base, grid),
            site_dir=site,
            results_root=tmp_path / "results",
            selected_date_str=DATE_STR,
        )
        assert executor._worker._revision_provider() == executor.scene_revision == 0


# ---------------------------------------------------------------------------
# u-e3c: post-reset tree routing (NF1) + the pre-v5-pruned routing rescue
# (predicate attack C). The routing predicate must never under-route a
# scenario whose executor state still carries family state, and the fresh
# rebuild after a reset must still run the replay window's tree edits.
# ---------------------------------------------------------------------------


def _reset_to_baseline(client: TestClient, scenario_id: str, *, key: str) -> dict:
    """Reset and wait for the re-published baseline (exact == scene)."""
    body = client.get(f"/api/v1/scenarios/{scenario_id}").json()
    response = client.post(
        f"/api/v1/scenarios/{scenario_id}/reset",
        json={"base_scene_version": body["scene_version"]},
        headers={"Idempotency-Key": key},
    )
    assert response.status_code == 200, response.text
    for _ in range(500):
        body = client.get(f"/api/v1/scenarios/{scenario_id}").json()
        if body["exact_result_version"] == body["scene_version"]:
            return body
        time.sleep(0.01)
    pytest.fail("reset baseline never published")


def _tree_edit(client: TestClient, scenario_id: str, tree: dict, base: int, *, key: str):
    return client.post(
        f"/api/v1/scenarios/{scenario_id}/edits",
        json={
            "base_scene_version": base,
            "edits": [{"operation": "add", "tree": tree}],
            "requested_result": {"time_indices": [1], "variables": ["utci", "tmrt"]},
        },
        headers={"Idempotency-Key": key},
    )


def _window_slices(job: dict) -> tuple[slice, slice]:
    window = job["window"]
    return (
        slice(window["row_start"], window["row_stop"]),
        slice(window["col_start"], window["col_stop"]),
    )


@pytest.mark.scientific
class TestPostResetTreeRouting:
    """u-e3c NF1: family edit -> reset -> TREE edit refused deterministically
    (``edit_rejected: batch must contain at least one validated edit``) on
    EVERY retry until a family edit healed it. The surviving pre-reset
    family event kept routing the executor while the fresh rebuild's replay
    floor (``max(covered, last_reset)``) excluded that event and
    ``replay_tree_events=False`` skipped the window's tree event — an empty
    batch, refused. The tree base after a reset is VOIDED (reset wipes
    ``scenario_trees``), so the window's tree events rebuild it exactly and
    must replay as vegetation commands.
    """

    def test_family_edit_reset_then_tree_edit_completes_bitwise(
        self, client: TestClient, family_site
    ) -> None:
        scenario_id = _scenario_ready(client)
        met = dict(_MET_T1_30)
        job1 = wait_for_job(
            client,
            _universal(client, scenario_id, [met], base=0, key="nf1-met").json()["job_id"],
        )
        assert job1["status"] == "complete"
        assert job1["metrics"]["transport"] == "executor"

        body = _reset_to_baseline(client, scenario_id, key="nf1-reset")

        tree = make_tree("tree_nf1", u=0.42, v=0.48)
        response = _tree_edit(
            client, scenario_id, tree, body["scene_version"], key="nf1-tree1"
        )
        assert response.status_code == 202, response.text
        job = wait_for_job(client, response.json()["job_id"])
        assert job["status"] == "complete", job.get("error")
        assert job["metrics"]["transport"] == "executor"

        # The NEXT tree edit must not refuse either (the wedge repeated on
        # every retry); it rides the ordinary snapshot-restore path.
        body = client.get(f"/api/v1/scenarios/{scenario_id}").json()
        tree2 = make_tree("tree_nf1b", u=0.36, v=0.52)
        response = _tree_edit(
            client, scenario_id, tree2, body["scene_version"], key="nf1-tree2"
        )
        job2 = wait_for_job(client, response.json()["job_id"])
        assert job2["status"] == "complete", job2.get("error")

        # Oracle twins: the reset VOIDED the met edit, so the served scene
        # is baseline+tree; the with-met twin must differ (non-vacuous).
        met_delta = _validated_delta(family_site, met)
        layer = TreeLayer(family_site.cache.tree_base, family_site.grid)
        layer.add_tree(api_tree_to_spec(tree, _grid_info(family_site)))
        rows, cols = _window_slices(job)
        bare = _full_run(
            family_site.cache,
            family_site.site,
            family_site.grid,
            family_site.root / "oracle_nf1_bare",
            layer=layer,
        )
        with_met = _full_run(
            family_site.cache,
            family_site.site,
            family_site.grid,
            family_site.root / "oracle_nf1_met",
            overlay=overlay_from_deltas([met_delta]),
            layer=layer,
        )
        served = _served_arrays(client, job)
        for name in VARIABLES:
            _assert_bitwise(
                served[name][0][rows, cols],
                bare[name][1][rows, cols],
                f"{name}: post-reset tree edit vs baseline+tree twin",
            )
            assert not np.array_equal(
                served[name][0][rows, cols],
                with_met[name][1][rows, cols],
                equal_nan=True,
            ), f"{name}: the met twins agree — the differential is vacuous"

    def test_pre_v5_pruned_row_routes_executor_via_state_dir(
        self, client: TestClient, family_site
    ) -> None:
        """u-e1 predicate attack C: a pre-v5 database whose family events
        were ALREADY pruned by retention before the upgrade migrates with
        ``carries_family_edits=0`` and NO surviving family event — the v5
        event backfill cannot see state that no longer exists. The next
        TREE edit must still route the executor (the on-disk executor state
        under ``results_root/<scenario>/executor-state`` proves the scenario
        ran family jobs) and publish the family state bitwise. On the
        unfixed code the row under-routes onto the legacy baseline-bound
        solver and publishes WITHOUT the met overlay — the F1 wrong-scene
        path resurrected."""
        scenario_id = _scenario_ready(client)
        met = dict(_MET_T1_30)
        job1 = wait_for_job(
            client,
            _universal(client, scenario_id, [met], base=0, key="attackc-met").json()[
                "job_id"
            ],
        )
        assert job1["status"] == "complete"
        assert job1["metrics"]["transport"] == "executor"

        store = client.app.state.context.store
        # OLD-code retention semantics: the family event is gone...
        deleted = store.sweep_retention(event_tail=0)
        assert deleted["edit_events"] >= 1
        assert not [
            e for e in store.list_events(scenario_id) if e.get("family")
        ], "family events survived the sweep — the repro is invalid"
        # ...and the pre-v5 row never carried the flag. (The reader as a
        # whole now returns True — the executor-state rescue under test is
        # exactly what this repro's row-state cannot express.)
        with store._write() as conn:
            conn.execute(
                "UPDATE scenarios SET carries_family_edits = 0 WHERE scenario_id = ?",
                (scenario_id,),
            )
        row = store._read(
            "SELECT carries_family_edits FROM scenarios WHERE scenario_id = ?",
            (scenario_id,),
        )
        assert row is not None and not int(row["carries_family_edits"]), (
            "the flag must be 0 for this repro to be valid"
        )

        body = client.get(f"/api/v1/scenarios/{scenario_id}").json()
        tree = make_tree("tree_attackc", u=0.42, v=0.48)
        response = _tree_edit(
            client, scenario_id, tree, body["scene_version"], key="attackc-tree"
        )
        assert response.status_code == 202, response.text
        job = wait_for_job(client, response.json()["job_id"])
        assert job["status"] == "complete", job.get("error")
        assert job["metrics"]["transport"] == "executor", (
            "the pre-v5-pruned row under-routed onto the legacy solver (attack C)"
        )

        # Served scene: the executor state carried the met overlay, so the
        # WITH-met twin is bitwise correct and the bare twin differs.
        met_delta = _validated_delta(family_site, met)
        layer = TreeLayer(family_site.cache.tree_base, family_site.grid)
        layer.add_tree(api_tree_to_spec(tree, _grid_info(family_site)))
        rows, cols = _window_slices(job)
        with_met = _full_run(
            family_site.cache,
            family_site.site,
            family_site.grid,
            family_site.root / "oracle_attackc_met",
            overlay=overlay_from_deltas([met_delta]),
            layer=layer,
        )
        bare = _full_run(
            family_site.cache,
            family_site.site,
            family_site.grid,
            family_site.root / "oracle_attackc_bare",
            layer=layer,
        )
        served = _served_arrays(client, job)
        for name in VARIABLES:
            _assert_bitwise(
                served[name][0][rows, cols],
                with_met[name][1][rows, cols],
                f"{name}: tree-after-prune vs WITH-met twin",
            )
            assert not np.array_equal(
                served[name][0][rows, cols],
                bare[name][1][rows, cols],
                equal_nan=True,
            ), f"{name}: the met twins agree — the differential is vacuous"

    def test_post_reset_pruned_row_keeps_legacy_routing_bitwise(
        self, client: TestClient, family_site
    ) -> None:
        """u-e3c attack-B guard: family edit -> reset -> retention sweep that
        prunes BOTH the family and the reset events -> tree edit. The
        executor-state directory survives the reset on disk (reset never
        touches it), but its coverage does not postdate the reset — the
        reset-aware dir signal must NOT arm, and the legacy baseline+tree
        publish is the bitwise-correct scene (the met state was voided).
        Pinning legacy here is what keeps the dir rescue from resurrecting
        pre-reset executor state."""
        scenario_id = _scenario_ready(client)
        met = dict(_MET_T1_30)
        job1 = wait_for_job(
            client,
            _universal(client, scenario_id, [met], base=0, key="guard-met").json()["job_id"],
        )
        assert job1["status"] == "complete"

        _reset_to_baseline(client, scenario_id, key="guard-reset")

        store = client.app.state.context.store
        deleted = store.sweep_retention(event_tail=0)
        assert deleted["edit_events"] >= 2, (
            "the sweep must prune the family event AND the reset event for "
            "this repro to be valid"
        )
        assert store.list_events(scenario_id) == []

        body = client.get(f"/api/v1/scenarios/{scenario_id}").json()
        tree = make_tree("tree_guard", u=0.42, v=0.48)
        response = _tree_edit(
            client, scenario_id, tree, body["scene_version"], key="guard-tree"
        )
        assert response.status_code == 202, response.text
        job = wait_for_job(client, response.json()["job_id"])
        assert job["status"] == "complete", job.get("error")
        assert (job["metrics"] or {}).get("transport") is None, (
            "post-reset with no surviving family state must route LEGACY "
            "(the executor state predates the reset and must not resurrect)"
        )

        # Served scene: baseline+tree (the met edit was voided); the
        # with-met twin must differ (non-vacuous).
        met_delta = _validated_delta(family_site, met)
        layer = TreeLayer(family_site.cache.tree_base, family_site.grid)
        layer.add_tree(api_tree_to_spec(tree, _grid_info(family_site)))
        rows, cols = _window_slices(job)
        bare = _full_run(
            family_site.cache,
            family_site.site,
            family_site.grid,
            family_site.root / "oracle_guard_bare",
            layer=layer,
        )
        with_met = _full_run(
            family_site.cache,
            family_site.site,
            family_site.grid,
            family_site.root / "oracle_guard_met",
            overlay=overlay_from_deltas([met_delta]),
            layer=layer,
        )
        served = _served_arrays(client, job)
        for name in VARIABLES:
            _assert_bitwise(
                served[name][0][rows, cols],
                bare[name][1][rows, cols],
                f"{name}: post-reset pruned tree edit vs baseline+tree twin",
            )
            assert not np.array_equal(
                served[name][0][rows, cols],
                with_met[name][1][rows, cols],
                equal_nan=True,
            ), f"{name}: the met twins agree — the differential is vacuous"
