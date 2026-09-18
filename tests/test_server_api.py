# SPDX-License-Identifier: GPL-3.0-only
"""Design-tool HTTP API contract tests (gates API-001..004).

All tests inject a fake solver through the app factory's dependency-injection
seam, so no real SOLWEIG physics runs here. The fake solver records every
call and returns deterministic arrays, which lets the tests assert:

* idempotency replay and key-reuse conflicts (API-001);
* explicit optimistic-concurrency conflicts (API-002);
* manifest + binary payload validation with checksums and headers (API-003);
* supersession, stale-publish rejection, and restart recovery (API-004).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import sqlite3
import threading
import time
import weakref
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from solweig_gpu.incremental.geometry import RasterWindow
from solweig_gpu.incremental.manifest import (
    CACHE_ARRAY_LAYOUT,
    CACHE_MODEL_VERSION,
    CACHE_SCHEMA_VERSION,
    ArrayEntry,
    AuxSourceEntry,
    RasterSourceEntry,
    REQUIRED_RASTER_KEYS,
    REQUIRED_SOURCE_KEYS,
    SiteManifest,
    SolarGeometrySpec,
)
from solweig_gpu.server import patch_codec
from solweig_gpu.server.app import create_app
from solweig_gpu.server.jobs import SolveResult, resolve_requested

SITE_ID = "test-site"
ROWS, COLS, PIXEL = 32, 48, 2.0
TIME_STEPS = 3
ORIGIN = (1000.0, 2000.0)


# ---------------------------------------------------------------------------
# Fixture site: a manifest loadable by incremental.load_manifest plus
# baseline result arrays (no site cache arrays are needed for API tests
# because the solver is faked).
# ---------------------------------------------------------------------------


def _hex64(seed: int) -> str:
    return f"{seed:064x}"[-64:]


def make_site_cache(
    root: Path,
    *,
    site_id: str = SITE_ID,
    baseline: bool = True,
    baseline_time_steps: int | None = None,
) -> Path:
    cache_dir = root / "cache" / site_id
    (cache_dir / "baseline_results").mkdir(parents=True, exist_ok=True)
    shape = (ROWS, COLS)
    rasters = {
        name: RasterSourceEntry(
            path=f"{name}/{name}_0_0.tif",
            sha256=_hex64(index),
            shape=shape,
            dtype="float32",
            nodata=None,
            geotransform=(ORIGIN[0], PIXEL, 0.0, ORIGIN[1], 0.0, -PIXEL),
            crs_wkt_hash="sha256:" + _hex64(99),
        )
        for index, name in enumerate(REQUIRED_RASTER_KEYS)
    }
    sources = {
        name: AuxSourceEntry(path=f"{name}_0_0.bin", sha256=_hex64(100 + index))
        for index, name in enumerate(REQUIRED_SOURCE_KEYS)
    }
    # Real cache array files with true sha256 hashes (P9: /health/ready loads
    # the site cache through SiteRegistry, whose self-test checksums every
    # small array), so the synthetic site passes startup validation.
    import hashlib
    import io

    arrays: dict[str, ArrayEntry] = {}
    for index, name in enumerate(CACHE_ARRAY_LAYOUT):
        array_shape = (TIME_STEPS, 7) if name == "solar" else shape
        values = np.full(array_shape, 0.05 + 0.001 * index, dtype=np.float32)
        buffer = io.BytesIO()
        np.lib.format.write_array(buffer, values, allow_pickle=False)
        blob = buffer.getvalue()
        (cache_dir / CACHE_ARRAY_LAYOUT[name]).parent.mkdir(
            parents=True, exist_ok=True
        )
        (cache_dir / CACHE_ARRAY_LAYOUT[name]).write_bytes(blob)
        arrays[name] = ArrayEntry(
            path=CACHE_ARRAY_LAYOUT[name],
            shape=array_shape,
            dtype="float32",
            nbytes=len(blob),
            sha256=hashlib.sha256(blob).hexdigest(),
        )
    manifest = SiteManifest(
        site_id=site_id,
        tile_key="0_0",
        model_version=CACHE_MODEL_VERSION,
        cache_schema_version=CACHE_SCHEMA_VERSION,
        rows=ROWS,
        cols=COLS,
        pixel_size_m=PIXEL,
        origin_x_m=ORIGIN[0],
        origin_y_m=ORIGIN[1],
        crs_wkt_hash="sha256:" + _hex64(7),
        time_steps=TIME_STEPS,
        patch_count=16,
        rasters=rasters,
        sources=sources,
        solar_geometry=SolarGeometrySpec(
            latitude=33.0, longitude=-87.0, altitude_m=0.0, utc_offset_hours=-5.0
        ),
        arrays=arrays,
    )
    manifest.write(cache_dir / "manifest.json")
    if baseline:
        rng = np.random.default_rng(20260901)
        steps = baseline_time_steps if baseline_time_steps is not None else TIME_STEPS
        for name, offset in (("utci", 20.0), ("tmrt", 40.0)):
            array = (offset + rng.uniform(0, 10, size=(steps, ROWS, COLS))).astype(
                np.float32
            )
            np.save(cache_dir / "baseline_results" / f"{name}.f32.npy", array)
        (cache_dir / "baseline_results" / "metadata.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "variables": ["utci", "tmrt"],
                    "time_steps": TIME_STEPS,
                    "source": "baseline",
                }
            )
        )
    return cache_dir


def load_baseline(cache_dir: Path, name: str) -> np.ndarray:
    return np.asarray(np.load(cache_dir / "baseline_results" / f"{name}.f32.npy"))


# ---------------------------------------------------------------------------
# Fake solver (dependency injection; never runs SOLWEIG physics)
# ---------------------------------------------------------------------------


class FakeSolver:
    """Records calls; returns deterministic window results per request.

    The produced value depends on the target scene version and tree count so
    tests can distinguish scenarios, versions, and stale results.
    """

    def __init__(self) -> None:
        self.calls: list[Any] = []
        self.lock = threading.Lock()
        #: set whenever a call begins; lets tests block until the solver is
        #: mid-flight inside the worker thread.
        self.entered = threading.Event()
        #: subtracted from the target version in the returned SolveResult to
        #: simulate a solver publishing an outdated scene revision.
        self.scene_version_offset = 0
        #: gate event; when set, each call blocks until cleared.
        self.gate: threading.Event | None = None
        self.fail = False

    def __call__(self, request, progress) -> SolveResult:
        with self.lock:
            self.calls.append(request)
        self.entered.set()
        if self.gate is not None:
            self.gate.wait()
        if self.fail:
            raise RuntimeError("fake solver explosion")
        time_indices, variables = resolve_requested(request.requested, request.grid)
        window = RasterWindow(4, 12, 6, 18)  # fixed 8x12 window inside the site
        value = 25.0 + request.target_scene_version + len(request.trees)
        progress("windowing", 0, 1)
        progress(
            "time_loop",
            len(time_indices),
            len(time_indices),
            mode="local",
            window=window,
        )
        arrays = {
            name: np.full(
                (len(time_indices), window.height, window.width),
                value + offset,
                dtype=np.float32,
            )
            for offset, name in enumerate(variables)
        }
        return SolveResult(
            status="published",
            scene_version=request.target_scene_version + self.scene_version_offset,
            mode="local",
            window=window,
            time_indices=time_indices,
            variables=variables,
            arrays=arrays,
            metrics={"mean_utci_delta_c": -0.84, "peak_utci_delta_c": -3.42},
        )


# ---------------------------------------------------------------------------
# App / client factories
# ---------------------------------------------------------------------------


def make_app(
    tmp_path: Path,
    solver: FakeSolver,
    *,
    site_cache: Path | None = None,
    state_root: Path | None = None,
    start_worker: bool = True,
    coalescing_window_ms: float = 20.0,
    **kwargs: Any,
):
    # P9 security controls are default-on in create_app (per-IP rate limit,
    # per-scenario edit budget); these tests poll far faster than those
    # budgets allow, so they opt out explicitly (the dedicated security
    # tests exercise the defaults).
    kwargs.setdefault("requests_per_minute_per_ip", None)
    kwargs.setdefault("edits_per_minute", None)
    return create_app(
        state_root=state_root or (tmp_path / "state"),
        sites={SITE_ID: {"cache_dir": site_cache or make_site_cache(tmp_path)}},
        solver_factory=lambda context: solver,
        coalescing_window_ms=coalescing_window_ms,
        start_worker=start_worker,
        **kwargs,
    )


@pytest.fixture()
def solver() -> FakeSolver:
    return FakeSolver()


@pytest.fixture()
def site_cache(tmp_path: Path) -> Path:
    return make_site_cache(tmp_path)


@pytest.fixture()
def client(tmp_path: Path, solver: FakeSolver, site_cache: Path):
    app = make_app(tmp_path, solver, site_cache=site_cache)
    with TestClient(app) as test_client:
        yield test_client


def make_tree(tree_id: str = "tree_01", *, u: float = 0.5, v: float = 0.4, **overrides):
    tree = {
        "tree_id": tree_id,
        "component_type": "broad_canopy",
        "u": u,
        "v": v,
        "height_m": 18.0,
        "canopy_diameter_m": 11.0,
        "trunk_ratio": 0.25,
        "transmissivity": 0.03,
        "phenology": "deciduous",
    }
    tree.update(overrides)
    return tree


def add_tree(client: TestClient, tree: dict, base: int, **kwargs):
    body = {
        "base_scene_version": base,
        "edits": [{"operation": "add", "tree": tree}],
        "requested_result": {"time_indices": [1], "variables": ["utci", "tmrt"]},
    }
    headers = {"Idempotency-Key": kwargs.pop("key", f"edit-{base}-{tree['tree_id']}")}
    headers.update(kwargs.pop("headers", {}))
    return client.post(
        f"/api/v1/scenarios/{kwargs.pop('scenario_id', SCENARIO0(client))}/edits",
        json=body,
        headers=headers,
    )


# Keyed by the client OBJECT (weak), never by ``id(client)``: a recycled
# address made a fresh TestClient inherit a DEAD app's scenario id — the
# helper then 404s on ``scenario_not_found`` and the victim test shifts
# with allocation order (flake seen at cfb7b39 and beb3bd0, different
# victims). A WeakKeyDictionary entry dies with its client, so cross-test
# reuse of this helper can only ever hit a LIVE app's own scenario.
SCENARIO_CACHE: "weakref.WeakKeyDictionary[TestClient, str]" = (
    weakref.WeakKeyDictionary()
)


def SCENARIO0(client: TestClient) -> str:
    """Create (or reuse) one scenario on this client for helper requests."""
    if client not in SCENARIO_CACHE:
        response = client.post(
            "/api/v1/scenarios",
            json={"site_id": SITE_ID, "name": "helper", "initial_state": "baseline"},
        )
        SCENARIO_CACHE[client] = response.json()["scenario_id"]
    return SCENARIO_CACHE[client]


def wait_for_job(
    client: TestClient, job_id: str, statuses: tuple[str, ...] = ("complete",), timeout: float = 10.0
) -> dict:
    deadline = time.monotonic() + timeout
    last: dict = {}
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 200, response.text
        last = response.json()
        if last["status"] in statuses:
            return last
        time.sleep(0.01)
    pytest.fail(f"job {job_id} did not reach {statuses}; last status: {last}")


def wait_for_exact(client: TestClient, scenario_id: str, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        scenario = client.get(f"/api/v1/scenarios/{scenario_id}").json()
        if scenario["exact_result_version"] == scenario["scene_version"]:
            return scenario
        time.sleep(0.01)
    pytest.fail(f"scenario never became exact: {scenario}")


@pytest.fixture(autouse=True)
def _clear_scenario_cache():
    SCENARIO_CACHE.clear()
    yield
    SCENARIO_CACHE.clear()


# ---------------------------------------------------------------------------
# API-001: idempotency
# ---------------------------------------------------------------------------


class TestScenarioDetailSiteGeometry:
    def test_get_scenario_exposes_site_geometry(self, client: TestClient) -> None:
        """The realtime client submits vegetation ops in world metres, but no
        endpoint exposed the site's grid placement — the studio could only
        build a span-local frame (x_m in [0, span]) whose ops land outside
        every real site raster (UTM origins). The scenario detail carries the
        geometry the scenario is already PINNED to, so the client's
        uv→world mapping can be exact from the first gesture (no dependence
        on calibration samples from funnelled legacy ops)."""
        scenario_id = SCENARIO0(client)
        body = client.get(f"/api/v1/scenarios/{scenario_id}").json()
        assert body["site_geometry"] == {
            "rows": ROWS,
            "cols": COLS,
            "pixel_size_m": PIXEL,
            "origin_x_m": ORIGIN[0],
            "origin_y_m": ORIGIN[1],
            "time_steps": TIME_STEPS,
        }


class TestIdempotency:
    def test_create_replay_returns_same_scenario(self, client: TestClient) -> None:
        body = {"site_id": SITE_ID, "name": "demo", "initial_state": "baseline"}
        headers = {"Idempotency-Key": "create-demo-01"}
        first = client.post("/api/v1/scenarios", json=body, headers=headers)
        assert first.status_code == 201, first.text
        second = client.post("/api/v1/scenarios", json=body, headers=headers)
        assert second.status_code == 201
        assert first.json() == second.json()
        # Single mutation: exactly one scenario exists.
        assert client.get(f"/api/v1/scenarios/{first.json()['scenario_id']}").status_code == 200
        metrics = client.get("/metrics").json()
        assert metrics["scenarios"] == 1

    def test_create_key_reused_with_different_body(self, client: TestClient) -> None:
        headers = {"Idempotency-Key": "create-42"}
        client.post(
            "/api/v1/scenarios",
            json={"site_id": SITE_ID, "name": "one"},
            headers=headers,
        )
        conflict = client.post(
            "/api/v1/scenarios",
            json={"site_id": SITE_ID, "name": "different"},
            headers=headers,
        )
        assert conflict.status_code == 409
        error = conflict.json()["error"]
        assert error["code"] == "idempotency_key_reused"
        assert error["request_id"].startswith("req_")

    def test_edit_replay_single_mutation_one_job(
        self, client: TestClient, solver: FakeSolver
    ) -> None:
        scenario_id = SCENARIO0(client)
        headers = {"Idempotency-Key": "edit-42"}
        body = {
            "base_scene_version": 0,
            "edits": [{"operation": "add", "tree": make_tree()}],
        }
        first = client.post(
            f"/api/v1/scenarios/{scenario_id}/edits", json=body, headers=headers
        )
        assert first.status_code == 202, first.text
        assert first.headers["ETag"] == '"scene-version-1"'
        second = client.post(
            f"/api/v1/scenarios/{scenario_id}/edits", json=body, headers=headers
        )
        assert second.status_code == 202
        assert second.json() == first.json()
        assert second.headers["ETag"] == first.headers["ETag"]
        scenario = client.get(f"/api/v1/scenarios/{scenario_id}").json()
        assert scenario["scene_version"] == 1
        assert len(scenario["trees"]) == 1
        wait_for_exact(client, scenario_id)
        assert len(solver.calls) == 1

    def test_edit_key_reused_with_different_body(self, client: TestClient) -> None:
        scenario_id = SCENARIO0(client)
        headers = {"Idempotency-Key": "edit-77"}
        client.post(
            f"/api/v1/scenarios/{scenario_id}/edits",
            json={"base_scene_version": 0, "edits": [{"operation": "add", "tree": make_tree("t1")}]},
            headers=headers,
        )
        conflict = client.post(
            f"/api/v1/scenarios/{scenario_id}/edits",
            json={"base_scene_version": 0, "edits": [{"operation": "add", "tree": make_tree("t2")}]},
            headers=headers,
        )
        assert conflict.status_code == 409
        assert conflict.json()["error"]["code"] == "idempotency_key_reused"

    def test_reset_replay_single_bump(self, client: TestClient) -> None:
        scenario_id = SCENARIO0(client)
        response = add_tree(client, make_tree(), 0)
        assert response.status_code == 202
        wait_for_exact(client, scenario_id)
        headers = {"Idempotency-Key": "reset-1", "If-Match": '"scene-version-1"'}
        first = client.post(f"/api/v1/scenarios/{scenario_id}/reset", headers=headers)
        assert first.status_code == 200, first.text
        second = client.post(f"/api/v1/scenarios/{scenario_id}/reset", headers=headers)
        assert second.status_code == 200
        assert second.json() == first.json()
        scenario = client.get(f"/api/v1/scenarios/{scenario_id}").json()
        assert scenario["scene_version"] == 2  # one bump only
        assert scenario["exact_result_version"] == 2


# ---------------------------------------------------------------------------
# API-002: optimistic concurrency and explicit conflicts
# ---------------------------------------------------------------------------


class TestOptimisticConcurrency:
    def test_stale_base_version_conflict_envelope(self, client: TestClient) -> None:
        scenario_id = SCENARIO0(client)
        first = add_tree(client, make_tree("t1"), base=0)
        assert first.status_code == 202
        stale = add_tree(client, make_tree("t2"), base=0)
        assert stale.status_code == 409
        error = stale.json()["error"]
        assert error["code"] == "scene_version_conflict"
        assert error["current_scene_version"] == 1
        assert error["scenario_url"] == f"/api/v1/scenarios/{scenario_id}"
        assert "request_id" in error
        # The client reloads authoritative state and reapplies at the new base.
        retried = add_tree(client, make_tree("t2"), base=1)
        assert retried.status_code == 202
        assert retried.json()["scene_version"] == 2

    def test_if_match_stale_conflict(self, client: TestClient) -> None:
        scenario_id = SCENARIO0(client)
        add_tree(client, make_tree("t1"), base=0)
        # If-Match agrees with the (now stale) body base: the store's
        # optimistic check is what rejects it.
        body = {
            "base_scene_version": 0,
            "edits": [{"operation": "add", "tree": make_tree("t2")}],
        }
        stale = client.post(
            f"/api/v1/scenarios/{scenario_id}/edits",
            json=body,
            headers={"If-Match": '"scene-version-0"'},
        )
        assert stale.status_code == 409
        assert stale.json()["error"]["code"] == "scene_version_conflict"
        assert stale.json()["error"]["current_scene_version"] == 1

    def test_if_match_disagreeing_with_body(self, client: TestClient) -> None:
        scenario_id = SCENARIO0(client)
        response = client.post(
            f"/api/v1/scenarios/{scenario_id}/edits",
            json={"base_scene_version": 0, "edits": [{"operation": "add", "tree": make_tree()}]},
            headers={"If-Match": '"scene-version-5"'},
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_request"

    def test_scenario_isolation(self, client: TestClient) -> None:
        first_id = client.post(
            "/api/v1/scenarios", json={"site_id": SITE_ID, "name": "a"}
        ).json()["scenario_id"]
        second_id = client.post(
            "/api/v1/scenarios", json={"site_id": SITE_ID, "name": "b"}
        ).json()["scenario_id"]
        assert first_id != second_id
        response_a = add_tree(client, make_tree("tree-a"), base=0, scenario_id=first_id)
        response_b = add_tree(client, make_tree("tree-b"), base=0, scenario_id=second_id)
        assert response_a.status_code == response_b.status_code == 202
        wait_for_exact(client, first_id)
        wait_for_exact(client, second_id)
        trees_a = client.get(f"/api/v1/scenarios/{first_id}").json()["trees"]
        trees_b = client.get(f"/api/v1/scenarios/{second_id}").json()["trees"]
        assert [t["tree_id"] for t in trees_a] == ["tree-a"]
        assert [t["tree_id"] for t in trees_b] == ["tree-b"]
        # A's result version is not visible under B and vice versa.
        assert client.get(f"/api/v1/scenarios/{second_id}/results/1").status_code == 200
        assert client.get(f"/api/v1/scenarios/{first_id}/results/1").status_code == 200
        manifest_a = client.get(f"/api/v1/scenarios/{first_id}/results/1").json()
        manifest_b = client.get(f"/api/v1/scenarios/{second_id}/results/1").json()
        assert manifest_a["scenario_id"] == first_id
        assert manifest_b["scenario_id"] == second_id
        # Fake values differ per scenario tree count? Both have one tree; the
        # payloads are equal-valued but the manifests stay scenario-scoped.
        assert manifest_a["checksum"] == manifest_b["checksum"]  # same fake values
        assert manifest_a["payload_url"] != manifest_b["payload_url"]

    def test_unknown_scenario_and_job(self, client: TestClient) -> None:
        missing = client.get("/api/v1/scenarios/scn_missing")
        assert missing.status_code == 404
        assert missing.json()["error"]["code"] == "scenario_not_found"
        missing_job = client.get("/api/v1/jobs/job_missing")
        assert missing_job.status_code == 404
        assert missing_job.json()["error"]["code"] == "job_not_found"
        missing_result = client.get("/api/v1/scenarios/scn_missing/results/0")
        assert missing_result.status_code == 404
        assert missing_result.json()["error"]["code"] == "scenario_not_found"

    def test_invalid_tree_geometry_field_paths(self, client: TestClient) -> None:
        scenario_id = SCENARIO0(client)
        cases = [
            (make_tree(canopy_diameter_m=40.0), "edits[0].tree.canopy_diameter_m"),
            (make_tree(height_m=50.0), "edits[0].tree.height_m"),
            (make_tree(u=1.5), "edits[0].tree.u"),
            (make_tree(v=-0.1), "edits[0].tree.v"),
            (make_tree(trunk_ratio=1.0), "edits[0].tree.trunk_ratio"),
            (make_tree(transmissivity=1.5), "edits[0].tree.transmissivity"),
        ]
        for tree, field in cases:
            response = client.post(
                f"/api/v1/scenarios/{scenario_id}/edits",
                json={"base_scene_version": 0, "edits": [{"operation": "add", "tree": tree}]},
            )
            assert response.status_code == 422, (field, response.text)
            error = response.json()["error"]
            assert error["code"] == "invalid_tree_geometry"
            assert error["field"] == field, error

    def test_unknown_component_type_rejected(self, client: TestClient) -> None:
        scenario_id = SCENARIO0(client)
        response = client.post(
            f"/api/v1/scenarios/{scenario_id}/edits",
            json={
                "base_scene_version": 0,
                "edits": [{"operation": "add", "tree": make_tree(component_type="mega_tree")}],
            },
        )
        assert response.status_code == 400
        assert response.json()["error"]["field"] == "edits[0].tree.component_type"

    def test_delete_unknown_tree_rejected(self, client: TestClient) -> None:
        scenario_id = SCENARIO0(client)
        response = client.post(
            f"/api/v1/scenarios/{scenario_id}/edits",
            json={
                "base_scene_version": 0,
                "edits": [{"operation": "delete", "tree_id": "ghost"}],
            },
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_request"
        assert response.json()["error"]["field"] == "edits[0].tree_id"

    def test_malformed_body_is_invalid_request(self, client: TestClient) -> None:
        scenario_id = SCENARIO0(client)
        response = client.post(
            f"/api/v1/scenarios/{scenario_id}/edits",
            json={"base_scene_version": "not-an-int", "edits": []},
        )
        assert response.status_code == 400
        error = response.json()["error"]
        assert error["code"] == "invalid_request"
        assert error["field"]

    def test_rate_limited(self, tmp_path: Path, solver: FakeSolver, site_cache: Path) -> None:
        app = make_app(
            tmp_path, solver, site_cache=site_cache, edits_per_minute=1
        )
        with TestClient(app) as limited:
            scenario_id = limited.post(
                "/api/v1/scenarios", json={"site_id": SITE_ID}
            ).json()["scenario_id"]
            url = f"/api/v1/scenarios/{scenario_id}/edits"
            body = {"base_scene_version": 0, "edits": [{"operation": "add", "tree": make_tree()}]}
            first = limited.post(url, json=body, headers={"Idempotency-Key": "k1"})
            assert first.status_code == 202
            second = limited.post(
                url,
                json={
                    "base_scene_version": 1,
                    "edits": [{"operation": "add", "tree": make_tree("t2")}],
                },
                headers={"Idempotency-Key": "k2"},
            )
            assert second.status_code == 429
            assert second.json()["error"]["code"] == "rate_limited"

    def test_site_limit_exceeded(self, tmp_path: Path, solver: FakeSolver, site_cache: Path) -> None:
        app = make_app(tmp_path, solver, site_cache=site_cache, max_scenarios=1)
        with TestClient(app) as limited:
            first = limited.post("/api/v1/scenarios", json={"site_id": SITE_ID})
            assert first.status_code == 201
            second = limited.post("/api/v1/scenarios", json={"site_id": SITE_ID})
            assert second.status_code == 403
            assert second.json()["error"]["code"] == "site_limit_exceeded"

    def test_unknown_site_rejected(self, client: TestClient) -> None:
        response = client.post("/api/v1/scenarios", json={"site_id": "not-a-site"})
        assert response.status_code == 400
        error = response.json()["error"]
        assert error["code"] == "invalid_request"
        assert error["field"] == "site_id"


# ---------------------------------------------------------------------------
# API-003: manifest + payload validation
# ---------------------------------------------------------------------------


class TestResults:
    def test_baseline_result_is_exact_scene_version_zero(
        self, client: TestClient, site_cache: Path
    ) -> None:
        scenario_id = SCENARIO0(client)
        scenario = client.get(f"/api/v1/scenarios/{scenario_id}").json()
        assert scenario["scene_version"] == 0
        assert scenario["exact_result_version"] == 0
        assert scenario["status"] == "exact"
        assert scenario["trees"] == []
        assert scenario["model_scope"]["limitations"]
        manifest_response = client.get(scenario["result_manifest_url"])
        assert manifest_response.status_code == 200
        manifest = manifest_response.json()
        assert manifest["schema_version"] == 1
        assert manifest["scene_version"] == 0
        assert manifest["exact"] is True
        assert manifest["site_cache_version"] == f"{SITE_ID}:cache-1"
        assert manifest["window"] == {
            "row_start": 0, "row_stop": ROWS, "col_start": 0, "col_stop": COLS,
        }
        assert manifest["time_indices"] == list(range(TIME_STEPS))
        assert {v["name"] for v in manifest["variables"]} == {"utci", "tmrt"}
        payload_response = client.get(manifest["payload_url"])
        assert payload_response.status_code == 200
        assert payload_response.headers["Content-Type"].startswith(
            patch_codec.PATCH_MEDIA_TYPE
        )
        assert payload_response.headers["X-SOLWEIG-Scene-Version"] == "0"
        assert payload_response.headers["X-SOLWEIG-Schema-Version"] == "1"
        assert payload_response.headers["ETag"] == patch_codec.payload_etag(
            manifest["checksum"]
        )
        assert int(payload_response.headers["Content-Length"]) == len(payload_response.content)
        arrays = patch_codec.decode_payload(manifest, payload_response.content)
        np.testing.assert_array_equal(arrays["utci"], load_baseline(site_cache, "utci"))
        np.testing.assert_array_equal(arrays["tmrt"], load_baseline(site_cache, "tmrt"))

    def test_job_lifecycle_queued_running_complete(
        self, client: TestClient, solver: FakeSolver
    ) -> None:
        scenario_id = SCENARIO0(client)
        gate = threading.Event()
        solver.gate = gate
        response = add_tree(client, make_tree(), base=0)
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        assert solver.entered.wait(timeout=5.0)  # solver is mid-flight
        running = client.get(f"/api/v1/jobs/{job_id}").json()
        assert running["status"] == "running"
        assert running["scenario_id"] == scenario_id
        assert running["target_scene_version"] == 1
        assert "stage" in running and "progress" in running and "mode" in running
        gate.set()
        complete = wait_for_job(client, job_id)
        assert complete["status"] == "complete"
        assert complete["result_manifest_url"].endswith(
            f"/scenarios/{scenario_id}/results/1"
        )
        assert complete["metrics"]["mode"] == "local"
        assert complete["metrics"]["duration_ms"] >= 0
        assert complete["metrics"]["window_fraction"] > 0

    def test_queued_shape_before_dispatch(
        self, tmp_path: Path, solver: FakeSolver, site_cache: Path
    ) -> None:
        app = make_app(
            tmp_path, solver, site_cache=site_cache, start_worker=False
        )
        with TestClient(app) as frozen:
            scenario_id = frozen.post(
                "/api/v1/scenarios", json={"site_id": SITE_ID}
            ).json()["scenario_id"]
            response = add_tree_frozen(frozen, scenario_id, base=0)
            job_id = response.json()["job_id"]
            queued = frozen.get(f"/api/v1/jobs/{job_id}").json()
            assert queued["status"] == "queued"
            assert queued["queue_position"] == 1
            assert queued["progress"] is None
            assert queued["target_scene_version"] == 1

    def test_manifest_matches_payload_and_fake_values(
        self, client: TestClient, solver: FakeSolver
    ) -> None:
        scenario_id = SCENARIO0(client)
        add_tree(client, make_tree(), base=0)
        scenario = wait_for_exact(client, scenario_id)
        assert scenario["exact_result_version"] == 1
        manifest = client.get(f"/api/v1/scenarios/{scenario_id}/results/1").json()
        assert manifest["scene_version"] == 1
        assert manifest["exact"] is True
        assert manifest["time_indices"] == [1]
        assert [v["name"] for v in manifest["variables"]] == ["utci", "tmrt"]
        assert manifest["variables"][0]["shape"] == [1, 8, 12]
        assert manifest["window"] == {"row_start": 4, "row_stop": 12, "col_start": 6, "col_stop": 18}
        payload = client.get(f"/api/v1/scenarios/{scenario_id}/results/1/payload")
        assert payload.headers["ETag"] == patch_codec.payload_etag(manifest["checksum"])
        assert payload.headers["X-SOLWEIG-Scene-Version"] == "1"
        arrays = patch_codec.decode_payload(manifest, payload.content)
        expected = 25.0 + 1 + 1  # target version 1 + one tree
        np.testing.assert_allclose(arrays["utci"], expected)
        np.testing.assert_allclose(arrays["tmrt"], expected + 1)
        assert manifest["metrics"]["mean_utci_delta_c"] == -0.84
        assert manifest["limitations"]

    def test_result_not_ready_for_future_version(self, client: TestClient) -> None:
        scenario_id = SCENARIO0(client)
        response = client.get(f"/api/v1/scenarios/{scenario_id}/results/9")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "result_not_ready"

    def test_payload_304_on_matching_etag(self, client: TestClient) -> None:
        scenario_id = SCENARIO0(client)
        manifest = client.get(f"/api/v1/scenarios/{scenario_id}/results/0").json()
        etag = patch_codec.payload_etag(manifest["checksum"])
        response = client.get(
            f"/api/v1/scenarios/{scenario_id}/results/0/payload",
            headers={"If-None-Match": etag},
        )
        assert response.status_code == 304


class TestPayloadNegotiation:
    """Binary payload Accept negotiation (zstd default, identity fallback)."""

    ZSTD = "application/vnd.solweig.patch+zstd"
    IDENTITY = "application/vnd.solweig.patch+identity"

    @staticmethod
    def _served(client: TestClient, scenario_id: str, accept: str | None):
        headers = {"Accept": accept} if accept is not None else {}
        return client.get(
            f"/api/v1/scenarios/{scenario_id}/results/0/payload",
            headers=headers,
        )

    def test_both_encodings_validate_and_decode_identically(
        self, client: TestClient
    ) -> None:
        import hashlib

        scenario_id = SCENARIO0(client)
        manifest = client.get(f"/api/v1/scenarios/{scenario_id}/results/0").json()

        zstd = self._served(client, scenario_id, self.ZSTD)
        assert zstd.status_code == 200
        assert zstd.headers["Content-Type"].startswith(self.ZSTD)
        # zstd checksum: over the served (compressed) bytes == manifest.
        assert zstd.headers["X-SOLWEIG-Checksum"] == manifest["checksum"]
        assert zstd.headers["ETag"] == patch_codec.payload_etag(manifest["checksum"])
        assert hashlib.sha256(zstd.content).hexdigest() in manifest["checksum"]

        identity = self._served(client, scenario_id, self.IDENTITY)
        assert identity.status_code == 200
        assert identity.headers["Content-Type"].startswith(self.IDENTITY)
        # identity checksum: over the served (uncompressed) bytes — different
        # value, same contract: sha256(body) == X-SOLWEIG-Checksum == ETag.
        identity_checksum = (
            "sha256:" + hashlib.sha256(identity.content).hexdigest()
        )
        assert identity.headers["X-SOLWEIG-Checksum"] == identity_checksum
        assert identity.headers["ETag"] == patch_codec.payload_etag(identity_checksum)
        assert identity_checksum != manifest["checksum"]
        assert len(identity.content) > len(zstd.content)

        # Both encodings decode to byte-identical arrays.
        zstd_arrays = patch_codec.decode_payload(manifest, zstd.content)
        identity_arrays = patch_codec.decode_identity_payload(
            manifest, identity.content
        )
        assert set(zstd_arrays) == set(identity_arrays)
        for name, array in zstd_arrays.items():
            np.testing.assert_array_equal(array, identity_arrays[name])

    def test_default_and_wildcard_serve_zstd(self, client: TestClient) -> None:
        scenario_id = SCENARIO0(client)
        for accept in (None, "*/*", "application/*"):
            response = self._served(client, scenario_id, accept)
            assert response.status_code == 200, accept
            assert response.headers["Content-Type"].startswith(self.ZSTD), accept

    def test_identity_chosen_only_when_zstd_not_acceptable(
        self, client: TestClient
    ) -> None:
        scenario_id = SCENARIO0(client)
        both = self._served(
            client, scenario_id, f"{self.ZSTD}, {self.IDENTITY}"
        )
        assert both.headers["Content-Type"].startswith(self.ZSTD)
        identity = self._served(client, scenario_id, self.IDENTITY)
        assert identity.headers["Content-Type"].startswith(self.IDENTITY)

    def test_unacceptable_encoding_is_406(self, client: TestClient) -> None:
        scenario_id = SCENARIO0(client)
        response = self._served(client, scenario_id, "application/json")
        assert response.status_code == 406
        assert response.json()["error"]["code"] == "not_acceptable"

    def test_304_per_encoding_etag(self, client: TestClient) -> None:
        scenario_id = SCENARIO0(client)
        identity = self._served(client, scenario_id, self.IDENTITY)
        etag = identity.headers["ETag"]
        again = client.get(
            f"/api/v1/scenarios/{scenario_id}/results/0/payload",
            headers={"Accept": self.IDENTITY, "If-None-Match": etag},
        )
        assert again.status_code == 304
        assert again.headers["ETag"] == etag
        # The identity ETag must NOT satisfy a zstd request: encodings have
        # distinct validators.
        zstd_request = client.get(
            f"/api/v1/scenarios/{scenario_id}/results/0/payload",
            headers={"Accept": self.ZSTD, "If-None-Match": etag},
        )
        assert zstd_request.status_code == 200
        assert zstd_request.headers["ETag"] != etag


def add_tree_frozen(client: TestClient, scenario_id: str, *, base: int) -> Any:
    return client.post(
        f"/api/v1/scenarios/{scenario_id}/edits",
        json={
            "base_scene_version": base,
            "edits": [{"operation": "add", "tree": make_tree()}],
        },
        headers={"Idempotency-Key": f"frozen-{base}"},
    )


# ---------------------------------------------------------------------------
# API-004: supersession, stale publish guard, restart recovery
# ---------------------------------------------------------------------------


class TestSupersession:
    def test_newer_edit_supersedes_queued_job(
        self, tmp_path: Path, solver: FakeSolver, site_cache: Path
    ) -> None:
        # Long coalescing window keeps job 1 queued while the second edit lands.
        app = make_app(
            tmp_path, solver, site_cache=site_cache, coalescing_window_ms=1500.0
        )
        with TestClient(app) as client:
            scenario_id = client.post(
                "/api/v1/scenarios", json={"site_id": SITE_ID}
            ).json()["scenario_id"]
            first = client.post(
                f"/api/v1/scenarios/{scenario_id}/edits",
                json={
                    "base_scene_version": 0,
                    "edits": [{"operation": "add", "tree": make_tree("t1")}],
                },
            )
            second = client.post(
                f"/api/v1/scenarios/{scenario_id}/edits",
                json={
                    "base_scene_version": 1,
                    "edits": [{"operation": "add", "tree": make_tree("t2")}],
                },
            )
            assert first.status_code == second.status_code == 202
            job1 = first.json()["job_id"]
            job2 = second.json()["job_id"]
            superseded = wait_for_job(client, job1, statuses=("superseded",))
            assert superseded["status"] == "superseded"
            complete = wait_for_job(client, job2, statuses=("complete", "superseded"))
            assert complete["status"] == "complete"
            scenario = wait_for_exact(client, scenario_id)
            assert scenario["scene_version"] == 2
            assert scenario["exact_result_version"] == 2
            # The superseded job never ran the solver.
            assert len(solver.calls) == 1
            # Its version was never published.
            missing = client.get(f"/api/v1/scenarios/{scenario_id}/results/1")
            assert missing.status_code == 404
            assert missing.json()["error"]["code"] == "result_not_ready"

    def test_stale_solver_result_never_published(
        self, client: TestClient, solver: FakeSolver
    ) -> None:
        scenario_id = SCENARIO0(client)
        solver.scene_version_offset = -1  # solver claims revision 0 for target 1
        response = add_tree(client, make_tree(), base=0)
        job_id = response.json()["job_id"]
        outcome = wait_for_job(client, job_id, statuses=("superseded", "failed", "complete"))
        assert outcome["status"] == "superseded"
        scenario = client.get(f"/api/v1/scenarios/{scenario_id}").json()
        assert scenario["exact_result_version"] == 0  # baseline still current
        assert scenario["status"] == "refining"
        assert (
            client.get(f"/api/v1/scenarios/{scenario_id}/results/1").status_code == 404
        )

    def test_running_job_superseded_when_edit_lands(
        self, client: TestClient, solver: FakeSolver
    ) -> None:
        scenario_id = SCENARIO0(client)
        gate = threading.Event()
        solver.gate = gate
        first = add_tree(client, make_tree("t1"), base=0)
        assert first.status_code == 202
        assert solver.entered.wait(timeout=5.0)  # job 1 is mid-solve
        second = add_tree(client, make_tree("t2"), base=1)
        assert second.status_code == 202
        gate.set()
        job1 = first.json()["job_id"]
        job2 = second.json()["job_id"]
        outcome1 = wait_for_job(client, job1, statuses=("superseded",))
        assert outcome1["status"] == "superseded"
        complete2 = wait_for_job(client, job2)
        assert complete2["status"] == "complete"
        scenario = wait_for_exact(client, scenario_id)
        assert scenario["exact_result_version"] == 2
        assert len(solver.calls) == 2

    def test_solver_failure_marks_job_failed(
        self, client: TestClient, solver: FakeSolver
    ) -> None:
        scenario_id = SCENARIO0(client)
        solver.fail = True
        response = add_tree(client, make_tree(), base=0)
        outcome = wait_for_job(client, response.json()["job_id"], statuses=("failed",))
        assert outcome["status"] == "failed"
        assert outcome["error"]["code"] == "job_failed"
        scenario = client.get(f"/api/v1/scenarios/{scenario_id}").json()
        assert scenario["status"] == "refining"  # design state preserved

    def test_restart_recovery_resumes_queued_job(
        self, tmp_path: Path, solver: FakeSolver, site_cache: Path
    ) -> None:
        state_root = tmp_path / "state"
        app1 = make_app(
            tmp_path,
            solver,
            site_cache=site_cache,
            state_root=state_root,
            start_worker=False,
        )
        with TestClient(app1) as client1:
            scenario_id = client1.post(
                "/api/v1/scenarios", json={"site_id": SITE_ID}
            ).json()["scenario_id"]
            response = add_tree_frozen(client1, scenario_id, base=0)
            assert response.status_code == 202
            job_id = response.json()["job_id"]
            assert client1.get(f"/api/v1/jobs/{job_id}").json()["status"] == "queued"
        # App closed (store closed, runner thread never started). Reopen from
        # the same SQLite file: the queued job must resume and complete.
        assert len(solver.calls) == 0
        app2 = make_app(
            tmp_path, solver, site_cache=site_cache, state_root=state_root
        )
        with TestClient(app2) as client2:
            scenario = wait_for_exact(client2, scenario_id)
            assert scenario["scene_version"] == 1
            assert scenario["exact_result_version"] == 1
            job = wait_for_job(client2, job_id)
            assert job["status"] == "complete"
            manifest = client2.get(f"/api/v1/scenarios/{scenario_id}/results/1").json()
            assert manifest["scene_version"] == 1
        assert len(solver.calls) == 1

    def test_restart_recovery_resumes_running_job(
        self, tmp_path: Path, solver: FakeSolver, site_cache: Path
    ) -> None:
        """A durable row found 'running' at startup resumes and completes.

        The previous test version waited for a graceful shutdown to observe
        the job running — but shutdown lets the runner finish in-flight
        dispatch, so recovery was never exercised (the test passed even with
        requeue-enqueue deleted). This version forces the crashed state
        directly: the job row is flipped to 'running' with no live runner,
        exactly like a process that died mid-solve.
        """
        state_root = tmp_path / "state"
        app1 = make_app(
            tmp_path,
            solver,
            site_cache=site_cache,
            state_root=state_root,
            start_worker=False,  # nothing dispatches: simulate a crash first
        )
        with TestClient(app1) as client1:
            scenario_id = client1.post(
                "/api/v1/scenarios", json={"site_id": SITE_ID}
            ).json()["scenario_id"]
            response = add_tree_frozen(client1, scenario_id, base=0)
            assert response.status_code == 202
            job_id = response.json()["job_id"]
            # Simulate the crash mid-solve: the durable row says running.
            app1.state.context.store.mark_job_running(job_id)
            assert client1.get(f"/api/v1/jobs/{job_id}").json()["status"] == "running"
        assert len(solver.calls) == 0  # nothing ran before the "crash"
        app2 = make_app(
            tmp_path, solver, site_cache=site_cache, state_root=state_root
        )
        with TestClient(app2) as client2:
            job = wait_for_job(client2, job_id)
            assert job["status"] == "complete"
            wait_for_exact(client2, scenario_id)
        assert len(solver.calls) == 1  # resumed exactly once

    def test_crash_between_publish_and_finish_recovers(
        self, tmp_path: Path, solver: FakeSolver, site_cache: Path
    ) -> None:
        """Re-running a job whose result was already published is idempotent.

        Simulates a crash in the window between ``publish_result`` and
        ``finish_job``: the result row exists, the job row still says
        'running'. On restart the job re-runs, hits the existing result, and
        — because the solver is deterministic — completes instead of looping
        as a permanent 'running' zombie.
        """
        state_root = tmp_path / "state"
        app1 = make_app(
            tmp_path, solver, site_cache=site_cache, state_root=state_root
        )
        with TestClient(app1) as client1:
            scenario_id = client1.post(
                "/api/v1/scenarios", json={"site_id": SITE_ID}
            ).json()["scenario_id"]
            response = add_tree_frozen(client1, scenario_id, base=0)
            job_id = response.json()["job_id"]
            assert wait_for_job(client1, job_id)["status"] == "complete"
            assert client1.get(
                f"/api/v1/scenarios/{scenario_id}/results/1"
            ).status_code == 200
        calls_before_restart = len(solver.calls)
        # Flip the finalized row back to 'running' (crash window).
        import sqlite3

        connection = sqlite3.connect(state_root / "store.sqlite3")
        try:
            connection.execute(
                "UPDATE jobs SET status = 'running', finished_at = NULL "
                "WHERE job_id = ?",
                (job_id,),
            )
            connection.commit()
        finally:
            connection.close()
        app2 = make_app(
            tmp_path, solver, site_cache=site_cache, state_root=state_root
        )
        with TestClient(app2) as client2:
            outcome = wait_for_job(client2, job_id)
            assert outcome["status"] == "complete"
            assert outcome["result_manifest_url"].endswith(
                f"/scenarios/{scenario_id}/results/1"
            )
            # The published result is untouched and still servable.
            manifest = client2.get(
                f"/api/v1/scenarios/{scenario_id}/results/1"
            ).json()
            payload = client2.get(
                f"/api/v1/scenarios/{scenario_id}/results/1/payload"
            )
            assert payload.status_code == 200
            arrays = patch_codec.decode_payload(manifest, payload.content)
            assert arrays["utci"].shape[1:] == (8, 12)
            scenario = client2.get(f"/api/v1/scenarios/{scenario_id}").json()
            assert scenario["exact_result_version"] == 1
        assert len(solver.calls) == calls_before_restart + 1  # re-ran once


# ---------------------------------------------------------------------------
# Reset, edits ledger, exports, ops endpoints
# ---------------------------------------------------------------------------


class TestResetAndLedger:
    def test_reset_restores_baseline_without_worker(
        self, client: TestClient, solver: FakeSolver, site_cache: Path
    ) -> None:
        scenario_id = SCENARIO0(client)
        add_tree(client, make_tree(), base=0)
        wait_for_exact(client, scenario_id)
        calls_before = len(solver.calls)
        response = client.post(
            f"/api/v1/scenarios/{scenario_id}/reset",
            headers={"Idempotency-Key": "reset-9", "If-Match": '"scene-version-1"'},
        )
        assert response.status_code == 200, response.text
        scenario = response.json()
        assert scenario["scene_version"] == 2
        assert scenario["exact_result_version"] == 2
        assert scenario["status"] == "exact"
        assert scenario["trees"] == []
        assert len(solver.calls) == calls_before  # no worker run for reset
        manifest = client.get(f"/api/v1/scenarios/{scenario_id}/results/2").json()
        assert manifest["scene_version"] == 2
        assert manifest["exact"] is True
        payload = client.get(f"/api/v1/scenarios/{scenario_id}/results/2/payload")
        arrays = patch_codec.decode_payload(manifest, payload.content)
        np.testing.assert_array_equal(arrays["utci"], load_baseline(site_cache, "utci"))

    def test_reset_with_stale_if_match_conflicts(self, client: TestClient) -> None:
        scenario_id = SCENARIO0(client)
        add_tree(client, make_tree(), base=0)
        response = client.post(
            f"/api/v1/scenarios/{scenario_id}/reset",
            headers={"If-Match": '"scene-version-0"'},
        )
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "scene_version_conflict"
        assert response.json()["error"]["current_scene_version"] == 1

    def test_move_and_update_replacement_objects(
        self, client: TestClient
    ) -> None:
        scenario_id = SCENARIO0(client)
        add_tree(client, make_tree("t1", u=0.2, v=0.2), base=0)
        moved = client.post(
            f"/api/v1/scenarios/{scenario_id}/edits",
            json={
                "base_scene_version": 1,
                "edits": [{"operation": "move", "tree": make_tree("t1", u=0.7, v=0.6)}],
            },
            headers={"Idempotency-Key": "move-1"},
        )
        assert moved.status_code == 202, moved.text
        updated = client.post(
            f"/api/v1/scenarios/{scenario_id}/edits",
            json={
                "base_scene_version": 2,
                "edits": [
                    {"operation": "update", "tree": make_tree("t1", u=0.7, v=0.6, height_m=25.0)}
                ],
            },
            headers={"Idempotency-Key": "update-1"},
        )
        assert updated.status_code == 202
        deleted = client.post(
            f"/api/v1/scenarios/{scenario_id}/edits",
            json={
                "base_scene_version": 3,
                "edits": [{"operation": "delete", "tree_id": "t1"}],
            },
            headers={"Idempotency-Key": "delete-1"},
        )
        assert deleted.status_code == 202
        scenario = wait_for_exact(client, scenario_id)
        assert scenario["scene_version"] == 4
        assert scenario["trees"] == []
        store = client.app.state.context.store
        events = store.list_events(scenario_id)
        assert [event["operation"] for event in events] == ["add", "move", "update", "delete"]
        move_event = events[1]
        assert move_event["old_tree"]["u"] == 0.2
        assert move_event["new_tree"]["u"] == 0.7
        assert move_event["event_id"].startswith("evt_")

    def test_export_stub_records_intent(self, client: TestClient) -> None:
        scenario_id = SCENARIO0(client)
        add_tree(client, make_tree(), base=0)
        wait_for_exact(client, scenario_id)
        response = client.post(
            f"/api/v1/scenarios/{scenario_id}/exports",
            json={"scene_version": 1, "format": "cog", "variables": ["utci"]},
        )
        assert response.status_code == 202
        body = response.json()
        assert body["export_id"].startswith("exp_")
        assert body["status"] == "accepted"
        assert "stub" in body["note"]  # v1 exports are explicitly a stub
        # Exporting an unpublished version is refused.
        missing = client.post(
            f"/api/v1/scenarios/{scenario_id}/exports",
            json={"scene_version": 5, "format": "cog"},
        )
        assert missing.status_code == 404
        assert missing.json()["error"]["code"] == "result_not_ready"

    def test_site_without_baseline_schedules_job(
        self, tmp_path: Path, solver: FakeSolver
    ) -> None:
        cache = make_site_cache(tmp_path, baseline=False)
        app = make_app(tmp_path, solver, site_cache=cache)
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/scenarios",
                json={"site_id": SITE_ID},
                headers={"Idempotency-Key": "create-no-baseline"},
            )
            assert response.status_code == 201
            body = response.json()
            assert body["status"] == "refining"
            assert body["job_id"].startswith("job_")
            # scene_version and exact_result_version both start at 0, so wait
            # on the baseline job itself rather than on version equality.
            outcome = wait_for_job(client, body["job_id"])
            assert outcome["status"] == "complete"
            scenario = client.get(f"/api/v1/scenarios/{body['scenario_id']}").json()
            assert scenario["status"] == "exact"
            assert scenario["exact_result_version"] == 0
            manifest = client.get(
                f"/api/v1/scenarios/{body['scenario_id']}/results/0"
            ).json()
            assert manifest["scene_version"] == 0
            assert manifest["metrics"]["mode"] == "local"  # faked full-solve

    def test_healthz_and_metrics(self, client: TestClient) -> None:
        health = client.get("/healthz")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        assert health.json()["sites"] == [SITE_ID]
        add_tree(client, make_tree(), base=0)
        wait_for_exact(client, SCENARIO0(client))
        metrics = client.get("/metrics").json()
        assert metrics["scenarios"] == 1
        assert metrics["jobs"]["complete"] >= 1
        assert metrics["results_published"] >= 2  # baseline + version 1


# ---------------------------------------------------------------------------
# Job cancellation ('cancelled' must be a reachable terminal status)
# ---------------------------------------------------------------------------


class TestJobCancellation:
    def test_cancel_running_job_discards_result(
        self, client: TestClient, solver: FakeSolver
    ) -> None:
        scenario_id = SCENARIO0(client)
        gate = threading.Event()
        solver.gate = gate
        response = add_tree(client, make_tree(), base=0)
        job_id = response.json()["job_id"]
        assert solver.entered.wait(timeout=5.0)  # mid-solve
        cancelled = client.post(f"/api/v1/jobs/{job_id}/cancel")
        assert cancelled.status_code == 200
        body = cancelled.json()
        assert body["status"] == "cancelled"
        assert body["error"]["code"] == "cancelled"
        gate.set()  # let the solver finish; the result must be discarded
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            job = client.get(f"/api/v1/jobs/{job_id}").json()
            if job["status"] == "cancelled":
                break
            time.sleep(0.01)
        assert job["status"] == "cancelled"
        missing = client.get(f"/api/v1/scenarios/{scenario_id}/results/1")
        assert missing.status_code == 404
        scenario = client.get(f"/api/v1/scenarios/{scenario_id}").json()
        assert scenario["scene_version"] == 1  # the edit still stands
        assert scenario["status"] == "refining"

    def test_cancel_queued_job(
        self, tmp_path: Path, solver: FakeSolver, site_cache: Path
    ) -> None:
        app = make_app(tmp_path, solver, site_cache=site_cache, start_worker=False)
        with TestClient(app) as frozen:
            scenario_id = frozen.post(
                "/api/v1/scenarios", json={"site_id": SITE_ID}
            ).json()["scenario_id"]
            response = add_tree_frozen(frozen, scenario_id, base=0)
            job_id = response.json()["job_id"]
            cancelled = frozen.post(f"/api/v1/jobs/{job_id}/cancel")
            assert cancelled.status_code == 200
            assert cancelled.json()["status"] == "cancelled"
        # Restart with a live runner: recovery only requeues queued/running
        # jobs, so the cancelled one is skipped entirely.
        app2 = make_app(tmp_path, solver, site_cache=site_cache)
        with TestClient(app2) as client2:
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                job = client2.get(f"/api/v1/jobs/{job_id}").json()
                assert job["status"] == "cancelled"
                time.sleep(0.05)
        assert len(solver.calls) == 0

    def test_cancel_finished_job_conflicts(self, client: TestClient) -> None:
        scenario_id = SCENARIO0(client)
        response = add_tree(client, make_tree(), base=0)
        job_id = response.json()["job_id"]
        wait_for_job(client, job_id)
        conflict = client.post(f"/api/v1/jobs/{job_id}/cancel")
        assert conflict.status_code == 409
        assert conflict.json()["error"]["code"] == "invalid_request"

    def test_cancel_unknown_job(self, client: TestClient) -> None:
        missing = client.post("/api/v1/jobs/job_missing/cancel")
        assert missing.status_code == 404
        assert missing.json()["error"]["code"] == "job_not_found"

    def test_recancel_cancelled_job_is_idempotent(
        self, client: TestClient, solver: FakeSolver
    ) -> None:
        """Retrying a successful cancel must not error out a retry loop."""
        scenario_id = SCENARIO0(client)
        gate = threading.Event()
        solver.gate = gate
        response = add_tree(client, make_tree(), base=0)
        job_id = response.json()["job_id"]
        assert solver.entered.wait(timeout=5.0)
        first = client.post(f"/api/v1/jobs/{job_id}/cancel")
        assert first.status_code == 200
        assert first.json()["status"] == "cancelled"
        for _ in range(2):
            replay = client.post(f"/api/v1/jobs/{job_id}/cancel")
            assert replay.status_code == 200, replay.text
            assert replay.json()["status"] == "cancelled"
        gate.set()
        final = wait_for_job(client, job_id, statuses=("cancelled",))
        assert final["status"] == "cancelled"

    def test_cancel_losing_race_reports_observed_terminal_status(
        self, tmp_path: Path, solver: FakeSolver, site_cache: Path
    ) -> None:
        """A cancel that read 'queued' but lost the race to a completion gets
        a 409 naming the observed status — never a 200 with someone else's
        outcome."""
        app = make_app(
            tmp_path, solver, site_cache=site_cache, start_worker=False
        )
        with TestClient(app) as frozen:
            scenario_id = frozen.post(
                "/api/v1/scenarios", json={"site_id": SITE_ID}
            ).json()["scenario_id"]
            job_id = add_tree_frozen(frozen, scenario_id, base=0).json()["job_id"]
            store = app.state.context.store
            # The job finishes out from under a cancel that already read
            # 'queued': the guarded UPDATE refuses to overwrite the outcome
            # and reports that nothing transitioned.
            assert store.finish_job(job_id, "complete") is True
            assert store.finish_job(job_id, "cancelled") is False
            assert store.require_job(job_id).status == "complete"
            cancel = frozen.post(f"/api/v1/jobs/{job_id}/cancel")
            assert cancel.status_code == 409
            assert cancel.json()["error"]["code"] == "invalid_request"
            assert "complete" in cancel.json()["error"]["message"]

    def test_publish_for_cancelled_job_refused_in_publish_txn(
        self, tmp_path: Path
    ) -> None:
        """The publish transaction itself refuses results for non-running
        jobs, closing the cancel-commits-between-guard-and-publish window."""
        from solweig_gpu.server.store import ResultNotPublishable, Store

        store = Store(
            tmp_path / "guard.sqlite3", results_root=tmp_path / "results"
        )
        try:
            record, job_id = store.create_scenario(site_id=SITE_ID, name="a")
            scenario_id = record.scenario_id
            store.mark_job_running(job_id)
            store.finish_job(
                job_id,
                "cancelled",
                error={"code": "cancelled", "message": "cancelled"},
            )
            with pytest.raises(ResultNotPublishable) as excinfo:
                store.publish_result(
                    scenario_id,
                    0,
                    manifest={"checksum": "sha256:0", "schema_version": 1},
                    payload=b"cancelled-job-bytes",
                    job_id=job_id,
                )
            assert excinfo.value.status == "cancelled"
            assert store.get_result(scenario_id, 0) is None  # nothing written
        finally:
            store.close()


# ---------------------------------------------------------------------------
# Contract sharpness: validation, fingerprints, cache mismatches, quotas
# ---------------------------------------------------------------------------


class TestContractEdges:
    def test_if_match_is_part_of_idempotency_fingerprint(
        self, client: TestClient
    ) -> None:
        """Same key + same body + different If-Match is a conflict, not a
        replay: the precondition changes what the mutation means."""
        scenario_id = SCENARIO0(client)
        url = f"/api/v1/scenarios/{scenario_id}/edits"
        body = {
            "base_scene_version": 0,
            "edits": [{"operation": "add", "tree": make_tree()}],
        }
        first = client.post(
            url, json=body,
            headers={"Idempotency-Key": "precondition", "If-Match": '"scene-version-0"'},
        )
        assert first.status_code == 202, first.text
        replay = client.post(
            url, json=body,
            headers={"Idempotency-Key": "precondition", "If-Match": '"scene-version-0"'},
        )
        assert replay.status_code == 202  # identical request replays
        conflict = client.post(
            url, json=body,
            headers={"Idempotency-Key": "precondition"},  # If-Match dropped
        )
        assert conflict.status_code == 409
        assert conflict.json()["error"]["code"] == "idempotency_key_reused"

    def test_unknown_result_variable_rejected(self, client: TestClient) -> None:
        scenario_id = SCENARIO0(client)
        response = client.post(
            f"/api/v1/scenarios/{scenario_id}/edits",
            json={
                "base_scene_version": 0,
                "edits": [{"operation": "add", "tree": make_tree()}],
                "requested_result": {"variables": ["wind_speed"]},
            },
        )
        assert response.status_code == 400
        error = response.json()["error"]
        assert error["code"] == "invalid_request"
        assert error["field"] == "requested_result.variables"

    def test_out_of_range_time_index_rejected(self, client: TestClient) -> None:
        scenario_id = SCENARIO0(client)
        response = client.post(
            f"/api/v1/scenarios/{scenario_id}/edits",
            json={
                "base_scene_version": 0,
                "edits": [{"operation": "add", "tree": make_tree()}],
                "requested_result": {"time_indices": [TIME_STEPS + 5]},
            },
        )
        assert response.status_code == 400
        error = response.json()["error"]
        assert error["code"] == "invalid_request"
        assert error["field"] == "requested_result.time_indices"

    def test_valid_requested_subset_accepted(self, client: TestClient) -> None:
        scenario_id = SCENARIO0(client)
        response = client.post(
            f"/api/v1/scenarios/{scenario_id}/edits",
            json={
                "base_scene_version": 0,
                "edits": [{"operation": "add", "tree": make_tree()}],
                "requested_result": {"time_indices": [1], "variables": ["utci"]},
            },
        )
        assert response.status_code == 202
        wait_for_exact(client, scenario_id)
        manifest = client.get(f"/api/v1/scenarios/{scenario_id}/results/1").json()
        assert manifest["time_indices"] == [1]
        assert [v["name"] for v in manifest["variables"]] == ["utci"]

    def test_baseline_shape_mismatch_is_cache_conflict(
        self, tmp_path: Path, solver: FakeSolver
    ) -> None:
        """A site whose stored baseline arrays disagree with its manifest is
        rejected as cache_version_mismatch (not an internal error)."""
        cache = make_site_cache(tmp_path, baseline_time_steps=TIME_STEPS + 1)
        app = make_app(tmp_path, solver, site_cache=cache)
        with TestClient(app) as client:
            response = client.post("/api/v1/scenarios", json={"site_id": SITE_ID})
            assert response.status_code == 409
            error = response.json()["error"]
            assert error["code"] == "cache_version_mismatch"
            assert "shape" in error["message"]
            # Nothing was created: the quota/metrics stay clean.
            assert client.get("/metrics").json()["scenarios"] == 0

    def test_reset_baseline_shape_mismatch_is_cache_conflict(
        self, tmp_path: Path, solver: FakeSolver, site_cache: Path
    ) -> None:
        # Create against a good cache, then swap in a mismatched baseline.
        state_root = tmp_path / "state"
        app = make_app(tmp_path, solver, site_cache=site_cache, state_root=state_root)
        with TestClient(app) as client:
            scenario_id = client.post(
                "/api/v1/scenarios", json={"site_id": SITE_ID}
            ).json()["scenario_id"]
        rng = np.random.default_rng(7)
        np.save(
            site_cache / "baseline_results" / "utci.f32.npy",
            rng.uniform(0, 10, size=(TIME_STEPS + 2, ROWS, COLS)).astype(np.float32),
        )
        app2 = make_app(
            tmp_path, solver, site_cache=site_cache, state_root=state_root
        )
        with TestClient(app2) as client2:
            response = client2.post(f"/api/v1/scenarios/{scenario_id}/reset")
            assert response.status_code == 409
            assert response.json()["error"]["code"] == "cache_version_mismatch"

    def test_scenario_quota_enforced_in_store_transaction(
        self, tmp_path: Path, solver: FakeSolver, site_cache: Path
    ) -> None:
        """The quota is enforced inside the create transaction, so two racing
        creates cannot both pass a read-then-write precheck."""
        from solweig_gpu.server.store import ScenarioQuotaExceeded, Store

        store = Store(
            tmp_path / "quota.sqlite3", results_root=tmp_path / "results"
        )
        try:
            first = store.create_scenario(site_id=SITE_ID, name="a", max_scenarios=1)
            assert first[0].site_id == SITE_ID
            with pytest.raises(ScenarioQuotaExceeded):
                store.create_scenario(site_id=SITE_ID, name="b", max_scenarios=1)
            assert store.count_scenarios() == 1
        finally:
            store.close()


# ---------------------------------------------------------------------------
# Real-adapter integration (default ExactWorker solver; real site cache)
# ---------------------------------------------------------------------------


class TestRealAdapterIntegration:
    """End-to-end runs through ``make_exact_worker_solver`` (no fake solver).

    Uses the same tiny synthetic prepared site as the Phase-5 worker tests:
    a real cache layout and real solver code, with randomized SVF fields so
    no expensive sky-view physics runs. These tests are the only place the
    watermark/replay/bootstrap wiring of the adapter is proven against the
    actual ExactWorker.
    """

    @staticmethod
    def _real_site(root: Path) -> dict:
        from tests.test_incremental_worker import (
            DATE_STR,
            _build_cache,
            _compute_baseline_svf,
            _make_tiny_site,
        )

        grid, site = _make_tiny_site(root / "site")
        # Replace the tiny fixture's random SVF numbers with a real
        # ``svf_calculator`` run: the windowed (LOCAL) solve reads patch
        # cubes (shadowmat et al.) from the cache, and the anisotropic
        # diffuse loop indexes them by the Perez patch count — random
        # 16-band cubes do not match and crash inside Solweig_2022a_calc.
        _compute_baseline_svf(site)
        met = site / "metfiles" / f"metfile_0_0_{DATE_STR}.txt"
        cache = _build_cache(site, root / "cache", met_path=met, site_id="tiny")
        return {
            "sites": {
                "tiny": {
                    "cache_dir": str(root / "cache"),
                    "site_dir": str(site),
                    "selected_date_str": DATE_STR,
                }
            },
            "rows": grid.rows,
            "cols": grid.cols,
        }

    def test_bootstrap_and_first_edit_with_real_worker(
        self, tmp_path: Path
    ) -> None:
        site = self._real_site(tmp_path)
        rows, cols = site["rows"], site["cols"]
        app = create_app(
            state_root=tmp_path / "state",
            sites=site["sites"],
            coalescing_window_ms=20.0,  # no fake solver: the default adapter
            requests_per_minute_per_ip=None,  # wait_for_job polls fast
        )
        with TestClient(app) as client:
            # (b) bootstrap: no stored baseline outputs -> baseline full-solve
            # job publishes an exact result for scene version 0.
            created = client.post("/api/v1/scenarios", json={"site_id": "tiny"})
            assert created.status_code == 201, created.text
            body = created.json()
            assert body["status"] == "refining"
            baseline_job = wait_for_job(client, body["job_id"], timeout=120.0)
            assert baseline_job["status"] == "complete"
            baseline_manifest = client.get(
                f"/api/v1/scenarios/{body['scenario_id']}/results/0"
            ).json()
            assert baseline_manifest["exact"] is True
            assert baseline_manifest["metrics"]["mode"] == "full"
            assert baseline_manifest["window"] == {
                "row_start": 0, "row_stop": rows, "col_start": 0, "col_stop": cols,
            }
            scenario_id = body["scenario_id"]

            # (a) first edit: the real worker must compute a *new* result —
            # with the old (post-batch) watermark bug every edit no-opped and
            # the baseline bytes were re-served as the edited scene.
            edit = client.post(
                f"/api/v1/scenarios/{scenario_id}/edits",
                json={
                    "base_scene_version": 0,
                    "edits": [
                        {
                            "operation": "add",
                            "tree": {
                                "tree_id": "t1",
                                "component_type": "broad_canopy",
                                # mirror TINY_ADD: off-centre so the job stays
                                # a strict sub-window (local mode)
                                "u": 40.5 / cols,
                                "v": 40.5 / rows,
                                "height_m": 4.0,
                                "canopy_diameter_m": 2.0,
                            },
                        }
                    ],
                },
                headers={"Idempotency-Key": "real-edit-1"},
            )
            assert edit.status_code == 202, edit.text
            outcome = wait_for_job(client, edit.json()["job_id"], timeout=120.0)
            assert outcome["status"] == "complete"
            manifest = client.get(
                f"/api/v1/scenarios/{scenario_id}/results/1"
            ).json()
            assert manifest["exact"] is True
            assert manifest["scene_version"] == 1
            edited = patch_codec.decode_payload(
                manifest,
                client.get(
                    f"/api/v1/scenarios/{scenario_id}/results/1/payload"
                ).content,
            )
            baseline_arrays = patch_codec.decode_payload(
                baseline_manifest,
                client.get(
                    f"/api/v1/scenarios/{scenario_id}/results/0/payload"
                ).content,
            )
            # This first edit's dirty window (48x48 px plus the 23 px write
            # margin) exceeds the 0.30 full-recompute fraction of the tiny
            # 128x128 site, so the worker legitimately routes it to a FULL
            # re-solve. What must NOT happen is the watermark bug: re-serving
            # the baseline bytes as an "exact" result for the edited scene.
            assert manifest["metrics"]["mode"] == "full"
            assert edited["utci"].shape == baseline_arrays["utci"].shape
            assert not np.array_equal(edited["utci"], baseline_arrays["utci"])
            # The recomputed scene differs around the new tree itself
            # (u=v=40.5 -> pixels 40..41).
            assert not np.array_equal(
                edited["utci"][:, 40:42, 40:42],
                baseline_arrays["utci"][:, 40:42, 40:42],
            )

            # A second edit far from the first keeps its invalidation window
            # below the full-recompute fraction, exercising the LOCAL path:
            # a strict sub-window patch composed over the prior full result.
            # The tree must stand on open ground: at (12.5, 12.5) it sits on
            # the 12 m building block, and the per-tree pedestal offset
            # (P8: canopy base = surface under the crown) honestly inflates
            # its influence window past the full-recompute fraction.
            edit2 = client.post(
                f"/api/v1/scenarios/{scenario_id}/edits",
                json={
                    "base_scene_version": 1,
                    "edits": [
                        {
                            "operation": "add",
                            "tree": {
                                "tree_id": "t2",
                                "component_type": "broad_canopy",
                                "u": 70.5 / cols,
                                "v": 30.5 / rows,
                                "height_m": 3.0,
                                "canopy_diameter_m": 2.0,
                            },
                        }
                    ],
                },
                headers={"Idempotency-Key": "real-edit-2"},
            )
            assert edit2.status_code == 202, edit2.text
            outcome2 = wait_for_job(client, edit2.json()["job_id"], timeout=120.0)
            assert outcome2["status"] == "complete"
            manifest2 = client.get(
                f"/api/v1/scenarios/{scenario_id}/results/2"
            ).json()
            assert manifest2["exact"] is True
            assert manifest2["scene_version"] == 2
            assert manifest2["metrics"]["mode"] == "local"
            window = manifest2["window"]
            window_area = (window["row_stop"] - window["row_start"]) * (
                window["col_stop"] - window["col_start"]
            )
            assert 0 < window_area < rows * cols  # strict sub-window
            edited2 = patch_codec.decode_payload(
                manifest2,
                client.get(
                    f"/api/v1/scenarios/{scenario_id}/results/2/payload"
                ).content,
            )
            # The patch is window-shaped; the prior result covers the full
            # site. Inside the window the edited scene differs where the new
            # tree changed things.
            r0, r1 = window["row_start"], window["row_stop"]
            c0, c1 = window["col_start"], window["col_stop"]
            assert edited2["utci"].shape == (
                edited["utci"].shape[0],
                r1 - r0,
                c1 - c0,
            )
            inside_prior = edited["utci"][:, r0:r1, c0:c1]
            assert not np.array_equal(edited2["utci"], inside_prior)
            # Client-side composition (prior + patch) must be byte-equal to
            # the prior result OUTSIDE the write window — the published patch
            # never claims changes it did not compute.
            composed = edited["utci"].copy()
            composed[:, r0:r1, c0:c1] = edited2["utci"]
            outside = np.ones((rows, cols), dtype=bool)
            outside[r0:r1, c0:c1] = False
            np.testing.assert_array_equal(
                composed[:, outside], edited["utci"][:, outside]
            )
            scenario = client.get(f"/api/v1/scenarios/{scenario_id}").json()
            assert scenario["exact_result_version"] == 2

    def test_bootstrap_after_superseded_baseline_then_second_edit(
        self, tmp_path: Path
    ) -> None:
        """Regression: a superseded baseline job must not brick the scenario.

        On a site without stored baseline outputs, an edit that lands before
        the baseline job (v0) runs supersedes it, so version 0 is NEVER
        published. The next job bootstraps with one full-tile solve at its
        own target version — a full-site-window exact result — and every
        later edit must compose from that lowest published version instead
        of failing with "no baseline result to compose against".
        """
        site = self._real_site(tmp_path)
        state_root = tmp_path / "state"
        # Frozen app: jobs commit durably but nothing dispatches, so the
        # edit deterministically supersedes the queued baseline job.
        app1 = create_app(
            state_root=state_root,
            sites=site["sites"],
            coalescing_window_ms=20.0,
            start_worker=False,
            requests_per_minute_per_ip=None,  # restart tests poll fast
        )
        with TestClient(app1) as client:
            created = client.post("/api/v1/scenarios", json={"site_id": "tiny"})
            assert created.status_code == 201, created.text
            body = created.json()
            scenario_id = body["scenario_id"]
            baseline_job = body["job_id"]
            edit1 = client.post(
                f"/api/v1/scenarios/{scenario_id}/edits",
                json={
                    "base_scene_version": 0,
                    "edits": [
                        {
                            "operation": "add",
                            "tree": {
                                "tree_id": "t1",
                                "component_type": "broad_canopy",
                                "u": 40.5 / site["cols"],
                                "v": 40.5 / site["rows"],
                                "height_m": 4.0,
                                "canopy_diameter_m": 2.0,
                            },
                        }
                    ],
                },
                headers={"Idempotency-Key": "gap-edit-1"},
            )
            assert edit1.status_code == 202, edit1.text
            edit1_job = edit1.json()["job_id"]

        # Restart with a live worker: the baseline job can no longer publish
        # version 0 (the scene is at 1), the first edit bootstraps at 1.
        app2 = create_app(
            state_root=state_root,
            sites=site["sites"],
            coalescing_window_ms=20.0,
            requests_per_minute_per_ip=None,  # restart tests poll fast
        )
        with TestClient(app2) as client:
            outcome0 = wait_for_job(
                client, baseline_job, timeout=120.0, statuses=("superseded",)
            )
            assert outcome0["status"] == "superseded"
            assert (
                client.get(f"/api/v1/scenarios/{scenario_id}/results/0").status_code
                == 404
            )  # version 0 is never published
            outcome1 = wait_for_job(client, edit1_job, timeout=120.0)
            assert outcome1["status"] == "complete"
            manifest1 = client.get(
                f"/api/v1/scenarios/{scenario_id}/results/1"
            ).json()
            assert manifest1["metrics"]["bootstrap"] is True
            assert manifest1["window"] == {
                "row_start": 0,
                "row_stop": site["rows"],
                "col_start": 0,
                "col_stop": site["cols"],
            }

            # The previously-bricking second edit: must compose from the
            # lowest published version (1) and complete.
            edit2 = client.post(
                f"/api/v1/scenarios/{scenario_id}/edits",
                json={
                    "base_scene_version": 1,
                    "edits": [
                        {
                            "operation": "add",
                            "tree": {
                                "tree_id": "t2",
                                "component_type": "broad_canopy",
                                "u": 12.5 / site["cols"],
                                "v": 12.5 / site["rows"],
                                "height_m": 3.0,
                                "canopy_diameter_m": 2.0,
                            },
                        }
                    ],
                },
                headers={"Idempotency-Key": "gap-edit-2"},
            )
            assert edit2.status_code == 202, edit2.text
            outcome2 = wait_for_job(client, edit2.json()["job_id"], timeout=120.0)
            assert outcome2["status"] == "complete"
            manifest2 = client.get(
                f"/api/v1/scenarios/{scenario_id}/results/2"
            ).json()
            assert manifest2["exact"] is True
            assert manifest2["scene_version"] == 2
            scenario = client.get(f"/api/v1/scenarios/{scenario_id}").json()
            assert scenario["exact_result_version"] == 2
            assert scenario["status"] == "exact"

    def test_real_adapter_reports_contract_progress(
        self, tmp_path: Path
    ) -> None:
        """(c) The default adapter emits the contract's time_loop shape."""
        from solweig_gpu.server.jobs import make_exact_worker_solver

        site = self._real_site(tmp_path)
        events: list[tuple] = []
        default_factory = make_exact_worker_solver()

        def factory(context):
            real_solver = default_factory(context)

            def capturing(request, progress):
                def capture(stage, completed, total, **kwargs):
                    events.append((stage, completed, total, kwargs))
                    progress(stage, completed, total, **kwargs)

                return real_solver(request, capture)

            return capturing

        app = create_app(
            state_root=tmp_path / "state",
            sites=site["sites"],
            solver_factory=factory,
            coalescing_window_ms=20.0,
            requests_per_minute_per_ip=None,  # wait_for_job polls fast
        )
        with TestClient(app) as client:
            created = client.post("/api/v1/scenarios", json={"site_id": "tiny"})
            scenario_id = created.json()["scenario_id"]
            wait_for_job(client, created.json()["job_id"], timeout=120.0)
            edit = client.post(
                f"/api/v1/scenarios/{scenario_id}/edits",
                json={
                    "base_scene_version": 0,
                    "edits": [
                        {
                            "operation": "add",
                            "tree": {
                                "tree_id": "t1",
                                "component_type": "broad_canopy",
                                "u": 40.5 / site["cols"],
                                "v": 40.5 / site["rows"],
                                "height_m": 4.0,
                                "canopy_diameter_m": 2.0,
                            },
                        }
                    ],
                },
            )
            assert edit.status_code == 202, edit.text
            outcome = wait_for_job(client, edit.json()["job_id"], timeout=120.0)
            assert outcome["status"] == "complete"
        stages = [event[0] for event in events]
        assert "windowing" in stages
        assert "time_loop" in stages
        assert "publishing" in stages
        time_loop = [e for e in events if e[0] == "time_loop"][-1]
        completed, total, kwargs = time_loop[1], time_loop[2], time_loop[3]
        assert total > 0 and completed == total
        assert kwargs.get("mode") in ("local", "full")
        window = kwargs.get("window")
        assert window is not None
        assert set(window) == {"row_start", "row_stop", "col_start", "col_stop"}

    @staticmethod
    def _guard_site(root: Path) -> dict:
        """A 256x256 flat site whose dominant structure is 8 m tall.

        Sized so a 9 m edit tree is (a) tall enough to raise the scene-wide
        march amplitude above the cached baseline (the local-solve guard must
        refuse it) while (b) its exact influence window stays far below the
        full-recompute fraction, i.e. routing policy says LOCAL.
        """
        import zipfile

        from tests.test_incremental_worker import (
            DATE_STR,
            SVF_ZIP_MEMBERS,
            _build_cache,
            _compute_baseline_svf,
            _write_tif,
        )

        rows = cols = 256
        pixel, origin, epsg = 2.0, (1000.0, 2000.0), 32616
        site = root / "processed_inputs"
        dem = np.zeros((rows, cols), dtype=np.float32)
        building = dem.copy()
        building[20:50, 15:50] = 8.0  # the dominant structure
        landcover = np.ones((rows, cols), dtype=np.uint8)

        def tif(kind: str, array: np.ndarray) -> None:
            _write_tif(
                site / kind / f"{kind}_0_0.tif", array,
                origin=origin, pixel=pixel, epsg=epsg,
            )

        tif("Building_DSM", building)
        tif("DEM", dem)
        tif("Trees", dem)  # no baseline vegetation
        tif("walls", dem)
        tif("aspect", dem)
        tif("Landcover", landcover)
        from tests.test_incremental_worker import _write_met

        _write_met(site / "metfiles" / f"metfile_0_0_{DATE_STR}.txt", range(10, 13))
        _compute_baseline_svf(site)

        rng = np.random.default_rng(20260901)
        svf_dir = site / "SVF"
        svf_dir.mkdir(parents=True, exist_ok=True)
        scratch = root / "zip_members"
        with zipfile.ZipFile(svf_dir / "svfs_0_0.zip", "w") as archive:
            for member in SVF_ZIP_MEMBERS:
                array = rng.uniform(0.0, 1.0, size=(rows, cols)).astype(np.float32)
                member_path = _write_tif(
                    scratch / f"{member}.tif", array,
                    origin=origin, pixel=pixel, epsg=epsg,
                )
                archive.write(member_path, arcname=f"{member}.tif")
        _write_tif(
            svf_dir / "SkyViewFactor_0_0.tif",
            rng.uniform(0.0, 1.0, size=(rows, cols)).astype(np.float32),
            origin=origin, pixel=pixel, epsg=epsg,
        )
        cubes = {
            name: rng.integers(0, 2, size=(rows, cols, 16)).astype(np.float32)
            for name in ("shadowmat", "vegshadowmat", "vbshmat")
        }
        np.savez(svf_dir / "shadowmats_0_0.npz", **cubes)

        cache = _build_cache(
            site, root / "cache",
            met_path=site / "metfiles" / f"metfile_0_0_{DATE_STR}.txt",
            site_id="guard",
        )
        return {
            "sites": {
                "guard": {
                    "cache_dir": str(root / "cache"),
                    "site_dir": str(site),
                    "selected_date_str": DATE_STR,
                }
            },
            "rows": rows,
            "cols": cols,
        }

    def test_local_policy_amaxvalue_guard_full_fallback_end_to_end(
        self, tmp_path: Path
    ) -> None:
        """SCI-004: routing says LOCAL, the guard refuses the local solve, the
        full fallback completes the job — all through the HTTP API."""
        from tests.test_incremental_worker import DATE_STR

        from solweig_gpu.incremental.cache import SiteCache
        from solweig_gpu.incremental.geometry import (
            RasterGrid,
            RecomputeMode,
            TreeSpec,
            choose_recompute_mode,
        )
        from solweig_gpu.incremental.trees import TreeLayer
        from solweig_gpu.incremental.worker import ExactWorker

        site = self._guard_site(tmp_path)
        rows, cols = site["rows"], site["cols"]
        tree_row, tree_col = 128.5, 128.5  # open ground, far from the 8 m block

        # (1) Routing policy check (windowing only — no solve): the edit's
        # exact influence window routes LOCAL.
        cache = SiteCache.load(site["sites"]["guard"]["cache_dir"])
        grid = RasterGrid(
            cache.rows, cache.cols, cache.pixel_size_m,
            cache.manifest.origin_x_m, cache.manifest.origin_y_m,
        )
        layer = TreeLayer(cache.tree_base, grid)
        layer.add_tree(
            TreeSpec(
                "tall",
                grid.origin_x_m + tree_col * grid.pixel_size_m,
                grid.origin_y_m - tree_row * grid.pixel_size_m,
                9.0, 2.0,
            )
        )
        worker = ExactWorker(
            cache, layer,
            site_dir=Path(site["sites"]["guard"]["site_dir"]),
            results_root=tmp_path / "worker-results",
            selected_date_str=DATE_STR,
        )
        dirty = worker.dirty_windows()
        union = dirty[0].expand(worker.write_margin_pixels).clamp(
            rows=grid.rows, cols=grid.cols
        )
        fraction = union.area / grid.area_pixels
        mode = choose_recompute_mode(
            union, grid,
            full_recompute_fraction=(
                worker.influence_config.full_recompute_fraction
            ),
        )
        assert fraction < 0.30, fraction
        assert mode is RecomputeMode.LOCAL, (mode, fraction)

        # (2) End-to-end: the guard refuses the local solve (the tree raises
        # the scene-wide march amplitude above the cached baseline) and the
        # worker's full fallback publishes an exact full-tile result.
        app = create_app(
            state_root=tmp_path / "state",
            sites=site["sites"],
            coalescing_window_ms=20.0,  # the default exact adapter
            requests_per_minute_per_ip=None,  # wait_for_job polls fast
        )
        with TestClient(app) as client:
            created = client.post("/api/v1/scenarios", json={"site_id": "guard"})
            assert created.status_code == 201, created.text
            scenario_id = created.json()["scenario_id"]
            wait_for_job(client, created.json()["job_id"], timeout=300.0)

            edit = client.post(
                f"/api/v1/scenarios/{scenario_id}/edits",
                json={
                    "base_scene_version": 0,
                    "edits": [
                        {
                            "operation": "add",
                            "tree": {
                                "tree_id": "tall",
                                "component_type": "broad_canopy",
                                "u": tree_col / cols,
                                "v": tree_row / rows,
                                "height_m": 9.0,
                                "canopy_diameter_m": 4.0,
                            },
                        }
                    ],
                },
                headers={"Idempotency-Key": "sci004-tall-edit"},
            )
            assert edit.status_code == 202, edit.text
            outcome = wait_for_job(client, edit.json()["job_id"], timeout=300.0)
            assert outcome["status"] == "complete", outcome

            manifest = client.get(
                f"/api/v1/scenarios/{scenario_id}/results/1"
            ).json()
            assert manifest["exact"] is True
            assert manifest["metrics"]["mode"] == "full"
            assert "amaxvalue" in manifest["metrics"]["fallback_reason"]
            assert manifest["window"] == {
                "row_start": 0, "row_stop": rows,
                "col_start": 0, "col_stop": cols,
            }
            scenario = client.get(f"/api/v1/scenarios/{scenario_id}").json()
            assert scenario["exact_result_version"] == 1
            assert scenario["status"] == "exact"


# ---------------------------------------------------------------------------
# P8 server hardening: composition cache, identity payload cache, retention
# ---------------------------------------------------------------------------


class TestStateCompositionCache:
    """Unit tests for the checksum-chain-validated LRU of composed states."""

    @staticmethod
    def _arrays(value: float) -> dict[str, np.ndarray]:
        return {"utci": np.full((1, 4, 4), value, dtype=np.float32)}

    def test_get_returns_private_copies(self) -> None:
        from solweig_gpu.server.jobs import _StateCompositionCache

        cache = _StateCompositionCache()
        cache.put("s", 2, ("utci",), ("c0", "c1"), self._arrays(7.0))
        handed = cache.get("s", 2, ("utci",), ("c0", "c1"))
        assert handed is not None
        handed["utci"][0, 0, 0] = -1.0  # caller-side mutation ...
        again = cache.get("s", 2, ("utci",), ("c0", "c1"))
        assert again is not None
        assert again["utci"][0, 0, 0] == 7.0  # ... must not leak into the cache

    def test_chain_mismatch_is_a_miss(self) -> None:
        from solweig_gpu.server.jobs import _StateCompositionCache

        cache = _StateCompositionCache()
        cache.put("s", 2, ("utci",), ("c0", "c1"), self._arrays(7.0))
        # Same version, different history (reset / re-publish) must not serve
        # the stale composition.
        assert cache.get("s", 2, ("utci",), ("c0", "cX")) is None
        # A longer chain sharing the prefix is a different key-length lookup:
        # only exact prefix matches count.
        assert cache.get("s", 2, ("utci",), ("c0",)) is None

    def test_lru_eviction_by_bytes(self) -> None:
        from solweig_gpu.server.jobs import _StateCompositionCache

        cache = _StateCompositionCache(max_total_bytes=self._arrays(0.0)["utci"].nbytes)
        cache.put("s1", 1, ("utci",), ("a",), self._arrays(1.0))
        cache.put("s2", 1, ("utci",), ("b",), self._arrays(2.0))  # evicts s1
        assert cache.get("s1", 1, ("utci",), ("a",)) is None
        assert cache.get("s2", 1, ("utci",), ("b",)) is not None

    def test_oversized_put_is_skipped(self) -> None:
        from solweig_gpu.server.jobs import _StateCompositionCache

        cache = _StateCompositionCache(max_total_bytes=8)
        cache.put("s", 1, ("utci",), ("a",), self._arrays(1.0))
        assert cache.get("s", 1, ("utci",), ("a",)) is None


class TestP8ServerFixes:
    def test_compose_reuses_cached_prefix(self, client: TestClient, monkeypatch) -> None:
        """A warm compose decodes zero durable payloads: the state resumes from
        the checksum-validated cache of the previous composition, so per-job
        cost is O(newly published versions) instead of O(all versions)."""
        from solweig_gpu.server.jobs import RunnerContext, _compose_current_state

        scenario_id = SCENARIO0(client)
        for version in range(3):
            response = add_tree(client, make_tree(f"tree_{version:02d}"), base=version)
            assert response.status_code == 202, response.text
            wait_for_exact(client, scenario_id)
        # versions 0..3 are durably published (baseline + three edits).

        context: RunnerContext = client.app.state.context
        grid_info = {
            "rows": ROWS, "cols": COLS, "time_steps": TIME_STEPS,
        }
        window = RasterWindow(0, ROWS, 0, COLS)
        variables = ("utci", "tmrt")

        calls = {"n": 0}
        real = patch_codec.decode_payload

        def counting(manifest, payload):
            calls["n"] += 1
            return real(manifest, payload)

        monkeypatch.setattr(patch_codec, "decode_payload", counting)
        cold = _compose_current_state(
            context, scenario_id, grid_info, variables, window, 4
        )
        assert calls["n"] == 4  # cold: one decode per published version
        warm = _compose_current_state(
            context, scenario_id, grid_info, variables, window, 4
        )
        assert calls["n"] == 4  # warm: the whole prefix comes from the cache
        for name in variables:
            np.testing.assert_array_equal(cold[name], warm[name])

    def test_identity_payload_is_decompressed_once(
        self, client: TestClient, monkeypatch
    ) -> None:
        scenario_id = SCENARIO0(client)
        wait_for_exact(client, scenario_id)
        calls = {"n": 0}
        real = patch_codec.decompress_payload

        def counting(payload, manifest):
            calls["n"] += 1
            return real(payload, manifest)

        monkeypatch.setattr(patch_codec, "decompress_payload", counting)
        url = f"/api/v1/scenarios/{scenario_id}/results/0/payload"
        first = client.get(url, headers={"Accept": self.IDENTITY})
        second = client.get(url, headers={"Accept": self.IDENTITY})
        assert first.status_code == second.status_code == 200
        assert calls["n"] == 1  # second request served from the byte cache
        assert first.content == second.content
        assert first.headers["ETag"] == second.headers["ETag"]
        assert first.headers["X-SOLWEIG-Checksum"] == second.headers["X-SOLWEIG-Checksum"]

    IDENTITY = TestPayloadNegotiation.IDENTITY

    def test_sweep_retention_trims_ledgers(self, tmp_path: Path) -> None:
        from solweig_gpu.server.store import StoredResponse, Store

        store = Store(tmp_path / "state.sqlite3", results_root=tmp_path / "results")
        try:
            scenario, _ = store.create_scenario(site_id=SITE_ID, name="s")
            scenario_id = scenario.scenario_id
            with store._write() as conn:
                for seq in range(1, 7):
                    conn.execute(
                        "INSERT INTO edit_events (scenario_id, sequence, event_id, "
                        "base_scene_version, operation, submitted_at) "
                        "VALUES (?, ?, ?, 0, 'add', '2026-09-01T00:00:00+00:00')",
                        (scenario_id, seq, f"evt-{seq}"),
                    )
                # Published results consumed events up to sequence 4.
                conn.execute(
                    "UPDATE scenarios SET acked_sequence = 4 WHERE scenario_id = ?",
                    (scenario_id,),
                )
                conn.execute(
                    "INSERT INTO idempotency_keys (scope, key, request_fingerprint, "
                    "response_json, created_at) VALUES "
                    "('edit', 'stale', 'fp1', ?, '2026-08-01T00:00:00+00:00'), "
                    "('edit', 'fresh', 'fp2', ?, '2026-09-01T12:00:00+00:00')",
                    (
                        StoredResponse(200, {"ok": 1}).json(),
                        StoredResponse(200, {"ok": 2}).json(),
                    ),
                )
            deleted = store.sweep_retention(
                idempotency_ttl_hours=24.0,
                event_tail=1,
                now="2026-09-01T20:00:00+00:00",
            )
            # acked=4, tail=1 => keep sequences > 3; 1..3 are consumed and old.
            assert deleted == {"idempotency_keys": 1, "edit_events": 3}
            assert [event["sequence"] for event in store.list_events(scenario_id)] == [
                4,
                5,
                6,
            ]
            # Above-watermark events are never deleted (coalescing needs them).
            assert store.lookup_idempotency("edit", "stale") is None
            fresh = store.lookup_idempotency("edit", "fresh")
            assert fresh is not None and fresh.body == {"ok": 2}
            # Idempotent: a second sweep deletes nothing.
            assert store.sweep_retention(
                idempotency_ttl_hours=24.0,
                event_tail=1,
                now="2026-09-01T20:00:00+00:00",
            ) == {"idempotency_keys": 0, "edit_events": 0}
        finally:
            store.close()

    def test_runner_sweep_is_throttled_and_failure_safe(self, client: TestClient) -> None:
        runner = client.app.state.runner
        store = runner.context.store
        calls = {"n": 0}

        def counting(**kwargs):
            calls["n"] += 1
            return {"idempotency_keys": 0, "edit_events": 0}

        runner._last_sweep = None
        original = store.sweep_retention
        store.sweep_retention = counting
        try:
            runner._maybe_sweep()
            assert calls["n"] == 1
            runner._maybe_sweep()  # within the hour: throttled
            assert calls["n"] == 1

            def exploding(**kwargs):
                raise RuntimeError("sweep failed")

            store.sweep_retention = exploding
            runner._last_sweep = None
            runner._maybe_sweep()  # must not raise; jobs keep flowing
            assert runner._last_sweep is not None
        finally:
            store.sweep_retention = original
            runner._last_sweep = None


# ---------------------------------------------------------------------------
# Shared world: the one workspace every visitor joins by default
# ---------------------------------------------------------------------------


class TestSharedWorld:
    """Boot-time find-or-create of the shared scenario plus discovery.

    ``create_app(shared_scenario_id=...)`` (always set by ``python -m
    solweig_gpu.server``) creates the scenario through the same path as
    POST /api/v1/scenarios — baseline materialization included — and
    ``/api/v1/capabilities`` advertises it as ``default_workspace_id`` so
    the studio joins it instead of minting a per-session scenario.
    """

    def test_boot_creates_the_shared_world_with_a_readable_baseline(
        self, tmp_path: Path, solver: FakeSolver, site_cache: Path
    ) -> None:
        app = make_app(
            tmp_path,
            solver,
            site_cache=site_cache,
            shared_scenario_id="scn_shared_world",
        )
        with TestClient(app) as client:
            response = client.get("/api/v1/scenarios/scn_shared_world")
            assert response.status_code == 200, response.text
            body = response.json()
            assert body["name"] == "Shared world"
            assert body["site_id"] == SITE_ID
            assert body["scene_version"] == 0
            # Baseline materialization is inline at boot: results/0 is
            # readable immediately, before any client edit.
            manifest = client.get("/api/v1/scenarios/scn_shared_world/results/0")
            assert manifest.status_code == 200, manifest.text
            assert manifest.json()["scene_version"] == 0

    def test_boot_ensure_is_idempotent_across_restarts(
        self, tmp_path: Path, solver: FakeSolver, site_cache: Path
    ) -> None:
        state_root = tmp_path / "state"
        for _ in range(2):
            app = make_app(
                tmp_path,
                solver,
                site_cache=site_cache,
                state_root=state_root,
                shared_scenario_id="scn_shared_world",
            )
            with TestClient(app) as client:
                # Second boot finds the scenario instead of re-creating it:
                # still exactly one scenario in the store.
                assert client.get("/metrics").json()["scenarios"] == 1
                shared = client.get("/api/v1/scenarios/scn_shared_world")
                assert shared.status_code == 200
                assert shared.json()["name"] == "Shared world"

    def test_capabilities_advertise_the_default_workspace(
        self, tmp_path: Path, solver: FakeSolver, site_cache: Path
    ) -> None:
        app = make_app(
            tmp_path,
            solver,
            site_cache=site_cache,
            shared_scenario_id="scn_shared_world",
        )
        with TestClient(app) as client:
            document = client.get("/api/v1/capabilities").json()
            assert document["default_workspace_id"] == "scn_shared_world"
            # The engine-side document is otherwise served verbatim.
            assert document["adapters"]

    def test_capabilities_omit_the_field_without_a_shared_world(
        self, client: TestClient
    ) -> None:
        document = client.get("/api/v1/capabilities").json()
        assert "default_workspace_id" not in document

    def test_per_scenario_minting_still_works_alongside_the_shared_world(
        self, tmp_path: Path, solver: FakeSolver, site_cache: Path
    ) -> None:
        app = make_app(
            tmp_path,
            solver,
            site_cache=site_cache,
            shared_scenario_id="scn_shared_world",
        )
        with TestClient(app) as client:
            minted = client.post("/api/v1/scenarios", json={"site_id": SITE_ID})
            assert minted.status_code == 201, minted.text
            assert minted.json()["scenario_id"] != "scn_shared_world"
            # The private sandbox is editable through the same contract.
            edited = add_tree(client, make_tree(), 0, scenario_id=minted.json()["scenario_id"])
            assert edited.status_code == 202, edited.text


    def test_boot_ensures_the_shared_world_even_when_the_scenario_quota_is_full(
        self, tmp_path: Path, solver: FakeSolver, site_cache: Path
    ) -> None:
        state_root = tmp_path / "state"
        app = make_app(tmp_path, solver, site_cache=site_cache, state_root=state_root)
        with TestClient(app) as client:
            minted = client.post("/api/v1/scenarios", json={"site_id": SITE_ID})
            assert minted.status_code == 201, minted.text

        # Quota is full (1/1) and no scenario-delete API exists: boot with a
        # shared world must not brick. The shared world is deployment
        # infrastructure, not a user session, so its creation is exempt from
        # the quota.
        app2 = make_app(
            tmp_path,
            solver,
            site_cache=site_cache,
            state_root=state_root,
            shared_scenario_id="scn_shared_world",
            max_scenarios=1,
        )
        with TestClient(app2) as client:
            assert client.get("/api/v1/scenarios/scn_shared_world").status_code == 200
            assert client.get("/metrics").json()["scenarios"] == 2

    def test_a_found_shared_world_pinned_to_an_unserved_site_is_not_advertised(
        self, tmp_path: Path, solver: FakeSolver
    ) -> None:
        state_root = tmp_path / "state"
        cache_a = make_site_cache(tmp_path / "a", site_id="site-a")
        cache_b = make_site_cache(tmp_path / "b", site_id="site-b")
        common = dict(
            state_root=state_root,
            solver_factory=lambda context: solver,
            coalescing_window_ms=20.0,
            requests_per_minute_per_ip=None,
            edits_per_minute=None,
        )
        app = create_app(sites={"site-a": {"cache_dir": cache_a}}, shared_scenario_id="scn_shared_world", **common)
        with TestClient(app):
            pass  # creates the shared world pinned to site-a

        # Restart serving only site-b: the found scenario's pin no longer
        # matches, so the boot must NOT advertise a default workspace (the
        # studio falls back to minting) instead of pointing every visitor at
        # a scenario this deployment cannot serve.
        app2 = create_app(sites={"site-b": {"cache_dir": cache_b}}, shared_scenario_id="scn_shared_world", **common)
        with TestClient(app2) as client:
            document = client.get("/api/v1/capabilities").json()
            assert "default_workspace_id" not in document
            assert client.get("/metrics").json()["scenarios"] == 1

    def test_boot_requeues_a_failed_shared_world_baseline_job(
        self, tmp_path: Path
    ) -> None:
        state_root = tmp_path / "state"
        # Isolated cache root (no `site_cache` fixture): the fixture writes
        # baseline outputs into tmp_path/cache, and a leftover metadata.json
        # would turn the boot into inline materialization instead of a
        # scheduled baseline job.
        no_baseline = make_site_cache(tmp_path / "nb", baseline=False)
        failing = FakeSolver()
        failing.fail = True
        app = make_app(
            tmp_path,
            failing,
            site_cache=no_baseline,
            state_root=state_root,
            shared_scenario_id="scn_shared_world",
        )
        with TestClient(app):
            # The boot-scheduled baseline job runs and fails.
            deadline = time.monotonic() + 5.0
            status = None
            while time.monotonic() < deadline:
                with sqlite3.connect(state_root / "store.sqlite3") as conn:
                    row = conn.execute(
                        "SELECT status FROM jobs"
                    ).fetchone()
                if row is not None and row[0] in ("failed", "cancelled", "complete"):
                    status = row[0]
                    break
                time.sleep(0.05)
            assert status == "failed"

        # A failed boot baseline would brick the shared world for every
        # visitor forever; the next boot re-queues it (durable-job path).
        healthy = FakeSolver()
        app2 = make_app(
            tmp_path,
            healthy,
            site_cache=no_baseline,
            state_root=state_root,
            shared_scenario_id="scn_shared_world",
        )
        with TestClient(app2) as client:
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                manifest = client.get("/api/v1/scenarios/scn_shared_world/results/0")
                if manifest.status_code == 200:
                    break
                time.sleep(0.05)
            assert manifest.status_code == 200, manifest.text
            assert manifest.json()["scene_version"] == 0

    def test_create_app_rejects_a_malformed_shared_scenario_id(
        self, tmp_path: Path, solver: FakeSolver, site_cache: Path
    ) -> None:
        # The id becomes a path segment under the state root, so the factory
        # validates the same convention the env entrypoint enforces.
        with pytest.raises(ValueError, match="shared_scenario_id"):
            make_app(
                tmp_path,
                solver,
                site_cache=site_cache,
                shared_scenario_id="shared_world",  # missing the scn_ prefix
            )

    def test_completed_job_body_exposes_the_recompute_mode(
        self, client: TestClient
    ) -> None:
        scenario_id = SCENARIO0(client)
        response = add_tree(client, make_tree(), base=0)
        assert response.status_code == 202, response.text
        complete = wait_for_job(client, response.json()["job_id"])
        assert complete["status"] == "complete"
        # The exact plane reads the mode off the JOB BODY (full-recompute
        # honesty): terminal bodies must carry the routed mode, not just the
        # metrics blob.
        assert complete["mode"] == "local"


# ---------------------------------------------------------------------------
# Optional-dependency guard
# ---------------------------------------------------------------------------


class TestOptionalServerDependency:
    def test_core_import_and_guard_without_fastapi(self) -> None:
        """`import solweig_gpu` works without FastAPI; the server subpackage
        raises a helpful ImportError instead."""
        code = "\n".join(
            [
                "import sys",
                "sys.modules['fastapi'] = None",
                "import solweig_gpu",
                "print('core-ok')",
                "try:",
                "    import solweig_gpu.server",
                "except ImportError as error:",
                "    assert 'solweig-gpu[server]' in str(error), error",
                "    print('guard-ok')",
                "else:",
                "    raise SystemExit('server import should have failed')",
            ]
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, "PYTHONPATH": os.pathsep.join(
                [str(Path(__file__).resolve().parents[1]), os.environ.get("PYTHONPATH", "")]
            )},
        )
        assert result.returncode == 0, result.stderr
        assert "core-ok" in result.stdout
        assert "guard-ok" in result.stdout
