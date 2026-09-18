# SPDX-License-Identifier: GPL-3.0-only
"""P9 security controls (REL-003) and ops health endpoints.

Covers the deployment hardening that ships default-on in ``create_app``:

* request body size cap → 413 ``payload_too_large`` (both the
  ``Content-Length`` precheck and the actual-size backstop for chunked
  uploads);
* per-client-IP sliding-window rate limit → 429 ``rate_limited`` with
  ``Retry-After`` (ops endpoints exempt; ``None`` disables; budgets are
  per-IP);
* per-scenario mutation budget (edits/resets) → 429 ``rate_limited``;
* tree-count cap (``max_trees_per_scenario``) and tree property bounds;
* ``/health/live`` / ``/health/ready`` / ``/health/worker`` shapes,
  including the not-ready 503 when the worker is not running or the store
  schema is newer than this code, and the worker diagnostics mid-job.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import httpx
from fastapi.testclient import TestClient

from solweig_gpu.server.app import (
    DEFAULT_EDITS_PER_MINUTE_PER_SCENARIO,
    DEFAULT_MAX_REQUEST_BODY_BYTES,
    DEFAULT_REQUESTS_PER_MINUTE_PER_IP,
    create_app,
)
from tests.test_server_api import SITE_ID, FakeSolver, make_site_cache

CLIENT_IP = "203.0.113.7"


def make_app(tmp_path: Path, solver: FakeSolver, **kwargs: Any):
    """App with the security defaults UNLESS a test overrides them."""
    return create_app(
        state_root=tmp_path / "state",
        sites={SITE_ID: {"cache_dir": make_site_cache(tmp_path)}},
        solver_factory=lambda context: solver,
        coalescing_window_ms=20.0,
        **kwargs,
    )


def _create_scenario(client: TestClient) -> str:
    response = client.post("/api/v1/scenarios", json={"site_id": SITE_ID})
    assert response.status_code == 201, response.text
    return response.json()["scenario_id"]


def _edit_body(tree_id: str = "t1") -> dict[str, Any]:
    return {
        "base_scene_version": 0,
        "edits": [
            {
                "operation": "add",
                "tree": {
                    "tree_id": tree_id,
                    "component_type": "broad_canopy",
                    "u": 0.5,
                    "v": 0.5,
                    "height_m": 8.0,
                    "canopy_diameter_m": 6.0,
                },
            }
        ],
    }


# ---------------------------------------------------------------------------
# Request body limit (413)
# ---------------------------------------------------------------------------


class TestRequestBodyLimit:
    def test_oversized_content_length_rejected_before_read(self, tmp_path: Path) -> None:
        solver = FakeSolver()
        app = make_app(tmp_path, solver, max_request_body_bytes=512)
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/scenarios",
                json={"site_id": SITE_ID, "name": "x" * 2048},
            )
            # rejected before any handler ran: no solver call happened
            assert solver.calls == []
        assert response.status_code == 413
        error = response.json()["error"]
        assert error["code"] == "payload_too_large"
        assert "512" in error["message"]
        assert response.headers["X-Request-ID"]

    def test_oversized_chunked_body_rejected_on_actual_size(self, tmp_path: Path) -> None:
        """No ``Content-Length`` (streamed upload): the actual-size check trips."""
        import asyncio

        solver = FakeSolver()
        app = make_app(tmp_path, solver, max_request_body_bytes=256)
        transport = httpx.ASGITransport(app=app, client=(CLIENT_IP, 44444))
        parts = [b'{"site_id": "', SITE_ID.encode(), b'", "name": "', b"x" * 4096, b'"}']

        async def chunks():
            for part in parts:
                yield part

        async def run() -> httpx.Response:
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as async_client:
                return await async_client.post("/api/v1/scenarios", content=chunks())

        response = asyncio.run(run())
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "payload_too_large"
        assert solver.calls == []

    def test_default_limit_is_one_mebibyte(self, tmp_path: Path) -> None:
        solver = FakeSolver()
        with TestClient(make_app(tmp_path, solver)) as client:
            # A normal create is far below the default cap.
            response = client.post("/api/v1/scenarios", json={"site_id": SITE_ID})
            assert response.status_code == 201
        assert DEFAULT_MAX_REQUEST_BODY_BYTES == 1 << 20

    def test_body_limit_disabled_when_none(self, tmp_path: Path) -> None:
        """With the cap off, an oversized body reaches the normal validation
        (``name`` is capped at 256 chars → 400), not the 413 gate."""
        solver = FakeSolver()
        app = make_app(tmp_path, solver, max_request_body_bytes=None)
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/scenarios",
                json={"site_id": SITE_ID, "name": "x" * 8192},
            )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_request"

    def test_body_limit_does_not_affect_responses(self, tmp_path: Path) -> None:
        """Payload downloads are responses; a small request cap must not
        throttle them even when the response exceeds the cap."""
        solver = FakeSolver()
        app = make_app(tmp_path, solver, max_request_body_bytes=512)
        with TestClient(app) as client:
            scenario_id = _create_scenario(client)
            edit = client.post(
                f"/api/v1/scenarios/{scenario_id}/edits", json=_edit_body()
            )
            assert edit.status_code == 202, edit.text
            deadline = time.monotonic() + 15.0
            while time.monotonic() < deadline:
                detail = client.get(f"/api/v1/jobs/{edit.json()['job_id']}").json()
                if detail["status"] not in ("queued", "running"):
                    break
                time.sleep(0.02)
            assert detail["status"] == "complete"
            payload = client.get(
                f"/api/v1/scenarios/{scenario_id}/results/1/payload",
                headers={"Accept": "application/vnd.solweig.patch+zstd"},
            )
            assert payload.status_code == 200
            assert len(payload.content) > 0


# ---------------------------------------------------------------------------
# Per-IP rate limit (429 + Retry-After)
# ---------------------------------------------------------------------------


class TestIpRateLimit:
    """Clients pinned to a source IP via TestClient's ``client=`` scope
    override; only ops-free endpoints are hit, so no lifespan is entered.
    """

    @staticmethod
    def _client(app, ip: str = CLIENT_IP) -> TestClient:
        return TestClient(app, client=(ip, 44444))

    def test_breach_returns_429(self, tmp_path: Path) -> None:
        solver = FakeSolver()
        app = make_app(
            tmp_path,
            solver,
            requests_per_minute_per_ip=5,
            edits_per_minute=None,
        )
        client = self._client(app)
        statuses = [client.get("/healthz").status_code for _ in range(7)]
        assert statuses[:5] == [200] * 5
        assert statuses[5:] == [429, 429]

    def test_429_envelope_and_retry_after_header(self, tmp_path: Path) -> None:
        solver = FakeSolver()
        app = make_app(
            tmp_path,
            solver,
            requests_per_minute_per_ip=2,
            edits_per_minute=None,
        )
        client = self._client(app)
        client.get("/healthz")
        client.get("/healthz")
        response = client.get("/healthz")
        assert response.status_code == 429
        error = response.json()["error"]
        assert error["code"] == "rate_limited"
        assert response.headers["Retry-After"].isdigit()
        assert int(response.headers["Retry-After"]) >= 1
        assert response.headers["X-Request-ID"]

    def test_ops_endpoints_exempt(self, tmp_path: Path) -> None:
        solver = FakeSolver()
        app = make_app(
            tmp_path,
            solver,
            requests_per_minute_per_ip=5,
            edits_per_minute=None,
        )
        with TestClient(app) as client:
            # Ops surface polls freely even with a tiny general budget.
            for _ in range(12):
                assert client.get("/health/live").status_code == 200
                assert client.get("/health/ready").status_code == 200
                assert client.get("/health/worker").status_code == 200
                assert client.get("/metrics").status_code == 200
            # ... while one non-ops call already burns a slot: with a budget
            # of 5, five /healthz calls land and the sixth 429s.
            assert [client.get("/healthz").status_code for _ in range(5)] == [200] * 5
            assert client.get("/healthz").status_code == 429

    def test_disabled_when_none(self, tmp_path: Path) -> None:
        solver = FakeSolver()
        with TestClient(
            make_app(
                tmp_path, solver, requests_per_minute_per_ip=None, edits_per_minute=None
            )
        ) as client:
            for _ in range(60):
                assert client.get("/healthz").status_code == 200

    def test_budgets_are_per_ip(self, tmp_path: Path) -> None:
        solver = FakeSolver()
        app = make_app(
            tmp_path,
            solver,
            requests_per_minute_per_ip=2,
            edits_per_minute=None,
        )
        client_a = self._client(app, "198.51.100.1")
        client_b = self._client(app, "198.51.100.2")
        assert client_a.get("/healthz").status_code == 200
        assert client_a.get("/healthz").status_code == 200
        assert client_a.get("/healthz").status_code == 429
        # A different client IP has its own window.
        assert client_b.get("/healthz").status_code == 200

    def test_default_is_30_per_minute(self) -> None:
        assert DEFAULT_REQUESTS_PER_MINUTE_PER_IP == 30


# ---------------------------------------------------------------------------
# Per-scenario mutation budget (429 rate_limited from the route limiter)
# ---------------------------------------------------------------------------


class TestPerScenarioEditBudget:
    def test_default_budget_enforced(self, tmp_path: Path) -> None:
        """Defaults on: the 21st per-scenario mutation inside a minute 429s."""
        solver = FakeSolver()
        with TestClient(
            make_app(tmp_path, solver, requests_per_minute_per_ip=None)
        ) as client:
            scenario_id = _create_scenario(client)
            statuses = []
            version = 0
            for index in range(DEFAULT_EDITS_PER_MINUTE_PER_SCENARIO + 1):
                tree_id = f"t{index}"
                response = client.post(
                    f"/api/v1/scenarios/{scenario_id}/edits",
                    json={
                        "base_scene_version": version,
                        "edits": [
                            {
                                "operation": "add",
                                "tree": {
                                    "tree_id": tree_id,
                                    "component_type": "broad_canopy",
                                    "u": 0.3 + index * 0.01,
                                    "v": 0.4,
                                    "height_m": 6.0,
                                    "canopy_diameter_m": 4.0,
                                },
                            }
                        ],
                    },
                )
                statuses.append(response.status_code)
                if response.status_code == 202:
                    version = response.json()["scene_version"]
            assert statuses[:-1] == [202] * DEFAULT_EDITS_PER_MINUTE_PER_SCENARIO
            assert statuses[-1] == 429
            assert client.get(f"/api/v1/scenarios/{scenario_id}").status_code == 200

    def test_budget_configurable(self, tmp_path: Path) -> None:
        solver = FakeSolver()
        with TestClient(
            make_app(
                tmp_path,
                solver,
                edits_per_minute=2,
                requests_per_minute_per_ip=None,
            )
        ) as client:
            scenario_id = _create_scenario(client)
            first = client.post(
                f"/api/v1/scenarios/{scenario_id}/edits", json=_edit_body("a1")
            )
            second = client.post(
                f"/api/v1/scenarios/{scenario_id}/edits",
                json={
                    "base_scene_version": first.json()["scene_version"],
                    "edits": [
                        {
                            "operation": "add",
                            "tree": {
                                "tree_id": "a2",
                                "component_type": "broad_canopy",
                                "u": 0.4,
                                "v": 0.4,
                                "height_m": 6.0,
                                "canopy_diameter_m": 4.0,
                            },
                        }
                    ],
                },
            )
            third = client.post(
                f"/api/v1/scenarios/{scenario_id}/edits",
                json={
                    "base_scene_version": second.json()["scene_version"],
                    "edits": [
                        {
                            "operation": "add",
                            "tree": {
                                "tree_id": "a3",
                                "component_type": "broad_canopy",
                                "u": 0.6,
                                "v": 0.4,
                                "height_m": 6.0,
                                "canopy_diameter_m": 4.0,
                            },
                        }
                    ],
                },
            )
            assert first.status_code == 202
            assert second.status_code == 202
            assert third.status_code == 429
            assert third.json()["error"]["code"] == "rate_limited"


# ---------------------------------------------------------------------------
# Tree-count cap and property bounds
# ---------------------------------------------------------------------------


class TestTreeBounds:
    def test_max_trees_per_scenario_enforced(self, tmp_path: Path) -> None:
        solver = FakeSolver()
        with TestClient(
            make_app(
                tmp_path,
                solver,
                max_trees_per_scenario=3,
                requests_per_minute_per_ip=None,
                edits_per_minute=None,
            )
        ) as client:
            scenario_id = _create_scenario(client)
            version = 0
            for index in range(3):
                response = client.post(
                    f"/api/v1/scenarios/{scenario_id}/edits",
                    json={
                        "base_scene_version": version,
                        "edits": [
                            {
                                "operation": "add",
                                "tree": {
                                    "tree_id": f"tree-{index}",
                                    "component_type": "broad_canopy",
                                    "u": 0.2 + index * 0.05,
                                    "v": 0.3,
                                    "height_m": 7.0,
                                    "canopy_diameter_m": 5.0,
                                },
                            }
                        ],
                    },
                )
                assert response.status_code == 202, response.text
                version = response.json()["scene_version"]
            over = client.post(
                f"/api/v1/scenarios/{scenario_id}/edits",
                json={
                    "base_scene_version": version,
                    "edits": [
                        {
                            "operation": "add",
                            "tree": {
                                "tree_id": "tree-3",
                                "component_type": "broad_canopy",
                                "u": 0.9,
                                "v": 0.3,
                                "height_m": 7.0,
                                "canopy_diameter_m": 5.0,
                            },
                        }
                    ],
                },
            )
            assert over.status_code == 403
            error = over.json()["error"]
            assert error["code"] == "site_limit_exceeded"
            assert "cap is 3" in error["message"]

    def test_tree_property_bounds_rejected(self, tmp_path: Path) -> None:
        """Height/canopy/uv/trunk/transmissivity bounds → 422 with field path."""
        solver = FakeSolver()
        with TestClient(
            make_app(
                tmp_path,
                solver,
                requests_per_minute_per_ip=None,
                edits_per_minute=None,
            )
        ) as client:
            scenario_id = _create_scenario(client)
            for field, value, expected_field in (
                ("height_m", 50.0, "edits[0].tree.height_m"),
                ("canopy_diameter_m", 40.0, "edits[0].tree.canopy_diameter_m"),
                ("u", 1.5, "edits[0].tree.u"),
                ("v", -0.1, "edits[0].tree.v"),
                ("trunk_ratio", 1.0, "edits[0].tree.trunk_ratio"),
                ("transmissivity", 1.5, "edits[0].tree.transmissivity"),
            ):
                tree = {
                    "tree_id": "bad-tree",
                    "component_type": "broad_canopy",
                    "u": 0.5,
                    "v": 0.5,
                    "height_m": 8.0,
                    "canopy_diameter_m": 6.0,
                }
                tree[field] = value
                response = client.post(
                    f"/api/v1/scenarios/{scenario_id}/edits",
                    json={
                        "base_scene_version": 0,
                        "edits": [{"operation": "add", "tree": tree}],
                    },
                )
                assert response.status_code == 422, (field, response.text)
                error = response.json()["error"]
                assert error["code"] == "invalid_tree_geometry"
                assert error["field"] == expected_field


# ---------------------------------------------------------------------------
# Ops health endpoints
# ---------------------------------------------------------------------------


class TestHealthEndpoints:
    def test_liveness_is_trivially_live(self, tmp_path: Path) -> None:
        solver = FakeSolver()
        with TestClient(make_app(tmp_path, solver)) as client:
            response = client.get("/health/live")
            assert response.status_code == 200
            body = response.json()
            assert body["status"] == "live"
            assert "time_utc" in body

    def test_readiness_passes_when_healthy(self, tmp_path: Path) -> None:
        solver = FakeSolver()
        with TestClient(make_app(tmp_path, solver)) as client:
            response = client.get("/health/ready")
            assert response.status_code == 200, response.text
            body = response.json()
            assert body["status"] == "ready"
            checks = body["checks"]
            assert checks["store"]["status"] == "ok"
            assert checks["worker"]["status"] == "ok"
            assert checks["sites"][SITE_ID] == "ok"
            assert checks["result_fs"]["status"] == "ok"
            assert "worker_running" in client.get("/healthz").json()

    def test_readiness_fails_without_worker(self, tmp_path: Path) -> None:
        solver = FakeSolver()
        app = make_app(
            tmp_path,
            solver,
            start_worker=False,
            requests_per_minute_per_ip=None,
            edits_per_minute=None,
        )
        with TestClient(app) as client:
            response = client.get("/health/ready")
            assert response.status_code == 503
            body = response.json()
            assert body["status"] == "not_ready"
            assert body["checks"]["worker"]["status"] == "stopped"

    def test_readiness_fails_on_newer_store_schema(self, tmp_path: Path) -> None:
        solver = FakeSolver()
        app = make_app(
            tmp_path,
            solver,
            requests_per_minute_per_ip=None,
            edits_per_minute=None,
        )
        # Simulate a store written by newer code.
        app.state.context.store.schema_version = 99
        with TestClient(app) as client:
            response = client.get("/health/ready")
            assert response.status_code == 503
            assert "newer" in response.json()["checks"]["store"]["error"]

    def test_worker_diagnostics_shape(self, tmp_path: Path) -> None:
        solver = FakeSolver()
        with TestClient(
            make_app(
                tmp_path, solver, requests_per_minute_per_ip=None, edits_per_minute=None
            )
        ) as client:
            response = client.get("/health/worker")
            assert response.status_code == 200
            body = response.json()
            assert body["worker_running"] is True
            assert isinstance(body["worker_pid"], int)
            assert body["worker_revision"]
            assert body["current_job_id"] is None
            assert body["idle"] is True
            assert body["seconds_since_heartbeat"] < 10
            assert "last_success_at_utc" in body
            assert "rss_bytes" in body
            assert body["site_cache_versions"][SITE_ID].startswith(SITE_ID)
            assert body["job_status_counts"] == {}

    def test_worker_diagnostics_report_running_job(self, tmp_path: Path) -> None:
        solver = FakeSolver()
        solver.gate = threading.Event()
        app = make_app(
            tmp_path,
            solver,
            requests_per_minute_per_ip=None,
            edits_per_minute=None,
        )
        with TestClient(app) as client:
            scenario_id = _create_scenario(client)  # baseline from cache: no job
            # Force a real job through the gated solver.
            edit = client.post(
                f"/api/v1/scenarios/{scenario_id}/edits", json=_edit_body()
            )
            assert edit.status_code == 202
            assert solver.entered.wait(timeout=10.0)
            mid = client.get("/health/worker").json()
            assert mid["current_job_id"] == edit.json()["job_id"]
            assert mid["idle"] is False
            solver.gate.set()  # release the gated solver (wait() blocks until set)
            deadline = time.monotonic() + 15.0
            while time.monotonic() < deadline:
                if client.get(f"/api/v1/jobs/{edit.json()['job_id']}").json()[
                    "status"
                ] == "complete":
                    break
                time.sleep(0.02)
            done = client.get("/health/worker").json()
            assert done["current_job_id"] is None
            assert done["idle"] is True
            assert done["jobs_dispatched"] >= 1
            assert done["job_status_counts"].get("complete", 0) >= 1
            assert done["last_success_at_utc"] is not None


# ---------------------------------------------------------------------------
# SSE events endpoint (unit level; the smoke test covers it over HTTP)
# ---------------------------------------------------------------------------


class TestEventsStream:
    def _parse_events(self, text: str) -> list[dict[str, str]]:
        events: list[dict[str, str]] = []
        current: dict[str, str] = {}
        for line in text.splitlines():
            if not line.strip():
                if current:
                    events.append(current)
                    current = {}
                continue
            if line.startswith(":"):
                continue
            kind, _, value = line.partition(": ")
            current[kind] = value
        return events

    def test_streams_progress_and_result_ready(self, tmp_path: Path) -> None:
        solver = FakeSolver()
        solver.gate = threading.Event()
        app = make_app(
            tmp_path,
            solver,
            requests_per_minute_per_ip=None,
            edits_per_minute=None,
        )
        with TestClient(app) as client:
            scenario_id = _create_scenario(client)
            edit = client.post(
                f"/api/v1/scenarios/{scenario_id}/edits", json=_edit_body()
            )
            assert solver.entered.wait(timeout=10.0)
            # Release the gated solver from a helper thread AFTER the stream
            # has seen the running job: the test transport buffers the SSE
            # body, so releasing from the main thread before reading would
            # deadlock the generator against the never-finishing job.
            threading.Timer(1.0, solver.gate.set).start()
            with client.stream(
                "GET",
                f"/api/v1/scenarios/{scenario_id}/events",
                headers={"Accept": "text/event-stream"},
                params={"wait_for_job_seconds": 5, "timeout_seconds": 30},
            ) as stream:
                assert stream.status_code == 200
                assert stream.headers["content-type"].startswith("text/event-stream")
                assert stream.headers["cache-control"] == "no-cache"
                assert stream.headers["x-accel-buffering"] == "no"
                text = "".join(chunk for chunk in stream.iter_text())
        events = self._parse_events(text)
        names = [event["event"] for event in events]
        assert "job-progress" in names
        assert names[-1] == "result-ready"
        ready = next(e for e in events if e["event"] == "result-ready")
        assert '"scene_version"' in ready["data"]
        assert ready["data"].count("result_manifest_url") == 1

    def test_terminal_replay_for_late_subscriber(self, tmp_path: Path) -> None:
        solver = FakeSolver()
        with TestClient(
            make_app(
                tmp_path,
                solver,
                requests_per_minute_per_ip=None,
                edits_per_minute=None,
            )
        ) as client:
            scenario_id = _create_scenario(client)
            edit = client.post(
                f"/api/v1/scenarios/{scenario_id}/edits", json=_edit_body()
            )
            job_id = edit.json()["job_id"]
            deadline = time.monotonic() + 15.0
            while time.monotonic() < deadline:
                if client.get(f"/api/v1/jobs/{job_id}").json()["status"] == "complete":
                    break
                time.sleep(0.02)
            response = client.get(
                f"/api/v1/scenarios/{scenario_id}/events",
                headers={"Accept": "text/event-stream"},
            )
            assert response.status_code == 200
            events = self._parse_events(response.text)
            assert events[-1]["event"] == "result-ready"

    def test_unknown_scenario_404(self, tmp_path: Path) -> None:
        solver = FakeSolver()
        with TestClient(
            make_app(
                tmp_path,
                solver,
                requests_per_minute_per_ip=None,
                edits_per_minute=None,
            )
        ) as client:
            response = client.get(
                "/api/v1/scenarios/does-not-exist/events",
                headers={"Accept": "text/event-stream"},
            )
            assert response.status_code == 404
            assert response.json()["error"]["code"] == "scenario_not_found"
