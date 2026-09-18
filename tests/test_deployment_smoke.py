# SPDX-License-Identifier: GPL-3.0-only
"""T6 deployment smoke: boot the real server stack and drive the full path.

Boots the FastAPI app through a REAL uvicorn server on an ephemeral port
(temp store + synthetic site cache, fake solver), then exercises every
deployment-critical behaviour end to end over HTTP:

* ops health surface (``/health/live``, ``/health/ready``, ``/health/worker``);
* create scenario → post edit → SSE event stream → job result → manifest;
* binary payload delivery with BOTH ``Accept`` encodings (zstd + identity),
  checksum / ETag verified per encoding, conditional 304;
* payload decodability against the manifest via ``patch_codec``;
* scenario reset;
* REL-003 controls over HTTP on a tight-profile server: 413 oversized body,
  429 per-IP rate breach with ``Retry-After``, and proof that
  ``uvicorn --proxy-headers`` makes the app see the proxy's
  ``X-Forwarded-For`` client (the TLS-termination trust model).

Skipped when the ``server`` extra is not installed. Run with::

    pytest -m deployment tests/test_deployment_smoke.py
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi", reason="server extra (fastapi) not installed")
pytest.importorskip("uvicorn", reason="server extra (uvicorn) not installed")
pytest.importorskip("httpx", reason="server extra (httpx) not installed")
pytest.importorskip("zstandard", reason="server extra (zstandard) not installed")

import httpx  # noqa: E402
import uvicorn  # noqa: E402
import zstandard  # noqa: E402

from solweig_gpu.server import patch_codec  # noqa: E402
from solweig_gpu.server.app import create_app  # noqa: E402
from tests.test_server_api import SITE_ID, FakeSolver, make_site_cache  # noqa: E402

import numpy as np  # noqa: E402

HOST = "127.0.0.1"
EDIT_BODY: dict[str, Any] = {
    "base_scene_version": 0,
    "edits": [
        {
            "operation": "add",
            "tree": {
                "tree_id": "smoke-tree",
                "component_type": "broad_canopy",
                "u": 0.5,
                "v": 0.5,
                "height_m": 9.0,
                "canopy_diameter_m": 6.0,
            },
        }
    ],
}


def start_server(app, *, proxy_headers: bool = False):
    """Run ``app`` under a real uvicorn server on an ephemeral port."""
    config = uvicorn.Config(
        app,
        host=HOST,
        port=0,  # ephemeral
        log_level="warning",
        lifespan="on",
        proxy_headers=proxy_headers,
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="uvicorn-smoke", daemon=True)
    thread.start()
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline and not server.started:
        time.sleep(0.02)
    if not server.started:
        server.should_exit = True
        raise AssertionError("uvicorn server did not start within 30s")
    port = server.servers[0].sockets[0].getsockname()[1]
    return server, thread, f"http://{HOST}:{port}"


def stop_server(server, thread) -> None:
    server.should_exit = True
    thread.join(timeout=15.0)


def wait_ready(client: httpx.Client, timeout: float = 30.0) -> dict[str, Any]:
    """Poll /health/ready (ops-exempt from rate limits) until 200."""
    deadline = time.monotonic() + timeout
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        try:
            response = client.get("/health/ready")
        except httpx.TransportError:
            time.sleep(0.05)
            continue
        if response.status_code == 200:
            return response.json()
        last = {"status": response.status_code, "body": response.text}
        time.sleep(0.05)
    raise AssertionError(f"server never became ready: {last}")


def parse_sse_events(lines: list[str]) -> list[dict[str, str]]:
    events: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for line in lines:
        if not line.strip():
            if current:
                events.append(current)
                current = {}
            continue
        if line.startswith(":"):
            continue
        kind, _, value = line.partition(": ")
        current[kind] = value
    if current:  # trailing event before the stream/break cut the blank line
        events.append(current)
    return events


@pytest.mark.deployment
@pytest.mark.integration
class TestDeploymentSmoke:
    def test_full_deployment_path_over_real_http(self, tmp_path: Path) -> None:
        """Production-profile server: default security controls ON."""
        solver = FakeSolver()
        solver.gate = threading.Event()  # blocks mid-solve until released
        app = create_app(
            state_root=tmp_path / "state",
            sites={SITE_ID: {"cache_dir": make_site_cache(tmp_path)}},
            solver_factory=lambda context: solver,
            coalescing_window_ms=20.0,
        )
        server, thread, base_url = start_server(app)
        try:
            with httpx.Client(base_url=base_url, timeout=30.0) as client:
                # -- ops surface ------------------------------------------------
                live = client.get("/health/live")
                assert live.status_code == 200
                assert live.json()["status"] == "live"

                ready = wait_ready(client)
                assert ready["status"] == "ready"
                assert ready["checks"]["worker"]["status"] == "ok"
                assert ready["checks"]["sites"][SITE_ID] == "ok"

                worker = client.get("/health/worker")
                assert worker.status_code == 200
                worker_body = worker.json()
                assert worker_body["worker_running"] is True
                assert worker_body["idle"] is True
                assert worker_body["worker_pid"] > 0

                # -- create → edit → SSE → result ------------------------------
                created = client.post("/api/v1/scenarios", json={"site_id": SITE_ID})
                assert created.status_code == 201, created.text
                scenario_id = created.json()["scenario_id"]

                edit = client.post(
                    f"/api/v1/scenarios/{scenario_id}/edits",
                    json=EDIT_BODY,
                    headers={"Idempotency-Key": "smoke-edit-1"},
                )
                assert edit.status_code == 202, edit.text
                job_id = edit.json()["job_id"]

                # SSE: subscribe while the gated solver holds the job, read
                # at least one job-progress, release, then read result-ready.
                # (Also keeps this test well under the default per-IP budget:
                # no polling loop.)
                sse_lines: list[str] = []
                with client.stream(
                    "GET",
                    f"/api/v1/scenarios/{scenario_id}/events",
                    headers={"Accept": "text/event-stream"},
                    params={"wait_for_job_seconds": 10},
                    timeout=30.0,
                ) as stream:
                    assert stream.status_code == 200
                    assert stream.headers["content-type"].startswith(
                        "text/event-stream"
                    )
                    assert stream.headers["x-accel-buffering"] == "no"
                    saw_progress = False
                    saw_ready = False
                    for line in stream.iter_lines():
                        sse_lines.append(line)
                        if "event: job-progress" in line:
                            saw_progress = True
                            solver.gate.set()  # let the job finish
                        if saw_progress and "event: result-ready" in line:
                            saw_ready = True
                        elif saw_ready and not line.strip():
                            break  # blank line closes the result-ready event
                events = parse_sse_events(sse_lines)
                names = [event["event"] for event in events]
                assert "job-progress" in names
                assert "result-ready" in names
                progress = next(e for e in events if e["event"] == "job-progress")
                assert f'"job_id":"{job_id}"' in progress["data"]
                ready_event = next(e for e in events if e["event"] == "result-ready")
                assert '"result_manifest_url"' in ready_event["data"]

                # Worker diagnostics saw the job and went idle again.
                worker_after = client.get("/health/worker").json()
                assert worker_after["current_job_id"] is None
                assert worker_after["job_status_counts"].get("complete", 0) >= 1

                # -- result manifest + payload, both encodings -----------------
                manifest_response = client.get(
                    f"/api/v1/scenarios/{scenario_id}/results/1"
                )
                assert manifest_response.status_code == 200
                manifest = manifest_response.json()
                assert manifest["scene_version"] == 1
                assert manifest["exact"] is True

                zstd = client.get(
                    f"/api/v1/scenarios/{scenario_id}/results/1/payload",
                    headers={"Accept": patch_codec.PATCH_MEDIA_TYPE},
                )
                assert zstd.status_code == 200
                assert zstd.headers["content-type"].startswith(
                    "application/vnd.solweig.patch+zstd"
                )
                assert zstd.headers["X-SOLWEIG-Checksum"] == manifest["checksum"]
                assert (
                    zstd.headers["X-SOLWEIG-Checksum"]
                    == patch_codec.checksum_payload(zstd.content)
                )
                assert zstd.headers["ETag"] == patch_codec.payload_etag(
                    manifest["checksum"]
                )
                assert zstd.headers["X-SOLWEIG-Scene-Version"] == "1"

                # Conditional GET: per-encoding ETag serves 304.
                not_modified = client.get(
                    f"/api/v1/scenarios/{scenario_id}/results/1/payload",
                    headers={
                        "Accept": patch_codec.PATCH_MEDIA_TYPE,
                        "If-None-Match": zstd.headers["ETag"],
                    },
                )
                assert not_modified.status_code == 304

                identity = client.get(
                    f"/api/v1/scenarios/{scenario_id}/results/1/payload",
                    headers={"Accept": patch_codec.IDENTITY_MEDIA_TYPE},
                )
                assert identity.status_code == 200
                assert identity.headers["content-type"].startswith(
                    "application/vnd.solweig.patch+identity"
                )
                # Identity checksum/ETag are over the SERVED (uncompressed)
                # bytes — a different value than the zstd manifest checksum.
                assert (
                    identity.headers["X-SOLWEIG-Checksum"]
                    == patch_codec.checksum_payload(identity.content)
                )
                assert (
                    identity.headers["X-SOLWEIG-Checksum"]
                    != zstd.headers["X-SOLWEIG-Checksum"]
                )
                # Identity bytes are exactly the decompressed zstd payload.
                decompressed = zstandard.ZstdDecompressor().decompressobj().decompress(
                    zstd.content
                )
                assert decompressed == identity.content

                # Decodability through the contract codec, both encodings.
                arrays = patch_codec.decode_payload(manifest, zstd.content)
                identity_arrays = patch_codec.decode_identity_payload(
                    manifest, identity.content
                )
                for decoded in (arrays, identity_arrays):
                    assert set(decoded) >= {"utci"}
                    window = manifest["window"]
                    rows = window["row_stop"] - window["row_start"]
                    cols = window["col_stop"] - window["col_start"]
                    for name, array in decoded.items():
                        assert array.shape[1:] == (rows, cols)
                        assert array.dtype == np.float32
                        assert bool(np.isfinite(array).all())

                # -- capabilities document (u-d1) ------------------------------
                capabilities = client.get("/api/v1/capabilities")
                assert capabilities.status_code == 200, capabilities.text
                document = capabilities.json()
                assert document["schema_version"] == 1
                adapter_ids = {adapter["id"] for adapter in document["adapters"]}
                # The four universal-transport families are integrated.
                assert {
                    "meteorological_forcing",
                    "landcover_surface",
                    "building_geometry",
                    "model_receptor_parameters",
                } <= adapter_ids
                assert document["view_only"]["zero_scientific_jobs"] is True

                # -- view-only endpoint: zero scientific jobs -------------------
                jobs_before = client.get("/metrics").json()["jobs"]

                view = client.post(
                    f"/api/v1/scenarios/{scenario_id}/views",
                    json={"operation": "select_layer", "layer": "utci", "time_index": 0},
                )
                assert view.status_code == 200, view.text
                view_body = view.json()
                assert view_body["job_enqueued"] is False
                assert view_body["zero_scientific_jobs"] is True
                assert view_body["layer"]["name"] == "utci"

                # Decode-free legend (manifest statistics, no payload scan).
                legend = client.post(
                    f"/api/v1/scenarios/{scenario_id}/views",
                    json={"operation": "legend", "layer": "utci", "time_index": 0},
                )
                assert legend.status_code == 200, legend.text
                assert set(legend.json()["legend"]) >= {"min", "max", "mean", "count"}

                jobs_after_view = client.get("/metrics").json()["jobs"]
                assert jobs_after_view == jobs_before, "views must not enqueue jobs"

                # -- universal edit transport (u-d4) ---------------------------
                # View-only items are answered inline: 200, zero jobs.
                universal_view = client.post(
                    f"/api/v1/scenarios/{scenario_id}/edits/universal",
                    json={
                        "base_scene_version": 1,
                        "edits": [
                            {
                                "adapter": "output_view",
                                "operation": "select_layer",
                                "values": {"layer": "utci"},
                            }
                        ],
                    },
                    headers={"Idempotency-Key": "smoke-universal-view"},
                )
                assert universal_view.status_code == 200, universal_view.text
                uv_body = universal_view.json()
                assert uv_body["job_enqueued"] is False
                assert uv_body["zero_scientific_jobs"] is True
                assert client.get("/metrics").json()["jobs"] == jobs_before

                # Typed refusals relay the registry's own vocabulary.
                tree_refusal = client.post(
                    f"/api/v1/scenarios/{scenario_id}/edits/universal",
                    json={
                        "base_scene_version": 1,
                        "edits": [
                            {
                                "adapter": "vegetation_geometry",
                                "operation": "add",
                                "values": {"height_m": 9.0},
                            }
                        ],
                    },
                    headers={"Idempotency-Key": "smoke-universal-tree"},
                )
                assert tree_refusal.status_code == 400, tree_refusal.text
                assert tree_refusal.json()["error"]["code"] == "unsupported_transport"
                assert "tree-edits-v1" in tree_refusal.json()["error"]["message"]

                water_refusal = client.post(
                    f"/api/v1/scenarios/{scenario_id}/edits/universal",
                    json={
                        "base_scene_version": 1,
                        "edits": [
                            {
                                "adapter": "landcover_surface",
                                "operation": "paint",
                                "values": {"class": 7},
                                "target": {
                                    "row_start": 0,
                                    "row_stop": 2,
                                    "col_start": 0,
                                    "col_stop": 2,
                                },
                            }
                        ],
                    },
                    headers={"Idempotency-Key": "smoke-universal-water"},
                )
                assert water_refusal.status_code == 422, water_refusal.text
                water_error = water_refusal.json()["error"]
                assert water_error["code"] == "invalid_edit_state"
                # The fence names itself — adapter verbatim, never restated.
                assert "fenced" in water_error["message"]
                assert water_error["adapter"] == "landcover_surface"

                # Refusals enqueued nothing and moved no version.
                assert client.get("/metrics").json()["jobs"] == jobs_before
                scenario_now = client.get(f"/api/v1/scenarios/{scenario_id}").json()
                assert scenario_now["scene_version"] == 1

                # -- reset ------------------------------------------------------
                reset = client.post(f"/api/v1/scenarios/{scenario_id}/reset")
                assert reset.status_code == 200, reset.text
                reset_body = reset.json()
                assert reset_body["trees"] == []
                assert reset_body["scene_version"] >= 1
        finally:
            stop_server(server, thread)

    def test_security_controls_and_proxy_headers_over_real_http(
        self, tmp_path: Path
    ) -> None:
        """Tight-profile server (launched like a proxy-fronted deployment):
        413 / 429 enforcement over HTTP, and X-Forwarded-For trust via
        ``uvicorn --proxy-headers``."""
        solver = FakeSolver()
        app = create_app(
            state_root=tmp_path / "state",
            sites={SITE_ID: {"cache_dir": make_site_cache(tmp_path)}},
            solver_factory=lambda context: solver,
            coalescing_window_ms=20.0,
            max_request_body_bytes=1024,
            requests_per_minute_per_ip=3,
            edits_per_minute=None,
        )
        server, thread, base_url = start_server(app, proxy_headers=True)
        try:
            with httpx.Client(base_url=base_url, timeout=30.0) as client:
                wait_ready(client)

                # 413: oversized body rejected at the gate.
                oversized = client.post(
                    "/api/v1/scenarios",
                    json={"site_id": SITE_ID, "name": "x" * 4096},
                )
                assert oversized.status_code == 413
                assert oversized.json()["error"]["code"] == "payload_too_large"

                # 429: the 413 POST already consumed one of the 3 per-IP
                # slots, so only two /healthz probes pass before the breach.
                statuses = [
                    client.get("/healthz").status_code for _ in range(4)
                ]
                assert statuses[:2] == [200, 200]
                assert statuses[2:] == [429, 429]
                breached = client.get("/healthz")
                assert breached.status_code == 429
                assert breached.json()["error"]["code"] == "rate_limited"
                assert int(breached.headers["Retry-After"]) >= 1

                # Proxy trust: uvicorn --proxy-headers rewrites the client
                # from X-Forwarded-For, so each proxied client gets its own
                # budget (127.0.0.1 is in the default forwarded allow-list).
                proxied = httpx.Client(
                    base_url=base_url,
                    timeout=30.0,
                    headers={"X-Forwarded-For": "198.51.100.77"},
                )
                try:
                    for _ in range(3):
                        assert proxied.get("/healthz").status_code == 200
                    assert proxied.get("/healthz").status_code == 429

                    rotated = httpx.Client(
                        base_url=base_url,
                        timeout=30.0,
                        headers={"X-Forwarded-For": "198.51.100.99"},
                    )
                    try:
                        # A different forwarded client is a different budget.
                        assert rotated.get("/healthz").status_code == 200
                    finally:
                        rotated.close()
                finally:
                    proxied.close()

                # X-Forwarded-Proto (TLS terminated upstream) is accepted
                # alongside; the app itself stays protocol-agnostic.
                assert (
                    client.get(
                        "/health/live",
                        headers={"X-Forwarded-Proto": "https"},
                    ).status_code
                    == 200
                )
        finally:
            stop_server(server, thread)
