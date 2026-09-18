# SPDX-License-Identifier: GPL-3.0-only
"""Capability endpoint + view-only operation tests (U-D1, UEDIT-007).

UEDIT-007's evidence lives here: a view-only request (the ``output_view``
family) is answered from capability metadata plus the published result
manifest and enqueues ZERO scientific jobs — asserted against BOTH the
store's durable job ledger and the injected solver's call record.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Literal, get_args, get_origin

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_server_api import (  # noqa: E402
    SITE_ID,
    FakeSolver,
    SCENARIO0,
    load_baseline,
    make_app,
    make_site_cache,
    make_tree,
    wait_for_job,
)

from solweig_gpu.incremental.capabilities import capability_document
from solweig_gpu.incremental.edit_registry import builtin_adapter_metadata
from solweig_gpu.server.models import ViewRequest


@pytest.fixture()
def view_client(tmp_path: Path):
    solver = FakeSolver()
    cache = make_site_cache(tmp_path)
    app = make_app(tmp_path, solver, site_cache=cache)
    with TestClient(app) as client:
        yield client, solver


def _job_count(client: TestClient) -> int:
    return int(client.get("/metrics").json()["jobs"]["queued"]) + sum(
        int(client.get("/metrics").json()["jobs"][status])
        for status in ("running", "complete", "superseded", "failed", "cancelled")
    )


def _views_url(scenario_id: str) -> str:
    return f"/api/v1/scenarios/{scenario_id}/views"


def test_capabilities_endpoint_serves_the_document(view_client) -> None:
    client, _solver = view_client
    response = client.get("/api/v1/capabilities")
    assert response.status_code == 200
    document = capability_document()
    assert response.json() == document
    # The short alias serves the same document (bootstrap without version
    # wiring).
    alias = client.get("/capabilities")
    assert alias.status_code == 200
    assert alias.json() == document
    assert len(document["adapters"]) == 9
    assert document["view_only"]["zero_scientific_jobs"] is True


def test_view_request_operation_literal_mirrors_the_registry() -> None:
    """u-d1b nit b: ``ViewRequest.operation``'s pydantic Literal is a
    first-parse mirror of the ``output_view`` adapter's registered
    operations (edit_registry). The route re-validates against the
    capability document (fail-closed), but a registry change must not
    silently diverge the mirror either way: a dropped operation would
    422 before the document could answer, an added one would be
    unreachable — this fails on both."""
    annotation = ViewRequest.model_fields["operation"].annotation
    assert get_origin(annotation) is Literal
    literal_operations = set(get_args(annotation))
    registry_operations = {
        metadata.operations
        for metadata in builtin_adapter_metadata()
        if metadata.id == "output_view"
    }
    assert len(registry_operations) == 1
    assert literal_operations == set(registry_operations.pop()), (
        "ViewRequest.operation Literal drifted from the output_view "
        "adapter's registered operations"
    )
    # And the served document agrees with the registry (the route's
    # authority), closing the three-way mirror.
    document = capability_document()
    assert set(document["view_only"]["operations"]) == literal_operations


def test_view_select_layer_enqueues_zero_scientific_jobs(view_client) -> None:
    """UEDIT-007: a view-only request answers without any solver job."""
    client, solver = view_client
    scenario_id = SCENARIO0(client)
    wait_for_scenario_exact(client, scenario_id)
    jobs_before = _job_count(client)
    calls_before = len(solver.calls)

    response = client.post(
        _views_url(scenario_id),
        json={"operation": "select_layer", "layer": "utci", "time_index": 1},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["operation"] == "select_layer"
    assert body["layer"]["name"] == "utci"
    assert body["layer"]["shape"][0] >= 2  # time dimension covers index 1
    assert body["scene_version"] == 0
    assert body["job_enqueued"] is False
    assert body["zero_scientific_jobs"] is True
    assert body["result_payload_url"].endswith("/payload")

    # The proof: the durable job ledger and the solver saw nothing new.
    assert _job_count(client) == jobs_before
    assert len(solver.calls) == calls_before


def test_view_rejects_layer_outside_capability(view_client) -> None:
    """The layer vocabulary is capability-gated, not a server constant."""
    client, solver = view_client
    scenario_id = SCENARIO0(client)
    jobs_before = _job_count(client)
    response = client.post(
        _views_url(scenario_id),
        json={"operation": "select_layer", "layer": "wbgt", "time_index": 0},
    )
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["field"] == "layer"
    assert "wbgt" in error["message"]
    assert _job_count(client) == jobs_before


def test_view_time_index_accepts_only_json_integers(view_client) -> None:
    """u-e3-review NIT-2 (view level): pydantic lax mode coerced ``True``->1,
    ``5.0``->5 and ``"07"``->7 at the model boundary. ``True`` selecting
    timestep 1 silently is a wrong-timestep view answer, the exact class
    the values-level strictness exists to kill."""
    client, _solver = view_client
    scenario_id = SCENARIO0(client)
    wait_for_scenario_exact(client, scenario_id)
    for bad in [5.0, True, "07"]:
        response = client.post(
            _views_url(scenario_id),
            json={"operation": "select_layer", "layer": "utci", "time_index": bad},
        )
        # Model-boundary refusal: the app's RequestValidationError envelope
        # (400 invalid_request naming the field), not a silent coercion and
        # never a 500.
        assert response.status_code == 400, (bad, response.text)
        error = response.json()["error"]
        assert error["code"] == "invalid_request"
        assert error["field"] == "time_index"


def test_view_rejects_producible_but_unpublished_layer(view_client) -> None:
    """`shadow` is producible by the engine but absent from this result."""
    client, _solver = view_client
    scenario_id = SCENARIO0(client)
    response = client.post(
        _views_url(scenario_id),
        json={"operation": "select_layer", "layer": "shadow", "time_index": 0},
    )
    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "view_not_available"
    assert "published_layers" in error


def test_view_after_edit_tracks_published_version(view_client) -> None:
    """Views ride the scenario's published versions without new jobs."""
    client, solver = view_client
    scenario_id = SCENARIO0(client)
    wait_for_scenario_exact(client, scenario_id)
    edit = client.post(
        f"/api/v1/scenarios/{scenario_id}/edits",
        json={
            "base_scene_version": 0,
            "edits": [{"operation": "add", "tree": make_tree("tree_01")}],
        },
    )
    assert edit.status_code == 202, edit.text
    job = wait_for_job(client, edit.json()["job_id"])
    assert job["status"] == "complete"
    jobs_at_edit = _job_count(client)

    response = client.post(
        _views_url(scenario_id),
        json={"operation": "select_layer", "layer": "utci", "time_index": 1},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["scene_version"] == 1
    assert _job_count(client) == jobs_at_edit
    assert len(solver.calls) == 1  # the edit's job only


def test_view_legend_computes_from_published_payload(view_client, tmp_path) -> None:
    client, _solver = view_client
    scenario_id = SCENARIO0(client)
    baseline = load_baseline(tmp_path / "cache" / SITE_ID, "utci")
    response = client.post(
        _views_url(scenario_id),
        json={"operation": "legend", "layer": "utci", "time_index": 1},
    )
    assert response.status_code == 200
    legend = response.json()["legend"]
    assert legend["min"] == pytest.approx(float(baseline[1].min()))
    assert legend["max"] == pytest.approx(float(baseline[1].max()))
    assert legend["mean"] == pytest.approx(float(baseline[1].mean()))
    assert legend["count"] == int(baseline[1].size)


def test_view_cached_time_lists_published_times(view_client) -> None:
    client, _solver = view_client
    scenario_id = SCENARIO0(client)
    response = client.post(
        _views_url(scenario_id),
        json={"operation": "cached_time", "layer": "utci", "time_index": 2},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["cached_time_indices"] == [0, 1, 2]
    assert body["time_index"] == 2

    missing = client.post(
        _views_url(scenario_id),
        json={"operation": "cached_time", "layer": "utci", "time_index": 99},
    )
    assert missing.status_code == 409
    assert missing.json()["error"]["code"] == "view_not_available"


def test_view_compare_carries_both_layer_metas(view_client) -> None:
    client, solver = view_client
    scenario_id = SCENARIO0(client)
    jobs_before = _job_count(client)
    response = client.post(
        _views_url(scenario_id),
        json={
            "operation": "compare",
            "layer": "utci",
            "compare_layer": "tmrt",
            "time_index": 0,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["layer"]["name"] == "utci"
    assert body["compare_layer"]["name"] == "tmrt"
    assert _job_count(client) == jobs_before

    missing = client.post(
        _views_url(scenario_id),
        json={"operation": "compare", "layer": "utci", "time_index": 0},
    )
    assert missing.status_code == 400
    assert missing.json()["error"]["field"] == "compare_layer"

    unknown = client.post(
        _views_url(scenario_id),
        json={
            "operation": "compare",
            "layer": "utci",
            "compare_layer": "wbgt",
            "time_index": 0,
        },
    )
    assert unknown.status_code == 400
    assert unknown.json()["error"]["field"] == "compare_layer"


def test_view_requires_a_published_result(view_client) -> None:
    client, _solver = view_client
    scenario_id = SCENARIO0(client)
    response = client.post(
        _views_url(scenario_id),
        json={
            "operation": "select_layer",
            "layer": "utci",
            "time_index": 0,
            "scene_version": 7,  # nothing published there yet
        },
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "result_not_ready"


def test_view_unknown_scenario_is_404(view_client) -> None:
    client, _solver = view_client
    response = client.post(
        _views_url("no-such-scenario"),
        json={"operation": "select_layer", "layer": "utci", "time_index": 0},
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "scenario_not_found"


def wait_for_scenario_exact(client: TestClient, scenario_id: str) -> None:
    """Wait until the scenario's baseline result is published."""
    for _ in range(200):
        body = client.get(f"/api/v1/scenarios/{scenario_id}").json()
        if body["exact_result_version"] == body["scene_version"]:
            return
        time.sleep(0.01)
    pytest.fail("scenario never reached an exact result")


# ---------------------------------------------------------------------------
# u-d4 view hardening: decode-free legends + per-scenario view budget
# ---------------------------------------------------------------------------


def _legend_body(time_index: int = 1) -> dict:
    return {"operation": "legend", "layer": "utci", "time_index": time_index}


def test_published_manifests_carry_per_time_statistics(view_client) -> None:
    client, _solver = view_client
    scenario = SCENARIO0(client)
    wait_for_scenario_exact(client, scenario)
    manifest = client.get(f"/api/v1/scenarios/{scenario}/results/0").json()
    assert "statistics" in manifest
    utci = manifest["statistics"]["utci"]
    assert [entry["time_index"] for entry in utci] == [0, 1, 2]
    assert all({"min", "max", "mean", "count"} <= set(entry) for entry in utci)


def test_legend_answered_without_any_payload_decode(view_client, monkeypatch) -> None:
    """Statistics-bearing manifests: legend is a pure-metadata answer."""
    from solweig_gpu.server import patch_codec

    def _forbidden(*args, **kwargs):  # pragma: no cover - fails the test
        pytest.fail("legend served by decoding the payload")

    monkeypatch.setattr(patch_codec, "decode_payload", _forbidden)
    monkeypatch.setattr(patch_codec, "decode_plane", _forbidden)
    client, _solver = view_client
    scenario = SCENARIO0(client)
    wait_for_scenario_exact(client, scenario)
    response = client.post(_views_url(scenario), json=_legend_body())
    assert response.status_code == 200
    legend = response.json()["legend"]
    assert legend["time_index"] == 1
    assert legend["count"] > 0
    assert legend["min"] <= legend["mean"] <= legend["max"]


def test_legend_falls_back_to_bounded_plane_decode_for_legacy_manifests(
    view_client,
) -> None:
    """Pre-statistics manifests still serve exact legends via decode_plane."""
    client, _solver = view_client
    scenario = SCENARIO0(client)
    wait_for_scenario_exact(client, scenario)
    modern = client.post(_views_url(scenario), json=_legend_body()).json()["legend"]

    # Simulate a legacy result: strip the statistics document from the
    # stored manifest (the same dict the store serves), keep everything else.
    store = client.app.state.context.store
    result = store.get_result(scenario, 0)
    result.manifest.pop("statistics", None)
    legacy = client.post(_views_url(scenario), json=_legend_body()).json()["legend"]

    assert legacy["count"] == modern["count"]
    assert legacy["min"] == pytest.approx(modern["min"])
    assert legacy["max"] == pytest.approx(modern["max"])
    assert legacy["mean"] == pytest.approx(modern["mean"])


def test_legend_rejects_time_index_outside_manifest_shape(view_client) -> None:
    client, _solver = view_client
    scenario = SCENARIO0(client)
    wait_for_scenario_exact(client, scenario)
    response = client.post(_views_url(scenario), json=_legend_body(time_index=99))
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "view_not_available"


def test_view_requests_carry_a_per_scenario_budget(tmp_path: Path) -> None:
    solver = FakeSolver()
    cache = make_site_cache(tmp_path)
    app = make_app(tmp_path, solver, site_cache=cache, views_per_minute=2)
    with TestClient(app) as client:
        scenario = SCENARIO0(client)
        wait_for_scenario_exact(client, scenario)
        body = {"operation": "select_layer", "layer": "utci", "time_index": 0}
        for _ in range(2):
            assert client.post(_views_url(scenario), json=body).status_code == 200
        limited = client.post(_views_url(scenario), json=body)
        assert limited.status_code == 429
        assert limited.json()["error"]["code"] == "rate_limited"
        # The view budget is separate from the mutation budget (disabled in
        # make_app): an edit still goes through while views are limited.
        edit = client.post(
            f"/api/v1/scenarios/{scenario}/edits",
            json={
                "base_scene_version": 0,
                "edits": [{"operation": "add", "tree": make_tree()}],
            },
        )
        assert edit.status_code == 202
