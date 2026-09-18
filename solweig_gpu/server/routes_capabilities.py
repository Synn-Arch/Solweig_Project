# SPDX-License-Identifier: GPL-3.0-only
"""Capability discovery and view-only scenario endpoints (UEDIT-007).

``GET /api/v1/capabilities`` (alias ``/capabilities``) serves the engine's
capability document verbatim from
:func:`solweig_gpu.incremental.capabilities.capability_document` — the
frontend's single discovery point (``frontend_spec.md``: "The frontend
retrieves capability metadata from the API. It should not hard-code the
complete tool set.").

``POST /api/v1/scenarios/{scenario_id}/views`` answers view-only
operations (the ``output_view`` family: select_layer / compare / legend /
cached_time) from capability metadata plus the LATEST PUBLISHED result —
no solver job is ever enqueued, which is exactly the UEDIT-007 contract
this module's tests prove: the store's job ledger and the injected solver
see zero new entries across a view request.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from solweig_gpu.incremental.capabilities import capability_document
from solweig_gpu.server import patch_codec
from solweig_gpu.server.models import ApiError, ViewRequest
from solweig_gpu.server.store import (
    ScenarioNotFound,
    Store,
    result_manifest_url,
    result_payload_url,
)

router = APIRouter(tags=["capabilities"])


def _document(request: Request) -> dict[str, Any]:
    """The capability document, computed once per app process.

    The document is a pure function of registry metadata + the graph +
    the adapter spec tables, none of which change at runtime; caching it
    keeps the discovery endpoint O(1) per request.
    """
    cached = getattr(request.app.state, "capability_document", None)
    if cached is None:
        cached = capability_document()
        request.app.state.capability_document = cached
    return cached


def _view_section(request: Request) -> dict[str, Any]:
    return _document(request)["view_only"]


def _capabilities_response(request: Request) -> dict[str, Any]:
    """The engine document, plus the app-level shared-world pointer.

    ``default_workspace_id`` is deployment state, not engine vocabulary, so
    it is merged here at the route — the engine-side
    :func:`capability_document` stays verbatim and apps without a shared
    world serve it unchanged. The merged document is cached per process
    alongside the engine document (both are immutable while serving).
    """
    document = _document(request)
    shared = getattr(request.app.state, "shared_scenario_id", None)
    if shared is None:
        return document
    cached = getattr(request.app.state, "capability_response_document", None)
    if cached is not None and cached.get("default_workspace_id") == shared:
        return cached
    merged = {**document, "default_workspace_id": shared}
    request.app.state.capability_response_document = merged
    return merged


@router.get("/api/v1/capabilities")
def get_capabilities(request: Request) -> JSONResponse:
    return JSONResponse(content=_capabilities_response(request))


@router.get("/capabilities", include_in_schema=False)
def get_capabilities_alias(request: Request) -> JSONResponse:
    """Stable short alias so a client can bootstrap without version wiring."""
    return get_capabilities(request)


# ---------------------------------------------------------------------------
# View-only scenario operations
# ---------------------------------------------------------------------------


def _require_scenario(request: Request, scenario_id: str):
    try:
        return _store(request).require_scenario(scenario_id)
    except ScenarioNotFound:
        raise ApiError(
            "scenario_not_found", f"scenario {scenario_id!r} does not exist", status=404
        ) from None


def _store(request: Request) -> Store:
    return request.app.state.context.store


def _validate_layer(
    request: Request,
    layer: str,
    view: dict[str, Any],
    *,
    field: str,
) -> None:
    """A layer is selectable only when the output_view metadata produces it."""
    producible = view["producible_layers"]
    if layer not in producible:
        raise ApiError(
            "invalid_request",
            f"layer {layer!r} is not a producible view layer; supported: "
            f"{producible}",
            field=field,
        )


def _variable_meta(manifest: dict[str, Any], layer: str) -> dict[str, Any] | None:
    for entry in manifest.get("variables", []):
        if entry.get("name") == layer:
            return entry
    return None


def create_view_payload(
    request: Request,
    record,
    payload: ViewRequest,
) -> dict[str, Any]:
    """Compose one view-only answer; raises :class:`ApiError` on refusals.

    Shared by the route and tests. Every branch reads ALREADY-PUBLISHED
    state only: the capability document, the scenario record, and the
    stored result manifest/payload. Nothing here submits, enqueues, or
    schedules a job.
    """
    view = _view_section(request)
    if payload.operation not in view["operations"]:
        raise ApiError(
            "invalid_request",
            f"unknown view operation {payload.operation!r}; supported: "
            f"{view['operations']}",
            field="operation",
        )
    _validate_layer(request, payload.layer, view, field="layer")
    if payload.operation == "compare":
        if payload.compare_layer is None:
            raise ApiError(
                "invalid_request",
                "compare views require compare_layer",
                field="compare_layer",
            )
        _validate_layer(request, payload.compare_layer, view, field="compare_layer")

    store = _store(request)
    scenario_id = record.scenario_id
    scene_version = (
        int(payload.scene_version)
        if payload.scene_version is not None
        else int(record.exact_result_version)
    )
    result = store.get_result(scenario_id, scene_version)
    if result is None:
        raise ApiError(
            "result_not_ready",
            f"scenario {scenario_id!r} has no published result at scene "
            f"version {scene_version} to view",
            status=404,
        )
    manifest = result.manifest
    body: dict[str, Any] = {
        "operation": payload.operation,
        "scenario_id": scenario_id,
        "scene_version": scene_version,
        "job_enqueued": False,
        "zero_scientific_jobs": True,
        "result_manifest_url": result_manifest_url(scenario_id, scene_version),
        "result_payload_url": result_payload_url(scenario_id, scene_version),
    }

    if payload.operation == "cached_time":
        cached = [int(index) for index in manifest.get("time_indices", [])]
        if payload.time_index not in cached:
            raise ApiError(
                "view_not_available",
                f"time index {payload.time_index} is not cached in the "
                f"published result (cached: {cached})",
                details={"cached_time_indices": cached},
                status=409,
            )
        body["cached_time_indices"] = cached
        body["time_index"] = payload.time_index
        return body

    meta = _variable_meta(manifest, payload.layer)
    if meta is None:
        published = [entry.get("name") for entry in manifest.get("variables", [])]
        raise ApiError(
            "view_not_available",
            f"layer {payload.layer!r} is producible by the engine but is not "
            f"carried by the published result at scene version {scene_version} "
            f"(published: {published}); request it via an edit's "
            "requested_result and view it once published",
            details={"published_layers": published},
            status=409,
        )
    body["layer"] = meta
    body["time_index"] = int(payload.time_index)

    if payload.operation == "compare":
        other = _variable_meta(manifest, payload.compare_layer)
        if other is None:
            published = [entry.get("name") for entry in manifest.get("variables", [])]
            raise ApiError(
                "view_not_available",
                f"compare layer {payload.compare_layer!r} is not carried by "
                f"the published result at scene version {scene_version} "
                f"(published: {published})",
                details={"published_layers": published},
                status=409,
            )
        body["compare_layer"] = other
    elif payload.operation == "legend":
        index = payload.time_index
        shape = [int(v) for v in meta.get("shape", [])]
        if len(shape) == 3 and not 0 <= index < shape[0]:
            raise ApiError(
                "view_not_available",
                f"time index {index} is outside the published result's "
                f"{shape[0]} time steps",
                status=409,
            )
        body["legend"] = _legend_document(
            manifest, result, payload.layer, index, nodata=meta.get("nodata")
        )
    return body


def _legend_document(
    manifest: dict[str, Any],
    result,
    layer: str,
    index: int,
    *,
    nodata: str | None,
) -> dict[str, Any]:
    """Legend for one published plane, cheapest path first.

    1. Manifest ``statistics`` (served by every result published since
       statistics were added): a pure-metadata answer, no decode at all.
    2. ``patch_codec.decode_plane``: streams exactly the requested plane
       out of the zstd payload (checksum verified over the compressed
       bytes first) and stops decompressing at the plane's end — used for
       older manifests. The pre-statistics full ``decode_payload`` path is
       gone; even the fallback never materializes the whole result.
    """
    statistics = manifest.get("statistics")
    if isinstance(statistics, dict):
        per_time = statistics.get(layer)
        for entry in per_time or []:
            if int(entry.get("time_index", -1)) == int(index):
                return {
                    "time_index": int(index),
                    "nodata": nodata,
                    "count": int(entry["count"]),
                    **(
                        {
                            "min": float(entry["min"]),
                            "max": float(entry["max"]),
                            "mean": float(entry["mean"]),
                        }
                        if int(entry["count"])
                        else {}
                    ),
                }
    plane = patch_codec.decode_plane(manifest, result.payload_bytes(), layer, index)
    finite = plane[np.isfinite(plane)]
    return {
        "time_index": int(index),
        "nodata": nodata,
        "count": int(finite.size),
        **(
            {
                "min": float(finite.min()),
                "max": float(finite.max()),
                "mean": float(finite.mean()),
            }
            if finite.size
            else {}
        ),
    }


@router.post("/api/v1/scenarios/{scenario_id}/views", status_code=200)
def create_view(
    payload: ViewRequest, scenario_id: str, request: Request
) -> JSONResponse:
    record = _require_scenario(request, scenario_id)
    _check_view_rate_limit(request, scenario_id)
    return JSONResponse(content=create_view_payload(request, record, payload))


def _check_view_rate_limit(request: Request, scenario_id: str) -> None:
    """Per-scenario view budget (same limiter machinery as the edit path)."""
    limiter = getattr(request.app.state, "views_rate_limiter", None)
    if limiter is not None:
        limiter.check(f"views:{scenario_id}")
