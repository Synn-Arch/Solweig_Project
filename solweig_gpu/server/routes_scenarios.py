# SPDX-License-Identifier: GPL-3.0-only
"""Scenario, edit, reset, result, and export endpoints.

Implements the ``/api/v1/scenarios`` surface of
``docs/incremental_design_tool/api_contract.md``: idempotent mutations,
optimistic scene-version concurrency, explicit conflict envelopes, server-side
tree validation through ``solweig_gpu.incremental`` (including UV-to-world
conversion via the site manifest), and binary result delivery with
``X-SOLWEIG-*`` headers.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Mapping

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response as FastAPIResponse
from fastapi.responses import StreamingResponse

from solweig_gpu.incremental.geometry import TreeSpec
from solweig_gpu.incremental.trees import (
    TREE_CANOPY_DIAMETER_RANGE_M,
    TREE_HEIGHT_RANGE_M,
    validate_tree_preset,
)
from solweig_gpu.server import patch_codec
from solweig_gpu.server.jobs import (
    RunnerContext,
    api_tree_to_spec,
    default_limitations,
    materialize_baseline_result,
)
from solweig_gpu.server.models import (
    ApiError,
    CreateScenarioRequest,
    EditRequest,
    ExportRequest,
    ResetRequest,
    TREE_PRESETS,
    UniversalEditRequest,
)
from solweig_gpu.server.store import (
    IdempotencyConflict,
    IdempotentReplay,
    SceneVersionConflict,
    ScenarioNotFound,
    ScenarioQuotaExceeded,
    SiteIdentityMismatch,
    Store,
    StoredResponse,
    request_fingerprint,
    result_manifest_url,
    result_payload_url,
    scenario_url,
)

router = APIRouter(prefix="/api/v1/scenarios", tags=["scenarios"])

SCENE_VERSION_ETAG_PREFIX = "scene-version-"


class _IdentityPayloadCache:
    """Bounded LRU of identity (decompressed) result payloads.

    Clients that cannot use the zstd ``DecompressionStream`` request the
    identity encoding; without this cache every such request re-decompresses
    the whole payload (hundreds of MB at full-site window sizes) and
    recomputes its checksum. Entries are keyed by the durable record
    checksum, so a re-published version never serves stale bytes. Bounded by
    total bytes with least-recently-used eviction.
    """

    def __init__(self, *, max_total_bytes: int = 256 << 20) -> None:
        import threading
        from collections import OrderedDict

        self._entries: "OrderedDict[str, tuple[bytes, str]]" = OrderedDict()
        self._total_bytes = 0
        self._max_total_bytes = int(max_total_bytes)
        # FastAPI serves requests from a threadpool, so get/put can race;
        # the lock keeps the OrderedDict LRU bookkeeping consistent (same
        # pattern as _StateCompositionCache in jobs.py).
        self._lock = threading.Lock()

    def get(self, checksum: str) -> tuple[bytes, str] | None:
        with self._lock:
            entry = self._entries.get(checksum)
            if entry is None:
                return None
            self._entries.move_to_end(checksum)
            return entry

    def put(self, checksum: str, payload: bytes, identity_checksum: str) -> None:
        if len(payload) > self._max_total_bytes:
            return
        with self._lock:
            old = self._entries.pop(checksum, None)
            if old is not None:
                self._total_bytes -= len(old[0])
            self._entries[checksum] = (payload, identity_checksum)
            self._total_bytes += len(payload)
            while self._total_bytes > self._max_total_bytes and self._entries:
                _k, evicted = self._entries.popitem(last=False)
                self._total_bytes -= len(evicted[0])


def _identity_cache(request: Request) -> _IdentityPayloadCache:
    cache = getattr(request.app.state, "identity_payload_cache", None)
    if cache is None:
        cache = _IdentityPayloadCache()
        request.app.state.identity_payload_cache = cache
    return cache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _context(request: Request) -> RunnerContext:
    return request.app.state.context


def _store(request: Request) -> Store:
    return request.app.state.context.store


def _raw_body(request: Request) -> bytes:
    return getattr(request.state, "raw_body", b"")


def _idempotency(request: Request) -> tuple[str | None, str]:
    """Return ``(key, fingerprint)``; fingerprint is ``""`` without a key.

    ``If-Match`` is part of the fingerprint: replaying a stored response for
    a request that now carries a different precondition would be wrong (the
    client changed the semantics of the mutation), so it must surface as an
    ``idempotency_key_reused`` conflict instead.
    """
    key = request.headers.get("Idempotency-Key") or None
    if not key:
        return None, ""
    fingerprint = request_fingerprint(
        request.method,
        request.url.path,
        _raw_body(request),
        extra=request.headers.get("If-Match"),
    )
    return key, fingerprint


def _idempotency_precheck(
    store: Store, scope: str, key: str | None, fingerprint: str
) -> FastAPIResponse | None:
    """Read-only replay/mismatch pre-check; the store re-checks atomically."""
    if not key:
        return None
    try:
        stored = store.check_idempotency(scope, key, fingerprint)
    except IdempotencyConflict:
        raise ApiError(
            "idempotency_key_reused",
            "this idempotency key was already used with a different request body",
            status=409,
        ) from None
    if stored is not None:
        return JSONResponse(
            content=stored.body,
            status_code=stored.status_code,
            headers=dict(stored.headers),
        )
    return None


def _conflict(scenario_id: str, error: SceneVersionConflict) -> ApiError:
    return ApiError(
        "scene_version_conflict",
        "The scenario changed after the client's base version.",
        details={
            "current_scene_version": error.current_scene_version,
            "scenario_url": scenario_url(scenario_id),
        },
        status=409,
    )


def _key_reused() -> ApiError:
    return ApiError(
        "idempotency_key_reused",
        "this idempotency key was already used with a different request body",
        status=409,
    )


def _etag_for_version(scene_version: int) -> str:
    return f'"{SCENE_VERSION_ETAG_PREFIX}{int(scene_version)}"'


def _parse_if_match(value: str | None) -> int | None:
    """Parse an ``If-Match: "scene-version-18"`` header into ``18``."""
    if value is None:
        return None
    text = value.strip()
    if text.startswith("W/"):
        text = text[2:]
    text = text.strip('"').strip()
    if text.startswith(SCENE_VERSION_ETAG_PREFIX):
        text = text[len(SCENE_VERSION_ETAG_PREFIX) :]
    try:
        return int(text)
    except ValueError:
        raise ApiError(
            "invalid_request",
            f"If-Match must be an ETag like '\"{SCENE_VERSION_ETAG_PREFIX}17\"', "
            f"got {value!r}",
            field="If-Match",
        ) from None


def _require_scenario(request: Request, scenario_id: str):
    try:
        return _store(request).require_scenario(scenario_id)
    except ScenarioNotFound:
        raise ApiError(
            "scenario_not_found", f"scenario {scenario_id!r} does not exist", status=404
        ) from None


def _tree_validation_error(index: int, error: ValueError) -> ApiError:
    """Convert incremental validation failures into field-specific 422 errors."""
    message = str(error)
    token = message.split(" must", 1)[0].strip().lower()
    field_map = {
        "x_m": "u",
        "y_m": "v",
        "height_m": "height_m",
        "canopy_radius_m": "canopy_diameter_m",
        "canopy diameter": "canopy_diameter_m",
        "trunk_ratio": "trunk_ratio",
        "transmissivity": "transmissivity",
    }
    field = field_map.get(token)
    text = message
    if field == "canopy_diameter_m":
        low, high = TREE_CANOPY_DIAMETER_RANGE_M
        text = f"canopy_diameter_m must be between {low:g} and {high:g}"
    elif field == "height_m":
        low, high = TREE_HEIGHT_RANGE_M
        text = f"height_m must be between {low:g} and {high:g}"
    return ApiError(
        "invalid_tree_geometry",
        text,
        field=f"edits[{index}].tree.{field}" if field else f"edits[{index}].tree",
        status=422,
    )


def _validate_tree(
    index: int, tree: Mapping[str, Any], grid: Mapping[str, float | int]
) -> TreeSpec:
    """Validate one API tree object and convert it to a world ``TreeSpec``."""
    component = tree.get("component_type")
    if component not in TREE_PRESETS:
        raise ApiError(
            "invalid_request",
            f"unknown component_type {component!r}; allowed: {sorted(TREE_PRESETS)}",
            field=f"edits[{index}].tree.component_type",
        )
    for uv_name in ("u", "v"):
        value = tree.get(uv_name)
        if value is None or not 0.0 <= float(value) <= 1.0:
            raise ApiError(
                "invalid_tree_geometry",
                f"{uv_name} must be between 0 and 1",
                field=f"edits[{index}].tree.{uv_name}",
                status=422,
            )
    try:
        spec = api_tree_to_spec(tree, grid)
        validate_tree_preset(spec)
    except ValueError as error:
        # TreeSpec.__post_init__ and validate_tree_preset both raise plain
        # ValueErrors; both are geometry-domain rejections.
        raise _tree_validation_error(index, error) from None
    return spec


def _check_rate_limit(request: Request, scenario_id: str) -> None:
    limiter = getattr(request.app.state, "rate_limiter", None)
    if limiter is not None:
        limiter.check(scenario_id)


def _verify_site_identity(request: Request, record, context) -> None:
    """U-D intake item g: the live site geometry must match the scenario's
    pinned identity (typed 409 otherwise — pixel-identical but
    geographically different data may not ride a live scenario)."""
    try:
        context.store.verify_site_identity(
            record.scenario_id,
            {"site_id": record.site_id, **context.sites.geometry(record.site_id)},
        )
    except SiteIdentityMismatch as error:
        raise ApiError(
            "site_identity_mismatch",
            str(error),
            status=409,
            details={"pinned_identity": error.expected, "current_identity": error.actual},
        ) from None


def _validate_requested(requested, grid: Mapping[str, float | int]) -> None:
    """Reject unsupported ``requested_result`` values with 400s.

    Without this the requested subset would be silently dropped/normalized
    by ``resolve_requested`` and the client would receive different data
    than it asked for.
    """
    if requested is None:
        return
    if requested.variables is not None:
        supported = list(patch_codec.SUPPORTED_VARIABLES)
        unknown = [name for name in requested.variables if name not in supported]
        if unknown:
            raise ApiError(
                "invalid_request",
                f"unknown result variables {unknown}; supported: {supported}",
                field="requested_result.variables",
            )
    if requested.time_indices is not None:
        time_steps = int(grid["time_steps"])
        out_of_range = [
            index for index in requested.time_indices if not 0 <= int(index) < time_steps
        ]
        if out_of_range:
            raise ApiError(
                "invalid_request",
                f"time indices {out_of_range} are outside the site's "
                f"{time_steps} time steps",
                field="requested_result.time_indices",
            )


def _json(response: StoredResponse) -> JSONResponse:
    return JSONResponse(
        content=response.body,
        status_code=response.status_code,
        headers=dict(response.headers),
    )


# ---------------------------------------------------------------------------
# Scenario creation and read
# ---------------------------------------------------------------------------


@router.post("", status_code=201)
def create_scenario(payload: CreateScenarioRequest, request: Request) -> FastAPIResponse:
    context = _context(request)
    sites = context.sites

    # Replay check first, mirroring the other mutation endpoints.
    key, fingerprint = _idempotency(request)
    replay = _idempotency_precheck(context.store, "_create", key, fingerprint)
    if replay is not None:
        return replay

    site_ids = sites.site_ids()
    if payload.site_id not in site_ids:
        raise ApiError(
            "invalid_request",
            f"unknown site_id {payload.site_id!r}; registered sites: {list(site_ids)}",
            field="site_id",
        )
    site_limiter = getattr(request.app.state, "site_limiter", None)
    if site_limiter is not None and not site_limiter.allow():
        raise ApiError(
            "site_limit_exceeded",
            "the maximum number of active scenarios for this server has been reached",
            status=403,
        )

    config = sites.config(payload.site_id)
    has_baseline = (config.cache_dir / "baseline_results" / "metadata.json").is_file()

    def baseline_publisher(scenario_id: str, scene_version: int):
        try:
            return materialize_baseline_result(
                sites, scenario_id, payload.site_id, scene_version
            )
        except ValueError as error:
            # Shape mismatch against the site manifest: the deployed cache
            # does not match the site (contract: cache_version_mismatch).
            raise ApiError(
                "cache_version_mismatch", str(error), status=409
            ) from None

    def response_builder(record, baseline_job_id) -> StoredResponse:
        body = _scenario_body(record)
        headers = {"Location": scenario_url(record.scenario_id)}
        if baseline_job_id is not None:
            body["job_id"] = baseline_job_id
            body["status_url"] = f"/api/v1/jobs/{baseline_job_id}"
            body["coalescing_window_ms"] = request.app.state.coalescing_window_ms
        return StoredResponse(201, body, headers)

    try:
        record, baseline_job_id = context.store.create_scenario(
            site_id=payload.site_id,
            name=payload.name,
            idempotency_key=key,
            request_fingerprint=fingerprint or None,
            max_scenarios=getattr(request.app.state, "max_scenarios", None),
            site_identity={
                "site_id": payload.site_id,
                **context.sites.geometry(payload.site_id),
            },
            baseline_publisher=baseline_publisher if has_baseline else None,
            response_builder=response_builder,
        )
    except ScenarioQuotaExceeded as error:
        raise ApiError(
            "site_limit_exceeded", str(error), status=403
        ) from None
    except IdempotentReplay as replayed:
        return _json(replayed.response)
    except IdempotencyConflict:
        raise _key_reused() from None
    if baseline_job_id is not None:
        # No stored baseline outputs: schedule a baseline full-solve job.
        request.app.state.runner.submit(baseline_job_id)
    return _json(response_builder(record, baseline_job_id))


def _scenario_body(record) -> dict[str, Any]:
    return {
        "scenario_id": record.scenario_id,
        "site_id": record.site_id,
        "name": record.name,
        "scene_version": record.scene_version,
        "exact_result_version": record.exact_result_version,
        "status": record.status,
        "trees": [],
        "result_manifest_url": result_manifest_url(
            record.scenario_id, record.exact_result_version
        ),
    }


@router.get("/{scenario_id}")
def get_scenario(scenario_id: str, request: Request) -> JSONResponse:
    store = _store(request)
    record = _require_scenario(request, scenario_id)
    context = _context(request)
    active = store.latest_job_for_scenario(scenario_id)
    active_job_id = (
        active.job_id if active is not None and active.status in ("queued", "running") else None
    )
    site_manifest = context.sites.manifest(record.site_id)
    body = _scenario_body(record)
    body["trees"] = store.list_trees(scenario_id)
    body["created_at"] = record.created_at
    body["updated_at"] = record.updated_at
    body["active_job_id"] = active_job_id
    body["model_scope"] = {
        "model_version": site_manifest.model_version,
        "site_cache_version": context.sites.site_cache_version(record.site_id),
        "limitations": list(default_limitations),
    }
    # The pinned site's grid placement: realtime vegetation ops carry world
    # metres, and no other surface exposes the origin — without it a client
    # can only build a span-local frame whose ops land outside the raster.
    body["site_geometry"] = dict(context.sites.geometry(record.site_id))
    return JSONResponse(content=body)


# ---------------------------------------------------------------------------
# Edits
# ---------------------------------------------------------------------------


@router.post("/{scenario_id}/edits", status_code=202)
def commit_edits(payload: EditRequest, scenario_id: str, request: Request) -> FastAPIResponse:
    record = _require_scenario(request, scenario_id)
    context = _context(request)
    store = context.store
    grid = context.sites.geometry(record.site_id)

    # Replay check comes first: a replayed request body refers to scene state
    # that has since advanced, so semantic validation would (wrongly) reject
    # it as stale/duplicated before we can return the stored response.
    key, fingerprint = _idempotency(request)
    replay = _idempotency_precheck(store, scenario_id, key, fingerprint)
    if replay is not None:
        return replay

    if_match = _parse_if_match(request.headers.get("If-Match"))
    if if_match is not None and if_match != payload.base_scene_version:
        raise ApiError(
            "invalid_request",
            "If-Match and base_scene_version disagree",
            field="base_scene_version",
        )
    # Site-identity pin (item g) BEFORE any semantic work: a swapped site
    # cache is refused here, whatever tree objects the request carries.
    _verify_site_identity(request, record, context)
    _validate_requested(payload.requested_result, grid)

    trees = {tree["tree_id"]: tree for tree in store.list_trees(scenario_id)}
    max_trees = getattr(request.app.state, "max_trees_per_scenario", None)
    adds = sum(1 for item in payload.edits if item.operation == "add")
    if max_trees is not None and len(trees) + adds > int(max_trees):
        raise ApiError(
            "site_limit_exceeded",
            f"this scenario already holds {len(trees)} trees; the per-scenario "
            f"cap is {int(max_trees)}",
            status=403,
        )
    applied: list[dict[str, Any]] = []
    for index, item in enumerate(payload.edits):
        if item.operation == "delete":
            if item.tree_id is None:
                raise ApiError(
                    "invalid_request",
                    "delete edits require tree_id",
                    field=f"edits[{index}].tree_id",
                )
            if item.tree_id not in trees:
                raise ApiError(
                    "invalid_request",
                    f"unknown tree_id {item.tree_id!r}",
                    field=f"edits[{index}].tree_id",
                )
            applied.append(
                {
                    "operation": "delete",
                    "tree_id": item.tree_id,
                    "old_tree": trees.pop(item.tree_id),
                    "new_tree": None,
                }
            )
            continue
        if item.tree is None:
            raise ApiError(
                "invalid_request",
                f"{item.operation} edits require a complete replacement tree object",
                field=f"edits[{index}].tree",
            )
        tree = item.tree.model_dump()
        _validate_tree(index, tree, grid)
        if item.operation == "add":
            if tree["tree_id"] in trees:
                raise ApiError(
                    "invalid_request",
                    f"tree_id {tree['tree_id']!r} already exists in this scenario",
                    field=f"edits[{index}].tree.tree_id",
                )
            applied.append(
                {
                    "operation": "add",
                    "tree_id": tree["tree_id"],
                    "old_tree": None,
                    "new_tree": tree,
                }
            )
        else:  # move / update carry a complete replacement object
            if tree["tree_id"] not in trees:
                raise ApiError(
                    "invalid_request",
                    f"unknown tree_id {tree['tree_id']!r}",
                    field=f"edits[{index}].tree.tree_id",
                )
            applied.append(
                {
                    "operation": item.operation,
                    "tree_id": tree["tree_id"],
                    "old_tree": trees[tree["tree_id"]],
                    "new_tree": tree,
                }
            )
        trees[tree["tree_id"]] = tree

    _check_rate_limit(request, scenario_id)

    requested = (
        payload.requested_result.model_dump(exclude_none=True)
        if payload.requested_result is not None
        else {}
    )

    def response_builder(record, job_id) -> StoredResponse:
        return StoredResponse(
            202,
            {
                "scenario_id": record.scenario_id,
                "scene_version": record.scene_version,
                "exact_result_version": record.exact_result_version,
                "status": record.status,
                "job_id": job_id,
                "coalescing_window_ms": request.app.state.coalescing_window_ms,
                "status_url": f"/api/v1/jobs/{job_id}",
            },
            headers={"ETag": _etag_for_version(record.scene_version)},
        )

    try:
        record, job_id = store.commit_edits(
            scenario_id,
            base_scene_version=payload.base_scene_version,
            applied_edits=applied,
            requested=requested,
            idempotency_key=key,
            request_fingerprint=fingerprint or None,
            response_builder=response_builder,
        )
    except IdempotentReplay as replayed:
        return _json(replayed.response)
    except IdempotencyConflict:
        raise _key_reused() from None
    except SceneVersionConflict as error:
        raise _conflict(scenario_id, error) from None
    request.app.state.runner.submit(job_id)
    return _json(response_builder(record, job_id))


# ---------------------------------------------------------------------------
# Universal (family) edits
# ---------------------------------------------------------------------------


@router.post("/{scenario_id}/edits/universal", status_code=202)
def commit_universal_edits(
    payload: UniversalEditRequest, scenario_id: str, request: Request
) -> FastAPIResponse:
    """Commit a batch of family edits through the universal transport.

    Same discipline as the tree endpoint — idempotent replay first,
    If-Match/base agreement, site-identity pin, typed refusals, one
    version bump, one job — with validation DELEGATED to the adapter
    registry (the capability document's validators are the single source
    of truth; this route never restates a domain or fence). A batch whose
    single item is an ``output_view`` operation is answered view-only:
    no event, no job, zero scientific jobs.
    """
    from solweig_gpu.server import universal
    from solweig_gpu.server.jobs import resolve_requested
    from solweig_gpu.server.models import ViewRequest
    from solweig_gpu.server.routes_capabilities import (
        _check_view_rate_limit,
        _document,
        create_view_payload,
    )

    record = _require_scenario(request, scenario_id)
    context = _context(request)
    store = context.store
    grid = context.sites.geometry(record.site_id)

    key, fingerprint = _idempotency(request)
    replay = _idempotency_precheck(store, scenario_id, key, fingerprint)
    if replay is not None:
        return replay

    if_match = _parse_if_match(request.headers.get("If-Match"))
    if if_match is not None and if_match != payload.base_scene_version:
        raise ApiError(
            "invalid_request",
            "If-Match and base_scene_version disagree",
            field="base_scene_version",
        )
    _verify_site_identity(request, record, context)
    _validate_requested(payload.requested_result, grid)

    # View-only batches: delegate to the published-result view machinery
    # (UEDIT-007 zero-jobs guarantee holds by construction — no store
    # mutation, no submit, anywhere in this branch).
    view_positions = [
        index
        for index, item in enumerate(payload.edits)
        if item.adapter == universal.VIEW_ADAPTER_ID
    ]
    if view_positions:
        if len(payload.edits) != 1:
            raise ApiError(
                "invalid_family_batch",
                "an output_view item is answered view-only (zero jobs) and "
                "must be the only item in a universal batch; commit family "
                "edits in a separate request",
                field="edits",
            )
        item = payload.edits[0]
        view_operations = _document(request)["view_only"]["operations"]
        if item.operation not in view_operations:
            raise ApiError(
                "invalid_request",
                f"unknown view operation {item.operation!r}; supported: "
                f"{view_operations}",
                field="edits[0].operation",
            )
        layer = item.target if isinstance(item.target, str) else item.values.get("layer")
        # u-e2 M2: strict JSON-integer parse of the view timestep. The
        # previous ``int(... or 0)`` coerced "07"->7 / True->1 and treated
        # a missing key as 0 — a served plane the client never asked for.
        # (The item-level field is already schema-typed by the model.)
        if item.time_index is not None:
            time_index = item.time_index
        elif "time_index" in item.values:
            time_index = universal.json_time_index(
                item.values["time_index"], field="edits[0].values.time_index"
            )
        else:
            time_index = 0
        view = ViewRequest(
            operation=item.operation,
            layer=str(layer or ""),
            time_index=time_index,
            compare_layer=item.values.get("compare_layer"),
        )
        _check_view_rate_limit(request, scenario_id)
        return JSONResponse(content=create_view_payload(request, record, view))

    requested_dump = (
        payload.requested_result.model_dump(exclude_none=True)
        if payload.requested_result is not None
        else {}
    )
    _, variables = resolve_requested(requested_dump, grid)
    requested_times = (
        tuple(payload.requested_result.time_indices)
        if payload.requested_result is not None and payload.requested_result.time_indices
        else None
    )
    site_context = universal.site_context(
        request.app.state, context.sites, record.site_id, record.scene_version
    )

    family_edits: list[dict[str, Any]] = []
    for index, item in enumerate(payload.edits):
        universal.check_transportable(item, index)
        command = universal.command_from_item(
            item,
            scenario_id=scenario_id,
            scene_revision=record.scene_version,
            edit_id=f"u{index}",
            index=index,
            requested_outputs=variables,
            requested_times=requested_times,
        )
        if command is None:  # pragma: no cover - view items returned above
            continue
        universal.validate_command(command, site_context, index=index)
        family_edits.append(
            {
                "operation": item.operation,
                "payload": {
                    "adapter": item.adapter,
                    "operation": item.operation,
                    "values": item.values,
                    "target": item.target,
                    "time_index": item.time_index,
                    "old_values": item.old_values,
                },
            }
        )

    _check_rate_limit(request, scenario_id)

    def response_builder(record, job_id) -> StoredResponse:
        return StoredResponse(
            202,
            {
                "scenario_id": record.scenario_id,
                "scene_version": record.scene_version,
                "exact_result_version": record.exact_result_version,
                "status": record.status,
                "job_id": job_id,
                "transport": "universal-edits-v1",
                "coalescing_window_ms": request.app.state.coalescing_window_ms,
                "status_url": f"/api/v1/jobs/{job_id}",
            },
            headers={"ETag": _etag_for_version(record.scene_version)},
        )

    try:
        record, job_id = store.commit_universal_edits(
            scenario_id,
            base_scene_version=payload.base_scene_version,
            family_edits=family_edits,
            requested=requested_dump,
            idempotency_key=key,
            request_fingerprint=fingerprint or None,
            response_builder=response_builder,
        )
    except IdempotentReplay as replayed:
        return _json(replayed.response)
    except IdempotencyConflict:
        raise _key_reused() from None
    except SceneVersionConflict as error:
        raise _conflict(scenario_id, error) from None
    request.app.state.runner.submit(job_id)
    return _json(response_builder(record, job_id))


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


@router.post("/{scenario_id}/reset", status_code=200)
def reset_scenario(
    scenario_id: str, request: Request, payload: ResetRequest | None = None
) -> FastAPIResponse:
    record = _require_scenario(request, scenario_id)
    context = _context(request)
    store = context.store
    sites = context.sites

    # Replay check first (see commit_edits): a replayed reset may carry a
    # now-stale If-Match that must not shadow the stored response.
    key, fingerprint = _idempotency(request)
    replay = _idempotency_precheck(store, scenario_id, key, fingerprint)
    if replay is not None:
        return replay

    if_match = _parse_if_match(request.headers.get("If-Match"))
    base = payload.base_scene_version if payload is not None else None
    if if_match is not None and base is not None and if_match != base:
        raise ApiError(
            "invalid_request",
            "If-Match and base_scene_version disagree",
            field="base_scene_version",
        )
    effective_base = if_match if if_match is not None else base

    # Site-identity pin (item g): resetting under swapped site data would
    # "restore" a baseline from a different place on Earth.
    _verify_site_identity(request, record, context)

    _check_rate_limit(request, scenario_id)

    def baseline_publisher(scenario_id_: str, scene_version: int):
        try:
            return materialize_baseline_result(
                sites, scenario_id_, record.site_id, scene_version
            )
        except FileNotFoundError:
            # Version 0 is the baseline by construction (data model invariant).
            baseline = store.get_result(scenario_id_, 0)
            if baseline is None:
                raise ApiError(
                    "cache_version_mismatch",
                    "the site cache has no baseline result to reset to",
                    status=409,
                ) from None
            manifest = dict(baseline.manifest)
            manifest["scene_version"] = int(scene_version)
            manifest["payload_url"] = result_payload_url(scenario_id_, scene_version)
            return manifest, baseline.payload_bytes()
        except ValueError as error:
            # Baseline arrays do not match the site manifest geometry.
            raise ApiError("cache_version_mismatch", str(error), status=409) from None

    def response_builder(record_) -> StoredResponse:
        return StoredResponse(
            200,
            {
                "scenario_id": record_.scenario_id,
                "scene_version": record_.scene_version,
                "exact_result_version": record_.exact_result_version,
                "status": record_.status,
                "trees": [],
                "result_manifest_url": result_manifest_url(
                    record_.scenario_id, record_.exact_result_version
                ),
            },
            headers={"ETag": _etag_for_version(record_.scene_version)},
        )

    try:
        record = store.reset_scenario(
            scenario_id,
            base_scene_version=effective_base,
            baseline_publisher=baseline_publisher,
            idempotency_key=key,
            request_fingerprint=fingerprint or None,
            response_builder=response_builder,
        )
    except IdempotentReplay as replayed:
        return _json(replayed.response)
    except IdempotencyConflict:
        raise _key_reused() from None
    except SceneVersionConflict as error:
        raise _conflict(scenario_id, error) from None
    return _json(response_builder(record))


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@router.get("/{scenario_id}/results/{scene_version}")
def get_result_manifest(
    scenario_id: str, scene_version: int, request: Request
) -> JSONResponse:
    _require_scenario(request, scenario_id)
    record = _store(request).get_result(scenario_id, scene_version)
    if record is None:
        raise ApiError(
            "result_not_ready",
            f"no exact result for scene version {scene_version} of scenario "
            f"{scenario_id!r} yet",
            status=404,
        )
    return JSONResponse(content=record.manifest)


@router.get("/{scenario_id}/results/{scene_version}/payload")
def get_result_payload(
    scenario_id: str, scene_version: int, request: Request
) -> FastAPIResponse:
    _require_scenario(request, scenario_id)
    record = _store(request).get_result(scenario_id, scene_version)
    if record is None:
        raise ApiError(
            "result_not_ready",
            f"no exact result for scene version {scene_version} of scenario "
            f"{scenario_id!r} yet",
            status=404,
        )
    payload = record.payload_bytes()
    # Content negotiation per the contract's Binary payload section: zstd is
    # the default; browsers without a zstd DecompressionStream request the
    # identity encoding and get the same byte order uncompressed. ETag and
    # checksum are computed over the bytes actually served, so the checksum
    # contract holds per encoding (the durable manifest checksum stays the
    # zstd one).
    media_type = patch_codec.select_payload_media_type(
        request.headers.get("accept")
    )
    if media_type is None:
        raise ApiError(
            "not_acceptable",
            "client does not accept either payload encoding "
            f"({patch_codec.PATCH_MEDIA_TYPE} or "
            f"{patch_codec.IDENTITY_MEDIA_TYPE})",
            status=406,
        )
    checksum = record.checksum
    if media_type == patch_codec.IDENTITY_MEDIA_TYPE:
        cached = _identity_cache(request).get(record.checksum)
        if cached is not None:
            payload, checksum = cached
        else:
            payload = patch_codec.decompress_payload(payload, record.manifest)
            checksum = patch_codec.checksum_payload(payload)
            _identity_cache(request).put(record.checksum, payload, checksum)
    etag = patch_codec.payload_etag(checksum)
    if request.headers.get("If-None-Match") == etag:
        return FastAPIResponse(status_code=304, headers={"ETag": etag})
    return FastAPIResponse(
        content=payload,
        media_type=media_type,
        headers={
            "ETag": etag,
            "X-SOLWEIG-Scene-Version": str(record.scene_version),
            "X-SOLWEIG-Schema-Version": str(patch_codec.PATCH_SCHEMA_VERSION),
            "X-SOLWEIG-Checksum": checksum,
            "X-SOLWEIG-Exact": "true" if record.exact else "false",
        },
    )


# ---------------------------------------------------------------------------
# SSE event stream (contract: optional surface)
# ---------------------------------------------------------------------------

SSE_POLL_INTERVAL_S = 0.2


def _sse_event(
    event: str, data: Mapping[str, Any], *, event_id: str | None = None
) -> str:
    lines: list[str] = []
    if event_id is not None:
        lines.append(f"id: {event_id}")
    lines.append(f"event: {event}")
    lines.append("data: " + json.dumps(data, separators=(",", ":")))
    return "\n".join(lines) + "\n\n"


@router.get("/{scenario_id}/events")
async def stream_events(
    scenario_id: str,
    request: Request,
    timeout_seconds: float = 300.0,
    wait_for_job_seconds: float = 2.0,
) -> StreamingResponse:
    """Server-sent events for a scenario's job pipeline.

    Emits ``job-progress`` whenever the latest job's status/stage/progress
    changes while it is queued or running, then one ``result-ready`` when it
    completes with a published result (or a final ``job-progress`` naming the
    terminal status for superseded/failed/cancelled) and closes. There is no
    push channel — this is a bounded poll over the same durable state the
    job endpoints serve, so a client that prefers polling loses nothing.

    Connecting when the latest job is already terminal replays that outcome
    immediately, which makes a late subscriber's ``result-ready`` retrieval
    deterministic. ``wait_for_job_seconds`` bounds how long an idle scenario
    stays open waiting for a first job before closing (the client simply
    reconnects).
    """
    _require_scenario(request, scenario_id)
    accept = (request.headers.get("accept") or "").lower()
    if accept and "text/event-stream" not in accept and "*/*" not in accept:
        raise ApiError(
            "not_acceptable",
            "the events endpoint serves text/event-stream",
            status=406,
        )
    store = _store(request)

    async def generator():
        deadline = time.monotonic() + max(timeout_seconds, 0.1)
        job_grace = time.monotonic() + max(wait_for_job_seconds, 0.0)
        last_signature: tuple[Any, ...] | None = None
        seq = 0
        yield ": connected\n\n"  # SSE comment line: not an event
        while time.monotonic() < deadline:
            job = store.latest_job_for_scenario(scenario_id)
            if job is not None:
                progress = job.progress or {}
                if job.status in ("queued", "running"):
                    signature = (
                        job.job_id,
                        job.status,
                        job.stage,
                        progress.get("completed_time_steps"),
                        progress.get("total_time_steps"),
                        job.mode,
                    )
                    if signature != last_signature:
                        last_signature = signature
                        seq += 1
                        yield _sse_event(
                            "job-progress",
                            {
                                "job_id": job.job_id,
                                "scene_version": job.target_scene_version,
                                "status": job.status,
                                "stage": job.stage,
                                "completed": progress.get(
                                    "completed_time_steps", 0
                                ),
                                "total": progress.get("total_time_steps", 1),
                            },
                            event_id=f"{job.job_id}:{seq}",
                        )
                    await asyncio.sleep(SSE_POLL_INTERVAL_S)
                    continue
                # Terminal outcome: report once and close the stream.
                if job.status == "complete":
                    result_version = (
                        job.result_scene_version
                        if job.result_scene_version is not None
                        else job.target_scene_version
                    )
                    seq += 1
                    yield _sse_event(
                        "result-ready",
                        {
                            "scene_version": int(result_version),
                            "result_manifest_url": result_manifest_url(
                                scenario_id, int(result_version)
                            ),
                            "job_id": job.job_id,
                        },
                        event_id=f"{job.job_id}:{seq}",
                    )
                else:
                    seq += 1
                    yield _sse_event(
                        "job-progress",
                        {
                            "job_id": job.job_id,
                            "scene_version": job.target_scene_version,
                            "status": job.status,
                            "stage": job.stage,
                            "completed": progress.get(
                                "completed_time_steps", 0
                            ),
                            "total": progress.get("total_time_steps", 1),
                        },
                        event_id=f"{job.job_id}:{seq}",
                    )
                return
            if time.monotonic() > job_grace:
                # Idle scenario and no job arrived within the grace window.
                return
            await asyncio.sleep(SSE_POLL_INTERVAL_S)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Exports — recorded intent only; export generation is a v1 STUB
# ---------------------------------------------------------------------------


@router.post("/{scenario_id}/exports", status_code=202)
def create_export(
    payload: ExportRequest, scenario_id: str, request: Request
) -> JSONResponse:
    """Record an export request and return 202 (v1 stub: intent only)."""
    _require_scenario(request, scenario_id)
    store = _store(request)
    if store.get_result(scenario_id, payload.scene_version) is None:
        raise ApiError(
            "result_not_ready",
            f"cannot export scene version {payload.scene_version}: no result published",
        )
    export_id = store.record_export(
        scenario_id,
        scene_version=payload.scene_version,
        format=payload.format,
        variables=payload.variables,
        time_indices=payload.time_indices,
    )
    return JSONResponse(
        content={
            "export_id": export_id,
            "scenario_id": scenario_id,
            "scene_version": payload.scene_version,
            "format": payload.format,
            "status": "accepted",
            "note": "Export intent is recorded; export generation is not "
            "implemented in v1 (stub).",
        },
        status_code=202,
    )
