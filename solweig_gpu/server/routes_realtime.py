# SPDX-License-Identifier: GPL-3.0-only
"""Realtime collaborative operation endpoints (R1 operation plane).

``POST /api/v1/workspaces/{scenario_id}/operations`` is the multi-writer
acceptance route: 1..128 idempotent operations per request, each durably
appended (one store transaction per batch) with a per-workspace contiguous
``server_sequence`` and assignment to the workspace's open epoch.
``base_revision`` is ADVISORY — divergence is logged and flagged, never a
rejection (``collaborative_state.md``: the server assigns final ordering;
no equality gate). Idempotency is per ``operation_id`` (globally unique),
not per request header: a retried operation replays its existing record, a
reused id with different content is a typed 409.

``GET .../operations/{operation_id}`` serves the idempotent retry lookup;
``GET .../epochs`` serves the workspace's durable epoch rows for
collaborator catch-up and R2 scheduler introspection.

Guards reused from the existing surface: the per-scenario mutation rate
budget and the site-identity pin (U-D item g). Validation at this layer is
transport-shaped only (frozen family/verb vocabularies, non-empty payload);
family payload validation stays with the executor bridge.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from solweig_gpu.server.models import ApiError, RealtimeOperationsRequest
from solweig_gpu.server.realtime import operations as rt
from solweig_gpu.server.realtime import selected_time
from solweig_gpu.server.routes_scenarios import (
    _check_rate_limit,
    _context,
    _require_scenario,
    _verify_site_identity,
)
from solweig_gpu.server.store import OperationFingerprintConflict, Store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/workspaces", tags=["realtime"])


def _now_utc() -> str:
    """Wall-clock ISO stamp, the store's ``_now_utc`` format exactly."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _store(request: Request) -> Store:
    return request.app.state.context.store


@router.post("/{scenario_id}/operations")
def post_operations(
    payload: RealtimeOperationsRequest, scenario_id: str, request: Request
) -> JSONResponse:
    """Durably accept a batch of collaborative operations (multi-writer)."""
    # t_receive (service_level_contract.md): stamped at route entry, before
    # any validation, so it measures the transport — not the guards.
    received_at = _now_utc()

    record = _require_scenario(request, scenario_id)
    context = _context(request)
    # Site-identity pin (U-D item g): a swapped site cache may not ride a
    # live workspace, whatever operations the request carries.
    _verify_site_identity(request, record, context)

    items = [item.model_dump() for item in payload.operations]
    for index, item in enumerate(items):
        rt.validate_operation_shape(item, index=index)

    _check_rate_limit(request, scenario_id)

    # r2b admission gate (the one sanctioned hook into this module; the
    # policy lives in realtime/admission.py): predicted fast cost + the
    # 1,000 ms envelope vs reserved fast capacity. Raises a typed
    # AdmissionRejected (429/503 + retry metadata) BEFORE the durable
    # append — outside-envelope work is rejected, never accepted-then-
    # dropped. Idempotent RETRIES of already-accepted operations may be
    # throttled the same way; the durable record stays the truth.
    admission = getattr(request.app.state, "fast_admission", None)
    if admission is not None:
        admission.check(scenario_id, items, received_at=received_at)

    try:
        accepted = _store(request).append_operations(
            scenario_id,
            [
                {**item, "actor_id": payload.actor_id, "received_at": received_at}
                for item in items
            ],
        )
    except OperationFingerprintConflict as error:
        raise ApiError(
            "operation_id_reused",
            str(error),
            details={"operation_id": error.operation_id},
            status=409,
        ) from None

    # t_accept: the durable commit is done; refresh the revision counters
    # from the row that just became durable.
    accepted_at = _now_utc()
    scenario = _store(request).require_scenario(scenario_id)
    # workspace_revision IS scene_version (migration v7 note); the
    # collaborative_state.md revision model rides the ack so clients can
    # reconcile without a second request.
    workspace_revision = scenario.scene_version

    accepted_bodies: list[dict[str, Any]] = []
    for item, operation in zip(items, accepted, strict=True):
        body: dict[str, Any] = {
            "operation_id": operation.operation_id,
            "server_sequence": operation.server_sequence,
            "epoch_id": operation.epoch_id,
            "duplicate": operation.duplicate,
        }
        # Advisory divergence: recorded, logged, NEVER a rejection.
        base_revision = item.get("base_revision")
        if base_revision is not None and base_revision != workspace_revision:
            body["base_divergent"] = True
            logger.info(
                "realtime operation %s (actor %s) carries advisory "
                "base_revision %s != workspace_revision %s; accepted anyway",
                operation.operation_id,
                payload.actor_id,
                base_revision,
                workspace_revision,
            )
        accepted_bodies.append(body)

    return JSONResponse(
        content={
            "workspace_id": scenario_id,
            "actor_id": payload.actor_id,
            "received_at": received_at,
            "accepted_at": accepted_at,
            "accepted": accepted_bodies,
            # The merged client reads ``acks ?? operations``; without this
            # alias every submit would stall on an undefined field.
            "acks": accepted_bodies,
            "workspace_revision": workspace_revision,
            "fast_revision": scenario.fast_revision,
            "exact_revision": scenario.exact_result_version,
            "exact_base_revision": scenario.exact_base_revision,
        }
    )


@router.get("/{scenario_id}/operations/{operation_id}")
def get_operation(
    scenario_id: str, operation_id: str, request: Request
) -> JSONResponse:
    """Idempotent retry lookup for one accepted operation."""
    _require_scenario(request, scenario_id)
    record = _store(request).get_operation(scenario_id, operation_id)
    if record is None:
        raise ApiError(
            "operation_not_found",
            f"no accepted operation {operation_id!r} in workspace {scenario_id!r}",
            status=404,
        )
    return JSONResponse(content=rt.operation_body(record))


@router.get("/{scenario_id}/operations")
def get_operations(
    scenario_id: str,
    request: Request,
    since_server_sequence: int = 0,
) -> JSONResponse:
    """Catch-up read: every operation past the client's high-water mark.

    The client's ``#catchUp`` (realtime_client.mjs) reads exactly
    ``body.operations`` plus the three revision fields — its SSE queue may
    have dropped frames (lagged subscriber), and this endpoint is the
    recovery path that makes SSE loss safe.
    """
    record = _require_scenario(request, scenario_id)
    operations = _store(request).operations_since(
        scenario_id, int(since_server_sequence)
    )
    scenario = _store(request).require_scenario(scenario_id)
    return JSONResponse(
        content={
            "workspace_id": scenario_id,
            "since_server_sequence": int(since_server_sequence),
            # Growth observability: how much history remains and how far back
            # it reaches. Additive fields — the payload's ledger table is
            # append-only today, but a future prune floor needs these to be
            # visible before any truncation contract ships.
            "operation_count": len(operations),
            "oldest_server_sequence": (
                int(operations[0].server_sequence) if operations else None
            ),
            "operations": [rt.operation_body(op) for op in operations],
            "workspace_revision": scenario.scene_version,
            "fast_revision": scenario.fast_revision,
            "exact_revision": scenario.exact_result_version,
        }
    )


@router.get("/{scenario_id}/events")
async def get_events(
    scenario_id: str,
    request: Request,
    time_indices: str | None = None,
    actor_id: str | None = None,
) -> StreamingResponse:
    """SSE stream of the workspace's realtime events.

    First frame is a state-snapshot ``canonical_revision`` (current
    revisions, EMPTY operations — history belongs to the catch-up GET, old
    operations are never replayed on the stream). Live frames: named SSE
    events (``event:`` lines; the client uses ``addEventListener``) for
    ``canonical_revision`` / ``fast_revision`` / ``exact_revision`` /
    ``heartbeat``.

    ``?actor_id=`` (best-effort presence): the hub registers the identity
    as a live subscriber and the periodic heartbeat carries the
    workspace's live-actor ``roster`` to everyone. Never authenticated —
    a malformed value degrades to an anonymous subscriber rather than
    failing the stream (presence is decoration; the revision frames are
    the contract).

    T15 (``?time_indices=1,2``, flag-gated): a selected-time stream. The
    subscriber additionally receives a first ``selected_time`` frame
    stating which of its requested times the server can serve bitwise-
    correctly right now, and its demand shrinks the epoch window toward
    the contract's latency floor until the stream ends. With the flag
    unset the parameter is a typed 400 naming the flag (never silently
    ignored — a client streaming selected times believes it is covered).
    """
    _require_scenario(request, scenario_id)
    hub = getattr(request.app.state, "broadcast_hub", None)
    if hub is None:  # pragma: no cover - every create_app wires a hub
        raise ApiError("realtime_disabled", "broadcast hub unavailable", status=503)
    requested_times: tuple[int, ...] = ()
    demand = None
    if time_indices is not None:
        if not selected_time.enabled():
            raise ApiError(
                "selected_time_streaming_disabled",
                f"time_indices streaming parameter requires "
                f"{selected_time.ENV_FLAG_NAME}=1 (selected-time streaming "
                "is disabled on this deployment)",
                status=400,
            )
        record = _require_scenario(request, scenario_id)
        grid = _context(request).sites.geometry(record.site_id)
        time_steps = int(grid["time_steps"])
        cleaned: list[int] = []
        for part in str(time_indices).split(","):
            part = part.strip()
            if not part:
                continue
            try:
                index = int(part)
            except ValueError:
                raise ApiError(
                    "invalid_request",
                    f"time_indices entry {part!r} is not an integer",
                    status=400,
                ) from None
            if not 0 <= index < time_steps:
                raise ApiError(
                    "invalid_request",
                    f"time index {index} is outside the site's {time_steps} "
                    "time steps",
                    status=400,
                )
            if index not in cleaned:
                cleaned.append(index)
        if not cleaned:
            raise ApiError(
                "invalid_request",
                "time_indices produced no valid site time steps",
                status=400,
            )
        requested_times = tuple(cleaned)
        demand = getattr(request.app.state, "selected_time_demand", None)
    scenario = _store(request).require_scenario(scenario_id)
    snapshot = {
        "workspace_id": scenario_id,
        "workspace_revision": scenario.scene_version,
        "operations": [],
        "fast_revision": scenario.fast_revision,
        "exact_revision": scenario.exact_result_version,
        "exact_base_revision": scenario.exact_base_revision,
        "snapshot": True,
    }
    # Presence identity: sanitize-or-degrade (see docstring) — length-capped
    # to a bounded roster display size, characters limited to the wire's
    # actor-id shape so a hostile string can't ride into every heartbeat.
    cleaned_actor_id: str | None = None
    if actor_id:
        candidate = str(actor_id).strip()[:128]
        if candidate and all(ch.isalnum() or ch in "_-:" for ch in candidate):
            cleaned_actor_id = candidate
    subscription = hub.subscribe(
        scenario_id, snapshot=snapshot, actor_id=cleaned_actor_id
    )
    # Subscribe TOCTOU (r1-epochs-review L1): an epoch may have closed
    # between the scenario read above and registration, leaving the
    # snapshot stale. Re-read and push a corrective snapshot frame so the
    # subscriber never strands on the old revision — revision-sync frames
    # arrive in non-decreasing revision order, live frames after both.
    current = _store(request).require_scenario(scenario_id)
    if current.scene_version > scenario.scene_version:
        subscription.push_snapshot(
            {
                "workspace_id": scenario_id,
                "workspace_revision": current.scene_version,
                "operations": [],
                "fast_revision": current.fast_revision,
                "exact_revision": current.exact_result_version,
                "exact_base_revision": current.exact_base_revision,
                "snapshot": True,
            }
        )
    if requested_times:
        # The coverage snapshot rides the SAME privileged push as the
        # corrective revision frame: it is the stream's contract statement
        # and must never be the frame a lagging subscriber drops.
        report = selected_time.coverage_report(
            _store(request), scenario_id, ("utci", "tmrt"), int(grid["time_steps"])
        )
        subscription.push_selected_time(
            selected_time.initial_frame(report, requested_times)
        )
        if demand is not None:
            demand.register(scenario_id)

    async def stream() -> AsyncIterator[str]:
        try:
            async for frame in subscription.events():
                yield frame
        finally:  # pragma: no cover - generator exit also unsubscribes
            subscription.close()
            if demand is not None and requested_times:
                demand.release(scenario_id)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/{scenario_id}/epochs")
def get_epochs(scenario_id: str, request: Request) -> JSONResponse:
    """Durable epoch rows for the workspace (open and historical)."""
    _require_scenario(request, scenario_id)
    epochs = _store(request).epoch_records(scenario_id)
    return JSONResponse(
        content={
            "workspace_id": scenario_id,
            "epochs": [rt.epoch_body(epoch) for epoch in epochs],
        }
    )


@router.get("/{scenario_id}/time-coverage")
def get_time_coverage(
    scenario_id: str, request: Request, variables: str | None = None
) -> JSONResponse:
    """Per-variable served-coverage report (T15, flag-gated).

    Answers, at the newest durable revision, exactly which times the
    server can serve bitwise-correctly for each variable: everything
    except UNHEALED request-cut gaps (a publication produced under a
    client's requested time subset whose uncovered times were computed
    then discarded). Also carries the liveness introspection the exit
    contract names: the newest state's age and the workspace's
    supersession count. With the flag unset the endpoint is a typed 404
    naming the flag — disabled features are discovered, not guessed.
    """
    if not selected_time.enabled():
        raise ApiError(
            "selected_time_streaming_disabled",
            f"selected-time coverage reporting requires "
            f"{selected_time.ENV_FLAG_NAME}=1 (selected-time streaming is "
            "disabled on this deployment)",
            status=404,
        )
    record = _require_scenario(request, scenario_id)
    grid = _context(request).sites.geometry(record.site_id)
    if variables:
        names = [name.strip() for name in str(variables).split(",") if name.strip()]
    else:
        names = ["utci", "tmrt"]
    report = selected_time.coverage_report(
        _store(request), scenario_id, tuple(names), int(grid["time_steps"])
    )
    report["flag"] = selected_time.ENV_FLAG_NAME
    return JSONResponse(content=report)
