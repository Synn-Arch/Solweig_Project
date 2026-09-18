# SPDX-License-Identifier: GPL-3.0-only
"""Job status and cancellation endpoints.

``GET /api/v1/jobs/{job_id}`` implements the contract's queued/running/
complete shapes. Terminal statuses are ``complete``, ``superseded``,
``failed``, and ``cancelled``; failed jobs expose the error envelope fields
inline, and ``result_manifest_url`` appears only once a result was actually
published.

``POST /api/v1/jobs/{job_id}/cancel`` makes ``cancelled`` reachable:

* a queued or running job becomes ``cancelled``; repeating the cancel on the
  now-terminal job is idempotent (200 with the ``cancelled`` body);
* cancelling any other finished job is a 409 ``invalid_request`` — the
  observed terminal outcome stands. A cancel that loses the race to a
  completion/supersession mid-request gets the same 409 naming the status
  it observed, never a 200 that silently reports someone else's outcome;
* a result that arrives for a job cancelled mid-flight is discarded — the
  runner's publish path re-checks job liveness inside the publish
  transaction itself, so a cancel committing in the guard-to-publish window
  still means the result is never published. The scenario stays
  ``refining`` until the next edit supersedes the cancelled work.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from solweig_gpu.server.jobs import job_eta_fields
from solweig_gpu.server.models import ApiError
from solweig_gpu.server.store import JobRecord, JobNotFound, result_manifest_url

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])


def _job_body(
    job: JobRecord,
    *,
    queue_position: int | None = None,
    eta_seconds: int | None = None,
    eta_basis: str | None = None,
) -> dict[str, Any]:
    """Render one job row in the contract's per-status shape.

    ``eta_seconds``/``eta_basis`` (routing policy wave) are ADDITIVE
    top-level fields on the queued and running shapes only: an honest
    estimate from the scenario's own completed-job history, or null when
    unknown. Terminal shapes never carry them (a finished job has no ETA)
    and deployed clients that ignore unknown fields are unaffected.
    """
    store_queue_position = queue_position
    body: dict[str, Any] = {
        "job_id": job.job_id,
        "scenario_id": job.scenario_id,
        "target_scene_version": job.target_scene_version,
        "status": job.status,
    }
    if job.status == "queued":
        body["queue_position"] = (
            store_queue_position if store_queue_position is not None else 1
        )
        body["progress"] = None
        body["eta_seconds"] = eta_seconds
        body["eta_basis"] = eta_basis
    elif job.status == "running":
        body["stage"] = job.stage or "starting"
        body["progress"] = job.progress
        body["mode"] = job.mode
        body["window"] = job.window
        body["eta_seconds"] = eta_seconds
        body["eta_basis"] = eta_basis
    elif job.status == "complete":
        body["result_manifest_url"] = result_manifest_url(
            job.scenario_id,
            job.result_scene_version
            if job.result_scene_version is not None
            else job.target_scene_version,
        )
        body["metrics"] = job.metrics
    else:  # superseded / failed / cancelled
        body["metrics"] = job.metrics
        if job.error:
            body["error"] = dict(job.error)
    if job.mode is not None and job.status != "running":
        body["mode"] = job.mode
    if job.window is not None and job.status != "running":
        body["window"] = job.window
    if job.plan is not None and job.status != "running":
        # The executed impact plan (u-d4): server-served per-node stages,
        # realized routing, and estimates vs actuals for executor-path
        # jobs. Tree-only jobs carry no plan; clients keep deriving.
        body["impact_plan"] = job.plan
    for field in ("queued_at", "started_at", "finished_at", "worker_revision"):
        value = getattr(job, field)
        if value is not None:
            body[field] = value
    return body


@router.get("/{job_id}")
def get_job(job_id: str, request: Request) -> JSONResponse:
    context = request.app.state.context
    store = context.store
    job = store.get_job(job_id)
    if job is None:
        raise ApiError("job_not_found", f"job {job_id!r} does not exist", status=404)
    position = store.queue_position(job.job_id) if job.status == "queued" else None
    eta_seconds, eta_basis = job_eta_fields(store, context.sites, job)
    return JSONResponse(
        content=_job_body(
            job, queue_position=position, eta_seconds=eta_seconds, eta_basis=eta_basis
        )
    )


@router.post("/{job_id}/cancel")
def cancel_job(job_id: str, request: Request) -> JSONResponse:
    """Cancel a queued or running job (idempotent, version-less mutation).

    Cancellation is not tied to a scene version, so no ``Idempotency-Key`` /
    ``If-Match`` preconditions apply. Re-cancelling a cancelled job replays
    the same 200 response; a cancel arriving after any other terminal status
    (including losing a race to a completion) is a 409 naming that status.

    Both bodies render through the same eta-wired path as ``GET``: the
    estimate is computed from the job the cancel observed (the queued or
    running row still carries the estimate it would have had). A cancelled
    body is terminal, so the rendered shape structurally omits the eta
    keys — the wiring keeps every ``_job_body`` call site on one contract.
    """
    context = request.app.state.context
    store = context.store
    job = store.get_job(job_id)
    if job is None:
        raise ApiError("job_not_found", f"job {job_id!r} does not exist", status=404)
    if job.status == "cancelled":
        # Idempotent replay: retrying a cancel that already took effect must
        # not error out from under a client retry loop.
        eta_seconds, eta_basis = job_eta_fields(store, context.sites, job)
        return JSONResponse(
            content=_job_body(job, eta_seconds=eta_seconds, eta_basis=eta_basis)
        )
    if job.status not in ("queued", "running"):
        raise ApiError(
            "invalid_request",
            f"job {job_id!r} already finished with status {job.status!r}; "
            "only queued or running jobs can be cancelled",
            field="status",
            status=409,
        )
    # The estimate the job would have carried live, computed from the
    # pre-cancel row before the terminal transition renders it away.
    eta_seconds, eta_basis = job_eta_fields(store, context.sites, job)
    cancelled_transition = store.finish_job(
        job_id,
        "cancelled",
        error={"code": "cancelled", "message": "cancelled by client request"},
    )
    if not cancelled_transition:
        # Lost the race: the job reached a terminal outcome between the
        # status check above and the guarded UPDATE. Report what actually
        # happened instead of echoing a cancelled body over someone else's
        # result.
        observed = store.require_job(job_id)
        raise ApiError(
            "invalid_request",
            f"job {job_id!r} already finished with status {observed.status!r}; "
            "only queued or running jobs can be cancelled",
            field="status",
            status=409,
        )
    # Retire any subscriber's countdown: an exact-lane cancel is a terminal
    # transition the SSE ``exact_progress`` plane must observe (the runner's
    # own chokepoint never sees a route-side cancel). Same additive payload
    # shape, best-effort like every hub publish.
    hub = getattr(request.app.state, "broadcast_hub", None)
    cancelled = store.require_job(job_id)
    cancel_request = (
        cancelled.request if isinstance(cancelled.request, dict) else {}
    )
    if hub is not None and cancel_request.get("exact_lane"):
        try:
            hub.broadcast_exact_progress(
                cancelled.scenario_id,
                {
                    "job_id": cancelled.job_id,
                    "status": cancelled.status,
                    "target_revision": int(cancelled.target_scene_version),
                    "eta_seconds": None,
                    "eta_basis": None,
                },
            )
        except Exception:  # pragma: no cover - telemetry never blocks the route
            logger.exception(
                "exact_progress (cancel) event failed for job %s", job_id
            )
    return JSONResponse(
        content=_job_body(
            store.require_job(job_id), eta_seconds=eta_seconds, eta_basis=eta_basis
        )
    )


__all__ = ["router", "JobNotFound"]
