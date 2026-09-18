# SPDX-License-Identifier: GPL-3.0-only
"""FastAPI application factory for the SOLWEIG incremental design tool.

The web framework stays optional for the core project: importing this module
requires the ``server`` extra (``pip install 'solweig-gpu[server]'``), while
``import solweig_gpu`` never touches FastAPI.

The factory wires together the SQLite store, the site registry (site manifests
loaded through ``solweig_gpu.incremental.load_manifest`` — the single source
of truth for UV-to-world conversion), the single background job runner with an
injectable solver (default: the Phase-5 ``ExactWorker`` adapter), and the
routes. Every mutation endpoint is idempotent through ``Idempotency-Key``
headers; optimistic concurrency is enforced through ``base_scene_version`` /
``If-Match`` with explicit ``scene_version_conflict`` envelopes.

Deployment note (P9, deployment_operations): this app ships with NO
authentication or transport security. It is designed to sit behind an
authenticating reverse proxy (TLS termination, bearer-token or mTLS auth)
on a trusted internal network; exposing it directly is unsupported until
that layer lands. In-app defence-in-depth controls (REL-003): a request
body-size cap (413), a per-IP sliding-window rate limit (429 +
``Retry-After``, ops endpoints exempt), and a per-scenario mutation budget
— all default-on and overridable via :func:`create_app`. The per-IP limit
is in-process state only; public / multi-process deployments must enforce
the real per-IP cap (and TLS) at the reverse proxy.
"""

from __future__ import annotations

import logging
import math
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from solweig_gpu.server import patch_codec
from solweig_gpu.server.executor_bridge import make_universal_dispatch_solver
from solweig_gpu.server.jobs import (
    JobRunner,
    RunnerContext,
    SiteRegistry,
    Solver,
    adaptive_coalescing_enabled,
    materialize_baseline_result,
)
from solweig_gpu.server.models import ApiError, SHARED_SCENARIO_ID_RE
from solweig_gpu.server.store import (
    JobNotFound,
    LATEST_SCHEMA_VERSION,
    ScenarioNotFound,
    ScenarioQuotaExceeded,
    SceneVersionConflict,
    SiteIdentityMismatch,
    Store,
    StoreError,
    new_request_id,
)
from solweig_gpu.server.realtime.admission import AdmissionEnvelope, FastAdmission
from solweig_gpu.server.realtime.broadcast import BroadcastHub
from solweig_gpu.server.realtime.epochs import (
    DEFAULT_TICK_MS,
    EPOCH_TICK_MAX_MS,
    EPOCH_TICK_MIN_MS,
    EpochScheduler,
)
from solweig_gpu.server.realtime.scheduler import FastLaneScheduler
from solweig_gpu.server.realtime.selected_time import SelectedTimeDemand, enabled as selected_time_enabled
from solweig_gpu.server.realtime.telemetry import default_registry
from solweig_gpu.server.routes_capabilities import router as capabilities_router
from solweig_gpu.server.routes_jobs import router as jobs_router
from solweig_gpu.server.routes_realtime import router as realtime_router
from solweig_gpu.server.routes_scenarios import router as scenarios_router

logger = logging.getLogger(__name__)

__all__ = [
    "create_app",
    "RateLimiter",
    "IpRateLimiter",
    "ScenarioQuotaLimiter",
    "DEFAULT_MAX_REQUEST_BODY_BYTES",
    "DEFAULT_REQUESTS_PER_MINUTE_PER_IP",
    "DEFAULT_EDITS_PER_MINUTE_PER_SCENARIO",
]

API_V1 = "/api/v1"


def _default_epoch_scheduler(
    store: Store,
    hub: BroadcastHub,
    tick_ms: float | None = None,
    window_hint_s: Any = None,
) -> EpochScheduler:
    """Production wiring: the epoch scheduler on the shared telemetry hub.

    ``tick_ms`` comes from ``create_app(epoch_tick_ms=...)`` (the
    service_level_contract.md "epoch duration: 100 ms nominal;
    configurable 50-200 ms" knob, validated there). ``None`` keeps the
    100 ms default byte-identical. ``window_hint_s`` (T15, flag-gated at
    the wiring site) is the adaptive-epoch hint clamped into the same
    50-200 ms domain by the scheduler itself.
    """
    return EpochScheduler(
        store,
        hub=hub,
        telemetry=default_registry(),
        tick_ms=tick_ms if tick_ms is not None else DEFAULT_TICK_MS,
        window_hint_s=window_hint_s,
    )


def _default_fast_lane(
    store: Store, hub: BroadcastHub, sites: Any = None
) -> FastLaneScheduler:
    """Production wiring: the deadline-isolated fast lane (r2b + R7).

    One per app, own daemon thread, telemetry on the shared registry so
    ``/metrics`` scrapes it. The epochs tail hook is attached in
    :func:`create_app` (also for injected factories building their own
    ``EpochScheduler``), so the fast lane wakes on every epoch close
    without the close pipeline ever waiting on fast work.

    R7: with a registered site source AND measured calibration evidence
    shipped in ``compensated.DEFAULT_QUALIFIER_EVIDENCE``, the lane's
    kernel is the compensated vegetation kernel (view epochs keep their
    structural ``fast_exact`` path inside it; the qualifier fence in the
    scheduler still guards every publication). Without measured evidence
    the kernel stays ``ViewCacheKernel`` — compensation is never wired on
    unmeasured numbers.
    """
    kernel: Any = None
    if sites is not None:
        try:
            from solweig_gpu.server.realtime.compensated import (
                DEFAULT_QUALIFIER_EVIDENCE,
                SiteRegistrySunSource,
                VegetationCompensationKernel,
            )

            if DEFAULT_QUALIFIER_EVIDENCE is not None:
                kernel = VegetationCompensationKernel(
                    store=store,
                    sites=SiteRegistrySunSource(sites),
                    evidence=DEFAULT_QUALIFIER_EVIDENCE,
                )
        except Exception:  # pragma: no cover - wiring must never block boot
            logger.exception("compensated fast kernel unavailable; view kernel only")
            kernel = None
    return FastLaneScheduler(
        store, hub=hub, telemetry=default_registry(), kernel=kernel
    )


def _ensure_shared_world(
    context: RunnerContext,
    runner: JobRunner,
    shared_scenario_id: str,
) -> str | None:
    """Find-or-create the shared-world scenario at boot (idempotent).

    Uses the same creation path as ``POST /api/v1/scenarios``: the site's
    stored baseline outputs are materialized inline as exact result 0 when
    present; a site without them gets a durable baseline job submitted to
    the runner. A scenario row with the shared id already exists -> found
    (its site pin is validated, and a failed boot baseline is re-queued).

    One shared world per server: when creating, it is pinned to the first
    configured site (site ids are sorted by the registry); multi-site
    deployments keep per-scenario minting for the other sites. Returns the
    workspace id to advertise, or ``None`` when a FOUND scenario is pinned
    to a site this deployment no longer serves — then capabilities omit the
    pointer (the studio falls back to minting) instead of pointing every
    visitor at an unservable scenario.

    Multi-process note: two workers racing the find-or-create both use the
    same fixed id; the loser's INSERT fails on the primary key and that
    worker simply serves the winner's row (single-writer state roots are
    the deployment norm; no cross-process lock is taken).
    """
    store = context.store
    sites = context.sites
    site_ids = sites.site_ids()
    if not site_ids:  # pragma: no cover - create_app requires sites
        return shared_scenario_id
    site_id = site_ids[0]
    existing = store.get_scenario(shared_scenario_id)
    if existing is not None:
        if existing.site_id not in site_ids:
            logger.warning(
                "shared world %s exists but is pinned to site %s, which this "
                "deployment does not serve (%s); not advertising a default "
                "workspace — visitors mint private scenarios until the state "
                "root is cleaned or the shared id is re-created on a served "
                "site",
                shared_scenario_id,
                existing.site_id,
                list(site_ids),
            )
            return None
        logger.info(
            "shared world: scenario %s found (site %s, scene version %d)",
            shared_scenario_id,
            existing.site_id,
            int(existing.scene_version),
        )
        latest = store.latest_job_for_scenario(shared_scenario_id)
        if (
            latest is not None
            and latest.status in ("failed", "cancelled")
            and store.latest_result(shared_scenario_id) is None
        ):
            # A failed boot baseline would brick the shared world for every
            # visitor forever (no client ever re-requests it); re-queue the
            # durable job so this boot heals it.
            if store.requeue_failed_job(latest.job_id):
                runner.submit(latest.job_id)
                logger.info(
                    "shared world: baseline job %s ended %s with no readable "
                    "result; re-queued at boot",
                    latest.job_id,
                    latest.status,
                )
        return shared_scenario_id
    if len(site_ids) > 1:
        logger.warning(
            "shared world %s is pinned to site %s; sites %s keep per-scenario "
            "minting (one shared id cannot span sites)",
            shared_scenario_id,
            site_id,
            list(site_ids[1:]),
        )
    config = sites.config(site_id)
    has_baseline = (config.cache_dir / "baseline_results" / "metadata.json").is_file()

    def baseline_publisher(scenario_id: str, scene_version: int):
        return materialize_baseline_result(sites, scenario_id, site_id, scene_version)

    record, baseline_job_id = store.create_scenario(
        site_id=site_id,
        name="Shared world",
        scenario_id=shared_scenario_id,
        # The shared world is deployment infrastructure, not a user session:
        # exempt from the scenario quota so a full quota can never brick boot
        # (no scenario-delete API exists to make room).
        max_scenarios=None,
        site_identity={"site_id": site_id, **sites.geometry(site_id)},
        baseline_publisher=baseline_publisher if has_baseline else None,
    )
    if baseline_job_id is not None:
        # No stored baseline outputs: schedule the baseline full-solve job
        # exactly like the create route does.
        runner.submit(baseline_job_id)
    logger.info(
        "shared world: created scenario %s for site %s (baseline %s)",
        shared_scenario_id,
        record.site_id,
        "materialized" if has_baseline else "job scheduled",
    )
    return shared_scenario_id

# ---------------------------------------------------------------------------
# P9 security controls (REL-003) — defaults are classroom-realistic and
# overridable per deployment (see create_app parameters). Multi-process
# deployments must enforce the per-IP limit at the proxy: this state is
# in-process only.
# ---------------------------------------------------------------------------

#: Requests with a body larger than this are rejected with 413 before the
#: body is read (edits and scenario creates are small JSON documents; large
#: payloads travel in RESPONSES, which this cap does not touch).
DEFAULT_MAX_REQUEST_BODY_BYTES = 1 << 20  # 1 MiB

#: General per-client-IP request budget (all routes except the ops surface).
DEFAULT_REQUESTS_PER_MINUTE_PER_IP = 30

#: Mutation budget per scenario (POST edits + resets).
DEFAULT_EDITS_PER_MINUTE_PER_SCENARIO = 20

#: View-only budget per scenario (POST views). Reads are cheaper than
#: solves, so the budget is larger than the mutation budget, but still
#: bounded: view requests hit the store and legacy manifests may stream
#: payloads for legends.
DEFAULT_VIEWS_PER_MINUTE_PER_SCENARIO = 120

#: Ops surface exempt from the per-IP budget: monitors poll these freely.
#: (``/healthz`` is the legacy alias and stays counted; deploy monitors on
#: the documented ``/health/*`` surface.)
OPS_EXEMPT_PATHS = frozenset(
    {"/health/live", "/health/ready", "/health/worker", "/metrics"}
)


def _loc_to_field(loc: list[Any]) -> str | None:
    """Format a pydantic error location as a contract-style field path."""
    segments: list[str] = []
    for part in loc:
        if isinstance(part, int):
            if segments:
                segments[-1] = f"{segments[-1]}[{part}]"
            else:
                segments.append(f"[{part}]")
        else:
            segments.append(str(part))
    return ".".join(segments) or None


class RateLimiter:
    """Small sliding-window limiter used to emit ``rate_limited`` errors."""

    def __init__(self, max_per_minute: int) -> None:
        self.max_per_minute = int(max_per_minute)
        self._events: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        with self._lock:
            now = time.monotonic()
            window = [t for t in self._events.get(key, []) if now - t < 60.0]
            if len(window) >= self.max_per_minute:
                self._events[key] = window
                raise ApiError(
                    "rate_limited",
                    f"more than {self.max_per_minute} mutations per minute for this "
                    "scenario; retry shortly",
                    status=429,
                )
            window.append(now)
            self._events[key] = window


class IpRateLimiter:
    """Sliding-window per-client-IP request limiter (REL-003).

    Unlike :class:`RateLimiter` (per-scenario, raises from route handlers)
    this one runs inside the HTTP middleware, where raising ``ApiError``
    would bypass the exception handlers — so it returns the ``Retry-After``
    seconds instead and the middleware builds the 429 envelope itself.

    State is in-process: behind a multi-process server (or when trusting a
    proxy's ``X-Forwarded-For`` via ``uvicorn --proxy-headers``) each worker
    keeps its own windows, so the effective budget is multiplied by the
    worker count. Enforce the per-IP cap at the reverse proxy for public
    deployments; this layer is defence in depth.
    """

    def __init__(self, max_per_minute: int) -> None:
        self.max_per_minute = int(max_per_minute)
        self._events: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def retry_after(self, key: str) -> float | None:
        """Consume one slot for ``key``; return seconds to wait, or ``None``
        when the request is allowed."""
        with self._lock:
            now = time.monotonic()
            window = [t for t in self._events.get(key, []) if now - t < 60.0]
            if len(window) >= self.max_per_minute:
                self._events[key] = window
                return max(1.0, math.ceil(60.0 - (now - window[0])))
            window.append(now)
            self._events[key] = window
            return None


class ScenarioQuotaLimiter:
    """Caps active scenarios for ``site_limit_exceeded`` errors."""

    def __init__(self, max_scenarios: int, store: Store) -> None:
        self.max_scenarios = int(max_scenarios)
        self._store = store

    def allow(self) -> bool:
        return self._store.count_scenarios() < self.max_scenarios


def _install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def handle_api_error(request: Request, error: ApiError) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None) or new_request_id()
        return JSONResponse(
            status_code=error.status, content=error.envelope(request_id)
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation(
        request: Request, error: RequestValidationError
    ) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None) or new_request_id()
        first = error.errors()[0] if error.errors() else {}
        parts = [part for part in first.get("loc", []) if part != "body"]
        field = _loc_to_field(parts)
        message = first.get("msg", "request validation failed")
        envelope = ApiError(
            "invalid_request",
            f"{message}" + (f" (field: {field})" if field else ""),
            field=field,
        ).envelope(request_id)
        return JSONResponse(status_code=400, content=envelope)

    @app.exception_handler(ScenarioNotFound)
    async def handle_scenario_not_found(
        request: Request, error: ScenarioNotFound
    ) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content=ApiError(
                "scenario_not_found", str(error), status=404
            ).envelope(getattr(request.state, "request_id", None)),
        )

    @app.exception_handler(JobNotFound)
    async def handle_job_not_found(request: Request, error: JobNotFound) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content=ApiError("job_not_found", str(error), status=404).envelope(
                getattr(request.state, "request_id", None)
            ),
        )

    @app.exception_handler(SceneVersionConflict)
    async def handle_version_conflict(
        request: Request, error: SceneVersionConflict
    ) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content=ApiError(
                "scene_version_conflict",
                "The scenario changed after the client's base version.",
                details={"current_scene_version": error.current_scene_version},
                status=409,
            ).envelope(getattr(request.state, "request_id", None)),
        )

    @app.exception_handler(SiteIdentityMismatch)
    async def handle_site_identity_mismatch(
        request: Request, error: SiteIdentityMismatch
    ) -> JSONResponse:
        # Normally mapped by the scenario routes; this handler covers any
        # other path that trips the store-level pin check (U-D item g).
        return JSONResponse(
            status_code=409,
            content=ApiError(
                "site_identity_mismatch",
                str(error),
                details={
                    "pinned_identity": error.expected,
                    "current_identity": error.actual,
                },
                status=409,
            ).envelope(getattr(request.state, "request_id", None)),
        )

    @app.exception_handler(ScenarioQuotaExceeded)
    async def handle_quota_exceeded(
        request: Request, error: ScenarioQuotaExceeded
    ) -> JSONResponse:
        # Normally mapped by the create route; this handler covers any other
        # path that trips the store-level quota check.
        return JSONResponse(
            status_code=403,
            content=ApiError(
                "site_limit_exceeded", str(error), status=403
            ).envelope(getattr(request.state, "request_id", None)),
        )

    @app.exception_handler(StoreError)
    async def handle_store_error(request: Request, error: StoreError) -> JSONResponse:
        logger.exception("store error")
        return JSONResponse(
            status_code=500,
            content=ApiError("internal_error", str(error), status=500).envelope(
                getattr(request.state, "request_id", None)
            ),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, error: Exception) -> JSONResponse:
        logger.exception("unhandled error")
        return JSONResponse(
            status_code=500,
            content=ApiError(
                "internal_error", f"{type(error).__name__}: {error}", status=500
            ).envelope(getattr(request.state, "request_id", None)),
        )


def create_app(
    *,
    state_root: str | Path,
    sites: Mapping[str, Any],
    solver_factory: Callable[[RunnerContext], Solver] | None = None,
    coalescing_window_ms: float = 500.0,
    adaptive_coalesce: bool | None = None,
    epoch_tick_ms: float | None = None,
    max_scenarios: int | None = None,
    max_trees_per_scenario: int = 200,
    edits_per_minute: int | None = DEFAULT_EDITS_PER_MINUTE_PER_SCENARIO,
    max_request_body_bytes: int | None = DEFAULT_MAX_REQUEST_BODY_BYTES,
    requests_per_minute_per_ip: int | None = DEFAULT_REQUESTS_PER_MINUTE_PER_IP,
    views_per_minute: int | None = DEFAULT_VIEWS_PER_MINUTE_PER_SCENARIO,
    worker_heartbeat_timeout_s: float = 60.0,
    start_worker: bool = True,
    epoch_scheduler_factory: Callable[[Store, "BroadcastHub"], Any] | None = None,
    start_epoch_scheduler: bool = True,
    fast_lane_factory: Callable[[Store, "BroadcastHub"], Any] | None = None,
    start_fast_lane: bool = True,
    admission_envelope: AdmissionEnvelope | None = None,
    shared_scenario_id: str | None = None,
    title: str = "SOLWEIG incremental design tool",
) -> FastAPI:
    """Build the design-tool API application.

    Parameters
    ----------
    state_root:
        Directory for the SQLite store and per-scenario result payloads.
    sites:
        Mapping of ``site_id`` to ``SiteConfig``-compatible values (or dicts
        with ``cache_dir``/``site_dir``/``selected_date_str`` keys).
    solver_factory:
        Dependency-injected solver factory. Defaults to the Phase-5
        ``ExactWorker`` adapter; tests inject a fake solver.
    coalescing_window_ms:
        Delay before dispatching a queued job so rapid edits batch together.
        (Legacy-lane knob ONLY: it does not reach the epoch scheduler —
        see ``epoch_tick_ms``. Mixing the two up was an R8 commissioning
        finding.)
    adaptive_coalesce:
        T24 (DESIGN §710): let the runner shrink that delay to the service
        contract's 50 ms latency floor for isolated jobs (no burst
        indicator ahead), keeping the full window under bursts. ``None``
        (default) reads the SEPARATE env flag
        ``SOLWEIG_RT_ADAPTIVE_COALESCE`` at construction, exactly like the
        T15 selected-time flag; unset keeps the fixed window byte-identical.
    epoch_tick_ms:
        The realtime epoch scheduler's tick (service_level_contract.md:
        "epoch duration: 100 ms nominal; configurable 50-200 ms"). Validated
        against the contract's own 50-200 ms domain — out-of-domain values
        raise ``ValueError`` at construction, before any wiring runs.
        ``None`` (default) keeps the 100 ms scheduler byte-identical. Only
        the DEFAULT epoch scheduler receives it; an injected
        ``epoch_scheduler_factory`` owns its scheduler's construction
        entirely (build your own ``EpochScheduler(tick_ms=...)`` there).
    max_scenarios / edits_per_minute:
        Optional caps producing ``site_limit_exceeded`` / ``rate_limited``
        (per-scenario mutation budget; ``None`` disables).
    max_request_body_bytes:
        Reject request bodies above this size with 413 ``payload_too_large``
        (checked against ``Content-Length`` before reading the body, then
        again on the actual bytes). ``None`` disables. Exports/payloads are
        responses and never affected.
    requests_per_minute_per_ip:
        Sliding-window budget per client IP across all non-ops routes,
        answering 429 ``rate_limited`` with ``Retry-After``. ``None``
        disables. Ops endpoints (``/health*``, ``/metrics``) are exempt so
        monitoring can poll freely. In-process only: multi-process or
        public deployments need the proxy-level limiter.
    worker_heartbeat_timeout_s:
        ``/health/ready`` fails when the worker's last heartbeat is older
        than this (long solves legitimately run minutes between progress
        callbacks, so the default is generous).
    start_worker:
        Start the background runner immediately (lifespan). Disable to enqueue
        jobs durably without executing them (restart-recovery testing).
    epoch_scheduler_factory:
        Builds the epoch scheduler from ``(store, broadcast_hub)``. Defaults
        to the real 100 ms :class:`EpochScheduler`; tests inject fake-clock
        or no-op schedulers.
    start_epoch_scheduler:
        Start the scheduler's daemon thread in the lifespan (recovery pass +
        tick loop). Disable to drive the injected scheduler manually.
    fast_lane_factory:
        Builds the fast lane scheduler from ``(store, broadcast_hub)``
        (r2b). Defaults to the real deadline-isolated
        :class:`FastLaneScheduler` on the shared telemetry registry; tests
        inject private-telemetry lanes for isolation (the module-level
        shared registry must not accumulate other apps' samples).
    start_fast_lane:
        Start the fast lane's daemon thread in the lifespan (recovery pass
        + wake/scan loop). Disable to drive the injected lane manually.
        The epochs tail hook is attached either way.
    admission_envelope:
        Override the realtime admission envelope (service_level_contract.md
        published bounds). ``None`` keeps the documented defaults.
    shared_scenario_id:
        Find-or-create ONE shared-world scenario with this id during the
        lifespan (the workspace every visitor joins by default), and serve
        it as ``default_workspace_id`` in ``/api/v1/capabilities``. The
        env-driven entrypoint always sets it (default
        ``scn_shared_world``); ``None`` keeps this app per-scenario only.
        Per-scenario minting (POST /api/v1/scenarios) stays available
        either way.
    """
    from contextlib import asynccontextmanager

    if shared_scenario_id is not None and not SHARED_SCENARIO_ID_RE.fullmatch(
        shared_scenario_id
    ):
        # Same convention the env entrypoint enforces: the id doubles as a
        # path segment under the state root, so a malformed value must never
        # reach the store (direct create_app callers bypass __main__).
        raise ValueError(
            f"shared_scenario_id must follow the scenario id convention "
            f"(scn_ prefix, then lowercase letters, digits, underscores or "
            f"hyphens), got {shared_scenario_id!r}"
        )

    if adaptive_coalesce is None:
        # T24 wiring follows the T15 precedent: the env flag is read HERE,
        # once, at construction — unset keeps the fixed window (the runner's
        # default) and the app behaves byte-identically to before.
        adaptive_coalesce = adaptive_coalescing_enabled()

    if epoch_tick_ms is not None:
        # service_level_contract.md: "Epoch duration | 100 ms nominal;
        # configurable 50-200 ms". Validate the contract's own domain
        # BEFORE any wiring runs (also when a scheduler factory is
        # injected — a bad value must never reach production silently).
        if not (EPOCH_TICK_MIN_MS <= epoch_tick_ms <= EPOCH_TICK_MAX_MS):
            raise ValueError(
                f"epoch_tick_ms={epoch_tick_ms} is outside the admitted "
                "epoch-duration domain 50-200 ms "
                "(service_level_contract.md: 100 ms nominal, configurable "
                "50-200 ms); refusing to build the app"
            )

    state_root = Path(state_root)
    store = Store(state_root / "store.sqlite3", results_root=state_root / "scenarios")
    site_registry = SiteRegistry(sites)
    context = RunnerContext(store=store, sites=site_registry, state_root=state_root)
    # Realtime plane (R1 epochs wave): one SSE hub + the epoch scheduler.
    # Built before the JobRunner so the exact lane can emit its named
    # ``exact_revision`` event through the same hub (verify bridge).
    broadcast_hub = BroadcastHub()
    runner = JobRunner(
        context,
        # The dispatcher keeps the ExactWorker path byte-for-byte for
        # tree-only scenarios and routes family-edit scenarios onto the
        # universal PlanExecutor bridge (u-d4).
        solver_factory=(
            make_universal_dispatch_solver(solver_factory)
            if solver_factory is not None
            else make_universal_dispatch_solver()
        ),
        coalescing_window_ms=coalescing_window_ms,
        adaptive_coalesce=adaptive_coalesce,
        broadcast_hub=broadcast_hub,
    )
    # T15 adaptive-epoch seam (flag-gated): while any selected-time stream
    # holds demand, epochs close at the contract's latency floor instead
    # of the configured cadence, so a freshly accepted operation reaches
    # its published fast/exact state at the admitted 50 ms bound. The
    # hint is CLAMPED into the 50-200 ms tick domain by the scheduler;
    # demand clears the moment the last selected-time stream ends.
    # Injected scheduler factories own their construction — they can wire
    # the hint themselves via the same ``app.state`` tracker below.
    selected_time_demand: SelectedTimeDemand | None = None
    epoch_window_hint = None
    if selected_time_enabled():
        selected_time_demand = SelectedTimeDemand()
        epoch_window_hint = lambda: (
            50.0 / 1000.0 if selected_time_demand.pending() else None
        )
    if epoch_scheduler_factory is not None:
        epoch_scheduler = epoch_scheduler_factory(store, broadcast_hub)
    else:
        # The contract's epoch-duration knob reaches the default
        # scheduler (R8b N3); injected factories own their construction.
        epoch_scheduler = _default_epoch_scheduler(
            store, broadcast_hub, tick_ms=epoch_tick_ms, window_hint_s=epoch_window_hint
        )
    # Fast lane (r2b): deadline-isolated scheduler + the ingest admission
    # gate that rejects outside-envelope work BEFORE durable acceptance.
    # Injected factories keep the (store, hub) protocol; only the default
    # wiring receives the site registry (R7 compensated kernel source).
    if fast_lane_factory is not None:
        fast_lane = fast_lane_factory(store, broadcast_hub)
    else:
        fast_lane = _default_fast_lane(store, broadcast_hub, site_registry)
    attach_after_close = getattr(epoch_scheduler, "attach_after_close", None)
    if attach_after_close is not None:
        attach_after_close(fast_lane.on_epoch_closed)
    fast_admission = FastAdmission(
        envelope=admission_envelope, capacity=fast_lane
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if start_worker:
            runner.start()
        if shared_scenario_id is not None:
            # Shared world: ensure the one workspace every visitor joins
            # exists (find-or-create through the POST /scenarios creation
            # path) before the app serves its first request. The result
            # re-assigns the advertised pointer: a found scenario pinned to
            # an unserved site clears it (capabilities then omit the field).
            app.state.shared_scenario_id = _ensure_shared_world(
                context,
                runner,
                shared_scenario_id,
            )
        if start_epoch_scheduler:
            epoch_scheduler.start()
        if start_fast_lane:
            fast_lane.start()
        yield
        # Fast lane first (it consumes epoch closes and writes through the
        # store), then the epoch scheduler (its close pipeline also writes
        # through the store) — both must quiesce before the store closes.
        fast_lane.stop()
        epoch_scheduler.stop()
        broadcast_hub.close()
        runner.stop()
        store.close()

    app = FastAPI(title=title, version="1", lifespan=lifespan)
    app.state.context = context
    app.state.runner = runner
    app.state.broadcast_hub = broadcast_hub
    app.state.epoch_scheduler = epoch_scheduler
    app.state.fast_lane = fast_lane
    app.state.fast_admission = fast_admission
    #: T15 selected-time demand tracker — None when the flag is off (the
    #: routes and epoch hint treat "absent" and "disabled" identically).
    app.state.selected_time_demand = selected_time_demand
    app.state.coalescing_window_ms = coalescing_window_ms
    #: Contract epoch-duration knob (validated 50-200 ms; None = 100 ms
    #: default). Diagnostics/commissioning surface, not a runtime control.
    app.state.epoch_tick_ms = epoch_tick_ms
    app.state.max_trees_per_scenario = max_trees_per_scenario
    #: Authoritative quota: enforced inside the store's create transaction so
    #: concurrent creates cannot both pass a read-then-write precheck.
    app.state.max_scenarios = max_scenarios
    app.state.rate_limiter = RateLimiter(edits_per_minute) if edits_per_minute else None
    #: View-only reads are far cheaper than solves but still touch the store
    #: and (for legacy manifests) the payload codec, so they carry their own
    #: per-scenario budget — distinct from the mutation budget so a polling
    #: view client cannot exhaust the edit path's limiter and vice versa.
    app.state.views_rate_limiter = (
        RateLimiter(views_per_minute) if views_per_minute else None
    )
    app.state.site_limiter = (
        ScenarioQuotaLimiter(max_scenarios, store) if max_scenarios else None
    )
    app.state.max_request_body_bytes = max_request_body_bytes
    app.state.ip_rate_limiter = (
        IpRateLimiter(requests_per_minute_per_ip)
        if requests_per_minute_per_ip
        else None
    )
    app.state.worker_heartbeat_timeout_s = float(worker_heartbeat_timeout_s)
    #: Shared-world pointer (None = this app serves no default workspace);
    #: the capabilities route merges it into the discovery document.
    app.state.shared_scenario_id = shared_scenario_id

    _install_error_handlers(app)

    @app.middleware("http")
    async def gate_and_stash_body(request: Request, call_next):
        """Security gate + request-id/body stash.

        This middleware runs OUTSIDE the exception handlers, so rejections
        return the contract envelope directly (raising ``ApiError`` here
        would surface as an opaque 500).

        Order: per-IP rate limit (cheap, no body read) → request body size
        cap (``Content-Length`` precheck, then actual bytes) → body stash.
        """
        request_id = new_request_id()
        request.state.request_id = request_id
        path = request.url.path

        ip_limiter: IpRateLimiter | None = getattr(
            request.app.state, "ip_rate_limiter", None
        )
        if ip_limiter is not None and path not in OPS_EXEMPT_PATHS:
            client_host = request.client.host if request.client else "unknown"
            wait_s = ip_limiter.retry_after(client_host)
            if wait_s is not None:
                response = JSONResponse(
                    status_code=429,
                    content=ApiError(
                        "rate_limited",
                        f"more than {ip_limiter.max_per_minute} requests per minute "
                        f"from this client; retry after {int(wait_s)}s",
                        status=429,
                    ).envelope(request_id),
                    headers={"Retry-After": str(int(wait_s))},
                )
                response.headers["X-Request-ID"] = request_id
                return response

        if request.method in ("POST", "PUT", "PATCH"):
            max_bytes = getattr(request.app.state, "max_request_body_bytes", None)
            declared = request.headers.get("content-length")
            if max_bytes is not None and declared is not None:
                try:
                    declared_bytes = int(declared)
                except ValueError:
                    declared_bytes = None
                if declared_bytes is not None and declared_bytes > int(max_bytes):
                    response = JSONResponse(
                        status_code=413,
                        content=ApiError(
                            "payload_too_large",
                            f"request body of {declared_bytes} bytes exceeds the "
                            f"{int(max_bytes)}-byte limit",
                            status=413,
                        ).envelope(request_id),
                    )
                    response.headers["X-Request-ID"] = request_id
                    return response
            body = await request.body()
            if max_bytes is not None and len(body) > int(max_bytes):
                response = JSONResponse(
                    status_code=413,
                    content=ApiError(
                        "payload_too_large",
                        f"request body of {len(body)} bytes exceeds the "
                        f"{int(max_bytes)}-byte limit",
                        status=413,
                    ).envelope(request_id),
                )
                response.headers["X-Request-ID"] = request_id
                return response
            request.state.raw_body = body

            async def receive():
                return {"type": "http.request", "body": body, "more_body": False}

            request._receive = receive
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    app.include_router(scenarios_router)
    app.include_router(jobs_router)
    app.include_router(capabilities_router)
    app.include_router(realtime_router)

    @app.get("/healthz", tags=["ops"])
    def healthz() -> dict[str, Any]:
        return {
            "status": "ok",
            "schema_version": patch_codec.PATCH_SCHEMA_VERSION,
            "worker_running": runner.running,
            "sites": list(site_registry.site_ids()),
        }

    @app.get("/metrics", tags=["ops"])
    def metrics() -> dict[str, Any]:
        counts = store.job_status_counts()
        return {
            "scenarios": store.count_scenarios(),
            "jobs": {
                status: counts.get(status, 0)
                for status in (
                    "queued",
                    "running",
                    "complete",
                    "superseded",
                    "failed",
                    "cancelled",
                )
            },
            "results_published": store.result_count(),
            # Realtime plane telemetry (r1-epochs flagged gap, wired r2b):
            # the process-wide solweig_rt_* registry — the epoch scheduler
            # and the fast lane both observe into it (deterministic
            # snapshot; Prometheus text rendering is render_text()).
            "realtime": default_registry().snapshot(),
        }

    # -- P9 ops endpoints (deployment_operations.md) -------------------------
    # ``/health/live``: answering at all proves the process and event loop
    # are responsive; it deliberately touches nothing else.

    @app.get("/health/live", tags=["ops"])
    async def health_live() -> dict[str, Any]:
        return {"status": "live", "time_utc": _utc_now_iso()}

    @app.get("/health/ready", tags=["ops"])
    def health_ready() -> JSONResponse:
        """Deployment readiness: every check must pass, else 503.

        Checks: store reachable, schema version this code understands, every
        registered site cache loadable, worker thread alive with a fresh
        heartbeat, and the result filesystem writable.
        """
        checks: dict[str, Any] = {}
        ready = True

        try:
            store.job_status_counts()
            checks["store"] = {"status": "ok", "schema_version": store.schema_version}
        except Exception as error:
            ready = False
            checks["store"] = {"status": "fail", "error": str(error)}
        else:
            if int(store.schema_version) > LATEST_SCHEMA_VERSION:
                ready = False
                checks["store"] = {
                    "status": "fail",
                    "error": (
                        f"store schema {store.schema_version} is newer than this "
                        f"code understands ({LATEST_SCHEMA_VERSION}); upgrade first"
                    ),
                }

        site_checks: dict[str, str] = {}
        for site_id in site_registry.site_ids():
            try:
                site_registry.manifest(site_id)
                site_registry.cache(site_id)
                site_checks[site_id] = "ok"
            except Exception as error:
                ready = False
                site_checks[site_id] = f"fail: {error}"
        checks["sites"] = site_checks

        status = runner.worker_status()
        heartbeat_stale = (
            status["seconds_since_heartbeat"] > app.state.worker_heartbeat_timeout_s
        )
        if not runner.running or heartbeat_stale:
            ready = False
            checks["worker"] = {
                "status": "stale" if heartbeat_stale else "stopped",
                "seconds_since_heartbeat": status["seconds_since_heartbeat"],
                "timeout_s": app.state.worker_heartbeat_timeout_s,
            }
        else:
            checks["worker"] = {
                "status": "ok",
                "seconds_since_heartbeat": status["seconds_since_heartbeat"],
            }

        try:
            probe = state_root / "scenarios" / f".ready-probe-{os.getpid()}"
            probe.parent.mkdir(parents=True, exist_ok=True)
            probe.write_bytes(b"ok")
            probe.unlink()
            checks["result_fs"] = {"status": "ok", "path": str(state_root / "scenarios")}
        except Exception as error:
            ready = False
            checks["result_fs"] = {"status": "fail", "error": str(error)}

        body: dict[str, Any] = {
            "status": "ready" if ready else "not_ready",
            "checks": checks,
        }
        return JSONResponse(status_code=200 if ready else 503, content=body)

    @app.get("/health/worker", tags=["ops"])
    def health_worker() -> dict[str, Any]:
        """Worker diagnostics (PID, revision, cache versions, current job,
        RSS, last success).

        This surface is operational detail, not API contract: restrict it to
        the ops network / proxy ACL in production (the app itself ships no
        authentication).
        """
        body = runner.worker_status()
        body["rss_bytes"] = _process_rss_bytes()
        body["site_cache_versions"] = {
            site_id: site_registry.site_cache_version(site_id)
            for site_id in site_registry.site_ids()
        }
        return body

    return app


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _process_rss_bytes() -> int | None:
    """Resident set size of this process; psutil when available."""
    try:
        import psutil  # type: ignore[import-not-found]

        return int(psutil.Process().memory_info().rss)
    except Exception:
        try:
            import resource

            return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (
                1024 if sys.platform != "darwin" else 1
            )
        except Exception:
            return None
