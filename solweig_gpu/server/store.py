# SPDX-License-Identifier: GPL-3.0-only
"""SQLite persistence for the SOLWEIG design-tool API service.

The store owns every durable fact the API needs:

* scenarios (site, monotonic ``scene_version``, ``exact_result_version``,
  status, consumed-edit watermark);
* the authoritative editable tree list per scenario;
* the append-only edit ledger (old and new tree objects);
* the ``(scope, idempotency_key) -> stored response`` table implementing
  replay and the ``idempotency_key_reused`` rule;
* job records (queued/running/complete/superseded/failed/cancelled) with
  progress, mode, window and metrics;
* published result manifests plus binary payloads on disk under
  ``<state_root>/scenarios/<scenario_id>/results/<scene_version>/``.

Concurrency: one shared connection guarded by an RLock; every mutation runs in
a single ``BEGIN IMMEDIATE`` transaction so idempotency checks, optimistic
version checks, the mutation itself and the stored response commit atomically.
SQLite runs in WAL mode so readers never block the worker thread.

Scientific-integrity guards enforced here (defense in depth behind the job
runner): ``publish_result`` refuses to publish a scene version other than the
scenario's current version and never decreases ``exact_result_version``.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, get_args

from solweig_gpu.server.realtime.types import CanonicalState, EpochStatus

__all__ = [
    "EpochRecord",
    "FastRevisionAdvance",
    "IdempotentReplay",
    "IdempotencyConflict",
    "JobRecord",
    "OperationFingerprintConflict",
    "OperationRecord",
    "ResultRecord",
    "ScenarioRecord",
    "SceneVersionConflict",
    "StaleResultError",
    "Store",
    "StoreError",
    "UnsupportedFamily",
    "uv_to_world",
    "world_to_uv",
]

TERMINAL_JOB_STATUSES = ("complete", "superseded", "failed", "cancelled")

#: Valid ``realtime_epochs.status`` values, derived from the frozen
#: :data:`solweig_gpu.server.realtime.types.EpochStatus` contract so the
#: schema and the interface cannot drift.
EPOCH_STATUSES: frozenset[str] = frozenset(get_args(EpochStatus))

#: Epoch statuses owned by the R2 compute lanes; the epoch close pipeline
#: never re-drives or downgrades an epoch in one of these.
TERMINAL_EPOCH_STATUSES: frozenset[str] = frozenset(
    {"fast_published", "exact_targeted"}
)
QUEUED = "queued"
RUNNING = "running"

MANIFEST_FILE = "manifest.json"
PAYLOAD_FILE = "payload.bin"


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def new_scenario_id() -> str:
    return "scn_" + uuid.uuid4().hex[:12]


def new_job_id() -> str:
    return "job_" + uuid.uuid4().hex[:12]


def new_event_id() -> str:
    return "evt_" + uuid.uuid4().hex[:12]


def new_export_id() -> str:
    return "exp_" + uuid.uuid4().hex[:12]


def new_request_id() -> str:
    return "req_" + uuid.uuid4().hex[:12]


def request_fingerprint(
    method: str, path: str, body: bytes | str | None, *, extra: str | None = None
) -> str:
    """Stable fingerprint of a mutation request for idempotency replay.

    Computed over the raw request body bytes so key reuse with any changed
    content — even semantically equal JSON with different key order — is
    detected deterministically. ``extra`` folds request headers that change
    the semantics of the mutation (currently ``If-Match``) into the
    fingerprint, so reusing a key with a different precondition is a
    conflict, not a silent replay.
    """
    raw = body if isinstance(body, (bytes, bytearray)) else (body or "").encode("utf-8")
    preimage = f"{method} {path}\n".encode("utf-8") + raw
    if extra:
        preimage += b"\nIf-Match: " + extra.encode("utf-8")
    digest = hashlib.sha256(preimage).hexdigest()
    return "sha256:" + digest


def uv_to_world(
    u: float,
    v: float,
    *,
    rows: int,
    cols: int,
    pixel_size_m: float,
    origin_x_m: float,
    origin_y_m: float,
) -> tuple[float, float]:
    """Convert site-normalized UV to projected world metres.

    ``(0, 0)`` is the site's top-left (north-west) corner, ``u`` grows east,
    ``v`` grows south, matching ``data_model.md``.
    """
    x_m = origin_x_m + float(u) * cols * pixel_size_m
    y_m = origin_y_m - float(v) * rows * pixel_size_m
    return x_m, y_m


def world_to_uv(
    x_m: float,
    y_m: float,
    *,
    rows: int,
    cols: int,
    pixel_size_m: float,
    origin_x_m: float,
    origin_y_m: float,
) -> tuple[float, float]:
    """Inverse of :func:`uv_to_world` (used for diagnostics)."""
    u = (float(x_m) - origin_x_m) / (cols * pixel_size_m)
    v = (origin_y_m - float(y_m)) / (rows * pixel_size_m)
    return u, v


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class StoreError(RuntimeError):
    """Base class for store failures."""


class ScenarioNotFound(StoreError):
    def __init__(self, scenario_id: str) -> None:
        self.scenario_id = scenario_id
        super().__init__(f"scenario {scenario_id!r} does not exist")


class JobNotFound(StoreError):
    def __init__(self, job_id: str) -> None:
        self.job_id = job_id
        super().__init__(f"job {job_id!r} does not exist")


class SceneVersionConflict(StoreError):
    """The client's base scene version is stale."""

    def __init__(self, current_scene_version: int) -> None:
        self.current_scene_version = int(current_scene_version)
        super().__init__(
            "the scenario changed after the client's base version "
            f"(current scene version {self.current_scene_version})"
        )


class IdempotencyConflict(StoreError):
    """An idempotency key was reused with a different request body."""

    def __init__(self, key: str) -> None:
        self.key = key
        super().__init__(f"idempotency key {key!r} was already used with a different body")


class OperationFingerprintConflict(StoreError):
    """An operation_id was reused with different operation content.

    The realtime plane's per-operation analogue of
    :class:`IdempotencyConflict`: a genuinely retried operation carries the
    same operation-defining content (source family, entity, verb, payload)
    and replays the EXISTING durable record (``duplicate=True``, never a
    second row); the same id with changed content — or under a different
    workspace, since ``operation_id`` is globally unique — is a typed
    refusal. Advisory metadata (``client_sequence``, ``base_revision``) is
    deliberately NOT part of the comparison: it must never become an
    equality gate.
    """

    def __init__(self, operation_id: str) -> None:
        self.operation_id = operation_id
        super().__init__(
            f"operation_id {operation_id!r} was already accepted with "
            "different operation content"
        )


class UnsupportedFamily(StoreError):
    """An operation's source family has no reducer fold mapping AND no
    adapter claim (r2a-review segment-skip adjudication, phased endgame).

    Raised by :meth:`Store.append_operations` BEFORE any durable write
    (reject-before-accept): such an operation used to be accepted and
    fold-skipped by the exact lane later, with metrics+manifest
    disclosure. That disclosure path remains for ALREADY-DURABLE
    historical rows — no retroactive breakage — but new acceptances are
    refused at the door. The refusal applies only where neither source
    claims the family: the reducer's fold registry
    (``realtime/reducer.py``) or the edit-registry adapters
    (``incremental/edit_registry.py``; an adapter-backed family is
    supported even when the reducer's fold of it is segment-skipped).

    ``family`` and :meth:`body` are machine-readable so a route-facing
    translation (a follow-up; the transport already refuses unknown
    families before the store is reached) can answer with a typed 4xx.
    """

    #: Advice a route-facing translation should surface to clients.
    advice = (
        "submit operations only for families the server supports (see "
        "GET /api/v1/capabilities); an adapter-backed family is accepted "
        "even before its exact-lane fold ships, and a refused batch "
        "writes nothing"
    )

    def __init__(self, family: str) -> None:
        self.family = str(family)
        super().__init__(
            f"source_family {self.family!r} is not supported by this server: "
            "it has neither a reducer fold mapping nor an adapter claim"
        )

    def body(self) -> dict[str, Any]:
        """Machine-readable refusal body for a route-facing translation."""
        return {
            "code": "unsupported_family",
            "family": self.family,
            "advice": self.advice,
        }


class IdempotentReplay(StoreError):
    """A stored idempotent response should be returned verbatim."""

    def __init__(self, response: "StoredResponse") -> None:
        self.response = response
        super().__init__("idempotent replay")


class StaleResultError(StoreError):
    """A publish attempted a scene version other than the current one."""

    def __init__(self, attempted: int, current: int) -> None:
        self.attempted = int(attempted)
        self.current = int(current)
        super().__init__(
            f"refusing to publish scene version {self.attempted}: the scenario is at "
            f"version {self.current} (stale results must never be published)"
        )


class ResultAlreadyPublished(StoreError):
    pass


class ResultNotPublishable(StoreError):
    """The job a result belongs to is no longer running (cancelled mid-flight).

    Raised by :meth:`Store.publish_result` when the job-status check inside
    the publish transaction observes a terminal job: the runner's pre-publish
    liveness check and the publish itself are separate transactions, so a
    cancel can land in between. The result is a discard by contract — never
    published for a cancelled job.
    """

    def __init__(self, job_id: str, status: str) -> None:
        self.job_id = str(job_id)
        self.status = str(status)
        super().__init__(
            f"job {self.job_id} is {self.status!r}, not running: its result "
            "is discarded (a cancel landed before the publish committed)"
        )


class SiteIdentityMismatch(StoreError):
    """The request's site geometry differs from the scenario's pinned
    identity (U-D intake item g): pixel-identical but geographically
    different data may not ride a live scenario."""

    def __init__(self, expected: Mapping[str, Any], actual: Mapping[str, Any]) -> None:
        self.expected = dict(expected)
        self.actual = dict(actual)
        differing = {
            key: {"pinned": self.expected.get(key), "current": self.actual.get(key)}
            for key in sorted(set(self.expected) | set(self.actual))
            if self.expected.get(key) != self.actual.get(key)
        }
        super().__init__(
            "the site data serving this scenario no longer matches the "
            f"identity pinned at creation: {differing}"
        )


class EpochBaselineStale(StoreError):
    """The canonical baseline an epoch was folded from is no longer the
    latest persisted revision at commit time (r1-epochs-review M2).

    A concurrent driver committed another epoch's reduction between this
    close's baseline read and its commit. Committing anyway would key the
    stale fold to a NEW revision and silently drop the other epoch's
    effects from the latest canonical state — refuse, and let the caller
    re-fold on the fresh baseline.
    """

    def __init__(self, workspace_id: str, expected: int, actual: int) -> None:
        self.workspace_id = workspace_id
        self.expected_baseline_revision = int(expected)
        self.actual_max_revision = int(actual)
        super().__init__(
            f"canonical baseline for {workspace_id!r} advanced during the "
            f"epoch close (folded from {expected}, latest is {actual}); "
            "re-fold on the fresh baseline"
        )


class ScenarioQuotaExceeded(StoreError):
    """The active-scenario cap was reached (authoritative, in-transaction)."""

    def __init__(self, max_scenarios: int) -> None:
        self.max_scenarios = int(max_scenarios)
        super().__init__(
            f"the maximum number of active scenarios ({self.max_scenarios}) "
            "for this server has been reached"
        )


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScenarioRecord:
    scenario_id: str
    site_id: str
    name: str
    scene_version: int
    exact_result_version: int
    status: str
    edit_sequence: int = 0
    acked_sequence: int = 0
    created_at: str = ""
    updated_at: str = ""
    #: Geographic identity pinned at creation (U-D intake item g);
    #: ``None`` for legacy rows pinned before this column existed.
    site_identity: Mapping[str, Any] | None = None
    #: Revision-triple members (R1, migration v7). ``scene_version`` above
    #: IS the workspace revision — not renamed, re-keyed semantically; these
    #: two complete the collaborative_state.md revision model
    #: (``exact_revision`` is the existing ``exact_result_version``).
    fast_revision: int = 0
    exact_base_revision: int = 0

    @property
    def exact(self) -> bool:
        return self.exact_result_version >= self.scene_version


@dataclass(frozen=True)
class JobRecord:
    job_id: str
    scenario_id: str
    target_scene_version: int
    status: str
    stage: str | None = None
    progress: dict[str, int] | None = None
    mode: str | None = None
    window: dict[str, int] | None = None
    request: dict[str, Any] | None = None
    edit_watermark: int = 0
    #: Store sequence watermark already consumed by a *published* result when
    #: this job was created. The solver replays ledger events with
    #: ``sequence <= base_edit_watermark`` into its consumed-batch watermark,
    #: so ``pending`` covers exactly this job's (and any superseded job's
    #: unconsumed) edits.
    base_edit_watermark: int = 0
    queued_at: str = ""
    started_at: str | None = None
    finished_at: str | None = None
    worker_revision: str | None = None
    metrics: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    result_scene_version: int | None = None
    #: The executed impact plan this job actually ran under (u-d4 plan
    #: exposure): per-node stages, realized routing, estimates vs actuals.
    #: Present only for jobs that ran through the executor bridge (family
    #: edits); tree-only jobs leave it NULL and clients derive stages.
    plan: dict[str, Any] | None = None


@dataclass(frozen=True)
class ResultRecord:
    scenario_id: str
    scene_version: int
    exact: bool
    manifest: dict[str, Any]
    payload_path: str
    checksum: str
    created_at: str = ""
    job_id: str | None = None

    def payload_bytes(self) -> bytes:
        return Path(self.payload_path).read_bytes()


@dataclass(frozen=True)
class StoredResponse:
    status_code: int
    body: Any
    headers: dict[str, str] = field(default_factory=dict)

    def json(self) -> str:
        return json.dumps(
            {"status_code": self.status_code, "body": self.body, "headers": self.headers}
        )

    @classmethod
    def from_json(cls, text: str) -> "StoredResponse":
        data = json.loads(text)
        return cls(
            status_code=int(data["status_code"]),
            body=data["body"],
            headers=dict(data.get("headers", {})),
        )


@dataclass(frozen=True)
class OperationRecord:
    """One durably accepted realtime operation (``realtime_operations`` row).

    ``duplicate`` is True on an idempotent replay: the record returned is
    the EXISTING durable row (same ``server_sequence``/``accepted_at``),
    never a second accepted effect. ``base_revision`` is advisory metadata
    exactly as stored — nothing in the store gates on it.
    """

    workspace_id: str
    operation_id: str
    server_sequence: int
    epoch_id: int
    actor_id: str
    client_sequence: int | None
    base_revision: int | None
    source_family: str
    entity_id: str | None
    verb: str
    payload: Mapping[str, Any]
    received_at: str
    accepted_at: str
    duplicate: bool = False


@dataclass(frozen=True)
class EpochRecord:
    """One per-workspace epoch row (``realtime_epochs``)."""

    workspace_id: str
    epoch_id: int
    status: str
    first_sequence: int | None
    last_sequence: int | None
    workspace_revision: int | None
    opened_at: str
    closed_at: str | None


@dataclass(frozen=True)
class CommittedEpoch:
    """Result of :meth:`Store.commit_epoch_reduction`.

    ``first_time`` is False on an idempotent re-drive (revision already
    assigned): the caller re-publishes but must not re-fold or re-bump —
    the original canonical state bytes stay authoritative.
    """

    workspace_id: str
    epoch_id: int
    workspace_revision: int
    first_time: bool


@dataclass(frozen=True)
class FastRevisionAdvance:
    """Result of :meth:`Store.advance_fast_revision`.

    ``exact_revision`` is the ``exact_result_version`` read INSIDE the
    advance transaction (the value the monotonic clamp actually used);
    ``fast_revision`` is the new watermark. ``None`` from the method means
    the stale fence refused the write.
    """

    workspace_id: str
    revision: int
    fast_revision: int
    exact_revision: int


# ---------------------------------------------------------------------------
# Schema migrations
# ---------------------------------------------------------------------------

_MIGRATION_1 = """
CREATE TABLE scenarios (
    scenario_id          TEXT PRIMARY KEY,
    site_id              TEXT NOT NULL,
    name                 TEXT NOT NULL DEFAULT '',
    scene_version        INTEGER NOT NULL DEFAULT 0,
    exact_result_version INTEGER NOT NULL DEFAULT 0,
    status               TEXT NOT NULL DEFAULT 'refining',
    edit_sequence        INTEGER NOT NULL DEFAULT 0,
    acked_sequence       INTEGER NOT NULL DEFAULT 0,
    created_at           TEXT NOT NULL,
    updated_at           TEXT NOT NULL
);

CREATE TABLE scenario_trees (
    scenario_id TEXT NOT NULL,
    tree_id     TEXT NOT NULL,
    tree_json   TEXT NOT NULL,
    PRIMARY KEY (scenario_id, tree_id)
);

CREATE TABLE edit_events (
    scenario_id       TEXT NOT NULL,
    sequence          INTEGER NOT NULL,
    event_id          TEXT NOT NULL,
    base_scene_version INTEGER NOT NULL,
    operation         TEXT NOT NULL,
    tree_id           TEXT,
    old_tree_json     TEXT,
    new_tree_json     TEXT,
    submitted_at      TEXT NOT NULL,
    idempotency_key   TEXT,
    PRIMARY KEY (scenario_id, sequence)
);

CREATE TABLE idempotency_keys (
    scope               TEXT NOT NULL,
    key                 TEXT NOT NULL,
    request_fingerprint TEXT NOT NULL,
    response_json       TEXT NOT NULL,
    created_at          TEXT NOT NULL,
    PRIMARY KEY (scope, key)
);

CREATE TABLE jobs (
    job_id               TEXT PRIMARY KEY,
    scenario_id          TEXT NOT NULL,
    target_scene_version INTEGER NOT NULL,
    status               TEXT NOT NULL DEFAULT 'queued',
    stage                TEXT,
    progress_json        TEXT,
    mode                 TEXT,
    window_json          TEXT,
    request_json         TEXT,
    edit_watermark       INTEGER NOT NULL DEFAULT 0,
    queued_at            TEXT NOT NULL,
    started_at           TEXT,
    finished_at          TEXT,
    worker_revision      TEXT,
    metrics_json         TEXT,
    error_json           TEXT,
    result_scene_version INTEGER
);
CREATE INDEX idx_jobs_scenario_status ON jobs (scenario_id, status);

CREATE TABLE results (
    scenario_id   TEXT NOT NULL,
    scene_version INTEGER NOT NULL,
    exact         INTEGER NOT NULL,
    manifest_json TEXT NOT NULL,
    payload_path  TEXT NOT NULL,
    checksum      TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    job_id        TEXT,
    PRIMARY KEY (scenario_id, scene_version)
);

CREATE TABLE exports (
    export_id         TEXT PRIMARY KEY,
    scenario_id       TEXT NOT NULL,
    scene_version     INTEGER NOT NULL,
    format            TEXT NOT NULL,
    variables_json    TEXT,
    time_indices_json TEXT,
    status            TEXT NOT NULL DEFAULT 'accepted',
    created_at        TEXT NOT NULL
);
"""

#: v1 -> v2: jobs gain ``base_edit_watermark`` (the store sequence watermark
#: already consumed by a published result when the job was created), which
#: the solver needs so its pending batch contains this job's edits.
_MIGRATION_2 = """
ALTER TABLE jobs ADD COLUMN base_edit_watermark INTEGER NOT NULL DEFAULT 0;
"""

#: v2 -> v3 (U-D intake item g): scenarios gain ``site_identity_json`` —
#: the geographic identity (site id, rows, cols, pixel size, ORIGIN
#: x/y) pinned at scenario creation. The grid guard previously checked
#: rows/cols/pixel_size only, so pixel-identical but geographically
#: DIFFERENT data (a swapped site cache with a different origin) could be
#: pointed at a live scenario; the pin closes that hole with a typed 409
#: on every subsequent mutation. NULL = a legacy row created before the
#: pin; it is pinned lazily on its next verified request.
_MIGRATION_3 = """
ALTER TABLE scenarios ADD COLUMN site_identity_json TEXT;
"""

#: v3 -> v4 (u-d4 universal edit transport): ``edit_events`` gains
#: ``family_json`` — the verbatim universal-edit payload (adapter, operation,
#: values, target, time_index, old_values) for events that are NOT tree
#: edits. Tree events keep NULL and their existing columns; the solver
#: dispatches a scenario onto the executor bridge as soon as any event
#: carries a family payload. ``jobs`` gains ``plan_json`` — the serialized
#: executed impact plan the job actually ran under (u-d4 plan exposure).
_MIGRATION_4 = """
ALTER TABLE edit_events ADD COLUMN family_json TEXT;
ALTER TABLE jobs ADD COLUMN plan_json TEXT;
"""

#: v4 -> v5 (u-e1 F1 durable routing): scenarios gain
#: ``carries_family_edits`` — a durable routing signal SET atomically with
#: the family-event insert in ``commit_universal_edits`` and CLEARED by
#: ``reset_scenario``. The routing predicate previously inferred
#: "this scenario carries family state" by scanning the ledger events, but
#: ``sweep_retention`` prunes events past the retention tail; once every
#: family event was pruned a later TREE edit re-routed to the legacy
#: baseline-bound solver, which publishes WITHOUT the family state (a
#: bitwise-wrong scene). The flag outlives the events: it is the memory
#: that survives pruning. The backfill UPDATE flags every scenario with a
#: surviving family event so a migrated deployment never under-routes;
#: scenarios whose family events were pruned BEFORE the upgrade keep legacy
#: routing until their next family edit (disclosed in
#: ``deployment_operations.md``), and ``scenario_carries_family_edits``
#: additionally ORs in any surviving family event.
_MIGRATION_5 = """
ALTER TABLE scenarios ADD COLUMN carries_family_edits INTEGER NOT NULL DEFAULT 0;
UPDATE scenarios SET carries_family_edits = 1 WHERE scenario_id IN (
    SELECT DISTINCT scenario_id FROM edit_events WHERE family_json IS NOT NULL
);
"""

#: v5 -> v6 (u-e3c attack C): scenarios gain ``last_reset_sequence`` — the
#: store sequence of the scenario's most recent reset, durable and
#: retention-immune like ``carries_family_edits`` (``sweep_retention``
#: prunes the reset EVENT itself once enough later edits ack past it, so
#: the ledger cannot be consulted). ``reset_scenario`` writes it going
#: forward. The routing predicate's executor-state rescue
#: (``scenario_carries_family_edits``) needs it: a pre-v5 row whose family
#: events were pruned BEFORE the upgrade keeps ``carries_family_edits=0``
#: and has no surviving family event — the v5 backfill cannot see state
#: that no longer exists — yet its executor-state directory proves the
#: scenario ran family jobs. The rescue arms only when that state
#: POSTDATES the reset watermark: a pre-reset snapshot must never
#: resurrect (the reset voided it). The backfill reads surviving reset
#: events; scenarios whose reset events were already pruned keep 0 and
#: are disclosed in ``deployment_operations.md`` (an operator reset after
#: upgrading re-arms the watermark).
_MIGRATION_6 = """
ALTER TABLE scenarios ADD COLUMN last_reset_sequence INTEGER NOT NULL DEFAULT 0;
UPDATE scenarios SET last_reset_sequence = (
    SELECT MAX(sequence) FROM edit_events
    WHERE edit_events.scenario_id = scenarios.scenario_id
      AND operation = 'reset'
) WHERE EXISTS (
    SELECT 1 FROM edit_events
    WHERE edit_events.scenario_id = scenarios.scenario_id
      AND operation = 'reset'
);
"""

#: v6 -> v7 (R1 realtime collaborative operation plane,
#: ``realtime_collaboration/collaborative_state.md`` + ``epoch_scheduler.md``):
#: three additive pieces, zero behavior change to existing tables/paths.
#:
#: 1. ``realtime_operations`` — the durable append-only operation audit.
#:    One row per ACCEPTED operation with the globally-UNIQUE ``operation_id``
#:    as the idempotency key (a retry never produces a second row; a reuse
#:    with different operation content is a typed conflict, never a silent
#:    second effect). ``server_sequence`` is per-workspace monotonic and
#:    CONTIGUOUS (the ``edit_events.sequence`` discipline, in its own
#:    ledger), so epoch reduction reads a gap-free prefix.
#:
#: 2. ``realtime_epochs`` — the per-workspace epoch state machine rows
#:    (``types.EpochStatus`` vocabulary). Ops are assigned to the
#:    workspace's currently-open epoch; ``workspace_revision`` is assigned
#:    at close (R2 owns closure semantics — ``mark_epoch_status`` is the
#:    only primitive here, there is NO ``close_epoch`` in this wave).
#:
#: 3. ``scenarios`` gains the revision-triple columns ``fast_revision`` and
#:    ``exact_base_revision`` (collaborative_state.md revision model).
#:    ``scene_version`` BECOMES ``workspace_revision`` semantically — the
#:    column is NOT renamed: every legacy solver/result path keyed on
#:    ``scene_version`` keeps its meaning, and the realtime plane reads it
#:    as the canonical workspace revision. Invariant maintained downstream:
#:    ``exact_revision <= fast_revision <= workspace_revision`` where
#:    ``exact_revision`` is the existing ``exact_result_version``.
#:
#: Retention: ``realtime_operations`` is APPEND-ONLY and is deliberately NOT
#: part of ``sweep_retention`` — the v5/v6 bug precedent (pruning a ledger
#: the routing logic consults forced retention-immune flags as repairs)
#: applies doubly here, where the audit IS the canonical history every
#: epoch reduction replays from. No pruning API exists for this table.
#:
#: Restart recovery: on Store open nothing auto-closes. Epochs whose status
#: is not terminal (``fast_published`` / ``exact_targeted``) remain
#: queryable exactly as recorded; R2's recovery owns closing/replaying
#: incomplete epochs (``epoch_scheduler.md`` "Recovery").
_MIGRATION_7 = """
CREATE TABLE realtime_operations (
    workspace_id    TEXT NOT NULL,
    server_sequence INTEGER NOT NULL,
    epoch_id        INTEGER NOT NULL,
    operation_id    TEXT NOT NULL UNIQUE,
    actor_id        TEXT NOT NULL,
    client_sequence INTEGER,
    base_revision   INTEGER,
    source_family   TEXT NOT NULL,
    entity_id       TEXT,
    verb            TEXT NOT NULL,
    payload_json    TEXT NOT NULL,
    received_at     TEXT NOT NULL,
    accepted_at     TEXT NOT NULL,
    PRIMARY KEY (workspace_id, server_sequence)
);
CREATE INDEX idx_realtime_operations_epoch
    ON realtime_operations (workspace_id, epoch_id);

CREATE TABLE realtime_epochs (
    workspace_id      TEXT NOT NULL,
    epoch_id          INTEGER NOT NULL,
    status            TEXT NOT NULL DEFAULT 'open',
    first_sequence    INTEGER,
    last_sequence     INTEGER,
    workspace_revision INTEGER,
    opened_at         TEXT NOT NULL,
    closed_at         TEXT,
    PRIMARY KEY (workspace_id, epoch_id)
);

ALTER TABLE scenarios ADD COLUMN fast_revision INTEGER NOT NULL DEFAULT 0;
ALTER TABLE scenarios ADD COLUMN exact_base_revision INTEGER NOT NULL DEFAULT 0;
"""

#: v7 -> v8 (R1 epoch scheduler wave, ``realtime_collaboration/
#: epoch_scheduler.md`` "Close pipeline" + "Recovery"): one additive table,
#: zero changes to existing tables.
#:
#: ``realtime_canonical_state`` — the per-workspace canonical family-state
#: snapshots, one row per assigned ``workspace_revision`` (PK
#: ``(workspace_id, workspace_revision)``). Every non-empty epoch close
#: persists the reducer's folded state here in the SAME transaction that
#: assigns the epoch's revision and bumps ``scenarios.scene_version`` (which
#: IS ``workspace_revision`` semantically — see the v7 note). The next
#: epoch's reduction folds from the latest row, so cross-epoch state
#: threading is durable, never in-memory.
#:
#: Idempotency: the insert is ``INSERT OR IGNORE`` and the revision
#: assignment guards on the epoch row's existing ``workspace_revision`` — a
#: re-driven close (crash-recovery republish, double tick) can never bump
#: twice nor overwrite the ORIGINAL state bytes with a re-fold from a newer
#: baseline. This table is append-only history (epoch N's fold stays
#: byte-identical for replay determinism) and, like
#: ``realtime_operations``, is deliberately outside ``sweep_retention``.
_MIGRATION_8 = """
CREATE TABLE realtime_canonical_state (
    workspace_id       TEXT NOT NULL,
    workspace_revision INTEGER NOT NULL,
    state_json         TEXT NOT NULL,
    created_at         TEXT NOT NULL,
    PRIMARY KEY (workspace_id, workspace_revision)
);
"""

#: v8 -> v9 (routing policy wave — idle reconciliation bookkeeping):
#: ``workspace_reconcile`` — per-workspace once-daily-full bookkeeping.
#: One row per workspace that has ever deferred or completed a
#: reconciliation full. ``last_completed_date`` is the UTC date of the
#: last COMPLETED reconcile (a SUPERSEDED attempt consumes nothing — only
#: completion owns a day); ``owed`` is the debt flag set when the exact
#: lane deferred a recoverable revision gap because no subscriber was
#: watching, persisting across restarts and day boundaries until a
#: reconcile discharges it.
_MIGRATION_9 = """
CREATE TABLE workspace_reconcile (
    workspace_id        TEXT PRIMARY KEY,
    last_completed_date TEXT NOT NULL,
    owed                INTEGER NOT NULL DEFAULT 0,
    updated_at          TEXT NOT NULL
);
"""

#: Sequential migration list; index i upgrades schema version i -> i + 1.
_MIGRATIONS: tuple[str, ...] = (
    _MIGRATION_1,
    _MIGRATION_2,
    _MIGRATION_3,
    _MIGRATION_4,
    _MIGRATION_5,
    _MIGRATION_6,
    _MIGRATION_7,
    _MIGRATION_8,
    _MIGRATION_9,
)

#: Schema version this code understands; a store reporting a HIGHER version
#: was written by newer code and must not be served (readiness check).
LATEST_SCHEMA_VERSION = len(_MIGRATIONS)

# Ledger retention: ``Store.sweep_retention`` deletes idempotency keys older
# than a ~24 h retry window and edit events below each scenario's
# ``acked_sequence`` minus a bounded audit tail. The JobRunner invokes it
# opportunistically (throttled to once per hour) instead of truncating live
# ledgers.

#: Cache for :func:`_supported_source_families`.
_SUPPORTED_SOURCE_FAMILIES: frozenset[str] | None = None


def _supported_source_families() -> frozenset[str]:
    """Source families the realtime plane accepts at operation-accept time.

    The union of the reducer's fold registry (``realtime/reducer.py``'s
    known-family set — the canonical-state fold) and the edit registry's
    adapter ids (``incremental/edit_registry.py``): an adapter-backed
    family is supported even when the reducer's fold of it is
    segment-skipped by the exact lane, because the fold-skip disclosure
    path covers it. Anything outside the union has neither a fold nor an
    adapter and is refused at accept (:class:`UnsupportedFamily`).
    """
    global _SUPPORTED_SOURCE_FAMILIES
    if _SUPPORTED_SOURCE_FAMILIES is None:
        from solweig_gpu.incremental.edit_registry import builtin_adapter_metadata
        from solweig_gpu.server.realtime import reducer

        _SUPPORTED_SOURCE_FAMILIES = frozenset(reducer._ALL_FAMILIES) | frozenset(
            metadata.id for metadata in builtin_adapter_metadata()
        )
    return _SUPPORTED_SOURCE_FAMILIES


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class Store:
    """Thread-safe SQLite persistence plus on-disk result payloads."""

    db_path: Path
    results_root: Path

    def __init__(self, db_path: str | Path, *, results_root: str | Path) -> None:
        self.db_path = Path(db_path)
        self.results_root = Path(results_root)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.results_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self.db_path), check_same_thread=False, timeout=30.0
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.isolation_level = None  # explicit transactions
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._migrate()

    # -- infrastructure ---------------------------------------------------

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _migrate(self) -> None:
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)"
            )
            row = self._conn.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
            current = int(row["v"]) if row is not None and row["v"] is not None else 0
            for version, script in enumerate(_MIGRATIONS, start=1):
                if version <= current:
                    continue
                self._conn.execute("BEGIN IMMEDIATE")
                try:
                    # sqlite3.executescript() would implicitly COMMIT the
                    # open transaction, so run the statements one by one to
                    # keep each migration atomic.
                    for statement in script.split(";"):
                        if statement.strip():
                            self._conn.execute(statement)
                    self._conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
                    self._conn.execute("COMMIT")
                except BaseException:
                    self._conn.execute("ROLLBACK")
                    raise
            self.schema_version = max(current, len(_MIGRATIONS))

    @contextmanager
    def _write(self):
        """One serialized write transaction."""
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    def _read(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(sql, params).fetchone()

    def _read_all(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    # -- scenarios ----------------------------------------------------------

    @staticmethod
    def _scenario_row(row: sqlite3.Row) -> ScenarioRecord:
        pinned = row["site_identity_json"] if "site_identity_json" in row.keys() else None
        keys = row.keys()
        return ScenarioRecord(
            scenario_id=row["scenario_id"],
            site_id=row["site_id"],
            name=row["name"],
            scene_version=int(row["scene_version"]),
            exact_result_version=int(row["exact_result_version"]),
            status=row["status"],
            edit_sequence=int(row["edit_sequence"]),
            acked_sequence=int(row["acked_sequence"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            site_identity=(
                json.loads(pinned)
                if isinstance(pinned, str) and pinned
                else None
            ),
            fast_revision=int(row["fast_revision"]) if "fast_revision" in keys else 0,
            exact_base_revision=(
                int(row["exact_base_revision"]) if "exact_base_revision" in keys else 0
            ),
        )

    def get_scenario(self, scenario_id: str) -> ScenarioRecord | None:
        row = self._read("SELECT * FROM scenarios WHERE scenario_id = ?", (scenario_id,))
        return self._scenario_row(row) if row is not None else None

    def require_scenario(self, scenario_id: str) -> ScenarioRecord:
        record = self.get_scenario(scenario_id)
        if record is None:
            raise ScenarioNotFound(scenario_id)
        return record

    def count_scenarios(self) -> int:
        row = self._read("SELECT COUNT(*) AS n FROM scenarios")
        return int(row["n"]) if row is not None else 0

    def current_scene_version(self, scenario_id: str) -> int:
        return self.require_scenario(scenario_id).scene_version

    def list_trees(self, scenario_id: str) -> list[dict[str, Any]]:
        rows = self._read_all(
            "SELECT tree_json FROM scenario_trees WHERE scenario_id = ? ORDER BY tree_id",
            (scenario_id,),
        )
        return [json.loads(row["tree_json"]) for row in rows]

    def list_events(
        self, scenario_id: str, *, after_sequence: int = 0
    ) -> list[dict[str, Any]]:
        rows = self._read_all(
            "SELECT * FROM edit_events WHERE scenario_id = ? AND sequence > ? "
            "ORDER BY sequence",
            (scenario_id, int(after_sequence)),
        )
        return [
            {
                "sequence": int(row["sequence"]),
                "event_id": row["event_id"],
                "base_scene_version": int(row["base_scene_version"]),
                "operation": row["operation"],
                "tree_id": row["tree_id"],
                "old_tree": json.loads(row["old_tree_json"]) if row["old_tree_json"] else None,
                "new_tree": json.loads(row["new_tree_json"]) if row["new_tree_json"] else None,
                "submitted_at": row["submitted_at"],
                "family": json.loads(row["family_json"]) if row["family_json"] else None,
            }
            for row in rows
        ]

    def sweep_retention(
        self,
        *,
        idempotency_ttl_hours: float = 24.0,
        event_tail: int = 1000,
        now: str | None = None,
    ) -> dict[str, int]:
        """Apply bounded-retention deletes to the append-only ledgers.

        * ``idempotency_keys``: entries older than the TTL (contract retry
          window ~24 h) are deleted — a key replayed after deletion simply
          re-executes, which is the documented behaviour once the retry
          window has closed.
        * ``edit_events``: events with ``sequence <= acked_sequence`` have
          been consumed by a published result; beyond an audit ``tail`` per
          scenario they are deleted. Events above the watermark are always
          kept (coalescing still needs them).

        Returns per-table delete counts. Never truncates live data.
        """
        from datetime import datetime, timedelta, timezone

        now_dt = (
            datetime.fromisoformat(now) if now is not None else datetime.now(timezone.utc)
        )
        if now_dt.tzinfo is None:
            now_dt = now_dt.replace(tzinfo=timezone.utc)
        # Format the cutoff exactly like _now_utc() (millisecond precision,
        # "Z" suffix): created_at is compared lexicographically in SQL, and
        # isoformat()'s "+00:00"+microseconds would skew the comparison.
        cutoff_dt = now_dt - timedelta(hours=idempotency_ttl_hours)
        cutoff = (
            cutoff_dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        )
        deleted = {"idempotency_keys": 0, "edit_events": 0}
        with self._write() as conn:
            cursor = conn.execute(
                "DELETE FROM idempotency_keys WHERE created_at < ?",
                (cutoff,),
            )
            deleted["idempotency_keys"] = cursor.rowcount or 0
            scenarios = conn.execute(
                "SELECT scenario_id, acked_sequence FROM scenarios"
            ).fetchall()
            for row in scenarios:
                watermark = int(row["acked_sequence"])
                cursor = conn.execute(
                    "DELETE FROM edit_events WHERE scenario_id = ? AND "
                    "sequence <= ? - ?",
                    (row["scenario_id"], watermark, int(event_tail)),
                )
                deleted["edit_events"] += cursor.rowcount or 0
        return deleted

    # -- idempotency ----------------------------------------------------------

    def lookup_idempotency(self, scope: str, key: str) -> StoredResponse | None:
        row = self._read(
            "SELECT response_json FROM idempotency_keys WHERE scope = ? AND key = ?",
            (scope, key),
        )
        return StoredResponse.from_json(row["response_json"]) if row is not None else None

    def check_idempotency(
        self, scope: str, key: str, fingerprint: str
    ) -> StoredResponse | None:
        """Read-only replay check: stored response, ``None``, or a conflict.

        Raises :class:`IdempotencyConflict` when the key exists with a
        different request fingerprint; returns the stored response for an
        exact match; ``None`` when the key is unknown. The mutating store
        methods repeat this check inside their transaction, which stays
        authoritative for races.
        """
        row = self._read(
            "SELECT request_fingerprint, response_json FROM idempotency_keys "
            "WHERE scope = ? AND key = ?",
            (scope, key),
        )
        if row is None:
            return None
        if row["request_fingerprint"] != fingerprint:
            raise IdempotencyConflict(key)
        return StoredResponse.from_json(row["response_json"])

    @staticmethod
    def _check_idempotency(
        conn: sqlite3.Connection,
        scope: str,
        key: str,
        fingerprint: str,
    ) -> None:
        """Raise replay/conflict if the key is already stored (inside a txn)."""
        row = conn.execute(
            "SELECT request_fingerprint, response_json FROM idempotency_keys "
            "WHERE scope = ? AND key = ?",
            (scope, key),
        ).fetchone()
        if row is None:
            return
        if row["request_fingerprint"] != fingerprint:
            raise IdempotencyConflict(key)
        raise IdempotentReplay(StoredResponse.from_json(row["response_json"]))

    @staticmethod
    def _store_idempotency(
        conn: sqlite3.Connection,
        scope: str,
        key: str,
        fingerprint: str,
        response: StoredResponse,
    ) -> None:
        conn.execute(
            "INSERT INTO idempotency_keys (scope, key, request_fingerprint, "
            "response_json, created_at) VALUES (?, ?, ?, ?, ?)",
            (scope, key, fingerprint, response.json(), _now_utc()),
        )

    # -- scenario creation ------------------------------------------------

    def create_scenario(
        self,
        *,
        site_id: str,
        name: str,
        scenario_id: str | None = None,
        idempotency_key: str | None = None,
        request_fingerprint: str | None = None,
        max_scenarios: int | None = None,
        site_identity: Mapping[str, Any] | None = None,
        baseline_publisher: Callable[[str, int], tuple[dict[str, Any], bytes]] | None = None,
        response_builder: Callable[[ScenarioRecord, str | None], StoredResponse] | None = None,
    ) -> tuple[ScenarioRecord, str | None]:
        """Create a scenario; optionally publish the baseline result inline.

        ``baseline_publisher(scenario_id, scene_version=0)`` returns
        ``(manifest, payload_bytes)`` from the site cache baseline outputs. When
        it is omitted (or returns without being called) the scenario starts in
        ``refining`` status and the returned sentinel asks the caller to enqueue
        a baseline full-solve job. ``max_scenarios`` is enforced inside this
        transaction (the authoritative quota check — the app-level limiter is
        only a fast pre-check), raising :class:`ScenarioQuotaExceeded`.
        ``site_identity`` (U-D intake item g) pins the site's geographic
        identity (site id + grid + ORIGIN x/y) at creation; later mutations
        verify against it through :meth:`verify_site_identity`.
        """
        scenario_id = scenario_id or new_scenario_id()
        scope = "_create"
        with self._write() as conn:
            if idempotency_key and request_fingerprint:
                self._check_idempotency(conn, scope, idempotency_key, request_fingerprint)
            if max_scenarios is not None:
                count = int(
                    conn.execute("SELECT COUNT(*) AS n FROM scenarios").fetchone()["n"]
                )
                if count >= int(max_scenarios):
                    raise ScenarioQuotaExceeded(int(max_scenarios))
            now = _now_utc()
            conn.execute(
                "INSERT INTO scenarios (scenario_id, site_id, name, scene_version, "
                "exact_result_version, status, created_at, updated_at, "
                "site_identity_json) "
                "VALUES (?, ?, ?, 0, 0, 'refining', ?, ?, ?)",
                (
                    scenario_id,
                    site_id,
                    name,
                    now,
                    now,
                    json.dumps(dict(site_identity), sort_keys=True)
                    if site_identity is not None
                    else None,
                ),
            )
            job_id: str | None = None
            if baseline_publisher is not None:
                manifest, payload = baseline_publisher(scenario_id, 0)
                self._publish_result_locked(
                    conn,
                    scenario_id,
                    0,
                    manifest=manifest,
                    payload=payload,
                    exact=True,
                    job_id=None,
                    acked_sequence=0,
                )
            else:
                job_id = self._insert_job_locked(
                    conn,
                    scenario_id,
                    target_scene_version=0,
                    request={"baseline": True},
                    edit_watermark=0,
                    base_edit_watermark=0,
                )
            record = self._require_scenario_locked(conn, scenario_id)
            response = (
                response_builder(record, job_id)
                if response_builder is not None
                else StoredResponse(201, {"scenario_id": scenario_id})
            )
            if idempotency_key and request_fingerprint:
                self._store_idempotency(conn, scope, idempotency_key, request_fingerprint, response)
        return record, job_id

    def verify_site_identity(
        self, scenario_id: str, current: Mapping[str, Any]
    ) -> ScenarioRecord:
        """Verify (and lazily pin) the scenario's site identity (item g).

        ``current`` is the LIVE site geometry from the site registry. A
        scenario pinned at creation compares site id, grid dimensions,
        pixel size, AND geographic origin against it; any difference
        raises :class:`SiteIdentityMismatch` — the typed refusal that
        stops pixel-identical-but-geographically-different data from
        riding a live scenario. Legacy rows created before the pin
        (``site_identity_json`` NULL) are pinned now, lazily, from the
        live geometry: their first post-upgrade request is by definition
        the identity they have been running under.

        Pure verification (a pinned scenario, matching or not) takes NO
        write transaction: the pin is read and compared outside any
        ``BEGIN IMMEDIATE``. Only the lazy pin — a NULL-pin legacy row —
        needs the write path.
        """
        # Read-only fast path: an existing pin answers without a write
        # transaction (verification is by far the common case — every
        # edit/reset request — and must not serialize behind writers).
        row = self._read(
            "SELECT * FROM scenarios WHERE scenario_id = ?", (scenario_id,)
        )
        if row is not None:
            scenario = self._scenario_row(row)
            if scenario.site_identity is not None:
                pinned = dict(scenario.site_identity)
                if self._identity_differs(pinned, current):
                    raise SiteIdentityMismatch(pinned, dict(current))
                return scenario
        with self._write() as conn:
            scenario = self._require_scenario_locked(conn, scenario_id)
            if scenario.site_identity is None:
                conn.execute(
                    "UPDATE scenarios SET site_identity_json = ?, updated_at = ? "
                    "WHERE scenario_id = ?",
                    (
                        json.dumps(dict(current), sort_keys=True),
                        _now_utc(),
                        scenario_id,
                    ),
                )
                return self._require_scenario_locked(conn, scenario_id)
            pinned = dict(scenario.site_identity)
            if self._identity_differs(pinned, current):
                raise SiteIdentityMismatch(pinned, dict(current))
            return scenario

    #: The identity fields the pin covers: the registry's site id plus the
    #: full grid geometry INCLUDING the geographic origin and the time
    #: axis length (``time_steps`` — a cache regenerated with a different
    #: number of hours is a different site) — the fields the old grid
    #: guard never checked. Fields a legacy pin never recorded are
    #: skipped by :meth:`_identity_differs`, so pre-``time_steps`` pins
    #: keep verifying without false conflicts.
    _IDENTITY_FIELDS = (
        "site_id",
        "rows",
        "cols",
        "pixel_size_m",
        "origin_x_m",
        "origin_y_m",
        "time_steps",
    )

    @classmethod
    def _identity_differs(
        cls, pinned: Mapping[str, Any], current: Mapping[str, Any]
    ) -> bool:
        for field in cls._IDENTITY_FIELDS:
            expected = pinned.get(field)
            if expected is None:
                continue  # a field the pin never recorded cannot conflict
            value = current.get(field)
            if isinstance(expected, bool) != isinstance(value, bool):
                return True
            if isinstance(expected, (int, float)) and isinstance(value, (int, float)):
                if float(expected) != float(value):
                    return True
            elif expected != value:
                return True
        return False

    # -- edits ---------------------------------------------------------------

    def commit_edits(
        self,
        scenario_id: str,
        *,
        base_scene_version: int,
        applied_edits: Sequence[dict[str, Any]],
        requested: dict[str, Any] | None,
        idempotency_key: str | None = None,
        request_fingerprint: str | None = None,
        response_builder: Callable[[ScenarioRecord, str], StoredResponse] | None = None,
    ) -> tuple[ScenarioRecord, str]:
        """Validate, persist, and version one edit batch atomically.

        ``applied_edits`` entries carry ``operation``, ``tree_id``, ``old_tree``
        and ``new_tree`` (API-shaped JSON or ``None``) and were validated by the
        route against the authoritative tree state. Raises
        :class:`SceneVersionConflict` when ``base_scene_version`` is stale and
        idempotency errors for replays.
        """
        with self._write() as conn:
            if idempotency_key and request_fingerprint:
                self._check_idempotency(conn, scenario_id, idempotency_key, request_fingerprint)
            scenario = self._require_scenario_locked(conn, scenario_id)
            if int(base_scene_version) != scenario.scene_version:
                raise SceneVersionConflict(scenario.scene_version)
            now = _now_utc()
            new_version = scenario.scene_version + 1
            sequence = scenario.edit_sequence
            for edit in applied_edits:
                sequence += 1
                conn.execute(
                    "INSERT INTO edit_events (scenario_id, sequence, event_id, "
                    "base_scene_version, operation, tree_id, old_tree_json, "
                    "new_tree_json, submitted_at, idempotency_key) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        scenario_id,
                        sequence,
                        new_event_id(),
                        int(base_scene_version),
                        edit["operation"],
                        edit["tree_id"],
                        json.dumps(edit["old_tree"]) if edit.get("old_tree") is not None else None,
                        json.dumps(edit["new_tree"]) if edit.get("new_tree") is not None else None,
                        now,
                        idempotency_key,
                    ),
                )
                if edit["operation"] == "delete":
                    conn.execute(
                        "DELETE FROM scenario_trees WHERE scenario_id = ? AND tree_id = ?",
                        (scenario_id, edit["tree_id"]),
                    )
                else:
                    conn.execute(
                        "INSERT INTO scenario_trees (scenario_id, tree_id, tree_json) "
                        "VALUES (?, ?, ?) "
                        "ON CONFLICT (scenario_id, tree_id) DO UPDATE SET tree_json = excluded.tree_json",
                        (scenario_id, edit["tree_id"], json.dumps(edit["new_tree"])),
                    )
            job_id = self._insert_job_locked(
                conn,
                scenario_id,
                target_scene_version=new_version,
                request=dict(requested or {}),
                edit_watermark=sequence,
                # The batch starts at the last *published* watermark, so a
                # job that supersedes an unpublished queued job inherits (and
                # recomputes) that job's unconsumed edits too.
                base_edit_watermark=scenario.acked_sequence,
            )
            self._supersede_queued_locked(conn, scenario_id, except_job_id=job_id)
            conn.execute(
                "UPDATE scenarios SET scene_version = ?, status = 'refining', "
                "edit_sequence = ?, updated_at = ? WHERE scenario_id = ?",
                (new_version, sequence, now, scenario_id),
            )
            record = self._require_scenario_locked(conn, scenario_id)
            response = (
                response_builder(record, job_id)
                if response_builder is not None
                else StoredResponse(202, {"scenario_id": scenario_id, "job_id": job_id})
            )
            if idempotency_key and request_fingerprint:
                self._store_idempotency(
                    conn, scenario_id, idempotency_key, request_fingerprint, response
                )
        return record, job_id

    # -- universal (family) edits ----------------------------------------------

    def commit_universal_edits(
        self,
        scenario_id: str,
        *,
        base_scene_version: int,
        family_edits: Sequence[dict[str, Any]],
        requested: dict[str, Any] | None,
        idempotency_key: str | None = None,
        request_fingerprint: str | None = None,
        response_builder: Callable[[ScenarioRecord, str], StoredResponse] | None = None,
    ) -> tuple[ScenarioRecord, str]:
        """Persist and version one universal (family) edit batch atomically.

        Identical discipline to :meth:`commit_edits` — idempotency replay
        check, optimistic ``base_scene_version`` agreement, one durable
        event per item, job insertion with the published-watermark base,
        supersession of queued jobs, version bump — but the events carry
        the verbatim universal payloads in ``family_json`` (validated by
        the route through the adapter registry BEFORE this call) instead
        of tree rows, and ``scenario_trees`` is untouched (tree edits keep
        the ``/edits`` contract; the two ledgers interleave through the
        shared sequence counter).
        """
        with self._write() as conn:
            if idempotency_key and request_fingerprint:
                self._check_idempotency(conn, scenario_id, idempotency_key, request_fingerprint)
            scenario = self._require_scenario_locked(conn, scenario_id)
            if int(base_scene_version) != scenario.scene_version:
                raise SceneVersionConflict(scenario.scene_version)
            now = _now_utc()
            new_version = scenario.scene_version + 1
            sequence = scenario.edit_sequence
            for edit in family_edits:
                sequence += 1
                conn.execute(
                    "INSERT INTO edit_events (scenario_id, sequence, event_id, "
                    "base_scene_version, operation, tree_id, submitted_at, "
                    "idempotency_key, family_json) "
                    "VALUES (?, ?, ?, ?, ?, NULL, ?, ?, ?)",
                    (
                        scenario_id,
                        sequence,
                        new_event_id(),
                        int(base_scene_version),
                        edit["operation"],
                        now,
                        idempotency_key,
                        json.dumps(edit["payload"], sort_keys=True),
                    ),
                )
            job_id = self._insert_job_locked(
                conn,
                scenario_id,
                target_scene_version=new_version,
                request=dict(requested or {}),
                edit_watermark=sequence,
                base_edit_watermark=scenario.acked_sequence,
            )
            self._supersede_queued_locked(conn, scenario_id, except_job_id=job_id)
            # u-e1 F1: the durable family-state signal rides the same
            # atomic write as the events — correct the instant the state
            # is born, independent of job scheduling or retention sweeps.
            conn.execute(
                "UPDATE scenarios SET scene_version = ?, status = 'refining', "
                "edit_sequence = ?, carries_family_edits = 1, updated_at = ? "
                "WHERE scenario_id = ?",
                (new_version, sequence, now, scenario_id),
            )
            record = self._require_scenario_locked(conn, scenario_id)
            response = (
                response_builder(record, job_id)
                if response_builder is not None
                else StoredResponse(202, {"scenario_id": scenario_id, "job_id": job_id})
            )
            if idempotency_key and request_fingerprint:
                self._store_idempotency(
                    conn, scenario_id, idempotency_key, request_fingerprint, response
                )
        return record, job_id

    def scenario_carries_family_edits(self, scenario_id: str) -> bool:
        """Durable family-state signal for solver routing (u-e1 F1).

        True when the scenario row itself carries the flag (set atomically
        with the family-event insert, cleared by reset — survives
        ``sweep_retention``) OR any family event still survives in the
        ledger (belt-and-braces for rows written before the v5 migration)
        OR the scenario's on-disk executor state carries family state the
        row cannot see (u-e3c attack C: a pre-v5 row whose family events
        were pruned BEFORE the upgrade — the v5 backfill cannot see state
        that no longer exists, but the executor-state directory under
        ``results_root/<scenario>/`` proves the scenario ran family jobs).
        The routing predicate must never UNDER-route: a false negative
        sends a tree edit to the legacy baseline-bound solver, which
        publishes without the family state — a bitwise-wrong scene.

        The dir signal is RESET-AWARE: it arms only when the executor
        state's coverage (the sidecar's adopted watermark, or the staged
        pending's) POSTDATES ``last_reset_sequence`` — a pre-reset snapshot
        was voided by the reset (baseline re-published, family overlays
        discarded by the fresh rebuild) and must never resurrect it.
        """
        row = self._read(
            "SELECT carries_family_edits, last_reset_sequence FROM scenarios "
            "WHERE scenario_id = ?",
            (scenario_id,),
        )
        if row is not None and int(row["carries_family_edits"]):
            return True
        surviving = self._read(
            "SELECT 1 FROM edit_events "
            "WHERE scenario_id = ? AND family_json IS NOT NULL LIMIT 1",
            (scenario_id,),
        )
        if surviving is not None:
            return True
        return self._executor_state_postdates_reset(scenario_id, row)

    def _executor_state_postdates_reset(self, scenario_id: str, row: Any) -> bool:
        """u-e3c attack C: the executor-state rescue, gated on the reset.

        Reads the coverage watermarks the bridge itself trusts — the
        adopted sidecar and the staged pending (lazy import: the bridge
        owns the layout and imports :mod:`.jobs`, which imports this
        module). A missing/corrupt document contributes 0, so state the
        bridge could not trust either cannot arm the rescue.
        """
        from solweig_gpu.server.executor_bridge import (
            COVERAGE_FILENAME,
            PENDING_DIRECTORY_NAME,
            PENDING_META_FILENAME,
            STATE_DIRECTORY_NAME,
        )

        state_dir = self.results_root / scenario_id / STATE_DIRECTORY_NAME
        if not state_dir.is_dir():
            return False

        covered = 0
        for path in (
            state_dir / COVERAGE_FILENAME,
            state_dir / PENDING_DIRECTORY_NAME / PENDING_META_FILENAME,
        ):
            try:
                covered = max(covered, int(json.loads(path.read_text())["covered_sequence"]))
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
        last_reset = int(row["last_reset_sequence"]) if row is not None else 0
        return covered > last_reset

    # -- realtime operations (R1 operation plane, migration v7) ---------------

    @staticmethod
    def _operation_row(row: sqlite3.Row, *, duplicate: bool = False) -> OperationRecord:
        return OperationRecord(
            workspace_id=row["workspace_id"],
            operation_id=row["operation_id"],
            server_sequence=int(row["server_sequence"]),
            epoch_id=int(row["epoch_id"]),
            actor_id=row["actor_id"],
            client_sequence=(
                int(row["client_sequence"]) if row["client_sequence"] is not None else None
            ),
            base_revision=(
                int(row["base_revision"]) if row["base_revision"] is not None else None
            ),
            source_family=row["source_family"],
            entity_id=row["entity_id"],
            verb=row["verb"],
            payload=json.loads(row["payload_json"]),
            received_at=row["received_at"],
            accepted_at=row["accepted_at"],
            duplicate=duplicate,
        )

    @staticmethod
    def _epoch_row(row: sqlite3.Row) -> EpochRecord:
        return EpochRecord(
            workspace_id=row["workspace_id"],
            epoch_id=int(row["epoch_id"]),
            status=str(row["status"]),
            first_sequence=(
                int(row["first_sequence"])
                if row["first_sequence"] is not None
                else None
            ),
            last_sequence=(
                int(row["last_sequence"]) if row["last_sequence"] is not None else None
            ),
            workspace_revision=(
                int(row["workspace_revision"])
                if row["workspace_revision"] is not None
                else None
            ),
            opened_at=row["opened_at"],
            closed_at=row["closed_at"],
        )

    @staticmethod
    def _open_or_create_epoch_locked(
        conn: sqlite3.Connection, workspace_id: str, now: str
    ) -> int:
        """Return the workspace's open epoch id, creating the next epoch
        (lazily epoch 0) when none is open.

        Any non-open status — terminal (``fast_published`` /
        ``exact_targeted``) or intermediate (``closed``/``reducing``/...) —
        means the next accepted operation starts a fresh epoch. Closure
        itself is R2's concern; this store only needs "which epoch is
        accepting operations right now".
        """
        row = conn.execute(
            "SELECT epoch_id, status FROM realtime_epochs "
            "WHERE workspace_id = ? ORDER BY epoch_id DESC LIMIT 1",
            (workspace_id,),
        ).fetchone()
        if row is not None and str(row["status"]) == "open":
            return int(row["epoch_id"])
        next_id = (int(row["epoch_id"]) + 1) if row is not None else 0
        conn.execute(
            "INSERT INTO realtime_epochs (workspace_id, epoch_id, status, opened_at) "
            "VALUES (?, ?, 'open', ?)",
            (workspace_id, next_id, now),
        )
        return next_id

    def append_operations(
        self,
        workspace_id: str,
        items: Sequence[Mapping[str, Any]],
    ) -> list[OperationRecord]:
        """Durably append a batch of realtime operations, ONE transaction.

        Per item, inside the same ``BEGIN IMMEDIATE`` (R0 queue
        cartography: the single-connection RLock + one-txn-per-batch
        discipline is what makes per-workspace sequences contiguous under
        interleaved multi-writer batches):

        * a NEW ``operation_id`` gets the next contiguous
          ``server_sequence`` for the workspace and is assigned to the
          workspace's currently-open epoch (lazily created; a batch of
          only-duplicates never opens an empty epoch row);
        * a retried ``operation_id`` with identical operation content
          (source family, entity, verb, payload — advisory metadata
          excluded by design) returns the EXISTING record with
          ``duplicate=True`` — never a second row;
        * a reused ``operation_id`` with DIFFERENT content (or under a
          different workspace: ids are globally unique) raises
          :class:`OperationFingerprintConflict` and rolls the whole batch
          back — all-or-nothing, like every other mutating store method;
        * a NEW operation whose ``source_family`` has neither a reducer
          fold mapping nor an adapter claim raises
          :class:`UnsupportedFamily` and rolls the whole batch back
          BEFORE any durable write (r2a-review adjudication: reject at
          accept, never accept-then-fold-skip). A RETRY of an
          already-durable operation replays first — the refusal never
          applies retroactively to historical rows, whose exact-lane
          fold-skip-with-disclosure treatment is unchanged.

        ``accepted_at`` is stamped inside this transaction: the instant the
        row becomes durable. ``payload`` is stored verbatim (canonical
        sort-key JSON, the ``family_json`` precedent); family-specific
        validation stays with the executor bridge — this ledger never
        restates it.

        ``accepted_at`` is stamped inside this transaction: the instant the
        row becomes durable. ``payload`` is stored verbatim (canonical
        sort-key JSON, the ``family_json`` precedent); family-specific
        validation stays with the executor bridge — this ledger never
        restates it.
        """
        records: list[OperationRecord] = []
        with self._write() as conn:
            # The workspace must be a live scenario row (defense in depth
            # behind the route's own check; also makes restart tests
            # meaningful — the ledger is workspace-scoped).
            self._require_scenario_locked(conn, workspace_id)
            now = _now_utc()
            sequence = int(
                conn.execute(
                    "SELECT COALESCE(MAX(server_sequence), 0) AS s "
                    "FROM realtime_operations WHERE workspace_id = ?",
                    (workspace_id,),
                ).fetchone()["s"]
            )
            epoch_id: int | None = None
            for item in items:
                operation_id = str(item["operation_id"])
                existing = conn.execute(
                    "SELECT * FROM realtime_operations WHERE operation_id = ?",
                    (operation_id,),
                ).fetchone()
                payload_json = json.dumps(dict(item["payload"]), sort_keys=True)
                if existing is not None:
                    same_content = (
                        str(existing["workspace_id"]) == workspace_id
                        and str(existing["source_family"]) == str(item["source_family"])
                        and existing["entity_id"] == item.get("entity_id")
                        and str(existing["verb"]) == str(item["verb"])
                        and str(existing["payload_json"]) == payload_json
                    )
                    if not same_content:
                        raise OperationFingerprintConflict(operation_id)
                    records.append(self._operation_row(existing, duplicate=True))
                    continue
                # Reject-before-accept (r2a endgame, phased): only NEWLY
                # accepted operations are gated — after the retry replay
                # above, so a historical row's retry stays a replay.
                if str(item["source_family"]) not in _supported_source_families():
                    raise UnsupportedFamily(str(item["source_family"]))
                sequence += 1
                if epoch_id is None:
                    epoch_id = self._open_or_create_epoch_locked(conn, workspace_id, now)
                conn.execute(
                    "INSERT INTO realtime_operations (workspace_id, server_sequence, "
                    "epoch_id, operation_id, actor_id, client_sequence, base_revision, "
                    "source_family, entity_id, verb, payload_json, received_at, "
                    "accepted_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        workspace_id,
                        sequence,
                        epoch_id,
                        operation_id,
                        str(item["actor_id"]),
                        item.get("client_sequence"),
                        item.get("base_revision"),
                        str(item["source_family"]),
                        item.get("entity_id"),
                        str(item["verb"]),
                        payload_json,
                        str(item["received_at"]),
                        now,
                    ),
                )
                conn.execute(
                    "UPDATE realtime_epochs SET "
                    "first_sequence = COALESCE(first_sequence, ?), last_sequence = ? "
                    "WHERE workspace_id = ? AND epoch_id = ?",
                    (sequence, sequence, workspace_id, epoch_id),
                )
                row = conn.execute(
                    "SELECT * FROM realtime_operations WHERE operation_id = ?",
                    (operation_id,),
                ).fetchone()
                records.append(self._operation_row(row))
        return records

    def get_operation(
        self, workspace_id: str, operation_id: str
    ) -> OperationRecord | None:
        """Idempotent-retry lookup for one operation of one workspace."""
        row = self._read(
            "SELECT * FROM realtime_operations "
            "WHERE workspace_id = ? AND operation_id = ?",
            (workspace_id, operation_id),
        )
        return self._operation_row(row) if row is not None else None

    def open_epochs(self, workspace_id: str) -> list[EpochRecord]:
        """Epochs of the workspace currently accepting operations."""
        rows = self._read_all(
            "SELECT * FROM realtime_epochs WHERE workspace_id = ? AND status = 'open' "
            "ORDER BY epoch_id",
            (workspace_id,),
        )
        return [self._epoch_row(row) for row in rows]

    def epoch_records(self, workspace_id: str) -> list[EpochRecord]:
        """Every epoch row of the workspace, oldest first.

        Restart recovery reads this: nothing auto-closes on Store open, so
        epochs caught mid-flight (``closed``/``reducing``/...) stay exactly
        as recorded for R2's recovery to close/replay them.
        """
        rows = self._read_all(
            "SELECT * FROM realtime_epochs WHERE workspace_id = ? ORDER BY epoch_id",
            (workspace_id,),
        )
        return [self._epoch_row(row) for row in rows]

    def operations_for_epoch(
        self, workspace_id: str, epoch_id: int
    ) -> list[OperationRecord]:
        """Every operation assigned to one epoch, in server-sequence order."""
        rows = self._read_all(
            "SELECT * FROM realtime_operations WHERE workspace_id = ? AND epoch_id = ? "
            "ORDER BY server_sequence",
            (workspace_id, int(epoch_id)),
        )
        return [self._operation_row(row) for row in rows]

    def operations_since(
        self, workspace_id: str, server_sequence: int
    ) -> list[OperationRecord]:
        """Catch-up read: operations with ``server_sequence`` greater than
        the client's high-water mark, in order."""
        rows = self._read_all(
            "SELECT * FROM realtime_operations WHERE workspace_id = ? "
            "AND server_sequence > ? ORDER BY server_sequence",
            (workspace_id, int(server_sequence)),
        )
        return [self._operation_row(row) for row in rows]

    def operations_in_span(
        self,
        workspace_id: str,
        after_sequence: int,
        through_sequence: int,
    ) -> list[OperationRecord]:
        """Span-bounded read: ``after_sequence`` < seq <= ``through_sequence``.

        Encodes the (after, through] window the exact-lane fold consumes,
        returning the same ordered rows the zero-loss fence inspects. The
        executor keeps its single whole-history read for today — the
        mixed-plane anchors and the consumed-chain baseline are
        contractually whole-history scans, and slicing them needs a prune
        floor that does not exist yet (blanket truncation was rejected in
        the stability review as silently-wrong-science territory). Callers
        that genuinely want one span (future prune-floor work, diagnostics)
        use this instead of re-deriving the boundary math.
        """
        rows = self._read_all(
            "SELECT * FROM realtime_operations WHERE workspace_id = ? "
            "AND server_sequence > ? AND server_sequence <= ? "
            "ORDER BY server_sequence",
            (workspace_id, int(after_sequence), int(through_sequence)),
        )
        return [self._operation_row(row) for row in rows]

    def mark_epoch_status(
        self,
        workspace_id: str,
        epoch_id: int,
        status: str,
        *,
        workspace_revision: int | None = None,
    ) -> bool:
        """Status-transition primitive for the R2 epoch scheduler.

        This wave deliberately does NOT implement ``close_epoch`` — closure
        semantics (which epochs close, when, what revision they are
        assigned) belong to the scheduler. The primitive only records what
        R2 decides: the new status (validated against
        :data:`EPOCH_STATUSES`), ``closed_at`` stamped on the first
        transition out of ``open`` (never overwritten afterwards), and the
        epoch's ``workspace_revision`` when supplied. Returns whether the
        epoch row existed.
        """
        if status not in EPOCH_STATUSES:
            raise StoreError(f"invalid epoch status {status!r}")
        with self._write() as conn:
            cursor = conn.execute(
                "UPDATE realtime_epochs SET status = ?, "
                "closed_at = COALESCE(closed_at, CASE WHEN status = 'open' "
                "THEN ? END), "
                "workspace_revision = COALESCE(?, workspace_revision) "
                "WHERE workspace_id = ? AND epoch_id = ?",
                (
                    status,
                    _now_utc() if status != "open" else None,
                    workspace_revision,
                    workspace_id,
                    int(epoch_id),
                ),
            )
            return cursor.rowcount > 0

    # -- epoch close pipeline (R1 scheduler wave, migration v8) ---------------

    def get_epoch(self, workspace_id: str, epoch_id: int) -> EpochRecord | None:
        """One epoch row or ``None`` (scheduler close/recovery reads)."""
        row = self._read(
            "SELECT * FROM realtime_epochs WHERE workspace_id = ? AND epoch_id = ?",
            (workspace_id, int(epoch_id)),
        )
        return self._epoch_row(row) if row is not None else None

    def open_epoch_workspaces(self) -> list[str]:
        """Workspaces with at least one ``open`` epoch, sorted."""
        rows = self._read_all(
            "SELECT DISTINCT workspace_id FROM realtime_epochs "
            "WHERE status = 'open' ORDER BY workspace_id"
        )
        return [row["workspace_id"] for row in rows]

    def recoverable_epochs(self) -> list[EpochRecord]:
        """Non-terminal epochs across ALL workspaces, oldest first.

        Every status that is not ``fast_published``/``exact_targeted`` is by
        definition still owed work (``open`` epochs wait on their window;
        ``closed``/``reducing`` are mid close-pipeline). The scheduler's
        ``recover()`` filters this set: open epochs are the live cadence's
        business, closed/reducing ones must be re-driven at boot.
        """
        rows = self._read_all(
            "SELECT * FROM realtime_epochs WHERE status NOT IN "
            "('fast_published', 'exact_targeted') "
            "ORDER BY workspace_id, epoch_id"
        )
        return [self._epoch_row(row) for row in rows]

    def commit_epoch_reduction(
        self,
        workspace_id: str,
        epoch_id: int,
        *,
        families: Mapping[str, Any],
        baseline_revision: int | None = None,
    ) -> "CommittedEpoch | None":
        """Durably land one epoch's canonical reduction in ONE transaction.

        Inside a single ``_write()`` (BEGIN IMMEDIATE under the store lock,
        so it serializes with ``append_operations`` — zero loss window):

        1. persist the folded ``families`` into ``realtime_canonical_state``
           keyed by the assigned revision (``INSERT OR IGNORE``: the original
           bytes always win a re-drive);
        2. assign the epoch's ``workspace_revision`` = ``scene_version`` + 1,
           computed INSIDE the transaction (race-free);
        3. bump ``scenarios.scene_version`` by exactly one — this bump IS the
           canonical revision assignment (accepted operations never bump it
           themselves);
        4. advance the epoch to ``reducing`` unless it already carries a
           terminal status (never downgrades R2's lane statuses).

        ``baseline_revision`` guards against the stale-baseline race
        (r1-epochs-review M2): pass the ``workspace_revision`` the epoch was
        folded FROM (``0`` when the fold started from the initial state and
        no canonical row existed). The commit then refuses — typed
        :class:`EpochBaselineStale`, transaction rolled back — if the latest
        persisted canonical revision is anything else, i.e. another driver
        landed a reduction after this fold read its baseline. ``None``
        disables the guard (the fold's provenance is unknown; legacy
        callers/tests).

        Idempotent: if the epoch row already has a ``workspace_revision``,
        nothing bumps and nothing overwrites — ``first_time`` is False so the
        caller re-broadcasts without re-folding. Returns ``None`` when the
        epoch (or its scenario) does not exist.
        """
        now = _now_utc()
        with self._write() as conn:
            epoch_row = conn.execute(
                "SELECT status, workspace_revision FROM realtime_epochs "
                "WHERE workspace_id = ? AND epoch_id = ?",
                (workspace_id, int(epoch_id)),
            ).fetchone()
            if epoch_row is None:
                return None
            existing_revision = epoch_row["workspace_revision"]
            if existing_revision is not None:
                # Idempotent re-drive: keep the original state bytes, never
                # bump, never touch status.
                state_json = json.dumps(
                    {"families": families}, separators=(",", ":")
                )
                conn.execute(
                    "INSERT OR IGNORE INTO realtime_canonical_state "
                    "(workspace_id, workspace_revision, state_json, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (workspace_id, int(existing_revision), state_json, now),
                )
                return CommittedEpoch(
                    workspace_id=workspace_id,
                    epoch_id=int(epoch_id),
                    workspace_revision=int(existing_revision),
                    first_time=False,
                )
            scenario_row = conn.execute(
                "SELECT scene_version FROM scenarios WHERE scenario_id = ?",
                (workspace_id,),
            ).fetchone()
            if scenario_row is None:
                return None
            if baseline_revision is not None:
                max_row = conn.execute(
                    "SELECT MAX(workspace_revision) FROM realtime_canonical_state "
                    "WHERE workspace_id = ?",
                    (workspace_id,),
                ).fetchone()
                latest = int(max_row[0]) if max_row[0] is not None else 0
                if latest != int(baseline_revision):
                    raise EpochBaselineStale(
                        workspace_id, int(baseline_revision), latest
                    )
            revision = int(scenario_row["scene_version"]) + 1
            state_json = json.dumps({"families": families}, separators=(",", ":"))
            conn.execute(
                "INSERT INTO realtime_canonical_state "
                "(workspace_id, workspace_revision, state_json, created_at) "
                "VALUES (?, ?, ?, ?)",
                (workspace_id, revision, state_json, now),
            )
            conn.execute(
                "UPDATE scenarios SET scene_version = ?, updated_at = ? "
                "WHERE scenario_id = ?",
                (revision, now, workspace_id),
            )
            status = str(epoch_row["status"])
            if status in TERMINAL_EPOCH_STATUSES:
                # Defensive: a terminal epoch without a revision still gets
                # its assignment recorded, but its lane status is R2's truth.
                conn.execute(
                    "UPDATE realtime_epochs SET workspace_revision = ? "
                    "WHERE workspace_id = ? AND epoch_id = ?",
                    (revision, workspace_id, int(epoch_id)),
                )
            else:
                conn.execute(
                    "UPDATE realtime_epochs SET status = 'reducing', "
                    "closed_at = COALESCE(closed_at, ?), workspace_revision = ? "
                    "WHERE workspace_id = ? AND epoch_id = ?",
                    (now, revision, workspace_id, int(epoch_id)),
                )
            return CommittedEpoch(
                workspace_id=workspace_id,
                epoch_id=int(epoch_id),
                workspace_revision=revision,
                first_time=True,
            )

    def _canonical_row(self, row: sqlite3.Row | None) -> CanonicalState | None:
        if row is None:
            return None
        state = json.loads(row["state_json"])
        return CanonicalState(
            workspace_id=row["workspace_id"],
            workspace_revision=int(row["workspace_revision"]),
            families=state.get("families", {}),
        )

    def latest_canonical_state(self, workspace_id: str) -> CanonicalState | None:
        """The workspace's newest canonical state row (next epoch's baseline)."""
        row = self._read(
            "SELECT * FROM realtime_canonical_state WHERE workspace_id = ? "
            "ORDER BY workspace_revision DESC LIMIT 1",
            (workspace_id,),
        )
        return self._canonical_row(row)

    def canonical_state_at(
        self, workspace_id: str, workspace_revision: int
    ) -> CanonicalState | None:
        """The canonical state row for one exact revision (replay/audit)."""
        row = self._read(
            "SELECT * FROM realtime_canonical_state WHERE workspace_id = ? "
            "AND workspace_revision = ?",
            (workspace_id, int(workspace_revision)),
        )
        return self._canonical_row(row)

    # -- fast revision advance (R2b fast lane, r2b-review F6) ------------------

    def scenarios_with_fast_lag(self) -> list[ScenarioRecord]:
        """Workspaces whose ``fast_revision`` lags their workspace revision
        (``scene_version``) — the fast lane's durable-state discovery scan
        (the typed replacement for the scheduler's raw scenarios SQL)."""
        rows = self._read_all(
            "SELECT * FROM scenarios WHERE fast_revision < scene_version"
        )
        return [self._scenario_row(row) for row in rows]

    def advance_fast_revision(
        self,
        workspace_id: str,
        revision: int,
        exact_result_version: int,
        *,
        updated_at: str | None = None,
    ) -> "FastRevisionAdvance | None":
        """Advance a workspace's ``fast_revision`` in ONE write transaction.

        The r2b fast lane's publish commit, as a typed accessor
        (r2b-review F6 — the scheduler no longer reaches into raw
        scenarios SQL). Semantics are the r2b implementation's, unchanged:

        * stale fence — ``None`` (nothing written) when the workspace does
          not exist or its ``scene_version`` has moved BELOW ``revision``;
          the caller owns the ``stale_publish_rejected_total`` observation;
        * monotonic clamp — the new ``fast_revision`` is
          ``max(old_fast, revision, exact_now)``, where ``exact_now`` is
          the ``exact_result_version`` read INSIDE this transaction, so a
          fast publish can never regress behind a concurrently published
          exact result;
        * ``exact_base_revision`` is compensated to at least
          ``exact_now`` in the same transaction.

        ``exact_result_version`` is the caller's observed exact revision —
        a defensive floor that the in-transaction ``exact_now`` subsumes
        by monotonicity (exact revisions never decrease). ``updated_at``
        defaults to now; the scheduler passes its publication stamp so the
        row's clock anchor is the publish decision, not the txn commit.
        """
        with self._write() as conn:
            row = conn.execute(
                "SELECT scene_version, fast_revision, exact_result_version "
                "FROM scenarios WHERE scenario_id = ?",
                (workspace_id,),
            ).fetchone()
            if row is None or int(row["scene_version"]) < int(revision):
                return None
            exact_now = int(row["exact_result_version"])
            new_fast = max(
                int(row["fast_revision"]),
                int(revision),
                exact_now,
                int(exact_result_version),
            )
            conn.execute(
                "UPDATE scenarios SET fast_revision = ?, "
                "exact_base_revision = max(exact_base_revision, ?), "
                "updated_at = ? WHERE scenario_id = ?",
                (
                    new_fast,
                    exact_now,
                    updated_at or _now_utc(),
                    workspace_id,
                ),
            )
            return FastRevisionAdvance(
                workspace_id=workspace_id,
                revision=int(revision),
                fast_revision=new_fast,
                exact_revision=exact_now,
            )

    # -- reset ----------------------------------------------------------------

    def reset_scenario(
        self,
        scenario_id: str,
        *,
        base_scene_version: int | None = None,
        baseline_publisher: Callable[[str, int], tuple[dict[str, Any], bytes]],
        idempotency_key: str | None = None,
        request_fingerprint: str | None = None,
        response_builder: Callable[[ScenarioRecord], StoredResponse] | None = None,
    ) -> ScenarioRecord:
        """Reset to the exact baseline scene without running the worker.

        Reset is a versioned mutation: editable trees are removed, the scene
        version advances, and the baseline result is re-published at the new
        version so ``exact_result_version == scene_version`` afterwards. Queued
        jobs targeting older versions are superseded. The workspace's
        revision-0 canonical row (the exact lane's bootstrap pin —
        ``jobs.ensure_exact_lane_bootstrap``, r2a-fix) is deleted in the
        same transaction (r2a-fix-review F2): it folds the legacy ledger
        history this reset just voided, so letting it survive would keep
        reset-deleted legacy objects in the chain re-fold baseline. The
        next bootstrap re-derives the pin from post-reset history only
        (its fold already restarts at the last reset event); until then
        the workspace simply has no rev-0 baseline, the pre-seam state of
        any never-bootstrapped workspace.
        """
        with self._write() as conn:
            if idempotency_key and request_fingerprint:
                self._check_idempotency(conn, scenario_id, idempotency_key, request_fingerprint)
            scenario = self._require_scenario_locked(conn, scenario_id)
            if base_scene_version is not None and int(base_scene_version) != scenario.scene_version:
                raise SceneVersionConflict(scenario.scene_version)
            now = _now_utc()
            new_version = scenario.scene_version + 1
            sequence = scenario.edit_sequence + 1
            conn.execute(
                "INSERT INTO edit_events (scenario_id, sequence, event_id, "
                "base_scene_version, operation, tree_id, submitted_at, idempotency_key) "
                "VALUES (?, ?, ?, ?, 'reset', NULL, ?, ?)",
                (scenario_id, sequence, new_event_id(), scenario.scene_version, now, idempotency_key),
            )
            conn.execute("DELETE FROM scenario_trees WHERE scenario_id = ?", (scenario_id,))
            # r2a-fix-review F2: the rev-0 canonical row is the exact
            # lane's bootstrap pin, folded from the legacy ledger history
            # this reset just voided. Delete it in the same transaction —
            # the chain re-fold baseline may not carry reset-deleted
            # legacy objects. The next ensure_exact_lane_bootstrap
            # re-derives it from post-reset history only.
            conn.execute(
                "DELETE FROM realtime_canonical_state "
                "WHERE workspace_id = ? AND workspace_revision = 0",
                (scenario_id,),
            )
            manifest, payload = baseline_publisher(scenario_id, new_version)
            # Advance the scenario row before publishing so the stale guard in
            # _publish_result_locked sees the new version as current. The
            # reset watermark rides the same write (u-e3c attack C): it is
            # the retention-immune record that executor state at or below
            # this sequence predates the reset and must never resurrect.
            conn.execute(
                "UPDATE scenarios SET scene_version = ?, exact_result_version = ?, "
                "status = 'exact', edit_sequence = ?, acked_sequence = ?, "
                "carries_family_edits = 0, last_reset_sequence = ?, updated_at = ? "
                "WHERE scenario_id = ?",
                (new_version, new_version, sequence, sequence, sequence, now, scenario_id),
            )
            self._publish_result_locked(
                conn,
                scenario_id,
                new_version,
                manifest=manifest,
                payload=payload,
                exact=True,
                job_id=None,
                acked_sequence=sequence,
            )
            self._supersede_queued_locked(conn, scenario_id, except_job_id=None)
            record = self._require_scenario_locked(conn, scenario_id)
            response = (
                response_builder(record)
                if response_builder is not None
                else StoredResponse(200, {"scenario_id": scenario_id})
            )
            if idempotency_key and request_fingerprint:
                self._store_idempotency(
                    conn, scenario_id, idempotency_key, request_fingerprint, response
                )
        return record

    # -- jobs -----------------------------------------------------------------

    @staticmethod
    def _insert_job_locked(
        conn: sqlite3.Connection,
        scenario_id: str,
        *,
        target_scene_version: int,
        request: dict[str, Any],
        edit_watermark: int,
        base_edit_watermark: int = 0,
    ) -> str:
        job_id = new_job_id()
        conn.execute(
            "INSERT INTO jobs (job_id, scenario_id, target_scene_version, status, "
            "request_json, edit_watermark, base_edit_watermark, queued_at) "
            "VALUES (?, ?, ?, 'queued', ?, ?, ?, ?)",
            (
                job_id,
                scenario_id,
                int(target_scene_version),
                json.dumps(request, sort_keys=True),
                int(edit_watermark),
                int(base_edit_watermark),
                _now_utc(),
            ),
        )
        return job_id

    @staticmethod
    def _job_row(row: sqlite3.Row) -> JobRecord:
        return JobRecord(
            job_id=row["job_id"],
            scenario_id=row["scenario_id"],
            target_scene_version=int(row["target_scene_version"]),
            status=row["status"],
            stage=row["stage"],
            progress=json.loads(row["progress_json"]) if row["progress_json"] else None,
            mode=row["mode"],
            window=json.loads(row["window_json"]) if row["window_json"] else None,
            request=json.loads(row["request_json"]) if row["request_json"] else None,
            edit_watermark=int(row["edit_watermark"]),
            base_edit_watermark=int(row["base_edit_watermark"]),
            queued_at=row["queued_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            worker_revision=row["worker_revision"],
            metrics=json.loads(row["metrics_json"]) if row["metrics_json"] else None,
            error=json.loads(row["error_json"]) if row["error_json"] else None,
            result_scene_version=(
                int(row["result_scene_version"]) if row["result_scene_version"] is not None else None
            ),
            plan=json.loads(row["plan_json"]) if row["plan_json"] else None,
        )

    def get_job(self, job_id: str) -> JobRecord | None:
        row = self._read("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
        return self._job_row(row) if row is not None else None

    def require_job(self, job_id: str) -> JobRecord:
        job = self.get_job(job_id)
        if job is None:
            raise JobNotFound(job_id)
        return job

    def latest_job_for_scenario(self, scenario_id: str) -> JobRecord | None:
        row = self._read(
            "SELECT * FROM jobs WHERE scenario_id = ? ORDER BY rowid DESC LIMIT 1",
            (scenario_id,),
        )
        return self._job_row(row) if row is not None else None

    def queue_position(self, job_id: str) -> int | None:
        job = self.get_job(job_id)
        if job is None or job.status != QUEUED:
            return None
        rows = self._read_all(
            "SELECT job_id FROM jobs WHERE status = 'queued' ORDER BY rowid"
        )
        ids = [row["job_id"] for row in rows]
        return ids.index(job_id) + 1 if job_id in ids else None

    def nonterminal_jobs(self) -> list[JobRecord]:
        rows = self._read_all(
            "SELECT * FROM jobs WHERE status IN ('queued', 'running') ORDER BY rowid"
        )
        return [self._job_row(row) for row in rows]

    @staticmethod
    def _supersede_queued_locked(
        conn: sqlite3.Connection, scenario_id: str, *, except_job_id: str | None
    ) -> int:
        """Mark queued jobs for a scenario superseded; return how many."""
        now = _now_utc()
        cursor = conn.execute(
            "UPDATE jobs SET status = 'superseded', finished_at = ? "
            "WHERE scenario_id = ? AND status = 'queued' AND job_id != ?",
            (now, scenario_id, except_job_id or ""),
        )
        return cursor.rowcount

    def mark_job_running(self, job_id: str) -> None:
        with self._write() as conn:
            conn.execute(
                "UPDATE jobs SET status = 'running', started_at = ?, stage = 'starting' "
                "WHERE job_id = ? AND status = 'queued'",
                (_now_utc(), job_id),
            )

    def requeue_failed_job(self, job_id: str) -> bool:
        """Reset a terminal ``failed``/``cancelled`` job back to ``queued``.

        Boot self-heal for the shared world's baseline: a failed boot
        baseline would otherwise brick the workspace for every visitor
        forever (no client ever re-requests it). Only terminal rows move;
        the durable job definition (edits, requested result) is untouched.
        """
        with self._write() as conn:
            cursor = conn.execute(
                "UPDATE jobs SET status = 'queued', started_at = NULL, stage = NULL, "
                "progress_json = NULL, mode = NULL, window_json = NULL, "
                "error_json = NULL, finished_at = NULL "
                "WHERE job_id = ? AND status IN ('failed', 'cancelled')",
                (job_id,),
            )
            return cursor.rowcount > 0

    def requeue_running_job(self, job_id: str) -> bool:
        """Reset a job found ``running`` at startup back to ``queued``.

        Used by restart recovery: the previous process died mid-flight, so the
        job resumes from its durable queue entry.
        """
        with self._write() as conn:
            cursor = conn.execute(
                "UPDATE jobs SET status = 'queued', started_at = NULL, stage = NULL, "
                "progress_json = NULL, mode = NULL, window_json = NULL, "
                "error_json = NULL, finished_at = NULL "
                "WHERE job_id = ? AND status = 'running'",
                (job_id,),
            )
            return cursor.rowcount > 0

    def update_job_progress(
        self,
        job_id: str,
        *,
        stage: str | None = None,
        progress: Mapping[str, int] | None = None,
        mode: str | None = None,
        window: Mapping[str, int] | None = None,
    ) -> None:
        with self._write() as conn:
            conn.execute(
                "UPDATE jobs SET "
                "stage = COALESCE(?, stage), "
                "progress_json = COALESCE(?, progress_json), "
                "mode = COALESCE(?, mode), "
                "window_json = COALESCE(?, window_json) "
                "WHERE job_id = ? AND status = 'running'",
                (
                    stage,
                    json.dumps(dict(progress)) if progress is not None else None,
                    mode,
                    json.dumps(dict(window)) if window is not None else None,
                    job_id,
                ),
            )

    def finish_job(
        self,
        job_id: str,
        status: str,
        *,
        mode: str | None = None,
        window: Mapping[str, int] | None = None,
        metrics: Mapping[str, Any] | None = None,
        error: Mapping[str, Any] | None = None,
        result_scene_version: int | None = None,
        worker_revision: str | None = None,
        plan: Mapping[str, Any] | None = None,
    ) -> bool:
        """Move a queued/running job to ``status``; report whether it did.

        Returns ``False`` when the row was already terminal — e.g. a cancel
        racing a completion finds the job ``complete`` and must not overwrite
        it (the observed terminal outcome stands).
        """
        if status not in TERMINAL_JOB_STATUSES:
            raise StoreError(f"invalid terminal job status {status!r}")
        with self._write() as conn:
            cursor = conn.execute(
                "UPDATE jobs SET status = ?, finished_at = ?, mode = COALESCE(?, mode), "
                "window_json = COALESCE(?, window_json), metrics_json = ?, error_json = ?, "
                "result_scene_version = ?, worker_revision = COALESCE(?, worker_revision), "
                "plan_json = COALESCE(?, plan_json) "
                "WHERE job_id = ? AND status IN ('queued', 'running')",
                (
                    status,
                    _now_utc(),
                    mode,
                    json.dumps(dict(window)) if window is not None else None,
                    json.dumps(dict(metrics), sort_keys=True) if metrics is not None else None,
                    json.dumps(dict(error), sort_keys=True) if error is not None else None,
                    result_scene_version,
                    worker_revision,
                    json.dumps(dict(plan), sort_keys=True) if plan is not None else None,
                    job_id,
                ),
            )
            return cursor.rowcount > 0

    # -- reconciliation bookkeeping (routing policy wave, migration v9) ------

    def mark_reconcile_owed(self, workspace_id: str) -> None:
        """Record deferred reconciliation debt (idempotent single write).

        The exact lane sets this when it defers a recoverable revision gap
        because the workspace has no subscribers. The ``WHERE owed = 0``
        guard keeps repeated deferrals read-mostly (no write storm while
        the debt stands). Debt persists across restarts and day
        boundaries; only :meth:`mark_reconcile_completed` discharges it.
        """
        now = _now_utc()
        with self._write() as conn:
            conn.execute(
                "INSERT INTO workspace_reconcile "
                "(workspace_id, last_completed_date, owed, updated_at) "
                "VALUES (?, '1970-01-01', 1, ?) "
                "ON CONFLICT(workspace_id) DO UPDATE SET owed = 1, updated_at = ? "
                "WHERE owed = 0",
                (workspace_id, now, now),
            )

    def mark_reconcile_completed(self, workspace_id: str, date: str) -> None:
        """Discharge the debt and consume the once-daily cap for ``date``.

        ``date`` is the UTC ``YYYY-MM-DD`` of the completed reconcile.
        Only a COMPLETED reconciliation may call this — a superseded
        attempt must leave both the cap and the debt untouched.
        """
        now = _now_utc()
        with self._write() as conn:
            conn.execute(
                "INSERT INTO workspace_reconcile "
                "(workspace_id, last_completed_date, owed, updated_at) "
                "VALUES (?, ?, 0, ?) "
                "ON CONFLICT(workspace_id) DO UPDATE SET "
                "last_completed_date = excluded.last_completed_date, "
                "owed = 0, updated_at = excluded.updated_at",
                (workspace_id, str(date), now),
            )

    def clear_reconcile_owed(self, workspace_id: str) -> None:
        """Discharge the debt WITHOUT consuming the once-daily cap.

        Called when an exact-lane chase settles the revision gap
        incrementally (a LOCAL or no-op completion): the debt is paid, but
        no full-tile pass ran, so the workspace keeps its daily full.
        """
        with self._write() as conn:
            conn.execute(
                "UPDATE workspace_reconcile SET owed = 0, updated_at = ? "
                "WHERE workspace_id = ? AND owed = 1",
                (_now_utc(), workspace_id),
            )

    def reconcile_state(self, workspace_id: str) -> tuple[str | None, bool]:
        """``(last_completed_date, owed)``; ``(None, False)`` when unknown."""
        row = self._read(
            "SELECT last_completed_date, owed FROM workspace_reconcile "
            "WHERE workspace_id = ?",
            (workspace_id,),
        )
        if row is None:
            return None, False
        return str(row["last_completed_date"]), bool(row["owed"])

    def workspaces_owing_reconcile(self) -> list[str]:
        """Workspaces carrying undischarged reconciliation debt, sorted."""
        rows = self._read_all(
            "SELECT workspace_id FROM workspace_reconcile WHERE owed = 1 "
            "ORDER BY workspace_id"
        )
        return [row["workspace_id"] for row in rows]

    def last_full_result_version(self, scenario_id: str) -> int | None:
        """Scene version of the most recent COMPLETED full-mode job.

        Anchors the patch-chain-depth heuristic (deep revision chains since
        the last authoritative full degrade incremental accuracy/coverage);
        ``None`` when the scenario has never completed a full solve.
        """
        row = self._read(
            "SELECT result_scene_version FROM jobs "
            "WHERE scenario_id = ? AND status = 'complete' AND mode = 'full' "
            "AND result_scene_version IS NOT NULL "
            "ORDER BY rowid DESC LIMIT 1",
            (scenario_id,),
        )
        return int(row["result_scene_version"]) if row is not None else None

    # -- honest solve ETA (routing policy wave) ---------------------------------

    #: Disclosed static fallback for full-solve ETAs with too little
    #: history (``eta_basis: "static"``): ~8 minutes for one full-tile pass
    #: on the single-core deployment the policy wave targets.
    STATIC_FULL_SOLVE_SECONDS = 480

    #: Completed-job sample horizon the estimator reads.
    ETA_SAMPLE_HORIZON = 50

    @staticmethod
    def _p80(values: Sequence[float]) -> float:
        ordered = sorted(values)
        index = max(0, min(len(ordered) - 1, int(math.ceil(0.8 * len(ordered))) - 1))
        return ordered[index]

    def estimate_solve_seconds(
        self,
        scenario_id: str,
        mode: str,
        *,
        window_fraction: float | None = None,
    ) -> tuple[int | None, str | None]:
        """Honest ETA from this scenario's own completed jobs.

        ``(seconds, basis)`` where basis is ``"history"`` (p80 of recent
        durations from ``jobs.metrics_json``), ``"static"`` (the disclosed
        fallback used only for full solves with fewer than two historical
        samples), or ``None`` — unknown, surfaced to clients as a null ETA
        rather than a guess. Durations round to the nearest 10 s.

        Local solves rescale each historical sample by the ratio of the
        target window fraction to the sample's own, restricted to samples
        within ±0.2 of the target so the linear rescaling stays honest.
        """
        rows = self._read_all(
            "SELECT metrics_json FROM jobs "
            "WHERE scenario_id = ? AND status = 'complete' "
            "AND metrics_json IS NOT NULL "
            "ORDER BY rowid DESC LIMIT ?",
            (scenario_id, self.ETA_SAMPLE_HORIZON),
        )
        samples: list[tuple[float, float | None]] = []
        for row in rows:
            try:
                metrics = json.loads(row["metrics_json"])
                if str(metrics.get("mode")) != mode:
                    continue
                duration_ms = float(metrics.get("duration_ms", 0.0))
                if duration_ms <= 0.0:
                    continue
                fraction = metrics.get("window_fraction")
                samples.append(
                    (duration_ms / 1000.0, float(fraction) if fraction is not None else None)
                )
            except (ValueError, TypeError):
                continue
        if mode == "full":
            durations = [seconds for seconds, _ in samples if seconds > 0.0]
            if len(durations) >= 2:
                return (
                    max(10, int(round(self._p80(durations) / 10.0) * 10)),
                    "history",
                )
            return (self.STATIC_FULL_SOLVE_SECONDS, "static")
        if mode == "local" and window_fraction is not None and window_fraction > 0.0:
            target = float(window_fraction)
            rescaled = [
                seconds * target / sample_fraction
                for seconds, sample_fraction in samples
                if sample_fraction is not None
                and sample_fraction > 0.0
                and abs(sample_fraction - target) <= 0.2
            ]
            if len(rescaled) >= 2:
                return (
                    max(10, int(round(self._p80(rescaled) / 10.0) * 10)),
                    "history",
                )
        return (None, None)

    # -- results ----------------------------------------------------------------

    def result_dir(self, scenario_id: str, scene_version: int) -> Path:
        return self.results_root / scenario_id / "results" / str(int(scene_version))

    def _publish_result_locked(
        self,
        conn: sqlite3.Connection,
        scenario_id: str,
        scene_version: int,
        *,
        manifest: dict[str, Any],
        payload: bytes,
        exact: bool,
        job_id: str | None,
        acked_sequence: int | None,
    ) -> None:
        scenario = self._require_scenario_locked(conn, scenario_id)
        if int(scene_version) != scenario.scene_version:
            raise StaleResultError(int(scene_version), scenario.scene_version)
        existing = conn.execute(
            "SELECT 1 FROM results WHERE scenario_id = ? AND scene_version = ?",
            (scenario_id, int(scene_version)),
        ).fetchone()
        if existing is not None:
            raise ResultAlreadyPublished(
                f"scenario {scenario_id} already has a result for version {scene_version}"
            )
        payload_path = self._write_payload(scenario_id, scene_version, manifest, payload)
        conn.execute(
            "INSERT INTO results (scenario_id, scene_version, exact, manifest_json, "
            "payload_path, checksum, created_at, job_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                scenario_id,
                int(scene_version),
                1 if exact else 0,
                json.dumps(manifest, sort_keys=True),
                str(payload_path),
                str(manifest.get("checksum", "")),
                _now_utc(),
                job_id,
            ),
        )
        # ``>=`` so publishing the baseline at version 0 also marks the
        # scenario exact (0 > 0 would never hold).
        if exact and int(scene_version) >= scenario.exact_result_version:
            conn.execute(
                "UPDATE scenarios SET exact_result_version = ?, status = 'exact', "
                "acked_sequence = MAX(acked_sequence, ?), updated_at = ? "
                "WHERE scenario_id = ?",
                (int(scene_version), int(acked_sequence or 0), _now_utc(), scenario_id),
            )

    def _write_payload(
        self, scenario_id: str, scene_version: int, manifest: dict[str, Any], payload: bytes
    ) -> Path:
        """Write manifest + payload atomically (tmp + rename, same directory)."""
        directory = self.result_dir(scenario_id, scene_version)
        directory.mkdir(parents=True, exist_ok=True)
        manifest_tmp = directory / (MANIFEST_FILE + ".tmp")
        payload_tmp = directory / (PAYLOAD_FILE + ".tmp")
        manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        payload_tmp.write_bytes(payload)
        manifest_tmp.replace(directory / MANIFEST_FILE)
        payload_tmp.replace(directory / PAYLOAD_FILE)
        return directory / PAYLOAD_FILE

    def publish_result(
        self,
        scenario_id: str,
        scene_version: int,
        *,
        manifest: dict[str, Any],
        payload: bytes,
        exact: bool = True,
        job_id: str | None = None,
        acked_sequence: int | None = None,
    ) -> None:
        """Publish a result; refuses stale versions (scientific-integrity guard).

        When ``job_id`` names the durable job this result belongs to, the job
        must still be ``running`` *inside this transaction*: the runner's
        liveness check and this publish are otherwise separate transactions,
        and a cancel landing in between would publish for a cancelled job.
        A non-running job raises :class:`ResultNotPublishable` and nothing is
        written (the whole transaction rolls back).
        """
        with self._write() as conn:
            if job_id is not None:
                row = conn.execute(
                    "SELECT status FROM jobs WHERE job_id = ?", (job_id,)
                ).fetchone()
                if row is None or str(row["status"]) != "running":
                    raise ResultNotPublishable(
                        job_id, "unknown" if row is None else str(row["status"])
                    )
            self._publish_result_locked(
                conn,
                scenario_id,
                scene_version,
                manifest=manifest,
                payload=payload,
                exact=exact,
                job_id=job_id,
                acked_sequence=acked_sequence,
            )

    def get_result(self, scenario_id: str, scene_version: int) -> ResultRecord | None:
        row = self._read(
            "SELECT * FROM results WHERE scenario_id = ? AND scene_version = ?",
            (scenario_id, int(scene_version)),
        )
        return self._result_row(row) if row is not None else None

    def latest_result(self, scenario_id: str) -> ResultRecord | None:
        row = self._read(
            "SELECT * FROM results WHERE scenario_id = ? ORDER BY scene_version DESC LIMIT 1",
            (scenario_id,),
        )
        return self._result_row(row) if row is not None else None

    def result_versions(self, scenario_id: str) -> list[int]:
        rows = self._read_all(
            "SELECT scene_version FROM results WHERE scenario_id = ? "
            "ORDER BY scene_version",
            (scenario_id,),
        )
        return [int(row["scene_version"]) for row in rows]

    @staticmethod
    def _result_row(row: sqlite3.Row) -> ResultRecord:
        return ResultRecord(
            scenario_id=row["scenario_id"],
            scene_version=int(row["scene_version"]),
            exact=bool(row["exact"]),
            manifest=json.loads(row["manifest_json"]),
            payload_path=row["payload_path"],
            checksum=row["checksum"],
            created_at=row["created_at"],
            job_id=row["job_id"],
        )

    def republish_result_at_version(
        self,
        scenario_id: str,
        source_version: int,
        target_version: int,
        *,
        job_id: str | None,
        manifest_override: Mapping[str, Any] | None = None,
    ) -> None:
        """Copy an existing result's bytes to a new scene version.

        Used when a job coalesces to a no-op: the scene equals the previous
        exact state, so the previous payload is re-served under the new version
        and ``exact_result_version`` advances without recomputation.

        ``manifest_override`` (r2a-fix M1) merges keys into the republished
        manifest — a DEEP COPY of the source manifest, so the caller's
        dict and the source row never alias the stored bytes. The
        ``limitations`` list MERGES (override entries append to the
        source's, e.g. a disclosure entry riding the r2a-fix
        ``families_skipped`` re-serve); every other override key replaces.
        ``scene_version`` and ``payload_url`` are store-owned and set last:
        an override may not re-key the result. Payload bytes and the
        manifest's ``checksum`` stay the source's — this republish did no
        work, and its provenance is the producing version's.
        """
        source = self.get_result(scenario_id, source_version)
        if source is None:
            raise StoreError(
                f"cannot republish version {target_version}: no stored result for "
                f"version {source_version}"
            )
        manifest = deepcopy(source.manifest)
        if manifest_override is not None:
            override = deepcopy(dict(manifest_override))
            merged_limitations = override.pop("limitations", None)
            manifest.update(override)
            if merged_limitations is not None:
                existing = manifest.get("limitations")
                if existing is None:
                    existing = ()
                elif isinstance(existing, str):
                    existing = (existing,)
                elif not isinstance(existing, Sequence):
                    existing = (existing,)
                manifest["limitations"] = [str(item) for item in existing] + [
                    str(item) for item in merged_limitations
                ]
        manifest["scene_version"] = int(target_version)
        manifest["payload_url"] = result_payload_url(scenario_id, target_version)
        self.publish_result(
            scenario_id,
            target_version,
            manifest=manifest,
            payload=source.payload_bytes(),
            exact=True,
            job_id=job_id,
        )

    # -- exports -----------------------------------------------------------------

    def record_export(
        self,
        scenario_id: str,
        *,
        scene_version: int,
        format: str,
        variables: Sequence[str] | None,
        time_indices: Sequence[int] | None,
    ) -> str:
        export_id = new_export_id()
        with self._write() as conn:
            conn.execute(
                "INSERT INTO exports (export_id, scenario_id, scene_version, format, "
                "variables_json, time_indices_json, status, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, 'accepted', ?)",
                (
                    export_id,
                    scenario_id,
                    int(scene_version),
                    format,
                    json.dumps(list(variables)) if variables is not None else None,
                    json.dumps([int(t) for t in time_indices]) if time_indices is not None else None,
                    _now_utc(),
                ),
            )
        return export_id

    def list_exports(self, scenario_id: str) -> list[dict[str, Any]]:
        rows = self._read_all(
            "SELECT * FROM exports WHERE scenario_id = ? ORDER BY rowid", (scenario_id,)
        )
        return [
            {
                "export_id": row["export_id"],
                "scene_version": int(row["scene_version"]),
                "format": row["format"],
                "variables": json.loads(row["variables_json"]) if row["variables_json"] else None,
                "time_indices": (
                    json.loads(row["time_indices_json"]) if row["time_indices_json"] else None
                ),
                "status": row["status"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    # -- introspection ---------------------------------------------------------

    def job_status_counts(self) -> dict[str, int]:
        rows = self._read_all("SELECT status, COUNT(*) AS n FROM jobs GROUP BY status")
        return {row["status"]: int(row["n"]) for row in rows}

    def latest_finished_job_at(self, status: str | None = None) -> str | None:
        """Most recent ``finished_at`` among jobs, optionally one status."""
        if status is None:
            row = self._read(
                "SELECT finished_at FROM jobs WHERE finished_at IS NOT NULL "
                "ORDER BY finished_at DESC LIMIT 1"
            )
        else:
            row = self._read(
                "SELECT finished_at FROM jobs "
                "WHERE finished_at IS NOT NULL AND status = ? "
                "ORDER BY finished_at DESC LIMIT 1",
                (status,),
            )
        return row["finished_at"] if row is not None else None

    def result_count(self) -> int:
        row = self._read("SELECT COUNT(*) AS n FROM results")
        return int(row["n"]) if row is not None else 0

    def _require_scenario_locked(
        self, conn: sqlite3.Connection, scenario_id: str
    ) -> ScenarioRecord:
        row = conn.execute(
            "SELECT * FROM scenarios WHERE scenario_id = ?", (scenario_id,)
        ).fetchone()
        if row is None:
            raise ScenarioNotFound(scenario_id)
        return self._scenario_row(row)


def result_manifest_url(scenario_id: str, scene_version: int) -> str:
    return f"/api/v1/scenarios/{scenario_id}/results/{int(scene_version)}"


def result_payload_url(scenario_id: str, scene_version: int) -> str:
    return result_manifest_url(scenario_id, scene_version) + "/payload"


def scenario_url(scenario_id: str) -> str:
    return f"/api/v1/scenarios/{scenario_id}"
