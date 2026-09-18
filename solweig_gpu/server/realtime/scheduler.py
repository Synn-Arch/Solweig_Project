# SPDX-License-Identifier: GPL-3.0-only
"""The realtime fast lane scheduler (R2b): deadline-isolated fast analysis.

Implements ``docs/incremental_design_tool/realtime_collaboration/
epoch_scheduler.md`` "Fast queue" on top of the R1 epoch plane and r2a's
exact lane:

* **Deadline-isolated thread** — one dedicated daemon thread per app (the
  ``jobs.py`` ticker pattern: ``Event.wait`` cadence + generation counter
  for prompt, idempotent stop). Fast kernels run in-process on that
  thread; the lane NEVER waits behind the exact solver's full-tile job
  (queue separation) and never blocks the epoch close pipeline — the
  epochs.py tail hook (:meth:`FastLaneScheduler.on_epoch_closed`) is
  enqueue-and-return.
* **Epoch consumption** — after an epoch close commits the canonical
  state, the lane plans from the EPOCH-FINAL canonical state. It advances
  the epoch row ``reducing -> fast_planned`` only: ``fast_published`` is a
  TERMINAL status whose semantics r2a's exact lane already owns (terminal
  epochs count as exact-consumed in ``jobs.py``), so fast-side completion
  is recorded in ``scenarios.fast_revision`` plus this scheduler's own
  per-epoch state (vocabulary gap flagged in the r2b handoff). An epoch
  the fast lane touched stays recoverable and exactly as consumable as
  before.
* **Result classes** (degradation ladder, service_level_contract.md):
  ``fast_exact`` for view/cached switches (view-only epochs; revisions the
  exact lane already covers), ``visual_pending`` as the default when no
  fast kernel can meet the deadline — including the case where the
  preferred kernel's PREDICTED cost does not fit the remaining deadline
  (the lane then publishes the lower class BEFORE the deadline without
  running the kernel). ``fast_qualified`` (R7) is the COMPENSATED class: a
  provisional anchored delta against the last exact planes. The lane
  publishes it only with a COMPLETE qualifier payload
  (:mod:`~solweig_gpu.server.realtime.qualification` validates structure —
  model version, qualification domain, measured holdout error evidence,
  anchor, base planes, reconciliation); a payload with any defect is
  downgraded to ``visual_pending`` loudly (typed counter + payload
  disclosure — never reduce precision silently). A published qualified
  frame is retired by the SUPERSESSION fence the moment the exact lane
  covers its revision (:meth:`FastLaneScheduler._supersede_stale_qualified`).
  The class rides the ``fast_revision`` SSE event.
* **Revision model** — publishes advance ``scenarios.fast_revision``
  monotonically (never past ``workspace_revision``, never below
  ``exact_revision``) in one store transaction, keeping
  ``exact_revision <= fast_revision <= workspace_revision``. Fast payloads
  are reproducible from the durable plane (canonical state + operation
  log), so v1 keeps them in memory + SSE only; no new store table.
* **Discovery from durable state** — every pass derives due work from
  ``scenarios.fast_revision < scene_version`` (plus committed epoch
  revisions), so a lost wake, a queue overflow, or a restart self-heals on
  the next pass (recovery = one pass at start). The hook only ACCELERATES
  the next pass to sub-tick latency. Scan-discovered revisions wait ONE
  pass (grace) before publishing: the scan can observe a revision inside
  the close pipeline's commit-to-broadcast window, and the grace
  guarantees a ``fast_revision`` frame never reaches subscribers before
  the ``canonical_revision`` frame for the same revision (review F1);
  hook-woken work is already post-broadcast and owes no grace.
* **Bounded queue** — pending wake/prediction state is bounded per
  workspace count; overflow wakes are counted and dropped (the scan
  re-derives them), and :mod:`~solweig_gpu.server.realtime.admission`
  rejects work BEFORE durable acceptance so the lane is the second line,
  not the only one.

Telemetry (standard names): ``fast_compute_ms``, ``publish_ms``,
``fast_total_ms`` (``t_fast_publish - t_accept`` of the epoch's oldest
operation), ``result_class_total{class}``,
``stale_publish_rejected_total``.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Mapping, Protocol, Sequence

from solweig_gpu.server.realtime.qualification import qualifier_defects
from solweig_gpu.server.realtime.telemetry import (
    StandardMetrics,
    TelemetryRegistry,
    register_standard_metrics,
)
from solweig_gpu.server.realtime.types import FastResultClass
from solweig_gpu.server.store import Store

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_FAST_DEADLINE_MS",
    "DEFAULT_MAX_PENDING_WORKSPACES",
    "DEFAULT_PUBLICATION_HISTORY_PER_WORKSPACE",
    "FAST_LANE_THREAD_NAME",
    "FastKernelResult",
    "FastLaneScheduler",
    "FastPlan",
    "ViewCacheKernel",
]

#: realtime_contract.yaml: fast_revision_p99_ms 1000.
DEFAULT_FAST_DEADLINE_MS = 1000.0

#: Bound on workspaces tracked in the pending queue (admission is the
#: primary bound; this is the lane's own memory fence).
DEFAULT_MAX_PENDING_WORKSPACES = 64

#: Per-workspace in-memory fast payload history (SSE catch-up is the
#: durable plane's job; this serves introspection/telemetry).
DEFAULT_PUBLICATION_HISTORY_PER_WORKSPACE = 64

FAST_LANE_THREAD_NAME = "solweig-fast-lane"

#: Idle cadence of the discovery scan when nothing wakes the lane (ms).
#: Matches the epoch tick: worst-case discovery lag equals one epoch
#: window when no hook fires (restart, hook drop).
_SCAN_CADENCE_MS = 100.0

_STORE_STAMP_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"


def _parse_stamp(stamp: str) -> datetime:
    """Parse a store-format wall stamp to an aware datetime."""
    return datetime.strptime(stamp, _STORE_STAMP_FORMAT).replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class FastPlan:
    """What the fast kernel plans from: one due epoch, epoch-final state.

    ``deadline_at`` is the absolute wall deadline derived from the OLDEST
    accepted operation in the epoch (``t_accept + deadline_ms``) — the
    same anchor the SLA's ``t_fast_publish - t_accept`` latency uses.
    """

    workspace_id: str
    epoch_id: int
    workspace_revision: int
    canonical_revision: int
    deadline_at: datetime
    now: datetime
    families: tuple[str, ...]
    operation_count: int
    #: Epoch-final canonical families snapshot (planning input).
    canonical_state: Mapping[str, Any] = field(default_factory=dict)

    def remaining_ms(self) -> float:
        return (self.deadline_at - self.now).total_seconds() * 1000.0


@dataclass(frozen=True)
class FastKernelResult:
    """One fast kernel's outcome: a result class plus a wire payload."""

    result_class: FastResultClass
    payload: Mapping[str, Any] = field(default_factory=dict)


class FastKernel(Protocol):
    """A deadline-bound fast kernel (in-process; numpy releases the GIL)."""

    name: str

    def predicted_cost_ms(self, plan: FastPlan) -> float:
        """The kernel's predicted compute for this plan (must be cheap)."""

    def run(self, plan: FastPlan) -> FastKernelResult:
        """Run the kernel; MUST stay inside the plan's remaining budget."""


class ViewCacheKernel:
    """Default v1 kernel: view/cached switches, else ``visual_pending``.

    ``fast_exact`` requires a thermal-neutral epoch — every operation
    view-only (per-user/workspace view selections never invalidate
    scientific state, collaborative_state.md) — or an exact result that
    already covers the revision (cache switch). Everything scientific has
    no bounded v1 kernel, so the lane discloses the pending visual state
    instead of guessing.
    """

    name = "view_cache_v1"

    def predicted_cost_ms(self, plan: FastPlan) -> float:
        # Classification over already-fetched plan data: effectively free.
        return 1.0

    def run(self, plan: FastPlan) -> FastKernelResult:
        return FastKernelResult(
            result_class=self.classify(plan),
            payload={"kernel": self.name},
        )

    def classify(self, plan: FastPlan) -> FastResultClass:
        if plan.families and all(f == "output_view" for f in plan.families):
            return "fast_exact"
        return "visual_pending"


@dataclass
class _PendingItem:
    """Discovered, not-yet-published fast work for one workspace."""

    workspace_id: str
    epoch_id: int
    workspace_revision: int
    deadline_at: datetime
    discovered_at: datetime
    predicted_ms: float = 0.0
    #: True when the DURABLE-STATE SCAN discovered this revision (it may
    #: have been observed between the epoch pipeline's durable commit and
    #: its canonical broadcast submission — see :meth:`_discover`).
    from_scan: bool = False
    #: Scan-discovered work waits ONE pass (grace) before publishing so
    #: the fast frame can never precede the canonical frame for the same
    #: revision. Cleared by the next pass (see ``_scan_grace``).
    deferred: bool = False


@dataclass(frozen=True)
class FastPublication:
    """One published fast revision (in-memory payload + wire shape)."""

    workspace_id: str
    epoch_id: int
    workspace_revision: int
    fast_revision: int
    result_class: FastResultClass
    exact_revision: int
    exact_base_revision: int
    met_deadline: bool
    latency_ms: float
    published_at: str
    payload: Mapping[str, Any] = field(default_factory=dict)


class FastLaneScheduler:
    """Deadline-ordered fast queue on a dedicated thread, one per app."""

    def __init__(
        self,
        store: Store,
        *,
        hub: Any = None,
        telemetry: TelemetryRegistry | None = None,
        kernel: FastKernel | None = None,
        deadline_ms: float = DEFAULT_FAST_DEADLINE_MS,
        max_pending_workspaces: int = DEFAULT_MAX_PENDING_WORKSPACES,
        publication_history: int = DEFAULT_PUBLICATION_HISTORY_PER_WORKSPACE,
        utc_now: Callable[[], str] | None = None,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        from solweig_gpu.server.store import _now_utc as _default_utc_now

        self._store = store
        self._hub = hub
        self._kernel = kernel if kernel is not None else ViewCacheKernel()
        self._telemetry = telemetry
        self._metrics: StandardMetrics | None = (
            register_standard_metrics(telemetry) if telemetry is not None else None
        )
        self.deadline_ms = float(deadline_ms)
        self.max_pending_workspaces = int(max_pending_workspaces)
        self._history = int(publication_history)
        self._utc_now = utc_now if utc_now is not None else _default_utc_now
        self._monotonic_ns = monotonic_ns
        self._lock = threading.Lock()
        self._pending: dict[str, _PendingItem] = {}
        self._dropped_wakes = 0
        #: Latest fast publication per workspace (revision), bounded. The
        #: tail hook consults it to skip re-broadcasts of work the lane
        #: already published (boot re-drive) — otherwise a pending entry
        #: no scan ever matches would strand a reservation (review F4).
        self._published_keys: dict[str, deque[int]] = {}
        #: Per workspace, the revision whose ONE-PASS scan grace is spent.
        #: Scan-discovered work defers exactly one pass so its fast frame
        #: can never precede the canonical frame (review F1).
        self._scan_grace: dict[str, int] = {}
        self._publications: dict[str, deque[FastPublication]] = {}
        #: Fairness cursor: workspaces at equal deadlines rotate.
        self._rr_cursor = 0
        self._wake = threading.Event()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._generation = 0

    # -- epochs.py tail hook (enqueue-and-return) ---------------------------

    def on_epoch_closed(
        self, workspace_id: str, epoch_id: int, event: Mapping[str, Any]
    ) -> None:
        """Tail hook entry: record the wake and return immediately.

        NEVER runs fast work inline — the epoch close pipeline must not
        block on the fast lane. The actual work happens on the lane's
        thread (or the next explicit ``run_once`` in tests). A full queue
        drops the wake (counted): the discovery scan re-derives it.
        """
        revision = event.get("workspace_revision")
        now = _parse_stamp(self._utc_now())
        # Deadline anchor: the epoch's oldest t_accept (the broadcast event
        # carries the operations in server order), same anchor the SLA's
        # t_fast_publish - t_accept latency uses. Falls back to now when
        # the event carries no operations (defensive).
        deadline = now + timedelta(milliseconds=self.deadline_ms)
        operations = event.get("operations")
        if isinstance(operations, Sequence) and operations:
            first = operations[0]
            if isinstance(first, Mapping) and first.get("accepted_at"):
                try:
                    accepted = _parse_stamp(str(first["accepted_at"]))
                    deadline = accepted + timedelta(milliseconds=self.deadline_ms)
                except ValueError:
                    pass
        with self._lock:
            published = self._published_keys.get(workspace_id)
            if published and int(revision or 0) <= published[-1]:
                # Review F4: a boot re-drive (or any re-broadcast) of a
                # revision the lane ALREADY fast-published. The scan will
                # never match it again (fast_revision is caught up), so
                # enqueueing it would strand a pending reservation until
                # the next epoch. Quietly skip: no new work exists.
                logger.debug(
                    "fast lane ignoring re-broadcast of published revision "
                    "%s/%s",
                    workspace_id,
                    revision,
                )
                return
            if (
                workspace_id not in self._pending
                and len(self._pending) >= self.max_pending_workspaces
            ):
                self._dropped_wakes += 1
                return
            current = self._pending.get(workspace_id)
            if current is None or int(revision or 0) > current.workspace_revision:
                # Hook-sourced (``from_scan`` stays False): the hook fires
                # AFTER the canonical broadcast was submitted, so this
                # work owes no ordering grace and may publish next pass.
                self._pending[workspace_id] = _PendingItem(
                    workspace_id=workspace_id,
                    epoch_id=int(epoch_id),
                    workspace_revision=int(revision or 0),
                    deadline_at=deadline,
                    discovered_at=now,
                )
        self._wake.set()

    # -- admission seam -------------------------------------------------------

    def reserved_ms(self, workspace_id: str) -> float:
        """Fast compute already booked for unpublished work (admission).

        Uses the SAME epoch-coalesced, PAYLOAD-AWARE formula as admission's
        :func:`~solweig_gpu.server.realtime.admission.predict_epoch_compute_ms`
        so reservations and checks agree — a land-cover paint reserves its
        window cell count exactly as admission books it (review F2:
        stripping the payload under-booked by the cell-derived cost).
        A lane that is not running reserves nothing: with the thread
        stopped nothing will be served, and admission's own prediction is
        the only bound that applies.
        """
        from solweig_gpu.server.realtime.admission import (
            predict_epoch_compute_ms,
        )

        if not self.running:
            return 0.0
        with self._lock:
            item = self._pending.get(workspace_id)
        if item is None:
            return 0.0
        operations = self._store.operations_for_epoch(workspace_id, item.epoch_id)
        return predict_epoch_compute_ms(
            [
                {
                    "source_family": record.source_family,
                    "payload": record.payload,
                }
                for record in operations
            ]
        )

    def saturated(self) -> bool:
        """Whether the pending queue is at its bound (503 signal)."""
        with self._lock:
            return len(self._pending) >= self.max_pending_workspaces

    def pending_workspaces(self) -> list[str]:
        with self._lock:
            return sorted(self._pending)

    # -- introspection --------------------------------------------------------

    def latest_publication(self, workspace_id: str) -> FastPublication | None:
        with self._lock:
            queue = self._publications.get(workspace_id)
        return queue[-1] if queue else None

    def publications(self, workspace_id: str) -> list[FastPublication]:
        with self._lock:
            queue = self._publications.get(workspace_id)
        return list(queue) if queue else []

    @property
    def dropped_wakes(self) -> int:
        with self._lock:
            return self._dropped_wakes

    # -- discovery / processing ------------------------------------------------

    def run_once(self) -> list[tuple[str, int]]:
        """One fast-lane pass: discover due work, publish deadline-first.

        Scan-discovered revisions in their FIRST pass are deferred
        (one-pass grace — see :meth:`_discover`) and publish on the next
        pass. Returns the ``(workspace_id, epoch_id)`` pairs published
        this pass. Per-workspace failures are contained (one workspace's
        fast work never silences another's); a persistently failing
        workspace is retried on the next pass because its revision stays
        due.
        """
        published: list[tuple[str, int]] = []
        published.extend(self._supersede_stale_qualified())
        items = self._discover()
        if not items:
            return published
        ready = [item for item in items if not item.deferred]
        deferred = [item for item in items if item.deferred]
        if deferred:
            logger.debug(
                "fast lane grace: deferring %d scan-discovered revision(s) "
                "one pass (canonical-before-fast ordering)",
                len(deferred),
            )
        if not ready:
            return published
        ready.sort(key=lambda item: (item.deadline_at, item.workspace_id))
        # Deadline order first; rotate equal-deadline batches for fairness
        # (deficit-free v1: equal deadlines interleave by cursor).
        ordered = self._fair_rotate(ready)
        for item in ordered:
            try:
                publication = self._publish_fast(item)
            except Exception:  # noqa: BLE001 - one workspace never stops the rest
                logger.exception(
                    "fast publish failed for %s/%s (revision stays due)",
                    item.workspace_id,
                    item.epoch_id,
                )
                continue
            if publication is not None:
                published.append((item.workspace_id, item.epoch_id))
        return published

    def _supersede_stale_qualified(self) -> list[tuple[str, int]]:
        """Exact-caught-up fence (R7): retire qualified frames the exact
        lane has overtaken.

        A ``fast_qualified`` frame is a provisional delta anchored to an
        exact base. Once the exact lane publishes a result COVERING the
        frame's own workspace revision, the frame must never survive as
        the authoritative view: re-publish the revision as ``fast_exact``
        (superseded disclosure), exactly once — the superseding publication
        replaces the qualified one in history, so this is idempotent by
        construction. No revision churn: ``fast_revision`` already covers
        the revision; only the class and payload move.
        """
        superseded: list[tuple[str, int]] = []
        with self._lock:
            workspaces = list(self._publications)
        for workspace_id in workspaces:
            publication = self.latest_publication(workspace_id)
            if publication is None or publication.result_class != "fast_qualified":
                continue
            scenario = self._store.get_scenario(workspace_id)
            if scenario is None:
                continue
            exact_revision = int(scenario.exact_result_version)
            if exact_revision < publication.workspace_revision:
                continue  # exact has not caught up to this revision yet
            published_at = self._utc_now()
            payload = {
                "kernel": "supersession_fence",
                "superseded_qualified": True,
                "superseded_result_class": "fast_qualified",
                "exact_revision": exact_revision,
                "qualified_exact_base_revision": publication.exact_base_revision,
            }
            replacement = FastPublication(
                workspace_id=workspace_id,
                epoch_id=publication.epoch_id,
                workspace_revision=publication.workspace_revision,
                fast_revision=publication.fast_revision,
                result_class="fast_exact",
                exact_revision=exact_revision,
                exact_base_revision=exact_revision,
                met_deadline=True,
                latency_ms=0.0,
                published_at=published_at,
                payload=payload,
            )
            self._record_publication(replacement)
            event: dict[str, Any] = {
                "workspace_id": workspace_id,
                "epoch_id": publication.epoch_id,
                "workspace_revision": publication.workspace_revision,
                "fast_revision": publication.fast_revision,
                "result_class": "fast_exact",
                "exact_revision": exact_revision,
                "exact_base_revision": exact_revision,
                "operations_count": 0,
                "families": [],
                "met_deadline": True,
                "latency_ms": 0.0,
                "published_at": published_at,
                "payload": dict(payload),
            }
            if self._hub is not None:
                try:
                    self._hub.broadcast_fast(workspace_id, event)
                except Exception:  # noqa: BLE001 - the fence must never kill the lane
                    logger.exception(
                        "supersession broadcast failed for %s (publication recorded)",
                        workspace_id,
                    )
            self._observe_class("fast_exact")
            logger.info(
                "superseded fast_qualified frame %s@r%d with fast_exact "
                "(exact revision %d caught up)",
                workspace_id,
                publication.workspace_revision,
                exact_revision,
            )
            superseded.append((workspace_id, publication.epoch_id))
        return superseded

    def _fair_rotate(self, items: list[_PendingItem]) -> list[_PendingItem]:
        """Round-robin rotate the equal-deadline prefix for fairness."""
        if len(items) < 2:
            return items
        head = items[0].deadline_at
        span = 1
        while span < len(items) and items[span].deadline_at == head:
            span += 1
        if span < 2:
            return items
        with self._lock:
            self._rr_cursor = (self._rr_cursor + 1) % span
            offset = self._rr_cursor
        return items[offset:span] + items[:offset] + items[span:]

    def _discover(self) -> list[_PendingItem]:
        """Derive due fast work from DURABLE state (self-healing scan).

        Due = a workspace whose ``fast_revision`` lags its
        ``workspace_revision`` (``scene_version``) with a committed epoch
        revision in the gap. The NEWEST due epoch is the target: its
        epoch-final canonical state subsumes every older due revision
        (canonical state is cumulative), so a lagging lane jumps to the
        newest revision instead of replaying each epoch.

        Ordering grace (review F1): the scan reads committed rows, so it
        can observe a revision INSIDE the close pipeline's
        commit-to-broadcast window and would broadcast ``fast_revision``
        before the canonical frame for the same revision reached
        subscribers. Scan-discovered work therefore waits exactly ONE
        pass before publishing (the hook-woken path fires after the
        canonical broadcast submission and owes no grace). Cost: at most
        one scan cadence (100 ms of the 1,000 ms budget); the
        durable-state self-heal is unchanged — a lost wake, an overflow
        drop, or a restart publishes one pass later than before.
        """
        rows = self._store.scenarios_with_fast_lag()
        scanned: list[_PendingItem] = []
        now = _parse_stamp(self._utc_now())
        for record in rows:
            workspace_id = record.scenario_id
            scene_version = int(record.scene_version)
            fast_revision = int(record.fast_revision)
            target = None
            for epoch in self._store.epoch_records(workspace_id):
                revision = epoch.workspace_revision
                if revision is None:
                    continue
                if fast_revision < int(revision) <= scene_version and (
                    target is None or int(revision) > target.workspace_revision
                ):
                    target = epoch
            if target is None:
                continue
            operations = self._store.operations_for_epoch(
                workspace_id, int(target.epoch_id)
            )
            if operations:
                anchor = _parse_stamp(operations[0].accepted_at)
            else:  # defensive: an assigned epoch always has operations
                anchor = now
            deadline = anchor + timedelta(milliseconds=self.deadline_ms)
            scanned.append(
                _PendingItem(
                    workspace_id=workspace_id,
                    epoch_id=int(target.epoch_id),
                    workspace_revision=int(target.workspace_revision or 0),
                    deadline_at=deadline,
                    discovered_at=now,
                    from_scan=True,
                )
            )
        # Mirror discovery into the pending map so admission sees the same
        # reservations the lane is about to serve (bounded; wake drops are
        # self-healing by construction). Grace bookkeeping shares the lock:
        # a revision's grace is spent exactly once.
        items: list[_PendingItem] = []
        with self._lock:
            for item in scanned:
                existing = self._pending.get(item.workspace_id)
                if (
                    existing is not None
                    and existing.workspace_revision == item.workspace_revision
                    and not existing.from_scan
                ):
                    # A hook already woke us for THIS revision
                    # (post-broadcast): the hook item publishes without
                    # grace; the scan twin is redundant.
                    items.append(existing)
                    continue
                if self._scan_grace.get(item.workspace_id) != (
                    item.workspace_revision
                ):
                    item.deferred = True  # one-pass grace, spent below
                    self._scan_grace[item.workspace_id] = item.workspace_revision
                self._pending[item.workspace_id] = item
                items.append(item)
        return items

    def _publish_fast(self, item: _PendingItem) -> FastPublication | None:
        """Plan, budget-check, run the kernel, advance ``fast_revision``.

        Returns the publication, or ``None`` when the work was found stale
        (another driver already advanced the revision past the fence).
        """
        store = self._store
        workspace_id = item.workspace_id
        epoch = store.get_epoch(workspace_id, item.epoch_id)
        if epoch is None or epoch.workspace_revision is None:
            return None
        revision = int(epoch.workspace_revision)
        scenario = store.get_scenario(workspace_id)
        if scenario is None or revision > int(scenario.scene_version):
            return None  # stale target: the workspace moved past it

        operations = store.operations_for_epoch(workspace_id, item.epoch_id)
        families = tuple(dict.fromkeys(str(op.source_family) for op in operations))
        canonical = store.latest_canonical_state(workspace_id)
        canonical_revision = (
            int(canonical.workspace_revision) if canonical is not None else 0
        )
        now = _parse_stamp(self._utc_now())
        plan = FastPlan(
            workspace_id=workspace_id,
            epoch_id=item.epoch_id,
            workspace_revision=revision,
            canonical_revision=canonical_revision,
            deadline_at=item.deadline_at,
            now=now,
            families=families,
            operation_count=len(operations),
            canonical_state=dict(canonical.families) if canonical else {},
        )

        # Advance the epoch row to fast_planned (NON-terminal on purpose:
        # fast completion must not settle the exact lane's books).
        if epoch.status == "reducing":
            store.mark_epoch_status(
                workspace_id, item.epoch_id, "fast_planned", workspace_revision=revision
            )

        # Deadline gate: if the preferred kernel cannot finish inside the
        # remaining budget, publish the LOWER class now instead of missing
        # the deadline (degradation ladder; the kernel never runs).
        compute_started_ns = self._monotonic_ns()
        predicted = float(self._kernel.predicted_cost_ms(plan))
        remaining = plan.remaining_ms()
        result_class: FastResultClass
        kernel_payload: Mapping[str, Any] = {}
        if predicted > remaining:
            result_class = "visual_pending"
            kernel_payload = {
                "kernel": getattr(self._kernel, "name", "kernel"),
                "downgraded": True,
                "predicted_ms": predicted,
                "remaining_ms": remaining,
            }
        else:
            kernel_result = self._kernel.run(plan)
            result_class = kernel_result.result_class
            kernel_payload = dict(kernel_result.payload)
            if result_class == "fast_qualified":
                # R7 un-reserve: the class may ride the wire ONLY with a
                # COMPLETE anchored qualifier payload. A fast_qualified
                # result with missing/malformed qualifier metadata is a
                # kernel defect: downgrade loudly (typed telemetry counter
                # + payload disclosure) and keep the deadline. The fence is
                # structural — "never reduce precision silently".
                defects = qualifier_defects(kernel_payload)
                if defects:
                    logger.error(
                        "fast kernel %r returned fast_qualified with "
                        "qualifier defects %s for %s/%s; downgrading to "
                        "visual_pending",
                        getattr(self._kernel, "name", "kernel"),
                        defects,
                        workspace_id,
                        item.epoch_id,
                    )
                    result_class = "visual_pending"
                    kernel_payload = {
                        **kernel_payload,
                        "downgraded": True,
                        "downgrade_reasons": list(defects),
                    }
                    for defect in defects:
                        self._observe_downgrade(defect)
        self._observe_ms("fast_compute_ms", compute_started_ns)

        # Cache-switch classes the default kernel cannot know about: an
        # exact result already covering the target revision IS the best
        # deadline-bound representation of it.
        exact_revision = int(scenario.exact_result_version)
        if exact_revision >= revision and result_class != "fast_exact":
            result_class = "fast_exact"

        publish_started_ns = self._monotonic_ns()
        published_at = self._utc_now()
        # ONE store transaction: stale-fenced, monotonic fast_revision
        # advance (never past workspace_revision, never below
        # exact_revision) + the compensated anchor — the typed accessor
        # (r2b-review F6: no raw scenarios SQL in this module).
        advance = store.advance_fast_revision(
            workspace_id, revision, exact_revision, updated_at=published_at
        )
        if advance is None:
            self._observe("stale_publish_rejected_total")
            return None
        new_fast = advance.fast_revision
        exact_now = advance.exact_revision
        # The ANCHOR rides qualified frames: the exact revision whose planes
        # the delta was computed against (validated payload field, clamped
        # to the transaction's exact floor — an anchor above the store's
        # exact revision is a kernel defect clamped to reality).
        exact_base = exact_now
        if result_class == "fast_qualified":
            anchor = kernel_payload.get("exact_base_revision")
            if (
                isinstance(anchor, int)
                and not isinstance(anchor, bool)
                and 0 <= anchor <= exact_now
            ):
                exact_base = anchor

        latency_ms = max(
            0.0,
            (
                _parse_stamp(published_at)
                - _parse_stamp(operations[0].accepted_at if operations else published_at)
            ).total_seconds()
            * 1000.0,
        )
        publication = FastPublication(
            workspace_id=workspace_id,
            epoch_id=item.epoch_id,
            workspace_revision=revision,
            fast_revision=new_fast,
            result_class=result_class,
            exact_revision=exact_now,
            exact_base_revision=exact_base,
            met_deadline=latency_ms <= self.deadline_ms,
            latency_ms=latency_ms,
            published_at=published_at,
            payload=kernel_payload,
        )
        self._record_publication(publication)
        event: dict[str, Any] = {
            "workspace_id": workspace_id,
            "epoch_id": item.epoch_id,
            "workspace_revision": revision,
            "fast_revision": new_fast,
            "result_class": result_class,
            "exact_revision": exact_now,
            "exact_base_revision": exact_base,
            "operations_count": len(operations),
            "families": list(families),
            "met_deadline": publication.met_deadline,
            "latency_ms": latency_ms,
            "published_at": published_at,
            **({"payload": dict(kernel_payload)} if kernel_payload else {}),
        }
        if self._hub is not None:
            try:
                self._hub.broadcast_fast(workspace_id, event)
            except Exception:  # noqa: BLE001 - publish must never kill the lane
                logger.exception(
                    "fast broadcast failed for %s/%s (revision stays advanced)",
                    workspace_id,
                    item.epoch_id,
                )
        self._observe_ms("publish_ms", publish_started_ns)
        self._observe_value("fast_total_ms", latency_ms)
        self._observe_class(result_class)
        with self._lock:
            self._pending.pop(workspace_id, None)
            grace_revision = self._scan_grace.get(workspace_id)
            if grace_revision is not None and grace_revision <= revision:
                del self._scan_grace[workspace_id]
        return publication

    def _record_publication(self, publication: FastPublication) -> None:
        with self._lock:
            queue = self._publications.setdefault(publication.workspace_id, deque())
            queue.append(publication)
            while len(queue) > self._history:
                queue.popleft()
            # The tail hook's skip index (review F4/F5): per-workspace
            # published REVISIONS in order (monotonic, so [-1] is the
            # latest) — a re-broadcast at or below the latest published
            # revision is stale boot re-drive, not new work.
            keys = self._published_keys.setdefault(publication.workspace_id, deque())
            keys.append(publication.workspace_revision)
            while len(keys) > 4 * self._history:
                keys.popleft()

    # -- recovery / daemon thread (jobs.py ticker pattern) -----------------

    def recover(self) -> list[tuple[str, int]]:
        """Re-publish fast revisions owed from before a restart.

        Idempotent by construction: everything derives from durable state
        (``fast_revision`` lag), and an in-flight crash between the status
        mark and the transaction simply leaves the revision due for the
        next pass.
        """
        return self.run_once()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Recovery pass + the daemon loop (idempotent)."""
        if self.running:
            return
        self._stop_event.clear()
        self._generation += 1
        generation = self._generation
        try:
            recovered = self.recover()
            if recovered:
                logger.info("fast lane recovery published %d epoch(s)", len(recovered))
        except Exception:  # noqa: BLE001 - never block startup on recovery
            logger.exception("fast lane recovery at start failed")
        self._thread = threading.Thread(
            target=self._run_loop,
            name=FAST_LANE_THREAD_NAME,
            daemon=True,
            kwargs={"generation": generation},
        )
        self._thread.start()

    def _run_loop(self, generation: int) -> None:
        scan_s = _SCAN_CADENCE_MS / 1000.0
        while True:
            # Wake-driven with a scan-cadence ceiling: the epochs tail hook
            # sets _wake for sub-tick latency; a lost wake (drop/restart)
            # still self-heals because the cadence pass re-scans.
            self._wake.wait(scan_s)
            if self._stop_event.is_set() or generation != self._generation:
                return
            self._wake.clear()
            try:
                self.run_once()
            except Exception:  # noqa: BLE001 - the lane must survive anything
                logger.exception("fast lane tick failed")

    def stop(self, timeout: float = 1.0) -> bool:
        """Signal the loop to stop and join it. Returns whether it exited."""
        self._generation += 1
        self._stop_event.set()
        self._wake.set()  # break the wait promptly
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout=timeout)
        exited = not thread.is_alive()
        if exited:
            self._thread = None
        return exited

    # -- telemetry (never blocks, never raises) -----------------------------

    def _observe(self, name: str) -> None:
        if self._metrics is None:
            return
        try:
            getattr(self._metrics, name).inc()
        except Exception:  # pragma: no cover - telemetry must never break flow
            logger.exception("telemetry observe failed for %s", name)

    def _observe_class(self, result_class: str) -> None:
        if self._metrics is None:
            return
        try:
            self._metrics.result_class_total.inc(labels={"class": result_class})
        except Exception:  # pragma: no cover
            logger.exception("telemetry observe failed for result_class")

    def _observe_downgrade(self, defect: str) -> None:
        """Count a qualifier downgrade by its bounded defect id (R7)."""
        if self._metrics is None:
            return
        try:
            self._metrics.fast_qualified_downgraded_total.inc(
                labels={"reason": defect}
            )
        except Exception:  # pragma: no cover
            logger.exception("telemetry observe failed for qualified downgrade")

    def _observe_value(self, name: str, value: float) -> None:
        if self._metrics is None:
            return
        try:
            getattr(self._metrics, name).observe(value)
        except Exception:  # pragma: no cover
            logger.exception("telemetry observe failed for %s", name)

    def _observe_ms(self, name: str, started_ns: int) -> None:
        self._observe_value(name, (self._monotonic_ns() - started_ns) / 1_000_000.0)

    # -- context manager ------------------------------------------------------

    def __enter__(self) -> "FastLaneScheduler":
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()
