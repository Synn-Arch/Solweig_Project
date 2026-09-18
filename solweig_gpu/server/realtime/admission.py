# SPDX-License-Identifier: GPL-3.0-only
"""Admission control for the realtime fast lane (R2b).

Implements the backpressure half of
``docs/incremental_design_tool/realtime_collaboration/epoch_scheduler.md``
("Backpressure") and ``service_level_contract.md`` ("Admission envelope" +
"Overload behavior"):

* admission runs at INGEST, BEFORE the durable append — an operation batch
  outside the guaranteed envelope is rejected with typed
  :class:`AdmissionRejected` (429 for per-workspace bursts the client can
  fix by backing off; 503 for server-wide saturation) carrying retry
  metadata. Once a batch IS accepted it is never dropped to save a
  deadline (that promise belongs to the operation plane, not this gate).
* the envelope is the 1,000 ms fast-deadline budget apportioned per the
  ``epoch_scheduler.md`` table — ingest 100 ms, epoch wait <= 100 ms,
  reduce+plan 100 ms, compute 550 ms, encode/publish 150 ms. Admission
  accounts for elapsed ingest time, the fixed stages, the fast compute
  RESERVED ahead of the batch (outstanding fast work for the workspace,
  reported by :class:`~solweig_gpu.server.realtime.scheduler.FastLaneScheduler`),
  and the batch's PREDICTED fast cost (per-family booking).
* the envelope fields mirror ``realtime_contract.yaml``
  ``admission_envelope_required_fields`` (rates, per-epoch bounds, payload
  size, active-workspace count, compute budget).

Predicted costs are BOOKED capacity, not measured kernel times: the lane
reserves the budget the degradation ladder may need for a family (a
geometry epoch can downgrade to ``visual_pending`` for free, but admission
must still bound how much fast work piles up per second). Constants are
declarative tuning knobs, safe to retune from load evidence.

JOINT ADMISSIBILITY CAVEAT (review F3, lead ruling this wave): the
envelope bounds are INDEPENDENT maxima, and a batch sitting at several
maxima at once can still exceed the compute budget — e.g. the maximal
128-operation batch containing one 100k-cell land-cover paint books
~656 ms (400 ms cells + 256 ms per-op margin) against the 550 ms compute
budget and is rejected even though each individual bound admits it. That
single-batch asymmetry is INTENTIONAL conservative behavior (a maximal
diversified batch is also the one with the deepest worst-case degradation
ladder), not a defect; the constants await measured kernel times from the
bounded fast kernels (R3/R7/R8) before any retune. Batches that need both
bounds at once should split across epochs.

This module never touches the store and never blocks: it is pure
bookkeeping over in-memory sliding windows plus the injected capacity
reporter.
"""

from __future__ import annotations

import json
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Protocol

from solweig_gpu.server.models import ApiError

__all__ = [
    "DEFAULT_FAST_DEADLINE_BUDGET_MS",
    "STAGE_INGEST_MS",
    "STAGE_EPOCH_WAIT_MS",
    "STAGE_REDUCE_PLAN_MS",
    "STAGE_COMPUTE_MS",
    "STAGE_ENCODE_PUBLISH_MS",
    "AdmissionEnvelope",
    "AdmissionRejected",
    "FastAdmission",
    "predict_operation_cost_ms",
]

#: The one-second fast deadline (realtime_contract.yaml
#: ``slo_targets.fast_revision_p99_ms``).
DEFAULT_FAST_DEADLINE_BUDGET_MS = 1000.0

# -- epoch_scheduler.md "Deadline budgeting" stage budgets --------------------
# Targets, not measured facts; telemetry adapts thresholds from p95/p99.
STAGE_INGEST_MS = 100.0  # network, auth, durable append
STAGE_EPOCH_WAIT_MS = 100.0  # wait to epoch close (<= 100 ms)
STAGE_REDUCE_PLAN_MS = 100.0  # reduce + dependency plan
STAGE_COMPUTE_MS = 550.0  # fast computation
STAGE_ENCODE_PUBLISH_MS = 150.0  # encode, publish, broadcast

#: Fixed (non-compute) stages of the budget.
FIXED_STAGE_MS = STAGE_EPOCH_WAIT_MS + STAGE_REDUCE_PLAN_MS + STAGE_ENCODE_PUBLISH_MS

#: Default retry hint floor (ms) — a hint, never a hold.
MIN_RETRY_AFTER_MS = 50.0


@dataclass(frozen=True)
class AdmissionEnvelope:
    """Published admission envelope (contract required fields).

    Defaults are deliberately permissive-but-real: they bound the obvious
    abuse cases without rejecting classroom-scale collaboration bursts.
    Production values must come from the commissioning load matrix
    (service_level_contract.md), not from this file.

    Joint admissibility caveat (review F3): these are INDEPENDENT maxima —
    a batch at several maxima at once (e.g. 128 operations including one
    100k-cell paint) can book past ``fast_compute_budget_ms_per_epoch``
    and be rejected although every single bound admits it. Intentional
    conservative behavior; retune only from measured kernel costs.
    """

    #: Operations in one batch / one epoch (route model caps batches at 128).
    max_operations_per_epoch: int = 128
    #: Per-workspace admitted operations per sliding second. Must be >= the
    #: per-epoch bound so ONE maximal batch always fits (the R1 wire
    #: contract accepts 128-operation batches); bursts beyond one maximal
    #: batch per second are the thing this bounds.
    max_accepted_operations_per_second: float = 128.0
    #: Raster cells one epoch's paint operations may touch.
    max_changed_cells_per_epoch: int = 100_000
    #: Geometry objects (building + vegetation ops) one epoch may touch.
    max_geometry_objects_touched_per_epoch: int = 1_000
    #: Per-operation payload size bound (bytes, JSON-encoded).
    max_payload_bytes_per_operation: int = 65_536
    #: Distinct workspaces admitted inside the sliding window.
    max_active_workspaces: int = 256
    #: Fast compute bookable for one epoch (the compute stage budget).
    fast_compute_budget_ms_per_epoch: float = STAGE_COMPUTE_MS


#: Families whose operations touch scientific state (vs view-only switches).
SCIENTIFIC_FAMILIES = frozenset(
    {
        "building_geometry",
        "vegetation_geometry",
        "landcover_surface",
        "meteorological_forcing",
        "model_receptor_parameters",
        "selected_date_time",
    }
)

#: Booked fast-compute cost per operation family (ms). Cheap families are
#: view/scalar switches; geometry books the most because its degradation
#: ladder is the deepest. Missing families book the worst case.
FAMILY_FAST_COST_MS: dict[str, float] = {
    "output_view": 5.0,
    "selected_date_time": 40.0,
    "meteorological_forcing": 40.0,
    "model_receptor_parameters": 40.0,
    "vegetation_geometry": 180.0,
    "building_geometry": 220.0,
    "landcover_surface": 150.0,
}

#: Worst-case booking for a family not in the table (frozen vocabulary
#: today; a new family without a measured cost books pessimistically).
UNKNOWN_FAMILY_COST_MS = 250.0

#: Landcover cell booking: per painted cell on top of the family base.
LANDCOVER_CELL_COST_MS = 0.004

#: Per-operation margin on top of the heaviest family base (an epoch is
#: ONE coalesced fast kernel invocation; the reducer already merged the
#: batch, so each extra operation books only its share of the plan).
PER_OPERATION_MARGIN_MS = 2.0


def predict_operation_cost_ms(item: Mapping[str, Any]) -> float:
    """Book the fast-compute cost of one operation (ms).

    Payload-aware for the family where payload size dominates the
    dependency graph: land-cover paint books its window's cell count,
    everything else books the family constant.
    """
    family = str(item.get("source_family", ""))
    base = FAMILY_FAST_COST_MS.get(family, UNKNOWN_FAMILY_COST_MS)
    if family != "landcover_surface":
        return base
    payload = item.get("payload")
    window = payload.get("window") if isinstance(payload, Mapping) else None
    if not isinstance(window, Mapping):
        return base
    try:
        rows = abs(int(window["row_end"]) - int(window["row_start"])) + 1
        cols = abs(int(window["col_end"]) - int(window["col_start"])) + 1
    except (KeyError, TypeError, ValueError):
        return base
    return max(base, rows * cols * LANDCOVER_CELL_COST_MS)


def predict_epoch_compute_ms(items: list[Mapping[str, Any]]) -> float:
    """Book the fast-compute cost of one EPOCH's operations (ms).

    An epoch is one coalesced kernel invocation (epoch_scheduler.md:
    "Coalesce operations into epoch-final source deltas"), so the booking
    is the HEAVIEST family base in the batch (the deepest degradation
    ladder the lane may have to climb) plus a small per-operation margin —
    never the sum of per-operation costs, which would reject exactly the
    128-operation batches the R1 wire contract accepts.
    """
    if not items:
        return 0.0
    heaviest = max(predict_operation_cost_ms(item) for item in items)
    return heaviest + PER_OPERATION_MARGIN_MS * len(items)


class AdmissionRejected(ApiError):
    """Typed rejection BEFORE durable acceptance, with retry metadata.

    ``status`` 429 = the workspace's own burst is outside the envelope
    (the client can fix it: back off, reduce brush size/rate, coarsen
    preview). ``status`` 503 = server-wide capacity (retry later).
    ``details`` always carries ``retry_after_ms``.
    """


class CapacityReporter(Protocol):
    """The fast lane's view of what is already reserved (admission seam)."""

    def reserved_ms(self, workspace_id: str) -> float: ...

    def saturated(self) -> bool: ...


class _NullCapacity:
    """No fast lane wired (unit tests / start_fast_lane=False apps)."""

    def reserved_ms(self, workspace_id: str) -> float:
        return 0.0

    def saturated(self) -> bool:
        return False


def _parse_stamp(stamp: str) -> datetime:
    """Parse the store/route ``_now_utc`` wall stamp (ms precision)."""
    return datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%S.%fZ").replace(
        tzinfo=timezone.utc
    )


def _default_utc_now() -> str:
    """Real wall clock in the routes' stamp format.

    Deliberately NOT the store's clock: ``received_at`` is stamped by the
    route (real wall), and the elapsed-ingest math must compare stamps
    from the same source — test monkeypatching of ``store._now_utc``
    (fake epoch clocks) must not skew the admission clock.
    """
    from datetime import datetime, timezone

    return (
        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    )


class FastAdmission:
    """Reject-before-accept gate for realtime operation batches.

    Thread-safe (the ingest route may run on any worker thread under the
    test client; uvicorn serves this app single-threaded per process but
    nothing here may depend on that).
    """

    def __init__(
        self,
        *,
        envelope: AdmissionEnvelope | None = None,
        capacity: CapacityReporter | None = None,
        utc_now: Any = None,
    ) -> None:
        self.envelope = envelope if envelope is not None else AdmissionEnvelope()
        self.capacity = capacity if capacity is not None else _NullCapacity()
        # Injectable wall clock (store stamp format) so envelope math is
        # deterministic under FakeClock. Defaults to the store's clock so
        # stamps stay comparable.
        self._utc_now = utc_now if utc_now is not None else _default_utc_now
        self._lock = threading.Lock()
        # Per-workspace sliding second of admitted operation counts.
        self._admitted_events: dict[str, deque[float]] = {}
        # Sliding second of (workspace, admitted_at_s) for the active bound.
        self._workspace_events: deque[tuple[str, float]] = deque()
        #: Rejection counter (ops-plane diagnostics; not a telemetry name).
        self.rejections_total = 0

    # -- the gate ------------------------------------------------------------

    def check(
        self,
        workspace_id: str,
        items: list[Mapping[str, Any]],
        *,
        received_at: str,
    ) -> None:
        """Admit the batch or raise :class:`AdmissionRejected`.

        Called by the ingest route AFTER transport validation, BEFORE the
        durable append — raising here means nothing was accepted, which is
        the whole contract: reject before acceptance, never accept-then-drop.
        """
        now = _parse_stamp(self._utc_now())
        received = _parse_stamp(received_at)
        elapsed_ingest_ms = max(0.0, (now - received).total_seconds() * 1000.0)

        # Server-wide capacity first: when the fast lane is saturated there
        # is no per-batch math worth doing — retry later.
        if self.capacity.saturated():
            self._reject(
                status=503,
                code="fast_lane_unavailable",
                message=(
                    "the fast lane's deadline queue is saturated; retry after "
                    "the hinted interval"
                ),
                details={"retry_after_ms": self._retry_after_ms(2.0 * MIN_RETRY_AFTER_MS)},
            )

        envelope = self.envelope
        if len(items) > envelope.max_operations_per_epoch:
            self._reject(
                message=(
                    f"a {len(items)}-operation batch exceeds the admitted "
                    f"envelope of {envelope.max_operations_per_epoch} operations "
                    "per epoch; reduce the brush size or submit rate"
                ),
                details={
                    "max_operations_per_epoch": envelope.max_operations_per_epoch,
                    "retry_after_ms": self._retry_after_ms(MIN_RETRY_AFTER_MS),
                    "advice": [
                        "reduce brush size or stroke rate",
                        "coarsen the preview window",
                    ],
                },
            )

        geometry_touched = sum(
            1
            for item in items
            if str(item.get("source_family", ""))
            in ("building_geometry", "vegetation_geometry")
        )
        if geometry_touched > envelope.max_geometry_objects_touched_per_epoch:
            self._reject(
                message=(
                    f"the batch touches {geometry_touched} geometry objects, "
                    f"above the admitted {envelope.max_geometry_objects_touched_per_epoch} "
                    "per epoch"
                ),
                details={
                    "max_geometry_objects_touched_per_epoch": (
                        envelope.max_geometry_objects_touched_per_epoch
                    ),
                    "retry_after_ms": self._retry_after_ms(MIN_RETRY_AFTER_MS),
                    "advice": ["reduce the number of objects edited per second"],
                },
            )

        changed_cells = _changed_cells(items)
        if changed_cells > envelope.max_changed_cells_per_epoch:
            self._reject(
                message=(
                    f"the batch paints {changed_cells} cells, above the "
                    f"admitted {envelope.max_changed_cells_per_epoch} per epoch"
                ),
                details={
                    "max_changed_cells_per_epoch": envelope.max_changed_cells_per_epoch,
                    "retry_after_ms": self._retry_after_ms(MIN_RETRY_AFTER_MS),
                    "advice": ["reduce brush size", "coarsen the preview window"],
                },
            )

        for index, item in enumerate(items):
            payload = item.get("payload")
            size = len(json.dumps(payload, default=str)) if payload else 0
            if size > envelope.max_payload_bytes_per_operation:
                self._reject(
                    message=(
                        f"operations[{index}] payload is {size} bytes, above "
                        f"the admitted {envelope.max_payload_bytes_per_operation}"
                    ),
                    details={
                        "max_payload_bytes_per_operation": (
                            envelope.max_payload_bytes_per_operation
                        ),
                        "retry_after_ms": self._retry_after_ms(MIN_RETRY_AFTER_MS),
                        "advice": ["split the edit into smaller operations"],
                    },
                )

        # Deadline envelope: elapsed ingest + fixed stages + reserved fast
        # work ahead of this batch + the epoch-coalesced predicted cost.
        reserved = float(self.capacity.reserved_ms(workspace_id))
        predicted_compute = predict_epoch_compute_ms(items)
        deadline_remaining_ms = max(
            0.0, DEFAULT_FAST_DEADLINE_BUDGET_MS - elapsed_ingest_ms
        )
        compute_available_ms = max(
            0.0,
            min(
                envelope.fast_compute_budget_ms_per_epoch,
                deadline_remaining_ms - FIXED_STAGE_MS,
            )
            - reserved,
        )
        predicted_fast_ms = (
            elapsed_ingest_ms
            + FIXED_STAGE_MS
            + reserved
            + predicted_compute
        )
        if predicted_compute > compute_available_ms:
            self._reject(
                message=(
                    f"the batch books {predicted_compute:.0f} ms of fast compute "
                    f"but only {compute_available_ms:.0f} ms of the 1,000 ms "
                    "deadline envelope remain unreserved; reduce the operation "
                    "rate or coarsen the preview"
                ),
                details={
                    "retry_after_ms": self._retry_after_ms(
                        predicted_compute - compute_available_ms
                    ),
                    "deadline_budget_ms": DEFAULT_FAST_DEADLINE_BUDGET_MS,
                    "elapsed_ingest_ms": elapsed_ingest_ms,
                    "reserved_ahead_ms": reserved,
                    "predicted_compute_ms": predicted_compute,
                    "compute_available_ms": compute_available_ms,
                    "predicted_fast_ms": predicted_fast_ms,
                    "advice": [
                        "reduce brush size or stroke rate",
                        "submit fewer operations per second",
                    ],
                },
            )

        # Per-workspace rate window (sliding second, wall-clock seconds).
        now_s = now.timestamp()
        with self._lock:
            events = self._admitted_events.setdefault(workspace_id, deque())
            while events and now_s - events[0] >= 1.0:
                events.popleft()
            if (
                len(events) + len(items)
                > envelope.max_accepted_operations_per_second
            ):
                wait_s = 1.0 - (now_s - events[0]) if events else 0.0
                retry_ms = self._retry_after_ms(wait_s * 1000.0)
                self._reject_locked(
                    message=(
                        f"admitting {len(items)} more operations would exceed the "
                        f"workspace's admitted rate of "
                        f"{envelope.max_accepted_operations_per_second:g} "
                        "operations per second; retry after the hinted interval"
                    ),
                    details={
                        "max_accepted_operations_per_second": (
                            envelope.max_accepted_operations_per_second
                        ),
                        "retry_after_ms": retry_ms,
                        "advice": ["reduce stroke rate", "coalesce edits client-side"],
                    },
                )
            events.extend([now_s] * len(items))
            while self._workspace_events and now_s - self._workspace_events[0][1] >= 1.0:
                self._workspace_events.popleft()
            active = {ws for ws, _ in self._workspace_events}
            if workspace_id not in active:
                if len(active) >= envelope.max_active_workspaces:
                    self._reject_locked(
                        status=503,
                        code="fast_lane_unavailable",
                        message=(
                            f"{len(active)} workspaces are active inside the "
                            "admission window, above the admitted "
                            f"{envelope.max_active_workspaces}; retry later"
                        ),
                        details={
                            "max_active_workspaces": envelope.max_active_workspaces,
                            "retry_after_ms": self._retry_after_ms(
                                MIN_RETRY_AFTER_MS
                            ),
                        },
                    )
                self._workspace_events.append((workspace_id, now_s))
            elif self._workspace_events:
                self._workspace_events.append((workspace_id, now_s))

    # -- internals --------------------------------------------------------

    @staticmethod
    def _retry_after_ms(raw_ms: float) -> int:
        return max(int(MIN_RETRY_AFTER_MS), int(round(raw_ms)))

    def _reject(
        self,
        message: str,
        *,
        details: dict[str, Any],
        status: int = 429,
        code: str = "fast_lane_overloaded",
    ) -> None:
        with self._lock:
            self._reject_locked(message, details=details, status=status, code=code)

    def _reject_locked(
        self,
        message: str,
        *,
        details: dict[str, Any],
        status: int = 429,
        code: str = "fast_lane_overloaded",
    ) -> None:
        self.rejections_total += 1
        raise AdmissionRejected(
            code,
            message,
            status=status,
            details=details,
        )


def _changed_cells(items: list[Mapping[str, Any]]) -> int:
    total = 0
    for item in items:
        if str(item.get("source_family", "")) != "landcover_surface":
            continue
        payload = item.get("payload")
        window = payload.get("window") if isinstance(payload, Mapping) else None
        if not isinstance(window, Mapping):
            continue
        try:
            rows = abs(int(window["row_end"]) - int(window["row_start"])) + 1
            cols = abs(int(window["col_end"]) - int(window["col_start"])) + 1
        except (KeyError, TypeError, ValueError):
            continue
        total += rows * cols
    return total
