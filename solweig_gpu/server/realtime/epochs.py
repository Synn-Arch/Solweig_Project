# SPDX-License-Identifier: GPL-3.0-only
"""The 100 ms epoch scheduler (R1 epochs wave).

Owns the epoch close pipeline described in
``docs/incremental_design_tool/realtime_collaboration/epoch_scheduler.md``:

    open --(window elapses)--> closed --(reduce+commit)--> reducing
        ... R2 advances to fast_planned/fast_published/exact_targeted

Per due epoch, in one pass (:meth:`EpochScheduler.close_epoch`):

1. mark ``open -> closed`` (``closed_at`` stamped; no-op if already closed);
2. fetch the epoch's operations (BEFORE any reduction; asserted sorted);
3. baseline = latest persisted canonical state, else ``initial_state``;
4. fold via the pure :class:`~solweig_gpu.server.realtime.types.EpochReducer`
   (server_sequence order — the reducer sorts and enforces uniqueness);
5. DURABLY, in ONE store transaction
   (:meth:`solweig_gpu.server.store.Store.commit_epoch_reduction`): persist
   the canonical state keyed by the new revision, assign the epoch's
   ``workspace_revision``, and bump ``scenarios.scene_version`` by exactly
   one. THAT bump is the canonical revision assignment — accepted
   operations never bump it themselves;
6. broadcast the ``canonical_revision`` SSE event and record telemetry.

Invariants this module is responsible for:

* ZERO LOSS — every durably accepted operation lands in exactly one epoch's
  broadcast payload (fenced before publish with a RuntimeError tripwire
  that survives ``python -O``; append/close serialize on the store lock, so
  an operation either joined the closing epoch or opened the next one —
  there is no in-between).
* IDEMPOTENT RE-DRIVE — an epoch that already carries a revision is
  re-broadcast only: no re-fold (which would read a newer baseline), no
  second bump, original state bytes untouched.
* STALE-BASELINE GUARD — the commit refuses (typed
  ``EpochBaselineStale``) when another driver landed a reduction after
  this fold read its baseline; the close re-folds on the fresh baseline
  (bounded retries), so the latest canonical state never silently loses a
  concurrent epoch's effects (r1-epochs-review M2).
* STRANDED-EPOCH RE-DRIVE — every tick re-drives closed-but-unassigned
  epochs OLDEST FIRST, before later epochs close: a transient failure
  between the closed-mark and the commit self-heals without a restart and
  epoch-id order keeps matching revision order (r1-epochs-review M1).
* RESTART RECOVERY — at start, epochs stuck ``closed``/``reducing`` are
  re-driven; terminal epochs are never touched.
* EMPTY EPOCHS — an epoch with zero operations closes without a revision
  (defensive: r1-ops only opens epochs for accepted batches).

Telemetry (``telemetry.py`` standard names) never blocks and never breaks
the pipeline. The daemon thread follows the ``jobs.py`` ticker pattern
(``Event.wait`` cadence + generation counter for prompt, idempotent stop).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable

from solweig_gpu.server.realtime import operations as rt
from solweig_gpu.server.realtime.reducer import DeterministicEpochReducer
from solweig_gpu.server.realtime.telemetry import (
    StandardMetrics,
    TelemetryRegistry,
    register_standard_metrics,
)
from solweig_gpu.server.realtime.types import (
    CanonicalState,
    ConflictRecord,
    EpochReducer,
    Operation,
)
from solweig_gpu.server.store import (
    EpochBaselineStale,
    OperationRecord,
    Store,
    _now_utc as _default_utc_now,
)

logger = logging.getLogger(__name__)

#: realtime_contract.yaml: micro_epoch_ms 100.
DEFAULT_TICK_MS = 100.0

#: service_level_contract.md: "Epoch duration | 100 ms nominal;
#: configurable 50-200 ms" — the admitted domain for any tick override
#: (create_app's ``epoch_tick_ms`` validates against exactly this).
EPOCH_TICK_MIN_MS = 50.0
EPOCH_TICK_MAX_MS = 200.0

#: SSE heartbeat cadence (stall detection + lagged-recovery notice).
DEFAULT_HEARTBEAT_INTERVAL_S = 5.0

EPOCH_SCHEDULER_THREAD_NAME = "solweig-epoch-scheduler"

#: Store wall-clock stamp format (``realtime_contract.yaml`` timestamps).
_STAMP_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"


def _parse_stamp(stamp: str) -> datetime:
    """Parse a store-format wall-clock stamp to an aware datetime."""
    return datetime.strptime(stamp, _STAMP_FORMAT).replace(tzinfo=timezone.utc)


def operation_from_record(record: OperationRecord) -> Operation:
    """Lift a durable :class:`OperationRecord` into the reducer's input type."""
    return Operation(
        workspace_id=record.workspace_id,
        operation_id=record.operation_id,
        actor_id=record.actor_id,
        client_sequence=record.client_sequence,
        base_revision=record.base_revision,
        source_family=record.source_family,  # type: ignore[arg-type]
        entity_id=record.entity_id,
        verb=record.verb,  # type: ignore[arg-type]
        payload=record.payload,
        received_at=record.received_at,
        accepted_at=record.accepted_at,
        server_sequence=record.server_sequence,
        epoch_id=record.epoch_id,
    )


def conflict_body(record: ConflictRecord) -> dict[str, Any]:
    """Render a :class:`ConflictRecord` for the broadcast event."""
    return {
        "kind": record.kind,
        "source_family": record.source_family,
        "entity_id": record.entity_id,
        "winner_operation_id": record.winner_operation_id,
        "loser_operation_ids": list(record.loser_operation_ids),
        "detail": dict(record.detail),
    }


class EpochScheduler:
    """Closes due epochs, assigns canonical revisions, broadcasts SSE events."""

    def __init__(
        self,
        store: Store,
        *,
        hub: Any = None,
        reducer: EpochReducer | None = None,
        telemetry: TelemetryRegistry | None = None,
        tick_ms: float = DEFAULT_TICK_MS,
        heartbeat_interval_s: float = DEFAULT_HEARTBEAT_INTERVAL_S,
        utc_now: Callable[[], str] | None = None,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        after_epoch_close: Callable[[str, int, dict[str, Any]], None] | None = None,
        window_hint_s: Callable[[], float | None] | None = None,
    ) -> None:
        self._store = store
        self._hub = hub
        self._reducer = reducer if reducer is not None else DeterministicEpochReducer()
        self._telemetry = telemetry
        self._metrics: StandardMetrics | None = (
            register_standard_metrics(telemetry) if telemetry is not None else None
        )
        self.tick_ms = float(tick_ms)
        self.heartbeat_interval_s = float(heartbeat_interval_s)
        # T15 adaptive-epoch seam (flag-gated at the WIRING site, not here):
        # an optional live hint for the CURRENT epoch window. While
        # selected-time demand is pending the app wires a hint at
        # EPOCH_TICK_MIN_MS so freshly accepted operations close (and
        # publish) at the latency floor; the hint is CLAMPED into the
        # service contract's [EPOCH_TICK_MIN_MS, EPOCH_TICK_MAX_MS] domain
        # so no caller can widen or shrink the cadence beyond the admitted
        # envelope. ``None`` (no hint, or hint returning None) keeps the
        # configured ``tick_ms`` — the flag-off cadence is untouched.
        self._window_hint_s = window_hint_s
        # Wall clock in the store's stamp format (epoch windowing). Defaults
        # to the store's own clock so stamps stay comparable.
        self._utc_now = utc_now if utc_now is not None else _default_utc_now
        self._monotonic_ns = monotonic_ns
        self._after_close = after_epoch_close
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._generation = 0  # invalidated by stop(); a stale loop exits
        self._last_heartbeat_ns = 0

    # -- after-close tail hook (r2b) -------------------------------------------

    def attach_after_close(
        self, callback: Callable[[str, int, dict[str, Any]], None]
    ) -> None:
        """Register the after-close tail hook (the fast lane's wake entry).

        The hook fires at the TAIL of :meth:`close_epoch`, AFTER the
        canonical revision is durable and broadcast. It must be
        enqueue-and-return by contract; a raising hook is logged and
        swallowed so the close pipeline is never blocked by fast work.
        """
        self._after_close = callback

    def _notify_closed(
        self, workspace_id: str, epoch: Any, event: dict[str, Any]
    ) -> None:
        if self._after_close is None:
            return
        try:
            self._after_close(workspace_id, int(epoch.epoch_id), event)
        except Exception:  # noqa: BLE001 - fast work must never block a close
            logger.exception(
                "after-close hook failed for %s/%s (revision stays durable)",
                workspace_id,
                epoch.epoch_id,
            )

    # -- cadence --------------------------------------------------------------

    def _window_s(self) -> float:
        """Current epoch window in seconds (configured cadence, or the
        live hint clamped into the contract's tick domain)."""
        if self._window_hint_s is not None:
            try:
                hint = self._window_hint_s()
            except Exception:  # noqa: BLE001 - a broken hint must not break cadence
                logger.exception("epoch window hint raised; using configured tick")
                hint = None
            if hint is not None:
                return min(
                    max(float(hint), EPOCH_TICK_MIN_MS / 1000.0),
                    EPOCH_TICK_MAX_MS / 1000.0,
                )
        return self.tick_ms / 1000.0

    def _epoch_due(self, epoch: Any) -> bool:
        """Whether the epoch's window has fully elapsed at the current time."""
        try:
            opened = _parse_stamp(epoch.opened_at)
        except ValueError:  # pragma: no cover - corrupt stamp: close it now
            return True
        now = _parse_stamp(self._utc_now())
        age = (now - opened).total_seconds()
        return age >= self._window_s() - 1e-9

    # -- close pipeline ---------------------------------------------------------

    def close_epoch(self, workspace_id: str, epoch_id: int) -> dict[str, Any] | None:
        """Run the full close pipeline for one epoch (idempotent).

        Returns the broadcast event dict, or ``None`` when the epoch does
        not exist, is terminal (R2's lanes own it), or closed empty.
        """
        epoch = self._store.get_epoch(workspace_id, int(epoch_id))
        if epoch is None:
            return None
        if epoch.status in rt.TERMINAL_EPOCH_STATUSES:
            return None

        # open -> closed (idempotent mark; closed_at stamps on this edge).
        if epoch.status == "open":
            self._store.mark_epoch_status(workspace_id, int(epoch_id), "closed")
            epoch = self._store.get_epoch(workspace_id, int(epoch_id))
            if epoch is None:  # pragma: no cover - raced delete
                return None

        # Already assigned on a previous drive: re-broadcast only. No
        # re-fold (the persisted bytes are the truth), no second bump.
        if epoch.workspace_revision is not None:
            event = self._publish(workspace_id, epoch)
            self._notify_closed(workspace_id, epoch, event)
            return event

        operations = self._store.operations_for_epoch(workspace_id, int(epoch_id))
        sequences = [record.server_sequence for record in operations]
        # Ordering fence (STRICTLY increasing — server_sequence is unique and
        # contiguous per workspace, so a repeat is as wrong as a swap): raise
        # (not assert) so the tripwire survives `python -O`; an unordered
        # epoch is a bug, never a runtime state.
        if any(
            sequences[i + 1] <= sequences[i] for i in range(len(sequences) - 1)
        ):
            raise RuntimeError(
                f"epoch {workspace_id}/{epoch_id} operations out of order: "
                f"{sequences}"
            )

        if not operations:
            # Empty epochs never carry a revision (r1-ops never creates one;
            # this is the defensive close). Stay 'closed'; still count it.
            self._observe("epochs_closed_total")
            return None

        # solweig_rt_epoch_wait_ms: t_epoch_close - t_accept of the OLDEST
        # operation in the epoch (the worst-case wait we owed anyone).
        oldest_accepted = _parse_stamp(operations[0].accepted_at)
        close_wait_ms = (
            _parse_stamp(self._utc_now()) - oldest_accepted
        ).total_seconds() * 1000.0

        # Fold + commit with the stale-baseline guard (r1-epochs-review M2):
        # a concurrent driver may land another epoch's reduction after this
        # fold read its baseline; the commit refuses (EpochBaselineStale) and
        # we re-fold on the fresh baseline instead of silently dropping the
        # other epoch's effects from the latest canonical state.
        reduction = None
        for _attempt in range(3):
            baseline: CanonicalState = (
                self._store.latest_canonical_state(workspace_id)
                or self._reducer.initial_state(workspace_id)
            )
            expected_baseline = int(baseline.workspace_revision or 0)
            reduce_started_ns = self._monotonic_ns()
            reduction = self._reducer.reduce_epoch(
                baseline,
                [operation_from_record(record) for record in operations],
            )
            self._observe_ms("reduce_ms", reduce_started_ns)
            try:
                committed = self._store.commit_epoch_reduction(
                    workspace_id,
                    int(epoch_id),
                    families=reduction.state.families,
                    baseline_revision=expected_baseline,
                )
                break
            except EpochBaselineStale as stale:
                logger.warning(
                    "epoch %s/%s folded a stale baseline (attempt %d): %s",
                    workspace_id,
                    epoch_id,
                    _attempt + 1,
                    stale,
                )
                continue
        else:
            raise RuntimeError(
                f"epoch {workspace_id}/{epoch_id} could not commit: the "
                "canonical baseline kept advancing across 3 attempts"
            )
        if committed is None:  # pragma: no cover - epoch vanished mid-close
            logger.error("epoch %s/%s vanished during close", workspace_id, epoch_id)
            return None
        self._observe("epochs_closed_total")
        self._observe_value("epoch_wait_ms", close_wait_ms)

        epoch = self._store.get_epoch(workspace_id, int(epoch_id))
        if epoch is None or epoch.workspace_revision is None:
            raise RuntimeError(
                f"epoch {workspace_id}/{epoch_id} has no revision after a "
                "first-time commit — commit_epoch_reduction broke its contract"
            )
        event = self._publish(
            workspace_id,
            epoch,
            operations=operations,
            conflicts=[conflict_body(record) for record in reduction.reduced.conflicts],
        )
        self._notify_closed(workspace_id, epoch, event)
        return event

    def _publish(
        self,
        workspace_id: str,
        epoch: Any,
        *,
        operations: list[OperationRecord] | None = None,
        conflicts: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Build + broadcast the canonical_revision event (idempotent).

        A broadcast failure is logged and swallowed: the revision is already
        durable, and late joiners recover through the catch-up GET.
        """
        if operations is None:
            operations = self._store.operations_for_epoch(
                workspace_id, int(epoch.epoch_id)
            )
        scenario = self._store.get_scenario(workspace_id)
        operation_bodies = [rt.operation_body(record) for record in operations]

        # ZERO-LOSS fence: everything durably assigned to this epoch must be
        # in the payload about to be published. A miss here is a bug, not a
        # runtime condition — raise (not assert) so the tripwire survives
        # `python -O` and never silently drops an accepted operation.
        expected = {record.operation_id for record in operations}
        rendered = {body["operation_id"] for body in operation_bodies}
        if rendered != expected:
            raise RuntimeError(
                f"zero-loss fence violated for {workspace_id}/{epoch.epoch_id}: "
                f"missing {expected - rendered}, extra {rendered - expected}"
            )

        event: dict[str, Any] = {
            "workspace_id": workspace_id,
            "epoch_id": int(epoch.epoch_id),
            "workspace_revision": int(epoch.workspace_revision),
            "operations": operation_bodies,
            "fast_revision": scenario.fast_revision if scenario else 0,
            "exact_revision": scenario.exact_result_version if scenario else 0,
            "exact_base_revision": scenario.exact_base_revision if scenario else 0,
        }
        if conflicts is not None:
            event["conflicts"] = conflicts
        publish_started_ns = self._monotonic_ns()
        if self._hub is not None:
            try:
                self._hub.broadcast_canonical(workspace_id, event)
            except Exception:  # noqa: BLE001 - publish must never kill the loop
                logger.exception(
                    "canonical broadcast failed for %s/%s (revision stays durable)",
                    workspace_id,
                    epoch.epoch_id,
                )
        self._observe_ms("publish_ms", publish_started_ns)
        return event

    # -- driving --------------------------------------------------------------

    def run_once(self) -> list[tuple[str, int]]:
        """One scheduler pass: close every due open epoch, every workspace.

        Stranded epochs first (r1-epochs-review M1): a transient failure
        between the closed-mark and the commit leaves an epoch ``closed``
        with no revision, invisible to the open-epoch scan — its accepted
        operations would sit outside every canonical revision until a
        restart. Every tick re-drives such epochs, OLDEST FIRST, so a
        stranded epoch always takes a LOWER revision than any epoch closed
        later in the same tick (no epoch-id/revision inversion). A
        persistently failing re-drive is logged loudly and retried next
        tick; it never blocks the live cadence.

        Returns the ``(workspace_id, epoch_id)`` pairs whose close pipeline
        ran (including defensive empty closes).
        """
        driven: list[tuple[str, int]] = []
        for epoch in self._store.recoverable_epochs():
            if epoch.status == "open" or epoch.workspace_revision is not None:
                continue  # live cadence / already assigned — not stranded
            try:
                self.close_epoch(epoch.workspace_id, epoch.epoch_id)
            except Exception:  # noqa: BLE001 - one bad epoch never stops the rest
                logger.exception(
                    "stranded epoch re-drive failed for %s/%s",
                    epoch.workspace_id,
                    epoch.epoch_id,
                )
                continue
            driven.append((epoch.workspace_id, int(epoch.epoch_id)))
        for workspace_id in self._store.open_epoch_workspaces():
            for epoch in self._store.open_epochs(workspace_id):
                if not self._epoch_due(epoch):
                    continue
                try:
                    self.close_epoch(workspace_id, epoch.epoch_id)
                except Exception:  # noqa: BLE001 - one bad epoch never stops the rest
                    logger.exception(
                        "epoch close failed for %s/%s", workspace_id, epoch.epoch_id
                    )
                    continue
                driven.append((workspace_id, int(epoch.epoch_id)))
        return driven

    def recover(self) -> list[tuple[str, int]]:
        """Re-drive epochs stuck mid close-pipeline (boot after a crash).

        ``open`` epochs are the live cadence's business (their window has
        simply not elapsed yet — or elapsed while we were down, in which
        case the first ``run_once`` closes them). Everything non-terminal
        and non-open gets one immediate drive. Returns the pairs that
        produced a broadcast event.
        """
        driven: list[tuple[str, int]] = []
        for epoch in self._store.recoverable_epochs():
            if epoch.status == "open":
                continue
            try:
                result = self.close_epoch(epoch.workspace_id, epoch.epoch_id)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "epoch recovery failed for %s/%s",
                    epoch.workspace_id,
                    epoch.epoch_id,
                )
                continue
            if result is not None:
                driven.append((epoch.workspace_id, int(epoch.epoch_id)))
        return driven

    # -- telemetry (never blocks, never raises) --------------------------------

    def _observe(self, name: str) -> None:
        if self._metrics is None:
            return
        try:
            getattr(self._metrics, name).inc()
        except Exception:  # pragma: no cover - telemetry must never break flow
            logger.exception("telemetry observe failed for %s", name)

    def _observe_value(self, name: str, value: float) -> None:
        if self._metrics is None:
            return
        try:
            getattr(self._metrics, name).observe(value)
        except Exception:  # pragma: no cover
            logger.exception("telemetry observe failed for %s", name)

    def _observe_ms(self, name: str, started_ns: int) -> None:
        self._observe_value(
            name, (self._monotonic_ns() - started_ns) / 1_000_000.0
        )

    # -- daemon thread ---------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Boot recovery + the daemon tick loop (idempotent)."""
        if self.running:
            return
        self._stop_event.clear()
        self._generation += 1
        generation = self._generation
        try:
            recovered = self.recover()
            if recovered:
                logger.info("epoch recovery re-drove %d epochs", len(recovered))
        except Exception:  # noqa: BLE001 - never block startup on recovery
            logger.exception("epoch recovery at start failed")
        self._thread = threading.Thread(
            target=self._run_loop,
            name=EPOCH_SCHEDULER_THREAD_NAME,
            daemon=True,
            kwargs={"generation": generation},
        )
        self._thread.start()

    def _run_loop(self, generation: int) -> None:
        interval = self._window_s()
        self._last_heartbeat_ns = self._monotonic_ns()
        interval_ns = int(interval * 1_000_000_000.0)
        tick_started_ns = self._monotonic_ns()
        while not self._stop_event.wait(interval):
            if generation != self._generation:
                return  # a stop() + start() replaced this loop
            woken_ns = self._monotonic_ns()
            # R8b: per-tick wake lateness — how much later than one
            # interval the loop actually fired (GIL/OS scheduling, never
            # negative). Direct evidence for visibility-tail attribution:
            # an epoch's wait beyond its [W, 2W) floor is bounded by this.
            self._observe_value(
                "epoch_tick_lateness_ms",
                max(0.0, (woken_ns - tick_started_ns - interval_ns) / 1_000_000.0),
            )
            tick_started_ns = woken_ns
            try:
                self.run_once()
            except Exception:  # noqa: BLE001 - the ticker must survive anything
                logger.exception("epoch scheduler tick failed")
            heartbeat_budget_ns = self.heartbeat_interval_s * 1_000_000_000.0
            if self._monotonic_ns() - self._last_heartbeat_ns >= heartbeat_budget_ns:
                self._last_heartbeat_ns = self._monotonic_ns()
                if self._metrics is not None:
                    try:
                        self._metrics.host_load.set(os.getloadavg()[0])
                    except Exception:  # pragma: no cover
                        logger.exception("host load sample failed")
                if self._hub is not None:
                    try:
                        self._hub.send_heartbeats()
                    except Exception:  # pragma: no cover
                        logger.exception("heartbeat send failed")

    def stop(self, timeout: float = 1.0) -> bool:
        """Signal the loop to stop and join it. Returns whether it exited."""
        self._generation += 1
        self._stop_event.set()
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout=timeout)
        exited = not thread.is_alive()
        if exited:
            self._thread = None
        return exited

    # -- context manager -------------------------------------------------------

    def __enter__(self) -> "EpochScheduler":
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()
