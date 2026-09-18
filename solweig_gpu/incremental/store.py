# SPDX-License-Identifier: GPL-3.0-only
"""Temporal result store: what is published per (node_id, time_index).

U-C intake M2: the scene graph's per-node version counters say *that* a
node changed, not *which timesteps* were (re)published. This store closes
that gap for the executor: every published product is recorded keyed by
``(node_id, time_index)``, so lookups, coverage checks, and targeted
invalidation (``invalidate("radiation", times=(14, 15))`` after a forcing
edit at t=14) work at timestep granularity.

Keys use OUTPUT node ids (``utci``, ``tmrt``, ...) plus transported
producible layer names (``shadow``, ...) — the same vocabulary the planner
keeps in ``ImpactPlan`` records.

One key may carry SEVERAL entries (u-c1b M2): a multi-window local job
publishes one patch per disjoint write window, and the store records one
entry per ``(window, patch)`` under the same ``(node, time)`` key, so
spatial coverage is never under-reported. ``lookup`` returns the first
recorded entry (deterministic window order); ``window_entries`` /
``lookup_window`` expose the per-window truth.

Publication discipline (mission invariants):

- **Atomic**: a ``publish`` batch is validated in full before any entry is
  written; a rejected batch leaves the store untouched. Writers serialize
  on an internal lock and commit by swapping in a fresh entry mapping, so
  concurrent readers always observe the old or the new snapshot, never a
  partial batch (u-c1b L6).
- **Never stale**: an entry whose ``scene_revision`` is older than the
  recorded revision for the same key is refused
  (:class:`StaleResultError`); a same-revision republish from a different
  job is a conflict (two jobs cannot own one revision), and a
  same-revision same-job republish with DIFFERENT content (window/mode/
  patch path) is a loud error, never a silent overwrite (u-c1b L7). A
  NEWER windowed batch supersedes CELLS, not entries (R5a): every
  recorded entry keeps its non-intersecting remainder — decomposed into
  disjoint narrowed rectangles referencing the SAME patch payload — so
  neither a fully untouched window (kept verbatim, R2 u-c1-review) nor
  the untouched sliver of a partially republished window loses
  provenance.
- **Single writer** (u-c1b M3, lead decision): a shared store driven by
  many executors is UNSUPPORTED. The store binds the first writer identity
  it is attributed to and refuses every other identity with
  :class:`StoreWriterError` — a loud refusal at the door, not per-key
  revision conflicts wedging the loser forever. Unattributed stores
  (``writer=None`` on every publish, the test/legacy path) stay anonymous
  and permissive.
- **Deterministic**: iteration order is sorted by key; no wall clock, no
  randomness.

Reader contract (U-D, u-c1-review R2a follow-up)
-------------------------------------------------

A windowed read whose requested window PARTIALLY overlaps the recorded
window entries must NEVER be served from the partial data: whatever the
supersession rule leaves behind, the recorded coverage is the only
provenance a reader has, and re-reading an uncovered sliver would
silently serve stale or missing cells. (Under R2 whole-entry supersession
the slivers came from entries being replaced ENTIRELY; R5a cell-remainder
retention narrows the recorded coverage to the surviving cells instead —
the contract below stays as the detection safety net either way.)
:meth:`window_coverage` makes that contract explicit and testable: it
classifies a request as ``full``, ``partial``, or ``absent`` and reports
the uncovered slivers, so a reader falls back to full recompute on
anything but ``full`` instead of discovering the gap in its output.
:meth:`revision_at` answers the reader's revision question with the MAX
published revision of the key (retained older windows must not drag it
down).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np

from .geometry import RasterGrid, RasterWindow

__all__ = [
    "CoverageStatus",
    "StaleResultError",
    "StoreError",
    "StoreWriterError",
    "TemporalResultEntry",
    "TemporalResultKey",
    "TemporalResultStore",
    "WindowCoverage",
]

#: Store key: one product node (or transported layer) at one timestep.
TemporalResultKey = tuple[str, int]

_VALID_MODES = ("local", "full")


class StoreError(RuntimeError):
    """The store rejected a structural or consistency violation."""


class StaleResultError(StoreError):
    """A publish attempted to move a key backwards in scene revisions."""


class StoreWriterError(StoreError):
    """A publish arrived from a writer identity the store never bound.

    One store has one writer (the executor that owns its scenario's
    revision chain). A second executor must build its own store; sharing
    one would collide every ``(node, time)`` key across scenarios and
    wedge the lower-revision writer into permanent same-revision
    conflicts (u-c1b M3 — refused loudly at attribution time instead).
    """


@dataclass(frozen=True, slots=True)
class TemporalResultEntry:
    """One published (node, time) product with its provenance."""

    node_id: str
    time_index: int
    scene_revision: int
    job_id: str
    mode: str
    write_window: RasterWindow
    patch_path: Path | None = None

    @property
    def key(self) -> TemporalResultKey:
        return (self.node_id, self.time_index)


class CoverageStatus(str, Enum):
    """Outcome of one windowed read against the recorded entries (U-D).

    The reader contract (see the module docstring): a ``partial`` result
    must trigger a full recompute, never a serve-from-partial — the
    uncovered sliver has no provenance, so serving it would guess.
    """

    ABSENT = "absent"
    PARTIAL = "partial"
    FULL = "full"


@dataclass(frozen=True, slots=True)
class WindowCoverage:
    """The reader-facing answer for one ``(node, time, window)`` request.

    ``entries`` are the recorded window entries that OVERLAP the request
    (deterministic window order, the same order :meth:`window_entries`
    reports); ``missing_windows`` decompose the uncovered sliver of the
    request into disjoint rectangles (empty when the status is ``full``,
    the whole request when ``absent``). ``covered_area`` counts the
    requested cells inside a recorded entry.
    """

    node_id: str
    time_index: int
    window: RasterWindow
    status: CoverageStatus
    entries: tuple[TemporalResultEntry, ...] = ()
    missing_windows: tuple[RasterWindow, ...] = ()
    covered_area: int = 0

    @property
    def missing_area(self) -> int:
        """Requested cells with no recorded coverage."""
        return self.window.area - self.covered_area


class TemporalResultStore:
    """In-memory ``(node_id, time_index)`` index over published products.

    The store holds *references* (provenance + patch location), never
    arrays: patches on disk remain the source of truth, checksum-verified
    by :mod:`solweig_gpu.incremental.result` on load.
    """

    def __init__(self, *, grid: RasterGrid | None = None) -> None:
        self._grid = grid
        #: key -> the window entries recorded for it (one per published
        #: write window, ordered deterministically by window). Publication
        #: replaces the whole mapping (copy-on-write snapshot swap under
        #: the writer lock), so lock-free readers always observe a
        #: consistent pre- or post-batch view (u-c1b L6).
        self._entries: dict[TemporalResultKey, tuple[TemporalResultEntry, ...]] = {}
        self._lock = threading.Lock()
        #: Bound writer identity (u-c1b M3): set by the first attributed
        #: publish or an explicit :meth:`bind_writer`; ``None`` is the
        #: unattributed legacy mode every pre-bind publish stays in.
        self._writer: str | None = None

    # ------------------------------------------------------------------
    # Writer binding (u-c1b M3: one store, one writer)
    # ------------------------------------------------------------------

    @property
    def writer(self) -> str | None:
        """The bound writer identity, or ``None`` while unattributed."""
        return self._writer

    def bind_writer(self, writer_id: str) -> str:
        """Bind (or re-confirm) this store's single writer identity.

        The first call binds; an identical re-bind is idempotent; a
        DIFFERENT identity raises :class:`StoreWriterError` — the loud
        refusal that replaces per-key revision conflicts wedging a second
        executor forever. Binding before any write is the earliest a
        sharing bug can be caught.
        """
        if not isinstance(writer_id, str) or not writer_id:
            raise StoreError(
                f"writer_id must be a non-empty string, got {writer_id!r}"
            )
        with self._lock:
            if self._writer is None:
                self._writer = writer_id
                return writer_id
            if self._writer != writer_id:
                raise StoreWriterError(
                    f"store is bound to writer {self._writer!r}; writer "
                    f"{writer_id!r} may not publish into it. One store has "
                    "one writer (the executor that owns its scenario's "
                    "revision chain); a second executor must build its own "
                    "store"
                )
            return writer_id

    # ------------------------------------------------------------------
    # Publication
    # ------------------------------------------------------------------

    def publish(
        self,
        entries: Sequence[TemporalResultEntry],
        *,
        writer: str | None = None,
    ) -> tuple[TemporalResultKey, ...]:
        """Record ``entries`` atomically; return the keys now covered.

        Validation is all-or-nothing: structural failures, duplicate
        windows for one key inside the batch, stale revisions,
        same-revision conflicts, and same-revision same-job replays with
        different content abort the whole batch with :class:`StoreError`
        (or :class:`StaleResultError`) before anything is written.

        ``writer`` attributes the batch to an executor identity. Once a
        store is bound, every publish must carry that identity
        (:class:`StoreWriterError` otherwise); an unbound store stays in
        the anonymous legacy mode and binds the first attributed batch it
        accepts.
        """
        if not entries:
            raise StoreError("publish batch must be non-empty")

        with self._lock:
            if self._writer is not None and writer != self._writer:
                raise StoreWriterError(
                    f"store is bound to writer {self._writer!r}; refusing a "
                    f"publish attributed to {writer!r}. One store has one "
                    "writer (the executor that owns its scenario's revision "
                    "chain)"
                )

            failures: list[str] = []
            by_key: dict[TemporalResultKey, list[TemporalResultEntry]] = {}
            for entry in entries:
                if not isinstance(entry.node_id, str) or not entry.node_id:
                    failures.append(f"node_id must be a non-empty string: {entry.node_id!r}")
                if not isinstance(entry.time_index, int) or entry.time_index < 0:
                    failures.append(f"time_index must be a non-negative int: {entry.time_index!r}")
                if not isinstance(entry.scene_revision, int) or entry.scene_revision < 0:
                    failures.append(
                        f"scene_revision must be a non-negative int: {entry.scene_revision!r}"
                    )
                if not isinstance(entry.job_id, str) or not entry.job_id:
                    failures.append(f"job_id must be a non-empty string: {entry.job_id!r}")
                if entry.mode not in _VALID_MODES:
                    failures.append(
                        f"mode must be one of {_VALID_MODES}, got {entry.mode!r}"
                    )
                if not isinstance(entry.write_window, RasterWindow):
                    failures.append("write_window must be a RasterWindow")
                    continue
                if self._grid is not None:
                    if not (
                        0 <= entry.write_window.row_start <= entry.write_window.row_stop <= self._grid.rows
                        and 0 <= entry.write_window.col_start <= entry.write_window.col_stop <= self._grid.cols
                    ):
                        failures.append(
                            f"write_window {entry.write_window} lies outside the "
                            f"{self._grid.rows}x{self._grid.cols} grid"
                        )
                        continue
                batch_for_key = by_key.setdefault(entry.key, [])
                if any(
                    prior.write_window == entry.write_window
                    for prior in batch_for_key
                ):
                    # One window per key per batch: a multi-window job
                    # publishes DISJOINT windows, so a repeated window is
                    # structural corruption, not a second opinion (u-c1b M2).
                    failures.append(
                        f"duplicate window in publish batch for {entry.key}: "
                        f"{entry.write_window}"
                    )
                else:
                    batch_for_key.append(entry)

            # Per-key consistency: every window entry of one key describes
            # the same publication (revision and mode are job-wide).
            for key, batch_for_key in by_key.items():
                revisions = {entry.scene_revision for entry in batch_for_key}
                if len(revisions) > 1:
                    failures.append(
                        f"entries for {key} carry mixed scene revisions "
                        f"{sorted(revisions)}; one batch publishes one revision"
                    )
                modes = {entry.mode for entry in batch_for_key}
                if len(modes) > 1:
                    failures.append(
                        f"entries for {key} carry mixed modes {sorted(modes)}; "
                        "one batch publishes one mode"
                    )

            if failures:
                raise StoreError(
                    "invalid publish batch (" + str(len(failures)) + " failures): "
                    + "; ".join(failures[:10])
                )

            # Revision discipline against already-recorded entries. One key
            # may carry entries at SEVERAL revisions (R2 retention below:
            # a newer windowed batch supersedes only the windows it
            # republishes), so the comparison revision is the HIGHEST
            # recorded one and the same-revision checks run against the
            # entries AT that revision.
            for key in sorted(by_key):
                batch_for_key = by_key[key]
                recorded = self._entries.get(key)
                if recorded is None:
                    continue
                batch_revision = batch_for_key[0].scene_revision
                recorded_revision = max(
                    entry.scene_revision for entry in recorded
                )
                if batch_revision < recorded_revision:
                    raise StaleResultError(
                        f"stale publish for {key}: revision {batch_revision} "
                        f"< recorded {recorded_revision} (job "
                        f"{recorded[0].job_id!r})"
                    )
                if batch_revision > recorded_revision:
                    # A newer revision supersedes only the cells it actually
                    # republishes — the commit step below keeps the
                    # non-republished remainder of every recorded entry (R2
                    # whole-window form, narrowed to cells by R5a).
                    continue
                recorded_at_revision = [
                    entry
                    for entry in recorded
                    if entry.scene_revision == batch_revision
                ]
                batch_jobs = {entry.job_id for entry in batch_for_key}
                recorded_jobs = {
                    entry.job_id for entry in recorded_at_revision
                }
                if batch_jobs != recorded_jobs:
                    raise StoreError(
                        f"revision conflict for {key}: revision "
                        f"{batch_revision} is already published by job(s) "
                        f"{sorted(recorded_jobs)}, refusing job(s) "
                        f"{sorted(batch_jobs)}"
                    )
                # Same revision, same job(s): an idempotent replay must
                # carry IDENTICAL content — a different window/mode/
                # patch-path set is a loud error, never a silent
                # overwrite (u-c1b L7).
                if _entry_content(
                    batch_for_key
                ) != _entry_content(recorded_at_revision):
                    raise StoreError(
                        f"same-revision same-job republish for {key} at "
                        f"revision {batch_revision} carries different "
                        "content (window/mode/patch path); refusing the "
                        "overwrite — republishing a published batch must "
                        "be a byte-identical replay"
                    )
                # Keep EVERY recorded entry — retained older-revision
                # windows included (they are still the live coverage for
                # their cells).
                by_key[key] = list(recorded)

            # Commit: copy-on-write snapshot swap. Readers (which take no
            # lock) observe either the old or the new mapping in full.
            committed: dict[TemporalResultKey, tuple[TemporalResultEntry, ...]] = dict(self._entries)
            for key, batch_for_key in by_key.items():
                ordered = tuple(sorted(batch_for_key, key=_window_sort_key))
                recorded = self._entries.get(key)
                if recorded is None:
                    committed[key] = ordered
                    continue
                if ordered[0].scene_revision <= max(
                    entry.scene_revision for entry in recorded
                ):
                    # Same-revision idempotent replay: the discipline pass
                    # already replaced the batch with the recorded entries.
                    committed[key] = ordered
                    continue
                # R2 (u-c1-review), narrowed to cells by R5a: a newer
                # WINDOWED batch supersedes exactly the cells it
                # republishes. The pre-R2 whole-entry drop opened a phantom
                # coverage gap (the first windowed batch after a full-tile
                # publication unrecorded the non-overlapping remainder of
                # the tile, so a full-window read degraded to PARTIAL and
                # strict readers fell back to a full re-solve) and
                # orphaned a perfectly valid patch directory. Retention
                # keeps the union: the later batch's own windows plus,
                # for every recorded entry, its non-intersecting
                # remainder — the SAME revision, job, mode, and patch
                # reference with the write window narrowed to the
                # rectangles the batch does not cover. The patch bytes
                # already hold the retained cells' values at that
                # revision; the narrowed window only declares which cells
                # a reader may take from it (the executor's plane
                # assembly slices payloads by ``entry.write_window −
                # patch.write_window`` offsets, so a narrowed entry
                # pointing into the original larger patch translates
                # correctly — and the met fast path's G1.1 per-cell
                # latest-validity gate composes mixed-provenance reads
                # under the G1.0 superset license, so retention widens
                # what readers may serve exactly as far as that proof
                # licenses).
                new_windows = [new.write_window for new in ordered]
                retained: list[TemporalResultEntry] = []
                for entry in recorded:
                    if not any(
                        entry.write_window.intersects(window)
                        for window in new_windows
                    ):
                        # R2: a window the batch never touches still
                        # depicts the newest value for its cells — keep
                        # it verbatim, original identity included.
                        retained.append(entry)
                        continue
                    for remainder in _subtract_windows(
                        entry.write_window, new_windows
                    ):
                        retained.append(
                            entry
                            if remainder == entry.write_window
                            # A zero-area (edge-touch) intersection
                            # removes no cells: the entry survives
                            # intact rather than being re-wrapped.
                            else replace(entry, write_window=remainder)
                        )
                committed[key] = tuple(
                    sorted((*retained, *ordered), key=_window_sort_key)
                )
            self._entries = committed
            if self._writer is None and writer is not None:
                self._writer = writer
            return tuple(sorted(by_key))

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def lookup(self, node_id: str, time_index: int) -> TemporalResultEntry | None:
        """Entry for ``(node_id, time_index)`` or ``None`` when unpublished.

        The returned entry is the first recorded window entry (deterministic
        window order) — the faithful answer for single-window publications;
        use :meth:`window_entries` / :meth:`lookup_window` for the complete
        multi-window truth (u-c1b M2).
        """
        entries = self._entries.get((node_id, int(time_index)))
        return None if not entries else entries[0]

    def window_entries(
        self, node_id: str, time_index: int
    ) -> tuple[TemporalResultEntry, ...]:
        """Every recorded window entry for ``(node_id, time_index)``.

        Multi-window local jobs record one entry per disjoint write window
        (each carrying its own patch path), so spatial coverage is never
        under-reported (u-c1b M2).
        """
        return self._entries.get((node_id, int(time_index)), ())

    def lookup_window(
        self, node_id: str, time_index: int, *, row: int, col: int
    ) -> TemporalResultEntry | None:
        """The entry whose write window contains ``(row, col)``, if any.

        The per-cell-region lookup a reader needs: with disjoint window
        entries recorded this returns the patch that actually covers the
        queried cell, not just the first window of the batch.
        """
        for entry in self.window_entries(node_id, time_index):
            window = entry.write_window
            if (
                window.row_start <= row < window.row_stop
                and window.col_start <= col < window.col_stop
            ):
                return entry
        return None

    def node_entries(self, node_id: str) -> tuple[TemporalResultEntry, ...]:
        """Every recorded entry for ``node_id``, ordered by time then window.

        The entry mapping is read ONCE (R3, u-c1-review): a concurrent
        publish or invalidate swaps ``self._entries`` wholesale, so a loop
        that re-read it per key could sort against one snapshot and index
        another — a KeyError TOCTOU between the two reads. The local
        reference keeps this reader on one immutable snapshot throughout.
        """
        entries = self._entries
        return tuple(
            entry
            for key in sorted(entries)
            if key[0] == node_id
            for entry in entries[key]
        )

    def available_times(self, node_id: str) -> tuple[int, ...]:
        """Timesteps with a recorded entry for ``node_id`` (sorted)."""
        return tuple(key[1] for key in sorted(self._entries) if key[0] == node_id)

    def revision_at(self, node_id: str, time_index: int) -> int | None:
        """MAX published scene revision of the key, ``None`` when unpublished.

        Reader-facing semantics (U-D, u-c1-review R2a follow-up): a key
        may carry window entries at SEVERAL revisions — retention keeps
        non-intersecting older windows alive after a newer windowed batch —
        and a reader asking "what revision is this product at?" needs the
        NEWEST published revision; the first recorded entry's revision can
        trail it. (Pre-U-D this returned the first entry's revision, which
        was indistinguishable on single-window publications and is what
        every then-existing caller asserted.) Callers that need one
        region's provenance should read the entry itself
        (:meth:`lookup_window`); callers that need a node-wide watermark
        have :meth:`max_revision`.
        """
        entries = self.window_entries(node_id, time_index)
        if not entries:
            return None
        return max(entry.scene_revision for entry in entries)

    def window_coverage(
        self, node_id: str, time_index: int, *, window: RasterWindow
    ) -> WindowCoverage:
        """Classify one windowed read as full/partial/absent (U-D contract).

        The R2a residual (u-c1-review round 3): supersession replaces
        whole entries, so a newer batch that partially overlaps a recorded
        window leaves the non-overlapping sliver of an overlapping READ
        without provenance — a reader that served the covered cells would
        silently drop the sliver. This method makes the incomplete
        coverage DETECTABLE: anything but :attr:`CoverageStatus.FULL`
        must fall back to full recompute, never serve from the entries.

        ``entries`` lists the overlapping recorded entries (window order),
        ``missing_windows`` the uncovered slivers of ``window`` as disjoint
        rectangles, and the status is ``ABSENT`` when no recorded entry
        overlaps the request with positive area (including an unpublished
        key or an entry that only edge-touches the request).
        """
        if not isinstance(window, RasterWindow):
            raise StoreError(
                f"window must be a RasterWindow, got {type(window).__name__}"
            )
        if window.is_empty:
            raise StoreError("coverage window must be non-empty")
        if self._grid is not None and (
            window.row_start < 0
            or window.col_start < 0
            or window.row_stop > self._grid.rows
            or window.col_stop > self._grid.cols
        ):
            raise StoreError(
                f"coverage window {window} lies outside the "
                f"{self._grid.rows}x{self._grid.cols} grid"
            )
        # Snapshot once (R3 discipline): a concurrent publish or invalidate
        # swaps the entry mapping wholesale; re-reading it between the
        # overlap pass and the mask fill could mix two snapshots.
        entries = self._entries
        recorded = entries.get((node_id, int(time_index)), ())
        overlapping = tuple(
            entry
            for entry in recorded
            if entry.write_window.intersects(window)
        )
        covered = _covered_mask(
            window, (entry.write_window for entry in overlapping)
        )
        covered_area = int(covered.sum())
        if covered_area == 0:
            return WindowCoverage(
                node_id=node_id,
                time_index=int(time_index),
                window=window,
                status=CoverageStatus.ABSENT,
                entries=(),
                missing_windows=(window,),
                covered_area=0,
            )
        if covered_area == window.area:
            return WindowCoverage(
                node_id=node_id,
                time_index=int(time_index),
                window=window,
                status=CoverageStatus.FULL,
                entries=overlapping,
                missing_windows=(),
                covered_area=covered_area,
            )
        return WindowCoverage(
            node_id=node_id,
            time_index=int(time_index),
            window=window,
            status=CoverageStatus.PARTIAL,
            entries=overlapping,
            missing_windows=_missing_rectangles(covered, window),
            covered_area=covered_area,
        )

    def coverage_gaps(self, node_id: str, times: Sequence[int]) -> tuple[int, ...]:
        """Requested times lacking a recorded entry (sorted)."""
        return tuple(
            time
            for time in sorted(set(times))
            if (node_id, int(time)) not in self._entries
        )

    def max_revision(self, node_id: str | None = None) -> int:
        """Highest recorded scene revision, optionally for one node only."""
        revisions = [
            entry.scene_revision
            for key, entries in self._entries.items()
            if node_id is None or key[0] == node_id
            for entry in entries
        ]
        return max(revisions, default=0)

    # ------------------------------------------------------------------
    # Invalidation
    # ------------------------------------------------------------------

    def invalidate(
        self, node_id: str, *, times: Sequence[int] | None = None
    ) -> tuple[TemporalResultKey, ...]:
        """Drop entries for ``node_id``; return the removed keys.

        ``times=None`` (default) drops every timestep for the node — the
        temporal analogue of "the node is dirty". Explicit times drop only
        those steps. Removing absent keys is a no-op.
        """
        node_times: set[int] | None = None
        if times is not None:
            node_times = {int(time) for time in times}
        with self._lock:
            removed = [
                key
                for key in sorted(self._entries)
                if key[0] == node_id and (node_times is None or key[1] in node_times)
            ]
            if removed:
                committed = dict(self._entries)
                for key in removed:
                    committed.pop(key, None)
                self._entries = committed
        return tuple(removed)

    # ------------------------------------------------------------------
    # Mapping protocol (read-only views)
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[TemporalResultKey]:
        return iter(sorted(self._entries))

    def __contains__(self, key: object) -> bool:
        return isinstance(key, tuple) and key in self._entries

    def as_mapping(self) -> Mapping[TemporalResultKey, TemporalResultEntry]:
        """Read-only snapshot: key -> its first (representative) entry."""
        return {key: entries[0] for key, entries in self._entries.items()}


def _window_sort_key(entry: TemporalResultEntry) -> tuple[int, int, int, int]:
    window = entry.write_window
    return (window.row_start, window.col_start, window.row_stop, window.col_stop)


def _covered_mask(
    window: RasterWindow, covering: Iterable[RasterWindow]
) -> np.ndarray:
    """Boolean mask of ``window``'s cells inside any window of ``covering``.

    The single definition of window-overlap mask math, shared by
    :meth:`window_coverage` (which cells of the request are recorded) and
    :func:`_subtract_windows` (which cells of a superseded entry survive).
    Windows that do not overlap ``window`` — including edge-touching
    ones, whose intersection has zero area — mark nothing; the
    request-relative slice bounds below are only sane for a positive-area
    overlap.
    """
    height = window.row_stop - window.row_start
    width = window.col_stop - window.col_start
    covered = np.zeros((height, width), dtype=bool)
    for other in covering:
        if not other.intersects(window):
            continue
        covered[
            max(window.row_start, other.row_start)
            - window.row_start : min(window.row_stop, other.row_stop)
            - window.row_start,
            max(window.col_start, other.col_start)
            - window.col_start : min(window.col_stop, other.col_stop)
            - window.col_start,
        ] = True
    return covered


def _subtract_windows(
    target: RasterWindow, cuts: Sequence[RasterWindow]
) -> tuple[RasterWindow, ...]:
    """Disjoint rectangle decomposition of ``target − union(cuts)``.

    The retained-remainder half of R5a supersession: which cells of a
    superseded entry the newer batch does NOT republish. Shares the
    run-decomposition of :func:`_missing_rectangles` (the same math
    ``window_coverage`` reports uncovered slivers with), so both callers
    decompose rectilinear remainders identically — deterministic
    ``(row_start, col_start)`` order, positive area, exact tiling. An
    empty ``cuts`` or a cut set that removes no cells (edge-touching
    windows only) yields ``(target,)``; a cut set covering every cell
    yields ``()``.
    """
    if not cuts:
        return (target,)
    return _missing_rectangles(_covered_mask(target, cuts), target)


def _missing_rectangles(
    covered: np.ndarray, window: RasterWindow
) -> tuple[RasterWindow, ...]:
    """Decompose the uncovered cells of ``covered`` (request-relative).

    Maximal-rectangle run decomposition (per-row horizontal runs merged
    downward while the column span continues), reported in absolute grid
    coordinates, sorted by ``(row_start, col_start)`` — deterministic, so
    identical store states report identical slivers. The rectangles are
    disjoint and tile exactly the uncovered cells.
    """
    rows, cols = covered.shape
    uncovered = ~covered
    open_runs: dict[tuple[int, int], list[int]] = {}
    emitted: list[RasterWindow] = []

    def _close(run: list[int]) -> None:
        emitted.append(
            RasterWindow(
                window.row_start + run[0],
                window.row_start + run[1],
                window.col_start + run[2],
                window.col_start + run[3],
            )
        )

    for row in range(rows):
        row_mask = uncovered[row]
        starts = np.flatnonzero(
            np.diff(np.concatenate(([0], row_mask.view(np.int8), [0]))) == 1
        )
        stops = np.flatnonzero(
            np.diff(np.concatenate(([0], row_mask.view(np.int8), [0]))) == -1
        )
        carried: dict[tuple[int, int], list[int]] = {}
        for start, stop in zip(starts, stops):
            span = (int(start), int(stop))
            run = open_runs.pop(span, None)
            if run is not None:
                run[1] = row + 1  # extend the rectangle down one row
            else:
                run = [row, row + 1, span[0], span[1]]
            carried[span] = run
        for run in open_runs.values():
            _close(run)  # no continuation row: the rectangle ends
        open_runs = carried
    for run in open_runs.values():
        _close(run)
    emitted.sort(key=lambda w: (w.row_start, w.col_start))
    return tuple(emitted)


def _entry_content(
    entries: Sequence[TemporalResultEntry],
) -> tuple[tuple[int, int, int, int, str, str | None], ...]:
    """Content fingerprint of a key's entries: window, mode, patch path.

    Two same-revision same-job batches whose fingerprints differ are NOT an
    idempotent replay (u-c1b L7).
    """
    return tuple(
        (
            entry.write_window.row_start,
            entry.write_window.col_start,
            entry.write_window.row_stop,
            entry.write_window.col_stop,
            entry.mode,
            None if entry.patch_path is None else str(entry.patch_path),
        )
        for entry in sorted(entries, key=_window_sort_key)
    )
