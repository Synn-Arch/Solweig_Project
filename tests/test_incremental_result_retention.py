# SPDX-License-Identifier: GPL-3.0-only
"""R5a cell-remainder retention tests (``TemporalResultStore`` supersession).

The R2 whole-entry rule drops a recorded entry WHOLE when a newer
windowed batch intersects its write window — so the FIRST windowed batch
after a full-tile publication kills full-tile coverage: the full-tile
entry intersects everything, is dropped wholesale, and the
non-overlapping remainder of its cells loses provenance
(``window_coverage(full)`` degrades to PARTIAL and the met fast path's
strict coverage gate refuses to serve; the r3a durability lever).

Cell-remainder retention supersedes CELLS, not entries: a recorded entry
that intersects the new batch keeps its non-intersecting remainder as
disjoint narrowed rectangles referencing the SAME patch payload (the
patch bytes already hold those cells' values at that revision — the
narrowed window only declares which cells a reader may take from it).

Invariants pinned here (one probe each):

1. full-tile @ rev N + windowed batch @ rev N+1 (sub-window) ⇒
   ``window_coverage(full)`` is FULL (whole-entry supersession: PARTIAL).
2. ``lookup_window`` outside the new windows resolves to the retained
   remainder entry; inside, to the new batch's entry.
3. remainder rectangles never overlap the new windows (a republished
   cell can never resolve back to the older entry) and ``revision_at``
   stays the max revision.
4. same-revision idempotent replay is untouched (retained entries are
   not perturbed by a byte-identical replay of the newer batch).
5. retained entries reference EXISTING patch paths — the original
   payload stays reachable and no new payload identity is invented.
6. the rectilinear decomposition is finite and disjoint: a full tile
   minus one interior window is EXACTLY four rectangles in the
   deterministic (row_start, col_start) order.
7. reader safety: the executor's plane assembly slices a patch payload
   by ``entry.write_window - patch.write_window`` offsets, so a narrowed
   entry pointing into the original larger patch translates correctly —
   pinned here by replicating that slicing against ramp payloads.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from solweig_gpu.incremental.geometry import RasterGrid, RasterWindow
from solweig_gpu.incremental.store import (
    CoverageStatus,
    TemporalResultEntry,
    TemporalResultStore,
)

FULL = RasterWindow(0, 10, 0, 10)
BATCH = RasterWindow(3, 6, 2, 7)  # an interior sub-window of FULL
FULL_JOB = "job-full"
BATCH_JOB = "job-local"
FULL_PATCH = Path("/data/patches/rev-000001-job-full")
BATCH_PATCH = Path("/data/patches/rev-000002-job-local")


def _wentry(
    window: RasterWindow,
    revision: int,
    job: str,
    *,
    patch: Path,
    node: str = "tmrt",
    time: int = 0,
) -> TemporalResultEntry:
    return TemporalResultEntry(
        node_id=node,
        time_index=time,
        scene_revision=revision,
        job_id=job,
        mode="full" if job == FULL_JOB else "local",
        write_window=window,
        patch_path=patch,
    )


def _store(rows: int = 10, cols: int = 10) -> TemporalResultStore:
    return TemporalResultStore(grid=RasterGrid(rows, cols, 1.0))


def _retained(entries: tuple[TemporalResultEntry, ...]) -> tuple[TemporalResultEntry, ...]:
    """The recorded entries still carrying the ORIGINAL (rev-1) provenance."""
    return tuple(entry for entry in entries if entry.scene_revision == 1)


def _overlap_area(a: RasterWindow, b: RasterWindow) -> int:
    rows = min(a.row_stop, b.row_stop) - max(a.row_start, b.row_start)
    cols = min(a.col_stop, b.col_stop) - max(a.col_start, b.col_start)
    return max(0, rows) * max(0, cols)


class TestFullTileSurvivesWindowedBatch:
    def test_full_tile_plus_windowed_batch_is_full_coverage(self) -> None:
        # Invariant 1 (THE RED): the first windowed batch after a
        # full-tile publication must not degrade full-window coverage.
        store = _store()
        store.publish([_wentry(FULL, 1, FULL_JOB, patch=FULL_PATCH)])
        store.publish([_wentry(BATCH, 2, BATCH_JOB, patch=BATCH_PATCH)])

        coverage = store.window_coverage("tmrt", 0, window=FULL)
        assert coverage.status is CoverageStatus.FULL
        assert coverage.missing_windows == ()
        assert coverage.covered_area == FULL.area
        assert coverage.missing_area == 0

    def test_cell_outside_new_windows_resolves_to_retained_remainder(self) -> None:
        # Invariant 2: a cell the batch never touched must still resolve
        # to the original payload — via the narrowed remainder entry.
        store = _store()
        store.publish([_wentry(FULL, 1, FULL_JOB, patch=FULL_PATCH)])
        store.publish([_wentry(BATCH, 2, BATCH_JOB, patch=BATCH_PATCH)])

        outside = store.lookup_window("tmrt", 0, row=0, col=0)
        assert outside is not None
        assert outside.scene_revision == 1
        assert outside.patch_path == FULL_PATCH

        inside = store.lookup_window("tmrt", 0, row=BATCH.row_start, col=BATCH.col_start)
        assert inside is not None
        assert inside.scene_revision == 2
        assert inside.patch_path == BATCH_PATCH

    def test_retained_entries_keep_original_provenance_fields(self) -> None:
        # A retained remainder declares a narrowed window only — revision,
        # job, mode, and patch reference are the original entry's.
        store = _store()
        store.publish([_wentry(FULL, 1, FULL_JOB, patch=FULL_PATCH)])
        store.publish([_wentry(BATCH, 2, BATCH_JOB, patch=BATCH_PATCH)])

        retained = _retained(store.window_entries("tmrt", 0))
        assert retained, "no rev-1 remainder retained after the windowed batch"
        for entry in retained:
            assert entry.job_id == FULL_JOB
            assert entry.mode == "full"
            assert entry.patch_path == FULL_PATCH
            assert entry.node_id == "tmrt"
            assert entry.time_index == 0


class TestRemainderGeometry:
    def test_full_tile_minus_one_window_is_four_disjoint_rectangles(self) -> None:
        # Invariant 6: the decomposition is finite, disjoint, and
        # deterministic — a full tile minus one interior window is
        # EXACTLY four rectangles in (row_start, col_start) order.
        store = _store()
        store.publish([_wentry(FULL, 1, FULL_JOB, patch=FULL_PATCH)])
        store.publish([_wentry(BATCH, 2, BATCH_JOB, patch=BATCH_PATCH)])

        entries = store.window_entries("tmrt", 0)
        assert _retained(entries) == (
            TemporalResultEntry(
                node_id="tmrt",
                time_index=0,
                scene_revision=1,
                job_id=FULL_JOB,
                mode="full",
                write_window=RasterWindow(0, 3, 0, 10),
                patch_path=FULL_PATCH,
            ),
            TemporalResultEntry(
                node_id="tmrt",
                time_index=0,
                scene_revision=1,
                job_id=FULL_JOB,
                mode="full",
                write_window=RasterWindow(3, 6, 0, 2),
                patch_path=FULL_PATCH,
            ),
            TemporalResultEntry(
                node_id="tmrt",
                time_index=0,
                scene_revision=1,
                job_id=FULL_JOB,
                mode="full",
                write_window=RasterWindow(3, 6, 7, 10),
                patch_path=FULL_PATCH,
            ),
            TemporalResultEntry(
                node_id="tmrt",
                time_index=0,
                scene_revision=1,
                job_id=FULL_JOB,
                mode="full",
                write_window=RasterWindow(6, 10, 0, 10),
                patch_path=FULL_PATCH,
            ),
        )

        # The remainders tile FULL minus BATCH exactly.
        remainders = _retained(entries)
        for i, first in enumerate(remainders):
            for second in remainders[i + 1 :]:
                assert _overlap_area(first.write_window, second.write_window) == 0
        assert sum(w.write_window.area for w in remainders) == FULL.area - BATCH.area

    def test_new_window_cells_never_resolve_to_retained_entry(self) -> None:
        # Invariant 3: the remainder is defined by subtraction — no
        # remainder rectangle overlaps any new window, so a republished
        # cell can never resolve back to the older entry.
        store = _store()
        store.publish([_wentry(FULL, 1, FULL_JOB, patch=FULL_PATCH)])
        store.publish([_wentry(BATCH, 2, BATCH_JOB, patch=BATCH_PATCH)])

        for entry in _retained(store.window_entries("tmrt", 0)):
            assert _overlap_area(entry.write_window, BATCH) == 0
        for row in range(BATCH.row_start, BATCH.row_stop):
            for col in range(BATCH.col_start, BATCH.col_stop):
                resolved = store.lookup_window("tmrt", 0, row=row, col=col)
                assert resolved is not None
                assert resolved.scene_revision == 2
        # The reader-facing revision watermark stays the newest one.
        assert store.revision_at("tmrt", 0) == 2

    def test_edge_touching_batch_keeps_entry_intact(self) -> None:
        # ``intersects`` counts edge-touching windows, but a zero-area
        # overlap removes no cells: the entry must survive verbatim (the
        # whole-entry rule dropped it, opening a fully phantom gap).
        store = _store(rows=12, cols=10)
        store.publish([_wentry(FULL, 1, FULL_JOB, patch=FULL_PATCH)])
        touching = RasterWindow(10, 12, 0, 10)  # shares only the row-10 edge
        store.publish([_wentry(touching, 2, BATCH_JOB, patch=BATCH_PATCH)])

        entries = store.window_entries("tmrt", 0)
        assert entries[0] == _wentry(FULL, 1, FULL_JOB, patch=FULL_PATCH)
        assert store.window_coverage("tmrt", 0, window=FULL).status is CoverageStatus.FULL

    def test_fully_covered_entry_is_fully_superseded(self) -> None:
        # A batch that republishes EVERY cell of an entry leaves no
        # remainder: the entry is dropped, no empty-window husk remains.
        store = _store()
        store.publish([_wentry(FULL, 1, FULL_JOB, patch=FULL_PATCH)])
        store.publish([_wentry(FULL, 2, BATCH_JOB, patch=BATCH_PATCH)])

        entries = store.window_entries("tmrt", 0)
        assert entries == (_wentry(FULL, 2, BATCH_JOB, patch=BATCH_PATCH),)
        assert store.window_coverage("tmrt", 0, window=FULL).status is CoverageStatus.FULL


class TestRetentionDiscipline:
    def test_identical_batch_replay_keeps_retained_entries(self) -> None:
        # Invariant 4: the same-revision idempotent replay path is
        # untouched — a byte-identical replay of the windowed batch must
        # keep the retained remainders exactly as they were.
        store = _store()
        store.publish([_wentry(FULL, 1, FULL_JOB, patch=FULL_PATCH)])
        store.publish([_wentry(BATCH, 2, BATCH_JOB, patch=BATCH_PATCH)])
        before = store.window_entries("tmrt", 0)

        store.publish([_wentry(BATCH, 2, BATCH_JOB, patch=BATCH_PATCH)])
        assert store.window_entries("tmrt", 0) == before
        assert store.window_coverage("tmrt", 0, window=FULL).status is CoverageStatus.FULL

    def test_retained_entries_reference_existing_patch_paths(self) -> None:
        # Invariant 5: retention is pure bookkeeping — the original
        # payload stays referenced (no orphaned patch dir) and no new
        # payload identity is invented for a remainder.
        store = _store()
        store.publish([_wentry(FULL, 1, FULL_JOB, patch=FULL_PATCH)])
        store.publish([_wentry(BATCH, 2, BATCH_JOB, patch=BATCH_PATCH)])

        paths = {entry.patch_path for entry in store.window_entries("tmrt", 0)}
        assert paths == {FULL_PATCH, BATCH_PATCH}

    def test_retention_is_deterministic_across_stores(self) -> None:
        # Same publish sequence, same recorded state — window order and
        # remainder decomposition included.
        windows = []
        for _ in range(2):
            store = _store()
            store.publish([_wentry(FULL, 1, FULL_JOB, patch=FULL_PATCH)])
            store.publish([_wentry(BATCH, 2, BATCH_JOB, patch=BATCH_PATCH)])
            windows.append(store.window_entries("tmrt", 0))
        assert windows[0] == windows[1]


class TestReaderPayloadTranslation:
    def test_narrowed_entries_slice_the_original_patch_correctly(self) -> None:
        # Invariant 7: the executor's plane assembly slices a patch
        # payload by ``entry.write_window - patch.write_window`` offsets.
        # A narrowed remainder points into the ORIGINAL (larger) patch,
        # so the translation must extract exactly the remainder's cells.
        store = _store()
        full_ramp = np.arange(100, dtype=np.float64).reshape(10, 10)
        # Patch arrays are window-relative: the batch's payload is shaped
        # to ITS write window, not to the grid.
        batch_payload = np.arange(100, 115, dtype=np.float64).reshape(3, 5)
        store.publish([_wentry(FULL, 1, FULL_JOB, patch=FULL_PATCH)])
        store.publish([_wentry(BATCH, 2, BATCH_JOB, patch=BATCH_PATCH)])

        coverage = store.window_coverage("tmrt", 0, window=FULL)
        assert coverage.status is CoverageStatus.FULL

        plane = np.full((10, 10), np.nan)
        for entry in coverage.entries:
            window = entry.write_window
            if entry.scene_revision == 1:
                payload, patch_window = full_ramp, FULL
            else:
                payload, patch_window = batch_payload, BATCH
            plane[
                window.row_start : window.row_stop,
                window.col_start : window.col_stop,
            ] = payload[
                window.row_start - patch_window.row_start : window.row_stop - patch_window.row_start,
                window.col_start - patch_window.col_start : window.col_stop - patch_window.col_start,
            ]
        expected = full_ramp.copy()
        expected[
            BATCH.row_start : BATCH.row_stop, BATCH.col_start : BATCH.col_stop
        ] = batch_payload
        np.testing.assert_array_equal(plane, expected)
