# SPDX-License-Identifier: GPL-3.0-only
"""U-D residual intake: one probe per closed review residual.

Covers the U-D items that live in this packet's owned files:

- **(b) store reader contract** — :meth:`TemporalResultStore.window_coverage`
  classifies a windowed read as full/partial/absent and decomposes the
  uncovered sliver into disjoint rectangles, so a reader can DETECT the
  R2a case (a newer windowed batch superseded only part of what a read
  needs) and fall back to full recompute instead of silently serving
  partial data. ``revision_at`` answers the MAX published revision of a
  key — a retained older window must not mask a newer one.
- **(d) identical-value resubmit refusals** — every adapter family refuses
  a declared old_state == new_state edit with a typed
  :class:`~solweig_gpu.incremental.edit_types.EditStateError` whose reason
  names the property (variable, window, parameter) and BOTH values.
- **(e) L4 worker footprint guard** — ``guard_patch_window`` /
  ``guard_read_covers_write`` make the worker's read/write footprint
  contract explicit and typed.
- **(e) L5 solver scratch staging** — an overlay that cannot be resolved
  refuses BEFORE the first scratch mutation: a failed staging pass leaves
  the previous scratch untouched, never a half-staged site.
- **(h) evidence-based no-op diagnostics** — the worker's no-op outcomes
  say WHY nothing is pending (nothing staged vs cancelled batch vs empty
  staged land-cover footprint) instead of a bare generic reason.

Item (f) is a recorded skip (``solweig_gpu.py``/``utci_process.py`` stay
upstream-faithful); item (i) citation repairs are comment-only and carry
no runtime probe.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from solweig_gpu.incremental.edit_types import EditStateError
from solweig_gpu.incremental.geometry import RasterGrid, RasterWindow
from solweig_gpu.incremental.solver import (
    SolverInputError,
    load_site_forcing,
    run_full_tile,
)
from solweig_gpu.incremental.store import (
    CoverageStatus,
    StoreError,
    TemporalResultEntry,
    TemporalResultStore,
    WindowCoverage,
)
from solweig_gpu.incremental.trees import TreeLayer
from solweig_gpu.incremental.worker import (
    WorkerError,
    guard_patch_window,
    guard_read_covers_write,
)

from tests.test_incremental_adapters_building import (
    BLOCK_A,
    building_command,
    context as building_context,
    make_adapter as make_building_adapter,
)
from tests.test_incremental_adapters_landcover import (
    WINDOW as LC_WINDOW,
    context as landcover_context,
    make_adapter as make_landcover_adapter,
    paint_command,
)
from tests.test_incremental_adapters_met_time import (
    context as met_context,
    make_adapter as make_met_adapter,
    met_command,
)
from tests.test_incremental_adapters_model_params import (
    context as params_context,
    make_adapter as make_params_adapter,
    param_command,
)
from tests.test_incremental_adapters_vegetation import (
    TREE_A,
    context as veg_context,
    make_adapter as make_veg_adapter,
    veg_command,
)
from tests.test_incremental_lc_integration import (
    DATE_STR,
    _lc_overlay,
    _nolc_site,
    _tiny_worker,
)
from tests.test_incremental_worker import TINY_ADD


# ---------------------------------------------------------------------------
# (b) store reader contract
# ---------------------------------------------------------------------------


def _wentry(
    window: RasterWindow,
    revision: int,
    job: str = "job-1",
    *,
    node: str = "utci",
    time: int = 0,
) -> TemporalResultEntry:
    return TemporalResultEntry(
        node_id=node,
        time_index=time,
        scene_revision=revision,
        job_id=job,
        mode="local",
        write_window=window,
        patch_path=Path(f"/tmp/rev-{revision:06d}-{job}"),
    )


def _store(rows: int = 8, cols: int = 8) -> TemporalResultStore:
    return TemporalResultStore(grid=RasterGrid(rows, cols, 1.0))


class TestWindowCoverage:
    def test_full_partial_absent_classification(self) -> None:
        store = _store()
        entry = _wentry(RasterWindow(0, 4, 0, 4), revision=1)
        store.publish([entry])

        full = store.window_coverage("utci", 0, window=RasterWindow(0, 4, 0, 4))
        assert full.status is CoverageStatus.FULL
        assert full.entries == (entry,)
        assert full.missing_windows == ()
        assert full.covered_area == 16
        assert full.missing_area == 0

        partial = store.window_coverage("utci", 0, window=RasterWindow(0, 4, 2, 6))
        assert partial.status is CoverageStatus.PARTIAL
        assert partial.entries == (entry,)  # overlapping entries, window order
        assert partial.covered_area == 8
        assert partial.missing_area == 8
        assert partial.missing_windows == (RasterWindow(0, 4, 4, 6),)

        absent = store.window_coverage("utci", 0, window=RasterWindow(4, 8, 4, 8))
        assert absent.status is CoverageStatus.ABSENT
        assert absent.entries == ()
        assert absent.covered_area == 0
        assert absent.missing_area == 16
        assert absent.missing_windows == (RasterWindow(4, 8, 4, 8),)

    def test_unpublished_key_is_absent(self) -> None:
        store = _store()
        coverage = store.window_coverage("tmrt", 0, window=RasterWindow(0, 2, 0, 2))
        assert coverage.status is CoverageStatus.ABSENT
        assert coverage.missing_windows == (RasterWindow(0, 2, 0, 2),)

    def test_partial_overlap_reader_falls_back_not_serves(self) -> None:
        # The R2a residual, updated by R5a cell-remainder retention: rev 1
        # publishes window A; rev 2 republishes a PARTIALLY overlapping
        # window B. Under the old whole-entry rule B's publish dropped A
        # and the sliver of A outside B lost provenance — the phantom
        # coverage gap. Now A keeps its remainder (same revision, job,
        # patch reference; narrowed window), a read spanning both
        # classifies FULL, and the composed serve over the request is
        # BITWISE: rev-2 values inside B, rev-1 values in the remainder
        # cells. The fallback contract SURVIVES where coverage is
        # genuinely incomplete (see the PARTIAL case at the end).
        store = _store()
        store.publish([_wentry(RasterWindow(0, 4, 0, 4), revision=1, job="j1")])
        store.publish([_wentry(RasterWindow(0, 4, 2, 6), revision=2, job="j2")])
        assert store.window_entries("utci", 0) == (
            _wentry(RasterWindow(0, 4, 0, 2), revision=1, job="j1"),
            _wentry(RasterWindow(0, 4, 2, 6), revision=2, job="j2"),
        )

        request = RasterWindow(0, 4, 0, 6)
        coverage = store.window_coverage("utci", 0, window=request)
        # The OLD hazard — a spanning read degraded to PARTIAL with an
        # unprovenanced sliver after the first windowed batch — is gone.
        assert coverage.status is CoverageStatus.FULL
        assert coverage.covered_area == request.area
        assert coverage.missing_area == 0
        assert coverage.missing_windows == ()

        # Composed serve, sliced per entry the way the executor's plane
        # assembly slices (offsets against the PATCH window; patch
        # arrays are window-relative).
        rev1_patch = np.arange(16, dtype=np.int64).reshape(4, 4)
        rev2_patch = 10_000 + np.arange(16, dtype=np.int64).reshape(4, 4)
        patch_windows = {
            Path("/tmp/rev-000001-j1"): RasterWindow(0, 4, 0, 4),
            Path("/tmp/rev-000002-j2"): RasterWindow(0, 4, 2, 6),
        }
        plane = np.full((4, 6), -1, dtype=np.int64)
        for entry in coverage.entries:
            payload = rev2_patch if entry.scene_revision == 2 else rev1_patch
            patch_window = patch_windows[entry.patch_path]
            window = entry.write_window
            plane[
                window.row_start : window.row_stop,
                window.col_start : window.col_stop,
            ] = payload[
                window.row_start - patch_window.row_start : window.row_stop - patch_window.row_start,
                window.col_start - patch_window.col_start : window.col_stop - patch_window.col_start,
            ]
        expected = np.full((4, 6), -1, dtype=np.int64)
        expected[:, 0:2] = rev1_patch[:, 0:2]
        expected[:, 2:6] = rev2_patch
        np.testing.assert_array_equal(plane, expected)

        # A republished cell NEVER resolves to the older entry.
        for col in range(2, 6):
            resolved = store.lookup_window("utci", 0, row=1, col=col)
            assert resolved is not None and resolved.scene_revision == 2

        # The reader fallback contract survives: cols [6, 8) are
        # recorded by nothing, so a read reaching there still classifies
        # PARTIAL with the sliver named — the caller recomputes, never
        # serves the covered cells alone.
        partial = store.window_coverage("utci", 0, window=RasterWindow(0, 4, 0, 8))
        assert partial.status is CoverageStatus.PARTIAL
        assert partial.missing_windows == (RasterWindow(0, 4, 6, 8),)
        # The reader-facing revision answer is the NEWEST published one.
        assert store.revision_at("utci", 0) == 2

    def test_missing_rectangles_tile_non_rectangular_holes_exactly(self) -> None:
        store = _store()
        # Covered: the middle-top block only. Uncovered: an L-shaped hole.
        store.publish([_wentry(RasterWindow(0, 2, 2, 4), revision=1)])
        request = RasterWindow(0, 4, 0, 6)
        coverage = store.window_coverage("utci", 0, window=request)
        assert coverage.status is CoverageStatus.PARTIAL
        assert coverage.missing_windows == (
            RasterWindow(0, 2, 0, 2),
            RasterWindow(0, 2, 4, 6),
            RasterWindow(2, 4, 0, 6),
        )
        # Pairwise cell-disjoint (edge-adjacent is fine — the rectangles
        # tile the hole), deterministic order, exact tiling of the
        # uncovered cells.
        windows = coverage.missing_windows

        def _overlap_area(a: RasterWindow, b: RasterWindow) -> int:
            rows = min(a.row_stop, b.row_stop) - max(a.row_start, b.row_start)
            cols = min(a.col_stop, b.col_stop) - max(a.col_start, b.col_start)
            return max(0, rows) * max(0, cols)

        for i, first in enumerate(windows):
            for second in windows[i + 1 :]:
                assert _overlap_area(first, second) == 0
        assert sum(w.area for w in windows) == coverage.missing_area
        rebuilt = np.zeros((4, 6), dtype=bool)
        for window in windows:
            rebuilt[
                window.row_start - request.row_start : window.row_stop - request.row_start,
                window.col_start - request.col_start : window.col_stop - request.col_start,
            ] = True
        expected = np.ones((4, 6), dtype=bool)
        expected[0:2, 2:4] = False
        assert np.array_equal(rebuilt, expected)

    def test_disjoint_newer_window_retained_and_revision_is_max(self) -> None:
        # R2 retention keeps non-intersecting older windows, so one key can
        # carry entries at several revisions. The reader-facing
        # ``revision_at`` must answer the MAX published revision, not the
        # first entry's (a retained rev-1 window ahead of it in window
        # order must not mask a rev-3 publication). Supersession's
        # intersection test counts edge-touching windows, so the retained
        # window must be separated by at least one cell.
        store = _store()
        store.publish([_wentry(RasterWindow(0, 4, 0, 4), revision=1, job="j1")])
        store.publish([_wentry(RasterWindow(5, 8, 5, 8), revision=3, job="j3")])
        entries = store.window_entries("utci", 0)
        assert len(entries) == 2
        assert entries[0].scene_revision == 1  # window order: rows 0-4 first
        assert store.revision_at("utci", 0) == 3
        assert store.revision_at("utci", 9) is None

    def test_invalid_windows_refused_with_store_error(self) -> None:
        store = _store()
        with pytest.raises(StoreError, match="must be a RasterWindow"):
            store.window_coverage("utci", 0, window={"row_start": 0})
        with pytest.raises(StoreError, match="non-empty"):
            store.window_coverage("utci", 0, window=RasterWindow(2, 2, 0, 4))
        with pytest.raises(StoreError, match="outside the 8x8 grid"):
            store.window_coverage("utci", 0, window=RasterWindow(0, 9, 0, 4))
        with pytest.raises(StoreError, match="outside the 8x8 grid"):
            store.window_coverage("utci", 0, window=RasterWindow(-1, 2, 0, 4))

    def test_coverage_object_is_the_documented_reader_answer(self) -> None:
        store = _store()
        store.publish([_wentry(RasterWindow(0, 4, 0, 4), revision=1)])
        coverage = store.window_coverage("utci", 0, window=RasterWindow(0, 4, 0, 4))
        assert isinstance(coverage, WindowCoverage)
        assert (coverage.node_id, coverage.time_index) == ("utci", 0)
        assert coverage.window == RasterWindow(0, 4, 0, 4)
        assert coverage.status.value == "full"


# ---------------------------------------------------------------------------
# (d) identical-value resubmit refusals, per family
# ---------------------------------------------------------------------------


class TestIdenticalValueResubmit:
    def test_vegetation_names_tree_properties_and_values(self) -> None:
        adapter = make_veg_adapter()
        with pytest.raises(
            EditStateError,
            match=r"no-op: tree 'tree-a'.*height_m: 10.0 -> 10.0",
        ) as excinfo:
            adapter.validate(
                veg_command(
                    "update",
                    old_state=dict(TREE_A),
                    new_state={
                        "tree_id": TREE_A["tree_id"],
                        "height_m": TREE_A["height_m"],
                    },
                ),
                veg_context(),
            )
        # Every declared property appears with both (equal) values.
        message = str(excinfo.value)
        for name in ("height_m", "canopy_radius_m"):
            assert f"{name}: " in message

    def test_building_names_properties_and_values(self) -> None:
        adapter = make_building_adapter()
        with pytest.raises(
            EditStateError,
            match=r"no-op: building .*already carries.*footprint_m",
        ) as excinfo:
            adapter.validate(
                building_command(
                    "move",
                    old_state=dict(BLOCK_A),
                    new_state=dict(BLOCK_A),
                ),
                building_context(),
            )
        message = str(excinfo.value)
        assert "height_m: " in message
        assert "identical-value resubmit" in message

    def test_meteorology_names_variable_timestep_and_values(self) -> None:
        adapter = make_met_adapter()
        with pytest.raises(
            EditStateError,
            match=r"no-op.*humidity@t=1: 50.0 -> 50.0",
        ):
            adapter.validate(
                met_command(
                    old_state={"time_index": 1, "values": {"humidity": 50.0}},
                    new_state={"time_index": 1, "values": {"humidity": 50.0}},
                ),
                met_context(),
            )

    def test_model_parameters_names_parameter_and_values(self) -> None:
        adapter = make_params_adapter()
        with pytest.raises(
            EditStateError,
            match=r"no-op.*albedo_b: 0.2 -> 0.2",
        ):
            adapter.validate(
                param_command(
                    "update",
                    old_state={"albedo_b": 0.2},
                    new_state={"albedo_b": 0.2},
                ),
                params_context(),
            )

    def test_landcover_names_window_and_class_sets(self) -> None:
        adapter = make_landcover_adapter()
        window_cells = (
            LC_WINDOW["row_stop"] - LC_WINDOW["row_start"]
        ) * (LC_WINDOW["col_stop"] - LC_WINDOW["col_start"])
        with pytest.raises(
            EditStateError,
            match=r"no-op.*window.*before = \[5\], after = \[5\]",
        ):
            adapter.validate(
                paint_command(
                    old_state={
                        "window": dict(LC_WINDOW),
                        "classes": [5] * window_cells,
                    },
                    new_state={"window": dict(LC_WINDOW), "classes": 5},
                ),
                landcover_context(),
            )

    def test_landcover_interception_is_whole_window_only(self) -> None:
        # u-c6b lead ruling: a paint whose before-state is only PARTIALLY
        # value-equal still executes over its whole window — per-cell or
        # partial interception is deliberately absent at the adapter (the
        # executor intercepts value-equal repaints of PUBLISHED state
        # whole-batch, one level up).
        adapter = make_landcover_adapter()
        window_cells = (
            LC_WINDOW["row_stop"] - LC_WINDOW["row_start"]
        ) * (LC_WINDOW["col_stop"] - LC_WINDOW["col_start"])
        edit = adapter.validate(
            paint_command(
                old_state={
                    "window": dict(LC_WINDOW),
                    "classes": [5] * (window_cells // 2)
                    + [6] * (window_cells - window_cells // 2),
                },
                new_state={"window": dict(LC_WINDOW), "classes": 5},
            ),
            landcover_context(),
        )
        (patch,) = edit.delta.patches
        assert patch.after_classes == (5,) * window_cells


# ---------------------------------------------------------------------------
# (e) L4: worker footprint guard
# ---------------------------------------------------------------------------


class TestWorkerFootprintGuard:
    def test_in_grid_window_passes(self) -> None:
        grid = RasterGrid(8, 8, 1.0)
        guard_patch_window(RasterWindow(0, 8, 0, 8), grid, label="write")
        guard_patch_window(RasterWindow(2, 5, 1, 3), grid, label="read")

    def test_empty_window_refused(self) -> None:
        with pytest.raises(WorkerError, match="write window is empty"):
            guard_patch_window(
                RasterWindow(2, 2, 0, 4), RasterGrid(8, 8, 1.0), label="write"
            )

    def test_out_of_grid_window_refused(self) -> None:
        grid = RasterGrid(8, 8, 1.0)
        with pytest.raises(WorkerError, match="escapes the site grid"):
            guard_patch_window(RasterWindow(0, 9, 0, 4), grid, label="write")
        with pytest.raises(WorkerError, match="escapes the site grid"):
            guard_patch_window(RasterWindow(-1, 4, 0, 4), grid, label="read")

    def test_read_must_cover_write(self) -> None:
        write = RasterWindow(4, 8, 4, 8)
        guard_read_covers_write(RasterWindow(0, 8, 0, 8), write)
        guard_read_covers_write(write, write)
        with pytest.raises(WorkerError, match="does not contain"):
            guard_read_covers_write(RasterWindow(4, 8, 4, 7), write)
        with pytest.raises(WorkerError, match="does not contain"):
            guard_read_covers_write(RasterWindow(5, 8, 4, 8), write)


# ---------------------------------------------------------------------------
# (e) L5: solver scratch staging refuses before the first mutation
# ---------------------------------------------------------------------------


class TestSolverScratchStaging:
    def test_unresolvable_overlay_leaves_previous_scratch_untouched(
        self, tmp_path: Path
    ) -> None:
        # A site whose cache carries no landcover raster cannot resolve an
        # overlay. The refusal must fire BEFORE the scratch wipe/copies: a
        # previous scratch (here: a sentinel from an earlier job) survives
        # untouched instead of being replaced by a half-staged site.
        grid, nolc, cache = _nolc_site(tmp_path)
        forcing = load_site_forcing(
            cache, site_dir=nolc, selected_date_str=DATE_STR
        )
        overlay = _lc_overlay(cache, window=RasterWindow(8, 16, 8, 16))
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        sentinel = scratch / "sentinel-from-previous-job.txt"
        sentinel.write_text("previous scratch state")

        with pytest.raises(SolverInputError, match="no landcover raster"):
            run_full_tile(
                cache,
                TreeLayer(cache.tree_base, grid),
                forcing=forcing,
                site_dir=nolc,
                scratch_dir=scratch,
                requested_variables=("utci",),
                landcover_overlay=overlay,
            )
        assert sentinel.read_text() == "previous scratch state"
        assert not (scratch / "Building_DSM").exists()
        assert not (scratch / "Trees").exists()


# ---------------------------------------------------------------------------
# (h): evidence-based no-op diagnostics
# ---------------------------------------------------------------------------


class TestWorkerNoOpDiagnostics:
    def test_nothing_staged_reason_names_the_checked_sources(
        self, tmp_path: Path
    ) -> None:
        worker, _site, _grid = _tiny_worker(tmp_path)
        outcome = worker.run()
        assert outcome.status == "no-op"
        assert "no batch is staged" in outcome.diagnostics["reason"]
        assert outcome.diagnostics["batch_sequences"] is None
        assert outcome.diagnostics["forcing_overlay_pending"] is False
        assert outcome.diagnostics["landcover_overlay_pending"] is False
        assert outcome.diagnostics["model_parameters_pending"] is False

    def test_cancelled_batch_reason_names_the_empty_batch(
        self, tmp_path: Path
    ) -> None:
        worker, _site, _grid = _tiny_worker(tmp_path)
        worker.layer.add_tree(TINY_ADD)
        worker.layer.delete_tree(TINY_ADD.tree_id)
        outcome = worker.run()
        assert outcome.status == "no-op"
        assert "staged batch carries no edits" in outcome.diagnostics["reason"]
        assert outcome.diagnostics["batch_sequences"] == (1, 2)

    def test_empty_landcover_footprint_reason_names_the_footprint(
        self, tmp_path: Path
    ) -> None:
        worker, _site, _grid = _tiny_worker(tmp_path)
        overlay = _lc_overlay(worker.cache, window=RasterWindow(8, 16, 8, 16))
        worker.stage_landcover_overlay(overlay, dirty_windows=[])
        outcome = worker.run()
        assert outcome.status == "no-op"
        assert outcome.diagnostics["reason"] == (
            "edits coalesced to no-op: staged land-cover footprint is empty"
        )
