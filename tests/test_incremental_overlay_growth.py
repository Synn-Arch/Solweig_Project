# SPDX-License-Identifier: GPL-3.0-only
"""U-C6 tests: bounded overlay growth (u-c4-review M3) and the dirty rule.

Fast suite (worker solvers stubbed, no SOLWEIG physics): the accumulated
land-cover overlay COALESCES per-cell last-write-wins at commit, so both
the accumulated patch count and each paint-only batch's dirty footprint
stay O(edited area) — never O(session strokes). The per-batch dirty rule
is the symmetric difference between the staged overlay and the PUBLISHED
overlay state: a repaint of already-published values dirties nothing
(clean no-op, revision unmoved), and a paint elsewhere never recomputes
the session's earlier footprints.

The forcing (met) equivalence check: per-(variable, time) records already
fold last-write-wins in the executor's batch fold, so the accumulated
overlay cannot grow with repaints — verified here alongside the paint
rule so the two overlay families carry the same discipline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from solweig_gpu.incremental.adapters.landcover import (
    LandCoverOverlay,
    coalesce_overlay,
    overlay_diff_windows,
)
from solweig_gpu.incremental.geometry import RasterWindow
from solweig_gpu.incremental.result import ResultPatch

from tests.test_incremental_lc_integration import (
    PAINT_WINDOW,
    PAINT_WINDOW_B,
    _covers,
    _fake_local_patches,
    _paint_command,
    _tiny_executor,
)
from tests.test_incremental_met_integration import _executor_met_edit

#: A repaint region strictly inside PAINT_WINDOW (rows/cols 43..46).
REPAINT_WINDOW = RasterWindow(43, 46, 43, 46)


# ---------------------------------------------------------------------------
# Canonical coalescing (unit level)
# ---------------------------------------------------------------------------


class TestCoalesceOverlay:
    def _baseline(self) -> np.ndarray:
        grid = np.full((16, 16), 5, dtype=np.int64)
        grid[4:8, 4:8] = 1
        return grid

    def _patch(self, window: RasterWindow, after: int):
        from solweig_gpu.incremental.edit_types import LandCoverPaintPatch

        return LandCoverPaintPatch(
            window=window,
            before_classes=(-1,) * window.area,
            after_classes=(after,) * window.area,
        )

    def test_n_overlapping_strokes_fold_to_one_patch(self):
        # The M3 shape: N repaints of overlapping rectangles used to
        # concatenate N patches forever; the canonical fold emits the
        # painted set as maximal rectangles (one here), values final.
        baseline = self._baseline()
        strokes = LandCoverOverlay(
            patches=tuple(
                self._patch(RasterWindow(4, 4 + n, 4, 4 + n), 2)
                for n in range(2, 9)
            )
        )
        folded = coalesce_overlay(strokes, baseline)
        assert folded is not None
        assert len(folded.patches) == 1
        resolved = folded.resolve(baseline)
        assert np.all(resolved[4:12, 4:12] == 2)
        assert np.all(resolved[:4, :] == baseline[:4, :])

    def test_resolve_equal_overlays_fold_to_equal_patches(self):
        baseline = self._baseline()
        one_stroke = LandCoverOverlay(patches=(self._patch(RasterWindow(4, 8, 4, 8), 2),))
        two_strokes = LandCoverOverlay(
            patches=(
                self._patch(RasterWindow(4, 8, 4, 8), 6),
                self._patch(RasterWindow(4, 8, 4, 8), 2),  # later stroke wins
            )
        )
        assert coalesce_overlay(one_stroke, baseline) == coalesce_overlay(
            two_strokes, baseline
        )

    def test_resolve_equal_overlays_with_different_painted_sets_fold_different(
        self,
    ):
        # The u-c6b M2 boundary: coalescing equality holds for the SAME
        # painted set plus the same resolved values. Two overlays that
        # merely RESOLVE equal — here both resolve to the baseline grid
        # byte-for-byte — but paint DIFFERENT cells fold to DIFFERENT
        # canonical patches, so fold equality can never be judged from
        # resolved grids alone (the painted set is the second axis).
        baseline = self._baseline()
        repaint_baseline_block = LandCoverOverlay(
            patches=(self._patch(RasterWindow(4, 8, 4, 8), 1),)
        )  # class 1 already lives in rows/cols 4..7 of the baseline
        paint_elsewhere_baseline_class = LandCoverOverlay(
            patches=(self._patch(RasterWindow(10, 14, 10, 14), 5),)
        )  # class 5 is the baseline fill everywhere else
        left = coalesce_overlay(repaint_baseline_block, baseline)
        right = coalesce_overlay(paint_elsewhere_baseline_class, baseline)
        assert left is not None and right is not None
        assert np.array_equal(left.resolve(baseline), baseline)
        assert np.array_equal(right.resolve(baseline), baseline)
        assert left != right  # same resolution, different painted sets
        assert left.patches[0].window == RasterWindow(4, 8, 4, 8)
        assert right.patches[0].window == RasterWindow(10, 14, 10, 14)

    def test_disjoint_strokes_union(self):
        baseline = self._baseline()
        folded = coalesce_overlay(
            LandCoverOverlay(
                patches=(
                    self._patch(RasterWindow(0, 2, 0, 2), 2),
                    self._patch(RasterWindow(10, 12, 10, 12), 6),
                )
            ),
            baseline,
        )
        assert folded is not None
        assert len(folded.patches) == 2  # disjoint cells union
        resolved = folded.resolve(baseline)
        assert np.all(resolved[0:2, 0:2] == 2)
        assert np.all(resolved[10:12, 10:12] == 6)

    def test_none_folds_to_none(self):
        assert coalesce_overlay(None, self._baseline()) is None

    def test_diff_windows_is_the_symmetric_difference(self):
        baseline = self._baseline()
        published = LandCoverOverlay(patches=(self._patch(RasterWindow(4, 8, 4, 8), 2),))
        # A repaint of the SAME values: zero dirty cells.
        assert overlay_diff_windows(published, published, baseline) == ()
        # A repaint of a SUBSET to a new value: the subset is the diff.
        subset = RasterWindow(6, 8, 6, 8)  # strictly inside the stroke
        staged = LandCoverOverlay(
            patches=(
                self._patch(RasterWindow(4, 8, 4, 8), 2),
                self._patch(subset, 6),
            )
        )
        windows = overlay_diff_windows(staged, published, baseline)
        cells = np.zeros((16, 16), dtype=bool)
        for window in windows:
            cells[window.row_start : window.row_stop, window.col_start : window.col_stop] = True
        assert cells.sum() == subset.area
        # The drop-overlay reset dirties exactly the published footprint.
        drop = overlay_diff_windows(None, published, baseline)
        assert len(drop) >= 1
        union = np.zeros((16, 16), dtype=bool)
        for window in drop:
            union[window.row_start : window.row_stop, window.col_start : window.col_stop] = True
        assert union.sum() == RasterWindow(4, 8, 4, 8).area


# ---------------------------------------------------------------------------
# Executor: paint-only batches stay O(edited area)
# ---------------------------------------------------------------------------


class TestPaintOnlyBatchesBounded:
    def test_paint_a_b_c_dirty_and_state_bounded(self, tmp_path: Path) -> None:
        ex = _tiny_executor(tmp_path)
        captured: dict = {}

        def fake_local(job_id, revision, forcing, write_windows, **kwargs):
            captured["windows"] = tuple(write_windows)
            return _fake_local_patches(job_id, revision, write_windows)

        ex._worker._solve_local = fake_local
        baseline_grid = np.asarray(ex.cache.landcover)

        # Paint A (large): publishes the paint footprint.
        first = ex.execute([ex.validate(_paint_command(classes=1))])
        assert first.published
        assert first.mode == "local"
        assert any(_covers(w, PAINT_WINDOW) for w in captured["windows"])

        # Paint B (small, elsewhere): the batch's dirty footprint is B
        # ONLY — A's cells are already published, so they are not staged
        # dirty and are not recomputed (the pre-u-c6 overlay recomputed
        # every prior footprint on every paint-only job).
        second = ex.execute(
            [
                ex.validate(
                    _paint_command(
                        revision=ex.scene_revision,
                        window=PAINT_WINDOW_B,
                        classes=2,
                        edit_id="lc-2",
                    )
                )
            ]
        )
        assert second.published
        for window in captured["windows"]:
            assert not (
                window.row_stop > PAINT_WINDOW.row_start
                and window.row_start < PAINT_WINDOW.row_stop
                and window.col_stop > PAINT_WINDOW.col_start
                and window.col_start < PAINT_WINDOW.col_stop
            ), f"paint B recomputed paint A's footprint: {window}"

        # The accumulated overlay after two disjoint paints is TWO
        # canonical patches (the pre-u-c6 fold appended one patch per
        # stroke; N paints would carry N forever).
        applied = ex._applied_landcover_overlay
        assert applied is not None
        assert len(applied.patches) == 2
        resolved = applied.resolve(baseline_grid)
        assert np.all(resolved[40:48, 40:48] == 1)
        assert np.all(resolved[110:116, 103:109] == 2)

        # Paint C overlapping A with the SAME values: a clean no-op. The
        # scene revision does not advance, nothing publishes, and the
        # accumulated overlay is unchanged (same canonical form).
        before_revision = ex.scene_revision
        before_store_revision = ex.store.max_revision()
        third = ex.execute(
            [
                ex.validate(
                    _paint_command(
                        revision=ex.scene_revision,
                        classes=1,  # same window, same class as published
                        edit_id="lc-3",
                    )
                )
            ]
        )
        assert third.status == "no-op"
        assert not third.published
        assert ex.scene_revision == before_revision
        assert ex.store.max_revision() == before_store_revision
        assert ex._applied_landcover_overlay == applied

    def test_repaint_changed_values_dirties_only_changed_cells(
        self, tmp_path: Path
    ) -> None:
        ex = _tiny_executor(tmp_path)
        captured: dict = {}

        def fake_local(job_id, revision, forcing, write_windows, **kwargs):
            captured["windows"] = tuple(write_windows)
            return _fake_local_patches(job_id, revision, write_windows)

        ex._worker._solve_local = fake_local
        first = ex.execute([ex.validate(_paint_command(classes=1))])
        assert first.published

        # Repaint a strict SUBSET of the published footprint to a new
        # class: the dirty footprint is the subset, not the whole stroke.
        second = ex.execute(
            [
                ex.validate(
                    _paint_command(
                        revision=ex.scene_revision,
                        window=REPAINT_WINDOW,
                        classes=6,
                        edit_id="lc-2",
                    )
                )
            ]
        )
        assert second.published
        changed_cells = np.zeros((ex.grid.rows, ex.grid.cols), dtype=bool)
        # Write windows carry the GVF margin; the DIRTY diagnostics carry
        # the exact pre-margin footprint.
        dirty = second.diagnostics["dirty_windows"]
        assert dirty, "a value-changing repaint must dirty its cells"
        for row_start, row_stop, col_start, col_stop in dirty:
            changed_cells[row_start:row_stop, col_start:col_stop] = True
        assert changed_cells.sum() == REPAINT_WINDOW.area
        baseline_grid = np.asarray(ex.cache.landcover)
        resolved = ex._applied_landcover_overlay.resolve(baseline_grid)
        assert np.all(resolved[43:46, 43:46] == 6)
        assert np.all(resolved[40:43, 40:48] == 1)  # earlier paint survives

    def test_accumulated_patch_count_bounded_across_repaints(
        self, tmp_path: Path
    ) -> None:
        # N repaints of the same window to alternating classes: the
        # accumulated overlay stays ONE canonical patch (last value
        # wins), never one patch per repaint.
        ex = _tiny_executor(tmp_path)
        ex._worker._solve_local = (
            lambda job_id, revision, forcing, write_windows, **kwargs: (
                _fake_local_patches(job_id, revision, write_windows)
            )
        )
        for index, classes in enumerate((1, 6, 1, 6, 1)):
            executed = ex.execute(
                [
                    ex.validate(
                        _paint_command(
                            revision=ex.scene_revision,
                            classes=classes,
                            edit_id=f"lc-{index}",
                        )
                    )
                ]
            )
            assert executed.published
            applied = ex._applied_landcover_overlay
            assert applied is not None
            assert len(applied.patches) == 1, (
                "accumulated overlay grew with repaint count "
                f"(index {index}: {len(applied.patches)} patches)"
            )

    def test_paint_back_to_baseline_repaint_is_dirty_but_state_kept(
        self, tmp_path: Path
    ) -> None:
        # Painting a cell back to its baseline class KEEPS the cell in the
        # canonical painted set (at the baseline value) — the overlay is
        # the paint STATE — but the batch is real work (values changed vs
        # published), and a SECOND such batch is a clean no-op.
        ex = _tiny_executor(tmp_path)
        ex._worker._solve_local = (
            lambda job_id, revision, forcing, write_windows, **kwargs: (
                _fake_local_patches(job_id, revision, write_windows)
            )
        )
        baseline_grid = np.asarray(ex.cache.landcover)
        baseline_class = int(baseline_grid[40, 40])
        first = ex.execute([ex.validate(_paint_command(classes=1))])
        assert first.published
        back = ex.execute(
            [
                ex.validate(
                    _paint_command(
                        revision=ex.scene_revision,
                        classes=baseline_class,
                        edit_id="lc-back",
                    )
                )
            ]
        )
        assert back.published
        resolved = ex._applied_landcover_overlay.resolve(baseline_grid)
        assert np.array_equal(resolved[40:48, 40:48], baseline_grid[40:48, 40:48])
        assert len(ex._applied_landcover_overlay.patches) == 1
        again = ex.execute(
            [
                ex.validate(
                    _paint_command(
                        revision=ex.scene_revision,
                        classes=baseline_class,
                        edit_id="lc-again",
                    )
                )
            ]
        )
        assert again.status == "no-op"
        assert not again.published


# ---------------------------------------------------------------------------
# Forcing (met) equivalence: no analogous growth
# ---------------------------------------------------------------------------


class TestForcingOverlayBounded:
    def test_republished_met_values_fold_not_append(self, tmp_path: Path) -> None:
        # The forcing fold is per-(variable, time) records — dict-fold
        # last-write-wins — so repaints REPLACE records instead of
        # appending, and a repaint of the published value is a no-op.
        # The edits are utci_only (wind_speed): a radiation-affecting edit
        # routes the G2.1 warm sparse path, whose real split solve no
        # tiny-site stub can serve, while the r3a refusal (no tmrt
        # published here) keeps the batches on the stubbed worker path.
        from tests.test_incremental_remediation_uc1b import _stub_full

        ex = _tiny_executor(tmp_path)
        _stub_full(ex)
        first = ex.execute(
            [
                _executor_met_edit(
                    ex, time_index=1, value=26.0, variable="wind_speed"
                )
            ]
        )
        assert first.published
        applied = ex._applied_forcing_overlay
        assert applied is not None
        assert len(applied.changes) == 1

        # Same (variable, time), new value: the record is REPLACED, not
        # appended.
        second = ex.execute(
            [
                _executor_met_edit(
                    ex, time_index=1, value=27.0, variable="wind_speed",
                    edit_id="met-2",
                )
            ]
        )
        assert second.published
        assert len(ex._applied_forcing_overlay.changes) == 1
        change = ex._applied_forcing_overlay.changes[0]
        assert change.after_value == 27.0

        # Same (variable, time), SAME value: the batch folds to the
        # published scenario state — a clean no-op, revision unmoved.
        before = ex.scene_revision
        third = ex.execute(
            [
                _executor_met_edit(
                    ex, time_index=1, value=27.0, variable="wind_speed",
                    edit_id="met-3",
                )
            ]
        )
        assert third.status == "no-op"
        assert not third.published
        assert ex.scene_revision == before
        assert len(ex._applied_forcing_overlay.changes) == 1

    def test_distinct_times_accumulate_per_record(self, tmp_path: Path) -> None:
        from tests.test_incremental_remediation_uc1b import _stub_full

        ex = _tiny_executor(tmp_path)
        _stub_full(ex)
        assert ex.execute(
            [
                _executor_met_edit(
                    ex, time_index=0, value=25.0, variable="wind_speed"
                )
            ]
        ).published
        assert ex.execute(
            [
                _executor_met_edit(
                    ex, time_index=1, value=26.0, variable="wind_speed",
                    edit_id="met-2",
                )
            ]
        ).published
        assert ex.execute(
            [
                _executor_met_edit(
                    ex, time_index=2, value=27.0, variable="wind_speed",
                    edit_id="met-3",
                )
            ]
        ).published
        # Three DISTINCT (variable, time) records — bounded by the series
        # length, not by the session's edit count.
        assert len(ex._applied_forcing_overlay.changes) == 3
