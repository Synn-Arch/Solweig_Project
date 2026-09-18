# SPDX-License-Identifier: GPL-3.0-only
"""U-C6 review remediation tests (agent u-c6b, lead ruling 2026-09-03).

The u-c6-review HIGH finding: the staged land-cover dirty footprint (the
VALUE symmetric difference vs the published overlay) broke the plan/scope
agreement because the plan promises the STROKE footprint (the planner sees
only deltas, never resolved values). Three repros, all of which published
at 574cdc3 (pre-u-c6) and all of which failed at the u-c6 merge:

- **(a) same-value repaint** — a paint whose cells already carry the target
  class (baseline or published) folds to a non-empty overlay whose resolved
  values equal the published ones: the value symmetric difference is EMPTY,
  so the worker saw a pending overlay with an empty dirty override and
  no-op'd, and the executor raised the misleading "found nothing pending"
  wedge-suspect error. Whole batch unexecutable.
- **(b) first paint partially over already-target cells** — a stroke that
  straddles cells already carrying the target class dirties only the
  value-changing sub-rectangle; the published write windows (diff + GVF
  margin) do not cover the plan's stroke-footprint promise, so
  ``_reconcile_scope`` refused the batch ("plan write window(s) not
  covered").
- **(c) mixed vegetation + same-value repaint** — the land-cover leg wedged
  as in (a)/(b) and the WHOLE batch (vegetation included) was refused.

Lead ruling implemented here:

1. DIRTY IS A SUPERSET OF THE PLAN'S PROMISE. Staged land-cover dirty
   windows = the batch's STROKE windows union the value-difference windows
   (diff ⊆ stroke normally; union for safety). Prior published footprints
   stay OUT of the current batch's dirty (only the current stroke dirties),
   so the u-c6 growth win — accumulation and per-batch dirty stay
   O(edited area) — is preserved.
2. NO-OP INTERCEPTION IS WHOLE-BATCH AND VALUE-BASED. A batch folds to a
   clean no-op only when EVERY family resolves value-equal to the published
   scenario state AND there are no other deltas (vegetation, forcing,
   parameters, building). The diagnostics text says "values already
   published"; the worker-side "nothing pending" error keeps naming the
   wedge suspects (double execution / mismatched transaction) and now
   distinguishes the two.
3. PRESENCE, NOT VALUE, MARKS SITE-GLOBAL PAYLOADS PENDING IN MIXED
   BATCHES. A forcing fold that resolves value-equal to the published one
   still marks the job pending when the batch carries any other real
   delta (the forcing closure is the full tile). Model parameters audited
   for the same wedge (dict folds compare BY VALUE already, and the
   params source is full-spatial so the plan itself routes full — pinned
   here).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from solweig_gpu.incremental.adapters.landcover import (
    LandCoverOverlay,
    coalesce_overlay,
)
from solweig_gpu.incremental.edit_types import EditCommand
from solweig_gpu.incremental.geometry import RasterWindow

from tests.test_incremental_executor import _veg_command
from tests.test_incremental_lc_integration import (
    PAINT_WINDOW,
    _covers,
    _fake_local_patches,
    _paint_command,
    _tiny_executor,
)
from tests.test_incremental_met_integration import _executor_met_edit
from tests.test_incremental_worker import TINY_ORIGIN, TINY_PIXEL

#: A paint window entirely over baseline-asphalt cells (class 1): the tiny
#: site's baseline is asphalt everywhere except the grass/bare-soil/water
#: patches, and this window avoids all of them AND the default vegetation
#: edit's influence window plus write margin (rows/cols ~9..71), so a mixed
#: vegetation batch can never accidentally cover the stroke.
ASPHALT_WINDOW = RasterWindow(100, 108, 10, 18)
#: A second far-away asphalt window (disjoint from ASPHALT_WINDOW even
#: after the GVF write margin), for prior-footprint exclusion probes.
ASPHALT_WINDOW_FAR = RasterWindow(100, 108, 60, 68)
#: A paint stroke that STRADDLES the baseline grass region (rows/cols
#: 38..51): rows/cols 38..50 are grass, the rest of the stroke is asphalt.
#: Painting asphalt over it changes only the grass corner — the reviewer's
#: "first paint partially over already-target cells" shape (target = the
#: baseline asphalt around the grass patch).
STRADDLE_WINDOW = RasterWindow(10, 50, 10, 50)
#: Grass cells inside STRADDLE_WINDOW (the only value-changing part).
STRADDLE_GRASS = RasterWindow(38, 50, 38, 50)


def _paint(
    executor,
    *,
    window: RasterWindow,
    classes,
    edit_id: str = "lc-1",
    revision: int | None = None,
):
    return executor.validate(
        _paint_command(
            revision=executor.scene_revision if revision is None else revision,
            window=window,
            classes=classes,
            edit_id=edit_id,
        )
    )


def _stub_local(executor, captured: dict) -> None:
    def fake_local(job_id, revision, forcing, write_windows, **kwargs):
        captured.setdefault("jobs", []).append(job_id)
        captured["windows"] = tuple(write_windows)
        return _fake_local_patches(job_id, revision, write_windows)

    executor._worker._solve_local = fake_local


def _stub_full(executor, captured: dict) -> None:
    def fake_full(job_id, revision, forcing):
        captured.setdefault("jobs", []).append(job_id)
        from tests.test_incremental_lc_integration import _stub_patch

        return [
            _stub_patch(
                ("utci",), job_id, revision, executor.grid.full_window, mode="full"
            )
        ]

    executor._worker._solve_full = fake_full


# ---------------------------------------------------------------------------
# Repro (a): same-value repaint (whole batch, clean no-op)
# ---------------------------------------------------------------------------


class TestReproSameValueRepaint:
    def test_first_paint_of_baseline_values_is_a_clean_noop(
        self, tmp_path: Path
    ) -> None:
        """Paint a window whose cells ALREADY carry the target class in the
        baseline: nothing any solve path could recompute differs, so the
        batch is a clean no-op — the revision does not advance and nothing
        publishes — never the misleading wedge-suspect ExecutorError."""
        ex = _tiny_executor(tmp_path)
        captured: dict = {}
        _stub_local(ex, captured)
        baseline_grid = np.asarray(ex.cache.landcover)
        # The window is baseline asphalt: painting asphalt is a same-value
        # repaint of never-published values. (Slice from the window constant
        # itself — u-c6-review round-2 F2: the literal [60:68, 20:28] pinned
        # an unrelated region that happened to be asphalt too.)
        assert np.all(
            baseline_grid[
                ASPHALT_WINDOW.row_start : ASPHALT_WINDOW.row_stop,
                ASPHALT_WINDOW.col_start : ASPHALT_WINDOW.col_stop,
            ]
            == 1
        )

        executed = ex.execute([_paint(ex, window=ASPHALT_WINDOW, classes=1)])

        assert executed.status == "no-op"
        assert not executed.published
        assert executed.mode is None
        assert ex.scene_revision == 0
        # No job ever dispatched: the stub records "jobs" only when the
        # worker calls a solve path, so an absent key IS the assertion.
        assert not captured.get("jobs")
        assert not list(ex.results_root.rglob("rev-*"))
        assert len(ex.store) == 0
        # The no-op text is the accurate one: values already published —
        # NOT the wedge suspects.
        reason = executed.diagnostics["reason"]
        assert "already published" in reason
        assert "double execution" not in reason

    def test_repaint_of_published_values_stays_a_clean_noop(
        self, tmp_path: Path
    ) -> None:
        """The u-c6 no-op discipline (repaint of already-PUBLISHED values)
        survives the footprint change unchanged."""
        ex = _tiny_executor(tmp_path)
        captured: dict = {}
        _stub_local(ex, captured)
        first = ex.execute([_paint(ex, window=PAINT_WINDOW, classes=1)])
        assert first.published
        before = ex.scene_revision
        applied = ex._applied_landcover_overlay

        again = ex.execute(
            [_paint(ex, window=PAINT_WINDOW, classes=1, edit_id="lc-2")]
        )
        assert again.status == "no-op"
        assert not again.published
        assert ex.scene_revision == before
        assert ex._applied_landcover_overlay == applied
        assert len(captured["jobs"]) == 1  # only the first paint dispatched


# ---------------------------------------------------------------------------
# Repro (b): first paint partially over already-target cells
# ---------------------------------------------------------------------------


class TestReproPartialOverTarget:
    def test_straddling_paint_publishes_and_covers_the_stroke(
        self, tmp_path: Path
    ) -> None:
        """A stroke whose value-diff is a strict sub-rectangle of the stroke
        still publishes windows covering the PLAN's stroke-footprint
        promise (dirty = stroke ∪ diff, not diff alone)."""
        ex = _tiny_executor(tmp_path)
        captured: dict = {}
        _stub_local(ex, captured)
        baseline_grid = np.asarray(ex.cache.landcover)
        # Fixture shape: the grass patch is the only value-changing corner.
        assert np.all(baseline_grid[38:50, 38:50] == 5)
        assert np.all(baseline_grid[10:38, 10:50] == 1)

        executed = ex.execute([_paint(ex, window=STRADDLE_WINDOW, classes=1)])

        assert executed.published
        assert executed.mode == "local"
        # The stroke footprint (the plan's write-window promise) is covered
        # by the published write windows.
        assert any(
            _covers(w, STRADDLE_WINDOW) for w in captured["windows"]
        ), captured["windows"]
        # The exact dirty footprint is the STROKE (superset of the diff):
        # the batch recomputes the whole stroke, wasted-but-correct over
        # the already-asphalt cells.
        dirty = executed.diagnostics["dirty_windows"]
        dirty_cells = np.zeros((ex.grid.rows, ex.grid.cols), dtype=bool)
        for row_start, row_stop, col_start, col_stop in dirty:
            dirty_cells[row_start:row_stop, col_start:col_stop] = True
        assert dirty_cells[STRADDLE_WINDOW.row_start : STRADDLE_WINDOW.row_stop,
                           STRADDLE_WINDOW.col_start : STRADDLE_WINDOW.col_stop].all()
        # The canonical overlay keeps the painted set (baseline-equal cells
        # included) at their final values.
        applied = ex._applied_landcover_overlay
        assert applied is not None
        resolved = applied.resolve(baseline_grid)
        assert np.all(
            resolved[
                STRADDLE_WINDOW.row_start : STRADDLE_WINDOW.row_stop,
                STRADDLE_WINDOW.col_start : STRADDLE_WINDOW.col_stop,
            ]
            == 1
        )

    def test_straddling_repaint_after_publish_is_a_clean_noop(
        self, tmp_path: Path
    ) -> None:
        """The same straddling stroke again folds value-equal to the
        published state: a clean whole-batch no-op."""
        ex = _tiny_executor(tmp_path)
        _stub_local(ex, {})
        first = ex.execute([_paint(ex, window=STRADDLE_WINDOW, classes=1)])
        assert first.published
        before = ex.scene_revision
        again = ex.execute(
            [_paint(ex, window=STRADDLE_WINDOW, classes=1, edit_id="lc-2")]
        )
        assert again.status == "no-op"
        assert ex.scene_revision == before


# ---------------------------------------------------------------------------
# Repro (c): mixed vegetation + same-value repaint
# ---------------------------------------------------------------------------


class TestReproMixedVegSameValueRepaint:
    def test_veg_plus_same_value_repaint_publishes_one_job(
        self, tmp_path: Path
    ) -> None:
        """A same-value paint riding a vegetation batch neither refuses the
        batch nor drops the paint: one job covers the tree dirty windows
        AND the paint stroke (the plan's promise for both)."""
        ex = _tiny_executor(tmp_path)
        captured: dict = {}
        _stub_local(ex, captured)
        veg = ex.validate(
            _veg_command(revision=ex.scene_revision, edit_id="veg-1")
        )
        repaint = _paint(
            ex, window=ASPHALT_WINDOW, classes=1, edit_id="lc-1"
        )

        executed = ex.execute([veg, repaint])

        assert executed.published
        assert len(captured["jobs"]) == 1
        assert [tree.tree_id for tree in ex.layer.current_trees()] == ["t1"]
        assert any(
            _covers(w, ASPHALT_WINDOW) for w in captured["windows"]
        ), captured["windows"]
        # The accumulated overlay committed: the painted set is the stroke.
        # (Slice from the window constant itself — F2: the literal
        # [60:68, 20:28] pinned an unrelated asphalt region.)
        applied = ex._applied_landcover_overlay
        assert applied is not None
        resolved = applied.resolve(np.asarray(ex.cache.landcover))
        assert np.all(
            resolved[
                ASPHALT_WINDOW.row_start : ASPHALT_WINDOW.row_stop,
                ASPHALT_WINDOW.col_start : ASPHALT_WINDOW.col_stop,
            ]
            == 1
        )

    def test_veg_plus_repaint_of_published_values_publishes_one_job(
        self, tmp_path: Path
    ) -> None:
        """u-c6-review round-2 F1 (HIGH): a mixed vegetation batch whose paint
        leg is a same-value repaint of ALREADY-PUBLISHED cells.

        The repaint folds overlay-EQUAL to the published state (same painted
        set, same values), so the worker's ``landcover_pending`` gate is
        False — yet the executor staged the stroke as the batch's real dirty
        footprint and the PLAN promises it as a write window. Joining the
        dirty set only under ``landcover_pending`` dropped the stroke: the
        job published the vegetation windows alone and ``_reconcile_scope``
        refused the whole batch forever (every retry failed the same way).
        The stroke must join whenever the executor staged a non-empty
        footprint, overlay-equal or not."""
        ex = _tiny_executor(tmp_path)
        captured: dict = {}
        _stub_local(ex, captured)
        # Publish a real value change first (asphalt -> bare soil): the
        # applied overlay afterwards carries ASPHALT_WINDOW painted class 6.
        first = ex.execute(
            [_paint(ex, window=ASPHALT_WINDOW, classes=6, edit_id="lc-0")]
        )
        assert first.published
        assert ex._applied_landcover_overlay is not None
        captured["jobs"] = []
        captured["windows"] = ()

        veg = ex.validate(
            _veg_command(revision=ex.scene_revision, edit_id="veg-1")
        )
        repaint = _paint(
            # Same window, same class as published: overlay-equal fold.
            ex, window=ASPHALT_WINDOW, classes=6, edit_id="lc-1"
        )

        executed = ex.execute([veg, repaint])

        assert executed.published
        assert executed.mode == "local"
        assert len(captured["jobs"]) == 1  # one job for the whole batch
        assert any(
            _covers(w, ASPHALT_WINDOW) for w in captured["windows"]
        ), captured["windows"]
        assert ex.scene_revision == 2
        # The overlay-equal repaint still commits (painted set unchanged).
        assert ex._applied_landcover_overlay is not None

    def test_veg_plus_same_value_repaint_dirty_bounded_by_stroke(
        self, tmp_path: Path
    ) -> None:
        """Growth discipline under stroke-dirty: the same-value repaint's
        contribution to the batch's dirty footprint is exactly its STROKE
        (never prior published footprints, never the whole session)."""
        ex = _tiny_executor(tmp_path)
        captured: dict = {}
        _stub_local(ex, captured)
        # Publish a disjoint far-away paint first: its footprint must NOT
        # recompute when the mixed batch runs.
        first = ex.execute(
            [_paint(ex, window=ASPHALT_WINDOW_FAR, classes=6, edit_id="lc-0")]
        )
        assert first.published
        captured["windows"] = ()

        veg = ex.validate(
            _veg_command(revision=ex.scene_revision, edit_id="veg-1")
        )
        repaint = _paint(
            ex, window=ASPHALT_WINDOW, classes=1, edit_id="lc-2"
        )
        executed = ex.execute([veg, repaint])
        assert executed.published
        for window in captured["windows"]:
            assert not (
                window.row_stop > ASPHALT_WINDOW_FAR.row_start
                and window.row_start < ASPHALT_WINDOW_FAR.row_stop
                and window.col_stop > ASPHALT_WINDOW_FAR.col_start
                and window.col_start < ASPHALT_WINDOW_FAR.col_stop
            ), f"prior published footprint recomputed: {window}"
        assert any(
            _covers(w, ASPHALT_WINDOW) for w in captured["windows"]
        ), captured["windows"]


# ---------------------------------------------------------------------------
# Presence-vs-value: site-global payloads in mixed batches
# ---------------------------------------------------------------------------


class TestPresenceNotValue:
    def test_mixed_veg_and_value_equal_met_routes_full(
        self, tmp_path: Path
    ) -> None:
        """The MET analog (u-c3-era): a forcing fold that resolves
        value-equal to the published overlay still marks a MIXED batch's
        job pending — pending by PRESENCE of forcing deltas, not by value
        difference — so the job recomputes the forcing closure (the full
        tile) instead of routing local on the vegetation windows alone."""
        ex = _tiny_executor(tmp_path)
        captured: dict = {}
        _stub_full(ex, captured)
        first = ex.execute(
            [_executor_met_edit(ex, time_index=1, value=26.0)]
        )
        assert first.published

        # Same met value again (value-equal fold) + a vegetation edit: the
        # forcing overlay is PRESENT, so the job is pending full-tile work.
        met_again = _executor_met_edit(
            ex, time_index=1, value=26.0, edit_id="met-2"
        )
        veg = ex.validate(
            _veg_command(revision=ex.scene_revision, edit_id="veg-1")
        )
        executed = ex.execute([met_again, veg])
        assert executed.published
        assert executed.mode == "full"
        assert executed.write_windows == (ex.grid.full_window,)

    def test_met_only_value_equal_still_clean_noop(self, tmp_path: Path) -> None:
        """Whole-batch all-no-op still intercepts cleanly: a met-only batch
        whose fold equals the published overlay never dispatches."""
        ex = _tiny_executor(tmp_path)
        captured: dict = {}
        _stub_full(ex, captured)
        assert ex.execute(
            [_executor_met_edit(ex, time_index=1, value=26.0)]
        ).published
        before = ex.scene_revision
        again = ex.execute(
            [
                _executor_met_edit(
                    ex, time_index=1, value=26.0, edit_id="met-2"
                )
            ]
        )
        assert again.status == "no-op"
        assert not again.published
        assert ex.scene_revision == before
        assert len(captured["jobs"]) == 1

    def test_params_mixed_value_equal_publishes_full(self, tmp_path: Path) -> None:
        """Params audit (u-c7 watermark): a parameter fold that equals the
        published mapping in a mixed batch publishes the full tile — the
        params source is full-spatial, so the plan itself routes full; the
        dict fold compares BY VALUE, so no structural/value seam opens."""
        from tests.test_incremental_params_integration import _params_edit

        ex = _tiny_executor(tmp_path)
        captured: dict = {}
        _stub_full(ex, captured)
        first = ex.execute([_params_edit(ex, new_state={"albedo_b": 0.35})])
        assert first.published

        veg = ex.validate(
            _veg_command(revision=ex.scene_revision, edit_id="veg-1")
        )
        same_again = _params_edit(
            ex, new_state={"albedo_b": 0.35}, edit_id="params-2"
        )
        executed = ex.execute([veg, same_again])
        assert executed.published
        assert executed.mode == "full"

    def test_params_only_value_equal_clean_noop(self, tmp_path: Path) -> None:
        from tests.test_incremental_params_integration import _params_edit

        ex = _tiny_executor(tmp_path)
        _stub_full(ex, {})
        assert ex.execute(
            [_params_edit(ex, new_state={"albedo_b": 0.35})]
        ).published
        before = ex.scene_revision
        again = ex.execute(
            [
                _params_edit(
                    ex, new_state={"albedo_b": 0.35}, edit_id="params-2"
                )
            ]
        )
        assert again.status == "no-op"
        assert ex.scene_revision == before


# ---------------------------------------------------------------------------
# M2: the coalesce-equality boundary (docstring property + boundary test)
# ---------------------------------------------------------------------------


class TestCoalesceEqualityBoundary:
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

    def test_resolve_equal_but_different_painted_sets_fold_different(self):
        """The M2 boundary: resolve-equality does NOT imply fold-equality.
        One overlay paints a cell to its BASELINE value (the cell stays in
        the painted set at the baseline class); the other leaves it
        unpainted. Both resolve to the same grid, but the canonical folds
        differ — the fold describes the paint STATE (the painted set at
        final values), not the resolved values alone."""
        baseline = self._baseline()
        window = RasterWindow(4, 8, 4, 8)
        # baseline[4:8, 4:8] == 1; painting 1 keeps the painted set.
        painted_to_baseline = LandCoverOverlay(
            patches=(self._patch(window, 1),)
        )
        never_painted = LandCoverOverlay(patches=())

        assert np.array_equal(
            painted_to_baseline.resolve(baseline),
            never_painted.resolve(baseline),
        )
        assert (
            coalesce_overlay(painted_to_baseline, baseline)
            != coalesce_overlay(never_painted, baseline)
        )

    def test_same_painted_set_same_resolved_values_fold_equal(self):
        """The property that DOES hold (the corrected docstring claim):
        same painted set + same resolved values fold to the same canonical
        patch tuple, no matter the stroke history that produced them."""
        baseline = self._baseline()
        one_stroke = LandCoverOverlay(
            patches=(self._patch(RasterWindow(4, 8, 4, 8), 2),)
        )
        many_strokes = LandCoverOverlay(
            patches=(
                self._patch(RasterWindow(4, 8, 4, 8), 6),
                self._patch(RasterWindow(4, 8, 4, 8), 2),  # later stroke wins
            )
        )
        assert coalesce_overlay(one_stroke, baseline) == coalesce_overlay(
            many_strokes, baseline
        )
        # And the differing-value case still folds different (the painted
        # set alone is not equality either).
        other_value = LandCoverOverlay(
            patches=(self._patch(RasterWindow(4, 8, 4, 8), 6),)
        )
        assert coalesce_overlay(one_stroke, baseline) != coalesce_overlay(
            other_value, baseline
        )


# ---------------------------------------------------------------------------
# L3: the worker-side no-op error distinguishes the two readings
# ---------------------------------------------------------------------------


class TestNoopErrorText:
    def test_worker_noop_names_both_readings(self, tmp_path, monkeypatch) -> None:
        """The loud path (u-c1b wedge probes) keeps naming the wedge
        suspects AND now distinguishes the value-noop reading from them."""
        from tests.test_incremental_remediation_uc1b import (
            _fail_store_publish,
            _stub_full,
        )

        from solweig_gpu.incremental.executor import ExecutorError

        ex = _tiny_executor(tmp_path)
        _stub_full(ex)
        # Wedge the worker exactly like the u-c1b M1 probe: a post-publish
        # store failure with the rollback hook disabled leaves the worker's
        # forcing watermark ahead of the executor state; the retry then
        # finds nothing pending and must fail LOUDLY.
        ex._worker.rollback_pending = lambda watermarks: None
        _fail_store_publish(monkeypatch, ex)
        with pytest.raises(Exception, match="store rejected"):
            ex.execute([_executor_met_edit(ex, time_index=1, value=26.0)])
        monkeypatch.undo()
        with pytest.raises(ExecutorError) as info:
            ex.execute(
                [
                    _executor_met_edit(
                        ex, time_index=1, value=26.0, edit_id="met-2"
                    )
                ]
            )
        message = str(info.value)
        assert "nothing pending" in message
        # The wedge suspects are still named ...
        assert "double execution" in message
        assert "mismatched transaction" in message
        # ... and the value-noop reading is distinguished from them.
        assert "already" in message


# ---------------------------------------------------------------------------
# Growth property preserved under stroke-dirty
# ---------------------------------------------------------------------------


class TestGrowthPreservedUnderStrokeDirty:
    def test_stroke_dirty_never_recomputes_prior_footprints(
        self, tmp_path: Path
    ) -> None:
        """The u-c6 growth win survives the superset fix: only the CURRENT
        stroke dirties, so a paint-only batch never recomputes footprints
        published by earlier batches — including when the current stroke
        carries same-value cells."""
        ex = _tiny_executor(tmp_path)
        captured: dict = {}
        _stub_local(ex, captured)
        # Two published paints (A straddles already-target cells, B is a
        # clean value change) ...
        assert ex.execute(
            [_paint(ex, window=STRADDLE_WINDOW, classes=1)]
        ).published
        assert ex.execute(
            [
                _paint(
                    ex,
                    window=ASPHALT_WINDOW,
                    classes=6,  # asphalt -> bare soil: a real value change
                    edit_id="lc-2",
                )
            ]
        ).published

        # ... then a THIRD paint elsewhere: its dirty footprint is its own
        # stroke only. Neither prior footprint recomputes.
        third_window = ASPHALT_WINDOW_FAR
        captured["windows"] = ()
        third = ex.execute(
            [_paint(ex, window=third_window, classes=2, edit_id="lc-3")]
        )
        assert third.published
        for prior in (STRADDLE_WINDOW, ASPHALT_WINDOW):
            for window in captured["windows"]:
                assert not (
                    window.row_stop > prior.row_start
                    and window.row_start < prior.row_stop
                    and window.col_stop > prior.col_start
                    and window.col_start < prior.col_stop
                ), f"prior footprint {prior} recomputed: {window}"

    def test_patch_count_bound_under_alternating_straddling_repaints(
        self, tmp_path: Path
    ) -> None:
        """N repaints of the straddling stroke to alternating classes: the
        accumulated overlay stays the canonical decomposition of ONE
        stroke's painted set — bounded by edited area, not stroke count."""
        ex = _tiny_executor(tmp_path)
        _stub_local(ex, {})
        for index, classes in enumerate((1, 6, 1, 6)):
            executed = ex.execute(
                [
                    _paint(
                        ex,
                        window=STRADDLE_WINDOW,
                        classes=classes,
                        edit_id=f"lc-{index}",
                    )
                ]
            )
            assert executed.published
            applied = ex._applied_landcover_overlay
            assert applied is not None
            # The straddling stroke is one canonical patch (uniform value).
            assert len(applied.patches) == 1, (
                "accumulated overlay grew with repaint count "
                f"(index {index}: {len(applied.patches)} patches)"
            )
