# SPDX-License-Identifier: GPL-3.0-only
"""U-C packet 4 tests: land-cover overlay into solver/worker/executor (E2E).

Closes the chain the packet names: committed ``LandCoverPaintDelta`` edits
reach the physics.

1. **The dropped-overlay hazard.** ``solve_window`` used to read only the
   BASELINE cache class grid and ``run_full_tile`` copied the baseline
   ``Landcover`` GeoTIFF into its scratch site — a scenario paint would be
   silently dropped on both paths. Now the overlay materializes through
   ``resolve_landcover_overlay`` (copy-on-write against the full baseline
   grid, masked re-check that refuses undeclared-cell tampers, and a loud
   refusal on sites whose cache carries no landcover raster at all), the
   windowed solve feeds the resolved slice to ``run_utci_window`` as the
   float32 ``landcover_grid`` tensor (halo cells outside the write window
   see painted cells exactly like the oracle), and the full-tile path
   stages the RESOLVED raster in its scratch site instead of copying the
   baseline file.
2. **The worker watermark.** ``stage_landcover_overlay`` mirrors the u-c3
   forcing discipline: a staged overlay that differs from the last
   PUBLISHED one marks the job pending (a paint-only batch has no tree
   edits to coalesce), its patch footprints join the dirty windows (a
   paint is windowed — not a full-tile recompute), publication consumes
   it, and superseded/failed jobs keep it pending. The drop-overlay reset
   recomputes the previously painted footprint.
3. **The executor seam.** ``_stage_overlays`` folds committed paint deltas
   (accumulated across published batches, patches concatenated in arrival
   order — last-write-wins per cell at apply time), re-runs the fence
   through ``landcover_overlay_from_deltas``, materializes the resolved
   grid fail-fast before any state advances, and stages it on the worker;
   the accumulated overlay commits only on publication. Mixed batches (lc
   + vegetation) run as ONE plan with both overlays active.

Fast suite (unmarked): overlay-dropped-proof through the NEW solver and
worker paths, resolved-grid identity (what the physics reads is bitwise
``overlay.resolve(baseline)``), scratch materialization (legacy copy
byte-identical, overlay path lossless), worker job lifecycle for
paint-only batches, executor accumulation across batches, last-write-wins
folding, the water fence end-to-end, and rollback.

``scientific`` suite: a real-SVF 128x128 site where a committed paint
published through the executor is compared BITWISE against the ground-truth
oracle — the same paint written physically into a twin site's Landcover
raster (and a fresh cache built from it) — plus changed/unchanged cells
against an un-edited baseline full run, and the mixed-batch differential
(tree add + paint in one plan vs a twin site carrying BOTH physical edits).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from osgeo import gdal

from solweig_gpu.incremental.adapters.landcover import (
    ADAPTER_ID as LC_ADAPTER_ID,
    LandCoverOverlay,
    LandCoverSurfaceAdapter,
    landcover_overlay_from_deltas,
)
from solweig_gpu.incremental.edit_registry import (
    AdapterSchemaError,
    LANDCOVER_SOURCE_NODE,
)
from solweig_gpu.incremental.edit_types import (
    EditCommand,
    LandCoverPaintDelta,
    LandCoverPaintPatch,
    SiteContext,
    SourceDeltaError,
    ValidatedEdit,
)
from solweig_gpu.incremental.executor import PlanExecutor
from solweig_gpu.incremental.geometry import RasterGrid, RasterWindow
from solweig_gpu.incremental.planner import PlanningError
from solweig_gpu.incremental.result import ResultPatch, load_patch
from solweig_gpu.incremental.solver import (
    SolverInputError,
    load_site_forcing,
    resolve_landcover_overlay,
    run_full_tile,
    solve_window,
)
from solweig_gpu.incremental import solver as solver_mod
from solweig_gpu.incremental.trees import TreeLayer
from solweig_gpu.incremental.worker import ExactWorker, WorkerError

from tests.test_incremental_worker import (
    DATE_STR,
    LOCAL_ALWAYS,
    TINY_ORIGIN,
    TINY_PIXEL,
    _build_cache,
    _compute_baseline_svf,
    _make_prepared_site,
    _make_tiny_site,
)

gdal.UseExceptions()

#: Paint footprint inside the tiny/scientific grass region (class 5) of
#: ``_scene_arrays`` (rows/cols 0.30-0.40 of the grid).
PAINT_WINDOW = RasterWindow(40, 48, 40, 48)
#: Second paint footprint inside the bare-soil region (class 6).
PAINT_WINDOW_B = RasterWindow(110, 116, 103, 109)

SCI_ROWS = SCI_COLS = 128
SCI_PIXEL = 2.0
SCI_ORIGIN = (300000.0, 4100000.0)
SCI_EPSG = 32616
SCI_VARIABLES = ("utci", "tmrt", "shadow")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _paint_state(window: RasterWindow, classes) -> dict:
    return {
        "window": {
            "row_start": window.row_start,
            "row_stop": window.row_stop,
            "col_start": window.col_start,
            "col_stop": window.col_stop,
        },
        "classes": classes,
    }


def _paint_command(
    *,
    revision: int = 0,
    scenario: str = "default",
    window: RasterWindow = PAINT_WINDOW,
    classes=1,
    edit_id: str = "lc-1",
    times: tuple[int, ...] = (1,),
    outputs: tuple[str, ...] = ("utci",),
) -> EditCommand:
    return EditCommand(
        edit_id=edit_id,
        scenario_id=scenario,
        base_scene_revision=revision,
        adapter_id=LC_ADAPTER_ID,
        operation="paint",
        old_state=None,
        new_state=_paint_state(window, classes),
        requested_outputs=outputs,
        requested_times=times,
    )


def _lc_overlay(
    cache, window: RasterWindow = PAINT_WINDOW, classes=1
) -> LandCoverOverlay:
    """An adapter-validated overlay carrying one paint stroke."""
    context = SiteContext(
        site_id=cache.site_id,
        grid=RasterGrid(cache.rows, cache.cols, cache.pixel_size_m),
        scene_revision=0,
        available_times=tuple(range(cache.time_steps)),
    )
    validated = LandCoverSurfaceAdapter().validate(
        _paint_command(window=window, classes=classes), context
    )
    return landcover_overlay_from_deltas([validated.delta])


def _covers(outer: RasterWindow, inner: RasterWindow) -> bool:
    return (
        outer.row_start <= inner.row_start
        and outer.row_stop >= inner.row_stop
        and outer.col_start <= inner.col_start
        and outer.col_stop >= inner.col_stop
    )


def _stub_patch(
    variables: tuple[str, ...], job_id: str, revision: int, window: RasterWindow, *,
    mode: str = "full",
) -> ResultPatch:
    arrays = {
        name: np.full((3, window.height, window.width), 1.0, dtype=np.float32)
        for name in variables
    }
    return ResultPatch(
        job_id=job_id,
        scene_revision=revision,
        mode=mode,
        write_window=window,
        read_window=window,
        site_id="tiny",
        tile_key="0_0",
        cache_manifest_sha256="0" * 64,
        model_version="test",
        variables=variables,
        arrays=arrays,
        time_start=0,
        time_stop=3,
    )


def _fake_local_patches(job_id: str, revision: int, write_windows) -> list[ResultPatch]:
    """Mirror the worker's local contract: one patch per write window,
    each with its own store id (disjoint footprints coexist in one job)."""
    return [
        _stub_patch(("utci",), f"{job_id}-w{index}", revision, window, mode="local")
        for index, window in enumerate(write_windows)
    ]


def _tiny_worker(tmp_path: Path):
    grid, site = _make_tiny_site(tmp_path / "site")
    cache = _build_cache(
        site,
        tmp_path / "cache",
        met_path=site / "metfiles" / f"metfile_0_0_{DATE_STR}.txt",
        site_id="tiny",
    )
    layer = TreeLayer(cache.tree_base, grid)
    worker = ExactWorker(
        cache,
        layer,
        site_dir=site,
        results_root=tmp_path / "results",
        selected_date_str=DATE_STR,
        influence_config=LOCAL_ALWAYS,
    )
    return worker, site, grid


def _tiny_executor(tmp_path: Path) -> PlanExecutor:
    grid, site = _make_tiny_site(tmp_path / "site")
    cache = _build_cache(
        site,
        tmp_path / "cache",
        met_path=site / "metfiles" / f"metfile_0_0_{DATE_STR}.txt",
        site_id="tiny",
    )
    return PlanExecutor(
        cache=cache,
        layer=TreeLayer(cache.tree_base, grid),
        site_dir=site,
        results_root=tmp_path / "results",
        selected_date_str=DATE_STR,
        influence_config=LOCAL_ALWAYS,
    )


def _nolc_site(tmp_path: Path):
    """A twin tiny site whose Landcover raster (and cache entry) is absent."""
    grid, site = _make_tiny_site(tmp_path / "site")
    nolc = shutil.copytree(site, tmp_path / "nolc_site")
    shutil.rmtree(nolc / "Landcover")
    cache = _build_cache(
        nolc,
        tmp_path / "nolc_cache",
        met_path=nolc / "metfiles" / f"metfile_0_0_{DATE_STR}.txt",
        site_id="nolc",
    )
    return grid, nolc, cache


class _TamperedOverlay(LandCoverOverlay):
    """Bypass: resolve() writes an UNDECLARED cell (corner of the grid)."""

    def resolve(self, baseline):
        grid = super().resolve(baseline)
        grid[0, 0] = 6
        return grid


class _FakeRunUtciWindow:
    """Capture run_utci_window kwargs; return write-shaped zeros."""

    def __init__(self, forcing):
        self._forcing = forcing
        self.captured: dict = {}

    def __call__(self, **kwargs):
        self.captured.update(kwargs)
        rows = kwargs["out_window"][1] - kwargs["out_window"][0]
        cols = kwargs["out_window"][3] - kwargs["out_window"][2]
        return {
            name: np.zeros((self._forcing.time_steps, rows, cols), dtype=np.float32)
            for name in kwargs["requested_variables"]
        }


# ---------------------------------------------------------------------------
# Solver: the grid actually read (fast; tiny site, physics stubbed)
# ---------------------------------------------------------------------------


class TestSolverMaterialization:
    def test_solve_window_reads_resolved_grid_identity(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        # What the windowed physics READS is the resolved class grid —
        # solve_window consumes the overlay and never falls back to the
        # baseline cache read (an overlay cannot be dropped there).
        worker, site, grid = _tiny_worker(tmp_path)
        overlay = _lc_overlay(worker.cache)
        worker.stage_landcover_overlay(overlay)
        forcing = worker.forcing()
        baseline = np.asarray(worker.cache.landcover)
        cache_before = np.array(baseline)

        fake = _FakeRunUtciWindow(forcing)
        monkeypatch.setattr(solver_mod, "run_utci_window", fake)
        write = RasterWindow(32, 96, 32, 96)
        solve_window(
            worker.cache,
            worker.layer,
            read_window=grid.full_window,
            write_window=write,
            forcing=forcing,
            requested_variables=("utci",),
            landcover_overlay=worker._landcover_overlay,
        )
        # Resolved-grid identity: bitwise overlay.resolve(baseline), cast
        # to the float32 tensor the physics consumes.
        assert np.array_equal(
            fake.captured["landcover_grid"].numpy(),
            overlay.resolve(baseline).astype(np.float32),
        )
        # Declared cells only: everything else is the untouched baseline.
        mask = overlay.changed_cell_mask(baseline.shape)
        resolved = overlay.resolve(baseline)
        assert np.array_equal(resolved[~mask], baseline[~mask])
        assert set(np.unique(resolved[40:48, 40:48])) == {1}
        # Copy-on-write: the cache-backed array is never mutated.
        assert np.array_equal(np.asarray(worker.cache.landcover), cache_before)
        # Temporal discipline: stateful surface replay from timestep 0.
        assert fake.captured["time_start"] == 0
        assert fake.captured["time_stop"] is None

    def test_read_halo_cells_see_paint_outside_write_window(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        # The read window is halo-inflated beyond the write window; painted
        # cells OUTSIDE the write window but INSIDE the read window must
        # reach the physics (the GVF walk integrates them), exactly like
        # the full-domain oracle.
        worker, site, grid = _tiny_worker(tmp_path)
        paint_far = RasterWindow(100, 108, 100, 108)  # outside every window
        overlay = _lc_overlay(worker.cache, window=paint_far)
        worker.stage_landcover_overlay(overlay)
        forcing = worker.forcing()

        fake = _FakeRunUtciWindow(forcing)
        monkeypatch.setattr(solver_mod, "run_utci_window", fake)
        write = RasterWindow(20, 28, 20, 28)  # disjoint from paint_far
        solve_window(
            worker.cache,
            worker.layer,
            read_window=grid.full_window,
            write_window=write,
            forcing=forcing,
            requested_variables=("utci",),
            landcover_overlay=worker._landcover_overlay,
        )
        grid_read = fake.captured["landcover_grid"].numpy()
        assert set(np.unique(grid_read[100:108, 100:108])) == {1}

    def test_overlay_none_is_the_legacy_bitwise_read(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        worker, site, grid = _tiny_worker(tmp_path)
        forcing = worker.forcing()
        fake = _FakeRunUtciWindow(forcing)
        monkeypatch.setattr(solver_mod, "run_utci_window", fake)
        solve_window(
            worker.cache,
            worker.layer,
            read_window=grid.full_window,
            write_window=RasterWindow(32, 96, 32, 96),
            forcing=forcing,
            requested_variables=("utci",),
        )
        assert np.array_equal(
            fake.captured["landcover_grid"].numpy(),
            np.array(
                worker.cache.window("landcover", grid.full_window),
                dtype=np.float32,
            ),
        )

    def test_tampered_and_foreign_overlays_refused(
        self, tmp_path: Path
    ) -> None:
        worker, site, grid = _tiny_worker(tmp_path)
        overlay = _lc_overlay(worker.cache)
        # A tampered resolve writing UNDECLARED cells is refused by the
        # masked re-check.
        with pytest.raises(SolverInputError, match="declared painted cells"):
            resolve_landcover_overlay(
                worker.cache, _TamperedOverlay(patches=overlay.patches)
            )
        # Type fence: only LandCoverOverlay rides the seam.
        with pytest.raises(SolverInputError, match="LandCoverOverlay"):
            resolve_landcover_overlay(worker.cache, {"classes": 1})
        # Off-grid patch windows are refused by the adapter's resolve.
        offgrid = LandCoverOverlay(
            patches=(
                LandCoverPaintPatch(
                    window=RasterWindow(120, 140, 10, 20),
                    before_classes=(-1,) * 200,
                    after_classes=(1,) * 200,
                ),
            )
        )
        with pytest.raises(SourceDeltaError, match="outside the baseline"):
            resolve_landcover_overlay(worker.cache, offgrid)

    def test_missing_landcover_cache_refused_on_both_paths(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        # Overlay-dropped-proof: a site whose cache carries no landcover
        # raster would run the physics with landcover_grid=None and drop
        # the paint — both solver paths refuse loudly instead.
        _grid, nolc, cache = _nolc_site(tmp_path)
        assert "landcover" not in cache
        layer = TreeLayer(cache.tree_base, _grid)
        forcing = load_site_forcing(
            cache, site_dir=nolc, selected_date_str=DATE_STR
        )
        overlay = _lc_overlay(cache)
        with pytest.raises(SolverInputError, match="no landcover raster"):
            resolve_landcover_overlay(cache, overlay)
        monkeypatch.setattr(
            solver_mod, "run_utci_window", _FakeRunUtciWindow(forcing)
        )
        with pytest.raises(SolverInputError, match="no landcover raster"):
            solve_window(
                cache,
                layer,
                read_window=_grid.full_window,
                write_window=RasterWindow(32, 64, 32, 64),
                forcing=forcing,
                requested_variables=("utci",),
                landcover_overlay=overlay,
            )
        monkeypatch.setattr(solver_mod, "compute_utci", lambda **kwargs: None)
        with pytest.raises(SolverInputError, match="no landcover raster"):
            run_full_tile(
                cache,
                layer,
                forcing=forcing,
                site_dir=nolc,
                scratch_dir=tmp_path / "scratch",
                requested_variables=("utci",),
                landcover_overlay=overlay,
            )

    def test_run_full_tile_scratch_stages_resolved_raster(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        worker, site, grid = _tiny_worker(tmp_path)
        overlay = _lc_overlay(worker.cache)
        forcing = load_site_forcing(
            worker.cache, site_dir=site, selected_date_str=DATE_STR
        )
        captured: dict = {}

        def fake_compute_utci(**kwargs):
            captured.update(kwargs)
            return None

        monkeypatch.setattr(solver_mod, "compute_utci", fake_compute_utci)
        scratch = tmp_path / "scratch"
        # fake_compute_utci writes no outputs: the reader refuses loudly.
        with pytest.raises(SolverInputError, match="did not produce"):
            run_full_tile(
                worker.cache,
                worker.layer,
                forcing=forcing,
                site_dir=site,
                scratch_dir=scratch,
                requested_variables=("utci",),
                landcover_overlay=overlay,
            )
        # The physics consumed the scratch raster ...
        staged = scratch / "Landcover" / "Landcover_0_0.tif"
        assert captured["landcover_path"] == str(staged)
        # ... and it re-reads bitwise equal to the resolved grid the
        # windowed path materializes (float32 class codes are exact).
        dataset = gdal.Open(str(staged))
        staged_grid = dataset.GetRasterBand(1).ReadAsArray().astype(np.float32)
        dataset = None
        assert np.array_equal(
            staged_grid, overlay.resolve(np.asarray(worker.cache.landcover))
        )
        # The baseline site raster itself stays untouched (COW staging).
        source = site / "Landcover" / "Landcover_0_0.tif"
        baseline_dataset = gdal.Open(str(source))
        source_grid = baseline_dataset.GetRasterBand(1).ReadAsArray()
        baseline_dataset = None
        assert not np.array_equal(staged_grid.astype(np.int64), source_grid)

    def test_run_full_tile_legacy_scratch_is_byte_identical_copy(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        worker, site, grid = _tiny_worker(tmp_path)
        monkeypatch.setattr(solver_mod, "compute_utci", lambda **kwargs: None)
        forcing = load_site_forcing(
            worker.cache, site_dir=site, selected_date_str=DATE_STR
        )
        scratch = tmp_path / "scratch"
        with pytest.raises(SolverInputError):
            run_full_tile(
                worker.cache,
                worker.layer,
                forcing=forcing,
                site_dir=site,
                scratch_dir=scratch,
                requested_variables=("utci",),
            )
        staged = scratch / "Landcover" / "Landcover_0_0.tif"
        assert (
            staged.read_bytes()
            == (site / "Landcover" / "Landcover_0_0.tif").read_bytes()
        )


# ---------------------------------------------------------------------------
# Worker: overlay threading (fast; tiny site, solvers stubbed)
# ---------------------------------------------------------------------------


class TestWorkerOverlayThreading:
    def test_stage_fence_rejects_wrong_type_missing_raster_offgrid(
        self, tmp_path: Path
    ) -> None:
        worker, site, grid = _tiny_worker(tmp_path)
        with pytest.raises(WorkerError, match="LandCoverOverlay"):
            worker.stage_landcover_overlay({"classes": 1})
        # A cache without a landcover raster can never stage a paint.
        _grid, nolc, nolc_cache = _nolc_site(tmp_path)
        nolc_worker = ExactWorker(
            nolc_cache,
            TreeLayer(nolc_cache.tree_base, _grid),
            site_dir=nolc,
            results_root=tmp_path / "results_nolc",
            selected_date_str=DATE_STR,
        )
        with pytest.raises(WorkerError, match="no landcover raster"):
            nolc_worker.stage_landcover_overlay(_lc_overlay(nolc_cache))
        # Off-grid patch windows are refused at staging (they become the
        # job's dirty windows).
        offgrid = LandCoverOverlay(
            patches=(
                LandCoverPaintPatch(
                    window=RasterWindow(120, 140, 10, 20),
                    before_classes=(-1,) * 200,
                    after_classes=(1,) * 200,
                ),
            )
        )
        with pytest.raises(WorkerError, match="outside the site grid"):
            worker.stage_landcover_overlay(offgrid)

    def test_paint_only_job_routes_local_and_consumes(self, tmp_path: Path) -> None:
        worker, site, grid = _tiny_worker(tmp_path)
        captured: dict = {}

        def fake_local(job_id, revision, forcing, write_windows, **kwargs):
            captured["windows"] = tuple(write_windows)
            captured["overlay"] = worker._landcover_overlay
            return _fake_local_patches(job_id, revision, write_windows)

        worker._solve_local = fake_local
        overlay = _lc_overlay(worker.cache)
        worker.stage_landcover_overlay(overlay)

        outcome = worker.run()
        assert outcome.published
        assert outcome.mode == "local"  # a paint is windowed, not site-global
        assert outcome.diagnostics["landcover_overlay_pending"] is True
        assert outcome.diagnostics["batch_sequences"] is None  # no tree batch
        # The write window covers the paint footprint plus the GVF margin.
        assert _covers(captured["windows"][0], PAINT_WINDOW)
        assert captured["windows"][0].area > PAINT_WINDOW.area
        assert captured["overlay"] is overlay
        # Publication consumed the staged overlay: a re-run without a new
        # stage is a no-op (the overlay is never re-published silently).
        assert worker.run().status == "no-op"
        # Drop-overlay reset: staging None re-arms the job over the
        # previously painted footprint (baseline restore semantics).
        worker.stage_landcover_overlay(None)
        second = worker.run()
        assert second.published
        assert _covers(captured["windows"][0], PAINT_WINDOW)
        assert worker.run().status == "no-op"  # both watermarks consumed

    def test_paint_job_superseded_keeps_pending(self, tmp_path: Path) -> None:
        worker, site, grid = _tiny_worker(tmp_path)
        worker._solve_local = (
            lambda job_id, revision, forcing, write_windows, **kwargs: (
                _fake_local_patches(job_id, revision, write_windows)
            )
        )
        worker.stage_landcover_overlay(_lc_overlay(worker.cache))
        # Revision moves mid-flight: the job publishes nothing, and the
        # staged overlay STAYS pending (it was never published).
        worker.bump_scene_revision()
        outcome = worker.run(target_revision=0)
        assert outcome.status == "superseded"
        assert not list((tmp_path / "results").rglob("rev-*"))
        assert worker.run().published  # retry at the live revision

    def test_paint_windows_merge_with_tree_dirty_windows(
        self, tmp_path: Path
    ) -> None:
        # A mixed pending batch (tree edit + staged paint) unions the two
        # dirty sources into one job's write windows.
        from solweig_gpu.incremental.trees import TreeSpec

        worker, site, grid = _tiny_worker(tmp_path)
        worker.layer.add_tree(
            TreeSpec(
                "t1",
                TINY_ORIGIN[0] + 60.5 * TINY_PIXEL,
                TINY_ORIGIN[1] - 60.5 * TINY_PIXEL,
                4.0,
                2.0,
            )
        )
        worker.stage_landcover_overlay(_lc_overlay(worker.cache))
        captured: dict = {}

        def fake_local(job_id, revision, forcing, write_windows, **kwargs):
            captured["windows"] = tuple(write_windows)
            return _fake_local_patches(job_id, revision, write_windows)

        worker._solve_local = fake_local
        outcome = worker.run()
        assert outcome.published
        assert any(_covers(w, PAINT_WINDOW) for w in captured["windows"])
        # The tree edit's window is covered too (its own dirty window).
        tree_cell_row = 60
        assert any(
            w.row_start <= tree_cell_row < w.row_stop
            and w.col_start <= tree_cell_row < w.col_stop
            for w in captured["windows"]
        )


# ---------------------------------------------------------------------------
# Executor: lc batches, accumulation, mixed batches, rollback (fast)
# ---------------------------------------------------------------------------


class TestExecutorLandcoverIntegration:
    def test_lc_batch_publishes_local_and_records_store(self, tmp_path: Path) -> None:
        ex = _tiny_executor(tmp_path)
        captured: dict = {}

        def fake_local(job_id, revision, forcing, write_windows, **kwargs):
            captured["windows"] = tuple(write_windows)
            captured["overlay"] = ex._worker._landcover_overlay
            return _fake_local_patches(job_id, revision, write_windows)

        ex._worker._solve_local = fake_local
        executed = ex.execute([ex.validate(_paint_command())])
        assert executed.published
        assert executed.mode == "local"  # windowed paint, not site-global
        assert executed.scene_revision == 1
        assert _covers(captured["windows"][0], PAINT_WINDOW)
        assert captured["overlay"] is ex._worker._landcover_overlay
        # The temporal store records every timestep at the publication
        # revision (the local solve replays 0..T-1 under the overlay).
        assert ex.store.available_times("utci") == (0, 1, 2)
        assert ex.store.revision_at("utci", 1) == 1
        assert ex.store.lookup("utci", 1).mode == "local"
        # The accumulated scenario overlay committed on publication.
        assert ex._applied_landcover_overlay is not None

    def test_overlay_accumulates_across_batches(self, tmp_path: Path) -> None:
        """The dropped-overlay regression: a later batch (even a vegetation
        batch) must materialize the class grid under the ACCUMULATED overlay
        — never silently revert an earlier paint to the baseline."""
        ex = _tiny_executor(tmp_path)
        captured: dict = {}

        def fake_local(job_id, revision, forcing, write_windows, **kwargs):
            captured["windows"] = tuple(write_windows)
            captured["overlay"] = ex._worker._landcover_overlay
            return _fake_local_patches(job_id, revision, write_windows)

        ex._worker._solve_local = fake_local
        baseline = np.asarray(ex.cache.landcover)

        first = ex.execute([ex.validate(_paint_command())])  # A -> asphalt
        assert first.published
        assert np.all(
            captured["overlay"].resolve(baseline)[40:48, 40:48] == 1
        )

        # Vegetation batch (windowed): the local solve still runs under the
        # accumulated paint.
        veg = ex.validate(
            EditCommand(
                edit_id="veg-1",
                scenario_id=ex._scenario_id,
                base_scene_revision=ex.scene_revision,
                adapter_id="vegetation_geometry",
                operation="add",
                old_state=None,
                new_state={
                    "tree_id": "t1",
                    "x_m": TINY_ORIGIN[0] + 80.5 * TINY_PIXEL,
                    "y_m": TINY_ORIGIN[1] - 80.5 * TINY_PIXEL,
                    "height_m": 4.0,
                    "canopy_radius_m": 2.0,
                },
                requested_outputs=("utci",),
                requested_times=(1,),
            )
        )
        second = ex.execute([veg])
        assert second.published
        assert second.mode == "local"
        assert np.all(
            captured["overlay"].resolve(baseline)[40:48, 40:48] == 1  # kept
        )

        # A second paint folds ON TOP (patch concatenation; a later stroke
        # wins per cell at apply time): both regions live in the third
        # batch's overlay.
        third = ex.execute(
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
        assert third.published
        resolved = captured["overlay"].resolve(baseline)
        assert np.all(resolved[40:48, 40:48] == 1)  # earlier paint survives
        assert np.all(resolved[110:116, 103:109] == 2)  # new paint applied

    def test_mixed_lc_and_vegetation_batch_runs_one_job(self, tmp_path: Path) -> None:
        ex = _tiny_executor(tmp_path)
        captured: dict = {}

        def fake_local(job_id, revision, forcing, write_windows, **kwargs):
            captured.setdefault("jobs", []).append(job_id)
            captured["windows"] = tuple(write_windows)
            captured["overlay"] = ex._worker._landcover_overlay
            return _fake_local_patches(job_id, revision, write_windows)

        ex._worker._solve_local = fake_local
        veg = ex.validate(
            EditCommand(
                edit_id="veg-1",
                scenario_id=ex._scenario_id,
                base_scene_revision=0,
                adapter_id="vegetation_geometry",
                operation="add",
                old_state=None,
                new_state={
                    "tree_id": "t1",
                    "x_m": TINY_ORIGIN[0] + 40.5 * TINY_PIXEL,
                    "y_m": TINY_ORIGIN[1] - 40.5 * TINY_PIXEL,
                    "height_m": 4.0,
                    "canopy_radius_m": 2.0,
                },
                requested_outputs=("utci",),
                requested_times=(1,),
            )
        )
        lc = ex.validate(_paint_command())
        executed = ex.execute([veg, lc])
        # ONE job carries both edits: the tree replayed into the layer AND
        # the paint staged as the scenario overlay.
        assert executed.published
        assert len(captured["jobs"]) == 1
        assert [tree.tree_id for tree in ex.layer.current_trees()] == ["t1"]
        assert captured["overlay"] is ex._worker._landcover_overlay
        assert any(_covers(w, PAINT_WINDOW) for w in captured["windows"])
        assert ex._applied_landcover_overlay is not None

    def test_water_fence_refused_end_to_end(self, tmp_path: Path) -> None:
        # Water (7) is fenced on EVERY path; the executor's validate seam
        # refuses the batch before anything is staged or published.
        ex = _tiny_executor(tmp_path)
        for bad in (7, 0, 3, 8, -1):
            with pytest.raises(AdapterSchemaError):
                ex.validate(_paint_command(classes=bad))
        with pytest.raises(AdapterSchemaError, match="water"):
            ex.validate(_paint_command(classes=7))
        # Hidden inside a sequence payload, same refusal.
        with pytest.raises(AdapterSchemaError, match="water"):
            ex.validate(_paint_command(classes=(7,) * PAINT_WINDOW.area))
        # End-to-end through execute(): even a hand-built ValidatedEdit that
        # bypassed adapter validation is refused at the overlay seam (the
        # fence re-runs on every construction), whole-batch, before any
        # state advances.
        smuggled = ValidatedEdit(
            command=_paint_command(classes=7),
            adapter_id=LC_ADAPTER_ID,
            schema_version=LandCoverSurfaceAdapter.schema_version,
            source_node_id=LANDCOVER_SOURCE_NODE,
            delta=LandCoverPaintDelta(
                source_node_id=LANDCOVER_SOURCE_NODE,
                adapter_id=LC_ADAPTER_ID,
                patches=(
                    LandCoverPaintPatch(
                        window=PAINT_WINDOW,
                        before_classes=(-1,) * PAINT_WINDOW.area,
                        after_classes=(7,) * PAINT_WINDOW.area,
                    ),
                ),
            ),
        )
        with pytest.raises(
            (AdapterSchemaError, SourceDeltaError, PlanningError), match="water"
        ):
            ex.execute([smuggled])
        assert ex.scene_revision == 0
        assert ex._applied_landcover_overlay is None
        assert not list(ex.results_root.rglob("rev-*"))

    def test_executor_failure_rolls_back_overlay_and_revision(
        self, tmp_path: Path
    ) -> None:
        ex = _tiny_executor(tmp_path)

        def exploding(job_id, revision, forcing, write_windows, **kwargs):
            raise RuntimeError("boom")

        ex._worker._solve_local = exploding
        with pytest.raises(RuntimeError, match="boom"):
            ex.execute([ex.validate(_paint_command())])
        assert ex.scene_revision == 0
        assert ex._applied_landcover_overlay is None
        assert not list(ex.results_root.rglob("rev-*"))
        # Retry on a healthy worker publishes and commits the overlay.
        ex._worker._solve_local = (
            lambda job_id, revision, forcing, write_windows, **kwargs: (
                _fake_local_patches(job_id, revision, write_windows)
            )
        )
        executed = ex.execute(
            [ex.validate(_paint_command(edit_id="lc-2"))]
        )
        assert executed.published
        assert ex._applied_landcover_overlay is not None

    def test_stage_overlays_seam_folds_last_write_wins(self, tmp_path: Path) -> None:
        ex = _tiny_executor(tmp_path)
        baseline = np.asarray(ex.cache.landcover)
        first = ex.validate(_paint_command(classes=1))
        later = ex.validate(
            _paint_command(classes=5, edit_id="lc-2")
        )  # repaint same window back to grass
        # Within one batch, arrival order wins per cell.
        overlay = ex._stage_overlays(
            {"landcover": (first.delta, later.delta)}
        )
        assert np.all(overlay.resolve(baseline)[40:48, 40:48] == 5)
        # Foreign deltas are refused at the seam (never trusted).
        foreign = LandCoverPaintDelta(
            "landcover", "some_other_adapter", later.delta.patches
        )
        with pytest.raises(SourceDeltaError, match="some_other_adapter"):
            ex._stage_overlays({"landcover": (foreign,)})
        # Without committed paint deltas the seam is the accumulated state.
        assert ex._stage_overlays({}) is None

    def test_lc_batch_on_landcoverless_site_refused(self, tmp_path: Path) -> None:
        _grid, nolc, cache = _nolc_site(tmp_path)
        ex = PlanExecutor(
            cache=cache,
            layer=TreeLayer(cache.tree_base, _grid),
            site_dir=nolc,
            results_root=tmp_path / "results",
            selected_date_str=DATE_STR,
            influence_config=LOCAL_ALWAYS,
        )
        with pytest.raises(SolverInputError, match="no landcover raster"):
            ex.execute([ex.validate(_paint_command())])
        assert ex.scene_revision == 0
        assert not list(ex.results_root.rglob("rev-*"))

    def test_value_equal_noop_path_leaves_worker_staging_clean(
        self, tmp_path: Path
    ) -> None:
        """u-e1 F3: the whole-batch value no-op returns AFTER the worker
        staging seams already ran (forcing presence + land-cover dirty
        override) but BEFORE the rollback discipline — a reused executor
        leaked ``_forcing_presence_pending=True`` and a stale
        ``_landcover_dirty_override``, so every later batch routed
        full-tile."""
        from tests.test_incremental_met_integration import _executor_met_edit

        ex = _tiny_executor(tmp_path)

        def fake_full(job_id, revision, forcing):
            return [
                _stub_patch(
                    ("utci",), job_id, revision, ex.grid.full_window, mode="full"
                )
            ]

        def fake_local(job_id, revision, forcing, write_windows, **kwargs):
            captured.setdefault("modes", []).append("local")
            return _fake_local_patches(job_id, revision, write_windows)

        captured: dict = {}
        ex._worker._solve_full = fake_full
        ex._worker._solve_local = fake_local

        # Mixed batch (met + paint) publishes once: the met presence rule
        # routes it full-tile and publication consumes the staged flags.
        first = ex.execute(
            [
                _executor_met_edit(ex, time_index=1, value=30.0, edit_id="met-1"),
                ex.validate(_paint_command(classes=1, edit_id="lc-1")),
            ]
        )
        assert first.published
        assert first.mode == "full"
        assert ex._worker._forcing_presence_pending is False

        # The SAME batch again folds value-equal to the published state:
        # a clean no-op. At base, the staging seams ran before the no-op
        # return and nobody rolled them back.
        repeat = ex.execute(
            [
                _executor_met_edit(ex, time_index=1, value=30.0, edit_id="met-2"),
                ex.validate(
                    _paint_command(
                        revision=ex.scene_revision, classes=1, edit_id="lc-2"
                    )
                ),
            ]
        )
        assert repeat.status == "no-op"
        assert ex._worker._forcing_presence_pending is False
        assert ex._worker._landcover_dirty_override is None

        # Behavioral scar at base: the leaked presence flag forced every
        # later batch full-tile. A fresh paint must route LOCAL again.
        follow = ex.execute(
            [
                ex.validate(
                    _paint_command(
                        revision=ex.scene_revision,
                        window=PAINT_WINDOW_B,
                        classes=2,
                        edit_id="lc-3",
                    )
                )
            ]
        )
        assert follow.published
        assert follow.mode == "local"
        assert captured["modes"] == ["local"]


# ---------------------------------------------------------------------------
# Scientific E2E: real physics, ground-truth differential
# ---------------------------------------------------------------------------


def _paint_landcover_raster(raster: Path, window: RasterWindow, after: int) -> None:
    """Physically paint ONE window of a Landcover raster in place."""
    dataset = gdal.Open(str(raster), gdal.GA_Update)
    band = dataset.GetRasterBand(1)
    grid = band.ReadAsArray()
    grid[window.row_start : window.row_stop, window.col_start : window.col_stop] = after
    band.WriteArray(grid)
    band.FlushCache()
    dataset.FlushCache()
    dataset = None


def _full_run(cache, site, grid, scratch: Path) -> dict[str, np.ndarray]:
    forcing = load_site_forcing(
        cache, site_dir=site, selected_date_str=DATE_STR
    )
    return run_full_tile(
        cache,
        TreeLayer(cache.tree_base, grid),
        forcing=forcing,
        site_dir=site,
        scratch_dir=scratch,
        requested_variables=SCI_VARIABLES,
    )


@pytest.fixture(scope="module")
def lc_site(tmp_path_factory):
    """One real-SVF site plus twins whose rasters are PHYSICALLY edited.

    The twins are the ground truth: the same edits written the pre-overlay
    way (edit the raster, rebuild the cache), run through the standard
    full-domain path. ``twin`` carries the paint; ``twin_dual`` carries the
    paint AND the added tree.
    """
    from solweig_gpu.incremental.trees import TreeSpec, rasterize_tree_patch

    root = tmp_path_factory.mktemp("lc_integration_site")
    base = TreeSpec(
        "b1",
        SCI_ORIGIN[0] + 20.5 * SCI_PIXEL,
        SCI_ORIGIN[1] - 20.5 * SCI_PIXEL,
        3.0,
        2.0,
    )
    grid, site = _make_prepared_site(
        root,
        rows=SCI_ROWS,
        cols=SCI_COLS,
        pixel=SCI_PIXEL,
        origin=SCI_ORIGIN,
        epsg=SCI_EPSG,
        base_trees=(base,),
        met_hours=range(10, 14),
    )
    _compute_baseline_svf(site)
    cache = _build_cache(
        site,
        root / "cache",
        met_path=site / "metfiles" / f"metfile_0_0_{DATE_STR}.txt",
        site_id="lc-int",
    )

    twin = shutil.copytree(site, root / "edited_site")
    _paint_landcover_raster(
        twin / "Landcover" / "Landcover_0_0.tif", PAINT_WINDOW, after=1
    )
    # A paint touches no geometry: the twin inherits the baseline SVF
    # untouched (the physically edited class raster is the only change).
    twin_cache = _build_cache(
        twin,
        root / "cache_edited",
        met_path=twin / "metfiles" / f"metfile_0_0_{DATE_STR}.txt",
        site_id="lc-int-edited",
    )

    twin_dual = shutil.copytree(site, root / "edited_dual_site")
    _paint_landcover_raster(
        twin_dual / "Landcover" / "Landcover_0_0.tif", PAINT_WINDOW, after=1
    )
    extra = TreeSpec(
        "t1",
        SCI_ORIGIN[0] + 44.5 * SCI_PIXEL,
        SCI_ORIGIN[1] - 44.5 * SCI_PIXEL,
        4.0,
        2.0,
    )
    trees_raster = twin_dual / "Trees" / "Trees_0_0.tif"
    dataset = gdal.Open(str(trees_raster), gdal.GA_Update)
    band = dataset.GetRasterBand(1)
    trees = band.ReadAsArray()
    canopy, _trunk = rasterize_tree_patch(extra, grid, grid.full_window)
    np.maximum(trees, canopy, out=trees)
    band.WriteArray(trees)
    band.FlushCache()
    dataset.FlushCache()
    dataset = None
    _compute_baseline_svf(twin_dual)
    twin_dual_cache = _build_cache(
        twin_dual,
        root / "cache_dual_edited",
        met_path=twin_dual / "metfiles" / f"metfile_0_0_{DATE_STR}.txt",
        site_id="lc-int-dual",
    )
    return SimpleNamespace(
        grid=grid,
        site=site,
        cache=cache,
        twin=twin,
        twin_cache=twin_cache,
        twin_dual=twin_dual,
        twin_dual_cache=twin_dual_cache,
        extra_tree=extra,
    )


def _executor_for(lc_site, tmp_path: Path) -> PlanExecutor:
    return PlanExecutor(
        cache=lc_site.cache,
        layer=TreeLayer(lc_site.cache.tree_base, lc_site.grid),
        site_dir=lc_site.site,
        results_root=tmp_path / "results",
        selected_date_str=DATE_STR,
        influence_config=LOCAL_ALWAYS,
    )


def _assert_patches_match(
    patch_paths, oracle: dict[str, np.ndarray], variables, *, note: str
) -> None:
    for path in patch_paths:
        patch = load_patch(path)
        window = patch.write_window
        for variable in variables:
            scenario = patch.variable(variable)
            expected = oracle[variable][
                :, window.row_start : window.row_stop, window.col_start : window.col_stop
            ]
            assert np.array_equal(scenario, expected, equal_nan=True), (
                f"{variable}: {note} differs from the physically edited "
                f"oracle inside write window {window}"
            )


@pytest.mark.scientific
class TestLandCoverEndToEndDifferential:
    def test_paint_matches_physically_edited_oracle_bitwise(
        self, tmp_path: Path, lc_site
    ) -> None:
        """Ground truth: the overlay-published products equal a full-domain
        run on a site whose Landcover RASTER was physically edited (and
        whose cache was rebuilt from it) — bitwise, every band, cell, and
        variable, inside the published write windows."""
        executor = _executor_for(lc_site, tmp_path)
        executed = executor.execute(
            [
                executor.validate(
                    _paint_command(
                        classes=1, times=(1,), outputs=SCI_VARIABLES
                    )
                )
            ]
        )
        assert executed.published
        assert executed.mode == "local"  # the paint window routes local
        assert executed.scene_revision == 1
        assert any(
            _covers(w, PAINT_WINDOW) for w in executed.write_windows
        )

        oracle = _full_run(
            lc_site.twin_cache,
            lc_site.twin,
            lc_site.grid,
            tmp_path / "oracle_scratch",
        )
        baseline = _full_run(
            lc_site.cache,
            lc_site.site,
            lc_site.grid,
            tmp_path / "baseline_scratch",
        )

        _assert_patches_match(
            executed.patch_paths,
            oracle,
            SCI_VARIABLES,
            note="paint overlay path",
        )

        # Physical plausibility and scope conservativeness. A paint is NOT
        # strictly per-cell: the changed surface properties radiatively
        # couple outward — on this site the physically edited twin differs
        # from the baseline over a halo of roughly one GVF march reach
        # (observed ~10 cells beyond the 8x8 footprint) — and the worker's
        # write margin (gvf_march_pixels + 1 = 12 cells) must CONTAIN that
        # reach: outside the published write windows the edited-scene
        # oracle equals the baseline bitwise, so the windowed recompute
        # neither misses nor invents influence (the trees suite's
        # conservativeness invariant, restated for paints).
        patch = load_patch(executed.patch_paths[0])
        window = patch.write_window
        write_union_mask = np.ones((SCI_ROWS, SCI_COLS), dtype=bool)
        for published in executed.write_windows:
            write_union_mask[
                published.row_start : published.row_stop,
                published.col_start : published.col_stop,
            ] = False
        for variable in SCI_VARIABLES:
            # NaN-stable change mask (NaN != NaN would flag every no-data
            # cell): a cell changed iff both finite and unequal, or exactly
            # one side is NaN.
            both_finite = np.isfinite(oracle[variable]) & np.isfinite(
                baseline[variable]
            )
            changed = np.zeros(oracle[variable].shape, dtype=bool)
            changed[both_finite] = (
                oracle[variable][both_finite] != baseline[variable][both_finite]
            )
            changed |= np.isnan(oracle[variable]) != np.isnan(baseline[variable])
            assert not changed[:, write_union_mask].any(), (
                f"{variable}: the physically edited scene moves cells "
                "outside the published write windows (the write margin "
                "does not contain the paint's radiative reach)"
            )
            reach_rows, reach_cols = np.where(changed.any(axis=0))
            max_abs = float(
                np.max(
                    np.abs(
                        np.where(
                            np.isfinite(oracle[variable])
                            & np.isfinite(baseline[variable]),
                            oracle[variable] - baseline[variable],
                            0.0,
                        )
                    )
                )
            )
            if reach_rows.size:
                print(
                    f"[lc-int] {variable}: changed-cell footprint rows "
                    f"{reach_rows.min()}..{reach_rows.max()} cols "
                    f"{reach_cols.min()}..{reach_cols.max()} (paint "
                    f"{PAINT_WINDOW.row_start}..{PAINT_WINDOW.row_stop - 1} / "
                    f"{PAINT_WINDOW.col_start}..{PAINT_WINDOW.col_stop - 1}); "
                    f"max|delta| = {max_abs:.6f}"
                )
            else:
                print(
                    f"[lc-int] {variable}: no changed cells "
                    f"(max|delta| = {max_abs:.6f})"
                )
        # Painted cells change (surface-property coupling through Tgmaps).
        paint_mask = np.zeros((SCI_ROWS, SCI_COLS), dtype=bool)
        paint_mask[
            PAINT_WINDOW.row_start : PAINT_WINDOW.row_stop,
            PAINT_WINDOW.col_start : PAINT_WINDOW.col_stop,
        ] = True
        for variable in ("utci", "tmrt"):
            scenario = patch.variable(variable)
            base = baseline[variable][
                :, window.row_start : window.row_stop,
                window.col_start : window.col_stop,
            ]
            painted_delta = float(
                np.max(
                    np.abs(
                        np.where(
                            np.isfinite(scenario) & np.isfinite(base),
                            scenario - base,
                            0.0,
                        )
                    )[:, paint_mask[window.row_start:window.row_stop,
                                    window.col_start:window.col_stop]]
                )
            )
            print(
                f"[lc-int] {variable}: max|delta| vs baseline over painted "
                f"cells = {painted_delta:.6f}"
            )
            assert painted_delta > 0.0, (
                f"{variable}: painting grass->asphalt must change the "
                "surface-property coupling (Tgmaps albedo/tsfc)"
            )
        # A paint touches no geometry: the shadow series never moves.
        assert np.array_equal(
            patch.variable("shadow"),
            baseline["shadow"][
                :, window.row_start : window.row_stop,
                window.col_start : window.col_stop,
            ],
            equal_nan=True,
        ), "a paint never touches geometry: shadow stays bitwise baseline"

    def test_mixed_lc_and_veg_matches_dual_edited_oracle_bitwise(
        self, tmp_path: Path, lc_site
    ) -> None:
        """Mixed batch E2E: a tree add and a paint in ONE plan publish ONE
        job (both overlays active) whose products equal a full-domain run
        on a twin site carrying BOTH physical edits — bitwise."""
        executor = _executor_for(lc_site, tmp_path)
        tree = lc_site.extra_tree
        veg = executor.validate(
            EditCommand(
                edit_id="veg-1",
                scenario_id=executor._scenario_id,
                base_scene_revision=0,
                adapter_id="vegetation_geometry",
                operation="add",
                old_state=None,
                new_state={
                    "tree_id": tree.tree_id,
                    "x_m": tree.x_m,
                    "y_m": tree.y_m,
                    "height_m": tree.height_m,
                    "canopy_radius_m": tree.canopy_radius_m,
                },
                requested_outputs=SCI_VARIABLES,
                requested_times=(1,),
            )
        )
        lc = executor.validate(
            _paint_command(classes=1, times=(1,), outputs=SCI_VARIABLES)
        )
        executed = executor.execute([veg, lc])
        assert executed.published
        assert executed.scene_revision == 1
        # The scenario layer carries the ADDED tree; the base tree lives in
        # the cache's baseline vegetation raster (tree_base), not the
        # layer's named edits.
        assert [t.tree_id for t in executor.layer.current_trees()] == ["t1"]
        assert executor._applied_landcover_overlay is not None

        oracle = _full_run(
            lc_site.twin_dual_cache,
            lc_site.twin_dual,
            lc_site.grid,
            tmp_path / "oracle_scratch",
        )
        _assert_patches_match(
            executed.patch_paths,
            oracle,
            SCI_VARIABLES,
            note="mixed (paint + tree) path",
        )
