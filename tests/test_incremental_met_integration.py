# SPDX-License-Identifier: GPL-3.0-only
"""U-C packet 3 tests: meteorology overlay into worker/executor (end-to-end).

Closes the two gaps the packet names:

1. **The dropped-overlay hazard.** ``run_full_tile`` and the worker paths
   used to materialize forcing from the baseline met text — a scenario
   overlay would be silently dropped. Now the overlay rides the job
   (:meth:`ExactWorker.stage_forcing_overlay`), the worker materializes the
   RESOLVED table through ``load_site_forcing(overlay=...)`` (masked
   re-check seam), and the full-tile path stages the resolved table in its
   scratch site instead of copying the baseline ``met_path``.
2. **The executor seam.** ``_scenario_forcing`` folds committed forcing
   deltas (accumulated across published batches, last-write-wins per
   (variable, time)) into the overlay and materializes it fail-fast; a
   forcing-only batch (no tree edits) routes to a full-tile recompute
   because forcing is site-global.

Fast suite (unmarked): overlay-dropped-proof through the NEW worker and
``run_full_tile`` paths (a hand-tampered overlay resolution is rejected by
the loader's masked re-check, sub-tolerance included), resolved-table
identity (what the solver reads is bitwise ``overlay.resolve(baseline)``),
scratch materialization (legacy copy byte-identical, overlay path lossless),
worker job lifecycle for forcing-only batches, executor accumulation across
batches, R1 union plan bounds, and rollback.

``scientific`` suite: a real-SVF 128x128 site where a committed met edit
published through the executor is compared BITWISE against the ground-truth
oracle — the same edit written physically into a twin site's met text file
(and a fresh cache built from it) — plus changed/unchanged bands against an
un-edited baseline full run, and the R1 union E2E (edit t=1, request
t=2,3 -> the union is recomputed and published under scene_revision+1).
"""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from solweig_gpu.incremental.edit_types import EditCommand, SpatialScope, TemporalScope
from solweig_gpu.incremental.executor import NotIntegratedError, PlanExecutor
from solweig_gpu.incremental.geometry import RasterWindow
from solweig_gpu.incremental.result import ResultPatch, load_patch
from solweig_gpu.incremental.solver import (
    SiteForcing,
    SolverInputError,
    load_site_forcing,
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
    TINY_ROWS,
    _build_cache,
    _compute_baseline_svf,
    _make_prepared_site,
    _make_tiny_site,
)
from tests.test_incremental_adapters_met_time import (
    ADAPTER_ID as MET_ADAPTER_ID,
    ForcingOverlay,
    MeteorologicalForcingAdapter,
    overlay_from_deltas,
)

SCI_ROWS = SCI_COLS = 128
SCI_PIXEL = 2.0
SCI_ORIGIN = (300000.0, 4100000.0)
SCI_EPSG = 32616
SCI_VARIABLES = ("utci", "tmrt", "shadow")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _baseline_table(site: Path) -> np.ndarray:
    met_path = site / "metfiles" / f"metfile_0_0_{DATE_STR}.txt"
    return np.loadtxt(met_path, skiprows=1, delimiter=" ")


def _met_overlay(cache, site: Path, *cells: tuple[str, int, float]) -> ForcingOverlay:
    """An adapter-validated overlay carrying ``(variable, time, value)`` cells."""
    from solweig_gpu.incremental.edit_types import ForcingChange, SiteContext
    from solweig_gpu.incremental.geometry import RasterGrid

    changes = []
    for variable, time_index, value in cells:
        command = EditCommand(
            edit_id=f"met-{variable}-{time_index}",
            scenario_id="default",
            base_scene_revision=0,
            adapter_id=MET_ADAPTER_ID,
            operation="update_time_row",
            old_state=None,
            new_state={
                "time_index": time_index,
                "values": {variable: value},
            },
            requested_outputs=("utci",),
            requested_times=(time_index,),
        )
        context = SiteContext(
            site_id=cache.site_id,
            grid=RasterGrid(cache.rows, cache.cols, cache.pixel_size_m),
            scene_revision=0,
            available_times=tuple(range(cache.time_steps)),
        )
        validated = MeteorologicalForcingAdapter().validate(command, context)
        changes.extend(validated.delta.changes)
    return overlay_from_deltas([_forcing_delta(changes)])


def _forcing_delta(changes):
    from solweig_gpu.incremental.edit_types import ForcingDelta

    return ForcingDelta(
        source_node_id="meteorology",
        adapter_id=MET_ADAPTER_ID,
        changes=tuple(changes),
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


def _executor_met_edit(
    executor: PlanExecutor,
    *,
    time_index: int,
    value: float,
    variable: str = "air_temperature",
    times: tuple[int, ...] | None = None,
    outputs: tuple[str, ...] = ("utci",),
    edit_id: str = "met-1",
):
    command = EditCommand(
        edit_id=edit_id,
        scenario_id=executor._scenario_id,
        base_scene_revision=executor.scene_revision,
        adapter_id=MET_ADAPTER_ID,
        operation="update_time_row",
        old_state=None,
        new_state={"time_index": time_index, "values": {variable: value}},
        requested_outputs=outputs,
        requested_times=(time_index,) if times is None else times,
    )
    return executor.validate(command)


class _TamperedOverlay(ForcingOverlay):
    """Bypass: resolve() writes an UNDECLARED cell (wind column)."""

    def resolve(self, baseline):
        table = super().resolve(baseline)
        table[:, 9] += 1.0
        return table


class _SubToleranceTamperedOverlay(ForcingOverlay):
    """Same bypass at 5e-4 (above rtol=1e-5/atol=1e-4, far below +1.0)."""

    def resolve(self, baseline):
        table = super().resolve(baseline)
        table[:, 9] += 5e-4
        return table


# ---------------------------------------------------------------------------
# Worker: overlay threading (fast; tiny site, solvers stubbed)
# ---------------------------------------------------------------------------


class TestWorkerOverlayThreading:
    def test_forcing_resolves_staged_overlay_bitwise(self, tmp_path: Path) -> None:
        worker, site, _grid = _tiny_worker(tmp_path)
        baseline = _baseline_table(site)
        cache_before = np.array(worker.cache.met)
        overlay = _met_overlay(worker.cache, site, ("air_temperature", 1, 26.0))

        worker.stage_forcing_overlay(overlay)
        forcing = worker.forcing()

        # Resolved-table identity: what the worker materializes is bitwise
        # overlay.resolve(baseline) — the declared cell only, everything
        # else the untouched baseline text.
        assert np.array_equal(forcing.met_table, overlay.resolve(baseline))
        assert forcing.met_table[1, 11] == 26.0
        mask = overlay.changed_cell_mask(baseline.shape)
        assert np.array_equal(forcing.met_table[~mask], baseline[~mask])
        # The forcing carries the overlay (the full path stages on it).
        assert forcing.overlay == overlay
        # Solar geometry is bitwise the baseline (fenced edit; the solar
        # cross-check was NOT relaxed for the overlay path).
        legacy = load_site_forcing(
            worker.cache, site_dir=site, selected_date_str=DATE_STR
        )
        assert np.array_equal(forcing.altitude, legacy.altitude)
        assert np.array_equal(forcing.azimuth, legacy.azimuth)
        # Copy-on-write: the cache-backed array is never mutated.
        assert np.array_equal(np.asarray(worker.cache.met), cache_before)
        # Cached materialization: the second call is the same object.
        assert worker.forcing() is forcing

    def test_forcing_rejects_tampered_overlay_through_worker(self, tmp_path: Path) -> None:
        # Overlay-dropped-proof through the NEW path: the worker's
        # materialization runs the loader's masked re-check, so an overlay
        # whose resolution writes UNDECLARED cells is refused — coarse
        # (+1.0) and sub-tolerance (5e-4) tampers alike.
        worker, site, _grid = _tiny_worker(tmp_path)
        overlay = _met_overlay(worker.cache, site, ("air_temperature", 1, 26.0))
        for tampered in (
            _TamperedOverlay(changes=overlay.changes),
            _SubToleranceTamperedOverlay(changes=overlay.changes),
        ):
            worker.stage_forcing_overlay(tampered)
            with pytest.raises(SolverInputError, match="outside the overlay"):
                worker.forcing()

    def test_stage_fence_and_baseline_reset(self, tmp_path: Path) -> None:
        worker, site, _grid = _tiny_worker(tmp_path)
        with pytest.raises(WorkerError, match="ForcingOverlay"):
            worker.stage_forcing_overlay({"air_temperature": 26.0})
        overlay = _met_overlay(worker.cache, site, ("air_temperature", 1, 26.0))
        worker.stage_forcing_overlay(overlay)
        worker.stage_forcing_overlay(None)  # drop-overlay reset
        forcing = worker.forcing()
        assert forcing.overlay is None
        assert np.array_equal(forcing.met_table, _baseline_table(site))

    def test_forcing_only_job_routes_full_and_consumes(self, tmp_path: Path) -> None:
        worker, site, grid = _tiny_worker(tmp_path)
        captured: dict = {}

        def fake_full(job_id, revision, forcing):
            captured["forcing"] = forcing
            return [
                _stub_patch(
                    ("utci",), job_id, revision, grid.full_window, mode="full"
                )
            ]

        worker._solve_full = fake_full
        worker.stage_forcing_overlay(
            _met_overlay(worker.cache, site, ("air_temperature", 1, 26.0))
        )

        outcome = worker.run()
        assert outcome.published
        assert outcome.mode == "full"  # forcing is site-global -> full tile
        assert outcome.write_windows == (grid.full_window,)
        assert outcome.diagnostics["forcing_overlay_pending"] is True
        assert outcome.diagnostics["batch_sequences"] is None  # no tree batch
        assert captured["forcing"].met_table[1, 11] == 26.0
        # Publication consumed the staged overlay: a re-run without a new
        # stage is a no-op (the overlay is never re-published silently).
        assert worker.run().status == "no-op"
        # A NEW overlay cell re-arms the job (accumulated state).
        worker.stage_forcing_overlay(
            _met_overlay(worker.cache, site, ("air_temperature", 1, 26.0), ("humidity", 2, 40.0))
        )
        second = worker.run()
        assert second.published
        assert captured["forcing"].met_table[2, 10] == 40.0

    def test_forcing_job_superseded_keeps_pending(self, tmp_path: Path) -> None:
        worker, site, grid = _tiny_worker(tmp_path)
        worker._solve_full = (
            lambda job_id, revision, forcing: [
                _stub_patch(("utci",), job_id, revision, grid.full_window)
            ]
        )
        worker.stage_forcing_overlay(
            _met_overlay(worker.cache, site, ("air_temperature", 1, 26.0))
        )
        # Revision moves mid-flight: the job publishes nothing, and the
        # staged overlay STAYS pending (it was never published).
        worker.bump_scene_revision()
        outcome = worker.run(target_revision=0)
        assert outcome.status == "superseded"
        assert not list((tmp_path / "results").rglob("rev-*"))
        assert worker.run().published  # retry at the live revision


# ---------------------------------------------------------------------------
# Solver: the table actually read + scratch materialization (fast)
# ---------------------------------------------------------------------------


class TestSolverMaterialization:
    def test_solve_window_reads_resolved_table_identity(self, tmp_path: Path, monkeypatch) -> None:
        # What the windowed solver READS is the resolved table itself —
        # solve_window consumes forcing.met_table and never falls back to
        # the baseline met_path (an overlay cannot be dropped there).
        worker, site, grid = _tiny_worker(tmp_path)
        overlay = _met_overlay(worker.cache, site, ("air_temperature", 1, 26.0))
        worker.stage_forcing_overlay(overlay)
        forcing = worker.forcing()

        captured: dict = {}

        def fake_run_utci_window(**kwargs):
            captured.update(kwargs)
            rows = kwargs["out_window"][1] - kwargs["out_window"][0]
            cols = kwargs["out_window"][3] - kwargs["out_window"][2]
            return {
                name: np.zeros((forcing.time_steps, rows, cols), dtype=np.float32)
                for name in kwargs["requested_variables"]
            }

        monkeypatch.setattr(solver_mod, "run_utci_window", fake_run_utci_window)
        write = RasterWindow(32, 96, 32, 96)
        outputs = solve_window(
            worker.cache,
            worker.layer,
            read_window=grid.full_window,
            write_window=write,
            forcing=forcing,
            requested_variables=("utci",),
        )
        assert set(outputs) == {"utci"}
        assert captured["met_file"] is forcing.met_table
        assert np.array_equal(
            captured["met_file"], overlay.resolve(_baseline_table(site))
        )
        # Temporal discipline: cold replay from timestep 0 over the whole
        # series (stateful ground heat — the R1/REPLAY semantics).
        assert captured["time_start"] == 0
        assert captured["time_stop"] is None

    def test_run_full_tile_scratch_stages_resolved_table(self, tmp_path: Path, monkeypatch) -> None:
        worker, site, grid = _tiny_worker(tmp_path)
        overlay = _met_overlay(worker.cache, site, ("air_temperature", 1, 26.0))
        forcing = load_site_forcing(
            worker.cache, site_dir=site, selected_date_str=DATE_STR, overlay=overlay
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
            )
        # The physics consumed the RESOLVED table ...
        assert np.array_equal(
            captured["met_file"], overlay.resolve(_baseline_table(site))
        )
        # ... and the scratch prepared site stages it, never the baseline
        # text at met_path (lossless %.17g round-trip).
        staged = scratch / "metfiles" / forcing.met_path.name
        assert staged.is_file()
        assert np.array_equal(
            np.loadtxt(staged, skiprows=1, delimiter=" "), forcing.met_table
        )

    def test_run_full_tile_legacy_scratch_is_byte_identical_copy(self, tmp_path: Path, monkeypatch) -> None:
        worker, site, grid = _tiny_worker(tmp_path)
        monkeypatch.setattr(
            solver_mod, "compute_utci", lambda **kwargs: None
        )
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
        staged = scratch / "metfiles" / forcing.met_path.name
        assert staged.read_bytes() == forcing.met_path.read_bytes()

    def test_forcing_with_overlay_kwarg(self, tmp_path: Path, monkeypatch) -> None:
        worker, site, grid = _tiny_worker(tmp_path)
        baseline_forcing = load_site_forcing(
            worker.cache, site_dir=site, selected_date_str=DATE_STR
        )
        overlay = _met_overlay(worker.cache, site, ("air_temperature", 1, 26.0))

        # None keeps the forcing as-is (legacy, byte-identical).
        assert (
            solver_mod._forcing_with_overlay(
                worker.cache,
                baseline_forcing,
                site_dir=site,
                overlay=None,
            )
            is baseline_forcing
        )
        # An overlay staged against a BASELINE forcing re-materializes
        # through the loader seam (masked re-check runs there).
        resolved = solver_mod._forcing_with_overlay(
            worker.cache,
            baseline_forcing,
            site_dir=site,
            overlay=overlay,
        )
        assert resolved.met_table[1, 11] == 26.0
        assert resolved.overlay == overlay
        # The forcing the seam already resolved is kept as-is ...
        assert (
            solver_mod._forcing_with_overlay(
                worker.cache, resolved, site_dir=site, overlay=overlay
            )
            is resolved
        )
        # ... but a DIFFERENT overlay on a resolved forcing is an ambiguity.
        other = _met_overlay(worker.cache, site, ("humidity", 2, 40.0))
        with pytest.raises(SolverInputError, match="different overlay kwarg"):
            solver_mod._forcing_with_overlay(
                worker.cache, resolved, site_dir=site, overlay=other
            )
        # A tampered overlay is refused by the seam's re-materialization.
        with pytest.raises(SolverInputError, match="outside the overlay"):
            solver_mod._forcing_with_overlay(
                worker.cache,
                baseline_forcing,
                site_dir=site,
                overlay=_TamperedOverlay(changes=overlay.changes),
            )


# ---------------------------------------------------------------------------
# Executor: met batches, accumulation, R1 union, rollback (fast)
# ---------------------------------------------------------------------------


class TestExecutorMetIntegration:
    def test_met_batch_publishes_full_and_records_store(self, tmp_path: Path) -> None:
        ex = _tiny_executor(tmp_path)
        captured: dict = {}

        def fake_full(job_id, revision, forcing):
            captured["forcing"] = forcing
            return [
                _stub_patch(
                    ("utci",), job_id, revision, ex.grid.full_window, mode="full"
                )
            ]

        ex._worker._solve_full = fake_full
        executed = ex.execute(
            [_executor_met_edit(ex, time_index=1, value=26.0)]
        )
        assert executed.published
        assert executed.mode == "full"
        assert executed.scene_revision == 1
        assert executed.write_windows == (ex.grid.full_window,)
        assert captured["forcing"].met_table[1, 11] == 26.0
        # The temporal store records every timestep at the publication
        # revision (the full solve replays 0..T-1 under the overlay).
        assert ex.store.available_times("utci") == (0, 1, 2)
        assert ex.store.revision_at("utci", 1) == 1
        assert ex.store.lookup("utci", 1).mode == "full"
        # The accumulated scenario overlay committed on publication.
        assert ex._applied_forcing_overlay is not None

    def test_overlay_accumulates_across_batches(self, tmp_path: Path) -> None:
        """The dropped-overlay regression: a later batch (even a vegetation
        batch) must materialize forcing under the ACCUMULATED overlay —
        never silently revert an earlier met edit to the baseline text."""
        ex = _tiny_executor(tmp_path)
        captured: dict = {}

        def fake_full(job_id, revision, forcing):
            captured["forcing"] = forcing
            return [
                _stub_patch(
                    ("utci",), job_id, revision, ex.grid.full_window, mode="full"
                )
            ]

        def fake_local(job_id, revision, forcing, write_windows, **kwargs):
            captured["forcing"] = forcing
            return [
                _stub_patch(
                    ("utci",), job_id, revision, write_windows[0], mode="local"
                )
            ]

        ex._worker._solve_full = fake_full
        ex._worker._solve_local = fake_local

        first = ex.execute([_executor_met_edit(ex, time_index=1, value=26.0)])
        assert first.published
        assert captured["forcing"].met_table[1, 11] == 26.0

        # Vegetation batch (windowed): the local solve still runs under the
        # accumulated overlay.
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
                    "x_m": TINY_ORIGIN[0] + 40.5 * TINY_PIXEL,
                    "y_m": TINY_ORIGIN[1] - 40.5 * TINY_PIXEL,
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
        assert captured["forcing"].met_table[1, 11] == 26.0  # kept

        # A later met edit folds ON TOP (last-write-wins per (variable,
        # time)): both cells live in the third batch's forcing. The edit is
        # utci_only (wind_speed) so the r3a fast path takes the batch and
        # refuses on the tmrt coverage this stubbed store cannot provide —
        # routing the ordinary worker path the stubs intercept. A
        # radiation-affecting edit would route the G2.1 warm sparse path,
        # whose real split solve no tiny-site stub can serve (the fake
        # 16-patch SVF cannot feed solve_window); the warm path's own
        # overlay chain is covered end-to-end on the prepared site by
        # tests/test_incremental_met_warm_path.py.
        third = ex.execute(
            [_executor_met_edit(ex, time_index=2, value=40.0, variable="wind_speed")]
        )
        assert third.published
        table = captured["forcing"].met_table
        assert table[1, 11] == 26.0  # earlier edit survives
        assert table[2, 9] == 40.0  # new edit applied

    def test_mixed_met_and_vegetation_batch_runs_one_full_job(self, tmp_path: Path) -> None:
        ex = _tiny_executor(tmp_path)
        captured: dict = {}

        def fake_full(job_id, revision, forcing):
            captured["forcing"] = forcing
            return [
                _stub_patch(
                    ("utci",), job_id, revision, ex.grid.full_window, mode="full"
                )
            ]

        ex._worker._solve_full = fake_full
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
        met = _executor_met_edit(ex, time_index=1, value=26.0)
        executed = ex.execute([veg, met])
        # Forcing's closure is FULL: the mixed batch publishes full tile in
        # ONE job, with the tree replayed AND the overlay materialized.
        assert executed.published
        assert executed.mode == "full"
        assert [tree.tree_id for tree in ex.layer.current_trees()] == ["t1"]
        assert captured["forcing"].met_table[1, 11] == 26.0

    def test_r1_union_plan_bounds_and_service(self, tmp_path: Path) -> None:
        # R1 (lead ruling 2026-09-02): edit t=1 and request t=2 -> the
        # recompute set is the UNION; the stateful REPLAY bound and the
        # time-varying bounds extend to its extremes; all served times
        # publish under scene_revision+1.
        ex = _tiny_executor(tmp_path)
        ex._worker._solve_full = (
            lambda job_id, revision, forcing: [
                _stub_patch(("utci",), job_id, revision, ex.grid.full_window)
            ]
        )
        edit = _executor_met_edit(ex, time_index=1, value=26.0, times=(1, 2))
        plan = ex.plan([edit])
        by_node = {impact.node_id: impact for impact in plan.node_impacts}
        assert by_node["surface_thermal_state"].temporal_scope is TemporalScope.REPLAY
        assert by_node["surface_thermal_state"].time_start == 0
        assert by_node["surface_thermal_state"].time_stop == 2  # union max
        assert by_node["utci"].time_start == 1  # union min
        assert by_node["utci"].time_stop == 2
        assert by_node["utci"].spatial_scope is SpatialScope.FULL

        executed = ex.execute([edit])
        assert executed.published
        assert executed.scene_revision == 1
        for time in (1, 2):  # the union is served at the new revision
            assert ex.store.revision_at("utci", time) == 1

    def test_executor_failure_rolls_back_overlay_and_revision(self, tmp_path: Path) -> None:
        ex = _tiny_executor(tmp_path)

        def exploding(job_id, revision, forcing):
            raise RuntimeError("boom")

        ex._worker._solve_full = exploding
        with pytest.raises(RuntimeError, match="boom"):
            ex.execute([_executor_met_edit(ex, time_index=1, value=26.0)])
        assert ex.scene_revision == 0
        assert ex._applied_forcing_overlay is None
        assert not list(ex.results_root.rglob("rev-*"))
        # Retry on a healthy worker publishes and commits the overlay.
        ex._worker._solve_full = (
            lambda job_id, revision, forcing: [
                _stub_patch(("utci",), job_id, revision, ex.grid.full_window)
            ]
        )
        executed = ex.execute(
            [_executor_met_edit(ex, time_index=1, value=26.0, edit_id="met-2")]
        )
        assert executed.published
        assert ex._applied_forcing_overlay is not None

    def test_scenario_forcing_seam_folds_last_write_wins(self, tmp_path: Path) -> None:
        ex = _tiny_executor(tmp_path)
        edit = _executor_met_edit(ex, time_index=1, value=26.0)
        committed = {"meteorology": (edit.delta,)}
        forcing = ex._scenario_forcing(committed)
        assert isinstance(forcing, SiteForcing)
        assert forcing.met_table[1, 11] == 26.0
        # A second committed delta on the SAME cell wins (arrival order).
        later = _executor_met_edit(ex, time_index=1, value=29.0, edit_id="met-2")
        folded = ex._scenario_forcing({"meteorology": (edit.delta, later.delta)})
        assert folded.met_table[1, 11] == 29.0
        # Without committed forcing deltas the seam is the baseline (the
        # accumulated overlay once one publishes — covered above).
        assert ex._scenario_forcing({}).overlay is None

    def test_selected_date_time_still_refused(self, tmp_path: Path) -> None:
        ex = _tiny_executor(tmp_path)
        edit = _executor_met_edit(ex, time_index=1, value=26.0)
        plan = ex.plan([edit])
        crafted = replace(plan, changed_sources=("selected_date_time",))
        with pytest.raises(NotIntegratedError, match="selected_date_time") as error:
            ex.execute_plan(crafted, [edit.delta])
        assert error.value.source_node == "selected_date_time"
        assert ex.scene_revision == 0


# ---------------------------------------------------------------------------
# Scientific E2E: real physics, ground-truth differential
# ---------------------------------------------------------------------------


def _edit_met_cell(met_path: Path, *, row: int, column: int, value: float) -> None:
    """Physically edit ONE cell of a met text file, preserving every other
    token's exact text (so the twin file parses bitwise-equal to the
    original everywhere except the edited cell)."""
    lines = met_path.read_text().splitlines()
    tokens = lines[row + 1].split()  # +1: header line
    tokens[column] = f"{value:.17g}"
    lines[row + 1] = " ".join(tokens)
    met_path.write_text("\n".join(lines) + "\n")


def _full_run(cache, site, grid, scratch: Path) -> dict[str, np.ndarray]:
    from solweig_gpu.incremental.solver import load_site_forcing, run_full_tile

    forcing = load_site_forcing(cache, site_dir=site, selected_date_str=DATE_STR)
    return run_full_tile(
        cache,
        TreeLayer(cache.tree_base, grid),
        forcing=forcing,
        site_dir=site,
        scratch_dir=scratch,
        requested_variables=SCI_VARIABLES,
    )


@pytest.fixture(scope="module")
def met_site(tmp_path_factory):
    """One real-SVF site plus a twin whose met file is PHYSICALLY edited.

    The twin is the ground truth: the same edit written the pre-overlay
    way (edit the text file, rebuild the cache), run through the standard
    full-domain path.
    """
    from solweig_gpu.incremental.trees import TreeSpec

    root = tmp_path_factory.mktemp("met_integration_site")
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
        site_id="met-int",
    )

    twin = shutil.copytree(site, root / "edited_site")
    met_path = twin / "metfiles" / f"metfile_0_0_{DATE_STR}.txt"
    _edit_met_cell(met_path, row=1, column=11, value=30.0)  # Ta at t=1
    twin_cache = _build_cache(
        twin,
        root / "cache_edited",
        met_path=met_path,
        site_id="met-int-edited",
    )
    return SimpleNamespace(
        grid=grid, site=site, cache=cache, twin=twin, twin_cache=twin_cache
    )


@pytest.mark.scientific
class TestMetEndToEndDifferential:
    def _execute_met_edit(self, met_site, tmp_path: Path, *, times):
        executor = PlanExecutor(
            cache=met_site.cache,
            layer=TreeLayer(met_site.cache.tree_base, met_site.grid),
            site_dir=met_site.site,
            results_root=tmp_path / "results",
            selected_date_str=DATE_STR,
            influence_config=LOCAL_ALWAYS,
        )
        executed = executor.execute(
            [
                _executor_met_edit(
                    executor,
                    time_index=1,
                    value=30.0,
                    times=times,
                    outputs=SCI_VARIABLES,
                )
            ]
        )
        assert executed.published
        assert executed.mode == "full"  # forcing is site-global
        assert executed.scene_revision == 1
        assert executed.write_windows == (met_site.grid.full_window,)
        return executor, executed, load_patch(executed.patch_paths[0])

    def test_met_edit_matches_physically_edited_oracle_bitwise(
        self, tmp_path: Path, met_site
    ) -> None:
        """Ground truth: the overlay-published products equal a full-domain
        run on a site whose met FILE was physically edited (and whose cache
        was rebuilt from it) — bitwise, every band, cell, and variable."""
        _executor, _executed, patch = self._execute_met_edit(
            met_site, tmp_path, times=(1,)
        )
        oracle = _full_run(
            met_site.twin_cache,
            met_site.twin,
            met_site.grid,
            tmp_path / "oracle_scratch",
        )
        baseline = _full_run(
            met_site.cache,
            met_site.site,
            met_site.grid,
            tmp_path / "baseline_scratch",
        )

        for variable in SCI_VARIABLES:
            scenario = patch.variable(variable)
            assert scenario.shape == oracle[variable].shape
            assert np.array_equal(scenario, oracle[variable], equal_nan=True), (
                f"{variable}: overlay path differs from the physically "
                "edited-met-file oracle"
            )

        # The committed edit changes the published products at the changed
        # time, and leaves every EARLIER time bitwise untouched (the state
        # carry runs forward from the edit, never backward). NB the oracle
        # tiles carry NaN water cells (~3.7% of utci) — identity claims use
        # equal_nan (a NaN equals itself bitwise).
        for variable in ("utci", "tmrt"):
            scenario = patch.variable(variable)
            assert np.array_equal(
                scenario[0], baseline[variable][0], equal_nan=True
            ), (
                f"{variable}: t=0 (before the edit) must stay bitwise "
                "identical to the baseline full run"
            )
            assert not np.array_equal(
                scenario[1], baseline[variable][1], equal_nan=True
            ), f"{variable}: t=1 (the edited timestep) must change"
            for band in range(1, scenario.shape[0]):
                both = np.isfinite(scenario[band]) & np.isfinite(
                    baseline[variable][band]
                )
                delta = float(
                    np.max(np.abs(np.where(both, scenario[band] - baseline[variable][band], 0.0)))
                )
                print(
                    f"[met-int] {variable} band t={band}: max|delta| vs "
                    f"baseline = {delta:.6f}"
                )

        # The registry fence pins every column that could move shadows:
        # an air-temperature edit leaves the shadow series bitwise the
        # baseline everywhere (and bitwise the oracle, above).
        assert np.array_equal(
            patch.variable("shadow"), baseline["shadow"], equal_nan=True
        )

    def test_r1_union_e2e_serves_requested_window(
        self, tmp_path: Path, met_site
    ) -> None:
        """R1 union E2E: edit t=1, request t=2,3 -> all three recomputed,
        published under scene_revision+1; the physics is bitwise the same
        overlay's plain edit (the requested window never changes values)."""
        executor, executed, patch = self._execute_met_edit(
            met_site, tmp_path, times=(1, 2, 3)
        )
        by_node = {impact.node_id: impact for impact in executed.plan.node_impacts}
        assert (
            by_node["surface_thermal_state"].temporal_scope
            is TemporalScope.REPLAY
        )
        assert by_node["surface_thermal_state"].time_stop == 3  # union max
        assert (by_node["utci"].time_start, by_node["utci"].time_stop) == (1, 3)

        # Every union time is served at the new revision (full-tile patch
        # carries the whole replayed series).
        for time in (1, 2, 3):
            assert executor.store.revision_at("utci", time) == 1
        assert executor.store.lookup("utci", 1).mode == "full"

        # Bitwise identity with the oracle twin (ground truth above).
        oracle = _full_run(
            met_site.twin_cache,
            met_site.twin,
            met_site.grid,
            tmp_path / "oracle_scratch",
        )
        for variable in SCI_VARIABLES:
            assert np.array_equal(
                patch.variable(variable), oracle[variable], equal_nan=True
            ), f"{variable}: R1 service run differs from the oracle"

        # Unchanged time before the edit stays bitwise the baseline.
        baseline = _full_run(
            met_site.cache,
            met_site.site,
            met_site.grid,
            tmp_path / "baseline_scratch",
        )
        for variable in SCI_VARIABLES:
            assert np.array_equal(
                patch.variable(variable)[0],
                baseline[variable][0],
                equal_nan=True,
            ), f"{variable}: t=0 must stay bitwise identical"
