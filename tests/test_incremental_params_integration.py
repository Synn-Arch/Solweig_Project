# SPDX-License-Identifier: GPL-3.0-only
"""U-C packet 7 tests: model-parameter executor wiring (end-to-end).

Closes the execution gap the packet names — ``model_parameters`` deltas
validated and planned (U-B) but never executed. Now the seam chain is
complete:

``ModelParameterDelta`` -> ``PlanExecutor._physics_kernel_arguments``
(:func:`kernel_arguments_from_deltas` fold over the ACCUMULATED published
parameter state, last-write-wins per name — the parameters are
time-invariant, so there is no time dimension) ->
:meth:`ExactWorker.stage_model_parameters` (the u-c3 watermark discipline,
mirrored: staged payload rides the job, publication consumes it, a failed
batch rewinds it) -> ``solve_window``/``run_full_tile`` ->
``run_utci_window(model_parameters=...)``. ``run_full_tile`` reroutes a
non-empty mapping through a faithful direct replication of the
``compute_utci`` core (the orchestrator itself has no parameter seam);
``None``/``{}`` keep the legacy ``compute_utci`` call byte-identical.

Fast suite (unmarked): worker staging fences (unknown name / non-mapping /
non-scalar refused at the staging door, payload frozen at stage time),
params-only job routing (site-global: full tile, watermark consumed on
publish, superseded keeps pending), solver forwarding (both solve paths
hand the staged mapping to ``run_utci_window``; ``None``/``{}`` never
leave the legacy path; the mapping type is fenced), executor accumulation
across batches (a later vegetation batch still runs under the folded
parameters), last-write-wins folding, repaint-to-published no-op, mixed
params+met and params+building batches, the blocked-name fence through
``execute_plan`` (hand-built delta, refused before any state advances),
and failure rollback.

``scientific`` suite: a real-SVF 64x64 site where a committed ``albedo_b``
edit published through the executor is compared BITWISE against the
independent oracle — a direct ``run_full_tile(model_parameters=...)`` run
— and against the legacy baseline through u-c2's decoupling table
(``albedo_b`` leaves lup/ldown/shadow bitwise identical while
utci/tmrt/kup/kdown change); the direct-path replication is pinned
bitwise against the legacy ``compute_utci`` run through an identity
override (``firstdayleaf=150`` stays inside the leaf-on window on the
DOY-172 fixture); accumulation and last-write-wins hold across TWO
published parameter batches; and a mixed params+met batch matches its
dual-edit oracle bitwise.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from solweig_gpu.incremental.edit_types import (
    EditCommand,
    ModelParameterChange,
    ModelParameterDelta,
    SourceDeltaError,
)
from solweig_gpu.incremental.executor import PlanExecutor
from solweig_gpu.incremental.result import load_patch
from solweig_gpu.incremental.solver import (
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
    _build_cache,
    _compute_baseline_svf,
    _make_prepared_site,
    _make_tiny_site,
)
from tests.test_incremental_adapters_model_params import ADAPTER_ID
from tests.test_incremental_met_integration import (
    _executor_met_edit,
    _met_overlay,
    _stub_patch,
)
from tests.test_incremental_params_physics import (
    EPSG,
    FIXTURE_TREE,
    MET_HOURS,
    ORIGIN,
    PIXEL,
    ROWS,
    COLS,
)

SCI_VARIABLES = ("utci", "tmrt", "shadow")
#: Every variable the direct path can carry (the legacy ``compute_utci``
#: read-back only ever produces the first three).
ORACLE_VARIABLES = (
    "utci", "tmrt", "kup", "kdown", "lup", "ldown", "shadow",
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _tiny_worker(tmp_path: Path):
    grid, site = _make_tiny_site(tmp_path / "site")
    cache = _build_cache(
        site,
        tmp_path / "cache",
        met_path=site / "metfiles" / f"metfile_0_0_{DATE_STR}.txt",
        site_id="tiny",
    )
    worker = ExactWorker(
        cache,
        TreeLayer(cache.tree_base, grid),
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


def _params_edit(
    executor: PlanExecutor,
    *,
    new_state: dict,
    old_state: dict | None = None,
    outputs: tuple[str, ...] = SCI_VARIABLES,
    edit_id: str = "params-1",
):
    command = EditCommand(
        edit_id=edit_id,
        scenario_id=executor._scenario_id,
        base_scene_revision=executor.scene_revision,
        adapter_id=ADAPTER_ID,
        operation="update",
        old_state=old_state,
        new_state=new_state,
        requested_outputs=outputs,
        requested_times=(1,),
    )
    return executor.validate(command)


def _veg_command(executor: PlanExecutor, *, edit_id: str = "veg-1"):
    command = EditCommand(
        edit_id=edit_id,
        scenario_id=executor._scenario_id,
        base_scene_revision=executor.scene_revision,
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
    return executor.validate(command)


def _stub_full(executor: PlanExecutor, captured: dict) -> None:
    def fake_full(job_id, revision, forcing):
        captured["params"] = executor._worker._model_parameters
        captured["forcing"] = forcing
        return [
            _stub_patch(
                ("utci", "tmrt", "shadow"),
                job_id,
                revision,
                executor.grid.full_window,
                mode="full",
            )
        ]

    executor._worker._solve_full = fake_full


# ---------------------------------------------------------------------------
# Worker: params threading (fast; tiny site, solvers stubbed)
# ---------------------------------------------------------------------------


class TestWorkerParamsThreading:
    def test_stage_fence_freezes_and_fences(self, tmp_path: Path) -> None:
        worker, _site, _grid = _tiny_worker(tmp_path)
        with pytest.raises(WorkerError, match="name -> value mapping"):
            worker.stage_model_parameters(["albedo_b"])
        with pytest.raises(WorkerError, match="unknown model parameters"):
            worker.stage_model_parameters({"bogus": 1.0})
        with pytest.raises(WorkerError, match="must be a scalar"):
            worker.stage_model_parameters({"albedo_b": [0.3]})
        # Fence names mirror the physics seam exactly.
        with pytest.raises(WorkerError, match="unknown model parameters"):
            worker.stage_model_parameters({"patch_option": 1})
        staged = {"albedo_b": 0.3, "cyl": False}
        worker.stage_model_parameters(staged)
        assert worker._model_parameters == {"albedo_b": 0.3, "cyl": False}
        # The payload is frozen at stage time: mutating the caller's
        # mapping afterwards cannot rewrite the job's physics.
        staged["albedo_b"] = 0.9
        staged["ewall"] = 0.1
        assert worker._model_parameters == {"albedo_b": 0.3, "cyl": False}
        # Dtype hygiene: np.float64 passes isinstance(float) but is coerced
        # to a plain double (bitwise-stable parameter differentials).
        worker.stage_model_parameters({"albedo_b": np.float64(0.35)})
        assert type(worker._model_parameters["albedo_b"]) is float
        # None restores the parameter-free baseline run.
        worker.stage_model_parameters(None)
        assert worker._model_parameters is None
        # An empty mapping is the parameter-free run too, never a pending
        # payload that would force a pointless full recompute.
        worker.stage_model_parameters({"albedo_b": 0.35})
        worker.stage_model_parameters({})
        assert worker._model_parameters is None

    def test_params_only_job_routes_full_and_consumes(self, tmp_path: Path) -> None:
        worker, _site, grid = _tiny_worker(tmp_path)
        captured: dict = {}
        worker._solve_full = (
            lambda job_id, revision, forcing: (
                captured.update(params=worker._model_parameters)
                or [
                    _stub_patch(
                        ("utci",), job_id, revision,
                        grid.full_window, mode="full",
                    )
                ]
            )
        )
        worker.stage_model_parameters({"albedo_b": 0.35})
        outcome = worker.run()
        assert outcome.published
        assert outcome.mode == "full"  # params are site-global -> full tile
        assert outcome.write_windows == (grid.full_window,)
        assert outcome.diagnostics["model_parameters_pending"] is True
        assert outcome.diagnostics["batch_sequences"] is None  # no tree batch
        assert captured["params"] == {"albedo_b": 0.35}
        # Publication consumed the staged payload: the watermark holds it
        # and a re-run without a new stage is a no-op.
        assert worker._published_model_parameters == {"albedo_b": 0.35}
        assert worker.run().status == "no-op"
        assert worker.publication_watermarks.model_parameters == {
            "albedo_b": 0.35
        }
        # A NEW payload re-arms the job (accumulated state).
        worker.stage_model_parameters({"albedo_b": 0.35, "ewall": 0.85})
        second = worker.run()
        assert second.published
        assert captured["params"] == {"albedo_b": 0.35, "ewall": 0.85}

    def test_params_job_superseded_keeps_pending(self, tmp_path: Path) -> None:
        worker, _site, grid = _tiny_worker(tmp_path)
        worker._solve_full = (
            lambda job_id, revision, forcing: [
                _stub_patch(("utci",), job_id, revision, grid.full_window)
            ]
        )
        worker.stage_model_parameters({"albedo_b": 0.35})
        # Revision moves mid-flight: the job publishes nothing, and the
        # staged payload STAYS pending (it was never published).
        worker.bump_scene_revision()
        outcome = worker.run(target_revision=0)
        assert outcome.status == "superseded"
        assert worker._published_model_parameters is None
        assert not list((tmp_path / "results").rglob("rev-*"))
        assert worker.run().published  # retry at the live revision

    def test_rollback_pending_rewinds_params_watermark(self, tmp_path: Path) -> None:
        worker, _site, _grid = _tiny_worker(tmp_path)
        worker.stage_model_parameters({"albedo_b": 0.35})
        worker._published_model_parameters = {"ewall": 0.85}
        snapshot = worker.publication_watermarks
        worker._published_model_parameters = {"bogus": "wedged"}
        worker.rollback_pending(snapshot)
        assert worker._published_model_parameters == {"ewall": 0.85}
        # Staging survives the rewind (only the PUBLISHED watermark rewinds).
        assert worker._model_parameters == {"albedo_b": 0.35}

    def test_solve_paths_forward_staged_params(self, tmp_path, monkeypatch) -> None:
        """Both solve paths hand the STAGED payload into the solver seam."""
        from solweig_gpu.incremental import worker as worker_mod

        worker, site, grid = _tiny_worker(tmp_path)
        captured: dict = {}

        def fake_solve_window(*args, **kwargs):
            captured["window_kwargs"] = kwargs
            window = kwargs["write_window"]
            return {
                name: np.zeros(
                    (worker.cache.time_steps, window.height, window.width),
                    dtype=np.float32,
                )
                for name in worker.requested_variables
            }

        def fake_run_full_tile(*args, **kwargs):
            captured["full_kwargs"] = kwargs
            arrays = {
                name: np.zeros(
                    (worker.cache.time_steps, grid.rows, grid.cols),
                    dtype=np.float32,
                )
                for name in worker.requested_variables
            }
            # G2.1: _solve_full always asks for the final thermal state.
            return arrays, {"next_step": worker.cache.time_steps}

        monkeypatch.setattr(worker_mod, "solve_window", fake_solve_window)
        monkeypatch.setattr(worker_mod, "run_full_tile", fake_run_full_tile)
        worker.stage_model_parameters({"albedo_b": 0.35})

        worker._solve_local("job", 0, worker.forcing(), (grid.full_window,))
        assert captured["window_kwargs"]["model_parameters"] == {"albedo_b": 0.35}
        worker._solve_full("job", 0, worker.forcing())
        assert captured["full_kwargs"]["model_parameters"] == {"albedo_b": 0.35}
        # Without a staged payload both paths forward None (legacy).
        worker.stage_model_parameters(None)
        worker._solve_full("job2", 0, worker.forcing())
        assert captured["full_kwargs"]["model_parameters"] is None


# ---------------------------------------------------------------------------
# Solver: forwarding + the direct-path branch (fast)
# ---------------------------------------------------------------------------


class TestSolverParamsMaterialization:
    def test_solve_window_forwards_model_parameters_verbatim(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        worker, _site, grid = _tiny_worker(tmp_path)
        forcing = worker.forcing()
        captured: dict = {}

        def fake_run_utci_window(**kwargs):
            captured.update(kwargs)
            rows = grid.rows
            cols = grid.cols
            return {
                name: np.zeros((forcing.time_steps, rows, cols), dtype=np.float32)
                for name in kwargs["requested_variables"]
            }

        monkeypatch.setattr(solver_mod, "run_utci_window", fake_run_utci_window)
        outputs = solve_window(
            worker.cache,
            worker.layer,
            read_window=grid.full_window,
            write_window=grid.full_window,
            forcing=forcing,
            requested_variables=("utci",),
        )
        assert set(outputs) == {"utci"}
        # Default None forwards None (the legacy byte-identical call).
        assert captured["model_parameters"] is None
        solve_window(
            worker.cache,
            worker.layer,
            read_window=grid.full_window,
            write_window=grid.full_window,
            forcing=forcing,
            requested_variables=("utci",),
            model_parameters={"albedo_b": 0.35, "cyl": False},
        )
        assert captured["model_parameters"] == {"albedo_b": 0.35, "cyl": False}

    def test_none_and_empty_keep_legacy_compute_utci(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        worker, site, _grid = _tiny_worker(tmp_path)
        forcing = worker.forcing()
        calls: list[dict] = []
        monkeypatch.setattr(
            solver_mod,
            "compute_utci",
            lambda **kwargs: calls.append(kwargs) or None,
        )
        def explode(**kwargs):
            raise AssertionError("run_utci_window must not run on the legacy path")

        monkeypatch.setattr(solver_mod, "run_utci_window", explode)
        for index, params in enumerate((None, {})):
            with pytest.raises(SolverInputError, match="did not produce"):
                run_full_tile(
                    worker.cache,
                    worker.layer,
                    forcing=forcing,
                    site_dir=site,
                    scratch_dir=tmp_path / f"scratch-{index}",
                    requested_variables=("utci",),
                    model_parameters=params,
                )
        assert len(calls) == 2  # both routed through compute_utci

    def test_params_run_direct_path_never_calls_compute_utci(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """A non-empty mapping reroutes through the direct replication:
        same scratch site, fresh svf_calculator, run_utci_window with the
        overrides — and compute_utci is never entered."""
        worker, site, _grid = _tiny_worker(tmp_path)
        forcing = worker.forcing()
        captured: dict = {}

        def explode(**kwargs):
            raise AssertionError("compute_utci must not run under parameters")

        def fake_svf(*args, **kwargs):
            captured["svf_args"] = args
            captured["svf_kwargs"] = kwargs
            rows, cols = worker.cache.rows, worker.cache.cols
            return tuple(
                torch_zeros(rows, cols) for _ in range(19)
            )

        def torch_zeros(rows, cols):
            import torch

            return torch.zeros((rows, cols))

        def fake_run_utci_window(**kwargs):
            captured.update(kwargs)
            return {
                name: np.zeros(
                    (forcing.time_steps, worker.cache.rows, worker.cache.cols),
                    dtype=np.float32,
                )
                for name in kwargs["requested_variables"]
            }

        monkeypatch.setattr(solver_mod, "compute_utci", explode)
        monkeypatch.setattr(solver_mod, "svf_calculator", fake_svf)
        monkeypatch.setattr(solver_mod, "run_utci_window", fake_run_utci_window)

        outputs = run_full_tile(
            worker.cache,
            worker.layer,
            forcing=forcing,
            site_dir=site,
            scratch_dir=tmp_path / "scratch",
            requested_variables=("utci", "kup", "shadow"),
            model_parameters={"albedo_b": 0.35},
        )
        # The direct path honors the full requested vocabulary ...
        assert set(outputs) == {"utci", "kup", "shadow"}
        # ... the physics core received the overrides verbatim ...
        assert captured["model_parameters"] == {"albedo_b": 0.35}
        assert captured["met_file"] is forcing.met_table
        assert captured["location"] == forcing.location
        # ... the SVF pass is the oracle's fresh patch-option-2 pass ...
        assert captured["svf_args"][0] == 2
        assert captured["svf_kwargs"]["save_rasters"] is False
        # ... and the land-cover grid is the cleaned scratch raster.
        assert captured["landcover_grid"] is not None
        assert captured["lc_class"] is not None

    def test_mapping_type_fenced(self, tmp_path: Path) -> None:
        worker, site, _grid = _tiny_worker(tmp_path)
        with pytest.raises(SolverInputError, match="name -> value mapping"):
            run_full_tile(
                worker.cache,
                worker.layer,
                forcing=worker.forcing(),
                site_dir=site,
                scratch_dir=tmp_path / "scratch",
                model_parameters=["albedo_b"],
            )


# ---------------------------------------------------------------------------
# Executor: params batches, accumulation, no-op, fences, rollback (fast)
# ---------------------------------------------------------------------------


class TestExecutorParamsIntegration:
    def test_params_batch_publishes_full_and_records_store(
        self, tmp_path: Path
    ) -> None:
        ex = _tiny_executor(tmp_path)
        captured: dict = {}
        _stub_full(ex, captured)
        executed = ex.execute(
            [_params_edit(ex, new_state={"albedo_b": 0.35})]
        )
        assert executed.published
        assert executed.mode == "full"  # params dirty set is FULL
        assert executed.scene_revision == 1
        assert executed.write_windows == (ex.grid.full_window,)
        # The fold is float()-coerced and the solve ran under it.
        assert captured["params"] == {"albedo_b": 0.35}
        assert type(captured["params"]["albedo_b"]) is float
        assert ex.store.available_times("utci") == (0, 1, 2)
        assert ex.store.revision_at("utci", 1) == 1
        assert ex.store.lookup("utci", 1).mode == "full"
        # The accumulated parameter state committed on publication.
        assert ex._applied_model_parameters == {"albedo_b": 0.35}
        assert ex._worker._published_model_parameters == {"albedo_b": 0.35}

    def test_params_accumulate_across_batches_last_write_wins(
        self, tmp_path: Path
    ) -> None:
        """The dropped-params regression: a later batch (even a windowed
        vegetation batch) must run under the ACCUMULATED parameters —
        never silently revert an earlier parameter edit."""
        ex = _tiny_executor(tmp_path)
        captured: dict = {}
        _stub_full(ex, captured)

        def fake_local(job_id, revision, forcing, write_windows, **kwargs):
            captured["params"] = ex._worker._model_parameters
            captured["forcing"] = forcing
            return [
                _stub_patch(
                    ("utci",), job_id, revision, write_windows[0], mode="local"
                )
            ]

        ex._worker._solve_local = fake_local

        first = ex.execute(
            [_params_edit(
                ex,
                new_state={"albedo_b": 0.3},
                old_state={"albedo_b": 0.2},
            )]
        )
        assert first.published
        assert ex._applied_model_parameters == {"albedo_b": 0.3}

        # Vegetation batch (windowed): the local solve still runs under the
        # accumulated parameters.
        second = ex.execute([_veg_command(ex)])
        assert second.published
        assert second.mode == "local"
        assert captured["params"] == {"albedo_b": 0.3}

        # A later params edit folds ON TOP (last-write-wins per name):
        # the override wins and the earlier edit's OTHER names survive.
        third = ex.execute(
            [_params_edit(
                ex,
                new_state={"albedo_b": 0.35, "ewall": 0.85},
                old_state={"albedo_b": 0.3, "ewall": 0.9},
                edit_id="params-2",
            )]
        )
        assert third.published
        assert captured["params"] == {"albedo_b": 0.35, "ewall": 0.85}
        assert ex._applied_model_parameters == {"albedo_b": 0.35, "ewall": 0.85}

    def test_repaint_of_published_params_is_noop(self, tmp_path: Path) -> None:
        ex = _tiny_executor(tmp_path)
        _stub_full(ex, {})
        first = ex.execute(
            [_params_edit(ex, new_state={"albedo_b": 0.35})]
        )
        assert first.published
        # Re-committing the SAME values folds to the published parameter
        # state: nothing any solve path could recompute differs.
        second = ex.execute(
            [_params_edit(ex, new_state={"albedo_b": 0.35}, edit_id="params-2")]
        )
        assert second.status == "no-op"
        assert second.mode is None
        assert ex.scene_revision == 1
        assert ex._applied_model_parameters == {"albedo_b": 0.35}

    def test_seam_fold_unit(self, tmp_path: Path) -> None:
        ex = _tiny_executor(tmp_path)
        # No committed params and no accumulated state: the baseline run.
        assert ex._physics_kernel_arguments({}) is None
        assert ex._physics_kernel_arguments({"vegetation_dsm": ()}) is None
        first = _params_edit(ex, new_state={"albedo_b": 0.3})
        second = _params_edit(
            ex, new_state={"albedo_b": 0.35}, edit_id="params-2"
        )
        committed = {"model_parameters": (first.delta, second.delta)}
        # Arrival order wins (the coalescer's last-write-wins semantics).
        assert ex._physics_kernel_arguments(committed) == {"albedo_b": 0.35}
        # Wrong-typed deltas are refused at the seam (fail-fast).
        from solweig_gpu.incremental.edit_types import ForcingDelta

        with pytest.raises(SourceDeltaError, match="ModelParameterDelta"):
            ex._physics_kernel_arguments(
                {"model_parameters": (ForcingDelta(
                    source_node_id="model_parameters",
                    adapter_id=ADAPTER_ID,
                ),)}
            )
        # Accumulated state is folded unchanged when the batch carries no
        # params deltas.
        ex._applied_model_parameters = {"ewall": 0.85}
        assert ex._physics_kernel_arguments({}) == {"ewall": 0.85}
        assert ex._physics_kernel_arguments(committed) == {
            "albedo_b": 0.35,
            "ewall": 0.85,
        }

    def test_mixed_params_and_met_batch_one_full_job(
        self, tmp_path: Path
    ) -> None:
        ex = _tiny_executor(tmp_path)
        captured: dict = {}
        _stub_full(ex, captured)
        param = _params_edit(ex, new_state={"albedo_b": 0.35})
        met = _executor_met_edit(ex, time_index=1, value=30.0)
        executed = ex.execute([param, met])
        # Both payloads are site-global: ONE full job carries the resolved
        # forcing AND the folded kernel arguments.
        assert executed.published
        assert executed.mode == "full"
        assert executed.scene_revision == 1
        assert captured["params"] == {"albedo_b": 0.35}
        assert captured["forcing"].met_table[1, 11] == 30.0
        assert captured["forcing"].overlay is not None
        assert ex._applied_model_parameters == {"albedo_b": 0.35}
        assert ex._applied_forcing_overlay == captured["forcing"].overlay

    def test_mixed_params_and_building_batch_threads_chain(
        self, tmp_path, monkeypatch
    ) -> None:
        from solweig_gpu.incremental import executor as executor_mod
        from solweig_gpu.incremental.buildings import BuildingRasterResult
        from solweig_gpu.incremental.regenerate import BuildingRegenerationResult

        ex = _tiny_executor(tmp_path)
        captured: dict = {}

        def fake_chain(**kwargs):
            captured.update(kwargs)
            zeros = np.zeros(
                (ex.cache.time_steps, ex.grid.rows, ex.grid.cols),
                dtype=np.float32,
            )
            return BuildingRegenerationResult(
                scenario_site_dir=ex.results_root / "default" / ".building" / "site",
                cache_dir=ex.results_root / "default" / ".building" / "cache",
                cache=ex.cache,
                raster=np.zeros((ex.grid.rows, ex.grid.cols), dtype=np.float32),
                records=BuildingRasterResult(
                    raster=np.zeros((ex.grid.rows, ex.grid.cols), dtype=np.float32),
                    records=(),
                ),
                outputs={"utci": zeros, "tmrt": zeros.copy(), "shadow": zeros.copy()},
                stages=("stage_scenario_site", "run_scenario_full_tile"),
            )

        monkeypatch.setattr(executor_mod, "regenerate_building_batch", fake_chain)
        building = ex.validate(
            EditCommand(
                edit_id="bld-1",
                scenario_id=ex._scenario_id,
                base_scene_revision=0,
                adapter_id="building_geometry",
                operation="add",
                old_state=None,
                new_state={
                    "building_id": "b1",
                    "footprint_m": (
                        (TINY_ORIGIN[0] + 10.0, TINY_ORIGIN[1] - 10.0),
                        (TINY_ORIGIN[0] + 14.0, TINY_ORIGIN[1] - 10.0),
                        (TINY_ORIGIN[0] + 14.0, TINY_ORIGIN[1] - 14.0),
                        (TINY_ORIGIN[0] + 10.0, TINY_ORIGIN[1] - 14.0),
                    ),
                    "height_m": 12.0,
                },
                requested_outputs=("utci",),
                requested_times=(1,),
            )
        )
        param = _params_edit(ex, new_state={"albedo_b": 0.35})
        executed = ex.execute([building, param])
        assert executed.published
        assert executed.mode == "full"
        # The chain received the folded kernel arguments alongside the
        # overlays — a mixed building+params batch never drops them.
        assert captured["model_parameters"] == {"albedo_b": 0.35}
        assert ex._applied_model_parameters == {"albedo_b": 0.35}
        # The worker adopted the publication (nothing left pending).
        assert (
            ex._worker.publication_watermarks.model_parameters
            == {"albedo_b": 0.35}
        )

    def test_blocked_param_refused_end_to_end(self, tmp_path: Path) -> None:
        """The fence, not adapter-only: a hand-built blocked delta fed
        through ``execute_plan`` is refused at the kernel-argument seam —
        before any state advances, nothing staged on the worker."""
        ex = _tiny_executor(tmp_path)
        edit = _params_edit(ex, new_state={"albedo_b": 0.35})
        plan = ex.plan([edit])
        blocked = ModelParameterDelta(
            source_node_id="model_parameters",
            adapter_id=ADAPTER_ID,
            parameters=(
                ModelParameterChange(
                    name="patch_option",
                    before_value=2,
                    after_value=1,
                ),
            ),
        )
        with pytest.raises(SourceDeltaError, match="BLOCKED"):
            ex.execute_plan(plan, [blocked])
        assert ex.scene_revision == 0
        assert ex._applied_model_parameters is None
        assert ex._worker._model_parameters is None
        assert not list(ex.results_root.rglob("rev-*"))

    def test_executor_failure_rolls_back_params_state(
        self, tmp_path: Path
    ) -> None:
        ex = _tiny_executor(tmp_path)

        def exploding(job_id, revision, forcing):
            raise RuntimeError("boom")

        ex._worker._solve_full = exploding
        with pytest.raises(RuntimeError, match="boom"):
            ex.execute([_params_edit(ex, new_state={"albedo_b": 0.35})])
        assert ex.scene_revision == 0
        assert ex._applied_model_parameters is None
        assert ex._worker._published_model_parameters is None
        assert not list(ex.results_root.rglob("rev-*"))
        # Retry on a healthy worker publishes and commits the parameters.
        captured: dict = {}
        _stub_full(ex, captured)
        executed = ex.execute(
            [_params_edit(ex, new_state={"albedo_b": 0.35}, edit_id="params-2")]
        )
        assert executed.published
        assert ex._applied_model_parameters == {"albedo_b": 0.35}


# ---------------------------------------------------------------------------
# Scientific E2E: real physics, ground-truth differentials
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def params_site(tmp_path_factory):
    """One real-SVF 64x64 site (the u-c2 physics fixture layout: two
    building blocks, one tree, land-cover classes, DOY-172 forcing)."""
    root = tmp_path_factory.mktemp("params_integration_site")
    grid, site = _make_prepared_site(
        root,
        rows=ROWS,
        cols=COLS,
        pixel=PIXEL,
        origin=ORIGIN,
        epsg=EPSG,
        base_trees=(FIXTURE_TREE,),
        met_hours=MET_HOURS,
    )
    _compute_baseline_svf(site)
    cache = _build_cache(
        site,
        root / "cache",
        met_path=site / "metfiles" / f"metfile_0_0_{DATE_STR}.txt",
        site_id="params-int",
    )
    return SimpleNamespace(grid=grid, site=site, cache=cache)


def _direct_run(
    params_site, scratch: Path, *, model_parameters=None, met_overlay=None
) -> dict[str, np.ndarray]:
    """The independent oracle: a direct ``run_full_tile`` run."""
    forcing = load_site_forcing(
        params_site.cache,
        site_dir=params_site.site,
        selected_date_str=DATE_STR,
        overlay=met_overlay,
    )
    return run_full_tile(
        params_site.cache,
        TreeLayer(params_site.cache.tree_base, params_site.grid),
        forcing=forcing,
        site_dir=params_site.site,
        scratch_dir=scratch,
        requested_variables=ORACLE_VARIABLES,
        model_parameters=model_parameters,
    )


def _params_executor(params_site, tmp_path: Path) -> PlanExecutor:
    return PlanExecutor(
        cache=params_site.cache,
        layer=TreeLayer(params_site.cache.tree_base, params_site.grid),
        site_dir=params_site.site,
        results_root=tmp_path / "results",
        selected_date_str=DATE_STR,
        influence_config=LOCAL_ALWAYS,
    )


def _execute_params_batch(executor: PlanExecutor, new_state, old_state):
    executed = executor.execute(
        [_params_edit(executor, new_state=new_state, old_state=old_state)]
    )
    assert executed.published
    assert executed.mode == "full"
    assert executed.write_windows == (executor.grid.full_window,)
    return executed, load_patch(executed.patch_paths[0])


@pytest.mark.scientific
class TestParamsEndToEndDifferential:
    def test_committed_albedo_edit_matches_direct_run_oracle_bitwise(
        self, tmp_path: Path, params_site
    ) -> None:
        """Ground truth: the executor-published products equal a direct
        ``run_full_tile(model_parameters=...)`` run — bitwise, every band,
        cell, and variable — and honor u-c2's decoupling table vs the
        baseline (albedo_b changes utci/tmrt/kup/kdown, leaves
        lup/ldown/shadow bitwise identical)."""
        executor = _params_executor(params_site, tmp_path)
        _executed, patch = _execute_params_batch(
            executor, {"albedo_b": 0.35}, {"albedo_b": 0.2}
        )

        oracle = _direct_run(
            params_site,
            tmp_path / "oracle_scratch",
            model_parameters={"albedo_b": 0.35},
        )
        for variable in SCI_VARIABLES:
            scenario = patch.variable(variable)
            assert scenario.shape == oracle[variable].shape
            assert np.array_equal(scenario, oracle[variable], equal_nan=True), (
                f"{variable}: executor path differs from the direct "
                "model-parameter oracle"
            )

        # The identity override (firstdayleaf=150 stays inside the leaf-on
        # window on the DOY-172 fixture) is physics-no-op through the SAME
        # direct path: it is the direct-path baseline the decoupling table
        # runs against (its faithfulness to the legacy path is pinned
        # separately below).
        identity = _direct_run(
            params_site,
            tmp_path / "identity_scratch",
            model_parameters={"firstdayleaf": 150},
        )

        # Coupled variables must CHANGE (wiring, not tolerance) ...
        for variable in ("utci", "tmrt", "kup", "kdown"):
            assert not np.array_equal(
                oracle[variable], identity[variable], equal_nan=True
            ), f"{variable}: albedo_b edit did not change {variable}"
        # ... decoupled variables stay BITWISE identical.
        for variable in ("lup", "ldown", "shadow"):
            assert np.array_equal(
                oracle[variable], identity[variable], equal_nan=True
            ), f"{variable}: albedo_b edit must leave {variable} bitwise"

        # Cross-path decoupling at the patch level: the executor's shadow
        # (direct path) is bitwise the LEGACY compute_utci baseline's.
        legacy = _direct_run(params_site, tmp_path / "legacy_scratch")
        assert np.array_equal(
            patch.variable("shadow"), legacy["shadow"], equal_nan=True
        )
        assert not np.array_equal(
            patch.variable("utci"), legacy["utci"], equal_nan=True
        )

    def test_direct_path_identity_override_is_bitwise_legacy(
        self, tmp_path: Path, params_site
    ) -> None:
        """The direct-path replication is faithful: an identity override
        (a parameter whose value keeps the physics unchanged) through the
        direct path equals the LEGACY ``compute_utci`` run bitwise."""
        identity = _direct_run(
            params_site,
            tmp_path / "identity_scratch",
            model_parameters={"firstdayleaf": 150},
        )
        legacy = _direct_run(params_site, tmp_path / "legacy_scratch")
        for variable in SCI_VARIABLES:
            assert np.array_equal(
                identity[variable], legacy[variable], equal_nan=True
            ), (
                f"{variable}: the direct-path replication differs from the "
                "legacy compute_utci run under an identity override"
            )

    def test_accumulated_last_write_wins_across_two_published_batches(
        self, tmp_path: Path, params_site
    ) -> None:
        """Two published parameter batches: the second batch's override of
        the first batch's name wins and the first batch's OTHER names
        survive — the final patch is bitwise the direct run under the
        folded mapping."""
        executor = _params_executor(params_site, tmp_path)
        _first, patch1 = _execute_params_batch(
            executor, {"albedo_b": 0.3}, {"albedo_b": 0.2}
        )
        assert executor._applied_model_parameters == {"albedo_b": 0.3}
        mid_oracle = _direct_run(
            params_site,
            tmp_path / "oracle_mid",
            model_parameters={"albedo_b": 0.3},
        )
        for variable in SCI_VARIABLES:
            assert np.array_equal(
                patch1.variable(variable),
                mid_oracle[variable],
                equal_nan=True,
            ), f"{variable}: first params batch differs from its oracle"

        second, patch2 = _execute_params_batch(
            executor,
            {"albedo_b": 0.35, "ewall": 0.85},
            {"albedo_b": 0.3, "ewall": 0.9},
        )
        assert second.scene_revision == 2
        assert executor._applied_model_parameters == {
            "albedo_b": 0.35,
            "ewall": 0.85,
        }
        final_oracle = _direct_run(
            params_site,
            tmp_path / "oracle_final",
            model_parameters={"albedo_b": 0.35, "ewall": 0.85},
        )
        for variable in SCI_VARIABLES:
            assert np.array_equal(
                patch2.variable(variable),
                final_oracle[variable],
                equal_nan=True,
            ), (
                f"{variable}: accumulated two-batch fold differs from the "
                "direct run under the folded mapping"
            )

    def test_mixed_params_and_met_batch_matches_dual_edit_oracle(
        self, tmp_path: Path, params_site
    ) -> None:
        """A mixed params+met batch: BOTH payloads active in ONE full job,
        bitwise equal to the direct run under the same overlay AND the
        same kernel arguments."""
        executor = _params_executor(params_site, tmp_path)
        param = _params_edit(executor, new_state={"albedo_b": 0.35})
        met = _executor_met_edit(executor, time_index=1, value=30.0)
        executed = executor.execute([param, met])
        assert executed.published
        assert executed.mode == "full"
        patch = load_patch(executed.patch_paths[0])
        assert executor._applied_model_parameters == {"albedo_b": 0.35}
        assert executor._applied_forcing_overlay is not None

        overlay = _met_overlay(
            params_site.cache,
            params_site.site,
            ("air_temperature", 1, 30.0),
        )
        oracle = _direct_run(
            params_site,
            tmp_path / "oracle_scratch",
            model_parameters={"albedo_b": 0.35},
            met_overlay=overlay,
        )
        for variable in SCI_VARIABLES:
            assert np.array_equal(
                patch.variable(variable), oracle[variable], equal_nan=True
            ), f"{variable}: mixed batch differs from the dual-edit oracle"
