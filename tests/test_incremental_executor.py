# SPDX-License-Identifier: GPL-3.0-only
"""U-C1 tests: plan executor, temporal store, engine LOW absorption (L2/L3/L5).

Fast suite (unmarked): the store contract, FrozenMapping constructor
normalization (L3), coalesced-window cancellation (L5), the requested-times
fence, wind-coefficient detection, the read-halo staging helper (L2), and
the executor lifecycle on the tiny site with the worker's solvers stubbed
(no SOLWEIG physics runs unmarked).

``scientific`` suite: vegetation E2E through the executor on a real-SVF
site — the executor must run the REAL worker and produce results
bitwise-consistent with a direct worker call on the same site.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from solweig_gpu.incremental.edit_types import (
    EditCommand,
    FrozenMapping,
    ObjectStateChange,
    OutputSelectionDelta,
    SpatialScope,
    ValidatedEdit,
    VegetationObjectDelta,
    coalesce_source_deltas,
)
from solweig_gpu.incremental.geometry import (
    InfluenceConfig,
    RasterGrid,
    RasterWindow,
)
from solweig_gpu.incremental.executor import (
    DEFAULT_TEMPORAL_STEPS,
    EXECUTABLE_SOURCE_NODES,
    ExecutorError,
    PlanExecutor,
    VARIABLE_TO_RESULT_NODE,
    detect_wind_coefficients,
    stage_read_window,
)
from solweig_gpu.incremental.planner import (
    PlanningError,
    fence_requested_times,
)
from solweig_gpu.incremental.result import ResultPatch, load_patch
from solweig_gpu.incremental.store import (
    StaleResultError,
    StoreError,
    TemporalResultEntry,
    TemporalResultStore,
)
from solweig_gpu.incremental.trees import TreeLayer
from solweig_gpu.incremental.worker import ExactWorker

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

# ---------------------------------------------------------------------------
# Store: (node_id, time_index) keying (U-C M2)
# ---------------------------------------------------------------------------


def _entry(node: str, time: int, revision: int, job: str = "job-1") -> TemporalResultEntry:
    return TemporalResultEntry(
        node_id=node,
        time_index=time,
        scene_revision=revision,
        job_id=job,
        mode="local",
        write_window=RasterWindow(0, 4, 0, 4),
        patch_path=Path("/tmp/rev-000001-job-1"),
    )


class TestTemporalResultStore:
    def test_publish_and_lookup_per_time(self) -> None:
        store = TemporalResultStore()
        keys = store.publish(
            [_entry("utci", t, revision=1) for t in range(3)]
        )
        assert keys == (("utci", 0), ("utci", 1), ("utci", 2))
        assert store.lookup("utci", 2).scene_revision == 1
        assert store.lookup("utci", 3) is None
        assert store.available_times("utci") == (0, 1, 2)
        assert store.available_times("tmrt") == ()
        assert store.revision_at("utci", 1) == 1
        assert store.revision_at("utci", 9) is None

    def test_different_times_carry_different_revisions(self) -> None:
        # The whole point of M2: per-node counters cannot express this.
        store = TemporalResultStore()
        store.publish([_entry("utci", 4, revision=1)])
        store.publish([_entry("utci", 5, revision=2)])
        assert store.revision_at("utci", 4) == 1
        assert store.revision_at("utci", 5) == 2

    def test_publish_is_atomic(self) -> None:
        store = TemporalResultStore()
        with pytest.raises(StoreError):
            store.publish(
                [
                    _entry("utci", 0, 1),
                    TemporalResultEntry(
                        node_id="",
                        time_index=0,
                        scene_revision=1,
                        job_id="j",
                        mode="local",
                        write_window=RasterWindow(0, 4, 0, 4),
                    ),
                ]
            )
        assert store.lookup("utci", 0) is None  # nothing written

    def test_stale_publish_refused(self) -> None:
        store = TemporalResultStore()
        store.publish([_entry("utci", 0, revision=2, job="a")])
        with pytest.raises(StaleResultError):
            store.publish([_entry("utci", 0, revision=1, job="b")])

    def test_same_revision_same_job_is_idempotent(self) -> None:
        store = TemporalResultStore()
        store.publish([_entry("utci", 0, revision=2, job="a")])
        keys = store.publish([_entry("utci", 0, revision=2, job="a")])
        assert keys == (("utci", 0),)
        assert len(store) == 1

    def test_same_revision_different_job_is_conflict(self) -> None:
        store = TemporalResultStore()
        store.publish([_entry("utci", 0, revision=2, job="a")])
        with pytest.raises(StoreError):
            store.publish([_entry("utci", 0, revision=2, job="b")])

    def test_newer_revision_supersedes(self) -> None:
        store = TemporalResultStore()
        store.publish([_entry("utci", 0, revision=1, job="a")])
        store.publish([_entry("utci", 0, revision=2, job="b")])
        entry = store.lookup("utci", 0)
        assert entry.scene_revision == 2 and entry.job_id == "b"

    def test_invalidate_by_time_and_whole_node(self) -> None:
        store = TemporalResultStore()
        store.publish([_entry("utci", t, 1) for t in range(4)])
        removed = store.invalidate("utci", times=(2, 3))
        assert removed == (("utci", 2), ("utci", 3))
        assert store.available_times("utci") == (0, 1)
        assert store.coverage_gaps("utci", (0, 1, 2, 3)) == (2, 3)
        assert store.invalidate("utci") == (("utci", 0), ("utci", 1))
        assert store.max_revision() == 0  # everything dropped

    def test_grid_out_of_bounds_rejected(self) -> None:
        grid = RasterGrid(8, 8, 1.0)
        store = TemporalResultStore(grid=grid)
        entry = TemporalResultEntry(
            node_id="utci", time_index=0, scene_revision=1, job_id="j",
            mode="local", write_window=RasterWindow(0, 16, 0, 4),
        )
        with pytest.raises(StoreError):
            store.publish([entry])


# ---------------------------------------------------------------------------
# Engine LOW absorption: L3 (FrozenMapping) and L5 (canceled windows)
# ---------------------------------------------------------------------------


class TestEngineLowAbsorption:
    def test_l3_constructor_normalizes_key_order(self) -> None:
        first = FrozenMapping((("b", 2), ("a", 1)))
        second = FrozenMapping((("a", 1), ("b", 2)))
        assert first == second
        assert hash(first) == hash(second)
        assert first._items == second._items  # sorted, not arrival order

    def test_l3_deduplicates_last_write_wins(self) -> None:
        mapping = FrozenMapping((("a", 1), ("a", 2)))
        assert dict(mapping) == {"a": 2}
        assert mapping._items == (("a", 2),)

    def test_l5_canceled_add_delete_drops_windows(self) -> None:
        window = RasterWindow(10, 20, 10, 20)
        add = VegetationObjectDelta(
            source_node_id="vegetation_dsm",
            adapter_id="vegetation_geometry",
            objects=(ObjectStateChange("t1", None, {"height_m": 5.0}),),
            windows=(window,),
        )
        delete = VegetationObjectDelta(
            source_node_id="vegetation_dsm",
            adapter_id="vegetation_geometry",
            objects=(ObjectStateChange("t1", {"height_m": 5.0}, None),),
            windows=(RasterWindow(30, 40, 30, 40),),
        )
        coalesced = coalesce_source_deltas([add, delete])
        assert coalesced.objects == ()
        assert coalesced.windows == ()
        assert coalesced.is_noop

    def test_l5_mixed_survival_keeps_windows_conservatively(self) -> None:
        # Delta 1 adds A; delta 2 deletes A and adds B. A cancels, B
        # survives — delta 2's windows must stay (attribution is only safe
        # when EVERY object of a delta canceled).
        w1, w2 = RasterWindow(10, 20, 10, 20), RasterWindow(30, 40, 30, 40)
        first = VegetationObjectDelta(
            source_node_id="vegetation_dsm",
            adapter_id="vegetation_geometry",
            objects=(ObjectStateChange("A", None, {"height_m": 5.0}),),
            windows=(w1,),
        )
        second = VegetationObjectDelta(
            source_node_id="vegetation_dsm",
            adapter_id="vegetation_geometry",
            objects=(
                ObjectStateChange("A", {"height_m": 5.0}, None),
                ObjectStateChange("B", None, {"height_m": 6.0}),
            ),
            windows=(w2,),
        )
        coalesced = coalesce_source_deltas([first, second])
        surviving = {change.object_id for change in coalesced.objects}
        assert surviving == {"B"}
        # w1 (all its objects canceled) drops; w2 (mixed delta) stays.
        assert w2 in coalesced.windows
        assert w1 not in coalesced.windows


# ---------------------------------------------------------------------------
# Fence (u-b-met LOW) + wind detection (scope items c/d)
# ---------------------------------------------------------------------------


_DEFAULT_STATE = object()  # distinguishes "unset" from an explicit None


def _veg_command(
    revision: int = 0,
    edit_id: str = "veg-1",
    times: tuple[int, ...] | None = (1,),
    outputs: tuple[str, ...] = ("utci",),
    operation: str = "add",
    old_state: dict | None = None,
    new_state: dict | None = _DEFAULT_STATE,
) -> EditCommand:
    return EditCommand(
        edit_id=edit_id,
        scenario_id="default",
        base_scene_revision=revision,
        adapter_id="vegetation_geometry",
        operation=operation,
        old_state=old_state,
        new_state={"tree_id": "t1",
                   "x_m": TINY_ORIGIN[0] + 40.5 * TINY_PIXEL,
                   "y_m": TINY_ORIGIN[1] - 40.5 * TINY_PIXEL,
                   "height_m": 4.0, "canopy_radius_m": 2.0}
        if new_state is _DEFAULT_STATE
        else new_state,
        requested_outputs=outputs,
        requested_times=times,
    )


def _met_edit(
    executor: PlanExecutor,
    *,
    revision: int | None = None,
    times: tuple[int, ...] | None = (1,),
    time_index: int = 1,
    value: float = 26.0,
    edit_id: str = "met-1",
):
    """A validated meteorology edit for the executor's scenario.

    ``time_index`` is the changed timestep (the delta cell); ``times`` is
    the batch's requested window (R1 service set).
    """
    from tests.test_incremental_adapters_met_time import (
        ADAPTER_ID as MET_ADAPTER_ID,
    )

    command = EditCommand(
        edit_id=edit_id,
        scenario_id=executor._scenario_id,
        base_scene_revision=(
            executor.scene_revision if revision is None else revision
        ),
        adapter_id=MET_ADAPTER_ID,
        operation="update_time_row",
        old_state={"time_index": time_index, "values": {"air_temperature": 20.0}},
        new_state={"time_index": time_index, "values": {"air_temperature": value}},
        requested_outputs=("utci",),
        requested_times=times,
    )
    return executor.validate(command)


class TestFenceAndDetection:
    def test_fence_rejects_out_of_series_times(self) -> None:
        validated = _validated(_veg_command(times=(1, 7)))
        with pytest.raises(PlanningError, match="outside the site forcing series"):
            fence_requested_times([validated], (0, 1, 2))
        assert fence_requested_times([validated], (0, 1, 2, 7)) == (1, 7)

    def test_fence_requires_available_times(self) -> None:
        with pytest.raises(PlanningError):
            fence_requested_times([_validated(_veg_command(times=(1,)))], ())

    def test_planner_applies_fence_when_context_has_times(self, tmp_path: Path) -> None:
        grid, site = _make_tiny_site(tmp_path / "site")
        executor = _executor(tmp_path, grid=grid, site=site)
        edit = _validated(_veg_command(times=(99,)))
        with pytest.raises(PlanningError, match="forcing series"):
            executor.plan([edit])

    def test_wind_detection_off_by_default(self, tmp_path: Path) -> None:
        assert detect_wind_coefficients(tmp_path, "0_0") is False

    def test_wind_detection_sees_oracle_layout(self, tmp_path: Path) -> None:
        (tmp_path / "WindCoeff").mkdir()
        (tmp_path / "WindCoeff" / "WindCoeff_dir000_0_0.tif").write_bytes(b"")
        (tmp_path / "WindCoeff" / "WindCoeff_dir030_1_2.tif").write_bytes(b"")
        assert detect_wind_coefficients(tmp_path, "0_0") is True
        assert detect_wind_coefficients(tmp_path, "9_9") is False

    def test_executor_site_context_carries_detection_and_fences(
        self, tmp_path: Path
    ) -> None:
        grid, site = _make_tiny_site(tmp_path / "site")
        executor = _executor(tmp_path, grid=grid, site=site)
        context = executor.site_context()
        assert context.available_times == tuple(range(executor.cache.time_steps))
        assert context.scene_revision == 0
        assert context.has_wind_coefficients is False

        # Now drop wind rasters into the site: detection must wire into the
        # SAME context the executor plans with (the planner's wind fence
        # fires — oracle parity, never silent).
        wind_dir = site / "WindCoeff"
        wind_dir.mkdir(exist_ok=True)
        (wind_dir / "WindCoeff_dir000_0_0.tif").write_bytes(b"")
        executor._has_wind_coefficients = None  # re-detect
        assert executor.has_wind_coefficients is True
        with pytest.raises(PlanningError, match="wind"):
            executor.plan([_validated(_veg_command())])


# ---------------------------------------------------------------------------
# Read-halo staging helper (L2)
# ---------------------------------------------------------------------------


class TestStageReadWindow:
    def test_unions_declared_read_windows_and_clamps(self) -> None:
        write = RasterWindow(50, 60, 50, 60)
        declared = (RasterWindow(40, 70, 40, 70), RasterWindow(52, 58, 52, 58))
        staged = stage_read_window(write, declared, rows=128, cols=128)
        assert (staged.row_start, staged.row_stop) == (40, 70)
        assert (staged.col_start, staged.col_stop) == (40, 70)

    def test_clamps_to_grid(self) -> None:
        write = RasterWindow(0, 10, 0, 10)
        declared = (RasterWindow(0, 10, 0, 200),)
        staged = stage_read_window(write, declared, rows=128, cols=128)
        assert staged.col_stop == 128

    def test_reads_never_shrink_and_writes_never_inflate(self) -> None:
        # The staged read is a superset of the write window (reads only
        # grow); the helper returns a read extent and the write window
        # object itself is never rewritten.
        write = RasterWindow(50, 60, 50, 60)
        before = (write.row_start, write.row_stop, write.col_start, write.col_stop)
        staged = stage_read_window(write, (), rows=128, cols=128)
        assert staged == write  # no declared halo: staged == write floor
        staged = stage_read_window(
            write, (RasterWindow(45, 65, 45, 65),), rows=128, cols=128
        )
        assert staged.area >= write.area
        assert (
            before
            == (write.row_start, write.row_stop, write.col_start, write.col_stop)
        )


# ---------------------------------------------------------------------------
# Executor lifecycle (tiny site; worker solvers stubbed — no physics)
# ---------------------------------------------------------------------------


def _validated(command: EditCommand) -> ValidatedEdit:
    return ValidatedEdit(
        command=command,
        adapter_id=command.adapter_id,
        schema_version=1,
        source_node_id="vegetation_dsm"
        if command.adapter_id == "vegetation_geometry"
        else "output_selection",
        delta=VegetationObjectDelta(
            source_node_id="vegetation_dsm",
            adapter_id="vegetation_geometry",
            objects=(ObjectStateChange("t1", None, dict(command.new_state)),),
            windows=(RasterWindow(0, 8, 0, 8),),
        )
        if command.adapter_id == "vegetation_geometry"
        else OutputSelectionDelta(
            "output_selection", command.adapter_id, ("utci",), ("tmrt",)
        ),
    )


def _executor(
    tmp_path: Path,
    *,
    grid: RasterGrid,
    site: Path,
    config: InfluenceConfig = LOCAL_ALWAYS,
) -> PlanExecutor:
    cache = _build_cache(
        site, tmp_path / "cache",
        met_path=site / "metfiles" / f"metfile_0_0_{DATE_STR}.txt",
        site_id="tiny",
    )
    layer = TreeLayer(cache.tree_base, grid)
    return PlanExecutor(
        cache=cache,
        layer=layer,
        site_dir=site,
        results_root=tmp_path / "results",
        selected_date_str=DATE_STR,
        influence_config=config,
    )


def _stub_patch(
    variables: tuple[str, ...],
    job_id: str,
    revision: int,
    window: RasterWindow,
    *,
    mode: str = "local",
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
        read_window=window.expand(2).clamp(rows=TINY_ROWS, cols=TINY_ROWS),
        site_id="tiny",
        tile_key="0_0",
        cache_manifest_sha256="0" * 64,
        model_version="test",
        variables=variables,
        arrays=arrays,
        time_start=0,
        time_stop=3,
    )


def _stub_local(executor: PlanExecutor, captured: dict) -> None:
    def fake(job_id, revision, forcing, write_windows, **kwargs):
        captured["windows"] = tuple(write_windows)
        captured["variables"] = tuple(executor._worker.requested_variables)
        return [_stub_patch(
            tuple(executor._worker.requested_variables),
            job_id, revision, write_windows[0],
        )]

    executor._worker._solve_local = fake


@pytest.fixture()
def executor(tmp_path: Path):
    grid, site = _make_tiny_site(tmp_path / "site")
    ex = _executor(tmp_path, grid=grid, site=site)
    return ex, grid, site


class TestExecutorLifecycle:
    def test_context_always_bound_with_diurnal_floor(self, executor) -> None:
        ex, _grid, _site = executor
        assert ex.default_temporal_steps == DEFAULT_TEMPORAL_STEPS == 24
        assert ex.available_times == tuple(range(ex.cache.time_steps))
        # A plan is impossible without the context: prove binding by fencing.
        with pytest.raises(PlanningError, match="forcing series"):
            ex.plan([_validated(_veg_command(times=(50,)))])

    def test_execute_publishes_records_and_advances(self, executor) -> None:
        ex, _grid, _site = executor
        captured: dict = {}
        _stub_local(ex, captured)

        edit = ex.validate(_veg_command())
        executed = ex.execute([edit])

        assert executed.published
        assert executed.status == "published"
        assert executed.mode == "local"
        assert executed.scene_revision == 1  # base 0 -> publishes at 1
        assert ex.scene_revision == 1
        assert captured["windows"][0].area > 0

        patch = load_patch(executed.patch_paths[0])
        assert patch.scene_revision == 1
        assert patch.variables == ("utci",)  # worker carries the request

        # The layer replayed the coalesced delta (worker watermark consumed).
        assert [tree.tree_id for tree in ex.layer.current_trees()] == ["t1"]
        assert ex._worker.pending_batch() is None

    def test_store_records_per_node_and_time(self, executor) -> None:
        ex, _grid, _site = executor
        _stub_local(ex, {})
        ex.execute([ex.validate(_veg_command(outputs=("utci", "tmrt", "shadow")))])

        keys = set(ex.store)
        # shadow transports under its graph node (time_shadow), tmrt/utci
        # under their own ids; three timesteps each.
        assert {("utci", 0), ("utci", 2), ("tmrt", 1), ("time_shadow", 2)} <= keys
        assert len(keys) == 9
        entry = ex.store.lookup("time_shadow", 1)
        assert entry.scene_revision == 1
        assert entry.mode == "local"
        assert entry.patch_path is not None
        assert ex.store.max_revision() == 1

    def test_zero_output_node_plan_transports_non_node_layer(self, executor) -> None:
        # NEW-1: a shadow-only request prunes every OUTPUT node from the
        # plan; the executor still runs the physics and transports shadow.
        ex, _grid, _site = executor
        captured: dict = {}
        _stub_local(ex, captured)
        edit = ex.validate(_veg_command(outputs=("shadow",)))
        plan = ex.plan([edit])
        assert "utci" not in {i.node_id for i in plan.node_impacts}
        executed = ex.execute([edit])
        assert executed.published
        assert "time_shadow" in {node for node, _t in executed.store_keys}
        assert "utci" in captured["variables"]  # physics floor

    def test_producible_extras_are_recorded_not_dropped(self, executor) -> None:
        ex, _grid, _site = executor
        _stub_local(ex, {})
        edit = ex.validate(_veg_command(outputs=("utci", "kup")))
        executed = ex.execute([edit])
        assert executed.published
        assert executed.unpublished_variables == (
            ("kup", executed.unpublished_variables[0][1]),
        )
        assert "kup" in executed.unpublished_variables[0][0]
        assert "kup" not in load_patch(executed.patch_paths[0]).variables

    def test_view_only_batch_is_accepted_noop(self, executor) -> None:
        ex, _grid, _site = executor
        command = EditCommand(
            edit_id="view-1",
            scenario_id="default",
            base_scene_revision=0,
            adapter_id="output_view",
            operation="select_layer",
            old_state={"layers": ("utci",)},
            new_state={"layers": ("tmrt",)},
            requested_outputs=("tmrt",),
            requested_times=None,
        )
        executed = ex.execute([_validated(command)])
        assert executed.status == "no-op"
        assert executed.patch_paths == ()
        assert ex.scene_revision == 0
        assert not list((ex.results_root).rglob("rev-*"))

    def test_view_plus_vegetation_batch_stages_only_vegetation(
        self, executor
    ) -> None:
        # Heterogeneous batch: the view selection plans but stages nothing
        # (no concrete output_view adapter exists yet); the vegetation edit
        # still executes.
        ex, _grid, _site = executor
        _stub_local(ex, {})
        veg = ex.validate(_veg_command())
        command = EditCommand(
            edit_id="view-1",
            scenario_id="default",
            base_scene_revision=0,
            adapter_id="output_view",
            operation="select_layer",
            old_state={"layers": ("utci",)},
            new_state={"layers": ("tmrt",)},
            requested_outputs=("tmrt",),
            requested_times=None,
        )
        executed = ex.execute([veg, _validated(command)])
        assert executed.published
        assert ex.layer.current_trees()[0].tree_id == "t1"

    def test_coalesced_noop_batch_never_touches_layer(self, executor) -> None:
        ex, _grid, _site = executor
        add = ex.validate(_veg_command(edit_id="veg-1"))
        tree = dict(add.command.new_state)
        delete = ex.validate(
            _veg_command(
                edit_id="veg-2",
                operation="delete",
                old_state=tree,
                new_state=None,
            )
        )
        executed = ex.execute([add, delete])
        assert executed.status == "no-op"
        assert ex.layer.current_trees() == ()
        assert ex.scene_revision == 0

    def test_met_batch_executes_through_u_c3_seam(self, executor) -> None:
        """u-c3 landed: a committed met edit no longer raises
        NotIntegratedError — it stages the scenario overlay on the worker
        and executes (physics stubbed here; the scientific E2E lives in
        tests/test_incremental_met_integration.py)."""
        ex, _grid, _site = executor
        captured: dict = {}
        ex._worker._solve_full = (
            lambda job_id, revision, forcing: captured.update(
                forcing=forcing,
                patch=_stub_patch(
                    ("utci",), job_id, revision,
                    RasterWindow(0, TINY_ROWS, 0, TINY_ROWS), mode="full",
                ),
            )
            or [captured["patch"]]
        )
        edit = _met_edit(ex, times=(1,))
        executed = ex.execute([edit])
        assert executed.published
        assert executed.mode == "full"
        assert executed.scene_revision == 1
        # The worker materialized the RESOLVED forcing (declared cell only).
        assert captured["forcing"].met_table[1, 11] == 26.0
        assert captured["forcing"].overlay is not None
        baseline = np.loadtxt(
            _site / "metfiles" / f"metfile_0_0_{DATE_STR}.txt",
            skiprows=1, delimiter=" ",
        )
        assert np.array_equal(
            captured["forcing"].met_table[0], baseline[0]
        )
        # Forcing-only batches touch no trees; the layer stays empty.
        assert ex.layer.current_trees() == ()

    def test_met_batch_out_of_series_refused_whole(self, executor) -> None:
        """The seam fail-fast: an overlay cell outside the site's forcing
        series (hand-built delta bypassing the adapter's timestep fence —
        the planner's delta check is names-only, time 3 on a 3-step
        series) is refused at the forcing materialization, before any
        state advances."""
        from solweig_gpu.incremental.edit_types import SourceDeltaError
        from tests.test_incremental_edit_engine import met_edit

        ex, _grid, _site = executor
        edit = met_edit(times=(1,))
        edit = replace(edit, command=replace(edit.command, scenario_id="default"))
        with pytest.raises(SourceDeltaError, match="outside the baseline"):
            ex.execute([edit])
        assert ex.scene_revision == 0
        assert ex._applied_forcing_overlay is None
        assert ex._worker._forcing_overlay is None
        assert not list(ex.results_root.rglob("rev-*"))

    def test_mixed_veg_params_batch_executes_through_u_c7_seam(
        self, executor
    ) -> None:
        """u-c7 landed: a mixed vegetation + model-parameter batch no longer
        raises NotIntegratedError — the committed parameter deltas fold
        (kernel_arguments_from_deltas) over the parameter-free baseline
        into the kernel-argument mapping staged on the worker, and the
        batch executes with the tree replay and the params both riding the
        job (physics stubbed here; the scientific E2E lives in
        tests/test_incremental_params_integration.py)."""
        ex, _grid, _site = executor
        captured: dict = {}
        ex._worker._solve_full = (
            lambda job_id, revision, forcing: captured.update(
                params=ex._worker._model_parameters,
                patch=_stub_patch(
                    ("utci",), job_id, revision,
                    RasterWindow(0, TINY_ROWS, 0, TINY_ROWS), mode="full",
                ),
            )
            or [captured["patch"]]
        )
        veg = ex.validate(_veg_command())
        param = ex.validate(
            EditCommand(
                edit_id="params-1",
                scenario_id="default",
                base_scene_revision=0,
                adapter_id="model_receptor_parameters",
                operation="update",
                old_state={"albedo_b": 0.2},
                new_state={"albedo_b": 0.3},
                requested_outputs=("utci",),
                requested_times=(1,),
            )
        )
        executed = ex.execute([veg, param])
        assert executed.published
        assert executed.mode == "full"  # params are site-global: full tile
        assert executed.scene_revision == 1
        # The tree replayed into the layer and the worker consumed it.
        assert [tree.tree_id for tree in ex.layer.current_trees()] == ["t1"]
        assert ex._worker.pending_batch() is None
        # The staged payload is the float()-coerced fold, and the solve
        # saw exactly it (both solve paths forward the staged mapping).
        assert captured["params"] == {"albedo_b": 0.3}
        assert type(captured["params"]["albedo_b"]) is float
        assert ex._worker._published_model_parameters == {"albedo_b": 0.3}
        # The executor committed the fold as the accumulated state.
        assert ex._applied_model_parameters == {"albedo_b": 0.3}

    def test_building_batch_executes_through_regeneration_seam(
        self, executor, monkeypatch
    ) -> None:
        ex, grid, _site = executor
        # u-c6 landed: building_dsm batches now EXECUTE — the regeneration
        # chain seam fires (real physics, executor-vs-oracle differential,
        # rollback, and mixed-batch overlay carry are covered by the u-c6
        # building integration suite). This pin keeps the DISPATCH honest
        # with a stubbed chain: FULL mode, publish at revision R+1, store
        # entries, scenario provenance, worker watermarks adopted. The
        # still-unintegrated families keep refusing whole-batch.
        from solweig_gpu.incremental import executor as executor_mod
        from solweig_gpu.incremental.buildings import BuildingRasterResult
        from solweig_gpu.incremental.regenerate import BuildingRegenerationResult

        captured: dict = {}

        def fake_chain(**kwargs):
            captured.update(kwargs)
            cache = ex.cache
            zeros = np.zeros(
                (cache.time_steps, grid.rows, grid.cols), dtype=np.float32
            )
            return BuildingRegenerationResult(
                scenario_site_dir=ex.results_root / "default" / ".building" / "site",
                cache_dir=ex.results_root / "default" / ".building" / "cache",
                cache=cache,
                raster=np.zeros((grid.rows, grid.cols), dtype=np.float32),
                records=BuildingRasterResult(
                    raster=np.zeros((grid.rows, grid.cols), dtype=np.float32),
                    records=(),
                ),
                outputs={"utci": zeros, "tmrt": zeros.copy()},
                stages=("stage_scenario_site", "run_scenario_full_tile"),
            )

        monkeypatch.setattr(executor_mod, "regenerate_building_batch", fake_chain)

        building = ex.validate(
            EditCommand(
                edit_id="bld-1",
                scenario_id="default",
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
        executed = ex.execute([building])
        assert executed.published
        assert executed.mode == "full"
        assert executed.scene_revision == 1
        assert ex.scene_revision == 1
        assert ex.store.revision_at("utci", 0) == 1
        assert ex.store.lookup("utci", 0).mode == "full"
        # The chain received the re-validated massing edits and the
        # executor's scratch scenario root, and the publication adopted on
        # the worker (nothing left pending, overlays synced).
        assert captured["scenario_root"] == ex.results_root / "default" / ".building"
        assert captured["selected_date_str"] == DATE_STR
        assert len(captured["edits"]) == 1
        assert captured["edits"][0].building_id == "b1"
        assert ex._worker.pending_batch() is None
        assert ex._worker.publication_watermarks.landcover_overlay is None
        # The building-only chain call above ran parameter-free: the u-c7
        # default threads None (the legacy compute_utci path).
        assert captured["model_parameters"] is None
        # u-c7 landed: model parameters now execute through the
        # kernel-argument seam (worker path, physics stubbed) and commit
        # the accumulated parameter state; the still-unintegrated families
        # (selected_date_time, dem) have no adapter to validate, so their
        # refusal holds by construction.
        ex._worker._solve_full = (
            lambda job_id, revision, forcing: [
                _stub_patch(
                    ("utci",), job_id, revision,
                    RasterWindow(0, TINY_ROWS, 0, TINY_ROWS), mode="full",
                )
            ]
        )
        param = ex.validate(
            EditCommand(
                edit_id="params-1",
                scenario_id="default",
                base_scene_revision=ex.scene_revision,
                adapter_id="model_receptor_parameters",
                operation="update",
                old_state={"albedo_b": 0.2},
                new_state={"albedo_b": 0.3},
                requested_outputs=("utci",),
                requested_times=(1,),
            )
        )
        executed = ex.execute([param])
        assert executed.published
        assert executed.mode == "full"
        assert executed.scene_revision == 2
        assert ex._applied_model_parameters == {"albedo_b": 0.3}

    def test_stale_plan_revision_refused(self, executor) -> None:
        ex, _grid, _site = executor
        _stub_local(ex, {})
        ex.execute([ex.validate(_veg_command())])  # revision 0 -> 1

        edit = ex.validate(_veg_command(edit_id="veg-2", revision=1))
        plan = ex.plan([edit])  # planned against revision 1
        assert plan.scene_revision == 1
        # Force a stale execution attempt: rebuild the same-shaped plan at
        # the old revision through the lower-level entry point.
        stale = replace(plan, scene_revision=0)
        with pytest.raises(ExecutorError, match="stale"):
            ex.execute_plan(stale, [])

    def test_plan_without_revision_refused(self, executor) -> None:
        ex, _grid, _site = executor
        edit = ex.validate(_veg_command())
        plan = replace(ex.plan([edit]), scene_revision=None)
        with pytest.raises(ExecutorError, match="scene_revision"):
            ex.execute_plan(plan, [])

    def test_double_execution_of_same_deltas_rejected(self, executor) -> None:
        ex, _grid, _site = executor
        _stub_local(ex, {})
        edit = ex.validate(_veg_command())
        executed = ex.execute([edit])
        assert executed.published

        # A second execution of the same committed deltas (fresh plan at
        # the new revision) must fail loudly in the replay pre-flight —
        # the tree already exists.
        second_edit = ex.validate(_veg_command(edit_id="veg-2", revision=1))
        second_plan = ex.plan([second_edit])
        transaction = second_edit.delta  # deltas are the staged payload
        with pytest.raises(ExecutorError, match="replay"):
            ex.execute_plan(second_plan, [transaction])

    def test_reconciliation_rejects_uncovered_plan_window(self, executor) -> None:
        ex, _grid, _site = executor
        _stub_local(ex, {})
        edit = ex.validate(_veg_command())
        plan = ex.plan([edit])
        assert any(i.node_id == "relative_geometry" for i in plan.node_impacts)
        inflated = tuple(
            replace(
                impact,
                write_windows=(RasterWindow(0, TINY_ROWS, 0, TINY_ROWS),),
                read_windows=(RasterWindow(0, TINY_ROWS, 0, TINY_ROWS),),
            )
            if impact.node_id == "relative_geometry"
            else impact
            for impact in plan.node_impacts
        )
        crafted = replace(plan, node_impacts=inflated)
        with pytest.raises(ExecutorError, match="does not cover the plan"):
            ex.execute_plan(crafted, [edit.delta])
        # Rollback: revision and layer state are restored.
        assert ex.scene_revision == 0
        assert ex.layer.current_trees() == ()

    def test_full_promise_forces_full_execution(self, executor) -> None:
        # The planner's safety policy legitimately demotes big batches to
        # FULL (dirty fraction >= 0.3); the executor must honor the plan's
        # FULL promise even when the worker's own routing would go local.
        ex, _grid, _site = executor
        _stub_local(ex, {})  # would route local on its own
        ex._worker._solve_full = (
            lambda job_id, revision, forcing: [
                _stub_patch(
                    tuple(ex._worker.requested_variables),
                    job_id, revision,
                    RasterWindow(0, TINY_ROWS, 0, TINY_ROWS), mode="full",
                )
            ]
        )
        # A tall tree's SVF radius forces a >30% dirty fraction: FULL plan.
        tall = ex.validate(
            _veg_command(
                new_state={
                    "tree_id": "t1",
                    "x_m": TINY_ORIGIN[0] + 60.5 * TINY_PIXEL,
                    "y_m": TINY_ORIGIN[1] - 60.5 * TINY_PIXEL,
                    "height_m": 25.0,
                    "canopy_radius_m": 2.0,
                },
            )
        )
        plan = ex.plan([tall])
        assert any(
            impact.spatial_scope is SpatialScope.FULL for impact in plan.node_impacts
        )
        executed = ex.execute([tall])
        assert executed.published
        assert executed.mode == "full"
        assert executed.write_windows[0] == RasterWindow(0, TINY_ROWS, 0, TINY_ROWS)
        assert ex.scene_revision == 1

    def test_reconciliation_defends_full_promise(self, executor) -> None:
        # Defense in depth: if a future routing change ever publishes local
        # windows under a FULL promise, reconciliation refuses loudly.
        from solweig_gpu.incremental.worker import JobOutcome

        ex, _grid, _site = executor
        edit = ex.validate(_veg_command())
        plan = ex.plan([edit])
        crafted = replace(
            plan,
            node_impacts=tuple(
                replace(
                    impact,
                    spatial_scope=SpatialScope.FULL,
                    write_windows=(),
                    read_windows=(),
                )
                if impact.node_id == "utci"
                else impact
                for impact in plan.node_impacts
            ),
        )
        ex._run_worker = (
            lambda *, target_revision, force_full: JobOutcome(
                status="published",
                mode="local",
                job_id="j",
                scene_revision=target_revision,
                write_windows=(RasterWindow(0, 8, 0, 8),),
                read_windows=(RasterWindow(0, 8, 0, 8),),
                patch_paths=(),
            )
        )
        with pytest.raises(ExecutorError, match="FULL"):
            ex.execute_plan(crafted, [edit.delta])
        assert ex.scene_revision == 0
        assert ex.layer.current_trees() == ()

    def test_second_batch_supersedes_store_entries(self, executor) -> None:
        ex, _grid, _site = executor
        _stub_local(ex, {})
        first = ex.execute([ex.validate(_veg_command())])
        assert first.scene_revision == 1

        # The height 4->9 update pushes the dirty fraction past the safety
        # policy's 0.30 threshold, so batch 2 plans (and executes) FULL.
        ex._worker._solve_full = (
            lambda job_id, revision, forcing: [
                _stub_patch(
                    ("utci", "tmrt", "shadow"), job_id, revision,
                    RasterWindow(0, TINY_ROWS, 0, TINY_ROWS), mode="full",
                )
            ]
        )
        update = ex.validate(
            _veg_command(
                edit_id="veg-2",
                revision=1,
                operation="update",
                old_state=dict(_veg_command().new_state),
                new_state={
                    "tree_id": "t1",
                    "x_m": TINY_ORIGIN[0] + 40.5 * TINY_PIXEL,
                    "y_m": TINY_ORIGIN[1] - 40.5 * TINY_PIXEL,
                    "height_m": 9.0,
                    "canopy_radius_m": 2.0,
                },
            )
        )
        second = ex.execute([update])
        assert second.published
        assert second.mode == "full"
        assert second.scene_revision == 2
        assert ex.store.revision_at("utci", 0) == 2
        assert ex.store.lookup("utci", 0).mode == "full"
        assert ex.store.max_revision() == 2

    def test_worker_failure_rolls_back_state_and_layer(self, executor) -> None:
        ex, _grid, _site = executor

        def exploding_run(**kwargs):
            raise RuntimeError("boom")

        ex._worker.run = exploding_run
        edit = ex.validate(_veg_command())
        with pytest.raises(RuntimeError, match="boom"):
            ex.execute([edit])
        assert ex.scene_revision == 0
        assert ex.layer.current_trees() == ()
        assert not list(ex.results_root.rglob("rev-*"))
        assert ex.store.max_revision() == 0

    def test_executable_families_constant(self) -> None:
        # u-c3: meteorology; u-c4: landcover; u-c6: building_dsm;
        # u-c7: model_parameters.
        assert EXECUTABLE_SOURCE_NODES == frozenset(
            {
                "vegetation_dsm",
                "meteorology",
                "landcover",
                "building_dsm",
                "model_parameters",
            }
        )
        assert VARIABLE_TO_RESULT_NODE["shadow"] == "time_shadow"


# ---------------------------------------------------------------------------
# Scientific E2E: the executor drives the REAL worker (u-c1 mission check)
# ---------------------------------------------------------------------------

SCI_ROWS = SCI_COLS = 128
SCI_PIXEL = 2.0
SCI_ORIGIN = (300000.0, 4100000.0)
SCI_EPSG = 32616


@pytest.fixture(scope="module")
def real_site(tmp_path_factory):
    root = tmp_path_factory.mktemp("executor_real_site")
    from solweig_gpu.incremental.trees import TreeSpec

    base = TreeSpec(
        "b1", SCI_ORIGIN[0] + 20.5 * SCI_PIXEL,
        SCI_ORIGIN[1] - 20.5 * SCI_PIXEL, 3.0, 2.0,
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
        site, root / "cache",
        met_path=site / "metfiles" / f"metfile_0_0_{DATE_STR}.txt",
        site_id="executor-real",
    )
    return grid, site, cache


@pytest.mark.scientific
class TestExecutorEndToEnd:
    def test_executor_matches_direct_worker_bitwise(self, tmp_path: Path, real_site):
        """The executor must run the real physics and agree with a direct
        ExactWorker call on the same site, tree, and influence policy."""
        from solweig_gpu.incremental.trees import TreeSpec

        grid, site, cache = real_site
        tree = TreeSpec(
            "t1", SCI_ORIGIN[0] + 64.5 * SCI_PIXEL,
            SCI_ORIGIN[1] - 64.5 * SCI_PIXEL, 4.0, 2.0,
        )

        # -- direct worker (the reference path) -------------------------
        direct_layer = TreeLayer(cache.tree_base, grid)
        direct = ExactWorker(
            cache, direct_layer,
            site_dir=site, results_root=tmp_path / "direct",
            selected_date_str=DATE_STR,
            influence_config=LOCAL_ALWAYS,
            requested_variables=("utci", "tmrt", "shadow"),
        )
        direct_layer.add_tree(tree)
        direct.bump_scene_revision()
        reference = direct.run()
        assert reference.published and reference.mode == "local"

        # -- executor ----------------------------------------------------
        executor = PlanExecutor(
            cache=cache,
            layer=TreeLayer(cache.tree_base, grid),
            site_dir=site,
            results_root=tmp_path / "executed",
            selected_date_str=DATE_STR,
            influence_config=LOCAL_ALWAYS,
        )
        command = EditCommand(
            edit_id="veg-1",
            scenario_id="default",
            base_scene_revision=0,
            adapter_id="vegetation_geometry",
            operation="add",
            old_state=None,
            new_state={
                "tree_id": "t1",
                "x_m": tree.x_m, "y_m": tree.y_m,
                "height_m": tree.height_m, "canopy_radius_m": tree.canopy_radius_m,
            },
            requested_outputs=("utci", "tmrt", "shadow"),
            requested_times=(1,),
        )
        executed = executor.execute([executor.validate(command)])

        assert executed.published
        assert executed.mode == reference.mode == "local"
        assert executed.scene_revision == reference.scene_revision == 1
        assert executed.write_windows == reference.write_windows
        assert executed.read_windows == reference.read_windows

        executor_patch = load_patch(executed.patch_paths[0])
        reference_patch = load_patch(reference.patch_paths[0])
        assert sorted(executor_patch.variables) == sorted(reference_patch.variables)
        for variable in reference_patch.variables:
            left = executor_patch.arrays[variable]
            right = reference_patch.arrays[variable]
            assert left.shape == right.shape and np.array_equal(
                left, right, equal_nan=True
            ), f"{variable} differs between executor and direct worker"

        # Temporal store coverage: every variable at every timestep, at
        # the publication revision.
        for node in ("utci", "tmrt", "time_shadow"):
            assert executor.store.available_times(node) == (0, 1, 2, 3)
            assert executor.store.revision_at(node, 0) == 1
        assert executor.scene_revision == 1
