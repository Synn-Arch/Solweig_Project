# SPDX-License-Identifier: GPL-3.0-only
"""R3a: meteorology variable-affinity split + UTCI-only fast path.

The met family's safe variables split by CODE-PROVEN affinity (r3a Step 0):

* ``utci_only`` — ``wind_speed`` (col 9) and ``uhii`` (col 24) enter ONLY
  the comfort/human calculation at time t (utci_process.py:693/:700-703
  reads; :880-884 wind plane; :887/:890 Ta+uhii plane; :917 the UTCI call).
  No radiation term, no shadow, no surface thermal state, no geometry
  consumes them: Solweig_2022a_calc receives plain ``Ta[i]`` without uhii
  and no wind argument at all.
* ``radiation_affecting`` — ``air_temperature`` (solweig.py:2042-2059,
  2109-2117, 2163, 2213, 2255-2274), ``humidity`` (:2042-2059),
  ``radiation`` (:2051-2059, 2111-2117, 2171-2179), ``pressure``
  (:869-877, :2051-2056).

For an ALL-``utci_only`` met-only batch the closure narrows to
``{utci @ changed timesteps}`` (planner), and the executor recomputes
exactly those planes from the scenario's PUBLISHED prior results (tmrt@t
from the (node, time) store) through the SAME utci code path — bitwise by
construction — and publishes a SPARSE per-timestep patch (non-prefix
``time_indices``, unlike W1's prefix truncation). Any unmet condition is
a typed, telemetry-visible refusal that routes today's full solve.

Fast suite (unmarked): affinity pinning, planner narrowing + negative
routing, sparse patch contract, executor fast-path wiring against stubbed
establishing solves. ``scientific`` suite: real-SVF bitwise differential
vs the physically-edited-met-file oracle, plus perf with host load. The
real ``site_500`` differential is env-gated (SOLWEIG_SITE_500_FAST_DIFF=1)
because it costs two 500x500 full solves.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from solweig_gpu.incremental.edit_registry import (
    METEOROLOGY_VARIABLES_SAFE,
    MET_UTCI_ONLY_VARIABLES,
    MET_VARIABLE_AFFINITY,
    MET_VARIABLE_AFFINITY_EVIDENCE,
)
from solweig_gpu.incremental.edit_graph import SceneGraphState, default_edit_graph
from solweig_gpu.incremental.edit_types import (
    EditCommand,
    ForcingChange,
    ForcingDelta,
    RasterGrid,
    RasterWindow,
    SpatialScope,
    TemporalScope,
    ValidatedEdit,
)
from solweig_gpu.incremental.planner import PlanningError
from solweig_gpu.incremental.result import (
    PatchError,
    ResultPatch,
    discard_staging,
    load_patch,
    load_patch_metadata,
    publish_staged_patch,
    stage_patch,
)
from solweig_gpu.incremental.store import CoverageStatus
from solweig_gpu.utci_process import recompute_utci_steps

from tests.test_incremental_worker import (
    DATE_STR,
    LOCAL_ALWAYS,
    TINY_COLS,
    TINY_ORIGIN,
    TINY_PIXEL,
    TINY_ROWS,
    _build_cache,
    _compute_baseline_svf,
    _make_prepared_site,
    _make_tiny_site,
)

# Reuse the server end-to-end harness's REAL-SVF prepared site for the
# sparse-scatter regression (C1): importing the fixture at MODULE scope is
# what registers it with pytest (a function-local import would not).
from tests.test_server_universal_edits import family_site  # noqa: F401

MET_ADAPTER_ID = "meteorological_forcing"
SCI_VARIABLES = ("utci", "tmrt", "shadow")

GRID = RasterGrid(64, 64, 2.0)
FULL_WINDOW = RasterWindow(0, 64, 0, 64)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def met_edit(
    variable: str,
    time_index: int,
    new_value: float,
    *,
    old_value: float = 20.0,
    times: tuple[int, ...] | None = None,
    outputs: tuple[str, ...] = ("utci",),
    edit_id: str = "met-1",
) -> ValidatedEdit:
    """A validated met edit carrying one ForcingChange (engine-level build,
    mirroring the adapter's delta shape — the planner schema-checks it)."""
    return ValidatedEdit(
        command=EditCommand(
            edit_id=edit_id,
            scenario_id="scenario-a",
            base_scene_revision=0,
            adapter_id=MET_ADAPTER_ID,
            operation="update_time_row",
            old_state={variable: old_value},
            new_state={"time_index": time_index, "values": {variable: new_value}},
            requested_outputs=outputs,
            requested_times=(time_index,) if times is None else times,
        ),
        adapter_id=MET_ADAPTER_ID,
        schema_version=1,
        source_node_id="meteorology",
        delta=ForcingDelta(
            "meteorology",
            MET_ADAPTER_ID,
            (ForcingChange(variable, time_index, old_value, new_value),),
        ),
    )


def initial_state() -> SceneGraphState:
    return SceneGraphState.initial(default_edit_graph())


def veg_edit(edit_id: str = "veg-1") -> ValidatedEdit:
    """A validated vegetation edit inside the 64x64 planner grid."""
    from solweig_gpu.incremental.edit_types import (
        ObjectStateChange,
        VegetationObjectDelta,
    )

    return ValidatedEdit(
        command=EditCommand(
            edit_id=edit_id,
            scenario_id="scenario-a",
            base_scene_revision=0,
            adapter_id="vegetation_geometry",
            operation="add",
            old_state=None,
            new_state={"x_m": 80.0, "y_m": 80.0, "height_m": 8.0},
            requested_outputs=("utci",),
            requested_times=(3,),
        ),
        adapter_id="vegetation_geometry",
        schema_version=1,
        source_node_id="vegetation_dsm",
        delta=VegetationObjectDelta(
            source_node_id="vegetation_dsm",
            adapter_id="vegetation_geometry",
            objects=(ObjectStateChange("tree-1", None, {"x_m": 80.0, "y_m": 80.0, "height_m": 8.0}),),
            windows=(RasterWindow(8, 24, 8, 24),),
        ),
    )


def make_planner(**kwargs):
    from solweig_gpu.incremental.edit_registry import builtin_registry
    from solweig_gpu.incremental.planner import EditPlanner

    defaults = dict(grid=GRID, registry=builtin_registry())
    defaults.update(kwargs)
    return EditPlanner(**defaults)


def _baseline_table(site: Path) -> np.ndarray:
    met_path = site / "metfiles" / f"metfile_0_0_{DATE_STR}.txt"
    return np.loadtxt(met_path, skiprows=1, delimiter=" ")


def _tiny_executor(tmp_path: Path):
    from solweig_gpu.incremental.executor import PlanExecutor
    from solweig_gpu.incremental.trees import TreeLayer

    grid, site = _make_tiny_site(tmp_path / "site")
    cache = _build_cache(
        site,
        tmp_path / "cache",
        met_path=site / "metfiles" / f"metfile_0_0_{DATE_STR}.txt",
        site_id="tiny",
    )
    executor = PlanExecutor(
        cache=cache,
        layer=TreeLayer(cache.tree_base, grid),
        site_dir=site,
        results_root=tmp_path / "results",
        selected_date_str=DATE_STR,
        influence_config=LOCAL_ALWAYS,
    )
    return executor, site, grid, cache


def _executor_met_edit(
    executor,
    *,
    time_index: int,
    value: float,
    variable: str = "wind_speed",
    old_value: float = 2.0,
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
    del old_value  # the executor path builds deltas through the adapter
    return executor.validate(command)


def _stub_full_patch(variables, job_id, revision, window, rows, cols, *, mode="full"):
    arrays = {
        name: np.zeros((3, window.height, window.width), dtype=np.float32)
        for name in variables
    }
    # Distinct, physically sane tmrt planes per timestep (25 + 5t degC).
    if "tmrt" in arrays:
        for t in range(3):
            arrays["tmrt"][t] = 25.0 + 5.0 * t
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


def _stub_solve_full(variables, grid):
    def solve(job_id, revision, forcing):
        return [
            _stub_full_patch(
                variables, job_id, revision, grid.full_window, grid.rows, grid.cols
            )
        ]

    return solve


def _executor_veg_edit(
    executor,
    *,
    outputs=("utci", "tmrt"),
    times=(1,),
    edit_id="veg-1",
):
    """A validated vegetation add at the tile's interior (the same edit
    test_mixed_family_batch_routes_full uses), executor-level."""
    from solweig_gpu.incremental.edit_types import EditCommand

    return executor.validate(
        EditCommand(
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
            requested_outputs=outputs,
            requested_times=times,
        )
    )


def _stub_window_patch(variables, job_id, revision, window, *, tmrt_offset=0.0):
    """A windowed local-solve patch whose tmrt is DISTINGUISHABLE from the
    establishing full patch (G1.1: the composed plane must provably mix the
    new window values with the retained rev-N remainder values, so equal
    stub values everywhere would make the composition unobservable)."""
    arrays = {
        name: np.zeros((3, window.height, window.width), dtype=np.float32)
        for name in variables
    }
    if "tmrt" in arrays:
        for t in range(3):
            arrays["tmrt"][t] = 25.0 + 5.0 * t + tmrt_offset
    return ResultPatch(
        job_id=job_id,
        scene_revision=revision,
        mode="local",
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


def _load_avgs() -> dict:
    out = subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()
    tail = out.split("load averages:")[-1]
    vals = [float(x) for x in tail.split()]
    return {"uptime_text": out, "load1": vals[0], "load5": vals[1], "load15": vals[2]}


# ---------------------------------------------------------------------------
# Step 0: the affinity table is pinned (and proven from code)
# ---------------------------------------------------------------------------


class TestMetVariableAffinity:
    def test_affinity_mapping_pinned(self) -> None:
        assert MET_VARIABLE_AFFINITY == {
            "air_temperature": "radiation_affecting",
            "humidity": "radiation_affecting",
            "pressure": "radiation_affecting",
            "radiation": "radiation_affecting",
            "uhii": "utci_only",
            "wind_speed": "utci_only",
        }
        assert MET_UTCI_ONLY_VARIABLES == frozenset({"wind_speed", "uhii"})
        assert set(MET_VARIABLE_AFFINITY) == set(METEOROLOGY_VARIABLES_SAFE)

    def test_every_affinity_carries_fileline_evidence(self) -> None:
        assert set(MET_VARIABLE_AFFINITY_EVIDENCE) == set(MET_VARIABLE_AFFINITY)
        for variable, affinity in MET_VARIABLE_AFFINITY.items():
            evidence = MET_VARIABLE_AFFINITY_EVIDENCE[variable]
            if affinity == "utci_only":
                # The claim "enters ONLY the comfort calculation" is proven by
                # utci_process.py consumption sites.
                assert "utci_process.py:" in evidence, variable
            else:
                # Radiation affinity is proven by solweig.py physics sites.
                assert "solweig.py:" in evidence, variable

    def test_adapter_declares_the_same_affinity(self) -> None:
        from solweig_gpu.incremental.adapters.met_time import (
            MET_VARIABLE_AFFINITY as ADAPTER_AFFINITY,
        )

        assert ADAPTER_AFFINITY == MET_VARIABLE_AFFINITY

    def test_unknown_affinity_value_is_refused(self) -> None:
        # L1 (r3a review): the drift guards pin KEY sets only; a VALUE typo
        # ("utci-only") would import clean and silently classify the
        # variable into no bucket. The import-time validator must refuse
        # any value outside the proven vocabulary.
        from solweig_gpu.incremental.edit_registry import (
            MET_VARIABLE_AFFINITY_EVIDENCE as EVIDENCE,
            AdapterRegistryError,
            _validate_met_affinity,
        )

        for typo in ("utci-only", "comfort", "", "UTCI_ONLY"):
            bad = dict(MET_VARIABLE_AFFINITY)
            bad["wind_speed"] = typo
            with pytest.raises(AdapterRegistryError, match="affinity"):
                _validate_met_affinity(bad, EVIDENCE)
        # The shipped table itself validates clean.
        _validate_met_affinity(MET_VARIABLE_AFFINITY, EVIDENCE)


# ---------------------------------------------------------------------------
# Step 1: planner narrows the met closure for all-utci_only met-only batches
# ---------------------------------------------------------------------------


class TestPlannerMetFastNarrowing:
    def test_wind_only_batch_narrows_to_utci_window_impact(self) -> None:
        plan = make_planner().plan([met_edit("wind_speed", 1, 5.0)], initial_state())
        assert plan.changed_sources == ("meteorology",)
        assert len(plan.node_impacts) == 1
        impact = plan.node_impacts[0]
        assert impact.node_id == "utci"
        assert impact.spatial_scope is SpatialScope.WINDOWS
        assert impact.write_windows == (FULL_WINDOW,)
        assert impact.read_windows == (FULL_WINDOW,)
        assert impact.temporal_scope is TemporalScope.ONE
        assert impact.time_start == 1
        assert "utci_only" in impact.reason
        # The radiation/shadow/thermal closure stays reusable: the changed
        # variables provably never enter it.
        for reusable in (
            "solar_atmospheric_state",
            "time_shadow",
            "radiation",
            "surface_thermal_state",
            "tmrt",
            "svf",
        ):
            assert reusable in plan.reusable_nodes, reusable
        assert "meteorology" not in plan.reusable_nodes
        assert "utci" not in plan.reusable_nodes
        # wbgt stays never-planned and is NOT silently reusable.
        assert "wbgt" not in plan.reusable_nodes
        assert "wbgt" not in {i.node_id for i in plan.node_impacts}

    def test_multiple_changed_times_plan_range(self) -> None:
        edits = [
            met_edit("wind_speed", 1, 5.0, edit_id="m1"),
            met_edit("uhii", 3, 2.0, old_value=0.0, edit_id="m2"),
        ]
        plan = make_planner().plan(edits, initial_state())
        impact = plan.node_impacts[0]
        assert impact.node_id == "utci"
        assert impact.temporal_scope is TemporalScope.RANGE
        assert (impact.time_start, impact.time_stop) == (1, 3)

    def test_requested_times_do_not_extend_fast_path_bounds(self) -> None:
        # R1's union rule exists because radiation_affecting met changes carry
        # stateful ground heat forward; utci_only changes carry NOTHING
        # forward, so a requested-but-unchanged timestep stays cached.
        edit = met_edit("wind_speed", 1, 5.0, times=(1, 3))
        plan = make_planner().plan([edit], initial_state())
        impact = plan.node_impacts[0]
        assert impact.temporal_scope is TemporalScope.ONE
        assert impact.time_start == 1

    def test_narrowing_is_policy_independent(self) -> None:
        # There is no locality to deny: the affinity proof makes the dirty
        # closure exactly {utci@t} and the write is already tile-wide, so a
        # strict safety policy must not demote it to FULL.
        from solweig_gpu.incremental.planner import ConservativeSafetyPolicy

        for fraction in (0.30, 0.10, 1.0):
            planner = make_planner(
                policy=ConservativeSafetyPolicy(full_recompute_fraction=fraction)
            )
            plan = planner.plan([met_edit("wind_speed", 1, 5.0)], initial_state())
            impact = plan.node_impacts[0]
            assert impact.spatial_scope is SpatialScope.WINDOWS, fraction
            assert impact.write_windows == (FULL_WINDOW,), fraction

    def test_air_temperature_stays_full_closure(self) -> None:
        plan = make_planner().plan(
            [met_edit("air_temperature", 1, 30.0, old_value=22.0)], initial_state()
        )
        scopes = {i.node_id: i.spatial_scope for i in plan.node_impacts}
        assert "utci" in scopes
        for node_id in (
            "solar_atmospheric_state",
            "time_shadow",
            "radiation",
            "surface_thermal_state",
            "tmrt",
            "utci",
        ):
            assert scopes[node_id] is SpatialScope.FULL, node_id
        assert "tmrt" not in plan.reusable_nodes

    def test_mixed_affinity_variables_stay_full_closure(self) -> None:
        edits = [
            met_edit("wind_speed", 1, 5.0, edit_id="m1"),
            met_edit("air_temperature", 2, 30.0, old_value=22.0, edit_id="m2"),
        ]
        plan = make_planner().plan(edits, initial_state())
        scopes = {i.node_id: i.spatial_scope for i in plan.node_impacts}
        assert scopes["utci"] is SpatialScope.FULL
        assert scopes["tmrt"] is SpatialScope.FULL
        assert "radiation" in scopes

    def test_mixed_family_batch_stays_full_closure(self) -> None:
        plan = make_planner().plan(
            [veg_edit(), met_edit("wind_speed", 3, 5.0, edit_id="met-1")],
            initial_state(),
        )
        scopes = {i.node_id: i.spatial_scope for i in plan.node_impacts}
        assert scopes["utci"] is SpatialScope.FULL
        assert scopes["tmrt"] is SpatialScope.FULL

    def test_tmrt_only_request_is_not_narrowed(self) -> None:
        # The fast path serves utci; a batch that does not keep utci keeps
        # today's conservative full closure.
        plan = make_planner().plan(
            [met_edit("wind_speed", 1, 5.0, outputs=("tmrt",))], initial_state()
        )
        assert "utci" not in {i.node_id for i in plan.node_impacts}

    def test_wbgt_request_still_refused(self) -> None:
        with pytest.raises(PlanningError, match="never be planned"):
            make_planner().plan(
                [met_edit("wind_speed", 1, 5.0, outputs=("wbgt",))], initial_state()
            )

    def test_full_only_utci_policy_disables_narrowing(self) -> None:
        # M1 (r3a review): ``full_only_nodes`` is the DOCUMENTED
        # solver-capability contract (a deployment pins a node there when
        # the fast evaluation must not run). The affinity narrowing must
        # respect it — policy-as-contract beats the proof — while the
        # demote-to-FULL heuristic lever (full_recompute_fraction) keeps
        # its documented policy independence above.
        from solweig_gpu.incremental.planner import ConservativeSafetyPolicy

        planner = make_planner(
            policy=ConservativeSafetyPolicy(full_only_nodes={"utci"})
        )
        plan = planner.plan([met_edit("wind_speed", 1, 5.0)], initial_state())
        scopes = {i.node_id: i.spatial_scope for i in plan.node_impacts}
        assert scopes["utci"] is SpatialScope.FULL, (
            "full_only_nodes={'utci'} must disable the fast narrowing"
        )
        assert scopes["tmrt"] is SpatialScope.FULL
        assert "tmrt" not in plan.reusable_nodes


# ---------------------------------------------------------------------------
# Step 2a: sparse (non-prefix) result patches
# ---------------------------------------------------------------------------


def _sparse_patch(time_indices: tuple[int, ...]) -> ResultPatch:
    window = RasterWindow(0, 8, 0, 8)
    arrays = {
        "utci": np.arange(
            len(time_indices) * 64, dtype=np.float32
        ).reshape(len(time_indices), 8, 8)
    }
    return ResultPatch(
        job_id="job-sparse",
        scene_revision=2,
        mode="local",
        write_window=window,
        read_window=window,
        site_id="site",
        tile_key="0_0",
        cache_manifest_sha256="0" * 64,
        model_version="test",
        variables=("utci",),
        arrays=arrays,
        time_start=time_indices[0] if time_indices else 0,
        time_stop=(time_indices[-1] + 1) if time_indices else 1,
        time_indices=time_indices,
    )


class TestSparseResultPatch:
    def test_stage_publish_load_roundtrip(self, tmp_path: Path) -> None:
        patch = _sparse_patch((1, 3))
        staged = stage_patch(patch, tmp_path)
        published = publish_staged_patch(staged, tmp_path, patch.scene_revision)
        loaded = load_patch(published)
        assert loaded.time_indices == (1, 3)
        assert loaded.n_time_steps == 2
        np.testing.assert_array_equal(loaded.variable("utci"), patch.arrays["utci"])
        metadata = load_patch_metadata(published)
        assert metadata.time_indices == (1, 3)
        assert metadata.n_time_steps == 2

    def test_apply_into_scatters_rows_to_declared_times(self) -> None:
        patch = _sparse_patch((1, 3))
        target = {"utci": np.zeros((5, 8, 8), dtype=np.float32)}
        before = target["utci"].copy()
        patch.apply_into(target)
        np.testing.assert_array_equal(target["utci"][1], patch.arrays["utci"][0])
        np.testing.assert_array_equal(target["utci"][3], patch.arrays["utci"][1])
        # Absent timesteps are NEVER written: consumers must not read them
        # as zeros produced by this patch.
        for absent in (0, 2, 4):
            np.testing.assert_array_equal(
                target["utci"][absent], before[absent], err_msg=f"t={absent}"
            )

    def test_legacy_prefix_patch_loads_with_none_indices(self, tmp_path: Path) -> None:
        window = RasterWindow(0, 4, 0, 4)
        arrays = {"utci": np.ones((3, 4, 4), dtype=np.float32)}
        patch = ResultPatch(
            job_id="job-legacy",
            scene_revision=1,
            mode="full",
            write_window=window,
            read_window=window,
            site_id="site",
            tile_key="0_0",
            cache_manifest_sha256="0" * 64,
            model_version="test",
            variables=("utci",),
            arrays=arrays,
            time_start=0,
            time_stop=3,
        )
        published = publish_staged_patch(stage_patch(patch, tmp_path), tmp_path, 1)
        loaded = load_patch(published)
        assert loaded.time_indices is None
        assert loaded.n_time_steps == 3
        assert load_patch_metadata(published).time_indices is None

    def test_invalid_time_indices_rejected(self) -> None:
        with pytest.raises(PatchError, match="strictly increasing"):
            _sparse_patch((3, 1))
        with pytest.raises(PatchError, match="non-empty"):
            _sparse_patch(())
        window = RasterWindow(0, 4, 0, 4)
        with pytest.raises(PatchError, match="shape"):
            ResultPatch(
                job_id="job-bad",
                scene_revision=1,
                mode="local",
                write_window=window,
                read_window=window,
                site_id="site",
                tile_key="0_0",
                cache_manifest_sha256="0" * 64,
                model_version="test",
                variables=("utci",),
                arrays={"utci": np.ones((3, 4, 4), dtype=np.float32)},
                time_start=1,
                time_stop=None,
                time_indices=(1, 2),  # two indices, three rows
            )
        with pytest.raises(PatchError, match="time_start"):
            ResultPatch(
                job_id="job-bad",
                scene_revision=1,
                mode="local",
                write_window=window,
                read_window=window,
                site_id="site",
                tile_key="0_0",
                cache_manifest_sha256="0" * 64,
                model_version="test",
                variables=("utci",),
                arrays={"utci": np.ones((2, 4, 4), dtype=np.float32)},
                time_start=0,
                time_stop=None,
                time_indices=(1, 2),  # starts after time_start
            )


# ---------------------------------------------------------------------------
# Step 2b: utci-only recompute helper (same ops, same order)
# ---------------------------------------------------------------------------


class TestRecomputeUtciSteps:
    def _scene(self):
        rows = cols = 16
        dsm = np.zeros((rows, cols), dtype=np.float32)
        dsm[4:8, 4:8] = 6.0  # building (invalid cells)
        dem = np.zeros((rows, cols), dtype=np.float32)
        met = np.zeros((3, 25), dtype=np.float64)
        met[:, 9] = 2.0   # wind
        met[:, 10] = 50.0  # RH
        met[:, 11] = 25.0  # Ta
        met[:, 24] = 0.0   # uhii
        planes = {
            t: np.full((rows, cols), 40.0 + 5.0 * t, dtype=np.float32)
            for t in range(3)
        }
        return met, planes, dsm, dem

    def test_shape_dtype_and_valid_mask(self) -> None:
        met, planes, dsm, dem = self._scene()
        out = recompute_utci_steps(met, (1, 2), planes, dsm, dem)
        assert out.shape == (2, 16, 16)
        assert out.dtype == np.float32
        # Building cells are NaN exactly like the solver's outputs.
        assert np.isnan(out[0][4:8, 4:8]).all()
        assert np.isfinite(out[0][0, 0])

    def test_wind_clamped_at_015_and_uhii_shifts_result(self) -> None:
        met, planes, dsm, dem = self._scene()
        calm_met = met.copy()
        calm_met[1, 9] = 0.0  # calm: va10m clamps to 0.15
        calm = recompute_utci_steps(calm_met, (1,), planes, dsm, dem)[0]
        breezy = recompute_utci_steps(met, (1,), planes, dsm, dem)[0]
        assert not np.array_equal(calm, breezy, equal_nan=True)
        # uhii adds to the Ta plane at exactly the edited timestep.
        met3 = met.copy()
        met3[1, 24] = 3.0
        with_uhii = recompute_utci_steps(met3, (1, 2), planes, dsm, dem)
        without = recompute_utci_steps(met, (1, 2), planes, dsm, dem)
        assert not np.array_equal(with_uhii[0], without[0], equal_nan=True)
        assert np.array_equal(with_uhii[1], without[1], equal_nan=True)

    def test_timestep_isolation(self) -> None:
        met, planes, dsm, dem = self._scene()
        edited = met.copy()
        edited[2, 9] = 9.0
        a = recompute_utci_steps(met, (0, 2), planes, dsm, dem)
        b = recompute_utci_steps(edited, (0, 2), planes, dsm, dem)
        assert np.array_equal(a[0], b[0], equal_nan=True)  # t=0 untouched
        assert not np.array_equal(a[1], b[1], equal_nan=True)


# ---------------------------------------------------------------------------
# Step 2c: executor fast path (establishing solve stubbed)
# ---------------------------------------------------------------------------


class TestExecutorMetFastPath:
    def _fast_executor(self, tmp_path: Path):
        executor, site, grid, cache = _tiny_executor(tmp_path)
        executor._worker._solve_full = _stub_solve_full(
            ("shadow", "tmrt", "utci"), grid
        )
        return executor, site, grid, cache

    def test_first_utci_only_edit_routes_full_without_coverage(self, tmp_path):
        # No published tmrt yet: the fast path refuses (typed, telemetry-
        # visible) and today's full solve runs instead.
        ex, site, grid, _cache = self._fast_executor(tmp_path)
        executed = ex.execute([_executor_met_edit(ex, time_index=1, value=5.0)])
        assert executed.published
        assert executed.mode == "full"
        assert executed.diagnostics.get("met_fast_path_refusal")

    def test_utci_only_edit_after_publication_routes_fast(self, tmp_path):
        ex, site, grid, cache = self._fast_executor(tmp_path)
        first = ex.execute([_executor_met_edit(ex, time_index=1, value=5.0)])
        assert first.mode == "full"

        second = ex.execute(
            [_executor_met_edit(ex, time_index=1, value=7.5, edit_id="met-2")]
        )
        assert second.published
        assert second.mode == "local"
        assert second.diagnostics.get("met_fast_path") is True
        assert second.diagnostics.get("met_fast_path_times") == (1,)

        patch = load_patch(second.patch_paths[0])
        assert patch.variables == ("utci",)
        assert patch.time_indices == (1,)
        assert patch.n_time_steps == 1

        # The published plane equals the same-code-path recompute from the
        # staged overlay at t and the ESTABLISHING publication's tmrt@t.
        baseline = _baseline_table(site)
        staged = ex._applied_forcing_overlay.resolve(baseline)
        establishing = load_patch(first.patch_paths[0])
        expected = recompute_utci_steps(
            staged,
            (1,),
            {1: establishing.variable("tmrt")[1]},
            np.array(cache.building_dsm),
            np.array(cache.dem),
        )
        np.testing.assert_array_equal(patch.variable("utci")[0], expected[0])

        # Store: the changed time advanced; earlier times keep provenance.
        assert ex.store.revision_at("utci", 1) == 2
        assert ex.store.revision_at("utci", 0) == 1
        assert ex.store.revision_at("tmrt", 1) == 1  # untouched by the fast path
        entry = ex.store.lookup("utci", 1)
        assert entry is not None and entry.patch_path == second.patch_paths[0]

        # The establishing tmrt plane is bitwise what the fast path read.
        np.testing.assert_array_equal(
            establishing.variable("tmrt")[1],
            load_patch(ex.store.lookup("tmrt", 1).patch_path).variable("tmrt")[1],
        )

    def test_fast_patch_covers_exactly_changed_times(self, tmp_path):
        ex, _site, _grid, _cache = self._fast_executor(tmp_path)
        ex.execute([_executor_met_edit(ex, time_index=1, value=5.0)])
        executed = ex.execute(
            [
                _executor_met_edit(ex, time_index=0, value=4.0, edit_id="m0"),
                _executor_met_edit(ex, time_index=2, value=6.0, edit_id="m2"),
            ]
        )
        assert executed.mode == "local"
        patch = load_patch(executed.patch_paths[0])
        assert patch.time_indices == (0, 2)
        for time_index in (0, 1, 2):
            assert ex.store.revision_at("utci", time_index) == (
                2 if time_index in (0, 2) else 1
            )

    def test_consecutive_fast_edits_stay_fast(self, tmp_path):
        ex, _site, _grid, _cache = self._fast_executor(tmp_path)
        ex.execute([_executor_met_edit(ex, time_index=1, value=5.0)])
        second = ex.execute(
            [_executor_met_edit(ex, time_index=1, value=7.5, edit_id="m2")]
        )
        third = ex.execute(
            [_executor_met_edit(ex, time_index=2, value=8.0, edit_id="m3")]
        )
        assert second.mode == "local" and third.mode == "local"
        assert load_patch(third.patch_paths[0]).time_indices == (2,)

    def test_watermarks_adopted_after_fast_publication(self, tmp_path):
        # Without the adoption the staged forcing overlay stays pending and
        # every later job would route full-tile.
        ex, _site, _grid, _cache = self._fast_executor(tmp_path)
        ex.execute([_executor_met_edit(ex, time_index=1, value=5.0)])
        ex.execute([_executor_met_edit(ex, time_index=1, value=7.5, edit_id="m2")])
        watermarks = ex._worker.publication_watermarks
        assert watermarks.forcing_overlay == ex._applied_forcing_overlay

    def test_radiation_variable_routes_full(self, tmp_path):
        # The edit lands at timestep 0: the G2.1 warm sparse path refuses
        # (no unchanged prefix to serve from the store), so the batch takes
        # today's exact full solve through the stubbed worker — keeping
        # this r3a fence testable on the tiny site. The warm path's own
        # routing (a row > 0 radiation edit serves its prefix and solves
        # the suffix warm) needs solve_window-real physics and is covered
        # end-to-end on the prepared site by tests/test_incremental_met_warm_path.py.
        ex, _site, _grid, _cache = self._fast_executor(tmp_path)
        ex.execute([_executor_met_edit(ex, time_index=1, value=5.0)])
        executed = ex.execute(
            [
                _executor_met_edit(
                    ex, time_index=0, value=30.0, variable="air_temperature",
                    edit_id="m2",
                )
            ]
        )
        assert executed.mode == "full"
        assert not executed.diagnostics.get("met_fast_path")
        # The warm consumer refused the row-0 edit and handed the batch to
        # the ordinary full solve — typed, telemetry-visible.
        assert "no unchanged prefix" in executed.diagnostics.get(
            "met_warm_path_refusal", ""
        )

    def test_mixed_family_batch_routes_full(self, tmp_path):
        from solweig_gpu.incremental.edit_types import EditCommand

        ex, _site, grid, _cache = self._fast_executor(tmp_path)
        ex.execute([_executor_met_edit(ex, time_index=1, value=5.0)])
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
        met = _executor_met_edit(ex, time_index=1, value=7.5, edit_id="m2")
        executed = ex.execute([veg, met])
        assert executed.mode == "full"

    def _mixed_provenance_state(self, tmp_path: Path, *, tmrt_offset: float = 7.0):
        """Full-tile tmrt publication @rev1, then a WINDOWED vegetation
        batch @rev2 whose patch carries ``tmrt_offset`` inside its write
        window: tmrt@1 ends FULL-coverage with MIXED provenance {1, 2},
        the rev-1 entry retained as narrowed cell remainders (R5a)."""
        ex, site, grid, cache = self._fast_executor(tmp_path)
        first = ex.execute([_executor_met_edit(ex, time_index=1, value=5.0)])
        assert first.mode == "full"

        def fake_local(job_id, revision, forcing, write_windows, **kwargs):
            return [
                _stub_window_patch(
                    ("shadow", "tmrt", "utci"),
                    job_id,
                    revision,
                    write_windows[0],
                    tmrt_offset=tmrt_offset,
                )
            ]

        ex._worker._solve_local = fake_local
        veg_run = ex.execute([_executor_veg_edit(ex)])
        assert veg_run.mode == "local"
        return ex, site, grid, cache, first, veg_run

    def test_windowed_tmrt_publication_composes_and_serves(self, tmp_path):
        # G1.1: after full-tile@N + windowed@N+1 the tmrt plane is
        # FULL-coverage mixed {N, N+1}; the fast path SERVES the per-cell
        # composition (new patch values inside the batch window, rev-N
        # values in the retained remainders) instead of refusing mixed
        # provenance. G1.0's superset differential proved this composition
        # bitwise equals the revision-head full recompute, tile-wide.
        (ex, site, grid, cache, first, veg_run) = self._mixed_provenance_state(
            tmp_path
        )

        met = _executor_met_edit(ex, time_index=1, value=7.5, edit_id="m2")
        executed = ex.execute([met])
        assert executed.published
        assert executed.mode == "local"
        assert executed.diagnostics.get("met_fast_path") is True
        assert executed.diagnostics.get("met_fast_path_times") == (1,)

        # The fast-path result contract is unchanged by the relaxation.
        patch = load_patch(executed.patch_paths[0])
        assert patch.variables == ("utci",)
        assert patch.time_indices == (1,)
        assert patch.write_window == grid.full_window

        # Expected composition from the REAL payloads: establishing plane
        # tile-wide, the window patch's values inside the veg write window.
        establishing = load_patch(first.patch_paths[0])
        window_patch = load_patch(veg_run.patch_paths[0])
        window = window_patch.write_window
        composed = establishing.variable("tmrt")[1].copy()
        composed[
            window.row_start : window.row_stop, window.col_start : window.col_stop
        ] = window_patch.variable("tmrt")[1]

        baseline = _baseline_table(site)
        staged = ex._applied_forcing_overlay.resolve(baseline)
        expected = recompute_utci_steps(
            staged,
            (1,),
            {1: composed},
            np.array(cache.building_dsm),
            np.array(cache.dem),
        )
        np.testing.assert_array_equal(patch.variable("utci")[0], expected[0])

        # The relaxation never touches the radiation products: tmrt@1
        # keeps its mixed provenance and its rev-2 head.
        assert ex.store.revision_at("tmrt", 1) == 2
        assert ex.store.revision_at("utci", 1) == 3

    def test_interior_window_retention_paints_every_remainder(self, tmp_path):
        # G1.1 dedup regression pin: a full-tile patch with an INTERIOR
        # batch window leaves exactly FOUR narrowed remainder entries
        # sharing ONE patch_path. The serve must paint ALL of them — the
        # pre-G1.1 seen_patches dedup skipped every entry after the first
        # sharing a patch path and would leave 3 of 4 remainders (here
        # hundreds of cells) silently NaN inside the composed plane.
        (ex, site, _grid, cache, first, veg_run) = self._mixed_provenance_state(
            tmp_path
        )

        entries = ex.store.window_entries("tmrt", 1)
        retained = [e for e in entries if e.scene_revision == 1]
        assert len(retained) == 4, "interior window: exactly four remainders"
        assert len({e.patch_path for e in retained}) == 1, (
            "the four remainders share the superseded full-tile patch"
        )

        met = _executor_met_edit(ex, time_index=1, value=7.5, edit_id="m2")
        executed = ex.execute([met])
        assert executed.mode == "local"

        establishing = load_patch(first.patch_paths[0])
        window_patch = load_patch(veg_run.patch_paths[0])
        window = window_patch.write_window
        # Anti-vacuity: the window really carries different tmrt values
        # than the retained remainder base.
        assert not np.array_equal(
            establishing.variable("tmrt")[1][
                window.row_start : window.row_stop,
                window.col_start : window.col_stop,
            ],
            window_patch.variable("tmrt")[1],
        )
        composed = establishing.variable("tmrt")[1].copy()
        composed[
            window.row_start : window.row_stop, window.col_start : window.col_stop
        ] = window_patch.variable("tmrt")[1]
        baseline = _baseline_table(site)
        staged = ex._applied_forcing_overlay.resolve(baseline)
        expected = recompute_utci_steps(
            staged,
            (1,),
            {1: composed},
            np.array(cache.building_dsm),
            np.array(cache.dem),
        )
        served = load_patch(executed.patch_paths[0]).variable("utci")[0]
        np.testing.assert_array_equal(served, expected[0])

    def test_partial_tmrt_coverage_still_refuses(self, tmp_path):
        # G1.1 fence: a windowed publication with NO full-tile base leaves
        # tmrt@t PARTIAL — the relaxation serves per-cell LATEST validity,
        # never partial coverage.
        ex, _site, grid, _cache = self._fast_executor(tmp_path)

        def fake_local(job_id, revision, forcing, write_windows, **kwargs):
            return [
                _stub_window_patch(
                    ("shadow", "tmrt", "utci"), job_id, revision, write_windows[0]
                )
            ]

        ex._worker._solve_local = fake_local
        veg_run = ex.execute([_executor_veg_edit(ex)])
        assert veg_run.mode == "local"

        coverage = ex.store.window_coverage("tmrt", 1, window=grid.full_window)
        assert coverage.status is CoverageStatus.PARTIAL

        met = _executor_met_edit(ex, time_index=1, value=7.5)
        executed = ex.execute([met])
        assert executed.mode == "full"
        refusal = executed.diagnostics.get("met_fast_path_refusal")
        assert refusal and "partial" in refusal

    def test_value_equal_overlay_is_a_clean_noop(self, tmp_path):
        ex, _site, _grid, _cache = self._fast_executor(tmp_path)
        ex.execute([_executor_met_edit(ex, time_index=1, value=5.0)])
        again = ex.execute(
            [_executor_met_edit(ex, time_index=1, value=5.0, edit_id="m2")]
        )
        assert again.status == "no-op"


# ---------------------------------------------------------------------------
# Scientific differential: real physics, bitwise vs the ground-truth oracle
# ---------------------------------------------------------------------------

SCI_ROWS = SCI_COLS = 128
SCI_PIXEL = 2.0
SCI_ORIGIN = (300000.0, 4100000.0)
SCI_EPSG = 32616


def _edit_met_cell(met_path: Path, *, row: int, column: int, value: float) -> None:
    """Physically edit ONE met cell, preserving every other token's text."""
    lines = met_path.read_text().splitlines()
    tokens = lines[row + 1].split()
    tokens[column] = f"{value:.17g}"
    lines[row + 1] = " ".join(tokens)
    met_path.write_text("\n".join(lines) + "\n")


def _full_run(cache, site, grid, scratch: Path) -> dict[str, np.ndarray]:
    from solweig_gpu.incremental.solver import load_site_forcing, run_full_tile
    from solweig_gpu.incremental.trees import TreeLayer

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
def fast_site(tmp_path_factory):
    """One real-SVF 128x128 site (the met_integration met_site pattern)."""
    from solweig_gpu.incremental.trees import TreeSpec

    root = tmp_path_factory.mktemp("met_fast_site")
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
        site_id="met-fast",
    )
    return SimpleNamespace(grid=grid, site=site, cache=cache)


def _fast_executor_for(fast_site, tmp_path: Path):
    from solweig_gpu.incremental.executor import PlanExecutor
    from solweig_gpu.incremental.trees import TreeLayer

    return PlanExecutor(
        cache=fast_site.cache,
        layer=TreeLayer(fast_site.cache.tree_base, fast_site.grid),
        site_dir=fast_site.site,
        results_root=tmp_path / "results",
        selected_date_str=DATE_STR,
        influence_config=LOCAL_ALWAYS,
    )


def _physically_edited_twin(fast_site, tmp_path: Path, *, column: int, row: int,
                            value: float):
    import shutil

    twin = shutil.copytree(fast_site.site, tmp_path / "edited_site")
    met_path = twin / "metfiles" / f"metfile_0_0_{DATE_STR}.txt"
    _edit_met_cell(met_path, row=row, column=column, value=value)
    twin_cache = _build_cache(
        twin,
        tmp_path / "cache_edited",
        met_path=met_path,
        site_id="met-fast-edited",
    )
    return twin, twin_cache


@pytest.mark.scientific
class TestMetFastPathScientific:
    def test_wind_fast_path_bitwise_vs_physically_edited_oracle(
        self, tmp_path: Path, fast_site
    ) -> None:
        executor = _fast_executor_for(fast_site, tmp_path)
        load = {"host_load": _load_avgs()}
        t0 = time.monotonic()
        establishing = executor.execute(
            [
                _executor_met_edit(
                    executor, time_index=1, value=5.0, outputs=SCI_VARIABLES
                )
            ]
        )
        full_seconds = time.monotonic() - t0
        assert establishing.published
        assert establishing.mode == "full"  # refused: no tmrt coverage yet

        t1 = time.monotonic()
        fast = executor.execute(
            [
                _executor_met_edit(
                    executor, time_index=1, value=7.5, outputs=SCI_VARIABLES,
                    edit_id="met-2",
                )
            ]
        )
        fast_seconds = time.monotonic() - t1
        assert fast.published
        assert fast.mode == "local"
        assert fast.diagnostics["met_fast_path"] is True
        assert fast.diagnostics["met_fast_path_times"] == (1,)
        load["host_load_after"] = _load_avgs()

        patch = load_patch(fast.patch_paths[0])
        assert patch.time_indices == (1,)
        assert patch.variables == ("utci",)

        # Ground truth: the same final met state written the pre-overlay
        # way (edit the text file, rebuild the cache), full-domain run.
        twin, twin_cache = _physically_edited_twin(
            fast_site, tmp_path, column=9, row=1, value=7.5
        )
        oracle = _full_run(twin_cache, twin, fast_site.grid, tmp_path / "oracle")

        assert np.array_equal(
            patch.variable("utci")[0], oracle["utci"][1], equal_nan=True
        ), "fast-path utci@1 differs from the physically-edited oracle"

        # Affinity evidence in the strong direction: the establishing run's
        # tmrt@1 (wind 5.0) is bitwise the oracle's tmrt@1 (wind 7.5) —
        # wind provably never enters the radiation physics.
        establishing_patch = load_patch(establishing.patch_paths[0])
        assert np.array_equal(
            establishing_patch.variable("tmrt")[1], oracle["tmrt"][1], equal_nan=True
        )
        # And the shadow series is bitwise untouched too.
        assert np.array_equal(
            establishing_patch.variable("shadow")[1],
            oracle["shadow"][1],
            equal_nan=True,
        )

        # Perf (host load recorded alongside every number).
        load["full_seconds"] = full_seconds
        load["fast_seconds"] = fast_seconds
        print(
            f"[met-fast] establishing full={full_seconds:.2f}s "
            f"fast={fast_seconds:.3f}s speedup={full_seconds / max(fast_seconds, 1e-9):.0f}x "
            f"load1={load['host_load']['load1']}"
        )
        assert fast_seconds < full_seconds

    def test_uhii_fast_path_bitwise(self, tmp_path: Path, fast_site) -> None:
        executor = _fast_executor_for(fast_site, tmp_path)
        establishing = executor.execute(
            [
                _executor_met_edit(
                    executor,
                    time_index=2,
                    value=0.0,
                    variable="uhii",
                    old_value=0.0,
                    outputs=SCI_VARIABLES,
                )
            ]
        )
        assert establishing.mode == "full"
        fast = executor.execute(
            [
                _executor_met_edit(
                    executor, time_index=2, value=3.0, variable="uhii",
                    outputs=SCI_VARIABLES, edit_id="met-2",
                )
            ]
        )
        assert fast.mode == "local"
        patch = load_patch(fast.patch_paths[0])
        assert patch.time_indices == (2,)

        twin, twin_cache = _physically_edited_twin(
            fast_site, tmp_path, column=24, row=2, value=3.0
        )
        oracle = _full_run(twin_cache, twin, fast_site.grid, tmp_path / "oracle")
        assert np.array_equal(
            patch.variable("utci")[0], oracle["utci"][2], equal_nan=True
        ), "fast-path utci@2 (uhii edit) differs from the oracle"

    def test_radiation_edit_after_fast_still_full_and_exact(
        self, tmp_path: Path, fast_site
    ) -> None:
        executor = _fast_executor_for(fast_site, tmp_path)
        established = executor.execute(
            [
                _executor_met_edit(
                    executor, time_index=1, value=5.0, outputs=SCI_VARIABLES
                )
            ]
        )
        # tmrt BEFORE the Ta edit (the establishing solve publishes it).
        established_patch = load_patch(established.patch_paths[0])
        established_tmrt = established_patch.variable("tmrt")
        executed = executor.execute(
            [
                _executor_met_edit(
                    executor, time_index=1, value=30.0, variable="air_temperature",
                    outputs=SCI_VARIABLES, edit_id="met-2",
                )
            ]
        )
        assert executed.mode == "full"
        assert not executed.diagnostics.get("met_fast_path")
        # G2.1: the radiation-affecting edit routes the warm sparse
        # consumer — the unchanged prefix (t=0) keeps its published values
        # and the suffix publishes as ONE sparse patch from the changed row.
        assert executed.diagnostics["met_warm_path"] is True
        assert executed.diagnostics["met_warm_r0"] == 1
        patch = load_patch(executed.patch_paths[0])
        assert patch.time_start == 1
        assert patch.time_stop == established_patch.time_stop

        # The accumulated overlay carries wind 5.0 AND Ta 30.0: the twin
        # encodes both cells physically and the composed serve — prefix
        # rows from the establishing publication, suffix rows from the
        # warm patch — must match it bitwise everywhere (the overlay path
        # stays exact on mixed state).
        import shutil

        twin = shutil.copytree(fast_site.site, tmp_path / "edited_site")
        met_path = twin / "metfiles" / f"metfile_0_0_{DATE_STR}.txt"
        _edit_met_cell(met_path, row=1, column=9, value=5.0)
        _edit_met_cell(met_path, row=1, column=11, value=30.0)
        twin_cache = _build_cache(
            twin,
            tmp_path / "cache_edited",
            met_path=met_path,
            site_id="met-fast-edited",
        )
        oracle = _full_run(twin_cache, twin, fast_site.grid, tmp_path / "oracle")
        r0 = patch.time_start
        for variable in SCI_VARIABLES:
            served = np.concatenate(
                [established_patch.variable(variable)[:r0], patch.variable(variable)],
                axis=0,
            )
            assert np.array_equal(
                served, oracle[variable], equal_nan=True
            ), f"{variable}: served (store prefix + warm suffix) differs from oracle"
        # Non-vacuity fence (r3a review L2): the oracle equality above is
        # only meaningful if the Ta edit actually MOVES tmrt — a silently
        # no-op edit would compare two identical stale solves and pass.
        assert not np.array_equal(
            established_tmrt[1],
            patch.variable("tmrt")[0],
            equal_nan=True,
        ), "air_temperature edit must move tmrt@1 (anti-vacuity)"


# ---------------------------------------------------------------------------
# r3a review remediation: sparse patches through the SERVER bridge
# ---------------------------------------------------------------------------


@pytest.mark.scientific
class TestServerBridgeSparseScatter:
    """C1 (r3a review, blocking): a sparse fast-path patch must land at its
    own global timestep through BOTH server state-assembly paths.

    At base the bridge scattered patches by dense offset — a patch for
    changed t=2 landed at global t=0 while t=2 kept its stale value — so
    the job served wrong arrays AND durably encoded them as its own result
    record. Reachable today: POST /edits/universal -> executor fast path.
    """

    def test_wind_fast_patch_lands_at_changed_timestep_compose_path(
        self, tmp_path: Path, family_site
    ) -> None:
        from tests.test_server_universal_edits import (
            _full_run as server_full_run,
            _make_client,
            _scenario_ready,
            _served_arrays,
            _universal,
            _validated_delta,
        )
        from solweig_gpu.incremental import overlay_from_deltas
        from test_server_api import wait_for_job

        met_table = np.loadtxt(
            family_site.site / "metfiles" / f"metfile_0_0_{DATE_STR}.txt",
            skiprows=1,
            delimiter=" ",
        )
        wind_t1 = float(round(met_table[1, 9] + 3.1, 4))
        wind_t2 = float(round(met_table[2, 9] + 4.2, 4))
        assert wind_t2 != met_table[2, 9]  # the edit is not value-equal

        with _make_client(tmp_path, family_site) as client:
            scenario = _scenario_ready(client)

            def wind_edit(time_index: int, value: float, base: int, key: str):
                return wait_for_job(
                    client,
                    _universal(
                        client,
                        scenario,
                        [
                            {
                                "adapter": "meteorological_forcing",
                                "operation": "update_time_row",
                                "time_index": time_index,
                                "values": {"wind_speed": value},
                            }
                        ],
                        base=base,
                        key=key,
                    ).json()["job_id"],
                )

            # t=1 first: full or fast, it establishes executor coverage.
            job1 = wind_edit(1, wind_t1, base=0, key="wind-c1-a")
            assert job1["status"] == "complete", job1
            # t=2 second: deterministically the sparse fast path.
            job2 = wind_edit(2, wind_t2, base=1, key="wind-c1-b")
            assert job2["status"] == "complete", job2
            assert job2["mode"] == "local", (
                "expected the sparse met fast path (utci_only wind edit); "
                f"got mode={job2['mode']!r}"
            )

            oracle = server_full_run(
                family_site.cache,
                family_site.site,
                family_site.grid,
                family_site.root / "oracle_wind_c1",
                overlay=overlay_from_deltas(
                    [
                        _validated_delta(
                            family_site,
                            {
                                "adapter": "meteorological_forcing",
                                "operation": "update_time_row",
                                "time_index": 1,
                                "values": {"wind_speed": wind_t1},
                            },
                        ),
                        _validated_delta(
                            family_site,
                            {
                                "adapter": "meteorological_forcing",
                                "operation": "update_time_row",
                                "time_index": 2,
                                "values": {"wind_speed": wind_t2},
                            },
                        ),
                    ]
                ),
            )
            served = _served_arrays(client, job2)
            # The recomputed plane lands at its OWN timestep...
            assert np.array_equal(
                served["utci"][2], oracle["utci"][2], equal_nan=True
            ), "changed t=2 must carry the recomputed utci plane"
            # ...and every OTHER timestep keeps its correct composed value
            # (t=0 never edited; t=1 from the first edit's durable record).
            for t in (0, 1):
                assert np.array_equal(
                    served["utci"][t], oracle["utci"][t], equal_nan=True
                ), f"unchanged t={t} must not receive a mis-scattered plane"
            # The fast path never touches radiation products.
            for t in range(met_table.shape[0]):
                assert np.array_equal(
                    served["tmrt"][t],
                    family_site.baseline["tmrt"][t],
                    equal_nan=True,
                ), f"tmrt@{t} must stay the baseline plane"

    def test_bootstrap_state_scatters_sparse_time_indices(
        self, tmp_path: Path
    ) -> None:
        """Direct fence on ``_bootstrap_state``'s scatter loop (the no-published-
        result path): the sparse patch's declared time_indices — not the
        dense block offset — select the destination timestep."""
        from solweig_gpu.server.executor_bridge import _bootstrap_state

        baseline_dir = tmp_path / "baseline_results"
        baseline_dir.mkdir()
        marker = -7.0
        np.save(
            baseline_dir / "utci.f32.npy",
            np.full((3, 8, 8), marker, dtype=np.float32),
        )
        context = SimpleNamespace(
            sites=SimpleNamespace(
                config=lambda site_id: SimpleNamespace(
                    cache_dir=tmp_path, site_dir=None
                )
            ),
            store=SimpleNamespace(results_root=tmp_path / "results"),
        )
        cache = SimpleNamespace(time_steps=3)
        grid = RasterGrid(rows=8, cols=8, pixel_size_m=2.0, origin_x_m=0.0,
                          origin_y_m=0.0)
        request = SimpleNamespace(
            site_id="tiny", scenario_id="scn_sparse", trees=(), grid=grid
        )
        state = _bootstrap_state(
            context, request, cache, grid, ("utci",), (_sparse_patch((2,)),)
        )
        assert np.array_equal(
            state["utci"][2], _sparse_patch((2,)).arrays["utci"][0]
        ), "sparse patch plane must land at its declared timestep t=2"
        for untouched in (0, 1):
            assert np.all(state["utci"][untouched] == marker), (
                f"t={untouched} must keep the stored baseline, not the "
                "offset-scattered plane"
            )


# ---------------------------------------------------------------------------
# Real site_500 differential (READ-ONLY cache; env-gated: two 500x500 full
# solves). Run: SOLWEIG_SITE_500_FAST_DIFF=1 pytest tests/test_incremental_met_fast_path.py -k site_500
# ---------------------------------------------------------------------------

SITE_500_CACHE = Path("/Users/alansynn/Workspace/solweig/site-cache/site_500")
SITE_500_SITE_DIR = Path("/Users/alansynn/Workspace/solweig/Input_subset/processed_inputs")


@pytest.mark.scientific
class TestSite500RealCacheDifferential:
    def test_wind_fast_path_bitwise_on_site_500(self, tmp_path: Path) -> None:
        if not os.environ.get("SOLWEIG_SITE_500_FAST_DIFF"):
            pytest.skip("SOLWEIG_SITE_500_FAST_DIFF not set")
        if not SITE_500_CACHE.is_dir() or not SITE_500_SITE_DIR.is_dir():
            pytest.skip("site_500 cache/site not present on this host")

        from solweig_gpu.incremental.cache import SiteCache
        from solweig_gpu.incremental.executor import PlanExecutor
        from solweig_gpu.incremental.solver import load_site_forcing, run_full_tile
        from solweig_gpu.incremental.trees import TreeLayer

        cache = SiteCache.load(SITE_500_CACHE)
        baseline = np.loadtxt(
            SITE_500_SITE_DIR / "metfiles" / "metfile_0_0.txt",
            skiprows=1,
            delimiter=" ",
        )
        import datetime as _dt

        selected_date = (
            _dt.datetime(int(baseline[0, 0]), 1, 1)
            + _dt.timedelta(days=int(baseline[0, 1]) - 1)
        ).strftime("%Y-%m-%d")
        grid = RasterGrid(
            cache.rows,
            cache.cols,
            cache.pixel_size_m,
            cache.manifest.origin_x_m,
            cache.manifest.origin_y_m,
        )
        executor = PlanExecutor(
            cache=cache,
            layer=TreeLayer(cache.tree_base, grid),
            site_dir=SITE_500_SITE_DIR,
            results_root=tmp_path / "results",
            selected_date_str=selected_date,
            influence_config=LOCAL_ALWAYS,
        )

        t = 12
        load_before = _load_avgs()
        t0 = time.monotonic()
        establishing = executor.execute(
            [
                _executor_met_edit(
                    executor, time_index=t, value=4.0, outputs=SCI_VARIABLES
                )
            ]
        )
        full_seconds = time.monotonic() - t0
        assert establishing.published and establishing.mode == "full"

        t1 = time.monotonic()
        fast = executor.execute(
            [
                _executor_met_edit(
                    executor, time_index=t, value=6.5, outputs=SCI_VARIABLES,
                    edit_id="met-2",
                )
            ]
        )
        fast_seconds = time.monotonic() - t1
        load_after = _load_avgs()
        assert fast.published and fast.mode == "local"
        assert fast.diagnostics["met_fast_path_times"] == (t,)

        patch = load_patch(fast.patch_paths[0])
        assert patch.time_indices == (t,)

        # Oracle: the SAME final met state as a full-tile run under the
        # equivalent overlay (the resolved table is staged in scratch, so
        # nothing but the tmp scratch dir is written; cache stays read-only).
        from solweig_gpu.incremental.adapters.met_time import ForcingOverlay

        oracle_forcing = load_site_forcing(
            cache,
            site_dir=SITE_500_SITE_DIR,
            selected_date_str=selected_date,
            overlay=ForcingOverlay(
                changes=(
                    ForcingChange("wind_speed", t, baseline[t, 9], 6.5),
                )
            ),
        )
        oracle = run_full_tile(
            cache,
            TreeLayer(cache.tree_base, grid),
            forcing=oracle_forcing,
            site_dir=SITE_500_SITE_DIR,
            scratch_dir=tmp_path / "oracle_scratch",
            requested_variables=("utci",),
        )
        assert np.array_equal(
            patch.variable("utci")[0], oracle["utci"][t], equal_nan=True
        ), "site_500: fast-path utci differs from the full-domain oracle"

        print(
            f"[met-fast site_500] full={full_seconds:.2f}s fast={fast_seconds:.3f}s "
            f"speedup={full_seconds / max(fast_seconds, 1e-9):.0f}x "
            f"load1 {load_before['load1']} -> {load_after['load1']}"
        )
        assert fast_seconds < full_seconds
