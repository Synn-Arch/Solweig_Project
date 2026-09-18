# SPDX-License-Identifier: GPL-3.0-only
"""Meteorological-forcing EditAdapter tests (U-B-met).

Covers the adapter contract for per-timestep, site-global met forcing: the
registry variable fence through BOTH paths (command states via
``adapter.validate`` -> ``validate_edit_state``, delta payloads via the
planner's ``validate_delta``), documented per-variable domains and type
spellings, timestep bounds from the site context, per-(variable, time)
no-op dropping, the refused ``preset``/``reset`` operations, the
copy-on-write :class:`ForcingOverlay` (baseline arrays immutable, declared
cells only) with its ``overlay_from_deltas`` executor seam, the solver's
``load_site_forcing(overlay=...)`` relaxation (legacy default byte-identical,
masked cross-check, tamper rejection), delta round-trips through the core
planner (full-downstream-only closure per the graph's ``meteorology`` edges
— including the graph-driven ``time_shadow`` recompute — with geometry
caches reusable and ``surface_thermal_state`` replaying), bitwise
adapter/planner plan equality with determinism repeats, the U-C mixed-scope
ruling (vegetation WINDOWS coexisting with forcing FULL in ONE plan), the
``wbgt`` never-planned invariant, and registry discoverability.
"""

from __future__ import annotations

import numpy as np
import pytest

from solweig_gpu.incremental import (
    ADAPTER_ID as VEG_ADAPTER_ID,
)
from solweig_gpu.incremental import (
    METEOROLOGY_VARIABLES_BLOCKED,
    METEOROLOGY_VARIABLES_SAFE,
    AdapterRegistryError,
    AdapterSchemaError,
    EditCommand,
    EditPlanner,
    EditStateError,
    ForcingChange,
    ForcingDelta,
    ForcingOverlay,
    MeteorologicalForcingAdapter,
    PlanningError,
    RasterGrid,
    ScenarioTransaction,
    SceneGraphState,
    SiteContext,
    SourceDeltaError,
    SpatialScope,
    TemporalScope,
    VARIABLE_SPECS,
    VegetationGeometryAdapter,
    builtin_registry,
    coalesce_source_deltas,
    default_edit_graph,
    overlay_from_deltas,
    register_default_adapters,
    register_meteorological_forcing_adapter,
)
from solweig_gpu.incremental.adapters.met_time import (
    ADAPTER_ID,
    ADAPTER_SCHEMA_VERSION,
)
from solweig_gpu.incremental.geometry import (
    InfluenceConfig,
    SunPosition,
)

GRID = RasterGrid(500, 500, 2.0, origin_x_m=0.0, origin_y_m=1000.0)
SUNS = (SunPosition(45.0, 180.0), SunPosition(35.0, 225.0))
CONFIG = InfluenceConfig(
    lowest_sky_patch_altitude_deg=45.0,
    minimum_direct_sun_altitude_deg=30.0,
    maximum_shadow_length_m=60.0,
    safety_margin_m=0.0,
    block_size_pixels=1,
)

TREE_A = {
    "tree_id": "tree-a",
    "x_m": 400.0,
    "y_m": 600.0,
    "height_m": 10.0,
    "canopy_radius_m": 3.0,
}

#: Every blocked variable from the registry ruling, with its reason.
BLOCKED_VARIABLES = {
    "year": "time column 0 (solar-cache-pinned)",
    "day_of_year": "time column 1 (solar-cache-pinned)",
    "hour": "time column 2 (solar-cache-pinned)",
    "minute": "time column 3 (solar-cache-pinned)",
    "diffuse_radiation": "dead met column 21 while onlyglobal = 1",
    "direct_radiation": "dead met column 22 while onlyglobal = 1",
    "wind_direction": "inert without wind-coefficient rasters",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_adapter(**kwargs) -> MeteorologicalForcingAdapter:
    return MeteorologicalForcingAdapter(**kwargs)


def context(revision: int = 0) -> SiteContext:
    return SiteContext(
        site_id="site-a",
        grid=GRID,
        scene_revision=revision,
        available_times=(0, 1, 2, 3),
    )


def met_command(
    operation: str = "update_time_row",
    *,
    edit_id: str = "met-1",
    revision: int = 0,
    old_state: dict | None = None,
    new_state: dict | None = None,
    times: tuple[int, ...] | None = (1,),
    outputs: tuple[str, ...] = ("utci",),
) -> EditCommand:
    return EditCommand(
        edit_id=edit_id,
        scenario_id="scenario-a",
        base_scene_revision=revision,
        adapter_id=ADAPTER_ID,
        operation=operation,
        old_state=old_state,
        new_state=new_state,
        requested_outputs=outputs,
        requested_times=times,
    )


def update_edit(
    new_state: dict,
    *,
    old_state: dict | None = None,
    edit_id: str = "met-1",
    operation: str = "update_time_row",
    times: tuple[int, ...] | None = (1,),
):
    return make_adapter().validate(
        met_command(
            operation,
            edit_id=edit_id,
            old_state=old_state,
            new_state=new_state,
            times=times,
        ),
        context(),
    )


def initial_state(revision: int = 0) -> SceneGraphState:
    return SceneGraphState(default_edit_graph(), revision, {})


def plan_node(plan, node_id):
    for impact in plan.node_impacts:
        if impact.node_id == node_id:
            return impact
    raise AssertionError(f"node {node_id} missing from plan")


def scopes_by_node(plan):
    return {impact.node_id: impact.spatial_scope for impact in plan.node_impacts}


def synthetic_edit(changes, *, new_state: dict | None = None):
    """A ValidatedEdit built directly from change records (bypasses the
    adapter's ``validate``) so the planner's own payload checks are tested
    without the adapter rejecting the edit first."""
    from solweig_gpu.incremental import ValidatedEdit

    return ValidatedEdit(
        command=met_command("update_time_row", new_state=new_state),
        adapter_id=ADAPTER_ID,
        schema_version=ADAPTER_SCHEMA_VERSION,
        source_node_id="meteorology",
        delta=ForcingDelta(
            source_node_id="meteorology",
            adapter_id=ADAPTER_ID,
            changes=tuple(changes),
        ),
    )


def veg_adapter() -> VegetationGeometryAdapter:
    return VegetationGeometryAdapter(sun_positions=SUNS, influence_config=CONFIG)


# ---------------------------------------------------------------------------
# Registry discovery and the spec table
# ---------------------------------------------------------------------------


def test_adapter_registers_idempotently_and_is_discoverable() -> None:
    registry = builtin_registry()
    register_meteorological_forcing_adapter(registry)
    register_meteorological_forcing_adapter(registry)
    adapter = registry.get_adapter(ADAPTER_ID)
    assert isinstance(adapter, MeteorologicalForcingAdapter)
    # Metadata stays the builtin entry verbatim (registration binds an
    # implementation to declared metadata; it never adds capability).
    assert adapter.metadata.operations == (
        "update_time_row",
        "update_range",
        "preset",
        "reset",
    )
    assert adapter.metadata.source_nodes == ("meteorology",)
    assert adapter.metadata.nominal_spatial_scope == "full_downstream_only"
    assert adapter.metadata.temporal_scope == "changed_and_dependent_times"
    assert adapter.metadata.preview == "environment_status"
    # The composed default registration binds it alongside the others.
    composed = register_default_adapters(builtin_registry())
    assert isinstance(
        composed.get_adapter(ADAPTER_ID), MeteorologicalForcingAdapter
    )
    assert len(composed.adapter_ids()) == 9


def test_variable_table_covers_the_registry_safe_set() -> None:
    specs = {spec.variable: spec for spec in VARIABLE_SPECS}
    assert set(specs) == set(METEOROLOGY_VARIABLES_SAFE)
    # Safe and blocked vocabularies are disjoint and together closed: the
    # fence cannot be routed around by a name that is neither.
    assert not (METEOROLOGY_VARIABLES_SAFE & METEOROLOGY_VARIABLES_BLOCKED)
    # Distinct met columns inside the 25-column layout.
    columns = [spec.column for spec in VARIABLE_SPECS]
    assert len(set(columns)) == len(columns)
    assert all(0 <= c < 25 for c in columns)
    # Every documented domain is finite and ordered (import-time checked by
    # the module's own drift guard; asserted here for the failure message).
    for spec in VARIABLE_SPECS:
        assert np.isfinite(spec.low) and np.isfinite(spec.high)
        assert spec.low < spec.high
        assert spec.units and spec.bounds_basis


# ---------------------------------------------------------------------------
# validate: state shapes, domains, fence
# ---------------------------------------------------------------------------


def test_update_time_row_builds_per_variable_time_changes() -> None:
    edit = update_edit(
        {"time_index": 1, "values": {"air_temperature": 26.0, "humidity": 55.0}}
    )
    assert edit.adapter_id == ADAPTER_ID
    assert edit.source_node_id == "meteorology"
    by_key = {
        (c.variable, c.time_index): c for c in edit.delta.changes
    }
    assert set(by_key) == {("air_temperature", 1), ("humidity", 1)}
    assert by_key[("air_temperature", 1)].after_value == 26.0
    assert by_key[("air_temperature", 1)].before_value is None
    # The delta's explicit time indices are exactly the changed steps.
    assert edit.delta.time_indices == (1,)


def test_update_time_row_records_declared_before_values() -> None:
    edit = update_edit(
        {"time_index": 2, "values": {"wind_speed": 5.0}},
        old_state={"time_index": 2, "values": {"wind_speed": 2.0}},
    )
    (change,) = edit.delta.changes
    assert change.before_value == 2.0
    assert change.after_value == 5.0


def test_update_range_scalar_and_series_spellings() -> None:
    edit = update_edit(
        {
            "time_start": 0,
            "time_stop": 2,
            "values": {
                "air_temperature": 30.0,  # scalar broadcast over the range
                "pressure": (1000.0, 1005.0, 1010.0),  # one value per step
            },
        },
        operation="update_range",
    )
    by_key = {(c.variable, c.time_index): c for c in edit.delta.changes}
    assert set(by_key) == {
        ("air_temperature", 0),
        ("air_temperature", 1),
        ("air_temperature", 2),
        ("pressure", 0),
        ("pressure", 1),
        ("pressure", 2),
    }
    assert by_key[("air_temperature", 2)].after_value == 30.0
    assert by_key[("pressure", 1)].after_value == 1005.0
    assert edit.delta.time_indices == (0, 1, 2)
    # A range may select a single step (start == stop).
    single = update_edit(
        {"time_start": 1, "time_stop": 1, "values": {"uhii": 1.5}},
        operation="update_range",
    )
    assert single.delta.time_indices == (1,)
    (change,) = single.delta.changes
    assert change.variable == "uhii"


def test_update_range_rejects_length_mismatch() -> None:
    with pytest.raises(EditStateError, match="per-step"):
        update_edit(
            {
                "time_start": 0,
                "time_stop": 2,
                "values": {"pressure": (1000.0, 1005.0)},
            },
            operation="update_range",
        )
    # A scalar is fine for update_time_row but a 1-tuple must match too.
    with pytest.raises(EditStateError, match="per-step"):
        update_edit(
            {"time_index": 1, "values": {"pressure": (1.0, 2.0)}},
        )


def test_every_blocked_variable_is_rejected_by_adapter_and_planner() -> None:
    assert set(BLOCKED_VARIABLES) == set(METEOROLOGY_VARIABLES_BLOCKED)
    adapter = make_adapter()
    planner = EditPlanner(grid=GRID, registry=builtin_registry())
    for variable in sorted(BLOCKED_VARIABLES):
        # Adapter path: the registry rejects the command state (both roles).
        with pytest.raises(AdapterSchemaError, match="blocks forcing variables"):
            adapter.validate(
                met_command(
                    new_state={"time_index": 1, "values": {variable: 2.0}}
                ),
                context(),
            )
        with pytest.raises(AdapterSchemaError, match="blocks forcing variables"):
            adapter.validate(
                met_command(
                    old_state={"time_index": 1, "values": {variable: 2.0}},
                    new_state={"time_index": 1, "values": {"humidity": 50.0}},
                ),
                context(),
            )
        # Planner path: a bypass-built delta payload never plans.
        smuggled = synthetic_edit((ForcingChange(variable, 1, 1.0, 2.0),))
        with pytest.raises(PlanningError, match="blocks forcing variables"):
            planner.plan([smuggled], initial_state())


def test_unknown_variables_are_rejected_by_adapter_and_planner() -> None:
    adapter = make_adapter()
    planner = EditPlanner(grid=GRID, registry=builtin_registry())
    for variable in ("transmissivity", "co2", ""):
        with pytest.raises(AdapterSchemaError, match="does not define"):
            adapter.validate(
                met_command(
                    new_state={"time_index": 1, "values": {variable: 1.0}}
                ),
                context(),
            )
    edit = synthetic_edit((ForcingChange("vapor_pressure", 1, 1.0, 2.0),))
    with pytest.raises(PlanningError, match="does not define"):
        planner.plan([edit], initial_state())
    # The values field itself must be a mapping, never a bare sequence.
    with pytest.raises(AdapterSchemaError, match="values"):
        adapter.validate(
            met_command(new_state={"time_index": 1, "values": [26.0]}),
            context(),
        )


def test_bounds_are_enforced_above_and_below() -> None:
    adapter = make_adapter()
    cases = [
        ("air_temperature", 61.0),  # degC, above
        ("air_temperature", -60.1),  # degC, below
        ("humidity", 100.5),  # percent, above
        ("humidity", -0.1),  # percent, below
        ("radiation", -1.0),  # W m-2, below
        ("radiation", 1501.0),  # W m-2, above
        ("wind_speed", 76.0),  # m s-1, above
        ("pressure", 299.0),  # hPa, below
        ("pressure", 1101.0),  # hPa, above
        ("uhii", 15.5),  # K, above
        ("uhii", -10.5),  # K, below
    ]
    for variable, value in cases:
        with pytest.raises(EditStateError, match=variable):
            adapter.validate(
                met_command(
                    new_state={"time_index": 1, "values": {variable: value}}
                ),
                context(),
            )
    # Boundaries themselves are legal.
    edit = adapter.validate(
        met_command(
            new_state={
                "time_index": 1,
                "values": {
                    "air_temperature": -60.0,
                    "humidity": 0.0,
                    "radiation": 0.0,
                    "wind_speed": 0.0,
                    "pressure": 1100.0,
                    "uhii": 15.0,
                },
            }
        ),
        context(),
    )
    assert len(edit.delta.changes) == 6
    # Domain checks also run per step of a series, not only on scalars.
    with pytest.raises(EditStateError, match="humidity"):
        adapter.validate(
            met_command(
                "update_range",
                new_state={
                    "time_start": 0,
                    "time_stop": 2,
                    "values": {"humidity": (50.0, 120.0, 50.0)},
                },
            ),
            context(),
        )


def test_type_spellings_are_single_domain() -> None:
    adapter = make_adapter()
    for bad in (True, "26", None, float("nan"), float("inf")):
        with pytest.raises(EditStateError):
            adapter.validate(
                met_command(
                    new_state={
                        "time_index": 1,
                        "values": {"air_temperature": bad},
                    }
                ),
                context(),
            )
    # int is an acceptable float spelling (numpy met tables are float64).
    edit = adapter.validate(
        met_command(new_state={"time_index": 1, "values": {"air_temperature": 26}}),
        context(),
    )
    (change,) = edit.delta.changes
    assert change.after_value == 26.0


def test_timesteps_are_fenced_to_the_site_context() -> None:
    adapter = make_adapter()
    with pytest.raises(EditStateError, match="available"):
        adapter.validate(
            met_command(new_state={"time_index": 4, "values": {"humidity": 50.0}}),
            context(),
        )
    with pytest.raises(EditStateError, match="available"):
        adapter.validate(
            met_command(
                "update_range",
                new_state={
                    "time_start": 2,
                    "time_stop": 5,
                    "values": {"humidity": 50.0},
                },
            ),
            context(),
        )
    with pytest.raises(EditStateError, match="integer"):
        adapter.validate(
            met_command(new_state={"time_index": 1.0, "values": {"humidity": 50.0}}),
            context(),
        )
    with pytest.raises(EditStateError, match="non-negative"):
        adapter.validate(
            met_command(new_state={"time_index": -1, "values": {"humidity": 50.0}}),
            context(),
        )
    # An empty context declares no bounds: any non-negative index is legal.
    unbounded = SiteContext(
        site_id="site-a", grid=GRID, scene_revision=0, available_times=()
    )
    edit = adapter.validate(
        met_command(new_state={"time_index": 99, "values": {"humidity": 50.0}}),
        unbounded,
    )
    assert edit.delta.time_indices == (99,)


def test_state_shapes_and_unknown_fields_are_rejected() -> None:
    adapter = make_adapter()
    with pytest.raises(EditStateError, match="unknown fields"):
        adapter.validate(
            met_command(new_state={"time_index": 1, "values": {"humidity": 50.0}, "zz": 1}),
            context(),
        )
    with pytest.raises(EditStateError, match="mixes"):
        adapter.validate(
            met_command(
                new_state={
                    "time_index": 1,
                    "time_start": 0,
                    "time_stop": 1,
                    "values": {"humidity": 50.0},
                }
            ),
            context(),
        )
    with pytest.raises(EditStateError, match="time_stop"):
        adapter.validate(
            met_command(
                "update_range",
                new_state={"time_start": 2, "time_stop": 1, "values": {"humidity": 50.0}},
            ),
            context(),
        )
    with pytest.raises(EditStateError, match="new_state"):
        adapter.validate(met_command(new_state=None), context())
    with pytest.raises(EditStateError, match="non-empty"):
        adapter.validate(
            met_command(new_state={"time_index": 1, "values": {}}), context()
        )
    with pytest.raises(EditStateError, match="select different timesteps"):
        adapter.validate(
            met_command(
                old_state={"time_index": 0, "values": {"humidity": 50.0}},
                new_state={"time_index": 1, "values": {"humidity": 51.0}},
            ),
            context(),
        )


def test_identity_edits_are_rejected_and_noops_drop() -> None:
    adapter = make_adapter()
    # Fully declared no-op: nothing survives, the edit is refused.
    with pytest.raises(EditStateError, match="no-op"):
        adapter.validate(
            met_command(
                old_state={"time_index": 1, "values": {"humidity": 50.0}},
                new_state={"time_index": 1, "values": {"humidity": 50.0}},
            ),
            context(),
        )
    # Per-variable no-ops drop while the real change survives.
    edit = adapter.validate(
        met_command(
            old_state={"time_index": 1, "values": {"humidity": 50.0, "uhii": 0.0}},
            new_state={"time_index": 1, "values": {"humidity": 50.0, "uhii": 1.0}},
        ),
        context(),
    )
    (change,) = edit.delta.changes
    assert change.variable == "uhii"
    assert change.before_value == 0.0


def test_preset_and_reset_are_refused_loudly() -> None:
    adapter = make_adapter()
    with pytest.raises(EditStateError, match="preset.*not implemented"):
        adapter.validate(
            met_command("preset", new_state={"time_index": 1, "values": {"humidity": 50.0}}),
            context(),
        )
    with pytest.raises(EditStateError, match="reset.*not implemented"):
        adapter.validate(
            met_command("reset", new_state={"time_index": 1, "values": {"humidity": 50.0}}),
            context(),
        )


def test_stale_revisions_and_foreign_commands_are_refused() -> None:
    adapter = make_adapter()
    with pytest.raises(EditStateError, match="stale scene revision"):
        adapter.validate(
            met_command(revision=3, new_state={"time_index": 1, "values": {"humidity": 50.0}}),
            context(revision=4),
        )
    foreign = met_command(new_state={"time_index": 1, "values": {"humidity": 50.0}})
    foreign = EditCommand(
        edit_id=foreign.edit_id,
        scenario_id=foreign.scenario_id,
        base_scene_revision=foreign.base_scene_revision,
        adapter_id=VEG_ADAPTER_ID,
        operation=foreign.operation,
        old_state=foreign.old_state,
        new_state=foreign.new_state,
        requested_outputs=foreign.requested_outputs,
        requested_times=foreign.requested_times,
    )
    with pytest.raises(EditStateError, match="targets adapter"):
        adapter.validate(foreign, context())
    with pytest.raises(AdapterRegistryError, match="does not support operation"):
        adapter.validate(
            met_command("paint", new_state={"time_index": 1, "values": {"humidity": 50.0}}),
            context(),
        )


# ---------------------------------------------------------------------------
# Planning: closure, scopes, reusability
# ---------------------------------------------------------------------------


def test_impact_plan_closures_and_reusable_geometry_caches() -> None:
    edit = update_edit({"time_index": 1, "values": {"air_temperature": 26.0}})
    plan = make_adapter().impact_plan(edit, context())
    assert plan.changed_sources == ("meteorology",)
    # Graph-driven closure: every meteorology edge, including the
    # solar_atmospheric_state -> time_shadow chain. time_shadow IS dirty —
    # the graph (not a per-adapter guess) decides, an over-approximation
    # the planner is entitled to under the fenced vocabulary.
    assert [i.node_id for i in plan.node_impacts] == [
        "solar_atmospheric_state",
        "time_shadow",
        "radiation",
        "surface_thermal_state",
        "tmrt",
        "utci",
    ]
    # Geometry and view-factor caches stay reusable.
    for reusable in (
        "walls",
        "wall_aspect",
        "building_visibility",
        "vegetation_visibility",
        "svf",
        "relative_geometry",
        "vegetation_dsm",
        "building_dsm",
    ):
        assert reusable in plan.reusable_nodes, reusable
    assert "time_shadow" not in plan.reusable_nodes
    # The full-spatial demotion is recorded, never silent.
    assert any(
        "source meteorology: full-spatial adapter" in reason
        for reason in plan.fallback_reasons
    )


def test_impact_plan_is_full_spatial_with_changed_and_dependent_times() -> None:
    # Requested step == changed step: the union (ruling R1) collapses to
    # the single changed step, preserving pure changed-step narrowing.
    edit = update_edit(
        {"time_index": 2, "values": {"radiation": 400.0}}, times=(2,)
    )
    plan = make_adapter().impact_plan(edit, context())
    scopes = scopes_by_node(plan)
    temporal = {i.node_id: i.temporal_scope for i in plan.node_impacts}
    for node_id in scopes:
        assert scopes[node_id] is SpatialScope.FULL, node_id
        assert plan_node(plan, node_id).write_windows == (), node_id
    # changed_and_dependent_times: the changed step narrows the
    # time-varying stages, while the stateful ground-heat accumulator
    # replays from timestep 0 to the changed bound.
    for node_id in (
        "solar_atmospheric_state",
        "time_shadow",
        "radiation",
        "tmrt",
        "utci",
    ):
        assert temporal[node_id] is TemporalScope.ONE, node_id
        assert plan_node(plan, node_id).time_start == 2
    assert temporal["surface_thermal_state"] is TemporalScope.REPLAY
    replay = plan_node(plan, "surface_thermal_state")
    assert replay.time_start == 0
    assert replay.time_stop == 2


def test_r1_forcing_edit_serves_requested_window_via_union() -> None:
    """Lead ruling R1 (2026-09-02, u-b-met review): carry-safe service.

    A forcing edit at t=k with requested_times beyond k must NOT serve
    the requested steps from cache: the ground-heat carry crosses k, so
    cached t>k products are carry-stale under the new scene_revision.
    The recompute set is the UNION of changed and requested steps and
    the stateful REPLAY bound extends to its max.
    """
    edit = update_edit(
        {"time_index": 1, "values": {"radiation": 400.0}}, times=(2, 3)
    )
    plan = make_adapter().impact_plan(edit, context())
    temporal = {i.node_id: i.temporal_scope for i in plan.node_impacts}
    starts = {i.node_id: i.time_start for i in plan.node_impacts}
    # Time-varying stages cover the union (1, 2, 3) as a RANGE.
    for node_id in (
        "solar_atmospheric_state",
        "time_shadow",
        "radiation",
        "tmrt",
        "utci",
    ):
        assert temporal[node_id] is TemporalScope.RANGE, node_id
        assert starts[node_id] == 1, node_id
        assert plan_node(plan, node_id).time_stop == 3, node_id
    # The stateful accumulator replays 0 -> max(union), absorbing the
    # overlay at t=1 before serving t=2,3 from replayed state.
    assert temporal["surface_thermal_state"] is TemporalScope.REPLAY
    replay = plan_node(plan, "surface_thermal_state")
    assert replay.time_start == 0
    assert replay.time_stop == 3


def test_r1_edit_beyond_requested_window_still_recomputes_at_k() -> None:
    """R1 flip side: an edit past the requested window is never skipped."""
    edit = update_edit(
        {"time_index": 3, "values": {"humidity": 60.0}}, times=(1,)
    )
    plan = make_adapter().impact_plan(edit, context())
    temporal = {i.node_id: i.temporal_scope for i in plan.node_impacts}
    starts = {i.node_id: i.time_start for i in plan.node_impacts}
    for node_id in ("radiation", "tmrt", "utci"):
        assert temporal[node_id] is TemporalScope.RANGE, node_id
        assert starts[node_id] == 1, node_id
        assert plan_node(plan, node_id).time_stop == 3, node_id
    assert plan_node(plan, "surface_thermal_state").time_stop == 3


def test_adapter_plan_equals_core_planner_plan_bitwise() -> None:
    edit = update_edit(
        {"time_index": 1, "values": {"air_temperature": 26.0, "humidity": 55.0}}
    )
    adapter_plan = make_adapter().impact_plan(edit, context())
    core_plan = EditPlanner(grid=GRID, registry=builtin_registry()).plan(
        [edit], initial_state()
    )
    assert adapter_plan == core_plan
    assert hash(adapter_plan) == hash(core_plan)
    assert repr(adapter_plan) == repr(core_plan)
    # Bitwise determinism on repeat, adapter and core alike.
    assert make_adapter().impact_plan(edit, context()) == adapter_plan
    assert make_adapter().validate(edit.command, context()) == edit


def test_full_planner_single_plan_for_forcing_batch() -> None:
    adapter = make_adapter()
    ctx = context()
    edits = [
        adapter.validate(
            met_command(
                edit_id="met-a",
                new_state={"time_index": 1, "values": {"humidity": 40.0}},
            ),
            ctx,
        ),
        adapter.validate(
            met_command(
                edit_id="met-b",
                old_state={"time_index": 2, "values": {"humidity": 40.0}},
                new_state={"time_index": 2, "values": {"humidity": 45.0}},
            ),
            ctx,
        ),
    ]
    planner = EditPlanner(grid=GRID, registry=builtin_registry())
    plan = planner.plan(edits, initial_state())
    assert plan.changed_sources == ("meteorology",)
    assert {i.node_id for i in plan.node_impacts} == {
        "solar_atmospheric_state",
        "time_shadow",
        "radiation",
        "surface_thermal_state",
        "tmrt",
        "utci",
    }
    # changed steps across the batch narrow the time-varying stages.
    assert plan_node(plan, "utci").time_start == 1
    assert plan_node(plan, "utci").time_stop == 2

    # The engine's coalescer keeps each cell's first before / last after;
    # staging keeps one delta per edit.
    transaction = ScenarioTransaction(scenario_id="scenario-a")
    for edit in edits:
        adapter.apply_source_delta(edit.delta, transaction)
    committed = transaction.commit()
    assert len(committed) == 2
    coalesced = coalesce_source_deltas(committed)
    assert isinstance(coalesced, ForcingDelta)
    by_key = {(c.variable, c.time_index): c for c in coalesced.changes}
    assert by_key[("humidity", 1)].before_value is None
    assert by_key[("humidity", 1)].after_value == 40.0
    assert by_key[("humidity", 2)].before_value == 40.0
    assert by_key[("humidity", 2)].after_value == 45.0
    assert planner.plan(edits, initial_state()) == plan


# ---------------------------------------------------------------------------
# U-C mixed scope: vegetation WINDOWS coexist with forcing FULL in ONE plan
# ---------------------------------------------------------------------------


def test_mixed_batch_one_plan_windows_and_full_coexist() -> None:
    met = make_adapter()
    veg = veg_adapter()
    ctx = context()
    met_edit = met.validate(
        met_command(edit_id="met-1", new_state={"time_index": 1, "values": {"humidity": 40.0}}),
        ctx,
    )
    veg_edit = veg.validate(
        EditCommand(
            edit_id="veg-1",
            scenario_id="scenario-a",
            base_scene_revision=0,
            adapter_id=VEG_ADAPTER_ID,
            operation="add",
            old_state=None,
            new_state=dict(TREE_A),
            requested_outputs=("utci",),
            requested_times=(3,),
        ),
        ctx,
    )

    planner = EditPlanner(grid=GRID, registry=builtin_registry())
    plan = planner.plan([veg_edit, met_edit], initial_state())

    # ONE plan for the heterogeneous batch, both sources recorded.
    assert plan.changed_sources == ("meteorology", "vegetation_dsm")
    scopes = scopes_by_node(plan)
    (veg_window,) = veg_edit.delta.windows

    # U-C mixed-scope ruling: the vegetation-local geometry chain keeps its
    # WINDOWS while the forcing-driven (and shared) nodes are FULL. Unlike
    # a parameters batch, forcing dirties solar_atmospheric_state itself,
    # so time_shadow goes FULL here (full-spatial source wins) — the
    # vegetation chain below it stays windowed.
    for node_id in ("relative_geometry", "vegetation_visibility", "svf"):
        assert scopes[node_id] is SpatialScope.WINDOWS, node_id
        assert plan_node(plan, node_id).write_windows == (veg_window,), node_id
    for node_id in (
        "solar_atmospheric_state",
        "time_shadow",
        "radiation",
        "surface_thermal_state",
        "tmrt",
        "utci",
    ):
        assert scopes[node_id] is SpatialScope.FULL, node_id
        assert plan_node(plan, node_id).write_windows == (), node_id
    assert any(
        "source meteorology: full-spatial adapter" in reason
        for reason in plan.fallback_reasons
    )
    assert plan.transport_window == veg_window
    assert plan.estimated_memory_bytes > 0
    assert planner.plan([veg_edit, met_edit], initial_state()) == plan


def test_wbgt_is_never_planned_in_any_resulting_plan() -> None:
    met = make_adapter()
    veg = veg_adapter()
    ctx = context()
    met_edit = met.validate(
        met_command(new_state={"time_index": 1, "values": {"humidity": 40.0}}), ctx
    )
    veg_edit = veg.validate(
        EditCommand(
            edit_id="veg-1",
            scenario_id="scenario-a",
            base_scene_revision=0,
            adapter_id=VEG_ADAPTER_ID,
            operation="add",
            old_state=None,
            new_state=dict(TREE_A),
            requested_outputs=("utci",),
            requested_times=(3,),
        ),
        ctx,
    )
    planner = EditPlanner(grid=GRID, registry=builtin_registry())
    for edits in ([met_edit], [veg_edit], [veg_edit, met_edit]):
        plan = planner.plan(edits, initial_state())
        assert "wbgt" not in {i.node_id for i in plan.node_impacts}
        assert "wbgt" not in plan.reusable_nodes
        assert any("output wbgt" in reason for reason in plan.fallback_reasons)

    # Requesting wbgt outright is refused loudly, never silently dropped.
    wbgt_edit = met.validate(
        met_command(
            new_state={"time_index": 1, "values": {"humidity": 40.0}},
            outputs=("wbgt",),
        ),
        ctx,
    )
    with pytest.raises(PlanningError, match="never be planned"):
        planner.plan([wbgt_edit], initial_state())


# ---------------------------------------------------------------------------
# The ForcingOverlay and the overlay_from_deltas executor seam
# ---------------------------------------------------------------------------


def test_overlay_from_deltas_folds_committed_deltas_last_write_wins() -> None:
    adapter = make_adapter()
    ctx = context()
    first = adapter.validate(
        met_command(
            edit_id="met-a",
            old_state={"time_index": 1, "values": {"humidity": 60.0}},
            new_state={"time_index": 1, "values": {"humidity": 40.0}},
        ),
        ctx,
    )
    # A later edit of the SAME cell: the fold keeps the first edit's
    # declared before and the last edit's after value.
    second = adapter.validate(
        met_command(
            edit_id="met-b",
            old_state={"time_index": 1, "values": {"humidity": 40.0}},
            new_state={"time_index": 1, "values": {"humidity": 45.0}},
        ),
        ctx,
    )
    # A distinct cell keeps its own first before.
    third = adapter.validate(
        met_command(
            edit_id="met-c",
            old_state={"time_index": 2, "values": {"air_temperature": 20.0}},
            new_state={"time_index": 2, "values": {"air_temperature": 30.0}},
        ),
        ctx,
    )
    overlay = overlay_from_deltas([first.delta, second.delta, third.delta])
    by_key = {(c.variable, c.time_index): c for c in overlay.changes}
    assert by_key[("humidity", 1)].before_value == 60.0
    assert by_key[("humidity", 1)].after_value == 45.0
    assert by_key[("air_temperature", 2)].before_value == 20.0
    assert by_key[("air_temperature", 2)].after_value == 30.0
    # Empty fold is a legal identity overlay.
    assert overlay_from_deltas([]).changes == ()


def test_overlay_from_deltas_rejects_foreign_deltas() -> None:
    adapter = make_adapter()
    edit = update_edit({"time_index": 1, "values": {"humidity": 40.0}})
    foreign_node = ForcingDelta(
        source_node_id="selected_date_time",
        adapter_id=ADAPTER_ID,
        changes=edit.delta.changes,
    )
    with pytest.raises(SourceDeltaError, match="source node"):
        overlay_from_deltas([foreign_node])
    foreign_adapter = ForcingDelta(
        source_node_id="meteorology",
        adapter_id="selected_date_time",
        changes=edit.delta.changes,
    )
    with pytest.raises(SourceDeltaError, match="adapter"):
        overlay_from_deltas([foreign_adapter])
    with pytest.raises(SourceDeltaError, match="ForcingDelta"):
        overlay_from_deltas([42])


def test_overlay_rejects_fenced_and_out_of_domain_values() -> None:
    # The overlay is self-certifying: bypass-built records cannot smuggle
    # blocked variables or out-of-domain values to the solver.
    with pytest.raises(SourceDeltaError, match="blocked"):
        ForcingOverlay(changes=(ForcingChange("hour", 1, 10.0, 11.0),))
    with pytest.raises(SourceDeltaError, match="unknown"):
        ForcingOverlay(changes=(ForcingChange("vapor_pressure", 1, 1.0, 2.0),))
    with pytest.raises(SourceDeltaError, match="above the documented upper"):
        ForcingOverlay(changes=(ForcingChange("humidity", 1, 50.0, 120.0),))
    with pytest.raises(SourceDeltaError, match="finite"):
        ForcingOverlay(
            changes=(ForcingChange("humidity", 1, 50.0, float("nan")),)
        )
    # Whole-series tuples are domain-checked per step too.
    with pytest.raises(SourceDeltaError, match="below the documented lower"):
        ForcingOverlay(
            changes=(ForcingChange("pressure", None, None, (1013.0, 50.0)),)
        )
    with pytest.raises(SourceDeltaError, match="duplicate"):
        ForcingOverlay(
            changes=(
                ForcingChange("humidity", 1, 50.0, 51.0),
                ForcingChange("humidity", 1, 51.0, 52.0),
            )
        )


def test_overlay_resolve_is_copy_on_write_and_declared_cells_only() -> None:
    baseline = np.arange(75, dtype=np.float64).reshape(3, 25)
    frozen = baseline.copy()
    overlay = ForcingOverlay(
        changes=(
            ForcingChange("air_temperature", 1, 20.0, 26.0),
            ForcingChange("humidity", None, None, 55.0),  # whole series
        )
    )
    resolved = overlay.resolve(baseline)
    # The baseline array is bitwise untouched (hard invariant).
    assert np.array_equal(baseline, frozen)
    assert resolved is not baseline
    # Declared cells carry the scenario values.
    assert resolved[1, 11] == 26.0
    assert np.all(resolved[:, 10] == 55.0)
    # Everything else is bitwise the baseline.
    mask = overlay.changed_cell_mask(baseline.shape)
    assert mask.sum() == 3 + 1  # whole humidity column + one Ta cell
    assert np.array_equal(resolved[~mask], baseline[~mask])
    # Resolving twice keeps the baseline intact and is deterministic.
    assert np.array_equal(overlay.resolve(baseline), resolved)
    assert np.array_equal(baseline, frozen)


def test_overlay_resolve_rejects_shape_violations() -> None:
    overlay = ForcingOverlay(changes=(ForcingChange("uhii", 1, 0.0, 1.0),))
    with pytest.raises(SourceDeltaError, match="2-D"):
        overlay.resolve(np.zeros(25))
    with pytest.raises(SourceDeltaError, match="columns"):
        overlay.resolve(np.zeros((3, 24)))  # uhii needs column 24
    series = ForcingOverlay(
        changes=(ForcingChange("uhii", None, None, (0.0, 1.0, 2.0)),)
    )
    with pytest.raises(SourceDeltaError, match="whole-series"):
        series.resolve(np.zeros((2, 25)))  # 3 values vs 2 rows


def test_overlay_resolve_time_index_bounds_are_checked() -> None:
    overlay = ForcingOverlay(changes=(ForcingChange("uhii", 5, 0.0, 1.0),))
    with pytest.raises(SourceDeltaError, match="outside the baseline"):
        overlay.resolve(np.zeros((3, 25)))


# ---------------------------------------------------------------------------
# The solver seam: load_site_forcing(overlay=...)
# ---------------------------------------------------------------------------


class TestSolverOverlaySeam:
    """Solver cross-check relaxation, exercised on the tiny-site fixture."""

    def _tiny(self, tmp_path):
        from tests.test_incremental_worker import (
            DATE_STR,
            _build_cache,
            _make_tiny_site,
        )

        grid, site = _make_tiny_site(tmp_path)
        cache = _build_cache(
            site,
            tmp_path / "cache",
            met_path=site / "metfiles" / f"metfile_0_0_{DATE_STR}.txt",
            site_id="tiny",
        )
        return site, cache, DATE_STR

    def _overlay(self, site, cache, date_str, *, variable="air_temperature", value=26.0):
        from solweig_gpu.incremental.edit_types import SiteContext

        baseline = cache.met.shape[0]
        ctx = SiteContext(
            site_id="tiny",
            grid=RasterGrid(128, 128, 2.0, origin_x_m=1000.0, origin_y_m=2000.0),
            scene_revision=0,
            available_times=tuple(range(baseline)),
        )
        edit = make_adapter().validate(
            met_command(
                new_state={"time_index": 1, "values": {variable: value}},
                times=(1,),
            ),
            ctx,
        )
        return overlay_from_deltas([edit.delta])

    def test_legacy_default_is_byte_identical(self, tmp_path):
        from solweig_gpu.incremental.solver import load_site_forcing

        site, cache, date_str = self._tiny(tmp_path)
        met_path = site / "metfiles" / f"metfile_0_0_{date_str}.txt"
        expected = np.loadtxt(met_path, skiprows=1, delimiter=" ")
        forcing = load_site_forcing(
            cache, site_dir=site, selected_date_str=date_str
        )
        # Bitwise: the legacy loader output is the text file, unrounded.
        assert np.array_equal(forcing.met_table, expected)
        # The explicit default spells the same legacy path.
        explicit = load_site_forcing(
            cache, site_dir=site, selected_date_str=date_str, overlay=None
        )
        assert np.array_equal(explicit.met_table, forcing.met_table)
        assert np.array_equal(explicit.altitude, forcing.altitude)

    def test_overlay_resolves_declared_cells_and_keeps_cache_baseline(
        self, tmp_path
    ):
        from solweig_gpu.incremental.solver import load_site_forcing

        site, cache, date_str = self._tiny(tmp_path)
        cache_before = np.array(cache.met)
        baseline = load_site_forcing(
            cache, site_dir=site, selected_date_str=date_str
        )
        overlay = self._overlay(site, cache, date_str)
        scenario = load_site_forcing(
            cache, site_dir=site, selected_date_str=date_str, overlay=overlay
        )
        # Declared cell carries the scenario value; everything else is
        # bitwise the baseline table.
        assert scenario.met_table[1, 11] == 26.0
        mask = overlay.changed_cell_mask(baseline.met_table.shape)
        assert np.array_equal(
            scenario.met_table[~mask], baseline.met_table[~mask]
        )
        # The cache-backed array was never mutated (copy-on-write).
        assert np.array_equal(np.asarray(cache.met), cache_before)
        # A subsequent legacy load still sees the pristine baseline.
        again = load_site_forcing(
            cache, site_dir=site, selected_date_str=date_str
        )
        assert np.array_equal(again.met_table, baseline.met_table)
        # The fenced vocabulary cannot move the solar geometry: the
        # recomputed solar series still matches the cache, i.e. the solar
        # cross-check was NOT loosened for the overlay path.
        assert np.array_equal(scenario.altitude, baseline.altitude)
        assert np.array_equal(scenario.azimuth, baseline.azimuth)

    def test_masked_cross_check_rejects_tampered_resolution(self, tmp_path):
        from solweig_gpu.incremental.solver import SolverInputError, load_site_forcing

        site, cache, date_str = self._tiny(tmp_path)
        overlay = self._overlay(site, cache, date_str)

        class Tampered(ForcingOverlay):
            """Bypass: resolve() writes an UNDECLARED cell (wind column)."""

            def resolve(self, baseline):
                table = super().resolve(baseline)
                table[:, 9] += 1.0
                return table

        with pytest.raises(SolverInputError, match="outside the overlay"):
            load_site_forcing(
                cache,
                site_dir=site,
                selected_date_str=date_str,
                overlay=Tampered(changes=overlay.changes),
            )

    def test_masked_cross_check_pins_tolerances_sub_tolerance(self, tmp_path):
        """u-b-met-reviewer MEDIUM-2: the masked re-check keeps the SAME
        tolerances as the baseline check — a SUB-tolerance tamper (5e-4,
        above rtol=1e-5/atol=1e-4 but far below the +1.0 the coarse test
        uses) must still be rejected. Loosening the masked check to
        rtol=1e-2/atol=1e-1 would accept this tamper and fail this test.
        """
        from solweig_gpu.incremental.solver import SolverInputError, load_site_forcing

        site, cache, date_str = self._tiny(tmp_path)
        overlay = self._overlay(site, cache, date_str)

        class SubToleranceTamper(ForcingOverlay):
            def resolve(self, baseline):
                table = super().resolve(baseline)
                table[:, 9] += 5e-4
                return table

        with pytest.raises(SolverInputError, match="outside the overlay"):
            load_site_forcing(
                cache,
                site_dir=site,
                selected_date_str=date_str,
                overlay=SubToleranceTamper(changes=overlay.changes),
            )

    def test_non_overlay_objects_are_refused(self, tmp_path):
        from solweig_gpu.incremental.solver import SolverInputError, load_site_forcing

        site, cache, date_str = self._tiny(tmp_path)
        for bad in ({"air_temperature": 26.0}, [26.0], 26.0):
            with pytest.raises(SolverInputError, match="ForcingOverlay"):
                load_site_forcing(
                    cache,
                    site_dir=site,
                    selected_date_str=date_str,
                    overlay=bad,
                )


# ---------------------------------------------------------------------------
# Transaction staging, preview, fixtures, exports
# ---------------------------------------------------------------------------


def test_apply_stages_rollback_safe_deltas() -> None:
    adapter = make_adapter()
    edit_a = update_edit(
        {"time_index": 1, "values": {"humidity": 40.0}}, edit_id="met-a"
    )
    edit_b = update_edit(
        {"time_index": 2, "values": {"humidity": 45.0}}, edit_id="met-b"
    )

    transaction = ScenarioTransaction(scenario_id="scenario-a")
    adapter.apply_source_delta(edit_a.delta, transaction)
    adapter.apply_source_delta(edit_b.delta, transaction)
    assert transaction.state == "open"
    assert transaction.staged_deltas() == (edit_a.delta, edit_b.delta)

    transaction.rollback()
    assert transaction.state == "rolled_back"
    assert transaction.staged_deltas() == ()
    from solweig_gpu.incremental import TransactionError

    with pytest.raises(TransactionError):
        adapter.apply_source_delta(edit_a.delta, transaction)

    committed = ScenarioTransaction(scenario_id="scenario-a")
    adapter.apply_source_delta(edit_a.delta, committed)
    staged = committed.commit()
    assert staged == (edit_a.delta,)


def test_apply_staging_door_revalidates_hand_built_payloads() -> None:
    """u-b-met-reviewer MEDIUM-1 (params precedent 3dc1f82): the staging
    door re-runs the full fence (names/finiteness/domains) on hand-built
    ForcingDelta payloads before they can enter a transaction — the
    planner's validate_delta is names-only and would otherwise defer
    rejection to overlay_from_deltas.
    """
    adapter = make_adapter()
    transaction = ScenarioTransaction(scenario_id="scenario-a")
    good = ForcingChange("humidity", 1, 50.0, 51.0)
    for label, changes in (
        ("non-finite nan", (ForcingChange("humidity", 1, 50.0, float("nan")),)),
        ("non-finite inf", (ForcingChange("humidity", 1, 50.0, float("inf")),)),
        ("above domain", (ForcingChange("humidity", 1, 50.0, 400.0),)),
        ("below domain", (ForcingChange("radiation", 1, 400.0, -5.0),)),
        ("blocked variable", (ForcingChange("wind_direction", 1, 10.0, 20.0),)),
        ("unknown variable", (ForcingChange("made_up_var", 1, 1.0, 2.0),)),
        ("duplicate change", (good, ForcingChange("humidity", 1, 52.0, 53.0))),
    ):
        hand_built = ForcingDelta(
            source_node_id="meteorology",
            adapter_id=ADAPTER_ID,
            changes=changes,
        )
        with pytest.raises(
            SourceDeltaError,
            match=r"finite|bound|blocked|unknown|duplicate|number",
        ):
            adapter.apply_source_delta(hand_built, transaction)
    assert transaction.staged_deltas() == ()


def test_apply_rejects_foreign_deltas() -> None:
    adapter = make_adapter()
    edit = update_edit({"time_index": 1, "values": {"humidity": 40.0}})
    transaction = ScenarioTransaction(scenario_id="scenario-a")
    with pytest.raises(SourceDeltaError, match="ForcingDelta"):
        adapter.apply_source_delta("not-a-delta", transaction)
    foreign_node = ForcingDelta(
        source_node_id="selected_date_time",
        adapter_id=ADAPTER_ID,
        changes=edit.delta.changes,
    )
    with pytest.raises(SourceDeltaError, match="source node"):
        adapter.apply_source_delta(foreign_node, transaction)
    foreign_adapter = ForcingDelta(
        source_node_id="meteorology",
        adapter_id="selected_date_time",
        changes=edit.delta.changes,
    )
    with pytest.raises(SourceDeltaError, match="adapter"):
        adapter.apply_source_delta(foreign_adapter, transaction)
    assert transaction.staged_deltas() == ()


def test_preview_descriptor_is_a_status_disclosure_and_not_exact() -> None:
    adapter = make_adapter()
    edit = update_edit({"time_index": 1, "values": {"humidity": 40.0}})
    descriptor = adapter.preview_descriptor(edit, context())
    assert descriptor.kind == "environment_status"
    assert descriptor.immediate is True
    text = " ".join(descriptor.limitations)
    assert "preview_is_not_exact" in text
    assert "humidity" in text
    assert "timestep" in text or "timesteps" in text
    assert "replay" in text  # changed_and_dependent_times is disclosed


def test_validation_fixtures_match_registry_entry() -> None:
    adapter = make_adapter()
    assert adapter.validation_fixtures() == (
        "air_temperature",
        "humidity",
        "radiation",
        "wind_speed_direction",
    )


def test_adapter_exports_from_package() -> None:
    from solweig_gpu.incremental import adapters

    assert adapters.METEOROLOGY_ADAPTER_ID == ADAPTER_ID
    assert adapters.METEOROLOGY_ADAPTER_SCHEMA_VERSION == ADAPTER_SCHEMA_VERSION
    assert adapters.MeteorologicalForcingAdapter is MeteorologicalForcingAdapter
    assert adapters.ForcingOverlay is ForcingOverlay
    assert adapters.overlay_from_deltas is overlay_from_deltas
    assert callable(adapters.register_meteorological_forcing_adapter)
    for name in (
        "MET_COLUMN_COUNT",
        "VARIABLE_SPECS",
        "ForcingVariableSpec",
    ):
        assert hasattr(adapters, name), name
        assert name in adapters.__all__, name
