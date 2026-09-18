# SPDX-License-Identifier: GPL-3.0-only
"""Model/receptor-parameter EditAdapter tests (U-B-params).

Covers the adapter contract for site-global model parameters: registry
schema enforcement through the adapter (all seven blocked names plus
unknown names, rejected by BOTH ``adapter.validate`` and the planner's
delta-payload checks), the documented per-parameter domains (below/above
every bounded parameter, type spellings, day-of-year integers), reset
semantics against :data:`PARAMETER_DEFAULTS`, delta round-trips through the
full planner (full-spatial downstream-only closure per the reconciled graph
edge ``model_parameters -> radiation``, geometry caches reusable), bitwise
adapter/engine plan equality with determinism repeats, the U-C mixed-scope
ruling (vegetation WINDOWS coexisting with parameter-driven FULL in ONE
plan), the ``wbgt`` never-planned invariant, transaction staging with the
kernel-argument executor seam, and registry discoverability.
"""

from __future__ import annotations

import pytest

from solweig_gpu.incremental import (
    ADAPTER_ID as VEG_ADAPTER_ID,
)
from solweig_gpu.incremental import (
    MODEL_PARAMETERS_ADAPTER_ID,
    MODEL_PARAMETERS_ADAPTER_SCHEMA_VERSION,
    MODEL_PARAMETERS_BLOCKED,
    MODEL_PARAMETERS_SAFE,
    PARAMETER_DEFAULTS,
    PARAMETER_SPECS,
    PLUMBING_CLASSIFICATION,
    AdapterRegistryError,
    AdapterSchemaError,
    EditCommand,
    EditPlanner,
    EditStateError,
    ModelParameterChange,
    ModelParameterDelta,
    ModelReceptorParametersAdapter,
    PlanningError,
    RasterGrid,
    ScenarioTransaction,
    SceneGraphState,
    SiteContext,
    SourceDeltaError,
    SpatialScope,
    TemporalScope,
    TransactionError,
    VegetationGeometryAdapter,
    before_values_from_deltas,
    builtin_registry,
    coalesce_source_deltas,
    default_edit_graph,
    kernel_arguments_from_deltas,
    register_default_adapters,
    register_model_parameters_adapter,
)
from solweig_gpu.incremental.adapters.model_parameters import ADAPTER_ID
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

#: Every blocked name from the registry ruling, with its coupling reason.
BLOCKED_PARAMS = {
    "patch_option": "SVF-cache-coupled",
    "scale": "cache-pinned",
    "location": "cache-pinned",
    "utc": "cache-pinned",
    "walllimit": "preprocessing-coupled",
    "onlyglobal": "dead met columns 21/22",
    "albedo_g": "redundant with the land-cover table",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_adapter(**kwargs) -> ModelReceptorParametersAdapter:
    return ModelReceptorParametersAdapter(**kwargs)


def context(revision: int = 0) -> SiteContext:
    return SiteContext(
        site_id="site-a",
        grid=GRID,
        scene_revision=revision,
        available_times=(0, 1, 2, 3),
    )


def param_command(
    operation: str = "update",
    *,
    edit_id: str = "params-1",
    revision: int = 0,
    old_state: dict | None = None,
    new_state: dict | None = None,
    times: tuple[int, ...] | None = (3,),
) -> EditCommand:
    return EditCommand(
        edit_id=edit_id,
        scenario_id="scenario-a",
        base_scene_revision=revision,
        adapter_id=ADAPTER_ID,
        operation=operation,
        old_state=old_state,
        new_state=new_state,
        requested_outputs=("utci",),
        requested_times=times,
    )


def update_edit(
    new_state: dict,
    *,
    old_state: dict | None = None,
    edit_id: str = "params-1",
):
    return make_adapter().validate(
        param_command(
            "update", edit_id=edit_id, old_state=old_state, new_state=new_state
        ),
        context(),
    )


def initial_state(revision: int = 0) -> SceneGraphState:
    return SceneGraphState(default_edit_graph(), revision, {})


def scopes_by_node(plan):
    return {impact.node_id: impact.spatial_scope for impact in plan.node_impacts}


def synthetic_edit(
    parameters, *, operation: str = "update", new_state: dict | None = None
):
    """A ValidatedEdit built directly from change records (bypasses the
    adapter's ``validate``) so the planner's own payload checks are tested
    without the adapter rejecting the edit first."""
    from solweig_gpu.incremental import ValidatedEdit

    command = param_command(operation, new_state=new_state)
    return ValidatedEdit(
        command=command,
        adapter_id=ADAPTER_ID,
        schema_version=MODEL_PARAMETERS_ADAPTER_SCHEMA_VERSION,
        source_node_id="model_parameters",
        delta=ModelParameterDelta(
            source_node_id="model_parameters",
            adapter_id=ADAPTER_ID,
            parameters=tuple(parameters),
        ),
    )


# ---------------------------------------------------------------------------
# Registry discovery and the defaults/spec tables
# ---------------------------------------------------------------------------


def test_adapter_registers_idempotently_and_is_discoverable() -> None:
    registry = builtin_registry()
    adapter = make_adapter(registry=registry)
    registry.register(adapter)
    registry.register(adapter)  # identical metadata: idempotent

    assert registry.get_adapter(ADAPTER_ID) is adapter
    metadata = registry.get_metadata(ADAPTER_ID)
    assert metadata is adapter.metadata
    assert metadata.operations == ("update", "reset")
    assert metadata.source_nodes == ("model_parameters",)
    assert metadata.preview == "parameter_scope_disclosure"
    assert metadata.is_full_spatial  # nominal scope starts with "full"

    # U-D capability listing shape: one entry per adapter, sorted by id.
    listing = registry.all_metadata()
    entries = [entry for entry in listing if entry.id == ADAPTER_ID]
    assert entries == [metadata]

    wired = register_default_adapters()
    assert isinstance(
        wired.get_adapter(ADAPTER_ID), ModelReceptorParametersAdapter
    )
    # The composed helper still binds the vegetation adapter.
    assert isinstance(
        wired.get_adapter(VEG_ADAPTER_ID), VegetationGeometryAdapter
    )
    solo = register_model_parameters_adapter()
    assert isinstance(
        solo.get_adapter(ADAPTER_ID), ModelReceptorParametersAdapter
    )


def test_parameter_tables_cover_the_registry_safe_set() -> None:
    assert set(PLUMBING_CLASSIFICATION) == set(MODEL_PARAMETERS_SAFE)
    assert set(PARAMETER_DEFAULTS) == set(MODEL_PARAMETERS_SAFE)
    assert {spec.name for spec in PARAMETER_SPECS} == set(MODEL_PARAMETERS_SAFE)
    assert not (set(MODEL_PARAMETERS_SAFE) & set(MODEL_PARAMETERS_BLOCKED))
    # Every default is code-cited today (no baseline-manifest entry exists).
    assert all(
        spec.default_source.startswith("code:")
        for spec in PARAMETER_SPECS
    ), "a baseline-manifest default must be refused at reset until a "
    "context carries the baseline manifest"
    # The defaults match the cited code constants (utci_process.py:50-68,
    # :649, :658, :716).
    assert PARAMETER_DEFAULTS == {
        "albedo_b": 0.2,
        "ewall": 0.9,
        "absK": 0.7,
        "absL": 0.95,
        "Fside": 0.22,
        "Fup": 0.06,
        "Fcyl": 0.28,
        "cyl": True,
        "height": 1.1,
        "transVeg": 0.03,
        "firstdayleaf": 97,
        "lastdayleaf": 300,
        "elvis": 0,
        "anisotropic_sky": 1,
    }
    # The plumbing classification splits exactly as the U-C friction list
    # documents (kernel args forwarded at utci_process.py:824-827 vs
    # loop-local prologue constants).
    kernel_args = sorted(
        k for k, v in PLUMBING_CLASSIFICATION.items() if v == "kernel_arg"
    )
    loop_local = sorted(
        k for k, v in PLUMBING_CLASSIFICATION.items() if v == "loop_local"
    )
    assert kernel_args == [
        "Fcyl",
        "Fside",
        "Fup",
        "absK",
        "absL",
        "albedo_b",
        "anisotropic_sky",
        "cyl",
        "elvis",
        "ewall",
    ]
    assert loop_local == ["firstdayleaf", "height", "lastdayleaf", "transVeg"]


def test_uncertain_bounds_are_marked_not_invented() -> None:
    uncertain = {spec.name for spec in PARAMETER_SPECS if spec.uncertain}
    # The weighting fractions have no code-enforced bound or sum constraint;
    # height has no documented upper bound. Both are flagged, not guessed.
    assert {"Fside", "Fup", "Fcyl", "height"} <= uncertain
    # Every spec carries a provenance string for its bounds and default.
    for spec in PARAMETER_SPECS:
        assert spec.bounds_basis
        assert spec.default_source


# ---------------------------------------------------------------------------
# Validation: registry schema, operations, domains
# ---------------------------------------------------------------------------


def test_update_round_trips_every_safe_parameter() -> None:
    adapter = make_adapter()
    ctx = context()
    new_values = {
        "albedo_b": 0.3,
        "ewall": 0.85,
        "absK": 0.6,
        "absL": 0.9,
        "Fside": 0.3,
        "Fup": 0.1,
        "Fcyl": 0.35,
        "cyl": False,
        "height": 1.2,
        "transVeg": 0.05,
        "firstdayleaf": 100,
        "lastdayleaf": 310,
        "elvis": 1,
        "anisotropic_sky": 0,
    }
    assert set(new_values) == set(MODEL_PARAMETERS_SAFE)
    edit = adapter.validate(
        param_command("update", new_state=new_values), ctx
    )

    assert edit.adapter_id == ADAPTER_ID
    assert edit.source_node_id == "model_parameters"
    assert edit.schema_version == adapter.schema_version == 1
    delta = edit.delta
    assert delta.source_node_id == "model_parameters"
    assert delta.spatial_windows == ()  # downstream-only: no local footprint
    assert delta.adapter_id == ADAPTER_ID
    by_name = {change.name: change for change in delta.parameters}
    assert set(by_name) == set(new_values)
    for name, value in new_values.items():
        assert by_name[name].before_value is None  # old_state not declared
        assert by_name[name].after_value == value
    # Canonical (sorted) parameter order keeps reprs deterministic.
    assert [change.name for change in delta.parameters] == sorted(new_values)

    # source_delta returns the validated payload (single derivation pass).
    assert adapter.source_delta(edit, ctx) is delta


def test_update_records_declared_before_values() -> None:
    edit = update_edit(
        {"albedo_b": 0.25, "ewall": 0.85},
        old_state={"albedo_b": 0.2, "ewall": 0.9, "absK": 0.7},
    )
    by_name = {change.name: change for change in edit.delta.parameters}
    assert by_name["albedo_b"].before_value == 0.2
    assert by_name["albedo_b"].after_value == 0.25
    assert by_name["ewall"].before_value == 0.9
    assert by_name["ewall"].after_value == 0.85
    # absK is only a before-claim for a parameter this edit does not touch:
    # no change record is manufactured for it.
    assert "absK" not in by_name


def test_update_rejects_undeclared_operation_and_bad_shapes() -> None:
    adapter = make_adapter()
    ctx = context()
    with pytest.raises(AdapterRegistryError, match="does not support"):
        adapter.validate(
            param_command("paint", new_state={"albedo_b": 0.3}), ctx
        )
    with pytest.raises(EditStateError, match="non-empty new_state"):
        adapter.validate(param_command("update", new_state=None), ctx)
    with pytest.raises(EditStateError, match="non-empty new_state"):
        adapter.validate(param_command("update", new_state={}), ctx)
    with pytest.raises(EditStateError, match="stale scene revision"):
        adapter.validate(
            param_command("update", revision=0, new_state={"albedo_b": 0.3}),
            context(revision=1),
        )
    with pytest.raises(EditStateError, match="targets adapter"):
        adapter.validate(
            EditCommand(
                edit_id="x",
                scenario_id="scenario-a",
                base_scene_revision=0,
                adapter_id="some_other_adapter",
                operation="update",
                old_state=None,
                new_state={"albedo_b": 0.3},
                requested_outputs=("utci",),
                requested_times=(3,),
            ),
            ctx,
        )


def test_every_blocked_parameter_is_rejected_by_adapter_and_planner() -> None:
    assert set(BLOCKED_PARAMS) == set(MODEL_PARAMETERS_BLOCKED)
    adapter = make_adapter()
    planner = EditPlanner(grid=GRID, registry=builtin_registry())
    for name in sorted(BLOCKED_PARAMS):
        # Adapter path: the registry rejects the command state.
        with pytest.raises(AdapterSchemaError, match="blocks parameters"):
            adapter.validate(
                param_command("update", new_state={name: 2}), context()
            )
        # Adapter path for old_state too (both states are schema-checked).
        with pytest.raises(AdapterSchemaError, match="blocks parameters"):
            adapter.validate(
                param_command(
                    "update",
                    old_state={name: 2},
                    new_state={"albedo_b": 0.3},
                ),
                context(),
            )
        # Planner path A: a fenced command state never plans.
        command = param_command("update", new_state={name: 2})
        frozen = EditCommand(
            edit_id=command.edit_id,
            scenario_id=command.scenario_id,
            base_scene_revision=command.base_scene_revision,
            adapter_id=command.adapter_id,
            operation=command.operation,
            old_state=command.old_state,
            new_state=command.new_state,
            requested_outputs=command.requested_outputs,
            requested_times=command.requested_times,
        )
        legitimate = update_edit({"albedo_b": 0.3}, edit_id="params-ok")
        smuggled = synthetic_edit(
            (ModelParameterChange(name, None, 2),),
            new_state=dict(frozen.new_state or {}),
        )
        with pytest.raises(PlanningError):
            planner.plan([legitimate, smuggled], initial_state())
        # Planner path B: the delta payload itself is schema-checked even
        # when the command states are clean.
        payload_only = synthetic_edit((ModelParameterChange(name, None, 2),))
        with pytest.raises(PlanningError, match="blocks parameters"):
            planner.plan([payload_only], initial_state())


def test_unknown_parameter_names_are_rejected_by_adapter_and_planner() -> None:
    adapter = make_adapter()
    planner = EditPlanner(grid=GRID, registry=builtin_registry())
    for name in ("transmissivity", "albedo_x", ""):
        with pytest.raises(AdapterSchemaError, match="does not define"):
            adapter.validate(
                param_command("update", new_state={name: 1}), context()
            )
    edit = synthetic_edit(
        (ModelParameterChange("transmissivity", None, 0.4),)
    )
    with pytest.raises(PlanningError, match="does not define"):
        planner.plan([edit], initial_state())


def test_bounds_are_enforced_above_and_below() -> None:
    adapter = make_adapter()
    cases = [
        ("albedo_b", 1.5),  # fraction, above
        ("albedo_b", -0.1),  # fraction, below
        ("ewall", 1.01),
        ("absK", -0.2),
        ("absL", 2.0),
        ("Fside", 1.1),
        ("Fup", -0.05),
        ("Fcyl", 3.0),
        ("transVeg", 1.2),
        ("transVeg", -0.01),
        ("height", 0.0),  # strictly positive
        ("height", -1.0),
        ("firstdayleaf", 400),  # day-of-year, above
        ("firstdayleaf", 0),  # day-of-year, below
        ("lastdayleaf", 367),
        ("lastdayleaf", -1),
        ("elvis", 2),  # switch flag
        ("anisotropic_sky", -1),
    ]
    for name, value in cases:
        with pytest.raises(EditStateError, match=name):
            adapter.validate(
                param_command("update", new_state={name: value}), context()
            )
    # Boundaries themselves are legal (height's lower bound is exclusive,
    # so a small positive value is the legal edge there).
    edit = adapter.validate(
        param_command(
            "update",
            new_state={
                "albedo_b": 0.0,
                "ewall": 1.0,
                "absK": 1.0,
                "transVeg": 1.0,
                "height": 0.05,
                "firstdayleaf": 1,
                "lastdayleaf": 366,
                "elvis": 1,
                "anisotropic_sky": 0,
            },
        ),
        context(),
    )
    assert len(edit.delta.parameters) == 9
    # height == 0 is rejected even though 0 is the nominal lower bound:
    # the documented domain is strictly positive.
    with pytest.raises(EditStateError, match="height"):
        adapter.validate(
            param_command("update", new_state={"height": 0.0}), context()
        )
    # Day-of-year wraparound is legal physics (leaf mask handles it,
    # utci_process.py:708-711): no cross-parameter ordering constraint.
    edit = adapter.validate(
        param_command(
            "update",
            old_state={"firstdayleaf": 97, "lastdayleaf": 300},
            new_state={"firstdayleaf": 330, "lastdayleaf": 60},
        ),
        context(),
    )
    assert len(edit.delta.parameters) == 2


def test_type_spellings_are_single_domain() -> None:
    adapter = make_adapter()
    ctx = context()
    # Numeric strings never smuggle into the physics layer.
    with pytest.raises(EditStateError, match="must be a number"):
        adapter.validate(
            param_command("update", new_state={"albedo_b": "0.3"}), ctx
        )
    # bool is an int subclass: a float parameter must reject it, an int
    # parameter must reject it too (one spelling per domain).
    with pytest.raises(EditStateError, match="must be a number"):
        adapter.validate(
            param_command("update", new_state={"albedo_b": True}), ctx
        )
    with pytest.raises(EditStateError, match="elvis"):
        adapter.validate(
            param_command("update", new_state={"elvis": True}), ctx
        )
    with pytest.raises(EditStateError, match="firstdayleaf"):
        adapter.validate(
            param_command("update", new_state={"firstdayleaf": 97.0}), ctx
        )
    # cyl is a boolean branch selector (default True, utci_process.py:62).
    with pytest.raises(EditStateError, match="boolean"):
        adapter.validate(param_command("update", new_state={"cyl": 1}), ctx)
    assert (
        adapter.validate(
            param_command("update", new_state={"cyl": False}), ctx
        ).delta.parameters[0].after_value
        is False
    )
    # old_state values are type-checked too, but NOT bounds-checked: an
    # out-of-range staged value must still be resettable.
    edit = adapter.validate(
        param_command(
            "reset",
            old_state={"albedo_b": 9.9},
        ),
        ctx,
    )
    (change,) = edit.delta.parameters
    assert change.before_value == 9.9
    assert change.after_value == 0.2


# ---------------------------------------------------------------------------
# Reset semantics
# ---------------------------------------------------------------------------


def test_reset_resolves_targets_to_documented_defaults() -> None:
    adapter = make_adapter()
    ctx = context()

    # Subset reset: old_state keys select the targets, values are the
    # before-claims.
    edit = adapter.validate(
        param_command("reset", old_state={"albedo_b": 0.4, "cyl": False}), ctx
    )
    by_name = {change.name: change for change in edit.delta.parameters}
    assert set(by_name) == {"albedo_b", "cyl"}
    assert by_name["albedo_b"].before_value == 0.4
    assert by_name["albedo_b"].after_value == PARAMETER_DEFAULTS["albedo_b"]
    assert by_name["cyl"].before_value is False
    assert by_name["cyl"].after_value is True

    # Full reset: no old_state resets every safe parameter, with before
    # unknown (None) throughout.
    full = adapter.validate(param_command("reset"), ctx)
    assert {change.name for change in full.delta.parameters} == set(
        PARAMETER_DEFAULTS
    )
    assert all(
        change.before_value is None for change in full.delta.parameters
    )
    assert all(
        change.after_value == PARAMETER_DEFAULTS[change.name]
        for change in full.delta.parameters
    )


def test_reset_never_accepts_user_supplied_after_values() -> None:
    adapter = make_adapter()
    with pytest.raises(EditStateError, match="must not carry a new_state"):
        adapter.validate(
            param_command("reset", new_state={"albedo_b": 0.3}), context()
        )
    # Blocked and unknown names are still rejected inside old_state (the
    # target selector is schema-checked like any other state).
    with pytest.raises(AdapterSchemaError, match="blocks parameters"):
        adapter.validate(
            param_command("reset", old_state={"patch_option": 2}), context()
        )
    with pytest.raises(AdapterSchemaError, match="does not define"):
        adapter.validate(
            param_command("reset", old_state={"transmissivity": 1}), context()
        )


def test_identity_edits_are_rejected() -> None:
    adapter = make_adapter()
    ctx = context()
    # update whose declared before equals the requested after.
    with pytest.raises(EditStateError, match="no-op"):
        adapter.validate(
            param_command(
                "update",
                old_state={"albedo_b": 0.2},
                new_state={"albedo_b": 0.2},
            ),
            ctx,
        )
    # reset of a parameter already at its documented default.
    with pytest.raises(EditStateError, match="no-op"):
        adapter.validate(
            param_command("reset", old_state={"albedo_b": 0.2}), ctx
        )
    # A multi-parameter update drops the per-parameter no-op and keeps the
    # real change rather than refusing the whole edit.
    edit = adapter.validate(
        param_command(
            "update",
            old_state={"albedo_b": 0.2, "ewall": 0.9},
            new_state={"albedo_b": 0.2, "ewall": 0.8},
        ),
        ctx,
    )
    (change,) = edit.delta.parameters
    assert change.name == "ewall"
    assert (change.before_value, change.after_value) == (0.9, 0.8)


# ---------------------------------------------------------------------------
# Impact plan: downstream-only closure, FULL scope, reusable geometry caches
# ---------------------------------------------------------------------------


def test_impact_plan_closures_and_reusable_geometry_caches() -> None:
    adapter = make_adapter()
    edit = update_edit({"albedo_b": 0.25}, old_state={"albedo_b": 0.2})
    plan = adapter.impact_plan(edit, context())

    # The reconciled graph has exactly one edge out of model_parameters
    # (model_parameters -> radiation), so the dirty closure is the
    # radiation -> surface_thermal_state -> tmrt -> utci chain.
    dirty = {impact.node_id for impact in plan.node_impacts} | set(
        plan.changed_sources
    )
    assert dirty == {
        "model_parameters",
        "radiation",
        "surface_thermal_state",
        "tmrt",
        "utci",
    }
    assert plan.changed_sources == ("model_parameters",)

    # Every geometry cache stays reusable: walls/wall_aspect, both
    # visibilities, svf, time_shadow, relative_geometry, the DSM sources,
    # the forcing sources, and the view node.
    assert set(plan.reusable_nodes) == {
        "dem",
        "building_dsm",
        "vegetation_dsm",
        "landcover",
        "meteorology",
        "selected_date_time",
        "wind_coefficients",
        "output_selection",
        "relative_geometry",
        "walls",
        "wall_aspect",
        "building_visibility",
        "vegetation_visibility",
        "svf",
        "solar_atmospheric_state",
        "time_shadow",
    }
    assert not (dirty & set(plan.reusable_nodes))
    for node_id in (
        "walls",
        "wall_aspect",
        "building_visibility",
        "vegetation_visibility",
        "svf",
        "time_shadow",
    ):
        assert node_id in plan.reusable_nodes, node_id


def test_impact_plan_is_full_spatial_with_downstream_temporal_scopes() -> None:
    edit = update_edit({"Fside": 0.3}, old_state={"Fside": 0.22})
    plan = make_adapter().impact_plan(edit, context())
    scopes = scopes_by_node(plan)

    assert set(scopes) == {
        "radiation",
        "surface_thermal_state",
        "tmrt",
        "utci",
    }
    for node_id, scope in scopes.items():
        assert scope is SpatialScope.FULL, node_id
    by_node = {impact.node_id: impact for impact in plan.node_impacts}
    for node_id in scopes:
        impact = by_node[node_id]
        assert impact.write_windows == (), node_id
        assert impact.read_windows == (), node_id
        assert "full-spatial" in impact.reason, node_id

    # The full-spatial source is recorded as the fallback reason.
    assert any(
        "source model_parameters: full-spatial adapter" in reason
        for reason in plan.fallback_reasons
    )

    # Requested-time stages narrow to ONE at the requested index; the
    # stateful accumulator replays from timestep 0 to the requested bound.
    assert by_node["radiation"].temporal_scope is TemporalScope.ONE
    assert by_node["radiation"].time_start == 3
    assert by_node["tmrt"].temporal_scope is TemporalScope.ONE
    assert by_node["utci"].temporal_scope is TemporalScope.ONE
    thermal = by_node["surface_thermal_state"]
    assert thermal.temporal_scope is TemporalScope.REPLAY
    assert thermal.time_start == 0
    assert thermal.time_stop == 3

    assert plan.estimated_memory_bytes > 0
    assert plan.estimated_work_units > 0.0
    assert plan.transport_window is None  # no windowed writes exist
    # Topological order: every producer precedes its consumers.
    graph = default_edit_graph()
    position = {i.node_id: idx for idx, i in enumerate(plan.node_impacts)}
    for producer, consumer in graph.edges:
        if producer in position and consumer in position:
            assert position[producer] < position[consumer]


# ---------------------------------------------------------------------------
# Adapter plan vs full EditPlanner: one plan, bitwise determinism
# ---------------------------------------------------------------------------


def test_adapter_plan_equals_core_planner_plan_bitwise() -> None:
    edit = update_edit({"absK": 0.6, "transVeg": 0.05})
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


def test_full_planner_single_plan_for_parameter_batch() -> None:
    adapter = make_adapter()
    ctx = context()
    edits = [
        adapter.validate(
            param_command(
                "update",
                edit_id="params-a",
                new_state={"albedo_b": 0.3},
            ),
            ctx,
        ),
        adapter.validate(
            param_command(
                "update",
                edit_id="params-b",
                old_state={"albedo_b": 0.3, "ewall": 0.9},
                new_state={"albedo_b": 0.35, "ewall": 0.8},
            ),
            ctx,
        ),
    ]
    planner = EditPlanner(grid=GRID, registry=builtin_registry())
    plan = planner.plan(edits, initial_state())
    assert plan.changed_sources == ("model_parameters",)
    assert {impact.node_id for impact in plan.node_impacts} == {
        "radiation",
        "surface_thermal_state",
        "tmrt",
        "utci",
    }

    # The engine's coalescer keeps each parameter's first before / last
    # after value; staging itself keeps one delta per edit. The first edit
    # declared no old_state, so albedo_b's before stays None (the executor
    # resolves it from the scenario parameter store).
    transaction = ScenarioTransaction(scenario_id="scenario-a")
    for edit in edits:
        adapter.apply_source_delta(edit.delta, transaction)
    committed = transaction.commit()
    assert len(committed) == 2
    coalesced = coalesce_source_deltas(committed)
    by_name = {change.name: change for change in coalesced.parameters}
    assert by_name["albedo_b"].before_value is None
    assert by_name["albedo_b"].after_value == 0.35
    assert by_name["ewall"].before_value == 0.9
    assert by_name["ewall"].after_value == 0.8

    assert planner.plan(edits, initial_state()) == plan


# ---------------------------------------------------------------------------
# U-C mixed scope: vegetation WINDOWS coexist with parameter FULL in ONE plan
# ---------------------------------------------------------------------------


def veg_adapter() -> VegetationGeometryAdapter:
    return VegetationGeometryAdapter(
        sun_positions=SUNS, influence_config=CONFIG
    )


def test_mixed_batch_one_plan_windows_and_full_coexist() -> None:
    params = make_adapter()
    veg = veg_adapter()
    ctx = context()
    param_edit = params.validate(
        param_command("update", edit_id="params-1", new_state={"Fup": 0.1}),
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
    plan = planner.plan([veg_edit, param_edit], initial_state())

    # ONE plan for the heterogeneous batch, both sources recorded.
    assert plan.changed_sources == ("model_parameters", "vegetation_dsm")
    scopes = scopes_by_node(plan)
    (veg_window,) = veg_edit.delta.windows

    # U-C mixed-scope ruling: the vegetation-local geometry chain keeps its
    # WINDOWS while the parameter-driven (and shared) nodes are FULL — full
    # for that node's products only, never a plan-level scope.
    for node_id in (
        "relative_geometry",
        "vegetation_visibility",
        "svf",
        "time_shadow",
    ):
        assert scopes[node_id] is SpatialScope.WINDOWS, node_id
        assert plan_node(plan, node_id).write_windows == (veg_window,), node_id
    for node_id in ("radiation", "surface_thermal_state", "tmrt", "utci"):
        assert scopes[node_id] is SpatialScope.FULL, node_id
    # radiation is dirty from both sources; the full-spatial parameter
    # source wins there, and that demotion is recorded on the node impact
    # (the source-level fallback reason names the full-spatial adapter).
    assert (
        "full: upstream model_parameters is full-spatial"
        in plan_node(plan, "radiation").reason
    )
    assert any(
        "source model_parameters: full-spatial adapter" in reason
        for reason in plan.fallback_reasons
    )
    # The transport bbox still covers the local writes.
    assert plan.transport_window == veg_window
    assert plan.estimated_memory_bytes > 0
    assert planner.plan([veg_edit, param_edit], initial_state()) == plan


def plan_node(plan, node_id):
    for impact in plan.node_impacts:
        if impact.node_id == node_id:
            return impact
    raise AssertionError(f"node {node_id} missing from plan")


def test_wbgt_is_never_planned_in_any_resulting_plan() -> None:
    params = make_adapter()
    veg = veg_adapter()
    ctx = context()
    param_edit = params.validate(
        param_command("update", new_state={"elvis": 1}), ctx
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

    for edits in ([param_edit], [veg_edit], [veg_edit, param_edit]):
        plan = planner.plan(edits, initial_state())
        assert "wbgt" not in {i.node_id for i in plan.node_impacts}
        assert "wbgt" not in plan.reusable_nodes
        assert any(
            "output wbgt" in reason for reason in plan.fallback_reasons
        )

    # Requesting wbgt outright is refused loudly, never silently dropped.
    command = EditCommand(
        edit_id="params-wbgt",
        scenario_id="scenario-a",
        base_scene_revision=0,
        adapter_id=ADAPTER_ID,
        operation="update",
        old_state=None,
        new_state={"elvis": 1},
        requested_outputs=("wbgt",),
        requested_times=(3,),
    )
    wbgt_edit = params.validate(command, ctx)
    with pytest.raises(PlanningError, match="never be planned"):
        planner.plan([wbgt_edit], initial_state())


# ---------------------------------------------------------------------------
# Transaction staging and the kernel-argument executor seam
# ---------------------------------------------------------------------------


def test_apply_stages_rollback_safe_deltas() -> None:
    adapter = make_adapter()
    edit_a = update_edit({"albedo_b": 0.3}, edit_id="params-a")
    edit_b = update_edit({"ewall": 0.8}, edit_id="params-b")

    transaction = ScenarioTransaction(scenario_id="scenario-a")
    adapter.apply_source_delta(edit_a.delta, transaction)
    adapter.apply_source_delta(edit_b.delta, transaction)
    assert transaction.state == "open"
    assert transaction.staged_deltas() == (edit_a.delta, edit_b.delta)

    transaction.rollback()
    assert transaction.state == "rolled_back"
    assert transaction.staged_deltas() == ()
    with pytest.raises(TransactionError):
        adapter.apply_source_delta(edit_a.delta, transaction)

    committed = ScenarioTransaction(scenario_id="scenario-a")
    adapter.apply_source_delta(edit_a.delta, committed)
    staged = committed.commit()
    assert staged == (edit_a.delta,)


def test_apply_rejects_foreign_deltas() -> None:
    adapter = make_adapter()
    transaction = ScenarioTransaction(scenario_id="scenario-a")
    with pytest.raises(SourceDeltaError, match="VegetationObjectDelta"):
        adapter.apply_source_delta(
            update_veg_delta(), transaction
        )
    with pytest.raises(SourceDeltaError, match="belongs to adapter"):
        adapter.apply_source_delta(
            ModelParameterDelta(
                "model_parameters",
                "some_other_adapter",
                (ModelParameterChange("albedo_b", None, 0.3),),
            ),
            transaction,
        )
    with pytest.raises(SourceDeltaError, match="source node"):
        adapter.apply_source_delta(
            ModelParameterDelta(
                "vegetation_dsm",
                ADAPTER_ID,
                (ModelParameterChange("albedo_b", None, 0.3),),
            ),
            transaction,
        )
    assert transaction.staged_deltas() == ()


def update_veg_delta():
    from solweig_gpu.incremental import ObjectStateChange, VegetationObjectDelta
    from solweig_gpu.incremental import RasterWindow

    return VegetationObjectDelta(
        "vegetation_dsm",
        VEG_ADAPTER_ID,
        (ObjectStateChange("tree-a", None, {"height_m": 10.0}),),
        (RasterWindow(0, 8, 0, 8),),
    )


def test_kernel_arguments_seam_coerces_numpy_float_scalars() -> None:
    """u-c2 review LOW: float-kind values fold to plain Python floats.

    ``np.float64`` subclasses ``float``, so it passes the domain gate and
    would otherwise be forwarded raw; a NumPy scalar reaching the physics
    could resolve dtype promotion differently from a plain double, which
    breaks the bitwise stability the parameter differentials rely on. The
    fold is the single hand-off point, so it is where every float-kind
    value becomes a Python double. Int and bool kinds keep their exact
    staged value (an int stays an int, ``cyl`` stays a real ``bool``).
    """
    import numpy as np

    delta = ModelParameterDelta(
        "model_parameters",
        ADAPTER_ID,
        (
            ModelParameterChange("albedo_b", None, np.float64(0.35)),
            ModelParameterChange("height", None, np.float64(1.2)),
            # int spelling of a float-kind parameter coerces too (one
            # canonical dtype per kind); int-kind parameters stay int.
            ModelParameterChange("transVeg", None, 0),
            ModelParameterChange("firstdayleaf", None, 200),
            ModelParameterChange("cyl", None, False),
        ),
    )
    args = kernel_arguments_from_deltas([delta])
    assert args == {
        "albedo_b": 0.35,
        "height": 1.2,
        "transVeg": 0.0,
        "firstdayleaf": 200,
        "cyl": False,
    }
    # Exact types, not just values: the physics always sees a plain double
    # for float kinds, a plain int for int kinds, and a real bool.
    assert type(args["albedo_b"]) is float
    assert type(args["height"]) is float
    assert type(args["transVeg"]) is float
    assert type(args["firstdayleaf"]) is int
    assert args["cyl"] is False
    # np.float64 is accepted (float subclass) but never forwarded as one.
    assert not isinstance(args["albedo_b"], np.float64)
    # Non-float NumPy scalars were already refused by the domain gate
    # before this seam (np.longdouble does not subclass float on
    # numpy >= 2), so the coercion cannot mask a dtype the gate rejects.
    with pytest.raises(SourceDeltaError, match="must be a number"):
        kernel_arguments_from_deltas(
            [
                ModelParameterDelta(
                    "model_parameters",
                    ADAPTER_ID,
                    (
                        ModelParameterChange(
                            "albedo_b", None, np.longdouble(0.35)
                        ),
                    ),
                )
            ]
        )


def test_kernel_arguments_seam_folds_committed_deltas() -> None:
    adapter = make_adapter()
    ctx = context()
    first = adapter.validate(
        param_command(
            "update",
            edit_id="params-a",
            new_state={"albedo_b": 0.3, "cyl": False},
        ),
        ctx,
    )
    second = adapter.validate(
        param_command(
            "update",
            edit_id="params-b",
            old_state={"albedo_b": 0.3},
            new_state={"albedo_b": 0.35, "transVeg": 0.05},
        ),
        ctx,
    )
    transaction = ScenarioTransaction(scenario_id="scenario-a")
    adapter.apply_source_delta(first.delta, transaction)
    adapter.apply_source_delta(second.delta, transaction)
    committed = transaction.commit()

    # The executor seam: committed deltas fold (last-write-wins) into the
    # flat name -> value mapping the U-C integration forwards into the
    # physics, re-validated against the documented domains.
    args = kernel_arguments_from_deltas(committed)
    assert args == {"albedo_b": 0.35, "cyl": False, "transVeg": 0.05}
    # Every folded name is classified for the U-C plumbing packet.
    assert set(args) <= set(PLUMBING_CLASSIFICATION)
    # before-values give the publication/undo path what was replaced;
    # parameters never declared keep None (executor resolves from the store).
    assert before_values_from_deltas(committed) == {"albedo_b": 0.3}

    # The seam re-validates: a fenced payload cannot reach the kernel. A
    # BLOCKED name is named as such; an unlisted name is "not in the safe
    # set" (this is the seam's own structural check, so it does not borrow
    # the registry's adapter-scoped error text).
    with pytest.raises(SourceDeltaError, match="BLOCKED"):
        kernel_arguments_from_deltas(
            [
                ModelParameterDelta(
                    "model_parameters",
                    ADAPTER_ID,
                    (ModelParameterChange("patch_option", None, 2),),
                )
            ]
        )
    with pytest.raises(SourceDeltaError, match="not in the documented safe"):
        kernel_arguments_from_deltas(
            [
                ModelParameterDelta(
                    "model_parameters",
                    ADAPTER_ID,
                    (ModelParameterChange("transmissivity", None, 0.4),),
                )
            ]
        )
    with pytest.raises(SourceDeltaError, match="expected ModelParameterDelta"):
        kernel_arguments_from_deltas([update_veg_delta()])
    with pytest.raises(SourceDeltaError, match="above the documented"):
        kernel_arguments_from_deltas(
            [
                ModelParameterDelta(
                    "model_parameters",
                    ADAPTER_ID,
                    (ModelParameterChange("albedo_b", None, 9.9),),
                )
            ]
        )
    with pytest.raises(SourceDeltaError, match="below the documented"):
        kernel_arguments_from_deltas(
            [
                ModelParameterDelta(
                    "model_parameters",
                    ADAPTER_ID,
                    (ModelParameterChange("transVeg", None, -0.5),),
                )
            ]
        )


# ---------------------------------------------------------------------------
# Preview and fixtures
# ---------------------------------------------------------------------------


def test_preview_descriptor_is_a_disclosure_and_not_exact() -> None:
    adapter = make_adapter()
    descriptor = adapter.preview_descriptor(
        update_edit({"Fside": 0.3}), context()
    )
    assert descriptor.kind == adapter.metadata.preview
    assert descriptor.kind == "parameter_scope_disclosure"
    assert descriptor.immediate is True
    assert any("no spatial preview" in note for note in descriptor.limitations)
    assert any("preview_is_not_exact" in note for note in descriptor.limitations)
    assert any("stay reusable" in note for note in descriptor.limitations)
    assert any("Fside" in note for note in descriptor.limitations)


def test_validation_fixtures_match_registry_entry() -> None:
    adapter = make_adapter()
    assert adapter.validation_fixtures() == (
        "baseline_restore",
        "lower_bound",
        "upper_bound",
    )
    assert adapter.validation_fixtures() == adapter.metadata.validation_fixtures


def test_adapter_exports_from_package() -> None:
    import solweig_gpu.incremental as incremental
    from solweig_gpu.incremental.adapters import (
        ModelReceptorParametersAdapter as direct,
    )

    assert direct is incremental.ModelReceptorParametersAdapter
    for name in (
        "MODEL_PARAMETERS_ADAPTER_ID",
        "MODEL_PARAMETERS_ADAPTER_SCHEMA_VERSION",
        "PARAMETER_DEFAULTS",
        "PARAMETER_SPECS",
        "PLUMBING_CLASSIFICATION",
        "ModelReceptorParametersAdapter",
        "ParameterSpec",
        "before_values_from_deltas",
        "kernel_arguments_from_deltas",
        "register_model_parameters_adapter",
    ):
        assert hasattr(incremental, name), name
        assert name in incremental.__all__
    assert incremental.MODEL_PARAMETERS_ADAPTER_ID == "model_receptor_parameters"
    # The vegetation adapter's generic constant names stay untouched.
    assert incremental.ADAPTER_ID == "vegetation_geometry"
    # Frozen delta payloads keep canonical (sorted) parameter order.
    edit = update_edit({"ewall": 0.8, "albedo_b": 0.3})
    assert [change.name for change in edit.delta.parameters] == [
        "albedo_b",
        "ewall",
    ]


def test_non_finite_values_rejected_by_adapter_and_seam() -> None:
    """u-b-params-reviewer HIGH: NaN and +inf must not pass the fences.

    NaN fails every relational comparison (nan < low and nan > high are
    both False) and +inf passes low-side-only bounds, so without an
    explicit finiteness check both reach the physics handoff. stdlib json
    parses NaN/Infinity literals, so a U-D API client can send them.
    """
    adapter = make_adapter()
    nan = float("nan")
    pos_inf = float("inf")
    neg_inf = float("-inf")
    for name, bad in (
        ("albedo_b", nan),
        ("albedo_b", pos_inf),
        ("height", nan),  # low-side-only bound: inf slips without the check
        ("height", pos_inf),
        ("transVeg", nan),
        ("ewall", neg_inf),
    ):
        # int-kind parameters (firstdayleaf/lastdayleaf) reject NaN/inf at
        # the type check already ("must be an integer in [1, 366]").
        with pytest.raises(EditStateError, match="not finite"):
            update_edit({name: bad})
    # The executor seam shares _check_value_in_domain: committed-fold
    # refuses the same payload at the last fence.
    from solweig_gpu.incremental import SourceDeltaError as _SDE

    hand_built = ModelParameterDelta(
        "model_parameters",
        ADAPTER_ID,
        (ModelParameterChange("albedo_b", None, nan),),
    )
    transaction = ScenarioTransaction(scenario_id="scenario-a")
    with pytest.raises((EditStateError, _SDE), match="not finite"):
        adapter.apply_source_delta(hand_built, transaction)
    assert transaction.staged_deltas() == ()


def test_apply_source_delta_revalidates_hand_built_payloads() -> None:
    """u-b-params-reviewer MEDIUM: the staging door re-checks domains.

    The planner's validate_delta checks NAMES; apply_source_delta is the
    last adapter-side door before commit, so a hand-built delta with an
    out-of-domain after_value or a blocked name must be refused there.
    """
    adapter = make_adapter()
    transaction = ScenarioTransaction(scenario_id="scenario-a")
    for bad_name, bad_value in (
        ("albedo_b", 9.9),  # above upper bound 1
        ("height", 0.0),  # exclusive low bound
        ("elvis", True),  # wrong kind
        ("patch_option", 2),  # blocked name
        ("albedo_g", 0.2),  # blocked name
        ("made_up_param", 1.0),  # unknown name
    ):
        delta = ModelParameterDelta(
            "model_parameters",
            ADAPTER_ID,
            (ModelParameterChange(bad_name, None, bad_value),),
        )
        with pytest.raises(EditStateError):
            adapter.apply_source_delta(delta, transaction)
    assert transaction.staged_deltas() == ()


def test_exclusive_lower_bound_error_text_is_not_inverted() -> None:
    """u-b-params-reviewer LOW-c: height=0.0 must say '> 0.0 required'."""
    with pytest.raises(EditStateError, match=r"> 0\.0 required"):
        update_edit({"height": 0.0})
    # Inclusive bounds keep the '>=' spelling.
    with pytest.raises(EditStateError, match=r">= 0\.0 required"):
        update_edit({"albedo_b": -0.1})
