# SPDX-License-Identifier: GPL-3.0-only
"""Building-massing EditAdapter tests (U-B-bld, adapter #5 — full-only).

Covers the adapter contract for building massing under the full-only
ruling: the operations vocabulary (add/move/update/delete honored;
``raster_patch`` refused loudly with the met precedent), the documented
footprint/height property schema (finiteness, domains, shapes, merge
semantics, identity edits), the FULL-ONLY planning shape on every
operation (zero windows, no transport bbox, every geometry/vision cache
invalidated, ``wbgt`` pruned with a reason), bitwise adapter/engine plan
equality against the hand-built H2 path (which carries a window the
planner must never read), determinism repeats, the U-C mixed-scope batch
(building + vegetation + parameters in ONE plan: building forces its
chain FULL, vegetation's chain goes FULL through the FULL
``relative_geometry`` parent, parameters stay full-spatial), the staging
door's payload re-validation (hand-built deltas with non-finite or
out-of-domain fields are refused pre-stage), the rasterizer executor
seam, and registry/package discoverability.
"""

from __future__ import annotations

import pytest

from solweig_gpu.incremental import (
    MODEL_PARAMETERS_ADAPTER_ID,
    RasterGrid,
    RasterWindow,
    BuildingMassingAdapter,
    BuildingMassingDelta,
    BuildingSpec,
    MassingEdit,
    ObjectStateChange,
    VegetationObjectDelta,
    EditCommand,
    EditPlanner,
    EditStateError,
    SceneGraphState,
    SiteContext,
    SourceDeltaError,
    SpatialScope,
    TemporalScope,
    ValidatedEdit,
    VegetationGeometryAdapter,
    builtin_adapter_metadata,
    builtin_registry,
    default_edit_graph,
    massing_edits_from_deltas,
    register_building_massing_adapter,
    register_default_adapters,
    ADAPTER_ID as VEG_ADAPTER_ID,
)
from solweig_gpu.incremental import (
    BUILDING_ADAPTER_ID,
    BUILDING_ADAPTER_SCHEMA_VERSION,
    BUILDING_EDITABLE_PROPERTIES,
    BUILDING_IDENTITY_PROPERTY,
    BUILDING_SOURCE_NODE,
    HEIGHT_MAX_M,
    HEIGHT_MIN_M,
)
from solweig_gpu.incremental.adapters.building import ADAPTER_ID
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

#: A valid massing block, in the UI's JSON-ish spelling (lists of pairs).
BLOCK_A = {
    "building_id": "block-a",
    "footprint_m": [
        [400.0, 500.0],
        [420.0, 500.0],
        [420.0, 480.0],
        [400.0, 480.0],
    ],
    "height_m": 15.0,
}

BLOCK_A_TALLER = {
    "building_id": "block-a",
    "height_m": 20.0,
}

TREE_A = {
    "tree_id": "tree-a",
    "x_m": 400.0,
    "y_m": 600.0,
    "height_m": 10.0,
    "canopy_radius_m": 3.0,
}

#: The exact dirty stages a building edit must invalidate (graph edges:
#: building_dsm -> relative_geometry/walls/wall_aspect/building_visibility;
#: relative_geometry -> vegetation_visibility; both visibilities -> svf;
#: svf/walls/wall_aspect -> time_shadow -> radiation ->
#: surface_thermal_state -> tmrt -> utci).
BUILDING_DIRTY_STAGES = (
    "relative_geometry",
    "walls",
    "wall_aspect",
    "building_visibility",
    "vegetation_visibility",
    "svf",
    "time_shadow",
    "radiation",
    "surface_thermal_state",
    "tmrt",
    "utci",
)

#: Everything else stays reusable: no unedited source, no forcing source,
#: no view node, and solar_atmospheric_state (meteorology/time driven
#: only) is building-invariant.
BUILDING_REUSABLE_NODES = (
    "dem",
    "vegetation_dsm",
    "landcover",
    "meteorology",
    "selected_date_time",
    "wind_coefficients",
    "model_parameters",
    "output_selection",
    "solar_atmospheric_state",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_adapter(**kwargs) -> BuildingMassingAdapter:
    return BuildingMassingAdapter(**kwargs)


def context(revision: int = 0) -> SiteContext:
    return SiteContext(
        site_id="site-a",
        grid=GRID,
        scene_revision=revision,
        available_times=(0, 1, 2, 3),
    )


def building_command(
    operation: str = "update",
    *,
    edit_id: str = "bld-1",
    revision: int = 0,
    old_state: dict | None = None,
    new_state: dict | None = None,
    outputs: tuple[str, ...] = ("utci",),
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
        requested_outputs=outputs,
        requested_times=times,
    )


def add_edit(**kwargs):
    return make_adapter().validate(
        building_command("add", new_state=dict(BLOCK_A), **kwargs), context()
    )


def update_edit(**kwargs):
    return make_adapter().validate(
        building_command(
            "update",
            old_state=dict(BLOCK_A),
            new_state=dict(BLOCK_A_TALLER),
            **kwargs,
        ),
        context(),
    )


def initial_state(revision: int = 0) -> SceneGraphState:
    return SceneGraphState(default_edit_graph(), revision, {})


def scopes_by_node(plan):
    return {impact.node_id: impact.spatial_scope for impact in plan.node_impacts}


def plan_node(plan, node_id):
    for impact in plan.node_impacts:
        if impact.node_id == node_id:
            return impact
    raise AssertionError(f"node {node_id} missing from plan")


def hand_built_h2_edit() -> ValidatedEdit:
    """The engine's H2 path verbatim: a delta smuggling a window in."""
    command = EditCommand(
        edit_id="h2-hand-built",
        scenario_id="scenario-a",
        base_scene_revision=0,
        adapter_id=ADAPTER_ID,
        operation="update",
        old_state={"height_m": 10.0},
        new_state={"height_m": 20.0},
        requested_outputs=("utci",),
        requested_times=(3,),
    )
    delta = BuildingMassingDelta(
        BUILDING_SOURCE_NODE,
        ADAPTER_ID,
        (ObjectStateChange("b1", {"height_m": 10.0}, {"height_m": 20.0}),),
        (RasterWindow(100, 200, 100, 200),),
    )
    return ValidatedEdit(command, ADAPTER_ID, 1, BUILDING_SOURCE_NODE, delta)


# ---------------------------------------------------------------------------
# Registry discovery, metadata parity, package exports
# ---------------------------------------------------------------------------


def test_adapter_registers_idempotently_and_is_discoverable() -> None:
    registry = builtin_registry()
    adapter = make_adapter(registry=registry)
    registry.register(adapter)
    registry.register(adapter)  # identical metadata: idempotent

    assert registry.get_adapter(ADAPTER_ID) is adapter
    # The composed helper binds building alongside every other adapter.
    composed = register_default_adapters()
    assert isinstance(
        composed.get_adapter(ADAPTER_ID), BuildingMassingAdapter
    )
    solo = register_building_massing_adapter()
    assert isinstance(solo.get_adapter(ADAPTER_ID), BuildingMassingAdapter)


def test_metadata_is_the_builtin_entry_verbatim() -> None:
    adapter = make_adapter()
    entry = next(
        metadata
        for metadata in builtin_adapter_metadata()
        if metadata.id == ADAPTER_ID
    )
    assert adapter.metadata == entry
    from solweig_gpu.incremental import AdapterStatus

    assert entry.status is AdapterStatus.FULL_ONLY_INITIALLY
    assert entry.source_nodes == (BUILDING_SOURCE_NODE,)
    assert entry.operations == (
        "add",
        "move",
        "update",
        "delete",
        "raster_patch",
    )
    assert entry.validation_fixtures == (
        "height",
        "footprint",
        "wall_aspect",
        "low_sun",
        "boundary",
    )
    assert entry.preview == "massing_and_shadow"
    assert adapter.validation_fixtures() == entry.validation_fixtures


def test_adapter_exports_from_package() -> None:
    # Import-origin proof: every public name resolves through the package
    # re-exports (adapters/__init__ and incremental/__init__ blocks).
    import solweig_gpu.incremental as pkg
    import solweig_gpu.incremental.adapters as adapters_pkg

    assert pkg.BuildingMassingAdapter is BuildingMassingAdapter
    assert adapters_pkg.BuildingMassingAdapter is BuildingMassingAdapter
    assert pkg.BUILDING_ADAPTER_ID == ADAPTER_ID == "building_geometry"
    assert pkg.BUILDING_ADAPTER_SCHEMA_VERSION == 1
    assert pkg.BUILDING_SOURCE_NODE == "building_dsm"
    assert pkg.BUILDING_IDENTITY_PROPERTY == "building_id"
    assert pkg.BUILDING_EDITABLE_PROPERTIES == ("footprint_m", "height_m")
    assert pkg.MassingEdit is MassingEdit
    assert pkg.BuildingSpec is BuildingSpec
    assert pkg.massing_edits_from_deltas is massing_edits_from_deltas
    for name in (
        "BUILDING_ADAPTER_ID",
        "BUILDING_ADAPTER_SCHEMA_VERSION",
        "BUILDING_EDITABLE_PROPERTIES",
        "BUILDING_IDENTITY_PROPERTY",
        "BUILDING_SOURCE_NODE",
        "BuildingMassingAdapter",
        "BuildingSpec",
        "HEIGHT_MAX_M",
        "HEIGHT_MIN_M",
        "MIN_FOOTPRINT_VERTICES",
        "MassingEdit",
        "massing_edits_from_deltas",
        "register_building_massing_adapter",
    ):
        assert name in pkg.__all__, name


# ---------------------------------------------------------------------------
# Operations vocabulary and state shapes
# ---------------------------------------------------------------------------


def test_raster_patch_is_refused_loudly_with_met_precedent() -> None:
    with pytest.raises(
        EditStateError, match="raster_patch.*not implemented"
    ) as excinfo:
        make_adapter().validate(
            building_command("raster_patch", new_state=dict(BLOCK_A)),
            context(),
        )
    message = str(excinfo.value)
    assert "raster" in message.lower()
    assert "refused rather than guessed" in message
    # The refusal names the honored vocabulary, not just the failure.
    assert "add/move/update/delete" in message


def test_undeclared_operation_is_rejected_by_the_registry() -> None:
    from solweig_gpu.incremental import AdapterRegistryError

    with pytest.raises(
        AdapterRegistryError, match="does not support operation"
    ):
        make_adapter().validate(
            building_command("sculpt", new_state=dict(BLOCK_A)), context()
        )


def test_operation_state_shapes_add_and_delete() -> None:
    adapter = make_adapter()
    with pytest.raises(EditStateError, match="add.*must not carry"):
        adapter.validate(
            building_command(
                "add", old_state=dict(BLOCK_A), new_state=dict(BLOCK_A)
            ),
            context(),
        )
    with pytest.raises(EditStateError, match="add.*requires a new_state"):
        adapter.validate(building_command("add"), context())
    with pytest.raises(EditStateError, match="delete.*must not carry"):
        adapter.validate(
            building_command(
                "delete", old_state=dict(BLOCK_A), new_state=dict(BLOCK_A)
            ),
            context(),
        )
    with pytest.raises(EditStateError, match="delete.*requires an old_state"):
        adapter.validate(building_command("delete"), context())

    delete = adapter.validate(
        building_command("delete", old_state=dict(BLOCK_A)), context()
    )
    (change,) = delete.delta.objects
    assert change.before is not None and change.after is None


def test_move_requires_both_states_and_rejects_identity() -> None:
    adapter = make_adapter()
    with pytest.raises(EditStateError, match="move.*requires both"):
        adapter.validate(
            building_command("move", new_state=dict(BLOCK_A)), context()
        )
    with pytest.raises(EditStateError, match="no-op"):
        adapter.validate(
            building_command(
                "move", old_state=dict(BLOCK_A), new_state=dict(BLOCK_A)
            ),
            context(),
        )


def test_update_and_move_merge_over_old_state_losslessly() -> None:
    for operation in ("update", "move"):
        edit = make_adapter().validate(
            building_command(
                operation,
                old_state=dict(BLOCK_A),
                new_state=dict(BLOCK_A_TALLER),  # height only
            ),
            context(),
        )
        (change,) = edit.delta.objects
        # Both delta states carry the FULL property set (lossless
        # round-trip); the unspecified footprint carries over.
        assert change.before["height_m"] == 15.0
        assert change.after["height_m"] == 20.0
        assert change.before["footprint_m"] == change.after["footprint_m"]
        assert len(change.after["footprint_m"]) == 4
        # JSON-ish lists are normalized to frozen tuple rings.
        assert isinstance(change.after["footprint_m"], tuple)
        assert change.after["footprint_m"][0] == (400.0, 500.0)


def test_id_mismatch_between_states_is_refused() -> None:
    other = dict(BLOCK_A_TALLER)
    other["building_id"] = "block-b"
    with pytest.raises(EditStateError, match="different buildings"):
        make_adapter().validate(
            building_command(
                "update", old_state=dict(BLOCK_A), new_state=other
            ),
            context(),
        )


def test_missing_required_properties_are_refused() -> None:
    no_footprint = {
        "building_id": "block-a",
        "height_m": 15.0,
    }
    with pytest.raises(EditStateError, match="missing required"):
        make_adapter().validate(
            building_command("add", new_state=no_footprint), context()
        )
    no_height = {
        "building_id": "block-a",
        "footprint_m": BLOCK_A["footprint_m"],
    }
    with pytest.raises(EditStateError, match="missing required"):
        make_adapter().validate(
            building_command("add", new_state=no_height), context()
        )


def test_unknown_fields_are_rejected() -> None:
    for key in ("num_floors", "transmissivity", "walllimit", "albedo_b"):
        state = dict(BLOCK_A)
        state[key] = 1
        with pytest.raises(EditStateError, match="unknown building fields"):
            make_adapter().validate(
                building_command("add", new_state=state), context()
            )


def test_stale_revision_and_foreign_adapter_are_refused() -> None:
    adapter = make_adapter()
    with pytest.raises(EditStateError, match="stale scene revision"):
        adapter.validate(
            building_command(
                "add", revision=1, new_state=dict(BLOCK_A)
            ),
            context(revision=0),
        )
    foreign = EditCommand(
        edit_id="foreign",
        scenario_id="scenario-a",
        base_scene_revision=0,
        adapter_id="vegetation_geometry",
        operation="add",
        old_state=None,
        new_state=dict(BLOCK_A),
        requested_outputs=("utci",),
        requested_times=(3,),
    )
    with pytest.raises(EditStateError, match="targets adapter"):
        adapter.validate(foreign, context())


# ---------------------------------------------------------------------------
# Property schema: footprint and height domains
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "footprint",
    (
        ((400.0, 500.0), (420.0, 500.0)),  # two vertices: not a polygon
        ((400.0, 500.0),),  # one vertex
        (),  # empty
        "polygon",  # not a sequence of pairs
        ((400.0, 500.0, 0.0), (420.0, 500.0, 0.0), (420.0, 480.0, 0.0)),
        (400.0, 500.0, 420.0),  # flat sequence of numbers
    ),
)
def test_footprint_shape_violations_are_refused(footprint) -> None:
    state = dict(BLOCK_A)
    state["footprint_m"] = footprint
    with pytest.raises(EditStateError, match="footprint_m"):
        make_adapter().validate(
            building_command("add", new_state=state), context()
        )


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), float("-inf")))
def test_non_finite_coordinates_are_refused(bad) -> None:
    ring = list(BLOCK_A["footprint_m"])
    ring[1] = [bad, 500.0]
    state = dict(BLOCK_A)
    state["footprint_m"] = ring
    with pytest.raises(EditStateError, match="must be finite"):
        make_adapter().validate(
            building_command("add", new_state=state), context()
        )


def test_coordinate_type_spellings_are_single_domain() -> None:
    for bad in (True, "400", None):
        ring = list(BLOCK_A["footprint_m"])
        ring[0] = [bad, 500.0]
        state = dict(BLOCK_A)
        state["footprint_m"] = ring
        with pytest.raises(EditStateError, match="must be a number"):
            make_adapter().validate(
                building_command("add", new_state=state), context()
            )


@pytest.mark.parametrize(
    "height",
    (
        0.0,  # zero-height is a delete, not a massing edit
        -5.0,  # negative height
        HEIGHT_MAX_M + 1.0,  # above the plausibility fence
        float("nan"),
        float("inf"),
        float("-inf"),
        True,  # bool is an int subclass; one spelling per domain
        "15",
    ),
)
def test_height_domain_is_enforced(height) -> None:
    state = dict(BLOCK_A)
    state["height_m"] = height
    with pytest.raises(EditStateError, match="height_m"):
        make_adapter().validate(
            building_command("add", new_state=state), context()
        )


def test_height_bounds_are_inclusive_at_the_top_only() -> None:
    # The upper plausibility bound itself is editable; only strict
    # positivity excludes the bottom.
    for height in (HEIGHT_MIN_M + 0.5, 3.0, 828.0, HEIGHT_MAX_M):
        state = dict(BLOCK_A)
        state["height_m"] = height
        edit = make_adapter().validate(
            building_command("add", new_state=state), context()
        )
        assert edit.delta.objects[0].after["height_m"] == float(height)


# ---------------------------------------------------------------------------
# FULL-ONLY planning shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("operation", ("add", "move", "update", "delete"))
def test_every_operation_plans_full_only(operation) -> None:
    adapter = make_adapter()
    if operation == "add":
        edit = adapter.validate(
            building_command("add", new_state=dict(BLOCK_A)), context()
        )
    elif operation == "delete":
        edit = adapter.validate(
            building_command("delete", old_state=dict(BLOCK_A)), context()
        )
    else:
        edit = adapter.validate(
            building_command(
                operation,
                old_state=dict(BLOCK_A),
                new_state=dict(BLOCK_A_TALLER),
            ),
            context(),
        )
    plan = adapter.impact_plan(edit, context())
    scopes = scopes_by_node(plan)
    assert scopes
    for node_id, scope in scopes.items():
        assert scope is SpatialScope.FULL, (operation, node_id)
        impact = plan_node(plan, node_id)
        assert impact.write_windows == ()
        assert impact.read_windows == ()
    assert plan.transport_window is None
    assert any(
        "full_only_initially" in reason for reason in plan.fallback_reasons
    ), plan.fallback_reasons


def test_full_dirty_set_and_reusable_set_both_directions() -> None:
    plan = make_adapter().impact_plan(update_edit(), context())
    planned = {impact.node_id for impact in plan.node_impacts}

    # Every geometry/vision-dependent stage is dirty and FULL...
    for node_id in BUILDING_DIRTY_STAGES:
        assert node_id in planned, node_id
        assert plan_node(plan, node_id).spatial_scope is SpatialScope.FULL
        assert node_id not in plan.reusable_nodes
    # ...and nothing else is reusable that should be dirty.
    assert planned == set(BUILDING_DIRTY_STAGES)
    # Every reusable candidate is reusable and absent from the impacts.
    for node_id in BUILDING_REUSABLE_NODES:
        assert node_id in plan.reusable_nodes, node_id
        assert node_id not in planned, node_id
    # wbgt: never planned, pruned with a recorded reason, never silently
    # dropped into either set.
    assert "wbgt" not in planned
    assert "wbgt" not in plan.reusable_nodes
    assert any("output wbgt" in reason for reason in plan.fallback_reasons)
    assert plan.changed_sources == (BUILDING_SOURCE_NODE,)


def test_temporal_scopes_replay_state_and_narrow_time_varying() -> None:
    plan = make_adapter().impact_plan(update_edit(), context())
    # Geometry caches are time-invariant (ALL = "no temporal restriction").
    for node_id in (
        "relative_geometry",
        "walls",
        "wall_aspect",
        "building_visibility",
        "vegetation_visibility",
        "svf",
    ):
        impact = plan_node(plan, node_id)
        assert impact.temporal_scope is TemporalScope.ALL, node_id
        assert impact.time_start is None and impact.time_stop is None
    # The stateful ground-heat accumulator replays from timestep 0 to the
    # requested bound; the time-varying stages narrow to the request.
    thermal = plan_node(plan, "surface_thermal_state")
    assert thermal.temporal_scope is TemporalScope.REPLAY
    assert (thermal.time_start, thermal.time_stop) == (0, 3)
    for node_id in ("time_shadow", "radiation", "tmrt", "utci"):
        impact = plan_node(plan, node_id)
        assert impact.temporal_scope is TemporalScope.ONE, node_id
        assert impact.time_start == 3
    assert plan.estimated_memory_bytes > 0


def test_delta_reports_no_windows_honestly() -> None:
    for edit in (add_edit(), update_edit()):
        delta = edit.delta
        assert delta.windows == ()
        assert delta.spatial_windows == ()
        assert not delta.is_noop


def test_walls_reason_cites_the_full_spatial_source() -> None:
    plan = make_adapter().impact_plan(update_edit(), context())
    assert (
        "full: upstream building_dsm is full-spatial"
        in plan_node(plan, "walls").reason
    )


# ---------------------------------------------------------------------------
# Engine parity and determinism
# ---------------------------------------------------------------------------


def test_adapter_plan_equals_engine_h2_plan_bitwise() -> None:
    # The engine's hand-built H2 delta smuggles a window in; the adapter's
    # real path reports no windows. The plans must still be identical:
    # the planner's full_only_initially remediation fires BEFORE any
    # window logic, and the adapter does not defeat it.
    adapter_plan = make_adapter().impact_plan(update_edit(), context())
    engine_plan = EditPlanner(grid=GRID, registry=builtin_registry()).plan(
        [hand_built_h2_edit()], initial_state()
    )
    assert adapter_plan == engine_plan
    assert hash(adapter_plan) == hash(engine_plan)
    assert adapter_plan.node_impacts == engine_plan.node_impacts
    assert adapter_plan.fallback_reasons == engine_plan.fallback_reasons
    assert adapter_plan.transport_window is None
    assert engine_plan.transport_window is None


def test_plan_determinism_repeat() -> None:
    adapter = make_adapter()
    edit = update_edit()
    first = adapter.impact_plan(edit, context())
    for _ in range(3):
        assert adapter.impact_plan(edit, context()) == first
    engine = EditPlanner(grid=GRID, registry=builtin_registry())
    once = engine.plan([edit], initial_state())
    assert engine.plan([edit], initial_state()) == once == first


def test_records_are_frozen_and_input_mutation_probe() -> None:
    payload = {
        "building_id": "block-a",
        "footprint_m": [
            [400.0, 500.0],
            [420.0, 500.0],
            [420.0, 480.0],
        ],
        "height_m": 15.0,
    }
    pristine = {
        "building_id": "block-a",
        "footprint_m": [
            [400.0, 500.0],
            [420.0, 500.0],
            [420.0, 480.0],
        ],
        "height_m": 15.0,
    }
    command = building_command("add", new_state=payload)
    # Mutating the caller's payload after command construction cannot
    # leak into the validated edit (frozen at every boundary).
    payload["height_m"] = 999.0
    payload["footprint_m"].append([0.0, 0.0])
    edit = make_adapter().validate(command, context())
    assert edit.delta.objects[0].after["height_m"] == 15.0
    assert len(edit.delta.objects[0].after["footprint_m"]) == 3

    delta = edit.delta
    same = make_adapter().validate(
        building_command("add", edit_id="bld-2", new_state=pristine),
        context(),
    ).delta
    assert delta == same and hash(delta) == hash(same)
    with pytest.raises(AttributeError):
        delta.objects = ()  # type: ignore[misc]
    # FrozenMapping payloads reject item assignment (TypeError from the
    # underlying MappingProxyType), so payload mutation is impossible too.
    with pytest.raises(TypeError):
        delta.objects[0].after["height_m"] = 1.0  # type: ignore[index]


# ---------------------------------------------------------------------------
# U-C mixed scope: building + vegetation + parameters in ONE plan
# ---------------------------------------------------------------------------


def veg_adapter() -> VegetationGeometryAdapter:
    return VegetationGeometryAdapter(sun_positions=SUNS, influence_config=CONFIG)


def mixed_edits():
    building = update_edit()
    vegetation = veg_adapter().validate(
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
        context(),
    )
    parameters = make_adapter_for_params().validate(
        EditCommand(
            edit_id="params-1",
            scenario_id="scenario-a",
            base_scene_revision=0,
            adapter_id=MODEL_PARAMETERS_ADAPTER_ID,
            operation="update",
            old_state=None,
            new_state={"Fup": 0.1},
            requested_outputs=("utci",),
            requested_times=(3,),
        ),
        context(),
    )
    return building, vegetation, parameters


def make_adapter_for_params():
    from solweig_gpu.incremental import ModelReceptorParametersAdapter

    return ModelReceptorParametersAdapter()


def test_mixed_batch_one_plan_building_forces_everything_full() -> None:
    building, vegetation, parameters = mixed_edits()
    planner = EditPlanner(grid=GRID, registry=builtin_registry())
    plan = planner.plan(
        [building, vegetation, parameters], initial_state()
    )

    # ONE valid topological plan; node order follows the graph's order.
    assert plan.changed_sources == (
        "building_dsm",
        "model_parameters",
        "vegetation_dsm",
    )
    order = default_edit_graph().topological_order()
    positions = [order.index(i.node_id) for i in plan.node_impacts]
    assert positions == sorted(positions)

    # The building edit forces the ENTIRE downstream chain FULL —
    # including the vegetation chain: relative_geometry is a shared FULL
    # parent, so vegetation_visibility cannot keep windows under a
    # building edit (engine-reviewer-verified mixed-scope semantics).
    scopes = scopes_by_node(plan)
    assert scopes
    for node_id, scope in scopes.items():
        assert scope is SpatialScope.FULL, node_id
        assert plan_node(plan, node_id).write_windows == ()
    assert plan.transport_window is None
    assert (
        "full: upstream relative_geometry is full-spatial"
        in plan_node(plan, "vegetation_visibility").reason
    )
    assert any(
        "full_only_initially" in reason for reason in plan.fallback_reasons
    )
    assert any(
        "full-spatial adapter model_receptor_parameters" in reason
        for reason in plan.fallback_reasons
    )
    assert "wbgt" not in {i.node_id for i in plan.node_impacts}
    assert planner.plan(
        [building, vegetation, parameters], initial_state()
    ) == plan


def test_mixed_batch_contrast_vegetation_keeps_windows_without_building() -> None:
    # The same vegetation + parameter batch WITHOUT the building edit
    # keeps vegetation's WINDOWS (the U-C mixed-scope ruling): it is the
    # building edit — not the batch shape — that demotes the chain.
    _, vegetation, parameters = mixed_edits()
    planner = EditPlanner(grid=GRID, registry=builtin_registry())
    plan = planner.plan([vegetation, parameters], initial_state())
    scopes = scopes_by_node(plan)
    (veg_window,) = vegetation.delta.windows
    for node_id in (
        "relative_geometry",
        "vegetation_visibility",
        "svf",
        "time_shadow",
    ):
        assert scopes[node_id] is SpatialScope.WINDOWS, node_id
        assert plan_node(plan, node_id).write_windows == (veg_window,)
    for node_id in ("radiation", "surface_thermal_state", "tmrt", "utci"):
        assert scopes[node_id] is SpatialScope.FULL, node_id
    assert plan.transport_window == veg_window


# ---------------------------------------------------------------------------
# Transaction staging and the staging door
# ---------------------------------------------------------------------------


def test_apply_stages_rollback_safe_deltas() -> None:
    from solweig_gpu.incremental import ScenarioTransaction, TransactionError

    adapter = make_adapter()
    edit = update_edit()
    transaction = ScenarioTransaction(scenario_id="scenario-a")
    adapter.apply_source_delta(edit.delta, transaction)
    assert transaction.staged_deltas() == (edit.delta,)
    committed = transaction.commit()
    assert committed == (edit.delta,)
    with pytest.raises(TransactionError):
        transaction.rollback()

    rolled = ScenarioTransaction(scenario_id="scenario-a")
    adapter.apply_source_delta(edit.delta, rolled)
    rolled.rollback()
    assert rolled.staged_deltas() == ()


def test_apply_rejects_foreign_deltas() -> None:
    from solweig_gpu.incremental import ScenarioTransaction

    adapter = make_adapter()
    transaction = ScenarioTransaction(scenario_id="scenario-a")
    with pytest.raises(SourceDeltaError, match="expected BuildingMassing"):
        adapter.apply_source_delta(
            VegetationObjectDelta("vegetation_dsm", VEG_ADAPTER_ID), transaction
        )
    with pytest.raises(SourceDeltaError, match="belongs to adapter"):
        adapter.apply_source_delta(
            BuildingMassingDelta(
                "building_dsm",
                "terrain_dem",
                (ObjectStateChange("b1", None, dict(BLOCK_A)),),
            ),
            transaction,
        )
    with pytest.raises(SourceDeltaError, match="targets source node"):
        adapter.apply_source_delta(
            BuildingMassingDelta(
                "dem",
                ADAPTER_ID,
                (ObjectStateChange("b1", None, dict(BLOCK_A)),),
            ),
            transaction,
        )
    assert transaction.staged_deltas() == ()


def hand_built_delta(
    before: dict | None = None,
    after: dict | None = None,
    object_id: str = "b1",
) -> BuildingMassingDelta:
    if before is None and after is None:
        after = dict(BLOCK_A)
    return BuildingMassingDelta(
        BUILDING_SOURCE_NODE,
        ADAPTER_ID,
        (ObjectStateChange(object_id, before, after),),
    )


@pytest.mark.parametrize(
    "before, after, match",
    (
        # Non-finite payloads ride the after-state (params HIGH fence).
        (None, {**BLOCK_A, "height_m": float("nan")}, "finite"),
        (None, {**BLOCK_A, "height_m": float("inf")}, "finite"),
        (
            {**BLOCK_A, "height_m": float("nan")},
            {**BLOCK_A, "height_m": 20.0},
            "finite",
        ),
        # Out-of-domain heights.
        (None, {**BLOCK_A, "height_m": 0.0}, "height_m"),
        (None, {**BLOCK_A, "height_m": -1.0}, "height_m"),
        (None, {**BLOCK_A, "height_m": HEIGHT_MAX_M + 1}, "height_m"),
        (None, {**BLOCK_A, "height_m": True}, "height_m"),
        # Footprint shape violations.
        (
            None,
            {**BLOCK_A, "footprint_m": ((400.0, 500.0), (420.0, 500.0))},
            "footprint_m",
        ),
        (
            None,
            {**BLOCK_A, "footprint_m": ((400.0, 500.0), ("x", 500.0), (1.0, 2.0))},
            "must be a number",
        ),
        # Unknown property smuggled into the payload.
        (None, {**BLOCK_A, "num_floors": 2}, "unknown building fields"),
        # Partial property set (the engine-test shape) is NOT a staged
        # payload this adapter ever produces.
        ({"height_m": 10.0}, {"height_m": 20.0}, "missing required"),
    ),
)
def test_staging_door_refuses_hand_built_payloads_pre_stage(
    before, after, match
) -> None:
    from solweig_gpu.incremental import ScenarioTransaction

    adapter = make_adapter()
    transaction = ScenarioTransaction(scenario_id="scenario-a")
    delta = hand_built_delta(before=before, after=after)
    with pytest.raises((EditStateError, SourceDeltaError), match=match):
        adapter.apply_source_delta(delta, transaction)
    # Refused PRE-stage: nothing landed in the transaction...
    assert transaction.staged_deltas() == ()
    # ...and the door stayed open for a legitimate payload afterwards.
    good = update_edit().delta
    adapter.apply_source_delta(good, transaction)
    assert transaction.staged_deltas() == (good,)


def test_staging_door_accepts_the_adapter_validated_payload() -> None:
    from solweig_gpu.incremental import ScenarioTransaction

    for edit in (add_edit(), update_edit()):
        transaction = ScenarioTransaction(scenario_id="scenario-a")
        make_adapter().apply_source_delta(edit.delta, transaction)
        assert transaction.staged_deltas() == (edit.delta,)


# ---------------------------------------------------------------------------
# The rasterizer executor seam
# ---------------------------------------------------------------------------


def test_massing_seam_round_trips_committed_deltas() -> None:
    edit = update_edit()
    (massing,) = massing_edits_from_deltas([edit.delta])
    assert isinstance(massing, MassingEdit)
    assert massing.building_id == "block-a"
    assert massing.before == BuildingSpec(
        "block-a",
        (
            (400.0, 500.0),
            (420.0, 500.0),
            (420.0, 480.0),
            (400.0, 480.0),
        ),
        15.0,
    )
    assert massing.after.height_m == 20.0
    assert massing.after.footprint_m == massing.before.footprint_m

    deleted = make_adapter().validate(
        building_command("delete", old_state=dict(BLOCK_A)), context()
    )
    (gone,) = massing_edits_from_deltas([deleted.delta])
    assert gone.before is not None and gone.after is None


def test_massing_seam_revalidates_and_rejects_foreign_deltas() -> None:
    with pytest.raises(SourceDeltaError, match="expected BuildingMassing"):
        massing_edits_from_deltas(
            [VegetationObjectDelta("vegetation_dsm", VEG_ADAPTER_ID)]
        )
    with pytest.raises(SourceDeltaError, match="belongs to adapter"):
        massing_edits_from_deltas(
            [
                BuildingMassingDelta(
                    "building_dsm",
                    "terrain_dem",
                    (ObjectStateChange("b1", None, dict(BLOCK_A)),),
                )
            ]
        )
    with pytest.raises(SourceDeltaError, match="targets source node"):
        massing_edits_from_deltas(
            [
                BuildingMassingDelta(
                    "dem",
                    ADAPTER_ID,
                    (ObjectStateChange("b1", None, dict(BLOCK_A)),),
                )
            ]
        )
    # The seam re-runs the full schema on hand-built payloads too.
    with pytest.raises((EditStateError, SourceDeltaError), match="finite"):
        massing_edits_from_deltas(
            [hand_built_delta(after={**BLOCK_A, "height_m": float("nan")})]
        )
    with pytest.raises((EditStateError, SourceDeltaError), match="height_m"):
        massing_edits_from_deltas(
            [hand_built_delta(after={**BLOCK_A, "height_m": 0.0})]
        )


def test_building_spec_construction_revalidates() -> None:
    with pytest.raises(EditStateError, match="at least"):
        BuildingSpec(
            "block-a",
            ((400.0, 500.0), (420.0, 500.0)),
            15.0,
        )
    with pytest.raises(EditStateError, match="finite"):
        BuildingSpec(
            "block-a",
            ((400.0, 500.0), (float("nan"), 500.0), (420.0, 480.0)),
            15.0,
        )
    with pytest.raises(EditStateError, match="height_m"):
        BuildingSpec(
            "block-a",
            ((400.0, 500.0), (420.0, 500.0), (420.0, 480.0)),
            0.0,
        )
    with pytest.raises(EditStateError, match="building_id"):
        BuildingSpec(
            "",
            ((400.0, 500.0), (420.0, 500.0), (420.0, 480.0)),
            15.0,
        )
    with pytest.raises(EditStateError, match="before or an after"):
        MassingEdit(building_id="block-a", before=None, after=None)


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------


def test_preview_descriptor_is_a_full_tile_disclosure() -> None:
    descriptor = make_adapter().preview_descriptor(update_edit(), context())
    assert descriptor.kind == "massing_and_shadow"
    assert descriptor.immediate is True
    text = " ".join(descriptor.limitations)
    assert "FULL tile" in text
    assert "preview_is_not_exact" in text
    # The wall threshold disclosure (walls_aspect.py walllimit = 3.0).
    assert "3 m" in text
    # The wind fence disclosure (UEDIT-010).
    assert "wind" in text
    # raster_patch refusal is disclosed to the frontend too.
    assert "raster_patch" in text
