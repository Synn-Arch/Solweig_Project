# SPDX-License-Identifier: GPL-3.0-only
"""Wave U-B engine-core tests: graph, registry, deltas, planner.

Covers the drift guards (code-defined graph/registry must mirror the YAML
specs), invalidation semantics per source family, heterogeneous-batch
planning with mixed scopes, the registry's enforced lead rulings
(land-cover water fence, vegetation transmissivity rejection, model
parameter fences, wind fence), and planner safety behavior (stale
revisions, locality demotion, view-only zero-stage plans, determinism).
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from solweig_gpu.incremental import (
    LANDCOVER_FENCED_CLASSES,
    LANDCOVER_VALID_CLASSES,
    MODEL_PARAMETERS_BLOCKED,
    MODEL_PARAMETERS_SAFE,
    AdapterMetadata,
    AdapterRegistry,
    AdapterRegistryError,
    AdapterSchemaError,
    AdapterStatus,
    BuildingMassingDelta,
    ConservativeSafetyPolicy,
    EditCommand,
    EditGraph,
    EditGraphError,
    EditPlanner,
    ForcingChange,
    ForcingDelta,
    GraphNode,
    ImpactPlan,
    LandCoverPaintDelta,
    LandCoverPaintPatch,
    ModelParameterChange,
    ModelParameterDelta,
    NodeImpact,
    ObjectStateChange,
    OutputSelectionDelta,
    PlanningError,
    RasterGrid,
    RasterWindow,
    SceneGraphState,
    SpatialScope,
    TemporalScope,
    ValidatedEdit,
    VegetationObjectDelta,
    builtin_adapter_metadata,
    builtin_registry,
    coalesce_source_deltas,
    default_edit_graph,
    freeze_mapping,
)

SPEC_DIR = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "incremental_design_tool"
    / "universal_editing"
)

GRID = RasterGrid(500, 500, 2.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def veg_command(
    revision: int = 0,
    edit_id: str = "veg-1",
    operation: str = "add",
    times: tuple[int, ...] | None = (3,),
) -> EditCommand:
    return EditCommand(
        edit_id=edit_id,
        scenario_id="scenario-a",
        base_scene_revision=revision,
        adapter_id="vegetation_geometry",
        operation=operation,
        old_state=None,
        new_state={"x_m": 200.0, "y_m": 200.0, "height_m": 10.0},
        requested_outputs=("utci",),
        requested_times=times,
    )


def veg_delta(
    before: dict | None = None,
    after: dict | None = None,
    windows: tuple[RasterWindow, ...] = (RasterWindow(96, 160, 96, 160),),
    object_id: str = "tree-1",
) -> VegetationObjectDelta:
    if before is None and after is None:
        after = {"x_m": 200.0, "y_m": 200.0, "height_m": 10.0}
    return VegetationObjectDelta(
        source_node_id="vegetation_dsm",
        adapter_id="vegetation_geometry",
        objects=(ObjectStateChange(object_id, before, after),),
        windows=windows,
    )


def veg_edit(
    revision: int = 0,
    delta: VegetationObjectDelta | None = None,
    edit_id: str = "veg-1",
    operation: str = "add",
    times: tuple[int, ...] | None = (3,),
) -> ValidatedEdit:
    return ValidatedEdit(
        command=veg_command(revision, edit_id, operation, times),
        adapter_id="vegetation_geometry",
        schema_version=1,
        source_node_id="vegetation_dsm",
        delta=delta if delta is not None else veg_delta(),
    )


def paint_patch(
    row: int = 200, col: int = 200, rows: int = 4, cols: int = 5
) -> LandCoverPaintPatch:
    area = rows * cols
    return LandCoverPaintPatch(
        window=RasterWindow(row, row + rows, col, col + cols),
        before_classes=(2,) * area,
        after_classes=(5,) * area,
    )


def landcover_edit(
    revision: int = 0,
    patches: tuple[LandCoverPaintPatch, ...] | None = None,
    edit_id: str = "lc-1",
) -> ValidatedEdit:
    return ValidatedEdit(
        command=EditCommand(
            edit_id=edit_id,
            scenario_id="scenario-a",
            base_scene_revision=revision,
            adapter_id="landcover_surface",
            operation="paint",
            old_state=None,
            new_state={"classes": (5,)},
            requested_outputs=("utci",),
            requested_times=(3,),
        ),
        adapter_id="landcover_surface",
        schema_version=1,
        source_node_id="landcover",
        delta=LandCoverPaintDelta(
            "landcover", "landcover_surface", patches or (paint_patch(),)
        ),
    )


def met_edit(
    revision: int = 0,
    times: tuple[int, ...] | None = (3,),
    edit_id: str = "met-1",
) -> ValidatedEdit:
    return ValidatedEdit(
        command=EditCommand(
            edit_id=edit_id,
            scenario_id="scenario-a",
            base_scene_revision=revision,
            adapter_id="meteorological_forcing",
            operation="update_time_row",
            old_state={"air_temperature": 20.0},
            new_state={"air_temperature": 26.0},
            requested_outputs=("utci",),
            requested_times=times,
        ),
        adapter_id="meteorological_forcing",
        schema_version=1,
        source_node_id="meteorology",
        delta=ForcingDelta(
            "meteorology",
            "meteorological_forcing",
            (ForcingChange("air_temperature", 3, 20.0, 26.0),),
        ),
    )


def view_edit(revision: int = 0, edit_id: str = "view-1") -> ValidatedEdit:
    return ValidatedEdit(
        command=EditCommand(
            edit_id=edit_id,
            scenario_id="scenario-a",
            base_scene_revision=revision,
            adapter_id="output_view",
            operation="select_layer",
            old_state={"layers": ("utci",)},
            new_state={"layers": ("tmrt",)},
            requested_outputs=("tmrt",),
            requested_times=None,
        ),
        adapter_id="output_view",
        schema_version=1,
        source_node_id="output_selection",
        delta=OutputSelectionDelta(
            "output_selection", "output_view", ("utci",), ("tmrt",)
        ),
    )


def make_planner(**kwargs) -> EditPlanner:
    defaults = dict(grid=GRID, registry=builtin_registry())
    defaults.update(kwargs)
    return EditPlanner(**defaults)


def initial_state() -> SceneGraphState:
    return SceneGraphState.initial(default_edit_graph())


def scopes_by_node(plan: ImpactPlan) -> dict[str, SpatialScope]:
    return {impact.node_id: impact.spatial_scope for impact in plan.node_impacts}


# ---------------------------------------------------------------------------
# Drift guard: code-defined graph mirrors dependency_graph.yaml
# ---------------------------------------------------------------------------


def test_graph_matches_yaml_nodes_and_edges() -> None:
    spec = yaml.safe_load((SPEC_DIR / "dependency_graph.yaml").read_text())
    graph = default_edit_graph()
    yaml_nodes = {node["id"]: node["kind"] for node in spec["nodes"]}
    code_nodes = {node.node_id: node.kind.value for node in graph.nodes}
    assert code_nodes == yaml_nodes
    assert len(code_nodes) == 22
    yaml_edge_list = [tuple(edge) for edge in spec["edges"]]
    # A duplicated edge in the YAML would vanish in the set comparison;
    # parity is on multisets, not just sets (review finding L4).
    assert len(yaml_edge_list) == len(set(yaml_edge_list)) == 36
    yaml_edges = set(yaml_edge_list)
    code_edges = set(graph.edges)
    assert code_edges == yaml_edges
    assert len(code_edges) == 36


def test_registry_matches_yaml_entries() -> None:
    spec = yaml.safe_load((SPEC_DIR / "edit_registry.yaml").read_text())
    yaml_adapters = {entry["id"]: entry for entry in spec["adapters"]}
    metadata_by_id = {entry.id: entry for entry in builtin_adapter_metadata()}
    assert set(metadata_by_id) == set(yaml_adapters)

    for adapter_id, entry in yaml_adapters.items():
        metadata = metadata_by_id[adapter_id]
        assert metadata.group == entry["group"]
        assert metadata.status.value == entry["status"]
        assert metadata.source_nodes == tuple(entry["source_nodes"])
        assert metadata.operations == tuple(entry["operations"])
        assert metadata.nominal_spatial_scope == entry["nominal_spatial_scope"]
        assert metadata.temporal_scope == entry["temporal_scope"]
        assert metadata.preview == entry["preview"]
        assert metadata.validation_fixtures == tuple(entry["validation_fixtures"])
        assert metadata.notes == entry.get("notes", "")
        assert metadata.valid_classes == frozenset(entry.get("valid_classes", ()))
        assert metadata.parameters_safe == frozenset(
            entry.get("parameters_safe", ())
        )
        assert metadata.parameters_blocked == frozenset(
            entry.get("parameters_blocked", ())
        )
        assert metadata.producible_incremental == tuple(
            entry.get("producible_incremental", ())
        )


def test_builtin_registry_bootstraps_all_entries() -> None:
    registry = builtin_registry()
    assert len(registry.adapter_ids()) == 9
    assert registry.get_metadata("vegetation_geometry").status is (
        AdapterStatus.ADAPTER_REQUIRED
    )


# ---------------------------------------------------------------------------
# Graph structure: acyclicity, determinism, closures
# ---------------------------------------------------------------------------


def test_topological_order_is_valid_and_deterministic() -> None:
    graph = default_edit_graph()
    order = graph.topological_order()
    assert len(order) == len(graph.node_ids)
    position = {node_id: index for index, node_id in enumerate(order)}
    for producer, consumer in graph.edges:
        assert position[producer] < position[consumer]
    # Independent reimplementation: greedy Kahn with lexicographic choice.
    remaining = {
        node_id: len(graph.parents(node_id)) for node_id in graph.node_ids
    }
    reference: list[str] = []
    pending = sorted(node_id for node_id, count in remaining.items() if count == 0)
    while pending:
        node_id = pending.pop(0)
        reference.append(node_id)
        for child in graph.children(node_id):
            remaining[child] -= 1
            if remaining[child] == 0:
                pending.append(child)
        pending.sort()
    assert tuple(reference) == order
    assert graph.topological_order() == order  # stable across calls


def test_cycle_raises_edit_graph_error() -> None:
    nodes = (
        GraphNode("a", _source()),
        GraphNode("b", _derived()),
        GraphNode("c", _derived()),
    )
    with pytest.raises(EditGraphError, match="cycle"):
        EditGraph(nodes, (("a", "b"), ("b", "c"), ("c", "b")))


def test_unknown_node_and_duplicate_edges_rejected() -> None:
    nodes = (GraphNode("a", _source()), GraphNode("b", _derived()))
    with pytest.raises(EditGraphError, match="unknown node"):
        EditGraph(nodes, (("a", "b"), ("b", "ghost")))
    with pytest.raises(EditGraphError, match="duplicate"):
        EditGraph(nodes, (("a", "b"), ("a", "b")))


def _source() -> object:
    from solweig_gpu.incremental import NodeKind

    return NodeKind.SOURCE


def _derived() -> object:
    from solweig_gpu.incremental import NodeKind

    return NodeKind.DERIVED


def test_scene_state_bumps_versions_along_closure() -> None:
    graph = default_edit_graph()
    state = SceneGraphState.initial(graph)
    assert state.scene_revision == 0
    advanced = state.advance(["vegetation_dsm"])
    assert advanced.scene_revision == 1
    dirty = {
        "vegetation_dsm",
        "relative_geometry",
        "vegetation_visibility",
        "svf",
        "time_shadow",
        "radiation",
        "surface_thermal_state",
        "tmrt",
        "utci",
        "wbgt",
    }
    for node_id in graph.node_ids:
        expected = 1 if node_id in dirty else 0
        assert advanced.version(node_id) == expected, node_id
    assert state.version("svf") == 0  # original snapshot untouched


# ---------------------------------------------------------------------------
# Invalidation semantics per source family
# ---------------------------------------------------------------------------


def dirty_nodes(plan: ImpactPlan) -> set[str]:
    graph = default_edit_graph()
    all_nodes = set(graph.node_ids)
    return all_nodes - set(plan.reusable_nodes)


def test_vegetation_edit_invalidates_vegetation_chain_only() -> None:
    plan = make_planner().plan([veg_edit()], initial_state())
    dirty = dirty_nodes(plan)
    assert {
        "vegetation_dsm",
        "relative_geometry",
        "vegetation_visibility",
        "svf",
        "time_shadow",
        "radiation",
        "surface_thermal_state",
        "tmrt",
        "utci",
        "wbgt",
    } <= dirty
    # Walls/building side stays reusable; svf depends on both visibilities
    # so it IS dirty when vegetation_visibility is (reconciled edges).
    for reusable in (
        "dem",
        "building_dsm",
        "landcover",
        "meteorology",
        "selected_date_time",
        "wind_coefficients",
        "model_parameters",
        "output_selection",
        "walls",
        "wall_aspect",
        "building_visibility",
        "solar_atmospheric_state",
    ):
        assert reusable in plan.reusable_nodes
    assert "svf" not in plan.reusable_nodes


def test_building_edit_invalidates_walls_to_time_shadow_chain() -> None:
    command = EditCommand(
        edit_id="b-1",
        scenario_id="scenario-a",
        base_scene_revision=0,
        adapter_id="building_geometry",
        operation="update",
        old_state={"height_m": 10.0},
        new_state={"height_m": 20.0},
        requested_outputs=("utci",),
        requested_times=(3,),
    )
    delta = BuildingMassingDelta(
        "building_dsm",
        "building_geometry",
        (ObjectStateChange("b1", {"height_m": 10.0}, {"height_m": 20.0}),),
        (RasterWindow(100, 200, 100, 200),),
    )
    edit = ValidatedEdit(
        command, "building_geometry", 1, "building_dsm", delta
    )
    plan = make_planner().plan([edit], initial_state())
    dirty = dirty_nodes(plan)
    for node_id in (
        "building_dsm",
        "relative_geometry",
        "walls",
        "wall_aspect",
        "building_visibility",
        "svf",
        "time_shadow",
        "radiation",
        "surface_thermal_state",
        "tmrt",
        "utci",
        "wbgt",
    ):
        assert node_id in dirty
    assert "dem" in plan.reusable_nodes
    assert "landcover" in plan.reusable_nodes
    assert "meteorology" in plan.reusable_nodes


def test_landcover_edit_skips_visibility_and_svf_nodes() -> None:
    plan = make_planner().plan([landcover_edit()], initial_state())
    dirty = dirty_nodes(plan)
    assert dirty == {
        "landcover",
        "radiation",
        "surface_thermal_state",
        "tmrt",
        "utci",
        "wbgt",
    }
    for reusable in (
        "relative_geometry",
        "walls",
        "wall_aspect",
        "building_visibility",
        "vegetation_visibility",
        "svf",
        "solar_atmospheric_state",
        "time_shadow",
    ):
        assert reusable in plan.reusable_nodes


def test_meteorology_edit_keeps_geometry_caches_reusable() -> None:
    plan = make_planner().plan([met_edit()], initial_state())
    dirty = dirty_nodes(plan)
    assert dirty == {
        "meteorology",
        "solar_atmospheric_state",
        "time_shadow",
        "radiation",
        "surface_thermal_state",
        "tmrt",
        "utci",
        "wbgt",
    }
    for reusable in (
        "relative_geometry",
        "walls",
        "wall_aspect",
        "building_visibility",
        "vegetation_visibility",
        "svf",
    ):
        assert reusable in plan.reusable_nodes


def test_time_edit_invalidates_solar_state_downstream() -> None:
    command = EditCommand(
        edit_id="t-1",
        scenario_id="scenario-a",
        base_scene_revision=0,
        adapter_id="selected_date_time",
        operation="select",
        old_state={"time_index": 3},
        new_state={"time_index": 5},
        requested_outputs=("utci",),
        requested_times=(5,),
    )
    delta = ForcingDelta(
        "selected_date_time",
        "selected_date_time",
        (ForcingChange("selected_time", 5, 3, 5),),
    )
    edit = ValidatedEdit(command, "selected_date_time", 1, "selected_date_time", delta)
    plan = make_planner().plan([edit], initial_state())
    dirty = dirty_nodes(plan)
    assert dirty == {
        "selected_date_time",
        "solar_atmospheric_state",
        "time_shadow",
        "radiation",
        "surface_thermal_state",
        "tmrt",
        "utci",
        "wbgt",
    }
    scopes = scopes_by_node(plan)
    # Full-downstream-only adapter: solar state onwards is full spatial.
    assert scopes["solar_atmospheric_state"] is SpatialScope.FULL
    assert "svf" in plan.reusable_nodes


# ---------------------------------------------------------------------------
# Heterogeneous batches, coalescing, mixed scopes
# ---------------------------------------------------------------------------


def test_heterogeneous_batch_single_plan_with_mixed_scopes() -> None:
    plan = make_planner().plan(
        [veg_edit(edit_id="v"), landcover_edit(edit_id="l"), met_edit(edit_id="m")],
        initial_state(),
    )
    assert plan.changed_sources == ("landcover", "meteorology", "vegetation_dsm")
    scopes = scopes_by_node(plan)
    # Local geometry stages keep their windows despite the global forcing.
    assert scopes["relative_geometry"] is SpatialScope.WINDOWS
    assert scopes["vegetation_visibility"] is SpatialScope.WINDOWS
    assert scopes["svf"] is SpatialScope.WINDOWS
    # Met-driven nodes are full for their own products only.
    for node_id in (
        "solar_atmospheric_state",
        "time_shadow",
        "radiation",
        "surface_thermal_state",
        "tmrt",
        "utci",
    ):
        assert scopes[node_id] is SpatialScope.FULL, node_id
    # wbgt is never planned (M1): every edit requested only utci, so the
    # disabled output is pruned with a recorded reason, not silently.
    assert "wbgt" not in scopes
    assert any(
        "output wbgt" in reason for reason in plan.fallback_reasons
    )
    # Per-node write windows survive on the local nodes (U-C ruling shape).
    records = dict(
        (node_id, scope) for node_id, scope, _ in plan.write_scope_records
    )
    assert records["vegetation_visibility"] is SpatialScope.WINDOWS
    veg_windows = dict(
        (node_id, windows) for node_id, _, windows in plan.write_scope_records
    )["vegetation_visibility"]
    assert veg_windows  # local node kept its windows
    assert plan.transport_window is not None
    # One plan, ordered topologically: every producer precedes consumers.
    graph = default_edit_graph()
    position = {impact.node_id: idx for idx, impact in enumerate(plan.node_impacts)}
    for producer, consumer in graph.edges:
        if producer in position and consumer in position:
            assert position[producer] < position[consumer]


def test_same_node_edits_coalesce_deterministically() -> None:
    added = {"x_m": 200.0, "y_m": 200.0, "height_m": 10.0}
    grown = {"x_m": 200.0, "y_m": 200.0, "height_m": 14.0}
    first = veg_delta(before=None, after=added, object_id="tree-1")
    second = veg_delta(before=added, after=grown, object_id="tree-1")
    coalesced = coalesce_source_deltas([first, second])
    assert isinstance(coalesced, VegetationObjectDelta)
    assert len(coalesced.objects) == 1
    assert coalesced.objects[0].before is None  # first old state retained
    assert coalesced.objects[0].after == freeze_mapping(grown)  # final new state

    plan = make_planner().plan(
        [
            veg_edit(delta=first, edit_id="v1"),
            veg_edit(delta=second, edit_id="v2", operation="update"),
        ],
        initial_state(),
    )
    assert plan.changed_sources == ("vegetation_dsm",)


def test_add_then_delete_coalesces_to_noop_zero_stage_plan() -> None:
    added = {"x_m": 200.0, "y_m": 200.0, "height_m": 10.0}
    add = veg_delta(before=None, after=added, object_id="tree-1")
    delete = veg_delta(before=added, after=None, object_id="tree-1")
    plan = make_planner().plan(
        [
            veg_edit(delta=add, edit_id="v1"),
            veg_edit(delta=delete, edit_id="v2", operation="delete"),
        ],
        initial_state(),
    )
    assert plan.node_impacts == ()
    assert plan.changed_sources == ()
    assert any("no-op" in reason for reason in plan.fallback_reasons)


def test_identical_batches_produce_identical_plans() -> None:
    planner = make_planner()
    edits = [veg_edit(), landcover_edit(), met_edit()]
    plan_a = planner.plan(edits, initial_state())
    plan_b = planner.plan(edits, initial_state())
    assert plan_a == plan_b
    assert hash(plan_a) == hash(plan_b)
    assert repr(plan_a) == repr(plan_b)


# ---------------------------------------------------------------------------
# Registry enforcement (UEDIT-001/002 shape + lead rulings)
# ---------------------------------------------------------------------------


def test_unknown_adapter_and_operation_rejected() -> None:
    registry = builtin_registry()
    with pytest.raises(AdapterRegistryError, match="unknown adapter"):
        registry.get_metadata("no_such_adapter")
    with pytest.raises(AdapterRegistryError, match="does not support"):
        registry.validate_operation("vegetation_geometry", "raster_patch")

    planner = make_planner()
    command = EditCommand(
        edit_id="x",
        scenario_id="scenario-a",
        base_scene_revision=0,
        adapter_id="ghost_adapter",
        operation="add",
        old_state=None,
        new_state={},
        requested_outputs=("utci",),
        requested_times=None,
    )
    edit = ValidatedEdit(
        command,
        "ghost_adapter",
        1,
        "vegetation_dsm",
        VegetationObjectDelta(
            "vegetation_dsm",
            "ghost_adapter",
            (ObjectStateChange("tree-1", None, {"height_m": 10.0}),),
            (RasterWindow(0, 8, 0, 8),),
        ),
    )
    with pytest.raises(PlanningError, match="UEDIT-001|unknown adapter"):
        planner.plan([edit], initial_state())

    bad_operation = EditCommand(
        edit_id="x",
        scenario_id="scenario-a",
        base_scene_revision=0,
        adapter_id="vegetation_geometry",
        operation="raster_patch",  # declared not-implemented for per-tree model
        old_state=None,
        new_state={},
        requested_outputs=("utci",),
        requested_times=None,
    )
    with pytest.raises(PlanningError, match="does not support"):
        planner.plan(
            [
                ValidatedEdit(
                    bad_operation, "vegetation_geometry", 1, "vegetation_dsm",
                    veg_delta(),
                )
            ],
            initial_state(),
        )


def test_water_class_7_is_fenced_everywhere() -> None:
    assert LANDCOVER_FENCED_CLASSES == frozenset({7})
    registry = builtin_registry()
    # Registering an adapter that exposes water is an error.
    metadata = AdapterMetadata(
        id="bad_landcover",
        group="surface",
        status=AdapterStatus.ADAPTER_REQUIRED,
        source_nodes=("landcover",),
        operations=("paint",),
        nominal_spatial_scope="windows",
        temporal_scope="replay_if_stateful",
        preview="surface_material",
        validation_fixtures=(),
        valid_classes=frozenset({1, 2, 5, 6, 7}),
    )
    with pytest.raises(AdapterRegistryError, match="water.*fenced|fenced"):
        registry.register_metadata(metadata)
    # Painting water through the builtin adapter is an error too.
    with pytest.raises(AdapterSchemaError, match="fenced"):
        registry.validate_edit_state("landcover_surface", {"classes": 7})
    with pytest.raises(AdapterSchemaError, match="fenced"):
        registry.validate_edit_state("landcover_surface", {"classes": (1, 7)})
    # Valid classes pass.
    registry.validate_edit_state("landcover_surface", {"classes": (1, 5)})
    registry.validate_edit_state("landcover_surface", {"classes": 6})
    assert LANDCOVER_VALID_CLASSES == frozenset({1, 2, 5, 6})


def test_vegetation_transmissivity_property_rejected() -> None:
    registry = builtin_registry()
    with pytest.raises(AdapterSchemaError, match="transmissivity"):
        registry.validate_edit_state(
            "vegetation_geometry", {"height_m": 12.0, "transmissivity": 0.2}
        )
    registry.validate_edit_state(
        "vegetation_geometry", {"height_m": 12.0, "trunk_ratio": 0.3}
    )
    # Registering a vegetation adapter without the declared rejection fails.
    metadata = AdapterMetadata(
        id="veg_no_rejection",
        group="geometry",
        status=AdapterStatus.ADAPTER_REQUIRED,
        source_nodes=("vegetation_dsm",),
        operations=("add",),
        nominal_spatial_scope="directional_windows",
        temporal_scope="replay_if_stateful",
        preview="vegetation_geometry_and_shadow",
        validation_fixtures=(),
    )
    with pytest.raises(AdapterRegistryError, match="transmissivity"):
        builtin_registry().register_metadata(metadata)


def test_new3_transmissivity_casing_variants_rejected() -> None:
    """u-b-reviewer NEW-3: engine-side fence is case-insensitive.

    The adapter layer's closed property vocabulary is the first line of
    defense; this backstop must also catch casing/nested-lowercase spellings
    that reach validate_edit_state directly.
    """
    registry = builtin_registry()
    for variant in ("Transmissivity", "TRANSMISSIVITY", "tRaNsMiSsIvItY"):
        # bare key...
        with pytest.raises(AdapterSchemaError, match=variant):
            registry.validate_edit_state(
                "vegetation_geometry", {"height_m": 12.0, variant: 0.2}
            )
    # the lowercase fence itself still fires with the physics rationale
    # (citation synced to the current utci_process.py line, u-d1 sweep)
    with pytest.raises(AdapterSchemaError, match="utci_process.py:649"):
        registry.validate_edit_state(
            "vegetation_geometry", {"height_m": 12.0, "TRANSMISSIVITY": 0.2}
        )


def test_model_parameters_safe_and_blocked_sets_enforced() -> None:
    registry = builtin_registry()
    blocked = sorted(MODEL_PARAMETERS_BLOCKED)
    assert blocked == [
        "albedo_g",
        "location",
        "onlyglobal",
        "patch_option",
        "scale",
        "utc",
        "walllimit",
    ]
    for name in ("patch_option", "scale", "location", "utc", "walllimit",
                 "onlyglobal", "albedo_g"):
        with pytest.raises(AdapterSchemaError, match="blocks parameters"):
            registry.validate_edit_state(
                "model_receptor_parameters", {name: 2}
            )
    with pytest.raises(AdapterSchemaError, match="does not define"):
        registry.validate_edit_state(
            "model_receptor_parameters", {"mystery_knob": 1.0}
        )
    registry.validate_edit_state(
        "model_receptor_parameters", {"albedo_b": 0.2, "transVeg": 0.03}
    )
    # A registration dropping any blocked parameter is refused.
    metadata = AdapterMetadata(
        id="params_leaky",
        group="advanced",
        status=AdapterStatus.ADAPTER_REQUIRED,
        source_nodes=("model_parameters",),
        operations=("update",),
        nominal_spatial_scope="full_downstream_stage_specific",
        temporal_scope="affected_times",
        preview="parameter_scope_disclosure",
        validation_fixtures=(),
        parameters_safe=MODEL_PARAMETERS_SAFE,
        parameters_blocked=frozenset({"scale"}),
    )
    with pytest.raises(AdapterRegistryError, match="must block"):
        builtin_registry().register_metadata(metadata)


def test_wind_coefficients_are_fenced() -> None:
    # Direct wind adapters must register as scientific_extension.
    metadata = AdapterMetadata(
        id="wind_direct",
        group="environment",
        status=AdapterStatus.ADAPTER_REQUIRED,
        source_nodes=("wind_coefficients",),
        operations=("paint",),
        nominal_spatial_scope="full",
        temporal_scope="all",
        preview="unsupported_disclosure",
        validation_fixtures=(),
    )
    with pytest.raises(AdapterRegistryError, match="scientific_extension"):
        builtin_registry().register_metadata(metadata)
    # Planning any scientific_extension batch is refused (never silent).
    command = EditCommand(
        edit_id="w",
        scenario_id="scenario-a",
        base_scene_revision=0,
        adapter_id="dynamic_wind_from_geometry",
        operation="rebuild",
        old_state=None,
        new_state={},
        requested_outputs=("utci",),
        requested_times=None,
    )
    delta = BuildingMassingDelta(
        "building_dsm",
        "dynamic_wind_from_geometry",
        (ObjectStateChange("b1", {"height_m": 10.0}, {"height_m": 12.0}),),
        (RasterWindow(0, 16, 0, 16),),
    )
    edit = ValidatedEdit(
        command, "dynamic_wind_from_geometry", 1, "building_dsm", delta
    )
    with pytest.raises(PlanningError, match="fenced"):
        make_planner().plan([edit], initial_state())


def test_view_only_adapter_registration_rules() -> None:
    metadata = AdapterMetadata(
        id="fake_output",
        group="analysis",
        status=AdapterStatus.VIEW_ONLY,
        source_nodes=("utci",),  # wrong node: must own output_selection
        operations=("select_layer",),
        nominal_spatial_scope="none",
        temporal_scope="none",
        preview="immediate",
        validation_fixtures=(),
    )
    with pytest.raises(AdapterRegistryError, match="output_selection"):
        builtin_registry().register_metadata(metadata)


def test_conflicting_duplicate_adapter_id_rejected() -> None:
    registry = builtin_registry()
    metadata = AdapterMetadata(
        id="vegetation_geometry",
        group="geometry",
        status=AdapterStatus.ADAPTER_REQUIRED,
        source_nodes=("vegetation_dsm",),
        operations=("add",),
        nominal_spatial_scope="full",  # differs from the builtin entry
        temporal_scope="replay_if_stateful",
        preview="vegetation_geometry_and_shadow",
        validation_fixtures=(),
        rejected_properties=frozenset({"transmissivity"}),
    )
    with pytest.raises(AdapterRegistryError, match="duplicate|different metadata"):
        registry.register_metadata(metadata)


def test_source_nodes_must_resolve_in_graph() -> None:
    metadata = AdapterMetadata(
        id="ghost_node_adapter",
        group="geometry",
        status=AdapterStatus.ADAPTER_REQUIRED,
        source_nodes=("not_a_node",),
        operations=("add",),
        nominal_spatial_scope="windows",
        temporal_scope="replay",
        preview="p",
        validation_fixtures=(),
    )
    with pytest.raises(AdapterRegistryError, match="unknown dependency node"):
        builtin_registry().register_metadata(metadata)


# ---------------------------------------------------------------------------
# Planner safety semantics
# ---------------------------------------------------------------------------


def test_view_only_edit_produces_zero_scientific_stages() -> None:
    plan = make_planner().plan([view_edit()], initial_state())
    assert plan.node_impacts == ()
    assert plan.estimated_work_units == 0.0
    assert plan.estimated_memory_bytes == 0
    assert plan.transport_window is None
    assert plan.changed_sources == ("output_selection",)
    assert "utci" in plan.reusable_nodes
    assert "svf" in plan.reusable_nodes


def test_stale_scene_revision_refuses_to_plan() -> None:
    state = initial_state().advance(["landcover"])  # revision now 1
    planner = make_planner()
    with pytest.raises(PlanningError, match="stale scene revision"):
        planner.plan([veg_edit(revision=0)], state)
    # Planning against the matching revision succeeds.
    plan = planner.plan([veg_edit(revision=1)], state)
    assert plan.scene_revision == 1


def test_mixed_scenarios_and_empty_batch_rejected() -> None:
    other_scenario = EditCommand(
        edit_id="x",
        scenario_id="scenario-b",
        base_scene_revision=0,
        adapter_id="vegetation_geometry",
        operation="add",
        old_state=None,
        new_state={},
        requested_outputs=("utci",),
        requested_times=None,
    )
    edit = ValidatedEdit(
        other_scenario, "vegetation_geometry", 1, "vegetation_dsm", veg_delta()
    )
    with pytest.raises(PlanningError, match="one scenario"):
        make_planner().plan([veg_edit(), edit], initial_state())
    with pytest.raises(PlanningError, match="non-empty|at least one"):
        make_planner().plan([], initial_state())


def test_unsafe_locality_demotes_node_to_full_with_reason() -> None:
    # A window covering 36% of the tile against a 0.10 threshold: the
    # first windowed stage is demoted, and FULL propagates downstream.
    big_window = RasterWindow(0, 300, 0, 300)
    planner = make_planner(policy=ConservativeSafetyPolicy(full_recompute_fraction=0.10))
    plan = planner.plan(
        [veg_edit(delta=veg_delta(windows=(big_window,)))], initial_state()
    )
    scopes = scopes_by_node(plan)
    for node_id, scope in scopes.items():
        assert scope is SpatialScope.FULL, node_id
    demotions = [
        reason for reason in plan.fallback_reasons if "demoted to full" in reason
    ]
    assert demotions
    assert any("relative_geometry" in reason for reason in demotions)
    # The demoted stages carry no windows (full for their products only).
    for impact in plan.node_impacts:
        assert impact.spatial_scope is SpatialScope.FULL
        assert impact.write_windows == ()


def test_local_windows_clamped_and_empty_windows_go_full() -> None:
    # Window fully outside the grid clamps away -> conservative FULL.
    outside = RasterWindow(600, 700, 600, 700)
    plan = make_planner().plan(
        [veg_edit(delta=veg_delta(windows=(outside,)))], initial_state()
    )
    scopes = scopes_by_node(plan)
    assert scopes["vegetation_visibility"] is SpatialScope.FULL
    assert any(
        "no local windows" in reason for reason in plan.fallback_reasons
    )


def test_stateful_stage_replays_from_zero() -> None:
    plan = make_planner().plan([veg_edit(times=(3,))], initial_state())
    by_node = {impact.node_id: impact for impact in plan.node_impacts}
    thermal = by_node["surface_thermal_state"]
    assert thermal.temporal_scope is TemporalScope.REPLAY
    assert thermal.time_start == 0
    assert thermal.time_stop == 3
    shadow = by_node["time_shadow"]
    assert shadow.temporal_scope is TemporalScope.ONE
    assert shadow.time_start == 3
    geometry = by_node["vegetation_visibility"]
    assert geometry.temporal_scope is TemporalScope.ALL


def test_estimates_follow_documented_formula() -> None:
    window = RasterWindow(96, 160, 96, 160)  # 64x64 = 4096 px
    plan = make_planner().plan(
        [veg_edit(delta=veg_delta(windows=(window,)), times=(3,))],
        initial_state(),
    )
    from solweig_gpu.incremental import NODE_EXECUTION_COSTS

    grid_pixels = GRID.rows * GRID.cols
    expected_memory = 0
    expected_work = 0.0
    for impact in plan.node_impacts:
        cost = NODE_EXECUTION_COSTS[impact.node_id]
        pixels = (
            grid_pixels
            if impact.spatial_scope is SpatialScope.FULL
            else sum(w.area for w in impact.write_windows)
        )
        if impact.temporal_scope is TemporalScope.ONE:
            steps = 1
        elif impact.temporal_scope is TemporalScope.RANGE:
            steps = impact.time_stop - impact.time_start + 1
        elif impact.temporal_scope is TemporalScope.REPLAY:
            steps = impact.time_stop + 1
        else:
            steps = 1
        expected_memory += cost.float32_layers_per_step * pixels * steps * 4
        expected_work += cost.relative_work * pixels * steps / 100_000
    assert plan.estimated_memory_bytes == expected_memory
    assert plan.estimated_work_units == pytest.approx(expected_work)
    assert plan.estimated_memory_bytes > 0
    # Hand-checked spot value: 8 stages over one 4096-px window at t=3
    # (2+8+2+1+4+1*4+1+1) layers * 4096 * 4 bytes = 376832. wbgt is not a
    # stage: outputs follow the request, and wbgt is never planned.
    assert plan.estimated_memory_bytes == 376832


def test_transport_window_unions_local_writes() -> None:
    plan = make_planner().plan(
        [
            veg_edit(delta=veg_delta(windows=(RasterWindow(96, 160, 96, 160),))),
            landcover_edit(patches=(paint_patch(200, 220),)),
        ],
        initial_state(),
    )
    assert plan.transport_window == RasterWindow(96, 204, 96, 225)


# ---------------------------------------------------------------------------
# Record semantics: frozen, hashable, transactional
# ---------------------------------------------------------------------------


def test_records_are_frozen_and_hashable() -> None:
    command = veg_command()
    same = veg_command()
    assert command == same
    assert hash(command) == hash(same)
    with pytest.raises(AttributeError):
        command.edit_id = "mutated"  # type: ignore[misc]

    impact = NodeImpact(
        node_id="svf",
        spatial_scope=SpatialScope.WINDOWS,
        read_windows=(RasterWindow(0, 8, 0, 8),),
        write_windows=(RasterWindow(0, 8, 0, 8),),
        temporal_scope=TemporalScope.ALL,
        time_start=None,
        time_stop=None,
        reason="test",
    )
    assert hash(impact) == hash(
        NodeImpact(
            "svf",
            SpatialScope.WINDOWS,
            (RasterWindow(0, 8, 0, 8),),
            (RasterWindow(0, 8, 0, 8),),
            TemporalScope.ALL,
            None,
            None,
            "test",
        )
    )
    with pytest.raises(Exception):
        NodeImpact(
            node_id="svf",
            spatial_scope=SpatialScope.FULL,
            read_windows=(RasterWindow(0, 8, 0, 8),),
            write_windows=(),
            temporal_scope=TemporalScope.ALL,
            time_start=None,
            time_stop=None,
            reason="full must carry no windows",
        )
    with pytest.raises(Exception):
        NodeImpact(
            node_id="svf",
            spatial_scope=SpatialScope.WINDOWS,
            read_windows=(),
            write_windows=(),
            temporal_scope=TemporalScope.ALL,
            time_start=None,
            time_stop=None,
            reason="windows need windows",
        )
    with pytest.raises(Exception):
        NodeImpact(
            node_id="svf",
            spatial_scope=SpatialScope.WINDOWS,
            read_windows=(),
            write_windows=(RasterWindow(0, 8, 0, 8),),
            temporal_scope=TemporalScope.RANGE,
            time_start=None,
            time_stop=None,
            reason="range needs bounds",
        )


def test_state_payloads_are_frozen_deterministically() -> None:
    command_a = EditCommand(
        "e", "s", 0, "a", "op", {"b": 1, "a": [1, 2]}, None, (), None
    )
    command_b = EditCommand(
        "e", "s", 0, "a", "op", {"a": (1, 2), "b": 1}, None, (), None
    )
    assert command_a == command_b
    assert hash(command_a) == hash(command_b)
    assert repr(command_a.old_state) == repr(command_b.old_state)


def test_scenario_transaction_staging_is_rollback_safe() -> None:
    from solweig_gpu.incremental import ScenarioTransaction, TransactionError

    transaction = ScenarioTransaction(scenario_id="scenario-a")
    transaction.stage(veg_delta())
    transaction.stage(landcover_edit().delta)
    assert transaction.state == "open"
    assert len(transaction.staged_deltas()) == 2
    transaction.rollback()
    assert transaction.state == "rolled_back"
    assert transaction.staged_deltas() == ()
    with pytest.raises(TransactionError):
        transaction.stage(veg_delta())

    fresh = ScenarioTransaction(scenario_id="scenario-a")
    fresh.stage(veg_delta())
    committed = fresh.commit()
    assert len(committed) == 1
    assert fresh.state == "committed"
    with pytest.raises(TransactionError):
        fresh.rollback()


def test_delta_family_must_match_source_node() -> None:
    command = EditCommand(
        edit_id="mixed-up",
        scenario_id="scenario-a",
        base_scene_revision=0,
        adapter_id="landcover_surface",
        operation="paint",
        old_state=None,
        new_state={},
        requested_outputs=("utci",),
        requested_times=None,
    )
    mismatch = ValidatedEdit(
        command,
        "landcover_surface",
        1,
        "landcover",
        VegetationObjectDelta(  # wrong family for the landcover node
            "landcover",
            "landcover_surface",
            (ObjectStateChange("tree-1", None, {"height_m": 10.0}),),
            (RasterWindow(0, 8, 0, 8),),
        ),
    )
    with pytest.raises(PlanningError, match="requires"):
        make_planner().plan([mismatch], initial_state())


def test_package_exports_engine_core() -> None:
    import solweig_gpu.incremental as incremental

    for name in (
        "EditPlanner",
        "EditGraph",
        "SceneGraphState",
        "AdapterRegistry",
        "AdapterMetadata",
        "EditAdapter",
        "EditCommand",
        "ValidatedEdit",
        "ImpactPlan",
        "NodeImpact",
        "SourceDelta",
        "SpatialScope",
        "TemporalScope",
        "ScenarioTransaction",
        "ConservativeSafetyPolicy",
        "SafetyPolicy",
        "builtin_registry",
        "default_edit_graph",
    ):
        assert hasattr(incremental, name), name
        assert name in incremental.__all__


def test_adapter_protocol_is_runtime_checkable() -> None:
    from solweig_gpu.incremental import EditAdapter

    class MinimalAdapter:
        adapter_id = "vegetation_geometry"
        schema_version = 1

        def validate(self, command, context): ...
        def preview_descriptor(self, edit, context): ...
        def source_delta(self, edit, context): ...
        def impact_plan(self, edit, context): ...
        def apply_source_delta(self, delta, transaction): ...
        def validation_fixtures(self): ...

    assert isinstance(MinimalAdapter(), EditAdapter)

    registry = builtin_registry()
    adapter = MinimalAdapter()
    adapter.metadata = registry.get_metadata("vegetation_geometry")
    registry.register(adapter)  # identical metadata: idempotent registration
    assert registry.get_adapter("vegetation_geometry") is adapter


# ---------------------------------------------------------------------------
# U-B review remediation (H1/H2/M1/M3/M4, 2026-09-02): every fence must hold
# on the DELTA payloads and the PLANNER path, not only on validate_edit_state
# ---------------------------------------------------------------------------


def test_h1_water_paints_through_delta_payload_rejected() -> None:
    # Reviewer repro: class 7 riding LandCoverPaintPatch.after_classes plans
    # fine unless the planner schema-checks the delta itself.
    window = RasterWindow(200, 204, 200, 205)
    area = window.area
    delta = LandCoverPaintDelta(
        "landcover",
        "landcover_surface",
        (
            LandCoverPaintPatch(
                window=window,
                before_classes=(2,) * area,
                after_classes=(7,) * area,
            ),
        ),
    )
    edit = landcover_edit(patches=delta.patches, edit_id="lc-water")
    # Swap the delta in (helper builds its own).
    edit = ValidatedEdit(
        edit.command, "landcover_surface", 1, "landcover", delta
    )
    with pytest.raises(PlanningError, match="fenced class 7"):
        make_planner().plan([edit], initial_state())


def test_h1_transmissivity_riding_object_delta_rejected() -> None:
    delta = VegetationObjectDelta(
        "vegetation_dsm",
        "vegetation_geometry",
        (
            ObjectStateChange(
                "tree-1",
                None,
                {"x_m": 200.0, "y_m": 200.0, "height_m": 10.0,
                 "transmissivity": 0.2},
            ),
        ),
        (RasterWindow(96, 160, 96, 160),),
    )
    edit = veg_edit(delta=delta, edit_id="veg-trans")
    with pytest.raises(PlanningError, match="transmissivity"):
        make_planner().plan([edit], initial_state())


def test_h1_blocked_and_unknown_model_params_via_delta_rejected() -> None:
    def param_edit(name: str, value: float) -> ValidatedEdit:
        command = EditCommand(
            edit_id=f"p-{name}",
            scenario_id="scenario-a",
            base_scene_revision=0,
            adapter_id="model_receptor_parameters",
            operation="update",
            old_state={name: 1.0},
            new_state={name: value},
            requested_outputs=("utci",),
            requested_times=(3,),
        )
        delta = ModelParameterDelta(
            "model_parameters",
            "model_receptor_parameters",
            (ModelParameterChange(name, 1.0, value),),
        )
        return ValidatedEdit(
            command, "model_receptor_parameters", 1, "model_parameters", delta
        )

    for name in ("patch_option", "Patch_Option", "utc", "albedo_g",
                 "mystery_knob"):
        if name == "patch_option":
            # Case-variant alias must not slip through either.
            with pytest.raises(PlanningError):
                make_planner().plan(
                    [param_edit("patch_option", 2.0)], initial_state()
                )
            continue
        with pytest.raises(PlanningError, match="blocks|does not define"):
            make_planner().plan([param_edit(name, 2.0)], initial_state())
    # The registry-level API rejects the alias case variant directly.
    with pytest.raises(AdapterSchemaError):
        builtin_registry().validate_edit_state(
            "model_receptor_parameters", {"patch_option": 2}
        )


def test_h1_command_states_schema_checked_by_planner() -> None:
    command = EditCommand(
        edit_id="veg-bad-state",
        scenario_id="scenario-a",
        base_scene_revision=0,
        adapter_id="vegetation_geometry",
        operation="add",
        old_state=None,
        new_state={"height_m": 10.0, "transmissivity": 0.3},
        requested_outputs=("utci",),
        requested_times=(3,),
    )
    edit = ValidatedEdit(
        command, "vegetation_geometry", 1, "vegetation_dsm", veg_delta()
    )
    with pytest.raises(PlanningError, match="transmissivity"):
        make_planner().plan([edit], initial_state())


def test_h2_building_and_terrain_edits_plan_full_only() -> None:
    # full_only_initially adapters must not emit windowed stages under the
    # default policy: walls/aspect have no in-process implementation and a
    # building edit invalidates site-wide SVF (unsafe locality -> full).
    for adapter_id, source, operation, delta in (
        (
            "building_geometry",
            "building_dsm",
            "update",
            BuildingMassingDelta(
                "building_dsm",
                "building_geometry",
                (ObjectStateChange("b1", {"height_m": 10.0},
                                   {"height_m": 20.0}),),
                (RasterWindow(100, 200, 100, 200),),
            ),
        ),
        (
            "terrain_dem",
            "dem",
            "sculpt",
            BuildingMassingDelta(  # family check skips unmapped `dem`
                "dem",
                "terrain_dem",
                (ObjectStateChange("d1", {"dz_m": 0.0}, {"dz_m": 2.0}),),
                (RasterWindow(100, 200, 100, 200),),
            ),
        ),
    ):
        command = EditCommand(
            edit_id="x",
            scenario_id="scenario-a",
            base_scene_revision=0,
            adapter_id=adapter_id,
            operation=operation,
            old_state={"height_m": 10.0},
            new_state={"height_m": 20.0},
            requested_outputs=("utci",),
            requested_times=(3,),
        )
        edit = ValidatedEdit(command, adapter_id, 1, source, delta)
        plan = make_planner().plan([edit], initial_state())
        scopes = scopes_by_node(plan)
        assert scopes, adapter_id
        for node_id, scope in scopes.items():
            assert scope is SpatialScope.FULL, (adapter_id, node_id)
        assert any(
            "full_only_initially" in reason for reason in plan.fallback_reasons
        ), adapter_id


def test_m1_wbgt_request_refused_and_view_selection_blocked() -> None:
    command = EditCommand(
        edit_id="veg-wbgt",
        scenario_id="scenario-a",
        base_scene_revision=0,
        adapter_id="vegetation_geometry",
        operation="add",
        old_state=None,
        new_state={"x_m": 200.0, "y_m": 200.0, "height_m": 10.0},
        requested_outputs=("wbgt",),
        requested_times=(3,),
    )
    edit = ValidatedEdit(
        command, "vegetation_geometry", 1, "vegetation_dsm", veg_delta()
    )
    with pytest.raises(PlanningError, match="never be planned"):
        make_planner().plan([edit], initial_state())

    # The view adapter cannot select wbgt either (not producible_incremental).
    with pytest.raises(AdapterSchemaError, match="not producible"):
        builtin_registry().validate_delta(
            "output_view",
            OutputSelectionDelta(
                "output_selection", "output_view", ("utci",), ("wbgt",)
            ),
        )


def test_m1_unrequested_outputs_pruned_with_reason() -> None:
    # Requesting only tmrt keeps tmrt (and its ancestors) but prunes utci;
    # the pruned output records a reason and is neither computed nor
    # reusable.
    command = EditCommand(
        edit_id="veg-tmrt-only",
        scenario_id="scenario-a",
        base_scene_revision=0,
        adapter_id="vegetation_geometry",
        operation="add",
        old_state=None,
        new_state={"x_m": 200.0, "y_m": 200.0, "height_m": 10.0},
        requested_outputs=("tmrt",),
        requested_times=(3,),
    )
    edit = ValidatedEdit(
        command, "vegetation_geometry", 1, "vegetation_dsm", veg_delta()
    )
    plan = make_planner().plan([edit], initial_state())
    scopes = scopes_by_node(plan)
    assert scopes["tmrt"] is SpatialScope.WINDOWS
    assert "utci" not in scopes
    assert "utci" not in plan.reusable_nodes
    assert any(
        "output utci: not requested" in reason for reason in plan.fallback_reasons
    )


def test_m3_available_times_bound_unbounded_estimates() -> None:
    from solweig_gpu.incremental.edit_types import SiteContext

    grid = GRID
    small_context = SiteContext("site-a", grid, 0, available_times=(0, 1, 2))
    big_context = SiteContext(
        "site-a", grid, 0, available_times=tuple(range(24))
    )
    edit = veg_edit(times=None)  # unbounded REPLAY/ALL temporal scopes
    plan_small = make_planner().plan([edit], initial_state(), small_context)
    plan_big = make_planner().plan([edit], initial_state(), big_context)
    assert plan_big.estimated_memory_bytes > plan_small.estimated_memory_bytes


def test_m4_wind_coefficient_site_fenced_via_context() -> None:
    from solweig_gpu.incremental.edit_types import SiteContext

    context = SiteContext(
        "windy-site", GRID, 0, has_wind_coefficients=True
    )
    with pytest.raises(PlanningError, match="wind-coefficient"):
        make_planner().plan([veg_edit()], initial_state(), context)
    # Grid/revision mismatches are refused too (stale context never plans).
    other_grid_context = SiteContext(
        "site-a",
        RasterGrid(64, 64, 2.0),
        0,
    )
    with pytest.raises(PlanningError, match="does not match"):
        make_planner().plan([veg_edit()], initial_state(), other_grid_context)
    stale_context = SiteContext("site-a", GRID, 7)
    with pytest.raises(PlanningError, match="stale"):
        make_planner().plan([veg_edit()], initial_state(), stale_context)


def test_new1_producible_non_node_outputs_do_not_crash() -> None:
    # Review NEW-1: "shadow"/"kup"/... are producible layer names but not
    # graph node ids — the ancestor walk must seed only node ids, and the
    # request must plan cleanly instead of raising EditGraphError.
    for layer in ("shadow", "kup", "kdown", "lup", "ldown", "ta", "wind"):
        command = EditCommand(
            edit_id=f"veg-{layer}",
            scenario_id="scenario-a",
            base_scene_revision=0,
            adapter_id="vegetation_geometry",
            operation="add",
            old_state=None,
            new_state={"x_m": 200.0, "y_m": 200.0, "height_m": 10.0},
            requested_outputs=(layer,),
            requested_times=(3,),
        )
        edit = ValidatedEdit(
            command, "vegetation_geometry", 1, "vegetation_dsm", veg_delta()
        )
        plan = make_planner().plan([edit], initial_state())
        # The layer rides the executor's transport plumbing (U-C); node
        # outputs still follow the request semantics.
        scopes = scopes_by_node(plan)
        assert "utci" not in scopes  # not requested, pruned with reason
        assert any(
            "output utci: not requested" in reason
            for reason in plan.fallback_reasons
        )
    # Truly unknown names are a clean PlanningError, never a graph crash.
    bad = EditCommand(
        edit_id="veg-ghost-layer",
        scenario_id="scenario-a",
        base_scene_revision=0,
        adapter_id="vegetation_geometry",
        operation="add",
        old_state=None,
        new_state={"x_m": 200.0, "y_m": 200.0, "height_m": 10.0},
        requested_outputs=("ghost_layer",),
        requested_times=(3,),
    )
    bad_edit = ValidatedEdit(
        bad, "vegetation_geometry", 1, "vegetation_dsm", veg_delta()
    )
    with pytest.raises(PlanningError, match="unknown requested outputs"):
        make_planner().plan([bad_edit], initial_state())


def test_new2_grid_mismatch_includes_pixel_size() -> None:
    # Review NEW-2: rows/cols equality alone must not pass a context whose
    # pixel size (or any other grid fact) differs.
    from solweig_gpu.incremental.edit_types import SiteContext

    coarse_context = SiteContext("site-a", RasterGrid(500, 500, 10.0), 0)
    with pytest.raises(PlanningError, match="does not match"):
        make_planner().plan(
            [veg_edit()], initial_state(), coarse_context
        )
