# SPDX-License-Identifier: GPL-3.0-only
"""Vegetation-geometry EditAdapter tests (first concrete adapter, U-B-veg).

Covers the adapter contract against the wrapped tree machinery: registry
schema enforcement through the adapter (transmissivity rejection, undeclared
operations, unknown fields), old-union-new delta windows cross-checked
against ``dirty_window_for_edit`` for every edit shape, delegation of
impact planning to the core planner (dirty/reusable sets, windowed scopes,
thermal replay, estimates, bitwise determinism, adapter/engine plan
equality), rollback-safe transaction staging with the TreeEdit commit seam,
and registry discoverability for the U-D capability listing.
"""

from __future__ import annotations

import pytest

from solweig_gpu.incremental import (
    ADAPTER_ID,
    AdapterRegistryError,
    AdapterSchemaError,
    BuildingMassingDelta,
    EditCommand,
    EditPlanner,
    EditStateError,
    ObjectStateChange,
    RasterGrid,
    RasterWindow,
    ScenarioTransaction,
    SceneGraphState,
    SiteContext,
    SourceDeltaError,
    SpatialScope,
    TemporalScope,
    TransactionError,
    TreeSpec,
    VegetationGeometryAdapter,
    VegetationObjectDelta,
    builtin_registry,
    coalesced_batch_from_deltas,
    default_edit_graph,
    dirty_window_for_edit,
    freeze_mapping,
    register_default_adapters,
    tree_edits_from_deltas,
)
from solweig_gpu.incremental.edits import EditOperation, dirty_windows_for_batch
from solweig_gpu.incremental.geometry import (
    InfluenceConfig,
    SunPosition,
    merge_windows,
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
TREE_B = {
    "tree_id": "tree-b",
    "x_m": 700.0,
    "y_m": 300.0,
    "height_m": 12.0,
    "canopy_radius_m": 4.0,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_adapter(**kwargs) -> VegetationGeometryAdapter:
    kwargs.setdefault("sun_positions", SUNS)
    kwargs.setdefault("influence_config", CONFIG)
    return VegetationGeometryAdapter(**kwargs)


def context(revision: int = 0) -> SiteContext:
    return SiteContext(
        site_id="site-a",
        grid=GRID,
        scene_revision=revision,
        available_times=(0, 1, 2, 3),
    )


def veg_command(
    operation: str = "add",
    *,
    edit_id: str = "veg-1",
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


def spec_a(**overrides) -> TreeSpec:
    props = {
        "x_m": TREE_A["x_m"],
        "y_m": TREE_A["y_m"],
        "height_m": TREE_A["height_m"],
        "canopy_radius_m": TREE_A["canopy_radius_m"],
        "trunk_ratio": 0.25,
    }
    props.update(overrides)
    return TreeSpec("tree-a", **props)


def spec_b(**overrides) -> TreeSpec:
    props = {
        "x_m": TREE_B["x_m"],
        "y_m": TREE_B["y_m"],
        "height_m": TREE_B["height_m"],
        "canopy_radius_m": TREE_B["canopy_radius_m"],
        "trunk_ratio": 0.25,
    }
    props.update(overrides)
    return TreeSpec("tree-b", **props)


def expected_window(
    old_tree: TreeSpec | None, new_tree: TreeSpec | None
) -> RasterWindow:
    return dirty_window_for_edit(
        GRID,
        old_tree=old_tree,
        new_tree=new_tree,
        sun_positions=SUNS,
        config=CONFIG,
    )


def add_edit(tree_state: dict, edit_id: str = "veg-1"):
    adapter = make_adapter()
    return adapter.validate(
        veg_command("add", edit_id=edit_id, new_state=dict(tree_state)), context()
    )


def initial_state(revision: int = 0) -> SceneGraphState:
    return SceneGraphState(default_edit_graph(), revision, {})


def scopes_by_node(plan):
    return {impact.node_id: impact.spatial_scope for impact in plan.node_impacts}


# ---------------------------------------------------------------------------
# Registry discovery and enforcement through the adapter
# ---------------------------------------------------------------------------


def test_adapter_registers_idempotently_and_is_discoverable() -> None:
    registry = builtin_registry()
    adapter = make_adapter(registry=registry)
    registry.register(adapter)
    registry.register(adapter)  # identical metadata: idempotent

    assert registry.get_adapter(ADAPTER_ID) is adapter
    metadata = registry.get_metadata(ADAPTER_ID)
    assert metadata is adapter.metadata
    assert metadata.operations == ("add", "move", "update", "delete")
    assert "raster_patch" not in metadata.operations
    assert metadata.source_nodes == ("vegetation_dsm",)
    assert metadata.preview == "vegetation_geometry_and_shadow"

    # U-D capability listing shape: one entry per adapter, sorted by id.
    listing = registry.all_metadata()
    veg_entries = [entry for entry in listing if entry.id == ADAPTER_ID]
    assert veg_entries == [metadata]

    wired = register_default_adapters()
    assert isinstance(wired.get_adapter(ADAPTER_ID), VegetationGeometryAdapter)


def test_validate_maps_every_legal_operation() -> None:
    adapter = make_adapter()
    ctx = context()

    cases = {
        "add": (
            veg_command("add", new_state=dict(TREE_A)),
            None,
            spec_a(),
        ),
        "move": (
            veg_command(
                "move",
                old_state=dict(TREE_A),
                new_state={"tree_id": "tree-a", "x_m": 460.0, "y_m": 560.0},
            ),
            spec_a(),
            spec_a(x_m=460.0, y_m=560.0),
        ),
        "update": (
            veg_command(
                "update",
                old_state=dict(TREE_A),
                new_state={"tree_id": "tree-a", "height_m": 14.0},
            ),
            spec_a(),
            spec_a(height_m=14.0),
        ),
        "delete": (
            veg_command("delete", old_state=dict(TREE_A)),
            spec_a(),
            None,
        ),
    }
    operations = {}
    for operation, (command, old_spec, new_spec) in cases.items():
        edit = adapter.validate(command, ctx)
        assert edit.adapter_id == ADAPTER_ID
        assert edit.source_node_id == "vegetation_dsm"
        assert edit.schema_version == adapter.schema_version == 1
        delta = edit.delta
        assert delta.source_node_id == "vegetation_dsm"
        (change,) = delta.objects
        assert change.object_id == "tree-a"
        if old_spec is None:
            assert change.before is None
        else:
            assert dict(change.before) == {
                "x_m": old_spec.x_m,
                "y_m": old_spec.y_m,
                "height_m": old_spec.height_m,
                "canopy_radius_m": old_spec.canopy_radius_m,
                "trunk_ratio": old_spec.trunk_ratio,
            }
        if new_spec is None:
            assert change.after is None
        else:
            assert dict(change.after) == {
                "x_m": new_spec.x_m,
                "y_m": new_spec.y_m,
                "height_m": new_spec.height_m,
                "canopy_radius_m": new_spec.canopy_radius_m,
                "trunk_ratio": new_spec.trunk_ratio,
            }
        # The TreeEdit view classifies into the machinery's operation enum.
        (tree_edit,) = adapter.tree_edits(command, ctx)
        operations[operation] = tree_edit.operation
    assert operations["add"] is EditOperation.ADD
    assert operations["move"] is EditOperation.MOVE
    assert operations["update"] is EditOperation.UPDATE
    assert operations["delete"] is EditOperation.DELETE


def test_validate_rejects_transmissivity_on_both_states() -> None:
    adapter = make_adapter()
    with pytest.raises(AdapterSchemaError, match="transmissivity"):
        adapter.validate(
            veg_command("add", new_state={**TREE_A, "transmissivity": 0.4}),
            context(),
        )
    with pytest.raises(AdapterSchemaError, match="transmissivity"):
        adapter.validate(
            veg_command("delete", old_state={**TREE_A, "transmissivity": 0.4}),
            context(),
        )


def test_validate_rejects_undeclared_operation() -> None:
    adapter = make_adapter()
    with pytest.raises(AdapterRegistryError, match="does not support"):
        adapter.validate(
            veg_command("raster_patch", new_state=dict(TREE_A)), context()
        )


def test_validate_rejects_unknown_fields() -> None:
    adapter = make_adapter()
    with pytest.raises(EditStateError, match="unknown vegetation fields"):
        adapter.validate(
            veg_command("add", new_state={**TREE_A, "species": "oak"}), context()
        )


def test_validate_enforces_operation_state_shapes() -> None:
    adapter = make_adapter()
    ctx = context()
    with pytest.raises(EditStateError, match="add.*must not carry an old_state"):
        adapter.validate(
            veg_command("add", old_state=dict(TREE_A), new_state=dict(TREE_A)), ctx
        )
    with pytest.raises(EditStateError, match="delete.*must not carry a new_state"):
        adapter.validate(
            veg_command("delete", old_state=dict(TREE_A), new_state=dict(TREE_A)), ctx
        )
    with pytest.raises(EditStateError, match="requires both"):
        adapter.validate(
            veg_command("update", new_state={"tree_id": "tree-a", "height_m": 8.0}),
            ctx,
        )
    with pytest.raises(EditStateError, match="height_m"):
        adapter.validate(
            veg_command(
                "add",
                new_state={k: v for k, v in TREE_A.items() if k != "height_m"},
            ),
            ctx,
        )
    with pytest.raises(EditStateError, match="different trees"):
        adapter.validate(
            veg_command(
                "move",
                old_state=dict(TREE_A),
                new_state={"tree_id": "tree-b", "x_m": 460.0, "y_m": 560.0},
            ),
            ctx,
        )


def test_validate_rejects_noop_and_stale_and_bad_values() -> None:
    adapter = make_adapter()
    with pytest.raises(EditStateError, match="no-op"):
        adapter.validate(
            veg_command(
                "update",
                old_state=dict(TREE_A),
                new_state={"tree_id": "tree-a", "height_m": TREE_A["height_m"]},
            ),
            context(),
        )
    with pytest.raises(EditStateError, match="stale scene revision"):
        adapter.validate(
            veg_command("add", revision=0, new_state=dict(TREE_A)), context(revision=1)
        )
    with pytest.raises(EditStateError, match="must be a number"):
        adapter.validate(
            veg_command("add", new_state={**TREE_A, "height_m": "tall"}), context()
        )
    with pytest.raises(ValueError, match="height_m must be between"):
        # Server preset range (3-40 m) enforced via the wrapped machinery.
        adapter.validate(
            veg_command("add", new_state={**TREE_A, "height_m": 60.0}), context()
        )


# ---------------------------------------------------------------------------
# Delta windows: old union new influence, cross-checked against the machinery
# ---------------------------------------------------------------------------


def test_delta_windows_equal_dirty_window_for_edit() -> None:
    adapter = make_adapter()
    ctx = context()

    add = adapter.validate(veg_command("add", new_state=dict(TREE_A)), ctx)
    assert add.delta.windows == (expected_window(None, spec_a()),)

    update = adapter.validate(
        veg_command(
            "update",
            old_state=dict(TREE_A),
            new_state={"tree_id": "tree-a", "height_m": 14.0},
        ),
        ctx,
    )
    assert update.delta.windows == (expected_window(spec_a(), spec_a(height_m=14.0)),)

    delete = adapter.validate(veg_command("delete", old_state=dict(TREE_A)), ctx)
    assert delete.delta.windows == (expected_window(spec_a(), None),)


def test_move_window_covers_old_and_new_influence() -> None:
    adapter = make_adapter()
    old_spec = spec_a()
    new_spec = spec_a(x_m=460.0, y_m=560.0)
    edit = adapter.validate(
        veg_command(
            "move",
            old_state=dict(TREE_A),
            new_state={"tree_id": "tree-a", "x_m": 460.0, "y_m": 560.0},
        ),
        context(),
    )
    (window,) = edit.delta.windows
    assert window == expected_window(old_spec, new_spec)
    old_only = expected_window(old_spec, None)
    new_only = expected_window(None, new_spec)
    # The union window covers both the abandoned and the new footprint...
    assert window.union(old_only) == window
    assert window.union(new_only) == window
    # ...and is strictly larger than either single-sided window.
    assert window.area > old_only.area
    assert window.area > new_only.area


def test_batch_write_windows_match_dirty_windows_for_batch() -> None:
    adapter = make_adapter()
    edits = [
        add_edit(TREE_A, edit_id="veg-a"),
        add_edit(TREE_B, edit_id="veg-b"),
    ]
    planner = EditPlanner(grid=GRID, registry=builtin_registry())
    plan = planner.plan(edits, initial_state())

    # Cross-check the plan's windowed write scope against the wrapped
    # batch helper (planner merges with gap 0).
    batch = coalesced_batch_from_deltas(
        [edit.delta for edit in edits], scenario_id="scenario-a"
    )
    assert batch is not None
    windows = dirty_windows_for_batch(
        batch,
        grid=GRID,
        sun_positions=SUNS,
        config=CONFIG,
        merge_gap_pixels=0,
    )
    assert len(windows) == 2  # disjoint trees stay separate windows
    by_node = {impact.node_id: impact for impact in plan.node_impacts}
    for node_id in ("relative_geometry", "vegetation_visibility", "svf"):
        assert by_node[node_id].write_windows == tuple(windows), node_id
    assert plan.transport_window == windows[0].union(windows[1])


# ---------------------------------------------------------------------------
# Impact plan: dirty/reusable sets, scopes, temporal semantics, estimates
# ---------------------------------------------------------------------------


def test_impact_plan_dirty_and_reusable_sets() -> None:
    plan = make_adapter().impact_plan(add_edit(TREE_A), context())

    dirty = {impact.node_id for impact in plan.node_impacts} | set(
        plan.changed_sources
    )
    # The vegetation chain per the reconciled graph. wbgt sits behind
    # time_shadow in the closure but is NEVER PLANNED (M1 ruling: wbgt is
    # disabled in incremental, refused when requested, pruned with a
    # recorded reason otherwise) — it appears in neither impacts nor
    # reusable_nodes.
    assert dirty == {
        "vegetation_dsm",
        "relative_geometry",
        "vegetation_visibility",
        "svf",
        "time_shadow",
        "radiation",
        "surface_thermal_state",
        "tmrt",
        "utci",
    }
    assert "wbgt" not in plan.reusable_nodes
    assert any(
        "output wbgt" in reason for reason in plan.fallback_reasons
    )
    # The building chain, terrain, forcing-side sources, and the view node
    # stay reusable.
    assert set(plan.reusable_nodes) == {
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
    }
    assert not (dirty & set(plan.reusable_nodes))


def test_impact_plan_windowed_scopes_and_temporal_semantics() -> None:
    edit = add_edit(TREE_A)
    plan = make_adapter().impact_plan(edit, context())
    scopes = scopes_by_node(plan)
    (window,) = edit.delta.windows

    for node_id in scopes:
        assert scopes[node_id] is SpatialScope.WINDOWS, node_id
    by_node = {impact.node_id: impact for impact in plan.node_impacts}
    for node_id in (
        "relative_geometry",
        "vegetation_visibility",
        "svf",
        "time_shadow",
        "radiation",
        "surface_thermal_state",
        "tmrt",
        "utci",
    ):
        impact = by_node[node_id]
        assert impact.write_windows == (window,), node_id
        assert impact.read_windows == (window,), node_id  # halo defaults to 0

    # Time-invariant geometry stages carry ALL (no temporal restriction).
    assert by_node["vegetation_visibility"].temporal_scope is TemporalScope.ALL
    # The stateful accumulator replays from timestep 0 (planner policy).
    thermal = by_node["surface_thermal_state"]
    assert thermal.temporal_scope is TemporalScope.REPLAY
    assert thermal.time_start == 0
    assert thermal.time_stop == 3  # max requested time
    # Requested-time stages narrow to ONE at the requested index.
    assert by_node["time_shadow"].temporal_scope is TemporalScope.ONE
    assert by_node["time_shadow"].time_start == 3
    assert by_node["tmrt"].temporal_scope is TemporalScope.ONE

    assert plan.estimated_memory_bytes > 0
    assert plan.estimated_work_units > 0.0
    assert plan.transport_window == window
    # Topological order: every producer precedes its consumers.
    graph = default_edit_graph()
    position = {i.node_id: idx for idx, i in enumerate(plan.node_impacts)}
    for producer, consumer in graph.edges:
        if producer in position and consumer in position:
            assert position[producer] < position[consumer]


def test_read_halo_expands_reads_not_writes() -> None:
    edit = add_edit(TREE_A)
    plain = make_adapter().impact_plan(edit, context())
    halo = make_adapter(read_halo_pixels=4).impact_plan(edit, context())

    plain_by_node = {impact.node_id: impact for impact in plain.node_impacts}
    for impact in halo.node_impacts:
        assert impact.spatial_scope is SpatialScope.WINDOWS
        assert impact.write_windows == plain_by_node[impact.node_id].write_windows
        assert impact.read_windows == tuple(
            window.expand(4).clamp(rows=GRID.rows, cols=GRID.cols)
            for window in impact.write_windows
        )
    # Estimates follow the core write-scope formula, unchanged by the halo.
    assert halo.estimated_memory_bytes == plain.estimated_memory_bytes


# ---------------------------------------------------------------------------
# Adapter plan vs full EditPlanner: one plan, bitwise determinism
# ---------------------------------------------------------------------------


def test_adapter_plan_equals_core_planner_plan_bitwise() -> None:
    edit = add_edit(TREE_A)
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


def test_full_planner_single_plan_for_vegetation_batch() -> None:
    adapter = make_adapter()
    ctx = context()
    edits = [
        adapter.validate(
            veg_command("add", edit_id="veg-a", new_state=dict(TREE_A)), ctx
        ),
        adapter.validate(
            veg_command(
                "update",
                edit_id="veg-b",
                old_state=dict(TREE_A),
                new_state={"tree_id": "tree-a", "height_m": 14.0},
            ),
            ctx,
        ),
    ]
    planner = EditPlanner(grid=GRID, registry=builtin_registry())
    plan = planner.plan(edits, initial_state())
    assert plan.changed_sources == ("vegetation_dsm",)
    # 8 stages, not 9: wbgt is never planned (M1) — pruned with a reason.
    assert len({impact.node_id for impact in plan.node_impacts}) == 8

    by_node = {impact.node_id: impact for impact in plan.node_impacts}
    # Overlapping old/new windows merge into one write window per the
    # planner's merge rule (gap 0).
    expected = tuple(
        merge_windows(
            [
                expected_window(None, spec_a()),
                expected_window(spec_a(), spec_a(height_m=14.0)),
            ]
        )
    )
    assert expected
    assert by_node["vegetation_visibility"].write_windows == expected

    assert planner.plan(edits, initial_state()) == plan

    # A non-contiguous chain (update built on a state that never existed)
    # is rejected by the engine's coalescing, not silently merged.
    broken = adapter.validate(
        veg_command(
            "update",
            edit_id="veg-x",
            old_state={**TREE_A, "height_m": 12.0},
            new_state={"tree_id": "tree-a", "height_m": 14.0},
        ),
        ctx,
    )
    with pytest.raises(SourceDeltaError, match="non-contiguous"):
        planner.plan([edits[0], broken], initial_state())


# ---------------------------------------------------------------------------
# Transaction staging and the ExactWorker commit seam
# ---------------------------------------------------------------------------


def test_apply_stages_rollback_safe_deltas() -> None:
    adapter = make_adapter()
    edit_a = add_edit(TREE_A, edit_id="veg-a")
    edit_b = add_edit(TREE_B, edit_id="veg-b")

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


def test_failed_second_edit_leaves_first_committed_only() -> None:
    adapter = make_adapter()
    ctx = context()
    first = adapter.validate(
        veg_command("add", edit_id="veg-a", new_state=dict(TREE_A)), ctx
    )
    second = adapter.validate(
        veg_command("add", edit_id="veg-b", new_state=dict(TREE_B)), ctx
    )

    committed_transaction = ScenarioTransaction(scenario_id="scenario-a")
    adapter.apply_source_delta(first.delta, committed_transaction)
    committed = committed_transaction.commit()
    assert committed == (first.delta,)
    with pytest.raises(TransactionError):
        committed_transaction.rollback()

    failed_transaction = ScenarioTransaction(scenario_id="scenario-a")
    adapter.apply_source_delta(second.delta, failed_transaction)
    failed_transaction.rollback()  # the second edit fails: discard it
    assert failed_transaction.staged_deltas() == ()

    batch = coalesced_batch_from_deltas(committed, scenario_id="scenario-a")
    assert batch is not None
    assert len(batch.edits) == 1
    assert batch.edits[0].tree_id == "tree-a"


def test_apply_rejects_foreign_deltas() -> None:
    adapter = make_adapter()
    transaction = ScenarioTransaction(scenario_id="scenario-a")
    with pytest.raises(SourceDeltaError, match="BuildingMassingDelta"):
        adapter.apply_source_delta(
            BuildingMassingDelta(
                "building_dsm",
                "building_geometry",
                (ObjectStateChange("b1", None, {"height_m": 10.0}),),
                (RasterWindow(0, 8, 0, 8),),
            ),
            transaction,
        )
    with pytest.raises(SourceDeltaError, match="belongs to adapter"):
        adapter.apply_source_delta(
            VegetationObjectDelta(
                "vegetation_dsm",
                "some_other_adapter",
                (ObjectStateChange("t", None, {"height_m": 10.0}),),
                (RasterWindow(0, 8, 0, 8),),
            ),
            transaction,
        )
    with pytest.raises(SourceDeltaError, match="source node"):
        adapter.apply_source_delta(
            VegetationObjectDelta(
                "building_dsm",
                ADAPTER_ID,
                (ObjectStateChange("t", None, {"height_m": 10.0}),),
                (RasterWindow(0, 8, 0, 8),),
            ),
            transaction,
        )
    assert transaction.staged_deltas() == ()


def test_committed_deltas_round_trip_to_worker_tree_edits() -> None:
    adapter = make_adapter()
    ctx = context()
    add = adapter.validate(
        veg_command("add", edit_id="veg-1", new_state=dict(TREE_A)), ctx
    )
    grow = adapter.validate(
        veg_command(
            "update",
            edit_id="veg-2",
            old_state=dict(TREE_A),
            new_state={"tree_id": "tree-a", "height_m": 14.0},
        ),
        ctx,
    )
    transaction = ScenarioTransaction(scenario_id="scenario-a")
    adapter.apply_source_delta(add.delta, transaction)
    adapter.apply_source_delta(grow.delta, transaction)
    committed = transaction.commit()

    edits = tree_edits_from_deltas(committed, scenario_id="scenario-a")
    assert [edit.sequence for edit in edits] == [1, 2]
    assert [edit.tree_id for edit in edits] == ["tree-a", "tree-a"]

    # The worker's coalesced view: add -> update collapses to one ADD at
    # the final state (first old state retained).
    batch = coalesced_batch_from_deltas(committed, scenario_id="scenario-a")
    assert batch is not None
    (edit,) = batch.edits
    assert edit.operation is EditOperation.ADD
    assert edit.old_tree is None
    assert edit.new_tree == spec_a(height_m=14.0)
    assert batch.scenario_id == "scenario-a"

    # A fully cancelled chain hands the executor a no-op (None), never a
    # synthetic edit.
    drop = adapter.validate(
        veg_command("delete", edit_id="veg-3", old_state=dict(TREE_A)), ctx
    )
    assert (
        coalesced_batch_from_deltas(
            [add.delta, drop.delta], scenario_id="scenario-a"
        )
        is None
    )


# ---------------------------------------------------------------------------
# Preview and fixtures
# ---------------------------------------------------------------------------


def test_preview_descriptor_is_marked_not_exact() -> None:
    adapter = make_adapter()
    descriptor = adapter.preview_descriptor(add_edit(TREE_A), context())
    assert descriptor.kind == adapter.metadata.preview
    assert descriptor.kind == "vegetation_geometry_and_shadow"
    assert descriptor.immediate is True
    assert any("not scientific output" in note for note in descriptor.limitations)
    assert any("transVeg" in note for note in descriptor.limitations)
    assert any("wind" in note for note in descriptor.limitations)


def test_validation_fixtures_match_registry_entry() -> None:
    adapter = make_adapter()
    assert adapter.validation_fixtures() == (
        "isolated",
        "overlap",
        "low_sun",
        "boundary",
        "delete",
    )
    assert adapter.validation_fixtures() == adapter.metadata.validation_fixtures


def test_adapter_exports_from_package() -> None:
    import solweig_gpu.incremental as incremental
    from solweig_gpu.incremental.adapters import VegetationGeometryAdapter as direct

    assert direct is incremental.VegetationGeometryAdapter
    for name in (
        "VegetationGeometryAdapter",
        "ADAPTER_ID",
        "ADAPTER_SCHEMA_VERSION",
        "EDITABLE_PROPERTIES",
        "IDENTITY_PROPERTY",
        "coalesced_batch_from_deltas",
        "register_default_adapters",
        "tree_edits_from_deltas",
    ):
        assert hasattr(incremental, name), name
        assert name in incremental.__all__
    assert incremental.ADAPTER_ID == "vegetation_geometry"
    assert incremental.EDITABLE_PROPERTIES == (
        "x_m",
        "y_m",
        "height_m",
        "canopy_radius_m",
        "trunk_ratio",
    )
    assert "transmissivity" not in incremental.EDITABLE_PROPERTIES
    # Frozen delta payloads keep key order canonical (deterministic reprs).
    edit = add_edit(TREE_A)
    assert edit.delta.objects[0].after == freeze_mapping(
        {
            "canopy_radius_m": 3.0,
            "height_m": 10.0,
            "trunk_ratio": 0.25,
            "x_m": 400.0,
            "y_m": 600.0,
        }
    )
