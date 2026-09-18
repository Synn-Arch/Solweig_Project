# SPDX-License-Identifier: GPL-3.0-only
"""Land-cover EditAdapter tests (U-B-lc).

Covers the adapter contract for region-local land-cover painting: the
water fence on EVERY path (command states in both roles via
``adapter.validate`` -> ``validate_edit_state``, delta payloads via the
planner's ``validate_delta``, the staging door inside
``apply_source_delta``, and the executor-side
:class:`LandCoverOverlay`), the window grammar (integer, non-negative,
in-grid, non-empty — user-supplied windows are refused off-grid, not
clamped), the class-payload grammar (broadcast vs row-major sequence),
plan round-trips through the core planner (the graph's two ``landcover``
edges dirty ``radiation -> surface_thermal_state -> tmrt -> utci`` while
every geometry/view-factor cache stays reusable; WINDOWS over the paint
footprint; ``surface_thermal_state`` replaying from timestep 0),
bitwise adapter/planner plan equality with determinism repeats, the
safety-policy dirty-fraction demotion, the U-C mixed-scope ruling
(vegetation WINDOWS coexisting with paint WINDOWS in ONE plan), the
copy-on-write overlay (baseline arrays — including cache-backed memmaps —
bitwise immutable, last-write-wins per cell), the refused
``polygon_assign``/``reset`` operations (documented deviations mirroring
the met adapter's refusals), the ``wbgt`` never-planned invariant, a
mutation-sensitivity probe proving the staging-door fence is
load-bearing, and registry discoverability.
"""

from __future__ import annotations

import numpy as np
import pytest

from solweig_gpu.incremental import (
    LANDCOVER_FENCED_CLASSES,
    LANDCOVER_VALID_CLASSES,
    ADAPTER_ID as VEG_ADAPTER_ID,
)
from solweig_gpu.incremental import (
    AdapterRegistryError,
    AdapterSchemaError,
    EditCommand,
    EditPlanner,
    EditStateError,
    LandCoverPaintDelta,
    LandCoverPaintPatch,
    LandCoverOverlay,
    LandCoverSurfaceAdapter,
    PlanningError,
    RasterGrid,
    RasterWindow,
    ScenarioTransaction,
    SceneGraphState,
    SiteContext,
    SourceDeltaError,
    SpatialScope,
    TemporalScope,
    VegetationGeometryAdapter,
    builtin_registry,
    coalesce_source_deltas,
    default_edit_graph,
    landcover_overlay_from_deltas,
    register_default_adapters,
    register_landcover_surface_adapter,
)
from solweig_gpu.incremental.adapters.landcover import (
    ADAPTER_ID,
    ADAPTER_SCHEMA_VERSION,
    FENCED_CLASSES,
    PAINTABLE_CLASSES,
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

#: A small, legal paint stroke far from TREE_A's influence window.
WINDOW = {"row_start": 200, "row_stop": 210, "col_start": 210, "col_stop": 220}
PAINT_WINDOW = RasterWindow(200, 210, 210, 220)

#: Every class the registry ruling keeps OUT of the paint vocabulary,
#: with its reason (0/3/4 are remapped at load, utci_process.py:1029-1041;
#: 7 is the fenced water class, solweig.py:130/:2214-2215 dead branch).
UNPAINTABLE_CLASSES = {
    0: "remapped to bare soil at load",
    3: "remapped to grass at load (and the water-test value)",
    4: "remapped to grass at load",
    7: "water — fenced, dead Twater branch upstream",
    8: "not a class",
    -2: "not a class",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_adapter(**kwargs) -> LandCoverSurfaceAdapter:
    return LandCoverSurfaceAdapter(**kwargs)


def context(revision: int = 0) -> SiteContext:
    return SiteContext(
        site_id="site-a",
        grid=GRID,
        scene_revision=revision,
        available_times=(0, 1, 2, 3),
    )


def paint_command(
    operation: str = "paint",
    *,
    edit_id: str = "lc-1",
    revision: int = 0,
    old_state: dict | None = None,
    new_state: dict | None = None,
    times: tuple[int, ...] | None = (3,),
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


def paint_edit(
    new_state: dict | None = None,
    *,
    old_state: dict | None = None,
    edit_id: str = "lc-1",
    times: tuple[int, ...] | None = (3,),
):
    if new_state is None:
        new_state = {"window": dict(WINDOW), "classes": 5}
    return make_adapter().validate(
        paint_command(
            old_state=old_state, new_state=new_state, edit_id=edit_id,
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


def paint_patch(
    row: int = 200, col: int = 210, rows: int = 4, cols: int = 4,
    after: int | tuple[int, ...] = 5, window: RasterWindow | None = None,
) -> LandCoverPaintPatch:
    area = rows * cols
    if window is None:
        window = RasterWindow(row, row + rows, col, col + cols)
    after_codes = (
        (after,) * window.area if isinstance(after, int) else tuple(after)
    )
    return LandCoverPaintPatch(
        window=window,
        before_classes=(-1,) * window.area,
        after_classes=after_codes,
    )


def synthetic_edit(patches, *, new_state: dict | None = None):
    """A ValidatedEdit built directly from patch records (bypasses the
    adapter's ``validate``) so the planner's own payload fence is tested
    without the adapter rejecting the edit first."""
    from solweig_gpu.incremental import ValidatedEdit

    return ValidatedEdit(
        command=paint_command(new_state=new_state),
        adapter_id=ADAPTER_ID,
        schema_version=ADAPTER_SCHEMA_VERSION,
        source_node_id="landcover",
        delta=LandCoverPaintDelta(
            source_node_id="landcover",
            adapter_id=ADAPTER_ID,
            patches=tuple(patches),
        ),
    )


def veg_adapter() -> VegetationGeometryAdapter:
    return VegetationGeometryAdapter(sun_positions=SUNS, influence_config=CONFIG)


# ---------------------------------------------------------------------------
# Registry discovery and the class vocabulary
# ---------------------------------------------------------------------------


def test_adapter_registers_idempotently_and_is_discoverable() -> None:
    registry = builtin_registry()
    register_landcover_surface_adapter(registry)
    register_landcover_surface_adapter(registry)
    adapter = registry.get_adapter(ADAPTER_ID)
    assert isinstance(adapter, LandCoverSurfaceAdapter)
    # Metadata stays the builtin entry verbatim (registration binds an
    # implementation to declared metadata; it never adds capability).
    assert adapter.metadata.source_nodes == ("landcover",)
    assert adapter.metadata.operations == ("paint", "polygon_assign", "reset")
    assert adapter.metadata.nominal_spatial_scope == "windows"
    assert adapter.metadata.temporal_scope == "replay_if_stateful"
    assert adapter.metadata.preview == "surface_material"
    assert adapter.metadata.valid_classes == frozenset({1, 2, 5, 6})
    # The composed default registration binds it alongside the others
    # (nine metadata entries; five implementations now).
    composed = register_default_adapters(builtin_registry())
    assert isinstance(composed.get_adapter(ADAPTER_ID), LandCoverSurfaceAdapter)
    assert len(composed.adapter_ids()) == 9


def test_class_vocabulary_matches_the_registry_ruling() -> None:
    assert PAINTABLE_CLASSES == LANDCOVER_VALID_CLASSES == {1, 2, 5, 6}
    assert FENCED_CLASSES == LANDCOVER_FENCED_CLASSES == {7}
    assert not (PAINTABLE_CLASSES & FENCED_CLASSES)
    # Every vocabulary class is a real (non-bool) integer — no float or
    # NaN can ride this family's payloads at all.
    for code in PAINTABLE_CLASSES | FENCED_CLASSES:
        assert isinstance(code, int) and not isinstance(code, bool)


# ---------------------------------------------------------------------------
# validate: window grammar, class grammar, fence
# ---------------------------------------------------------------------------


def test_paint_broadcast_builds_one_patch() -> None:
    edit = paint_edit({"window": dict(WINDOW), "classes": 2})
    assert edit.adapter_id == ADAPTER_ID
    assert edit.source_node_id == "landcover"
    (patch,) = edit.delta.patches
    assert patch.window == PAINT_WINDOW
    # Broadcast: every cell carries the class; the undeclared before is
    # the data model's unspecified marker (-1), never a guessed class.
    assert patch.after_classes == (2,) * PAINT_WINDOW.area
    assert patch.before_classes == (-1,) * PAINT_WINDOW.area
    assert edit.delta.spatial_windows == (PAINT_WINDOW,)


def test_paint_mixed_sequence_matches_window_area() -> None:
    # mixed_mask fixture: one stroke, row-major per-cell classes.
    codes = tuple(
        1 if (row + col) % 2 else 5
        for row in range(4)
        for col in range(5)
    )
    window = {"row_start": 10, "row_stop": 14, "col_start": 0, "col_stop": 5}
    edit = paint_edit({"window": window, "classes": list(codes)})
    (patch,) = edit.delta.patches
    assert patch.after_classes == codes
    assert patch.window == RasterWindow(10, 14, 0, 5)
    # A single-cell window accepts a 1-element sequence.
    single = paint_edit(
        {"window": {"row_start": 0, "row_stop": 1, "col_start": 0,
                    "col_stop": 1}, "classes": (6,)}
    )
    (patch,) = single.delta.patches
    assert patch.after_classes == (6,)


def test_declared_before_is_recorded_and_identity_paints_refused() -> None:
    before = [1] * PAINT_WINDOW.area
    edit = paint_edit(
        {"window": dict(WINDOW), "classes": 5},
        old_state={"window": dict(WINDOW), "classes": before},
    )
    (patch,) = edit.delta.patches
    assert patch.before_classes == tuple(before)
    # A declared identity paint is refused, never a silent empty plan.
    with pytest.raises(EditStateError, match="no-op"):
        paint_edit(
            {"window": dict(WINDOW), "classes": 5},
            old_state={"window": dict(WINDOW), "classes": [5] * 100},
        )
    # old_state and new_state must select the SAME window.
    with pytest.raises(EditStateError, match="different windows"):
        paint_edit(
            {"window": dict(WINDOW), "classes": 5},
            old_state={
                "window": {"row_start": 0, "row_stop": 2, "col_start": 0,
                           "col_stop": 2},
                "classes": 1,
            },
        )


def test_window_grammar_is_enforced() -> None:
    adapter = make_adapter()
    bad_windows = [
        {},  # missing entirely
        {"row_start": 0, "row_stop": 2},  # partial
        {"row_start": 0, "row_stop": 2, "col_start": 0, "col_stop": 2, "z": 1},
        {"row_start": 0.0, "row_stop": 2, "col_start": 0, "col_stop": 2},
        {"row_start": True, "row_stop": 2, "col_start": 0, "col_stop": 2},
        {"row_start": -1, "row_stop": 2, "col_start": 0, "col_stop": 2},
        {"row_start": 2, "row_stop": 1, "col_start": 0, "col_stop": 2},
        {"row_start": 5, "row_stop": 5, "col_start": 0, "col_stop": 2},  # empty
        {"row_start": 0, "row_stop": 501, "col_start": 0, "col_stop": 2},
        {"row_start": 0, "row_stop": 2, "col_start": 0, "col_stop": 5000},
    ]
    for bad in bad_windows:
        with pytest.raises(EditStateError):
            adapter.validate(
                paint_command(new_state={"window": bad, "classes": 5}),
                context(),
            )
    # Unknown state fields — including case variants of the two legal
    # names — are rejected with the closed vocabulary.
    for state in (
        {"window": dict(WINDOW), "classes": 5, "mask": [1, 0]},
        {"Window": dict(WINDOW), "classes": 5},
        {"window": dict(WINDOW), "Classes": 5},
    ):
        with pytest.raises(EditStateError, match="unknown fields"):
            adapter.validate(paint_command(new_state=state), context())
    # A non-mapping window or a missing window field is refused.
    with pytest.raises(EditStateError, match="window"):
        adapter.validate(
            paint_command(new_state={"classes": 5}), context()
        )


def test_class_payload_grammar_is_enforced() -> None:
    adapter = make_adapter()
    cases = [
        None,
        "5",
        5.0,
        True,
        [5] * 99,  # wrong length for the 100-cell window
        [5, "1"] + [5] * 98,
        [5, 1.0] + [5] * 98,
        [5, True] + [5] * 98,
    ]
    for bad in cases:
        # Non-integer spellings may be rejected by the registry's own
        # class check before the adapter's grammar check — either error
        # proves the payload never becomes a patch.
        with pytest.raises((EditStateError, AdapterSchemaError)):
            adapter.validate(
                paint_command(
                    new_state={"window": dict(WINDOW), "classes": bad}
                ),
                context(),
            )
    # new_state is mandatory for a paint.
    with pytest.raises(EditStateError, match="new_state"):
        adapter.validate(paint_command(new_state=None), context())


def test_water_is_rejected_on_command_states_both_roles() -> None:
    adapter = make_adapter()
    # Scalar spelling.
    with pytest.raises(AdapterSchemaError, match="water"):
        adapter.validate(
            paint_command(new_state={"window": dict(WINDOW), "classes": 7}),
            context(),
        )
    # Hidden inside a sequence.
    with pytest.raises(AdapterSchemaError, match="water"):
        adapter.validate(
            paint_command(
                new_state={
                    "window": dict(WINDOW),
                    "classes": [5] * 99 + [7],
                }
            ),
            context(),
        )
    # old_state is a command state too: a claimed history carrying water
    # is rejected exactly like a paint request.
    with pytest.raises(AdapterSchemaError, match="water"):
        adapter.validate(
            paint_command(
                new_state={"window": dict(WINDOW), "classes": 5},
                old_state={"window": dict(WINDOW), "classes": 7},
            ),
            context(),
        )
    with pytest.raises(AdapterSchemaError, match="water"):
        adapter.validate(
            paint_command(
                new_state={"window": dict(WINDOW), "classes": 5},
                old_state={
                    "window": dict(WINDOW),
                    "classes": [7] + [1] * 99,
                },
            ),
            context(),
        )


def test_remapped_and_unknown_classes_are_rejected_on_command_states() -> None:
    assert set(UNPAINTABLE_CLASSES) - {7} == {0, 3, 4, 8, -2}
    adapter = make_adapter()
    for code in (0, 3, 4, 8, -2):
        # new_state scalar...
        with pytest.raises(AdapterSchemaError, match="not valid"):
            adapter.validate(
                paint_command(
                    new_state={"window": dict(WINDOW), "classes": code}
                ),
                context(),
            )
        # ...and old_state (both command states are fenced).
        with pytest.raises(AdapterSchemaError, match="not valid"):
            adapter.validate(
                paint_command(
                    new_state={"window": dict(WINDOW), "classes": 5},
                    old_state={"window": dict(WINDOW), "classes": code},
                ),
                context(),
            )
    # Sequence spellings are fenced per element.
    with pytest.raises(AdapterSchemaError, match="not valid"):
        adapter.validate(
            paint_command(
                new_state={
                    "window": dict(WINDOW),
                    "classes": [5] * 98 + [4, 6],
                }
            ),
            context(),
        )


def test_planner_rejects_bypass_built_paint_payloads() -> None:
    """Delta-payload fence: a hand-built delta never plans (u-b wave)."""
    planner = EditPlanner(grid=GRID, registry=builtin_registry())
    area = 16
    for code, pattern in ((7, "water"), (0, "invalid classes"),
                          (3, "invalid classes"), (8, "invalid classes")):
        patch = LandCoverPaintPatch(
            window=RasterWindow(0, 4, 0, 4),
            before_classes=(-1,) * area,
            after_classes=(code,) * area,
        )
        with pytest.raises(PlanningError, match=pattern):
            planner.plan([synthetic_edit((patch,))], initial_state())
    # Negative codes are the data model's unspecified markers: they never
    # paint, so the fence deliberately skips them.
    marker = LandCoverPaintPatch(
        window=RasterWindow(0, 4, 0, 4),
        before_classes=(-1,) * area,
        after_classes=(5, -1) * 8,
    )
    plan = planner.plan([synthetic_edit((marker,))], initial_state())
    assert [i.node_id for i in plan.node_impacts] == [
        "radiation", "surface_thermal_state", "tmrt", "utci",
    ]


def test_staging_door_revalidates_the_payload() -> None:
    """The staging door re-runs the fence + window sanity (wave standard)."""
    adapter = make_adapter()
    transaction = ScenarioTransaction(scenario_id="scenario-a")
    bad_payloads = [
        # Water in the after payload.
        paint_patch(after=7),
        # Remapped/unknown classes in the after payload.
        paint_patch(after=0),
        paint_patch(after=3),
        paint_patch(after=8),
        paint_patch(after=[5, 7] + [5] * 14),
        # Structural window violations.
        paint_patch(
            window=RasterWindow(-4, 4, 0, 4), rows=4, cols=4, after=5
        ),
        paint_patch(
            window=RasterWindow(5, 5, 0, 4), rows=0, cols=4, after=5
        ),
    ]
    for patch in bad_payloads:
        delta = LandCoverPaintDelta("landcover", ADAPTER_ID, (patch,))
        with pytest.raises(SourceDeltaError):
            adapter.apply_source_delta(delta, transaction)
    assert transaction.staged_deltas() == ()


def test_staging_door_fence_is_load_bearing(monkeypatch) -> None:
    """Mutation-sensitivity probe: neuter the staging-door fence and a
    water-carrying delta WOULD stage — the dedicated refusal test above
    fails exactly when this fence call is removed, so it is not vacuous."""
    import solweig_gpu.incremental.adapters.landcover as landcover_module

    monkeypatch.setattr(
        landcover_module, "_check_paint_payload", lambda delta: None
    )
    adapter = make_adapter()
    hand_built = LandCoverPaintDelta(
        "landcover", ADAPTER_ID, (paint_patch(after=7),)
    )
    transaction = ScenarioTransaction(scenario_id="scenario-a")
    adapter.apply_source_delta(hand_built, transaction)
    assert transaction.staged_deltas() == (hand_built,)


# ---------------------------------------------------------------------------
# Planning: closure, scopes, reusability
# ---------------------------------------------------------------------------


def test_impact_plan_closures_and_reusable_geometry_caches() -> None:
    edit = paint_edit()
    plan = make_adapter().impact_plan(edit, context())
    assert plan.changed_sources == ("landcover",)
    # Graph-driven closure: the ONLY landcover edges are
    # [landcover, radiation] and [landcover, surface_thermal_state];
    # everything below follows transitively (radiation -> tmrt,
    # surface_thermal_state -> tmrt, tmrt -> utci/wbgt-pruned).
    assert [i.node_id for i in plan.node_impacts] == [
        "radiation",
        "surface_thermal_state",
        "tmrt",
        "utci",
    ]
    # Geometry and view-factor caches stay reusable: no landcover edge
    # reaches walls/wall_aspect (building_dsm only), the visibilities
    # (geometry sources only), svf (visibilities only), time_shadow
    # (solar + visibilities + walls family), or solar_atmospheric_state
    # (meteorology/selected_date_time only).
    for reusable in (
        "walls",
        "wall_aspect",
        "building_visibility",
        "vegetation_visibility",
        "svf",
        "relative_geometry",
        "time_shadow",
        "solar_atmospheric_state",
        "vegetation_dsm",
        "building_dsm",
        "dem",
        "meteorology",
        "model_parameters",
    ):
        assert reusable in plan.reusable_nodes, reusable
    assert "time_shadow" not in {i.node_id for i in plan.node_impacts}
    # wbgt is never planned; the pruning is recorded, never silent.
    assert "wbgt" not in {i.node_id for i in plan.node_impacts}
    assert "wbgt" not in plan.reusable_nodes
    assert any("output wbgt" in reason for reason in plan.fallback_reasons)


def test_impact_plan_is_windowed_over_the_paint_footprint() -> None:
    edit = paint_edit()
    plan = make_adapter().impact_plan(edit, context())
    scopes = scopes_by_node(plan)
    # The registry's nominal scope is `windows` and the delta's windows
    # ARE the paint footprint: every dirty stage stays local over it.
    for node_id in ("radiation", "surface_thermal_state", "tmrt", "utci"):
        impact = plan_node(plan, node_id)
        assert scopes[node_id] is SpatialScope.WINDOWS, node_id
        assert impact.write_windows == (PAINT_WINDOW,), node_id
        # Default halo 0: reads equal writes (the GVF-walk halo is the
        # executor's read-window concern, never a write expansion).
        assert impact.read_windows == (PAINT_WINDOW,), node_id
    assert plan.transport_window == PAINT_WINDOW
    assert plan.estimated_memory_bytes > 0
    assert plan.estimated_work_units > 0


def test_temporal_scopes_replay_stateful_and_narrow_requested_times() -> None:
    # With requested times, the time-varying stages narrow to ONE and the
    # ground-heat accumulator replays from timestep 0 to the bound.
    edit = paint_edit(times=(3,))
    plan = make_adapter().impact_plan(edit, context())
    temporal = {i.node_id: i.temporal_scope for i in plan.node_impacts}
    for node_id in ("radiation", "tmrt", "utci"):
        assert temporal[node_id] is TemporalScope.ONE, node_id
        assert plan_node(plan, node_id).time_start == 3
    assert temporal["surface_thermal_state"] is TemporalScope.REPLAY
    replay = plan_node(plan, "surface_thermal_state")
    assert replay.time_start == 0
    assert replay.time_stop == 3
    # Without requested times, time-varying stages stay ALL (never a
    # guessed sub-range) and the replay is unbounded above.
    unbounded = make_adapter().impact_plan(paint_edit(times=None), context())
    t2 = {i.node_id: i.temporal_scope for i in unbounded.node_impacts}
    for node_id in ("radiation", "tmrt", "utci"):
        assert t2[node_id] is TemporalScope.ALL, node_id
    assert t2["surface_thermal_state"] is TemporalScope.REPLAY
    replay2 = plan_node(unbounded, "surface_thermal_state")
    assert replay2.time_start == 0
    assert replay2.time_stop is None


def test_large_paint_is_demoted_to_full_by_safety_policy() -> None:
    # 300x300 = 90000 cells over 250000 = 0.36 dirty fraction >= 0.30:
    # the conservative policy demotes every stage to FULL, recorded.
    big = {
        "row_start": 0, "row_stop": 300, "col_start": 0, "col_stop": 300,
    }
    edit = paint_edit({"window": big, "classes": 6})
    plan = make_adapter().impact_plan(edit, context())
    scopes = scopes_by_node(plan)
    for node_id in ("radiation", "surface_thermal_state", "tmrt", "utci"):
        assert scopes[node_id] is SpatialScope.FULL, node_id
        assert plan_node(plan, node_id).write_windows == (), node_id
    assert any(
        "demoted to full by safety policy" in reason
        for reason in plan.fallback_reasons
    )
    assert plan.transport_window is None


def test_adapter_plan_equals_core_planner_plan_bitwise() -> None:
    edit = paint_edit({"window": dict(WINDOW), "classes": 1})
    adapter_plan = make_adapter().impact_plan(edit, context())
    core_plan = EditPlanner(grid=GRID, registry=builtin_registry()).plan(
        [edit], initial_state()
    )
    assert adapter_plan == core_plan
    assert hash(adapter_plan) == hash(core_plan)
    assert repr(adapter_plan) == repr(core_plan)
    # Bitwise determinism on repeat, adapter and core alike; validation
    # is a pure function of the command too.
    assert make_adapter().impact_plan(edit, context()) == adapter_plan
    assert make_adapter().validate(edit.command, context()) == edit


def test_read_halo_expands_reads_never_writes() -> None:
    edit = paint_edit()
    halo = LandCoverSurfaceAdapter(read_halo_pixels=10)
    plan = halo.impact_plan(edit, context())
    expanded = RasterWindow(190, 220, 200, 230)
    for node_id in ("radiation", "surface_thermal_state", "tmrt", "utci"):
        impact = plan_node(plan, node_id)
        assert impact.spatial_scope is SpatialScope.WINDOWS, node_id
        assert impact.read_windows == (expanded,), node_id
        assert impact.write_windows == (PAINT_WINDOW,), node_id
    # Estimates keep the write footprint (the halo is not extra work in
    # the plan's cost model) and the transport bbox is unchanged.
    core = make_adapter().impact_plan(edit, context())
    assert plan.estimated_memory_bytes == core.estimated_memory_bytes
    assert plan.transport_window == core.transport_window
    # The halo parameter is bounded like the vegetation adapter's.
    with pytest.raises(ValueError):
        LandCoverSurfaceAdapter(read_halo_pixels=-1)
    with pytest.raises(ValueError):
        LandCoverSurfaceAdapter(read_halo_pixels=True)


# ---------------------------------------------------------------------------
# U-C mixed scope: vegetation WINDOWS coexist with paint WINDOWS
# ---------------------------------------------------------------------------


def test_mixed_batch_one_plan_windows_and_windows_coexist() -> None:
    lc = make_adapter()
    veg = veg_adapter()
    ctx = context()
    lc_edit = lc.validate(
        paint_command(
            edit_id="lc-1",
            new_state={
                "window": {"row_start": 400, "row_stop": 404,
                           "col_start": 400, "col_stop": 404},
                "classes": 5,
            },
        ),
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
    plan = planner.plan([veg_edit, lc_edit], initial_state())

    # ONE plan for the heterogeneous batch, both sources recorded.
    assert plan.changed_sources == ("landcover", "vegetation_dsm")
    (veg_window,) = veg_edit.delta.windows
    (paint_win,) = lc_edit.delta.patches[0].window,
    paint_window = lc_edit.delta.patches[0].window
    assert paint_win == paint_window

    # U-C mixed-scope ruling: the vegetation-local geometry chain keeps
    # ITS windows; the shared landcover-driven stages carry the MERGED
    # window set (both strokes are local, so both coexist as WINDOWS).
    for node_id in ("relative_geometry", "vegetation_visibility", "svf",
                    "time_shadow"):
        impact = plan_node(plan, node_id)
        assert impact.spatial_scope is SpatialScope.WINDOWS, node_id
        assert impact.write_windows == (veg_window,), node_id
    for node_id in ("radiation", "surface_thermal_state", "tmrt", "utci"):
        impact = plan_node(plan, node_id)
        assert impact.spatial_scope is SpatialScope.WINDOWS, node_id
        # The strokes are disjoint, so merging keeps both windows.
        assert len(impact.write_windows) == 2, node_id
        assert {veg_window, paint_window} == set(impact.write_windows), node_id
    assert plan.transport_window == veg_window.union(paint_window)
    assert planner.plan([veg_edit, lc_edit], initial_state()) == plan


def test_wbgt_is_never_planned_in_any_resulting_plan() -> None:
    lc = make_adapter()
    veg = veg_adapter()
    ctx = context()
    lc_edit = lc.validate(
        paint_command(
            new_state={
                "window": {"row_start": 400, "row_stop": 404,
                           "col_start": 400, "col_stop": 404},
                "classes": 5,
            }
        ),
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
    for edits in ([lc_edit], [veg_edit], [veg_edit, lc_edit]):
        plan = planner.plan(edits, initial_state())
        assert "wbgt" not in {i.node_id for i in plan.node_impacts}
        assert "wbgt" not in plan.reusable_nodes
        assert any("output wbgt" in reason for reason in plan.fallback_reasons)

    # Requesting wbgt outright is refused loudly, never silently dropped.
    wbgt_edit = lc.validate(
        paint_command(
            new_state={"window": dict(WINDOW), "classes": 5},
            outputs=("wbgt",),
        ),
        ctx,
    )
    with pytest.raises(PlanningError, match="never be planned"):
        planner.plan([wbgt_edit], initial_state())


# ---------------------------------------------------------------------------
# The LandCoverOverlay and the landcover_overlay_from_deltas seam
# ---------------------------------------------------------------------------


def test_overlay_from_deltas_concatenates_and_rejects_foreign() -> None:
    lc = make_adapter()
    ctx = context()
    first = lc.validate(
        paint_command(
            edit_id="lc-a", new_state={"window": dict(WINDOW), "classes": 5}
        ),
        ctx,
    )
    second = lc.validate(
        paint_command(
            edit_id="lc-b",
            new_state={
                "window": {"row_start": 300, "row_stop": 302,
                           "col_start": 10, "col_stop": 12},
                "classes": 1,
            },
        ),
        ctx,
    )
    overlay = landcover_overlay_from_deltas([first.delta, second.delta])
    assert overlay.patches == (
        first.delta.patches[0],
        second.delta.patches[0],
    )
    # Empty fold is a legal identity overlay.
    assert landcover_overlay_from_deltas([]).patches == ()
    # Foreign payloads are refused at the seam.
    with pytest.raises(SourceDeltaError, match="LandCoverPaintDelta"):
        landcover_overlay_from_deltas([42])
    foreign_node = LandCoverPaintDelta(
        source_node_id="vegetation_dsm",
        adapter_id=ADAPTER_ID,
        patches=first.delta.patches,
    )
    with pytest.raises(SourceDeltaError, match="source node"):
        landcover_overlay_from_deltas([foreign_node])
    foreign_adapter = LandCoverPaintDelta(
        source_node_id="landcover",
        adapter_id="vegetation_geometry",
        patches=first.delta.patches,
    )
    with pytest.raises(SourceDeltaError, match="adapter"):
        landcover_overlay_from_deltas([foreign_adapter])
    # The overlay constructor re-runs the fence: bypass-built patches
    # carrying water never construct.
    with pytest.raises(SourceDeltaError, match="water"):
        LandCoverOverlay(patches=(paint_patch(after=7),))
    with pytest.raises(SourceDeltaError, match="invalid classes"):
        LandCoverOverlay(patches=(paint_patch(after=3),))
    with pytest.raises(SourceDeltaError, match="negative raster indices"):
        LandCoverOverlay(
            patches=(
                paint_patch(window=RasterWindow(-4, 0, 0, 4), after=5),
            )
        )


def test_overlay_resolve_is_copy_on_write_declared_cells_only() -> None:
    baseline = np.ones((500, 500), dtype=np.uint8)
    frozen = baseline.copy()
    overlay = LandCoverOverlay(
        patches=(
            paint_patch(row=200, col=210, rows=10, cols=10, after=5),
            paint_patch(row=200, col=210, rows=10, cols=10, after=6),
        )
    )
    resolved = overlay.resolve(baseline)
    # The baseline array is bitwise untouched (hard invariant).
    assert np.array_equal(baseline, frozen)
    assert resolved is not baseline
    assert resolved.dtype == np.uint8
    # Last-write-wins per cell across patches in arrival order.
    assert np.all(resolved[200:210, 210:220] == 6)
    mask = overlay.changed_cell_mask(baseline.shape)
    assert mask.sum() == 100
    assert np.all(resolved[mask] == 6)
    assert np.array_equal(resolved[~mask], baseline[~mask])
    # Resolving twice keeps the baseline intact and is deterministic.
    assert np.array_equal(overlay.resolve(baseline), resolved)
    assert np.array_equal(baseline, frozen)


def test_overlay_resolve_skips_unspecified_markers() -> None:
    baseline = np.ones((8, 8), dtype=np.uint8)
    # -1 markers (the data model's partial-coverage contract) never paint.
    overlay = LandCoverOverlay(
        patches=(
            paint_patch(row=0, col=0, rows=2, cols=2, after=(5, -1, 6, -1)),
        )
    )
    resolved = overlay.resolve(baseline)
    assert resolved[0, 0] == 5
    assert resolved[0, 1] == 1  # untouched
    assert resolved[1, 0] == 6
    assert resolved[1, 1] == 1  # untouched
    assert overlay.changed_cell_mask((8, 8)).sum() == 2


def test_overlay_resolve_on_memmap_keeps_file_bytes_unchanged(tmp_path) -> None:
    path = tmp_path / "landcover.u8.npy"
    mm = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.uint8, shape=(64, 64)
    )
    mm[:] = 1
    mm.flush()
    del mm
    # Reopen read-only: this is the cache-backed access pattern
    # (solver.py:1099-1106 reads `cache.window("landcover", ...)`).
    baseline = np.lib.format.open_memmap(path, mode="r")
    overlay = LandCoverOverlay(
        patches=(paint_patch(row=10, col=12, rows=2, cols=2, after=2),)
    )
    resolved = overlay.resolve(baseline)
    assert np.all(resolved[10:12, 12:14] == 2)
    del baseline, resolved
    # The file on disk never changed: copy-on-write even for memmaps.
    reloaded = np.load(path)
    assert np.all(reloaded[10:12, 12:14] == 1)
    assert np.all(reloaded == 1)


def test_overlay_resolve_rejects_shape_violations() -> None:
    overlay = LandCoverOverlay(
        patches=(paint_patch(row=0, col=0, rows=2, cols=2, after=5),)
    )
    with pytest.raises(SourceDeltaError, match="2-D"):
        overlay.resolve(np.zeros(4, dtype=np.uint8))
    outside = LandCoverOverlay(
        patches=(paint_patch(row=62, col=0, rows=4, cols=2, after=5),)
    )
    with pytest.raises(SourceDeltaError, match="outside the baseline"):
        outside.resolve(np.ones((64, 64), dtype=np.uint8))


def test_engine_coalescing_concatenates_paint_patches() -> None:
    lc = make_adapter()
    ctx = context()
    edits = [
        lc.validate(
            paint_command(
                edit_id="lc-a", new_state={"window": dict(WINDOW), "classes": 5}
            ),
            ctx,
        ),
        lc.validate(
            paint_command(
                edit_id="lc-b",
                new_state={
                    "window": {"row_start": 300, "row_stop": 302,
                               "col_start": 10, "col_stop": 12},
                    "classes": 1,
                },
            ),
            ctx,
        ),
    ]
    transaction = ScenarioTransaction(scenario_id="scenario-a")
    for edit in edits:
        lc.apply_source_delta(edit.delta, transaction)
    committed = transaction.commit()
    coalesced = coalesce_source_deltas(committed)
    assert isinstance(coalesced, LandCoverPaintDelta)
    assert len(coalesced.patches) == 2  # arrival order, last-write-wins
    overlay = landcover_overlay_from_deltas([coalesced])
    assert len(overlay.patches) == 2


# ---------------------------------------------------------------------------
# Transaction staging, refused operations, guards
# ---------------------------------------------------------------------------


def test_apply_stages_rollback_safe_deltas() -> None:
    adapter = make_adapter()
    edit_a = paint_edit(
        {"window": dict(WINDOW), "classes": 5}, edit_id="lc-a"
    )
    edit_b = paint_edit(
        {
            "window": {"row_start": 300, "row_stop": 302,
                       "col_start": 10, "col_stop": 12},
            "classes": 1,
        },
        edit_id="lc-b",
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


def test_apply_rejects_foreign_deltas() -> None:
    adapter = make_adapter()
    edit = paint_edit()
    transaction = ScenarioTransaction(scenario_id="scenario-a")
    with pytest.raises(SourceDeltaError, match="LandCoverPaintDelta"):
        adapter.apply_source_delta("not-a-delta", transaction)
    foreign_node = LandCoverPaintDelta(
        source_node_id="vegetation_dsm",
        adapter_id=ADAPTER_ID,
        patches=edit.delta.patches,
    )
    with pytest.raises(SourceDeltaError, match="source node"):
        adapter.apply_source_delta(foreign_node, transaction)
    foreign_adapter = LandCoverPaintDelta(
        source_node_id="landcover",
        adapter_id="vegetation_geometry",
        patches=edit.delta.patches,
    )
    with pytest.raises(SourceDeltaError, match="adapter"):
        adapter.apply_source_delta(foreign_adapter, transaction)
    assert transaction.staged_deltas() == ()


def test_polygon_assign_and_reset_are_refused_loudly() -> None:
    adapter = make_adapter()
    with pytest.raises(EditStateError, match="polygon_assign.*not implemented"):
        adapter.validate(
            paint_command("polygon_assign", new_state={"classes": 5}), context()
        )
    with pytest.raises(EditStateError, match="reset.*not implemented"):
        adapter.validate(paint_command("reset", new_state={"classes": 5}), context())


def test_stale_revisions_and_foreign_commands_are_refused() -> None:
    adapter = make_adapter()
    with pytest.raises(EditStateError, match="stale scene revision"):
        adapter.validate(
            paint_command(revision=3, new_state={"window": dict(WINDOW),
                                                 "classes": 5}),
            context(revision=4),
        )
    foreign = paint_command(new_state={"window": dict(WINDOW), "classes": 5})
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
            paint_command("update", new_state={"window": dict(WINDOW),
                                               "classes": 5}),
            context(),
        )


# ---------------------------------------------------------------------------
# Preview, fixtures, exports
# ---------------------------------------------------------------------------


def test_preview_descriptor_is_material_feedback_and_not_exact() -> None:
    adapter = make_adapter()
    edit = paint_edit()
    descriptor = adapter.preview_descriptor(edit, context())
    assert descriptor.kind == "surface_material"
    assert descriptor.immediate is True
    text = " ".join(descriptor.limitations)
    assert "preview_is_not_exact" in text
    assert "Water" in text and "class 7" in text  # the fence is disclosed
    assert "replay" in text  # replay_if_stateful is disclosed
    assert "reusable" in text  # geometry caches stay valid


def test_validation_fixtures_match_registry_entry() -> None:
    adapter = make_adapter()
    assert adapter.validation_fixtures() == ("class_replace", "mixed_mask",
                                             "reset")


def test_adapter_exports_from_package() -> None:
    from solweig_gpu.incremental import adapters

    assert adapters.LANDCOVER_ADAPTER_ID == ADAPTER_ID
    assert adapters.LANDCOVER_ADAPTER_SCHEMA_VERSION == ADAPTER_SCHEMA_VERSION
    assert adapters.LandCoverSurfaceAdapter is LandCoverSurfaceAdapter
    assert adapters.LandCoverOverlay is LandCoverOverlay
    assert adapters.landcover_overlay_from_deltas is landcover_overlay_from_deltas
    assert callable(adapters.register_landcover_surface_adapter)
    for name in ("PAINTABLE_CLASSES", "FENCED_CLASSES"):
        assert hasattr(adapters, name), name
        assert name in adapters.__all__, name

    import solweig_gpu.incremental as incremental

    assert incremental.LandCoverSurfaceAdapter is LandCoverSurfaceAdapter
    assert incremental.LANDCOVER_SOURCE_NODE == "landcover"
    assert incremental.LANDCOVER_ADAPTER_ID == ADAPTER_ID
