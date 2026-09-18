# SPDX-License-Identifier: GPL-3.0-only
"""Vegetation-geometry edit adapter: the first concrete EditAdapter.

This adapter implements the edit-adapter contract
(``docs/incremental_design_tool/universal_editing/edit_adapter_contract.md``)
for the ``vegetation_geometry`` registry entry by *wrapping* the existing
tree machinery — :mod:`solweig_gpu.incremental.trees`,
:mod:`solweig_gpu.incremental.edits`, and
:func:`solweig_gpu.incremental.geometry.dirty_window_for_edit` — instead of
duplicating any physics or invalidation logic:

- **validate** maps user state (position, height, canopy radius,
  ``trunk_ratio``) to :class:`~solweig_gpu.incremental.edits.TreeEdit`
  old/new specs, enforces the registry's property schema through
  :meth:`solweig_gpu.incremental.edit_registry.AdapterRegistry.validate_edit_state`
  (so the transmissivity rejection, water fences, etc. are the *registry's*
  rulings, never re-implemented here), and returns a
  :class:`~solweig_gpu.incremental.edit_types.ValidatedEdit` carrying a
  :class:`~solweig_gpu.incremental.edit_types.VegetationObjectDelta` whose
  windows are the old-union-new influence windows.
- **source_delta** returns the delta built during validation (single
  validation pass; the planner never re-derives payloads).
- **impact_plan** delegates to the core
  :class:`~solweig_gpu.incremental.planner.EditPlanner` for the single-edit
  batch, so adapter plans and engine plans are bitwise-identical by
  construction (same dirty closure, same spatial/temporal scopes, same
  estimate formulas, same determinism).
- **apply_source_delta** stages the typed delta in a
  :class:`~solweig_gpu.incremental.edit_types.ScenarioTransaction`;
  :func:`tree_edits_from_deltas` /
  :func:`coalesced_batch_from_deltas` convert committed deltas back into the
  coalesced :class:`~solweig_gpu.incremental.edits.TreeEdit` list the
  :class:`~solweig_gpu.incremental.worker.ExactWorker` consumes today (the
  U-C executor seam).
- **preview_descriptor** declares the registry's
  ``vegetation_geometry_and_shadow`` preview with mandatory
  not-exact/wind/transmissivity disclosures (mission invariant
  ``preview_is_not_exact``).

Design notes
------------

**Registry metadata is the single source of truth.** The adapter's
``metadata`` attribute *is* the builtin ``vegetation_geometry`` entry, so
:meth:`AdapterRegistry.register` is idempotent against a
:func:`~solweig_gpu.incremental.edit_registry.builtin_registry` and any
accidental redefinition of the entry is an error, not a silent override.
``validation_fixtures`` and the preview kind are read from that metadata, so
the adapter can never drift from ``edit_registry.yaml`` (the drift-guard
test enforces the YAML side).

**Why the adapter does not own planning logic.** The mission invariant
``dependency_graph_is_versioned`` means the *graph*, not a per-adapter
guess, decides the dirty closure: a vegetation edit dirties
``relative_geometry -> vegetation_visibility -> svf -> time_shadow ->
radiation -> surface_thermal_state -> tmrt -> utci`` (and ``wbgt``, which
sits behind ``time_shadow`` in the reconciled graph), while the building
chain (``walls``, ``wall_aspect``, ``building_visibility``), ``dem``, the
forcing sources, and ``output_selection`` stay reusable. Delegating to the
core planner keeps that decision in one place; the adapter only supplies
the delta and the windows.

Conservative choices (contract ambiguities resolved the safe way):

1. **Unknown state fields are rejected.** Only ``tree_id`` plus
   :data:`EDITABLE_PROPERTIES` are legal; anything else — including
   ``transmissivity`` — is an error naming the registry schema. The
   registry's own ``validate_edit_state`` runs first, so the
   transmissivity error text cites the lead ruling (inert in physics,
   global ``transVeg = 0.03``, ``utci_process.py:606``).
2. **Property values must be real numbers** (``int``/``float``, never
   ``bool`` or numeric strings). ``TreeSpec`` would accept ``float("10")``,
   which would silently smuggle strings into the geometry layer.
3. **Both states are schema-checked.** ``old_state`` passes through the
   same registry + field validation as ``new_state`` (a delete whose
   ``old_state`` carries ``transmissivity`` is rejected too).
4. **Server preset ranges apply to every spec built from user state** via
   :func:`solweig_gpu.incremental.trees.validate_tree_preset` (height
   3-40 m, canopy diameter 1-30 m), matching the documented server-side
   tree validation.
5. **Identity edits are rejected at validation.** A ``move``/``update``
   whose merged new state equals the old state is an explicit
   :class:`~solweig_gpu.incremental.edit_types.EditStateError` naming the
   property and both (equal) values (U-D item d: an identical-value
   resubmit must refuse cleanly instead of churning dirty windows for a
   physically identical scene; add+delete cancellation across *separate
   commands* remains the planner's no-op path).
6. **``move``/``update`` merge over the old state.** Unspecified
   properties carry over from ``old_state`` exactly like
   :meth:`solweig_gpu.incremental.trees.TreeLayer.update_tree`
   (``dataclasses.replace`` semantics); before/after mappings in the delta
   always carry the *full* property set so the TreeEdit round-trip is
   lossless.
7. **Stale revisions are refused early.** ``validate`` requires
   ``command.base_scene_revision == context.scene_revision`` (UEDIT-006);
   the planner re-checks against the authoritative
   :class:`~solweig_gpu.incremental.edit_graph.SceneGraphState`.
8. **Delta windows keep empty results.** A tree whose influence clamps
   away off-grid yields an empty window rather than a guessed full window;
   the planner's conservative clamp rule then decides (FULL, never a
   zero-area job).
9. **Read-window halo is opt-in.** Plan write windows are the core
   planner's (grid-clamped merged delta windows, so adapter and engine
   plans compare bitwise-equal); with ``read_halo_pixels > 0`` the adapter
   additionally expands each windowed stage's *read* windows by that halo
   (clamped), leaving write scopes and estimates untouched. The executor
   (U-C) expands read windows further via
   :func:`solweig_gpu.incremental.solver.required_halo_pixels` /
   the worker's ``write_margin_pixels`` — this adapter never claims the
   halo is exact.

Sun positions and the influence policy are injected at construction
(``sun_positions``, ``influence_config``): the adapter holds no site state.
The U-C integration wires them from the site cache exactly as
``ExactWorker._sun_positions`` / ``ExactWorker._site_influence_config`` do;
without sun positions the windows are the geometry/SVF-disc bound only
(the directional corridor term of
:func:`solweig_gpu.incremental.geometry.dirty_window_for_edit` is empty),
which is documented here rather than papered over.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

from ..edit_graph import EditGraph, SceneGraphState, default_edit_graph
from ..edit_registry import (
    VEGETATION_SOURCE_NODE,
    AdapterRegistry,
    EditAdapter,
    builtin_adapter_metadata,
    builtin_registry,
)
from ..edit_types import (
    EditCommand,
    EditStateError,
    ImpactPlan,
    NodeImpact,
    ObjectStateChange,
    PreviewDescriptor,
    ScenarioTransaction,
    SiteContext,
    SourceDelta,
    SourceDeltaError,
    SpatialScope,
    ValidatedEdit,
    VegetationObjectDelta,
    identical_value_clause,
)
from ..edits import CoalescedEditBatch, TreeEdit, coalesce_tree_edits
from ..geometry import (
    InfluenceConfig,
    RasterGrid,
    RasterWindow,
    SunPosition,
    TreeSpec,
    dirty_window_for_edit,
)
from ..planner import EditPlanner, SafetyPolicy
from ..trees import validate_tree_preset

__all__ = [
    "ADAPTER_ID",
    "ADAPTER_SCHEMA_VERSION",
    "EDITABLE_PROPERTIES",
    "IDENTITY_PROPERTY",
    "VegetationGeometryAdapter",
    "coalesced_batch_from_deltas",
    "register_default_adapters",
    "tree_edits_from_deltas",
]

#: Registry id this adapter implements (must match the builtin metadata).
ADAPTER_ID = "vegetation_geometry"

#: Adapter property-schema version; bump on any user-state schema change
#: (the contract: an adapter schema change invalidates incompatible edit
#: events and cached source overlays).
ADAPTER_SCHEMA_VERSION = 1

#: State key holding the object identity (maps to ``TreeSpec.tree_id``).
IDENTITY_PROPERTY = "tree_id"

#: Editable vegetation geometry properties, in canonical order. Per the
#: registry lead ruling this deliberately EXCLUDES ``transmissivity``.
EDITABLE_PROPERTIES: tuple[str, ...] = (
    "x_m",
    "y_m",
    "height_m",
    "canopy_radius_m",
    "trunk_ratio",
)

_ALLOWED_STATE_KEYS = frozenset((IDENTITY_PROPERTY, *EDITABLE_PROPERTIES))
_OPERATIONS = frozenset({"add", "move", "update", "delete"})
#: Properties every newly supplied state must carry (trunk_ratio defaults).
_REQUIRED_PROPERTIES = ("x_m", "y_m", "height_m", "canopy_radius_m")
_DEFAULT_TRUNK_RATIO = 0.25

_PREVIEW_LIMITATIONS = (
    "Preview renders canopy/trunk geometry and placeholder shadows only; it "
    "is interaction feedback, not scientific output (preview_is_not_exact)",
    "Preview shadows use an approximate sun direction; exact shade requires "
    "the solver job",
    "Per-tree transmissivity is not represented: the physics consumes the "
    "global transVeg = 0.03 (registry lead ruling 2026-09-02)",
    "The wind field is fixed: vegetation edits never update wind "
    "coefficients (UEDIT-010: dynamic wind needs a separate scientific "
    "extension)",
)


# ---------------------------------------------------------------------------
# State -> TreeSpec mapping (shared by validate and tree_edits)
# ---------------------------------------------------------------------------


def _check_state_mapping(state: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(state, Mapping):
        raise EditStateError(
            f"{label} must be a mapping of vegetation properties, got "
            f"{type(state).__name__}"
        )
    return state


def _check_known_fields(state: Mapping[str, Any], label: str) -> None:
    unknown = sorted(set(state) - _ALLOWED_STATE_KEYS)
    if unknown:
        raise EditStateError(
            f"{label} carries unknown vegetation fields {unknown}; editable "
            f"properties are {list(EDITABLE_PROPERTIES)} plus "
            f"{IDENTITY_PROPERTY!r}. The registry schema for "
            f"{ADAPTER_ID!r} excludes everything else — notably "
            "transmissivity (inert in the physics; global transVeg = 0.03)"
        )


def _require_number(value: Any, key: str, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EditStateError(
            f"{label}: {key} must be a number, got {type(value).__name__}"
        )
    return float(value)


def _tree_id_from(state: Mapping[str, Any], label: str) -> str:
    tree_id = state.get(IDENTITY_PROPERTY)
    if not isinstance(tree_id, str) or not tree_id:
        raise EditStateError(
            f"{label} requires a non-empty string {IDENTITY_PROPERTY!r}"
        )
    return tree_id


def _merged_properties(
    old_state: Mapping[str, Any], new_state: Mapping[str, Any]
) -> dict[str, float]:
    """Merge ``new_state`` over ``old_state`` (``replace`` semantics)."""
    merged: dict[str, Any] = dict(old_state)
    merged.update(new_state)
    merged.setdefault("trunk_ratio", _DEFAULT_TRUNK_RATIO)
    return merged


def _spec_from_properties(
    tree_id: str, properties: Mapping[str, Any], label: str
) -> TreeSpec:
    for key in _REQUIRED_PROPERTIES:
        if key not in properties:
            raise EditStateError(f"{label} is missing required property {key!r}")
    spec = TreeSpec(
        tree_id=tree_id,
        x_m=_require_number(properties["x_m"], "x_m", label),
        y_m=_require_number(properties["y_m"], "y_m", label),
        height_m=_require_number(properties["height_m"], "height_m", label),
        canopy_radius_m=_require_number(
            properties["canopy_radius_m"], "canopy_radius_m", label
        ),
        trunk_ratio=_require_number(
            properties["trunk_ratio"], "trunk_ratio", label
        ),
    )
    # Structural checks first (finiteness, positivity), then the documented
    # server preset ranges — the same validation every edited tree passes
    # on its way into TreeLayer.
    validate_tree_preset(spec)
    return spec


def _properties_from_spec(spec: TreeSpec) -> dict[str, float]:
    """Full, lossless property mapping for one tree spec."""
    return {key: getattr(spec, key) for key in EDITABLE_PROPERTIES}


def _specs_from_command(
    command: EditCommand, registry: AdapterRegistry
) -> tuple[TreeSpec | None, TreeSpec | None]:
    """Validate states and map one command to ``(old_spec, new_spec)``."""
    operation = command.operation
    if operation not in _OPERATIONS:
        # The registry check below reports undeclared operations; this guard
        # keeps the shape checks honest for custom registries.
        raise EditStateError(
            f"operation must be one of {sorted(_OPERATIONS)}, got {operation!r}"
        )
    old_state = command.old_state
    new_state = command.new_state
    if operation == "add" and old_state is not None:
        raise EditStateError("an 'add' must not carry an old_state")
    if operation == "delete" and new_state is not None:
        raise EditStateError("a 'delete' must not carry a new_state")
    if operation in ("move", "update") and (old_state is None or new_state is None):
        raise EditStateError(
            f"a {operation!r} requires both old_state and new_state"
        )
    if operation == "add" and new_state is None:
        raise EditStateError("an 'add' requires a new_state")
    if operation == "delete" and old_state is None:
        raise EditStateError("a 'delete' requires an old_state")

    # Registry schema enforcement first (rejected properties, fences) on
    # BOTH states, then the adapter's own field-level checks.
    for label, state in (("old_state", old_state), ("new_state", new_state)):
        if state is None:
            continue
        mapping = _check_state_mapping(state, label)
        registry.validate_edit_state(ADAPTER_ID, mapping)
        _check_known_fields(mapping, label)

    old_spec: TreeSpec | None = None
    new_spec: TreeSpec | None = None
    if old_state is not None:
        old_id = _tree_id_from(old_state, "old_state")
        base = dict(old_state)
        base.setdefault("trunk_ratio", _DEFAULT_TRUNK_RATIO)
        old_spec = _spec_from_properties(old_id, base, "old_state")
    if new_state is not None:
        new_id = _tree_id_from(new_state, "new_state")
        if old_spec is not None and new_id != old_spec.tree_id:
            raise EditStateError(
                f"old_state and new_state describe different trees "
                f"({old_spec.tree_id!r} vs {new_id!r})"
            )
        if old_state is not None:
            merged = _merged_properties(old_state, new_state)
        else:
            merged = dict(new_state)
            merged.setdefault("trunk_ratio", _DEFAULT_TRUNK_RATIO)
        new_spec = _spec_from_properties(new_id, merged, "new_state")
    if old_spec == new_spec:
        # U-D item (d): an identical-value resubmit (declared old_state ==
        # merged new_state) must refuse naming the property and both
        # values; without the refusal the edit churns dirty windows for a
        # physically identical scene.
        properties = (
            _properties_from_spec(old_spec) if old_spec is not None else {}
        )
        raise EditStateError(
            "edit is a no-op: tree "
            f"{(new_spec or old_spec).tree_id!r} already carries the "
            "requested state — every declared property is value-equal "
            "between old_state and new_state ("
            + identical_value_clause(
                (name, properties[name], properties[name])
                for name in EDITABLE_PROPERTIES
                if name in properties
            )
            + "); an identical-value resubmit would churn dirty windows "
            "for a physically identical scene (add+delete cancellation "
            "across separate commands is the planner's job, not a valid "
            "single edit)"
        )
    return old_spec, new_spec


# ---------------------------------------------------------------------------
# Committed deltas -> TreeEdit batch (the ExactWorker seam)
# ---------------------------------------------------------------------------


def _spec_from_frozen(
    tree_id: str, properties: Mapping[str, Any] | None, label: str
) -> TreeSpec | None:
    if properties is None:
        return None
    missing = [
        key for key in (*_REQUIRED_PROPERTIES, "trunk_ratio") if key not in properties
    ]
    if missing:
        raise SourceDeltaError(
            f"{label} for object {tree_id!r} is missing properties {missing}; "
            "deltas staged by this adapter always carry the full property set"
        )
    return TreeSpec(
        tree_id=tree_id,
        x_m=float(properties["x_m"]),
        y_m=float(properties["y_m"]),
        height_m=float(properties["height_m"]),
        canopy_radius_m=float(properties["canopy_radius_m"]),
        trunk_ratio=float(properties["trunk_ratio"]),
    )


def tree_edits_from_deltas(
    deltas: Sequence[VegetationObjectDelta],
    *,
    scenario_id: str,
    start_sequence: int = 1,
) -> tuple[TreeEdit, ...]:
    """Convert committed vegetation deltas to the worker's ``TreeEdit`` list.

    One :class:`~solweig_gpu.incremental.edits.TreeEdit` per object change,
    in arrival order, with deterministic global sequence numbers starting at
    ``start_sequence`` (unique within the batch, as
    :func:`~solweig_gpu.incremental.edits.coalesce_tree_edits` requires).
    The originating commands' ``base_scene_revision`` (single-scenario,
    UEDIT-006-checked by the planner) travels beside this list — the delta
    family deliberately carries no revision of its own.
    """
    if not scenario_id:
        raise SourceDeltaError("scenario_id must be non-empty")
    if start_sequence < 1:
        raise SourceDeltaError("start_sequence must be >= 1")
    edits: list[TreeEdit] = []
    sequence = start_sequence
    for delta in deltas:
        if not isinstance(delta, VegetationObjectDelta):
            raise SourceDeltaError(
                f"expected VegetationObjectDelta, got {type(delta).__name__}"
            )
        if delta.adapter_id != ADAPTER_ID:
            raise SourceDeltaError(
                f"delta belongs to adapter {delta.adapter_id!r}, not {ADAPTER_ID!r}"
            )
        if delta.source_node_id != VEGETATION_SOURCE_NODE:
            raise SourceDeltaError(
                f"delta targets source node {delta.source_node_id!r}, not "
                f"{VEGETATION_SOURCE_NODE!r}"
            )
        for change in delta.objects:
            old_tree = _spec_from_frozen(change.object_id, change.before, "before")
            new_tree = _spec_from_frozen(change.object_id, change.after, "after")
            edits.append(
                TreeEdit(
                    scenario_id=scenario_id,
                    sequence=sequence,
                    tree_id=change.object_id,
                    old_tree=old_tree,
                    new_tree=new_tree,
                )
            )
            sequence += 1
    return tuple(edits)


def coalesced_batch_from_deltas(
    deltas: Sequence[VegetationObjectDelta],
    *,
    scenario_id: str,
    start_sequence: int = 1,
) -> CoalescedEditBatch | None:
    """Coalesced worker batch for committed deltas (``None`` when a no-op).

    This is the commit-path hand-off the U-C executor uses: the
    transaction's committed deltas become the coalesced
    :class:`~solweig_gpu.incremental.edits.CoalescedEditBatch` (plus the
    batch's base scene revision) that
    :class:`~solweig_gpu.incremental.worker.ExactWorker` consumes today.
    The executor replays the batch's edits into the scenario's
    :class:`~solweig_gpu.incremental.trees.TreeLayer` so the layer's own
    sequence counter and the worker's published-watermark stay
    authoritative.
    """
    edits = tree_edits_from_deltas(
        deltas, scenario_id=scenario_id, start_sequence=start_sequence
    )
    if not edits:
        return None
    batch = coalesce_tree_edits(edits)
    if not batch.edits:  # every edit cancelled (add + delete)
        return None
    return batch


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------


class VegetationGeometryAdapter(EditAdapter):
    """EditAdapter for per-tree vegetation geometry (registry id above).

    The adapter is a stateless service: site facts (sun positions, the
    influence policy, the safety policy) are injected at construction, and
    every protocol method derives its result from the command, the
    :class:`~solweig_gpu.incremental.edit_types.SiteContext`, and the
    wrapped tree machinery. All outputs are frozen records, so identical
    inputs reproduce bitwise-identical deltas and plans.
    """

    adapter_id: str = ADAPTER_ID
    schema_version: int = ADAPTER_SCHEMA_VERSION

    def __init__(
        self,
        *,
        registry: AdapterRegistry | None = None,
        graph: EditGraph | None = None,
        sun_positions: Sequence[SunPosition] = (),
        influence_config: InfluenceConfig | None = None,
        policy: SafetyPolicy | None = None,
        read_halo_pixels: int = 0,
    ) -> None:
        self._registry = registry if registry is not None else builtin_registry()
        self._graph = graph if graph is not None else default_edit_graph()
        self._sun_positions: tuple[SunPosition, ...] = tuple(sun_positions)
        self._influence_config = (
            influence_config if influence_config is not None else InfluenceConfig()
        )
        self._policy = policy
        if isinstance(read_halo_pixels, bool) or read_halo_pixels < 0:
            raise ValueError("read_halo_pixels must be a non-negative integer")
        self._read_halo_pixels = int(read_halo_pixels)
        # The builtin entry, verbatim: registering this adapter against a
        # builtin_registry() is then idempotent, and any divergent
        # redefinition of the entry is a registry error, not a silent win.
        self.metadata = next(
            entry
            for entry in builtin_adapter_metadata()
            if entry.id == ADAPTER_ID
        )

    # -- protocol: validation -------------------------------------------------

    def validate(
        self, command: EditCommand, context: SiteContext
    ) -> ValidatedEdit:
        """Map one command to a validated edit carrying a typed delta."""
        if command.adapter_id != self.adapter_id:
            raise EditStateError(
                f"command targets adapter {command.adapter_id!r}, not "
                f"{self.adapter_id!r}"
            )
        if command.base_scene_revision != context.scene_revision:
            raise EditStateError(
                f"edit {command.edit_id!r} targets stale scene revision "
                f"{command.base_scene_revision}; context is at revision "
                f"{context.scene_revision} (UEDIT-006: stale revisions "
                "never plan, so never publish)"
            )
        self._check_command(command, context)
        old_spec, new_spec = _specs_from_command(command, self._registry)
        tree_id = (new_spec or old_spec).tree_id
        windows = self._delta_windows(old_spec, new_spec, context.grid)
        delta = VegetationObjectDelta(
            source_node_id=VEGETATION_SOURCE_NODE,
            adapter_id=self.adapter_id,
            objects=(
                ObjectStateChange(
                    object_id=tree_id,
                    before=(
                        _properties_from_spec(old_spec) if old_spec else None
                    ),
                    after=_properties_from_spec(new_spec) if new_spec else None,
                ),
            ),
            windows=windows,
        )
        return ValidatedEdit(
            command=command,
            adapter_id=self.adapter_id,
            schema_version=self.schema_version,
            source_node_id=VEGETATION_SOURCE_NODE,
            delta=delta,
        )

    def tree_edits(
        self, command: EditCommand, context: SiteContext
    ) -> tuple[TreeEdit, ...]:
        """The command's :class:`TreeEdit` view (what the worker consumes).

        Runs the same validation as :meth:`validate` and returns at most one
        edit with a placeholder ``sequence=1``; batch-global sequence
        numbering belongs to :func:`tree_edits_from_deltas` / the TreeLayer
        replay, which is the authoritative staged path.
        """
        self._check_command(command, context)
        old_spec, new_spec = _specs_from_command(command, self._registry)
        tree_id = (new_spec or old_spec).tree_id
        return (
            TreeEdit(
                scenario_id=command.scenario_id,
                sequence=1,
                tree_id=tree_id,
                old_tree=old_spec,
                new_tree=new_spec,
            ),
        )

    # -- protocol: delta, plan, apply, preview, fixtures -----------------------

    def source_delta(
        self, edit: ValidatedEdit, context: SiteContext
    ) -> VegetationObjectDelta:
        """Return the delta produced at validation (single derivation)."""
        self._require_own_delta(edit)
        return edit.delta

    def impact_plan(
        self, edit: ValidatedEdit, context: SiteContext
    ) -> ImpactPlan:
        """The single-edit impact plan, delegated to the core planner.

        The dirty closure, spatial/temporal scopes, safety demotions, and
        estimate formulas are the planner's (and therefore the engine's):
        ``surface_thermal_state`` replays from timestep 0 because the
        planner's ``STATEFUL_NODES`` rule says so, and windowed stages keep
        their merged grid-clamped delta windows. With the default
        ``read_halo_pixels = 0`` the returned plan equals the core
        :meth:`EditPlanner.plan` output for ``[edit]`` bitwise.
        """
        self._require_own_delta(edit)
        state = SceneGraphState(self._graph, context.scene_revision, {})
        planner = EditPlanner(
            grid=context.grid,
            graph=self._graph,
            registry=self._registry,
            policy=self._policy,
        )
        plan = planner.plan([edit], state)
        return self._with_read_halo(plan, context.grid)

    def apply_source_delta(
        self, delta: SourceDelta, transaction: ScenarioTransaction
    ) -> None:
        """Stage the typed delta in a rollback-safe transaction.

        Staging never touches baseline state: the transaction either
        commits (returning the staged deltas for publication — the U-C
        executor turns them into the worker's coalesced TreeEdit batch via
        :func:`coalesced_batch_from_deltas`) or rolls back cleanly. No-op
        deltas (e.g. an add+delete that coalesced away) stage like any
        other delta; the planner already reports them as zero-stage plans.
        """
        if not isinstance(delta, VegetationObjectDelta):
            raise SourceDeltaError(
                f"expected VegetationObjectDelta, got {type(delta).__name__}"
            )
        if delta.adapter_id != self.adapter_id:
            raise SourceDeltaError(
                f"delta belongs to adapter {delta.adapter_id!r}, not "
                f"{self.adapter_id!r}"
            )
        if delta.source_node_id != VEGETATION_SOURCE_NODE:
            raise SourceDeltaError(
                f"delta targets source node {delta.source_node_id!r}, not "
                f"{VEGETATION_SOURCE_NODE!r}"
            )
        transaction.stage(delta)

    def preview_descriptor(
        self, edit: ValidatedEdit, context: SiteContext
    ) -> PreviewDescriptor:
        """Vegetation-geometry preview with mandatory disclosures.

        The preview is immediate client-side feedback (canopy/trunk geometry
        plus placeholder shadows) and is explicitly NOT the scientific
        output — the registry kind and the limitations tuple say so.
        """
        self._require_own_delta(edit)
        return PreviewDescriptor(
            kind=self.metadata.preview,
            immediate=True,
            limitations=_PREVIEW_LIMITATIONS,
        )

    def validation_fixtures(self) -> tuple[str, ...]:
        """Scientific fixture names, verbatim from the registry metadata."""
        return self.metadata.validation_fixtures

    # -- internals ---------------------------------------------------------------

    def _check_command(
        self, command: EditCommand, context: SiteContext
    ) -> None:
        if command.adapter_id != self.adapter_id:
            raise EditStateError(
                f"command targets adapter {command.adapter_id!r}, not "
                f"{self.adapter_id!r}"
            )
        if command.base_scene_revision != context.scene_revision:
            raise EditStateError(
                f"edit {command.edit_id!r} targets stale scene revision "
                f"{command.base_scene_revision}; context is at revision "
                f"{context.scene_revision} (UEDIT-006)"
            )
        self._registry.validate_operation(self.adapter_id, command.operation)

    def _require_own_delta(self, edit: ValidatedEdit) -> VegetationObjectDelta:
        if not isinstance(edit.delta, VegetationObjectDelta):
            raise SourceDeltaError(
                "edit must carry a VegetationObjectDelta, got "
                f"{type(edit.delta).__name__}"
            )
        if edit.delta.adapter_id != self.adapter_id:
            raise SourceDeltaError(
                f"edit delta belongs to adapter {edit.delta.adapter_id!r}"
            )
        return edit.delta

    def _delta_windows(
        self,
        old_spec: TreeSpec | None,
        new_spec: TreeSpec | None,
        grid: RasterGrid,
    ) -> tuple[RasterWindow, ...]:
        """Conservative old-union-new influence window, as one window.

        Thin delegation to
        :func:`solweig_gpu.incremental.geometry.dirty_window_for_edit`
        (block-aligned, grid-clamped): the wrapped machinery unions the old
        and new influence bounds internally, which is what makes moves and
        deletes safe.
        """
        window = dirty_window_for_edit(
            grid,
            old_tree=old_spec,
            new_tree=new_spec,
            sun_positions=self._sun_positions,
            config=self._influence_config,
        )
        return (window,)

    def _with_read_halo(self, plan: ImpactPlan, grid: RasterGrid) -> ImpactPlan:
        """Expand windowed stages' read windows by the configured halo."""
        if not self._read_halo_pixels:
            return plan
        impacts = tuple(
            self._expand_impact(impact, grid) for impact in plan.node_impacts
        )
        return replace(plan, node_impacts=impacts)

    def _expand_impact(self, impact: NodeImpact, grid: RasterGrid) -> NodeImpact:
        if impact.spatial_scope is not SpatialScope.WINDOWS:
            return impact  # FULL carries no windows by contract
        read = tuple(
            window.expand(self._read_halo_pixels).clamp(
                rows=grid.rows, cols=grid.cols
            )
            for window in impact.write_windows
        )
        return replace(impact, read_windows=read)


def register_default_adapters(
    registry: AdapterRegistry | None = None,
) -> AdapterRegistry:
    """Register every implemented adapter (idempotent) and return the registry.

    Capability listing for the U-D API is then just
    ``registry.all_metadata()``; concrete adapters resolve through
    ``registry.get_adapter``.
    """
    target = registry if registry is not None else builtin_registry()
    target.register(VegetationGeometryAdapter(registry=target))
    return target
