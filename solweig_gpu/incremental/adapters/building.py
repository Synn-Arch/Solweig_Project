# SPDX-License-Identifier: GPL-3.0-only
"""Building-massing edit adapter (``building_geometry``, wave U-B adapter #5).

This adapter implements the edit-adapter contract
(``docs/incremental_design_tool/universal_editing/edit_adapter_contract.md``)
for the ``building_geometry`` registry entry. Buildings are the one edit
family whose every downstream product is *preprocessing-derived*: walls and
wall aspects come from a separate wall pass over the Building_DSM
(:func:`solweig_gpu.walls_aspect.run_parallel_processing`), and the SVF /
shadow-matrix artefacts come from a site-wide SVF calculation
(:func:`solweig_gpu.solweig_gpu.calculate_svf`). No in-process incremental
path regenerates any of them, and a massing change shifts sky-view scalars
far beyond its footprint. The registry therefore pins the entry at
``full_only_initially`` and the planner honors that **before** any
window/scope logic (H2 remediation,
:meth:`solweig_gpu.incremental.planner.EditPlanner._resolve_source_scope`):
every building edit — every operation, any payload shape — resolves its
source scope to FULL and propagates FULL down the whole dirty closure.

So this adapter wraps no physics and computes no windows:

- **validate** maps the user state (a building id, a footprint polygon, a
  height) through :meth:`AdapterRegistry.validate_edit_state` first, then
  enforces the documented property schema
  (:data:`EDITABLE_PROPERTIES`/:data:`HEIGHT_*`) and builds a
  :class:`~solweig_gpu.incremental.edit_types.BuildingMassingDelta`.
- **source_delta** returns the delta built during validation (single
  validation pass); the delta's ``windows`` stay empty —
  :attr:`SourceDelta.spatial_windows` is ``()`` exactly like the
  model-parameter and forcing adapters, because the spatial effect of a
  massing change is site-wide and entirely downstream. No fake windows.
- **impact_plan** delegates to the core planner for the single-edit batch,
  so adapter plans and engine plans are bitwise-identical by construction.
  The planner's ``full_only_initially`` rule (not this adapter) forces
  FULL; the adapter merely refuses to defeat it.
- **apply_source_delta** stages the typed delta in a
  :class:`~solweig_gpu.incremental.edit_types.ScenarioTransaction` after
  re-validating the whole payload at the staging door (finiteness and
  domains, the model-parameter adapter's MEDIUM remediation): a hand-built
  delta with a non-finite or out-of-domain field is refused *pre-stage*,
  never staged on trust.
- **preview_descriptor** declares the registry's ``massing_and_shadow``
  kind with the mandatory not-exact/full-tile/wind disclosures (mission
  invariant ``preview_is_not_exact``).

Operations vocabulary (registry declares ``add, move, update, delete,
raster_patch``)
-------------------------------------------------------------------------

``add``/``move``/``update``/``delete`` are honored as object-model edits on
conceptual massing blocks (editable_scope.md: "add or remove a building
volume; move a conceptual massing block; change footprint or height").
``raster_patch`` is REFUSED loudly, with the meteorological adapter's
``preset``/``reset`` precedent: the typed delta family carries object
states, not raster cells, and no Building_DSM raster-mutation path exists
anywhere in the incremental code — honoring it would mean inventing a
patch shape the executor has no consumer for. Refused, never guessed.

Executor seam and U-C integration friction
------------------------------------------

The FULL plan is the *demand*; regenerating the preprocessing chain is the
executor's job. Concretely, for one committed building edit the U-C
executor must run (in this order):

1. **Rasterize the massing edits into the scenario Building_DSM** — no
   incremental building rasterizer exists (contrast
   :class:`solweig_gpu.incremental.trees.TreeLayer` for vegetation). The
   executor writes each block's footprint cells as an absolute surface;
   the elevation rule (ground + height, with the ground sampled from the
   DEM under the footprint) is U-C's to fix and validate.
   :func:`massing_edits_from_deltas` hands it the re-validated
   (before, after) :class:`BuildingSpec` pairs.
2. **Re-run the wall pass**: ``run_walls_aspect(preprocess_dir)``
   (solweig_gpu.py:255) → ``run_parallel_processing`` over
   ``Building_DSM/`` writing ``walls/`` and ``aspect/``
   (walls_aspect.py:271). ``walllimit = 3.0`` (walls_aspect.py:25,
   applied at :51): a block below 3 m produces NO walls/wall_aspect
   entries — disclosed in the preview, not silently dropped.
3. **Re-run SVF**: ``calculate_svf(base_path, overwrite=True)``
   (solweig_gpu.py:274) rewrites ``SVF/SkyViewFactor_{tile_key}.tif``,
   ``SVF/svfs_{tile_key}.zip`` and ``SVF/shadowmats_{tile_key}.npz``
   (svf_calculator, shadow.py:409).
4. **Rebuild the site cache**: :func:`solweig_gpu.incremental.cache_builder.build_site_cache`
   (cache_builder.py:82) — the cache hashes the walls/aspect rasters
   (:128) and the SVF artefacts (:137-146) and maps them to the
   ``walls``/``wall_aspect`` layers (:201-202) the solver reads
   (solver.py:1086-1090).
5. **Run the full tile**: :func:`solweig_gpu.incremental.solver.run_full_tile`
   (solver.py:1258) already stages ``Building_DSM/DEM/walls/aspect/
   Landcover`` into its scratch site (solver.py:1189) — the FULL plan maps
   one-to-one onto this path, and no windowed solve may be used (the plan
   carries no windows at all).

Conservative choices (contract ambiguities resolved the safe way):

1. **Full-only, honestly.** The delta reports ``spatial_windows == ()``.
   The engine's hand-built H2 path (which smuggles a window into the
   delta) and this adapter's real path produce *bitwise-identical* plans,
   because the planner drops to FULL on the adapter status before it ever
   looks at windows; a test pins that equality.
2. **One property spelling per domain.** ``height_m`` is a real number
   (never ``bool`` or a numeric string), finite, in ``(0, 1000]`` m — a
   loose plausibility fence (the code enforces no bound anywhere), with
   the 3 m wall threshold disclosed rather than enforced (a short garden
   wall still casts shade). ``footprint_m`` is a sequence of >= 3
   ``(x_m, y_m)`` vertex pairs, each a finite real number; polygon
   simplicity/self-intersection is NOT validated here (no geometry
   library in the incremental layer) — that is the rasterizer's contract
   item, disclosed rather than guessed.
3. **States carry the full property set.** ``move``/``update`` merge
   ``new_state`` over ``old_state`` (``dataclasses.replace`` semantics,
   like the vegetation adapter) and both delta states always carry the
   full property set, so the executor round-trip is lossless. An
   ``old_state`` missing a property is refused (the UI displays the
   current object; it always knows the full state).
4. **Identity edits are rejected at validation**; add/delete state-shape
   violations (an ``add`` with an ``old_state``, a ``delete`` with a
   ``new_state``) are refused exactly like the vegetation adapter's.
5. **Grid overlap is NOT fenced** (mirrors the vegetation sibling, which
   leaves off-grid geometry to window clamping): a footprint wholly
   outside the site regenerates an unchanged Building_DSM. That is an
   executor rasterization concern, documented here rather than silently
   claimed as validated.
6. **The staging door re-validates the payload.** ``apply_source_delta``
   rebuilds every :class:`BuildingSpec` from the frozen before/after
   mappings (full property set + domains + finiteness) before
   ``transaction.stage``; the planner's registry-level delta check has no
   building fence, so this door is the payload's only schema gate.
7. **Stale revisions are refused early.** ``validate`` requires
   ``command.base_scene_revision == context.scene_revision`` (UEDIT-006);
   the planner re-checks against the authoritative scene state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..edit_graph import EditGraph, SceneGraphState, default_edit_graph
from ..edit_registry import (
    AdapterRegistry,
    EditAdapter,
    builtin_adapter_metadata,
    builtin_registry,
)
from ..edit_types import (
    BuildingMassingDelta,
    EditCommand,
    EditStateError,
    ImpactPlan,
    ObjectStateChange,
    PreviewDescriptor,
    ScenarioTransaction,
    SiteContext,
    SourceDelta,
    SourceDeltaError,
    ValidatedEdit,
    identical_value_clause,
)
from ..planner import EditPlanner, SafetyPolicy

__all__ = [
    "ADAPTER_ID",
    "ADAPTER_SCHEMA_VERSION",
    "BUILDING_SOURCE_NODE",
    "BuildingMassingAdapter",
    "BuildingSpec",
    "EDITABLE_PROPERTIES",
    "HEIGHT_MAX_M",
    "HEIGHT_MIN_M",
    "IDENTITY_PROPERTY",
    "MIN_FOOTPRINT_VERTICES",
    "MassingEdit",
    "massing_edits_from_deltas",
    "register_building_massing_adapter",
]

#: Registry id this adapter implements (must match the builtin metadata).
ADAPTER_ID = "building_geometry"

#: Adapter property-schema version; bump on any user-state schema change
#: (the contract: an adapter schema change invalidates incompatible edit
#: events and cached source overlays).
ADAPTER_SCHEMA_VERSION = 1

#: Source node this adapter owns (module-local constant; no registry-level
#: building fence exists, so the shared registry constants stay untouched).
BUILDING_SOURCE_NODE = "building_dsm"

#: State key holding the object identity (the massing block's id).
IDENTITY_PROPERTY = "building_id"

#: Editable massing properties, in canonical order (editable_scope.md:
#: "change footprint or height").
EDITABLE_PROPERTIES: tuple[str, ...] = (
    "footprint_m",
    "height_m",
)

#: Properties every state must carry in full (no defaults exist).
_REQUIRED_PROPERTIES: tuple[str, ...] = ("footprint_m", "height_m")

#: A closed polygon needs at least three distinct vertices; the executor
#: closes the ring itself (first == last is accepted but never required).
MIN_FOOTPRINT_VERTICES = 3

#: Documented height domain (m above the block's base). The code enforces
#: no height bound anywhere; (0, 1000] is a loose plausibility fence — the
#: tallest built structure is ~830 m — flagged rather than invented tight.
#: NOTE: walls_aspect's ``walllimit = 3.0`` (walls_aspect.py:25) is a
#: *disclosure*, not a lower bound: blocks below 3 m are legitimate massing
#: (they still shade) but produce no walls/wall_aspect entries downstream.
HEIGHT_MIN_M = 0.0
HEIGHT_MAX_M = 1000.0

_ALLOWED_STATE_KEYS = frozenset((IDENTITY_PROPERTY, *EDITABLE_PROPERTIES))
_OPERATIONS = frozenset({"add", "move", "update", "delete"})

_PREVIEW_LIMITATIONS = (
    "Preview renders the massing block and an approximate cast shadow "
    "only; it is interaction feedback, not scientific output "
    "(preview_is_not_exact)",
    "Every building edit recomputes the FULL tile: walls, wall aspects, "
    "visibilities, SVF and every time step regenerate through the "
    "preprocessing pipeline (full_only_initially; no validated local "
    "implementation exists)",
    "Blocks below the wall threshold (3 m, walls_aspect.py walllimit) "
    "produce no walls/wall_aspect entries in the regenerated outputs — "
    "that is preprocessing behaviour, disclosed here, not dropped "
    "silently",
    "The wind field is fixed: building edits never update wind "
    "coefficients (UEDIT-010: dynamic wind needs a separate scientific "
    "extension)",
    "'raster_patch' is not implemented: massing edits are object-model "
    "(footprint + height) edits; direct Building_DSM cell edits are "
    "refused rather than guessed",
)


# ---------------------------------------------------------------------------
# The property schema (shared by validate, the staging door, and the seam)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BuildingSpec:
    """One massing block's validated, canonical geometry.

    ``footprint_m`` is a ring of ``(x_m, y_m)`` site-CRS vertices (>= 3,
    finite, not closed by this type — the rasterizer closes it);
    ``height_m`` is the block's height above its base in
    ``(HEIGHT_MIN_M, HEIGHT_MAX_M]``. Construction re-validates
    everything, so a spec that exists is a spec that passed the schema —
    the property the executor seam and the staging door rely on.
    """

    building_id: str
    footprint_m: tuple[tuple[float, float], ...]
    height_m: float

    def __post_init__(self) -> None:
        if not isinstance(self.building_id, str) or not self.building_id:
            raise EditStateError(
                "building_id must be a non-empty string, got "
                f"{type(self.building_id).__name__}"
            )
        object.__setattr__(
            self, "footprint_m", _validated_footprint(self.footprint_m, "spec")
        )
        height = _validated_height(self.height_m, "spec")
        object.__setattr__(self, "height_m", height)


def _require_number(value: Any, label: str) -> float:
    # One spelling per domain: real numbers only — bool is an int subclass
    # and numeric strings would smuggle into the rasterizer unchecked
    # (vegetation/model-parameter precedent).
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EditStateError(
            f"{label} must be a number, got {type(value).__name__}"
        )
    number = float(value)
    # Finiteness BEFORE the domain comparisons (params HIGH remediation):
    # NaN fails every relational check and +inf passes low-side-only
    # bounds, so either would slip to the physics handoff unchecked.
    if not math.isfinite(number):
        raise EditStateError(f"{label} must be finite, got {number!r}")
    return number


def _validated_height(value: Any, label: str) -> float:
    number = _require_number(value, f"{label}: height_m")
    if number <= HEIGHT_MIN_M:
        raise EditStateError(
            f"{label}: height_m = {number!r} must be > {HEIGHT_MIN_M} m; a "
            "zero-height block is a delete, not a massing edit"
        )
    if number > HEIGHT_MAX_M:
        raise EditStateError(
            f"{label}: height_m = {number!r} is above the documented "
            f"plausibility bound {HEIGHT_MAX_M} m (the code enforces no "
            "height bound; this is a loose fence, not a model-validity "
            "range)"
        )
    return number


def _validated_footprint(
    value: Any, label: str
) -> tuple[tuple[float, float], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(
        value, (list, tuple)
    ):
        raise EditStateError(
            f"{label}: footprint_m must be a sequence of (x_m, y_m) "
            f"vertex pairs, got {type(value).__name__}"
        )
    if len(value) < MIN_FOOTPRINT_VERTICES:
        raise EditStateError(
            f"{label}: footprint_m carries {len(value)} vertices; a "
            f"polygon needs at least {MIN_FOOTPRINT_VERTICES}"
        )
    ring: list[tuple[float, float]] = []
    for index, vertex in enumerate(value):
        if isinstance(vertex, (str, bytes)) or not isinstance(
            vertex, (list, tuple)
        ):
            raise EditStateError(
                f"{label}: footprint_m vertex {index} must be an "
                f"(x_m, y_m) pair, got {type(vertex).__name__}"
            )
        if len(vertex) != 2:
            raise EditStateError(
                f"{label}: footprint_m vertex {index} must be an "
                f"(x_m, y_m) pair, got {len(vertex)} coordinates"
            )
        ring.append(
            (
                _require_number(vertex[0], f"{label}: footprint_m x[{index}]"),
                _require_number(vertex[1], f"{label}: footprint_m y[{index}]"),
            )
        )
    return tuple(ring)


def _check_known_fields(state: Mapping[str, Any], label: str) -> None:
    unknown = sorted(set(state) - _ALLOWED_STATE_KEYS)
    if unknown:
        raise EditStateError(
            f"{label} carries unknown building fields {unknown}; editable "
            f"properties are {list(EDITABLE_PROPERTIES)} plus "
            f"{IDENTITY_PROPERTY!r}. There is no building property "
            "vocabulary beyond footprint/height — direct DSM cell edits "
            "('raster_patch') are refused, not modeled as properties"
        )


def _building_id_from(state: Mapping[str, Any], label: str) -> str:
    building_id = state.get(IDENTITY_PROPERTY)
    if not isinstance(building_id, str) or not building_id:
        raise EditStateError(
            f"{label} requires a non-empty string {IDENTITY_PROPERTY!r}"
        )
    return building_id


def _spec_from_state(
    building_id: str, state: Mapping[str, Any], label: str
) -> BuildingSpec:
    # Unknown-field check FIRST so the staging door and the executor seam
    # (which route hand-built payloads through here) reject smuggled
    # properties exactly like the command path does.
    _check_known_fields(state, label)
    missing = [key for key in _REQUIRED_PROPERTIES if key not in state]
    if missing:
        raise EditStateError(f"{label} is missing required properties {missing}")
    # BuildingSpec.__post_init__ re-runs the full domain/finiteness schema
    # (conservative choice 6: one validation path for every entry point).
    return BuildingSpec(
        building_id=building_id,
        footprint_m=state["footprint_m"],  # type: ignore[arg-type]
        height_m=state["height_m"],  # type: ignore[arg-type]
    )


def _properties_from_spec(spec: BuildingSpec) -> dict[str, Any]:
    """Full, lossless property mapping for one massing spec."""
    return {
        "footprint_m": spec.footprint_m,
        "height_m": spec.height_m,
    }


def _specs_from_command(
    command: EditCommand, registry: AdapterRegistry
) -> tuple[BuildingSpec | None, BuildingSpec | None]:
    """Validate states and map one command to ``(old_spec, new_spec)``."""
    operation = command.operation
    if operation not in _OPERATIONS:
        if operation == "raster_patch":
            # Registry-declared but refused (met adapter's preset/reset
            # precedent): no Building_DSM raster-mutation path exists.
            raise EditStateError(
                "operation 'raster_patch' is not implemented: the typed "
                "delta family carries object states (footprint + height), "
                "not raster cells, and no incremental Building_DSM "
                "raster-mutation path exists — inventing a patch shape "
                "the executor cannot consume is refused rather than "
                "guessed (massing ops: add/move/update/delete)"
            )
        # Undeclared operations are reported by the registry check in
        # validate(); this guard keeps the shape checks honest for custom
        # registries.
        raise EditStateError(
            f"operation must be one of {sorted(_OPERATIONS)}, got "
            f"{operation!r}"
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
    # BOTH states, then the adapter's own field-level schema.
    for label, state in (("old_state", old_state), ("new_state", new_state)):
        if state is None:
            continue
        registry.validate_edit_state(ADAPTER_ID, dict(state))
        _check_known_fields(state, label)

    old_spec: BuildingSpec | None = None
    new_spec: BuildingSpec | None = None
    if old_state is not None:
        old_spec = _spec_from_state(
            _building_id_from(old_state, "old_state"), old_state, "old_state"
        )
    if new_state is not None:
        new_id = _building_id_from(new_state, "new_state")
        if old_spec is not None and new_id != old_spec.building_id:
            raise EditStateError(
                f"old_state and new_state describe different buildings "
                f"({old_spec.building_id!r} vs {new_id!r}); a massing edit "
                "revalues one block (move the footprint vertices instead)"
            )
        if old_state is not None:
            merged: dict[str, Any] = dict(old_state)
            merged.update(new_state)
        else:
            merged = dict(new_state)
        new_spec = _spec_from_state(new_id, merged, "new_state")
    if old_spec == new_spec:
        # U-D item (d): an identical-value resubmit (declared old_state ==
        # merged new_state) must refuse naming the property and both
        # values; without the refusal the edit churns the massing
        # regeneration chain for a physically identical scene.
        properties = (
            _properties_from_spec(old_spec) if old_spec is not None else {}
        )
        raise EditStateError(
            "edit is a no-op: building "
            f"{(new_spec or old_spec).building_id!r} already carries the "
            "requested state — every declared property is value-equal "
            "between old_state and new_state ("
            + identical_value_clause(
                (name, properties[name], properties[name])
                for name in ("footprint_m", "height_m")
                if name in properties
            )
            + "); an identical-value resubmit would churn dirty windows "
            "for a physically identical scene (add+delete cancellation "
            "across separate commands is the planner's job, not a valid "
            "single edit)"
        )
    return old_spec, new_spec


# ---------------------------------------------------------------------------
# Committed deltas -> validated massing edits (the U-C executor seam)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MassingEdit:
    """One committed massing change as re-validated ``(before, after)`` specs.

    ``None`` marks a not-yet-existing (add) or removed (delete) block.
    Construction re-validates both specs, so the executor's inputs are
    schema-checked on the seam, not only inside adapter ``validate()``
    calls (the model-parameter adapter's ``kernel_arguments_from_deltas``
    discipline).
    """

    building_id: str
    before: BuildingSpec | None
    after: BuildingSpec | None

    def __post_init__(self) -> None:
        if not isinstance(self.building_id, str) or not self.building_id:
            raise EditStateError("building_id must be a non-empty string")
        if self.before is None and self.after is None:
            raise EditStateError(
                "a massing edit needs a before or an after spec"
            )
        for role, spec in (("before", self.before), ("after", self.after)):
            if spec is not None and spec.building_id != self.building_id:
                raise EditStateError(
                    f"{role} spec id {spec.building_id!r} does not match "
                    f"edit id {self.building_id!r}"
                )


def _spec_from_frozen(
    building_id: str,
    properties: Mapping[str, Any] | None,
    label: str,
) -> BuildingSpec | None:
    """Rebuild one spec from a delta's frozen mapping (staging door + seam).

    The full property set is required — deltas staged by this adapter
    always carry it — so a hand-built partial payload is refused here
    rather than half-interpreted.
    """
    if properties is None:
        return None
    return _spec_from_state(building_id, properties, label)


def massing_edits_from_deltas(
    deltas: Sequence[BuildingMassingDelta],
) -> tuple[MassingEdit, ...]:
    """Fold committed building deltas into re-validated :class:`MassingEdit` records.

    One record per object change, in arrival order. Every before/after
    mapping is rebuilt into a :class:`BuildingSpec` — the full domain and
    finiteness schema runs again on this seam — so the rasterizer (U-C
    step 1 in the module friction list) receives self-certifying inputs:
    a bypass-built delta with a non-finite or out-of-domain payload fails
    here, never inside the raster write.
    """
    edits: list[MassingEdit] = []
    for delta in deltas:
        if not isinstance(delta, BuildingMassingDelta):
            raise SourceDeltaError(
                f"expected BuildingMassingDelta, got {type(delta).__name__}"
            )
        if delta.adapter_id != ADAPTER_ID:
            raise SourceDeltaError(
                f"delta belongs to adapter {delta.adapter_id!r}, not "
                f"{ADAPTER_ID!r}"
            )
        if delta.source_node_id != BUILDING_SOURCE_NODE:
            raise SourceDeltaError(
                f"delta targets source node {delta.source_node_id!r}, not "
                f"{BUILDING_SOURCE_NODE!r}"
            )
        for change in delta.objects:
            edits.append(
                MassingEdit(
                    building_id=change.object_id,
                    before=_spec_from_frozen(
                        change.object_id, change.before, "before-state"
                    ),
                    after=_spec_from_frozen(
                        change.object_id, change.after, "after-state"
                    ),
                )
            )
    return tuple(edits)


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------


class BuildingMassingAdapter(EditAdapter):
    """EditAdapter for building massing (registry id above).

    A stateless service: the registry, dependency graph, and safety policy
    are injected at construction, and every protocol method derives its
    result from the command and the
    :class:`~solweig_gpu.incremental.edit_types.SiteContext`. All outputs
    are frozen records, so identical inputs reproduce bitwise-identical
    deltas and plans. The adapter holds no site state and no building
    inventory: the scenario's current blocks live in the scenario store,
    reached only by the executor.
    """

    adapter_id: str = ADAPTER_ID
    schema_version: int = ADAPTER_SCHEMA_VERSION

    def __init__(
        self,
        *,
        registry: AdapterRegistry | None = None,
        graph: EditGraph | None = None,
        policy: SafetyPolicy | None = None,
    ) -> None:
        self._registry = registry if registry is not None else builtin_registry()
        self._graph = graph if graph is not None else default_edit_graph()
        self._policy = policy
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
        self._registry.validate_operation(self.adapter_id, command.operation)
        old_spec, new_spec = _specs_from_command(command, self._registry)
        building_id = (new_spec or old_spec).building_id
        # windows deliberately empty: a massing change's spatial effect is
        # site-wide and entirely downstream (full_only_initially) —
        # reporting a footprint window here would be a fake local scope.
        delta = BuildingMassingDelta(
            source_node_id=BUILDING_SOURCE_NODE,
            adapter_id=self.adapter_id,
            objects=(
                ObjectStateChange(
                    object_id=building_id,
                    before=(
                        _properties_from_spec(old_spec) if old_spec else None
                    ),
                    after=_properties_from_spec(new_spec) if new_spec else None,
                ),
            ),
            windows=(),
        )
        return ValidatedEdit(
            command=command,
            adapter_id=self.adapter_id,
            schema_version=self.schema_version,
            source_node_id=BUILDING_SOURCE_NODE,
            delta=delta,
        )

    # -- protocol: delta, plan, apply, preview, fixtures -----------------------

    def source_delta(
        self, edit: ValidatedEdit, context: SiteContext
    ) -> BuildingMassingDelta:
        """Return the delta produced at validation (single derivation)."""
        return self._require_own_delta(edit)

    def impact_plan(
        self, edit: ValidatedEdit, context: SiteContext
    ) -> ImpactPlan:
        """The single-edit impact plan, delegated to the core planner.

        The dirty closure, spatial/temporal scopes, and estimate formulas
        are the planner's (and therefore the engine's): the graph's
        ``building_dsm ->`` edges make the closure ``relative_geometry ->
        vegetation_visibility``, ``walls``, ``wall_aspect``,
        ``building_visibility -> svf -> time_shadow -> radiation ->
        surface_thermal_state -> tmrt -> utci`` (plus ``wbgt``, pruned as
        never-planned), the registry's ``full_only_initially`` status
        makes the source — and therefore every one of those stages — FULL
        with no windows and no transport bounding box, and every
        geometry/vision-dependent cache is invalidated (nothing downstream
        of ``building_dsm`` stays reusable; ``solar_atmospheric_state``
        and the unedited sources do). The returned plan equals the core
        :meth:`EditPlanner.plan` output for ``[edit]`` bitwise —
        including the engine's hand-built H2 path, which carries a window
        in its delta that the planner never reads.
        """
        self._require_own_delta(edit)
        state = SceneGraphState(self._graph, context.scene_revision, {})
        planner = EditPlanner(
            grid=context.grid,
            graph=self._graph,
            registry=self._registry,
            policy=self._policy,
        )
        return planner.plan([edit], state)

    def apply_source_delta(
        self, delta: SourceDelta, transaction: ScenarioTransaction
    ) -> None:
        """Stage the typed delta in a rollback-safe transaction.

        Staging never touches baseline state: the transaction either
        commits (returning the staged deltas, which the U-C executor folds
        via :func:`massing_edits_from_deltas` into the rasterizer's
        inputs) or rolls back cleanly. The registry has no building fence,
        so THIS door is the payload's only schema gate (conservative
        choice 6): every before/after mapping is rebuilt into a
        :class:`BuildingSpec` — full property set, domains, finiteness —
        before anything can be staged. A hand-built delta with a
        non-finite height, a two-vertex footprint, or a partial property
        set is refused here, pre-stage.
        """
        if not isinstance(delta, BuildingMassingDelta):
            raise SourceDeltaError(
                f"expected BuildingMassingDelta, got {type(delta).__name__}"
            )
        if delta.adapter_id != self.adapter_id:
            raise SourceDeltaError(
                f"delta belongs to adapter {delta.adapter_id!r}, not "
                f"{self.adapter_id!r}"
            )
        if delta.source_node_id != BUILDING_SOURCE_NODE:
            raise SourceDeltaError(
                f"delta targets source node {delta.source_node_id!r}, not "
                f"{BUILDING_SOURCE_NODE!r}"
            )
        # Staging door (model-parameter MEDIUM remediation): re-validate
        # the payload, then stage. Failure leaves the transaction empty.
        massing_edits_from_deltas([delta])
        transaction.stage(delta)

    def preview_descriptor(
        self, edit: ValidatedEdit, context: SiteContext
    ) -> PreviewDescriptor:
        """Massing/shadow preview with mandatory not-exact caveats.

        The registry kind is ``massing_and_shadow``: the preview renders
        the block and an approximate cast shadow, and is explicitly NOT
        the scientific output — the full-tile regeneration demand is the
        plan's, and the values come from the solver job.
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

    def _require_own_delta(self, edit: ValidatedEdit) -> BuildingMassingDelta:
        if not isinstance(edit.delta, BuildingMassingDelta):
            raise SourceDeltaError(
                "edit must carry a BuildingMassingDelta, got "
                f"{type(edit.delta).__name__}"
            )
        if edit.delta.adapter_id != self.adapter_id:
            raise SourceDeltaError(
                f"edit delta belongs to adapter {edit.delta.adapter_id!r}"
            )
        return edit.delta


def register_building_massing_adapter(
    registry: AdapterRegistry | None = None,
) -> AdapterRegistry:
    """Register this adapter (idempotent) and return the registry.

    Kept separate from the earlier adapters' helpers so they stay
    untouched; ``adapters.register_default_adapters`` composes all of them.
    """
    target = registry if registry is not None else builtin_registry()
    target.register(BuildingMassingAdapter(registry=target))
    return target
