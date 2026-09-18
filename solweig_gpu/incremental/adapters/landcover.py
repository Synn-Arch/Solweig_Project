# SPDX-License-Identifier: GPL-3.0-only
"""Land-cover surface edit adapter (``landcover_surface``).

This adapter implements the edit-adapter contract
(``docs/incremental_design_tool/universal_editing/edit_adapter_contract.md``)
for the ``landcover_surface`` registry entry. Land cover is a *region-local*
class grid: painting replaces class codes inside a rectangular window and
nothing else. The adapter wraps no physics — the class grid's effect on the
model is entirely downstream (surface-property maps at
``utci_process.py:608-641`` via :func:`solweig_gpu.Tgmaps_v1.Tgmaps_v1`,
the GVF walk at ``solweig.py:2124-2133``, and the ground-heat accumulator
via ``Tgmap1``/TsWaveDelay ``solweig.py:2155-2160``) — so the adapter stages
typed :class:`~solweig_gpu.incremental.edit_types.LandCoverPaintDelta`
records and delegates every planning decision to the core
:class:`~solweig_gpu.incremental.planner.EditPlanner`:

- **validate** maps the user state (a half-open ``window`` plus ``classes``)
  through
  :meth:`solweig_gpu.incremental.edit_registry.AdapterRegistry.validate_edit_state`
  first, so the water/invalid-class rulings are the *registry's*, never
  re-implemented here. On top of the registry schema the adapter enforces
  the window grammar (integer, non-negative, in-grid, non-empty) and the
  class-payload grammar (one code broadcast, or a row-major sequence of
  exactly ``window.area`` codes).
- **source_delta** returns the delta built during validation (single
  validation pass); ``spatial_windows`` are the patch windows themselves —
  unlike vegetation there is no influence corridor to compute: a paint's
  write scope *is* its footprint (see "Spatial semantics" below).
- **impact_plan** delegates to the core planner for the single-edit batch,
  so adapter plans and engine plans are bitwise-identical by construction.
- **apply_source_delta** re-validates the full paint payload through the
  same class fence before ``transaction.stage`` (the staging door; the
  model-parameter adapter's reviewer finding, now the wave standard) and
  stages the typed delta in a rollback-safe transaction.
  :func:`landcover_overlay_from_deltas` is the executor seam — it folds
  committed deltas into the copy-on-write :class:`LandCoverOverlay` the
  U-C integration resolves against the baseline class grid (see the
  friction list below).

The water fence (lead ruling 2026-09-02)
----------------------------------------

Water (class 7) is rejected on EVERY path, and so is every other
non-paintable code:

1. **Command states, both roles.**
   :meth:`AdapterRegistry.validate_edit_state` rejects ``classes``
   carrying 7 (named out: the oracle's water physics tests ``lc_grid == 3``
   — ``solweig.py:130`` and ``:2214-2215`` — a branch that is dead after
   the load-time cleaning, so painting 7 would deliver grass-class physics
   while claiming water) and any code outside ``valid_classes`` (0/3/4 are
   silently remapped at load — ``<1 | >7`` to 6, ``3|4`` to 5,
   ``utci_process.py:1029-1041`` — so painting them would misrepresent the
   result; 8+ and negatives are not classes at all).
2. **Delta payloads.** The planner calls
   :meth:`AdapterRegistry.validate_delta` for every edit in a batch, which
   fences each patch's ``after_classes`` (negative codes are the data
   model's "unspecified/never painted" markers and are skipped): a
   bypass-built delta can never plan.
3. **The staging door.** :meth:`LandCoverSurfaceAdapter.apply_source_delta`
   re-runs the same fence on the payload before ``transaction.stage``.
4. **The executor seam.** :class:`LandCoverOverlay` re-runs it on
   construction, so the object handed toward the solver is
   self-certifying. Each layer exists because the previous one can be
   bypassed by a hand-built record.

Spatial semantics (what recomputes, from the graph — not guessed)
-----------------------------------------------------------------

The dependency graph has exactly two edges out of ``landcover``:
``[landcover, radiation]`` and ``[landcover, surface_thermal_state]``
(dependency_graph.yaml; ``utci_process.py:608-641`` for the property grids
that justify the radiation edge, ``solweig.py:2155-2160`` for the
ground-heat accumulation that justifies the stateful edge). The planner's
closure is therefore ``radiation -> surface_thermal_state -> tmrt -> utci``
(``wbgt`` pruned as never planned), and every geometry/view-factor node
stays reusable — ``walls`` and ``wall_aspect`` derive from
``building_dsm`` only, ``building_visibility``/``vegetation_visibility``
from geometry sources only, ``svf`` from the visibilities, and
``time_shadow`` from ``solar_atmospheric_state`` + visibilities + walls
family: none of those chains contains ``landcover``, so the cached SVF
bundle, walls, and shadows are reusable for a paint (VERIFIED against the
YAML edge list; the engine test
``test_landcover_edit_skips_visibility_and_svf_nodes`` pins the same split).

Write scope: the registry entry declares ``nominal_spatial_scope:
windows`` and the patch window is the delta's ``spatial_windows``, so the
planner resolves the whole landcover-driven chain to WINDOWS over the
(clamped, merged) paint footprint. Read scope: land-cover enters the GVF
walk (``gvf_2018a`` consumes ``alb_grid``/``emis_grid``/``Tg`` up to the
walk reach ``second`` pixels, ``solweig.py:2124-2133``), so receptors just
outside the footprint still SEE painted cells; that is a READ halo, not a
write expansion — exactly the vegetation adapter's
``read_halo_pixels`` treatment (default 0 keeps adapter plans
bitwise-equal to engine plans; the U-C executor widens read windows via
:func:`solweig_gpu.incremental.solver.required_halo_pixels`). The stateful
``surface_thermal_state`` always replays from timestep 0 (planner's
``STATEFUL_NODES`` rule; ``Tgmap1`` accumulation cannot be summarized).

Temporal semantics (registry ``temporal_scope: replay_if_stateful``)
--------------------------------------------------------------------

A paint is time-invariant as an input but stateful in effect: the changed
surface properties feed every timestep's radiation and the ground-heat
accumulator, so the time-varying stages recompute at the requested times
(``TemporalScope.ONE``/``RANGE`` from ``requested_times``, else ``ALL``)
while ``surface_thermal_state`` replays from timestep 0 to the requested
bound — never a guessed sub-range (planner conservative choices 4-6).

Operations
----------

- ``paint`` is implemented: one command = one paint stroke = one
  :class:`~solweig_gpu.incremental.edit_types.LandCoverPaintPatch`.
- ``polygon_assign`` is REFUSED LOUDLY (documented deviation, mirroring
  the met adapter's ``preset`` refusal): no polygon rasterization
  semantics exist anywhere in the incremental stack (the tree machinery
  rasterizes discs only; shapely appears in preprocessing-side scripts
  with their own conventions). Choosing a fill rule (pixel-centre vs
  any-part), a coordinate convention (pixel vs world), and hole handling
  would silently define mask edges for scientific output — refused rather
  than guessed. The registry entry still declares the operation; clients
  get an explicit error, never a silent accept.
- ``reset`` is REFUSED LOUDLY (mirroring the met adapter's ``reset``
  refusal): baseline restore needs the site's baseline class grid, which
  no :class:`~solweig_gpu.incremental.edit_types.SiteContext` carries.
  The executor implements reset as overlay-drop instead — staging a
  "back to unknown values" paint is refused, never guessed.

Executor seam and U-C integration friction
------------------------------------------

How painted classes reach the physics (consumption sites, with lines):

1. ``solver.py:1099-1108`` — ``solve_window`` reads the BASELINE
   ``cache.window("landcover", read_window)`` and converts it to a float32
   tensor. The scenario overlay must materialize between the cache read
   and the tensor conversion (the solver has no overlay seam for land
   cover yet; the forcing overlay's ``load_site_forcing(overlay=...)``
   pattern is the template).
2. ``utci_process.py:608-641`` — ``run_utci_window`` re-cleans the grid
   (``<1 | >7`` -> 6, ``3|4`` -> 5) and calls
   :func:`solweig_gpu.Tgmaps_v1.Tgmaps_v1` (``Tgmaps_v1.py:15-53``) to
   build ``TgK/Tstart/alb_grid/emis_grid/TmaxLST(+wall)``. Painted classes
   {1, 2, 5, 6} survive this cleaning unchanged — it is exactly why 0/3/4
   are not paintable (silent remap) and why the fence vocabulary is the
   registry's ``valid_classes``.
3. ``solweig.py:2131-2133`` — the GVF walk consumes the property grids;
   its read reach is the halo concern above.
4. ``solweig.py:130`` / ``:2214-2215`` — the ``lc_grid == 3`` water
   branches (dead after cleaning): the water-fence evidence.
5. ``solver.py:1213-1216`` — ``run_full_tile`` passes the baseline
   ``landcover_path`` into the scratch site; a scenario overlay must also
   materialize the resolved grid there or the full fallback silently
   ignores the paint (same shape as the forcing overlay's full-fallback
   friction).
6. State replay: painted surfaces change ``Tgmap1`` history
   (``TsWaveDelay_2015a``, ``solweig.py:2155-2160``), so replay from
   timestep 0 is mandatory — the planner's STATEFUL rule already forbids
   "optimizing" it away.

Conservative choices (contract ambiguities resolved the safe way):

1. **The registry fence runs first on BOTH command states; the adapter
   never re-implements it.** Unknown state fields (including case
   variants like ``Window``/``Classes``) are rejected with the closed
   vocabulary — a paint state is exactly ``{'window', 'classes'}``.
2. **Windows are user-supplied, so out-of-grid is a refusal, not a
   clamp.** The paint window must be integer, non-negative, non-empty,
   and inside the context grid; a partially off-grid stroke is a user
   error (adapter-computed influence windows are clamped instead —
   different contract, vegetation's choice 8).
3. **One spelling per domain.** Window bounds and class codes must be
   real integers (``int``, never ``bool`` and never ``float`` — ``2.0``
   is not a class). There are no float-valued fields in this family, so
   no NaN/Infinity can ride the payload at all (the classes are ints by
   :class:`LandCoverPaintPatch` construction); this is stated here
   because the sibling adapters needed explicit ``math.isfinite`` checks
   for their float domains.
4. **Bounds bind the new value only.** ``old_state`` is the caller's
   *claim* about the current classes: it is shape- and fence-checked (the
   registry rejects water/invalid codes in either role — a claimed
   history is still a command state) but never compared against site
   truth; undeclared before-cells are recorded as ``-1`` (the data
   model's unspecified marker) for the executor to resolve from the
   scenario overlay store.
5. **Declared identity paints are refused.** A paint whose declared
   before equals its after everywhere is an explicit
   :class:`~solweig_gpu.incremental.edit_types.EditStateError`; without a
   declared before there is nothing to compare and the edit stands (the
   executor resolves the true before at apply time).
6. **``after_classes`` never carries negative codes from user state.**
   The user surface paints only real classes; ``-1`` markers exist at the
   delta/overlay layer for partial coverage built programmatically (the
   data model's contract), and the fence skips them — they never paint.
7. **Stale revisions are refused early.** ``validate`` requires
   ``command.base_scene_revision == context.scene_revision`` (UEDIT-006);
   the planner re-checks against the authoritative scene state.
8. **The overlay never mutates its input.** :meth:`LandCoverOverlay.resolve`
   writes on a fresh copy, so cache-backed (memmap) and loader-backed
   arrays stay bitwise unchanged no matter how often an overlay resolves.
9. **The registry facts self-check at import.** The builtin entry must
   own the ``landcover`` source, carry exactly the ruling's valid/fenced
   class sets, declare exactly the three operations, and be windowed (a
   future registry edit that forgets this adapter fails at import time
   rather than at a user's edit).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np

from ..edit_graph import EditGraph, SceneGraphState, default_edit_graph
from ..edit_registry import (
    LANDCOVER_FENCED_CLASSES,
    LANDCOVER_SOURCE_NODE,
    LANDCOVER_VALID_CLASSES,
    AdapterRegistry,
    EditAdapter,
    builtin_adapter_metadata,
    builtin_registry,
)
from ..edit_types import (
    EditCommand,
    EditStateError,
    ImpactPlan,
    LandCoverPaintDelta,
    LandCoverPaintPatch,
    NodeImpact,
    PreviewDescriptor,
    ScenarioTransaction,
    SiteContext,
    SourceDelta,
    SourceDeltaError,
    SpatialScope,
    ValidatedEdit,
)
from ..geometry import RasterGrid, RasterWindow
from ..planner import EditPlanner, SafetyPolicy

__all__ = [
    "ADAPTER_ID",
    "ADAPTER_SCHEMA_VERSION",
    "FENCED_CLASSES",
    "PAINTABLE_CLASSES",
    "LandCoverOverlay",
    "LandCoverSurfaceAdapter",
    "coalesce_overlay",
    "landcover_overlay_from_deltas",
    "overlay_diff_windows",
    "register_landcover_surface_adapter",
]

#: Registry id this adapter implements (must match the builtin metadata).
ADAPTER_ID = "landcover_surface"

#: Adapter property-schema version; bump on any user-state schema change
#: (the contract: an adapter schema change invalidates incompatible edit
#: events and cached source overlays).
ADAPTER_SCHEMA_VERSION = 1

#: Registry-declared operations (mirrored for honest error messages).
_OPERATIONS = frozenset({"paint", "polygon_assign", "reset"})

#: The paint vocabulary, from the binding registry ruling (lead ruling
#: 2026-09-02): asphalt (1), roofs (2), grass (5), bare soil (6).
PAINTABLE_CLASSES = frozenset(LANDCOVER_VALID_CLASSES)

#: Water (7): fenced — the oracle tests ``lc_grid == 3``, never 7, so the
#: Twater branch is dead upstream and water semantics cannot be delivered.
FENCED_CLASSES = frozenset(LANDCOVER_FENCED_CLASSES)

#: The data model's unspecified/never-painted marker (edit_types.py).
UNSPECIFIED_CODE = -1

_WINDOW_FIELDS = ("row_start", "row_stop", "col_start", "col_stop")
_STATE_FIELDS = frozenset({"window", "classes"})

_PREVIEW_LIMITATIONS = (
    "Preview recolours the painted cells by class only; it is interaction "
    "feedback, not scientific output (preview_is_not_exact)",
    "Water (class 7) is not paintable in v1: the oracle never applies "
    "Twater to painted water (lc_grid == 3 dead branch), so the adapter "
    "refuses it rather than claiming semantics it cannot deliver (lead "
    "ruling 2026-09-02)",
    "Classes 0/3/4 are not paintable: load-time cleaning remaps them "
    "(<1|>7 to bare soil, 3/4 to grass; utci_process.py:1029-1041), so a "
    "paint would silently deliver a different class",
    "surface_thermal_state replays from timestep 0: ground-heat "
    "accumulation cannot be summarized, so an earlier timestep is never "
    "skipped silently",
    "Geometry and view-factor caches stay reusable: a paint never "
    "invalidates walls, wall_aspect, visibilities, svf, or time_shadow",
)


def _check_registry_facts() -> None:
    """Drift guard: the builtin entry carries the ruling's facts verbatim.

    A registry edit that changes the land-cover source node, the class
    fence, the operation set, or the windowed nominal scope must update
    this adapter in the same commit — failing at import beats failing at
    a user's edit.
    """
    matches = [
        entry
        for entry in builtin_adapter_metadata()
        if entry.id == ADAPTER_ID
    ]
    if len(matches) != 1:
        raise EditStateError(
            f"the builtin registry must declare {ADAPTER_ID!r} exactly once; "
            f"found {len(matches)}"
        )
    entry = matches[0]
    if tuple(entry.source_nodes) != (LANDCOVER_SOURCE_NODE,):
        raise EditStateError(
            f"{ADAPTER_ID!r} must own source node {LANDCOVER_SOURCE_NODE!r}, "
            f"got {tuple(entry.source_nodes)}"
        )
    if frozenset(entry.valid_classes) != PAINTABLE_CLASSES:
        raise EditStateError(
            f"{ADAPTER_ID!r} valid classes must be exactly "
            f"{sorted(PAINTABLE_CLASSES)}, got {sorted(entry.valid_classes)}"
        )
    if PAINTABLE_CLASSES & FENCED_CLASSES:
        raise EditStateError(
            "the paintable and fenced class sets overlap: "
            f"{sorted(PAINTABLE_CLASSES & FENCED_CLASSES)}"
        )
    if frozenset(entry.operations) != _OPERATIONS:
        raise EditStateError(
            f"{ADAPTER_ID!r} must declare exactly {sorted(_OPERATIONS)}, "
            f"got {sorted(entry.operations)}"
        )
    if entry.is_full_spatial:
        raise EditStateError(
            f"{ADAPTER_ID!r} nominal spatial scope must be windowed "
            "(the paint footprint is the write scope), got "
            f"{entry.nominal_spatial_scope!r}"
        )


_check_registry_facts()


# ---------------------------------------------------------------------------
# The paint-payload fence (shared by the staging door and the overlay)
# ---------------------------------------------------------------------------


def _check_paintable(
    codes: Sequence[int], label: str
) -> None:
    """Fence one after-classes payload exactly like ``validate_delta``.

    Non-negative codes must be paintable (water named out with the ruling
    text); negative codes are the data model's unspecified markers and are
    skipped — they never paint. ``SourceDeltaError`` (not
    ``EditStateError``) because every caller is an executor-side seam.
    """
    water = sorted(
        {code for code in codes if code in FENCED_CLASSES}
    )
    if water:
        raise SourceDeltaError(
            f"{label} carries fenced class 7 (water): the oracle never "
            "applies Twater to class 7 (lc_grid == 3 dead branch), so "
            "water semantics cannot be delivered (lead ruling 2026-09-02)"
        )
    invalid = sorted(
        {
            code
            for code in codes
            if code >= 0 and code not in PAINTABLE_CLASSES
        }
    )
    if invalid:
        raise SourceDeltaError(
            f"{label} carries invalid classes {invalid}; paintable: "
            f"{sorted(PAINTABLE_CLASSES)} (0/3/4 are remapped at load, "
            "utci_process.py:1029-1041)"
        )


def _check_patch_window(patch: LandCoverPaintPatch, label: str) -> None:
    """Structural window sanity at the executor seams (no grid available).

    Indices must be non-negative and the window non-empty; a hand-built
    patch with off-grid indices is refused here, and grid fit is checked
    against the real baseline shape at :meth:`LandCoverOverlay.resolve`.
    """
    window = patch.window
    if (
        window.row_start < 0
        or window.col_start < 0
        or window.row_stop < 0
        or window.col_stop < 0
    ):
        raise SourceDeltaError(
            f"{label} window {window} carries negative raster indices"
        )
    if window.is_empty:
        raise SourceDeltaError(f"{label} window {window} is empty")


def _check_paint_payload(delta: LandCoverPaintDelta) -> None:
    """Re-validate a whole delta payload (fence + window sanity)."""
    for index, patch in enumerate(delta.patches):
        label = f"patch {index}"
        _check_patch_window(patch, label)
        _check_paintable(patch.after_classes, f"{label} after_classes")


# ---------------------------------------------------------------------------
# Command-state parsing
# ---------------------------------------------------------------------------


def _window_from_state(
    state: Mapping[str, Any], label: str, grid: RasterGrid
) -> RasterWindow:
    """Parse and bound-check one state's ``window`` (user-supplied)."""
    spec = state.get("window")
    if not isinstance(spec, Mapping):
        raise EditStateError(
            f"{label} requires a 'window' mapping with fields "
            f"{list(_WINDOW_FIELDS)} (half-open, raster indices)"
        )
    unknown = sorted(set(spec) - set(_WINDOW_FIELDS))
    if unknown:
        raise EditStateError(
            f"{label}.window carries unknown fields {unknown}; the window "
            f"grammar is exactly {list(_WINDOW_FIELDS)}"
        )
    missing = [field for field in _WINDOW_FIELDS if field not in spec]
    if missing:
        raise EditStateError(
            f"{label}.window is missing fields {missing}"
        )
    values: dict[str, int] = {}
    for field in _WINDOW_FIELDS:
        value = spec[field]
        if isinstance(value, bool) or not isinstance(value, int):
            raise EditStateError(
                f"{label}.window.{field} must be an integer raster index, "
                f"got {type(value).__name__}"
            )
        if value < 0:
            raise EditStateError(
                f"{label}.window.{field} must be non-negative, got {value}"
            )
        values[field] = value
    if values["row_stop"] < values["row_start"]:
        raise EditStateError(
            f"{label}.window row_stop ({values['row_stop']}) must be >= "
            f"row_start ({values['row_start']}) (half-open bounds)"
        )
    if values["col_stop"] < values["col_start"]:
        raise EditStateError(
            f"{label}.window col_stop ({values['col_stop']}) must be >= "
            f"col_start ({values['col_start']}) (half-open bounds)"
        )
    window = RasterWindow(
        values["row_start"],
        values["row_stop"],
        values["col_start"],
        values["col_stop"],
    )
    if window.is_empty:
        raise EditStateError(
            f"{label}.window {window} is empty; a paint stroke needs at "
            "least one cell"
        )
    if window.row_stop > grid.rows or window.col_stop > grid.cols:
        raise EditStateError(
            f"{label}.window {window} is outside the site grid "
            f"({grid.rows}x{grid.cols}); a user-supplied paint stroke is "
            "refused rather than silently clamped"
        )
    return window


def _classes_from_state(
    state: Mapping[str, Any], label: str, window: RasterWindow
) -> tuple[int, ...]:
    """Expand one state's ``classes`` into a row-major ``window.area`` tuple.

    A single code broadcasts over the window (fixture ``class_replace``); a
    sequence must carry exactly ``window.area`` codes (fixture
    ``mixed_mask``). The registry fence has already rejected water and
    invalid codes; this parses shape and integer spelling only.
    """
    raw = state.get("classes")
    if raw is None:
        raise EditStateError(
            f"{label} requires a 'classes' field (a paintable class code or "
            "a row-major sequence of them)"
        )
    area = window.area
    if isinstance(raw, bool) or not isinstance(raw, int):
        if not isinstance(raw, (list, tuple)):
            raise EditStateError(
                f"{label}.classes must be a class code or a sequence of "
                f"class codes, got {type(raw).__name__}"
            )
        if len(raw) != area:
            raise EditStateError(
                f"{label}.classes carries {len(raw)} codes but the window "
                f"holds {area} cells (row-major flattening)"
            )
        codes: list[int] = []
        for code in raw:
            if isinstance(code, bool) or not isinstance(code, int):
                raise EditStateError(
                    f"{label}.classes codes must be integers, got "
                    f"{type(code).__name__} ({code!r})"
                )
            codes.append(code)
        return tuple(codes)
    return (raw,) * area


def _check_state_fields(state: Mapping[str, Any], label: str) -> None:
    unknown = sorted(set(state) - _STATE_FIELDS)
    if unknown:
        raise EditStateError(
            f"{label} carries unknown fields {unknown}; a paint state is "
            "exactly {'window', 'classes'}"
        )


def _patches_from_command(
    command: EditCommand, registry: AdapterRegistry, grid: RasterGrid
) -> tuple[LandCoverPaintPatch, ...]:
    """Validate one paint command and build its canonical patch tuple."""
    operation = command.operation
    if operation not in _OPERATIONS:
        # The registry check below reports undeclared operations; this guard
        # keeps the shape checks honest for custom registries.
        raise EditStateError(
            f"operation must be one of {sorted(_OPERATIONS)}, got "
            f"{operation!r}"
        )
    if operation == "polygon_assign":
        # Documented deviation, mirroring the met adapter's preset refusal.
        raise EditStateError(
            "operation 'polygon_assign' is not implemented: no polygon "
            "rasterization semantics exist in the incremental stack (the "
            "tree machinery rasterizes discs only), and choosing a fill "
            "rule, coordinate convention, and hole handling would silently "
            "define mask edges for scientific output — refused rather than "
            "guessed (paint into a window instead)"
        )
    if operation == "reset":
        # Documented deviation, mirroring the met adapter's reset refusal.
        raise EditStateError(
            "operation 'reset' is not implemented as a delta: baseline "
            "restore needs the site's baseline class grid, which no "
            "SiteContext carries — the executor drops the scenario overlay "
            "instead (refused rather than guessed)"
        )

    old_state = command.old_state
    new_state = command.new_state
    # Registry fence first (water/invalid classes) on BOTH states, then the
    # adapter's own grammar checks.
    for label, state in (("old_state", old_state), ("new_state", new_state)):
        if state is None:
            continue
        _check_state_fields(state, label)
        registry.validate_edit_state(ADAPTER_ID, dict(state))

    if new_state is None:
        raise EditStateError("a 'paint' requires a new_state")

    window = _window_from_state(new_state, "new_state", grid)
    after = _classes_from_state(new_state, "new_state", window)

    if old_state is None:
        # Undeclared before: the data model's unspecified marker; the
        # executor resolves true before-values from the scenario overlay.
        before: tuple[int, ...] = (UNSPECIFIED_CODE,) * window.area
    else:
        old_window = _window_from_state(old_state, "old_state", grid)
        if old_window != window:
            raise EditStateError(
                "old_state and new_state select different windows "
                f"({old_window} vs {window}); a paint revalues one window"
            )
        before = _classes_from_state(old_state, "old_state", old_window)
        if before == after:
            # Whole-window interception only (u-c6b lead ruling): a paint
            # whose declared before-state equals its after-state on EVERY
            # cell is refused — naming the class property and both (equal)
            # value sets (U-D item d). Per-cell or partial interception is
            # deliberately NOT done here: the plan's promise is the STROKE
            # footprint, and a straddling stroke still executes over its
            # whole window (value-equal repaints of PUBLISHED state are
            # intercepted whole-batch one level up, in the executor).
            classes = sorted(set(before))
            raise EditStateError(
                "edit is a no-op: every cell of the declared before-state "
                "already carries the requested class (window "
                f"{window}: class before = {classes}, after = {classes}, "
                "value-equal on every cell); an identical-value resubmit "
                "would churn the paint recompute footprint for a "
                "physically identical scene"
            )
    return (
        LandCoverPaintPatch(
            window=window,
            before_classes=before,
            after_classes=after,
        ),
    )


# ---------------------------------------------------------------------------
# Committed deltas -> LandCoverOverlay (the U-C executor seam)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LandCoverOverlay:
    """Copy-on-write land-cover class overlay for one scenario.

    Built from committed :class:`LandCoverPaintDelta` records via
    :func:`landcover_overlay_from_deltas` (the executor seam) — never from
    raw user state. Construction re-runs the class fence on every patch,
    so the object handed toward the solver is self-certifying:

    - :meth:`resolve` returns a *new* array that equals the baseline class
      grid everywhere except the painted cells; the baseline itself is
      never written (hard invariant — it is typically a cache-backed
      memmap). Patches apply in arrival order, so later strokes win per
      cell (the coalescer's ordered last-write-wins semantics).
    - :meth:`changed_cell_mask` exposes exactly the cells the overlay
      paints (unspecified ``-1`` markers never paint), which is how a
      cross-check can narrow to the undeclared remainder.
    """

    patches: tuple[LandCoverPaintPatch, ...] = ()

    def __post_init__(self) -> None:
        for patch in self.patches:
            if not isinstance(patch, LandCoverPaintPatch):
                raise SourceDeltaError(
                    "overlay patches must be LandCoverPaintPatch records, "
                    f"got {type(patch).__name__}"
                )
        object.__setattr__(self, "patches", tuple(self.patches))
        # Full payload fence (classes + window sanity) on the seam.
        for index, patch in enumerate(self.patches):
            _check_patch_window(patch, f"overlay patch {index}")
            _check_paintable(
                patch.after_classes, f"overlay patch {index} after_classes"
            )

    # -- resolution ----------------------------------------------------------

    def resolve(self, baseline: np.ndarray) -> np.ndarray:
        """Return the overlay-resolved class grid; ``baseline`` is untouched.

        Copy-on-write: the resolved grid is a fresh copy of ``baseline``
        (same dtype), so caller arrays — including cache-backed memmaps —
        are bitwise unchanged no matter how often an overlay resolves.
        Unspecified ``-1`` after-codes leave their cells unchanged.
        """
        grid = np.asarray(baseline)
        if grid.ndim != 2:
            raise SourceDeltaError(
                f"baseline land-cover grid must be 2-D, got shape "
                f"{grid.shape}"
            )
        resolved = np.array(grid, copy=True)
        for index, patch in enumerate(self.patches):
            window = patch.window
            if (
                window.row_stop > grid.shape[0]
                or window.col_stop > grid.shape[1]
            ):
                raise SourceDeltaError(
                    f"overlay patch {index} window {window} is outside the "
                    f"baseline grid shape {grid.shape}"
                )
            after = np.asarray(
                patch.after_classes, dtype=np.int64
            ).reshape(window.height, window.width)
            target = resolved[
                window.row_start : window.row_stop,
                window.col_start : window.col_stop,
            ]
            painted = after >= 0  # unspecified markers never paint
            target[painted] = after[painted]
        return resolved

    def changed_cell_mask(self, shape: tuple[int, ...]) -> np.ndarray:
        """Boolean mask, True exactly at cells a patch paints."""
        mask = np.zeros(shape, dtype=bool)
        for patch in self.patches:
            window = patch.window
            after = np.asarray(
                patch.after_classes, dtype=np.int64
            ).reshape(window.height, window.width)
            target = mask[
                window.row_start : window.row_stop,
                window.col_start : window.col_stop,
            ]
            target |= after >= 0
        return mask


def landcover_overlay_from_deltas(
    deltas: Sequence[LandCoverPaintDelta],
) -> LandCoverOverlay:
    """Fold committed paint deltas into the solver's :class:`LandCoverOverlay`.

    Patches concatenate in arrival order (the coalescer's semantics:
    last-write-wins per cell at apply time), with every payload re-checked
    against the fence and window sanity — the executor's inputs are
    validated on this seam, not only inside adapter ``validate()`` calls.
    This is the hand-off the U-C integration resolves against the baseline
    class grid (solver.py:1099-1108 is the materialization site).
    """
    patches: list[LandCoverPaintPatch] = []
    for delta in deltas:
        if not isinstance(delta, LandCoverPaintDelta):
            raise SourceDeltaError(
                f"expected LandCoverPaintDelta, got {type(delta).__name__}"
            )
        if delta.adapter_id != ADAPTER_ID:
            raise SourceDeltaError(
                f"delta belongs to adapter {delta.adapter_id!r}, not "
                f"{ADAPTER_ID!r}"
            )
        if delta.source_node_id != LANDCOVER_SOURCE_NODE:
            raise SourceDeltaError(
                f"delta targets source node {delta.source_node_id!r}, not "
                f"{LANDCOVER_SOURCE_NODE!r}"
            )
        patches.extend(delta.patches)
    return LandCoverOverlay(patches=tuple(patches))


# ---------------------------------------------------------------------------
# Overlay coalescing (u-c4-review M3: bounded accumulated state)
# ---------------------------------------------------------------------------


def _mask_run_patches(
    mask: np.ndarray, values: np.ndarray | None
) -> tuple[LandCoverPaintPatch, ...]:
    """Decompose a painted-cell mask into maximal-rectangle paint patches.

    Per-row horizontal runs are merged vertically with the run above when
    the column span AND the painted values continue exactly, so a solid
    rectangle painted once (or repainted to one uniform class) comes back
    as ONE patch no matter how many strokes built it, while value-divergent
    regions split at the divergence. ``before_classes`` are the data
    model's unspecified markers: the canonical form describes the CURRENT
    paint, not the history that produced it. Output order is deterministic
    (sorted by ``(row_start, col_start)``), so two overlays with the same
    resolved grid coalesce to the same patch tuple — equality of canonical
    overlays is semantic equality.
    """
    rows, cols = mask.shape
    emitted: list[LandCoverPaintPatch] = []
    # (col_start, col_stop) -> [row_start, row_stop, values-tuple|None]
    open_runs: dict[tuple[int, int], list] = {}

    def _close(run: list) -> None:
        row_start, row_stop, col_start, col_stop = (
            int(run[0]), int(run[1]), int(run[2]), int(run[3]),
        )
        height = row_stop - row_start
        width = col_stop - col_start
        after = (
            (UNSPECIFIED_CODE,) * (height * width)
            if run[4] is None
            # A run's painted values are one row's slice that continues
            # unchanged down the whole rectangle (the merge condition), so
            # the flattened row-major codes are that row repeated.
            else tuple(int(v) for v in run[4]) * height
        )
        emitted.append(
            LandCoverPaintPatch(
                window=RasterWindow(row_start, row_stop, col_start, col_stop),
                before_classes=(UNSPECIFIED_CODE,) * (height * width),
                after_classes=after,
            )
        )

    for row in range(rows):
        row_mask = mask[row]
        starts = np.flatnonzero(
            np.diff(np.concatenate(([0], row_mask.view(np.int8), [0]))) == 1
        )
        stops = np.flatnonzero(
            np.diff(np.concatenate(([0], row_mask.view(np.int8), [0]))) == -1
        )
        carried: dict[tuple[int, int], list] = {}
        for start, stop in zip(starts, stops):
            col_start, col_stop = int(start), int(stop)
            values_run = (
                None
                if values is None
                else tuple(int(v) for v in values[row, col_start:col_stop])
            )
            run = open_runs.pop((col_start, col_stop), None)
            if run is not None and (values_run is None or run[4] == values_run):
                run[1] = row + 1  # extend the rectangle down one row
                if values_run is not None:
                    run[4] = values_run
                carried[(col_start, col_stop)] = run
            else:
                if run is not None:
                    _close(run)  # value divergence: the upper patch ends
                carried[(col_start, col_stop)] = [
                    row, row + 1, col_start, col_stop, values_run,
                ]
        for run in open_runs.values():
            _close(run)  # no continuation row: the patch ends
        open_runs = carried
    for run in open_runs.values():
        _close(run)
    emitted.sort(key=lambda p: (p.window.row_start, p.window.col_start))
    return tuple(emitted)


def coalesce_overlay(
    overlay: LandCoverOverlay | None,
    baseline: np.ndarray,
    *,
    resolved: np.ndarray | None = None,
) -> LandCoverOverlay | None:
    """Canonical per-cell last-write-wins fold of one overlay (u-c6).

    The u-c4-review M3 fix: an accumulated overlay used to concatenate one
    patch per committed stroke forever, so every paint-only job re-staged
    (and recomputed) every footprint the session ever painted — session-
    monotone cost. This fold resolves the overlay once against
    ``baseline`` (the ordered last-write-wins application, exactly what
    :meth:`LandCoverOverlay.resolve` feeds the physics) and re-emits the
    PAINTED cells — the union of every painted footprint — as a canonical
    rectangle decomposition carrying each cell's FINAL class. Properties:

    - EQUALITY of canonical overlays holds iff two overlays carry the
      SAME PAINTED SET and the SAME RESOLVED VALUES: the fold is the
      canonical paint STATE, not the resolved grid alone. Two overlays
      that merely RESOLVE equal can fold DIFFERENT — painting a cell to
      its baseline value keeps it in the painted set (at the baseline
      class) while never painting it leaves it out, yet both resolve to
      the same grid (the u-c6b M2 boundary: resolve-equality is NOT
      fold-equality at the painted-set boundary). "The batch changed no
      VALUES" is therefore NOT an ``==`` test on overlays; it is the
      value-difference test in :func:`overlay_diff_windows`;
    - a repaint of already-published values folds to the SAME overlay
      (same painted set, same final values);
    - painting a cell back to its baseline class KEEPS the cell in the
      painted set (at its baseline value): the overlay is the canonical
      paint STATE, and the batch-level no-op decision lives one level up
      in :func:`overlay_diff_windows` (repaint-to-baseline is a dirty
      footprint of exactly those cells, not a silent drop);
    - the patch count is bounded by the edited region's shape (a rectangle
      stays one patch no matter how many strokes built it), never by the
      session's stroke count.

    ``None`` (or an overlay that paints nothing) folds to ``None`` — the
    legacy no-overlay state, byte-identical semantics.

    ``resolved`` (u-c6b L5) optionally supplies the already-materialized
    ``overlay.resolve(baseline)`` grid so a caller that resolved for
    validation does not pay a second full-grid copy-and-resolve here; the
    default ``None`` resolves internally, byte-identical.
    """
    if overlay is None:
        return None
    grid = np.asarray(baseline)
    resolved_grid = overlay.resolve(grid) if resolved is None else resolved
    painted = overlay.changed_cell_mask(grid.shape)
    if not painted.any():
        return None
    return LandCoverOverlay(
        patches=_mask_run_patches(painted, resolved_grid)
    )


def overlay_diff_windows(
    new: LandCoverOverlay | None,
    old: LandCoverOverlay | None,
    baseline: np.ndarray,
    *,
    resolved_new: np.ndarray | None = None,
    resolved_old: np.ndarray | None = None,
) -> tuple[RasterWindow, ...]:
    """Windows covering exactly the cells ``new`` changes vs ``old``.

    The u-c4-review M3 dirty rule: a batch's dirty footprint is the
    symmetric difference between the overlay the batch would run under and
    the overlay whose results are already PUBLISHED — both resolved
    against the same baseline — so repainting already-published values
    dirties nothing. The windows are the canonical rectangle decomposition
    of the difference mask (deterministic; a rectangular change is one
    window), ready for the worker's dirty-window merge.

    u-c6b note: this is the VALUE difference — the dirty footprint the
    executor stages is its UNION with the batch's STROKE windows (the
    plan's promise basis; the planner sees deltas, never resolved
    values), computed by the executor. ``resolved_new``/``resolved_old``
    (u-c6b L5) optionally supply pre-materialized resolved grids so the
    executor hoists the per-batch full-grid resolves to one pass; the
    defaults resolve internally, byte-identical.
    """
    grid = np.asarray(baseline)
    new_grid = grid if new is None else (
        new.resolve(grid) if resolved_new is None else resolved_new
    )
    old_grid = grid if old is None else (
        old.resolve(grid) if resolved_old is None else resolved_old
    )
    diff = new_grid != old_grid
    if not diff.any():
        return ()
    return tuple(
        patch.window for patch in _mask_run_patches(diff, None)
    )


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------


class LandCoverSurfaceAdapter(EditAdapter):
    """EditAdapter for region-local land-cover painting (registry id above).

    A stateless service: the registry, dependency graph, and safety policy
    are injected at construction, and every protocol method derives its
    result from the command and the
    :class:`~solweig_gpu.incremental.edit_types.SiteContext`. All outputs
    are frozen records, so identical inputs reproduce bitwise-identical
    deltas and plans. The adapter holds no site state and no class grids:
    the scenario's resolved grid lives in the :class:`LandCoverOverlay`
    the executor derives from committed deltas.
    """

    adapter_id: str = ADAPTER_ID
    schema_version: int = ADAPTER_SCHEMA_VERSION

    def __init__(
        self,
        *,
        registry: AdapterRegistry | None = None,
        graph: EditGraph | None = None,
        policy: SafetyPolicy | None = None,
        read_halo_pixels: int = 0,
    ) -> None:
        self._registry = registry if registry is not None else builtin_registry()
        self._graph = graph if graph is not None else default_edit_graph()
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
        self._registry.validate_operation(self.adapter_id, command.operation)
        patches = _patches_from_command(
            command, self._registry, context.grid
        )
        delta = LandCoverPaintDelta(
            source_node_id=LANDCOVER_SOURCE_NODE,
            adapter_id=self.adapter_id,
            patches=patches,
        )
        return ValidatedEdit(
            command=command,
            adapter_id=self.adapter_id,
            schema_version=self.schema_version,
            source_node_id=LANDCOVER_SOURCE_NODE,
            delta=delta,
        )

    # -- protocol: delta, plan, apply, preview, fixtures -----------------------

    def source_delta(
        self, edit: ValidatedEdit, context: SiteContext
    ) -> LandCoverPaintDelta:
        """Return the delta produced at validation (single derivation)."""
        return self._require_own_delta(edit)

    def impact_plan(
        self, edit: ValidatedEdit, context: SiteContext
    ) -> ImpactPlan:
        """The single-edit impact plan, delegated to the core planner.

        The dirty closure, spatial/temporal scopes, and estimate formulas
        are the planner's (and therefore the engine's): the graph's two
        ``landcover ->`` edges ([landcover, radiation],
        [landcover, surface_thermal_state]) make the closure
        ``radiation -> surface_thermal_state -> tmrt -> utci`` (plus
        ``wbgt``, pruned as never-planned), the family's windowed nominal
        scope keeps every one of those stages WINDOWS over the paint
        footprint (until the safety policy's dirty-fraction rule demotes a
        stage to FULL, recorded in ``fallback_reasons``), and every
        geometry/view-factor cache (``walls``, ``wall_aspect``, both
        visibilities, ``svf``, ``relative_geometry``, ``time_shadow``,
        ``solar_atmospheric_state``) stays reusable. ``surface_thermal_state``
        replays from timestep 0. With the default ``read_halo_pixels = 0``
        the returned plan equals the core :meth:`EditPlanner.plan` output
        for ``[edit]`` bitwise.
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
        commits (returning the staged deltas, which the U-C executor folds
        via :func:`landcover_overlay_from_deltas` into the solver's class
        overlay) or rolls back cleanly. Defense-in-depth at the staging
        door (the model-parameter reviewer's finding, now the wave
        standard): the planner's ``validate_delta`` fences payloads at plan
        time, and this door re-runs the same fence (classes + window
        sanity) on whatever reaches it, so a hand-built delta routed
        straight to apply can never stage unpaintable content.
        """
        if not isinstance(delta, LandCoverPaintDelta):
            raise SourceDeltaError(
                f"expected LandCoverPaintDelta, got {type(delta).__name__}"
            )
        if delta.adapter_id != self.adapter_id:
            raise SourceDeltaError(
                f"delta belongs to adapter {delta.adapter_id!r}, not "
                f"{self.adapter_id!r}"
            )
        if delta.source_node_id != LANDCOVER_SOURCE_NODE:
            raise SourceDeltaError(
                f"delta targets source node {delta.source_node_id!r}, not "
                f"{LANDCOVER_SOURCE_NODE!r}"
            )
        _check_paint_payload(delta)
        transaction.stage(delta)

    def preview_descriptor(
        self, edit: ValidatedEdit, context: SiteContext
    ) -> PreviewDescriptor:
        """Surface-material preview with mandatory not-exact caveats.

        The registry kind is ``surface_material``: the preview recolours
        the painted cells client-side and is explicitly NOT the scientific
        output — the surface-property grids and downstream products come
        from the solver job (mission invariant ``preview_is_not_exact``).
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

    def _require_own_delta(self, edit: ValidatedEdit) -> LandCoverPaintDelta:
        if not isinstance(edit.delta, LandCoverPaintDelta):
            raise SourceDeltaError(
                "edit must carry a LandCoverPaintDelta, got "
                f"{type(edit.delta).__name__}"
            )
        if edit.delta.adapter_id != self.adapter_id:
            raise SourceDeltaError(
                f"edit delta belongs to adapter {edit.delta.adapter_id!r}"
            )
        return edit.delta

    def _with_read_halo(self, plan: ImpactPlan, grid: RasterGrid) -> ImpactPlan:
        """Expand windowed stages' read windows by the configured halo.

        The GVF walk (solweig.py:2124-2133) integrates ground properties
        up to the walk reach, so receptors outside the paint footprint
        still see painted cells: a READ halo, never a write expansion
        (write scopes and estimates stay the planner's, so plans compare
        bitwise-equal at halo 0). The U-C executor widens read windows
        further via ``required_halo_pixels`` — this adapter never claims
        the halo is exact.
        """
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


def register_landcover_surface_adapter(
    registry: AdapterRegistry | None = None,
) -> AdapterRegistry:
    """Register this adapter (idempotent) and return the registry.

    Kept separate from the earlier adapters' helpers so they stay
    untouched; ``adapters.register_default_adapters`` composes all of them.
    """
    target = registry if registry is not None else builtin_registry()
    target.register(LandCoverSurfaceAdapter(registry=target))
    return target
