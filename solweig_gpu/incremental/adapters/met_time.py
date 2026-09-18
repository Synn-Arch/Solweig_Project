# SPDX-License-Identifier: GPL-3.0-only
"""Meteorological-forcing edit adapter (``meteorological_forcing``).

This adapter implements the edit-adapter contract
(``docs/incremental_design_tool/universal_editing/edit_adapter_contract.md``)
for the ``meteorological_forcing`` registry entry. Met/time forcing is a
*per-timestep, site-global* input: the 25-column met text file
(preprocessor.py:1130-1146) is overlaid scenario-side — the baseline arrays
are immutable (copy-on-write, a hard invariant) and the overlay is applied
at solver input assembly (:func:`solweig_gpu.incremental.solver.load_site_forcing`
``overlay=`` seam). The adapter wraps no physics; it stages typed
:class:`~solweig_gpu.incremental.edit_types.ForcingDelta` records and
delegates every planning decision to the core
:class:`~solweig_gpu.incremental.planner.EditPlanner`:

- **validate** maps the user state (operation, timestep selection, and a
  ``values`` mapping of forcing variable -> new value, plus an optional
  ``old_state`` of claimed current values) through
  :meth:`solweig_gpu.incremental.edit_registry.AdapterRegistry.validate_edit_state`
  first, then enforces the documented per-variable domain
  (:data:`VARIABLE_SPECS`) and the timestep bounds from the
  :class:`~solweig_gpu.incremental.edit_types.SiteContext`. The safe/unknown/
  blocked variable vocabulary itself is the registry's
  (:data:`~solweig_gpu.incremental.edit_registry.METEOROLOGY_VARIABLES_SAFE`
  / ``..._BLOCKED``); the same fence guards the delta payload at the planner
  (:meth:`AdapterRegistry.validate_delta`), so a bypass-built delta can never
  plan.
- **source_delta** returns the delta built during validation (single
  validation pass); the base class's ``spatial_windows`` stays ``()``
  because forcing has no local footprint: its spatial effect is entirely
  downstream, so the planner resolves the whole family to FULL (registry
  ``nominal_spatial_scope`` is ``full_downstream_only``).
- **impact_plan** delegates to the core planner for the single-edit batch,
  so adapter plans and engine plans are bitwise-identical by construction.
  The dirty closure follows the reconciled graph edges —
  ``meteorology -> solar_atmospheric_state -> time_shadow -> radiation ->
  surface_thermal_state -> tmrt -> utci`` (plus every other direct
  ``meteorology ->`` edge) — while the geometry caches (``walls``,
  ``wall_aspect``, both visibilities, ``svf``, ``relative_geometry``) stay
  reusable. ``time_shadow`` IS dirty: the graph edge
  ``[meteorology, solar_atmospheric_state]`` exists, so the planner
  recomputes it even though the fenced variable set never shifts a time
  column — an over-approximation the planner is entitled to (the graph, not
  a per-adapter guess, decides the closure).
- **apply_source_delta** stages the typed delta in a
  :class:`~solweig_gpu.incremental.edit_types.ScenarioTransaction`;
  :func:`overlay_from_deltas` is the executor seam — it folds committed
  deltas into the :class:`ForcingOverlay` the solver resolves against the
  baseline met table (see the friction list below).
- **preview_descriptor** declares the registry's ``environment_status``
  kind: the preview is a *status disclosure* (which variables at which
  timesteps changed and what recomputes), never a spatial rendering, and is
  explicitly NOT the scientific output.

Temporal semantics (registry ``temporal_scope: changed_and_dependent_times``)
----------------------------------------------------------------------------

Forcing is per-timestep: one :class:`ForcingChange` per (variable, time
index) records the edit, so ``ForcingDelta.time_indices`` carries exactly
the changed steps. The core planner then (a) narrows every forcing-driven
time-varying stage to ONE/RANGE over the changed steps and (b) replays the
stateful ``surface_thermal_state`` from timestep 0 to the changed bound —
accumulated ground-heat state cannot be summarized, which is what
"dependent times" means for this family. The adapter never undercuts the
context's ``available_times``: every declared timestep must be one of them.

Executor seam and U-C integration friction
------------------------------------------

``load_site_forcing`` accepts the overlay and resolves it copy-on-write, but
the worker does not build one yet (:func:`ExactWorker` calls the loader with
``overlay=None`` — the legacy, bit-identical path). U-C plumbing needed for
a forcing edit to reach the kernel:

1. ``worker.py`` (forcing load site): build the scenario overlay from the
   transaction's committed forcing deltas via :func:`overlay_from_deltas`
   and pass it into ``load_site_forcing``; the cross-check relaxation is
   keyed on that explicit overlay object and nothing else.
2. ``run_utci_window`` consumes the met table wholesale
   (utci_process.py:671-690); no per-argument plumbing is needed for the
   fenced variables — the resolved table already carries them — but
   ``time_start``/``time_stop`` windowing must be widened from the write
   window's timesteps to the plan's temporal scope (REPLAY from 0) for
   stateful correctness (solve_window already replays from 0 for
   vegetation; forcing replay uses the same path).
3. ``run_full_tile`` (solver.py:1165) copies ``forcing.met_path`` — the
   BASELINE text — into the scratch site; a scenario forcing overlay must
   also materialize the resolved table there (or the full fallback silently
   ignores the overlay).

Conservative choices (contract ambiguities resolved the safe way):

1. **The registry fence runs first on BOTH command states and the delta
   payload; the adapter never re-implements it.** Unknown variables are
   rejected with the registry vocabulary in the message; blocked ones
   (time columns, ``diffuse_radiation``/``direct_radiation``,
   ``wind_direction``) name their coupling reason exactly like the
   model-parameter adapter's blocked set — so a bypass-built command
   *and* a bypass-built delta are both rejected at the planner.
   ``wind_direction`` is fenced as INERT, not merely unknown: the
   incremental path passes ``windcoeff_by_dir=None``
   (solver.py:1090-1091, :1179), so exposing it would misrepresent a
   no-op as an edit (the transmissivity precedent, lead ruling
   2026-09-02).
2. **One spelling per domain.** Values must be real numbers (``int``/
   ``float``, never ``bool`` or numeric strings); per-step series must be
   sequences of real numbers exactly as long as the range they cover.
3. **Bounds bind the new value only.** ``old_state`` is the caller's
   *claim* about the current series: it is name- and type-checked but not
   bounds-checked, so an out-of-range staged value can still be edited
   away. The new/after values are always fully bounded (finite + physical
   domain).
4. **Per-(variable, time) no-ops drop; empty edits are refused.** A
   declared ``before`` equal to the ``after`` disappears from the delta;
   an edit left with no changes is an explicit
   :class:`~solweig_gpu.incremental.edit_types.EditStateError`.
5. **Timesteps are fenced to the context.** When
   ``context.available_times`` is non-empty, every declared timestep must
   be one of them; the planner's estimate bound (``len(available_times)``)
   is therefore never undercut by an out-of-series edit.
6. **``preset`` is refused.** No preset catalog exists anywhere in the
   code or the registry; inventing one is a scientific decision, so the
   operation is rejected loudly rather than guessed.
7. **``reset`` is refused.** Baseline restore needs the site's baseline
   met series, which no :class:`SiteContext` carries (the model-parameter
   adapter has code-cited defaults; met baselines are site-specific). The
   executor should implement reset as overlay-drop instead — staging a
   whole-series "back to unknown values" delta is refused, never guessed.
8. **Stale revisions are refused early.** ``validate`` requires
   ``command.base_scene_revision == context.scene_revision`` (UEDIT-006);
   the planner re-checks against the authoritative scene state.
9. **The variable table self-checks.** Every spec's column index is a
   distinct index inside the 25-column met layout, its bounds are finite
   and ordered, and the table covers exactly the registry safe set — a
   registry change that forgets this adapter fails at import time rather
   than at a user's edit.
10. **The overlay never mutates its input.** :meth:`ForcingOverlay.resolve`
    copies the baseline table before the first write (copy-on-write), so
    cache-backed and loader-backed arrays stay immutable no matter how
    often an overlay resolves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ..edit_graph import EditGraph, SceneGraphState, default_edit_graph
from ..edit_registry import (
    METEOROLOGY_SOURCE_NODE,
    METEOROLOGY_VARIABLES_BLOCKED,
    METEOROLOGY_VARIABLES_SAFE,
    MET_UTCI_ONLY_VARIABLES,
    MET_VARIABLE_AFFINITY,
    MET_VARIABLE_AFFINITY_EVIDENCE,
    AdapterRegistry,
    EditAdapter,
    builtin_adapter_metadata,
    builtin_registry,
)
from ..edit_types import (
    EditCommand,
    EditStateError,
    ForcingChange,
    ForcingDelta,
    ImpactPlan,
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
    "ForcingOverlay",
    "ForcingVariableSpec",
    "MET_COLUMN_COUNT",
    "MET_UTCI_ONLY_VARIABLES",
    "MET_VARIABLE_AFFINITY",
    "MET_VARIABLE_AFFINITY_EVIDENCE",
    "VARIABLE_SPECS",
    "MeteorologicalForcingAdapter",
    "overlay_from_deltas",
    "register_meteorological_forcing_adapter",
]

#: Registry id this adapter implements (must match the builtin metadata).
ADAPTER_ID = "meteorological_forcing"

#: Adapter property-schema version; bump on any user-state schema change
#: (the contract: an adapter schema change invalidates incompatible edit
#: events and cached source overlays).
ADAPTER_SCHEMA_VERSION = 1

#: Registry-declared operations (mirrored for honest error messages).
_OPERATIONS = frozenset({"update_time_row", "update_range", "preset", "reset"})

#: Column count of the met text layout (preprocessor.py:1130-1146).
MET_COLUMN_COUNT = 25


@dataclass(frozen=True, slots=True)
class ForcingVariableSpec:
    """One fenced forcing variable's column, units, and physical domain.

    ``column`` is the met table column index the overlay writes;
    ``low``/``high`` bound every *new* value (conservative choice 3). The
    code enforces no met-value bounds anywhere, so every range here is a
    loose physical plausibility fence flagged in ``uncertain`` rather than
    an invented tight one (mirrors the model-parameter adapter's treatment
    of its unbounded parameters).
    """

    variable: str
    column: int
    units: str
    low: float
    high: float
    bounds_basis: str
    uncertain: str = ""


#: Documented per-variable domains, in registry safe-set order. Sources:
#: the U-A audit §(c) forcing-column table (consumption sites cited) and
#: ``solweig_gpu/utci_process.py:671-690`` (column reads) / ``:787-788`` /
#: ``:870-877`` (physics use).
VARIABLE_SPECS: tuple[ForcingVariableSpec, ...] = (
    ForcingVariableSpec(
        variable="air_temperature",
        column=11,
        units="degC",
        low=-60.0,
        high=60.0,
        bounds_basis="dry-bulb air temperature (Td, read as Ta at "
        "utci_process.py:674): drives vapour pressure, clearness index, "
        "wall/surface temperature parameterisation, nocturnal Lup, UTCI; "
        "the UTCI polynomial is fitted far inside this range",
        uncertain="the code enforces no bound; +/-60 degC is a loose "
        "plausibility fence (record extremes), not a model-validity range",
    ),
    ForcingVariableSpec(
        variable="humidity",
        column=10,
        units="percent",
        low=0.0,
        high=100.0,
        bounds_basis="relative humidity (RH, utci_process.py:675): enters "
        "physics as RH/100 (clearness index :787-788), so the physical "
        "[0, 1] fraction bounds the percent spelling to [0, 100] exactly",
    ),
    ForcingVariableSpec(
        variable="radiation",
        column=14,
        units="W m-2",
        low=0.0,
        high=1500.0,
        bounds_basis="global shortwave radiation (Kdn/radG, "
        "utci_process.py:676): drives the clearness index and, under "
        "onlyglobal = 1, the estimated diffuse/direct split "
        "(solweig.py:2055-2059)",
        uncertain="the code enforces no bound; 1500 W m-2 covers the solar "
        "constant plus cloud-enhancement excursions, loosely",
    ),
    ForcingVariableSpec(
        variable="wind_speed",
        column=9,
        units="m s-1",
        low=0.0,
        high=75.0,
        bounds_basis="wind speed (Wind/Ws, utci_process.py:680): the "
        "near-ground wind is coeff * Ws clamped to >= 0.15 m/s "
        "(:870-873) for UTCI",
        uncertain="the code enforces no upper bound; 75 m s-1 is a loose "
        "plausibility fence well above any sustained observation",
    ),
    ForcingVariableSpec(
        variable="pressure",
        column=12,
        units="hPa",
        low=300.0,
        high=1100.0,
        bounds_basis="station pressure (press/P, utci_process.py:679; the "
        "met column carries hPa, standard atmosphere 1013): clearness "
        "index (:787-788) and the WBGT wet-bulb input (:694); the fence "
        "spans high-altitude stations (~300 hPa) to above any recorded "
        "surface high",
        uncertain="the code enforces no bound; the range is a loose "
        "plausibility fence",
    ),
    ForcingVariableSpec(
        variable="uhii",
        column=24,
        units="K",
        low=-10.0,
        high=15.0,
        bounds_basis="urban-heat-island intensity (uhii, "
        "utci_process.py:688): added to the Ta plane only (:874/:877 -> "
        "UTCI/ta outputs) and to the WBGT wet-bulb input (:692, disabled "
        "incrementally); it never enters the radiation physics",
        uncertain="the code enforces no bound; -10/+15 K is a loose "
        "plausibility fence around observed UHI intensities",
    ),
)

_SPEC_BY_VARIABLE: dict[str, ForcingVariableSpec] = {
    spec.variable: spec for spec in VARIABLE_SPECS
}


def _check_variable_table() -> None:
    """Drift guard: the spec table mirrors the registry fence exactly.

    Also checks column indices are distinct and inside the 25-column met
    layout and that every documented domain is finite and ordered, so a
    registry change (or a column renumbering in the met format) that
    forgets this adapter fails at import instead of at a user's edit.
    """
    spec_variables = {spec.variable for spec in VARIABLE_SPECS}
    if spec_variables != set(METEOROLOGY_VARIABLES_SAFE):
        missing = sorted(set(METEOROLOGY_VARIABLES_SAFE) - spec_variables)
        extra = sorted(spec_variables - set(METEOROLOGY_VARIABLES_SAFE))
        raise EditStateError(
            f"VARIABLE_SPECS must cover the registry safe set exactly; "
            f"missing={missing}, extra={extra}"
        )
    if set(MET_VARIABLE_AFFINITY) != spec_variables:
        raise EditStateError(
            "MET_VARIABLE_AFFINITY drifted from VARIABLE_SPECS "
            f"({sorted(set(MET_VARIABLE_AFFINITY) ^ spec_variables)}); every "
            "editable variable must carry a code-proven solver affinity"
        )
    columns = [spec.column for spec in VARIABLE_SPECS]
    if len(set(columns)) != len(columns):
        raise EditStateError(
            f"forcing variable columns must be distinct, got {sorted(columns)}"
        )
    for spec in VARIABLE_SPECS:
        if not 0 <= spec.column < MET_COLUMN_COUNT:
            raise EditStateError(
                f"forcing variable {spec.variable!r} writes column "
                f"{spec.column}, outside the {MET_COLUMN_COUNT}-column met "
                "layout (preprocessor.py:1130-1146)"
            )
        if not (math.isfinite(spec.low) and math.isfinite(spec.high)):
            raise EditStateError(
                f"forcing variable {spec.variable!r} has non-finite bounds"
            )
        if spec.low >= spec.high:
            raise EditStateError(
                f"forcing variable {spec.variable!r} has unordered bounds "
                f"[{spec.low}, {spec.high}]"
            )


_check_variable_table()


# ---------------------------------------------------------------------------
# Command-state parsing (shared by validate and the fixture vocabulary)
# ---------------------------------------------------------------------------


def _require_number(value: Any, variable: str, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EditStateError(
            f"{label}: {variable} must be a number, got "
            f"{type(value).__name__}"
        )
    number = float(value)
    if not math.isfinite(number):
        raise EditStateError(
            f"{label}: {variable} must be finite, got {number!r}"
        )
    return number


def _check_domain(spec: ForcingVariableSpec, value: float, label: str) -> None:
    if value < spec.low:
        raise EditStateError(
            f"{label}: {spec.variable} = {value!r} is below the documented "
            f"lower bound {spec.low} {spec.units} ({spec.bounds_basis})"
        )
    if value > spec.high:
        raise EditStateError(
            f"{label}: {spec.variable} = {value!r} is above the documented "
            f"upper bound {spec.high} {spec.units} ({spec.bounds_basis})"
        )


def _require_index(value: Any, label: str, available: Sequence[int]) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EditStateError(
            f"{label} must be an integer timestep index, got "
            f"{type(value).__name__}"
        )
    if value < 0:
        raise EditStateError(f"{label} must be non-negative, got {value}")
    if available and value not in available:
        raise EditStateError(
            f"{label} = {value} is not one of the site's available "
            f"timesteps; the edit cannot undercut the series bounds "
            f"(available: {len(available)} steps, "
            f"[{min(available)}, {max(available)}])"
        )
    return value


def _values_mapping(state: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    values = state.get("values")
    if not isinstance(values, Mapping) or not dict(values):
        raise EditStateError(
            f"{label} requires a non-empty 'values' mapping of forcing "
            "variable to new value"
        )
    return values


def _parse_state(
    state: Mapping[str, Any],
    label: str,
    *,
    bounds: bool,
    available: Sequence[int],
) -> tuple[range | int, dict[str, Any]]:
    """Validate one command state; return its timestep selection + values.

    ``bounds=False`` is the old_state pass (a claim about the current
    series): names and types are checked, values are not bounded
    (conservative choice 3).
    """
    if not isinstance(state, Mapping):
        raise EditStateError(
            f"{label} must be a mapping, got {type(state).__name__}"
        )
    unknown_keys = sorted(set(state) - {"time_index", "time_start", "time_stop", "values"})
    if unknown_keys:
        raise EditStateError(
            f"{label} carries unknown fields {unknown_keys}; a forcing "
            "state is {'time_index', 'values'} for update_time_row and "
            "{'time_start', 'time_stop', 'values'} for update_range"
        )
    values = dict(_values_mapping(state, label))

    if "time_index" in state:
        if "time_start" in state or "time_stop" in state:
            raise EditStateError(
                f"{label} mixes 'time_index' with 'time_start'/'time_stop'; "
                "a state selects either one timestep or one contiguous range"
            )
        selection: range | int = _require_index(
            state["time_index"], f"{label}.time_index", available
        )
    else:
        if "time_start" not in state or "time_stop" not in state:
            raise EditStateError(
                f"{label} requires either 'time_index' or both 'time_start' "
                "and 'time_stop'"
            )
        start = _require_index(state["time_start"], f"{label}.time_start", available)
        stop = _require_index(state["time_stop"], f"{label}.time_stop", available)
        if stop < start:
            raise EditStateError(
                f"{label}.time_stop ({stop}) must be >= time_start ({start})"
            )
        selection = range(start, stop + 1)

    checked: dict[str, Any] = {}
    for variable in sorted(values):
        spec = _SPEC_BY_VARIABLE[variable]
        value = values[variable]
        if isinstance(value, (list, tuple)):
            width = 1 if isinstance(selection, int) else len(selection)
            if len(value) != width:
                raise EditStateError(
                    f"{label}: {variable} carries {len(value)} per-step "
                    f"values but the edit selects {width} timestep(s)"
                )
            steps = [
                _require_number(item, variable, label) for item in value
            ]
            if bounds:
                for item in steps:
                    _check_domain(spec, item, label)
            checked[variable] = tuple(steps)
        else:
            number = _require_number(value, variable, label)
            if bounds:
                _check_domain(spec, number, label)
            checked[variable] = number
    return selection, checked


def _expand_changes(
    selection: range | int,
    values: Mapping[str, Any],
    before: Mapping[str, Any] | None = None,
) -> tuple[ForcingChange, ...]:
    """One ForcingChange per (variable, timestep), canonical sorted order.

    A scalar ``before`` claim is copied to every selected timestep of its
    variable; per-step (sequence) claims cannot be aligned to one cell
    without an index convention, so they record ``None`` (conservative
    choice 3: the executor resolves true before-values from the scenario).
    """
    times = (selection,) if isinstance(selection, int) else tuple(selection)
    changes = []
    for time_index in times:
        for variable in sorted(values):
            after = values[variable]
            if isinstance(after, tuple):
                after = after[times.index(time_index)]
            declared = before.get(variable) if before else None
            if isinstance(declared, tuple):
                declared = None
            changes.append(
                ForcingChange(
                    variable=variable,
                    time_index=time_index,
                    before_value=declared,
                    after_value=after,
                )
            )
    return tuple(sorted(changes, key=lambda c: (c.variable, c.time_index)))


def _changes_from_command(
    command: EditCommand,
    registry: AdapterRegistry,
    available: Sequence[int],
) -> tuple[ForcingChange, ...]:
    """Validate states and build one command's canonical change records."""
    operation = command.operation
    if operation not in _OPERATIONS:
        # The registry check below reports undeclared operations; this guard
        # keeps the shape checks honest for custom registries.
        raise EditStateError(
            f"operation must be one of {sorted(_OPERATIONS)}, got "
            f"{operation!r}"
        )
    if operation in ("preset", "reset"):
        # Conservative choices 6/7: refused loudly with the reason; the
        # registry declares the operations, the v1 adapter implements only
        # the two whose semantics are fully determined.
        if operation == "preset":
            raise EditStateError(
                "operation 'preset' is not implemented: no forcing preset "
                "catalog exists in the code or the registry, and inventing "
                "one is a scientific decision — refused rather than guessed"
            )
        raise EditStateError(
            "operation 'reset' is not implemented as a delta: baseline "
            "restore needs the site's baseline met series, which no "
            "SiteContext carries — the executor drops the scenario overlay "
            "instead (refused rather than guessed, mirroring the "
            "model-parameter baseline-manifest refusal)"
        )

    old_state = command.old_state
    new_state = command.new_state
    # Registry schema enforcement first (rejected properties, fences) on
    # BOTH states, then the adapter's own field-level checks.
    for label, state in (("old_state", old_state), ("new_state", new_state)):
        if state is None:
            continue
        registry.validate_edit_state(ADAPTER_ID, dict(state))

    if new_state is None:
        raise EditStateError(
            f"an {operation!r} requires a new_state"
        )
    old_selection: range | int | None = None
    before: dict[str, Any] = {}
    if old_state is not None:
        old_selection, before = _parse_state(
            old_state, "old_state", bounds=False, available=available
        )
    new_selection, after = _parse_state(
        new_state, "new_state", bounds=True, available=available
    )
    if old_selection is not None and old_selection != new_selection:
        raise EditStateError(
            "old_state and new_state select different timesteps "
            f"({old_selection!r} vs {new_selection!r}); a forcing edit "
            "revalues one timestep selection"
        )

    changes = _expand_changes(new_selection, after, before)
    # Per-(variable, time) no-ops drop (conservative choice 4): a declared
    # before equal to the after value disappears; an edit left with no
    # changes is an explicit error, not a silent empty plan. The refusal
    # names every (variable, timestep) property and both equal values
    # (U-D item d) — an identical-value resubmit must refuse cleanly
    # instead of churning the full-tile forcing closure for a physically
    # identical scene.
    kept = [
        change
        for change in changes
        if change.before_value is None
        or change.before_value != change.after_value
    ]
    if not kept:
        raise EditStateError(
            "edit is a no-op: every targeted variable already carries the "
            "requested value at every selected timestep ("
            + identical_value_clause(
                (
                    f"{change.variable}@t={change.time_index}",
                    change.before_value,
                    change.after_value,
                )
                for change in changes
            )
            + "); an identical-value resubmit would churn the forcing "
            "recompute closure for a physically identical scene"
        )
    return tuple(kept)


# ---------------------------------------------------------------------------
# Committed deltas -> ForcingOverlay (the solver executor seam)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ForcingOverlay:
    """Copy-on-write met forcing overlay for one scenario.

    Built from committed :class:`ForcingDelta` records via
    :func:`overlay_from_deltas` (the executor seam) — never from raw user
    state. Construction re-validates the fence and every value domain, so
    the object handed to the solver is self-certifying:

    - :meth:`resolve` returns a *new* float64 table that equals the
      baseline everywhere except the declared (variable, time) cells; the
      baseline array itself is never written (hard invariant — it is
      typically loader- or cache-backed).
    - :meth:`changed_cell_mask` exposes exactly the cells the overlay
      declares, which is how the solver's met cross-check narrows to the
      undeclared remainder instead of trusting the overlay wholesale.
    """

    changes: tuple[ForcingChange, ...] = ()

    def __post_init__(self) -> None:
        seen: dict[tuple[str, int | None], ForcingChange] = {}
        for change in self.changes:
            if not isinstance(change, ForcingChange):
                raise SourceDeltaError(
                    "overlay changes must be ForcingChange records, got "
                    f"{type(change).__name__}"
                )
            if change.variable in METEOROLOGY_VARIABLES_BLOCKED:
                raise SourceDeltaError(
                    f"overlay carries blocked forcing variable "
                    f"{change.variable!r}: time columns are pinned to the "
                    "cached solar series, Kdiff/Kdir are dead while "
                    "onlyglobal = 1, and wind_direction is inert without "
                    "wind coefficients (u-b-met, 2026-09-02)"
                )
            spec = _SPEC_BY_VARIABLE.get(change.variable)
            if spec is None:
                raise SourceDeltaError(
                    f"overlay carries unknown forcing variable "
                    f"{change.variable!r}; the safe vocabulary is "
                    f"{sorted(METEOROLOGY_VARIABLES_SAFE)}"
                )
            for role, value in (
                ("after_value", change.after_value),
                ("before_value", change.before_value),
            ):
                if value is None:
                    continue
                values = value if isinstance(value, tuple) else (value,)
                for item in values:
                    if isinstance(item, bool) or not isinstance(item, (int, float)):
                        raise SourceDeltaError(
                            f"overlay {change.variable!r} {role} must be a "
                            f"number, got {type(item).__name__}"
                        )
                    if not math.isfinite(float(item)):
                        raise SourceDeltaError(
                            f"overlay {change.variable!r} {role} must be "
                            f"finite, got {item!r}"
                        )
            if change.after_value is not None:
                items = (
                    change.after_value
                    if isinstance(change.after_value, tuple)
                    else (change.after_value,)
                )
                for item in items:
                    try:
                        _check_domain(spec, float(item), "overlay")
                    except EditStateError as error:
                        # The overlay is an executor-side object: its
                        # refusals are SourceDeltaError, like every other
                        # rejection on this seam.
                        raise SourceDeltaError(str(error)) from error
            key = (change.variable, change.time_index)
            if key in seen:
                raise SourceDeltaError(
                    f"overlay carries duplicate change for variable "
                    f"{change.variable!r} at time_index {change.time_index!r}; "
                    "coalesce deltas before building the overlay "
                    "(overlay_from_deltas does this)"
                )
            seen[key] = change
        object.__setattr__(self, "changes", tuple(self.changes))

    # -- resolution ----------------------------------------------------------

    def resolve(self, baseline: np.ndarray) -> np.ndarray:
        """Return the overlay-resolved table; ``baseline`` stays untouched.

        Copy-on-write: the first declared write happens on a fresh float64
        copy, so the caller's array (typically the loader's or the cache's)
        is bitwise unchanged no matter how often an overlay resolves.
        """
        table = np.asarray(baseline)
        if table.ndim != 2:
            raise SourceDeltaError(
                f"baseline met table must be 2-D, got shape {table.shape}"
            )
        rows, columns = table.shape
        resolved = np.array(table, dtype=np.float64, copy=True)
        for change in self.changes:
            spec = _SPEC_BY_VARIABLE[change.variable]
            if spec.column >= columns:
                raise SourceDeltaError(
                    f"baseline met table has {columns} columns; variable "
                    f"{change.variable!r} needs column {spec.column} "
                    "(25-column met layout, preprocessor.py:1130-1146)"
                )
            value = change.after_value
            if change.time_index is None:
                if isinstance(value, tuple):
                    if len(value) != rows:
                        raise SourceDeltaError(
                            f"whole-series change for {change.variable!r} "
                            f"carries {len(value)} values but the series "
                            f"has {rows} timesteps"
                        )
                    resolved[:, spec.column] = np.asarray(value, dtype=np.float64)
                else:
                    resolved[:, spec.column] = float(value)
            else:
                if change.time_index >= rows:
                    raise SourceDeltaError(
                        f"time_index {change.time_index} is outside the "
                        f"baseline series ({rows} timesteps)"
                    )
                resolved[change.time_index, spec.column] = (
                    float(value[change.time_index])
                    if isinstance(value, tuple)
                    else float(value)
                )
        return resolved

    def changed_cell_mask(self, shape: tuple[int, ...]) -> np.ndarray:
        """Boolean (T, C) mask, True exactly at the declared cells."""
        mask = np.zeros(shape, dtype=bool)
        for change in self.changes:
            spec = _SPEC_BY_VARIABLE[change.variable]
            if change.time_index is None:
                mask[:, spec.column] = True
            else:
                mask[change.time_index, spec.column] = True
        return mask


def overlay_from_deltas(
    deltas: Sequence[ForcingDelta],
) -> ForcingOverlay:
    """Fold committed forcing deltas into the solver's :class:`ForcingOverlay`.

    Last-write-wins per (variable, time_index) in arrival order (the
    coalescer's semantics for one batch), with every variable and value
    re-checked against the fence and the documented domains — the
    executor's inputs are validated on this seam, not only inside adapter
    ``validate()`` calls. This is the hand-off the U-C integration passes
    as ``load_site_forcing(..., overlay=...)``.
    """
    folded: dict[tuple[str, int | None], ForcingChange] = {}
    for delta in deltas:
        if not isinstance(delta, ForcingDelta):
            raise SourceDeltaError(
                f"expected ForcingDelta, got {type(delta).__name__}"
            )
        if delta.adapter_id != ADAPTER_ID:
            raise SourceDeltaError(
                f"delta belongs to adapter {delta.adapter_id!r}, not "
                f"{ADAPTER_ID!r}"
            )
        if delta.source_node_id != METEOROLOGY_SOURCE_NODE:
            raise SourceDeltaError(
                f"delta targets source node {delta.source_node_id!r}, not "
                f"{METEOROLOGY_SOURCE_NODE!r}"
            )
        for change in delta.changes:
            key = (change.variable, change.time_index)
            previous = folded.get(key)
            first_before = (
                change.before_value if previous is None else previous.before_value
            )
            folded[key] = ForcingChange(
                change.variable,
                change.time_index,
                first_before,
                change.after_value,
            )
    # Whole-series records (time_index None) sort first so a per-timestep
    # record overrides them in ForcingOverlay.resolve; -1 is a safe sort
    # stand-in because real time indices are non-negative.
    changes = tuple(
        sorted(
            folded.values(),
            key=lambda c: (c.variable, -1 if c.time_index is None else c.time_index),
        )
    )
    return ForcingOverlay(changes=changes)


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------


class MeteorologicalForcingAdapter(EditAdapter):
    """EditAdapter for per-timestep meteorological forcing (registry id above).

    A stateless service: the registry, dependency graph, and safety policy
    are injected at construction, and every protocol method derives its
    result from the command and the
    :class:`~solweig_gpu.incremental.edit_types.SiteContext`. All outputs
    are frozen records, so identical inputs reproduce bitwise-identical
    deltas and plans. The adapter holds no site state and no met values:
    the scenario's resolved series live in the
    :class:`ForcingOverlay` the executor derives from committed deltas.
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
        changes = _changes_from_command(
            command, self._registry, context.available_times
        )
        delta = ForcingDelta(
            source_node_id=METEOROLOGY_SOURCE_NODE,
            adapter_id=self.adapter_id,
            changes=changes,
        )
        return ValidatedEdit(
            command=command,
            adapter_id=self.adapter_id,
            schema_version=self.schema_version,
            source_node_id=METEOROLOGY_SOURCE_NODE,
            delta=delta,
        )

    # -- protocol: delta, plan, apply, preview, fixtures -----------------------

    def source_delta(
        self, edit: ValidatedEdit, context: SiteContext
    ) -> ForcingDelta:
        """Return the delta produced at validation (single derivation)."""
        return self._require_own_delta(edit)

    def impact_plan(
        self, edit: ValidatedEdit, context: SiteContext
    ) -> ImpactPlan:
        """The single-edit impact plan, delegated to the core planner.

        The dirty closure, spatial/temporal scopes, and estimate formulas
        are the planner's (and therefore the engine's): the graph's
        ``meteorology ->`` edges make the closure ``solar_atmospheric_state
        -> time_shadow -> radiation -> surface_thermal_state -> tmrt ->
        utci`` (plus ``wbgt``, pruned as never-planned), the family's
        full-downstream-only nominal scope makes every one of those stages
        FULL, and the registry's ``changed_and_dependent_times`` narrows
        them to the changed steps while ``surface_thermal_state`` replays
        from timestep 0. Every geometry cache (``walls``, ``wall_aspect``,
        the visibilities, ``svf``, ``relative_geometry``) stays reusable.
        The returned plan equals the core :meth:`EditPlanner.plan` output
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
        return planner.plan([edit], state)

    def apply_source_delta(
        self, delta: SourceDelta, transaction: ScenarioTransaction
    ) -> None:
        """Stage the typed delta in a rollback-safe transaction.

        Staging never touches baseline state: the transaction either
        commits (returning the staged deltas, which the U-C executor folds
        via :func:`overlay_from_deltas` into the solver's forcing overlay)
        or rolls back cleanly. Wiring the overlay into the worker's
        ``load_site_forcing`` call is the U-C executor's job — this adapter
        only stages the typed, domain-checked record.
        """
        if not isinstance(delta, ForcingDelta):
            raise SourceDeltaError(
                f"expected ForcingDelta, got {type(delta).__name__}"
            )
        if delta.adapter_id != self.adapter_id:
            raise SourceDeltaError(
                f"delta belongs to adapter {delta.adapter_id!r}, not "
                f"{self.adapter_id!r}"
            )
        if delta.source_node_id != METEOROLOGY_SOURCE_NODE:
            raise SourceDeltaError(
                f"delta targets source node {delta.source_node_id!r}, not "
                f"{METEOROLOGY_SOURCE_NODE!r}"
            )
        # Staging door (u-b-met-reviewer MEDIUM-1; params precedent at
        # model_parameters.py, commit 3dc1f82): the planner's
        # validate_delta is names-only, so a hand-built ForcingDelta with a
        # non-finite or out-of-domain after_value would otherwise stage and
        # commit unchecked (rejection deferred to overlay_from_deltas).
        # ForcingOverlay.__post_init__ is the canonical fence on this seam
        # (names / finiteness / domains / duplicates) — run it before the
        # delta can enter a transaction.
        ForcingOverlay(changes=delta.changes)
        transaction.stage(delta)

    def preview_descriptor(
        self, edit: ValidatedEdit, context: SiteContext
    ) -> PreviewDescriptor:
        """Environment-status disclosure with mandatory not-exact caveats.

        The registry kind is ``environment_status``: there is no spatial
        preview for site-global forcing, so the preview *is* the status
        disclosure — which variables changed at which timesteps, what the
        temporal semantics are, and what the preview cannot tell you. It
        is interaction feedback, never scientific output (mission
        invariant ``preview_is_not_exact``).
        """
        delta = self._require_own_delta(edit)
        variables = ", ".join(
            sorted({change.variable for change in delta.changes})
        )
        times = delta.time_indices
        when = (
            f"timesteps {list(times)}"
            if times
            else "the whole series (no explicit timestep)"
        )
        return PreviewDescriptor(
            kind=self.metadata.preview,
            immediate=True,
            limitations=(
                "Forcing is site-global (full_downstream_only): there is "
                "no spatial preview, only this status disclosure (which "
                "stages recompute and why they are full-tile)",
                f"This edit changes {variables} at {when}; preview does "
                "not recompute any output — the scientific values come "
                "from the solver job (preview_is_not_exact)",
                "changed_and_dependent_times: the changed timesteps "
                "recompute and the ground-heat accumulator replays from "
                "timestep 0 to the changed bound — an earlier edit is "
                "never skipped silently",
                "Geometry and view-factor caches stay reusable: walls, "
                "wall_aspect, the visibilities and svf are forcing-"
                "invariant, so the disclosure never claims a geometry "
                "change",
                "wind_direction is not editable (inert until the wind "
                "scientific extension, UEDIT-010) and uhii shifts only "
                "the Ta planes, never the radiation physics",
            ),
        )

    def validation_fixtures(self) -> tuple[str, ...]:
        """Scientific fixture names, verbatim from the registry metadata."""
        return self.metadata.validation_fixtures

    # -- internals ---------------------------------------------------------------

    def _require_own_delta(self, edit: ValidatedEdit) -> ForcingDelta:
        if not isinstance(edit.delta, ForcingDelta):
            raise SourceDeltaError(
                "edit must carry a ForcingDelta, got "
                f"{type(edit.delta).__name__}"
            )
        if edit.delta.adapter_id != self.adapter_id:
            raise SourceDeltaError(
                f"edit delta belongs to adapter {edit.delta.adapter_id!r}"
            )
        return edit.delta


def register_meteorological_forcing_adapter(
    registry: AdapterRegistry | None = None,
) -> AdapterRegistry:
    """Register this adapter (idempotent) and return the registry.

    Kept separate from the earlier adapters' helpers so they stay
    untouched; ``adapters.register_default_adapters`` composes all of them.
    """
    target = registry if registry is not None else builtin_registry()
    target.register(MeteorologicalForcingAdapter(registry=target))
    return target
