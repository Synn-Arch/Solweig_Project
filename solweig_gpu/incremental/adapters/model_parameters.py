# SPDX-License-Identifier: GPL-3.0-only
"""Model/receptor-parameter edit adapter (``model_receptor_parameters``).

This adapter implements the edit-adapter contract
(``docs/incremental_design_tool/universal_editing/edit_adapter_contract.md``)
for the ``model_receptor_parameters`` registry entry. Parameters are
*site-global scalars* (albedos, emissivities, absorption fractions, receptor
geometry, phenology, model switches) consumed inside
:func:`solweig_gpu.utci_process.run_utci_window`; the adapter wraps no
physics — it stages typed
:class:`~solweig_gpu.incremental.edit_types.ModelParameterDelta` records and
delegates every planning decision to the core
:class:`~solweig_gpu.incremental.planner.EditPlanner`:

- **validate** maps the user state (parameter name -> new value, plus an
  optional ``old_state`` of claimed current values) through
  :meth:`solweig_gpu.incremental.edit_registry.AdapterRegistry.validate_edit_state`
  first, so the safe-set-only / blocked-set / unknown-name rulings are the
  *registry's*, never re-implemented here. On top of the registry schema the
  adapter enforces a documented per-parameter domain
  (:data:`PARAMETER_SPECS`) — albedos/emissivities/absorptivities and the
  weighting fractions in ``[0, 1]``, ``cyl`` a bool, day-of-year integers in
  ``[1, 366]``, switch flags in ``{0, 1}``, ``height > 0`` — and rejects
  identity edits.
- **source_delta** returns the delta built during validation (single
  validation pass); the base class's ``spatial_windows``
  stays ``()`` because a parameter change has no local footprint: its
  spatial effect is entirely downstream, so the planner resolves the whole
  family to FULL (registry ``nominal_spatial_scope`` starts with ``full``).
- **impact_plan** delegates to the core planner for the single-edit batch,
  so adapter plans and engine plans are bitwise-identical by construction.
- **apply_source_delta** stages the typed delta in a
  :class:`~solweig_gpu.incremental.edit_types.ScenarioTransaction`;
  :func:`kernel_arguments_from_deltas` is the executor seam — it folds
  committed deltas into the flat ``name -> value`` mapping the U-C
  integration forwards into the physics (see the friction list below).
- **preview_descriptor** declares the registry's
  ``parameter_scope_disclosure`` kind: the preview is a *disclosure* (which
  stages recompute and why), never a spatial rendering, and is explicitly
  NOT the scientific output.

Reset semantics
---------------

``reset`` resolves each target parameter to the value in
:data:`PARAMETER_DEFAULTS`. Every safe parameter has an explicit code-cited
default constant, so no entry is ``baseline-manifest`` today; the marker and
its rule are kept in the table's vocabulary because a future parameter
without a code constant must resolve against the site's baseline manifest,
and :class:`~solweig_gpu.incremental.edit_types.SiteContext` carries no
parameter store yet — that combination is refused loudly rather than guessed
(conservative choice 5).

``reset`` therefore restores the *documented code defaults*, not the values
the site was initialised with: if a scenario was created with non-default
parameters, ``reset`` is not a baseline restore for them. That stronger
semantic needs the baseline manifest in the site context (friction item 4).

Executor seam and U-C integration friction
------------------------------------------

None of the fourteen safe parameters is an argument of
``run_utci_window`` today; all of them are module-level constants
(``solweig_gpu/utci_process.py:50-68``) read inside its body. They split into
two plumbing tiers (:data:`PLUMBING_CLASSIFICATION`):

1. ``kernel_arg`` (10) — already forwarded to the physics kernel
   :func:`solweig_gpu.solweig.Solweig_2022a_calc` at
   ``utci_process.py:824-827`` (``albedo_b, absK, absL, ewall, Fside, Fup,
   Fcyl`` at :824; ``cyl, elvis`` at :825; ``anisotropic_sky`` at :827).
   U-C work: add the parameter to ``run_utci_window`` and forward it.
2. ``loop_local`` (4) — consumed in the loop prologue before any kernel
   call, so they need body plumbing as well as a forwarded argument:
   ``transVeg`` (:649 -> ``psi`` :713, ``svfbuveg`` :730, ``diffsh`` :734),
   ``height`` (:716 -> ``first``/``second`` GVF walk reach :719-722), and
   ``firstdayleaf``/``lastdayleaf`` (:67-68 -> leaf-on mask :708-711 -> ``psi``).

``anisotropic_sky`` is classified ``kernel_arg`` (it is forwarded at :827)
but its value is also assigned loop-locally at :658, so the override must
replace that assignment too. ``patch_option`` is assigned at :659/:1120 and
is BLOCKED precisely because it also keys the SVF cache.

Conservative choices (contract ambiguities resolved the safe way):

1. **The registry schema runs first on both states.** Blocked names
   (``patch_option``, ``scale``, ``location``, ``utc``, ``walllimit``,
   ``onlyglobal``, ``albedo_g``) and unknown names are rejected by
   :meth:`AdapterRegistry.validate_edit_state` with the registry's ruling
   text; the adapter adds its domain checks afterwards.
2. **One spelling per domain.** ``int``-valued parameters reject ``bool``
   (``True`` is an ``int`` subclass, so ``elvis=True`` and ``elvis=1`` must
   not both spell the same switch); ``cyl`` accepts only ``bool``
   (``cyl = True``, utci_process.py:62); float parameters reject ``bool``
   and numeric strings exactly like the vegetation adapter's property
   checks.
3. **Bounds bind the new value only.** ``old_state`` is the caller's *claim*
   about the current value: it is name- and type-checked but not
   bounds-checked, so an out-of-range staged value can still be reset. The
   new/after value is always fully bounded.
4. **Per-parameter no-ops drop; empty edits are refused.** A parameter whose
   declared ``before`` equals its ``after`` disappears from the delta (the
   engine's coalescer would drop it anyway); an edit left with no changes is
   an explicit :class:`~solweig_gpu.incremental.edit_types.EditStateError`,
   mirroring the vegetation adapter's identity-edit rejection.
5. **``reset`` never accepts user-supplied after-values.** A ``reset``
   carrying a ``new_state`` is rejected: the after-values are the documented
   defaults, so nothing else can be smuggled through the operation.
   ``old_state`` selects the reset subset (its keys); ``old_state=None``
   resets every safe parameter.
6. **``before`` may be ``None``.** Without ``old_state`` the adapter cannot
   know the scenario's current value; it records ``None`` and the executor
   resolves the true before-value from the scenario parameter store at
   apply time (documented, never guessed here).
7. **Stale revisions are refused early.** ``validate`` requires
   ``command.base_scene_revision == context.scene_revision`` (UEDIT-006);
   the planner re-checks against the authoritative scene state.
8. **The defaults table self-checks.** Every code-cited default must satisfy
   its own documented domain; a code drift that breaks that invariant fails
   at import time rather than at a user's edit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from ..edit_graph import EditGraph, SceneGraphState, default_edit_graph
from ..edit_registry import (
    MODEL_PARAMETERS_BLOCKED,
    MODEL_PARAMETERS_SAFE,
    MODEL_PARAMETERS_SOURCE_NODE,
    AdapterRegistry,
    EditAdapter,
    builtin_adapter_metadata,
    builtin_registry,
)
from ..edit_types import (
    EditCommand,
    EditStateError,
    ImpactPlan,
    ModelParameterChange,
    ModelParameterDelta,
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
    "PARAMETER_DEFAULTS",
    "PARAMETER_SPECS",
    "PLUMBING_CLASSIFICATION",
    "ParameterSpec",
    "ModelReceptorParametersAdapter",
    "check_parameter_value",
    "kernel_arguments_from_deltas",
    "register_model_parameters_adapter",
]

#: Registry id this adapter implements (must match the builtin metadata).
ADAPTER_ID = "model_receptor_parameters"

#: Adapter property-schema version; bump on any user-state schema change
#: (the contract: an adapter schema change invalidates incompatible edit
#: events and cached source overlays).
ADAPTER_SCHEMA_VERSION = 1

#: Registry-declared operations (mirrored for honest error messages).
_OPERATIONS = frozenset({"update", "reset"})

#: Day-of-year bound: 366 covers leap years; the physics leaf mask
#: (utci_process.py:708-711) handles wraparound, so no ordering constraint
#: between firstdayleaf and lastdayleaf is imposed.
_DOY_LOW, _DOY_HIGH = 1, 366


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    """One safe parameter's documented domain, default, and provenance.

    ``low``/``high`` bound the *new* value (conservative choice 3); ``None``
    means unbounded on that side. ``default_source`` cites the code constant
    (``"code:utci_process.py:NN"``) or marks the entry
    ``"baseline-manifest"`` — a parameter with no code constant must be
    resolved against the site's baseline manifest, which no context carries
    today, so such an entry is refused at reset time rather than guessed.
    """

    name: str
    kind: str  # "float" | "int" | "bool"
    low: float | None
    high: float | None
    default: Any
    default_source: str
    bounds_basis: str
    uncertain: str = ""
    low_exclusive: bool = False

    @property
    def is_baseline_manifest_default(self) -> bool:
        return self.default_source == "baseline-manifest"


#: Documented per-parameter domains, in registry safe-set order. Sources:
#: ``solweig_gpu/utci_process.py:50-68`` (module constants) and
#: ``:649-734`` (loop-local constants); audit evidence
#: ``docs/incremental_design_tool/universal_editing/audit/2026-09-02-u-a-audit.md``
#: section (b). Where the code enforces no bound, the loose physical bound is
#: used and flagged in ``uncertain`` rather than inventing a tight one.
PARAMETER_SPECS: tuple[ParameterSpec, ...] = (
    ParameterSpec(
        name="albedo_b",
        kind="float",
        low=0.0,
        high=1.0,
        default=0.2,
        default_source="code:utci_process.py:50",
        bounds_basis="albedo is a reflectance fraction; forwarded as a "
        "kernel arg at utci_process.py:824",
    ),
    ParameterSpec(
        name="ewall",
        kind="float",
        low=0.0,
        high=1.0,
        default=0.9,
        default_source="code:utci_process.py:52",
        bounds_basis="emissivity is a fraction; kernel arg at :824",
    ),
    ParameterSpec(
        name="absK",
        kind="float",
        low=0.0,
        high=1.0,
        default=0.7,
        default_source="code:utci_process.py:54",
        bounds_basis="shortwave absorptivity fraction; kernel arg at :824",
    ),
    ParameterSpec(
        name="absL",
        kind="float",
        low=0.0,
        high=1.0,
        default=0.95,
        default_source="code:utci_process.py:55",
        bounds_basis="longwave absorptivity fraction; kernel arg at :824",
    ),
    ParameterSpec(
        name="Fside",
        kind="float",
        low=0.0,
        high=1.0,
        default=0.22,
        default_source="code:utci_process.py:58",
        bounds_basis="Sstr side-weighting fraction; kernel arg at :824",
        uncertain="the code enforces no bound and no Fside/Fup sum "
        "constraint; only the per-parameter [0, 1] fraction bound is "
        "documented",
    ),
    ParameterSpec(
        name="Fup",
        kind="float",
        low=0.0,
        high=1.0,
        default=0.06,
        default_source="code:utci_process.py:59",
        bounds_basis="Sstr up-weighting fraction; kernel arg at :824",
        uncertain="see Fside: no cross-parameter sum constraint is enforced",
    ),
    ParameterSpec(
        name="Fcyl",
        kind="float",
        low=0.0,
        high=1.0,
        default=0.28,
        default_source="code:utci_process.py:60",
        bounds_basis="cylinder-side weighting fraction; kernel arg at :824",
        uncertain="see Fside: no cross-parameter sum constraint is enforced",
    ),
    ParameterSpec(
        name="cyl",
        kind="bool",
        low=None,
        high=None,
        default=True,
        default_source="code:utci_process.py:62",
        bounds_basis="boolean branch selector for the Sstr/Lside merge; "
        "kernel arg at :825",
    ),
    ParameterSpec(
        name="height",
        kind="float",
        low=0.0,
        high=None,
        default=1.1,
        default_source="code:utci_process.py:716",
        bounds_basis="height of calculation (m); drives the GVF walk reach "
        "first/second at :719-722",
        uncertain="no upper bound is documented in the code (the march "
        "reach grows as round(height*20)); only strict positivity is "
        "enforced",
        low_exclusive=True,
    ),
    ParameterSpec(
        name="transVeg",
        kind="float",
        low=0.0,
        high=1.0,
        default=0.03,
        default_source="code:utci_process.py:649",
        bounds_basis="vegetation transmissivity fraction (3/100); feeds "
        "psi :713, svfbuveg :730, diffsh :734",
    ),
    ParameterSpec(
        name="firstdayleaf",
        kind="int",
        low=float(_DOY_LOW),
        high=float(_DOY_HIGH),
        default=97,
        default_source="code:utci_process.py:67",
        bounds_basis="day of year; leaf-on mask at :708-711 (wraparound "
        "allowed when firstdayleaf > lastdayleaf)",
    ),
    ParameterSpec(
        name="lastdayleaf",
        kind="int",
        low=float(_DOY_LOW),
        high=float(_DOY_HIGH),
        default=300,
        default_source="code:utci_process.py:68",
        bounds_basis="day of year; leaf-on mask at :708-711",
    ),
    ParameterSpec(
        name="elvis",
        kind="int",
        low=0.0,
        high=1.0,
        default=0,
        default_source="code:utci_process.py:63",
        bounds_basis="esky-correction switch (solweig.py:2046); kernel "
        "arg at utci_process.py:825",
    ),
    ParameterSpec(
        name="anisotropic_sky",
        kind="int",
        low=0.0,
        high=1.0,
        default=1,
        default_source="code:utci_process.py:658",
        bounds_basis="Perez diffuse + anisotropic longwave switch; kernel "
        "arg at :827, loop-local assignment at :658",
    ),
)

_SPEC_BY_NAME: dict[str, ParameterSpec] = {
    spec.name: spec for spec in PARAMETER_SPECS
}

#: Reset targets: every safe parameter's documented default, keyed by name.
#: All fourteen are code-cited today (see ``default_source``); a future
#: ``baseline-manifest`` entry must be resolved from the site's baseline
#: manifest at reset time and is refused until a context carries one.
PARAMETER_DEFAULTS: Mapping[str, Any] = MappingProxyType(
    {spec.name: spec.default for spec in PARAMETER_SPECS}
)

#: Executor plumbing tier per parameter (see the module docstring's friction
#: list): ``kernel_arg`` = already forwarded into Solweig_2022a_calc, U-C
#: only threads it through run_utci_window's signature; ``loop_local`` =
#: consumed in the loop prologue, so the body plumbing is needed too.
PLUMBING_CLASSIFICATION: Mapping[str, str] = MappingProxyType(
    {
        "albedo_b": "kernel_arg",
        "ewall": "kernel_arg",
        "absK": "kernel_arg",
        "absL": "kernel_arg",
        "Fside": "kernel_arg",
        "Fup": "kernel_arg",
        "Fcyl": "kernel_arg",
        "cyl": "kernel_arg",
        "elvis": "kernel_arg",
        # NOTE: anisotropic_sky is kernel_arg but ALSO loop-local-assigned
        # at utci_process.py:658 — the U-C override must replace BOTH sites
        # (see the module docstring; u-b-params-reviewer LOW-a).
        "anisotropic_sky": "kernel_arg",
        "transVeg": "loop_local",
        "height": "loop_local",
        "firstdayleaf": "loop_local",
        "lastdayleaf": "loop_local",
    }
)


def _check_defaults_table() -> None:
    """Drift guard: every code-cited default satisfies its own domain.

    Also checks the table covers exactly the registry safe set and that the
    plumbing classification is total, so a registry change that forgets this
    adapter fails at import instead of at a user's edit.
    """
    spec_names = {spec.name for spec in PARAMETER_SPECS}
    if spec_names != set(MODEL_PARAMETERS_SAFE):
        missing = sorted(set(MODEL_PARAMETERS_SAFE) - spec_names)
        extra = sorted(spec_names - set(MODEL_PARAMETERS_SAFE))
        raise EditStateError(
            f"PARAMETER_SPECS must cover the registry safe set exactly; "
            f"missing={missing}, extra={extra}"
        )
    if set(PLUMBING_CLASSIFICATION) != spec_names:
        raise EditStateError(
            "PLUMBING_CLASSIFICATION must classify every documented "
            "parameter"
        )
    for spec in PARAMETER_SPECS:
        if spec.is_baseline_manifest_default:
            continue  # no code constant to check against
        _check_value_in_domain(spec, spec.default, f"default({spec.name})")


def _domain_description(spec: ParameterSpec) -> str:
    if spec.kind == "bool":
        return "a boolean"
    if spec.kind == "int":
        if spec.low == spec.high:
            return f"an integer equal to {int(spec.low)}"
        return f"an integer in [{int(spec.low)}, {int(spec.high)}]"
    if spec.low is not None and spec.high is not None:
        return f"a number in [{spec.low}, {spec.high}]"
    if spec.low is not None:
        bracket = ">" if spec.low_exclusive else ">="
        return f"a number {bracket} {spec.low}"
    if spec.high is not None:
        return f"a number <= {spec.high}"
    return "any finite number"


def _check_value_in_domain(
    spec: ParameterSpec, value: Any, label: str
) -> None:
    """Enforce one value against its documented domain (bounds on new only)."""
    if spec.kind == "bool":
        if not isinstance(value, bool):
            raise EditStateError(
                f"{label}: {spec.name} must be {_domain_description(spec)}, "
                f"got {type(value).__name__}"
            )
        return
    if spec.kind == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            raise EditStateError(
                f"{label}: {spec.name} must be "
                f"{_domain_description(spec)}, got "
                f"{type(value).__name__}"
            )
    else:  # float
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise EditStateError(
                f"{label}: {spec.name} must be a number, got "
                f"{type(value).__name__}"
            )
    number = float(value)
    # Finiteness BEFORE the bound comparisons: NaN fails every relational
    # comparison (nan < low is False, nan > high is False) and +inf passes
    # low-side-only bounds, so without this check both would slip through
    # to the physics handoff (u-b-params-reviewer HIGH, 2026-09-02).
    # stdlib json parses NaN/Infinity literals by default, so a U-D API
    # client can send them; the sibling vegetation standard requires
    # finiteness too (trees.py:71-72, vegetation.py:282).
    if not math.isfinite(number):
        raise EditStateError(
            f"{label}: {spec.name} = {value!r} is not finite; parameter "
            f"values must be finite ({spec.bounds_basis})"
        )
    if spec.low is not None and (
        number < spec.low or (spec.low_exclusive and number == spec.low)
    ):
        bracket = "> " if spec.low_exclusive else ">= "
        raise EditStateError(
            f"{label}: {spec.name} = {value!r} is below the documented "
            f"lower bound ({bracket}{spec.low} required; "
            f"{spec.bounds_basis})"
        )
    if spec.high is not None and number > spec.high:
        raise EditStateError(
            f"{label}: {spec.name} = {value!r} is above the documented "
            f"upper bound {spec.high} ({spec.bounds_basis})"
        )


def check_parameter_value(name: str, value: Any) -> None:
    """Public one-name value-domain fence for a staged model parameter.

    The staging doors the executor's payload rides through —
    :meth:`solweig_gpu.incremental.worker.ExactWorker.stage_model_parameters`
    and the ``model_parameters`` branch of
    :func:`solweig_gpu.incremental.solver.run_full_tile` — fence NAMES and
    scalar KINDS themselves; before u-c8 they trusted every scalar that
    passed the kind check, so ``{albedo_b: True}`` (a bool where a float is
    required), ``{albedo_b: 50.0}`` (outside the documented ``[0, 1]``), and
    ``{transVeg: inf}`` (non-finite) were staged and forwarded into the
    physics (u-c7-review MEDIUM). This wrapper exposes the adapter's own
    domain check (the same :func:`_check_value_in_domain` every internal
    seam runs) so those doors apply the FULL documented domain — type
    spelling, finiteness, bounds — without reaching into adapter internals.

    Raises :class:`~solweig_gpu.incremental.edit_types.EditStateError` for
    an unknown name or an out-of-domain value; the doors re-raise that as
    their own typed error (``WorkerError`` / ``SolverInputError``). The
    check is value-only: it never mutates, never coerces (the dtype hygiene
    at the fold and staging seams keeps its own spelling).
    """
    spec = _SPEC_BY_NAME.get(name)
    if spec is None:
        raise EditStateError(
            f"unknown model parameter {name!r}; accepted names are "
            f"{sorted(_SPEC_BY_NAME)}"
        )
    _check_value_in_domain(spec, value, "model parameter")


def _check_declared_values(
    state: Mapping[str, Any], label: str, *, bounds: bool
) -> dict[str, Any]:
    """Name- and type-check one state mapping (bounds optional)."""
    if not isinstance(state, Mapping):
        raise EditStateError(
            f"{label} must be a mapping of parameter names to values, got "
            f"{type(state).__name__}"
        )
    values: dict[str, Any] = {}
    for name in sorted(state):
        spec = _SPEC_BY_NAME.get(name)
        if spec is None:
            # Unreachable after the registry's unknown-name rejection; kept
            # so the adapter stays safe under a custom registry.
            raise EditStateError(
                f"{label} carries parameter {name!r}, which is not in the "
                f"documented safe set {sorted(_SPEC_BY_NAME)}"
            )
        value = state[name]
        if bounds:
            _check_value_in_domain(spec, value, label)
        else:
            # Type check only (conservative choice 3: old_state is a claim
            # about the current value, not a value the adapter is asserting).
            if spec.kind == "bool":
                if not isinstance(value, bool):
                    raise EditStateError(
                        f"{label}: {name} must be a boolean, got "
                        f"{type(value).__name__}"
                    )
            elif spec.kind == "int":
                if isinstance(value, bool) or not isinstance(value, int):
                    raise EditStateError(
                        f"{label}: {name} must be an integer, got "
                        f"{type(value).__name__}"
                    )
            elif isinstance(value, bool) or not isinstance(
                value, (int, float)
            ):
                raise EditStateError(
                    f"{label}: {name} must be a number, got "
                    f"{type(value).__name__}"
                )
        values[name] = value
    return values


def _changes_from_command(
    command: EditCommand, registry: AdapterRegistry
) -> tuple[ModelParameterChange, ...]:
    """Validate states and build one command's canonical change records."""
    operation = command.operation
    if operation not in _OPERATIONS:
        # The registry check below reports undeclared operations; this guard
        # keeps the shape checks honest for custom registries.
        raise EditStateError(
            f"operation must be one of {sorted(_OPERATIONS)}, got "
            f"{operation!r}"
        )
    old_state = command.old_state
    new_state = command.new_state

    # Registry schema enforcement first (blocked/unknown names) on BOTH
    # states, then the adapter's documented domains.
    for label, state in (("old_state", old_state), ("new_state", new_state)):
        if state is None:
            continue
        registry.validate_edit_state(ADAPTER_ID, dict(state))

    if operation == "update":
        if new_state is None or not dict(new_state):
            raise EditStateError(
                "an 'update' requires a non-empty new_state of parameter "
                "values"
            )
        before = (
            _check_declared_values(old_state, "old_state", bounds=False)
            if old_state is not None
            else {}
        )
        after = _check_declared_values(new_state, "new_state", bounds=True)
    else:  # reset
        if new_state is not None:
            raise EditStateError(
                "a 'reset' must not carry a new_state: after-values are the "
                "documented defaults (PARAMETER_DEFAULTS) and can never be "
                "user-supplied"
            )
        before = (
            _check_declared_values(old_state, "old_state", bounds=False)
            if old_state is not None
            else {}
        )
        targets = sorted(before) or sorted(PARAMETER_DEFAULTS)
        after = {}
        for name in targets:
            spec = _SPEC_BY_NAME[name]
            if spec.is_baseline_manifest_default:
                raise EditStateError(
                    f"parameter {name!r} has no code-cited default; "
                    "reset-to-baseline requires the site's baseline "
                    "manifest, which no SiteContext carries today — "
                    "refusing rather than guessing a value"
                )
            after[name] = spec.default

    # Per-parameter no-ops drop (conservative choice 4); an edit left with
    # no changes is an explicit error, not a silent empty plan. The refusal
    # names every parameter property and both equal values (U-D item d) —
    # an identical-value resubmit must refuse cleanly instead of churning
    # the site-global parameter recompute for a physically identical
    # scene.
    changes = [
        ModelParameterChange(
            name=name,
            before_value=before.get(name),
            after_value=after[name],
        )
        for name in sorted(after)
        if name not in before or before[name] != after[name]
    ]
    if not changes:
        raise EditStateError(
            "edit is a no-op: every targeted parameter already carries the "
            "requested value ("
            + identical_value_clause(
                (name, before[name], after[name]) for name in sorted(after)
            )
            + "); an identical-value resubmit would churn the parameter "
            "recompute closure for a physically identical scene"
        )
    return tuple(changes)


# ---------------------------------------------------------------------------
# Committed deltas -> physics kernel arguments (the U-C executor seam)
# ---------------------------------------------------------------------------


def kernel_arguments_from_deltas(
    deltas: Sequence[ModelParameterDelta],
) -> dict[str, Any]:
    """Fold committed deltas into the flat ``name -> value`` physics mapping.

    Last-write-wins in arrival order (the coalescer's semantics for one
    batch), with every name and value re-checked against the documented
    domain — the executor's inputs are validated on this seam, not only
    inside adapter ``validate()`` calls. Float-kind values are coerced to
    plain ``float`` here: ``np.float64`` is a ``float`` subclass, so it
    passes the domain gate, and forwarding it raw would hand the physics a
    NumPy scalar whose dtype rules can diverge from a plain double — the
    one spelling that keeps parameter differentials bitwise-stable is a
    Python double. Int and bool kinds keep their exact staged value. This
    is the hand-off the U-C integration forwards into
    ``run_utci_window``: per
    :data:`PLUMBING_CLASSIFICATION` the ten ``kernel_arg`` parameters become
    ``Solweig_2022a_calc`` arguments (utci_process.py:824-827) and the four
    ``loop_local`` parameters feed the loop prologue (:649, :708-722). Use
    :func:`before_values_from_deltas` when the publication path also needs
    what each parameter replaced.
    """
    values: dict[str, Any] = {}
    for delta in deltas:
        if not isinstance(delta, ModelParameterDelta):
            raise SourceDeltaError(
                f"expected ModelParameterDelta, got {type(delta).__name__}"
            )
        if delta.adapter_id != ADAPTER_ID:
            raise SourceDeltaError(
                f"delta belongs to adapter {delta.adapter_id!r}, not "
                f"{ADAPTER_ID!r}"
            )
        if delta.source_node_id != MODEL_PARAMETERS_SOURCE_NODE:
            raise SourceDeltaError(
                f"delta targets source node {delta.source_node_id!r}, not "
                f"{MODEL_PARAMETERS_SOURCE_NODE!r}"
            )
        for change in delta.parameters:
            spec = _SPEC_BY_NAME.get(change.name)
            if spec is None:
                if change.name in MODEL_PARAMETERS_BLOCKED:
                    raise SourceDeltaError(
                        f"delta carries parameter {change.name!r}, which is "
                        "BLOCKED (cache-coupled, preprocessing-coupled, or "
                        "dead-branch-activating; lead ruling 2026-09-02)"
                    )
                raise SourceDeltaError(
                    f"delta carries parameter {change.name!r}, which is not "
                    f"in the documented safe set {sorted(MODEL_PARAMETERS_SAFE)}"
                )
            try:
                _check_value_in_domain(spec, change.after_value, "delta")
            except EditStateError as error:
                # The seam validates typed deltas, so its failures are
                # SourceDeltaError like every other check here.
                raise SourceDeltaError(str(error)) from error
            if spec.kind == "float":
                # Fold-seam dtype hygiene (u-c2 review LOW): coerce
                # float-kind values (np.float64 passes the domain gate
                # because it subclasses float) to a plain Python double so
                # a NumPy scalar dtype can never change dtype promotion
                # inside the physics — parameter differentials must stay
                # bitwise-stable. int/bool kinds keep their exact staged
                # value.
                values[change.name] = float(change.after_value)
            else:
                values[change.name] = change.after_value
    return values


def before_values_from_deltas(
    deltas: Sequence[ModelParameterDelta],
) -> dict[str, Any]:
    """First known ``before`` value per parameter (``None`` when unknown).

    The publication/undo path needs what each parameter replaced; the
    executor resolves ``None`` entries from the scenario's parameter store,
    which is the only authoritative source (conservative choice 6).
    """
    values: dict[str, Any] = {}
    for delta in deltas:
        if not isinstance(delta, ModelParameterDelta):
            raise SourceDeltaError(
                f"expected ModelParameterDelta, got {type(delta).__name__}"
            )
        for change in delta.parameters:
            if change.before_value is not None:
                values.setdefault(change.name, change.before_value)
    return values


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------


class ModelReceptorParametersAdapter(EditAdapter):
    """EditAdapter for site-global model/receptor parameters.

    A stateless service: the registry, dependency graph, and safety policy
    are injected at construction, and every protocol method derives its
    result from the command and the
    :class:`~solweig_gpu.incremental.edit_types.SiteContext`. All outputs
    are frozen records, so identical inputs reproduce bitwise-identical
    deltas and plans. The adapter holds no site state and no parameter
    values: the scenario's current values live in the scenario parameter
    store, reached only by the executor.
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
        changes = _changes_from_command(command, self._registry)
        delta = ModelParameterDelta(
            source_node_id=MODEL_PARAMETERS_SOURCE_NODE,
            adapter_id=self.adapter_id,
            parameters=changes,
        )
        return ValidatedEdit(
            command=command,
            adapter_id=self.adapter_id,
            schema_version=self.schema_version,
            source_node_id=MODEL_PARAMETERS_SOURCE_NODE,
            delta=delta,
        )

    # -- protocol: delta, plan, apply, preview, fixtures -----------------------

    def source_delta(
        self, edit: ValidatedEdit, context: SiteContext
    ) -> ModelParameterDelta:
        """Return the delta produced at validation (single derivation)."""
        return self._require_own_delta(edit)

    def impact_plan(
        self, edit: ValidatedEdit, context: SiteContext
    ) -> ImpactPlan:
        """The single-edit impact plan, delegated to the core planner.

        The dirty closure, spatial/temporal scopes, and estimate formulas
        are the planner's (and therefore the engine's): the graph's single
        ``model_parameters -> radiation`` edge makes the closure
        ``radiation -> surface_thermal_state -> tmrt -> utci`` (plus ``wbgt``,
        pruned as never-planned), the family's full-spatial nominal scope
        makes every one of those stages FULL, and every geometry cache
        (``walls``, ``wall_aspect``, the visibilities, ``svf``,
        ``time_shadow``) stays reusable. The returned plan equals the core
        :meth:`EditPlanner.plan` output for ``[edit]`` bitwise. There is
        deliberately no ``read_halo_pixels`` option: a halo only expands
        WINDOWS stages, and this family has none.
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
        via :func:`kernel_arguments_from_deltas` into the physics kernel
        arguments) or rolls back cleanly. Wiring the staged values into
        ``run_utci_window`` is the U-C executor's job — this adapter only
        stages the typed, domain-checked record.
        """
        if not isinstance(delta, ModelParameterDelta):
            raise SourceDeltaError(
                f"expected ModelParameterDelta, got {type(delta).__name__}"
            )
        if delta.adapter_id != self.adapter_id:
            raise SourceDeltaError(
                f"delta belongs to adapter {delta.adapter_id!r}, not "
                f"{self.adapter_id!r}"
            )
        if delta.source_node_id != MODEL_PARAMETERS_SOURCE_NODE:
            raise SourceDeltaError(
                f"delta targets source node {delta.source_node_id!r}, not "
                f"{MODEL_PARAMETERS_SOURCE_NODE!r}"
            )
        # Defense-in-depth at the staging door (u-b-params-reviewer
        # MEDIUM): the planner's validate_delta checks NAMES; a hand-built
        # delta routed straight to apply would otherwise stage
        # out-of-domain after_values unchecked. Re-run the full
        # name+domain validation on the delta payload before it can be
        # committed to a transaction.
        _check_declared_values(
            {change.name: change.after_value for change in delta.parameters},
            f"delta for {delta.source_node_id!r}",
            bounds=True,
        )
        transaction.stage(delta)

    def preview_descriptor(
        self, edit: ValidatedEdit, context: SiteContext
    ) -> PreviewDescriptor:
        """Parameter-scope disclosure with mandatory not-exact caveats.

        The registry kind is ``parameter_scope_disclosure``: there is no
        spatial preview for a site-global scalar, so the preview *is* the
        disclosure — which stages recompute, why they are full-tile, and
        what the preview cannot tell you. It is interaction feedback, never
        scientific output (mission invariant ``preview_is_not_exact``).
        """
        delta = self._require_own_delta(edit)
        names = ", ".join(sorted(change.name for change in delta.parameters))
        return PreviewDescriptor(
            kind=self.metadata.preview,
            immediate=True,
            limitations=(
                "Parameters are site-global scalars: there is no spatial "
                "preview, only this scope disclosure (which stages "
                "recompute and why they are full-tile)",
                f"This edit changes {names}; preview does not recompute "
                "any output — the scientific values come from the solver "
                "job (preview_is_not_exact)",
                "'before' values shown to the user come from the "
                "scenario parameter store at apply time; the delta records "
                "None wherever the caller did not declare the current value",
                "Geometry and view-factor caches stay reusable: the "
                "recompute is downstream of radiation only, so the "
                "disclosure never claims a geometry change",
            ),
        )

    def validation_fixtures(self) -> tuple[str, ...]:
        """Scientific fixture names, verbatim from the registry metadata."""
        return self.metadata.validation_fixtures

    # -- internals ---------------------------------------------------------------

    def _require_own_delta(self, edit: ValidatedEdit) -> ModelParameterDelta:
        if not isinstance(edit.delta, ModelParameterDelta):
            raise SourceDeltaError(
                "edit must carry a ModelParameterDelta, got "
                f"{type(edit.delta).__name__}"
            )
        if edit.delta.adapter_id != self.adapter_id:
            raise SourceDeltaError(
                f"edit delta belongs to adapter {edit.delta.adapter_id!r}"
            )
        return edit.delta


def register_model_parameters_adapter(
    registry: AdapterRegistry | None = None,
) -> AdapterRegistry:
    """Register this adapter (idempotent) and return the registry.

    Kept separate from :func:`vegetation.register_default_adapters` so the
    vegetation helper stays untouched; ``adapters.register_default_adapters``
    composes both.
    """
    target = registry if registry is not None else builtin_registry()
    target.register(ModelReceptorParametersAdapter(registry=target))
    return target


_check_defaults_table()
