# SPDX-License-Identifier: GPL-3.0-only
"""Topological planner: one valid execution plan per heterogeneous batch.

The planner is the only component that turns validated edits into an
:class:`~solweig_gpu.incremental.edit_types.ImpactPlan`. It implements the
runtime sequence "coalesce compatible edits -> source deltas ->
dependency-graph resolution -> per-node spatial/temporal impact ->
local/full safety" from the execution-pipeline spec, and enforces:

- **UEDIT-005** (heterogeneous batches produce ONE valid topological plan):
  deltas from all adapters are coalesced per source node first, the
  dependency closure is resolved once, and ``node_impacts`` is emitted in
  the graph's deterministic topological order;
- **UEDIT-006 shape** (stale scene revisions never plan, so never publish):
  every command's ``base_scene_revision`` must equal the current
  :class:`~solweig_gpu.incremental.edit_graph.SceneGraphState` revision;
- **UEDIT-007 shape** (view-only edits enqueue zero scientific stages): a
  batch whose scientific edits coalesce to nothing — or contains only
  view-only edits — yields a zero-stage plan;
- **UEDIT-010/mission invariant** (dynamic wind is fenced): batches
  containing a ``scientific_extension`` adapter (or touching
  ``wind_coefficients``) are refused unless the planner was constructed
  with ``allow_wind_extension=True`` — never silently planned.

Mixed scopes (U-C ruling, binding): the plan carries per-node write-scope
records (the ``node_impacts`` themselves) plus one transport bounding box.
A node is FULL when any dirty input (transitively, back to a source) is
full-spatial or when the safety policy denies locality — full *for that
node's products only*; coexisting local nodes keep their windows.

Determinism: no wall-clock, no randomness, no iteration over unordered
sets; all collections are sorted or graph-ordered. Identical batches
produce bitwise-identical plans.

Estimate formulas (cheap planning heuristics — NOT measured constants;
calibration belongs to the performance packets)::

    pixels(node)  = grid rows*cols            (FULL)
                  = Σ write-window areas      (WINDOWS)
                  = 0                        (NONE)
    steps(node)   = 1                          (ONE)
                  = time_stop - time_start + 1 (RANGE)
                  = time_stop + 1             (REPLAY with bound)
                  = max(1, len(requested_times), default_temporal_steps)
                                               (ALL / unbounded REPLAY)
    estimated_memory_bytes = Σ layers(node) · pixels(node) · steps(node) · 4
    estimated_work_units   = Σ weight(node) · pixels(node) · steps(node) / 100_000

Conservative choices (ambiguities resolved the safe way, documented here
and in the mission report):

1. A dirty derived node inherits FULL from any dirty parent (transitive);
   there is no "small enough to stay local" exception.
2. A source whose adapter declares a full-spatial nominal scope, or whose
   delta carries no windows, is FULL for its whole downstream share.
3. Policy-denied locality demotes the node to FULL and records the reason
   in ``fallback_reasons`` — nodes are never silently dropped.
4. ``surface_thermal_state`` always replays from timestep 0 when dirty
   (accumulated state cannot be summarized).
5. Time-varying stages narrow to explicit changed/requested times when
   known (ONE/RANGE); otherwise ALL — never a guessed sub-range.
6. Time-invariant geometry stages carry ``TemporalScope.ALL`` (the enum has
   no NONE member); read it as "no temporal restriction".
7. Windows are clamped to the grid; a delta whose windows clamp away to
   nothing is treated as FULL, not as a zero-area job.
8. Unmapped source families (``dem``, wind rebuild) skip the strict
   delta-family check rather than guessing one.
9. ``full_only_initially`` adapters (building, terrain) plan FULL for the
   whole downstream share — no windowed stages exist for them yet
   (walls/aspect are preprocessing; site-wide SVF invalidation).
10. Fenced content in *delta payloads* is rejected exactly like fenced
    command state (``AdapterRegistry.validate_delta``): the executor's
    inputs are schema-checked on every plan, never only inside adapter
    ``validate()`` calls.
11. Output products follow the request: ``wbgt`` is refused when requested
    and pruned-with-reason otherwise (:data:`NEVER_PLANNED_OUTPUTS`);
    explicit requests keep their output ancestors (``utci`` needs
    ``tmrt``) and prune unrelated outputs with a recorded reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

from .edit_graph import EditGraph, NodeKind, SceneGraphState, default_edit_graph
from .edit_registry import (
    METEOROLOGY_SOURCE_NODE,
    MET_UTCI_ONLY_VARIABLES,
    MET_VARIABLE_AFFINITY,
    AdapterRegistry,
    AdapterRegistryError,
    AdapterStatus,
    WIND_SOURCE_NODE,
    builtin_registry,
)
from .edit_types import (
    BuildingMassingDelta,
    ForcingDelta,
    ImpactPlan,
    LandCoverPaintDelta,
    ModelParameterDelta,
    NodeImpact,
    OutputSelectionDelta,
    SiteContext,
    SourceDelta,
    SpatialScope,
    TemporalScope,
    ValidatedEdit,
    VegetationObjectDelta,
    coalesce_source_deltas,
)
from .geometry import RasterGrid, RasterWindow, merge_windows

__all__ = [
    "ConservativeSafetyPolicy",
    "EditPlanner",
    "LocalityDecision",
    "NODE_EXECUTION_COSTS",
    "NodeExecutionCost",
    "PlanningError",
    "SafetyPolicy",
    "TIME_VARYING_NODES",
    "fence_requested_times",
]


class PlanningError(RuntimeError):
    """The batch cannot be planned (stale revision, unknown adapter, fence)."""


def fence_requested_times(
    edits: Sequence[ValidatedEdit], available_times: Sequence[int]
) -> tuple[int, ...]:
    """Refuse out-of-series ``requested_times`` before any planning effects.

    U-C intake (u-b-met LOW): a requested time outside the site's forcing
    series would silently extend REPLAY past the series end. The executor
    always calls this with the site cache's time range before planning;
    :meth:`EditPlanner.plan` also applies it defensively whenever the
    supplied :class:`~solweig_gpu.incremental.edit_types.SiteContext`
    carries non-empty ``available_times``. Raises ``PlanningError``
    listing every offending index — never a silent extension. Returns the
    sorted union of the batch's requested times when they are all in
    series (empty when no edit requests specific times).
    """
    known = tuple(available_times)
    if not known:
        raise PlanningError("available_times must be non-empty to fence requests")
    requested = sorted(
        {
            time
            for edit in edits
            if edit.command.requested_times is not None
            for time in edit.command.requested_times
        }
    )
    known_set = set(known)
    out_of_series = [time for time in requested if time not in known_set]
    if out_of_series:
        raise PlanningError(
            "requested_times outside the site forcing series "
            f"(available 0..{max(known)}): {out_of_series}. The replay set "
            "is never silently extended past the series end "
            "(u-b-met LOW; U-C executor fence)"
        )
    return tuple(requested)


# ---------------------------------------------------------------------------
# Safety policy interface
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LocalityDecision:
    """A safety policy's verdict for one node's windowed recompute."""

    allowed: bool
    reason: str


@runtime_checkable
class SafetyPolicy(Protocol):
    """Decides whether a node may recompute locally.

    Implementations must be deterministic (no clock/randomness): the same
    ``(node_id, windows, dirty_fraction)`` must always yield the same
    verdict and reason string.
    """

    def allows_local(
        self,
        node_id: str,
        windows: tuple[RasterWindow, ...],
        dirty_fraction: float,
    ) -> LocalityDecision:
        ...  # pragma: no cover - protocol body


@dataclass(frozen=True, slots=True)
class ConservativeSafetyPolicy:
    """Default conservative policy reusing the worker's dirty-fraction rule.

    A node stays local only while its dirty fraction stays strictly below
    ``full_recompute_fraction`` (default 0.30, matching
    :func:`solweig_gpu.incremental.geometry.choose_recompute_mode`). The
    adapter/integration packets wire site-specific thresholds and
    solver-capability sets (``full_only_nodes``) through this same type.
    """

    full_recompute_fraction: float = 0.30
    full_only_nodes: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not 0.0 < self.full_recompute_fraction <= 1.0:
            raise ValueError("full_recompute_fraction must be in (0, 1]")
        object.__setattr__(self, "full_only_nodes", frozenset(self.full_only_nodes))

    def allows_local(
        self,
        node_id: str,
        windows: tuple[RasterWindow, ...],
        dirty_fraction: float,
    ) -> LocalityDecision:
        if node_id in self.full_only_nodes:
            return LocalityDecision(
                False, f"node {node_id!r} is forced full by policy configuration"
            )
        if not windows:
            return LocalityDecision(False, "no write windows remain after clamping")
        if dirty_fraction >= self.full_recompute_fraction:
            return LocalityDecision(
                False,
                f"dirty fraction {dirty_fraction:.4f} >= "
                f"{self.full_recompute_fraction:g}",
            )
        return LocalityDecision(
            True,
            f"dirty fraction {dirty_fraction:.4f} < "
            f"{self.full_recompute_fraction:g}",
        )


# ---------------------------------------------------------------------------
# Cost model (planning heuristics only)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NodeExecutionCost:
    """Cheap per-node cost model: float32 layers per step + relative work."""

    float32_layers_per_step: int
    relative_work: float


#: Heuristic execution costs for every computed stage. Sources and views
#: have no stage and no entry. Values are planning constants, deliberately
#: coarse: they rank jobs, they do not predict runtime.
NODE_EXECUTION_COSTS: dict[str, NodeExecutionCost] = {
    "relative_geometry": NodeExecutionCost(2, 1.0),
    "walls": NodeExecutionCost(1, 1.0),
    "wall_aspect": NodeExecutionCost(1, 1.0),
    "building_visibility": NodeExecutionCost(8, 4.0),
    "vegetation_visibility": NodeExecutionCost(8, 4.0),
    "svf": NodeExecutionCost(2, 2.0),
    "solar_atmospheric_state": NodeExecutionCost(1, 0.5),
    "time_shadow": NodeExecutionCost(1, 3.0),
    "radiation": NodeExecutionCost(4, 6.0),
    "surface_thermal_state": NodeExecutionCost(1, 2.0),
    "tmrt": NodeExecutionCost(1, 1.0),
    "utci": NodeExecutionCost(1, 0.5),
    "wbgt": NodeExecutionCost(1, 0.5),
}

#: Stages whose value differs per timestep (everything else is a geometry
#: cache valid for all times).
TIME_VARYING_NODES = frozenset(
    {
        "solar_atmospheric_state",
        "time_shadow",
        "radiation",
        "surface_thermal_state",
        "tmrt",
        "utci",
        "wbgt",
    }
)

#: Stages that accumulate across timesteps (always replay from 0).
STATEFUL_NODES = frozenset({"surface_thermal_state"})

#: Output products the incremental system can never produce. ``wbgt`` is
#: disabled in both the local solve and the full fallback (solver.py:1095,
#: :1191) — the lead ruling is "never promised": requesting it is refused
#: loudly (PlanningError), and it is silently dropped from no plan either.
NEVER_PLANNED_OUTPUTS = frozenset({"wbgt"})

#: Sources whose edits drive *when* downstream stages must recompute.
FORCING_SOURCE_NODES = frozenset({"meteorology", "selected_date_time"})

#: Strict delta-family expectations per source node; unmapped sources
#: (dem, wind rebuild) skip the check (conservative choice 8).
SOURCE_DELTA_FAMILY: dict[str, type[SourceDelta]] = {
    "vegetation_dsm": VegetationObjectDelta,
    "building_dsm": BuildingMassingDelta,
    "landcover": LandCoverPaintDelta,
    "meteorology": ForcingDelta,
    "selected_date_time": ForcingDelta,
    "model_parameters": ModelParameterDelta,
    "output_selection": OutputSelectionDelta,
}

BYTES_PER_LAYER_ELEMENT = 4
WORK_UNIT_PIXEL_STEPS = 100_000


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _NodeScope:
    """Resolved spatial scope for one dirty node during the topo walk."""

    spatial_scope: SpatialScope
    windows: tuple[RasterWindow, ...]


class EditPlanner:
    """Produces exactly one deterministic, topologically ordered plan."""

    def __init__(
        self,
        *,
        grid: RasterGrid,
        graph: EditGraph | None = None,
        registry: AdapterRegistry | None = None,
        policy: SafetyPolicy | None = None,
        allow_wind_extension: bool = False,
        default_temporal_steps: int = 1,
    ) -> None:
        if not isinstance(grid, RasterGrid):
            raise PlanningError("grid must be a RasterGrid")
        self._grid = grid
        self._graph = graph if graph is not None else default_edit_graph()
        self._registry = registry if registry is not None else builtin_registry()
        self._policy: SafetyPolicy = (
            policy if policy is not None else ConservativeSafetyPolicy()
        )
        self._allow_wind_extension = bool(allow_wind_extension)
        if isinstance(default_temporal_steps, bool) or default_temporal_steps < 1:
            raise PlanningError("default_temporal_steps must be >= 1")
        self._default_temporal_steps = int(default_temporal_steps)

    # -- public API -----------------------------------------------------------

    @property
    def graph(self) -> EditGraph:
        return self._graph

    @property
    def registry(self) -> AdapterRegistry:
        return self._registry

    def plan(
        self,
        edits: Sequence[ValidatedEdit],
        state: SceneGraphState,
        context: SiteContext | None = None,
    ) -> ImpactPlan:
        """Plan one heterogeneous batch against the current graph state.

        ``context`` (optional but expected in production) supplies the site
        facts the fences and estimates need: the wind-parity fence
        (``has_wind_coefficients`` — sites with WindCoeff rasters are refused
        until a wind adapter passes validation; the incremental path assumes
        coeff=1, solver.py:1090-1091/:1179) and honest temporal estimates
        (``available_times`` bounds unbounded REPLAY/ALL step counts).
        """
        if not edits:
            raise PlanningError("batch must contain at least one validated edit")
        self._check_batch_invariants(edits, state)
        self._check_context(edits, state, context)
        self._check_registry_facts(edits)
        self._check_wind_fence(edits)

        # -- coalesce per (source node, adapter) in arrival order ----------
        grouped: dict[tuple[str, str], list[SourceDelta]] = {}
        for edit in edits:
            grouped.setdefault((edit.source_node_id, edit.adapter_id), []).append(
                edit.delta
            )
        changed: dict[str, list[SourceDelta]] = {}
        for (node_id, adapter_id) in sorted(grouped):
            coalesced = coalesce_source_deltas(grouped[(node_id, adapter_id)])
            if not coalesced.is_noop:
                changed.setdefault(node_id, []).append(coalesced)

        view_changed = sorted(
            node_id
            for node_id in changed
            if self._graph.kind(node_id) is NodeKind.VIEW
        )
        scientific_sources = sorted(
            node_id
            for node_id in changed
            if self._graph.kind(node_id) is not NodeKind.VIEW
        )
        had_scientific_edits = any(
            self._graph.kind(edit.source_node_id) is not NodeKind.VIEW
            for edit in edits
        )

        all_nodes = set(self._graph.node_ids)
        changed_nodes = set(changed)
        if not scientific_sources:
            # View-only (or fully coalesced no-op) batch: zero scientific
            # stages (UEDIT-007 shape).
            reasons: list[str] = []
            if had_scientific_edits:
                reasons.append(
                    "batch coalesced to a no-op: every scientific edit cancels"
                )
            return ImpactPlan(
                changed_sources=tuple(sorted(changed_nodes)),
                node_impacts=(),
                reusable_nodes=tuple(sorted(all_nodes - changed_nodes)),
                fallback_reasons=tuple(sorted(reasons)),
                estimated_memory_bytes=0,
                estimated_work_units=0.0,
                scene_revision=state.scene_revision,
                transport_window=None,
            )

        # -- spatial scope per changed source --------------------------------
        fallback_reasons: list[str] = []
        resolved: dict[str, _NodeScope] = {}
        for node_id in scientific_sources:
            scope, windows, reasons = self._resolve_source_scope(
                node_id, changed[node_id]
            )
            resolved[node_id] = _NodeScope(scope, windows)
            fallback_reasons.extend(reasons)

        # -- dirty closure + topological walk --------------------------------
        dirty = self._graph.downstream(scientific_sources)
        requested_times = tuple(
            sorted(
                {
                    time
                    for edit in edits
                    if edit.command.requested_times is not None
                    for time in edit.command.requested_times
                }
            )
        )
        explicit_times = self._explicit_forcing_times(scientific_sources, changed)
        if explicit_times and requested_times:
            # Lead ruling R1 (2026-09-02, u-b-met review open question): a
            # forcing edit at t=k must also serve the batch's requested
            # window. The ground-heat/stateful carry crosses k, so serving
            # requested t>k steps from cache would publish carry-stale
            # products under the new scene_revision. The recompute set is
            # the UNION of changed and requested steps; the stateful REPLAY
            # bound and the time-varying bounds both extend to its extremes.
            # Steps outside this union stay cached and unpublished (R2: the
            # tail beyond the service window is a documented limitation;
            # replay-from-0 keeps every later batch consistent).
            explicit_times = tuple(
                sorted(set(explicit_times) | set(requested_times))
            )
        forcing_dirty = self._graph.downstream(
            source
            for source in scientific_sources
            if source in FORCING_SOURCE_NODES
        )

        # -- output selection (M1): produce only what is requested ----------
        requested_outputs = tuple(
            sorted(
                {
                    output
                    for edit in edits
                    for output in edit.command.requested_outputs
                }
            )
        )
        refused = sorted(set(requested_outputs) & NEVER_PLANNED_OUTPUTS)
        if refused:
            raise PlanningError(
                f"outputs {refused} can never be planned: wbgt is disabled "
                "in the incremental system (solver.py:1095, :1191) and is "
                "never promised (lead ruling 2026-09-02); refusing the "
                "request rather than silently dropping it"
            )
        keep_outputs = self._needed_outputs(requested_outputs, all_nodes)
        if not requested_outputs:
            # Nobody requested specific outputs: plan every producible
            # output except the never-planned set (recorded, not silent).
            for output in sorted(NEVER_PLANNED_OUTPUTS & all_nodes):
                fallback_reasons.append(
                    f"output {output}: never planned (disabled in "
                    "incremental; never promised — lead ruling 2026-09-02)"
                )

        # -- met utci-only affinity narrowing (r3a, 2026-09-04) ----------------
        fast_plan = self._met_utci_fast_plan(
            changed,
            scientific_sources,
            explicit_times,
            keep_outputs,
            requested_times,
            all_nodes,
            fallback_reasons,
            state,
            context,
        )
        if fast_plan is not None:
            return fast_plan

        impacts: list[NodeImpact] = []
        for node_id in self._graph.topological_order():
            if node_id not in dirty:
                continue
            if self._graph.kind(node_id) is NodeKind.OUTPUT and (
                node_id not in keep_outputs
            ):
                fallback_reasons.append(
                    f"output {node_id}: not requested in this batch; skipped"
                )
                continue
            if node_id in resolved:
                node_scope = resolved[node_id]
            else:
                node_scope = self._resolve_derived_scope(node_id, dirty, resolved)
            if not self._graph.is_executable(node_id):
                continue
            impact = self._build_impact(
                node_id,
                node_scope,
                dirty,
                resolved,
                node_id in forcing_dirty,
                requested_times,
                explicit_times,
                fallback_reasons,
            )
            resolved[node_id] = _NodeScope(
                impact.spatial_scope, impact.write_windows
            )
            impacts.append(impact)

        # -- transport bbox (one per plan; U-C ruling) ------------------------
        transport = self._transport_window(impacts)

        # -- reusable: everything the closure did not reach -------------------
        reusable = tuple(sorted(all_nodes - dirty - set(view_changed)))

        memory_bytes, work_units = self._estimate(
            impacts, requested_times, context
        )
        return ImpactPlan(
            changed_sources=tuple(sorted(changed_nodes)),
            node_impacts=tuple(impacts),
            reusable_nodes=reusable,
            fallback_reasons=tuple(sorted(fallback_reasons)),
            estimated_memory_bytes=memory_bytes,
            estimated_work_units=work_units,
            scene_revision=state.scene_revision,
            transport_window=transport,
        )

    # -- met utci-only affinity narrowing (r3a) --------------------------------

    def _met_utci_fast_plan(
        self,
        changed: dict[str, list[SourceDelta]],
        scientific_sources: list[str],
        explicit_times: tuple[int, ...],
        keep_outputs: frozenset[str],
        requested_times: tuple[int, ...],
        all_nodes: set[str],
        fallback_reasons: list[str],
        state: SceneGraphState,
        context: SiteContext | None,
    ) -> ImpactPlan | None:
        """Narrow an ALL-``utci_only`` met-only batch to ``{utci @ t}``.

        Eligibility is proven, not assumed (r3a Step 0): the batch changes
        ONLY the meteorology source, every changed variable carries the
        ``utci_only`` affinity (edit_registry.MET_VARIABLE_AFFINITY — it
        enters ONLY the comfort calculation at time t: no radiation term,
        no shadow, no surface thermal state, no geometry), every change is
        a discrete timestep (no whole-series resets), and the batch keeps
        ``utci`` as an output. Anything else keeps today's conservative
        FULL closure.

        Consequences of the proof:

        * the dirty closure collapses to exactly ``{utci @ changed t}`` —
          tmrt/radiation/shadow/thermal state provably consume none of the
          changed inputs, so they stay reusable;
        * requested-but-unchanged timesteps stay cached: unlike R1's union
          rule (radiation_affecting met changes carry stateful ground heat
          across k), a utci_only change carries NOTHING forward, so this
          plan never dirties a requested t it did not change. Those
          timesteps are served by the composition layer from the durable
          result history (the server's compose raises a typed StoreError
          when a requested variable is not covered — never a silent
          stale-or-shifted value); the fast-path patch itself carries only
          the changed timesteps' planes.
        * the write is tile-wide (UTCI's boolean-compaction +
          exp/fractional-pow extent sensitivity pins the evaluation
          extent), so :meth:`ConservativeSafetyPolicy.allows_local` is
          deliberately bypassed: its locality question — "is a windowed
          write cheaper than the full recompute it dirties?" — is
          inapplicable when the affinity proof already shrank the dirty
          closure to one node and the write window cannot be smaller than
          the tile. A WINDOWS impact carrying the full window can never
          arise from the generic path (``_build_impact`` demotes it), so
          the shape itself is the executor-side signature of this proof.
        """
        if scientific_sources != [METEOROLOGY_SOURCE_NODE]:
            return None
        deltas = changed.get(METEOROLOGY_SOURCE_NODE, [])
        if not deltas:
            return None
        changed_variables: set[str] = set()
        changed_times: set[int] = set()
        for delta in deltas:
            if not isinstance(delta, ForcingDelta):
                return None
            for change in delta.changes:
                if MET_VARIABLE_AFFINITY.get(change.variable) != "utci_only":
                    return None
                if change.time_index is None:
                    # Whole-series change: keep the conservative closure.
                    return None
                changed_variables.add(change.variable)
                changed_times.add(change.time_index)
        if "utci" in self._policy.full_only_nodes:
            # M1 (r3a review): full_only_nodes is the documented
            # solver-capability contract — a deployment that pins utci
            # there has declared the fast evaluation must not run, and
            # the policy beats the proof. (The demote-to-FULL heuristic
            # lever, full_recompute_fraction, stays policy-independent
            # as documented above: there is no locality to deny.)
            fallback_reasons.append(
                "met utci-only batch did not narrow: policy pins utci to "
                "full-only (full_only_nodes) — capability contract "
                "overrides the affinity proof"
            )
            return None
        if not changed_variables or "utci" not in keep_outputs:
            # The fast path serves utci; a batch that does not keep it
            # (e.g. a tmrt-only request) keeps today's closure.
            if changed_variables and changed_variables <= MET_UTCI_ONLY_VARIABLES:
                fallback_reasons.append(
                    "met utci-only batch did not narrow: utci is not a kept "
                    "output this batch"
                )
            return None
        times = tuple(sorted(changed_times))
        variables = ", ".join(sorted(changed_variables))
        if len(times) == 1:
            temporal_scope = TemporalScope.ONE
            time_stop: int | None = None
        else:
            temporal_scope = TemporalScope.RANGE
            time_stop = times[-1]
        full = self._grid.full_window
        impact = NodeImpact(
            node_id="utci",
            spatial_scope=SpatialScope.WINDOWS,
            read_windows=(full,),
            write_windows=(full,),
            temporal_scope=temporal_scope,
            time_start=times[0],
            time_stop=time_stop,
            reason=(
                f"met utci_only affinity ({variables}): enters only the "
                f"comfort calculation at t, so the closure narrows to "
                f"utci at changed timesteps {list(times)} (r3a Step 0 "
                "code-proven split; requested-but-unchanged timesteps stay "
                "cached — no stateful carry under affinity)"
            ),
        )
        transport = self._transport_window([impact])
        # Dirty = the changed source, the narrowed node, and the outputs the
        # generic closure would have dirtied but can never serve from cache
        # (wbgt is never planned and never silently reusable).
        dirty = {METEOROLOGY_SOURCE_NODE, "utci"} | (
            NEVER_PLANNED_OUTPUTS & all_nodes
        )
        reusable = tuple(sorted(all_nodes - dirty))
        fallback_reasons.append(
            f"met utci_only affinity ({variables}): radiation/shadow/thermal "
            "closure stays reusable (changed variables enter only the "
            "comfort calculation)"
        )
        memory_bytes, work_units = self._estimate(
            [impact], requested_times, context
        )
        return ImpactPlan(
            changed_sources=(METEOROLOGY_SOURCE_NODE,),
            node_impacts=(impact,),
            reusable_nodes=reusable,
            fallback_reasons=tuple(sorted(fallback_reasons)),
            estimated_memory_bytes=memory_bytes,
            estimated_work_units=work_units,
            scene_revision=state.scene_revision,
            transport_window=transport,
        )

    # -- invariant checks ------------------------------------------------------

    def _check_batch_invariants(
        self, edits: Sequence[ValidatedEdit], state: SceneGraphState
    ) -> None:
        scenarios = {edit.command.scenario_id for edit in edits}
        if len(scenarios) != 1:
            raise PlanningError(
                "all edits in a batch must target one scenario, got "
                f"{sorted(scenarios)}"
            )
        if state.graph is not self._graph:
            raise PlanningError(
                "scene state belongs to a different graph instance"
            )
        for edit in edits:
            base = edit.command.base_scene_revision
            if base != state.scene_revision:
                raise PlanningError(
                    f"edit {edit.command.edit_id!r} targets stale scene revision "
                    f"{base}; current revision is {state.scene_revision} "
                    "(UEDIT-006: stale revisions never publish)"
                )

    def _check_context(
        self,
        edits: Sequence[ValidatedEdit],
        state: SceneGraphState,
        context: SiteContext | None,
    ) -> None:
        """Site-fact fences that need the SiteContext (wind parity, grid)."""
        if context is None:
            return
        if (
            context.grid.rows != self._grid.rows
            or context.grid.cols != self._grid.cols
            or context.grid.pixel_size_m != self._grid.pixel_size_m
        ):
            raise PlanningError(
                "site context grid "
                f"({context.grid.rows}x{context.grid.cols}@"
                f"{context.grid.pixel_size_m}m) does not match planner grid "
                f"({self._grid.rows}x{self._grid.cols}@"
                f"{self._grid.pixel_size_m}m)"
            )
        if context.scene_revision != state.scene_revision:
            raise PlanningError(
                f"site context revision {context.scene_revision} is stale: "
                f"current scene revision is {state.scene_revision} "
                "(UEDIT-006: stale revisions never publish)"
            )
        if context.has_wind_coefficients and not self._allow_wind_extension:
            raise PlanningError(
                "site uses wind-coefficient rasters: the incremental path "
                "assumes coeff=1 (solver.py:1090-1091, :1179) so it is not "
                "oracle-equal on wind-dependent products; planning is "
                "refused until a wind-coefficient adapter passes "
                "independent validation (lead ruling 2026-09-02; UEDIT-010; "
                "never silent)"
            )
        if context.available_times:
            # U-C fence (u-b-met LOW): refuse out-of-series requested times
            # before any staging or solving side effect.
            fence_requested_times(edits, context.available_times)

    def _needed_outputs(
        self, requested_outputs: tuple[str, ...], all_nodes: set[str]
    ) -> frozenset[str]:
        """Which OUTPUT nodes the plan must produce.

        Requested names may be OUTPUT node ids (``utci``, ``tmrt``) or
        producible layer names that are not graph nodes (``shadow``,
        ``kup``, ... — the transport vocabulary of
        ``output_view.producible_incremental``). Non-node layer names are
        accepted but drive no ancestor walk; unknown names (neither a node
        nor producible anywhere in the registry) are a clean PlanningError,
        never an uncontrolled graph lookup. With explicit requests: the
        requested output nodes plus every output that is an ancestor of one
        (requesting ``utci`` needs ``tmrt``). With no requests: every
        output except :data:`NEVER_PLANNED_OUTPUTS`. Only dirty outputs
        outside this set are pruned, each with a recorded reason.
        """
        outputs = {
            node_id
            for node_id in all_nodes
            if self._graph.kind(node_id) is NodeKind.OUTPUT
        }
        producible: set[str] = set()
        for metadata in self._registry.all_metadata():
            producible.update(metadata.producible_incremental)
        unknown = sorted(
            name
            for name in requested_outputs
            if name not in all_nodes and name not in producible
        )
        if unknown:
            raise PlanningError(
                f"unknown requested outputs {unknown}; requestable names are "
                f"output nodes {sorted(outputs)} or producible layers "
                f"{sorted(producible)}"
            )
        if not requested_outputs:
            return frozenset(outputs - NEVER_PLANNED_OUTPUTS)
        node_requests = {name for name in requested_outputs if name in outputs}
        keep = set(node_requests)
        seen: set[str] = set()
        stack = sorted(node_requests)
        while stack:
            node_id = stack.pop()
            for parent in self._graph.parents(node_id):
                if parent in seen:
                    continue
                seen.add(parent)
                if parent in outputs:
                    keep.add(parent)
                stack.append(parent)
        return frozenset(keep & outputs) - NEVER_PLANNED_OUTPUTS

    def _check_registry_facts(self, edits: Sequence[ValidatedEdit]) -> None:
        for edit in edits:
            try:
                metadata = self._registry.get_metadata(edit.adapter_id)
                self._registry.validate_operation(
                    edit.adapter_id, edit.command.operation
                )
            except AdapterRegistryError as error:
                # Registry failures surface as planning failures: the batch
                # never reaches execution (UEDIT-001/002 shape).
                raise PlanningError(str(error)) from error
            # Command states AND the delta payload the executor consumes are
            # both schema-checked here: the fences are enforced on every
            # production path, never only inside adapter validate() calls.
            try:
                for role, command_state in (
                    ("old_state", edit.command.old_state),
                    ("new_state", edit.command.new_state),
                ):
                    if command_state is not None:
                        self._registry.validate_edit_state(
                            edit.adapter_id, dict(command_state)
                        )
                self._registry.validate_delta(edit.adapter_id, edit.delta)
            except AdapterRegistryError as error:
                raise PlanningError(
                    f"edit {edit.command.edit_id!r}: {error}"
                ) from error
            if edit.source_node_id not in metadata.source_nodes:
                raise PlanningError(
                    f"adapter {edit.adapter_id!r} does not own source node "
                    f"{edit.source_node_id!r} (owns {list(metadata.source_nodes)})"
                )
            if not self._graph.contains(edit.source_node_id):
                raise PlanningError(
                    f"source node {edit.source_node_id!r} is not in the "
                    "dependency graph (UEDIT-002)"
                )
            expected = SOURCE_DELTA_FAMILY.get(edit.source_node_id)
            if expected is not None and not isinstance(edit.delta, expected):
                raise PlanningError(
                    f"source node {edit.source_node_id!r} requires "
                    f"{expected.__name__}, got {type(edit.delta).__name__}"
                )

    def _check_wind_fence(self, edits: Sequence[ValidatedEdit]) -> None:
        if self._allow_wind_extension:
            return
        for edit in edits:
            try:
                metadata = self._registry.get_metadata(edit.adapter_id)
            except AdapterRegistryError as error:
                raise PlanningError(str(error)) from error
            if edit.source_node_id == WIND_SOURCE_NODE:
                raise PlanningError(
                    "wind-coefficient changes are fenced: dynamic wind from "
                    "geometry requires separate scientific validation "
                    "(UEDIT-010; never silent)"
                )
            if metadata.status is AdapterStatus.SCIENTIFIC_EXTENSION:
                raise PlanningError(
                    f"adapter {edit.adapter_id!r} is a scientific extension: "
                    "planning is fenced until it passes independent "
                    "validation (UEDIT-010; never silent)"
                )

    # -- scope resolution --------------------------------------------------------

    def _resolve_source_scope(
        self, node_id: str, deltas: Sequence[SourceDelta]
    ) -> tuple[SpatialScope, tuple[RasterWindow, ...], list[str]]:
        """Spatial scope for one changed source node.

        ``full_only_initially`` adapters (building, terrain) are FULL
        regardless of their nominal scope: their local semantics have no
        in-process implementation yet (walls/aspect are preprocessing;
        building edits invalidate site-wide SVF scalars and the amaxvalue
        guard), so planning anything windowed for them would emit stages
        the executor cannot honour (mission invariant: unsafe locality
        falls back to full). Full-spatial adapters (forcing, time,
        parameters, terrain) and window-less deltas are FULL; otherwise the
        merged, grid-clamped delta windows.
        """
        reasons: list[str] = []
        metadatas = [
            self._registry.get_metadata(delta.adapter_id) for delta in deltas
        ]
        initial_only = sorted(
            {
                metadata.id
                for metadata in metadatas
                if metadata.status is AdapterStatus.FULL_ONLY_INITIALLY
            }
        )
        if initial_only:
            reasons.append(
                f"source {node_id}: adapter(s) {', '.join(initial_only)} are "
                "full_only_initially (no validated local implementation; "
                "preprocessing/site-wide invalidation) — full tile"
            )
            return SpatialScope.FULL, (), reasons
        full_adapters = sorted(
            {
                metadata.id
                for metadata in metadatas
                if metadata.is_full_spatial
            }
        )
        if full_adapters:
            reasons.append(
                f"source {node_id}: full-spatial adapter "
                f"{', '.join(full_adapters)}"
            )
            return SpatialScope.FULL, (), reasons
        windows: list[RasterWindow] = []
        for delta in deltas:
            for window in delta.spatial_windows:
                clamped = window.clamp(rows=self._grid.rows, cols=self._grid.cols)
                if not clamped.is_empty:
                    windows.append(clamped)
        if not windows:
            reasons.append(
                f"source {node_id}: no local windows declared; conservative full"
            )
            return SpatialScope.FULL, (), reasons
        merged = tuple(merge_windows(windows))
        return SpatialScope.WINDOWS, merged, reasons

    def _resolve_derived_scope(
        self,
        node_id: str,
        dirty: frozenset[str],
        resolved: dict[str, _NodeScope],
    ) -> _NodeScope:
        """A dirty derived/stateful/output node inherits from dirty inputs.

        FULL wins (conservative choice 1): any full-spatial dirty parent
        makes this node full for its products; windowed parents merge
        their write windows into this node's candidate windows.
        """
        dirty_parents = sorted(
            parent for parent in self._graph.parents(node_id) if parent in dirty
        )
        full_parents = [
            parent
            for parent in dirty_parents
            if resolved[parent].spatial_scope is SpatialScope.FULL
        ]
        if full_parents:
            return _NodeScope(SpatialScope.FULL, ())
        windows: list[RasterWindow] = []
        for parent in dirty_parents:
            windows.extend(resolved[parent].windows)
        merged = tuple(merge_windows(windows)) if windows else ()
        if not merged:
            # A dirty node with no dirty windowed inputs (should not happen
            # given FULL propagation, but never emit an empty WINDOWS stage).
            return _NodeScope(SpatialScope.FULL, ())
        return _NodeScope(SpatialScope.WINDOWS, merged)

    def _build_impact(
        self,
        node_id: str,
        node_scope: _NodeScope,
        dirty: frozenset[str],
        resolved: dict[str, _NodeScope],
        is_forcing_driven: bool,
        requested_times: tuple[int, ...],
        explicit_times: tuple[int, ...],
        fallback_reasons: list[str],
    ) -> NodeImpact:
        dirty_parents = sorted(
            parent for parent in self._graph.parents(node_id) if parent in dirty
        )
        scope = node_scope.spatial_scope
        windows = node_scope.windows
        if scope is SpatialScope.WINDOWS:
            dirty_fraction = (
                sum(window.area for window in windows) / self._grid.area_pixels
            )
            decision = self._policy.allows_local(node_id, windows, dirty_fraction)
            if not decision.allowed:
                fallback_reasons.append(
                    f"node {node_id}: demoted to full by safety policy "
                    f"({decision.reason})"
                )
                scope = SpatialScope.FULL
                windows = ()
                reason = f"full: policy denied local ({decision.reason})"
            else:
                reason = (
                    f"windows via {','.join(dirty_parents)}; "
                    f"policy: {decision.reason}"
                )
        else:
            cause_parts = [
                parent
                for parent in dirty_parents
                if resolved[parent].spatial_scope is SpatialScope.FULL
            ]
            cause = (
                f"upstream {', '.join(cause_parts)} is full-spatial"
                if cause_parts
                else "full-spatial source"
            )
            reason = f"full: {cause}"

        temporal_scope, time_start, time_stop = self._temporal_scope(
            node_id, is_forcing_driven, requested_times, explicit_times
        )
        read_windows = windows if scope is SpatialScope.WINDOWS else ()
        return NodeImpact(
            node_id=node_id,
            spatial_scope=scope,
            read_windows=read_windows,
            write_windows=windows,
            temporal_scope=temporal_scope,
            time_start=time_start,
            time_stop=time_stop,
            reason=reason,
        )

    def _temporal_scope(
        self,
        node_id: str,
        is_forcing_driven: bool,
        requested_times: tuple[int, ...],
        explicit_times: tuple[int, ...],
    ) -> tuple[TemporalScope, int | None, int | None]:
        """Temporal scope per conservative choices 4-6 (see module notes)."""
        if node_id in STATEFUL_NODES:
            stop = None
            if explicit_times:
                stop = max(explicit_times)
            elif requested_times:
                stop = max(requested_times)
            return TemporalScope.REPLAY, 0, stop
        if node_id not in TIME_VARYING_NODES:
            return TemporalScope.ALL, None, None
        bounds: tuple[int, int] | None = None
        if is_forcing_driven and explicit_times:
            bounds = (min(explicit_times), max(explicit_times))
        elif requested_times:
            bounds = (min(requested_times), max(requested_times))
        if bounds is None:
            return TemporalScope.ALL, None, None
        start, stop = bounds
        if start == stop:
            return TemporalScope.ONE, start, None
        return TemporalScope.RANGE, start, stop

    # -- estimates ---------------------------------------------------------------

    def _explicit_forcing_times(
        self, scientific_sources: Sequence[str], changed: dict[str, list[SourceDelta]]
    ) -> tuple[int, ...]:
        indices: set[int] = set()
        for node_id in scientific_sources:
            if node_id not in FORCING_SOURCE_NODES:
                continue
            for delta in changed[node_id]:
                if isinstance(delta, ForcingDelta):
                    indices.update(delta.time_indices)
        return tuple(sorted(indices))

    def _estimate(
        self,
        impacts: Sequence[NodeImpact],
        requested_times: tuple[int, ...],
        context: SiteContext | None = None,
    ) -> tuple[int, float]:
        # Available site timesteps bound unbounded REPLAY/ALL estimates so
        # they never understate by ignoring the series length (M3): the
        # estimate may overstate, never understate.
        available_steps = (
            len(context.available_times)
            if context is not None and context.available_times
            else 0
        )
        memory_bytes = 0
        work_units = 0.0
        for impact in impacts:
            if impact.spatial_scope is SpatialScope.NONE:
                continue
            cost = NODE_EXECUTION_COSTS.get(impact.node_id)
            if cost is None:
                raise PlanningError(
                    f"no execution cost model for node {impact.node_id!r}"
                )
            if impact.spatial_scope is SpatialScope.FULL:
                pixels = self._grid.area_pixels
            else:
                pixels = sum(window.area for window in impact.write_windows)
            if impact.temporal_scope is TemporalScope.ONE:
                steps = 1
            elif impact.temporal_scope is TemporalScope.RANGE:
                assert impact.time_start is not None and impact.time_stop is not None
                steps = impact.time_stop - impact.time_start + 1
            elif impact.temporal_scope is TemporalScope.REPLAY:
                steps = (
                    impact.time_stop + 1
                    if impact.time_stop is not None
                    else max(
                        1,
                        len(requested_times),
                        self._default_temporal_steps,
                        available_steps,
                    )
                )
            else:  # ALL
                steps = max(
                    1,
                    len(requested_times),
                    self._default_temporal_steps,
                    available_steps,
                )
            memory_bytes += cost.float32_layers_per_step * pixels * steps * BYTES_PER_LAYER_ELEMENT
            work_units += (
                cost.relative_work * pixels * steps / WORK_UNIT_PIXEL_STEPS
            )
        return memory_bytes, work_units

    def _transport_window(
        self, impacts: Sequence[NodeImpact]
    ) -> RasterWindow | None:
        union = RasterWindow(0, 0, 0, 0)
        found = False
        for impact in impacts:
            for window in impact.write_windows:
                union = union.union(window)
                found = True
        return union if found else None
