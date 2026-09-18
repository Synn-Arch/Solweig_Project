# SPDX-License-Identifier: GPL-3.0-only
"""Plan executor: ``ImpactPlan`` -> real recomputes -> published results.

This module closes the loop the universal editing engine left open: the
planner produces a valid :class:`~solweig_gpu.incremental.edit_types.ImpactPlan`,
the adapters stage typed deltas in a rollback-safe
:class:`~solweig_gpu.incremental.edit_types.ScenarioTransaction`, and the
executor binds the two to the mission-validated scientific machinery
(:class:`~solweig_gpu.incremental.worker.ExactWorker`,
:func:`~solweig_gpu.incremental.solver.solve_window`,
:func:`~solweig_gpu.incremental.solver.run_full_tile`,
:mod:`solweig_gpu.incremental.result`) so a plan becomes published,
checksummed patches plus :mod:`solweig_gpu.incremental.store` entries keyed
``(node_id, time_index)``.

What the executor owns (packet u-c1):

(a) accepts ANY valid ``ImpactPlan``, including zero-output-node-impact
    plans (non-node producible layer requests such as a shadow-only batch
    ride the same transport plumbing; NEW-1 reviewer contract note);
(b) publishes/looks up/invalidates results per ``(node, time)`` through
    :class:`~solweig_gpu.incremental.store.TemporalResultStore` (M2);
(c) ALWAYS binds a :class:`~solweig_gpu.incremental.edit_types.SiteContext`
    (available times, grid, scene revision, wind-coefficient detection) and
    plans with ``default_temporal_steps=24`` so estimates never understate
    a diurnal series (M3 residual);
(d) fences ``requested_times`` to the site's available times BEFORE
    planning (u-b-met LOW) — a clean
    :class:`~solweig_gpu.incremental.planner.PlanningError`, never a silent
    replay extension past the series end;
(e) read-halo wiring (L2): staged reads are inflated by the
    adapter-declared read windows (``NodeImpact.read_windows``, the
    ``read_halo_pixels`` the adapter baked in at plan time) via
    :func:`stage_read_window`; writes are NEVER inflated by a read halo;
(f) scene-revision discipline: a plan executes only while its
    ``scene_revision`` is current (stale plans are refused outright), and
    the batch's results publish at the revision the batch CREATES
    (``plan.scene_revision + 1`` — the worker's own bump-then-publish
    convention; publishing the edited scene under the base revision would
    relabel the baseline revision). A failed or superseded execution rolls
    the scene state and the layer replay back wholesale, so a revision
    number never advances without its published results (oracle
    invariant).

Execution model (this packet): vegetation batches run against the EXISTING
windowed path — the executor replays the committed coalesced deltas into
the scenario's
:class:`~solweig_gpu.incremental.trees.TreeLayer` (public mutators only, so
the layer's own sequence counter and the worker's published-watermark stay
authoritative) and then drives
:class:`~solweig_gpu.incremental.worker.ExactWorker.run`.  Meteorology
batches (u-c3) carry no tree edits: their committed forcing deltas fold
into the scenario overlay staged on the worker, whose pending-overlay flag
makes the job a full-tile recompute (forcing is site-global), and the
accumulated overlay commits only on publication.  Land-cover batches
(u-c4) are the windowed mirror: their committed paint deltas fold into the
scenario :class:`~solweig_gpu.incremental.adapters.landcover.LandCoverOverlay`
staged on the worker, whose pending flag joins the patch footprints to the
job's dirty windows, and the accumulated overlay commits only on
publication.  Model-parameter batches (u-c7) fold their committed deltas
over the accumulated parameter state into the flat kernel-argument mapping
staged on the worker — site-global like forcing, so the pending payload
routes the job full-tile, and the accumulated mapping commits only on
publication.  Building geometry (u-d4c) ACCUMULATES like every other
family: the executor keeps the per-building-id fold of every massing edit
committed by a PUBLISHED batch, and while that fold is non-empty EVERY
batch — building-only, mixed, or other-family-only — routes through the
building regeneration chain staged from BASELINE with the FULL fold (the
chain is the only solve path whose site/cache carry the edited massing;
the baseline-bound worker would silently revert the buildings).  A batch
whose fold is empty routes the worker exactly as before (routing identity
for building-free scenarios).  Every other source family raises
:class:`NotIntegratedError` (clean, typed, naming the packet that wires
it).

SEAM SIGNATURES for the remaining U-C packets (each plugs into exactly one
hook below; nothing else in this file needs to change):

- **u-c2** (model-parameter physics plumbing into ``run_utci_window``) —
  LANDED (u-c7 wiring)::

      PlanExecutor._physics_kernel_arguments(
          committed: Mapping[str, tuple[SourceDelta, ...]]
      ) -> dict[str, Any] | None

  wired from ``adapters.model_parameters.kernel_arguments_from_deltas`` (10
  ``kernel_arg`` parameters -> ``Solweig_2022a_calc`` call arguments,
  utci_process.py:823-828; 4 ``loop_local`` parameters feed the loop
  prologue). The batch's committed parameter deltas fold over the
  ACCUMULATED published parameter state (last-write-wins per name — the
  parameters are time-invariant, so there is no time dimension), the fold
  re-validates every name/domain and float()-coerces float-kind values at
  the seam (fail-fast before any state advances), and the mapping rides the
  job via ``ExactWorker.stage_model_parameters``: a staged payload that
  differs from the published one marks the job pending and routes it
  full-tile (the planner scopes the params dirty set FULL — the parameters
  are site-global), and both solve paths forward it into
  ``run_utci_window``. ``run_full_tile`` hands a non-empty mapping to its
  direct-path replication of the ``compute_utci`` core (the orchestrator
  itself has no parameter seam), and the building regeneration chain
  receives the same mapping so a mixed building+params batch never drops
  the parameter edits. ``None`` (never set) is the byte-identical
  parameter-free run.

- **u-c3** (meteorological overlay into the worker) — LANDED::

      PlanExecutor._scenario_forcing(
          committed: Mapping[str, tuple[SourceDelta, ...]]
      ) -> SiteForcing

  wired from ``adapters.met_time.overlay_from_deltas`` ->
  ``solver.load_site_forcing(cache, site_dir=..., selected_date_str=...,
  overlay=overlay)``. The batch's committed forcing deltas fold
  last-write-wins on top of the scenario's ACCUMULATED overlay (every
  forcing change already published), the seam materializes and
  re-verifies the resolved table before any state advances, and the same
  overlay rides the job via ``ExactWorker.stage_forcing_overlay`` (the
  worker's ``forcing()`` is the single materialization point for both the
  local and the full solve; a forcing-only batch has no tree edits, so the
  worker's staged-overlay pending flag routes it to a full-tile recompute
  — forcing is site-global). ``run_full_tile`` stages the RESOLVED table
  in its scratch site, never the baseline met text.

- **u-c4** (landcover overlay wiring) — LANDED::

      PlanExecutor._stage_overlays(
          committed: Mapping[str, tuple[SourceDelta, ...]]
      ) -> LandCoverOverlay | None

  wired from ``adapters.landcover.landcover_overlay_from_deltas``; the
  overlay materializes at the solver seam (``solve_window``'s class-grid
  read and the full-tile fallback's scratch Landcover staging, both via
  ``solver.resolve_landcover_overlay``; Tgmaps_v1.py:15-53 consumes the
  class grid unchanged). ``stage_read_window`` already accepts the
  landcover adapter's inflated ``read_windows`` (its
  ``read_halo_pixels``), and the solver resolves against the FULL baseline
  grid before slicing, so reads outside the write window but inside the
  halo see painted cells exactly like the oracle.

- **u-c5** (building rasterizer + SVF regeneration chain) — LANDED (u-c6)::

      PlanExecutor._regenerate_building_chain(
          edits: Sequence[MassingEdit],
          *,
          target_revision: int,
          forcing_overlay: ForcingOverlay | None,
          landcover_overlay: LandCoverOverlay | None,
      ) -> JobOutcome

  wired from ``adapters.building.massing_edits_from_deltas`` ->
  ``regenerate.regenerate_building_batch`` (rasterizer, walls/aspect, SVF,
  cache rebuild, full-tile solve). Building batches stay FULL-scope per
  the planner's ``full_only_initially`` ruling: the whole scenario chain
  regenerates under the executor's scratch root and the full-tile results
  publish through the same two-phase patch discipline the worker uses
  (stage -> rename into ``results/<scenario>/rev-{N}-<job>``), so the
  u-c1b publication contract holds unchanged. The scenario's CURRENT
  state rides the regeneration: the live tree layer, the effective
  forcing overlay, and the effective land-cover overlay (the land-cover
  seam is threaded through ``run_scenario_full_tile`` ->
  ``run_full_tile(landcover_overlay=...)``; a mixed building+paint batch
  never silently drops the paint). u-d4c: the chain's edit set is the
  executor's ACCUMULATED building fold (every massing edit committed by a
  PUBLISHED batch, folded per building id) — not just the current batch's
  edits — so a batch after a published building edit (any family) can
  never silently revert it; see :attr:`PlanExecutor._applied_massing_edits`.

Determinism: no wall clock reads in any decision path (job ids and patch
timestamps live in provenance records only); all collections are sorted or
graph-ordered; identical batches execute identically.
"""

from __future__ import annotations

import inspect
import logging
import shutil
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from uuid import uuid4

import numpy as np

from solweig_gpu.utci_process import map_windcoeff_files_by_key

from .adapters import register_default_adapters
from .adapters.building import (
    BUILDING_SOURCE_NODE,
    MassingEdit,
    massing_edits_from_deltas,
)
from .adapters.landcover import (
    LandCoverOverlay,
    coalesce_overlay,
    landcover_overlay_from_deltas,
    overlay_diff_windows,
)
from .adapters.met_time import ForcingOverlay, overlay_from_deltas
from .adapters.model_parameters import kernel_arguments_from_deltas
from .adapters.vegetation import (
    VegetationGeometryAdapter,
    coalesced_batch_from_deltas,
)
from .cache import SiteCache
from .edit_graph import EditGraph, NodeKind, SceneGraphState, default_edit_graph
from .edit_registry import (
    LANDCOVER_SOURCE_NODE,
    METEOROLOGY_SOURCE_NODE,
    MET_UTCI_ONLY_VARIABLES,
    MODEL_PARAMETERS_SOURCE_NODE,
    AdapterRegistry,
    builtin_registry,
)
from .edit_types import (
    EditCommand,
    ForcingChange,
    ForcingDelta,
    ImpactPlan,
    LandCoverPaintDelta,
    LandCoverPaintPatch,
    ModelParameterDelta,
    ScenarioTransaction,
    SiteContext,
    SourceDelta,
    SourceDeltaError,
    SpatialScope,
    ValidatedEdit,
    VegetationObjectDelta,
    coalesce_source_deltas,
)
from .edits import TreeEdit
from .geometry import (
    InfluenceConfig,
    RasterGrid,
    RasterWindow,
    merge_windows,
)
from .planner import (
    ConservativeSafetyPolicy,
    EditPlanner,
    PlanningError,
    fence_requested_times,
)
from .regenerate import RegenerationError, regenerate_building_batch
from .result import (
    SUPPORTED_VARIABLES,
    ResultPatch,
    discard_staging,
    load_patch,
    load_patch_metadata,
    new_job_id,
    publish_staged_patch,
    stage_patch,
)
from .solver import (
    SiteForcing,
    compose_full_scene_tensors,
    load_site_forcing,
    resolve_landcover_overlay,
    solve_window,
)
from .store import (
    CoverageStatus,
    TemporalResultEntry,
    TemporalResultKey,
    TemporalResultStore,
)
from .checkpoints import (
    THERMAL_TENSOR_NAMES,
    CheckpointError,
    coverage_checkpoint,
    list_checkpoints,
    load_checkpoint,
    thermal_checkpoint,
    write_checkpoint,
)
from .trees import TreeLayer, validate_tree_spec
from .worker import (
    ExactWorker,
    JobOutcome,
    PublicationWatermarks,
    SupersededJobError,
    discard_published_patches,
)
from ..utci_process import recompute_utci_steps

__all__ = [
    "DEFAULT_TEMPORAL_STEPS",
    "EXECUTABLE_SOURCE_NODES",
    "ExecutedPlan",
    "ExecutorError",
    "MetFastPathRefused",
    "NotIntegratedError",
    "PlanExecutor",
    "VARIABLE_TO_RESULT_NODE",
    "detect_wind_coefficients",
    "stage_read_window",
]

#: u-d1 fence (U-D intake item c): the building regeneration chain's own
#: validated variable vocabulary, read from ``regenerate_building_batch``'s
#: signature default — the single source of truth, so widening the chain
#: widens the fence with it. The executor's transport vocabulary
#: (``worker.requested_variables``) must stay inside this set: a future
#: widening of the patch transport without widening the chain would
#: otherwise silently ask the regeneration for variables it cannot
#: produce (executor seam, ~this module's ``_regenerate_building_chain``).
_BUILDING_CHAIN_VARIABLES = frozenset(
    inspect.signature(regenerate_building_batch).parameters[
        "requested_variables"
    ].default
)

#: Best-effort orphan cleanup (u-c1b M4) reports here when a failed
#: batch's patch directory survives the rollback.
_LOG = logging.getLogger(__name__)

#: M3 residual: a diurnal series floor. When a context carries no
#: available_times the planner's ALL/unbounded-REPLAY estimates use this
#: many steps instead of 1, so estimates never understate a real site.
DEFAULT_TEMPORAL_STEPS = 24

#: Source families this packet can execute end-to-end: vegetation through
#: the existing windowed worker path, meteorology through the u-c3
#: forcing-overlay seam (committed forcing deltas fold into the scenario
#: overlay the worker materializes via ``load_site_forcing(overlay=...)``;
#: a forcing-only batch has no tree edits, so the worker's staged-overlay
#: pending flag routes it to a full-tile recompute — forcing is
#: site-global), land cover through the u-c4 paint-overlay seam (committed
#: paint deltas fold, per-cell last-write-wins at commit, into the
#: :class:`~solweig_gpu.incremental.adapters.landcover.LandCoverOverlay`
#: the worker stages; a paint-only batch recomputes the batch's actual
#: dirty footprint — the STROKE windows union the value symmetric
#: difference vs the published overlay, a superset of the plan's
#: stroke-footprint promise (u-c6b) that still excludes every prior
#: published footprint — as windowed dirty work), buildings through the
#: u-c5 regeneration
#: chain (u-c6 wiring): committed massing edits regenerate the whole
#: scenario chain and publish FULL-tile results at ``plan.scene_revision
#: + 1``, and model parameters through the u-c7 kernel-argument seam:
#: committed parameter deltas fold over the scenario's accumulated
#: parameter state (last-write-wins per name; the parameters are
#: time-invariant, so there is no time dimension to the payload) into the
#: flat mapping the worker stages site-globally — a params-only batch
#: routes full-tile, exactly the plan's FULL dirty set. Everything else
#: raises :class:`NotIntegratedError` naming the
#: packet that wires it.
EXECUTABLE_SOURCE_NODES = frozenset(
    {
        "vegetation_dsm",
        "meteorology",
        "landcover",
        "building_dsm",
        MODEL_PARAMETERS_SOURCE_NODE,
    }
)

#: Where each unintegrated family lands.
NOT_INTEGRATED_PACKETS: dict[str, str] = {
    "selected_date_time": (
        "no selected_date_time adapter exists yet (the registry declares "
        "the source; its EditAdapter lands with the U-D API intake)"
    ),
    "dem": "no terrain delta family exists yet (U-C intake L1 pending)",
}

#: Patch variable -> store node id. Published patches carry layer names
#: (``shadow``); the store and the plans speak graph node ids
#: (``time_shadow``). utci/tmrt are both.
VARIABLE_TO_RESULT_NODE: dict[str, str] = {
    "utci": "utci",
    "tmrt": "tmrt",
    "shadow": "time_shadow",
}


class ExecutorError(RuntimeError):
    """The plan could not be executed (inconsistent state, uncovered scope)."""


class MetFastPathRefused(ExecutorError):
    """The met utci-only fast path declined this batch (r3a).

    Never raised past :meth:`PlanExecutor.execute_plan` for a healthy
    batch: a refusal is a ROUTING decision — the batch falls back to
    today's exact full solve, with the reason attached to the fallback
    outcome's diagnostics as ``met_fast_path_refusal`` (typed and
    telemetry-visible, never silently wrong). Subclassing
    :class:`ExecutorError` means an unexpected escape (a refusal leaking
    where no fallback can run) is still a loud failure.
    """


class NotIntegratedError(ExecutorError):
    """A source family's scientific path lands in a later U-C packet."""

    def __init__(self, source_node: str, packet: str) -> None:
        self.source_node = source_node
        self.packet = packet
        super().__init__(
            f"source family {source_node!r} has no integrated execution "
            f"path yet; it lands in {packet}. Stage the batch without it or "
            "wait for that packet (never silently skipped)"
        )


# ---------------------------------------------------------------------------
# Site facts
# ---------------------------------------------------------------------------


def detect_wind_coefficients(
    site_dir: str | Path, tile_key: str
) -> bool:
    """Whether the site carries WindCoeff rasters for ``tile_key``.

    Oracle parity: ``compute_utci`` maps ``WindCoeff_dirXXX_{tile}.tif``
    files with ``map_windcoeff_files_by_key`` (utci_process.py:174-215).
    The incremental path assumes coeff=1 (solver.py:1090-1091, :1179), so a
    wind-coefficient site must be fenced, and the fence needs this exact
    detection — the cache manifest has no wind entry (checked; the manifest
    schema rejects unknown keys), so the oracle's own mapping is the source
    of truth. Searched directories: ``site_dir`` and ``site_dir/WindCoeff``.
    """
    for directory in (Path(site_dir), Path(site_dir) / "WindCoeff"):
        mapping = map_windcoeff_files_by_key(directory)
        if tile_key in mapping and mapping[tile_key]:
            return True
    return False


def stage_read_window(
    write_window: RasterWindow,
    declared_read_windows: Sequence[RasterWindow],
    *,
    rows: int,
    cols: int,
) -> RasterWindow:
    """Staged read extent for one write window (L2 read-halo wiring).

    The staged read is the write window inflated by the adapter-declared
    read halo — i.e. the union of ``declared_read_windows`` (the
    ``NodeImpact.read_windows`` the adapter expanded by its
    ``read_halo_pixels`` at plan time), clamped to the grid. It is a
    *floor*: the solver's exact-influence window
    (``read_window_for_write_window``) is unioned on top by the solve path,
    and may only grow. Writes are NEVER inflated here — this function
    returns a read window and touches no write scope.
    """
    staged = write_window
    for declared in declared_read_windows:
        staged = staged.union(declared)
    return staged.clamp(rows=rows, cols=cols)


# ---------------------------------------------------------------------------
# Execution outcome
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExecutedPlan:
    """What the executor did with one plan."""

    plan: ImpactPlan
    status: str  # "published" | "no-op" | "superseded"
    mode: str | None  # "local" | "full" | None
    job_id: str | None
    scene_revision: int | None
    write_windows: tuple[RasterWindow, ...] = ()
    read_windows: tuple[RasterWindow, ...] = ()
    patch_paths: tuple[Path, ...] = ()
    store_keys: tuple[TemporalResultKey, ...] = ()
    requested_variables: tuple[str, ...] = ()
    #: Variables the batch requested that the patch layer cannot carry yet,
    #: as ``(name, reason)`` — recorded, never silently dropped.
    unpublished_variables: tuple[tuple[str, str], ...] = ()
    fallback_reason: str | None = None
    diagnostics: dict = field(default_factory=dict)

    @property
    def published(self) -> bool:
        return self.status == "published"


#: Stable schema version of the serialized plan document (u-d4).
PLAN_DOCUMENT_SCHEMA_VERSION = 1


def _plan_window_dict(window: RasterWindow) -> dict[str, int]:
    return {
        "row_start": int(window.row_start),
        "row_stop": int(window.row_stop),
        "col_start": int(window.col_start),
        "col_stop": int(window.col_stop),
    }


def executed_plan_document(executed: ExecutedPlan) -> dict:
    """Serialize one :class:`ExecutedPlan` into a stable JSON document.

    Pure serialization of an ALREADY executed plan (no behavior change in
    any execute path). The document is what the HTTP job record surfaces
    as ``impact_plan``: per-node stages (``changed`` for sources the batch
    edited, ``reused`` for nodes the planner proved clean, ``recomputed``
    for everything the dirty closure forced), the realized routing, and
    estimated-vs-actual work. Field names are contract-stable; the schema
    version gates future evolution.
    """
    plan = executed.plan
    impacts = {impact.node_id: impact for impact in plan.node_impacts}
    changed = set(plan.changed_sources)
    reusable = set(plan.reusable_nodes)
    nodes: list[dict] = []
    for node_id in sorted(set(impacts) | changed | reusable):
        impact = impacts.get(node_id)
        if node_id in changed:
            stage = "changed"
            why = (
                impact.reason
                if impact is not None
                else "the batch edited this source directly"
            )
        elif node_id in reusable:
            stage = "reused"
            why = (
                impact.reason
                if impact is not None
                else "outside the batch's dirty closure: prior published "
                "results remain valid"
            )
        else:
            stage = "recomputed"
            why = (
                impact.reason
                if impact is not None
                else "inside the batch's dirty closure: recomputed from the "
                "edited sources"
            )
        entry: dict = {
            "node": str(node_id),
            "stage": stage,
            "why": why,
        }
        if impact is not None:
            entry["spatial_scope"] = str(impact.spatial_scope)
            entry["temporal_scope"] = str(impact.temporal_scope)
            if impact.time_start is not None:
                entry["time_start"] = int(impact.time_start)
            if impact.time_stop is not None:
                entry["time_stop"] = int(impact.time_stop)
            if impact.write_windows:
                entry["windows"] = [
                    _plan_window_dict(window) for window in impact.write_windows
                ]
        nodes.append(entry)
    document: dict = {
        "schema_version": PLAN_DOCUMENT_SCHEMA_VERSION,
        "status": executed.status,
        "mode": executed.mode,
        "scene_revision": executed.scene_revision,
        "job_id": executed.job_id,
        "nodes": nodes,
        "routing": {
            "mode": executed.mode,
            "write_windows": [
                _plan_window_dict(window) for window in executed.write_windows
            ],
            "transport_window": (
                _plan_window_dict(plan.transport_window)
                if plan.transport_window is not None
                else None
            ),
            "fallback_reason": executed.fallback_reason,
            "fallback_reasons": list(plan.fallback_reasons),
        },
        "estimates": {
            "work_units": int(plan.estimated_work_units),
            "memory_bytes": int(plan.estimated_memory_bytes),
        },
        "realized": {
            "patch_count": len(executed.patch_paths),
            "published_variables": list(executed.requested_variables),
            "unpublished_variables": [
                {"name": name, "reason": reason}
                for name, reason in executed.unpublished_variables
            ],
            "diagnostics": dict(executed.diagnostics),
        },
    }
    return document


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class PlanExecutor:
    """Execute plans against the real scientific worker.

    One executor owns one scenario: the editable
    :class:`~solweig_gpu.incremental.trees.TreeLayer`, the scene-graph
    state (revisions), the bound
    :class:`~solweig_gpu.incremental.planner.EditPlanner`, the composed
    :class:`~solweig_gpu.incremental.worker.ExactWorker`, and the
    ``(node, time)`` result store. The full lifecycle is
    :meth:`execute`; :meth:`plan` and :meth:`execute_plan` are the lower
    halves for callers that stage transactions themselves.
    """

    def __init__(
        self,
        *,
        cache: SiteCache,
        layer: TreeLayer,
        site_dir: str | Path,
        results_root: str | Path,
        selected_date_str: str,
        registry: AdapterRegistry | None = None,
        graph: EditGraph | None = None,
        influence_config: InfluenceConfig | None = None,
        sun_positions: Sequence[Any] | None = None,
        store: TemporalResultStore | None = None,
        state: SceneGraphState | None = None,
        scratch_root: str | Path | None = None,
        default_temporal_steps: int = DEFAULT_TEMPORAL_STEPS,
        revision_provider: Callable[[], int] | None = None,
        full_recompute_fraction: float | None = None,
    ) -> None:
        self._graph = graph if graph is not None else default_edit_graph()
        self._state = (
            state if state is not None else SceneGraphState.initial(self._graph)
        )
        if self._state.graph is not self._graph:
            raise ExecutorError(
                "state graph must be the executor's graph (plan/execute must "
                "share one node vocabulary)"
            )

        self.cache = cache
        self.layer = layer
        self.site_dir = Path(site_dir)
        self.results_root = Path(results_root)
        self.selected_date_str = selected_date_str
        self._scratch_root = scratch_root
        self._scenario_id = layer.scenario_id

        self._grid = RasterGrid(
            rows=cache.rows,
            cols=cache.cols,
            pixel_size_m=cache.pixel_size_m,
            origin_x_m=cache.manifest.origin_x_m,
            origin_y_m=cache.manifest.origin_y_m,
        )
        self._available_times = tuple(range(cache.time_steps))
        self._default_temporal_steps = int(default_temporal_steps)

        # The composed worker is the mission-validated windowed path: it
        # owns dirty windows, the GVF write margin, local/full routing,
        # patch staging/publishing, and the supersession checkpoints. The
        # executor never rebuilds those. The worker's supersession provider
        # defaults to the executor's INTERNAL scene revision; a caller
        # whose durability is decided elsewhere (the server bridge binds it
        # to the store's committed scene version, so a concurrent commit
        # aborts the batch mid-solve like the legacy solver's) can override
        # it — ``None`` keeps the internal default.
        #
        # ``full_recompute_fraction`` (exact-lane routing policy) overrides
        # ONLY the worker's mode-choice threshold: the fraction at which a
        # solve's dirty-write union demotes to a full-tile recompute. It is
        # threaded to BOTH coupled decision sites (the worker's
        # ``choose_recompute_mode`` and the planner's safety policy) so
        # plan-FULL and worker-FULL agree; ``None`` keeps each site's
        # default. The adapter-facing ``self._influence_config`` below stays
        # keyed on the CALLER's config so its site-adjusted derivation
        # (sun altitude, shadow length) is untouched by a routing override.
        worker_influence_config = influence_config
        if full_recompute_fraction is not None:
            worker_influence_config = replace(
                influence_config if influence_config is not None else InfluenceConfig(),
                full_recompute_fraction=float(full_recompute_fraction),
            )
        self._worker = ExactWorker(
            cache,
            layer,
            site_dir=site_dir,
            results_root=results_root,
            selected_date_str=selected_date_str,
            influence_config=worker_influence_config,
            revision_provider=revision_provider
            or (lambda: self._state.scene_revision),
            scratch_root=scratch_root,
        )
        # Site facts for the adapters, derived exactly like the worker
        # derives its own (single source of truth, no duplication).
        self._sun_positions = (
            tuple(sun_positions)
            if sun_positions is not None
            else tuple(self._worker._sun_positions())
        )
        self._influence_config = (
            influence_config
            if influence_config is not None
            else self._worker._site_influence_config()
        )

        # Registry: bind every concrete adapter, then hot-swap the
        # vegetation implementation to the site-configured instance (the
        # registry documents hot-swapping; metadata stays identical).
        self._registry = (
            registry if registry is not None else builtin_registry()
        )
        register_default_adapters(self._registry)
        self._registry.register(
            VegetationGeometryAdapter(
                registry=self._registry,
                sun_positions=self._sun_positions,
                influence_config=self._influence_config,
            )
        )

        self._planner = EditPlanner(
            grid=self._grid,
            graph=self._graph,
            registry=self._registry,
            allow_wind_extension=False,
            default_temporal_steps=self._default_temporal_steps,
            policy=(
                ConservativeSafetyPolicy(
                    full_recompute_fraction=float(full_recompute_fraction)
                )
                if full_recompute_fraction is not None
                else None
            ),
        )

        self.store = store if store is not None else TemporalResultStore(
            grid=self._grid
        )
        # u-c1b M3 (lead decision: single writer, loud refusal): bind this
        # executor's identity on the store NOW — a second executor handed
        # the same store is refused at construction (and again at every
        # attributed publish) instead of wedging into per-key revision
        # conflicts. Sharing one store across executors is unsupported:
        # their ``(node, time)`` keys and revision chains collide.
        #
        # The identity carries a per-INSTANCE token (u-c1-review R1): the
        # scene-address prefix alone (site:tile:scenario) made the id
        # IDEMPOTENT for identical addresses, so two executors built for the
        # same scenario against one shared store both bound successfully and
        # then published interleaved revision chains silently — disjoint
        # ``(node, time)`` keys never even collided. The token makes the ids
        # distinct, so the second construction is refused loudly. It cannot
        # wedge a restarted legitimate owner: the store is in-memory and
        # lives exactly one process, so a new process always builds a new
        # store and binds fresh. Like ``job_id`` (result.new_job_id), the
        # token is identity/provenance, never a decision-path input.
        self._store_writer_id = (
            f"executor:{self.cache.site_id}:{self.cache.tile_key}"
            f":{self._scenario_id}:{uuid4().hex[:8]}"
        )
        self.store.bind_writer(self._store_writer_id)
        self._has_wind_coefficients: bool | None = None
        #: Accumulated scenario forcing overlay (u-c3): the fold of every
        #: forcing delta committed by a PUBLISHED batch. The next job's
        #: overlay is this plus the batch's committed forcing deltas
        #: (last-write-wins per (variable, time)), so a later batch can
        #: never silently revert an earlier met edit.
        self._applied_forcing_overlay: ForcingOverlay | None = None
        #: Accumulated scenario land-cover overlay (u-c4): the fold of
        #: every paint delta committed by a PUBLISHED batch, coalesced
        #: per-cell last-write-wins AT COMMIT (u-c6, u-c4-review M3) into
        #: a canonical rectangle decomposition of the session's painted
        #: cells carrying their final classes. The next job's overlay is
        #: this plus the batch's committed paint patches re-coalesced, so
        #: a later batch can never silently revert an earlier paint — and
        #: the accumulated patch count stays O(edited area), never
        #: O(session strokes).
        self._applied_landcover_overlay: LandCoverOverlay | None = None
        #: Resolved class grid of ``_applied_landcover_overlay`` (u-c6b L5):
        #: committed at publication next to the overlay itself (it is the
        #: staged fold's single validation resolve — coalescing preserves
        #: the resolved grid), so the per-batch value-difference test
        #: against the published state needs no second full-grid resolve.
        #: ``None`` is the never-painted baseline.
        self._applied_landcover_resolved: np.ndarray | None = None
        #: The staged batch's VALUE-difference windows vs the published
        #: overlay state (u-c6b): set by :meth:`_stage_overlays` on every
        #: batch and read by the whole-batch no-op interception — a batch
        #: folds to a clean no-op only when it resolves VALUE-equal to the
        #: published scenario state (overlay ``==`` is painted-set equality
        #: and would miss a same-value repaint of never-painted cells).
        self._staged_landcover_value_diff: tuple[RasterWindow, ...] = ()
        #: The staged fold's resolved grid (u-c6b L5), committed into
        #: ``_applied_landcover_resolved`` on publication.
        self._staged_landcover_resolved: np.ndarray | None = None
        #: Accumulated scenario model parameters (u-c7): the last-write-wins
        #: fold of every parameter delta committed by a PUBLISHED batch.
        #: The next job's kernel arguments are this plus the batch's
        #: committed parameter deltas (last-write-wins per name — the
        #: parameters are time-invariant, so there is no time dimension),
        #: so a later batch can never silently revert an earlier parameter
        #: edit. ``None`` (never set) is the parameter-free baseline run.
        self._applied_model_parameters: dict[str, Any] | None = None
        #: Accumulated scenario building-massing fold (u-d4c): the
        #: per-building-id net edit of every massing edit committed by a
        #: PUBLISHED batch, in first-appearance order.
        #:
        #: FOLD SEMANTICS (verified against
        #: :meth:`solweig_gpu.incremental.buildings.BuildingLayer.apply_edits`
        #: — the rasterizer is PLACEMENT-style: a ``before`` footprint is
        #: reset to DEM ground, an ``after`` footprint is painted at
        #: ``ground + height_m``; nothing validates ``before`` against the
        #: staged raster). Because the regeneration chain always stages the
        #: scenario site FROM BASELINE, the fold is kept at the DELTA level
        #: with BASELINE as the "old" anchor:
        #:
        #: * each id's entry is ``MassingEdit(id, before=<the before-spec
        #:   of the id's FIRST session edit>, after=<the after-spec of the
        #:   id's LAST session edit>)`` — last-write-wins per id, so
        #:   ``add A(h=15)`` then ``update A(h=20)`` folds to
        #:   ``add A(h=20)`` (``before=None``: the id never existed in the
        #:   baseline), and a baseline building's ``update`` keeps the
        #:   baseline state as the anchor so re-staging from baseline
        #:   reproduces the session's net change exactly;
        #: * DELETES STAY DELETES: an id whose net state is absent keeps an
        #:   ``after=None`` entry forever. A deleted BASELINE building
        #:   anchors at its baseline spec, so the chain keeps erasing it on
        #:   every re-stage; a deleted SESSION-ADDED building keeps a
        #:   TOMBSTONE anchored at the deleted session spec — a raster no-op
        #:   against the baseline-staged site (its footprint resets to DEM
        #:   ground the baseline already carries) that exists so the fold
        #:   never empties by deletion: ``regenerate_building_batch``
        #:   refuses empty edit tuples, and routing the worker instead
        #:   would hit its nothing-pending no-op gate (a building delta is
        #:   not worker state) and raise.
        #:
        #: While the fold is non-empty, EVERY batch routes through
        #: :meth:`_regenerate_building_chain` with the full fold (executed
        #: scope may EXCEED the plan promise — full ⊇ windows — never
        #: undercut, the ``force_full`` principle): the chain is the only
        #: solve path whose site/cache carry the edited massing, so a
        #: baseline-bound worker batch would silently revert the buildings
        #: (the u-d4c defect). It commits only on publication and is
        #: re-derived per batch, so a failed or superseded batch leaves it
        #: untouched. ``{}`` (never touched by a published building edit)
        #: keeps the worker routing byte-identical to pre-u-d4c.
        self._applied_massing_edits: dict[str, MassingEdit] = {}
        #: Building-chain scenario root (u-c5 wiring, landed by u-c6):
        #: ``results/<scenario_id>/.building`` — a PER-SCENARIO path (one
        #: directory per scenario under that scenario's results root, NOT
        #: per executor instance: the path carries no per-instance token),
        #: holding the regeneration chain's staged site/cache/
        #: solver_scratch. DURABILITY CHOICE: on SUCCESS the tree is kept
        #: until the next chain-routed batch (u-d4c: any batch while the
        #: building fold is non-empty; ``stage_scenario_site`` wipes
        #: and rebuilds it, so it is bounded by one batch and doubles as
        #: an audit artifact); on ROLLBACK it is wiped best-effort — the
        #: failed batch's partial chain has no consumers, and the stage
        #: list in the chained RegenerationError already names how far
        #: the chain got. The baseline site/cache the chain reads are
        #: read-only by construction (u-c5's hash-verified invariant).
        #:
        #: SHARING EXPOSURE (u-c6b L4, pre-existing and documented): every
        #: executor bound to the SAME scenario id AND results root shares
        #: this one directory. The store's single-writer fence
        #: (:meth:`store.TemporalResultStore.bind_writer`) refuses a
        #: second executor only when the STORE is shared; two executors
        #: with the same scenario + results root but SEPARATE stores are
        #: not fenced anywhere — one executor's rollback ``rmtree`` of
        #: this root would wipe the other's in-flight chain mid-batch.
        #: That configuration is unsupported: share the store (so
        #: ``bind_writer`` refuses the second executor at construction)
        #: or give each executor its own results root.
        self._scenario_regenerate_root = (
            self.results_root / self._scenario_id / ".building"
        )

    # ------------------------------------------------------------------
    # Site context (scope item c)
    # ------------------------------------------------------------------

    @property
    def grid(self) -> RasterGrid:
        return self._grid

    @property
    def state(self) -> SceneGraphState:
        return self._state

    @property
    def scene_revision(self) -> int:
        return self._state.scene_revision

    @property
    def available_times(self) -> tuple[int, ...]:
        return self._available_times

    @property
    def default_temporal_steps(self) -> int:
        """The planner's ALL/unbounded-REPLAY estimate floor (24 by default)."""
        return self._default_temporal_steps

    @property
    def has_wind_coefficients(self) -> bool:
        """Wind detection wired from the site directory (oracle parity)."""
        if self._has_wind_coefficients is None:
            self._has_wind_coefficients = detect_wind_coefficients(
                self.site_dir, self.cache.tile_key
            )
        return self._has_wind_coefficients

    def site_context(self, state: SceneGraphState | None = None) -> SiteContext:
        """The ALWAYS-bound context (scope item c)."""
        current = state if state is not None else self._state
        return SiteContext(
            site_id=self.cache.site_id,
            grid=self._grid,
            scene_revision=current.scene_revision,
            available_times=self._available_times,
            has_wind_coefficients=self.has_wind_coefficients,
        )

    # ------------------------------------------------------------------
    # Validation + planning
    # ------------------------------------------------------------------

    def adapter_for(self, adapter_id: str):
        """Resolve a concrete adapter through the bound registry."""
        return self._registry.get_adapter(adapter_id)

    def validate(self, command: EditCommand) -> ValidatedEdit:
        """Validate one command against the current site context."""
        return self.adapter_for(command.adapter_id).validate(
            command, self.site_context()
        )

    def plan(self, edits: Sequence[ValidatedEdit]) -> ImpactPlan:
        """Fence (d), then plan with the context ALWAYS bound (c)."""
        if not edits:
            raise PlanningError("batch must contain at least one validated edit")
        # Scope item (d): fence BEFORE planning — a clean PlanningError for
        # out-of-series requested times, never a silent replay extension.
        fence_requested_times(edits, self._available_times)
        return self._planner.plan(edits, self._state, self.site_context())

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, edits: Sequence[ValidatedEdit]) -> ExecutedPlan:
        """Full lifecycle: plan -> stage -> commit -> execute -> record.

        Zero-stage (view-only or fully coalesced no-op) plans are accepted
        and executed as no-ops: nothing is staged into the layer, nothing
        publishes, and the scene revision does not advance.
        """
        plan = self.plan(edits)
        scientific = self._scientific_sources(plan)
        if not scientific:
            return ExecutedPlan(
                plan=plan,
                status="no-op",
                mode=None,
                job_id=None,
                scene_revision=self.scene_revision,
                requested_variables=self._resolve_variables(edits)[0],
                diagnostics={
                    "reason": (
                        "zero scientific stages (view-only or coalesced "
                        "no-op batch): nothing to execute"
                    )
                },
            )

        # Integrability is checked BEFORE anything is staged: a mixed batch
        # containing an unintegrated family is refused whole (partial
        # execution would advance the revision for half a batch).
        self._check_integrable(plan)

        scenario_ids = {edit.command.scenario_id for edit in edits}
        if scenario_ids != {self._scenario_id}:
            raise ExecutorError(
                f"batch scenarios {sorted(scenario_ids)} do not match the "
                f"executor scenario {self._scenario_id!r}"
            )

        transaction = ScenarioTransaction(scenario_id=self._scenario_id)
        try:
            for edit in edits:
                if self._graph.kind(edit.source_node_id) is NodeKind.VIEW:
                    # View-layer selections ride the batch (the planner
                    # plans them) but stage nothing: they select among
                    # already-published layers and have no source delta to
                    # apply. Their concrete adapter lands with the U-D API.
                    continue
                self.adapter_for(edit.adapter_id).apply_source_delta(
                    edit.delta, transaction
                )
            committed = transaction.commit()
        except BaseException:
            transaction.rollback()
            raise
        requested, unpublished = self._resolve_variables(edits)
        return self.execute_plan(
            plan,
            committed,
            requested_variables=requested,
            unpublished_variables=unpublished,
        )

    def execute_plan(
        self,
        plan: ImpactPlan,
        committed_deltas: Sequence[SourceDelta],
        *,
        requested_variables: Sequence[str] | None = None,
        unpublished_variables: Sequence[tuple[str, str]] = (),
    ) -> ExecutedPlan:
        """Execute one plan against already-committed deltas.

        Lower half of :meth:`execute`: the transaction is the caller's
        responsibility; the executor dispatches per source family and
        records the outcome. Stale plans (``plan.scene_revision`` not the
        current scene revision, or unset) are refused outright — scope
        item (f). ``requested_variables`` is the batch's transport
        vocabulary; when omitted only the OUTPUT product nodes present in
        the plan (utci/tmrt, plus the ``utci`` physics floor) are carried,
        so callers transporting non-node producible layers (``shadow``,
        ``kup``, ...) must pass the vocabulary explicitly —
        :meth:`execute` always does.
        """
        self._check_integrable(plan)
        if plan.scene_revision is None:
            raise ExecutorError(
                "plan carries no scene_revision; the executor only executes "
                "context-bound plans (UEDIT-006: stale revisions never publish)"
            )
        if plan.scene_revision != self.scene_revision:
            raise ExecutorError(
                f"plan revision {plan.scene_revision} is stale: the executor "
                f"is at revision {self.scene_revision} (UEDIT-006)"
            )

        # Group committed deltas per source node exactly like the planner
        # grouped them (arrival order per (node, adapter)).
        grouped: dict[tuple[str, str], list[SourceDelta]] = {}
        for delta in committed_deltas:
            grouped.setdefault(
                (delta.source_node_id, delta.adapter_id), []
            ).append(delta)
        committed: dict[str, tuple[SourceDelta, ...]] = {}
        for (node_id, _adapter_id), deltas in sorted(grouped.items()):
            committed[node_id] = (coalesce_source_deltas(deltas),)

        scientific = self._scientific_sources(plan)
        if not scientific:
            return ExecutedPlan(
                plan=plan,
                status="no-op",
                mode=None,
                job_id=None,
                scene_revision=self.scene_revision,
                diagnostics={"reason": "plan has zero scientific stages"},
            )

        if requested_variables is None:
            requested, plan_unpublished = self._variables_from_plan(plan)
            unpublished = tuple(unpublished_variables) + plan_unpublished
        else:
            requested, derived_unpublished = self._split_variables(
                set(requested_variables) | {"utci"}
            )
            unpublished = tuple(unpublished_variables) + derived_unpublished
        self._worker.requested_variables = requested

        # u-c3 seam: committed forcing deltas -> the scenario overlay the
        # job materializes its forcing from. When the batch carries forcing
        # deltas the seam is exercised BEFORE any state advances: the
        # loader's masked cross-check refuses a tampered or inconsistent
        # resolve whole-batch (fail-fast), so a bad overlay can never leave
        # a half-executed revision behind. The staged overlay rides the job
        # (the worker materializes it); it commits into the accumulated
        # scenario state only on publication below.
        forcing_overlay = self._effective_forcing_overlay(committed)
        if METEOROLOGY_SOURCE_NODE in committed:
            self._scenario_forcing(committed)
        # u-c6b presence rule (u-c6-review met analog): in a MIXED batch,
        # a forcing fold that resolves value-equal to the published
        # overlay still marks the job pending — pending by PRESENCE of
        # the batch's forcing deltas, not by value difference — so the
        # job recomputes the forcing closure (the full tile; forcing is
        # site-global) instead of riding a local job scoped to the other
        # families' windows. A forcing-ONLY batch never sets it: the
        # whole-batch value no-op is intercepted below (clean no-op), and
        # a wedged retry (worker watermarks stuck ahead of the executor
        # state, u-c1b M1) must keep finding nothing pending so the loud
        # ExecutorError fires.
        forcing_present = METEOROLOGY_SOURCE_NODE in committed and any(
            node_id != METEOROLOGY_SOURCE_NODE and deltas
            for node_id, deltas in committed.items()
        )
        self._worker.stage_forcing_overlay(
            forcing_overlay, batch_pending=forcing_present
        )
        # SEAM (u-c2, landed by u-c7): committed model-parameter deltas
        # fold (over the published parameter state, last-write-wins per
        # name) into the flat kernel-argument mapping staged on the
        # worker. The fold re-validates every name and domain
        # (``kernel_arguments_from_deltas``) BEFORE any state advances, so
        # a blocked or out-of-domain parameter is refused whole-batch; the
        # staged payload rides the job (both solve paths forward it into
        # ``run_utci_window``) and commits into the accumulated parameter
        # state only on publication below.
        model_parameters = self._physics_kernel_arguments(committed)
        self._worker.stage_model_parameters(model_parameters)
        # u-c4 seam (landed): committed paint deltas fold into the scenario
        # land-cover overlay staged on the worker. The seam materializes
        # and re-verifies the resolved grid BEFORE any state advances (a
        # tampered or unstageable overlay is refused whole-batch), and the
        # staged overlay commits into the accumulated scenario state only
        # on publication below.
        landcover_overlay = self._stage_overlays(committed)

        # Publication revision (scope item f). The plan is planned against
        # ``plan.scene_revision`` = R (the batch's base); its results depict
        # the scene the batch CREATES, which is revision R+1 — the same
        # convention the worker's own lifecycle uses (bump, then publish).
        # Publishing the edited scene under R would relabel the baseline
        # revision, so the executor advances FIRST (gating already proved
        # R is current), publishes at R+1, and rolls both the state and the
        # layer replay back wholesale if anything fails or the job is
        # superseded — a failed batch never advances the revision.
        batch_edits = self._prepare_vegetation_batch(committed)
        building_edits = self._prepare_building_batch(committed)
        # u-d4c: the batch's building edits fold into the ACCUMULATED
        # building state BEFORE dispatch. The fold below is a staged copy
        # (never the applied dict itself): it commits into
        # ``_applied_massing_edits`` only at publication, so a failed or
        # superseded batch leaves the published fold untouched — the same
        # staged-then-commit discipline the forcing/land-cover/parameter
        # overlays follow.
        massing_fold = self._staged_massing_fold(building_edits)
        if (
            not building_edits
            and not batch_edits
            and forcing_overlay == self._applied_forcing_overlay
            and not self._staged_landcover_value_diff
            and model_parameters == self._applied_model_parameters
        ):
            # u-e1 F3: the staging seams above already ran for a batch
            # that will now never dispatch — the worker carries the
            # forcing-presence flag (a MIXED value-equal batch sets it by
            # PRESENCE) and a land-cover dirty override with no job to
            # consume them. Mirror the rollback discipline every other
            # non-publishing exit follows: rewind the CURRENT watermarks
            # (an identity rewind — nothing published) and clear both
            # staged flags, so a reused executor does not route every
            # later batch full-tile off a leaked pending marker.
            self._worker.rollback_pending(self._worker.publication_watermarks)
            # u-c6 (u-c4-review M3) + u-c6b (u-c6-review H1/L3): the batch
            # folds VALUE-equal to the PUBLISHED scenario state in every
            # family — a repaint of values already published, or a first
            # paint of cells that already carry the target class in the
            # baseline — with no other deltas (vegetation, forcing,
            # parameters, building). Nothing any solve path could
            # recompute differs, so the batch is a clean no-op: the scene
            # revision does not advance and nothing publishes. The
            # land-cover test is VALUE-based (overlay ``==`` is painted-set
            # equality and would miss a same-value repaint of
            # never-published cells — the u-c6-review repro (a) wedge),
            # and every comparison is against the EXECUTOR's committed
            # state (not the worker's watermarks): a wedged worker
            # (u-c1b M1) has its watermarks stuck AHEAD of the executor
            # state, and a value-changing retry must still dispatch into
            # the worker so ``_record_outcome`` raises the loud "nothing
            # pending" ExecutorError instead of silently swallowing the
            # wedge.
            return ExecutedPlan(
                plan=plan,
                status="no-op",
                mode=None,
                job_id=None,
                scene_revision=self.scene_revision,
                requested_variables=requested,
                unpublished_variables=unpublished,
                diagnostics={
                    "reason": (
                        "batch folds to the published scenario state "
                        "(values already published): nothing to recompute"
                    )
                },
            )
        force_full = any(
            impact.spatial_scope is SpatialScope.FULL
            for impact in plan.node_impacts
        )
        previous_state = self._state
        self._state = self._state.advance(plan.changed_sources)
        replayed: tuple[TreeEdit, ...] = ()
        # u-c1b M1: snapshot the worker's publication watermarks BEFORE the
        # batch is dispatched. The worker advances them the moment its own
        # publish loop completes, while the executor commits its state only
        # after recording the outcome — a failure in between must rewind
        # BOTH sides or every retry finds "nothing pending" forever.
        worker_watermarks = self._worker.publication_watermarks
        family_outcome: JobOutcome | None = None
        met_fast_published = False
        try:
            if batch_edits:
                self._replay_tree_edits(batch_edits)
                replayed = batch_edits
            met_fast_refusal: str | None = None
            met_warm_refusal: str | None = None
            if not batch_edits and not massing_fold:
                if self._met_fast_path_plan_shape(plan):
                    # r3a: an all-utci_only met-only batch whose plan the
                    # planner narrowed to {utci @ changed t} reuses the
                    # PUBLISHED radiation state and recomputes only the
                    # comfort planes. Any unmet condition is a typed
                    # refusal (MetFastPathRefused) that routes today's
                    # exact full solve instead — never a silently wrong
                    # shortcut.
                    try:
                        family_outcome, met_fast_refusal = (
                            self._run_met_utci_fast_path(
                                plan,
                                committed,
                                forcing_overlay,
                                model_parameters,
                                target_revision=self._state.scene_revision,
                            )
                        )
                    except MetFastPathRefused as refusal:
                        met_fast_refusal = str(refusal)
                elif self._met_warm_path_eligible(committed):
                    # G2.1 item 4 — the first warm consumer, on the r3a
                    # REFUSED branch: a radiation-affecting met-only batch
                    # (the shape the planner keeps at FULL closure) serves
                    # its unchanged prefix from the store and solves the
                    # suffix warm from a fingerprint-matching checkpoint.
                    # Structural refusals route today's exact full solve
                    # with the reason attached, exactly like r3a.
                    try:
                        family_outcome, _ = (
                            self._run_met_warm_sparse_path(
                                committed,
                                model_parameters,
                                target_revision=self._state.scene_revision,
                            )
                        )
                    except MetFastPathRefused as refusal:
                        met_warm_refusal = str(refusal)
                met_fast_published = (
                    family_outcome is not None and family_outcome.published
                )
            if family_outcome is None:
                if massing_fold:
                    # u-d4c route-through: ANY batch while the accumulated
                    # building fold is non-empty — building-only, mixed, or
                    # other-family-only — rides the regeneration chain with
                    # the FULL fold. The chain stages the scenario site from
                    # BASELINE and applies the fold deterministically, so the
                    # baseline stays read-only and the published buildings
                    # survive every later batch (the u-d4c defect: routing a
                    # post-building batch through the baseline-bound worker
                    # silently reverted the massing). Executed scope may
                    # EXCEED the plan promise (full ⊇ windows), never
                    # undercut — the ``force_full`` principle.
                    family_outcome = self._regenerate_building_chain(
                        tuple(massing_fold.values()),
                        target_revision=self._state.scene_revision,
                        forcing_overlay=forcing_overlay,
                        landcover_overlay=landcover_overlay,
                        model_parameters=model_parameters,
                        chain_carries_new_edits=bool(building_edits),
                    )
                else:
                    family_outcome = self._run_worker(
                        target_revision=self._state.scene_revision,
                        force_full=force_full,
                    )
                if met_fast_refusal is not None:
                    # Refusal telemetry rides the fallback outcome (never a
                    # silent detour): diagnostics is a plain dict by design.
                    family_outcome.diagnostics["met_fast_path_refusal"] = (
                        met_fast_refusal
                    )
                if met_warm_refusal is not None:
                    # Same contract for the warm consumer's structural
                    # refusals (prefix coverage miss, row-0 edit, ...).
                    family_outcome.diagnostics["met_warm_path_refusal"] = (
                        met_warm_refusal
                    )
            executed = self._record_outcome(
                plan, family_outcome, requested, unpublished
            )
        except BaseException as error:
            rollback_failure = self._rollback_failed_batch(
                replayed=replayed,
                outcome=family_outcome,
                previous_state=previous_state,
                worker_watermarks=worker_watermarks,
                chain_routed=bool(massing_fold),
            )
            if rollback_failure is not None:
                # T10: the ORIGINAL error stays the actionable one; a
                # rollback failure rides as its cause instead of masking it.
                raise error from rollback_failure
            raise
        if not executed.published:
            # Superseded: nothing was published under the advanced
            # revision, so the scene state rolls back with the layer — and
            # the batch's forcing and land-cover overlays stay uncommitted
            # (the next batch re-stages the accumulated scenario state).
            rollback_failure = self._rollback_failed_batch(
                replayed=replayed,
                outcome=family_outcome,
                previous_state=previous_state,
                worker_watermarks=worker_watermarks,
                chain_routed=bool(massing_fold),
            )
            if rollback_failure is not None:
                raise ExecutorError(
                    "superseded batch rollback did not complete cleanly; "
                    f"layer/worker state may be inconsistent: {rollback_failure}"
                ) from rollback_failure
            return executed
        # Published: the overlays and kernel arguments the results were
        # computed under become the scenario's accumulated forcing,
        # land-cover, model-parameter, and building-massing state (the
        # land-cover resolve memo rides the overlay commit, u-c6b L5; the
        # building fold is the staged copy, u-d4c).
        self._applied_forcing_overlay = forcing_overlay
        self._applied_landcover_overlay = landcover_overlay
        self._applied_landcover_resolved = self._staged_landcover_resolved
        self._applied_model_parameters = (
            dict(model_parameters) if model_parameters is not None else None
        )
        self._applied_massing_edits = massing_fold
        if massing_fold or met_fast_published:
            # The batch published OUTSIDE ``ExactWorker.run`` (u-d4c: ANY
            # chain-routed batch, not just batches carrying new building
            # edits): adopt the publication on the worker so its staged
            # overlays are not left "pending" (they would force every
            # later job full-tile) and so a chain-routed batch's replayed
            # tree edits are consumed — their effect is published in this
            # batch's full-tile patch, not left pending for the next job.
            pending = self._worker.pending_batch()
            acked_sequence = worker_watermarks.acked_sequence
            if pending is not None and pending.edits:
                acked_sequence = max(acked_sequence, pending.last_sequence)
            self._worker.adopt_published_state(
                PublicationWatermarks(
                    acked_sequence=acked_sequence,
                    forcing_overlay=forcing_overlay,
                    landcover_overlay=landcover_overlay,
                    model_parameters=model_parameters,
                )
            )
        return executed

    def _rollback_failed_batch(
        self,
        *,
        replayed: Sequence[TreeEdit],
        outcome: JobOutcome | None,
        previous_state: SceneGraphState,
        worker_watermarks: PublicationWatermarks,
        chain_routed: bool = False,
    ) -> BaseException | None:
        """Wholesale rollback of a failed/superseded batch (u-c1b M1/M4/T10).

        Mirrors the publication commit points in reverse: scene-state
        rebind, layer replay undo, worker watermark rewind, and — when the
        worker had already renamed patches into the results root —
        best-effort removal of the failed batch's orphan ``rev-*``
        directories (M4). Every step runs even when an earlier one failed,
        so one broken step can never leave the others advanced (that is
        the M1 wedge). The FIRST step failure is returned for the caller
        to chain onto the original error (T10: an undo failure must never
        mask the actionable exception); ``None`` means the rollback was
        clean.

        ``chain_routed`` (u-c6; u-d4c: ANY batch dispatched through the
        regeneration chain, not just batches carrying new building edits)
        additionally wipes the regeneration chain's scenario tree
        (best-effort): the failed batch's partial chain has no consumers,
        the retry rebuilds it from scratch, and the stage list in the
        chained :exc:`RegenerationError` already records how far the
        chain got. The accumulated building fold needs no explicit rewind
        here — it commits only on publication (u-d4c staged-copy
        discipline), so a failed or superseded batch leaves it at the
        previous published state by construction.
        """
        if chain_routed:
            shutil.rmtree(self._scenario_regenerate_root, ignore_errors=True)
        self._state = previous_state  # pure rebind: cannot fail
        failures: list[BaseException] = []
        if replayed:
            try:
                self._undo_tree_edits(replayed)
            except BaseException as undo_error:  # noqa: BLE001 - reported, never raised here
                _LOG.error(
                    "tree-edit undo failed during batch rollback "
                    "(u-c1b T10); the layer may carry replayed edits: %s",
                    undo_error,
                )
                failures.append(undo_error)
        try:
            self._worker.rollback_pending(worker_watermarks)
        except BaseException as worker_error:  # noqa: BLE001
            _LOG.error(
                "worker watermark rewind failed during batch rollback "
                "(u-c1b M1); the worker may see published edits as "
                "consumed: %s",
                worker_error,
            )
            failures.append(worker_error)
        if outcome is not None and outcome.patch_paths:
            # M4: the worker publishes per-window patch directories before
            # the executor commits. A post-publish failure leaves them as
            # never-current orphans that rev-globbing readers could pick
            # up; removal is best-effort and survivors are logged loudly.
            leftovers = discard_published_patches(outcome.patch_paths)
            for leftover in leftovers:
                _LOG.error(
                    "orphan patch directory %s of failed job %s survived "
                    "best-effort cleanup (u-c1b M4); it depicts a scene "
                    "revision that never became current — readers must "
                    "not consume it",
                    leftover,
                    outcome.job_id,
                )
        return failures[0] if failures else None

    # ------------------------------------------------------------------
    # Family dispatch
    # ------------------------------------------------------------------

    def _check_integrable(self, plan: ImpactPlan) -> None:
        for node_id in self._scientific_sources(plan):
            if node_id not in EXECUTABLE_SOURCE_NODES:
                raise NotIntegratedError(
                    node_id, NOT_INTEGRATED_PACKETS.get(node_id, "a later U-C packet")
                )

    def _prepare_vegetation_batch(
        self, committed: Mapping[str, tuple[SourceDelta, ...]]
    ) -> tuple[TreeEdit, ...]:
        """Committed vegetation deltas -> the worker's coalesced batch.

        The adapter's documented U-C hand-off
        (:func:`~solweig_gpu.incremental.adapters.vegetation.coalesced_batch_from_deltas`):
        the returned edits are replayed into the scenario layer through its
        public mutators (the layer assigns its own sequences; the worker's
        watermark stays authoritative), and the worker runs the real solve.
        A batch with no vegetation deltas (u-c3: a forcing-only batch)
        replays nothing — its recompute is driven by the staged forcing
        overlay instead.

        Pre-flight spec validation (u-c1b L5): every tree spec the coalesced
        batch would replay is structurally validated BEFORE any layer
        mutation. The layer's own mutators do validate, but only as they
        apply — a mid-chain refusal (e.g. ``trunk_ratio=1.0``) would leave
        the earlier edits replayed and the batch's only recourse the undo
        path. Refusing here keeps a malformed spec a pure transaction
        rejection: state, layer, and worker watermarks all untouched.
        """
        veg_deltas = [
            delta
            for deltas in committed.values()
            for delta in deltas
            if isinstance(delta, VegetationObjectDelta)
        ]
        if not veg_deltas:
            return ()
        batch = coalesced_batch_from_deltas(
            veg_deltas, scenario_id=self._scenario_id
        )
        if batch is None or not batch.edits:
            raise ExecutorError(
                "plan demands vegetation recomputation but the committed "
                "deltas coalesce to a no-op: plan and transaction disagree "
                "(double execution or mismatched batch)"
            )
        for edit in batch.edits:
            for role, spec in (("old_tree", edit.old_tree), ("new_tree", edit.new_tree)):
                if spec is None:
                    continue
                try:
                    validate_tree_spec(spec)
                except ValueError as error:
                    raise ExecutorError(
                        f"refusing batch: tree {edit.tree_id!r} carries an "
                        f"invalid {role} spec ({error}); validated in "
                        "pre-flight, before any layer mutation (u-c1b L5)"
                    ) from error
        return tuple(batch.edits)

    def _run_worker(self, *, target_revision: int, force_full: bool) -> JobOutcome:
        """Run the composed worker, honoring a FULL plan promise.

        The plan is the contract: when it marks ANY node FULL (policy
        demotion, full-only adapter), the products must be full tile. The
        worker's own locality routing is bypassed with the minimal valid
        full-recompute threshold, so the plan and the execution can never
        disagree on scope. A WINDOWS plan keeps the worker's routing,
        including its SolverInputError fallback to full — executed scope
        may exceed the promise, never undercut it.

        Temporal scope (perf wave 1 STEP 2a): the dispatched solve keeps
        the FULL series — the executor's node store records full-series
        coverage per publication (a partial-coverage record would need
        read-side reconciliation this seam does not have), and a
        full-series patch is a superset of any requested prefix. The
        causal-prefix truncation lives in the tree-path adapter
        (``make_exact_worker_solver``) and in the bridge's command
        construction (``_pending_commands``), where the plan's temporal
        replay is derived from the same requested series.
        """
        if force_full:
            original = self._worker.influence_config
            self._worker.influence_config = replace(
                original, full_recompute_fraction=1e-12
            )
            try:
                return self._worker.run(target_revision=target_revision)
            finally:
                self._worker.influence_config = original
        return self._worker.run(target_revision=target_revision)

    # ------------------------------------------------------------------
    # Met utci-only fast path (r3a)
    # ------------------------------------------------------------------

    def _met_fast_path_plan_shape(self, plan: ImpactPlan) -> bool:
        """True only for the plan shape the affinity narrowing produces.

        Exactly one impact — ``utci``, WINDOWS, carrying the FULL tile as
        its write window — and no FULL impacts. A WINDOWS impact with a
        tile-wide window can never come out of the generic planning path
        (``_build_impact`` demotes it via the safety policy), so this shape
        is itself the signature of the planner's affinity proof; anything
        else keeps the ordinary dispatch.
        """
        if len(plan.node_impacts) != 1:
            return False
        impact = plan.node_impacts[0]
        return (
            impact.node_id == "utci"
            and impact.spatial_scope is SpatialScope.WINDOWS
            and tuple(impact.write_windows) == (self.grid.full_window,)
        )

    def _run_met_utci_fast_path(
        self,
        plan: ImpactPlan,
        committed: dict[str, tuple[SourceDelta, ...]],
        forcing_overlay: ForcingOverlay | None,
        model_parameters: dict | None,
        *,
        target_revision: int,
    ) -> tuple[JobOutcome | None, str | None]:
        """Recompute ONLY utci at the changed timesteps (r3a fast path).

        Returns ``(outcome, None)`` on a fast publication or supersession;
        any unmet condition RAISES :exc:`MetFastPathRefused` (the caller
        catches it, routes today's exact full solve, and attaches the
        reason to that fallback outcome's diagnostics).

        Soundness: the planner narrowed this batch only because EVERY
        changed met variable carries the ``utci_only`` affinity (it enters
        only the comfort calculation at t), so the correct new utci@t is a
        pure function of (a) the resolved met table at t and (b) the tmrt
        plane@t the scenario already published. (a) is re-derived from the
        baseline text through the same overlay resolve the solver's loader
        runs; (b) is served from the ``(node, time)`` store under
        per-cell latest validity (G1.1): the full-tile window must be
        covered FULLY, and the covering entries are painted in ascending
        ``scene_revision`` order so every cell carries the value of its
        NEWEST entry. R5a retention leaves a superseded full-tile
        publication beside a newer windowed batch as mixed provenance
        {N, N+1}, and the G1.0 superset differential proved that
        composition bitwise equals the revision-head full recompute
        tile-wide — exactly the published state the affinity proof says
        is still valid. Any coverage miss or unpainted cell refuses;
        nothing is ever guessed.
        """
        started = time.monotonic()

        def refuse(reason: str) -> None:
            _LOG.info("met utci fast path refused: %s", reason)
            raise MetFastPathRefused(reason)

        if set(committed) != {METEOROLOGY_SOURCE_NODE}:
            refuse(
                f"batch changes {sorted(committed)}: the fast path serves "
                "met-only batches"
            )
        changed_variables: set[str] = set()
        for delta in committed[METEOROLOGY_SOURCE_NODE]:
            if not isinstance(delta, ForcingDelta):
                refuse("batch carries a non-forcing meteorology delta")
            for change in delta.changes:
                if change.variable not in MET_UTCI_ONLY_VARIABLES:
                    refuse(
                        f"forcing variable {change.variable!r} is "
                        "radiation_affecting: full closure required"
                    )
                changed_variables.add(change.variable)
        if self._staged_landcover_value_diff:
            refuse("batch stages land-cover edits: full closure required")
        if model_parameters != self._applied_model_parameters:
            refuse(
                "batch changes model parameters the published radiation "
                "state did not run under"
            )

        # (a) the resolved met tables: staged (post-batch) vs applied
        # (published) — the changed rows are exactly the recompute set.
        baseline = load_site_forcing(
            self.cache,
            site_dir=self.site_dir,
            selected_date_str=self.selected_date_str,
        ).met_table
        staged_table = (
            forcing_overlay.resolve(baseline)
            if forcing_overlay is not None
            else baseline
        )
        applied_table = (
            self._applied_forcing_overlay.resolve(baseline)
            if self._applied_forcing_overlay is not None
            else baseline
        )
        changed_rows = np.any(
            staged_table != applied_table, axis=1
        )
        changed_times = tuple(int(t) for t in np.flatnonzero(changed_rows))
        if not changed_times:
            refuse("overlay resolves value-equal to the published state")

        # (b) published tmrt planes at the changed times, composed per cell
        # under the G1.1 per-cell latest validity rule.
        full = self.grid.full_window
        planes: dict[int, np.ndarray] = {}
        for t in changed_times:
            coverage = self.store.window_coverage("tmrt", t, window=full)
            if coverage.status is not CoverageStatus.FULL:
                refuse(
                    f"tmrt@{t} coverage is {coverage.status.value}: the "
                    "radiation state is not fully published"
                )
            # G1.1 per-cell latest validity: paint the entries in ASCENDING
            # scene_revision order (later revisions overwrite earlier ones
            # cell-by-cell), building a max-revision raster the same way.
            # R5a retention keeps a superseded full-tile entry beside a
            # newer windowed batch as narrowed remainders, and the G1.0
            # superset differential proved the composed plane equals the
            # revision-head full recompute bitwise, tile-wide (batch write
            # windows subsume the causal impact, so a retained rev-N cell
            # already carries its newest value). The raster is the fence:
            # every cell must end up painted by its newest entry — a hole
            # means a coverage/store defect and refuses, never a guess.
            plane = np.full(
                (full.height, full.width), np.nan, dtype=np.float32
            )
            newest_revision = np.full(
                (full.height, full.width), -1, dtype=np.int64
            )
            patches: dict[Path, ResultPatch] = {}
            for entry in sorted(
                coverage.entries, key=lambda item: item.scene_revision
            ):
                # Dedup the LOAD, never the slice: one patch file can own
                # several narrowed remainder entries (R5a retention), and
                # every entry paints its own window.
                if entry.patch_path not in patches:
                    patches[entry.patch_path] = load_patch(entry.patch_path)
                patch = patches[entry.patch_path]
                row = (
                    patch.time_indices.index(t)
                    if patch.time_indices is not None
                    else t - patch.time_start
                )
                w = entry.write_window
                pw = patch.write_window  # patch arrays are window-relative
                plane[
                    w.row_start : w.row_stop, w.col_start : w.col_stop
                ] = patch.variable("tmrt")[
                    row,
                    w.row_start - pw.row_start : w.row_stop - pw.row_start,
                    w.col_start - pw.col_start : w.col_stop - pw.col_start,
                ]
                newest_revision[
                    w.row_start : w.row_stop, w.col_start : w.col_stop
                ] = entry.scene_revision
            if (newest_revision < 0).any():
                # Defect tripwire, not a live staleness check: FULL
                # coverage means the entry windows tile the request, so
                # an unpainted cell is unreachable through the public
                # API (review N1, G1.1). Staleness (a planner undershoot
                # leaving a wrong-valued cell) is fenced by the G1.0
                # superset differential, not here.
                refuse(
                    f"tmrt@{t} composed coverage leaves "
                    f"{int((newest_revision < 0).sum())} cells unpainted: "
                    "per-cell latest validity is incomplete"
                )
            planes[t] = plane

        # Same utci code path as the solver, full-tile extent: bitwise by
        # construction (utci_process.recompute_utci_steps replicates the
        # per-step ops verbatim; any missing input raises loudly).
        try:
            utci_planes = recompute_utci_steps(
                staged_table,
                changed_times,
                planes,
                np.asarray(self.cache.building_dsm),
                np.asarray(self.cache.dem),
            )
        except ValueError as error:
            refuse(f"utci recompute input recovery failed: {error}")

        patch = ResultPatch(
            job_id=new_job_id(),
            scene_revision=target_revision,
            mode="local",
            write_window=full,
            read_window=full,
            site_id=self.cache.site_id,
            tile_key=self.cache.tile_key,
            cache_manifest_sha256=self.cache.metadata()["manifest_sha256"],
            model_version=self.cache.model_version,
            variables=("utci",),
            arrays={"utci": utci_planes},
            time_start=changed_times[0],
            time_stop=changed_times[-1] + 1,
            time_indices=changed_times,
        )

        # Two-phase publication mirroring ExactWorker.run: staged, then the
        # supersession checkpoint, then the atomic rename — with the same
        # best-effort cleanup on any failure.
        scenario_root = self.results_root / self._scenario_id
        staging_root = scenario_root / ".staging"
        staging_root.mkdir(parents=True, exist_ok=True)
        try:
            staged_dir = stage_patch(patch, staging_root)
            self._worker._check_not_superseded(target_revision)
            published_dir = publish_staged_patch(
                staged_dir, scenario_root, patch.scene_revision
            )
        except SupersededJobError as reason:
            discard_staging(staging_root)
            return (
                JobOutcome(
                    status="superseded",
                    mode=None,
                    job_id=patch.job_id,
                    scene_revision=target_revision,
                    diagnostics={
                        "reason": str(reason),
                        "met_fast_path": True,
                    },
                ),
                None,
            )
        except BaseException:
            discard_staging(staging_root)
            raise

        elapsed = round(time.monotonic() - started, 6)
        return (
            JobOutcome(
                status="published",
                mode="local",
                job_id=patch.job_id,
                scene_revision=target_revision,
                write_windows=(full,),
                read_windows=(full,),
                patch_paths=(published_dir,),
                diagnostics={
                    "met_fast_path": True,
                    "met_fast_path_times": changed_times,
                    "met_fast_path_variables": tuple(sorted(changed_variables)),
                    "met_fast_path_seconds": elapsed,
                },
            ),
            None,
        )

    def _met_warm_path_eligible(self, committed: Mapping[str, tuple]) -> bool:
        """A met-only batch of any variable affinity (G2.1 item 4 gate).

        The r3a fast path serves ``utci_only`` batches whose plan the
        planner narrowed; RADIATION-AFFECTING met edits keep the
        conservative FULL closure (the thermal state chain carries the
        changed inputs forward across k), which is exactly the case the
        warm-sparse consumer addresses. Mixed batches (vegetation, paint,
        parameters, massing) never enter — their dirty closure is not
        time-axis-local.
        """
        return set(committed) == {METEOROLOGY_SOURCE_NODE}

    def _run_met_warm_sparse_path(
        self,
        committed: dict[str, tuple],
        model_parameters: dict | None,
        *,
        target_revision: int,
    ) -> tuple[JobOutcome | None, str | None]:
        """Serve a radiation-affecting met edit's suffix warm (G2.1 item 4).

        The first consumer of the temporal checkpoints: an edit whose
        first changed row is ``r0`` pays only the suffix today — rows
        ``t < r0`` are provably unchanged (the thermal chain is causal:
        state@k depends only on inputs ``0..k-1``) and keep being served
        from the store under the SAME strict FULL-coverage gate the r3a
        fast path uses, while the suffix ``r0..T`` is solved through the
        warm seam from the newest fingerprint-matching checkpoint@k
        (``k <= r0``; the met-prefix digest over rows ``0..k-1`` is the
        applicability key — a checkpoint whose prefix covers an edited
        row can never match, whatever its revision). The split at ``r0``
        re-anchors: the state@r0 the suffix consumed is staged for the
        publication hook, so the NEXT edit at a row ``>= r0`` warms from
        it (~24 -> ~4 full-tile steps on the honest row-20 shape).

        No match, a torn checkpoint, or a missing prefix coverage is
        never an error surfaced to the edit: an unmet condition RAISES
        :exc:`MetFastPathRefused` (the caller routes today's exact full
        solve and attaches the reason), and a warm miss without a
        structural refusal still publishes — the prefix simply replays
        COLD from step 0 (the split solve is bitwise the cold replay by
        the item-2 parity fence) with the reason as fallback telemetry.
        """
        started = time.monotonic()

        def refuse(reason: str) -> None:
            _LOG.info("met warm sparse path refused: %s", reason)
            raise MetFastPathRefused(reason)

        if not self._met_warm_path_eligible(committed):
            refuse(
                f"batch changes {sorted(committed)}: the warm path serves "
                "met-only batches"
            )
        for delta in committed[METEOROLOGY_SOURCE_NODE]:
            if not isinstance(delta, ForcingDelta):
                refuse("batch carries a non-forcing meteorology delta")
        if self._staged_landcover_value_diff:
            refuse("batch stages land-cover edits: full closure required")
        if model_parameters != self._applied_model_parameters:
            refuse(
                "batch changes model parameters the published radiation "
                "state did not run under"
            )

        # r0 — the first row the staged forcing actually changes: rows
        # below it are unchanged, so their published values stay valid.
        # ONE forcing materialization serves the diff, the fingerprint
        # digests, and both solve phases (the loader is not free).
        baseline = load_site_forcing(
            self.cache,
            site_dir=self.site_dir,
            selected_date_str=self.selected_date_str,
        ).met_table
        forcing = self._worker.forcing()
        staged_table = forcing.met_table
        applied_table = (
            self._applied_forcing_overlay.resolve(baseline)
            if self._applied_forcing_overlay is not None
            else baseline
        )
        changed_rows = np.any(staged_table != applied_table, axis=1)
        changed = np.flatnonzero(changed_rows)
        if changed.size == 0:
            refuse("overlay resolves value-equal to the published state")
        r0 = int(changed[0])
        total = int(staged_table.shape[0])
        if r0 == 0:
            refuse(
                "edit starts at timestep 0: no unchanged prefix to serve "
                "from the store"
            )

        # The existing strict serve gate, per prefix row and per published
        # variable: every t < r0 must be FULLY covered or the store cannot
        # serve the prefix and the batch replays full. The store speaks
        # NODE ids (``shadow`` patches index under ``time_shadow``), so the
        # vocabulary maps through VARIABLE_TO_RESULT_NODE — a name/id mixup
        # here would refuse every shadow-carrying vocabulary even when the
        # prefix is fully published.
        full = self.grid.full_window
        requested = tuple(self._worker.requested_variables)
        for variable in requested:
            node_id = VARIABLE_TO_RESULT_NODE.get(variable, variable)
            for t in range(r0):
                coverage = self.store.window_coverage(node_id, t, window=full)
                if coverage.status is not CoverageStatus.FULL:
                    refuse(
                        f"{variable}@{t} coverage is "
                        f"{coverage.status.value}: the unchanged prefix is "
                        "not fully published"
                    )

        # Checkpoint scan, newest-first: the newest record whose input
        # fingerprint matches the CURRENT inputs at its own next_step
        # wins. Torn/corrupt records are skipped (typed rejection
        # consumed, never surfaced); records past r0 cannot match — their
        # met prefix covers the edited row — and are skipped up front.
        # T10: the applicability decision routes through the torch-free
        # classifier (solweig_core.numba_cpu.thermal.classify_anchor),
        # which SEPARATES geometry-history invalidation (the scene
        # lineage digests — a previous scene's checkpoint can never
        # serve, whatever its clock position) from met-prefix
        # invalidation (a time-axis boundary property, digested at the
        # CANDIDATE's own next_step). The digest SPELLING stays the
        # worker's shared ``thermal_fingerprint``; the scene-lineage
        # digests are next_step independent and computed once per batch.
        from solweig_core.numba_cpu.thermal import (
            GEOMETRY_FINGERPRINT_KEYS as _GEOMETRY_KEYS,
            CheckpointCandidate as _CheckpointCandidate,
            classify_anchor as _classify_anchor,
        )

        checkpoint_root = self.results_root / self._scenario_id / "checkpoints"
        selected = None
        skip_reason: str | None = None
        geometry_fingerprint = {
            key: digest
            for key, digest in self._worker._thermal_fingerprint_for(
                forcing, {"next_step": r0}
            ).items()
            if key in _GEOMETRY_KEYS
        }

        def _met_digest(next_step: int) -> str:
            return self._worker._thermal_fingerprint_for(
                forcing, {"next_step": next_step}
            )["met_prefix"]

        for revision in reversed(list_checkpoints(checkpoint_root)):
            try:
                record = load_checkpoint(
                    checkpoint_root / f"rev-{revision:06d}"
                )
            except CheckpointError as error:
                if skip_reason is None:
                    skip_reason = (
                        f"newest readable-looking checkpoint "
                        f"rev-{revision:06d} is torn/corrupt ({error})"
                    )
                continue
            reason, _invalidation = _classify_anchor(
                _CheckpointCandidate(
                    scene_revision=record.scene_revision,
                    next_step=record.next_step,
                    thermal=bool(record.thermal_tensors),
                    input_fingerprint=record.input_fingerprint,
                ),
                geometry_fingerprint=geometry_fingerprint,
                met_prefix_digest=_met_digest,
                r0=r0,
            )
            if reason is None:
                selected = record
                break
            if skip_reason is None:
                skip_reason = reason
        if selected is None and skip_reason is None:
            skip_reason = f"no checkpoint under {checkpoint_root}"

        if selected is not None:
            k = int(selected.next_step)
            initial_state = {
                name: selected.thermal_tensors[name]
                for name in THERMAL_TENSOR_NAMES
            }
            initial_state["CI"] = selected.thermal_scalars["CI"]
            initial_state["firstdaytime"] = selected.thermal_scalars[
                "firstdaytime"
            ]
            initial_state["timeadd"] = selected.thermal_scalars["timeadd"]
            # Absent Twater = not yet established (the pre-midnight []):
            # the warm core resumes it as not-established, never a guess.
            initial_state["Twater"] = selected.thermal_scalars.get("Twater")
            initial_state["next_step"] = k
        else:
            k = 0
            initial_state = None

        # The split solve. Phase 1 advances the state to r0 (warm from k
        # when an anchor matched, cold from 0 otherwise); phase 2 is the
        # published suffix. Bitwise the cold replay by the item-2 parity
        # fence applied at both split points.
        scene = compose_full_scene_tensors(self.cache, self.layer)
        solve_kwargs = dict(
            read_window=full,
            write_window=full,
            forcing=forcing,
            requested_variables=requested,
            landcover_overlay=self._worker._landcover_overlay,
            model_parameters=self._worker._model_parameters,
            scene=scene,
            required_read_window=full,
        )
        if k < r0:
            _prefix_outputs, state_r0 = solve_window(
                self.cache,
                self.layer,
                time_start=k,
                time_stop=r0,
                initial_state=initial_state,
                return_final_state=True,
                **solve_kwargs,
            )
        else:
            state_r0 = initial_state
        suffix = solve_window(
            self.cache,
            self.layer,
            time_start=r0,
            initial_state=state_r0,
            **solve_kwargs,
        )

        patch = ResultPatch(
            job_id=new_job_id(),
            scene_revision=target_revision,
            # Full-tile spatial recompute (read == write == the grid): the
            # plan's FULL promise is honored; time-sparse by construction.
            mode="full",
            write_window=full,
            read_window=full,
            site_id=self.cache.site_id,
            tile_key=self.cache.tile_key,
            cache_manifest_sha256=self.cache.metadata()["manifest_sha256"],
            model_version=self.cache.model_version,
            variables=requested,
            arrays={
                name: np.ascontiguousarray(array)
                for name, array in suffix.items()
            },
            time_start=r0,
            time_stop=total,
        )

        # Two-phase publication mirroring the r3a fast path and
        # ExactWorker.run: staged, the supersession checkpoint, the atomic
        # rename — then the suffix anchor is staged for the publication
        # hook (superseded/refused jobs stage nothing).
        scenario_root = self.results_root / self._scenario_id
        staging_root = scenario_root / ".staging"
        staging_root.mkdir(parents=True, exist_ok=True)
        try:
            staged_dir = stage_patch(patch, staging_root)
            self._worker._check_not_superseded(target_revision)
            published_dir = publish_staged_patch(
                staged_dir, scenario_root, patch.scene_revision
            )
        except SupersededJobError as reason:
            discard_staging(staging_root)
            return (
                JobOutcome(
                    status="superseded",
                    mode=None,
                    job_id=patch.job_id,
                    scene_revision=target_revision,
                    diagnostics={"reason": str(reason), "met_warm_path": True},
                ),
                None,
            )
        except BaseException:
            discard_staging(staging_root)
            raise

        self._worker.stage_thermal_capture(
            state_r0,
            self._worker._thermal_fingerprint_for(forcing, state_r0),
        )
        elapsed = round(time.monotonic() - started, 6)
        return (
            JobOutcome(
                status="published",
                mode="full",
                job_id=patch.job_id,
                scene_revision=target_revision,
                write_windows=(full,),
                read_windows=(full,),
                patch_paths=(published_dir,),
                fallback_reason=None if selected is not None else skip_reason,
                diagnostics={
                    "met_warm_path": True,
                    "met_warm_r0": r0,
                    "met_warm_resume_step": k,
                    "met_warm_steps_solved": total - k,
                    "met_warm_checkpoint_revision": (
                        selected.scene_revision if selected is not None else None
                    ),
                    "met_warm_refusal": (
                        None if selected is not None else skip_reason
                    ),
                    "met_warm_seconds": elapsed,
                },
            ),
            None,
        )

    def _undo_tree_edits(self, edits: Sequence[TreeEdit]) -> None:
        """Reverse a replayed batch (public mutators, reverse order)."""
        for edit in reversed(tuple(edits)):
            if edit.old_tree is None and edit.new_tree is not None:
                self.layer.delete_tree(edit.tree_id)
            elif edit.new_tree is None and edit.old_tree is not None:
                self.layer.add_tree(edit.old_tree)
            else:
                old, new = edit.old_tree, edit.new_tree
                if (new.x_m, new.y_m) != (old.x_m, old.y_m):
                    self.layer.move_tree(edit.tree_id, x_m=old.x_m, y_m=old.y_m)
                changes = {
                    name: getattr(old, name)
                    for name in ("height_m", "canopy_radius_m", "trunk_ratio", "transmissivity")
                    if getattr(new, name) != getattr(old, name)
                }
                if changes:
                    self.layer.update_tree(edit.tree_id, **changes)

    def _replay_tree_edits(self, edits: Sequence[TreeEdit]) -> None:
        """Replay coalesced edits into the layer, atomically.

        Pre-flight simulates the whole chain against the live tree map (the
        coalesced batch guarantees contiguous before/after states, so the
        simulation is exact); only a chain that fully applies is replayed
        into the layer. The layer's own ``_record`` assigns sequences.
        """
        scenario_mismatch = [
            edit for edit in edits if edit.scenario_id != self._scenario_id
        ]
        if scenario_mismatch:
            raise ExecutorError(
                "tree edits target scenario "
                f"{scenario_mismatch[0].scenario_id!r}, not {self._scenario_id!r}"
            )

        # Pre-flight: apply to a scratch map first.
        live = {
            tree.tree_id: tree for tree in self.layer.current_trees()
        }
        for edit in edits:
            current = live.get(edit.tree_id)
            if edit.old_tree is None:
                if current is not None:
                    raise ExecutorError(
                        f"replay would re-add existing tree {edit.tree_id!r}"
                    )
                live[edit.tree_id] = edit.new_tree
            elif edit.new_tree is None:
                if current is None:
                    raise ExecutorError(
                        f"replay would delete unknown tree {edit.tree_id!r}"
                    )
                del live[edit.tree_id]
            else:
                if current is None or current.tree_id != edit.old_tree.tree_id:
                    raise ExecutorError(
                        f"replay chain for tree {edit.tree_id!r} is not "
                        "contiguous with the layer state"
                    )
                live[edit.tree_id] = edit.new_tree

        # Real pass: public mutators only.
        for edit in edits:
            if edit.old_tree is None:
                self.layer.add_tree(edit.new_tree)
            elif edit.new_tree is None:
                self.layer.delete_tree(edit.tree_id)
            else:
                new = edit.new_tree
                old = edit.old_tree
                if (new.x_m, new.y_m) != (old.x_m, old.y_m):
                    self.layer.move_tree(edit.tree_id, x_m=new.x_m, y_m=new.y_m)
                changes = {
                    name: getattr(new, name)
                    for name in ("height_m", "canopy_radius_m", "trunk_ratio", "transmissivity")
                    if getattr(new, name) != getattr(old, name)
                }
                if changes:
                    self.layer.update_tree(edit.tree_id, **changes)

    # -- SEAMS (u-c2/u-c3/u-c4): overridden/extended by their packets ----

    def _physics_kernel_arguments(
        self, committed: Mapping[str, tuple[SourceDelta, ...]]
    ) -> dict[str, Any] | None:
        """u-c2 seam (LANDED by u-c7): committed params -> kernel arguments.

        The batch's committed ``model_parameters`` deltas fold over the
        scenario's ACCUMULATED parameter state — the last-write-wins fold
        of every parameter delta already published — through
        :func:`kernel_arguments_from_deltas`, which re-validates every
        name and domain and float()-coerces float-kind values at THIS
        seam: a bypass-built or blocked delta is refused here
        (fail-fast), before any state advances, so a bad parameter can
        never leave a half-executed revision behind. The parameters are
        time-invariant, so the payload is a flat ``name -> value``
        mapping with no time dimension.

        Returns ``None`` when no parameter edit has ever been published
        and the batch carries none — the parameter-free baseline run,
        byte-identical to the pre-u-c7 physics. The staged payload rides
        the job (the worker forwards it into ``run_utci_window`` on both
        solve paths) and commits into the accumulated state only on
        publication.
        """
        batch = committed.get(MODEL_PARAMETERS_SOURCE_NODE, ())
        accumulated = self._applied_model_parameters
        if not batch:
            # No new parameter edits: the job re-runs under the published
            # parameter state (folded unchanged), or the baseline when no
            # parameter edit has ever been published.
            if not accumulated:
                return None
            return dict(accumulated)
        if not all(
            isinstance(delta, ModelParameterDelta) for delta in batch
        ):
            bad = sorted(
                type(delta).__name__
                for delta in batch
                if not isinstance(delta, ModelParameterDelta)
            )
            raise SourceDeltaError(
                f"model_parameters deltas must be ModelParameterDelta "
                f"records, got {bad}"
            )
        # Batch fold on top of the accumulated state: batch changes win
        # (last-write-wins per name, arrival order — the coalescer's
        # semantics), and the adapter's fold re-validates every name and
        # domain so nothing staged downstream can bypass the fence.
        folded = dict(accumulated) if accumulated else {}
        folded.update(kernel_arguments_from_deltas(batch))
        return folded

    def _scenario_forcing(
        self, committed: Mapping[str, tuple[SourceDelta, ...]]
    ) -> SiteForcing:
        """u-c3 seam: ``overlay_from_deltas`` -> ``load_site_forcing``.

        The batch's committed forcing deltas fold (last-write-wins per
        (variable, time)) on top of the scenario's accumulated overlay, and
        the loader materializes the resolved float64 table: the baseline
        staleness check first, then the masked re-check OUTSIDE the
        overlay's declared cells, then the UNRELAXED solar cross-check —
        the resolve is verified at this seam, never trusted. The returned
        forcing carries the overlay (:attr:`SiteForcing.overlay`), which is
        how the full-tile fallback knows to stage the RESOLVED table in its
        scratch site instead of copying the baseline met text. The worker
        consumes the same overlay through
        :meth:`ExactWorker.stage_forcing_overlay` (single materialization
        point, identical physics on the local and the full path).
        """
        return load_site_forcing(
            self.cache,
            site_dir=self.site_dir,
            selected_date_str=self.selected_date_str,
            overlay=self._effective_forcing_overlay(committed),
        )

    def _effective_forcing_overlay(
        self, committed: Mapping[str, tuple[SourceDelta, ...]]
    ) -> ForcingOverlay | None:
        """The scenario forcing overlay the next job runs under (u-c3).

        Accumulated scenario state: the batch's committed ``meteorology``
        deltas (re-validated by :func:`overlay_from_deltas` — the fence,
        the documented domains, and the duplicate check run on this seam,
        so a bypass-built delta is refused here, not trusted) fold on top
        of every overlay change already published, last-write-wins per
        (variable, time). Without either source this is ``None`` — the
        legacy baseline forcing, byte-identical.
        """
        batch_deltas = tuple(
            delta
            for delta in committed.get(METEOROLOGY_SOURCE_NODE, ())
            if isinstance(delta, ForcingDelta)
        )
        if not batch_deltas:
            return self._applied_forcing_overlay
        batch_overlay = overlay_from_deltas(batch_deltas)
        folded: dict[tuple[str, int | None], ForcingChange] = {}
        if self._applied_forcing_overlay is not None:
            for change in self._applied_forcing_overlay.changes:
                folded[(change.variable, change.time_index)] = change
        for change in batch_overlay.changes:
            folded[(change.variable, change.time_index)] = change
        # ARRIVAL order across the accumulated/batch boundary (u-c3 L2):
        # accumulated changes first (in the order they were published),
        # then this batch's changes. The overlay resolves its list
        # last-write-wins per variable, so this fold makes a NEWER
        # whole-series batch edit override OLDER accumulated per-step
        # values — sorting whole-series records first here would invert
        # that priority and resurrect published per-step values the user
        # just overwrote wholesale ([25, 26, 25] instead of [25, 25, 25]).
        # Within one batch, overlay_from_deltas already applied its own
        # deterministic whole-series-then-per-step order.
        merged = tuple(folded.values())
        return ForcingOverlay(changes=merged)

    def _stage_overlays(
        self, committed: Mapping[str, tuple[SourceDelta, ...]]
    ) -> LandCoverOverlay | None:
        """u-c4 seam (LANDED): ``landcover_overlay_from_deltas`` -> solver.

        The batch's committed paint deltas fold (per-cell last-write-wins
        AT COMMIT, coalesced — see :meth:`_effective_landcover_overlay`)
        on top of the scenario's accumulated overlay, and the fence
        re-runs inside :func:`landcover_overlay_from_deltas` (a
        bypass-built delta is refused at this seam, never trusted). The
        seam then materializes the resolved grid through the solver seam
        (:func:`solver.resolve_landcover_overlay`) fail-fast — the masked
        re-check and the landcover-raster presence check run BEFORE any
        state advances, so a bad overlay can never leave a half-executed
        revision behind — and stages it on the worker, whose pending
        watermark rides it into the job. Returns the effective overlay so
        :meth:`execute_plan` can commit it on publication.

        u-c6 (u-c4-review M3) + u-c6b (u-c6-review H1, lead ruling): the
        staged dirty footprint is a SUPERSET of the plan's promise — the
        batch's STROKE windows (the committed paint patches the planner
        scoped the plan's write windows from; the planner never sees
        resolved values) UNION the symmetric difference between the staged
        and PUBLISHED overlay states. The value-diff alone under-promised:
        a paint over cells already carrying the target class dirtied only
        the changing sub-rectangle, so the published write windows
        (diff + margin) did not cover the plan's stroke-footprint windows
        and the whole batch was refused. Recomputing same-value stroke
        cells is wasted-but-correct; under-recomputing breaks the plan
        contract. The u-c6 growth win is preserved: only the CURRENT
        stroke dirties — prior published footprints stay out of the
        current batch's dirty windows, and the accumulated overlay stays
        coalesced (O(edited area), never O(session strokes)).

        u-c6b L5: ONE baseline copy and ONE staged resolve per batch. The
        fail-fast validation resolve is the staged fold's only full-grid
        resolve — it feeds the coalescing fold and the value difference
        unchanged — and the published side reads the resolve memo
        committed at the last publication
        (``_applied_landcover_resolved``).
        """
        batch_deltas = tuple(
            delta
            for delta in committed.get(LANDCOVER_SOURCE_NODE, ())
            if isinstance(delta, LandCoverPaintDelta)
        )
        fold = self._effective_landcover_overlay(committed)
        baseline: np.ndarray | None = None
        if fold is not None:
            # Fail-fast materialization of the RAW fold FIRST: the
            # landcover-raster presence check (typed refusal on a
            # landcoverless site) and the masked re-check run BEFORE any
            # state advances — and BEFORE the baseline copy the coalescing
            # fold reads — so the u-c4 refusal contract is unchanged and a
            # site without a land-cover raster never reaches the baseline
            # seam at all. L5: the returned grid is the staged fold's
            # one-and-only resolve; coalescing preserves it (the canonical
            # patches carry its values on the painted mask).
            staged_resolved = resolve_landcover_overlay(self.cache, fold)
            baseline = np.array(self.cache.landcover)
            overlay = coalesce_overlay(
                fold, baseline, resolved=staged_resolved
            )
        else:
            overlay = self._applied_landcover_overlay
            staged_resolved = self._applied_landcover_resolved
        published_grid = self._applied_landcover_resolved
        if published_grid is None and staged_resolved is not None:
            published_grid = baseline
        if staged_resolved is None or staged_resolved is published_grid:
            # No land-cover content can differ: either the scenario has no
            # land-cover source at all, or this batch re-stages the
            # accumulated overlay unchanged (no paint deltas committed), so
            # the staged and published states resolve to the SAME grid
            # object without materializing the baseline copy.
            self._staged_landcover_value_diff = ()
        else:
            # Reachable only when a fold staged this batch, so ``baseline``
            # is materialized here by construction.
            self._staged_landcover_value_diff = overlay_diff_windows(
                overlay,
                self._applied_landcover_overlay,
                baseline,
                resolved_new=staged_resolved,
                resolved_old=published_grid,
            )
        # Stroke ∪ value-diff (diff ⊆ stroke for a rectangular stroke over
        # changing values; the union covers partial-over-target strokes
        # and any diff cells a later stroke reverts). Prior published
        # footprints are NOT here: only the current stroke dirties.
        stroke_windows = tuple(
            patch.window for delta in batch_deltas for patch in delta.patches
        )
        dirty_windows = (
            tuple(
                merge_windows(
                    [*stroke_windows, *self._staged_landcover_value_diff],
                    gap_pixels=0,
                )
            )
            if stroke_windows or self._staged_landcover_value_diff
            else ()
        )
        self._staged_landcover_resolved = staged_resolved
        self._worker.stage_landcover_overlay(overlay, dirty_windows=dirty_windows)
        return overlay

    def _effective_landcover_overlay(
        self, committed: Mapping[str, tuple[SourceDelta, ...]]
    ) -> LandCoverOverlay | None:
        """The scenario land-cover overlay fold for the next job (u-c4).

        Accumulated scenario state: the batch's committed ``landcover``
        deltas (re-validated by :func:`landcover_overlay_from_deltas` — the
        class fence and window sanity run on this seam) append their
        patches after every paint already published, last-write-wins per
        cell at apply time. The returned fold is RAW (patch
        concatenation); :meth:`_stage_overlays` materializes it fail-fast
        through the solver seam and COALESCES it per-cell against the
        baseline grid (:func:`.adapters.landcover.coalesce_overlay`,
        u-c6), so the accumulated overlay that commits on publication is
        the canonical rectangle decomposition of the session's painted
        cells, not its stroke history. Without either source this is
        ``None`` — the legacy baseline class grid, byte-identical.
        """
        batch_deltas = tuple(
            delta
            for delta in committed.get(LANDCOVER_SOURCE_NODE, ())
            if isinstance(delta, LandCoverPaintDelta)
        )
        if not batch_deltas:
            return self._applied_landcover_overlay
        batch_overlay = landcover_overlay_from_deltas(batch_deltas)
        patches: list[LandCoverPaintPatch] = []
        if self._applied_landcover_overlay is not None:
            patches.extend(self._applied_landcover_overlay.patches)
        patches.extend(batch_overlay.patches)
        return LandCoverOverlay(patches=tuple(patches))

    def _prepare_building_batch(
        self, committed: Mapping[str, tuple[SourceDelta, ...]]
    ) -> tuple[MassingEdit, ...]:
        """Committed building deltas -> re-validated massing edits (u-c5).

        ``massing_edits_from_deltas`` re-runs the full property schema on
        this seam (the staging door's discipline), so a bypass-built delta
        is refused here, never inside the raster write. Empty return means
        the batch carries no building family — the windowed worker path
        runs instead. A non-empty return routes the WHOLE batch through
        :meth:`_regenerate_building_chain`: building edits are FULL-only
        (the planner's ``full_only_initially`` ruling), so even a mixed
        batch regenerates full tile with the batch's other families
        riding the chain as overlays/layer.
        """
        building_deltas = committed.get(BUILDING_SOURCE_NODE, ())
        if not building_deltas:
            return ()
        return massing_edits_from_deltas(building_deltas)

    def _staged_massing_fold(
        self, batch_edits: Sequence[MassingEdit]
    ) -> dict[str, MassingEdit]:
        """Applied massing fold + this batch's edits -> staged fold (u-d4c).

        The fold is the executor-level ACCUMULATION of every building edit
        published this session, keyed by ``building_id``, LAST-WRITE-WINS
        per id:

        * a NEW id folds in as-is (its ``before`` is the BASELINE anchor
          for session-added buildings — ``None``);
        * an id already in the fold keeps its ORIGINAL ``before`` (the
          fold's anchor must never drift: the rasterizer applies edits
          placement-style against the BASELINE-staged site, so ``before``
          is a disclosure anchor, not a raster lookup) and takes the
          incoming ``after``;
        * a DELETE stays a delete — for a BASELINE id the entry keeps
          ``before`` = baseline spec with ``after is None`` forever; for a
          SESSION-ADDED id the entry becomes a TOMBSTONE carrying the
          deleted session spec (``before`` = the added spec,
          ``after is None``) instead of vanishing. A tombstone never
          empties the fold: ``regenerate_building_batch`` refuses empty
          edit tuples, and the alternative — routing the worker — would
          hit its nothing-pending no-op gate (the batch's building delta
          is not worker state) and raise. Raster-wise the tombstone is a
          no-op against the baseline-staged site (its footprint is reset
          to DEM ground the baseline already carries), so the chain
          publishes baseline-equal products with SCENARIO provenance at
          the new revision — exactly what "I deleted my added building"
          owes the client.

        ORDER-AND-OVERLAP DISCLOSURE (u-d4c remediation): the chain
        applies the fold as ``tuple(massing_fold.values())`` — dict
        FIRST-INSERTION order, so each id's position is fixed by its
        FIRST appearance this session, NOT by the session order of its
        latest edit. The rasterizer
        (:meth:`~solweig_gpu.incremental.buildings.BuildingLayer.apply_edits`)
        replays that tuple sequentially: a delete's before-footprint
        reset is UNCONDITIONAL (``raster[reset_mask] = ground[reset_mask]``
        — it erases bystander paint at overlaps) and after-paints combine
        by ``np.fmax`` (taller wins). Overlapping DISTINCT building ids
        therefore resolve by APPLICATION ORDER, which can differ from
        session-order sequential replay: add A (footprint FA), add B
        (FB overlapping FA), delete A folds to
        ``[A_tombstone(reset FA), B(paint FB)]`` and keeps B INTACT at
        FA∩FB, while a session-order replay ``[A, B, A_delete]`` would
        TRUNCATE B there (the delete's reset lands last). Deleting the
        LATER-inserted id instead (``[A, B_tombstone]``) truncates A at
        the overlap in BOTH orderings — the reset lands after A's paint —
        so the divergence is asymmetric. It is bounded to the overlap
        cells of distinct building ids (never same-id cells, which
        last-write-wins resolves exactly): the adapter carries no
        building inventory, so the fold cannot express cross-id
        occlusion, and per-id last-write-wins vs the baseline anchor is
        the mission-endorsed semantics.

        The returned dict is a STAGED COPY: the caller commits it into
        :attr:`_applied_massing_edits` only on publication, so a failed or
        superseded batch leaves the published fold untouched.
        """
        fold = dict(self._applied_massing_edits)
        for edit in batch_edits:
            current = fold.get(edit.building_id)
            if current is None:
                fold[edit.building_id] = edit
            elif edit.after is None and current.before is None:
                # Session-added id, now deleted: keep a tombstone so the
                # fold (and the chain routing it mandates) survives.
                fold[edit.building_id] = MassingEdit(
                    edit.building_id, before=edit.before, after=None
                )
            else:
                fold[edit.building_id] = MassingEdit(
                    edit.building_id, before=current.before, after=edit.after
                )
        return fold

    def _regenerate_building_chain(
        self,
        edits: Sequence[MassingEdit],
        *,
        target_revision: int,
        forcing_overlay: ForcingOverlay | None,
        landcover_overlay: LandCoverOverlay | None,
        model_parameters: Mapping[str, Any] | None = None,
        chain_carries_new_edits: bool = False,
    ) -> JobOutcome:
        """u-c5 seam (LANDED by u-c6): accumulated massing -> full chain.

        The whole scenario chain regenerates under the executor's scratch
        root (the baseline site and cache are read-only inputs — u-c5's
        hash-verified invariant), with the scenario's CURRENT state riding
        the regeneration: the ACCUMULATED building fold (u-d4c: ``edits``
        is the full published fold, not just this batch's new deltas, so
        buildings published by EARLIER batches survive every later batch),
        the live tree layer (``tree_layer=self.layer``,
        so vegetation accumulated by earlier published batches survives a
        building batch), the effective forcing overlay, the effective
        land-cover overlay (threaded through ``run_scenario_full_tile`` ->
        ``run_full_tile(landcover_overlay=...)``; the chain had no
        land-cover seam before u-c6, and a mixed building+paint batch
        would have silently dropped the paint), and the effective model
        parameters (u-c7, threaded the same way into
        ``run_full_tile(model_parameters=...)`` so a mixed
        building+params batch never silently drops the parameter edits).
        ``chain_carries_new_edits`` records whether this batch itself
        contributed new building deltas (routing disclosure only — a
        chain-routed batch with NO new building edits is the u-d4c
        accumulation path: the other family rides the chain precisely so
        the published buildings survive it).

        Results publish through the SAME two-phase discipline the worker
        uses (``stage_patch`` -> ``publish_staged_patch`` into
        ``results/<scenario>/rev-{N:06d}-{job}``), so the u-c1b
        publication contract holds unchanged: per-(variable, time,
        window) store entries via :meth:`_record_outcome`, watermark
        rollback via :meth:`_rollback_failed_batch` (which also wipes the
        scenario tree), single-writer store attribution, and orphan
        cleanup. The patch's provenance is the SCENARIO cache the chain
        rebuilt (site id, manifest hash, model version), so readers can
        never mistake a regenerated product for a baseline-cache product.
        """
        job_id = new_job_id()
        started = time.time()  # provenance only; never a decision input
        # T6 (u-c6-review): one read of the worker's requested-variable
        # vocabulary for the whole chain — the regeneration call and the
        # patch record below consume the same tuple.
        variables = tuple(self._worker.requested_variables)
        # u-d1 fence (U-D intake item c): fail LOUDLY when the requested
        # vocabulary has widened beyond what the regeneration chain was
        # validated to produce, instead of silently forwarding names the
        # chain would drop or mis-render deep inside the raster solve.
        unsupported = sorted(set(variables) - _BUILDING_CHAIN_VARIABLES)
        if unsupported:
            raise ExecutorError(
                "building regeneration chain cannot produce requested "
                f"variables {unsupported}; the chain's validated vocabulary "
                f"is {sorted(_BUILDING_CHAIN_VARIABLES)}. Widen "
                "regenerate_building_batch first (u-d1 intake c fence) — "
                "never silently request variables the chain does not "
                "implement."
            )
        try:
            result = regenerate_building_batch(
                edits=edits,
                baseline_site_dir=self.site_dir,
                baseline_cache=self.cache,
                scenario_root=self._scenario_regenerate_root,
                tree_layer=self.layer,
                selected_date_str=self.selected_date_str,
                requested_variables=variables,
                forcing_overlay=forcing_overlay,
                landcover_overlay=landcover_overlay,
                model_parameters=model_parameters,
            )
        except RegenerationError as error:
            raise ExecutorError(
                "building regeneration chain failed (the baseline is "
                "untouched and the scenario tree is wiped on rollback): "
                f"{error}"
            ) from error

        def _superseded(reason: str) -> JobOutcome:
            discard_staging(self.results_root / self._scenario_id / ".staging")
            shutil.rmtree(self._scenario_regenerate_root, ignore_errors=True)
            return JobOutcome(
                status="superseded",
                mode=None,
                job_id=job_id,
                scene_revision=target_revision,
                diagnostics={"reason": reason},
            )

        # Supersession checkpoint 1 (worker parity): the revision must
        # still be the one this batch advanced to.
        if self.scene_revision != target_revision:
            return _superseded("scene revision advanced mid-flight")

        scenario_cache = result.cache
        patch = ResultPatch(
            job_id=job_id,
            scene_revision=target_revision,
            mode="full",
            write_window=self._grid.full_window,
            read_window=self._grid.full_window,
            site_id=scenario_cache.site_id,
            tile_key=scenario_cache.tile_key,
            cache_manifest_sha256=scenario_cache.metadata()["manifest_sha256"],
            model_version=scenario_cache.model_version,
            variables=variables,
            arrays={
                name: np.ascontiguousarray(result.outputs[name])
                for name in variables
            },
            time_start=0,
            time_stop=scenario_cache.time_steps,
        )
        scenario_root = self.results_root / self._scenario_id
        staging_root = scenario_root / ".staging"
        staging_root.mkdir(parents=True, exist_ok=True)
        staging_dir = stage_patch(patch, staging_root)
        # Supersession checkpoint 2: before the rename into the results
        # root (a revision bump mid-publish must leave nothing published).
        if self.scene_revision != target_revision:
            return _superseded("scene revision advanced mid-flight")
        try:
            patch_path = publish_staged_patch(
                staging_dir, scenario_root, patch.scene_revision
            )
        except BaseException:
            discard_staging(staging_root)
            raise
        full = self._grid.full_window
        return JobOutcome(
            status="published",
            mode="full",
            job_id=job_id,
            scene_revision=target_revision,
            write_windows=(patch.write_window,),
            read_windows=(patch.read_window,),
            patch_paths=(patch_path,),
            diagnostics={
                "dirty_windows": [
                    (
                        full.row_start,
                        full.row_stop,
                        full.col_start,
                        full.col_stop,
                    )
                ],
                "dirty_fraction": 1.0,
                "elapsed_seconds": time.time() - started,
                "building_batch": True,
                "building_fold_size": len(edits),
                "chain_carries_new_edits": chain_carries_new_edits,
                "regeneration_stages": list(result.stages),
                "building_disclosures": list(result.records.disclosures),
                "forcing_overlay_pending": (
                    forcing_overlay != self._applied_forcing_overlay
                ),
                "landcover_overlay_pending": (
                    landcover_overlay != self._applied_landcover_overlay
                ),
            },
        )

    # ------------------------------------------------------------------
    # Outcome recording (store + reconciliation)
    # ------------------------------------------------------------------

    def _record_outcome(
        self,
        plan: ImpactPlan,
        outcome: JobOutcome,
        requested: tuple[str, ...],
        unpublished: tuple[tuple[str, str], ...],
    ) -> ExecutedPlan:
        if outcome.status == "superseded":
            return ExecutedPlan(
                plan=plan,
                status="superseded",
                mode=None,
                job_id=outcome.job_id,
                scene_revision=outcome.scene_revision,
                requested_variables=requested,
                unpublished_variables=unpublished,
                diagnostics=outcome.diagnostics,
            )
        if outcome.status == "no-op":
            raise ExecutorError(
                "plan demanded recomputation but the worker found nothing "
                f"pending ({outcome.diagnostics.get('reason', 'no-op')}). "
                "Either every value in this batch already matched the "
                "published scenario state (a value-equality no-op the "
                "executor intercepts whole-batch — reaching this point "
                "means that interception was bypassed), or this is double "
                "execution, a mismatched transaction, or worker watermarks "
                "left advanced by an earlier failed batch (see "
                "ExactWorker.rollback_pending)"
            )

        self._reconcile_scope(plan, outcome)

        # Metadata-only patch reads (u-c1b L8): recording indexes each
        # patch into the (node, time) store — it needs provenance,
        # windows, and bounds, not array content. Full checksum
        # verification stays in load_patch at the point of content reads.
        patch_records = [
            (path, load_patch_metadata(path)) for path in outcome.patch_paths
        ]
        store_keys = self._record_temporal_entries(outcome, patch_records)

        # G2.0 write-policy hook: the batch is fully published at this
        # revision, so offer a temporal checkpoint. G2.1 item 3: a
        # FULL-solve publication carries the worker's captured thermal
        # state (with the input fingerprint computed at capture) — the
        # warm-state anchor; everything else (local jobs, no-op batches)
        # stays coverage-only. Checkpoints are reproduce aids, NEVER a
        # source of truth ahead of the op log / result store: a failed
        # write must not fail the published batch, so it is best-effort
        # here.
        try:
            capture = self._worker.take_pending_thermal_capture()
            if capture is not None:
                record = thermal_checkpoint(
                    scene_revision=outcome.scene_revision,
                    store=self.store,
                    final_state=capture["final_state"],
                    input_fingerprint=capture["fingerprint"],
                )
            else:
                record = coverage_checkpoint(
                    scene_revision=outcome.scene_revision, store=self.store
                )
            write_checkpoint(
                record,
                self.results_root / self._scenario_id / "checkpoints",
            )
        except Exception as error:  # noqa: BLE001 — best-effort by contract
            # Publication stands, ALWAYS: a checkpoint is a reproduce aid,
            # never a source of truth, so any fault class here is telemetry
            # (G2.2 chaos suite consumes these), never a failed edit.
            _LOG.warning(
                "checkpoint write fault after publish at revision %s "
                "(publication stands): %s",
                outcome.scene_revision,
                error,
            )

        return ExecutedPlan(
            plan=plan,
            status="published",
            mode=outcome.mode,
            job_id=outcome.job_id,
            scene_revision=outcome.scene_revision,
            write_windows=outcome.write_windows,
            read_windows=outcome.read_windows,
            patch_paths=outcome.patch_paths,
            store_keys=store_keys,
            requested_variables=requested,
            unpublished_variables=unpublished,
            fallback_reason=outcome.fallback_reason,
            diagnostics=dict(outcome.diagnostics),
        )

    def _reconcile_scope(self, plan: ImpactPlan, outcome: JobOutcome) -> None:
        """The executed scope must cover the plan's promise — never less.

        Coverage (scope items e/f): every plan write window must lie inside
        a published write window (or the published mode is full tile), and
        every adapter-declared read window must lie inside the published
        read union — the L2 wiring's enforcement point: the staged read may
        only grow (see :func:`stage_read_window`), and an under-published
        patch set is a loud error, never a silent partial result.
        """
        published_write_union = outcome.write_windows
        published_read_union = outcome.read_windows
        full_tile = outcome.mode == "full"

        uncovered_writes: list[RasterWindow] = []
        uncovered_reads: list[RasterWindow] = []
        full_promised: list[str] = []
        for impact in plan.node_impacts:
            if impact.spatial_scope is SpatialScope.FULL and not full_tile:
                # The plan promised full-tile products for this node; the
                # worker routed local. That is an under-publication.
                full_promised.append(impact.node_id)
                continue
            if full_tile:
                continue
            for window in impact.write_windows:
                if not any(
                    _window_covers(published, window)
                    for published in published_write_union
                ):
                    uncovered_writes.append(window)
            for window in impact.read_windows:
                if not any(
                    _window_covers(published, window)
                    for published in published_read_union
                ):
                    uncovered_reads.append(window)

        problems: list[str] = []
        if full_promised:
            problems.append(
                "plan marks FULL scope for nodes "
                f"{sorted(set(full_promised))} but the worker published "
                f"{outcome.mode!r}: configure the executor influence policy "
                "so FULL promises route full (never under-publish)"
            )
        if uncovered_writes:
            problems.append(
                f"{len(uncovered_writes)} plan write window(s) not covered "
                "by the published write windows"
            )
        if uncovered_reads:
            problems.append(
                f"{len(uncovered_reads)} adapter-declared read window(s) "
                "not covered by the published read windows (L2 read-halo "
                "wiring: reads may only grow)"
            )
        if problems:
            raise ExecutorError(
                "executed scope does not cover the plan: " + "; ".join(problems)
            )

    def _record_temporal_entries(
        self,
        outcome: JobOutcome,
        patch_records: Sequence[tuple[Path, Any]],
    ) -> tuple[TemporalResultKey, ...]:
        """Record ``(node, time)`` entries for every published patch window.

        One entry per (variable, timestep, write window) (u-c1b M2): a
        multi-window local job publishes one patch per disjoint write
        window (patch ids ``job-w0``, ``job-w1``, ...), and each window's
        entry carries ITS OWN patch path, patch job id, and mode. The
        previous merge-to-one-entry form under-reported coverage — every
        time entry pointed at ``patch[0]`` and claimed only its footprint
        — so ``coverage_gaps`` reported the other windows as holes and
        per-cell readers loaded the wrong patch. The store records the
        window entries together under the same ``(node, time)`` key;
        ``lookup_window`` resolves a cell to the patch that covers it.
        """
        entries: list[TemporalResultEntry] = []
        for path, patch in patch_records:
            if patch.time_indices is not None:
                # Sparse patch (r3a met fast path): it carries EXACTLY the
                # recomputed timesteps — never a prefix, so absent rows must
                # not be recorded (a reader would treat them as zeros).
                step_indices: Iterable[int] = patch.time_indices
            else:
                time_stop = (
                    patch.time_stop
                    if patch.time_stop is not None
                    else self.cache.time_steps
                )
                step_indices = range(patch.time_start, time_stop)
            for variable in patch.variables:
                node_id = VARIABLE_TO_RESULT_NODE.get(variable, variable)
                for time_index in step_indices:
                    entries.append(
                        TemporalResultEntry(
                            node_id=node_id,
                            time_index=time_index,
                            scene_revision=patch.scene_revision,
                            job_id=patch.job_id,
                            mode=patch.mode,
                            write_window=patch.write_window,
                            patch_path=path,
                        )
                    )
        if not entries:
            return ()
        return self.store.publish(entries, writer=self._store_writer_id)

    # ------------------------------------------------------------------
    # Scenario state persistence (U-D intake item a)
    # ------------------------------------------------------------------

    def save_scenario_state(self, directory: str | Path) -> Path:
        """Snapshot the complete scenario state into ``directory``.

        Call BETWEEN batches (after :meth:`execute` returned). The save
        asserts the u-c1b pairing invariant (worker publication
        watermarks equal the executor's accumulated overlays) and refuses
        to write contradictory state; see
        :mod:`solweig_gpu.incremental.scenario_state`.
        """
        from .scenario_state import snapshot_from_executor, write_snapshot

        snapshot = snapshot_from_executor(self)
        return write_snapshot(
            snapshot,
            directory,
            landcover_resolved=self._applied_landcover_resolved,
        )

    def restore_scenario_state(self, directory: str | Path) -> None:
        """Restore a snapshot written by :meth:`save_scenario_state`.

        The executor must be freshly constructed for the SAME scenario
        (site cache, selected date, influence policy are verified; any
        mismatch refuses the whole restore). A fresh process resumes the
        scenario bitwise: the worker adopts the published watermarks and
        re-stages the SAME overlays as its between-batch steady state
        (staged == published arms nothing — restoring never re-arms
        published overlays as pending), the accumulated overlays/resolve
        memo are reloaded, the tree log replays, and the store index is
        rebuilt through the store's own publish discipline.
        """
        from .scenario_state import read_snapshot, restore_into_executor

        restore_into_executor(self, read_snapshot(directory), directory=directory)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _scientific_sources(self, plan: ImpactPlan) -> tuple[str, ...]:
        return tuple(
            node_id
            for node_id in plan.changed_sources
            if self._graph.kind(node_id) is not NodeKind.VIEW
        )

    def _resolve_variables(
        self, edits: Sequence[ValidatedEdit]
    ) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
        """Requested patch variables plus the not-yet-carried extras."""
        names = {"utci"}  # physics floor: run_utci_window always computes it
        for edit in edits:
            names.update(edit.command.requested_outputs)
        return self._split_variables(names)

    def _variables_from_plan(
        self, plan: ImpactPlan
    ) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
        """Resolve variables from the plan alone (fallback vocabulary).

        Only OUTPUT product nodes (utci/tmrt) are recoverable from a plan:
        non-node producible layers (e.g. ``shadow``) are indistinguishable
        from ancestry (``time_shadow`` is a utci ancestor), so callers
        transporting non-node layers must pass ``requested_variables``
        explicitly — :meth:`execute` always does.
        """
        names = {"utci"}
        for impact in plan.node_impacts:
            if impact.node_id in ("utci", "tmrt"):
                names.add(impact.node_id)
        return self._split_variables(names)

    def _split_variables(
        self, names: set[str]
    ) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
        requested = tuple(sorted(names & set(SUPPORTED_VARIABLES)))
        unpublished = tuple(
            (name, UNPATCHABLE_VARIABLE_REASON)
            for name in sorted(names - set(SUPPORTED_VARIABLES))
        )
        return requested, unpublished


#: Reasons recorded for producible-but-not-yet-carried variables.
UNPATCHABLE_VARIABLE_REASON = (
    "producible by run_utci_window but the ResultPatch layer carries only "
    "utci/tmrt/shadow; transport lands with u-c2"
)


def _window_covers(outer: RasterWindow, inner: RasterWindow) -> bool:
    return (
        outer.row_start <= inner.row_start
        and outer.row_stop >= inner.row_stop
        and outer.col_start <= inner.col_start
        and outer.col_stop >= inner.col_stop
    )
