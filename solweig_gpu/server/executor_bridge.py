# SPDX-License-Identifier: GPL-3.0-only
"""Family-aware solver dispatch: ledger-driven routing onto PlanExecutor.

The default solver (:func:`solweig_gpu.server.jobs.make_exact_worker_solver`)
replays the tree ledger into a :class:`TreeLayer` and runs the Phase-5
ExactWorker. That path cannot express the other integrated families —
their edits are not tree events. This module wraps the default solver
with a dispatcher:

* a scenario whose ledger carries **no** family event AND whose store row
  carries **no** durable family flag keeps the legacy solver
  byte-for-byte (zero behavior change for tree scenarios);
* any other scenario routes onto the universal-editing
  :class:`PlanExecutor`, rebuilt per job from the durable snapshot under
  ``<results_root>/<scenario>/executor-state`` plus the ledger events the
  snapshot has not consumed yet. The routing signal is DURABLE (u-e1 F1):
  ``Store.carries_family_edits`` (set atomically with the family-event
  commit, cleared by reset — see ``scenario_carries_family_edits``)
  outlives ``sweep_retention``. Scanning only the request's surviving
  events under-routed: once retention pruned every family event, a later
  tree edit fell back to the legacy baseline-bound solver and published
  WITHOUT the family state — a bitwise-wrong scene.

Rebuild protocol (deterministic, restart-safe):

* ``executor-state/scenario-state.json`` + memo = the standard scenario
  snapshot (schema v2, written only between batches);
* ``executor-state/ledger-coverage.json`` = a sidecar recording the store
  sequence the snapshot already consumed (the snapshot schema is frozen,
  so coverage rides beside it — never inside it);
* a ``reset`` event after the snapshot's coverage invalidates the
  snapshot: the executor is rebuilt fresh from the authoritative tree
  list (``scenario_trees`` — reset wipes it) and only post-reset family
  events replay, which also voids accumulated family overlays, matching
  the reset contract's baseline re-publish exactly;
* when the snapshot restored, pending tree events replay as vegetation
  commands (the snapshot's tree log predates them); when the rebuild is
  fresh the tree base decides: post-reset the base is VOIDED (reset wipes
  ``scenario_trees``) and the window's tree events rebuild it exactly from
  empty (u-e3c NF1 — a tree-only window then publishes a real patch
  instead of refusing an empty batch); any other fresh rebuild seeds the
  authoritative tree list, which already carries every committed tree
  edit, so tree events are not replayed again;
* retention must not create a gap inside the replay window: if the ledger
  no longer carries the full contiguous range ``(covered, watermark]``,
  the job fails with a typed ``scenario_state_unrecoverable`` error
  instead of recomputing from the wrong base.

Coverage promotion protocol (u-d4 remediation H1 — crash-safe by
construction). The runner — not the solver — decides when a result is
durable: a solve returns arrays that the runner may still discard (stale
scene version, a cancel in the guard-to-publish window, a crash before
``Store.publish_result``, or a crash between publish and job completion).
The sidecar's ``covered_sequence`` therefore advances ONLY when a staged
snapshot's producing job is durably **complete** in the store, never
inside the solve that produced it:

* a successful solve STAGES its post-batch snapshot into
  ``executor-state/pending/`` (snapshot + ``pending.json`` metadata),
  written via a staging directory and one atomic ``os.replace`` swap;
* the NEXT family job's reconcile step adopts the staged snapshot (rename
  ``pending/`` into ``generations/gen-<job_id>``, then one atomic sidecar
  flip naming that generation) if and only if its producing job is a
  DIFFERENT job whose durable row is ``complete`` — i.e. its outcome
  (published result, or a no-op re-serve) is durable — and its coverage
  does not pass this request's own watermark; otherwise the staged
  snapshot is dropped;
* a job whose result was discarded (superseded, cancelled, failed, or
  killed mid-flight) never reaches ``complete``, so its staged snapshot is
  dropped and its ledger events stay > ``covered`` — exactly the state it
  would sit in had it never run. The next job replays them
  (``routes_jobs`` cancel contract: "the next edit supersedes the
  cancelled work");
* a RE-RUN of the same job (restart recovery requeues a ``running`` row)
  ignores its own staged snapshot and recomputes deterministically — this
  is what makes the requeued job publish (or hit the runner's checksum
  -verified ``ResultAlreadyPublished`` idempotent completion) instead of
  finding an empty window and failing with a misleading
  ``edit_rejected``.

The invariant: the replay window is ``(covered, edit_watermark]``, and
every event in it is replayed by the next family job. ``covered`` never
passes the watermark of a job whose outcome is not durable, so no
committed family edit can be consumed by the executor state while missing
from the published composition. Starting the window at the STORE's
published watermark instead would be WRONG: the snapshot's tree log
predates tree events published by the legacy path, which must replay as
vegetation commands (``covered`` may legitimately trail the store).

Crash points all converge: a crash during staging leaves the previous
pending (or none) — the sidecar never names staged state; a crash after
the staging swap but before adoption leaves a pending that the next
reconcile adopts only if the job completed (crash before publish →
dropped → replay; crash after publish → the requeue re-runs and completes
idempotently); a crash between the promotion rename and the sidecar flip
leaves the old sidecar naming the old (still-present) generation — the
next job replays from it and the orphan generation is collected later;
the sidecar flip itself is one ``os.replace``.

Supersession parity (H1/R1): the executor's worker is built with a
store-bound revision provider — while the store's scene version stays at
the value observed when the job started, it reports the executor's
internal revision (identical to the executor's default); any concurrent
store commit (edit or reset) makes it report a revision that matches no
target, so the worker's cooperative supersession checkpoints abort the
batch mid-solve exactly like the legacy solver's store-bound provider,
instead of burning the whole solve to have the runner discard it.

The executor path reuses the transport mapping in
:mod:`solweig_gpu.server.universal` — the SAME ``EditCommand`` assembly
served HTTP validation — and the executor re-validates every command
through its own registry-bound adapters (single source of truth).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from solweig_gpu.incremental.edit_registry import AdapterSchemaError
from solweig_gpu.incremental.edit_types import (
    EditCommand,
    EditStateError,
    SourceDeltaError,
)
from solweig_gpu.incremental.executor import (
    ExecutorError,
    PlanExecutor,
    executed_plan_document,
)
from solweig_gpu.incremental.geometry import InfluenceConfig, RasterGrid, TreeSpec
from solweig_gpu.incremental.planner import PlanningError
from solweig_gpu.incremental.result import load_patch
from solweig_gpu.incremental.scenario_state import (
    ScenarioStateError,
    read_snapshot,
    restore_into_executor,
    snapshot_from_executor,
    write_snapshot,
)
from solweig_gpu.incremental.solver import SolverInputError
from solweig_gpu.incremental.trees import TreeLayer
from solweig_gpu.server import patch_codec, universal
from solweig_gpu.server.jobs import (
    LEGACY_ACTOR_ID,
    RunnerContext,
    SolveRequest,
    SolveResult,
    _compose_current_state,
    api_tree_to_spec,
    default_limitations,
    ensure_exact_lane_bootstrap,
    exact_lane_skip_disclosure,
    make_exact_worker_solver,
    resolve_requested,
    resolve_time_stop,
    stage_timing_metrics,
    union_window_area_fraction,
)
from solweig_gpu.server.realtime import selected_time, types as rt_types
from solweig_gpu.server.realtime.reducer import DeterministicEpochReducer

logger = logging.getLogger(__name__)

STATE_DIRECTORY_NAME = "executor-state"
COVERAGE_FILENAME = "ledger-coverage.json"
SNAPSHOT_FILENAME = "scenario-state.json"

#: Staged (not yet adoptable) snapshot awaiting durable-completion review.
PENDING_DIRECTORY_NAME = "pending"
PENDING_STAGING_NAME = "pending.tmp"
PENDING_META_FILENAME = "pending.json"

#: Adopted snapshots (immutable; the sidecar names one by its generation).
GENERATIONS_DIRECTORY_NAME = "generations"

#: Default requested variables for the executor's worker: the patch
#: transport's own vocabulary
#: (:data:`solweig_gpu.server.patch_codec.SUPPORTED_VARIABLES`), derived
#: rather than restated so a widened transport widens the replay with it.
EXECUTOR_REQUESTED_VARIABLES = tuple(patch_codec.SUPPORTED_VARIABLES)

#: Effective ``full_recompute_fraction`` for EXACT-LANE solves (routing
#: policy wave): the dirty-union fraction at which a solve escapes windowed
#: mode and pays one full-tile pass. The legacy lanes keep the 0.30 default
#: (:attr:`InfluenceConfig.full_recompute_fraction` /
#: :attr:`ConservativeSafetyPolicy.full_recompute_fraction`); the exact
#: lane prefers short incremental solves, so only a near-whole-tile dirty
#: union (>= 0.95) escapes the window overhead it would no longer amortize.
#: Threaded to BOTH coupled decision sites (worker mode choice + planner
#: safety policy) via :func:`_exact_lane_recompute_fraction`.
EXACT_LANE_FULL_RECOMPUTE_FRACTION = 0.95

#: Effective fraction for an idle RECONCILIATION full (once-daily
#: subscriber-free maintenance): the whole point is one authoritative
#: full-tile solve, so the threshold collapses to ~0 (every non-empty
#: dirty union routes full). Same 1e-12 idiom the executor's own
#: ``force_full`` swap uses.
RECONCILE_FULL_RECOMPUTE_FRACTION = 1e-12


def _exact_lane_recompute_fraction(request: "SolveRequest") -> float | None:
    """Routing threshold for this request, or ``None`` for the legacy lanes.

    Exact-lane requests carry their marker in ``request.exact_lane``; a
    marker with ``reconcile`` truthy is the once-daily maintenance full
    (see :data:`RECONCILE_FULL_RECOMPUTE_FRACTION`), any other exact-lane
    marker gets :data:`EXACT_LANE_FULL_RECOMPUTE_FRACTION`, and everything
    else (legacy ``/edits`` jobs) keeps the 0.30 default at both sites.
    """
    lane = request.exact_lane
    if not lane:
        return None
    if lane.get("reconcile"):
        return RECONCILE_FULL_RECOMPUTE_FRACTION
    return EXACT_LANE_FULL_RECOMPUTE_FRACTION


class _Unrecoverable(RuntimeError):
    """The durable state cannot rebuild the scenario exactly."""


def make_universal_dispatch_solver(base_factory=None):
    """Wrap the default solver factory with the family dispatcher."""
    base = base_factory if base_factory is not None else make_exact_worker_solver()

    def factory(context: RunnerContext):
        base_solver = base(context)

        def solve(request: SolveRequest, progress) -> SolveResult:
            # r2a: exact-lane jobs ALWAYS take the executor path — their
            # batch is derived from the realtime operation span, not the
            # tree ledger, and the executor bridge is the only place that
            # can fold the epoch plane onto scene state. This fires before
            # the durable family predicate: a workspace may carry realtime
            # history without any surviving family event.
            if request.exact_lane is not None:
                return _solve_exact_lane(context, request, progress)
            # u-e1 F1: route on the DURABLE family signal, never on the
            # retention-pruned event list alone. Once sweep_retention
            # prunes every family event past the tail, an events-only
            # predicate sends a later TREE edit to the legacy baseline-
            # bound solver, which publishes without the family state. The
            # belt-and-braces event scan stays (it catches family events
            # in THIS request and pre-migration rows whose surviving
            # ledger still names them); the store flag is the memory that
            # outlives the sweep. The predicate errs toward OVER-routing:
            # a tree-only scenario wrongly flagged pays one executor
            # rebuild, a family scenario wrongly legacy-published pays a
            # bitwise-wrong scene.
            if any(event.get("family") for event in request.events) or (
                context.store.scenario_carries_family_edits(request.scenario_id)
            ):
                return _solve_with_executor(context, request, progress)
            return base_solver(request, progress)

        return solve

    return factory


# ---------------------------------------------------------------------------
# Coverage sidecar + pending-adoption protocol
# ---------------------------------------------------------------------------


def _state_directory(context: RunnerContext, scenario_id: str) -> Path:
    return context.store.results_root / scenario_id / STATE_DIRECTORY_NAME


@dataclass(frozen=True)
class _Coverage:
    """The snapshot the sidecar trusts: its ledger watermark + location.

    ``generation`` is ``None`` for the legacy root layout (snapshot files
    directly under the state directory, written before coverage promotion
    existed — such a snapshot stays readable/adoptable); otherwise the
    directory name under ``generations/`` holding the snapshot.
    """

    covered_sequence: int
    generation: str | None


def _read_coverage(state_dir: Path) -> _Coverage:
    path = state_dir / COVERAGE_FILENAME
    if not path.is_file():
        return _Coverage(0, None)
    try:
        document = json.loads(path.read_text())
        covered = int(document["covered_sequence"])
        generation = document.get("generation")
        if generation is not None:
            generation = str(generation)
            if not (state_dir / GENERATIONS_DIRECTORY_NAME / generation).is_dir():
                # The named snapshot is gone (manual interference or a
                # crash mid-collection): distrust the coverage entirely.
                return _Coverage(0, None)
        return _Coverage(covered, generation)
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        # A corrupt sidecar is treated as no coverage: the snapshot is not
        # trusted and the executor rebuilds from the ledger (a typed
        # unrecoverable error surfaces if the ledger is insufficient).
        return _Coverage(0, None)


def _write_coverage(state_dir: Path, coverage: _Coverage) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    tmp = state_dir / (COVERAGE_FILENAME + ".tmp")
    tmp.write_text(
        json.dumps(
            {
                "covered_sequence": int(coverage.covered_sequence),
                "generation": coverage.generation,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    os.replace(tmp, state_dir / COVERAGE_FILENAME)


def _snapshot_directory(state_dir: Path, coverage: _Coverage) -> Path:
    """Directory holding the snapshot the sidecar names (root or a
    generation)."""
    if coverage.generation is None:
        return state_dir
    return state_dir / GENERATIONS_DIRECTORY_NAME / coverage.generation


def _read_pending_meta(state_dir: Path) -> Mapping[str, Any] | None:
    """The staged snapshot's producer (job id) and coverage, or ``None``.

    ``None`` also covers a corrupt meta file: a pending without readable
    metadata cannot prove its producer completed, so it is dropped.
    """
    path = state_dir / PENDING_DIRECTORY_NAME / PENDING_META_FILENAME
    if not path.is_file():
        return None
    try:
        meta = json.loads(path.read_text())
        return {
            "job_id": str(meta["job_id"]),
            "covered_sequence": int(meta["covered_sequence"]),
        }
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return None


def _discard_pending(state_dir: Path) -> None:
    """Remove staged (unadoptable) state — its events stay pending replay."""
    shutil.rmtree(state_dir / PENDING_DIRECTORY_NAME, ignore_errors=True)
    shutil.rmtree(state_dir / PENDING_STAGING_NAME, ignore_errors=True)


def _collect_generations(generations_dir: Path, *, keep: str) -> None:
    """Best-effort GC of superseded snapshot generations.

    A superseded generation is never read again: the sidecar names the
    current one, and a re-run recomputes from it rather than from older
    state. Never called before the sidecar flip naming ``keep``, so a
    crash here cannot orphan the trusted snapshot.
    """
    try:
        entries = list(generations_dir.iterdir())
    except OSError:
        return
    for entry in entries:
        if entry.name == keep:
            continue
        try:
            if entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink()
        except OSError:
            pass


def _promote_pending(
    context: RunnerContext, state_dir: Path, meta: Mapping[str, Any]
) -> _Coverage:
    """Adopt the staged snapshot: rename it under ``generations/``, then
    flip the sidecar (one atomic ``os.replace``) to name it.

    A crash before the flip leaves the sidecar naming the previous —
    still present — generation (GC only runs after a flip), so the next
    job replays from it and the orphan is collected later.
    """
    generations_dir = state_dir / GENERATIONS_DIRECTORY_NAME
    generations_dir.mkdir(parents=True, exist_ok=True)
    generation = f"gen-{meta['job_id']}"
    target = generations_dir / generation
    pending_dir = state_dir / PENDING_DIRECTORY_NAME
    if target.exists():
        # A same-named generation survived an earlier crash-after-rename:
        # replace it wholesale rather than merge directories.
        shutil.rmtree(target, ignore_errors=True)
    os.replace(pending_dir, target)
    coverage = _Coverage(int(meta["covered_sequence"]), generation)
    _write_coverage(state_dir, coverage)
    _collect_generations(generations_dir, keep=generation)
    return coverage


def _pending_restorable_by(
    snapshot: Any, request: SolveRequest
) -> bool:
    """Whether THIS request's executor could restore the staged snapshot.

    ``restore_into_executor`` verifies the snapshot's influence config
    against the executor's (locality policy is part of the scenario's
    computational identity), and the exact lane runs a DIFFERENT
    ``full_recompute_fraction`` than the legacy path (0.95 vs the site
    default; a reconcile marker lower still). A staged snapshot produced
    under one fraction can therefore never be restored by a lane running
    another: adopting it would wedge that lane's next span behind the
    restore's typed ``scenario_state_unrecoverable`` refusal and the
    tick's churn guard (incident-2). The staged snapshot is dropped
    instead — its ledger events stay pending replay, and the lane
    rebuilds fresh (the exact lane re-derives its own layer from the
    legacy tree list plus the fold baseline; window family events replay
    as commands), which loses nothing.
    """
    try:
        staged = float(
            (snapshot.influence_config or {}).get(
                "full_recompute_fraction", InfluenceConfig().full_recompute_fraction
            )
        )
    except (TypeError, ValueError):
        return False
    override = _exact_lane_recompute_fraction(request)
    lane = (
        float(override)
        if override is not None
        else float(InfluenceConfig().full_recompute_fraction)
    )
    return staged == lane


def _promoted_snapshot_restorable(
    snapshot_dir: Path, request: SolveRequest
) -> bool:
    """Whether this request's lane could restore the PROMOTED snapshot
    the coverage sidecar names (#139: the fraction gate's scope covers
    the promoted door too).

    ``_pending_restorable_by`` guards the pending-adoption door alone,
    but the coverage sidecar is per-SCENARIO, shared by every lane: a
    snapshot promoted by the legacy lane (0.30) — or, symmetrically, by
    the exact lane (0.95), blocking a later family job — is named by the
    sidecar this request reads, and restoring it trips
    ``_verify_identity``'s influence-config fence, whose typed
    ``scenario_state_unrecoverable`` refusal repeats idempotently until
    a reset: the incident-2 wedge class, one door later. A promoted
    snapshot whose locality fraction this request's lane cannot restore
    is therefore treated as ABSENT for branch selection — the job takes
    the same fresh rebuild the pending door's documented drop produces
    ("never adopted into a coverage its executor must then refuse"),
    never restoring.

    A snapshot that cannot be READ (torn pair, corrupt document) keeps
    the file-existence decision: the restore attempt itself surfaces
    the typed refusal (u-e1 F2), exactly as before this guard.
    """
    try:
        snapshot = read_snapshot(snapshot_dir)
    except ScenarioStateError:
        return True
    return _pending_restorable_by(snapshot, request)


def _reconcile_coverage(
    context: RunnerContext, state_dir: Path, request: SolveRequest
) -> _Coverage:
    """Adopt or drop the staged pending snapshot — never mid-job.

    Runs at the START of every family solve, before the executor is
    built, so the snapshot decision is made exactly once per job and
    always against durable store state. Adoption requires the producing
    job to be a DIFFERENT job whose durable row is ``complete``: that is
    the one status whose outcome (a published result, or a no-op
    re-serve) the store has already committed, and (incident-2) a
    snapshot whose locality fraction this request's own lane could
    actually restore — a cross-lane staged snapshot is dropped, never
    adopted into a coverage its executor must then refuse. Everything
    else — superseded, cancelled, failed, crashed-mid-flight (still
    ``running``), or this same job re-queued after a crash — drops the
    staged snapshot, leaving its ledger events > ``covered`` and thus
    pending replay by THIS job.
    """
    meta = _read_pending_meta(state_dir)
    if meta is not None:
        adopt = False
        if meta["job_id"] != request.job_id:
            row = context.store.get_job(meta["job_id"])
            if (
                row is not None
                and row.status == "complete"
                and meta["covered_sequence"] <= int(request.edit_watermark)
            ):
                try:
                    # Torn pairs refuse HERE, before anything is promoted.
                    snapshot = read_snapshot(state_dir / PENDING_DIRECTORY_NAME)
                except ScenarioStateError:
                    pass
                else:
                    adopt = _pending_restorable_by(snapshot, request)
        if adopt:
            return _promote_pending(context, state_dir, meta)
    _discard_pending(state_dir)
    return _read_coverage(state_dir)


# ---------------------------------------------------------------------------
# Rebuild + batch assembly
# ---------------------------------------------------------------------------


class _StoreSupersessionWatch:
    """Store-bound supersession checkpoint for the executor's worker (H1/R1).

    The executor's default revision provider reports its INTERNAL scene
    revision, so a store commit landing while a family job runs never
    trips the worker's cooperative supersession checkpoints — unlike the
    legacy solver, whose provider is bound to
    ``store.current_scene_version`` (see ``jobs.py``). This watch restores
    that parity: while the store's scene version stays at the value
    observed when the job started building, it reports the executor's
    internal revision (byte-identical behavior to the default); the
    moment the store moves (any commit — edit or reset), it reports a
    revision that matches no target, so every
    ``ExactWorker._check_not_superseded`` checkpoint aborts the batch and
    the executor's wholesale rollback runs. The discarded result never
    publishes, its snapshot never stages, and the committed edit stays
    pending replay — the same mid-flight outcome the legacy solver has.
    """

    def __init__(self, store, scenario_id: str) -> None:
        self._store = store
        self._scenario_id = scenario_id
        self._baseline = int(store.current_scene_version(scenario_id))
        self._executor: PlanExecutor | None = None

    def bind(self, executor: PlanExecutor) -> None:
        self._executor = executor

    def tripped(self) -> bool:
        """True once the store's scene version moved off the job's base."""
        return int(self._store.current_scene_version(self._scenario_id)) != self._baseline

    def __call__(self) -> int:
        executor = self._executor
        if executor is None:  # defensive: never called unbound
            return self._baseline
        if self.tripped():
            return -1
        return executor.scene_revision


def _tree_state(tree: Mapping[str, Any], grid_info: Mapping[str, float | int]) -> dict:
    spec = api_tree_to_spec(tree, grid_info)
    return {
        "tree_id": spec.tree_id,
        "x_m": spec.x_m,
        "y_m": spec.y_m,
        "height_m": spec.height_m,
        "canopy_radius_m": spec.canopy_radius_m,
        "trunk_ratio": spec.trunk_ratio,
    }


def _exact_lane_baseline_tree_specs(
    baseline: rt_types.CanonicalState,
) -> tuple[TreeSpec, ...]:
    """Live vegetation objects of the exact lane's fold baseline, as specs.

    The canonical state at ``consumed_revision`` is the realtime plane's
    authoritative tree list (pre-gate legacy trees ride the pinned
    revision-0 bootstrap; every post-gate tree change is an operation the
    scheduler folded). The executor's layer for an exact-lane job must
    carry exactly this list: the span's family delta is the net change
    VERSUS this baseline, so replaying it against any other layer breaks
    the batch's before/after contiguity contract.

    ``trunk_ratio`` defaults to the layer's own default when an object's
    history never carried one (adds predating the field, folded pins); a
    structurally malformed object (the reducer folds accepted payloads,
    but a torn row would surface here) refuses typed rather than seeding
    a wrong tree silently.
    """
    family = (baseline.families or {}).get("vegetation_geometry") or {}
    objects: Mapping[str, Mapping[str, Any]] = family.get("objects") or {}
    specs: list[TreeSpec] = []
    for entity_id in sorted(objects):
        fields = objects[entity_id]
        try:
            specs.append(
                TreeSpec(
                    tree_id=str(entity_id),
                    x_m=float(fields["x_m"]),
                    y_m=float(fields["y_m"]),
                    height_m=float(fields["height_m"]),
                    canopy_radius_m=float(fields["canopy_radius_m"]),
                    trunk_ratio=float(fields.get("trunk_ratio", 0.25)),
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            raise EditStateError(
                f"exact lane: fold baseline object {entity_id!r} carries an "
                f"unusable vegetation spec ({error!r}); refusing to seed the "
                "executor layer from a malformed canonical state"
            ) from error
    return tuple(specs)


def _seed_exact_lane_baseline(
    executor: PlanExecutor,
    baseline: rt_types.CanonicalState,
    *,
    exclude: frozenset[str] = frozenset(),
) -> None:
    """Reconcile the layer with the fold baseline's vegetation, add-only.

    Whatever tree base an exact-lane executor starts from — the legacy
    tree list (fresh rebuild), a restored snapshot, or the post-reset
    void — the span's family delta is the net change VERSUS the canonical
    baseline at ``consumed_revision``, so the layer must also carry that
    baseline's live vegetation objects or the batch's before/after
    contiguity contract breaks (``_replay_tree_edits`` refuses the first
    structural op on an id the layer has never seen). Add-only, exactly
    like the legacy-list seeding: an id the base already carries is left
    at the base's own state, never re-seeded.

    Incident-2 (live 2026-09-07): 1e07f1c seeded the baseline on the
    fresh branch alone. A foreign staged snapshot (a family job's
    coverage) or a reset-voided base left the layer native-blind on the
    OTHER two branches, and the span refused with the production
    ``replay chain ... not contiguous`` message, wedging the lane behind
    the churn guard.

    Incident-3 (live 2026-09-07): the incident-2 reasoning "a window
    event's epoch is unsettled at mint time, so the baseline can never
    contain its id" does NOT survive the B1b typed settle — a span
    refused as unfoldable still SETTLES its epoch, so the next mint's
    baseline (settled revisions only) can very well contain an id whose
    ledger event the same job's window still replays (coverage never
    advanced past the refused job). Seeding that id would collide with
    the window's own ADD replay. ``exclude`` carries exactly those ids
    (``_ledger_window_anchors``): ids whose FIRST window event
    establishes them (an add). Ids whose window opens with a
    move/update/delete still need their pre-state in the layer for the
    replay to mutate, and are seeded normally.
    """
    live = {tree.tree_id for tree in executor.layer.current_trees()}
    for spec in _exact_lane_baseline_tree_specs(baseline):
        if spec.tree_id in live or spec.tree_id in exclude:
            continue
        executor.layer.add_tree(spec)


@dataclass(frozen=True)
class _LedgerAnchor:
    """One tree id this job's ledger window replays, anchored for the fold.

    Incident-3 (live 2026-09-07, ``tree-mtrl1chf-01-dmnp``): the exact
    span's executor batch is assembled from TWO planes — the ledger
    window replay (``_pending_commands``) and the epoch-span fold
    (``reduce_epoch``). When the same object id appears in both, the
    batch carries two ObjectStateChange records for it and the executor's
    ``_coalesce_object_changes`` demands chain contiguity: the fold's
    before-state must BE the ledger replay's after-state. At base the
    fold chained onto the consumed canonical state (the native op-1
    position) while the ledger replay produced the ADD position — the
    production ``non-contiguous change chain`` refusal, escaping untyped
    (``SourceDeltaError`` is ``EditStateError``'s sibling, not its
    subclass) and wedging the lane behind the churn guard.

    The anchor re-derives the fold for exactly these ids: the baseline's
    copy of the object is replaced by the LEDGER's after-state (the last
    window event's ``new_tree``), and the object's native operations
    SINCE that event (the funnelled op's sequence) re-fold onto it — so
    the fold's delta is the coalesced ``[ledger_after -> final]`` net
    change, contiguous with the window replay by construction (both
    states derive from the same ``_tree_state``/``uv_to_world`` floats).

    Review BLOCKER 2 extended the re-derivation BELOW the anchor for
    move anchors: the event's trees are a LEGACY-plane capture (native
    writes never touch ``scenario_trees``) while the funnelled move
    writes position only in canonical, so committed native FIELD writes
    predating the funnel survive canonical and must be recovered —
    see :func:`_anchored_backfill`'s recovery range and its non-
    convergent-delete refusal.
    """

    tree_id: str
    #: Five vegetation properties of the last window event's ``new_tree``
    #: (reducer vocabulary, no tree id), or None when the window's last
    #: event for the id deletes it (a tombstone anchor).
    after: dict[str, float] | None
    #: ``server_sequence`` of the event's funnelled realtime op (0 when
    #: the event never funnelled): the fold's high-water mark for this
    #: id's already-committed native operations.
    funnel_seq: int
    #: Whether the window's FIRST event for the id establishes it (an
    #: add with no old tree): the replay itself puts the id in the layer,
    #: so the baseline seed must not pre-plant it (seed coherence).
    established_by_window: bool
    #: Whether the LAST window event canonically writes POSITION ONLY (a
    #: legacy move: the funnelled op is reducer-filtered to
    #: ``_POSITION_FIELDS``, while the event's own trees are a full
    #: LEGACY-plane capture). Only then can pre-anchor native FIELD
    #: writes survive canonical yet vanish from the anchored fold — the
    #: recovery range exists exactly for this anchor kind (incident-3
    #: review BLOCKER 2).
    partial_write_set: bool
    #: ``server_sequence`` of the funnelled op of the LAST window event
    #: whose canonical write-set is FULL (an add, or an update/replace):
    #: every native op at or below it is superseded in canonical by that
    #: event, so recovery starts above it. 0 when the window never saw a
    #: full-write event for the id (it pre-existed the window in the
    #: legacy plane) — recovery then reaches the whole committed native
    #: history of the id.
    recovery_floor: int


def _ledger_window_anchors(
    request: SolveRequest,
    window: "_ReplayWindow",
    records: Sequence[Any],
) -> dict[str, _LedgerAnchor]:
    """Anchors for every tree id THIS job's ledger window replays.

    The filter mirrors :func:`_pending_commands`' tree-event replay
    predicate exactly (``covered < sequence <= edit_watermark``, no
    family, not a reset) — an id anchors if and only if the same batch
    will replay one of its events. Per id: the LAST window event wins
    (the ledger's authoritative state for it).
    """
    if not window.replay_tree_events:
        return {}
    watermark = int(request.edit_watermark)
    events_by_id: dict[str, list[Mapping[str, Any]]] = {}
    established: dict[str, bool] = {}
    for event in request.events:
        sequence = int(event["sequence"])
        if sequence <= window.covered or sequence > watermark:
            continue
        if event.get("family") or event.get("operation") == "reset":
            continue
        tree_id = event.get("tree_id")
        if not tree_id:
            continue
        tree_id = str(tree_id)
        if tree_id not in established:
            established[tree_id] = bool(
                event.get("new_tree") and not event.get("old_tree")
            )
        events_by_id.setdefault(tree_id, []).append(event)  # ascending
    if not events_by_id:
        return {}

    # Funnel sequences for the events the anchor fields need: the LAST
    # event per id (the anchor point) and the last FULL-write event per
    # id (the recovery floor).
    wanted: dict[str, str] = {}
    for tree_id, id_events in events_by_id.items():
        wanted[f"legacy-{id_events[-1]['event_id']}"] = tree_id
        for event in id_events:
            if str(event["operation"]) in ("add", "update"):
                wanted[f"legacy-{event['event_id']}"] = tree_id
    funnel_seq: dict[str, int] = {}
    for record in records:
        if str(record.operation_id) in wanted:
            funnel_seq[str(record.operation_id)] = int(record.server_sequence)

    def _event_funnel(event: Mapping[str, Any]) -> int:
        return funnel_seq.get(f"legacy-{event['event_id']}", 0)

    anchors: dict[str, _LedgerAnchor] = {}
    for tree_id, id_events in events_by_id.items():
        last = id_events[-1]
        recovery_floor = 0
        for event in id_events:
            if str(event["operation"]) in ("add", "update"):
                recovery_floor = _event_funnel(event)
        new_tree = last.get("new_tree")
        after = (
            {
                name: value
                for name, value in _tree_state(new_tree, request.grid).items()
                if name != "tree_id"
            }
            if new_tree
            else None
        )
        anchors[tree_id] = _LedgerAnchor(
            tree_id=tree_id,
            after=after,
            funnel_seq=_event_funnel(last),
            established_by_window=bool(established.get(tree_id)),
            partial_write_set=str(last["operation"]) == "move" and after is not None,
            recovery_floor=recovery_floor,
        )
    return anchors


def _anchored_baseline(
    baseline: rt_types.CanonicalState,
    anchors: dict[str, _LedgerAnchor],
) -> rt_types.CanonicalState:
    """The fold baseline with every anchored id reset to its ledger state.

    Generations are preserved (the reducer's conflict audit reads them);
    every other field of an anchored id becomes the ledger after-state,
    erasing the canonical row's own version of that id's history so the
    id's native operations since its last ledger event can re-fold onto
    it exactly once. Non-anchored families and objects pass through
    untouched.
    """
    if not anchors:
        return baseline
    families = dict(baseline.families or {})
    vegetation = dict(families.get("vegetation_geometry") or {})
    objects = {
        str(entity): dict(fields)
        for entity, fields in (vegetation.get("objects") or {}).items()
    }
    tombstones = dict(vegetation.get("tombstones") or {})
    for anchor in anchors.values():
        if anchor.after is None:
            # The window's last event deletes the tree: anchor a
            # tombstone so any later native re-add folds as a recreation.
            prior = objects.pop(anchor.tree_id, None)
            generation = int(
                (prior or {}).get(
                    "generation", tombstones.get(anchor.tree_id, 0)
                )
            )
            tombstones[anchor.tree_id] = generation
        else:
            prior = objects.get(anchor.tree_id)
            if prior is not None:
                generation = int(prior.get("generation", 0))
            else:
                buried = tombstones.pop(anchor.tree_id, None)
                # Resurrection parity (review MINOR): the funnelled add
                # over a tombstone raises canonical's generation by one
                # (GENERATION_RECREATED); anchor the SAME generation the
                # canonical row at this revision carries. Deltas exclude
                # generation either way — this only keeps the reducer's
                # conflict audit reading the same number canonical wrote.
                generation = (int(buried) + 1) if buried is not None else 0
            objects[anchor.tree_id] = {**anchor.after, "generation": generation}
    families["vegetation_geometry"] = {
        **vegetation,
        "objects": objects,
        "tombstones": tombstones,
    }
    return rt_types.CanonicalState(
        workspace_id=baseline.workspace_id,
        workspace_revision=baseline.workspace_revision,
        families=families,
    )


#: The funnelled move's actual canonical write-set for tree payloads
#: (``x_m``, ``y_m``): the fields a move-anchor's recovery must NOT
#: re-apply (the event itself is their latest causal writer). Deliberately
#: NOT ``reducer._POSITION_FIELDS`` verbatim — that set is a superset (it
#: also carries ``u``/``v``); this constant names only what a funnelled
#: MOVE canonically writes. If the vocabulary ever carries ``u``/``v``,
#: this constant and the per-field convergence argument must be
#: re-audited.
_ANCHOR_POSITION_FIELDS = frozenset({"x_m", "y_m"})


def _anchored_backfill(
    anchors: dict[str, _LedgerAnchor],
    records: Sequence[Any],
    span_sequences: Sequence[int],
    epoch_revisions: Mapping[int, int],
    consumed_revision: int,
) -> list[rt_types.Operation]:
    """Anchored ids' native operations since their last ledger event.

    Zero-loss discipline, same as the chain re-fold: an operation
    backfills only when its epoch carries a committed revision in
    ``(0, consumed_revision]`` (the exact set the scheduler folded —
    never an unsettled epoch) AND it is not already in the span. Span
    operations arrive separately; the reducer's total order
    (``server_sequence``) merges the two lists deterministically.

    PRE-ANCHOR RECOVERY (incident-3 review BLOCKER 2): a legacy MOVE
    event's ``new_tree`` is a LEGACY-plane capture — native realtime
    writes never touch ``scenario_trees`` — while the funnelled move
    applies POSITION ONLY in canonical (reducer ``_POSITION_FIELDS``).
    A committed native FIELD write below the anchor's funnel seq
    therefore survives canonical but would silently vanish from the
    anchored fold: the fold would revert the field to the client's
    stale capture. For move anchors (``partial_write_set``), the
    committed native history between ``recovery_floor`` and the anchor
    re-folds onto the anchor state with each op's write-set RESTRICTED
    to the fields the event does not own (everything but position):

    * a native move (position-only) contributes nothing — the legacy
      move IS the later causal position writer, so the anchor's
      position already carries the merged truth;
    * a native field write/replace/add carries its non-position fields
      (an add is re-expressed as a replace: the reducer's add REPLACES
      the whole row, which would drop the anchor's position; a replace
      merges onto the live row, writing exactly the carried fields).

    Absolute per-field writes re-applied in ``server_sequence`` order
    converge to the per-field last writer — the same merge canonical
    performed — including a native op landing between the client's
    capture and the funnel accept (its fields re-apply over the stale
    capture, which is exactly what canonical ordered).

    THE NON-CONVERGENT CASE (refuse, never diverge): a terminal native
    DELETE in the recovery range — canonical keeps the id dead (the
    funnelled move over a tombstone is an audited no-op) while the
    event's trees never saw the death; folding would silently
    resurrect the id. Raises :class:`SourceDeltaError` for the caller's
    typed ``edit_rejected`` envelope (which settles).
    """
    if not anchors:
        return []
    span = frozenset(int(sequence) for sequence in span_sequences)
    backfill: list[rt_types.Operation] = []
    recovery: dict[str, list[Any]] = {}
    for record in records:
        if record.actor_id == LEGACY_ACTOR_ID:
            continue
        anchor = anchors.get(str(record.entity_id))
        if anchor is None or record.source_family != "vegetation_geometry":
            continue
        sequence = int(record.server_sequence)
        if sequence in span:
            continue
        revision = epoch_revisions.get(int(record.epoch_id))
        if revision is None or not (0 < int(revision) <= consumed_revision):
            continue
        if sequence > anchor.funnel_seq:
            backfill.append(_operation_from_record(record))
        elif (
            anchor.partial_write_set
            and sequence > anchor.recovery_floor
        ):
            recovery.setdefault(anchor.tree_id, []).append(record)

    for tree_id, records_for_id in recovery.items():
        alive = True  # the legacy plane sees the id (the route moved it)
        for record in sorted(records_for_id, key=lambda item: item.server_sequence):
            operation = _operation_from_record(record)
            if operation.verb == "delete":
                alive = False
                continue
            if operation.verb == "add":
                alive = True
            values = dict((operation.payload or {}).get("values") or {})
            carried = {
                name: value
                for name, value in values.items()
                if name not in _ANCHOR_POSITION_FIELDS
            }
            if not carried:
                continue  # a position-only write: the anchor owns position
            if operation.verb == "add":
                # A reducer add REPLACES the row wholesale, which would
                # drop the anchor's position; re-express it as a replace
                # so only the carried (non-position) fields land.
                operation = replace(operation, verb="replace")
            backfill.append(replace(operation, payload={"values": carried}))
        if not alive:
            anchor = anchors[tree_id]
            raise SourceDeltaError(
                f"mixed-plane anchor cannot converge for object "
                f"{tree_id!r}: a native delete precedes its ledger "
                f"window event (sequence at or below the anchor's "
                f"funnel op {anchor.funnel_seq}), and the event's "
                f"legacy-plane capture never saw the death — canonical "
                f"keeps the object dead while the ledger would "
                f"resurrect it. Reset the scenario or re-add the tree "
                f"to converge"
            )
    backfill.sort(key=lambda operation: operation.server_sequence)
    return backfill


def _vegetation_command(
    event: Mapping[str, Any],
    grid_info: Mapping[str, float | int],
    *,
    scenario_id: str,
    scene_revision: int,
    edit_id: str,
    requested_outputs: Sequence[str],
    requested_times: tuple[int, ...],
) -> EditCommand:
    operation = str(event["operation"])
    old_state = (
        _tree_state(event["old_tree"], grid_info) if event.get("old_tree") else None
    )
    new_state = (
        _tree_state(event["new_tree"], grid_info) if event.get("new_tree") else None
    )
    return EditCommand(
        edit_id=edit_id,
        scenario_id=scenario_id,
        base_scene_revision=int(scene_revision),
        adapter_id="vegetation_geometry",
        operation=operation,
        old_state=old_state,
        new_state=new_state,
        requested_outputs=tuple(requested_outputs),
        requested_times=requested_times,
    )


@dataclass(frozen=True)
class _ReplayWindow:
    """The ledger replay window decision of :func:`_build_executor`.

    Pure function of durable state (the coverage sidecar, the snapshot
    directory, the event list) plus the request's own lane (the #139
    promoted-door fraction gate) so the exact lane can resolve it
    EAGERLY — before the executor is built — and derive its mixed-plane
    anchors from the same window ``_pending_commands`` will replay
    (incident-3).
    """

    #: The window's floor: tree events at ``sequence <= covered`` are
    #: already reflected in the executor's tree base.
    covered: int
    #: Whether the window's tree events replay as vegetation commands.
    replay_tree_events: bool
    #: Whether the decision restored a promoted snapshot (the first
    #: branch); False on the post-reset void and fresh-rebuild branches.
    snapshot_restored: bool


def _ledger_replay_window(
    state_dir: Path,
    coverage: _Coverage,
    request: SolveRequest,
    ledger: Sequence[Mapping[str, Any]],
) -> _ReplayWindow:
    """Which of ``_build_executor``'s three branches this job takes.

    Branch semantics (incident-3 refactor of the inline conditions; the
    mapping is exact):

    * a promotable snapshot whose coverage postdates the last reset AND
      whose locality fraction this request's own lane could restore
      (#139: the fraction gate's scope covers the promoted door — see
      :func:`_promoted_snapshot_restorable`) -> restore it, replay the
      window above its coverage;
    * a reset the coverage never passed -> the post-reset VOID: replay
      every tree event after the reset (the u-e3c NF1 rebuild);
    * anything else -> fresh rebuild, the authoritative tree list
      already reflects every committed tree edit, no tree replay.
    """
    last_reset = max(
        (int(event["sequence"]) for event in ledger if event.get("operation") == "reset"),
        default=0,
    )
    covered = coverage.covered_sequence
    snapshot_dir = _snapshot_directory(state_dir, coverage)
    if (
        (snapshot_dir / SNAPSHOT_FILENAME).is_file()
        and covered > last_reset
        and _promoted_snapshot_restorable(snapshot_dir, request)
    ):
        return _ReplayWindow(covered, True, True)
    if last_reset > 0 and covered <= last_reset:
        return _ReplayWindow(last_reset, True, False)
    return _ReplayWindow(max(covered, last_reset), False, False)


def _build_executor(
    context: RunnerContext,
    request: SolveRequest,
    *,
    grid: RasterGrid,
    coverage: _Coverage,
    exact_lane_baseline: rt_types.CanonicalState | None = None,
    exact_lane_seed_exclusions: frozenset[str] = frozenset(),
) -> tuple[PlanExecutor, _StoreSupersessionWatch, int, bool]:
    """Fresh-or-restored executor, its consumed sequence, replay flag.

    ``coverage`` comes from :func:`_reconcile_coverage` (the pending
    adoption decision is made exactly once per job, before this). The
    flag is the ``replay_tree_events`` decision: True when the tree base
    the executor starts from predates the replay window (a restored
    snapshot's tree log, or the post-reset void), so the window's tree
    events must replay as vegetation commands.

    ``exact_lane_baseline`` (exact-lane jobs only) is the canonical fold
    baseline at ``consumed_revision``; its live vegetation objects seed
    the layer on EVERY branch (fresh, restored, void) alongside whatever
    tree base that branch carries (see :func:`_seed_exact_lane_baseline`).
    ``exact_lane_seed_exclusions`` (incident-3) withholds ids whose
    ledger-window events this same job replays as ADDs — the replay
    itself establishes them in the layer, so seeding the baseline's copy
    would collide with it.
    """
    site = context.sites.config(request.site_id)
    cache = context.sites.cache(request.site_id)
    scenario_root = context.store.results_root / request.scenario_id
    state_dir = _state_directory(context, request.scenario_id)

    window = _ledger_replay_window(state_dir, coverage, request, list(request.events))
    snapshot_dir = _snapshot_directory(state_dir, coverage)

    layer = TreeLayer(
        np.asarray(cache.tree_base), grid, scenario_id=request.scenario_id
    )
    watch = _StoreSupersessionWatch(context.store, request.scenario_id)
    executor = PlanExecutor(
        cache=cache,
        layer=layer,
        site_dir=site.site_dir or site.cache_dir,
        results_root=scenario_root / "executor",
        selected_date_str=site.selected_date_str,
        revision_provider=watch,
        full_recompute_fraction=_exact_lane_recompute_fraction(request),
    )
    watch.bind(executor)

    if window.snapshot_restored:
        restore_into_executor(executor, read_snapshot(snapshot_dir), directory=snapshot_dir)
        if exact_lane_baseline is not None:
            # Incident-2: the restored snapshot's layer may be native-blind
            # (a producer that never seeded the fold baseline — see
            # _reconcile_coverage's lane guard for the fraction-mismatch
            # case this cannot be). Seed the baseline's objects so the
            # span's first structural op on a native tree replays
            # contiguously instead of refusing.
            _seed_exact_lane_baseline(
                executor, exact_lane_baseline, exclude=exact_lane_seed_exclusions
            )
        return executor, watch, window.covered, window.replay_tree_events

    if window.replay_tree_events:
        # u-e3c NF1: a reset VOIDED the executor's tree base — reset wipes
        # ``scenario_trees``, so the only trees the scene can carry are the
        # ones the post-reset ledger window commits. Seed NOTHING and let
        # the window's tree events replay as vegetation commands: replaying
        # the same events, in sequence order, from the voided base
        # reconstructs the authoritative tree list exactly, AND a tree-only
        # window (family edit -> reset -> tree edit) now produces a real
        # batch and publishes the tree patch instead of refusing an empty
        # one. Without the replay the batch validated to zero commands and
        # the executor refused with ``edit_rejected`` on every retry until
        # a family edit healed the scenario.
        if exact_lane_baseline is not None:
            # Incident-2 (void twin): the void seeds no LEGACY trees, but
            # the fold baseline at ``consumed_revision`` still carries the
            # realtime-native trees (reset voids ``scenario_trees``, never
            # the realtime fold) — seed them or the span's structural op
            # on one refuses with the production message.
            _seed_exact_lane_baseline(
                executor, exact_lane_baseline, exclude=exact_lane_seed_exclusions
            )
        return executor, watch, window.covered, window.replay_tree_events

    # Fresh rebuild whose tree base is NOT voided (first family job, or a
    # snapshot lost while its coverage survives): the authoritative tree
    # list already reflects every committed tree edit, so tree events are
    # seeded once from it and pending tree events are not replayed again.
    for tree in request.trees:
        executor.layer.add_tree(api_tree_to_spec(tree, request.grid))
    if exact_lane_baseline is not None:
        # Exact-lane seeding (live-incident fix, 1e07f1c): the span's
        # family delta is the net change VERSUS the canonical baseline at
        # ``consumed_revision``, so the layer must carry that baseline's
        # vegetation objects or the batch's before/after contiguity
        # contract breaks. On a realtime-native workspace the legacy
        # ledger watermark never advances (realtime operations do not
        # touch ``edit_sequence``), so ``covered`` stays 0, the snapshot
        # restore gate above never fires, and WITHOUT this seed every
        # span folds against a layer missing every tree added by an
        # earlier span — add-only spans publish blind and the first
        # replace/move/delete of an earlier span's tree is refused with
        # ``engine_refused`` (replay chain not contiguous), wedging the
        # exact lane behind the churn guard. The canonical baseline is a
        # pure re-fold of durable state (the pinned revision-0 bootstrap
        # carries pre-gate legacy trees, so those seed exactly once — the
        # ``add_tree`` duplicate check skips an id the legacy list already
        # seeded), and the delta commands then replay contiguously.
        _seed_exact_lane_baseline(
            executor, exact_lane_baseline, exclude=exact_lane_seed_exclusions
        )
    return executor, watch, window.covered, window.replay_tree_events


def _pending_commands(
    executor: PlanExecutor,
    request: SolveRequest,
    grid_info: Mapping[str, float | int],
    covered: int,
    *,
    replay_tree_events: bool,
    variables: Sequence[str],
    all_times: tuple[int, ...],
) -> list:
    """Validated edits for the ledger window ``(covered, edit_watermark]``.

    Two requested-variable rules, on purpose:

    * FAMILY commands request the fixed transport triple
      (:data:`EXECUTOR_REQUESTED_VARIABLES` = the patch codec's
      vocabulary) and the job's time series — the full series by default,
      or the causal prefix ``t = 0..max`` when the request names explicit
      ``time_indices`` (perf wave 1: the prefix is bit-identical for every
      computed step, and each published patch self-describes its time
      coverage so the windowed composition stays aligned);
    * VEGETATION replays (tree events the loaded tree base predates: a
      restored snapshot's tree log, or the post-reset void — u-e3c NF1)
      request the job's RESOLVED variable subset on the same time series:
      the executor merges them into one batch with the family commands,
      and the union of requested variables drives the batch.

    The served result honors the client's ``requested_result`` subset
    when the composed state is sliced. ``all_times`` is the resolved time
    series (computed once by the caller — the exact lane's derived
    commands share it).
    """
    commands: list[EditCommand] = []
    revision = executor.scene_revision
    for position, event in enumerate(request.events):
        sequence = int(event["sequence"])
        if sequence <= covered or sequence > int(request.edit_watermark):
            continue
        if event.get("family"):
            item = universal.item_from_event(event)
            command = universal.command_from_item(
                item,
                scenario_id=request.scenario_id,
                scene_revision=revision,
                edit_id=f"seq{sequence}",
                index=position,
                requested_outputs=EXECUTOR_REQUESTED_VARIABLES,
                requested_times=all_times,
            )
            if command is not None:
                commands.append(executor.validate(command))
        elif replay_tree_events and event.get("operation") != "reset":
            commands.append(
                executor.validate(
                    _vegetation_command(
                        event,
                        grid_info,
                        scenario_id=request.scenario_id,
                        scene_revision=revision,
                        edit_id=f"seq{sequence}",
                        requested_outputs=variables,
                        requested_times=all_times,
                    )
                )
            )
    return commands


def _guard_ledger_completeness(request: SolveRequest, covered: int) -> None:
    """Refuse typed when retention pruned inside the replay window.

    The ledger handed to the solver lists every surviving event; the
    FULL window ``(covered, edit_watermark]`` must be present and
    contiguous — a gap anywhere (checked per sequence, not just at the
    head) means an event the snapshot never consumed is gone with no way
    to replay it.
    """
    sequences = sorted(
        int(event["sequence"])
        for event in request.events
        if int(event["sequence"]) > covered
    )
    missing = [
        sequence
        for sequence in range(covered + 1, int(request.edit_watermark) + 1)
        if sequence not in set(sequences)
    ]
    if missing:
        raise _Unrecoverable(
            f"ledger no longer carries the full replay window (retention "
            f"pruned {len(missing)} event(s) inside it, first missing "
            f"sequence {missing[0]}) and no executor snapshot covers "
            f"them; the scenario's family state cannot be rebuilt exactly"
        )


# ---------------------------------------------------------------------------
# Exact lane (r2a): fold the operation span, derive the executor batch
# ---------------------------------------------------------------------------

#: Reducer delta verb -> adapter operation for the object families. The
#: reducer's ``replace`` is the adapter's ``update`` (full-field write);
#: everything else shares the name.
_OBJECT_DELTA_OPERATIONS = {
    "add": "add",
    "delete": "delete",
    "move": "move",
    "replace": "update",
}

#: Object-family id field inside adapter command state (the delta carries
#: the id as ``target``; the adapters want it inside the state payload).
_OBJECT_ID_FIELDS = {
    "vegetation_geometry": "tree_id",
    "building_geometry": "building_id",
}

#: Segment families and output_view are folded into canonical state but
#: NOT yet mapped onto adapter commands this wave: the segment grammar
#: (field -> [[start, stop, value], ...]) has no one-to-one adapter
#: operation, and view selection never executes. Deltas here are skipped
#: WITH DISCLOSURE (metrics + log), never silently dropped.
#: ``selected_date_time`` is segment-shaped state without an integrated
#: adapter at all; same disclosure.
_EXACT_LANE_UNMAPPED_FAMILIES = (
    "meteorological_forcing",
    "model_receptor_parameters",
    "selected_date_time",
    "output_view",
)


def _operation_from_record(record) -> rt_types.Operation:
    """Durable row -> reducer :class:`Operation` (verbatim fold input)."""
    return rt_types.Operation(
        workspace_id=record.workspace_id,
        operation_id=record.operation_id,
        actor_id=record.actor_id,
        client_sequence=record.client_sequence,
        base_revision=record.base_revision,
        source_family=record.source_family,
        entity_id=record.entity_id,
        verb=record.verb,
        payload=dict(record.payload),
        received_at=record.received_at,
        accepted_at=record.accepted_at,
        server_sequence=int(record.server_sequence),
        epoch_id=int(record.epoch_id),
    )


#: Verbs whose object-family semantics REQUIRE a known live target: the
#: reducer audits delete/replace/move of an unknown or tombstoned id as a
#: no-state-change (its vocabulary has no refusal), so the BRIDGE must
#: fence them before the fold — an accepted operation whose semantics
#: imply a state change may never fold to silence.
_OBJECT_UNFOLDABLE_VERBS = ("delete", "replace", "move")


def _unfoldable_native_object_ops(
    baseline: rt_types.CanonicalState, operations: Sequence[rt_types.Operation]
) -> list[rt_types.Operation]:
    """Native object ops the fold baseline can never resolve (r2a-fix
    BLOCKING-1b).

    Simulates the reducer's known-id tracking over the span in
    ``server_sequence`` order, seeded from the fold baseline's objects and
    tombstones: ``add`` registers the id; delete/replace/move of an id in
    NEITHER set cannot produce the state change its semantics promise
    (the reducer would audit it as a no-op — the accepted-but-unfolded
    case that published wrong science silently). Returns those
    operations, in order; empty when the whole span folds cleanly.

    Deletes of TOMBSTONED ids are foldable by design (deleting a deleted
    object legitimately no-ops) — only ids the fold has never seen trip
    the fence.
    """
    known: dict[str, set[str]] = {}
    for family in _OBJECT_ID_FIELDS:
        state = (baseline.families or {}).get(family) or {}
        known[family] = set(state.get("objects") or {}) | set(
            state.get("tombstones") or {}
        )
    unfoldable: list[rt_types.Operation] = []
    for op in sorted(operations, key=lambda item: item.server_sequence):
        family = op.source_family
        if family not in _OBJECT_ID_FIELDS:
            continue  # landcover/met/segment families have no object ids
        entity = op.entity_id
        if not entity:
            continue
        if op.verb == "add":
            known[family].add(entity)
            continue
        if op.verb in _OBJECT_UNFOLDABLE_VERBS and entity not in known[family]:
            unfoldable.append(op)
    return unfoldable


def _object_delta_command(
    family: str,
    command: Mapping[str, Any],
    *,
    scenario_id: str,
    scene_revision: int,
    edit_id: str,
    requested_outputs: Sequence[str],
    requested_times: tuple[int, ...],
) -> EditCommand:
    operation = _OBJECT_DELTA_OPERATIONS.get(str(command["operation"]))
    if operation is None:
        # The reducer's object grammar is closed; an unknown verb here is
        # a reducer/bridge version skew, not user input.
        raise RuntimeError(
            f"exact lane: unmappable object delta verb {command['operation']!r} "
            f"for {family} (reducer/executor_bridge version skew)"
        )
    target = command["target"]
    id_field = _OBJECT_ID_FIELDS[family]
    old_values = command.get("old_values")
    new_values = command.get("values") or {}
    return EditCommand(
        edit_id=edit_id,
        scenario_id=scenario_id,
        base_scene_revision=int(scene_revision),
        adapter_id=family,
        operation=operation,
        old_state=({id_field: target, **dict(old_values)} if old_values else None),
        new_state=(
            None if operation == "delete" else {id_field: target, **dict(new_values)}
        ),
        requested_outputs=tuple(requested_outputs),
        requested_times=requested_times,
    )


def _landcover_delta_command(
    command: Mapping[str, Any],
    *,
    scenario_id: str,
    scene_revision: int,
    edit_id: str,
    requested_outputs: Sequence[str],
    requested_times: tuple[int, ...],
) -> EditCommand:
    target = command["target"] or {}
    cells = [(int(row), int(col)) for row, col in target.get("cells", ())]
    if not cells:
        raise RuntimeError(
            "exact lane: landcover delta paint carries no cells — the "
            "reducer never emits empty paints (version skew)"
        )
    rows = [row for row, _ in cells]
    cols = [col for _, col in cells]
    window = {
        "row_start": min(rows),
        "row_stop": max(rows) + 1,
        "col_start": min(cols),
        "col_stop": max(cols) + 1,
    }
    return EditCommand(
        edit_id=edit_id,
        scenario_id=scenario_id,
        base_scene_revision=int(scene_revision),
        adapter_id="landcover_surface",
        operation="paint",
        old_state=None,
        new_state={"window": window, "classes": command["values"].get("class")},
        requested_outputs=tuple(requested_outputs),
        requested_times=requested_times,
    )


def _exact_ledger_replay_window(
    context: RunnerContext, request: SolveRequest
) -> _ReplayWindow:
    """The replay window resolved EAGERLY for the exact lane (incident-3).

    Same inputs as ``_solve_with_executor``'s own call — the coverage
    reconcile is idempotent (a promoted pending consumes itself; a
    dropped one is gone; the second call reads the settled sidecar), so
    the window computed here IS the window ``_pending_commands`` replays
    against a moment later.
    """
    state_dir = _state_directory(context, request.scenario_id)
    coverage = _reconcile_coverage(context, state_dir, request)
    return _ledger_replay_window(state_dir, coverage, request, list(request.events))


def _solve_exact_lane(context: RunnerContext, request: SolveRequest, progress) -> SolveResult:
    """Exact-lane solve: the batch IS the realtime operation span (r2a).

    The lane marker (``SolveRequest.exact_lane``) carries the span the
    runner derived from the epoch plane: ``(consumed_seq, span_hi]`` must
    resolve to exactly the contiguous, complete operation list (ZERO-LOSS
    FENCE — a partial fold publishes wrong science, so it raises, a
    tripwire that survives ``python -O``), every span operation must
    belong to an epoch assigned a revision inside ``(consumed_revision,
    target_revision]`` (per-op epoch fence), and operations funnelled
    from the legacy edit paths (``LEGACY_ACTOR_ID``) are counted and
    SKIPPED — their scene effects ride the authoritative ``edit_events``
    replay in the same job's ledger window, so folding them here too
    would double-apply.

    The native operations fold onto the canonical baseline at
    ``consumed_revision`` through :class:`DeterministicEpochReducer` (pure
    function, deterministic total order); the resulting family deltas
    become executor commands. Mapped this wave: vegetation/building
    object deltas, landcover paints. Segment families and output_view
    are disclosed as skipped (see :data:`_EXACT_LANE_UNMAPPED_FAMILIES`).

    Incident-3 (live 2026-09-07): the batch also replays the ledger
    window (``_pending_commands``). For a tree id that appears in BOTH
    planes the fold is LEDGER-ANCHORED — the id's baseline state becomes
    the ledger after-state and its native operations since its last
    ledger event re-fold onto it (``_ledger_window_anchors`` /
    ``_anchored_baseline`` / ``_anchored_backfill``) — so the two planes
    chain contiguously per object instead of colliding in
    ``_coalesce_object_changes``; the executor seed withholds ids the
    window itself establishes (``exact_lane_seed_exclusions``).

    Incident-3 review BLOCKER 2: for MOVE anchors the re-fold also
    RECOVERS the id's pre-anchor committed native field writes
    (restricted to the fields the move does not canonically own), and
    the one shape the durable state cannot order — a terminal
    pre-anchor native delete — refuses as a typed ``edit_rejected``
    that settles instead of folding a silent divergence.

    Everything else — coverage reconcile, rebuild, supersession watch,
    staging, composition — is the standard executor path, reused via the
    ``extra_commands_factory`` seam so the lane inherits its crash-safety
    and typed-refusal contracts unchanged.
    """
    lane = dict(request.exact_lane or {})
    target_revision = int(lane.get("target_revision") or request.target_scene_version)
    consumed_revision = int(lane.get("consumed_revision") or 0)
    consumed_seq = int(lane.get("consumed_seq") or 0)
    span_hi = int(lane.get("span_hi") or consumed_seq)
    store = context.store
    workspace = request.scenario_id

    # One bounded O(history) read serves BOTH consumers: the fold's span
    # slice below AND the mixed-plane anchors / consumed-chain baseline
    # (contractually whole-history scans — see _ledger_window_anchors).
    # Store.operations_in_span encodes the (after, through] semantics for
    # span-only callers; here the slice derives from the same read.
    all_records = store.operations_since(workspace, 0)
    records = [
        record
        for record in all_records
        if consumed_seq < int(record.server_sequence) <= span_hi
    ]
    sequences = [int(record.server_sequence) for record in records]
    if sequences != list(range(consumed_seq + 1, span_hi + 1)):
        # ZERO-LOSS FENCE: raising (not a typed envelope) — this is an
        # internal invariant break, and the job must fail loudly.
        raise RuntimeError(
            f"exact lane zero-loss fence: operation span "
            f"({consumed_seq}, {span_hi}] resolved to sequences {sequences}; "
            f"every accepted operation in the span must fold exactly once — "
            "refusing to publish a partial fold"
        )
    epoch_revisions = {
        epoch.epoch_id: epoch.workspace_revision
        for epoch in store.epoch_records(workspace)
    }
    native: list[rt_types.Operation] = []
    legacy_skipped = 0
    for record in records:
        revision = epoch_revisions.get(int(record.epoch_id))
        if revision is None or not (
            consumed_revision < int(revision) <= target_revision
        ):
            raise RuntimeError(
                f"exact lane zero-loss fence: operation {record.server_sequence} "
                f"belongs to epoch {record.epoch_id} whose revision "
                f"({revision}) falls outside the span "
                f"({consumed_revision}, {target_revision}]"
            )
        if record.actor_id == LEGACY_ACTOR_ID:
            legacy_skipped += 1  # audit copy: effects ride the ledger replay
            continue
        native.append(_operation_from_record(record))

    reducer = DeterministicEpochReducer()
    bootstrap = store.canonical_state_at(workspace, 0)
    if bootstrap is None:
        # r2a-fix F1: a solve can arrive with epochs already committed
        # while the tick's pinning failed transiently (missing site
        # geometry, a restart race). The scheduler rows at those revisions
        # were folded from ``initial_state`` — legacy-blind — so re-derive
        # and pin the baseline HERE (cheap, idempotent) rather than
        # consulting those rows blindly. Without this the B1b fence would
        # refuse the span loudly; the blip is avoidable.
        bootstrap = ensure_exact_lane_bootstrap(
            store, workspace, grid=request.grid
        )
    if consumed_revision > 0:
        if bootstrap is not None:
            # r2a-fix BLOCKING-1a (chain re-fold): a pinned revision-0
            # bootstrap means the workspace had PRE-GATE legacy history.
            # The scheduler's committed canonical rows were folded from
            # whatever baseline existed at THEIR close time — rows committed
            # before the pin are legacy-blind — so the row AT
            # ``consumed_revision`` cannot be trusted as this fold's
            # baseline. Re-fold the whole consumed chain from the pin
            # instead: a pure, deterministic function of durable state
            # (the pinned baseline plus every operation of every epoch
            # with revision in (0, consumed_revision], native and
            # funnelled alike — exactly the set the scheduler folded, in
            # the reducer's own total order). Equivalent to the scheduler
            # rows whenever no pre-gate history exists.
            chain = [
                _operation_from_record(record)
                for record in all_records
                if (
                    revision := epoch_revisions.get(int(record.epoch_id))
                )
                is not None
                and 0 < int(revision) <= consumed_revision
            ]
            baseline = reducer.reduce_epoch(bootstrap, chain).state
        else:
            baseline = store.canonical_state_at(workspace, consumed_revision)
            if baseline is None:
                raise RuntimeError(
                    f"exact lane zero-loss fence: canonical state at revision "
                    f"{consumed_revision} is missing — the fold baseline is part "
                    "of the zero-loss contract"
                )
    else:
        # r2a-fix BLOCKING-1a (gate-open bootstrap): at the start of the
        # collaborative era the fold baseline must carry the legacy scene.
        # The tick pins it before creating the job; the re-derivation
        # above (same idempotent function) covers jobs whose tick could
        # not pin — a crashed tick, a restart race, a direct call. A None
        # here means no foldable legacy history: the empty initial state
        # is correct.
        baseline = bootstrap if bootstrap is not None else reducer.initial_state(workspace)

    # Incident-3 (live 2026-09-07): the mixed-plane anchor. Resolve the
    # SAME replay window ``_pending_commands`` will use — eagerly, via
    # the pure branch decision (the coverage reconcile is idempotent;
    # ``_solve_with_executor`` re-runs it and observes the same result) —
    # and for every tree id whose ledger event this window replays,
    # re-anchor the fold: the id's baseline state becomes the LEDGER's
    # after-state and its native operations since that event re-fold
    # onto it, so the fold's delta chains contiguously onto the ledger
    # replay's after-state instead of colliding with it.
    window = _exact_ledger_replay_window(context, request)
    anchors = _ledger_window_anchors(request, window, all_records)
    fold_baseline = _anchored_baseline(baseline, anchors)
    seed_exclusions = frozenset(
        anchor.tree_id for anchor in anchors.values() if anchor.established_by_window
    )

    lane_metrics: dict[str, Any] = {
        "target_revision": target_revision,
        "consumed_revision": consumed_revision,
        "span_first_seq": consumed_seq + 1,
        "span_last_seq": span_hi,
        "span_native_ops": len(native),
        "legacy_ops_skipped": legacy_skipped,
        "ledger_anchored_objects": len(anchors),
    }
    try:
        backfill = _anchored_backfill(
            anchors, all_records, sequences, epoch_revisions, consumed_revision
        )
    except SourceDeltaError as error:
        # Review BLOCKER 2's caveat: a cross-plane divergence the durable
        # state cannot order (a terminal pre-anchor native delete under a
        # move-anchor window event). Refuse TYPED — the edit_rejected
        # envelope settles the span downstream — never fold a silent
        # divergence, never rethrow raw.
        logger.error(
            "exact lane: refusing span (%d, %d] for %s — %s",
            consumed_seq,
            span_hi,
            workspace,
            error,
        )
        return SolveResult(
            status="superseded",
            scene_version=request.target_scene_version,
            error={"code": "edit_rejected", "message": str(error)},
            metrics={"exact_lane": {**lane_metrics, "anchor_refusals": 1}},
        )
    lane_metrics["anchored_backfill_ops"] = len(backfill)
    unfoldable = _unfoldable_native_object_ops(
        fold_baseline, [*native, *backfill]
    )
    if unfoldable:
        # r2a-fix BLOCKING-1b fence: an accepted native delete/replace/move
        # of an object id NO fold baseline can ever know (no pre-gate
        # legacy event carried it, no earlier native op added it) must
        # never silently fold to no delta — the reducer would audit it as
        # a no-op and the published result would keep the object's shadow,
        # undisclosed. Refuse LOUDLY (typed edit_rejected + lane metric);
        # the runner settles the span and fails the job rather than
        # wedging behind the churn guard.
        operation_ids = [op.operation_id for op in unfoldable]
        entities = sorted({op.entity_id for op in unfoldable if op.entity_id})
        logger.error(
            "exact lane: refusing span (%d, %d] for %s — %d accepted "
            "native operation(s) reference object id(s) the fold history "
            "can never resolve (%s); folding would silently drop their "
            "promised state change",
            consumed_seq,
            span_hi,
            workspace,
            len(unfoldable),
            ", ".join(entities),
        )
        return SolveResult(
            status="superseded",
            scene_version=request.target_scene_version,
            error={
                "code": "edit_rejected",
                "message": (
                    "exact lane: accepted native operation(s) "
                    f"{operation_ids} reference object id(s) {entities} "
                    "unknown to every fold baseline (pre-gate legacy "
                    "history the realtime plane cannot reconstruct); "
                    "refusing to publish a silently-partial fold"
                ),
                "unfoldable_operations": operation_ids,
            },
            metrics={
                "exact_lane": {
                    **lane_metrics,
                    "unfoldable_native_ops": len(unfoldable),
                }
            },
        )
    reduction = reducer.reduce_epoch(fold_baseline, [*native, *backfill])

    families_skipped: set[str] = set()

    def exact_lane_commands(
        executor: PlanExecutor,
        variables: Sequence[str],
        all_times: tuple[int, ...],
    ) -> list[EditCommand]:
        commands: list[EditCommand] = []
        revision = executor.scene_revision
        for family, delta in sorted(reduction.reduced.family_deltas.items()):
            if delta.is_noop:
                continue
            if family in _EXACT_LANE_UNMAPPED_FAMILIES:
                families_skipped.add(family)
                logger.warning(
                    "exact lane: skipping %d %s delta command(s) for %s — "
                    "segment/view families are not yet mapped onto adapter "
                    "commands (r2a scope; disclosed in metrics)",
                    len(delta.commands),
                    family,
                    workspace,
                )
                continue
            for position, delta_command in enumerate(delta.commands):
                common = {
                    "scenario_id": request.scenario_id,
                    "scene_revision": revision,
                    "edit_id": f"{family}-seq{consumed_seq}-{position}",
                    "requested_times": all_times,
                }
                if family in _OBJECT_ID_FIELDS:
                    commands.append(
                        _object_delta_command(
                            family,
                            delta_command,
                            requested_outputs=EXECUTOR_REQUESTED_VARIABLES,
                            **common,
                        )
                    )
                elif family == "landcover_surface":
                    commands.append(
                        _landcover_delta_command(
                            delta_command,
                            requested_outputs=EXECUTOR_REQUESTED_VARIABLES,
                            **common,
                        )
                    )
                else:
                    families_skipped.add(family)
        return commands

    result = _solve_with_executor(
        context,
        request,
        progress,
        extra_commands_factory=exact_lane_commands,
        exact_lane_baseline=baseline,
        exact_lane_seed_exclusions=seed_exclusions,
    )
    if families_skipped:
        lane_metrics["families_skipped"] = sorted(families_skipped)
    result = replace(
        result, metrics={**dict(result.metrics or {}), "exact_lane": lane_metrics}
    )
    if families_skipped and not result.error:
        # r2a-fix MEDIUM-1 (published path): the skip disclosure travels
        # INSIDE the result's limitations, so ``materialize_result`` bakes
        # it into the published manifest where a client reading the result
        # sees it — job metrics and logs alone let a bitwise-incomplete
        # scene pass as complete.
        disclosure = exact_lane_skip_disclosure(
            sorted(families_skipped), target_revision
        )
        limitations = tuple(result.limitations) or default_limitations
        result = replace(result, limitations=limitations + (disclosure,))
    return result


# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------


def _solve_with_executor(
    context: RunnerContext,
    request: SolveRequest,
    progress,
    *,
    extra_commands_factory: Callable[..., Sequence[EditCommand]] | None = None,
    exact_lane_baseline: rt_types.CanonicalState | None = None,
    exact_lane_seed_exclusions: frozenset[str] = frozenset(),
) -> SolveResult:
    """Executor-path solve; ``extra_commands_factory`` (r2a exact lane)
    appends commands derived from the realtime operation span.

    The factory runs inside this function's guarded region, bound to the
    live executor (its ``scene_revision`` is the commands' mandatory
    ``base_scene_revision``), and its commands pass the same
    ``executor.validate`` gate as the ledger-derived ones.
    ``exact_lane_baseline`` (exact-lane jobs only) is the canonical fold
    baseline the executor's layer is seeded from on a fresh rebuild;
    ``exact_lane_seed_exclusions`` (incident-3) withholds ids whose
    ledger-window events this same job replays as ADDs.
    """
    cache = context.sites.cache(request.site_id)
    grid_info = context.sites.geometry(request.site_id)
    grid = RasterGrid(
        rows=int(grid_info["rows"]),
        cols=int(grid_info["cols"]),
        pixel_size_m=float(grid_info["pixel_size_m"]),
        origin_x_m=float(grid_info["origin_x_m"]),
        origin_y_m=float(grid_info["origin_y_m"]),
    )
    time_indices, variables = resolve_requested(request.requested, grid_info)
    total_time_steps = int(cache.time_steps)
    state_dir = _state_directory(context, request.scenario_id)

    try:
        # Pending adoption happens exactly once per job, before anything is
        # built, against durable store state (see _reconcile_coverage).
        coverage = _reconcile_coverage(context, state_dir, request)
        executor, watch, covered, replay_tree_events = _build_executor(
            context, request, grid=grid, coverage=coverage,
            exact_lane_baseline=exact_lane_baseline,
            exact_lane_seed_exclusions=exact_lane_seed_exclusions,
        )
        _guard_ledger_completeness(request, covered)
        progress("windowing", 0, 1)
        # Perf wave 1 (STEP 2a): an explicit, non-``refine_full_day`` time
        # slice truncates the commands' requested series to the causal
        # prefix ``t = 0..max`` — the plan's temporal replay and the
        # worker's solve bound then agree, and every computed step stays
        # bit-identical to the full-series run. Computed ONCE here; both
        # the ledger window and any exact-lane span commands share it.
        time_stop = resolve_time_stop(request.requested, grid_info)
        all_times = (
            tuple(range(time_stop))
            if time_stop is not None
            else tuple(range(int(grid_info["time_steps"])))
        )
        commands = _pending_commands(
            executor,
            request,
            grid_info,
            covered,
            replay_tree_events=replay_tree_events,
            variables=variables,
            all_times=all_times,
        )
        if extra_commands_factory is not None:
            commands.extend(
                executor.validate(command)
                for command in extra_commands_factory(
                    executor, variables, all_times
                )
            )
        if extra_commands_factory is not None and not commands:
            # r2a-fix BLOCKING-2: an exact-lane span whose every delta maps
            # to no executor command (a met-only or all-unmapped-family
            # epoch) is a LEGITIMATE no-op, not a plan error — the
            # PlanExecutor refuses empty batches by contract, which at
            # base surfaced as edit_rejected, superseded the job, and
            # wedged the epoch behind the tick's churn guard forever.
            # Mirror the executor's own no-op branch: stage the snapshot
            # (the ledger window is consumed) and let the runner re-serve
            # the latest exact result at the new version.
            _stage_state(executor, state_dir, request)
            return SolveResult(
                status="no-op",
                scene_version=request.target_scene_version,
                metrics={"mode": "no-op"},
            )
        executed = executor.execute(commands)
    except _Unrecoverable as error:
        return SolveResult(
            status="superseded",
            scene_version=request.target_scene_version,
            error={"code": "scenario_state_unrecoverable", "message": str(error)},
        )
    except ScenarioStateError as error:
        # u-e1 F2: a corrupt/unrestorable snapshot (torn write, tampered
        # document, unreadable memo) is the SAME class of failure as the
        # bridge's own _Unrecoverable — the durable state cannot rebuild
        # the scenario exactly. At base it escaped this list and wedged
        # every later family job behind a generic job_failed until reset;
        # the typed envelope names the failure class, repeats
        # idempotently, and heals on reset (the restore gate bypasses the
        # corrupt snapshot after a reset event).
        return SolveResult(
            status="superseded",
            scene_version=request.target_scene_version,
            error={"code": "scenario_state_unrecoverable", "message": str(error)},
        )
    except (ExecutorError, SolverInputError) as error:
        # u-e1 F4: the executor/solver refusing its own machinery (an
        # inconsistent worker state, an unsafe local job input) is a
        # typed ENGINE refusal — distinct from edit_rejected (the batch
        # was valid; the engine could not run it) and never a generic
        # job_failed. Deterministic failures must surface with their
        # message so an operator can tell a state problem (F2) from an
        # engine problem (here).
        return SolveResult(
            status="superseded",
            scene_version=request.target_scene_version,
            error={"code": "engine_refused", "message": str(error)},
        )
    except (EditStateError, SourceDeltaError, AdapterSchemaError, PlanningError) as error:
        # The executor re-validates every command through the registry and
        # fences the batch at plan time; a refusal here means state drifted
        # since HTTP validation (both validator vocabularies relay).
        # SourceDeltaError (incident-3): ``_coalesce_object_changes``
        # refusing a non-contiguous per-object chain is EditStateError's
        # SIBLING, not its subclass — at base it escaped this list as a raw
        # job_failed that never settled its span, wedging the exact lane
        # behind the churn guard (live 2026-09-07). The mixed-plane anchor
        # makes the chain contiguous for window-carried ids; any residual
        # refusal (cross-plane divergences the durable state cannot order)
        # surfaces as this typed edit_rejected and SETTLES.
        return SolveResult(
            status="superseded",
            scene_version=request.target_scene_version,
            error={"code": "edit_rejected", "message": str(error)},
        )

    if not executed.published and executed.status != "no-op":
        # The executor's own supersession checkpoint tripped (concurrent
        # commit advanced the scene): the store's next job replays the
        # whole window from the unchanged snapshot.
        return SolveResult(
            status="superseded",
            scene_version=request.target_scene_version,
            metrics={"mode": executed.status},
        )

    if watch.tripped():
        # A store commit landed mid-solve at a checkpoint the executor's
        # internal revision could not see (e.g. inside the building
        # regeneration chain, whose internal checkpoints track the executor
        # revision only): the runner would discard this result as stale, so
        # refuse it here before any state is staged — legacy parity.
        return SolveResult(
            status="superseded",
            scene_version=request.target_scene_version,
            metrics={"mode": executed.mode if executed.published else executed.status},
        )

    if executed.status == "no-op" or not executed.patch_paths:
        # Value-equal batch (or view-only): the runner re-serves the latest
        # published result at the target version. Stage the snapshot so the
        # consumed events never replay again — it is adopted only once this
        # job's row reaches ``complete`` (the no-op re-serve is durable
        # exactly then).
        _stage_state(executor, state_dir, request)
        return SolveResult(
            status="no-op",
            scene_version=request.target_scene_version,
            metrics={"mode": "no-op"},
        )

    healed = False
    try:
        progress("time_loop", total_time_steps, total_time_steps, mode=executed.mode)
        progress("publishing", 0, 1)
        patches = [load_patch(path) for path in executed.patch_paths]
        union = patches[0].write_window
        for patch in patches[1:]:
            union = union.union(patch.write_window)

        if context.store.result_versions(request.scenario_id):
            state = _compose_current_state(
                context,
                request.scenario_id,
                grid_info,
                variables,
                union,
                request.target_scene_version,
                extra_patches=patches,
            )
            window = union
        else:
            state = _bootstrap_state(context, request, cache, grid, variables, patches)
            window = grid.full_window
    except selected_time.TimeCoverageGap:
        # T15 heal (witness: accepted operation loss): the durable history
        # carries an UNHEALED request-cut record, so the incremental compose
        # refused rather than silently keep the cut revision's uncovered
        # times stale. Rebuild from the fold + patches (_bootstrap_state)
        # and publish FULL coverage — full site window, every time step —
        # so this publication heals the gap. The rebinding of
        # ``time_indices`` below makes the request-cut marker evaluate
        # False for this result (the payload is no longer cut).
        progress("time_loop", total_time_steps, total_time_steps, mode="full")
        state = _bootstrap_state(context, request, cache, grid, variables, patches)
        window = grid.full_window
        time_indices = tuple(range(total_time_steps))
        healed = True
    except _Unrecoverable as error:
        return SolveResult(
            status="superseded",
            scene_version=request.target_scene_version,
            error={"code": "scenario_state_unrecoverable", "message": str(error)},
        )

    selected = {
        name: np.ascontiguousarray(state[name][list(time_indices)], dtype=np.float32)
        for name in variables
    }
    # Stage (never promote): the sidecar advances only when the NEXT family
    # job observes this one durably complete in the store.
    _stage_state(executor, state_dir, request)
    # Honest read-side telemetry (perf wave 1 STEP 1a): the executed plan
    # carries the read union its solves touched. Same additive-field
    # contract as the tree-path adapter above.
    read_window_fraction = union_window_area_fraction(
        executed.read_windows, grid_info
    )
    if read_window_fraction is None:
        read_window_fraction = (
            1.0 if executed.mode == "full" else round(window.area / grid.area_pixels, 6)
        )
    # Routing observability (policy wave): lift the worker's TRUE pre-mode
    # dirty fraction and its demotion verdict into the persisted metrics —
    # a full-mode job must always be able to say WHY it is full (fraction
    # demotion vs a legitimately whole-tile dirty union), instead of
    # silently demoting at the threshold.
    worker_diagnostics = executed.diagnostics or {}
    metrics = {
        "fallback_reason": executed.fallback_reason,
        "window_fraction": round(window.area / grid.area_pixels, 6),
        "read_window_fraction": read_window_fraction,
        "dirty_fraction": worker_diagnostics.get("dirty_fraction"),
        "demoted_by_fraction": bool(
            worker_diagnostics.get("demoted_by_fraction", False)
        ),
        "dirty_windows": [
            {
                "row_start": w.row_start,
                "row_stop": w.row_stop,
                "col_start": w.col_start,
                "col_stop": w.col_stop,
            }
            for w in executed.write_windows
        ],
        "transport": "executor",
        **stage_timing_metrics(
            (executed.diagnostics or {}).get("stage_timings")
        ),
    }
    if healed:
        metrics["fallback_reason"] = "time_coverage_heal"
        metrics["time_coverage_heal"] = True
    return SolveResult(
        status="published",
        scene_version=request.target_scene_version,
        mode=executed.mode,
        window=window,
        time_indices=time_indices,
        variables=variables,
        arrays=selected,
        metrics=metrics,
        # T15 cut site (executor path): the composed state was cut to the
        # REQUESTED times before publishing — unless the heal above fired,
        # which rebound ``time_indices`` to the full axis (marker False).
        requested_time_subset=len(time_indices) < total_time_steps,
        plan=executed_plan_document(executed),
    )


def _stage_state(executor: PlanExecutor, state_dir: Path, request: SolveRequest) -> None:
    """Stage the quiescent executor's snapshot as PENDING adoption.

    Snapshots are written only between batches (``execute`` returned), so
    the u-c1b pairing always holds. Ordering discipline (crash-safe): the
    snapshot + its ``pending.json`` metadata are written into a staging
    directory (``write_snapshot`` itself is npy-first/JSON-last), then
    swapped into ``pending/`` with ONE atomic ``os.replace``. A crash at
    any point leaves either the previous pending (consumed by this job's
    own reconcile) or none — the sidecar never names staged state, so a
    crashed solve's events stay pending replay.
    """
    staging = state_dir / PENDING_STAGING_NAME
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    snapshot = snapshot_from_executor(executor)
    write_snapshot(
        snapshot, staging, landcover_resolved=executor._applied_landcover_resolved
    )
    (staging / PENDING_META_FILENAME).write_text(
        json.dumps(
            {
                "job_id": str(request.job_id),
                "covered_sequence": int(request.edit_watermark),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    # The previous pending (if any) was consumed by this job's reconcile,
    # so the target is free; removing it first is pure paranoia for the
    # pathological never-reconciled case.
    shutil.rmtree(state_dir / PENDING_DIRECTORY_NAME, ignore_errors=True)
    os.replace(staging, state_dir / PENDING_DIRECTORY_NAME)


def _bootstrap_state(
    context: RunnerContext,
    request: SolveRequest,
    cache,
    grid: RasterGrid,
    variables: Sequence[str],
    patches: Sequence[Any],
) -> dict[str, np.ndarray]:
    """Full-site state for a scenario with no published results yet.

    Base = the site cache's stored baseline outputs when the deployment
    ships them (the same source ``materialize_baseline_result`` serves);
    the executor's fresh patches overlay them in their write windows.
    """
    site = context.sites.config(request.site_id)
    baseline_dir = site.cache_dir / "baseline_results"
    state: dict[str, np.ndarray] = {}
    stored = [
        name
        for name in variables
        if (baseline_dir / f"{name}.f32.npy").is_file()
    ]
    for name in stored:
        array = np.asarray(np.load(baseline_dir / f"{name}.f32.npy"))
        expected = (int(cache.time_steps), grid.rows, grid.cols)
        if array.shape != expected:
            # u-e1 F5: a stored baseline whose shape does not match the
            # site extent would otherwise be padded (or truncated) into
            # the composition — a silent NaN tail served as science. The
            # mis-built cache is named so the operator can rebuild it;
            # refusing is the only safe outcome.
            raise _Unrecoverable(
                f"stored baseline {baseline_dir / (name + '.f32.npy')} has "
                f"shape {array.shape}, expected {expected} "
                f"(time_steps x rows x cols); the site cache is mis-built — "
                "refusing instead of serving a padded composition"
            )
        full = np.full(expected, np.nan, dtype=np.float32)
        full[: array.shape[0]] = array
        state[name] = full
    missing = [name for name in variables if name not in stored]
    if missing:
        # No (or partial) stored baseline outputs: solve the missing
        # variables with ONE full-tile run of the current scene — baseline
        # forcing, baseline landcover, the authoritative tree list, NO
        # family overlays — mirroring the legacy solver's documented
        # bootstrap ("the next job bootstraps with one full-tile solve at
        # its own target version"): a superseded baseline job must not
        # brick the scenario.
        from solweig_gpu.incremental.solver import load_site_forcing, run_full_tile

        layer = TreeLayer(np.asarray(cache.tree_base), grid, scenario_id=request.scenario_id)
        for tree in request.trees:
            layer.add_tree(api_tree_to_spec(tree, request.grid))
        base = run_full_tile(
            cache,
            layer,
            forcing=load_site_forcing(
                cache,
                site_dir=site.site_dir or site.cache_dir,
                selected_date_str=site.selected_date_str,
            ),
            site_dir=site.site_dir or site.cache_dir,
            scratch_dir=context.store.results_root
            / request.scenario_id
            / "executor"
            / "bootstrap-scratch",
            requested_variables=missing,
        )
        for name in missing:
            state[name] = np.asarray(base[name])
    for patch in patches:
        window = patch.write_window
        region = (
            slice(window.row_start, window.row_stop),
            slice(window.col_start, window.col_stop),
        )
        for name in variables:
            if name not in patch.arrays:
                continue
            block = np.asarray(patch.arrays[name])
            # Sparse patches (r3a met fast path) carry their global
            # timesteps explicitly and cover ONLY those rows; the block
            # offset is NOT the destination timestep (a changed-t plane
            # would land at t=0 while the changed t kept its stale
            # value). Legacy patches cover t = 0..n contiguously.
            global_times = (
                patch.time_indices
                if patch.time_indices is not None
                else range(block.shape[0])
            )
            for offset in range(block.shape[0]):
                state[name][(global_times[offset],) + region] = block[offset]
    return state
