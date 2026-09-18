# SPDX-License-Identifier: GPL-3.0-only
"""Background job runner and solver contract for the design-tool API.

The :class:`JobRunner` is a single background worker thread (one per
application) that executes exact-analysis jobs serialized through a queue:

* every job is durable in SQLite before it is queued;
* a configurable coalescing window (default 500 ms) runs before dispatch so
  rapid edits batch into one job (behind the T24 flag, the window shrinks to
  the service contract's latency floor when no burst indicator is present —
  see :func:`adaptive_coalescing_enabled`);
* when a newer edit arrives, older *queued* jobs are marked ``superseded`` and
  never dispatched;
* a job that finishes after the scene moved on is *not published*: the runner
  re-checks that ``result.scene_version == current scene version`` before
  calling :meth:`Store.publish_result` (which enforces the same guard again).

The scientific solver is injected. The app factory accepts any
``solver_factory``; the default :func:`make_exact_worker_solver` adapts the
Phase-5 :class:`~solweig_gpu.incremental.worker.ExactWorker`. Tests inject a
fake solver, so no SOLWEIG physics runs in the API unit tests.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from solweig_gpu.incremental.cache import SiteCache
from solweig_gpu.incremental.geometry import (
    RasterGrid,
    RasterWindow,
    TreeSpec,
    merge_windows,
)
from solweig_gpu.incremental.manifest import CACHE_MODEL_VERSION, SiteManifest, load_manifest
from solweig_gpu.incremental.result import load_patch
from solweig_gpu.incremental.solver import run_full_tile
from solweig_gpu.incremental.trees import TreeLayer
from solweig_gpu.incremental.worker import ExactWorker
from solweig_gpu.server import patch_codec
from solweig_gpu.server.realtime import publish_fence, selected_time, types as rt_types
from solweig_gpu.server.realtime.epochs import EPOCH_TICK_MIN_MS
from solweig_gpu.server.realtime.reducer import DeterministicEpochReducer
from solweig_gpu.server.store import (
    TERMINAL_EPOCH_STATUSES,
    EpochRecord,
    JobRecord,
    ResultAlreadyPublished,
    ResultNotPublishable,
    ResultRecord,
    ScenarioRecord,
    Store,
    StoreError,
    StaleResultError,
    _now_utc,
    result_payload_url,
    uv_to_world,
)
from solweig_gpu.server.patch_codec import build_manifest, encode_payload

logger = logging.getLogger(__name__)

__all__ = [
    "JobRunner",
    "LEGACY_ACTOR_ID",
    "RECONCILE_ATTEMPTS_PER_DAY",
    "RunnerContext",
    "SiteConfig",
    "SiteRegistry",
    "SiteNotRegistered",
    "SolveRequest",
    "SolveResult",
    "make_exact_worker_solver",
    "ensure_exact_lane_bootstrap",
    "exact_lane_skip_disclosure",
    "materialize_baseline_result",
    "resolve_requested",
    "resolve_time_stop",
    "union_window_area_fraction",
    "default_limitations",
]

#: ``actor_id`` marking realtime operations funnelled from the legacy edit
#: paths (r2a). The exact lane recognizes the marker and SKIPS these
#: operations when deriving its executor commands — their scene effects
#: come from the authoritative ``edit_events`` replay inside the same job,
#: so folding them into commands too would double-apply. The operations
#: still land in the epoch plane: the op log is the collaborative audit
#: and the reducer's canonical-state input, regardless of origin.
LEGACY_ACTOR_ID = "legacy_edits"

#: Model-scope limitations disclosed with every result manifest.
default_limitations = (
    "Tree-induced local wind-field changes are not recomputed.",
    "Results are exact within the published window; cells outside it are "
    "unchanged from the baseline scene.",
)

#: Reconcile attempt ceiling per workspace per UTC day (routing-policy
#: review follow-up): without a COMPLETION, at most this many reconcile
#: mints run per day — a crash-class poison request must not loop a full
#: solve roughly continuously on the 600 s backoff. In-process like the
#: backoff map; a restart clears it (tolerated: the durable debt survives).
RECONCILE_ATTEMPTS_PER_DAY = 5


def exact_lane_skip_disclosure(
    families: Sequence[str], revision: int
) -> str:
    """Manifest limitation disclosing realtime families the exact lane
    folded into canonical state but could NOT map onto executor commands
    (r2a-fix MEDIUM-1: the disclosure belongs in the PUBLISHED manifest,
    where a client reading the result sees it — job metrics and logs alone
    let a bitwise-incomplete scene pass as complete).

    ``revision`` is the canonical workspace revision the result covers, so
    the limitation is auditable against the epoch plane.
    """
    return (
        "exact lane: realtime families "
        + ", ".join(sorted(str(family) for family in families))
        + " have no executor command mapping yet; their changes are "
        f"excluded from this result (canonical revision {int(revision)})"
    )


# ---------------------------------------------------------------------------
# Site registry
# ---------------------------------------------------------------------------


class SiteNotRegistered(KeyError):
    def __init__(self, site_id: str) -> None:
        self.site_id = site_id
        super().__init__(f"site {site_id!r} is not registered with this server")


@dataclass(frozen=True)
class SiteConfig:
    """Deployment configuration for one prepared site cache."""

    cache_dir: Path
    site_dir: Path | None = None
    selected_date_str: str = "2024-06-20"

    def __post_init__(self) -> None:
        object.__setattr__(self, "cache_dir", Path(self.cache_dir))
        if self.site_dir is not None:
            object.__setattr__(self, "site_dir", Path(self.site_dir))


class SiteRegistry:
    """Immutable site catalog; manifests and caches are loaded lazily once."""

    def __init__(self, configs: Mapping[str, SiteConfig | Mapping[str, Any]]) -> None:
        self._configs: dict[str, SiteConfig] = {}
        for site_id, config in dict(configs).items():
            if isinstance(config, SiteConfig):
                self._configs[site_id] = config
            else:
                self._configs[site_id] = SiteConfig(
                    cache_dir=Path(config["cache_dir"]),
                    site_dir=Path(config["site_dir"]) if config.get("site_dir") else None,
                    selected_date_str=str(config.get("selected_date_str", "2024-06-20")),
                )
        self._manifests: dict[str, SiteManifest] = {}
        self._caches: dict[str, SiteCache] = {}
        self._lock = threading.Lock()

    def site_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._configs))

    def __contains__(self, site_id: object) -> bool:
        return site_id in self._configs

    def config(self, site_id: str) -> SiteConfig:
        try:
            return self._configs[site_id]
        except KeyError:
            raise SiteNotRegistered(site_id) from None

    def manifest(self, site_id: str) -> SiteManifest:
        """Load and cache the site manifest via ``incremental.load_manifest``.

        The manifest is the single source of truth for UV-to-world conversion;
        no frontend constants are involved anywhere.
        """
        with self._lock:
            manifest = self._manifests.get(site_id)
            if manifest is None:
                config = self.config(site_id)
                manifest = load_manifest(config.cache_dir / "manifest.json")
                if manifest.site_id != site_id:
                    raise SiteNotRegistered(
                        f"cache at {config.cache_dir} declares site "
                        f"{manifest.site_id!r}, expected {site_id!r}"
                    )
                self._manifests[site_id] = manifest
            return manifest

    def cache(self, site_id: str) -> SiteCache:
        with self._lock:
            cache = self._caches.get(site_id)
            if cache is None:
                config = self.config(site_id)
                cache = SiteCache.load(config.cache_dir)
                self._caches[site_id] = cache
            return cache

    def geometry(self, site_id: str) -> dict[str, float | int]:
        manifest = self.manifest(site_id)
        return {
            "rows": manifest.rows,
            "cols": manifest.cols,
            "pixel_size_m": manifest.pixel_size_m,
            "origin_x_m": manifest.origin_x_m,
            "origin_y_m": manifest.origin_y_m,
            "time_steps": manifest.time_steps,
        }

    def site_cache_version(self, site_id: str) -> str:
        manifest = self.manifest(site_id)
        return f"{manifest.site_id}:cache-{manifest.cache_schema_version}"


# ---------------------------------------------------------------------------
# Solver contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SolveRequest:
    """Everything a solver needs to evaluate one job."""

    job_id: str
    scenario_id: str
    site_id: str
    target_scene_version: int
    #: Store sequence of the LAST edit included in this job.
    edit_watermark: int
    #: Store sequence watermark already consumed by a *published* result when
    #: this job was created; the solver's pending batch is everything in
    #: ``(base_edit_watermark, edit_watermark]``. (Using ``edit_watermark``
    #: here instead would make the batch empty: every job would no-op.)
    base_edit_watermark: int = 0
    trees: tuple[Mapping[str, Any], ...] = ()  # authoritative API tree objects
    events: tuple[Mapping[str, Any], ...] = ()  # full edit ledger (ascending)
    requested: Mapping[str, Any] = field(default_factory=dict)
    grid: Mapping[str, float | int] = field(default_factory=dict)
    #: Exact-lane routing marker (r2a). Present (non-None) when the job was
    #: created by the runner's exact-lane tick from the epoch plane; the
    #: dispatcher routes it onto the executor bridge, which derives the
    #: batch from the operation span below. ``None`` for every legacy job
    #: (baseline, /edits, /edits/universal) — byte-identical routing.
    exact_lane: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class SolveResult:
    """What one solver run produced."""

    status: str  # "published" | "superseded" | "no-op"
    scene_version: int
    mode: str | None = None
    window: RasterWindow | None = None
    time_indices: tuple[int, ...] = ()
    variables: tuple[str, ...] = ()
    arrays: Mapping[str, np.ndarray] = field(default_factory=dict)
    metrics: Mapping[str, Any] = field(default_factory=dict)
    limitations: tuple[str, ...] = ()
    error: Mapping[str, Any] | None = None
    #: Serialized executed impact plan (executor-path jobs only, u-d4):
    #: per-node stages, realized routing, estimates vs actuals. Tree-only
    #: jobs leave it None; clients derive stages for those.
    plan: Mapping[str, Any] | None = None
    #: T15 request-cut marker: the durable payload was cut to a STRICT
    #: SUBSET of the site's time axis by the client's
    #: ``requested_result.time_indices`` (computed then discarded). Flag-
    #: independent (the fact is true regardless); only the DISCLOSURE,
    #: guard, and coverage surface check the flag. Native sparse patches
    #: (r3a met fast path) leave it False — their time indices ARE their
    #: exact change set.
    requested_time_subset: bool = False


ProgressCallback = Callable[..., None]
Solver = Callable[[SolveRequest, ProgressCallback], SolveResult]


@dataclass
class RunnerContext:
    """Dependencies handed to solver factories."""

    store: Store
    sites: SiteRegistry
    state_root: Path
    #: Process-local cache of composed full-site states (P8: makes result
    #: composition O(new versions) instead of O(all versions) per job).
    state_cache: "_StateCompositionCache" = field(
        default_factory=lambda: _StateCompositionCache()
    )


class _StateCompositionCache:
    """LRU cache of composed full-site state arrays, keyed by checksum chain.

    An entry for ``(scenario, version, variables)`` holds the full-site
    arrays after applying every published result up to ``version``, together
    with the ordered tuple of the applied records' checksums. A cached entry
    is only reused when the store's checksum chain for that version prefix
    still matches, so a scenario reset / re-publish (same version, different
    history) is detected without any store hooks and degrades to a miss.

    Entries are large (time_steps x rows x cols float32 per variable), so
    the cache is bounded by total bytes, evicting least-recently-used
    compositions first.
    """

    def __init__(self, *, max_total_bytes: int = 512 << 20) -> None:
        from collections import OrderedDict

        self._entries: "OrderedDict[tuple[str, int, tuple[str, ...]], tuple[tuple[str, ...], dict[str, np.ndarray]]]" = (
            OrderedDict()
        )
        self._total_bytes = 0
        self._max_total_bytes = int(max_total_bytes)
        self._lock = threading.Lock()

    @staticmethod
    def _nbytes(arrays: Mapping[str, np.ndarray]) -> int:
        return int(sum(arr.nbytes for arr in arrays.values()))

    def get(
        self,
        scenario_id: str,
        version: int,
        variables: Sequence[str],
        chain: Sequence[str],
    ) -> dict[str, np.ndarray] | None:
        key = (scenario_id, int(version), tuple(variables))
        with self._lock:
            entry = self._entries.get(key)
            if entry is None or entry[0] != tuple(chain):
                return None
            cached_chain, arrays = entry
            self._entries.move_to_end(key)
            # Hand out copies: the composer mutates the state in place while
            # applying newer patches; the cached entry must stay pristine.
            return {name: arrays[name].copy() for name in arrays}

    def put(
        self,
        scenario_id: str,
        version: int,
        variables: Sequence[str],
        chain: Sequence[str],
        arrays: Mapping[str, np.ndarray],
    ) -> None:
        key = (scenario_id, int(version), tuple(variables))
        payload = {name: np.ascontiguousarray(arrays[name]) for name in variables}
        size = self._nbytes(payload)
        if size > self._max_total_bytes:
            return
        with self._lock:
            old = self._entries.pop(key, None)
            if old is not None:
                self._total_bytes -= self._nbytes(old[1])
            self._entries[key] = (tuple(chain), payload)
            self._total_bytes += size
            while self._total_bytes > self._max_total_bytes and self._entries:
                _k, evicted = self._entries.popitem(last=False)
                self._total_bytes -= self._nbytes(evicted[1])


def resolve_requested(
    requested: Mapping[str, Any] | None,
    grid: Mapping[str, float | int],
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """Normalize a ``requested_result`` against the site grid.

    Defaults to every site time step and the ``utci``/``tmrt`` variables.
    """
    time_steps = int(grid["time_steps"])
    if requested and requested.get("time_indices"):
        candidates = [int(t) for t in requested["time_indices"]]
    else:
        candidates = list(range(time_steps))
    cleaned: list[int] = []
    for index in candidates:
        if 0 <= index < time_steps and index not in cleaned:
            cleaned.append(index)
    if not cleaned:
        cleaned = list(range(time_steps))
    requested_variables = (requested or {}).get("variables") or ["utci", "tmrt"]
    variables = tuple(
        name
        for name in patch_codec.SUPPORTED_VARIABLES
        if name in set(requested_variables)
    )
    if not variables:
        variables = ("utci",)
    return tuple(cleaned), variables


def resolve_time_stop(
    requested: Mapping[str, Any] | None,
    grid: Mapping[str, float | int],
) -> int | None:
    """Causal time-prefix stop for one ``requested_result`` (perf wave 1).

    The temporal replay inside the solver is forward-only (Tgmap1/TgOut1/
    CI/Twater/firstdaytime/timeadd state chains all carry causally forward
    from timestep 0), so a request for ``time_indices`` can be served by
    replaying only the PREFIX ``t = 0..max``: every computed step is
    bit-identical to the full-day run, and nothing after ``max`` is
    consumed. Returns ``None`` (full series) whenever the request carries no
    explicit ``time_indices``, asks for ``refine_full_day``, or names only
    out-of-series indices (``resolve_requested`` falls those back to the
    full range too).
    """
    if not requested or not requested.get("time_indices"):
        return None
    if requested.get("refine_full_day"):
        return None
    time_steps = int(grid["time_steps"])
    valid = [
        int(index)
        for index in requested["time_indices"]
        if 0 <= int(index) < time_steps
    ]
    if not valid:
        return None
    return max(valid) + 1


def union_window_area_fraction(
    windows: Sequence[RasterWindow],
    grid: Mapping[str, float | int],
) -> float | None:
    """Site-area fraction of the UNION of ``windows``.

    Used for the additive ``read_window_fraction`` job metric (the honest
    counterpart of ``window_fraction``, which covers only the write union):
    the read union is what the solve actually touched. Returns ``None`` for
    an empty sequence so callers can omit the field when nothing ran.
    """
    if not windows:
        return None
    # merge_windows returns the DISJOINT partition of the union (mutually
    # non-intersecting pieces), so the union area is the piece sum — [0]
    # alone would silently drop every window disjoint from the first.
    union_area = sum(piece.area for piece in merge_windows(list(windows)))
    area = int(grid["rows"]) * int(grid["cols"])
    return round(union_area / area, 6)


def stage_timing_metrics(timings: Mapping[str, Any] | None) -> dict[str, float]:
    """Lift worker stage timings into additive metric fields.

    ``svf_seconds`` / ``time_loop_seconds`` come from the worker's solver
    seams (``window_svf_bundle`` / ``run_utci_window``) whenever the path
    reports them; paths without seams (the legacy full-tile orchestrator)
    simply omit the fields. ``svf_march_routed`` / ``svf_march_refused``
    are the T19b lane-router counts for the solve (numba marches executed /
    typed refusals that fell back to the torch kernel) — counts, not
    seconds, riding the same float-typed channel.
    """
    if not timings:
        return {}
    metrics: dict[str, float] = {}
    for name in ("svf_seconds", "time_loop_seconds"):
        value = timings.get(name)
        if value is not None:
            metrics[name] = round(float(value), 3)
    for name in ("svf_march_routed", "svf_march_refused"):
        value = timings.get(name)
        if value is not None:
            metrics[name] = float(value)
    return metrics


# ---------------------------------------------------------------------------
# Result materialization
# ---------------------------------------------------------------------------


def _window_dict(window: RasterWindow) -> dict[str, int]:
    return {
        "row_start": int(window.row_start),
        "row_stop": int(window.row_stop),
        "col_start": int(window.col_start),
        "col_stop": int(window.col_stop),
    }


def job_eta_fields(
    store: Store, sites: SiteRegistry, job: JobRecord
) -> tuple[int | None, str | None]:
    """Honest ETA for a QUEUED or RUNNING job body (routing policy wave).

    ``(eta_seconds, eta_basis)`` from :meth:`Store.estimate_solve_seconds`
    over the scenario's own completed-job history. Running jobs estimate
    once their mode is known (executor-path jobs report mode only at the
    final stage; earlier stages stay honestly null). Queued jobs are
    estimable only when the job's own routing is already known — an
    exact-lane reconcile marker always means a full-tile pass; everything
    else waits for dispatch. Unknown is ``(None, None)``, never a guess.
    """
    if job.status == "running":
        if job.mode not in ("full", "local"):
            return (None, None)
        window_fraction = None
        if job.mode == "local" and job.window:
            scenario = store.get_scenario(job.scenario_id)
            if scenario is None:
                return (None, None)
            try:
                geometry = sites.geometry(scenario.site_id)
                area = int(geometry["rows"]) * int(geometry["cols"])
            except Exception:  # pragma: no cover - defensive
                return (None, None)
            window = dict(job.window)
            height = max(0, int(window.get("row_stop", 0)) - int(window.get("row_start", 0)))
            width = max(0, int(window.get("col_stop", 0)) - int(window.get("col_start", 0)))
            if area > 0 and height > 0 and width > 0:
                window_fraction = round((height * width) / area, 6)
        return store.estimate_solve_seconds(
            job.scenario_id, job.mode, window_fraction=window_fraction
        )
    if job.status == "queued":
        marker = job.request.get("exact_lane") if isinstance(job.request, dict) else None
        if isinstance(marker, dict) and marker.get("reconcile"):
            return store.estimate_solve_seconds(job.scenario_id, "full")
        return (None, None)
    return (None, None)


def materialize_result(
    context: RunnerContext,
    scenario: ScenarioRecord,
    result: SolveResult,
    *,
    duration_ms: float,
) -> tuple[dict[str, Any], bytes]:
    """Encode a published :class:`SolveResult` into an API manifest + payload."""
    if result.window is None or not result.variables or not result.arrays:
        raise StoreError("published solve result is missing window, variables, or arrays")
    manifest_site = context.sites.manifest(scenario.site_id)
    grid = context.sites.geometry(scenario.site_id)
    meta, payload, checksum = encode_payload(result.arrays, list(result.variables))
    window_fraction = round(result.window.area / (int(grid["rows"]) * int(grid["cols"])), 6)
    metrics = {
        **dict(result.metrics),
        "duration_ms": round(duration_ms, 3),
        "mode": result.mode,
        "window_fraction": window_fraction,
    }
    manifest = build_manifest(
        scenario_id=scenario.scenario_id,
        scene_version=int(result.scene_version),
        window=result.window,
        time_indices=tuple(result.time_indices),
        variables=meta,
        payload_url=result_payload_url(scenario.scenario_id, int(result.scene_version)),
        checksum=checksum,
        model_version=manifest_site.model_version,
        site_cache_version=context.sites.site_cache_version(scenario.site_id),
        site_id=scenario.site_id,
        exact=True,
        metrics=metrics,
        limitations=tuple(result.limitations) or default_limitations,
        statistics=patch_codec.result_statistics(result.arrays),
    )
    # T15 request-cut disclosure: flag ON + the solve itself reports its
    # payload was cut to a strict time subset. The block marks the record
    # so no later composition (or client) mistakes it for full coverage.
    # Flag OFF stays byte-identical — the marker alone writes nothing.
    if selected_time.enabled() and result.requested_time_subset:
        manifest[selected_time.MANIFEST_KEY] = selected_time.disclosure_block(
            tuple(result.time_indices), int(grid["time_steps"])
        )
    return manifest, payload


def materialize_baseline_result(
    sites: SiteRegistry,
    scenario_id: str,
    site_id: str,
    scene_version: int = 0,
    *,
    variables: Sequence[str] = ("utci", "tmrt"),
) -> tuple[dict[str, Any], bytes]:
    """Encode the site cache baseline outputs as an exact result.

    Reads ``<cache_dir>/baseline_results/{variable}.f32.npy`` plus
    ``metadata.json``. Shapes are validated against the site manifest (a
    mismatch means the cache does not match the deployed site). Raises
    ``FileNotFoundError`` when the site has no stored baseline outputs, which
    tells the caller to schedule a baseline full-solve job instead.
    """
    config = sites.config(site_id)
    baseline_dir = config.cache_dir / "baseline_results"
    if not (baseline_dir / "metadata.json").is_file():
        raise FileNotFoundError(f"no baseline_results under {config.cache_dir}")
    manifest_site = sites.manifest(site_id)
    grid = sites.geometry(site_id)
    present = [name for name in variables if (baseline_dir / f"{name}.f32.npy").is_file()]
    if not present:
        raise FileNotFoundError(f"baseline_results has none of {list(variables)}")
    expected = (int(grid["time_steps"]), int(grid["rows"]), int(grid["cols"]))
    arrays: dict[str, np.ndarray] = {}
    for name in present:
        array = np.asarray(np.load(baseline_dir / f"{name}.f32.npy", mmap_mode="r"))
        if array.shape != expected:
            raise ValueError(
                f"baseline array {name!r} shape {array.shape} does not match the "
                f"site manifest {expected} (cache_version_mismatch)"
            )
        arrays[name] = np.ascontiguousarray(array, dtype=np.float32)
    meta, payload, checksum = encode_payload(arrays, present)
    manifest = build_manifest(
        scenario_id=scenario_id,
        scene_version=int(scene_version),
        window=RasterWindow(0, int(grid["rows"]), 0, int(grid["cols"])),
        time_indices=tuple(range(int(grid["time_steps"]))),
        variables=meta,
        payload_url=result_payload_url(scenario_id, int(scene_version)),
        checksum=checksum,
        model_version=manifest_site.model_version,
        site_cache_version=sites.site_cache_version(site_id),
        site_id=site_id,
        exact=True,
        metrics={"mode": "baseline"},
        limitations=default_limitations,
        statistics=patch_codec.result_statistics(arrays),
    )
    return manifest, payload


# ---------------------------------------------------------------------------
# Legacy edit funnel (r2a)
# ---------------------------------------------------------------------------


#: Legacy universal-adapter operation -> realtime verb. The realtime plane's
#: verb vocabulary (``realtime.types.OperationVerb``) is coarser than the
#: universal adapters': ``remove``/``delete`` both tombstone, ``update*``
#: and ``preset``/``reset`` are value writes (``set``), and ``move``/
#: ``paint``/``select`` map one-to-one. Anything unmappable is refused
#: (returns ``None``) rather than silently mis-verb'd.
_LEGACY_VERBS: Mapping[str, str] = {
    "add": "add",
    "delete": "delete",
    "remove": "delete",
    "move": "move",
    "paint": "paint",
    "select": "select",
    "update": "replace",
    "update_time_row": "set",
    "update_range": "set",
    "preset": "set",
    "reset": "set",
    "set": "set",
}

#: Integrated adapters whose ids are also ``realtime.types.SourceFamily``
#: members (every universal-edit event the route accepted already carries
#: one of these; ``vegetation_geometry`` travels /edits, not universal).
_LEGACY_FAMILY_ADAPTERS = (
    "building_geometry",
    "landcover_surface",
    "meteorological_forcing",
    "model_receptor_parameters",
    "output_view",
)


def _legacy_tree_values(tree: Mapping[str, Any], grid: Mapping[str, float | int]) -> dict[str, Any]:
    """World-metre field mapping of one API tree object (adapter-side
    decomposition: UV -> world via the site manifest, diameter -> radius)."""
    x_m, y_m = uv_to_world(
        float(tree["u"]),
        float(tree["v"]),
        rows=int(grid["rows"]),
        cols=int(grid["cols"]),
        pixel_size_m=float(grid["pixel_size_m"]),
        origin_x_m=float(grid["origin_x_m"]),
        origin_y_m=float(grid["origin_y_m"]),
    )
    values: dict[str, Any] = {"x_m": x_m, "y_m": y_m}
    if "height_m" in tree:
        values["height_m"] = float(tree["height_m"])
    if "canopy_diameter_m" in tree:
        values["canopy_radius_m"] = float(tree["canopy_diameter_m"]) / 2.0
    if "trunk_ratio" in tree:
        values["trunk_ratio"] = float(tree["trunk_ratio"])
    return values


def _legacy_operation_item(
    event: Mapping[str, Any], grid: Mapping[str, float | int]
) -> dict[str, Any] | None:
    """One legacy ledger event -> one realtime operation item, or None.

    The ``operation_id`` is deterministic in the event id, so a redelivered
    submit (or the store's idempotent replay) never produces a second
    operation. ``base_revision`` carries the event's ``base_scene_version``
    ADVISORY — the route's If-Match equality gate is untouched and nothing
    downstream gates on it (collaborative_state.md revision contract).
    """
    common = {
        "operation_id": f"legacy-{event['event_id']}",
        "actor_id": LEGACY_ACTOR_ID,
        "client_sequence": int(event["sequence"]),
        "base_revision": int(event["base_scene_version"]),
        "received_at": event["submitted_at"],
    }
    family = event.get("family")
    if family:
        adapter = str(family["adapter"])
        if adapter not in _LEGACY_FAMILY_ADAPTERS:
            return None  # not transportable to the realtime plane
        verb = _LEGACY_VERBS.get(str(family["operation"]))
        if verb is None:
            return None
        item: dict[str, Any] = {**common, "source_family": adapter, "verb": verb}
        values = dict(family.get("values") or {})
        if adapter == "building_geometry":
            item["entity_id"] = family.get("target")
            item["payload"] = {"values": values}
        elif adapter == "landcover_surface":
            item["entity_id"] = None
            target = family.get("target") or {}
            payload: dict[str, Any] = {}
            if isinstance(target, Mapping) and "row_start" in target:
                payload["window"] = dict(target)
            if "class" in values:
                payload["class"] = values["class"]
            item["payload"] = payload
        elif adapter == "meteorological_forcing":
            item["entity_id"] = None
            payload = {"values": values}
            if family.get("time_index") is not None:
                payload["time_index"] = family["time_index"]
            item["payload"] = payload
        else:  # model_receptor_parameters, output_view
            item["entity_id"] = None
            item["payload"] = {"values": values}
        if family.get("old_values"):
            item["payload"]["old_values"] = dict(family["old_values"])
        return item

    # Tree event (/edits): decompose into a vegetation operation.
    operation = str(event["operation"])
    verb = _LEGACY_VERBS.get(operation)
    if verb is None:
        return None
    source = event.get("new_tree") or event.get("old_tree")
    if not isinstance(source, Mapping):
        return None
    return {
        **common,
        "source_family": "vegetation_geometry",
        "entity_id": event.get("tree_id"),
        "verb": verb,
        "payload": {"values": _legacy_tree_values(source, grid)},
    }


def ensure_exact_lane_bootstrap(
    store: Store, workspace_id: str, *, grid: Mapping[str, float | int]
) -> Any:
    """Pin the exact lane's revision-0 fold baseline from the legacy scene
    (r2a-fix BLOCKING-1a, PRIMARY fix).

    The hole: an object created via legacy ``/edits`` BEFORE the workspace
    had any realtime operation exists only in the legacy ledgers. The
    epoch scheduler folds from ``initial_state`` when no canonical row
    exists, so every committed canonical revision of such a workspace is
    blind to pre-gate legacy objects — a native delete/replace/move of one
    folded to an audited no-op and the published exact result silently
    kept the deleted object's shadow.

    The fix materializes that history ONCE, at gate-open: when the
    workspace carries realtime operations but no revision-0 canonical row,
    every un-mirrored legacy ledger event (pre-gate ones by definition;
    post-gate edits are mirrored into the operation log by the funnel and
    are recognized by their deterministic ``legacy-<event_id>`` ids, so
    they must NOT fold here — their effects ride the op log and the
    ledger replay) is decomposed through :func:`_legacy_operation_item`
    and folded onto the reducer's initial state. The result is pinned at
    revision 0 (``INSERT OR IGNORE`` — first writer wins, forever), which
    later canonical revisions never collide with (epoch commits assign
    ``scene_version + 1 >= 1``).

    Deterministic + replayable: the pin is a pure function of durable
    state (legacy ledger + operation log), so a crash between the pin and
    the exact job's creation loses nothing — the row is durable and the
    next tick reuses it; a crash before the pin simply re-derives it. The
    solve path re-folds the committed chain from this pin (see
    ``_solve_exact_lane``), so scheduler rows folded legacy-blind before
    the pin are never trusted as the fold baseline. Retention pruning of
    pre-gate events under-knows the fold (documented residual): the
    unfoldable-op fence in the bridge then refuses such spans loudly
    instead of publishing a silently-wrong scene. Reset-after-pin (a
    reset must invalidate the pinned row) lives in
    ``Store.reset_scenario`` — store.py ownership.

    Returns the pinned :class:`CanonicalState`, or ``None`` when there is
    nothing to pin (gate not open, no foldable legacy history, or a
    reset-voided ledger).
    """
    pinned = store.canonical_state_at(workspace_id, 0)
    if pinned is not None:
        return pinned
    operations = store.operations_since(workspace_id, 0)
    if not operations:
        return None  # the collaborative gate has not opened: nothing to pin
    mirrored = {record.operation_id for record in operations}
    fold_events: list[Mapping[str, Any]] = []
    for event in store.list_events(workspace_id):
        if event["operation"] == "reset":
            # A reset voids every earlier object: only post-reset history
            # belongs in the baseline (and the fresh baseline re-publish
            # already matched the reset scene).
            fold_events = []
            continue
        if f"legacy-{event['event_id']}" in mirrored:
            continue  # already in the operation log; the chain re-fold carries it
        fold_events.append(event)
    reducer = DeterministicEpochReducer()
    fold_ops: list[rt_types.Operation] = []
    for event in fold_events:
        item = _legacy_operation_item(event, grid)
        if item is None:
            continue
        fold_ops.append(
            rt_types.Operation(
                workspace_id=workspace_id,
                operation_id=str(item["operation_id"]),
                actor_id=LEGACY_ACTOR_ID,
                client_sequence=item.get("client_sequence"),
                base_revision=item.get("base_revision"),
                source_family=str(item["source_family"]),
                entity_id=item.get("entity_id"),
                verb=str(item["verb"]),
                payload=dict(item["payload"]),
                received_at=str(item["received_at"]),
                accepted_at=str(item["received_at"]),
                server_sequence=int(event["sequence"]),
                epoch_id=-1,
            )
        )
    if not fold_ops:
        return None
    folded = reducer.reduce_epoch(reducer.initial_state(workspace_id), fold_ops).state
    state_json = json.dumps({"families": folded.families}, separators=(",", ":"))
    with store._write() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO realtime_canonical_state "
            "(workspace_id, workspace_revision, state_json, created_at) "
            "VALUES (?, 0, ?, ?)",
            (workspace_id, state_json, _now_utc()),
        )
    pinned = store.canonical_state_at(workspace_id, 0)
    assert pinned is not None  # the INSERT OR IGNORE above just made it
    logger.info(
        "exact lane: pinned revision-0 bootstrap for %s from %d un-mirrored "
        "legacy event(s)",
        workspace_id,
        len(fold_ops),
    )
    return pinned


# ---------------------------------------------------------------------------
# T24 adaptive coalescing flag (DESIGN §710)
# ---------------------------------------------------------------------------

#: The SEPARATE flag (TASKS T15's "별도 flag와 coverage test" discipline,
#: applied to §710's debounce half): unset/empty/"0"/"false" keep the fixed
#: coalescing window byte-identical. The epoch-cadence half of §710 already
#: shipped behind its own flag (T15's ``SOLWEIG_RT_SELECTED_TIME_STREAMING``
#: adaptive epoch hint); this one governs ONLY the JobRunner's dispatch
#: debounce.
ADAPTIVE_COALESCE_ENV_FLAG = "SOLWEIG_RT_ADAPTIVE_COALESCE"

#: The latency floor the window shrinks to under the flag when no burst
#: indicator is present. Deliberately the service contract's epoch-duration
#: floor (``EPOCH_TICK_MIN_MS``, service_level_contract.md's admitted
#: 50-200 ms domain) — the same bound T15's adaptive epoch hint uses, so the
#: debounce and the epoch cadence shrink to ONE admitted floor, not two
#: invented numbers.
ADAPTIVE_COALESCE_FLOOR_MS = EPOCH_TICK_MIN_MS

_ADAPTIVE_TRUTHY = {"1", "true", "yes", "on"}


def adaptive_coalescing_enabled(
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Whether adaptive coalescing is on (read at call time, like T15)."""
    source = os.environ if environ is None else environ
    return str(source.get(ADAPTIVE_COALESCE_ENV_FLAG, "")).strip().lower() in (
        _ADAPTIVE_TRUTHY
    )


# ---------------------------------------------------------------------------
# Job runner
# ---------------------------------------------------------------------------


class JobRunner:
    """Single background worker thread executing durable jobs."""

    def __init__(
        self,
        context: RunnerContext,
        *,
        solver_factory: Callable[[RunnerContext], Solver],
        coalescing_window_ms: float = 500.0,
        adaptive_coalesce: bool = False,
        adaptive_coalesce_floor_ms: float = ADAPTIVE_COALESCE_FLOOR_MS,
        worker_revision: str = CACHE_MODEL_VERSION,
        heartbeat_interval_s: float = 5.0,
        broadcast_hub: Any = None,
        reconcile_idle_grace_s: float = 300.0,
    ) -> None:
        self.context = context
        self.solver_factory = solver_factory
        self.coalescing_window_ms = float(coalescing_window_ms)
        #: T24 adaptive debounce (DESIGN §710, flag-gated): when True the
        #: dispatch wait shrinks to ``adaptive_coalesce_floor_ms`` for an
        #: isolated job (no burst indicator ahead) instead of the full
        #: configured window. Default False = the fixed window, unchanged.
        self.adaptive_coalesce = bool(adaptive_coalesce)
        self.adaptive_coalesce_floor_ms = float(adaptive_coalesce_floor_ms)
        self.worker_revision = worker_revision
        self.heartbeat_interval_s = float(heartbeat_interval_s)
        #: Sustained subscriber-free time before a workspace is eligible
        #: for the once-daily idle reconciliation full (routing policy
        #: wave). Constructor-injectable so tests shrink it to zero.
        self.reconcile_idle_grace_s = float(reconcile_idle_grace_s)
        # SSE fan-out for the exact lane's named ``exact_revision`` event
        # (the client's verify bridge). None keeps the runner broadcast-
        # free — unit harnesses drive finalize paths without a hub.
        self.broadcast_hub = broadcast_hub
        self._queue: "queue.Queue[str | None]" = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._solver: Solver | None = None
        self._last_sweep: float | None = None
        # Heartbeat ticker (perf wave 1 STEP 1b): the worker thread only
        # heartbeats at queue-empty idle and after dispatch, so a long
        # solve (300-650 s) starves /health/ready into failingStreak 503s.
        # The daemon ticker below keeps the heartbeat fresh at the SAME
        # interval while the worker is busy, and stops promptly once the
        # worker thread is gone. The generation counter makes a stale
        # ticker from a previous start/stop cycle exit instead of serving
        # a restarted runner (the stop event is reused by design).
        self._ticker: threading.Thread | None = None
        self._ticker_stop = threading.Event()
        self._ticker_generation = 0
        #: Worker diagnostics (P9 /health/worker). Written only by the worker
        #: thread; read by the ops endpoints. Terminal job outcome counts and
        #: last-success timestamps are read durably from the store.
        self.started_monotonic = time.monotonic()
        self.started_at_utc = _now_utc()
        self.last_heartbeat_monotonic = self.started_monotonic
        self.current_job_id: str | None = None
        self.jobs_dispatched = 0
        #: Exact-lane churn guard (r2a): highest canonical revision this
        #: process already created an exact job for, per workspace. A typed
        #: refusal (``engine_refused`` on a cache the executor cannot serve,
        #: say) must not respawn on every tick — one attempt per revision.
        #: In-memory by design: after a restart one extra job may run before
        #: the guard re-arms, which the supersession/finalize guards bound.
        self._exact_attempted: dict[str, int] = {}
        #: Idle-reconcile scan throttle (monotonic timestamp of the last
        #: scan; ~15 s cadence so a quiet deployment idles cheaply). None
        #: until the first scan — a fresh runner scans immediately, never
        #: "throttled" by a fabricated epoch-zero timestamp.
        self._last_reconcile_scan: float | None = None
        #: Per workspace, when the subscriber count last dropped to zero
        #: (sustained-emptiness clock for the idle reconcile's grace). A
        #: restart clears it, delaying eligibility by at most one grace —
        #: the durable debt itself survives in ``workspace_reconcile``.
        self._empty_since: dict[str, float] = {}
        #: Per-workspace backoff timestamp for reconciliation attempts: a
        #: superseded or failed attempt retries later WITHOUT consuming
        #: the once-daily cap (only a completion does). Absent key = never
        #: attempted (an absent entry must not read as "attempted at 0").
        self._reconcile_attempted: dict[str, float] = {}
        #: Per workspace, ``(utc_day, minted_attempt_count)`` for the daily
        #: reconcile ceiling (RECONCILE_ATTEMPTS_PER_DAY): every idle
        #: reconcile MINT that does not reach a completion counts; a
        #: completion that consumes the day resets it (the workspace is
        #: proven not poisoned); a new UTC day starts fresh.
        self._reconcile_failures: dict[str, tuple[str, int]] = {}

    # -- lifecycle -----------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Recover durable nonterminal jobs, then start the worker thread."""
        if self._thread is not None:
            return
        recovered = 0
        for job in self.context.store.nonterminal_jobs():
            if job.status == "running":
                # The previous process died mid-flight: reset to queued and
                # enqueue for dispatch (forgetting the enqueue would leave the
                # durable row queued forever).
                self.context.store.requeue_running_job(job.job_id)
                self._queue.put(job.job_id)
                recovered += 1
            else:
                self._queue.put(job.job_id)
                recovered += 1
        if recovered:
            logger.info("recovered %d durable job(s) after restart", recovered)
        self._solver = self.solver_factory(self.context)
        # r2a: epochs that committed while the process was down owe exact
        # work — catch up before the first dispatch, not after the next
        # queue arrival.
        self._exact_lane_tick()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run_loop, name="solweig-job-runner", daemon=True
        )
        self._thread.start()
        self._ticker_generation += 1
        self._ticker_stop.clear()
        self._ticker = threading.Thread(
            target=self._ticker_loop,
            args=(self._ticker_generation,),
            name="solweig-heartbeat-ticker",
            daemon=True,
        )
        self._ticker.start()

    def _ticker_loop(self, generation: int) -> None:
        """Heartbeat while the worker thread is busy solving (STEP 1b).

        Pure liveness telemetry: touches nothing but
        ``last_heartbeat_monotonic``, so it can never interact with the
        physics or the job state machine. Exits as soon as the runner is
        stopped (generation bump + stop event) or the worker thread is
        gone, so a dead worker never keeps reporting healthy.
        """
        interval = max(self.heartbeat_interval_s, 0.01)
        while not self._ticker_stop.wait(interval):
            thread = self._thread
            if (
                generation != self._ticker_generation
                or thread is None
                or not thread.is_alive()
            ):
                return
            self._heartbeat()

    def stop(self, *, timeout: float = 10.0) -> None:
        self._stop.set()
        self._queue.put(None)
        # Stop the ticker FIRST and promptly: a stopped runner must not
        # keep reporting a fresh heartbeat (a dead worker would look
        # healthy for up to one interval otherwise). The generation bump
        # retires any ticker mid-wait even before the event is observed.
        self._ticker_generation += 1
        self._ticker_stop.set()
        ticker = self._ticker
        if ticker is not None and ticker.is_alive():
            ticker.join(timeout=1.0)
        self._ticker = None
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        self._thread = None

    def submit(self, job_id: str) -> None:
        """Queue one durable job, funnelling its legacy edits (r2a) and
        chasing the canonical revision (r2a exact lane).

        The funnel and the tick are best-effort by design: a failure logs
        and falls through to the plain queue put, because the durable job
        row is already the source of truth — the epoch plane catches up on
        the next tick.
        """
        try:
            self._funnel_legacy_edits(job_id)
        except Exception:  # pragma: no cover - defensive
            logger.exception("legacy edit funnel failed for job %s", job_id)
        try:
            self._exact_lane_tick()
        except Exception:  # pragma: no cover - defensive
            logger.exception("exact-lane tick failed at job %s submit", job_id)
        self._queue.put(job_id)

    # -- dispatch loop ---------------------------------------------------------

    def _run_loop(self) -> None:
        while True:
            try:
                # Idle poll cadence (r2a): epochs commit on the scheduler
                # thread, NOT through this queue, so the lane must re-check
                # the recoverable set at a short fixed cadence rather than
                # waiting out a heartbeat interval. The ticker thread keeps
                # /health/worker honest independently of this timeout.
                item = self._queue.get(
                    timeout=max(min(self.heartbeat_interval_s, 0.25), 0.05)
                )
            except queue.Empty:
                # Idle heartbeat: /health/worker distinguishes "thread alive
                # and idle" from "thread dead" via last_heartbeat_monotonic.
                self._heartbeat()
                self._exact_lane_tick()
                self._idle_reconcile_tick()
                continue
            try:
                if item is None or self._stop.is_set():
                    break
                # Pre-sleep supersede scan (perf wave 1 STEP 2b): a queued
                # job a newer edit already made redundant is aborted BEFORE
                # the coalescing delay — its solve's result would be
                # discarded, so never pay it. The scan runs again after the
                # sleep (an edit may land DURING it) inside _dispatch.
                if self._supersede_scan(item):
                    continue
                window_ms = self._effective_coalescing_window_ms(item)
                if window_ms > 0:
                    # Interruptible coalescing delay: rapid edits batch together.
                    # (§710 adaptive debounce: with the flag off this is the
                    # unchanged fixed window — see the policy method.)
                    self._stop.wait(window_ms / 1000.0)
                if self._stop.is_set():
                    break
                self._dispatch(item)
                self._heartbeat()
                self._maybe_sweep()
                # A dispatch may have consumed epochs whose successors were
                # already committed: re-check immediately (r2a).
                self._exact_lane_tick()
            except Exception:  # pragma: no cover - defensive
                logger.exception("unexpected runner error for job %s", item)
            finally:
                self._queue.task_done()

    def _heartbeat(self) -> None:
        self.last_heartbeat_monotonic = time.monotonic()

    def worker_status(self) -> dict[str, Any]:
        """/health/worker payload. Job outcome counts and the last-success
        timestamp are durable (read from the store) so they survive restarts;
        heartbeat freshness is judged by the caller (it depends on the
        readiness policy, not the runner)."""
        now = time.monotonic()
        try:
            job_counts = self.context.store.job_status_counts()
            last_success = self.context.store.latest_finished_job_at("complete")
            recovered = sum(
                1 for job in self.context.store.nonterminal_jobs()
            )
        except Exception:  # pragma: no cover - diagnostics must never raise
            logger.exception("worker status store read failed")
            job_counts, last_success, recovered = {}, None, 0
        return {
            "worker_running": self.running,
            "worker_pid": os.getpid(),
            "worker_revision": self.worker_revision,
            "started_at_utc": self.started_at_utc,
            "uptime_s": round(now - self.started_monotonic, 3),
            "current_job_id": self.current_job_id,
            "idle": self.running and self.current_job_id is None,
            "seconds_since_heartbeat": round(
                max(now - self.last_heartbeat_monotonic, 0.0), 3
            ),
            "heartbeat_interval_s": self.heartbeat_interval_s,
            "jobs_dispatched": self.jobs_dispatched,
            "job_status_counts": job_counts,
            "nonterminal_jobs": recovered,
            "last_success_at_utc": last_success,
        }

    def _maybe_sweep(self, *, interval_s: float = 3600.0) -> None:
        """Opportunistically trim ledgers, at most once per hour.

        Failures are logged and never block dispatch; the next eligible
        sweep retries on the following job.
        """
        now = time.monotonic()
        if self._last_sweep is not None and now - self._last_sweep < interval_s:
            return
        try:
            deleted = self.context.store.sweep_retention()
            if any(deleted.values()):
                logger.info("ledger sweep deleted %s", deleted)
        except Exception:  # pragma: no cover - defensive
            logger.exception("ledger retention sweep failed")
        # Stamp only after the attempt: a failed sweep should retry on the
        # next job instead of waiting out the full interval.
        self._last_sweep = now

    # -- exact lane (r2a) -----------------------------------------------------

    def _funnel_legacy_edits(self, job_id: str) -> None:
        """Mirror one job's unconsumed legacy ledger events into the op log.

        The collaborative gate: ONLY workspaces that already carry realtime
        operations are funnelled, so a pure-legacy deployment (the r1 world)
        sees no operations, no epochs, and no extra version bumps — the
        funnel is additive history for the epoch plane, never a second
        source of scene truth. Effects are NOT double-applied: the exact
        lane skips ``LEGACY_ACTOR_ID`` operations when deriving commands
        (the authoritative ``edit_events`` replay inside the same job
        carries their scene effects) — the funnelled rows are the audit
        copy and the reducer's canonical-state input.

        Idempotent by construction: ``operation_id`` is deterministic in
        the ledger event id, so a redelivered submit re-appends duplicates
        the store already deduplicates.
        """
        store = self.context.store
        job = store.get_job(job_id)
        if job is None:
            return
        request = job.request if isinstance(job.request, dict) else {}
        if request.get("exact_lane"):
            return  # the lane's own jobs are the op log's consumer, not producer
        if int(job.edit_watermark) <= int(job.base_edit_watermark):
            return  # baseline job / nothing unconsumed to mirror
        if not store.operations_since(job.scenario_id, 0):
            return  # collaborative gate: no realtime history, no funnel
        scenario = store.get_scenario(job.scenario_id)
        if scenario is None:
            return
        grid = self.context.sites.geometry(scenario.site_id)
        items: list[dict[str, Any]] = []
        for event in store.list_events(
            job.scenario_id, after_sequence=int(job.base_edit_watermark)
        ):
            if int(event["sequence"]) > int(job.edit_watermark):
                break  # ledger is ascending; a later job owns the rest
            if event["operation"] == "reset":
                # Whole-scene resets are not decomposable into per-entity
                # operations; deferred with disclosure (r2a handoff).
                continue
            item = _legacy_operation_item(event, grid)
            if item is not None:
                items.append(item)
        if items:
            store.append_operations(job.scenario_id, items)
            logger.info(
                "funnelled %d legacy edit(s) of job %s into the operation log",
                len(items),
                job.job_id,
            )

    def _exact_lane_tick(self) -> None:
        """One exact-lane scheduling pass over the recoverable epoch set.

        Called from :meth:`submit` (an edit may have closed an epoch's
        successors), the run loop's idle cadence, and :meth:`start`
        (catch-up after restart). Per-workspace failures are contained: one
        workspace's tick must not silence another's.
        """
        store = self.context.store
        by_workspace: dict[str, list[EpochRecord]] = {}
        for epoch in store.recoverable_epochs():
            by_workspace.setdefault(epoch.workspace_id, []).append(epoch)
        for workspace_id in sorted(by_workspace):
            try:
                self._exact_lane_tick_workspace(workspace_id, by_workspace[workspace_id])
            except Exception:  # pragma: no cover - defensive
                logger.exception(
                    "exact-lane tick failed for workspace %s", workspace_id
                )

    def _exact_lane_audience(self, workspace_id: str) -> bool:
        """Is anyone watching this workspace's exact progress?

        True when the broadcast hub is absent (hub-free deployments and
        the unit harnesses keep the legacy ungated behavior), when at
        least one SSE subscriber is connected, or when a nonterminal
        exact-lane job for the workspace already exists (its owner is
        owed the chase regardless of live sockets). False means a minted
        chase would spend a solve nobody is watching.
        """
        hub = self.broadcast_hub
        if hub is None:
            return True
        try:
            if hub.subscriber_count(workspace_id) > 0:
                return True
        except Exception:  # pragma: no cover - defensive
            logger.debug(
                "subscriber count unavailable for %s; treating as watched",
                workspace_id,
                exc_info=True,
            )
            return True
        for job in self.context.store.nonterminal_jobs():
            if job.scenario_id != workspace_id:
                continue
            if isinstance(job.request, dict) and job.request.get("exact_lane"):
                return True
        return False

    def _exact_lane_tick_workspace(
        self,
        workspace_id: str,
        epochs: Sequence[EpochRecord],
        *,
        reconcile: bool = False,
    ) -> str | None:
        """Create at most ONE pending exact job for ``workspace_id``.

        The lane chases the CANONICAL revision (``scenarios.scene_version``
        — epoch commits and legacy edits interleave on that counter), never
        an edit batch: every non-terminal epoch with an assigned revision
        at or below the job's target is settled by its completion. The
        span marker (``consumed_seq``/``span_hi``) tells the executor
        bridge exactly which operations the fold owes.

        Returns the minted job id, or ``None`` when this pass minted
        nothing (the idle reconcile's daily attempt ceiling counts only
        real mints — single-flight no-ops must not burn it).

        Routing policy: without ``reconcile`` the mint is gated on an
        audience (:meth:`_exact_lane_audience`) — dead sessions must not
        pile full-tile solves ahead of live users. A gated-out workspace
        keeps its recoverable epochs and gains durable reconciliation
        debt instead. With ``reconcile`` (the once-daily idle
        maintenance full) the audience gate and the churn guard are
        bypassed — the idle tick has already established emptiness and
        applies its own eligibility, cap, and backoff.
        """
        store = self.context.store
        assigned = [
            int(epoch.workspace_revision)
            for epoch in epochs
            if epoch.workspace_revision is not None
        ]
        if not assigned:
            return  # nothing committed yet: the open epoch keeps collecting
        newest_revision = max(assigned)
        records = store.epoch_records(workspace_id)
        consumed_revision = 0
        consumed_seq = 0
        for record in records:
            if record.status not in TERMINAL_EPOCH_STATUSES:
                continue
            revision = int(record.workspace_revision or 0)
            consumed_revision = max(consumed_revision, revision)
            last = int(record.last_sequence or record.first_sequence or 0)
            consumed_seq = max(consumed_seq, last)
        if newest_revision <= consumed_revision:
            return
        if not reconcile and not self._exact_lane_audience(workspace_id):
            # Subscriber-gated (routing policy wave): the chase would mint
            # a solve nobody is watching. The epochs stay recoverable and
            # the revision gap becomes durable reconciliation debt,
            # discharged by the once-daily idle reconcile (or instantly by
            # the next subscribed chase).
            store.mark_reconcile_owed(workspace_id)
            return
        # Churn guard: a typed refusal must not respawn per tick. Re-arms
        # the moment a NEWER revision is committed. The reconcile path
        # skips it: an idle full must be re-mintable at the SAME revision
        # after a superseded attempt, which is exactly the case the guard
        # exists to block.
        if (
            not reconcile
            and newest_revision <= self._exact_attempted.get(workspace_id, 0)
        ):
            return
        # Gate-open bootstrap (r2a-fix BLOCKING-1a): pin the revision-0
        # fold baseline from the legacy scene BEFORE this tick's write
        # transaction (``Store._write`` is BEGIN IMMEDIATE and not
        # nestable). Best-effort here by design — the pin is idempotent
        # and the solve path re-derives it from ``request.grid`` when the
        # tick could not, for a fresh gate (consumed_revision 0) and an
        # already-committed chain alike (r2a-fix F1), so a missing site
        # geometry (or any transient failure) must not silence the tick's
        # scheduling work.
        try:
            scenario = store.get_scenario(workspace_id)
            if scenario is not None:
                ensure_exact_lane_bootstrap(
                    store,
                    workspace_id,
                    grid=self.context.sites.geometry(scenario.site_id),
                )
        except Exception:  # pragma: no cover - defensive
            logger.debug(
                "exact-lane bootstrap deferred to solve time for %s",
                workspace_id,
                exc_info=True,
            )
        with store._write() as conn:
            row = conn.execute(
                "SELECT scene_version, edit_sequence, acked_sequence "
                "FROM scenarios WHERE scenario_id = ?",
                (workspace_id,),
            ).fetchone()
            if row is None:
                return
            target = int(row["scene_version"])
            if target <= consumed_revision:
                return
            for job_row in conn.execute(
                "SELECT status, target_scene_version, request_json FROM jobs "
                "WHERE scenario_id = ?",
                (workspace_id,),
            ).fetchall():
                try:
                    marker = (json.loads(job_row["request_json"]) or {}).get(
                        "exact_lane"
                    )
                except (TypeError, ValueError):
                    continue
                if not marker:
                    continue
                if job_row["status"] in ("queued", "running") and int(
                    job_row["target_scene_version"]
                ) >= target:
                    # Single-flight: a live exact job already owns this
                    # (or a newer) target; a strictly older live target is
                    # allowed to queue behind the running job.
                    self._exact_attempted[workspace_id] = target
                    return
            span_hi = 0
            for record in records:
                revision = record.workspace_revision
                if revision is None:
                    continue
                if consumed_revision < int(revision) <= target:
                    span_hi = max(
                        span_hi, int(record.last_sequence or record.first_sequence or 0)
                    )
            job_id = Store._insert_job_locked(
                conn,
                workspace_id,
                target_scene_version=target,
                request={
                    "exact_lane": {
                        "target_revision": target,
                        "consumed_revision": consumed_revision,
                        "consumed_seq": consumed_seq,
                        "span_hi": span_hi,
                        **({"reconcile": True} if reconcile else {}),
                    }
                },
                edit_watermark=int(row["edit_sequence"]),
                base_edit_watermark=int(row["acked_sequence"]),
            )
            Store._supersede_queued_locked(conn, workspace_id, except_job_id=job_id)
        self._exact_attempted[workspace_id] = target
        self._queue.put(job_id)
        logger.info(
            "exact lane: job %s targets revision %d (span %d..%d) for %s",
            job_id,
            target,
            consumed_seq + 1,
            span_hi,
            workspace_id,
        )
        return job_id

    def _idle_reconcile_tick(self, now_monotonic: float | None = None) -> None:
        """Once-daily maintenance full for subscriber-free workspaces (R4).

        Runs only from the run loop's queue-empty idle cadence (right
        after the exact-lane tick) — a busy worker is never idle by
        definition. The scan itself is throttled (~15 s) and every
        per-workspace decision — sustained emptiness, the once-daily cap,
        chain depth, attempt backoff — lives in
        :meth:`_idle_reconcile_workspace`. Per-workspace failures are
        contained like the exact-lane tick's.
        """
        if self.current_job_id is not None or not self._queue.empty():
            return
        now = time.monotonic() if now_monotonic is None else float(now_monotonic)
        if (
            self._last_reconcile_scan is not None
            and now - self._last_reconcile_scan < 15.0
        ):
            return
        self._last_reconcile_scan = now
        store = self.context.store
        candidates: dict[str, list[EpochRecord]] = {}
        for epoch in store.recoverable_epochs():
            candidates.setdefault(epoch.workspace_id, []).append(epoch)
        for workspace_id in store.workspaces_owing_reconcile():
            candidates.setdefault(workspace_id, [])
        today = _now_utc()[:10]
        for workspace_id in sorted(candidates):
            try:
                self._idle_reconcile_workspace(
                    workspace_id, candidates[workspace_id], today, now
                )
            except Exception:  # pragma: no cover - defensive
                logger.exception(
                    "idle reconcile scan failed for workspace %s", workspace_id
                )

    def _idle_reconcile_workspace(
        self,
        workspace_id: str,
        epochs: Sequence[EpochRecord],
        today: str,
        now: float,
    ) -> None:
        """Mint ONE full-tile reconcile for a quiet, indebted workspace.

        Eligibility: nobody watching (the emptiness clock has run past the
        grace, or the patch chain since the last completed full is deeper
        than ~50 revisions while quiet), no completed reconcile today
        (only a COMPLETION consumes the day; superseded/failed attempts
        retry on the 600 s backoff), at most
        :data:`RECONCILE_ATTEMPTS_PER_DAY` minted attempts today without a
        completion, and a recoverable revision gap the reconcile can
        actually fold. The mint reuses the exact lane with
        ``reconcile=True`` so the executor bridge routes it a forced full
        (:data:`RECONCILE_FULL_RECOMPUTE_FRACTION`).
        """
        store = self.context.store
        if self._exact_lane_audience(workspace_id):
            # Someone is watching again: the emptiness clock resets and the
            # regular gated chase takes over on its own ticks.
            self._empty_since.pop(workspace_id, None)
            return
        empty_since = self._empty_since.get(workspace_id)
        if empty_since is None:
            self._empty_since[workspace_id] = now
            return
        last_completed, _owed = store.reconcile_state(workspace_id)
        if last_completed == today:
            return
        failures_day, failures = self._reconcile_failures.get(workspace_id, ("", 0))
        if failures_day == today and failures >= RECONCILE_ATTEMPTS_PER_DAY:
            # Daily attempt ceiling (routing-policy review follow-up): a
            # crash-class poison request must not loop a full solve
            # continuously. Held until the next UTC day; the hold was
            # logged ONCE when the ceiling was reached, not per scan.
            return
        attempted_at = self._reconcile_attempted.get(workspace_id)
        if attempted_at is not None and now - attempted_at < 600.0:
            return
        scenario = store.get_scenario(workspace_id)
        if scenario is None:
            return
        eligible = (now - empty_since) >= self.reconcile_idle_grace_s
        if not eligible:
            # Secondary trigger: a deep revision chain since the last
            # completed full, still while quiet — incremental accuracy
            # degrades with chain depth even when every window is small.
            last_full = store.last_full_result_version(workspace_id)
            if last_full is not None and scenario.scene_version - last_full > 50:
                eligible = True
        if not eligible:
            return
        self._reconcile_attempted[workspace_id] = now
        minted = self._exact_lane_tick_workspace(workspace_id, epochs, reconcile=True)
        if minted is None:
            return  # single-flight no-op: no attempt was spent
        attempts = (failures + 1) if failures_day == today else 1
        self._reconcile_failures[workspace_id] = (today, attempts)
        if attempts >= RECONCILE_ATTEMPTS_PER_DAY:
            logger.info(
                "idle reconcile: workspace %s reached its attempt ceiling "
                "(%d minted attempt(s) today without a completion); holding "
                "further reconcile mints until the next UTC day",
                workspace_id,
                attempts,
            )

    def _consume_exact_epochs(self, workspace_id: str, through_revision: int) -> None:
        """Settle every assigned epoch at or below the published revision.

        ``exact_targeted`` is terminal: the epoch leaves the recoverable
        set and the scheduler never re-drives it. Epochs above the
        published revision stay untouched — they belong to the next exact
        job's span.
        """
        store = self.context.store
        for epoch in store.epoch_records(workspace_id):
            if epoch.status in TERMINAL_EPOCH_STATUSES:
                continue
            revision = epoch.workspace_revision
            if revision is None or int(revision) > int(through_revision):
                continue
            store.mark_epoch_status(
                workspace_id,
                epoch.epoch_id,
                "exact_targeted",
                workspace_revision=int(revision),
            )

    def _finish_job(self, job_id: str, status: str, **kwargs: Any) -> None:
        """Terminal-state chokepoint: finish durably, then emit progress.

        Every JobRunner terminal transition funnels here so the SSE
        ``exact_progress`` plane always observes the ending (a cancelled or
        superseded job must retire a client's countdown, not strand it).
        The store call itself is unchanged.
        """
        self.context.store.finish_job(job_id, status, **kwargs)
        self._emit_exact_progress(job_id)

    def _emit_exact_progress(self, job_id: str) -> None:
        """Broadcast one additive ``exact_progress`` SSE event, if warranted.

        Emitted for exact-lane jobs only (the native realtime plane's live
        chase): legacy clients own their job ids and poll
        ``GET /jobs/{id}``, whose eta fields remain the source of truth.
        Transitions: dispatch (running), mode-known (eta becomes possible),
        terminal (complete/failed/superseded — via :meth:`_finish_job`).
        Never raises: progress telemetry must not fail a solve or a settle.
        """
        hub = self.broadcast_hub
        if hub is None:
            return
        try:
            store = self.context.store
            job = store.get_job(job_id)
            if job is None:
                return
            request = job.request if isinstance(job.request, dict) else {}
            if not request.get("exact_lane"):
                return
            eta_seconds, eta_basis = job_eta_fields(store, self.context.sites, job)
            payload: dict[str, Any] = {
                "job_id": job.job_id,
                "status": job.status,
                "target_revision": int(job.target_scene_version),
                "eta_seconds": eta_seconds,
                "eta_basis": eta_basis,
            }
            if job.status == "queued":
                position = store.queue_position(job.job_id)
                if position is not None:
                    payload["queue_position"] = int(position)
            hub.broadcast_exact_progress(job.scenario_id, payload)
        except Exception:  # noqa: BLE001 - telemetry must never block the lane
            logger.exception(
                "exact_progress event failed for job %s", job_id
            )

    def _broadcast_exact_revision(
        self, workspace_id: str, exact_revision: int
    ) -> None:
        """Emit the named ``exact_revision`` SSE event (verify bridge).

        The studio's verify stage (badge → Exact, exact-result fetch) rides
        this EVENT, not the ``exact_revision`` field fast frames carry: a
        field reconciles the revision triple, but nothing tells the client
        the exact lane just published. Emitted only when a result actually
        EXISTS at the settled revision — failed/superseded settles advance
        no exact state and must never claim exactness. Idempotent by
        content: a crash between publish and terminal finish re-runs the
        settle and simply re-emits the same revisions (clients take the
        max, exactly like a snapshot frame).
        """
        if self.broadcast_hub is None:
            return
        store = self.context.store
        try:
            # Existence gate INSIDE the fence: a store read that raises must
            # skip the event, never escape into the settle path (a raise here
            # would strand the job row "running" until a process restart).
            if store.get_result(workspace_id, int(exact_revision)) is None:
                return
            workspace_revision = int(store.current_scene_version(workspace_id))
        except Exception:  # noqa: BLE001 - the event must never block settle
            logger.exception(
                "exact_revision event: workspace revision unreadable for %s",
                workspace_id,
            )
            workspace_revision = int(exact_revision)
        try:
            self.broadcast_hub.broadcast_exact(
                workspace_id,
                {
                    "workspace_id": workspace_id,
                    "exact_revision": int(exact_revision),
                    "workspace_revision": workspace_revision,
                },
            )
        except Exception:  # noqa: BLE001 - the fence must never kill settlement
            logger.exception(
                "exact_revision event failed for %s (result stays published)",
                workspace_id,
            )
        # T15 (flag ON only): the coverage frame rides the exact
        # publication — a selected-time streaming client learns WHICH
        # times the freshly published revision serves bitwise-correctly
        # instead of trusting the revision number alone. Same fence: a
        # failure logs, never blocks the settle. getattr-guarded because
        # test hubs implement only the events they assert on.
        if not selected_time.enabled():
            return
        broadcast_selected = getattr(
            self.broadcast_hub, "broadcast_selected_time", None
        )
        if broadcast_selected is None:
            return
        try:
            scenario = store.get_scenario(workspace_id)
            if scenario is None:
                return
            geometry = self.context.sites.geometry(scenario.site_id)
            time_steps = int(geometry["time_steps"])
            report = selected_time.coverage_report(
                store, workspace_id, ("utci", "tmrt"), time_steps
            )
            broadcast_selected(
                workspace_id,
                selected_time.initial_frame(report, list(range(time_steps))),
            )
        except Exception:  # noqa: BLE001 - telemetry must never kill settlement
            logger.exception(
                "selected_time coverage frame failed for %s "
                "(exact result stays published)",
                workspace_id,
            )

    def _supersede_scan(self, job_id: str) -> bool:
        """Mark a redundant queued job superseded; report whether to abort.

        Shared by the run loop's pre-sleep check and :meth:`_dispatch`
        (STEP 2b): True means "do not dispatch this job" — it is already
        gone/cancelled, or a strictly newer queued job for the same
        scenario made it redundant.
        """
        store = self.context.store
        job = store.get_job(job_id)
        if job is None or job.status != "queued":
            return True  # superseded or cancelled while queued
        # Revision chasing (r2a): an exact-lane job targets the CANONICAL
        # revision, not an edit batch, so "a newer edit queued behind me"
        # (the legacy test below) is irrelevant to it — what makes it stale
        # is the workspace revision itself moving past its target. The
        # tick already dropped such rows; this is the safe boundary that
        # also catches the race between tick and dispatch.
        request = job.request if isinstance(job.request, dict) else {}
        if request.get("exact_lane"):
            current = int(store.current_scene_version(job.scenario_id))
            if current > int(job.target_scene_version):
                self._finish_job(
                    job.job_id,
                    "superseded",
                    error={
                        "code": "superseded",
                        "message": (
                            f"canonical revision advanced to {current} past "
                            f"exact target {job.target_scene_version}; a newer "
                            "exact job owns this span"
                        ),
                    },
                )
                return True
            return False
        for other in store.nonterminal_jobs():
            if (
                other.job_id != job.job_id
                and other.scenario_id == job.scenario_id
                and other.status == "queued"
                and other.target_scene_version > job.target_scene_version
            ):
                self._finish_job(
                    job.job_id,
                    "superseded",
                    error={
                        "code": "superseded",
                        "message": "a newer edit superseded this queued job",
                    },
                )
                return True
        return False

    def _effective_coalescing_window_ms(self, job_id: str) -> float:
        """The dispatch wait for this job (§710 adaptive debounce, T24).

        The coalescing wait's ONLY role is to give newer work a chance to
        make this queued job redundant before its solve is paid for (the
        pre/post-sleep supersede scans are the layers that act on it). No
        acceptance, folding, conflict-resolution, or publication path reads
        the window — that is what makes any wait duration semantics-free,
        which is §710's invariant: "Epoch를 adaptive하게 줄이더라도 reducer
        determinism, idempotency, conflict resolution은 변하지 않아야 한다".

        Flag off (the default): the configured window, exactly as before —
        one store-free attribute read, byte-identical behavior. Flag on: the
        window shrinks to ``adaptive_coalesce_floor_ms`` (the contract's
        50 ms latency floor) ONLY when no burst indicator is present; any
        indicator keeps the full configured window, so rapid-edit batching
        is preserved exactly where the fixed window earns its keep.
        """
        if not self.adaptive_coalesce or self.coalescing_window_ms <= 0:
            return self.coalescing_window_ms
        if self._coalesce_burst_ahead(job_id):
            return self.coalescing_window_ms
        return min(self.adaptive_coalesce_floor_ms, self.coalescing_window_ms)

    def _coalesce_burst_ahead(self, job_id: str) -> bool:
        """Whether an edit burst looks in progress for this dispatch.

        Durable-signal only (no wall-clock heuristics feed the decision):

        * work is already queued behind this job (any lane);
        * a nonterminal peer job exists for the same scenario — a running
          single-flight exact chase or a second queued legacy edit;
        * an epoch is OPEN for the workspace: an operation was accepted
          within the last epoch window (or the closer is down, in which
          case staying conservative is the honest fallback).

        A missing job row reads as a burst (the conservative full window —
        the post-sleep scan aborts the dispatch anyway).
        """
        if self._queue.qsize() > 0:
            return True
        store = self.context.store
        job = store.get_job(job_id)
        if job is None:
            return True
        for other in store.nonterminal_jobs():
            if other.job_id != job.job_id and other.scenario_id == job.scenario_id:
                return True
        try:
            return bool(store.open_epochs(job.scenario_id))
        except Exception:  # pragma: no cover - defensive: never break dispatch
            logger.exception("open-epoch probe failed; using full window")
            return True

    def _dispatch(self, job_id: str) -> None:
        store = self.context.store
        job = store.get_job(job_id)
        if job is None or job.status != "queued":
            return  # superseded or cancelled while queued
        scenario = store.get_scenario(job.scenario_id)
        if scenario is None:
            self._finish_job(
                job_id,
                "failed",
                error={"code": "scenario_not_found", "message": "scenario disappeared"},
            )
            return
        # Post-sleep scan: an edit that arrived DURING the coalescing delay
        # supersedes this job before its solve is paid for.
        if self._supersede_scan(job_id):
            return
        self._execute(job, scenario)

    def _execute(self, job, scenario: ScenarioRecord) -> None:
        store = self.context.store
        assert self._solver is not None
        grid = self.context.sites.geometry(scenario.site_id)
        job_request = job.request if isinstance(job.request, dict) else {}
        exact_lane = job_request.get("exact_lane")
        request = SolveRequest(
            job_id=job.job_id,
            scenario_id=scenario.scenario_id,
            site_id=scenario.site_id,
            target_scene_version=job.target_scene_version,
            edit_watermark=job.edit_watermark,
            base_edit_watermark=job.base_edit_watermark,
            trees=tuple(store.list_trees(scenario.scenario_id)),
            events=tuple(store.list_events(scenario.scenario_id)),
            requested=job.request or {},
            grid=grid,
            exact_lane=dict(exact_lane) if exact_lane else None,
        )
        store.mark_job_running(job.job_id)
        self.current_job_id = job.job_id
        self.jobs_dispatched += 1
        self._heartbeat()
        self._emit_exact_progress(job.job_id)

        mode_emitted = {"value": False}

        def progress(
            stage: str,
            completed: int,
            total: int,
            *,
            mode: str | None = None,
            window: RasterWindow | Mapping[str, int] | None = None,
        ) -> None:
            self._heartbeat()  # a busy worker is alive even mid-solve
            window_dict = (
                _window_dict(window)
                if isinstance(window, RasterWindow)
                else (dict(window) if window is not None else None)
            )
            store.update_job_progress(
                job.job_id,
                stage=stage,
                progress={
                    "completed_time_steps": int(completed),
                    "total_time_steps": int(total),
                },
                mode=mode,
                window=window_dict,
            )
            # Mode-known transition: the eta basis exists only once the mode
            # is on the durable row (executor-path jobs report it late), so
            # this is the moment a subscriber's countdown becomes possible.
            if mode is not None and not mode_emitted["value"]:
                mode_emitted["value"] = True
                self._emit_exact_progress(job.job_id)

        started = time.perf_counter()
        try:
            result = self._solver(request, progress)
        except Exception as error:  # solver failure marks the job failed
            logger.exception("solver failed for job %s", job.job_id)
            self._finish_job(
                job.job_id,
                "failed",
                error={"code": "job_failed", "message": f"{type(error).__name__}: {error}"},
                worker_revision=self.worker_revision,
            )
            return
        finally:
            self.current_job_id = None
        duration_ms = (time.perf_counter() - started) * 1000.0
        self._finalize(job, scenario, result, duration_ms=duration_ms)

    def _finalize(
        self,
        job,
        scenario: ScenarioRecord,
        result: SolveResult,
        *,
        duration_ms: float,
    ) -> None:
        store = self.context.store
        window_dict = _window_dict(result.window) if result.window is not None else None
        exact_lane = (
            job.request.get("exact_lane")
            if isinstance(job.request, dict)
            else None
        )

        def consume_exact_epochs() -> None:
            """Advance every epoch the published revision covers (r2a).

            Runs BEFORE the job's terminal finish so a crash between the
            two leaves the epoch ``reducing`` — recoverable — rather than
            terminal-but-unrecorded; the re-run's idempotent completion
            path finishes the consumption. Legacy jobs carry no marker and
            never consume (the epoch plane is the exact lane's to settle).
            """
            if exact_lane is not None:
                self._consume_exact_epochs(
                    job.scenario_id, int(job.target_scene_version)
                )
                self._broadcast_exact_revision(
                    job.scenario_id, int(job.target_scene_version)
                )

        def mark_reconcile_settlement(mode: str | None) -> None:
            """R4 bookkeeping, called ONLY after a ``complete`` finish.

            A completed reconcile marker — or any completed exact job that
            actually ran FULL — owns the day: the durable cap is consumed
            and the debt discharged. A LOCAL (or no-op) completion settled
            the revision gap incrementally, so the debt is paid without
            consuming the once-daily full. Superseded and failed attempts
            touch nothing (their callers never reach here).
            """
            if exact_lane is None:
                return
            if exact_lane.get("reconcile") or mode == "full":
                store.mark_reconcile_completed(job.scenario_id, _now_utc()[:10])
                # The day is consumed AND the workspace is proven not
                # poisoned: tomorrow's attempts start from a fresh ledger.
                self._reconcile_failures.pop(job.scenario_id, None)
            else:
                store.clear_reconcile_owed(job.scenario_id)

        if result.status == "published":
            live = store.get_job(job.job_id)
            if live is None or live.status != "running":
                # Cancelled mid-flight: the client asked to stop, so the
                # result is discarded rather than published.
                logger.info(
                    "job %s left status 'running' (%s) before finalize; discarding result",
                    job.job_id,
                    live.status if live is not None else "deleted",
                )
                return
            current = store.current_scene_version(scenario.scenario_id)
            if int(result.scene_version) != int(current) or int(result.scene_version) != int(
                job.target_scene_version
            ):
                # Scientific-integrity guard: never publish a stale result.
                self._finish_job(
                    job.job_id,
                    "superseded",
                    mode=result.mode,
                    window=window_dict,
                    error={
                        "code": "superseded",
                        "message": (
                            f"result for scene version {result.scene_version} arrived "
                            f"after the scenario moved to version {current}; not published"
                        ),
                    },
                    worker_revision=self.worker_revision,
                )
                return
            try:
                manifest, payload = materialize_result(
                    self.context,
                    scenario,
                    result,
                    duration_ms=duration_ms,
                )
                # T15: the publication rides a fenced lease (T13
                # OutputLease's CPU analogue) — mint from the LIVE row,
                # mark completion, publish at most once. The store's own
                # fences (version staleness, in-transaction running
                # check, idempotent-complete) remain authoritative and
                # surface unchanged below.
                live_row = store.get_job(job.job_id)
                if live_row is None or live_row.status != "running":
                    raise publish_fence.LeaseStaleError(
                        job.job_id,
                        live_row.status if live_row is not None else "deleted",
                    )
                lease = publish_fence.ResultPublishLease.from_job(store, live_row)
                lease.mark_complete()
                lease.publish(
                    manifest=manifest,
                    payload=payload,
                    acked_sequence=job.edit_watermark,
                )
            except StaleResultError:
                self._finish_job(
                    job.job_id,
                    "superseded",
                    mode=result.mode,
                    window=window_dict,
                    error={"code": "superseded", "message": "scene advanced before publish"},
                    worker_revision=self.worker_revision,
                )
                return
            except (ResultNotPublishable, publish_fence.LeaseStaleError) as error:
                # A cancel committed between the liveness check above and the
                # publish transaction: the result is a discard by contract,
                # and the job row already carries its cancelled outcome.
                logger.info(
                    "job %s: discarding result for scene version %s (%s)",
                    job.job_id,
                    result.scene_version,
                    error,
                )
                return
            except ResultAlreadyPublished:
                # Crash window between publish_result and finish_job: the
                # durable row says running, the re-run reproduced the same
                # result. Deterministic solvers make this an idempotent
                # success when (and only when) the bytes match.
                existing = store.get_result(
                    scenario.scenario_id, int(result.scene_version)
                )
                if existing is not None and existing.checksum == manifest.get("checksum"):
                    logger.info(
                        "job %s re-published identical result for scene version %s; "
                        "treating as idempotent completion",
                        job.job_id,
                        result.scene_version,
                    )
                    consume_exact_epochs()
                    self._finish_job(
                        job.job_id,
                        "complete",
                        mode=result.mode,
                        window=window_dict,
                        metrics=manifest["metrics"],
                        result_scene_version=int(result.scene_version),
                        worker_revision=self.worker_revision,
                        plan=result.plan,
                    )
                    mark_reconcile_settlement(result.mode)
                else:
                    self._finish_job(
                        job.job_id,
                        "failed",
                        mode=result.mode,
                        window=window_dict,
                        error={
                            "code": "internal_error",
                            "message": (
                                f"scene version {result.scene_version} is already "
                                "published with different content; refusing to "
                                "overwrite"
                            ),
                        },
                        worker_revision=self.worker_revision,
                    )
                return
            consume_exact_epochs()
            self._finish_job(
                job.job_id,
                "complete",
                mode=result.mode,
                window=window_dict,
                metrics=manifest["metrics"],
                result_scene_version=int(result.scene_version),
                worker_revision=self.worker_revision,
                plan=result.plan,
            )
            mark_reconcile_settlement(result.mode)
            return
        if result.status == "no-op":
            # Coalesced to no semantic change: re-serve the latest exact
            # state under the new version without recomputation. Every
            # exit SETTLES the epochs first (r2a-fix LOW-1): an unsettled
            # epoch plus the tick's churn guard (one attempt per revision)
            # would wedge the lane at this revision forever — there would
            # be no later job left to consume it.
            noop_metrics = {
                "duration_ms": round(duration_ms, 3),
                "mode": "no-op",
                **dict(result.metrics or {}),
            }
            latest = store.latest_result(scenario.scenario_id)
            if latest is None:
                consume_exact_epochs()
                self._finish_job(
                    job.job_id,
                    "failed",
                    error={
                        "code": "job_failed",
                        "message": "no-op job has no earlier result to re-serve",
                    },
                    worker_revision=self.worker_revision,
                )
                return
            if latest.scene_version == job.target_scene_version:
                # Crash window between the re-serve publish and this
                # finish: the target version's result already exists —
                # idempotent completion (the ResultAlreadyPublished
                # semantics, surfaced without re-paying the publish).
                consume_exact_epochs()
                self._finish_job(
                    job.job_id,
                    "complete",
                    mode="no-op",
                    metrics=noop_metrics,
                    result_scene_version=job.target_scene_version,
                    worker_revision=self.worker_revision,
                )
                mark_reconcile_settlement("no-op")
                return
            # r2a-fix MEDIUM-1: when the lane folded realtime families it
            # could not map onto executor commands, the RE-SERVED manifest
            # must disclose it. ``republish_result_at_version`` merges the
            # disclosure into the republished manifest via
            # ``manifest_override`` (store seam): payload bytes and the
            # checksum stay the producing version's — this re-serve did no
            # work, so their provenance is unchanged.
            lane_metrics = (result.metrics or {}).get("exact_lane") or {}
            skipped = lane_metrics.get("families_skipped") or []
            try:
                store.republish_result_at_version(
                    scenario.scenario_id,
                    latest.scene_version,
                    job.target_scene_version,
                    job_id=job.job_id,
                    manifest_override=(
                        {
                            "limitations": [
                                exact_lane_skip_disclosure(
                                    skipped, int(job.target_scene_version)
                                )
                            ]
                        }
                        if skipped
                        else None
                    ),
                )
            except ResultAlreadyPublished:
                # Idempotent re-drive: the publish committed, the finish
                # did not. The epochs are owed their settlement either way.
                consume_exact_epochs()
                self._finish_job(
                    job.job_id,
                    "complete",
                    mode="no-op",
                    metrics=noop_metrics,
                    result_scene_version=job.target_scene_version,
                    worker_revision=self.worker_revision,
                )
                mark_reconcile_settlement("no-op")
                return
            except (StoreError, StaleResultError) as error:
                consume_exact_epochs()
                self._finish_job(
                    job.job_id,
                    "superseded",
                    error={"code": "superseded", "message": str(error)},
                    worker_revision=self.worker_revision,
                )
                return
            consume_exact_epochs()
            self._finish_job(
                job.job_id,
                "complete",
                mode="no-op",
                metrics=noop_metrics,
                result_scene_version=job.target_scene_version,
                worker_revision=self.worker_revision,
            )
            mark_reconcile_settlement("no-op")
            return
        if (
            exact_lane is not None
            and result.error
            and result.error.get("code") in ("edit_rejected", "engine_refused")
        ):
            # r2a-fix BLOCKING-1b anti-wedge (incident-3 review BLOCKER 3:
            # extended to engine_refused): a typed refusal on the exact
            # lane is DETERMINISTIC — the same fold refuses the same way,
            # whether the batch was refused as an edit (edit_rejected) or
            # by the engine's own machinery (engine_refused:
            # ExecutorError/SolverInputError — an empty-batch contract
            # refusal, an inconsistent worker state; never a crash-class
            # exception, which keeps the reconcile debt below) — and the
            # tick's churn guard blocks a retry at the same canonical
            # revision. Falling through to the generic superseded finish
            # would leave the span's epochs ``reducing`` forever (the
            # wedge). Settle the span (``exact_targeted``) and record the
            # refusal as a loud FAILED job; the next canonical revision
            # re-arms the lane.
            #
            # Disclosure: SolverInputError also covers filesystem-missing
            # conditions (site building raster missing, met source file
            # not found) — with inputs on live-synced storage, a transient
            # missing input now SETTLES the span (one skipped publication)
            # instead of wedging, and heals on the next revision.
            consume_exact_epochs()
            self._finish_job(
                job.job_id,
                "failed",
                mode=result.mode,
                window=window_dict,
                metrics=dict(result.metrics) if result.metrics else None,
                error=dict(result.error),
                worker_revision=self.worker_revision,
            )
            # The refusal consumed the span's epochs, so the revision gap
            # the reconcile debt exists to fold is gone: clear the owed
            # flag here (this path never reaches mark_reconcile_settlement)
            # or it lingers forever as idle-scan candidate noise. Crash-
            # class failures (the exception path in _execute) keep the
            # debt — their retry is wanted.
            store.clear_reconcile_owed(job.scenario_id)
            return
        self._finish_job(
            job.job_id,
            "superseded",
            mode=result.mode,
            window=window_dict,
            error=dict(result.error) if result.error else None,
            worker_revision=self.worker_revision,
        )


# ---------------------------------------------------------------------------
# Default solver: ExactWorker adapter
# ---------------------------------------------------------------------------


def api_tree_to_spec(tree: Mapping[str, Any], grid: Mapping[str, float | int]) -> TreeSpec:
    """Convert an API UV tree object into a world-coordinate ``TreeSpec``.

    UV-to-world conversion runs through the site manifest geometry — never
    through frontend constants.
    """
    x_m, y_m = uv_to_world(
        float(tree["u"]),
        float(tree["v"]),
        rows=int(grid["rows"]),
        cols=int(grid["cols"]),
        pixel_size_m=float(grid["pixel_size_m"]),
        origin_x_m=float(grid["origin_x_m"]),
        origin_y_m=float(grid["origin_y_m"]),
    )
    return TreeSpec(
        tree_id=str(tree["tree_id"]),
        x_m=x_m,
        y_m=y_m,
        height_m=float(tree["height_m"]),
        canopy_radius_m=float(tree["canopy_diameter_m"]) / 2.0,
        trunk_ratio=float(tree.get("trunk_ratio", 0.25)),
        transmissivity=float(tree.get("transmissivity", 0.03)),
    )


def _replay_event(
    layer: TreeLayer, event: Mapping[str, Any], grid_info: Mapping[str, float | int]
) -> int:
    """Apply one ledger event to ``layer``; return how many layer edits it
    produced (``move``/``update`` fan out into two layer edits, ``reset``
    into one per live tree), which keeps the layer's internal sequence
    numbering aligned with store watermarks."""
    if event.get("family"):
        # Universal (family) events carry no tree payload; they route the
        # scenario onto the executor bridge, which never calls this.
        return 0
    if event["operation"] == "reset":
        count = 0
        for tree in layer.current_trees():
            layer.delete_tree(tree.tree_id)
            count += 1
        return count
    new_tree = event.get("new_tree")
    if new_tree is None:
        layer.delete_tree(event["tree_id"])
        return 1
    spec = api_tree_to_spec(new_tree, grid_info)
    existing = next(
        (t for t in layer.current_trees() if t.tree_id == spec.tree_id), None
    )
    if existing is None:
        layer.add_tree(spec)
        return 1
    layer.move_tree(spec.tree_id, x_m=spec.x_m, y_m=spec.y_m)
    layer.update_tree(
        spec.tree_id,
        height_m=spec.height_m,
        canopy_radius_m=spec.canopy_radius_m,
        trunk_ratio=spec.trunk_ratio,
        transmissivity=spec.transmissivity,
    )
    return 2


def make_exact_worker_solver(
    *,
    requested_variables: Sequence[str] = ("utci", "tmrt", "shadow"),
) -> Callable[[RunnerContext], Solver]:
    """Default solver factory adapting the Phase-5 :class:`ExactWorker`.

    The adapter replays the durable edit ledger into a fresh
    :class:`TreeLayer` and primes the worker's consumed-edit watermark from
    the job's *base* watermark, so the worker's pending batch is exactly this
    job's edits — plus any unconsumed edits inherited from superseded jobs.
    (Priming from ``edit_watermark`` instead would leave the batch empty and
    every edit job would no-op.) It runs the worker against the shared site
    cache with a revision provider bound to the store's current scene
    version (so the worker's cooperative supersession checkpoints trip on
    concurrent edits). Published worker patches are composed with the
    baseline and prior published results into one API patch covering the
    union write window. When no baseline result exists at all (bootstrap),
    the adapter performs one standard full-tile solve of the current scene.
    """

    def factory(context: RunnerContext) -> Solver:
        def solve(request: SolveRequest, progress: ProgressCallback) -> SolveResult:
            site = context.sites.config(request.site_id)
            cache = context.sites.cache(request.site_id)
            grid_info = context.sites.geometry(request.site_id)
            grid = RasterGrid(
                rows=int(grid_info["rows"]),
                cols=int(grid_info["cols"]),
                pixel_size_m=float(grid_info["pixel_size_m"]),
                origin_x_m=float(grid_info["origin_x_m"]),
                origin_y_m=float(grid_info["origin_y_m"]),
            )
            layer = TreeLayer(
                np.asarray(cache.tree_base), grid, scenario_id=request.scenario_id
            )
            # Replay the ledger, counting layer edits consumed by published
            # results: the worker's pending batch becomes exactly the edits
            # in (base_edit_watermark, edit_watermark].
            acked_layer_edits = 0
            for event in request.events:
                produced = _replay_event(layer, event, grid_info)
                if int(event["sequence"]) <= int(request.base_edit_watermark):
                    acked_layer_edits += produced
            worker = ExactWorker(
                cache,
                layer,
                site_dir=site.site_dir or site.cache_dir,
                results_root=context.store.results_root / request.scenario_id / "worker",
                selected_date_str=site.selected_date_str,
                requested_variables=tuple(requested_variables),
                revision_provider=lambda: context.store.current_scene_version(
                    request.scenario_id
                ),
            )
            worker._acked_sequence = acked_layer_edits

            total_time_steps = int(cache.time_steps)
            time_indices, variables = resolve_requested(request.requested, grid_info)
            # Perf wave 1 STEP 2a: an explicit time_indices request bounds
            # the local solve at its causal prefix t = 0..max(index) — every
            # computed step stays bit-identical to the full-series run (the
            # temporal state chains only carry forward). Full-day requests
            # (``refine_full_day``, no explicit indices) keep None.
            time_stop = resolve_time_stop(request.requested, grid_info)
            variables = tuple(v for v in variables if v in requested_variables) or (
                requested_variables[0],
            )

            if not context.store.result_versions(request.scenario_id):
                # Bootstrap: no baseline result exists to compose against, so
                # run one standard full-tile solve of the (possibly edited)
                # scene. This covers both fresh scenarios whose site cache has
                # no stored baseline outputs and first jobs racing ahead of a
                # superseded baseline job.
                progress("windowing", 0, 1)
                forcing = worker.forcing()
                full_arrays = run_full_tile(
                    cache,
                    layer,
                    forcing=forcing,
                    site_dir=worker.site_dir,
                    scratch_dir=worker.scratch_root / request.job_id,
                    requested_variables=tuple(requested_variables),
                )
                window = grid.full_window
                progress(
                    "time_loop", total_time_steps, total_time_steps,
                    mode="full", window=_window_dict(window),
                )
                progress("publishing", 0, 1)
                return SolveResult(
                    status="published",
                    scene_version=request.target_scene_version,
                    mode="full",
                    window=window,
                    time_indices=time_indices,
                    variables=variables,
                    arrays={
                        name: np.ascontiguousarray(
                            full_arrays[name][list(time_indices)], dtype=np.float32
                        )
                        for name in variables
                    },
                    metrics={
                        "fallback_reason": None,
                        "window_fraction": 1.0,
                        "read_window_fraction": 1.0,
                        "bootstrap": True,
                        "dirty_windows": [],
                    },
                    # T15 cut site (bootstrap): the full tile was computed,
                    # then the payload was cut to the REQUESTED times — the
                    # unrequested planes are discarded, not durable.
                    requested_time_subset=len(time_indices) < total_time_steps,
                )

            progress("windowing", 0, 1)
            outcome = worker.run(
                job_id=request.job_id,
                target_revision=request.target_scene_version,
                time_stop=time_stop,
            )
            if outcome.status != "published":
                return SolveResult(
                    status=outcome.status,
                    scene_version=request.target_scene_version,
                    metrics={"mode": outcome.status},
                )
            patches = [load_patch(path) for path in outcome.patch_paths]
            union = patches[0].write_window
            for patch in patches[1:]:
                union = union.union(patch.write_window)
            # The solve loop itself runs every site time step inside
            # ``worker.run`` (no per-step hook exists there), so report it in
            # the contract's time_loop shape once it completed: mode and
            # window are known only after the worker returns.
            progress(
                "time_loop", total_time_steps, total_time_steps,
                mode=outcome.mode, window=_window_dict(union),
            )
            progress("publishing", 0, 1)
            try:
                state = _compose_current_state(
                    context,
                    request.scenario_id,
                    grid_info,
                    variables,
                    union,
                    request.target_scene_version,
                    extra_patches=patches,
                )
            except selected_time.TimeCoverageGap:
                # T15 heal (witness: accepted operation loss): the durable
                # history carries an UNHEALED request-cut record, so the
                # incremental compose would silently keep the cut revision's
                # uncovered times at their stale values. Recompute the full
                # tile and publish FULL coverage (full site window, every
                # time step) — this publication heals the gap and the scene
                # is exact again, bitwise.
                progress("time_loop", total_time_steps, total_time_steps, mode="full")
                forcing = worker.forcing()
                full_arrays = run_full_tile(
                    cache,
                    layer,
                    forcing=forcing,
                    site_dir=worker.site_dir,
                    scratch_dir=worker.scratch_root / request.job_id,
                    requested_variables=tuple(requested_variables),
                )
                window = grid.full_window
                progress(
                    "publishing", 0, 1, mode="full", window=_window_dict(window)
                )
                return SolveResult(
                    status="published",
                    scene_version=request.target_scene_version,
                    mode="full",
                    window=window,
                    time_indices=tuple(range(total_time_steps)),
                    variables=variables,
                    arrays={
                        name: np.ascontiguousarray(full_arrays[name], dtype=np.float32)
                        for name in variables
                    },
                    metrics={
                        "fallback_reason": "time_coverage_heal",
                        "window_fraction": 1.0,
                        "read_window_fraction": 1.0,
                        "dirty_windows": [],
                        "time_coverage_heal": True,
                    },
                    requested_time_subset=False,
                )
            selected = {
                name: np.ascontiguousarray(state[name][list(time_indices)], dtype=np.float32)
                for name in variables
            }
            # Honest read-side telemetry: the worker reports the read union
            # its solves actually touched; a full-tile outcome reads the
            # whole site. (Additive field; ``window_fraction`` above keeps
            # its write-union meaning.)
            read_window_fraction = union_window_area_fraction(
                outcome.read_windows, grid_info
            )
            if read_window_fraction is None:
                read_window_fraction = (
                    1.0
                    if outcome.mode == "full"
                    else round(union.area / grid.area_pixels, 6)
                )
            return SolveResult(
                status="published",
                scene_version=outcome.scene_revision,
                mode=outcome.mode,
                window=union,
                time_indices=time_indices,
                variables=variables,
                arrays=selected,
                metrics={
                    "fallback_reason": outcome.fallback_reason,
                    "window_fraction": round(union.area / grid.area_pixels, 6),
                    "read_window_fraction": read_window_fraction,
                    "dirty_windows": [_window_dict(w) for w in (outcome.write_windows or ())],
                    **stage_timing_metrics(
                        (outcome.diagnostics or {}).get("stage_timings")
                    ),
                },
                # T15 cut site (legacy incremental): the composed state was
                # cut to the REQUESTED times below before publishing.
                requested_time_subset=len(time_indices) < total_time_steps,
            )

        return solve

    return factory


def _apply_result_into_state(
    state: dict[str, np.ndarray],
    manifest: Mapping[str, Any],
    patch_arrays: Mapping[str, np.ndarray],
    variables: Sequence[str],
) -> None:
    """Apply one decoded result payload into full-site state arrays.

    Sparse-time payloads scatter at their manifest GLOBAL ``time_indices``
    — never at dense offsets. The legacy dense-prefix fallback stays legal
    exactly when the payload covers the site's full time axis; a
    partial-time payload with missing/mismatched indices is refused with
    :class:`patch_codec.PartialTimeScatterError` (T15 mis-scatter fence:
    guessing ``t = 0..n`` would land a changed-t plane at the wrong global
    time while the changed time kept its stale value).
    """
    window = manifest["window"]
    r0, r1 = int(window["row_start"]), int(window["row_stop"])
    c0, c1 = int(window["col_start"]), int(window["col_stop"])
    time_indices = [int(t) for t in manifest.get("time_indices", [])]
    for name in variables:
        patch = patch_arrays.get(name)
        if patch is None:
            continue
        target = state[name]
        region = target[:, r0:r1, c0:c1]
        if not time_indices or len(time_indices) != patch.shape[0]:
            if patch.shape[0] != target.shape[0]:
                raise patch_codec.PartialTimeScatterError(
                    name, patch.shape[0], target.shape[0]
                )
            time_indices = list(range(patch.shape[0]))
        region[time_indices] = patch


def _compose_current_state(
    context: RunnerContext,
    scenario_id: str,
    grid_info: Mapping[str, float | int],
    variables: Sequence[str],
    window: RasterWindow,
    target_version: int,
    *,
    extra_patches: Sequence[Any] = (),
) -> dict[str, np.ndarray]:
    """Lowest published result + every result <= target, then ``extra_patches``.

    Incremental semantics: the result for scene N equals the composition base
    plus every change published up to version N inside their write windows;
    cells outside any write window are unchanged. ``extra_patches`` are the
    freshly computed worker patches for the target version itself.

    The base is the LOWEST published version, not necessarily 0: when a
    scenario's baseline job is superseded before publishing (an edit lands
    while it is in flight on a site without stored baseline outputs), the
    next job bootstraps with one full-tile solve at its own target version —
    a full-site-window exact result, which is exactly as valid a composition
    base as a version-0 baseline. Requiring version 0 would permanently brick
    such scenarios: nothing can ever publish 0 again once the scene advanced.

    P8 performance: composition resumes from the cached state of the newest
    compatible version prefix (validated against the store's checksum chain,
    so resets / re-publishes invalidate stale entries automatically), so the
    per-job cost is O(newly published versions) decodes instead of
    O(all versions). The freshly composed state is cached for the next job
    BEFORE ``extra_patches`` are applied — extra patches are not durable
    yet, and a deterministic solver makes the eventual durable record equal
    to what this job computed.
    """
    store = context.store
    time_steps = int(grid_info["time_steps"])
    rows, cols = int(grid_info["rows"]), int(grid_info["cols"])
    versions = [v for v in store.result_versions(scenario_id) if v <= target_version]
    if not versions:
        raise StoreError(f"scenario {scenario_id} has no published result to compose against")
    records: dict[int, ResultRecord] = {}
    chain: list[str] = []
    for version in versions:
        record: ResultRecord | None = store.get_result(scenario_id, version)
        assert record is not None
        records[version] = record
        chain.append(record.checksum)
    # T15 composition guard, BEFORE the cache probe: a cached entry for a
    # gapped history must not be served either (the cache predates the gap
    # only if the checksum chain changed, but the guard is cheap and the
    # refusal is the contract — never a silently-wrong compose). Flag OFF
    # composes exactly as before.
    if selected_time.enabled():
        selected_time.guard_composition(
            [records[version].manifest for version in versions],
            variables,
            time_steps=time_steps,
            rows=rows,
            cols=cols,
        )

    state: dict[str, np.ndarray] | None = None
    applied = 0
    # Newest compatible cached prefix (chain match => identical history).
    for start in range(len(versions) - 1, -1, -1):
        cached = context.state_cache.get(
            scenario_id, versions[start], variables, chain[: start + 1]
        )
        if cached is not None:
            state = cached
            applied = start + 1
            break
    if state is None:
        state = {
            name: np.zeros((time_steps, rows, cols), dtype=np.float32)
            for name in variables
        }
    # A cache hit implies this exact composition succeeded before, so every
    # requested variable was covered by the durable history; otherwise the
    # replay below re-derives coverage from the decoded patches.
    have_values = (
        {name: True for name in variables}
        if applied
        else {name: False for name in variables}
    )
    for version in versions[applied:]:
        record = records[version]
        patch_arrays = patch_codec.decode_payload(record.manifest, record.payload_bytes())
        _apply_result_into_state(state, record.manifest, patch_arrays, variables)
        for name in variables:
            if name in patch_arrays:
                have_values[name] = True
    # Cache the durable composition for the next job — a private copy taken
    # BEFORE extra patches mutate ``state``, and only when the durable
    # history itself covers every variable (a composition propped up by
    # not-yet-durable extras must never be cached).
    if all(have_values.values()):
        context.state_cache.put(
            scenario_id,
            versions[-1],
            variables,
            chain,
            {name: state[name].copy() for name in variables},
        )
    for patch in extra_patches:
        manifest = {
            "window": _window_dict(patch.write_window),
            # Sparse patches (r3a met fast path) carry their global
            # timesteps explicitly and cover ONLY those rows; a dense
            # offset here would land a changed-t plane at t=0 while the
            # changed t kept its stale value. Legacy patches keep the
            # dense prefix semantics.
            "time_indices": (
                [int(t) for t in patch.time_indices]
                if patch.time_indices is not None
                else list(range(patch.n_time_steps))
            ),
        }
        _apply_result_into_state(state, manifest, patch.arrays, variables)
        for name in variables:
            if name in patch.arrays:
                have_values[name] = True
    missing = [name for name, have in have_values.items() if not have]
    if missing:
        raise StoreError(f"variables {missing} missing from the baseline result")
    return {
        name: state[name][
            :, window.row_start : window.row_stop, window.col_start : window.col_stop
        ]
        for name in variables
    }
