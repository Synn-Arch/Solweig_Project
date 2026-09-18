# SPDX-License-Identifier: GPL-3.0-only
"""G1.0 -- the window-planner superset differential (R5 design brief).

Scope: prove or refute the superset property G1.1's serve-gate relaxation
is admissible ONLY under. After a windowed revision-N+1 batch supersedes a
revision-N full-tile publication, the COMPOSED STORE SERVE (revision-N+1
values inside the batch write windows, retained revision-N values in the
cell remainders R5a keeps) must be BITWISE EQUAL to an INDEPENDENT
full-tile recompute at scene revision N+1, TILE-WIDE -- not just inside
the batch windows. Decomposed, that is the window-planner superset
property: for every revision transition r -> r+1 that published windows
W(r+1), every cell whose recomputed value changed between scene r and
scene r+1 lies inside W(r+1); cells outside the windows then provably
hold their rev-r values, and painting entries in ascending-revision order
reconstructs the recompute exactly. R5a's re-pinned tests prove the STORE
semantics (remainder values are rev-N because the retained entry IS
rev-N); nothing before this harness compared those remainder values to a
full recompute at N+1 -- that differential is the proof here.

Every case drives the REAL machinery end to end (no monkeypatched store,
no hand-crafted write windows -- the planner and worker derive them):

1. a real prepared site (real ``svf_calculator`` baseline, real forcing)
   and a real :class:`PlanExecutor`;
2. a SEEDING batch of tall vegetation edits whose planned influence union
   routes FULL -- a full-tile publication at revision 1 through the
   worker's own full path, recorded by the store as tile-wide entries for
   ``utci`` / ``tmrt`` / ``time_shadow``;
3. k seeded PROBE batches of small vegetation edits (add / move / resize
   / delete / multi-op, heights capped so the routed write windows stay a
   strict sub-window of the tile) -- each supersedes exactly the cells it
   repaints and publishes at the next revision;
4. after EVERY publication, the differential: the composed serve
   (``window_coverage`` FULL check, then newest-covering-entry painting
   sliced exactly like the executor's own plane assembly -- per-entry
   offsets against the PATCH window, payloads checksum-verified through
   ``load_patch``) vs an independent ``run_full_tile`` on the post-batch
   scene (fresh forcing, fresh scratch directory), for every published
   node at every timestep.

Verdict classes:

- ``undershoot`` (the defect G1.0 exists to catch): a mismatching cell
  OUTSIDE the step's published write windows -- the planner's window
  derivation missed a cell whose value changed. Fix the window
  derivation, never the serve gate.
- ``in_window_mismatch``: a mismatching cell INSIDE the write windows --
  a windowed-solve parity failure (a different defect class, equally
  fatal to the sweep).
- ``calibration_mismatch``: the post-seed differential (rev-1 only, the
  trivial regime) fails -- the worker's full-tile publication does not
  equal a fresh full solve; distinguishes harness/pathology from planner
  undershoot.
- ``coverage_not_full``: a compared (node, t) is not FULL after
  retention -- a store coverage defect.

Everything is seeded and deterministic (``numpy.random.default_rng``; no
wall clock in any decision path). NaN handling: utci planes legally carry
NaN cells (out-of-domain inputs); comparisons are ``np.array_equal(...,
equal_nan=True)`` on float32 arrays -- NaN-vs-NaN counts as equal, any
NaN-vs-value or value-vs-NaN cell is a mismatch like any other. Mismatch
COUNTS for reporting use a NaN-aware difference (``!=`` alone would count
NaN-vs-NaN pairs).

Profiles: ``smoke`` (1 site, fast) / ``suite`` (all sites) / ``thorough``
(all sites x seeds, longer probe sequences); the ``G1_SWEEP_SCALE``
environment variable overrides the CLI default (mirroring the V4 gate's
``R4B_SWEEP_SCALE``). A sweep whose probes ALL routed full-tile (no
windowed batch observed) or whose serves never carried mixed provenance
{N, N+1} proves nothing about retention and exits 1 (vacuity guard).

Honest non-coverage: single-tile scenarios, vegetation-family edits only
(building/massing batches always route the full regeneration chain and
meteorology batches route full by design -- neither produces the
windowed-over-full transition), probe trees capped below the site's
baseline ray-march amplitude so the local path's amplitude fences are
never the reason a probe routed full, and tiles <= 160 px per side at
pixel >= 4 m (the GVF write margin, ceil(22 m / pixel) + 1 cells, shrinks
the windowed regime below that).

Evidence: ``python docs/incremental_design_tool/realtime_collaboration/
design/proofs/g1_superset_diff.py`` writes
``/tmp/g1_proof/g1_superset_diff.json`` (+ stdout log).
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from solweig_gpu.incremental.edit_types import EditCommand  # noqa: E402
from solweig_gpu.incremental.executor import (  # noqa: E402
    PlanExecutor,
    VARIABLE_TO_RESULT_NODE,
)
from solweig_gpu.incremental.geometry import RasterWindow  # noqa: E402
from solweig_gpu.incremental.result import load_patch  # noqa: E402
from solweig_gpu.incremental.solver import (  # noqa: E402
    load_site_forcing,
    run_full_tile,
)
from solweig_gpu.incremental.store import CoverageStatus  # noqa: E402
from solweig_gpu.incremental.trees import TreeLayer, TreeSpec  # noqa: E402

# The site-construction helpers are the SAME idiom the canonical executor
# integration suites use (real tif site, real svf_calculator baseline,
# real met forcing) -- importing them keeps the harness's fixtures
# bit-identical to the tests' instead of growing a parallel builder.
from tests.test_incremental_worker import (  # noqa: E402
    DATE_STR,
    _build_cache,
    _compute_baseline_svf,
    _make_prepared_site,
)

#: The (variable, store-node) pairs the batches transport and the
#: differential compares. ``shadow`` publishes under the ``time_shadow``
#: node id (VARIABLE_TO_RESULT_NODE, executor.py).
COMPARED_VARIABLES: tuple[tuple[str, str], ...] = (
    ("utci", "utci"),
    ("tmrt", "tmrt"),
    ("shadow", "time_shadow"),
)

#: node id -> the patch variable name that publishes under it.
NODE_TO_VARIABLE: dict[str, str] = dict(
    (node, variable) for variable, node in COMPARED_VARIABLES
)

UNDERSHOOT = "undershoot"
IN_WINDOW_MISMATCH = "in_window_mismatch"
CALIBRATION_MISMATCH = "calibration_mismatch"
COVERAGE_NOT_FULL = "coverage_not_full"

_BASE_SEED = 20260905


# ---------------------------------------------------------------------------
# Site specs: real prepared sites, calibrated so both regimes are reachable
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SiteSpec:
    """One synthetic prepared site plus its edit-calibration envelope.

    Two regimes must both be REACHABLE through the real planner/worker
    routing, or the sweep proves nothing:

    - ``seed_trees`` (tall, spread): their influence union -- adapter
      windows plus the worker's GVF write margin, ``ceil(22 m / pixel)
      + 1`` cells per side -- covers >= the worker's 30% dirty fraction,
      so the seeding batch publishes a FULL-TILE revision-1 patch.
    - ``probe_height_range`` / ``probe_radius_range``: caps keeping a
      single small edit's routed window a strict sub-window of the tile
      (the 6-degree sky floor makes a tree's SVF radius ~9.5x its height;
      the caps are derived from that fence, never around it). Probe trees
      also stay at or below the baseline block height (12 m), so an edit
      can never raise the site-wide ray-march amplitude -- the local
      solve's amplitude fences are never the reason a probe routed full.
    """

    name: str
    rows: int
    cols: int
    pixel: float
    origin: tuple[float, float]
    epsg: int
    base_tree: tuple[float, float, float, float]  # (row, col, h, r)
    seed_trees: tuple[tuple[float, float, float, float], ...]
    probe_height_range: tuple[float, float]
    probe_radius_range: tuple[float, float]
    met_hours: tuple[int, ...] = (10, 11, 12)


SITES: dict[str, SiteSpec] = {
    "square4m": SiteSpec(
        name="square4m", rows=128, cols=128, pixel=4.0,
        origin=(1000.0, 2000.0), epsg=32616,
        base_tree=(20.0, 20.0, 5.0, 2.0),
        seed_trees=((90.0, 100.0, 12.0, 3.0), (30.0, 40.0, 11.0, 3.0)),
        probe_height_range=(3.5, 5.5),
        probe_radius_range=(1.5, 2.5),
    ),
    "wide4m": SiteSpec(
        name="wide4m", rows=96, cols=160, pixel=4.0,
        origin=(3000.0, 4000.0), epsg=32616,
        base_tree=(18.0, 24.0, 4.0, 2.0),
        seed_trees=(
            (70.0, 130.0, 12.0, 3.0),
            (24.0, 30.0, 11.0, 3.0),
            (48.0, 80.0, 12.0, 3.0),
        ),
        probe_height_range=(3.5, 5.0),
        probe_radius_range=(1.5, 2.5),
    ),
    "coarse6m": SiteSpec(
        name="coarse6m", rows=128, cols=128, pixel=6.0,
        origin=(5000.0, 6000.0), epsg=32616,
        base_tree=(22.0, 22.0, 5.0, 2.0),
        # four spread dominants: at 6 m pixels one 12 m tree's window is
        # ~0.15 of the tile (local), so the seeding union needs several.
        seed_trees=(
            (28.0, 28.0, 12.0, 3.0),
            (28.0, 100.0, 12.0, 3.0),
            (100.0, 28.0, 12.0, 3.0),
            (100.0, 100.0, 12.0, 3.0),
        ),
        probe_height_range=(4.0, 7.5),
        probe_radius_range=(1.5, 3.0),
    ),
}


PROFILES: dict[str, dict[str, Any]] = {
    "smoke": {
        "sequences": {"square4m": 1}, "probes": (2, 2),
    },
    "suite": {
        "sequences": {name: 1 for name in SITES}, "probes": (3, 4),
    },
    "thorough": {
        "sequences": {name: 3 for name in SITES}, "probes": (4, 6),
    },
}


_ROOT_COUNTER = itertools.count()


def fresh_root(label: str) -> Path:
    """A FRESH per-case root: leftover revisions of an earlier case must
    never be visible to a later one (results, cache, scratch alike)."""
    return Path(tempfile.mkdtemp(prefix=f"g1_{label}_{next(_ROOT_COUNTER)}_"))


# ---------------------------------------------------------------------------
# Case machinery
# ---------------------------------------------------------------------------


@dataclass
class StepRecord:
    step: int  # 0 = seeding batch; 1.. = probe batches
    verbs: list[str]
    outcome: str  # "full" | "windowed" | "no-op" | "failed"
    write_windows: list[str] = field(default_factory=list)
    window_fraction: float = 0.0
    fallback_reason: str | None = None
    mixed_revisions: list[int] = field(default_factory=list)
    detail: str = ""


@dataclass
class CaseReport:
    site: str
    seed: int
    n_probes: int
    steps: list[StepRecord] = field(default_factory=list)
    defects: list[dict[str, Any]] = field(default_factory=list)
    windowed_probes: int = 0
    full_probes: int = 0
    compared_serves: int = 0
    full_coverage_serves: int = 0
    mixed_serves: int = 0
    cells_bitwise_compared: int = 0
    nan_cells_compared: int = 0
    calibration: str = "not_run"

    def summary(self) -> dict[str, Any]:
        return {
            "site": self.site,
            "seed": self.seed,
            "n_probes": self.n_probes,
            "steps": [vars(record) for record in self.steps],
            "defects": self.defects,
            "windowed_probes": self.windowed_probes,
            "full_probes": self.full_probes,
            "compared_serves": self.compared_serves,
            "full_coverage_serves": self.full_coverage_serves,
            "mixed_serves": self.mixed_serves,
            "cells_bitwise_compared": self.cells_bitwise_compared,
            "nan_cells_compared": self.nan_cells_compared,
            "calibration": self.calibration,
        }


def _defect(
    kind: str, case: CaseReport, step: int, detail: str
) -> dict[str, Any]:
    entry = {
        "kind": kind,
        "site": case.site,
        "seed": case.seed,
        "step": step,
        "detail": detail,
        "repro": (
            f"g1_superset_diff.run_case({case.site!r}, seed={case.seed}, "
            f"n_probes={case.n_probes}, stop_before={step + 1})"
        ),
    }
    case.defects.append(entry)
    return entry


def _tree_state(spec: SiteSpec, row: float, col: float, h: float, r: float,
                tree_id: str) -> dict[str, Any]:
    return {
        "tree_id": tree_id,
        "x_m": spec.origin[0] + (col + 0.5) * spec.pixel,
        "y_m": spec.origin[1] - (row + 0.5) * spec.pixel,
        "height_m": h,
        "canopy_radius_m": r,
    }


def _spec_dict(tree: TreeSpec) -> dict[str, Any]:
    return {
        "tree_id": tree.tree_id,
        "x_m": tree.x_m,
        "y_m": tree.y_m,
        "height_m": tree.height_m,
        "canopy_radius_m": tree.canopy_radius_m,
    }


def _open_cell(spec: SiteSpec, rng: np.random.Generator) -> tuple[int, int]:
    """An away-from-buildings cell (the prepared-site scene puts blocks at
    fixed row/col fractions); clamped to a tile-edge margin."""
    margin = 10
    for _attempt in range(50):
        row = int(rng.integers(margin, spec.rows - margin))
        col = int(rng.integers(margin, spec.cols - margin))
        if (0.06 * spec.rows - 2 <= row < 0.26 * spec.rows + 2
                and 0.0 * spec.cols - 2 <= col < 0.30 * spec.cols + 2):
            continue
        if (0.56 * spec.rows - 2 <= row < 0.81 * spec.rows + 2
                and 0.54 * spec.cols - 2 <= col < 0.81 * spec.cols + 2):
            continue
        return row, col
    return spec.rows // 2, spec.cols // 2


class ProbeGenerator:
    """Seeded probe-batch generator over the executor's LIVE tree layer.

    Draws verbs from the live scene each step (the layer is the ground
    truth the executor's own replay checks against), so a seed fully
    determines the sequence. Probe trees are tracked so deletes/resizes/
    moves only ever target SMALL trees -- a probe aimed at a 12 m seed
    tree would dirty a dominant's full influence window and route full,
    exercising nothing about retention.
    """

    def __init__(self, spec: SiteSpec, rng: np.random.Generator) -> None:
        self.spec = spec
        self.rng = rng
        self.counter = 0
        self.probe_ids: list[str] = []

    def _next_id(self) -> str:
        self.counter += 1
        return f"p{self.counter}"

    def _height(self) -> float:
        lo, hi = self.spec.probe_height_range
        return round(float(self.rng.uniform(lo, hi)), 2)

    def _radius(self) -> float:
        lo, hi = self.spec.probe_radius_range
        return round(float(self.rng.uniform(lo, hi)), 2)

    def _add(self) -> tuple[str, str, dict[str, Any] | None, dict[str, Any] | None]:
        tree_id = self._next_id()
        row, col = _open_cell(self.spec, self.rng)
        state = _tree_state(
            self.spec, float(row), float(col), self._height(), self._radius(),
            tree_id,
        )
        self.probe_ids.append(tree_id)
        return ("add", f"add({tree_id}@r{row}c{col})", None, state)

    def _probe_tree(self, layer: TreeLayer) -> TreeSpec | None:
        live = {t.tree_id: t for t in layer.current_trees()}
        owned = [live[tid] for tid in self.probe_ids if tid in live]
        return owned[int(self.rng.integers(len(owned)))] if owned else None

    def _move(self, layer: TreeLayer):
        tree = self._probe_tree(layer)
        if tree is None:
            return self._add()
        # short moves only: a far move dirties BOTH windows and the union
        # routes full -- which proves nothing about retention.
        col = (tree.x_m - self.spec.origin[0]) / self.spec.pixel - 0.5
        row = (self.spec.origin[1] - tree.y_m) / self.spec.pixel - 0.5
        d_row = int(self.rng.integers(-6, 7))
        d_col = int(self.rng.integers(-6, 7))
        new_row = float(min(max(row + d_row, 8), self.spec.rows - 8))
        new_col = float(min(max(col + d_col, 8), self.spec.cols - 8))
        state = _tree_state(
            self.spec, new_row, new_col, tree.height_m, tree.canopy_radius_m,
            tree.tree_id,
        )
        return ("update", f"move({tree.tree_id}->r{new_row:.0f}c{new_col:.0f})",
                _spec_dict(tree), state)

    def _resize(self, layer: TreeLayer):
        tree = self._probe_tree(layer)
        if tree is None:
            return self._add()
        state = _tree_state(
            self.spec, 0.0, 0.0, self._height(), self._radius(), tree.tree_id
        )
        state["x_m"], state["y_m"] = tree.x_m, tree.y_m
        return ("update", f"resize({tree.tree_id},h={state['height_m']})",
                _spec_dict(tree), state)

    def _delete(self, layer: TreeLayer):
        tree = self._probe_tree(layer)
        if tree is None:
            return self._add()
        self.probe_ids.remove(tree.tree_id)
        return ("delete", f"delete({tree.tree_id})", _spec_dict(tree), None)

    def next_batch(self, layer: TreeLayer) -> list[tuple[str, str, dict | None, dict | None]]:
        """Draw one probe batch (a single verb, or a close pair)."""
        roll = self.rng.random()
        single: tuple[str, str, dict | None, dict | None]
        if roll < 0.30 or not self.probe_ids:
            single = self._add()
        elif roll < 0.50:
            single = self._move(layer)
        elif roll < 0.70:
            single = self._resize(layer)
        else:
            single = self._delete(layer)
        if self.rng.random() < 0.20:
            # multi-op batch: two small ops the planner coalesces into one
            # publication (windows merged -- still one revision bump).
            second = self._add()
            return [single, second]
        return [single]


def composed_serve(
    executor: PlanExecutor, node_id: str, t: int, rows: int, cols: int
) -> tuple[np.ndarray | None, list[int], str]:
    """The store's real read path, composed over the entry set.

    Mirrors the executor's own plane-assembly slicing (the met fast
    path, executor.py): ``window_coverage`` classifies the full-window
    request, then every overlapping entry's payload slice is painted at
    ``entry.write_window`` offsets against its PATCH window. Entries are
    pre-sorted ascending by window; later-revision entries overwrite
    earlier ones exactly where their windows overlap -- which, by the
    supersession discipline, is nowhere (disjoint remainders), so the
    paint order is deterministic regardless.

    Returns ``(plane, revisions_present, status_detail)``; ``plane`` is
    ``None`` when coverage is not FULL (never a serve).
    """
    window = RasterWindow(0, rows, 0, cols)
    coverage = executor.store.window_coverage(node_id, t, window=window)
    if coverage.status is not CoverageStatus.FULL:
        return None, [], (
            f"coverage {coverage.status.value} with missing "
            f"{[str(w) for w in coverage.missing_windows]}"
        )
    plane = np.full((rows, cols), np.nan, dtype=np.float32)
    revisions: set[int] = set()
    variable = NODE_TO_VARIABLE[node_id]
    for entry in coverage.entries:
        revisions.add(entry.scene_revision)
        patch = load_patch(entry.patch_path)  # checksum-verified load
        row = (
            patch.time_indices.index(t)
            if patch.time_indices is not None
            else t - patch.time_start
        )
        w, pw = entry.write_window, patch.write_window
        plane[
            w.row_start : w.row_stop, w.col_start : w.col_stop
        ] = patch.variable(variable)[
            row,
            w.row_start - pw.row_start : w.row_stop - pw.row_start,
            w.col_start - pw.col_start : w.col_stop - pw.col_start,
        ]
    if not revisions:
        return None, [], "FULL coverage with zero entries"
    return plane, sorted(revisions), "full"


def run_differential(
    case: CaseReport,
    executor: PlanExecutor,
    site: Path,
    scratch_root: Path,
    *,
    step: int,
    probe_windows: tuple[RasterWindow, ...] | None,
) -> bool:
    """Compare the composed serve against a fresh full-tile recompute.

    ``probe_windows``: the current step's PUBLISHED write windows (``None``
    for the post-seed calibration step, where every cell is inside the
    publication by construction). Mismatching cells inside them are
    ``in_window_mismatch``; outside them, ``undershoot``.
    """
    cache = executor.cache
    rows, cols = cache.rows, cache.cols
    forcing = load_site_forcing(
        cache, site_dir=site, selected_date_str=executor.selected_date_str
    )
    oracle = run_full_tile(
        cache,
        executor.layer,
        forcing=forcing,
        site_dir=site,
        scratch_dir=scratch_root / f"oracle-{step}",
        requested_variables=tuple(v for v, _ in COMPARED_VARIABLES),
    )
    clean = True
    for variable, node in COMPARED_VARIABLES:
        oracle_plane_series = oracle[variable]
        for t in range(cache.time_steps):
            plane, revisions, detail = composed_serve(
                executor, node, t, rows, cols
            )
            case.compared_serves += 1
            if plane is None:
                _defect(
                    COVERAGE_NOT_FULL, case, step,
                    f"{node}@t{t}: {detail}",
                )
                clean = False
                continue
            case.full_coverage_serves += 1
            if len(revisions) > 1:
                case.mixed_serves += 1
            oracle_plane = np.asarray(oracle_plane_series[t])
            if plane.dtype != np.float32 or oracle_plane.dtype != np.float32:
                _defect(
                    IN_WINDOW_MISMATCH, case, step,
                    f"{node}@t{t}: dtype {plane.dtype} vs "
                    f"{oracle_plane.dtype} (float32-exact required)",
                )
                clean = False
                continue
            case.cells_bitwise_compared += int(plane.size)
            case.nan_cells_compared += int(np.isnan(plane).sum())
            if np.array_equal(plane, oracle_plane, equal_nan=True):
                continue
            clean = False
            # NaN-aware cell difference (a raw != counts NaN-vs-NaN).
            left_nan = np.isnan(plane)
            right_nan = np.isnan(oracle_plane)
            differs = (plane != oracle_plane) & ~(left_nan & right_nan)
            for row, col in zip(*np.nonzero(differs)):
                if step == 0:
                    # The trivial regime (rev-1 full publication only):
                    # any mismatch is a calibration failure, not a window
                    # judgement.
                    kind = CALIBRATION_MISMATCH
                    detail_suffix = " -- calibration step (rev-1 full serve)"
                else:
                    inside = any(
                        w.row_start <= row < w.row_stop
                        and w.col_start <= col < w.col_stop
                        for w in (probe_windows or ())
                    )
                    kind = IN_WINDOW_MISMATCH if inside else UNDERSHOOT
                    detail_suffix = (
                        " -- cell inside the step's write windows"
                        if inside
                        else " -- cell OUTSIDE the step's published write "
                        "windows"
                    )
                _defect(
                    kind, case, step,
                    f"{node}@t{t} cell (r{int(row)},c{int(col)}): composed "
                    f"{plane[row, col]!r} (entry revisions {revisions}) vs "
                    f"recompute {oracle_plane[row, col]!r}" + detail_suffix,
                )
                break  # first mismatching cell per (node, t) is the pin
    return clean


def run_case(
    site_name: str,
    seed: int,
    n_probes: int,
    *,
    stop_before: int | None = None,
    verbose: bool = False,
) -> CaseReport:
    """Run one seeded case; returns the case report.

    ``stop_before`` truncates the probe sequence (defect reproduction:
    re-run the pinned prefix that constructs the failure; step 0 is the
    seeding batch, steps 1.. are probes).
    """
    spec = SITES[site_name]
    rng = np.random.default_rng(seed)
    case = CaseReport(site=site_name, seed=seed, n_probes=n_probes)

    root = fresh_root(site_name)
    grid, site = _make_prepared_site(
        root,
        rows=spec.rows, cols=spec.cols, pixel=spec.pixel,
        origin=spec.origin, epsg=spec.epsg,
        base_trees=(
            TreeSpec(
                "base0",
                spec.origin[0] + (spec.base_tree[1] + 0.5) * spec.pixel,
                spec.origin[1] - (spec.base_tree[0] + 0.5) * spec.pixel,
                spec.base_tree[2], spec.base_tree[3],
            ),
        ),
        met_hours=spec.met_hours,
    )
    _compute_baseline_svf(site)
    cache = _build_cache(
        site, root / "cache",
        met_path=site / "metfiles" / f"metfile_0_0_{DATE_STR}.txt",
        site_id=f"g1-{site_name}",
    )
    executor = PlanExecutor(
        cache=cache,
        layer=TreeLayer(cache.tree_base, grid),
        site_dir=site,
        results_root=root / "results",
        selected_date_str=DATE_STR,
    )

    def execute(tag: str, ops: list[tuple[str, str, dict | None, dict | None]]):
        commands = [
            executor.validate(
                EditCommand(
                    edit_id=f"{tag}-{i}",
                    scenario_id=executor._scenario_id,
                    base_scene_revision=executor.scene_revision,
                    adapter_id="vegetation_geometry",
                    operation=op,
                    old_state=old,
                    new_state=new,
                    requested_outputs=tuple(
                        v for v, _ in COMPARED_VARIABLES
                    ),
                    requested_times=None,  # vegetation is time-varying: ALL
                )
            )
            for i, (op, _desc, old, new) in enumerate(ops)
        ]
        return executor.execute(commands)

    # -- step 0: the seeding batch (full-tile publication at revision 1) --
    seed_ops = [
        (
            "add",
            f"seed_add(s{i}@r{row:.0f}c{col:.0f},h={h})",
            None,
            _tree_state(spec, row, col, h, r, f"s{i}"),
        )
        for i, (row, col, h, r) in enumerate(spec.seed_trees)
    ]
    record = StepRecord(
        step=0,
        verbs=[desc for (_op, desc, _o, _n) in seed_ops],
        outcome="full",
    )
    executed = execute("seed", seed_ops)
    if not executed.published or executed.mode != "full":
        _defect(
            CALIBRATION_MISMATCH, case, 0,
            f"seeding batch did not publish full-tile: status "
            f"{executed.status} mode {executed.mode}",
        )
        case.steps.append(record)
        return case
    record.write_windows = [str(w) for w in executed.write_windows]
    record.window_fraction = 1.0
    case.full_probes += 1
    if run_differential(
        case, executor, site, root, step=0, probe_windows=None
    ):
        case.calibration = "ok"
    else:
        case.calibration = "MISMATCH"
    case.steps.append(record)
    if verbose:
        print(
            f"    [step 0] seed FULL rev={executed.scene_revision} "
            f"calibration={case.calibration}"
        )

    # -- steps 1..n_probes: seeded windowed probe batches ---------------
    generator = ProbeGenerator(spec, rng)
    for step in range(1, n_probes + 1):
        if stop_before is not None and step >= stop_before:
            break
        ops = generator.next_batch(executor.layer)
        executed = execute(f"p{step}", ops)
        record = StepRecord(
            step=step,
            verbs=[desc for (_op, desc, _o, _n) in ops],
            outcome="failed" if not executed.published else executed.mode
            or "no-op",
        )
        if not executed.published:
            # A probe failure is a defect of a different family (edit
            # validation, supersession wedges) -- record it loudly.
            _defect(
                CALIBRATION_MISMATCH, case, step,
                f"probe batch did not publish: status={executed.status} "
                f"diagnostics={executed.diagnostics}",
            )
            case.steps.append(record)
            continue
        record.write_windows = [str(w) for w in executed.write_windows]
        record.window_fraction = (
            sum(w.area for w in executed.write_windows)
            / (spec.rows * spec.cols)
            if executed.write_windows
            else 1.0
        )
        record.fallback_reason = executed.fallback_reason
        if executed.mode == "full":
            case.full_probes += 1
            record.detail = (
                "probe routed full-tile (influence union >= the 30% dirty "
                "fraction): the differential still runs, but this step "
                "exercises no retention"
            )
        else:
            case.windowed_probes += 1
            if any(
                w == RasterWindow(0, spec.rows, 0, spec.cols)
                for w in executed.write_windows
            ) and len(executed.write_windows) == 1:
                record.detail = (
                    "probe published local mode with a tile-wide window "
                    "(windowed in name only)"
                )
        run_differential(
            case, executor, site, root, step=step,
            probe_windows=tuple(executed.write_windows),
        )
        _plane, revisions, _detail = composed_serve(
            executor, "utci", 0, spec.rows, spec.cols
        )
        record.mixed_revisions = revisions
        case.steps.append(record)
        if verbose:
            print(
                f"    [step {step}] {executed.mode} "
                f"{'+'.join(v.split('(')[0] for v in record.verbs)} "
                f"frac={record.window_fraction:.3f} revs={revisions} "
                f"defects={len(case.defects)}"
            )
    return case


def run_sweep(profile: str, *, verbose: bool = True) -> dict[str, Any]:
    """Run the profile's seeded sweep; returns the evidence dict.

    Per-case scratch/results roots are self-managed fresh tempdirs (the
    V4 gate's ``roundtrip_root`` discipline): leftover revisions of an
    earlier case must never be visible to a later one.
    """
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}")
    plan = PROFILES[profile]
    cases: list[CaseReport] = []
    started = time.perf_counter()
    for site_name, n_sequences in plan["sequences"].items():
        for i in range(n_sequences):
            seed = _BASE_SEED + i
            lo, hi = plan["probes"]
            n_probes = int(lo + (hi - lo) * (i % 3) / 2)
            case = run_case(site_name, seed, n_probes, verbose=verbose)
            cases.append(case)
            if verbose:
                s = case.summary()
                print(
                    f"[case] {site_name} seed={seed} steps={len(s['steps'])} "
                    f"windowed={s['windowed_probes']} full={s['full_probes']} "
                    f"mixed_serves={s['mixed_serves']} "
                    f"calibration={s['calibration']} "
                    f"defects={len(s['defects'])}"
                )
    elapsed = time.perf_counter() - started
    totals = {
        "profile": profile,
        "base_seed": _BASE_SEED,
        "cases": len(cases),
        "sites": len(plan["sequences"]),
        "steps": sum(len(c.steps) for c in cases),
        "windowed_probes": sum(c.windowed_probes for c in cases),
        "full_probes": sum(c.full_probes for c in cases),
        "compared_serves": sum(c.compared_serves for c in cases),
        "full_coverage_serves": sum(c.full_coverage_serves for c in cases),
        "mixed_serves": sum(c.mixed_serves for c in cases),
        "cells_bitwise_compared": sum(
            c.cells_bitwise_compared for c in cases
        ),
        "nan_cells_compared": sum(c.nan_cells_compared for c in cases),
        "calibrations_ok": sum(c.calibration == "ok" for c in cases),
        "defects": [d for case in cases for d in case.defects],
        "undershoots": sum(
            d["kind"] == UNDERSHOOT for c in cases for d in c.defects
        ),
        "load_average": tuple(round(x, 2) for x in os.getloadavg()),
        "cpu_count": os.cpu_count(),
        "wall_seconds": round(elapsed, 1),
    }
    return {"totals": totals, "cases": [c.summary() for c in cases]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--profile", choices=("smoke", "suite", "thorough"),
        default=os.environ.get("G1_SWEEP_SCALE", "thorough"),
    )
    parser.add_argument(
        "--out", default="/tmp/g1_proof/g1_superset_diff.json"
    )
    args = parser.parse_args(argv)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    result = run_sweep(args.profile)
    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=1, default=str)
    totals = result["totals"]
    print(
        f"=== G1 superset differential: profile={totals['profile']} "
        f"cases={totals['cases']} steps={totals['steps']} "
        f"windowed={totals['windowed_probes']} full={totals['full_probes']} "
        f"mixed_serves={totals['mixed_serves']} "
        f"cells={totals['cells_bitwise_compared']} "
        f"defects={len(totals['defects'])} "
        f"(undershoots={totals['undershoots']}) "
        f"({totals['wall_seconds']}s)"
    )
    print(f"wrote {args.out}")
    vacuous = totals["windowed_probes"] == 0 or totals["mixed_serves"] == 0
    if vacuous:
        print(
            "VACUOUS SWEEP: no windowed probe batch or no mixed-provenance "
            "serve was observed -- the superset property was NOT exercised"
        )
    return 1 if totals["defects"] or vacuous else 0


if __name__ == "__main__":
    sys.exit(main())
