# SPDX-License-Identifier: GPL-3.0-only
"""G2.2 -- the crash-restart proof harness (R5 design brief, final R5 step).

Scope: prove the temporal-checkpoint + scenario-state machinery survives a
process kill at every durable seam. A kill is modeled the honest way: the
victim runs in a REAL SUBPROCESS that arms a fault at a named seam inside
the real commit path and then ``os._exit``\\ s -- no Python exception
unwinding, no rollback handlers, no cleanup. The parent then inspects the
disk the dead process left behind, performs the RESTART exactly as a fresh
process would (fresh executor, ``restore_scenario_state`` from the newest
snapshot, replay of the op log -- the op log is truth, checkpoints are
reproduce aids), and proves four properties:

1. **Zero accepted ops lost** -- every op in the op log through the kill
   lands in the restarted scenario: the final scene revision and the
   per-op routing (status, mode, r0) reproduce the uninterrupted oracle
   run's exactly; the warm resume step may only be HIGHER, and only via
   the one legal mechanism (the killed run durably checkpointed the
   replayed op's own revision and the replay warm-resumes from that
   self-anchor -- idempotent-replay convergence).
2. **Revision monotonicity** -- no scene revision advances without its
   published results: at every kill point the newest durable snapshot's
   claimed revision has full published patch bytes on disk (every store
   entry's patch path exists and checksum-loads).
3. **No torn checkpoints survive** -- the checkpoint write is staged
   (``rev-XXXXXX.tmp``) + ``os.rename``; a kill in the window leaves the
   OLD record, the NEW record, or (kill between the same-revision rmtree
   and the rename) NEITHER -- never a torn hybrid. Every checkpoint the
   directory listing exposes must checksum-load. Byte-level tampering
   (flipped tensor byte; truncated ``checkpoint.json``) is rejected with
   the typed ``CheckpointError`` and the loader falls back to the previous
   valid record -- and the science still reproduces the oracle bitwise.
4. **Restart determinism** -- the restarted scenario's served planes
   (store-composed serve, G1.1 ascending-revision per-cell paint, payloads
   checksum-verified through ``load_patch``) are BITWISE equal to the
   uninterrupted canonical execution of the same op sequence, for every
   published variable at every timestep (``np.array_equal(...,
   equal_nan=True)`` on float32).

The op sequence (deterministic, seeded values) is built to exercise the
whole warm machinery across a restart:

* op 0: a radiation-affecting met edit at t=1 -- the warm consumer's
  prefix gate refuses (nothing published yet), the establishing full solve
  publishes revision 1 and a full-series thermal checkpoint;
* op 1: a second Ta edit at t=1 -- r0=1, the prefix (t=0) serves from the
  store, no anchor matches (the rev-1 checkpoint's met prefix covers the
  edited row) so the split solves COLD and re-anchors at r0=1 (rev 2);
* op 2: a Ta edit at t=2 -- r0=2 and the rev-2 anchor's fingerprint
  matches, so the suffix solves WARM from step 1 (the flagship case: a
  restarted process re-warms from a checkpoint that only exists on disk);
* op 3: a small tree add -- on this 128 px tile under the DEFAULT
  influence policy the GVF-margin-inflated dirty fraction lands above
  the full-recompute bar for any tree, so the add takes a full solve
  (measured, not assumed) whose thermal anchor@4 the next op skips as
  beyond-r0, while the scene edit invalidates every earlier anchor;
* op 4: a humidity edit at t=3 -- the warm consumer runs, every anchor is
  skipped (coverage-only / stale scene digest / covered prefix), and the
  split solves cold. Cold and warm must agree bitwise (the G2.1 parity
  fence), so the oracle equality holds either way.

Kill seams (each wraps the REAL function and ``os._exit``\\ s at the named
window; staged ``.tmp`` + rename are the real commit path, never mocked):

* ``solve_mid_series`` -- inside the physics series (after the Nth real
  ``Solweig_2022a_calc`` step of the killed op's solve);
* ``warm_split_suffix_entry`` -- on entry of the warm consumer's suffix
  ``solve_window`` call (prefix solved, nothing published: INSIDE the
  split); ``call_index=1`` is the phase-1 entry;
* ``pre_publish`` -- on entry of ``publish_staged_patch`` (staged ``.tmp``
  patch directory on disk, rename never ran; patched on BOTH the executor
  and worker module bindings so a worker-path publish is caught too);
* ``post_publish_pre_checkpoint`` -- patch renamed in, checkpoint absent;
* ``checkpoint_staged_pre_rename`` -- staging fully written, rename never
  ran (the mandated torn window: old record intact, staging invisible to
  listings);
* ``checkpoint_final_removed_pre_rename`` -- kill AFTER the same-revision
  ``rmtree`` of the old record and BEFORE the rename (leaves NEITHER; the
  loader must fall back to an earlier valid record). This window only
  opens when a RESTART rewrites a revision the killed run already
  checkpointed, so the case is a two-crash chain: child 1 dies after op 2
  wrote its checkpoint (post_checkpoint), child 2 restores from the op-1
  snapshot in a fresh process, replays op 2, and dies in the rmtree
  window of the rewrite -- which also proves a genuine cross-process
  restore reaches the identical code path;
* ``post_checkpoint`` -- everything durable; the restart replays forward.

Non-coverage (honesty): kills fire at Python seams inside the real commit
path, not at arbitrary machine-code points (the APFS rename atomicity
argument makes the durability surface equivalent, but kernel-level torn
writes are out of scope); the op log is harness-owned (the server's
durable accept-queue lands with R8's chaos suite); one site shape
(128x128 @ 2 m, T=4), sequential single-writer batches (no competing
writers mid-solve); the store is in-memory by design and is rebuilt from
the snapshot exactly as ``scenario_state.restore_into_executor`` does;
corruption covers one tensor-byte flip and one truncated-metadata variant
on the newest revision; checkpoints are consumed only through the warm
consumer's own scan (``select_replay_checkpoint`` semantics are pinned by
the G2.0/G2.1 suites); every vegetation job in the sequence routes FULL
under the default influence policy on this tile (fraction >= 0.30 once
the GVF write margin inflates the window), so windowed/local veg
publications and coverage-only checkpoints are NOT exercised here --
G1.1's superset gate and the G2.0 checkpoint suite own those paths.

Profiles: ``smoke`` (the eight mandated kill points + the corruption
case) / ``suite`` (smoke + four extra seams) / ``thorough`` (suite x 2
seeds); ``G22_SWEEP_SCALE`` overrides the CLI default (the R4B/G1 idiom).

Evidence: ``python docs/incremental_design_tool/realtime_collaboration/
design/proofs/g22_crash_restart.py`` writes
``/tmp/g22_proof/g22_crash_restart.json`` (+ stdout log).
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.test_incremental_worker import (  # noqa: E402
    DATE_STR,
    _build_cache,
    _compute_baseline_svf,
    _make_prepared_site,
)
from solweig_gpu.incremental.edit_types import EditCommand  # noqa: E402
from solweig_gpu.incremental.geometry import RasterGrid  # noqa: E402
from solweig_gpu.incremental.result import load_patch  # noqa: E402
from solweig_gpu.incremental.store import CoverageStatus  # noqa: E402
from solweig_gpu.incremental.trees import TreeLayer, TreeSpec  # noqa: E402

PROOF_ROOT = Path("/tmp/g22_proof")
EVIDENCE_JSON = PROOF_ROOT / "g22_crash_restart.json"

_BASE_SEED = 20260905

#: The victim subprocess's exit code / stderr marker on a fired kill.
KILL_EXIT = 70
KILL_MARKER = "G22KILL:"
#: The victim reached the kill op but the seam never fired (a harness
#: calibration problem: the op routed somewhere the seam cannot see).
NOOP_EXIT = 71

MET_ADAPTER_ID = "meteorological_forcing"
VEG_ADAPTER_ID = "vegetation_geometry"

#: (patch variable, store node) pairs the differential compares -- the
#: transported vocabulary of the canonical integration suites.
COMPARED_VARIABLES: tuple[tuple[str, str], ...] = (
    ("utci", "utci"),
    ("tmrt", "tmrt"),
    ("shadow", "time_shadow"),
)
VARIABLES = tuple(variable for variable, _node in COMPARED_VARIABLES)

#: The one prepared-site shape (the met fast-path suite's real-SVF site):
#: 128x128 @ 2 m, met hours 10..13 (T=4), one baseline tree.
SITE_ROWS = SITE_COLS = 128
SITE_PIXEL = 2.0
SITE_ORIGIN = (300000.0, 4100000.0)
SITE_EPSG = 32616
SITE_ID = "g22-crash"

# met table columns (the r3a affinity evidence): humidity=10, Ta=11.
_COL_HUMIDITY = 10
_COL_AIR_TEMPERATURE = 11

_REVISION_DIR = re.compile(r"^rev-(\d{6})$")
_STAGING_DIR = re.compile(r"^rev-(\d{6})\.tmp$")


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------


KILL_CASES: dict[str, list[dict[str, Any]]] = {
    "smoke": [
        dict(
            name="op0_solve_mid_series",
            kill_op=0,
            seam="solve_mid_series",
            param={"after_steps": 1},
        ),
        dict(
            name="op0_pre_publish_cold_restart",
            kill_op=0,
            seam="pre_publish",
        ),
        dict(
            name="op2_warm_split_suffix_entry",
            kill_op=2,
            seam="warm_split_suffix_entry",
            param={"call_index": 2},
        ),
        dict(name="op2_pre_publish", kill_op=2, seam="pre_publish"),
        dict(
            name="op2_post_publish_pre_checkpoint",
            kill_op=2,
            seam="post_publish_pre_checkpoint",
        ),
        dict(
            name="op2_checkpoint_staged_pre_rename",
            kill_op=2,
            seam="checkpoint_staged_pre_rename",
        ),
        dict(
            name="op2_checkpoint_final_removed_pre_rename",
            kill_op=2,
            seam="post_checkpoint",
            second_kill=dict(
                kill_op=2, seam="checkpoint_final_removed_pre_rename"
            ),
        ),
        dict(name="op4_post_checkpoint", kill_op=4, seam="post_checkpoint"),
    ],
    "suite": [
        # appended to smoke by PROFILES
        dict(
            name="op1_solve_mid_series",
            kill_op=1,
            seam="solve_mid_series",
            param={"after_steps": 2},
        ),
        dict(
            name="op3_solve_mid_series",
            kill_op=3,
            seam="solve_mid_series",
            param={"after_steps": 1},
        ),
        dict(
            name="op2_warm_split_phase1_entry",
            kill_op=2,
            seam="warm_split_suffix_entry",
            param={"call_index": 1},
        ),
        dict(name="op4_pre_publish", kill_op=4, seam="pre_publish"),
    ],
    "thorough": [],
}

PROFILES: dict[str, dict[str, Any]] = {
    "smoke": {"cases": KILL_CASES["smoke"], "seeds": (_BASE_SEED,)},
    "suite": {
        "cases": KILL_CASES["smoke"] + KILL_CASES["suite"],
        "seeds": (_BASE_SEED,),
    },
    "thorough": {
        "cases": KILL_CASES["smoke"] + KILL_CASES["suite"],
        "seeds": (_BASE_SEED, _BASE_SEED + 1),
    },
}

_ROOT_COUNTER = itertools.count()


def fresh_root(label: str) -> Path:
    """A FRESH per-case root: no earlier case's revisions may leak in."""
    return Path(tempfile.mkdtemp(prefix=f"g22_{label}_{next(_ROOT_COUNTER)}_"))


# ---------------------------------------------------------------------------
# Site + op sequence (deterministic; seeded values, fixed shape)
# ---------------------------------------------------------------------------


@dataclass
class SiteContext:
    """The once-per-sweep prepared site every process rebuilds its cache
    from (deterministic bytes -> identical cache identity everywhere)."""

    root: Path
    site_dir: Path
    grid: RasterGrid
    scenario_id: str = "default"

    @property
    def met_path(self) -> Path:
        return self.site_dir / "metfiles" / f"metfile_0_0_{DATE_STR}.txt"


def build_site() -> SiteContext:
    root = PROOF_ROOT / "site"
    site = root / "processed_inputs"
    grid = RasterGrid(
        rows=SITE_ROWS, cols=SITE_COLS, pixel_size_m=SITE_PIXEL,
        origin_x_m=SITE_ORIGIN[0], origin_y_m=SITE_ORIGIN[1],
    )
    if site.is_dir():
        return SiteContext(root=root, site_dir=site, grid=grid)
    root.mkdir(parents=True, exist_ok=True)
    base = TreeSpec(
        "b1",
        SITE_ORIGIN[0] + 20.5 * SITE_PIXEL,
        SITE_ORIGIN[1] - 20.5 * SITE_PIXEL,
        3.0,
        2.0,
    )
    grid, site = _make_prepared_site(
        root,
        rows=SITE_ROWS,
        cols=SITE_COLS,
        pixel=SITE_PIXEL,
        origin=SITE_ORIGIN,
        epsg=SITE_EPSG,
        base_trees=(base,),
        met_hours=range(10, 14),
    )
    _compute_baseline_svf(site)
    return SiteContext(root=root, site_dir=site, grid=grid)



def build_ops(site_ctx: SiteContext, seed: int) -> list[dict[str, Any]]:
    """The five-op sequence (fixed shape; seeded values, derived from the
    BASELINE met table so every edit provably changes its row)."""
    from solweig_gpu.incremental.solver import load_site_forcing

    cache = _build_cache(
        site_ctx.site_dir,
        site_ctx.root / "cache_values",
        met_path=site_ctx.met_path,
        site_id=SITE_ID,
    )
    base = load_site_forcing(
        cache, site_dir=site_ctx.site_dir, selected_date_str=DATE_STR
    ).met_table
    rng = np.random.default_rng(seed)
    ta1 = float(base[1, _COL_AIR_TEMPERATURE]) + 10.0 + float(rng.integers(0, 4))
    ta2 = float(base[2, _COL_AIR_TEMPERATURE]) + 12.0 + float(rng.integers(0, 4))
    hu3 = float(base[3, _COL_HUMIDITY]) + 15.0 + float(rng.integers(0, 6))
    veg_h = 4.0 + 0.25 * float(rng.integers(0, 3))
    shutil.rmtree(site_ctx.root / "cache_values", ignore_errors=True)
    return [
        {
            "kind": "met", "id": "m0", "variable": "air_temperature",
            "time_index": 1, "value": ta1,
        },
        {
            "kind": "met", "id": "m1", "variable": "air_temperature",
            "time_index": 1, "value": ta1 + 2.0,
        },
        {
            "kind": "met", "id": "m2", "variable": "air_temperature",
            "time_index": 2, "value": ta2,
        },
        {
            "kind": "veg", "id": "v3", "height_m": veg_h,
            "x_m": SITE_ORIGIN[0] + 40.5 * SITE_PIXEL,
            "y_m": SITE_ORIGIN[1] - 40.5 * SITE_PIXEL,
        },
        {
            "kind": "met", "id": "m4", "variable": "humidity",
            "time_index": 3, "value": hu3,
        },
    ]


def build_edit(executor, op: dict[str, Any]):
    """One validated edit for ``op`` against the executor's CURRENT state
    (the same construction every process uses, so replays are identical)."""
    if op["kind"] == "met":
        command = EditCommand(
            edit_id=op["id"],
            scenario_id=executor._scenario_id,
            base_scene_revision=executor.scene_revision,
            adapter_id=MET_ADAPTER_ID,
            operation="update_time_row",
            old_state=None,
            new_state={
                "time_index": op["time_index"],
                "values": {op["variable"]: op["value"]},
            },
            requested_outputs=VARIABLES,
            requested_times=(op["time_index"],),
        )
    else:
        # A tree ADD (the prepared site's base canopy is raster-only: it
        # carries no editable TreeSpec, so the sequence adds its own).
        new = {
            "tree_id": "g22-tree",
            "x_m": float(op["x_m"]),
            "y_m": float(op["y_m"]),
            "height_m": float(op["height_m"]),
            "canopy_radius_m": 2.5,
        }
        command = EditCommand(
            edit_id=op["id"],
            scenario_id=executor._scenario_id,
            base_scene_revision=executor.scene_revision,
            adapter_id=VEG_ADAPTER_ID,
            operation="add",
            old_state=None,
            new_state=new,
            requested_outputs=VARIABLES,
            requested_times=None,  # vegetation is time-varying: ALL
        )
    return executor.validate(command)


def make_executor(site_ctx: SiteContext, cache_dir: Path, results_root: Path):
    """A fresh executor exactly as every process builds one (default
    influence policy: small veg edits route windowed/local)."""
    from solweig_gpu.incremental.executor import PlanExecutor

    cache = _build_cache(
        site_ctx.site_dir,
        cache_dir,
        met_path=site_ctx.met_path,
        site_id=SITE_ID,
    )
    return PlanExecutor(
        cache=cache,
        layer=TreeLayer(cache.tree_base, site_ctx.grid),
        site_dir=site_ctx.site_dir,
        results_root=results_root,
        selected_date_str=DATE_STR,
    )


# ---------------------------------------------------------------------------
# Serve + diagnostics (the G1.1 ascending-revision per-cell paint)
# ---------------------------------------------------------------------------


def composed_serve(executor, variable: str, time_index: int) -> np.ndarray:
    """The store-composed plane: FULL-coverage gate, entries painted in
    ascending scene_revision order, payloads checksum-verified through
    ``load_patch`` and sliced exactly like the executor's own assembly."""
    from solweig_gpu.incremental.executor import VARIABLE_TO_RESULT_NODE

    full = executor.grid.full_window
    node_id = VARIABLE_TO_RESULT_NODE.get(variable, variable)
    coverage = executor.store.window_coverage(node_id, time_index, window=full)
    if coverage.status is not CoverageStatus.FULL:
        raise RuntimeError(
            f"{variable}@{time_index} coverage is {coverage.status.value}: "
            "the final state must be fully published"
        )
    plane = np.full((full.height, full.width), np.nan, dtype=np.float32)
    patches: dict[Path, Any] = {}
    for entry in sorted(coverage.entries, key=lambda item: item.scene_revision):
        if entry.patch_path not in patches:
            patches[entry.patch_path] = load_patch(entry.patch_path)
        patch = patches[entry.patch_path]
        row = (
            patch.time_indices.index(time_index)
            if patch.time_indices is not None
            else time_index - patch.time_start
        )
        w, pw = entry.write_window, patch.write_window
        plane[
            w.row_start : w.row_stop, w.col_start : w.col_stop
        ] = patch.variable(variable)[
            row,
            w.row_start - pw.row_start : w.row_stop - pw.row_start,
            w.col_start - pw.col_start : w.col_stop - pw.col_start,
        ]
    return plane


def op_diag(executed) -> dict[str, Any]:
    """The per-op routing fingerprint the restart must reproduce."""
    diag = {
        "status": executed.status,
        "mode": executed.mode,
        "revision": executed.scene_revision,
    }
    for key in (
        "met_warm_path",
        "met_warm_r0",
        "met_warm_resume_step",
        "met_warm_checkpoint_revision",
        "met_fast_path",
    ):
        diag[key] = executed.diagnostics.get(key)
    return diag


def final_serves(executor, time_steps: int) -> dict[str, list[np.ndarray]]:
    return {
        variable: [composed_serve(executor, variable, t) for t in range(time_steps)]
        for variable in VARIABLES
    }


# ---------------------------------------------------------------------------
# The victim subprocess
# ---------------------------------------------------------------------------


def _is_checkpoint_final(target: Any) -> bool:
    path = Path(target)
    return path.parent.name == "checkpoints" and _REVISION_DIR.match(path.name)


def arm_kill(spec: dict[str, Any]) -> None:
    """Wrap the REAL seam function; ``os._exit`` at the named window.

    Every wrapper calls the real implementation for every invocation before
    the target one (the commit path exercised is the production one; only
    the process death is injected).
    """
    import solweig_gpu.incremental.executor as executor_module
    import solweig_gpu.incremental.worker as worker_module
    import solweig_gpu.utci_process as utci_module

    seam = spec["seam"]
    param = spec.get("param") or {}

    def die(tag: str) -> None:
        sys.stderr.write(f"{KILL_MARKER}{tag}\n")
        sys.stderr.flush()
        os._exit(KILL_EXIT)

    if seam == "solve_mid_series":
        real = utci_module.Solweig_2022a_calc
        state = {"steps": 0}
        target = int(param.get("after_steps", 1))

        def step_wrapper(*args, **kwargs):
            out = real(*args, **kwargs)
            state["steps"] += 1
            if state["steps"] >= target:
                die(f"solve_mid_series:after_{state['steps']}_steps")
            return out

        utci_module.Solweig_2022a_calc = step_wrapper
        return

    if seam == "warm_split_suffix_entry":
        real = executor_module.solve_window
        state = {"calls": 0}
        target = int(param.get("call_index", 2))

        def solve_wrapper(*args, **kwargs):
            state["calls"] += 1
            if state["calls"] == target:
                die(f"warm_split:call_{state['calls']}_entry")
            return real(*args, **kwargs)

        executor_module.solve_window = solve_wrapper
        return

    if seam == "pre_publish":
        # Both module bindings: the warm consumer publishes through the
        # executor module, the worker's own jobs through the worker module.
        def publish_wrapper(*args, **kwargs):
            die("pre_publish")

        executor_module.publish_staged_patch = publish_wrapper
        worker_module.publish_staged_patch = publish_wrapper
        return

    if seam == "post_publish_pre_checkpoint":
        def checkpoint_wrapper(*args, **kwargs):
            die("post_publish_pre_checkpoint")

        executor_module.write_checkpoint = checkpoint_wrapper
        return

    if seam == "post_checkpoint":
        real = executor_module.write_checkpoint
        state = {"writes": 0}
        target = int(param.get("after_writes", 1))

        def after_wrapper(*args, **kwargs):
            out = real(*args, **kwargs)
            state["writes"] += 1
            if state["writes"] >= target:
                die(f"post_checkpoint:after_{state['writes']}_writes")
            return out

        executor_module.write_checkpoint = after_wrapper
        return

    if seam == "checkpoint_staged_pre_rename":
        real_rename = os.rename

        def rename_wrapper(src, dst, *args, **kwargs):
            if _is_checkpoint_final(dst):
                die("checkpoint_staged_pre_rename")
            return real_rename(src, dst, *args, **kwargs)

        os.rename = rename_wrapper  # the dying process; global is fine
        return

    if seam == "checkpoint_final_removed_pre_rename":
        real_rmtree = shutil.rmtree

        def rmtree_wrapper(target, *args, **kwargs):
            if _is_checkpoint_final(target):
                real_rmtree(target, *args, **kwargs)
                die("checkpoint_final_removed_pre_rename")
            return real_rmtree(target, *args, **kwargs)

        shutil.rmtree = rmtree_wrapper  # the dying process; global is fine
        return

    raise ValueError(f"unknown kill seam {seam!r}")


def run_child(spec_path: str) -> int:
    """The victim: replay ops (optionally restored from a snapshot), die
    at the armed seam, save a snapshot after every COMPLETED op."""
    spec = json.loads(Path(spec_path).read_text())
    site = json.loads(spec["site_json"])
    site_ctx = SiteContext(
        root=Path(site["root"]),
        site_dir=Path(site["site_dir"]),
        grid=RasterGrid(
            rows=site["rows"], cols=site["cols"],
            pixel_size_m=site["pixel"],
            origin_x_m=site["origin_x"], origin_y_m=site["origin_y"],
        ),
    )
    results_root = Path(spec["results_root"])
    executor = make_executor(site_ctx, Path(spec["cache_dir"]), results_root)
    ops = spec["ops"]
    start = int(spec.get("start_op", 0))
    if spec.get("restore_snapshot"):
        executor.restore_scenario_state(spec["restore_snapshot"])
    kill_op = spec["kill_op"]
    armed = False
    for index in range(start, len(ops)):
        if index == kill_op and not armed:
            arm_kill(spec)
            armed = True
        executed = executor.execute([build_edit(executor, ops[index])])
        if not executed.published:
            sys.stderr.write(
                f"G22CHILD: op {index} did not publish "
                f"({executed.status}: {executed.diagnostics.get('reason')})\n"
            )
            return 72
        executor.save_scenario_state(Path(spec["snap_root"]) / f"op{index:02d}")
        if index == kill_op:
            # Reached here only if the seam never fired.
            sys.stderr.write(f"{KILL_MARKER}UNREACHED:{spec['seam']}\n")
            return NOOP_EXIT
    return 0


# ---------------------------------------------------------------------------
# Parent-side disk inspection + restart
# ---------------------------------------------------------------------------


def inspect_disk(
    results_root: Path,
    snap_root: Path,
    *,
    expect_torn: frozenset[int] = frozenset(),
) -> tuple[dict[str, Any], str | None]:
    """What the dead process left behind. Returns ``(report, defect)``.

    Property 3 (no torn checkpoint survives) and property 2 (revision
    monotonicity) are asserted HERE, against the disk alone.
    ``expect_torn`` exempts revisions the harness itself tampered with
    (the corruption case): their typed rejection is recorded as evidence,
    not as a surviving-tear defect.
    """
    from solweig_gpu.incremental.checkpoints import (
        CheckpointError,
        list_checkpoints,
        load_checkpoint,
    )
    from solweig_gpu.incremental.scenario_state import read_snapshot

    scenario_root = results_root / "default"
    checkpoint_root = scenario_root / "checkpoints"
    report: dict[str, Any] = {"checkpoints": [], "torn_rejects": []}
    revisions = list_checkpoints(checkpoint_root)
    for revision in revisions:
        try:
            record = load_checkpoint(checkpoint_root / f"rev-{revision:06d}")
        except CheckpointError as error:
            if revision in expect_torn:
                report["torn_rejects"].append(
                    f"rev-{revision:06d}: {str(error)[:120]}"
                )
                continue
            # A record the LISTING exposes must never be torn: staging is
            # invisible by name, so a loadable-looking rev that fails its
            # checksums is a torn hybrid that survived -- the defect class
            # this harness exists to catch.
            return report, f"torn_checkpoint_survived: rev-{revision:06d}: {error}"
        report["checkpoints"].append(
            {
                "revision": revision,
                "next_step": record.next_step,
                "thermal": bool(record.thermal_tensors),
            }
        )
    report["staging_tmp"] = sorted(
        path.name
        for path in checkpoint_root.iterdir()
        if _STAGING_DIR.match(path.name)
    ) if checkpoint_root.is_dir() else []

    snapshots = sorted(
        path for path in snap_root.iterdir() if path.is_dir()
    ) if snap_root.is_dir() else []
    report["snapshots"] = [path.name for path in snapshots]
    latest = snapshots[-1] if snapshots else None
    report["restored_from"] = latest.name if latest else None
    if latest is not None:
        try:
            snapshot = read_snapshot(latest)
        except Exception as error:  # noqa: BLE001 - any failure is a defect
            return report, f"snapshot_unreadable: {latest.name}: {error}"
        # Property 2: the snapshot's claimed revision must be fully
        # published -- every store entry's patch bytes exist and load.
        seen_paths: set[Path] = set()
        at_revision = 0
        for item in snapshot.store_entries:
            if int(item["scene_revision"]) == snapshot.scene_revision:
                at_revision += 1
            raw = item.get("patch_path")
            if not raw:
                continue
            path = Path(raw)
            if path in seen_paths:
                continue
            seen_paths.add(path)
            if not path.is_dir():
                return report, f"snapshot_patch_missing: {path}"
            try:
                load_patch(path)
            except Exception as error:  # noqa: BLE001
                return report, f"snapshot_patch_unloadable: {path}: {error}"
        if at_revision == 0:
            return report, (
                "revision_without_results: snapshot claims revision "
                f"{snapshot.scene_revision} but records no entries at it"
            )
        report["snapshot_revision"] = snapshot.scene_revision

    # Informational: patch directories the snapshot does not reference
    # (expected after post-publish kills; the store never serves them).
    referenced = {str(path) for path in seen_paths} if latest else set()
    orphans = [
        path.name
        for path in scenario_root.glob("rev-*")
        if path.is_dir() and str(path) not in referenced
    ]
    report["orphan_patch_dirs"] = orphans
    dot_staging = scenario_root / ".staging"
    report["dot_staging_entries"] = (
        sorted(path.name for path in dot_staging.iterdir())
        if dot_staging.is_dir()
        else []
    )
    return report, None


def restart_and_compare(
    site_ctx: SiteContext,
    case_root: Path,
    disk: dict[str, Any],
    ops: list[dict[str, Any]],
    oracle_ops: list[dict[str, Any]],
    oracle_planes: dict[str, list[np.ndarray]],
    time_steps: int,
    *,
    expect_routing_divergence: str | None,
) -> tuple[dict[str, Any], str | None]:
    """The restart: fresh executor, restore the newest snapshot (if any),
    replay the op log from there, prove bitwise equality + routing parity."""
    results_root = case_root / "results"
    snap_root = case_root / "snaps"
    executor = make_executor(site_ctx, case_root / "cache_restart", results_root)
    restored = disk.get("restored_from")
    start = 0
    if restored is not None:
        executor.restore_scenario_state(snap_root / restored)
        start = int(restored[2:]) + 1
    replayed: list[dict[str, Any]] = []
    warm_after_restart = False
    for index in range(start, len(ops)):
        executed = executor.execute([build_edit(executor, ops[index])])
        if not executed.published:
            return {}, f"restart_op_failed: op {index}: {executed.status}"
        diag = op_diag(executed)
        replayed.append(diag)
        if diag.get("met_warm_resume_step") and diag["met_warm_resume_step"] > 0:
            warm_after_restart = True

    # Property 1/4 helpers: routing parity + bitwise serve equality.
    # Parity is exact on status/mode/revision/r0. The resume step may be
    # HIGHER than the oracle's by exactly one legal mechanism: the killed
    # run already durably checkpointed THIS op's own revision (a
    # post-checkpoint kill), and the replay warm-resumes from that
    # self-anchor -- idempotent-replay convergence, the entire point of
    # checkpoints. It may never be LOWER (a durable anchor the oracle had
    # can only be missing by corruption, which runs under the exempted
    # corruption case).
    if expect_routing_divergence is None:
        for offset, diag in enumerate(replayed):
            oracle = oracle_ops[start + offset]
            for key in ("status", "mode", "revision", "met_warm_r0"):
                if diag.get(key) != oracle.get(key):
                    return {}, (
                        f"routing_divergence: op {start + offset} "
                        f"{key}: restarted {diag.get(key)!r} != oracle "
                        f"{oracle.get(key)!r}"
                    )
            resumed = diag.get("met_warm_resume_step") or 0
            oracle_resumed = oracle.get("met_warm_resume_step") or 0
            if resumed < oracle_resumed:
                return {}, (
                    f"routing_divergence: op {start + offset} resumed at "
                    f"{resumed} < oracle {oracle_resumed}: an anchor the "
                    "uninterrupted run used is missing or unusable"
                )
            if resumed > oracle_resumed:
                anchor_rev = diag.get("met_warm_checkpoint_revision")
                if anchor_rev != oracle.get("revision"):
                    return {}, (
                        f"routing_divergence: op {start + offset} resumed "
                        f"further ({resumed} > {oracle_resumed}) from "
                        f"rev-{anchor_rev!r}, which is not the killed "
                        "run's own self-anchor"
                    )
    planes = final_serves(executor, time_steps)
    cells = 0
    for variable in VARIABLES:
        for t in range(time_steps):
            if not np.array_equal(
                planes[variable][t], oracle_planes[variable][t], equal_nan=True
            ):
                mismatches = int(
                    np.sum(
                        planes[variable][t] != oracle_planes[variable][t],
                    )
                )
                return {}, (
                    f"serve_mismatch: {variable}@{t}: {mismatches} cells "
                    "differ from the uninterrupted oracle"
                )
            cells += int(planes[variable][t].size)
    revision = executor.scene_revision
    if revision != len(ops):
        return {}, (
            f"ops_lost: final revision {revision} != {len(ops)} accepted ops"
        )
    return (
        {
            "restored_from": restored,
            "ops_replayed": len(replayed),
            "replayed": replayed,
            "warm_after_restart": warm_after_restart,
            "cells_bitwise_compared": cells,
            "final_revision": revision,
        },
        None,
    )


# ---------------------------------------------------------------------------
# Corruption sub-case (property 3: checksum reject -> previous-valid)
# ---------------------------------------------------------------------------


def run_corruption_case(
    site_ctx: SiteContext,
    ops: list[dict[str, Any]],
    oracle_ops: list[dict[str, Any]],
    oracle_planes: dict[str, list[np.ndarray]],
    time_steps: int,
) -> tuple[dict[str, Any], list[str]]:
    """Tamper the newest checkpoint's bytes and prove the typed rejection,
    the previous-valid fallback, and bitwise-correct science after restart.

    Runs ops 0..1 cleanly (rev-000002 = the warm anchor), then tears the
    rev-000002 record two ways -- a flipped tensor byte, then a truncated
    ``checkpoint.json`` -- and finally restarts from the op-1 snapshot and
    replays ops 2..4. The torn newest record must force the warm consumer
    onto the cold split (routing divergence BY DESIGN) whose science is
    still bitwise the oracle's (the G2.1 warm==cold parity fence).
    """
    from solweig_gpu.incremental.checkpoints import (
        CheckpointError,
        list_checkpoints,
        load_checkpoint,
        load_latest_checkpoint,
    )

    defects: list[str] = []
    report: dict[str, Any] = {
        "name": "corruption_torn_newest",
        "kill_op": None,
        "seam": "byte_tamper",
        "verdict": "FAIL",
        "defect": None,
    }
    started = time.monotonic()
    case_root = fresh_root("corrupt")
    results_root = case_root / "results"
    snap_root = case_root / "snaps"
    executor = make_executor(site_ctx, case_root / "cache", results_root)
    for index in (0, 1):
        executed = executor.execute([build_edit(executor, ops[index])])
        if not executed.published:
            # calibration: the sequence must publish every op
            defects.append(f"calibration: op {index} did not publish")
            return report, defects
        executor.save_scenario_state(snap_root / f"op{index:02d}")

    checkpoint_root = results_root / "default" / "checkpoints"
    torn = checkpoint_root / "rev-000002"
    if 2 not in list_checkpoints(checkpoint_root):
        defects.append("calibration: rev-000002 checkpoint absent after op 1")
        return report, defects

    # Variant A: flip one byte inside a thermal tensor payload.
    tensor_path = sorted((torn / "tensors").glob("*.npy"))[0]
    raw = bytearray(tensor_path.read_bytes())
    raw[-1] ^= 0x01
    tensor_path.write_bytes(bytes(raw))
    rejected = 0
    try:
        load_checkpoint(torn)
        defects.append("torn_tensor_accepted: flipped byte loaded cleanly")
    except CheckpointError as error:
        report["tensor_reject"] = str(error)[:160]
        rejected += 1
    latest = load_latest_checkpoint(checkpoint_root)
    if latest is None or latest.scene_revision != 1:
        defects.append(
            "previous_valid_fallback_failed: load_latest returned "
            f"{None if latest is None else latest.scene_revision}, wanted 1"
        )

    # Variant B: truncate the metadata JSON (a genuinely crash-torn file).
    meta_path = torn / "checkpoint.json"
    text = meta_path.read_text()
    meta_path.write_text(text[: len(text) // 2])
    try:
        load_checkpoint(torn)
        defects.append("torn_metadata_accepted: truncated JSON loaded cleanly")
    except CheckpointError as error:
        report["metadata_reject"] = str(error)[:160]
        rejected += 1
    latest = load_latest_checkpoint(checkpoint_root)
    if latest is None or latest.scene_revision != 1:
        defects.append("previous_valid_fallback_failed_after_metadata_tear")

    disk, defect = inspect_disk(results_root, snap_root, expect_torn=frozenset({2}))
    if defect:
        defects.append(f"disk_after_tamper: {defect}")
    report["disk"] = disk
    try:
        restart, defect = restart_and_compare(
            site_ctx,
            case_root,
            disk,
            ops,
            oracle_ops,
            oracle_planes,
            time_steps,
            expect_routing_divergence=(
                "torn newest checkpoint forces the cold split on op 2 "
                "(the design-mandated fallback; science must still match)"
            ),
        )
    except Exception as error:  # noqa: BLE001 - a crash IS the finding
        restart, defect = {}, f"restart_error: {error!r}"
    if defect:
        defects.append(f"restart_after_tamper: {defect}")
    else:
        restart.pop("replayed", None)
        report["restart"] = restart
        if not restart.get("ops_replayed"):
            defects.append("restart_after_tamper: replayed zero ops")
    report["torn_rejections"] = rejected
    report["seconds"] = round(time.monotonic() - started, 1)
    return report, defects


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def run_kill_case(
    name: str,
    site_ctx: SiteContext,
    ops: list[dict[str, Any]],
    oracle_ops: list[dict[str, Any]],
    oracle_planes: dict[str, list[np.ndarray]],
    time_steps: int,
    kill: dict[str, Any],
) -> dict[str, Any]:
    """One kill point: victim subprocess(es) -> disk inspection ->
    restart -> bitwise comparison. Never raises; verdicts ride the case."""
    started = time.monotonic()
    case: dict[str, Any] = {
        "name": name,
        "kill_op": kill["kill_op"],
        "seam": kill["seam"],
        "param": kill.get("param"),
        "verdict": "FAIL",
        "defect": None,
    }
    case_root = fresh_root(name)
    spec = {
        "site_json": json.dumps(
            {
                "root": str(site_ctx.root),
                "site_dir": str(site_ctx.site_dir),
                "rows": site_ctx.grid.rows,
                "cols": site_ctx.grid.cols,
                "pixel": site_ctx.grid.pixel_size_m,
                "origin_x": site_ctx.grid.origin_x_m,
                "origin_y": site_ctx.grid.origin_y_m,
            }
        ),
        "results_root": str(case_root / "results"),
        "snap_root": str(case_root / "snaps"),
        "ops": ops,
        "kill_op": kill["kill_op"],
        "seam": kill["seam"],
        "param": kill.get("param"),
        "cache_dir": str(case_root / "cache_child"),
    }
    spec_path = case_root / "child_spec.json"
    spec_path.write_text(json.dumps(spec))
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--child", str(spec_path)],
        capture_output=True,
        text=True,
        timeout=1200,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
    )
    case["child_rc"] = proc.returncode
    case["child_stderr_tail"] = proc.stderr.strip().splitlines()[-3:]
    fired = proc.returncode == KILL_EXIT and KILL_MARKER in proc.stderr
    case["kill_fired"] = fired
    marker = ""
    for line in proc.stderr.splitlines():
        if line.startswith(KILL_MARKER):
            marker = line[len(KILL_MARKER):]
    case["marker"] = marker
    if not fired:
        case["defect"] = (
            "kill_did_not_fire" if proc.returncode in (0, NOOP_EXIT) else "child_error"
        )
        case["seconds"] = round(time.monotonic() - started, 1)
        return case

    second = kill.get("second_kill")
    if second:
        # The two-crash chain: a SECOND fresh process restores from the
        # newest snapshot and dies rewriting the revision child 1 already
        # checkpointed (the rmtree-then-rename window only opens there).
        restore = None
        snaps = sorted(
            path for path in (case_root / "snaps").iterdir() if path.is_dir()
        )
        if snaps:
            restore = str(snaps[-1])
        spec2 = dict(spec)
        spec2.update(
            {
                "kill_op": second["kill_op"],
                "seam": second["seam"],
                "param": second.get("param"),
                "restore_snapshot": restore,
                "start_op": second["kill_op"],
                "cache_dir": str(case_root / "cache_child2"),
            }
        )
        spec2_path = case_root / "child2_spec.json"
        spec2_path.write_text(json.dumps(spec2))
        proc2 = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--child",
                str(spec2_path),
            ],
            capture_output=True,
            text=True,
            timeout=1200,
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        )
        case["child2_rc"] = proc2.returncode
        fired2 = proc2.returncode == KILL_EXIT and KILL_MARKER in proc2.stderr
        case["kill2_fired"] = fired2
        if not fired2:
            case["defect"] = (
                "second_kill_did_not_fire"
                if proc2.returncode in (0, NOOP_EXIT)
                else "child2_error"
            )
            case["seconds"] = round(time.monotonic() - started, 1)
            return case

    disk, defect = inspect_disk(case_root / "results", case_root / "snaps")
    case["disk"] = disk
    if defect:
        case["defect"] = defect
        case["seconds"] = round(time.monotonic() - started, 1)
        return case

    try:
        restart, defect = restart_and_compare(
            site_ctx,
            case_root,
            disk,
            ops,
            oracle_ops,
            oracle_planes,
            time_steps,
            expect_routing_divergence=None,
        )
    except Exception as error:  # noqa: BLE001 - a crash IS the finding
        restart, defect = {}, f"restart_error: {error!r}"
    if defect:
        case["defect"] = defect
    else:
        case["restart"] = {
            key: value for key, value in restart.items() if key != "replayed"
        }
        case["verdict"] = "PASS"
    case["seconds"] = round(time.monotonic() - started, 1)
    return case


def run_sweep(profile_name: str) -> dict[str, Any]:
    """One sweep: build the site once, run the uninterrupted oracle, then
    every kill case and the corruption case. Returns the evidence dict."""
    profile = PROFILES[profile_name]
    PROOF_ROOT.mkdir(parents=True, exist_ok=True)
    sweep: dict[str, Any] = {
        "harness": "g22_crash_restart",
        "profile": profile_name,
        "cases": [],
        "totals": {
            "cases": 0,
            "passed": 0,
            "failed": 0,
            "defects": [],
            "warm_after_restart_cases": 0,
            "torn_rejections": 0,
            "staged_tmp_observed": 0,
            "dot_staging_observed": 0,
            "mid_series_confirmed": 0,
            "cells_bitwise_compared": 0,
        },
    }

    site_ctx = build_site()
    first_ops = build_ops(site_ctx, profile["seeds"][0])
    time_steps = 4

    # -- the uninterrupted oracle --------------------------------------
    oracle_root = fresh_root("oracle")
    executor = make_executor(site_ctx, oracle_root / "cache", oracle_root / "results")
    oracle_ops: list[dict[str, Any]] = []
    for index, op in enumerate(first_ops):
        executed = executor.execute([build_edit(executor, op)])
        if not executed.published:
            sweep["totals"]["defects"].append(
                f"oracle_calibration: op {index} ({op['id']}) did not publish: "
                f"{executed.status} {executed.diagnostics}"
            )
            return sweep
        oracle_ops.append(op_diag(executed))
        executor.save_scenario_state(oracle_root / "snaps" / f"op{index:02d}")
    oracle_planes = final_serves(executor, time_steps)
    sweep["oracle_ops"] = oracle_ops
    warm_oracle = any(
        (op.get("met_warm_resume_step") or 0) > 0 for op in oracle_ops
    )
    if not warm_oracle:
        sweep["totals"]["defects"].append(
            "oracle_calibration: no op warmed from an anchor -- the "
            "sequence no longer exercises the warm path"
        )
        return sweep

    for seed in profile["seeds"]:
        ops = first_ops if seed == profile["seeds"][0] else build_ops(site_ctx, seed)
        if seed != profile["seeds"][0]:
            # The oracle is per-seed: re-run it for the perturbed values.
            oracle_root = fresh_root(f"oracle_{seed}")
            executor = make_executor(
                site_ctx, oracle_root / "cache", oracle_root / "results"
            )
            oracle_ops = []
            for index, op in enumerate(ops):
                executed = executor.execute([build_edit(executor, op)])
                if not executed.published:
                    sweep["totals"]["defects"].append(
                        f"oracle_calibration(seed {seed}): op {index} failed"
                    )
                    return sweep
                oracle_ops.append(op_diag(executed))
                executor.save_scenario_state(
                    oracle_root / "snaps" / f"op{index:02d}"
                )
            oracle_planes = final_serves(executor, time_steps)
            sweep.setdefault("oracle_ops_by_seed", {})[str(seed)] = oracle_ops

        for kill in profile["cases"]:
            name = kill["name"] if seed == profile["seeds"][0] else f"{kill['name']}_s{seed}"
            print(f"  [case] {name} (op {kill['kill_op']}, {kill['seam']}) ...")
            case = run_kill_case(
                name,
                site_ctx,
                ops,
                oracle_ops,
                oracle_planes,
                time_steps,
                kill,
            )
            totals = sweep["totals"]
            totals["cases"] += 1
            if case["verdict"] == "PASS":
                totals["passed"] += 1
                restart = case.get("restart", {})
                if restart.get("warm_after_restart"):
                    totals["warm_after_restart_cases"] += 1
                totals["cells_bitwise_compared"] += restart.get(
                    "cells_bitwise_compared", 0
                )
                disk = case.get("disk", {})
                if disk.get("staging_tmp"):
                    totals["staged_tmp_observed"] += 1
                if disk.get("dot_staging_entries"):
                    totals["dot_staging_observed"] += 1
            else:
                totals["failed"] += 1
                totals["defects"].append(f"{name}: {case.get('defect')}")
            if case.get("marker", "").startswith("solve_mid_series"):
                totals["mid_series_confirmed"] += 1
            sweep["cases"].append(case)
            print(
                f"    -> {case['verdict']}"
                + (f" ({case['defect']})" if case.get("defect") else "")
                + f" [{case['seconds']}s]"
            )

        # -- the corruption case (once per seed) -------------------------
        print(f"  [case] corruption_torn_newest (seed {seed}) ...")
        corruption, defects = run_corruption_case(
            site_ctx, ops, oracle_ops, oracle_planes, time_steps
        )
        totals = sweep["totals"]
        totals["cases"] += 1
        if defects:
            totals["failed"] += 1
            totals["defects"].extend(
                f"corruption(seed {seed}): {defect}" for defect in defects
            )
            corruption["verdict"] = "FAIL"
        else:
            totals["passed"] += 1
            corruption["verdict"] = "PASS"
            totals["torn_rejections"] += int(corruption.get("torn_rejections", 0))
            totals["cells_bitwise_compared"] += corruption.get(
                "restart", {}
            ).get("cells_bitwise_compared", 0)
        sweep["cases"].append(corruption)
        print(f"    -> {corruption['verdict']} [{corruption.get('seconds', 0)}s]")

    # -- vacuity guard --------------------------------------------------
    totals = sweep["totals"]
    vacuity = []
    if totals["warm_after_restart_cases"] < 1:
        vacuity.append("no restart ever warmed from a disk checkpoint")
    if totals["torn_rejections"] < 2:
        vacuity.append("fewer than two torn-checkpoint rejections observed")
    if totals["staged_tmp_observed"] < 1:
        vacuity.append("no staged rev-*.tmp observed at a checkpoint kill")
    if totals["mid_series_confirmed"] < 1:
        vacuity.append("no mid-series kill confirmed")
    if totals["dot_staging_observed"] < 1:
        vacuity.append("no staged patch directory observed at a pre-publish kill")
    totals["vacuous"] = bool(vacuity)
    totals["vacuity_reasons"] = vacuity
    return sweep


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        default=os.environ.get("G22_SWEEP_SCALE", "thorough"),
        choices=sorted(PROFILES),
    )
    parser.add_argument("--child", metavar="SPEC_JSON")
    args = parser.parse_args(argv)
    if args.child:
        return run_child(args.child)
    started = time.monotonic()
    sweep = run_sweep(args.profile)
    sweep["seconds_total"] = round(time.monotonic() - started, 1)
    PROOF_ROOT.mkdir(parents=True, exist_ok=True)
    EVIDENCE_JSON.write_text(json.dumps(sweep, indent=2, sort_keys=True))
    ok = (
        not sweep["totals"]["defects"]
        and not sweep["totals"]["vacuous"]
        and sweep["totals"]["failed"] == 0
    )
    print(
        f"[g22] profile={args.profile} cases={sweep['totals']['cases']} "
        f"passed={sweep['totals']['passed']} failed={sweep['totals']['failed']} "
        f"warm_restarts={sweep['totals']['warm_after_restart_cases']} "
        f"torn_rejects={sweep['totals']['torn_rejections']} "
        f"cells={sweep['totals']['cells_bitwise_compared']} "
        f"vacuous={sweep['totals']['vacuous']} "
        f"total_s={sweep['seconds_total']}"
    )
    if sweep["totals"]["defects"]:
        print("[g22] DEFECTS:")
        for defect in sweep["totals"]["defects"]:
            print(f"    - {defect}")
    print(f"[g22] evidence: {EVIDENCE_JSON}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
