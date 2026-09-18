# SPDX-License-Identifier: GPL-3.0-only
"""T08 radiation bench: stage-fusion ablation + full time-loop wall time.

Modes (torch allowed HERE only; runtime modules stay torch-free):
  loop    --engine mirror|fused [--reps N] [--clear-cache]
        Full 24-timestep loop over the capture bundle (state threaded),
        n reps, median + min/max spread, per-timestep medians. Cold JIT
        = first invocation after --clear-cache in a fresh process.
  oracle-stage
        Live pinned-oracle solve (read-only tree): exclusive
        Solweig_2022a_calc (radiation) wall time + full solve wall +
        host load. The "original" column of the ablation.
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import time
from pathlib import Path

REPO = Path("/Users/alansynn/Workspace/solweig")
sys.path.insert(0, str(REPO))

import numpy as np

ART = Path("/Users/alansynn/Workspace/solweig_ultrafast_artifacts/t08")
CAP = ART / "capture"
ORACLE_TREE = Path("/Users/alansynn/Workspace/solweig_oracle_e0d19fc")


def run_loop(engine: str, reps: int) -> dict:
    from solweig_core.numba_cpu import radiation as rad

    st = rad.rad_static_from_capture(CAP)
    manifest = json.loads((CAP / "manifest.json").read_text())
    n_t = manifest["n_timesteps"]
    fn = (rad.mirror_radiation_timestep if engine == "mirror"
          else rad.fused_radiation_timestep)

    # warm-load bundles once (npz reads are not part of the measurement)
    bundles = [rad.rad_bundle_from_capture(CAP, t) for t in range(n_t)]
    init_state = rad.rad_state_from_capture(CAP, 0, rows=st.rows,
                                            cols=st.cols)

    first_wall = None
    per_t_totals = []
    rep_walls = []
    for rep in range(reps):
        t_walls = []
        t0 = time.perf_counter()
        state = init_state
        for t in range(n_t):
            ts = time.perf_counter()
            _, state = fn(st, bundles[t], state)
            t_walls.append(time.perf_counter() - ts)
        wall = time.perf_counter() - t0
        if rep == 0:
            first_wall = wall
            per_t_totals = t_walls
        rep_walls.append(wall)
    day_walls = [per_t_totals[t] for t in range(n_t)
                 if bundles[t].is_day]
    return {
        "engine": engine,
        "reps": reps,
        "loop_wall_s": rep_walls,
        "loop_wall_median_s": statistics.median(rep_walls),
        "loop_wall_min_s": min(rep_walls),
        "loop_wall_max_s": max(rep_walls),
        "first_call_wall_s": first_wall,
        "per_t_median_day_s": statistics.median(day_walls),
        "per_t_max_day_s": max(day_walls),
        "host_load": __import__("os").getloadavg(),
    }


def run_oracle_stage() -> dict:
    tree = str(ORACLE_TREE.resolve())
    if tree not in sys.path:
        sys.path.insert(0, tree)
    import solweig_gpu.solweig as sw
    import solweig_gpu.utci_process as up
    from solweig_gpu.incremental.cache import SiteCache
    from solweig_gpu.incremental.geometry import RasterGrid, RasterWindow
    from solweig_gpu.incremental.solver import (
        compose_full_scene_tensors, read_window_for_write_window, solve_window,
    )
    from solweig_gpu.incremental.trees import TreeLayer
    from solweig_gpu.incremental.worker import ExactWorker

    # exclusive radiation timing: wrap Solweig_2022a_calc (the per-t
    # radiation chain; march outputs arrive as precomputed_shadows)
    orig = up.Solweig_2022a_calc
    acc = {"radiation_s": 0.0, "n_calls": 0}

    def timed(*a, **k):
        t0 = time.perf_counter()
        r = orig(*a, **k)
        acc["radiation_s"] += time.perf_counter() - t0
        acc["n_calls"] += 1
        return r

    up.Solweig_2022a_calc = timed

    cache = SiteCache.load(ORACLE_TREE / "site-cache" / "site_500")
    grid = RasterGrid(cache.rows, cache.cols, cache.pixel_size_m,
                      cache.manifest.origin_x_m, cache.manifest.origin_y_m)
    layer = TreeLayer(cache.tree_base, grid)
    worker = ExactWorker(cache, layer,
                         site_dir=ORACLE_TREE / "Input_subset" / "processed_inputs",
                         results_root=ART / "scratch_results_bench",
                         selected_date_str="2009-08-11")
    forcing = worker.forcing()
    scene = compose_full_scene_tensors(cache, layer)
    write = RasterWindow(0, cache.rows, 0, cache.cols)
    read = read_window_for_write_window(write, cache, scene, forcing)

    t0 = time.perf_counter()
    solve_window(cache, layer, read_window=read, write_window=write,
                 forcing=forcing, requested_variables=("utci", "tmrt", "shadow"),
                 stage_timings={}, scene=scene)
    solve_s = time.perf_counter() - t0
    up.Solweig_2022a_calc = orig
    return {
        "oracle_full_solve_s": solve_s,
        "oracle_radiation_stage_s": acc["radiation_s"],
        "oracle_radiation_calls": acc["n_calls"],
        "oracle_nonradiation_s": solve_s - acc["radiation_s"],
        "host_load": __import__("os").getloadavg(),
    }


def clear_numba_cache() -> None:
    pycache = REPO / "solweig_core" / "numba_cpu" / "__pycache__"
    if pycache.exists():
        for f in pycache.glob("radiation.*.nbi"):
            f.unlink()
        for f in pycache.glob("radiation.*.nbc"):
            f.unlink()
    print("numba cache for solweig_core.numba_cpu.radiation cleared")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_loop = sub.add_parser("loop")
    p_loop.add_argument("--engine", default="fused",
                        choices=("mirror", "fused"))
    p_loop.add_argument("--reps", type=int, default=5)
    p_loop.add_argument("--clear-cache", action="store_true")
    sub.add_parser("oracle-stage")
    args = ap.parse_args()

    if args.cmd == "loop":
        if args.clear_cache:
            clear_numba_cache()
        res = run_loop(args.engine, args.reps)
    else:
        res = run_oracle_stage()
    print(json.dumps(res, indent=2))
    out = ART / f"bench_{args.cmd}_{getattr(args, 'engine', 'oracle')}.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"written {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
