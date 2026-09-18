# SPDX-License-Identifier: GPL-3.0-only
"""T09 UTCI bench: dense vs sparse numba kernels vs the original torch
stage, site_500-shaped extents (n_valid = 209422).

Modes (torch allowed HERE only; runtime modules stay torch-free —
bench_radiation.py precedent):
  kernels [--reps N] [--timesteps 0 5 12 23]
        Per-timestep wall of: dense kernel (full (500,500) planes,
        masked-select compact + compute + scatter), sparse kernel
        (pre-compacted 1-D vectors — the R6 pipeline shape), and the
        original torch utci_calculator on the same tensors. reps >= 5,
        median + min/max. Cold JIT = first-call wall in a fresh process
        (T14a flipped the family to cache=True, so a fresh process with a
        warm NUMBA_CACHE_DIR replays instead of recompiling; a genuinely
        cold cache dir still pays the first compile once).
  oracle-stage
        Live recompute_utci_steps wall on the captured timesteps (the
        original UTCI stage this task replaces) for the non-radiation
        budget comparison with T08.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

REPO = Path("/Users/alansynn/Workspace/solweig")
sys.path.insert(0, str(REPO))

import numpy as np

ART = Path("/Users/alansynn/Workspace/solweig_ultrafast_artifacts")
T09 = ART / "t09"
ORACLE_TREE = Path("/Users/alansynn/Workspace/solweig_oracle_e0d19fc")
SITE = ORACLE_TREE / "site-cache/site_500"
CAP = ART / "t08/capture"


def site_inputs(ts_list):
    from solweig_core.numba_cpu import utci as nb

    met = np.loadtxt(SITE / "metfiles/metfile_0_0.txt", skiprows=1,
                     dtype=np.float64)
    bd = np.load(SITE / "static/building_dsm.f32.npy")
    dem = np.load(SITE / "static/dem.f32.npy")
    buildings = nb.buildings_from_dsm(bd, dem)
    steps = []
    for ts in ts_list:
        plane = np.ascontiguousarray(
            np.load(CAP / f"t{ts:02d}.npz")["ret_Tmrt"], dtype=np.float32)
        steps.append((ts, plane, met[ts, 11], met[ts, 10], met[ts, 9]))
    return buildings, steps


def run_kernels(reps: int, ts_list) -> dict:
    import torch

    from solweig_gpu.calculate_utci import utci_calculator
    from solweig_core.numba_cpu import utci as nb

    torch.set_num_threads(8)
    buildings, steps = site_inputs(ts_list)

    rows_out = {}
    first_calls = {}
    for ts, plane, ta, rh, ws in steps:
        t_ta = torch.tensor(np.broadcast_to(
            np.float32(ta), plane.shape).copy())
        t_rh = torch.tensor(np.broadcast_to(
            np.float32(rh), plane.shape).copy())
        t_ws = torch.tensor(np.clip(
            np.full(plane.shape, np.float32(ws), dtype=np.float32),
            0.15, None))
        t_tm = torch.tensor(plane)
        tv, rv, mv, xv = nb.compact_valid(
            t_ta.numpy(), t_rh.numpy(), plane, t_ws.numpy())

        walls = {"dense": [], "sparse": []}
        walls["torch"] = []
        for rep in range(reps):
            t0 = time.perf_counter()
            dense = nb.utci_met_step(plane, ta, rh, ws, uhii_i=0.0,
                                     torch_threads=8)
            walls["dense"].append(time.perf_counter() - t0)
            if rep == 0:
                first_calls.setdefault("dense_cold_jit_s",
                                       walls["dense"][0])

            t0 = time.perf_counter()
            sparse = nb.utci_calculator_sparse(tv, rv, mv, xv,
                                               torch_threads=8)
            walls["sparse"].append(time.perf_counter() - t0)
            if rep == 0:
                first_calls.setdefault("sparse_cold_jit_s",
                                       walls["sparse"][0])

            t0 = time.perf_counter()
            t_out = utci_calculator(t_ta, t_rh, t_tm, t_ws)
            walls["torch"].append(time.perf_counter() - t0)
            del t_out
        rows_out[f"t{ts}"] = {
            k: {
                "median_s": statistics.median(v),
                "min_s": min(v),
                "max_s": max(v),
                "all_s": v,
            } for k, v in walls.items()
        }

    med = {
        k: statistics.median(rows_out[f"t{ts}"][k]["median_s"]
                             for ts in ts_list)
        for k in ("dense", "sparse", "torch")
    }
    return {
        "reps": reps,
        "n_compacted_lanes": int(tv.size),
        "per_timestep": rows_out,
        "median_per_step": med,
        "speedup_vs_torch": {k: med["torch"] / med[k]
                             for k in ("dense", "sparse")},
        **first_calls,
        "host_load": os.getloadavg(),
    }


def run_oracle_stage(reps: int, ts_list) -> dict:
    import torch

    from solweig_gpu.utci_process import recompute_utci_steps

    torch.set_num_threads(8)
    met = np.loadtxt(SITE / "metfiles/metfile_0_0.txt", skiprows=1,
                     dtype=np.float64)
    bd = np.load(SITE / "static/building_dsm.f32.npy")
    dem = np.load(SITE / "static/dem.f32.npy")
    planes = {ts: np.ascontiguousarray(
        np.load(CAP / f"t{ts:02d}.npz")["ret_Tmrt"], dtype=np.float32)
        for ts in ts_list}
    walls = []
    for _ in range(reps):
        t0 = time.perf_counter()
        recompute_utci_steps(met, list(ts_list), planes, bd, dem)
        walls.append(time.perf_counter() - t0)
    return {
        "reps": reps,
        "timesteps": list(ts_list),
        "median_s": statistics.median(walls),
        "min_s": min(walls),
        "max_s": max(walls),
        "all_s": walls,
        "host_load": os.getloadavg(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_k = sub.add_parser("kernels")
    p_k.add_argument("--reps", type=int, default=7)
    p_k.add_argument("--timesteps", type=int, nargs="+",
                     default=[0, 5, 12, 23])
    p_o = sub.add_parser("oracle-stage")
    p_o.add_argument("--reps", type=int, default=5)
    p_o.add_argument("--timesteps", type=int, nargs="+",
                     default=[0, 5, 12, 23])
    args = ap.parse_args()

    if args.cmd == "kernels":
        res = run_kernels(args.reps, tuple(args.timesteps))
        out = T09 / f"bench_kernels_r{args.reps}.json"
    else:
        res = run_oracle_stage(args.reps, tuple(args.timesteps))
        out = T09 / f"bench_oracle_stage_r{args.reps}.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(f"written {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
