# SPDX-License-Identifier: GPL-3.0-only
"""T11 bench: full-domain fallback solve + warm replay lane, n>=3 medians.

Writes (never only /tmp):
  t11/bench_full_solve.json      — raw per-run timings, host load, env
  t11/bench_costs.json           — the router's MeasuredCosts table

Usage: .venv python tests/ultrafast/bench_full_solve.py [--runs 3]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

WT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WT))
import numpy as np  # noqa: E402

from solweig_core.numba_cpu import full_solve as fs  # noqa: E402
from solweig_core.numba_cpu import radiation  # noqa: E402
from solweig_core.numba_cpu import thermal  # noqa: E402

CAP = Path("/Users/alansynn/Workspace/solweig_ultrafast_artifacts/t08/capture")
ART = Path("/Users/alansynn/Workspace/solweig_ultrafast_artifacts/t11")
LATITUDE = 30.312645
PROFILE = "site_500-default"


def bench_full_solve(rows: int, cols: int) -> dict:
    cs = fs.CaptureSet(cap_dir=CAP, profile=PROFILE, latitude=LATITUDE)
    res = fs.full_solve_capture(cs, None, profile=PROFILE, rows=rows,
                                cols=cols, publish_anchor=False)
    return res


def bench_replay_lane(rows: int, cols: int) -> float:
    """T10 warm lane cost/step: bundle load + fused advance (no met
    recompute — warm replay consumes the frozen bundles)."""
    st = radiation.rad_static_from_capture(CAP)
    n = json.loads((CAP / "manifest.json").read_text())["n_timesteps"]
    state = radiation.rad_state_from_capture(CAP, 0, rows=rows, cols=cols)
    t0 = time.perf_counter()
    for t in range(n):
        t_in = radiation.rad_bundle_from_capture(CAP, t)
        _, state = radiation.fused_radiation_timestep(st, t_in, state)
    return (time.perf_counter() - t0) * 1e3 / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()
    man = json.loads((CAP / "manifest.json").read_text())
    rows = int(man.get("rows", 500))
    cols = int(man.get("cols", 500))
    n_t = int(man["n_timesteps"])

    # warm-up (numba JIT + page cache) — never counted
    warm = bench_full_solve(rows, cols)
    out_bytes = sum(
        int(np.asarray(v).nbytes) for v in warm.outputs[-1].values()
        if isinstance(v, np.ndarray))

    full_runs = []
    for i in range(args.runs):
        t0 = time.perf_counter()
        res = bench_full_solve(rows, cols)
        wall = (time.perf_counter() - t0) * 1e3
        full_runs.append({
            "run": i,
            "wall_ms": wall,
            "per_step_median_ms": statistics.median(res.timings_ms),
            "per_step_min_ms": min(res.timings_ms),
            "per_step_max_ms": max(res.timings_ms),
            "loadavg": list(os.getloadavg()),
        })
    replay_runs = []
    for i in range(args.runs):
        replay_runs.append({
            "run": i,
            "per_step_ms": bench_replay_lane(rows, cols),
            "loadavg": list(os.getloadavg()),
        })

    full_median = statistics.median(r["per_step_median_ms"]
                                    for r in full_runs)
    replay_median = statistics.median(r["per_step_ms"]
                                      for r in replay_runs)
    report = {
        "capture": str(CAP),
        "rows": rows, "cols": cols, "n_timesteps": n_t,
        "runs": args.runs,
        "full_solve_runs": full_runs,
        "replay_lane_runs": replay_runs,
        "full_ms_per_step_median": full_median,
        "replay_ms_per_step_median": replay_median,
        "output_bytes_per_t": out_bytes,
        "host_load_at_start": list(os.getloadavg()),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "bench_full_solve.json").write_text(json.dumps(
        report, indent=2))
    (ART / "bench_costs.json").write_text(json.dumps({
        "full_ms_per_step": full_median,
        "sparse_ms_per_step": replay_median,
        "output_bytes_per_t": out_bytes,
    }, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
