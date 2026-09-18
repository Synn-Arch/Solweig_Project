# SPDX-License-Identifier: GPL-3.0-only
"""T10 thermal replay bench: warm vs cold-split vs fresh-full.

Runs the torch-free temporal replay core (solweig_core.numba_cpu.thermal)
over the t08 capture bundle (site_500, 500x500, 24 timesteps) for the
canonical met-edit shapes and records, per route:

  * steps_solved — timesteps actually advanced (the exit criterion; a
    coverage-only marker produces NO reduction and is not a thermal hit);
  * wall seconds, with per-timestep npz bundle reads PRE-LOADED (the
    measurement isolates the replay core, not disk IO);
  * a raw-bit equality spot-check against the capture's own return
    planes (the bench re-verifies what the test suite pins).

Routes (default r0 = 20, the honest late-edit shape):
  fresh-full  cold state@0, collect 0..24            (24 steps)
  cold-split  no anchor: advance 0..20, solve 20..24 (24 steps)
  warm-hit    anchor@20 (the ORACLE's recorded entry state), edit at
              row 21: advance 20..21, solve 21..24    (4 steps)
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np

ART = Path("/Users/alansynn/Workspace/solweig_ultrafast_artifacts/t10")
CAP = Path("/Users/alansynn/Workspace/solweig_ultrafast_artifacts/t08/capture")


def oracle_anchor(t_entry: int):
    """A ThermalState built from the capture's recorded ENTRY state of
    timestep ``t_entry`` (independent of this process's own replay)."""
    from solweig_core.numba_cpu import thermal

    z = np.load(CAP / f"t{t_entry:02d}.npz")
    payload = {
        name: z[f"arg_{name}"] for name in thermal.THERMAL_PLANE_NAMES
    }
    if "arg_CI__0d" in z:
        payload["CI"] = float(np.float32(z["arg_CI__0d"][()]))
    else:
        payload["CI"] = float(np.float32(z["arg_CI__f64"][()]))
    payload["firstdaytime"] = int(z["arg_firstdaytime__int"][()])
    payload["timeadd"] = float(z["arg_timeadd__f64"][()])
    payload["Twater"] = None  # not-established spelling
    payload["next_step"] = int(t_entry)
    return thermal.thermal_state_from_payload(payload)


def _timed(fn, reps: int):
    walls = []
    result = None
    for _ in range(reps):
        t0 = time.perf_counter()
        result = fn()
        walls.append(time.perf_counter() - t0)
    return result, walls


def run(reps: int, r0: int, anchor_t: int) -> dict:
    from solweig_core.numba_cpu import radiation as rad
    from solweig_core.numba_cpu import thermal

    st = rad.rad_static_from_capture(CAP)
    manifest = json.loads((CAP / "manifest.json").read_text())
    n_t = int(manifest["n_timesteps"])
    # bundle reads are PRE-LOADED: the measurement is the replay core
    bundles = [rad.rad_bundle_from_capture(CAP, t) for t in range(n_t)]

    def bundle_fn(t: int):
        return bundles[t]

    cold = thermal.cold_thermal_state(rows=st.rows, cols=st.cols)
    anchor = oracle_anchor(anchor_t)

    fresh, fresh_walls = _timed(
        lambda: thermal.replay_thermal(
            st, bundle_fn, cold, collect_ts=range(n_t), t_stop=n_t
        ),
        reps,
    )
    cold_split, cold_walls = _timed(
        lambda: thermal.split_replay_thermal(
            st, bundle_fn, cold, r0=r0, t_stop=n_t
        ),
        reps,
    )
    # the warm hit: an edit at r0+1 (rows < r0+1 unchanged for the
    # anchor@r0) — the executor's m3 shape (~24 -> ~4 steps)
    warm, warm_walls = _timed(
        lambda: thermal.split_replay_thermal(
            st, bundle_fn, anchor, r0=r0 + 1, t_stop=n_t
        ),
        reps,
    )

    # raw-bit spot check vs the capture's own returns (uint32 views).
    # The warm suffix owns t >= r0+1; t <= r0 is the unchanged prefix
    # the executor serves from the store (out of this kernel's scope).
    mismatch = []
    for t in (r0 + 1, n_t - 1):
        z = np.load(CAP / f"t{t:02d}.npz")
        for name in ("Tmrt", "Kdown", "Ldown"):
            got = warm.outputs[t][name]
            want = z[f"ret_{name}"]
            if (np.ascontiguousarray(got).view(np.uint32)
                    != np.ascontiguousarray(want).view(np.uint32)).any():
                mismatch.append(f"{name}@t{t}")

    def route(result, walls, **extra):
        return {
            "steps_solved": result.steps_solved,
            "wall_median_s": round(statistics.median(walls), 6),
            "wall_min_s": round(min(walls), 6),
            **extra,
        }

    return {
        "grid": f"{st.rows}x{st.cols}",
        "n_timesteps": n_t,
        "r0": r0,
        "anchor_next_step": anchor.next_step,
        "reps": reps,
        "routes": {
            "fresh_full": route(fresh, fresh_walls),
            "cold_split": route(cold_split, cold_walls),
            "warm_hit": route(
                warm, warm_walls, resume_step=anchor.next_step
            ),
        },
        "warm_vs_capture_bit_mismatches": mismatch,
        "steps_reduction": {
            "fresh_minus_warm": (
                fresh.steps_solved - warm.steps_solved
            ),
            "warm_over_fresh": round(
                warm.steps_solved / fresh.steps_solved, 4
            ),
        },
        "host_load": _loadavg(),
    }


def _loadavg():
    try:
        one, five, fifteen = os.getloadavg()
        return [round(one, 2), round(five, 2), round(fifteen, 2)]
    except OSError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--r0", type=int, default=20)
    ap.add_argument("--anchor-t", type=int, default=20)
    args = ap.parse_args()
    record = run(args.reps, args.r0, args.anchor_t)
    ART.mkdir(parents=True, exist_ok=True)
    out = ART / "bench_thermal.json"
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record["routes"], indent=2))
    print(json.dumps(record["steps_reduction"], indent=2))
    print(f"written: {out}")


if __name__ == "__main__":
    main()
