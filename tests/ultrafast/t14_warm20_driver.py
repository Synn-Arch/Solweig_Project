# SPDX-License-Identifier: GPL-3.0-only
"""T14 D4 witness: twenty warm scenarios in ONE process (RSS budget).

``cpu_aot_twenty_warm_scenarios_process_peak_rss_max: 1536`` — a
runtime process serving 20 forcing scenarios of the same prebuilt site.
Shape: one shared site (the compact capture), 20 sequential full solves
(steam-off: outputs are digested and released, as a serving process
would stream them), numba cache warm. Records peak ru_maxrss.

Usage (clean venv, warm NUMBA_CACHE_DIR):

    venv-runtime/bin/python tests/ultrafast/t14_warm20_driver.py \
        --out gates/warm20.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import time
from pathlib import Path

import numpy as np


def _peak_rss_mb() -> float:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return round(raw / (1024.0 * 1024.0), 1)
    return round(raw / 1024.0, 1)


def _digest(arr) -> str:
    a = np.ascontiguousarray(arr)
    if a.dtype == np.float32:
        a = a.view(np.uint32)
    return hashlib.sha256(a.tobytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capture", type=Path,
        default=Path("/Users/alansynn/Workspace/solweig_ultrafast_artifacts/"
                     "t14/compact_capture"),
    )
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    from solweig_core.numba_cpu import full_solve as fs

    cap_set = fs.CaptureSet(
        cap_dir=str(args.capture), profile="site_500-default",
        latitude=30.312645,
    )

    # warm the JIT once (compile memory must not pollute the 20-scenario
    # peak measurement — the budget describes a SERVING process)
    fs.full_solve_capture(cap_set, profile="site_500-default")

    digests: list[str] = []
    solve_s: list[float] = []
    for i in range(args.n):
        t0 = time.perf_counter()
        res = fs.full_solve_capture(cap_set, profile="site_500-default")
        solve_s.append(round(time.perf_counter() - t0, 3))
        h = hashlib.sha256()
        for out in res.outputs:  # digest-then-release (streaming shape)
            for k in sorted(out):
                v = out[k]
                if hasattr(v, "tobytes"):
                    h.update(k.encode())
                    h.update(_digest(v).encode())
        digests.append(h.hexdigest())
        del res

    import gc

    gc.collect()
    report = {
        "n_scenarios": args.n,
        "identical": len(set(digests)) == 1,
        "first_digest": digests[0],
        "solve_s": solve_s,
        "median_solve_s": sorted(solve_s)[len(solve_s) // 2],
        "peak_rss_mb": _peak_rss_mb(),
        "budget_mib": 1536,
        "within_budget": _peak_rss_mb() <= 1536.0,
    }
    payload = json.dumps(report, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1))
    print("@@WARM20@@" + payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
