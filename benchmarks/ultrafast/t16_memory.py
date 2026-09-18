# SPDX-License-Identifier: GPL-3.0-only
"""T16 CPU-lane memory measurements (yaml ``memory_mib`` rows).

Subprocess-RSS protocol (T14 shape; ru_maxrss is peak, current RSS via
``ps`` — macOS has no /proc):

* ``steady``       — cpu_aot_single_site_single_scenario_steady_rss_max
                     (256): a serving-shaped process holding ONE site's
                     capture (runtime import + static + one warm solve +
                     gc), reporting CURRENT resident size (steady, not
                     peak) plus its peak for context.
* ``job_peak``     — cpu_aot_representative_job_peak_rss_max (768): peak
                     ru_maxrss of ONE representative job (the 24-step
                     full solve over the site_500 capture; includes
                     resident mmap scratch/checkpoints/threads/code per
                     the yaml's accounting note).
* ``warm20``       — cpu_aot_twenty_warm_scenarios_process_peak_rss_max
                     (1536): twenty sequential full solves in ONE process,
                     digest-then-release streaming shape. Reuses
                     ``tests/ultrafast/t14_warm20_driver.py`` (the T14
                     witness driver) against the REPRESENTATIVE site_500
                     capture — measured, not assumed from T14's compact
                     capture.
* ``server_job``   — INFORMATIONAL: peak RSS of one real selected-time
                     edit job through the JobRunner path (the torch
                     server lane; outside the cpu_aot target's scope,
                     recorded for the release review's completeness).

Usage:
    .venv/bin/python benchmarks/ultrafast/t16_memory.py \
        [--artifacts-dir DIR] (default /Users/alansynn/Workspace/
        solweig_ultrafast_artifacts/t16/cpu)
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t16_cpu_lane import (  # noqa: E402
    CAPTURE_DEFAULT, CAPTURE_LATITUDE, CAPTURE_PROFILE,
    certified_source_hashes, env_record, hw_record, _utcnow)

REPO_ROOT = Path(__file__).resolve().parents[2]
T14_WARM20 = REPO_ROOT / "tests" / "ultrafast" / "t14_warm20_driver.py"

_STEADY_RUNNER = r'''
import json, subprocess, sys, time
sys.path.insert(0, {root!r})
t0 = time.perf_counter()
from solweig_core import runtime as rt          # serving-shape import
from solweig_core.numba_cpu import full_solve as fs
t_import = time.perf_counter() - t0
import gc
# NOTE: rt.Runtime(...).solve() is currently unconstructible upstream —
# Runtime.cap_set() calls CaptureSet without the required `latitude`
# (pre-existing; flagged in the T16 report). The steady measurement uses
# the same stack directly: CaptureSet + one full solve.
cs = fs.CaptureSet(cap_dir={cap!r}, profile={profile!r}, latitude={lat!r})
t0 = time.perf_counter()
res = fs.full_solve_capture(cs, None, profile={profile!r},
                            publish_anchor=False)  # ONE warm solve
t_solve = time.perf_counter() - t0
del res
gc.collect()
def cur_rss_mb():
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(__import__("os").getpid())],
                         capture_output=True, text=True).stdout.strip()
    return round(int(out) / 1024.0, 1) if out else None
import platform, resource
raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
peak = round(raw / (1024.0 * 1024.0), 1) if platform.system() == "Darwin" \
    else round(raw / 1024.0, 1)
print("@@MEM@@" + json.dumps({{
    "import_runtime_s": round(t_import, 3),
    "one_solve_s": round(t_solve, 3),
    "steady_rss_mib": cur_rss_mb(),
    "peak_rss_mib": peak,
    "torch_in_modules": "torch" in sys.modules,
}}))
'''

_JOBPEAK_RUNNER = r'''
import json, platform, resource, sys, time
sys.path.insert(0, {root!r})
from solweig_core.numba_cpu import full_solve as fs
cs = fs.CaptureSet(cap_dir={cap!r}, profile={profile!r},
                   latitude={lat!r})
t0 = time.perf_counter()
res = fs.full_solve_capture(cs, None, profile={profile!r},
                            publish_anchor=False)
wall = time.perf_counter() - t0
raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
peak = round(raw / (1024.0 * 1024.0), 1) if platform.system() == "Darwin" \
    else round(raw / 1024.0, 1)
n_out = sum(1 for out in res.outputs for k in out if hasattr(out[k], "tobytes"))
print("@@MEM@@" + json.dumps({{
    "job": "one_representative_full_solve",
    "wall_s": round(wall, 3), "n_output_planes": n_out,
    "peak_rss_mib": peak,
}}))
'''

_SERVER_JOB_RUNNER = r'''
import json, platform, resource, sys, time
sys.path.insert(0, {root!r}); sys.path.insert(0, {bench!r})
from pathlib import Path
from t16_cpu_lane import bench_selected_exact
t0 = time.perf_counter()
summary = bench_selected_exact(
    cache_dir=Path({cache!r}), site_dir=Path({site!r}),
    out_dir=Path({out!r}), warmup=0, repeats=1, pairs=0, seed=1)
wall = time.perf_counter() - t0
raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
peak = round(raw / (1024.0 * 1024.0), 1) if platform.system() == "Darwin" \
    else round(raw / 1024.0, 1)
print("@@MEM@@" + json.dumps({{
    "job": "one_selected_time_edit_job_real_path",
    "cold_e2e_s": round(summary["e2e_s"]["p50"], 3),
    "subprocess_wall_s": round(wall, 3),
    "peak_rss_mib": peak,
}}))
'''


def _run_marker(code: str) -> dict:
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, cwd=str(REPO_ROOT))
    if proc.returncode != 0:
        raise RuntimeError(f"memory subprocess failed: {proc.stderr[-2000:]}")
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("@@MEM@@")]
    if not line:
        raise RuntimeError(f"no @@MEM@@ marker: {proc.stdout[-500:]} "
                           f"{proc.stderr[-500:]}")
    return json.loads(line[0][len("@@MEM@@"):])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capture", default=str(CAPTURE_DEFAULT))
    ap.add_argument("--site-cache",
                    default="/Users/alansynn/Workspace/solweig/site-cache/site_500")
    ap.add_argument("--site-dir",
                    default="/Users/alansynn/Workspace/solweig/"
                            "Input_subset/processed_inputs")
    ap.add_argument("--artifacts-dir",
                    default="/Users/alansynn/Workspace/"
                            "solweig_ultrafast_artifacts/t16/cpu")
    ap.add_argument("--warm20-n", type=int, default=20)
    args = ap.parse_args(argv)

    art = Path(args.artifacts_dir)
    art.mkdir(parents=True, exist_ok=True)
    capture = Path(args.capture)
    report: dict = {
        "schema_version": 1,
        "task": "T16 memory (cpu lane)",
        "recorded_at_utc": _utcnow(),
        "capture": str(capture),
        "protocol": ("subprocess RSS; ru_maxrss=peak (macOS reports bytes), "
                     "current resident via ps; accounting includes resident "
                     "mmap scratch/checkpoints/threads/code per yaml"),
        "targets_mib": {
            "steady_rss_max": 256,
            "representative_job_peak_rss_max": 768,
            "twenty_warm_scenarios_process_peak_rss_max": 1536,
        },
        "hw_at_start": hw_record(),
        "env": env_record(),
        "source_sha256": certified_source_hashes(),
    }

    steady_code = _STEADY_RUNNER.format(root=str(REPO_ROOT), cap=str(capture),
                                        profile=CAPTURE_PROFILE,
                                        lat=CAPTURE_LATITUDE)
    report["steady"] = _run_marker(steady_code)
    report["steady"]["verdict"] = (
        "PASS" if (report["steady"]["steady_rss_mib"] or 1e9) <= 256
        else "TARGET_MISSED")

    jobpeak_code = _JOBPEAK_RUNNER.format(root=str(REPO_ROOT),
                                          cap=str(capture),
                                          profile=CAPTURE_PROFILE,
                                          lat=CAPTURE_LATITUDE)
    report["representative_job_peak"] = _run_marker(jobpeak_code)
    report["representative_job_peak"]["verdict"] = (
        "PASS" if report["representative_job_peak"]["peak_rss_mib"] <= 768
        else "TARGET_MISSED")

    warm20_out = art / "warm20_site500.json"
    # the T14 witness driver imports solweig_core without a path insert
    # (T14 ran it from an env where the repo was importable); supply the
    # repo root explicitly rather than editing the shared witness file
    import os

    warm20_env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    proc = subprocess.run(
        [sys.executable, str(T14_WARM20), "--capture", str(capture),
         "--n", str(args.warm20_n), "--out", str(warm20_out)],
        capture_output=True, text=True, cwd=str(REPO_ROOT), env=warm20_env)
    if proc.returncode != 0:
        raise RuntimeError(f"warm20 driver failed: {proc.stderr[-2000:]}")
    warm20 = json.loads(warm20_out.read_text())
    warm20["verdict"] = ("PASS" if warm20["within_budget"]
                         else "TARGET_MISSED")
    warm20["note"] = ("representative site_500 capture (measured; T14's "
                      "original witness used the compact capture)")
    report["twenty_warm_scenarios"] = warm20

    server_code = _SERVER_JOB_RUNNER.format(
        root=str(REPO_ROOT), bench=str(Path(__file__).resolve().parent),
        cache=args.site_cache, site=args.site_dir,
        out=str(art / "server_job_scratch"))
    report["informational_server_job_peak"] = _run_marker(server_code)
    report["informational_server_job_peak"]["note"] = (
        "torch server lane (JobRunner real path); NOT the cpu_aot target's "
        "scope — recorded for release-review completeness")

    out_path = art / "t16_memory.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    worktree_copy = REPO_ROOT / "benchmarks" / "ultrafast" / "bench" / \
        "t16_memory.json"
    worktree_copy.parent.mkdir(parents=True, exist_ok=True)
    worktree_copy.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "steady_rss_mib": report["steady"]["steady_rss_mib"],
        "job_peak_mib": report["representative_job_peak"]["peak_rss_mib"],
        "warm20_peak_mib": warm20["peak_rss_mb"],
        "server_job_peak_mib":
            report["informational_server_job_peak"]["peak_rss_mib"],
        "report": str(out_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
