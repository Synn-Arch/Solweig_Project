# SPDX-License-Identifier: GPL-3.0-only
"""T16 CPU-lane startup measurements (yaml ``startup_ms`` rows).

* ``process_to_ready`` — cpu_aot_prebuilt_site_process_to_ready_p95_max
  (3000 ms), COLD and WARM numba disk cache, >=5 samples each. Reuses the
  T14a protocol verbatim (``tests/ultrafast/bench_startup_t14a.py``) with
  T14A_RUNS raised to 6; p95 is recomputed here over the raw per-run rows
  (the T14a script itself reports medians only).
* ``unseen_signature`` — jit_compile_and_unseen_signature_latency (report
  required): first ``runtime.step_table`` call on an angle/scale unseen
  anywhere in the frozen diets, in a fresh process (cold) vs a process
  that already built one table (warm).
* ``cold_first_edit`` — cold_first_edit_end_to_end (report required): one
  selected-time edit job through the REAL JobRunner path in a fresh
  subprocess (torch import + JIT included).
* ``site_cache_build`` — site_cache_build (report required): recorded
  evidence — the t08 capture-regeneration wall (its manifest) and the T00
  oracle full-domain baseline solve (raw_runs.jsonl). The SVF-dominated
  site-cache preprocessing has no one-command builder in this tree; that
  gap is disclosed, not papered over.
* ``server_lane_p2r`` — INFORMATIONAL: import of the server job stack +
  SiteCache.load(site_500) in a fresh process (the torch lane's
  process-to-ready shape; outside the cpu_aot target's scope).

Usage:
    .venv/bin/python benchmarks/ultrafast/t16_startup.py \
        [--artifacts-dir DIR] [--runs 6]
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t16_cpu_lane import (  # noqa: E402
    certified_source_hashes, env_record, hw_record, percentile_checked,
    _utcnow)

REPO_ROOT = Path(__file__).resolve().parents[2]
T14A_BENCH = REPO_ROOT / "tests" / "ultrafast" / "bench_startup_t14a.py"
T00_RAW_RUNS = REPO_ROOT / "benchmarks" / "ultrafast" / "baseline" / \
    "raw_runs.jsonl"
CAPTURE_MANIFEST = Path(
    "/Users/alansynn/Workspace/solweig_ultrafast_artifacts/t08/capture/"
    "manifest.json")

_UNSEEN_RUNNER = r'''
import json, sys, time
sys.path.insert(0, {root!r})
t0 = time.perf_counter()
from solweig_core import runtime as rt
t_import = time.perf_counter() - t0
t0 = time.perf_counter()
tab = rt.step_table("svf_shadow", 61.0, 30.0, 1.0 / 7.5, 32, 48, 12.0)
t_first = time.perf_counter() - t0
t0 = time.perf_counter()
tab2 = rt.step_table("svf_shadow", 62.5, 41.25, 1.0 / 7.5, 32, 48, 12.0)
t_second = time.perf_counter() - t0
print("@@START@@" + json.dumps({{
    "import_runtime_s": round(t_import, 4),
    "first_unseen_table_s": round(t_first, 4),
    "second_unseen_table_s": round(t_second, 4),
    "counts": [int(tab.count), int(tab2.count)],
}}))
'''

_SERVER_P2R_RUNNER = r'''
import json, sys, time
sys.path.insert(0, {root!r})
t0 = time.perf_counter()
from solweig_gpu.incremental.cache import SiteCache
from solweig_gpu.server.jobs import JobRunner, RunnerContext, SiteRegistry
t_import = time.perf_counter() - t0
t0 = time.perf_counter()
cache = SiteCache.load({cache!r})
sites = SiteRegistry({{"site_500": {{"cache_dir": {cache_s!r}}}}})
geometry = sites.geometry("site_500")
t_load = time.perf_counter() - t0
print("@@START@@" + json.dumps({{
    "import_server_stack_s": round(t_import, 3),
    "site_cache_load_s": round(t_load, 3),
    "process_to_ready_s": round(t_import + t_load, 3),
    "grid": {{"rows": geometry["rows"], "cols": geometry["cols"]}},
}}))
'''

_COLD_EDIT_RUNNER = r'''
import json, sys, time
sys.path.insert(0, {root!r}); sys.path.insert(0, {bench!r})
from pathlib import Path
from t16_cpu_lane import bench_selected_exact
t0 = time.perf_counter()
summary = bench_selected_exact(
    cache_dir=Path({cache!r}), site_dir=Path({site!r}),
    out_dir=Path({out!r}), warmup=0, repeats=1, pairs=0, seed=1)
subprocess_wall = time.perf_counter() - t0
row = json.loads(open(Path({rec!r})).read().splitlines()[0])
print("@@START@@" + json.dumps({{
    "cold_first_edit_e2e_s": round(row["e2e_s"], 3),
    "cold_stage_s": {{k: round(v, 3) for k, v in row["stage_s"].items()}},
    "subprocess_wall_s": round(subprocess_wall, 3),
}}))
'''


def _run_marker(code: str, env: dict | None = None) -> dict:
    import os

    merged = dict(os.environ)
    if env:
        merged.update(env)
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, cwd=str(REPO_ROOT), env=merged)
    if proc.returncode != 0:
        raise RuntimeError(f"startup subprocess failed: {proc.stderr[-2000:]}")
    line = [ln for ln in proc.stdout.splitlines()
            if ln.startswith("@@START@@")]
    if not line:
        raise RuntimeError(f"no @@START@@ marker: {proc.stdout[-500:]} "
                           f"{proc.stderr[-500:]}")
    return json.loads(line[0][len("@@START@@"):])


def _site_cache_build_evidence() -> dict:
    evidence: dict = {
        "definition": ("offline preprocessing that produced the site cache "
                       "(SVF marches, baseline_results) and the frozen "
                       "capture — excluded from runtime budgets, reported "
                       "per the yaml"),
        "capture_regeneration_s": None,
        "oracle_full_domain_baseline_solve_s": None,
        "svf_preprocess_builder_in_tree": False,
        "disclosure": (
            "no one-command site_500 SVF-preprocessing builder exists in "
            "this tree (the cache predates the ultrafast work); its wall "
            "is therefore not re-measured here. The two recorded walls "
            "below bound the offline cost that IS recorded."),
    }
    try:
        man = json.loads(CAPTURE_MANIFEST.read_text())
        evidence["capture_regeneration_s"] = man.get("solve_wall_s")
        evidence["capture_manifest"] = str(CAPTURE_MANIFEST)
    except (OSError, json.JSONDecodeError):
        pass
    try:
        for line in T00_RAW_RUNS.read_text().splitlines():
            row = json.loads(line)
            if row.get("mode") == "full":
                evidence["oracle_full_domain_baseline_solve_s"] = round(
                    row["timings_s"]["total_s"], 3)
                evidence["oracle_run_id"] = row.get("run_id")
                break
    except (OSError, json.JSONDecodeError):
        pass
    return evidence


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts-dir",
                    default="/Users/alansynn/Workspace/"
                            "solweig_ultrafast_artifacts/t16/cpu")
    ap.add_argument("--runs", type=int, default=6,
                    help="samples per cold/warm scenario (>=5 per yaml)")
    args = ap.parse_args(argv)

    art = Path(args.artifacts_dir)
    art.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "schema_version": 1,
        "task": "T16 startup (cpu lane)",
        "recorded_at_utc": _utcnow(),
        "target_ms": {"cpu_aot_prebuilt_site_process_to_ready_p95_max": 3000},
        "runs_per_scenario": args.runs,
        "hw_at_start": hw_record(),
        "env": env_record(),
        "source_sha256": certified_source_hashes(),
    }

    # -- process-to-ready via the T14a protocol (cold + warm cache) -------
    import os
    import tempfile

    with tempfile.TemporaryDirectory(prefix="t16_t14a_") as td:
        env = dict(os.environ, T14A_ART=str(art), T14A_RUNS=str(args.runs),
                   T14A_TAG="_t16", T14A_ROOT=str(REPO_ROOT))
        proc = subprocess.run(
            [sys.executable, str(T14A_BENCH)], capture_output=True,
            text=True, cwd=str(REPO_ROOT), env=env)
        if proc.returncode != 0:
            raise RuntimeError(f"t14a bench failed: {proc.stderr[-2000:]}")
    t14a_path = art / "t14a" / "timing" / "startup_timing_t16.json"
    t14a = json.loads(t14a_path.read_text())
    p2r: dict = {"driver": "tests/ultrafast/bench_startup_t14a.py "
                           "(T14a protocol, reused verbatim)",
                 "definition": ("process start -> torch-free CPU stack "
                                "imported + first-call JIT loads (+ capture "
                                "loads for _full) ; p95 recomputed over raw "
                                "per-run rows"),
                 "scenarios": t14a["scenarios"]}
    for arm in ("cold", "warm"):
        rows = t14a["scenarios"][arm]
        for key in ("p2r_jit_s", "p2r_full_s"):
            vals = [r[key] for r in rows if isinstance(r.get(key), float)]
            p2r[f"{arm}_{key}"] = {
                "n": len(vals),
                "p50": round(statistics.median(vals), 4),
                "p95": round(percentile_checked(vals, 95), 4),
                "min": round(min(vals), 4), "max": round(max(vals), 4),
            }
    p2r["verdict_warm_prebuilt_vs_3000ms"] = (
        "PASS" if p2r["warm_p2r_full_s"]["p95"] * 1000.0 <= 3000.0
        else "TARGET_MISSED")
    p2r["note"] = ("the yaml target describes the PREBUILT (shipped numba "
                   "disk cache) arm = 'warm'; the cold arm is the "
                   "first-compile/JIT disclosure")
    report["process_to_ready"] = p2r

    # -- unseen-signature latency -----------------------------------------
    unseen_rows = []
    for i in range(3):
        row = _run_marker(_UNSEEN_RUNNER.format(root=str(REPO_ROOT)))
        row["rep"] = i
        unseen_rows.append(row)
    report["unseen_signature"] = {
        "rows": unseen_rows,
        "first_table_ms_p50": round(statistics.median(
            [r["first_unseen_table_s"] for r in unseen_rows]) * 1e3, 2),
        "second_table_ms_p50": round(statistics.median(
            [r["second_unseen_table_s"] for r in unseen_rows]) * 1e3, 2),
        "note": ("first vs second unseen-angle table in the same fresh "
                 "process: signature-compile cost shows as the gap; "
                 "process-cold JIT is the t14a cold arm above"),
    }

    # -- cold first edit E2E (real job path, fresh subprocess) ------------
    scratch = art / "startup_cold_edit_scratch"
    cold_edit = _run_marker(_COLD_EDIT_RUNNER.format(
        root=str(REPO_ROOT), bench=str(Path(__file__).resolve().parent),
        cache="/Users/alansynn/Workspace/solweig/site-cache/site_500",
        site="/Users/alansynn/Workspace/solweig/Input_subset/"
              "processed_inputs",
        out=str(scratch), rec=str(scratch / "records_raw.jsonl")))
    report["cold_first_edit"] = cold_edit

    # -- server-lane process-to-ready (informational) ----------------------
    report["informational_server_lane_p2r"] = _run_marker(
        _SERVER_P2R_RUNNER.format(
            root=str(REPO_ROOT),
            cache="/Users/alansynn/Workspace/solweig/site-cache/site_500",
            cache_s="/Users/alansynn/Workspace/solweig/site-cache/site_500"))

    # -- site-cache build (recorded evidence + disclosure) -----------------
    report["site_cache_build"] = _site_cache_build_evidence()

    out_path = art / "t16_startup.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    worktree_copy = REPO_ROOT / "benchmarks" / "ultrafast" / "bench" / \
        "t16_startup.json"
    worktree_copy.parent.mkdir(parents=True, exist_ok=True)
    worktree_copy.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "warm_p2r_full_p95_ms":
            round(p2r["warm_p2r_full_s"]["p95"] * 1e3, 1),
        "cold_p2r_full_p95_ms":
            round(p2r["cold_p2r_full_s"]["p95"] * 1e3, 1),
        "cold_first_edit_e2e_s": cold_edit["cold_first_edit_e2e_s"],
        "report": str(out_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
