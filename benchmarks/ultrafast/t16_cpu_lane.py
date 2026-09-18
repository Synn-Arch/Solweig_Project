# SPDX-License-Identifier: GPL-3.0-only
"""T16 CPU-lane measurement library (TASKS.ko.md §T16).

The factorized ablation + final workload matrix for the CPU lane, run
against ``docs/incremental_design_tool/ultrafast_bitwise/acceptance_targets.yaml``
(targets are NEVER relaxed here — this module only measures).

Variants (invoked via ``run.py bench``):
  cpu-numba-selected-exact    one selected-time EDIT JOB through the REAL
                              JobRunner/Store path (geometry edit E1 at the
                              representative site_500 fixture, requested
                              ``time_indices=[12]``), accept -> exact result
                              published -> client decode+apply. In-process:
                              HTTP transport is out of the CPU lane's scope
                              (network_scope local_test_transport); every
                              stage that IS in-process is timed exclusively.
  cpu-numba-full-day          24-timestep full solve over the frozen
                              site_500 capture — cold (fresh subprocess,
                              shipped numba disk cache) and warm (in-process
                              repeats), per cpu_full_day_local.
  cpu-numba-full-recompute-warm
                              warm-process FULL 24-step recompute (the
                              idle-reconcile / fallback full-recompute
                              path), per cpu_full_recompute_warm. The
                              anchor-resume suffix replay (T14 W5 shape)
                              is recorded as an auxiliary row, never as the
                              primary target number.

Honesty scaffolding (the harness must not be blind to its own
mismeasurement):
  * :func:`assert_exclusive_stage_sum` — exclusive stage timings must sum
    to the measured E2E (a double-counted stage aborts the run).
  * :func:`percentile_checked` / :func:`summarize_samples` — p50/p95 are
    computed ONLY over ``phase == "sample"`` rows and the sample count is
    asserted against the requested minimum (a p95 over warmup-only rows
    aborts).
  * Every run record carries the hardware affinity/load disclosure and the
    certified sha256 of the scientific sources that executed.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The frozen site_500 capture (T08) — the representative CPU-lane capture.
CAPTURE_DEFAULT = Path(
    "/Users/alansynn/Workspace/solweig_ultrafast_artifacts/t08/capture")
CAPTURE_PROFILE = "site_500-default"
CAPTURE_LATITUDE = 30.312645

#: Scientific sources whose sha256 certifies each run record. The numba
#: lane + the real job path (server) + this harness.
CERTIFIED_SOURCES = (
    "solweig_core/numba_cpu/full_solve.py",
    "solweig_core/numba_cpu/radiation.py",
    "solweig_core/numba_cpu/met_recompute.py",
    "solweig_core/numba_cpu/thermal.py",
    "solweig_core/runtime.py",
    "solweig_gpu/server/jobs.py",
    "solweig_gpu/server/store.py",
    "solweig_gpu/server/patch_codec.py",
    "solweig_gpu/incremental/worker.py",
    "solweig_gpu/incremental/solver.py",
    "benchmarks/ultrafast/t16_cpu_lane.py",
)


class StageSumMismatch(AssertionError):
    """Exclusive stage timings do not sum to the measured E2E."""


class SamplePhaseError(AssertionError):
    """A percentile was requested over rows that are not samples."""


class SampleCountError(AssertionError):
    """Fewer sample rows than the protocol minimum."""


# ---------------------------------------------------------------------------
# honesty scaffolding
# ---------------------------------------------------------------------------
def percentile_checked(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile over SAMPLE values only.

    Callers pass ``phase == "sample"`` rows (enforced by
    :func:`summarize_samples`); empty input is a hard error — an absent
    percentile must never read as 0.0.
    """
    if not values:
        raise SamplePhaseError("percentile over zero sample rows")
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * (float(q) / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def summarize_samples(rows: Sequence[Mapping[str, Any]], key: str,
                      *, runs_requested: int) -> dict:
    """p50/p95/min/max/n over ``key`` across SAMPLE rows, with the protocol
    assertions: only ``phase == "sample"`` rows enter a percentile (a p95
    over warmup-only rows raises — zero sample rows is a hard error, never
    a quiet 0.0), and the sample count must reach what the protocol
    requested (a silently truncated run must not produce a confident
    p95)."""
    samples = [r for r in rows if r.get("phase") == "sample"]
    if not samples:
        raise SamplePhaseError(
            "zero sample rows carry phase=='sample' (warmup-only rows "
            f"must never produce a p95 for {key!r})")
    if len(samples) < runs_requested:
        raise SampleCountError(
            f"{len(samples)} sample rows < requested {runs_requested}; "
            "refusing to summarize a truncated run")
    vals = [float(r[key]) for r in samples]
    return {
        "n": len(vals),
        "p50": percentile_checked(vals, 50),
        "p95": percentile_checked(vals, 95),
        "min": min(vals),
        "max": max(vals),
    }


def assert_exclusive_stage_sum(stage_s: Mapping[str, float], e2e_s: float,
                              *, tol_rel: float = 0.02,
                              tol_abs_s: float = 0.05) -> None:
    """Exclusive stages must EXPLAIN the E2E: |sum(stages) - e2e| within
    max(rel*|e2e|, abs). A stage table that double-counts (or omits) work
    fails the run instead of publishing an unexplainable number."""
    total = sum(float(v) for v in stage_s.values())
    tol = max(tol_rel * abs(float(e2e_s)), tol_abs_s)
    if abs(total - float(e2e_s)) > tol:
        raise StageSumMismatch(
            f"exclusive stages sum to {total:.6f}s but E2E is "
            f"{float(e2e_s):.6f}s (tol {tol:.6f}s); stages={dict(stage_s)}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def certified_source_hashes() -> dict[str, str]:
    return {rel: sha256_file(REPO_ROOT / rel) for rel in CERTIFIED_SOURCES
            if (REPO_ROOT / rel).is_file()}


def hw_record() -> dict:
    """Hardware affinity / load disclosure per the yaml's
    report_hardware_affinity_load_and_coverage."""
    affinity = None
    affinity_available = False
    if hasattr(os, "sched_getaffinity"):
        try:
            affinity = sorted(os.sched_getaffinity(0))
            affinity_available = True
        except OSError:
            affinity = None
    record = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "mac_model": None,
        "cpu_count": os.cpu_count(),
        "sched_getaffinity_available": affinity_available,
        "affinity": affinity,
        "loadavg": list(os.getloadavg()),
        "pid": os.getpid(),
    }
    try:  # macOS host model (the dev-host disclosure the report needs)
        record["mac_model"] = subprocess.run(
            ["sysctl", "-n", "hw.model"], capture_output=True,
            text=True, timeout=5).stdout.strip()
    except Exception:  # noqa: BLE001 - disclosure only
        record["mac_model"] = None
    if not record["mac_model"]:  # sandboxed sysctl fallback
        record["mac_model"] = platform.mac_ver()[0] or None
    try:
        record["cpu_brand"] = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
            text=True, timeout=5).stdout.strip()
    except Exception:  # noqa: BLE001
        record["cpu_brand"] = None
    return record


def env_record() -> dict:
    import numpy

    record = {"python": sys.version.split()[0], "numpy": numpy.__version__}
    for name in ("numba", "torch"):
        try:
            record[name] = __import__(name).__version__
        except Exception:  # noqa: BLE001 - absent dep is a fact, not error
            record[name] = None
    return record


def _append_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())


# ---------------------------------------------------------------------------
# variant: cpu-numba-selected-exact (the REAL job path)
# ---------------------------------------------------------------------------
#: E1 at the representative fixture: add t1 h10 r4 at cell (166,102)
#: (benchmarks/ultrafast/harness_solve.py T1) expressed in API tree shape.
E1_CELL = (166, 102)
E1_TREE = {
    "tree_id": "t1",
    "height_m": 10.0,
    "canopy_diameter_m": 8.0,   # TreeSpec radius 4.0 m
    "trunk_ratio": 0.25,
    "transmissivity": 0.03,
}
SELECTED_TIME_INDEX = 12  # yaml representative_selected_global_times


def _e1_api_tree(site_geometry: Mapping[str, Any]) -> dict:
    from solweig_gpu.server.store import world_to_uv

    rows = int(site_geometry["rows"])
    cols = int(site_geometry["cols"])
    pixel = float(site_geometry["pixel_size_m"])
    ox = float(site_geometry["origin_x_m"])
    oy = float(site_geometry["origin_y_m"])
    x_m = ox + (E1_CELL[1] + 0.5) * pixel   # harness_solve.cell_xy
    y_m = oy - (E1_CELL[0] + 0.5) * pixel
    u, v = world_to_uv(x_m, y_m, rows=rows, cols=cols, pixel_size_m=pixel,
                       origin_x_m=ox, origin_y_m=oy)
    return {**E1_TREE, "u": u, "v": v}


def _client_apply(manifest: Mapping[str, Any],
                  arrays: Mapping[str, Any], grid: Mapping[str, Any]) -> dict:
    """The client-side apply: scatter the decoded patch into a full-site
    state at the manifest's GLOBAL time indices (jobs._apply_result_into_
    state semantics, client copy so the server path stays untouched)."""
    import numpy as np

    window = manifest["window"]
    r0, r1 = int(window["row_start"]), int(window["row_stop"])
    c0, c1 = int(window["col_start"]), int(window["col_stop"])
    time_indices = [int(t) for t in manifest.get("time_indices", [])]
    state = {}
    for name, patch in arrays.items():
        target = np.zeros(
            (int(grid["time_steps"]), int(grid["rows"]), int(grid["cols"])),
            dtype=np.float32)
        region = target[:, r0:r1, c0:c1]
        idx = time_indices if len(time_indices) == patch.shape[0] else \
            list(range(patch.shape[0]))
        region[idx] = patch
        state[name] = target
    return state


def bench_selected_exact(*, cache_dir: Path, site_dir: Path, out_dir: Path,
                         warmup: int, repeats: int, pairs: int,
                         seed: int, poll_s: float = 0.002) -> dict:
    """Real-path selected-time edit job: accept -> published -> applied.

    One :class:`JobRunner` (real solver factory, real Store in tmp dirs,
    default 500 ms coalescing window — queue/epoch wait is part of the
    measured scope per the yaml's ``include_queue_epoch_wait``). Per rep a
    FRESH scenario is created (stored site baseline published inline, as
    ``routes_scenarios`` does for caches with baseline_results), one E1
    geometry edit is committed with ``requested_result.time_indices=[12]``
    and the T15 flag ON, and the wall is split into exclusive stages that
    must sum to the E2E.
    """
    os.environ.setdefault("SOLWEIG_RT_SELECTED_TIME_STREAMING", "1")
    from solweig_gpu.server.jobs import (
        JobRunner, RunnerContext, SiteRegistry, make_exact_worker_solver,
        materialize_baseline_result)
    from solweig_gpu.server.store import Store
    from solweig_gpu.server import patch_codec

    site_id = "site_500"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    record_out = out_dir / "records_raw.jsonl"

    sites = SiteRegistry({site_id: {
        "cache_dir": str(cache_dir), "site_dir": str(site_dir),
        "selected_date_str": "2009-08-11",
    }})
    store = Store(out_dir / "store.sqlite3",
                  results_root=out_dir / "scenarios")
    context = RunnerContext(store=store, sites=sites,
                            state_root=out_dir / "state")
    stamps: dict[str, float] = {}

    def timing_factory(ctx):  # wraps the REAL factory; measures dispatch
        inner = make_exact_worker_solver()(ctx)

        def solve(request, progress):
            stamps["solve_start"] = time.perf_counter()
            try:
                return inner(request, progress)
            finally:
                stamps["solve_end"] = time.perf_counter()

        return solve

    runner = JobRunner(context, solver_factory=timing_factory)
    runner.start()
    grid = sites.geometry(site_id)
    e1_tree = _e1_api_tree(grid)

    def one_job(rep: int, phase: str, requested: dict) -> dict:
        scenario_id = f"t16_sel_{phase}_{rep}"
        store.create_scenario(
            site_id=site_id, name=f"t16-{phase}-{rep}",
            scenario_id=scenario_id,
            baseline_publisher=lambda sid, ver: (
                materialize_baseline_result(sites, sid, site_id,
                                            scene_version=ver)),
        )
        t0 = time.perf_counter()
        _record, job_id = store.commit_edits(
            scenario_id,
            base_scene_version=0,
            applied_edits=[{
                "operation": "add", "tree_id": e1_tree["tree_id"],
                "old_tree": None, "new_tree": e1_tree,
            }],
            requested=requested,
        )
        t_accept = time.perf_counter()
        runner.submit(job_id)
        deadline = time.monotonic() + 1800.0
        while time.monotonic() < deadline:
            job = store.get_job(job_id)
            if job is not None and job.status in (
                    "complete", "failed", "superseded", "cancelled"):
                break
            time.sleep(poll_s)
        else:
            raise RuntimeError(f"job {job_id} did not finish in 1800 s")
        t_complete = time.perf_counter()
        if job.status != "complete":
            raise RuntimeError(
                f"job {job_id} finished {job.status!r}: {job.error}")
        result = store.get_result(scenario_id, job.target_scene_version)
        if result is None:
            raise RuntimeError(f"no published result for {scenario_id}")
        payload = result.payload_bytes()
        arrays = patch_codec.decode_payload(result.manifest, payload)
        _state = _client_apply(result.manifest, arrays, grid)
        t_applied = time.perf_counter()

        stage_s = {
            "accept_s": t_accept - t0,
            "queue_wait_s": stamps["solve_start"] - t_accept,
            "solve_s": stamps["solve_end"] - stamps["solve_start"],
            "publish_s": t_complete - stamps["solve_end"],
            "fetch_decode_apply_s": t_applied - t_complete,
        }
        e2e = t_applied - t0
        assert_exclusive_stage_sum(stage_s, e2e)
        return {
            "run_id": f"{scenario_id}_{_utcnow()}",
            "phase": phase, "rep": rep,
            "requested": requested,
            "e2e_s": e2e,
            "stage_s": stage_s,
            "payload_bytes": len(payload),
            "time_indices": list(result.manifest.get("time_indices", [])),
            "window": result.manifest.get("window"),
            "job_metrics": result.manifest.get("metrics"),
            "hw": hw_record(),
        }

    rows: list[dict] = []
    for i in range(warmup):
        rows.append(one_job(i, "warmup", {
            "time_indices": [SELECTED_TIME_INDEX],
            "variables": ["utci", "tmrt"]}))
        print(f"[selected-exact] warmup {i + 1}/{warmup} done", flush=True)
    for i in range(repeats):
        rows.append(one_job(i, "sample", {
            "time_indices": [SELECTED_TIME_INDEX],
            "variables": ["utci", "tmrt"]}))
        print(f"[selected-exact] sample {i + 1}/{repeats} "
              f"(e2e {rows[-1]['e2e_s']:.2f}s)", flush=True)

    # paired A/B with randomized order (sampling.randomize_paired_ab_order):
    # A = the same E1 edit job with NO time cut (full-day payload), B = the
    # selected-time job. Same process, same warm state; the pair isolates
    # the T15 causal-prefix cut's contribution.
    rng = random.Random(seed)
    pair_rows: list[dict] = []
    for i in range(pairs):
        order = ["A", "B"] if rng.random() < 0.5 else ["B", "A"]
        for arm in order:
            requested = ({} if arm == "A" else {
                "time_indices": [SELECTED_TIME_INDEX],
                "variables": ["utci", "tmrt"]})
            row = one_job(i, f"pair_{arm}", requested)
            row["pair"] = i
            row["arm"] = arm
            row["order"] = order
            pair_rows.append(row)
        print(f"[selected-exact] pair {i + 1}/{pairs} order={order}",
              flush=True)

    runner.stop()
    _append_jsonl(record_out, rows + pair_rows)
    summary = {
        "schema_version": 1,
        "variant": "cpu-numba-selected-exact",
        "recorded_at_utc": _utcnow(),
        "site": {"cache_dir": str(cache_dir), "site_dir": str(site_dir),
                 "grid": {k: grid[k] for k in (
                     "rows", "cols", "pixel_size_m", "time_steps")}},
        "edit": "E1 add t1 h10 r4 at cell (166,102) via real JobRunner path",
        "selected_time_index": SELECTED_TIME_INDEX,
        "flag": "SOLWEIG_RT_SELECTED_TIME_STREAMING=1",
        "coalescing_window_ms": runner.coalescing_window_ms,
        "sampling": {
            "warmup": warmup, "repeats": repeats, "pairs": pairs,
            "paired_order_randomized": True, "seed": seed,
            "poll_s": poll_s,
            "repeats_justification": (
                "single run ~40-60 s, at/above the ~60 s "
                "paired_expensive threshold; >=30 samples per the yaml's "
                "expensive regime" if repeats < 100 else None),
        },
        "e2e_s": summarize_samples(rows, "e2e_s", runs_requested=repeats),
        "stage_s": {
            name: _stage_summary(rows, name, repeats)
            for name in (rows[0]["stage_s"] if rows else {})
        },
        "pairs": [
            {"pair": p, "order": [r["arm"] for r in pair_rows if r["pair"] == p],
             "A_e2e_s": next(r["e2e_s"] for r in pair_rows
                             if r["pair"] == p and r["arm"] == "A"),
             "B_e2e_s": next(r["e2e_s"] for r in pair_rows
                             if r["pair"] == p and r["arm"] == "B")}
            for p in range(pairs)
        ],
        "hw_at_start": hw_record(),
        "env": env_record(),
        "source_sha256": certified_source_hashes(),
        "run_dir": str(out_dir),
    }
    return summary


def _stage_summary(rows: Sequence[Mapping[str, Any]], name: str,
                   repeats: int) -> dict:
    """Exclusive-stage stats over sample rows (stage values lifted out of
    each row's stage dict; warmup rows excluded by construction)."""
    lifted = [{"phase": r["phase"], "value": r["stage_s"][name]}
              for r in rows if name in r.get("stage_s", {})]
    return summarize_samples(lifted, "value", runs_requested=repeats)


# ---------------------------------------------------------------------------
# variants: full-day (cold/warm) + full-recompute-warm (numba capture lane)
# ---------------------------------------------------------------------------
_COLD_RUNNER = r'''
import json, sys, time
sys.path.insert(0, {root!r})
t_import0 = time.perf_counter()
import numpy as np
from solweig_core.numba_cpu import full_solve as fs
t_import = time.perf_counter() - t_import0
cs = fs.CaptureSet(cap_dir={cap!r}, profile={profile!r}, latitude={lat!r})
t0 = time.perf_counter()
res = fs.full_solve_capture(cs, None, profile={profile!r},
                            rows={rows!r}, cols={cols!r}, publish_anchor=False)
wall = time.perf_counter() - t0
print("@@T16@@" + json.dumps({{
    "import_s": t_import, "solve_wall_s": wall,
    "per_step_ms": [round(x, 3) for x in res.timings_ms],
    "first_step_ms": round(res.timings_ms[0], 3),
}}))
'''


def _run_cold_subprocess(capture: Path, rows: int, cols: int) -> dict:
    code = _COLD_RUNNER.format(root=str(REPO_ROOT), cap=str(capture),
                               profile=CAPTURE_PROFILE, lat=CAPTURE_LATITUDE,
                               rows=rows, cols=cols)
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, cwd=str(REPO_ROOT))
    if proc.returncode != 0:
        raise RuntimeError(f"cold subprocess failed: {proc.stderr[-2000:]}")
    line = [ln for ln in proc.stdout.splitlines()
            if ln.startswith("@@T16@@")]
    if not line:
        raise RuntimeError(f"cold subprocess produced no marker: "
                           f"{proc.stdout[-500:]} {proc.stderr[-500:]}")
    return json.loads(line[0][len("@@T16@@"):])


def bench_full_day(*, capture: Path, out_dir: Path, warmup: int,
                   repeats: int, cold_repeats: int | None) -> dict:
    """24-timestep full solve, cold (fresh subprocess per sample, shipped
    numba disk cache) and warm (in-process repeats after warmup), with a
    one-pass exclusive stage ablation that must explain the warm wall."""
    from solweig_core.numba_cpu import full_solve as fs

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    record_out = out_dir / "records_raw.jsonl"
    man = json.loads((capture / "manifest.json").read_text())
    n_t = int(man["n_timesteps"])
    rows = int(man.get("rows", 500))
    cols = int(man.get("cols", 500))
    cs = fs.CaptureSet(cap_dir=str(capture), profile=CAPTURE_PROFILE,
                       latitude=CAPTURE_LATITUDE)

    rows_out: list[dict] = []
    for i in range(warmup):
        t0 = time.perf_counter()
        res = fs.full_solve_capture(cs, None, profile=CAPTURE_PROFILE,
                                    rows=rows, cols=cols, publish_anchor=False)
        rows_out.append({"phase": "warmup", "rep": i,
                         "wall_s": time.perf_counter() - t0,
                         "per_step_median_ms": statistics.median(
                             res.timings_ms),
                         "hw": hw_record()})
        print(f"[full-day] warmup {i + 1}/{warmup} "
              f"({rows_out[-1]['wall_s']:.2f}s)", flush=True)
    for i in range(repeats):
        t0 = time.perf_counter()
        res = fs.full_solve_capture(cs, None, profile=CAPTURE_PROFILE,
                                    rows=rows, cols=cols, publish_anchor=False)
        rows_out.append({"phase": "sample", "rep": i,
                         "wall_s": time.perf_counter() - t0,
                         "per_step_median_ms": statistics.median(
                             res.timings_ms),
                         "hw": hw_record()})
        print(f"[full-day] warm sample {i + 1}/{repeats} "
              f"({rows_out[-1]['wall_s']:.2f}s)", flush=True)

    cold_rows: list[dict] = []
    if cold_repeats:
        for i in range(cold_repeats):
            mark = _run_cold_subprocess(capture, rows, cols)
            cold_rows.append({"phase": "sample", "rep": i,
                              "wall_s": mark["import_s"]
                              + mark["solve_wall_s"],
                              **mark, "hw": hw_record()})
            print(f"[full-day] cold sample {i + 1}/{cold_repeats} "
                  f"({cold_rows[-1]['wall_s']:.2f}s)", flush=True)

    ablation = _ablate_full_solve_once(capture, rows, cols)
    warm_wall = statistics.median(
        [r["wall_s"] for r in rows_out if r["phase"] == "sample"])
    _append_jsonl(record_out, rows_out + [
        # spread FIRST so the cold tag is not clobbered by r's "sample"
        {**r, "phase": "cold_subprocess"} for r in cold_rows])
    return {
        "schema_version": 1,
        "variant": "cpu-numba-full-day",
        "recorded_at_utc": _utcnow(),
        "capture": str(capture), "rows": rows, "cols": cols,
        "n_timesteps": n_t,
        "warm": summarize_samples(rows_out, "wall_s",
                                  runs_requested=repeats),
        "warm_per_step_median_ms": summarize_samples(
            rows_out, "per_step_median_ms", runs_requested=repeats),
        "cold": (summarize_samples(
            [{"phase": r["phase"], "wall_s": r["wall_s"]}
             for r in cold_rows], "wall_s", runs_requested=cold_repeats)
            if cold_repeats else None),
        "cold_definition": ("fresh subprocess per sample: interpreter + "
                            "import + capture load + full 24-step solve; "
                            "numba disk cache PRESENT (shipped/AOT shape)"),
        "stage_ablation_s": ablation,
        "stage_ablation_vs_warm_wall_s": warm_wall,
        "sampling": {"warmup": warmup, "repeats": repeats,
                     "cold_repeats": cold_repeats},
        "hw_at_start": hw_record(),
        "env": env_record(),
        "source_sha256": certified_source_hashes(),
        "run_dir": str(out_dir),
    }


def _ablate_full_solve_once(capture: Path, rows: int, cols: int) -> dict:
    """One instrumented replay of the full-solve operation sequence with
    EXCLUSIVE per-stage timers (the E2E comes from the real
    ``full_solve_capture`` wall; this pass explains it). Stages:
    static_load / bundle_load / met_recompute / fused_kernel; the sum is
    asserted against this pass's own wall so the table cannot
    double-count."""
    from solweig_core.numba_cpu.met_recompute import (
        met_from_capture, recompute_timestep, site_params_from_capture,
        time_geom_from_capture)
    from solweig_core.numba_cpu.radiation import (
        fused_radiation_timestep, rad_bundle_from_capture,
        rad_state_from_capture, rad_static_from_capture)

    capture = Path(capture)
    n = int(json.loads((capture / "manifest.json").read_text())["n_timesteps"])
    stage = {k: 0.0 for k in ("static_load_s", "bundle_load_s",
                              "met_recompute_s", "fused_kernel_s")}
    t_wall0 = time.perf_counter()
    z0 = np_load(capture / "t00.npz")
    site = site_params_from_capture(z0, CAPTURE_LATITUDE)
    t0 = time.perf_counter()
    st = rad_static_from_capture(capture)
    stage["static_load_s"] += time.perf_counter() - t0
    t0 = time.perf_counter()
    state = rad_state_from_capture(capture, 0, rows=rows, cols=cols)
    stage["static_load_s"] += time.perf_counter() - t0
    ci = float(state.CI)
    ci_is_tensor = False
    for t in range(n):
        t0 = time.perf_counter()
        z = np_load(capture / f"t{t:02d}.npz")
        tg = time_geom_from_capture(z)
        met = met_from_capture(z)
        t_in = rad_bundle_from_capture(capture, t)
        stage["bundle_load_s"] += time.perf_counter() - t0
        shadow = (z["sunon_in_shadow"]
                  if t_in.is_day and "sunon_in_shadow" in z.files else None)
        t0 = time.perf_counter()
        res = recompute_timestep(site, met, tg, ci, shadow,
                                 CI_thread_is_tensor=ci_is_tensor)
        stage["met_recompute_s"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        if t_in.is_day:
            from solweig_core.numba_cpu import full_solve as _fs
            _fs._apply_day_overrides(t_in, res, met)
        else:
            from solweig_core.numba_cpu import full_solve as _fs
            _fs._apply_night_overrides(t_in, res, met)
        out, state = fused_radiation_timestep(st, t_in, state)
        stage["fused_kernel_s"] += time.perf_counter() - t0
        ci = float(res["ret_CI"])
        ci_is_tensor = bool(res["CI_flavor_is_tensor"])
    wall = time.perf_counter() - t_wall0
    assert_exclusive_stage_sum(stage, wall, tol_rel=0.15, tol_abs_s=1.0)
    return {"stages_s": {k: round(v, 4) for k, v in stage.items()},
            "pass_wall_s": round(wall, 4)}


def np_load(path: Path):
    import numpy as np

    return np.load(path)


def bench_full_recompute_warm(*, capture: Path, out_dir: Path, warmup: int,
                              repeats: int) -> dict:
    """Warm-process FULL 24-step recompute (idle-reconcile / fallback full
    recompute), plus the anchor-resume suffix replay as an AUXILIARY row
    (never the primary target number — the target row is the full 24-step
    recompute)."""
    from solweig_core.numba_cpu import full_solve as fs

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    record_out = out_dir / "records_raw.jsonl"
    cs = fs.CaptureSet(cap_dir=str(capture), profile=CAPTURE_PROFILE,
                       latitude=CAPTURE_LATITUDE)

    rows_out: list[dict] = []
    for i in range(warmup):
        t0 = time.perf_counter()
        fs.full_solve_capture(cs, None, profile=CAPTURE_PROFILE,
                              publish_anchor=False)
        rows_out.append({"phase": "warmup", "rep": i,
                         "wall_s": time.perf_counter() - t0,
                         "hw": hw_record()})
        print(f"[recompute-warm] warmup {i + 1}/{warmup}", flush=True)
    for i in range(repeats):
        t0 = time.perf_counter()
        res = fs.full_solve_capture(cs, None, profile=CAPTURE_PROFILE,
                                    publish_anchor=False)
        rows_out.append({"phase": "sample", "rep": i,
                         "wall_s": time.perf_counter() - t0,
                         "hw": hw_record()})
        print(f"[recompute-warm] sample {i + 1}/{repeats} "
              f"({rows_out[-1]['wall_s']:.2f}s)", flush=True)

    # auxiliary: anchor-resume suffix replay (T14 W5 shape) — warm start
    # from a MID-WINDOW anchor at step 12, replaying only the suffix.
    anchor_run = fs.full_solve_capture(cs, None, profile=CAPTURE_PROFILE,
                                       publish_anchor_at=[11])
    aux_rows: list[dict] = []
    if anchor_run.anchors:
        anchor_state, anchor_fp = anchor_run.anchors[0]
        aux_samples = 5
        for i in range(warmup + aux_samples):
            t0 = time.perf_counter()
            warm = fs.full_solve_capture(
                cs, None, profile=CAPTURE_PROFILE,
                warm_state=anchor_state, warm_fingerprint=anchor_fp,
                r0=int(anchor_state.next_step), publish_anchor=False)
            aux_rows.append({"phase": ("warmup" if i < warmup else "sample"),
                             "rep": i,
                             "wall_s": time.perf_counter() - t0})
    overlap_equal = None
    if anchor_run.anchors:
        warm = fs.full_solve_capture(
            cs, None, profile=CAPTURE_PROFILE, warm_state=anchor_state,
            warm_fingerprint=anchor_fp, r0=int(anchor_state.next_step))
        offset = len(anchor_run.outputs) - len(warm.outputs)
        overlap_equal = all(
            _per_t_digest(anchor_run.outputs[offset + i])
            == _per_t_digest(warm.outputs[i])
            for i in range(len(warm.outputs)))

    _append_jsonl(record_out, rows_out)
    return {
        "schema_version": 1,
        "variant": "cpu-numba-full-recompute-warm",
        "recorded_at_utc": _utcnow(),
        "capture": str(capture),
        "definition": ("warm-process FULL 24-step recompute (publish_anchor "
                       "off; JIT + page cache warm)"),
        "full_recompute_warm": summarize_samples(
            rows_out, "wall_s", runs_requested=repeats),
        "auxiliary_anchor_resume_suffix": (
            summarize_samples(
                [r for r in aux_rows if r["phase"] == "sample"], "wall_s",
                runs_requested=5)
            if len(aux_rows) > warmup else None),
        "auxiliary_overlap_raw_bit_equal": overlap_equal,
        "sampling": {"warmup": warmup, "repeats": repeats},
        "hw_at_start": hw_record(),
        "env": env_record(),
        "source_sha256": certified_source_hashes(),
        "run_dir": str(out_dir),
    }


def _per_t_digest(out: Mapping[str, Any]) -> str:
    import numpy as np

    h = hashlib.sha256()
    for k in sorted(out):
        v = out[k]
        if hasattr(v, "tobytes"):
            a = np.ascontiguousarray(v)
            if a.dtype == np.float32:
                a = a.view(np.uint32)
            h.update(k.encode())
            h.update(a.tobytes())
    return h.hexdigest()
