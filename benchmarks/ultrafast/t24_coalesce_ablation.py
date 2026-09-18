# SPDX-License-Identifier: GPL-3.0-only
"""T24 coalescing-window ablation (DESIGN §710, GOAL_PROMPT_T18_CPU T24).

Measures what the dispatch debounce contributes to the measured
cpu_selected_exact E2E, by replaying the T16 selected-exact job flow (the
REAL JobRunner/Store path, E1 at the representative site_500 fixture,
``time_indices=[12]``) under three window policies:

  fixed500   the shipped default — 500 ms fixed coalescing wait (flag off;
             the T16 baseline shape; T16 measured queue_wait 0.511 s).
  fixed50    candidate A — pure config reduction (SOLWEIG_COALESCE_MS=50
             shape; zero code change), expressed here as window=50.
  adaptive   candidate B — the T24 flag (SOLWEIG_RT_ADAPTIVE_COALESCE):
             window stays 500, an isolated job dispatches at the 50 ms
             contract floor (the policy's burst indicators keep 500 under
             load; this bench's single isolated edit is exactly the
             no-burst case the floor targets).

PROVISIONAL by design: reduced repeats (default pairs=5), other agents
share the host, and the clean-session re-confirmation belongs to T29. The
point is the queue_wait stage DELTA (deterministic 450 ms) and its
pass-through into E2E — not a new p95 record.

Usage:
  python benchmarks/ultrafast/t24_coalesce_ablation.py \
      [--pairs 5] [--seed 20260909] \
      [--out ~/Workspace/solweig_ultrafast_artifacts/t24/ablation]

Every rep is appended incrementally to ``ablation_raw.jsonl`` so a killed
run keeps its completed rows.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.ultrafast.run import BASELINE_DIR, load_site_paths  # noqa: E402
from benchmarks.ultrafast.t16_cpu_lane import (  # noqa: E402
    SELECTED_TIME_INDEX,
    _client_apply,
    _e1_api_tree,
    assert_exclusive_stage_sum,
    certified_source_hashes,
    env_record,
    hw_record,
    percentile_checked,
)

#: Arm table: (arm, coalescing_window_ms, adaptive_coalesce).
ARMS = (
    ("fixed500", 500.0, False),
    ("fixed50", 50.0, False),
    ("adaptive", 500.0, True),
)


def _arm_runner_kwargs(arm: str) -> dict:
    for name, window, adaptive in ARMS:
        if name == arm:
            return {
                "coalescing_window_ms": window,
                "adaptive_coalesce": adaptive,
            }
    raise KeyError(arm)


def one_job(*, store, sites, runner_factory, grid, e1_tree, scenario_id,
            requested, poll_s: float = 0.002) -> dict:
    """One T16-shaped selected-exact edit job through a REAL runner.

    Same exclusive-stage split as ``t16_cpu_lane.bench_selected_exact``:
    accept / queue_wait / solve / publish / fetch_decode_apply must sum to
    the measured E2E.
    """
    from solweig_gpu.server.jobs import materialize_baseline_result
    from solweig_gpu.server import patch_codec

    stamps: dict[str, float] = {}
    runner = runner_factory(stamps)
    store.create_scenario(
        site_id="site_500",
        name=scenario_id,
        scenario_id=scenario_id,
        # Inline baseline publish, exactly as routes_scenarios does for
        # caches with baseline_results (and as the T16 harness does): it
        # happens BEFORE t0, outside the measured E2E stages.
        baseline_publisher=lambda sid, ver: materialize_baseline_result(
            sites, sid, "site_500", scene_version=ver),
    )
    runner.start()
    try:
        t0 = time.perf_counter()
        _record, job_id = store.commit_edits(
            scenario_id,
            base_scene_version=0,
            applied_edits=[{
                "operation": "add",
                "tree_id": e1_tree["tree_id"],
                "old_tree": None,
                "new_tree": e1_tree,
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
            "scenario_id": scenario_id,
            "e2e_s": e2e,
            "stage_s": stage_s,
            "payload_bytes": len(payload),
            "time_indices": list(result.manifest.get("time_indices", [])),
            "hw": hw_record(),
        }
    finally:
        runner.stop()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--out", type=Path,
        default=Path.home() / "Workspace/solweig_ultrafast_artifacts/t24")
    args = parser.parse_args()

    # Parity with the T16 selected-exact flow (the T15 request-cut flag).
    os.environ.setdefault("SOLWEIG_RT_SELECTED_TIME_STREAMING", "1")

    out_dir = args.out / f"ablation_{time.strftime('%Y%m%dT%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "ablation_raw.jsonl"

    from solweig_gpu.server.jobs import (
        JobRunner, RunnerContext, SiteRegistry, make_exact_worker_solver)
    from solweig_gpu.server.store import Store

    paths = load_site_paths(BASELINE_DIR / "input_manifest.json")
    sites = SiteRegistry({"site_500": {
        "cache_dir": str(paths["cache_dir"]),
        "site_dir": str(paths["site_dir"]),
        "selected_date_str": "2009-08-11",
    }})
    store = Store(out_dir / "store.sqlite3",
                  results_root=out_dir / "scenarios")
    context = RunnerContext(store=store, sites=sites,
                            state_root=out_dir / "state")
    grid = sites.geometry("site_500")
    e1_tree = _e1_api_tree(grid)

    def runner_factory(arm: str, stamps: dict):
        def factory(ctx):
            inner = make_exact_worker_solver()(ctx)

            def solve(request, progress):
                stamps["solve_start"] = time.perf_counter()
                try:
                    return inner(request, progress)
                finally:
                    stamps["solve_end"] = time.perf_counter()

            return solve

        kwargs = _arm_runner_kwargs(arm)
        return JobRunner(context, solver_factory=factory, **kwargs)

    rows: list[dict] = []

    def emit(row: dict) -> None:
        rows.append(row)
        with open(raw_path, "a") as fh:  # incremental: a kill keeps rows
            fh.write(json.dumps(row) + "\n")
        print(
            f"[t24] {row['phase']} pair={row['pair']} arm={row['arm']} "
            f"e2e={row['e2e_s']:.3f}s queue_wait={row['stage_s']['queue_wait_s']:.3f}s "
            f"solve={row['stage_s']['solve_s']:.3f}s",
            flush=True)

    requested = {"time_indices": [SELECTED_TIME_INDEX],
                 "variables": ["utci", "tmrt"]}

    # Warmup: one fixed500 job warms JIT + page cache for ALL arms (same
    # process, same solver factory chain).
    for i in range(args.warmup):
        emit({**one_job(
            store=store, sites=sites,
            runner_factory=lambda stamps: runner_factory("fixed500", stamps),
            grid=grid, e1_tree=e1_tree,
            scenario_id=f"t24_warmup_{i}",
            requested=requested),
            "phase": "warmup", "pair": -1, "arm": "fixed500"})

    rng = random.Random(args.seed)
    for pair in range(args.pairs):
        order = [arm for arm, _, _ in ARMS]
        rng.shuffle(order)
        for arm in order:
            emit({**one_job(
                store=store, sites=sites,
                runner_factory=lambda stamps, a=arm: runner_factory(a, stamps),
                grid=grid, e1_tree=e1_tree,
                scenario_id=f"t24_{arm}_p{pair}",
                requested=requested),
                "phase": "sample", "pair": pair, "arm": arm, "order": order})

    # Summary: per-arm stage/E2E stats over SAMPLE rows only (the T16
    # honesty rules, applied to this reduced provisional run).
    summary = {
        "schema_version": 1,
        "variant": "t24-coalesce-ablation",
        "provisional": True,
        "provisional_reason": (
            "reduced repeats (pairs="
            f"{args.pairs}); shared host (other agents); T29 re-measures "
            "clean-session"),
        "recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00",
                                         time.gmtime()),
        "arms": {name: {"coalescing_window_ms": w, "adaptive_coalesce": a}
                 for name, w, a in ARMS},
        "sampling": {"warmup": args.warmup, "pairs": args.pairs,
                     "order_randomized": True, "seed": args.seed},
        "per_arm": {},
        "pairs": [],
        "hw_at_start": hw_record(),
        "env": env_record(),
        "source_sha256": certified_source_hashes(),
        "run_dir": str(out_dir),
    }
    for arm, _, _ in ARMS:
        arm_rows = [r for r in rows
                    if r["phase"] == "sample" and r["arm"] == arm]
        vals = [r["e2e_s"] for r in arm_rows]
        summary["per_arm"][arm] = {
            "n": len(arm_rows),
            "e2e_s": {"p50": percentile_checked(vals, 50),
                      "p95": percentile_checked(vals, 95),
                      "min": min(vals), "max": max(vals)},
            **{name: {
                "p50": percentile_checked(
                    [r["stage_s"][name] for r in arm_rows], 50),
                "p95": percentile_checked(
                    [r["stage_s"][name] for r in arm_rows], 95),
            } for name in ("queue_wait_s", "solve_s", "publish_s",
                           "accept_s", "fetch_decode_apply_s")},
        }
    for pair in range(args.pairs):
        summary["pairs"].append({
            "pair": pair,
            **{r["arm"]: {"e2e_s": r["e2e_s"],
                          "queue_wait_s": r["stage_s"]["queue_wait_s"]}
               for r in rows if r["phase"] == "sample" and r["pair"] == pair},
        })
    summary_path = out_dir / "ablation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[t24] summary -> {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
