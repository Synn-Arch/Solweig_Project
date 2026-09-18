"""P9 REL-002: real-time classroom queue load test (REAL solver).

testing_validation.md "Queue load test" specifies the classroom session:
20 scenarios, one edit per scenario every 120 s on average, bursts of five
edits within 10 s, one exact worker, 30-minute run. The P9 mission bounds
the wall time at < 10 minutes, so this harness keeps the spec's SHAPE on a
smaller footprint and a REAL solve path:

* 10 scenarios on the tests' synthetic "tiny" fixture site (128x128 @ 2 m,
  3 met hours) with the DEFAULT exact solver (ExactWorker) — real physics,
  real incremental windows, no fake latency;
* arrival rate derived from the MEASURED service time: the spec ratio is
  120 s arrival / ~25 s service ~ 4.8 (P8 evidence, 500x500 @ 2 m). This
  harness measures the tiny site's real per-edit service time S_cal and
  sets the steady inter-edit interval to 4.8 * S_cal, preserving the
  arrival/service ratio (the quantity that determines queue dynamics);
* two bursts of 5 edits within 2 s to one scenario each (spec: 5 in 10 s,
  scaled to the compressed service time so all 5 actually overlap in the
  queue — otherwise the burst cannot exercise coalescing).

PASS criteria (teaching tolerance, documented because the spec names no
number; every criterion is derived from the spec's own language):
  1. every accepted (202) edit's job reaches a terminal status — no
     silently dropped committed edits;
  2. the queue fully drains: zero nonterminal jobs after the load stops
     (bounded backlog);
  3. every scenario reaches exact at its latest scene version;
  4. bursts coalesce: superseded count > 0;
  5. steady-state queue wait p95 <= 2x service time (arrival rate is ~1/5
     of service rate, so waits should stay near one coalescing window).

Run:  python measure_load.py            (full load test, writes JSON)
      python measure_load.py --calibrate (timings only)
"""
from __future__ import annotations

import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(WORKTREE))
sys.path.insert(0, str(WORKTREE / "tests"))

from fastapi.testclient import TestClient  # noqa: E402

from solweig_gpu.server.app import create_app  # noqa: E402
from tests.test_incremental_worker import _build_cache, _make_tiny_site  # noqa: E402

OUT_DIR = Path("/tmp/p9reliability")
SPEC_RATIO = 4.8        # 120 s arrival / 25 s service (P8, 500x500 site)
N_SCENARIOS = 10
BURSTS = 2
BURST_EDITS = 5
BURST_SPAN_S = 2.0
STEADY_BUDGET_S = 240.0
DRAIN_TIMEOUT_S = 120.0


def iso_to_epoch(stamp: str | None) -> float | None:
    if not stamp:
        return None
    return (
        datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%S.%fZ").replace(
            tzinfo=None
        )
        - datetime(1970, 1, 1)
    ).total_seconds()


def wait_terminal(client: TestClient, job_id: str, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    body: dict = {}
    while time.monotonic() < deadline:
        body = client.get(f"/api/v1/jobs/{job_id}").json()
        if body.get("status") not in ("queued", "running"):
            return body
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} never finished: {body}")


def edit_body(base: int, tree_id: str, u: float) -> dict:
    return {
        "base_scene_version": base,
        "edits": [
            {
                "operation": "add",
                "tree": {
                    "tree_id": tree_id,
                    "component_type": "broad_canopy",
                    "u": u,
                    "v": 0.45,
                    "height_m": 6.0,
                    "canopy_diameter_m": 4.0,
                },
            }
        ],
    }


def main() -> None:
    import tempfile

    import psutil

    tmp = Path(tempfile.mkdtemp(prefix="rel002-"))
    print("building tiny fixture site (real cache build)...", flush=True)
    grid, site = _make_tiny_site(tmp)
    cache_dir = tmp / "cache"
    _build_cache(
        site,
        cache_dir,
        met_path=site / "metfiles" / "metfile_0_0_2024-06-20.txt",
        site_id="tiny",
    )

    app = create_app(
        state_root=tmp / "state",
        sites={"tiny": {"cache_dir": cache_dir, "site_dir": site}},
        coalescing_window_ms=500.0,  # production default
        requests_per_minute_per_ip=None,  # harness traffic, not clients
        edits_per_minute=None,
    )
    process = psutil.Process()

    with TestClient(app) as client:
        store = client.app.state.context.store

        # -- calibration: one baseline full-solve + one incremental edit ----
        t0 = time.monotonic()
        created = client.post("/api/v1/scenarios", json={"site_id": "tiny"})
        assert created.status_code == 201, created.text
        calibration = created.json()
        baseline = wait_terminal(client, calibration["job_id"], timeout=300.0)
        assert baseline["status"] == "complete", baseline
        s_full = time.monotonic() - t0

        t1 = time.monotonic()
        edit = client.post(
            f"/api/v1/scenarios/{calibration['scenario_id']}/edits",
            json=edit_body(0, "calib-1", 0.42),
        )
        assert edit.status_code == 202, edit.text
        first = wait_terminal(client, edit.json()["job_id"], timeout=300.0)
        assert first["status"] == "complete", first
        s_cal = time.monotonic() - t1
        print(
            f"calibration: baseline full-solve {s_full:.2f}s, "
            f"incremental edit {s_cal:.2f}s",
            flush=True,
        )
        if "--calibrate" in sys.argv:
            return

        steady_interval = max(SPEC_RATIO * s_cal, 1.0)
        n_steady = max(int(STEADY_BUDGET_S / steady_interval), 8)
        print(
            f"schedule: steady 1 edit / {steady_interval:.2f}s x {n_steady}, "
            f"{BURSTS} bursts x {BURST_EDITS} within {BURST_SPAN_S}s",
            flush=True,
        )

        # -- warmup: create the remaining scenarios, then BARRIER until every
        #    baseline job has drained so steady-state queue waits measure the
        #    classroom load, not the deployment warm-up backlog.
        scenarios = [calibration["scenario_id"]]
        for index in range(1, N_SCENARIOS):
            created = client.post(
                "/api/v1/scenarios",
                json={"site_id": "tiny", "name": f"classroom-{index}"},
            )
            assert created.status_code == 201, created.text
            scenarios.append(created.json()["scenario_id"])
        warmup_deadline = time.monotonic() + 300.0
        while store.nonterminal_jobs():
            if time.monotonic() > warmup_deadline:
                raise AssertionError("warm-up baselines never drained")
            time.sleep(0.25)
        warmup_s = None  # warm-up is outside the measured load window

        samples: list[dict] = []
        stop = threading.Event()

        def sampler() -> None:
            while not stop.is_set():
                nonterminal = store.nonterminal_jobs()
                samples.append(
                    {
                        "t": round(time.monotonic() - t_start, 2),
                        "nonterminal": len(nonterminal),
                        "queued": sum(
                            1 for j in nonterminal if j.status == "queued"
                        ),
                        "running": sum(
                            1 for j in nonterminal if j.status == "running"
                        ),
                        "rss_bytes": process.memory_info().rss,
                    }
                )
                time.sleep(0.25)

        accepted: list[dict] = []
        submitted = {"steady": 0, "burst": 0}
        t_start = time.monotonic()
        sampler_thread = threading.Thread(target=sampler, daemon=True)
        sampler_thread.start()

        def submit(scenario_id: str, n: int, kind: str) -> None:
            submitted[kind] += 1
            base = client.get(f"/api/v1/scenarios/{scenario_id}").json()[
                "scene_version"
            ]
            response = client.post(
                f"/api/v1/scenarios/{scenario_id}/edits",
                json=edit_body(base, f"tree-{n}", 0.15 + 0.03 * (n % 25)),
                headers={"Idempotency-Key": f"edit-{scenario_id}-{n}"},
            )
            assert response.status_code == 202, response.text
            accepted.append(
                {
                    "scenario": scenario_id,
                    "job_id": response.json()["job_id"],
                    "t_submit": round(time.monotonic() - t_start, 3),
                }
            )

        # -- steady phase -----------------------------------------------------
        edit_n = 0
        next_steady = t_start + 1.0
        burst_due = [t_start + STEADY_BUDGET_S * 0.35, t_start + STEADY_BUDGET_S * 0.75]
        burst_index = 0
        while True:
            now = time.monotonic()
            if burst_index < BURSTS and now >= burst_due[burst_index]:
                burst_scenario = scenarios[(edit_n + 3) % N_SCENARIOS]
                for k in range(BURST_EDITS):
                    submit(burst_scenario, n=edit_n + k, kind="burst")
                    time.sleep(BURST_SPAN_S / BURST_EDITS)
                edit_n += BURST_EDITS
                burst_index += 1
                next_steady = max(next_steady, time.monotonic() + steady_interval)
                continue
            if now >= next_steady and edit_n < n_steady:
                submit(scenarios[edit_n % N_SCENARIOS], n=edit_n, kind="steady")
                edit_n += 1
                next_steady += steady_interval
                continue
            if edit_n >= n_steady and burst_index >= BURSTS:
                break
            time.sleep(0.02)

        # -- drain ------------------------------------------------------------
        drain_start = time.monotonic()
        deadline = drain_start + DRAIN_TIMEOUT_S
        while time.monotonic() < deadline and store.nonterminal_jobs():
            time.sleep(0.25)
        drain_s = time.monotonic() - drain_start
        stop.set()
        sampler_thread.join(timeout=2.0)

        # -- per-job latency bookkeeping from the durable ledger --------------
        per_job: list[dict] = []
        for item in accepted:
            record = store.get_job(item["job_id"])
            assert record is not None, item
            queued = iso_to_epoch(record.queued_at)
            started = iso_to_epoch(record.started_at)
            finished = iso_to_epoch(record.finished_at)
            per_job.append(
                {
                    "scenario": item["scenario"],
                    "job_id": record.job_id,
                    "status": record.status,
                    "t_submit_s": item["t_submit"],
                    "queue_wait_s": (
                        round(started - queued, 3)
                        if started is not None and queued is not None
                        else None
                    ),
                    "total_s": (
                        round(finished - queued, 3)
                        if finished is not None and queued is not None
                        else None
                    ),
                }
            )

        steady_jobs = [
            job
            for job in per_job
            if job["status"] in ("complete", "superseded")
        ]
        waits = sorted(
            job["queue_wait_s"]
            for job in per_job
            if job["queue_wait_s"] is not None
        )

        def percentile(values: list[float], fraction: float) -> float | None:
            if not values:
                return None
            index = min(int(fraction * len(values)), len(values) - 1)
            return values[index]

        status_counts: dict[str, int] = {}
        for job in per_job:
            status_counts[job["status"]] = status_counts.get(job["status"], 0) + 1

        nonterminal_final = len(store.nonterminal_jobs())
        stranded = []
        for scenario_id in scenarios:
            body = client.get(f"/api/v1/scenarios/{scenario_id}").json()
            if body["exact_result_version"] != body["scene_version"]:
                stranded.append(scenario_id)

        rss_samples = [s["rss_bytes"] for s in samples]
        p95_wait = percentile(waits, 0.95)
        criteria = {
            "no_dropped_edits": all(
                job["status"] in ("complete", "superseded", "failed", "cancelled")
                for job in per_job
            )
            and status_counts.get("failed", 0) == 0,
            "queue_drains": nonterminal_final == 0,
            "all_scenarios_exact": not stranded,
            "bursts_coalesce": status_counts.get("superseded", 0) > 0,
            "steady_p95_wait_within_2x_service": (
                p95_wait is not None and p95_wait <= 2.0 * s_cal
            ),
        }
        report = {
            "record_type": "p9-reliability-load",
            "date": "2026-09-02",
            "fixture": {
                "site_id": "tiny",
                "grid": f"{grid.rows}x{grid.cols}@{grid.pixel_size_m:g}m",
                "source": "tests/test_incremental_worker.py::_make_tiny_site",
                "solver": "default make_exact_worker_solver (ExactWorker)",
                "met_hours": 3,
            },
            "calibration": {
                "baseline_full_solve_s": round(s_full, 3),
                "incremental_edit_s": round(s_cal, 3),
                "spec_ratio_arrival_over_service": SPEC_RATIO,
                "steady_interval_s": round(steady_interval, 3),
            },
            "load": {
                "scenarios": N_SCENARIOS,
                "steady_edits_planned": n_steady,
                "steady_edits": submitted["steady"],
                "burst_edits": submitted["burst"],
                "bursts": {
                    "count": BURSTS,
                    "edits_per_burst": BURST_EDITS,
                    "span_s": BURST_SPAN_S,
                },
                "edits_accepted": len(accepted),
            },
            "results": {
                "job_status_counts": status_counts,
                "per_job": per_job,
                "queue_wait_p50_s": percentile(waits, 0.50),
                "queue_wait_p95_s": p95_wait,
                "max_queue_depth": max(s["queued"] for s in samples),
                "max_nonterminal_depth": max(s["nonterminal"] for s in samples),
                "drain_seconds": round(drain_s, 2),
                "final_nonterminal_jobs": nonterminal_final,
                "stranded_scenarios": stranded,
                "worker_rss_mb": {
                    "min": round(min(rss_samples) / 1e6, 1),
                    "max": round(max(rss_samples) / 1e6, 1),
                    "final": round(rss_samples[-1] / 1e6, 1),
                },
                "steady_job_count": len(steady_jobs),
            },
            "gate_REL_002": all(criteria.values()),
            "criteria": criteria,
            "wall_clock_s": round(time.monotonic() - t_start, 1),
            "note": (
                "Real-time run (no time compression) with the real ExactWorker "
                "on the tests' synthetic tiny site; arrival interval derived "
                "from the measured service time to keep the spec's "
                "arrival/service ratio (~4.8). Superseded jobs are burst "
                "coalescing, not dropped edits: every accepted edit's target "
                "scene version is covered by the latest completed job."
            ),
        }
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "load_report.json").write_text(json.dumps(report, indent=2))
        (OUT_DIR / "load_samples.json").write_text(json.dumps(samples))
        summary = dict(report)
        summary["results"] = {
            k: v for k, v in report["results"].items() if k != "per_job"
        }
        print(json.dumps(summary, indent=2))
        print(f"wall clock {report['wall_clock_s']}s (budget < 600s)")
        print(f"gate REL-002: {report['gate_REL_002']}")


if __name__ == "__main__":
    main()
