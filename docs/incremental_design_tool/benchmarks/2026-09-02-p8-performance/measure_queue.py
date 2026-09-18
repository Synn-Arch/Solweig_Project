"""P8.4 QUEUE-001: latest-version coalescing under load (compressed time).

Spec: 20 scenarios, one edit per 120 s, bursts of 5 edits per 10 s, one
worker, 30 minutes; simulated time may be compressed if the ratio is kept
and documented.

Compression used here: 20x (30 min -> 90 s wall). Solve latency is injected
into a fake solver at the same 20x (25 s typical local job -> 1.25 s), so
queue dynamics (arrival rate vs service rate, coalescing, supersession)
match the spec's ratios. The physics is irrelevant to this gate; the real
worker's solve is measured separately (PERF-001/002).

PASS (QUEUE-001): with one worker,
  * every scenario reaches exact at its latest version (no stranded edits);
  * burst edits coalesce: the superseded count is nonzero and the queue
    fully drains after the load stops (bounded backlog);
  * no accepted job waits forever: every job reaches a terminal status.
"""
import json
import sys
import threading
import time
from pathlib import Path

WORKTREE = Path("/Users/alansynn/Workspace/solweig/.claude/worktrees/p8-perf")
sys.path.insert(0, str(WORKTREE))
sys.path.insert(0, str(WORKTREE / "tests"))

from fastapi.testclient import TestClient

from solweig_gpu.server.app import create_app
from tests.test_server_api import FakeSolver, SITE_ID, make_site_cache

COMPRESS = 20.0            # 30 min -> 90 s
WALL_SECONDS = 90.0
EDIT_INTERVAL_S = 120.0 / COMPRESS     # 6 s between steady edits
BURST_EVERY_S = 600.0 / COMPRESS       # a burst every 30 s of wall time
BURST_SPAN_S = 10.0 / COMPRESS         # 5 edits within 0.5 s
SOLVE_S = 25.0 / COMPRESS              # 1.25 s per solve
N_SCENARIOS = 20


class SlowFakeSolver(FakeSolver):
    """The contract-test fake with the compressed real-job latency."""

    def __call__(self, request, progress):
        time.sleep(SOLVE_S)
        return super().__call__(request, progress)


def main() -> None:
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="queue001-"))
    solver = SlowFakeSolver()
    app = create_app(
        state_root=tmp / "state",
        sites={SITE_ID: {"cache_dir": str(make_site_cache(tmp))}},
        solver_factory=lambda context: solver,
        coalescing_window_ms=500.0,  # production default
    )
    events: list[dict] = []
    samples: list[dict] = []
    stop = threading.Event()

    with TestClient(app) as client:
        store = client.app.state.context.store

        def sampler() -> None:
            while not stop.is_set():
                nonterminal = store.nonterminal_jobs()
                counts = store.job_status_counts()
                samples.append({
                    "t": round(time.monotonic() - t0, 2),
                    "nonterminal": len(nonterminal),
                    "queued": sum(
                        1 for j in nonterminal if j.status == "queued"
                    ),
                    "running": sum(
                        1 for j in nonterminal if j.status == "running"
                    ),
                    "complete": counts.get("complete", 0),
                    "superseded": counts.get("superseded", 0),
                })
                time.sleep(0.25)

        t0 = time.monotonic()
        sampler_thread = threading.Thread(target=sampler, daemon=True)
        sampler_thread.start()

        # 20 scenarios; creation queues each baseline job (drains during
        # the first seconds of the load window, like a warmed deployment).
        scenarios = []
        for i in range(N_SCENARIOS):
            created = client.post(
                "/api/v1/scenarios",
                json={"site_id": SITE_ID, "name": f"load-{i}"},
                headers={"Idempotency-Key": f"create-{i}"},
            )
            assert created.status_code == 201, created.text
            scenarios.append(created.json()["scenario_id"])
        events.append({"t": 0.0, "kind": "created", "count": len(scenarios)})

        def submit_edit(scenario_id: str, n: int) -> None:
            base = client.get(
                f"/api/v1/scenarios/{scenario_id}"
            ).json()["scene_version"]
            body = {
                "base_scene_version": base,
                "edits": [{
                    "operation": "add",
                    "tree": {
                        "tree_id": f"tree-{n}",
                        "component_type": "broad_canopy",
                        "u": 0.3 + 0.01 * (n % 20),
                        "v": 0.4,
                        "height_m": 6.0,
                        "canopy_diameter_m": 4.0,
                    },
                }],
            }
            response = client.post(
                f"/api/v1/scenarios/{scenario_id}/edits",
                json=body,
                headers={"Idempotency-Key": f"edit-{scenario_id}-{n}"},
            )
            events.append({
                "t": round(time.monotonic() - t0, 2),
                "kind": "edit",
                "scenario": scenario_id,
                "status": response.status_code,
                "job": (response.json().get("job_id") if response.status_code == 202 else None),
            })

        # Steady cadence: one edit every 6 s, round-robin over scenarios.
        # Bursts: 5 edits within 0.5 s at t ~ 20/50/80 s.
        edit_n = 0
        burst_count = 0
        next_steady = t0 + 3.0
        next_burst = t0 + 20.0
        while True:
            now = time.monotonic()
            if now - t0 >= WALL_SECONDS:
                break
            if burst_count < 3 and now >= next_burst:
                # Burst = 5 rapid edits to ONE scenario: supersession is
                # per-scenario (latest scene version wins), so spreading a
                # burst across scenarios can never produce coalescing.
                burst_scenario = scenarios[(edit_n + 2) % N_SCENARIOS]
                for k in range(5):
                    submit_edit(burst_scenario, n=edit_n + k)
                    time.sleep(BURST_SPAN_S / 5.0)
                edit_n += 5
                burst_count += 1
                next_burst += BURST_EVERY_S
                events.append({"t": round(time.monotonic() - t0, 2),
                               "kind": "burst", "index": burst_count})
                next_steady = max(next_steady, time.monotonic() + EDIT_INTERVAL_S)
                continue
            if now >= next_steady:
                submit_edit(scenarios[edit_n % N_SCENARIOS], n=edit_n)
                edit_n += 1
                next_steady += EDIT_INTERVAL_S
                continue
            time.sleep(0.02)

        # Drain: every scenario must reach exact at its latest version.
        deadline = time.monotonic() + 180.0
        pending = set(scenarios)
        while pending and time.monotonic() < deadline:
            for scenario_id in list(pending):
                body = client.get(f"/api/v1/scenarios/{scenario_id}").json()
                if body["exact_result_version"] == body["scene_version"]:
                    pending.discard(scenario_id)
            time.sleep(0.25)
        stop.set()
        sampler_thread.join(timeout=2.0)

        # Per-job outcomes from the durable ledger.
        jobs = []
        for event in events:
            if event.get("kind") == "edit" and event.get("status") == 202:
                record = store.get_job(event["job"])
                assert record is not None, event
                jobs.append({
                    "scenario": event["scenario"],
                    "status": record.status,
                })
        status_counts: dict[str, int] = {}
        for job in jobs:
            status_counts[job["status"]] = status_counts.get(job["status"], 0) + 1

        nonterminal_final = len(store.nonterminal_jobs())
        exact_all = not pending
        superseded = status_counts.get("superseded", 0) + sum(
            s["superseded"] for s in samples[-1:]
        )
        report = {
            "compression": f"{COMPRESS:g}x (30 min -> {WALL_SECONDS:g} s)",
            "solve_seconds": SOLVE_S,
            "scenarios": N_SCENARIOS,
            "edits_submitted": len([e for e in events if e.get("kind") == "edit"]),
            "job_status_counts": status_counts,
            "final_nonterminal_jobs": nonterminal_final,
            "all_scenarios_exact": exact_all,
            "stranded_scenarios": sorted(pending),
            "max_queued_depth": max(s["queued"] for s in samples),
            "max_nonterminal_depth": max(s["nonterminal"] for s in samples),
            "final_sample": samples[-1] if samples else None,
            "gate_QUEUE_001": bool(
                exact_all and nonterminal_final == 0 and superseded > 0
            ),
            "note": "coalescing observed via superseded jobs during bursts; "
                    "bounded backlog = queue drains to zero nonterminal",
        }
        print(json.dumps(report, indent=2))
        Path("/tmp/p8perf/queue_report.json").write_text(json.dumps(report, indent=2))
        Path("/tmp/p8perf/queue_samples.json").write_text(json.dumps(samples))


if __name__ == "__main__":
    main()
