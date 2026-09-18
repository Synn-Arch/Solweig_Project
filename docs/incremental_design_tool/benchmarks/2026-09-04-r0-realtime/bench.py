"""R0 independent reproduction of handoff section 10.1 live HTTP benchmark.

Two single-step (time_indices [12]) tree-add jobs on fresh scenarios, one
full-day (range(24)) job, scenario-creation latency x3, host load before/after
every job. Evidence -> /tmp/r0_perf/evidence.json.
"""
import json
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone

B = "http://127.0.0.1:8001/api/v1"
POLL_INTERVAL_S = 3.0
RATE_LIMIT_SLEEP_S = 60.0


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def req(method, path, body=None, timeout=60):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(B + path, data=data, method=method)
    r.add_header("Content-Type", "application/json")
    if body is not None:
        r.add_header("Idempotency-Key", str(uuid.uuid4()))
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return json.loads(resp.read() or b"{}")


def load_avgs():
    out = subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()
    # macOS: "... load averages: 1.23 4.56 7.89"
    tail = out.split("load averages:")[-1]
    vals = [float(x) for x in tail.split()]
    return {"uptime_text": out, "load1": vals[0], "load5": vals[1], "load15": vals[2]}


def wait_complete(job_id, log):
    """Poll GET /jobs/{id} every >=3 s until status == 'complete'. 429 -> sleep 60 s."""
    polls = 0
    status = None
    t0 = time.monotonic()
    while True:
        try:
            job = req("GET", f"/jobs/{job_id}")
        except urllib.error.HTTPError as e:
            if e.code == 429:
                log.append({"event": "429", "at": now_iso(), "slept_s": RATE_LIMIT_SLEEP_S})
                time.sleep(RATE_LIMIT_SLEEP_S)
                continue
            raise
        polls += 1
        status = job.get("status")
        if status in ("complete", "failed", "cancelled"):
            return job, polls, time.monotonic() - t0, status
        time.sleep(POLL_INTERVAL_S)


def tree_edit_body(scene_version, time_indices):
    return {
        "base_scene_version": scene_version,
        "edits": [{
            "operation": "add",
            "tree": {
                "tree_id": "bm-1",
                "component_type": "broad_canopy",
                "u": 0.575,
                "v": 0.345,
                "height_m": 18.0,
                "canopy_diameter_m": 11.0,
                "trunk_ratio": 0.25,
                "transmissivity": 0.03,
                "phenology": "deciduous",
            },
        }],
        "requested_result": {"time_indices": time_indices},
    }


def run_benchmark(label, time_indices, log):
    entry = {"label": label, "time_indices": time_indices, "started_at": now_iso()}

    entry["load_before"] = load_avgs()

    t0 = time.monotonic()
    scn = req("POST", "/scenarios", {"site_id": "site_500"})
    entry["scenario_create_seconds"] = round(time.monotonic() - t0, 3)
    entry["scenario_id"] = scn.get("scenario_id")
    entry["scenario_status"] = scn.get("status")
    entry["scene_version"] = scn.get("scene_version")

    edit = req("POST", f"/scenarios/{scn['scenario_id']}/edits",
               tree_edit_body(scn["scene_version"], time_indices))
    entry["job_id"] = edit.get("job_id")
    entry["edit_submit_at"] = now_iso()

    job, polls, poll_wall_s, final_status = wait_complete(entry["job_id"], log)
    entry["final_status"] = final_status
    entry["polls"] = polls
    entry["poll_wall_seconds"] = round(poll_wall_s, 2)
    entry["metrics"] = job.get("metrics")
    entry["job_status_fields"] = {k: job.get(k) for k in ("status", "error", "created_at", "updated_at") if k in job}
    entry["finished_at"] = now_iso()
    entry["load_after"] = load_avgs()
    return entry


def main():
    evidence = {
        "agent": "R0 performance-reproduction",
        "started_at": now_iso(),
        "api_base": B,
        "poll_interval_s": POLL_INTERVAL_S,
        "commit_sha": subprocess.run(["git", "-C", "/Users/alansynn/Workspace/solweig", "rev-parse", "HEAD"],
                                     capture_output=True, text=True).stdout.strip(),
        "host": subprocess.run(["uname", "-a"], capture_output=True, text=True).stdout.strip(),
        "python_version": subprocess.run(["/Users/alansynn/Workspace/solweig/.venv/bin/python", "--version"],
                                         capture_output=True, text=True).stdout.strip(),
        "server_process": subprocess.run(
            ["ps", "-p", open("/tmp/r0_perf/server_pid").read().strip(), "-o", "pid,etime,command"],
            capture_output=True, text=True).stdout.strip(),
    }
    log = []
    runs = []

    # Scenario-creation latency x3 + two single-step jobs + one full-day job:
    # creation timings come from the 3 scenario POSTs below (one per run).
    runs.append(run_benchmark("single-step rep1", [12], log))
    runs.append(run_benchmark("single-step rep2", [12], log))
    runs.append(run_benchmark("full-day", list(range(24)), log))

    evidence["runs"] = runs
    evidence["rate_limit_events"] = log
    evidence["finished_at"] = now_iso()
    evidence["load_at_end"] = load_avgs()

    with open("/tmp/r0_perf/evidence.json", "w") as f:
        json.dump(evidence, f, indent=2)

    # Console summary
    for r in runs:
        m = r.get("metrics") or {}
        print(f"{r['label']}: status={r['final_status']} "
              f"create={r['scenario_create_seconds']}s "
              f"duration_ms={m.get('duration_ms')} svf={m.get('svf_seconds')} "
              f"tloop={m.get('time_loop_seconds')} "
              f"read_wf={m.get('read_window_fraction')} write_wf={m.get('window_fraction')} "
              f"load {r['load_before']['load1']} -> {r['load_after']['load1']}")


if __name__ == "__main__":
    main()
