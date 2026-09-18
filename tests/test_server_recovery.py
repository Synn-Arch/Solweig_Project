# SPDX-License-Identifier: GPL-3.0-only
"""REL-001: the worker recovers from a hard process crash mid-job.

The JobRunner's restart recovery (``start()`` requeues nonterminal jobs and
resets orphaned ``running`` rows) is exercised against a REAL process death:
a child process boots the app with a solver that sleeps mid-solve, the parent
observes the job flip to ``running`` in the durable SQLite store, then
SIGKILLs the child (no cleanup, no graceful shutdown — the closest
simulation to a worker/OOM death a deployment sees).

The parent then reopens the same state root in-process and asserts the
recovery contract:

* the interrupted job is requeued and completes after restart;
* no job row is left ``running`` (the classic stuck-row hole);
* the published-result set is exactly what the history says (no orphan
  versions, scene version == exact result version);
* idempotency keys recorded before the crash still replay/conflict after;
* store migrations are idempotent across repeated reopenings.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from solweig_gpu.server.app import create_app
from solweig_gpu.server.store import LATEST_SCHEMA_VERSION, Store

from tests.test_server_api import SITE_ID, FakeSolver, make_site_cache

ROOT_DIR = Path(__file__).resolve().parents[1]
DB_RELATIVE = "store.sqlite3"


def test_migration_5_backfills_pre_v5_family_scenarios() -> None:
    """u-e3-review NIT-1: on a pre-v5 database the new column arrives as 0
    for every scenario. Without a backfill, a deployment that had ALREADY
    pruned its family events past the retention tail (the exact prune the
    flag exists to survive) upgrades to ``carries_family_edits=0`` and
    resurrects the F1 under-route: the next TREE edit routes to the
    baseline-bound solver and publishes without the family state. The
    migration itself must set the flag from surviving family events."""
    from solweig_gpu.server.store import _MIGRATION_5

    conn = sqlite3.connect(":memory:")
    try:
        conn.execute(
            "CREATE TABLE scenarios ("
            " scenario_id TEXT PRIMARY KEY)"
        )
        conn.execute(
            "CREATE TABLE edit_events ("
            " sequence INTEGER PRIMARY KEY,"
            " scenario_id TEXT,"
            " family_json TEXT)"
        )
        conn.execute("INSERT INTO scenarios (scenario_id) VALUES ('s1')")
        conn.execute("INSERT INTO scenarios (scenario_id) VALUES ('s2')")
        conn.execute("INSERT INTO scenarios (scenario_id) VALUES ('s3')")
        # s1: family event survives the upgrade. s2: pruned before upgrade
        # (unrecoverable from events — backfill cannot help, disclosed).
        # s3: never had family edits.
        conn.execute(
            "INSERT INTO edit_events (sequence, scenario_id, family_json)"
            " VALUES (1, 's1', '{}')"
        )
        conn.execute(
            "INSERT INTO edit_events (sequence, scenario_id, family_json)"
            " VALUES (2, 's2', NULL)"
        )
        conn.executescript(_MIGRATION_5)
        flags = dict(
            conn.execute(
                "SELECT scenario_id, carries_family_edits FROM scenarios"
            ).fetchall()
        )
        assert flags["s1"] == 1, "surviving family event must set the flag"
        assert flags["s2"] == 0
        assert flags["s3"] == 0
    finally:
        conn.close()


def test_migration_6_backfills_last_reset_sequence() -> None:
    """u-e3c attack C: the routing predicate's executor-state rescue (a
    pre-v5 row whose family events were pruned before the upgrade still
    routes the executor because its executor-state directory exists) must
    be reset-aware: a pre-reset snapshot must never resurrect. The durable
    reset watermark ``last_reset_sequence`` backfills from SURVIVING reset
    events at migration time (and ``reset_scenario`` writes it going
    forward, retention-immune like the flag)."""
    from solweig_gpu.server.store import _MIGRATION_6

    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("CREATE TABLE scenarios (scenario_id TEXT PRIMARY KEY)")
        conn.execute(
            "CREATE TABLE edit_events ("
            " sequence INTEGER PRIMARY KEY,"
            " scenario_id TEXT,"
            " operation TEXT)"
        )
        conn.execute("INSERT INTO scenarios (scenario_id) VALUES ('r1')")
        conn.execute("INSERT INTO scenarios (scenario_id) VALUES ('r0')")
        conn.execute(
            "INSERT INTO edit_events (sequence, scenario_id, operation)"
            " VALUES (1, 'r1', 'update_time_row')"
        )
        conn.execute(
            "INSERT INTO edit_events (sequence, scenario_id, operation)"
            " VALUES (2, 'r1', 'reset')"
        )
        conn.executescript(_MIGRATION_6)
        watermark = dict(
            conn.execute(
                "SELECT scenario_id, last_reset_sequence FROM scenarios"
            ).fetchall()
        )
        assert watermark["r1"] == 2, "surviving reset event must set the watermark"
        assert watermark["r0"] == 0, "no surviving reset stays at the 0 default"
    finally:
        conn.close()


class SleepingSolver(FakeSolver):
    """Blocks mid-solve so the parent can kill the process mid-job."""

    def __call__(self, request, progress):
        time.sleep(30.0)  # killed long before this returns
        return super().__call__(request, progress)


def _read_job_statuses(db_path: Path) -> list[tuple[str, str]]:
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=5.0)
    try:
        rows = conn.execute(
            "SELECT job_id, status FROM jobs ORDER BY rowid"
        ).fetchall()
    finally:
        conn.close()
    return [(row[0], row[1]) for row in rows]


def _wait_for_running(db_path: Path, timeout: float = 90.0) -> str:
    """Poll the child's store (WAL readers are fine) until a job runs.

    Tolerates the file not existing yet while the child boots.
    """
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            statuses = _read_job_statuses(db_path)
        except sqlite3.OperationalError as error:
            last_error = error
            statuses = []
        for job_id, status in statuses:
            if status == "running":
                return job_id
        time.sleep(0.05)
    raise AssertionError(
        f"no job reached 'running' within {timeout}s "
        f"(last error: {last_error}; statuses={statuses})"
    )


def _child_main(state_root: Path, cache_dir: Path) -> None:
    """Child process: boot the app, submit an edit, hold mid-solve."""
    from fastapi.testclient import TestClient as ChildTestClient

    app = create_app(
        state_root=state_root,
        sites={SITE_ID: {"cache_dir": cache_dir}},
        solver_factory=lambda context: SleepingSolver(),
        coalescing_window_ms=20.0,
        requests_per_minute_per_ip=None,
        edits_per_minute=None,
    )
    edit_body = {
        "base_scene_version": 0,
        "edits": [
            {
                "operation": "add",
                "tree": {
                    "tree_id": "crash-tree",
                    "component_type": "broad_canopy",
                    "u": 0.5,
                    "v": 0.5,
                    "height_m": 9.0,
                    "canopy_diameter_m": 6.0,
                },
            }
        ],
    }
    with ChildTestClient(app) as client:
        created = client.post("/api/v1/scenarios", json={"site_id": SITE_ID})
        assert created.status_code == 201, created.text
        scenario_id = created.json()["scenario_id"]
        edit = client.post(
            f"/api/v1/scenarios/{scenario_id}/edits",
            json=edit_body,
            headers={"Idempotency-Key": "crash-edit-1"},
        )
        assert edit.status_code == 202, edit.text
        # Job is durable; the solver now sleeps. Hold the process open.
        time.sleep(60.0)


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{ROOT_DIR}{os.pathsep}{existing}" if existing else str(ROOT_DIR)
    return env


@pytest.mark.slow
class TestWorkerCrashRecovery:
    def test_sigkill_mid_job_recovers_on_restart(self, tmp_path: Path) -> None:
        cache_dir = make_site_cache(tmp_path)
        state_root = tmp_path / "state"

        child = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "child",
                str(state_root),
                str(cache_dir),
            ],
            env=_child_env(),
            cwd=str(ROOT_DIR),
        )
        try:
            db_path = state_root / DB_RELATIVE
            job_id = _wait_for_running(db_path)
        finally:
            child.kill()  # SIGKILL: no atexit, no graceful stop
            child.wait(timeout=10.0)

        # Hard-dead store: the job row is frozen at 'running' before recovery.
        statuses_before = dict(_read_job_statuses(db_path))
        assert statuses_before[job_id] == "running"

        # Migration idempotence: reopen the same store twice more.
        probe_store = Store(state_root / DB_RELATIVE, results_root=state_root / "scenarios")
        assert probe_store.schema_version == LATEST_SCHEMA_VERSION
        probe_store.close()
        probe_store2 = Store(state_root / DB_RELATIVE, results_root=state_root / "scenarios")
        assert probe_store2.schema_version == LATEST_SCHEMA_VERSION
        probe_store2.close()

        # Restart in-process on the SAME state root with a fast solver.
        solver = FakeSolver()
        app = create_app(
            state_root=state_root,
            sites={SITE_ID: {"cache_dir": cache_dir}},
            solver_factory=lambda context: solver,
            coalescing_window_ms=20.0,
            requests_per_minute_per_ip=None,
            edits_per_minute=None,
        )
        with TestClient(app) as client:
            # The interrupted job completed after being requeued.
            deadline = time.monotonic() + 30.0
            detail = {}
            while time.monotonic() < deadline:
                detail = client.get(f"/api/v1/jobs/{job_id}").json()
                if detail.get("status") not in ("queued", "running", None):
                    break
                time.sleep(0.05)
            assert detail["status"] == "complete", detail

            # Recovery invariants straight from the durable store.
            store = app.state.context.store
            assert store.nonterminal_jobs() == []  # no stuck 'running' rows
            scenario = store.get_scenario(store.get_job(job_id).scenario_id)
            assert scenario.scene_version == 1
            assert scenario.exact_result_version == 1

            # No orphan published versions: exactly baseline (0) + edit (1).
            versions = store.result_versions(scenario.scenario_id)
            assert versions == [0, 1]
            for version in versions:
                record = store.get_result(scenario.scenario_id, version)
                assert record is not None
                assert record.checksum == record.manifest["checksum"]

            # The published result for the interrupted job is servable.
            payload = client.get(
                f"/api/v1/scenarios/{scenario.scenario_id}/results/1/payload"
            )
            assert payload.status_code == 200
            assert payload.headers["X-SOLWEIG-Scene-Version"] == "1"

            # Idempotency keys recorded before the crash still work.
            replay = client.post(
                f"/api/v1/scenarios/{scenario.scenario_id}/edits",
                json={
                    "base_scene_version": 0,
                    "edits": [
                        {
                            "operation": "add",
                            "tree": {
                                "tree_id": "crash-tree",
                                "component_type": "broad_canopy",
                                "u": 0.5,
                                "v": 0.5,
                                "height_m": 9.0,
                                "canopy_diameter_m": 6.0,
                            },
                        }
                    ],
                },
                headers={"Idempotency-Key": "crash-edit-1"},
            )
            assert replay.status_code == 202
            assert replay.json()["job_id"] == job_id  # stored response replayed

            conflict = client.post(
                f"/api/v1/scenarios/{scenario.scenario_id}/edits",
                json={
                    "base_scene_version": 1,
                    "edits": [
                        {
                            "operation": "add",
                            "tree": {
                                "tree_id": "different-tree",
                                "component_type": "broad_canopy",
                                "u": 0.4,
                                "v": 0.4,
                                "height_m": 5.0,
                                "canopy_diameter_m": 4.0,
                            },
                        }
                    ],
                },
                headers={"Idempotency-Key": "crash-edit-1"},
            )
            assert conflict.status_code == 409
            assert conflict.json()["error"]["code"] == "idempotency_key_reused"

            # The runner's recovery did not double-dispatch anything: exactly
            # one solver call for the requeued job (the crashed attempt was in
            # another process and never reached this solver factory).
            assert len(solver.calls) == 1
            assert solver.calls[0].job_id == job_id


if __name__ == "__main__":  # pragma: no cover - child process driver
    if len(sys.argv) == 4 and sys.argv[1] == "child":
        _child_main(Path(sys.argv[2]), Path(sys.argv[3]))
    else:  # pragma: no cover
        raise SystemExit("usage: test_server_recovery.py child <state_root> <cache_dir>")
