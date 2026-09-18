# SPDX-License-Identifier: GPL-3.0-only
"""exact_progress SSE emission tests (operational-stability wave ①).

The studio can only show a countdown if the server TELLS it what is in
flight: the exact lane mints jobs server-side, job ids never reach SSE
clients, and before this wave the exact_revision event fired at COMPLETION
only. These tests pin the additive ``exact_progress`` named event:

* emitted at the transitions that matter — mint (queued + queue position),
  dispatch (running), mode-known (eta becomes possible), terminal
  (complete/failed/superseded/cancelled via the ``_finish_job`` chokepoint
  and the cancel route);
* exact-lane jobs ONLY — legacy funnel jobs keep their poll-based contract
  and emit nothing;
* never raises: a broken hub or store must not fail a solve (telemetry is
  best-effort by construction);
* roster + heartbeat-cap companions live in test_realtime_broadcast.py;
  the catch-up span query (wave ②) is pinned at the bottom.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from solweig_gpu.server.jobs import JobRunner, RunnerContext, SiteRegistry
from solweig_gpu.server.store import Store

from tests.test_realtime_epochs import WS, make_store, make_workspace, op_item
from tests.test_realtime_r2_exact_lane import close_epoch
from tests.test_server_api import SITE_ID, make_site_cache


class RecordingHub:
    """A broadcast hub seam that records exact_progress publishes.

    Only the methods the runner may call matter; anything else failing
    loudly is a test bug, not a production path.
    """

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def broadcast_exact_progress(self, workspace_id: str, event: dict[str, Any]) -> None:
        self.events.append((workspace_id, json.loads(json.dumps(event))))


def make_emission_runner(
    tmp_path: Path, store: Store, hub: RecordingHub | None
) -> JobRunner:
    context = RunnerContext(
        store=store,
        sites=SiteRegistry({SITE_ID: {"cache_dir": make_site_cache(tmp_path)}}),
        state_root=tmp_path / "state",
    )
    return JobRunner(
        context,
        solver_factory=lambda ctx: (lambda request, progress: None),
        coalescing_window_ms=0.0,
        broadcast_hub=hub,
    )


def mint_exact_job(
    tmp_path: Path, store: Store, hub: RecordingHub | None = None
) -> tuple[JobRunner, str]:
    """One queued exact-lane job, minted through the runner's real tick."""
    make_workspace(store, WS)
    store.append_operations(WS, [op_item("op_1")])
    close_epoch(store)
    runner = make_emission_runner(tmp_path, store, hub)
    runner._exact_lane_tick()
    jobs = store.nonterminal_jobs()
    exact = [
        job
        for job in jobs
        if isinstance(job.request, dict) and job.request.get("exact_lane")
    ]
    assert len(exact) == 1, f"tick should mint exactly one exact job, got {exact}"
    return runner, exact[0].job_id


def test_emit_silently_noops_without_a_hub(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    runner, job_id = mint_exact_job(tmp_path, store, hub=None)
    runner._emit_exact_progress(job_id)  # must not raise
    runner._finish_job(job_id, "superseded")  # must not raise


def test_emit_requires_exact_lane(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    make_workspace(store, WS)
    hub = RecordingHub()
    runner = make_emission_runner(tmp_path, store, hub)
    # The workspace's baseline job (legacy lane, no exact_lane marker) is
    # the negative case: it must never appear on the exact_progress plane.
    legacy = [job for job in store.nonterminal_jobs()]
    assert legacy, "baseline job expected"
    for job in legacy:
        runner._emit_exact_progress(job.job_id)
    assert hub.events == []


def test_emit_queued_carries_queue_position(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    hub = RecordingHub()
    runner, job_id = mint_exact_job(tmp_path, store, hub)
    hub.events.clear()
    runner._emit_exact_progress(job_id)
    assert len(hub.events) == 1
    workspace, event = hub.events[0]
    assert workspace == WS
    assert event["job_id"] == job_id
    assert event["status"] == "queued"
    assert isinstance(event["target_revision"], int)
    assert event["queue_position"] >= 1
    # No mode yet → no invented estimate.
    assert event["eta_seconds"] is None


def test_emit_running_then_mode_known(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    hub = RecordingHub()
    runner, job_id = mint_exact_job(tmp_path, store, hub)
    hub.events.clear()

    store.mark_job_running(job_id)
    runner._emit_exact_progress(job_id)
    assert hub.events[-1][1]["status"] == "running"

    store.update_job_progress(job_id, mode="windowed", progress={"stage": "solve"})
    runner._emit_exact_progress(job_id)
    event = hub.events[-1][1]
    assert event["status"] == "running"
    # eta fields mirror the job body's additive contract: a number with a
    # basis, or an honest null — never a missing key.
    assert "eta_seconds" in event and "eta_basis" in event


def test_finish_job_emits_terminal_for_every_status(tmp_path: Path) -> None:
    for status in ("complete", "failed", "superseded", "cancelled"):
        root = tmp_path / f"term-{status}"
        root.mkdir()
        store = make_store(root)
        hub = RecordingHub()
        runner, job_id = mint_exact_job(root, store, hub)
        hub.events.clear()
        runner._finish_job(job_id, status)
        assert len(hub.events) == 1, f"status {status}: one terminal event"
        event = hub.events[0][1]
        assert event["status"] == status
        assert event["job_id"] == job_id
        # Terminal shapes carry no estimate (mirrors GET /jobs/{id}).
        assert event["eta_seconds"] is None and event["eta_basis"] is None
        assert store.get_job(job_id).status == status


def test_emit_never_raises_on_a_broken_hub(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    runner, job_id = mint_exact_job(tmp_path, store, hub=None)

    class ExplodingHub:
        def broadcast_exact_progress(self, workspace_id: str, event: dict[str, Any]) -> None:
            raise RuntimeError("hub is down")

    runner.broadcast_hub = ExplodingHub()
    runner._emit_exact_progress(job_id)  # swallowed, logged
    runner._finish_job(job_id, "failed")  # finish still durably lands
    assert store.get_job(job_id).status == "failed"


# ---------------------------------------------------------------------------
# Wave ② — catch-up span query (zero-loss fence reads a closed interval)
# ---------------------------------------------------------------------------


def test_operations_in_span_returns_inclusive_upper_bound(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    make_workspace(store, WS)
    for index in range(5):
        records = store.append_operations(
            WS, [op_item(f"span-op-{index}", entity_id=f"tree-{index}")]
        )
        assert records[0].server_sequence == index + 1

    # (after, through] semantics: the executor-bridge re-fold reads the ops
    # the ledger event span covers, never re-reading the anchor itself.
    span = store.operations_in_span(WS, 0, 3)
    assert [op.server_sequence for op in span] == [1, 2, 3]
    span = store.operations_in_span(WS, 3, 5)
    assert [op.server_sequence for op in span] == [4, 5]
    assert store.operations_in_span(WS, 5, 5) == []
    # Other workspaces never leak in.
    assert store.operations_in_span("ws-none", 0, 5) == []
