# SPDX-License-Identifier: GPL-3.0-only
"""T15 liveness tests: the publication lease and the restart drill.

Witness 2 (stale exact publication): the T13 ``OutputLease`` discipline
integrated CPU-side as :mod:`solweig_gpu.server.realtime.publish_fence`
— publish only after completion, refuse stale job-row generations, one
publication per lease, ``release()`` abandons.

Witness 6 (pending epoch permanently stuck after restart): the
crash-shape restart recovery drill — a durable ``running`` exact job row
plus its recoverable epoch must settle after a fresh runner boots.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from solweig_gpu.server import patch_codec
from solweig_gpu.server.jobs import JobRunner, RunnerContext, SiteRegistry, SolveResult
from solweig_gpu.server.realtime.publish_fence import (
    LeaseStaleError,
    PublishPendingError,
    ResultPublishLease,
)
from solweig_gpu.server.store import Store, StoreError

from tests.test_realtime_epochs import veg_add
from tests.test_realtime_r2_exact_lane import poll_until
from tests.test_server_api import SITE_ID, make_site_cache

# ---------------------------------------------------------------------------
# Witness 2: the publication lease (T13 OutputLease discipline, CPU side)
# ---------------------------------------------------------------------------


def _lease_fixture(tmp_path: Path):
    """Store + one RUNNING job row targeting scene version 1."""
    store = Store(tmp_path / "store.sqlite3", results_root=tmp_path / "scenarios")
    store.create_scenario(site_id="site", name="t15", scenario_id="ws_l")
    with store._write() as conn:
        conn.execute(
            "UPDATE scenarios SET scene_version = 1 WHERE scenario_id = 'ws_l'"
        )
        job_id = Store._insert_job_locked(
            conn, "ws_l", target_scene_version=1, request={}, edit_watermark=0
        )
    store.mark_job_running(job_id)
    return store, job_id


def _lease_payload(workspace: str, version: int):
    from solweig_gpu.incremental.geometry import RasterWindow

    arrays = {"utci": np.full((1, 2, 2), 5.0, dtype=np.float32)}
    meta, payload, checksum = patch_codec.encode_payload(arrays, ["utci"])
    manifest = patch_codec.build_manifest(
        scenario_id=workspace,
        scene_version=version,
        window=RasterWindow(0, 2, 0, 2),
        time_indices=(0,),
        variables=meta,
        payload_url=f"/api/v1/scenarios/{workspace}/results/{version}/payload",
        checksum=checksum,
        model_version="test-model",
        site_cache_version="test-cache",
    )
    return manifest, payload


def test_publish_lease_refuses_publish_before_completion(tmp_path: Path) -> None:
    """RED witness (stale exact publication, ordering half): publishing
    before the lease's completion protocol runs is refused — the result's
    job-row validity was never re-checked. Designated mutation: deleting
    the ``_completed`` gate in ``publish()`` is killed by this test."""
    store, job_id = _lease_fixture(tmp_path)
    lease = ResultPublishLease.from_job(store, store.get_job(job_id))
    manifest, payload = _lease_payload("ws_l", 1)
    with pytest.raises(PublishPendingError) as error:
        lease.publish(manifest=manifest, payload=payload)
    assert "mark_complete" in str(error.value)
    assert store.get_result("ws_l", 1) is None


def test_publish_lease_is_stale_when_the_job_row_moves_on(tmp_path: Path) -> None:
    """RED witness (stale exact publication, freshness half): the job left
    ``running`` between completion and publish — the slot was reused, so
    the result is a discard and NOTHING reaches the store. Designated
    mutation: deleting the ``_require_running()`` call inside ``publish``
    is killed by this test (the store would still refuse via
    ``ResultNotPublishable`` with the job superseded — but with a
    different, unfenced semantics; the lease must carry its own typed
    refusal)."""
    store, job_id = _lease_fixture(tmp_path)
    lease = ResultPublishLease.from_job(store, store.get_job(job_id))
    lease.mark_complete()
    store.finish_job(job_id, "superseded")
    manifest, payload = _lease_payload("ws_l", 1)
    with pytest.raises(LeaseStaleError) as error:
        lease.publish(manifest=manifest, payload=payload)
    assert error.value.status == "superseded"
    assert store.get_result("ws_l", 1) is None


def test_publish_lease_mint_refuses_a_terminal_job_row(tmp_path: Path) -> None:
    """A terminal row can never mint a lease: its outcome is the truth and
    a late solve's result must not overwrite it."""
    store, job_id = _lease_fixture(tmp_path)
    store.finish_job(job_id, "failed")
    with pytest.raises(LeaseStaleError):
        ResultPublishLease.from_job(store, store.get_job(job_id))


def test_publish_lease_publishes_exactly_once(tmp_path: Path) -> None:
    """One lease, one publication (the T13 double-publish fence)."""
    store, job_id = _lease_fixture(tmp_path)
    lease = ResultPublishLease.from_job(store, store.get_job(job_id))
    lease.mark_complete()
    manifest, payload = _lease_payload("ws_l", 1)
    lease.publish(manifest=manifest, payload=payload)
    record = store.get_result("ws_l", 1)
    assert record is not None and record.checksum == manifest["checksum"]
    with pytest.raises(StoreError):
        lease.publish(manifest=manifest, payload=payload)


def test_publish_lease_release_abandons_without_publishing(tmp_path: Path) -> None:
    """``release()`` abandons: a completed-but-released lease can no
    longer publish anything."""
    store, job_id = _lease_fixture(tmp_path)
    lease = ResultPublishLease.from_job(store, store.get_job(job_id))
    lease.mark_complete()
    lease.release()
    manifest, payload = _lease_payload("ws_l", 1)
    with pytest.raises(PublishPendingError):
        lease.publish(manifest=manifest, payload=payload)
    assert store.get_result("ws_l", 1) is None


# ---------------------------------------------------------------------------
# Witness 6: pending epoch permanently stuck after restart — the crash-shape
# restart recovery drill (running exact job + recoverable epochs at boot)
# ---------------------------------------------------------------------------


def test_running_exact_job_and_pending_epoch_settle_after_restart(
    tmp_path: Path,
) -> None:
    """Crash shape: the process died with the exact job mid-flight (durable
    row ``running``) and its epoch still recoverable. A fresh runner's
    :meth:`JobRunner.start` must recover the job (requeue) and settle the
    epoch through the job's completion — nothing may remain recoverable.
    Designated mutation: dropping the ``requeue_running_job`` call in
    ``start()`` leaves the row ``running`` forever (the tick's
    single-flight cannot mint a second job at the same target) and this
    drill times out."""
    from solweig_gpu.incremental.geometry import RasterWindow

    workspace = "ws_restart"
    store = Store(tmp_path / "store.sqlite3", results_root=tmp_path / "scenarios")
    store.create_scenario(site_id=SITE_ID, name="t15", scenario_id=workspace)
    store.append_operations(workspace, [veg_add("op_restart_1", "tree-r1")])
    store.mark_epoch_status(workspace, 0, "closed")
    committed = store.commit_epoch_reduction(
        workspace, 0, families={"vegetation_geometry": {"objects": {}}}
    )
    assert committed is not None and committed.first_time
    revision = committed.workspace_revision

    def solving(request, progress):
        arrays = {"utci": np.full((3, 32, 48), 7.0, dtype=np.float32)}
        return SolveResult(
            status="published",
            scene_version=request.target_scene_version,
            mode="full",
            window=RasterWindow(0, 32, 0, 48),
            time_indices=(0, 1, 2),
            variables=("utci",),
            arrays=arrays,
        )

    context = RunnerContext(
        store=store,
        sites=SiteRegistry({SITE_ID: {"cache_dir": make_site_cache(tmp_path)}}),
        state_root=tmp_path / "state",
    )
    crashed = JobRunner(
        context, solver_factory=lambda ctx: solving, coalescing_window_ms=0.0
    )
    epochs = [e for e in store.recoverable_epochs() if e.workspace_id == workspace]
    job_id = crashed._exact_lane_tick_workspace(workspace, epochs)
    assert job_id is not None
    # Crash: the durable row froze mid-flight.
    store.mark_job_running(job_id)
    assert store.get_job(job_id).status == "running"
    assert [e.workspace_id for e in store.recoverable_epochs()] == [workspace]

    restarted = JobRunner(
        context, solver_factory=lambda ctx: solving, coalescing_window_ms=0.0
    )
    restarted.start()
    try:
        assert poll_until(
            lambda: store.get_job(job_id).status
            in ("complete", "superseded", "failed", "cancelled"),
            timeout=30.0,
        ), f"recovered exact job never finished: {store.get_job(job_id)}"
        assert store.get_job(job_id).status == "complete"
        assert poll_until(
            lambda: not [
                e for e in store.recoverable_epochs() if e.workspace_id == workspace
            ],
            timeout=30.0,
        ), "pending epoch stuck after restart"
        assert store.get_result(workspace, revision) is not None
    finally:
        restarted.stop()
