# SPDX-License-Identifier: GPL-3.0-only
"""LIVE-INCIDENT 2 regression (2026-09-07, post 579008e deploy): the exact
lane refused the span ``(5, 11]`` with ``engine_refused: replay chain for
tree 'tree-gate-probe-1' is not contiguous with the layer state`` while
``exact_revision`` had already advanced 5 -> 10 without the exact lane
publishing — a FOREIGN full recompute (a family/universal edit job) had
published exact results past the lane's settled epochs and staged executor
coverage the next exact span then restored from, a snapshot whose layer
never carried the realtime-native trees.

Root cause (the incident-1 seed of 1e07f1c covers only ONE branch): the
fold-baseline seed in ``executor_bridge._build_executor`` exists only on
the fresh-rebuild branch. Any exact span whose executor restores from a
snapshot (the foreign job's staged coverage) — or rebuilds from the
post-reset void — folds its delta against a layer missing every
baseline tree, so the first structural op on one is refused with the
production message and the lane wedges behind the churn guard.

Watermark pins (hypothesis B of the incident): a LANE-OWNED publication
(the reconcile completion) settles its epochs, so the next span starts at
that revision; a FOREIGN publication settles nothing by design — the next
span stays FAT over the gap (zero-loss), never clamped to
``exact_result_version``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from solweig_gpu.server.executor_bridge import (
    PENDING_DIRECTORY_NAME,
    SNAPSHOT_FILENAME,
    STATE_DIRECTORY_NAME,
)
from solweig_gpu.server.jobs import SolveResult
from solweig_gpu.server.store import _now_utc

from tests.test_realtime_r2_exact_lane import (
    close_epoch,
    exact_jobs,
    job_rows,
    poll_until,
    sse_subscribed,
)
from tests.test_realtime_epochs import WS, make_store, make_workspace, op_item
from tests.test_realtime_routing_policy import (
    CountingHub,
    _grace_zero_runner,
    today_utc,
)
from tests.test_server_api import wait_for_job
from tests.test_server_universal_edits import (
    SITE_ID as FAMILY_SITE,
)
from tests.test_server_universal_edits import (
    _make_client,
    _scenario_ready,
    _universal,
    family_site,  # noqa: F401  (module-scoped real-site fixture)
)


def _make_incident_client(tmp_path: Path, family_site):
    from solweig_gpu.server.realtime.epochs import EpochScheduler
    from solweig_gpu.server.realtime.scheduler import FastLaneScheduler
    from solweig_gpu.server.realtime.telemetry import TelemetryRegistry

    return _make_client(
        tmp_path,
        family_site,
        epoch_scheduler_factory=lambda store, hub: EpochScheduler(
            store, hub=hub, telemetry=TelemetryRegistry()
        ),
        fast_lane_factory=lambda store, hub: FastLaneScheduler(
            store, hub=hub, telemetry=TelemetryRegistry()
        ),
        start_fast_lane=False,
    )


def _post_native_op(
    client: TestClient,
    workspace: str,
    actor: str,
    seq: int,
    base: int,
    entity: str,
    verb: str,
    payload: dict,
) -> None:
    posted = client.post(
        f"/api/v1/workspaces/{workspace}/operations",
        json={
            "actor_id": actor,
            "operations": [
                {
                    "operation_id": f"{actor}:op-{seq}",
                    "client_sequence": seq,
                    "base_revision": base,
                    "source_family": "vegetation_geometry",
                    "entity_id": entity,
                    "verb": verb,
                    "payload": payload,
                }
            ],
        },
    )
    assert posted.status_code == 200, posted.text


def _center(client: TestClient) -> tuple[float, float, float]:
    geometry = dict(client.app.state.context.sites.geometry(FAMILY_SITE))
    px = float(geometry["pixel_size_m"])
    cx = float(geometry["origin_x_m"]) + 0.5 * int(geometry["cols"]) * px
    cy = float(geometry["origin_y_m"]) - 0.5 * int(geometry["rows"]) * px
    return cx, cy, px


def _caught_up(client: TestClient, workspace: str, expected: int) -> bool:
    body = client.get(f"/api/v1/workspaces/{workspace}/operations").json()
    return (
        int(body["workspace_revision"]) >= expected
        and int(body["exact_revision"]) == int(body["workspace_revision"])
    )


def _exact_job_errors(client: TestClient, workspace: str) -> list[dict]:
    store = client.app.state.context.store
    return [
        row["error"]
        for row in job_rows(store, workspace)
        if row["request"].get("exact_lane") and row.get("error")
    ]


def _span_marker_for(
    client: TestClient, workspace: str, target_revision: int
) -> dict | None:
    """The minted exact-lane job's span marker for ``target_revision``."""
    store = client.app.state.context.store

    def minted() -> bool:
        return any(
            row["request"]["exact_lane"]["target_revision"] == target_revision
            for row in exact_jobs(store, workspace)
        )

    if not poll_until(minted, timeout=60.0):
        return None
    for row in exact_jobs(store, workspace):
        marker = row["request"]["exact_lane"]
        if marker["target_revision"] == target_revision:
            return marker
    return None


_MET_T1_30 = {
    "adapter": "meteorological_forcing",
    "operation": "update_time_row",
    "time_index": 1,
    "values": {"air_temperature": 30.0},
}


class _FreezableEpochClock:
    """Epoch-window clock the scheduler reads (its ``utc_now`` seam).

    A family edit on a realtime workspace is mirrored into the operation
    log, and its audit epoch closes within the (<= 200 ms) window — under
    a live clock the in-flight foreign job is superseded by that commit
    before it can publish. On the live box the foreign full recompute
    COMPLETED (exact_revision advanced 5 -> 10 with nothing settled), so
    the honest deterministic equivalent is to hold the epoch window open
    for the foreign job's duration: freeze on posting, release on
    completion. Store stamps (accepted_at, job rows) keep the real clock.
    """

    def __init__(self) -> None:
        self._frozen_at: str | None = None

    def freeze(self) -> None:
        self._frozen_at = _now_utc()

    def release(self) -> None:
        self._frozen_at = None

    def __call__(self) -> str:
        return self._frozen_at if self._frozen_at is not None else _now_utc()


def _make_foreign_race_client(tmp_path: Path, family_site, clock):
    from solweig_gpu.server.realtime.epochs import EpochScheduler
    from solweig_gpu.server.realtime.scheduler import FastLaneScheduler
    from solweig_gpu.server.realtime.telemetry import TelemetryRegistry

    return _make_client(
        tmp_path,
        family_site,
        epoch_scheduler_factory=lambda store, hub: EpochScheduler(
            store, hub=hub, telemetry=TelemetryRegistry(), utc_now=clock
        ),
        fast_lane_factory=lambda store, hub: FastLaneScheduler(
            store, hub=hub, telemetry=TelemetryRegistry()
        ),
        start_fast_lane=False,
    )


@pytest.mark.scientific
def test_foreign_family_edit_does_not_poison_the_next_exact_span(
    tmp_path: Path, family_site
) -> None:
    """INCIDENT-2 mechanism, end to end: a foreign family edit (the
    universal transport — a full recompute the exact lane never asked for)
    publishes exact past the lane's settled epochs AND stages executor
    coverage whose snapshot layer never carried the realtime-native trees.
    The next exact span restores from that snapshot, so its first
    structural op on a native tree must still fold — at base the restored
    layer is native-blind and the span is refused with the production
    ``replay chain ... not contiguous`` message, wedging the lane."""
    clock = _FreezableEpochClock()
    with _make_foreign_race_client(tmp_path, family_site, clock) as client:
        workspace = _scenario_ready(client)
        cx, cy, px = _center(client)
        with sse_subscribed(client, workspace):
            _post_native_op(
                client, workspace, "studio-a", 1, 0, "tree-gate-probe-1", "add",
                {"tree_id": "tree-gate-probe-1", "x_m": cx, "y_m": cy,
                 "height_m": 14.0, "canopy_radius_m": 6.0,
                 "trunk_ratio": 0.25},
            )
            assert poll_until(
                lambda: _caught_up(client, workspace, 1), timeout=180.0
            ), "the native add's exact chase never published"
            assert _exact_job_errors(client, workspace) == []

            # The foreign full recompute: a family edit through the
            # universal transport, with the epoch window held open so the
            # job completes (the live box's timing). It publishes
            # exact=True at the bumped scene version WITHOUT settling any
            # epoch (the production "heal" that advanced exact_revision
            # past the lane) and its staged snapshot — a layer built from
            # the legacy tree list alone — becomes the next exact span's
            # restore base.
            clock.freeze()
            try:
                foreign = _universal(
                    client, workspace, [_MET_T1_30], base=1, key="inc2-met"
                )
                assert foreign.status_code == 202, foreign.text
                job = wait_for_job(
                    client, foreign.json()["job_id"], timeout=240.0
                )
            finally:
                clock.release()
            assert job["status"] == "complete", job.get("error")
            scenario = client.get(f"/api/v1/scenarios/{workspace}").json()
            assert scenario["exact_result_version"] >= 2, (
                "the repro is invalid: the foreign edit did not publish "
                "exact past the lane's settled epochs"
            )

            # Zero-loss pin: the foreign publication settles NOTHING, so
            # the next span must stay FAT over the un-settled gap — its
            # consumed watermark is the settled-epoch revision (1), never
            # the foreign exact_result_version. Clamping the span start
            # to exact_revision would silently drop the gap's native
            # operations from the fold.
            _post_native_op(
                client, workspace, "studio-b", 2, scenario["scene_version"],
                "tree-gate-probe-1", "move",
                {"tree_id": "tree-gate-probe-1",
                 "x_m": cx + 3 * px, "y_m": cy - 3 * px},
            )
            marker = _span_marker_for(
                client, workspace, scenario["scene_version"] + 1
            )
            assert marker is not None, "the move's exact span never minted"
            assert marker["consumed_revision"] == 1, (
                "a foreign publication must not advance the span start "
                f"(zero-loss): {marker}"
            )

            # The span folds against the restored (foreign-staged) layer:
            # at base the move of the native tree is refused and the exact
            # lane wedges behind the churn guard.
            caught = poll_until(
                lambda: _caught_up(
                    client, workspace, scenario["scene_version"] + 1
                ),
                timeout=240.0,
            )
            errors = _exact_job_errors(client, workspace)
            assert caught and not errors, (
                "exact lane never caught up after the foreign edit "
                "(INCIDENT-2: engine_refused refusal of the restored "
                f"layer's missing native tree wedges the lane): {errors}"
            )

            store = client.app.state.context.store
            body = client.get(f"/api/v1/workspaces/{workspace}/operations").json()
            canonical = store.canonical_state_at(
                workspace, int(body["workspace_revision"])
            )
            family = (canonical.families or {}).get("vegetation_geometry") or {}
            victim = (family.get("objects") or {}).get("tree-gate-probe-1")
            assert victim is not None
            assert victim["x_m"] == pytest.approx(cx + 3 * px), victim
            assert victim["y_m"] == pytest.approx(cy - 3 * px), victim


@pytest.mark.scientific
def test_post_reset_span_folds_against_the_fold_baseline(
    tmp_path: Path, family_site
) -> None:
    """The void branch of the same poison: a scenario reset bumps the
    legacy edit watermark past every staged coverage, so every later
    exact span rebuilds from the post-reset VOID. The fold baseline at
    ``consumed_revision`` still carries the realtime-native trees (reset
    voids the LEGACY tree list, never the realtime fold), so the span's
    structural op on one must fold — at base the void layer is seeded
    with nothing and the span is refused with the production message."""
    with _make_incident_client(tmp_path, family_site) as client:
        workspace = _scenario_ready(client)
        cx, cy, px = _center(client)
        with sse_subscribed(client, workspace):
            _post_native_op(
                client, workspace, "studio-a", 1, 0, "tree-pre-reset", "add",
                {"tree_id": "tree-pre-reset", "x_m": cx, "y_m": cy,
                 "height_m": 14.0, "canopy_radius_m": 6.0,
                 "trunk_ratio": 0.25},
            )
            assert poll_until(
                lambda: _caught_up(client, workspace, 1), timeout=180.0
            ), "the pre-reset add's exact chase never published"

            reset = client.post(
                f"/api/v1/scenarios/{workspace}/reset",
                json={"base_scene_version": 1},
                headers={"Idempotency-Key": "inc2-reset"},
            )
            assert reset.status_code == 200, reset.text
            scenario = client.get(f"/api/v1/scenarios/{workspace}").json()
            assert scenario["scene_version"] == 2
            assert scenario["exact_result_version"] == 2

            # Post-reset adds replay from the void (their own events) and
            # publish; the baseline's pre-reset tree rides only the fold.
            _post_native_op(
                client, workspace, "studio-b", 2, 2, "tree-post-reset", "add",
                {"tree_id": "tree-post-reset", "x_m": cx - 6 * px,
                 "y_m": cy + 2 * px, "height_m": 12.0,
                 "canopy_radius_m": 3.5, "trunk_ratio": 0.25},
            )
            assert poll_until(
                lambda: _caught_up(client, workspace, 3), timeout=180.0
            ), "the post-reset add's exact span never published"

            # Structural op on the BASELINE-carried (pre-reset) tree: the
            # void layer must carry it or the span refuses (INCIDENT-2's
            # void twin).
            _post_native_op(
                client, workspace, "studio-c", 3, 3, "tree-pre-reset", "move",
                {"tree_id": "tree-pre-reset",
                 "x_m": cx + 4 * px, "y_m": cy - 4 * px},
            )
            caught = poll_until(
                lambda: _caught_up(client, workspace, 4), timeout=180.0
            )
            errors = _exact_job_errors(client, workspace)
            assert caught and not errors, (
                "exact lane never caught up after the reset-void span "
                "(INCIDENT-2 void twin: engine_refused refusal of the "
                f"baseline tree the void layer never seeded): {errors}"
            )

            store = client.app.state.context.store
            body = client.get(f"/api/v1/workspaces/{workspace}/operations").json()
            canonical = store.canonical_state_at(
                workspace, int(body["workspace_revision"])
            )
            family = (canonical.families or {}).get("vegetation_geometry") or {}
            victim = (family.get("objects") or {}).get("tree-pre-reset")
            assert victim is not None
            assert victim["x_m"] == pytest.approx(cx + 4 * px), victim


# ---------------------------------------------------------------------------
# Follow-up #139: the restore branch's own seed line, and the fraction
# gate's scope over the promoted-coverage door
# ---------------------------------------------------------------------------


def _state_dir(client: TestClient, workspace: str) -> Path:
    return (
        client.app.state.context.store.results_root
        / workspace
        / STATE_DIRECTORY_NAME
    )


def _pending_document(client: TestClient, workspace: str) -> dict | None:
    document = _state_dir(client, workspace) / PENDING_DIRECTORY_NAME / SNAPSHOT_FILENAME
    if not document.is_file():
        return None
    return json.loads(document.read_text())


def _settled(client: TestClient, workspace: str, *, min_revision: int) -> bool:
    body = client.get(f"/api/v1/workspaces/{workspace}/operations").json()
    revision = int(body["workspace_revision"])
    return revision >= min_revision and int(body["exact_revision"]) == revision


@pytest.mark.scientific
def test_restored_native_blind_snapshot_is_reseeded_from_the_fold_baseline(
    tmp_path: Path, family_site
) -> None:
    """#139 (a): the RESTORE branch's seed line, pinned directly.

    The e2e repro above stages a FOREIGN snapshot: its producer ran the
    legacy lane's 0.30 fraction, so the incident-2 fraction gate DROPS it
    and the span rebuilds on the FRESH branch, whose own seed (1e07f1c)
    carries the fold — the restore branch's seed call is never
    load-bearing anywhere in the suite (the review's surviving mutation
    m1: neutering that one call left every incident test green). This
    test builds the load-bearing shape instead: a SAME-lane (0.95)
    staged snapshot whose layer is native-blind — exactly what a
    pre-incident-1 exact producer staged on the live box — restored by
    the next exact span, whose structural op on the native tree must
    then fold through the restore-branch seed alone."""
    clock = _FreezableEpochClock()
    with _make_foreign_race_client(tmp_path, family_site, clock) as client:
        workspace = _scenario_ready(client)
        cx, cy, px = _center(client)
        with sse_subscribed(client, workspace):
            # One legacy ledger event (a completed family edit through the
            # frozen-clock foreign race) so the exact lane's staged
            # snapshots carry covered_sequence >= 1 — without it the
            # restore branch can never fire (covered 0 is never > the
            # reset floor 0).
            clock.freeze()
            try:
                foreign = _universal(
                    client, workspace, [_MET_T1_30], base=0, key="i139a-met"
                )
                assert foreign.status_code == 202, foreign.text
                job = wait_for_job(
                    client, foreign.json()["job_id"], timeout=240.0
                )
            finally:
                clock.release()
            assert job["status"] == "complete", job.get("error")

            # Let the held-open audit epoch close and settle first, so
            # the native add lands in its OWN epoch (deterministic
            # revisions for the polls below).
            assert poll_until(
                lambda: _settled(client, workspace, min_revision=1), timeout=180.0
            ), "the foreign edit's audit epoch never settled"

            # The native add: its exact chase adopts the post-release
            # audit chase's same-lane staged snapshot, PROMOTES it, and
            # restores from it (same lane, no fence), then publishes and
            # stages its own snapshot — whose layer now carries the
            # native tree.
            _post_native_op(
                client, workspace, "studio-a", 1, 1, "tree-i139a", "add",
                {"tree_id": "tree-i139a", "x_m": cx, "y_m": cy,
                 "height_m": 14.0, "canopy_radius_m": 6.0,
                 "trunk_ratio": 0.25},
            )
            assert poll_until(
                lambda: _settled(client, workspace, min_revision=2)
                and any(
                    edit.get("tree_id") == "tree-i139a"
                    for edit in (_pending_document(client, workspace) or {}).get(
                        "tree_edits", []
                    )
                ),
                timeout=180.0,
            ), "the native add's exact chase never published and staged"

            # Fabricate the pre-incident-1 producer's defect: strip the
            # native tree from the STAGED snapshot's tree log, so the
            # restored layer is native-blind exactly like the live box's
            # pre-fix staged coverage. The next span adopts it (same
            # lane: the fraction gate passes) and takes the RESTORE
            # branch — only the restore-branch seed can give the layer
            # the fold baseline's copy of the tree.
            pending_path = (
                _state_dir(client, workspace)
                / PENDING_DIRECTORY_NAME
                / SNAPSHOT_FILENAME
            )
            document = json.loads(pending_path.read_text())
            before = len(document["tree_edits"])
            document["tree_edits"] = [
                edit
                for edit in document["tree_edits"]
                if edit.get("tree_id") != "tree-i139a"
            ]
            assert len(document["tree_edits"]) < before, (
                "the repro is invalid: the staged snapshot never carried "
                "the native tree"
            )
            pending_path.write_text(
                json.dumps(document, indent=2, sort_keys=True) + "\n"
            )

            # Structural op on the baseline-carried native tree: the
            # span restores the native-blind snapshot, so the move folds
            # only if the restore branch seeded the fold baseline's copy
            # (mutation m1: with that seed line neutered the span refuses
            # with the production 'replay chain ... not contiguous'
            # message and the lane wedges).
            moved_x, moved_y = cx + 3 * px, cy - 3 * px
            _post_native_op(
                client, workspace, "studio-b", 2, 3, "tree-i139a", "move",
                {"tree_id": "tree-i139a", "x_m": moved_x, "y_m": moved_y},
            )
            caught = poll_until(
                lambda: _settled(client, workspace, min_revision=3), timeout=240.0
            )
            errors = _exact_job_errors(client, workspace)
            assert caught and not errors, (
                "the exact span restoring a native-blind same-lane snapshot "
                "never folded its structural op (#139a: the restore branch's "
                f"baseline seed is missing): {errors}"
            )

            store = client.app.state.context.store
            body = client.get(f"/api/v1/workspaces/{workspace}/operations").json()
            canonical = store.canonical_state_at(
                workspace, int(body["workspace_revision"])
            )
            family = (canonical.families or {}).get("vegetation_geometry") or {}
            victim = (family.get("objects") or {}).get("tree-i139a")
            assert victim is not None
            assert victim["x_m"] == pytest.approx(moved_x), victim
            assert victim["y_m"] == pytest.approx(moved_y), victim


@pytest.mark.scientific
def test_exact_span_does_not_restore_a_cross_lane_promoted_snapshot(
    tmp_path: Path, family_site
) -> None:
    """#139 (b): the fraction gate's scope covers the promoted door.

    ``_pending_restorable_by`` guards the pending-adoption door alone,
    but the coverage sidecar is per-SCENARIO, shared by every lane: once
    a legacy family job promotes a same-fraction (0.30) snapshot into
    the generations the sidecar names, a later exact span's branch
    decision sees a promoted snapshot whose fraction its 0.95 executor
    can never restore. ``restore_into_executor``'s identity fence then
    refuses with the typed ``scenario_state_unrecoverable``, which
    repeats idempotently until a reset — the exact incident-2 wedge
    class, one door later. The gate's stated intent (a snapshot this
    lane cannot restore is never adopted into a coverage its executor
    must then refuse) must hold for the promoted door too: the span
    treats the cross-lane snapshot as absent and rebuilds fresh."""
    clock = _FreezableEpochClock()
    with _make_foreign_race_client(tmp_path, family_site, clock) as client:
        workspace = _scenario_ready(client)
        cx, cy, px = _center(client)
        with sse_subscribed(client, workspace):
            # Two completed foreign family edits, both under the frozen
            # clock: the first stages a 0.30 snapshot, the second (same
            # lane, gate passes) ADOPTS and PROMOTES it — the sidecar now
            # names a cross-lane snapshot the exact lane cannot restore.
            met_two = {"adapter": "meteorological_forcing",
                       "operation": "update_time_row",
                       "time_index": 1,
                       "values": {"air_temperature": 31.0}}
            clock.freeze()
            try:
                first = _universal(
                    client, workspace, [_MET_T1_30], base=0, key="i139b-met1"
                )
                assert first.status_code == 202, first.text
                job_one = wait_for_job(
                    client, first.json()["job_id"], timeout=240.0
                )
                assert job_one["status"] == "complete", job_one.get("error")
                scene_version = client.get(
                    f"/api/v1/scenarios/{workspace}"
                ).json()["scene_version"]
                second = _universal(
                    client, workspace, [met_two], base=scene_version,
                    key="i139b-met2",
                )
                assert second.status_code == 202, second.text
                job_two = wait_for_job(
                    client, second.json()["job_id"], timeout=240.0
                )
            finally:
                clock.release()
            assert job_two["status"] == "complete", job_two.get("error")

            # The promotion precondition (non-vacuity): the sidecar must
            # actually name a promoted generation, not mere staged state.
            coverage = json.loads(
                (_state_dir(client, workspace) / "ledger-coverage.json").read_text()
            )
            assert coverage.get("generation"), (
                "the repro is invalid: the second family edit never "
                f"promoted the first one's snapshot: {coverage}"
            )

            # A native add mints the exact span against that promoted
            # cross-lane coverage. The span must rebuild FRESH (the gate
            # treats the 0.30 generation as absent for the 0.95 lane),
            # never attempt the restore whose identity fence would wedge
            # the lane behind the churn guard. NOTE: revision-parity
            # polls cannot discriminate here — the foreign publications
            # already advanced exact_revision past the settled epochs
            # (the incident's own "heal" symptom) — so the outcome is
            # read from the chase's own terminal job row.
            _post_native_op(
                client, workspace, "studio-a", 1, 2, "tree-i139b", "add",
                {"tree_id": "tree-i139b", "x_m": cx - 4 * px, "y_m": cy + 2 * px,
                 "height_m": 12.0, "canopy_radius_m": 4.0,
                 "trunk_ratio": 0.25},
            )
            # The add's epoch commits within the coalescing window after
            # the POST returns, so the chase targets scene_version + 1.
            target = int(
                client.get(f"/api/v1/scenarios/{workspace}").json()["scene_version"]
            ) + 1

            def chase_terminal() -> list[dict]:
                store = client.app.state.context.store
                return [
                    row
                    for row in exact_jobs(store, workspace)
                    if row["request"]["exact_lane"]["target_revision"] >= target
                    and row["status"] in ("complete", "superseded", "failed")
                ]

            assert poll_until(lambda: bool(chase_terminal()), timeout=240.0), (
                "the native add's exact span never minted a chase"
            )
            terminal = chase_terminal()
            errors = [row.get("error") for row in terminal if row.get("error")]
            assert not errors, (
                "an exact span faced with a promoted cross-lane snapshot "
                "wedged instead of rebuilding fresh (#139b guard scope): "
                f"{errors}"
            )
            assert not any(
                "scenario_state_unrecoverable" in json.dumps(error)
                for error in errors
            )
            assert any(row["status"] == "complete" for row in terminal), terminal

            store = client.app.state.context.store
            body = client.get(f"/api/v1/workspaces/{workspace}/operations").json()
            canonical = store.canonical_state_at(
                workspace, int(body["workspace_revision"])
            )
            assert canonical is not None, (
                "the span completed but its revision never settled "
                "canonically (#139b)"
            )
            family = (canonical.families or {}).get("vegetation_geometry") or {}
            victim = (family.get("objects") or {}).get("tree-i139b")
            assert victim is not None
            assert victim["x_m"] == pytest.approx(cx - 4 * px), victim
            assert victim["y_m"] == pytest.approx(cy + 2 * px), victim


# ---------------------------------------------------------------------------
# Watermark pins (hypothesis B): which publications may advance the span
# start, and which must leave the span fat (zero-loss)
# ---------------------------------------------------------------------------


def test_lane_owned_reconcile_publication_advances_the_span_watermark(
    tmp_path: Path,
) -> None:
    """A reconcile completion is LANE-OWNED: it settles its epochs, so the
    NEXT span mint starts at that published revision (R_pub) — the
    consumed watermark can never sit behind a publication the lane itself
    made."""
    store = make_store(tmp_path)
    make_workspace(store, WS)
    hub = CountingHub(count=0)
    runner = _grace_zero_runner(tmp_path, store, hub)

    store.publish_result(
        WS, 0, manifest={"checksum": "c0", "scene_version": 0}, payload=b"p0", exact=True
    )
    store.append_operations(WS, [op_item("op_wm_1")])
    close_epoch(store, epoch_id=0)
    store.append_operations(WS, [op_item("op_wm_2")])
    close_epoch(store, epoch_id=1)
    runner._exact_lane_tick()  # gated: debt recorded
    assert store.reconcile_state(WS)[1] is True

    runner._idle_reconcile_tick(now_monotonic=100.0)
    runner._idle_reconcile_tick(now_monotonic=200.0)
    pending = [row for row in exact_jobs(store, WS) if row["status"] == "queued"]
    assert len(pending) == 1, job_rows(store, WS)
    reconcile = pending[0]["request"]["exact_lane"]
    r_pub = reconcile["target_revision"]
    assert r_pub == 2

    # The lane-owned publication: the reconcile job completes and settles
    # every epoch through R_pub.
    store.mark_job_running(pending[0]["job_id"])
    runner._finalize(
        store.require_job(pending[0]["job_id"]),
        store.require_scenario(WS),
        SolveResult(status="no-op", scene_version=r_pub),
        duration_ms=1.0,
    )
    assert store.require_job(pending[0]["job_id"]).status == "complete"
    assert {e.workspace_id for e in store.recoverable_epochs()} == set()
    assert store.reconcile_state(WS) == (today_utc(), False)

    # The next span starts AT R_pub — never at the pre-reconcile
    # watermark the superseded attempt would have left behind.
    hub.count = 1
    store.append_operations(WS, [op_item("op_wm_3")])
    close_epoch(store, epoch_id=2)
    runner._exact_lane_tick()
    marker = _store_span_marker(store, target_revision=3)
    assert marker is not None, "the post-reconcile span never minted"
    assert marker["consumed_revision"] == r_pub, (
        "a lane-owned publication must advance the consumed watermark to "
        f"R_pub={r_pub}: {marker}"
    )
    assert marker["span_hi"] == 3


def test_foreign_publication_leaves_the_span_zero_loss_fat(
    tmp_path: Path,
) -> None:
    """A FOREIGN publication (a non-lane job publishing exact — the
    incident's "heal") settles NO epochs: the next span must still cover
    the whole gap from the settled watermark, folding the foreign
    revision's native operations exactly once. This pins the design
    decision the incident fix must NOT break: clamping the span start to
    ``exact_result_version`` would silently drop those operations."""
    store = make_store(tmp_path)
    make_workspace(store, WS)
    hub = CountingHub(count=1)  # the gate is open: plain chases mint
    runner = _grace_zero_runner(tmp_path, store, hub)

    store.publish_result(
        WS, 0, manifest={"checksum": "c0", "scene_version": 0}, payload=b"p0", exact=True
    )
    store.append_operations(WS, [op_item("op_fat_1")])
    close_epoch(store, epoch_id=0)
    runner._exact_lane_tick()
    chase = [row for row in exact_jobs(store, WS) if row["status"] == "queued"]
    assert len(chase) == 1, job_rows(store, WS)
    store.mark_job_running(chase[0]["job_id"])
    runner._finalize(
        store.require_job(chase[0]["job_id"]),
        store.require_scenario(WS),
        SolveResult(status="no-op", scene_version=1),
        duration_ms=1.0,
    )
    assert store.require_job(chase[0]["job_id"]).status == "complete"

    # The foreign heal: revision 2 commits (a new epoch) and a NON-lane
    # job publishes exact at scene_version=2 — no epoch settlement.
    store.append_operations(WS, [op_item("op_fat_2")])
    close_epoch(store, epoch_id=1)
    committed = store.commit_epoch_reduction(
        WS, 1, families={"vegetation_geometry": {"objects": {}}}
    )
    assert committed is not None and committed.workspace_revision == 2
    store.publish_result(
        WS,
        2,
        manifest={"checksum": "c2", "scene_version": 2},
        payload=b"p2",
        exact=True,
    )
    scenario = store.require_scenario(WS)
    assert scenario.exact_result_version == 2

    # Revision 3 commits; the next span is FAT: it starts at the SETTLED
    # watermark (1), spans over the foreign revision, and folds both gaps.
    store.append_operations(WS, [op_item("op_fat_3")])
    close_epoch(store, epoch_id=2)
    store.commit_epoch_reduction(
        WS, 2, families={"vegetation_geometry": {"objects": {}}}
    )
    runner._exact_lane_tick()
    marker = _store_span_marker(store, target_revision=3)
    assert marker is not None, "the post-heal span never minted"
    assert marker["consumed_revision"] == 1, (
        "the foreign publication settled nothing — the span must stay "
        f"fat over the gap (zero-loss): {marker}"
    )
    assert marker["span_hi"] == 3


def _store_span_marker(store, *, target_revision: int) -> dict | None:
    for row in exact_jobs(store, WS):
        marker = row["request"]["exact_lane"]
        if marker["target_revision"] == target_revision:
            return marker
    return None
