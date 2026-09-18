# SPDX-License-Identifier: GPL-3.0-only
"""Live incident 3: the mixed-plane poisoned span (failing-first).

PRODUCTION SHAPE (scn_shared_world @ 2026-09-07, read-only forensics): the
exact lane wedged at ``exact_result_version 10`` while
``scene_version/fast_revision 17``. Every exact job since 18:37Z failed at
stage ``windowing`` in ~8s with::

    SourceDeltaError: non-contiguous change chain for object
    'tree-mtrl1chf-01-dmnp': before-state does not match the preceding
    after-state

The op/evidence chain (verbatim): realtime ops seq 1-10 native healthy;
seq 11 ``legacy-evt_...`` actor ``legacy_edits`` verb ``add`` entity
``tree-mtrl1chf-01-dmnp`` epoch 10 (height 12.0 m, canopy radius 3.5 m,
trunk ratio 0.25); seq 12 studio move SAME epoch 10; seq 13 move NEXT
epoch 11; seq 14-16 native moves of other trees. Ledger: reset seq 1,
add seq 2. Scenario: ``last_reset_sequence 1``, coverage stuck at the
reset, exact lane ``{consumed_revision 13, consumed_seq 12, span_hi 16,
target_revision 17}``.

ROOT CAUSE: the exact span's executor batch is assembled from TWO planes
— the ledger window replay (``_pending_commands``: the ADD event,
``None -> addpos``) and the epoch-span fold (``reducer.reduce_epoch``:
the native moves, ``op1_after -> op2_after``). For the poisoned id both
planes emit ObjectStateChange records in ONE batch; the native op-1 sits
BETWEEN them in the scene's history; ``_coalesce_object_changes``
demands chain contiguity and refuses; ``SourceDeltaError`` escapes the
bridge's typed catch ladder (it is a sibling of ``EditStateError``, not
a subclass), the job dies as a raw ``job_failed`` WITHOUT settling its
span, and the chase re-mints the same poisoned span forever while any
subscriber watches.

CONTRACT UNDER TEST (fix semantics):

1. The span job must COMPLETE with the tree at the final native
   position (native ops after the legacy add win as ordered).
2. PER-OBJECT CHAIN ANCHORING: for ids whose ledger-window events the
   same job replays, the fold's delta must chain onto the LEDGER's
   after-state — the fold re-derives that object's native ops since its
   last ledger event (here BOTH op-1 and op-2: ``[addpos -> op2]``),
   never since ``consumed_seq`` alone.
3. SEED COHERENCE: the void/fresh executor seed must not pre-plant an
   id whose ledger-window event the same job will replay (add-only
   contract for already-present ids).
4. ZERO-LOSS FENCES stay: span completeness, per-op epoch revision,
   B1b unfoldable-native refusal, and the ``legacy_ops_skipped`` /
   ``span_native_ops`` lane metrics keep their meaning.
5. Any residual coalesce refusal surfaces as a TYPED ``edit_rejected``
   that settles — never a raw ``SourceDeltaError`` rethrow.

Positions below are mapped into the 256 m tiny site preserving the
production edit structure (same-epoch legacy add + native move, next
epoch move, healthy-tree moves riding the same span) and the production
tree's non-positional fields verbatim (height 12.0 m, canopy diameter
7.0 m -> radius 3.5 m, trunk ratio 0.25).
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from solweig_gpu.server.jobs import LEGACY_ACTOR_ID, SolveResult
from solweig_gpu.server.store import Store, uv_to_world

from tests.test_realtime_epochs import (
    WS,
    epoch_of,
    make_store,
    make_workspace,
    op_item,
)
from tests.test_realtime_r2_exact_lane import (
    SseSession,
    _arrays_at,
    _assert_nan_aware_bitwise_equal,
    _capture_commands_stub,
    _isolated_scheduler_factory,
    _running_exact_job,
    _wait_exact_caught_up,
    close_epoch,
    exact_jobs,
    exact_request,
    make_runner,
    poll_until,
)
from tests.test_server_api import make_tree
from tests.test_server_universal_edits import (
    SITE_ID as FAMILY_SITE,
)
from tests.test_server_universal_edits import (
    _make_client,
    _reset_to_baseline,
    _scenario_ready,
    family_site,
)

# ---------------------------------------------------------------------------
# The production poison, mapped into the tiny site
# ---------------------------------------------------------------------------

#: The exact production tree id (the error message names it verbatim).
POISON_ID = "tree-mtrl1chf-01-dmnp"
HEALTHY_ID = "tree-healthy-lead-01"

#: Production tree fields, verbatim (height 12 m, canopy diameter 7 m).
POISON_FIELDS = {
    "height_m": 12.0,
    "canopy_diameter_m": 7.0,
    "trunk_ratio": 0.25,
}
HEALTHY_FIELDS = {
    "height_m": 13.0,
    "canopy_diameter_m": 8.0,
    "trunk_ratio": 0.25,
}

#: Site-UV targets for the poison tree: legacy add position, native move
#: op-1 (same epoch, production seq 12), native move op-2 (next epoch,
#: production seq 13). Tens of metres apart so their influence windows
#: overlap — the parity twins below are non-vacuous differentials.
ADD_UV = (0.5027, 0.5215)
P1_UV = (0.4261, 0.6642)
P2_UV = (0.6128, 0.5803)

#: The healthy tree's add position and its second-span move.
HEALTHY_UV = (0.3125, 0.4218)
HEALTHY2_UV = (0.3611, 0.4722)

CHAIN_NEEDLE = "non-contiguous change chain"


def _site_geometry(client: TestClient) -> dict[str, Any]:
    return dict(client.app.state.context.sites.geometry(FAMILY_SITE))


def _world_of(client: TestClient, uv: tuple[float, float]) -> tuple[float, float]:
    """The EXACT float pair every lane derives the same position from.

    Legacy /edits posts the uv; the funnel and the executor replay derive
    ``uv_to_world(u, v)``; native operations post these pre-computed
    world metres — one float derivation, identical bits in every lane.
    """
    geometry = _site_geometry(client)
    return uv_to_world(
        uv[0],
        uv[1],
        rows=int(geometry["rows"]),
        cols=int(geometry["cols"]),
        pixel_size_m=float(geometry["pixel_size_m"]),
        origin_x_m=float(geometry["origin_x_m"]),
        origin_y_m=float(geometry["origin_y_m"]),
    )


def _poison_tree(uv: tuple[float, float]) -> dict[str, Any]:
    return make_tree(POISON_ID, u=uv[0], v=uv[1], **POISON_FIELDS)


def _healthy_tree(uv: tuple[float, float]) -> dict[str, Any]:
    return make_tree(HEALTHY_ID, u=uv[0], v=uv[1], **HEALTHY_FIELDS)


def _native_add_item(
    client: TestClient,
    operation_id: str,
    entity_id: str,
    uv: tuple[float, float],
    fields: dict[str, Any],
    *,
    client_sequence: int,
) -> dict[str, Any]:
    x_m, y_m = _world_of(client, uv)
    return {
        "operation_id": operation_id,
        "client_sequence": client_sequence,
        "base_revision": 0,
        "source_family": "vegetation_geometry",
        "entity_id": entity_id,
        "verb": "add",
        "payload": {
            "values": {
                "x_m": x_m,
                "y_m": y_m,
                "height_m": float(fields["height_m"]),
                "canopy_radius_m": float(fields["canopy_diameter_m"]) / 2.0,
                "trunk_ratio": float(fields["trunk_ratio"]),
            }
        },
    }


def _native_move_item(
    client: TestClient,
    operation_id: str,
    entity_id: str,
    uv: tuple[float, float],
    *,
    client_sequence: int,
) -> dict[str, Any]:
    x_m, y_m = _world_of(client, uv)
    return {
        "operation_id": operation_id,
        "client_sequence": client_sequence,
        "base_revision": 0,
        "source_family": "vegetation_geometry",
        "entity_id": entity_id,
        "verb": "move",
        "payload": {"values": {"x_m": x_m, "y_m": y_m}},
    }


def _post_native(client: TestClient, workspace: str, operations: list[dict]) -> None:
    sanitized = [
        {k: v for k, v in op.items() if k not in ("actor_id", "received_at")}
        for op in operations
    ]
    posted = client.post(
        f"/api/v1/workspaces/{workspace}/operations",
        json={"actor_id": "incident3", "operations": sanitized},
    )
    assert posted.status_code == 200, posted.text


def _scene_version(client: TestClient, workspace: str) -> int:
    return int(client.get(f"/api/v1/scenarios/{workspace}").json()["scene_version"])


def _legacy_tree_edit(
    client: TestClient,
    workspace: str,
    edits: list[dict],
    *,
    key: str,
) -> dict:
    response = client.post(
        f"/api/v1/scenarios/{workspace}/edits",
        json={
            "base_scene_version": _scene_version(client, workspace),
            "edits": edits,
        },
        headers={"Idempotency-Key": key},
    )
    assert response.status_code == 202, response.text
    return response.json()


def _exact_chain_failures(store: Store, workspace: str) -> list[dict[str, Any]]:
    """Exact-lane job errors carrying the production coalesce refusal."""
    failures = []
    for row in exact_jobs(store, workspace):
        error = row.get("error") or {}
        if CHAIN_NEEDLE in json.dumps(error):
            failures.append(error)
    return failures


def _op_record(store: Store, workspace: str, operation_id: str):
    for record in store.operations_since(workspace, 0):
        if record.operation_id == operation_id:
            return record
    return None


def _epoch_status(store: Store, workspace: str, epoch_id: int) -> str | None:
    for epoch in store.epoch_records(workspace):
        if int(epoch.epoch_id) == int(epoch_id):
            return str(epoch.status)
    return None


def _wait_epoch_settled(
    store: Store, workspace: str, epoch_id: int, *, timeout: float = 120.0
) -> str:
    status = poll_until(
        lambda: _epoch_status(store, workspace, epoch_id) == "exact_targeted"
        and "exact_targeted",
        timeout=timeout,
    )
    assert status == "exact_targeted", (
        f"epoch {epoch_id} never settled "
        f"(last: {_epoch_status(store, workspace, epoch_id)})"
    )
    return status


def _canonical_object(store: Store, workspace: str, tree_id: str) -> dict[str, Any]:
    scenario = store.require_scenario(workspace)
    state = store.canonical_state_at(workspace, int(scenario.scene_version))
    assert state is not None, "no canonical state at the current revision"
    objects = state.families["vegetation_geometry"]["objects"]
    assert tree_id in objects, (sorted(objects), "poisoned tree missing from canon")
    return dict(objects[tree_id])


def _assert_tree_state(
    fields: dict[str, Any],
    *,
    x_m: float,
    y_m: float,
    height_m: float,
    canopy_radius_m: float,
    trunk_ratio: float,
) -> None:
    """Bitwise state pin (float equality — never a tolerance)."""
    assert fields["x_m"] == x_m, (fields, x_m)
    assert fields["y_m"] == y_m, (fields, y_m)
    assert fields["height_m"] == height_m, fields
    assert fields["canopy_radius_m"] == canopy_radius_m, fields
    assert fields["trunk_ratio"] == trunk_ratio, fields


@contextmanager
def _hold_epochs(client: TestClient):
    """Deterministic same-epoch landings (review MINOR, timing-fragility).

    The same-epoch repro gate below raced the scheduler's 200 ms close
    cadence: a close between the /edits POST and the native-op POST
    drains the ledger window and defuses the poison (a false outcome,
    not a caught bug). Holding the scheduler loop around the critical
    POST pair makes the epoch assignment deterministic — appends go to
    the open epoch, nothing closes it mid-pair. The tick domain is
    service-contract-bound (50-200 ms), so widening the tick is not an
    option; stopping the loop is.
    """
    scheduler = client.app.state.epoch_scheduler
    scheduler.stop()
    try:
        yield scheduler
    finally:
        scheduler.start()


def _drive_mixed_plane_poison(client: TestClient, workspace: str) -> None:
    """The production sequence, through the public routes, on one scenario.

    reset -> native healthy add (gate open, exact settles) -> legacy
    /edits ADD of the poisoned tree -> native move op-1 in the SAME epoch
    (production seq 11 + 12) -> epoch settles -> native move op-2 in the
    NEXT epoch plus a healthy-tree move (production seq 13 + 14-16).
    """
    store = client.app.state.context.store
    _reset_to_baseline(client, workspace, key="inc3-reset")

    # The collaborative era opens with a healthy native add; the exact
    # lane settles it (coverage + canonical state advance).
    _post_native(
        client,
        workspace,
        [
            _native_add_item(
                client,
                "inc3_healthy_add",
                HEALTHY_ID,
                HEALTHY_UV,
                HEALTHY_FIELDS,
                client_sequence=1,
            )
        ],
    )
    assert _wait_exact_caught_up(client, workspace) is not None, (
        "healthy native add never settled"
    )

    # The legacy /edits ADD and the same-epoch native move (production:
    # seq 12 joined seq 11's epoch), posted under a HELD scheduler so the
    # pair lands in one epoch deterministically (never a cadence race).
    with _hold_epochs(client):
        _legacy_tree_edit(
            client,
            workspace,
            [{"operation": "add", "tree": _poison_tree(ADD_UV)}],
            key="studio-fc37391f-edit-1",
        )
        _post_native(
            client,
            workspace,
            [
                _native_move_item(
                    client, "inc3_move_op1", POISON_ID, P1_UV, client_sequence=2
                )
            ],
        )
    # Repro gate: the funnel op and op-1 must share one epoch (asserted
    # on the durable rows — epoch ids are assigned at append time).
    funnel_op = next(
        (
            record
            for record in store.operations_since(workspace, 0)
            if record.actor_id == LEGACY_ACTOR_ID
            and record.entity_id == POISON_ID
        ),
        None,
    )
    assert funnel_op is not None, "the legacy add never funnelled into the op log"
    op1 = _op_record(store, workspace, "inc3_move_op1")
    assert op1 is not None
    assert int(funnel_op.epoch_id) == int(op1.epoch_id), (
        "repro gate missed: the funnel add and the native move landed in "
        f"different epochs ({funnel_op.epoch_id} vs {op1.epoch_id}) — the "
        "same-epoch production shape was not reproduced; rerun"
    )

    # The funnel epoch settles: at base via the B1b typed refusal (the
    # span's native move names an id no fold baseline knows), at the fix
    # via the anchored fold's real publication. Either way the span is
    # settled and the canonical state carries the id at op-1's position.
    _wait_epoch_settled(store, workspace, int(funnel_op.epoch_id))

    # NEXT epoch: the second native move of the poisoned tree plus the
    # healthy tree's move (production seq 13 + 14-16 riding one span).
    _post_native(
        client,
        workspace,
        [
            _native_move_item(
                client, "inc3_move_op2", POISON_ID, P2_UV, client_sequence=3
            ),
            _native_move_item(
                client,
                "inc3_healthy_move",
                HEALTHY_ID,
                HEALTHY2_UV,
                client_sequence=4,
            ),
        ],
    )


# ---------------------------------------------------------------------------
# T1: the exact production shape — RED at base with the verbatim error
# ---------------------------------------------------------------------------


@pytest.mark.scientific
def test_t1_production_shape_mixed_plane_span_completes(
    tmp_path: Path, family_site
) -> None:
    """The wedged span must COMPLETE: tree at op-2's position, exact
    caught up, no raw SourceDeltaError ever.

    At base this is RED with the production error verbatim: the span's
    batch carries the ledger ADD (``None -> addpos``) plus the fold MOVE
    (``op1_after -> op2_after``) for the same object; the executor's
    coalesce refuses; the error escapes untyped; the job dies without
    settling; the lane never catches up.
    """
    with _make_client(
        tmp_path,
        family_site,
        epoch_scheduler_factory=_isolated_scheduler_factory,
        epoch_tick_ms=200.0,
    ) as client:
        store = client.app.state.context.store
        workspace = _scenario_ready(client)
        # The subscription object stays referenced for the whole test —
        # the exact chase only mints while somebody watches.
        subscriptions = [SseSession(client, workspace)]  # noqa: F841

        _drive_mixed_plane_poison(client, workspace)
        version_before = _scene_version(client, workspace)

        # The poisoned span resolves one way or the other: either the
        # production refusal appears (base) or the lane catches up (fix).
        def _outcome() -> str | None:
            if _exact_chain_failures(store, workspace):
                return "poisoned"
            body = client.get(f"/api/v1/scenarios/{workspace}").json()
            if (
                body["exact_result_version"] == body["scene_version"]
                and int(body["scene_version"]) > version_before
            ):
                return "caught_up"
            return None

        outcome = poll_until(_outcome, timeout=150.0, interval=0.2)
        failures = _exact_chain_failures(store, workspace)
        assert not failures, (
            "the exact span was poisoned by the mixed-plane batch "
            f"(production incident 3): {failures}"
        )
        assert outcome == "caught_up", (
            "the exact lane neither poisoned nor caught up (timeout)"
        )
        assert _wait_exact_caught_up(client, workspace) is not None

        # No raw rethrow on ANY exact job: residual refusals must be the
        # typed edit_rejected envelope that settles (contract 5).
        for row in exact_jobs(store, workspace):
            error = row.get("error") or {}
            assert "SourceDeltaError" not in json.dumps(error), (
                row["job_id"],
                error,
                "a coalesce refusal escaped untyped",
            )

        # The tree's authoritative state: op-2's position, production
        # fields intact, exactly once (native ops after the legacy add
        # win, as ordered).
        poison = _canonical_object(store, workspace, POISON_ID)
        _assert_tree_state(
            poison,
            x_m=_world_of(client, P2_UV)[0],
            y_m=_world_of(client, P2_UV)[1],
            height_m=12.0,
            canopy_radius_m=3.5,
            trunk_ratio=0.25,
        )
        objects = store.canonical_state_at(
            workspace, store.require_scenario(workspace).scene_version
        ).families["vegetation_geometry"]["objects"]
        assert sum(1 for key in objects if key == POISON_ID) == 1

        # Zero-loss lane metrics keep their meaning: the funnel op was
        # counted as skipped exactly once (the epoch-A span), and the
        # final span folded its two native ops.
        completed = [
            row
            for row in exact_jobs(store, workspace)
            if row["status"] == "complete" and (row.get("metrics") or {}).get("exact_lane")
        ]
        assert completed, "no completed exact job carrying lane metrics"
        by_skipped = [
            int(row["metrics"]["exact_lane"].get("legacy_ops_skipped", 0))
            for row in completed
        ]
        assert 1 in by_skipped, (
            [row["metrics"]["exact_lane"] for row in completed],
            "the funnelled legacy add was never accounted as skipped",
        )


# ---------------------------------------------------------------------------
# T2: bitwise parity twins — all-legacy and all-native oracles
# ---------------------------------------------------------------------------


def _drive_all_native_twin(client: TestClient, workspace: str) -> None:
    _reset_to_baseline(client, workspace, key="t2-native-reset")
    _post_native(
        client,
        workspace,
        [
            _native_add_item(
                client, "t2n_healthy", HEALTHY_ID, HEALTHY_UV, HEALTHY_FIELDS,
                client_sequence=1,
            ),
            _native_add_item(
                client, "t2n_poison", POISON_ID, ADD_UV, POISON_FIELDS,
                client_sequence=2,
            ),
        ],
    )
    assert _wait_exact_caught_up(client, workspace) is not None
    _post_native(
        client,
        workspace,
        [_native_move_item(client, "t2n_move1", POISON_ID, P1_UV, client_sequence=3)],
    )
    assert _wait_exact_caught_up(client, workspace) is not None
    _post_native(
        client,
        workspace,
        [
            _native_move_item(
                client, "t2n_move2", POISON_ID, P2_UV, client_sequence=4
            ),
            _native_move_item(
                client, "t2n_healthy_move", HEALTHY_ID, HEALTHY2_UV,
                client_sequence=5,
            ),
        ],
    )
    assert _wait_exact_caught_up(client, workspace) is not None


def _drive_all_legacy_twin(client: TestClient, workspace: str) -> None:
    from tests.test_server_api import wait_for_job

    _reset_to_baseline(client, workspace, key="t2-legacy-reset")

    def _edit(edits: list[dict], key: str) -> None:
        body = _legacy_tree_edit(client, workspace, edits, key=key)
        job = wait_for_job(client, body["job_id"], timeout=180.0)
        assert job["status"] == "complete", job.get("error")

    _edit([{"operation": "add", "tree": _healthy_tree(HEALTHY_UV)}], "t2l-1")
    _edit([{"operation": "add", "tree": _poison_tree(ADD_UV)}], "t2l-2")
    _edit([{"operation": "move", "tree": _poison_tree(P1_UV)}], "t2l-3")
    _edit(
        [
            {"operation": "move", "tree": _poison_tree(P2_UV)},
            {"operation": "move", "tree": _healthy_tree(HEALTHY2_UV)},
        ],
        "t2l-4",
    )


@pytest.mark.scientific
def test_t2_parity_twins_all_legacy_and_all_native(
    tmp_path: Path, family_site
) -> None:
    """The mixed-plane scene must be bitwise the same scene as BOTH
    single-plane oracles: every tree added/edited only through /edits
    (all-legacy) and only through realtime operations (all-native).
    Same edit structure, same positions (one float derivation per
    position shared by all lanes), same final tree layer.
    """
    with _make_client(
        tmp_path,
        family_site,
        epoch_scheduler_factory=_isolated_scheduler_factory,
        epoch_tick_ms=200.0,
    ) as client:
        store = client.app.state.context.store

        mixed = _scenario_ready(client)
        native_twin = _scenario_ready(client)
        legacy_twin = _scenario_ready(client)
        subscriptions = [  # noqa: F841
            SseSession(client, mixed),
            SseSession(client, native_twin),
        ]

        _drive_mixed_plane_poison(client, mixed)
        assert _wait_exact_caught_up(client, mixed) is not None
        _drive_all_native_twin(client, native_twin)
        _drive_all_legacy_twin(client, legacy_twin)

        mixed_arrays = _arrays_at(client, mixed, _scene_version(client, mixed))
        native_arrays = _arrays_at(
            client, native_twin, _scene_version(client, native_twin)
        )
        legacy_arrays = _arrays_at(
            client, legacy_twin, _scene_version(client, legacy_twin)
        )
        _assert_nan_aware_bitwise_equal(mixed_arrays, native_arrays)
        _assert_nan_aware_bitwise_equal(mixed_arrays, legacy_arrays)

        # All three lanes leave the poisoned tree at op-2's position.
        p2 = _world_of(client, P2_UV)
        for workspace in (mixed, native_twin):
            poison = _canonical_object(store, workspace, POISON_ID)
            _assert_tree_state(
                poison,
                x_m=p2[0],
                y_m=p2[1],
                height_m=12.0,
                canopy_radius_m=3.5,
                trunk_ratio=0.25,
            )
        legacy_trees = {
            tree["tree_id"]: tree
            for tree in client.get(f"/api/v1/scenarios/{legacy_twin}").json()["trees"]
        }
        assert POISON_ID in legacy_trees
        assert legacy_trees[POISON_ID]["height_m"] == 12.0
        assert legacy_trees[POISON_ID]["canopy_diameter_m"] == 7.0
        # Non-vacuous: the legacy lane's tree ended at op-2's uv target.
        assert legacy_trees[POISON_ID]["u"] == P2_UV[0]
        assert legacy_trees[POISON_ID]["v"] == P2_UV[1]


# ---------------------------------------------------------------------------
# T3: the incident-2 regression gate — NOT an in-file test. The four tests
# of tests/test_live_incident2_exact_lane_poison.py (foreign-family staging,
# post-reset void seeding, both watermark pins) must stay green; the suite
# run is the gate. (Numbering kept: T3 names that gate.)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# T4: legacy add with NO native ops — folded/replayed exactly once
# ---------------------------------------------------------------------------


@pytest.mark.scientific
def test_t4_legacy_add_without_native_ops_replays_exactly_once(
    tmp_path: Path, family_site
) -> None:
    """A funnelled legacy add whose tree never moves natively: the add
    lands EXACTLY ONCE (tree present, at the ledger position, production
    fields) and later spans never double-apply it or move it.
    """
    with _make_client(
        tmp_path,
        family_site,
        epoch_scheduler_factory=_isolated_scheduler_factory,
        epoch_tick_ms=200.0,
    ) as client:
        store = client.app.state.context.store
        workspace = _scenario_ready(client)
        subscriptions = [SseSession(client, workspace)]  # noqa: F841

        _reset_to_baseline(client, workspace, key="t4-reset")
        _post_native(
            client,
            workspace,
            [
                _native_add_item(
                    client, "t4_healthy", HEALTHY_ID, HEALTHY_UV, HEALTHY_FIELDS,
                    client_sequence=1,
                )
            ],
        )
        assert _wait_exact_caught_up(client, workspace) is not None

        _legacy_tree_edit(
            client, workspace, [{"operation": "add", "tree": _poison_tree(ADD_UV)}],
            key="t4-legacy-add",
        )
        funnel_op = next(
            (
                record
                for record in store.operations_since(workspace, 0)
                if record.actor_id == LEGACY_ACTOR_ID
                and record.entity_id == POISON_ID
            ),
            None,
        )
        assert funnel_op is not None, "the legacy add never funnelled"
        _wait_epoch_settled(store, workspace, int(funnel_op.epoch_id))

        # Exactly one funnelled op for the tree, ever.
        funnelled = [
            record
            for record in store.operations_since(workspace, 0)
            if record.actor_id == LEGACY_ACTOR_ID and record.entity_id == POISON_ID
        ]
        assert len(funnelled) == 1, [r.operation_id for r in funnelled]
        assert not _exact_chain_failures(store, workspace)

        add_x, add_y = _world_of(client, ADD_UV)
        poison = _canonical_object(store, workspace, POISON_ID)
        _assert_tree_state(
            poison,
            x_m=add_x,
            y_m=add_y,
            height_m=12.0,
            canopy_radius_m=3.5,
            trunk_ratio=0.25,
        )

        # A LATER span (healthy tree moves, poisoned tree untouched):
        # idempotent — the add is not re-applied, the tree stays put,
        # exactly one object row, still no coalesce refusal.
        _post_native(
            client,
            workspace,
            [
                _native_move_item(
                    client, "t4_healthy_move", HEALTHY_ID, HEALTHY2_UV,
                    client_sequence=2,
                )
            ],
        )
        assert _wait_exact_caught_up(client, workspace) is not None
        assert not _exact_chain_failures(store, workspace)
        poison = _canonical_object(store, workspace, POISON_ID)
        _assert_tree_state(
            poison,
            x_m=add_x,
            y_m=add_y,
            height_m=12.0,
            canopy_radius_m=3.5,
            trunk_ratio=0.25,
        )
        completed = [
            row
            for row in exact_jobs(store, workspace)
            if row["status"] == "complete" and (row.get("metrics") or {}).get("exact_lane")
        ]
        assert completed
        assert 1 in [
            int(row["metrics"]["exact_lane"].get("legacy_ops_skipped", 0))
            for row in completed
        ], "the funnelled add was never accounted as a skipped legacy op"


# ---------------------------------------------------------------------------
# T5: the per-object chain anchor, unit-pinned at the exact production
# marker shape (consumed_revision 1-ish analog of 13/12/16/17)
# ---------------------------------------------------------------------------


def _insert_reset_event(store: Store, workspace: str, *, sequence: int) -> None:
    with store._write() as conn:
        conn.execute(
            "INSERT INTO edit_events (scenario_id, sequence, event_id, "
            "base_scene_version, operation, tree_id, submitted_at) "
            "VALUES (?, ?, ?, 0, 'reset', NULL, ?)",
            (workspace, sequence, f"evt-reset-{sequence}", "2026-09-07T18:00:00.000Z"),
        )


def _insert_legacy_add_event(
    store: Store, workspace: str, tree: dict[str, Any], *, sequence: int
) -> None:
    tree_json = json.dumps(tree)
    with store._write() as conn:
        conn.execute(
            "INSERT INTO edit_events (scenario_id, sequence, event_id, "
            "base_scene_version, operation, tree_id, new_tree_json, submitted_at) "
            "VALUES (?, ?, ?, 1, 'add', ?, ?, ?)",
            (
                workspace,
                sequence,
                f"evt_{sequence}",
                tree["tree_id"],
                tree_json,
                "2026-09-07T18:37:00.000Z",
            ),
        )
        conn.execute(
            "INSERT INTO scenario_trees (scenario_id, tree_id, tree_json) "
            "VALUES (?, ?, ?)",
            (workspace, tree["tree_id"], tree_json),
        )


def test_t5_fold_command_chains_onto_the_ledger_after_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CONTRACT 2, unit pin at the production marker shape.

    The fold's delta for an id whose ledger-window event the same job
    replays must chain onto the LEDGER's after-state: the fold derives
    the object's native ops since its LAST LEDGER EVENT (both op-1 and
    op-2 -> ``[addpos -> op2]``), producing a move command whose
    old_state IS the ledger add's five properties — not the consumed
    canonical state at op-1's position (the un-anchored base behavior
    that produced the non-contiguous refusal).
    """
    from solweig_gpu.server.executor_bridge import make_universal_dispatch_solver

    # The production grid mapping (tiny site geometry is irrelevant here;
    # the executor solve is stubbed — only the float derivation of the
    # event's uv must be the SAME one the anchor uses).
    grid = {
        "rows": 8,
        "cols": 8,
        "pixel_size_m": 2.0,
        "origin_x_m": 0.0,
        "origin_y_m": 0.0,
        "time_steps": 1,
    }

    def _uv(u: float, v: float) -> tuple[float, float]:
        return uv_to_world(
            u, v,
            rows=8, cols=8, pixel_size_m=2.0, origin_x_m=0.0, origin_y_m=0.0,
        )

    add_tree = make_tree(POISON_ID, u=0.50, v=0.50, **POISON_FIELDS)
    add_x, add_y = _uv(0.50, 0.50)
    p1_x, p1_y = _uv(0.3125, 0.375)
    p2_x, p2_y = _uv(0.6875, 0.4375)

    store = make_store(tmp_path)
    make_workspace(store, WS)
    runner = make_runner(tmp_path, store)

    # Ledger: reset seq 1, add seq 2 (production: last_reset_sequence 1,
    # the add is the only tree event in the replay window).
    _insert_reset_event(store, WS, sequence=1)
    _insert_legacy_add_event(store, WS, add_tree, sequence=2)

    # Op log: funnel add (seq 1) + native move op-1 (seq 2) in epoch 0;
    # native move op-2 (seq 3) in epoch 1 — production seq 11/12/13.
    store.append_operations(
        WS,
        [
            op_item(
                "legacy-evt_2",
                actor_id=LEGACY_ACTOR_ID,
                entity_id=POISON_ID,
                verb="add",
                payload={
                    "values": {
                        "x_m": add_x,
                        "y_m": add_y,
                        "height_m": 12.0,
                        "canopy_radius_m": 3.5,
                        "trunk_ratio": 0.25,
                    }
                },
            ),
            op_item(
                "op_move_1",
                entity_id=POISON_ID,
                verb="move",
                client_sequence=2,
                payload={"values": {"x_m": p1_x, "y_m": p1_y}},
            ),
        ],
    )
    # The scheduler's committed canonical rows: epoch 0 folded the funnel
    # add + op-1 (id at op-1's position, generation 0), epoch 1 carries
    # the same object untouched (its op-2 fold is the exact lane's to
    # derive — the canonical row at the CONSUMED revision is what the
    # span folds onto).
    store.mark_epoch_status(WS, 0, "closed")
    committed_1 = store.commit_epoch_reduction(
        WS,
        0,
        families={
            "vegetation_geometry": {
                "objects": {
                    POISON_ID: {
                        "x_m": p1_x,
                        "y_m": p1_y,
                        "height_m": 12.0,
                        "canopy_radius_m": 3.5,
                        "trunk_ratio": 0.25,
                        "generation": 0,
                    }
                },
                "tombstones": {},
            }
        },
    )
    assert committed_1 is not None and committed_1.workspace_revision == 1
    store.append_operations(
        WS,
        [
            op_item(
                "op_move_2",
                entity_id=POISON_ID,
                verb="move",
                client_sequence=3,
                payload={"values": {"x_m": p2_x, "y_m": p2_y}},
            )
        ],
    )
    store.mark_epoch_status(WS, 1, "closed")
    committed_2 = store.commit_epoch_reduction(
        WS,
        1,
        families={
            "vegetation_geometry": {
                "objects": {
                    POISON_ID: {
                        "x_m": p2_x,
                        "y_m": p2_y,
                        "height_m": 12.0,
                        "canopy_radius_m": 3.5,
                        "trunk_ratio": 0.25,
                        "generation": 0,
                    }
                },
                "tombstones": {},
            }
        },
    )
    assert committed_2 is not None and committed_2.workspace_revision == 2

    captured: dict[str, Any] = {}
    _capture_commands_stub(monkeypatch, captured)
    solve = make_universal_dispatch_solver(lambda ctx: None)(runner.context)

    # The production marker shape: the span (consumed_seq, span_hi] holds
    # ONLY op-2; the replay window (last_reset, watermark] holds the ADD.
    request = exact_request(
        target_scene_version=2,
        edit_watermark=2,
        events=tuple(store.list_events(WS)),
        trees=(add_tree,),
        grid=grid,
        exact_lane={
            "target_revision": 2,
            "consumed_revision": 1,
            "consumed_seq": 2,
            "span_hi": 3,
        },
    )
    result = solve(request, lambda *args, **kwargs: None)
    assert result.error is None, result.error

    commands = captured.get("commands", [])
    moves = [
        command
        for command in commands
        if command.adapter_id == "vegetation_geometry"
        and (command.old_state or {}).get("tree_id") == POISON_ID
    ]
    assert moves, (
        [(c.adapter_id, c.operation) for c in commands],
        "the poisoned tree's fold delta vanished from the batch",
    )

    # THE ANCHOR: the fold command's before-state is the LEDGER add's
    # after-state (all five properties, bitwise), and its after-state is
    # op-2's position — the coalesced [addpos -> op2] net change.
    expected_old = {
        "tree_id": POISON_ID,
        "x_m": add_x,
        "y_m": add_y,
        "height_m": 12.0,
        "canopy_radius_m": 3.5,
        "trunk_ratio": 0.25,
    }
    assert moves[0].old_state == expected_old, (
        moves[0].old_state,
        "the fold did not chain onto the ledger's after-state "
        "(per-object chain anchoring, contract 2)",
    )
    assert moves[0].new_state["x_m"] == p2_x
    assert moves[0].new_state["y_m"] == p2_y
    assert moves[0].operation in ("move", "replace")


def _insert_legacy_move_event(
    store: Store,
    workspace: str,
    old_tree: dict[str, Any],
    new_tree: dict[str, Any],
    *,
    sequence: int,
) -> None:
    old_json = json.dumps(old_tree)
    new_json = json.dumps(new_tree)
    with store._write() as conn:
        conn.execute(
            "INSERT INTO edit_events (scenario_id, sequence, event_id, "
            "base_scene_version, operation, tree_id, old_tree_json, "
            "new_tree_json, submitted_at) "
            "VALUES (?, ?, ?, 1, 'move', ?, ?, ?, ?)",
            (
                workspace,
                sequence,
                f"evt_{sequence}",
                new_tree["tree_id"],
                old_json,
                new_json,
                "2026-09-07T18:40:00.000Z",
            ),
        )
        conn.execute(
            "INSERT INTO scenario_trees (scenario_id, tree_id, tree_json) "
            "VALUES (?, ?, ?) "
            "ON CONFLICT (scenario_id, tree_id) DO UPDATE SET "
            "tree_json = excluded.tree_json",
            (workspace, new_tree["tree_id"], new_json),
        )


def _legacy_funnel_item(
    operation_id: str,
    entity_id: str,
    verb: str,
    values: dict[str, Any],
    *,
    client_sequence: int,
) -> dict[str, Any]:
    """A funnelled legacy op exactly as jobs._legacy_operation_item mints it."""
    return op_item(
        operation_id,
        actor_id=LEGACY_ACTOR_ID,
        entity_id=entity_id,
        verb=verb,
        client_sequence=client_sequence,
        payload={"values": values},
    )


def _unit_grid() -> dict[str, Any]:
    return {
        "rows": 8,
        "cols": 8,
        "pixel_size_m": 2.0,
        "origin_x_m": 0.0,
        "origin_y_m": 0.0,
        "time_steps": 1,
    }


def _unit_uv(u: float, v: float) -> tuple[float, float]:
    return uv_to_world(
        u, v, rows=8, cols=8, pixel_size_m=2.0, origin_x_m=0.0, origin_y_m=0.0
    )


def _fold_commands_for(captured: dict[str, Any], tree_id: str) -> list:
    commands = captured.get("commands", [])
    return [
        command
        for command in commands
        if command.adapter_id == "vegetation_geometry"
        and (command.old_state or {}).get("tree_id") == tree_id
    ]


# ---------------------------------------------------------------------------
# T5b: the backfill must carry post-anchor native FIELD writes — the
# backfill-mutant killer (review BLOCKER 1a: pre-span moves coalesce
# identically without a backfill; a field write does not).
# ---------------------------------------------------------------------------


def test_t5b_backfill_carries_post_anchor_native_field_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A native REPLACE between the funnelled add and the span must ride
    the anchored backfill: the fold's after-state carries the new height,
    not the ledger add's (a backfill emptied to ``[]`` silently reverts
    the field write — exactly the mutant this test exists to kill)."""
    from solweig_gpu.server.executor_bridge import make_universal_dispatch_solver

    add_x, add_y = _unit_uv(0.50, 0.50)
    p2_x, p2_y = _unit_uv(0.6875, 0.4375)
    add_tree = make_tree(POISON_ID, u=0.50, v=0.50, **POISON_FIELDS)

    store = make_store(tmp_path)
    make_workspace(store, WS)
    runner = make_runner(tmp_path, store)

    _insert_reset_event(store, WS, sequence=1)
    _insert_legacy_add_event(store, WS, add_tree, sequence=2)

    # Epoch 0: funnel add (seq 1) + a committed native HEIGHT write
    # (seq 2) — the op the backfill must carry. Epoch 1: the span's move.
    store.append_operations(
        WS,
        [
            _legacy_funnel_item(
                "legacy-evt_2", POISON_ID, "add",
                {
                    "x_m": add_x, "y_m": add_y, "height_m": 12.0,
                    "canopy_radius_m": 3.5, "trunk_ratio": 0.25,
                },
                client_sequence=1,
            ),
            op_item(
                "op_height_write",
                entity_id=POISON_ID,
                verb="replace",
                client_sequence=2,
                payload={"values": {"height_m": 15.0}},
            ),
        ],
    )
    store.mark_epoch_status(WS, 0, "closed")
    committed_1 = store.commit_epoch_reduction(
        WS,
        0,
        families={
            "vegetation_geometry": {
                "objects": {
                    POISON_ID: {
                        "x_m": add_x, "y_m": add_y, "height_m": 15.0,
                        "canopy_radius_m": 3.5, "trunk_ratio": 0.25,
                        "generation": 0,
                    }
                },
                "tombstones": {},
            }
        },
    )
    assert committed_1 is not None and committed_1.workspace_revision == 1
    store.append_operations(
        WS,
        [
            op_item(
                "op_move_span",
                entity_id=POISON_ID,
                verb="move",
                client_sequence=3,
                payload={"values": {"x_m": p2_x, "y_m": p2_y}},
            )
        ],
    )
    store.mark_epoch_status(WS, 1, "closed")
    committed_2 = store.commit_epoch_reduction(
        WS,
        1,
        families={
            "vegetation_geometry": {
                "objects": {
                    POISON_ID: {
                        "x_m": p2_x, "y_m": p2_y, "height_m": 15.0,
                        "canopy_radius_m": 3.5, "trunk_ratio": 0.25,
                        "generation": 0,
                    }
                },
                "tombstones": {},
            }
        },
    )
    assert committed_2 is not None and committed_2.workspace_revision == 2

    captured: dict[str, Any] = {}
    _capture_commands_stub(monkeypatch, captured)
    solve = make_universal_dispatch_solver(lambda ctx: None)(runner.context)

    request = exact_request(
        target_scene_version=2,
        edit_watermark=2,
        events=tuple(store.list_events(WS)),
        trees=(add_tree,),
        grid=_unit_grid(),
        exact_lane={
            "target_revision": 2,
            "consumed_revision": 1,
            "consumed_seq": 2,
            "span_hi": 3,
        },
    )
    result = solve(request, lambda *args, **kwargs: None)
    assert result.error is None, result.error

    moves = _fold_commands_for(captured, POISON_ID)
    assert moves, (
        "the poisoned tree's fold delta vanished from the batch"
    )
    # The fold chains onto the ledger add's state (height 12.0 — the
    # client's own capture) and its AFTER-state carries the native height
    # write (15.0): the backfill folded it. An emptied backfill leaves
    # 12.0 here — the silent field revert this test kills.
    assert moves[0].old_state is not None
    assert moves[0].old_state["height_m"] == 12.0, moves[0].old_state
    assert moves[0].new_state is not None
    assert moves[0].new_state["height_m"] == 15.0, (
        moves[0].new_state,
        "the post-anchor native height write was silently dropped from "
        "the fold (the backfill mutant this test kills)",
    )
    assert moves[0].new_state["x_m"] == p2_x
    assert moves[0].new_state["y_m"] == p2_y


# ---------------------------------------------------------------------------
# T7 (review BLOCKER 2): a MOVE anchor must recover the id's pre-anchor
# native FIELD writes — the funnelled move applies position-only in the
# reducer, so the client's full-tree capture is blind to them.
# ---------------------------------------------------------------------------


def test_t7_move_anchor_recovers_pre_anchor_native_field_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The silent divergence the review traced (BLOCKER 2).

    A legacy MOVE event's ``new_tree`` carries all five client fields
    (captured from the LEGACY plane, which native realtime writes never
    touch), but the funnelled move applies POSITION ONLY in the reducer
    (reducer ``_POSITION_FIELDS``). A committed native height write that
    predates the funnel therefore survives in canonical — and must
    survive in the published fold too: the fold's after-state carries the
    native height, the move event's position.

    At the pre-rework HEAD this test is RED: the backfill floors at the
    anchor's funnel seq, the pre-anchor write is excluded, and the fold
    silently reverts the height to the client's stale capture.
    """
    from solweig_gpu.server.executor_bridge import make_universal_dispatch_solver

    add_x, add_y = _unit_uv(0.50, 0.50)
    move_x, move_y = _unit_uv(0.375, 0.5625)
    span_x, span_y = _unit_uv(0.6875, 0.4375)

    add_tree = make_tree(POISON_ID, u=0.50, v=0.50, **POISON_FIELDS)
    # The client's move capture: the legacy-plane row — blind to the
    # native height write (still height 12.0), position moved.
    move_tree = make_tree(POISON_ID, u=0.375, v=0.5625, **POISON_FIELDS)

    store = make_store(tmp_path)
    make_workspace(store, WS)
    runner = make_runner(tmp_path, store)

    _insert_reset_event(store, WS, sequence=1)
    _insert_legacy_add_event(store, WS, add_tree, sequence=2)
    _insert_legacy_move_event(store, WS, add_tree, move_tree, sequence=3)

    # Epoch 0: funnel add (seq 1) + the pre-anchor native height write
    # (seq 2). Canonical rev 1: (addpos, height 15.0).
    store.append_operations(
        WS,
        [
            _legacy_funnel_item(
                "legacy-evt_2", POISON_ID, "add",
                {
                    "x_m": add_x, "y_m": add_y, "height_m": 12.0,
                    "canopy_radius_m": 3.5, "trunk_ratio": 0.25,
                },
                client_sequence=1,
            ),
            op_item(
                "op_height_write",
                entity_id=POISON_ID,
                verb="replace",
                client_sequence=2,
                payload={"values": {"height_m": 15.0}},
            ),
        ],
    )
    store.mark_epoch_status(WS, 0, "closed")
    committed_1 = store.commit_epoch_reduction(
        WS,
        0,
        families={
            "vegetation_geometry": {
                "objects": {
                    POISON_ID: {
                        "x_m": add_x, "y_m": add_y, "height_m": 15.0,
                        "canopy_radius_m": 3.5, "trunk_ratio": 0.25,
                        "generation": 0,
                    }
                },
                "tombstones": {},
            }
        },
    )
    assert committed_1 is not None and committed_1.workspace_revision == 1

    # Epoch 1: the funnelled MOVE (seq 3). Canonical rev 2: the move
    # applies position only — (movepos, height 15.0). The event's own
    # trees stay legacy-plane (height 12.0).
    store.append_operations(
        WS,
        [
            _legacy_funnel_item(
                "legacy-evt_3", POISON_ID, "move",
                {"x_m": move_x, "y_m": move_y},
                client_sequence=3,
            ),
        ],
    )
    store.mark_epoch_status(WS, 1, "closed")
    committed_2 = store.commit_epoch_reduction(
        WS,
        1,
        families={
            "vegetation_geometry": {
                "objects": {
                    POISON_ID: {
                        "x_m": move_x, "y_m": move_y, "height_m": 15.0,
                        "canopy_radius_m": 3.5, "trunk_ratio": 0.25,
                        "generation": 0,
                    }
                },
                "tombstones": {},
            }
        },
    )
    assert committed_2 is not None and committed_2.workspace_revision == 2

    # Epoch 2: the span's native move (seq 4) — target revision 3.
    store.append_operations(
        WS,
        [
            op_item(
                "op_span_move",
                entity_id=POISON_ID,
                verb="move",
                client_sequence=4,
                payload={"values": {"x_m": span_x, "y_m": span_y}},
            )
        ],
    )
    store.mark_epoch_status(WS, 2, "closed")
    committed_3 = store.commit_epoch_reduction(
        WS,
        2,
        families={
            "vegetation_geometry": {
                "objects": {
                    POISON_ID: {
                        "x_m": span_x, "y_m": span_y, "height_m": 15.0,
                        "canopy_radius_m": 3.5, "trunk_ratio": 0.25,
                        "generation": 0,
                    }
                },
                "tombstones": {},
            }
        },
    )
    assert committed_3 is not None and committed_3.workspace_revision == 3

    captured: dict[str, Any] = {}
    _capture_commands_stub(monkeypatch, captured)
    solve = make_universal_dispatch_solver(lambda ctx: None)(runner.context)

    request = exact_request(
        target_scene_version=3,
        edit_watermark=3,
        events=tuple(store.list_events(WS)),
        trees=(move_tree,),
        grid=_unit_grid(),
        exact_lane={
            "target_revision": 3,
            "consumed_revision": 2,
            "consumed_seq": 3,
            "span_hi": 4,
        },
    )
    result = solve(request, lambda *args, **kwargs: None)
    assert result.error is None, result.error

    moves = _fold_commands_for(captured, POISON_ID)
    assert moves, "the poisoned tree's fold delta vanished from the batch"

    # The fold chains onto the MOVE event's after-state (the client's
    # five fields — position moved, height 12.0) and its after-state is
    # the canonical truth: the native height write recovered, the span's
    # position applied. Faithfulness: canonical rev 3 for this object is
    # exactly (spanpos, height 15.0).
    expected_old = {
        "tree_id": POISON_ID,
        "x_m": move_x,
        "y_m": move_y,
        "height_m": 12.0,
        "canopy_radius_m": 3.5,
        "trunk_ratio": 0.25,
    }
    assert moves[0].old_state == expected_old, (
        moves[0].old_state,
        "the fold did not chain onto the move event's after-state",
    )
    assert moves[0].new_state is not None
    assert moves[0].new_state["x_m"] == span_x, moves[0].new_state
    assert moves[0].new_state["y_m"] == span_y, moves[0].new_state
    assert moves[0].new_state["height_m"] == 15.0, (
        moves[0].new_state,
        "the pre-anchor native height write was silently reverted to the "
        "client's stale capture (review BLOCKER 2: the funnelled move is "
        "position-only in canonical, so the native write must survive)",
    )


# ---------------------------------------------------------------------------
# T8 (review BLOCKER 2 caveat): a terminal pre-anchor native DELETE is a
# cross-plane ordering the durable state cannot resolve — the fold must
# TAKE THE TYPED REFUSAL, never silently resurrect the tree.
# ---------------------------------------------------------------------------


def test_t8_pre_anchor_native_delete_takes_typed_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A legacy-plane MOVE of a tree the native plane already deleted:
    canonical keeps it dead (the funnelled move over a tombstone is an
    audited no-op), the event's trees never saw the death. The anchor
    cannot converge this — folding would resurrect the tree silently —
    so the span must refuse with the TYPED edit_rejected envelope naming
    the id (which settles), never publish a resurrection."""
    from solweig_gpu.server.executor_bridge import make_universal_dispatch_solver

    add_x, add_y = _unit_uv(0.50, 0.50)
    move_x, move_y = _unit_uv(0.375, 0.5625)
    healthy_x, healthy_y = _unit_uv(0.3125, 0.4218)
    healthy2_x, healthy2_y = _unit_uv(0.3611, 0.4722)

    add_tree = make_tree(POISON_ID, u=0.50, v=0.50, **POISON_FIELDS)
    move_tree = make_tree(POISON_ID, u=0.375, v=0.5625, **POISON_FIELDS)
    healthy_tree = make_tree(HEALTHY_ID, u=0.3125, v=0.4218, **HEALTHY_FIELDS)

    store = make_store(tmp_path)
    make_workspace(store, WS)
    runner = make_runner(tmp_path, store)

    _insert_reset_event(store, WS, sequence=1)
    _insert_legacy_add_event(store, WS, add_tree, sequence=2)
    _insert_legacy_move_event(store, WS, add_tree, move_tree, sequence=3)

    # Epoch 0: funnel add (seq 1), the native DELETE of the poisoned tree
    # (seq 2), a native healthy add (seq 3, keeps the span meaningful).
    # Canonical rev 1: poison tombstoned.
    store.append_operations(
        WS,
        [
            _legacy_funnel_item(
                "legacy-evt_2", POISON_ID, "add",
                {
                    "x_m": add_x, "y_m": add_y, "height_m": 12.0,
                    "canopy_radius_m": 3.5, "trunk_ratio": 0.25,
                },
                client_sequence=1,
            ),
            op_item(
                "op_native_delete",
                entity_id=POISON_ID,
                verb="delete",
                client_sequence=2,
                payload={"values": {}},
            ),
            op_item(
                "op_healthy_add",
                entity_id=HEALTHY_ID,
                verb="add",
                client_sequence=3,
                payload={
                    "values": {
                        "x_m": healthy_x, "y_m": healthy_y,
                        "height_m": 13.0, "canopy_radius_m": 4.0,
                        "trunk_ratio": 0.25,
                    }
                },
            ),
        ],
    )
    store.mark_epoch_status(WS, 0, "closed")
    committed_1 = store.commit_epoch_reduction(
        WS,
        0,
        families={
            "vegetation_geometry": {
                "objects": {
                    HEALTHY_ID: {
                        "x_m": healthy_x, "y_m": healthy_y,
                        "height_m": 13.0, "canopy_radius_m": 4.0,
                        "trunk_ratio": 0.25, "generation": 0,
                    }
                },
                "tombstones": {POISON_ID: 0},
            }
        },
    )
    assert committed_1 is not None and committed_1.workspace_revision == 1

    # Epoch 1: the funnelled MOVE of the natively-deleted tree (seq 4) —
    # over a tombstone it is an audited no-op in canonical (rev 2 keeps
    # the tombstone; the legacy plane never saw the death).
    store.append_operations(
        WS,
        [
            _legacy_funnel_item(
                "legacy-evt_3", POISON_ID, "move",
                {"x_m": move_x, "y_m": move_y},
                client_sequence=4,
            ),
        ],
    )
    store.mark_epoch_status(WS, 1, "closed")
    committed_2 = store.commit_epoch_reduction(
        WS,
        1,
        families={
            "vegetation_geometry": {
                "objects": {
                    HEALTHY_ID: {
                        "x_m": healthy_x, "y_m": healthy_y,
                        "height_m": 13.0, "canopy_radius_m": 4.0,
                        "trunk_ratio": 0.25, "generation": 0,
                    }
                },
                "tombstones": {POISON_ID: 0},
            }
        },
    )
    assert committed_2 is not None and committed_2.workspace_revision == 2

    # Epoch 2: the span (seq 5) — a healthy-tree move, target revision 3.
    store.append_operations(
        WS,
        [
            op_item(
                "op_healthy_span_move",
                entity_id=HEALTHY_ID,
                verb="move",
                client_sequence=5,
                payload={"values": {"x_m": healthy2_x, "y_m": healthy2_y}},
            )
        ],
    )
    store.mark_epoch_status(WS, 2, "closed")
    committed_3 = store.commit_epoch_reduction(
        WS,
        2,
        families={
            "vegetation_geometry": {
                "objects": {
                    HEALTHY_ID: {
                        "x_m": healthy2_x, "y_m": healthy2_y,
                        "height_m": 13.0, "canopy_radius_m": 4.0,
                        "trunk_ratio": 0.25, "generation": 0,
                    }
                },
                "tombstones": {POISON_ID: 0},
            }
        },
    )
    assert committed_3 is not None and committed_3.workspace_revision == 3

    _capture_commands_stub(monkeypatch, {})
    solve = make_universal_dispatch_solver(lambda ctx: None)(runner.context)

    request = exact_request(
        target_scene_version=3,
        edit_watermark=3,
        events=tuple(store.list_events(WS)),
        trees=(healthy_tree,),
        grid=_unit_grid(),
        exact_lane={
            "target_revision": 3,
            "consumed_revision": 2,
            "consumed_seq": 4,
            "span_hi": 5,
        },
    )
    result = solve(request, lambda *args, **kwargs: None)

    # THE REFUSAL: typed edit_rejected, naming the id and the divergence
    # class. At the pre-rework HEAD this is either a silent success or an
    # unrelated coalesce refusal — never this named, settling envelope.
    assert result.error is not None, (
        "the fold silently converged (or published a resurrection of a "
        "natively-deleted tree) — BLOCKER 2's caveat case must refuse typed"
    )
    assert result.error.get("code") == "edit_rejected", result.error
    message = str(result.error.get("message", ""))
    assert POISON_ID in message, result.error
    assert "native delete precedes" in message, (
        result.error,
        "the refusal must name the divergence class (a pre-anchor native "
        "delete under a ledger window event) so operators can tell it "
        "from a coalesce refusal",
    )


# ---------------------------------------------------------------------------
# T6 (review BLOCKER 1b): the post-typed-settle wedge lineage — baseline
# CONTAINS the id while the window carries its ADD. Kills the
# seed-exclusion mutant (pre-planting the id collides with the replay's
# own add: the layer's add_tree raises, the job dies raw, the lane wedges).
# ---------------------------------------------------------------------------


GHOST_ID = "tree-ghost-wedge-01"


def test_t6_wedge_lineage_window_add_over_baseline_id_completes(
    tmp_path: Path, family_site
) -> None:
    """The production genesis, forced deterministically.

    The funnel epoch also carries a native op on an id NO fold baseline
    knows (a ghost move) — the span takes the B1b typed refusal, which
    SETTLES the epoch (canonical advances: the baseline now contains the
    poisoned tree at op-1's position) while coverage never stages (the
    window still carries the ADD). The NEXT span is the exclusion
    trigger: seeding the baseline's copy of the id would pre-plant it
    under the replay's own ADD. With the exclusion the span completes,
    the tree lands at op-2; with the exclusion emptied the replay add
    collides with the seeded layer row (a raw ``already exists``), the
    job dies without settling, and the lane wedges — this test's RED.
    """
    with _make_client(
        tmp_path,
        family_site,
        epoch_scheduler_factory=_isolated_scheduler_factory,
        epoch_tick_ms=200.0,
    ) as client:
        store = client.app.state.context.store
        workspace = _scenario_ready(client)
        subscriptions = [SseSession(client, workspace)]  # noqa: F841

        _reset_to_baseline(client, workspace, key="inc3b-reset")
        _post_native(
            client,
            workspace,
            [
                _native_add_item(
                    client, "t6_healthy", HEALTHY_ID, HEALTHY_UV,
                    HEALTHY_FIELDS, client_sequence=1,
                )
            ],
        )
        assert _wait_exact_caught_up(client, workspace) is not None

        # The legacy ADD, then — in ONE operations POST — op-1 AND the
        # ghost move, under a HELD scheduler: all three share one epoch
        # deterministically (never a cadence race).
        with _hold_epochs(client):
            _legacy_tree_edit(
                client,
                workspace,
                [{"operation": "add", "tree": _poison_tree(ADD_UV)}],
                key="studio-inc3b-edit-1",
            )
            _post_native(
                client,
                workspace,
                [
                    _native_move_item(
                        client, "t6_move_op1", POISON_ID, P1_UV, client_sequence=2
                    ),
                    _native_move_item(
                        client, "t6_ghost_move", GHOST_ID, P2_UV, client_sequence=3
                    ),
                ],
            )
        funnel_op = next(
            (
                record
                for record in store.operations_since(workspace, 0)
                if record.actor_id == LEGACY_ACTOR_ID
                and record.entity_id == POISON_ID
            ),
            None,
        )
        op1 = _op_record(store, workspace, "t6_move_op1")
        ghost = _op_record(store, workspace, "t6_ghost_move")
        assert funnel_op is not None and op1 is not None and ghost is not None
        assert (
            int(funnel_op.epoch_id) == int(op1.epoch_id) == int(ghost.epoch_id)
        ), (
            "repro gate missed: the funnel add, op-1 and the ghost move "
            f"must share one epoch ({funnel_op.epoch_id} / {op1.epoch_id} / "
            f"{ghost.epoch_id}) — the typed-settle lineage was not "
            "reproduced; rerun"
        )

        # Epoch A settles THROUGH the typed refusal (B1b: the ghost move
        # is unfoldable for every fold baseline) — canonical advances,
        # coverage does not.
        _wait_epoch_settled(store, workspace, int(funnel_op.epoch_id))
        lineage = [
            row
            for row in exact_jobs(store, workspace)
            if (row.get("error") or {}).get("code") == "edit_rejected"
            and GHOST_ID in json.dumps(row.get("error"))
        ]
        assert lineage, (
            "vacuity guard: no typed settle named the ghost — the "
            "baseline-advanced/coverage-stuck lineage never happened"
        )

        # Epoch B: the second native move — the wedge-lineage span.
        _post_native(
            client,
            workspace,
            [
                _native_move_item(
                    client, "t6_move_op2", POISON_ID, P2_UV, client_sequence=4
                )
            ],
        )

        assert _wait_exact_caught_up(client, workspace) is not None, (
            "the wedge-lineage span never completed: the baseline's copy "
            "of the id collided with the window's own ADD replay (the "
            "seed-exclusion mutant this test kills)"
        )
        assert not _exact_chain_failures(store, workspace)

        p2 = _world_of(client, P2_UV)
        poison = _canonical_object(store, workspace, POISON_ID)
        _assert_tree_state(
            poison,
            x_m=p2[0], y_m=p2[1],
            height_m=12.0, canopy_radius_m=3.5, trunk_ratio=0.25,
        )
        objects = store.canonical_state_at(
            workspace, store.require_scenario(workspace).scene_version
        ).families["vegetation_geometry"]["objects"]
        assert sum(1 for key in objects if key == POISON_ID) == 1


# ---------------------------------------------------------------------------
# T9 (review BLOCKER 3): an engine_refused exact job must SETTLE its span
# — the generic superseded fall-through leaves the epochs ``reducing``
# forever behind the churn guard (the wedge the code's own comment names).
# ---------------------------------------------------------------------------


def test_t9_engine_refused_exact_job_settles_its_span(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    make_workspace(store, WS)
    runner = make_runner(tmp_path, store)
    store.append_operations(WS, [op_item("op_t9_1")])
    close_epoch(store, epoch_id=0)

    job_id = _running_exact_job(store, 1)
    runner._finalize(
        store.require_job(job_id),
        store.require_scenario(WS),
        SolveResult(
            status="superseded",
            scene_version=1,
            error={
                "code": "engine_refused",
                "message": "engine machinery refused the batch",
            },
        ),
        duration_ms=1.0,
    )
    row = store.require_job(job_id)
    assert row.status == "failed", (
        row.status,
        "an engine_refused exact job must finish as a loud FAILED job "
        "(the deterministic-refusal settle), not a silent superseded",
    )
    assert epoch_of(store, WS, 0).status == "exact_targeted", (
        "the engine_refused refusal left its span's epoch 'reducing' — "
        "the churn guard wedges the lane at this revision forever "
        "(review BLOCKER 3)"
    )
    assert {e.workspace_id for e in store.recoverable_epochs()} == set()
