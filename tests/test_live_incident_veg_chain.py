# SPDX-License-Identifier: GPL-3.0-only
"""LIVE-INCIDENT regression (2026-09-07, scn_shared_world): the exact
lane refused every chase at/after a native ``replace`` of a
realtime-added tree with ``engine_refused: replay chain for tree ... is
not contiguous with the layer state``, wedging the exact result at the
pre-replace revision behind the churn guard.

Root cause (fixed in ``executor_bridge._build_executor``): on a
realtime-native workspace the legacy ledger watermark never advances, so
every exact job took the fresh-rebuild branch and seeded the executor's
layer from the LEGACY tree list only — the layer never carried trees
added by earlier realtime spans. Add-only spans publish blind; the first
replace/move/delete of an earlier span's tree trips the replay
contiguity guard. The fix seeds the layer from the exact lane's
canonical fold baseline at ``consumed_revision``, which is the span
delta's own before-state.
"""

from __future__ import annotations

from pathlib import Path

from tests.test_realtime_r2_exact_lane import (
    job_rows,
    poll_until,
    sse_subscribed,
)
from tests.test_server_universal_edits import (
    SITE_ID as FAMILY_SITE,
)
from tests.test_server_universal_edits import (
    _make_client,
    _scenario_ready,
    family_site,  # noqa: F401  (module-scoped real-site fixture)
)

import pytest

REPLACED_HEIGHT = 34.9826299398625  # the production height-slider value


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


def _drive(client, workspace: str, ops: list[dict]) -> None:
    """Post ``ops`` one per epoch, waiting for the exact lane each time.

    ``ops`` items: ``(actor, client_sequence, base_revision, entity,
    verb, payload, expected_revision_after_catchup)``. The catch-up wait
    is keyed on the EXPECTED revision, so an engine refusal (exact stuck)
    fails the step that wedged instead of silently passing on the
    previous revision.
    """
    for actor, seq, base, entity, verb, payload, expected in ops:
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

        def caught_up(expected: int = expected) -> bool:
            body = client.get(
                f"/api/v1/workspaces/{workspace}/operations"
            ).json()
            return (
                int(body["workspace_revision"]) >= expected
                and int(body["exact_revision"]) == int(body["workspace_revision"])
            )

        assert poll_until(caught_up, timeout=120.0), (
            f"exact lane never caught up after {verb} {entity} "
            "(LIVE INCIDENT: engine_refused refusal wedges the exact lane "
            "at the previous revision)"
        )


def _exact_job_errors(client, workspace: str) -> list[dict]:
    store = client.app.state.context.store
    return [
        row["error"]
        for row in job_rows(store, workspace)
        if row["request"].get("exact_lane") and row.get("error")
    ]


def _victim(client, workspace: str) -> dict | None:
    store = client.app.state.context.store
    body = client.get(f"/api/v1/workspaces/{workspace}/operations").json()
    canonical = store.canonical_state_at(workspace, int(body["workspace_revision"]))
    if canonical is None:
        return None
    family = (canonical.families or {}).get("vegetation_geometry") or {}
    return (family.get("objects") or {}).get("tree-mtr78qbn-05-5187")


def test_exact_lane_survives_native_replace_then_partial_move(
    tmp_path: Path, family_site
) -> None:
    """The full production sequence (live_veg_ops seq 4-8 on the three
    earlier test adds): every op its own epoch, exact lane catching up
    after each."""
    with _make_incident_client(tmp_path, family_site) as client:
        workspace = _scenario_ready(client)
        with sse_subscribed(client, workspace):
            geometry = dict(client.app.state.context.sites.geometry(FAMILY_SITE))
            px = float(geometry["pixel_size_m"])
            cx = float(geometry["origin_x_m"]) + 0.5 * int(geometry["cols"]) * px
            cy = float(geometry["origin_y_m"]) - 0.5 * int(geometry["rows"]) * px
            x5, y5 = cx + 4 * px, cy - 2 * px

            _drive(
                client,
                workspace,
                [
                    # seq 1-3: the earlier co-located test adds + gate probe
                    ("studio-a", 1, 0, "tree-exact-1", "add",
                     {"tree_id": "tree-exact-1", "x_m": cx, "y_m": cy,
                      "height_m": 12.0, "canopy_radius_m": 5.5,
                      "trunk_ratio": 0.25}, 1),
                    ("studio-b", 1, 1, "tree-exact-2", "add",
                     {"tree_id": "tree-exact-2", "x_m": cx, "y_m": cy,
                      "height_m": 12.0, "canopy_radius_m": 5.5,
                      "trunk_ratio": 0.25}, 2),
                    ("studio-c", 1, 2, "tree-gate-probe-1", "add",
                     {"tree_id": "tree-gate-probe-1", "x_m": cx, "y_m": cy,
                      "height_m": 14.0, "canopy_radius_m": 6.0,
                      "trunk_ratio": 0.25}, 3),
                    # seq 4: add tree-mtr3ucjx-04-eieb
                    ("studio-l5b8h03k", 1, 3, "tree-mtr3ucjx-04-eieb", "add",
                     {"tree_id": "tree-mtr3ucjx-04-eieb",
                      "x_m": cx - 8 * px, "y_m": cy + 3 * px,
                      "height_m": 12.0, "canopy_radius_m": 3.5,
                      "trunk_ratio": 0.25}, 4),
                    # seq 5: add tree-mtr78qbn-05-5187 (h 18)
                    ("studio-wxiilpu1", 1, 4, "tree-mtr78qbn-05-5187", "add",
                     {"tree_id": "tree-mtr78qbn-05-5187", "x_m": x5, "y_m": y5,
                      "height_m": 18.0, "canopy_radius_m": 5.5,
                      "trunk_ratio": 0.25}, 5),
                    # seq 6: REPLACE — the height slider (full payload, same x/y)
                    ("studio-wxiilpu1", 2, 5, "tree-mtr78qbn-05-5187", "replace",
                     {"tree_id": "tree-mtr78qbn-05-5187", "x_m": x5, "y_m": y5,
                      "height_m": REPLACED_HEIGHT, "canopy_radius_m": 5.5,
                      "trunk_ratio": 0.25}, 6),
                    # seq 7: MOVE — the canvas drag (PARTIAL payload: x/y only)
                    ("studio-zzxu5djb", 1, 6, "tree-mtr78qbn-05-5187", "move",
                     {"tree_id": "tree-mtr78qbn-05-5187",
                      "x_m": x5 + 2 * px, "y_m": y5 - 18 * px}, 7),
                    # seq 8: add tree-mtr8vqm0-06-jmte
                    ("studio-9s5aek8u", 1, 7, "tree-mtr8vqm0-06-jmte", "add",
                     {"tree_id": "tree-mtr8vqm0-06-jmte",
                      "x_m": cx - 20 * px, "y_m": cy + 10 * px,
                      "height_m": 12.0, "canopy_radius_m": 3.5,
                      "trunk_ratio": 0.25}, 8),
                ],
            )

            errors = _exact_job_errors(client, workspace)
            assert errors == [], errors
            victim = _victim(client, workspace)
            assert victim is not None
            assert victim["height_m"] == pytest.approx(REPLACED_HEIGHT), victim
            assert victim["x_m"] == pytest.approx(x5 + 2 * px), victim
            assert victim["y_m"] == pytest.approx(y5 - 18 * px), victim


def test_exact_lane_survives_replace_of_prior_span_tree_minimal_prefix(
    tmp_path: Path, family_site
) -> None:
    """Minimal prefix of the incident: one add (its own epoch, minted),
    then a native replace of that tree. At base the replace is refused
    with the exact production message and the exact result stays at the
    add's revision."""
    with _make_incident_client(tmp_path, family_site) as client:
        workspace = _scenario_ready(client)
        with sse_subscribed(client, workspace):
            geometry = dict(client.app.state.context.sites.geometry(FAMILY_SITE))
            px = float(geometry["pixel_size_m"])
            cx = float(geometry["origin_x_m"]) + 0.5 * int(geometry["cols"]) * px
            cy = float(geometry["origin_y_m"]) - 0.5 * int(geometry["rows"]) * px

            _drive(
                client,
                workspace,
                [
                    ("studio-wxiilpu1", 1, 0, "tree-mtr78qbn-05-5187", "add",
                     {"tree_id": "tree-mtr78qbn-05-5187", "x_m": cx, "y_m": cy,
                      "height_m": 18.0, "canopy_radius_m": 5.5,
                      "trunk_ratio": 0.25}, 1),
                    ("studio-wxiilpu1", 2, 1, "tree-mtr78qbn-05-5187", "replace",
                     {"tree_id": "tree-mtr78qbn-05-5187", "x_m": cx, "y_m": cy,
                      "height_m": REPLACED_HEIGHT, "canopy_radius_m": 5.5,
                      "trunk_ratio": 0.25}, 2),
                ],
            )

            errors = _exact_job_errors(client, workspace)
            assert errors == [], errors
            victim = _victim(client, workspace)
            assert victim is not None
            assert victim["height_m"] == pytest.approx(REPLACED_HEIGHT), victim


def test_baseline_tree_specs_trunk_ratio_fallback_is_pinned():
    """The seed helper's ``trunk_ratio`` fallback (an add predating the field
    folds without one) must equal the layer default everywhere else — the
    0.25 literal is duplicated across the codebase, and a silent drift here
    would seed executors with different trunk geometry than every other
    default site. Malformed specs must refuse typed, never seed a wrong
    tree (review mutation-b was green before this pin)."""
    from solweig_gpu.incremental.edit_types import EditStateError
    from solweig_gpu.server.executor_bridge import _exact_lane_baseline_tree_specs
    from solweig_gpu.server.realtime.types import CanonicalState

    complete = {
        "x_m": 10.0, "y_m": 20.0, "height_m": 12.0, "canopy_radius_m": 3.5
    }
    baseline = CanonicalState(
        workspace_id="w",
        workspace_revision=3,
        families={
            "vegetation_geometry": {
                "objects": {
                    "tree-a": dict(complete),  # no trunk_ratio — the fallback path
                    "tree-b": dict(complete, trunk_ratio=0.4),
                }
            }
        },
    )
    specs = _exact_lane_baseline_tree_specs(baseline)
    assert [(s.tree_id, s.trunk_ratio) for s in specs] == [
        ("tree-a", 0.25),
        ("tree-b", 0.4),
    ]

    torn = CanonicalState(
        workspace_id="w",
        workspace_revision=3,
        families={"vegetation_geometry": {"objects": {"tree-c": {"x_m": 1.0}}}},
    )
    with pytest.raises(EditStateError, match="tree-c"):
        _exact_lane_baseline_tree_specs(torn)
