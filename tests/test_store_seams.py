# SPDX-License-Identifier: GPL-3.0-only
"""Store-owner seam packet (r2a endgame / r2a-fix M1+F2 / r2b F6).

Four failing-first seams, one per review adjudication:

* ``UnsupportedFamily`` typed refusal at :meth:`Store.append_operations`
  (r2a-review segment-skip adjudication, PHASED: refuses only families
  with NEITHER a reducer fold mapping NOR an adapter claim, and only for
  NEWLY accepted operations — already-durable historical rows keep the
  exact lane's fold-skip-with-disclosure path, and retries of them
  replay; reject-before-accept, never append-then-refuse).
* ``republish_result_at_version(manifest_override=...)`` (r2a-fix M1):
  the no-op re-serve merges a manifest disclosure without hand-copying
  the manifest through the public publish path; payload bytes and
  checksum provenance stay the producing version's.
* :meth:`Store.advance_fast_revision` (r2b-review F6): the typed,
  stale-fenced, monotonic fast_revision advance in ONE store
  transaction, byte-identical to the scheduler's raw-SQL block.
* :meth:`Store.reset_scenario` rev-0 pin invalidation (r2a-fix-review
  F2): the exact lane's revision-0 bootstrap baseline must not survive
  the reset whose events voided it; the next bootstrap re-derives from
  post-reset history only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, get_args

import pytest

from solweig_gpu.server.jobs import ensure_exact_lane_bootstrap
from solweig_gpu.server.realtime.reducer import DeterministicEpochReducer
from solweig_gpu.server.realtime.types import SourceFamily
from solweig_gpu.server.store import (
    Store,
    UnsupportedFamily,
    result_payload_url,
)

from tests.test_realtime_epochs import WS, make_store, make_workspace, op_item


def _commit_tree_edit(store: Store, base_scene_version: int, tree_id: str) -> None:
    """One legacy ``/edits`` tree add, store-level (the pre-gate shape)."""
    store.commit_edits(
        WS,
        base_scene_version=base_scene_version,
        applied_edits=[
            {
                "operation": "add",
                "tree_id": tree_id,
                "old_tree": None,
                "new_tree": {
                    "tree_id": tree_id,
                    "u": 2.0,
                    "v": 3.0,
                    "height_m": 8.0,
                    "canopy_diameter_m": 4.0,
                },
            }
        ],
        requested=None,
    )


def _publish_baseline(store: Store) -> tuple[dict[str, Any], bytes]:
    def publish(scenario_id: str, scene_version: int) -> tuple[dict[str, Any], bytes]:
        manifest = {
            "scene_version": int(scene_version),
            "payload_url": result_payload_url(scenario_id, scene_version),
            "checksum": "sha256:baseline",
        }
        return manifest, b"baseline-payload"

    return publish


def _insert_raw_operation(store: Store, item: dict[str, Any]) -> None:
    """Simulate an ALREADY-DURABLE pre-change row (append refuses now)."""
    with store._write() as conn:
        conn.execute(
            "INSERT INTO realtime_operations (workspace_id, server_sequence, "
            "epoch_id, operation_id, actor_id, client_sequence, base_revision, "
            "source_family, entity_id, verb, payload_json, received_at, "
            "accepted_at) VALUES (?, 1, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                WS,
                item["operation_id"],
                item["actor_id"],
                item.get("client_sequence"),
                item.get("base_revision"),
                item["source_family"],
                item.get("entity_id"),
                item["verb"],
                json.dumps(dict(item["payload"]), sort_keys=True),
                item["received_at"],
                item["received_at"],
            ),
        )


# ---------------------------------------------------------------------------
# Seam 1: unsupported_family typed refusal (r2a endgame, phased)
# ---------------------------------------------------------------------------


class TestUnsupportedFamilyRefusal:
    def test_new_operation_of_unsupported_family_refused_before_any_write(
        self, tmp_path
    ):
        store = make_store(tmp_path)
        make_workspace(store, WS)
        with pytest.raises(UnsupportedFamily) as excinfo:
            store.append_operations(
                WS, [op_item("op-1", source_family="weather_alchemy")]
            )
        # machine-readable refusal body (route translation is a follow-up)
        assert excinfo.value.family == "weather_alchemy"
        body = excinfo.value.body()
        assert body["code"] == "unsupported_family"
        assert body["family"] == "weather_alchemy"
        assert body["advice"]
        # reject-before-accept: nothing durable, nothing half-consumed
        assert store.operations_since(WS, 0) == []
        assert store.epoch_records(WS) == []
        # the refused batch consumed no sequence: the next acceptance is 1
        records = store.append_operations(WS, [op_item("op-2")])
        assert [record.server_sequence for record in records] == [1]

    def test_batch_is_all_or_nothing(self, tmp_path):
        store = make_store(tmp_path)
        make_workspace(store, WS)
        with pytest.raises(UnsupportedFamily):
            store.append_operations(
                WS,
                [op_item("op-good"), op_item("op-bad", source_family="nope")],
            )
        assert store.operations_since(WS, 0) == []
        assert store.epoch_records(WS) == []

    def test_transport_vocabulary_never_refused(self, tmp_path):
        # The frozen operation vocabulary (realtime.types.SourceFamily) is
        # exactly what the HTTP transport accepts; the store must never
        # refuse any of it (defense in depth, never a second gate).
        store = make_store(tmp_path)
        make_workspace(store, WS)
        for index, family in enumerate(get_args(SourceFamily)):
            records = store.append_operations(
                WS, [op_item(f"op-{index}", source_family=family)]
            )
            assert records[0].duplicate is False

    def test_adapter_claimed_family_accepted_even_without_reducer_fold(
        self, tmp_path
    ):
        # Phasing (a): an adapter-backed family is supported even when the
        # reducer has no fold for it (the exact lane's fold-skip
        # disclosure covers those downstream).
        from solweig_gpu.incremental.edit_registry import builtin_adapter_metadata
        from solweig_gpu.server.realtime import reducer

        adapter_ids = {metadata.id for metadata in builtin_adapter_metadata()}
        extra = adapter_ids - set(reducer._ALL_FAMILIES)
        assert extra  # the premise holds: adapters claim beyond the fold set
        store = make_store(tmp_path)
        make_workspace(store, WS)
        for index, family in enumerate(sorted(extra)):
            records = store.append_operations(
                WS, [op_item(f"op-{index}", source_family=family)]
            )
            assert records[0].source_family == family

    def test_retry_of_durable_historical_unsupported_op_replays(self, tmp_path):
        # Phasing (b): already-durable rows of a now-refused family keep
        # replaying (no retroactive breakage); only NEW acceptances refuse.
        store = make_store(tmp_path)
        make_workspace(store, WS)
        _insert_raw_operation(
            store, op_item("historic-op", source_family="prehistoric_family")
        )
        replayed = store.append_operations(
            WS, [op_item("historic-op", source_family="prehistoric_family")]
        )
        assert replayed[0].duplicate is True
        with pytest.raises(UnsupportedFamily):
            store.append_operations(
                WS, [op_item("fresh-op", source_family="prehistoric_family")]
            )

    def test_durable_refused_family_rows_still_serve_existing_consumers(
        self, tmp_path
    ):
        # Phasing (b): a row already durable under a now-refused family
        # keeps serving every existing consumer unchanged — catch-up
        # reads, epoch membership, retry replay. The store adds no
        # retroactive refusal, and the exact lane's fold machinery is
        # untouched (reducer-known families — the only ones real
        # workspaces can carry, since the transport and the funnel only
        # ever produced those — are all inside the supported set).
        store = make_store(tmp_path)
        make_workspace(store, WS)
        _insert_raw_operation(
            store, op_item("historic-op", source_family="prehistoric_family")
        )
        assert [
            op.operation_id for op in store.operations_since(WS, 0)
        ] == ["historic-op"]
        assert [
            op.operation_id for op in store.operations_for_epoch(WS, 0)
        ] == ["historic-op"]
        assert store.get_operation(WS, "historic-op") is not None
        replayed = store.append_operations(
            WS, [op_item("historic-op", source_family="prehistoric_family")]
        )
        assert replayed[0].duplicate is True

    def test_mixed_known_family_workspaces_replay_unchanged(self, tmp_path):
        # Phasing (b), the realistic population: a workspace whose durable
        # history is all reducer-known (what the transport and the funnel
        # always produced) replays and folds exactly as before the seam.
        store = make_store(tmp_path)
        make_workspace(store, WS)
        store.append_operations(
            WS,
            [op_item("op-veg"), op_item("op-view", source_family="output_view")],
        )
        reducer = DeterministicEpochReducer()
        folded = reducer.reduce_epoch(
            reducer.initial_state(WS), store.operations_for_epoch(WS, 0)
        )
        committed = store.commit_epoch_reduction(
            WS, 0, families=folded.state.families, baseline_revision=0
        )
        assert committed is not None and committed.first_time is True
        assert store.canonical_state_at(WS, 1) is not None


# ---------------------------------------------------------------------------
# Seam 2: republish_result_at_version(manifest_override=...) (r2a-fix M1)
# ---------------------------------------------------------------------------


class TestRepublishManifestOverride:
    SOURCE_MANIFEST = {
        "scene_version": 0,
        "payload_url": None,  # asserted via result_payload_url
        "checksum": "sha256:source",
        "limitations": ["L1"],
        "metrics": {"duration_ms": 5, "nested": {"kept": True}},
    }

    def _workspace_with_source(self, tmp_path: Path) -> Store:
        store = make_store(tmp_path)
        source = json.loads(json.dumps(self.SOURCE_MANIFEST))
        store.create_scenario(
            site_id="site",
            name="s",
            scenario_id=WS,
            baseline_publisher=lambda sid, version: (source, b"payload-0"),
        )
        return store

    def test_override_merges_limitations_and_keeps_payload_provenance(
        self, tmp_path
    ):
        store = self._workspace_with_source(tmp_path)
        store.commit_edits(
            WS,
            base_scene_version=0,
            applied_edits=[
                {
                    "operation": "add",
                    "tree_id": "t1",
                    "old_tree": None,
                    "new_tree": {"tree_id": "t1", "u": 1.0, "v": 1.0},
                }
            ],
            requested=None,
        )
        override = {"limitations": ["exact lane: families skipped (rev 1)"]}
        store.republish_result_at_version(
            WS, 0, 1, job_id=None, manifest_override=override
        )
        result = store.get_result(WS, 1)
        assert result is not None
        # store-owned identity keys
        assert result.manifest["scene_version"] == 1
        assert result.manifest["payload_url"] == result_payload_url(WS, 1)
        # limitations MERGE: source first, override appended
        assert result.manifest["limitations"] == [
            "L1",
            "exact lane: families skipped (rev 1)",
        ]
        # payload bytes and checksum provenance stay the source's
        assert result.payload_bytes() == b"payload-0"
        assert result.manifest["checksum"] == "sha256:source"
        assert result.checksum == "sha256:source"
        assert result.exact is True
        # untouched keys copy verbatim
        assert result.manifest["metrics"] == {"duration_ms": 5, "nested": {"kept": True}}

    def test_override_merges_onto_manifest_without_limitations(self, tmp_path):
        store = make_store(tmp_path)
        source = json.loads(json.dumps(self.SOURCE_MANIFEST))
        del source["limitations"]
        store.create_scenario(
            site_id="site",
            name="s",
            scenario_id=WS,
            baseline_publisher=lambda sid, version: (source, b"payload-0"),
        )
        store.commit_edits(
            WS,
            base_scene_version=0,
            applied_edits=[
                {
                    "operation": "add",
                    "tree_id": "t1",
                    "old_tree": None,
                    "new_tree": {"tree_id": "t1", "u": 1.0, "v": 1.0},
                }
            ],
            requested=None,
        )
        store.republish_result_at_version(
            WS,
            0,
            1,
            job_id=None,
            manifest_override={"limitations": ["disclosure-only"]},
        )
        result = store.get_result(WS, 1)
        assert result.manifest["limitations"] == ["disclosure-only"]

    def test_override_is_deep_copied_and_store_keys_win(self, tmp_path):
        store = self._workspace_with_source(tmp_path)
        store.commit_edits(
            WS,
            base_scene_version=0,
            applied_edits=[
                {
                    "operation": "add",
                    "tree_id": "t1",
                    "old_tree": None,
                    "new_tree": {"tree_id": "t1", "u": 1.0, "v": 1.0},
                }
            ],
            requested=None,
        )
        override: dict[str, Any] = {
            # an override may not lie about the version or the payload URL
            "scene_version": 999,
            "payload_url": "/evil",
            "limitations": ["disclosure"],
        }
        store.republish_result_at_version(
            WS, 0, 1, job_id=None, manifest_override=override
        )
        # mutating the caller's override afterwards cannot reach the store
        override["limitations"].append("late mutation")
        override["metrics"] = {"smuggled": True}
        result = store.get_result(WS, 1)
        assert result.manifest["scene_version"] == 1
        assert result.manifest["payload_url"] == result_payload_url(WS, 1)
        assert result.manifest["limitations"] == ["L1", "disclosure"]
        assert result.manifest["metrics"] == {
            "duration_ms": 5,
            "nested": {"kept": True},
        }

    def test_without_override_copies_verbatim(self, tmp_path):
        store = self._workspace_with_source(tmp_path)
        store.commit_edits(
            WS,
            base_scene_version=0,
            applied_edits=[
                {
                    "operation": "add",
                    "tree_id": "t1",
                    "old_tree": None,
                    "new_tree": {"tree_id": "t1", "u": 1.0, "v": 1.0},
                }
            ],
            requested=None,
        )
        store.republish_result_at_version(WS, 0, 1, job_id=None)
        result = store.get_result(WS, 1)
        assert result.manifest["limitations"] == ["L1"]
        assert result.manifest["scene_version"] == 1
        assert result.manifest["payload_url"] == result_payload_url(WS, 1)
        assert result.payload_bytes() == b"payload-0"


# ---------------------------------------------------------------------------
# Seam 3: Store.advance_fast_revision typed accessor (r2b-review F6)
# ---------------------------------------------------------------------------


def _advance_fast_raw(
    store: Store,
    workspace_id: str,
    revision: int,
    exact_result_version: int,
    *,
    published_at: str,
) -> tuple[int, int] | None:
    """The r2b scheduler's raw-SQL block, VERBATIM (the parity reference).

    Returns ``(new_fast, exact_now)`` or ``None`` on the stale fence —
    exactly what ``Store.advance_fast_revision`` must reproduce.
    """
    with store._write() as conn:
        row = conn.execute(
            "SELECT scene_version, fast_revision, exact_result_version "
            "FROM scenarios WHERE scenario_id = ?",
            (workspace_id,),
        ).fetchone()
        if row is None or int(row["scene_version"]) < revision:
            return None
        exact_now = int(row["exact_result_version"])
        new_fast = max(int(row["fast_revision"]), revision, exact_now)
        conn.execute(
            "UPDATE scenarios SET fast_revision = ?, "
            "exact_base_revision = max(exact_base_revision, ?), "
            "updated_at = ? WHERE scenario_id = ?",
            (new_fast, exact_now, published_at, workspace_id),
        )
    return new_fast, exact_now


class TestAdvanceFastRevision:
    STAMP = "2026-09-04T12:00:00.000Z"

    def _workspace_at(self, tmp_path: Path, edits: int) -> Store:
        """Workspace at ``scene_version == edits``, exact at 0, fast at 0."""
        store = make_store(tmp_path)
        store.create_scenario(
            site_id="site",
            name="s",
            scenario_id=WS,
            baseline_publisher=lambda sid, version: (
                {
                    "scene_version": int(version),
                    "payload_url": result_payload_url(sid, version),
                    "checksum": "sha256:baseline",
                },
                b"baseline",
            ),
        )
        for index in range(edits):
            store.commit_edits(
                WS,
                base_scene_version=index,
                applied_edits=[
                    {
                        "operation": "add",
                        "tree_id": f"t{index}",
                        "old_tree": None,
                        "new_tree": {"tree_id": f"t{index}", "u": 1.0, "v": 1.0},
                    }
                ],
                requested=None,
            )
        return store

    def _workspace_exact_midway(self, tmp_path: Path) -> Store:
        """Workspace at scene_version 2 with an exact result at version 1
        (exact_result_version 1, fast_revision 0, exact_base_revision 0)."""
        store = self._workspace_at(tmp_path, edits=1)
        store.publish_result(
            WS, 1, manifest={"checksum": "sha256:x"}, payload=b"p", exact=True
        )
        store.commit_edits(
            WS,
            base_scene_version=1,
            applied_edits=[
                {
                    "operation": "add",
                    "tree_id": "t1",
                    "old_tree": None,
                    "new_tree": {"tree_id": "t1", "u": 2.0, "v": 2.0},
                }
            ],
            requested=None,
        )
        return store

    def test_stale_revision_refused_and_write_refused(self, tmp_path):
        store = self._workspace_at(tmp_path, edits=2)  # scene_version 2
        # race probe: a revision above the workspace's current version is
        # the stale write the r2b fence refuses — the typed accessor
        # refuses the SAME write, touching nothing.
        assert store.advance_fast_revision(WS, 3, 0) is None
        scenario = store.require_scenario(WS)
        assert scenario.fast_revision == 0
        assert scenario.exact_base_revision == 0
        # unknown workspace: the same refusal, not an error
        assert store.advance_fast_revision("ws_missing", 1, 0) is None

    def test_monotonic_clamped_advance(self, tmp_path):
        store = self._workspace_exact_midway(tmp_path)
        advance = store.advance_fast_revision(WS, 2, 1)
        assert advance is not None
        assert advance.workspace_id == WS
        assert advance.revision == 2
        # max(old_fast=0, revision=2, exact_now=1)
        assert advance.fast_revision == 2
        assert advance.exact_revision == 1
        scenario = store.require_scenario(WS)
        assert scenario.fast_revision == 2
        assert scenario.exact_base_revision == 1
        # a LATER call for an older revision never moves fast backwards
        stale_late = store.advance_fast_revision(WS, 1, 1)
        assert stale_late is not None
        assert stale_late.fast_revision == 2
        assert store.require_scenario(WS).fast_revision == 2

    def test_updated_at_stamp_honored(self, tmp_path):
        store = self._workspace_at(tmp_path, edits=1)
        store.advance_fast_revision(WS, 1, 0, updated_at=self.STAMP)
        row = store._read(
            "SELECT updated_at FROM scenarios WHERE scenario_id = ?", (WS,)
        )
        assert row["updated_at"] == self.STAMP

    def test_byte_identical_to_r2b_raw_sql_block(self, tmp_path):
        # The parity pin: identical workspaces driven through the r2b
        # scheduler's raw-SQL block and through the typed accessor end on
        # the SAME scenario row (revision math, fence, exact_base bump).
        raw_store = self._workspace_exact_midway(tmp_path / "raw")
        typed_store = self._workspace_exact_midway(tmp_path / "typed")
        raw = _advance_fast_raw(
            raw_store, WS, 2, 1, published_at=self.STAMP
        )
        typed = typed_store.advance_fast_revision(
            WS, 2, 1, updated_at=self.STAMP
        )
        assert raw == (2, 1)
        assert typed is not None
        assert (typed.fast_revision, typed.exact_revision) == raw
        raw_row = dict(raw_store._read(
            "SELECT scene_version, fast_revision, exact_result_version, "
            "exact_base_revision, updated_at FROM scenarios WHERE scenario_id = ?",
            (WS,),
        ))
        typed_row = dict(typed_store._read(
            "SELECT scene_version, fast_revision, exact_result_version, "
            "exact_base_revision, updated_at FROM scenarios WHERE scenario_id = ?",
            (WS,),
        ))
        assert typed_row == raw_row
        # and the fence agrees on the stale case too
        raw_store2 = self._workspace_at(tmp_path / "raw2", edits=1)
        typed_store2 = self._workspace_at(tmp_path / "typed2", edits=1)
        assert (
            _advance_fast_raw(
                raw_store2, WS, 5, 0, published_at=self.STAMP
            )
            is None
        )
        assert typed_store2.advance_fast_revision(WS, 5, 0) is None

    def test_scheduler_discovers_lagging_workspace_via_typed_read(self, tmp_path):
        store = self._workspace_at(tmp_path, edits=2)
        lagging = store.scenarios_with_fast_lag()
        assert [record.scenario_id for record in lagging] == [WS]
        assert lagging[0].scene_version == 2
        assert lagging[0].fast_revision == 0
        store.advance_fast_revision(WS, 2, 0)
        assert store.scenarios_with_fast_lag() == []


# ---------------------------------------------------------------------------
# Seam 4: reset_scenario rev-0 pin invalidation (r2a-fix-review F2)
# ---------------------------------------------------------------------------


class TestResetRev0PinInvalidation:
    GRID = {
        "rows": 10,
        "cols": 10,
        "pixel_size_m": 1.0,
        "origin_x_m": 0.0,
        "origin_y_m": 0.0,
    }

    def _pre_gate_workspace(self, store: Store, tree_id: str = "tree-1") -> None:
        """A workspace with an un-mirrored legacy tree edit whose
        collaborative gate is open (realtime operations exist)."""
        make_workspace(store, WS)
        _commit_tree_edit(store, base_scene_version=0, tree_id=tree_id)
        store.append_operations(WS, [op_item("op-1")])

    def test_reset_deletes_the_rev0_pin(self, tmp_path):
        store = make_store(tmp_path)
        self._pre_gate_workspace(store)
        pinned = ensure_exact_lane_bootstrap(store, WS, grid=self.GRID)
        assert pinned is not None and pinned.workspace_revision == 0
        assert "tree-1" in pinned.families["vegetation_geometry"]["objects"]
        store.reset_scenario(WS, baseline_publisher=_publish_baseline(store))
        # THE FIX: the pin carried the reset-deleted legacy object; it
        # must not survive the reset that voided it.
        assert store.canonical_state_at(WS, 0) is None

    def test_rebootstrap_after_reset_rederives_post_reset_history_only(
        self, tmp_path
    ):
        store = make_store(tmp_path)
        self._pre_gate_workspace(store)
        ensure_exact_lane_bootstrap(store, WS, grid=self.GRID)
        store.reset_scenario(WS, baseline_publisher=_publish_baseline(store))
        # a reset-voided ledger: nothing post-reset to fold, nothing pinned
        assert ensure_exact_lane_bootstrap(store, WS, grid=self.GRID) is None
        # a POST-reset legacy edit re-pins — carrying only post-reset objects
        _commit_tree_edit(store, base_scene_version=2, tree_id="tree-2")
        store.append_operations(WS, [op_item("op-2")])
        repinned = ensure_exact_lane_bootstrap(store, WS, grid=self.GRID)
        assert repinned is not None
        objects = repinned.families["vegetation_geometry"]["objects"]
        assert "tree-2" in objects
        assert "tree-1" not in objects  # the pre-reset object never resurrects

    def test_reset_idempotence_and_repeated_reset(self, tmp_path):
        store = make_store(tmp_path)
        self._pre_gate_workspace(store)
        ensure_exact_lane_bootstrap(store, WS, grid=self.GRID)
        store.reset_scenario(WS, baseline_publisher=_publish_baseline(store))
        store.reset_scenario(WS, baseline_publisher=_publish_baseline(store))
        assert store.canonical_state_at(WS, 0) is None
        # later canonical revisions are untouched by the seam: the pin is
        # the only row reset invalidates
        pinned = store.canonical_state_at(WS, 0)
        assert pinned is None
        store.append_operations(WS, [op_item("op-3")])
        reducer = DeterministicEpochReducer()
        folded = reducer.reduce_epoch(
            reducer.initial_state(WS), store.operations_for_epoch(WS, 0)
        )
        committed = store.commit_epoch_reduction(
            WS, 0, families=folded.state.families, baseline_revision=0
        )
        assert committed is not None
        # the post-reset close lands at scene_version + 1 (two resets: 3 + 1)
        assert committed.workspace_revision == 4
        assert store.canonical_state_at(WS, 4) is not None
