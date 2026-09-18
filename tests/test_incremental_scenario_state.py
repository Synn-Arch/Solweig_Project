# SPDX-License-Identifier: GPL-3.0-only
"""Scenario state persistence tests (U-D1, U-D intake item a).

Every test builds a real executor on the tiny-site fixture (solvers
stubbed for the fast suites) and proves the two halves of the contract:

* save/restore round-trips the COMPLETE scenario state (overlays, tree
  log + watermarks, scene revision, resolve memo, store index);
* a fresh executor restored from the snapshot continues the scenario
  BITWISE identically to the never-restarted sequence, without re-arming
  the published overlays as pending (u-c1b) and without weakening the
  store's publication discipline.
"""

from __future__ import annotations

import inspect
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_incremental_worker import (  # noqa: E402
    DATE_STR,
    TINY_ORIGIN,
    TINY_PIXEL,
    _build_cache,
    _make_tiny_site,
)
from test_incremental_met_integration import (  # noqa: E402
    _executor_met_edit,
    _stub_patch,
)
from test_incremental_building_integration import (  # noqa: E402
    NEW_BLOCK_B_STATE,
    NEW_BLOCK_STATE,
)

from solweig_gpu.incremental.edit_types import EditCommand
from solweig_gpu.incremental.executor import PlanExecutor
from solweig_gpu.incremental.geometry import InfluenceConfig
from solweig_gpu.incremental.result import load_patch
from solweig_gpu.incremental.scenario_state import (
    ScenarioStateError,
    read_snapshot,
    rebase_results_root,
    restore_into_executor,
    snapshot_from_executor,
    write_snapshot,
)
from solweig_gpu.incremental.store import StaleResultError, TemporalResultEntry
from solweig_gpu.incremental.trees import TreeLayer
from solweig_gpu.incremental.worker import PublicationWatermarks

LOCAL_ALWAYS = InfluenceConfig(full_recompute_fraction=1.0)
#: A paint class that differs from the baseline in the painted window (a
#: same-class repaint would be intercepted as a whole-batch value no-op).
PAINT_CLASSES = 2


def _make_executor(tmp_path: Path, **overrides) -> PlanExecutor:
    grid, site = _make_tiny_site(tmp_path / "site")
    cache = _build_cache(
        site,
        tmp_path / "cache",
        met_path=site / "metfiles" / f"metfile_0_0_{DATE_STR}.txt",
        site_id="tiny",
    )
    executor = PlanExecutor(
        cache=cache,
        layer=TreeLayer(cache.tree_base, grid),
        site_dir=site,
        results_root=overrides.pop("results_root", tmp_path / "results"),
        selected_date_str=overrides.pop("selected_date_str", DATE_STR),
        influence_config=overrides.pop("influence_config", LOCAL_ALWAYS),
        **overrides,
    )
    captured: dict = {}

    def fake_full(job_id, revision, forcing):
        captured["forcing"] = forcing
        return [
            _stub_patch(("utci",), job_id, revision, executor.grid.full_window, mode="full")
        ]

    def fake_local(job_id, revision, forcing, write_windows, **kwargs):
        captured["forcing"] = forcing
        return [
            _stub_patch(("utci",), job_id, revision, window, mode="local")
            for window in write_windows
        ]

    executor._worker._solve_full = fake_full
    executor._worker._solve_local = fake_local
    executor._captured = captured
    return executor


def _veg_edit(executor: PlanExecutor, *, edit_id="veg-1", tree_id="t1", n=40.5):
    return executor.validate(
        EditCommand(
            edit_id=edit_id,
            scenario_id=executor._scenario_id,
            base_scene_revision=executor.scene_revision,
            adapter_id="vegetation_geometry",
            operation="add",
            old_state=None,
            new_state={
                "tree_id": tree_id,
                "x_m": TINY_ORIGIN[0] + n * TINY_PIXEL,
                "y_m": TINY_ORIGIN[1] - n * TINY_PIXEL,
                "height_m": 4.0,
                "canopy_radius_m": 2.0,
            },
            requested_outputs=("utci",),
            requested_times=(1,),
        )
    )


def _paint_edit(executor: PlanExecutor, *, edit_id="lc-1"):
    return executor.validate(
        EditCommand(
            edit_id=edit_id,
            scenario_id=executor._scenario_id,
            base_scene_revision=executor.scene_revision,
            adapter_id="landcover_surface",
            operation="paint",
            old_state=None,
            new_state={
                "window": {
                    "row_start": 40,
                    "row_stop": 48,
                    "col_start": 40,
                    "col_stop": 48,
                },
                "classes": PAINT_CLASSES,
            },
            requested_outputs=("utci",),
            requested_times=(1,),
        )
    )


def _paint_at(executor: PlanExecutor, *, edit_id, row, cls):
    """A paint of a 4x4 window at ``row`` (the reviewer's repro shapes)."""
    return executor.validate(
        EditCommand(
            edit_id=edit_id,
            scenario_id=executor._scenario_id,
            base_scene_revision=executor.scene_revision,
            adapter_id="landcover_surface",
            operation="paint",
            old_state=None,
            new_state={
                "window": {
                    "row_start": row,
                    "row_stop": row + 4,
                    "col_start": 40,
                    "col_stop": 44,
                },
                "classes": cls,
            },
            requested_outputs=("utci",),
            requested_times=(1,),
        )
    )


def _params_edit(executor: PlanExecutor, *, edit_id="params-1"):
    return executor.validate(
        EditCommand(
            edit_id=edit_id,
            scenario_id=executor._scenario_id,
            base_scene_revision=executor.scene_revision,
            adapter_id="model_receptor_parameters",
            operation="update",
            old_state=None,
            new_state={"albedo_b": 0.35},
            requested_outputs=("utci",),
            requested_times=(1,),
        )
    )


def _save(executor: PlanExecutor, directory: Path) -> None:
    write_snapshot(
        snapshot_from_executor(executor),
        directory,
        landcover_resolved=executor._applied_landcover_resolved,
    )


def _restored_twin(source: PlanExecutor, tmp_path: Path) -> PlanExecutor:
    """A fresh executor (fresh store, fresh layer) restored from a save."""
    twin = _make_executor(
        tmp_path,
        results_root=source.results_root,
        selected_date_str=source.selected_date_str,
        influence_config=LOCAL_ALWAYS,
    )
    restore_into_executor(twin, read_snapshot(tmp_path / "state"), directory=tmp_path / "state")
    return twin


def _assert_state_equal(a: PlanExecutor, b: PlanExecutor) -> None:
    assert b.scene_revision == a.scene_revision
    assert dict(b.state.node_versions) == dict(a.state.node_versions)
    assert b.layer.current_trees() == a.layer.current_trees()
    assert len(b.layer.edits) == len(a.layer.edits)
    assert b._applied_forcing_overlay == a._applied_forcing_overlay
    assert b._applied_landcover_overlay == a._applied_landcover_overlay
    assert b._applied_model_parameters == a._applied_model_parameters
    # u-d4c: the accumulated building fold round-trips (order included —
    # it is the chain's edit order on the next batch).
    assert list(b._applied_massing_edits) == list(a._applied_massing_edits)
    assert b._applied_massing_edits == a._applied_massing_edits
    if a._applied_landcover_resolved is None:
        assert b._applied_landcover_resolved is None
    else:
        assert np.array_equal(
            b._applied_landcover_resolved, a._applied_landcover_resolved
        )
    assert b._worker.publication_watermarks == a._worker.publication_watermarks
    assert sorted(b.store) == sorted(a.store)
    for key in sorted(a.store):
        assert b.store.revision_at(*key) == a.store.revision_at(*key)
        assert b.store.lookup(*key).mode == a.store.lookup(*key).mode
        assert len(b.store.window_entries(*key)) == len(a.store.window_entries(*key))


class TestScenarioStatePersistence:
    def test_mixed_sequence_resumes_bitwise(self, tmp_path: Path) -> None:
        """The full accumulated state survives a restart bitwise.

        Sequence: vegetation, met, paint, parameters (four families), then
        a fresh executor restores and continues with one more met edit;
        every scientific input (forcing table, staged parameters) and the
        published store state must match the never-restarted twin exactly.
        """
        ex = _make_executor(tmp_path)
        assert ex.execute([_veg_edit(ex)]).published
        assert ex.execute([_executor_met_edit(ex, time_index=1, value=26.0)]).published
        assert ex.execute([_paint_edit(ex)]).published
        assert ex.execute([_params_edit(ex)]).published
        _save(ex, tmp_path / "state")

        twin = _restored_twin(ex, tmp_path)
        _assert_state_equal(ex, twin)

        # The continuing edit is utci_only (wind_speed): a
        # radiation-affecting edit would route the G2.1 warm sparse path,
        # whose real split solve no tiny-site stub can serve, while the
        # r3a refusal (no tmrt published here) keeps the batch on the
        # stubbed worker path this suite tests state on.
        continued = ex.execute(
            [
                _executor_met_edit(
                    ex, time_index=2, value=40.0, variable="wind_speed",
                    edit_id="m2",
                )
            ]
        )
        resumed = twin.execute(
            [
                _executor_met_edit(
                    twin, time_index=2, value=40.0, variable="wind_speed",
                    edit_id="m2",
                )
            ]
        )
        assert continued.published and resumed.published
        assert resumed.scene_revision == continued.scene_revision
        # Bitwise-identical scientific inputs.
        assert np.array_equal(
            twin._captured["forcing"].met_table, ex._captured["forcing"].met_table
        )
        assert twin._worker._model_parameters == ex._worker._model_parameters
        # Bitwise-identical published products at the same (node, time).
        for key in sorted(ex.store):
            left = load_patch(ex.store.lookup(*key).patch_path)
            right = load_patch(twin.store.lookup(*key).patch_path)
            assert np.array_equal(
                left.arrays["utci"], right.arrays["utci"]
            ), f"patch arrays differ at {key}"
            assert left.scene_revision == right.scene_revision
            assert left.mode == right.mode

    def test_restore_does_not_rearm_published_overlays(self, tmp_path: Path) -> None:
        """u-c1b: restoring adopts watermarks; it never re-arms them PENDING.

        The post-restore worker must sit in the exact steady state a
        never-restarted twin holds between batches: the accumulated
        overlays are staged AND published (staged == published is "nothing
        pending" — the worker's pending detection is staged != published,
        plus the presence/dirty-override flags, all off). Staging something
        DIFFERENT from the published watermark would mark it pending
        forever and every later job would run FULL tile. After a correct
        restore, a pure vegetation edit executes as a LOCAL job under the
        accumulated overlay.
        """
        ex = _make_executor(tmp_path)
        assert ex.execute([_executor_met_edit(ex, time_index=1, value=26.0)]).published
        _save(ex, tmp_path / "state")
        twin = _restored_twin(ex, tmp_path)

        # Nothing pending: staged == published, flags off, no tree edits.
        assert twin._worker.pending_batch() is None
        assert twin._worker._forcing_presence_pending is False
        assert twin._worker._landcover_dirty_override in (None, ())
        assert twin._worker._forcing_overlay == twin._worker._published_forcing_overlay
        assert (
            twin._worker._landcover_overlay
            == twin._worker._published_landcover_overlay
        )
        assert (
            twin._worker._model_parameters
            == twin._worker._published_model_parameters
        )
        # ...and the staged state is the ACCUMULATED scenario, identical to
        # the never-restarted executor's steady state (the twin can
        # materialize the overlay forcing without a new batch).
        assert twin._worker._forcing_overlay == ex._worker._forcing_overlay
        assert np.array_equal(
            twin._worker.forcing().met_table, ex._worker.forcing().met_table
        )

        resumed = twin.execute([_veg_edit(twin)])
        assert resumed.published
        assert resumed.mode == "local"  # full would mean re-armed overlays
        # And the vegetation job still ran under the accumulated overlay.
        assert twin._captured["forcing"].met_table[1, 11] == 26.0

    def test_vegetation_only_roundtrip(self, tmp_path: Path) -> None:
        ex = _make_executor(tmp_path)
        assert ex.execute([_veg_edit(ex)]).published
        _save(ex, tmp_path / "state")
        twin = _restored_twin(ex, tmp_path)
        _assert_state_equal(ex, twin)
        assert twin._worker.pending_batch() is None

        # A second vegetation edit becomes exactly one pending layer edit
        # (the watermark consumed the restored history correctly).
        resumed = twin.execute([_veg_edit(twin, edit_id="veg-2", tree_id="t2", n=60.5)])
        assert resumed.published
        straight = ex.execute([_veg_edit(ex, edit_id="veg-2", tree_id="t2", n=60.5)])
        assert straight.published
        assert resumed.mode == straight.mode
        assert len(twin.layer.current_trees()) == len(ex.layer.current_trees()) == 2

    def test_meteorology_only_roundtrip(self, tmp_path: Path) -> None:
        ex = _make_executor(tmp_path)
        assert ex.execute([_executor_met_edit(ex, time_index=1, value=26.0)]).published
        _save(ex, tmp_path / "state")
        twin = _restored_twin(ex, tmp_path)
        _assert_state_equal(ex, twin)

        # utci_only continuing edit: keeps the batch on the stubbed worker
        # path (see test_mixed_sequence_resumes_bitwise for the routing
        # note) while still folding ON TOP of the restored overlay.
        continued = ex.execute(
            [
                _executor_met_edit(
                    ex, time_index=2, value=30.0, variable="wind_speed",
                    edit_id="m2",
                )
            ]
        )
        resumed = twin.execute(
            [
                _executor_met_edit(
                    twin, time_index=2, value=30.0, variable="wind_speed",
                    edit_id="m2",
                )
            ]
        )
        assert continued.published and resumed.published
        table = twin._captured["forcing"].met_table
        assert table[1, 11] == 26.0  # the earlier edit survived the restart
        assert table[2, 9] == 30.0
        assert np.array_equal(table, ex._captured["forcing"].met_table)

    def test_landcover_only_roundtrip(self, tmp_path: Path) -> None:
        ex = _make_executor(tmp_path)
        assert ex.execute([_paint_edit(ex)]).published
        assert ex._applied_landcover_resolved is not None
        _save(ex, tmp_path / "state")
        twin = _restored_twin(ex, tmp_path)
        _assert_state_equal(ex, twin)

        resolved = twin._applied_landcover_resolved
        assert resolved is not None
        assert np.array_equal(resolved[40:48, 40:48], np.full((8, 8), PAINT_CLASSES))
        # The restored resolve memo is what the next job stages.
        resumed = twin.execute([_veg_edit(twin)])
        assert resumed.published
        staged = twin._worker._landcover_overlay
        assert staged is not None
        assert np.array_equal(
            staged.resolve(np.asarray(twin.cache.landcover)), resolved
        )

    def test_model_parameters_only_roundtrip(self, tmp_path: Path) -> None:
        ex = _make_executor(tmp_path)
        assert ex.execute([_params_edit(ex)]).published
        _save(ex, tmp_path / "state")
        twin = _restored_twin(ex, tmp_path)
        _assert_state_equal(ex, twin)
        assert twin._applied_model_parameters == {"albedo_b": 0.35}

        # A later parameter edit folds ON TOP of the restored value.
        second = twin.validate(
            EditCommand(
                edit_id="params-2",
                scenario_id=twin._scenario_id,
                base_scene_revision=twin.scene_revision,
                adapter_id="model_receptor_parameters",
                operation="update",
                old_state=None,
                new_state={"albedo_b": 0.42, "ewall": 0.85},
                requested_outputs=("utci",),
                requested_times=(1,),
            )
        )
        assert twin.execute([second]).published
        straight = ex.validate(
            EditCommand(
                edit_id="params-2",
                scenario_id=ex._scenario_id,
                base_scene_revision=ex.scene_revision,
                adapter_id="model_receptor_parameters",
                operation="update",
                old_state=None,
                new_state={"albedo_b": 0.42, "ewall": 0.85},
                requested_outputs=("utci",),
                requested_times=(1,),
            )
        )
        assert ex.execute([straight]).published
        # The fold landed identically on both sides.
        assert twin._worker._model_parameters == ex._worker._model_parameters
        assert twin._worker._model_parameters == {"albedo_b": 0.42, "ewall": 0.85}

    def test_restore_rebuilds_store_publication_discipline(self, tmp_path: Path) -> None:
        """The replayed store index re-arms the stale-revision guard."""
        ex = _make_executor(tmp_path)
        assert ex.execute([_veg_edit(ex)]).published  # revision 1
        _save(ex, tmp_path / "state")
        twin = _restored_twin(ex, tmp_path)
        top_revision = twin.store.max_revision()
        assert top_revision == 1
        stale_entry = TemporalResultEntry(
            node_id="utci",
            time_index=0,
            scene_revision=top_revision - 1,
            job_id="stale-job",
            mode="full",
            write_window=twin.grid.full_window,
        )
        with pytest.raises(StaleResultError):
            twin.store.publish([stale_entry], writer=twin._store_writer_id)

    def test_save_refuses_contradictory_pairing(self, tmp_path: Path) -> None:
        """u-c1b pairing: a worker advanced past the executor state never
        snapshots (restoring that would re-arm or revert silently)."""
        ex = _make_executor(tmp_path)
        assert ex.execute([_executor_met_edit(ex, time_index=1, value=26.0)]).published
        wedge = _executor_met_edit(ex, time_index=2, value=30.0, edit_id="wedge")
        from solweig_gpu.incremental.adapters.met_time import overlay_from_deltas

        poisoned = overlay_from_deltas([wedge.delta])
        ex._worker.adopt_published_state(
            PublicationWatermarks(
                acked_sequence=ex._worker.publication_watermarks.acked_sequence,
                forcing_overlay=poisoned,
                landcover_overlay=ex._worker.publication_watermarks.landcover_overlay,
                model_parameters=ex._worker.publication_watermarks.model_parameters,
            )
        )
        with pytest.raises(ScenarioStateError, match="pairing"):
            snapshot_from_executor(ex)

    def test_restore_refuses_foreign_executor(self, tmp_path: Path) -> None:
        ex = _make_executor(tmp_path)
        assert ex.execute([_veg_edit(ex)]).published
        _save(ex, tmp_path / "state")

        # Different influence policy: a different computational identity.
        foreign = _make_executor(
            tmp_path / "foreign",
            results_root=ex.results_root,
            influence_config=InfluenceConfig(full_recompute_fraction=0.5),
        )
        with pytest.raises(ScenarioStateError, match="influence config"):
            restore_into_executor(
                foreign, read_snapshot(tmp_path / "state"), directory=tmp_path / "state"
            )

        # Different selected date: a different solar day.
        other_day = _make_executor(
            tmp_path / "otherday",
            results_root=ex.results_root,
            selected_date_str="2024-12-21",
        )
        with pytest.raises(ScenarioStateError, match="selected_date_str"):
            restore_into_executor(
                other_day, read_snapshot(tmp_path / "state"), directory=tmp_path / "state"
            )

    def test_restore_maps_acked_over_rollback_history(self, tmp_path: Path) -> None:
        """A scenario with a FAILED-then-retried batch: the layer log
        carries replay+undo records with sequence gaps, and the restored
        watermark must still consume exactly the published history."""
        ex = _make_executor(tmp_path)
        edit = _veg_edit(ex)

        def exploding(job_id, revision, forcing, write_windows, **kwargs):
            raise RuntimeError("first attempt explodes")

        ex._worker._solve_local = exploding
        with pytest.raises(RuntimeError, match="explodes"):
            ex.execute([edit])
        # The rollback undid the layer: the log now carries add+delete.
        assert len(ex.layer.edits) == 2
        assert not ex.layer.current_trees()

        def working(job_id, revision, forcing, write_windows, **kwargs):
            ex._captured["forcing"] = forcing
            return [
                _stub_patch(("utci",), job_id, revision, window, mode="local")
                for window in write_windows
            ]

        ex._worker._solve_local = working
        retry = _veg_edit(ex, edit_id="veg-retry")
        assert ex.execute([retry]).published
        log_length = len(ex.layer.edits)  # add, delete(undo), add(retry)
        assert log_length == 3
        assert ex._worker.publication_watermarks.acked_sequence == 3

        _save(ex, tmp_path / "state")
        twin = _restored_twin(ex, tmp_path)
        _assert_state_equal(ex, twin)
        assert twin._worker.pending_batch() is None  # retry consumed

        resumed = twin.execute([_veg_edit(twin, edit_id="veg-2", tree_id="t2", n=60.5)])
        assert resumed.published
        assert resumed.mode == "local"

    def test_save_and_restore_at_revision_zero(self, tmp_path: Path) -> None:
        """A never-executed scenario round-trips too (empty state)."""
        ex = _make_executor(tmp_path)
        _save(ex, tmp_path / "state")
        twin = _restored_twin(ex, tmp_path)
        _assert_state_equal(ex, twin)
        assert twin.scene_revision == 0
        assert twin.execute([_veg_edit(twin)]).published

    def test_honest_overwrite_save_round_trips(self, tmp_path: Path) -> None:
        """A COMPLETE re-save (memo v2 + JSON v2) is exactly restorable.

        The crash-window fix pairs every JSON document with a hash of the
        memo it was written against; this pins the other side of that
        contract — a second save into the same directory, after further
        paint batches, must not be refused and must restore bitwise.
        """
        ex = _make_executor(tmp_path)
        assert ex.execute([_paint_at(ex, edit_id="p1", row=40, cls=2)]).published
        _save(ex, tmp_path / "state")

        # Continue the scenario and RE-SAVE into the same directory.
        assert ex.execute([_paint_at(ex, edit_id="p2", row=60, cls=5)]).published
        _save(ex, tmp_path / "state")

        twin = _restored_twin(ex, tmp_path)
        _assert_state_equal(ex, twin)
        resumed = twin.execute([_veg_edit(twin)])
        assert resumed.published

    def test_torn_overwrite_save_is_refused(self, tmp_path: Path) -> None:
        """u-d1b HIGH: JSON v1 + memo v2 must NEVER load as a half-state.

        Mirrors the reviewer's repro: save #1 clean (paint A published);
        paint B published; the re-save crashes after the memo replace but
        before the JSON replace. The torn directory must be refused by a
        typed error — on the unfixed code it loaded, feeding the
        checkpoint-2 memo to a checkpoint-1 executor, and the operator's
        re-submitted paint B was intercepted as a whole-batch no-op while
        the clean twin PUBLISHED it.
        """
        ex = _make_executor(tmp_path)
        assert ex.execute([_paint_at(ex, edit_id="p1", row=40, cls=2)]).published
        _save(ex, tmp_path / "state")
        shutil.copytree(tmp_path / "state", tmp_path / "clean")

        assert ex.execute([_paint_at(ex, edit_id="p2", row=60, cls=5)]).published
        memo2 = ex._applied_landcover_resolved

        # The crash: ONLY the new memo lands; the JSON document stays v1.
        tmp_memo = tmp_path / "state" / "landcover-resolved.npy.tmp.npy"
        np.save(tmp_memo, memo2, allow_pickle=False)
        tmp_memo.replace(tmp_path / "state" / "landcover-resolved.npy")

        with pytest.raises(ScenarioStateError, match="memo/JSON pair"):
            read_snapshot(tmp_path / "state")

        # The control: the clean checkpoint restores, and the operator's
        # re-submitted paint B PUBLISHES there — the behavior the torn
        # restore used to silently diverge from.
        clean = _make_executor(
            tmp_path / "clean_root",
            results_root=ex.results_root,
            selected_date_str=ex.selected_date_str,
            influence_config=LOCAL_ALWAYS,
        )
        restore_into_executor(
            clean, read_snapshot(tmp_path / "clean"), directory=tmp_path / "clean"
        )
        outcome = clean.execute([_paint_at(clean, edit_id="p2", row=60, cls=5)])
        assert outcome.published, outcome.diagnostics
        assert np.unique(clean._applied_landcover_resolved[60:64, 40:44]) == 5

    def test_memo_swapped_after_read_is_refused_at_restore(self, tmp_path: Path) -> None:
        """Defense in depth: the pairing hash re-verifies at LOAD time.

        read_snapshot validated the pair, but the memo bytes can change
        between the read and the restore (another writer, tampering); the
        restore must refuse rather than feed a foreign memo as
        resolved_old.
        """
        ex = _make_executor(tmp_path)
        assert ex.execute([_paint_edit(ex)]).published
        _save(ex, tmp_path / "state")

        snapshot = read_snapshot(tmp_path / "state")  # pair verifies here
        foreign = np.zeros_like(ex._applied_landcover_resolved)
        np.save(tmp_path / "state" / "landcover-resolved.npy", foreign, allow_pickle=False)
        twin = _make_executor(
            tmp_path / "twin",
            results_root=ex.results_root,
            selected_date_str=ex.selected_date_str,
            influence_config=LOCAL_ALWAYS,
        )
        with pytest.raises(ScenarioStateError, match="memo/JSON pair"):
            restore_into_executor(twin, snapshot, directory=tmp_path / "state")
        assert twin._applied_landcover_resolved is None  # refused whole

    def test_read_snapshot_types_missing_keys(self, tmp_path: Path) -> None:
        """A document missing a key is a typed refusal, never a KeyError."""
        ex = _make_executor(tmp_path)
        assert ex.execute([_veg_edit(ex)]).published
        _save(ex, tmp_path / "state")

        document_path = tmp_path / "state" / "scenario-state.json"
        document = json.loads(document_path.read_text())
        del document["scene_revision"]
        document_path.write_text(json.dumps(document))

        with pytest.raises(ScenarioStateError, match="scene_revision"):
            read_snapshot(tmp_path / "state")

    def test_restore_refuses_tampered_trunk_ratio_typed(self, tmp_path: Path) -> None:
        """u-e1 L1: the restore path rebuilt ``TreeSpec`` records with no
        schema validation, so a tampered snapshot (``trunk_ratio`` = 1.0,
        outside the half-open [0, 1) interval every live edit enforces)
        surfaced as a RAW ValueError mid-replay instead of a typed
        restore refusal."""
        ex = _make_executor(tmp_path)
        assert ex.execute([_veg_edit(ex)]).published
        _save(ex, tmp_path / "state")

        path = tmp_path / "state" / "scenario-state.json"
        document = json.loads(path.read_text())
        assert document["tree_edits"], "no tree edit in the snapshot"
        document["tree_edits"][0]["new_tree"]["trunk_ratio"] = 1.0
        path.write_text(json.dumps(document))

        # read_snapshot stays permissive (it is a document reader); the
        # RESTORE gate is where the spec re-enters the live layer and
        # must refuse it typed.
        twin = _make_executor(
            tmp_path / "twin",
            results_root=ex.results_root,
            influence_config=LOCAL_ALWAYS,
        )
        with pytest.raises(ScenarioStateError, match="trunk_ratio"):
            restore_into_executor(
                twin, read_snapshot(tmp_path / "state"), directory=tmp_path / "state"
            )
        # The refusal is whole-restore: no tree reached the twin's layer.
        assert twin.layer.current_trees() == ()

    def test_restore_refuses_divergent_results_root(self, tmp_path: Path) -> None:
        """The snapshot pins the save-time results root (store entries
        carry absolute patch paths under it); a restore under a different
        root is a typed refusal, not a silent re-point."""
        ex = _make_executor(tmp_path)
        assert ex.execute([_veg_edit(ex)]).published
        _save(ex, tmp_path / "state")

        elsewhere = _make_executor(
            tmp_path / "elsewhere",
            results_root=tmp_path / "elsewhere" / "results",
            influence_config=LOCAL_ALWAYS,
        )
        with pytest.raises(ScenarioStateError, match="results_root"):
            restore_into_executor(
                elsewhere,
                read_snapshot(tmp_path / "state"),
                directory=tmp_path / "state",
            )

    # -- u-d4: explicit operator rebase for moved deployments ---------------

    def _moved_deployment(self, tmp_path: Path) -> tuple[PlanExecutor, Path, Path]:
        """Publish one batch, save, and physically relocate the results tree."""
        ex = _make_executor(tmp_path)
        assert ex.execute([_veg_edit(ex)]).published
        _save(ex, tmp_path / "state")
        moved = tmp_path / "moved-results"
        shutil.move(str(tmp_path / "results"), str(moved))
        return ex, moved, tmp_path / "state"

    def test_rebase_requires_explicit_opt_in(self, tmp_path: Path) -> None:
        """Without allow_root_move the rebase is a typed refusal, not a move."""
        ex, moved, state = self._moved_deployment(tmp_path)
        snapshot = read_snapshot(state)
        with pytest.raises(ScenarioStateError, match="allow_root_move"):
            rebase_results_root(snapshot, moved)
        assert read_snapshot(state).results_root == str(tmp_path / "results")

    def test_rebase_translates_paths_and_enables_restore(
        self, tmp_path: Path
    ) -> None:
        """The documented operator path: move the tree, rebase with the
        explicit flag, and the snapshot restores under the new root — with
        every translated patch path verified to exist first."""
        ex, moved, state = self._moved_deployment(tmp_path)
        snapshot = read_snapshot(state)
        assert snapshot.store_entries
        assert all(
            entry["patch_path"].startswith(str(tmp_path / "results"))
            for entry in snapshot.store_entries
        )

        rebased = rebase_results_root(
            snapshot, moved, allow_root_move=True, snapshot_directory=state
        )
        assert rebased.results_root == str(moved)
        assert all(
            entry["patch_path"].startswith(str(moved))
            for entry in rebased.store_entries
        )
        # The document on disk was rewritten atomically and reads back equal.
        assert read_snapshot(state) == rebased

        # A fresh executor under the new root now restores — the pre-rebase
        # refusal is gone — and continues the scenario bitwise.
        twin = _make_executor(
            tmp_path / "elsewhere",
            results_root=moved,
            influence_config=LOCAL_ALWAYS,
        )
        restore_into_executor(twin, read_snapshot(state), directory=state)
        _assert_state_equal(ex, twin)
        assert twin.execute([_veg_edit(twin, edit_id="veg-2", tree_id="t2")]).published

    def test_rebase_refuses_missing_patches_and_keeps_document(
        self, tmp_path: Path
    ) -> None:
        """A target root missing the published patches is refused and the
        on-disk document is untouched (single atomic rewrite, only after
        every check passed)."""
        _ex, _moved, state = self._moved_deployment(tmp_path)
        before = (state / "scenario-state.json").read_text()
        empty = tmp_path / "empty-root"
        empty.mkdir()
        snapshot = read_snapshot(state)
        with pytest.raises(ScenarioStateError, match="do not exist"):
            rebase_results_root(
                snapshot, empty, allow_root_move=True, snapshot_directory=state
            )
        assert (state / "scenario-state.json").read_text() == before

    def test_rebase_refuses_nonexistent_root(self, tmp_path: Path) -> None:
        _ex, _moved, state = self._moved_deployment(tmp_path)
        snapshot = read_snapshot(state)
        with pytest.raises(ScenarioStateError, match="does not exist"):
            rebase_results_root(
                snapshot,
                tmp_path / "not-there",
                allow_root_move=True,
                snapshot_directory=state,
            )

    def test_rebase_reverifies_memo_hash(self, tmp_path: Path) -> None:
        """A snapshot declaring the resolve memo re-verifies it against the
        recorded sha256 before rewriting anything."""
        ex = _make_executor(tmp_path)
        assert ex.execute([_paint_edit(ex)]).published
        assert ex._applied_landcover_resolved is not None
        _save(ex, tmp_path / "state")
        moved = tmp_path / "moved-results"
        shutil.move(str(tmp_path / "results"), str(moved))
        snapshot = read_snapshot(tmp_path / "state")

        # Tamper with the memo: the rebase must refuse (the moved
        # deployment's snapshot directory is not intact).
        memo = tmp_path / "state" / "landcover-resolved.npy"
        np.save(memo, np.zeros_like(np.load(memo)), allow_pickle=False)
        with pytest.raises(ScenarioStateError, match="landcover_resolved_sha256"):
            rebase_results_root(
                snapshot, moved, allow_root_move=True, snapshot_directory=tmp_path / "state"
            )

    def test_rebase_requires_snapshot_directory_for_memo_snapshots(
        self, tmp_path: Path
    ) -> None:
        ex = _make_executor(tmp_path)
        assert ex.execute([_paint_edit(ex)]).published
        _save(ex, tmp_path / "state")
        moved = tmp_path / "moved-results"
        shutil.move(str(tmp_path / "results"), str(moved))
        snapshot = read_snapshot(tmp_path / "state")
        with pytest.raises(ScenarioStateError, match="snapshot_directory"):
            rebase_results_root(snapshot, moved, allow_root_move=True)

    def test_write_refuses_undeclared_memo(self, tmp_path: Path) -> None:
        """The declaration/supply pairing is symmetric: a memo supplied to
        write while the snapshot does not declare one is refused (the
        document would deny a memo the directory carries)."""
        from solweig_gpu.incremental.scenario_state import (
            snapshot_from_executor as _sfe,
        )

        ex = _make_executor(tmp_path)
        assert ex.execute([_veg_edit(ex)]).published
        snapshot = _sfe(ex)
        assert snapshot.landcover_resolved is None
        with pytest.raises(ScenarioStateError, match="does not declare"):
            write_snapshot(
                snapshot,
                tmp_path / "state",
                landcover_resolved=np.zeros((ex.cache.rows, ex.cache.cols), dtype=np.int16),
            )


# ---------------------------------------------------------------------------
# Scientific restart: real SVF, real solver — bitwise A/B against the
# never-restarted sequence (the only oracle that matters for persistence).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sci_site(tmp_path_factory):
    """A real-SVF 128x128 prepared site and its cache (module-scoped: the
    SVF precompute is the expensive part and every restart test shares it)."""
    from types import SimpleNamespace

    from solweig_gpu.incremental.geometry import TreeSpec
    from test_incremental_worker import (
        _compute_baseline_svf,
        _make_prepared_site,
    )

    root = tmp_path_factory.mktemp("scenario_state_site")
    base = TreeSpec(
        "b1",
        300000.0 + 20.5 * 2.0,
        4100000.0 - 20.5 * 2.0,
        3.0,
        2.0,
    )
    grid, site = _make_prepared_site(
        root,
        rows=128,
        cols=128,
        pixel=2.0,
        origin=(300000.0, 4100000.0),
        epsg=32616,
        base_trees=(base,),
        met_hours=range(10, 14),
    )
    _compute_baseline_svf(site)
    cache = _build_cache(
        site,
        root / "cache",
        met_path=site / "metfiles" / f"metfile_0_0_{DATE_STR}.txt",
        site_id="state-int",
    )
    return SimpleNamespace(grid=grid, site=site, cache=cache)


@pytest.mark.scientific
class TestScientificRestart:

    def _real_executor(self, sci_site, root: Path) -> PlanExecutor:
        return PlanExecutor(
            cache=sci_site.cache,
            layer=TreeLayer(sci_site.cache.tree_base, sci_site.grid),
            site_dir=sci_site.site,
            results_root=root / "results",
            selected_date_str=DATE_STR,
            influence_config=LOCAL_ALWAYS,
        )

    def test_mixed_sequence_restart_is_bitwise(self, sci_site, tmp_path: Path) -> None:
        """Real physics: met + paint + vegetation in one mixed batch, save,
        restart into a fresh process-equivalent executor, continue with a
        second met batch — every published band matches the
        never-restarted twin bitwise."""
        ex = self._real_executor(sci_site, tmp_path)

        def paint(executor):
            return executor.validate(
                EditCommand(
                    edit_id="lc-1",
                    scenario_id=executor._scenario_id,
                    base_scene_revision=executor.scene_revision,
                    adapter_id="landcover_surface",
                    operation="paint",
                    old_state=None,
                    new_state={
                        "window": {
                            "row_start": 40,
                            "row_stop": 48,
                            "col_start": 40,
                            "col_stop": 48,
                        },
                        # The prepared site's baseline class there is 5;
                        # painting 1 is a value change (no no-op intercept).
                        "classes": 1,
                    },
                    requested_outputs=("utci",),
                    requested_times=(1,),
                )
            )

        def tree(executor):
            return executor.validate(
                EditCommand(
                    edit_id="veg-1",
                    scenario_id=executor._scenario_id,
                    base_scene_revision=executor.scene_revision,
                    adapter_id="vegetation_geometry",
                    operation="add",
                    old_state=None,
                    new_state={
                        "tree_id": "t2",
                        "x_m": sci_site.grid.origin_x_m
                        + 60.5 * sci_site.grid.pixel_size_m,
                        "y_m": sci_site.grid.origin_y_m
                        - 60.5 * sci_site.grid.pixel_size_m,
                        "height_m": 6.0,
                        "canopy_radius_m": 2.5,
                    },
                    requested_outputs=("utci",),
                    requested_times=(1,),
                )
            )

        def met(executor, *, edit_id):
            return executor.validate(
                EditCommand(
                    edit_id=edit_id,
                    scenario_id=executor._scenario_id,
                    base_scene_revision=executor.scene_revision,
                    adapter_id="meteorological_forcing",
                    operation="update_time_row",
                    old_state=None,
                    new_state={
                        "time_index": 1 if edit_id == "met-1" else 2,
                        "values": {
                            "air_temperature"
                            if edit_id == "met-1"
                            else "humidity": 30.0
                            if edit_id == "met-1"
                            else 45.0
                        },
                    },
                    requested_outputs=("utci",),
                    requested_times=(1,) if edit_id == "met-1" else (2,),
                )
            )

        first = ex.execute([met(ex, edit_id="met-1"), paint(ex), tree(ex)])
        assert first.published
        assert first.scene_revision == 1

        # Save; restart into a fresh executor (fresh store, fresh layer,
        # fresh worker — the state a new process would build).
        _save(ex, tmp_path / "state")
        twin = self._real_executor(sci_site, tmp_path)
        restore_into_executor(
            twin, read_snapshot(tmp_path / "state"), directory=tmp_path / "state"
        )
        _assert_state_equal(ex, twin)

        # The restart materializes the SAME forcing (accumulated overlay).
        assert np.array_equal(
            twin._worker.forcing().met_table, ex._worker.forcing().met_table
        )

        # Continue on both; the published products must match bitwise.
        continued = ex.execute([met(ex, edit_id="met-2")])
        resumed = twin.execute([met(twin, edit_id="met-2")])
        assert continued.published and resumed.published
        assert resumed.scene_revision == continued.scene_revision == 2
        assert np.array_equal(
            twin._worker.forcing().met_table, ex._worker.forcing().met_table
        )
        assert sorted(twin.store) == sorted(ex.store)
        for key in sorted(ex.store):
            left = load_patch(ex.store.lookup(*key).patch_path)
            right = load_patch(twin.store.lookup(*key).patch_path)
            assert sorted(left.arrays) == sorted(right.arrays)
            assert "utci" in left.arrays
            for name in sorted(left.arrays):
                # equal_nan (a NaN equals itself bitwise): the solver's
                # nodata cells are NaN on BOTH sides and must sit at the
                # identical positions — the repo's bitwise convention for
                # scientific arrays.
                assert np.array_equal(
                    left.arrays[name], right.arrays[name], equal_nan=True
                ), f"{name} differs at {key} after restart"
            assert left.scene_revision == right.scene_revision
            assert left.mode == right.mode


# ---------------------------------------------------------------------------
# u-d4c: the accumulated building fold is scenario state — it round-trips
# through the v3 snapshot and drives chain routing after a restart.
# ---------------------------------------------------------------------------


class TestBuildingFoldPersistence:
    """Building-edit accumulation across save/restore (u-d4c, schema v3)."""

    def _building_add(self, executor: PlanExecutor, state: dict, edit_id: str):
        from tests.test_incremental_building_integration import _building_command

        return executor.validate(
            _building_command(
                revision=executor.scene_revision, state=state, edit_id=edit_id
            )
        )

    def _building_delete(self, executor: PlanExecutor, state: dict, edit_id: str):
        from tests.test_incremental_building_integration import _delete_command

        return executor.validate(
            _delete_command(
                revision=executor.scene_revision, state=state, edit_id=edit_id
            )
        )

    def test_building_fold_round_trip_and_resume_routes_chain(
        self, tmp_path: Path
    ) -> None:
        """A published building edit survives the restart, and the restored
        executor routes the NEXT (other-family) batch through the chain
        with the fold — never back to the baseline-bound worker."""
        ex = _make_executor(tmp_path)
        assert ex.execute(
            [self._building_add(ex, NEW_BLOCK_STATE, "bld-a")]
        ).published
        _save(ex, tmp_path / "state")

        snapshot = read_snapshot(tmp_path / "state")
        assert snapshot.schema_version == 3
        assert [item["building_id"] for item in snapshot.massing_edits] == [
            "new-block"
        ]
        assert snapshot.massing_edits[0]["before"] is None
        assert snapshot.massing_edits[0]["after"]["height_m"] == 15.0

        twin = _restored_twin(ex, tmp_path)
        _assert_state_equal(ex, twin)

        outputs = []
        for executor, edit_id in ((ex, "met-src"), (twin, "met-twn")):
            executed = executor.execute(
                [
                    _executor_met_edit(
                        executor,
                        time_index=1,
                        value=26.0,
                        outputs=("utci", "tmrt", "shadow"),
                        edit_id=edit_id,
                    )
                ]
            )
            assert executed.published, "the resumed batch must publish"
            assert executed.mode == "full", (
                "a met batch after a published building edit must ride the "
                "chain (u-d4c), on BOTH the source and the restored twin"
            )
            assert executed.diagnostics["building_fold_size"] == 1
            assert executed.diagnostics["chain_carries_new_edits"] is False
            outputs.append(load_patch(executed.patch_paths[0]))
        left, right = outputs
        assert sorted(left.arrays) == sorted(right.arrays)
        for name in sorted(left.arrays):
            assert np.array_equal(
                left.arrays[name], right.arrays[name], equal_nan=True
            ), f"{name} differs between the source and the resumed scenario"

    def test_delete_tombstone_round_trips(self, tmp_path: Path) -> None:
        """add -> delete: the fold's tombstone survives the restart too
        (deletes stay deletes across processes).

        u-d4c remediation F5: the delete batch is the one publication
        made with ONLY a tombstone in the fold — its products must equal
        BASELINE-equal products bitwise (an independent chain whose net
        massing is the empty edit: add A then delete A stages a scene
        numerically identical to baseline) while carrying SCENARIO (not
        baseline) provenance, closing the docstring's claim."""
        ex = _make_executor(tmp_path)
        add_executed = ex.execute(
            [self._building_add(ex, NEW_BLOCK_STATE, "bld-a")]
        )
        assert add_executed.published
        delete_executed = ex.execute(
            [self._building_delete(ex, NEW_BLOCK_STATE, "bld-del")]
        )
        assert delete_executed.published
        _save(ex, tmp_path / "state")

        snapshot = read_snapshot(tmp_path / "state")
        assert [item["building_id"] for item in snapshot.massing_edits] == [
            "new-block"
        ]
        assert snapshot.massing_edits[0]["after"] is None
        assert snapshot.massing_edits[0]["before"] is not None

        twin = _restored_twin(ex, tmp_path)
        _assert_state_equal(ex, twin)
        tombstone = twin._applied_massing_edits["new-block"]
        assert tombstone.after is None
        assert tombstone.before is not None

        # --- F5: publish once with ONLY the tombstone, prove the claim.
        from solweig_gpu.incremental.regenerate import (
            _read_tile,
            regenerate_building_batch,
        )
        from tests.test_incremental_building_integration import _massing_edit

        # The only fold entry IS the tombstone; the batch rode the chain.
        assert list(ex._applied_massing_edits) == ["new-block"]
        assert ex._applied_massing_edits["new-block"].after is None
        assert delete_executed.mode == "full"

        # Baseline-equal oracle: the SAME chain over a scene whose net
        # session massing is EMPTY (add A then delete A stages the
        # baseline tiles back: the unconditional before-footprint reset
        # restores DEM ground where the add painted).
        baseline_equal = regenerate_building_batch(
            edits=(
                _massing_edit(NEW_BLOCK_STATE),
                _massing_edit(NEW_BLOCK_STATE, deleted=True),
            ),
            baseline_site_dir=ex.site_dir,
            baseline_cache=ex.cache,
            scenario_root=tmp_path / "f5_twin" / "scenario",
            selected_date_str=DATE_STR,
        )
        add_patch = load_patch(add_executed.patch_paths[0])
        delete_patch = load_patch(delete_executed.patch_paths[0])
        for name in sorted(delete_patch.arrays):
            assert np.array_equal(
                delete_patch.arrays[name],
                baseline_equal.outputs[name],
                equal_nan=True,
            ), (
                f"{name}: a tombstone-only fold must publish baseline-"
                "equal products (the docstring's claim)"
            )
            # Non-vacuous: the ADD batch (building present) differs from
            # the tombstone-only products — the equality above is the
            # building being removed again, not products that never move.
            assert not np.array_equal(
                delete_patch.arrays[name], add_patch.arrays[name],
                equal_nan=True,
            ), f"{name}: the add/delete products are identical; vacuous"

        # Raster seam: the staged scenario Building_DSM tile is bitwise
        # the baseline tile (the tombstone's reset nets to baseline).
        staged_site = ex.results_root / "default" / ".building" / "site"
        staged_dsm, *_ = _read_tile(
            staged_site, "Building_DSM", ex.cache.tile_key
        )
        baseline_dsm, *_ = _read_tile(
            ex.site_dir, "Building_DSM", ex.cache.tile_key
        )
        assert np.array_equal(staged_dsm, baseline_dsm)

        # SCENARIO provenance, never a baseline-cache product: the patch
        # carries the rebuilt scenario cache's manifest, at the delete
        # batch's new revision.
        assert delete_patch.mode == "full"
        assert delete_executed.scene_revision == 2
        assert (
            delete_patch.cache_manifest_sha256
            != ex.cache.metadata()["manifest_sha256"]
        )
        for time_index in range(ex.cache.time_steps):
            assert ex.store.revision_at("utci", time_index) == 2
            assert ex.store.lookup("utci", time_index).mode == "full"

    def test_v2_document_is_refused(self, tmp_path: Path) -> None:
        """A pre-u-d4c (v2) document cannot prove the session carried no
        published buildings — it is refused, never restored unverifiable
        (the same discipline that refuses v1)."""
        ex = _make_executor(tmp_path)
        assert ex.execute([_veg_edit(ex)]).published
        _save(ex, tmp_path / "state")
        path = tmp_path / "state" / "scenario-state.json"
        document = json.loads(path.read_text())
        assert document["schema_version"] == 3
        # Forge the strongest possible v2 document: version 2 AND no
        # massing_edits key (exactly what a pre-u-d4c writer produced).
        document["schema_version"] = 2
        document.pop("massing_edits", None)
        path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
        with pytest.raises(ScenarioStateError, match="not the supported version"):
            read_snapshot(tmp_path / "state")

    def test_deterministic_rebuild_reaches_the_same_fold(
        self, tmp_path: Path
    ) -> None:
        """executor_bridge's fresh-rebuild path (no snapshot: first family
        job, or a reset invalidated it) replays the ledger's building
        events as commands and must reach the SAME accumulated fold as the
        snapshot-restored twin."""
        ex = _make_executor(tmp_path)
        assert ex.execute(
            [self._building_add(ex, NEW_BLOCK_STATE, "bld-a")]
        ).published
        assert ex.execute(
            [self._building_add(ex, NEW_BLOCK_B_STATE, "bld-b")]
        ).published
        _save(ex, tmp_path / "state")

        rebuilt = _make_executor(tmp_path, results_root=tmp_path / "results_twin")
        assert rebuilt.execute(
            [self._building_add(rebuilt, NEW_BLOCK_STATE, "bld-a")]
        ).published
        assert rebuilt.execute(
            [self._building_add(rebuilt, NEW_BLOCK_B_STATE, "bld-b")]
        ).published

        assert list(rebuilt._applied_massing_edits) == list(
            ex._applied_massing_edits
        )
        assert rebuilt._applied_massing_edits == ex._applied_massing_edits
        assert (
            snapshot_from_executor(rebuilt).massing_edits
            == read_snapshot(tmp_path / "state").massing_edits
        )
