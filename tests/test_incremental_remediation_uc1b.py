# SPDX-License-Identifier: GPL-3.0-only
"""U-C1/U-C3 review remediation tests (agent u-c1b).

Every fix lands with the probe that pins it:

- **M1 (the publication wedge)** — a post-publish EXECUTOR failure (the
  store rejecting the batch) used to leave the worker's publication
  watermarks advanced while the executor rolled its own state back, so
  every retry found "nothing pending" forever. Each watermark class
  (``_acked_sequence`` for vegetation, ``_published_forcing_overlay`` for
  met, ``_published_landcover_overlay`` for paint) is probed
  before/after: with the rollback hook disabled the retry wedges; with
  it active the same failure recovers and publishes.
- **M2 (multi-window store coverage)** — a multi-window local job
  publishes one patch per write window; the store records one entry per
  (variable, timestep, window) carrying its own patch path, so coverage
  spans every disjoint footprint and a cell resolves to the patch that
  actually covers it.
- **M3 (single writer)** — a store bound to one executor identity refuses
  a second identity at bind time AND at publish time
  (:class:`StoreWriterError`); unbound stores stay anonymous (legacy).
- **M4 (orphan patches)** — a failed batch's already-renamed ``rev-*``
  directories are removed best-effort by the executor rollback.
- **L5** — ``validate_tree_spec`` runs in the executor's pre-flight, so a
  malformed spec (``trunk_ratio=1.0``) is refused BEFORE any layer
  mutation.
- **L6** — store publication is lock + snapshot-swap: a concurrent reader
  sees the old or the new state, never a partial batch.
- **L7** — a same-revision same-job republish with different content is a
  loud :class:`StoreError`, never a silent overwrite.
- **L8** — :func:`load_patch_metadata` indexes a patch without reading or
  hashing its arrays; full verification stays in :func:`load_patch`.
- **L9** — the executor-level superseded branch rolls state, layer, and
  worker watermarks back and publishes nothing.
- **T10** — a failure inside the rollback handler (tree undo breaking)
  rides as the ORIGINAL error's ``__cause__`` and never masks it.
- **u-c3 L2** — forcing folds in arrival order across the
  accumulated/batch boundary: a NEWER whole-series batch edit overrides
  an OLDER accumulated per-step value ([25, 25, 25], not [25, 26, 25]).
- **u-c3 L3** — ``regenerate`` threads ``forcing_overlay`` through to the
  solver's forcing materialization (default ``None`` stays legacy).
- **u-c3 L5** — the resolved-met header states the column count it
  actually wrote; the text round-trips bitwise.
"""

from __future__ import annotations

import inspect
import re
import threading
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from solweig_gpu.incremental.adapters.met_time import ForcingOverlay
from solweig_gpu.incremental.edit_types import (
    EditCommand,
    ObjectStateChange,
    ValidatedEdit,
    VegetationObjectDelta,
    ForcingChange,
    ForcingDelta,
)
from solweig_gpu.incremental.executor import ExecutorError, PlanExecutor
from solweig_gpu.incremental.geometry import RasterWindow
from solweig_gpu.incremental import solver as solver_mod
from solweig_gpu.incremental.regenerate import (
    regenerate_building_batch,
    run_scenario_full_tile,
)
from solweig_gpu.incremental.result import (
    PatchChecksumError,
    PatchError,
    ResultPatch,
    load_patch,
    load_patch_metadata,
    publish_staged_patch,
    stage_patch,
)
from solweig_gpu.incremental.store import (
    StoreError,
    StoreWriterError,
    TemporalResultEntry,
    TemporalResultStore,
)
from solweig_gpu.incremental.trees import TreeLayer, TreeSpec
from solweig_gpu.incremental.worker import (
    JobOutcome,
    discard_published_patches,
)

from tests.test_incremental_adapters_met_time import ADAPTER_ID as MET_ADAPTER_ID
from tests.test_incremental_executor import _veg_command
from tests.test_incremental_lc_integration import (
    PAINT_WINDOW,
    PAINT_WINDOW_B,
    _covers,
    _paint_command,
)
from tests.test_incremental_met_integration import _executor_met_edit
from tests.test_incremental_worker import (
    DATE_STR,
    LOCAL_ALWAYS,
    TINY_ORIGIN,
    TINY_PIXEL,
    _build_cache,
    _make_tiny_site,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _tiny_executor(tmp_path: Path) -> PlanExecutor:
    grid, site = _make_tiny_site(tmp_path / "site")
    cache = _build_cache(
        site,
        tmp_path / "cache",
        met_path=site / "metfiles" / f"metfile_0_0_{DATE_STR}.txt",
        site_id="tiny",
    )
    return PlanExecutor(
        cache=cache,
        layer=TreeLayer(cache.tree_base, grid),
        site_dir=site,
        results_root=tmp_path / "results",
        selected_date_str=DATE_STR,
        influence_config=LOCAL_ALWAYS,
    )


def _stub_patch(
    variables: tuple[str, ...],
    job_id: str,
    revision: int,
    window: RasterWindow,
    *,
    mode: str = "local",
) -> ResultPatch:
    arrays = {
        name: np.full((3, window.height, window.width), 1.0, dtype=np.float32)
        for name in variables
    }
    return ResultPatch(
        job_id=job_id,
        scene_revision=revision,
        mode=mode,
        write_window=window,
        read_window=window,
        site_id="tiny",
        tile_key="0_0",
        cache_manifest_sha256="0" * 64,
        model_version="test",
        variables=variables,
        arrays=arrays,
        time_start=0,
        time_stop=3,
    )


def _fake_local_patches(job_id: str, revision: int, write_windows) -> list[ResultPatch]:
    """The worker's local contract: one patch per write window, each with
    its own store id (mirrors the multi-window job ids ``job-wN``)."""
    return [
        _stub_patch(("utci",), f"{job_id}-w{index}", revision, window, mode="local")
        for index, window in enumerate(write_windows)
    ]


def _stub_local_every_window(executor: PlanExecutor) -> None:
    executor._worker._solve_local = (
        lambda job_id, revision, forcing, write_windows, **kwargs: (
            _fake_local_patches(job_id, revision, write_windows)
        )
    )


def _stub_full(executor: PlanExecutor) -> None:
    executor._worker._solve_full = (
        lambda job_id, revision, forcing: [
            _stub_patch(
                ("utci",), job_id, revision, executor.grid.full_window, mode="full"
            )
        ]
    )


def _veg_edit(executor: PlanExecutor, *, edit_id: str = "veg-1"):
    return executor.validate(_veg_command(edit_id=edit_id))


class _PostPublishFailure(RuntimeError):
    """Injected at the executor's store publish — AFTER the worker published."""


def _fail_store_publish(monkeypatch, executor: PlanExecutor) -> None:
    def exploding(entries, **kwargs):
        raise _PostPublishFailure(
            "store rejected the batch after the worker published"
        )

    monkeypatch.setattr(executor.store, "publish", exploding)


def _no_rev_dirs(results_root: Path) -> bool:
    return not list(Path(results_root).rglob("rev-*"))


def _store_entry(
    node: str,
    time: int,
    revision: int,
    *,
    job: str = "job-1",
    window: RasterWindow | None = None,
    path: Path | None = None,
) -> TemporalResultEntry:
    return TemporalResultEntry(
        node_id=node,
        time_index=time,
        scene_revision=revision,
        job_id=job,
        mode="local",
        write_window=window if window is not None else RasterWindow(0, 4, 0, 4),
        patch_path=path,
    )


# ---------------------------------------------------------------------------
# M1: the publication wedge (one probe per watermark class)
# ---------------------------------------------------------------------------


class TestM1PublicationWedge:
    def test_vegetation_watermark_rollback_recovers(self, tmp_path, monkeypatch):
        """``_acked_sequence``: before-fails / after-recovers, in one probe."""
        # --- BEFORE: without the worker rollback hook the retry wedges.
        wedged = _tiny_executor(tmp_path / "wedge")
        _stub_local_every_window(wedged)
        original_hook = wedged._worker.rollback_pending
        wedged._worker.rollback_pending = lambda watermarks: None
        _fail_store_publish(monkeypatch, wedged)
        with pytest.raises(_PostPublishFailure):
            wedged.execute([_veg_edit(wedged)])
        monkeypatch.undo()
        with pytest.raises(ExecutorError, match="nothing pending"):
            wedged.execute([_veg_edit(wedged)])
        # The wedge is permanent for this executor (nothing rewound it).
        with pytest.raises(ExecutorError, match="nothing pending"):
            wedged.execute([_veg_edit(wedged)])
        wedged._worker.rollback_pending = original_hook

        # --- AFTER: the same failure with the hook active recovers.
        ex = _tiny_executor(tmp_path / "fix")
        _stub_local_every_window(ex)
        watermarks_before = ex._worker.publication_watermarks
        assert watermarks_before.acked_sequence == 0
        _fail_store_publish(monkeypatch, ex)
        with pytest.raises(_PostPublishFailure):
            ex.execute([_veg_edit(ex)])
        monkeypatch.undo()
        # Rollback evidence: every commit point rewound together.
        assert ex.scene_revision == 0
        assert ex._worker.publication_watermarks == watermarks_before
        assert ex._worker.pending_batch() is not None  # edits pending again
        assert ex.layer.current_trees() == ()  # layer replay undone
        assert _no_rev_dirs(tmp_path / "fix" / "results")  # M4
        # Retry re-executes and publishes (the wedge is gone).
        executed = ex.execute([_veg_edit(ex)])
        assert executed.published
        assert ex.store.revision_at("utci", 1) == 1

    def test_forcing_watermark_rollback_recovers(self, tmp_path, monkeypatch):
        """``_published_forcing_overlay``: before-fails / after-recovers."""
        wedged = _tiny_executor(tmp_path / "wedge")
        _stub_full(wedged)
        wedged._worker.rollback_pending = lambda watermarks: None
        _fail_store_publish(monkeypatch, wedged)
        with pytest.raises(_PostPublishFailure):
            wedged.execute(
                [_executor_met_edit(wedged, time_index=1, value=26.0)]
            )
        monkeypatch.undo()
        with pytest.raises(ExecutorError, match="nothing pending"):
            wedged.execute(
                [_executor_met_edit(wedged, time_index=1, value=26.0)]
            )

        ex = _tiny_executor(tmp_path / "fix")
        _stub_full(ex)
        watermarks_before = ex._worker.publication_watermarks
        assert watermarks_before.forcing_overlay is None
        _fail_store_publish(monkeypatch, ex)
        with pytest.raises(_PostPublishFailure):
            ex.execute([_executor_met_edit(ex, time_index=1, value=26.0)])
        monkeypatch.undo()
        assert ex.scene_revision == 0
        # The forcing watermark rewound: the staged overlay is pending
        # again instead of being seen as already published.
        assert ex._worker.publication_watermarks.forcing_overlay is None
        assert ex._worker._forcing_overlay is not None
        assert (
            ex._worker._forcing_overlay
            != ex._worker._published_forcing_overlay
        )
        assert ex._applied_forcing_overlay is None  # never committed
        assert _no_rev_dirs(tmp_path / "fix" / "results")
        executed = ex.execute([_executor_met_edit(ex, time_index=1, value=26.0)])
        assert executed.published
        assert ex.store.revision_at("utci", 1) == 1

    def test_landcover_watermark_rollback_recovers(self, tmp_path, monkeypatch):
        """``_published_landcover_overlay``: before-fails / after-recovers."""
        wedged = _tiny_executor(tmp_path / "wedge")
        _stub_local_every_window(wedged)
        wedged._worker.rollback_pending = lambda watermarks: None
        _fail_store_publish(monkeypatch, wedged)
        with pytest.raises(_PostPublishFailure):
            wedged.execute([wedged.validate(_paint_command())])
        monkeypatch.undo()
        with pytest.raises(ExecutorError, match="nothing pending"):
            wedged.execute([wedged.validate(_paint_command())])

        ex = _tiny_executor(tmp_path / "fix")
        _stub_local_every_window(ex)
        watermarks_before = ex._worker.publication_watermarks
        assert watermarks_before.landcover_overlay is None
        _fail_store_publish(monkeypatch, ex)
        with pytest.raises(_PostPublishFailure):
            ex.execute([ex.validate(_paint_command())])
        monkeypatch.undo()
        assert ex.scene_revision == 0
        assert ex._worker.publication_watermarks.landcover_overlay is None
        assert (
            ex._worker._landcover_overlay
            != ex._worker._published_landcover_overlay
        )
        assert ex._applied_landcover_overlay is None
        assert _no_rev_dirs(tmp_path / "fix" / "results")
        executed = ex.execute([ex.validate(_paint_command())])
        assert executed.published
        assert ex.store.revision_at("utci", 1) == 1


# ---------------------------------------------------------------------------
# M2: one store entry per patch window
# ---------------------------------------------------------------------------


class TestM2MultiWindowStoreCoverage:
    def test_two_disjoint_paint_windows_record_per_patch_entries(
        self, tmp_path
    ) -> None:
        ex = _tiny_executor(tmp_path)
        captured: dict = {}

        def fake_local(job_id, revision, forcing, write_windows, **kwargs):
            captured["windows"] = tuple(write_windows)
            return _fake_local_patches(job_id, revision, write_windows)

        ex._worker._solve_local = fake_local
        paint_a = ex.validate(_paint_command(edit_id="lc-a"))
        paint_b = ex.validate(
            _paint_command(edit_id="lc-b", window=PAINT_WINDOW_B)
        )
        executed = ex.execute([paint_a, paint_b])

        assert executed.published
        assert executed.mode == "local"
        # The two footprints are disjoint: one local job, TWO patches.
        assert len(captured["windows"]) == 2
        assert len(executed.patch_paths) == 2

        # One store entry per (variable, timestep, window) — each carrying
        # its own patch path and its own per-window job id (job-wN), never
        # a merged union window pointing at the first patch.
        entries = ex.store.window_entries("utci", 1)
        assert len(entries) == 2
        assert {e.write_window for e in entries} == set(captured["windows"])
        assert {e.job_id for e in entries} == {
            f"{executed.job_id}-w0",
            f"{executed.job_id}-w1",
        }
        # Coverage spans BOTH disjoint footprints (no phantom gaps).
        assert ex.store.coverage_gaps("utci", (0, 1, 2)) == ()
        assert ex.store.available_times("utci") == (0, 1, 2)

        # A cell resolves to the patch that actually covers its region.
        a_entry = ex.store.lookup_window("utci", 1, row=44, col=44)
        b_entry = ex.store.lookup_window("utci", 1, row=113, col=106)
        assert a_entry is not None and b_entry is not None
        assert a_entry.patch_path != b_entry.patch_path
        meta_a = load_patch_metadata(a_entry.patch_path)
        meta_b = load_patch_metadata(b_entry.patch_path)
        assert _covers(meta_a.write_window, PAINT_WINDOW)
        assert _covers(meta_b.write_window, PAINT_WINDOW_B)
        assert not _covers(meta_a.write_window, PAINT_WINDOW_B)
        # The uncovered between-region cell maps to no entry.
        assert ex.store.lookup_window("utci", 1, row=80, col=80) is None

    def test_lookup_window_resolves_region_to_its_patch(self) -> None:
        store = TemporalResultStore()
        west = RasterWindow(0, 10, 0, 10)
        east = RasterWindow(20, 30, 20, 30)
        store.publish(
            [
                _store_entry("utci", 0, 1, job="job-w0", window=west, path=Path("/p/w0")),
                _store_entry("utci", 0, 1, job="job-w1", window=east, path=Path("/p/w1")),
            ]
        )
        assert len(store.window_entries("utci", 0)) == 2
        assert store.lookup_window("utci", 0, row=5, col=5).patch_path == Path("/p/w0")
        assert store.lookup_window("utci", 0, row=25, col=25).patch_path == Path("/p/w1")
        assert store.lookup_window("utci", 0, row=15, col=15) is None


# ---------------------------------------------------------------------------
# M3: one store, one writer (loud refusal)
# ---------------------------------------------------------------------------


class TestM3SingleWriter:
    def test_second_executor_identity_refused_at_construction(
        self, tmp_path
    ) -> None:
        ex = _tiny_executor(tmp_path)
        assert ex.store.writer is not None
        twin_layer = TreeLayer(
            ex.cache.tree_base, ex.grid, scenario_id="intruder"
        )
        with pytest.raises(StoreWriterError, match="One store has one writer"):
            PlanExecutor(
                cache=ex.cache,
                layer=twin_layer,
                site_dir=ex.site_dir,
                results_root=tmp_path / "results2",
                selected_date_str=DATE_STR,
                influence_config=LOCAL_ALWAYS,
                store=ex.store,
            )

    def test_second_writer_refused_at_bind_and_publish(self, tmp_path) -> None:
        ex = _tiny_executor(tmp_path)
        with pytest.raises(StoreWriterError):
            ex.store.bind_writer("executor:someone:else")
        with pytest.raises(StoreWriterError):
            ex.store.publish(
                [_store_entry("utci", 0, 1)], writer="executor:someone:else"
            )
        # An attributed publish from the BOUND identity still works, and
        # the identical re-bind is idempotent.
        assert (
            ex.store.bind_writer(ex._store_writer_id) == ex._store_writer_id
        )
        keys = ex.store.publish(
            [_store_entry("utci", 0, 1)], writer=ex._store_writer_id
        )
        assert keys == (("utci", 0),)

    def test_unbound_store_stays_anonymous_legacy(self) -> None:
        store = TemporalResultStore()
        assert store.writer is None
        store.publish([_store_entry("utci", 0, 1)])  # writer=None: legacy
        assert store.writer is None
        # Once attributed, the store is bound — an anonymous publish (the
        # shape a second, unmodified executor would produce) is refused.
        store.publish([_store_entry("utci", 1, 1)], writer="w1")
        assert store.writer == "w1"
        with pytest.raises(StoreWriterError):
            store.publish([_store_entry("utci", 2, 1)])


# ---------------------------------------------------------------------------
# M4: orphan patch directories
# ---------------------------------------------------------------------------


class TestM4OrphanPatchCleanup:
    def test_discard_published_patches_removes_dirs_and_reports_leftovers(
        self, tmp_path
    ) -> None:
        a = tmp_path / "rev-000001-job-a"
        b = tmp_path / "rev-000001-job-b"
        for directory in (a, b):
            directory.mkdir()
            (directory / "patch.json").write_text("{}")
        leftovers = discard_published_patches(
            (a, b, tmp_path / "rev-000002-missing")
        )
        assert leftovers == ()
        assert not a.exists()
        assert not b.exists()

    def test_failed_batch_leaves_no_rev_dirs(self, tmp_path, monkeypatch) -> None:
        # Full evidence lives in the M1 probes; this pins the executor's
        # post-publish failure cleanup with an explicit staging check.
        ex = _tiny_executor(tmp_path)
        _stub_local_every_window(ex)
        _fail_store_publish(monkeypatch, ex)
        with pytest.raises(_PostPublishFailure):
            ex.execute([_veg_edit(ex)])
        monkeypatch.undo()
        results = tmp_path / "results"
        assert _no_rev_dirs(results)
        assert not list(results.rglob("*.tmp"))  # staging discarded too


# ---------------------------------------------------------------------------
# L5: pre-flight tree-spec validation
# ---------------------------------------------------------------------------


def _hand_built_veg_delta(tree_state: dict, tree_id: str = "t1") -> ValidatedEdit:
    """A ValidatedEdit whose delta BYPASSES adapter validation.

    The review's L5 probe: the executor's pre-flight must refuse a
    malformed spec even when it arrives through the transaction seam
    rather than ``executor.validate``.
    """
    return ValidatedEdit(
        command=EditCommand(
            edit_id="veg-bad",
            scenario_id="default",
            base_scene_revision=0,
            adapter_id="vegetation_geometry",
            operation="add",
            old_state=None,
            new_state=dict(tree_state),
            requested_outputs=("utci",),
            requested_times=(1,),
        ),
        adapter_id="vegetation_geometry",
        schema_version=1,
        source_node_id="vegetation_dsm",
        delta=VegetationObjectDelta(
            source_node_id="vegetation_dsm",
            adapter_id="vegetation_geometry",
            objects=(ObjectStateChange(tree_id, None, dict(tree_state)),),
            windows=(RasterWindow(0, 8, 0, 8),),
        ),
    )


class TestL5PreflightSpecValidation:
    def test_trunk_ratio_one_refused_before_any_layer_mutation(
        self, tmp_path
    ) -> None:
        ex = _tiny_executor(tmp_path)
        _stub_local_every_window(ex)
        bad = _hand_built_veg_delta(
            {
                "tree_id": "t1",
                "x_m": TINY_ORIGIN[0] + 40.5 * TINY_PIXEL,
                "y_m": TINY_ORIGIN[1] - 40.5 * TINY_PIXEL,
                "height_m": 4.0,
                "canopy_radius_m": 2.0,
                "trunk_ratio": 1.0,  # structurally invalid (must be < 1)
            }
        )
        with pytest.raises(ExecutorError, match="invalid new_tree spec") as info:
            ex.execute([bad])
        assert isinstance(info.value.__cause__, ValueError)
        assert "trunk_ratio" in str(info.value.__cause__)
        # Refused BEFORE any mutation: state, layer, worker, and store.
        assert ex.scene_revision == 0
        assert ex.layer.current_trees() == ()
        assert ex._worker.pending_batch() is None
        assert len(ex.store) == 0
        assert _no_rev_dirs(tmp_path / "results")

    def test_update_path_refused_and_layer_unchanged(self, tmp_path) -> None:
        ex = _tiny_executor(tmp_path)
        _stub_local_every_window(ex)
        good = TreeSpec(
            "t1",
            TINY_ORIGIN[0] + 40.5 * TINY_PIXEL,
            TINY_ORIGIN[1] - 40.5 * TINY_PIXEL,
            4.0,
            2.0,
            trunk_ratio=0.2,
        )
        ex.layer.add_tree(good)
        before = {
            "x_m": good.x_m,
            "y_m": good.y_m,
            "height_m": 4.0,
            "canopy_radius_m": 2.0,
            "trunk_ratio": 0.2,
        }
        after = dict(before, trunk_ratio=1.0)  # structurally invalid update
        bad = replace(
            _hand_built_veg_delta(before),
            delta=VegetationObjectDelta(
                source_node_id="vegetation_dsm",
                adapter_id="vegetation_geometry",
                objects=(ObjectStateChange("t1", before, after),),
                windows=(RasterWindow(0, 8, 0, 8),),
            ),
        )
        with pytest.raises(ExecutorError, match="invalid"):
            ex.execute([bad])
        trees = ex.layer.current_trees()
        assert len(trees) == 1
        assert trees[0].trunk_ratio == 0.2  # original spec intact
        assert ex.scene_revision == 0


# ---------------------------------------------------------------------------
# L6: atomic publication under concurrency
# ---------------------------------------------------------------------------


class TestL6AtomicPublishSnapshot:
    def test_concurrent_reader_never_sees_partial_batch(self) -> None:
        store = TemporalResultStore()
        store.publish([_store_entry("utci", t, 1) for t in range(40)])
        observations: list[tuple[int, bool]] = []
        stop = threading.Event()

        def reader() -> None:
            while not stop.is_set():
                snapshot = store.as_mapping()
                observations.append(
                    (
                        len(snapshot),
                        all(
                            entry.scene_revision == 1 and entry.node_id == "utci"
                            for entry in snapshot.values()
                        ),
                    )
                )

        thread = threading.Thread(target=reader)
        thread.start()
        try:
            for extra in range(10):
                store.publish([_store_entry("utci", 40 + extra, 1)])
        finally:
            stop.set()
            thread.join()
        lengths = {length for length, _ in observations}
        # Old count, new count, or anything complete in between — never a
        # partial batch (which would show up as a torn mapping).
        assert lengths <= set(range(40, 51))
        assert all(consistent for _, consistent in observations)
        assert len(store) == 50


# ---------------------------------------------------------------------------
# L7: same-revision same-job republish discipline
# ---------------------------------------------------------------------------


class TestL7RepublishContentConflict:
    def test_different_patch_path_is_a_loud_error(self) -> None:
        store = TemporalResultStore()
        window = RasterWindow(0, 4, 0, 4)
        store.publish(
            [_store_entry("utci", 0, 2, job="a", window=window, path=Path("/p/first"))]
        )
        with pytest.raises(StoreError, match="different content"):
            store.publish(
                [
                    _store_entry(
                        "utci", 0, 2, job="a", window=window, path=Path("/p/other")
                    )
                ]
            )
        assert store.lookup("utci", 0).patch_path == Path("/p/first")

    def test_different_window_set_is_a_loud_error(self) -> None:
        store = TemporalResultStore()
        store.publish(
            [_store_entry("utci", 0, 2, job="a", window=RasterWindow(0, 4, 0, 4))]
        )
        with pytest.raises(StoreError, match="different content"):
            store.publish(
                [
                    _store_entry(
                        "utci", 0, 2, job="a", window=RasterWindow(4, 8, 0, 4)
                    )
                ]
            )

    def test_identical_replay_stays_idempotent(self) -> None:
        store = TemporalResultStore()
        entry = _store_entry("utci", 0, 2, job="a", path=Path("/p/same"))
        keys = store.publish([entry])
        keys_again = store.publish([entry])
        assert keys_again == keys == (("utci", 0),)
        assert len(store) == 1


# ---------------------------------------------------------------------------
# L8: metadata-only patch loading
# ---------------------------------------------------------------------------


class TestL8PatchMetadata:
    def _published(self, tmp_path: Path, *, time_stop: int | None = 3):
        patch = _stub_patch(("utci", "tmrt"), "job-x", 3, RasterWindow(4, 12, 8, 16))
        patch = replace(patch, time_stop=time_stop)
        staging = stage_patch(patch, tmp_path / "staging")
        (tmp_path / "results").mkdir(parents=True, exist_ok=True)
        return publish_staged_patch(staging, tmp_path / "results", 3)

    def test_metadata_round_trip_without_array_verification(self, tmp_path) -> None:
        published = self._published(tmp_path)
        meta = load_patch_metadata(published)
        assert meta.job_id == "job-x"
        assert meta.scene_revision == 3
        assert meta.mode == "local"
        assert meta.write_window == RasterWindow(4, 12, 8, 16)
        assert meta.variables == ("utci", "tmrt")
        assert meta.time_start == 0 and meta.time_stop == 3
        assert meta.n_time_steps == 3
        assert set(meta.checksums) == {"utci", "tmrt"}
        # No arrays were read: tampering the bytes does not affect the
        # metadata view, while the full load still refuses (checksums).
        array_file = published / "variables" / "utci.f32.npy"
        data = bytearray(array_file.read_bytes())
        data[-1] ^= 0xFF
        array_file.write_bytes(bytes(data))
        assert load_patch_metadata(published).job_id == "job-x"
        with pytest.raises(PatchChecksumError):
            load_patch(published)

    def test_n_time_steps_requires_recorded_time_stop(self, tmp_path) -> None:
        published = self._published(tmp_path, time_stop=None)
        meta = load_patch_metadata(published)
        assert meta.time_stop is None
        with pytest.raises(PatchError, match="time_stop"):
            meta.n_time_steps


# ---------------------------------------------------------------------------
# L9: executor-level superseded branch
# ---------------------------------------------------------------------------


class TestL9ExecutorSuperseded:
    def test_superseded_outcome_rolls_back_everything(self, tmp_path) -> None:
        ex = _tiny_executor(tmp_path)
        watermarks_before = ex._worker.publication_watermarks

        def superseded_worker(*, target_revision: int, force_full: bool):
            return JobOutcome(
                status="superseded",
                mode=None,
                job_id="job-superseded",
                scene_revision=target_revision,
                diagnostics={"reason": "the scene revision moved on"},
            )

        ex._run_worker = superseded_worker
        executed = ex.execute([_veg_edit(ex)])
        assert executed.status == "superseded"
        assert not executed.published
        assert executed.job_id == "job-superseded"
        # State, layer, and worker watermarks all rewound; nothing on disk.
        assert ex.scene_revision == 0
        assert ex.layer.current_trees() == ()
        assert ex._worker.publication_watermarks == watermarks_before
        assert len(ex.store) == 0
        assert _no_rev_dirs(tmp_path / "results")
        # The executor stays usable: the retry at the live revision runs.
        del ex._run_worker
        _stub_local_every_window(ex)
        retried = ex.execute([_veg_edit(ex)])
        assert retried.published


# ---------------------------------------------------------------------------
# T10: rollback failures never mask the original error
# ---------------------------------------------------------------------------


class TestT10UndoFailureChaining:
    def test_undo_failure_rides_as_cause_original_propagates(
        self, tmp_path, monkeypatch
    ) -> None:
        ex = _tiny_executor(tmp_path)
        _stub_local_every_window(ex)
        watermarks_before = ex._worker.publication_watermarks
        _fail_store_publish(monkeypatch, ex)

        def broken_undo(edits):
            raise ValueError("tree undo broke mid-rollback")

        ex._undo_tree_edits = broken_undo
        with pytest.raises(_PostPublishFailure) as info:
            ex.execute([_veg_edit(ex)])
        # The ORIGINAL error is the actionable one that propagates.
        assert isinstance(info.value.__cause__, ValueError)
        assert "tree undo broke" in str(info.value.__cause__)
        # Every other rollback step still ran despite the undo failure.
        assert ex.scene_revision == 0
        assert ex._worker.publication_watermarks == watermarks_before
        assert _no_rev_dirs(tmp_path / "results")
        # The layer keeps the replayed edit — the documented residue of a
        # failed undo (reported loudly, never silently swallowed).
        assert [tree.tree_id for tree in ex.layer.current_trees()] == ["t1"]


# ---------------------------------------------------------------------------
# u-c3 L2: forcing fold priority (arrival order across batches)
# ---------------------------------------------------------------------------


class TestUc3L2ForcingFoldPriority:
    def test_newer_whole_series_overrides_older_per_step(self, tmp_path) -> None:
        ex = _tiny_executor(tmp_path)
        captured: dict = {}

        def fake_full(job_id, revision, forcing):
            captured["table"] = forcing.met_table
            return [
                _stub_patch(
                    ("utci",), job_id, revision, ex.grid.full_window, mode="full"
                )
            ]

        ex._worker._solve_full = fake_full

        # Batch 1 (adapter-validated): per-step ta[1] = 26 published.
        first = ex.execute([_executor_met_edit(ex, time_index=1, value=26.0)])
        assert first.published
        assert captured["table"][1, 11] == 26.0

        # Batch 2: a WHOLE-series ta = 25 edit. Adapter v1 implements no
        # whole-series operation, so this is the hand-built record the
        # seam re-validates (overlay_from_deltas + ForcingOverlay fence).
        validated = _executor_met_edit(ex, time_index=1, value=25.0)
        whole = replace(
            validated,
            delta=ForcingDelta(
                source_node_id="meteorology",
                adapter_id=MET_ADAPTER_ID,
                changes=(
                    ForcingChange("air_temperature", None, 26.0, 25.0),
                ),
            ),
        )
        second = ex.execute([whole])
        assert second.published
        # Arrival order: the newer whole-series batch edit overrides the
        # OLDER accumulated per-step value — [25, 25, 25], never the
        # priority-inverted [25, 26, 25].
        assert captured["table"][:, 11].tolist() == [25.0, 25.0, 25.0]


# ---------------------------------------------------------------------------
# u-c3 L3: regenerate threads forcing_overlay (compile-time plumbing)
# ---------------------------------------------------------------------------


class TestUc3L3RegenerateOverlayPlumbing:
    def test_run_scenario_full_tile_threads_overlay(self, tmp_path, monkeypatch):
        grid, site = _make_tiny_site(tmp_path / "site")
        cache = _build_cache(
            site,
            tmp_path / "cache",
            met_path=site / "metfiles" / f"metfile_0_0_{DATE_STR}.txt",
            site_id="tiny",
        )
        layer = TreeLayer(cache.tree_base, grid)
        captured: dict = {}
        sentinel_forcing = object()

        def fake_load_site_forcing(cache_, **kwargs):
            captured["overlay"] = kwargs.get("overlay")
            return sentinel_forcing

        def fake_run_full_tile(cache_, layer_, *, forcing, **kwargs):
            captured["forcing"] = forcing
            return {"utci": None}

        # run_scenario_full_tile imports both names from .solver at CALL
        # time, so patching the module attributes is sufficient.
        monkeypatch.setattr(solver_mod, "load_site_forcing", fake_load_site_forcing)
        monkeypatch.setattr(solver_mod, "run_full_tile", fake_run_full_tile)

        overlay = ForcingOverlay(
            changes=(ForcingChange("air_temperature", None, None, 25.0),)
        )
        result = run_scenario_full_tile(
            cache,
            layer,
            site_dir=site,
            scratch_dir=tmp_path / "scratch",
            selected_date_str=DATE_STR,
            forcing_overlay=overlay,
        )
        assert captured["overlay"] is overlay
        assert captured["forcing"] is sentinel_forcing
        assert result == {"utci": None}

        # Legacy default: omitting the overlay materializes baseline
        # forcing (overlay=None), byte-identical to the pre-L3 behavior.
        run_scenario_full_tile(
            cache,
            layer,
            site_dir=site,
            scratch_dir=tmp_path / "scratch",
            selected_date_str=DATE_STR,
        )
        assert captured["overlay"] is None

    def test_regeneration_batch_param_accepted_and_forwarded(self) -> None:
        signature = inspect.signature(regenerate_building_batch)
        assert "forcing_overlay" in signature.parameters
        assert signature.parameters["forcing_overlay"].default is None
        # The batch orchestrator forwards its overlay into the full-tile
        # stage's forcing materialization.
        batch_source = inspect.getsource(regenerate_building_batch)
        assert "forcing_overlay=forcing_overlay" in batch_source
        wrapper_source = inspect.getsource(run_scenario_full_tile)
        assert "overlay=forcing_overlay" in wrapper_source


# ---------------------------------------------------------------------------
# u-c3 L5: resolved-met header names the width it wrote
# ---------------------------------------------------------------------------


class TestUc3L5ResolvedMetHeader:
    def test_header_count_matches_row_width_and_round_trips(self, tmp_path):
        table = np.arange(3 * 25, dtype=np.float64).reshape(3, 25) / 7.0
        path = tmp_path / "resolved_met.txt"
        solver_mod._write_resolved_met(path, table)

        header = path.read_text().splitlines()[0]
        stated = int(re.search(r"(\d+) columns", header).group(1))
        assert stated == table.shape[1] == 25  # the real layout, not 15
        # The staged text re-reads bitwise equal (loader contract).
        loaded = np.loadtxt(path, skiprows=1, delimiter=" ")
        assert loaded.shape == table.shape
        np.testing.assert_array_equal(loaded, table)
