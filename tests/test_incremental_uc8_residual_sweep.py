# SPDX-License-Identifier: GPL-3.0-only
"""u-c8 combined residual sweep: two review verdicts, one probe each.

Every fix from the u-c7-review (APPROVE-WITH-NITS) and u-c1-review
round-2 (R1-R4) residuals — plus the u-c6-review round-2 scope addition
F1, whose repro lives in ``test_incremental_uc6b_remediation.py`` — is
pinned here by the probe that fails without it:

- **Value-domain fence (u-c7 MEDIUM)** — the staging doors
  (:meth:`ExactWorker.stage_model_parameters` and the ``model_parameters``
  branch of :func:`run_full_tile`) used to check names and scalar KINDS
  only, so ``{albedo_b: True}`` (a bool where a float is required),
  ``{albedo_b: 50.0}`` (outside the documented ``[0, 1]``), and
  ``{transVeg: inf}`` (non-finite) staged and forwarded into the physics.
  Both doors now run the adapter's public
  :func:`check_parameter_value` — the probes are refused at each door
  with the door's own typed error.
- **Identity drift pin (u-c7 MEDIUM)** — the direct-path vs legacy
  ``compute_utci`` identity guarantee, swept over fixtures x payloads
  (two sites, four identity payloads, bitwise, :func:`np.array_equal`).
- **``solve_window`` Mapping fence (u-c7 LOW)** — same typed refusal the
  ``run_full_tile`` door has had.
- **Requested-variable symmetry (u-c7 LOW)** — the legacy full-tile path
  used to SILENTLY drop kup/kdown/lup/ldown requests from the published
  patch; the worker's solve door now refuses them with a typed error in
  the same "did not produce" family the direct path raises (the solver
  level stays permissive: the sanctioned oracle-helper pattern requests
  the full vocabulary through both branches).
- **Publish-by-copy (u-c7 TRIVIAL)** — the published model-parameter
  watermark is a copy, so mutating the staged mapping in place after a
  publication cannot rewrite what the job ran under.
- **R1 writer token (u-c1 MEDIUM)** — the executor writer id carries a
  per-instance token, so two executors on the SAME site/tile/scenario
  sharing one store no longer construct idempotently: the second is
  refused by the store's single-writer fence at construction.
- **R2 windowed supersession retention (u-c1 MEDIUM)** — a newer batch
  superseding only SOME windows of a key no longer drops the others
  (narrowed by R5a to the surviving CELLS of an intersecting window).
- **R3 ``node_entries`` TOCTOU (u-c1 LOW)** — the entry mapping is read
  once; a hammering invalidate/publish thread cannot wedge the reader.
- **R4 orphan cleanup on any failure (u-c1 LOW)** — a non-supersede
  failure between two window publishes discards the already-renamed
  patch directories instead of leaving never-current revisions behind.
"""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from solweig_gpu.incremental.adapters.model_parameters import (
    PARAMETER_DEFAULTS,
    check_parameter_value,
)
from solweig_gpu.incremental.edit_types import EditStateError
from solweig_gpu.incremental.executor import PlanExecutor
from solweig_gpu.incremental.geometry import RasterWindow
from solweig_gpu.incremental.solver import (
    SolverInputError,
    load_site_forcing,
    run_full_tile,
    solve_window,
)
from solweig_gpu.incremental import solver as solver_mod
from solweig_gpu.incremental.store import (
    CoverageStatus,
    StaleResultError,
    StoreWriterError,
    TemporalResultEntry,
    TemporalResultStore,
)
from solweig_gpu.incremental.trees import TreeLayer, TreeSpec
from solweig_gpu.incremental import worker as worker_mod
from solweig_gpu.incremental.worker import ExactWorker, WorkerError

from tests.test_incremental_params_integration import (
    _tiny_executor,
    _tiny_worker,
    params_site,
)
from tests.test_incremental_worker import (
    DATE_STR,
    LOCAL_ALWAYS,
    TINY_ADD,
    _build_cache,
    _compute_baseline_svf,
    _derive_lon_lat,
    _derive_utc_offset,
    _make_prepared_site,
    _make_tiny_site,
    _met_row,
)
from tests.test_incremental_params_physics import (
    EPSG,
    ORIGIN,
    PIXEL,
)
from tests.test_incremental_params_integration import _stub_patch

# ---------------------------------------------------------------------------
# Item 1 (u-c7 MEDIUM): value-domain fence at the two staging doors
# ---------------------------------------------------------------------------

#: The reviewer's three probe payloads: right scalar KIND, wrong VALUE.
VALUE_PROBES = (
    ("bool-for-float", {"albedo_b": True}),
    ("out-of-bounds", {"albedo_b": 50.0}),
    ("non-finite", {"transVeg": float("inf")}),
)


class TestParameterValueDomainFence:
    def test_public_wrapper_is_the_adapters_own_domain_check(self) -> None:
        """The sanctioned adapter surface: rejects every probe payload,
        accepts every documented default (the fence is value-domain only
        — it must not over-refuse the legitimate mapping)."""
        for _name, probe in VALUE_PROBES:
            (probe_name,) = probe
            with pytest.raises(EditStateError, match=probe_name):
                check_parameter_value(probe_name, probe[probe_name])
        with pytest.raises(EditStateError, match="unknown model parameter"):
            check_parameter_value("bogus", 1.0)
        for name, default in PARAMETER_DEFAULTS.items():
            check_parameter_value(name, default)

    def test_worker_door_refuses_probe_payloads(self, tmp_path: Path) -> None:
        worker, _site, _grid = _tiny_worker(tmp_path)
        for _name, probe in VALUE_PROBES:
            (probe_name,) = probe
            with pytest.raises(WorkerError) as caught:
                worker.stage_model_parameters(probe)
            # The door's typed error carries the adapter's domain reason.
            assert probe_name in str(caught.value)
            assert isinstance(caught.value.__cause__, EditStateError)
        # Nothing staged: a refused payload never advances state.
        assert worker._model_parameters is None

    def test_solver_door_refuses_probe_payloads(self, tmp_path: Path) -> None:
        worker, site, _grid = _tiny_worker(tmp_path)
        for index, (_name, probe) in enumerate(VALUE_PROBES):
            (probe_name,) = probe
            with pytest.raises(SolverInputError) as caught:
                run_full_tile(
                    worker.cache,
                    worker.layer,
                    forcing=worker.forcing(),
                    site_dir=site,
                    scratch_dir=tmp_path / f"scratch-{index}",
                    model_parameters=probe,
                )
            assert probe_name in str(caught.value)
            assert isinstance(caught.value.__cause__, EditStateError)

    def test_valid_values_pass_both_doors(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Guard against over-fencing: documented in-domain values still
        stage and reach the solver seam."""
        worker, _site, _grid = _tiny_worker(tmp_path)
        worker.stage_model_parameters({"albedo_b": 0.3, "cyl": False})
        assert worker._model_parameters == {"albedo_b": 0.3, "cyl": False}
        captured: dict = {}

        def fake_run_utci_window(**kwargs):
            captured.update(kwargs)
            rows, cols = worker.cache.rows, worker.cache.cols
            return {
                name: np.zeros(
                    (worker.cache.time_steps, rows, cols), dtype=np.float32
                )
                for name in kwargs["requested_variables"]
            }

        def fake_svf(*args, **kwargs):
            return tuple(
                torch.zeros((worker.cache.rows, worker.cache.cols))
                for _ in range(19)
            )

        monkeypatch.setattr(solver_mod, "run_utci_window", fake_run_utci_window)
        monkeypatch.setattr(solver_mod, "svf_calculator", fake_svf)
        outputs = run_full_tile(
            worker.cache,
            worker.layer,
            forcing=worker.forcing(),
            site_dir=worker.site_dir,
            scratch_dir=tmp_path / "scratch-ok",
            requested_variables=("utci",),
            model_parameters={"albedo_b": 0.35},
        )
        assert set(outputs) == {"utci"}
        assert captured["model_parameters"] == {"albedo_b": 0.35}


# ---------------------------------------------------------------------------
# Item 3 (u-c7 LOW): solve_window Mapping fence
# ---------------------------------------------------------------------------


class TestSolveWindowMappingFence:
    def test_non_mapping_refused_typed(self, tmp_path: Path) -> None:
        worker, _site, grid = _tiny_worker(tmp_path)
        with pytest.raises(
            SolverInputError, match="name -> value mapping, got list"
        ):
            solve_window(
                worker.cache,
                worker.layer,
                read_window=grid.full_window,
                write_window=grid.full_window,
                forcing=worker.forcing(),
                model_parameters=["albedo_b"],
            )


# ---------------------------------------------------------------------------
# Item 4 (u-c7 LOW): requested-variable symmetry on the legacy branch
# ---------------------------------------------------------------------------


class TestLegacyRequestedVariableFence:
    def test_legacy_worker_solve_refuses_flux_variables_loudly(
        self, tmp_path: Path
    ) -> None:
        """kup/kdown/lup/ldown were silently dropped from the published
        patch on the legacy full-tile path; the worker's solve door now
        refuses them with the same typed error family the direct path's
        "did not produce" uses — before any solve state advances."""
        worker, _site, _grid = _tiny_worker(tmp_path)
        assert worker._model_parameters is None  # legacy branch
        for flux in ("kup", "kdown", "lup", "ldown"):
            worker.requested_variables = ("utci", flux)
            with pytest.raises(
                WorkerError, match="legacy full-tile run cannot produce"
            ):
                worker._solve_full("job-flux", 1, worker.forcing())

    def test_legacy_solver_level_stays_permissive_for_the_oracle_helper(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """The sanctioned oracle-helper pattern requests the FULL variable
        vocabulary through BOTH solver branches and reads only the
        legacy-producible subset from the legacy result — so the fence
        lives at the worker door, and a solver-level legacy run with flux
        variables in the request enters compute_utci unaffected. (The
        stubbed orchestrator writes nothing, so the run still ends in the
        pre-existing "did not produce" read-back error for the legacy
        vocabulary — the flux names never cause a refusal here.)"""
        from tests.test_incremental_params_integration import ORACLE_VARIABLES

        worker, site, _grid = _tiny_worker(tmp_path)
        calls: list[dict] = []
        monkeypatch.setattr(
            solver_mod,
            "compute_utci",
            lambda **kwargs: calls.append(kwargs) or None,
        )
        with pytest.raises(SolverInputError, match="did not produce"):
            run_full_tile(
                worker.cache,
                worker.layer,
                forcing=worker.forcing(),
                site_dir=site,
                scratch_dir=tmp_path / "scratch-legacy",
                requested_variables=ORACLE_VARIABLES,
            )
        assert len(calls) == 1  # the flux names never refused the request

    def test_direct_solver_path_still_carries_flux_variables(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """The fence is legacy-only: under model parameters the direct
        solver path keeps honoring the whole vocabulary (no
        over-refusal) — it carries every run_utci_window variable. (The
        patch layer's own SUPPORTED_VARIABLES whitelist stays the gate
        for what may be PUBLISHED; this pins the solver seam.)"""
        worker, site, _grid = _tiny_worker(tmp_path)
        forcing = worker.forcing()

        def explode(**kwargs):  # pragma: no cover - must not run
            raise AssertionError("compute_utci must not run under parameters")

        def fake_run_utci_window(**kwargs):
            rows, cols = worker.cache.rows, worker.cache.cols
            return {
                name: np.zeros((forcing.time_steps, rows, cols), dtype=np.float32)
                for name in kwargs["requested_variables"]
            }

        monkeypatch.setattr(solver_mod, "compute_utci", explode)
        monkeypatch.setattr(solver_mod, "run_utci_window", fake_run_utci_window)
        monkeypatch.setattr(
            solver_mod,
            "svf_calculator",
            lambda *a, **k: tuple(
                torch.zeros((worker.cache.rows, worker.cache.cols))
                for _ in range(19)
            ),
        )
        outputs = run_full_tile(
            worker.cache,
            worker.layer,
            forcing=forcing,
            site_dir=site,
            scratch_dir=tmp_path / "scratch-direct",
            requested_variables=("utci", "kup", "kdown"),
            model_parameters={"albedo_b": 0.35},
        )
        assert set(outputs) == {"utci", "kup", "kdown"}


# ---------------------------------------------------------------------------
# Item 6 (u-c7 TRIVIAL): publish-by-copy of the params watermark
# ---------------------------------------------------------------------------


class TestPublishedParamsWatermarkCopy:
    def test_in_place_mutation_of_staged_mapping_cannot_rewrite_watermark(
        self, tmp_path: Path
    ) -> None:
        worker, _site, grid = _tiny_worker(tmp_path)
        worker._solve_full = (
            lambda job_id, revision, forcing: [
                _stub_patch(("utci",), job_id, revision, grid.full_window)
            ]
        )
        worker.stage_model_parameters({"albedo_b": 0.35})
        assert worker.run().published
        assert worker._published_model_parameters == {"albedo_b": 0.35}
        # Mutating the STAGED mapping in place after publication must not
        # alias into the published watermark (publish-by-reference did).
        staged = worker._model_parameters
        assert staged is not None
        staged["albedo_b"] = 0.9
        staged["ewall"] = 0.1
        assert worker._published_model_parameters == {"albedo_b": 0.35}
        assert worker.publication_watermarks.model_parameters == {
            "albedo_b": 0.35
        }


# ---------------------------------------------------------------------------
# Item 7 (u-c1-review R1, MEDIUM): per-instance writer token
# ---------------------------------------------------------------------------


class TestR1WriterIdentityToken:
    def test_same_scenario_second_executor_refused_at_construction(
        self, tmp_path: Path
    ) -> None:
        ex1 = _tiny_executor(tmp_path / "one")
        assert ex1.store.writer is not None
        # Same site, same tile, SAME scenario: pre-R1 the writer id was a
        # deterministic site:tile:scenario string, so the second executor
        # re-bound it IDEMPOTENTLY and both published into one store with
        # disjoint keys silently. The per-instance token makes the ids
        # distinct, so the store's single-writer fence now fires at the
        # second construction.
        with pytest.raises(StoreWriterError, match="One store has one writer"):
            PlanExecutor(
                cache=ex1.cache,
                layer=TreeLayer(
                    ex1.cache.tree_base, ex1.grid, scenario_id=ex1.layer.scenario_id
                ),
                site_dir=ex1.site_dir,
                results_root=tmp_path / "results2",
                selected_date_str=DATE_STR,
                influence_config=LOCAL_ALWAYS,
                store=ex1.store,
            )

    def test_writer_id_carries_address_plus_per_instance_token(
        self, tmp_path: Path
    ) -> None:
        ex = _tiny_executor(tmp_path)
        prefix = (
            f"executor:{ex.cache.site_id}:{ex.cache.tile_key}"
            f":{ex.layer.scenario_id}:"
        )
        assert ex._store_writer_id.startswith(prefix)
        token = ex._store_writer_id.removeprefix(prefix)
        assert len(token) == 8
        int(token, 16)  # hex token, not a counter two twins could share
        # The identical re-bind stays idempotent for the SAME executor.
        assert ex.store.bind_writer(ex._store_writer_id) == ex._store_writer_id


# ---------------------------------------------------------------------------
# Item 8 (u-c1-review R2, MEDIUM): windowed supersession retention
# ---------------------------------------------------------------------------


def _e(
    node: str,
    time: int,
    revision: int,
    *,
    job: str,
    window: RasterWindow,
    path: Path,
) -> TemporalResultEntry:
    return TemporalResultEntry(
        node_id=node,
        time_index=time,
        scene_revision=revision,
        job_id=job,
        mode="local",
        write_window=window,
        patch_path=path,
    )


class TestR2WindowedSupersessionRetention:
    W1 = RasterWindow(0, 10, 0, 10)
    W2 = RasterWindow(20, 30, 20, 30)

    def test_newer_windowed_batch_retains_nonintersecting_coverage(self) -> None:
        """The reviewer's exact scenario: rev-1 covers W1+W2 under
        ('utci', 0); rev-2 republishes W2 only. W1's entry is still the
        live coverage for its cells — dropping it wholesale (the pre-R2
        behavior) opened a phantom coverage gap and orphaned a valid
        patch directory."""
        store = TemporalResultStore()
        store.publish(
            [
                _e("utci", 0, 1, job="job1", window=self.W1, path=Path("/p/rev1-w1")),
                _e("utci", 0, 1, job="job1", window=self.W2, path=Path("/p/rev1-w2")),
            ]
        )
        store.publish(
            [_e("utci", 0, 2, job="job2", window=self.W2, path=Path("/p/rev2-w2"))]
        )
        entries = store.window_entries("utci", 0)
        assert len(entries) == 2
        w1_entry = store.lookup_window("utci", 0, row=5, col=5)
        assert w1_entry is not None
        assert w1_entry.scene_revision == 1
        assert w1_entry.patch_path == Path("/p/rev1-w1")
        w2_entry = store.lookup_window("utci", 0, row=25, col=25)
        assert w2_entry is not None
        assert w2_entry.scene_revision == 2
        assert w2_entry.patch_path == Path("/p/rev2-w2")
        # Coverage is unchanged-empty: no phantom gap over W1.
        assert store.coverage_gaps("utci", (0,)) == ()
        assert store.max_revision("utci") == 2

    def test_intersecting_windows_are_still_superseded(self) -> None:
        """R5a cell-granular supersession (of the R2 whole-entry rule):
        a partial overlap of W1 supersedes exactly the INTERSECTING
        CELLS — W1 keeps its non-intersecting remainder as narrowed
        entries referencing the SAME patch — while W2 survives verbatim
        and a full-tile batch still supersedes everything. The composed
        serve over W1 is bitwise: rev-2 values inside the batch window,
        rev-1 values in the retained remainder cells."""
        store = TemporalResultStore()
        w1_patch = Path("/p/1a")
        w2_patch = Path("/p/1b")
        batch_window = RasterWindow(5, 15, 5, 15)
        batch_patch = Path("/p/2")
        store.publish(
            [
                _e("utci", 0, 1, job="job1", window=self.W1, path=w1_patch),
                _e("utci", 0, 1, job="job1", window=self.W2, path=w2_patch),
            ]
        )
        # Overlaps W1 only.
        store.publish(
            [
                _e(
                    "utci", 0, 2, job="job2",
                    window=batch_window, path=batch_patch,
                )
            ]
        )
        entries = store.window_entries("utci", 0)
        # W1 minus the batch keeps TWO remainder rectangles; the batch's
        # own window and W2 (never touched) stay verbatim.
        assert {(e.scene_revision, e.write_window, e.patch_path) for e in entries} == {
            (1, RasterWindow(0, 5, 0, 10), w1_patch),
            (1, RasterWindow(5, 10, 0, 5), w1_patch),
            (1, self.W2, w2_patch),
            (2, batch_window, batch_patch),
        }
        # No phantom coverage gap over the SUPERSEDED window: retained
        # remainder + the batch's overlap tile W1 completely (the old
        # whole-entry drop degraded exactly this read).
        assert (
            store.window_coverage("utci", 0, window=self.W1).status
            is CoverageStatus.FULL
        )
        # A cell inside the batch window NEVER resolves to the older
        # entry — remainder rectangles are defined by subtraction.
        batch_entry = store.lookup_window("utci", 0, row=5, col=5)
        assert batch_entry is not None and batch_entry.scene_revision == 2
        for row in range(batch_window.row_start, batch_window.row_stop):
            for col in range(batch_window.col_start, batch_window.col_stop):
                resolved = store.lookup_window("utci", 0, row=row, col=col)
                assert resolved == batch_entry
        # Cells outside the batch resolve to their ORIGINAL payloads.
        assert store.lookup_window("utci", 0, row=0, col=0).patch_path == w1_patch
        assert store.lookup_window("utci", 0, row=25, col=25).scene_revision == 1

        # Composed serve, sliced the way the executor's plane assembly
        # slices (per-entry offsets against the PATCH window; patch
        # arrays are window-relative). Over the union bounding box the
        # composition is exact: rev-1 payload in the retained remainder
        # cells, rev-2 payload inside the batch window, and the
        # never-recorded corner stays at the sentinel (no phantom serve).
        rev1_plane = np.arange(100, dtype=np.int64).reshape(10, 10)
        rev2_plane = 10_000 + np.arange(100, dtype=np.int64).reshape(10, 10)
        payload_windows = {w1_patch: self.W1, batch_patch: batch_window}
        box = RasterWindow(0, 15, 0, 15)
        composed = np.full((15, 15), -1, dtype=np.int64)
        for entry in store.window_coverage("utci", 0, window=box).entries:
            payload = rev2_plane if entry.scene_revision == 2 else rev1_plane
            patch_window = payload_windows[entry.patch_path]
            window = entry.write_window
            composed[
                window.row_start : window.row_stop,
                window.col_start : window.col_stop,
            ] = payload[
                window.row_start - patch_window.row_start : window.row_stop - patch_window.row_start,
                window.col_start - patch_window.col_start : window.col_stop - patch_window.col_start,
            ]
        expected = np.full((15, 15), -1, dtype=np.int64)
        expected[0:10, 0:10] = rev1_plane  # W1's surviving cells
        expected[5:15, 5:15] = rev2_plane  # the batch republished its window
        np.testing.assert_array_equal(composed, expected)

        # A full-tile batch intersects every recorded window: every
        # remainder vanishes (nothing lies outside the tile), nothing
        # stale survives.
        store.publish(
            [
                _e(
                    "utci", 0, 3, job="job3",
                    window=RasterWindow(0, 40, 0, 40), path=Path("/p/3"),
                )
            ]
        )
        entries = store.window_entries("utci", 0)
        assert len(entries) == 1
        assert entries[0].scene_revision == 3

    def test_stale_guard_uses_the_highest_recorded_revision(self) -> None:
        """With mixed recorded revisions on one key, stale detection
        compares against the HIGHEST — a rev-1 publish against a key
        whose W2 already moved to rev-2 is stale even though W1 still
        carries rev-1."""
        store = TemporalResultStore()
        store.publish(
            [
                _e("utci", 0, 1, job="job1", window=self.W1, path=Path("/p/1a")),
                _e("utci", 0, 1, job="job1", window=self.W2, path=Path("/p/1b")),
            ]
        )
        store.publish(
            [_e("utci", 0, 2, job="job2", window=self.W2, path=Path("/p/2"))]
        )
        with pytest.raises(StaleResultError, match="stale publish"):
            store.publish(
                [
                    _e(
                        "utci", 0, 1, job="job1b",
                        window=RasterWindow(30, 35, 30, 35), path=Path("/p/1c"),
                    )
                ]
            )

    def test_same_revision_replay_keeps_retained_entries_verbatim(self) -> None:
        """An idempotent replay of the rev-2 batch against a key that
        carries retained rev-1 windows keeps EVERY recorded entry (the
        retained old windows are still live coverage, not replay
        content)."""
        store = TemporalResultStore()
        store.publish(
            [
                _e("utci", 0, 1, job="job1", window=self.W1, path=Path("/p/1a")),
                _e("utci", 0, 1, job="job1", window=self.W2, path=Path("/p/1b")),
            ]
        )
        second = [
            _e("utci", 0, 2, job="job2", window=self.W2, path=Path("/p/2"))
        ]
        store.publish(second)
        keys = store.publish(second)  # byte-identical replay
        assert keys == (("utci", 0),)
        entries = store.window_entries("utci", 0)
        assert {(e.scene_revision, e.job_id) for e in entries} == {
            (1, "job1"),
            (2, "job2"),
        }


# ---------------------------------------------------------------------------
# Item 9 (u-c1-review R3, LOW): node_entries snapshot-once
# ---------------------------------------------------------------------------


class TestR3NodeEntriesSnapshotOnce:
    def test_hammer_invalidate_publish_never_wedges_reader(self) -> None:
        """The reader sorted one snapshot and indexed another — a KeyError
        TOCTOU whenever an invalidate/publish swapped the mapping between
        the two reads. Hammer both sides; no exception is tolerated."""
        store = TemporalResultStore()
        store.publish(
            [
                _e(
                    "utci", t, 1, job="seed",
                    window=RasterWindow(0, 4, 0, 4), path=Path(f"/p/seed-{t}"),
                )
                for t in range(8)
            ]
        )
        errors: list[BaseException] = []
        start = threading.Barrier(3)
        ROUNDS = 5000

        def churn():
            try:
                start.wait()
                for _ in range(ROUNDS):
                    # Swap the mapping twice per round: remove key 0, then
                    # put it back (readers race a live snapshot swap).
                    store.invalidate("utci", times=(0,))
                    store.publish(
                        [
                            _e(
                                "utci", 0, 1, job="seed",
                                window=RasterWindow(0, 4, 0, 4),
                                path=Path("/p/seed-0"),
                            )
                        ]
                    )
            except BaseException as error:  # pragma: no cover - failure path
                errors.append(error)

        def reader():
            try:
                start.wait()
                for _ in range(ROUNDS):
                    entries = store.node_entries("utci")
                    if not entries:
                        errors.append(AssertionError("node_entries lost its seed"))
                        return
            except BaseException as error:  # pragma: no cover - failure path
                errors.append(error)

        threads = [threading.Thread(target=churn), threading.Thread(target=reader)]
        for thread in threads:
            thread.start()
        start.wait()
        for thread in threads:
            thread.join()
        assert not errors


# ---------------------------------------------------------------------------
# Item 10 (u-c1-review R4, LOW): orphan cleanup on ANY failure
# ---------------------------------------------------------------------------


class TestR4OrphanCleanupOnAnyFailure:
    def test_failure_between_window_publishes_discards_partial_patches(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        grid, site = _make_tiny_site(tmp_path / "site")
        cache = _build_cache(
            site,
            tmp_path / "cache",
            met_path=site / "metfiles" / f"metfile_0_0_{DATE_STR}.txt",
            site_id="tiny",
        )
        layer = TreeLayer(cache.tree_base, grid)
        worker = ExactWorker(
            cache,
            layer,
            site_dir=site,
            results_root=tmp_path / "results",
            selected_date_str=DATE_STR,
            influence_config=LOCAL_ALWAYS,
        )
        worker.bump_scene_revision()
        layer.add_tree(TINY_ADD)

        w1 = RasterWindow(0, 10, 0, 10)
        w2 = RasterWindow(40, 50, 40, 50)

        def fake_solve_local(job_id, revision, forcing, write_windows, **kwargs):
            return [
                _stub_patch(
                    ("utci",), f"{job_id}-w{i}", revision, window, mode="local"
                )
                for i, window in enumerate((w1, w2))
            ]

        worker._solve_local = fake_solve_local

        real_publish = worker_mod.publish_staged_patch
        renames: list[Path] = []

        def publish_then_fail(staging_dir, results_root, scene_revision):
            if renames:
                raise OSError("injected non-supersede failure mid-publish-loop")
            published = real_publish(
                staging_dir, results_root, scene_revision
            )
            renames.append(published)
            return published

        monkeypatch.setattr(
            worker_mod, "publish_staged_patch", publish_then_fail
        )
        with pytest.raises(OSError, match="injected non-supersede failure"):
            worker.run()

        # The FIRST window's already-renamed patch directory — an orphan
        # of a job that will never complete — is gone, not left behind as
        # a never-current revision a rev-globbing reader could trust.
        assert renames, "the probe never published its first window"
        assert not any(path.exists() for path in renames)
        assert not list((tmp_path / "results").rglob("rev-*"))
        # The unrenamed second staging directory is discarded too.
        staging = tmp_path / "results" / layer.scenario_id / ".staging"
        assert not list(staging.rglob("*.tmp"))
        # The edit is still pending: nothing was consumed by the failed job.
        assert worker.pending_batch() is not None
        assert worker._acked_sequence == 0
        # Recovery: without the injected failure the same batch publishes.
        monkeypatch.setattr(worker_mod, "publish_staged_patch", real_publish)
        outcome = worker.run()
        assert outcome.published
        assert len(outcome.patch_paths) == 2


# ---------------------------------------------------------------------------
# Item 2 (u-c7 MEDIUM): identity drift pin, swept fixtures x payloads
# ---------------------------------------------------------------------------

#: Identity payloads: every value EQUALS its documented default, so the
#: direct-path run under the payload must equal the legacy
#: ``compute_utci`` run bitwise. The all-defaults cell ({}) exercises the
#: parameter-free run itself (model_parameters=None routes legacy).
IDENTITY_PAYLOADS = (
    ("cyl-default", {"cyl": True}),
    ("height-default", {"height": 1.1}),
    ("anisotropic-sky-default", {"anisotropic_sky": 1}),
    ("all-defaults", {}),
)

#: Bitwise compare over the legacy path's own vocabulary: compute_utci
#: writes only utci/tmrt/shadow (its flux switches are hardwired False).
SWEEP_VARIABLES = ("utci", "tmrt", "shadow")


def _met_row_doy(hour: int, doy: float) -> list[float]:
    row = list(_met_row(hour))
    row[1] = doy
    return row


def _write_met_doy(path: Path, hours, doy: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "# year doy hour minute placeholder wind rh ta p radg rad diff radI wdir uhii"
    lines = [header] + [
        " ".join(f"{v:g}" for v in _met_row_doy(h, doy)) for h in hours
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


def _build_cache_dated(
    site_dir: Path,
    cache_dir: Path,
    *,
    met_path: Path,
    site_id: str,
    date_str: str,
):
    from solweig_gpu.incremental.cache_builder import build_site_cache

    building = site_dir / "Building_DSM" / "Building_DSM_0_0.tif"
    lon, lat = _derive_lon_lat(building)
    utc = _derive_utc_offset(lat, lon, date_str)
    build_site_cache(
        site_dir,
        cache_dir,
        tile_key="0_0",
        site_id=site_id,
        latitude=lat,
        longitude=lon,
        altitude_m=0.0,
        utc_offset_hours=utc,
        met_file=met_path,
    )
    from solweig_gpu.incremental.cache import SiteCache

    return SiteCache.load(cache_dir)


def _make_dated_site(
    root: Path,
    *,
    rows: int,
    cols: int,
    date_str: str,
    doy: float,
    tree: TreeSpec,
    site_id: str,
) -> SimpleNamespace:
    """A second harness site at a DIFFERENT DOY: the identity guarantee is
    a property of the physics replication, not of one DOY-172 layout."""
    from tests.test_incremental_params_physics import MET_HOURS

    grid, site = _make_prepared_site(
        root,
        rows=rows,
        cols=cols,
        pixel=PIXEL,
        origin=ORIGIN,
        epsg=EPSG,
        base_trees=(tree,),
        met_hours=MET_HOURS,
    )
    dated_met = _write_met_doy(
        site / "metfiles" / f"metfile_0_0_{date_str}.txt", MET_HOURS, doy
    )
    (site / "metfiles" / f"metfile_0_0_{DATE_STR}.txt").unlink()
    _compute_baseline_svf(site)
    cache = _build_cache_dated(
        site,
        root / "cache",
        met_path=dated_met,
        site_id=site_id,
        date_str=date_str,
    )
    return SimpleNamespace(grid=grid, site=site, cache=cache, date_str=date_str)


def _direct_run_dated(
    site: SimpleNamespace, scratch: Path, *, model_parameters=None
) -> dict[str, np.ndarray]:
    """The independent oracle at the site's OWN date (the shared
    ``_direct_run`` helper hardcodes the DOY-172 date string)."""
    forcing = load_site_forcing(
        site.cache,
        site_dir=site.site,
        selected_date_str=site.date_str,
    )
    return run_full_tile(
        site.cache,
        TreeLayer(site.cache.tree_base, site.grid),
        forcing=forcing,
        site_dir=site.site,
        scratch_dir=scratch,
        requested_variables=SWEEP_VARIABLES,
        model_parameters=(model_parameters or None),
    )


@pytest.fixture(scope="module")
def sweep_sites(tmp_path_factory, params_site):
    """The 64x64 DOY-172 harness site (shared with the params integration
    suite) plus a 48x48 DOY-300 (2024-10-26) site built the same way."""
    doy300_tree = TreeSpec(
        "sweep-tree",
        ORIGIN[0] + 12.5 * PIXEL,
        ORIGIN[1] - 24.5 * PIXEL,
        8.0,
        3.0,
    )
    doy300 = _make_dated_site(
        tmp_path_factory.mktemp("sweep_doy300_site"),
        rows=48,
        cols=48,
        date_str="2024-10-26",
        doy=300.0,
        tree=doy300_tree,
        site_id="sweep-doy300",
    )
    return {
        "doy172-64x64": SimpleNamespace(
            grid=params_site.grid,
            site=params_site.site,
            cache=params_site.cache,
            date_str=DATE_STR,
        ),
        "doy300-48x48": doy300,
    }


@pytest.fixture(scope="module")
def legacy_baselines(sweep_sites, tmp_path_factory) -> dict[str, dict]:
    """One legacy compute_utci baseline per site, memoized for the sweep."""
    baselines: dict[str, dict] = {}
    for name, site in sweep_sites.items():
        baselines[name] = _direct_run_dated(
            site, tmp_path_factory.mktemp(f"sweep_legacy_{name}")
        )
    return baselines


@pytest.mark.scientific
class TestDirectPathIdentitySweep:
    @pytest.mark.parametrize("payload_name,payload", IDENTITY_PAYLOADS)
    def test_identity_payload_matches_legacy_bitwise(
        self,
        sweep_sites,
        legacy_baselines,
        tmp_path: Path,
        payload_name: str,
        payload: dict,
    ) -> None:
        """The drift pin, swept: on EVERY site and EVERY identity payload
        (each value equals its documented default), the direct
        model-parameter path and the legacy compute_utci path agree
        BITWISE — cell for cell, band for band, over every variable the
        legacy path writes. A single differing float anywhere in the
        replication (a dtype, a fold order, a default misspelling) fails
        this exactly; no tolerance absorbs it."""
        for site_name, site in sweep_sites.items():
            identity = _direct_run_dated(
                site,
                tmp_path / f"identity-{site_name}-{payload_name}",
                model_parameters=payload,
            )
            legacy = legacy_baselines[site_name]
            for variable in SWEEP_VARIABLES:
                assert identity[variable].shape == legacy[variable].shape
                assert np.array_equal(
                    identity[variable], legacy[variable], equal_nan=True
                ), (
                    f"{site_name} / {payload_name} / {variable}: the direct "
                    "path drifted from the legacy compute_utci baseline "
                    "under an identity payload"
                )
