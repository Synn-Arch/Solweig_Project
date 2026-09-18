# SPDX-License-Identifier: GPL-3.0-only
"""Perf wave 1: honest telemetry, heartbeat liveness, time-prefix truncation.

Covers the four confirmed, parity-safe optimizations:

* **STEP 1a (honest telemetry)** — job metrics carry the READ-window union
  fraction next to the write ``window_fraction``, plus per-stage
  ``svf_seconds`` / ``time_loop_seconds`` timings when the worker reports
  them (additive fields; the write fraction keeps its name and meaning).
* **STEP 1b (heartbeat starvation)** — a daemon ticker keeps
  ``last_heartbeat_monotonic`` fresh while the single worker thread is
  blocked inside a long solve, so ``/health/ready`` cannot flap 503 for the
  whole solve; a dead/stopped worker stops heartbeating.
* **STEP 2a (time-prefix truncation)** — an explicit, non-``refine_full_day``
  ``time_indices`` request solves only the causal prefix ``t = 0..max``.
  The forward-only temporal replay (Tgmap1/TgOut1/CI/Twater/firstdaytime/
  timeadd state chains) makes every computed step bit-identical to the
  full-day run, asserted here at the ``run_utci_window``, worker, and
  end-to-end server levels.
* **STEP 2b (trivial dedups)** — one scene composition + read-window
  derivation per window (the halo guard consumes the shared value), the
  SVF patch cubes sliced at the write extent they are consumed at, and the
  supersede scan running BEFORE the coalescing sleep as well as after it.
"""

from __future__ import annotations

import dataclasses
import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from solweig_gpu.incremental.geometry import RasterWindow
from solweig_gpu.server.jobs import JobRunner

from tests.test_incremental_params_physics import (
    build_physics_fixture,
    run_kwargs,
)
from tests.test_server_api import (
    SCENARIO0,
    FakeSolver,
    add_tree,
    make_app,
    make_tree,
    wait_for_job,
)
from tests.test_incremental_worker import science_site  # noqa: F401  (fixture)

DATE_STR = "2024-06-20"


# ---------------------------------------------------------------------------
# STEP 2a: requested-time-prefix resolution (fast units)
# ---------------------------------------------------------------------------


class TestResolveTimeStop:
    @staticmethod
    def _resolve(requested, grid):
        from solweig_gpu.server.jobs import resolve_time_stop

        return resolve_time_stop(requested, grid)

    def test_explicit_indices_prefix_stop(self) -> None:
        grid = {"time_steps": 24}
        assert self._resolve({"time_indices": [12]}, grid) == 13

    def test_explicit_indices_unsorted_prefix_stop(self) -> None:
        grid = {"time_steps": 24}
        assert self._resolve({"time_indices": [17, 3, 5]}, grid) == 18

    def test_max_index_equals_last_step(self) -> None:
        grid = {"time_steps": 24}
        assert self._resolve({"time_indices": [23]}, grid) == 24

    def test_no_request_is_full_range(self) -> None:
        assert self._resolve(None, {"time_steps": 24}) is None
        assert self._resolve({}, {"time_steps": 24}) is None
        assert self._resolve({"variables": ["utci"]}, {"time_steps": 24}) is None

    def test_refine_full_day_is_full_range(self) -> None:
        grid = {"time_steps": 24}
        requested = {"time_indices": [12], "refine_full_day": True}
        assert self._resolve(requested, grid) is None

    def test_indices_out_of_series_fall_back_to_full_range(self) -> None:
        grid = {"time_steps": 10}
        assert self._resolve({"time_indices": [12]}, grid) is None
        assert self._resolve({"time_indices": [-1, 99]}, grid) is None

    def test_mixed_valid_and_invalid_uses_valid_maximum(self) -> None:
        grid = {"time_steps": 24}
        assert self._resolve({"time_indices": [99, 4]}, grid) == 5


# ---------------------------------------------------------------------------
# STEP 1a: read-window fraction helper (fast unit)
# ---------------------------------------------------------------------------


class TestReadWindowFraction:
    @staticmethod
    def _fraction(windows, grid):
        from solweig_gpu.server.jobs import union_window_area_fraction

        return union_window_area_fraction(windows, grid)

    def test_disjoint_windows_sum(self) -> None:
        grid = {"rows": 10, "cols": 10}
        windows = (RasterWindow(0, 2, 0, 5), RasterWindow(8, 10, 0, 5))
        assert self._fraction(windows, grid) == pytest.approx(0.2)

    def test_overlapping_windows_union_not_sum(self) -> None:
        grid = {"rows": 10, "cols": 10}
        windows = (RasterWindow(0, 5, 0, 10), RasterWindow(3, 8, 0, 10))
        assert self._fraction(windows, grid) == pytest.approx(0.8)

    def test_empty_returns_none(self) -> None:
        assert self._fraction((), {"rows": 10, "cols": 10}) is None

    def test_full_window_is_one(self) -> None:
        grid = {"rows": 4, "cols": 6}
        assert self._fraction((RasterWindow(0, 4, 0, 6),), grid) == 1.0


# ---------------------------------------------------------------------------
# STEP 1b: heartbeat starvation
# ---------------------------------------------------------------------------


class TestHeartbeatTicker:
    def test_heartbeat_stays_fresh_during_long_solve(self, tmp_path: Path) -> None:
        solver = FakeSolver()
        solver.gate = threading.Event()
        app = make_app(tmp_path, solver, start_worker=False)
        # A dedicated runner with a SHORT heartbeat interval: without a
        # ticker the heartbeat goes stale for the whole gated solve.
        runner = JobRunner(
            app.state.context,
            solver_factory=lambda _context: solver,
            coalescing_window_ms=20.0,
            heartbeat_interval_s=0.05,
        )
        with TestClient(app) as client:
            scenario_id = SCENARIO0(client)
            submitted = add_tree(client, make_tree("hb-tree"), base=0)
            assert submitted.status_code == 202, submitted.text
            job_id = submitted.json()["job_id"]

            runner.start()
            try:
                deadline = time.monotonic() + 10.0
                while not solver.entered.is_set():
                    assert time.monotonic() < deadline, "solver never started"
                    time.sleep(0.005)
                # Sample heartbeat freshness across ~6 ticker intervals while
                # the worker thread is blocked inside the solve.
                worst = 0.0
                for _ in range(40):
                    worst = max(
                        worst, runner.worker_status()["seconds_since_heartbeat"]
                    )
                    time.sleep(0.01)
                assert worst < 0.2, (
                    f"heartbeat went stale mid-solve ({worst:.3f}s without a "
                    "tick): /health/ready would flap 503 for the whole solve"
                )
                solver.gate.set()  # release the gated solve
                wait_for_job(client, job_id, timeout=10.0)
            finally:
                runner.stop()

    def test_stopped_runner_stops_heartbeating(self, tmp_path: Path) -> None:
        solver = FakeSolver()
        app = make_app(tmp_path, solver, start_worker=False)
        runner = JobRunner(
            app.state.context,
            solver_factory=lambda _context: solver,
            coalescing_window_ms=20.0,
            heartbeat_interval_s=0.05,
        )
        with TestClient(app) as client:
            scenario_id = SCENARIO0(client)
            submitted = add_tree(client, make_tree("dead-tree"), base=0)
            assert submitted.status_code == 202, submitted.text
            runner.start()
            deadline = time.monotonic() + 10.0
            while not solver.entered.is_set():
                assert time.monotonic() < deadline, "solver never started"
                time.sleep(0.005)
            runner.stop()
            last = runner.last_heartbeat_monotonic
            time.sleep(0.25)  # several ticker intervals
            assert runner.last_heartbeat_monotonic == last, (
                "a stopped runner's ticker kept heartbeating: liveness "
                "semantics must not outlive the worker thread"
            )
            assert not runner.running


# ---------------------------------------------------------------------------
# STEP 2b (item 3): supersede scan before the coalescing sleep
# ---------------------------------------------------------------------------


class TestSupersedeBeforeSleep:
    def test_newer_queued_job_supersedes_older_without_dispatch(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        solver = FakeSolver()
        app = make_app(tmp_path, solver, start_worker=False, coalescing_window_ms=400.0)
        dispatched: list[str] = []
        original_dispatch = JobRunner._dispatch

        def spy(self, job_id: str):
            dispatched.append(job_id)
            return original_dispatch(self, job_id)

        monkeypatch.setattr(JobRunner, "_dispatch", spy)
        with TestClient(app) as client:
            scenario_id = SCENARIO0(client)
            first = add_tree(client, make_tree("older-tree"), base=0)
            assert first.status_code == 202, first.text
            job1 = first.json()["job_id"]
            second = add_tree(client, make_tree("newer-tree"), base=1)
            assert second.status_code == 202, second.text
            job2 = second.json()["job_id"]

            app.state.runner.start()
            detail1 = wait_for_job(client, job1, statuses=("superseded",), timeout=15.0)
            detail2 = wait_for_job(client, job2, timeout=15.0)

            assert detail1["status"] == "superseded"
            # The store's submit path already supersedes older queued jobs
            # when the newer edit lands (no error payload on that path);
            # the runner's pre-sleep scan is the second layer. Either way
            # the invariant under test holds: the older job's solve is
            # never paid for.
            assert job1 not in dispatched, (
                "the older job was dispatched even though a strictly newer "
                "queued job already obsoleted it: the pre-sleep supersede "
                "scan must abort it before any solve work"
            )
            assert job2 in dispatched
            assert detail2["status"] == "complete"

    def test_supersede_scan_marks_redundant_job_and_aborts(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Unit level: the scan itself marks + aborts (the marking layer).

        The public API cannot leave two queued jobs for one scenario (every
        submit supersedes older queued rows), so the scan's marking branch
        is exercised directly against the durable store.
        """
        solver = FakeSolver()
        app = make_app(tmp_path, solver, start_worker=False)
        runner = app.state.runner
        store = app.state.context.store
        with TestClient(app) as client:
            scenario_id = SCENARIO0(client)
            submitted = add_tree(client, make_tree("scan-tree"), base=0)
            job_id = submitted.json()["job_id"]
            job = store.require_job(job_id)
            assert job.status == "queued"

            # Nothing newer queued: the scan must let the job through.
            assert runner._supersede_scan(job_id) is False
            assert store.require_job(job_id).status == "queued"

            # A strictly newer queued job for the same scenario makes this
            # one redundant: the scan marks it superseded (with the error
            # code) and reports "do not dispatch".
            newer = dataclasses.replace(
                job,
                job_id="job_newer_shim",
                target_scene_version=job.target_scene_version + 1,
            )
            monkeypatch.setattr(
                store, "nonterminal_jobs", lambda: [job, newer]
            )
            assert runner._supersede_scan(job_id) is True
            marked = store.require_job(job_id)
            assert marked.status == "superseded"
            assert marked.error == {
                "code": "superseded",
                "message": "a newer edit superseded this queued job",
            }
            # A second scan on the now-terminal job stays an abort (idempotent
            # for the dispatch decision) without touching the store again.
            assert runner._supersede_scan(job_id) is True
            assert store.require_job(job_id).finished_at == marked.finished_at

    def test_edit_arriving_during_sleep_supersedes_pending_job(
        self, tmp_path: Path
    ) -> None:
        solver = FakeSolver()
        app = make_app(tmp_path, solver, start_worker=False, coalescing_window_ms=400.0)
        with TestClient(app) as client:
            scenario_id = SCENARIO0(client)
            first = add_tree(client, make_tree("pending-tree"), base=0)
            job1 = first.json()["job_id"]
            app.state.runner.start()
            # The newer edit lands while the runner is inside the coalescing
            # sleep for job1: the post-sleep scan must obsolete job1.
            second = add_tree(client, make_tree("during-sleep-tree"), base=1)
            job2 = second.json()["job_id"]
            detail1 = wait_for_job(client, job1, statuses=("superseded",), timeout=15.0)
            detail2 = wait_for_job(client, job2, timeout=15.0)
            assert detail1["status"] == "superseded"
            assert detail2["status"] == "complete"


# ---------------------------------------------------------------------------
# STEP 2a (run level) + STEP 2b (item 2): run_utci_window seams
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def physics_fx(tmp_path_factory):
    return build_physics_fixture(tmp_path_factory.mktemp("wave1_physics"))


class TestRunUtciWindowTimePrefix:
    def test_prefix_steps_match_full_run_bitwise(self, physics_fx) -> None:
        import solweig_gpu.utci_process as utci_process

        full = utci_process.run_utci_window(**run_kwargs(physics_fx))
        prefix_kwargs = run_kwargs(physics_fx)
        prefix_kwargs["time_stop"] = 2
        prefix = utci_process.run_utci_window(**prefix_kwargs)
        assert prefix["utci"].shape[0] == 2
        for name, array in prefix.items():
            assert np.array_equal(array, full[name][:2], equal_nan=True), (
                f"{name}: causal prefix steps differ from the full-day run"
            )


class TestPreCroppedSvfCubes:
    def test_pre_cropped_cubes_match_late_crop_bitwise(self, physics_fx) -> None:
        import solweig_gpu.utci_process as utci_process

        out_window = (16, 40, 20, 48)
        legacy_kwargs = run_kwargs(physics_fx, out_window=out_window)
        legacy = utci_process.run_utci_window(**legacy_kwargs)

        cropped_kwargs = run_kwargs(physics_fx, out_window=out_window)
        r0, r1, c0, c1 = out_window
        bundle = list(cropped_kwargs["svf_bundle"])
        # cube positions 15/16/17 (vegshmat, vbshvegshmat, shmat) are
        # pre-cropped to the write window; per-cell scalars stay read-sized.
        for position in (15, 16, 17):
            bundle[position] = bundle[position][r0:r1, c0:c1]
        cropped_kwargs["svf_bundle"] = tuple(bundle)
        cropped_kwargs["svf_bundle_cubes_cropped"] = True
        cropped = utci_process.run_utci_window(**cropped_kwargs)

        for name, array in cropped.items():
            assert np.array_equal(array, legacy[name], equal_nan=True), (
                f"{name}: cropping the SVF shadow cubes early (write extent) "
                "changed the outputs"
            )


# ---------------------------------------------------------------------------
# STEP 2a / 2b-1 / 2b-2 / 1a (worker level, real physics)
# ---------------------------------------------------------------------------


@pytest.mark.scientific
class TestWorkerTimePrefixAndDedup:
    def test_worker_time_stop_prefix_bitwise_and_metrics(self, science_site) -> None:
        from solweig_gpu.incremental.worker import ExactWorker
        from tests.test_incremental_worker import (
            VARIABLES,
            _local_worker,
            _state_layer,
            load_patch,
        )

        layer_prefix = _state_layer(science_site, "add")
        worker_prefix = _local_worker(
            science_site, layer_prefix, science_site.results_root("prefix")
        )
        worker_prefix.bump_scene_revision()
        outcome = worker_prefix.run(time_stop=13)
        assert outcome.published and outcome.mode == "local"
        patch_prefix = load_patch(outcome.patch_paths[0])
        assert patch_prefix.n_time_steps == 13, "prefix patch must cover t=0..12"
        assert patch_prefix.time_start == 0
        assert patch_prefix.time_stop == 13

        layer_full = _state_layer(science_site, "add")
        worker_full = _local_worker(
            science_site, layer_full, science_site.results_root("fullday")
        )
        worker_full.bump_scene_revision()
        outcome_full = worker_full.run()
        assert outcome_full.published and outcome_full.mode == "local"
        patch_full = load_patch(outcome_full.patch_paths[0])
        assert patch_full.n_time_steps == science_site.cache_a.time_steps

        for name in VARIABLES:
            assert np.array_equal(
                patch_prefix.arrays[name][12],
                patch_full.arrays[name][12],
                equal_nan=True,
            ), f"{name}: step 12 differs between the prefix and full-day runs"

        timings = (outcome.diagnostics or {}).get("stage_timings")
        assert timings is not None, "worker diagnostics must carry stage timings"
        assert timings["svf_seconds"] >= 0.0
        assert timings["time_loop_seconds"] >= 0.0

    def test_local_solve_composes_scene_once_per_window(
        self, science_site, monkeypatch
    ) -> None:
        import solweig_gpu.incremental.worker as worker_mod
        from tests.test_incremental_worker import _local_worker, _state_layer

        compose_calls = []
        read_window_calls = []

        original_compose = worker_mod.compose_full_scene_tensors
        original_read_window = worker_mod.read_window_for_write_window

        def counting_compose(cache, layer):
            compose_calls.append(1)
            return original_compose(cache, layer)

        def counting_read_window(write_window, cache, scene, forcing):
            read_window_calls.append(1)
            return original_read_window(write_window, cache, scene, forcing)

        monkeypatch.setattr(
            worker_mod, "compose_full_scene_tensors", counting_compose
        )
        monkeypatch.setattr(
            worker_mod, "read_window_for_write_window", counting_read_window
        )

        layer = _state_layer(science_site, "add")
        worker = _local_worker(science_site, layer, science_site.results_root("dedup"))
        worker.bump_scene_revision()
        outcome = worker.run()
        assert outcome.published and outcome.mode == "local"

        assert compose_calls == [1], (
            f"scene composed {len(compose_calls)}x for a single-window local "
            "job: the worker derivation and the solver must share one "
            "composition"
        )
        assert read_window_calls == [1], (
            f"read window derived {len(read_window_calls)}x for a "
            "single-window local job"
        )

    def test_window_svf_bundle_cube_window_equivalence(self, science_site) -> None:
        from solweig_gpu.incremental.solver import (
            compose_full_scene_tensors,
            window_svf_bundle,
        )
        from tests.test_incremental_worker import SVF_BUNDLE_INDEX, _state_layer

        cache = science_site.cache_a
        # veg_changed=False requires the scene vegetation to equal the cached
        # baseline exactly (the stale-SVF guard); the edited scene covers the
        # recompute branch.
        scenes = {
            True: _state_layer(science_site, "add"),
            False: _state_layer(science_site, "baseline"),
        }
        full = science_site.grid.full_window
        read_window = RasterWindow(
            max(full.row_start, 40),
            min(full.row_stop, 140),
            max(full.col_start, 40),
            min(full.col_stop, 140),
        )
        write_window = RasterWindow(60, 100, 70, 120)

        for veg_changed in (True, False):
            scene = compose_full_scene_tensors(cache, scenes[veg_changed])
            legacy = window_svf_bundle(
                cache, scene, read_window, veg_changed=veg_changed
            )
            cropped = window_svf_bundle(
                cache,
                scene,
                read_window,
                veg_changed=veg_changed,
                cube_window=write_window,
            )
            r0 = write_window.row_start - read_window.row_start
            r1 = write_window.row_stop - read_window.row_start
            c0 = write_window.col_start - read_window.col_start
            c1 = write_window.col_stop - read_window.col_start
            for name in ("vegshmat", "vbshvegshmat", "shmat"):
                position = SVF_BUNDLE_INDEX[name]
                assert np.array_equal(
                    legacy[position][r0:r1, c0:c1],
                    cropped[position],
                    equal_nan=True,
                ), f"{name} (veg_changed={veg_changed}): cube_window slice mismatch"
            # per-cell scalars keep the read extent; only the cubes shrink
            for name in ("svf", "svfveg", "svfaveg", "svftotal"):
                position = SVF_BUNDLE_INDEX[name]
                assert cropped[position].shape == legacy[position].shape


# ---------------------------------------------------------------------------
# STEP 2a + 1a: end-to-end server parity (the acceptance oracle)
# ---------------------------------------------------------------------------


def _seed_baseline_results(cache_dir: Path, time_steps: int, rows: int, cols: int) -> None:
    """Zero-valued baseline outputs so scenario creation publishes inline.

    The parity comparison only reads freshly solved write-window cells, so
    the baseline content is irrelevant; the shapes must match the manifest.
    """
    baseline_dir = cache_dir / "baseline_results"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    for name in ("utci", "tmrt"):
        np.save(
            baseline_dir / f"{name}.f32.npy",
            np.zeros((time_steps, rows, cols), dtype=np.float32),
        )
    (baseline_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "variables": ["utci", "tmrt"],
                "time_steps": time_steps,
                "source": "wave1-parity-seed",
            }
        )
    )


@pytest.mark.scientific
class TestTimePrefixEndToEndParity:
    def test_time_indices_12_matches_full_day_step_12(self, science_site, tmp_path) -> None:
        from solweig_gpu.server import patch_codec
        from solweig_gpu.server.app import create_app

        cache = science_site.cache_a
        # SiteCache maps the directory privately; the registry needs the dir.
        cache_dir = Path(cache._cache_dir)
        grid = science_site.grid
        _seed_baseline_results(cache_dir, cache.time_steps, grid.rows, grid.cols)

        u, v = _world_to_uv(science_site, science_site.add_position_m)

        app = create_app(
            state_root=tmp_path / "state",
            sites={
                "science": {
                    "cache_dir": cache_dir,
                    "site_dir": science_site.site_dir,
                    "selected_date_str": DATE_STR,
                }
            },
            coalescing_window_ms=50.0,
            requests_per_minute_per_ip=None,
            edits_per_minute=None,
        )
        with TestClient(app) as client:
            payloads = {}
            for label, requested in (
                ("prefix", {"time_indices": [12], "variables": ["utci", "tmrt"]}),
                ("full", {"variables": ["utci", "tmrt"]}),
            ):
                created = client.post(
                    "/api/v1/scenarios", json={"site_id": "science"}
                )
                assert created.status_code == 201, created.text
                scenario_id = created.json()["scenario_id"]
                body = {
                    "base_scene_version": 0,
                    "edits": [
                        {
                            "operation": "add",
                            "tree": {
                                "tree_id": "parity-tree",
                                "component_type": "broad_canopy",
                                "u": u,
                                "v": v,
                                "height_m": 6.0,
                                "canopy_diameter_m": 10.0,
                            },
                        }
                    ],
                    "requested_result": requested,
                }
                submitted = client.post(
                    f"/api/v1/scenarios/{scenario_id}/edits",
                    json=body,
                    headers={"Idempotency-Key": f"wave1-{label}"},
                )
                assert submitted.status_code == 202, submitted.text
                job_id = submitted.json()["job_id"]
                detail = wait_for_job(client, job_id, timeout=300.0)
                assert detail["status"] == "complete", detail

                store = app.state.context.store
                scenario = store.get_scenario(scenario_id)
                record = store.get_result(scenario.scenario_id, 1)
                assert record is not None
                manifest = record.manifest
                arrays = patch_codec.decode_payload(manifest, record.payload_bytes())
                payloads[label] = (manifest, arrays, detail)

            manifest_prefix, arrays_prefix, detail_prefix = payloads["prefix"]
            manifest_full, arrays_full, _ = payloads["full"]

            assert list(manifest_prefix["time_indices"]) == [12]
            assert list(manifest_full["time_indices"]) == list(
                range(science_site.cache_a.time_steps)
            )

            # THE parity gate: the served step-12 arrays are bit-identical.
            for name in ("utci", "tmrt"):
                assert arrays_prefix[name].shape[0] == 1
                assert np.array_equal(
                    arrays_prefix[name][0],
                    arrays_full[name][12],
                    equal_nan=True,
                ), f"{name}: served step 12 differs between [12] and full-day jobs"

            metrics = manifest_prefix["metrics"]
            assert "window_fraction" in metrics  # unchanged field, still present
            assert "read_window_fraction" in metrics
            assert 0.0 < metrics["read_window_fraction"] <= 1.0
            assert metrics["read_window_fraction"] >= metrics["window_fraction"]
            # The parity gate is only meaningful on the WINDOWED local path:
            # a bootstrap full-tile run would trivially match itself.
            assert metrics["window_fraction"] < 1.0
            assert "svf_seconds" in metrics
            assert "time_loop_seconds" in metrics
            assert metrics["svf_seconds"] >= 0.0
            assert metrics["time_loop_seconds"] > 0.0

            job_metrics = detail_prefix["metrics"]
            assert "read_window_fraction" in job_metrics
            assert "svf_seconds" in job_metrics
            assert "time_loop_seconds" in job_metrics
            # Worker path, not the bootstrap full-tile run (its metrics
            # carry "bootstrap": true and read_window_fraction 1.0).
            assert "bootstrap" not in job_metrics


def _world_to_uv(science_site, position_m) -> tuple[float, float]:
    from solweig_gpu.server.store import world_to_uv

    grid = science_site.grid
    x_m, y_m = position_m
    return world_to_uv(
        x_m,
        y_m,
        rows=grid.rows,
        cols=grid.cols,
        pixel_size_m=grid.pixel_size_m,
        origin_x_m=grid.origin_x_m,
        origin_y_m=grid.origin_y_m,
    )
