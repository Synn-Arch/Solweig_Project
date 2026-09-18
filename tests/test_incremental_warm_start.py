# SPDX-License-Identifier: GPL-3.0-only
"""G2.1 warm-start plumbing: default-None bit-identity + warm==cold parity.

Design: ``docs/incremental_design_tool/realtime_collaboration/design/
r5-temporal-checkpoints-and-fast-serve.md`` section G2 (binding).

The seam under test is the additive ``initial_state`` /
``return_final_state`` / ``time_start`` plumbing on
``run_utci_window`` / ``solve_window`` / ``run_full_tile``. Two fences are
HARD here:

- default-None bit-identity: the flags alone must not perturb the cold
  path by a single bit;
- warm==cold parity: ``warm(k..N, state@k)`` is bitwise
  ``cold(0..N)[k..N]`` — the carried state is exactly the loop's
  cross-timestep dependency (the six thermal planes + CI/firstdaytime/
  timeadd/Twater; ``timestepdec`` is derived per run, never carried).
  The day-spanning case exercises the midnight rows that recompute
  ``CI`` and (under land cover) ``Twater`` — the exact hazard that makes
  a naive ``time_start > 0`` warm start silently wrong science.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from solweig_gpu.incremental.geometry import RasterWindow
from solweig_gpu.incremental.solver import (
    SolverInputError,
    run_full_tile,
    solve_window,
)
from solweig_gpu.incremental.trees import TreeLayer
from solweig_gpu.incremental.worker import ExactWorker

from tests.test_incremental_worker import (
    DATE_STR,
    LOCAL_ALWAYS,
    TINY_COLS,
    TINY_EPSG,
    TINY_ORIGIN,
    TINY_PIXEL,
    TINY_ROWS,
    TINY_TREE,
    _build_cache,
    _compute_baseline_svf,
    _make_prepared_site,
)

_THERMAL_PLANES = (
    "Tgmap1", "Tgmap1E", "Tgmap1S", "Tgmap1W", "Tgmap1N", "TgOut1",
)


def _worker_with_hours(
    tmp_path: Path, met_hours
):
    """A real-physics worker whose site spans the given met hours.

    Unlike ``_make_tiny_site`` (random SVF numbers, 16 fake patches — its
    cache can never feed the physics), this stages a prepared site with the
    REAL baseline SVF (``_compute_baseline_svf``), so ``solve_window``'s
    no-edit cache-slice path serves the oracle's own 153-patch bundle and
    every array that leaves the seam is physics, not fixture noise.
    """
    grid, site = _make_prepared_site(
        tmp_path / "site",
        rows=TINY_ROWS,
        cols=TINY_COLS,
        pixel=TINY_PIXEL,
        origin=TINY_ORIGIN,
        epsg=TINY_EPSG,
        base_trees=(TINY_TREE,),
        met_hours=met_hours,
    )
    _compute_baseline_svf(site)
    cache = _build_cache(
        site,
        tmp_path / "cache",
        met_path=site / "metfiles" / f"metfile_0_0_{DATE_STR}.txt",
        site_id="warm",
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
    return worker, site, grid


def _solve(worker, grid, *, time_start=0, time_stop=None, initial_state=None,
           return_final_state=False, write_window=None):
    write = write_window if write_window is not None else grid.full_window
    return solve_window(
        worker.cache,
        worker.layer,
        read_window=grid.full_window,
        write_window=write,
        forcing=worker.forcing(),
        requested_variables=("utci", "tmrt", "shadow"),
        time_start=time_start,
        time_stop=time_stop,
        initial_state=initial_state,
        return_final_state=return_final_state,
    )


def _assert_bitwise_equal(actual: dict, expected: dict) -> None:
    assert sorted(actual) == sorted(expected)
    for name in expected:
        got, want = actual[name], expected[name]
        assert got.dtype == want.dtype == np.float32
        assert got.shape == want.shape, name
        # Byte equality: NaN payloads and signed zeros must match exactly.
        assert got.tobytes() == want.tobytes(), name


class TestDefaultNoneBitIdentity:
    def test_flags_alone_do_not_perturb_the_cold_path(self, tmp_path: Path) -> None:
        worker, site, grid = _worker_with_hours(tmp_path, range(10, 13))
        cold = _solve(worker, grid)
        flagged = _solve(worker, grid, return_final_state=True)
        if isinstance(flagged, tuple):
            flagged = flagged[0]
        _assert_bitwise_equal(flagged, cold)

    def test_captured_state_shape_and_vocabulary(self, tmp_path: Path) -> None:
        worker, site, grid = _worker_with_hours(tmp_path, range(10, 13))
        total = worker.forcing().time_steps
        outputs, state = _solve(
            worker, grid, time_stop=total, return_final_state=True
        )
        assert set(state) == set(_THERMAL_PLANES) | {
            "CI", "firstdaytime", "timeadd", "Twater", "next_step",
        }
        assert state["next_step"] == total
        for name in _THERMAL_PLANES:
            plane = state[name]
            assert isinstance(plane, torch.Tensor)
            assert plane.dtype == torch.float32
            assert tuple(plane.shape) == (grid.rows, grid.cols)
        for name in ("CI", "firstdaytime", "timeadd"):
            assert isinstance(state[name], float)


class TestWarmEqualsColdParity:
    def test_full_tile_suffix_bitwise(self, tmp_path: Path) -> None:
        worker, site, grid = _worker_with_hours(tmp_path, range(10, 13))
        cold = _solve(worker, grid)
        k = 1  # mid-series resume (after the first daytime step)
        _, state_k = _solve(worker, grid, time_stop=k, return_final_state=True)
        warm = _solve(worker, grid, time_start=k, initial_state=state_k)
        _assert_bitwise_equal(
            warm, {name: array[k:] for name, array in cold.items()}
        )

    def test_warm_across_day_boundary_bitwise(self, tmp_path: Path) -> None:
        # 27 hourly rows = day 172 (rows 0..23) + day 173 (rows 24..26).
        # Row 0 is a midnight row (CI recomputed, Twater established under
        # land cover); row 24 is the NEXT midnight row inside the warm
        # suffix — the warm run must recompute CI/Twater there exactly as
        # the cold run did.
        worker, site, grid = _worker_with_hours(tmp_path, range(0, 27))
        cold = _solve(worker, grid)
        k = 20
        _, state_k = _solve(worker, grid, time_stop=k, return_final_state=True)
        warm = _solve(worker, grid, time_start=k, initial_state=state_k)
        _assert_bitwise_equal(
            warm, {name: array[k:] for name, array in cold.items()}
        )

    def test_windowed_suffix_bitwise(self, tmp_path: Path) -> None:
        # The write-window crop property: a windowed warm solve is the
        # crop of the full-tile warm trajectory, itself bitwise the cold
        # trajectory (the thermal chain is elementwise).
        worker, site, grid = _worker_with_hours(tmp_path, range(10, 13))
        write = RasterWindow(32, 96, 32, 96)
        cold_windowed = _solve(worker, grid, write_window=write)
        k = 1
        _, state_k = _solve(worker, grid, time_stop=k, return_final_state=True)
        warm_windowed = _solve(
            worker, grid, time_start=k, initial_state=state_k,
            write_window=write,
        )
        _assert_bitwise_equal(
            warm_windowed,
            {name: array[k:] for name, array in cold_windowed.items()},
        )

    def test_full_tile_capture_matches_cold_state(self, tmp_path: Path) -> None:
        # The capture route the worker uses (run_full_tile) must land on
        # the same state the windowed seam captures — and its OUTPUTS must
        # be bitwise the legacy orchestrator's (the documented
        # replication license for routing capture through the direct
        # path).
        worker, site, grid = _worker_with_hours(tmp_path, range(10, 13))
        legacy = run_full_tile(
            worker.cache,
            worker.layer,
            forcing=worker.forcing(),
            site_dir=site,
            scratch_dir=Path(tmp_path) / "scratch_legacy",
            requested_variables=("utci", "tmrt", "shadow"),
        )
        direct, state = run_full_tile(
            worker.cache,
            worker.layer,
            forcing=worker.forcing(),
            site_dir=site,
            scratch_dir=Path(tmp_path) / "scratch_capture",
            requested_variables=("utci", "tmrt", "shadow"),
            return_final_state=True,
        )
        _assert_bitwise_equal(direct, legacy)
        _, state_seam = _solve(
            worker, grid, time_stop=worker.forcing().time_steps,
            return_final_state=True,
        )
        for name in _THERMAL_PLANES:
            assert (
                state[name].detach().cpu().numpy().tobytes()
                == state_seam[name].detach().cpu().numpy().tobytes()
            ), name
        for name in ("CI", "firstdaytime", "timeadd", "next_step"):
            assert state[name] == state_seam[name]
        assert state["Twater"] == state_seam["Twater"]


class TestWarmStartFences:
    def test_next_step_must_match_time_start(self, tmp_path: Path) -> None:
        worker, site, grid = _worker_with_hours(tmp_path, range(10, 13))
        _, state_k = _solve(worker, grid, time_stop=1, return_final_state=True)
        tampered = dict(state_k)
        tampered["next_step"] = 2
        with pytest.raises((SolverInputError, ValueError), match="next_step"):
            _solve(worker, grid, time_start=1, initial_state=tampered)

    def test_full_tile_shape_required_windowed_slice_refuses_ragged(
        self, tmp_path: Path
    ) -> None:
        worker, site, grid = _worker_with_hours(tmp_path, range(10, 13))
        _, state_k = _solve(worker, grid, time_stop=1, return_final_state=True)
        ragged = dict(state_k)
        ragged["TgOut1"] = state_k["TgOut1"][:-1, :-1].contiguous()
        with pytest.raises(SolverInputError, match="full-tile"):
            _solve(worker, grid, time_start=1, initial_state=ragged)

    def test_missing_plane_refused_loudly(self, tmp_path: Path) -> None:
        worker, site, grid = _worker_with_hours(tmp_path, range(10, 13))
        _, state_k = _solve(worker, grid, time_stop=1, return_final_state=True)
        partial = {
            name: plane for name, plane in state_k.items()
            if name != "TgOut1"
        }
        with pytest.raises((SolverInputError, ValueError), match="TgOut1"):
            _solve(worker, grid, time_start=1, initial_state=partial)

    def test_time_start_out_of_range_refused(self, tmp_path: Path) -> None:
        worker, site, grid = _worker_with_hours(tmp_path, range(10, 13))
        with pytest.raises(SolverInputError, match="time_start"):
            _solve(worker, grid, time_start=99)
