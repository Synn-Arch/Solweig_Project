# SPDX-License-Identifier: GPL-3.0-only
"""T1-level unit tests for the T3 differential metrics on synthetic arrays.

These validate the measurement machinery itself (masks, rings, bands,
compare functions, exact outside-window semantics). Real-oracle differential
scenarios S01-S12 land with the P5 worker.
"""

from __future__ import annotations

import numpy as np
import pytest

from solweig_gpu.incremental.geometry import RasterWindow

from .metrics import (
    assert_outside_window_unchanged,
    binary_region_metrics,
    boundary_ring_mask,
    compare_binary,
    compare_continuous,
    continuous_region_metrics,
    distance_band_masks,
    error_by_distance_band,
    inward_edge_distance,
    outside_window_mask,
)
from .tolerances import BOUNDARY_RING_CELLS, DISTANCE_BANDS

GRID = (24, 32)
WINDOW = RasterWindow(8, 20, 12, 28)  # rows [8,20), cols [12,28)


class TestInwardEdgeDistance:
    def test_zero_on_window_border(self) -> None:
        dist = inward_edge_distance(GRID, WINDOW)
        assert dist[8, 12] == 0
        assert dist[8, 27] == 0
        assert dist[19, 12] == 0
        assert dist[19, 27] == 0

    def test_min_of_axis_distances_inside(self) -> None:
        dist = inward_edge_distance(GRID, WINDOW)
        assert dist[10, 20] == 2  # min(2, 9, 8, 7)
        assert dist[14, 13] == 1  # min(6, 5, 1, 14)
        assert dist[9, 26] == 1

    def test_outside_is_minus_one(self) -> None:
        dist = inward_edge_distance(GRID, WINDOW)
        assert dist[0, 0] == -1
        assert dist[8, 11] == -1
        assert dist[20, 12] == -1
        assert (dist[:8, :] == -1).all()
        assert (dist[:, 28:] == -1).all()

    def test_window_covering_grid(self) -> None:
        full = RasterWindow(0, 24, 0, 32)
        dist = inward_edge_distance(GRID, full)
        assert dist[0, 0] == 0
        assert dist[12, 16] == 11  # min(12, 11, 16, 15)
        assert (dist >= 0).all()


class TestMasks:
    def test_outside_mask_complements_window(self) -> None:
        outside = outside_window_mask(GRID, WINDOW)
        inside = ~outside
        assert inside.sum() == 12 * 16
        assert outside[:8, :].all()
        assert inside[8:20, 12:28].all()

    def test_ring_is_outer_two_cells_of_window(self) -> None:
        ring = boundary_ring_mask(GRID, WINDOW, ring_inner=0, ring_outer=2)
        assert ring[8, 12] and ring[9, 13] and ring[8, 19]
        assert not ring[11, 15]  # 3 cells from every edge
        assert ring.sum() == 12 * 16 - 6 * 10  # interior erodes by 3 per side

    def test_ring_custom_width(self) -> None:
        ring = boundary_ring_mask(GRID, WINDOW, ring_inner=2, ring_outer=4)
        assert not ring[8, 12]  # distance 0 not in [2, 4]
        assert ring[10, 14]  # distance 2
        assert not ring[13, 17]  # distance 5

    def test_ring_rejects_bad_geometry(self) -> None:
        with pytest.raises(ValueError):
            boundary_ring_mask(GRID, WINDOW, ring_inner=3, ring_outer=3)
        with pytest.raises(ValueError):
            boundary_ring_mask(GRID, WINDOW, ring_inner=-1, ring_outer=2)

    def test_distance_bands_partition_window(self) -> None:
        masks = dict(distance_band_masks(GRID, WINDOW, bands=DISTANCE_BANDS))
        union = np.zeros(GRID, dtype=bool)
        for label, mask in masks.items():
            assert not (union & mask).any(), f"band {label} overlaps earlier bands"
            union |= mask
        window = ~outside_window_mask(GRID, WINDOW)
        assert (union == window).all()
        assert masks["0-2"][8, 12]
        assert masks["3-8"][13, 20]  # distance 5
        # Window is 12x16: max inward distance is 5, so 9-16 and interior are empty.
        assert not masks["9-16"].any()
        assert not masks["interior"].any()

    def test_distance_bands_on_full_grid(self) -> None:
        full = RasterWindow(0, 24, 0, 32)
        masks = dict(distance_band_masks(GRID, full, bands=DISTANCE_BANDS))
        assert masks["0-2"][0, 0]
        assert masks["3-8"][3, 4]
        assert masks["9-16"][9, 9]
        assert masks["9-16"][11, 15]
        assert not masks["interior"].any()  # max distance is 11

    def test_interior_band_beyond_deepest(self) -> None:
        masks = dict(distance_band_masks(GRID, WINDOW, bands=((0, 2),)))
        assert set(masks) == {"0-2", "interior"}
        assert masks["interior"][13, 18]  # distance 5 > 2
        assert not masks["interior"][8, 12]


class TestContinuousMetrics:
    def test_zero_error_when_identical(self) -> None:
        rng = np.random.default_rng(3)
        oracle = rng.normal(size=GRID).astype(np.float32)
        valid = np.ones(GRID, dtype=bool)
        m = continuous_region_metrics(oracle, oracle, valid, tolerance=0.0)
        assert m.n_compared == 24 * 32
        assert m.max_abs_error == 0.0
        assert m.fraction_above_tolerance == 0.0

    def test_statistics_on_known_error(self) -> None:
        oracle = np.zeros(GRID)
        candidate = np.zeros(GRID)
        candidate[0, 0] = 2.0
        candidate[1, 1] = 4.0
        valid = np.ones(GRID, dtype=bool)
        m = continuous_region_metrics(candidate, oracle, valid, tolerance=1.0)
        assert m.max_abs_error == 4.0
        assert m.mean_abs_error == pytest.approx(6.0 / (24 * 32))
        assert m.count_above_tolerance == 2
        assert m.abs_error_p50 == 0.0

    def test_valid_mask_restricts_comparison(self) -> None:
        oracle = np.zeros(GRID)
        candidate = np.full(GRID, 5.0)
        valid = np.zeros(GRID, dtype=bool)
        valid[4, 5] = True
        m = continuous_region_metrics(candidate, oracle, valid, tolerance=1.0)
        assert m.n_compared == 1
        assert m.max_abs_error == 5.0

    def test_float64_upcast_of_mixed_dtypes(self) -> None:
        oracle = np.full(GRID, 1.0 + 1e-6)  # float64
        candidate = np.full(GRID, 1.0, dtype=np.float32)
        valid = np.ones(GRID, dtype=bool)
        m = continuous_region_metrics(candidate, oracle, valid, tolerance=0.0)
        # A float32 subtraction would collapse this to 0; float64 keeps it.
        assert m.max_abs_error == pytest.approx(1e-6, rel=1e-3)

    def test_empty_valid_mask_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            continuous_region_metrics(
                np.zeros(GRID), np.zeros(GRID), np.zeros(GRID, dtype=bool), 1.0
            )

    def test_shape_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            continuous_region_metrics(
                np.zeros(GRID), np.zeros((5, 5)), np.ones(GRID, dtype=bool), 1.0
            )


class TestBinaryMetrics:
    def test_mismatch_fraction(self) -> None:
        oracle = np.zeros(GRID, dtype=np.uint8)
        candidate = np.zeros(GRID, dtype=np.uint8)
        candidate[3, 4] = 1
        candidate[10, 20] = 1
        valid = np.ones(GRID, dtype=bool)
        m = binary_region_metrics(candidate, oracle, valid)
        assert m.mismatch_count == 2
        assert m.mismatch_fraction == pytest.approx(2.0 / (24 * 32))

    def test_semantic_values_coerced(self) -> None:
        oracle = np.zeros(GRID, dtype=bool)
        candidate = np.zeros(GRID, dtype=np.float32)
        candidate[1, 1] = 0.5  # astype(bool) -> True
        valid = np.ones(GRID, dtype=bool)
        assert binary_region_metrics(candidate, oracle, valid).mismatch_count == 1


class TestCompareReports:
    def test_continuous_report_regions(self) -> None:
        rng = np.random.default_rng(5)
        oracle = rng.normal(size=GRID).astype(np.float32)
        candidate = oracle.copy()
        candidate[8, 12] += 100.0  # corner of window (ring + inside)
        valid = np.ones(GRID, dtype=bool)
        report = compare_continuous(candidate, oracle, valid, WINDOW, tolerance=1.0)
        d = report.to_dict()
        assert d["variable"] == "continuous"
        assert d["inside"]["count_above_tolerance"] == 1
        assert d["boundary_ring"]["count_above_tolerance"] == 1
        assert d["outside"]["max_abs_error"] == 0.0

    def test_binary_report_outside_zero(self) -> None:
        rng = np.random.default_rng(7)
        oracle = rng.integers(0, 2, size=GRID).astype(np.uint8)
        candidate = oracle.copy()
        candidate[15, 20] ^= 1  # deep inside window
        valid = np.ones(GRID, dtype=bool)
        report = compare_binary(candidate, oracle, valid, WINDOW)
        d = report.to_dict()
        assert d["inside"]["mismatch_count"] == 1
        assert d["outside"]["mismatch_count"] == 0

    def test_outside_none_when_window_is_full_grid(self) -> None:
        full = RasterWindow(0, 24, 0, 32)
        oracle = np.zeros(GRID)
        valid = np.ones(GRID, dtype=bool)
        report = compare_continuous(oracle.copy(), oracle, valid, full, 1.0)
        assert report.outside is None
        assert report.to_dict()["outside"] is None

    def test_error_by_distance_band_localises(self) -> None:
        oracle = np.zeros(GRID)
        candidate = np.zeros(GRID)
        candidate[10, 14] = 3.0  # distance 2 -> band 0-2
        candidate[13, 20] = 7.0  # distance 5 -> band 3-8
        valid = np.ones(GRID, dtype=bool)
        bands = error_by_distance_band(
            candidate, oracle, valid, WINDOW, 1.0, bands=DISTANCE_BANDS
        )
        assert set(bands) == {"0-2", "3-8"}
        assert bands["0-2"].max_abs_error == 3.0
        assert bands["3-8"].max_abs_error == 7.0


class TestOutsideWindowUnchanged:
    def test_passes_when_only_window_touched(self) -> None:
        rng = np.random.default_rng(11)
        before = rng.normal(size=GRID).astype(np.float32)
        after = before.copy()
        after[8:20, 12:28] += 1.0
        assert_outside_window_unchanged(before, after, WINDOW)

    def test_fails_and_names_first_mismatch(self) -> None:
        before = np.zeros(GRID, dtype=np.float32)
        after = before.copy()
        after[0, 0] = 9.0
        with pytest.raises(AssertionError, match="outside the write window"):
            assert_outside_window_unchanged(before, after, WINDOW)

    def test_nan_change_outside_detected(self) -> None:
        before = np.zeros(GRID, dtype=np.float32)
        after = before.copy()
        after[3, 3] = np.nan
        with pytest.raises(AssertionError):
            assert_outside_window_unchanged(before, after, WINDOW)

    def test_full_grid_window_is_noop(self) -> None:
        full = RasterWindow(0, 24, 0, 32)
        arr = np.ones(GRID)
        assert_outside_window_unchanged(arr, arr * 2, full)


class TestToleranceConstants:
    def test_ring_and_band_constants_match_spec(self) -> None:
        assert BOUNDARY_RING_CELLS == (0, 2)
        assert DISTANCE_BANDS == ((0, 2), (3, 8), (9, 16))
