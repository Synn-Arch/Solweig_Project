# SPDX-License-Identifier: GPL-3.0-only
"""Pure-numpy comparison metrics for the T3 scientific differential suite.

Conventions (docs/incremental_design_tool/agent/validation_pipeline.md):
- absolute errors are computed in float64 upcast;
- quantiles use ``np.quantile`` over the absolute-error array;
- boundary rings and distance bands are measured inward from the
  write-window edge (region definitions may overlap; they are diagnostic
  regions, not partitions);
- the outside-window region uses exact-equality semantics, never tolerance;
- integer masks / patch layout are always compared exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from solweig_gpu.incremental.geometry import RasterWindow


@dataclass(frozen=True, slots=True)
class RegionMetrics:
    n_compared: int
    max_abs_error: float
    mean_abs_error: float
    rmse: float
    abs_error_p50: float
    abs_error_p95: float
    abs_error_p99: float
    count_above_tolerance: int
    fraction_above_tolerance: float

    def to_dict(self) -> dict:
        return {
            "n_compared": self.n_compared,
            "max_abs_error": self.max_abs_error,
            "mean_abs_error": self.mean_abs_error,
            "rmse": self.rmse,
            "abs_error_p50": self.abs_error_p50,
            "abs_error_p95": self.abs_error_p95,
            "abs_error_p99": self.abs_error_p99,
            "count_above_tolerance": self.count_above_tolerance,
            "fraction_above_tolerance": self.fraction_above_tolerance,
        }


@dataclass(frozen=True, slots=True)
class BinaryRegionMetrics:
    n_compared: int
    mismatch_count: int
    mismatch_fraction: float

    def to_dict(self) -> dict:
        return {
            "n_compared": self.n_compared,
            "mismatch_count": self.mismatch_count,
            "mismatch_fraction": self.mismatch_fraction,
        }


def continuous_region_metrics(
    candidate: np.ndarray,
    oracle: np.ndarray,
    valid: np.ndarray,
    tolerance: float,
) -> RegionMetrics:
    """Error statistics over valid cells between candidate and oracle."""
    candidate = np.asarray(candidate)
    oracle = np.asarray(oracle)
    valid = np.asarray(valid, dtype=bool)
    if candidate.shape != oracle.shape or candidate.shape != valid.shape:
        raise ValueError(
            f"shape mismatch: candidate {candidate.shape}, oracle {oracle.shape}, "
            f"valid {valid.shape}"
        )
    if not valid.any():
        raise ValueError("valid mask is empty")
    abs_error = np.abs(
        candidate.astype(np.float64)[valid] - oracle.astype(np.float64)[valid]
    )
    above = int((abs_error > tolerance).sum())
    return RegionMetrics(
        n_compared=int(abs_error.size),
        max_abs_error=float(abs_error.max()),
        mean_abs_error=float(abs_error.mean()),
        rmse=float(np.sqrt(np.mean(abs_error ** 2))),
        abs_error_p50=float(np.quantile(abs_error, 0.50)),
        abs_error_p95=float(np.quantile(abs_error, 0.95)),
        abs_error_p99=float(np.quantile(abs_error, 0.99)),
        count_above_tolerance=above,
        fraction_above_tolerance=float(above / abs_error.size),
    )


def binary_region_metrics(
    candidate: np.ndarray,
    oracle: np.ndarray,
    valid: np.ndarray,
) -> BinaryRegionMetrics:
    """Disagreement statistics for binary planes (e.g. shadow)."""
    candidate = np.asarray(candidate)
    oracle = np.asarray(oracle)
    valid = np.asarray(valid, dtype=bool)
    if candidate.shape != oracle.shape or candidate.shape != valid.shape:
        raise ValueError("shape mismatch in binary comparison")
    if not valid.any():
        raise ValueError("valid mask is empty")
    mismatch = candidate.astype(bool)[valid] != oracle.astype(bool)[valid]
    count = int(mismatch.sum())
    return BinaryRegionMetrics(
        n_compared=int(mismatch.size),
        mismatch_count=count,
        mismatch_fraction=float(count / mismatch.size),
    )


def inward_edge_distance(
    grid_shape: tuple[int, int],
    write_window: RasterWindow,
) -> np.ndarray:
    """Distance (cells) from each cell to the nearest write-window edge.

    Inside the window: ``min(i - r0, r1-1 - i, j - c0, c1-1 - j) >= 0``.
    Outside the window: ``-1`` (excluded from every inward region).
    """
    rows, cols = grid_shape
    r0 = max(0, min(write_window.row_start, rows))
    r1 = max(r0, min(write_window.row_stop, rows))
    c0 = max(0, min(write_window.col_start, cols))
    c1 = max(c0, min(write_window.col_stop, cols))

    row_idx = np.arange(rows)[:, None]
    col_idx = np.arange(cols)[None, :]
    row_dist = np.minimum(row_idx - r0, (r1 - 1) - row_idx)
    col_dist = np.minimum(col_idx - c0, (c1 - 1) - col_idx)
    inside = (row_dist >= 0) & (col_dist >= 0)
    return np.where(inside, np.minimum(row_dist, col_dist), -1)


def outside_window_mask(
    grid_shape: tuple[int, int],
    write_window: RasterWindow,
) -> np.ndarray:
    """Boolean mask of cells strictly outside the half-open write window."""
    return inward_edge_distance(grid_shape, write_window) < 0


def boundary_ring_mask(
    grid_shape: tuple[int, int],
    write_window: RasterWindow,
    *,
    ring_inner: int = 0,
    ring_outer: int = 2,
) -> np.ndarray:
    """Band of cells just inside the write-window edge.

    ``ring_inner=0, ring_outer=2`` selects the outermost 2 cells of the
    window on each side (the seam where a local solve meets stored results).
    """
    if ring_inner < 0 or ring_outer <= ring_inner:
        raise ValueError("require ring_outer > ring_inner >= 0")
    dist = inward_edge_distance(grid_shape, write_window)
    return (dist >= ring_inner) & (dist <= ring_outer)


def distance_band_masks(
    grid_shape: tuple[int, int],
    write_window: RasterWindow,
    *,
    bands: tuple[tuple[int, int], ...] = ((0, 2), (3, 8), (9, 16)),
) -> list[tuple[str, np.ndarray]]:
    """Concentric distance bands measured inward from the write-window edge.

    Returns ``(label, mask)`` pairs for each band plus the ``"interior"``
    band (deeper than the outermost band edge).
    """
    dist = inward_edge_distance(grid_shape, write_window)
    result: list[tuple[str, np.ndarray]] = []
    deepest = 0
    for low, high in bands:
        if low < 0 or high < low:
            raise ValueError(f"invalid band ({low}, {high})")
        result.append((f"{low}-{high}", (dist >= low) & (dist <= high)))
        deepest = max(deepest, high)
    result.append(("interior", dist > deepest))
    return result


@dataclass(frozen=True, slots=True)
class ComparisonReport:
    variable: str
    inside: RegionMetrics | BinaryRegionMetrics
    boundary_ring: RegionMetrics | BinaryRegionMetrics
    outside: RegionMetrics | BinaryRegionMetrics | None

    def to_dict(self) -> dict:
        return {
            "variable": self.variable,
            "inside": self.inside.to_dict(),
            "boundary_ring": self.boundary_ring.to_dict(),
            "outside": self.outside.to_dict() if self.outside is not None else None,
        }


def _window_mask(grid_shape: tuple[int, int], write_window: RasterWindow) -> np.ndarray:
    return inward_edge_distance(grid_shape, write_window) >= 0


def compare_continuous(
    candidate: np.ndarray,
    oracle: np.ndarray,
    valid: np.ndarray,
    write_window: RasterWindow,
    tolerance: float,
    *,
    ring: tuple[int, int] = (0, 2),
) -> ComparisonReport:
    """Continuous-variable report: inside / boundary ring / outside regions.

    The inside and boundary-ring regions intentionally overlap (the ring is
    a subset of the window); each is reported independently so seam-localised
    errors are visible without diluting the window-wide statistics.
    """
    shape = candidate.shape
    window_mask = _window_mask(shape, write_window)
    ring_mask = boundary_ring_mask(
        shape, write_window, ring_inner=ring[0], ring_outer=ring[1]
    )
    outside_mask = outside_window_mask(shape, write_window)
    outside_report = None
    if (outside_mask & valid).any():
        outside_report = continuous_region_metrics(
            candidate, oracle, outside_mask & valid, tolerance
        )
    return ComparisonReport(
        variable="continuous",
        inside=continuous_region_metrics(
            candidate, oracle, window_mask & valid, tolerance
        ),
        boundary_ring=continuous_region_metrics(
            candidate, oracle, ring_mask & valid, tolerance
        ),
        outside=outside_report,
    )


def compare_binary(
    candidate: np.ndarray,
    oracle: np.ndarray,
    valid: np.ndarray,
    write_window: RasterWindow,
    *,
    ring: tuple[int, int] = (0, 2),
) -> ComparisonReport:
    """Binary-plane report (shadow) with mismatch fractions per region."""
    shape = candidate.shape
    window_mask = _window_mask(shape, write_window)
    ring_mask = boundary_ring_mask(
        shape, write_window, ring_inner=ring[0], ring_outer=ring[1]
    )
    outside_mask = outside_window_mask(shape, write_window)
    outside_report = None
    if (outside_mask & valid).any():
        outside_report = binary_region_metrics(
            candidate, oracle, outside_mask & valid
        )
    return ComparisonReport(
        variable="binary",
        inside=binary_region_metrics(candidate, oracle, window_mask & valid),
        boundary_ring=binary_region_metrics(candidate, oracle, ring_mask & valid),
        outside=outside_report,
    )


def error_by_distance_band(
    candidate: np.ndarray,
    oracle: np.ndarray,
    valid: np.ndarray,
    write_window: RasterWindow,
    tolerance: float,
    *,
    bands: tuple[tuple[int, int], ...] = ((0, 2), (3, 8), (9, 16)),
) -> dict[str, RegionMetrics]:
    """Per-band error metrics to localise seam problems."""
    result: dict[str, RegionMetrics] = {}
    for label, mask in distance_band_masks(candidate.shape, write_window, bands=bands):
        band_valid = mask & valid
        if band_valid.any():
            result[label] = continuous_region_metrics(
                candidate, oracle, band_valid, tolerance
            )
    return result


def assert_outside_window_unchanged(
    before: np.ndarray,
    after: np.ndarray,
    write_window: RasterWindow,
) -> None:
    """Exact equality outside the write window; names the first mismatch."""
    if before.shape != after.shape:
        raise AssertionError(f"shape change: {before.shape} -> {after.shape}")
    mask = outside_window_mask(before.shape, write_window)
    if not mask.any():
        return
    if np.array_equal(before[mask], after[mask]):
        return
    flat_before = before[mask]
    flat_after = after[mask]
    bad = np.flatnonzero(flat_before != flat_after)
    raise AssertionError(
        f"{bad.size} cells changed outside the write window; first flat index "
        f"{int(bad[0])}: {flat_before[bad[0]]} -> {flat_after[bad[0]]}"
    )
