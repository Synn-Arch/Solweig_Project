# SPDX-License-Identifier: GPL-3.0-only
"""T2 tests for window-typed raster I/O and the windowed SOLWEIG core (P1).

Fast tests cover the window I/O contract (crop-equals-slice, outside-window
invariance on write, edge clamping) and the core's argument validation.
The bitwise full-window equivalence gate (WIN-002) runs against the real
500x500 fixture and is therefore executed as the documented gate command
(see docs/incremental_design_tool/agent/WORKLOG.md), not as part of the
default unit suite.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import pytest
import rasterio
from rasterio.transform import from_origin

from solweig_gpu.incremental.geometry import RasterWindow
from solweig_gpu.incremental.io import (
    copy_window,
    read_window,
    read_window_like,
    window_bounds_world,
    write_window,
)

ROWS = COLS = 64
PIXEL = 2.0
ORIGIN_X = 621734.7066
ORIGIN_Y = 3354614.3479


def _write_grid(path: Path, array: np.ndarray) -> None:
    profile = {
        "driver": "GTiff",
        "height": array.shape[0],
        "width": array.shape[1],
        "count": 1,
        "dtype": "float32",
        "transform": from_origin(ORIGIN_X, ORIGIN_Y, PIXEL, PIXEL),
        "nodata": None,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(np.float32), 1)


@pytest.fixture()
def ramp_path(tmp_path: Path) -> Path:
    path = tmp_path / "ramp.tif"
    _write_grid(path, np.arange(ROWS * COLS, dtype=np.float64).reshape(ROWS, COLS))
    return path


class TestReadWindow:
    def test_crop_equals_slice(self, ramp_path: Path) -> None:
        """Window read must be identical to slicing the full array."""
        full = read_window(ramp_path, RasterWindow(0, ROWS, 0, COLS))
        for window in (
            RasterWindow(0, 10, 0, 10),
            RasterWindow(7, 31, 5, 47),
            RasterWindow(ROWS - 1, ROWS, COLS - 3, COLS),
        ):
            got = read_window(ramp_path, window)
            expected = full[window.row_start:window.row_stop, window.col_start:window.col_stop]
            assert got.shape == (
                window.row_stop - window.row_start,
                window.col_stop - window.col_start,
            )
            np.testing.assert_array_equal(got, expected)

    def test_empty_window_after_clamp_raises_or_empty(self, ramp_path: Path) -> None:
        """A fully out-of-extent window clamps to empty and reads empty."""
        got = read_window(ramp_path, RasterWindow(ROWS + 5, ROWS + 9, 0, 4))
        assert got.shape == (0, 4)

    def test_clamp_at_edge(self, ramp_path: Path) -> None:
        """Halo overhang past the domain edge clamps to the raster extent."""
        full = read_window(ramp_path, RasterWindow(0, ROWS, 0, COLS))
        got = read_window(ramp_path, RasterWindow(ROWS - 2, ROWS + 6, COLS - 2, COLS + 6))
        np.testing.assert_array_equal(got, full[ROWS - 2:, COLS - 2:])

    def test_read_window_like_shape_mismatch(self, ramp_path: Path) -> None:
        with pytest.raises(ValueError, match="does not match reference"):
            read_window_like(
                ramp_path,
                RasterWindow(0, 8, 0, 8),
                reference=np.zeros((3, 3)),
            )


class TestWriteWindow:
    def test_only_window_touched(self, ramp_path: Path) -> None:
        """Writing a window must not modify any pixel outside it."""
        before = read_window(ramp_path, RasterWindow(0, ROWS, 0, COLS))
        window = RasterWindow(10, 20, 30, 44)
        patch = np.full((10, 14), -7.0, dtype=np.float32)
        write_window(ramp_path, window, patch)

        after = read_window(ramp_path, RasterWindow(0, ROWS, 0, COLS))
        np.testing.assert_array_equal(after[window.row_start:window.row_stop,
                                           window.col_start:window.col_stop], patch)
        outside = np.ones_like(before, dtype=bool)
        outside[window.row_start:window.row_stop, window.col_start:window.col_stop] = False
        assert outside.any()
        np.testing.assert_array_equal(after[outside], before[outside])

    def test_shape_mismatch_rejected(self, ramp_path: Path) -> None:
        with pytest.raises(ValueError, match="does not match window shape"):
            write_window(ramp_path, RasterWindow(0, 4, 0, 4), np.zeros((5, 5)))

    def test_copy_window_bitwise(self, ramp_path: Path, tmp_path: Path) -> None:
        dst = tmp_path / "copy.tif"
        _write_grid(dst, np.zeros((ROWS, COLS)))
        window = RasterWindow(5, 15, 6, 16)
        copy_window(ramp_path, dst, window)
        src_arr = read_window(ramp_path, window)
        dst_arr = read_window(dst, window)
        np.testing.assert_array_equal(src_arr, dst_arr)
        assert (read_window(dst, RasterWindow(0, 5, 0, COLS)) == 0).all()


class TestWindowBounds:
    def test_world_bounds(self, ramp_path: Path) -> None:
        window = RasterWindow(0, 10, 0, 10)
        left, bottom, right, top = window_bounds_world(ramp_path, window)
        assert left == pytest.approx(ORIGIN_X)
        assert right == pytest.approx(ORIGIN_X + 10 * PIXEL)
        assert top == pytest.approx(ORIGIN_Y)
        assert bottom == pytest.approx(ORIGIN_Y - 10 * PIXEL)


class TestRunUtciWindowContract:
    """Validation-only tests for the windowed core (no scientific compute)."""

    def test_import(self) -> None:
        from solweig_gpu.utci_process import run_utci_window  # noqa: F401

    def test_unknown_variable_rejected(self) -> None:
        from solweig_gpu.utci_process import run_utci_window

        with pytest.raises(ValueError, match="unknown requested variables"):
            run_utci_window(
                a=None, temp1=None, temp2=None, walls=None, dirwalls=None,
                svf_bundle=None, met_file=np.zeros((4, 30)),
                altitude=None, azimuth=None, zen=None, jday=None,
                dectime=np.zeros(4), altmax=None, location={}, scale=0.5,
                requested_variables=("utci", "banana"),
            )

    def test_time_range_validated(self) -> None:
        from solweig_gpu.utci_process import run_utci_window

        with pytest.raises(ValueError, match="invalid time range"):
            run_utci_window(
                a=None, temp1=None, temp2=None, walls=None, dirwalls=None,
                svf_bundle=None, met_file=np.zeros((4, 30)),
                altitude=None, azimuth=None, zen=None, jday=None,
                dectime=np.zeros(4), altmax=None, location={}, scale=0.5,
                time_start=3, time_stop=3,
            )

    def test_out_window_validated_before_any_compute(self) -> None:
        from solweig_gpu.utci_process import run_utci_window

        # out_window is checked against the tensor shape before the heavy
        # pre-loop runs, so a malformed window fails fast.
        with pytest.raises(ValueError, match="outside tensor shape"):
            run_utci_window(
                a=torch.zeros((4, 5)), temp1=None, temp2=None, walls=None,
                dirwalls=None, svf_bundle=None, met_file=np.zeros((4, 30)),
                altitude=None, azimuth=None, zen=None, jday=None,
                dectime=np.zeros(4), altmax=None, location={}, scale=0.5,
                out_window=(2, 9, 0, 4),  # row_stop beyond the 4-row tensor
            )
