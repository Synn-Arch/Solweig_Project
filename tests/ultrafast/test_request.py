"""Tests for solweig_core.request — logical domain vs physical windows.

RED witnesses (TASKS T02): wrong origin/shape rejected at the REQUEST
level — windows escaping the logical grid, a write window not contained in
the read window, inverted grids, GLOBAL time coverage beyond the forcing
series, unparseable dtype plans, empty profile identity, and devices
outside the declared vocabulary. GREEN guards pin the harness-compared
metadata (global time indices, window origin) the request must carry.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from solweig_core.request import (  # noqa: E402
    DEFAULT_DTYPE_PLAN,
    DEFAULT_OUTPUT_DTYPE_PLAN,
    PhysicalWindow,
    SolveRequestView,
    TimeCoverage,
)
from solweig_core.status import RefusalReason, RequestValidationError  # noqa: E402


def make_request(**overrides):
    """A valid request over a 100x80 domain; overrides applied last."""
    kwargs = dict(
        logical_domain_id="site:test:100x80",
        rows=100,
        cols=80,
        origin_x_m=1000.0,
        origin_y_m=2000.0,
        pixel_size_m=2.0,
        read_window=PhysicalWindow(0, 100, 0, 80),
        write_window=PhysicalWindow(10, 20, 10, 20),
        time=TimeCoverage(0, None, 24),
        profile_id="canonical_cpu_v1",
        device="cpu",
    )
    kwargs.update(overrides)
    return SolveRequestView(**kwargs)


# ---------------------------------------------------------------------------
# PhysicalWindow
# ---------------------------------------------------------------------------
class TestPhysicalWindow:
    def test_dimensions_and_origin(self):
        window = PhysicalWindow(5, 15, 7, 21)
        assert window.height == 10
        assert window.width == 14
        assert window.origin == (5, 7)
        assert window.list() == [5, 15, 7, 21]

    def test_valid_window_passes(self):
        PhysicalWindow(0, 10, 0, 10).validate(rows=10, cols=10)

    def test_window_escaping_grid_rejected(self):
        with pytest.raises(RequestValidationError, match="exceeds the logical domain"):
            PhysicalWindow(0, 101, 0, 80).validate(rows=100, cols=80)

    def test_column_escape_rejected(self):
        with pytest.raises(RequestValidationError, match="exceeds"):
            PhysicalWindow(0, 100, 0, 81).validate(rows=100, cols=80)

    def test_negative_origin_rejected(self):
        with pytest.raises(RequestValidationError, match="negative"):
            PhysicalWindow(-1, 10, 0, 10).validate(rows=100, cols=80)

    def test_empty_window_rejected(self):
        with pytest.raises(RequestValidationError, match="empty or inverted"):
            PhysicalWindow(5, 5, 0, 10).validate(rows=100, cols=80)

    def test_inverted_window_rejected(self):
        with pytest.raises(RequestValidationError, match="empty or inverted"):
            PhysicalWindow(10, 5, 0, 10).validate(rows=100, cols=80)

    def test_containment(self):
        outer = PhysicalWindow(0, 50, 0, 50)
        inner = PhysicalWindow(10, 20, 10, 20)
        poking = PhysicalWindow(10, 20, 10, 60)
        assert outer.contains(inner)
        assert not outer.contains(poking)
        assert not inner.contains(outer)


# ---------------------------------------------------------------------------
# TimeCoverage — GLOBAL time indices, never array offsets
# ---------------------------------------------------------------------------
class TestTimeCoverage:
    def test_none_stop_normalizes_to_total(self):
        coverage = TimeCoverage(0, None, 24)
        assert coverage.time_stop == 24

    def test_global_indices_are_timestep_numbers(self):
        coverage = TimeCoverage(5, 9, 24)
        assert coverage.global_indices == (5, 6, 7, 8)

    def test_stop_beyond_series_rejected(self):
        with pytest.raises(RequestValidationError, match="exceeds the forcing series"):
            TimeCoverage(0, 25, 24).validate()

    def test_empty_range_rejected(self):
        with pytest.raises(RequestValidationError, match="replay range"):
            TimeCoverage(7, 7, 24).validate()

    def test_start_at_stop_rejected(self):
        with pytest.raises(RequestValidationError, match="replay range"):
            TimeCoverage(24, 24, 24).validate()

    def test_negative_start_rejected(self):
        with pytest.raises(RequestValidationError, match="replay range"):
            TimeCoverage(-1, 4, 24).validate()

    def test_nonpositive_series_rejected(self):
        with pytest.raises(RequestValidationError, match="total_steps"):
            TimeCoverage(0, 0, 0).validate()


# ---------------------------------------------------------------------------
# SolveRequestView
# ---------------------------------------------------------------------------
class TestSolveRequestView:
    def test_valid_request_passes(self):
        make_request().validate()

    def test_read_window_must_contain_write_window(self):
        with pytest.raises(RequestValidationError, match="not contained in the read"):
            make_request(
                read_window=PhysicalWindow(0, 15, 0, 15),
                write_window=PhysicalWindow(10, 20, 10, 20),
            ).validate()

    def test_row_only_escape_rejected(self):
        with pytest.raises(RequestValidationError, match="not contained"):
            make_request(
                read_window=PhysicalWindow(0, 19, 0, 80),
            ).validate()

    def test_empty_domain_rejected(self):
        with pytest.raises(RequestValidationError, match="empty"):
            make_request(rows=0).validate()

    def test_rows_cols_transposition_caught_by_window_checks(self):
        # rows/cols swapped: the read window's column bound now exceeds the
        # (transposed) grid — a silent swap would shift every global index.
        with pytest.raises(RequestValidationError, match="exceeds"):
            make_request(rows=80, cols=100).validate()

    def test_nonpositive_pixel_size_rejected(self):
        with pytest.raises(RequestValidationError, match="pixel_size_m"):
            make_request(pixel_size_m=0.0).validate()

    def test_bad_dtype_plan_rejected(self):
        with pytest.raises(RequestValidationError, match="numpy dtype"):
            make_request(dtype_plan={"building_dsm": "float33"}).validate()

    def test_empty_profile_id_rejected(self):
        with pytest.raises(RequestValidationError, match="profile_id"):
            make_request(profile_id="").validate()

    def test_unknown_device_rejected(self):
        with pytest.raises(RequestValidationError, match="vocabulary"):
            make_request(device="tpu").validate()

    def test_empty_requested_variables_rejected(self):
        with pytest.raises(RequestValidationError, match="requested_variables"):
            make_request(requested_variables=()).validate()

    def test_default_dtype_plans_are_the_declared_contracts(self):
        assert DEFAULT_DTYPE_PLAN["landcover"] == "|u1"
        assert DEFAULT_DTYPE_PLAN["building_dsm"] == "<f4"
        assert set(DEFAULT_OUTPUT_DTYPE_PLAN) == {"utci", "tmrt", "shadow"}

    def test_describe_carries_harness_metadata(self):
        described = make_request(
            time=TimeCoverage(5, 9, 24)
        ).describe()
        # The exact request-side fields the T01 bitwise harness compares.
        assert described["write_window"] == [10, 20, 10, 20]
        assert described["read_window"] == [0, 100, 0, 80]
        assert described["time_start"] == 5
        assert described["time_stop"] == 9
        assert described["device"] == "cpu"
        assert described["profile_id"] == "canonical_cpu_v1"
