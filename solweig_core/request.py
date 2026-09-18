"""Typed solve requests: logical domain, physical windows, time coverage.

DESIGN.ko.md 5.3: ``logical_domain`` defines reference arithmetic and
boundary rules; ``physical_work_window`` defines which cells are actually
computed. They are separate fields here and must stay separate — a window
shrink that silently redefines the domain is the class of bug that changes
ray termination and reductions.

Time coverage is expressed in GLOBAL time indices (timestep numbers within
the forcing series), NEVER as array offsets: ``time_start=0, time_stop=24``
means global timesteps 0..23, whatever buffer the consumer slices. The T01
harness compares exactly this metadata (global time index, window origin),
so the request carries it first-class (see
``tests/ultrafast/bitwise_harness.PlaneMetadata``).

The dtype plan is a *declared contract* per plane (categorical uint8 /
coordinate int32 / scientific float32 per DESIGN 5.3's recommended buffer
types), validated as parseable numpy dtypes; it is not a performance hint.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from solweig_core.status import (
    RefusalReason,
    RequestValidationError,
    TypedRefusal,
)

__all__ = [
    "PhysicalWindow",
    "TimeCoverage",
    "SolveRequestView",
    "DEFAULT_DTYPE_PLAN",
    "DEFAULT_OUTPUT_DTYPE_PLAN",
]

#: Input planes the T02 adapter materializes at the ABI boundary, with the
#: dtypes the legacy path actually consumes (float32 scientific planes; the
#: land-cover class grid is uint8 at the boundary and cast to float32 at
#: the legacy seam exactly like ``solve_window`` does).
DEFAULT_DTYPE_PLAN: dict[str, str] = {
    "building_dsm": "<f4",
    "dem": "<f4",
    "veg_canopy": "<f4",
    "walls": "<f4",
    "wall_aspect": "<f4",
    "landcover": "|u1",
}

#: Output planes the solver produces (T00 harness vocabulary), float32.
DEFAULT_OUTPUT_DTYPE_PLAN: dict[str, str] = {
    "utci": "<f4",
    "tmrt": "<f4",
    "shadow": "<f4",
}


def _refusal(detail: str, context: dict[str, Any] | None = None):
    return RequestValidationError(TypedRefusal(RefusalReason.REQUEST_INVALID, detail, context))


@dataclass(frozen=True)
class PhysicalWindow:
    """Half-open cell window [row_start, row_stop) x [col_start, col_stop).

    This is the PHYSICAL work window — which cells get computed. The
    LOGICAL domain (rows/cols/origin of the whole grid) lives on
    :class:`SolveRequestView`; the two are deliberately different objects.
    """

    row_start: int
    row_stop: int
    col_start: int
    col_stop: int

    def validate(self, *, rows: int, cols: int, label: str = "window") -> None:
        if self.row_start < 0 or self.col_start < 0:
            raise _refusal(
                f"{label} origin ({self.row_start}, {self.col_start}) is negative",
                {"window": [self.row_start, self.row_stop, self.col_start, self.col_stop]},
            )
        if self.row_stop <= self.row_start or self.col_stop <= self.col_start:
            raise _refusal(
                f"{label} is empty or inverted: {self}", {"window": list(self)}
            )
        if self.row_stop > rows or self.col_stop > cols:
            raise _refusal(
                f"{label} {self} exceeds the logical domain grid "
                f"({rows} rows x {cols} cols)",
                {"window": list(self), "grid": [rows, cols]},
            )

    @property
    def height(self) -> int:
        return self.row_stop - self.row_start

    @property
    def width(self) -> int:
        return self.col_stop - self.col_start

    @property
    def origin(self) -> tuple[int, int]:
        return (self.row_start, self.col_start)

    def list(self) -> list[int]:
        return [self.row_start, self.row_stop, self.col_start, self.col_stop]

    def __iter__(self):
        return iter((self.row_start, self.row_stop, self.col_start, self.col_stop))

    def contains(self, other: "PhysicalWindow") -> bool:
        return (
            self.row_start <= other.row_start
            and self.row_stop >= other.row_stop
            and self.col_start <= other.col_start
            and self.col_stop >= other.col_stop
        )


@dataclass(frozen=True)
class TimeCoverage:
    """GLOBAL time indices covered by a request (never array offsets).

    ``time_start``/``time_stop`` are half-open global timestep indices into
    the site's forcing series; ``time_stop=None`` means "to the end of the
    series" and normalizes to ``total_steps`` on construction.
    """

    time_start: int
    time_stop: int | None
    total_steps: int

    def __post_init__(self) -> None:
        if self.time_stop is None:
            object.__setattr__(self, "time_stop", int(self.total_steps))

    def validate(self) -> None:
        if int(self.total_steps) <= 0:
            raise _refusal(f"total_steps must be positive, got {self.total_steps}")
        if int(self.time_start) < 0 or int(self.time_start) >= int(self.time_stop):
            raise _refusal(
                f"time_start {self.time_start} outside the replay range "
                f"[0, {self.time_stop})",
                {"time": [self.time_start, self.time_stop]},
            )
        if int(self.time_stop) > int(self.total_steps):
            raise _refusal(
                f"time_stop {self.time_stop} exceeds the forcing series "
                f"({self.total_steps} global timesteps)",
                {"time": [self.time_start, self.time_stop]},
            )

    @property
    def global_indices(self) -> tuple[int, ...]:
        """The request-side metadata the T01 harness compares."""
        return tuple(range(int(self.time_start), int(self.time_stop)))


@dataclass(frozen=True)
class SolveRequestView:
    """Everything the core needs to route one window solve.

    Built by the adapter seam from the site cache/worker inputs; validated
    before any numerical work. The device is DECLARED here (default
    ``'cpu'``) — :mod:`solweig_core.dispatch` resolves it explicitly and
    never auto-selects by availability.
    """

    logical_domain_id: str
    rows: int
    cols: int
    origin_x_m: float
    origin_y_m: float
    pixel_size_m: float
    read_window: PhysicalWindow
    write_window: PhysicalWindow
    time: TimeCoverage
    requested_variables: tuple[str, ...] = ("utci", "tmrt", "shadow")
    dtype_plan: Mapping[str, str] = field(
        default_factory=lambda: dict(DEFAULT_DTYPE_PLAN)
    )
    output_dtype_plan: Mapping[str, str] = field(
        default_factory=lambda: dict(DEFAULT_OUTPUT_DTYPE_PLAN)
    )
    profile_id: str = ""
    device: str = "cpu"

    def validate(self) -> None:
        """Structural validation; raises :class:`RequestValidationError`.

        RED witnesses pinned by tests/ultrafast/test_request.py: windows
        escaping the grid, a write window not contained in the read window,
        inverted rows/cols, time coverage beyond the series, unparseable
        dtype plans, and an unknown device vocabulary entry.
        """
        if not isinstance(self.logical_domain_id, str) or not self.logical_domain_id:
            raise _refusal("logical_domain_id must be a non-empty string")
        if self.rows <= 0 or self.cols <= 0:
            raise _refusal(f"logical domain shape ({self.rows}, {self.cols}) is empty")
        if not (self.pixel_size_m > 0.0):
            raise _refusal(f"pixel_size_m must be positive, got {self.pixel_size_m!r}")
        self.read_window.validate(rows=self.rows, cols=self.cols, label="read window")
        self.write_window.validate(rows=self.rows, cols=self.cols, label="write window")
        if not self.read_window.contains(self.write_window):
            raise _refusal(
                f"write window {self.write_window} is not contained in the read "
                f"window {self.read_window}",
                {
                    "read": self.read_window.list(),
                    "write": self.write_window.list(),
                },
            )
        self.time.validate()
        if not self.requested_variables:
            raise _refusal("requested_variables must be non-empty")
        for plan_name, plan in (
            ("dtype_plan", self.dtype_plan),
            ("output_dtype_plan", self.output_dtype_plan),
        ):
            for plane, dtype_str in dict(plan).items():
                try:
                    np.dtype(dtype_str)
                except TypeError as error:
                    raise _refusal(
                        f"{plan_name}[{plane!r}] = {dtype_str!r} is not a numpy "
                        "dtype"
                    ) from error
        if not self.profile_id:
            raise _refusal("profile_id is required (T01 profile identity)")
        if self.device not in ("cpu", "cuda"):
            raise _refusal(
                f"device {self.device!r} is outside the request vocabulary "
                "('cpu' or 'cuda'); dispatch never guesses"
            )

    def describe(self) -> dict[str, Any]:
        """JSON-friendly summary for run records."""
        return {
            "logical_domain_id": self.logical_domain_id,
            "rows": self.rows,
            "cols": self.cols,
            "origin_x_m": self.origin_x_m,
            "origin_y_m": self.origin_y_m,
            "pixel_size_m": self.pixel_size_m,
            "read_window": self.read_window.list(),
            "write_window": self.write_window.list(),
            "time_start": self.time.time_start,
            "time_stop": self.time.time_stop,
            "total_steps": self.time.total_steps,
            "requested_variables": list(self.requested_variables),
            "dtype_plan": dict(self.dtype_plan),
            "output_dtype_plan": dict(self.output_dtype_plan),
            "profile_id": self.profile_id,
            "device": self.device,
        }
