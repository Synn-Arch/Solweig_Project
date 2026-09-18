# SPDX-License-Identifier: GPL-3.0-only
"""Geometry primitives for incremental, CPU-oriented SOLWEIG updates.

The functions in this module are intentionally independent of GDAL and PyTorch.
They define the coordinate conventions and conservative invalidation rules that
an incremental solver can share across the API, worker, and validation code.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import ceil, cos, floor, isfinite, radians, sin, tan
from numbers import Integral
from typing import Callable, Iterable, Sequence


@dataclass(frozen=True, slots=True)
class SunPosition:
    """Solar position using the SOLWEIG/GIS azimuth convention.

    Parameters
    ----------
    altitude_deg:
        Solar altitude above the horizon in degrees.
    azimuth_deg:
        Direction *toward the sun*, clockwise from north. Therefore, the cast
        shadow extends in the opposite horizontal direction.
    """

    altitude_deg: float
    azimuth_deg: float

    def __post_init__(self) -> None:
        if not isfinite(float(self.altitude_deg)):
            raise ValueError("altitude_deg must be finite")
        if not isfinite(float(self.azimuth_deg)):
            raise ValueError("azimuth_deg must be finite")


@dataclass(frozen=True, slots=True)
class TreeSpec:
    """Minimal tree parameters required by the incremental invalidation layer."""

    tree_id: str
    x_m: float
    y_m: float
    height_m: float
    canopy_radius_m: float
    trunk_ratio: float = 0.25
    transmissivity: float = 0.03

    def __post_init__(self) -> None:
        if not self.tree_id:
            raise ValueError("tree_id must be non-empty")
        for field_name, value in (
            ("x_m", self.x_m),
            ("y_m", self.y_m),
            ("height_m", self.height_m),
            ("canopy_radius_m", self.canopy_radius_m),
            ("trunk_ratio", self.trunk_ratio),
            ("transmissivity", self.transmissivity),
        ):
            if not isfinite(float(value)):
                raise ValueError(f"{field_name} must be finite")
        if self.height_m <= 0:
            raise ValueError("height_m must be positive")
        if self.canopy_radius_m <= 0:
            raise ValueError("canopy_radius_m must be positive")
        if not 0.0 <= self.trunk_ratio <= 1.0:
            raise ValueError("trunk_ratio must be in [0, 1]")
        if not 0.0 <= self.transmissivity <= 1.0:
            raise ValueError("transmissivity must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class RasterWindow:
    """Half-open raster window: rows ``[row_start, row_stop)`` and columns alike."""

    row_start: int
    row_stop: int
    col_start: int
    col_stop: int

    def __post_init__(self) -> None:
        for field_name, value in (
            ("row_start", self.row_start),
            ("row_stop", self.row_stop),
            ("col_start", self.col_start),
            ("col_stop", self.col_stop),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{field_name} must be an integer")
        if self.row_stop < self.row_start or self.col_stop < self.col_start:
            raise ValueError("window stops must be greater than or equal to starts")

    @property
    def width(self) -> int:
        return self.col_stop - self.col_start

    @property
    def height(self) -> int:
        return self.row_stop - self.row_start

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def is_empty(self) -> bool:
        return self.width == 0 or self.height == 0

    def intersects(self, other: "RasterWindow", *, gap_pixels: int = 0) -> bool:
        if gap_pixels < 0:
            raise ValueError("gap_pixels must be non-negative")
        return not (
            self.row_stop + gap_pixels < other.row_start
            or other.row_stop + gap_pixels < self.row_start
            or self.col_stop + gap_pixels < other.col_start
            or other.col_stop + gap_pixels < self.col_start
        )

    def union(self, other: "RasterWindow") -> "RasterWindow":
        if self.is_empty:
            return other
        if other.is_empty:
            return self
        return RasterWindow(
            row_start=min(self.row_start, other.row_start),
            row_stop=max(self.row_stop, other.row_stop),
            col_start=min(self.col_start, other.col_start),
            col_stop=max(self.col_stop, other.col_stop),
        )

    def expand(self, pixels: int) -> "RasterWindow":
        if pixels < 0:
            raise ValueError("pixels must be non-negative")
        return RasterWindow(
            self.row_start - pixels,
            self.row_stop + pixels,
            self.col_start - pixels,
            self.col_stop + pixels,
        )

    def clamp(self, *, rows: int, cols: int) -> "RasterWindow":
        if rows <= 0 or cols <= 0:
            raise ValueError("rows and cols must be positive")
        return RasterWindow(
            row_start=max(0, min(rows, self.row_start)),
            row_stop=max(0, min(rows, self.row_stop)),
            col_start=max(0, min(cols, self.col_start)),
            col_stop=max(0, min(cols, self.col_stop)),
        )

    def align(self, block_size: int) -> "RasterWindow":
        """Expand the window to block boundaries without clipping it."""
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        return RasterWindow(
            row_start=floor(self.row_start / block_size) * block_size,
            row_stop=ceil(self.row_stop / block_size) * block_size,
            col_start=floor(self.col_start / block_size) * block_size,
            col_stop=ceil(self.col_stop / block_size) * block_size,
        )


@dataclass(frozen=True, slots=True)
class RasterGrid:
    """North-up square-pixel raster grid.

    ``origin_x_m`` and ``origin_y_m`` identify the outer upper-left corner.
    Columns increase eastward. Rows increase southward.
    """

    rows: int
    cols: int
    pixel_size_m: float
    origin_x_m: float = 0.0
    origin_y_m: float = 0.0

    def __post_init__(self) -> None:
        for field_name, value in (("rows", self.rows), ("cols", self.cols)):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{field_name} must be an integer")
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError("rows and cols must be positive")
        for field_name, value in (
            ("pixel_size_m", self.pixel_size_m),
            ("origin_x_m", self.origin_x_m),
            ("origin_y_m", self.origin_y_m),
        ):
            if not isfinite(float(value)):
                raise ValueError(f"{field_name} must be finite")
        if self.pixel_size_m <= 0:
            raise ValueError("pixel_size_m must be positive")

    @property
    def full_window(self) -> RasterWindow:
        return RasterWindow(0, self.rows, 0, self.cols)

    @property
    def area_pixels(self) -> int:
        return self.rows * self.cols

    def world_bounds_to_window(
        self,
        *,
        min_x_m: float,
        min_y_m: float,
        max_x_m: float,
        max_y_m: float,
    ) -> RasterWindow:
        """Convert inclusive world bounds to a conservative half-open window."""
        bounds = (min_x_m, min_y_m, max_x_m, max_y_m)
        if not all(isfinite(float(value)) for value in bounds):
            raise ValueError("world bounds must be finite")
        if max_x_m < min_x_m or max_y_m < min_y_m:
            raise ValueError("maximum bounds must not be smaller than minimum bounds")

        col_start = floor((min_x_m - self.origin_x_m) / self.pixel_size_m)
        col_stop = ceil((max_x_m - self.origin_x_m) / self.pixel_size_m)
        row_start = floor((self.origin_y_m - max_y_m) / self.pixel_size_m)
        row_stop = ceil((self.origin_y_m - min_y_m) / self.pixel_size_m)

        return RasterWindow(row_start, row_stop, col_start, col_stop).clamp(
            rows=self.rows,
            cols=self.cols,
        )


@dataclass(frozen=True, slots=True)
class InfluenceConfig:
    """Conservative invalidation policy shared by local and full recomputation."""

    lowest_sky_patch_altitude_deg: float = 6.0
    minimum_direct_sun_altitude_deg: float = 5.0
    maximum_shadow_length_m: float = 300.0
    safety_margin_m: float = 6.0
    block_size_pixels: int = 16
    full_recompute_fraction: float = 0.30
    #: Extra vertical amplitude (m) added to every tree's height when
    #: deriving influence bounds. On sloped terrain a tree's canopy stands
    #: (ground_elevation + height) above datum while the lowest target
    #: surface sits near the site minimum, so the true shadow/SVF reach is
    #: governed by (elevation difference + height), not by height alone.
    #: The worker sets this to the site's DEM relief; 0.0 reproduces the
    #: flat-site behaviour.
    elevation_offset_m: float = 0.0

    def __post_init__(self) -> None:
        for field_name, value in (
            ("lowest_sky_patch_altitude_deg", self.lowest_sky_patch_altitude_deg),
            ("minimum_direct_sun_altitude_deg", self.minimum_direct_sun_altitude_deg),
            ("maximum_shadow_length_m", self.maximum_shadow_length_m),
            ("safety_margin_m", self.safety_margin_m),
            ("full_recompute_fraction", self.full_recompute_fraction),
            ("elevation_offset_m", self.elevation_offset_m),
        ):
            if not isfinite(float(value)):
                raise ValueError(f"{field_name} must be finite")
        if isinstance(self.block_size_pixels, bool) or not isinstance(
            self.block_size_pixels, Integral
        ):
            raise TypeError("block_size_pixels must be an integer")
        if self.elevation_offset_m < 0:
            raise ValueError("elevation_offset_m must be non-negative")
        if not 0.0 < self.lowest_sky_patch_altitude_deg < 90.0:
            raise ValueError("lowest_sky_patch_altitude_deg must be in (0, 90)")
        if not 0.0 < self.minimum_direct_sun_altitude_deg < 90.0:
            raise ValueError("minimum_direct_sun_altitude_deg must be in (0, 90)")
        if self.maximum_shadow_length_m <= 0:
            raise ValueError("maximum_shadow_length_m must be positive")
        if self.safety_margin_m < 0:
            raise ValueError("safety_margin_m must be non-negative")
        if self.block_size_pixels <= 0:
            raise ValueError("block_size_pixels must be positive")
        if not 0.0 < self.full_recompute_fraction <= 1.0:
            raise ValueError("full_recompute_fraction must be in (0, 1]")


class RecomputeMode(str, Enum):
    LOCAL = "local"
    FULL = "full"


#: Sentinel returned by :func:`local_elevation_offset_m` when the iterative
#: local-relief search cannot stabilise inside the configured iteration cap
#: (e.g. a monotone slope running to the site edge). Callers must then fall
#: back to a conservative global bound for that tree.
LOCAL_OFFSET_UNSTABLE = None


def local_elevation_offset_m(
    *,
    base_elevation_m: float,
    bounds_for_offset: Callable[[float], RasterWindow],
    region_minimum_m: Callable[[RasterWindow], float],
    max_iterations: int = 3,
) -> float | None:
    """Per-tree local relief for influence bounds (deterministic iteration).

    The oracle's shadow/SVF march compares the tree's canopy top
    (``base_elevation_m + height``) against the *target* surface elevation of
    every cell it can reach, so the tree's effective vertical amplitude is its
    height plus the elevation drop to the lowest reachable target. Using the
    global site relief for every tree (the flat default's conservative
    alternative) makes single small trees invalidate whole hillsides; this
    function derives the drop *around this tree* instead.

    Scheme (bounded, deterministic, from-below):

    1. start from the flat-site candidate region (offset 0);
    2. take the minimum target-surface elevation inside that region;
    3. recompute the candidate region with the implied offset; if the region
       grew and the minimum dropped, repeat;
    4. stop when the minimum stabilises (the region already contains its own
       low basin) or after ``max_iterations`` expansions.

    The iteration starts at zero offset and both step inputs are monotone in
    the offset, so every iterate stays at or below the true local relief of
    the tree's influence region. Returns ``None`` when the minimum is still
    dropping at the iteration cap: the honest bound is then unknown-locally
    and callers must use a conservative global offset.

    Parameters
    ----------
    base_elevation_m:
        Target-surface (building-DSM datum) elevation at the tree's cell.
    bounds_for_offset:
        Maps an elevation offset (m) to the candidate influence window for
        the tree. Called with 0.0 first; must be monotone non-shrinking in
        the offset.
    region_minimum_m:
        Minimum target-surface elevation (m) inside a candidate window.
        Geometry stays raster-free: the caller supplies the array lookup.
    max_iterations:
        Fixed expansion cap; must be >= 1.
    """
    if not isfinite(float(base_elevation_m)):
        raise ValueError("base_elevation_m must be finite")
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")

    window = bounds_for_offset(0.0)
    offset = 0.0
    for _ in range(max_iterations):
        minimum = float(region_minimum_m(window))
        candidate = max(0.0, float(base_elevation_m) - minimum)
        if candidate <= offset:
            # The region with this offset already contains its own minimum:
            # a stable interior basin. (A far-away deep cell can still lie
            # outside — callers must verify the final window against the
            # exact per-cell influence condition; the worker does.)
            return offset
        offset = candidate
        window = bounds_for_offset(offset)
    return LOCAL_OFFSET_UNSTABLE


def estimate_svf_radius_m(
    tree_height_m: float,
    lowest_patch_altitude_deg: float,
    *,
    maximum_radius_m: float | None = None,
) -> float:
    """Return a conservative radius within which a tree can alter sky visibility."""
    if not isfinite(float(tree_height_m)) or tree_height_m <= 0:
        raise ValueError("tree_height_m must be finite and positive")
    if not isfinite(float(lowest_patch_altitude_deg)):
        raise ValueError("lowest_patch_altitude_deg must be finite")
    if not 0.0 < lowest_patch_altitude_deg < 90.0:
        raise ValueError("lowest_patch_altitude_deg must be in (0, 90)")
    if maximum_radius_m is not None and (
        not isfinite(float(maximum_radius_m)) or maximum_radius_m <= 0
    ):
        raise ValueError("maximum_radius_m must be finite and positive")
    radius = tree_height_m / tan(radians(lowest_patch_altitude_deg))
    return min(radius, maximum_radius_m) if maximum_radius_m is not None else radius


def shadow_vector_m(
    tree_height_m: float,
    sun: SunPosition,
    *,
    minimum_altitude_deg: float = 5.0,
    maximum_length_m: float = 300.0,
) -> tuple[float, float]:
    """Return east/north displacement from the tree to its shadow endpoint."""
    if not isfinite(float(tree_height_m)) or tree_height_m <= 0:
        raise ValueError("tree_height_m must be finite and positive")
    if not isfinite(float(minimum_altitude_deg)):
        raise ValueError("minimum_altitude_deg must be finite")
    if not 0.0 < minimum_altitude_deg < 90.0:
        raise ValueError("minimum_altitude_deg must be in (0, 90)")
    if not isfinite(float(maximum_length_m)) or maximum_length_m <= 0:
        raise ValueError("maximum_length_m must be finite and positive")

    altitude = max(float(sun.altitude_deg), minimum_altitude_deg)
    altitude = min(altitude, 89.999)
    length = min(tree_height_m / tan(radians(altitude)), maximum_length_m)
    azimuth = radians(float(sun.azimuth_deg) % 360.0)

    # Sun vector toward azimuth: east=sin(az), north=cos(az).
    # A cast shadow extends exactly opposite that vector.
    return -length * sin(azimuth), -length * cos(azimuth)


def tree_influence_bounds_m(
    tree: TreeSpec,
    sun_positions: Sequence[SunPosition],
    config: InfluenceConfig,
    *,
    elevation_offset_m: float | None = None,
) -> tuple[float, float, float, float]:
    """Return a conservative axis-aligned world-space invalidation bound.

    The bound includes the canopy footprint, direct-shadow corridors for the
    supplied timesteps, and a sky-view radius based on the lowest sky patch.
    The vertical amplitude is the tree height plus ``config.elevation_offset_m``:
    on sloped terrain the canopy's reach is measured from the lowest target
    surface, not from its own base (see ``InfluenceConfig.elevation_offset_m``).
    ``elevation_offset_m`` optionally overrides the config value per tree —
    the worker derives it from the terrain *around this tree*
    (:func:`local_elevation_offset_m`) instead of the global site relief.
    """
    effective_height_m = tree.height_m + (
        config.elevation_offset_m
        if elevation_offset_m is None
        else float(elevation_offset_m)
    )
    svf_radius = estimate_svf_radius_m(
        effective_height_m,
        config.lowest_sky_patch_altitude_deg,
        maximum_radius_m=config.maximum_shadow_length_m,
    )
    radius = max(tree.canopy_radius_m, svf_radius) + config.safety_margin_m

    min_x = tree.x_m - radius
    max_x = tree.x_m + radius
    min_y = tree.y_m - radius
    max_y = tree.y_m + radius

    for sun in sun_positions:
        dx, dy = shadow_vector_m(
            effective_height_m,
            sun,
            minimum_altitude_deg=config.minimum_direct_sun_altitude_deg,
            maximum_length_m=config.maximum_shadow_length_m,
        )
        corridor_radius = tree.canopy_radius_m + config.safety_margin_m
        end_x = tree.x_m + dx
        end_y = tree.y_m + dy
        min_x = min(min_x, tree.x_m - corridor_radius, end_x - corridor_radius)
        max_x = max(max_x, tree.x_m + corridor_radius, end_x + corridor_radius)
        min_y = min(min_y, tree.y_m - corridor_radius, end_y - corridor_radius)
        max_y = max(max_y, tree.y_m + corridor_radius, end_y + corridor_radius)

    return min_x, min_y, max_x, max_y


def dirty_window_for_edit(
    grid: RasterGrid,
    *,
    old_tree: TreeSpec | None,
    new_tree: TreeSpec | None,
    sun_positions: Sequence[SunPosition],
    config: InfluenceConfig = InfluenceConfig(),
    elevation_offset_for_tree: Callable[[TreeSpec], float] | None = None,
) -> RasterWindow:
    """Compute the invalidation window for add, update, move, or delete.

    Both old and new influence regions are invalidated. This is essential for
    moves and deletes, because the old tree's effects must be removed.
    ``elevation_offset_for_tree`` optionally supplies a per-tree local relief
    (see :func:`local_elevation_offset_m`); without it every tree uses
    ``config.elevation_offset_m``.
    """
    if old_tree is None and new_tree is None:
        raise ValueError("at least one of old_tree or new_tree must be provided")

    def offset_for(tree: TreeSpec) -> float | None:
        if elevation_offset_for_tree is None:
            return None
        return float(elevation_offset_for_tree(tree))

    bounds: list[tuple[float, float, float, float]] = []
    if old_tree is not None:
        bounds.append(
            tree_influence_bounds_m(
                old_tree, sun_positions, config,
                elevation_offset_m=offset_for(old_tree),
            )
        )
    if new_tree is not None:
        bounds.append(
            tree_influence_bounds_m(
                new_tree, sun_positions, config,
                elevation_offset_m=offset_for(new_tree),
            )
        )

    min_x = min(bound[0] for bound in bounds)
    min_y = min(bound[1] for bound in bounds)
    max_x = max(bound[2] for bound in bounds)
    max_y = max(bound[3] for bound in bounds)

    window = grid.world_bounds_to_window(
        min_x_m=min_x,
        min_y_m=min_y,
        max_x_m=max_x,
        max_y_m=max_y,
    )
    return window.align(config.block_size_pixels).clamp(rows=grid.rows, cols=grid.cols)


def window_fraction(window: RasterWindow, grid: RasterGrid) -> float:
    return window.area / grid.area_pixels


def choose_recompute_mode(
    window: RasterWindow,
    grid: RasterGrid,
    *,
    full_recompute_fraction: float = 0.30,
) -> RecomputeMode:
    if not isfinite(float(full_recompute_fraction)):
        raise ValueError("full_recompute_fraction must be finite")
    if not 0.0 < full_recompute_fraction <= 1.0:
        raise ValueError("full_recompute_fraction must be in (0, 1]")
    return (
        RecomputeMode.FULL
        if window_fraction(window, grid) >= full_recompute_fraction
        else RecomputeMode.LOCAL
    )


def merge_windows(
    windows: Iterable[RasterWindow],
    *,
    gap_pixels: int = 0,
) -> list[RasterWindow]:
    """Merge intersecting or nearly adjacent windows to reduce job overhead."""
    if gap_pixels < 0:
        raise ValueError("gap_pixels must be non-negative")

    merged: list[RasterWindow] = []
    for window in sorted(
        (candidate for candidate in windows if not candidate.is_empty),
        key=lambda item: (item.row_start, item.col_start, item.row_stop, item.col_stop),
    ):
        candidate = window
        changed = True
        while changed:
            changed = False
            retained: list[RasterWindow] = []
            for existing in merged:
                if candidate.intersects(existing, gap_pixels=gap_pixels):
                    candidate = candidate.union(existing)
                    changed = True
                else:
                    retained.append(existing)
            merged = retained
        merged.append(candidate)

    return sorted(merged, key=lambda item: (item.row_start, item.col_start))
