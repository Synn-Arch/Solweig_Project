# SPDX-License-Identifier: GPL-3.0-only
"""Spatial index over editable trees for deterministic window queries.

The index answers two questions for any world rectangle (or the world
rectangle covered by a raster window):

* which trees' crown discs intersect the rectangle (rasterization queries);
* which trees' conservative influence footprints intersect the rectangle
  (invalidation queries), where footprints come from
  :func:`solweig_gpu.incremental.geometry.tree_influence_bounds_m` with the
  supplied :class:`InfluenceConfig` and sun positions.

A uniform grid of world-space buckets keeps queries sublinear without
approximation: every tree is inserted into each bucket overlapped by its
influence footprint, and bucket candidates are filtered with exact
closed-rectangle and disc tests. Results are always sorted by ``tree_id``
so queries are reproducible regardless of iteration order.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor, isfinite
from typing import Iterable, Sequence

from .edits import TreeEdit
from .geometry import (
    InfluenceConfig,
    RasterGrid,
    RasterWindow,
    SunPosition,
    TreeSpec,
    tree_influence_bounds_m,
)

DEFAULT_BUCKET_SIZE_M = 64.0


@dataclass(frozen=True, slots=True)
class WorldRect:
    """Closed axis-aligned world rectangle in metres (UTM-like, y grows north)."""

    min_x_m: float
    min_y_m: float
    max_x_m: float
    max_y_m: float

    def __post_init__(self) -> None:
        bounds = (self.min_x_m, self.min_y_m, self.max_x_m, self.max_y_m)
        if not all(isfinite(float(value)) for value in bounds):
            raise ValueError("world bounds must be finite")
        if self.max_x_m < self.min_x_m or self.max_y_m < self.min_y_m:
            raise ValueError("maximum bounds must not be smaller than minimum bounds")

    @classmethod
    def from_window(cls, window: RasterWindow, grid: RasterGrid) -> "WorldRect":
        """Outer (pixel-edge) world bounds covered by a raster window."""
        pixel = grid.pixel_size_m
        return cls(
            min_x_m=grid.origin_x_m + window.col_start * pixel,
            min_y_m=grid.origin_y_m - window.row_stop * pixel,
            max_x_m=grid.origin_x_m + window.col_stop * pixel,
            max_y_m=grid.origin_y_m - window.row_start * pixel,
        )

    def intersects(self, other: "WorldRect") -> bool:
        return not (
            self.max_x_m < other.min_x_m
            or other.max_x_m < self.min_x_m
            or self.max_y_m < other.min_y_m
            or other.max_y_m < self.min_y_m
        )

    def union(self, other: "WorldRect") -> "WorldRect":
        return WorldRect(
            min_x_m=min(self.min_x_m, other.min_x_m),
            min_y_m=min(self.min_y_m, other.min_y_m),
            max_x_m=max(self.max_x_m, other.max_x_m),
            max_y_m=max(self.max_y_m, other.max_y_m),
        )


class TreeSpatialIndex:
    """Uniform-grid spatial index over the current editable trees.

    Parameters
    ----------
    trees:
        Current live trees. Duplicate ``tree_id`` values are rejected.
    grid:
        Optional grid enabling the ``*_window`` query helpers.
    config, sun_positions:
        Influence policy used for footprints (canopy plus sky-view radius,
        shadow corridors, and safety margin). Sun positions default to an
        empty sequence, which yields the canopy/sky-view footprint only.
    bucket_size_m:
        World-space edge length of the uniform buckets.
    """

    def __init__(
        self,
        trees: Iterable[TreeSpec],
        *,
        grid: RasterGrid | None = None,
        config: InfluenceConfig = InfluenceConfig(),
        sun_positions: Sequence[SunPosition] = (),
        bucket_size_m: float = DEFAULT_BUCKET_SIZE_M,
    ) -> None:
        if not isfinite(float(bucket_size_m)) or bucket_size_m <= 0:
            raise ValueError("bucket_size_m must be finite and positive")
        self._grid = grid
        self._config = config
        self._sun_positions = tuple(sun_positions)
        self._bucket_size_m = float(bucket_size_m)

        ordered = tuple(sorted(trees, key=lambda tree: tree.tree_id))
        tree_ids = [tree.tree_id for tree in ordered]
        if len(tree_ids) != len(set(tree_ids)):
            raise ValueError("duplicate tree_id values are not allowed")

        self._trees = ordered
        self._footprints: dict[str, WorldRect] = {}
        self._buckets: dict[tuple[int, int], tuple[str, ...]] = {}
        bucket_members: dict[tuple[int, int], set[str]] = {}
        for tree in ordered:
            bounds = tree_influence_bounds_m(tree, self._sun_positions, self._config)
            rect = WorldRect(*bounds)
            self._footprints[tree.tree_id] = rect
            for ix in range(
                floor(rect.min_x_m / bucket_size_m),
                floor(rect.max_x_m / bucket_size_m) + 1,
            ):
                for iy in range(
                    floor(rect.min_y_m / bucket_size_m),
                    floor(rect.max_y_m / bucket_size_m) + 1,
                ):
                    bucket_members.setdefault((ix, iy), set()).add(tree.tree_id)
        # Freeze bucket membership as sorted tuples so queries never depend on
        # set iteration order.
        self._buckets = {
            key: tuple(sorted(members)) for key, members in bucket_members.items()
        }

    @property
    def trees(self) -> tuple[TreeSpec, ...]:
        return self._trees

    @property
    def tree_ids(self) -> tuple[str, ...]:
        return tuple(tree.tree_id for tree in self._trees)

    def query_crown(self, rect: WorldRect) -> tuple[TreeSpec, ...]:
        """Trees whose crown disc intersects the closed rectangle."""
        return self._query(rect, influence=False)

    def query_influence(self, rect: WorldRect) -> tuple[TreeSpec, ...]:
        """Trees whose influence footprint intersects the closed rectangle."""
        return self._query(rect, influence=True)

    def query_crown_window(self, window: RasterWindow) -> tuple[TreeSpec, ...]:
        return self.query_crown(self._window_rect(window))

    def query_influence_window(self, window: RasterWindow) -> tuple[TreeSpec, ...]:
        return self.query_influence(self._window_rect(window))

    def affected_trees(self, edit: TreeEdit) -> tuple[TreeSpec, ...]:
        """Trees whose influence intersects the region dirtied by one edit.

        The dirty region is the union of the old and new influence footprints
        (add uses only the new tree, delete only the old tree), matching
        :func:`solweig_gpu.incremental.geometry.dirty_window_for_edit`. The
        result may include the edited tree itself when it is indexed.
        """
        rects: list[WorldRect] = []
        if edit.old_tree is not None:
            rects.append(
                WorldRect(
                    *tree_influence_bounds_m(
                        edit.old_tree, self._sun_positions, self._config
                    )
                )
            )
        if edit.new_tree is not None:
            rects.append(
                WorldRect(
                    *tree_influence_bounds_m(
                        edit.new_tree, self._sun_positions, self._config
                    )
                )
            )
        if not rects:
            raise ValueError("edit must contain an old or new tree")
        dirty = rects[0]
        for rect in rects[1:]:
            dirty = dirty.union(rect)
        return self.query_influence(dirty)

    def _window_rect(self, window: RasterWindow) -> WorldRect:
        if self._grid is None:
            raise ValueError("window queries require a grid; construct with grid=...")
        return WorldRect.from_window(window, self._grid)

    def _query(self, rect: WorldRect, *, influence: bool) -> tuple[TreeSpec, ...]:
        by_id = {tree.tree_id: tree for tree in self._trees}
        candidates: set[str] = set()
        for ix in range(
            floor(rect.min_x_m / self._bucket_size_m),
            floor(rect.max_x_m / self._bucket_size_m) + 1,
        ):
            for iy in range(
                floor(rect.min_y_m / self._bucket_size_m),
                floor(rect.max_y_m / self._bucket_size_m) + 1,
            ):
                candidates.update(self._buckets.get((ix, iy), ()))

        matches: list[TreeSpec] = []
        for tree_id in sorted(candidates):
            tree = by_id[tree_id]
            if influence:
                if self._footprints[tree_id].intersects(rect):
                    matches.append(tree)
            elif _disc_intersects_rect(tree, rect):
                matches.append(tree)
        return tuple(matches)


def _disc_intersects_rect(tree: TreeSpec, rect: WorldRect) -> bool:
    """Exact closed disc-versus-rectangle intersection test."""
    closest_x = min(max(tree.x_m, rect.min_x_m), rect.max_x_m)
    closest_y = min(max(tree.y_m, rect.min_y_m), rect.max_y_m)
    dx = closest_x - tree.x_m
    dy = closest_y - tree.y_m
    return dx * dx + dy * dy <= tree.canopy_radius_m**2
