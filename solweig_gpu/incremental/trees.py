# SPDX-License-Identifier: GPL-3.0-only
"""Editable tree layer: immutable base vegetation plus ordered tree edits.

The layer reconstructs vegetation inputs for any raster window exactly as the
full pipeline constructs them globally. ``Trees.tif`` stores canopy-top height
*above ground*; crown footprints are discs evaluated at cell centres; and
overlapping canopies combine with ``maximum`` (never a sum), matching
``solweig_gpu/utci_process.py`` (``vegdem = trees + dem``,
``vegdem2 = trees * 0.25 + dem``) and
:func:`solweig_gpu.incremental.fixtures.rasterize_tree`.

The trunk-zone raster generalizes the pipeline's fixed ``0.25`` fraction to a
per-tree ``trunk_ratio``: each tree paints its trunk zone at
``trunk_ratio * height_m`` and overlapping contributions combine with
``maximum``. With the default ``trunk_ratio = 0.25`` this reduces exactly to
the oracle rule ``vegdem2 = trees * 0.25 + dem``.

Window rasterization is local *by construction*: every cell value is computed
from world coordinates alone, so rasterizing a window always equals slicing a
full-domain rasterization to the same window.
"""

from __future__ import annotations

from dataclasses import replace
from math import ceil, floor, isfinite
from typing import Iterable, Sequence

import numpy as np

from .edits import CoalescedEditBatch, TreeEdit, coalesce_tree_edits
from .geometry import (
    InfluenceConfig,
    RasterGrid,
    RasterWindow,
    SunPosition,
    TreeSpec,
    dirty_window_for_edit,
)
from .spatial_index import TreeSpatialIndex

# The full pipeline derives the trunk zone as trees * 0.25; the base raster
# has no per-tree ratios, so it always uses this fraction.
BASE_TRUNK_FRACTION = 0.25
_BASE_TRUNK_FRACTION_32 = np.float32(BASE_TRUNK_FRACTION)

# Server-side preset ranges documented in the API contract and fixtures.
TREE_HEIGHT_RANGE_M = (3.0, 40.0)
TREE_CANOPY_DIAMETER_RANGE_M = (1.0, 30.0)

#: Trunk-zone share of tree height, a HALF-OPEN ``[low, high)`` interval:
#: a ratio of ``high`` would raise the trunk zone to the canopy top and
#: is rejected (:func:`validate_tree_spec`). The capability document
#: serves this range by import — never a second hand-written copy.
TRUNK_RATIO_RANGE = (0.0, 1.0)


def validate_tree_spec(tree: TreeSpec) -> TreeSpec:
    """Validate one editable tree; return it unchanged.

    Structural checks (server-side tree validation):

    * all coordinates and dimensions must be finite;
    * ``height_m > 0`` and ``canopy_radius_m > 0``;
    * ``trunk_ratio`` in ``[0, 1)`` (a ratio of 1 would raise the trunk zone
      to the canopy top and is rejected);
    * ``transmissivity`` in ``[0, 1]``.
    """
    for field_name, value in (
        ("x_m", tree.x_m),
        ("y_m", tree.y_m),
        ("height_m", tree.height_m),
        ("canopy_radius_m", tree.canopy_radius_m),
        ("trunk_ratio", tree.trunk_ratio),
        ("transmissivity", tree.transmissivity),
    ):
        if not isfinite(float(value)):
            raise ValueError(f"{field_name} must be finite")
    if tree.height_m <= 0:
        raise ValueError("height_m must be positive")
    if tree.canopy_radius_m <= 0:
        raise ValueError("canopy_radius_m must be positive")
    ratio_low, ratio_high = TRUNK_RATIO_RANGE
    if not ratio_low <= tree.trunk_ratio < ratio_high:
        raise ValueError(f"trunk_ratio must be in [{ratio_low:g}, {ratio_high:g})")
    if not 0.0 <= tree.transmissivity <= 1.0:
        raise ValueError("transmissivity must be in [0, 1]")
    return tree


def validate_tree_preset(tree: TreeSpec) -> TreeSpec:
    """Apply the documented server preset ranges on top of the structural checks."""
    validate_tree_spec(tree)
    low, high = TREE_HEIGHT_RANGE_M
    if not low <= tree.height_m <= high:
        raise ValueError(f"height_m must be between {low:g} and {high:g} metres")
    low, high = TREE_CANOPY_DIAMETER_RANGE_M
    if not low <= 2.0 * tree.canopy_radius_m <= high:
        raise ValueError(f"canopy diameter must be between {low:g} and {high:g} metres")
    return tree


def rasterize_tree_patch(
    tree: TreeSpec,
    grid: RasterGrid,
    window: RasterWindow,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize one tree clipped to ``window``.

    Returns ``(canopy_above_ground, trunk_above_ground)`` float32 arrays of
    the window shape: canopy cells carry ``height_m`` and trunk cells carry
    ``trunk_ratio * height_m``. Cells outside the crown disc are zero. Cell
    centres are computed from world coordinates using the same expression as
    the full-domain rasterizer, so local and sliced-full results agree
    bit-for-bit.
    """
    canopy = np.zeros((window.height, window.width), dtype=np.float32)
    trunk = np.zeros((window.height, window.width), dtype=np.float32)

    pixel = grid.pixel_size_m
    radius = tree.canopy_radius_m

    col_lo = floor((tree.x_m - radius - grid.origin_x_m) / pixel - 0.5) - 1
    col_hi = ceil((tree.x_m + radius - grid.origin_x_m) / pixel - 0.5) + 1
    row_lo = floor((grid.origin_y_m - tree.y_m - radius) / pixel - 0.5) - 1
    row_hi = ceil((grid.origin_y_m - tree.y_m + radius) / pixel - 0.5) + 1

    col_lo = max(col_lo, window.col_start)
    col_hi = min(col_hi, window.col_stop)
    row_lo = max(row_lo, window.row_start)
    row_hi = min(row_hi, window.row_stop)
    if col_lo >= col_hi or row_lo >= row_hi:
        return canopy, trunk

    cols = np.arange(col_lo, col_hi, dtype=np.int64)
    rows = np.arange(row_lo, row_hi, dtype=np.int64)
    centre_x = grid.origin_x_m + (cols + 0.5) * pixel
    centre_y = grid.origin_y_m - (rows + 0.5) * pixel
    inside = (centre_x[np.newaxis, :] - tree.x_m) ** 2 + (
        centre_y[:, np.newaxis] - tree.y_m
    ) ** 2 <= radius**2

    canopy_patch = canopy[
        row_lo - window.row_start : row_hi - window.row_start,
        col_lo - window.col_start : col_hi - window.col_start,
    ]
    trunk_patch = trunk[
        row_lo - window.row_start : row_hi - window.row_start,
        col_lo - window.col_start : col_hi - window.col_start,
    ]
    canopy_patch[inside] = np.float32(tree.height_m)
    trunk_patch[inside] = np.float32(tree.trunk_ratio * tree.height_m)
    return canopy, trunk


class TreeLayer:
    """Base vegetation raster plus an ordered log of tree edits.

    The base raster is treated as immutable: window rasterization crops it
    and copies, never writes into it. Edits are recorded as
    :class:`~solweig_gpu.incremental.edits.TreeEdit` entries so they can be
    coalesced and turned into dirty windows by the incremental worker.
    """

    def __init__(
        self,
        base_vegetation: np.ndarray,
        grid: RasterGrid,
        *,
        scenario_id: str = "default",
    ) -> None:
        if not scenario_id:
            raise ValueError("scenario_id must be non-empty")
        base = np.asarray(base_vegetation)
        if base.ndim != 2 or base.shape != (grid.rows, grid.cols):
            raise ValueError(
                f"base_vegetation must have shape ({grid.rows}, {grid.cols})"
            )
        if base.dtype != np.float32:
            base = base.astype(np.float32)
        self._base = base
        self._grid = grid
        self._scenario_id = scenario_id
        self._edits: list[TreeEdit] = []
        self._live: dict[str, TreeSpec] = {}
        self._sequence = 0

    @classmethod
    def from_geotiff(cls, path, *, scenario_id: str = "default") -> "TreeLayer":
        """Build a layer from a ``Trees.tif`` raster (height above ground)."""
        import rasterio

        with rasterio.open(path) as dataset:
            base = dataset.read(1).astype("float32")
            transform = dataset.transform
            grid = RasterGrid(
                rows=dataset.height,
                cols=dataset.width,
                pixel_size_m=float(abs(transform.a)),
                origin_x_m=float(transform.c),
                origin_y_m=float(transform.f),
            )
        return cls(base, grid, scenario_id=scenario_id)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def grid(self) -> RasterGrid:
        return self._grid

    @property
    def scenario_id(self) -> str:
        return self._scenario_id

    @property
    def base_vegetation(self) -> np.ndarray:
        """Read-only view of the immutable base canopy raster (above ground)."""
        view = self._base.view()
        view.flags.writeable = False
        return view

    @property
    def edits(self) -> tuple[TreeEdit, ...]:
        return tuple(self._edits)

    def current_trees(self) -> tuple[TreeSpec, ...]:
        """Live edited trees, deterministically sorted by ``tree_id``."""
        return tuple(self._live[tree_id] for tree_id in sorted(self._live))

    def coalesced_batch(self) -> CoalescedEditBatch | None:
        """Coalesced equivalent of the edit log, or ``None`` when empty."""
        if not self._edits:
            return None
        return coalesce_tree_edits(self._edits)

    def build_index(
        self,
        *,
        config: InfluenceConfig = InfluenceConfig(),
        sun_positions: Sequence[SunPosition] = (),
    ) -> TreeSpatialIndex:
        """Spatial index over the current live trees."""
        return TreeSpatialIndex(
            self.current_trees(),
            grid=self._grid,
            config=config,
            sun_positions=sun_positions,
        )

    # ------------------------------------------------------------------
    # Edits
    # ------------------------------------------------------------------

    def add_tree(self, tree: TreeSpec) -> TreeEdit:
        validate_tree_spec(tree)
        if tree.tree_id in self._live:
            raise ValueError(f"tree {tree.tree_id!r} already exists")
        return self._record(old_tree=None, new_tree=tree)

    def move_tree(self, tree_id: str, *, x_m: float, y_m: float) -> TreeEdit:
        current = self._require_tree(tree_id)
        if not isfinite(float(x_m)) or not isfinite(float(y_m)):
            raise ValueError("x_m and y_m must be finite")
        return self._record(
            old_tree=current,
            new_tree=replace(current, x_m=float(x_m), y_m=float(y_m)),
        )

    def update_tree(self, tree_id: str, **changes: float) -> TreeEdit:
        """Update editable properties (height, radius, trunk ratio, transmissivity)."""
        current = self._require_tree(tree_id)
        allowed = {"height_m", "canopy_radius_m", "trunk_ratio", "transmissivity"}
        unknown = set(changes) - allowed
        if unknown:
            raise ValueError(f"unknown fields: {sorted(unknown)}")
        if not changes:
            raise ValueError("at least one field is required")
        return self._record(old_tree=current, new_tree=replace(current, **changes))

    def delete_tree(self, tree_id: str) -> TreeEdit:
        current = self._require_tree(tree_id)
        return self._record(old_tree=current, new_tree=None)

    def affected_window(
        self,
        edits: Iterable[TreeEdit] | None = None,
        *,
        sun_positions: Sequence[SunPosition] = (),
        config: InfluenceConfig = InfluenceConfig(),
    ) -> RasterWindow:
        """Union dirty window (old union new influence) for the given edits.

        Defaults to every edit recorded so far. The result is block-aligned
        and clamped to the grid by :func:`dirty_window_for_edit`. An empty
        edit list yields an empty window.
        """
        edit_list = list(self._edits if edits is None else edits)
        if not edit_list:
            return RasterWindow(0, 0, 0, 0)
        union = RasterWindow(0, 0, 0, 0)
        for edit in edit_list:
            union = union.union(
                dirty_window_for_edit(
                    self._grid,
                    old_tree=edit.old_tree,
                    new_tree=edit.new_tree,
                    sun_positions=sun_positions,
                    config=config,
                )
            )
        return union

    # ------------------------------------------------------------------
    # Rasterization
    # ------------------------------------------------------------------

    def rasterize_window(self, window: RasterWindow) -> np.ndarray:
        """Window-local canopy-top height above ground (``Trees.tif`` layout)."""
        return self.vegetation_rasters_window(window)[0]

    def vegetation_rasters_window(
        self,
        window: RasterWindow,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(canopy_above_ground, trunk_above_ground)`` for a window.

        Both arrays are float32 with the window shape. The base raster is
        cropped (copied, never written) and every live edited tree is
        rasterized into the window in world coordinates, combining with
        ``maximum`` exactly like the full input construction.
        """
        self._check_window(window)
        base_crop = self._base[
            window.row_start : window.row_stop, window.col_start : window.col_stop
        ]
        canopy = base_crop.copy()
        trunk = base_crop * _BASE_TRUNK_FRACTION_32
        for tree in self.current_trees():
            tree_canopy, tree_trunk = rasterize_tree_patch(tree, self._grid, window)
            np.maximum(canopy, tree_canopy, out=canopy)
            np.maximum(trunk, tree_trunk, out=trunk)
        return canopy, trunk

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _require_tree(self, tree_id: str) -> TreeSpec:
        if tree_id not in self._live:
            raise ValueError(f"unknown tree {tree_id!r}")
        return self._live[tree_id]

    def _record(self, *, old_tree: TreeSpec | None, new_tree: TreeSpec | None) -> TreeEdit:
        tree_id = (new_tree or old_tree).tree_id  # type: ignore[union-attr]
        self._sequence += 1
        edit = TreeEdit(
            scenario_id=self._scenario_id,
            sequence=self._sequence,
            tree_id=tree_id,
            old_tree=old_tree,
            new_tree=new_tree,
        )
        self._edits.append(edit)
        if new_tree is None:
            self._live.pop(tree_id, None)
        else:
            self._live[tree_id] = new_tree
        return edit

    def _check_window(self, window: RasterWindow) -> None:
        rows, cols = self._grid.rows, self._grid.cols
        if not (
            0 <= window.row_start <= window.row_stop <= rows
            and 0 <= window.col_start <= window.col_stop <= cols
        ):
            raise ValueError(
                f"window {window} must lie inside the {rows}x{cols} grid"
            )
