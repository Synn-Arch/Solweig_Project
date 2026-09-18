# SPDX-License-Identifier: GPL-3.0-only
"""Editable building layer: immutable base massing plus ordered massing edits.

This module is U-C packet 5's rasterizer for committed building-massing
edits. The typed hand-off comes from the building adapter's executor seam
(:func:`solweig_gpu.incremental.adapters.building.massing_edits_from_deltas`,
which re-validates every ``(before, after)`` spec pair), and the output is
a scenario ``Building_DSM`` raster in the pipeline's absolute-elevation
layout — the layer :mod:`solweig_gpu.incremental.regenerate` feeds into the
full preprocessing chain.

Rasterization rules (fixed here, cited to their precedents)
-----------------------------------------------------------

* **Absolute elevation, grounded per cell.** ``Building_DSM.tif`` stores
  surface elevation above datum, so a block paints
  ``dem[row, col] + height_m`` in every footprint cell. The ground is
  sampled *per cell* (nearest-cell, no interpolation): that is exactly the
  convention the vegetation path uses — ``Trees.tif`` stores height above
  ground and the pipeline combines it with the DEM cell-by-cell
  (``vegdem = temp1 + temp2``, ``solweig_gpu/utci_process.py``), and
  :func:`solweig_gpu.incremental.trees.rasterize_tree_patch` never
  interpolates the DEM either. A block on sloped ground therefore follows
  the slope, exactly like an equivalently-placed canopy would.
* **Cell-centre fill, boundary-inclusive, trees orientation.** A cell is
  inside the footprint iff its centre — computed with the tree layer's
  expression ``centre_x = origin_x + (col + 0.5) * pixel``,
  ``centre_y = origin_y - (row + 0.5) * pixel``
  (:mod:`solweig_gpu.incremental.trees`, ``rasterize_tree_patch``) — lies
  inside the polygon, boundaries included. The disc test's ``<= radius**2``
  boundary inclusion (trees.py) becomes an on-edge test with a tolerance
  scaled to the edge length. The even-odd crossing rule makes the result
  independent of ring winding (CW and CCW paint identically) and
  deterministic for self-intersecting rings, which the adapter explicitly
  leaves to this contract (no geometry library in the incremental layer).
* **Combination: ``maximum``, resets excepted.** Painted blocks combine
  with the base raster and each other through ``np.fmax`` (the trees layer
  uses ``np.maximum``; ``fmax`` additionally lets a block legitimately
  cover NaN nodata cells in the base DSM without erasing itself). Only a
  removed block's ``before`` footprint lowers the surface, back to DEM
  ground.
* **Copy-on-write.** The base Building_DSM and DEM are copied at
  construction and exposed read-only; rasterization always builds a fresh
  array. The baseline raster is never written through.
* **Non-finite ground is refused, not guessed.** A footprint cell over a
  non-finite DEM value (NaN / inf nodata) aborts rasterization with the
  block id and cell count: painting a building on undefined ground would
  fabricate data. Finite nodata sentinels (e.g. ``-9999``) are the
  caller's contract, exactly like the tree layer, which ignores nodata
  metadata entirely.
* **Before-footprint resets are unconditional.** For a ``move`` /
  ``update`` / ``delete``, every cell of the block's validated ``before``
  footprint returns to DEM ground before the ``after`` footprint (if any)
  is painted. A before footprint that overlaps *other* baseline structures
  resets those cells too — the raster-only baseline cannot distinguish
  owners; the adapter's object model guarantees the before state, and the
  per-edit ``cells_reset`` counts make the action auditable. Disclosed
  here rather than fenced (the adapter's conservative choice 5 applies).

``walllimit`` disclosure
------------------------

``solweig_gpu/walls_aspect.py`` fixes ``walllimit = 3.0`` (module line 25,
applied in ``findwalls``): a block whose height above ground stays below
3 m produces NO walls/wall_aspect entries of its own in the regenerated
outputs. Such blocks remain legitimate massing (they still shade), so this
layer rasterizes them normally and reports
``MassingRasterRecord.walls_expected = False`` — surfaced, never silently
dropped (the adapter's preview limitation, made executable).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor
from typing import Sequence

import numpy as np

from .adapters.building import BuildingSpec, MassingEdit
from .geometry import RasterGrid, RasterWindow

__all__ = [
    "WALL_LIMIT_M",
    "BuildingLayer",
    "BuildingRasterResult",
    "MassingRasterRecord",
    "footprint_window",
    "polygon_cell_mask",
]

#: Wall-height threshold from ``solweig_gpu/walls_aspect.py`` (line 25,
#: ``walllimit``, applied in ``findwalls``). Duplicated as a value here so
#: this module stays importable without GDAL (walls_aspect imports osgeo at
#: module scope); the citation is the contract, the float is the courtesy.
WALL_LIMIT_M = 3.0


# ---------------------------------------------------------------------------
# Footprint geometry (world coordinates -> cell-centre mask)
# ---------------------------------------------------------------------------


def _closed_ring(
    footprint_m: tuple[tuple[float, float], ...],
) -> tuple[tuple[float, float], ...]:
    """Return the footprint with the ring closed (first == last)."""
    if footprint_m[0] == footprint_m[-1]:
        return footprint_m
    return (*footprint_m, footprint_m[0])


def _raw_bounds(
    footprint_m: Sequence[tuple[float, float]],
    grid: RasterGrid,
) -> tuple[int, int, int, int]:
    """Conservative (row_lo, row_hi, col_lo, col_hi) cell bounds.

    Uses the tree layer's cell-centre bounding recipe
    (``rasterize_tree_patch``: ``floor/ceil(value / pixel - 0.5) ∓ 1``), so
    a cell centre is never missed by rounding. Bounds may fall outside the
    grid; callers clip.
    """
    xs = [float(vertex[0]) for vertex in footprint_m]
    ys = [float(vertex[1]) for vertex in footprint_m]
    pixel = grid.pixel_size_m
    col_lo = floor((min(xs) - grid.origin_x_m) / pixel - 0.5) - 1
    col_hi = ceil((max(xs) - grid.origin_x_m) / pixel - 0.5) + 1
    row_lo = floor((grid.origin_y_m - max(ys)) / pixel - 0.5) - 1
    row_hi = ceil((grid.origin_y_m - min(ys)) / pixel - 0.5) + 1
    return row_lo, row_hi, col_lo, col_hi


def footprint_window(
    footprint_m: Sequence[tuple[float, float]],
    grid: RasterGrid,
    window: RasterWindow,
) -> RasterWindow:
    """Conservative candidate window for a footprint, clipped to ``window``.

    The result may be empty (footprint outside the site or window);
    clamping to both the window and the grid keeps it a valid window for
    direct slicing.
    """
    row_lo, row_hi, col_lo, col_hi = _raw_bounds(footprint_m, grid)
    # A footprint outside the clipped region can leave stop < start; the
    # empty window (stop == start) is the honest result, never an error.
    row_start = max(row_lo, window.row_start, 0)
    row_stop = max(min(row_hi, window.row_stop, grid.rows), row_start)
    col_start = max(col_lo, window.col_start, 0)
    col_stop = max(min(col_hi, window.col_stop, grid.cols), col_start)
    return RasterWindow(
        row_start=row_start,
        row_stop=row_stop,
        col_start=col_start,
        col_stop=col_stop,
    )


def polygon_cell_mask(
    footprint_m: Sequence[tuple[float, float]],
    grid: RasterGrid,
    window: RasterWindow,
) -> tuple[np.ndarray, bool]:
    """Even-odd, boundary-inclusive cell-centre mask for one footprint.

    Returns ``(mask, edge_touching)`` where ``mask`` has the window shape
    and ``edge_touching`` reports whether any painted cell lies on the
    grid's border row/column — where ``findwalls`` zeroes wall cells
    (``walls[:, 0] = 0`` etc.), so edge blocks lose their border wall
    entries downstream.

    The mask is evaluated over the FULL grid and sliced to the window, so
    a window rasterization is a slice of the full-domain rasterization by
    construction, and ``edge_touching`` is exact rather than a padded
    bounding-box approximation. Cell centres use the tree layer's
    world-coordinate expression verbatim (``rasterize_tree_patch``), so
    masks agree with the disc test under the same grid.
    """
    shape = (window.height, window.width)
    candidate = footprint_window(footprint_m, grid, grid.full_window)
    if candidate.is_empty:
        return np.zeros(shape, dtype=bool), False

    rows = np.arange(candidate.row_start, candidate.row_stop, dtype=np.int64)
    cols = np.arange(candidate.col_start, candidate.col_stop, dtype=np.int64)
    pixel = grid.pixel_size_m
    centre_x = grid.origin_x_m + (cols + 0.5) * pixel  # trees.py convention
    centre_y = grid.origin_y_m - (rows + 0.5) * pixel
    # Broadcast to (rows, cols): x varies along columns, y along rows.
    cx = centre_x[np.newaxis, :]
    cy = centre_y[:, np.newaxis]

    ring = _closed_ring(tuple((float(x), float(y)) for x, y in footprint_m))
    # Broadcast target: rows along axis 0, cols along axis 1.
    inside = np.zeros((cy.shape[0], cx.shape[1]), dtype=bool)
    for index in range(len(ring) - 1):
        x1, y1 = ring[index]
        x2, y2 = ring[index + 1]
        # On-edge test (boundary-inclusive, mirroring the disc's <=):
        # tolerance scales with the edge so float rounding never eats an
        # exact hit, and a hair outside stays outside.
        edge_len = abs(x2 - x1) + abs(y2 - y1) + 1.0
        tol = 1e-9 * edge_len
        cross = (x2 - x1) * (cy - y1) - (y2 - y1) * (cx - x1)
        on_segment = (
            (np.abs(cross) <= tol)
            & (cx >= min(x1, x2) - tol)
            & (cx <= max(x1, x2) + tol)
            & (cy >= min(y1, y2) - tol)
            & (cy <= max(y1, y2) + tol)
        )
        # Even-odd crossing test, ray cast toward +x. Horizontal edges
        # never satisfy the straddle condition; their points are covered
        # by on_segment. Zero-length edges (a closed ring's seam) are
        # inert in both tests.
        dy = y2 - y1
        straddle = (y1 > cy) != (y2 > cy)
        safe_dy = np.where(dy == 0.0, 1.0, dy)
        x_cross = (x2 - x1) * (cy - y1) / safe_dy + x1
        inside ^= straddle & (cx < x_cross)
        inside |= on_segment

    full_mask = np.zeros((grid.rows, grid.cols), dtype=bool)
    full_mask[
        candidate.row_start : candidate.row_stop,
        candidate.col_start : candidate.col_stop,
    ] = inside
    border = np.zeros((grid.rows, grid.cols), dtype=bool)
    border[0, :] = border[-1, :] = True
    border[:, 0] = border[:, -1] = True
    edge_touching = bool((full_mask & border).any())
    return (
        full_mask[
            window.row_start : window.row_stop, window.col_start : window.col_stop
        ].copy(),
        edge_touching,
    )


# ---------------------------------------------------------------------------
# Rasterization records (auditable metadata per applied edit)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MassingRasterRecord:
    """What one massing edit did to the raster (and what to disclose).

    Counts are relative to the rasterized window: the full-domain
    rasterization reports site-wide counts. ``operation`` is derived from
    the spec pair (``add``: no before; ``delete``: no after; ``update``:
    both — the adapter's ``move`` is an update whose footprint moved).
    ``walls_expected`` is whether the AFTER block's height clears
    ``WALL_LIMIT_M``; it is ``False`` for a ``delete`` (no after spec, so
    nothing of this block remains to wall) — a disclosure never fires for
    deletes.
    """

    building_id: str
    operation: str
    cells_painted: int
    cells_reset: int
    walls_expected: bool
    edge_touching: bool

    @property
    def disclosures(self) -> tuple[str, ...]:
        notes: list[str] = []
        if self.operation in ("add", "update") and not self.walls_expected:
            notes.append(
                f"building {self.building_id!r}: height below the wall "
                f"threshold ({WALL_LIMIT_M:g} m, walls_aspect.py walllimit); "
                "the regenerated walls/wall_aspect rasters will carry no "
                "entries for this block"
            )
        if self.operation in ("add", "update") and self.cells_painted == 0:
            notes.append(
                f"building {self.building_id!r}: footprint covers no cell "
                "centres in this window (outside the site or narrower than "
                "one pixel); the Building_DSM is unchanged by this block"
            )
        if self.edge_touching:
            notes.append(
                f"building {self.building_id!r}: footprint touches the raster "
                "edge; findwalls zeroes border cells, so wall entries along "
                "that edge are dropped downstream"
            )
        return tuple(notes)


@dataclass(frozen=True, slots=True)
class BuildingRasterResult:
    """One rasterization pass: the scenario raster plus per-edit records."""

    raster: np.ndarray
    records: tuple[MassingRasterRecord, ...]

    @property
    def disclosures(self) -> tuple[str, ...]:
        notes: list[str] = []
        for record in self.records:
            notes.extend(record.disclosures)
        return tuple(notes)


# ---------------------------------------------------------------------------
# The layer
# ---------------------------------------------------------------------------


class BuildingLayer:
    """Base massing raster plus an ordered log of committed massing edits.

    Mirrors :class:`solweig_gpu.incremental.trees.TreeLayer`: the base
    Building_DSM and DEM are immutable snapshots, edits are recorded in
    arrival order, and rasterization reconstructs the scenario surface from
    world coordinates alone (so a window rasterization always equals the
    matching slice of the full-domain rasterization). Unlike the tree
    layer, blocks live in *absolute* elevation, so the DEM travels with the
    layer and grounds every paint and reset.
    """

    def __init__(
        self,
        base_building_dsm: np.ndarray,
        dem: np.ndarray,
        grid: RasterGrid,
        *,
        scenario_id: str = "default",
    ) -> None:
        if not scenario_id:
            raise ValueError("scenario_id must be non-empty")
        base = np.array(base_building_dsm, dtype=np.float32, copy=True)
        ground = np.array(dem, dtype=np.float32, copy=True)
        expected = (grid.rows, grid.cols)
        if base.shape != expected:
            raise ValueError(
                f"base_building_dsm must have shape {expected}, got {base.shape}"
            )
        if ground.shape != expected:
            raise ValueError(
                f"dem must have shape {expected}, got {ground.shape}"
            )
        self._base = base
        self._dem = ground
        self._grid = grid
        self._scenario_id = scenario_id
        self._edits: list[MassingEdit] = []
        self._live: dict[str, BuildingSpec] = {}

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
    def base_building_dsm(self) -> np.ndarray:
        """Read-only view of the immutable base surface (absolute metres)."""
        view = self._base.view()
        view.flags.writeable = False
        return view

    @property
    def dem(self) -> np.ndarray:
        """Read-only view of the immutable ground raster (absolute metres)."""
        view = self._dem.view()
        view.flags.writeable = False
        return view

    @property
    def edits(self) -> tuple[MassingEdit, ...]:
        return tuple(self._edits)

    def current_buildings(self) -> tuple[BuildingSpec, ...]:
        """Live edited blocks, deterministically sorted by ``building_id``."""
        return tuple(self._live[id_] for id_ in sorted(self._live))

    # ------------------------------------------------------------------
    # Edits
    # ------------------------------------------------------------------

    def apply_edit(self, edit: MassingEdit) -> None:
        """Record one committed massing edit (validated hand-off).

        Refusals (all pre-record, layer state untouched):

        * a non-:class:`MassingEdit` payload;
        * an ``add`` over an id this layer already carries;
        * a ``before`` spec that disagrees with the layer's live spec for
          that id (a stale hand-off — the executor folded edits out of
          order or from a different scene);
        * a no-op record (``before == after``).
        """
        if not isinstance(edit, MassingEdit):
            raise ValueError(
                f"expected MassingEdit, got {type(edit).__name__}"
            )
        if edit.before is not None and edit.before == edit.after:
            raise ValueError(
                f"building {edit.building_id!r}: before and after specs are "
                "identical (a no-op edit)"
            )
        live = self._live.get(edit.building_id)
        if edit.after is not None and edit.before is None:
            if live is not None:
                raise ValueError(
                    f"building {edit.building_id!r} already exists in "
                    f"scenario {self._scenario_id!r}"
                )
        if edit.before is not None and live is not None and live != edit.before:
            raise ValueError(
                f"building {edit.building_id!r}: stale before spec (layer "
                f"holds {live!r}, edit claims {edit.before!r})"
            )
        self._edits.append(edit)
        if edit.after is None:
            self._live.pop(edit.building_id, None)
        else:
            self._live[edit.building_id] = edit.after

    def apply_edits(self, edits: Sequence[MassingEdit]) -> None:
        """Record edits in arrival order (first refusal wins)."""
        for edit in edits:
            self.apply_edit(edit)

    # ------------------------------------------------------------------
    # Rasterization
    # ------------------------------------------------------------------

    def rasterize_window(self, window: RasterWindow) -> np.ndarray:
        """Window-local scenario Building_DSM (absolute elevation, float32)."""
        return self.rasterize_window_with_records(window).raster

    def rasterize_window_with_records(
        self, window: RasterWindow,
    ) -> BuildingRasterResult:
        """Rasterize the window, returning the raster and per-edit records.

        Every edit is replayed in arrival order: the ``before`` footprint
        (if any) is reset to DEM ground, then the ``after`` footprint (if
        any) is painted at ``ground + height_m``, combining with ``fmax``.
        """
        self._check_window(window)
        raster = self._base[
            window.row_start : window.row_stop, window.col_start : window.col_stop
        ].copy()
        ground = self._dem[
            window.row_start : window.row_stop, window.col_start : window.col_stop
        ]
        records: list[MassingRasterRecord] = []
        for edit in self._edits:
            records.append(
                self._apply_edit(raster, ground, window, edit)
            )
        return BuildingRasterResult(
            raster=raster, records=tuple(records)
        )

    def rasterize(self) -> np.ndarray:
        """Full-domain scenario Building_DSM (absolute elevation, float32)."""
        return self.rasterize_window(self._grid.full_window)

    def rasterize_with_records(self) -> BuildingRasterResult:
        """Full-domain rasterization with per-edit records and disclosures."""
        return self.rasterize_window_with_records(self._grid.full_window)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _apply_edit(
        self,
        raster: np.ndarray,
        ground: np.ndarray,
        window: RasterWindow,
        edit: MassingEdit,
    ) -> MassingRasterRecord:
        cells_reset = 0
        edge_touching = False
        if edit.before is not None:
            reset_mask, edge_before = polygon_cell_mask(
                edit.before.footprint_m, self._grid, window
            )
            edge_touching = edge_before
            if reset_mask.any():
                self._require_finite_ground(
                    ground, reset_mask, edit.building_id, "before"
                )
                raster[reset_mask] = ground[reset_mask]
                cells_reset = int(reset_mask.sum())
        cells_painted = 0
        walls_expected = False
        if edit.after is not None:
            paint_mask, edge_after = polygon_cell_mask(
                edit.after.footprint_m, self._grid, window
            )
            edge_touching = edge_touching or edge_after
            if paint_mask.any():
                self._require_finite_ground(
                    ground, paint_mask, edit.building_id, "after"
                )
                surface = ground + np.float32(edit.after.height_m)
                # fmax ONLY on painted cells: the block wins over a NaN
                # nodata base cell it legitimately covers, while unpainted
                # cells (NaN or not) keep their base value untouched.
                region = raster[paint_mask]
                raster[paint_mask] = np.fmax(region, surface[paint_mask])
                cells_painted = int(paint_mask.sum())
            walls_expected = edit.after.height_m >= WALL_LIMIT_M
        operation = (
            "add"
            if edit.before is None
            else ("delete" if edit.after is None else "update")
        )
        return MassingRasterRecord(
            building_id=edit.building_id,
            operation=operation,
            cells_painted=cells_painted,
            cells_reset=cells_reset,
            walls_expected=walls_expected,
            edge_touching=bool(edge_touching),
        )

    @staticmethod
    def _require_finite_ground(
        ground: np.ndarray,
        mask: np.ndarray,
        building_id: str,
        role: str,
    ) -> None:
        bad = int(np.count_nonzero(~np.isfinite(ground[mask])))
        if bad:
            raise ValueError(
                f"building {building_id!r}: {bad} footprint cells ({role}) "
                "carry non-finite DEM ground; refusing to rasterize massing "
                "onto undefined ground (fix or nodata-fill the DEM first)"
            )

    def _check_window(self, window: RasterWindow) -> None:
        rows, cols = self._grid.rows, self._grid.cols
        if not (
            0 <= window.row_start <= window.row_stop <= rows
            and 0 <= window.col_start <= window.col_stop <= cols
        ):
            raise ValueError(
                f"window {window} must lie inside the {rows}x{cols} grid"
            )
