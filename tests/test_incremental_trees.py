# SPDX-License-Identifier: GPL-3.0-only
"""Unit tests for editable-tree rasterization and spatial queries (Phase 4)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from solweig_gpu.incremental.edits import EditOperation
from solweig_gpu.incremental.fixtures import rasterize_tree
from solweig_gpu.incremental.geometry import (
    InfluenceConfig,
    RasterGrid,
    RasterWindow,
    SunPosition,
    TreeSpec,
)
from solweig_gpu.incremental.spatial_index import TreeSpatialIndex, WorldRect
from solweig_gpu.incremental.trees import (
    TreeLayer,
    validate_tree_preset,
    validate_tree_spec,
)

ROWS = COLS = 64
PIXEL = 2.0
ORIGIN_X = 621734.7066
ORIGIN_Y = 3354614.3479


@pytest.fixture()
def grid() -> RasterGrid:
    return RasterGrid(
        rows=ROWS,
        cols=COLS,
        pixel_size_m=PIXEL,
        origin_x_m=ORIGIN_X,
        origin_y_m=ORIGIN_Y,
    )


def world_centre(grid: RasterGrid, row: int, col: int) -> tuple[float, float]:
    return (
        grid.origin_x_m + (col + 0.5) * PIXEL,
        grid.origin_y_m - (row + 0.5) * PIXEL,
    )


@pytest.fixture()
def base_vegetation() -> np.ndarray:
    """Scattered synthetic base vegetation (height above ground, metres)."""
    rng = np.random.default_rng(11)
    base = np.zeros((ROWS, COLS), dtype="float32")
    for row, col, height in (
        (10, 12, 7.0),
        (11, 13, 9.5),
        (40, 50, 12.0),
        (55, 6, 6.5),
    ):
        base[row - 2 : row + 3, col - 2 : col + 3] = rng.uniform(
            height - 1.0, height + 1.0, (5, 5)
        ).astype("float32")
    return base


@pytest.fixture()
def trees_path(tmp_path: Path, base_vegetation: np.ndarray) -> Path:
    import rasterio
    from rasterio.transform import from_origin

    transform = from_origin(ORIGIN_X, ORIGIN_Y, PIXEL, PIXEL)
    profile = {
        "driver": "GTiff",
        "height": ROWS,
        "width": COLS,
        "count": 1,
        "dtype": "float32",
        "transform": transform,
        "crs": "EPSG:32614",
    }
    with rasterio.open(tmp_path / "Trees.tif", "w", **profile) as dataset:
        dataset.write(base_vegetation, 1)
    return tmp_path / "Trees.tif"


@pytest.fixture()
def layer(trees_path: Path) -> TreeLayer:
    return TreeLayer.from_geotiff(trees_path)


def make_tree(tree_id: str, row: int, col: int, **overrides: float) -> TreeSpec:
    x_m = ORIGIN_X + (col + 0.5) * PIXEL
    y_m = ORIGIN_Y - (row + 0.5) * PIXEL
    defaults = {
        "height_m": 10.0,
        "canopy_radius_m": 6.0,
        "trunk_ratio": 0.25,
        "transmissivity": 0.03,
    }
    return TreeSpec(tree_id=tree_id, x_m=x_m, y_m=y_m, **{**defaults, **overrides})


# ----------------------------------------------------------------------
# Local rasterization equals full rasterization sliced to the window
# ----------------------------------------------------------------------


def test_local_rasterization_equals_full_sliced(layer: TreeLayer, grid: RasterGrid) -> None:
    layer.add_tree(make_tree("t_centre", 32, 32))
    layer.add_tree(make_tree("t_overlap", 32, 34, height_m=16.0))
    layer.add_tree(make_tree("t_edge", 63, 63))
    layer.move_tree("t_centre", x_m=ORIGIN_X + 20.5 * PIXEL, y_m=ORIGIN_Y - 44.5 * PIXEL)

    full_canopy, full_trunk = layer.vegetation_rasters_window(grid.full_window)
    windows = [
        RasterWindow(0, 0, 0, 0),
        RasterWindow(0, 0, 0, 1),
        RasterWindow(0, 4, 0, 4),          # top-left corner
        RasterWindow(60, 64, 60, 64),      # bottom-right corner
        RasterWindow(0, 64, 60, 64),       # full-height right edge strip
        RasterWindow(28, 40, 28, 40),      # old tree position
        RasterWindow(40, 50, 16, 26),      # new moved-tree position
        RasterWindow(63, 64, 63, 64),      # single corner pixel
        grid.full_window,
    ]
    for window in windows:
        canopy, trunk = layer.vegetation_rasters_window(window)
        assert canopy.shape == (window.height, window.width)
        assert canopy.dtype == np.float32 and trunk.dtype == np.float32
        np.testing.assert_array_equal(
            canopy,
            full_canopy[window.row_start : window.row_stop, window.col_start : window.col_stop],
            err_msg=f"canopy mismatch for window {window}",
        )
        np.testing.assert_array_equal(
            trunk,
            full_trunk[window.row_start : window.row_stop, window.col_start : window.col_stop],
            err_msg=f"trunk mismatch for window {window}",
        )


def test_boundary_clipping_matches_full_convention(
    trees_path: Path, grid: RasterGrid
) -> None:
    # Tree centred exactly on the outer north-west corner of the domain.
    corner = TreeSpec("t_corner", ORIGIN_X, ORIGIN_Y, height_m=10.0, canopy_radius_m=8.0)
    # Tree straddling the eastern edge.
    edge = TreeSpec("t_edge", ORIGIN_X + 64 * PIXEL, ORIGIN_Y - 32 * PIXEL, 12.0, 6.0)

    layer = TreeLayer.from_geotiff(trees_path)
    layer.add_tree(corner)
    layer.add_tree(edge)

    full = layer.rasterize_window(grid.full_window)
    # Cross-check the canopy combination against the full-domain fixture
    # rasterizer painted over the same base.
    reference = rasterize_tree(layer.base_vegetation, corner, grid)
    reference = rasterize_tree(reference, edge, grid)
    np.testing.assert_array_equal(full, reference)
    assert full[0, 0] == 10.0  # corner cell inside the quarter disc

    window = RasterWindow(0, 8, 0, 8)
    np.testing.assert_array_equal(
        layer.rasterize_window(window), full[0:8, 0:8]
    )


def test_translation_invariance() -> None:
    rng = np.random.default_rng(42)
    for _ in range(8):
        dx_px = int(rng.integers(-6, 7))
        dy_px = int(rng.integers(-6, 7))
        tree = make_tree("t", 20, 20)
        grid_a = RasterGrid(ROWS, COLS, PIXEL, ORIGIN_X, ORIGIN_Y)
        grid_b = RasterGrid(
            ROWS, COLS, PIXEL, ORIGIN_X + dx_px * PIXEL, ORIGIN_Y - dy_px * PIXEL
        )
        zeros_a = np.zeros((ROWS, COLS), dtype="float32")
        zeros_b = np.zeros((ROWS, COLS), dtype="float32")
        layer_a = TreeLayer(zeros_a, grid_a)
        layer_a.add_tree(tree)
        layer_b = TreeLayer(zeros_b, grid_b)
        layer_b.add_tree(tree)

        window_a = RasterWindow(20 - 12, 20 + 13, 20 - 12, 20 + 13).clamp(
            rows=ROWS, cols=COLS
        )
        # Shifting the grid origin by (dx, dy) pixels moves the tree to
        # (row - dy, col - dx) in index space, so the matching window shifts
        # the opposite way.
        window_b = RasterWindow(
            window_a.row_start - dy_px,
            window_a.row_stop - dy_px,
            window_a.col_start - dx_px,
            window_a.col_stop - dx_px,
        )
        canopy_a, trunk_a = layer_a.vegetation_rasters_window(window_a)
        canopy_b, trunk_b = layer_b.vegetation_rasters_window(window_b)
        assert canopy_a.sum() > 0
        np.testing.assert_array_equal(canopy_a, canopy_b)
        np.testing.assert_array_equal(trunk_a, trunk_b)


# ----------------------------------------------------------------------
# Combination rules
# ----------------------------------------------------------------------


def test_overlapping_crowns_combine_with_maximum(layer: TreeLayer, grid: RasterGrid) -> None:
    short = make_tree("short", 32, 30, height_m=8.0, trunk_ratio=0.9)
    tall = make_tree("tall", 32, 33, height_m=15.0, trunk_ratio=0.25)
    layer.add_tree(short)
    layer.add_tree(tall)

    canopy, trunk = layer.vegetation_rasters_window(RasterWindow(24, 40, 24, 44))
    # Overlap cell: maximum height, never a sum (8 + 15 = 23 must not appear).
    assert canopy.max() == 15.0
    assert canopy.min() >= 0.0
    assert not np.any(canopy > 15.0)
    # Trunk zone paints per tree at trunk_ratio * height and combines with max:
    # short -> 0.9 * 8 = 7.2 wins over tall -> 0.25 * 15 = 3.75.
    assert trunk.max() == pytest.approx(7.2, abs=1e-6)
    assert np.all(trunk <= canopy + 1e-6)

    # Order independence: adding in the opposite order gives identical rasters.
    layer_reversed = TreeLayer(layer.base_vegetation.copy(), grid)
    layer_reversed.add_tree(tall)
    layer_reversed.add_tree(short)
    canopy_r, trunk_r = layer_reversed.vegetation_rasters_window(RasterWindow(24, 40, 24, 44))
    np.testing.assert_array_equal(canopy, canopy_r)
    np.testing.assert_array_equal(trunk, trunk_r)


def test_trunk_zone_matches_oracle_fraction_for_default_ratio(
    trees_path: Path, grid: RasterGrid
) -> None:
    layer = TreeLayer.from_geotiff(trees_path)
    layer.add_tree(make_tree("t", 30, 30))
    canopy, trunk = layer.vegetation_rasters_window(grid.full_window)
    # vegdem2 = trees * 0.25 + dem, so trunk-above-ground == canopy * 0.25.
    np.testing.assert_allclose(trunk, canopy * np.float32(0.25), rtol=0, atol=1e-7)


def test_canopy_height_painted_across_disc_on_raised_ground(grid: RasterGrid) -> None:
    """P8 review follow-up: the whole crown disc carries ``height_m``.

    ``ExactWorker._exact_influence_window`` derives the march's tallest
    occluder as ``base = max(a)`` over the crown-reach box plus the tree
    height — i.e. it assumes the rasterizer paints the canopy top at the
    tree height on EVERY cell of the footprint disc, each on that cell's
    own ground datum (``vegdsm = canopy + building_dsm``,
    ``vegdsm2 = trunk + dem``, max-combination per P4). A rasterizer that
    painted only the centre cell would break that assumption silently:
    the tallest occluder would be ``a[centre] + height`` instead of
    ``max(a under disc) + height``.

    The terrain here raises one cell inside the disc 5 m above the tree's
    centre cell, so the two rules disagree: this test fails under
    centre-cell-only painting.
    """
    from math import ceil

    centre_row, centre_col = 32, 32
    tree = make_tree("t", centre_row, centre_col, height_m=10.0, canopy_radius_m=6.0)
    high_row, high_col = 30, 34  # 5.66 m from the centre: inside the 6 m disc
    surface = np.zeros((ROWS, COLS), dtype="float32")  # building DSM == DEM
    surface[high_row, high_col] = 5.0
    assert surface[centre_row, centre_col] == 0.0  # higher cell is not the centre

    layer = TreeLayer(np.zeros((ROWS, COLS), dtype="float32"), grid)
    layer.add_tree(tree)
    canopy, trunk = layer.vegetation_rasters_window(grid.full_window)

    # Canopy top painted on the whole disc, including the raised cell.
    assert canopy[centre_row, centre_col] == np.float32(10.0)
    assert canopy[high_row, high_col] == np.float32(10.0)
    # Trunk zone likewise painted across the disc (P4 max-combination).
    assert trunk[high_row, high_col] == np.float32(0.25 * 10.0)

    # vegdsm / vegdem2 datums at the raised cell: its own ground + canopy.
    vegdsm = canopy + surface
    vegdsm[vegdsm == surface] = 0.0  # the pipeline's no-canopy quirk
    vegdem2 = trunk + surface
    assert vegdsm[high_row, high_col] == np.float32(5.0 + 10.0)
    assert vegdem2[high_row, high_col] == np.float32(5.0 + 0.25 * 10.0)

    # The tallest occluder under the crown equals exactly the value the
    # influence mask computes: max(a) over the reach box, plus height.
    config = InfluenceConfig()
    reach_px = ceil((tree.canopy_radius_m + config.safety_margin_m) / PIXEL)
    box = surface[
        centre_row - reach_px : centre_row + reach_px + 1,
        centre_col - reach_px : centre_col + reach_px + 1,
    ]
    base = float(box.max())
    assert base == 5.0
    disc = (canopy > 0) & (
        np.abs(np.arange(ROWS)[:, None] - centre_row) <= reach_px
    ) & (np.abs(np.arange(COLS)[None, :] - centre_col) <= reach_px)
    assert float(vegdsm[disc].max()) == np.float32(base + tree.height_m)


# ----------------------------------------------------------------------
# TREE-001: every editable property is represented in rasterization
# ----------------------------------------------------------------------


def test_tree001_every_editable_property_changes_raster(
    trees_path: Path, grid: RasterGrid
) -> None:
    def raster(tree: TreeSpec) -> tuple[np.ndarray, np.ndarray]:
        layer = TreeLayer.from_geotiff(trees_path)
        layer.add_tree(tree)
        return layer.vegetation_rasters_window(grid.full_window)

    base_tree = make_tree("t", 32, 32)
    base_canopy, base_trunk = raster(base_tree)

    variants = {
        "x_m": replace(base_tree, x_m=base_tree.x_m + 4 * PIXEL),
        "y_m": replace(base_tree, y_m=base_tree.y_m - 4 * PIXEL),
        "height_m": replace(base_tree, height_m=14.0),
        "canopy_radius_m": replace(base_tree, canopy_radius_m=9.0),
    }
    for name, tree in variants.items():
        canopy, trunk = raster(tree)
        assert not np.array_equal(canopy, base_canopy), name
        assert not np.array_equal(trunk, base_trunk), name

    # trunk_ratio changes only the trunk zone.
    canopy, trunk = raster(replace(base_tree, trunk_ratio=0.6))
    np.testing.assert_array_equal(canopy, base_canopy)
    assert not np.array_equal(trunk, base_trunk)
    assert trunk.max() == pytest.approx(6.0, abs=1e-6)


# ----------------------------------------------------------------------
# TREE-002: base raster immutability
# ----------------------------------------------------------------------


def test_tree002_base_raster_unchanged_after_edits(
    layer: TreeLayer, base_vegetation: np.ndarray, grid: RasterGrid
) -> None:
    assert not layer.base_vegetation.flags.writeable
    layer.add_tree(make_tree("t", 32, 32))
    layer.move_tree("t", x_m=ORIGIN_X + 40.5 * PIXEL, y_m=ORIGIN_Y - 20.5 * PIXEL)
    layer.update_tree("t", height_m=18.0, canopy_radius_m=8.0)
    layer.vegetation_rasters_window(grid.full_window)
    layer.affected_window(sun_positions=(SunPosition(30.0, 90.0),))
    layer.delete_tree("t")
    # Bitwise comparison against the caller-owned array and the read-only view.
    assert layer.base_vegetation.tobytes() == base_vegetation.tobytes()
    np.testing.assert_array_equal(layer.base_vegetation, base_vegetation)
    # And rasterizing still works afterwards.
    canopy, _ = layer.vegetation_rasters_window(grid.full_window)
    np.testing.assert_array_equal(canopy, base_vegetation)


# ----------------------------------------------------------------------
# TREE-003: move and delete restore old regions
# ----------------------------------------------------------------------


def test_tree003_move_and_delete_restore_base(
    layer: TreeLayer, base_vegetation: np.ndarray, grid: RasterGrid
) -> None:
    layer.add_tree(make_tree("t", 18, 18))
    layer.move_tree("t", x_m=ORIGIN_X + 46.5 * PIXEL, y_m=ORIGIN_Y - 46.5 * PIXEL)

    # The old-position window is exactly the base raster again (the moved
    # crown no longer touches it).
    old_window = RasterWindow(12, 25, 12, 25)
    np.testing.assert_array_equal(
        layer.rasterize_window(old_window), base_vegetation[12:25, 12:25]
    )
    # The new position is painted.
    new_window = RasterWindow(40, 54, 40, 54)
    assert not np.array_equal(
        layer.rasterize_window(new_window), base_vegetation[40:54, 40:54]
    )

    # Deleting the added tree restores the full raster to the base.
    layer.delete_tree("t")
    canopy, trunk = layer.vegetation_rasters_window(grid.full_window)
    np.testing.assert_array_equal(canopy, base_vegetation)
    np.testing.assert_array_equal(trunk, base_vegetation * np.float32(0.25))


# ----------------------------------------------------------------------
# Edit log, coalescing, and dirty windows
# ----------------------------------------------------------------------


def test_edit_log_operations_and_coalescing(layer: TreeLayer, grid: RasterGrid) -> None:
    tree = make_tree("t", 30, 30)
    layer.add_tree(tree)
    layer.move_tree("t", x_m=tree.x_m + 6 * PIXEL, y_m=tree.y_m)
    layer.update_tree("t", height_m=16.0, canopy_radius_m=7.0)

    assert [edit.operation for edit in layer.edits] == [
        EditOperation.ADD,
        EditOperation.MOVE,
        EditOperation.UPDATE,
    ]
    batch = layer.coalesced_batch()
    assert batch is not None
    assert len(batch.edits) == 1
    coalesced = batch.edits[0]
    assert coalesced.operation == EditOperation.ADD
    assert coalesced.old_tree is None
    assert coalesced.new_tree == layer.current_trees()[0]

    dirty = layer.affected_window()
    assert not dirty.is_empty
    # Covers both the original and the moved/updated crowns.
    assert dirty.row_start <= 30 and dirty.row_stop >= 34
    assert dirty.col_start <= 30 and dirty.col_stop >= 37

    # add -> delete coalesces away entirely.
    layer.delete_tree("t")
    batch = layer.coalesced_batch()
    assert batch is not None
    assert batch.edits == ()
    assert layer.current_trees() == ()
    # Empty edit list yields an empty window.
    empty_layer = TreeLayer(layer.base_vegetation, layer.grid)
    assert empty_layer.affected_window().is_empty


def test_window_outside_grid_rejected(layer: TreeLayer) -> None:
    with pytest.raises(ValueError):
        layer.vegetation_rasters_window(RasterWindow(-2, 4, 0, 4))
    with pytest.raises(ValueError):
        layer.vegetation_rasters_window(RasterWindow(0, 4, 0, COLS + 1))


# ----------------------------------------------------------------------
# Spatial queries
# ----------------------------------------------------------------------


def test_spatial_query_excludes_distant_trees_and_includes_influence(
    grid: RasterGrid,
) -> None:
    # Compact footprints so exclusion on a 128 m synthetic domain is possible.
    config = InfluenceConfig(
        lowest_sky_patch_altitude_deg=45.0,
        maximum_shadow_length_m=60.0,
        safety_margin_m=2.0,
    )
    west = make_tree("west", 32, 10)  # height 10 m -> footprint radius 12 m
    east = make_tree("east", 32, 40, height_m=20.0)  # footprint radius 22 m
    window = RasterWindow(30, 35, 55, 60)
    rect = WorldRect.from_window(window, grid)

    index = TreeSpatialIndex([west, east], grid=grid, config=config)
    assert index.query_crown_window(window) == ()
    assert index.query_crown(rect) == ()
    # Without the low sun the influence footprints stop short of the window
    # (the eastern box reaches x ~ 103 m; the window starts at x = 110 m).
    assert east not in index.query_influence(rect)
    assert west not in index.query_influence(rect)

    # With a low western sun (azimuth 270 deg) shadows extend east: the
    # eastern tree's corridor reaches past the window, the western tree's
    # does not.
    low_sun = (SunPosition(altitude_deg=6.0, azimuth_deg=270.0),)
    index_sun = TreeSpatialIndex(
        [west, east], grid=grid, config=config, sun_positions=low_sun
    )
    affected = index_sun.query_influence_window(window)
    assert east in affected
    assert west not in affected
    # Influence queries are supersets of crown queries.
    assert set(index.query_crown(rect)) <= set(index.query_influence(rect))
    # Deterministic ordering by tree_id regardless of insertion order.
    both = TreeSpatialIndex([east, west], grid=grid).query_influence(
        WorldRect.from_window(grid.full_window, grid)
    )
    assert [tree.tree_id for tree in both] == ["east", "west"]


def test_spatial_index_per_edit_affected_trees(layer: TreeLayer, grid: RasterGrid) -> None:
    # Compact footprints keep the affected sets small on the synthetic site.
    config = InfluenceConfig(
        lowest_sky_patch_altitude_deg=45.0,
        maximum_shadow_length_m=20.0,
        safety_margin_m=2.0,
    )
    tree_a = make_tree("a", 30, 30)
    tree_b = make_tree("b", 32, 32)
    layer.add_tree(tree_a)
    layer.add_tree(tree_b)
    index = layer.build_index(config=config)

    # Move tree a next to b: b is affected by the edit (dirty region overlaps
    # b's footprint); a itself is also reported because it is indexed.
    x_new, y_new = world_centre(grid, 33, 31)
    move_edit = layer.move_tree("a", x_m=x_new, y_m=y_new)
    ids = [tree.tree_id for tree in index.affected_trees(move_edit)]
    assert "b" in ids
    assert "a" in ids

    # An add whose influence stays clear of b does not affect it; a nearby
    # add does.
    far_edit = layer.add_tree(make_tree("far", 8, 8))
    assert "b" not in [tree.tree_id for tree in index.affected_trees(far_edit)]
    near_edit = layer.add_tree(make_tree("near", 32, 33))
    assert "b" in [tree.tree_id for tree in index.affected_trees(near_edit)]

    # Delete edit dirties the deleted tree's old region.
    delete_edit = layer.delete_tree("b")
    assert "a" in [tree.tree_id for tree in index.affected_trees(delete_edit)]


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------


def test_validation_rejects_invalid_specs(layer: TreeLayer) -> None:
    # Structural rejects already enforced at construction time.
    for overrides in (
        {"height_m": 0.0},
        {"height_m": -5.0},
        {"canopy_radius_m": 0.0},
        {"canopy_radius_m": -1.0},
        {"trunk_ratio": -0.1},
        {"transmissivity": 1.5},
        {"x_m": float("nan")},
        {"y_m": float("inf")},
    ):
        fields = {
            "x_m": 0.0,
            "y_m": 0.0,
            "height_m": 10.0,
            "canopy_radius_m": 5.0,
        }
        fields.update(overrides)
        with pytest.raises(ValueError):
            TreeSpec(tree_id="bad", **fields)

    # trunk_ratio == 1 constructs but is rejected server-side.
    full_trunk = make_tree("full_trunk", 20, 20, trunk_ratio=1.0)
    with pytest.raises(ValueError, match="trunk_ratio"):
        validate_tree_spec(full_trunk)
    with pytest.raises(ValueError, match="trunk_ratio"):
        layer.add_tree(full_trunk)

    # Documented preset ranges.
    with pytest.raises(ValueError, match="height_m"):
        validate_tree_preset(make_tree("small", 20, 20, height_m=2.0))
    with pytest.raises(ValueError, match="canopy diameter"):
        validate_tree_preset(make_tree("wide", 20, 20, canopy_radius_m=20.0))
    validate_tree_preset(make_tree("ok", 20, 20))
    assert layer.current_trees() == ()
