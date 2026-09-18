# SPDX-License-Identifier: GPL-3.0-only
"""Unit tests for deterministic scientific fixture generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from solweig_gpu.incremental.fixtures import (
    FIXTURE_TREE_BASE,
    build_fixture_states,
    find_open_cell,
    rasterize_tree,
)
from solweig_gpu.incremental.geometry import RasterGrid, TreeSpec

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


@pytest.fixture()
def source_dir(tmp_path: Path, grid: RasterGrid) -> Path:
    """Synthetic flat open-terrain site with one distant building cluster."""
    import rasterio
    from rasterio.transform import from_origin

    rng = np.random.default_rng(7)
    dem = np.full((ROWS, COLS), 190.0, dtype="float32")
    building = dem + rng.normal(0, 0.1, (ROWS, COLS)).astype("float32")
    # small building cluster far from the centre
    building[4:10, 4:10] += 12.0
    trees = np.zeros((ROWS, COLS), dtype="float32")

    transform = from_origin(ORIGIN_X, ORIGIN_Y, PIXEL, PIXEL)
    for name, array in (
        ("Trees.tif", trees),
        ("Building_DSM.tif", building),
        ("DEM.tif", dem),
    ):
        profile = {
            "driver": "GTiff",
            "height": ROWS,
            "width": COLS,
            "count": 1,
            "dtype": "float32",
            "transform": transform,
            "crs": "EPSG:32614",
        }
        with rasterio.open(tmp_path / name, "w", **profile) as dataset:
            dataset.write(array, 1)
    return tmp_path


def centre_world(grid: RasterGrid) -> tuple[float, float]:
    row = ROWS // 2
    col = COLS // 2
    return (
        grid.origin_x_m + (col + 0.5) * PIXEL,
        grid.origin_y_m - (row + 0.5) * PIXEL,
    )


def test_rasterize_tree_paints_disc_of_canopy_height(grid: RasterGrid) -> None:
    x_m, y_m = centre_world(grid)
    tree = TreeSpec(
        tree_id="t1", x_m=x_m, y_m=y_m, height_m=10.0, canopy_radius_m=6.0
    )
    vegetation = np.zeros((ROWS, COLS), dtype="float32")
    out = rasterize_tree(vegetation, tree, grid)
    painted = out > 0
    assert painted.sum() > 0
    # every painted cell carries the canopy top height
    assert np.all(out[painted] == 10.0)
    # input untouched
    assert np.all(vegetation == 0)
    # cell containing the trunk is painted
    row = ROWS // 2
    col = COLS // 2
    assert out[row, col] == 10.0


def test_rasterize_tree_combines_overlaps_with_maximum(grid: RasterGrid) -> None:
    x_m, y_m = centre_world(grid)
    short = TreeSpec("t1", x_m, y_m, height_m=8.0, canopy_radius_m=6.0)
    tall = TreeSpec("t2", x_m + 4.0, y_m, height_m=15.0, canopy_radius_m=6.0)
    vegetation = np.zeros((ROWS, COLS), dtype="float32")
    out = rasterize_tree(rasterize_tree(vegetation, short, grid), tall, grid)
    row = ROWS // 2
    col = COLS // 2
    # overlap cell: never sum heights, take the maximum
    assert out[row, col] == 15.0
    assert out.max() == 15.0


def test_rasterize_tree_clips_at_site_boundary(grid: RasterGrid) -> None:
    tree = TreeSpec("t1", grid.origin_x_m + 1.0, grid.origin_y_m - 1.0, 10.0, 8.0)
    vegetation = np.zeros((ROWS, COLS), dtype="float32")
    out = rasterize_tree(vegetation, tree, grid)
    assert out[0, 0] == 10.0
    # nothing painted outside the array (guaranteed by construction)
    assert out.shape == vegetation.shape


def test_find_open_cell_avoids_buildings_and_vegetation(
    source_dir: Path, grid: RasterGrid
) -> None:
    import rasterio

    with rasterio.open(source_dir / "Trees.tif") as dataset:
        vegetation = dataset.read(1)
    with rasterio.open(source_dir / "Building_DSM.tif") as dataset:
        building = dataset.read(1)
    with rasterio.open(source_dir / "DEM.tif") as dataset:
        dem = dataset.read(1)
    x_m, y_m = find_open_cell(vegetation, building, dem, grid, clearance_m=12.0)
    col = int((x_m - grid.origin_x_m) / PIXEL)
    row = int((grid.origin_y_m - y_m) / PIXEL)
    clearance_px = int(12.0 / PIXEL)
    region_bld = building[
        row - clearance_px : row + clearance_px + 1,
        col - clearance_px : col + clearance_px + 1,
    ]
    region_dem = dem[
        row - clearance_px : row + clearance_px + 1,
        col - clearance_px : col + clearance_px + 1,
    ]
    assert not np.any(region_bld - region_dem > 2.0)
    assert vegetation[
        row - clearance_px : row + clearance_px + 1,
        col - clearance_px : col + clearance_px + 1,
    ].max() == 0


def test_build_fixture_states_manifest_and_invariants(
    source_dir: Path, tmp_path: Path
) -> None:
    target = tmp_path / "fixture"
    manifest = build_fixture_states(source_dir, target)

    assert manifest["schema_version"] == 1
    assert set(manifest["states"]) == {"baseline", "add", "move", "resize", "delete"}
    assert manifest["invariants"]["delete_equals_baseline"]
    assert manifest["invariants"]["add_changes_cells"]
    assert manifest["invariants"]["move_changes_cells"]
    assert manifest["invariants"]["resize_changes_cells"]

    import rasterio

    hashes = {name: record["sha256"] for name, record in manifest["states"].items()}
    files = {name: record["file"] for name, record in manifest["states"].items()}
    assert hashes["baseline"] == hashes["delete"]
    with rasterio.open(target / files["baseline"]) as baseline:
        base = baseline.read(1)
    with rasterio.open(target / files["add"]) as add:
        added = add.read(1)
    with rasterio.open(target / files["move"]) as move:
        moved = move.read(1)
    with rasterio.open(target / files["resize"]) as resize:
        resized = resize.read(1)

    # add/move/resize change only cells inside their own crown footprints
    for state, tree_key in (
        (added, "add"),
        (moved, "move_to"),
        (resized, "resize"),
    ):
        changed = state != base
        assert changed.any()
        tree = manifest["trees"][tree_key]
        grid = grid_of(manifest)
        radius = max(
            manifest["trees"]["add"]["canopy_radius_m"],
            manifest["trees"]["resize"]["canopy_radius_m"],
        ) + 2 * PIXEL
        row = int((grid.origin_y_m - tree["y_m"]) / PIXEL)
        col = int((tree["x_m"] - grid.origin_x_m) / PIXEL)
        pad = int(np.ceil(radius / PIXEL))
        outside = np.ones_like(changed)
        outside[
            max(0, row - pad) : row + pad + 1, max(0, col - pad) : col + pad + 1
        ] = False
        assert not np.any(changed & outside), tree_key


def grid_of(manifest: dict) -> RasterGrid:
    grid = manifest["grid"]
    return RasterGrid(
        rows=grid["rows"],
        cols=grid["cols"],
        pixel_size_m=grid["pixel_size_m"],
        origin_x_m=grid["origin_x_m"],
        origin_y_m=grid["origin_y_m"],
    )


def test_default_fixture_tree_matches_documented_ranges() -> None:
    assert 3.0 <= FIXTURE_TREE_BASE.height_m <= 40.0
    assert 1.0 <= 2 * FIXTURE_TREE_BASE.canopy_radius_m <= 30.0
    assert 0.0 <= FIXTURE_TREE_BASE.trunk_ratio <= 1.0
    assert 0.0 <= FIXTURE_TREE_BASE.transmissivity <= 1.0
