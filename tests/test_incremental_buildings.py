# SPDX-License-Identifier: GPL-3.0-only
"""U-C packet 5 tests: building rasterizer + full regeneration chain.

Fast suite (no GDAL/torch): the rasterization geometry of
:class:`~solweig_gpu.incremental.buildings.BuildingLayer` — footprint ->
cell-centre masks under the trees.py conventions (cell centres, boundary
inclusion, orientation independence), per-cell DEM grounding of the
absolute-elevation rule, nodata refusal, before-footprint resets, ``fmax``
combination, copy-on-write, window slicing, edit-refusals, walllimit
disclosure, and the adapter-seam hand-off
(:func:`massing_edits_from_deltas` -> ``BuildingLayer``).

Chain suite (GDAL + torch, one tiny 48x48 prepared site): the staged
regeneration chain of
:mod:`solweig_gpu.incremental.regenerate` — scenario site staging with
overlap guards, rasterized Building_DSM / Trees tiles, the real walls and
aspect pass, the real standalone SVF recompute, the rebuilt scenario site
cache, and ``run_full_tile`` — plus the two mission invariants: the
baseline stays byte-identical through the whole chain, and a failing stage
leaves it byte-identical too.
"""

from __future__ import annotations

import hashlib
import math
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from osgeo import gdal, osr

from solweig_gpu.incremental import (
    BUILDING_ADAPTER_ID,
    BUILDING_SOURCE_NODE,
    BuildingLayer,
    BuildingMassingDelta,
    BuildingSpec,
    MassingEdit,
    ObjectStateChange,
    RasterGrid,
    RasterWindow,
    RegenerationError,
    TreeLayer,
    TreeSpec,
    massing_edits_from_deltas,
    regenerate_building_batch,
    stage_scenario_site,
    write_scenario_building_dsm,
    write_scenario_trees,
)
from solweig_gpu.incremental.buildings import WALL_LIMIT_M, polygon_cell_mask
from solweig_gpu.incremental.trees import rasterize_tree_patch

gdal.UseExceptions()

DATE_STR = "2024-06-20"

# ---------------------------------------------------------------------------
# Small pure-numpy geometry fixtures
# ---------------------------------------------------------------------------

GRID = RasterGrid(10, 12, 2.0, origin_x_m=1000.0, origin_y_m=2000.0)


def _rect(x0: float, x1: float, y0: float, y1: float):
    """World-axis rectangle covering cells whose centres lie in [x0,x1]x[y0,y1]."""
    return ((x0, y1), (x1, y1), (x1, y0), (x0, y0))


#: cols 2..5, rows 3..5 on GRID (cell centres 1005..1011 x 1987..1993).
BLOCK_FOOT = _rect(1004.0, 1012.0, 1988.0, 1994.0)
BLOCK_CELLS = np.zeros((10, 12), dtype=bool)
BLOCK_CELLS[3:6, 2:6] = True


def _spec(building_id: str, footprint, height: float) -> BuildingSpec:
    return BuildingSpec(building_id=building_id, footprint_m=footprint, height_m=height)


def _add(building_id: str, footprint, height: float) -> MassingEdit:
    return MassingEdit(building_id=building_id, before=None, after=_spec(building_id, footprint, height))


def _flat_dem(value: float = 5.0) -> np.ndarray:
    return np.full((10, 12), value, dtype=np.float32)


class TestRasterizationGeometry:
    def test_rectangle_paints_exact_cells_at_absolute_elevation(self):
        dem = _flat_dem(5.0)
        base = dem.copy()
        layer = BuildingLayer(base, dem, GRID)
        layer.apply_edit(_add("b1", BLOCK_FOOT, 15.0))
        raster = layer.rasterize()
        assert raster.dtype == np.float32
        assert np.array_equal(raster == 20.0, BLOCK_CELLS)
        assert np.array_equal(raster[~BLOCK_CELLS], base[~BLOCK_CELLS])

    def test_per_cell_dem_grounding_follows_slope(self):
        # Absolute elevation = per-cell DEM ground + height_m, exactly the
        # per-cell (nearest, no interpolation) convention the vegetation
        # path uses (vegdem = trees + dem; rasterize_tree_patch never
        # interpolates the DEM).
        dem = np.linspace(250.0, 261.2, 120, dtype=np.float32).reshape(10, 12)
        layer = BuildingLayer(dem.copy(), dem, GRID)
        layer.apply_edit(_add("b1", BLOCK_FOOT, 12.5))
        raster = layer.rasterize()
        assert np.allclose(raster[BLOCK_CELLS], dem[BLOCK_CELLS] + 12.5)
        # the block top follows the slope, not a single base elevation
        assert raster[BLOCK_CELLS].max() - raster[BLOCK_CELLS].min() > 1.0

    def test_winding_and_ring_closure_do_not_matter(self):
        dem = _flat_dem()
        forward = BuildingLayer(dem.copy(), dem, GRID)
        forward.apply_edit(_add("b1", BLOCK_FOOT, 10.0))
        reversed_ring = tuple(reversed(BLOCK_FOOT))
        backward = BuildingLayer(dem.copy(), dem, GRID)
        backward.apply_edit(_add("b1", reversed_ring, 10.0))
        closed = BuildingLayer(dem.copy(), dem, GRID)
        closed.apply_edit(_add("b1", (*BLOCK_FOOT, BLOCK_FOOT[0]), 10.0))
        assert np.array_equal(forward.rasterize(), backward.rasterize())
        assert np.array_equal(forward.rasterize(), closed.rasterize())

    def test_polygon_fill_matches_trees_disc_convention(self):
        # A 64-gon inscribed in a disc must reproduce the tree layer's
        # cell-centre disc mask cell for cell: same centre expression,
        # same boundary-inclusive rule, same grid orientation.
        centre_x, centre_y, radius = 1010.0, 1990.0, 4.7
        ngon = tuple(
            (
                centre_x + radius * math.cos(2 * math.pi * i / 64.0),
                centre_y + radius * math.sin(2 * math.pi * i / 64.0),
            )
            for i in range(64)
        )
        mask, _ = polygon_cell_mask(ngon, GRID, GRID.full_window)
        canopy, _trunk = rasterize_tree_patch(
            TreeSpec("t", centre_x, centre_y, 9.0, radius), GRID, GRID.full_window
        )
        assert np.array_equal(mask, canopy > 0)

    def test_edge_through_cell_centre_is_inside(self):
        # Boundary-inclusive like the disc test's ``<= radius**2``: an edge
        # running exactly along a column of cell centres keeps that column.
        dem = _flat_dem()
        east_edge_on_centres = 1011.0  # centre x of column 5
        foot = _rect(1004.0, east_edge_on_centres, 1988.0, 1994.0)
        layer = BuildingLayer(dem.copy(), dem, GRID)
        layer.apply_edit(_add("b1", foot, 9.0))
        raster = layer.rasterize()
        assert np.all(raster[3:6, 5] == 14.0)  # the on-edge column is painted
        assert np.all(raster[3:6, 6] == 5.0)  # strictly outside stays ground

    def test_off_site_footprint_is_a_disclosed_noop(self):
        dem = _flat_dem()
        layer = BuildingLayer(dem.copy(), dem, GRID)
        far_away = _rect(5000.0, 5012.0, 4988.0, 4994.0)
        layer.apply_edit(_add("b1", far_away, 10.0))
        result = layer.rasterize_with_records()
        assert np.array_equal(result.raster, dem)
        record = result.records[0]
        assert record.cells_painted == 0
        assert any("no cell centres" in note for note in result.disclosures)

    def test_zero_area_polygon_paints_nothing(self):
        dem = _flat_dem()
        layer = BuildingLayer(dem.copy(), dem, GRID)
        sliver = ((1004.0, 1992.0), (1008.0, 1990.0), (1012.0, 1988.0))
        layer.apply_edit(_add("b1", sliver, 10.0))
        result = layer.rasterize_with_records()
        assert result.records[0].cells_painted == 0
        assert np.array_equal(result.raster, dem)

    def test_nonfinite_dem_under_footprint_refused(self):
        dem = _flat_dem()
        dem[4, 4] = np.nan
        layer = BuildingLayer(dem.copy(), dem, GRID)
        layer.apply_edit(_add("b1", BLOCK_FOOT, 10.0))
        with pytest.raises(ValueError, match="b1.*non-finite DEM ground"):
            layer.rasterize()

    def test_block_covers_nan_base_cells_without_losing_itself(self):
        # fmax combination: a NaN nodata cell inside the footprint is
        # legitimately covered by the block; NaN elsewhere survives.
        dem = _flat_dem()
        base = _flat_dem()
        base[4, 3] = np.nan
        base[8, 8] = np.nan
        layer = BuildingLayer(base, dem, GRID)
        layer.apply_edit(_add("b1", BLOCK_FOOT, 10.0))
        raster = layer.rasterize()
        assert raster[4, 3] == 15.0
        assert np.isnan(raster[8, 8])

    def test_edge_touching_footprint_flagged(self):
        dem = _flat_dem()
        layer = BuildingLayer(dem.copy(), dem, GRID)
        touching = _rect(1000.0, 1012.0, 1988.0, 1994.0)  # includes col 0
        layer.apply_edit(_add("b1", touching, 10.0))
        result = layer.rasterize_with_records()
        assert result.records[0].edge_touching is True
        assert any("raster edge" in note for note in result.disclosures)
        interior = BuildingLayer(dem.copy(), dem, GRID)
        interior.apply_edit(_add("b1", BLOCK_FOOT, 10.0))
        assert interior.rasterize_with_records().records[0].edge_touching is False


class TestLayerSemantics:
    def test_delete_resets_before_footprint_to_ground(self):
        dem = _flat_dem(5.0)
        base = dem + np.where(BLOCK_CELLS, 15.0, 0.0).astype(np.float32)
        layer = BuildingLayer(base, dem, GRID)
        layer.apply_edit(
            MassingEdit("b1", before=_spec("b1", BLOCK_FOOT, 15.0), after=None)
        )
        result = layer.rasterize_with_records()
        record = result.records[0]
        assert record.operation == "delete"
        assert record.cells_reset == int(BLOCK_CELLS.sum())
        assert np.array_equal(result.raster, dem)  # fully restored ground

    def test_update_resets_then_paints(self):
        dem = _flat_dem(5.0)
        base = dem + np.where(BLOCK_CELLS, 15.0, 0.0).astype(np.float32)
        smaller = _rect(1004.0, 1008.0, 1988.0, 1994.0)  # cols 2..3 only
        smaller_cells = np.zeros_like(BLOCK_CELLS)
        smaller_cells[3:6, 2:4] = True
        layer = BuildingLayer(base, dem, GRID)
        layer.apply_edit(
            MassingEdit(
                "b1",
                before=_spec("b1", BLOCK_FOOT, 15.0),
                after=_spec("b1", smaller, 20.0),
            )
        )
        result = layer.rasterize_with_records()
        record = result.records[0]
        assert record.operation == "update"
        assert record.cells_reset == int(BLOCK_CELLS.sum())
        assert record.cells_painted == int(smaller_cells.sum())
        assert np.array_equal(result.raster == 25.0, smaller_cells)
        # the abandoned part of the old footprint is ground again
        assert np.array_equal(result.raster[3:6, 4:6], dem[3:6, 4:6])

    def test_paint_never_lowers_a_taller_baseline(self):
        dem = _flat_dem(5.0)
        base = dem + np.where(BLOCK_CELLS, 30.0, 0.0).astype(np.float32)
        layer = BuildingLayer(base, dem, GRID)
        layer.apply_edit(_add("b1", BLOCK_FOOT, 10.0))
        assert np.array_equal(layer.rasterize(), base)  # max, never a carve-down

    def test_copy_on_write_invariants(self):
        dem = _flat_dem(5.0)
        base = dem + 1.0
        base_snapshot = base.copy()
        dem_snapshot = dem.copy()
        layer = BuildingLayer(base, dem, GRID)
        assert not layer.base_building_dsm.flags.writeable
        assert not layer.dem.flags.writeable
        layer.apply_edit(_add("b1", BLOCK_FOOT, 40.0))
        first = layer.rasterize()
        second = layer.rasterize()
        assert np.array_equal(first, second)  # rasterization is pure
        assert np.array_equal(base, base_snapshot)
        assert np.array_equal(dem, dem_snapshot)
        base[0, 0] = 99.0  # caller mutation cannot reach the layer's copy
        assert np.array_equal(layer.rasterize(), first)

    def test_window_rasterization_equals_full_slice(self):
        dem = np.linspace(250.0, 261.2, 120, dtype=np.float32).reshape(10, 12)
        base = dem + 3.0
        layer = BuildingLayer(base, dem, GRID)
        layer.apply_edit(_add("b1", BLOCK_FOOT, 15.0))
        layer.apply_edit(
            MassingEdit("b1", _spec("b1", BLOCK_FOOT, 15.0), None)
        )
        layer.apply_edit(_add("b2", _rect(1006.0, 1014.0, 1986.0, 1992.0), 8.0))
        full = layer.rasterize()
        for window in (
            RasterWindow(0, 10, 0, 12),
            RasterWindow(2, 7, 3, 9),
            RasterWindow(0, 4, 0, 4),
        ):
            assert np.array_equal(
                layer.rasterize_window(window),
                full[window.row_start : window.row_stop, window.col_start : window.col_stop],
            )

    def test_record_operations_and_counts(self):
        dem = _flat_dem(5.0)
        layer = BuildingLayer(dem.copy(), dem, GRID)
        layer.apply_edit(_add("b1", BLOCK_FOOT, 12.0))
        layer.apply_edit(
            MassingEdit("b1", _spec("b1", BLOCK_FOOT, 12.0), _spec("b1", BLOCK_FOOT, 18.0))
        )
        layer.apply_edit(
            MassingEdit("b2", _spec("b2", _rect(1006.0, 1010.0, 1990.0, 1994.0), 9.0), None)
        )
        records = layer.rasterize_with_records().records
        assert [r.operation for r in records] == ["add", "update", "delete"]
        assert [r.building_id for r in records] == ["b1", "b1", "b2"]
        assert records[0].cells_painted == int(BLOCK_CELLS.sum())
        assert records[1].cells_reset == int(BLOCK_CELLS.sum())
        assert [r.walls_expected for r in records] == [True, True, False]

    def test_walllimit_disclosure(self):
        dem = _flat_dem(5.0)
        layer = BuildingLayer(dem.copy(), dem, GRID)
        layer.apply_edit(_add("short", BLOCK_FOOT, WALL_LIMIT_M - 0.5))
        layer.apply_edit(_add("tall", _rect(1006.0, 1010.0, 1990.0, 1994.0), 12.0))
        result = layer.rasterize_with_records()
        by_id = {r.building_id: r for r in result.records}
        assert by_id["short"].walls_expected is False
        assert by_id["tall"].walls_expected is True
        notes = " | ".join(result.disclosures)
        assert "wall" in notes and "3" in notes and "short" in notes
        assert "tall" not in notes  # no spurious disclosure for the tall block

    def test_refusals_leave_layer_state_unchanged(self):
        dem = _flat_dem(5.0)
        layer = BuildingLayer(dem.copy(), dem, GRID)
        layer.apply_edit(_add("b1", BLOCK_FOOT, 12.0))
        with pytest.raises(ValueError, match="already exists"):
            layer.apply_edit(_add("b1", BLOCK_FOOT, 20.0))
        with pytest.raises(ValueError, match="stale before spec"):
            layer.apply_edit(
                MassingEdit("b1", _spec("b1", BLOCK_FOOT, 20.0), _spec("b1", BLOCK_FOOT, 25.0))
            )
        with pytest.raises(ValueError, match="no-op"):
            layer.apply_edit(
                MassingEdit("b1", _spec("b1", BLOCK_FOOT, 12.0), _spec("b1", BLOCK_FOOT, 12.0))
            )
        with pytest.raises(ValueError, match="MassingEdit"):
            layer.apply_edit("not an edit")
        assert len(layer.edits) == 1
        assert [s.building_id for s in layer.current_buildings()] == ["b1"]

    def test_sequential_edits_of_one_block_apply_in_order(self):
        dem = _flat_dem(5.0)
        layer = BuildingLayer(dem.copy(), dem, GRID)
        layer.apply_edit(_add("b1", BLOCK_FOOT, 12.0))
        moved = _rect(1015.0, 1019.0, 1986.0, 1992.0)  # cols 7..9, rows 4..6
        layer.apply_edit(
            MassingEdit("b1", _spec("b1", BLOCK_FOOT, 12.0), _spec("b1", moved, 12.0))
        )
        raster = layer.rasterize()
        assert np.all(raster[BLOCK_CELLS] == 5.0)  # old footprint is ground
        moved_cells = np.zeros_like(BLOCK_CELLS)
        moved_cells[4:7, 7:10] = True
        assert np.array_equal(raster == 17.0, moved_cells)

    def test_adapter_seam_handoff_feeds_the_layer(self):
        # The u-b-bld contract: committed BuildingMassingDelta records fold
        # into re-validated MassingEdit pairs, which rasterize directly.
        delta = BuildingMassingDelta(
            source_node_id=BUILDING_SOURCE_NODE,
            adapter_id=BUILDING_ADAPTER_ID,
            objects=(
                ObjectStateChange(
                    object_id="b1",
                    before=None,
                    after={
                        "footprint_m": [list(v) for v in BLOCK_FOOT],
                        "height_m": 14.0,
                    },
                ),
            ),
            windows=(),
        )
        edits = massing_edits_from_deltas([delta])
        assert len(edits) == 1
        dem = _flat_dem(5.0)
        layer = BuildingLayer(dem.copy(), dem, GRID)
        layer.apply_edits(edits)
        assert np.array_equal(layer.rasterize() == 19.0, BLOCK_CELLS)


# ---------------------------------------------------------------------------
# Tiny prepared-site fixture for the regeneration chain
# ---------------------------------------------------------------------------

TINY_ROWS = TINY_COLS = 48
TINY_PIXEL = 2.0
TINY_ORIGIN = (1000.0, 2000.0)
TINY_EPSG = 32616
SVF_ZIP_MEMBERS = (
    "svf", "svfE", "svfS", "svfW", "svfN",
    "svfveg", "svfEveg", "svfSveg", "svfWveg", "svfNveg",
    "svfaveg", "svfEaveg", "svfSaveg", "svfWaveg", "svfNaveg",
)

#: Baseline 12 m block: rows 8..15, cols 6..13.
SITE_BLOCK_FOOT = _rect(1013.0, 1027.0, 1969.0, 1983.0)
#: Scenario addition: a new 15 m block at rows 28..35, cols 26..33.
NEW_BLOCK_FOOT = _rect(1053.0, 1067.0, 1929.0, 1943.0)
SITE_TREE = TreeSpec(
    "base-tree", TINY_ORIGIN[0] + 10.5 * TINY_PIXEL,
    TINY_ORIGIN[1] - 40.5 * TINY_PIXEL, 6.0, 4.0,
)


def _write_tif(path: Path, array: np.ndarray, *, origin=TINY_ORIGIN, pixel: float = TINY_PIXEL, epsg: int = TINY_EPSG):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows, cols = array.shape
    dtype = gdal.GDT_Byte if array.dtype == np.uint8 else gdal.GDT_Float32
    dataset = gdal.GetDriverByName("GTiff").Create(str(path), cols, rows, 1, dtype)
    dataset.SetGeoTransform((origin[0], pixel, 0.0, origin[1], 0.0, -pixel))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    dataset.SetProjection(srs.ExportToWkt())
    dataset.GetRasterBand(1).WriteArray(array)
    dataset.FlushCache()
    dataset = None
    return path


def _read_tif_array(path: Path) -> np.ndarray:
    dataset = gdal.Open(str(path))
    try:
        return dataset.GetRasterBand(1).ReadAsArray()
    finally:
        dataset = None


def _write_met(path: Path, hours) -> Path:
    def row(hour: int) -> list[float]:
        daytime = 6 <= hour <= 19
        ramp = math.sin(math.pi * (hour - 6) / 13.0) if daytime else -0.3
        radg = 850.0 * max(0.0, ramp) if daytime else 0.0
        ta = 22.0 + 8.0 * ramp
        rh = 60.0 - 20.0 * ramp
        values = (
            [2024.0, 172.0, float(hour), 0.0]
            + [0.0] * 5
            + [2.0, rh, ta, 1013.0, 0.0, radg]
            + [0.0] * 6
            + [150.0, 600.0, 270.0, 0.0]
        )
        assert len(values) == 25
        return values

    header = "# year doy hour minute placeholder wind rh ta p radg rad diff radI wdir uhii"
    lines = [header] + [" ".join(f"{v:g}" for v in row(h)) for h in hours]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def _derive_lon_lat(building_tif: Path) -> tuple[float, float]:
    dataset = gdal.Open(str(building_tif))
    gt = dataset.GetGeoTransform()
    widthx, heightx = dataset.RasterXSize, dataset.RasterYSize
    old_cs = osr.SpatialReference()
    old_cs.ImportFromWkt(dataset.GetProjection())
    old_cs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    new_cs = osr.SpatialReference()
    new_cs.ImportFromEPSG(4326)
    new_cs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    transform = osr.CoordinateTransformation(old_cs, new_cs)
    lon, lat = transform.TransformPoint(
        gt[0] + gt[1] * widthx / 2.0, gt[3] + gt[5] * heightx / 2.0
    )[:2]
    dataset = None
    return lon, lat


def _derive_utc_offset(lat: float, lon: float) -> float:
    import datetime

    from pytz import timezone as pytz_timezone
    from timezonefinder import TimezoneFinder

    name = TimezoneFinder().timezone_at(lat=lat, lng=lon) or "UTC"
    local = pytz_timezone(name).localize(
        datetime.datetime.strptime(DATE_STR, "%Y-%m-%d")
    )
    return local.utcoffset().total_seconds() / 3600


def _hash_tree(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path(root).rglob("*"))
        if path.is_file()
    }


@pytest.fixture(scope="module")
def baseline(tmp_path_factory) -> SimpleNamespace:
    """One tiny prepared site + baseline cache (synthetic baseline SVF)."""
    from solweig_gpu.incremental.cache_builder import build_site_cache
    from solweig_gpu.incremental.cache import SiteCache

    root = tmp_path_factory.mktemp("uc5_baseline")
    site = root / "processed_inputs"
    grid = RasterGrid(TINY_ROWS, TINY_COLS, TINY_PIXEL, *TINY_ORIGIN)

    dem = np.zeros((TINY_ROWS, TINY_COLS), dtype=np.float32)
    building = dem.copy()
    building[8:16, 6:14] = 12.0
    landcover = np.ones((TINY_ROWS, TINY_COLS), dtype=np.uint8)
    vegetation = np.zeros((TINY_ROWS, TINY_COLS), dtype=np.float32)
    canopy, _trunk = rasterize_tree_patch(SITE_TREE, grid, grid.full_window)
    np.maximum(vegetation, canopy, out=vegetation)

    _write_tif(site / "Building_DSM" / "Building_DSM_0_0.tif", building)
    _write_tif(site / "DEM" / "DEM_0_0.tif", dem)
    _write_tif(site / "Trees" / "Trees_0_0.tif", vegetation)
    _write_tif(site / "walls" / "walls_0_0.tif", np.zeros_like(dem))
    _write_tif(site / "aspect" / "aspect_0_0.tif", np.zeros_like(dem))
    _write_tif(site / "Landcover" / "Landcover_0_0.tif", landcover)
    _write_met(site / "metfiles" / f"metfile_0_0_{DATE_STR}.txt", range(10, 13))

    # Synthetic baseline SVF artifacts: nothing in the chain reads baseline
    # SVF values, so cheap random fields stand in for the real calculator.
    rng = np.random.default_rng(20260902)
    svf_dir = site / "SVF"
    svf_dir.mkdir(parents=True)
    scratch = root / "zip_members"
    with zipfile.ZipFile(svf_dir / "svfs_0_0.zip", "w") as archive:
        for member in SVF_ZIP_MEMBERS:
            member_path = _write_tif(
                scratch / f"{member}.tif",
                rng.uniform(0.0, 1.0, size=(TINY_ROWS, TINY_COLS)).astype(np.float32),
            )
            archive.write(member_path, arcname=f"{member}.tif")
    _write_tif(
        svf_dir / "SkyViewFactor_0_0.tif",
        rng.uniform(0.0, 1.0, size=(TINY_ROWS, TINY_COLS)).astype(np.float32),
    )
    np.savez(
        svf_dir / "shadowmats_0_0.npz",
        **{
            name: rng.integers(0, 2, size=(TINY_ROWS, TINY_COLS, 16)).astype(np.float32)
            for name in ("shadowmat", "vegshadowmat", "vbshmat")
        },
    )

    lon, lat = _derive_lon_lat(site / "Building_DSM" / "Building_DSM_0_0.tif")
    cache_dir = root / "cache"
    build_site_cache(
        site,
        cache_dir,
        tile_key="0_0",
        site_id="uc5-baseline",
        latitude=lat,
        longitude=lon,
        altitude_m=0.0,  # flat zero DEM -> the oracle altitude rule
        utc_offset_hours=_derive_utc_offset(lat, lon),
    )
    return SimpleNamespace(
        root=root,
        site=site,
        cache_dir=cache_dir,
        cache=SiteCache.load(cache_dir),
        grid=grid,
        site_hashes=_hash_tree(site),
        cache_hashes=_hash_tree(cache_dir),
    )


def _chain_edits() -> tuple[MassingEdit, ...]:
    """One committed batch: raise the baseline block, add a new block."""
    deltas = (
        BuildingMassingDelta(
            source_node_id=BUILDING_SOURCE_NODE,
            adapter_id=BUILDING_ADAPTER_ID,
            objects=(
                ObjectStateChange(
                    object_id="site-block",
                    before={
                        "footprint_m": [list(v) for v in SITE_BLOCK_FOOT],
                        "height_m": 12.0,
                    },
                    after={
                        "footprint_m": [list(v) for v in SITE_BLOCK_FOOT],
                        "height_m": 18.0,
                    },
                ),
            ),
            windows=(),
        ),
        BuildingMassingDelta(
            source_node_id=BUILDING_SOURCE_NODE,
            adapter_id=BUILDING_ADAPTER_ID,
            objects=(
                ObjectStateChange(
                    object_id="new-block",
                    before=None,
                    after={
                        "footprint_m": [list(v) for v in NEW_BLOCK_FOOT],
                        "height_m": 15.0,
                    },
                ),
            ),
            windows=(),
        ),
    )
    return massing_edits_from_deltas(deltas)


class TestRegenerationStages:
    def test_stage_scenario_site_copies_baseline_and_guards_overlap(self, baseline, tmp_path):
        root = tmp_path / "scenario"
        site = stage_scenario_site(baseline.site, root, tile_key="0_0")
        assert site == root / "site"
        assert _hash_tree(site) == baseline.site_hashes  # faithful copy
        # a rerun wipes and rebuilds
        (site / "stale_marker.txt").write_text("stale")
        stage_scenario_site(baseline.site, root, tile_key="0_0")
        assert not (site / "stale_marker.txt").exists()
        # overlap guards: never able to delete baseline inputs
        with pytest.raises(RegenerationError, match="overlap"):
            stage_scenario_site(baseline.site, baseline.site, tile_key="0_0")
        with pytest.raises(RegenerationError, match="overlap"):
            stage_scenario_site(baseline.site, baseline.root, tile_key="0_0")
        with pytest.raises(RegenerationError, match="missing"):
            stage_scenario_site(baseline.site, tmp_path / "x2", tile_key="9_9")

    def test_write_scenario_building_dsm_and_trees(self, baseline, tmp_path):
        site = stage_scenario_site(baseline.site, tmp_path / "scenario", tile_key="0_0")
        result = write_scenario_building_dsm(site, _chain_edits(), tile_key="0_0")
        by_id = {r.building_id: r for r in result.records}
        assert by_id["site-block"].operation == "update"
        assert by_id["new-block"].operation == "add"
        written = _read_tif_array(site / "Building_DSM" / "Building_DSM_0_0.tif")
        assert np.array_equal(written, result.raster)
        assert (written == 18.0).sum() == 64
        assert (written == 15.0).sum() == 64

        layer = TreeLayer(np.asarray(baseline.cache.tree_base), baseline.grid)
        layer.add_tree(TreeSpec("t9", TINY_ORIGIN[0] + 30.5 * TINY_PIXEL, TINY_ORIGIN[1] - 20.5 * TINY_PIXEL, 7.0, 3.0))
        write_scenario_trees(site, layer, tile_key="0_0")
        trees = _read_tif_array(site / "Trees" / "Trees_0_0.tif")
        expected, _ = rasterize_tree_patch(layer.current_trees()[0], baseline.grid, baseline.grid.full_window)
        base_canopy, _ = rasterize_tree_patch(SITE_TREE, baseline.grid, baseline.grid.full_window)
        assert np.array_equal(trees, np.maximum(base_canopy, expected))
        # the baseline tiles were never touched
        assert _hash_tree(baseline.site) == baseline.site_hashes

    def test_empty_batch_refused(self, baseline, tmp_path):
        with pytest.raises(RegenerationError, match="at least one MassingEdit"):
            regenerate_building_batch(
                edits=(),
                baseline_site_dir=baseline.site,
                baseline_cache=baseline.cache,
                scenario_root=tmp_path / "scenario",
            )

    def test_full_chain_smoke_and_baseline_immutability(self, baseline, tmp_path):
        result = regenerate_building_batch(
            edits=_chain_edits(),
            baseline_site_dir=baseline.site,
            baseline_cache=baseline.cache,
            scenario_root=tmp_path / "scenario",
        )
        # every stage ran, in order
        assert result.stages == (
            "stage_scenario_site",
            "write_scenario_building_dsm",
            "regenerate_walls_aspect",
            "regenerate_svf",
            "rebuild_scenario_cache",
            "run_scenario_full_tile",
        )

        site = result.scenario_site_dir
        # walls + aspect regenerated from the scenario Building_DSM
        walls = _read_tif_array(site / "walls" / "walls_0_0.tif")
        aspect = _read_tif_array(site / "aspect" / "aspect_0_0.tif")
        assert walls.shape == (TINY_ROWS, TINY_COLS)
        assert walls.max() >= WALL_LIMIT_M  # the 18 m and 15 m blocks wall up
        assert np.any(walls > 0) and np.any(aspect > 0)

        # SVF artifacts genuinely rewritten (not the copied baseline files)
        for name in ("SkyViewFactor_0_0.tif", "svfs_0_0.zip", "shadowmats_0_0.npz"):
            scenario_hash = hashlib.sha256((site / "SVF" / name).read_bytes()).hexdigest()
            baseline_hash = hashlib.sha256(
                (baseline.site / "SVF" / name).read_bytes()
            ).hexdigest()
            assert scenario_hash != baseline_hash, name
        svftotal = _read_tif_array(site / "SVF" / "SkyViewFactor_0_0.tif")
        assert svftotal.min() >= 0.0 and svftotal.max() <= 1.0

        # scenario cache rebuilt over the regenerated products
        assert result.cache.manifest.site_id == "uc5-baseline-scenario"
        assert np.array_equal(np.asarray(result.cache.building_dsm), result.raster)
        assert (
            result.cache.manifest.rasters["building_dsm"].sha256
            != baseline.cache.manifest.rasters["building_dsm"].sha256
        )
        assert (
            result.cache.manifest.rasters["walls"].sha256
            != baseline.cache.manifest.rasters["walls"].sha256
        )
        assert result.cache.time_steps == 3

        # the full tile completed: real outputs with the right shape
        assert set(result.outputs) == {"utci", "tmrt", "shadow"}
        for name, stack in result.outputs.items():
            assert stack.shape == (3, TINY_ROWS, TINY_COLS), name
            assert np.isfinite(stack).any(), name

        # mission invariant: the baseline is byte-identical end to end
        assert _hash_tree(baseline.site) == baseline.site_hashes
        assert _hash_tree(baseline.cache_dir) == baseline.cache_hashes

    def test_failing_stage_leaves_baseline_pristine(self, baseline, tmp_path, monkeypatch):
        import solweig_gpu.solweig_gpu as pipeline

        def boom(*args, **kwargs):
            raise RuntimeError("SVF exploded")

        monkeypatch.setattr(pipeline, "calculate_svf", boom)
        with pytest.raises(RegenerationError, match="regenerate_svf") as info:
            regenerate_building_batch(
                edits=_chain_edits(),
                baseline_site_dir=baseline.site,
                baseline_cache=baseline.cache,
                scenario_root=tmp_path / "scenario",
            )
        # the error names how far the chain got; the staged partial tree
        # remains for inspection and the baseline is untouched.
        assert "regenerate_walls_aspect" in str(info.value)
        scenario_site = tmp_path / "scenario" / "site"
        assert (scenario_site / "Building_DSM" / "Building_DSM_0_0.tif").is_file()
        assert _hash_tree(baseline.site) == baseline.site_hashes
        assert _hash_tree(baseline.cache_dir) == baseline.cache_hashes
