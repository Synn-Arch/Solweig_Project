# SPDX-License-Identifier: GPL-3.0-only
"""Phase 5 tests: exact CPU window worker (worker / solver / result).

Fast suite (unmarked) covers the result-patch contract, halo geometry, the
stale-SVF guard, forcing validation, and the worker job lifecycle (no-op,
watermark, supersession, fallback routing) with the solvers stubbed — no
SOLWEIG physics runs unmarked.

The ``scientific`` suite builds one synthetic prepared site (192x192, 4 m
pixels) with a real ``svf_calculator`` sky-view field and runs the full
differential matrix against the standard full-domain oracle:

* SCI-001 (exactness): add/move/resize/delete patches are bit-identical to
  the oracle inside the write window;
* SCI-002 (outside-window invariance): the stored result never changes
  outside the write window, and the edited-scene oracle itself is unchanged
  there (influence windows are conservative);
* SCI-003 (seam diagnostics): boundary-ring and distance-band metrics are
  evaluated with tests/scientific/metrics.py;
* SCI-004 (safe fallback): an oversized local job routes to the full-tile
  path and still matches the oracle bit-for-bit.
"""

from __future__ import annotations

import datetime
import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from osgeo import gdal, osr

from solweig_gpu.incremental.cache import SiteCache
from solweig_gpu.incremental.cache_builder import build_site_cache
from solweig_gpu.incremental.geometry import (
    InfluenceConfig,
    RasterGrid,
    RasterWindow,
    TreeSpec,
    LOCAL_OFFSET_UNSTABLE,
    local_elevation_offset_m,
)
from solweig_gpu.incremental.result import (
    PATCH_METADATA_FILE,
    PatchChecksumError,
    PatchError,
    ResultPatch,
    discard_staging,
    load_patch,
    publish_staged_patch,
    stage_patch,
)
from solweig_gpu.incremental import solver as solver_mod
from solweig_gpu.incremental.solver import (
    GVF_MARCH_METERS,
    FullSceneTensors,
    SiteForcing,
    SolverInputError,
    StaleSvfError,
    compose_full_scene_tensors,
    gvf_march_pixels,
    load_site_forcing,
    read_window_for_write_window,
    required_halo_pixels,
    window_svf_bundle,
    _march_reach_pixels,
)
from solweig_gpu.incremental.trees import TreeLayer
from solweig_gpu.incremental.worker import ExactWorker
from tests.scientific.metrics import (
    assert_outside_window_unchanged,
    compare_binary,
    compare_continuous,
    distance_band_masks,
    error_by_distance_band,
    outside_window_mask,
)
from tests.scientific.tolerances import (
    DISTANCE_BANDS,
    SHADOW_MISMATCH_FRACTION_MAX,
    TMRT_MAX_ABS,
    UTCI_MAX_ABS,
)

gdal.UseExceptions()

DATE_STR = "2024-06-20"
VARIABLES = ("utci", "tmrt", "shadow")

SVF_ZIP_MEMBERS = (
    "svf", "svfE", "svfS", "svfW", "svfN",
    "svfveg", "svfEveg", "svfSveg", "svfWveg", "svfNveg",
    "svfaveg", "svfEaveg", "svfSaveg", "svfWaveg", "svfNaveg",
)

# window_svf_bundle positions (svf_calculator return order)
SVF_BUNDLE_INDEX = {
    "svf": 0, "svfaveg": 1, "svfE": 2, "svfEaveg": 3, "svfEveg": 4,
    "svfN": 5, "svfNaveg": 6, "svfNveg": 7, "svfS": 8, "svfSaveg": 9,
    "svfSveg": 10, "svfveg": 11, "svfW": 12, "svfWaveg": 13, "svfWveg": 14,
    "vegshmat": 15, "vbshvegshmat": 16, "shmat": 17, "svftotal": 18,
}


# ---------------------------------------------------------------------------
# Synthetic prepared-site construction
# ---------------------------------------------------------------------------


def _write_tif(path: Path, array: np.ndarray, *, origin, pixel: float, epsg: int):
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


def _met_row(hour: int) -> list[float]:
    daytime = 6 <= hour <= 19
    ramp = math.sin(math.pi * (hour - 6) / 13.0) if daytime else -0.3
    radg = 850.0 * max(0.0, ramp) if daytime else 0.0
    ta = 22.0 + 8.0 * ramp
    rh = 60.0 - 20.0 * ramp
    row = (
        [2024.0, 172.0, float(hour), 0.0]
        + [0.0] * 5
        + [2.0, rh, ta, 1013.0, 0.0, radg]
        + [0.0] * 6
        + [150.0, 600.0, 270.0, 0.0]
    )
    assert len(row) == 25
    return row


def _write_met(path: Path, hours) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "# year doy hour minute placeholder wind rh ta p radg rad diff radI wdir uhii"
    lines = [header] + [" ".join(f"{v:g}" for v in _met_row(h)) for h in hours]
    path.write_text("\n".join(lines) + "\n")
    return path


def _derive_lon_lat(building_tif: Path) -> tuple[float, float]:
    """Replicate compute_utci's tile-centre location derivation exactly."""
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


def _derive_utc_offset(lat: float, lon: float, date_str: str) -> float:
    """Replicate compute_utci's TimezoneFinder/pytz UTC-offset derivation."""
    from timezonefinder import TimezoneFinder
    import pytz

    name = TimezoneFinder().timezone_at(lat=lat, lng=lon) or "UTC"
    local_dt = pytz.timezone(name).localize(
        datetime.datetime.strptime(date_str, "%Y-%m-%d")
    )
    return local_dt.utcoffset().total_seconds() / 3600


def _scene_arrays(rows: int, cols: int):
    """Deterministic built scene: flat ground and two building blocks.

    The north-west block is deliberately the tallest structure (12 m, above
    every tree the tests add): the cached building-only SVF terms are only
    scene-invariant while an edit cannot raise the site-wide ray-march
    amplitude (see solver.solve_window's baseline-amaxvalue guard).
    """
    dem = np.zeros((rows, cols), dtype=np.float32)
    building = dem.copy()
    building[int(rows * 0.08) : int(rows * 0.20), int(cols * 0.06) : int(cols * 0.20)] = 12.0
    building[int(rows * 0.62) : int(rows * 0.75), int(cols * 0.60) : int(cols * 0.75)] = 4.0
    landcover = np.ones((rows, cols), dtype=np.uint8)
    landcover[int(rows * 0.3) : int(rows * 0.4), int(cols * 0.3) : int(cols * 0.4)] = 5
    landcover[int(rows * 0.85) : int(rows * 0.92), int(cols * 0.05) : int(cols * 0.15)] = 7
    landcover[int(rows * 0.85) : int(rows * 0.92), int(cols * 0.8) : int(cols * 0.9)] = 6
    return dem, building, landcover


def _make_prepared_site(
    root: Path,
    *,
    rows: int,
    cols: int,
    pixel: float,
    origin,
    epsg: int,
    base_trees: tuple[TreeSpec, ...],
    met_hours,
) -> tuple[RasterGrid, Path]:
    """Write a prepared ``processed_inputs`` site with the given base trees."""
    from solweig_gpu.incremental.trees import rasterize_tree_patch

    site = root / "processed_inputs"
    grid = RasterGrid(rows, cols, pixel, origin[0], origin[1])
    dem, building, landcover = _scene_arrays(rows, cols)
    vegetation = np.zeros((rows, cols), dtype=np.float32)
    for tree in base_trees:
        canopy, _trunk = rasterize_tree_patch(tree, grid, grid.full_window)
        np.maximum(vegetation, canopy, out=vegetation)

    def tif(kind: str, array: np.ndarray) -> Path:
        return _write_tif(
            site / kind / f"{kind}_0_0.tif", array,
            origin=origin, pixel=pixel, epsg=epsg,
        )

    tif("Building_DSM", building)
    tif("DEM", dem)
    tif("Trees", vegetation)
    tif("walls", np.zeros((rows, cols), dtype=np.float32))
    tif("aspect", np.zeros((rows, cols), dtype=np.float32))
    tif("Landcover", landcover)
    _write_met(site / "metfiles" / f"metfile_0_0_{DATE_STR}.txt", met_hours)
    return grid, site


def _compute_baseline_svf(site: Path) -> None:
    """Run the real ``svf_calculator`` on the baseline scene (standalone layout).

    This is the same computation ``compute_utci`` performs on a cold SVF path
    (fresh scratch directory, no SVF cache), so a cache built from these
    outputs carries the oracle's building-only and baseline vegetation SVF.
    """
    from solweig_gpu.shadow import load_raster_to_tensor, svf_calculator

    building = site / "Building_DSM" / "Building_DSM_0_0.tif"
    a, dataset = load_raster_to_tensor(str(building))
    temp1, _ = load_raster_to_tensor(str(site / "Trees" / "Trees_0_0.tif"))
    temp2, _ = load_raster_to_tensor(str(site / "DEM" / "DEM_0_0.tif"))
    scale = 1.0 / dataset.GetGeoTransform()[1]

    temp1[temp1 < 0.0] = 0.0
    vegdem = temp1 + temp2
    vegdem2 = torch.add(temp1 * 0.25, temp2)
    bush = torch.logical_not(vegdem2 * vegdem) * vegdem
    vegdsm = temp1 + a
    vegdsm[vegdsm == a] = 0.0
    vegdsm2 = temp1 * 0.25 + a
    vegdsm2[vegdsm2 == a] = 0.0
    amaxvalue = torch.maximum(a.max(), vegdem.max())
    svf_dir = site / "SVF"
    svf_dir.mkdir(parents=True, exist_ok=True)
    svf_calculator(
        2, amaxvalue, a, vegdsm, vegdsm2, bush, scale,
        save_rasters=True, output_dir=str(svf_dir), number="0_0",
        gdal_dsm=dataset,
    )
    dataset = None


def _build_cache(site_dir: Path, cache_dir: Path, *, met_path: Path, site_id: str) -> SiteCache:
    building = site_dir / "Building_DSM" / "Building_DSM_0_0.tif"
    lon, lat = _derive_lon_lat(building)
    utc = _derive_utc_offset(lat, lon, DATE_STR)
    build_site_cache(
        site_dir,
        cache_dir,
        tile_key="0_0",
        site_id=site_id,
        latitude=lat,
        longitude=lon,
        altitude_m=0.0,  # flat zero DEM -> the oracle altitude rule yields 0.0
        utc_offset_hours=utc,
        met_file=met_path,
    )
    return SiteCache.load(cache_dir)


# ---------------------------------------------------------------------------
# Fast fixture: tiny site with arbitrary (random) SVF numbers — no physics
# ---------------------------------------------------------------------------

TINY_ROWS = TINY_COLS = 128
TINY_PIXEL = 2.0
TINY_ORIGIN = (1000.0, 2000.0)
TINY_EPSG = 32616
TINY_TREE = TreeSpec(
    "b1", TINY_ORIGIN[0] + 10.5 * TINY_PIXEL,
    TINY_ORIGIN[1] - 10.5 * TINY_PIXEL, 3.0, 2.0,
)
# Off-centre position so the dirty + write-margin window stays a strict
# sub-window of the tiny grid (otherwise every job routes to FULL).
TINY_ADD = TreeSpec(
    "t1", TINY_ORIGIN[0] + 40.5 * TINY_PIXEL,
    TINY_ORIGIN[1] - 40.5 * TINY_PIXEL, 4.0, 2.0,
)
# Forces the LOCAL branch in lifecycle tests (window fraction stays below 1).
LOCAL_ALWAYS = InfluenceConfig(full_recompute_fraction=1.0)


def _make_tiny_site(root: Path) -> tuple[RasterGrid, Path]:
    grid, site = _make_prepared_site(
        root,
        rows=TINY_ROWS,
        cols=TINY_COLS,
        pixel=TINY_PIXEL,
        origin=TINY_ORIGIN,
        epsg=TINY_EPSG,
        base_trees=(TINY_TREE,),
        met_hours=range(10, 13),
    )
    rng = np.random.default_rng(20260901)
    svf_dir = site / "SVF"
    svf_dir.mkdir(parents=True, exist_ok=True)
    scratch = root / "zip_members"
    with zipfile.ZipFile(svf_dir / "svfs_0_0.zip", "w") as archive:
        for member in SVF_ZIP_MEMBERS:
            array = rng.uniform(0.0, 1.0, size=(TINY_ROWS, TINY_COLS)).astype(np.float32)
            member_path = _write_tif(scratch / f"{member}.tif", array, origin=TINY_ORIGIN, pixel=TINY_PIXEL, epsg=TINY_EPSG)
            archive.write(member_path, arcname=f"{member}.tif")
    _write_tif(
        svf_dir / "SkyViewFactor_0_0.tif",
        rng.uniform(0.0, 1.0, size=(TINY_ROWS, TINY_COLS)).astype(np.float32),
        origin=TINY_ORIGIN, pixel=TINY_PIXEL, epsg=TINY_EPSG,
    )
    cubes = {
        name: rng.integers(0, 2, size=(TINY_ROWS, TINY_COLS, 16)).astype(np.float32)
        for name in ("shadowmat", "vegshadowmat", "vbshmat")
    }
    np.savez(svf_dir / "shadowmats_0_0.npz", **cubes)
    return grid, site


@pytest.fixture()
def tiny(tmp_path: Path):
    grid, site = _make_tiny_site(tmp_path)
    cache = _build_cache(
        site, tmp_path / "cache",
        met_path=site / "metfiles" / f"metfile_0_0_{DATE_STR}.txt",
        site_id="tiny",
    )
    return SimpleNamespace(grid=grid, site=site, cache=cache)


def _flat_scene(a_tensor: torch.Tensor, amax: float) -> FullSceneTensors:
    zeros = torch.zeros_like(a_tensor)
    return FullSceneTensors(
        a=a_tensor, canopy=zeros.clone(), dem=zeros.clone(),
        vegdem=zeros.clone(), vegdem2=zeros.clone(), bush=zeros.clone(),
        vegdsm=zeros.clone(), vegdsm2=zeros.clone(),
        amaxvalue=torch.tensor(amax),
    )


def _forcing(altitudes) -> SiteForcing:
    altitudes = list(altitudes)
    t_steps = len(altitudes)
    return SiteForcing(
        met_table=np.zeros((t_steps, 25)),
        altitude=np.asarray([altitudes], dtype=np.float64),
        azimuth=np.full((1, t_steps), 180.0),
        zen=np.full((1, t_steps), 0.8),
        jday=np.full((1, t_steps), 172.0),
        dectime=np.asarray([172.0 + 0.25 * t for t in range(t_steps)]),
        altmax=np.full((1, t_steps), 80.0),
        location={"longitude": -87.0, "latitude": 33.0, "altitude": 0.0},
        utc_offset_hours=-5.0,
        selected_date_str=DATE_STR,
        met_path=Path("met"),
    )


# ---------------------------------------------------------------------------
# ResultPatch contract
# ---------------------------------------------------------------------------


def _sample_patch(write: RasterWindow, *, mode: str = "local") -> ResultPatch:
    arrays = {
        name: np.full((4, write.height, write.width), 12.5, dtype=np.float32)
        for name in VARIABLES
    }
    return ResultPatch(
        job_id="job-test",
        scene_revision=3,
        mode=mode,
        write_window=write,
        read_window=write.expand(4).clamp(rows=64, cols=64),
        site_id="site",
        tile_key="0_0",
        cache_manifest_sha256="0" * 64,
        model_version="test",
        variables=VARIABLES,
        arrays=arrays,
        time_start=0,
        time_stop=4,
    )


class TestResultPatchContract:
    def test_stage_publish_roundtrip(self, tmp_path: Path) -> None:
        patch = _sample_patch(RasterWindow(8, 16, 4, 12))
        staging = stage_patch(patch, tmp_path)
        assert staging.name.endswith(".tmp")
        published = publish_staged_patch(staging, tmp_path, patch.scene_revision)
        loaded = load_patch(published)
        assert loaded.job_id == patch.job_id
        assert loaded.scene_revision == 3
        assert loaded.mode == "local"
        assert loaded.write_window == patch.write_window
        assert loaded.variables == VARIABLES
        for name in VARIABLES:
            np.testing.assert_array_equal(loaded.variable(name), patch.arrays[name])
            assert loaded.variable(name).dtype == np.float32

    def test_checksum_verification_rejects_tampering(self, tmp_path: Path) -> None:
        patch = _sample_patch(RasterWindow(0, 4, 0, 4))
        published = publish_staged_patch(
            stage_patch(patch, tmp_path), tmp_path, patch.scene_revision
        )
        target = published / "variables" / "tmrt.f32.npy"
        data = bytearray(target.read_bytes())
        data[-1] ^= 0xFF
        target.write_bytes(bytes(data))
        with pytest.raises(PatchChecksumError, match="tmrt"):
            load_patch(published)

    def test_missing_checksum_entry_rejected(self, tmp_path: Path) -> None:
        # Metadata corruption that DROPS a checksum must not silently skip
        # verification for that variable.
        patch = _sample_patch(RasterWindow(0, 4, 0, 4))
        published = publish_staged_patch(
            stage_patch(patch, tmp_path), tmp_path, patch.scene_revision
        )
        metadata = json.loads((published / PATCH_METADATA_FILE).read_text())
        del metadata["checksums"]["tmrt"]
        (published / PATCH_METADATA_FILE).write_text(json.dumps(metadata))
        with pytest.raises(PatchError, match="no checksum for variable 'tmrt'"):
            load_patch(published)

    def test_unknown_schema_version_rejected(self, tmp_path: Path) -> None:
        patch = _sample_patch(RasterWindow(0, 4, 0, 4))
        published = publish_staged_patch(
            stage_patch(patch, tmp_path), tmp_path, patch.scene_revision
        )
        metadata = json.loads((published / PATCH_METADATA_FILE).read_text())
        metadata["schema_version"] = 99
        (published / PATCH_METADATA_FILE).write_text(json.dumps(metadata))
        with pytest.raises(PatchError, match="schema version"):
            load_patch(published)

    def test_apply_into_writes_only_inside_window(self) -> None:
        patch = _sample_patch(RasterWindow(4, 8, 4, 8))
        target = {name: np.zeros((4, 32, 32), dtype=np.float32) for name in VARIABLES}
        before = {name: arr.copy() for name, arr in target.items()}
        patch.apply_into(target)
        w = patch.write_window
        inside = np.zeros((32, 32), dtype=bool)
        inside[w.row_start:w.row_stop, w.col_start:w.col_stop] = True
        for name in VARIABLES:
            np.testing.assert_array_equal(
                target[name][:, w.row_start:w.row_stop, w.col_start:w.col_stop],
                patch.arrays[name],
            )
            np.testing.assert_array_equal(target[name][:, ~inside], before[name][:, ~inside])

    def test_invalid_patch_modes_and_shapes_rejected(self) -> None:
        with pytest.raises(PatchError, match="mode"):
            _sample_patch(RasterWindow(0, 4, 0, 4), mode="approximate")
        patch = _sample_patch(RasterWindow(0, 4, 0, 4))
        bad_arrays = dict(patch.arrays)
        bad_arrays["utci"] = bad_arrays["utci"][:, :, :3]
        with pytest.raises(PatchError, match="shape"):
            ResultPatch(
                job_id=patch.job_id, scene_revision=patch.scene_revision,
                mode="local", write_window=patch.write_window,
                read_window=patch.read_window, site_id="site", tile_key="0_0",
                cache_manifest_sha256="0" * 64, model_version="test",
                variables=VARIABLES, arrays=bad_arrays,
                time_start=0, time_stop=4,
            )

    def test_discard_staging_removes_temp_dirs(self, tmp_path: Path) -> None:
        patch = _sample_patch(RasterWindow(0, 4, 0, 4))
        staging = stage_patch(patch, tmp_path)
        discard_staging(tmp_path)
        assert not staging.exists()


# ---------------------------------------------------------------------------
# Halo geometry
# ---------------------------------------------------------------------------


class TestHaloGeometry:
    def test_march_reach_formula(self) -> None:
        # amplitude 10 m, scale 0.5 px/m, 45 degrees
        expected = math.ceil(10.0 * 0.5 / math.tan(math.radians(45.0)))
        assert _march_reach_pixels(
            height_amplitude_m=10.0, scale=0.5, altitude_deg=45.0
        ) == expected
        # a night sun needs no march
        assert _march_reach_pixels(height_amplitude_m=10.0, scale=0.5, altitude_deg=-3.0) == 0
        # near-horizontal rays march unboundedly far
        assert _march_reach_pixels(height_amplitude_m=10.0, scale=0.5, altitude_deg=0.5) > 500

    def test_required_halo_covers_patches_gvf_and_margin(self, tiny) -> None:
        a = torch.zeros((TINY_ROWS, TINY_COLS))
        a[8:12, 8:12] = 5.0  # 5 m buildings on flat ground: amplitude 5
        scene = _flat_scene(a, 5.0)
        forcing = _forcing([6.0, 45.0])  # daytime altitudes in the series
        halo = required_halo_pixels(tiny.cache, scene, forcing)
        scale = 1.0 / TINY_PIXEL
        reach_6deg = math.ceil(5.0 * scale / math.tan(math.radians(6.0)))
        assert halo == min(
            reach_6deg + gvf_march_pixels(TINY_PIXEL) + 2, TINY_ROWS
        )

    def test_gvf_march_pixels_scales_with_pixel_size(self) -> None:
        # The oracle marches round(22 m / pixel_size_m) CELLS; the halo term
        # must scale the same way (ceil, never below the oracle's reach).
        # 22 cells is correct ONLY at 1 m pixels.
        assert gvf_march_pixels(1.0) == 22
        assert gvf_march_pixels(0.5) == 44  # sub-metre site: old constant truncated
        assert gvf_march_pixels(2.0) == 11
        assert gvf_march_pixels(4.0) == 6
        assert GVF_MARCH_METERS == 22.0
        import math as _math
        import pytest as _pytest

        with _pytest.raises(ValueError):
            gvf_march_pixels(0.0)
        with _pytest.raises(ValueError):
            gvf_march_pixels(float("nan"))
        assert gvf_march_pixels(3.0) == _math.ceil(22.0 / 3.0)

    def test_read_window_expands_and_clamps(self, tiny) -> None:
        a = torch.zeros((TINY_ROWS, TINY_COLS))
        a[8:12, 8:12] = 5.0
        scene = _flat_scene(a, 5.0)
        forcing = _forcing([45.0])
        write = RasterWindow(10, 20, 10, 20)
        read = read_window_for_write_window(write, tiny.cache, scene, forcing)
        halo = required_halo_pixels(tiny.cache, scene, forcing)
        # exact semantics: targets (write + GVF ring) plus any occluder that
        # can flip a march for them, always inside the conservative halo
        targets = write.expand(
            gvf_march_pixels(TINY_PIXEL) + 2
        ).clamp(rows=TINY_ROWS, cols=TINY_COLS)
        assert read == targets  # flat surroundings flip nothing beyond targets
        conservative = write.expand(halo).clamp(rows=TINY_ROWS, cols=TINY_COLS)
        assert (
            read.row_start >= conservative.row_start
            and read.row_stop <= conservative.row_stop
            and read.col_start >= conservative.col_start
            and read.col_stop <= conservative.col_stop
        )
        # the GVF ring clamps the lower/left edges to the grid while the
        # upper/right edges expand freely
        stop = min(20 + gvf_march_pixels(TINY_PIXEL) + 2, TINY_ROWS)
        assert read == RasterWindow(0, stop, 0, stop)

    def test_read_window_includes_only_reachable_occluders(self, tiny) -> None:
        # a 5 m occluder 8 px beyond the target ring still reaches it at the
        # 6-degree sky-patch altitude and must be read; flat cells between
        # the ring and the occluder cannot flip anything and are skipped.
        a = torch.zeros((TINY_ROWS, TINY_COLS))
        a[8:12, 8:12] = 5.0
        a[40:42, 40:42] = 5.0
        scene = _flat_scene(a, 5.0)
        forcing = _forcing([45.0])
        write = RasterWindow(10, 20, 10, 20)
        read = read_window_for_write_window(write, tiny.cache, scene, forcing)
        ring = gvf_march_pixels(TINY_PIXEL) + 2
        assert read == RasterWindow(0, 42, 0, 42)  # bbox through the occluder

    def test_read_window_drops_occluders_below_flip_threshold(self, tiny) -> None:
        # same geometry but only 0.1 m tall: 0.1 m cannot beat
        # 8 px * 2 m * tan(6 deg) ~ 1.68 m, so the mask stays empty.
        a = torch.zeros((TINY_ROWS, TINY_COLS))
        a[8:12, 8:12] = 5.0
        a[40:42, 40:42] = 0.1
        scene = _flat_scene(a, 5.0)
        forcing = _forcing([45.0])
        write = RasterWindow(10, 20, 10, 20)
        read = read_window_for_write_window(write, tiny.cache, scene, forcing)
        targets = write.expand(
            gvf_march_pixels(TINY_PIXEL) + 2
        ).clamp(rows=TINY_ROWS, cols=TINY_COLS)
        assert read == targets

    def test_read_window_covers_site_when_write_covers_site(self, tiny) -> None:
        a = torch.zeros((TINY_ROWS, TINY_COLS))
        scene = _flat_scene(a, 0.0)
        forcing = _forcing([45.0])
        write = RasterWindow(0, TINY_ROWS, 0, TINY_COLS)
        read = read_window_for_write_window(write, tiny.cache, scene, forcing)
        assert read == RasterWindow(0, TINY_ROWS, 0, TINY_COLS)


# ---------------------------------------------------------------------------
# Per-tree local elevation offsets (P8: re-enable local mode)
# ---------------------------------------------------------------------------


class TestLocalElevationOffset:
    def _sloped_surface(self, rows: int = 40, cols: int = 40) -> np.ndarray:
        """Terrain with a 100 m plateau (Chebyshev ring <= 2) dropping
        1 m per pixel to ring 14, then rising: the deepest reachable
        surface is 86 m at ring 14, so the stable local relief is 14 m."""
        row_index = np.arange(rows)[:, None]
        col_index = np.arange(cols)[None, :]
        cheb = np.maximum(
            np.abs(row_index - rows // 2), np.abs(col_index - cols // 2)
        ).astype(float)
        surface = 100.0 - cheb
        surface[cheb <= 2.0] = 100.0
        surface[cheb >= 15.0] = 101.0  # rises: no deeper relief in reach
        return surface

    def _bounds_for(self, surface: np.ndarray, centre: int):
        def bounds_for_offset(offset_m: float) -> RasterWindow:
            radius_px = 5 + int(offset_m)  # region grows with the offset
            return RasterWindow(
                max(0, centre - radius_px),
                min(surface.shape[0], centre + radius_px + 1),
                max(0, centre - radius_px),
                min(surface.shape[1], centre + radius_px + 1),
            )

        return bounds_for_offset

    @staticmethod
    def _region_min(surface: np.ndarray):
        def region_minimum(window: RasterWindow) -> float:
            return float(
                surface[
                    window.row_start : window.row_stop,
                    window.col_start : window.col_stop,
                ].min()
            )

        return region_minimum

    def test_flat_region_returns_zero(self) -> None:
        surface = np.full((20, 20), 100.0)
        offset = local_elevation_offset_m(
            base_elevation_m=100.0,
            bounds_for_offset=lambda _o: RasterWindow(0, 20, 0, 20),
            region_minimum_m=self._region_min(surface),
        )
        assert offset == 0.0

    def test_basin_offset_stabilises_at_local_relief(self) -> None:
        surface = self._sloped_surface()
        offset = local_elevation_offset_m(
            base_elevation_m=100.0,
            bounds_for_offset=self._bounds_for(surface, 20),
            region_minimum_m=self._region_min(surface),
            max_iterations=6,
        )
        # the slope bottoms at 86 m (ring 14) and rises beyond: the stable
        # local relief is exactly 14 m, reached once the region spans the
        # basin rim; each iteration must expand before that.
        assert offset == 14.0

    def test_monotone_slope_hits_iteration_cap_and_reports_unstable(self) -> None:
        # 100x100 terrain falling 1 m/px to the edges: every expansion
        # reveals a lower cell, so the candidate grows each iteration and
        # cannot stabilise inside the cap.
        surface = self._sloped_surface(100, 100)
        # remove the rim so the drop continues to the site edge
        row_index = np.arange(100)[:, None]
        col_index = np.arange(100)[None, :]
        cheb = np.maximum(np.abs(row_index - 50), np.abs(col_index - 50))
        surface = 100.0 - cheb

        offset = local_elevation_offset_m(
            base_elevation_m=100.0,
            bounds_for_offset=self._bounds_for(surface, 50),
            region_minimum_m=self._region_min(surface),
            max_iterations=3,
        )
        assert offset is LOCAL_OFFSET_UNSTABLE

    def test_iteration_cap_smaller_than_basin_expansion_is_unstable(self) -> None:
        # same basin as the stabilising test, but only one iteration: the
        # first region sees relief, the candidate would grow on the second
        # pass — the cap must refuse to guess and return the sentinel so
        # callers fall back to a conservative bound.
        surface = self._sloped_surface()
        offset = local_elevation_offset_m(
            base_elevation_m=100.0,
            bounds_for_offset=self._bounds_for(surface, 20),
            region_minimum_m=self._region_min(surface),
            max_iterations=1,
        )
        assert offset is LOCAL_OFFSET_UNSTABLE

    def test_offset_never_smaller_than_final_region_relief(self) -> None:
        surface = self._sloped_surface()
        bounds_for_offset = self._bounds_for(surface, 20)
        offset = local_elevation_offset_m(
            base_elevation_m=100.0,
            bounds_for_offset=bounds_for_offset,
            region_minimum_m=self._region_min(surface),
            max_iterations=6,
        )
        final_window = bounds_for_offset(offset)
        final_relief = 100.0 - float(
            surface[
                final_window.row_start : final_window.row_stop,
                final_window.col_start : final_window.col_stop,
            ].min()
        )
        assert offset >= final_relief

    def test_flat_site_worker_offset_is_zero(self, tiny) -> None:
        from solweig_gpu.incremental.worker import ExactWorker as _W

        worker = _W(
            tiny.cache,
            TreeLayer(tiny.cache.tree_base, tiny.grid),
            site_dir=tiny.site,
            results_root=tiny.site.parent / "results",
            selected_date_str=DATE_STR,
        )
        config = worker._site_influence_config()
        tree = TreeSpec(
            "t",
            tiny.grid.origin_x_m + 60.5 * TINY_PIXEL,
            tiny.grid.origin_y_m - 60.5 * TINY_PIXEL,
            3.0,
            1.0,
        )
        assert worker._influence_elevation_offset(tree, config) == 0.0

    def test_unstable_search_falls_back_to_global_relief(self, tiny) -> None:
        from unittest.mock import patch

        from solweig_gpu.incremental import worker as worker_mod
        from solweig_gpu.incremental.worker import ExactWorker as _W

        worker = _W(
            tiny.cache,
            TreeLayer(tiny.cache.tree_base, tiny.grid),
            site_dir=tiny.site,
            results_root=tiny.site.parent / "results",
            selected_date_str=DATE_STR,
        )
        config = worker._site_influence_config()
        a = np.asarray(tiny.cache.building_dsm)
        # a tree on the 12 m block: global drop = 12 - 0
        tree = TreeSpec(
            "t",
            tiny.grid.origin_x_m + 12.5 * TINY_PIXEL,
            tiny.grid.origin_y_m - 12.5 * TINY_PIXEL,
            3.0,
            1.0,
        )
        with patch.object(
            worker_mod, "local_elevation_offset_m", return_value=None
        ):
            offset = worker._influence_elevation_offset(tree, config)
        assert offset == float(a[12, 12]) - float(a.min())

    def test_pedestal_tree_routes_full_but_open_tree_routes_local(
        self, tiny
    ) -> None:
        from solweig_gpu.incremental.geometry import choose_recompute_mode
        from solweig_gpu.incremental.worker import ExactWorker as _W

        def windows_for(x_px: float, y_px: float) -> list[RasterWindow]:
            layer = TreeLayer(tiny.cache.tree_base, tiny.grid)
            layer.add_tree(
                TreeSpec(
                    "t",
                    tiny.grid.origin_x_m + x_px * TINY_PIXEL,
                    tiny.grid.origin_y_m - y_px * TINY_PIXEL,
                    3.0,
                    1.0,
                )
            )
            worker = _W(
                tiny.cache,
                layer,
                site_dir=tiny.site,
                results_root=tiny.site.parent / "results",
                selected_date_str=DATE_STR,
            )
            return worker.dirty_windows()

        # open ground: strict sub-window, well under the 0.30 fraction
        open_windows = windows_for(70.5, 30.5)
        assert open_windows
        fraction = sum(w.area for w in open_windows) / (
            tiny.grid.rows * tiny.grid.cols
        )
        assert fraction < 0.30
        assert choose_recompute_mode(open_windows[0], tiny.grid).value == "local"
        # same tree on the 12 m building block: the pedestal offset makes the
        # honest influence window whole-hillside -> full recompute by policy
        pedestal_windows = windows_for(12.5, 12.5)
        fraction_p = sum(w.area for w in pedestal_windows) / (
            tiny.grid.rows * tiny.grid.cols
        )
        assert fraction_p >= 0.30
        assert (
            choose_recompute_mode(pedestal_windows[0], tiny.grid).value
            == "full"
        )

    def test_exact_influence_window_matches_flip_condition(self, tiny) -> None:
        from solweig_gpu.incremental.worker import ExactWorker as _W

        layer = TreeLayer(tiny.cache.tree_base, tiny.grid)
        tree = TreeSpec(
            "t",
            tiny.grid.origin_x_m + 40.5 * TINY_PIXEL,
            tiny.grid.origin_y_m - 40.5 * TINY_PIXEL,
            3.0,
            1.0,
        )
        layer.add_tree(tree)
        worker = _W(
            tiny.cache,
            layer,
            site_dir=tiny.site,
            results_root=tiny.site.parent / "results",
            selected_date_str=DATE_STR,
        )
        config = worker._site_influence_config()
        exact = worker._exact_influence_window([tree], config)
        assert exact is not None
        # flat open ground, 3 m tree, 2 m pixels. tan_min is tan of
        # min(6-deg sky-patch floor, series-min positive solar altitude);
        # here that is the 6 deg patch floor, so flips stay within
        # ceil(3 / (2 * tan(6 deg))) = 15 px of cell (40, 40) (centre
        # floor(40.5)), aligned to the 16 px block grid.
        assert exact == RasterWindow(16, 64, 16, 64)


# ---------------------------------------------------------------------------
# Stale-SVF guard (Stage 7)
# ---------------------------------------------------------------------------


class TestStaleSvfGuard:
    def test_unchanged_scene_slices_cache(self, tiny) -> None:
        layer = TreeLayer(tiny.cache.tree_base, tiny.grid)
        scene = compose_full_scene_tensors(tiny.cache, layer)
        window = RasterWindow(4, 20, 4, 20)
        bundle = window_svf_bundle(tiny.cache, scene, window, veg_changed=False)
        for name in ("svf", "svfveg", "svfEaveg"):
            np.testing.assert_array_equal(
                bundle[SVF_BUNDLE_INDEX[name]].numpy(),
                np.asarray(tiny.cache.svf[name])[4:20, 4:20],
                err_msg=name,
            )
        np.testing.assert_array_equal(
            bundle[SVF_BUNDLE_INDEX["shmat"]].numpy(),
            np.asarray(tiny.cache.svf_patches["shadowmat"])[4:20, 4:20, :],
        )

    def test_negative_nodata_treebase_no_edit_scene_slices_cache(self, tiny, monkeypatch) -> None:
        """A baseline Trees raster carrying negative nodata (e.g. -999 in
        LiDAR products) is stored verbatim in the cache; the no-edit guard
        compares clamped-vs-clamped, so nodata alone must never trip it."""
        from solweig_gpu.incremental.cache import SiteCache as _SiteCache

        tree_base = np.array(tiny.cache.tree_base)
        tree_base[0, 0] = -999.0
        monkeypatch.setattr(
            _SiteCache, "tree_base", property(lambda self: tree_base)
        )
        layer = TreeLayer(tree_base, tiny.grid)
        scene = compose_full_scene_tensors(tiny.cache, layer)
        bundle = window_svf_bundle(tiny.cache, scene, RasterWindow(4, 20, 4, 20), veg_changed=False)
        np.testing.assert_array_equal(
            bundle[SVF_BUNDLE_INDEX["svf"]].numpy(),
            np.asarray(tiny.cache.svf["svf"])[4:20, 4:20],
        )

    def test_veg_edit_rejects_cached_terms(self, tiny) -> None:
        layer = TreeLayer(tiny.cache.tree_base, tiny.grid)
        layer.add_tree(TreeSpec(
            "t9", TINY_ORIGIN[0] + 20.5 * TINY_PIXEL,
            TINY_ORIGIN[1] - 20.5 * TINY_PIXEL, 4.0, 2.0,
        ))
        scene = compose_full_scene_tensors(tiny.cache, layer)
        with pytest.raises(StaleSvfError, match="vegetation"):
            window_svf_bundle(tiny.cache, scene, RasterWindow(4, 20, 4, 20), veg_changed=False)

    def test_veg_edit_recomputes_terms(self, tiny) -> None:
        layer = TreeLayer(tiny.cache.tree_base, tiny.grid)
        layer.add_tree(TreeSpec(
            "t9", TINY_ORIGIN[0] + 20.5 * TINY_PIXEL,
            TINY_ORIGIN[1] - 20.5 * TINY_PIXEL, 4.0, 2.0,
        ))
        scene = compose_full_scene_tensors(tiny.cache, layer)
        bundle = window_svf_bundle(tiny.cache, scene, RasterWindow(4, 20, 4, 20), veg_changed=True)
        svfveg = bundle[SVF_BUNDLE_INDEX["svfveg"]]
        assert svfveg.shape == (16, 16)
        # the recomputed vegetation SVF differs from the stale cache somewhere
        stale = np.asarray(tiny.cache.svf["svfveg"])[4:20, 4:20]
        assert not np.array_equal(svfveg.numpy(), stale)


class TestAmplitudeGuard:
    def test_taller_than_baseline_rejected_for_local_solve(self, tiny) -> None:
        """A tree taller than every baseline structure raises the site-wide
        ray-march amplitude, so cached building SVF terms would be stale; the
        solver must refuse instead of producing a silently-wrong window."""
        from solweig_gpu.incremental.solver import solve_window

        layer = TreeLayer(tiny.cache.tree_base, tiny.grid)
        layer.add_tree(TreeSpec(
            "tall", TINY_ORIGIN[0] + 40.5 * TINY_PIXEL,
            TINY_ORIGIN[1] - 40.5 * TINY_PIXEL, 50.0, 3.0,
        ))
        with pytest.raises(SolverInputError, match="amaxvalue"):
            solve_window(
                tiny.cache,
                layer,
                read_window=tiny.grid.full_window,
                write_window=RasterWindow(4, 20, 4, 20),
                forcing=_forcing([45.0]),
                requested_variables=("utci",),
            )

    def test_amplitude_lowering_on_negative_dsm_rejected(self, tiny) -> None:
        """Deleting the dominant tree lowers the oracle's march amplitude; on
        a site with below-datum surfaces the extra cached march steps can
        still contribute, so the guard must refuse there too."""
        from solweig_gpu.incremental.solver import _check_baseline_amplitude

        scene = compose_full_scene_tensors(tiny.cache, TreeLayer(tiny.cache.tree_base, tiny.grid))
        # Lower the amplitude and push the DSM below datum.
        scene.a[0, 0] = -5.0
        scene.amaxvalue = torch.tensor(11.0)  # below the baseline 12 m block
        with pytest.raises(SolverInputError, match="amaxvalue"):
            _check_baseline_amplitude(scene, tiny.cache)

    def test_amplitude_lowering_on_flat_site_allowed(self, tiny) -> None:
        """With non-negative surfaces the shorter cached march cannot change
        any result (extra steps never pass the occluder test), so a
        lowering edit stays safe locally."""
        from solweig_gpu.incremental.solver import _check_baseline_amplitude

        scene = compose_full_scene_tensors(tiny.cache, TreeLayer(tiny.cache.tree_base, tiny.grid))
        assert float(scene.a.min()) >= 0.0
        scene.amaxvalue = torch.tensor(11.0)  # below the baseline 12 m block
        _check_baseline_amplitude(scene, tiny.cache)  # must not raise

    def test_unchanged_amplitude_allowed(self, tiny) -> None:
        from solweig_gpu.incremental.solver import _check_baseline_amplitude

        scene = compose_full_scene_tensors(tiny.cache, TreeLayer(tiny.cache.tree_base, tiny.grid))
        scene.a[0, 0] = -5.0
        _check_baseline_amplitude(scene, tiny.cache)  # equal amplitude: fine


# ---------------------------------------------------------------------------
# Forcing validation
# ---------------------------------------------------------------------------


class TestForcing:
    def test_load_validates_and_recomputes_float64(self, tiny) -> None:
        forcing = load_site_forcing(tiny.cache, site_dir=tiny.site, selected_date_str=DATE_STR)
        assert forcing.met_table.shape == (3, 25)
        assert forcing.met_table.dtype == np.float64
        assert forcing.altitude.shape == (1, 3)
        assert forcing.location["altitude"] == 0.0

    def test_wrong_date_rejected(self, tiny) -> None:
        with pytest.raises(SolverInputError, match="selected_date_str"):
            load_site_forcing(tiny.cache, site_dir=tiny.site, selected_date_str="2024-06-21")

    def test_tampered_met_rejected(self, tiny) -> None:
        met = tiny.site / "metfiles" / f"metfile_0_0_{DATE_STR}.txt"
        _write_met(met, range(13, 16))  # different rows than the cache saw
        with pytest.raises(SolverInputError, match="meteorology"):
            load_site_forcing(tiny.cache, site_dir=tiny.site, selected_date_str=DATE_STR)

    def _tiny_cache(self, root: Path, **overrides) -> tuple[SiteCache, Path]:
        grid, site = _make_tiny_site(root / "site")
        building = site / "Building_DSM" / "Building_DSM_0_0.tif"
        lon, lat = _derive_lon_lat(building)
        kwargs = dict(
            site_dir=site, cache_dir=root / "cache", tile_key="0_0", site_id="tiny",
            latitude=lat, longitude=lon, altitude_m=0.0,
            utc_offset_hours=_derive_utc_offset(lat, lon, DATE_STR),
            met_file=site / "metfiles" / f"metfile_0_0_{DATE_STR}.txt",
        )
        kwargs.update(overrides)
        build_site_cache(**kwargs)
        return SiteCache.load(root / "cache"), site

    def test_utc_mismatch_rejected(self, tmp_path: Path) -> None:
        cache, site = self._tiny_cache(tmp_path, utc_offset_hours=1.0)
        with pytest.raises(SolverInputError, match="UTC offset"):
            load_site_forcing(cache, site_dir=site, selected_date_str=DATE_STR)

    def test_altitude_rule_mismatch_rejected(self, tmp_path: Path) -> None:
        cache, site = self._tiny_cache(tmp_path, altitude_m=3.0)  # wrong for 0 m DEM
        with pytest.raises(SolverInputError, match="altitude"):
            load_site_forcing(cache, site_dir=site, selected_date_str=DATE_STR)

    def test_manifest_lonlat_mismatch_rejected(self, tmp_path: Path) -> None:
        # A cache whose manifest lon/lat disagrees with its own rasters
        # (operator error at build time, or re-staged rasters with a shifted
        # extent) would silently fork solar geometry between local solves
        # and the full-domain oracle.
        grid, site = _make_tiny_site(tmp_path / "site")
        building = site / "Building_DSM" / "Building_DSM_0_0.tif"
        lon, lat = _derive_lon_lat(building)
        build_site_cache(
            site_dir=site, cache_dir=tmp_path / "cache", tile_key="0_0",
            site_id="tiny",
            latitude=lat, longitude=lon + 0.01,  # shifted vs the raster centre
            altitude_m=0.0,
            utc_offset_hours=_derive_utc_offset(lat, lon, DATE_STR),
            met_file=site / "metfiles" / f"metfile_0_0_{DATE_STR}.txt",
        )
        cache = SiteCache.load(tmp_path / "cache")
        with pytest.raises(SolverInputError, match="lon/lat"):
            load_site_forcing(cache, site_dir=site, selected_date_str=DATE_STR)

    def test_solar_rowcount_mismatch_rejected(self, tiny, monkeypatch) -> None:
        # A cache whose solar series has a different row count than the met
        # table (cache-build corruption) must never SKIP the solar
        # cross-check silently.
        from solweig_gpu.incremental.cache import SiteCache as _SiteCache

        truncated = np.asarray(tiny.cache.solar)[:-1]
        monkeypatch.setattr(
            _SiteCache, "solar", property(lambda self: truncated)
        )
        with pytest.raises(SolverInputError, match="solar series"):
            load_site_forcing(tiny.cache, site_dir=tiny.site, selected_date_str=DATE_STR)


# ---------------------------------------------------------------------------
# Worker job lifecycle (solvers stubbed; no physics)
# ---------------------------------------------------------------------------


def _stub_patch(job_id: str, revision: int, window: RasterWindow, mode: str = "local") -> ResultPatch:
    arrays = {
        name: np.full((3, window.height, window.width), 1.0, dtype=np.float32)
        for name in VARIABLES
    }
    return ResultPatch(
        job_id=job_id, scene_revision=revision, mode=mode,
        write_window=window,
        read_window=window.expand(2).clamp(rows=TINY_ROWS, cols=TINY_COLS),
        site_id="tiny", tile_key="0_0", cache_manifest_sha256="0" * 64,
        model_version="test", variables=VARIABLES, arrays=arrays,
        time_start=0, time_stop=3,
    )


def _stub_solve_local(job_id, revision, forcing, write_windows, **kwargs):
    return [_stub_patch(job_id, revision, write_windows[0])]


class TestWorkerLifecycle:
    @staticmethod
    def _worker(tmp_path: Path, *, config: InfluenceConfig = LOCAL_ALWAYS) -> tuple[ExactWorker, TreeLayer]:
        grid, site = _make_tiny_site(tmp_path / "site")
        cache = _build_cache(
            site, tmp_path / "cache",
            met_path=site / "metfiles" / f"metfile_0_0_{DATE_STR}.txt",
            site_id="tiny",
        )
        layer = TreeLayer(cache.tree_base, grid)
        worker = ExactWorker(
            cache, layer,
            site_dir=site, results_root=tmp_path / "results",
            selected_date_str=DATE_STR,
            influence_config=config,
        )
        worker.bump_scene_revision()
        return worker, layer

    def test_write_margin_derived_from_pixel_size(self, tmp_path: Path) -> None:
        # The oracle marches round(22 m / pixel) cells: 22 cells ONLY at 1 m
        # pixels. The write margin must scale with the site pixel size.
        worker, _layer = self._worker(tmp_path)  # tiny site: TINY_PIXEL = 2 m
        assert worker.write_margin_pixels == math.ceil(22.0 / TINY_PIXEL) + 1

    def test_site_influence_config_accounts_for_relief(self, tmp_path: Path) -> None:
        worker, _layer = self._worker(tmp_path)
        config = worker._site_influence_config()
        # Flat tiny fixture (DEM all zeros): no relief offset.
        assert config.elevation_offset_m == 0.0
        # The length cap is at least the oracle's own march bound at the
        # lowest configured altitudes, never the bare 300 m default.
        bound = max(
            12.0 / math.tan(math.radians(config.minimum_direct_sun_altitude_deg)),
            12.0 / math.tan(math.radians(config.lowest_sky_patch_altitude_deg)),
        )
        assert config.maximum_shadow_length_m >= bound

    def test_no_edits_is_noop(self, tmp_path: Path) -> None:
        worker, layer = self._worker(tmp_path)
        outcome = worker.run()
        assert outcome.status == "no-op"
        assert outcome.mode is None
        assert outcome.patch_paths == ()
        assert not list((tmp_path / "results").rglob("rev-*"))

    def test_add_then_delete_before_publish_is_noop(self, tmp_path: Path) -> None:
        worker, layer = self._worker(tmp_path)
        layer.add_tree(TINY_ADD)
        layer.delete_tree("t1")
        assert worker.run().status == "no-op"

    def test_publish_flow_and_watermark(self, tmp_path: Path) -> None:
        worker, layer = self._worker(tmp_path)
        captured = {}

        def fake_solve_local(job_id, revision, forcing, write_windows, **kwargs):
            captured["windows"] = write_windows
            return [_stub_patch(job_id, revision, write_windows[0])]

        layer.add_tree(TINY_ADD)
        worker._solve_local = fake_solve_local
        outcome = worker.run()
        assert outcome.status == "published"
        assert outcome.mode == "local"
        assert captured["windows"][0].area > 0
        patch = load_patch(outcome.patch_paths[0])
        assert patch.scene_revision == 1
        # the watermark consumed the add: a re-run with no new edits is a no-op
        assert worker.run().status == "no-op"
        assert worker.pending_batch() is None

    def test_delete_after_publish_is_a_real_job(self, tmp_path: Path) -> None:
        worker, layer = self._worker(tmp_path)
        tree = TINY_ADD
        layer.add_tree(tree)
        worker._solve_local = _stub_solve_local
        assert worker.run().status == "published"

        worker.bump_scene_revision()
        layer.delete_tree("t1")
        second = {}

        def fake_solve_local(job_id, revision, forcing, write_windows, **kwargs):
            second["windows"] = write_windows
            return [_stub_patch(job_id, revision, write_windows[0])]

        worker._solve_local = fake_solve_local
        outcome = worker.run()
        # without the edit watermark this would coalesce to a no-op and the
        # published add patch would stay in the store forever
        assert outcome.status == "published"
        window = second["windows"][0]
        col = int((tree.x_m - TINY_ORIGIN[0]) / TINY_PIXEL)
        row = int((TINY_ORIGIN[1] - tree.y_m) / TINY_PIXEL)
        assert window.row_start <= row < window.row_stop
        assert window.col_start <= col < window.col_stop
        assert len(list((tmp_path / "results" / layer.scenario_id).glob("rev-*"))) == 2

    def test_superseded_job_publishes_nothing(self, tmp_path: Path) -> None:
        worker, layer = self._worker(tmp_path)
        layer.add_tree(TINY_ADD)
        calls = {"n": 0}

        def provider():
            calls["n"] += 1
            return 1 if calls["n"] == 1 else 2  # the scene advances mid-flight

        worker._revision_provider = provider
        worker._solve_local = _stub_solve_local
        outcome = worker.run(target_revision=1)
        assert outcome.status == "superseded"
        scenario_root = tmp_path / "results" / layer.scenario_id
        assert not list(scenario_root.rglob("rev-*"))
        assert not list((scenario_root / ".staging").glob("*.tmp"))

    def test_already_superseded_aborts_immediately(self, tmp_path: Path) -> None:
        worker, layer = self._worker(tmp_path)
        layer.add_tree(TINY_ADD)
        worker._revision_provider = lambda: 99
        outcome = worker.run(target_revision=1)
        assert outcome.status == "superseded"
        assert worker.pending_batch() is not None  # the edit remains pending

    def test_unsafe_local_routes_to_full(self, tmp_path: Path, monkeypatch) -> None:
        worker, layer = self._worker(tmp_path)
        layer.add_tree(TINY_ADD)

        def broken_solve_window(*args, **kwargs):
            raise SolverInputError("halo insufficient")

        monkeypatch.setattr("solweig_gpu.incremental.worker.solve_window", broken_solve_window)
        worker._solve_full = lambda job_id, revision, forcing: [
            _stub_patch(job_id, revision, RasterWindow(0, TINY_ROWS, 0, TINY_COLS), mode="full")
        ]
        outcome = worker.run()
        assert outcome.status == "published"
        assert outcome.mode == "full"
        assert outcome.fallback_reason == "halo insufficient"
        assert outcome.write_windows[0] == RasterWindow(0, TINY_ROWS, 0, TINY_COLS)


# ---------------------------------------------------------------------------
# Scientific differential suite
# ---------------------------------------------------------------------------

SCI_ROWS = SCI_COLS = 192
SCI_PIXEL = 4.0
SCI_ORIGIN = (500000.0, 3750000.0)
SCI_EPSG = 32616

BASE_TREES = (
    TreeSpec("b1", SCI_ORIGIN[0] + 30.5 * SCI_PIXEL, SCI_ORIGIN[1] - 30.5 * SCI_PIXEL, 3.0, 2.0),
    TreeSpec("b2", SCI_ORIGIN[0] + 160.5 * SCI_PIXEL, SCI_ORIGIN[1] - 150.5 * SCI_PIXEL, 2.5, 2.0),
)


@dataclass
class ScienceSite:
    site_dir: Path
    grid: RasterGrid
    cache_a: SiteCache  # 24-hour forcing
    cache_b: SiteCache  # 9..14 h forcing (sub-full read windows)
    forcing_a: SiteForcing
    forcing_b: SiteForcing
    add_position_m: tuple[float, float]
    move_position_m: tuple[float, float]
    valid_mask: np.ndarray  # non-building cells

    def layer(self, scenario_id: str = "sci") -> TreeLayer:
        return TreeLayer(self.cache_a.tree_base, self.grid, scenario_id=scenario_id)

    def results_root(self, name: str) -> Path:
        return self.site_dir.parent / f"results_{name}"


@pytest.fixture(scope="module")
def science_site(tmp_path_factory) -> ScienceSite:
    root = tmp_path_factory.mktemp("science_site")
    grid, site = _make_prepared_site(
        root,
        rows=SCI_ROWS,
        cols=SCI_COLS,
        pixel=SCI_PIXEL,
        origin=SCI_ORIGIN,
        epsg=SCI_EPSG,
        base_trees=BASE_TREES,
        met_hours=range(24),
    )
    _compute_baseline_svf(site)
    cache_a = _build_cache(
        site, root / "cache_a",
        met_path=site / "metfiles" / f"metfile_0_0_{DATE_STR}.txt",
        site_id="science",
    )
    met_b = _write_met(
        site / "metfiles" / f"metfile_midday_0_0_{DATE_STR}.txt", range(9, 15)
    )
    cache_b = _build_cache(site, root / "cache_b", met_path=met_b, site_id="science-b")

    from solweig_gpu.incremental.fixtures import find_open_cell

    veg = np.asarray(cache_a.tree_base)
    building = np.asarray(cache_a.building_dsm)
    dem = np.asarray(cache_a.dem)
    centre = find_open_cell(veg, building, dem, grid, clearance_m=48.0)
    target_row, target_col = SCI_ROWS / 2 - 20, SCI_COLS / 2 + 24
    east = find_open_cell(
        veg, building, dem, grid, clearance_m=48.0,
        preference=lambda r, c: 100.0 * ((r - target_row) ** 2 + (c - target_col) ** 2),
    )
    assert east != centre, "move target must differ from the add position"
    valid_mask = (building.astype(np.float64) - dem.astype(np.float64)) < 2.0
    return ScienceSite(
        site_dir=site,
        grid=grid,
        cache_a=cache_a,
        cache_b=cache_b,
        forcing_a=load_site_forcing(cache_a, site_dir=site, selected_date_str=DATE_STR),
        forcing_b=load_site_forcing(cache_b, site_dir=site, selected_date_str=DATE_STR),
        add_position_m=centre,
        move_position_m=east,
        valid_mask=valid_mask,
    )


def _edited_tree(position_m, *, resize: bool) -> TreeSpec:
    x, y = position_m
    if resize:
        return TreeSpec("t1", x, y, 8.0, 6.0)
    return TreeSpec("t1", x, y, 6.0, 5.0)


def _state_layer(site: ScienceSite, state: str, scenario_id: str = "sci") -> TreeLayer:
    """Layer whose live vegetation equals the named scene state."""
    layer = TreeLayer(site.cache_a.tree_base, site.grid, scenario_id=scenario_id)
    if state == "add":
        layer.add_tree(_edited_tree(site.add_position_m, resize=False))
    elif state == "move":
        layer.add_tree(_edited_tree(site.add_position_m, resize=False))
        layer.move_tree("t1", x_m=site.move_position_m[0], y_m=site.move_position_m[1])
    elif state == "resize":
        layer.add_tree(_edited_tree(site.add_position_m, resize=False))
        layer.update_tree("t1", height_m=8.0, canopy_radius_m=6.0)
    elif state in ("baseline", "delete"):
        pass
    else:
        raise ValueError(state)
    return layer


@dataclass
class OracleStore:
    site: ScienceSite
    runs: dict[tuple[str, str], dict[str, np.ndarray]]

    def get(self, state: str, met: str = "a") -> dict[str, np.ndarray]:
        key = (state, met)
        if key not in self.runs:
            cache = self.site.cache_a if met == "a" else self.site.cache_b
            forcing = self.site.forcing_a if met == "a" else self.site.forcing_b
            scratch = self.site.site_dir.parent / f"oracle_{state}_{met}"
            self.runs[key] = solver_mod.run_full_tile(
                cache,
                _state_layer(self.site, state),
                forcing=forcing,
                site_dir=self.site.site_dir,
                scratch_dir=scratch,
                requested_variables=VARIABLES,
            )
        return self.runs[key]


@pytest.fixture(scope="module")
def oracles(science_site) -> OracleStore:
    return OracleStore(site=science_site, runs={})


# -- scientific assertion helpers -------------------------------------------


def _assert_bitwise_inside(
    candidate: np.ndarray, oracle: np.ndarray, write_window: RasterWindow, variable: str
) -> None:
    w = write_window
    cand = candidate[:, w.row_start:w.row_stop, w.col_start:w.col_stop]
    ref = oracle[:, w.row_start:w.row_stop, w.col_start:w.col_stop]
    assert np.array_equal(cand, ref, equal_nan=True), _mismatch_report(variable, cand, ref, w)


def _mismatch_report(variable: str, candidate: np.ndarray, oracle: np.ndarray, w: RasterWindow) -> str:
    same = np.array_equal(candidate, oracle, equal_nan=True)
    diff = np.abs(candidate.astype(np.float64) - oracle.astype(np.float64))
    bad = np.zeros(candidate.shape, dtype=bool)
    bad[~np.isnan(candidate) | ~np.isnan(oracle)] = True
    bad &= ~same
    per_step = [f"t={t}:{int(bad[t].sum())}/{bad[t].size}" for t in range(bad.shape[0])]
    worst = float(np.nanmax(np.where(bad, diff, 0.0))) if bad.any() else 0.0
    return (
        f"{variable}: not bitwise inside window {w}; mismatched cells per step: "
        f"{', '.join(per_step)}; worst |diff| = {worst:.6g}"
    )


def _assert_outside_identical(before: np.ndarray, after: np.ndarray, window: RasterWindow) -> None:
    """SCI-002: exact outside-window invariance, NaN-stable."""
    mask = outside_window_mask(before.shape[1:], window)
    if not mask.any():
        return
    left = before.transpose(1, 2, 0)[mask]
    right = after.transpose(1, 2, 0)[mask]
    assert np.array_equal(left, right, equal_nan=True), (
        "stored result changed outside the write window"
    )
    # cross-check the shared T3 helper on the first NaN-free plane
    for t in range(before.shape[0]):
        if not np.isnan(before[t]).any():
            assert_outside_window_unchanged(before[t], after[t], window)
            break


def _seam_metrics(
    candidate: np.ndarray,
    oracle: np.ndarray,
    valid: np.ndarray,
    window: RasterWindow,
    tolerance: float,
) -> dict:
    """SCI-003: worst inside / ring / band statistics across timesteps."""
    report = {
        "inside_max_abs": 0.0,
        "ring_max_abs": 0.0,
        "bands_max_abs": {
            label: 0.0
            for label, _mask in distance_band_masks(valid.shape, window, bands=DISTANCE_BANDS)
        },
    }
    for t in range(candidate.shape[0]):
        plane_valid = valid & ~np.isnan(candidate[t]) & ~np.isnan(oracle[t])
        if not plane_valid.any():
            continue
        step = compare_continuous(candidate[t], oracle[t], plane_valid, window, tolerance)
        report["inside_max_abs"] = max(report["inside_max_abs"], step.inside.max_abs_error)
        report["ring_max_abs"] = max(report["ring_max_abs"], step.boundary_ring.max_abs_error)
        for label, metrics in error_by_distance_band(
            candidate[t], oracle[t], plane_valid, window, tolerance, bands=DISTANCE_BANDS
        ).items():
            report["bands_max_abs"][label] = max(
                report["bands_max_abs"][label], metrics.max_abs_error
            )
    return report


def _differential(
    site: ScienceSite,
    oracle: dict[str, np.ndarray],
    baseline: dict[str, np.ndarray],
    store: dict[str, np.ndarray],
    write_window: RasterWindow,
    *,
    label: str,
) -> dict:
    """Full SCI-001/002/003 assertion block for one published store state."""
    w = write_window
    outside = outside_window_mask((SCI_ROWS, SCI_COLS), w)
    diagnostics: dict = {
        "label": label,
        "write_window": (w.row_start, w.row_stop, w.col_start, w.col_stop),
    }

    for name in VARIABLES:
        # SCI-001: bitwise identical to the oracle inside the write window
        _assert_bitwise_inside(store[name], oracle[name], w, name)
        # SCI-002: the store never changes outside the write window
        _assert_outside_identical(baseline[name], store[name], w)
        # conservativeness: outside the influence window the edited-scene
        # oracle equals the baseline oracle, so the patch is complete
        assert np.array_equal(
            oracle[name].transpose(1, 2, 0)[outside],
            baseline[name].transpose(1, 2, 0)[outside],
            equal_nan=True,
        ), f"{name}: oracle changed outside the influence window (non-conservative)"

    for name, tolerance in (("utci", UTCI_MAX_ABS), ("tmrt", TMRT_MAX_ABS)):
        valid = (
            site.valid_mask
            & ~np.isnan(store[name]).any(axis=0)
            & ~np.isnan(oracle[name]).any(axis=0)
        )
        metrics = _seam_metrics(store[name], oracle[name], valid, w, tolerance)
        assert metrics["inside_max_abs"] == 0.0, f"{name} inside: {metrics}"
        assert metrics["ring_max_abs"] == 0.0, f"{name} ring: {metrics}"
        assert all(v == 0.0 for v in metrics["bands_max_abs"].values()), (
            f"{name} bands: {metrics}"
        )
        diagnostics[f"{name}_seam"] = metrics

    shadow_mismatch = 0.0
    for t in range(store["shadow"].shape[0]):
        plane_valid = (
            site.valid_mask
            & ~np.isnan(store["shadow"][t])
            & ~np.isnan(oracle["shadow"][t])
        )
        if not plane_valid.any():
            continue
        step = compare_binary(store["shadow"][t], oracle["shadow"][t], plane_valid, w)
        shadow_mismatch = max(shadow_mismatch, step.inside.mismatch_fraction)
    assert shadow_mismatch <= SHADOW_MISMATCH_FRACTION_MAX
    assert shadow_mismatch == 0.0  # the exact worker targets zero disagreement
    diagnostics["shadow_mismatch_fraction"] = shadow_mismatch
    return diagnostics


def _local_worker(
    science_site: ScienceSite, layer: TreeLayer, results_root: Path, *,
    config: InfluenceConfig | None = None,
) -> ExactWorker:
    return ExactWorker(
        science_site.cache_a,
        layer,
        site_dir=science_site.site_dir,
        results_root=results_root,
        selected_date_str=DATE_STR,
        influence_config=config or InfluenceConfig(),
    )


def _apply_patches(baseline: dict[str, np.ndarray], patches) -> dict[str, np.ndarray]:
    store = {name: arr.copy() for name, arr in baseline.items()}
    for patch in patches:
        patch.apply_into(store)
    return store


@pytest.mark.scientific
class TestScientificLocalDifferential:
    def test_add_tree_local_bitwise(self, science_site, oracles) -> None:
        layer = _state_layer(science_site, "add")
        worker = _local_worker(science_site, layer, science_site.results_root("add"))
        worker.bump_scene_revision()
        outcome = worker.run()
        assert outcome.published
        assert outcome.mode == "local"
        patch = load_patch(outcome.patch_paths[0])
        assert patch.write_window.area < science_site.grid.area_pixels

        baseline = oracles.get("baseline", "a")
        store = _apply_patches(baseline, [patch])
        diagnostics = _differential(
            science_site, oracles.get("add", "a"), baseline,
            store, patch.write_window, label="add",
        )
        assert diagnostics["utci_seam"]["ring_max_abs"] == 0.0

    def _two_job_sequence(
        self, science_site, oracles, root_name: str, second_edit, *,
        config: InfluenceConfig | None = None,
    ):
        """add -> publish -> second edit -> publish; returns both patches."""
        layer = _state_layer(science_site, "add")
        worker = _local_worker(
            science_site, layer, science_site.results_root(root_name), config=config
        )
        worker.bump_scene_revision()
        first = worker.run()
        assert first.published and first.mode == "local"
        patch1 = load_patch(first.patch_paths[0])

        worker.bump_scene_revision()
        second_edit(layer)
        second = worker.run()
        assert second.published and second.mode == "local"
        patch2 = load_patch(second.patch_paths[0])
        return patch1, patch2

    def test_move_tree_local_bitwise(self, science_site, oracles) -> None:
        patch1, patch2 = self._two_job_sequence(
            science_site, oracles, "move",
            lambda layer: layer.move_tree(
                "t1",
                x_m=science_site.move_position_m[0],
                y_m=science_site.move_position_m[1],
            ),
            config=InfluenceConfig(full_recompute_fraction=0.8),
        )
        baseline = oracles.get("baseline", "a")
        # job 2 differential: intermediate store (baseline + add patch) vs move oracle
        _differential(
            science_site, oracles.get("move", "a"), _apply_patches(baseline, [patch1]),
            _apply_patches(baseline, [patch1, patch2]), patch2.write_window, label="move",
        )
        # the move job's window covers both the old and the new position
        for position in (science_site.add_position_m, science_site.move_position_m):
            col = int((position[0] - SCI_ORIGIN[0]) / SCI_PIXEL)
            row = int((SCI_ORIGIN[1] - position[1]) / SCI_PIXEL)
            w = patch2.write_window
            assert w.row_start <= row < w.row_stop
            assert w.col_start <= col < w.col_stop

    def test_resize_tree_local_bitwise(self, science_site, oracles) -> None:
        patch1, patch2 = self._two_job_sequence(
            science_site, oracles, "resize",
            lambda layer: layer.update_tree("t1", height_m=8.0, canopy_radius_m=6.0),
            config=InfluenceConfig(full_recompute_fraction=0.8),
        )
        baseline = oracles.get("baseline", "a")
        _differential(
            science_site, oracles.get("resize", "a"), _apply_patches(baseline, [patch1]),
            _apply_patches(baseline, [patch1, patch2]), patch2.write_window, label="resize",
        )

    def test_delete_tree_restores_baseline_bitwise(self, science_site, oracles) -> None:
        patch1, patch2 = self._two_job_sequence(
            science_site, oracles, "delete", lambda layer: layer.delete_tree("t1"),
        )
        baseline = oracles.get("baseline", "a")
        store = _apply_patches(baseline, [patch1, patch2])
        # the add -> delete round trip restores the baseline exactly everywhere
        for name in VARIABLES:
            assert np.array_equal(store[name], baseline[name], equal_nan=True), (
                f"{name}: add+delete round trip did not restore the baseline"
            )
        # the delete job's window covers the removed tree
        tree = _edited_tree(science_site.add_position_m, resize=False)
        col = int((tree.x_m - SCI_ORIGIN[0]) / SCI_PIXEL)
        row = int((SCI_ORIGIN[1] - tree.y_m) / SCI_PIXEL)
        w = patch2.write_window
        assert w.row_start <= row < w.row_stop and w.col_start <= col < w.col_stop

    def test_unsafe_local_falls_back_to_full_tile(self, science_site, oracles) -> None:
        layer = _state_layer(science_site, "add")
        worker = _local_worker(
            science_site, layer, science_site.results_root("full"),
            config=InfluenceConfig(full_recompute_fraction=0.02),
        )
        worker.bump_scene_revision()
        outcome = worker.run()
        assert outcome.published
        assert outcome.mode == "full"
        patch = load_patch(outcome.patch_paths[0])
        assert patch.write_window == science_site.grid.full_window

        baseline = oracles.get("baseline", "a")
        store = _apply_patches(baseline, [patch])
        # SCI-004: the fallback recomputed everything and still matches the
        # oracle bit-for-bit across the entire tile
        for name in VARIABLES:
            assert np.array_equal(store[name], oracles.get("add", "a")[name], equal_nan=True), (
                f"{name}: full-tile fallback differs from the oracle"
            )

    def test_midday_forcing_reads_subfull_window(self, science_site, oracles) -> None:
        layer = TreeLayer(science_site.cache_b.tree_base, science_site.grid, scenario_id="sci_b")
        layer.add_tree(_edited_tree(science_site.add_position_m, resize=False))
        worker = ExactWorker(
            science_site.cache_b, layer,
            site_dir=science_site.site_dir,
            results_root=science_site.results_root("midday"),
            selected_date_str=DATE_STR,
        )
        worker.bump_scene_revision()
        outcome = worker.run()
        assert outcome.published
        assert outcome.mode == "local"
        patch = load_patch(outcome.patch_paths[0])
        full = science_site.grid.full_window
        assert patch.read_window.area < full.area  # genuinely windowed read

        baseline = oracles.get("baseline", "b")
        store = _apply_patches(baseline, [patch])
        _differential(
            science_site, oracles.get("add", "b"), baseline,
            store, patch.write_window, label="midday-add",
        )

    def test_superseded_real_job_publishes_nothing(self, science_site) -> None:
        layer = _state_layer(science_site, "add")
        results_root = science_site.results_root("superseded")
        worker = _local_worker(science_site, layer, results_root)
        worker.bump_scene_revision()
        calls = {"n": 0}

        def provider():
            calls["n"] += 1
            return 1 if calls["n"] <= 2 else 2  # advances after windowing

        worker._revision_provider = provider
        outcome = worker.run(target_revision=1)
        assert outcome.status == "superseded"
        scenario_root = results_root / layer.scenario_id
        assert not list(scenario_root.rglob("rev-*"))
        assert not list((scenario_root / ".staging").glob("*.tmp"))
        assert worker.pending_batch() is not None

    def test_veg_svf_recompute_matches_cached_bitwise(self, science_site) -> None:
        """No-edit scene: the replayed accumulation equals the cached SVF."""
        layer = science_site.layer()
        scene = compose_full_scene_tensors(science_site.cache_a, layer)
        r0 = c0 = 60
        window = RasterWindow(r0, r0 + 48, c0, c0 + 48)
        bundle = window_svf_bundle(science_site.cache_a, scene, window, veg_changed=True)
        for name in (
            "svfveg", "svfEveg", "svfSveg", "svfWveg", "svfNveg",
            "svfaveg", "svfEaveg", "svfSaveg", "svfWaveg", "svfNaveg",
            "svftotal",
        ):
            np.testing.assert_array_equal(
                bundle[SVF_BUNDLE_INDEX[name]].numpy(),
                np.asarray(science_site.cache_a.svf[name])[r0:r0 + 48, c0:c0 + 48],
                err_msg=name,
            )
        np.testing.assert_array_equal(
            bundle[SVF_BUNDLE_INDEX["vegshmat"]].numpy(),
            np.asarray(science_site.cache_a.svf_patches["vegshadowmat"])[r0:r0 + 48, c0:c0 + 48, :],
        )
        np.testing.assert_array_equal(
            bundle[SVF_BUNDLE_INDEX["vbshvegshmat"]].numpy(),
            np.asarray(science_site.cache_a.svf_patches["vbshmat"])[r0:r0 + 48, c0:c0 + 48, :],
        )
