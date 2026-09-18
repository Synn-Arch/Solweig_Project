import json
import zipfile

import numpy as np
import pytest
from osgeo import gdal, osr

from solweig_gpu.incremental.cache import CacheValidationError, SiteCache
from solweig_gpu.incremental.cache_builder import build_site_cache
from solweig_gpu.incremental.cache_builder import main as cache_builder_main
from solweig_gpu.incremental.geometry import RasterWindow
from solweig_gpu.incremental.manifest import (
    CACHE_MODEL_VERSION,
    ManifestMismatchError,
    MissingManifestKeyError,
    load_manifest,
)

ROWS = 6
COLS = 8
PATCH_COUNT = 16
TIME_STEPS = 3
MET_COLUMNS = 25
PIXEL_SIZE = 2.0
ORIGIN = (1000.0, 2000.0)
EPSG = 32616

SVF_MEMBERS = (
    "svf",
    "svfE",
    "svfS",
    "svfW",
    "svfN",
    "svfveg",
    "svfEveg",
    "svfSveg",
    "svfWveg",
    "svfNveg",
    "svfaveg",
    "svfEaveg",
    "svfSaveg",
    "svfWaveg",
    "svfNaveg",
)


def _write_tif(path, array, *, dtype=gdal.GDT_Float32, nodata=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows, cols = array.shape
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(str(path), cols, rows, 1, dtype)
    dataset.SetGeoTransform(
        (ORIGIN[0], PIXEL_SIZE, 0.0, ORIGIN[1], 0.0, -PIXEL_SIZE)
    )
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(EPSG)
    dataset.SetProjection(srs.ExportToWkt())
    band = dataset.GetRasterBand(1)
    band.WriteArray(array)
    if nodata is not None:
        band.SetNoDataValue(float(nodata))
    dataset.FlushCache()
    dataset = None
    return path


def _write_met_file(path):
    header = "# year doy hour minute placeholder wind rh ta p radg rad diff radI wdir uhii"
    lines = [header]
    for hour in range(10, 10 + TIME_STEPS):
        row = (
            [2024.0, 172.0, float(hour), 0.0]
            + [0.0] * 5
            + [2.0, 55.0, 28.0, 1013.0, 0.0, 800.0]
            + [0.0] * 6
            + [150.0, 600.0, 270.0, 0.0]
        )
        assert len(row) == MET_COLUMNS
        lines.append(" ".join(f"{value:g}" for value in row))
    path.write_text("\n".join(lines) + "\n")
    return path


def _make_site(tmp_path, *, with_landcover=True):
    """Create a tiny synthetic prepared site in the processed_inputs layout."""
    site = tmp_path / "processed_inputs"
    for name in ("Building_DSM", "DEM", "Trees", "walls", "aspect", "SVF", "metfiles"):
        (site / name).mkdir(parents=True)

    rng = np.random.default_rng(20260901)
    dem = rng.uniform(250.0, 260.0, size=(ROWS, COLS)).astype(np.float32)
    building = dem + rng.choice(np.array([0.0, 12.0], dtype=np.float32), size=(ROWS, COLS))
    trees = rng.uniform(0.0, 18.0, size=(ROWS, COLS)) * (rng.random((ROWS, COLS)) < 0.3)
    walls = rng.uniform(0.0, 8.0, size=(ROWS, COLS)).astype(np.float32)
    aspect = rng.uniform(0.0, 360.0, size=(ROWS, COLS)).astype(np.float32)

    sources = {
        "dem": dem,
        "building_dsm": building.astype(np.float32),
        "trees": trees.astype(np.float32),
        "walls": walls,
        "wall_aspect": aspect,
    }
    _write_tif(site / "Building_DSM" / "Building_DSM_0_0.tif", sources["building_dsm"], nodata=-9999.0)
    _write_tif(site / "DEM" / "DEM_0_0.tif", dem, nodata=-9999.0)
    _write_tif(site / "Trees" / "Trees_0_0.tif", sources["trees"])
    _write_tif(site / "walls" / "walls_0_0.tif", walls)
    _write_tif(site / "aspect" / "aspect_0_0.tif", aspect)
    if with_landcover:
        (site / "Landcover").mkdir()
        landcover = rng.integers(1, 8, size=(ROWS, COLS)).astype(np.uint8)
        _write_tif(site / "Landcover" / "Landcover_0_0.tif", landcover, dtype=gdal.GDT_Byte)
        sources["landcover"] = landcover

    svf_dir = site / "SVF"
    for member in SVF_MEMBERS:
        sources[member] = rng.uniform(0.0, 1.0, size=(ROWS, COLS)).astype(np.float32)
    sources["svftotal"] = rng.uniform(0.0, 1.0, size=(ROWS, COLS)).astype(np.float32)
    _write_tif(svf_dir / "SkyViewFactor_0_0.tif", sources["svftotal"])
    scratch = tmp_path / "zip_members"
    with zipfile.ZipFile(svf_dir / "svfs_0_0.zip", "w") as archive:
        for member in SVF_MEMBERS:
            member_path = _write_tif(scratch / f"{member}.tif", sources[member])
            archive.write(member_path, arcname=f"{member}.tif")
    for name in ("shadowmat", "vegshadowmat", "vbshmat"):
        sources[name] = rng.integers(0, 2, size=(ROWS, COLS, PATCH_COUNT)).astype(np.float32)
    np.savez(
        svf_dir / "shadowmats_0_0.npz",
        **{name: sources[name] for name in ("shadowmat", "vegshadowmat", "vbshmat")},
    )

    _write_met_file(site / "metfiles" / "metfile_0_0_2024-06-20.txt")
    return site, sources


def _build_cache(site, cache_dir, **overrides):
    kwargs = dict(
        tile_key="0_0",
        site_id="synthetic-site",
        latitude=33.75,
        longitude=-84.39,
        altitude_m=250.0,
        utc_offset_hours=-5.0,
    )
    kwargs.update(overrides)
    return build_site_cache(site, cache_dir, **kwargs)


def _flip_last_byte(path):
    data = bytearray(path.read_bytes())
    data[-1] ^= 0xFF
    path.write_bytes(bytes(data))


@pytest.fixture
def built(tmp_path):
    site, sources = _make_site(tmp_path)
    cache_dir = tmp_path / "cache"
    _build_cache(site, cache_dir)
    return site, cache_dir, sources


def test_default_altitude_follows_oracle_rule(tmp_path):
    # The synthetic DEM is positive (250-260 m), so the oracle rule clamps
    # the site altitude to 3.0; a build without an explicit altitude must
    # record exactly that, or the solver's location validation rejects
    # every job on the cache.
    import torch

    site, sources = _make_site(tmp_path)
    cache_dir = tmp_path / "cache"
    manifest = _build_cache(site, cache_dir, altitude_m=None)
    assert manifest.solar_geometry.altitude_m == 3.0

    expected = float(torch.median(torch.from_numpy(sources["dem"])).item())
    assert expected > 0.0  # guard the test's own premise


def test_default_altitude_negative_dem_not_clamped():
    from solweig_gpu.incremental.cache_builder import oracle_altitude_m

    dem = np.full((6, 6), -5.0, dtype=np.float32)
    assert oracle_altitude_m(dem) == -5.0
    flat = np.zeros((6, 6), dtype=np.float32)
    assert oracle_altitude_m(flat) == 0.0


def test_explicit_altitude_recorded_verbatim(tmp_path):
    site, _ = _make_site(tmp_path)
    cache_dir = tmp_path / "cache"
    manifest = _build_cache(site, cache_dir, altitude_m=250.0)
    assert manifest.solar_geometry.altitude_m == 250.0


def test_manifest_missing_key_error_names_the_field(built):
    _, cache_dir, _ = built
    data = json.loads((cache_dir / "manifest.json").read_text())
    del data["rows"]
    (cache_dir / "manifest.json").write_text(json.dumps(data))
    with pytest.raises(MissingManifestKeyError, match="'rows'"):
        load_manifest(cache_dir / "manifest.json")


def test_manifest_hash_mismatch_detected(built):
    _, cache_dir, _ = built
    _flip_last_byte(cache_dir / "svf" / "svfE.f32.npy")
    manifest = load_manifest(cache_dir / "manifest.json")
    with pytest.raises(ManifestMismatchError, match=r"arrays\.svfE\.sha256"):
        manifest.validate(cache_dir)


def test_manifest_shape_mismatch_detected(built):
    _, cache_dir, _ = built
    data = json.loads((cache_dir / "manifest.json").read_text())
    data["arrays"]["dem"]["shape"] = [ROWS + 1, COLS]
    (cache_dir / "manifest.json").write_text(json.dumps(data))
    with pytest.raises(ManifestMismatchError, match=r"arrays\.dem\.shape"):
        load_manifest(cache_dir / "manifest.json").validate(cache_dir)


@pytest.mark.parametrize(
    "window",
    [
        RasterWindow(0, ROWS, 0, COLS),
        RasterWindow(1, 4, 2, 7),
        RasterWindow(0, 1, 0, 1),
        RasterWindow(ROWS - 2, ROWS, COLS - 3, COLS),
    ],
)
def test_window_slices_match_written_arrays(built, window):
    _, cache_dir, sources = built
    cache = SiteCache.load(cache_dir)
    expected = sources["dem"][
        window.row_start : window.row_stop, window.col_start : window.col_stop
    ]
    np.testing.assert_array_equal(cache.window("dem", window), expected)


def test_window_slice_of_patch_cube_keeps_patch_axis(built):
    _, cache_dir, sources = built
    cache = SiteCache.load(cache_dir)
    window = RasterWindow(1, 4, 2, 7)
    patch = cache.window("shadowmat", window)
    assert patch.shape == (window.height, window.width, PATCH_COUNT)
    np.testing.assert_array_equal(patch, sources["shadowmat"][1:4, 2:7, :])


def test_window_outside_grid_raises(built):
    _, cache_dir, _ = built
    cache = SiteCache.load(cache_dir)
    with pytest.raises(ValueError, match="exceeds array"):
        cache.window("dem", RasterWindow(ROWS - 2, ROWS + 1, 0, 2))


def test_cache_arrays_are_read_only(built):
    _, cache_dir, _ = built
    cache = SiteCache.load(cache_dir)
    dem = cache.dem
    assert isinstance(dem, np.memmap)
    assert not dem.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        dem[0, 0] = 99.0
    view = cache.window("dem", RasterWindow(0, 2, 0, 2))
    with pytest.raises(ValueError, match="read-only"):
        view[...] = 0.0


def test_builder_is_deterministic(tmp_path):
    site, _ = _make_site(tmp_path)
    first = tmp_path / "cache-a"
    second = tmp_path / "cache-b"
    _build_cache(site, first)
    _build_cache(site, second)
    assert (first / "manifest.json").read_bytes() == (second / "manifest.json").read_bytes()
    assert (
        first / "static" / "dem.f32.npy"
    ).read_bytes() == (second / "static" / "dem.f32.npy").read_bytes()
    SiteCache.load(second)


def test_warm_load_performs_no_raster_reads(built, monkeypatch):
    _, cache_dir, _ = built
    import rasterio
    from osgeo import gdal

    def _forbidden(*args, **kwargs):
        raise AssertionError("warm cache load opened a raster")

    monkeypatch.setattr(rasterio, "open", _forbidden)
    monkeypatch.setattr(gdal, "Open", _forbidden)
    monkeypatch.setattr(gdal, "OpenEx", _forbidden)

    cache = SiteCache.load(cache_dir)  # self-test still runs at load
    window = cache.window("building_dsm", RasterWindow(1, 4, 2, 5))
    assert window.shape == (3, 3)
    assert cache.met.shape == (TIME_STEPS, MET_COLUMNS)
    assert cache.solar.shape[0] == TIME_STEPS


def test_self_test_detects_tampering(built):
    _, cache_dir, _ = built
    SiteCache.load(cache_dir)
    _flip_last_byte(cache_dir / "static" / "tree_base.f32.npy")
    with pytest.raises(CacheValidationError, match="tree_base"):
        SiteCache.load(cache_dir)


def test_metadata_exposes_layout_and_model_version(built):
    _, cache_dir, _ = built
    metadata = SiteCache.load(cache_dir).metadata()
    assert metadata["model_version"] == CACHE_MODEL_VERSION
    assert metadata["cache_schema_version"] == 1
    assert metadata["site_id"] == "synthetic-site"
    assert metadata["patch_count"] == PATCH_COUNT
    assert metadata["time_steps"] == TIME_STEPS
    assert metadata["layout"]["dem"] == "static/dem.f32.npy"
    assert metadata["layout"]["solar"] == "forcing/solar_geometry.f32.npy"


def test_site_without_landcover_builds_and_loads(tmp_path):
    site, _ = _make_site(tmp_path, with_landcover=False)
    cache_dir = tmp_path / "cache"
    _build_cache(site, cache_dir)
    cache = SiteCache.load(cache_dir)
    assert "landcover" not in cache.array_names()
    with pytest.raises(KeyError, match="landcover"):
        cache.landcover


def test_builder_reports_missing_tile_raster(tmp_path):
    site, _ = _make_site(tmp_path)
    (site / "DEM" / "DEM_0_0.tif").unlink()
    with pytest.raises(FileNotFoundError, match="DEM"):
        _build_cache(site, tmp_path / "cache")


def test_cli_validate_only_reports_success(built, capsys):
    site, cache_dir, _ = built
    code = cache_builder_main(
        ["--site", str(site), "--out", str(cache_dir), "--validate-only"]
    )
    assert code == 0
    assert "[OK]" in capsys.readouterr().out


def test_cli_validate_only_detects_tampering(built, capsys):
    _, cache_dir, _ = built
    _flip_last_byte(cache_dir / "svf" / "svfveg.f32.npy")
    code = cache_builder_main(["--out", str(cache_dir), "--validate-only"])
    assert code == 1
    assert "svfveg" in capsys.readouterr().err
