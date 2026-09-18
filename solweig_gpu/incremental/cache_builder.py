# SPDX-License-Identifier: GPL-3.0-only
"""Deterministic builder for the incremental SOLWEIG baseline site cache.

Converts a prepared site directory (the ``processed_inputs`` layout produced
by the preprocessor) into the memory-mapped ``.npy`` cache described in
``docs/incremental_design_tool/data_model.md``:

```text
site/                              cache/
├── Building_DSM/Building_DSM_X_Y.tif      ├── manifest.json
├── DEM/DEM_X_Y.tif                         ├── static/*.npy
├── Trees/Trees_X_Y.tif                     ├── svf/*.npy
├── walls/walls_X_Y.tif                     └── forcing/*.npy
├── aspect/aspect_X_Y.tif
├── Landcover/Landcover_X_Y.tif (optional)
├── SVF/SkyViewFactor_X_Y.tif
├── SVF/svfs_X_Y.zip
├── SVF/shadowmats_X_Y.npz
└── metfiles/metfile_X_Y_<date>.txt
```

The build is deterministic for identical inputs: files are discovered through
sorted directory listings, arrays are written with ``numpy.save``, and the
manifest is serialized as sorted-key JSON with paths relative to their root.

Command line usage:

```text
python -m solweig_gpu.incremental.cache_builder --site <dir> --out <cache_dir> \
    [--validate] [--validate-only] [--tile 0_0] [--latitude ..] [--longitude ..]
```
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .cache import SiteCache
from .manifest import (
    CACHE_ARRAY_LAYOUT,
    CACHE_MODEL_VERSION,
    CACHE_SCHEMA_VERSION,
    MANIFEST_FILENAME,
    SOLAR_GEOMETRY_COLUMNS,
    SVF_PATCH_ARRAYS,
    SVF_SCALAR_ARRAYS,
    ArrayEntry,
    AuxSourceEntry,
    RasterSourceEntry,
    SiteManifest,
    SolarGeometrySpec,
    hash_text,
    load_manifest,
    sha256_file,
)

_TILE_SUFFIX_RE = re.compile(r"_(\d+)_(\d+)$")

_TILE_KEY_RE = re.compile(r"^\d+_\d+$")

#: Minimum met-table column count consumed by ``utci_process.compute_utci``.
MIN_MET_COLUMNS = 23

#: Site subdirectory and file-name prefix for each required raster input.
_RASTER_DIRECTORIES: dict[str, str] = {
    "building_dsm": "Building_DSM",
    "dem": "DEM",
    "trees": "Trees",
    "walls": "walls",
    "aspect": "aspect",
    "landcover": "Landcover",
}


def build_site_cache(
    site_dir: str | Path,
    cache_dir: str | Path,
    *,
    tile_key: str = "0_0",
    site_id: str | None = None,
    met_file: str | Path | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    altitude_m: float | None = None,
    utc_offset_hours: float = 0.0,
    model_version: str = CACHE_MODEL_VERSION,
    validate: bool = False,
) -> SiteManifest:
    """Build a site cache from a prepared site directory.

    Args:
        site_dir: Prepared ``processed_inputs`` directory for one site.
        cache_dir: Output directory for the cache; created when absent.
        tile_key: Tile identifier such as ``"0_0"``.
        site_id: Stable site identifier; defaults to the site directory name.
        met_file: Explicit met forcing text file; auto-discovered when None.
        latitude, longitude, altitude_m, utc_offset_hours: Location used to
            persist the solar-geometry series exactly once. When
            ``altitude_m`` is None it is derived from the site DEM with the
            oracle's altitude rule (median, clamped to 3 m when positive) —
            the value the solver's location validation demands.
        model_version: Model stamp recorded in the manifest.
        validate: Run the cache self-test immediately after building.

    Returns:
        The written :class:`SiteManifest`.
    """
    if not _TILE_KEY_RE.fullmatch(tile_key):
        raise ValueError(f"tile_key must look like '<int>_<int>', got '{tile_key}'")
    if latitude is None or longitude is None:
        raise ValueError(
            "latitude and longitude are required so the solar-geometry series "
            "can be persisted once per cache"
        )
    site_root = Path(site_dir).resolve()
    if not site_root.is_dir():
        raise FileNotFoundError(f"site directory not found: {site_root}")
    cache_root = Path(cache_dir)

    raster_paths: dict[str, Path] = {}
    for name in ("building_dsm", "dem", "trees", "walls", "aspect"):
        raster_paths[name] = _find_tile_raster(
            site_root / _RASTER_DIRECTORIES[name], tile_key
        )
    landcover_path = _find_tile_raster(
        site_root / _RASTER_DIRECTORIES["landcover"], tile_key, required=False
    )
    if landcover_path is not None:
        raster_paths["landcover"] = landcover_path
    svftotal_path = site_root / "SVF" / f"SkyViewFactor_{tile_key}.tif"
    if not svftotal_path.is_file():
        raise FileNotFoundError(f"SVF total raster not found: {svftotal_path}")
    svf_zip_path = site_root / "SVF" / f"svfs_{tile_key}.zip"
    if not svf_zip_path.is_file():
        raise FileNotFoundError(f"SVF zip not found: {svf_zip_path}")
    svf_npz_path = site_root / "SVF" / f"shadowmats_{tile_key}.npz"
    if not svf_npz_path.is_file():
        raise FileNotFoundError(f"shadowmats NPZ not found: {svf_npz_path}")
    met_path = (
        Path(met_file)
        if met_file is not None
        else _find_met_file(site_root / "metfiles", tile_key)
    )

    # ---- read source rasters (the only GDAL reads in the cache lifetime) ----
    rasters: dict[str, tuple[np.ndarray, float | None, tuple[float, ...], str]] = {}
    for name in sorted(raster_paths):
        rasters[name] = _read_raster(raster_paths[name])
    rasters["svftotal"] = _read_raster(svftotal_path)

    reference = rasters["building_dsm"]
    rows, cols = reference[0].shape
    pixel_size_m = _validate_raster_grid(rasters)

    # ---- read the SVF bundle in the existing standalone cache layout ----
    svf_arrays: dict[str, np.ndarray] = {}
    for member in SVF_SCALAR_ARRAYS:
        if member == "svftotal":
            svf_arrays[member] = np.asarray(rasters["svftotal"][0], dtype=np.float32)
        else:
            svf_arrays[member] = _read_zip_raster(svf_zip_path, f"{member}.tif")
    patch_arrays: dict[str, np.ndarray] = {}
    with np.load(svf_npz_path) as npz:
        for name in SVF_PATCH_ARRAYS:
            if name not in npz.files:
                raise KeyError(f"shadowmats NPZ {svf_npz_path.name} is missing key '{name}'")
            patch_arrays[name] = np.asarray(npz[name], dtype=np.float32)
    patch_count = int(patch_arrays["shadowmat"].shape[-1]) if patch_arrays["shadowmat"].ndim else 0
    for name, array in patch_arrays.items():
        if array.ndim != 3 or array.shape[:2] != (rows, cols) or array.shape[2] <= 0:
            raise ValueError(
                f"SVF patch array '{name}' must be [rows={rows}, cols={cols}, patch], "
                f"got shape {array.shape}"
            )

    # ---- met forcing and solar geometry, persisted once ----
    if altitude_m is None:
        altitude_m = oracle_altitude_m(rasters["dem"][0])
    met_table = _read_met_table(met_path)
    time_steps = int(met_table.shape[0])
    solar_series = _compute_solar_geometry(
        met_table,
        latitude=float(latitude),
        longitude=float(longitude),
        altitude_m=float(altitude_m),
        utc_offset_hours=float(utc_offset_hours),
    )

    # ---- assemble the deterministic array set ----
    raster_to_cache_names: dict[str, str] = {
        "building_dsm": "building_dsm",
        "dem": "dem",
        "trees": "tree_base",
        "walls": "walls",
        "aspect": "wall_aspect",
    }
    prepared: dict[str, np.ndarray] = {
        cache_name: np.asarray(rasters[source_name][0], dtype=np.float32)
        for source_name, cache_name in raster_to_cache_names.items()
    }
    if "landcover" in rasters:
        prepared["landcover"] = _cast_uint8(rasters["landcover"][0], "landcover")
    prepared.update(svf_arrays)
    prepared.update(patch_arrays)
    prepared["met"] = met_table.astype(np.float32)
    prepared["solar"] = solar_series
    for name, array in sorted(prepared.items()):
        if name in ("met", "solar"):
            if array.ndim != 2 or array.shape[0] != time_steps:
                raise ValueError(
                    f"cache array '{name}' must start with {time_steps} time "
                    f"steps, got shape {array.shape}"
                )
            continue
        expected = (
            (rows, cols, patch_count) if name in SVF_PATCH_ARRAYS else (rows, cols)
        )
        if array.shape != expected:
            raise ValueError(
                f"cache array '{name}' has shape {array.shape}, expected {expected}"
            )

    array_entries: dict[str, ArrayEntry] = {}
    for name in sorted(prepared):
        array_entries[name] = _write_cache_array(cache_root, name, prepared[name])

    # ---- manifest with site-relative source paths ----
    raster_entries = {
        name: RasterSourceEntry(
            path=_relative(site_root, raster_paths[name]),
            sha256=sha256_file(raster_paths[name]),
            shape=(int(rasters[name][0].shape[0]), int(rasters[name][0].shape[1])),
            dtype=str(rasters[name][0].dtype),
            nodata=rasters[name][1],
            geotransform=rasters[name][2],
            crs_wkt_hash=hash_text(rasters[name][3]),
        )
        for name in sorted(raster_paths)
    }
    raster_entries["svftotal"] = RasterSourceEntry(
        path=_relative(site_root, svftotal_path),
        sha256=sha256_file(svftotal_path),
        shape=(rows, cols),
        dtype=str(rasters["svftotal"][0].dtype),
        nodata=rasters["svftotal"][1],
        geotransform=rasters["svftotal"][2],
        crs_wkt_hash=hash_text(rasters["svftotal"][3]),
    )
    source_entries = {
        "svf_zip": AuxSourceEntry(
            path=_relative(site_root, svf_zip_path), sha256=sha256_file(svf_zip_path)
        ),
        "svf_npz": AuxSourceEntry(
            path=_relative(site_root, svf_npz_path), sha256=sha256_file(svf_npz_path)
        ),
        "met": AuxSourceEntry(
            path=_relative(site_root, met_path),
            sha256=sha256_file(met_path),
            shape=(time_steps, int(met_table.shape[1])),
            dtype="float64",
        ),
    }
    manifest = SiteManifest(
        site_id=site_id if site_id is not None else site_root.name,
        tile_key=tile_key,
        model_version=model_version,
        cache_schema_version=CACHE_SCHEMA_VERSION,
        rows=int(rows),
        cols=int(cols),
        pixel_size_m=pixel_size_m,
        origin_x_m=reference[2][0],
        origin_y_m=reference[2][3],
        crs_wkt_hash=hash_text(reference[3]),
        time_steps=time_steps,
        patch_count=patch_count,
        rasters=raster_entries,
        sources=source_entries,
        solar_geometry=SolarGeometrySpec(
            latitude=float(latitude),
            longitude=float(longitude),
            altitude_m=float(altitude_m),
            utc_offset_hours=float(utc_offset_hours),
        ),
        arrays=array_entries,
    )
    manifest.write(cache_root / MANIFEST_FILENAME)

    if validate:
        SiteCache.load(cache_root)
    return manifest


# ----------------------------------------------------------------------
# Source discovery
# ----------------------------------------------------------------------


def oracle_altitude_m(dem: "np.ndarray") -> float:
    """The oracle's site-altitude rule: median DEM, 3 m when positive.

    Replicates ``compute_utci``'s location derivation (see the solver's
    ``_oracle_location``) bit-for-bit, including ``torch.median`` semantics
    for even-sized rasters. The solver rejects caches whose recorded
    ``altitude_m`` disagrees with this rule, so an unspecified build default
    must be the rule itself — a hard-coded 0.0 silently bricks every worker
    job on positive-relief sites.
    """
    dem_t = torch.from_numpy(np.array(dem, dtype=np.float32))
    alt = float(torch.median(dem_t).item())
    return 3.0 if alt > 0 else alt


def _tile_suffix_key(stem: str) -> str | None:
    match = _TILE_SUFFIX_RE.search(stem)
    return f"{match.group(1)}_{match.group(2)}" if match else None


def _find_tile_raster(
    directory: Path, tile_key: str, *, required: bool = True
) -> Path | None:
    """Find the unique ``*_<x>_<y>.tif`` tile raster for ``tile_key``."""
    if not directory.is_dir():
        if required:
            raise FileNotFoundError(f"site raster directory not found: {directory}")
        return None
    matches = sorted(
        path
        for path in directory.glob("*.tif")
        if _tile_suffix_key(path.stem) == tile_key
    )
    if not matches:
        if required:
            raise FileNotFoundError(
                f"no '{tile_key}' tile raster found in {directory}"
            )
        return None
    if len(matches) > 1:
        names = [path.name for path in matches]
        raise ValueError(f"ambiguous '{tile_key}' tile rasters in {directory}: {names}")
    return matches[0]


def _find_met_file(directory: Path, tile_key: str) -> Path:
    """Find the unique ``metfile_<x>_<y>*.txt`` forcing file for ``tile_key``."""
    if not directory.is_dir():
        raise FileNotFoundError(f"metfiles directory not found: {directory}")
    matches = sorted(
        path
        for path in directory.glob("*.txt")
        if path.stem.startswith(f"metfile_{tile_key}")
    )
    if not matches:
        raise FileNotFoundError(f"no metfile for tile '{tile_key}' found in {directory}")
    if len(matches) > 1:
        names = [path.name for path in matches]
        raise ValueError(
            f"ambiguous metfiles for tile '{tile_key}' in {directory}: {names}; "
            "pass met_file explicitly"
        )
    return matches[0]


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root).as_posix()


# ----------------------------------------------------------------------
# Source readers
# ----------------------------------------------------------------------


def _read_raster(
    path: Path,
) -> tuple[np.ndarray, float | None, tuple[float, float, float, float, float, float], str]:
    """Read one GeoTIFF band with GDAL, returning (array, nodata, geotransform, WKT)."""
    from osgeo import gdal  # imported lazily so warm workers never load GDAL

    dataset = gdal.Open(str(path))
    if dataset is None:
        raise ValueError(f"could not open raster: {path}")
    band = dataset.GetRasterBand(1)
    array = band.ReadAsArray()
    if array is None:
        raise ValueError(f"could not read band 1 of raster: {path}")
    nodata = band.GetNoDataValue()
    geotransform = tuple(float(value) for value in dataset.GetGeoTransform())
    wkt = dataset.GetProjection()
    dataset = None
    return array, nodata, geotransform, wkt  # type: ignore[return-value]


def _read_zip_raster(zip_path: Path, member_name: str) -> np.ndarray:
    """Read a GeoTIFF member of an SVF zip via GDAL ``/vsizip/``."""
    from osgeo import gdal

    dataset = gdal.Open(f"/vsizip/{zip_path}/{member_name}")
    if dataset is None:
        raise ValueError(f"could not open {member_name} inside {zip_path}")
    array = dataset.GetRasterBand(1).ReadAsArray()
    dataset = None
    if array is None:
        raise ValueError(f"could not read {member_name} inside {zip_path}")
    return np.asarray(array, dtype=np.float32)


def _read_met_table(path: Path) -> np.ndarray:
    """Read a met forcing file the same way ``run_utci_tiles`` does."""
    table = np.loadtxt(path, skiprows=1, delimiter=" ")
    table = np.atleast_2d(np.asarray(table, dtype=np.float64))
    if table.ndim != 2 or table.shape[1] < MIN_MET_COLUMNS:
        raise ValueError(
            f"met file {path.name} must supply at least {MIN_MET_COLUMNS} columns, "
            f"got {table.shape[1] if table.ndim == 2 else table.ndim} "
            "(expected space-separated rows after one header line)"
        )
    return table


def _compute_solar_geometry(
    met_table: np.ndarray,
    *,
    latitude: float,
    longitude: float,
    altitude_m: float,
    utc_offset_hours: float,
) -> np.ndarray:
    """Persist the solar-geometry series once using the SOLWEIG metdata kernel."""
    from ..sun_position import Solweig_2015a_metdata_noload  # lazy heavy import

    location = {
        "latitude": latitude,
        "longitude": longitude,
        "altitude": altitude_m,
    }
    (
        _yyyy,
        altitude,
        azimuth,
        zenith,
        jday,
        leafon,
        dectime,
        altmax,
    ) = Solweig_2015a_metdata_noload(met_table, location, utc_offset_hours)
    series = np.column_stack(
        [
            np.asarray(altitude).ravel(),
            np.asarray(azimuth).ravel(),
            np.asarray(zenith).ravel(),
            np.asarray(jday).ravel(),
            np.asarray(dectime).ravel(),
            np.asarray(altmax).ravel(),
            np.asarray(leafon).ravel(),
        ]
    ).astype(np.float32)
    if series.shape != (met_table.shape[0], len(SOLAR_GEOMETRY_COLUMNS)):
        raise ValueError(
            f"solar-geometry series has shape {series.shape}, expected "
            f"({met_table.shape[0]}, {len(SOLAR_GEOMETRY_COLUMNS)})"
        )
    return series


# ----------------------------------------------------------------------
# Validation helpers
# ----------------------------------------------------------------------


def _validate_raster_grid(
    rasters: dict[str, tuple[np.ndarray, float | None, tuple[float, ...], str]],
) -> float:
    """Check shared shape, north-up square-pixel transform, and CRS."""
    shapes = {
        name: tuple(int(value) for value in entry[0].shape)
        for name, entry in rasters.items()
    }
    if len(set(shapes.values())) != 1:
        raise ValueError(f"source raster shapes disagree: {shapes}")
    geotransforms = {name: entry[2] for name, entry in rasters.items()}
    reference_gt = next(iter(geotransforms.values()))
    for name, gt in sorted(geotransforms.items()):
        if any(abs(left - right) > 1e-9 for left, right in zip(gt, reference_gt)):
            raise ValueError(
                f"raster '{name}' geotransform {gt} does not match reference "
                f"{reference_gt}"
            )
    if reference_gt[1] <= 0 or reference_gt[5] >= 0:
        raise ValueError(
            f"geotransform must be north-up with positive pixel size, got {reference_gt}"
        )
    if abs(abs(reference_gt[5]) - reference_gt[1]) > 1e-9 or reference_gt[2] != 0.0 or reference_gt[4] != 0.0:
        raise ValueError(
            f"geotransform must use square, axis-aligned pixels, got {reference_gt}"
        )
    wkt_hashes = {name: hash_text(entry[3]) for name, entry in rasters.items()}
    if len(set(wkt_hashes.values())) != 1:
        raise ValueError("source rasters do not share one CRS")
    return reference_gt[1]


def _cast_uint8(array: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(array)
    if values.size and (values.min() < 0 or values.max() > 255):
        raise ValueError(f"categorical array '{name}' values do not fit uint8")
    return values.astype(np.uint8)


def _write_cache_array(cache_root: Path, name: str, array: np.ndarray) -> ArrayEntry:
    """Write one deterministic ``.npy`` and describe it for the manifest."""
    if name not in CACHE_ARRAY_LAYOUT:
        raise ValueError(f"no cache layout path defined for array '{name}'")
    relative = CACHE_ARRAY_LAYOUT[name]
    path = cache_root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        np.save(handle, array)
    return ArrayEntry(
        path=relative,
        shape=tuple(int(value) for value in array.shape),
        dtype=str(array.dtype),
        nbytes=int(array.nbytes),
        sha256=sha256_file(path),
    )


# ----------------------------------------------------------------------
# Command line interface
# ----------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m solweig_gpu.incremental.cache_builder",
        description="Build or validate the incremental SOLWEIG baseline site cache.",
    )
    parser.add_argument("--site", help="prepared processed_inputs site directory")
    parser.add_argument("--out", required=True, help="cache directory to build or validate")
    parser.add_argument("--tile", default="0_0", help="tile key such as 0_0 (default: 0_0)")
    parser.add_argument("--site-id", default=None, help="stable site identifier")
    parser.add_argument("--met-file", default=None, help="explicit met forcing text file")
    parser.add_argument("--latitude", type=float, default=None, help="site latitude")
    parser.add_argument("--longitude", type=float, default=None, help="site longitude")
    parser.add_argument(
        "--altitude",
        type=float,
        default=None,
        help="site elevation in meters (default: oracle rule — median DEM, 3 m when positive)",
    )
    parser.add_argument("--utc-offset", type=float, default=0.0, help="UTC offset in hours")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="run the cache self-test after building",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate the existing cache at --out without building",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.validate_only:
            cache_root = Path(args.out)
            manifest = load_manifest(cache_root / MANIFEST_FILENAME)
            manifest.validate(cache_root)
            SiteCache.load(cache_root)
            if args.site:
                manifest.verify_sources(args.site)
            print(f"[OK] cache at {cache_root} matches its manifest")
            return 0
        if not args.site:
            parser.error("--site is required when building a cache")
        manifest = build_site_cache(
            args.site,
            args.out,
            tile_key=args.tile,
            site_id=args.site_id,
            met_file=args.met_file,
            latitude=args.latitude,
            longitude=args.longitude,
            altitude_m=args.altitude,
            utc_offset_hours=args.utc_offset,
            validate=args.validate,
        )
    except (ValueError, OSError) as error:
        print(f"[error] {error}", file=sys.stderr)
        return 1
    print(
        f"[OK] built cache for site '{manifest.site_id}' tile {manifest.tile_key} "
        f"({manifest.rows}x{manifest.cols}, {manifest.time_steps} steps, "
        f"{manifest.patch_count} patches) at {Path(args.out)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
