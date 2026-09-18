# API Reference

This section provides a detailed reference for the public functions of the SOLWEIG-GPU package.

**Entry points (exported from `solweig_gpu`):**

- **`thermal_comfort(...)`** – One-shot: runs wind-coefficient calculation (optional), preprocessing, wall/aspect, SVF, and UTCI in one call. Use this for normal runs (the `thermal_comfort` CLI calls this).
- **`preprocess(...)`**, **`run_walls_aspect(preprocess_dir)`**, **`calculate_svf(...)`**, **`run_utci_tiles(...)`** – Staged execution: call these when you need to run preprocessing once, then wall/aspect, then SVF, then UTCI (optionally for a subset of tiles). See [Developer Guide – Pipeline stages](developer_guide.md#pipeline-stages).
- **`build_inputs(...)`** – Download and build the required input rasters and meteorological data for a location from near-globally available datasets (requires Google Earth Engine authentication).
- **`build_wind_ext_coeff(...)`** – Compute direction-based wind-extension coefficient rasters (GLIDE-SOL feature; requires ERA5 forecast surface roughness).

---

## `thermal_comfort()`

Main function to run a full SOLWEIG-GPU simulation (optional wind coefficients → preprocess → wall/aspect → SVF → UTCI).

```python
from solweig_gpu import thermal_comfort

thermal_comfort(
    base_path,
    selected_date_str,
    building_dsm_filename='Building_DSM.tif',
    dem_filename='DEM.tif',
    trees_filename='Trees.tif',
    landcover_filename=None,
    ERA_5_z0_find=True,
    tile_size=3600,
    overlap=20,
    use_own_met=True,
    start_time=None,
    end_time=None,
    data_source_type=None,
    data_folder=None,
    own_met_file=None,
    use_uhi=True,
    save_tmrt=True,
    save_svf=False,
    save_kup=False,
    save_kdown=False,
    save_lup=False,
    save_ldown=False,
    save_shadow=False,
    save_wbgt=False,
    save_ta=False,
    save_wind=False,
)
```

### Parameters

-   `base_path` (str): Base directory for `output_folder/` and `processed_inputs/`; also used to resolve relative raster paths. To write outputs elsewhere, set this to that directory and pass complete paths for the raster parameters.
-   `selected_date_str` (str): The date for the simulation in `YYYY-MM-DD` format (local time of the study area).
-   `building_dsm_filename` (str): Path or filename of the Building DSM raster (relative to `base_path`, or a complete path). Defaults to `'Building_DSM.tif'`.
-   `dem_filename` (str): Path or filename of the DEM raster. Defaults to `'DEM.tif'`.
-   `trees_filename` (str): Path or filename of the Tree DSM raster. Defaults to `'Trees.tif'`.
-   `landcover_filename` (str, optional): Path or filename of the land cover raster. Defaults to `None`.
-   `ERA_5_z0_find` (bool): If `True`, SOLWEIG-GPU calls `build_wind_ext_coeff()` before preprocessing to compute directional wind-extension coefficients, and expects the ERA5 file `data_stream-oper_stepType-instant.nc` (containing forecast surface roughness, `fsr`) to be present in `data_folder`. If the file is not found, a warning is printed and the run continues without wind coefficients. Set `False` to skip wind coefficients. Defaults to `True`.
-   `tile_size` (int): The size of the tiles in pixels. Defaults to `3600`.
-   `overlap` (int): The overlap between tiles in pixels (used for shadow transfer between neighboring tiles). Defaults to `20`.
-   `use_own_met` (bool): Whether to use a custom meteorological file. Defaults to `True`.
-   `start_time` (str, optional): The start time of the simulation in `YYYY-MM-DD HH:MM:SS` format (UTC). Required if `use_own_met` is `False`.
-   `end_time` (str, optional): The end time of the simulation in `YYYY-MM-DD HH:MM:SS` format (UTC). Required if `use_own_met` is `False`.
-   `data_source_type` (str, optional): The type of meteorological data source (`'ERA5'` or `'wrfout'`). Required if `use_own_met` is `False`.
-   `data_folder` (str, optional): The directory containing the meteorological data files. Required if `use_own_met` is `False`.
-   `own_met_file` (str, optional): The path to the custom meteorological file. Required if `use_own_met` is `True`.
-   `use_uhi` (bool): If `True`, use UHI-aware ERA5 processing and write the diagnostic urban heat island intensity (`UHI_CYCLE`/`uhii`) into the generated metfiles. Use only with ERA5 forcing; keep `False` for WRF or user-provided met files. Defaults to `True`.
-   `save_tmrt` (bool, optional): Whether to save the Mean Radiant Temperature output. Defaults to `True`.
-   `save_svf` (bool, optional): Whether to save the Sky View Factor output. Defaults to `False`.
-   `save_kup` (bool, optional): Whether to save the upwelling shortwave radiation output. Defaults to `False`.
-   `save_kdown` (bool, optional): Whether to save the downwelling shortwave radiation output. Defaults to `False`.
-   `save_lup` (bool, optional): Whether to save the upwelling longwave radiation output. Defaults to `False`.
-   `save_ldown` (bool, optional): Whether to save the downwelling longwave radiation output. Defaults to `False`.
-   `save_shadow` (bool, optional): Whether to save the shadow map output. Defaults to `False`.
-   `save_wbgt` (bool, optional): Whether to save the Wet Bulb Globe Temperature (WBGT) output. Defaults to `False`.
-   `save_ta` (bool, optional): Whether to save the diagnostic air temperature field. Defaults to `False`.
-   `save_wind` (bool, optional): Whether to save the diagnostic wind speed field. Defaults to `False`.

UTCI is always computed and saved.

---

## `build_inputs()`

Downloads and builds the static and meteorological inputs required by SOLWEIG-GPU for a given location, using near-globally available urban datasets (WorldCover, tree canopy, DEM, LCZ, OSM/GBA building vectors). Requires an authenticated Google Earth Engine project and the optional dependencies `earthengine-api`, `geemap`, `geopandas`, and `osmnx`.

```python
import os
from solweig_gpu import build_inputs

os.environ["EE_PROJECT"] = "your-gee-project-id"

base_path = build_inputs(
    lat=30.27,
    lon=-97.74,
    city="Austin",
    km_buffer=8.0,
    km_reduced_lat=3.0,
    km_reduced_lon=1.0,
    base_folder="/path/to/save/inputs",
    resolution=2.0,
)
```

**Parameters:**

-   `lat`, `lon` (float): Center of the area in degrees.
-   `city` (str, optional): Name for the output folder. If `None`, derived from reverse geocoding.
-   `km_buffer` (float): Half-size of the initial bounding box in km. Defaults to `8.0`.
-   `km_reduced_lat`, `km_reduced_lon` (float): Shrink (N/S and E/W) from the bounding box for the SOLWEIG domain, in km. Defaults to `3.0` and `1.0`.
-   `base_folder` (str, optional): Workspace root for the downloads and outputs.
-   `resolution` (float): Reference grid resolution in meters. Defaults to `2.0`.

**Returns:** Path to the output directory (e.g. `{base_folder}/{city}_for_solweig`) containing `Building_DSM.tif`, `DEM.tif`, `Trees.tif`, `Landuse.tif`, and meteorological NetCDF data. Use this path as `base_path` for `preprocess()` or `thermal_comfort()`.

---

## `build_wind_ext_coeff()`

Builds full-domain, direction-based wind-extension coefficient rasters (`WindCoeff_dir000.tif` … `WindCoeff_dir330.tif`, every 30°) from the building and tree rasters and the ERA5 roughness length. This implements the GLIDE-SOL wind scheme (Zonato et al., 2026).

```python
from solweig_gpu import build_wind_ext_coeff

build_wind_ext_coeff(
    "/path/to/solweig/input",  # folder with Building_DSM.tif and Trees.tif
    "/path/to/era5",           # folder with data_stream-oper_stepType-instant.nc
)
```

**Parameters:**

-   `input_dir` (str | Path): Directory containing the processed SOLWEIG raster inputs (`Buildings.tif` or `Building_DSM.tif`, and `Trees.tif`). The output `WindCoeff_dir*.tif` rasters are written into this same directory.
-   `era5_dir` (str | Path): Directory containing the ERA5 NetCDF file `data_stream-oper_stepType-instant.nc` with forecast surface roughness (`fsr`). The roughness is extracted at the midpoint of the building raster and averaged over all available times.
-   Keyword-only tuning parameters (optional): `directions`, `z0_ref`, `hmin_b`, `hmin_t`, `z_eval`, `zref`, `LAI_t`, `a0_t`, `a1_t`, `alpha_min_t`, `alpha_max_t`, `coeff_min`, `coeff_max`, `lp_min_open`, `max_workers`. See the auto-generated API documentation for details.

**Returns:** Path to the input/output directory (string).

---

## `preprocess()`

Runs only preprocessing: validates rasters, creates tiles, and prepares metfiles. Use when you want to call the pipeline in stages.

```python
from solweig_gpu import preprocess

preprocess_dir = preprocess(
    base_path="/path/to/data",
    selected_date_str="2020-08-13",
    building_dsm_filename="Building_DSM.tif",
    dem_filename="DEM.tif",
    trees_filename="Trees.tif",
    landcover_filename=None,
    windcoeff_folder=None,     # folder/glob/raster with WindCoeff_dir*.tif, or None
    tile_size=1000,
    overlap=100,
    use_own_met=True,
    own_met_file="/path/to/met.txt",
    preprocess_dir=None,       # optional; default is base_path/processed_inputs
    use_uhi=True,              # ERA5 only: write diagnostic UHII into metfiles
)
```

**Returns:** Path to the preprocessing directory (string).

**Parameters:** Same as the corresponding part of `thermal_comfort` (base_path, selected_date_str, raster filenames, tile_size, overlap, met options), plus:

-   `windcoeff_folder` (str, optional): Wind coefficient input. Can be `None` (do not use wind coefficients), a folder containing `WindCoeff_dir*.tif`, a glob pattern such as `"WindCoeff_dir*.tif"`, or a single legacy wind coefficient raster. Relative paths are resolved against `base_path`. Generate the directional rasters with `build_wind_ext_coeff()`.
-   `use_uhi` (bool): If `True`, use UHI-aware ERA5 processing and write `UHI_CYCLE`/`uhii` into the generated metfiles; if `False`, write `uhii = 0.0`. Defaults to `True`.
-   `preprocess_dir` (str, optional): Overrides the default output directory `{base_path}/processed_inputs`.

---

## `run_walls_aspect()`

Computes wall heights and aspects for all tiles in the preprocessing directory. Call after `preprocess()`.

```python
from solweig_gpu import run_walls_aspect

run_walls_aspect(preprocess_dir)
```

**Parameters:**

-   `preprocess_dir` (str): Path returned by `preprocess()` (contains Building_DSM/, DEM/, Trees/, etc.).

**Returns:** None. Writes to `{preprocess_dir}/walls/` and `{preprocess_dir}/aspect/`.

---

## `calculate_svf()`

Calculates standalone Sky View Factor outputs for all raster tiles in a preprocessing directory. Call after `preprocess()` (the tiles must exist). Existing outputs are skipped unless `overwrite=True`.

```python
from solweig_gpu import calculate_svf

calculate_svf(
    base_path=preprocess_dir,  # the preprocessing directory containing Building_DSM/, DEM/, Trees/
    patch_option=2,
    overwrite=False,
)
```

**Parameters:**

-   `base_path` (str): Directory containing the tiled `Building_DSM/`, `DEM/`, and `Trees/` folders (i.e. the preprocessing directory).
-   `patch_option` (int): Sky patch discretization option passed to the SVF calculator. Defaults to `2`.
-   `overwrite` (bool): If `False`, tiles whose SVF outputs already exist are skipped. Defaults to `False`.

**Returns:** None. Writes `SkyViewFactor_{tile_key}.tif`, `svfs_{tile_key}.zip`, and `shadowmats_{tile_key}.npz` to `{base_path}/SVF/`.

---

## `run_utci_tiles()`

Runs the SOLWEIG/UTCI computation for tiles. Call after `preprocess()` and `run_walls_aspect()`.

```python
from solweig_gpu import run_utci_tiles

run_utci_tiles(
    base_path="/path/to/data",
    preprocess_dir=preprocess_dir,
    selected_date_str="2020-08-13",
    tile_keys=None,       # None = all tiles; or ["0_0", "1000_0"] for a subset
    save_tmrt=True,
    save_svf=False,
    save_kup=False,
    save_kdown=False,
    save_lup=False,
    save_ldown=False,
    save_shadow=False,
    save_wbgt=False,
    save_ta=False,
    save_wind=False,
)
```

**Parameters:**

-   `base_path` (str): Base directory; `output_folder/` is created under it.
-   `preprocess_dir` (str): Path returned by `preprocess()`.
-   `selected_date_str` (str): Simulation date `YYYY-MM-DD`.
-   `tile_keys` (list, optional): If `None`, process all tiles. If a list (e.g. `["0_0", "1000_0"]`), process only those tile keys.
-   `save_tmrt`, `save_svf`, `save_kup`, `save_kdown`, `save_lup`, `save_ldown`, `save_shadow`, `save_wbgt`, `save_ta`, `save_wind`: Same as in `thermal_comfort()`.

If directional wind coefficients were provided during preprocessing, the model selects the nearest 30° wind coefficient raster for each timestep based on the wind direction (`Wd`) column of the metfile.

**Returns:** None. Writes GeoTIFFs to `{base_path}/output_folder/{tile_key}/`.
