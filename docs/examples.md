# Examples (Using the sample data)

This section provides a collection of examples demonstrating how to use SOLWEIG-GPU across different scenarios.

Sample data is available in [Zenodo](https://doi.org/10.5281/zenodo.21081622)

## Example 0 (Optional): Download Input Data with `build_inputs`

New in Version 2: download and build the required input rasters and meteorological data for any location from near-globally available urban datasets. Google Earth Engine must be authenticated before this step.

```python
import os
from solweig_gpu import build_inputs

os.environ["EE_PROJECT"] = "your-gee-project-id"  # Your own GEE/GCP project ID

base_path = build_inputs(
    lat=30.27,
    lon=-97.74,
    city="Austin",
    km_buffer=2,        # km from the central lat-lon to set the download extent
    km_reduced_lat=1,
    km_reduced_lon=1,
    base_folder="/path/to/save/inputs",
    resolution=2,       # spatial resolution of the generated rasters in meters
)

print("SOLWEIG input folder:", base_path)
```

## Compute Direction-Based Wind Coefficients (Optional)

New in Version 2 (GLIDE-SOL): requires ERA5 data with the variable *forecast surface roughness* (`fsr`).

```python
from solweig_gpu import build_wind_ext_coeff

build_wind_ext_coeff(
    "/path/to/solweig/input",  # base path where input rasters are present
    "/path/to/era5",           # folder containing data_stream-oper_stepType-instant.nc
)
```

This writes `WindCoeff_dir000.tif` … `WindCoeff_dir330.tif` (every 30°) into the input folder. Pass this folder as `windcoeff_folder` to `preprocess()`, or simply use `ERA_5_z0_find=True` in `thermal_comfort()` to do this automatically.

## Example 1: Using WRF Data

This example shows how to run a simulation using meteorological data from WRF output files.

```python
from solweig_gpu import thermal_comfort

thermal_comfort(
    base_path='/path/to/input',
    selected_date_str='2020-08-13',
    building_dsm_filename='Building_DSM.tif',
    dem_filename='DEM.tif',
    trees_filename='Trees.tif',
    landcover_filename=None,
    tile_size=1000,
    overlap=100,
    use_own_met=False,
    start_time='2020-08-13 06:00:00',
    end_time='2020-08-14 05:00:00',
    data_source_type='wrfout',
    data_folder='/path/to/wrfout',
)
```
**See the interactive** [Jupyter notebook](notebooks/Example_wrfout.ipynb)

## Example 2: Using ERA5 Data

This example demonstrates how to use ERA5 reanalysis data for the simulation.

```python
from solweig_gpu import thermal_comfort

thermal_comfort(
    base_path='/path/to/input',
    selected_date_str='2020-08-13',
    building_dsm_filename='Building_DSM.tif',
    dem_filename='DEM.tif',
    trees_filename='Trees.tif',
    landcover_filename=None,
    tile_size=1000,
    overlap=100,
    use_own_met=False,
    start_time='2020-08-13 06:00:00',
    end_time='2020-08-13 23:00:00',
    data_source_type='ERA5',
    data_folder='/path/to/era5',
    ERA_5_z0_find=True,   # directional wind coefficients from ERA5 roughness (new in v2)
    use_uhi=True,         # diagnostic urban heat island intensity (ERA5 only, new in v2)
    save_wbgt=True,       # Wet Bulb Globe Temperature output (new in v2)
)
```
**See the interactive** [Jupyter notebook](notebooks/Example_ERA5.ipynb).

## How to install CDS API

ECMWF Climate Data Store (CDS) API key is required to download ERA5 data programmatically. To set-up API, please follow: <https://cds.climate.copernicus.eu/how-to-api> . Alternatively, the data can be downloaded directly from CDS: <https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview>

## You can download ERA5 as below
```python
import cdsapi

dataset = "reanalysis-era5-single-levels"
request = {
    "product_type": ["reanalysis"],
    "variable": [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "2m_dewpoint_temperature",
        "2m_temperature",
        "surface_pressure",
        "surface_solar_radiation_downwards",
        "surface_thermal_radiation_downwards",
        "forecast_surface_roughness" # required only if ERA_5_z0_find=True (directional wind coefficients)
    ],
    "year": ["2020"], # change to the desired year
    "month": ["08"], # change to the desired month
    "day": ["13", "14"], # change to the desired date
    "time": [
        "00:00", "01:00", "02:00",
        "03:00", "04:00", "05:00",
        "06:00", "07:00", "08:00",
        "09:00", "10:00", "11:00",
        "12:00", "13:00", "14:00",
        "15:00", "16:00", "17:00",
        "18:00", "19:00", "20:00",
        "21:00", "22:00", "23:00"
    ],
    "data_format": "netcdf",
    "download_format": "unarchived",
    "area": [31, -98, 29, -97] #change according to your location
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()
```

## Example 3: Using a Custom Meteorological File

This example shows how to use your own meteorological data in the UMEP text file format.

```python
from solweig_gpu import thermal_comfort

thermal_comfort(
    base_path='/path/to/input',
    selected_date_str='2020-08-13',
    building_dsm_filename='Building_DSM.tif',
    dem_filename='DEM.tif',
    trees_filename='Trees.tif',
    landcover_filename=None,
    tile_size=1000,
    overlap=100,
    use_own_met=True,
    own_met_file='/path/to/met.txt',
    ERA_5_z0_find=False,  # set True only if data_folder contains the ERA5 file data_stream-oper_stepType-instant.nc
    use_uhi=False,        # not recommended with user-provided meteorological files
)
```
**See the interactive** [Jupyter notebook](notebooks/Example_ownmetfile.ipynb)

## Example 4: Running the Pipeline in Stages (New in Version 2)

For finer control (e.g. running a subset of tiles, or reusing preprocessed data), run the four stages separately:

```python
from solweig_gpu import preprocess, run_walls_aspect, calculate_svf, run_utci_tiles

# Step 1: Preprocess and create inputs in the required format
preprocess_dir = preprocess(
    base_path="/path/to/solweig/input",
    selected_date_str="2020-08-13",
    building_dsm_filename="Building_DSM.tif",
    dem_filename="DEM.tif",
    trees_filename="Trees.tif",
    landcover_filename="Landuse.tif",          # None if land cover is not used
    windcoeff_folder="/path/to/solweig/input", # None if wind coefficients are not used
    tile_size=400,
    overlap=0,
    use_own_met=False,
    start_time="2020-08-13 06:00:00",
    end_time="2020-08-14 05:00:00",
    data_source_type="ERA5",
    data_folder="/path/to/era5",
    own_met_file=None,
    use_uhi=True,  # ERA5 only: diagnostic urban heat island intensity
)

# Step 2: Calculate wall height and aspect
run_walls_aspect(preprocess_dir)

# Step 3: Calculate the sky-view factor
calculate_svf(preprocess_dir, patch_option=2, overwrite=False)

# Step 4: Run the SOLWEIG-GPU model
run_utci_tiles(
    base_path="/path/to/solweig/input",
    preprocess_dir=preprocess_dir,
    selected_date_str="2020-08-13",
    tile_keys=None,   # or e.g. ["0_0", "400_0"] for a subset of tiles
    save_tmrt=True,
    save_wbgt=False,
)
```

