#SOLWEIG-GPU: GPU-accelerated SOLWEIG model for urban thermal comfort simulation
#Copyright (C) 2022–2025 Harsh Kamath and Naveen Sudharsan

#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.

#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#GNU General Public License for more details.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import numpy as np
from math import radians
from copy import deepcopy
from osgeo import gdal, osr
import datetime
import calendar
import scipy.ndimage.interpolation as sc
import torch
import torch.nn.functional as F
from scipy.ndimage import rotate
import time
from timezonefinder import TimezoneFinder
import pytz
import datetime
from .Tgmaps_v1 import Tgmaps_v1
from .sun_position import Solweig_2015a_metdata_noload
from .shadow import svf_calculator, create_patches
from .solweig import Solweig_2022a_calc, clearnessindex_2013b
from .calculate_utci import utci_calculator
from .calculate_wbgt import isobaric_wet_bulb_temperature_from_rh, black_globe_temperature
import os
import re
import zipfile
# from .preprocessor import ppr
from .walls_aspect import run_parallel_processing
gdal.UseExceptions()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

script_dir = os.path.dirname(__file__)
landcover_classes_path = os.path.join(script_dir, 'landcoverclasses_2016a.txt')

# Wall and ground emissivity and albedo
albedo_b = 0.2
albedo_g = 0.15
ewall = 0.9
eground = 0.95
absK = 0.7
absL = 0.95

# Standing position
Fside = 0.22
Fup = 0.06
Fcyl = 0.28

cyl = True
elvis = 0
usevegdem = 1
onlyglobal = 1

firstdayleaf = 97
lastdayleaf = 300
conifer_bool = False

#: Model/receptor parameter names ``run_utci_window`` accepts in its
#: ``model_parameters`` override mapping (incremental design tool, U-C
#: packet 2). Mirrors ``solweig_gpu.incremental.adapters.model_parameters.
#: PLUMBING_CLASSIFICATION``; kept a literal here so the physics module
#: never imports the editing framework (tests assert the two sets stay in
#: sync). Unknown names are refused by ``run_utci_window`` so a mistyped
#: override can never silently miss the physics.
_MODEL_PARAMETER_NAMES = frozenset({
    "albedo_b", "ewall", "absK", "absL", "Fside", "Fup", "Fcyl", "cyl",
    "elvis", "anisotropic_sky", "transVeg", "height",
    "firstdayleaf", "lastdayleaf",
})

def load_raster_to_tensor(dem_path):
    """
    Load a GeoTIFF raster file into a PyTorch tensor.
    
    Args:
        dem_path (str): Path to GeoTIFF file
    
    Returns:
        tuple: (tensor, dataset) where:
            - tensor: PyTorch tensor on GPU/CPU with raster data
            - dataset: GDAL dataset object (for accessing metadata)
    """
    dataset = gdal.Open(dem_path)
    band = dataset.GetRasterBand(1)
    array = band.ReadAsArray().astype(np.float32)
    return torch.tensor(array, device=device), dataset

def extract_key(filename, is_metfile=False):
    """
    Extract numerical key from filename for tile matching.
    
    Args:
        filename (str): Filename to parse
        is_metfile (bool): True if filename is a metfile, False if raster tile
    
    Returns:
        str: Extracted key (e.g., "0_0" from "Building_DSM_0_0.tif")
    """

    if is_metfile:
        # look for metfile_X_Y_DATE
        match = re.search(r'metfile_(\d+)_(\d+)_\d{4}-\d{2}-\d{2}', filename)
    else:
        # look for ..._X_Y.tif
        match = re.search(r'_(\d+)_(\d+)', filename)

    if match:
        return f"{match.group(1)}_{match.group(2)}"
    return None

# Function to list matching files in a directory
def get_matching_files(directory, extension):
    """
    Get sorted list of files with given extension from directory.
    
    Args:
        directory (str): Directory path to search
        extension (str): File extension to filter (e.g., '.tif')
    
    Returns:
        list: Sorted list of filenames matching extension
    """
    return sorted([f for f in os.listdir(directory) if f.endswith(extension)])

def map_files_by_key(directory, extension, is_metfile=False):
    """
    Create mapping of tile keys to filenames.
    
    Groups files by their tile coordinates (e.g., "0_0", "1000_0") to match
    corresponding raster tiles with their meteorological files.
    
    Args:
        directory (str): Directory containing files
        extension (str): File extension to filter
        is_metfile (bool): True if files are metfiles
    
    Returns:
        dict: Dictionary mapping keys to filenames
    """
    files = get_matching_files(directory, extension)
    mapping = {}
    for f in files:
        key = extract_key(f, is_metfile=is_metfile)
        if key:
            mapping[key] = os.path.join(directory, f)
    return mapping

def extract_number_from_filename(filename):
    """
    Extract tile number from Building_DSM filename.
    
    Args:
        filename (str): Filename in format "Building_DSM_X_Y.tif"
    
    Returns:
        str: Extracted number portion (e.g., "0_0")
    """
    number = filename[13:-4] # change according to the naming of building DSM files
    return number

def map_windcoeff_files_by_key(directory, extension=".tif"):
    """
    Map directional wind coefficient tiles by tile key.

    Expected filenames:
      WindCoeff_dir000_0_0.tif
      WindCoeff_dir030_0_0.tif
      ...
      WindCoeff_dir330_0_0.tif

    Returns:
      {
        "0_0": {
            0:   "/.../WindCoeff_dir000_0_0.tif",
            30:  "/.../WindCoeff_dir030_0_0.tif",
            ...
            330: "/.../WindCoeff_dir330_0_0.tif",

        },
        "0_1000": {...}

      }
    """
    if not os.path.isdir(directory):
        return {}

    files = get_matching_files(directory, extension)
    mapping = {}

    pattern = re.compile(r"^WindCoeff_dir(\d{3})_(-?\d+)_(-?\d+)\.tif$")

    for f in files:
        m = pattern.match(f)
        if not m:
            continue

        direction = int(m.group(1)) % 360
        key = f"{m.group(2)}_{m.group(3)}"

        mapping.setdefault(key, {})[direction] = os.path.join(directory, f)

    return mapping
    
def nearest_wind_dir_30(wd):
    """
    Round meteorological wind direction to nearest 30-degree bin.

    wd is wind FROM direction:
      0=N, 90=E, 180=S, 270=W
    """
    if not np.isfinite(wd) or wd < 0:
        return None

    return int((np.floor(((wd % 360.0) + 15.0) / 30.0) * 30.0) % 360.0)

def _svf_cache_paths_from_building_dsm(building_dsm_path: str, number: str):
    """
    Infer base_path/SVF cache paths from the Building_DSM tile path.

    Example:
        base_path/Building_DSM/Building_DSM_0_0.tif

    Gives:
        base_path/SVF/SkyViewFactor_0_0.tif
        base_path/SVF/svfs_0_0.zip
        base_path/SVF/shadowmats_0_0.npz
    """
    building_dsm_dir = os.path.dirname(building_dsm_path)
    base_path = os.path.dirname(building_dsm_dir)
    svf_dir = os.path.join(base_path, "SVF")

    svftotal_path = os.path.join(svf_dir, f"SkyViewFactor_{number}.tif")
    zip_path = os.path.join(svf_dir, f"svfs_{number}.zip")
    npz_path = os.path.join(svf_dir, f"shadowmats_{number}.npz")

    return base_path, svf_dir, svftotal_path, zip_path, npz_path

def _geotransforms_equal(gt_a, gt_b, tol: float = 1e-6) -> bool:
    """True when two GDAL geotransforms agree within ``tol`` (map units)."""
    return (
        len(gt_a) == 6
        and len(gt_b) == 6
        and all(abs(a - b) <= tol for a, b in zip(gt_a, gt_b))
    )


def _svf_cache_extent_matches(building_dsm_path: str, svftotal_path: str, zip_path: str) -> bool:
    """
    Check whether cached SVF rasters share the Building_DSM tile's extent.

    The SVF cache is keyed by tile number only, so a cache produced for a
    different window of the source mosaic (e.g. after re-staging the subset)
    can otherwise be silently reused. SVF rasters are stamped with the
    geotransform of the DSM they were computed from, so comparing extents
    catches stale/foreign caches before they contaminate results.

    Corrupt-but-present files (truncated tif, damaged zip) raise under
    gdal.UseExceptions(); they must degrade to "cache unusable" so the
    wrapper recomputes, never crash the run.
    """
    try:
        ds = gdal.Open(building_dsm_path)
        if ds is None:
            return False
        ds_gt = ds.GetGeoTransform()
        ds_size = (ds.RasterXSize, ds.RasterYSize)
        ds = None

        for raster_path in (svftotal_path, f"/vsizip/{zip_path}/svf.tif"):
            cache_ds = gdal.Open(raster_path)
            if cache_ds is None:
                return False
            matches = (
                cache_ds.RasterXSize == ds_size[0]
                and cache_ds.RasterYSize == ds_size[1]
                and _geotransforms_equal(cache_ds.GetGeoTransform(), ds_gt)
            )
            cache_ds = None
            if not matches:
                return False
    except RuntimeError:
        return False
    return True


def _svf_cache_exists(building_dsm_path: str, number: str) -> bool:
    """
    Check whether all standalone SVF cache files exist AND match the tile extent.
    """
    _, _, svftotal_path, zip_path, npz_path = _svf_cache_paths_from_building_dsm(
        building_dsm_path,
        number,
    )

    return (
        os.path.isfile(svftotal_path)
        and os.path.isfile(zip_path)
        and os.path.isfile(npz_path)
        and _svf_cache_extent_matches(building_dsm_path, svftotal_path, zip_path)
    )

def _load_raster_to_tensor_from_path(raster_path: str):
    """
    Load a normal GeoTIFF path into a GPU/CPU tensor.
    """
    ds = gdal.Open(raster_path)
    if ds is None:
        raise FileNotFoundError(f"Could not open raster: {raster_path}")

    arr = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    ds = None

    return torch.tensor(arr, device=device)


def _load_raster_to_tensor_from_zip(zip_path: str, internal_tif_name: str):
    """
    Load a GeoTIFF inside svfs_*.zip directly using GDAL /vsizip/.
    """
    vsi_path = f"/vsizip/{zip_path}/{internal_tif_name}"

    ds = gdal.Open(vsi_path)
    if ds is None:
        raise FileNotFoundError(
            f"Could not open {internal_tif_name} inside ZIP: {zip_path}"
        )

    arr = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    ds = None

    return torch.tensor(arr, device=device)


def load_cached_svf_outputs(building_dsm_path: str, number: str):
    """
    Load cached standalone SVF outputs from base_path/SVF onto GPU.

    Expected files:
        base_path/SVF/SkyViewFactor_<number>.tif
        base_path/SVF/svfs_<number>.zip
        base_path/SVF/shadowmats_<number>.npz

    Raises ValueError when the cached rasters' extent does not match the
    Building_DSM tile (stale/foreign cache; recompute SVF or restore the
    matching cache).
    """
    _, _, svftotal_path, zip_path, npz_path = _svf_cache_paths_from_building_dsm(
        building_dsm_path,
        number,
    )

    if not _svf_cache_extent_matches(building_dsm_path, svftotal_path, zip_path):
        raise ValueError(
            f"Cached SVF outputs for tile {number} do not match the extent of "
            f"{building_dsm_path}; refusing to load a stale/foreign SVF cache. "
            "Recompute SVF for this tile or restore the matching cache."
        )

    svf = _load_raster_to_tensor_from_zip(zip_path, "svf.tif")
    svfE = _load_raster_to_tensor_from_zip(zip_path, "svfE.tif")
    svfS = _load_raster_to_tensor_from_zip(zip_path, "svfS.tif")
    svfW = _load_raster_to_tensor_from_zip(zip_path, "svfW.tif")
    svfN = _load_raster_to_tensor_from_zip(zip_path, "svfN.tif")

    svfveg = _load_raster_to_tensor_from_zip(zip_path, "svfveg.tif")
    svfEveg = _load_raster_to_tensor_from_zip(zip_path, "svfEveg.tif")
    svfSveg = _load_raster_to_tensor_from_zip(zip_path, "svfSveg.tif")
    svfWveg = _load_raster_to_tensor_from_zip(zip_path, "svfWveg.tif")
    svfNveg = _load_raster_to_tensor_from_zip(zip_path, "svfNveg.tif")

    svfaveg = _load_raster_to_tensor_from_zip(zip_path, "svfaveg.tif")
    svfEaveg = _load_raster_to_tensor_from_zip(zip_path, "svfEaveg.tif")
    svfSaveg = _load_raster_to_tensor_from_zip(zip_path, "svfSaveg.tif")
    svfWaveg = _load_raster_to_tensor_from_zip(zip_path, "svfWaveg.tif")
    svfNaveg = _load_raster_to_tensor_from_zip(zip_path, "svfNaveg.tif")

    svftotal = _load_raster_to_tensor_from_path(svftotal_path)

    with np.load(npz_path) as npz:
        shmat = torch.tensor(npz["shadowmat"].astype(np.float32), device=device)
        vegshmat = torch.tensor(npz["vegshadowmat"].astype(np.float32), device=device)
        vbshvegshmat = torch.tensor(npz["vbshmat"].astype(np.float32), device=device)

    return (
        svf, svfaveg, svfE, svfEaveg, svfEveg, svfN, svfNaveg, svfNveg, svfS, svfSaveg, svfSveg, svfveg,
        svfW, svfWaveg, svfWveg, vegshmat, vbshvegshmat, shmat, svftotal,)

#: The cross-timestep thermal planes a warm start must carry (G2.1).
#: Exactly the state allocated at :614-619 below and threaded k -> k+1 by
#: ``TsWaveDelay_2015a`` — see the ``initial_state`` docstring.
_THERMAL_STATE_PLANES = (
    "Tgmap1", "Tgmap1E", "Tgmap1S", "Tgmap1W", "Tgmap1N", "TgOut1",
)

#: The carried scalars a warm start must carry (G2.1): ``CI`` and
#: ``Twater`` are recomputed only at midnight rows (:790-805), so a mid-day
#: warm start needs the values the cold run carried INTO that step.
_THERMAL_STATE_SCALARS = ("CI", "firstdaytime", "timeadd", "Twater")


def _warm_thermal_planes(initial_state, rows, cols, device, time_start):
    """Validate/bridge an ``initial_state`` into the six working planes.

    The planes may arrive as float32 torch tensors or float32 ndarrays
    (``torch.from_numpy`` shares the bits — no dtype change, no reorder);
    any other dtype, a missing plane, or a plane not at THIS function's
    per-cell working extent ``(rows, cols)`` is refused with
    ``ValueError`` — never narrowed, never cropped to fit. A
    ``next_step`` that disagrees with ``time_start`` is a silent-science
    hazard and is refused the same way.
    """
    if not isinstance(initial_state, dict):
        raise ValueError(
            "initial_state must be a dict of thermal planes + scalars, got "
            f"{type(initial_state).__name__}"
        )
    recorded_step = initial_state.get("next_step")
    if recorded_step is not None and int(recorded_step) != int(time_start):
        raise ValueError(
            f"initial_state was captured at next_step={recorded_step!r} but "
            f"time_start={time_start!r}; a state may only resume exactly "
            "where it was captured — refusing rather than guessing"
        )
    planes = []
    for name in _THERMAL_STATE_PLANES:
        if name not in initial_state:
            raise ValueError(f"initial_state is missing the plane {name!r}")
        plane = initial_state[name]
        if isinstance(plane, torch.Tensor):
            if plane.dtype != torch.float32:
                raise ValueError(
                    f"initial_state plane {name!r} must be float32, got "
                    f"{plane.dtype!r}"
                )
            plane = plane.detach().to(device)
        elif isinstance(plane, np.ndarray):
            if plane.dtype != np.float32:
                raise ValueError(
                    f"initial_state plane {name!r} must be float32, got "
                    f"{plane.dtype!r}"
                )
            plane = torch.from_numpy(np.ascontiguousarray(plane)).to(device)
        else:
            raise ValueError(
                f"initial_state plane {name!r} must be a torch tensor or "
                f"ndarray, got {type(plane).__name__}"
            )
        if tuple(plane.shape) != (rows, cols):
            raise ValueError(
                f"initial_state plane {name!r} shape {tuple(plane.shape)} "
                f"does not match the working extent {(rows, cols)} — slice "
                "the full-tile state to the write window before the warm "
                "start"
            )
        planes.append(plane)
    for name in ("CI", "firstdaytime", "timeadd"):
        if name not in initial_state:
            raise ValueError(
                f"initial_state is missing the carried scalar {name!r}"
            )
    return tuple(planes)


def run_utci_window(
    *,
    a,
    temp1,
    temp2,
    walls,
    dirwalls,
    svf_bundle,
    met_file,
    altitude,
    azimuth,
    zen,
    jday,
    dectime,
    altmax,
    location,
    scale,
    amaxvalue=None,
    landcover_grid=None,
    lc_class=None,
    windcoeff=None,
    windcoeff_by_dir=None,
    time_start=0,
    time_stop=None,
    requested_variables=("utci",),
    save_wbgt=False,
    out_window=None,
    model_parameters=None,
    svf_bundle_cubes_cropped=False,
    initial_state=None,
    return_final_state=False,
):
    """Windowed SOLWEIG temporal core extracted from :func:`compute_utci`.

    Runs the full physics time loop over one raster window. All spatial
    arguments are tensors of identical (rows, cols); site-level scalars
    (``location``, ``scale``, ``altmax`` and the solar-geometry series)
    always describe the full site and are passed unchanged regardless of
    window extent. Temporal state always starts cold at ``time_start`` —
    that is the full cross-timestep state (Tgmap1 family, TgOut1, CI=1.0,
    Twater=[], firstdaytime=1., timeadd=0.), not just the ground-heat
    maps. The documented warm-up policy for incremental evaluation is to
    replay from timestep 0: with ``time_start > 0`` the first day's
    clearness index would not be recomputed (it is only derived on
    midnight rows) and ``Twater`` would stay empty.

    Args:
        a: Building DSM tensor (window).
        temp1: Vegetation-above-ground tensor (window); values < 0 clamped
            to 0, matching the original pipeline.
        temp2: DEM tensor (window).
        walls, dirwalls: Wall height / wall aspect tensors (window).
        svf_bundle: SVF tuple from ``svf_calculator`` /
            ``load_cached_svf_outputs`` cropped to the window:
            ``(svf, svfaveg, svfE, svfEaveg, svfEveg, svfN, svfNaveg,
            svfNveg, svfS, svfSaveg, svfSveg, svfveg, svfW, svfWaveg,
            svfWveg, vegshmat, vbshvegshmat, shmat, svftotal)``.
        met_file: Meteorological forcing array (rows = timesteps).
        altitude, azimuth, zen, jday, dectime, altmax: Full-site solar
            geometry series from ``Solweig_2015a_metdata_noload``.
        location: Site location dict (longitude, latitude, altitude).
        scale: Site raster scale (pixels per metre).
        amaxvalue: Optional full-domain ``max(building DSM, vegdem)``;
            derived from the supplied window arrays when omitted.
        landcover_grid: Cleaned land-cover tensor (window) or ``None`` to
            disable land cover.
        lc_class: Land-cover class table (required with ``landcover_grid``).
        windcoeff, windcoeff_by_dir: Wind coefficient tensor / directional
            dict cropped to the window, or ``None``.
        time_start, time_stop: Half-open timestep index range to evaluate.
        requested_variables: Subset of ``{"utci", "tmrt", "kup", "kdown",
            "lup", "ldown", "shadow", "ta", "wind"}``; only requested
            variables are allocated and returned. ``"utci"`` is always
            included.
        save_wbgt: Also compute and return the WBGT diagnostic.
        out_window: Optional ``(row0, row1, col0, col1)`` sub-window of the
            supplied spatial tensors. When given, the per-timestep shadow
            marches still run over the full (read) tensors — the march is a
            per-output-cell ray trace, so outputs inside ``out_window`` are
            bit-identical — while every per-cell computation and every
            returned array is cropped to ``out_window``. ``None`` keeps the
            exact pre-existing behaviour (window == tensors).
        model_parameters: Optional ``name -> value`` mapping of model/
            receptor parameter overrides (incremental design tool, U-C
            packet 2). The executor folds committed
            ``model_receptor_parameters`` edits into this mapping via
            :func:`solweig_gpu.incremental.adapters.model_parameters.
            kernel_arguments_from_deltas`; accepted names mirror
            ``PLUMBING_CLASSIFICATION`` (:data:`_MODEL_PARAMETER_NAMES`) and
            unknown names raise ``ValueError`` so a mistyped override can
            never silently miss the physics. Every absent name keeps the
            module/loop-local constant, so ``None`` (the default) is
            bit-identical to the pre-existing behaviour.
        initial_state: G2.1 warm start (default ``None`` = the exact cold
            behaviour above). A dict carrying the FULL cross-timestep
            thermal state as of ``time_start`` — the six float32 planes
            ``Tgmap1``/``Tgmap1E``/``Tgmap1S``/``Tgmap1W``/``Tgmap1N``/
            ``TgOut1`` (torch tensors or float32 ndarrays at THIS
            function's per-cell working extent, i.e. the write window when
            ``out_window`` is given) plus the scalars ``CI``,
            ``firstdaytime``, ``timeadd``, ``Twater`` (``None`` = not yet
            established, the pre-midnight ``[]``) and the ``next_step`` the
            state was captured at (must equal ``time_start`` — a mismatched
            pair is refused loudly, never guessed). The warm==cold parity
            fence (tests/test_incremental_warm_start.py) is what licenses
            this seam: ``warm(k..N, state@k)`` is bitwise
            ``cold(0..N)[k..N]`` because the carried state is exactly the
            loop's cross-timestep dependency (``timestepdec`` is derived
            per run, never carried).
        return_final_state: G2.1 state capture (default ``False`` = the
            exact previous return). When ``True`` the return is
            ``(outputs, final_state)`` where ``final_state`` is the same
            dict shape ``initial_state`` accepts, describing the state
            AFTER completing ``time_stop`` steps (planes as float32 torch
            tensors at the working extent — a windowed run's state is the
            write-window crop of the full-tile trajectory; only FULL-TILE
            states are checkpoint-persistable, windowed consumers slice).

    Returns:
        dict mapping variable name to ``numpy.ndarray`` of shape
        ``(time_stop - time_start, rows, cols)``; or, when
        ``return_final_state`` is set, ``(outputs, final_state)``.
    """
    valid_variables = {
        "utci", "tmrt", "kup", "kdown", "lup", "ldown",
        "shadow", "ta", "wind",
    }
    requested = set(requested_variables) | {"utci"}
    if not requested.issubset(valid_variables):
        unknown = sorted(requested - valid_variables)
        raise ValueError(f"unknown requested variables: {unknown}")
    if time_stop is None:
        time_stop = met_file.shape[0]
    if not 0 <= time_start < time_stop <= met_file.shape[0]:
        raise ValueError(
            f"invalid time range [{time_start}, {time_stop}) for "
            f"{met_file.shape[0]} forcing rows"
        )

    rows, cols = a.shape
    if out_window is not None:
        r0, r1, c0, c1 = out_window
        if not (0 <= r0 < r1 <= rows and 0 <= c0 < c1 <= cols):
            raise ValueError(
                f"out_window {out_window} outside tensor shape {(rows, cols)}"
            )
    if model_parameters is None:
        model_parameters = {}
    else:
        model_parameters = dict(model_parameters)
        unknown = sorted(set(model_parameters) - _MODEL_PARAMETER_NAMES)
        if unknown:
            raise ValueError(
                f"unknown model parameters {unknown}; accepted names are "
                f"{sorted(_MODEL_PARAMETER_NAMES)}"
            )

    (
        svf, svfaveg, svfE, svfEaveg, svfEveg, svfN, svfNaveg, svfNveg,
        svfS, svfSaveg, svfSveg, svfveg, svfW, svfWaveg, svfWveg,
        vegshmat, vbshvegshmat, shmat, svftotal,
    ) = svf_bundle

    if out_window is not None:
        _sl = (slice(out_window[0], out_window[1]),
               slice(out_window[2], out_window[3]))
        # Per-cell SVF terms: elementwise functions of these slices, so the
        # write-window values are bit-identical to the full-window run.
        svf, svfaveg, svfE, svfEaveg, svfEveg, svfN, svfNaveg, svfNveg, \
        svfS, svfSaveg, svfSveg, svfveg, svfW, svfWaveg, svfWveg = (
            t[_sl] for t in (svf, svfaveg, svfE, svfEaveg, svfEveg, svfN,
                             svfNaveg, svfNveg, svfS, svfSaveg, svfSveg,
                             svfveg, svfW, svfWaveg, svfWveg)
        )
        if not svf_bundle_cubes_cropped:
            # The patch cubes are cropped here unless the caller already
            # sliced them at the write extent (perf wave 1 STEP 2b: the
            # solver crops them at the memmap instead of materializing the
            # full read-extent cube first — same values, pure data
            # movement). Refused without an out_window because the flag
            # only makes sense against a known write extent.
            vegshmat = vegshmat[_sl]
            vbshvegshmat = vbshvegshmat[_sl]
            shmat = shmat[_sl]
    elif svf_bundle_cubes_cropped:
        raise ValueError(
            "svf_bundle_cubes_cropped=True requires out_window: the "
            "pre-cropped cube extent must match a declared write window"
        )

    temp1[temp1 < 0.] = 0.
    vegdem = temp1 + temp2
    vegdem2 = torch.add(temp1 * 0.25, temp2)
    bush = torch.logical_not(vegdem2 * vegdem) * vegdem
    vegdsm = temp1 + a
    vegdsm[vegdsm == a] = 0
    vegdsm2 = temp1 * 0.25 + a
    vegdsm2[vegdsm2 == a] = 0
    if amaxvalue is None:
        amaxvalue = torch.maximum(a.max(), vegdem.max())
    buildings = a - temp2
    buildings[buildings < 2.] = 1.
    buildings[buildings >= 2.] = 0.
    if windcoeff is None and windcoeff_by_dir is None:
        windcoeff = torch.ones((rows, cols), device=device)

    if out_window is not None:
        # buildings feeds the GVF ray walk (see Solweig_2022a_calc): it must
        # stay read-sized. The wind fields keep a read-sized copy too:
        # utci_calculator boolean-compacts its inputs and then applies
        # exp/fractional-pow, whose vectorised path depends on the
        # compacted length — the UTCI evaluation therefore has to run at
        # the read extent to stay bit-identical to the reference.
        windcoeff_read = windcoeff
        windcoeff_by_dir_read = windcoeff_by_dir
        if windcoeff is not None:
            windcoeff = windcoeff[_sl]
        elif windcoeff_by_dir is not None:
            windcoeff_by_dir = {
                k: v[_sl] for k, v in windcoeff_by_dir.items()
            }
        # rows/cols now describe the per-cell domain; the march and GVF
        # inputs (a, vegdsm/vegdsm2, bush, walls, dirwalls, buildings,
        # landcover) keep the read shape.
        read_rows, read_cols = rows, cols
        rows, cols = r1 - r0, c1 - c0
    else:
        windcoeff_read = None
        windcoeff_by_dir_read = None
    # Read-shaped (uncropped) so the UTCI valid-cell compaction matches the
    # reference run exactly; see the wind-field note above.
    valid_mask_read = (buildings == 1)
    valid_mask = valid_mask_read[_sl] if out_window is not None else valid_mask_read

    # Knight is the base of the no-landcover surface-property maps, which
    # feed the read-sized GVF walk; keep it at the read extent.
    Knight = torch.zeros(
        (read_rows, read_cols) if out_window is not None else (rows, cols),
        device=device,
    )
    if initial_state is None:
        Tgmap1 = torch.zeros((rows, cols), device=device)
        Tgmap1E = torch.zeros((rows, cols), device=device)
        Tgmap1S = torch.zeros((rows, cols), device=device)
        Tgmap1W = torch.zeros((rows, cols), device=device)
        Tgmap1N = torch.zeros((rows, cols), device=device)
        TgOut1 = torch.zeros((rows, cols), device=device)
    else:
        # G2.1 warm start: resume from the captured cross-timestep state
        # (validated/bridged above); the loop below runs unchanged.
        (
            Tgmap1, Tgmap1E, Tgmap1S, Tgmap1W, Tgmap1N, TgOut1,
        ) = _warm_thermal_planes(
            initial_state, rows, cols, device, time_start
        )

    if landcover_grid is not None:
        landcover = 1
        lcgrid_torch = landcover_grid
        lcgrid_np = lcgrid_torch.cpu().numpy()

        mask_invalid = (lcgrid_np < 1) | (lcgrid_np > 7)
        if mask_invalid.any():
            lcgrid_np[mask_invalid] = 6

        mask_vegetation = (lcgrid_np == 3) | (lcgrid_np == 4)
        if mask_vegetation.any():
            lcgrid_np[mask_vegetation] = 5

        (TgK_np, Tstart_np, alb_np, emis_np, TgK_wall_np, Tstart_wall_np,
         TmaxLST_np, TmaxLST_wall_np) = Tgmaps_v1(lcgrid_np, lc_class)

        TgK           = torch.from_numpy(TgK_np).to(device).float()
        Tstart        = torch.from_numpy(Tstart_np).to(device).float()
        alb_grid      = torch.from_numpy(alb_np).to(device).float()
        emis_grid     = torch.from_numpy(emis_np).to(device).float()
        TgK_wall = torch.as_tensor(TgK_wall_np, device=device).float()
        Tstart_wall = torch.as_tensor(Tstart_wall_np, device=device).float()
        TmaxLST = torch.as_tensor(TmaxLST_np, device=device).float()
        TmaxLST_wall = torch.as_tensor(TmaxLST_wall_np, device=device).float()
    else:
        landcover = 0
        TgK = Knight + 0.37
        Tstart = Knight - 3.41
        alb_grid = Knight + albedo_g
        emis_grid = Knight + eground
        TgK_wall = 0.37
        Tstart_wall = -3.41
        TmaxLST = 15.
        TmaxLST_wall = 15.

    # Model/receptor parameter overrides (U-C packet 2): every absent name
    # keeps its module/loop-local default, so an empty mapping (and the
    # default None) is bit-identical to the pre-change code. Each comment
    # cites the overridden constant's default value and source (module
    # constants by current line, loop-local constants by their pre-plumbing
    # line, matching PARAMETER_SPECS' default_source citations).
    transVeg = model_parameters.get("transVeg", 3. / 100.)  # loop-local default 0.03 (pre-plumbing :608)
    if landcover == 1:
        lcgrid = lcgrid_torch
    else:
        lcgrid = False
    # anisotropic_sky is DOUBLE-SITED (loop-local assignment here, kernel
    # argument at the Solweig_2022a_calc call below): this assignment is the
    # single source of the forwarded value, so overriding it here replaces
    # both sites (adapters.model_parameters.PLUMBING_CLASSIFICATION note).
    anisotropic_sky = model_parameters.get("anisotropic_sky", 1)  # loop-local default 1 (pre-plumbing :613)
    patch_option = 2
    # kernel-arg overrides (constants read at the Solweig_2022a_calc call):
    # defaults cited from the module constants at utci_process.py:50-63.
    albedo_b = model_parameters.get("albedo_b", 0.2)    # default 0.2 (utci_process.py:50)
    ewall = model_parameters.get("ewall", 0.9)          # default 0.9 (utci_process.py:52)
    absK = model_parameters.get("absK", 0.7)            # default 0.7 (utci_process.py:54)
    absL = model_parameters.get("absL", 0.95)           # default 0.95 (utci_process.py:55)
    Fside = model_parameters.get("Fside", 0.22)         # default 0.22 (utci_process.py:58)
    Fup = model_parameters.get("Fup", 0.06)             # default 0.06 (utci_process.py:59)
    Fcyl = model_parameters.get("Fcyl", 0.28)           # default 0.28 (utci_process.py:60)
    cyl = model_parameters.get("cyl", True)             # default True (utci_process.py:62)
    elvis = model_parameters.get("elvis", 0)            # default 0 (utci_process.py:63)
    DOY = torch.tensor(met_file[:, 1], device=device)
    hours = torch.tensor(met_file[:, 2], device=device)
    minu = torch.tensor(met_file[:, 3], device=device)
    Ta = torch.tensor(met_file[:, 11], device=device)
    RH = torch.tensor(met_file[:, 10], device=device)
    radG = torch.tensor(met_file[:, 14], device=device)
    radD = torch.tensor(met_file[:, 21], device=device)
    radI = torch.tensor(met_file[:, 22], device=device)
    P = torch.tensor(met_file[:, 12], device=device)
    Ws = torch.tensor(met_file[:, 9], device=device)

    if met_file.shape[1] > 23:
        Wdirection = torch.tensor(met_file[:, 23], device=device)
    else:
        Wdirection = torch.full((met_file.shape[0],), -999.0, device=device)

    if met_file.shape[1] > 24:
        uhii = torch.tensor(met_file[:, 24], device=device)
    else:
        uhii = torch.zeros(met_file.shape[0], device=device)

    if save_wbgt:
        Ta_np = (Ta + uhii).cpu().numpy()
        RH_np = RH.cpu().numpy()
        P_np = P.cpu().numpy()

        wbt_np = isobaric_wet_bulb_temperature_from_rh(p=P_np * 1000.0, T=Ta_np + 273.15, rh=RH_np, phase='liquid', method='Romps', limit=True)
        wbt = torch.tensor(wbt_np, device=device, dtype=Ta.dtype)
    # Prepare leafon based on vegetation type
    # Leaf-on window overrides: module defaults firstdayleaf=97,
    # lastdayleaf=300 (utci_process.py:67-68).
    firstdayleaf = model_parameters.get("firstdayleaf", 97)
    lastdayleaf = model_parameters.get("lastdayleaf", 300)
    if conifer_bool:
        leafon = torch.ones((1, DOY.shape[0]), device=device)
    else:
        leafon = torch.zeros((1, DOY.shape[0]), device=device)
        if firstdayleaf > lastdayleaf:
            leaf_bool = ((DOY > firstdayleaf) | (DOY < lastdayleaf))
        else:
            leaf_bool = ((DOY > firstdayleaf) & (DOY < lastdayleaf))
        leafon[0, leaf_bool] = 1
    psi = leafon * transVeg
    psi[leafon == 0] = 0.5
    if initial_state is None:
        Twater = []
    else:
        # G2.1: ``None`` is the not-yet-established spelling (the cold
        # run's pre-midnight ``[]``); a captured scalar resumes it.
        carried_water = initial_state.get("Twater")
        Twater = [] if carried_water is None else float(carried_water)
    height = model_parameters.get("height", 1.1)  # loop-local default 1.1 m (pre-plumbing :656)
    height = torch.tensor(height, device=device)
    #first = torch.round(torch.tensor(height, device=device))
    first = torch.round(height.clone().detach().to(device))
    if first == 0.:
        first = torch.tensor(1., device=device)
    second = torch.round(height * 20.)
    if len(Ta) == 1:
        timestepdec = 0
    else:
        timestepdec = dectime[1] - dectime[0]
    if initial_state is None:
        timeadd = 0.
        firstdaytime = 1.
    else:
        # G2.1: the carried day-cycle scalars resume exactly where the
        # cold run left them (midnight rows below recompute as usual).
        timeadd = float(initial_state["timeadd"])
        firstdaytime = float(initial_state["firstdaytime"])

    svfbuveg = svf - (1.0 - svfveg) * (1.0 - transVeg)
    asvf = torch.acos(torch.sqrt(svf))
    diffsh = torch.zeros((rows, cols, shmat.shape[2]), device=device)
    for i in range(shmat.shape[2]):
        diffsh[:, :, i] = shmat[:, :, i] - (1 - vegshmat[:, :, i]) * (1 - transVeg)

    if out_window is not None:
        # Patch masks used by define_patch_characteristics are pure
        # functions of the timestep-invariant shadow cubes; evaluate them
        # once instead of once per timestep (identical bits, far fewer
        # kernel launches).
        n_patches = shmat.shape[2]
        sky_masks = (
            torch.empty((n_patches, rows, cols), dtype=torch.bool, device=device),
            torch.empty((n_patches, rows, cols), dtype=torch.bool, device=device),
            torch.empty((n_patches, rows, cols), dtype=torch.bool, device=device),
            torch.empty((n_patches, rows, cols), dtype=torch.bool, device=device),
        )
        for i in range(n_patches):
            sky_masks[0][i] = (shmat[:, :, i] == 1) & (vegshmat[:, :, i] == 1)
            sky_masks[1][i] = (vegshmat[:, :, i] == 0) | (vbshvegshmat[:, :, i] == 0)
            sky_masks[2][i] = ((1 - shmat[:, :, i]) * vbshvegshmat[:, :, i]) == 1
            sky_masks[3][i] = (
                (shmat[:, :, i] == 0) | (vegshmat[:, :, i] == 0)
                | (vbshvegshmat[:, :, i] == 0)
            )
    else:
        sky_masks = None
    tmp = svf + svfveg - 1.0
    tmp[tmp < 0.0] = 0.0
    svfalfa = torch.asin(torch.exp(torch.log(1.0 - tmp) / 2.0))

    # Allocate only the requested output volumes (windowed, (n_t, rows, cols)).
    n_out = time_stop - time_start
    outputs = {
        name: np.empty((n_out, rows, cols), dtype=np.float32)
        for name in requested
    }
    if save_wbgt:
        outputs["wbgt"] = np.empty((n_out, rows, cols), dtype=np.float32)

    if initial_state is None:
        CI = 1.0
    else:
        # G2.1: the clearness index is recomputed only at midnight rows;
        # mid-day warm starts must resume the carried value.
        CI = float(initial_state["CI"])
    if out_window is not None:
        from solweig_gpu.solweig import shadowingfunction_wallheight_23

    for i in np.arange(time_start, time_stop):
        out_index = int(i - time_start)
        if landcover == 1:
            if ((dectime[i] - np.floor(dectime[i]))) == 0 or (i == 0):
                Ta_      = Ta.cpu().numpy()  # Added
                Twater = np.mean(Ta_[jday[0] == np.floor(dectime[i])])  # Added
        if (dectime[i] - np.floor(dectime[i])) == 0:
            daylines = np.where(np.floor(dectime) == dectime[i])
            if daylines.__len__() > 1:
                alt = altitude[0][daylines]
                alt2 = np.where(alt > 1)
                rise = alt2[0][0]
                [_, CI, _, _, _] = clearnessindex_2013b(zen[0, i + rise + 1], jday[0, i + rise + 1], Ta[i + rise + 1],
                                                        RH[i + rise + 1] / 100., radG[i + rise + 1], location, P[i + rise + 1])
                if (CI > 1.) or (CI == np.inf):
                    CI = 1.
            else:
                CI = 1.
        if out_window is not None:
            # March over the read window: the wall-height shadow trace is
            # per-output-cell, so outputs inside out_window are
            # bit-identical to a march the oracle would run with the same
            # read-window surroundings. The scalar conversions replicate
            # the ones Solweig_2022a_calc performs internally.
            # Solweig_2022a_calc consumes precomputed_shadows only inside
            # its daytime branch (altitude > 0); at night it sets shadow to
            # zeros without touching the march outputs, so the march is
            # skipped there entirely.
            if altitude[0][i] > 0:
                vegsh_m, sh_m, _, wallsh_m, wallsun_m, wallshve_m, _, facesun_m = shadowingfunction_wallheight_23(
                    a, vegdsm, vegdsm2,
                    torch.tensor(azimuth[0][i]).item(), torch.tensor(altitude[0][i]).item(),
                    scale, amaxvalue.item(), bush, walls, dirwalls * np.pi / 180.)
                precomputed = (vegsh_m, sh_m, wallsh_m, wallsun_m,
                               wallshve_m, facesun_m)
            else:
                precomputed = None
            calc_walls = walls
            calc_dirwalls = dirwalls
            out_slice = _sl
        else:
            precomputed = None
            calc_walls = walls
            calc_dirwalls = dirwalls
            out_slice = None

        Tmrt, Kdown, Kup, Ldown, Lup, Tg, ea, esky, I0, CI, shadow, firstdaytime, timestepdec, timeadd, \
        Tgmap1, Tgmap1E, Tgmap1S, Tgmap1W, Tgmap1N, Keast, Ksouth, Kwest, Knorth, Least, Lsouth, Lwest, Lnorth, \
        KsideI, TgOut1, TgOut, radIout, radDout, Lside, Lsky_patch_characteristics, CI_Tg, CI_TgG, KsideD, dRad, Kside = Solweig_2022a_calc(
            i, a, scale, rows, cols, svf, svfN, svfW, svfE, svfS, svfveg, svfNveg, svfEveg, svfSveg, svfWveg, svfaveg, svfEaveg, svfSaveg, svfWaveg, svfNaveg, vegdsm, vegdsm2, albedo_b, absK, absL, ewall, Fside, Fup, Fcyl,
            altitude[0][i], azimuth[0][i], zen[0][i], jday[0][i], usevegdem, onlyglobal, buildings, location, psi[0][i], landcover, lcgrid, dectime[i], altmax[0][i], calc_dirwalls, calc_walls, cyl, elvis, Ta[i], RH[i], radG[i], radD[i], radI[i], P[i],
            amaxvalue, bush, Twater, TgK, Tstart, alb_grid, emis_grid, TgK_wall, Tstart_wall, TmaxLST, TmaxLST_wall, first, second, svfalfa, svfbuveg, firstdaytime, timeadd, timestepdec, Tgmap1, Tgmap1E, Tgmap1S, Tgmap1W, Tgmap1N,
            CI, TgOut1, diffsh, shmat, vegshmat, vbshvegshmat, anisotropic_sky, asvf, patch_option,
            precomputed_shadows=precomputed, out_slice=out_slice, sky_masks=sky_masks)
        # Create matrices for meteorological parameters for the current time step
        if out_window is not None:
            # Read-shaped met mats: utci_calculator's boolean compaction
            # plus exp/pow makes its bit-level output depend on the tensor
            # extent, so the UTCI evaluation must match the reference
            # extent. Values outside out_window are never stored.
            RH_mat = torch.zeros((read_rows, read_cols), device=device) + RH[i]
            Tmrt_mat = torch.zeros((read_rows, read_cols), device=device)
            Tmrt_mat[_sl] = Tmrt
            if precomputed is not None:
                shadow_read = sh_m - (1 - vegsh_m) * (1 - psi[0][i])
            else:
                # Night: Solweig_2022a_calc returns a zero shadow plane, so
                # the wbgt sun/shade condition below is True everywhere;
                # replicate that exactly instead of marching.
                shadow_read = torch.zeros(
                    (read_rows, read_cols), device=device
                )
        else:
            RH_mat = torch.zeros((rows, cols), device=device) + RH[i]
            Tmrt_mat = torch.zeros((rows, cols), device=device) + Tmrt
        if windcoeff_by_dir is not None:
            wd_i = float(Wdirection[i].detach().cpu().item())
            dir_bin = nearest_wind_dir_30(wd_i)

            if dir_bin is None:
                coeff_i = torch.ones((rows, cols), device=device)
                coeff_i_read = (
                    torch.ones((read_rows, read_cols), device=device)
                    if out_window is not None else coeff_i
                )
            else:
                coeff_i = windcoeff_by_dir[dir_bin]
                coeff_i_read = (
                    windcoeff_by_dir_read[dir_bin]
                    if out_window is not None else coeff_i
                )
        else:
            coeff_i = windcoeff
            coeff_i_read = windcoeff_read if out_window is not None else coeff_i

        va10m_mat = coeff_i * Ws[i]
        va10m_mat = torch.clamp(va10m_mat, min=0.15) #WGBT works for u > 0.15 m/s
        if out_window is not None:
            va10m_mat_read = torch.clamp(coeff_i_read * Ws[i], min=0.15)
            Ta_mat = torch.zeros((read_rows, read_cols), device=device) + Ta[i] + uhii[i]
        else:
            va10m_mat_read = va10m_mat
            Ta_mat = torch.zeros((rows, cols), device=device) + Ta[i] + uhii[i]

        if save_wbgt:
            coef = 6.3 / 0.46821
            if out_window is not None:
                hcg = coef*torch.pow(va10m_mat_read, 0.6)
                bgt_mat = black_globe_temperature(hcg, Tmrt_mat, Ta_mat, emissivity=0.95)
                cond = shadow_read < 0.1
            else:
                hcg = coef*torch.pow(va10m_mat, 0.6)
                bgt_mat = black_globe_temperature(hcg, Tmrt_mat, Ta_mat, emissivity=0.95)
                cond = shadow < 0.1
            wbgt_sun = 0.7 * wbt[i] + 0.3 * bgt_mat
            wbgt_shade = 0.7 * wbt[i] + 0.2 * bgt_mat + 0.1 * Ta_mat
            wbgt_mat = torch.where(cond, wbgt_sun, wbgt_shade)
            if out_window is not None:
                outputs["wbgt"][out_index] = wbgt_mat.cpu().numpy()[_sl]
            else:
                outputs["wbgt"][out_index] = wbgt_mat.cpu().numpy()

        if "utci" in outputs:
            if out_window is not None:
                UTCI_mat = utci_calculator(Ta_mat, RH_mat, Tmrt_mat, va10m_mat_read)
                UTCI = torch.full(UTCI_mat.shape, float('nan'), device=device)
                UTCI[valid_mask_read] = UTCI_mat[valid_mask_read]
                outputs["utci"][out_index] = UTCI.cpu().numpy()[_sl]
            else:
                UTCI_mat = utci_calculator(Ta_mat, RH_mat, Tmrt_mat, va10m_mat)
                UTCI = torch.full(UTCI_mat.shape, float('nan'), device=device)
                UTCI[valid_mask] = UTCI_mat[valid_mask]
                outputs["utci"][out_index] = UTCI.cpu().numpy()
        if "tmrt" in outputs:
            outputs["tmrt"][out_index] = Tmrt.cpu().numpy()
        if "kup" in outputs:
            outputs["kup"][out_index] = Kup.cpu().numpy()
        if "kdown" in outputs:
            outputs["kdown"][out_index] = Kdown.cpu().numpy()
        if "lup" in outputs:
            outputs["lup"][out_index] = Lup.cpu().numpy()
        if "ldown" in outputs:
            outputs["ldown"][out_index] = Ldown.cpu().numpy()
        if "shadow" in outputs:
            outputs["shadow"][out_index] = shadow.cpu().numpy()
        if "ta" in outputs:
            ta_plane = Ta_mat.cpu().numpy()
            outputs["ta"][out_index] = ta_plane[_sl] if out_window is not None else ta_plane
        if "wind" in outputs:
            wind_plane = va10m_mat_read.cpu().numpy()
            outputs["wind"][out_index] = wind_plane[_sl] if out_window is not None else wind_plane

    if return_final_state:
        # G2.1 state capture: exactly what ``initial_state`` accepts — the
        # six planes at the per-cell working extent (a windowed run's state
        # is the write-window crop of the full-tile trajectory; the chain
        # is elementwise, so only full-tile states are composable) plus the
        # carried scalars. ``Twater`` is ``None`` while un-established
        # (the cold run's pre-midnight ``[]``).
        final_state = {
            "Tgmap1": Tgmap1,
            "Tgmap1E": Tgmap1E,
            "Tgmap1S": Tgmap1S,
            "Tgmap1W": Tgmap1W,
            "Tgmap1N": Tgmap1N,
            "TgOut1": TgOut1,
            "CI": float(CI),
            "firstdaytime": float(firstdaytime),
            "timeadd": float(timeadd),
            "Twater": None if isinstance(Twater, list) else float(Twater),
            "next_step": int(time_stop),
        }
        return outputs, final_state
    return outputs


def recompute_utci_steps(met_file, time_indices, tmrt_planes, buildings_dsm, dem):
    """Recompute ONLY the utci planes at ``time_indices`` (r3a met fast path).

    Replicates :func:`run_utci_window`'s per-timestep comfort ops verbatim
    at the FULL-TILE extent (the ``out_window``-equals-tile case the solver
    runs at write=read=full), so the result is bitwise the value a full
    solve would publish for the same met table and the same published tmrt
    planes:

    - scene prep (:574-578): ``buildings = a - temp2`` thresholded at 2 m,
      ``windcoeff = ones`` (the incremental paths pass no wind-coefficient
      rasters);
    - met reads (:687-703): ``Ta`` col 11, ``RH`` col 10, ``Ws`` col 9 as
      ``torch.tensor`` (float64) series, ``uhii`` col 24 (zeros when the
      column is absent);
    - per timestep (:843-920): ``RH_mat = zeros + RH[i]``,
      ``Tmrt_mat[_sl] = Tmrt`` (full-tile slice), ``va10m = clamp(ones *
      Ws[i], 0.15)``, ``Ta_mat = zeros + Ta[i] + uhii[i]`` (left-assoc),
      ``utci_calculator(...)``, NaN outside ``buildings == 1``.

    The radiation/thermal state is NOT recomputed: each timestep's tmrt
    plane arrives from the scenario's published prior results (the
    ``(node, time)`` store), which is exactly what the affinity proof
    (edit_registry.MET_VARIABLE_AFFINITY: wind_speed/uhii enter only these
    lines) makes sound for utci-only met changes.

    Parameters
    ----------
    met_file : (T, >=25) float64 ndarray — the RESOLVED met table (overlay
        applied) the solver would read.
    time_indices : sequence of int — the changed timesteps to recompute.
    tmrt_planes : mapping int -> (rows, cols) float32 ndarray — the
        published tmrt plane for every requested timestep.
    buildings_dsm, dem : (rows, cols) float32 ndarrays — the cached scene
        rasters (solver passes ``a``/``temp2`` crops; full tile here).

    Returns
    -------
    (len(time_indices), rows, cols) float32 ndarray, NaN on invalid cells.

    Raises
    ------
    ValueError — any missing tmrt plane or shape/domain mismatch (the
        executor treats this as a fast-path refusal, never a silent
        fallback to stale data).
    """
    rows, cols = buildings_dsm.shape
    if dem.shape != buildings_dsm.shape:
        raise ValueError(
            f"dem shape {dem.shape} != buildings_dsm shape {buildings_dsm.shape}"
        )
    indices = tuple(int(i) for i in time_indices)
    if not indices:
        raise ValueError("time_indices must be non-empty")
    for i in indices:
        if not 0 <= i < met_file.shape[0]:
            raise ValueError(
                f"time index {i} outside the met table ({met_file.shape[0]} steps)"
            )
        plane = tmrt_planes.get(i)
        if plane is None:
            raise ValueError(f"no published tmrt plane for timestep {i}")
        if plane.shape != (rows, cols):
            raise ValueError(
                f"tmrt plane t={i} shape {plane.shape} != tile {(rows, cols)}"
            )

    device = torch.device("cpu")
    a = torch.from_numpy(
        np.array(buildings_dsm, dtype=np.float32, order="C", copy=True)
    )
    temp2 = torch.from_numpy(
        np.array(dem, dtype=np.float32, order="C", copy=True)
    )
    buildings = a - temp2                                    # :574
    buildings[buildings < 2.] = 1.                           # :575
    buildings[buildings >= 2.] = 0.                          # :576
    valid_mask_read = buildings == 1                         # :605
    windcoeff = torch.ones((rows, cols), device=device)      # :577-578

    Ta = torch.tensor(met_file[:, 11])                       # :687
    RH = torch.tensor(met_file[:, 10])                       # :688
    Ws = torch.tensor(met_file[:, 9])                        # :693
    if met_file.shape[1] > 24:                               # :700-703
        uhii = torch.tensor(met_file[:, 24])
    else:
        uhii = torch.zeros(met_file.shape[0])

    out = np.empty((len(indices), rows, cols), dtype=np.float32)
    for k, i in enumerate(indices):
        RH_mat = torch.zeros((rows, cols), device=device) + RH[i]          # :848
        Tmrt_mat = torch.zeros((rows, cols), device=device)                # :849
        Tmrt_mat[:] = torch.from_numpy(
            np.array(tmrt_planes[i], dtype=np.float32, order="C", copy=True)
        )                                                                  # :850
        va10m_mat_read = torch.clamp(windcoeff * Ws[i], min=0.15)          # :886
        Ta_mat = (                                                          # :887
            torch.zeros((rows, cols), device=device) + Ta[i] + uhii[i]
        )
        UTCI_mat = utci_calculator(Ta_mat, RH_mat, Tmrt_mat, va10m_mat_read)  # :912
        UTCI = torch.full(UTCI_mat.shape, float("nan"), device=device)        # :913
        UTCI[valid_mask_read] = UTCI_mat[valid_mask_read]                     # :914
        out[k] = UTCI.cpu().numpy()                                           # :915
    return out


def compute_utci(building_dsm_path, tree_path, dem_path, walls_path, aspect_path, landcover_path, windcoeff_path, met_file,
                output_path,number,selected_date_str,save_tmrt=False,save_svf=False, save_kup=False,save_kdown=False,save_lup=False,
                save_ldown=False,save_shadow=False,save_wbgt=False,save_ta=False,save_wind=False):
    """
    Compute UTCI and related thermal comfort outputs for a single tile.
    
    This is the main computation function that integrates shadow modeling, radiation
    calculations, and UTCI computation for urban microclimate analysis.
    
    Args:
        building_dsm_path (str): Path to Building DSM raster
        tree_path (str): Path to tree/vegetation DSM raster
        dem_path (str): Path to Digital Elevation Model raster
        walls_path (str): Path to wall height raster
        aspect_path (str): Path to wall aspect raster  
        landcover_path (str): Path to land cover raster (can be None)
        windcoeff_path (str): Path to wind coefficient raster (can be None)
        met_file (str): Path to meteorological forcing file
        output_path (str): Directory for saving output rasters
        number (str): Tile identifier (e.g., "0_0")
        selected_date_str (str): Date string (YYYY-MM-DD)
        save_tmrt (bool): Save mean radiant temperature output
        save_svf (bool): Save sky view factor output
        save_kup (bool): Save upward shortwave radiation
        save_kdown (bool): Save downward shortwave radiation
        save_lup (bool): Save upward longwave radiation
        save_ldown (bool): Save downward longwave radiation
        save_shadow (bool): Save shadow maps
        save_ta (bool): Save diagnostic air temperature field
        save_wind (bool): Save diagnostic wind speed field

    Returns:
        None: Outputs are saved as GeoTIFF files in output_path
    
    Notes:
        - Automatically uses GPU if available
        - Outputs are multi-band rasters (one band per hour)
        - UTCI is always computed and saved
        - Other outputs are optional based on save_* flags
    """
    a, dataset = load_raster_to_tensor(building_dsm_path)
    temp1, dataset2 = load_raster_to_tensor(tree_path)
    temp2, dataset3 = load_raster_to_tensor(dem_path)
    walls, dataset4 = load_raster_to_tensor(walls_path)
    dirwalls, dataset5 = load_raster_to_tensor(aspect_path)
          
    windcoeff = None
    windcoeff_by_dir = None
    dataset6 = None
    dataset7 = None

    # windcoeff_path can now be:
    #   None              -> no wind coefficient
    #   str               -> legacy single WindCoeff tile
    #   dict[int, str]    -> directional wind coefficient tiles for this tile
    if isinstance(windcoeff_path, dict):
        expected_dirs = list(range(0, 360, 30))
        missing_dirs = [d for d in expected_dirs if d not in windcoeff_path]

        if missing_dirs:
            raise FileNotFoundError(
                f"Tile {number} is missing directional wind coefficient files: "
                + ", ".join(f"WindCoeff_dir{d:03d}_{number}.tif" for d in missing_dirs)
            )

        windcoeff_by_dir = {}

        for d in expected_dirs:
            coeff_tensor, _ = load_raster_to_tensor(windcoeff_path[d])

            if coeff_tensor.shape != a.shape:
                raise ValueError(
                    f"Wind coefficient raster shape {tuple(coeff_tensor.shape)} for direction {d:03d} "
                    f"does not match Building DSM shape {tuple(a.shape)}"
                )

            windcoeff_by_dir[d] = coeff_tensor

    elif windcoeff_path is not None:
        # Legacy single-raster mode
        windcoeff, dataset7 = load_raster_to_tensor(windcoeff_path)

        if windcoeff.shape != a.shape:
            raise ValueError(
                f"Wind coefficient raster shape {tuple(windcoeff.shape)} does not match "
                f"Building DSM shape {tuple(a.shape)}"
            )
 
    # Added
    landcover = 0
    lcgrid_torch = None
    lc_class = None

    if landcover_path is not None:
        landcover = 1
        lcgrid_torch, dataset6 = load_raster_to_tensor(landcover_path)
        lcgrid_np = lcgrid_torch.cpu().numpy()
        #lcgrid_np = lcgrid_np.astype(int)

        mask_invalid = (lcgrid_np < 1) | (lcgrid_np > 7)
        if mask_invalid.any():
            print("Warning: land-cover grid contains values outside 1-7. "
                "Invalid cells are set to 6 (bare soil). ")
            lcgrid_np[mask_invalid] = 6

        mask_vegetation = (lcgrid_np == 3) | (lcgrid_np == 4)
        if mask_vegetation.any():
            print("Attention!",
                  "The land cover grid includes values (deciduous and/or conifer) not appropriate for the SOLWEIG-formatted land cover grid (should not include 3 or 4). "
                  "Land cover under the vegetation is required. "
                  "Setting the invalid landcover types to grass.")
            lcgrid_np[mask_vegetation] = 5

        # Re-wrap so cleaned values are carried into the windowed core on any
        # device (on CPU tensors .numpy() already aliases, so this is a no-op).
        lcgrid_torch = torch.as_tensor(
            lcgrid_np, device=lcgrid_torch.device, dtype=lcgrid_torch.dtype
        )

        with open(landcover_classes_path) as f:
            lines = f.readlines()[1:]                            # skip header line
        lc_class = np.empty((len(lines), 6), dtype=float)
        for i, ln in enumerate(lines):
            lc_class[i, :] = [float(x) for x in ln.split()[1:]]  # cols 1-6
    # Added
    
    base_date = datetime.datetime.strptime(selected_date_str, "%Y-%m-%d")
    rows, cols = a.shape
    geotransform = dataset.GetGeoTransform()
    scale = 1 / geotransform[1]
    projection_wkt = dataset.GetProjection()
    old_cs = osr.SpatialReference()
    old_cs.ImportFromWkt(projection_wkt) 
    old_cs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    wgs84_wkt = """GEOGCS["WGS 84",
        DATUM["WGS_1984",
            SPHEROID["WGS 84",6378137,298.257223563,
                AUTHORITY["EPSG","7030"]],
            AUTHORITY["EPSG","6326"]],
        PRIMEM["Greenwich",0,
            AUTHORITY["EPSG","8901"]],
        UNIT["degree",0.01745329251994328,
            AUTHORITY["EPSG","9122"]],
        AUTHORITY["EPSG","4326"]]"""
    new_cs = osr.SpatialReference()
    new_cs.ImportFromWkt(wgs84_wkt)
    new_cs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    transform = osr.CoordinateTransformation(old_cs, new_cs)
    widthx = dataset.RasterXSize
    heightx = dataset.RasterYSize
    geotransform = dataset.GetGeoTransform()
    #minx = geotransform[0]
    #miny = geotransform[3] + widthx * geotransform[4] + heightx * geotransform[5]
    #lonlat = transform.TransformPoint(minx, miny)
    #gdalver = float(gdal.__version__[0])
    #if gdalver == 3.:
    #    lon = lonlat[1]  # changed to gdal 3
    #    lat = lonlat[0]  # changed to gdal 3
    #else:
    #    lon = lonlat[0]  # changed to gdal 2
    #    lat = lonlat[1]  # changed to gdal 2
    centre_x = geotransform[0] + geotransform[1] * widthx  / 2.0
    centre_y = geotransform[3] + geotransform[5] * heightx / 2.0
    lon, lat = transform.TransformPoint(centre_x, centre_y)[:2]
    alt = torch.median(temp2)
    alt = alt.cpu().item()
    if alt > 0:
        alt = 3.
    location = {'longitude': lon, 'latitude': lat, 'altitude': alt}
    # After computing lat and lon
    tf = TimezoneFinder()
    timezone_name = tf.timezone_at(lat=lat, lng=lon) or "UTC"
    local_tz = pytz.timezone(timezone_name)
    # Use a sample date (today or specific) to get current UTC offset
    local_dt = local_tz.localize(base_date)
    utc = local_dt.utcoffset().total_seconds() / 3600
    print(f"[INFO] Timezone: {timezone_name}, UTC offset: {utc} hours")
    YYYY, altitude, azimuth, zen, jday, leafon, dectime, altmax = Solweig_2015a_metdata_noload(met_file, location, utc)
    temp1[temp1 < 0.] = 0.
    vegdem = temp1 + temp2
    vegdem2 = torch.add(temp1 * 0.25, temp2)
    bush = torch.logical_not(vegdem2 * vegdem) * vegdem
    vegdsm = temp1 + a
    vegdsm[vegdsm == a] = 0
    vegdsm2 = temp1 * 0.25 + a
    vegdsm2[vegdsm2 == a] = 0
    amaxvalue = torch.maximum(a.max(), vegdem.max())

    start_time = time.time()
    # Calculate SVF and related parameters (remains unchanged)
    patch_option = 2

    base_path, svf_cache_dir, svftotal_cache_path, svf_zip_path, svf_npz_path = (_svf_cache_paths_from_building_dsm(building_dsm_path, number))
    
    svf_cache_available = _svf_cache_exists(building_dsm_path, number)
    
    if svf_cache_available:
        print(f"[INFO] Loading cached SVF outputs for tile {number} from {svf_cache_dir}")

        (
        svf, svfaveg, svfE, svfEaveg, svfEveg, svfN, svfNaveg, svfNveg, svfS, svfSaveg, svfSveg,
        svfveg, svfW, svfWaveg, svfWveg, vegshmat, vbshvegshmat, shmat, svftotal,) = load_cached_svf_outputs(building_dsm_path, number)

    else:
        print(f"[INFO] Cached SVF outputs not found for tile {number}. Calculating SVF.")

        (
        svf, svfaveg, svfE, svfEaveg, svfEveg, svfN, svfNaveg, svfNveg, svfS, svfSaveg, svfSveg,
        svfveg, svfW, svfWaveg, svfWveg, vegshmat, vbshvegshmat, shmat, svftotal,) = svf_calculator(
        patch_option, amaxvalue, a, vegdsm, vegdsm2, bush, scale, save_rasters=True,
        output_dir=svf_cache_dir, number=number, gdal_dsm=dataset,)
    
    svf_bundle = (
        svf, svfaveg, svfE, svfEaveg, svfEveg, svfN, svfNaveg, svfNveg, svfS, svfSaveg, svfSveg,
        svfveg, svfW, svfWaveg, svfWveg, vegshmat, vbshvegshmat, shmat, svftotal,)

    requested_variables = ["utci"]
    if save_tmrt:
        requested_variables.append("tmrt")
    if save_kup:
        requested_variables.append("kup")
    if save_kdown:
        requested_variables.append("kdown")
    if save_lup:
        requested_variables.append("lup")
    if save_ldown:
        requested_variables.append("ldown")
    if save_shadow:
        requested_variables.append("shadow")
    if save_ta:
        requested_variables.append("ta")
    if save_wind:
        requested_variables.append("wind")

    outputs = run_utci_window(
        a=a,
        temp1=temp1,
        temp2=temp2,
        walls=walls,
        dirwalls=dirwalls,
        svf_bundle=svf_bundle,
        met_file=met_file,
        altitude=altitude,
        azimuth=azimuth,
        zen=zen,
        jday=jday,
        dectime=dectime,
        altmax=altmax,
        location=location,
        scale=scale,
        amaxvalue=amaxvalue,
        landcover_grid=lcgrid_torch,
        lc_class=lc_class,
        windcoeff=windcoeff,
        windcoeff_by_dir=windcoeff_by_dir,
        requested_variables=tuple(requested_variables),
        save_wbgt=save_wbgt,
    )

    UTCI_all  = outputs["utci"]
    TMRT_all  = outputs.get("tmrt")
    Kup_all   = outputs.get("kup")
    Kdown_all = outputs.get("kdown")
    Lup_all   = outputs.get("lup")
    Ldown_all = outputs.get("ldown")
    Shadow_all= outputs.get("shadow")
    wbgt_all  = outputs.get("wbgt")
    Ta_all    = outputs.get("ta")
    Wind_all  = outputs.get("wind")

    hours = torch.tensor(met_file[:, 2], device=device)
    minu = torch.tensor(met_file[:, 3], device=device)

    # Write a multi-band GeoTIFF for UTCI (each band corresponds to one time step)
    driver = gdal.GetDriverByName('GTiff')
    out_file_path = os.path.join(output_path, f'UTCI_{number}.tif')
    num_bands = UTCI_all.shape[0]
    out_dataset = driver.Create(out_file_path, cols, rows, num_bands, gdal.GDT_Float32)
    out_dataset.SetGeoTransform(dataset.GetGeoTransform())
    out_dataset.SetProjection(dataset.GetProjection())
    for band in range(num_bands):
        out_band = out_dataset.GetRasterBand(band + 1)
        out_band.WriteArray(UTCI_all[band])
        out_band.FlushCache()
        hour = int(hours[band].cpu().item())
        minute = int(minu[band].cpu().item())
        timestamp = base_date.replace(hour=hour, minute=minute).isoformat()
        out_band.SetMetadata({'Time': timestamp})
    out_dataset = None
    # Optionally, you can similarly write TMRT to a single multi-band file:
    if save_tmrt:
        out_file_path_op = os.path.join(output_path, f'TMRT_{number}.tif')
        num_bands_op = TMRT_all.shape[0]
        out_dataset_op = driver.Create(out_file_path_op, cols, rows, num_bands_op, gdal.GDT_Float32)
        out_dataset_op.SetGeoTransform(dataset.GetGeoTransform())
        out_dataset_op.SetProjection(dataset.GetProjection())
        for band in range(num_bands_op):
            out_band = out_dataset_op.GetRasterBand(band + 1)
            out_band.WriteArray(TMRT_all[band])
            out_band.FlushCache()
            hour = int(hours[band].cpu().item())
            minute = int(minu[band].cpu().item())
            timestamp = base_date.replace(hour=hour, minute=minute).isoformat()
            out_band.SetMetadata({'Time': timestamp})
        out_dataset_op = None

    if save_svf and not svf_cache_available:
        out_file_path_op = os.path.join(output_path, f"SVF_{number}.tif")
        SVF = svftotal.detach().cpu().numpy().astype(np.float32)

        out_dataset_op = driver.Create(
            out_file_path_op,
            cols,
            rows,
            1,
            gdal.GDT_Float32,
        )
        out_dataset_op.SetGeoTransform(dataset.GetGeoTransform())
        out_dataset_op.SetProjection(dataset.GetProjection())

        out_band = out_dataset_op.GetRasterBand(1)
        out_band.WriteArray(SVF)
        out_band.FlushCache()

        out_dataset_op = None
    elif save_svf and svf_cache_available:
        print(
            f"[INFO] SVF cache already exists for tile {number}; "
            f"skipping duplicate SVF_{number}.tif write."
        )
                    
    if save_kup:
        out_file_path_op = os.path.join(output_path, f'Kup_{number}.tif')
        num_bands_op = Kup_all.shape[0]
        out_dataset_op = driver.Create(out_file_path_op, cols, rows, num_bands_op, gdal.GDT_Float32)
        out_dataset_op.SetGeoTransform(dataset.GetGeoTransform())
        out_dataset_op.SetProjection(dataset.GetProjection())
        for band in range(num_bands_op):
            out_band = out_dataset_op.GetRasterBand(band + 1)
            out_band.WriteArray(Kup_all[band])
            out_band.FlushCache()
            hour = int(hours[band].cpu().item())
            minute = int(minu[band].cpu().item())
            timestamp = base_date.replace(hour=hour, minute=minute).isoformat()
            out_band.SetMetadata({'Time': timestamp})
        out_dataset_op = None
    if save_kdown:
        out_file_path_op = os.path.join(output_path, f'Kdown_{number}.tif')
        num_bands_op = Kdown_all.shape[0]
        out_dataset_op = driver.Create(out_file_path_op, cols, rows, num_bands_op, gdal.GDT_Float32)
        out_dataset_op.SetGeoTransform(dataset.GetGeoTransform())
        out_dataset_op.SetProjection(dataset.GetProjection())
        for band in range(num_bands_op):
            out_band = out_dataset_op.GetRasterBand(band + 1)
            out_band.WriteArray(Kdown_all[band])
            out_band.FlushCache()
            hour = int(hours[band].cpu().item())
            minute = int(minu[band].cpu().item())
            timestamp = base_date.replace(hour=hour, minute=minute).isoformat()
            out_band.SetMetadata({'Time': timestamp})
        out_dataset_op = None
    if save_lup:
        out_file_path_op = os.path.join(output_path, f'Lup_{number}.tif')
        num_bands_op = Lup_all.shape[0]
        out_dataset_op = driver.Create(out_file_path_op, cols, rows, num_bands_op, gdal.GDT_Float32)
        out_dataset_op.SetGeoTransform(dataset.GetGeoTransform())
        out_dataset_op.SetProjection(dataset.GetProjection())
        for band in range(num_bands_op):
            out_band = out_dataset_op.GetRasterBand(band + 1)
            out_band.WriteArray(Lup_all[band])
            out_band.FlushCache()
            hour = int(hours[band].cpu().item())
            minute = int(minu[band].cpu().item())
            timestamp = base_date.replace(hour=hour, minute=minute).isoformat()
            out_band.SetMetadata({'Time': timestamp})
        out_dataset_op = None
    if save_ldown:
        out_file_path_op = os.path.join(output_path, f'Ldown_{number}.tif')
        num_bands_op = Ldown_all.shape[0]
        out_dataset_op = driver.Create(out_file_path_op, cols, rows, num_bands_op, gdal.GDT_Float32)
        out_dataset_op.SetGeoTransform(dataset.GetGeoTransform())
        out_dataset_op.SetProjection(dataset.GetProjection())
        for band in range(num_bands_op):
            out_band = out_dataset_op.GetRasterBand(band + 1)
            out_band.WriteArray(Ldown_all[band])
            out_band.FlushCache()
            hour = int(hours[band].cpu().item())
            minute = int(minu[band].cpu().item())
            timestamp = base_date.replace(hour=hour, minute=minute).isoformat()
            out_band.SetMetadata({'Time': timestamp})
        out_dataset_op = None
    if save_shadow:
        out_file_path_op = os.path.join(output_path, f'Shadow_{number}.tif')
        num_bands_op = Shadow_all.shape[0]
        out_dataset_op = driver.Create(out_file_path_op, cols, rows, num_bands_op, gdal.GDT_Float32)
        out_dataset_op.SetGeoTransform(dataset.GetGeoTransform())
        out_dataset_op.SetProjection(dataset.GetProjection())
        for band in range(num_bands_op):
            out_band = out_dataset_op.GetRasterBand(band + 1)
            out_band.WriteArray(Shadow_all[band])
            out_band.FlushCache()
            hour = int(hours[band].cpu().item())
            minute = int(minu[band].cpu().item())
            timestamp = base_date.replace(hour=hour, minute=minute).isoformat()
            out_band.SetMetadata({'Time': timestamp})
        out_dataset_op = None
    if save_wbgt:
        out_file_path_op = os.path.join(output_path, f'WBGT_{number}.tif')
        num_bands_op = wbgt_all.shape[0]
        out_dataset_op = driver.Create(out_file_path_op, cols, rows, num_bands_op, gdal.GDT_Float32)
        out_dataset_op.SetGeoTransform(dataset.GetGeoTransform())
        out_dataset_op.SetProjection(dataset.GetProjection())
        for band in range(num_bands_op):
            out_band = out_dataset_op.GetRasterBand(band + 1)
            out_band.WriteArray(wbgt_all[band])
            out_band.FlushCache()
            hour = int(hours[band].cpu().item())
            minute = int(minu[band].cpu().item())
            timestamp = base_date.replace(hour=hour, minute=minute).isoformat()
            out_band.SetMetadata({'Time': timestamp})
        out_dataset_op = None

    if save_ta:
        out_file_path_op = os.path.join(output_path, f'Ta_{number}.tif')
        num_bands_op = Ta_all.shape[0]
        out_dataset_op = driver.Create(out_file_path_op, cols, rows, num_bands_op, gdal.GDT_Float32)
        out_dataset_op.SetGeoTransform(dataset.GetGeoTransform())
        out_dataset_op.SetProjection(dataset.GetProjection())
        for band in range(num_bands_op):
            out_band = out_dataset_op.GetRasterBand(band + 1)
            out_band.WriteArray(Ta_all[band])
            out_band.FlushCache()
            hour = int(hours[band].cpu().item())
            minute = int(minu[band].cpu().item())
            timestamp = base_date.replace(hour=hour, minute=minute).isoformat()
            out_band.SetMetadata({'Time': timestamp})
        out_dataset_op = None

    if save_wind:
        out_file_path_op = os.path.join(output_path, f'Wind_{number}.tif')
        num_bands_op = Wind_all.shape[0]
        out_dataset_op = driver.Create(out_file_path_op, cols, rows, num_bands_op, gdal.GDT_Float32)
        out_dataset_op.SetGeoTransform(dataset.GetGeoTransform())
        out_dataset_op.SetProjection(dataset.GetProjection())
        for band in range(num_bands_op):
            out_band = out_dataset_op.GetRasterBand(band + 1)
            out_band.WriteArray(Wind_all[band])
            out_band.FlushCache()
            hour = int(hours[band].cpu().item())
            minute = int(minu[band].cpu().item())
            timestamp = base_date.replace(hour=hour, minute=minute).isoformat()
            out_band.SetMetadata({'Time': timestamp})
        out_dataset_op = None

    # Clean up datasets
    dataset = None
    dataset2 = None
    dataset3 = None
    dataset4 = None
    dataset5 = None
    dataset6 = None
    dataset7 = None
    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Time taken to execute tile {number}: {time_taken:.2f} seconds")


