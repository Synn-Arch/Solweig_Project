# SPDX-License-Identifier: GPL-3.0-only
"""Exact local evaluation for the incremental design-tool worker (Phase 5).

This module implements Stages 4, 7 and 8 of docs/incremental_design_tool/
incremental_algorithm.md on top of the P1-P4 building blocks:

* :class:`SiteForcing` / :func:`load_site_forcing` recompute the float64
  meteorological and solar-geometry series from the original met text file
  exactly the way the full-domain oracle (``compute_utci``) does. The cache
  stores those series as float32 for warm-path access; physics inputs must
  stay float64 end-to-end, so the worker never feeds cached float32 forcing
  into the solver.
* :func:`required_halo_pixels` derives a conservative read-window halo from
  a shadow occlusion bound (see the derivation in the module docs of
  :func:`_march_reach_pixels`); :func:`read_window_for_write_window` refines
  it to the exact targets-plus-occluders bounding box, and
  :func:`effective_march_amplitude` shortens the marches themselves with a
  bit-identical early exit.
* MEDIUM-2 invariant: the veg-SVF replay's per-patch march REGIME (one-step
  vs multi-step at ``shadow()``'s step-1 accumulator zero) is a function of
  the FULL-TILE composed scene only — never of the read window.  The replay
  amplitude is therefore computed from the full-tile ``scene.a`` /
  ``scene.vegdsm`` / ``scene.vegdsm2`` (the same expression the oracle and
  ``veg_svf_state._scene_amplitude`` use), not from the window crop — this
  is the SVF recompute only; the time-loop march keeps its window-RELATIVE
  amplitude (the W2 design, strictly shorter marches) and escalates to the
  absolute ``scene.amaxvalue`` solely when a timestep bands.  R6
  escalation: a patch whose first-step distance lands in
  ``(A_eff, A_abs]`` — one-step at the effective amplitude, multi-step
  under the oracle's absolute stop — marches at the ABSOLUTE
  ``scene.amaxvalue`` (the oracle's own stop) instead of refusing, so its
  executed step set is identical to the oracle's and the serve is
  bitwise-equal by construction; see :func:`_banded_patch_mask` and
  :class:`MarchRegimeError` (kept as the typed refusal home, currently
  not raised).
* :func:`window_svf_bundle` implements the Stage 7 static/dynamic SVF split:
  building-only SVF scalars and the building shadow cube are sliced from the
  P2 cache (scene-invariant, bitwise identical to the oracle's), while every
  vegetation-dependent term is recomputed over the read window by replaying
  the exact ``svf_calculator`` accumulation loop. A cached vegetation SVF can
  never be reused after a vegetation edit (see :class:`StaleSvfError`).
* :func:`solve_window` runs :func:`run_utci_window` over the read window with
  cold temporal state replayed from timestep 0 (Stage 8) and a
  window-relative march amplitude (bit-identical to the oracle's scene-wide
  stop, see :func:`effective_march_amplitude`) that escalates to the
  absolute ``scene.amaxvalue`` whenever a timestep's sun first step would
  band (see :func:`time_loop_band_present`).
* :func:`run_full_tile` is the standard full-domain path used both as the
  scientific oracle and as the safe fallback for unsafe local jobs: it stages
  a scratch prepared-site directory (fresh SVF, so no stale-cache reuse),
  calls ``compute_utci``, and reads the multi-band GeoTIFFs back. A scenario
  forcing overlay (u-c3) rides the :class:`SiteForcing` the worker/executor
  materialized through :func:`load_site_forcing`: ``compute_utci`` receives
  the RESOLVED table and the scratch site stages it, never the baseline met
  text (an overlay must never be dropped to ``met_path``). A scenario
  land-cover overlay (u-c4) rides the same paths through
  :func:`resolve_landcover_overlay`: the windowed solve materializes the
  resolved class grid between the cache read and the tensor conversion
  (the :func:`solve_window` overlay seam), and the full-tile scratch site
  stages the RESOLVED raster, never the baseline ``Landcover`` GeoTIFF.
"""

from __future__ import annotations

import datetime
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np
import torch
from osgeo import gdal, osr

from solweig_gpu.shadow import (
    annulus_weight,
    create_patches,
    shadow as shadow_fn,
    svf_calculator,
)
from solweig_gpu.sun_position import Solweig_2015a_metdata_noload
from solweig_gpu.utci_process import (
    compute_utci,
    landcover_classes_path,
    load_raster_to_tensor,
    run_utci_window,
)
from solweig_gpu.incremental.cache import SiteCache
from solweig_gpu.incremental.geometry import RasterGrid, RasterWindow
from solweig_gpu.incremental import march_router
from solweig_gpu.incremental.trees import TreeLayer
from solweig_gpu.incremental.adapters.landcover import LandCoverOverlay
from solweig_gpu.incremental.adapters.met_time import ForcingOverlay
from solweig_gpu.incremental.adapters.model_parameters import (
    check_parameter_value,
)
from solweig_gpu.incremental.edit_types import EditStateError

__all__ = [
    "GVF_MARCH_METERS",
    "HALO_SAFETY_PIXELS",
    "gvf_march_pixels",
    "SiteForcing",
    "SolverInputError",
    "StaleSvfError",
    "load_site_forcing",
    "resolve_landcover_overlay",
    "required_halo_pixels",
    "read_window_for_write_window",
    "effective_march_amplitude",
    "window_svf_bundle",
    "solve_window",
    "solve_with_core_adapter",
    "run_full_tile",
    "compose_full_scene_tensors",
    "load_landcover_classes",
]

#: ``sunonsurface_2018a`` integrates ground-view factors by marching a
#: horizontal distance of ``second = round(1.1 * 20) = 22`` METRES (from
#: ``height = 1.1``), then scaling to pixels inside the kernel:
#: ``second = torch.round(second * scale)`` with ``scale`` in pixels per
#: metre. The march reach in CELLS is therefore pixel-size dependent --
#: ``round(22 / pixel_size_m)`` -- and is 22 cells only at 1 m pixels.
#: Lup/albedo/shadow values shift by up to that many cells into results,
#: so both the read halo and the write-window expansion use
#: :func:`gvf_march_pixels`.
GVF_MARCH_METERS = 22.0

#: Extra cells beyond the analytic bounds: guards rounding at the ray-march
#: step grid (``torch.round`` on the shift) and off-by-one clamps.
HALO_SAFETY_PIXELS = 2


def gvf_march_pixels(pixel_size_m: float) -> int:
    """Ground-view-factor march reach in cells for this site's pixel size.

    The oracle marches ``torch.round(22 m * scale)`` cells; ``ceil`` is used
    here so the halo and write margin are never shorter than the oracle's
    reach (``ceil(x) >= round(x)`` for every positive scale).
    """
    pixel = float(pixel_size_m)
    if not math.isfinite(pixel) or pixel <= 0.0:
        raise ValueError(f"pixel_size_m must be finite and positive: {pixel!r}")
    return int(math.ceil(GVF_MARCH_METERS / pixel))

#: Altitude of the lowest annulus for ``patch_option = 2`` (6 degrees); the
#: veg-SVF recompute march only needs to cover this once, statically.
LOWEST_SKY_PATCH_ALTITUDE_DEG = 6.0


class SolverInputError(ValueError):
    """Raised when solver inputs are inconsistent with the site cache."""


class StaleSvfError(RuntimeError):
    """A vegetation-dependent SVF was requested from a stale cache.

    The standalone SVF cache (``SVF/svfs_<n>.zip``) is keyed by file name
    only, so after any vegetation edit its vegetation terms describe the OLD
    scene. The solver refuses to slice those terms; they must be recomputed
    (Stage 7) or the job must fall back to the full path.
    """


class MarchRegimeError(SolverInputError):
    """A structural march-regime refusal (reserved; currently not raised).

    Historically raised when the scene's effective march amplitude left a
    sky patch in the ONE-STEP regime (``dz_1 > A_eff``) where the oracle's
    absolute stop (``scene.amaxvalue``) is multi-step (``dz_1 <= A_abs``)
    — a ``vbsh == 2.0`` value divergence the fold would silently consume.
    R6 replaced that refusal with the adjudicated escalation: banded
    patches march at the oracle's absolute ``scene.amaxvalue`` (identical
    executed step set per patch, bitwise-safe BY CONSTRUCTION), so no
    scene is refused on this ground anymore. The class stays as the typed
    home for STRUCTURAL march-regime refusals should a future lever
    reintroduce one (e.g. narrowing marches on bushed scenes); worker
    routing treats it like any :class:`SolverInputError` and falls back
    to the full-tile path.
    """


# ---------------------------------------------------------------------------
# Site forcing (Stage 8 inputs, float64, oracle-identical)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SiteForcing:
    """Float64 forcing series exactly as the full-domain oracle derives them."""

    met_table: np.ndarray  # (T, C) float64, np.loadtxt of the met text file
    altitude: np.ndarray  # (1, T)
    azimuth: np.ndarray  # (1, T)
    zen: np.ndarray  # (1, T)
    jday: np.ndarray  # (1, T)
    dectime: np.ndarray  # (T,)
    altmax: np.ndarray  # (1, T)
    location: dict  # {'longitude', 'latitude', 'altitude'} float64
    utc_offset_hours: float
    selected_date_str: str
    met_path: Path
    #: Scenario overlay (u-c3) whose resolution is already folded into
    #: ``met_table`` by ``load_site_forcing(overlay=...)``. ``None`` (the
    #: default) is the legacy baseline forcing, byte-identical to the
    #: pre-overlay construction. ``met_path`` always names the BASELINE text
    #: file; when ``overlay`` is set the full-tile path materializes the
    #: scratch met text from the RESOLVED table instead of copying it, so a
    #: prepared scratch site can never silently carry the baseline forcing
    #: the solver did not run with.
    overlay: ForcingOverlay | None = None

    @property
    def time_steps(self) -> int:
        return int(self.met_table.shape[0])


def _oracle_location(cache: SiteCache, site_dir: str | Path) -> dict:
    """Replicate ``compute_utci``'s location derivation bit-for-bit.

    Longitude/latitude are re-derived from the site Building_DSM raster's
    geotransform centre exactly the way the oracle does on every run, then
    checked against the manifest: a cache whose manifest lon/lat disagrees
    with its own rasters (operator error at build time, or re-staged rasters
    with a shifted extent) would make local solves and full-tile runs use
    different solar geometry. The altitude rule (median DEM, clamped to 3 m
    when positive) is re-derived from the cached DEM.
    """
    solar = cache.manifest.solar_geometry

    dsm_path = _site_raster_path(Path(site_dir).resolve(), "Building_DSM", cache.tile_key)
    if not dsm_path.is_file():
        raise SolverInputError(f"site building raster missing: {dsm_path}")
    dataset = gdal.Open(str(dsm_path))
    if dataset is None:
        raise SolverInputError(f"could not open site building raster: {dsm_path}")
    try:
        geotransform = dataset.GetGeoTransform()
        centre_x = geotransform[0] + geotransform[1] * dataset.RasterXSize / 2.0
        centre_y = geotransform[3] + geotransform[5] * dataset.RasterYSize / 2.0
        old_cs = osr.SpatialReference()
        old_cs.ImportFromWkt(dataset.GetProjection())
        old_cs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        wgs84 = osr.SpatialReference()
        wgs84.ImportFromEPSG(4326)
        wgs84.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        lon, lat = osr.CoordinateTransformation(old_cs, wgs84).TransformPoint(
            centre_x, centre_y
        )[:2]
    except RuntimeError as error:
        raise SolverInputError(
            f"could not derive the oracle location from {dsm_path}: {error}"
        ) from error
    finally:
        dataset = None

    dem_t = torch.from_numpy(np.array(cache.dem))  # copy: read-only memmap out
    alt = torch.median(dem_t).item()
    if alt > 0:
        alt = 3.0
    location = {
        "longitude": float(lon),
        "latitude": float(lat),
        "altitude": float(alt),
    }
    if (
        abs(location["longitude"] - float(solar.longitude)) > 1e-6
        or abs(location["latitude"] - float(solar.latitude)) > 1e-6
    ):
        raise SolverInputError(
            "cache manifest solar_geometry lon/lat "
            f"({solar.longitude}, {solar.latitude}) disagrees with the site "
            f"raster centre ({location['longitude']:.6f}, "
            f"{location['latitude']:.6f}); local solves and the full-domain "
            "oracle would use different solar geometry — rebuild the cache "
            "from these rasters"
        )
    if location["altitude"] != float(solar.altitude_m):
        raise SolverInputError(
            "cache manifest altitude_m "
            f"({solar.altitude_m}) disagrees with the oracle altitude rule "
            f"({location['altitude']}); the cache and the full-domain oracle "
            "would use different solar geometry"
        )
    return location


def _oracle_utc_offset(cache: SiteCache, selected_date_str: str) -> float:
    """Replicate ``compute_utci``'s timezone derivation (TimezoneFinder+pytz)."""
    from timezonefinder import TimezoneFinder
    import pytz

    solar = cache.manifest.solar_geometry
    base_date = datetime.datetime.strptime(selected_date_str, "%Y-%m-%d")
    tf = TimezoneFinder()
    timezone_name = (
        tf.timezone_at(lat=float(solar.latitude), lng=float(solar.longitude))
        or "UTC"
    )
    local_tz = pytz.timezone(timezone_name)
    local_dt = local_tz.localize(base_date)
    utc = local_dt.utcoffset().total_seconds() / 3600
    if utc != float(solar.utc_offset_hours):
        raise SolverInputError(
            f"UTC offset for {selected_date_str} ({timezone_name}, {utc} h) "
            "disagrees with the cache manifest "
            f"({solar.utc_offset_hours} h); solar geometry would differ "
            "between the local solve and the oracle"
        )
    return utc


def _met_path(cache: SiteCache, site_dir: str | Path) -> Path:
    entry = cache.manifest.sources.get("met")
    if entry is None:
        raise SolverInputError("cache manifest has no 'met' source entry")
    path = Path(site_dir).resolve() / entry.path
    if not path.is_file():
        raise SolverInputError(f"met source file not found: {path}")
    return path


def load_site_forcing(
    cache: SiteCache,
    *,
    site_dir: str | Path,
    selected_date_str: str,
    overlay: ForcingOverlay | None = None,
) -> SiteForcing:
    """Recompute float64 met + solar series the way ``compute_utci`` does.

    The cache's float32 copies are cross-checked (not used): a mismatch means
    the cache was built from different forcing and every derived result would
    be invalid.

    ``overlay`` carries a scenario's committed forcing edits
    (:class:`~solweig_gpu.incremental.adapters.met_time.ForcingOverlay`,
    built by that adapter's executor seam). The default ``None`` is the
    legacy path, byte-identical to the pre-overlay loader. With an overlay
    the *baseline* staleness check above still runs first (an overlay must
    ride a fresh cache, not paper over a stale one), then the resolved
    table is re-checked against the cache with the overlay's declared
    (variable, time) cells masked out — resolve discipline is verified at
    this seam, not trusted. The solar-geometry cross-check is NOT relaxed:
    the registry fence pins the met time columns, so a resolved table
    still yields the cached solar series (and a tampered overlay that
    moved them is rejected here, defensively). The overlay never mutates
    the loader's baseline array (copy-on-write, hard invariant).
    """
    met_path = _met_path(cache, site_dir)
    met_table = np.loadtxt(met_path, skiprows=1, delimiter=" ")
    if met_table.ndim != 2:
        raise SolverInputError(f"met table {met_path} is not 2-D")
    if met_table.shape[0] != cache.time_steps:
        raise SolverInputError(
            f"met table has {met_table.shape[0]} rows but the cache was built "
            f"with {cache.time_steps} timesteps"
        )
    cached_met = np.asarray(cache.met)
    if cached_met.shape != met_table.shape:
        raise SolverInputError(
            f"cache met shape {cached_met.shape} does not match the source "
            f"met table {met_table.shape}"
        )
    if not np.allclose(cached_met, met_table, rtol=1e-5, atol=1e-4):
        raise SolverInputError(
            "cached meteorology disagrees with the met source file; rebuild "
            "the site cache before running incremental jobs"
        )
    if overlay is not None:
        if not isinstance(overlay, ForcingOverlay):
            raise SolverInputError(
                "overlay must be a ForcingOverlay, got "
                f"{type(overlay).__name__}"
            )
        resolved = overlay.resolve(met_table)
        if resolved.shape != met_table.shape:
            raise SolverInputError(
                "overlay-resolved met table shape "
                f"{resolved.shape} does not match the baseline "
                f"{met_table.shape}"
            )
        unchanged = ~overlay.changed_cell_mask(met_table.shape)
        if not np.allclose(
            cached_met[unchanged], resolved[unchanged], rtol=1e-5, atol=1e-4
        ):
            raise SolverInputError(
                "overlay-resolved meteorology disagrees with the met source "
                "outside the overlay's declared (variable, time) cells; "
                "refusing the scenario forcing overlay"
            )
        met_table = resolved

    location = _oracle_location(cache, site_dir)
    utc = _oracle_utc_offset(cache, selected_date_str)
    (
        _YYYY,
        altitude,
        azimuth,
        zen,
        jday,
        _leafon,
        dectime,
        altmax,
    ) = Solweig_2015a_metdata_noload(met_table, location, utc)

    solar = np.asarray(cache.solar)
    if solar.shape[0] != met_table.shape[0]:
        # Never skip the cross-check silently: a row-count mismatch means the
        # cache was built from a different forcing series.
        raise SolverInputError(
            f"cache solar series has {solar.shape[0]} rows but the met table "
            f"has {met_table.shape[0]}; rebuild the site cache"
        )
    # float32 storage rounding is expected; only gross mismatches reject.
    for column, derived in (
        (0, altitude[0]),
        (1, azimuth[0]),
        (5, altmax[0]),
    ):
        if not np.allclose(solar[:, column], derived, rtol=1e-4, atol=1e-2):
            raise SolverInputError(
                "cached solar geometry disagrees with the recomputed "
                f"series (column {column}); rebuild the site cache"
            )
    first_row = met_table[0]
    derived_date = datetime.datetime(
        int(first_row[0]), 1, 1
    ) + datetime.timedelta(days=int(first_row[1]) - 1)
    if derived_date.strftime("%Y-%m-%d") != selected_date_str:
        raise SolverInputError(
            f"selected_date_str {selected_date_str!r} does not match the met "
            f"table date {derived_date.strftime('%Y-%m-%d')}"
        )

    return SiteForcing(
        met_table=met_table,
        altitude=np.asarray(altitude),
        azimuth=np.asarray(azimuth),
        zen=np.asarray(zen),
        jday=np.asarray(jday),
        dectime=np.asarray(dectime),
        altmax=np.asarray(altmax),
        location=location,
        utc_offset_hours=utc,
        selected_date_str=selected_date_str,
        met_path=met_path,
        overlay=overlay,
    )


def load_landcover_classes() -> np.ndarray:
    """Parse the land-cover class table exactly like ``compute_utci``."""
    with open(landcover_classes_path) as f:
        lines = f.readlines()[1:]
    lc_class = np.empty((len(lines), 6), dtype=float)
    for i, ln in enumerate(lines):
        lc_class[i, :] = [float(x) for x in ln.split()[1:]]
    return lc_class


def resolve_landcover_overlay(
    cache: SiteCache, overlay: LandCoverOverlay | None
) -> np.ndarray:
    """Materialize the scenario class grid (u-c4 solver seam).

    Single materialization point for both solve paths, mirroring
    ``load_site_forcing(overlay=...)``: the resolved grid is
    ``overlay.resolve(baseline)`` — copy-on-write, so the cache-backed
    memmap is bitwise unchanged — and the resolve is VERIFIED here, never
    trusted: outside the overlay's changed-cell mask the resolved grid must
    equal the baseline bitwise (an undeclared-cell tamper is refused loudly,
    exactly like the forcing loader's masked re-check). ``overlay=None``
    never reaches this seam; callers keep their legacy cache reads.

    A site whose cache carries no landcover raster raises here: the physics
    would run with ``landcover_grid=None`` and the paint could never reach
    it — refused rather than silently dropped.
    """
    if not isinstance(overlay, LandCoverOverlay):
        raise SolverInputError(
            "landcover overlay must be a LandCoverOverlay, got "
            f"{type(overlay).__name__}"
        )
    if "landcover" not in cache:
        raise SolverInputError(
            "scenario landcover overlay on a site whose cache has no "
            "landcover raster; the paint could never reach the physics "
            "(utci_process.py:608 consumes the class grid only when the "
            "site carries one) — refusing the overlay"
        )
    baseline = np.array(cache.landcover)  # copy: read-only memmap out
    resolved = overlay.resolve(baseline)
    mask = overlay.changed_cell_mask(baseline.shape)
    if not np.array_equal(resolved[~mask], baseline[~mask]):
        raise SolverInputError(
            "overlay-resolved landcover disagrees with the baseline class "
            "grid outside the overlay's declared painted cells; refusing "
            "the scenario paint overlay"
        )
    return resolved


# ---------------------------------------------------------------------------
# Full-scene composition (shared by halo, local solve, and full fallback)
# ---------------------------------------------------------------------------


@dataclass
class FullSceneTensors:
    """Full-site tensors derived exactly like ``compute_utci`` (float32)."""

    a: torch.Tensor  # building DSM
    canopy: torch.Tensor  # vegetation above ground (edited scene)
    dem: torch.Tensor
    vegdem: torch.Tensor  # canopy + dem (ground-relative)
    vegdem2: torch.Tensor  # trunk zone + dem (ground-relative)
    bush: torch.Tensor
    vegdsm: torch.Tensor  # canopy + building DSM
    vegdsm2: torch.Tensor
    amaxvalue: torch.Tensor  # scalar tensor: max(a.max(), vegdem.max())


def _compose_scene_from_canopy(
    cache: SiteCache, canopy_np: np.ndarray
) -> FullSceneTensors:
    """Compose the scene tensors from a KNOWN canopy raster.

    Byte-identical body of :func:`compose_full_scene_tensors`, extracted so
    the vegetation-occlusion state (R4) can recompose a past scene from its
    stored canopy snapshot without going through a tree layer.
    """
    # np.array copies: the memmaps are read-only and torch.from_numpy would
    # warn on (and forbid) writes.
    a = torch.from_numpy(np.array(cache.building_dsm))
    temp2 = torch.from_numpy(np.array(cache.dem))
    temp1 = torch.from_numpy(np.ascontiguousarray(canopy_np))
    temp1[temp1 < 0.0] = 0.0

    vegdem = temp1 + temp2
    vegdem2 = torch.add(temp1 * 0.25, temp2)
    bush = torch.logical_not(vegdem2 * vegdem) * vegdem
    vegdsm = temp1 + a
    vegdsm[vegdsm == a] = 0
    vegdsm2 = temp1 * 0.25 + a
    vegdsm2[vegdsm2 == a] = 0
    amaxvalue = torch.maximum(a.max(), vegdem.max())
    return FullSceneTensors(
        a=a,
        canopy=temp1,
        dem=temp2,
        vegdem=vegdem,
        vegdem2=vegdem2,
        bush=bush,
        vegdsm=vegdsm,
        vegdsm2=vegdsm2,
        amaxvalue=amaxvalue,
    )


def compose_full_scene_tensors(cache: SiteCache, layer: TreeLayer) -> FullSceneTensors:
    """Compose the edited scene exactly the way the oracle pipeline does.

    Derivation order and expressions replicate ``compute_utci`` (and the
    standalone branch of ``svf_calculator``) line for line so the resulting
    tensors are bitwise identical to the oracle's on the same scene.
    """
    grid = RasterGrid(
        rows=cache.rows,
        cols=cache.cols,
        pixel_size_m=cache.pixel_size_m,
        origin_x_m=cache.manifest.origin_x_m,
        origin_y_m=cache.manifest.origin_y_m,
    )
    canopy_np = layer.vegetation_rasters_window(grid.full_window)[0]
    return _compose_scene_from_canopy(cache, canopy_np)


def _crop(
    tensor: torch.Tensor, window: RasterWindow
) -> torch.Tensor:
    return tensor[
        window.row_start : window.row_stop, window.col_start : window.col_stop
    ]


# ---------------------------------------------------------------------------
# Stage 4: halo / read window
# ---------------------------------------------------------------------------


def _march_reach_pixels(
    *,
    height_amplitude_m: float,
    scale: float,
    altitude_deg: float,
) -> int:
    """Pixels a ray march can span at ``altitude_deg``.

    A march step at index ``i`` shifts by ``(dx, dy)`` with
    ``max(|dx|, |dy|) == i`` and compares occluder height minus
    ``dz = ds * i * tan(altitude) / scale`` against the target surface. An
    occluder beyond Chebyshev distance ``N`` can still block the target only
    if ``height_amplitude - ds*i*tan(alt)/scale > 0`` for some step ``i <=
    N``, i.e. only if ``N < amplitude * scale / tan(altitude)``. Cells
    further than ``N`` from any candidate occluder see bit-identical march
    results on any window that contains them.
    """
    if altitude_deg <= 0.0:
        return 0
    tan_alt = math.tan(math.radians(min(altitude_deg, 89.999)))
    if tan_alt <= 0.0:
        return int(1e9)
    return int(math.ceil(height_amplitude_m * scale / tan_alt))


def required_halo_pixels(
    cache: SiteCache,
    scene: FullSceneTensors,
    forcing: SiteForcing,
) -> int:
    """Halo (cells) around the write window for a bitwise-exact local solve.

    The bound must cover, for every cell whose inputs must be correct (write
    window plus the GVF shift reach), all ray marches that can influence it:

    * every positive daytime solar altitude in the forcing series (direct
      shadow, ``shadowingfunction_wallheight_23``);
    * the lowest sky-patch altitude (6 degrees for ``patch_option=2``,
      covering the veg-SVF recompute marches).

    Plus the pixel-size-dependent ground-view-factor march
    (:func:`gvf_march_pixels`) and a small safety margin. Capped at the site
    diagonal: a halo that large simply makes the read window the full site
    (still exact).
    """
    scale = 1.0 / float(cache.pixel_size_m)
    amplitude = float(scene.amaxvalue.item()) - float(
        torch.min(scene.a).item()
    )
    altitude_degrees = [LOWEST_SKY_PATCH_ALTITUDE_DEG]
    altitude_degrees.extend(
        float(value) for value in np.asarray(forcing.altitude[0]) if value > 0.0
    )
    reach = max(
        _march_reach_pixels(
            height_amplitude_m=amplitude, scale=scale, altitude_deg=altitude
        )
        for altitude in altitude_degrees
    )
    halo = reach + gvf_march_pixels(cache.pixel_size_m) + HALO_SAFETY_PIXELS
    return int(min(halo, max(cache.rows, cache.cols)))


def _minimum_march_altitude_deg(forcing: SiteForcing) -> float:
    """Lowest altitude any march can run at: sky patches or the sun series."""
    altitudes = [LOWEST_SKY_PATCH_ALTITUDE_DEG]
    altitudes.extend(
        float(value) for value in np.asarray(forcing.altitude[0]) if value > 0.0
    )
    return min(altitudes)


def effective_march_amplitude(
    a: torch.Tensor,
    vegdsm: torch.Tensor,
    vegdsm2: torch.Tensor,
    *,
    scene_amaxvalue: torch.Tensor,
) -> torch.Tensor:
    """Early-exit march amplitude that is bit-identical to the oracle's.

    In ``shadowingfunction_wallheight_23`` the ``amaxvalue`` argument is used
    only as the while-loop stop (``amaxvalue >= dz``); no value computed
    inside the loop depends on it.  Every per-step update is either

    * a strict comparison against ``a`` (``temp > a`` for ``fabovea`` /
      ``gabovea``, and the ``f``/``shvoveg`` running maxima start at ``a`` /
      ``vegdsm`` whose final use is gated), or
    * a running maximum whose result only survives through the ``vegsh``
      gate, and the gate can only be open at cells where some executed
      candidate already exceeded ``a`` there.

    Therefore a step with ``dz > max(a, vegdsm, vegdsm2) - min(a)`` over the
    arrays actually being marched cannot change any output: its candidates
    ``surface[o] - dz < a[t]`` flip no strict comparison, cannot raise the
    running maxima at open-gate cells (those already hold a value
    ``> a[t]``), cannot open the gate (the one-step-lagged ``lastfabovea``
    of a skipped step equals the previous step's ``fabovea``, so the
    per-step ``vegsh2`` is non-increasing once candidates fall below
    ``a``), and the ``vbshvegsh`` accumulator adds only the unchanged
    running ``vegsh``, which cannot flip its final ``> 0`` classification.

    The bound is *relative* (max minus min), not the oracle's absolute
    ``max(a, vegdem)``, so on sites with high base elevations marches stop
    after the local relief instead of the absolute elevation.  The result is
    clamped to ``[0, scene_amaxvalue]``: never negative on flat sites, and
    never longer than the oracle's own march.
    """
    bound = (
        torch.maximum(
            torch.maximum(a.max(), vegdsm.max()),
            vegdsm2.max(),
        )
        - torch.min(a)
    )
    bound = torch.maximum(bound, torch.zeros((), dtype=bound.dtype))
    return torch.minimum(bound, scene_amaxvalue.to(dtype=bound.dtype))


#: First-step march distances per sky patch, memoized per pixel scale (the
#: geometry is a pure function of ``create_patches(2)`` and the scale).
_FIRST_STEP_DZ: dict[float, np.ndarray] = {}


def _patch_first_step_distances(scale: float) -> np.ndarray:
    """dz of the FIRST march step per patch, association-matched arithmetic.

    Copied from ``veg_svf_state._patch_first_step_dz`` (r4a-owned) rather
    than imported: ``veg_svf_state`` already imports this module, so the
    import would be circular, and the standalone replay must not depend on
    the state module (it has to work with the occluder store absent or
    kill-switched). The arithmetic is association-matched to ``shadow()``:
    step 1 always executes (dz starts at 0.0), so a patch is in the
    ONE-STEP regime at amplitude ``A`` iff ``dz_1 > A`` — the regime whose
    step-1 accumulator zero lets ``vbsh`` reach 2.0. Same branch/ds
    expressions as the march, and the SAME dz association
    (``ds * index * (tan(alt)/scale)`` with ``tan/scale`` folded FIRST —
    bitwise-equal to the naive ``ds*tan(alt)/scale`` only at power-of-two
    scales, review NEW-LOW). The zenith patch carries the -1.0 sentinel
    (it never marches).
    """
    cached = _FIRST_STEP_DZ.get(float(scale))
    if cached is not None:
        # Copy on lend: callers must treat the cache as immutable — a
        # mutated return would poison the fence process-wide
        # (w2-review F1).
        return cached.copy()
    patches, _rings = _sky_patch_geometry(2)
    degrees = torch.pi / 180.0
    pibyfour = torch.pi / 4.0
    out = np.zeros(len(patches), dtype=np.float32)
    for i, patch in enumerate(patches):
        altitude, azimuth = patch[0], patch[1]
        if float(altitude) >= 90.0:
            out[i] = -1.0  # sentinel: the zenith patch never marches
            continue
        az = azimuth
        if float(az) == 0.0:
            az = az * 0.0 + 1e-12
        az = az * degrees
        alt = altitude * degrees
        sinazimuth = torch.sin(az)
        cosazimuth = torch.cos(az)
        if bool(pibyfour <= az < 3.0 * pibyfour) or bool(
            5.0 * pibyfour <= az < 7.0 * pibyfour
        ):
            ds = torch.abs(1.0 / sinazimuth)
        else:
            ds = torch.abs(1.0 / cosazimuth)
        # association-matched to the march: dz = ds * index * tbs with
        # tbs = tan(alt)/scale computed FIRST (review NEW-LOW)
        tanaltitudebyscale = torch.tan(alt) / scale
        out[i] = float(ds * 1.0 * tanaltitudebyscale)
    _FIRST_STEP_DZ[float(scale)] = out
    return out


def _banded_patch_mask(
    amplitude: float, absolute_amaxvalue: float, scale: float
) -> np.ndarray:
    """Sky patches whose first march step lands in the escalation band.

    A patch with ``dz_1`` in ``(A_eff, A_abs]`` is ONE-STEP at the
    effective amplitude but MULTI-STEP under the oracle's absolute stop
    (``svf_calculator`` marches every patch at ``amaxvalue``): its step-1
    ``firstvegdem`` raise would survive un-buffered into ``vbshvegsh`` in
    a regime the oracle never runs. The R6 replacement for the W2 refusal
    escalates exactly these patches to the absolute stop (see
    :func:`_recompute_veg_svf_window`) — identical executed step set per
    patch, so the serve is bitwise-equal to the oracle BY CONSTRUCTION.
    """
    dz1 = _patch_first_step_distances(scale)
    return (dz1 > float(amplitude)) & (dz1 <= float(absolute_amaxvalue))


def _sun_first_step_distance(
    azimuth_deg: float, altitude_deg: float, scale: float
) -> float:
    """First NONZERO march distance of ``shadowingfunction_wallheight_23``.

    Association-matched to the time-loop march: ``index`` starts at 0
    (``dz`` 0, the self-comparison), the first shifted step is index 1
    with ``dz = (ds * index) * (tan(altitude) / scale)`` — the
    ``tan/scale`` quotient folded FIRST, ``ds`` from the same azimuth
    branch the march takes, and NO azimuth-zero substitution (unlike
    ``shadow()``; the wall-height march does not apply one).
    """
    degrees = torch.pi / 180.0
    pibyfour = torch.pi / 4.0
    az = torch.tensor(float(azimuth_deg)) * degrees
    alt = torch.tensor(float(altitude_deg)) * degrees
    sinazimuth = torch.sin(az)
    cosazimuth = torch.cos(az)
    if bool(pibyfour <= az < 3.0 * pibyfour) or bool(
        5.0 * pibyfour <= az < 7.0 * pibyfour
    ):
        ds = torch.abs(1.0 / sinazimuth)
    else:
        ds = torch.abs(1.0 / cosazimuth)
    tanaltitudebyscale = torch.tan(alt) / scale
    return float((ds * 1.0) * tanaltitudebyscale)


def time_loop_band_present(
    window_amplitude: float,
    absolute_amaxvalue: float,
    scale: float,
    altitude: np.ndarray,
    azimuth: np.ndarray,
    *,
    time_start: int = 0,
    time_stop: int | None = None,
) -> bool:
    """Whether any timestep the time loop will run sits in the sun band.

    The time-loop march (``shadowingfunction_wallheight_23`` at the
    ``precomputed_shadows`` seam) runs while ``amaxvalue >= dz``; a
    timestep whose sun first-step distance lands in
    ``(window_amplitude, absolute_amaxvalue]`` would execute ONE step
    under the windowed effective amplitude where the oracle's absolute
    stop runs several — a step-regime difference previously immune only
    through a missing-accumulator quirk of the wall-height march. R6
    closes the latent note by escalating such loops to the absolute
    amplitude (the same ``scene.amaxvalue`` source the SVF escalation
    uses). Night timesteps (altitude <= 0) never march and never band;
    only the requested ``[time_start, time_stop)`` prefix is examined —
    the exact range ``run_utci_window`` will replay.
    """
    total_steps = int(np.asarray(altitude).shape[1])
    stop = total_steps if time_stop is None else int(time_stop)
    for t in range(int(time_start), stop):
        alt_t = float(np.asarray(altitude)[0][t])
        if alt_t <= 0.0:
            continue
        dz1 = _sun_first_step_distance(
            float(np.asarray(azimuth)[0][t]), alt_t, scale
        )
        if window_amplitude < dz1 <= absolute_amaxvalue:
            return True
    return False


# Conservative superset margin (metres) for occluder-mask inclusion.
_OCCLUDER_MASK_SLACK_M = 0.01


def read_window_for_write_window(
    write_window: RasterWindow,
    cache: SiteCache,
    scene: FullSceneTensors,
    forcing: SiteForcing,
) -> RasterWindow:
    """Exact read window: state ring around the write window plus occluders.

    Two cell classes must be inside the read window:

    * *targets* — cells whose values must be correct.  Temporal state
      (``Tgmap1`` family) propagates only through the ground-view-factor
      march (:func:`gvf_march_pixels` plus a safety margin), so the targets
      are ``write_window`` expanded by that ring (the P5-validated halo
      structure).
    * *occluders* — cells whose surface heights can flip any march
      comparison for a target.  A march step at Chebyshev distance ``d``
      metres applies ``dz = d * tan(altitude)``; it can flip target ``t``
      only if ``max(a, vegdsm, vegdsm2)[o] - a[t] > d * tan(alt)`` for some
      marched altitude.  All marched altitudes (sky patches down to 6
      degrees, the positive solar series) are at least
      :func:`_minimum_march_altitude_deg`, whose tangent gives the largest
      reach, so a single mask

          ``occluder_h - min(a over targets) > cheb_m * tan(alt_min)``

      is an exact (per-amplitude, slack-widened) superset of the occluders
      any march can consult.  Using the minimum target elevation keeps the
      mask a superset for every target cell individually.

    The returned window is the bounding box of targets and the mask,
    clamped to the site.  It is always contained in the old conservative
    ``write.expand(required_halo_pixels(...))`` window, whose exactness the
    scientific differential already validated: every cell dropped here
    contributes march candidates at or below ``a[t]`` for all targets.
    """
    pixel = float(cache.pixel_size_m)
    targets = write_window.expand(
        gvf_march_pixels(pixel) + HALO_SAFETY_PIXELS
    ).clamp(rows=cache.rows, cols=cache.cols)
    if targets.area >= cache.rows * cache.cols:
        return targets

    tan_min = math.tan(math.radians(_minimum_march_altitude_deg(forcing)))
    if tan_min <= 0.0:  # pragma: no cover - alt_min >= 6 degrees always
        return RasterWindow(
            0, cache.rows, 0, cache.cols
        )
    target_floor_m = float(
        scene.a[
            targets.row_start : targets.row_stop,
            targets.col_start : targets.col_stop,
        ].min()
    )

    rows = torch.arange(cache.rows)
    cols = torch.arange(cache.cols)
    row_dist = torch.clamp(
        targets.row_start - rows, min=0
    ) + torch.clamp(rows - (targets.row_stop - 1), min=0)
    col_dist = torch.clamp(
        targets.col_start - cols, min=0
    ) + torch.clamp(cols - (targets.col_stop - 1), min=0)
    cheb_m = (
        torch.maximum(row_dist[:, None], col_dist[None, :]).to(torch.float32)
        * pixel
    )

    occluder_h = torch.maximum(
        scene.a, torch.maximum(scene.vegdsm, scene.vegdsm2)
    )
    mask = (occluder_h - target_floor_m) > (
        cheb_m * tan_min - _OCCLUDER_MASK_SLACK_M
    )
    mask[
        targets.row_start : targets.row_stop,
        targets.col_start : targets.col_stop,
    ] = True

    rows_any = torch.nonzero(mask.any(dim=1))
    cols_any = torch.nonzero(mask.any(dim=0))
    return RasterWindow(
        row_start=int(rows_any[0].item()),
        row_stop=int(rows_any[-1].item()) + 1,
        col_start=int(cols_any[0].item()),
        col_stop=int(cols_any[-1].item()) + 1,
    )


# ---------------------------------------------------------------------------
# Stage 7: windowed SVF bundle (static split + dynamic veg recompute)
# ---------------------------------------------------------------------------

_SVF_BUILDING_SCALARS = ("svf", "svfE", "svfS", "svfW", "svfN")


def window_svf_bundle(
    cache: SiteCache,
    scene: FullSceneTensors,
    window: RasterWindow,
    *,
    veg_changed: bool,
    cube_window: RasterWindow | None = None,
    veg_state=None,
) -> tuple:
    """Assemble the 19-tuple SVF bundle cropped to ``window``.

    Building-only terms (``svf``, directional ``svfE/S/W/N`` and the
    building shadow cube ``shmat``) are scene-invariant: they are sliced from
    the cache and are bitwise identical to what the oracle computes on any
    tree-edit state of the same built scene. Every vegetation-dependent term
    (``svfveg`` family, ``svf*aveg`` family, ``vegshmat``, ``vbshvegshmat``,
    ``svftotal``) is recomputed over the window by replaying the exact
    ``svf_calculator`` accumulation loop with the composed scene.

    When ``veg_changed`` is false and the layer has no vegetation edits the
    cached vegetation terms may be sliced instead (identical arithmetic), but
    this solver stays conservative: it always recomputes vegetation terms
    locally. Passing ``veg_changed=False`` while the scene differs from the
    cached baseline raises :class:`StaleSvfError` — that guard exists so the
    worker can never silently consume the stale standalone SVF cache after a
    vegetation edit (WORKLOG hazard: the SVF cache is keyed by file name).

    ``cube_window`` (perf wave 1 STEP 2b, default ``None`` = ``window``)
    narrows the spatial extent of the three patch CUBES
    (``vegshmat``/``vbshvegshmat``/``shmat``, positions 15-17): the SVF
    scalars keep the full read extent because the physics consumes them
    halo-wide, while ``run_utci_window`` crops every cube to its write
    window before use, so slicing the memmap at the extent actually
    consumed is pure data movement, bit-identical to the late crop. Must
    be contained in ``window``; callers passing it must also tell
    ``run_utci_window`` (``svf_bundle_cubes_cropped=True``) so the crop
    is not applied twice.

    In the recompute branch (``veg_changed=True``) the cube window has a
    second role (perf wave 2 lever 1): it is also the replay's ACCUMULATE
    extent — every vegetation accumulator, the vegetation cubes and each
    per-patch ``shadow()`` march are evaluated for that extent only (the
    marches on quadrant sub-windows that cover it; see
    :func:`_patch_march_window`). The per-cell scalars are zero-padded
    back to the read extent so the bundle contract above is unchanged.

    ``veg_state`` (R4 Phase A, default ``None`` = the replay above) hands
    in a validated vegetation occluder state
    (:class:`solweig_gpu.incremental.veg_svf_state.VegOcclusionState`,
    duck-typed here): the per-patch planes are unpacked from its packed
    bits at the cube window — values identical to the replay's
    ``vegshmat``/``vbshvegshmat`` — and folded through the SAME fold
    (:func:`_fold_veg_svf_from_planes`), so every scalar is bit-identical
    to the replay on the same bits. A state that does not validate against
    this cache and scene raises :class:`StaleSvfError` (loud; the state
    layer's typed refusals route the batch to the full replay BEFORE
    reaching this seam).
    """
    if window.row_start < 0 or window.col_start < 0:
        raise SolverInputError(f"window must be clamped to the site: {window}")
    if window.row_stop > cache.rows or window.col_stop > cache.cols:
        raise SolverInputError(f"window exceeds the site grid: {window}")
    if window.is_empty:
        raise SolverInputError("window must be non-empty")
    if cube_window is None:
        cube_window = window
    if (
        cube_window.row_start < window.row_start
        or cube_window.row_stop > window.row_stop
        or cube_window.col_start < window.col_start
        or cube_window.col_stop > window.col_stop
    ):
        raise SolverInputError(
            f"cube window {cube_window} escapes the read window {window}"
        )

    building_scalars = {
        name: _slice_cache_to_tensor(cache, name, window)
        for name in _SVF_BUILDING_SCALARS
    }
    shmat_w = _slice_patch_cube_to_tensor(cache, "shadowmat", cube_window)

    if not veg_changed:
        # Only legitimate when the composed scene equals the cached baseline
        # vegetation. Verify instead of trusting the caller. The cached Trees
        # raster is stored verbatim (it may carry negative nodata, e.g. -999
        # in LiDAR products), while ``scene.canopy`` is clamped at zero --
        # compare clamped against clamped, mirroring the baseline-amaxvalue
        # guard, so a no-edit scene never trips this check on nodata alone.
        cached_tree = np.asarray(cache.tree_base)
        cached_tree = np.where(cached_tree < 0.0, 0.0, cached_tree)
        if not np.array_equal(cached_tree, scene.canopy.numpy()):
            raise StaleSvfError(
                "vegetation-dependent SVF terms requested from the cache "
                "after a vegetation edit; recompute them (Stage 7) or fall "
                "back to the full-tile path"
            )
        veg_scalars = {
            name: _slice_cache_to_tensor(cache, name, window)
            for name in (
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
        }
        vegshmat_w = _slice_patch_cube_to_tensor(cache, "vegshadowmat", cube_window)
        vbshvegshmat_w = _slice_patch_cube_to_tensor(cache, "vbshmat", cube_window)
        svftotal_w = _slice_cache_to_tensor(cache, "svftotal", window)
    else:
        if veg_state is not None:
            # R4 Phase A: packed occluder bits -> per-patch planes -> the
            # SAME fold as the replay. Validation is the state layer's
            # contract; a mismatch here is a loud invariant breach.
            veg_state.validate_against(cache, scene)
            vegshmat_w, vbshvegshmat_w = veg_state.unpack_planes(cube_window)
            veg_scalars, vegshmat_w, vbshvegshmat_w, svftotal_w = (
                _fold_veg_svf_from_planes(
                    vegshmat=vegshmat_w,
                    vbshvegshmat=vbshvegshmat_w,
                    vegdem2_read=_crop(scene.vegdsm2, window),
                    svf_building_window=building_scalars["svf"],
                    acc_r0=cube_window.row_start - window.row_start,
                    acc_r1=cube_window.row_stop - window.row_start,
                    acc_c0=cube_window.col_start - window.col_start,
                    acc_c1=cube_window.col_stop - window.col_start,
                    rows=window.height,
                    cols=window.width,
                )
            )
        else:
            # Perf wave 2: the replay accumulates only over the consumed extent
            # (cube_window) with per-patch march windows, and returns the
            # vegetation cubes directly at that extent — the same data movement
            # the veg_changed=False branch does at the memmap. The per-cell
            # scalars keep the read extent (zero-padded outside the accumulate
            # window, where nothing consumes them).
            veg_scalars, vegshmat_w, vbshvegshmat_w, svftotal_w = (
                _recompute_veg_svf_window(
                    cache,
                    scene,
                    window,
                    building_scalars["svf"],
                    accumulate_window=cube_window,
                )
            )

    # svf_calculator return order:
    # (svf, svfaveg, svfE, svfEaveg, svfEveg, svfN, svfNaveg, svfNveg,
    #  svfS, svfSaveg, svfSveg, svfveg, svfW, svfWaveg, svfWveg,
    #  vegshmat, vbshvegshmat, shmat, SVFtotal)
    return (
        building_scalars["svf"],
        veg_scalars["svfaveg"],
        building_scalars["svfE"],
        veg_scalars["svfEaveg"],
        veg_scalars["svfEveg"],
        building_scalars["svfN"],
        veg_scalars["svfNaveg"],
        veg_scalars["svfNveg"],
        building_scalars["svfS"],
        veg_scalars["svfSaveg"],
        veg_scalars["svfSveg"],
        veg_scalars["svfveg"],
        building_scalars["svfW"],
        veg_scalars["svfWaveg"],
        veg_scalars["svfWveg"],
        vegshmat_w,
        vbshvegshmat_w,
        shmat_w,
        svftotal_w,
    )


def _slice_cache_to_tensor(
    cache: SiteCache, name: str, window: RasterWindow
) -> torch.Tensor:
    array = np.array(cache.window(name, window))  # copy: read-only memmap out
    return torch.from_numpy(array)


def _slice_patch_cube_to_tensor(
    cache: SiteCache, name: str, window: RasterWindow
) -> torch.Tensor:
    array = np.array(
        cache.svf_patches[name][
            window.row_start : window.row_stop,
            window.col_start : window.col_stop,
            :,
        ]
    )  # copy: writable, contiguous, and independent of the memmap
    return torch.from_numpy(array)


def _patch_march_reach_pixels(
    height_amplitude_m: float, scale: float, altitude_deg: float
) -> int:
    """Upper bound on ``shadow()``'s march step count for one sky patch.

    ``shadow()`` steps while ``amaxvalue >= ds * (i - 1) * tan(alt) / scale``
    with ``ds >= 1``, so the last executed step index satisfies
    ``i <= height_amplitude_m * scale / tan(alt) + 1``. The ``ceil`` plus one
    extra step guards the float32 evaluation of the comparison inside the
    march; over-covering a march window is harmless (the march is a pure
    per-cell function of its input rasters), under-covering it would drop
    steps. Mirrors :func:`_march_reach_pixels` at the patch-altitude
    granularity (perf wave 2 lever 1).
    """
    if altitude_deg >= 90.0:
        return 1
    if altitude_deg <= 0.0:
        return 0
    tan_alt = math.tan(math.radians(min(altitude_deg, 89.999)))
    if tan_alt <= 0.0:
        return 0
    return int(math.ceil(float(height_amplitude_m) * float(scale) / tan_alt)) + 1


def _quadrant_read_direction(azimuth_deg: float) -> tuple[int, int]:
    """Axes ``shadow()``'s march reads toward for a patch azimuth.

    In ``shadow()`` the row shift is ``dx = -sign(cos(az))`` and the column
    shift is ``dy = +sign(sin(az))``: a target cell reads occluders at
    ``(r + dx, c + dy)``. An axis whose trig factor is zero never shifts
    (the shift is ``round(i * tan)`` of a sub-tolerance angle, zero for any
    realistic step count), so it needs no expansion. Within a tolerance of
    a trig zero the classification may return 0 where the float32 sign is
    formally nonzero, never the opposite quadrant — and a zeroed axis is
    exact (see above), so both readings are safe.
    """
    az = math.fmod(float(azimuth_deg), 360.0)
    if az < 0.0:
        az += 360.0
    sin_v = math.sin(math.radians(az))
    cos_v = math.cos(math.radians(az))
    # treat |trig| <= eps as an exact zero: the float32 shift for such an
    # angle is round(i * tan(angle)) = 0 for every realistic step count
    eps = 1.0e-12
    row_dir = -(1 if cos_v > eps else (-1 if cos_v < -eps else 0))
    col_dir = 1 if sin_v > eps else (-1 if sin_v < -eps else 0)
    return row_dir, col_dir


def _patch_march_window(
    accumulate_window: RasterWindow,
    read_window: RasterWindow,
    *,
    azimuth_deg: float,
    reach_pixels: int,
) -> RasterWindow:
    """March extent for one sky patch (perf wave 2 lever 1).

    ``bbox(accumulate window expanded by ``reach_pixels`` in the patch's
    READ directions only)`` clamped to the read window. For every target
    cell inside the accumulate window this window reproduces the
    full-read-window march bit for bit:

    * every step ``i <= reach_pixels`` stays inside it (the per-axis shift
      is bounded by ``i`` and only the shifted axes expand), and
    * reads that clamp at its edge clamp at the read window's edge in the
      full march too whenever the two edges coincide (the expansion is
      clipped by the read window), and never reach an interior edge.

    Cells outside the accumulate window are simply not marched; their
    values were never consumed.
    """
    row_dir, col_dir = _quadrant_read_direction(azimuth_deg)
    return RasterWindow(
        max(
            accumulate_window.row_start - (reach_pixels if row_dir < 0 else 0),
            read_window.row_start,
        ),
        min(
            accumulate_window.row_stop + (reach_pixels if row_dir > 0 else 0),
            read_window.row_stop,
        ),
        max(
            accumulate_window.col_start - (reach_pixels if col_dir < 0 else 0),
            read_window.col_start,
        ),
        min(
            accumulate_window.col_stop + (reach_pixels if col_dir > 0 else 0),
            read_window.col_stop,
        ),
    )


def _bush_free_marches(bush: torch.Tensor) -> bool:
    """Whether march extents may be narrowed for this scene.

    ``shadow()`` gates two branches on global reductions over the marched
    extent (``bush.max() > 0`` per step and post-march); with an all-zero
    bush layer those predicates are False for EVERY extent, so a narrowed
    march evaluates them identically. Any nonzero bush cell (including
    negative ones) forces the full-window march instead — refuse rather
    than risk flipping an extent-dependent predicate.
    """
    return not bool(bush.any())


#: Memoized sky-patch geometry keyed by ``create_patches`` option (perf
#: wave 2 lever 2): ``(patches, rings)`` where ``patches[index]`` is
#: ``(altitude, azimuth, ring)`` and ``rings[ring]`` is
#: ``(altitude, patch_count, weights)`` with ``weights[k - span_start]`` the
#: ``(isotropic, anisotropic)`` annulus-weight pair for annulus ``k``.
_SKY_PATCH_GEOMETRY: dict[int, tuple[tuple, tuple]] = {}


def _sky_patch_geometry(patch_option: int = 2):
    """Build (once per process) the constant sky-patch replay geometry.

    ``create_patches``, the ``iazimuth`` fill, ``aziintervalaniso`` and every
    ``annulus_weight(k, ...)`` value depend only on the patch option, so the
    per-solve reconstruction the replay used to perform is pure
    recomputation. The memoized values are produced by the same expressions
    on the same inputs — bitwise identical tensors, merely not rebuilt.
    """
    cached = _SKY_PATCH_GEOMETRY.get(patch_option)
    if cached is not None:
        return cached

    device = torch.device("cpu")
    (
        _skyvaultalt,
        _skyvaultazi,
        annulino,
        skyvaultaltint,
        aziinterval,
        _skyvaultaziint,
        azistart,
    ) = create_patches(patch_option)
    skyvaultaziint = torch.tensor(
        [360 / patches for patches in aziinterval], device=device
    )
    iazimuth = torch.zeros((1, torch.sum(aziinterval).item()), device=device)

    index = 0
    for j in range(skyvaultaltint.shape[0]):
        for k in range(int(360 / skyvaultaziint[j])):
            iazimuth[0, index] = k * skyvaultaziint[j] + azistart[j]
            if iazimuth[0, index] > 360.0:
                iazimuth[0, index] = iazimuth[0, index] - 360.0
            index += 1

    aziintervalaniso = torch.ceil(aziinterval / 2.0)

    patches = []
    rings = []
    index = 0
    for i in range(skyvaultaltint.shape[0]):
        ring_patches = int(aziinterval[i].int())
        weights = tuple(
            (
                annulus_weight(k, aziinterval[i], device),
                annulus_weight(k, aziintervalaniso[i], device),
            )
            for k in range(annulino[i] + 1, annulino[i + 1] + 1)
        )
        rings.append((skyvaultaltint[i], ring_patches, weights))
        for _j in range(ring_patches):
            patches.append((skyvaultaltint[i], iazimuth[0, index], i))
            index += 1

    cached = (tuple(patches), tuple(rings))
    _SKY_PATCH_GEOMETRY[patch_option] = cached
    return cached


def _fold_veg_svf_from_planes(
    *,
    vegshmat: torch.Tensor,
    vbshvegshmat: torch.Tensor,
    vegdem2_read: torch.Tensor,
    svf_building_window: torch.Tensor,
    acc_r0: int,
    acc_r1: int,
    acc_c0: int,
    acc_c1: int,
    rows: int,
    cols: int,
) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fold per-patch vegetation planes into the SVF scalar family.

    ONE code path (R4 Phase A fold refactor) for both producers of the
    per-patch planes: the replay (:func:`_recompute_veg_svf_window`, planes
    straight from each ``shadow()`` march) and the packed-occluder-state
    path (planes unpacked from ``VegOcclusionState`` bits). The op sequence
    per accumulator is unchanged — same weights, same patch order, same
    ``last``/clamp/``svftotal`` post corrections, same zero padding — so
    both paths return bit-identical scalars for identical planes.

    ``vegshmat``/``vbshvegshmat`` carry the ACCUMULATE extent; every scalar
    is zero-padded back to the READ extent (``rows`` x ``cols``), where
    values outside the accumulate window are never consumed.
    """
    device = torch.device("cpu")
    acc_rows = acc_r1 - acc_r0
    acc_cols = acc_c1 - acc_c0

    zeros = lambda: torch.zeros((acc_rows, acc_cols), device=device)  # noqa: E731
    svfveg = zeros()
    svfEveg = zeros()
    svfSveg = zeros()
    svfWveg = zeros()
    svfNveg = zeros()
    svfaveg = zeros()
    svfEaveg = zeros()
    svfSaveg = zeros()
    svfWaveg = zeros()
    svfNaveg = zeros()

    patches, rings = _sky_patch_geometry(2)

    index = 0
    for _altitude, _ring_patches, weights in rings:
        for _j in range(_ring_patches):
            azimuth = patches[index][1]
            vegsh_acc = vegshmat[:, :, index]
            vbsh_acc = vbshvegshmat[:, :, index]

            for weight, weight_aniso in weights:
                svfveg = svfveg + weight * vegsh_acc
                svfaveg = svfaveg + weight * vbsh_acc
                if 0 <= azimuth < 180:
                    svfEveg = svfEveg + weight_aniso * vegsh_acc
                    svfEaveg = svfEaveg + weight_aniso * vbsh_acc
                if 90 <= azimuth < 270:
                    svfSveg = svfSveg + weight_aniso * vegsh_acc
                    svfSaveg = svfSaveg + weight_aniso * vbsh_acc
                if 180 <= azimuth < 360:
                    svfWveg = svfWveg + weight_aniso * vegsh_acc
                    svfWaveg = svfWaveg + weight_aniso * vbsh_acc
                if azimuth >= 270 or azimuth < 90:
                    svfNveg = svfNveg + weight_aniso * vegsh_acc
                    svfNaveg = svfNaveg + weight_aniso * vbsh_acc

            index += 1

    last = torch.zeros((acc_rows, acc_cols), device=device)
    last[vegdem2_read[acc_r0:acc_r1, acc_c0:acc_c1] == 0.0] = 3.0459e-004
    svfSveg = svfSveg + last
    svfWveg = svfWveg + last
    svfSaveg = svfSaveg + last
    svfWaveg = svfWaveg + last
    svfveg[svfveg > 1.0] = 1.0
    svfEveg[svfEveg > 1.0] = 1.0
    svfSveg[svfSveg > 1.0] = 1.0
    svfWveg[svfWveg > 1.0] = 1.0
    svfNveg[svfNveg > 1.0] = 1.0
    svfaveg[svfaveg > 1.0] = 1.0
    svfEaveg[svfEaveg > 1.0] = 1.0
    svfSaveg[svfSaveg > 1.0] = 1.0
    svfWaveg[svfWaveg > 1.0] = 1.0
    svfNaveg[svfNaveg > 1.0] = 1.0

    trans = torch.tensor(0.03, device=device)
    svftotal = (
        svf_building_window[acc_r0:acc_r1, acc_c0:acc_c1]
        - (1 - svfveg)
        * (1 - trans)
    )

    acc_slice = (slice(acc_r0, acc_r1), slice(acc_c0, acc_c1))

    def _read_extent(accumulator: torch.Tensor) -> torch.Tensor:
        """Zero-pad an accumulator back to the read extent (inert padding:
        values outside the accumulate window are never consumed)."""
        if acc_rows == rows and acc_cols == cols:
            return accumulator
        padded = torch.zeros((rows, cols), device=device)
        padded[acc_slice] = accumulator
        return padded

    veg_scalars = {
        "svfveg": _read_extent(svfveg),
        "svfEveg": _read_extent(svfEveg),
        "svfSveg": _read_extent(svfSveg),
        "svfWveg": _read_extent(svfWveg),
        "svfNveg": _read_extent(svfNveg),
        "svfaveg": _read_extent(svfaveg),
        "svfEaveg": _read_extent(svfEaveg),
        "svfSaveg": _read_extent(svfSaveg),
        "svfWaveg": _read_extent(svfWaveg),
        "svfNaveg": _read_extent(svfNaveg),
    }
    return (
        veg_scalars,
        vegshmat,
        vbshvegshmat,
        _read_extent(svftotal),
    )


def _recompute_veg_svf_window(
    cache: SiteCache,
    scene: FullSceneTensors,
    window: RasterWindow,
    svf_building_window: torch.Tensor,
    *,
    accumulate_window: RasterWindow | None = None,
) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replay ``svf_calculator``'s accumulation for the vegetation terms.

    The loop below mirrors ``svf_calculator`` (shadow.py) patch for patch:
    same ``create_patches`` layout, same ``iazimuth`` construction, same
    annulus weights (aniso weights for the directional accumulators), same
    accumulation order, same post-corrections and clamps. Only the
    vegetation-dependent accumulators and the vegetation patch cubes are
    produced; the building terms are sliced from the cache instead (the
    underlying per-patch building shadow is recomputed as part of each
    ``shadow()`` call and simply not accumulated).

    Perf wave 2, both levers bit-identical by construction:

    * ``accumulate_window`` (default ``None`` = ``window``) narrows the
      spatial extent of every accumulator and patch cube to the extent the
      physics consumes, and each patch's ``shadow()`` march runs on the
      accumulate window expanded by the patch's march reach in its READ
      directions only (:func:`_patch_march_window`), gated off by any
      nonzero bush cell (:func:`_bush_free_marches`). Cells inside the
      accumulate window see the identical step sequence and reads as the
      full-window march (see :func:`_patch_march_window`).
    * the sky-patch geometry and annulus weights are memoized constants
      (:func:`_sky_patch_geometry`) instead of being rebuilt per solve and
      per patch.

    The returned scalars keep the READ extent as before: accumulators are
    zero-padded back into read-extent tensors. Values outside the
    accumulate window are never consumed (``run_utci_window`` crops every
    scalar to its write window at entry), so the padding is inert.
    """
    device = torch.device("cpu")
    scale = 1.0 / float(cache.pixel_size_m)
    a = _crop(scene.a, window).clone()
    vegdem = _crop(scene.vegdsm, window).clone()
    vegdem2 = _crop(scene.vegdsm2, window).clone()
    bush = _crop(scene.bush, window).clone()
    # MEDIUM-2 invariant: the march regime is a function of the FULL-TILE
    # composed scene, never of this window. The crop's relative bound drops
    # whenever the exact occluder mask excludes the site's tallest surface,
    # and a bound below a patch's first-step distance flips that patch into
    # the one-step regime (step-1 accumulator zero -> vbsh 2.0) that the
    # oracle's full-tile march never runs. Compute the amplitude from the
    # full-tile tensors — the same expression as the oracle's stop choice
    # and veg_svf_state._scene_amplitude.
    #
    # R6 escalation (adjudicated, replaces the W2 refusal): patches whose
    # first step lands in (A_eff, A_abs] march at the ABSOLUTE stop
    # ``scene.amaxvalue`` — the very tensor value ``svf_calculator`` passes
    # — so their executed step set is identical to the oracle's and the
    # serve is bitwise-equal BY CONSTRUCTION. Never A_eff (the disproved
    # approximation). Non-banded patches keep the early-exit amplitude:
    # their skipped steps (dz > full-tile A_eff >= any crop bound) are
    # provably inert, and dz_1 <= A_eff keeps them multi-step like the
    # oracle (dz_1 > A_abs keeps them one-step like the oracle).
    amaxvalue = effective_march_amplitude(
        scene.a, scene.vegdsm, scene.vegdsm2, scene_amaxvalue=scene.amaxvalue
    )
    escalated = _banded_patch_mask(
        float(amaxvalue), float(scene.amaxvalue), scale
    )

    rows, cols = a.shape
    if accumulate_window is None:
        accumulate_window = window
    acc_r0 = accumulate_window.row_start - window.row_start
    acc_r1 = accumulate_window.row_stop - window.row_start
    acc_c0 = accumulate_window.col_start - window.col_start
    acc_c1 = accumulate_window.col_stop - window.col_start
    acc_rows = acc_r1 - acc_r0
    acc_cols = acc_c1 - acc_c0

    patches, rings = _sky_patch_geometry(2)
    n_patches = len(patches)
    vegshmat = torch.zeros((acc_rows, acc_cols, n_patches), device=device)
    vbshvegshmat = torch.zeros((acc_rows, acc_cols, n_patches), device=device)

    # Lever 1: per-patch march windows. One reach per altitude ring, one
    # window per (ring, read-direction) — patches in the same ring with the
    # same quadrant march the identical grid. A ring carrying any escalated
    # patch takes the ABSOLUTE amplitude's reach: over-covering a march
    # window is harmless (the march is a pure per-cell function of its
    # input rasters), while the escalated patches march to the absolute
    # stop inside it and non-banded patches in the ring still stop at
    # their own effective amplitude (the stop is per-call, not per-grid).
    march_narrowed = _bush_free_marches(bush)
    ring_amplitude: list[torch.Tensor] = []
    _ring_start = 0
    for _altitude, ring_patches, _weights in rings:
        ring_escalated = bool(
            escalated[_ring_start : _ring_start + ring_patches].any()
        )
        ring_amplitude.append(
            scene.amaxvalue if ring_escalated else amaxvalue
        )
        _ring_start += int(ring_patches)
    ring_reach = [
        _patch_march_reach_pixels(float(amp), scale, float(altitude))
        for amp, (altitude, _count, _weights) in zip(ring_amplitude, rings)
    ]
    march_grids: dict[tuple[int, int, int], tuple[tuple[slice, slice], tuple[slice, slice]]] = {}

    def _patch_grid(ring: int, azimuth) -> tuple[tuple[slice, slice], tuple[slice, slice]]:
        """(march slices, accumulate-inside-march slices), window-relative."""
        full = (slice(None), slice(None))
        if not march_narrowed:
            return full, (slice(acc_r0, acc_r1), slice(acc_c0, acc_c1))
        key = (ring, *_quadrant_read_direction(float(azimuth)))
        entry = march_grids.get(key)
        if entry is None:
            march_window = _patch_march_window(
                accumulate_window,
                window,
                azimuth_deg=float(azimuth),
                reach_pixels=ring_reach[ring],
            )
            grid = (
                slice(
                    march_window.row_start - window.row_start,
                    march_window.row_stop - window.row_start,
                ),
                slice(
                    march_window.col_start - window.col_start,
                    march_window.col_stop - window.col_start,
                ),
            )
            inside = (
                slice(
                    accumulate_window.row_start - march_window.row_start,
                    accumulate_window.row_stop - march_window.row_start,
                ),
                slice(
                    accumulate_window.col_start - march_window.col_start,
                    accumulate_window.col_stop - march_window.col_start,
                ),
            )
            entry = (grid, inside)
            march_grids[key] = entry
        return entry

    index = 0
    for i, (altitude, ring_patches, _weights) in enumerate(rings):
        for _j in range(ring_patches):
            azimuth = patches[index][1]
            grid, inside = _patch_grid(i, azimuth)
            # R6: banded patches march at the oracle's absolute stop (the
            # same tensor value svf_calculator passes) — their executed
            # step set IS the oracle's. Non-banded patches keep the
            # early-exit amplitude and may stop EARLIER than the oracle:
            # their skipped steps (dz above the full-tile effective
            # bound) are value-inert (effective_march_amplitude's
            # argument), so the step SETS can differ while the VALUES
            # cannot. shadow()'s index-1 accumulator zero makes the stop
            # REGIME itself value-load-bearing — pinned against the true
            # svf_calculator oracle by TestEscalationSensitivity.
            patch_amplitude = (
                scene.amaxvalue if bool(escalated[index]) else amaxvalue
            )
            # T19b: the per-patch march runs through the registered lane
            # router — the bit-pinned numba march (T04 kernel + R3 general
            # step tables keyed on THIS grid's shape and amplitude) when
            # the scene is inside its domain, the torch kernel byte-
            # identically on refusal or the declared torch lane. The torch
            # callable is handed in from THIS module's binding so test
            # spies on ``solver.shadow_fn`` keep witnessing the fallback
            # lane, and the amplitude policy records which R6 policy chose
            # the executed stop (escalated patches march at the oracle's
            # absolute ``scene.amaxvalue``).
            sh, vegsh, vbshvegsh = march_router.shadow_march(
                patch_amplitude,
                a[grid[0], grid[1]],
                vegdem[grid[0], grid[1]],
                vegdem2[grid[0], grid[1]],
                bush[grid[0], grid[1]],
                azimuth,
                altitude,
                scale,
                amplitude_policy=(
                    "r6_band_escalated"
                    if bool(escalated[index])
                    else "effective_windowed"
                ),
                torch_shadow=shadow_fn,
            )
            del sh

            vegshmat[:, :, index] = vegsh[inside[0], inside[1]]
            vbshvegshmat[:, :, index] = vbshvegsh[inside[0], inside[1]]
            index += 1

    return _fold_veg_svf_from_planes(
        vegshmat=vegshmat,
        vbshvegshmat=vbshvegshmat,
        vegdem2_read=vegdem2,
        svf_building_window=svf_building_window,
        acc_r0=acc_r0,
        acc_r1=acc_r1,
        acc_c0=acc_c0,
        acc_c1=acc_c1,
        rows=rows,
        cols=cols,
    )


# ---------------------------------------------------------------------------
# Stage 8: exact window solve
# ---------------------------------------------------------------------------


def _check_window_nesting(
    write_window: RasterWindow, read_window: RasterWindow, cache: SiteCache
) -> None:
    full = RasterWindow(0, cache.rows, 0, cache.cols)
    for label, window in (("read", read_window), ("write", write_window)):
        if window.is_empty:
            raise SolverInputError(f"{label} window is empty: {window}")
        if (
            window.row_start < 0
            or window.col_start < 0
            or window.row_stop > cache.rows
            or window.col_stop > cache.cols
        ):
            raise SolverInputError(
                f"{label} window {window} exceeds the site grid {full}"
            )
    nested = (
        read_window.row_start <= write_window.row_start
        and read_window.row_stop >= write_window.row_stop
        and read_window.col_start <= write_window.col_start
        and read_window.col_stop >= write_window.col_stop
    )
    if not nested:
        raise SolverInputError(
            f"write window {write_window} is not contained in read window "
            f"{read_window}"
        )


def _check_baseline_amplitude(scene: FullSceneTensors, cache: SiteCache) -> None:
    """Refuse local solves that change the global march amplitude.

    The cached building-only SVF terms (and the baseline amaxvalue they were
    marched with) are only scene-invariant while the composed scene does not
    change the global march amplitude. A taller tree raises the GLOBAL
    ray-march truncation bound, so building shadows far from the edit can
    change in the oracle while the cache still holds the shorter-march
    values. An amplitude-LOWERING edit (deleting the dominant tree)
    shortens the oracle's march: the extra cached steps only contribute
    where an occluder still exceeds the target's surface, which requires
    below-datum (negative) surfaces — guarded for those sites too. Any
    amplitude change on such a site is unsafe locally: refuse and let the
    worker fall back to full tile.
    """
    baseline_canopy = torch.from_numpy(np.array(cache.tree_base))
    baseline_canopy[baseline_canopy < 0.0] = 0.0
    baseline_amaxvalue = torch.maximum(
        scene.a.max(), (baseline_canopy + scene.dem).max()
    )
    if bool(scene.amaxvalue != baseline_amaxvalue) and (
        bool(scene.amaxvalue > baseline_amaxvalue)
        or float(scene.a.min()) < 0.0
    ):
        raise SolverInputError(
            f"scene amaxvalue ({float(scene.amaxvalue)}) differs from the "
            f"cached baseline ({float(baseline_amaxvalue)}); building-only "
            "SVF terms would be stale — full-tile recompute required"
        )


#: The cross-timestep thermal planes a G2.1 warm start threads through
#: ``solve_window`` — exactly the state ``run_utci_window`` allocates and
#: returns under ``return_final_state`` (see utci_process.py; the carried
#: scalars ``CI``/``firstdaytime``/``timeadd``/``Twater`` ride alongside).
_THERMAL_STATE_PLANES = (
    "Tgmap1", "Tgmap1E", "Tgmap1S", "Tgmap1W", "Tgmap1N", "TgOut1",
)


def solve_window(
    cache: SiteCache,
    layer: TreeLayer,
    *,
    read_window: RasterWindow,
    write_window: RasterWindow,
    forcing: SiteForcing,
    requested_variables: Sequence[str] = ("utci", "tmrt", "shadow"),
    veg_changed: bool | None = None,
    landcover_overlay: LandCoverOverlay | None = None,
    model_parameters: Mapping[str, Any] | None = None,
    time_start: int = 0,
    time_stop: int | None = None,
    stage_timings: MutableMapping[str, float] | None = None,
    scene: FullSceneTensors | None = None,
    required_read_window: RasterWindow | None = None,
    veg_state=None,
    initial_state: Mapping[str, Any] | None = None,
    return_final_state: bool = False,
) -> dict[str, np.ndarray] | tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Evaluate the exact SOLWEIG physics over the read window.

    Temporal state is replayed cold from timestep 0 over the read window
    (Stage 8) and only the write-window crop is returned. Marches run with
    the window-relative :func:`effective_march_amplitude` — bit-identical to
    the oracle's scene-wide stop, just without the unreachable tail —
    escalating to the absolute ``scene.amaxvalue`` when a timestep's sun
    first step would band (:func:`time_loop_band_present`; the R6 closure
    of the latent window-amplitude time-loop note).
    ``veg_changed`` defaults to "the layer has any pending vegetation edits"
    and selects Stage-7 recompute over cached vegetation SVF terms.

    ``landcover_overlay`` (u-c4, default ``None`` = the legacy
    byte-identical cache read) is the scenario paint overlay: the resolved
    class grid is materialized once against the FULL baseline grid (so any
    read window — including halo cells outside the write window — sees every
    painted cell inside it) and the read-window slice of it is what reaches
    ``run_utci_window`` as the float32 ``landcover_grid`` tensor.
    ``model_parameters`` (u-c7, default ``None`` = the legacy byte-identical
    call) is the folded kernel-argument mapping from committed model
    parameter edits; it is forwarded verbatim to ``run_utci_window``, which
    re-fences names and applies per-name overrides at the module/loop-local
    constant sites. Like :func:`run_full_tile`, the mapping TYPE is fenced
    here (u-c7-review LOW): a non-Mapping payload is refused with
    :class:`SolverInputError` before any windowing work runs.

    ``time_stop`` (perf wave 1, default ``None`` = the full series) bounds
    the causal time-prefix replay at ``t = 0..time_stop``; every computed
    step is bit-identical to the full-series run because the temporal
    state chains only carry forward. ``stage_timings`` (optional out-param)
    accumulates ``svf_seconds`` / ``time_loop_seconds`` across calls so the
    worker can report per-stage telemetry; it never affects the solve.

    ``initial_state`` / ``return_final_state`` / ``time_start`` (G2.1,
    defaults ``None`` / ``False`` / ``0`` = the exact cold behaviour) are
    the warm-start seam. ``initial_state`` carries the FULL-TILE
    cross-timestep thermal state captured at ``next_step == time_start``
    (the six planes + ``CI``/``firstdaytime``/``timeadd``/``Twater`` from
    :func:`solweig_gpu.utci_process.run_utci_window`'s
    ``return_final_state``); the planes are sliced to the write window
    here (the thermal chain is elementwise, so a windowed warm state is
    the crop of the full-tile trajectory) and handed to the windowed core,
    which re-fences extent/``next_step`` loudly. The parity fence
    (tests/test_incremental_warm_start.py) licenses the seam:
    ``warm(k..N, state@k)`` is bitwise ``cold(0..N)[k..N]``. With
    ``return_final_state`` the return is ``(outputs, final_state)`` where
    the state planes sit at the WRITE-window extent — only full-tile
    solves (read == write == the grid) produce persistable states.

    ``scene`` and ``required_read_window`` (STEP 2b dedup, default
    ``None`` = computed here) let a caller that ALREADY composed the scene
    tensors and derived the exact-influence read window hand both in: the
    worker computes them for its halo guard, and this seam consumes the
    same values instead of recomputing them (identical deterministic
    functions of the same inputs, so the guard keeps checking the very
    value it would have computed). The patch cubes in the SVF bundle are
    sliced at the WRITE window here (STEP 2b): ``run_utci_window`` crops
    them to exactly that extent before use, so the narrower memmap slice
    is pure data movement, bit-identical to the legacy late crop.
    ``veg_state`` (R4 Phase A, default ``None``) forwards to
    :func:`window_svf_bundle` — the vegetation SVF terms are folded from
    the packed occluder bits instead of replaying the 153-patch march.
    """
    if model_parameters is not None and not isinstance(
        model_parameters, Mapping
    ):
        raise SolverInputError(
            "model_parameters must be a name -> value mapping, got "
            f"{type(model_parameters).__name__}"
        )
    if time_stop is not None:
        # Causal-prefix fence (perf wave 1): the replay must cover at least
        # one timestep and never exceed the forcing series.
        total_steps = int(np.asarray(forcing.met_table).shape[0])
        if not 0 < int(time_stop) <= total_steps:
            raise SolverInputError(
                f"time_stop {time_stop} outside the causal prefix range "
                f"[1, {total_steps}] for this forcing series"
            )
    if time_start != 0 or initial_state is not None:
        # G2.1 warm-start fence: a suffix replay must start inside the
        # series and actually cover steps (time_stop None = full series).
        total_steps = int(np.asarray(forcing.met_table).shape[0])
        effective_stop = (
            total_steps if time_stop is None else int(time_stop)
        )
        if not 0 <= int(time_start) < effective_stop:
            raise SolverInputError(
                f"time_start {time_start} outside the replay range "
                f"[0, {effective_stop}) for this forcing series"
            )
    if initial_state is not None and not isinstance(
        initial_state, Mapping
    ):
        raise SolverInputError(
            "initial_state must be a mapping of thermal planes + scalars, "
            f"got {type(initial_state).__name__}"
        )
    _check_window_nesting(write_window, read_window, cache)
    if scene is None:
        scene = compose_full_scene_tensors(cache, layer)

    # Cached building-only SVF terms are only valid while the global march
    # amplitude is unchanged (see _check_baseline_amplitude).
    _check_baseline_amplitude(scene, cache)

    grid = RasterGrid(
        rows=cache.rows,
        cols=cache.cols,
        pixel_size_m=cache.pixel_size_m,
        origin_x_m=cache.manifest.origin_x_m,
        origin_y_m=cache.manifest.origin_y_m,
    )
    if grid.rows != layer.grid.rows or grid.cols != layer.grid.cols:
        raise SolverInputError("tree layer grid does not match the cache grid")

    # Halo sufficiency check: refuse silently-incorrect local solves.
    # STEP 2b dedup: a caller that already derived the exact-influence
    # window (the worker does, for its own guard) hands it in; the value
    # is the same deterministic function of the same inputs.
    required = (
        required_read_window
        if required_read_window is not None
        else read_window_for_write_window(write_window, cache, scene, forcing)
    )
    if (
        read_window.row_start > required.row_start
        or read_window.row_stop < required.row_stop
        or read_window.col_start > required.col_start
        or read_window.col_stop < required.col_stop
    ):
        raise SolverInputError(
            f"read window {read_window} is smaller than the exact-influence "
            f"window {required}; the local solve would not be exact"
        )

    a = _crop(scene.a, read_window).clone()
    temp1 = _crop(scene.canopy, read_window).clone()  # run_utci_window mutates temp1
    temp2 = _crop(scene.dem, read_window).clone()
    # Bit-identical early exit for the time-loop marches: run_utci_window
    # clamps temp1 to be non-negative, then rebuilds vegdsm/vegdsm2 from
    # these same crops (temp1 + a with the ==a -> 0 quirk), so the
    # window-relative amplitude bounds every march comparison the oracle
    # makes with the scene-wide amaxvalue.
    temp1_clamped = torch.where(
        temp1 < 0.0, torch.zeros((), dtype=temp1.dtype), temp1
    )
    march_amplitude = effective_march_amplitude(
        a,
        temp1_clamped + a,  # vegdsm pre-quirk; the quirk only lowers cells
        temp1_clamped * 0.25 + a,
        scene_amaxvalue=scene.amaxvalue,
    )
    # R6 (latent-note closure): a timestep whose SUN first-step distance
    # lands in (windowed effective, absolute] would run one step at the
    # windowed amplitude where the oracle's absolute stop runs several —
    # a step-regime difference that was immune only through a
    # missing-accumulator quirk of shadowingfunction_wallheight_23.
    # Escalate the whole loop to the same absolute-amplitude source the
    # SVF escalation uses (scene.amaxvalue, the oracle's own stop); the
    # non-banded timesteps marching farther is value-inert (their extra
    # steps sit above the marched arrays' relative bound, the W2
    # inertness argument), and the loop's executed step set becomes the
    # oracle's per timestep by construction.
    #
    # Value note (r6-review mutation testing): at a banded timestep the
    # extra bodies are provably all-zero raises (every occluder in the
    # marched crop sits below the windowed amplitude, hence below dz_1)
    # and this kernel re-derives its final vbsh from vegsh alone — no
    # shadow()-style index-1 accumulator zero — so the escalation is
    # regime-parity insurance, not a value change. A byte-flip fence for
    # it cannot exist; the inertness itself is pinned by
    # test_banded_timestep_marches_are_value_inert.
    _total_steps = int(np.asarray(forcing.met_table).shape[0])
    _effective_stop = (
        _total_steps if time_stop is None else int(time_stop)
    )
    if time_loop_band_present(
        float(march_amplitude),
        float(scene.amaxvalue),
        1.0 / float(cache.pixel_size_m),
        forcing.altitude,
        forcing.azimuth,
        time_start=int(time_start),
        time_stop=_effective_stop,
    ):
        march_amplitude = scene.amaxvalue
    walls = torch.from_numpy(
        np.array(cache.window("walls", read_window))
    )
    dirwalls = torch.from_numpy(
        np.array(cache.window("wall_aspect", read_window))
    )

    if veg_changed is None:
        veg_changed = layer.coalesced_batch() is not None
    _svf_started = time.perf_counter()
    _march_before = march_router.lane_stats()
    svf_bundle = window_svf_bundle(
        cache, scene, read_window, veg_changed=veg_changed,
        cube_window=write_window, veg_state=veg_state,
    )
    if stage_timings is not None:
        stage_timings["svf_seconds"] = (
            stage_timings.get("svf_seconds", 0.0)
            + (time.perf_counter() - _svf_started)
        )
        # T19b routing telemetry (floats only — the stage-timings channel
        # is typed float): per-solve deltas of the router's process-level
        # counters, so concurrent solves never misattribute each other's
        # marches. ACCUMULATE (+=): the worker folds the prepare-side
        # corridor counts in first, and solve_window itself runs once per
        # write window on a shared stage dict. Refusal REASONS ride the
        # router's lane_stats.
        _march_after = march_router.lane_stats()
        stage_timings["svf_march_routed"] = (
            stage_timings.get("svf_march_routed", 0.0)
            + float(
                _march_after["numba_routed"] - _march_before["numba_routed"]
            )
        )
        stage_timings["svf_march_refused"] = (
            stage_timings.get("svf_march_refused", 0.0)
            + float(
                _march_after["numba_refused"] - _march_before["numba_refused"]
            )
        )

    landcover_grid = None
    lc_class = None
    if landcover_overlay is not None:
        # u-c4: resolve against the FULL baseline grid (the masked re-check
        # runs inside, and a cache without a landcover raster is refused —
        # the paint could never reach the physics) and slice the read
        # window out of it, so halo cells outside the write window see
        # painted cells exactly like the full-domain oracle would.
        resolved_lc = resolve_landcover_overlay(cache, landcover_overlay)
        landcover_grid = torch.from_numpy(
            np.array(
                resolved_lc[
                    read_window.row_start : read_window.row_stop,
                    read_window.col_start : read_window.col_stop,
                ],
                dtype=np.float32,
            )
        )
        lc_class = load_landcover_classes()
    elif "landcover" in cache:
        # Cast to float32: the oracle's load_raster_to_tensor also produces a
        # float32 tensor, and Tgmaps_v1 copies the grid to build the surface
        # property maps — a uint8 grid would truncate emissivity/albedo to 0.
        landcover_grid = torch.from_numpy(
            np.array(cache.window("landcover", read_window), dtype=np.float32)
        )
        lc_class = load_landcover_classes()

    _loop_started = time.perf_counter()
    warm_state = None
    if initial_state is not None:
        # Slice the FULL-TILE captured planes to the write window (the
        # thermal chain is elementwise: the windowed warm state is the crop
        # of the full-tile trajectory). Extent/next_step fences run inside
        # the windowed core on the sliced planes.
        warm_state = dict(initial_state)
        for name in _THERMAL_STATE_PLANES:
            try:
                plane = initial_state[name]
            except KeyError as error:
                raise SolverInputError(
                    f"initial_state is missing the thermal plane {name!r}"
                ) from error
            if isinstance(plane, torch.Tensor):
                pass
            elif isinstance(plane, np.ndarray):
                plane = torch.from_numpy(np.ascontiguousarray(plane))
            else:
                raise SolverInputError(
                    f"initial_state plane {name!r} must be a tensor or "
                    f"ndarray, got {type(plane).__name__}"
                )
            if tuple(plane.shape) != (cache.rows, cache.cols):
                raise SolverInputError(
                    f"initial_state plane {name!r} shape "
                    f"{tuple(plane.shape)} is not the full-tile grid "
                    f"{(cache.rows, cache.cols)} — capture states are "
                    "full-tile, windowed consumers slice here"
                )
            warm_state[name] = plane[
                write_window.row_start : write_window.row_stop,
                write_window.col_start : write_window.col_stop,
            ]
    outputs = run_utci_window(
        a=a,
        temp1=temp1,
        temp2=temp2,
        walls=walls,
        dirwalls=dirwalls,
        svf_bundle=svf_bundle,
        met_file=forcing.met_table,
        altitude=forcing.altitude,
        azimuth=forcing.azimuth,
        zen=forcing.zen,
        jday=forcing.jday,
        dectime=forcing.dectime,
        altmax=forcing.altmax,
        location=forcing.location,
        scale=1.0 / float(cache.pixel_size_m),
        amaxvalue=march_amplitude,
        landcover_grid=landcover_grid,
        lc_class=lc_class,
        windcoeff=None,
        windcoeff_by_dir=None,
        time_start=time_start,
        time_stop=time_stop,
        requested_variables=tuple(requested_variables),
        save_wbgt=False,
        model_parameters=model_parameters,
        initial_state=warm_state,
        return_final_state=return_final_state,
        svf_bundle_cubes_cropped=True,
        out_window=(
            write_window.row_start - read_window.row_start,
            write_window.row_stop - read_window.row_start,
            write_window.col_start - read_window.col_start,
            write_window.col_stop - read_window.col_start,
        ),
    )
    if return_final_state:
        outputs, final_state = outputs
    if stage_timings is not None:
        stage_timings["time_loop_seconds"] = (
            stage_timings.get("time_loop_seconds", 0.0)
            + (time.perf_counter() - _loop_started)
        )

    # With out_window the arrays already have the write-window shape.
    results = {
        name: np.ascontiguousarray(array) for name, array in outputs.items()
    }
    if return_final_state:
        # G2.1: the captured state planes sit at the WRITE-window extent
        # (a crop of the full-tile trajectory); only full-tile solves
        # (read == write == the grid) yield persistable states.
        return results, final_state
    return results


# ---------------------------------------------------------------------------
# Full-tile fallback / oracle (standard full path, fresh SVF)
# ---------------------------------------------------------------------------

_SCRATCH_DIRS = (
    "Building_DSM",
    "DEM",
    "Trees",
    "walls",
    "aspect",
    "Landcover",
    "metfiles",
)


def _site_raster_path(site_dir: Path, kind: str, tile_key: str) -> Path:
    return site_dir / kind / f"{kind}_{tile_key}.tif"


def _write_resolved_met(path: Path, table: np.ndarray) -> None:
    """Write an overlay-resolved forcing table as a loadable met text file.

    ``%.17g`` round-trips float64 exactly, and the loader parses with
    ``np.loadtxt(skiprows=1, delimiter=" ")`` — so the staged text re-reads
    to a table bitwise equal to the resolved array the solver actually ran
    with. The scratch prepared-site layout keeps a met file at this path;
    materializing the resolved table there (instead of copying the baseline
    ``met_path``) is what keeps a scenario overlay visible on the full-tile
    path (u-c3: an overlay must never be dropped to the baseline text).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Header states the layout it actually wrote. It once listed 15 column
    # names for the 25-column met layout (u-c3 L5): a reader trusting the
    # names over the row width would mis-map every column from 15 on. The
    # count is derived from the table itself, so it can never drift again.
    columns = int(np.asarray(table).shape[1])
    header = (
        f"# overlay-resolved scenario met table (u-c3); {columns} columns "
        "per row (baseline met layout: preprocessor.py:1130-1146); "
        "row width is authoritative, not this note"
    )
    np.savetxt(
        path,
        np.asarray(table, dtype=np.float64),
        fmt="%.17g",
        delimiter=" ",
        header=header,
        comments="",
    )


def _run_full_tile_with_model_parameters(
    cache: SiteCache,
    *,
    forcing: SiteForcing,
    scratch: Path,
    tile_key: str,
    building_name: str,
    requested: Sequence[str],
    model_parameters: Mapping[str, Any],
    return_final_state: bool = False,
) -> dict[str, np.ndarray] | tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Run the staged scratch site through ``run_utci_window`` with overrides.

    Line-for-line replication of ``compute_utci``'s core (utci_process.py):
    the same scratch-raster loads, the same land-cover cleaning (invalid → 6,
    vegetation → 5), the same ``vegdem``/``bush``/``vegdsm``/``amaxvalue``
    expressions, and the same fresh ``svf_calculator(patch_option=2, ...)``
    pass — only the GeoTIFF write/read round-trip is skipped, because
    ``run_utci_window`` already returns the exact float32 stacks the legacy
    path would read back (the write is lossless float32). Site-level scalars
    come from ``forcing`` (``load_site_forcing`` re-derives them the oracle
    way and validates them against the manifest), and ``scale`` comes from
    the same scratch Building_DSM geotransform the oracle reads. The only
    behavioural difference from the legacy path is
    ``model_parameters=model_parameters`` on the windowed core.
    """
    a, dataset = load_raster_to_tensor(str(scratch / "Building_DSM" / building_name))
    temp1, _ = load_raster_to_tensor(str(scratch / "Trees" / f"Trees_{tile_key}.tif"))
    temp2, _ = load_raster_to_tensor(str(_site_raster_path(scratch, "DEM", tile_key)))
    walls, _ = load_raster_to_tensor(str(_site_raster_path(scratch, "walls", tile_key)))
    dirwalls, _ = load_raster_to_tensor(
        str(_site_raster_path(scratch, "aspect", tile_key))
    )
    try:
        scale = 1 / dataset.GetGeoTransform()[1]
    except (RuntimeError, AttributeError, IndexError) as error:
        raise SolverInputError(
            f"could not read the scratch Building_DSM geotransform: {error}"
        ) from error
    finally:
        dataset = None

    lcgrid_torch = None
    lc_class = None
    landcover_path = _site_raster_path(scratch, "Landcover", tile_key)
    if landcover_path.is_file():
        # compute_utci's cleaning rules, applied to the same staged raster
        # (baseline copy or u-c4 resolved paint) the legacy path would load.
        lcgrid_torch, _ = load_raster_to_tensor(str(landcover_path))
        lcgrid_np = lcgrid_torch.cpu().numpy()
        mask_invalid = (lcgrid_np < 1) | (lcgrid_np > 7)
        if mask_invalid.any():
            lcgrid_np[mask_invalid] = 6
        mask_vegetation = (lcgrid_np == 3) | (lcgrid_np == 4)
        if mask_vegetation.any():
            lcgrid_np[mask_vegetation] = 5
        lcgrid_torch = torch.as_tensor(
            lcgrid_np, device=lcgrid_torch.device, dtype=lcgrid_torch.dtype
        )
        lc_class = load_landcover_classes()

    if a.shape != (cache.rows, cache.cols):
        raise SolverInputError(
            f"scratch Building_DSM shape {tuple(a.shape)} does not match the "
            f"cache grid {(cache.rows, cache.cols)}"
        )

    # compute_utci's vegetation expressions, in its order (temp1 clamp first:
    # every downstream tensor reads the clamped canopy).
    temp1[temp1 < 0.0] = 0.0
    vegdem = temp1 + temp2
    vegdem2 = torch.add(temp1 * 0.25, temp2)
    bush = torch.logical_not(vegdem2 * vegdem) * vegdem
    vegdsm = temp1 + a
    vegdsm[vegdsm == a] = 0
    vegdsm2 = temp1 * 0.25 + a
    vegdsm2[vegdsm2 == a] = 0
    amaxvalue = torch.maximum(a.max(), vegdem.max())

    # Fresh SVF over the composed scene — the same pass compute_utci runs on
    # a freshly staged scratch site (its SVF cache dir never exists there).
    # save_rasters=False skips only the GeoTIFF writes, never the math.
    svf_bundle = svf_calculator(
        2, amaxvalue, a, vegdsm, vegdsm2, bush, scale, save_rasters=False
    )

    outputs = run_utci_window(
        a=a,
        temp1=temp1,
        temp2=temp2,
        walls=walls,
        dirwalls=dirwalls,
        svf_bundle=svf_bundle,
        met_file=forcing.met_table,
        altitude=forcing.altitude,
        azimuth=forcing.azimuth,
        zen=forcing.zen,
        jday=forcing.jday,
        dectime=forcing.dectime,
        altmax=forcing.altmax,
        location=forcing.location,
        scale=scale,
        amaxvalue=amaxvalue,
        landcover_grid=lcgrid_torch,
        lc_class=lc_class,
        windcoeff=None,
        windcoeff_by_dir=None,
        requested_variables=tuple(requested),
        save_wbgt=False,
        model_parameters=model_parameters,
        return_final_state=return_final_state,
    )
    if return_final_state:
        outputs, final_state = outputs

    results: dict[str, np.ndarray] = {}
    for name in ("utci", "tmrt", "kup", "kdown", "lup", "ldown", "shadow"):
        if name not in requested:
            continue
        array = outputs.get(name)
        if array is None:
            raise SolverInputError(
                f"full-tile model-parameter run did not produce {name}"
            )
        if array.shape != (cache.time_steps, cache.rows, cache.cols):
            raise SolverInputError(
                f"{name} shape {array.shape} does not match the expected "
                f"({cache.time_steps}, {cache.rows}, {cache.cols})"
            )
        results[name] = np.ascontiguousarray(array)
    if return_final_state:
        # Full-tile run (out_window is never set on this path): the state
        # planes are already the full grid — directly persistable.
        return results, final_state
    return results


def _run_full_tile_impl(
    cache: SiteCache,
    layer: TreeLayer,
    *,
    forcing: SiteForcing,
    site_dir: Path,
    scratch_dir: Path,
    requested_variables: Sequence[str] = ("utci", "tmrt", "shadow"),
    landcover_overlay: LandCoverOverlay | None = None,
    model_parameters: Mapping[str, Any] | None = None,
    return_final_state: bool = False,
) -> dict[str, np.ndarray] | tuple[dict[str, np.ndarray], dict[str, Any]]:
    tile_key = cache.tile_key
    source_building = _site_raster_path(site_dir, "Building_DSM", tile_key)
    if not source_building.is_file():
        raise SolverInputError(f"site building raster missing: {source_building}")

    # L5 (u-c4-review item 5): resolve and vet the overlay BEFORE the first
    # scratch mutation. A resolve failure must leave the previous scratch
    # untouched — never a half-staged site — and a validated overlay makes
    # the baseline Landcover copy below pure waste, because the resolved
    # grid is staged at that exact path further down. The cache check runs
    # first so a cacheless site still hears the "no landcover raster"
    # refusal (utci_process.py:608 consumes the class grid only when the
    # site carries one).
    resolved_lc = None
    if landcover_overlay is not None:
        resolved_lc = resolve_landcover_overlay(cache, landcover_overlay)
        if not (
            site_dir / "Landcover" / f"Landcover_{tile_key}.tif"
        ).is_file():
            raise SolverInputError(
                "scenario landcover overlay on a site without a Landcover "
                "raster; the paint could never reach the physics"
            )

    scratch = Path(scratch_dir)
    if scratch.exists():
        shutil.rmtree(scratch)
    for name in _SCRATCH_DIRS:
        (scratch / name).mkdir(parents=True, exist_ok=True)
    output_path = scratch / "out"
    output_path.mkdir(parents=True, exist_ok=True)

    for kind in ("Building_DSM", "DEM", "walls", "aspect", "Landcover"):
        if resolved_lc is not None and kind == "Landcover":
            # L5: skip the baseline copy the resolved grid overwrites below.
            continue
        source = _site_raster_path(site_dir, kind, tile_key)
        if source.is_file():
            shutil.copy2(source, scratch / kind / source.name)

    scene = compose_full_scene_tensors(cache, layer)
    gdal_dsm = gdal.Open(str(source_building))
    from solweig_gpu.shadow import save_raster_like_gdal

    save_raster_like_gdal(
        gdal_dsm,
        str(scratch / "Trees" / f"Trees_{tile_key}.tif"),
        scene.canopy,
    )

    # u-c4: under a scenario paint overlay the scratch prepared site stages
    # the RESOLVED class grid, never the baseline Landcover GeoTIFF (an
    # overlay must never be dropped to the baseline raster).
    #
    # L6 (u-c4-review item 6) — dtype asymmetry, deliberately kept and
    # documented rather than normalized: the site Landcover GeoTIFF is
    # uint8, while ``save_raster_like_gdal`` (shadow.py) always writes
    # float32 (``astype(np.float32)`` + ``GDT_Float32``), so the staged
    # scratch Landcover is float32 under an overlay and uint8 (verbatim
    # ``copy2``) without one. The two spellings are physics-equal: the
    # physics loads rasters through ``load_raster_to_tensor`` (float32) and
    # the class codes 1..7 are exact in float32, so the staged raster
    # re-reads bitwise equal to the resolved grid the windowed path
    # materializes. TRAP for future consumers: read the staged raster
    # through that same loader; never assume the site's uint8 dtype.
    # ``overlay=None`` keeps the verbatim file copy — byte-identical to the
    # legacy scratch site.
    if resolved_lc is not None:
        save_raster_like_gdal(
            gdal_dsm,
            str(_site_raster_path(scratch, "Landcover", tile_key)),
            resolved_lc,
        )

    # The scratch prepared-site layout carries the met file the run consumed.
    # Under a scenario overlay that is the RESOLVED table, never the baseline
    # text at ``met_path`` (u-c3: copying the baseline would silently drop
    # the overlay from the staged site record).
    met_scratch = scratch / "metfiles" / forcing.met_path.name
    if forcing.overlay is not None:
        _write_resolved_met(met_scratch, forcing.met_table)
    else:
        shutil.copy2(forcing.met_path, met_scratch)

    requested = tuple(requested_variables)
    # NOTE on requested-variable symmetry (u-c7-review LOW): the legacy
    # ``compute_utci`` path below hardwires save_kup/kdown/lup/ldown=False,
    # so flux variables requested here are silently ABSENT from the
    # returned mapping (the read-back loop carries utci/tmrt/shadow only,
    # and it refuses a missing one of those loudly). The loud refusal for
    # a legacy flux request lives one door up, at the worker's
    # ``_solve_full`` (every production caller routes through it): the
    # solver level itself must stay permissive because the sanctioned
    # oracle-helper pattern (tests/test_incremental_params_integration
    # ``_direct_run``) requests the full variable vocabulary through BOTH
    # branches and reads only the legacy-producible subset from the legacy
    # result.
    if model_parameters or return_final_state:
        # u-c7: ``compute_utci`` has no ``model_parameters`` seam (it is the
        # upstream orchestrator and out of scope for this packet), so a
        # scenario with model parameter overrides runs the SAME staged
        # scratch site through the windowed core directly — a faithful
        # replication of ``compute_utci``'s pre-``run_utci_window`` core
        # (same raster loads, same landcover cleaning, same vegdem/bush/
        # amaxvalue expressions, same fresh ``svf_calculator`` pass) with
        # the folded overrides handed to ``run_utci_window``. An empty or
        # ``None`` mapping never reaches this branch: the legacy
        # ``compute_utci`` call below stays byte-identical.
        #
        # G2.1: ``return_final_state`` routes here too — the legacy
        # orchestrator has no thermal-state return seam, while this path
        # IS the documented replication of it (bit-identical outputs; the
        # GeoTIFF write/read round-trip it skips is lossless float32). The
        # warm==cold and direct==legacy parity fences
        # (tests/test_incremental_warm_start.py) pin that license.
        return _run_full_tile_with_model_parameters(
            cache,
            forcing=forcing,
            scratch=scratch,
            tile_key=tile_key,
            building_name=source_building.name,
            requested=requested,
            model_parameters=model_parameters or {},
            return_final_state=return_final_state,
        )
    compute_utci(
        building_dsm_path=str(scratch / "Building_DSM" / source_building.name),
        tree_path=str(scratch / "Trees" / f"Trees_{tile_key}.tif"),
        dem_path=str(_site_raster_path(scratch, "DEM", tile_key)),
        walls_path=str(_site_raster_path(scratch, "walls", tile_key)),
        aspect_path=str(_site_raster_path(scratch, "aspect", tile_key)),
        landcover_path=(
            str(_site_raster_path(scratch, "Landcover", tile_key))
            if _site_raster_path(scratch, "Landcover", tile_key).is_file()
            else None
        ),
        windcoeff_path=None,
        met_file=forcing.met_table,
        output_path=str(output_path),
        number=tile_key,
        selected_date_str=forcing.selected_date_str,
        save_tmrt="tmrt" in requested,
        save_svf=False,
        save_kup=False,
        save_kdown=False,
        save_lup=False,
        save_ldown=False,
        save_shadow="shadow" in requested,
        save_wbgt=False,
        save_ta=False,
        save_wind=False,
    )

    results: dict[str, np.ndarray] = {}
    for name, prefix in (("utci", "UTCI"), ("tmrt", "TMRT"), ("shadow", "Shadow")):
        if name not in requested:
            continue
        path = output_path / f"{prefix}_{tile_key}.tif"
        if not path.is_file():
            raise SolverInputError(f"full-tile run did not produce {path}")
        dataset = gdal.Open(str(path))
        try:
            bands = dataset.RasterCount
            stack = np.empty((bands, cache.rows, cache.cols), dtype=np.float32)
            for band in range(bands):
                stack[band] = dataset.GetRasterBand(band + 1).ReadAsArray()
        finally:
            dataset = None
        if bands != cache.time_steps:
            raise SolverInputError(
                f"{path} has {bands} bands, expected {cache.time_steps}"
            )
        results[name] = stack
    return results


def _forcing_with_overlay(
    cache: SiteCache,
    forcing: SiteForcing,
    *,
    site_dir: Path,
    overlay: ForcingOverlay | None,
) -> SiteForcing:
    """Reconcile a forcing object with an explicit ``overlay`` kwarg (u-c3).

    ``overlay=None`` (or an overlay the forcing already carries) keeps the
    forcing as-is: the worker/executor path materializes through
    :func:`load_site_forcing` exactly once and the full path must not
    re-derive it. A forcing that already carries a DIFFERENT overlay is an
    ambiguity, refused loudly. An overlay staged against a BASELINE forcing
    is materialized through :func:`load_site_forcing` so the masked
    re-check validates the resolve (resolve discipline is verified at that
    seam, never trusted) and the resolved table — not the baseline met path
    — is what the full run reads and stages.
    """
    if overlay is None or overlay == forcing.overlay:
        return forcing
    if forcing.overlay is not None:
        raise SolverInputError(
            "run_full_tile received a forcing already resolved under a "
            "scenario overlay AND a different overlay kwarg; pass the "
            "overlay to load_site_forcing once (the seam), not both"
        )
    return load_site_forcing(
        cache,
        site_dir=site_dir,
        selected_date_str=forcing.selected_date_str,
        overlay=overlay,
    )


def run_full_tile(
    cache: SiteCache,
    layer: TreeLayer,
    *,
    forcing: SiteForcing,
    site_dir: str | Path,
    scratch_dir: str | Path,
    requested_variables: Sequence[str] = ("utci", "tmrt", "shadow"),
    overlay: ForcingOverlay | None = None,
    landcover_overlay: LandCoverOverlay | None = None,
    model_parameters: Mapping[str, Any] | None = None,
    return_final_state: bool = False,
) -> dict[str, np.ndarray] | tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Run the standard full-domain SOLWEIG path on the composed scene.

    See :func:`_run_full_tile_impl`; this wrapper validates inputs.
    ``overlay`` (u-c3, default ``None`` = the legacy byte-identical path)
    is the scenario forcing overlay the full run must honor: either the
    forcing already carries it (``load_site_forcing(overlay=...)`` — the
    sanctioned materialization seam) or it is resolved here against the
    baseline forcing through the same loader. Either way ``compute_utci``
    reads the RESOLVED table and the scratch site stages it — an overlay is
    never silently dropped to the baseline met file.
    ``landcover_overlay`` (u-c4, default ``None`` = the legacy verbatim
    Landcover copy) is the scenario paint overlay: the full run materializes
    it through :func:`resolve_landcover_overlay` and stages the RESOLVED
    class grid in the scratch site, never the baseline raster.
    ``model_parameters`` (u-c7, default ``None`` = the legacy byte-identical
    ``compute_utci`` path) is the folded kernel-argument mapping from
    committed model parameter edits. A non-empty mapping reroutes the run
    through :func:`_run_full_tile_with_model_parameters` — a faithful
    replication of the ``compute_utci`` core over the SAME staged scratch
    site with the overrides handed to ``run_utci_window`` (the orchestrator
    itself has no parameter seam). Every name AND value in the mapping is
    domain-fenced here (the adapter's public
    :func:`~solweig_gpu.incremental.adapters.model_parameters.\
check_parameter_value`, re-raised as :class:`SolverInputError`) — the
    physics seam re-fences names only, so an out-of-domain scalar is
    refused at this door, never forwarded (u-c7-review MEDIUM).
    ``return_final_state`` (G2.1, default ``False`` = the exact previous
    return) additionally returns the full-tile cross-timestep thermal state
    at the end of the series — the capture side of the warm-start seam
    (routes the run through the documented ``compute_utci`` replication;
    see ``_run_full_tile_impl``).
    """
    if not Path(site_dir).is_dir():
        raise SolverInputError(f"site directory not found: {site_dir}")
    if model_parameters is not None and not isinstance(
        model_parameters, Mapping
    ):
        raise SolverInputError(
            "model_parameters must be a name -> value mapping, got "
            f"{type(model_parameters).__name__}"
        )
    if model_parameters:
        # Value-domain fence on the params branch (u-c7-review MEDIUM): the
        # mapping-type check above is not enough — {albedo_b: True},
        # {albedo_b: 50.0}, and {transVeg: inf} are all scalar Mappings that
        # ``run_utci_window`` would accept (it re-fences NAMES only), so the
        # adapter's own documented domain check runs here through its public
        # wrapper, before the scratch site is staged or any state advances.
        # An empty mapping skips this loop and stays on the legacy path.
        for name, value in sorted(model_parameters.items()):
            try:
                check_parameter_value(name, value)
            except EditStateError as error:
                raise SolverInputError(str(error)) from error
    return _run_full_tile_impl(
        cache,
        layer,
        forcing=_forcing_with_overlay(
            cache,
            forcing,
            site_dir=Path(site_dir).resolve(),
            overlay=overlay,
        ),
        site_dir=Path(site_dir).resolve(),
        scratch_dir=Path(scratch_dir),
        requested_variables=requested_variables,
        landcover_overlay=landcover_overlay,
        model_parameters=model_parameters,
        return_final_state=return_final_state,
    )


# ---------------------------------------------------------------------------
# T02 opt-in adapter seam: the solweig_core ABI boundary around the SAME
# numerical path (DESIGN.ko.md 5.2/5.3, TASKS T02). This is a SEAM, not new
# math: inputs become ABI views at the boundary, the ORIGINAL solve_window
# runs untouched, outputs become views back. Default behaviour of every
# existing caller is unchanged — the seam is reached only through
# ``solve_with_core_adapter`` or ``ExactWorker(core_adapter=True)``.
# ---------------------------------------------------------------------------

_CORE_DOMAIN_ID = "site:{tile}:{rows}x{cols}"


def _core_input_views(
    cache: SiteCache,
    canopy_np: np.ndarray,
    domain_id: str,
    dtype_plan: Mapping[str, str],
) -> dict[str, Any]:
    """Wrap the solve's input planes as ABI views at the boundary (zero-copy).

    Cache planes are borrowed read-only and FROZEN: the solve must not be
    able to write through the boundary, and ``assert_not_mutated`` re-checks
    every frozen view after the solve returns. The canopy is deliberately
    NOT frozen — ``_compose_scene_from_canopy`` clamps negative canopy
    through the shared ``torch.from_numpy`` buffer IN PLACE (LiDAR nodata),
    a documented-allowed mutation of a caller-owned plane.

    Every plane must be the FULL logical domain grid: a crop handed in as a
    full-grid plane is refused (a shape/origin lie would shift every global
    index downstream).
    """
    from solweig_core.abi import ArrayView
    from solweig_core.status import AbiValidationError

    rows, cols = int(cache.rows), int(cache.cols)
    planes: dict[str, np.ndarray] = {
        "building_dsm": np.asarray(cache.building_dsm),
        "dem": np.asarray(cache.dem),
        "walls": np.asarray(cache.walls),
        "wall_aspect": np.asarray(cache.wall_aspect),
    }
    if "landcover" in cache:
        planes["landcover"] = np.asarray(cache.landcover)
    views: dict[str, Any] = {}
    for name, array in planes.items():
        expected = np.dtype(dtype_plan[name])
        if array.dtype != expected:
            raise AbiValidationError(
                _core_refusal(
                    f"plane {name!r} dtype {array.dtype.str} disagrees with "
                    f"the dtype plan ({expected.str})"
                )
            )
        if tuple(array.shape) != (rows, cols):
            raise AbiValidationError(
                _core_refusal(
                    f"plane {name!r} shape {tuple(array.shape)} is not the "
                    f"full logical domain ({rows}, {cols}); a crop handed as "
                    "the full grid would shift every global index"
                )
            )
        views[name] = ArrayView.from_numpy(
            array,
            global_origin=(0, 0),  # full-grid borrows: origin is the grid corner
            logical_domain_id=domain_id,
            freeze=True,
        )
    canopy = np.asarray(canopy_np)
    expected_canopy = np.dtype(dtype_plan["veg_canopy"])
    if canopy.dtype != expected_canopy or tuple(canopy.shape) != (rows, cols):
        raise AbiValidationError(
            _core_refusal(
                f"veg_canopy shape/dtype ({tuple(canopy.shape)}, "
                f"{canopy.dtype.str}) is not the full logical domain plan "
                f"(({rows}, {cols}), {expected_canopy.str})"
            )
        )
    views["veg_canopy"] = ArrayView.from_numpy(
        canopy,
        global_origin=(0, 0),
        logical_domain_id=domain_id,
    )
    return views


def _core_refusal(detail: str):
    from solweig_core.status import RefusalReason, TypedRefusal

    return TypedRefusal(RefusalReason.ABI_INVALID, detail)


def solve_with_core_adapter(
    cache: SiteCache,
    layer: TreeLayer,
    *,
    read_window: RasterWindow,
    write_window: RasterWindow,
    forcing: SiteForcing,
    requested_variables: Sequence[str] = ("utci", "tmrt", "shadow"),
    veg_changed: bool | None = None,
    landcover_overlay: LandCoverOverlay | None = None,
    model_parameters: Mapping[str, Any] | None = None,
    time_start: int = 0,
    time_stop: int | None = None,
    stage_timings: MutableMapping[str, float] | None = None,
    veg_state: Any = None,
    device: str = "cpu",
    profile_id: str = "canonical_cpu_v1",
    output_views: MutableMapping[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """Opt-in solve through the solweig_core ABI/dispatch boundary.

    This is the T02 SEAM, not new physics: the request/windows/time are
    validated as a :class:`solweig_core.request.SolveRequestView` (logical
    domain, physical windows, GLOBAL time indices, dtype plan, profile),
    the device is resolved EXPLICITLY (``'cuda'`` is a typed BLOCKED
    refusal on this host — never a silent CPU-as-success substitute), the
    input planes cross the boundary as frozen read-only ABI views, the
    ORIGINAL :func:`solve_window` runs untouched, and the outputs cross
    back as frozen read-only views whose ``global_origin`` carries the
    write-window origin and the GLOBAL first timestep (the T01 harness
    metadata contract).

    Exactly ONE boundary conversion happens on the way in (the scene
    compose from the validated canopy view) and ZERO on the way out —
    there is no per-step or per-window tensor round-trip anywhere in this
    function (asserted by tests/ultrafast/test_adapter_roundtrip.py, both
    statically and by runtime conversion counters).

    ``output_views`` (optional) receives the output :class:`ArrayView`\\ s
    for metadata inspection; the returned mapping borrows the same arrays
    (no copy).
    """
    from solweig_core import dispatch as core_dispatch
    from solweig_core.abi import ArrayView
    from solweig_core.request import (
        DEFAULT_DTYPE_PLAN,
        PhysicalWindow,
        SolveRequestView,
        TimeCoverage,
    )

    domain_id = _CORE_DOMAIN_ID.format(
        tile=cache.tile_key, rows=cache.rows, cols=cache.cols
    )
    request = SolveRequestView(
        logical_domain_id=domain_id,
        rows=cache.rows,
        cols=cache.cols,
        origin_x_m=cache.manifest.origin_x_m,
        origin_y_m=cache.manifest.origin_y_m,
        pixel_size_m=cache.pixel_size_m,
        read_window=PhysicalWindow(
            read_window.row_start,
            read_window.row_stop,
            read_window.col_start,
            read_window.col_stop,
        ),
        write_window=PhysicalWindow(
            write_window.row_start,
            write_window.row_stop,
            write_window.col_start,
            write_window.col_stop,
        ),
        time=TimeCoverage(int(time_start), time_stop, cache.time_steps),
        requested_variables=tuple(requested_variables),
        dtype_plan=DEFAULT_DTYPE_PLAN,
        profile_id=profile_id,
        device=device,
    )
    request.validate()
    # Declarative dispatch: a CUDA request is BLOCKED here with a typed
    # refusal, before any buffer moves or solve work — never re-targeted.
    core_dispatch.resolve(request).raise_if_blocked()

    # Boundary IN — zero-copy borrows. The canopy is composed from the
    # LAYER (the same full-window rasterization compose_full_scene_tensors
    # performs), wrapped through the ABI, and composed into the scene: one
    # conversion, at the boundary, never inside a loop.
    canopy_np = layer.vegetation_rasters_window(
        RasterWindow(0, cache.rows, 0, cache.cols)
    )[0]
    input_views = _core_input_views(cache, canopy_np, domain_id, DEFAULT_DTYPE_PLAN)
    scene = _compose_scene_from_canopy(cache, input_views["veg_canopy"].to_numpy())

    outputs = solve_window(
        cache,
        layer,
        read_window=read_window,
        write_window=write_window,
        forcing=forcing,
        requested_variables=requested_variables,
        veg_changed=veg_changed,
        landcover_overlay=landcover_overlay,
        model_parameters=model_parameters,
        time_start=time_start,
        time_stop=time_stop,
        stage_timings=stage_timings,
        required_read_window=read_window,
        veg_state=veg_state,
    )

    # Boundary OUT — zero-copy wrap; global_origin carries the write-window
    # origin and the GLOBAL first timestep (T01 harness metadata contract).
    views: dict[str, Any] = {}
    for name, array in outputs.items():
        views[name] = ArrayView.from_numpy(
            array,
            global_origin=(
                int(time_start),
                write_window.row_start,
                write_window.col_start,
            ),
            logical_domain_id=domain_id,
            read_only=True,
            freeze=True,
        )
    if output_views is not None:
        output_views.update(views)
    # Freeze-on-lend witnesses: the read-only cache borrows must be bitwise
    # unchanged by the solve (the solver copies at every cache read; a
    # mutation here would mean someone wrote through the ABI boundary).
    for view in input_views.values():
        if view.frozen:
            view.assert_not_mutated()
    return {name: view.to_numpy() for name, view in views.items()}
