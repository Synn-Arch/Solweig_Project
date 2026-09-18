# SPDX-License-Identifier: GPL-3.0-only
"""Full-preprocessing regeneration for committed building-massing batches.

This module is U-C packet 5's orchestration of the chain the building
adapter documents as its executor friction (``solweig_gpu/incremental/
adapters/building.py``, "Executor seam and U-C integration friction"):
building edits are **FULL-only** — every downstream product is
preprocessing-derived — so a committed batch must regenerate, in order:

1. **Stage** a scenario site (copy of the baseline prepared site; the
   baseline is never written through — the module's hard invariant).
2. **Rasterize** the committed :class:`MassingEdit` records into the
   staged ``Building_DSM`` (``BuildingLayer``, absolute elevation =
   per-cell DEM ground + ``height_m``). When a vegetation scenario layer
   is supplied its composed canopy is written into the staged ``Trees``
   so mixed batches regenerate coherently.
3. **Walls/aspect pass**: ``run_walls_aspect(site)``
   (:func:`solweig_gpu.solweig_gpu.run_walls_aspect` ->
   ``walls_aspect.run_parallel_processing``) rewrites ``walls/`` and
   ``aspect/`` from the scenario Building_DSM.
4. **SVF recompute**: ``calculate_svf(site, overwrite=True)``
   (:func:`solweig_gpu.solweig_gpu.calculate_svf` -> ``svf_calculator``,
   ``solweig_gpu/shadow.py``) rewrites ``SVF/SkyViewFactor_{tile}.tif``,
   ``SVF/svfs_{tile}.zip`` and ``SVF/shadowmats_{tile}.npz`` for the
   scenario scene.
5. **Site-cache rebuild**: :func:`solweig_gpu.incremental.cache_builder.
   build_site_cache` over the scenario site, hashing the regenerated
   walls/aspect/SVF into a scenario cache (solar geometry carried over
   from the baseline manifest so the solver's location/timezone
   cross-checks hold).
6. **Full tile solve**: :func:`solweig_gpu.incremental.solver.run_full_tile`
   on the scenario cache + site (it stages its own scratch and recomputes
   SVF cold — the standard oracle path; the regenerated SVF artifacts are
   for the *cache*, later local jobs, and audit, not a solver shortcut).

Failure atomicity
-----------------

Every stage writes only under ``scenario_root`` (wiped at the start of a
batch, exactly how ``run_full_tile`` wipes its scratch). The baseline site
and baseline cache are opened read-only; a failing stage therefore leaves
them untouched by construction, and the partial scenario tree remains
under ``scenario_root`` as forensic evidence (the stage list in the raised
:exc:`RegenerationError` names how far the chain got). A rerun wipes and
rebuilds.

Executor seam (u-c1)
--------------------

:func:`regenerate_building_batch` is the function the executor calls:

.. code-block:: python

    edits = massing_edits_from_deltas(committed_building_deltas)
    result = regenerate_building_batch(
        edits=edits,
        baseline_site_dir=site_dir,
        baseline_cache=cache,
        scenario_root=scenario_root,
    )  # -> BuildingRegenerationResult(outputs, cache, records, ...)

The individual stages stay public and independently callable for testing
and for finer-grained executor policies.
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .adapters.building import MassingEdit
from .adapters.landcover import LandCoverOverlay
from .adapters.met_time import ForcingOverlay
from .buildings import BuildingLayer, BuildingRasterResult
from .cache import SiteCache
from .trees import TreeLayer

__all__ = [
    "STAGE_SITE",
    "STAGE_BUILDING_DSM",
    "STAGE_TREES",
    "STAGE_WALLS_ASPECT",
    "STAGE_SVF",
    "STAGE_CACHE",
    "STAGE_FULL_TILE",
    "BuildingRegenerationResult",
    "RegenerationError",
    "rebuild_scenario_cache",
    "regenerate_building_batch",
    "regenerate_svf",
    "regenerate_walls_aspect",
    "run_scenario_full_tile",
    "stage_scenario_site",
    "write_scenario_building_dsm",
    "write_scenario_trees",
]

#: Stage names, in execution order (reported in results and errors).
STAGE_SITE = "stage_scenario_site"
STAGE_BUILDING_DSM = "write_scenario_building_dsm"
STAGE_TREES = "write_scenario_trees"
STAGE_WALLS_ASPECT = "regenerate_walls_aspect"
STAGE_SVF = "regenerate_svf"
STAGE_CACHE = "rebuild_scenario_cache"
STAGE_FULL_TILE = "run_scenario_full_tile"

_MET_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


class RegenerationError(RuntimeError):
    """A regeneration stage failed; the baseline is untouched (by design)."""


@dataclass(frozen=True)
class BuildingRegenerationResult:
    """Everything one committed building batch regenerated."""

    scenario_site_dir: Path
    cache_dir: Path
    cache: SiteCache
    raster: np.ndarray
    records: BuildingRasterResult
    outputs: dict[str, np.ndarray]
    stages: tuple[str, ...]

    @property
    def disclosures(self) -> tuple[str, ...]:
        """Flattened per-edit disclosures (walllimit, empty fills, edges)."""
        return self.records.disclosures


# ---------------------------------------------------------------------------
# Tile raster IO (GDAL, imported lazily like cache_builder's readers)
# ---------------------------------------------------------------------------


def _tile_path(site_dir: Path, kind: str, tile_key: str) -> Path:
    return Path(site_dir) / kind / f"{kind}_{tile_key}.tif"


def _read_tile(
    site_dir: Path, kind: str, tile_key: str
) -> tuple[np.ndarray, tuple[float, ...], str, float | None, Path]:
    """Read one staged tile raster: (array, geotransform, wkt, nodata, path)."""
    from osgeo import gdal  # lazy: warm workers never load GDAL for this

    path = _tile_path(site_dir, kind, tile_key)
    dataset = gdal.Open(str(path))
    if dataset is None:
        raise RegenerationError(f"could not open tile raster: {path}")
    try:
        band = dataset.GetRasterBand(1)
        array = band.ReadAsArray()
        if array is None:
            raise RegenerationError(f"could not read band 1 of {path}")
        nodata = band.GetNoDataValue()
        geotransform = tuple(float(v) for v in dataset.GetGeoTransform())
        wkt = dataset.GetProjection()
    finally:
        dataset = None
    return array, geotransform, wkt, nodata, path


def _write_tile(
    path: Path,
    array: np.ndarray,
    geotransform: tuple[float, ...],
    wkt: str,
    nodata: float | None,
) -> None:
    """Write one float32 tile raster atomically (temp file + replace)."""
    from osgeo import gdal

    data = np.asarray(array, dtype=np.float32)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".regenerating.tif")
    dataset = None
    try:
        dataset = gdal.GetDriverByName("GTiff").Create(
            str(tmp), data.shape[1], data.shape[0], 1, gdal.GDT_Float32
        )
        if dataset is None:
            raise RegenerationError(f"could not create raster {tmp}")
        dataset.SetGeoTransform(geotransform)
        dataset.SetProjection(wkt)
        band = dataset.GetRasterBand(1)
        if nodata is not None:
            band.SetNoDataValue(float(nodata))
        band.WriteArray(data)
        band.FlushCache()
        band = None
        dataset.FlushCache()
    finally:
        dataset = None
    os.replace(tmp, path)


def _grid_from_geotransform(
    geotransform: tuple[float, ...], rows: int, cols: int, label: str
):
    from .geometry import RasterGrid

    if (
        geotransform[1] <= 0
        or geotransform[5] >= 0
        or abs(abs(geotransform[5]) - geotransform[1]) > 1e-9
        or geotransform[2] != 0.0
        or geotransform[4] != 0.0
    ):
        raise RegenerationError(
            f"{label} geotransform {geotransform} must be north-up with "
            "square axis-aligned pixels"
        )
    return RasterGrid(
        rows=rows,
        cols=cols,
        pixel_size_m=geotransform[1],
        origin_x_m=geotransform[0],
        origin_y_m=geotransform[3],
    )


def _site_grid(site_dir: Path, tile_key: str):
    """The site's RasterGrid, from the Building_DSM tile header."""
    _array, geotransform, _wkt, _nodata, path = _read_tile(
        site_dir, "Building_DSM", tile_key
    )
    return _grid_from_geotransform(
        geotransform, _array.shape[0], _array.shape[1], f"{path.name}"
    )


def _require_same_grid(
    left: tuple[float, ...], right: tuple[float, ...], left_name: str, right_name: str
) -> None:
    if any(abs(a - b) > 1e-9 for a, b in zip(left, right)):
        raise RegenerationError(
            f"{left_name} geotransform {left} does not match {right_name} "
            f"{right}; a regeneration batch needs one shared grid"
        )


# ---------------------------------------------------------------------------
# Stage 1: scenario site staging (baseline stays read-only)
# ---------------------------------------------------------------------------


def stage_scenario_site(
    baseline_site_dir: str | Path,
    scenario_root: str | Path,
    *,
    tile_key: str,
) -> Path:
    """Copy the baseline prepared site into ``scenario_root/site``.

    Refuses (before touching anything) when the baseline is missing a
    ``Building_DSM``/``DEM``/``Trees`` tile or the ``metfiles`` directory,
    or when ``scenario_root`` overlaps the baseline site (the wipe at the
    start of staging must never be able to delete baseline inputs).
    """
    baseline = Path(baseline_site_dir).resolve()
    root = Path(scenario_root).resolve()
    if not baseline.is_dir():
        raise RegenerationError(f"baseline site directory not found: {baseline}")
    for kind in ("Building_DSM", "DEM", "Trees"):
        if not _tile_path(baseline, kind, tile_key).is_file():
            raise RegenerationError(
                f"baseline {kind}/{kind}_{tile_key}.tif missing in {baseline}"
            )
    if not (baseline / "metfiles").is_dir():
        raise RegenerationError(f"baseline metfiles directory missing: {baseline}")
    if root == baseline or baseline in root.parents or root in baseline.parents:
        raise RegenerationError(
            f"scenario_root {root} must not overlap the baseline site "
            f"{baseline}; staging wipes scenario_root on every run"
        )
    if root.exists():
        shutil.rmtree(root)
    site = root / "site"
    shutil.copytree(baseline, site)
    return site


# ---------------------------------------------------------------------------
# Stage 2: rasterize massing edits (and any vegetation scenario) in place
# ---------------------------------------------------------------------------


def write_scenario_building_dsm(
    scenario_site_dir: str | Path,
    edits: Sequence[MassingEdit],
    *,
    tile_key: str,
) -> BuildingRasterResult:
    """Apply committed massing edits to the STAGED Building_DSM tile.

    Reads the staged ``Building_DSM``/``DEM`` pair (grid cross-checked),
    rasterizes through :class:`BuildingLayer` (per-cell DEM grounding,
    ``fmax`` combination, before-footprint resets), and rewrites the tile
    in place atomically. The baseline site is never touched — callers pass
    the staged copy from :func:`stage_scenario_site`.
    """
    site = Path(scenario_site_dir)
    base, gt_base, wkt_base, nodata, building_path = _read_tile(
        site, "Building_DSM", tile_key
    )
    dem, gt_dem, _wkt_dem, _nd_dem, dem_path = _read_tile(site, "DEM", tile_key)
    if base.shape != dem.shape:
        raise RegenerationError(
            f"staged Building_DSM {building_path} shape {base.shape} does "
            f"not match DEM {dem_path} shape {dem.shape}"
        )
    _require_same_grid(gt_base, gt_dem, "Building_DSM", "DEM")
    grid = _grid_from_geotransform(
        gt_base, base.shape[0], base.shape[1], building_path.name
    )
    layer = BuildingLayer(base, dem, grid, scenario_id="scenario")
    layer.apply_edits(edits)
    result = layer.rasterize_with_records()
    _write_tile(building_path, result.raster, gt_base, wkt_base, nodata)
    return result


def write_scenario_trees(
    scenario_site_dir: str | Path,
    tree_layer: TreeLayer,
    *,
    tile_key: str,
) -> None:
    """Write a vegetation scenario's composed canopy into the staged Trees tile.

    The canopy array is exactly what ``run_full_tile`` recomposes from the
    same layer (``vegetation_rasters_window(full)[0]``), so the staged
    site, the regenerated SVF, and the solver's own scene composition stay
    one scene.
    """
    site = Path(scenario_site_dir)
    _existing, geotransform, wkt, nodata, trees_path = _read_tile(
        site, "Trees", tile_key
    )
    grid = _grid_from_geotransform(
        geotransform, _existing.shape[0], _existing.shape[1], trees_path.name
    )
    if (grid.rows, grid.cols) != (tree_layer.grid.rows, tree_layer.grid.cols):
        raise RegenerationError(
            f"tree layer grid {tree_layer.grid.rows}x{tree_layer.grid.cols} "
            f"does not match the site grid {grid.rows}x{grid.cols}"
        )
    canopy = tree_layer.vegetation_rasters_window(grid.full_window)[0]
    _write_tile(trees_path, canopy, geotransform, wkt, nodata)


# ---------------------------------------------------------------------------
# Stages 3-4: preprocessing passes over the scenario site
# ---------------------------------------------------------------------------


def regenerate_walls_aspect(site_dir: str | Path) -> None:
    """Re-run the wall pass (walls/ + aspect/) over the scenario site.

    Thin wrapper over :func:`solweig_gpu.solweig_gpu.run_walls_aspect`
    (-> ``walls_aspect.run_parallel_processing``); imported lazily so this
    module stays importable without GDAL/torch/scipy.
    """
    from ..solweig_gpu import run_walls_aspect

    run_walls_aspect(str(Path(site_dir).resolve()))


def regenerate_svf(site_dir: str | Path, *, patch_option: int = 2) -> None:
    """Recompute the standalone SVF artifacts (overwrite=True).

    Rewrites ``SVF/SkyViewFactor_{tile}.tif``, ``SVF/svfs_{tile}.zip`` and
    ``SVF/shadowmats_{tile}.npz`` for every tile of the scenario site via
    :func:`solweig_gpu.solweig_gpu.calculate_svf` (``svf_calculator``).
    ``patch_option`` defaults to 2 — the option ``compute_utci`` itself
    pins.
    """
    from ..solweig_gpu import calculate_svf

    calculate_svf(
        str(Path(site_dir).resolve()), patch_option=patch_option, overwrite=True
    )


# ---------------------------------------------------------------------------
# Stage 5: scenario site cache
# ---------------------------------------------------------------------------


def rebuild_scenario_cache(
    scenario_site_dir: str | Path,
    cache_dir: str | Path,
    baseline_cache: SiteCache | str | Path,
    *,
    met_file: str | Path | None = None,
) -> SiteCache:
    """Build the scenario site cache from the regenerated site.

    Solar geometry (latitude/longitude/UTC offset) is carried over from the
    baseline manifest so the solver's oracle cross-checks (site-raster
    centre, timezone) hold exactly as they did for the baseline cache; the
    altitude is re-derived with the oracle rule from the (unchanged) DEM.
    Walllimit note: blocks under 3 m contribute no walls/aspect entries —
    the cache faithfully hashes whatever the passes produced.
    """
    from .cache_builder import build_site_cache

    baseline = _load_baseline_cache(baseline_cache)
    solar = baseline.manifest.solar_geometry
    site_id = f"{baseline.manifest.site_id}-scenario"
    build_site_cache(
        scenario_site_dir,
        cache_dir,
        tile_key=baseline.tile_key,
        site_id=site_id,
        met_file=met_file,
        latitude=float(solar.latitude),
        longitude=float(solar.longitude),
        altitude_m=None,  # oracle rule over the unchanged scenario DEM
        utc_offset_hours=float(solar.utc_offset_hours),
        model_version=baseline.manifest.model_version,
    )
    return SiteCache.load(Path(cache_dir))


def _load_baseline_cache(
    baseline_cache: SiteCache | str | Path,
) -> SiteCache:
    if isinstance(baseline_cache, SiteCache):
        return baseline_cache
    return SiteCache.load(Path(baseline_cache))


# ---------------------------------------------------------------------------
# Stage 6: full-tile solve on the scenario site
# ---------------------------------------------------------------------------


def run_scenario_full_tile(
    cache: SiteCache,
    layer: TreeLayer,
    *,
    site_dir: str | Path,
    scratch_dir: str | Path,
    selected_date_str: str,
    requested_variables: Sequence[str] = ("utci", "tmrt", "shadow"),
    forcing_overlay: ForcingOverlay | None = None,
    landcover_overlay: LandCoverOverlay | None = None,
    model_parameters: Mapping[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """Run :func:`run_full_tile` over the scenario cache and site.

    ``run_full_tile`` stages its own scratch (fresh SVF — no stale cache
    reuse) and reads the scenario Building_DSM / walls / aspect from
    ``site_dir``; this wrapper only derives the float64 forcing the way
    the oracle does and forwards.

    ``forcing_overlay`` (u-c3 L3) carries the scenario's committed
    forcing edits into the materialized forcing. The default ``None``
    keeps the legacy behavior — the BASELINE table, no overlay — so
    callers that never pass one are byte-identical to before. Passing an
    overlay here is what keeps a mixed building+met batch's forcing
    visible on the regenerated full-tile path (an overlay silently
    dropped to baseline text would invalidate the recomputed products).

    ``landcover_overlay`` (u-c4, threaded by u-c6) carries the scenario's
    committed paint edits into the full run: ``run_full_tile`` resolves
    it against the scenario cache's landcover (a baseline copy, so the
    resolve matches the baseline resolution discipline) and stages the
    RESOLVED class grid. Default ``None`` keeps the legacy verbatim-copy
    path, byte-identical to before.

    ``model_parameters`` (u-c7) carries the scenario's committed
    model-parameter edits into the full run's windowed core: a non-empty
    mapping reroutes ``run_full_tile`` through its direct-path
    replication of the ``compute_utci`` core with the overrides handed to
    ``run_utci_window``. Default ``None`` keeps the legacy ``compute_utci``
    path, byte-identical to before.
    """
    from .solver import load_site_forcing, run_full_tile

    forcing = load_site_forcing(
        cache,
        site_dir=Path(site_dir).resolve(),
        selected_date_str=selected_date_str,
        overlay=forcing_overlay,
    )
    return run_full_tile(
        cache,
        layer,
        forcing=forcing,
        site_dir=site_dir,
        scratch_dir=scratch_dir,
        requested_variables=requested_variables,
        landcover_overlay=landcover_overlay,
        model_parameters=model_parameters,
    )


# ---------------------------------------------------------------------------
# The batch orchestrator (the u-c1 executor seam)
# ---------------------------------------------------------------------------


def _derive_selected_date_str(met_path: Path) -> str:
    match = _MET_DATE_RE.search(met_path.name)
    if match is None:
        raise RegenerationError(
            f"could not derive a YYYY-MM-DD simulation date from the met "
            f"file name {met_path.name}; pass selected_date_str explicitly"
        )
    return match.group(1)


def regenerate_building_batch(
    *,
    edits: Sequence[MassingEdit],
    baseline_site_dir: str | Path,
    baseline_cache: SiteCache | str | Path,
    scenario_root: str | Path,
    tree_layer: TreeLayer | None = None,
    selected_date_str: str | None = None,
    requested_variables: Sequence[str] = ("utci", "tmrt", "shadow"),
    patch_option: int = 2,
    met_file: str | Path | None = None,
    forcing_overlay: ForcingOverlay | None = None,
    landcover_overlay: LandCoverOverlay | None = None,
    model_parameters: Mapping[str, Any] | None = None,
) -> BuildingRegenerationResult:
    """Regenerate the full preprocessing chain for one building batch.

    Runs stages 1-6 (see the module docstring) under ``scenario_root``
    (``site/``, ``cache/``, ``solver_scratch/``), returning the scenario
    cache, the rasterization records, and the full-tile outputs. The
    baseline site and cache are read-only inputs: any stage failure raises
    :exc:`RegenerationError` naming the stage and leaves the baseline
    byte-identical (verified by tests) with the partial scenario tree kept
    under ``scenario_root`` for inspection.

    ``edits`` are the adapter's re-validated records
    (:func:`~solweig_gpu.incremental.adapters.building.massing_edits_from_deltas`
    output). ``tree_layer`` optionally carries the scenario's committed
    vegetation (mixed batches); the default ``None`` keeps the baseline
    vegetation raster as staged. ``forcing_overlay`` (u-c3 L3) optionally
    carries the scenario's committed forcing edits through to the
    full-tile stage's forcing materialization; the default ``None`` is
    the legacy baseline-forcing path, byte-identical to before.
    ``landcover_overlay`` (u-c4, threaded by u-c6) optionally carries the
    scenario's committed paint edits through to the full-tile stage's
    class-grid materialization; the default ``None`` is the legacy
    verbatim-copy path, byte-identical to before.
    ``model_parameters`` (u-c7) optionally carries the scenario's
    committed model-parameter edits through to the full-tile stage's
    windowed core; the default ``None`` is the legacy ``compute_utci``
    path, byte-identical to before.
    """
    edit_tuple = tuple(edits)
    if not edit_tuple:
        raise RegenerationError(
            "a regeneration batch needs at least one MassingEdit "
            "(an empty batch is a no-op the executor never plans)"
        )
    cache_in = _load_baseline_cache(baseline_cache)
    tile_key = cache_in.tile_key
    root = Path(scenario_root).resolve()
    stages: list[str] = []

    def _stage(name: str, run):
        try:
            outcome = run()
        except Exception as error:  # noqa: BLE001 - wrapped with context
            raise RegenerationError(
                f"regeneration stage {name!r} failed after "
                f"{tuple(stages)}: {error}"
            ) from error
        stages.append(name)
        return outcome

    site = _stage(
        STAGE_SITE,
        lambda: stage_scenario_site(
            baseline_site_dir, root, tile_key=tile_key
        ),
    )
    records = _stage(
        STAGE_BUILDING_DSM,
        lambda: write_scenario_building_dsm(site, edit_tuple, tile_key=tile_key),
    )
    grid = _site_grid(site, tile_key)
    layer = tree_layer
    if layer is None:
        layer = TreeLayer(
            np.array(cache_in.tree_base), grid, scenario_id="scenario"
        )
    elif (layer.grid.rows, layer.grid.cols) != (grid.rows, grid.cols):
        raise RegenerationError(
            f"tree layer grid {layer.grid.rows}x{layer.grid.cols} does not "
            f"match the site grid {grid.rows}x{grid.cols}"
        )
    if tree_layer is not None:
        _stage(
            STAGE_TREES,
            lambda: write_scenario_trees(site, layer, tile_key=tile_key),
        )
    _stage(STAGE_WALLS_ASPECT, lambda: regenerate_walls_aspect(site))
    _stage(STAGE_SVF, lambda: regenerate_svf(site, patch_option=patch_option))
    cache_dir = root / "cache"
    scenario_cache = _stage(
        STAGE_CACHE,
        lambda: rebuild_scenario_cache(
            site, cache_dir, cache_in, met_file=met_file
        ),
    )
    if selected_date_str is None:
        met_entry = cache_in.manifest.sources.get("met")
        if met_entry is None:
            raise RegenerationError(
                "baseline cache manifest carries no 'met' source entry; "
                "pass selected_date_str explicitly"
            )
        selected_date_str = _derive_selected_date_str(
            Path(baseline_site_dir).resolve() / met_entry.path
        )
    outputs = _stage(
        STAGE_FULL_TILE,
        lambda: run_scenario_full_tile(
            scenario_cache,
            layer,
            site_dir=site,
            scratch_dir=root / "solver_scratch",
            selected_date_str=selected_date_str,
            requested_variables=requested_variables,
            forcing_overlay=forcing_overlay,
            landcover_overlay=landcover_overlay,
            model_parameters=model_parameters,
        ),
    )
    return BuildingRegenerationResult(
        scenario_site_dir=site,
        cache_dir=cache_dir,
        cache=scenario_cache,
        raster=records.raster,
        records=records,
        outputs=dict(outputs),
        stages=tuple(stages),
    )
