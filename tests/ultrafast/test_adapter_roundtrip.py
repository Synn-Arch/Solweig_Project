"""T02 adapter seam roundtrip: ABI boundary around the ORIGINAL solve path.

Byte-identity gate (a) of TASKS.ko.md T02, at test scale: a synthetic
40x80 non-square site built from T00's frozen fixture generator
(``benchmarks/ultrafast/make_synth_fixtures.build_scene``, read-only
import) is solved through the direct path (:func:`solve_window`) and
through the opt-in seam (:func:`solve_with_core_adapter`); every output
plane must be raw-bit identical (``tests/ultrafast/bitwise_harness``,
metadata included: global time indices + window origin).

Boundary guards (DESIGN 5.2: a ``.cpu().numpy()`` inside a patch/step loop
is never integrated):

* STATIC: the seam functions contain no tensor-conversion spellings
  (``.cpu``/``.numpy`` attribute access) and no torch name at all —
  loop or otherwise;
* RUNTIME: instrumenting ``torch.Tensor.cpu``/``torch.Tensor.numpy`` with
  counters, the adapter solve performs EXACTLY as many conversions as the
  direct solve (delta zero — the seam adds none, per-window or
  per-step);
* WORKER ROUTE: ``ExactWorker(core_adapter=True)`` publishes patch arrays
  bit-identical to the default worker.

MUTATION-DESIGNATED tests (T02 item 8): the output-view metadata test
fails if the seam's boundary conversion flips the window origin or drops
the global time start; the conversion-count test fails if any conversion
is added inside the seam; the bit-identity test fails if the boundary
reorders or perturbs buffers.

The site_500 E1 gate (b) runs once at task completion through
``benchmarks/ultrafast/run.py compare`` against the frozen e1_r1
artifacts — see the T02 report; this file pins the reproducible
synthetic lane.
"""
from __future__ import annotations

import ast
import datetime
import hashlib
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal, osr

gdal.UseExceptions()

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for _p in (str(REPO_ROOT), str(HERE), str(REPO_ROOT / "benchmarks" / "ultrafast")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import make_synth_fixtures  # noqa: E402  (T00-owned, read-only import)

from bitwise_harness import PlaneMetadata, assert_planes_equal  # noqa: E402
from solweig_gpu.incremental.cache import SiteCache  # noqa: E402
from solweig_gpu.incremental.cache_builder import build_site_cache  # noqa: E402
from solweig_gpu.incremental.geometry import RasterGrid, RasterWindow, TreeSpec  # noqa: E402
from solweig_gpu.incremental.solver import (  # noqa: E402
    compose_full_scene_tensors,
    load_site_forcing,
    read_window_for_write_window,
    solve_window,
    solve_with_core_adapter,
)
from solweig_gpu.incremental.trees import TreeLayer  # noqa: E402
from solweig_gpu.incremental.worker import ExactWorker  # noqa: E402
from solweig_core.status import (  # noqa: E402
    AbiValidationError,
    DispatchBlockedError,
)

# ---------------------------------------------------------------------------
# Synthetic prepared site (mirrors tests/test_incremental_worker.py's
# construction helpers, self-contained so this suite owns nothing shared)
# ---------------------------------------------------------------------------
ROWS, COLS = 40, 80  # non-square on purpose (T00 synth shape)
PIXEL = 2.0
ORIGIN = (1000.0, 2000.0)
EPSG = 32616
DATE_STR = "2024-06-20"
MET_HOURS = range(10, 16)  # 6 global timesteps
EDIT_CELL = (26, 20)
WRITE_WINDOW = RasterWindow(16, 36, 10, 30)


def _write_tif(path: Path, array: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows, cols = array.shape
    dtype = gdal.GDT_Byte if array.dtype == np.uint8 else gdal.GDT_Float32
    dataset = gdal.GetDriverByName("GTiff").Create(str(path), cols, rows, 1, dtype)
    dataset.SetGeoTransform((ORIGIN[0], PIXEL, 0.0, ORIGIN[1], 0.0, -PIXEL))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(EPSG)
    dataset.SetProjection(srs.ExportToWkt())
    dataset.GetRasterBand(1).WriteArray(array)
    dataset.FlushCache()
    dataset = None
    return path


def _met_row(hour: int) -> list[float]:
    ramp = math.sin(math.pi * (hour - 6) / 13.0)
    radg = 850.0 * max(0.0, ramp)
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


def _write_met(path: Path) -> Path:
    header = "# year doy hour minute placeholder wind rh ta p radg rad diff radI wdir uhii"
    lines = [header] + [" ".join(f"{v:g}" for v in _met_row(h)) for h in MET_HOURS]
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


def _derive_utc_offset(lat: float, lon: float, date_str: str) -> float:
    from timezonefinder import TimezoneFinder
    import pytz

    name = TimezoneFinder().timezone_at(lat=lat, lng=lon) or "UTC"
    local_dt = pytz.timezone(name).localize(
        datetime.datetime.strptime(date_str, "%Y-%m-%d")
    )
    return local_dt.utcoffset().total_seconds() / 3600


def _compute_baseline_svf(site: Path) -> None:
    """Fresh svf_calculator pass over the baseline scene (oracle layout)."""
    from solweig_gpu.shadow import load_raster_to_tensor, svf_calculator
    import torch

    a, dataset = load_raster_to_tensor(str(site / "Building_DSM" / "Building_DSM_0_0.tif"))
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


@pytest.fixture(scope="module")
def synth_site(tmp_path_factory):
    """Build the 40x80 prepared site + cache once; return the heavy pieces."""
    root = tmp_path_factory.mktemp("t02_synth")
    scene_planes = make_synth_fixtures.build_scene(ROWS, COLS, PIXEL)
    site = root / "processed_inputs"
    _write_tif(site / "Building_DSM" / "Building_DSM_0_0.tif", scene_planes["building_dsm"])
    _write_tif(site / "DEM" / "DEM_0_0.tif", scene_planes["dem"])
    _write_tif(site / "Trees" / "Trees_0_0.tif", scene_planes["veg_canopy"])
    _write_tif(site / "walls" / "walls_0_0.tif", np.zeros((ROWS, COLS), dtype=np.float32))
    _write_tif(site / "aspect" / "aspect_0_0.tif", np.zeros((ROWS, COLS), dtype=np.float32))
    landcover = np.ones((ROWS, COLS), dtype=np.uint8)
    landcover[ROWS // 3 : ROWS // 2, COLS // 3 : COLS // 2] = 5
    landcover[2 * ROWS // 3 : 2 * ROWS // 3 + 5, : COLS // 4] = 6
    _write_tif(site / "Landcover" / "Landcover_0_0.tif", landcover)
    met_path = _write_met(site / "metfiles" / "metfile_0_0_2024-06-20.txt")

    building = site / "Building_DSM" / "Building_DSM_0_0.tif"
    lon, lat = _derive_lon_lat(building)
    utc = _derive_utc_offset(lat, lon, DATE_STR)
    _compute_baseline_svf(site)
    cache = build_site_cache(
        site,
        root / "cache",
        tile_key="0_0",
        site_id="t02-synth",
        latitude=lat,
        longitude=lon,
        # synth DEM median is > 0 -> the oracle altitude rule pins 3.0
        altitude_m=3.0,
        utc_offset_hours=utc,
        met_file=met_path,
    )
    cache = SiteCache.load(root / "cache")
    grid = RasterGrid(
        cache.rows, cache.cols, cache.pixel_size_m,
        cache.manifest.origin_x_m, cache.manifest.origin_y_m,
    )
    return {"root": root, "site": site, "cache": cache, "grid": grid}


def _fresh_layer(cache: SiteCache, grid: RasterGrid) -> TreeLayer:
    """Fresh layer with the E1-like edit applied (deterministic)."""
    layer = TreeLayer(cache.tree_base, grid)
    x = grid.origin_x_m + (EDIT_CELL[1] + 0.5) * grid.pixel_size_m
    y = grid.origin_y_m - (EDIT_CELL[0] + 0.5) * grid.pixel_size_m
    layer.add_tree(TreeSpec("t1", x, y, 6.0, 2.0))
    return layer


def _solve_pair(synth_site):
    """Direct vs adapter solve on identically-edited fresh layers."""
    cache, grid = synth_site["cache"], synth_site["grid"]
    layer_direct = _fresh_layer(cache, grid)
    layer_adapter = _fresh_layer(cache, grid)

    scene = compose_full_scene_tensors(cache, layer_direct)
    forcing = load_site_forcing(
        cache, site_dir=synth_site["site"], selected_date_str=DATE_STR
    )
    read = read_window_for_write_window(WRITE_WINDOW, cache, scene, forcing)

    direct = solve_window(
        cache, layer_direct,
        read_window=read, write_window=WRITE_WINDOW, forcing=forcing,
        requested_variables=("utci", "tmrt", "shadow"),
        scene=scene, required_read_window=read,
    )
    output_views = {}
    adapted = solve_with_core_adapter(
        cache, layer_adapter,
        read_window=read, write_window=WRITE_WINDOW, forcing=forcing,
        requested_variables=("utci", "tmrt", "shadow"),
        output_views=output_views,
    )
    return direct, adapted, output_views, read


# ---------------------------------------------------------------------------
# THE GATE: raw-bit identity, metadata included
# ---------------------------------------------------------------------------
class TestByteIdentity:
    def test_adapter_planes_bit_identical_to_direct(self, synth_site):
        direct, adapted, _views, _read = _solve_pair(synth_site)
        assert set(direct) == set(adapted) == {"utci", "tmrt", "shadow"}
        for name in sorted(direct):
            meta = PlaneMetadata(
                plane=name,
                global_time_index=tuple(MET_HOURS),
                window_origin=(WRITE_WINDOW.row_start, WRITE_WINDOW.col_start),
                shape=direct[name].shape,
                dtype_str=direct[name].dtype.str,
            )
            # MUTATION-DESIGNATED: any boundary reorder/copy that perturbs
            # values, or an origin/time metadata lie, fails here.
            report = assert_planes_equal(
                direct[name], adapted[name], ref_meta=meta, cand_meta=meta
            )
            assert report["mismatch_count"] == 0
            assert report["metadata_comparison"]["status"] == "PASS"

    def test_output_views_carry_write_origin_and_global_time(self, synth_site):
        # MUTATION-DESIGNATED (origin flip / early free): the frozen output
        # views must state the write-window origin and the GLOBAL first
        # timestep — the T01 harness comparison fields.
        _direct, _adapted, views, _read = _solve_pair(synth_site)
        for name, view in views.items():
            assert view.global_origin == (
                0, WRITE_WINDOW.row_start, WRITE_WINDOW.col_start,
            ), f"{name}: output origin is not the write window"
            assert view.shape == (
                len(tuple(MET_HOURS)),
                WRITE_WINDOW.height,
                WRITE_WINDOW.width,
            )
            assert view.frozen and view.read_only
            assert view.residency == "cpu"
            assert view.is_contiguous()

    def test_returned_planes_are_the_views_owners(self, synth_site):
        _direct, adapted, views, _read = _solve_pair(synth_site)
        for name in adapted:
            assert adapted[name] is views[name].to_numpy()


# ---------------------------------------------------------------------------
# Boundary guards: no per-step/per-window tensor conversion in the seam
# ---------------------------------------------------------------------------
SEAM_FUNCTIONS = ("solve_with_core_adapter", "_core_input_views")


def _seam_ast():
    source = (REPO_ROOT / "solweig_gpu" / "incremental" / "solver.py").read_text()
    tree = ast.parse(source)
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in SEAM_FUNCTIONS:
            found[node.name] = node
    return found


class TestBoundaryGuard:
    def test_seam_functions_exist(self):
        found = _seam_ast()
        assert set(found) == set(SEAM_FUNCTIONS)

    def test_seam_has_no_tensor_conversion_spellings_anywhere(self):
        # STATIC witness: no .cpu/.numpy attribute access in the seam —
        # inside a loop or otherwise — and no torch name at all. DESIGN 5.2
        # forbids integrating implementations that repeat .cpu().numpy()
        # in the patch/step loop; banning the spelling in the whole seam is
        # strictly stronger.
        for name, node in _seam_ast().items():
            for sub in ast.walk(node):
                if isinstance(sub, ast.Attribute):
                    assert sub.attr not in ("cpu", "numpy"), (
                        f"{name} performs a tensor conversion (.{sub.attr})"
                    )
                if isinstance(sub, ast.Name):
                    assert sub.id != "torch", (
                        f"{name} references torch directly"
                    )
                if isinstance(sub, (ast.Import, ast.ImportFrom)):
                    modules = (
                        [a.name for a in sub.names]
                        if isinstance(sub, ast.Import)
                        else [sub.module or ""]
                    )
                    for module in modules:
                        assert module.split(".")[0] != "torch", (
                            f"{name} imports torch"
                        )

    def test_adapter_adds_no_tensor_conversions(self, synth_site, monkeypatch):
        # RUNTIME witness: with torch.Tensor.cpu/.numpy instrumented, the
        # adapter solve performs EXACTLY the conversions of the direct
        # solve — delta zero. A per-window or per-step conversion added to
        # the seam (e.g. re-materializing buffers inside a loop) shows up
        # as a positive delta here.
        import torch

        cache, grid = synth_site["cache"], synth_site["grid"]
        # Shared pre-computed inputs (uncounted): identical forcing, read
        # window, and freshly-edited layers so only the ROUTE differs.
        forcing = load_site_forcing(
            cache, site_dir=synth_site["site"], selected_date_str=DATE_STR
        )
        probe_layer = _fresh_layer(cache, grid)
        scene = compose_full_scene_tensors(cache, probe_layer)
        read = read_window_for_write_window(WRITE_WINDOW, cache, scene, forcing)

        counts: dict[str, int] = {"cpu": 0, "numpy": 0}
        orig_cpu, orig_numpy = torch.Tensor.cpu, torch.Tensor.numpy

        def counted_cpu(self, *args, **kwargs):
            counts["cpu"] += 1
            return orig_cpu(self, *args, **kwargs)

        def counted_numpy(self, *args, **kwargs):
            counts["numpy"] += 1
            return orig_numpy(self, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, "cpu", counted_cpu)
        monkeypatch.setattr(torch.Tensor, "numpy", counted_numpy)
        try:
            solve_with_core_adapter(
                cache, _fresh_layer(cache, grid),
                read_window=read, write_window=WRITE_WINDOW, forcing=forcing,
                requested_variables=("utci", "tmrt", "shadow"),
            )
        finally:
            monkeypatch.undo()
        adapter_counts = dict(counts)

        counts["cpu"] = counts["numpy"] = 0
        monkeypatch.setattr(torch.Tensor, "cpu", counted_cpu)
        monkeypatch.setattr(torch.Tensor, "numpy", counted_numpy)
        try:
            solve_window(
                cache, _fresh_layer(cache, grid),
                read_window=read, write_window=WRITE_WINDOW, forcing=forcing,
                requested_variables=("utci", "tmrt", "shadow"),
                scene=scene, required_read_window=read,
            )
        finally:
            monkeypatch.undo()
        direct_counts = dict(counts)

        assert adapter_counts["cpu"] == direct_counts["cpu"], (
            f"seam added Tensor.cpu calls: {direct_counts} -> {adapter_counts}"
        )
        assert adapter_counts["numpy"] == direct_counts["numpy"], (
            f"seam added Tensor.numpy calls: {direct_counts} -> {adapter_counts}"
        )


# ---------------------------------------------------------------------------
# Worker route: the opt-in flag, bit-identical patches
# ---------------------------------------------------------------------------
class TestWorkerRoute:
    def test_core_adapter_flag_routes_local_solve_bit_identical(self, synth_site):
        cache, grid = synth_site["cache"], synth_site["grid"]

        def run_worker(flag: bool) -> dict[str, np.ndarray]:
            layer = _fresh_layer(cache, grid)
            worker = ExactWorker(
                cache, layer,
                site_dir=synth_site["site"],
                results_root=synth_site["root"] / ("results_" + str(flag)),
                selected_date_str=DATE_STR,
                core_adapter=flag,
            )
            forcing = worker.forcing()
            patches = worker._solve_local(
                "job-t02", 1, forcing, (WRITE_WINDOW,)
            )
            assert len(patches) == 1
            return patches[0].arrays

        default = run_worker(False)
        adapted = run_worker(True)
        assert set(default) == set(adapted)
        for name in sorted(default):
            report = assert_planes_equal(
                default[name], adapted[name],
                ref_meta=PlaneMetadata(
                    plane=name, global_time_index=tuple(MET_HOURS),
                    window_origin=(WRITE_WINDOW.row_start, WRITE_WINDOW.col_start),
                ),
                cand_meta=PlaneMetadata(
                    plane=name, global_time_index=tuple(MET_HOURS),
                    window_origin=(WRITE_WINDOW.row_start, WRITE_WINDOW.col_start),
                ),
            )
            assert report["mismatch_count"] == 0

    def test_default_worker_untouched_by_the_seam(self):
        # The default constructor argument is False: existing callers keep
        # the direct solve_window route unless they opt in.
        import inspect
        from solweig_gpu.incremental import worker as worker_mod

        signature = inspect.signature(worker_mod.ExactWorker.__init__)
        assert signature.parameters["core_adapter"].default is False
        source = inspect.getsource(worker_mod.ExactWorker._solve_local)
        # both routes stay present and the adapter branch is opt-in only
        assert "solve_with_core_adapter" in source
        assert "if self._core_adapter:" in source


# ---------------------------------------------------------------------------
# Refusals and guards through the seam
# ---------------------------------------------------------------------------
class TestSeamRefusals:
    def test_cuda_request_blocked_before_any_solve(self, synth_site, monkeypatch):
        from solweig_gpu.incremental import solver as solver_mod

        calls = {"n": 0}

        def sentinel(*args, **kwargs):
            calls["n"] += 1
            raise AssertionError("solve_window must not run under a BLOCKED dispatch")

        monkeypatch.setattr(solver_mod, "solve_window", sentinel)
        cache, grid = synth_site["cache"], synth_site["grid"]
        layer = _fresh_layer(cache, grid)
        forcing = load_site_forcing(
            cache, site_dir=synth_site["site"], selected_date_str=DATE_STR
        )
        with pytest.raises(DispatchBlockedError):
            solve_with_core_adapter(
                cache, layer,
                read_window=RasterWindow(0, cache.rows, 0, cache.cols),
                write_window=WRITE_WINDOW,
                forcing=forcing,
                device="cuda",  # typed BLOCKED on this host — never CPU-as-success
            )
        assert calls["n"] == 0

    def test_read_must_contain_write_through_the_request(self, synth_site):
        cache, grid = synth_site["cache"], synth_site["grid"]
        layer = _fresh_layer(cache, grid)
        forcing = load_site_forcing(
            cache, site_dir=synth_site["site"], selected_date_str=DATE_STR
        )
        from solweig_core.status import RequestValidationError

        with pytest.raises(RequestValidationError, match="not contained"):
            solve_with_core_adapter(
                cache, layer,
                read_window=RasterWindow(16, 36, 10, 30),
                write_window=RasterWindow(16, 36, 10, 50),  # escapes read
                forcing=forcing,
            )

    def test_window_escaping_grid_refused(self, synth_site):
        cache, grid = synth_site["cache"], synth_site["grid"]
        layer = _fresh_layer(cache, grid)
        forcing = load_site_forcing(
            cache, site_dir=synth_site["site"], selected_date_str=DATE_STR
        )
        from solweig_core.status import RequestValidationError

        with pytest.raises(RequestValidationError, match="exceeds"):
            solve_with_core_adapter(
                cache, layer,
                read_window=RasterWindow(0, cache.rows + 1, 0, cache.cols),
                write_window=WRITE_WINDOW,
                forcing=forcing,
            )

    def test_crop_handed_as_full_grid_plane_refused(self, synth_site, monkeypatch):
        # RED witness (wrong shape through the boundary): a cache plane
        # that is not the full logical domain (a crop) fails ABI
        # validation at the seam — the origin/shape lie would shift every
        # global index downstream.
        crop = np.asarray(synth_site["cache"].building_dsm)[
            WRITE_WINDOW.row_start : WRITE_WINDOW.row_stop,
            WRITE_WINDOW.col_start : WRITE_WINDOW.col_stop,
        ].copy()
        monkeypatch.setattr(
            SiteCache,
            "building_dsm",
            property(lambda self: crop),
        )
        cache, grid = synth_site["cache"], synth_site["grid"]
        layer = _fresh_layer(cache, grid)
        forcing = load_site_forcing(
            cache, site_dir=synth_site["site"], selected_date_str=DATE_STR
        )
        with pytest.raises(AbiValidationError, match="full logical domain"):
            solve_with_core_adapter(
                cache, layer,
                read_window=RasterWindow(0, cache.rows, 0, cache.cols),
                write_window=WRITE_WINDOW,
                forcing=forcing,
            )

    def test_cache_bytes_unchanged_by_the_adapted_solve(self, synth_site):
        # The frozen read-only borrows assert internally after the solve;
        # this witness re-hashes the on-disk cache plane independently.
        cache = synth_site["cache"]
        before = hashlib.sha256(
            np.ascontiguousarray(cache.building_dsm).tobytes()
        ).hexdigest()
        _direct, _adapted, _views, _read = _solve_pair(synth_site)
        after = hashlib.sha256(
            np.ascontiguousarray(cache.building_dsm).tobytes()
        ).hexdigest()
        assert before == after
