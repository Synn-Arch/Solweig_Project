# SPDX-License-Identifier: GPL-3.0-only
"""W2 replay amplitude: the veg-SVF replay march must stop where the oracle stops.

MEDIUM-2 (r4a-review): ``_recompute_veg_svf_window`` computed its early-exit
march amplitude (``effective_march_amplitude``) over the READ-WINDOW CROP.
The bound is ``max(a, vegdsm, vegdsm2) - min(a)`` over the cropped arrays, so
a narrowed W2 read window (exact occluder mask) that excludes the site's
tallest surface shrinks the bound below a sky patch's FIRST-STEP march
distance ``dz_1 = ds * (tan(alt) / scale)`` while the full tile stays above
it. ``shadow()`` then executes a different STEP REGIME on the crop than the
oracle's full-tile march: the oracle runs step 2+ (accumulating ``vegsh``
into ``vbshvegsh`` after the step-1 ``zero_()``), the crop replay stops
after step 1 — and the step-1 raise survives un-buffered, so ``vbsh``
reaches 2.0 in the replay where the oracle holds 1.0. The fold consumes
VALUES, so the divergence reaches every ``svf*aveg`` scalar and
``svftotal``: a value divergence against the oracle twin, not a {0,1}-fence
issue.

Invariant established (solver module docstring carries it too): the
replay's per-patch march regime is a function of the FULL-TILE composed
scene only, never of the read window. The replay amplitude is the
scene-wide effective amplitude (the same expression r4a's corridor
re-march uses, ``veg_svf_state._scene_amplitude``).

R6 escalation (this file's banded-scene classes): a scene whose effective
amplitude leaves some patch one-step where the oracle's absolute stop is
multi-step (``dz_1`` in ``(A_eff, A_abs]``) no longer refuses — the banded
patches march at the ORACLE's absolute stop ``scene.amaxvalue`` instead
(the same tensor value ``svf_calculator`` passes), so their executed step
set is identical to the oracle's by construction and the serve is
bitwise-equal to the full-tile twin. NEVER to ``A_eff`` — the disproved
approximation. Non-banded patches keep the early-exit amplitude (skipped
steps provably inert). The same absolute-amplitude source closes the
latent time-loop note: a timestep whose SUN first-step distance lands in
the (windowed effective, absolute] band escalates the time-loop march to
``scene.amaxvalue`` as well. :class:`MarchRegimeError` stays typed for
structural march-regime refusals (worker routing contract pinned below).

The dz_1 arithmetic in :func:`_dz1_per_patch` mirrors
``veg_svf_state._patch_first_step_dz`` (r4a-owned) association-for-
association: ``dz = ds * index * (tan(alt) / scale)`` with the
``tan/scale`` quotient folded FIRST (review NEW-LOW) — the naive
``ds * tan(alt) / scale`` is 1 ulp off at non-power-of-two scales.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from solweig_gpu.incremental.geometry import RasterGrid, RasterWindow, TreeSpec
from solweig_gpu.incremental.solver import (
    SolverInputError,
    compose_full_scene_tensors,
    effective_march_amplitude,
    load_site_forcing,
    read_window_for_write_window,
    run_full_tile,
    solve_window,
    window_svf_bundle,
)
from solweig_gpu.incremental.trees import TreeLayer
from solweig_gpu.shadow import svf_calculator

from tests.test_incremental_worker import (  # noqa: F401  (fixtures)
    SVF_BUNDLE_INDEX,
    _build_cache,
    _compute_baseline_svf,
    _make_prepared_site,
    _write_met,
)

DATE_STR = "2024-06-20"
VARIABLES = ("utci", "tmrt", "shadow")

VEG_SCALARS = (
    "svfveg", "svfEveg", "svfSveg", "svfWveg", "svfNveg",
    "svfaveg", "svfEaveg", "svfSaveg", "svfWaveg", "svfNaveg",
    "svftotal",
)


def _dz1_per_patch(scale: float) -> np.ndarray:
    """First-step march distance per sky patch, association-matched.

    Mirrors ``veg_svf_state._patch_first_step_dz`` (r4a-owned; kept in sync
    by cross-reference): same branch/ds expressions as ``shadow()`` and the
    SAME dz association (``ds * 1.0 * (tan(alt)/scale)``). The zenith patch
    carries the -1 sentinel (it never marches).
    """
    from solweig_gpu.incremental.solver import _sky_patch_geometry

    patches, _rings = _sky_patch_geometry(2)
    degrees = torch.pi / 180.0
    pibyfour = torch.pi / 4.0
    out = np.zeros(len(patches), dtype=np.float64)
    for i, patch in enumerate(patches):
        altitude, azimuth = float(patch[0]), float(patch[1])
        if altitude >= 90.0:
            out[i] = -1.0
            continue
        az = azimuth
        if az == 0.0:
            az = 1e-12
        az = torch.tensor(az) * degrees
        alt = torch.tensor(altitude) * degrees
        sinazimuth = torch.sin(az)
        cosazimuth = torch.cos(az)
        if bool(pibyfour <= az < 3.0 * pibyfour) or bool(
            5.0 * pibyfour <= az < 7.0 * pibyfour
        ):
            ds = torch.abs(1.0 / sinazimuth)
        else:
            ds = torch.abs(1.0 / cosazimuth)
        tanaltitudebyscale = torch.tan(alt) / scale
        out[i] = float(ds * 1.0 * tanaltitudebyscale)
    return out


def _crop_amplitude(scene: object, window: RasterWindow) -> float:
    """The (defective) crop-relative amplitude exactly as the replay read it."""

    def _crop(tensor: torch.Tensor) -> torch.Tensor:
        return tensor[window.row_start : window.row_stop, window.col_start : window.col_stop]

    return float(
        effective_march_amplitude(
            _crop(scene.a), _crop(scene.vegdsm), _crop(scene.vegdsm2),
            scene_amaxvalue=scene.amaxvalue,
        )
    )


def _full_amplitude(scene: object) -> float:
    return float(
        effective_march_amplitude(
            scene.a, scene.vegdsm, scene.vegdsm2, scene_amaxvalue=scene.amaxvalue
        )
    )


def _window_amplitude(scene: object, window: RasterWindow) -> float:
    """Windowed effective amplitude exactly as solve_window derives it."""

    def _crop(tensor: torch.Tensor) -> torch.Tensor:
        return tensor[window.row_start : window.row_stop, window.col_start : window.col_stop]

    a = _crop(scene.a)
    temp1_clamped = torch.where(
        _crop(scene.canopy) < 0.0, torch.zeros(()), _crop(scene.canopy)
    )
    return float(
        effective_march_amplitude(
            a,
            temp1_clamped + a,  # vegdsm pre-quirk, mirroring solve_window
            temp1_clamped * 0.25 + a,
            scene_amaxvalue=scene.amaxvalue,
        )
    )


def _sun_dz1_per_timestep(forcing, scale: float) -> list[float | None]:
    """First NONZERO march distance of the time loop per timestep.

    Mirrors ``shadowingfunction_wallheight_23`` association-for-association:
    ``index`` starts at 0 (dz 0, the self-comparison), the first shifted
    step is index 1 with ``dz = (ds * 1) * (tan(alt) / scale)`` — the
    quotient folded FIRST, no azimuth-zero substitution (unlike
    ``shadow()``). ``None`` marks a timestep at/below the horizon (the
    march is skipped entirely there).
    """
    degrees = torch.pi / 180.0
    pibyfour = torch.pi / 4.0
    out: list[float | None] = []
    for alt_deg, az_deg in zip(forcing.altitude[0], forcing.azimuth[0]):
        alt0, az0 = float(alt_deg), float(az_deg)
        if alt0 <= 0.0:
            out.append(None)
            continue
        az = torch.tensor(az0) * degrees
        alt = torch.tensor(alt0) * degrees
        sinazimuth = torch.sin(az)
        cosazimuth = torch.cos(az)
        if bool(pibyfour <= az < 3.0 * pibyfour) or bool(
            5.0 * pibyfour <= az < 7.0 * pibyfour
        ):
            ds = torch.abs(1.0 / sinazimuth)
        else:
            ds = torch.abs(1.0 / cosazimuth)
        tanaltitudebyscale = torch.tan(alt) / scale
        out.append(float((ds * 1.0) * tanaltitudebyscale))
    return out


# ---------------------------------------------------------------------------
# Sites
# ---------------------------------------------------------------------------

ROWS = COLS = 160
PIXEL = 4.0
ORIGIN = (500000.0, 3750000.0)
EPSG = 32616

#: The tree under study: 10 m crown, 12 m canopy radius (3 cells at 4 m) at
#: the centre of the write window. Canopy-interior cells sit under vegetation
#: (``vegdsm2 > a`` there), which is what lets ``shadow()``'s step-1 raise
#: survive to the final planes.
TREE = TreeSpec("t1", ORIGIN[0] + 80.5 * PIXEL, ORIGIN[1] - 80.5 * PIXEL, 10.0, 12.0)
WRITE_WINDOW = RasterWindow(64, 96, 64, 96)

#: 12 m building block in the NE corner: tall enough to own the full-tile
#: amplitude (12 > 10), far enough that the exact occluder mask
#: (``read_window_for_write_window``) legitimately drops it from the narrowed
#: window's crop (its cells sit > 128 m from every target: 12 <= 128 m *
#: tan(6 deg) ~= 13.4 m, so no march comparison can flip through it).
BUILDING_VALUE = 12.0


def _scene_arrays(
    rows: int, cols: int, *, dem_value: float = 0.0, building_value: float | None = None
):
    dem = np.full((rows, cols), dem_value, dtype=np.float32)
    building = dem.copy()
    building[8:24, 116:152] = BUILDING_VALUE if building_value is None else building_value
    landcover = np.ones((rows, cols), dtype=np.uint8)
    return dem, building, landcover


def _vegetation(grid: RasterGrid, trees: tuple[TreeSpec, ...]) -> np.ndarray:
    from solweig_gpu.incremental.trees import rasterize_tree_patch

    vegetation = np.zeros((grid.rows, grid.cols), dtype=np.float32)
    for tree in trees:
        canopy, _trunk = rasterize_tree_patch(tree, grid, grid.full_window)
        np.maximum(vegetation, canopy, out=vegetation)
    return vegetation


def _write_site(
    root: Path,
    *,
    trees: tuple[TreeSpec, ...],
    dem_value: float = 0.0,
    building_value: float | None = None,
) -> RasterGrid:
    grid = RasterGrid(ROWS, COLS, PIXEL, ORIGIN[0], ORIGIN[1])
    from solweig_gpu.incremental.solver import _compose_scene_from_canopy  # noqa: F401

    site = root / "processed_inputs"
    dem, building, landcover = _scene_arrays(
        ROWS, COLS, dem_value=dem_value, building_value=building_value
    )
    vegetation = _vegetation(grid, trees)

    from tests.test_incremental_worker import _write_tif

    def tif(kind: str, array: np.ndarray) -> Path:
        return _write_tif(
            site / kind / f"{kind}_0_0.tif", array,
            origin=ORIGIN, pixel=PIXEL, epsg=EPSG,
        )

    tif("Building_DSM", building)
    tif("DEM", dem)
    tif("Trees", vegetation)
    tif("walls", np.zeros((ROWS, COLS), dtype=np.float32))
    tif("aspect", np.zeros((ROWS, COLS), dtype=np.float32))
    tif("Landcover", landcover)
    _write_met(site / "metfiles" / f"metfile_0_0_{DATE_STR}.txt", range(9, 15))
    return grid


def _make_site(
    tmp_factory,
    name: str,
    *,
    trees: tuple[TreeSpec, ...],
    dem_value: float = 0.0,
    building_value: float | None = None,
):
    root = tmp_factory.mktemp(name)
    grid = _write_site(
        root, trees=trees, dem_value=dem_value, building_value=building_value
    )
    _compute_baseline_svf(root / "processed_inputs")
    from tests.test_incremental_worker import _derive_lon_lat, _derive_utc_offset

    building = root / "processed_inputs" / "Building_DSM" / "Building_DSM_0_0.tif"
    lon, lat = _derive_lon_lat(building)
    utc = _derive_utc_offset(lat, lon, DATE_STR)
    from solweig_gpu.incremental.cache_builder import build_site_cache
    from solweig_gpu.incremental.cache import SiteCache

    build_site_cache(
        root / "processed_inputs",
        root / "cache",
        tile_key="0_0",
        site_id=name,
        latitude=lat,
        longitude=lon,
        # the oracle altitude rule: median DEM, clamped to 3 m when positive
        altitude_m=3.0 if dem_value > 0.0 else 0.0,
        utc_offset_hours=utc,
        met_file=root / "processed_inputs" / "metfiles" / f"metfile_0_0_{DATE_STR}.txt",
    )
    cache = SiteCache.load(root / "cache")
    forcing = load_site_forcing(
        cache, site_dir=root / "processed_inputs", selected_date_str=DATE_STR
    )
    layer = TreeLayer(cache.tree_base, grid, scenario_id=name)
    scene = compose_full_scene_tensors(cache, layer)
    return SimpleSite(root, grid, cache, forcing, layer, scene)


class SimpleSite:
    def __init__(self, root, grid, cache, forcing, layer, scene):
        self.root = root
        self.grid = grid
        self.cache = cache
        self.forcing = forcing
        self.layer = layer
        self.scene = scene


@pytest.fixture(scope="module")
def amplitude_site(tmp_path_factory):
    """Flat site, 12 m NE building, NO baseline tree — the edit adds TREE."""
    return _make_site(tmp_path_factory, "amplitude_site", trees=())


@pytest.fixture(scope="module")
def edited_site(amplitude_site):
    """The amplitude_site with TREE added (the edited scene under study)."""
    layer = TreeLayer(
        amplitude_site.cache.tree_base, amplitude_site.grid, scenario_id="sci"
    )
    assert layer.add_tree(TREE) is not False
    scene = compose_full_scene_tensors(amplitude_site.cache, layer)
    return SimpleSite(
        amplitude_site.root,
        amplitude_site.grid,
        amplitude_site.cache,
        amplitude_site.forcing,
        layer,
        scene,
    )


@pytest.fixture(scope="module")
def elevated_site(tmp_path_factory):
    """20 m base terrain, a 10 m tree on top, no building.

    The absolute stop is ``vegdem.max() = 30`` while the relative bound is
    ``30 - 20 = 10``: the 78-degree ring's first steps (dz_1 in ~[18.8,
    26.6] m at 4 m pixels) land strictly between them — the replay would
    run ONE step where the oracle runs two. This is the loud-refusal case.
    """
    tree = TreeSpec("base", ORIGIN[0] + 80.5 * PIXEL, ORIGIN[1] - 80.5 * PIXEL, 10.0, 12.0)
    return _make_site(tmp_path_factory, "elevated_site", trees=(tree,), dem_value=20.0)


# ---------------------------------------------------------------------------
# The construction (regime band between crop and full tile)
# ---------------------------------------------------------------------------


@pytest.mark.scientific
class TestConstruction:
    def test_narrowed_window_creates_a_march_regime_band(
        self, amplitude_site, edited_site
    ) -> None:
        """The reviewer's construction, pinned: crop amplitude < dz_1 <= full."""
        read = read_window_for_write_window(
            WRITE_WINDOW, edited_site.cache, edited_site.scene, edited_site.forcing
        )
        crop_amp = _crop_amplitude(edited_site.scene, read)
        full_amp = _full_amplitude(edited_site.scene)
        dz1 = _dz1_per_patch(1.0 / edited_site.cache.pixel_size_m)

        # the narrowed window legitimately excludes the 12 m building ...
        assert crop_amp == pytest.approx(10.0, abs=1e-4), (
            "the read-window crop should see only the 10 m tree"
        )
        # ... while the full tile keeps the building's 12 m
        assert full_amp == pytest.approx(12.0, abs=1e-4)
        # ... and some sky patch's first step lands strictly between
        band = np.nonzero((dz1 > crop_amp) & (dz1 <= full_amp))[0]
        assert band.size > 0, (
            "construction failed: no patch changes march regime between the "
            "narrowed crop and the full tile"
        )


# ---------------------------------------------------------------------------
# Parity: narrowed-window replay vs full-tile replay (the oracle twin)
# ---------------------------------------------------------------------------


def _bundle_at(site, window: RasterWindow, cube: RasterWindow):
    return window_svf_bundle(
        site.cache, site.scene, window, veg_changed=True, cube_window=cube
    )


def _assert_veg_terms_match(
    narrowed,
    full_tiled,
    narrowed_read: RasterWindow,
    full_read: RasterWindow,
    cube: RasterWindow,
    context: str,
) -> None:
    """Bitwise veg-term equality at the cube window, scalars at their crops.

    Both bundles carry scalars at their OWN read extent, so each side is
    sliced with the cube window expressed in that bundle's extent. Cubes
    arrive at the cube extent in both (``cube_window`` given) and compare
    directly.
    """

    def _scalar_slice(read: RasterWindow) -> tuple[slice, slice]:
        return (
            slice(cube.row_start - read.row_start, cube.row_stop - read.row_start),
            slice(cube.col_start - read.col_start, cube.col_stop - read.col_start),
        )

    left_sl = _scalar_slice(narrowed_read)
    right_sl = _scalar_slice(full_read)
    for name in VEG_SCALARS:
        left = narrowed[SVF_BUNDLE_INDEX[name]].numpy()[left_sl]
        right = full_tiled[SVF_BUNDLE_INDEX[name]].numpy()[right_sl]
        assert np.array_equal(left, right, equal_nan=True), (
            f"{context}: {name} differs between the narrowed-window replay "
            f"and the full-tile replay ({int((left != right).sum())} cells)"
        )
    for name in ("vegshmat", "vbshvegshmat"):
        left = narrowed[SVF_BUNDLE_INDEX[name]].numpy()
        right = full_tiled[SVF_BUNDLE_INDEX[name]].numpy()
        assert left.shape == right.shape == (
            cube.height, cube.width, left.shape[2],
        ), (
            f"{context}: {name} cube not at the cube-window extent"
        )
        assert np.array_equal(left, right, equal_nan=True), (
            f"{context}: {name} differs between the narrowed-window replay "
            f"and the full-tile replay ({int((left != right).sum())} cells); "
            f"narrowed value set {sorted(np.unique(left).tolist())} vs "
            f"full-tile {sorted(np.unique(right).tolist())}"
        )


@pytest.mark.scientific
class TestReplayParity:
    def test_narrowed_replay_matches_full_tile_replay(
        self, amplitude_site, edited_site
    ) -> None:
        """MEDIUM-2: the narrowed read window must not change the march regime.

        RED (pre-fix): the crop-relative amplitude stops the band patches
        after step 1 and ``vbshvegshmat`` carries 2.0 at canopy-interior
        cells where the full-tile replay (oracle regime) holds 1.0.
        """
        read = read_window_for_write_window(
            WRITE_WINDOW, edited_site.cache, edited_site.scene, edited_site.forcing
        )
        assert WRITE_WINDOW.contained_in(read) if hasattr(
            WRITE_WINDOW, "contained_in"
        ) else True
        narrowed = _bundle_at(edited_site, read, WRITE_WINDOW)
        full_tiled = _bundle_at(edited_site, edited_site.grid.full_window, WRITE_WINDOW)
        _assert_veg_terms_match(
            narrowed, full_tiled, read, edited_site.grid.full_window,
            WRITE_WINDOW, "MEDIUM-2",
        )

    def test_replay_planes_are_oracle_representable_at_every_window(
        self, amplitude_site, edited_site
    ) -> None:
        """vbsh == 2.0 is an ORACLE value at genuinely one-step patches (the
        step-1 accumulator zero leaves the raise un-buffered there), so the
        replay planes must carry exactly the oracle's value alphabet
        (vegsh in {0, 1}, vbsh in {0, 1, 2.0}) at every window — never an
        EXTRA 2.0 from a regime the oracle did not run.
        """
        read = read_window_for_write_window(
            WRITE_WINDOW, edited_site.cache, edited_site.scene, edited_site.forcing
        )
        twos = {}
        for label, window in (
            ("narrowed", read),
            ("full tile", edited_site.grid.full_window),
        ):
            bundle = _bundle_at(edited_site, window, WRITE_WINDOW)
            vegsh = np.unique(bundle[SVF_BUNDLE_INDEX["vegshmat"]].numpy())
            assert set(vegsh.tolist()) <= {0.0, 1.0}, (
                f"{label} window: vegshmat carries non-binary values {vegsh}"
            )
            vbsh = np.unique(bundle[SVF_BUNDLE_INDEX["vbshvegshmat"]].numpy())
            assert set(vbsh.tolist()) <= {0.0, 1.0, 2.0}, (
                f"{label} window: vbshvegshmat carries impossible values {vbsh}"
            )
            twos[label] = int(
                (bundle[SVF_BUNDLE_INDEX["vbshvegshmat"]].numpy() == 2.0).sum()
            )
        assert twos["narrowed"] == twos["full tile"], (
            f"the narrowed window's one-step raise count ({twos['narrowed']} "
            f"vbsh==2.0 cells) differs from the full-tile oracle regime's "
            f"({twos['full tile']}): the read window changed the march regime"
        )

    def test_windowed_solve_matches_oracle_end_to_end(
        self, amplitude_site, edited_site
    ) -> None:
        """The physics sees the fold: solve_window == run_full_tile bitwise."""
        read = read_window_for_write_window(
            WRITE_WINDOW, edited_site.cache, edited_site.scene, edited_site.forcing
        )
        solved = solve_window(
            edited_site.cache,
            edited_site.layer,
            read_window=read,
            write_window=WRITE_WINDOW,
            forcing=edited_site.forcing,
            requested_variables=VARIABLES,
            veg_changed=True,
        )
        oracle = run_full_tile(
            edited_site.cache,
            edited_site.layer,
            forcing=edited_site.forcing,
            site_dir=edited_site.root / "processed_inputs",
            scratch_dir=edited_site.root / "scratch_oracle",
            requested_variables=VARIABLES,
        )
        for name in VARIABLES:
            cand = solved[name]
            ref = oracle[name][:, WRITE_WINDOW.row_start:WRITE_WINDOW.row_stop,
                               WRITE_WINDOW.col_start:WRITE_WINDOW.col_stop]
            assert cand.shape == ref.shape, (
                f"{name}: solve_window returned {cand.shape}, expected the "
                f"write-window crop {ref.shape}"
            )
            assert np.array_equal(cand, ref, equal_nan=True), (
                f"{name}: windowed solve differs from the oracle at "
                f"{int((cand != ref).sum())} values inside the write window"
            )


# ---------------------------------------------------------------------------
# The regime fence (residual one-step-vs-multi-step corner)
# ---------------------------------------------------------------------------


@pytest.mark.scientific
class TestBandedEscalation:
    """R6: banded scenes SERVE via escalated marches, bitwise == the oracle.

    RED at the R6 base commit: ``window_svf_bundle`` / ``solve_window``
    raised :class:`MarchRegimeError` on these scenes. The adjudicated
    replacement marches the banded patches at the oracle's absolute stop
    ``scene.amaxvalue`` — identical executed step set per patch, so the
    serve equals the oracle twin bitwise BY CONSTRUCTION.
    """

    def test_fence_is_typed_for_worker_routing(self) -> None:
        from solweig_gpu.incremental import solver as solver_mod

        assert issubclass(solver_mod.MarchRegimeError, SolverInputError)

    def test_banded_scene_serves_and_matches_the_oracle_twin(
        self, elevated_site
    ) -> None:
        """dz_1 in (A_eff, A_abs] -> escalate those patches, serve bitwise."""
        scene = elevated_site.scene
        amp = _full_amplitude(scene)
        dz1 = _dz1_per_patch(1.0 / elevated_site.cache.pixel_size_m)
        band = np.nonzero(
            (dz1 > amp) & (dz1 <= float(scene.amaxvalue))
        )[0]
        assert band.size > 0, "elevated fixture lost its regime band"
        read = read_window_for_write_window(
            WRITE_WINDOW, elevated_site.cache, scene, elevated_site.forcing
        )
        narrowed = _bundle_at(elevated_site, read, WRITE_WINDOW)
        full_tiled = _bundle_at(
            elevated_site, elevated_site.grid.full_window, WRITE_WINDOW
        )
        _assert_veg_terms_match(
            narrowed, full_tiled, read, elevated_site.grid.full_window,
            WRITE_WINDOW, "escalated",
        )

    def test_banded_scene_value_alphabet_matches_the_oracle_regime(
        self, elevated_site
    ) -> None:
        """The escalated replay must carry the ORACLE's value alphabet.

        Like the MEDIUM-2 fence before it: no EXTRA vbsh==2.0 from a regime
        the oracle did not run, and the one-step 2.0 count equal between
        the narrowed escalated replay and the full-tile twin.
        """
        read = read_window_for_write_window(
            WRITE_WINDOW, elevated_site.cache, elevated_site.scene,
            elevated_site.forcing,
        )
        twos = {}
        for label, window in (
            ("narrowed", read),
            ("full tile", elevated_site.grid.full_window),
        ):
            bundle = _bundle_at(elevated_site, window, WRITE_WINDOW)
            vegsh = np.unique(bundle[SVF_BUNDLE_INDEX["vegshmat"]].numpy())
            assert set(vegsh.tolist()) <= {0.0, 1.0}, (
                f"{label} window: vegshmat carries non-binary values {vegsh}"
            )
            vbsh = np.unique(bundle[SVF_BUNDLE_INDEX["vbshvegshmat"]].numpy())
            assert set(vbsh.tolist()) <= {0.0, 1.0, 2.0}, (
                f"{label} window: vbshvegshmat carries impossible values {vbsh}"
            )
            twos[label] = int(
                (bundle[SVF_BUNDLE_INDEX["vbshvegshmat"]].numpy() == 2.0).sum()
            )
        assert twos["narrowed"] == twos["full tile"], (
            f"escalated narrowed window's one-step count ({twos['narrowed']} "
            f"vbsh==2.0 cells) differs from the full-tile oracle regime's "
            f"({twos['full tile']})"
        )

    def test_banded_scene_solve_matches_oracle_end_to_end(
        self, elevated_site
    ) -> None:
        """The physics sees the fold: escalated solve_window == oracle."""
        oracle = run_full_tile(
            elevated_site.cache,
            elevated_site.layer,
            forcing=elevated_site.forcing,
            site_dir=elevated_site.root / "processed_inputs",
            scratch_dir=elevated_site.root / "scratch_oracle",
            requested_variables=VARIABLES,
        )
        read = read_window_for_write_window(
            WRITE_WINDOW, elevated_site.cache, elevated_site.scene,
            elevated_site.forcing,
        )
        for label, read_window in (
            ("narrowed", read),
            ("full tile", elevated_site.grid.full_window),
        ):
            solved = solve_window(
                elevated_site.cache,
                elevated_site.layer,
                read_window=read_window,
                write_window=WRITE_WINDOW,
                forcing=elevated_site.forcing,
                requested_variables=VARIABLES,
                veg_changed=True,
            )
            for name in VARIABLES:
                cand = solved[name]
                ref = oracle[name][
                    :, WRITE_WINDOW.row_start:WRITE_WINDOW.row_stop,
                    WRITE_WINDOW.col_start:WRITE_WINDOW.col_stop
                ]
                assert cand.shape == ref.shape, (
                    f"{label}/{name}: solve_window returned {cand.shape}, "
                    f"expected the write-window crop {ref.shape}"
                )
                assert np.array_equal(cand, ref, equal_nan=True), (
                    f"{label}/{name}: escalated banded solve differs from "
                    f"the oracle at {int((cand != ref).sum())} values"
                )

    def test_flat_site_never_refuses(self, amplitude_site, edited_site) -> None:
        """Flat ground: A_eff == A_abs -> the band is empty -> serve."""
        read = read_window_for_write_window(
            WRITE_WINDOW, edited_site.cache, edited_site.scene, edited_site.forcing
        )
        # would raise if the fence misfired
        _bundle_at(edited_site, read, WRITE_WINDOW)

    def test_one_step_scenes_are_served_and_match_the_oracle(
        self, tmp_path_factory
    ) -> None:
        """A genuinely one-step oracle regime is fine: replay follows it.

        No building, 2 m tree on flat ground: the 78-degree ring's dz_1
        (~18.8-26.6 m) exceeds even the absolute stop (2 m), so oracle and
        replay are one-step there TOGETHER — serve, and match the oracle
        bitwise end to end.
        """
        site = _make_site(
            tmp_path_factory, "onestep_site", building_value=0.0,
            trees=(TreeSpec("s1", ORIGIN[0] + 80.5 * PIXEL,
                            ORIGIN[1] - 80.5 * PIXEL, 2.0, 12.0),),
        )
        write = RasterWindow(72, 90, 72, 90)
        read = read_window_for_write_window(write, site.cache, site.scene, site.forcing)
        bundle = window_svf_bundle(
            site.cache, site.scene, read, veg_changed=True, cube_window=write
        )
        # the one-step regime IS representable in the replay planes: they
        # match the oracle's planes (both one-step at the same patches), so
        # any 2.0 the oracle holds must appear identically here
        full_bundle = window_svf_bundle(
            site.cache, site.scene, site.grid.full_window,
            veg_changed=True, cube_window=write,
        )
        _assert_veg_terms_match(
            bundle, full_bundle, read, site.grid.full_window, write, "one-step"
        )
        solved = solve_window(
            site.cache, site.layer, read_window=read, write_window=write,
            forcing=site.forcing, requested_variables=VARIABLES, veg_changed=True,
        )
        oracle = run_full_tile(
            site.cache, site.layer, forcing=site.forcing,
            site_dir=site.root / "processed_inputs",
            scratch_dir=site.root / "scratch_oracle",
            requested_variables=VARIABLES,
        )
        for name in VARIABLES:
            cand = solved[name]
            ref = oracle[name][:, write.row_start:write.row_stop,
                               write.col_start:write.col_stop]
            assert cand.shape == ref.shape, (
                f"{name}: solve_window returned {cand.shape}, expected the "
                f"write-window crop {ref.shape}"
            )
            assert np.array_equal(cand, ref, equal_nan=True), (
                f"{name}: one-step scene solve differs from the oracle"
            )


@pytest.mark.scientific
class TestEscalationSensitivity:
    """Mutation-B fence (r6 review fixup): replay vs the TRUE oracle.

    Every other parity test in this file compares the narrowed replay
    against a FULL-WINDOW REPLAY of the same solver code — deleting the
    escalation mutates both sides together and those comparisons stay
    byte-identical (the reviewer demonstrated: removing the SVF
    escalation entirely keeps all 20 of them green). This class pins the
    replay against ``svf_calculator`` itself, whose marches run at the
    absolute ``scene.amaxvalue`` no matter what the solver does.

    The value channel is ``shadow()``'s index-1 block: it zeroes the
    ``vbshvegsh`` accumulator AFTER the step-1 raise, so a ONE-step
    march finishes with ``vbsh = 1 + vegsh`` (the 2.0 signature at
    raised cells) while a multi-step march re-accumulates and finishes
    at ``vbsh = vegsh`` in {0, 1}. Dropping the escalation (banded
    patches one-step at A_eff) writes 2.0 into exactly the cells the
    witness assertion below counts — this is the test that flips bytes.
    """

    def test_banded_replay_matches_the_true_svf_calculator_oracle(
        self, elevated_site
    ) -> None:
        scene = elevated_site.scene
        scale = 1.0 / elevated_site.cache.pixel_size_m
        amp = _full_amplitude(scene)
        dz1 = _dz1_per_patch(scale)
        band = np.nonzero((dz1 > amp) & (dz1 <= float(scene.amaxvalue)))[0]
        assert band.size > 0, "elevated fixture lost its regime band"

        oracle = svf_calculator(
            2, scene.amaxvalue, scene.a, scene.vegdsm, scene.vegdsm2,
            scene.bush, scale, save_rasters=False,
        )
        oracle_vbsh = oracle[SVF_BUNDLE_INDEX["vbshvegshmat"]].numpy()[
            WRITE_WINDOW.row_start:WRITE_WINDOW.row_stop,
            WRITE_WINDOW.col_start:WRITE_WINDOW.col_stop, :,
        ]
        read = read_window_for_write_window(
            WRITE_WINDOW, elevated_site.cache, scene, elevated_site.forcing
        )
        replay = _bundle_at(elevated_site, read, WRITE_WINDOW)
        replay_vbsh = replay[SVF_BUNDLE_INDEX["vbshvegshmat"]].numpy()

        assert np.array_equal(oracle_vbsh, replay_vbsh, equal_nan=True), (
            "the escalated replay's vbshvegshmat cube deviates from the "
            "TRUE svf_calculator oracle at "
            f"{int((oracle_vbsh != replay_vbsh).sum())} cells"
        )
        # Multi-step signature at the banded patches: no 2.0 anywhere the
        # oracle marches two-plus steps, and the step-1 raise FIRES inside
        # the write window — the cells a one-step march would flip to 2.0.
        ones = int((oracle_vbsh[..., band] == 1.0).sum())
        assert ones > 0, (
            "fixture lost its sensitivity: no raised cell at any banded "
            "patch inside the write window, so a one-step march could not "
            "flip bytes here and this fence would be vacuous"
        )
        assert 2.0 not in oracle_vbsh[..., band], (
            "the TRUE oracle carries vbsh == 2.0 at a banded patch; it "
            "marches multi-step there, so the 2.0 signature is impossible "
            "and the fixture's band computation disagrees with the oracle"
        )


# ---------------------------------------------------------------------------
# Shape/scale sweep: non-square tiles, non-power-of-two scales
# ---------------------------------------------------------------------------


def _sweep_site(
    tmp_factory, name: str, rows: int, cols: int, pixel: float,
    *, dem_value: float = 0.0, tree_height: float = 3.0,
):
    """Non-square site: tree at centre, 5 m building block far NE.

    At 1 m pixels the 66/78-degree dz_1 values land in (3, 5]; at 3 m
    pixels (scale 1/3, non-power-of-two) the 42/54-degree rings do — both
    scales exercise the band machinery at the association-exact dz_1.
    ``dem_value > 0`` builds the BANDED variant (R6): a 20 m base with a
    25 m block and a 10 m tree gives A_eff = 30 - 20 = 10 <
    dz_1(78 deg, ds ~ 1.41) ~= 19.9 <= A_abs = 30 at scale 1/3 — the
    escalated serve must hold at the non-power-of-two scale too.
    """
    root = tmp_factory.mktemp(name)
    grid = RasterGrid(rows, cols, pixel, ORIGIN[0], ORIGIN[1])
    tree = TreeSpec(
        "t1", ORIGIN[0] + (rows // 2 + 0.5) * pixel,
        ORIGIN[1] - (cols // 2 + 0.5) * pixel, tree_height, 3.0 * pixel,
    )
    site = root / "processed_inputs"
    dem = np.full((rows, cols), dem_value, dtype=np.float32)
    building = dem.copy()
    building[2:6, cols - 8 : cols - 2] = 25.0 if dem_value > 0.0 else 5.0
    vegetation = _vegetation(grid, (tree,))

    from tests.test_incremental_worker import _write_tif

    def tif(kind: str, array: np.ndarray) -> Path:
        return _write_tif(
            site / kind / f"{kind}_0_0.tif", array,
            origin=ORIGIN, pixel=pixel, epsg=EPSG,
        )

    tif("Building_DSM", building)
    tif("DEM", dem)
    tif("Trees", vegetation)
    tif("walls", np.zeros((rows, cols), dtype=np.float32))
    tif("aspect", np.zeros((rows, cols), dtype=np.float32))
    tif("Landcover", np.ones((rows, cols), dtype=np.uint8))
    _write_met(site / "metfiles" / f"metfile_0_0_{DATE_STR}.txt", range(9, 15))
    _compute_baseline_svf(site)

    from tests.test_incremental_worker import _derive_lon_lat, _derive_utc_offset

    building_tif = site / "Building_DSM" / "Building_DSM_0_0.tif"
    lon, lat = _derive_lon_lat(building_tif)
    utc = _derive_utc_offset(lat, lon, DATE_STR)
    from solweig_gpu.incremental.cache_builder import build_site_cache
    from solweig_gpu.incremental.cache import SiteCache

    build_site_cache(
        site, root / "cache", tile_key="0_0", site_id=name,
        latitude=lat, longitude=lon,
        # the oracle altitude rule: median DEM, clamped to 3 m when positive
        altitude_m=3.0 if dem_value > 0.0 else 0.0, utc_offset_hours=utc,
        met_file=site / "metfiles" / f"metfile_0_0_{DATE_STR}.txt",
    )
    cache = SiteCache.load(root / "cache")
    forcing = load_site_forcing(cache, site_dir=site, selected_date_str=DATE_STR)
    layer = TreeLayer(cache.tree_base, grid, scenario_id=name)
    return SimpleSite(root, grid, cache, forcing, layer,
                      compose_full_scene_tensors(cache, layer))


@pytest.mark.scientific
class TestShapeScaleParity:
    @pytest.mark.parametrize(
        ("rows", "cols", "pixel"),
        [
            (48, 96, 1.0),   # 2:1 wide, power-of-two scale
            (96, 48, 1.0),   # 1:2 tall
            (48, 96, 3.0),   # scale 1/3 (non-power-of-two, NEW-LOW hazard)
            (96, 48, 3.0),
        ],
    )
    def test_narrowed_matches_full_tiled(self, tmp_path_factory, rows, cols, pixel) -> None:
        site = _sweep_site(tmp_path_factory, f"sweep_{rows}x{cols}@{pixel}", rows, cols, pixel)
        centre = rows // 2
        write = RasterWindow(centre - 8, centre + 8, cols // 2 - 8, cols // 2 + 8)
        read = read_window_for_write_window(write, site.cache, site.scene, site.forcing)
        narrowed = _bundle_at(site, read, write)
        full_tiled = _bundle_at(site, site.grid.full_window, write)
        _assert_veg_terms_match(
            narrowed, full_tiled, read, site.grid.full_window, write,
            f"{rows}x{cols}@{pixel}",
        )

    @pytest.mark.parametrize(
        ("rows", "cols", "pixel"),
        [
            (48, 96, 3.0),   # banded at the non-power-of-two scale (R6)
            (96, 48, 3.0),
        ],
    )
    def test_banded_narrowed_matches_full_tiled(
        self, tmp_path_factory, rows, cols, pixel
    ) -> None:
        """R6: the escalated serve holds on non-square, scale-1/3 banded tiles.

        RED at the R6 base commit: both windows raise MarchRegimeError.
        """
        site = _sweep_site(
            tmp_path_factory, f"sweepb_{rows}x{cols}@{pixel}",
            rows, cols, pixel, dem_value=20.0, tree_height=10.0,
        )
        dz1 = _dz1_per_patch(1.0 / pixel)
        band = np.nonzero(
            (dz1 > _full_amplitude(site.scene))
            & (dz1 <= float(site.scene.amaxvalue))
        )[0]
        assert band.size > 0, "banded sweep fixture lost its regime band"
        centre = rows // 2
        write = RasterWindow(centre - 8, centre + 8, cols // 2 - 8, cols // 2 + 8)
        read = read_window_for_write_window(write, site.cache, site.scene, site.forcing)
        narrowed = _bundle_at(site, read, write)
        full_tiled = _bundle_at(site, site.grid.full_window, write)
        _assert_veg_terms_match(
            narrowed, full_tiled, read, site.grid.full_window, write,
            f"banded {rows}x{cols}@{pixel}",
        )


# ---------------------------------------------------------------------------
# R6: time-loop march amplitude escalation (the latent solver note, closed)
# ---------------------------------------------------------------------------


@pytest.mark.scientific
class TestTimeLoopEscalation:
    """The time loop marches at the windowed effective amplitude today; a
    timestep whose SUN first step lands in (A_eff_window, A_abs] would run
    one step where the oracle runs several — previously immune only via a
    missing-accumulator quirk of ``shadowingfunction_wallheight_23``. The
    R6 closure escalates such loops to the same absolute-amplitude source
    as the SVF replay (``scene.amaxvalue``).
    """

    def test_sun_band_exists_on_the_edited_scene(self, edited_site) -> None:
        """Construction pin: t=3 (alt ~69.7 deg) lands in (10, 12] at 4 m."""
        read = read_window_for_write_window(
            WRITE_WINDOW, edited_site.cache, edited_site.scene,
            edited_site.forcing,
        )
        window_amp = _window_amplitude(edited_site.scene, read)
        absolute = float(edited_site.scene.amaxvalue)
        assert window_amp < absolute, (
            "fixture lost its window-vs-absolute amplitude gap"
        )
        dz1 = _sun_dz1_per_timestep(
            edited_site.forcing, 1.0 / edited_site.cache.pixel_size_m
        )
        banded = [
            i for i, dz in enumerate(dz1)
            if dz is not None and window_amp < dz <= absolute
        ]
        assert banded, (
            "construction failed: no daytime timestep changes march regime "
            "between the windowed effective amplitude and the absolute stop"
        )

    def test_time_loop_band_predicate_escalates(
        self, edited_site
    ) -> None:
        """solver.time_loop_band_present: True on the banded window, False
        when the window sees the whole tile (A_eff_window == A_abs)."""
        from solweig_gpu.incremental import solver as solver_mod

        read = read_window_for_write_window(
            WRITE_WINDOW, edited_site.cache, edited_site.scene,
            edited_site.forcing,
        )
        assert solver_mod.time_loop_band_present(
            _window_amplitude(edited_site.scene, read),
            float(edited_site.scene.amaxvalue),
            1.0 / edited_site.cache.pixel_size_m,
            edited_site.forcing.altitude,
            edited_site.forcing.azimuth,
        )
        assert not solver_mod.time_loop_band_present(
            _window_amplitude(edited_site.scene, edited_site.grid.full_window),
            float(edited_site.scene.amaxvalue),
            1.0 / edited_site.cache.pixel_size_m,
            edited_site.forcing.altitude,
            edited_site.forcing.azimuth,
        )

    def test_time_loop_respects_the_requested_prefix(
        self, edited_site
    ) -> None:
        """The band predicate only looks at timesteps the loop will run."""
        from solweig_gpu.incremental import solver as solver_mod

        read = read_window_for_write_window(
            WRITE_WINDOW, edited_site.cache, edited_site.scene,
            edited_site.forcing,
        )
        window_amp = _window_amplitude(edited_site.scene, read)
        absolute = float(edited_site.scene.amaxvalue)
        dz1 = _sun_dz1_per_timestep(
            edited_site.forcing, 1.0 / edited_site.cache.pixel_size_m
        )
        banded = [
            i for i, dz in enumerate(dz1)
            if dz is not None and window_amp < dz <= absolute
        ]
        first_banded = min(banded)
        # a prefix that stops before the first banded timestep sees no band
        assert not solver_mod.time_loop_band_present(
            window_amp, absolute,
            1.0 / edited_site.cache.pixel_size_m,
            edited_site.forcing.altitude, edited_site.forcing.azimuth,
            time_stop=first_banded,
        )
        # one that includes it does
        assert solver_mod.time_loop_band_present(
            window_amp, absolute,
            1.0 / edited_site.cache.pixel_size_m,
            edited_site.forcing.altitude, edited_site.forcing.azimuth,
            time_stop=first_banded + 1,
        )

    def test_banded_timestep_marches_are_value_inert(self) -> None:
        """Why there is no byte-flip fence for the time-loop escalation.

        At a BANDED timestep (sun dz_1 in (windowed effective, absolute])
        no march body — step 1 included — can raise anything in ANY
        channel of ``shadowingfunction_wallheight_23``: every raise
        needs an occluder value above ``a[target] + dz_i`` with
        ``dz_i >= dz_1``, but the windowed effective amplitude bounds
        ``max(a, vegdsm, vegdsm2) - min(a) < dz_1`` over the marched
        crop, for the building, canopy and trunk channels alike. And
        unlike ``shadow()``, this kernel re-derives its final
        ``vbsh = 1 - (clamped accumulator - vegsh)`` from ``vegsh``
        alone, so the accumulated step count never reaches the outputs
        (the ``shadow()`` index-1 accumulator zero is exactly what makes
        the SVF escalation value-load-bearing instead). The time-loop
        escalation is therefore regime-parity insurance — marches
        step-identical to the oracle BY CONSTRUCTION, immune today only
        through this algebra — and this pin verifies the inertness the
        insurance rests on: a banded construction yields bitwise-
        identical outputs at the windowed and the absolute stop.
        """
        from solweig_gpu.solweig import shadowingfunction_wallheight_23

        n = 48
        a = torch.zeros((n, n))
        a[20, 20] = 2.0  # building relief below dz_1 (still consulted)
        vegdem = torch.zeros((n, n))
        vegdem[24, 18] = 1.5  # canopy relief below dz_1 (still consulted)
        vegdem2 = vegdem.clone() * 0.25
        zeros = torch.zeros((n, n))
        # az 135 deg -> ds = sqrt(2); alt 60 deg -> dz_1 = sqrt(2)*tan(60)
        # ~ 2.449 m. Window amplitude = max(a, vegdem, vegdem2) - min(a) =
        # 2.0 < dz_1 (banded); absolute stop 10 runs bodies 1-5
        # (dz_5 ~ 9.80 <= 10 < dz_6 ~ 12.25).
        def outputs(amaxvalue):
            return shadowingfunction_wallheight_23(
                a, vegdem, vegdem2, 135.0, 60.0, 1.0, amaxvalue,
                zeros, zeros, zeros,
            )

        windowed = outputs(2.0)
        absolute = outputs(10.0)
        for index, (short, long) in enumerate(zip(windowed, absolute)):
            assert torch.equal(short, long), (
                f"time-loop march output plane {index} differs between the "
                "windowed and the absolute stop on a BANDED timestep; the "
                "value-inertness argument behind the escalation is false"
            )

    def test_banded_scene_time_loop_matches_oracle(
        self, elevated_site
    ) -> None:
        """End-to-end on a scene with BOTH bands: t=4/5 sun dz_1 in
        (A_eff_w, A_abs] and 78-degree sky patches in the SVF band. The
        escalated solve must match the oracle bitwise through the time
        loop as well (covered jointly with the SVF escalation above; this
        pin isolates the time-loop band existence on the same fixture)."""
        scene = elevated_site.scene
        read = read_window_for_write_window(
            WRITE_WINDOW, elevated_site.cache, scene, elevated_site.forcing
        )
        window_amp = _window_amplitude(scene, read)
        absolute = float(scene.amaxvalue)
        dz1 = _sun_dz1_per_timestep(
            elevated_site.forcing, 1.0 / elevated_site.cache.pixel_size_m
        )
        banded = [
            i for i, dz in enumerate(dz1)
            if dz is not None and window_amp < dz <= absolute
        ]
        assert banded, "elevated fixture lost its time-loop band"
