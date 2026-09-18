# SPDX-License-Identifier: GPL-3.0-only
"""Perf wave 2: per-patch march windows + invariant hoists in the veg-SVF replay.

The vegetation SVF replay (``_recompute_veg_svf_window``) replays
``svf_calculator``'s 153-patch accumulation loop after every tree edit; at
W1 HEAD every patch marches over the FULL read window even though a patch at
azimuth theta only ever reads cells in ONE quadrant of itself. Wave 2:

* **Lever 1 (per-patch march windows)** — each patch's ``shadow()`` march
  runs on ``bbox(accumulate window expanded by the patch's march reach in
  its READ direction)`` clipped to the read window. The march is a pure
  per-cell function of the input rasters (no cross-cell state), so cells
  inside the accumulate window see IDENTICAL arithmetic — same reads, same
  step sequence, same accumulation order — while the far quadrants of the
  read window are simply not marched. Gated OFF when the bush layer is
  non-zero anywhere in the read window (``shadow()``'s bush branches are
  global reductions over the marched extent; a shrunk extent could flip
  their predicates — refuse rather than risk it).
* **Lever 2 (invariant hoists)** — the sky-patch geometry
  (``create_patches(2)``, the ``iazimuth`` fill, ``aziintervalaniso``) and
  every ``annulus_weight(k, ...)`` value depend only on constants, so they
  are memoized per process instead of being rebuilt per solve / per patch.
  Pure recomputation elimination: identical values, identical arithmetic.

Every optimization here is bit-identical by construction and asserted
bitwise against the un-windowed replay (march windowing A/B, cube-window
vs legacy bundle crops, and the pre-existing worker-vs-oracle differential
suite that runs on top of this path).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from solweig_gpu.incremental.geometry import RasterWindow

from tests.test_incremental_worker import (  # noqa: F401  (fixture)
    SVF_BUNDLE_INDEX,
    _state_layer,
    science_site,
)

# The window pair used by the wave-1 bundle-equivalence test: a read window
# with a comfortably smaller interior write window, on the science site.
READ_WINDOW = RasterWindow(40, 140, 40, 140)
WRITE_WINDOW = RasterWindow(60, 100, 70, 120)


def _bundle(scene_cache, scene, *, cube_window):
    from solweig_gpu.incremental.solver import window_svf_bundle

    return window_svf_bundle(
        scene_cache, scene, READ_WINDOW, veg_changed=True,
        cube_window=cube_window,
    )


def _edited_scene(science_site):
    from solweig_gpu.incremental.solver import compose_full_scene_tensors

    return compose_full_scene_tensors(
        science_site.cache_a, _state_layer(science_site, "add")
    )


def _write_crop(window: RasterWindow) -> tuple[slice, slice]:
    return (
        slice(window.row_start - READ_WINDOW.row_start,
              window.row_stop - READ_WINDOW.row_start),
        slice(window.col_start - READ_WINDOW.col_start,
              window.col_stop - READ_WINDOW.col_start),
    )


_SCALARS = frozenset(
    {
        "svfveg", "svfEveg", "svfSveg", "svfWveg", "svfNveg",
        "svfaveg", "svfEaveg", "svfSaveg", "svfWaveg", "svfNaveg",
        "svftotal",
    }
)


def _assert_bundle_matches_legacy(windowed, legacy, cube: RasterWindow) -> None:
    """Bitwise oracle: windowed bundle vs un-windowed replay.

    Scalars keep the read extent in both bundles (the windowed one is
    zero-padded outside the accumulate window — values there are never
    consumed); cubes arrive at the cube extent in the windowed bundle and
    must equal the legacy read-extent crop.
    """
    sl = _write_crop(cube)
    for name in (
        "svfveg", "svfEveg", "svfSveg", "svfWveg", "svfNveg",
        "svfaveg", "svfEaveg", "svfSaveg", "svfWaveg", "svfNaveg",
        "svftotal",
    ):
        position = SVF_BUNDLE_INDEX[name]
        assert np.array_equal(
            windowed[position].numpy()[sl],
            legacy[position].numpy()[sl],
            equal_nan=True,
        ), f"{name}: windowed replay differs from the legacy crop"
    # the padding outside the accumulate window is exactly zero and the
    # legacy value there is NOT consumed anywhere downstream
    padded = windowed[SVF_BUNDLE_INDEX["svfveg"]].numpy().copy()
    padded[sl] = 0.0
    assert not padded.any(), "scalar padding outside the cube must be zero"
    for name in ("vegshmat", "vbshvegshmat"):
        position = SVF_BUNDLE_INDEX[name]
        assert windowed[position].shape == (
            cube.height, cube.width, legacy[position].shape[2],
        ), f"{name}: cube not at the requested extent"
        assert np.array_equal(
            windowed[position].numpy(),
            legacy[position].numpy()[sl],
            equal_nan=True,
        ), f"{name}: windowed cube differs from the legacy crop"


# ---------------------------------------------------------------------------
# Lever 1 units: march reach bound + read-direction classification
# ---------------------------------------------------------------------------


class TestPatchMarchReachBound:
    @staticmethod
    def _reach(amplitude_m, scale, altitude_deg):
        from solweig_gpu.incremental.solver import _patch_march_reach_pixels

        return _patch_march_reach_pixels(amplitude_m, scale, altitude_deg)

    def test_flat_site_runs_one_step(self) -> None:
        assert self._reach(0.0, 1.0, 6.0) == 1

    def test_reach_scales_with_amplitude_and_scale(self) -> None:
        low = self._reach(10.0, 0.5, 6.0)
        high = self._reach(20.0, 0.5, 6.0)
        assert high > low
        # amplitude 20 m, 0.5 px/m, tan(6 deg): ~95 steps (+1 loop margin)
        assert low == math.ceil(10.0 * 0.5 / math.tan(math.radians(6.0))) + 1
        assert high == math.ceil(20.0 * 0.5 / math.tan(math.radians(6.0))) + 1

    def test_higher_altitude_patches_reach_less(self) -> None:
        assert self._reach(30.0, 1.0, 18.0) < self._reach(30.0, 1.0, 6.0)

    def test_zenith_patch_reaches_one_cell(self) -> None:
        assert self._reach(30.0, 1.0, 90.0) == 1


class TestQuadrantReadDirection:
    @staticmethod
    def _direction(azimuth_deg):
        from solweig_gpu.incremental.solver import _quadrant_read_direction

        return _quadrant_read_direction(azimuth_deg)

    def test_cardinal_directions(self) -> None:
        # shadow(): dx is the ROW shift (= -sign(cos)), dy is the COL shift
        # (= +sign(sin)); a target reads cells at (r + dx, c + dy).
        assert self._direction(0.0) == (-1, 0)     # north: rows above
        assert self._direction(90.0) == (0, 1)     # east: cols right
        assert self._direction(180.0) == (1, 0)    # south: rows below
        assert self._direction(270.0) == (0, -1)   # west: cols left

    def test_intercardinal_directions(self) -> None:
        assert self._direction(45.0) == (-1, 1)
        assert self._direction(135.0) == (1, 1)
        assert self._direction(225.0) == (1, -1)
        assert self._direction(315.0) == (-1, -1)

    def test_360_wraps_to_north(self) -> None:
        assert self._direction(360.0) == (-1, 0)

    def test_boundary_azimuths_classify_conservatively(self) -> None:
        # Near a sin/cos zero the classification must never point the
        # WRONG way: an epsilon-neighbour may classify as 0 or as the tiny
        # true sign (both safe -- the tiny-sign shift rounds to zero
        # anyway, so the extra expansion is inert over-cover), never as
        # the opposite quadrant.
        for az in (1e-9, 1e-9 - 360.0):          # just above north
            row, col = self._direction(az)
            assert row in (-1, 0)
            assert col in (0, 1)
        az = 180.0 - 1e-9                         # just south of south
        row, col = self._direction(az)
        assert row in (0, 1)
        assert col in (0, 1)
        for az in (90.0 + 1e-9, 270.0 - 1e-9):
            row, col = self._direction(az)
            assert row in (0, 1)
            assert col in (-1, 0, 1)


class TestPatchMarchWindow:
    @staticmethod
    def _window(acc, read, azimuth_deg, reach):
        from solweig_gpu.incremental.solver import _patch_march_window

        return _patch_march_window(acc, read, azimuth_deg=azimuth_deg,
                                   reach_pixels=reach)

    def test_expands_only_into_the_read_direction(self) -> None:
        acc = RasterWindow(60, 100, 70, 120)
        read = RasterWindow(40, 140, 40, 140)
        # azimuth 135 reads rows below and cols right; expansions go only
        # into those directions, clamped to the read window (140)
        window = self._window(acc, read, 135.0, 30)
        assert window == RasterWindow(60, 130, 70, 140)

    def test_clamps_to_the_read_window(self) -> None:
        acc = RasterWindow(45, 55, 45, 55)
        read = RasterWindow(40, 140, 40, 140)
        window = self._window(acc, read, 225.0, 500)  # reads rows below, cols left
        assert window == RasterWindow(45, 140, 40, 55)

    def test_contains_the_accumulate_window(self) -> None:
        acc = RasterWindow(60, 100, 70, 120)
        read = RasterWindow(40, 140, 40, 140)
        for azimuth in (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0):
            window = self._window(acc, read, azimuth, 25)
            assert window.row_start <= acc.row_start
            assert window.row_stop >= acc.row_stop
            assert window.col_start <= acc.col_start
            assert window.col_stop >= acc.col_stop


# ---------------------------------------------------------------------------
# Lever 1 (scientific): march grids actually shrink, stay covering, and the
# windowed replay is bitwise the un-windowed one
# ---------------------------------------------------------------------------


@pytest.mark.scientific
class TestPatchMarchWindows:
    @staticmethod
    def _march_grids(science_site, scene, monkeypatch):
        # T19b: the replay marches through march_router.shadow_march (numba
        # lane by default, torch on refusal/flag) — the router is the seam
        # every lane crosses, so the spy lives there. solver.shadow_fn alone
        # is only the torch fallback callable and is not called on the
        # default lane.
        import solweig_gpu.incremental.march_router as march_router_mod

        grids = []
        original = march_router_mod.shadow_march

        def spy(amaxvalue, a, vegdem, vegdem2, bush, azimuth, altitude, scale,
                **kwargs):
            grids.append(tuple(a.shape))
            return original(amaxvalue, a, vegdem, vegdem2, bush, azimuth,
                            altitude, scale, **kwargs)

        monkeypatch.setattr(march_router_mod, "shadow_march", spy)
        _bundle(science_site.cache_a, scene, cube_window=WRITE_WINDOW)
        return grids

    def test_march_grids_narrow_below_the_read_window(self, science_site, monkeypatch) -> None:
        scene = _edited_scene(science_site)
        grids = self._march_grids(science_site, scene, monkeypatch)
        read_shape = (READ_WINDOW.height, READ_WINDOW.width)
        write_shape = (WRITE_WINDOW.height, WRITE_WINDOW.width)

        assert len(grids) == 153, "one march per sky patch"
        for shape in grids:
            assert shape[0] <= read_shape[0] and shape[1] <= read_shape[1]
            # every march must still cover the cells whose results are
            # accumulated (the write/cube window)
            assert shape[0] >= write_shape[0] and shape[1] >= write_shape[1], (
                f"march grid {shape} does not cover the write window "
                f"{write_shape}: the accumulation would read uncomputed cells"
            )
        distinct = sorted(set(grids))
        assert len(distinct) > 1, (
            "every patch marched at the read-window extent: the per-patch "
            "march windowing is not engaged"
        )
        smallest = min(r * c for r, c in grids)
        assert smallest < read_shape[0] * read_shape[1], (
            "no patch marched on a grid smaller than the read window"
        )

    def test_march_windowing_bitwise_matches_legacy_replay(
        self, science_site
    ) -> None:
        scene = _edited_scene(science_site)
        legacy = _bundle(science_site.cache_a, scene, cube_window=None)
        windowed = _bundle(science_site.cache_a, scene, cube_window=WRITE_WINDOW)
        _assert_bundle_matches_legacy(windowed, legacy, WRITE_WINDOW)

    def test_march_windows_correct_at_read_window_corners(
        self, science_site
    ) -> None:
        """The quadrant expansion must point the RIGHT way.

        A wrong sign cuts reads on the wrong side of the accumulate window;
        placing it in opposite corners of the read window makes every sign
        error visible as a bitwise mismatch against the legacy replay.
        """
        scene = _edited_scene(science_site)
        legacy = _bundle(science_site.cache_a, scene, cube_window=None)
        for cube in (
            RasterWindow(45, 75, 45, 75),     # top-left of the read window
            RasterWindow(105, 135, 105, 135),  # bottom-right
        ):
            windowed = _bundle(science_site.cache_a, scene, cube_window=cube)
            _assert_bundle_matches_legacy(windowed, legacy, cube)

    def test_bush_in_read_window_disables_march_windowing(
        self, science_site, monkeypatch
    ) -> None:
        """shadow()'s bush branches reduce over the marched extent; a shrunk
        extent could flip their predicates, so a non-zero bush layer must
        fall back to full-window marches (and still match the legacy replay
        bitwise)."""
        scene = _edited_scene(science_site)
        # compose_full_scene_tensors builds bush as a fresh tensor (no cache
        # aliasing), so planting bush cells inside the read window is safe.
        scene.bush[50, 80] = 5.0
        scene.bush[120, 60] = 2.0
        read_slice = (
            slice(READ_WINDOW.row_start, READ_WINDOW.row_stop),
            slice(READ_WINDOW.col_start, READ_WINDOW.col_stop),
        )
        assert float(scene.bush[read_slice].max()) > 0.0

        grids = self._march_grids(science_site, scene, monkeypatch)
        read_shape = (READ_WINDOW.height, READ_WINDOW.width)
        assert set(grids) == {read_shape}, (
            "a bushy read window must march every patch at the full read "
            "extent (the global bush predicates are extent-dependent)"
        )

        legacy = _bundle(science_site.cache_a, scene, cube_window=None)
        windowed = _bundle(science_site.cache_a, scene, cube_window=WRITE_WINDOW)
        _assert_bundle_matches_legacy(windowed, legacy, WRITE_WINDOW)

    def test_march_windowing_ab_gate_bitwise(self, science_site, monkeypatch) -> None:
        """A/B oracle: force the march windowing OFF and compare bitwise.

        This is the direct proof that lever 1 changes no value: the same
        solve, once with narrowed marches (default) and once with every
        patch marched at the read extent (gate forced off), must agree on
        every accumulated term.
        """
        import solweig_gpu.incremental.solver as solver_mod

        scene = _edited_scene(science_site)
        windowed = _bundle(science_site.cache_a, scene, cube_window=WRITE_WINDOW)

        monkeypatch.setattr(
            solver_mod, "_bush_free_marches", lambda bush: False
        )
        forced_legacy = _bundle(science_site.cache_a, scene, cube_window=WRITE_WINDOW)

        for name in (
            "svfveg", "svfEveg", "svfSveg", "svfWveg", "svfNveg",
            "svfaveg", "svfEaveg", "svfSaveg", "svfWaveg", "svfNaveg",
            "svftotal", "vegshmat", "vbshvegshmat",
        ):
            position = SVF_BUNDLE_INDEX[name]
            assert np.array_equal(
                windowed[position].numpy(),
                forced_legacy[position].numpy(),
                equal_nan=True,
            ), f"{name}: windowed marches differ from full-window marches"


# ---------------------------------------------------------------------------
# Lever 2 (scientific): hoist counters
# ---------------------------------------------------------------------------


@pytest.mark.scientific
class TestInvariantHoists:
    @staticmethod
    def _clear_geometry_memo(monkeypatch) -> None:
        import solweig_gpu.incremental.solver as solver_mod

        monkeypatch.setattr(
            solver_mod, "_SKY_PATCH_GEOMETRY", {}, raising=False
        )

    def test_annulus_weights_compute_once_per_ring_not_per_patch(
        self, science_site, monkeypatch
    ) -> None:
        import solweig_gpu.incremental.solver as solver_mod

        self._clear_geometry_memo(monkeypatch)
        calls = []
        original = solver_mod.annulus_weight

        def spy(altitude, aziinterval, device=None):
            calls.append((altitude, aziinterval))
            return original(altitude, aziinterval, device)

        monkeypatch.setattr(solver_mod, "annulus_weight", spy)

        scene = _edited_scene(science_site)
        _bundle(science_site.cache_a, scene, cube_window=WRITE_WINDOW)

        # annulus spans per ring: 12,12,12,12,12,12,12,6 (annulino diffs),
        # each evaluated once isotropic + once anisotropic = 2 x 90
        assert len(calls) == 2 * (12 * 7 + 6), (
            f"annulus weights computed {len(calls)} times for one solve: "
            "they depend only on the ring geometry and must be hoisted out "
            "of the per-patch loop"
        )

    def test_patch_geometry_memoized_across_solves(
        self, science_site, monkeypatch
    ) -> None:
        import solweig_gpu.incremental.solver as solver_mod

        self._clear_geometry_memo(monkeypatch)
        calls = []
        original = solver_mod.create_patches

        def spy(patch_option):
            calls.append(patch_option)
            return original(patch_option)

        monkeypatch.setattr(solver_mod, "create_patches", spy)

        scene = _edited_scene(science_site)
        first = _bundle(science_site.cache_a, scene, cube_window=WRITE_WINDOW)
        second = _bundle(science_site.cache_a, scene, cube_window=None)

        assert calls == [2], (
            f"create_patches ran {len(calls)} times for two solves: the "
            "sky-patch geometry is constant and must be memoized"
        )
        # The memoized geometry still yields the legacy replay bitwise.
        _assert_bundle_matches_legacy(first, second, WRITE_WINDOW)
