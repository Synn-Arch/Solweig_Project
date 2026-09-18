# SPDX-License-Identifier: GPL-3.0-only
"""R6 item 4 fence: why the per-(revision, t) march cache is NOT shipped.

The evaluated lever (R6 dispatch, honest-outcome clause) was a cache of the
``precomputed_shadows`` seam march (``shadowingfunction_wallheight_23`` at
utci_process.py, the out_window branch) keyed on ``(scene revision, t)``.
Two gates had to pass before implementation:

* (a) the profiled win is real. It is not: on the site_500 replay
  (/tmp/r6_proof/profile.json, commit f22c1e6, 8 threads, load ~5-6) the
  seam march costs 1.757 s of the 30.46 s time loop — a 5.8 % share — and
  a cache only ever hits on a REPEAT solve whose march inputs are all
  bitwise-identical. The single-edit hot path (one tree edit, one solve)
  is always cold: zero benefit. The dominant cost is the SVF replay
  (16.37 s svf_seconds), not this march.
* (b) the ``(revision, t)`` key is sound. It is not: the march is a pure
  function of its FULL input tuple — ``(a, vegdsm, vegdsm2, azimuth,
  altitude, scale, amaxvalue, bush, walls, dirwalls)`` evaluated on a
  given GRID EXTENT — and three of those dimensions are not captured by
  ``(scene revision, t)``:

  - the resolved FORCING at ``t`` (a met overlay or any future forcing
    edit changes the marched sun position at the same timestep);
  - the march AMPLITUDE (``solve_window`` escalates the whole loop to
    the absolute ``scene.amaxvalue`` when a timestep bands — the R6
    closure — so the same ``(scene, t)`` marches at a different stop
    depending on the read window's effective amplitude);
  - the read-window EXTENT (edge cells clamp differently; a narrower
    window that drops an occluder changes values at cells it contains).

  A stale serve from any of those is a WRONG-SCIENCE bug class (wrong
  shadows presented as exact), so the dispatch's rule applies: typed
  refusal on any doubt, never a guess.

The only sound key is a hash of the full input content, at which point
the hit set is exactly "bitwise-identical repeat solves" — the measured
5.8 % of one stage, on repeats only. The producer-side revision identity
that a cheap trustworthy key would need lives in executor.py / worker.py,
which are frozen this wave (R5 machinery). Lever refused; these fences
pin the seam properties any future cache must respect, so the refusal is
documented at the march itself rather than in prose alone.

Constructions below are fully synthetic (no site fixtures): a 48x48 flat
grid, one 12 m building spike at (20, 20), target cell (16, 16), sun at
azimuth 135 deg (the march reads diagonally (+row, +col), ``ds = sqrt(2)``)
and altitude 45 deg, scale 1 (1 m pixels). Step ``i`` reaches the spike at
``i = 4`` with ``dz_4 = 4 * sqrt(2) ~ 5.66 m``; the loop's stop is checked
against the PREVIOUS step's dz, so amplitude 4.0 stops after index 3 (the
spike is never consulted) while 12.0 runs index 4+ (the spike shadows the
target). Verified against the live kernel before pinning.
"""

from __future__ import annotations

import torch

from solweig_gpu.solweig import shadowingfunction_wallheight_23

N = 48
SPIKE = (20, 20)
TARGET = (16, 16)
SPIKE_HEIGHT = 12.0
#: amplitude that stops after index 3 (dz_3 = 3*sqrt(2) ~ 4.24 < 4.0 is
#: false, so index 4 never runs: the spike is not consulted)
AMP_SHORT = 4.0
#: amplitude that runs index 4 (dz_3 ~ 4.24 <= 12.0): the spike is consulted
AMP_LONG = 12.0
SUN_AZ = 135.0
SUN_ALT = 45.0


def _scene() -> tuple[torch.Tensor, ...]:
    """(a, vegdem, vegdem2, bush, walls, aspect) with one building spike."""
    a = torch.zeros((N, N))
    a[SPIKE] = SPIKE_HEIGHT
    zeros = torch.zeros((N, N))
    return a, zeros, zeros, zeros, zeros, zeros


def _march(a, veg, veg2, bush, walls, asp, *, azimuth, altitude, amaxvalue):
    return shadowingfunction_wallheight_23(
        a, veg, veg2, azimuth, altitude, 1.0, amaxvalue, bush, walls, asp
    )


def _full_march(*, azimuth=SUN_AZ, altitude=SUN_ALT, amaxvalue=AMP_LONG):
    a, veg, veg2, bush, walls, asp = _scene()
    return _march(
        a, veg, veg2, bush, walls, asp,
        azimuth=azimuth, altitude=altitude, amaxvalue=amaxvalue,
    )


# ---------------------------------------------------------------------------
# Purity: the ONLY sound cache key is the full input content
# ---------------------------------------------------------------------------


class TestMarchIsPure:
    def test_identical_inputs_produce_bitwise_identical_outputs(self) -> None:
        """Determinism pin: same full input tuple -> same 8 output planes.

        This is the property that makes a full-content hash a sound key —
        and simultaneously the property a ``(revision, t)`` key lacks (the
        classes below break each incomplete dimension).
        """
        first = _full_march()
        second = _full_march()
        assert len(first) == 8
        for index, (x, y) in enumerate(zip(first, second)):
            assert torch.equal(x, y), (
                f"output plane {index} is not a pure function of the march "
                "inputs; no cache key of any shape is sound"
            )

    def test_shadow_semantics_of_the_construction(self) -> None:
        """The spike shadows the target at AMP_LONG (sh 0 = shadowed post
        ``1 - sh``), and does not at AMP_SHORT — the value the amplitude
        and extent fences below flip."""
        sh_long = _full_march(amaxvalue=AMP_LONG)[1]
        sh_short = _full_march(amaxvalue=AMP_SHORT)[1]
        assert sh_long[TARGET].item() == 0.0, (
            "construction drifted: the spike should shadow the target at "
            f"AMP_LONG (got sh={sh_long[TARGET].item()})"
        )
        assert sh_short[TARGET].item() == 1.0, (
            "construction drifted: the spike must not be consulted at "
            f"AMP_SHORT (got sh={sh_short[TARGET].item()})"
        )


# ---------------------------------------------------------------------------
# Fence 1: forcing identity — (scene, t) alone does not determine the march
# ---------------------------------------------------------------------------


class TestCacheKeyMustIncludeForcing:
    def test_same_scene_same_t_different_sun_changes_the_march(self) -> None:
        """Same arrays, same amplitude, same timestep INDEX — a different
        resolved sun position (the u-c3 forcing-overlay case, or any future
        forcing edit that moves the marched geometry) flips 16 shadow cells.

        A ``(scene revision, t)``-keyed cache would serve the first sun's
        planes for the second forcing: wrong shadows presented as exact.
        """
        south_sun = _full_march(azimuth=SUN_AZ)[1]
        north_sun = _full_march(azimuth=315.0)[1]
        assert south_sun[TARGET].item() != north_sun[TARGET].item(), (
            "construction drifted: the two sun positions should disagree "
            "at the target cell"
        )
        assert not torch.equal(south_sun, north_sun), (
            "two different sun positions produced identical shadow planes; "
            "the forcing-identity fence has nothing to pin"
        )


# ---------------------------------------------------------------------------
# Fence 2: amplitude identity — the R6 escalation changes the selection
# ---------------------------------------------------------------------------


class TestCacheKeyMustIncludeAmplitude:
    def test_same_scene_same_t_different_amplitude_changes_the_march(self) -> None:
        """``solve_window`` picks the march amplitude per solve: the windowed
        effective amplitude, or the absolute ``scene.amaxvalue`` when a
        timestep bands (the R6 time-loop escalation). Both values below are
        live selections for the SAME ``(scene, t)`` on different read
        windows — a cache primed under one and keyed without the amplitude
        would serve the other's step regime.
        """
        sh_long = _full_march(amaxvalue=AMP_LONG)[1]
        sh_short = _full_march(amaxvalue=AMP_SHORT)[1]
        differing = int((sh_long != sh_short).sum())
        assert differing > 0, (
            "the march output does not depend on the amaxvalue argument; "
            "the amplitude fence has nothing to pin"
        )
        assert sh_long[TARGET].item() != sh_short[TARGET].item()


# ---------------------------------------------------------------------------
# Fence 3: extent identity — a narrower window that drops an occluder
# ---------------------------------------------------------------------------


class TestCacheKeyMustIncludeWindowExtent:
    def test_same_inputs_different_extent_changes_values_in_shared_cells(self) -> None:
        """Sub-window rows/cols 4..20 contains the target but not the spike:
        its edge-clamped march never consults the occluder, so the SAME
        cell carries a different value than the full-grid march — same
        scene, same sun, same amplitude, same ``t``.

        Any plane cache must key on the extent it was marched at (the
        seam's planes are read-window crops, not scene quantities).
        """
        a, veg, veg2, bush, walls, asp = _scene()
        full_sh = _march(
            a, veg, veg2, bush, walls, asp,
            azimuth=SUN_AZ, altitude=SUN_ALT, amaxvalue=AMP_LONG,
        )[1]
        sub = (slice(4, 20), slice(4, 20))
        sub_sh = _march(
            a[sub].clone(), veg[sub], veg2[sub], bush[sub], walls[sub],
            asp[sub], azimuth=SUN_AZ, altitude=SUN_ALT, amaxvalue=AMP_LONG,
        )[1]
        target_sub = (TARGET[0] - 4, TARGET[1] - 4)
        assert full_sh[TARGET].item() != sub_sh[target_sub].item(), (
            "construction drifted: the sub-window march should miss the "
            "spike its edge clamp replaces"
        )
        assert not torch.equal(full_sh[sub], sub_sh)
